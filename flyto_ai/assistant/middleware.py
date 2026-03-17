# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""AssistantMiddleware — single entry point for all assistant intelligence.

Usage in Agent::

    self._assistant = AssistantMiddleware()

    # Before LLM call:
    prompt_hint = self._assistant.prepare(message, mode)

    # Wrap dispatch:
    dispatch_fn = self._assistant.wrap(base_dispatch)

    # After LLM call:
    pending_input = self._assistant.post_process(tool_calls, exec_results, message, mode)
"""
import logging
import re
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from flyto_ai.assistant import router, interactive, resilience
from flyto_ai.assistant.output_tracker import OutputTracker, extract_output_paths
from flyto_ai.assistant.safety import CircuitBreaker, BoundedHistory, mask_sensitive

logger = logging.getLogger(__name__)


def _url_to_context_key(url: str) -> str:
    """Derive a vault context_key from a URL.

    Examples:
        https://x.com/i/flow/login    → x_login
        https://github.com/login      → github_login
        https://thsrc.com.tw/member   → thsrc_login
        https://accounts.google.com   → google_login
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        # Strip www. and common TLDs
        host = re.sub(r'^www\.', '', host)
        # Take the main domain part
        parts = host.split('.')
        if len(parts) >= 2:
            # Use first meaningful part: 'x' from 'x.com', 'github' from 'github.com'
            domain = parts[0]
            if domain in ('accounts', 'login', 'auth', 'auth0', 'sso', 'id', 'signin', 'oauth'):
                # 'accounts.google.com' → 'google'
                domain = parts[1] if len(parts) > 1 else parts[0]
        else:
            domain = parts[0]
        return "{}_login".format(domain) if domain else ""
    except Exception:
        return ""


def _derive_context_key(current_url: str, ask_question: str) -> str:
    """Derive context_key from available runtime signals. Zero hardcoding.

    Priority:
    1. Current browser URL (most reliable — already navigated)
    2. URL inside the ask_user question text
    3. Fuzzy match against existing vault keys

    This function never parses the original user message and never
    uses hardcoded site lists. It only uses:
    - The actual browser URL (set by browser.goto)
    - The LLM's ask_user question (which usually mentions the site)
    - Existing vault keys (dynamic, grows as user saves credentials)
    """
    # 1. Current browser URL → domain_login
    if current_url:
        key = _url_to_context_key(current_url)
        if key:
            return key

    # 2. URL inside ask_user question
    url_match = re.search(r'https?://[^\s]+', ask_question)
    if url_match:
        key = _url_to_context_key(url_match.group(0))
        if key:
            return key

    # 3. Fuzzy match: check if any vault key's domain appears in the question
    try:
        from flyto_ai.vault import Vault
        vault = Vault()
        vault.load()
        q_lower = ask_question.lower()
        for vault_key in vault.list_keys():
            # Extract domain from vault key: 'github_login' → 'github'
            domain = vault_key.split("_")[0]
            if domain and len(domain) >= 2 and domain in q_lower:
                return vault_key
            # Single-char domains (like "x") need word boundary check
            if domain and len(domain) == 1 and re.search(r'(?:^|\s)' + re.escape(domain) + r'(?:\s|$)', q_lower):
                return vault_key
    except Exception:
        pass

    return ""


class AssistantMiddleware:
    """Transparent intelligence layer between Agent and tool dispatch.

    Three hooks — prepare / wrap / post_process — cover the full lifecycle.
    """

    def __init__(self) -> None:
        router.init_storage()
        self._output_tracker: Optional[OutputTracker] = None

    # ── Before LLM call ──────────────────────────────────────────

    def prepare(self, message: str, mode: str = "execute") -> str:
        """Pre-resolve blueprint and return a prompt snippet to inject."""
        if mode != "execute":
            return ""
        return router.pre_resolve(message)

    # ── Wrap dispatch ────────────────────────────────────────────

    def wrap(self, base_dispatch: Callable, user_message: str = "") -> Callable:
        """Wrap the tool dispatch function with assistant intelligence.

        Layers:
        1. Blueprint guard — redirect to use_blueprint on first call
        2. URL tracker — track current URL for context_key derivation
        3. ask_user enrichment — auto-inject context_key from URL
        4. Snapshot guard — auto-inject snapshot before browser interact
        5. Anti-bot detection — auto-retry with system Chrome
        6. Selector healing — auto-fix broken CSS after failure
        """
        guard_done = {"v": False}
        snap_guard = resilience.SnapshotGuard()
        antibot_guard = resilience.AntibotGuard()
        breaker = CircuitBreaker(max_failures=3)
        history = BoundedHistory(max_size=20)
        current_url = {"v": ""}
        output_paths = extract_output_paths(user_message)
        self._output_tracker = OutputTracker(output_paths) if output_paths else None

        async def assistant_dispatch(func_name: str, func_args: dict) -> dict:
            # Layer 1: Blueprint guard (first call only)
            if not guard_done["v"]:
                guard_done["v"] = True
                redirect = await router.guard(
                    func_name, func_args, user_message, base_dispatch,
                )
                if redirect is not None:
                    return redirect

            # Layer 2: ask_user enrichment — auto-inject context_key
            if func_name == "ask_user":
                args = func_args if isinstance(func_args, dict) else {}
                if not args.get("context_key"):
                    question = args.get("question", "")
                    derived_key = _derive_context_key(current_url["v"], question)
                    if derived_key:
                        args["context_key"] = derived_key
                        logger.info("Auto-derived context_key=%s", derived_key)

            # Layer 3: Snapshot guard — auto-inject snapshot before interact
            if snap_guard.needs_snapshot(func_name, func_args):
                logger.info("Auto-injecting browser.snapshot before %s",
                            func_args.get("module_id", func_name))
                snap_result = await base_dispatch("execute_module", {
                    "module_id": "browser.snapshot",
                    "params": {},
                })
                if snap_result.get("ok", False):
                    snap_guard.record_snapshot(snap_result)
                    snap_result["_auto_snapshot"] = True
                    snap_result["message"] = (
                        "AUTO-SNAPSHOT: You tried to interact without seeing the page first. "
                        "Here is the current page content. Find the REAL selector from the "
                        "hints below, then retry your action with the correct selector."
                    )
                    return snap_result

            # Track snapshot calls
            snap_guard.on_tool_call(func_name, func_args)

            # Layer 4: Delegate to flyto-core — validate + auto-correct before execution
            # flyto-core's validate_params now returns corrections, not just errors
            if func_name == "execute_module" and isinstance(func_args, dict):
                module_id = func_args.get("module_id", "")
                params = func_args.get("params", {})
                if module_id:
                    try:
                        vr = await base_dispatch("validate_params", {
                            "module_id": module_id, "params": params,
                        })
                        if isinstance(vr, dict) and not vr.get("valid", True):
                            suggestions = vr.get("suggestions", {})
                            # Auto-correct: use corrected_params from flyto-core
                            if "corrected_params" in suggestions:
                                func_args = dict(func_args)
                                func_args["params"] = suggestions["corrected_params"]
                                logger.info("Params auto-corrected by flyto-core for %s", module_id)
                            # Module not found: use alternatives from flyto-core
                            elif "alternatives" in suggestions and suggestions["alternatives"]:
                                func_args = dict(func_args)
                                func_args["module_id"] = suggestions["alternatives"][0]
                                logger.info("Module redirected by flyto-core: %s → %s",
                                            module_id, func_args["module_id"])
                    except Exception:
                        pass

            # Variable resolution (${steps.x.result} → actual value)
            if func_name == "execute_module" and isinstance(func_args, dict):
                params = func_args.get("params", {})
                if any("${" in str(v) for v in params.values()):
                    try:
                        from flyto_ai.assistant.param_fixer import _resolve_variables
                        resolved = _resolve_variables(params, history.items())
                        if resolved is not params:
                            func_args = dict(func_args)
                            func_args["params"] = resolved
                    except Exception:
                        pass

            # Circuit breaker: block modules that keep failing
            if func_name == "execute_module" and isinstance(func_args, dict):
                mid = func_args.get("module_id", "")
                if breaker.is_tripped(mid):
                    return {"ok": False, "error": breaker.get_message(mid)}

            # Normal dispatch
            result = await base_dispatch(func_name, func_args)

            # Mask sensitive data in ask_user auto-fill results
            if func_name == "ask_user" and isinstance(result, dict):
                if result.get("auto_filled"):
                    # Mask passwords before LLM sees them
                    result["data"] = mask_sensitive(result.get("data", {}))

            # Track execution results + circuit breaker (with empty detection)
            if func_name == "execute_module" and isinstance(result, dict):
                mid = func_args.get("module_id", "") if isinstance(func_args, dict) else ""
                breaker.record_result(mid, result.get("ok", False), result)
                history.append(result)
                if self._output_tracker:
                    self._output_tracker.on_tool_call(func_name, func_args, result)

            # Auto-retry on failure: use flyto-core's corrected_params
            if (func_name == "execute_module"
                    and isinstance(result, dict)
                    and not result.get("ok", True)
                    and isinstance(func_args, dict)):
                error_msg = str(result.get("error", "")).lower()
                if "missing" in error_msg or "required" in error_msg:
                    try:
                        vr = await base_dispatch("validate_params", {
                            "module_id": func_args.get("module_id", ""),
                            "params": func_args.get("params", {}),
                        })
                        corrected = vr.get("suggestions", {}).get("corrected_params")
                        if corrected:
                            retry_args = dict(func_args)
                            retry_args["params"] = corrected
                            retry_result = await base_dispatch(func_name, retry_args)
                            if retry_result.get("ok", False):
                                history.append(retry_result)
                                return retry_result
                    except Exception:
                        pass

            # Track URL from goto results
            if (func_name == "execute_module"
                    and func_args.get("module_id") == "browser.goto"
                    and isinstance(result, dict)):
                new_url = result.get("url", func_args.get("params", {}).get("url", ""))
                if new_url:
                    current_url["v"] = new_url

            # Track snapshot results
            if (func_name == "execute_module"
                    and func_args.get("module_id") in ("browser.snapshot",)
                    and isinstance(result, dict) and result.get("ok")):
                snap_guard.record_snapshot(result)

            # Layer 5: Anti-bot detection — auto-retry with system Chrome
            if antibot_guard.check_result(func_name, func_args, result):
                url = func_args.get("params", {}).get("url", "")
                retried = await antibot_guard.retry_with_system_chrome(base_dispatch, url)
                if retried is not None:
                    return retried

            # Layer 6: Selector healing (on failure)
            if resilience.should_heal(func_name, func_args, result):
                healed = await resilience.try_heal(base_dispatch, func_args)
                if healed is not None:
                    return healed

            return result

        # Wrap with output auto-save on session end
        original_dispatch = assistant_dispatch
        _output_tracker = self._output_tracker

        async def dispatch_with_output(func_name: str, func_args: dict) -> dict:
            return await original_dispatch(func_name, func_args)

        # Store references for post_process to use the same dispatch chain
        self._wrapped_dispatch = dispatch_with_output
        self._exec_history = history

        return dispatch_with_output

    # ── After LLM call ───────────────────────────────────────────

    async def post_process(
        self,
        tool_calls: List[Dict[str, Any]],
        execution_results: List[Dict[str, Any]],
        user_message: str,
        mode: str = "execute",
        dispatch: Optional[Callable] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run post-execution logic. Returns pending_input if ask_user was triggered."""
        if mode == "execute" and execution_results:
            router.feedback(tool_calls, execution_results, user_message)

        # Auto-write missing output files using the SAME dispatch chain
        # (preserves SANDBOX_DIR and other context)
        write_dispatch = getattr(self, '_wrapped_dispatch', None) or dispatch
        hist = getattr(self, '_exec_history', None)
        full_results = hist.items() if hist else execution_results

        if self._output_tracker and write_dispatch and full_results:
            try:
                auto_results = await self._output_tracker.auto_write_missing(
                    write_dispatch, full_results,
                )
                if auto_results:
                    execution_results.extend(auto_results)
                    logger.info("Auto-saved %d missing output files", len(auto_results))
            except Exception as e:
                logger.debug("Auto-save failed: %s", e)

        return interactive.extract_pending_input(tool_calls)
