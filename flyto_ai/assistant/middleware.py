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
from flyto_ai.assistant.choice_detector import detect_choices

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
        self._last_choices: Optional[Dict[str, Any]] = None

    # ── Before LLM call ──────────────────────────────────────────

    def prepare(self, message: str, mode: str = "execute") -> str:
        """Pre-resolve blueprint and return a prompt snippet to inject."""
        if mode != "execute":
            return ""
        return router.pre_resolve(message)

    async def auto_navigate_choice(self, message: str, dispatch: Any) -> Optional[Dict[str, Any]]:
        """If user's message matches a stored page choice, auto-navigate.

        System-level: directly clicks/gotos the element without LLM involvement.
        Returns the navigation result, or None if no match.
        """
        if not self._last_choices or not dispatch:
            return None

        msg = message.strip()
        for field in self._last_choices.get("fields", []):
            selectors = field.get("_selectors", {})
            options = field.get("options", [])
            for opt in options:
                if opt in msg or msg in opt:
                    target = selectors.get(opt, "")
                    if not target:
                        continue

                    logger.info("Auto-navigating choice: %s → %s", opt, target[:60])

                    # Re-snapshot to re-stamp hints, then click
                    if target.startswith("[data-flyto-hint"):
                        await dispatch("execute_module", {
                            "module_id": "browser.snapshot",
                            "params": {},
                        })

                    # Try hint selector first, then text selector
                    for selector in [target, 'text="{}"'.format(opt)]:
                        try:
                            click_result = await dispatch("execute_module", {
                                "module_id": "browser.click",
                                "params": {"selector": selector},
                            })
                            if isinstance(click_result, dict) and (
                                click_result.get("ok") or click_result.get("status") == "success"
                            ):
                                self._last_choices = None
                                import asyncio as _aio
                                await _aio.sleep(1)
                                return click_result
                        except Exception:
                            continue

        return None

    # ── Wrap dispatch ────────────────────────────────────────────

    def wrap(self, base_dispatch: Callable, user_message: str = "") -> Callable:
        """Wrap the tool dispatch function with assistant intelligence.

        Layers:
        1. Blueprint guard — redirect to use_blueprint on first call
        2. ask_user enrichment — auto-inject context_key from URL
        3. Snapshot guard — auto-inject snapshot before browser interact
        4. Param correction — validate + auto-correct before execution
        5. Circuit breaker — block modules that keep failing
        6. Post-result — healing, anti-bot retry, URL/snapshot tracking
        """
        guard_done = {"v": False}
        snap_guard = resilience.SnapshotGuard()
        antibot_guard = resilience.AntibotGuard()
        breaker = CircuitBreaker(max_failures=3)
        history = BoundedHistory(max_size=20)
        current_url = {"v": ""}
        has_browsed = {"v": False}
        output_paths = extract_output_paths(user_message)
        self._output_tracker = OutputTracker(output_paths) if output_paths else None
        # NOTE: do NOT reset self._last_choices here — it persists across rounds

        async def assistant_dispatch(func_name: str, func_args: dict) -> dict:
            # Layer 1: Blueprint guard (first call only)
            if not guard_done["v"]:
                guard_done["v"] = True
                redirect = await self._apply_blueprint_guard(
                    func_name, func_args, user_message, base_dispatch,
                )
                if redirect is not None:
                    return redirect

            # Layer 2: ask_user enrichment + premature ask_user guard
            ask_redirect = await self._apply_ask_user_enrichment(
                func_name, func_args, current_url, base_dispatch, has_browsed,
            )
            if ask_redirect is not None:
                return ask_redirect

            # Layer 3: Snapshot guard
            snap_redirect = await self._apply_snapshot_guard(
                func_name, func_args, snap_guard, base_dispatch,
            )
            if snap_redirect is not None:
                return snap_redirect
            snap_guard.on_tool_call(func_name, func_args)

            # Layer 4: Param correction (validate + variable resolution)
            func_args = await self._apply_param_correction(
                func_name, func_args, base_dispatch, history,
            )

            # Layer 5: Circuit breaker
            blocked = self._apply_circuit_breaker(func_name, func_args, breaker)
            if blocked is not None:
                return blocked

            # Normal dispatch
            result = await base_dispatch(func_name, func_args)

            # Post-result processing
            result = await self._on_result(
                func_name, func_args, result,
                base_dispatch, breaker, history, snap_guard,
                antibot_guard, current_url, has_browsed,
            )

            return result

        self._wrapped_dispatch = assistant_dispatch
        self._exec_history = history

        return assistant_dispatch

    # ── Dispatch layer helpers ──────────────────────────────────

    async def _apply_blueprint_guard(
        self, func_name: str, func_args: dict,
        user_message: str, base_dispatch: Callable,
    ) -> Optional[dict]:
        """Layer 1: Redirect to use_blueprint on first call if applicable."""
        return await router.guard(func_name, func_args, user_message, base_dispatch)

    async def _apply_ask_user_enrichment(
        self, func_name: str, func_args: dict,
        current_url: Dict[str, str],
        base_dispatch: Callable,
        has_browsed: Dict[str, bool],
    ) -> Optional[dict]:
        """Layer 2: Enrich ask_user + guard against premature ask_user.

        If LLM calls ask_user before visiting any website, and the fields
        have no options (LLM is guessing what to ask), redirect to
        inspect_page first so the system can detect real choices.
        """
        if func_name != "ask_user":
            return None

        args = func_args if isinstance(func_args, dict) else {}

        # Auto-derive context_key
        if not args.get("context_key"):
            question = args.get("question", "")
            derived_key = _derive_context_key(current_url["v"], question)
            if derived_key:
                args["context_key"] = derived_key
                logger.info("Auto-derived context_key=%s", derived_key)

        # Guard: if no browser activity yet and fields have no options,
        # LLM is guessing. Return a hint to use browser first.
        fields = args.get("fields", [])
        has_options = any(f.get("options") for f in fields)
        if not has_browsed.get("v") and not has_options:
            logger.info("ask_user called without browsing — suggesting browser first")
            return {
                "ok": True,
                "__ASK_USER__": False,
                "message": (
                    "You haven't visited any website yet. Before asking the user, "
                    "go to the relevant website first, read the page, and present "
                    "the ACTUAL options from the page — not guessed questions."
                ),
            }

        return None

    async def _apply_snapshot_guard(
        self, func_name: str, func_args: dict,
        snap_guard: "resilience.SnapshotGuard",
        base_dispatch: Callable,
    ) -> Optional[dict]:
        """Layer 3: Auto-inject snapshot before browser interact."""
        if not snap_guard.needs_snapshot(func_name, func_args):
            return None
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
        return None

    async def _apply_param_correction(
        self, func_name: str, func_args: dict,
        base_dispatch: Callable, history: "BoundedHistory",
    ) -> dict:
        """Layer 4: Validate + auto-correct params, resolve variables, before execution."""
        if func_name != "execute_module" or not isinstance(func_args, dict):
            return func_args

        # Delegate to flyto-core for validation + auto-correction
        module_id = func_args.get("module_id", "")
        params = func_args.get("params", {})
        if module_id:
            try:
                vr = await base_dispatch("validate_params", {
                    "module_id": module_id, "params": params,
                })
                if isinstance(vr, dict) and not vr.get("valid", True):
                    suggestions = vr.get("suggestions", {})
                    if "corrected_params" in suggestions:
                        func_args = dict(func_args)
                        func_args["params"] = suggestions["corrected_params"]
                        logger.info("Params auto-corrected by flyto-core for %s", module_id)
                    elif "alternatives" in suggestions and suggestions["alternatives"]:
                        func_args = dict(func_args)
                        func_args["module_id"] = suggestions["alternatives"][0]
                        logger.info("Module redirected by flyto-core: %s → %s",
                                    module_id, func_args["module_id"])
            except Exception:
                pass

        # Variable resolution (${steps.x.result} → actual value)
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

        return func_args

    def _apply_circuit_breaker(
        self, func_name: str, func_args: dict,
        breaker: "CircuitBreaker",
    ) -> Optional[dict]:
        """Layer 5: Block modules that keep failing."""
        if func_name != "execute_module" or not isinstance(func_args, dict):
            return None
        mid = func_args.get("module_id", "")
        if breaker.is_tripped(mid):
            return {"ok": False, "error": breaker.get_message(mid)}
        return None

    async def _on_result(
        self, func_name: str, func_args: dict, result: dict,
        base_dispatch: Callable,
        breaker: "CircuitBreaker", history: "BoundedHistory",
        snap_guard: "resilience.SnapshotGuard",
        antibot_guard: "resilience.AntibotGuard",
        current_url: Dict[str, str],
        has_browsed: Optional[Dict[str, bool]] = None,
    ) -> dict:
        """Post-dispatch: mask, track, auto-retry, heal."""
        # Mask sensitive data in ask_user auto-fill results
        if func_name == "ask_user" and isinstance(result, dict):
            if result.get("auto_filled"):
                result["data"] = mask_sensitive(result.get("data", {}))

        # Track execution results + circuit breaker
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

        # Track URL + browser activity + detect choices
        if func_name == "inspect_page" and has_browsed:
            has_browsed["v"] = True
        if (func_name == "execute_module"
                and func_args.get("module_id") == "browser.goto"
                and isinstance(result, dict)):
            if has_browsed:
                has_browsed["v"] = True
            new_url = result.get("url", func_args.get("params", {}).get("url", ""))
            if new_url:
                current_url["v"] = new_url

            # Choice detection on goto result (has links)
            try:
                choices = detect_choices(result)
                if choices:
                    result["_choices_detected"] = choices
                    self._last_choices = choices
                    opts_summary = [
                        f.get("label", "") + ": " + str(f.get("options", [])[:5])
                        for f in choices.get("fields", [])
                    ]
                    result["message"] = (
                        (result.get("message", "") or "") +
                        "\n\nCHOICES DETECTED on this page. "
                        "Call ask_user to let the user pick: " + str(opts_summary)
                    )
            except Exception:
                pass

        # Track snapshot results + detect choices
        if (func_name == "execute_module"
                and func_args.get("module_id") in ("browser.snapshot",)
                and isinstance(result, dict) and result.get("ok")):
            snap_guard.record_snapshot(result)

            # Choice detection: if page has selectable groups, inject ask_user hint
            try:
                snap_data = result.get("result", result)
                if isinstance(snap_data, str):
                    import json as _json
                    try:
                        snap_data = _json.loads(snap_data)
                    except (ValueError, TypeError):
                        snap_data = result
                choices = detect_choices(snap_data)
                if choices:
                    result["_choices_detected"] = choices
                    self._last_choices = choices
                    result["message"] = (
                        result.get("message", "") +
                        "\n\nINTERACTIVE CHOICES DETECTED on this page. "
                        "Call ask_user to let the user pick: " +
                        str([f.get("label") + ": " + str(f.get("options", [])[:5])
                             for f in choices.get("fields", [])])
                    )
            except Exception:
                pass

        # Also detect choices from inspect_page results
        if (func_name == "inspect_page"
                and isinstance(result, dict) and result.get("ok", True)):
            try:
                choices = detect_choices(result)
                if choices:
                    result["_choices_detected"] = choices
                    self._last_choices = choices
                    result["message"] = (
                        result.get("message", "") +
                        "\n\nINTERACTIVE CHOICES DETECTED. "
                        "Call ask_user to let the user pick: " +
                        str([f.get("label") + ": " + str(f.get("options", [])[:5])
                             for f in choices.get("fields", [])])
                    )
            except Exception:
                pass

        # Anti-bot detection — auto-retry with system Chrome
        if antibot_guard.check_result(func_name, func_args, result):
            url = func_args.get("params", {}).get("url", "")
            retried = await antibot_guard.retry_with_system_chrome(base_dispatch, url)
            if retried is not None:
                return retried

        # Selector healing (on failure)
        if resilience.should_heal(func_name, func_args, result):
            healed = await resilience.try_heal(base_dispatch, func_args)
            if healed is not None:
                return healed

        return result

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
