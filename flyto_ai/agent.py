# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Agent class — chat loop orchestrator."""
import logging
from typing import Any, Dict, List, Optional, Tuple

from flyto_ai.config import AgentConfig
from flyto_ai.models import ChatResponse, StreamCallback, StreamEvent, StreamEventType
from flyto_ai.prompt.policies import is_module_allowed, is_tool_allowed
from flyto_ai.prompt.system_prompt import build_system_prompt, detect_language
from flyto_ai.providers.base import LLMProvider
from flyto_ai.validation import extract_yaml_from_response, validate_workflow_steps

logger = logging.getLogger(__name__)


def _init_blueprint_storage():
    """Initialize flyto-blueprint with SQLite storage for local persistence."""
    try:
        from flyto_blueprint import get_engine
        from flyto_blueprint.storage.sqlite import SQLiteBackend
        get_engine(storage=SQLiteBackend())
    except ImportError:
        pass


def _blueprint_feedback(tool_calls: List[Dict[str, Any]], execution_results: List[Dict[str, Any]], user_message: str):
    """Closed-loop blueprint learning. Pure code — zero LLM involvement.

    1. If a blueprint was used (use_blueprint in tool_calls):
       - All executions OK  → report_outcome(success=True)  → score +5
       - Any execution FAIL → report_outcome(success=False) → score -10
       - Score < 10 → auto-retired, never suggested again
    2. If execution succeeded with 3+ steps (no blueprint):
       - learn_from_execution() → save as verified blueprint (score 70)
       - Duplicate workflow → boosted_existing → score +3
    """
    try:
        from flyto_blueprint import get_engine
    except ImportError:
        return

    engine = get_engine()
    all_ok = all(r.get("ok", False) for r in execution_results)

    # --- Phase 1: Report outcome if a blueprint was used ---
    used_blueprint_id = None
    for tc in tool_calls:
        if tc.get("function") == "use_blueprint":
            used_blueprint_id = tc.get("arguments", {}).get("blueprint_id", "")
            break

    if used_blueprint_id:
        try:
            engine.report_outcome(used_blueprint_id, success=all_ok)
            logger.info("Blueprint outcome: %s %s", used_blueprint_id, "OK" if all_ok else "FAIL")
        except Exception as e:
            logger.debug("Blueprint report_outcome failed: %s", e)

    # --- Phase 2: Learn new blueprint from successful execution ---
    if not all_ok or len(execution_results) < 3:
        return

    steps = []
    for i, r in enumerate(execution_results):
        mid = r.get("module_id", "")
        if not mid:
            continue  # skip entries with empty module_id
        params = r.get("arguments", {}).get("params", {})
        steps.append({
            "id": "step_{}".format(i + 1),
            "module": mid,
            "params": params,
        })

    if len(steps) < 3:
        return  # not enough meaningful steps to save

    workflow = {"name": user_message[:80], "steps": steps}
    categories = list({s["module"].split(".")[0] for s in steps if "." in s["module"]})

    try:
        engine.learn_from_execution(workflow=workflow, name=user_message[:80], tags=categories)
        logger.info("Blueprint learned: %s (%d steps)", user_message[:40], len(steps))
    except Exception as e:
        logger.debug("Blueprint learn failed: %s", e)


class Agent:
    """High-level AI agent that translates natural language to Flyto workflows.

    Wires together: config → provider → tools → system prompt → chat loop.
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[List[Dict]] = None,
        dispatch_fn=None,
        system_prompt: Optional[str] = None,
        policies: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._config = config
        self._provider = self._make_provider()
        self._tools = tools or []
        self._dispatch_fn = dispatch_fn
        self._system_prompt = system_prompt
        self._policies = policies

        # Auto-discover tools when nothing injected
        if not self._tools and not self._dispatch_fn:
            self._auto_discover_tools()

    def _auto_discover_tools(self):
        """Auto-detect and register available tools (core, blueprint, inspect)."""
        from flyto_ai.tools.registry import ToolRegistry

        registry = ToolRegistry()

        try:
            from flyto_ai.tools.core_tools import get_core_tool_defs, dispatch_core_tool
            defs = get_core_tool_defs()
            if defs:
                registry.register_many(defs, dispatch_core_tool)
        except Exception:
            pass

        try:
            from flyto_ai.tools.blueprint_tools import get_blueprint_tool_defs, dispatch_blueprint_tool
            _init_blueprint_storage()
            defs = get_blueprint_tool_defs()
            if defs:
                registry.register_many(defs, dispatch_blueprint_tool)
        except Exception:
            pass

        try:
            from flyto_ai.tools.inspect_page import INSPECT_PAGE_TOOL, dispatch_inspect_page
            registry.register(INSPECT_PAGE_TOOL, dispatch_inspect_page)
        except Exception:
            pass

        if registry.tools:
            self._tools = registry.tools
            self._dispatch_fn = registry.dispatch

    def _make_provider(self) -> LLMProvider:
        """Create the LLM provider from config."""
        cfg = self._config
        if cfg.provider == "anthropic":
            from flyto_ai.providers.anthropic import AnthropicProvider
            return AnthropicProvider(
                api_key=cfg.api_key,
                model=cfg.resolved_model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
        elif cfg.provider == "ollama":
            from flyto_ai.providers.ollama import OllamaProvider
            return OllamaProvider(
                model=cfg.resolved_model,
                base_url=cfg.base_url or "http://localhost:11434/v1",
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
        else:
            # openai, openai-compatible, or default
            from flyto_ai.providers.openai import OpenAIProvider
            return OpenAIProvider(
                api_key=cfg.api_key,
                model=cfg.resolved_model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                base_url=cfg.base_url,
            )

    def _make_safe_dispatch(self):
        """Create a dispatch function that enforces policies."""
        base_dispatch = self._dispatch_fn
        policies = self._policies

        async def safe_dispatch(func_name: str, func_args: dict) -> dict:
            if policies and not is_tool_allowed(func_name, policies):
                return {"ok": False, "error": "Tool not allowed: {}".format(func_name)}
            if policies and func_name == "execute_module":
                module_id = func_args.get("module_id", "")
                if not is_module_allowed(module_id, policies):
                    category = module_id.split(".")[0] if "." in module_id else module_id
                    return {
                        "ok": False,
                        "error": "Module category '{}' is not allowed.".format(category),
                    }
            return await base_dispatch(func_name, func_args)

        return safe_dispatch if base_dispatch else None

    async def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
        template_context: Optional[Dict[str, Any]] = None,
        mode: str = "execute",
        on_tool_call=None,
        on_stream: Optional[StreamCallback] = None,
    ) -> ChatResponse:
        """Run one chat turn: send message → tool loop → validation → response.

        Parameters
        ----------
        mode : str
            ``"execute"`` — run modules directly, skip YAML nudge/validation.
            ``"yaml"`` — only generate workflow YAML (original behaviour).
        on_tool_call : callable, optional
            ``on_tool_call(func_name, func_args)`` — called before each tool
            dispatch.  Use for progress display.
        on_stream : callable, optional
            ``on_stream(StreamEvent)`` — called for each streaming event
            (tokens, tool start/end, done).  When set, providers enable
            streaming for LLM responses.
        """
        if not self._config.api_key and self._config.provider != "ollama":
            return ChatResponse(
                ok=False,
                message="No API key configured.",
                session_id="",
                error="no_api_key",
            )

        # Build messages
        messages = list(history or [])
        messages.append({"role": "user", "content": message})

        # Build dispatch (with optional progress + stream callbacks)
        dispatch_fn = self._make_safe_dispatch()
        if dispatch_fn and (on_tool_call or on_stream):
            _base = dispatch_fn

            async def _instrumented(func_name: str, func_args: dict) -> dict:
                if on_tool_call:
                    try:
                        on_tool_call(func_name, func_args)
                    except Exception:
                        pass  # callback failure must not break tool loop
                if on_stream:
                    try:
                        on_stream(StreamEvent(
                            type=StreamEventType.TOOL_START,
                            tool_name=func_name,
                            tool_args=func_args,
                        ))
                    except Exception:
                        pass
                result = await _base(func_name, func_args)
                if on_stream:
                    try:
                        on_stream(StreamEvent(
                            type=StreamEventType.TOOL_END,
                            tool_name=func_name,
                            tool_result=result if isinstance(result, dict) else None,
                        ))
                    except Exception:
                        pass
                return result

            dispatch_fn = _instrumented
        has_tools = bool(self._tools and dispatch_fn)

        # Build system prompt (with deterministic language detection)
        reply_language = detect_language(message)
        system_prompt = self._system_prompt or build_system_prompt(
            module_count=300, context=template_context, has_tools=has_tools,
            mode=mode, reply_language=reply_language,
        )

        # Call LLM (toolless mode if no tools available)
        if has_tools:
            response_content, tool_calls = await self._call_llm(messages, system_prompt, dispatch_fn, on_stream=on_stream)
        else:
            response_content, tool_calls = await self._call_llm_toolless(messages, system_prompt, on_stream=on_stream)

        if not response_content:
            return ChatResponse(
                ok=False,
                message="AI provider call failed. Please try again.",
                session_id="",
                error="provider_call_failed",
            )

        # --- yaml mode: nudge + validation (original behaviour) ---
        if mode == "yaml":
            # No-YAML nudge
            if not extract_yaml_from_response(response_content):
                nudge_messages = messages + [
                    {"role": "assistant", "content": response_content},
                    {"role": "user", "content": (
                        "You must always output a Flyto Workflow YAML. "
                        "Please generate the workflow YAML now using the modules and blueprints available."
                    )},
                ]
                nudge_content, nudge_tc = await self._call_llm(nudge_messages, system_prompt, dispatch_fn)
                if nudge_content and extract_yaml_from_response(nudge_content):
                    response_content = nudge_content
                    tool_calls.extend(nudge_tc)

            # YAML validation loop
            for _attempt in range(self._config.max_validation_rounds):
                yaml_str = extract_yaml_from_response(response_content)
                if not yaml_str:
                    break
                errors = validate_workflow_steps(yaml_str)
                if not errors:
                    break

                error_list = "\n".join("- {}".format(e) for e in errors)
                retry_messages = messages + [
                    {"role": "assistant", "content": response_content},
                    {"role": "user", "content": (
                        "The workflow YAML you generated has validation errors:\n"
                        "{}\n\n"
                        "Please call get_module_info() for each failing module to "
                        "verify the correct param names, then regenerate the YAML.".format(error_list)
                    )},
                ]
                retry_content, retry_tc = await self._call_llm(retry_messages, system_prompt, dispatch_fn)
                if retry_content:
                    response_content = retry_content
                    tool_calls.extend(retry_tc)
                else:
                    break

        # Collect execute_module results from tool calls
        execution_results = [
            tc for tc in tool_calls
            if tc.get("function") == "execute_module"
        ]

        # Closed-loop blueprint feedback (no LLM involved)
        if mode == "execute" and execution_results:
            _blueprint_feedback(tool_calls, execution_results, message)

        return ChatResponse(
            ok=True,
            message=response_content,
            session_id="",
            tool_calls=tool_calls,
            execution_results=execution_results,
            provider=self._config.provider,
            model=self._config.resolved_model,
        )

    async def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        dispatch_fn,
        on_stream: Optional[StreamCallback] = None,
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Call the LLM provider with tools. Returns (content, tool_call_log)."""
        try:
            return await self._provider.chat(
                messages, system_prompt, self._tools,
                dispatch_fn, self._config.max_tool_rounds,
                on_stream=on_stream,
            )
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return None, []

    async def _call_llm_toolless(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        on_stream: Optional[StreamCallback] = None,
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Call the LLM provider without tools (pure conversation)."""
        try:
            async def _noop_dispatch(name: str, args: dict) -> dict:
                return {"ok": False, "error": "No tools available"}

            return await self._provider.chat(
                messages, system_prompt, [], _noop_dispatch, max_rounds=1,
                on_stream=on_stream,
            )
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return None, []
