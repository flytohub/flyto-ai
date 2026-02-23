# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Agent class — chat loop orchestrator."""
import logging
from typing import Any, Dict, List, Optional, Tuple

from flyto_ai.config import AgentConfig
from flyto_ai.models import ChatResponse
from flyto_ai.prompt.policies import is_module_allowed, is_tool_allowed
from flyto_ai.prompt.system_prompt import build_system_prompt
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
    ) -> ChatResponse:
        """Run one chat turn: send message → tool loop → validation → response."""
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

        # Build dispatch
        dispatch_fn = self._make_safe_dispatch()
        has_tools = bool(self._tools and dispatch_fn)

        # Build system prompt
        system_prompt = self._system_prompt or build_system_prompt(
            module_count=300, context=template_context, has_tools=has_tools,
        )

        # Call LLM (toolless mode if no tools available)
        if has_tools:
            response_content, tool_calls = await self._call_llm(messages, system_prompt, dispatch_fn)
        else:
            response_content, tool_calls = await self._call_llm_toolless(messages, system_prompt)

        if not response_content:
            return ChatResponse(
                ok=False,
                message="AI provider call failed. Please try again.",
                session_id="",
                error="provider_call_failed",
            )

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

        return ChatResponse(
            ok=True,
            message=response_content,
            session_id="",
            tool_calls=tool_calls,
            provider=self._config.provider,
            model=self._config.resolved_model,
        )

    async def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        dispatch_fn,
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Call the LLM provider with tools. Returns (content, tool_call_log)."""
        try:
            return await self._provider.chat(
                messages, system_prompt, self._tools,
                dispatch_fn, self._config.max_tool_rounds,
            )
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return None, []

    async def _call_llm_toolless(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Call the LLM provider without tools (pure conversation)."""
        try:
            async def _noop_dispatch(name: str, args: dict) -> dict:
                return {"ok": False, "error": "No tools available"}

            return await self._provider.chat(
                messages, system_prompt, [], _noop_dispatch, max_rounds=1,
            )
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return None, []
