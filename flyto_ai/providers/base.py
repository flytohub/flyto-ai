# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Abstract LLM provider interface."""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from flyto_ai.models import StreamCallback, StreamEvent, StreamEventType

logger = logging.getLogger(__name__)

# Type alias for the dispatch function
DispatchFn = Callable[[str, dict], Coroutine[Any, Any, dict]]

# Shared constants for tool result handling
MAX_RESULT_LEN = 8000
MAX_PREVIEW_LEN = 500
TRUNCATION_NOTE = "...(truncated)"


def fire_stream(on_stream: Optional[StreamCallback], event: StreamEvent) -> None:
    """Safely invoke the stream callback. Shared across all providers."""
    if on_stream is None:
        return
    try:
        on_stream(event)
    except Exception as e:
        logger.debug("Stream callback error: %s", e)


async def dispatch_and_log_tool(
    func_name: str,
    func_args: dict,
    dispatch_fn: DispatchFn,
    round_num: int,
    on_stream: Optional[StreamCallback] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Dispatch a tool call, log it, truncate result. Shared by all providers.

    Returns (result_str, log_entry).
    """
    from flyto_ai.redaction import redact_args

    logger.info(
        "Tool call [%d]: %s(%s)",
        round_num + 1, func_name,
        json.dumps(redact_args(func_args))[:200],
    )

    fire_stream(on_stream, StreamEvent(
        type=StreamEventType.TOOL_START,
        tool_name=func_name,
        tool_args=func_args,
    ))

    result = await dispatch_fn(func_name, func_args)
    result_str = json.dumps(result, ensure_ascii=False, default=str)

    if len(result_str) > MAX_RESULT_LEN:
        result_str = result_str[:MAX_RESULT_LEN] + TRUNCATION_NOTE

    fire_stream(on_stream, StreamEvent(
        type=StreamEventType.TOOL_END,
        tool_name=func_name,
        tool_result=result if isinstance(result, dict) else {"raw": result_str[:MAX_PREVIEW_LEN]},
    ))

    log_entry: Dict[str, Any] = {
        "function": func_name,
        "arguments": func_args,
        "result_preview": result_str[:MAX_PREVIEW_LEN],
    }
    if func_name == "execute_module":
        log_entry["module_id"] = func_args.get("module_id", "")
        # flyto-core modules return {"status": "success"} without "ok" field.
        # Normalize: treat status=="success" as ok=True.
        if isinstance(result, dict):
            log_entry["ok"] = result.get("ok", result.get("status") == "success")
        else:
            log_entry["ok"] = False

    return result_str, log_entry


class LLMProvider(ABC):
    """Abstract base class for LLM providers (OpenAI, Anthropic, etc.)."""

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict],
        dispatch_fn: DispatchFn,
        max_rounds: int = 30,
        on_stream: Optional[StreamCallback] = None,
    ) -> Tuple[str, List[Dict[str, Any]], int, Dict[str, int]]:
        """Run a chat loop with function calling.

        Parameters
        ----------
        on_stream : callable, optional
            ``on_stream(StreamEvent)`` — called for each streaming event.
            When set, providers should enable streaming for LLM responses.
            When None, behaviour is unchanged (non-streaming).

        Returns (final_message, tool_call_log, rounds_used, usage_dict).
        """
