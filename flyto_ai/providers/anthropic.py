# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Anthropic provider (tool use loop)."""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from flyto_ai.models import StreamCallback, StreamEvent, StreamEventType
from flyto_ai.providers.base import DispatchFn, LLMProvider, fire_stream as _fire
from flyto_ai.redaction import redact_args

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with tool use loop."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    def __repr__(self) -> str:
        key_hint = "{}...".format(self._api_key[:4]) if self._api_key and len(self._api_key) > 4 else "***"
        return "AnthropicProvider(model={!r}, api_key={!r})".format(self._model, key_hint)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict],
        dispatch_fn: DispatchFn,
        max_rounds: int = 15,
        on_stream: Optional[StreamCallback] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        client = self._client

        # Convert to Anthropic tool format
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["inputSchema"],
            }
            for t in tools
        ]

        # Strip system messages (Anthropic uses system param)
        claude_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg["role"] in ("user", "assistant")
        ]

        tool_call_log: List[Dict[str, Any]] = []

        for round_num in range(max_rounds):
            create_kwargs = {
                "model": self._model,
                "system": system_prompt,
                "messages": claude_messages,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            }
            if anthropic_tools:
                create_kwargs["tools"] = anthropic_tools
                create_kwargs["tool_choice"] = {"type": "auto"}

            # ── Streaming path ──────────────────────────────────
            if on_stream is not None:
                async with client.messages.stream(**create_kwargs) as stream:
                    async for text in stream.text_stream:
                        _fire(on_stream, StreamEvent(
                            type=StreamEventType.TOKEN,
                            content=text,
                        ))
                    response = await stream.get_final_message()
            else:
                # ── Non-streaming path (unchanged) ──────────────
                response = await client.messages.create(**create_kwargs)

            has_tool_use = any(block.type == "tool_use" for block in response.content)

            if not has_tool_use:
                text_parts = [block.text for block in response.content if block.type == "text"]
                content = "\n".join(text_parts)
                if response.stop_reason == "max_tokens":
                    content += "\n\n[Note: Response was truncated due to token limit.]"
                _fire(on_stream, StreamEvent(type=StreamEventType.DONE))
                return content, tool_call_log

            claude_messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                func_name = block.name
                func_args = block.input if isinstance(block.input, dict) else {}

                logger.info(
                    "Tool call [%d]: %s(%s)",
                    round_num + 1, func_name,
                    json.dumps(redact_args(func_args))[:200],
                )

                _fire(on_stream, StreamEvent(
                    type=StreamEventType.TOOL_START,
                    tool_name=func_name,
                    tool_args=func_args,
                ))

                result = await dispatch_fn(func_name, func_args)
                result_str = json.dumps(result, ensure_ascii=False, default=str)

                if len(result_str) > 8000:
                    result_str = result_str[:8000] + "...(truncated)"

                _fire(on_stream, StreamEvent(
                    type=StreamEventType.TOOL_END,
                    tool_name=func_name,
                    tool_result=result if isinstance(result, dict) else {"raw": result_str[:500]},
                ))

                log_entry: Dict[str, Any] = {
                    "function": func_name,
                    "arguments": func_args,
                    "result_preview": result_str[:500],
                }
                # Structured result for execute_module tracking
                if func_name == "execute_module":
                    log_entry["module_id"] = func_args.get("module_id", "")
                    log_entry["ok"] = result.get("ok", False) if isinstance(result, dict) else False
                tool_call_log.append(log_entry)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            claude_messages.append({"role": "user", "content": tool_results})

        # Force final answer — append to last user message to avoid consecutive user roles
        summary_text = {"type": "text", "text": "Please summarize the results and provide your final answer."}
        if claude_messages and claude_messages[-1]["role"] == "user":
            content = claude_messages[-1]["content"]
            if isinstance(content, list):
                content.append(summary_text)
            else:
                claude_messages.append({"role": "assistant", "content": "I'll summarize now."})
                claude_messages.append({"role": "user", "content": [summary_text]})
        else:
            claude_messages.append({"role": "user", "content": [summary_text]})
        response = await client.messages.create(
            model=self._model,
            system=system_prompt,
            messages=claude_messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        text_parts = [block.text for block in response.content if block.type == "text"]
        _fire(on_stream, StreamEvent(type=StreamEventType.DONE))
        return "\n".join(text_parts), tool_call_log
