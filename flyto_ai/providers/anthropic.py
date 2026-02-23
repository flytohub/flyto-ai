# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Anthropic provider (tool use loop)."""
import json
import logging
from typing import Any, Dict, List, Tuple

from flyto_ai.providers.base import DispatchFn, LLMProvider
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

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict],
        dispatch_fn: DispatchFn,
        max_rounds: int = 15,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)

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
            response = await client.messages.create(**create_kwargs)

            has_tool_use = any(block.type == "tool_use" for block in response.content)

            if not has_tool_use:
                text_parts = [block.text for block in response.content if block.type == "text"]
                return "\n".join(text_parts), tool_call_log

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

                result = await dispatch_fn(func_name, func_args)
                result_str = json.dumps(result, ensure_ascii=False, default=str)

                if len(result_str) > 8000:
                    result_str = result_str[:8000] + "...(truncated)"

                tool_call_log.append({
                    "tool": func_name,
                    "args": redact_args(func_args),
                    "result_preview": result_str[:500],
                })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            claude_messages.append({"role": "user", "content": tool_results})

        # Force final answer
        claude_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "Please summarize the results and provide your final answer."}],
        })
        response = await client.messages.create(
            model=self._model,
            system=system_prompt,
            messages=claude_messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        text_parts = [block.text for block in response.content if block.type == "text"]
        return "\n".join(text_parts), tool_call_log
