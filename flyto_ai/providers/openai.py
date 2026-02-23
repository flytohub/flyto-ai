# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""OpenAI and OpenAI-compatible provider (function calling loop)."""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from flyto_ai.providers.base import DispatchFn, LLMProvider
from flyto_ai.redaction import redact_args

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI provider with function calling loop."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        base_url: Optional[str] = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._base_url = base_url

    def _make_client(self):
        import openai
        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return openai.AsyncOpenAI(**kwargs)

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict],
        dispatch_fn: DispatchFn,
        max_rounds: int = 15,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        client = self._make_client()

        # Convert to OpenAI function format
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"],
                },
            }
            for t in tools
        ]

        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        tool_call_log: List[Dict[str, Any]] = []

        for round_num in range(max_rounds):
            create_kwargs = {
                "model": self._model,
                "messages": full_messages,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            }
            if openai_tools:
                create_kwargs["tools"] = openai_tools
                create_kwargs["tool_choice"] = "auto"
            response = await client.chat.completions.create(**create_kwargs)

            choice = response.choices[0]

            if not choice.message.tool_calls:
                return choice.message.content or "", tool_call_log

            full_messages.append(choice.message)

            for tc in choice.message.tool_calls:
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

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

                full_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

        # Hit max rounds — ask for summary
        full_messages.append({
            "role": "user",
            "content": "Please summarize the results so far and provide your final answer.",
        })
        response = await client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return response.choices[0].message.content or "", tool_call_log
