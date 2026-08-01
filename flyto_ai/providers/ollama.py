# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Local Ollama provider (OpenAI-compatible endpoint)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from flyto_ai.models import StreamCallback, StreamEvent, StreamEventType
from flyto_ai.providers.base import DispatchFn, dispatch_and_log_tool, fire_stream
from flyto_ai.providers.openai import (
    OpenAIProvider,
    _content_with_finish_note,
    _to_openai_tools,
)


MAX_STRUCTURED_RESPONSE_BYTES = 512 * 1024


class OllamaStructuredOutputError(RuntimeError):
    """Raised when Ollama cannot return one bounded native completion."""


class OllamaProvider(OpenAIProvider):
    """Ollama provider using the native bounded chat endpoint."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        think: bool | str = False,
    ) -> None:
        super().__init__(
            api_key="ollama",  # Ollama doesn't need a real key
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
        )
        self._ollama_native_url = self._native_chat_url(base_url)
        if not isinstance(think, bool) and think not in {"low", "medium", "high", "max"}:
            raise ValueError("think must be a boolean or low, medium, high, or max")
        self._think = think

    @staticmethod
    def _native_chat_url(base_url: str) -> str:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Ollama base_url must be an HTTP(S) URL")
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return parsed._replace(
            path=f"{path}/api/chat",
            params="",
            query="",
            fragment="",
        ).geturl()

    async def complete_json_schema(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        timeout_seconds: float = 120.0,
    ) -> dict[str, Any]:
        """Return Ollama's native JSON-schema completion and provider counters."""

        if not 1.0 <= timeout_seconds <= 300.0:
            raise ValueError("timeout_seconds must be between 1 and 300")
        normalized_messages = []
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role not in {"system", "user", "assistant"}:
                raise ValueError("structured messages contain an unsupported role")
            if not content or len(content) > 256_000:
                raise ValueError("structured message content is empty or too large")
            normalized_messages.append({"role": role, "content": content})
        payload = {
            "model": self._model,
            "stream": False,
            "think": False,
            "format": dict(schema),
            "messages": normalized_messages,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        return await asyncio.to_thread(
            self._post_native_json,
            payload,
            timeout_seconds,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[dict],
        dispatch_fn: DispatchFn,
        max_rounds: int = 30,
        on_stream: StreamCallback | None = None,
    ) -> tuple[str, list[dict[str, Any]], int, dict[str, int]]:
        """Run Ollama's native tool loop with explicit thinking control."""

        full_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        full_messages.extend(dict(message) for message in messages)
        native_tools = _to_openai_tools(tools)
        tool_call_log: list[dict[str, Any]] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for round_num in range(max_rounds):
            response = await self._native_chat_request(full_messages, native_tools)
            self._accumulate_native_usage(usage, response)
            message = self._native_message(response)
            tool_calls = self._native_tool_calls(message)
            if not tool_calls:
                content = str(message.get("content") or "")
                content = _content_with_finish_note(
                    content,
                    str(response.get("done_reason") or "") or None,
                )
                if content:
                    fire_stream(
                        on_stream,
                        StreamEvent(type=StreamEventType.TOKEN, content=content),
                    )
                fire_stream(on_stream, StreamEvent(type=StreamEventType.DONE))
                return content, tool_call_log, round_num + 1, usage

            full_messages.append(self._native_assistant_tool_message(tool_calls))
            for call in tool_calls:
                function = call["function"]
                result_str, log_entry, _images = await dispatch_and_log_tool(
                    function["name"],
                    function["arguments"],
                    dispatch_fn,
                    round_num,
                    on_stream,
                )
                tool_call_log.append(log_entry)
                full_messages.append(
                    {
                        "role": "tool",
                        "tool_name": function["name"],
                        "content": result_str,
                    }
                )
                if log_entry.get("result", {}).get("__ASK_USER__"):
                    return (
                        "I need some information from you before I can continue.",
                        tool_call_log,
                        round_num + 1,
                        usage,
                    )

        return await self._finish_native_after_max_rounds(
            full_messages,
            tool_call_log,
            max_rounds,
            usage,
            on_stream,
        )

    async def _native_chat_request(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": False,
            "think": self._think,
            "messages": [dict(message) for message in messages],
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }
        if tools:
            payload["tools"] = [dict(tool) for tool in tools]
        return await asyncio.to_thread(self._post_native_json, payload, 300.0)

    @staticmethod
    def _native_message(response: Mapping[str, Any]) -> Mapping[str, Any]:
        message = response.get("message")
        if not isinstance(message, Mapping):
            raise OllamaStructuredOutputError(
                "Ollama native completion is missing message"
            )
        return message

    @staticmethod
    def _native_tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise OllamaStructuredOutputError(
                "Ollama native completion has invalid tool_calls"
            )
        calls: list[dict[str, Any]] = []
        for index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, Mapping):
                raise OllamaStructuredOutputError(
                    "Ollama native completion has an invalid tool call"
                )
            function = raw_call.get("function")
            if not isinstance(function, Mapping) or not isinstance(
                function.get("name"), str
            ):
                raise OllamaStructuredOutputError(
                    "Ollama native completion has an invalid tool function"
                )
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise OllamaStructuredOutputError(
                        "Ollama native completion has invalid tool arguments"
                    ) from exc
            if not isinstance(arguments, Mapping):
                raise OllamaStructuredOutputError(
                    "Ollama native completion has non-object tool arguments"
                )
            calls.append(
                {
                    "id": str(raw_call.get("id") or f"call_{index}"),
                    "function": {
                        "name": function["name"],
                        "arguments": dict(arguments),
                    },
                }
            )
        return calls

    @staticmethod
    def _native_assistant_tool_message(
        tool_calls: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "index": index,
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    },
                }
                for index, call in enumerate(tool_calls)
            ],
        }

    @staticmethod
    def _accumulate_native_usage(
        usage: dict[str, int],
        response: Mapping[str, Any],
    ) -> None:
        prompt_tokens = response.get("prompt_eval_count") or 0
        completion_tokens = response.get("eval_count") or 0
        if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
            raise OllamaStructuredOutputError(
                "Ollama native completion has invalid token counters"
            )
        usage["prompt_tokens"] += prompt_tokens
        usage["completion_tokens"] += completion_tokens
        usage["total_tokens"] += prompt_tokens + completion_tokens

    async def _finish_native_after_max_rounds(
        self,
        messages: list[dict[str, Any]],
        tool_call_log: list[dict[str, Any]],
        max_rounds: int,
        usage: dict[str, int],
        on_stream: StreamCallback | None,
    ) -> tuple[str, list[dict[str, Any]], int, dict[str, int]]:
        completed = [call["function"] for call in tool_call_log if call.get("ok", True)]
        failed = [call["function"] for call in tool_call_log if not call.get("ok", True)]
        summary = [
            f"You have used all {max_rounds} tool rounds.",
            "Completed: {}".format(", ".join(completed[-5:]) if completed else "none"),
        ]
        if failed:
            summary.append("Failed: {}".format(", ".join(failed[-3:])))
        summary.append(
            "Please summarize what was accomplished, what remains incomplete, "
            "and suggest next steps."
        )
        messages.append({"role": "user", "content": " ".join(summary)})
        response = await self._native_chat_request(messages, [])
        self._accumulate_native_usage(usage, response)
        message = self._native_message(response)
        content = _content_with_finish_note(
            str(message.get("content") or ""),
            str(response.get("done_reason") or "") or None,
        )
        if content:
            fire_stream(
                on_stream,
                StreamEvent(type=StreamEventType.TOKEN, content=content),
            )
        fire_stream(on_stream, StreamEvent(type=StreamEventType.DONE))
        return content, tool_call_log, max_rounds, usage

    def _post_native_json(
        self,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        request = Request(
            self._ollama_native_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read(MAX_STRUCTURED_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            detail = ""
            try:
                error_payload = exc.read(4097)
                if len(error_payload) <= 4096:
                    decoded_error = json.loads(error_payload.decode("utf-8"))
                    if isinstance(decoded_error, dict) and isinstance(
                        decoded_error.get("error"),
                        str,
                    ):
                        detail = decoded_error["error"][:1000]
            except (UnicodeError, json.JSONDecodeError, OSError):
                pass
            message = "Ollama structured completion failed"
            if detail:
                message = f"{message}: {detail}"
            raise OllamaStructuredOutputError(message) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaStructuredOutputError(
                "Ollama structured completion failed"
            ) from exc
        if len(raw) > MAX_STRUCTURED_RESPONSE_BYTES:
            raise OllamaStructuredOutputError(
                "Ollama structured completion exceeded the response limit"
            )
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise OllamaStructuredOutputError(
                "Ollama structured completion was not UTF-8 JSON"
            ) from exc
        if not isinstance(result, dict):
            raise OllamaStructuredOutputError(
                "Ollama structured completion must be an object"
            )
        message = result.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaStructuredOutputError(
                "Ollama structured completion is missing message.content"
            )
        return result
