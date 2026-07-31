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

from flyto_ai.providers.openai import OpenAIProvider


MAX_STRUCTURED_RESPONSE_BYTES = 512 * 1024


class OllamaStructuredOutputError(RuntimeError):
    """Raised when Ollama cannot return one bounded structured completion."""


class OllamaProvider(OpenAIProvider):
    """Ollama provider using the OpenAI-compatible API endpoint."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(
            api_key="ollama",  # Ollama doesn't need a real key
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
        )
        self._ollama_native_url = self._native_chat_url(base_url)

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
