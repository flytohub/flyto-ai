# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""OpenAI and OpenAI-compatible provider (function calling loop)."""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from flyto_ai.models import StreamCallback, StreamEvent, StreamEventType
from flyto_ai.providers.base import (
    DispatchFn, LLMProvider, dispatch_and_log_tool, fire_stream as _fire,
)

logger = logging.getLogger(__name__)


_BROWSER_INTENT_PATTERNS = [
    "http://", "https://", ".com", ".org", ".net", ".tw", ".io", ".dev",
    "搜尋", "幫我到", "打開", "瀏覽", "上網", "查詢", "查一下", "看看",
    "google", "search", "browse", "website", "scrape", "go to",
    "tixcraft", "wikipedia", "youtube",
]


def _looks_like_browser_task(messages: list) -> bool:
    """Check if the user query requires browser automation."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = msg.get("content", "")
            if isinstance(text, str):
                lower = text.lower()
                return any(p in lower for p in _BROWSER_INTENT_PATTERNS)
            break
    return False


def _fill_remaining_tool_responses(
    tc_list: list,
    current_tc: Any,
    full_messages: list,
    *,
    id_key: str = "id",
) -> None:
    """Fill remaining tool calls with 'paused' responses after ask_user break.

    Works with both dict-style (streaming) and object-style (non-streaming) tool calls.
    """
    idx = tc_list.index(current_tc)
    for remaining in tc_list[idx + 1:]:
        tc_id = remaining[id_key] if isinstance(remaining, dict) else getattr(remaining, id_key)
        full_messages.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": '{"ok": false, "error": "Execution paused — waiting for user input"}',
        })


def _to_openai_tools(tools: List[Dict]) -> List[Dict[str, Any]]:
    """Convert Flyto2 tool definitions to OpenAI function tool definitions."""
    return [
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


def _accumulate_usage(total_usage: Dict[str, int], usage: Any) -> None:
    """Accumulate OpenAI usage objects into the provider usage dict."""
    if not usage:
        return
    total_usage["prompt_tokens"] += usage.prompt_tokens or 0
    total_usage["completion_tokens"] += usage.completion_tokens or 0
    total_usage["total_tokens"] += usage.total_tokens or 0


def _content_with_finish_note(content: str, finish_reason: Optional[str]) -> str:
    """Append the token-limit note when OpenAI truncated the response."""
    if finish_reason == "length":
        return content + "\n\n[Note: Response was truncated due to token limit.]"
    return content


def _tool_call_id(tc: Any) -> str:
    return tc["id"] if isinstance(tc, dict) else tc.id


def _tool_call_name(tc: Any) -> str:
    return tc["function"]["name"] if isinstance(tc, dict) else tc.function.name


def _tool_call_arguments(tc: Any) -> str:
    return tc["function"]["arguments"] if isinstance(tc, dict) else tc.function.arguments


def _parse_tool_args(arguments: str) -> Dict[str, Any]:
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {}


def _assistant_message_from_tool_calls(tool_calls: list):
    """Build an OpenAI assistant message containing tool calls and no text."""
    import openai.types.chat as _cht

    tool_call_objs = [
        _cht.ChatCompletionMessageToolCall(
            id=_tool_call_id(tc),
            type="function",
            function=_cht.chat_completion_message_tool_call.Function(
                name=_tool_call_name(tc),
                arguments=_tool_call_arguments(tc),
            ),
        )
        for tc in tool_calls
    ]
    return _cht.ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=tool_call_objs,
    )


def _vision_user_message(func_name: str, images: List[Dict[str, str]]) -> Dict[str, Any]:
    """Build a native OpenAI vision message from tool image sidebands."""
    vision_content = [{"type": "text", "text": "[Screenshot from {}]".format(func_name)}]
    for img in images:
        vision_content.append({
            "type": "image_url",
            "image_url": {
                "url": "data:{};base64,{}".format(img.get("media_type", "image/png"), img["base64"]),
                "detail": "low",
            },
        })
    return {"role": "user", "content": vision_content}


def _stream_tool_call_list(collected_tool_calls: Dict[int, Dict[str, Any]]) -> list:
    """Convert accumulated streaming tool deltas into OpenAI tool-call dicts."""
    return [
        {
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": tc["arguments"],
            },
        }
        for _, tc in sorted(collected_tool_calls.items())
    ]


async def _collect_stream_response(stream: Any, total_usage: Dict[str, int], on_stream: StreamCallback):
    """Collect text and tool calls from an OpenAI streaming response."""
    content_parts: List[str] = []
    collected_tool_calls: Dict[int, Dict[str, Any]] = {}
    finish_reason = None

    async for chunk in stream:
        if hasattr(chunk, "usage") and chunk.usage:
            _accumulate_usage(total_usage, chunk.usage)

        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue

        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

        if delta.content:
            content_parts.append(delta.content)
            _fire(on_stream, StreamEvent(
                type=StreamEventType.TOKEN,
                content=delta.content,
            ))

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in collected_tool_calls:
                    collected_tool_calls[idx] = {"id": tc_delta.id or "", "name": "", "arguments": ""}
                entry = collected_tool_calls[idx]
                if tc_delta.id:
                    entry["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        entry["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        entry["arguments"] += tc_delta.function.arguments

    return content_parts, collected_tool_calls, finish_reason


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
        self._client = None

    def __repr__(self) -> str:
        key_hint = "{}...".format(self._api_key[:4]) if self._api_key and len(self._api_key) > 4 else "***"
        return "OpenAIProvider(model={!r}, api_key={!r})".format(self._model, key_hint)

    def _make_client(self):
        if self._client is None:
            import openai
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def complete_json_schema(
        self,
        *,
        messages,
        schema,
        timeout_seconds: float = 120.0,
    ) -> Dict[str, Any]:
        """Return a schema-constrained completion, shaped like the other providers.

        Satisfies the StructuredJsonProvider boundary that
        ``flyto_ai.robotics_planning`` defines, so bounded planning has one
        provider interface rather than one per vendor.

        ``strict`` structured outputs rather than a prompt asking for JSON:
        the model then cannot return prose, a fence, or an extra field, which
        removes the three shapes a caller would otherwise have to reject. The
        caller still validates the *content* — a schema constrains the shape,
        and an enum baked into this call can be satisfied by a value the caller
        no longer accepts.

        Returned in the provider-native shape (``message.content``) that the
        adapter reads, so a caller does not have to know which vendor answered.
        """
        if not 1.0 <= timeout_seconds <= 300.0:
            raise ValueError("timeout_seconds must be between 1 and 300")

        normalized: List[Dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role not in {"system", "user", "assistant"}:
                raise ValueError("structured messages contain an unsupported role")
            if not content or len(content) > 256_000:
                raise ValueError("structured message content is empty or too large")
            normalized.append({"role": role, "content": content})
        if not normalized:
            raise ValueError("structured completion needs at least one message")

        client = self._make_client()
        response = await client.chat.completions.create(
            model=self._model,
            messages=normalized,
            temperature=0,
            max_tokens=self._max_tokens,
            timeout=timeout_seconds,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_reply",
                    "strict": True,
                    "schema": dict(schema),
                },
            },
        )
        choice = response.choices[0] if response.choices else None
        if choice is None:
            raise RuntimeError("structured completion returned no choices")
        # A length stop means the JSON is truncated, and truncated JSON parses
        # as a refusal downstream rather than as the wrong answer — say which it
        # was so an operator is not left guessing at a bad reply.
        if getattr(choice, "finish_reason", None) == "length":
            raise RuntimeError("structured completion was cut off by max_tokens")
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise RuntimeError("model refused the structured request: {}".format(str(refusal)[:200]))

        usage = getattr(response, "usage", None)
        return {
            "message": {"content": choice.message.content or ""},
            "model": getattr(response, "model", self._model),
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        }

    def _is_native_openai(self) -> bool:
        """True if talking to real OpenAI API (not Ollama / custom base_url)."""
        if not self._base_url:
            return True
        from urllib.parse import urlparse
        host = urlparse(self._base_url).hostname or ""
        return host == "api.openai.com" or host.endswith(".openai.com")

    def _create_kwargs(
        self,
        full_messages: List[Any],
        openai_tools: List[Dict[str, Any]],
        source_messages: List[Dict[str, Any]],
        round_num: int,
    ) -> Dict[str, Any]:
        create_kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if openai_tools:
            create_kwargs["tools"] = openai_tools
            if round_num == 0 and _looks_like_browser_task(source_messages):
                create_kwargs["tool_choice"] = "required"
            else:
                create_kwargs["tool_choice"] = "auto"
        return create_kwargs

    async def _dispatch_tool_calls(
        self,
        tool_calls: list,
        full_messages: List[Any],
        tool_call_log: List[Dict[str, Any]],
        dispatch_fn: DispatchFn,
        round_num: int,
        on_stream: Optional[StreamCallback] = None,
    ) -> bool:
        """Dispatch model-requested tools. Returns True when ask_user pauses."""
        tc_list = list(tool_calls)
        for tc in tc_list:
            func_name = _tool_call_name(tc)
            func_args = _parse_tool_args(_tool_call_arguments(tc))

            result_str, log_entry, images = await dispatch_and_log_tool(
                func_name, func_args, dispatch_fn, round_num, on_stream,
            )
            tool_call_log.append(log_entry)

            full_messages.append({
                "role": "tool",
                "tool_call_id": _tool_call_id(tc),
                "content": result_str,
            })

            if log_entry.get("result", {}).get("__ASK_USER__"):
                _fill_remaining_tool_responses(tc_list, tc, full_messages)
                return True

            if images and self._is_native_openai():
                full_messages.append(_vision_user_message(func_name, images))

        return False

    async def _finish_after_max_rounds(
        self,
        client: Any,
        full_messages: List[Any],
        tool_call_log: List[Dict[str, Any]],
        max_rounds: int,
        total_usage: Dict[str, int],
        on_stream: Optional[StreamCallback],
    ) -> Tuple[str, List[Dict[str, Any]], int, Dict[str, int]]:
        completed = [tc["function"] for tc in tool_call_log if tc.get("ok", True)]
        failed = [tc["function"] for tc in tool_call_log if not tc.get("ok", True)]
        summary_parts = [
            "You have used all {} tool rounds.".format(max_rounds),
            "Completed: {}".format(", ".join(completed[-5:]) if completed else "none"),
        ]
        if failed:
            summary_parts.append("Failed: {}".format(", ".join(failed[-3:])))
        summary_parts.append(
            "Please summarize what was accomplished, what remains incomplete, "
            "and suggest next steps."
        )
        full_messages.append({
            "role": "user",
            "content": " ".join(summary_parts),
        })
        response = await client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        _accumulate_usage(total_usage, response.usage)

        _fire(on_stream, StreamEvent(type=StreamEventType.DONE))
        return response.choices[0].message.content or "", tool_call_log, max_rounds, total_usage

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict],
        dispatch_fn: DispatchFn,
        max_rounds: int = 30,
        on_stream: Optional[StreamCallback] = None,
    ) -> Tuple[str, List[Dict[str, Any]], int, Dict[str, int]]:
        client = self._make_client()

        openai_tools = _to_openai_tools(tools)
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        tool_call_log: List[Dict[str, Any]] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for round_num in range(max_rounds):
            create_kwargs = self._create_kwargs(full_messages, openai_tools, messages, round_num)

            # ── Streaming path ──────────────────────────────────
            if on_stream is not None:
                create_kwargs["stream"] = True
                if self._is_native_openai():
                    create_kwargs["stream_options"] = {"include_usage": True}

                stream = await client.chat.completions.create(**create_kwargs)
                content_parts, collected_tool_calls, finish_reason = await _collect_stream_response(
                    stream, total_usage, on_stream,
                )
                if not collected_tool_calls:
                    content = _content_with_finish_note("".join(content_parts), finish_reason)
                    _fire(on_stream, StreamEvent(type=StreamEventType.DONE))
                    return content, tool_call_log, round_num + 1, total_usage

                tc_list = _stream_tool_call_list(collected_tool_calls)
                full_messages.append(_assistant_message_from_tool_calls(tc_list))
                if await self._dispatch_tool_calls(tc_list, full_messages, tool_call_log, dispatch_fn, round_num, on_stream):
                    text = "I need some information from you before I can continue."
                    _fire(on_stream, StreamEvent(type=StreamEventType.DONE))
                    return text, tool_call_log, round_num + 1, total_usage

                continue  # next round

            # ── Non-streaming path ──────────────────────────────
            response = await client.chat.completions.create(**create_kwargs)

            # Accumulate usage
            _accumulate_usage(total_usage, response.usage)

            choice = response.choices[0]

            if not choice.message.tool_calls:
                content = choice.message.content or ""
                return _content_with_finish_note(content, choice.finish_reason), tool_call_log, round_num + 1, total_usage

            # Strip text content from intermediate rounds to prevent fabrication.
            # Reconstruct message with content=None so the LLM generates text
            # only AFTER seeing actual tool results.
            tool_calls = list(choice.message.tool_calls)
            full_messages.append(_assistant_message_from_tool_calls(tool_calls))

            if await self._dispatch_tool_calls(tool_calls, full_messages, tool_call_log, dispatch_fn, round_num):
                text = "I need some information from you before I can continue."
                return text, tool_call_log, round_num + 1, total_usage

        return await self._finish_after_max_rounds(client, full_messages, tool_call_log, max_rounds, total_usage, on_stream)
