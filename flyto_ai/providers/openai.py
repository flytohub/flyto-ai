# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""OpenAI and OpenAI-compatible provider (function calling loop)."""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from flyto_ai.models import StreamCallback, StreamEvent, StreamEventType
from flyto_ai.providers.base import DispatchFn, LLMProvider
from flyto_ai.redaction import redact_args

logger = logging.getLogger(__name__)


def _fire(on_stream: Optional[StreamCallback], event: StreamEvent) -> None:
    """Safely invoke the stream callback."""
    if on_stream is None:
        return
    try:
        on_stream(event)
    except Exception:
        pass  # callback crash must never break the chat loop


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

    def _make_client(self):
        if self._client is None:
            import openai
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict],
        dispatch_fn: DispatchFn,
        max_rounds: int = 15,
        on_stream: Optional[StreamCallback] = None,
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

            # ── Streaming path ──────────────────────────────────
            if on_stream is not None:
                create_kwargs["stream"] = True
                content_parts: List[str] = []
                collected_tool_calls: Dict[int, Dict[str, Any]] = {}
                finish_reason = None

                stream = await client.chat.completions.create(**create_kwargs)
                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue

                    # Finish reason
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

                    # Text content
                    if delta.content:
                        content_parts.append(delta.content)
                        _fire(on_stream, StreamEvent(
                            type=StreamEventType.TOKEN,
                            content=delta.content,
                        ))

                    # Tool call deltas — accumulate by index
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in collected_tool_calls:
                                collected_tool_calls[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": "",
                                    "arguments": "",
                                }
                            entry = collected_tool_calls[idx]
                            if tc_delta.id:
                                entry["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    entry["name"] += tc_delta.function.name
                                if tc_delta.function.arguments:
                                    entry["arguments"] += tc_delta.function.arguments

                # No tool calls → return content
                if not collected_tool_calls:
                    content = "".join(content_parts)
                    if finish_reason == "length":
                        content += "\n\n[Note: Response was truncated due to token limit.]"
                    _fire(on_stream, StreamEvent(type=StreamEventType.DONE))
                    return content, tool_call_log

                # Reconstruct assistant message for conversation history
                tc_list = []
                for idx in sorted(collected_tool_calls.keys()):
                    tc = collected_tool_calls[idx]
                    tc_list.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    })

                # Build a message object compatible with the OpenAI API
                import openai.types.chat as _cht
                assistant_tc_objs = [
                    _cht.ChatCompletionMessageToolCall(
                        id=tc["id"],
                        type="function",
                        function=_cht.chat_completion_message_tool_call.Function(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                    for tc in tc_list
                ]
                assistant_msg = _cht.ChatCompletionMessage(
                    role="assistant",
                    content="".join(content_parts) or None,
                    tool_calls=assistant_tc_objs,
                )
                full_messages.append(assistant_msg)

                # Dispatch each tool call
                for tc in tc_list:
                    func_name = tc["function"]["name"]
                    try:
                        func_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        func_args = {}

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
                    if func_name == "execute_module":
                        log_entry["module_id"] = func_args.get("module_id", "")
                        log_entry["ok"] = result.get("ok", False) if isinstance(result, dict) else False
                    tool_call_log.append(log_entry)

                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_str,
                    })

                continue  # next round

            # ── Non-streaming path (unchanged) ──────────────────
            response = await client.chat.completions.create(**create_kwargs)

            choice = response.choices[0]

            if not choice.message.tool_calls:
                content = choice.message.content or ""
                if choice.finish_reason == "length":
                    content += "\n\n[Note: Response was truncated due to token limit.]"
                return content, tool_call_log

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
        _fire(on_stream, StreamEvent(type=StreamEventType.DONE))
        return response.choices[0].message.content or "", tool_call_log
