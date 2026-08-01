# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Native Ollama provider transport tests."""

from __future__ import annotations

from typing import Any

import pytest

from flyto_ai.models import StreamEventType
from flyto_ai.providers.ollama import OllamaProvider, OllamaStructuredOutputError


def test_ollama_provider_validates_think_control() -> None:
    assert OllamaProvider(think=False)._think is False
    assert OllamaProvider(think="low")._think == "low"

    with pytest.raises(ValueError, match="think"):
        OllamaProvider(think="unbounded")


@pytest.mark.asyncio
async def test_native_chat_runs_tool_loop_without_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OllamaProvider(model="qwen3:8b", temperature=0.0, max_tokens=512)
    payloads: list[dict[str, Any]] = []
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "write_file",
                                "arguments": {"path": "answer.txt", "content": "ok"},
                            },
                        }
                    ],
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 40,
                "eval_count": 12,
            },
            {
                "message": {"role": "assistant", "content": "Done."},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 55,
                "eval_count": 8,
            },
        ]
    )

    def fake_post(payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
        assert timeout_seconds == 300.0
        payloads.append(payload)
        return next(responses)

    monkeypatch.setattr(provider, "_post_native_json", fake_post)
    dispatched: list[tuple[str, dict[str, Any]]] = []

    async def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        dispatched.append((name, arguments))
        return {"ok": True}

    events = []
    answer, calls, rounds, usage = await provider.chat(
        messages=[{"role": "user", "content": "Write the answer."}],
        system_prompt="You edit files.",
        tools=[
            {
                "name": "write_file",
                "description": "Write one file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            }
        ],
        dispatch_fn=dispatch,
        on_stream=events.append,
    )

    assert answer == "Done."
    assert rounds == 2
    assert usage == {"prompt_tokens": 95, "completion_tokens": 20, "total_tokens": 115}
    assert dispatched == [("write_file", {"path": "answer.txt", "content": "ok"})]
    assert calls[0]["function"] == "write_file"
    assert all(payload["think"] is False for payload in payloads)
    assert payloads[0]["stream"] is False
    assert payloads[0]["options"] == {"temperature": 0.0, "num_predict": 512}
    assert payloads[0]["tools"][0]["function"]["name"] == "write_file"
    assert payloads[1]["messages"][-2]["tool_calls"][0]["function"]["arguments"] == {
        "path": "answer.txt",
        "content": "ok",
    }
    assert payloads[1]["messages"][-1] == {
        "role": "tool",
        "tool_name": "write_file",
        "content": '{"ok": true}',
    }
    assert [event.type for event in events] == [
        StreamEventType.TOOL_START,
        StreamEventType.TOOL_END,
        StreamEventType.TOKEN,
        StreamEventType.DONE,
    ]


@pytest.mark.asyncio
async def test_native_chat_pauses_after_ask_user_and_fills_parallel_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OllamaProvider()
    monkeypatch.setattr(
        provider,
        "_post_native_json",
        lambda payload, timeout_seconds: {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "ask_1",
                        "function": {"name": "ask_user", "arguments": {"question": "Choose"}},
                    },
                    {
                        "id": "write_1",
                        "function": {"name": "write_file", "arguments": {"path": "x"}},
                    },
                ],
            },
            "done": True,
            "prompt_eval_count": 3,
            "eval_count": 2,
        },
    )

    async def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"__ASK_USER__": True, "question": arguments["question"]}

    answer, calls, rounds, _usage = await provider.chat(
        messages=[{"role": "user", "content": "Help"}],
        system_prompt="Ask when blocked.",
        tools=[
            {"name": "ask_user", "description": "Ask", "inputSchema": {"type": "object"}},
            {"name": "write_file", "description": "Write", "inputSchema": {"type": "object"}},
        ],
        dispatch_fn=dispatch,
    )

    assert answer == "I need some information from you before I can continue."
    assert rounds == 1
    assert [call["function"] for call in calls] == ["ask_user"]


@pytest.mark.asyncio
async def test_native_chat_rejects_malformed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = OllamaProvider()
    monkeypatch.setattr(
        provider,
        "_post_native_json",
        lambda payload, timeout_seconds: {"done": True},
    )

    async def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    with pytest.raises(OllamaStructuredOutputError, match="missing message"):
        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="Be useful.",
            tools=[],
            dispatch_fn=dispatch,
        )
