from types import SimpleNamespace

import pytest

from flyto_ai.providers.openai import OpenAIProvider


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No fake OpenAI responses left")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def _usage(prompt=1, completion=2, total=3):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def _message(content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _choice(message, finish_reason="stop"):
    return SimpleNamespace(message=message, finish_reason=finish_reason)


def _response(message, *, finish_reason="stop", usage=None):
    return SimpleNamespace(
        usage=usage,
        choices=[_choice(message, finish_reason=finish_reason)],
    )


def _tool_call(call_id="call_1", name="demo_tool", arguments='{"x": 1}'):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


async def _dispatch(name, args):
    return {"ok": True, "name": name, "args": args}


@pytest.mark.asyncio
async def test_openai_chat_returns_text_and_usage_without_tools():
    provider = OpenAIProvider(api_key="test")
    provider._client = FakeClient([
        _response(_message("done"), usage=_usage(3, 4, 7)),
    ])

    content, tool_log, rounds, usage = await provider.chat(
        [{"role": "user", "content": "hello"}],
        "system",
        [],
        _dispatch,
    )

    assert content == "done"
    assert tool_log == []
    assert rounds == 1
    assert usage == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}


@pytest.mark.asyncio
async def test_openai_chat_dispatches_non_streaming_tool_call_then_finishes():
    provider = OpenAIProvider(api_key="test")
    first = _response(_message(tool_calls=[_tool_call(name="demo_tool", arguments='{"x": 42}')]))
    second = _response(_message("final"), usage=_usage(2, 5, 7))
    provider._client = FakeClient([first, second])

    content, tool_log, rounds, usage = await provider.chat(
        [{"role": "user", "content": "run tool"}],
        "system",
        [{"name": "demo_tool", "description": "Demo", "inputSchema": {"type": "object"}}],
        _dispatch,
        max_rounds=3,
    )

    assert content == "final"
    assert rounds == 2
    assert usage["total_tokens"] == 7
    assert tool_log[0]["function"] == "demo_tool"
    assert tool_log[0]["arguments"] == {"x": 42}
    messages_after_first_call = provider._client.completions.calls[0]["messages"]
    assert any(msg.get("role") == "tool" and msg["tool_call_id"] == "call_1" for msg in messages_after_first_call if isinstance(msg, dict))


@pytest.mark.asyncio
async def test_openai_chat_pauses_and_fills_remaining_tool_calls_for_ask_user():
    provider = OpenAIProvider(api_key="test")
    first = _response(_message(tool_calls=[
        _tool_call(call_id="call_1", name="ask_user", arguments='{"question": "email?"}'),
        _tool_call(call_id="call_2", name="demo_tool", arguments='{"x": 2}'),
    ]))
    provider._client = FakeClient([first])

    async def dispatch(name, args):
        assert name == "ask_user"
        return {"__ASK_USER__": True}

    content, tool_log, rounds, usage = await provider.chat(
        [{"role": "user", "content": "need input"}],
        "system",
        [
            {"name": "ask_user", "description": "Ask", "inputSchema": {"type": "object"}},
            {"name": "demo_tool", "description": "Demo", "inputSchema": {"type": "object"}},
        ],
        dispatch,
        max_rounds=3,
    )

    assert content == "I need some information from you before I can continue."
    assert rounds == 1
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert tool_log[0]["result"]["__ASK_USER__"] is True
    messages_after_first_call = provider._client.completions.calls[0]["messages"]
    paused = [
        msg for msg in messages_after_first_call
        if isinstance(msg, dict) and msg.get("tool_call_id") == "call_2"
    ]
    assert paused and "Execution paused" in paused[0]["content"]
