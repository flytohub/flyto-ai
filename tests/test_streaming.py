# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for streaming callback support (Phase 1)."""
import pytest

from flyto_ai import Agent, AgentConfig
from flyto_ai.models import StreamEvent, StreamEventType


def _make_agent(monkeypatch, mock_chat_fn):
    """Helper: create agent with mocked LLM provider."""
    config = AgentConfig(provider="ollama", api_key="test")
    agent = Agent(config=config)
    agent._tools = [{"name": "execute_module", "description": "run", "inputSchema": {}}]

    async def _dispatch(name, args):
        return {"ok": True, "data": {"title": "Example"}}

    agent._dispatch_fn = _dispatch
    monkeypatch.setattr(agent._provider, "chat", mock_chat_fn)
    return agent


@pytest.mark.asyncio
async def test_on_stream_receives_tokens(monkeypatch):
    """on_stream callback receives TOKEN events from the provider."""
    events = []

    async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15, on_stream=None):
        # Simulate provider emitting tokens
        if on_stream:
            on_stream(StreamEvent(type=StreamEventType.TOKEN, content="Hello"))
            on_stream(StreamEvent(type=StreamEventType.TOKEN, content=" world"))
            on_stream(StreamEvent(type=StreamEventType.DONE))
        return "Hello world", []

    agent = _make_agent(monkeypatch, mock_chat)

    def _cb(event):
        events.append(event)

    result = await agent.chat("hi", mode="execute", on_stream=_cb)

    assert result.ok
    assert result.message == "Hello world"
    token_events = [e for e in events if e.type == StreamEventType.TOKEN]
    assert len(token_events) == 2
    assert token_events[0].content == "Hello"
    assert token_events[1].content == " world"
    assert any(e.type == StreamEventType.DONE for e in events)


@pytest.mark.asyncio
async def test_on_stream_none_backward_compat(monkeypatch):
    """on_stream=None preserves existing behaviour — no errors."""
    async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15, on_stream=None):
        assert on_stream is None
        return "Result", []

    agent = _make_agent(monkeypatch, mock_chat)
    result = await agent.chat("hi", mode="execute")

    assert result.ok
    assert result.message == "Result"


@pytest.mark.asyncio
async def test_tool_events_emitted(monkeypatch):
    """Agent dispatch wrapper emits TOOL_START and TOOL_END events."""
    events = []

    async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15, on_stream=None):
        # Simulate the provider calling dispatch_fn (which is instrumented)
        await dispatch_fn("search_modules", {"query": "browser"})
        await dispatch_fn("execute_module", {"module_id": "browser.goto"})
        return "Done!", [
            {"function": "search_modules"},
            {"function": "execute_module"},
        ]

    agent = _make_agent(monkeypatch, mock_chat)

    def _cb(event):
        events.append(event)

    result = await agent.chat("test", mode="execute", on_stream=_cb)

    assert result.ok
    tool_starts = [e for e in events if e.type == StreamEventType.TOOL_START]
    tool_ends = [e for e in events if e.type == StreamEventType.TOOL_END]
    assert len(tool_starts) == 2
    assert len(tool_ends) == 2
    assert tool_starts[0].tool_name == "search_modules"
    assert tool_starts[1].tool_name == "execute_module"


@pytest.mark.asyncio
async def test_callback_crash_safe(monkeypatch):
    """If on_stream callback raises in the provider _fire() wrapper, chat still completes.

    Also verifies agent-level dispatch wrapper catches callback crashes.
    """
    async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15, on_stream=None):
        # Real providers use _fire() which catches exceptions.
        # Simulate that: try/except around callback invocation.
        if on_stream:
            try:
                on_stream(StreamEvent(type=StreamEventType.TOKEN, content="x"))
            except Exception:
                pass  # provider _fire() catches this
        # Dispatch to trigger agent-level wrapper (tests TOOL_START/END crash safety)
        await dispatch_fn("search_modules", {"query": "test"})
        return "Done!", [{"function": "search_modules"}]

    agent = _make_agent(monkeypatch, mock_chat)

    def _bad_cb(event):
        raise RuntimeError("callback exploded")

    result = await agent.chat("test", mode="execute", on_stream=_bad_cb)
    assert result.ok
    assert result.message == "Done!"


@pytest.mark.asyncio
async def test_coexists_with_on_tool_call(monkeypatch):
    """on_stream and on_tool_call both fire on the same dispatch."""
    stream_events = []
    tool_calls = []

    async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15, on_stream=None):
        await dispatch_fn("search_modules", {"query": "test"})
        return "Done!", [{"function": "search_modules"}]

    agent = _make_agent(monkeypatch, mock_chat)

    def _stream_cb(event):
        stream_events.append(event)

    def _tool_cb(func_name, func_args):
        tool_calls.append(func_name)

    result = await agent.chat(
        "test", mode="execute",
        on_tool_call=_tool_cb,
        on_stream=_stream_cb,
    )

    assert result.ok
    assert tool_calls == ["search_modules"]
    assert any(e.type == StreamEventType.TOOL_START for e in stream_events)
    assert any(e.type == StreamEventType.TOOL_END for e in stream_events)
