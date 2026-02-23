# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Deep integration tests — 10 scenarios covering streaming, browser retry,
model config, and cross-cutting concerns. All 10 must pass consecutively."""
import json
import pytest

from flyto_ai import Agent, AgentConfig, StreamEvent, StreamEventType
from flyto_ai.config import FUNCTION_CALLING_SUPPORT
from flyto_ai.models import StreamCallback
from flyto_ai.tools.core_tools import (
    _is_transient_error,
    _is_session_dead,
    dispatch_core_tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agent(monkeypatch, mock_chat_fn, *, provider="ollama"):
    config = AgentConfig(provider=provider, api_key="test")
    agent = Agent(config=config)
    agent._tools = [
        {"name": "execute_module", "description": "run", "inputSchema": {}},
        {"name": "search_modules", "description": "search", "inputSchema": {}},
    ]
    async def _dispatch(name, args):
        return {"ok": True, "data": {}}
    agent._dispatch_fn = _dispatch
    monkeypatch.setattr(agent._provider, "chat", mock_chat_fn)
    return agent


# ============================================================================
# TEST 1: Streaming tokens arrive in order, DONE event fires last
# ============================================================================
@pytest.mark.asyncio
async def test_01_streaming_token_ordering(monkeypatch):
    """Tokens arrive in order and DONE is the last event type."""
    events = []
    chunks = ["The ", "answer ", "is ", "42."]

    async def mock_chat(messages, system_prompt, tools, dispatch_fn,
                        max_rounds=30, on_stream=None):
        for c in chunks:
            if on_stream:
                on_stream(StreamEvent(type=StreamEventType.TOKEN, content=c))
        if on_stream:
            on_stream(StreamEvent(type=StreamEventType.DONE))
        return "The answer is 42.", [], 1, {}

    agent = _agent(monkeypatch, mock_chat)
    result = await agent.chat("what is 6*7?", on_stream=lambda e: events.append(e))

    assert result.ok
    token_contents = [e.content for e in events if e.type == StreamEventType.TOKEN]
    assert token_contents == chunks
    assert events[-1].type == StreamEventType.DONE


# ============================================================================
# TEST 2: Streaming + tool dispatch interleaved correctly
# ============================================================================
@pytest.mark.asyncio
async def test_02_streaming_with_tools_interleaved(monkeypatch):
    """Streaming interleaves TOKEN and TOOL_START/END in correct order."""
    events = []

    async def mock_chat(messages, system_prompt, tools, dispatch_fn,
                        max_rounds=30, on_stream=None):
        # Simulate: search → tokens → execute → more tokens
        await dispatch_fn("search_modules", {"query": "http"})
        if on_stream:
            on_stream(StreamEvent(type=StreamEventType.TOKEN, content="Found "))
        await dispatch_fn("execute_module", {"module_id": "api.get"})
        if on_stream:
            on_stream(StreamEvent(type=StreamEventType.TOKEN, content="result."))
            on_stream(StreamEvent(type=StreamEventType.DONE))
        return "Found result.", [
            {"function": "search_modules"},
            {"function": "execute_module"},
        ], 1, {}

    agent = _agent(monkeypatch, mock_chat)
    result = await agent.chat("test", on_stream=lambda e: events.append(e))

    assert result.ok
    types = [e.type for e in events]
    # Agent dispatch wrapper fires TOOL_START and TOOL_END around each dispatch
    assert StreamEventType.TOOL_START in types
    assert StreamEventType.TOOL_END in types
    assert StreamEventType.TOKEN in types
    # TOOL_START should come before corresponding TOOL_END
    starts = [i for i, t in enumerate(types) if t == StreamEventType.TOOL_START]
    ends = [i for i, t in enumerate(types) if t == StreamEventType.TOOL_END]
    assert len(starts) == len(ends) == 2
    for s, e in zip(starts, ends):
        assert s < e


# ============================================================================
# TEST 3: Multiple on_stream callbacks crash — chat survives all
# ============================================================================
@pytest.mark.asyncio
async def test_03_multiple_callback_crashes(monkeypatch):
    """Even if on_stream raises on every event type, chat completes."""
    crash_count = [0]

    async def mock_chat(messages, system_prompt, tools, dispatch_fn,
                        max_rounds=30, on_stream=None):
        await dispatch_fn("search_modules", {"query": "test"})
        return "OK", [{"function": "search_modules"}], 1, {}

    agent = _agent(monkeypatch, mock_chat)

    def _crasher(event):
        crash_count[0] += 1
        raise ValueError("crash #{}".format(crash_count[0]))

    result = await agent.chat("test", on_stream=_crasher)
    assert result.ok
    assert crash_count[0] >= 2  # At least TOOL_START + TOOL_END


# ============================================================================
# TEST 4: Browser retry — timeout then success, context preserved
# ============================================================================
@pytest.mark.asyncio
async def test_04_browser_retry_preserves_context(monkeypatch):
    """After transient retry, the final result contains correct data."""
    attempt = [0]

    async def mock_execute(module_id, params, context, browser_sessions):
        attempt[0] += 1
        if attempt[0] == 1:
            return {"ok": False, "error": "Navigation timeout of 30000ms exceeded"}
        return {"ok": True, "data": {"text": "Hello World", "url": params.get("url", "")}}

    fake_handler = {"execute_module": mock_execute}
    monkeypatch.setattr("flyto_ai.tools.core_tools._get_mcp_handler", lambda: fake_handler)

    result = await dispatch_core_tool("execute_module", {
        "module_id": "browser.goto",
        "params": {"url": "https://example.com"},
    })
    assert result["ok"] is True
    assert result["data"]["text"] == "Hello World"
    assert result["data"]["url"] == "https://example.com"


# ============================================================================
# TEST 5: Browser retry — session dead → relaunch → retry → success
# ============================================================================
@pytest.mark.asyncio
async def test_05_session_dead_full_recovery(monkeypatch):
    """Dead session triggers relaunch, then retries the original call."""
    call_log = []

    async def mock_execute(module_id, params, context, browser_sessions):
        call_log.append({"module": module_id, "params": params})
        if module_id == "browser.launch":
            return {"ok": True, "data": {"session_id": "sess-new"}}
        if len([c for c in call_log if c["module"] == "browser.extract"]) == 1:
            return {"ok": False, "error": "Session closed unexpectedly"}
        return {"ok": True, "data": {"items": ["a", "b", "c"]}}

    fake_handler = {"execute_module": mock_execute}
    monkeypatch.setattr("flyto_ai.tools.core_tools._get_mcp_handler", lambda: fake_handler)

    result = await dispatch_core_tool("execute_module", {
        "module_id": "browser.extract",
        "params": {"selector": ".item"},
    })
    assert result["ok"] is True
    assert result["data"]["items"] == ["a", "b", "c"]
    modules_called = [c["module"] for c in call_log]
    assert modules_called == ["browser.extract", "browser.launch", "browser.extract"]


# ============================================================================
# TEST 6: Non-browser errors NEVER retry (even with transient-like messages)
# ============================================================================
@pytest.mark.asyncio
async def test_06_non_browser_never_retries(monkeypatch):
    """api.get with a timeout error does not retry."""
    calls = [0]

    async def mock_execute(module_id, params, context, browser_sessions):
        calls[0] += 1
        return {"ok": False, "error": "Connection timed out after 30s"}

    fake_handler = {"execute_module": mock_execute}
    monkeypatch.setattr("flyto_ai.tools.core_tools._get_mcp_handler", lambda: fake_handler)

    result = await dispatch_core_tool("execute_module", {
        "module_id": "api.get",
        "params": {"url": "https://slow-api.example.com"},
    })
    assert result["ok"] is False
    assert calls[0] == 1


# ============================================================================
# TEST 7: Streaming + on_tool_call + on_stream coexist across full chat
# ============================================================================
@pytest.mark.asyncio
async def test_07_full_chat_both_callbacks(monkeypatch):
    """Complete chat flow with both on_tool_call and on_stream active."""
    stream_events = []
    tool_calls = []

    async def mock_chat(messages, system_prompt, tools, dispatch_fn,
                        max_rounds=30, on_stream=None):
        # 3 tool calls + streamed response
        await dispatch_fn("search_modules", {"query": "browser"})
        await dispatch_fn("execute_module", {"module_id": "browser.launch"})
        await dispatch_fn("execute_module", {"module_id": "browser.goto"})
        if on_stream:
            on_stream(StreamEvent(type=StreamEventType.TOKEN, content="Done!"))
            on_stream(StreamEvent(type=StreamEventType.DONE))
        return "Done!", [
            {"function": "search_modules"},
            {"function": "execute_module", "module_id": "browser.launch", "ok": True},
            {"function": "execute_module", "module_id": "browser.goto", "ok": True},
        ], 1, {}

    agent = _agent(monkeypatch, mock_chat)
    result = await agent.chat(
        "scrape example.com",
        mode="execute",
        on_tool_call=lambda name, args: tool_calls.append(name),
        on_stream=lambda e: stream_events.append(e),
    )

    assert result.ok
    assert len(tool_calls) == 3
    assert tool_calls == ["search_modules", "execute_module", "execute_module"]

    start_events = [e for e in stream_events if e.type == StreamEventType.TOOL_START]
    end_events = [e for e in stream_events if e.type == StreamEventType.TOOL_END]
    token_events = [e for e in stream_events if e.type == StreamEventType.TOKEN]
    assert len(start_events) == 3
    assert len(end_events) == 3
    assert len(token_events) == 1
    assert token_events[0].content == "Done!"


# ============================================================================
# TEST 8: FUNCTION_CALLING_SUPPORT config completeness
# ============================================================================
def test_08_model_compatibility_table():
    """FUNCTION_CALLING_SUPPORT has all expected models and valid ratings."""
    valid_ratings = {"excellent", "good", "fair", "poor"}

    # Must have cloud models
    assert "gpt-4o" in FUNCTION_CALLING_SUPPORT
    assert "gpt-4o-mini" in FUNCTION_CALLING_SUPPORT
    assert "claude-sonnet-4-5-20250929" in FUNCTION_CALLING_SUPPORT

    # Must have local models
    assert "qwen2.5:7b" in FUNCTION_CALLING_SUPPORT
    assert "llama3.2" in FUNCTION_CALLING_SUPPORT

    # All ratings are valid
    for model, rating in FUNCTION_CALLING_SUPPORT.items():
        assert rating in valid_ratings, "{}: invalid rating '{}'".format(model, rating)

    # At least 8 models
    assert len(FUNCTION_CALLING_SUPPORT) >= 8


# ============================================================================
# TEST 9: Error classification edge cases — mixed case, substrings, unicode
# ============================================================================
def test_09_error_classification_edge_cases():
    """Transient/session error classification handles edge cases."""
    # Mixed case
    assert _is_transient_error("Navigation TIMEOUT of 30000ms")
    assert _is_transient_error("TARGET CLOSED due to crash")

    # Substring in longer message
    assert _is_transient_error("Error: net::ERR_CONNECTION_REFUSED at https://example.com")
    assert _is_transient_error("Page crashed while loading resources")

    # Session dead subset of transient
    assert _is_session_dead("Browser has been closed by the user")
    assert _is_transient_error("Target closed")  # also transient
    assert _is_session_dead("Target closed")     # also session dead

    # Definitely NOT transient
    assert not _is_transient_error("Element '#btn' not found in DOM")
    assert not _is_transient_error("Invalid URL format")
    assert not _is_transient_error("Module 'browser.foo' does not exist")
    assert not _is_transient_error("")

    # Empty string is not session dead
    assert not _is_session_dead("")


# ============================================================================
# TEST 10: Full agent lifecycle — no API key for non-ollama, then ollama OK
# ============================================================================
@pytest.mark.asyncio
async def test_10_agent_lifecycle_cross_provider(monkeypatch):
    """Non-ollama without key → error; ollama → OK; streaming through both."""
    # Part A: no key → error
    config_no_key = AgentConfig(provider="openai", api_key="")
    agent_no_key = Agent(config=config_no_key)
    result_err = await agent_no_key.chat("hello")
    assert not result_err.ok
    assert result_err.error == "no_api_key"

    # Part B: ollama → ok, with streaming
    events = []

    async def mock_chat(messages, system_prompt, tools, dispatch_fn,
                        max_rounds=30, on_stream=None):
        if on_stream:
            on_stream(StreamEvent(type=StreamEventType.TOKEN, content="Hi"))
            on_stream(StreamEvent(type=StreamEventType.DONE))
        return "Hi from Ollama!", [], 1, {}

    config_ok = AgentConfig(provider="ollama", api_key="ollama")
    agent_ok = Agent(config=config_ok)
    agent_ok._tools = []
    agent_ok._dispatch_fn = None
    monkeypatch.setattr(agent_ok._provider, "chat", mock_chat)

    result_ok = await agent_ok.chat("hello", on_stream=lambda e: events.append(e))
    assert result_ok.ok
    assert result_ok.message == "Hi from Ollama!"
    assert result_ok.provider == "ollama"
    token_events = [e for e in events if e.type == StreamEventType.TOKEN]
    assert len(token_events) == 1
    assert token_events[0].content == "Hi"
