# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for Agent execute vs yaml mode behaviour."""
import pytest

from flyto_ai import Agent, AgentConfig


def _make_agent(monkeypatch, mock_chat_fn):
    """Helper: create agent with mocked LLM provider."""
    config = AgentConfig(provider="ollama", api_key="test")
    agent = Agent(config=config)
    agent._tools = [{"name": "execute_module", "description": "run", "inputSchema": {}}]

    async def _dispatch(name, args):
        return {"ok": True, "data": {"title": "Example Domain"}}

    agent._dispatch_fn = _dispatch
    monkeypatch.setattr(agent._provider, "chat", mock_chat_fn)
    return agent


@pytest.mark.asyncio
async def test_execute_mode_skips_yaml_nudge(monkeypatch):
    """In execute mode, no YAML nudge is sent even when response has no YAML."""
    call_count = 0

    async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15):
        nonlocal call_count
        call_count += 1
        # Return plain text (no YAML) — in yaml mode this triggers nudge
        return "I executed the task. The title is Example Domain.", []

    agent = _make_agent(monkeypatch, mock_chat)
    result = await agent.chat("scrape example.com", mode="execute")

    assert result.ok
    assert call_count == 1  # No nudge follow-up call
    assert "Example Domain" in result.message


@pytest.mark.asyncio
async def test_yaml_mode_triggers_nudge(monkeypatch):
    """In yaml mode, missing YAML triggers a nudge follow-up call."""
    call_count = 0

    async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "Sure, I can help with that.", []  # No YAML → triggers nudge
        else:
            return "```yaml\nname: scrape\nsteps:\n  - id: s1\n    module: browser.goto\n    label: Go\n    params:\n      url: https://example.com\n```", []

    agent = _make_agent(monkeypatch, mock_chat)
    result = await agent.chat("scrape example.com", mode="yaml")

    assert result.ok
    assert call_count == 2  # Original + nudge
    assert "```yaml" in result.message


@pytest.mark.asyncio
async def test_execute_mode_collects_execution_results(monkeypatch):
    """Execute mode populates execution_results from execute_module tool calls."""
    async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15):
        return "Done!", [
            {"function": "search_modules", "args": {"query": "browser"}, "result": {"ok": True}},
            {"function": "execute_module", "args": {"module_id": "browser.goto"}, "result": {"ok": True}},
            {"function": "execute_module", "args": {"module_id": "browser.extract"}, "result": {"ok": True, "data": {"text": "Hello"}}},
        ]

    agent = _make_agent(monkeypatch, mock_chat)
    result = await agent.chat("scrape example.com", mode="execute")

    assert result.ok
    assert len(result.execution_results) == 2
    assert all(r["function"] == "execute_module" for r in result.execution_results)
    assert len(result.tool_calls) == 3  # All tool calls preserved


@pytest.mark.asyncio
async def test_execute_mode_uses_execute_prompt(monkeypatch):
    """Execute mode uses EXECUTE_SYSTEM_PROMPT, not DEFAULT."""
    captured_prompt = {}

    async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15):
        captured_prompt["value"] = system_prompt
        return "Done!", []

    agent = _make_agent(monkeypatch, mock_chat)
    await agent.chat("hello", mode="execute")

    assert "EXECUTE" in captured_prompt["value"]
    assert "ONLY generate" not in captured_prompt["value"]


@pytest.mark.asyncio
async def test_yaml_mode_uses_default_prompt(monkeypatch):
    """Yaml mode uses DEFAULT_SYSTEM_PROMPT."""
    captured_prompt = {}

    async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15):
        captured_prompt["value"] = system_prompt
        return "```yaml\nname: test\nsteps: []\n```", []

    agent = _make_agent(monkeypatch, mock_chat)
    await agent.chat("hello", mode="yaml")

    assert "YAML GENERATION LOOP" in captured_prompt["value"]
    assert "EXECUTION LOOP" not in captured_prompt["value"]


@pytest.mark.asyncio
async def test_custom_system_prompt_ignores_mode(monkeypatch):
    """When custom system_prompt is set, mode parameter doesn't affect prompt selection."""
    captured_prompt = {}
    custom = "You are a custom assistant."

    async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15):
        captured_prompt["value"] = system_prompt
        return "OK", []

    config = AgentConfig(provider="ollama", api_key="test")
    agent = Agent(config=config, system_prompt=custom)
    agent._tools = []
    agent._dispatch_fn = None
    monkeypatch.setattr(agent._provider, "chat", mock_chat)

    await agent.chat("hello", mode="execute")
    assert captured_prompt["value"] == custom

    await agent.chat("hello", mode="yaml")
    assert captured_prompt["value"] == custom


@pytest.mark.asyncio
async def test_on_tool_call_callback(monkeypatch):
    """on_tool_call callback is invoked for each tool dispatch."""
    calls = []

    async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15):
        # Simulate the provider calling dispatch_fn (which is instrumented)
        await dispatch_fn("search_modules", {"query": "browser"})
        await dispatch_fn("execute_module", {"module_id": "browser.goto"})
        return "Done!", [
            {"function": "search_modules"},
            {"function": "execute_module"},
        ]

    agent = _make_agent(monkeypatch, mock_chat)

    def _cb(func_name, func_args):
        calls.append(func_name)

    result = await agent.chat("test", mode="execute", on_tool_call=_cb)

    assert result.ok
    assert calls == ["search_modules", "execute_module"]
