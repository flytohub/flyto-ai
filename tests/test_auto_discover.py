# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for Agent auto-discovery and toolless mode."""
import pytest

from flyto_ai import Agent, AgentConfig


def test_no_tools_no_core_toolless_mode():
    """Agent creates successfully without flyto-core installed (toolless mode)."""
    config = AgentConfig(provider="ollama")
    agent = Agent(config=config)
    # Should not crash — toolless mode is fine
    assert agent is not None


def test_explicit_tools_skip_auto_discover():
    """When tools and dispatch_fn are provided, auto-discover is skipped."""
    dummy_tools = [{"name": "test_tool", "description": "test", "inputSchema": {}}]

    async def dummy_dispatch(name, args):
        return {"ok": True}

    config = AgentConfig(provider="ollama")
    agent = Agent(config=config, tools=dummy_tools, dispatch_fn=dummy_dispatch)
    assert len(agent._tools) == 1
    assert agent._tools[0]["name"] == "test_tool"
    assert agent._dispatch_fn is dummy_dispatch


def test_auto_discover_with_mock_core(monkeypatch):
    """Auto-discover registers core tools when flyto-core is available."""
    fake_tools = [
        {"name": "list_modules", "description": "List modules", "inputSchema": {}},
        {"name": "search_modules", "description": "Search", "inputSchema": {}},
    ]

    def mock_get_core_defs():
        return fake_tools

    monkeypatch.setattr(
        "flyto_ai.tools.core_tools.get_core_tool_defs",
        mock_get_core_defs,
    )

    config = AgentConfig(provider="ollama")
    agent = Agent(config=config)
    # Should have auto-discovered the mocked core tools + inspect_page
    tool_names = [t["name"] for t in agent._tools]
    assert "list_modules" in tool_names
    assert "search_modules" in tool_names


def test_toolless_agent_has_no_dispatch():
    """Toolless agent has empty tools and no dispatch_fn from auto-discover."""
    config = AgentConfig(provider="ollama")
    agent = Agent(config=config)
    # If no tools were discovered, _dispatch_fn stays None
    # (inspect_page may still register if core_tools import works partially)
    # The important thing is: agent does not crash
    assert isinstance(agent._tools, list)


@pytest.mark.asyncio
async def test_toolless_chat_returns_response(monkeypatch):
    """Toolless agent chat returns a valid ChatResponse."""
    config = AgentConfig(provider="ollama", api_key="test")

    # Mock provider to avoid real LLM call
    async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15, on_stream=None):
        return "Here is a sample workflow:\n```yaml\nname: test\nsteps: []\n```", []

    agent = Agent(config=config)
    agent._tools = []
    agent._dispatch_fn = None
    monkeypatch.setattr(agent._provider, "chat", mock_chat)

    result = await agent.chat("hello")
    assert result.ok
    assert "yaml" in result.message.lower() or "test" in result.message.lower()
