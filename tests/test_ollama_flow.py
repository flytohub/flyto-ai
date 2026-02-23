# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for Ollama provider flow."""
import os

import pytest

from flyto_ai import Agent, AgentConfig


def test_from_env_ollama_no_key(monkeypatch):
    """FLYTO_AI_PROVIDER=ollama auto-fills api_key='ollama'."""
    monkeypatch.setenv("FLYTO_AI_PROVIDER", "ollama")
    monkeypatch.delenv("FLYTO_AI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    config = AgentConfig.from_env()
    assert config.provider == "ollama"
    assert config.api_key == "ollama"


def test_resolved_model_ollama():
    """Ollama defaults to llama3.2."""
    config = AgentConfig(provider="ollama")
    assert config.resolved_model == "llama3.2"


def test_resolved_model_ollama_custom():
    """Custom model overrides default."""
    config = AgentConfig(provider="ollama", model="mistral")
    assert config.resolved_model == "mistral"


def test_agent_ollama_no_api_key_error():
    """Ollama agent doesn't return 'no_api_key' error."""
    config = AgentConfig(provider="ollama")
    agent = Agent(config=config)
    # Agent should be created without error
    assert agent is not None
    assert agent._config.provider == "ollama"


@pytest.mark.asyncio
async def test_ollama_chat_no_api_key_error(monkeypatch):
    """chat() with ollama provider doesn't return no_api_key error."""
    config = AgentConfig(provider="ollama")
    agent = Agent(config=config)

    # Mock provider to avoid real LLM call
    async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15):
        return "Hello! I'm running on Ollama.", []

    monkeypatch.setattr(agent._provider, "chat", mock_chat)

    result = await agent.chat("hello")
    assert result.ok
    assert result.error != "no_api_key"


def test_from_env_fallback_openai(monkeypatch):
    """Falls back to openai when OPENAI_API_KEY is set."""
    monkeypatch.delenv("FLYTO_AI_PROVIDER", raising=False)
    monkeypatch.delenv("FLYTO_AI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    config = AgentConfig.from_env()
    assert config.provider == "openai"
    assert config.api_key == "sk-test"


def test_from_env_fallback_anthropic(monkeypatch):
    """Falls back to anthropic when ANTHROPIC_API_KEY is set."""
    monkeypatch.delenv("FLYTO_AI_PROVIDER", raising=False)
    monkeypatch.delenv("FLYTO_AI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    config = AgentConfig.from_env()
    assert config.provider == "anthropic"
    assert config.api_key == "sk-ant-test"
