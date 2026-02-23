# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for AgentConfig — env/dict loading, defaults."""
import os

from flyto_ai.config import AgentConfig


class TestAgentConfig:

    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.provider == ""
        assert cfg.api_key == ""
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096

    def test_from_dict(self):
        cfg = AgentConfig.from_dict({
            "provider": "openai",
            "api_key": "sk-test",
            "model": "gpt-4o",
            "temperature": 0.5,
        })
        assert cfg.provider == "openai"
        assert cfg.api_key == "sk-test"
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.5

    def test_from_dict_defaults(self):
        cfg = AgentConfig.from_dict({})
        assert cfg.provider == ""
        assert cfg.max_tokens == 4096

    def test_resolved_model_openai(self):
        cfg = AgentConfig(provider="openai")
        assert cfg.resolved_model == "gpt-4o-mini"

    def test_resolved_model_anthropic(self):
        cfg = AgentConfig(provider="anthropic")
        assert cfg.resolved_model == "claude-sonnet-4-5-20250929"

    def test_resolved_model_custom(self):
        cfg = AgentConfig(provider="openai", model="gpt-4o")
        assert cfg.resolved_model == "gpt-4o"

    def test_from_env_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("FLYTO_AI_PROVIDER", raising=False)
        monkeypatch.delenv("FLYTO_AI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = AgentConfig.from_env()
        assert cfg.provider == "openai"
        assert cfg.api_key == "sk-test"

    def test_from_env_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("FLYTO_AI_PROVIDER", raising=False)
        monkeypatch.delenv("FLYTO_AI_API_KEY", raising=False)
        cfg = AgentConfig.from_env()
        assert cfg.provider == "anthropic"
        assert cfg.api_key == "sk-ant-test"

    def test_from_env_explicit_provider(self, monkeypatch):
        monkeypatch.setenv("FLYTO_AI_PROVIDER", "anthropic")
        monkeypatch.setenv("FLYTO_AI_API_KEY", "sk-explicit")
        cfg = AgentConfig.from_env()
        assert cfg.provider == "anthropic"
        assert cfg.api_key == "sk-explicit"
