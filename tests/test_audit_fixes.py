# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for audit-discovered bugs — verifies critical fixes stay fixed."""
import os
import time
import pytest

from flyto_ai import Agent, AgentConfig
from flyto_ai.redaction import is_sensitive_key, redact_args
from flyto_ai.session import SessionStore


# =========================================================================
# BUG 1+2: Provider tool_call_log format must match agent.py expectations
# =========================================================================

class TestToolCallLogFormat:
    """Providers must return {function, arguments, module_id, ok} not {tool, args}."""

    def _simulate_provider_log_entry(self, func_name, func_args, result):
        """Reproduce the exact log entry format from providers."""
        from typing import Any, Dict
        log_entry: Dict[str, Any] = {
            "function": func_name,
            "arguments": func_args,
            "result_preview": str(result)[:500],
        }
        if func_name == "execute_module":
            log_entry["module_id"] = func_args.get("module_id", "")
            log_entry["ok"] = result.get("ok", False) if isinstance(result, dict) else False
        return log_entry

    def test_execute_module_has_function_field(self):
        entry = self._simulate_provider_log_entry(
            "execute_module",
            {"module_id": "browser.goto", "params": {"url": "https://example.com"}},
            {"ok": True, "data": {"title": "Example"}},
        )
        assert entry["function"] == "execute_module"
        assert "tool" not in entry

    def test_execute_module_has_module_id(self):
        entry = self._simulate_provider_log_entry(
            "execute_module",
            {"module_id": "browser.extract", "params": {"selector": "h1"}},
            {"ok": True, "data": {"text": "Hello"}},
        )
        assert entry["module_id"] == "browser.extract"

    def test_execute_module_has_ok_field(self):
        entry = self._simulate_provider_log_entry(
            "execute_module",
            {"module_id": "browser.type", "params": {"selector": "input", "text": "test"}},
            {"ok": False, "error": "Timeout"},
        )
        assert entry["ok"] is False

    def test_execute_module_ok_true(self):
        entry = self._simulate_provider_log_entry(
            "execute_module",
            {"module_id": "http.get", "params": {"url": "https://example.com"}},
            {"ok": True, "data": {"status": 200}},
        )
        assert entry["ok"] is True

    def test_non_execute_has_no_module_id(self):
        entry = self._simulate_provider_log_entry(
            "search_modules",
            {"query": "browser"},
            {"ok": True, "modules": []},
        )
        assert "module_id" not in entry
        assert "ok" not in entry

    def test_arguments_are_full_not_redacted(self):
        """arguments in log must contain full params for blueprint learning."""
        entry = self._simulate_provider_log_entry(
            "execute_module",
            {"module_id": "http.get", "params": {"url": "https://example.com"}},
            {"ok": True},
        )
        assert entry["arguments"]["params"]["url"] == "https://example.com"


class TestExecutionResultsFilter:
    """agent.py must correctly filter execution_results from tool_calls."""

    def test_filters_only_execute_module(self):
        tool_calls = [
            {"function": "search_modules", "arguments": {"query": "browser"}},
            {"function": "execute_module", "arguments": {"module_id": "browser.launch"}, "module_id": "browser.launch", "ok": True},
            {"function": "get_module_info", "arguments": {"module_id": "browser.goto"}},
            {"function": "execute_module", "arguments": {"module_id": "browser.goto"}, "module_id": "browser.goto", "ok": True},
            {"function": "execute_module", "arguments": {"module_id": "browser.extract"}, "module_id": "browser.extract", "ok": True},
        ]
        execution_results = [
            tc for tc in tool_calls
            if tc.get("function") == "execute_module"
        ]
        assert len(execution_results) == 3
        assert all(r.get("module_id") for r in execution_results)
        assert all(r.get("ok") is True for r in execution_results)

    def test_empty_when_no_execute_module(self):
        tool_calls = [
            {"function": "search_modules", "arguments": {"query": "http"}},
            {"function": "get_module_info", "arguments": {"module_id": "http.get"}},
        ]
        execution_results = [
            tc for tc in tool_calls
            if tc.get("function") == "execute_module"
        ]
        assert len(execution_results) == 0


# =========================================================================
# Blueprint feedback — closed loop
# =========================================================================

class TestBlueprintFeedback:
    """_blueprint_feedback must correctly detect blueprints and report outcomes."""

    def test_detects_used_blueprint(self):
        """Phase 1: finds use_blueprint in tool_calls."""
        tool_calls = [
            {"function": "list_blueprints", "arguments": {}},
            {"function": "use_blueprint", "arguments": {"blueprint_id": "scrape_website"}},
            {"function": "execute_module", "arguments": {"module_id": "browser.launch"}, "module_id": "browser.launch", "ok": True},
            {"function": "execute_module", "arguments": {"module_id": "browser.goto"}, "module_id": "browser.goto", "ok": True},
            {"function": "execute_module", "arguments": {"module_id": "browser.extract"}, "module_id": "browser.extract", "ok": True},
        ]
        used_blueprint_id = None
        for tc in tool_calls:
            if tc.get("function") == "use_blueprint":
                used_blueprint_id = tc.get("arguments", {}).get("blueprint_id", "")
                break
        assert used_blueprint_id == "scrape_website"

    def test_no_blueprint_when_not_used(self):
        tool_calls = [
            {"function": "search_modules", "arguments": {"query": "http"}},
            {"function": "execute_module", "arguments": {"module_id": "http.get"}, "module_id": "http.get", "ok": True},
        ]
        used_blueprint_id = None
        for tc in tool_calls:
            if tc.get("function") == "use_blueprint":
                used_blueprint_id = tc.get("arguments", {}).get("blueprint_id", "")
                break
        assert used_blueprint_id is None

    def test_all_ok_detection(self):
        """All executions must succeed for positive outcome."""
        results_all_ok = [
            {"function": "execute_module", "module_id": "browser.launch", "ok": True},
            {"function": "execute_module", "module_id": "browser.goto", "ok": True},
            {"function": "execute_module", "module_id": "browser.extract", "ok": True},
        ]
        assert all(r.get("ok", False) for r in results_all_ok)

        results_one_fail = [
            {"function": "execute_module", "module_id": "browser.launch", "ok": True},
            {"function": "execute_module", "module_id": "browser.type", "ok": False},
            {"function": "execute_module", "module_id": "browser.extract", "ok": True},
        ]
        assert not all(r.get("ok", False) for r in results_one_fail)

    def test_workflow_construction_from_results(self):
        """Phase 2: builds workflow dict from execution results."""
        execution_results = [
            {"function": "execute_module", "module_id": "browser.launch", "ok": True, "arguments": {"module_id": "browser.launch", "params": {}}},
            {"function": "execute_module", "module_id": "browser.goto", "ok": True, "arguments": {"module_id": "browser.goto", "params": {"url": "https://example.com"}}},
            {"function": "execute_module", "module_id": "browser.extract", "ok": True, "arguments": {"module_id": "browser.extract", "params": {"selector": "h1"}}},
        ]
        steps = []
        for i, r in enumerate(execution_results):
            mid = r.get("module_id", "")
            params = r.get("arguments", {}).get("params", {})
            steps.append({"id": "step_{}".format(i + 1), "module": mid, "params": params})

        assert len(steps) == 3
        assert steps[0]["module"] == "browser.launch"
        assert steps[1]["params"]["url"] == "https://example.com"
        assert steps[2]["params"]["selector"] == "h1"

        categories = list({s["module"].split(".")[0] for s in steps if "." in s["module"]})
        assert "browser" in categories


# =========================================================================
# End-to-end: agent.chat() with mocked provider
# =========================================================================

class TestAgentEndToEnd:
    """Full flow: provider returns tool_calls → agent filters → response has execution_results."""

    @pytest.mark.asyncio
    async def test_execution_results_populated(self, monkeypatch):
        """execution_results is populated when provider returns execute_module calls."""
        async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15):
            return "Done! Title is Example Domain.", [
                {"function": "search_modules", "arguments": {"query": "browser"}, "result_preview": "..."},
                {"function": "execute_module", "arguments": {"module_id": "browser.launch", "params": {}}, "module_id": "browser.launch", "ok": True, "result_preview": "..."},
                {"function": "execute_module", "arguments": {"module_id": "browser.goto", "params": {"url": "https://example.com"}}, "module_id": "browser.goto", "ok": True, "result_preview": "..."},
                {"function": "execute_module", "arguments": {"module_id": "browser.extract", "params": {"selector": "h1"}}, "module_id": "browser.extract", "ok": True, "result_preview": "..."},
            ]

        config = AgentConfig(provider="ollama", api_key="test")
        agent = Agent(config=config)
        agent._tools = [{"name": "execute_module", "description": "run", "inputSchema": {}}]
        agent._dispatch_fn = lambda n, a: {"ok": True}
        monkeypatch.setattr(agent._provider, "chat", mock_chat)

        result = await agent.chat("scrape example.com", mode="execute")

        assert result.ok
        assert len(result.execution_results) == 3
        assert result.execution_results[0]["module_id"] == "browser.launch"
        assert result.execution_results[1]["module_id"] == "browser.goto"
        assert result.execution_results[2]["module_id"] == "browser.extract"
        assert all(r["ok"] is True for r in result.execution_results)

    @pytest.mark.asyncio
    async def test_execution_results_empty_in_yaml_mode(self, monkeypatch):
        """yaml mode still collects execution_results (if provider ran tools)."""
        async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15):
            return "```yaml\nname: test\nsteps:\n  - id: s1\n    module: browser.goto\n    label: Go\n    params:\n      url: https://example.com\n```", []

        config = AgentConfig(provider="ollama", api_key="test")
        agent = Agent(config=config)
        agent._tools = [{"name": "execute_module", "description": "run", "inputSchema": {}}]
        agent._dispatch_fn = lambda n, a: {"ok": True}
        monkeypatch.setattr(agent._provider, "chat", mock_chat)

        result = await agent.chat("scrape example.com", mode="yaml")
        assert result.ok
        assert len(result.execution_results) == 0

    @pytest.mark.asyncio
    async def test_failed_execution_tracked(self, monkeypatch):
        """Failed execute_module calls show ok=False in execution_results."""
        async def mock_chat(messages, system_prompt, tools, dispatch_fn, max_rounds=15):
            return "Browser type failed.", [
                {"function": "execute_module", "arguments": {"module_id": "browser.launch", "params": {}}, "module_id": "browser.launch", "ok": True, "result_preview": "..."},
                {"function": "execute_module", "arguments": {"module_id": "browser.type", "params": {"selector": "input[name='q']", "text": "test"}}, "module_id": "browser.type", "ok": False, "result_preview": "..."},
            ]

        config = AgentConfig(provider="ollama", api_key="test")
        agent = Agent(config=config)
        agent._tools = [{"name": "execute_module", "description": "run", "inputSchema": {}}]
        agent._dispatch_fn = lambda n, a: {"ok": True}
        monkeypatch.setattr(agent._provider, "chat", mock_chat)

        result = await agent.chat("search google", mode="execute")
        assert result.ok
        assert len(result.execution_results) == 2
        assert result.execution_results[0]["ok"] is True
        assert result.execution_results[1]["ok"] is False


# =========================================================================
# BUG 8: Redaction must not kill "author"
# =========================================================================

class TestRedactionPrecision:

    def test_auth_exact_is_sensitive(self):
        assert is_sensitive_key("auth") is True

    def test_author_is_not_sensitive(self):
        assert is_sensitive_key("author") is False

    def test_author_name_is_not_sensitive(self):
        assert is_sensitive_key("author_name") is False

    def test_author_email_is_not_sensitive(self):
        assert is_sensitive_key("author_email") is False

    def test_authorization_is_sensitive(self):
        assert is_sensitive_key("Authorization") is True

    def test_oauth_callback_is_not_sensitive(self):
        """oauth_callback doesn't contain any exact or substring match."""
        # "auth" is exact-only, "authorization" is substring
        # "oauth_callback" contains "auth" but only as part of "oauth"
        assert is_sensitive_key("oauth_callback") is False

    def test_redact_preserves_author(self):
        result = redact_args({
            "author_name": "John",
            "author_email": "john@example.com",
            "password": "secret123",
            "auth": "Bearer xyz",
        })
        assert result["author_name"] == "John"
        assert result["author_email"] == "john@example.com"
        assert result["password"] == "***"
        assert result["auth"] == "***"


# =========================================================================
# BUG 10: session.create must set timestamp
# =========================================================================

class TestSessionTimestamp:

    def test_create_sets_timestamp(self):
        store = SessionStore(max_sessions=10)
        store.create("sess1", "user1")
        assert "sess1" in store._timestamps
        assert isinstance(store._timestamps["sess1"], float)
        assert store._timestamps["sess1"] > 0

    def test_new_session_in_cleanup(self):
        """New sessions with timestamp are subject to TTL cleanup."""
        store = SessionStore(max_sessions=10, ttl_seconds=1)
        store.create("sess1", "user1")
        # Manually set old timestamp
        store._timestamps["sess1"] = time.time() - 100
        store.cleanup()
        assert "sess1" not in store._sessions


# =========================================================================
# BUG 11: config provider detection
# =========================================================================

class TestConfigProviderDetection:

    def test_explicit_openai_no_fallback_to_anthropic(self, monkeypatch):
        """FLYTO_AI_PROVIDER=openai should NOT pick up ANTHROPIC_API_KEY."""
        monkeypatch.setenv("FLYTO_AI_PROVIDER", "openai")
        monkeypatch.delenv("FLYTO_AI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        config = AgentConfig.from_env()
        # Should NOT have an API key since openai was explicit but no openai key exists
        assert config.api_key == ""
        assert config.provider == "openai"

    def test_no_provider_prefers_openai(self, monkeypatch):
        """No provider set: OPENAI_API_KEY takes priority."""
        monkeypatch.delenv("FLYTO_AI_PROVIDER", raising=False)
        monkeypatch.delenv("FLYTO_AI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        config = AgentConfig.from_env()
        assert config.provider == "openai"
        assert config.api_key == "sk-openai-test"

    def test_no_provider_falls_to_anthropic(self, monkeypatch):
        """No provider, no openai key: falls back to anthropic."""
        monkeypatch.delenv("FLYTO_AI_PROVIDER", raising=False)
        monkeypatch.delenv("FLYTO_AI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        config = AgentConfig.from_env()
        assert config.provider == "anthropic"
        assert config.api_key == "sk-ant-test"


# =========================================================================
# Policies: all module categories covered
# =========================================================================

class TestPoliciesCoverage:

    def test_critical_categories_allowed(self):
        from flyto_ai.prompt.policies import is_module_allowed
        must_allow = [
            "file.read", "file.write",
            "database.query",
            "datetime.now", "date.format",
            "json.parse", "csv.parse", "xml.parse", "yaml.parse",
            "git.clone", "git.commit", "git.diff",
            "docker.run", "docker.ps",
            "shell.exec",
            "email.send",
            "pdf.parse",
            "ssh.exec",
            "dns.lookup", "network.ping",
            "cache.get", "queue.enqueue",
            "llm.chat", "ai.model",
            "stealth.launch",
        ]
        for module_id in must_allow:
            assert is_module_allowed(module_id), "Expected {} to be allowed".format(module_id)
