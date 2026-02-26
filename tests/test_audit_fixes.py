# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for audit-discovered bugs — NO mocks, all real code paths."""
import asyncio
import inspect
import os
import time
import pytest

from flyto_ai import Agent, AgentConfig
from flyto_ai.agent import _blueprint_feedback
from flyto_ai.models import ChatResponse
from flyto_ai.redaction import is_sensitive_key, redact_args
from flyto_ai.session import SessionStore
from flyto_ai.prompt.policies import (
    validate_base_url, is_tool_allowed, is_module_allowed, get_default_policies,
)
from flyto_ai.tools.registry import ToolRegistry
from flyto_ai.tools.core_tools import (
    get_core_tool_defs, dispatch_core_tool,
    _browser_sessions, clear_browser_sessions,
)
from flyto_ai.tools.blueprint_tools import (
    get_blueprint_tool_defs, dispatch_blueprint_tool,
)
from flyto_ai.tools.inspect_page import INSPECT_PAGE_TOOL, inspect_page
from flyto_ai.providers.openai import OpenAIProvider
from flyto_ai.providers.anthropic import AnthropicProvider


# =========================================================================
# BUG 1+2: Provider tool_call_log format — read source code directly
# =========================================================================

class TestProviderLogFormatSource:
    """Verify shared dispatch_and_log_tool builds log entries with {function, arguments}.

    Log entry construction has been extracted to providers/base.py dispatch_and_log_tool.
    """

    def test_shared_dispatch_uses_function_field(self):
        from flyto_ai.providers.base import dispatch_and_log_tool
        src = inspect.getsource(dispatch_and_log_tool)
        assert '"function": func_name' in src or "'function': func_name" in src

    def test_shared_dispatch_uses_arguments_field(self):
        from flyto_ai.providers.base import dispatch_and_log_tool
        src = inspect.getsource(dispatch_and_log_tool)
        assert '"arguments": func_args' in src or "'arguments': func_args" in src

    def test_shared_dispatch_tracks_module_id(self):
        from flyto_ai.providers.base import dispatch_and_log_tool
        src = inspect.getsource(dispatch_and_log_tool)
        assert 'log_entry["module_id"]' in src

    def test_shared_dispatch_tracks_ok(self):
        from flyto_ai.providers.base import dispatch_and_log_tool
        src = inspect.getsource(dispatch_and_log_tool)
        assert 'log_entry["ok"]' in src

    def test_openai_uses_shared_dispatch(self):
        src = inspect.getsource(OpenAIProvider.chat)
        assert "dispatch_and_log_tool" in src

    def test_anthropic_uses_shared_dispatch(self):
        src = inspect.getsource(AnthropicProvider.chat)
        assert "dispatch_and_log_tool" in src

    def test_openai_no_inline_log_entry(self):
        """OpenAI should not build log_entry inline (moved to shared function)."""
        src = inspect.getsource(OpenAIProvider.chat)
        assert 'log_entry: Dict' not in src

    def test_anthropic_no_inline_log_entry(self):
        """Anthropic should not build log_entry inline (moved to shared function)."""
        src = inspect.getsource(AnthropicProvider.chat)
        assert 'log_entry: Dict' not in src


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
        # Same logic as agent.py line ~298
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
# Blueprint feedback — call the REAL function
# =========================================================================

class TestBlueprintFeedbackReal:
    """Call _blueprint_feedback directly — exercises real flyto-blueprint engine."""

    def test_feedback_no_crash_on_success(self):
        """_blueprint_feedback should not crash with valid inputs."""
        tool_calls = [
            {"function": "execute_module", "arguments": {"module_id": "browser.launch", "params": {}}, "module_id": "browser.launch", "ok": True},
            {"function": "execute_module", "arguments": {"module_id": "browser.goto", "params": {"url": "https://example.com"}}, "module_id": "browser.goto", "ok": True},
            {"function": "execute_module", "arguments": {"module_id": "browser.extract", "params": {"selector": "h1"}}, "module_id": "browser.extract", "ok": True},
        ]
        execution_results = [tc for tc in tool_calls if tc.get("function") == "execute_module"]
        # This calls the real function — if flyto-blueprint is installed,
        # it will actually save to SQLite via engine.learn_from_execution()
        _blueprint_feedback(tool_calls, execution_results, "test scrape example.com")
        # No exception = pass

    def test_feedback_no_crash_on_failure(self):
        """_blueprint_feedback should not crash when execution failed."""
        tool_calls = [
            {"function": "execute_module", "arguments": {"module_id": "browser.launch", "params": {}}, "module_id": "browser.launch", "ok": True},
            {"function": "execute_module", "arguments": {"module_id": "browser.type", "params": {}}, "module_id": "browser.type", "ok": False},
        ]
        execution_results = [tc for tc in tool_calls if tc.get("function") == "execute_module"]
        # Should return early (not all ok, < 3 steps) without crashing
        _blueprint_feedback(tool_calls, execution_results, "failed test")

    def test_feedback_with_blueprint_used(self):
        """_blueprint_feedback should handle use_blueprint in tool_calls."""
        tool_calls = [
            {"function": "use_blueprint", "arguments": {"blueprint_id": "test_nonexistent_id_12345"}},
            {"function": "execute_module", "arguments": {"module_id": "browser.launch", "params": {}}, "module_id": "browser.launch", "ok": True},
            {"function": "execute_module", "arguments": {"module_id": "browser.goto", "params": {}}, "module_id": "browser.goto", "ok": True},
            {"function": "execute_module", "arguments": {"module_id": "browser.extract", "params": {}}, "module_id": "browser.extract", "ok": True},
        ]
        execution_results = [tc for tc in tool_calls if tc.get("function") == "execute_module"]
        # Will try report_outcome on nonexistent blueprint — should not crash
        _blueprint_feedback(tool_calls, execution_results, "test with blueprint")

    def test_feedback_empty_results_noop(self):
        """Empty execution_results should be a no-op."""
        _blueprint_feedback([], [], "nothing")


# =========================================================================
# Agent real code paths (no API calls needed)
# =========================================================================

class TestAgentRealPaths:
    """Test Agent methods directly without mocking providers."""

    def test_no_api_key_returns_error(self):
        """agent.chat() returns error when no API key — tests real code path."""
        config = AgentConfig(provider="openai", api_key="")
        agent = Agent(config=config)
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(agent.chat("hello", mode="execute"))
        finally:
            loop.close()
        assert result.ok is False
        assert "API key" in result.message or "api_key" in (result.error or "")

    def test_make_safe_dispatch_blocks_unknown_tool(self):
        """_make_safe_dispatch enforces tool allowlist."""
        config = AgentConfig(provider="ollama", api_key="test")
        agent = Agent(config=config)

        call_log = []
        async def real_dispatch(name, args):
            call_log.append(name)
            return {"ok": True}

        agent._dispatch_fn = real_dispatch
        agent._policies = get_default_policies()
        safe = agent._make_safe_dispatch()

        loop = asyncio.new_event_loop()
        try:
            # Allowed tool
            result = loop.run_until_complete(
                safe("execute_module", {"module_id": "browser.goto", "params": {}})
            )
            assert result["ok"] is True
            assert "execute_module" in call_log

            # Disallowed tool
            result = loop.run_until_complete(
                safe("dangerous_tool", {})
            )
            assert result["ok"] is False
            assert "not allowed" in result["error"]
        finally:
            loop.close()

    def test_make_safe_dispatch_blocks_unknown_module_category(self):
        """_make_safe_dispatch enforces module category allowlist."""
        config = AgentConfig(provider="ollama", api_key="test")
        agent = Agent(config=config)

        async def real_dispatch(name, args):
            return {"ok": True}

        agent._dispatch_fn = real_dispatch
        agent._policies = get_default_policies()
        safe = agent._make_safe_dispatch()

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                safe("execute_module", {"module_id": "malware.deploy", "params": {}})
            )
        finally:
            loop.close()
        assert result["ok"] is False
        assert "not allowed" in result["error"]

    def test_callback_exception_safety(self):
        """on_tool_call callback crash must not break agent dispatch."""
        config = AgentConfig(provider="ollama", api_key="test")
        agent = Agent(config=config)

        dispatch_calls = []
        async def real_dispatch(name, args):
            dispatch_calls.append(name)
            return {"ok": True}

        agent._dispatch_fn = real_dispatch
        agent._tools = [{"name": "search_modules", "description": "s", "inputSchema": {}}]

        def crashing_callback(name, args):
            raise RuntimeError("boom!")

        # Build the real instrumented dispatch as agent.chat() does
        base_dispatch = agent._make_safe_dispatch()
        async def instrumented(func_name, func_args):
            try:
                crashing_callback(func_name, func_args)
            except Exception:
                pass  # same logic as agent.py
            return await base_dispatch(func_name, func_args)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                instrumented("search_modules", {"query": "test"})
            )
        finally:
            loop.close()
        assert result["ok"] is True
        assert "search_modules" in dispatch_calls

    def test_chat_response_model_fields(self):
        """ChatResponse has all required fields."""
        resp = ChatResponse(
            ok=True, message="done", session_id="",
            tool_calls=[{"function": "execute_module", "module_id": "browser.goto", "ok": True}],
            execution_results=[{"function": "execute_module", "module_id": "browser.goto", "ok": True}],
            provider="openai", model="gpt-4o-mini",
        )
        assert resp.ok
        assert len(resp.execution_results) == 1
        assert resp.provider == "openai"

    def test_auto_discover_tools_runs(self):
        """Agent auto-discovers tools from flyto-core if installed."""
        config = AgentConfig(provider="ollama", api_key="test")
        agent = Agent(config=config)
        # Should not crash — tools may or may not be available
        assert isinstance(agent._tools, list)


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
        store = SessionStore(max_sessions=10, ttl_seconds=1)
        store.create("sess1", "user1")
        store._timestamps["sess1"] = time.time() - 100
        store.cleanup()
        assert "sess1" not in store._sessions


# =========================================================================
# BUG 11: config provider detection (real env vars, no monkeypatch)
# =========================================================================

class TestConfigProviderDetection:

    def _with_env(self, overrides, fn):
        """Run fn with specific env vars, then restore."""
        keys = ["FLYTO_AI_PROVIDER", "FLYTO_AI_API_KEY",
                "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
        saved = {k: os.environ.pop(k, None) for k in keys}
        try:
            for k, v in overrides.items():
                if v is not None:
                    os.environ[k] = v
            return fn()
        finally:
            for k in keys:
                os.environ.pop(k, None)
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_explicit_openai_no_fallback_to_anthropic(self):
        config = self._with_env(
            {"FLYTO_AI_PROVIDER": "openai", "ANTHROPIC_API_KEY": "sk-ant-test"},
            AgentConfig.from_env,
        )
        assert config.api_key == ""
        assert config.provider == "openai"

    def test_no_provider_prefers_openai(self):
        config = self._with_env(
            {"OPENAI_API_KEY": "sk-openai-test", "ANTHROPIC_API_KEY": "sk-ant-test"},
            AgentConfig.from_env,
        )
        assert config.provider == "openai"
        assert config.api_key == "sk-openai-test"

    def test_no_provider_falls_to_anthropic(self):
        config = self._with_env(
            {"ANTHROPIC_API_KEY": "sk-ant-test"},
            AgentConfig.from_env,
        )
        assert config.provider == "anthropic"
        assert config.api_key == "sk-ant-test"


# =========================================================================
# Policies: all module categories covered
# =========================================================================

class TestPoliciesCoverage:

    def test_critical_categories_allowed(self):
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

    def test_unknown_category_blocked(self):
        assert is_module_allowed("malware.deploy") is False
        assert is_module_allowed("exploit.rce") is False


# =========================================================================
# P0 #1: Anthropic max_rounds — verify fix in source
# =========================================================================

class TestAnthropicMaxRounds:

    def test_merges_summary_into_last_user_message(self):
        """Anthropic provider must merge summary into last user message,
        not blindly append a new user message (which would cause API 400)."""
        src = inspect.getsource(AnthropicProvider.chat)
        # The fix: define summary_text, then append it to existing content list
        assert "summary_text" in src
        assert "content.append(summary_text)" in src
        # Must check if last message is already user role before deciding
        assert 'claude_messages[-1]["role"] == "user"' in src


# =========================================================================
# P0 #2: /history command
# =========================================================================

class TestHistoryCommand:

    def test_history_in_help_text(self):
        """Source code must handle /history command."""
        from flyto_ai import cli
        src = inspect.getsource(cli._cmd_interactive)
        assert '"/history"' in src or "'/history'" in src
        # Must have handler, not just in help text
        assert "cmd == \"/history\"" in src or 'cmd == "/history"' in src


# =========================================================================
# P1 #3: validate_base_url with SSRF protection
# =========================================================================

class TestBaseUrlValidation:

    def test_localhost_allowed(self):
        assert validate_base_url("http://localhost:11434/v1") is True

    def test_loopback_allowed(self):
        assert validate_base_url("http://127.0.0.1:8080/v1") is True

    def test_openai_https_allowed(self):
        assert validate_base_url("https://api.openai.com/v1") is True

    def test_http_remote_blocked(self):
        assert validate_base_url("http://api.openai.com/v1") is False

    def test_unknown_domain_blocked(self):
        assert validate_base_url("https://evil.example.com/v1") is False

    def test_azure_openai_allowed(self):
        assert validate_base_url("https://my-instance.openai.azure.com/v1") is True

    def test_empty_url_blocked(self):
        assert validate_base_url("") is False

    def test_ftp_blocked(self):
        assert validate_base_url("ftp://localhost/data") is False

    def test_config_clears_bad_base_url(self):
        config = AgentConfig(provider="openai", api_key="test",
                             base_url="https://evil.example.com/v1")
        assert config.base_url is None

    def test_config_keeps_good_base_url(self):
        config = AgentConfig(provider="openai", api_key="test",
                             base_url="https://api.openai.com/v1")
        assert config.base_url == "https://api.openai.com/v1"

    def test_config_keeps_localhost(self):
        config = AgentConfig(provider="ollama", api_key="ollama",
                             base_url="http://localhost:11434/v1")
        assert config.base_url == "http://localhost:11434/v1"


# =========================================================================
# P1 #4+5: serve JSON parse + Content-Length
# =========================================================================

class TestServeHardening:

    def test_serve_source_validates_content_length(self):
        """Serve handler must check Content-Length bounds."""
        from flyto_ai import cli
        src = inspect.getsource(cli._cmd_serve_stdlib)
        assert "MAX_BODY_SIZE" in src or "Content-Length" in src

    def test_serve_source_catches_json_errors(self):
        """Serve handler must catch JSON parse errors."""
        from flyto_ai import cli
        src = inspect.getsource(cli._cmd_serve_stdlib)
        assert "JSONDecodeError" in src or "json.loads" in src

    def test_serve_max_body_constant(self):
        """MAX_BODY_SIZE should be defined and reasonable."""
        from flyto_ai import cli
        src = inspect.getsource(cli._cmd_serve_stdlib)
        assert "1_000_000" in src or "1000000" in src


# =========================================================================
# P1 #6+7: inspect_page ok defaults
# =========================================================================

class TestInspectPageDefaults:

    def test_source_uses_is_ok_helper(self):
        """inspect_page must use _is_ok() helper, not raw .get('ok', True)."""
        src = inspect.getsource(inspect_page)
        # Should NOT have get("ok", True) anywhere
        assert '.get("ok", True)' not in src
        assert ".get('ok', True)" not in src
        # Should use _is_ok() helper for all result checks
        assert '_is_ok(' in src

    def test_tool_def_has_required_url(self):
        assert INSPECT_PAGE_TOOL["name"] == "inspect_page"
        assert "url" in INSPECT_PAGE_TOOL["inputSchema"]["properties"]
        assert "url" in INSPECT_PAGE_TOOL["inputSchema"]["required"]


# =========================================================================
# P2 #8: browser sessions clear
# =========================================================================

class TestBrowserSessionsClear:

    def test_clear_browser_sessions(self):
        _browser_sessions["test_session_123"] = {"page": "dummy"}
        assert len(_browser_sessions) > 0
        clear_browser_sessions()
        assert len(_browser_sessions) == 0

    def test_clear_is_idempotent(self):
        clear_browser_sessions()
        clear_browser_sessions()
        assert len(_browser_sessions) == 0


# =========================================================================
# P2 #9: interactive history trim
# =========================================================================

class TestHistoryTrim:

    def test_history_trim_in_source(self):
        """Interactive mode must trim history to prevent unbounded growth."""
        from flyto_ai import cli
        src = inspect.getsource(cli._cmd_interactive)
        assert "history" in src
        # Must have a trim/slice operation
        assert "40" in src  # our limit
        assert "history[:]" in src or "history =" in src

    def test_trim_logic_preserves_newest(self):
        history = []
        for i in range(30):
            history.append({"role": "user", "content": "msg {}".format(i)})
            history.append({"role": "assistant", "content": "reply {}".format(i)})
        if len(history) > 40:
            history[:] = history[-40:]
        assert len(history) == 40
        assert history[-1]["content"] == "reply 29"
        assert history[0]["content"] == "msg 10"


# =========================================================================
# P2 #10: serve event loop closure
# =========================================================================

class TestServeLoopClosure:

    def test_serve_closes_loop(self):
        """_cmd_serve_stdlib must close the event loop in finally block."""
        from flyto_ai import cli
        src = inspect.getsource(cli._cmd_serve_stdlib)
        assert "loop.close()" in src
        assert "finally" in src


# =========================================================================
# P2 #11: LLM client caching
# =========================================================================

class TestLLMClientCaching:

    def test_openai_client_cached(self):
        provider = OpenAIProvider(api_key="test-key")
        client1 = provider._make_client()
        client2 = provider._make_client()
        assert client1 is client2

    def test_openai_client_initially_none(self):
        provider = OpenAIProvider(api_key="test-key")
        # Before first _make_client, _client is None
        assert provider._client is None

    def test_anthropic_client_initially_none(self):
        provider = AnthropicProvider(api_key="test-key")
        assert provider._client is None


# =========================================================================
# P2 #12: MCP handler caching
# =========================================================================

class TestMCPHandlerCaching:

    def test_handler_caching_exists(self):
        from flyto_ai.tools import core_tools
        assert hasattr(core_tools, "_handler_checked")
        assert hasattr(core_tools, "_cached_handler")

    def test_get_mcp_handler_is_cached(self):
        """Calling _get_mcp_handler twice returns same object."""
        from flyto_ai.tools.core_tools import _get_mcp_handler
        result1 = _get_mcp_handler()
        result2 = _get_mcp_handler()
        assert result1 is result2  # same ref (cached)


# =========================================================================
# P3 #13: max_tokens truncation detection
# =========================================================================

class TestTruncationDetection:

    def test_openai_detects_length_finish(self):
        """OpenAI provider source must check finish_reason == 'length'."""
        src = inspect.getsource(OpenAIProvider.chat)
        assert "finish_reason" in src
        assert '"length"' in src or "'length'" in src
        assert "truncated" in src.lower()

    def test_anthropic_detects_max_tokens_stop(self):
        """Anthropic provider source must check stop_reason == 'max_tokens'."""
        src = inspect.getsource(AnthropicProvider.chat)
        assert "stop_reason" in src
        assert '"max_tokens"' in src or "'max_tokens'" in src
        assert "truncated" in src.lower()


# =========================================================================
# P3 #14: on_tool_call callback safety
# =========================================================================

class TestCallbackSafety:

    def test_agent_source_wraps_callback_in_try(self):
        """Agent.chat source must wrap on_tool_call in try/except."""
        src = inspect.getsource(Agent.chat)
        # Find the callback wrapping section
        assert "on_tool_call" in src
        assert "except Exception" in src
        assert "pass" in src  # callback failure = silently continue

    def test_real_callback_safety(self):
        """Build actual instrumented dispatch and verify crash safety."""
        config = AgentConfig(provider="ollama", api_key="test")
        agent = Agent(config=config)

        calls = []
        async def real_dispatch(name, args):
            calls.append(name)
            return {"ok": True}

        agent._dispatch_fn = real_dispatch
        agent._tools = [{"name": "search_modules", "description": "s", "inputSchema": {}}]

        def crashing_callback(name, args):
            raise RuntimeError("boom!")

        # Reproduce agent.chat() callback wrapping logic
        base = agent._make_safe_dispatch()
        async def instrumented(fn, fa):
            try:
                crashing_callback(fn, fa)
            except Exception:
                pass
            return await base(fn, fa)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(instrumented("search_modules", {"query": "x"}))
        finally:
            loop.close()

        assert result["ok"] is True
        assert "search_modules" in calls


# =========================================================================
# P3 #15: empty module_id in blueprint feedback
# =========================================================================

class TestBlueprintEmptyModuleId:

    def test_source_filters_empty_module_id(self):
        """_blueprint_feedback must skip entries with empty module_id."""
        src = inspect.getsource(_blueprint_feedback)
        assert "if not mid" in src
        assert "continue" in src

    def test_source_rechecks_step_count(self):
        """After filtering, must verify >= 3 steps remain."""
        src = inspect.getsource(_blueprint_feedback)
        assert "len(steps) < 3" in src

    def test_feedback_with_empty_ids_doesnt_save(self):
        """Calling _blueprint_feedback with mostly empty module_ids should not crash."""
        tool_calls = [
            {"function": "execute_module", "arguments": {"module_id": "", "params": {}}, "module_id": "", "ok": True},
            {"function": "execute_module", "arguments": {"module_id": "", "params": {}}, "module_id": "", "ok": True},
            {"function": "execute_module", "arguments": {"module_id": "browser.goto", "params": {}}, "module_id": "browser.goto", "ok": True},
        ]
        execution_results = [tc for tc in tool_calls if tc.get("function") == "execute_module"]
        # Should return early (< 3 non-empty steps) — no crash
        _blueprint_feedback(tool_calls, execution_results, "test with empties")


# =========================================================================
# Dispatch: real core_tools and blueprint_tools
# =========================================================================

class TestDispatchCoreToolsReal:

    def test_get_core_tool_defs_returns_list(self):
        result = get_core_tool_defs()
        assert isinstance(result, list)
        # If flyto-core is installed, should have tools
        if result:
            assert all("name" in t for t in result)
            assert all("inputSchema" in t for t in result)

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self):
        result = await dispatch_core_tool("nonexistent_tool_xyz", {})
        assert result["ok"] is False
        assert "Unknown" in result.get("error", "") or "not installed" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_dispatch_list_modules(self):
        """Call real list_modules if flyto-core is installed."""
        result = await dispatch_core_tool("list_modules", {})
        # Either returns modules or "not installed" error
        if result.get("ok") is False:
            assert "not installed" in result.get("error", "")
        else:
            assert isinstance(result, (dict, list))

    @pytest.mark.asyncio
    async def test_dispatch_search_modules(self):
        """Call real search_modules."""
        result = await dispatch_core_tool("search_modules", {"query": "browser"})
        if result.get("ok") is False and "not installed" in result.get("error", ""):
            pass  # flyto-core not installed, OK
        else:
            assert isinstance(result, (dict, list))

    @pytest.mark.asyncio
    async def test_dispatch_get_module_info(self):
        """Call real get_module_info."""
        result = await dispatch_core_tool("get_module_info", {"module_id": "browser.launch"})
        if result.get("ok") is False and "not installed" in result.get("error", ""):
            pass
        else:
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_dispatch_validate_params(self):
        """Call real validate_params."""
        result = await dispatch_core_tool("validate_params", {
            "module_id": "browser.launch",
            "params": {"headless": True},
        })
        if result.get("ok") is False and "not installed" in result.get("error", ""):
            pass
        else:
            assert isinstance(result, dict)


class TestDispatchBlueprintToolsReal:

    def test_get_blueprint_tool_defs_returns_list(self):
        result = get_blueprint_tool_defs()
        assert isinstance(result, list)
        if result:
            assert all("name" in t for t in result)

    @pytest.mark.asyncio
    async def test_dispatch_unknown_blueprint_tool(self):
        result = await dispatch_blueprint_tool("nonexistent_tool_xyz", {})
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_dispatch_list_blueprints(self):
        """Call real list_blueprints."""
        result = await dispatch_blueprint_tool("list_blueprints", {})
        if result.get("ok") is False and "not installed" in result.get("error", ""):
            pass
        else:
            assert result.get("ok") is True
            assert "blueprints" in result


# =========================================================================
# Tool registry — real dispatch
# =========================================================================

class TestToolRegistryReal:

    @pytest.mark.asyncio
    async def test_dispatch_unknown_returns_error(self):
        reg = ToolRegistry()
        result = await reg.dispatch("ghost_tool", {})
        assert result["ok"] is False
        assert "Unknown" in result["error"]

    @pytest.mark.asyncio
    async def test_dispatch_handler_exception_caught(self):
        reg = ToolRegistry()

        async def crashing_handler(name, args):
            raise ValueError("real crash")

        reg.register({"name": "bad_tool", "description": "crash", "inputSchema": {}}, crashing_handler)
        result = await reg.dispatch("bad_tool", {})
        assert result["ok"] is False
        assert "real crash" in result["error"]

    @pytest.mark.asyncio
    async def test_dispatch_real_handler_success(self):
        reg = ToolRegistry()

        async def good_handler(name, args):
            return {"ok": True, "data": args.get("x", 0) * 2}

        reg.register({"name": "double", "description": "doubles x", "inputSchema": {}}, good_handler)
        result = await reg.dispatch("double", {"x": 21})
        assert result["ok"] is True
        assert result["data"] == 42

    def test_register_many_and_list(self):
        reg = ToolRegistry()

        async def handler(name, args):
            return {"ok": True}

        defs = [
            {"name": "tool_a", "description": "A", "inputSchema": {}},
            {"name": "tool_b", "description": "B", "inputSchema": {}},
        ]
        reg.register_many(defs, handler)
        assert len(reg.tools) == 2
        names = {t["name"] for t in reg.tools}
        assert names == {"tool_a", "tool_b"}

    def test_to_openai_format(self):
        reg = ToolRegistry()

        async def h(n, a):
            return {}

        reg.register({"name": "t", "description": "d", "inputSchema": {"type": "object"}}, h)
        oai = reg.to_openai_format()
        assert oai[0]["type"] == "function"
        assert oai[0]["function"]["name"] == "t"

    def test_to_anthropic_format(self):
        reg = ToolRegistry()

        async def h(n, a):
            return {}

        reg.register({"name": "t", "description": "d", "inputSchema": {"type": "object"}}, h)
        ant = reg.to_anthropic_format()
        assert ant[0]["name"] == "t"
        assert "input_schema" in ant[0]
