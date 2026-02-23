# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for Sprint 1 (Code Hardening) + Sprint 2 (Architecture Quick Wins)."""
import asyncio
import inspect

import pytest

from flyto_ai.config import AgentConfig
from flyto_ai.providers.openai import OpenAIProvider
from flyto_ai.providers.anthropic import AnthropicProvider
from flyto_ai.providers.base import fire_stream
from flyto_ai.models import StreamEvent, StreamEventType
from flyto_ai.tools.registry import ToolRegistry, MAX_DISPATCH_DEPTH
from flyto_ai.validation import validate_workflow_steps


# =========================================================================
# Sprint 1 #1: Tool discovery logs warnings (not silent pass)
# =========================================================================

class TestToolDiscoveryWarning:

    def test_auto_discover_uses_logger_warning(self):
        """except blocks in _auto_discover_tools must log, not pass silently."""
        from flyto_ai.agent import Agent
        src = inspect.getsource(Agent._auto_discover_tools)
        # Should NOT have bare 'except Exception:\n            pass'
        assert "except Exception:\n            pass" not in src
        # Should use logger.warning
        assert "logger.warning" in src


# =========================================================================
# Sprint 1 #2: _fire() uses debug log (not silent pass)
# =========================================================================

class TestFireStreamLog:

    def test_fire_stream_in_base(self):
        """fire_stream should be importable from providers.base."""
        assert callable(fire_stream)

    def test_fire_stream_none_callback_noop(self):
        """fire_stream with None callback does nothing."""
        event = StreamEvent(type=StreamEventType.TOKEN, content="hi")
        fire_stream(None, event)  # no exception

    def test_fire_stream_catches_callback_error(self):
        """fire_stream should catch callback exceptions."""
        def bad_callback(event):
            raise RuntimeError("boom")

        event = StreamEvent(type=StreamEventType.TOKEN, content="hi")
        fire_stream(bad_callback, event)  # should not raise

    def test_openai_imports_fire_stream(self):
        """OpenAI provider should import fire_stream from base."""
        import flyto_ai.providers.openai as oai_mod
        src = inspect.getsource(oai_mod)
        assert "fire_stream" in src

    def test_anthropic_imports_fire_stream(self):
        """Anthropic provider should import fire_stream from base."""
        import flyto_ai.providers.anthropic as ant_mod
        src = inspect.getsource(ant_mod)
        assert "fire_stream" in src


# =========================================================================
# Sprint 1 #3: Browser relaunch error message includes both errors
# =========================================================================

class TestBrowserRelaunchErrorMsg:

    def test_relaunch_error_combines_messages(self):
        """Relaunch failure must include both original + relaunch error."""
        from flyto_ai.tools import core_tools
        src = inspect.getsource(core_tools.dispatch_core_tool)
        assert "Browser session dead" in src
        assert "Relaunch also failed" in src


# =========================================================================
# Sprint 1 #4: Webhook catches ValueError + OSError
# =========================================================================

class TestWebhookExceptionHandling:

    def test_webhook_catches_valueerror(self):
        from flyto_ai.cli import _post_webhook
        src = inspect.getsource(_post_webhook)
        assert "ValueError" in src

    def test_webhook_catches_oserror(self):
        from flyto_ai.cli import _post_webhook
        src = inspect.getsource(_post_webhook)
        assert "OSError" in src


# =========================================================================
# Sprint 1 #5: YAML validation reports structural errors
# =========================================================================

class TestYamlValidationStructure:

    def test_non_dict_returns_error(self):
        errors = validate_workflow_steps("- item1\n- item2")
        assert len(errors) == 1
        assert "mapping" in errors[0].lower() or "dict" in errors[0].lower()

    def test_string_returns_error(self):
        errors = validate_workflow_steps('"just a string"')
        assert len(errors) == 1
        assert "dict" in errors[0].lower() or "mapping" in errors[0].lower()

    def test_missing_steps_key(self):
        errors = validate_workflow_steps("name: test\ndescription: hello")
        assert len(errors) == 1
        assert "steps" in errors[0].lower()

    def test_steps_not_list(self):
        errors = validate_workflow_steps("name: test\nsteps: not_a_list")
        assert len(errors) == 1
        assert "list" in errors[0].lower()

    def test_valid_empty_steps(self):
        errors = validate_workflow_steps("name: test\nsteps: []")
        assert len(errors) == 0


# =========================================================================
# Sprint 1 #6: Pipe input support
# =========================================================================

class TestPipeMode:

    def test_pipe_handler_exists(self):
        from flyto_ai import cli
        assert hasattr(cli, "_cmd_pipe")

    def test_main_routes_pipe(self):
        from flyto_ai.cli import main
        src = inspect.getsource(main)
        assert "_cmd_pipe" in src
        assert "sys.stdin.isatty()" in src


# =========================================================================
# Sprint 2 #7: Provider registry pattern
# =========================================================================

class TestProviderRegistry:

    def test_registry_has_openai(self):
        from flyto_ai.providers import PROVIDER_REGISTRY
        assert "openai" in PROVIDER_REGISTRY

    def test_registry_has_anthropic(self):
        from flyto_ai.providers import PROVIDER_REGISTRY
        assert "anthropic" in PROVIDER_REGISTRY

    def test_registry_has_ollama(self):
        from flyto_ai.providers import PROVIDER_REGISTRY
        assert "ollama" in PROVIDER_REGISTRY

    def test_create_provider_openai(self):
        from flyto_ai.providers import create_provider
        provider = create_provider("openai", api_key="test", model="gpt-4o-mini")
        assert isinstance(provider, OpenAIProvider)

    def test_create_provider_anthropic(self):
        from flyto_ai.providers import create_provider
        provider = create_provider("anthropic", api_key="test")
        assert isinstance(provider, AnthropicProvider)

    def test_create_provider_unknown_defaults_openai(self):
        from flyto_ai.providers import create_provider
        provider = create_provider("unknown_provider", api_key="test")
        assert isinstance(provider, OpenAIProvider)

    def test_agent_uses_registry(self):
        from flyto_ai.agent import Agent
        src = inspect.getsource(Agent._make_provider)
        assert "create_provider" in src
        assert "if cfg.provider == \"anthropic\"" not in src


# =========================================================================
# Sprint 2 #8: _fire() extracted to base.py
# =========================================================================

class TestFireStreamShared:

    def test_no_local_fire_in_openai(self):
        """openai.py should not define its own _fire function."""
        import flyto_ai.providers.openai as oai_mod
        src = inspect.getsource(oai_mod)
        # Should import, not define
        assert "def _fire(" not in src
        assert "fire_stream as _fire" in src

    def test_no_local_fire_in_anthropic(self):
        """anthropic.py should not define its own _fire function."""
        import flyto_ai.providers.anthropic as ant_mod
        src = inspect.getsource(ant_mod)
        assert "def _fire(" not in src
        assert "fire_stream as _fire" in src


# =========================================================================
# Sprint 2 #9: Thread-safe browser sessions
# =========================================================================

class TestThreadSafeBrowserSessions:

    def test_lock_exists(self):
        from flyto_ai.tools.core_tools import _browser_sessions_lock
        assert hasattr(_browser_sessions_lock, "acquire")
        assert hasattr(_browser_sessions_lock, "release")

    def test_clear_uses_lock(self):
        from flyto_ai.tools import core_tools
        src = inspect.getsource(core_tools.clear_browser_sessions)
        assert "_browser_sessions_lock" in src


# =========================================================================
# Sprint 2 #10: Config bounds validation
# =========================================================================

class TestConfigBoundsValidation:

    def test_temperature_negative_clamped(self):
        cfg = AgentConfig(temperature=-1.0)
        assert cfg.temperature == 0.0

    def test_temperature_over_two_clamped(self):
        cfg = AgentConfig(temperature=5.0)
        assert cfg.temperature == 2.0

    def test_temperature_valid_unchanged(self):
        cfg = AgentConfig(temperature=1.2)
        assert cfg.temperature == 1.2

    def test_max_tokens_zero_clamped(self):
        cfg = AgentConfig(max_tokens=0)
        assert cfg.max_tokens == 1

    def test_max_tokens_negative_clamped(self):
        cfg = AgentConfig(max_tokens=-100)
        assert cfg.max_tokens == 1

    def test_max_tokens_huge_clamped(self):
        cfg = AgentConfig(max_tokens=999_999)
        assert cfg.max_tokens == 200_000

    def test_max_tokens_valid_unchanged(self):
        cfg = AgentConfig(max_tokens=8192)
        assert cfg.max_tokens == 8192


# =========================================================================
# Sprint 2 #11: Dispatch recursion depth limit
# =========================================================================

class TestDispatchRecursionLimit:

    @pytest.mark.asyncio
    async def test_normal_dispatch_works(self):
        reg = ToolRegistry()

        async def handler(name, args):
            return {"ok": True}

        reg.register({"name": "test", "description": "t", "inputSchema": {}}, handler)
        result = await reg.dispatch("test", {})
        assert result["ok"] is True

    @pytest.mark.asyncio
    async def test_deep_recursion_blocked(self):
        """dispatch() must block recursion beyond MAX_DISPATCH_DEPTH."""
        reg = ToolRegistry()
        call_count = [0]

        async def recursive_handler(name, args):
            call_count[0] += 1
            return await reg.dispatch("recurse", {})

        reg.register({"name": "recurse", "description": "r", "inputSchema": {}}, recursive_handler)
        result = await reg.dispatch("recurse", {})
        assert result["ok"] is False
        assert "recursion" in result["error"].lower() or "limit" in result["error"].lower()
        assert call_count[0] <= MAX_DISPATCH_DEPTH

    def test_max_dispatch_depth_constant(self):
        assert MAX_DISPATCH_DEPTH == 10


# =========================================================================
# Sprint 2 #12: API key __repr__ redaction
# =========================================================================

class TestApiKeyRedaction:

    def test_openai_repr_hides_key(self):
        provider = OpenAIProvider(api_key="sk-1234567890abcdef")
        r = repr(provider)
        assert "sk-1234567890abcdef" not in r
        assert "sk-1..." in r

    def test_anthropic_repr_hides_key(self):
        provider = AnthropicProvider(api_key="sk-ant-secret-key-here")
        r = repr(provider)
        assert "sk-ant-secret-key-here" not in r
        assert "sk-a..." in r

    def test_openai_repr_short_key(self):
        provider = OpenAIProvider(api_key="abc")
        r = repr(provider)
        assert "abc" not in r
        assert "***" in r

    def test_openai_repr_empty_key(self):
        provider = OpenAIProvider(api_key="")
        r = repr(provider)
        assert "***" in r


# =========================================================================
# Sprint 2 #13: HTTP server schema validation
# =========================================================================

class TestServeSchemaValidation:

    def test_serve_uses_chat_request_validation(self):
        from flyto_ai import cli
        src = inspect.getsource(cli._cmd_serve)
        assert "ChatRequest" in src
        assert "model_validate" in src


# =========================================================================
# Sprint 2 #14: Session cleanup in serve
# =========================================================================

class TestServeSessionCleanup:

    def test_serve_has_cleanup(self):
        from flyto_ai import cli
        src = inspect.getsource(cli._cmd_serve)
        assert "clear_browser_sessions" in src
        assert "CLEANUP_INTERVAL" in src


# =========================================================================
# Sprint 2 #15: inspect_page cleanup
# =========================================================================

class TestInspectPageCleanup:

    def test_finally_clears_sessions(self):
        from flyto_ai.tools.inspect_page import inspect_page
        src = inspect.getsource(inspect_page)
        assert "sessions.clear()" in src

    def test_finally_logs_cleanup_failure(self):
        from flyto_ai.tools.inspect_page import inspect_page
        src = inspect.getsource(inspect_page)
        assert "logger.debug" in src
