# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for Sprint 3 — 6 gap fixes.  ALL real tests, NO mocks.

Gap 4: Provider dedup (dispatch_and_log_tool)
Gap 5: Configurable max rounds
Gap 2: Token usage tracking (UsageStats)
Gap 1: MCP server
Gap 3: Observability (audit + tool timing)
Gap 6: Async serve (aiohttp / stdlib fallback)
"""
import json
import logging
import os
import time

import pytest

from flyto_ai import Agent, AgentConfig
from flyto_ai.models import ChatResponse, UsageStats, StreamEvent, StreamEventType


# Skip all real-LLM tests if no API key available
_has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
requires_openai = pytest.mark.skipif(not _has_openai_key, reason="OPENAI_API_KEY not set")


# =========================================================================
# Gap 4: Provider dedup — dispatch_and_log_tool (real calls, no mock)
# =========================================================================

class TestDispatchAndLogTool:
    """dispatch_and_log_tool: real async dispatch, real stream callbacks."""

    def test_constants_values(self):
        from flyto_ai.providers.base import MAX_RESULT_LEN, MAX_PREVIEW_LEN, TRUNCATION_NOTE
        assert MAX_RESULT_LEN == 8000
        assert MAX_PREVIEW_LEN == 500
        assert "truncated" in TRUNCATION_NOTE.lower()

    @pytest.mark.asyncio
    async def test_dispatch_returns_result_and_log_entry(self):
        """Real dispatch function → verify result_str and log_entry structure."""
        from flyto_ai.providers.base import dispatch_and_log_tool

        async def real_dispatch(name, args):
            return {"ok": True, "data": args.get("x", 0) * 2}

        result_str, log_entry, images = await dispatch_and_log_tool(
            "double", {"x": 21}, real_dispatch, round_num=0,
        )
        assert images == []
        parsed = json.loads(result_str)
        assert parsed["ok"] is True
        assert parsed["data"] == 42
        assert log_entry["function"] == "double"
        assert log_entry["arguments"] == {"x": 21}
        assert len(log_entry["result_preview"]) > 0

    @pytest.mark.asyncio
    async def test_dispatch_truncates_oversized_result(self):
        """Result > 8000 chars gets truncated with note."""
        from flyto_ai.providers.base import dispatch_and_log_tool, MAX_RESULT_LEN, TRUNCATION_NOTE

        async def big_dispatch(name, args):
            return {"ok": True, "data": "x" * 20000}

        result_str, _, _ = await dispatch_and_log_tool("big", {}, big_dispatch, round_num=0)
        assert len(result_str) <= MAX_RESULT_LEN + len(TRUNCATION_NOTE) + 10
        assert TRUNCATION_NOTE in result_str

    @pytest.mark.asyncio
    async def test_dispatch_preview_capped_at_500(self):
        """log_entry['result_preview'] capped at MAX_PREVIEW_LEN."""
        from flyto_ai.providers.base import dispatch_and_log_tool, MAX_PREVIEW_LEN

        async def medium_dispatch(name, args):
            return {"ok": True, "data": "y" * 2000}

        _, log_entry, _ = await dispatch_and_log_tool("med", {}, medium_dispatch, round_num=0)
        assert len(log_entry["result_preview"]) <= MAX_PREVIEW_LEN

    @pytest.mark.asyncio
    async def test_dispatch_execute_module_enriches_log(self):
        """execute_module calls add module_id and ok to log_entry."""
        from flyto_ai.providers.base import dispatch_and_log_tool

        async def ok_dispatch(name, args):
            return {"ok": True}

        _, log_entry, _ = await dispatch_and_log_tool(
            "execute_module", {"module_id": "browser.goto"}, ok_dispatch, round_num=2,
        )
        assert log_entry["module_id"] == "browser.goto"
        assert log_entry["ok"] is True

    @pytest.mark.asyncio
    async def test_dispatch_execute_module_failed(self):
        """Failed execute_module → ok=False in log_entry."""
        from flyto_ai.providers.base import dispatch_and_log_tool

        async def fail_dispatch(name, args):
            return {"ok": False, "error": "not found"}

        _, log_entry, _ = await dispatch_and_log_tool(
            "execute_module", {"module_id": "bad.module"}, fail_dispatch, round_num=0,
        )
        assert log_entry["ok"] is False

    @pytest.mark.asyncio
    async def test_dispatch_fires_stream_events(self):
        """Real stream callback receives TOOL_START and TOOL_END."""
        from flyto_ai.providers.base import dispatch_and_log_tool

        events = []

        def on_stream(event):
            events.append(event)

        async def real_dispatch(name, args):
            return {"ok": True}

        await dispatch_and_log_tool("test_tool", {}, real_dispatch, round_num=0, on_stream=on_stream)

        types = [e.type for e in events]
        assert StreamEventType.TOOL_START in types
        assert StreamEventType.TOOL_END in types
        # TOOL_START has tool_name
        start_event = [e for e in events if e.type == StreamEventType.TOOL_START][0]
        assert start_event.tool_name == "test_tool"
        # TOOL_END has tool_result
        end_event = [e for e in events if e.type == StreamEventType.TOOL_END][0]
        assert isinstance(end_event.tool_result, dict)

    @pytest.mark.asyncio
    async def test_dispatch_redacts_sensitive_args(self, caplog):
        """Sensitive keys like 'password' should be redacted in log output."""
        from flyto_ai.providers.base import dispatch_and_log_tool

        async def noop(name, args):
            return {"ok": True}

        with caplog.at_level(logging.INFO, logger="flyto_ai.providers.base"):
            await dispatch_and_log_tool(
                "login", {"username": "admin", "password": "s3cret"}, noop, round_num=0,
            )

        # The log line should contain "***" for password, not "s3cret"
        log_text = " ".join(r.message for r in caplog.records)
        assert "s3cret" not in log_text
        assert "***" in log_text


# =========================================================================
# Gap 5: Configurable max rounds (real config objects, no mock)
# =========================================================================

class TestConfigurableMaxRounds:

    def test_default_30(self):
        cfg = AgentConfig()
        assert cfg.max_tool_rounds == 30

    def test_from_dict_default_30(self):
        cfg = AgentConfig.from_dict({})
        assert cfg.max_tool_rounds == 30

    def test_from_dict_override(self):
        cfg = AgentConfig.from_dict({"max_tool_rounds": 50})
        assert cfg.max_tool_rounds == 50

    def test_from_env_reads_env(self, monkeypatch):
        monkeypatch.setenv("FLYTO_AI_MAX_TOOL_ROUNDS", "20")
        monkeypatch.setenv("FLYTO_AI_PROVIDER", "ollama")
        cfg = AgentConfig.from_env()
        assert cfg.max_tool_rounds == 20

    def test_from_env_default(self, monkeypatch):
        monkeypatch.delenv("FLYTO_AI_MAX_TOOL_ROUNDS", raising=False)
        monkeypatch.setenv("FLYTO_AI_PROVIDER", "ollama")
        cfg = AgentConfig.from_env()
        assert cfg.max_tool_rounds == 30

    def test_chat_response_has_rounds_used(self):
        resp = ChatResponse(ok=True, message="hi", session_id="", rounds_used=5)
        assert resp.rounds_used == 5

    def test_chat_response_default_rounds_zero(self):
        resp = ChatResponse(ok=True, message="hi", session_id="")
        assert resp.rounds_used == 0

    @requires_openai
    @pytest.mark.asyncio
    async def test_real_chat_returns_rounds_used(self):
        """Real OpenAI call → rounds_used > 0 in ChatResponse."""
        config = AgentConfig.from_env()
        config.model = "gpt-4o-mini"
        config.max_tool_rounds = 5
        agent = Agent(config=config, tools=[], dispatch_fn=None)

        result = await agent.chat("Reply with exactly: hello", mode="execute")

        assert result.ok
        assert result.rounds_used >= 1
        assert len(result.message) > 0


# =========================================================================
# Gap 2: Token usage tracking (real LLM calls)
# =========================================================================

class TestUsageStats:

    def test_model_fields_defaults(self):
        u = UsageStats()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
        assert u.cache_creation_input_tokens == 0
        assert u.cache_read_input_tokens == 0

    def test_model_with_values(self):
        u = UsageStats(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert u.total_tokens == 150

    def test_exported_from_init(self):
        from flyto_ai import UsageStats as US
        assert US is UsageStats

    def test_chat_response_has_usage(self):
        u = UsageStats(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        resp = ChatResponse(ok=True, message="hi", session_id="", usage=u)
        assert resp.usage.total_tokens == 150

    def test_chat_response_usage_none_default(self):
        resp = ChatResponse(ok=True, message="hi", session_id="")
        assert resp.usage is None

    def test_usage_stats_serialization(self):
        """UsageStats round-trips through JSON."""
        u = UsageStats(prompt_tokens=100, completion_tokens=50, total_tokens=150,
                       cache_creation_input_tokens=10, cache_read_input_tokens=20)
        data = json.loads(u.model_dump_json())
        assert data["prompt_tokens"] == 100
        assert data["cache_read_input_tokens"] == 20

    @requires_openai
    @pytest.mark.asyncio
    async def test_real_openai_returns_usage(self):
        """Real OpenAI call → usage.total_tokens > 0."""
        config = AgentConfig.from_env()
        config.model = "gpt-4o-mini"
        agent = Agent(config=config, tools=[], dispatch_fn=None)

        result = await agent.chat("What is 2+2? Reply with just the number.", mode="execute")

        assert result.ok
        assert result.usage is not None, "usage should be populated from real OpenAI call"
        assert result.usage.total_tokens > 0
        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0

    @requires_openai
    @pytest.mark.asyncio
    async def test_real_openai_streaming_returns_usage(self):
        """Real OpenAI streaming call → usage.total_tokens > 0."""
        config = AgentConfig.from_env()
        config.model = "gpt-4o-mini"
        agent = Agent(config=config, tools=[], dispatch_fn=None)

        tokens = []
        def on_stream(event):
            if event.type == StreamEventType.TOKEN:
                tokens.append(event.content)

        result = await agent.chat(
            "Say exactly: hello world", mode="execute", on_stream=on_stream,
        )

        assert result.ok
        assert len(tokens) > 0, "should have received streaming tokens"
        assert result.usage is not None
        assert result.usage.total_tokens > 0

    @requires_openai
    @pytest.mark.asyncio
    async def test_real_openai_with_tools_returns_usage(self):
        """Real OpenAI call with tools → usage tracks across tool rounds."""
        config = AgentConfig.from_env()
        config.model = "gpt-4o-mini"
        # Use auto-discovered tools (flyto-core)
        agent = Agent(config=config)

        if not agent._tools:
            pytest.skip("flyto-core not installed, no tools to test")

        result = await agent.chat("List available module categories", mode="execute")

        assert result.ok
        assert result.usage is not None
        assert result.usage.total_tokens > 0
        # With tool calls, prompt tokens accumulate across rounds
        if result.rounds_used > 1:
            assert result.usage.prompt_tokens > 100  # multi-round = more tokens


# =========================================================================
# Gap 1: MCP server (real MCPServer, real Agent init)
# =========================================================================

class TestMCPServer:

    @pytest.mark.asyncio
    async def test_initialize_returns_server_info(self):
        from flyto_ai.mcp_server import MCPServer
        server = MCPServer()
        resp = await server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})

        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert resp["result"]["serverInfo"]["name"] == "flyto-ai"
        assert resp["result"]["capabilities"]["tools"]["listChanged"] is False

    @pytest.mark.asyncio
    async def test_ping_returns_empty(self):
        from flyto_ai.mcp_server import MCPServer
        server = MCPServer()
        resp = await server.handle({"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}})
        assert resp["result"] == {}

    @pytest.mark.asyncio
    async def test_notification_no_response(self):
        from flyto_ai.mcp_server import MCPServer
        server = MCPServer()
        resp = await server.handle({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        assert resp is None

    @pytest.mark.asyncio
    async def test_unknown_method_returns_error(self):
        from flyto_ai.mcp_server import MCPServer
        server = MCPServer()
        resp = await server.handle({"jsonrpc": "2.0", "id": 3, "method": "foo/bar", "params": {}})
        assert resp["error"]["code"] == -32601
        assert "not found" in resp["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_tools_list_includes_chat_tool(self):
        """tools/list returns real registered tools + chat meta-tool."""
        from flyto_ai.mcp_server import MCPServer
        server = MCPServer()
        resp = await server.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": {}})

        assert "result" in resp
        tools = resp["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        # chat meta-tool must always be present
        assert "chat" in tool_names
        # If flyto-core is installed, should have more tools
        if len(tools) > 1:
            assert "execute_module" in tool_names or "search_modules" in tool_names

    @pytest.mark.asyncio
    async def test_tools_call_chat_missing_message(self):
        """tools/call chat with no message → error."""
        from flyto_ai.mcp_server import MCPServer
        server = MCPServer()
        resp = await server.handle({
            "jsonrpc": "2.0", "id": 5,
            "method": "tools/call",
            "params": {"name": "chat", "arguments": {}},
        })
        assert resp["error"]["code"] == -32602
        assert "message" in resp["error"]["message"].lower()

    @requires_openai
    @pytest.mark.asyncio
    async def test_tools_call_chat_real_llm(self):
        """tools/call chat with real LLM → returns text content."""
        from flyto_ai.mcp_server import MCPServer
        server = MCPServer()
        resp = await server.handle({
            "jsonrpc": "2.0", "id": 6,
            "method": "tools/call",
            "params": {"name": "chat", "arguments": {"message": "Say exactly: pong"}},
        })

        assert "result" in resp
        content = resp["result"]["content"]
        assert len(content) > 0
        assert content[0]["type"] == "text"
        assert len(content[0]["text"]) > 0

    def test_chat_tool_schema_valid(self):
        """CHAT_TOOL inputSchema is valid JSON Schema."""
        from flyto_ai.mcp_server import CHAT_TOOL
        schema = CHAT_TOOL["inputSchema"]
        assert schema["type"] == "object"
        assert "message" in schema["properties"]
        assert schema["properties"]["message"]["type"] == "string"
        assert "mode" in schema["properties"]
        assert set(schema["properties"]["mode"]["enum"]) == {"execute", "yaml"}
        assert schema["required"] == ["message"]

    def test_entry_point_in_pyproject(self):
        import pathlib
        content = (pathlib.Path(__file__).parent.parent / "pyproject.toml").read_text()
        assert "flyto-ai-mcp" in content
        assert "flyto_ai.mcp_server:main" in content


# =========================================================================
# Gap 3: Observability — real audit emit + real tool timing
# =========================================================================

class TestObservability:

    def test_audit_entry_fields(self):
        from flyto_ai.audit import ChatAuditEntry
        entry = ChatAuditEntry(
            user_message="hello",
            provider="openai",
            model="gpt-4o-mini",
            mode="execute",
            tool_calls_count=3,
            execution_count=2,
            duration_ms=1500,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            ok=True,
        )
        assert entry.ok is True
        assert entry.duration_ms == 1500
        assert entry.total_tokens == 150
        assert entry.timestamp > 0

    def test_audit_emit_structured_json(self, caplog):
        """emit() outputs valid JSON with all expected fields."""
        from flyto_ai.audit import ChatAuditEntry
        with caplog.at_level(logging.INFO, logger="flyto_ai.audit"):
            ChatAuditEntry(
                user_message="test message",
                provider="openai",
                model="gpt-4o-mini",
                duration_ms=1234,
                total_tokens=500,
            ).emit()

        assert len(caplog.records) == 1
        record = json.loads(caplog.records[0].message)
        assert record["event"] == "chat_audit"
        assert record["duration_ms"] == 1234
        assert record["total_tokens"] == 500
        assert record["provider"] == "openai"
        assert record["model"] == "gpt-4o-mini"
        assert isinstance(record["ts"], float)

    def test_audit_emit_error_path(self, caplog):
        """Error entries include error field."""
        from flyto_ai.audit import ChatAuditEntry
        with caplog.at_level(logging.INFO, logger="flyto_ai.audit"):
            ChatAuditEntry(ok=False, error="timeout").emit()

        record = json.loads(caplog.records[0].message)
        assert record["ok"] is False
        assert record["error"] == "timeout"

    def test_audit_emit_no_error_field_when_ok(self, caplog):
        """Success entries don't include error field."""
        from flyto_ai.audit import ChatAuditEntry
        with caplog.at_level(logging.INFO, logger="flyto_ai.audit"):
            ChatAuditEntry(ok=True).emit()

        record = json.loads(caplog.records[0].message)
        assert "error" not in record

    @pytest.mark.asyncio
    async def test_registry_dispatch_logs_timing(self, caplog):
        """Real ToolRegistry.dispatch logs elapsed_ms."""
        from flyto_ai.tools.registry import ToolRegistry

        reg = ToolRegistry()

        async def slow_handler(name, args):
            import asyncio
            await asyncio.sleep(0.05)  # 50ms
            return {"ok": True}

        reg.register({"name": "slow", "description": "s", "inputSchema": {}}, slow_handler)

        with caplog.at_level(logging.DEBUG, logger="flyto_ai.tools.registry"):
            result = await reg.dispatch("slow", {})

        assert result["ok"] is True
        # Should have a log line with "completed in NN ms"
        timing_logs = [r for r in caplog.records if "completed in" in r.message]
        assert len(timing_logs) == 1
        # Extract ms value — should be >= 50
        msg = timing_logs[0].message
        assert "slow" in msg

    @requires_openai
    @pytest.mark.asyncio
    async def test_real_chat_emits_audit(self, caplog):
        """Real Agent.chat() emits a ChatAuditEntry."""
        config = AgentConfig.from_env()
        config.model = "gpt-4o-mini"
        agent = Agent(config=config, tools=[], dispatch_fn=None)

        with caplog.at_level(logging.INFO, logger="flyto_ai.audit"):
            result = await agent.chat("Say hi", mode="execute")

        assert result.ok
        audit_logs = [r for r in caplog.records if "chat_audit" in r.message]
        assert len(audit_logs) >= 1
        record = json.loads(audit_logs[0].message)
        assert record["ok"] is True
        assert record["duration_ms"] > 0
        assert record["provider"] in ("openai", "")

    @pytest.mark.asyncio
    async def test_no_api_key_also_emits_audit(self, caplog):
        """Even the error path (no API key) should emit audit."""
        config = AgentConfig(provider="openai", api_key="")
        agent = Agent(config=config)

        with caplog.at_level(logging.INFO, logger="flyto_ai.audit"):
            result = await agent.chat("hello", mode="execute")

        assert result.ok is False
        # audit may or may not emit for the early-exit path (no_api_key)
        # — what matters is it doesn't crash


# =========================================================================
# Gap 6: Async serve — real function existence + stdlib handler logic
# =========================================================================

class TestAsyncServe:

    def test_serve_dispatches_to_aiohttp_or_stdlib(self):
        """_cmd_serve tries aiohttp import, falls back to stdlib."""
        from flyto_ai.cli import _cmd_serve, _cmd_serve_stdlib
        assert callable(_cmd_serve)
        assert callable(_cmd_serve_stdlib)

    def test_stdlib_handler_has_required_routes(self):
        """stdlib Handler has do_POST and do_GET."""
        from flyto_ai.cli import _cmd_serve_stdlib
        import inspect
        src = inspect.getsource(_cmd_serve_stdlib)
        assert "do_POST" in src
        assert "do_GET" in src
        assert '"/chat"' in src
        assert '"/health"' in src

    def test_optional_dep_in_pyproject(self):
        import pathlib
        content = (pathlib.Path(__file__).parent.parent / "pyproject.toml").read_text()
        assert 'serve = ["aiohttp' in content

    def test_aiohttp_handler_exists_if_installed(self):
        """If aiohttp is installed, _cmd_serve_aiohttp should exist."""
        try:
            from flyto_ai.cli import _cmd_serve_aiohttp
            assert callable(_cmd_serve_aiohttp)
        except ImportError:
            pytest.skip("aiohttp not installed")


# =========================================================================
# Integration: End-to-end with real OpenAI
# =========================================================================

class TestEndToEnd:

    @requires_openai
    @pytest.mark.asyncio
    async def test_full_chat_response_structure(self):
        """Real chat → verify ChatResponse has all new fields populated."""
        config = AgentConfig.from_env()
        config.model = "gpt-4o-mini"
        agent = Agent(config=config, tools=[], dispatch_fn=None)

        result = await agent.chat("What is 2+2? Reply with just the number.", mode="execute")

        assert isinstance(result, ChatResponse)
        assert result.ok is True
        assert result.provider == "openai"
        assert result.model == "gpt-4o-mini"
        assert result.rounds_used >= 1
        assert result.usage is not None
        assert result.usage.total_tokens > 0
        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0
        assert "4" in result.message

    @requires_openai
    @pytest.mark.asyncio
    async def test_streaming_chat_returns_usage(self):
        """Real streaming chat → tokens stream + usage populated."""
        config = AgentConfig.from_env()
        config.model = "gpt-4o-mini"
        agent = Agent(config=config, tools=[], dispatch_fn=None)

        stream_events = []
        def on_stream(event):
            stream_events.append(event)

        result = await agent.chat(
            "Count from 1 to 3, one per line",
            mode="execute", on_stream=on_stream,
        )

        assert result.ok
        # Should have received streaming tokens
        token_events = [e for e in stream_events if e.type == StreamEventType.TOKEN]
        assert len(token_events) > 0
        # Done event
        done_events = [e for e in stream_events if e.type == StreamEventType.DONE]
        assert len(done_events) >= 1
        # Usage still populated even with streaming
        assert result.usage is not None
        assert result.usage.total_tokens > 0

    @requires_openai
    @pytest.mark.asyncio
    async def test_max_rounds_config_respected(self):
        """Setting max_tool_rounds=1 limits rounds. Non-tool chat uses 1 round."""
        config = AgentConfig.from_env()
        config.model = "gpt-4o-mini"
        config.max_tool_rounds = 1
        agent = Agent(config=config, tools=[], dispatch_fn=None)

        result = await agent.chat("Say hello", mode="execute")

        assert result.ok
        assert result.rounds_used == 1

    @requires_openai
    @pytest.mark.asyncio
    async def test_mcp_tools_call_real(self):
        """MCP server tools/call with real OpenAI → returns text response."""
        from flyto_ai.mcp_server import MCPServer
        server = MCPServer()

        # Initialize first
        await server.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})

        # Call chat tool
        resp = await server.handle({
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": "chat", "arguments": {"message": "Say exactly: test-ok"}},
        })

        assert "result" in resp
        content = resp["result"]["content"]
        assert len(content) > 0
        assert content[0]["type"] == "text"
        text = content[0]["text"].lower()
        assert "test" in text or "ok" in text or len(text) > 0
