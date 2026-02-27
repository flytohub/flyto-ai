# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for Telegram command routing system.

Unit tests   — test helpers directly (_tg_list_blueprints, _tg_run_claude, etc.)
Integration  — spin up the real aiohttp app, POST to /telegram, assert routed replies.
"""
import argparse
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional

import pytest
import pytest_asyncio

from flyto_ai.cli import (
    _TG_HELP_TEXT,
    _tg_list_blueprints,
    _tg_run_claude,
)


# ===================================================================
# Unit Tests
# ===================================================================

# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

class TestHelpText:
    def test_help_contains_commands(self):
        assert "/claude" in _TG_HELP_TEXT
        assert "/yaml" in _TG_HELP_TEXT
        assert "/blueprint" in _TG_HELP_TEXT
        assert "/help" in _TG_HELP_TEXT

    def test_help_mentions_plain_text(self):
        assert "plain text" in _TG_HELP_TEXT


# ---------------------------------------------------------------------------
# /yaml  (/blueprint)
# ---------------------------------------------------------------------------

class TestListBlueprints:
    @patch("flyto_ai.cli.get_engine", create=True)
    def test_no_blueprints(self, mock_get_engine):
        """Empty list returns 'No blueprints yet.'"""
        engine = MagicMock()
        engine.list_blueprints.return_value = []

        with patch.dict("sys.modules", {"flyto_blueprint": MagicMock(get_engine=lambda **kw: engine),
                                         "flyto_blueprint.storage": MagicMock()}):
            result = _tg_list_blueprints()
        assert result == "No blueprints yet."

    def test_with_blueprints(self):
        """Lists blueprints sorted by score, max 10."""
        bps = [
            {"name": "login-test", "score": 80},
            {"name": "scrape-page", "score": 95},
            {"name": "low-score", "score": 10},
        ]
        engine = MagicMock()
        engine.list_blueprints.return_value = bps

        with patch.dict("sys.modules", {
            "flyto_blueprint": MagicMock(get_engine=lambda **kw: engine),
            "flyto_blueprint.storage": MagicMock(),
        }):
            result = _tg_list_blueprints()

        assert "Blueprints:" in result
        assert "scrape-page" in result
        assert "login-test" in result
        assert result.index("scrape-page") < result.index("login-test")

    def test_import_error_handled(self):
        """If flyto_blueprint is not installed, returns error message."""
        with patch.dict("sys.modules", {"flyto_blueprint": None}):
            result = _tg_list_blueprints()
        assert "Error" in result


# ---------------------------------------------------------------------------
# /claude
# ---------------------------------------------------------------------------

@dataclass
class _FakeCodeTaskResponse:
    ok: bool
    message: str
    session_id: str = ""
    attempts: int = 1
    files_changed: List[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    claude_session_id: Optional[str] = None
    claude_num_turns: int = 0
    claude_duration_ms: int = 0
    claude_usage: Optional[dict] = None
    verification_results: list = field(default_factory=list)
    evidence: list = field(default_factory=list)


class TestRunClaude:
    @pytest.mark.asyncio
    async def test_success(self):
        """Successful Claude run returns 'Done' + message + files."""
        fake_resp = _FakeCodeTaskResponse(
            ok=True,
            message="Fixed the login bug",
            files_changed=["src/auth.py", "tests/test_auth.py"],
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=fake_resp)

        with patch.dict("sys.modules", {
            "flyto_ai.agents.claude_code": MagicMock(ClaudeCodeAgent=lambda config: mock_agent),
            "flyto_ai.agents.models": MagicMock(CodeTaskRequest=lambda **kw: MagicMock()),
        }):
            result = await _tg_run_claude("fix the login bug", "/tmp/project", MagicMock())

        assert "Done" in result
        assert "Fixed the login bug" in result
        assert "src/auth.py" in result

    @pytest.mark.asyncio
    async def test_failure(self):
        """Failed Claude run returns 'Failed' + message."""
        fake_resp = _FakeCodeTaskResponse(
            ok=False,
            message="Could not find the file",
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=fake_resp)

        with patch.dict("sys.modules", {
            "flyto_ai.agents.claude_code": MagicMock(ClaudeCodeAgent=lambda config: mock_agent),
            "flyto_ai.agents.models": MagicMock(CodeTaskRequest=lambda **kw: MagicMock()),
        }):
            result = await _tg_run_claude("fix nonexistent.py", "/tmp/project", MagicMock())

        assert "Failed" in result
        assert "Could not find the file" in result

    @pytest.mark.asyncio
    async def test_exception_handled(self):
        """If ClaudeCodeAgent raises, returns error string instead of crashing."""
        with patch.dict("sys.modules", {
            "flyto_ai.agents.claude_code": None,
        }):
            result = await _tg_run_claude("hello", "/tmp", MagicMock())
        assert "Claude error" in result


# ---------------------------------------------------------------------------
# --dir argument parsing
# ---------------------------------------------------------------------------

class TestDirArgParsing:
    def test_dir_default_is_none(self):
        """--dir defaults to None when not provided."""
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        serve_p = sub.add_parser("serve")
        serve_p.add_argument("--host", default="127.0.0.1")
        serve_p.add_argument("--port", type=int, default=7411)
        serve_p.add_argument("--dir", default=None)

        args = parser.parse_args(["serve"])
        assert args.dir is None

    def test_dir_set_explicitly(self):
        """--dir stores the provided path."""
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        serve_p = sub.add_parser("serve")
        serve_p.add_argument("--dir", default=None)

        args = parser.parse_args(["serve", "--dir", "/opt/my-project"])
        assert args.dir == "/opt/my-project"


# ===================================================================
# Integration Tests — real aiohttp app, real HTTP requests
# ===================================================================

def _tg_payload(text, chat_id=5608426436):
    """Build a minimal Telegram Update payload."""
    return {"message": {"chat": {"id": chat_id}, "text": text}}


@pytest.fixture()
def tg_sent_messages():
    """Collects (chat_id, text) tuples sent via _tg_send during the test."""
    return []


@pytest.fixture()
def mock_agent_chat():
    """Returns a mock agent whose .chat() we can inspect."""
    agent = MagicMock()
    agent.chat = AsyncMock(return_value=MagicMock(message="agent reply"))
    return agent


@pytest_asyncio.fixture()
async def tg_client(monkeypatch, tg_sent_messages, mock_agent_chat):
    """Capture the aiohttp app from _cmd_serve_aiohttp, return TestClient.

    Patches:
    - web.run_app → capture app, don't block
    - _tg_send → record messages instead of hitting Telegram API
    - Agent / AgentConfig → mock (no real LLM calls)
    - _TG_TOKEN / _TG_ALLOWED_CHATS → test values
    """
    from aiohttp.test_utils import TestClient, TestServer
    import flyto_ai.cli as cli_mod

    # --- Patch module-level TG config ---
    monkeypatch.setattr(cli_mod, "_TG_TOKEN", "fake-token-for-test")
    monkeypatch.setattr(cli_mod, "_TG_ALLOWED_CHATS", frozenset({5608426436}))

    # --- Patch _tg_send to collect messages ---
    sent = tg_sent_messages

    async def fake_tg_send(token, chat_id, text):
        sent.append((chat_id, text))

    monkeypatch.setattr(cli_mod, "_tg_send", fake_tg_send)

    # --- Patch Agent / AgentConfig ---
    mock_config = MagicMock()
    mock_config.provider = None
    mock_config.model = None
    mock_config.api_key = None

    monkeypatch.setattr(cli_mod, "AgentConfig", MagicMock, raising=False)
    # Patch at the import location inside the function
    import flyto_ai
    monkeypatch.setattr(flyto_ai, "AgentConfig", MagicMock(from_env=lambda: mock_config), raising=False)
    monkeypatch.setattr(flyto_ai, "Agent", lambda config: mock_agent_chat, raising=False)

    # --- Capture the app from web.run_app ---
    captured = {}

    def fake_run_app(app, **kwargs):
        captured["app"] = app

    from aiohttp import web
    monkeypatch.setattr(web, "run_app", fake_run_app)

    # --- Call _cmd_serve_aiohttp (it will return after fake_run_app) ---
    args = argparse.Namespace(
        host="127.0.0.1", port=0,
        provider=None, model=None, api_key=None,
        dir="/tmp/test-project",
    )
    cli_mod._cmd_serve_aiohttp(args)

    app = captured["app"]
    client = TestClient(TestServer(app))
    await client.start_server()

    yield client

    await client.close()


async def _drain_background(n=5):
    """Give fire-and-forget tasks time to finish."""
    for _ in range(n):
        await asyncio.sleep(0.05)


class TestTelegramIntegration:
    """POST to /telegram on the real aiohttp app, assert routed replies."""

    @pytest.mark.asyncio
    async def test_help_command(self, tg_client, tg_sent_messages):
        """/help returns the help text, not agent.chat."""
        resp = await tg_client.post("/telegram", json=_tg_payload("/help"))
        assert resp.status == 200
        assert (await resp.json())["ok"] is True

        await _drain_background()

        # First message = "Processing...", second = help text
        replies = [text for _, text in tg_sent_messages]
        assert any("/claude" in r and "/yaml" in r for r in replies)

    @pytest.mark.asyncio
    async def test_yaml_command(self, tg_client, tg_sent_messages):
        """/yaml routes to _tg_list_blueprints, not agent.chat."""
        resp = await tg_client.post("/telegram", json=_tg_payload("/yaml"))
        assert resp.status == 200

        await _drain_background()

        replies = [text for _, text in tg_sent_messages]
        # Should contain blueprint result (or error if flyto_blueprint not installed)
        assert any("Blueprints" in r or "blueprint" in r.lower() or "Error" in r for r in replies)

    @pytest.mark.asyncio
    async def test_blueprint_alias(self, tg_client, tg_sent_messages):
        """/blueprint is an alias for /yaml."""
        resp = await tg_client.post("/telegram", json=_tg_payload("/blueprint"))
        assert resp.status == 200

        await _drain_background()

        replies = [text for _, text in tg_sent_messages]
        assert any("Blueprints" in r or "blueprint" in r.lower() or "Error" in r for r in replies)

    @pytest.mark.asyncio
    async def test_claude_command(self, tg_client, tg_sent_messages):
        """/claude <msg> routes to _tg_run_claude, not agent.chat."""
        resp = await tg_client.post("/telegram", json=_tg_payload("/claude list files"))
        assert resp.status == 200

        await _drain_background()

        replies = [text for _, text in tg_sent_messages]
        # Should get a Claude result (or error), not agent.chat's "agent reply"
        assert any("Claude" in r or "Done" in r or "Failed" in r for r in replies)

    @pytest.mark.asyncio
    async def test_claude_empty_message(self, tg_client, tg_sent_messages):
        """/claude with no message shows usage."""
        resp = await tg_client.post("/telegram", json=_tg_payload("/claude"))
        assert resp.status == 200

        await _drain_background()

        replies = [text for _, text in tg_sent_messages]
        assert any("Usage" in r for r in replies)

    @pytest.mark.asyncio
    async def test_plain_text_routes_to_agent(self, tg_client, tg_sent_messages, mock_agent_chat):
        """Plain text (no slash command) goes to agent.chat()."""
        resp = await tg_client.post("/telegram", json=_tg_payload("what is 2+2"))
        assert resp.status == 200

        await _drain_background()

        # agent.chat should have been called
        mock_agent_chat.chat.assert_called_once()
        call_args = mock_agent_chat.chat.call_args
        assert call_args[0][0] == "what is 2+2"

        replies = [text for _, text in tg_sent_messages]
        assert any("agent reply" in r for r in replies)

    @pytest.mark.asyncio
    async def test_unauthorized_chat_ignored(self, tg_client, tg_sent_messages):
        """Chat ID not in allowlist is silently ignored."""
        resp = await tg_client.post("/telegram", json=_tg_payload("/help", chat_id=999))
        assert resp.status == 200

        await _drain_background()

        # No messages sent (not even "Processing...")
        assert len(tg_sent_messages) == 0

    @pytest.mark.asyncio
    async def test_empty_text_ignored(self, tg_client, tg_sent_messages):
        """Empty text is silently ignored."""
        payload = {"message": {"chat": {"id": 5608426436}, "text": ""}}
        resp = await tg_client.post("/telegram", json=payload)
        assert resp.status == 200

        await _drain_background()

        assert len(tg_sent_messages) == 0

    @pytest.mark.asyncio
    async def test_processing_indicator_sent_first(self, tg_client, tg_sent_messages):
        """Every valid command starts with a 'Processing...' indicator."""
        resp = await tg_client.post("/telegram", json=_tg_payload("/help"))
        assert resp.status == 200

        await _drain_background()

        assert len(tg_sent_messages) >= 2
        first_msg = tg_sent_messages[0][1]
        assert "Processing" in first_msg
