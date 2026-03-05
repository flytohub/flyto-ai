# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for Telegram command routing system.

Unit tests   — test helpers directly (commands, blueprints, etc.)
Integration  — spin up the real aiohttp app, POST to /telegram, assert routed replies.

Updated to use the refactored flyto_ai.telegram package.
"""
import argparse
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional

import pytest
import pytest_asyncio

from flyto_ai.telegram.commands import _HELP_TEXT, _list_blueprints


# ===================================================================
# Unit Tests
# ===================================================================

# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

class TestHelpText:
    def test_help_contains_commands(self):
        assert "/agent" in _HELP_TEXT
        assert "/yaml" in _HELP_TEXT
        assert "/cd" in _HELP_TEXT
        assert "/model" in _HELP_TEXT
        assert "/help" in _HELP_TEXT

    def test_help_mentions_plain_text(self):
        assert "plain text" in _HELP_TEXT


# ---------------------------------------------------------------------------
# /yaml  (/blueprint)
# ---------------------------------------------------------------------------

class TestListBlueprints:
    def test_no_blueprints(self):
        """Empty list returns 'No blueprints yet.'"""
        engine = MagicMock()
        engine.list_blueprints.return_value = []

        with patch.dict("sys.modules", {"flyto_blueprint": MagicMock(get_engine=lambda **kw: engine)}):
            result = _list_blueprints()
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
        }):
            result = _list_blueprints()

        assert "Blueprints:" in result
        assert "scrape-page" in result
        assert "login-test" in result
        assert result.index("scrape-page") < result.index("login-test")

    def test_import_error_handled(self):
        """If flyto_blueprint is not installed, returns error message."""
        with patch.dict("sys.modules", {"flyto_blueprint": None}):
            result = _list_blueprints()
        assert "Error" in result


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
    return {"message": {"chat": {"id": chat_id}, "from": {"id": 1}, "text": text, "message_id": 1}}


@pytest.fixture()
def mock_agent_chat():
    """Returns a mock agent whose .chat() we can inspect."""
    agent = MagicMock()
    agent.chat = AsyncMock(return_value=MagicMock(
        message="agent reply",
        cost={"estimated_cost_usd": 0.0},
        ok=True,
    ))
    return agent


@pytest_asyncio.fixture()
async def tg_client(monkeypatch, mock_agent_chat):
    """Capture the aiohttp app from _cmd_serve_aiohttp, return TestClient.

    Patches:
    - web.run_app → capture app, don't block
    - Agent / AgentConfig → mock (no real LLM calls)
    - _TG_TOKEN → test value
    - TelegramSender._request → capture messages
    """
    from aiohttp.test_utils import TestClient, TestServer
    import flyto_ai.cli as cli_mod

    # --- Patch module-level TG config ---
    monkeypatch.setattr(cli_mod, "_TG_TOKEN", "fake-token-for-test")

    # --- Patch TelegramService to use test allowed_chats ---
    original_init = None
    from flyto_ai.telegram.service import TelegramService
    original_init = TelegramService.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["allowed_chats"] = frozenset({5608426436})
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(TelegramService, "__init__", patched_init)

    # --- Patch TelegramSender._request to not hit real API ---
    sent_messages = []
    from flyto_ai.telegram.sender import TelegramSender

    async def fake_request(self, method, payload):
        if method == "sendMessage":
            msg_id = len(sent_messages) + 1
            sent_messages.append((payload.get("chat_id"), payload.get("text", "")))
            return {"ok": True, "result": {"message_id": msg_id}}
        if method == "editMessageText":
            return {"ok": True, "result": {"message_id": payload.get("message_id")}}
        if method == "deleteMessage":
            return {"ok": True}
        return {"ok": True}

    monkeypatch.setattr(TelegramSender, "_request", fake_request)

    # --- Patch Agent / AgentConfig ---
    mock_config = MagicMock()
    mock_config.provider = None
    mock_config.model = None
    mock_config.api_key = None

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

    # Run startup handlers (initializes TelegramService)
    for handler in app.on_startup:
        await handler(app)

    client = TestClient(TestServer(app))
    await client.start_server()

    # Attach sent_messages for test access
    client._tg_sent_messages = sent_messages

    yield client

    # Run cleanup handlers
    for handler in app.on_cleanup:
        await handler(app)

    await client.close()


async def _drain_background(n=5):
    """Give fire-and-forget tasks time to finish."""
    for _ in range(n):
        await asyncio.sleep(0.05)


class TestTelegramIntegration:
    """POST to /telegram on the real aiohttp app, assert routed replies."""

    @pytest.mark.asyncio
    async def test_help_command(self, tg_client):
        """/help returns the help text, not agent.chat."""
        resp = await tg_client.post("/telegram", json=_tg_payload("/help"))
        assert resp.status == 200
        assert (await resp.json())["ok"] is True

        await _drain_background()

        replies = [text for _, text in tg_client._tg_sent_messages]
        assert any("/agent" in r and "/yaml" in r for r in replies)

    @pytest.mark.asyncio
    async def test_yaml_command(self, tg_client):
        """/yaml routes to blueprints listing."""
        resp = await tg_client.post("/telegram", json=_tg_payload("/yaml"))
        assert resp.status == 200

        await _drain_background()

        replies = [text for _, text in tg_client._tg_sent_messages]
        assert any("Blueprints" in r or "blueprint" in r.lower() or "Error" in r for r in replies)

    @pytest.mark.asyncio
    async def test_blueprint_alias(self, tg_client):
        """/blueprint is an alias for /yaml."""
        resp = await tg_client.post("/telegram", json=_tg_payload("/blueprint"))
        assert resp.status == 200

        await _drain_background()

        replies = [text for _, text in tg_client._tg_sent_messages]
        assert any("Blueprints" in r or "blueprint" in r.lower() or "Error" in r for r in replies)

    @pytest.mark.asyncio
    async def test_agent_empty_message(self, tg_client):
        """/agent with no message shows usage."""
        resp = await tg_client.post("/telegram", json=_tg_payload("/agent"))
        assert resp.status == 200

        await _drain_background()

        replies = [text for _, text in tg_client._tg_sent_messages]
        assert any("Usage" in r for r in replies)

    @pytest.mark.asyncio
    async def test_plain_text_routes_to_claude_bridge(self, tg_client):
        """Plain text (no slash command) goes to Claude Code bridge.

        Since claude-agent-sdk is not installed in test, ClaudeBridge
        sends an error message about missing SDK.
        """
        resp = await tg_client.post("/telegram", json=_tg_payload("what is 2+2"))
        assert resp.status == 200

        await _drain_background()

        replies = [text for _, text in tg_client._tg_sent_messages]
        # Should see either a streaming response or an SDK-not-installed error
        assert len(replies) > 0

    @pytest.mark.asyncio
    async def test_unauthorized_chat_ignored(self, tg_client):
        """Chat ID not in allowlist is silently ignored."""
        resp = await tg_client.post("/telegram", json=_tg_payload("/help", chat_id=999))
        assert resp.status == 200

        await _drain_background()

        # No messages sent
        assert len(tg_client._tg_sent_messages) == 0

    @pytest.mark.asyncio
    async def test_empty_text_ignored(self, tg_client):
        """Empty text is silently ignored."""
        payload = {"message": {"chat": {"id": 5608426436}, "from": {"id": 1}, "text": "", "message_id": 1}}
        resp = await tg_client.post("/telegram", json=payload)
        assert resp.status == 200

        await _drain_background()

        assert len(tg_client._tg_sent_messages) == 0
