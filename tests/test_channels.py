# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for multi-channel adapter framework."""
import pytest

from flyto_ai.channels.telegram import TelegramAdapter
from flyto_ai.channels.discord import DiscordAdapter
from flyto_ai.channels.slack import SlackAdapter
from flyto_ai.channels.webhook import WebhookAdapter
from flyto_ai.channels.router import ChannelRouter


# --- Telegram ---

@pytest.mark.asyncio
async def test_telegram_parse_message():
    adapter = TelegramAdapter()
    payload = {
        "message": {
            "text": "hello flyto",
            "from": {"id": 123, "username": "testuser", "first_name": "Test"},
            "chat": {"id": 456, "type": "private"},
            "message_id": 789,
        }
    }
    msg = await adapter.parse_incoming(payload)
    assert msg is not None
    assert msg.text == "hello flyto"
    assert msg.user_id == "123"
    assert msg.session_id == "456"
    assert msg.channel == "telegram"


@pytest.mark.asyncio
async def test_telegram_parse_no_text():
    adapter = TelegramAdapter()
    msg = await adapter.parse_incoming({"message": {"photo": []}})
    assert msg is None


@pytest.mark.asyncio
async def test_telegram_parse_no_message():
    adapter = TelegramAdapter()
    msg = await adapter.parse_incoming({"callback_query": {}})
    assert msg is None


# --- Discord ---

@pytest.mark.asyncio
async def test_discord_parse_message():
    adapter = DiscordAdapter()
    payload = {
        "t": "MESSAGE_CREATE",
        "d": {
            "content": "hello discord",
            "author": {"id": "111", "username": "bot_test", "bot": False},
            "channel_id": "222",
            "guild_id": "333",
            "id": "444",
        },
    }
    msg = await adapter.parse_incoming(payload)
    assert msg is not None
    assert msg.text == "hello discord"
    assert msg.channel == "discord"


@pytest.mark.asyncio
async def test_discord_ignore_bot():
    adapter = DiscordAdapter()
    payload = {
        "t": "MESSAGE_CREATE",
        "d": {
            "content": "bot message",
            "author": {"id": "111", "bot": True},
            "channel_id": "222",
        },
    }
    msg = await adapter.parse_incoming(payload)
    assert msg is None


@pytest.mark.asyncio
async def test_discord_parse_interaction():
    adapter = DiscordAdapter()
    payload = {
        "type": 2,
        "data": {"name": "ask", "options": [{"value": "what is python"}]},
        "member": {"user": {"id": "111"}},
        "channel_id": "222",
        "id": "333",
        "token": "tok",
    }
    msg = await adapter.parse_incoming(payload)
    assert msg is not None
    assert "python" in msg.text
    assert msg.metadata.get("command_name") == "ask"


# --- Slack ---

@pytest.mark.asyncio
async def test_slack_parse_message():
    adapter = SlackAdapter()
    payload = {
        "event": {
            "type": "message",
            "text": "hello slack",
            "user": "U123",
            "channel": "C456",
            "ts": "1234567890.123",
        },
        "team_id": "T789",
    }
    msg = await adapter.parse_incoming(payload)
    assert msg is not None
    assert msg.text == "hello slack"
    assert msg.channel == "slack"


@pytest.mark.asyncio
async def test_slack_ignore_bot():
    adapter = SlackAdapter()
    payload = {
        "event": {
            "type": "message",
            "text": "bot message",
            "bot_id": "B123",
            "channel": "C456",
        },
    }
    msg = await adapter.parse_incoming(payload)
    assert msg is None


@pytest.mark.asyncio
async def test_slack_url_verification():
    adapter = SlackAdapter()
    msg = await adapter.parse_incoming({"type": "url_verification", "challenge": "abc"})
    assert msg is None


# --- Webhook ---

@pytest.mark.asyncio
async def test_webhook_parse():
    adapter = WebhookAdapter()
    payload = {
        "message": "custom webhook message",
        "user_id": "user1",
        "session_id": "sess1",
    }
    msg = await adapter.parse_incoming(payload)
    assert msg is not None
    assert msg.text == "custom webhook message"
    assert msg.channel == "webhook"


@pytest.mark.asyncio
async def test_webhook_parse_empty():
    adapter = WebhookAdapter()
    msg = await adapter.parse_incoming({})
    assert msg is None


# --- Router ---

@pytest.mark.asyncio
async def test_router_register():
    router = ChannelRouter()
    router.register(TelegramAdapter())
    router.register(SlackAdapter())
    assert sorted(router.channels) == ["slack", "telegram"]


@pytest.mark.asyncio
async def test_router_unknown_channel():
    router = ChannelRouter()
    result = await router.handle("unknown", {})
    assert result is None


@pytest.mark.asyncio
async def test_router_no_handler():
    router = ChannelRouter()
    router.register(WebhookAdapter())
    result = await router.handle("webhook", {"message": "test"})
    assert result is None


@pytest.mark.asyncio
async def test_router_handle():
    router = ChannelRouter()
    router.register(WebhookAdapter())

    async def handler(msg):
        return "Echo: {}".format(msg.text)

    router.set_handler(handler)
    result = await router.handle("webhook", {"message": "hello"})
    assert result == "Echo: hello"


def test_router_unregister():
    router = ChannelRouter()
    router.register(TelegramAdapter())
    assert "telegram" in router.channels
    router.unregister("telegram")
    assert "telegram" not in router.channels
