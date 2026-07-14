# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Discord channel adapter."""
import logging
import os
from typing import Any, Dict, Optional

from flyto_ai.channels.base import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH = 2000


class DiscordAdapter(ChannelAdapter):
    """Discord Bot adapter (webhook-based).

    Supports Discord Interactions API (slash commands and message components)
    and traditional bot message events.
    """

    def __init__(self, bot_token: Optional[str] = None, webhook_url: Optional[str] = None) -> None:
        self._bot_token = bot_token or os.getenv("DISCORD_BOT_TOKEN", "")
        self._webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")

    @property
    def channel_name(self) -> str:
        return "discord"

    async def parse_incoming(self, raw_payload: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Parse Discord event into IncomingMessage.

        Handles both MESSAGE_CREATE events and Interaction payloads.
        """
        # Interaction (slash command)
        if raw_payload.get("type") == 2:
            data = raw_payload.get("data", {})
            options = data.get("options", [])
            text = " ".join(
                str(opt.get("value", "")) for opt in options
            ) or data.get("name", "")
            user = raw_payload.get("member", {}).get("user", {}) or raw_payload.get("user", {})
            return IncomingMessage(
                channel="discord",
                user_id=str(user.get("id", "")),
                text=text,
                session_id=str(raw_payload.get("channel_id", "")),
                metadata={
                    "interaction_id": raw_payload.get("id", ""),
                    "interaction_token": raw_payload.get("token", ""),
                    "guild_id": raw_payload.get("guild_id", ""),
                    "command_name": data.get("name", ""),
                },
            )

        # MESSAGE_CREATE event
        if raw_payload.get("t") == "MESSAGE_CREATE" or "content" in raw_payload:
            content = raw_payload.get("content") or raw_payload.get("d", {}).get("content", "")
            if not content:
                return None
            msg_data = raw_payload.get("d", raw_payload)
            author = msg_data.get("author", {})
            # Ignore bot messages
            if author.get("bot", False):
                return None
            return IncomingMessage(
                channel="discord",
                user_id=str(author.get("id", "")),
                text=content,
                session_id=str(msg_data.get("channel_id", "")),
                metadata={
                    "username": author.get("username", ""),
                    "guild_id": str(msg_data.get("guild_id", "")),
                    "message_id": str(msg_data.get("id", "")),
                },
            )

        return None

    async def send(self, message: OutgoingMessage) -> bool:
        """Send message to Discord channel."""
        try:
            text = message.text
            chunks = [text[i:i + _MAX_MESSAGE_LENGTH] for i in range(0, len(text), _MAX_MESSAGE_LENGTH)]

            # Prefer interaction response if token available
            interaction_token = message.metadata.get("interaction_token", "")
            if interaction_token:
                return await self._send_interaction_response(interaction_token, chunks)

            # Fall back to webhook
            if self._webhook_url:
                return await self._send_webhook(chunks)

            # Fall back to bot API
            if self._bot_token and message.session_id:
                return await self._send_bot_message(message.session_id, chunks)

            logger.warning("Discord: no sending method configured")
            return False
        except ImportError:
            logger.warning("Discord: aiohttp not installed")
            return False
        except Exception as e:
            logger.warning("Discord send error: %s", e)
            return False

    async def _send_webhook(self, chunks):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            for chunk in chunks:
                async with session.post(self._webhook_url, json={"content": chunk}) as resp:
                    if resp.status not in (200, 204):
                        return False
        return True

    async def _send_bot_message(self, channel_id, chunks):
        import aiohttp
        url = "https://discord.com/api/v10/channels/{}/messages".format(channel_id)
        headers = {"Authorization": "Bot {}".format(self._bot_token)}
        async with aiohttp.ClientSession(headers=headers) as session:
            for chunk in chunks:
                async with session.post(url, json={"content": chunk}) as resp:
                    if resp.status != 200:
                        return False
        return True

    async def _send_interaction_response(self, token, chunks):
        import aiohttp
        # Respond to first chunk as interaction followup
        url = "https://discord.com/api/v10/webhooks/{app_id}/{token}".format(
            app_id="self", token=token
        )
        async with aiohttp.ClientSession() as session:
            for chunk in chunks:
                async with session.post(url, json={"content": chunk}) as resp:
                    if resp.status not in (200, 204):
                        return False
        return True
