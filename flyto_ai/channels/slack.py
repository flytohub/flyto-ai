# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Slack channel adapter."""
import logging
import os
from typing import Any, Dict, Optional

from flyto_ai.channels.base import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH = 3000  # Slack block text limit


class SlackAdapter(ChannelAdapter):
    """Slack Bot adapter (Events API + Web API)."""

    def __init__(self, bot_token: Optional[str] = None, webhook_url: Optional[str] = None) -> None:
        self._bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN", "")
        self._webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")

    @property
    def channel_name(self) -> str:
        return "slack"

    async def parse_incoming(self, raw_payload: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Parse Slack Events API payload into IncomingMessage."""
        # URL verification challenge
        if raw_payload.get("type") == "url_verification":
            return None

        event = raw_payload.get("event", {})
        if not event:
            return None

        # Only handle message events (not bot messages)
        if event.get("type") != "message" or event.get("subtype"):
            return None
        if event.get("bot_id"):
            return None

        text = event.get("text", "")
        if not text:
            return None

        return IncomingMessage(
            channel="slack",
            user_id=event.get("user", ""),
            text=text,
            session_id=event.get("channel", ""),
            metadata={
                "team_id": raw_payload.get("team_id", ""),
                "thread_ts": event.get("thread_ts", ""),
                "ts": event.get("ts", ""),
            },
        )

    async def send(self, message: OutgoingMessage) -> bool:
        """Send message to Slack."""
        try:
            text = message.text
            chunks = [text[i:i + _MAX_MESSAGE_LENGTH] for i in range(0, len(text), _MAX_MESSAGE_LENGTH)]

            # Prefer webhook for simple messages
            if self._webhook_url:
                return await self._send_webhook(chunks)

            # Bot API
            if self._bot_token and message.session_id:
                return await self._send_bot_message(
                    message.session_id, chunks,
                    thread_ts=message.metadata.get("thread_ts", ""),
                )

            logger.warning("Slack: no sending method configured")
            return False
        except ImportError:
            logger.warning("Slack: aiohttp not installed")
            return False
        except Exception as e:
            logger.warning("Slack send error: %s", e)
            return False

    async def _send_webhook(self, chunks):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            for chunk in chunks:
                async with session.post(self._webhook_url, json={"text": chunk}) as resp:
                    if resp.status != 200:
                        return False
        return True

    async def _send_bot_message(self, channel, chunks, thread_ts=""):
        import aiohttp
        headers = {
            "Authorization": "Bearer {}".format(self._bot_token),
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession(headers=headers) as session:
            for chunk in chunks:
                payload = {"channel": channel, "text": chunk}
                if thread_ts:
                    payload["thread_ts"] = thread_ts
                async with session.post(
                    "https://slack.com/api/chat.postMessage", json=payload
                ) as resp:
                    data = await resp.json()
                    if not data.get("ok"):
                        logger.warning("Slack API error: %s", data.get("error", ""))
                        return False
        return True
