# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Telegram channel adapter."""
import logging
import os
from typing import Any, Dict, Optional

from flyto_ai.channels.base import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

# Telegram message length limit
_MAX_MESSAGE_LENGTH = 4096


class TelegramAdapter(ChannelAdapter):
    """Telegram Bot API adapter.

    Handles webhook payloads from Telegram and sends responses
    via the Bot API.
    """

    def __init__(self, bot_token: Optional[str] = None) -> None:
        self._bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")

    @property
    def channel_name(self) -> str:
        return "telegram"

    async def parse_incoming(self, raw_payload: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Parse Telegram webhook update into IncomingMessage.

        Handles both regular messages and callback_query (inline keyboard presses).
        """
        # Handle callback_query (inline keyboard button press)
        callback = raw_payload.get("callback_query")
        if callback:
            user = callback.get("from", {})
            msg = callback.get("message", {})
            chat = msg.get("chat", {})
            return IncomingMessage(
                channel="telegram",
                user_id=str(user.get("id", "")),
                text=callback.get("data", ""),
                session_id=str(chat.get("id", "")),
                metadata={
                    "type": "callback_query",
                    "callback_query_id": callback.get("id", ""),
                    "chat_type": chat.get("type", ""),
                    "username": user.get("username", ""),
                    "first_name": user.get("first_name", ""),
                    "message_id": msg.get("message_id", 0),
                },
            )

        # Handle regular messages
        message = raw_payload.get("message") or raw_payload.get("edited_message")
        if not message:
            return None

        text = message.get("text", "")
        caption = message.get("caption", "")
        attachments: list = []

        # Photo — pick highest resolution (last in array)
        photos = message.get("photo")
        if photos:
            best = photos[-1]
            attachments.append({"type": "photo", "file_id": best["file_id"]})
            text = text or caption

        # Voice message
        voice = message.get("voice")
        if voice:
            attachments.append({
                "type": "voice",
                "file_id": voice["file_id"],
                "duration": str(voice.get("duration", 0)),
            })

        if not text and not attachments:
            return None

        chat = message.get("chat", {})
        user = message.get("from", {})

        return IncomingMessage(
            channel="telegram",
            user_id=str(user.get("id", "")),
            text=text,
            session_id=str(chat.get("id", "")),
            attachments=attachments,
            metadata={
                "type": "message",
                "chat_type": chat.get("type", ""),
                "username": user.get("username", ""),
                "first_name": user.get("first_name", ""),
                "message_id": message.get("message_id", 0),
            },
        )

    async def send(self, message: OutgoingMessage) -> bool:
        """Send message via Telegram Bot API."""
        if not self._bot_token:
            logger.warning("Telegram: no bot token configured")
            return False

        try:
            import aiohttp

            chat_id = message.session_id or message.user_id
            text = message.text

            # Split long messages
            chunks = [text[i:i + _MAX_MESSAGE_LENGTH] for i in range(0, len(text), _MAX_MESSAGE_LENGTH)]

            url = "https://api.telegram.org/bot{}/sendMessage".format(self._bot_token)
            async with aiohttp.ClientSession() as session:
                for chunk in chunks:
                    payload = {
                        "chat_id": chat_id,
                        "text": chunk,
                        "parse_mode": "Markdown",
                    }
                    async with session.post(url, json=payload) as resp:
                        if resp.status != 200:
                            body = await resp.text()
                            logger.warning("Telegram send failed: %s %s", resp.status, body[:200])
                            return False
            return True
        except ImportError:
            logger.warning("Telegram: aiohttp not installed")
            return False
        except Exception as e:
            logger.warning("Telegram send error: %s", e)
            return False
