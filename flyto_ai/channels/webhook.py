# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Generic webhook channel adapter."""
import logging
from typing import Any, Dict, Optional

from flyto_ai.channels.base import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


class WebhookAdapter(ChannelAdapter):
    """Generic webhook adapter for custom integrations.

    Incoming: expects JSON with 'message' and optional 'user_id', 'session_id'.
    Outgoing: POSTs JSON to a configured callback URL.
    """

    def __init__(self, callback_url: Optional[str] = None) -> None:
        self._callback_url = callback_url

    @property
    def channel_name(self) -> str:
        return "webhook"

    async def parse_incoming(self, raw_payload: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Parse generic webhook payload.

        Expected format:
        {
            "message": "user's text",
            "user_id": "optional",
            "session_id": "optional",
            "metadata": {}
        }
        """
        text = raw_payload.get("message", "") or raw_payload.get("text", "")
        if not text:
            return None

        return IncomingMessage(
            channel="webhook",
            user_id=raw_payload.get("user_id", "anonymous"),
            text=text,
            session_id=raw_payload.get("session_id", ""),
            metadata=raw_payload.get("metadata", {}),
        )

    async def send(self, message: OutgoingMessage) -> bool:
        """Send response to callback URL."""
        if not self._callback_url:
            logger.debug("Webhook: no callback URL, skipping send")
            return False

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                payload = {
                    "message": message.text,
                    "user_id": message.user_id,
                    "session_id": message.session_id,
                    "metadata": message.metadata,
                }
                async with session.post(self._callback_url, json=payload) as resp:
                    return resp.status in (200, 201, 202, 204)
        except ImportError:
            logger.warning("Webhook: aiohttp not installed")
            return False
        except Exception as e:
            logger.warning("Webhook send error: %s", e)
            return False
