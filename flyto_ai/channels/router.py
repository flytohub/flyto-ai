# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Channel router — dispatches messages to the correct adapter."""
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

from flyto_ai.channels.base import ChannelAdapter, IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

# Type alias for the message handler
MessageHandler = Callable[[IncomingMessage], Coroutine[Any, Any, str]]


class ChannelRouter:
    """Routes incoming messages to the appropriate channel adapter.

    Registers adapters by channel name, parses incoming payloads,
    dispatches to the agent, and sends responses back.

    Usage::

        router = ChannelRouter()
        router.register(TelegramAdapter(bot_token="..."))
        router.register(SlackAdapter(bot_token="..."))

        # Set handler (typically Agent.chat wrapped)
        router.set_handler(my_handler)

        # Handle incoming webhook
        response_text = await router.handle("telegram", raw_payload)
    """

    def __init__(self) -> None:
        self._adapters: Dict[str, ChannelAdapter] = {}
        self._handler: Optional[MessageHandler] = None

    def register(self, adapter: ChannelAdapter) -> None:
        """Register a channel adapter."""
        self._adapters[adapter.channel_name] = adapter
        logger.info("Channel registered: %s", adapter.channel_name)

    def unregister(self, channel_name: str) -> None:
        """Unregister a channel adapter."""
        self._adapters.pop(channel_name, None)

    def set_handler(self, handler: MessageHandler) -> None:
        """Set the message handler (called for each incoming message)."""
        self._handler = handler

    @property
    def channels(self) -> List[str]:
        """List of registered channel names."""
        return sorted(self._adapters.keys())

    def get_adapter(self, channel_name: str) -> Optional[ChannelAdapter]:
        """Get a specific adapter by channel name."""
        return self._adapters.get(channel_name)

    async def handle(
        self,
        channel_name: str,
        raw_payload: Dict[str, Any],
    ) -> Optional[str]:
        """Handle an incoming webhook payload.

        1. Find the adapter for this channel
        2. Parse the payload into IncomingMessage
        3. Call the handler to get a response
        4. Send the response back through the adapter

        Returns the response text, or None if the payload was not a message.
        """
        adapter = self._adapters.get(channel_name)
        if not adapter:
            logger.warning("No adapter for channel: %s", channel_name)
            return None

        # Parse incoming
        incoming = await adapter.parse_incoming(raw_payload)
        if not incoming:
            return None

        if not self._handler:
            logger.warning("No message handler set")
            return None

        # Handle message
        try:
            response_text = await self._handler(incoming)
        except Exception as e:
            logger.warning("Message handler error: %s", e)
            response_text = "An error occurred while processing your message."

        # Send response back
        outgoing = OutgoingMessage(
            text=response_text,
            channel=channel_name,
            user_id=incoming.user_id,
            session_id=incoming.session_id,
            metadata=incoming.metadata,
        )
        await adapter.send(outgoing)

        return response_text

    async def start_all(self) -> None:
        """Start all registered adapters."""
        for adapter in self._adapters.values():
            await adapter.start()

    async def stop_all(self) -> None:
        """Stop all registered adapters."""
        for adapter in self._adapters.values():
            await adapter.stop()
