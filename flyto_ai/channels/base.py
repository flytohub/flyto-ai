# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Channel adapter base class — normalize messaging across platforms."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IncomingMessage:
    """Normalized incoming message from any channel."""
    channel: str         # "telegram" | "discord" | "slack" | "webhook"
    user_id: str         # platform-specific user ID
    text: str            # message text
    session_id: str = "" # conversation/thread ID
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional: images, files, etc.
    attachments: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class OutgoingMessage:
    """Normalized outgoing message to any channel."""
    text: str
    channel: str = ""
    user_id: str = ""
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional: buttons, embeds, etc.
    extras: Dict[str, Any] = field(default_factory=dict)


class ChannelAdapter(ABC):
    """Abstract base class for channel adapters.

    Each adapter normalizes platform-specific messaging into
    IncomingMessage/OutgoingMessage and handles sending responses.
    """

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Unique channel identifier (e.g. 'telegram', 'discord')."""

    @abstractmethod
    async def parse_incoming(self, raw_payload: Dict[str, Any]) -> Optional[IncomingMessage]:
        """Parse a raw webhook/event payload into an IncomingMessage.

        Returns None if the payload is not a user message (e.g. status update).
        """

    @abstractmethod
    async def send(self, message: OutgoingMessage) -> bool:
        """Send a message through this channel. Returns True on success."""

    async def start(self) -> None:
        """Optional: start listening (for long-polling channels)."""

    async def stop(self) -> None:
        """Optional: cleanup resources."""
