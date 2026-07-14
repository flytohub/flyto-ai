# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Steer mode — inject user messages mid-execution.

Allows users to redirect or refine the agent's behavior during
a multi-tool execution, without waiting for completion.
"""
import asyncio
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class SteerQueue:
    """Thread-safe message queue for mid-execution steering.

    The agent checks this queue between tool calls. If a message
    is waiting, it gets injected into the conversation so the LLM
    can adjust its behavior.

    Usage::

        queue = SteerQueue()

        # User side (e.g., WebSocket handler):
        queue.push("Actually, only search the first 3 results")

        # Agent side (between tool calls):
        steer_msg = queue.pop()
        if steer_msg:
            messages.append({"role": "user", "content": steer_msg})
    """

    def __init__(self, max_pending: int = 10) -> None:
        self._queue: deque = deque(maxlen=max_pending)
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()

    def push(self, message: str) -> None:
        """Push a steering message (from user/external source)."""
        self._queue.append(message)
        self._event.set()
        logger.info("Steer message queued: %s", message[:80])

    def pop(self) -> Optional[str]:
        """Pop the next steering message, or None if empty."""
        try:
            msg = self._queue.popleft()
            if not self._queue:
                self._event.clear()
            return msg
        except IndexError:
            return None

    def pop_all(self) -> list:
        """Pop all pending steering messages."""
        msgs = list(self._queue)
        self._queue.clear()
        self._event.clear()
        return msgs

    @property
    def has_pending(self) -> bool:
        """Check if there are pending steering messages."""
        return len(self._queue) > 0

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    async def wait_for_message(self, timeout: float = 0.1) -> Optional[str]:
        """Wait for a steering message with timeout.

        Returns the message if one arrives within timeout, else None.
        Used by the agent between tool calls for non-blocking check.
        """
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return self.pop()
        except asyncio.TimeoutError:
            return None

    def clear(self) -> None:
        """Clear all pending messages."""
        self._queue.clear()
        self._event.clear()


def build_steer_injection(message: str) -> dict:
    """Build a user message dict for injecting a steer message into conversation."""
    return {
        "role": "user",
        "content": (
            "[USER STEERING — mid-execution redirect]\n"
            "{}\n"
            "[Continue with the adjusted instructions above. "
            "Do NOT restart from scratch — adapt your current approach.]"
        ).format(message),
    }
