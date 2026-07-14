# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Dangerous operation confirmation via Telegram inline keyboard."""
import asyncio
import logging
import time
from typing import Callable, Dict

from flyto_ai.telegram.sender import TelegramSender

logger = logging.getLogger(__name__)

# Timeout for user confirmation (seconds)
CONFIRMATION_TIMEOUT = 120

# Modules that require explicit user confirmation
DANGEROUS_MODULES = frozenset({
    "file.delete", "file.write", "file.move",
    "shell.exec", "process.kill",
    "database.execute", "database.drop",
    "git.push", "git.reset",
    "docker.remove", "docker.stop",
    "ssh.exec",
})

# Entire categories that are dangerous
DANGEROUS_CATEGORIES = frozenset({
    "shell", "process", "ssh", "docker", "k8s",
})


def _is_dangerous(module_id: str) -> bool:
    """Check if a module requires user confirmation."""
    if module_id in DANGEROUS_MODULES:
        return True
    category = module_id.split(".")[0] if "." in module_id else ""
    return category in DANGEROUS_CATEGORIES


class ConfirmationManager:
    """Manages dangerous operation confirmations via Telegram inline keyboard.

    When a dangerous module is about to execute, sends an inline keyboard
    asking the user to approve or deny. Waits up to CONFIRMATION_TIMEOUT
    seconds for a response.
    """

    def __init__(self, sender: TelegramSender) -> None:
        self._sender = sender
        # Pending confirmations: confirmation_id → asyncio.Future
        self._pending: Dict[str, asyncio.Future] = {}

    def make_dispatch_wrapper(self, chat_id: int) -> Callable:
        """Create a dispatch_wrapper that intercepts dangerous module calls.

        Returns a function suitable for Agent.chat(dispatch_wrapper=...).
        """
        def wrapper(original_dispatch: Callable) -> Callable:
            async def wrapped_dispatch(func_name: str, func_args: dict) -> dict:
                if _is_dangerous(func_name):
                    approved = await self._request_confirmation(chat_id, func_name, func_args)
                    if not approved:
                        return {"ok": False, "error": "User denied operation: {}".format(func_name)}
                return await original_dispatch(func_name, func_args)
            return wrapped_dispatch
        return wrapper

    async def _request_confirmation(
        self,
        chat_id: int,
        module_id: str,
        args: dict,
    ) -> bool:
        """Send inline keyboard and wait for user response."""
        confirmation_id = "{}_{}".format(chat_id, int(time.time() * 1000))

        # Build a readable summary of the operation
        args_summary = ""
        for key in ("path", "command", "url", "query", "selector"):
            if key in args:
                val = str(args[key])
                if len(val) > 80:
                    val = val[:80] + "..."
                args_summary = "\n`{}: {}`".format(key, val)
                break

        text = "Dangerous operation:\n`{}`{}".format(module_id, args_summary)

        buttons = [[
            {"text": "Execute", "callback_data": "confirm:{}:approve".format(confirmation_id)},
            {"text": "Cancel", "callback_data": "confirm:{}:deny".format(confirmation_id)},
        ]]

        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[confirmation_id] = future

        await self._sender.send_with_keyboard(chat_id, text, buttons)

        try:
            result = await asyncio.wait_for(future, timeout=CONFIRMATION_TIMEOUT)
            return result
        except asyncio.TimeoutError:
            logger.info("Confirmation %s timed out", confirmation_id)
            await self._sender.send(chat_id, "Operation timed out — cancelled.", parse_mode="")
            return False
        finally:
            self._pending.pop(confirmation_id, None)

    async def handle_callback(self, callback_data: str, callback_query_id: str) -> bool:
        """Handle an incoming callback_query from an inline keyboard button.

        Returns True if the callback was handled (was a confirmation callback).
        """
        if not callback_data.startswith("confirm:"):
            return False

        parts = callback_data.split(":")
        if len(parts) != 3:
            return False

        _, confirmation_id, action = parts
        future = self._pending.get(confirmation_id)

        if future is None or future.done():
            await self._sender.answer_callback(callback_query_id, "Expired or already handled.")
            return True

        approved = action == "approve"
        future.set_result(approved)

        ack_text = "Approved" if approved else "Cancelled"
        await self._sender.answer_callback(callback_query_id, ack_text)
        return True
