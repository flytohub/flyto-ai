# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Telegram command router — /help /status /cancel /cost /clear /yaml /cd /model /agent."""
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, TYPE_CHECKING

from flyto_ai.telegram.sender import TelegramSender

if TYPE_CHECKING:
    from flyto_ai.telegram.claude_bridge import CLIBridge
    from flyto_ai.telegram.jobs import JobQueue
    from flyto_ai.cost import CostTracker
    from flyto_ai.memory.sqlite_store import SQLiteSessionStore

logger = logging.getLogger(__name__)

_HELP_TEXT = (
    "Available commands:\n"
    "\n"
    "  (plain text) — Claude Code (read/write code)\n"
    "  /agent <msg> — flyto-ai agent (automation)\n"
    "  /cd <path> — change working directory\n"
    "  /model <name> — switch model (sonnet/opus/haiku)\n"
    "  /status — view active tasks\n"
    "  /cancel — interrupt Claude Code\n"
    "  /cost — view spending\n"
    "  /clear — clear session\n"
    "  /yaml — list learned blueprints\n"
    "  /help — show this message"
)


class CommandRouter:
    """Routes slash commands to handlers."""

    def __init__(
        self,
        sender: TelegramSender,
        job_queue: "JobQueue",
        session_store: Optional["SQLiteSessionStore"] = None,
        cost_tracker: Optional["CostTracker"] = None,
        claude_bridge: Optional["CLIBridge"] = None,
        agent_handler: Optional[Callable] = None,
    ) -> None:
        self._sender = sender
        self._jobs = job_queue
        self._session_store = session_store
        self._cost_tracker = cost_tracker
        self._claude_bridge = claude_bridge
        self._agent_handler = agent_handler

    async def handle(self, chat_id: int, text: str) -> bool:
        """Try to handle a slash command. Returns True if handled."""
        stripped = text.strip()
        cmd = stripped.split()[0].lower() if stripped.startswith("/") else ""

        if cmd == "/help":
            await self._sender.send(chat_id, _HELP_TEXT, parse_mode="")
            return True

        if cmd in ("/yaml", "/blueprint"):
            reply = _list_blueprints()
            await self._sender.send(chat_id, reply, parse_mode="")
            return True

        if cmd == "/status":
            await self._handle_status(chat_id)
            return True

        if cmd == "/cancel":
            await self._handle_cancel(chat_id)
            return True

        if cmd == "/cost":
            await self._handle_cost(chat_id)
            return True

        if cmd == "/clear":
            await self._handle_clear(chat_id)
            return True

        if cmd == "/cd":
            path = stripped[3:].strip()
            if not path:
                await self._sender.send(chat_id, "Usage: /cd <path>", parse_mode="")
            elif self._claude_bridge:
                reply = await self._claude_bridge.set_cwd(chat_id, path)
                await self._sender.send(chat_id, reply, parse_mode="")
            else:
                await self._sender.send(chat_id, "Claude Code not configured.", parse_mode="")
            return True

        if cmd == "/model":
            model = stripped[6:].strip().lower()
            if not model:
                await self._sender.send(chat_id, "Usage: /model <sonnet|opus|haiku>", parse_mode="")
            elif self._claude_bridge:
                reply = await self._claude_bridge.set_model(chat_id, model)
                await self._sender.send(chat_id, reply, parse_mode="")
            else:
                await self._sender.send(chat_id, "Claude Code not configured.", parse_mode="")
            return True

        if cmd == "/agent":
            msg = stripped[6:].strip()
            if not msg:
                await self._sender.send(chat_id, "Usage: /agent <message>", parse_mode="")
            elif self._agent_handler:
                await self._agent_handler(chat_id, msg)
            else:
                await self._sender.send(chat_id, "Agent not configured.", parse_mode="")
            return True

        return False

    async def _handle_status(self, chat_id: int) -> None:
        jobs = await self._jobs.get_recent(chat_id, limit=5)
        if not jobs:
            await self._sender.send(chat_id, "No recent tasks.", parse_mode="")
            return

        lines = []
        for j in jobs:
            status_icon = {
                "running": "...",
                "pending": "...",
                "completed": "OK",
                "failed": "ERR",
                "cancelled": "X",
            }.get(j["status"], "?")
            text_preview = j["text"][:40]
            lines.append("[{}] {} — {}".format(status_icon, j["job_id"][:8], text_preview))
        await self._sender.send(chat_id, "\n".join(lines), parse_mode="")

    async def _handle_cancel(self, chat_id: int) -> None:
        # Try Claude Code interrupt first
        if self._claude_bridge and self._claude_bridge.is_busy(chat_id):
            await self._claude_bridge.interrupt(chat_id)
            return

        # Fallback to job cancel
        active = await self._jobs.get_active(chat_id)
        if not active:
            await self._sender.send(chat_id, "No active task to cancel.", parse_mode="")
            return

        cancelled = await self._jobs.cancel(active["job_id"])
        if cancelled:
            await self._sender.send(chat_id, "Cancelled: {}".format(active["job_id"][:8]), parse_mode="")
        else:
            await self._sender.send(chat_id, "Could not cancel — task may have finished.", parse_mode="")

    async def _handle_cost(self, chat_id: int) -> None:
        if not self._cost_tracker:
            await self._sender.send(chat_id, "Cost tracking not available.", parse_mode="")
            return

        ct = self._cost_tracker
        total = getattr(ct, "_session_total_usd", 0.0)
        prompt = getattr(ct, "_total_prompt_tokens", 0)
        completion = getattr(ct, "_total_completion_tokens", 0)
        msg = "Cost: ${:.4f}\nTokens: {:,} in / {:,} out".format(total, prompt, completion)
        await self._sender.send(chat_id, msg, parse_mode="")

    async def _handle_clear(self, chat_id: int) -> None:
        # Clear Claude Code session
        if self._claude_bridge:
            await self._claude_bridge.clear(chat_id)

        # Clear agent conversation history
        session_id = "tg_{}".format(chat_id)
        if self._session_store:
            await self._session_store.delete_session(session_id)
            if not self._claude_bridge:
                await self._sender.send(chat_id, "Conversation history cleared.", parse_mode="")
        elif not self._claude_bridge:
            await self._sender.send(chat_id, "Session store not available.", parse_mode="")


def _list_blueprints() -> str:
    """List top blueprints from flyto-blueprint engine."""
    try:
        from flyto_blueprint import get_engine
        engine = get_engine()
        bps = engine.list_blueprints()
    except Exception as e:
        return "Error loading blueprints: {}".format(e)

    if not bps:
        return "No blueprints yet."

    lines = []
    sorted_bps = sorted(bps, key=lambda b: b.get("score", 0), reverse=True)[:10]
    for bp in sorted_bps:
        name = bp.get("name", "?")
        score = bp.get("score", 0)
        lines.append("  {} (score: {})".format(name, score))
    return "Blueprints:\n" + "\n".join(lines)
