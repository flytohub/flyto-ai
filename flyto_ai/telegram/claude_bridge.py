# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""ClaudeBridge — per-chat Claude Code SDK session manager for Telegram."""
import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from flyto_ai.telegram.confirmation import ConfirmationManager
    from flyto_ai.telegram.sender import TelegramSender

logger = logging.getLogger(__name__)

# TG API rate limit: ~30 edits/s per chat, stay conservative
_EDIT_INTERVAL = 0.8
# Tail length for status message updates (avoid TG 4096 limit)
_STATUS_TAIL = 3000
# Timeout for a single query (10 min)
_QUERY_TIMEOUT = 600
# Dangerous bash patterns that need user confirmation
_DANGEROUS_BASH = ("rm ", "rm\t", "rmdir ", "git push", "git reset", "DROP ", "DELETE FROM ")


class ClaudeBridge:
    """Per-chat Claude Code SDK session manager.

    Each Telegram chat gets its own ClaudeSDKClient instance that maintains
    conversation context across messages. Streams partial output back to TG
    by editing a status message in real time.
    """

    def __init__(
        self,
        sender: "TelegramSender",
        confirmation: Optional["ConfirmationManager"] = None,
        working_dir: str = ".",
        model: str = "sonnet",
        mcp_servers: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._sender = sender
        self._confirmation = confirmation
        self._default_cwd = working_dir
        self._default_model = model
        self._mcp_servers = mcp_servers or self._detect_mcp_servers()

        # Per-chat state
        self._sessions: Dict[int, str] = {}       # chat_id → session_id
        self._cwds: Dict[int, str] = {}            # chat_id → working dir
        self._models: Dict[int, str] = {}          # chat_id → model
        self._locks: Dict[int, asyncio.Lock] = {}  # prevent concurrent queries per chat
        self._busy: Dict[int, bool] = {}           # chat_id → currently running
        self._pending_followups: Dict[int, List[str]] = {}
        self._active_tasks: Dict[int, asyncio.Task] = {}  # for interrupt

    # ── Public API ──────────────────────────────────────────────

    def is_busy(self, chat_id: int) -> bool:
        return self._busy.get(chat_id, False)

    def add_followup(self, chat_id: int, text: str) -> None:
        self._pending_followups.setdefault(chat_id, []).append(text)

    async def query(self, chat_id: int, text: str) -> None:
        """Send a message to Claude Code and stream the response to TG."""
        lock = self._locks.setdefault(chat_id, asyncio.Lock())

        if lock.locked():
            # Already running — queue as follow-up
            self.add_followup(chat_id, text)
            await self._sender.send(chat_id, "Queued (Claude is busy).", parse_mode="")
            return

        async with lock:
            self._busy[chat_id] = True
            try:
                await self._run_query(chat_id, text)
                # Process follow-ups
                while True:
                    followups = self._pending_followups.pop(chat_id, [])
                    if not followups:
                        break
                    for fu in followups:
                        await self._run_query(chat_id, fu)
            finally:
                self._busy[chat_id] = False

    async def interrupt(self, chat_id: int) -> None:
        """Interrupt the running Claude Code query."""
        task = self._active_tasks.get(chat_id)
        if task and not task.done():
            task.cancel()
            await self._sender.send(chat_id, "Interrupted.", parse_mode="")
        else:
            await self._sender.send(chat_id, "Nothing to interrupt.", parse_mode="")

    async def set_cwd(self, chat_id: int, path: str) -> str:
        """Change working directory for next query."""
        import os
        resolved = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(resolved):
            return "Not a directory: {}".format(resolved)
        self._cwds[chat_id] = resolved
        # Clear session so next query picks up new cwd
        self._sessions.pop(chat_id, None)
        return "Working directory: {}".format(resolved)

    async def set_model(self, chat_id: int, model: str) -> str:
        """Change model for this chat."""
        valid = ("sonnet", "opus", "haiku")
        if model not in valid:
            return "Unknown model. Choose: {}".format(", ".join(valid))
        self._models[chat_id] = model
        return "Model: {}".format(model)

    async def clear(self, chat_id: int) -> None:
        """Clear session — next message starts fresh."""
        self._sessions.pop(chat_id, None)
        self._cwds.pop(chat_id, None)
        self._models.pop(chat_id, None)
        self._pending_followups.pop(chat_id, None)
        await self._sender.send(chat_id, "Session cleared.", parse_mode="")

    # ── MCP Discovery ────────────────────────────────────────────

    @staticmethod
    def _detect_mcp_servers() -> Dict[str, Any]:
        """Auto-detect flyto MCP servers in the monorepo."""
        servers: Dict[str, Any] = {}
        # Resolve relative to this file: flyto-ai/flyto_ai/telegram/ → monorepo root
        mono = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

        core_src = os.path.join(mono, "flyto-core", "src")
        if os.path.isfile(os.path.join(core_src, "core", "mcp_server.py")):
            servers["flyto-core"] = {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "core.mcp_server"],
                "cwd": core_src,
            }

        indexer_dir = os.path.join(mono, "flyto-indexer")
        if os.path.isfile(os.path.join(indexer_dir, "src", "mcp_server.py")):
            servers["flyto-indexer"] = {
                "type": "stdio",
                "command": "python",
                "args": ["-m", "src.mcp_server"],
                "cwd": indexer_dir,
            }

        if servers:
            logger.info("Detected MCP servers: %s", list(servers.keys()))
        return servers

    # ── Internal ────────────────────────────────────────────────

    async def _run_query(self, chat_id: int, text: str) -> None:
        """Execute a single query against Claude Code SDK."""
        try:
            from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
        except ImportError:
            await self._sender.send(
                chat_id,
                "Error: claude-agent-sdk not installed.\n`pip install claude-agent-sdk`",
            )
            return

        # Send status message
        status_msg_id = await self._sender.send(chat_id, "Thinking...", parse_mode="")
        if not status_msg_id:
            return

        cwd = self._cwds.get(chat_id, self._default_cwd)
        model = self._models.get(chat_id, self._default_model)
        session_id = self._sessions.get(chat_id)

        can_use_tool = self._make_can_use_tool(chat_id)

        # Auto-allow MCP tools (read-only, safe)
        allowed = ["Read", "Glob", "Grep", "Bash", "Edit", "Write"]
        for name in self._mcp_servers:
            # e.g. mcp__flyto-core__list_modules, mcp__flyto-indexer__find_references
            allowed.append("mcp__{}__*".format(name))

        options = ClaudeAgentOptions(
            cwd=cwd,
            model=model,
            allowed_tools=allowed,
            permission_mode="default",
            can_use_tool=can_use_tool,
            include_partial_messages=True,
            resume=session_id,
            max_budget_usd=5.0,
            mcp_servers=self._mcp_servers if self._mcp_servers else {},
        )

        try:
            async with ClaudeSDKClient(options=options) as client:
                # Wrap in a task so interrupt() can cancel it
                task = asyncio.current_task()
                self._active_tasks[chat_id] = task

                await client.query(text)
                final_text, new_session_id, cost = await self._stream_response(
                    client, chat_id, status_msg_id,
                )

                if new_session_id:
                    self._sessions[chat_id] = new_session_id

                # Delete status message and send final reply
                await self._sender.delete_message(chat_id, status_msg_id)

                if final_text:
                    footer = ""
                    if cost > 0:
                        footer = "\n\n(${:.4f})".format(cost)
                    await self._sender.send_long(chat_id, final_text + footer)
                else:
                    await self._sender.send(chat_id, "Done (no output).", parse_mode="")

        except asyncio.CancelledError:
            await self._sender.delete_message(chat_id, status_msg_id)
            raise
        except Exception as e:
            logger.exception("ClaudeBridge query failed for chat %d", chat_id)
            await self._sender.delete_message(chat_id, status_msg_id)
            await self._sender.send(chat_id, "Error: {}".format(e), parse_mode="")
        finally:
            self._active_tasks.pop(chat_id, None)

    async def _stream_response(
        self,
        client: Any,
        chat_id: int,
        status_msg_id: int,
    ) -> tuple:
        """Consume streaming messages, edit TG status in real time.

        Returns (final_text, session_id, cost_usd).
        """
        from claude_agent_sdk import AssistantMessage, ResultMessage, StreamEvent

        buffer = ""
        last_edit_time = 0.0
        session_id = None
        cost = 0.0
        final_text = ""

        async for msg in client.receive_response():
            if isinstance(msg, StreamEvent):
                event = msg.event
                # Text delta
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    buffer += delta.get("text", "")
                    now = time.monotonic()
                    if now - last_edit_time > _EDIT_INTERVAL and buffer:
                        display = buffer[-_STATUS_TAIL:] if len(buffer) > _STATUS_TAIL else buffer
                        await self._sender.edit_message(
                            chat_id, status_msg_id, display, parse_mode="",
                        )
                        last_edit_time = now

                # Tool use start
                elif event.get("type") == "content_block_start":
                    cb = event.get("content_block", {})
                    if cb.get("type") == "tool_use":
                        tool_info = "Running: {}".format(cb.get("name", "?"))
                        tool_input = cb.get("input", {})
                        # Show key detail
                        for key in ("command", "file_path", "pattern", "query"):
                            if key in tool_input:
                                val = str(tool_input[key])
                                if len(val) > 60:
                                    val = val[:60] + "..."
                                tool_info += "\n`{}`".format(val)
                                break
                        await self._sender.edit_message(
                            chat_id, status_msg_id, tool_info, parse_mode="",
                        )

            elif isinstance(msg, AssistantMessage):
                # Extract text from content blocks
                for block in getattr(msg, "content", []):
                    if hasattr(block, "text"):
                        if buffer:
                            buffer += "\n"
                        buffer += block.text

            elif isinstance(msg, ResultMessage):
                session_id = getattr(msg, "session_id", None)
                cost = getattr(msg, "total_cost_usd", 0) or 0
                final_text = getattr(msg, "result", "") or buffer
                break

        if not final_text:
            final_text = buffer

        return final_text, session_id, cost

    def _make_can_use_tool(self, chat_id: int):
        """Build a can_use_tool callback for this chat."""
        confirmation = self._confirmation

        async def can_use_tool(tool_name: str, input_data: dict, context: Any):
            from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny

            # Check dangerous bash commands
            if tool_name == "Bash":
                cmd = input_data.get("command", "")
                if any(p in cmd for p in _DANGEROUS_BASH):
                    if confirmation:
                        approved = await confirmation._request_confirmation(
                            chat_id, "Bash", {"command": cmd},
                        )
                        if not approved:
                            return PermissionResultDeny(message="User denied.")
                    else:
                        return PermissionResultDeny(message="Dangerous command blocked.")

            return PermissionResultAllow(updated_input=input_data)

        return can_use_tool
