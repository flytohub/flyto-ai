# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""CLIBridge — generic per-chat CLI subprocess bridge for Telegram.

Wraps any AI CLI (claude, codex, etc.) as a subprocess with streaming
JSON output. CLI-specific details live in CLIProfile; the bridge itself
is CLI-agnostic.
"""
import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from flyto_ai.telegram.sender import TelegramSender

logger = logging.getLogger(__name__)

# TG API rate limit: ~30 edits/s per chat, stay conservative
_EDIT_INTERVAL = 0.8
# Tail length for status message updates (avoid TG 4096 limit)
_STATUS_TAIL = 3000
# Default query timeout (10 min)
_QUERY_TIMEOUT = 600


# ── CLI Profile ─────────────────────────────────────────────────

@dataclass
class CLIProfile:
    """Describes how to invoke and parse an AI CLI."""

    name: str
    # How to find the binary (resolved via shutil.which)
    command: str
    # Build subprocess args for a query
    # Signature: (prompt, cwd, model, session_id, extra) → list[str]
    build_args: Callable[..., List[str]] = None
    # Parse a single stdout JSON line → (event_type, data)
    # event_type: "text", "tool", "result", "ignore"
    parse_line: Callable[[dict], tuple] = None
    # Extra env vars to inject (e.g. unset CLAUDECODE for nested calls)
    env_overrides: Dict[str, str] = field(default_factory=dict)
    # Valid model names
    valid_models: tuple = ("sonnet", "opus", "haiku")
    # Timeout in seconds
    timeout: int = _QUERY_TIMEOUT


def _claude_build_args(
    prompt: str,
    cwd: str,
    model: Optional[str] = None,
    session_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> List[str]:
    """Build claude CLI args."""
    args = [
        "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
    ]
    if model:
        args.extend(["--model", model])
    if session_id:
        args.extend(["--resume", session_id])
    # MCP config
    mcp_config = (extra or {}).get("mcp_config")
    if mcp_config:
        args.extend(["--mcp-config", mcp_config])
    return args


def _claude_parse_line(data: dict) -> tuple:
    """Parse a claude stream-json line.

    Returns (event_type, payload) where event_type is one of:
    - "text": payload = text string
    - "tool": payload = tool name string
    - "result": payload = {"text": str, "session_id": str, "cost": float}
    - "ignore": payload = None
    """
    msg_type = data.get("type", "")

    if msg_type == "assistant":
        message = data.get("message", {})
        content = message.get("content", [])
        texts = []
        for block in content:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                return ("tool", block.get("name", "?"))
        if texts:
            return ("text", "".join(texts))
        return ("ignore", None)

    if msg_type == "result":
        return ("result", {
            "text": data.get("result", ""),
            "session_id": data.get("session_id", ""),
            "cost": data.get("total_cost_usd", 0) or 0,
        })

    return ("ignore", None)


CLAUDE_PROFILE = CLIProfile(
    name="claude",
    command="claude",
    build_args=_claude_build_args,
    parse_line=_claude_parse_line,
    env_overrides={"CLAUDECODE": ""},  # allow nested invocation
    valid_models=("sonnet", "opus", "haiku"),
)


# ── CLIBridge ───────────────────────────────────────────────────

class CLIBridge:
    """Per-chat CLI subprocess bridge for Telegram.

    Spawns a CLI process per query, streams JSON output back to TG
    by editing a status message in real time. Manages per-chat sessions,
    working directories, and model selection.

    CLI-agnostic — all CLI-specific logic lives in CLIProfile.
    """

    def __init__(
        self,
        sender: "TelegramSender",
        profile: Optional[CLIProfile] = None,
        working_dir: str = ".",
        model: str = "sonnet",
    ) -> None:
        self._sender = sender
        self._profile = profile or CLAUDE_PROFILE
        self._default_cwd = working_dir
        self._default_model = model

        # Per-chat state
        self._sessions: Dict[int, str] = {}       # chat_id → session_id
        self._cwds: Dict[int, str] = {}            # chat_id → working dir
        self._models: Dict[int, str] = {}          # chat_id → model
        self._locks: Dict[int, asyncio.Lock] = {}  # prevent concurrent queries per chat
        self._busy: Dict[int, bool] = {}           # chat_id → currently running
        self._pending_followups: Dict[int, List[str]] = {}
        self._active_procs: Dict[int, asyncio.subprocess.Process] = {}

    # ── Public API ──────────────────────────────────────────────

    def is_busy(self, chat_id: int) -> bool:
        return self._busy.get(chat_id, False)

    def add_followup(self, chat_id: int, text: str) -> None:
        self._pending_followups.setdefault(chat_id, []).append(text)

    async def query(self, chat_id: int, text: str) -> None:
        """Send a message to the CLI and stream the response to TG."""
        lock = self._locks.setdefault(chat_id, asyncio.Lock())

        if lock.locked():
            self.add_followup(chat_id, text)
            await self._sender.send(chat_id, "Queued (busy).", parse_mode="")
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
        """Kill the running CLI process."""
        proc = self._active_procs.get(chat_id)
        if proc and proc.returncode is None:
            proc.terminate()
            await self._sender.send(chat_id, "Interrupted.", parse_mode="")
        else:
            await self._sender.send(chat_id, "Nothing to interrupt.", parse_mode="")

    async def set_cwd(self, chat_id: int, path: str) -> str:
        """Change working directory for next query."""
        resolved = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(resolved):
            return "Not a directory: {}".format(resolved)
        self._cwds[chat_id] = resolved
        self._sessions.pop(chat_id, None)
        return "Working directory: {}".format(resolved)

    async def set_model(self, chat_id: int, model: str) -> str:
        """Change model for this chat."""
        if model not in self._profile.valid_models:
            return "Unknown model. Choose: {}".format(", ".join(self._profile.valid_models))
        self._models[chat_id] = model
        return "Model: {}".format(model)

    async def clear(self, chat_id: int) -> None:
        """Clear session — next message starts fresh."""
        self._sessions.pop(chat_id, None)
        self._cwds.pop(chat_id, None)
        self._models.pop(chat_id, None)
        self._pending_followups.pop(chat_id, None)
        await self._sender.send(chat_id, "Session cleared.", parse_mode="")

    # ── Internal ────────────────────────────────────────────────

    async def _run_query(self, chat_id: int, text: str) -> None:
        """Spawn CLI subprocess, stream output to TG."""
        profile = self._profile

        # Resolve binary
        binary = shutil.which(profile.command)
        if not binary:
            await self._sender.send(
                chat_id,
                "Error: `{}` CLI not found.".format(profile.command),
                parse_mode="",
            )
            return

        # Send status message
        status_msg_id = await self._sender.send(chat_id, "Thinking...", parse_mode="")
        if not status_msg_id:
            return

        cwd = self._cwds.get(chat_id, self._default_cwd)
        model = self._models.get(chat_id, self._default_model)
        session_id = self._sessions.get(chat_id)

        args = profile.build_args(
            prompt=text,
            cwd=cwd,
            model=model,
            session_id=session_id,
        )

        # Build env
        env = os.environ.copy()
        env.update(profile.env_overrides)

        try:
            proc = await asyncio.create_subprocess_exec(
                binary, *args,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            self._active_procs[chat_id] = proc

            final_text, new_session_id, cost = await self._stream_proc(
                proc, chat_id, status_msg_id,
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
                # Check stderr for error info
                stderr_data = await proc.stderr.read()
                err = stderr_data.decode("utf-8", errors="replace").strip() if stderr_data else ""
                if err and proc.returncode != 0:
                    await self._sender.send(chat_id, "Error: {}".format(err[:1000]), parse_mode="")
                else:
                    await self._sender.send(chat_id, "Done (no output).", parse_mode="")

        except asyncio.CancelledError:
            if proc and proc.returncode is None:
                proc.terminate()
            await self._sender.delete_message(chat_id, status_msg_id)
            raise
        except Exception as e:
            logger.exception("CLIBridge query failed for chat %d", chat_id)
            await self._sender.delete_message(chat_id, status_msg_id)
            await self._sender.send(chat_id, "Error: {}".format(e), parse_mode="")
        finally:
            self._active_procs.pop(chat_id, None)

    async def _stream_proc(
        self,
        proc: asyncio.subprocess.Process,
        chat_id: int,
        status_msg_id: int,
    ) -> tuple:
        """Read subprocess stdout line by line, parse JSON, update TG.

        Returns (final_text, session_id, cost_usd).
        """
        profile = self._profile
        buffer = ""
        last_edit_time = 0.0
        session_id = None
        cost = 0.0
        final_text = ""

        try:
            async for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # Plain text fallback (non-JSON CLI)
                    buffer += line + "\n"
                    now = time.monotonic()
                    if now - last_edit_time > _EDIT_INTERVAL:
                        display = buffer[-_STATUS_TAIL:]
                        await self._sender.edit_message(
                            chat_id, status_msg_id, display, parse_mode="",
                        )
                        last_edit_time = now
                    continue

                event_type, payload = profile.parse_line(data)

                if event_type == "text":
                    buffer += payload
                    now = time.monotonic()
                    if now - last_edit_time > _EDIT_INTERVAL and buffer:
                        display = buffer[-_STATUS_TAIL:]
                        await self._sender.edit_message(
                            chat_id, status_msg_id, display, parse_mode="",
                        )
                        last_edit_time = now

                elif event_type == "tool":
                    tool_info = "Running: {}".format(payload)
                    await self._sender.edit_message(
                        chat_id, status_msg_id, tool_info, parse_mode="",
                    )

                elif event_type == "result":
                    final_text = payload.get("text", "") or buffer
                    session_id = payload.get("session_id")
                    cost = payload.get("cost", 0)

        except asyncio.CancelledError:
            raise

        await proc.wait()

        if not final_text:
            final_text = buffer.strip()

        return final_text, session_id, cost


# ── Backward compat alias ──────────────────────────────────────
ClaudeBridge = CLIBridge
