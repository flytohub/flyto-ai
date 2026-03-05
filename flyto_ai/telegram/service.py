# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""TelegramService — main coordinator for the Telegram bot.

Routes webhooks to commands, confirmation callbacks, Claude Code bridge,
or flyto-ai agent jobs. Claude Code is the default; agent is via /agent.
"""
import asyncio
import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from flyto_ai.channels.telegram import TelegramAdapter
from flyto_ai.models import StreamEvent, StreamEventType
from flyto_ai.steer import SteerQueue, build_steer_injection
from flyto_ai.telegram.commands import CommandRouter
from flyto_ai.telegram.confirmation import ConfirmationManager
from flyto_ai.telegram.jobs import JobQueue
from flyto_ai.telegram.sender import TelegramSender

if TYPE_CHECKING:
    from flyto_ai.agent import Agent
    from flyto_ai.cost import CostTracker
    from flyto_ai.memory.sqlite_store import SQLiteSessionStore
    from flyto_ai.telegram.claude_bridge import CLIBridge

logger = logging.getLogger(__name__)


class TelegramService:
    """Telegram bot service — coordinates all TG subsystems.

    Lifecycle::

        svc = TelegramService(agent=agent, config=config)
        await svc.init()
        # ... handle webhooks ...
        await svc.close()
    """

    def __init__(
        self,
        agent: "Agent",
        bot_token: Optional[str] = None,
        allowed_chats: Optional[frozenset] = None,
        session_store: Optional["SQLiteSessionStore"] = None,
        cost_tracker: Optional["CostTracker"] = None,
        claude_bridge: Optional["CLIBridge"] = None,
        working_dir: Optional[str] = None,
    ) -> None:
        self._agent = agent
        self._bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")

        raw = os.getenv("TELEGRAM_ALLOWED_CHATS", "")
        self._allowed_chats = allowed_chats or (
            frozenset(int(c.strip()) for c in raw.split(",") if c.strip())
            if raw else frozenset()
        )

        self._working_dir = working_dir or os.getcwd()

        # Subsystems (initialized in init())
        self._sender: Optional[TelegramSender] = None
        self._adapter: Optional[TelegramAdapter] = None
        self._jobs: Optional[JobQueue] = None
        self._confirmation: Optional[ConfirmationManager] = None
        self._commands: Optional[CommandRouter] = None
        self._session_store = session_store
        self._cost_tracker = cost_tracker
        self._claude_bridge = claude_bridge

        # Per-chat steer queues: chat_id → SteerQueue
        self._steer_queues: Dict[int, SteerQueue] = {}
        # Per-chat status message IDs: chat_id → msg_id
        self._status_messages: Dict[int, int] = {}
        # Active asyncio tasks: job_id → Task
        self._active_tasks: Dict[str, asyncio.Task] = {}

    async def init(self) -> None:
        """Initialize all subsystems."""
        self._sender = TelegramSender(self._bot_token)
        self._adapter = TelegramAdapter(self._bot_token)
        self._jobs = JobQueue()
        await self._jobs.init()
        await self._jobs.resume_incomplete()

        self._confirmation = ConfirmationManager(self._sender)

        # Initialize CLIBridge with sender if provided
        if self._claude_bridge:
            self._claude_bridge._sender = self._sender

        # Agent handler for /agent command — starts a flyto-ai agent job
        async def _agent_handler(chat_id: int, msg: str) -> None:
            await self._start_new_job(chat_id, msg)

        self._commands = CommandRouter(
            sender=self._sender,
            job_queue=self._jobs,
            session_store=self._session_store,
            cost_tracker=self._cost_tracker,
            claude_bridge=self._claude_bridge,
            agent_handler=_agent_handler,
        )

    async def close(self) -> None:
        """Clean up resources."""
        # Cancel active tasks
        for task in self._active_tasks.values():
            task.cancel()
        if self._jobs:
            await self._jobs.close()

    # ── Webhook entry point ────────────────────────────────────

    async def handle_aiohttp(self, request) -> Any:
        """aiohttp request handler for /telegram webhook."""
        from aiohttp import web

        if not self._bot_token:
            return web.json_response(
                {"ok": False, "error": "TELEGRAM_BOT_TOKEN not set"}, status=503,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": True})

        await self.handle_webhook(body)
        return web.json_response({"ok": True})

    async def handle_webhook(self, body: Dict[str, Any]) -> None:
        """Process a Telegram webhook update."""
        incoming = await self._adapter.parse_incoming(body)
        if not incoming:
            return

        chat_id = int(incoming.session_id) if incoming.session_id else 0
        if not chat_id:
            return

        # Whitelist check
        if self._allowed_chats and chat_id not in self._allowed_chats:
            return

        msg_type = incoming.metadata.get("type", "message")

        # Route callback queries to confirmation manager
        if msg_type == "callback_query":
            callback_query_id = incoming.metadata.get("callback_query_id", "")
            if self._confirmation and callback_query_id:
                await self._confirmation.handle_callback(incoming.text, callback_query_id)
            return

        # Route text messages (with optional attachments)
        await self._handle_message(chat_id, incoming.text, incoming.attachments)

    async def _handle_message(
        self, chat_id: int, text: str, attachments: Optional[List[Dict]] = None,
    ) -> None:
        """Route a text message: command → Claude Code → agent fallback."""
        # 0. Process attachments → rewrite text before routing
        if attachments:
            text = await self._process_attachments(chat_id, text, attachments)
            if text is None:
                return  # error already reported to user

        # 1. Try slash commands
        if text and text.strip().startswith("/") and self._commands:
            handled = await self._commands.handle(chat_id, text)
            if handled:
                return

        if not text:
            return

        # 2. Claude Code is busy → follow-up queue
        if self._claude_bridge and self._claude_bridge.is_busy(chat_id):
            self._claude_bridge.add_followup(chat_id, text)
            await self._sender.send(chat_id, "Queued (Claude is busy).", parse_mode="")
            return

        # 3. Default → Claude Code
        if self._claude_bridge:
            asyncio.create_task(self._claude_bridge.query(chat_id, text))
            return

        # 4. If there's an active agent job, push to steer queue
        active_job = await self._jobs.get_active(chat_id) if self._jobs else None
        if active_job and active_job["status"] == "running":
            queue = self._steer_queues.get(chat_id)
            if queue:
                queue.push(text)
                await self._sender.send(chat_id, "Steering message received.", parse_mode="")
                return

        # 5. Fallback → new agent job
        await self._start_new_job(chat_id, text)

    # ── Attachment processing ──────────────────────────────────

    async def _process_attachments(
        self, chat_id: int, text: str, attachments: List[Dict],
    ) -> Optional[str]:
        """Process photo/voice attachments and return rewritten prompt text.

        Returns None if processing failed (error already sent to user).
        """
        for att in attachments:
            att_type = att.get("type")
            if att_type == "photo":
                result = await self._process_photo(chat_id, att["file_id"], text)
                if result is None:
                    return None
                text = result
            elif att_type == "voice":
                result = await self._process_voice(chat_id, att["file_id"])
                if result is None:
                    return None
                # Append caption text if present
                if text:
                    text = "{}\n\n{}".format(result, text)
                else:
                    text = result
        return text

    async def _process_photo(
        self, chat_id: int, file_id: str, caption: str,
    ) -> Optional[str]:
        """Download photo and build a prompt that tells Claude Code to read it.

        Returns rewritten prompt or None on failure.
        """
        suffix = ".jpg"
        fd, path = tempfile.mkstemp(suffix=suffix, prefix="tg_photo_")
        os.close(fd)

        ok = await self._sender.download_file(file_id, path)
        if not ok:
            await self._sender.send(chat_id, "Failed to download photo.", parse_mode="")
            _safe_remove(path)
            return None

        # Schedule cleanup after 5 minutes
        loop = asyncio.get_event_loop()
        loop.call_later(300, _safe_remove, path)

        user_text = caption or "Describe this image"
        prompt = (
            "The user sent a photo. Read the image at: {}\n\n"
            "User: {}"
        ).format(path, user_text)
        return prompt

    async def _process_voice(self, chat_id: int, file_id: str) -> Optional[str]:
        """Download voice, transcribe via Whisper, return transcribed text.

        Returns transcribed text or None on failure.
        """
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            await self._sender.send(
                chat_id,
                "Voice messages require OPENAI_API_KEY (for Whisper).",
                parse_mode="",
            )
            return None

        fd, path = tempfile.mkstemp(suffix=".ogg", prefix="tg_voice_")
        os.close(fd)

        ok = await self._sender.download_file(file_id, path)
        if not ok:
            await self._sender.send(chat_id, "Failed to download voice.", parse_mode="")
            _safe_remove(path)
            return None

        try:
            transcript = await _transcribe_whisper(path, api_key)
        except Exception as e:
            logger.warning("Whisper transcription failed: %s", e)
            await self._sender.send(
                chat_id, "Voice transcription failed: {}".format(e), parse_mode="",
            )
            return None
        finally:
            _safe_remove(path)

        if not transcript:
            await self._sender.send(chat_id, "Could not transcribe voice.", parse_mode="")
            return None

        # Show user what was heard
        preview = transcript[:200]
        if len(transcript) > 200:
            preview += "..."
        await self._sender.send(chat_id, "Voice: {}".format(preview), parse_mode="")

        return transcript

    async def _start_new_job(self, chat_id: int, text: str) -> None:
        """Enqueue and start a new agent job."""
        job_id = await self._jobs.enqueue(chat_id, text)

        # Send processing indicator
        msg_id = await self._sender.send(chat_id, "Processing...", parse_mode="")
        if msg_id:
            self._status_messages[chat_id] = msg_id

        # Create steer queue for this chat
        self._steer_queues[chat_id] = SteerQueue()

        # Spawn background task
        task = asyncio.create_task(self._run_agent_job(job_id, chat_id, text))
        self._active_tasks[job_id] = task

        def _cleanup(t):
            self._active_tasks.pop(job_id, None)
            self._steer_queues.pop(chat_id, None)
            self._status_messages.pop(chat_id, None)

        task.add_done_callback(_cleanup)

    async def _run_agent_job(self, job_id: str, chat_id: int, text: str) -> None:
        """Execute an agent chat and send the result."""
        await self._jobs.start(job_id)

        tool_count = 0
        tool_names = []

        def on_stream(event: StreamEvent) -> None:
            nonlocal tool_count
            if event.type == StreamEventType.TOOL_START and event.tool_name:
                tool_count += 1
                tool_names.append(event.tool_name)
                # Update status message (fire-and-forget)
                msg_id = self._status_messages.get(chat_id)
                if msg_id and self._sender:
                    status = "Running: {}".format(event.tool_name)
                    asyncio.create_task(
                        self._sender.edit_message(chat_id, msg_id, status, parse_mode="")
                    )

        # Build dispatch wrapper for confirmation
        dispatch_wrapper = None
        if self._confirmation:
            dispatch_wrapper = self._confirmation.make_dispatch_wrapper(chat_id)

        # Build steer-aware dispatch wrapper
        steer_queue = self._steer_queues.get(chat_id)
        if steer_queue and dispatch_wrapper:
            base_wrapper = dispatch_wrapper

            def steer_and_confirm_wrapper(original_dispatch):
                confirmed_dispatch = base_wrapper(original_dispatch)

                async def with_steer(func_name: str, func_args: dict) -> dict:
                    result = await confirmed_dispatch(func_name, func_args)
                    # After each tool call, check steer queue
                    steer_msg = steer_queue.pop()
                    if steer_msg and isinstance(result, dict):
                        result["_steer"] = build_steer_injection(steer_msg)
                    return result
                return with_steer

            dispatch_wrapper = steer_and_confirm_wrapper
        elif steer_queue:
            def steer_only_wrapper(original_dispatch):
                async def with_steer(func_name: str, func_args: dict) -> dict:
                    result = await original_dispatch(func_name, func_args)
                    steer_msg = steer_queue.pop()
                    if steer_msg and isinstance(result, dict):
                        result["_steer"] = build_steer_injection(steer_msg)
                    return result
                return with_steer

            dispatch_wrapper = steer_only_wrapper

        try:
            # Load conversation history
            session_id = "tg_{}".format(chat_id)
            history = []
            if self._session_store:
                stored = await self._session_store.get_messages(session_id, limit=20)
                history = [{"role": m["role"], "content": m["content"]} for m in stored]

            # Save user message
            if self._session_store:
                await self._session_store.add_message(session_id, "user", text)

            result = await self._agent.chat(
                text,
                history=history,
                mode="execute",
                on_stream=on_stream,
                dispatch_wrapper=dispatch_wrapper,
            )

            reply = result.message or "Done."

            # Save assistant response
            if self._session_store:
                await self._session_store.add_message(session_id, "assistant", reply)

            # Build final status
            cost_info = ""
            if result.cost and result.cost.get("estimated_cost_usd"):
                cost_info = ", ${:.4f}".format(result.cost["estimated_cost_usd"])

            # Delete status message and send final reply
            status_msg_id = self._status_messages.get(chat_id)
            if status_msg_id:
                await self._sender.delete_message(chat_id, status_msg_id)

            if tool_count > 0:
                footer = "\n\n({} tools{})".format(tool_count, cost_info)
                reply += footer

            await self._sender.send(chat_id, reply)
            await self._jobs.complete(job_id, result=reply[:500])

        except asyncio.CancelledError:
            await self._jobs.cancel(job_id)
            await self._sender.send(chat_id, "Task cancelled.", parse_mode="")
        except Exception as e:
            logger.exception("Job %s failed", job_id)
            await self._jobs.fail(job_id, error=str(e))
            await self._sender.send(chat_id, "Error: {}".format(e), parse_mode="")


# ── Module-level helpers ──────────────────────────────────────


def _safe_remove(path: str) -> None:
    """Remove a file if it exists, silently ignoring errors."""
    try:
        os.remove(path)
    except OSError:
        pass


async def _transcribe_whisper(file_path: str, api_key: str) -> str:
    """Transcribe an audio file using OpenAI Whisper API. Returns transcript text."""
    import aiohttp

    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": "Bearer {}".format(api_key)}

    data = aiohttp.FormData()
    data.add_field("model", "whisper-1")
    data.add_field(
        "file",
        open(file_path, "rb"),
        filename=os.path.basename(file_path),
        content_type="audio/ogg",
    )

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=data) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError("Whisper API {}: {}".format(resp.status, body[:200]))
            result = await resp.json()
            return result.get("text", "")
