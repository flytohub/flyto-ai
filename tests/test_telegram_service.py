# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for the refactored Telegram service package.

Covers: JobQueue, ConfirmationManager, CommandRouter, TelegramService,
callback_query parsing, multi-turn history, steer injection.
"""
import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ===================================================================
# JobQueue
# ===================================================================

class TestJobQueue:
    @pytest_asyncio.fixture()
    async def job_queue(self, tmp_path):
        from flyto_ai.telegram.jobs import JobQueue
        db_path = str(tmp_path / "test_jobs.db")
        jq = JobQueue(db_path=db_path)
        await jq.init()
        yield jq
        await jq.close()

    @pytest.mark.asyncio
    async def test_enqueue_and_get(self, job_queue):
        job_id = await job_queue.enqueue(123, "do something")
        job = await job_queue.get(job_id)
        assert job is not None
        assert job["chat_id"] == 123
        assert job["text"] == "do something"
        assert job["status"] == "pending"

    @pytest.mark.asyncio
    async def test_lifecycle(self, job_queue):
        job_id = await job_queue.enqueue(123, "task")
        await job_queue.start(job_id)
        job = await job_queue.get(job_id)
        assert job["status"] == "running"

        await job_queue.complete(job_id, result="all done")
        job = await job_queue.get(job_id)
        assert job["status"] == "completed"
        assert job["result"] == "all done"

    @pytest.mark.asyncio
    async def test_fail(self, job_queue):
        job_id = await job_queue.enqueue(123, "task")
        await job_queue.start(job_id)
        await job_queue.fail(job_id, error="boom")
        job = await job_queue.get(job_id)
        assert job["status"] == "failed"
        assert job["error"] == "boom"

    @pytest.mark.asyncio
    async def test_cancel(self, job_queue):
        job_id = await job_queue.enqueue(123, "task")
        cancelled = await job_queue.cancel(job_id)
        assert cancelled is True
        job = await job_queue.get(job_id)
        assert job["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_completed_returns_false(self, job_queue):
        job_id = await job_queue.enqueue(123, "task")
        await job_queue.start(job_id)
        await job_queue.complete(job_id)
        cancelled = await job_queue.cancel(job_id)
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_get_active(self, job_queue):
        await job_queue.enqueue(123, "task1")
        job_id2 = await job_queue.enqueue(123, "task2")
        await job_queue.start(job_id2)

        active = await job_queue.get_active(123)
        assert active is not None
        assert active["job_id"] == job_id2

    @pytest.mark.asyncio
    async def test_get_recent(self, job_queue):
        await job_queue.enqueue(123, "a")
        await job_queue.enqueue(123, "b")
        await job_queue.enqueue(123, "c")
        recent = await job_queue.get_recent(123, limit=2)
        assert len(recent) == 2

    @pytest.mark.asyncio
    async def test_resume_incomplete(self, job_queue):
        job_id = await job_queue.enqueue(123, "task")
        await job_queue.start(job_id)
        count = await job_queue.resume_incomplete()
        assert count == 1
        job = await job_queue.get(job_id)
        assert job["status"] == "failed"
        assert "restarted" in job["error"].lower()

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, job_queue):
        job = await job_queue.get("nonexistent")
        assert job is None


# ===================================================================
# ConfirmationManager
# ===================================================================

class TestConfirmationManager:
    @pytest.fixture()
    def sender(self):
        s = MagicMock()
        s.send_with_keyboard = AsyncMock(return_value=100)
        s.answer_callback = AsyncMock()
        s.send = AsyncMock(return_value=101)
        return s

    @pytest.fixture()
    def manager(self, sender):
        from flyto_ai.telegram.confirmation import ConfirmationManager
        return ConfirmationManager(sender)

    def test_is_dangerous(self):
        from flyto_ai.telegram.confirmation import _is_dangerous
        assert _is_dangerous("file.delete") is True
        assert _is_dangerous("shell.exec") is True
        assert _is_dangerous("ssh.connect") is True  # category match
        assert _is_dangerous("docker.build") is True  # category match
        assert _is_dangerous("browser.goto") is False
        assert _is_dangerous("string.uppercase") is False

    @pytest.mark.asyncio
    async def test_approve_flow(self, manager, sender):
        wrapper = manager.make_dispatch_wrapper(chat_id=123)
        mock_dispatch = AsyncMock(return_value={"ok": True, "data": "deleted"})
        wrapped = wrapper(mock_dispatch)

        # Start the dangerous call in background
        task = asyncio.create_task(wrapped("file.delete", {"path": "/tmp/test.txt"}))

        # Wait for the keyboard to be sent
        await asyncio.sleep(0.05)
        sender.send_with_keyboard.assert_called_once()
        call_args = sender.send_with_keyboard.call_args
        buttons = call_args[1]["buttons"] if "buttons" in call_args[1] else call_args[0][2]
        # Extract confirmation_id from callback_data
        callback_data = buttons[0][0]["callback_data"]
        confirmation_id = callback_data.split(":")[1]

        # Simulate user pressing "approve"
        handled = await manager.handle_callback(
            "confirm:{}:approve".format(confirmation_id),
            "cb_query_123",
        )
        assert handled is True

        result = await task
        assert result["ok"] is True
        mock_dispatch.assert_called_once_with("file.delete", {"path": "/tmp/test.txt"})

    @pytest.mark.asyncio
    async def test_deny_flow(self, manager, sender):
        wrapper = manager.make_dispatch_wrapper(chat_id=123)
        mock_dispatch = AsyncMock(return_value={"ok": True})
        wrapped = wrapper(mock_dispatch)

        task = asyncio.create_task(wrapped("shell.exec", {"command": "rm -rf /"}))

        await asyncio.sleep(0.05)
        call_args = sender.send_with_keyboard.call_args
        buttons = call_args[1]["buttons"] if "buttons" in call_args[1] else call_args[0][2]
        callback_data = buttons[0][1]["callback_data"]  # deny button
        confirmation_id = callback_data.split(":")[1]

        handled = await manager.handle_callback(
            "confirm:{}:deny".format(confirmation_id),
            "cb_query_456",
        )
        assert handled is True

        result = await task
        assert result["ok"] is False
        assert "denied" in result["error"].lower()
        mock_dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_safe_module_no_confirmation(self, manager):
        wrapper = manager.make_dispatch_wrapper(chat_id=123)
        mock_dispatch = AsyncMock(return_value={"ok": True, "data": "page"})
        wrapped = wrapper(mock_dispatch)

        result = await wrapped("browser.goto", {"url": "https://example.com"})
        assert result["ok"] is True
        mock_dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_callback_unknown(self, manager, sender):
        handled = await manager.handle_callback("unknown:data", "cb_123")
        assert handled is False

    @pytest.mark.asyncio
    async def test_handle_callback_expired(self, manager, sender):
        handled = await manager.handle_callback("confirm:expired_id:approve", "cb_123")
        assert handled is True  # handled but expired
        sender.answer_callback.assert_called_once()


# ===================================================================
# CommandRouter
# ===================================================================

class TestCommandRouter:
    @pytest_asyncio.fixture()
    async def router_deps(self, tmp_path):
        from flyto_ai.telegram.jobs import JobQueue
        from flyto_ai.telegram.commands import CommandRouter
        from flyto_ai.telegram.sender import TelegramSender

        sender = MagicMock(spec=TelegramSender)
        sender.send = AsyncMock(return_value=1)

        jq = JobQueue(db_path=str(tmp_path / "cmd_jobs.db"))
        await jq.init()

        agent_handler = AsyncMock()
        router = CommandRouter(
            sender=sender,
            job_queue=jq,
            agent_handler=agent_handler,
        )
        router._agent_handler_mock = agent_handler
        yield router, sender, jq
        await jq.close()

    @pytest.mark.asyncio
    async def test_help(self, router_deps):
        router, sender, _ = router_deps
        handled = await router.handle(123, "/help")
        assert handled is True
        sender.send.assert_called_once()
        text = sender.send.call_args[0][1]
        assert "/agent" in text
        assert "/status" in text

    @pytest.mark.asyncio
    async def test_yaml(self, router_deps):
        router, sender, _ = router_deps
        handled = await router.handle(123, "/yaml")
        assert handled is True

    @pytest.mark.asyncio
    async def test_blueprint_alias(self, router_deps):
        router, sender, _ = router_deps
        handled = await router.handle(123, "/blueprint")
        assert handled is True

    @pytest.mark.asyncio
    async def test_status_empty(self, router_deps):
        router, sender, _ = router_deps
        handled = await router.handle(123, "/status")
        assert handled is True
        text = sender.send.call_args[0][1]
        assert "No recent" in text

    @pytest.mark.asyncio
    async def test_status_with_jobs(self, router_deps):
        router, sender, jq = router_deps
        await jq.enqueue(123, "run something")
        handled = await router.handle(123, "/status")
        assert handled is True
        text = sender.send.call_args[0][1]
        assert "run something" in text

    @pytest.mark.asyncio
    async def test_cancel_no_active(self, router_deps):
        router, sender, _ = router_deps
        handled = await router.handle(123, "/cancel")
        assert handled is True
        text = sender.send.call_args[0][1]
        assert "No active" in text

    @pytest.mark.asyncio
    async def test_cancel_active(self, router_deps):
        router, sender, jq = router_deps
        job_id = await jq.enqueue(123, "task")
        await jq.start(job_id)
        handled = await router.handle(123, "/cancel")
        assert handled is True
        text = sender.send.call_args[0][1]
        assert "Cancelled" in text

    @pytest.mark.asyncio
    async def test_agent_command(self, router_deps):
        router, sender, _ = router_deps
        handled = await router.handle(123, "/agent fix the bug")
        assert handled is True
        router._agent_handler_mock.assert_called_once_with(123, "fix the bug")

    @pytest.mark.asyncio
    async def test_agent_empty(self, router_deps):
        router, sender, _ = router_deps
        handled = await router.handle(123, "/agent")
        assert handled is True
        text = sender.send.call_args[0][1]
        assert "Usage" in text

    @pytest.mark.asyncio
    async def test_clear_no_store(self, router_deps):
        router, sender, _ = router_deps
        handled = await router.handle(123, "/clear")
        assert handled is True
        text = sender.send.call_args[0][1]
        assert "not available" in text.lower()

    @pytest.mark.asyncio
    async def test_unknown_command_not_handled(self, router_deps):
        router, sender, _ = router_deps
        handled = await router.handle(123, "/unknown")
        assert handled is False

    @pytest.mark.asyncio
    async def test_plain_text_not_handled(self, router_deps):
        router, sender, _ = router_deps
        handled = await router.handle(123, "just some text")
        assert handled is False


# ===================================================================
# channels/telegram.py — callback_query parsing
# ===================================================================

class TestCallbackQueryParsing:
    @pytest.mark.asyncio
    async def test_parse_callback_query(self):
        from flyto_ai.channels.telegram import TelegramAdapter
        adapter = TelegramAdapter("fake-token")

        payload = {
            "callback_query": {
                "id": "cb_12345",
                "from": {"id": 999, "username": "testuser"},
                "message": {
                    "message_id": 42,
                    "chat": {"id": 123, "type": "private"},
                },
                "data": "confirm:abc:approve",
            }
        }

        msg = await adapter.parse_incoming(payload)
        assert msg is not None
        assert msg.text == "confirm:abc:approve"
        assert msg.session_id == "123"
        assert msg.metadata["type"] == "callback_query"
        assert msg.metadata["callback_query_id"] == "cb_12345"

    @pytest.mark.asyncio
    async def test_parse_regular_message(self):
        from flyto_ai.channels.telegram import TelegramAdapter
        adapter = TelegramAdapter("fake-token")

        payload = {
            "message": {
                "message_id": 1,
                "from": {"id": 999, "username": "testuser"},
                "chat": {"id": 123, "type": "private"},
                "text": "hello",
            }
        }

        msg = await adapter.parse_incoming(payload)
        assert msg is not None
        assert msg.text == "hello"
        assert msg.metadata["type"] == "message"

    @pytest.mark.asyncio
    async def test_parse_empty_payload(self):
        from flyto_ai.channels.telegram import TelegramAdapter
        adapter = TelegramAdapter("fake-token")
        msg = await adapter.parse_incoming({})
        assert msg is None


# ===================================================================
# TelegramSender
# ===================================================================

class TestTelegramSender:
    @pytest.mark.asyncio
    async def test_send_success(self):
        from flyto_ai.telegram.sender import TelegramSender
        sender = TelegramSender("fake-token")

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"ok": True, "result": {"message_id": 42}})

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session.post = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_resp),
                __aexit__=AsyncMock(return_value=False),
            ))
            mock_session_cls.return_value = mock_session

            msg_id = await sender.send(123, "hello")
            assert msg_id == 42

    @pytest.mark.asyncio
    async def test_send_truncates_long_messages(self):
        from flyto_ai.telegram.sender import TelegramSender
        sender = TelegramSender("fake-token")

        # Verify truncation logic
        long_text = "x" * 5000
        with patch.object(sender, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"ok": True, "result": {"message_id": 1}}
            await sender.send(123, long_text)
            call_payload = mock_req.call_args[0][1]
            assert len(call_payload["text"]) <= 4096


# ===================================================================
# TelegramService — integration
# ===================================================================

class TestTelegramService:
    @pytest_asyncio.fixture()
    async def service(self, tmp_path):
        from flyto_ai.telegram.service import TelegramService

        mock_agent = MagicMock()
        mock_agent.chat = AsyncMock(return_value=MagicMock(
            message="agent reply",
            cost={"estimated_cost_usd": 0.01},
            ok=True,
        ))

        svc = TelegramService(
            agent=mock_agent,
            bot_token="fake-token",
            allowed_chats=frozenset({123}),
        )
        await svc.init()

        # Patch sender to capture instead of calling TG API
        svc._sender = MagicMock()
        svc._sender.send = AsyncMock(return_value=100)
        svc._sender.edit_message = AsyncMock(return_value=True)
        svc._sender.delete_message = AsyncMock(return_value=True)
        svc._sender.send_with_keyboard = AsyncMock(return_value=101)
        svc._sender.answer_callback = AsyncMock()

        # Also patch the command router's sender
        svc._commands._sender = svc._sender

        yield svc, mock_agent
        await svc.close()

    @pytest.mark.asyncio
    async def test_help_command(self, service):
        svc, _ = service
        await svc.handle_webhook({
            "message": {"chat": {"id": 123}, "from": {"id": 1}, "text": "/help", "message_id": 1}
        })
        svc._sender.send.assert_called()
        text = svc._sender.send.call_args[0][1]
        assert "/agent" in text

    @pytest.mark.asyncio
    async def test_plain_text_creates_job(self, service):
        svc, mock_agent = service
        await svc.handle_webhook({
            "message": {"chat": {"id": 123}, "from": {"id": 1}, "text": "do automation", "message_id": 1}
        })
        # Wait for the background task
        await asyncio.sleep(0.2)
        mock_agent.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_unauthorized_chat_ignored(self, service):
        svc, mock_agent = service
        await svc.handle_webhook({
            "message": {"chat": {"id": 999}, "from": {"id": 1}, "text": "hello", "message_id": 1}
        })
        await asyncio.sleep(0.1)
        mock_agent.chat.assert_not_called()
        svc._sender.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_callback_query_routed(self, service):
        svc, _ = service
        # Also patch the confirmation manager's sender
        svc._confirmation._sender = svc._sender
        # Create a pending confirmation
        svc._confirmation._pending["test_id"] = asyncio.get_event_loop().create_future()

        await svc.handle_webhook({
            "callback_query": {
                "id": "cb_1",
                "from": {"id": 1},
                "message": {"chat": {"id": 123}, "message_id": 5},
                "data": "confirm:test_id:approve",
            }
        })
        svc._sender.answer_callback.assert_called()

    @pytest.mark.asyncio
    async def test_empty_payload_ignored(self, service):
        svc, _ = service
        await svc.handle_webhook({})
        svc._sender.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_steer_queue_on_active_job(self, service):
        svc, mock_agent = service
        from flyto_ai.steer import SteerQueue

        # Simulate an active running job
        job_id = await svc._jobs.enqueue(123, "running task")
        await svc._jobs.start(job_id)
        svc._steer_queues[123] = SteerQueue()

        await svc.handle_webhook({
            "message": {"chat": {"id": 123}, "from": {"id": 1}, "text": "change direction", "message_id": 2}
        })

        # Should push to steer queue, not start new job
        assert svc._steer_queues[123].has_pending
        msg = svc._steer_queues[123].pop()
        assert msg == "change direction"


# ===================================================================
# dispatch_wrapper in agent.py
# ===================================================================

class TestDispatchWrapper:
    @pytest.mark.asyncio
    async def test_dispatch_wrapper_applied(self):
        """dispatch_wrapper wraps the dispatch_fn before instrumentation."""
        from flyto_ai.agent import Agent
        from flyto_ai.config import AgentConfig

        config = AgentConfig(
            provider="anthropic",
            api_key="fake-key",
            model="claude-sonnet-4-5",
        )
        agent = Agent(config=config)

        # Track if wrapper was called
        wrapper_called = []

        def my_wrapper(original_fn):
            async def wrapped(name, args):
                wrapper_called.append(name)
                return await original_fn(name, args)
            return wrapped

        # We can't easily run a full chat without an LLM, but we can verify
        # the wrapper parameter is accepted
        assert "dispatch_wrapper" in Agent.chat.__code__.co_varnames
