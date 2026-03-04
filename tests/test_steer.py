# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for steer mode (mid-execution message injection)."""
import asyncio
import pytest

from flyto_ai.steer import SteerQueue, build_steer_injection


def test_steer_queue_push_pop():
    q = SteerQueue()
    q.push("change direction")
    assert q.has_pending is True
    assert q.pending_count == 1

    msg = q.pop()
    assert msg == "change direction"
    assert q.has_pending is False


def test_steer_queue_pop_empty():
    q = SteerQueue()
    assert q.pop() is None


def test_steer_queue_pop_all():
    q = SteerQueue()
    q.push("msg1")
    q.push("msg2")
    q.push("msg3")

    msgs = q.pop_all()
    assert len(msgs) == 3
    assert q.has_pending is False


def test_steer_queue_max_pending():
    q = SteerQueue(max_pending=3)
    q.push("1")
    q.push("2")
    q.push("3")
    q.push("4")  # oldest should be dropped

    assert q.pending_count == 3
    msgs = q.pop_all()
    assert "4" in msgs  # newest kept
    assert "1" not in msgs  # oldest dropped


def test_steer_queue_clear():
    q = SteerQueue()
    q.push("msg")
    q.clear()
    assert q.has_pending is False
    assert q.pop() is None


@pytest.mark.asyncio
async def test_steer_queue_wait_timeout():
    q = SteerQueue()
    msg = await q.wait_for_message(timeout=0.05)
    assert msg is None


@pytest.mark.asyncio
async def test_steer_queue_wait_receives():
    q = SteerQueue()

    async def push_later():
        await asyncio.sleep(0.02)
        q.push("late message")

    asyncio.create_task(push_later())
    msg = await q.wait_for_message(timeout=0.5)
    assert msg == "late message"


def test_build_steer_injection():
    result = build_steer_injection("focus on results page only")
    assert result["role"] == "user"
    assert "focus on results page only" in result["content"]
    assert "STEERING" in result["content"]
