# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for the proactive scheduling system."""
import asyncio
import time
import pytest

from flyto_ai.scheduler.tasks import (
    ScheduledTask, TaskSchedule, TaskResult,
    ScheduleType, TaskState,
)
from flyto_ai.scheduler.engine import Scheduler


# --- TaskSchedule tests ---

def test_schedule_interval_first_run():
    s = TaskSchedule(type=ScheduleType.INTERVAL, interval_seconds=60)
    next_run = s.next_run_time(last_run=0.0)
    assert abs(next_run - time.time()) < 2


def test_schedule_interval_subsequent():
    s = TaskSchedule(type=ScheduleType.INTERVAL, interval_seconds=60)
    last_run = time.time() - 30  # 30 seconds ago
    next_run = s.next_run_time(last_run)
    assert next_run > time.time()  # not yet due


def test_schedule_one_shot_immediate():
    s = TaskSchedule(type=ScheduleType.ONE_SHOT, run_at=0)
    next_run = s.next_run_time()
    assert abs(next_run - time.time()) < 2


def test_schedule_one_shot_future():
    future = time.time() + 3600
    s = TaskSchedule(type=ScheduleType.ONE_SHOT, run_at=future)
    assert s.next_run_time() == future


# --- ScheduledTask tests ---

def test_task_from_dict():
    task = ScheduledTask.from_dict({
        "name": "Check email",
        "instruction": "Check my Gmail inbox",
        "schedule": {"type": "interval", "interval_seconds": 1800},
        "budget_usd": 0.10,
        "tags": ["email"],
    })
    assert task.name == "Check email"
    assert task.schedule.interval_seconds == 1800
    assert task.budget_usd == 0.10


def test_task_is_due():
    task = ScheduledTask(
        name="test",
        instruction="test",
        schedule=TaskSchedule(type=ScheduleType.INTERVAL, interval_seconds=1),
    )
    # First run: immediate
    assert task.is_due is True


def test_task_not_due():
    task = ScheduledTask(
        name="test",
        instruction="test",
        schedule=TaskSchedule(type=ScheduleType.INTERVAL, interval_seconds=3600),
    )
    task.last_run = time.time()  # just ran
    assert task.is_due is False


def test_task_disabled_not_due():
    task = ScheduledTask(
        name="test",
        instruction="test",
        enabled=False,
    )
    assert task.is_due is False


def test_task_record_result():
    task = ScheduledTask(name="test", instruction="test")
    result = TaskResult(task_id=task.task_id, ok=True, cost_usd=0.05)
    task.record_result(result)

    assert task.run_count == 1
    assert task.success_count == 1
    assert task.success_rate == 1.0
    assert task.total_cost_usd == 0.05


def test_task_success_rate():
    task = ScheduledTask(name="test", instruction="test")
    task.record_result(TaskResult(task_id=task.task_id, ok=True))
    task.record_result(TaskResult(task_id=task.task_id, ok=False))
    task.record_result(TaskResult(task_id=task.task_id, ok=True))
    assert task.success_rate == pytest.approx(2/3)


def test_task_history_limit():
    task = ScheduledTask(name="test", instruction="test")
    for i in range(60):
        task.record_result(TaskResult(task_id=task.task_id, ok=True))
    assert len(task.history) == 50  # capped


# --- Scheduler tests ---

def test_scheduler_add_remove():
    s = Scheduler()
    task = ScheduledTask(name="test", instruction="test")
    tid = s.add_task(task)
    assert s.task_count == 1
    assert s.remove_task(tid) is True
    assert s.task_count == 0


def test_scheduler_enable_disable():
    s = Scheduler()
    task = ScheduledTask(name="test", instruction="test")
    s.add_task(task)
    s.disable_task(task.task_id)
    assert task.enabled is False
    assert task.state == TaskState.DISABLED
    s.enable_task(task.task_id)
    assert task.enabled is True


@pytest.mark.asyncio
async def test_scheduler_run_once():
    results = []

    async def executor(instruction):
        results.append(instruction)
        return {"ok": True, "message": "done", "cost_usd": 0.01}

    s = Scheduler(executor=executor)
    task = ScheduledTask(
        name="test",
        instruction="do something",
        schedule=TaskSchedule(type=ScheduleType.ONE_SHOT),
    )
    s.add_task(task)

    task_results = await s.run_once()
    assert len(task_results) == 1
    assert task_results[0].ok is True
    assert "do something" in results


@pytest.mark.asyncio
async def test_scheduler_one_shot_disables():
    async def executor(instruction):
        return {"ok": True}

    s = Scheduler(executor=executor)
    task = ScheduledTask(
        name="one-shot",
        instruction="test",
        schedule=TaskSchedule(type=ScheduleType.ONE_SHOT),
    )
    s.add_task(task)

    await s.run_once()
    assert task.enabled is False
    assert task.state == TaskState.DISABLED


@pytest.mark.asyncio
async def test_scheduler_executor_error():
    async def executor(instruction):
        raise RuntimeError("something broke")

    s = Scheduler(executor=executor)
    task = ScheduledTask(name="test", instruction="fail")
    s.add_task(task)

    results = await s.run_once()
    assert len(results) == 1
    assert results[0].ok is False
    assert "broke" in results[0].error


@pytest.mark.asyncio
async def test_scheduler_no_executor():
    s = Scheduler()
    task = ScheduledTask(name="test", instruction="fail")
    s.add_task(task)

    results = await s.run_once()
    assert results[0].ok is False


def test_scheduler_summary():
    s = Scheduler()
    s.add_task(ScheduledTask(name="t1", instruction="test1"))
    s.add_task(ScheduledTask(name="t2", instruction="test2"))
    summary = s.summary()
    assert summary["task_count"] == 2
    assert len(summary["tasks"]) == 2


@pytest.mark.asyncio
async def test_scheduler_start_stop():
    async def executor(instruction):
        return {"ok": True}

    s = Scheduler(executor=executor, check_interval=0.1)
    await s.start()
    assert s.running is True
    await asyncio.sleep(0.05)
    await s.stop()
    assert s.running is False
