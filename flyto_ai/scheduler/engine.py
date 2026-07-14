# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Scheduler engine — runs scheduled tasks using AsyncIO."""
import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

from flyto_ai.scheduler.tasks import ScheduledTask, TaskResult, TaskState, ScheduleType

logger = logging.getLogger(__name__)

# Type for the task executor function
TaskExecutor = Callable[[str], Coroutine[Any, Any, Dict[str, Any]]]


class Scheduler:
    """AsyncIO-based task scheduler.

    Runs scheduled tasks at configured intervals, with per-task
    budget caps and result tracking.

    Usage::

        scheduler = Scheduler(executor=my_agent_chat)
        scheduler.add_task(ScheduledTask(
            name="Check emails",
            instruction="Check my Gmail for urgent emails",
            schedule=TaskSchedule(type=ScheduleType.INTERVAL, interval_seconds=1800),
            budget_usd=0.10,
        ))
        await scheduler.start()
    """

    def __init__(
        self,
        executor: Optional[TaskExecutor] = None,
        check_interval: float = 10.0,
    ) -> None:
        self._executor = executor
        self._check_interval = check_interval
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    @property
    def running(self) -> bool:
        return self._running

    def add_task(self, task: ScheduledTask) -> str:
        """Add a scheduled task. Returns the task_id."""
        self._tasks[task.task_id] = task
        logger.info("Scheduled task added: %s (%s)", task.name, task.task_id)
        return task.task_id

    def remove_task(self, task_id: str) -> bool:
        """Remove a scheduled task. Returns True if found."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[ScheduledTask]:
        return list(self._tasks.values())

    def enable_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task:
            task.enabled = True
            task.state = TaskState.PENDING
            return True
        return False

    def disable_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task:
            task.enabled = False
            task.state = TaskState.DISABLED
            return True
        return False

    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started (%d tasks, check every %.0fs)",
                     self.task_count, self._check_interval)

    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        logger.info("Scheduler stopped")

    async def run_once(self) -> List[TaskResult]:
        """Check and execute all due tasks once. Returns results."""
        results = []
        for task in list(self._tasks.values()):
            if task.is_due:
                result = await self._execute_task(task)
                results.append(result)
        return results

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self.run_once()
            except Exception as e:
                logger.warning("Scheduler loop error: %s", e)

            try:
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break

    async def _execute_task(self, task: ScheduledTask) -> TaskResult:
        """Execute a single scheduled task."""
        task.state = TaskState.RUNNING
        t0 = time.monotonic()

        logger.info("Running scheduled task: %s (%s)", task.name, task.task_id)

        try:
            if not self._executor:
                raise RuntimeError("No executor configured")

            # Execute the task instruction
            result_data = await self._executor(task.instruction)
            duration_ms = int((time.monotonic() - t0) * 1000)

            ok = result_data.get("ok", True)
            cost = result_data.get("cost_usd", 0.0)

            # Budget check
            if task.budget_usd and cost > task.budget_usd:
                logger.warning(
                    "Task %s exceeded budget: $%.4f / $%.2f",
                    task.task_id, cost, task.budget_usd,
                )

            result = TaskResult(
                task_id=task.task_id,
                ok=ok,
                message=result_data.get("message", "")[:500],
                duration_ms=duration_ms,
                cost_usd=cost,
                error=result_data.get("error"),
            )

            task.state = TaskState.COMPLETED if ok else TaskState.FAILED
            task.record_result(result)

            # One-shot tasks: disable after execution
            if task.schedule.type == ScheduleType.ONE_SHOT:
                task.enabled = False
                task.state = TaskState.DISABLED

            return result

        except Exception as e:
            duration_ms = int((time.monotonic() - t0) * 1000)
            result = TaskResult(
                task_id=task.task_id,
                ok=False,
                duration_ms=duration_ms,
                error=str(e)[:500],
            )
            task.state = TaskState.FAILED
            task.record_result(result)
            logger.warning("Task %s failed: %s", task.task_id, str(e)[:200])
            return result

    def summary(self) -> Dict[str, Any]:
        """Return scheduler summary."""
        return {
            "running": self._running,
            "task_count": self.task_count,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "name": t.name,
                    "state": t.state.value,
                    "enabled": t.enabled,
                    "run_count": t.run_count,
                    "success_rate": round(t.success_rate, 2),
                    "total_cost_usd": round(t.total_cost_usd, 4),
                }
                for t in self._tasks.values()
            ],
        }
