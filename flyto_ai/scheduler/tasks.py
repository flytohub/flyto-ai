# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Scheduled task definitions."""
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ScheduleType(str, Enum):
    CRON = "cron"
    INTERVAL = "interval"
    ONE_SHOT = "one_shot"


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class TaskSchedule:
    """Schedule definition for a task.

    For cron: uses simplified cron syntax (minute hour day_of_month month day_of_week)
    For interval: seconds between runs
    For one_shot: runs once at the specified time (or immediately if 0)
    """
    type: ScheduleType = ScheduleType.INTERVAL
    interval_seconds: int = 3600          # for interval type
    cron_expression: str = ""             # for cron type (simplified)
    run_at: float = 0.0                   # for one_shot (epoch timestamp, 0=immediate)

    def next_run_time(self, last_run: float = 0.0) -> float:
        """Calculate the next run time based on schedule type."""
        now = time.time()

        if self.type == ScheduleType.ONE_SHOT:
            if self.run_at <= 0:
                return now  # immediate
            return self.run_at

        if self.type == ScheduleType.INTERVAL:
            if last_run <= 0:
                return now  # first run: immediate
            return last_run + self.interval_seconds

        if self.type == ScheduleType.CRON:
            # Simplified: for now just treat as interval
            # Full cron parsing can be added later
            if last_run <= 0:
                return now
            return last_run + 60  # minimum 1 minute for cron

        return now


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    timestamp: float = field(default_factory=time.time)
    ok: bool = True
    message: str = ""
    duration_ms: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None


@dataclass
class ScheduledTask:
    """A scheduled task definition.

    Combines a natural language instruction with a schedule
    and optional budget cap.

    Better than OpenClaw's HEARTBEAT:
    - Independent budget cap per task (OpenClaw burned $50/day on email checks)
    - Auto-converts to blueprint after stable pattern → zero-cost subsequent runs
    """
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    instruction: str = ""           # natural language task description
    schedule: TaskSchedule = field(default_factory=TaskSchedule)
    budget_usd: Optional[float] = None  # per-execution budget cap
    enabled: bool = True
    tags: List[str] = field(default_factory=list)

    # Runtime state
    state: TaskState = TaskState.PENDING
    last_run: float = 0.0
    run_count: int = 0
    success_count: int = 0
    total_cost_usd: float = 0.0
    history: List[TaskResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)."""
        if self.run_count == 0:
            return 0.0
        return self.success_count / self.run_count

    @property
    def is_due(self) -> bool:
        """Check if this task is due to run."""
        if not self.enabled or self.state == TaskState.DISABLED:
            return False
        if self.state == TaskState.RUNNING:
            return False
        next_run = self.schedule.next_run_time(self.last_run)
        return time.time() >= next_run

    def record_result(self, result: TaskResult) -> None:
        """Record an execution result."""
        self.last_run = result.timestamp
        self.run_count += 1
        if result.ok:
            self.success_count += 1
        self.total_cost_usd += result.cost_usd
        self.history.append(result)
        # Keep only last 50 results
        if len(self.history) > 50:
            self.history = self.history[-50:]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledTask":
        """Create a task from a dict (parsed from YAML config)."""
        schedule_data = data.get("schedule", {})
        schedule = TaskSchedule(
            type=ScheduleType(schedule_data.get("type", "interval")),
            interval_seconds=schedule_data.get("interval_seconds", 3600),
            cron_expression=schedule_data.get("cron", ""),
            run_at=schedule_data.get("run_at", 0.0),
        )
        return cls(
            task_id=data.get("task_id", uuid.uuid4().hex[:12]),
            name=data.get("name", ""),
            instruction=data.get("instruction", ""),
            schedule=schedule,
            budget_usd=data.get("budget_usd"),
            enabled=data.get("enabled", True),
            tags=data.get("tags", []),
        )
