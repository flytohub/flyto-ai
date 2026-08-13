# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Validated scheduled-task definitions and deterministic UTC slot calculation."""
from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Set

MAX_NAME = 200
MAX_INSTRUCTION = 16_384
MAX_TASK_ID = 128
MAX_TAGS = 32
MAX_TAG = 64
MAX_INTERVAL = 366 * 24 * 3600
MAX_RUN_AT = 253402300799.0
MAX_BUDGET_USD = 1_000_000.0
MAX_RESULT_TEXT = 500
MAX_HISTORY = 50
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


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


def _number(value: Any, name: str, *, maximum: float, zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < (0 if zero else 1) or result > maximum:
        raise ValueError(f"{name} is out of range")
    return result


def _integer(value: Any, name: str, *, maximum: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise ValueError(f"{name} must be an integer in range")
    return value


def _text(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or any(
        ord(char) < 32 and char not in "\t\n\r" for char in value
    ):
        raise ValueError(f"{name} is invalid")
    return value


def _cron_values(field: str, low: int, high: int, *, sunday: bool = False) -> Set[int]:
    if not field or len(field) > 128:
        raise ValueError("invalid cron field")
    values: Set[int] = set()
    for part in field.split(","):
        if not part:
            raise ValueError("invalid cron list")
        base, slash, step_text = part.partition("/")
        if slash:
            if not step_text.isdigit() or int(step_text) < 1 or int(step_text) > high - low + 1:
                raise ValueError("invalid cron step")
            step = int(step_text)
        else:
            step = 1
        if base == "*":
            start, end = low, high
        elif "-" in base:
            pieces = base.split("-")
            if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
                raise ValueError("invalid cron range")
            start, end = map(int, pieces)
        elif base.isdigit() and not slash:
            start = end = int(base)
        else:
            raise ValueError("unsupported cron syntax")
        if start < low or end > high or start > end:
            raise ValueError("cron value out of range")
        values.update(range(start, end + 1, step))
    if sunday and 7 in values:
        values.remove(7)
        values.add(0)
    return values


@dataclass
class TaskSchedule:
    """A strict interval, one-shot, or five-field UTC cron definition.

    Cron fields accept only ``*``, integers, comma lists, inclusive ranges,
    and positive steps on a wildcard or range.
    """

    type: ScheduleType = ScheduleType.INTERVAL
    interval_seconds: int = 3600
    cron_expression: str = ""
    run_at: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.type, ScheduleType):
            try:
                self.type = ScheduleType(self.type)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid schedule type") from exc
        if self.type == ScheduleType.INTERVAL:
            self.interval_seconds = _integer(
                self.interval_seconds, "interval_seconds", maximum=MAX_INTERVAL
            )
            if self.cron_expression or self.run_at != 0:
                raise ValueError("interval schedule has incompatible fields")
        elif self.type == ScheduleType.ONE_SHOT:
            self.run_at = _number(self.run_at, "run_at", maximum=MAX_RUN_AT)
            if self.cron_expression or self.interval_seconds != 3600:
                raise ValueError("one-shot schedule has incompatible fields")
        else:
            if not isinstance(self.cron_expression, str) or len(self.cron_expression) > 640:
                raise ValueError("invalid cron expression")
            _, _, day, month, weekday, day_restricted, weekday_restricted = self._cron_parts()
            if not self._calendar_satisfiable(
                day, month, weekday, day_restricted, weekday_restricted
            ):
                raise ValueError("cron is unsatisfiable within the Gregorian calendar cycle")
            if self.interval_seconds != 3600 or self.run_at != 0:
                raise ValueError("cron schedule has incompatible fields")

    def _cron_parts(self) -> tuple[Set[int], Set[int], Set[int], Set[int], Set[int], bool, bool]:
        fields = self.cron_expression.split()
        if len(fields) != 5:
            raise ValueError("cron must contain exactly five fields")
        minute = _cron_values(fields[0], 0, 59)
        hour = _cron_values(fields[1], 0, 23)
        day = _cron_values(fields[2], 1, 31)
        month = _cron_values(fields[3], 1, 12)
        weekday = _cron_values(fields[4], 0, 7, sunday=True)
        return (
            minute, hour, day, month, weekday,
            day != set(range(1, 32)),
            weekday != set(range(7)),
        )

    def _cron_sets(self) -> tuple[Set[int], ...]:
        return self._cron_parts()[:5]

    @staticmethod
    def _calendar_satisfiable(
        day: Set[int],
        month: Set[int],
        weekday: Set[int],
        day_restricted: bool,
        weekday_restricted: bool,
    ) -> bool:
        """Check exactly one finite 400-year Gregorian calendar cycle."""
        candidate = datetime(2000, 1, 1, tzinfo=timezone.utc)
        for _ in range(146_097):
            cron_weekday = (candidate.weekday() + 1) % 7
            day_match = candidate.day in day
            weekday_match = cron_weekday in weekday
            calendar_match = (
                day_match or weekday_match
                if day_restricted and weekday_restricted
                else day_match and weekday_match
            )
            if candidate.month in month and calendar_match:
                return True
            candidate += timedelta(days=1)
        return False

    def next_slot(self, cursor: float, *, now: Optional[float] = None) -> float:
        """Return the next deterministic slot; ``now`` is only the initial anchor."""
        current = time.time() if now is None else _number(now, "now", maximum=MAX_RUN_AT)
        cursor = _number(cursor, "cursor", maximum=MAX_RUN_AT)
        if self.type == ScheduleType.ONE_SHOT:
            slot = self.run_at or current
            return slot if cursor < slot else math.inf
        if self.type == ScheduleType.INTERVAL:
            return current if cursor == 0 else cursor + self.interval_seconds
        minute, hour, day, month, weekday, day_restricted, weekday_restricted = (
            self._cron_parts()
        )
        anchor = current if cursor == 0 else cursor
        stamp = int(anchor) // 60 * 60
        stamp += 0 if cursor == 0 and anchor == stamp else 60
        candidate = datetime.fromtimestamp(stamp, timezone.utc)
        # Every Gregorian date/weekday combination repeats within 400 years.
        # Searching one complete cycle is finite and proves an absent slot unsatisfiable.
        for _ in range(146_097 + 1):
            value = candidate
            cron_weekday = (value.weekday() + 1) % 7
            day_match = value.day in day
            weekday_match = cron_weekday in weekday
            calendar_match = (
                day_match or weekday_match
                if day_restricted and weekday_restricted
                else day_match and weekday_match
            )
            if value.month in month and calendar_match:
                day_start = int(value.replace(hour=0, minute=0, second=0).timestamp())
                for allowed_hour in sorted(hour):
                    for allowed_minute in sorted(minute):
                        slot = day_start + allowed_hour * 3600 + allowed_minute * 60
                        if slot >= stamp:
                            return float(slot)
            try:
                candidate = datetime.fromtimestamp(
                    int(value.replace(hour=0, minute=0, second=0).timestamp()) + 86_400,
                    timezone.utc,
                )
            except (OverflowError, OSError, ValueError) as exc:
                raise ValueError("cron search exceeds the supported UTC range") from exc
            stamp = int(candidate.timestamp())
        raise ValueError("cron is unsatisfiable within the Gregorian calendar cycle")

    def next_run_time(self, last_run: float = 0.0) -> float:
        return self.next_slot(last_run)

    def to_definition(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"type": self.type.value}
        if self.type == ScheduleType.INTERVAL:
            result["interval_seconds"] = self.interval_seconds
        elif self.type == ScheduleType.CRON:
            result["cron"] = self.cron_expression
        else:
            result["run_at"] = self.run_at
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSchedule":
        if not isinstance(data, Mapping):
            raise ValueError("schedule must be a mapping")
        try:
            kind = ScheduleType(data.get("type", "interval"))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid schedule type") from exc
        allowed = {
            ScheduleType.INTERVAL: {"type", "interval_seconds"},
            ScheduleType.CRON: {"type", "cron"},
            ScheduleType.ONE_SHOT: {"type", "run_at"},
        }[kind]
        if set(data) - allowed:
            raise ValueError("unknown or incompatible schedule fields")
        kwargs: Dict[str, Any] = {"type": kind}
        if kind == ScheduleType.INTERVAL:
            kwargs["interval_seconds"] = data.get("interval_seconds", 3600)
        elif kind == ScheduleType.CRON:
            kwargs["cron_expression"] = data.get("cron", "")
        else:
            kwargs["run_at"] = data.get("run_at", 0.0)
        return cls(**kwargs)


@dataclass
class TaskResult:
    task_id: str
    timestamp: float = field(default_factory=time.time)
    ok: bool = True
    message: str = ""
    duration_ms: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None


@dataclass
class ScheduledTask:
    """A public immutable definition plus ephemeral compatibility counters."""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    instruction: str = ""
    schedule: TaskSchedule = field(default_factory=TaskSchedule)
    budget_usd: Optional[float] = None
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    state: TaskState = TaskState.PENDING
    last_run: float = 0.0
    run_count: int = 0
    success_count: int = 0
    total_cost_usd: float = 0.0
    history: List[TaskResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.task_id = _text(self.task_id, "task_id", MAX_TASK_ID)
        if not _TOKEN.fullmatch(self.task_id):
            raise ValueError("task_id must be a safe token")
        self.name = _text(self.name, "name", MAX_NAME)
        self.instruction = _text(self.instruction, "instruction", MAX_INSTRUCTION)
        if not isinstance(self.schedule, TaskSchedule):
            raise ValueError("schedule must be a TaskSchedule")
        if self.budget_usd is not None:
            self.budget_usd = _number(
                self.budget_usd, "budget_usd", maximum=MAX_BUDGET_USD
            )
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if not isinstance(self.tags, list) or len(self.tags) > MAX_TAGS:
            raise ValueError("tags are invalid")
        checked = [_text(tag, "tag", MAX_TAG) for tag in self.tags]
        if len(set(checked)) != len(checked) or any(not _TOKEN.fullmatch(tag) for tag in checked):
            raise ValueError("tags must be unique safe tokens")
        self.tags = checked

    @property
    def success_rate(self) -> float:
        return self.success_count / self.run_count if self.run_count else 0.0

    @property
    def is_due(self) -> bool:
        now = time.time()
        return bool(self.enabled and self.state not in (TaskState.DISABLED, TaskState.RUNNING)
                    and now >= self.schedule.next_slot(self.last_run, now=now))

    def record_result(self, result: TaskResult) -> None:
        self.last_run = result.timestamp
        self.run_count += 1
        self.success_count += int(result.ok)
        self.total_cost_usd += result.cost_usd
        self.history = (self.history + [result])[-MAX_HISTORY:]

    def to_definition(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "name": self.name,
                "instruction": self.instruction, "schedule": self.schedule.to_definition(),
                "budget_usd": self.budget_usd, "enabled": self.enabled, "tags": list(self.tags)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScheduledTask":
        if not isinstance(data, Mapping):
            raise ValueError("task must be a mapping")
        allowed = {"task_id", "name", "instruction", "schedule", "budget_usd", "enabled", "tags"}
        if set(data) - allowed:
            raise ValueError("unknown or server-owned task fields")
        return cls(task_id=data.get("task_id", uuid.uuid4().hex[:12]),
                   name=data.get("name", ""), instruction=data.get("instruction", ""),
                   schedule=TaskSchedule.from_dict(data.get("schedule", {})),
                   budget_usd=data.get("budget_usd"), enabled=data.get("enabled", True),
                   tags=list(data.get("tags", [])) if isinstance(data.get("tags", []), list) else data.get("tags"))
