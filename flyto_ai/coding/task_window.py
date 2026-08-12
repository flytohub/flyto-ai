# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Read-only, secret-free host task window for many coding frontends.

The window joins two durable facts without creating a second scheduler:
MissionStore owns main-axis/branch/order state, while coding job records own the
implementation/audit lifecycle and repository-set lease digests.  Raw prompts,
paths, provider session ids, evidence and worker identities never cross this
surface.  It is a local operator view, not a fourth MCP tool and not authority
for any mutation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from flyto_ai.coding.mission_runtime import CodingMissionRuntime


TASK_WINDOW_SCHEMA = "flyto.coding-task-window.v1"
MAX_TASK_WINDOW_ITEMS = 200
MAX_TASK_WINDOW_SCAN = 5_000
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_JOB = re.compile(r"^job_[a-f0-9]{24}$")
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_TERMINAL = frozenset({"completed", "failed", "codex_accepted"})
_STATE_ORDER = {
    "running": 0,
    "rework_running": 1,
    "awaiting_codex_audit": 2,
    "queued": 3,
    "codex_accepted": 4,
    "completed": 5,
    "failed": 6,
}


class TaskWindowCorrupt(RuntimeError):
    """Durable task state could not be projected without guessing."""


def _safe(value: Any, *, default: str = "") -> str:
    if value in (None, ""):
        return default
    if not isinstance(value, str) or not _SAFE.fullmatch(value):
        raise TaskWindowCorrupt("task window state contains an unsafe identifier")
    return value


def _bounded_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TaskWindowCorrupt("task window state contains an invalid counter")
    return value


def _mission_scope(value: Any) -> str:
    """Carry the kernel's already-public bounded scope token."""

    if value in (None, ""):
        return ""
    if (
        not isinstance(value, str)
        or len(value) > 256
        or not value.isprintable()
        or any(character.isspace() for character in value)
    ):
        raise TaskWindowCorrupt("task window state contains an invalid mission scope")
    return value


def _bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TaskWindowCorrupt("task window state contains an invalid boolean")
    return value


def _projection(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("mission")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TaskWindowCorrupt("task window state contains an invalid mission projection")
    return value


def _repository_digests(record: Mapping[str, Any]) -> List[str]:
    value = record.get("repository_digests")
    if value is None:
        value = [record.get("workspace_sha256")]
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 16
        or any(not isinstance(item, str) or not _DIGEST.fullmatch(item) for item in value)
    ):
        raise TaskWindowCorrupt("task window state contains invalid repository digests")
    return sorted(set(value))


def _job_projection(
    record: Mapping[str, Any],
    *,
    work_items: Mapping[str, Mapping[str, Any]],
    scheduler_ranks: Mapping[str, int],
) -> Dict[str, Any]:
    job_id = record.get("job_id")
    if not isinstance(job_id, str) or not _JOB.fullmatch(job_id):
        raise TaskWindowCorrupt("task window state contains an invalid job id")
    state = _safe(record.get("state"))
    mission = _projection(record)
    work_item_id = _safe(mission.get("work_item_id"))
    work = work_items.get(work_item_id, {})
    owner_ref = record.get("owner_ref") or "unassigned"
    owner_ref = _safe(owner_ref)
    updated_at = record.get("updated_at", 0)
    if isinstance(updated_at, bool) or not isinstance(updated_at, (int, float)):
        raise TaskWindowCorrupt("task window state contains an invalid timestamp")
    return {
        "job_id": job_id,
        "owner_ref": owner_ref,
        "state": state,
        "terminal": state in _TERMINAL,
        "mission_id": _safe(mission.get("mission_id")),
        "mission_status": _safe(mission.get("mission_status")),
        "main_axis_sha256": _safe(mission.get("main_axis_sha256")),
        "work_item_id": work_item_id,
        "scope": _mission_scope(work.get("scope", mission.get("scope"))),
        "lane": _safe(work.get("lane", mission.get("lane"))),
        "priority": _bounded_int(work.get("priority", mission.get("priority", 0))),
        "scheduler_rank": scheduler_ranks.get(work_item_id),
        "parent_id": _safe(mission.get("parent_id")),
        "return_to_id": _safe(mission.get("return_to_id")),
        "returned_to_main_axis": _bool(mission.get("returned_to_main_axis", False)),
        "repository_digests": _repository_digests(record),
        "implementation_backend": _safe(record.get("implementation_backend")),
        "implementation_session_bound": bool(record.get("implementation_session_id")),
        "implementer_started": _bool(record.get("implementer_started", False)),
        "audit_count": _bounded_int(record.get("audit_count", 0)),
        "rework_count": _bounded_int(record.get("rework_count", 0)),
        "failure_code": _safe(record.get("failure_code")),
        "landable": _bool(record.get("landable", False)),
        "updated_at": float(updated_at),
    }


def read_task_window(state_root: Any, *, limit: int = 50) -> Dict[str, Any]:
    """Return one bounded local snapshot without starting a coding service."""

    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_TASK_WINDOW_ITEMS:
        raise ValueError("task window limit must be between 1 and 200")
    root = Path(os.path.abspath(os.path.expanduser(str(state_root))))
    runtime = CodingMissionRuntime(root, worker="task-window-reader")
    # Current job records carry their own mission projection, so a long history
    # cannot make the newest main axis disappear when the generic store's
    # bounded work-item enrichment snapshot contains only older rows.
    fleet = runtime.fleet(limit=MAX_TASK_WINDOW_ITEMS)
    scheduler_order = runtime.scheduler_order(limit=MAX_TASK_WINDOW_ITEMS)
    scheduler_ranks = {item: index + 1 for index, item in enumerate(scheduler_order)}
    work_items = {
        str(item.get("work_item_id") or ""): item
        for item in fleet.get("work_items", [])
        if isinstance(item, Mapping)
    }

    paths: List[Path] = []
    tenants = root / "tenants"
    try:
        for tenant in sorted(tenants.iterdir()) if tenants.is_dir() else ():
            jobs = tenant / "jobs"
            if jobs.is_dir():
                paths.extend(sorted(jobs.glob("*.json")))
    except OSError as exc:
        raise TaskWindowCorrupt("task window state could not be read") from exc
    if len(paths) > MAX_TASK_WINDOW_SCAN:
        try:
            paths = sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)[
                :MAX_TASK_WINDOW_SCAN
            ]
        except OSError as exc:
            raise TaskWindowCorrupt("task window state could not be read") from exc
    truncated = len(paths) > limit or bool(fleet.get("truncated"))
    tasks = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise TaskWindowCorrupt("task window state could not be read") from exc
        if not isinstance(record, Mapping):
            raise TaskWindowCorrupt("task window state contains an invalid job record")
        tasks.append(
            _job_projection(
                record, work_items=work_items, scheduler_ranks=scheduler_ranks,
            )
        )
    tasks.sort(key=lambda item: (
        item["scheduler_rank"] is None,
        item["scheduler_rank"] or 0,
        _STATE_ORDER.get(item["state"], 99),
        -item["priority"],
        -item["updated_at"],
        item["job_id"],
    ))
    tasks = tasks[:limit]
    mission_groups: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        mission_id = task["mission_id"]
        if not mission_id:
            continue
        group = mission_groups.setdefault(mission_id, {
            "mission_id": mission_id,
            "scope": task["scope"],
            "status": task["mission_status"],
            "main_axis_sha256": task["main_axis_sha256"],
            "tasks": 0,
            "open_tasks": 0,
        })
        group["tasks"] += 1
        if not task["terminal"]:
            group["open_tasks"] += 1
    missions = list(mission_groups.values())
    return {
        "schema": TASK_WINDOW_SCHEMA,
        "state_root_sha256": hashlib.sha256(str(root).encode()).hexdigest(),
        "available": bool(fleet.get("available", False)),
        "truncated": truncated,
        "metrics": dict(fleet.get("metrics", {})),
        "missions": missions,
        "tasks": tasks,
    }
