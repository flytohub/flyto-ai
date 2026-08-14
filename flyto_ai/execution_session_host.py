# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Durable, fail-closed admission host for prepared execution sessions."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from flyto_ai.execution_session import ExecutionAuthority, prepare_execution_session
from flyto_ai.scheduler import ScheduledTask, Scheduler, TaskSchedule
from flyto_ai.scheduler.tasks import ScheduleType

EXECUTION_SESSION_HOST_INSTRUCTION_VERSION = "flyto.execution-session-host-instruction.v1"
EXECUTION_SESSION_HOST_RECEIPT_VERSION = "flyto.execution-session-host-receipt.v1"
_TASK_PREFIX = "execution-session-"
_DIGEST_PREFIX = "sha256:"
_DIGEST_LENGTH = len(_DIGEST_PREFIX) + 64
_MAX_RECEIPT_BYTES = 16_384


class ExecutionSessionHostError(ValueError):
    """Raised when durable admission cannot be proven safe and unambiguous."""


def _task_id(session_id: str) -> str:
    """Map the exact, already-validated session identifier to a safe token."""
    return _TASK_PREFIX + hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _instruction(prepared: Mapping[str, Any]) -> dict[str, Any]:
    attestations = prepared.get("attestations")
    planning_input = prepared.get("planning_input")
    if not isinstance(attestations, Mapping) or not isinstance(planning_input, Mapping):
        raise ExecutionSessionHostError("prepared session has an invalid shape")
    session_id = planning_input.get("session_id")
    digests = {
        "request": attestations.get("request"),
        "authority": attestations.get("authority"),
        "route": attestations.get("route"),
        "overall": prepared.get("overall_digest"),
    }
    if not isinstance(session_id, str) or any(
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or not value.startswith(_DIGEST_PREFIX)
        for value in digests.values()
    ):
        raise ExecutionSessionHostError("prepared session has invalid correlation")
    return {
        "contract_version": EXECUTION_SESSION_HOST_INSTRUCTION_VERSION,
        "session": {"session_id": session_id, "task_id": _task_id(session_id)},
        "digests": digests,
    }


def _encode(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if len(encoded.encode("utf-8")) > _MAX_RECEIPT_BYTES:
        raise ExecutionSessionHostError("admission data exceeds its durable bound")
    return encoded


async def _admission_executor(_instruction_json: str) -> dict[str, Any]:
    return {"ok": False, "error": "execution_not_connected"}


def _receipt(
    instruction: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    if set(result) != {"slot", "ok", "cost_usd", "message", "error", "evidence_ref"}:
        raise ExecutionSessionHostError("durable occurrence result has an invalid shape")
    if (
        result.get("ok") is not False
        or result.get("error")
        not in {"execution_not_connected", "execution_outcome_unknown"}
        or result.get("message") != ""
        or result.get("cost_usd") != 0.0
    ):
        raise ExecutionSessionHostError("durable occurrence did not close blocked")
    receipt = {
        "contract_version": EXECUTION_SESSION_HOST_RECEIPT_VERSION,
        "session": dict(instruction["session"]),
        "digests": dict(instruction["digests"]),
        "status": "blocked",
        "result": dict(result),
    }
    # Round-trip through strict JSON to return containers detached from scheduler state.
    return json.loads(_encode(receipt))


async def admit_execution_session(
    state_root: os.PathLike[str] | str,
    untrusted_request: Mapping[str, Any],
    trusted_manifests: Sequence[Mapping[str, Any]],
    authority: ExecutionAuthority,
    now_ms: int,
    *,
    trusted_blueprints: Sequence[Mapping[str, Any]] = (),
    limit: int = 8,
) -> Mapping[str, Any]:
    """Prepare, durably admit, and truthfully block one execution occurrence."""
    if state_root is None or isinstance(state_root, bool):
        raise ExecutionSessionHostError("state_root must select durable state")
    try:
        state_path = os.fspath(state_root)
    except TypeError as exc:
        raise ExecutionSessionHostError("state_root must be a filesystem path") from exc
    if not state_path:
        raise ExecutionSessionHostError("state_root must select durable state")

    prepared = prepare_execution_session(
        untrusted_request,
        trusted_manifests,
        authority,
        now_ms,
        trusted_blueprints=trusted_blueprints,
        limit=limit,
    )
    instruction = _instruction(prepared)
    instruction_json = _encode(instruction)
    task_id = instruction["session"]["task_id"]
    scheduler = Scheduler(executor=_admission_executor, state_root=state_root)

    existing = scheduler.get_task(task_id)
    if existing is None:
        task = ScheduledTask(
            task_id=task_id,
            name="Execution session admission",
            instruction=instruction_json,
            schedule=TaskSchedule(type=ScheduleType.ONE_SHOT, run_at=0),
            tags=["execution-session-admission"],
        )
        try:
            scheduler.add_task(task)
        except ValueError:
            # A concurrent admission may have won the durable unique-key race.
            existing = scheduler.get_task(task_id)
            if existing is None:
                raise
    if existing is not None and existing.instruction != instruction_json:
        raise ExecutionSessionHostError("session admission conflicts with durable state")

    await scheduler.run_once()
    persisted = scheduler.get_task(task_id)
    if persisted is None or persisted.instruction != instruction_json:
        raise ExecutionSessionHostError("session admission durable state is unresolved")
    summary = scheduler.summary()
    matching = [item for item in summary["tasks"] if item["task_id"] == task_id]
    if len(matching) != 1 or len(matching[0]["results"]) != 1:
        raise ExecutionSessionHostError("session admission occurrence is unresolved")
    return _receipt(instruction, matching[0]["results"][0])


__all__ = [
    "EXECUTION_SESSION_HOST_INSTRUCTION_VERSION",
    "EXECUTION_SESSION_HOST_RECEIPT_VERSION",
    "ExecutionSessionHostError",
    "admit_execution_session",
]
