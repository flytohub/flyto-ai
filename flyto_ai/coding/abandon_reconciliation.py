# Copyright 2025 Flyto2 Inc.
# Licensed under the Apache License, Version 2.0
"""Fail-closed Mission reconciliation for the host abandon valve."""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

from flyto_ai.coding.contracts import (
    MISSION_STATUS_CLOSED,
    MISSION_STATUS_DISPATCHED,
    MISSION_STATUS_READY,
    CodingJobState,
)
from flyto_ai.coding.errors import (
    AbandonStateConflict,
    CodingServiceBusy,
)
from flyto_ai.coding.mission_runtime import (
    DISPOSITION_BLOCKED,
    DISPOSITION_DEFERRED,
    CodingMissionRuntime,
    DispatchedWork,
    _translated,
    worker_identity,
)


@contextmanager
def _reconcile_dispatched(
    runtime: CodingMissionRuntime, work_item_id: str,
) -> Iterator[DispatchedWork | None]:
    """Reacquire one orphaned dispatch without selecting unrelated work."""

    runtime.require_supported()
    if not runtime.reclaim(work_item_id):
        yield None
        return
    item = runtime.work_item(work_item_id)
    if item is None or item.status != MISSION_STATUS_READY:
        yield None
        return
    key = runtime._key("reconcile-dispatch", work_item_id, uuid.uuid4().hex[:16])
    try:
        with _translated():
            dispatcher = runtime.store.dispatch_expected(
                operation=key,
                worker=runtime.worker,
                work_item_id=work_item_id,
                expected_attempt=item.attempts + 1,
            )
            with dispatcher as handle:
                yield None if handle is None else runtime._resolve(handle)
    finally:
        runtime._forget(key)


def settle_abandoned_mission(
    service: Any,
    record: Mapping[str, Any],
    tenant_ref: str,
    job_id: str,
    state: str,
    historical_split_brain: bool,
    settlement_factory: Any,
) -> dict[str, Any]:
    """Prove and, when needed, owner-close the job's projected work item."""

    projection = service._record_projection(record)
    item = (
        CodingMissionRuntime._persisted_work_item(
            service.state_root, projection.work_item_id,
        )
        if projection is not None else None
    )
    if projection is None or item is None or (
        item.mission_id != projection.mission_id
        or item.coordinates.project != tenant_ref
        or item.coordinates.location != job_id
    ):
        raise AbandonStateConflict(
            "the job's exact mission work item cannot be proven",
        )

    if item.status == MISSION_STATUS_DISPATCHED:
        transient_mission = service._mission is None
        if transient_mission:
            service._mission = CodingMissionRuntime(
                service.state_root, worker=worker_identity(service.instance_id),
            )
        try:
            with _reconcile_dispatched(service._mission, item.work_item_id) as work:
                if work is None:
                    raise CodingServiceBusy(
                        "the mission work item is still being executed",
                    )
                changes = settlement_factory(
                    service, work, tenant_ref, job_id,
                )(
                    state=CodingJobState.FAILED.value,
                    failure_code="job_abandoned",
                )
        finally:
            if transient_mission:
                service._mission = None
        if not changes:
            raise AbandonStateConflict(
                "the mission work item could not be settled",
            )
        return changes

    if historical_split_brain:
        if (
            item.status != MISSION_STATUS_CLOSED
            or item.disposition not in (DISPOSITION_BLOCKED, DISPOSITION_DEFERRED)
        ):
            raise AbandonStateConflict(
                "only this job's non-landable abandoned mission item can be reconciled",
            )
        return {"mission": CodingMissionRuntime.advance(
            record["mission"],
            status=MISSION_STATUS_CLOSED,
            disposition=str(item.disposition),
        )}

    if state != CodingJobState.AWAITING_CODEX_AUDIT.value and (
        item.status != MISSION_STATUS_CLOSED
        or item.disposition not in (DISPOSITION_BLOCKED, DISPOSITION_DEFERRED)
    ):
        raise AbandonStateConflict(
            "queued work must be closed blocked or deferred before abandonment",
        )
    return {}
