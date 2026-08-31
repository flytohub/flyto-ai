# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Cold-start reconciliation without history-multiplied MissionStore reads."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Mapping

from flyto_ai.coding.contracts import (
    MISSION_STATUS_DISPATCHED,
    MISSION_STATUS_READY,
    TERMINAL_CODING_JOB_STATES,
    CodingJobState,
    CodingMissionProjection,
)
from flyto_ai.coding.mission_reconciliation import (
    terminal_orphan_ready_projection_from_item,
)
from flyto_ai.orchestration.mission_batch import read_work_items_fail_closed


_TERMINAL_STATES = frozenset(state.value for state in TERMINAL_CODING_JOB_STATES)
_PUMPABLE_STATES = frozenset({
    CodingJobState.QUEUED.value,
    CodingJobState.REWORK_QUEUED.value,
})
_EXECUTING_STATES = frozenset({
    CodingJobState.RUNNING.value,
    CodingJobState.REWORK_RUNNING.value,
})


def _terminal_candidate(
    service: Any,
    record: Mapping[str, Any],
) -> CodingMissionProjection | None:
    if str(record.get("state") or "") not in _TERMINAL_STATES:
        return None
    projection = service._record_projection(record)
    if projection is None or projection.status not in {
        MISSION_STATUS_READY,
        MISSION_STATUS_DISPATCHED,
    }:
        return None
    return projection


def reconcile_interrupted_jobs(service: Any) -> None:
    """Reconcile one service restart while validating MissionStore only once.

    Job records remain independently parsed and attributed. Only the known
    work-item rows used by terminal-orphan projection share a validated store
    snapshot; lease-based reclaim and every record mutation keep their existing
    exact checks.
    """

    candidates = _scan_interrupted_records(service)
    identifiers = tuple(dict.fromkeys(
        projection.work_item_id for _path, _record, _tenant, projection in candidates
    ))
    items = read_work_items_fail_closed(service._mission.store, identifiers)
    for path, record, tenant_ref, projection in candidates:
        _reclaim_terminal_candidate(
            service, path, record, tenant_ref, projection, items,
        )

    service._sweep_workspace_claims()
    service._reconcile_continuation_claims()
    for _ in range(service._reclaimed):
        service._prime_pump()
    service._reclaimed = 0


def _scan_interrupted_records(
    service: Any,
) -> list[tuple[Any, Dict[str, Any], str, CodingMissionProjection]]:
    """Parse every job once and perform the unchanged nonterminal recovery."""

    candidates: list[tuple[Any, Dict[str, Any], str, CodingMissionProjection]] = []
    tenants = service.state_root / "tenants"
    if not tenants.is_dir():
        return candidates
    for path in tenants.glob("*/jobs/job_*.json"):
        _scan_interrupted_record(service, path, candidates)
    return candidates


def _scan_interrupted_record(
    service: Any,
    path: Any,
    candidates: list[tuple[Any, Dict[str, Any], str, CodingMissionProjection]],
) -> None:
    """Route one parsed record to terminal batching or ordinary recovery."""

    try:
        record = service._read_json(path)
        projection = _terminal_candidate(service, record)
        if str(record.get("state") or "") in _TERMINAL_STATES:
            if projection is not None:
                candidates.append((path, record, path.parent.parent.name, projection))
            return
        if record.get("state") in _PUMPABLE_STATES:
            if service._may_execute(record):
                service._reconcile_queued_job(path, record)
            return
        if record.get("state") in _EXECUTING_STATES:
            _fail_interrupted_execution(service, path, record)
    except (OSError, ValueError, json.JSONDecodeError):
        return


def _reclaim_terminal_candidate(
    service: Any,
    path: Any,
    record: Dict[str, Any],
    tenant_ref: str,
    projection: CodingMissionProjection,
    items: Mapping[str, Any],
) -> None:
    """Persist one exact ready projection; malformed candidates stay untouched."""

    try:
        ready = terminal_orphan_ready_projection_from_item(
            service._mission,
            record,
            item=items.get(projection.work_item_id),
            tenant_ref=tenant_ref,
            projection=projection,
        )
        if ready is not None:
            service._update_record_locked(path, mission=ready)
            service._reclaimed += 1
    except (OSError, ValueError, json.JSONDecodeError):
        return


def _fail_interrupted_execution(
    service: Any,
    path: Any,
    record: Dict[str, Any],
) -> None:
    job_id = str(record.get("job_id") or "")
    if not service._acquire_job_lease(job_id):
        return
    try:
        record.update({
            "state": CodingJobState.FAILED.value,
            "updated_at": time.time(),
            "landable": False,
            "failure_code": "service_restarted",
        })
        service._write_json(path, record)
        service._publish_status(record)
        service._discard_resume(path.parent.parent.name, job_id)
        service._reclaim_mission_item(path, record)
    finally:
        service._release_job_lease(job_id)
