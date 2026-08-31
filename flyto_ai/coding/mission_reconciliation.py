# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Fail-closed reconciliation for persisted coding mission projections."""
from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Sequence

from flyto_ai.coding.contracts import (
    MISSION_STATUS_DISPATCHED,
    MISSION_STATUS_READY,
    TERMINAL_CODING_JOB_STATES,
    CodingMissionProjection,
)
from flyto_ai.coding.mission_runtime import CodingMissionRuntime, MissionRouteError
from flyto_ai.orchestration.mission_control import WorkItem

_JOB_ID = re.compile(r"job_[0-9a-f]{24}")
_TERMINAL_STATES = frozenset(state.value for state in TERMINAL_CODING_JOB_STATES)


class RoundSettlement:
    """Close one dispatched round once, before publishing its settled state."""

    __slots__ = ("_service", "_work", "_tenant_ref", "_job_id", "_settled", "_changes")

    def __init__(self, service: Any, work: Any, tenant_ref: str, job_id: str) -> None:
        self._service = service
        self._work = work
        self._tenant_ref = tenant_ref
        self._job_id = job_id
        self._settled = work is None
        self._changes: Dict[str, Any] = {}

    @property
    def settled(self) -> bool:
        return self._settled

    @property
    def work_item_id(self) -> str:
        return "" if self._work is None else str(self._work.work_item_id)

    def __call__(
        self,
        *,
        revision: str = "",
        files: Sequence[str] = (),
        state: str = "",
        failure_code: str = "",
    ) -> Dict[str, Any]:
        if self._settled:
            return dict(self._changes)
        self._settled = True
        assert self._work is not None
        self._changes = self._service._close_round_item(
            self._work,
            self._tenant_ref,
            self._job_id,
            revision=revision,
            files=files,
            state=state,
            failure_code=failure_code,
        )
        return dict(self._changes)


def terminal_orphan_ready_projection(
    runtime: CodingMissionRuntime,
    record: Mapping[str, Any],
    *,
    tenant_ref: str,
) -> Optional[dict[str, Any]]:
    """Reclaim one exact terminal/dispatched orphan and project it as ready.

    The kernel lease is the execution fence.  This function neither closes a
    work item nor completes a mission: after a successful reclaim the caller
    must persist the returned projection and use the ordinary dispatcher for
    owner-bound deferred accounting.
    """

    validated = _terminal_projection(record)
    if validated is None:
        return None
    projection, _job_id = validated
    item = runtime.work_item(projection.work_item_id)
    return terminal_orphan_ready_projection_from_item(
        runtime,
        record,
        item=item,
        tenant_ref=tenant_ref,
        projection=projection,
    )


def terminal_orphan_ready_projection_from_item(
    runtime: CodingMissionRuntime,
    record: Mapping[str, Any],
    *,
    item: Optional[WorkItem],
    tenant_ref: str,
    projection: Optional[CodingMissionProjection] = None,
) -> Optional[dict[str, Any]]:
    """Project one terminal orphan from an already validated store snapshot."""

    validated = _terminal_projection(record, projection)
    if validated is None:
        return None
    projection, job_id = validated
    if not _item_matches(item, projection, tenant_ref, job_id):
        return None
    assert item is not None
    if not _reclaim_ready_item(runtime, item, projection.work_item_id):
        return None
    stored = dict(record.get("mission") or {})
    stored.update({"status": MISSION_STATUS_READY, "disposition": ""})
    try:
        return CodingMissionProjection.from_mapping(stored).to_mapping()
    except (TypeError, ValueError):
        return None


def _terminal_projection(
    record: Mapping[str, Any],
    projection: Optional[CodingMissionProjection] = None,
) -> Optional[tuple[CodingMissionProjection, str]]:
    """Validate the immutable job/projection coordinates shared by both paths."""

    if str(record.get("state") or "") not in _TERMINAL_STATES:
        return None
    if projection is None:
        try:
            projection = CodingMissionProjection.from_mapping(record.get("mission"))
        except (TypeError, ValueError):
            return None
    if projection.status not in {MISSION_STATUS_READY, MISSION_STATUS_DISPATCHED}:
        return None
    job_id = str(record.get("job_id") or "")
    return (projection, job_id) if _JOB_ID.fullmatch(job_id) else None


def _item_matches(
    item: Optional[WorkItem],
    projection: CodingMissionProjection,
    tenant_ref: str,
    job_id: str,
) -> bool:
    """Bind the validated store row to this exact tenant job projection."""

    return bool(
        item is not None
        and item.mission_id == projection.mission_id
        and item.coordinates.project == tenant_ref
        and item.coordinates.location == job_id
    )


def _reclaim_ready_item(
    runtime: CodingMissionRuntime,
    item: WorkItem,
    work_item_id: str,
) -> bool:
    """Keep dispatched reclaim live and accept an already-ready exact row."""

    if item.status != MISSION_STATUS_DISPATCHED:
        return item.status == MISSION_STATUS_READY
    try:
        return bool(runtime.reclaim(work_item_id))
    except MissionRouteError:
        return False
