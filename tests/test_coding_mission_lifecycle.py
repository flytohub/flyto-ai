# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Hostile tests for the *wired* coding mission lifecycle.

The contract slice is tested next door in `test_coding_mission_contract.py` and
asserts only value types. This module asserts the opposite half: that a real
`MissionStore` is the authority a real `CodingService` actually runs through.
Nothing here stubs the kernel, and several tests run two `CodingService`
instances over one state root and one workspace, because the properties that
matter - global queue order, resource exclusion, no stolen lease, no duplicated
provider round - are only observable when more than one process authority
exists.

Every test that needs the kernel's two host primitives is skipped where they are
absent, and the boundary itself is asserted unconditionally.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from flyto_ai.coding import CodingTaskRequest, FlytoCodingAgent
from flyto_ai.coding.contracts import (
    MISSION_COMPLETED,
    MISSION_DISPOSITION_FIXED,
    MISSION_OPEN,
    MISSION_PROJECTION_FIELDS,
    MISSION_STATUS_CLOSED,
    CodingAuditFinding,
    CodingAuditSeverity,
    CodingAuditVerdict,
    CodingJobState,
    CodingMissionEnvelope,
    CodingMissionProjection,
    TERMINAL_CODING_JOB_STATES,
)
from flyto_ai.coding.mission_runtime import (
    CRITERION_AUDIT,
    CRITERION_CHECKS,
    CRITERION_REVISION,
    MISSION_DESIRED_RESULT,
    MISSION_RESOURCE_KIND,
    MISSION_RESOURCE_NAMESPACE,
    CodingMissionRuntime,
    DispatchedWork,
    MissionAuthorityRefused,
    MissionRouteError,
    coding_scope,
    synthesize_envelope,
    worker_identity,
)
from flyto_ai.coding.service import (
    AUTHORITY_MARKER_NAME,
    AUTHORITY_MARKER_VERSION,
    MAX_AUTHORITY_MARKER_BYTES,
    EXECUTION_AUTHORITY_UNBOUND,
    MISSION_ITEM_UNAVAILABLE,
    CodingAuthorityConflict,
    CodingAuthorityUnavailable,
    CodingService,
    MissionRouteRefused,
    _DISPATCH_FOREIGN,
    receipt_to_mapping,
)
from flyto_ai.orchestration.mission_control import (
    DISPOSITION_BLOCKED,
    DISPOSITION_DEFERRED,
    DISPOSITION_FIXED,
    LANE_PRIMARY,
    LANE_REPAIR,
    STATUS_CLOSED,
    STATUS_DISPATCHED,
    STATUS_READY,
    AcceptanceCriterion,
    Closure,
    MissionConflict,
    MissionStore,
    MissionUnauthorized,
    MissionResource,
    WorkCoordinates,
    inspect_host,
)

_HOST = inspect_host()
needs_host = pytest.mark.skipif(
    not _HOST.supported,
    reason="this host lacks the mission kernel's required primitives: {}".format(
        ", ".join(_HOST.missing) or "unknown",
    ),
)

_SETTLED = TERMINAL_CODING_JOB_STATES | {CodingJobState.AWAITING_CODEX_AUDIT}
_TENANT = "tenant-mission"
_SECRET = "sk-" + "live-do-not-leak-0123456789"


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


class _Provider:
    """Deterministic provider; every effect goes through the real tool boundary."""

    #: Live rounds across every provider instance, and the high-water mark.
    #: Resource exclusion is a statement about concurrency, so it is measured
    #: rather than inferred from which fixture happened to run which job.
    active = 0
    max_active = 0
    census = threading.Lock()

    @classmethod
    def reset(cls) -> None:
        with cls.census:
            cls.active = 0
            cls.max_active = 0

    def __init__(self, delay: float = 0.0, tag: str = "solo") -> None:
        self.delay = delay
        #: Every provider writes into its *own* file with its own bytes. Two
        #: providers that wrote identical content to identical paths would make
        #: the second job's first round a legitimate no-op - no attributable
        #: change - and the agent would correctly take another round. That is a
        #: fixture artefact, not a scheduling fact, and it would silently weaken
        #: the exactly-one-round assertion this suite depends on.
        self.tag = tag
        self.rounds = 0
        self.lock = threading.Lock()
        self.prompts: List[str] = []

    async def chat(self, **kwargs: Any):
        with self.lock:
            self.rounds += 1
            index = self.rounds
        self.prompts.append(json.dumps(kwargs.get("messages", []), ensure_ascii=False))
        cls = type(self)
        with cls.census:
            cls.active += 1
            cls.max_active = max(cls.max_active, cls.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            for path, content in (
                ("result.txt", "verified\n"),
                ("notes-{}.txt".format(self.tag),
                 "{} round {}\n".format(self.tag, index)),
            ):
                outcome = await kwargs["dispatch_fn"](
                    "coding_write_file",
                    {"path": path, "content": content, "overwrite": True},
                )
                assert outcome["ok"]
        finally:
            with cls.census:
                cls.active -= 1
        return "done", [{"function": "coding_write_file", "ok": True}], 1, {"total_tokens": 1}


def _declare(workspace: Path) -> None:
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: real_file_check\n"
        "    argv: {}\n".format(json.dumps([
            sys.executable,
            "-c",
            "from pathlib import Path; assert Path('result.txt').read_text()"
            " == 'verified\\n'",
        ])),
        encoding="utf-8",
    )


def _service(
    state_root: Path,
    workspace: Path,
    *,
    provider: Optional[_Provider] = None,
    require_codex_audit: bool = False,
    max_workers: int = 2,
    max_rework_rounds: int = 3,
) -> CodingService:
    return CodingService(
        lambda store: FlytoCodingAgent(provider or _Provider(), store=store),
        state_root=str(state_root),
        workspace_roots=(str(workspace),),
        max_workers=max_workers,
        max_queued=8,
        require_codex_audit=require_codex_audit,
        max_rework_rounds=max_rework_rounds,
    )


def _request(workspace: Path, *, message: str = "write verified result", **kwargs: Any):
    return CodingTaskRequest(message=message, working_dir=str(workspace), **kwargs)


def _wait(service: CodingService, job_id: str, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        receipt = service.get(_TENANT, job_id)
        if receipt.state in _SETTLED:
            return receipt
        time.sleep(0.02)
    raise AssertionError(
        "coding job {} never settled (last state {})".format(
            job_id, service.get(_TENANT, job_id).state,
        ),
    )


def _store(state_root: Path) -> MissionStore:
    """A third, independent view of the same durable store."""

    return MissionStore(state_root)


def _items(state_root: Path) -> Dict[str, Any]:
    return {
        item.work_item_id: item for item in _store(state_root).snapshot(limit=200).work_items
    }


# --------------------------------------------------------------------------
# boundary: asserted on every host, including unsupported ones
# --------------------------------------------------------------------------


def test_the_host_boundary_is_answered_without_raising() -> None:
    """`inspect_host` is what lets a service refuse at start-up, so it never raises."""

    capabilities = inspect_host()
    assert isinstance(capabilities.supported, bool)
    assert capabilities.supported == CodingMissionRuntime.supported()
    assert capabilities.supported != bool(capabilities.missing)


def test_a_worker_identity_is_bounded_and_process_distinct() -> None:
    token = worker_identity("abc123")
    assert token.isprintable() and not any(part.isspace() for part in token)
    assert len(token) <= 128
    assert token == worker_identity("abc123")


def test_the_runtime_refuses_an_unbounded_worker(tmp_path: Path) -> None:
    for worker in ("", " ", "has space", "x" * 200, "line\nbreak"):
        with pytest.raises(ValueError):
            CodingMissionRuntime(tmp_path, worker=worker)


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------


def test_a_job_without_a_mission_synthesizes_the_coding_adapter_contract() -> None:
    envelope = synthesize_envelope("refactor the parser", workspace_sha256="ab" * 32)
    assert envelope.objective == "refactor the parser"
    assert envelope.desired_result == MISSION_DESIRED_RESULT
    assert envelope.criteria_ids == (CRITERION_REVISION, CRITERION_CHECKS, CRITERION_AUDIT)
    assert envelope.is_root and envelope.lane == LANE_PRIMARY
    assert envelope.scope == coding_scope("ab" * 32)
    # The scope groups by worktree without ever being a worktree.
    assert "/" not in envelope.scope and len(envelope.scope) <= 35


def test_a_multiline_objective_is_folded_and_a_long_one_is_visibly_cut() -> None:
    """The kernel stores one printable line, so folding is honest truncation."""

    folded = synthesize_envelope("first\n\tsecond   third", workspace_sha256="cd" * 32)
    assert folded.objective == "first second third"
    long_message = " ".join("word{}".format(index) for index in range(2000))
    cut = synthesize_envelope(long_message, workspace_sha256="cd" * 32)
    assert cut.objective.endswith(" [...]")
    assert len(cut.objective) <= 2000
    assert cut.objective.startswith("word0 word1")


def test_synthesis_never_reaches_the_workload_neutral_kernel() -> None:
    """The coding vocabulary lives here; the store has no API that knows it."""

    import flyto_ai.orchestration.mission_control as kernel

    exported = set(kernel.__all__)
    for name in (
        "CRITERION_AUDIT", "CRITERION_REVISION", "CRITERION_CHECKS",
        "synthesize_envelope", "coding_scope", "CodingMissionRuntime",
    ):
        assert name not in exported, name
    # And the criteria a coding mission declares are this module's constants,
    # not something the kernel could have supplied.
    assert not hasattr(kernel, "CRITERION_AUDIT")


# --------------------------------------------------------------------------
# admission
# --------------------------------------------------------------------------


@needs_host
def test_every_new_job_is_admitted_to_a_real_mission(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    service = _service(tmp_path / "state", workspace)
    try:
        receipt = service.submit(_TENANT, "key-1", _request(workspace))
        assert receipt.mission is not None
        projection = CodingMissionProjection.from_mapping(receipt.mission)
        assert projection.is_root
        assert projection.lane == LANE_PRIMARY
        mission = _store(tmp_path / "state").get_mission(projection.mission_id)
        assert mission.status == MISSION_OPEN
        assert mission.criteria_ids == (
            CRITERION_REVISION, CRITERION_CHECKS, CRITERION_AUDIT,
        )
        assert mission.objective == "write verified result"
        _wait(service, receipt.job_id)
    finally:
        service.close()


@needs_host
def test_admission_is_idempotent_and_places_exactly_one_work_item(
    tmp_path: Path,
) -> None:
    """A replayed idempotency key must not fork the mission graph."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    service = _service(state, workspace)
    try:
        first = service.submit(_TENANT, "key-1", _request(workspace))
        second = service.submit(_TENANT, "key-1", _request(workspace))
        assert first.job_id == second.job_id
        assert first.mission == second.mission
        _wait(service, first.job_id)
        snapshot = _store(state).snapshot(limit=200)
        assert len(snapshot.missions) == 1
        assert len(snapshot.work_items) == 1
    finally:
        service.close()


@needs_host
def test_admission_binds_tenant_job_and_workspace_without_a_path(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    service = _service(state, workspace)
    try:
        receipt = service.submit(_TENANT, "key-1", _request(workspace))
        projection = CodingMissionProjection.from_mapping(receipt.mission or {})
        item = _store(state).get_work_item(projection.work_item_id)
        assert isinstance(item.coordinates, WorkCoordinates)
        # Private identity, bounded, and never the caller's path or prose.
        assert item.coordinates.location == receipt.job_id
        assert len(item.coordinates.project) == 64
        assert str(workspace) not in json.dumps(list(item.coordinates.__dict__.values()))
        # The canonical workspace is claimed by digest.
        assert len(item.resources) == 1
        resource = item.resources[0]
        assert isinstance(resource, MissionResource)
        assert resource.namespace == MISSION_RESOURCE_NAMESPACE
        assert resource.kind == MISSION_RESOURCE_KIND
        assert str(workspace) not in resource.identity
        _wait(service, receipt.job_id)
    finally:
        service.close()


@needs_host
def test_a_named_mission_is_honoured_and_a_forged_contract_is_refused(
    tmp_path: Path,
) -> None:
    """Attaching to an existing mission validates its immutable contract."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    runtime = CodingMissionRuntime(state, worker="w-test-1")
    envelope = CodingMissionEnvelope(
        scope="scope-token",
        objective="reach the stated objective",
        desired_result="the objective is demonstrably reached",
        acceptance_criteria=(AcceptanceCriterion("c1", "the declared checks pass"),),
    )
    admission = runtime.admit(
        tenant_ref="a" * 64,
        job_id="job_" + "b" * 24,
        workspace_sha256="c" * 64,
        envelope=envelope,
        message="ignored when an envelope is named",
    )
    stored = runtime.store.get_mission(admission.mission_id)
    assert stored.objective == "reach the stated objective"
    # A side item naming that mission but stating a different contract is
    # refused: a mission is validated, never amended.
    forged = CodingMissionEnvelope(
        scope="scope-token",
        objective="reach a different objective",
        desired_result="the objective is demonstrably reached",
        acceptance_criteria=(AcceptanceCriterion("c1", "the declared checks pass"),),
        mission_id=admission.mission_id,
        parent_id=admission.work_item_id,
        return_to_id=admission.work_item_id,
    )
    with pytest.raises(MissionRouteError):
        runtime.admit(
            tenant_ref="a" * 64,
            job_id="job_" + "d" * 24,
            workspace_sha256="c" * 64,
            envelope=forged,
            message="",
        )


@needs_host
def test_a_legacy_request_without_a_mission_takes_the_same_route(
    tmp_path: Path,
) -> None:
    """Compatibility is not a bypass: an unnamed mission is still a mission."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    service = _service(state, workspace)
    try:
        request = _request(workspace)
        assert request.mission is None
        receipt = service.submit(_TENANT, "key-1", request)
        settled = _wait(service, receipt.job_id)
        assert settled.state is CodingJobState.COMPLETED
        assert settled.mission is not None
        item = _items(state)[
            CodingMissionProjection.from_mapping(settled.mission).work_item_id
        ]
        assert item.status == STATUS_CLOSED
        assert item.attempts == 1
    finally:
        service.close()


# --------------------------------------------------------------------------
# cross-instance queue: order, exclusion, handoff
# --------------------------------------------------------------------------


@needs_host
def test_two_instances_share_one_queue_and_never_run_one_workspace_twice(
    tmp_path: Path,
) -> None:
    """The failing handoff, asserted directly.

    Two service instances, one state root, one workspace. The mission store
    holds that worktree as an exclusive resource, so the second job cannot start
    until the first closes - and the release happens in the *other* instance, so
    nothing in-process can announce it. Both jobs must still finish.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    # Distinct attributable bytes per fixture, so a round is never a no-op that
    # would earn a legitimate extra model round.
    _Provider.reset()
    first_provider = _Provider(delay=0.3, tag="one")
    second_provider = _Provider(tag="two")
    first = _service(state, workspace, provider=first_provider)
    second = _service(state, workspace, provider=second_provider)
    try:
        one = first.submit(_TENANT, "key-1", _request(workspace, message="first task"))
        two = second.submit(_TENANT, "key-2", _request(workspace, message="second task"))
        settled_one = _wait(first, one.job_id)
        settled_two = _wait(second, two.job_id)
        assert settled_one.state is CodingJobState.COMPLETED
        assert settled_two.state is CodingJobState.COMPLETED
        # Exactly two provider rounds across the fleet - one per job. *Which*
        # instance ran a given job is deliberately not asserted: admission no
        # longer pins a job to its submitter, so either compatible worker may
        # execute either job. The global total is the exactly-once invariant.
        assert first_provider.rounds + second_provider.rounds == 2
        # Exclusivity is measured, not inferred: the two rounds never overlapped
        # on this one worktree.
        assert _Provider.max_active == 1
        # Two rounds really did land attributable bytes.
        assert len(list(workspace.glob("notes-*.txt"))) >= 1
        items = _items(state)
        assert len(items) == 2
        # Every item left the queue explicitly, and each was really dispatched.
        assert all(item.status == STATUS_CLOSED for item in items.values())
        assert all(item.disposition for item in items.values())
        # The strong assertion, restored. A queued job holds no lease, so no
        # worker can be offered work it must refuse, so nothing requeues and no
        # attempt is burnt on contention.
        assert sorted(item.attempts for item in items.values()) == [1, 1]
    finally:
        first.close()
        second.close()


@needs_host
def test_the_store_prefers_repair_work_over_primary_work(tmp_path: Path) -> None:
    """Repair beats primary inside one scope, and does not cut through conflict."""

    state = tmp_path / "state"
    runtime = CodingMissionRuntime(state, worker="w-order")
    tenant = "a" * 64
    root = runtime.admit(
        tenant_ref=tenant,
        job_id="job_" + "1" * 24,
        workspace_sha256="1" * 64,
        envelope=None,
        message="root task",
    )
    # Close the root so its repair child becomes runnable and its worktree free.
    with runtime.dispatch() as work:
        assert work is not None and work.work_item_id == root.work_item_id
        runtime.close_fixed(
            work,
            tenant_ref=tenant,
            job_id=work.job_id,
            mission_id=work.mission_id,
            work_item_id=work.work_item_id,
        )
    # A plain primary item in the same scope, submitted *first*...
    later = runtime.admit(
        tenant_ref=tenant,
        job_id="job_" + "2" * 24,
        workspace_sha256="1" * 64,
        envelope=None,
        message="unrelated task",
    )
    # ...and a repair child of the original mission, submitted second.
    repair = runtime.submit_repair(
        tenant_ref=tenant,
        job_id="job_" + "1" * 24,
        workspace_sha256="1" * 64,
        projection=root.projection,
        round_index=1,
    )
    assert repair.projection.lane == LANE_REPAIR
    assert repair.projection.parent_id == root.work_item_id
    assert repair.projection.return_to_id == root.work_item_id
    with runtime.dispatch() as work:
        assert work is not None
        # Same scope, older primary item queued first, and repair still wins.
        assert work.work_item_id == repair.work_item_id
        assert work.is_repair
        runtime.close_fixed(
            work,
            tenant_ref=tenant,
            job_id=work.job_id,
            mission_id=work.mission_id,
            work_item_id=work.work_item_id,
        )
    with runtime.dispatch() as work:
        assert work is not None and work.work_item_id == later.work_item_id


@needs_host
def test_one_workspace_is_never_held_by_two_running_work_items(
    tmp_path: Path,
) -> None:
    """Resource exclusion, proved against the store rather than against timing."""

    state = tmp_path / "state"
    runtime = CodingMissionRuntime(state, worker="w-excl")
    tenant = "a" * 64
    shared = "f" * 64
    first = runtime.admit(
        tenant_ref=tenant, job_id="job_" + "1" * 24,
        workspace_sha256=shared, envelope=None, message="one",
    )
    runtime.admit(
        tenant_ref=tenant, job_id="job_" + "2" * 24,
        workspace_sha256=shared, envelope=None, message="two",
    )
    with runtime.dispatch() as held:
        assert held is not None and held.work_item_id == first.work_item_id
        assert runtime.is_workspace_claimed(shared)
        # A second dispatch, even from a different worker, finds nothing to take.
        second = CodingMissionRuntime(state, worker="w-excl-2")
        with second.dispatch() as blocked:
            assert blocked is None
        runtime.close_fixed(
            held, tenant_ref=tenant, job_id=held.job_id,
            mission_id=held.mission_id, work_item_id=held.work_item_id,
        )
    assert not runtime.is_workspace_claimed(shared)
    second = CodingMissionRuntime(state, worker="w-excl-2")
    with second.dispatch() as freed:
        assert freed is not None


# --------------------------------------------------------------------------
# authority: handles, heartbeats, fencing
# --------------------------------------------------------------------------


@needs_host
def test_a_handle_cannot_be_forged_from_identifiers_in_a_receipt(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    runtime = CodingMissionRuntime(state, worker="w-auth")
    tenant = "a" * 64
    placed = runtime.admit(
        tenant_ref=tenant, job_id="job_" + "1" * 24,
        workspace_sha256="1" * 64, envelope=None, message="one",
    )
    forged = DispatchedWork(
        handle=None,  # type: ignore[arg-type]
        tenant_ref=tenant,
        job_id="job_" + "1" * 24,
        workspace_sha256="1" * 64,
    )
    with pytest.raises(MissionAuthorityRefused):
        runtime.close_fixed(
            forged, tenant_ref=tenant, job_id="job_" + "1" * 24,
            mission_id=placed.mission_id, work_item_id=placed.work_item_id,
        )
    # The item is untouched: no closure, still ready.
    assert runtime.store.get_work_item(placed.work_item_id).status == STATUS_READY


@needs_host
def test_a_live_handle_may_not_close_another_tenants_or_jobs_work(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    runtime = CodingMissionRuntime(state, worker="w-auth")
    tenant = "a" * 64
    placed = runtime.admit(
        tenant_ref=tenant, job_id="job_" + "1" * 24,
        workspace_sha256="1" * 64, envelope=None, message="one",
    )
    with runtime.dispatch() as work:
        assert work is not None
        for kwargs in (
            {"tenant_ref": "b" * 64},
            {"job_id": "job_" + "9" * 24},
            {"mission_id": "m-000000000099"},
            {"work_item_id": "w-000000000099"},
        ):
            arguments = {
                "tenant_ref": tenant,
                "job_id": "job_" + "1" * 24,
                "mission_id": placed.mission_id,
                "work_item_id": placed.work_item_id,
            }
            arguments.update(kwargs)
            with pytest.raises(MissionAuthorityRefused):
                runtime.close_fixed(work, **arguments)
        runtime.close_fixed(
            work, tenant_ref=tenant, job_id="job_" + "1" * 24,
            mission_id=placed.mission_id, work_item_id=placed.work_item_id,
        )


@needs_host
def test_a_stale_handle_is_rejected_after_its_era_ended(tmp_path: Path) -> None:
    """A dispatch that was requeued cannot write into the era that replaced it."""

    state = tmp_path / "state"
    runtime = CodingMissionRuntime(state, worker="w-fence")
    tenant = "a" * 64
    placed = runtime.admit(
        tenant_ref=tenant, job_id="job_" + "1" * 24,
        workspace_sha256="1" * 64, envelope=None, message="one",
    )
    escaped: Dict[str, Any] = {}
    with runtime.dispatch() as work:
        assert work is not None
        escaped["work"] = work
        escaped["fence"] = work.fence
    # Left without closing: requeued, and the *next* dispatch burns a higher
    # token, which is what makes the escaped handle stale.
    with runtime.dispatch() as newer:
        assert newer is not None
        assert newer.fence > escaped["fence"]
        with pytest.raises((MissionRouteError, MissionUnauthorized)):
            runtime.close_fixed(
                escaped["work"], tenant_ref=tenant, job_id="job_" + "1" * 24,
                mission_id=placed.mission_id, work_item_id=placed.work_item_id,
            )
        runtime.close_fixed(
            newer, tenant_ref=tenant, job_id=newer.job_id,
            mission_id=newer.mission_id, work_item_id=newer.work_item_id,
        )


@needs_host
def test_heartbeats_are_recorded_and_never_confer_or_cost_authority(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    runtime = CodingMissionRuntime(state, worker="w-beat")
    tenant = "a" * 64
    runtime.admit(
        tenant_ref=tenant, job_id="job_" + "1" * 24,
        workspace_sha256="1" * 64, envelope=None, message="one",
    )
    from flyto_ai.coding.mission_runtime import MissionHeartbeat

    with runtime.dispatch() as work:
        assert work is not None
        pulse = MissionHeartbeat(work.handle, interval=0.05)
        pulse.beat()
        pulse.start()
        time.sleep(0.2)
        pulse.stop()
        assert pulse.beats >= 2
        item = runtime.store.get_work_item(work.work_item_id)
        assert item.heartbeats >= 2
        # Nothing about authority moved: same fence, same status, same worker.
        assert item.fence == work.fence
        assert item.status == STATUS_DISPATCHED
        runtime.close_fixed(
            work, tenant_ref=tenant, job_id=work.job_id,
            mission_id=work.mission_id, work_item_id=work.work_item_id,
        )


@needs_host
def test_a_live_lease_is_never_stolen_by_silence(tmp_path: Path) -> None:
    """No TTL, no grace period, no guess from a heartbeat that stopped."""

    state = tmp_path / "state"
    runtime = CodingMissionRuntime(state, worker="w-live")
    other = CodingMissionRuntime(state, worker="w-thief")
    tenant = "a" * 64
    runtime.admit(
        tenant_ref=tenant, job_id="job_" + "1" * 24,
        workspace_sha256="1" * 64, envelope=None, message="one",
    )
    with runtime.dispatch() as work:
        assert work is not None
        # Deliberately quiet: no heartbeat at all for well past any plausible
        # timeout a time-based scheduler would have used.
        time.sleep(0.3)
        with pytest.raises(MissionRouteError):
            other.reclaim(work.work_item_id)
        with pytest.raises(MissionConflict):
            other.store.reclaim(work.work_item_id, operation="steal-attempt")
        assert runtime.store.get_work_item(work.work_item_id).status == STATUS_DISPATCHED
        runtime.close_fixed(
            work, tenant_ref=tenant, job_id=work.job_id,
            mission_id=work.mission_id, work_item_id=work.work_item_id,
        )


# --------------------------------------------------------------------------
# closure, audit, rework, side return
# --------------------------------------------------------------------------


@needs_host
def test_an_audited_job_closes_fixed_and_acceptance_completes_the_mission(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    service = _service(state, workspace, require_codex_audit=True)
    try:
        receipt = service.submit(_TENANT, "key-1", _request(workspace))
        ready = _wait(service, receipt.job_id)
        assert ready.state is CodingJobState.AWAITING_CODEX_AUDIT
        projection = CodingMissionProjection.from_mapping(ready.mission or {})
        # An attributable auditable revision exists, so the item closed fixed.
        assert projection.status == MISSION_STATUS_CLOSED
        assert projection.disposition == MISSION_DISPOSITION_FIXED
        assert projection.mission_status == MISSION_OPEN
        item = _store(state).get_work_item(projection.work_item_id)
        assert item.disposition == DISPOSITION_FIXED

        accepted = service.audit(
            _TENANT, receipt.job_id, ready.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
        final = CodingMissionProjection.from_mapping(accepted.mission or {})
        assert final.mission_status == MISSION_COMPLETED
        mission = _store(state).get_mission(final.mission_id)
        # Evidence for exactly every criterion, and no criterion left unanswered.
        assert {key for key, _ in mission.acceptance_evidence} == set(
            mission.criteria_ids,
        )
        assert dict(mission.acceptance_evidence)[CRITERION_REVISION] == (
            ready.implementation_revision_sha256
        )
    finally:
        service.close()


@needs_host
def test_a_pre_audit_terminal_failure_closes_with_the_whole_accounting(
    tmp_path: Path,
) -> None:
    """No attributable revision means no `fixed`, and never a silent close."""

    class _NoChange:
        async def chat(self, **_kwargs: Any):
            return "nothing to do", [], 1, {"total_tokens": 1}

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    service = _service(
        state, workspace, provider=_NoChange(), require_codex_audit=True,
    )
    try:
        receipt = service.submit(_TENANT, "key-1", _request(workspace))
        settled = _wait(service, receipt.job_id)
        assert settled.state is CodingJobState.FAILED
        projection = CodingMissionProjection.from_mapping(settled.mission or {})
        assert projection.status == MISSION_STATUS_CLOSED
        assert projection.disposition == DISPOSITION_BLOCKED
        item = _store(state).get_work_item(projection.work_item_id)
        closure = item.closure
        assert closure is not None
        assert closure.disposition == DISPOSITION_BLOCKED
        # Every field the kernel demands of a non-delivering closure is present.
        assert closure.rationale and closure.risk and closure.owner
        assert closure.evidence_refs
        assert closure.revisit_at and closure.revisit_at > int(time.time())
        # And the mission is not completed on the strength of a failure.
        assert _store(state).get_mission(item.mission_id).status == MISSION_OPEN
    finally:
        service.close()


@needs_host
def test_rework_places_one_repair_child_in_the_same_session(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    provider = _Provider()
    service = _service(state, workspace, provider=provider, require_codex_audit=True)
    try:
        receipt = service.submit(_TENANT, "key-1", _request(workspace))
        ready = _wait(service, receipt.job_id)
        first_item = CodingMissionProjection.from_mapping(ready.mission or {})
        session = ready.implementation_session_id
        assert session

        service.audit(
            _TENANT, receipt.job_id, ready.implementation_revision_sha256,
            CodingAuditVerdict.REWORK,
            (CodingAuditFinding(
                code="needs_more", message="do more work",
                severity=CodingAuditSeverity.MAJOR,
            ),),
        )
        reworked = _wait(service, receipt.job_id)
        assert reworked.rework_count == 1
        assert reworked.implementation_session_id == session
        child = CodingMissionProjection.from_mapping(reworked.mission or {})
        assert child.work_item_id != first_item.work_item_id
        assert child.lane == LANE_REPAIR
        assert child.parent_id == first_item.work_item_id
        # The route home points at the main axis.
        assert child.return_to_id == first_item.work_item_id
        items = _items(state)
        assert len(items) == 2
        # Exactly one repair child: a retried rework never forks the graph.
        assert sum(1 for item in items.values() if item.lane == LANE_REPAIR) == 1
    finally:
        service.close()


@needs_host
def test_a_side_item_accept_returns_home_and_completes_nothing(
    tmp_path: Path,
) -> None:
    """The fixed-root rule binds the axis; a branch may not declare victory."""

    state = tmp_path / "state"
    runtime = CodingMissionRuntime(state, worker="w-side")
    tenant = "a" * 64
    root = runtime.admit(
        tenant_ref=tenant, job_id="job_" + "1" * 24,
        workspace_sha256="1" * 64, envelope=None, message="root",
    )
    with runtime.dispatch() as work:
        assert work is not None
        runtime.close_fixed(
            work, tenant_ref=tenant, job_id=work.job_id,
            mission_id=work.mission_id, work_item_id=work.work_item_id,
        )
    child = runtime.submit_repair(
        tenant_ref=tenant, job_id="job_" + "1" * 24,
        workspace_sha256="1" * 64, projection=root.projection, round_index=1,
    )
    # A side item's accept completes nothing at all.
    assert runtime.complete(
        tenant_ref=tenant, job_id="job_" + "1" * 24,
        mission_id=root.mission_id, work_item_id=child.work_item_id,
        evidence={CRITERION_REVISION: "ab" * 32},
    ) is None
    assert runtime.store.get_mission(root.mission_id).status == MISSION_OPEN
    # The root's own accept is refused too, while the sibling is still open.
    assert runtime.complete(
        tenant_ref=tenant, job_id="job_" + "1" * 24,
        mission_id=root.mission_id, work_item_id=root.work_item_id,
        evidence={CRITERION_REVISION: "ab" * 32},
    ) is None
    assert runtime.store.get_mission(root.mission_id).status == MISSION_OPEN
    # Only once every item is closed does the axis complete.
    with runtime.dispatch() as work:
        assert work is not None and work.work_item_id == child.work_item_id
        runtime.close_fixed(
            work, tenant_ref=tenant, job_id=work.job_id,
            mission_id=work.mission_id, work_item_id=work.work_item_id,
        )
    completed = runtime.complete(
        tenant_ref=tenant, job_id="job_" + "1" * 24,
        mission_id=root.mission_id, work_item_id=root.work_item_id,
        evidence={CRITERION_REVISION: "ab" * 32},
    )
    assert completed is not None and completed.status == MISSION_COMPLETED


# --------------------------------------------------------------------------
# publication order: settled state and owner-closed item are one fact
# --------------------------------------------------------------------------


@needs_host
@pytest.mark.parametrize("audited", [True, False])
def test_no_settled_state_is_observable_before_its_item_is_owner_closed(
    tmp_path: Path, audited: bool,
) -> None:
    """The invariant, polled hostilely rather than waited out.

    A round used to publish `failed` or `awaiting_codex_audit` and only then
    close its work item, so a poller could catch a terminal job whose item this
    process still held dispatched. This hammers `get()` for the whole run and
    asserts that the *first* settled observation already carries a closed
    projection - and that the store agrees the item is closed.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    service = _service(
        state, workspace, provider=_Provider(delay=0.2, tag="sync"),
        require_codex_audit=audited,
    )
    try:
        receipt = service.submit(_TENANT, "key-1", _request(workspace))
        observations = 0
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            seen = service.get(_TENANT, receipt.job_id)
            observations += 1
            if seen.state not in _SETTLED:
                continue
            projection = CodingMissionProjection.from_mapping(seen.mission or {})
            # The record and the kernel must agree, at the very first settled
            # observation, with no sleep in between.
            assert projection.status == MISSION_STATUS_CLOSED, seen.state
            assert projection.disposition, seen.state
            item = _store(state).get_work_item(projection.work_item_id)
            assert item.status == STATUS_CLOSED
            assert item.disposition == projection.disposition
            break
        else:
            raise AssertionError("job never settled")
        assert observations > 1
    finally:
        service.close()


@needs_host
def test_an_audit_racing_an_unsettled_item_cannot_schedule(tmp_path: Path) -> None:
    """An audit-ready record whose item is not closed is refused, not acted on.

    The publication order makes this unreachable in a live round, so the record
    is edited in private tenant state to reproduce exactly what an older build
    could publish. Both verdicts must refuse: accepting would land a revision
    whose work item is still dispatched, and reworking would place a repair
    child under a parent that has not settled.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    service = _service(state, workspace, require_codex_audit=True)
    try:
        receipt = service.submit(_TENANT, "key-1", _request(workspace))
        ready = _wait(service, receipt.job_id)
        assert ready.state is CodingJobState.AWAITING_CODEX_AUDIT
        before = _items(state)

        path = (
            state / "tenants" / CodingService._tenant_ref(_TENANT)
            / "jobs" / (receipt.job_id + ".json")
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        record["mission"] = dict(record["mission"], status="dispatched", disposition="")
        path.write_text(json.dumps(record), encoding="utf-8")

        from flyto_ai.coding.service import AuditStateConflict

        for verdict, findings in (
            (CodingAuditVerdict.ACCEPT, ()),
            (CodingAuditVerdict.REWORK, (CodingAuditFinding(
                code="needs_more", message="do more work",
                severity=CodingAuditSeverity.MAJOR,
            ),)),
        ):
            with pytest.raises(AuditStateConflict):
                service.audit(
                    _TENANT, receipt.job_id,
                    ready.implementation_revision_sha256, verdict, findings,
                )
        # Nothing was scheduled: no repair child, no new work item at all.
        assert set(_items(state)) == set(before)
        assert service.get(_TENANT, receipt.job_id).rework_count == 0
    finally:
        service.close()


# --------------------------------------------------------------------------
# multi-worker contention: authority is execution, not admission
# --------------------------------------------------------------------------


@contextmanager
def _resource_held(state: Path, workspace: Path):
    """Hold one workspace's mission resource, so real jobs cannot dispatch.

    A deterministic stand-in for "some other worker is busy on this worktree",
    built out of the same kernel primitive the service itself uses. Nothing is
    mocked: the item really is dispatched and really does hold the claim.
    """

    blocker = CodingMissionRuntime(state, worker="w-blocker")
    digest = CodingService._workspace_digest(str(workspace))
    blocker.admit(
        tenant_ref="b" * 64, job_id="job_" + "b" * 24,
        workspace_sha256=digest, envelope=None, message="hold the worktree",
    )
    with blocker.dispatch() as held:
        assert held is not None
        assert blocker.is_workspace_claimed(digest)
        try:
            yield blocker
        finally:
            blocker.close_accounted(
                held, tenant_ref="b" * 64, job_id=held.job_id,
                mission_id=held.mission_id, work_item_id=held.work_item_id,
                disposition=DISPOSITION_BLOCKED,
                rationale="the test released this worktree",
                risk="none; this item never ran a round",
                evidence_refs=("test-blocker",),
            )


@needs_host
def test_several_instances_execute_each_job_once_without_burning_attempts(
    tmp_path: Path,
) -> None:
    """The thundering-herd proof, inverted into an assertion.

    Three current-build services share one state root and one workspace, and
    every job contends for that single exclusive resource. Admission no longer
    pins a job to the instance that accepted it, so the store's chosen worker is
    the one that runs it. Attempts and fencing tokens must therefore grow with
    completed jobs, not with how many pumps went looking.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    providers = [_Provider(tag="p{}".format(index)) for index in range(3)]
    services = [_service(state, workspace, provider=provider) for provider in providers]
    try:
        receipts = [
            service.submit(_TENANT, "key-{}".format(index), _request(
                workspace, message="task {}".format(index),
            ))
            for index, service in enumerate(services)
        ]
        for service, receipt in zip(services, receipts):
            assert _wait(service, receipt.job_id).state is CodingJobState.COMPLETED
        # Exactly one provider round per job, wherever it ran.
        assert sum(provider.rounds for provider in providers) == 3
        items = _items(state)
        assert len(items) == 3
        assert all(item.status == STATUS_CLOSED for item in items.values())
        # The strong assertion, restored: no item was ever dispatched twice.
        assert sorted(item.attempts for item in items.values()) == [1, 1, 1]
        # And fencing tokens are O(completed jobs), not O(pumps). Every burnt
        # token is one dispatch; a herd of refusals would show up here first.
        assert max(item.fence for item in items.values()) <= len(items) + 1
    finally:
        for service in services:
            service.close()


@needs_host
def test_queued_work_survives_the_submitter_and_is_run_by_another_worker(
    tmp_path: Path,
) -> None:
    """Queued is not interrupted, and a restart pumps it instead of failing it."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    with _resource_held(state, workspace):
        submitter = _service(state, workspace, provider=_Provider(tag="gone"))
        try:
            receipt = submitter.submit(_TENANT, "key-1", _request(workspace))
            assert submitter.get(_TENANT, receipt.job_id).state is CodingJobState.QUEUED
        finally:
            submitter.close()
        # The submitter is gone and the job is still queued, not failed.
        worker = _service(state, workspace, provider=_Provider(tag="worker"))
        after = worker.get(_TENANT, receipt.job_id)
        assert after.state is CodingJobState.QUEUED
        assert after.failure_code is None
    # Resource released; the surviving worker must complete it from durable
    # state alone - the job record, the resume envelope and the round envelope.
    try:
        settled = _wait(worker, receipt.job_id)
        assert settled.state is CodingJobState.COMPLETED
        assert (workspace / "notes-worker.txt").exists()
        item = _items(state)[
            CodingMissionProjection.from_mapping(settled.mission or {}).work_item_id
        ]
        assert item.status == STATUS_CLOSED and item.attempts == 1
    finally:
        worker.close()


@needs_host
def test_queued_job_reclaims_a_dispatch_lost_before_the_record_advanced(
    tmp_path: Path,
) -> None:
    """A crash between mission dispatch and job transition cannot strand work."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    with _resource_held(state, workspace):
        submitter = _service(state, workspace, provider=_Provider(tag="gone"))
        try:
            receipt = submitter.submit(_TENANT, "key-dispatch-crash", _request(workspace))
            assert submitter.get(_TENANT, receipt.job_id).state is CodingJobState.QUEUED
        finally:
            submitter.close()

    # Reproduce the exact durable split: MissionStore committed `dispatched`,
    # then the process vanished before CodingService could write `running`.
    script = (
        "import os\n"
        "from flyto_ai.coding.mission_runtime import CodingMissionRuntime\n"
        "runtime = CodingMissionRuntime({!r}, worker='w-crashed')\n"
        "with runtime.dispatch() as work:\n"
        "    assert work is not None\n"
        "    os._exit(0)\n"
    ).format(str(state))
    subprocess.run([sys.executable, "-c", script], check=True)
    stranded = _items(state)
    work_item_id = CodingMissionProjection.from_mapping(
        receipt.mission or {},
    ).work_item_id
    item = stranded[work_item_id]
    assert item.status == STATUS_DISPATCHED
    assert item.attempts == 1

    provider = _Provider(tag="recovered")
    worker = _service(state, workspace, provider=provider)
    try:
        settled = _wait(worker, receipt.job_id)
        assert settled.state is CodingJobState.COMPLETED
        assert provider.rounds == 1
        recovered = _items(state)[item.work_item_id]
        assert recovered.status == STATUS_CLOSED
        assert recovered.attempts == 2
    finally:
        worker.close()


def _stranger(state: Path, workspace: Path, **overrides: Any) -> CodingService:
    fields: Dict[str, Any] = {
        "state_root": str(state),
        "workspace_roots": (str(workspace),),
        "max_workers": 2,
        "max_queued": 8,
        "implementation_backend": "some-other-backend",
    }
    fields.update(overrides)
    return CodingService(
        lambda store: FlytoCodingAgent(_Provider(tag="stranger"), store=store),
        **fields,
    )


@needs_host
def test_an_incompatible_service_is_refused_before_it_can_sweep_or_pump(
    tmp_path: Path,
) -> None:
    """One semantic startup authority owns a state root while work is live.

    Refusal at construction is what makes "does not burn attempts" an invariant
    rather than a budget: a service that never starts is never offered an item,
    so it can neither consume a dispatch attempt nor sweep a workspace claim it
    has no standing over.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    with _resource_held(state, workspace):
        owner = _service(
            state, workspace, provider=_Provider(tag="owner2"),
            require_codex_audit=True,
        )
        receipt = owner.submit(_TENANT, "key-1", _request(workspace))
        projection = CodingMissionProjection.from_mapping(receipt.mission or {})
        claim = owner._workspace_claim_path(str(workspace))
        assert claim.exists()
        owner.close()

        for overrides in (
            {"implementation_backend": "some-other-backend"},
            {"implementation_backend": "native", "require_codex_audit": False},
            {"implementation_backend": "native", "config_path": ".flyto/other.yaml"},
            {"implementation_backend": "native", "max_rework_rounds": 2},
        ):
            with pytest.raises(CodingAuthorityConflict):
                _stranger(
                    state, workspace,
                    **dict({"require_codex_audit": True}, **overrides),
                )
        # Nothing was consumed and nothing was released by the refusals.
        item = _store(state).get_work_item(projection.work_item_id)
        assert item.attempts == 0 and item.fence == 0
        assert item.status == STATUS_READY
        assert claim.exists()
        assert not (workspace / "notes-stranger.txt").exists()

    # A compatible heir finishes it from durable state alone.
    heir = _service(
        state, workspace, provider=_Provider(tag="heir"), require_codex_audit=True,
    )
    try:
        settled = _wait(heir, receipt.job_id)
        assert settled.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert (workspace / "notes-heir.txt").exists()
        assert _store(state).get_work_item(projection.work_item_id).attempts == 1
    finally:
        heir.close()


@needs_host
def test_two_different_authorities_cannot_coexist_on_an_empty_root(
    tmp_path: Path,
) -> None:
    """The race a record scan could never see.

    An empty root has no job records, so an inference from history lets both
    services construct - and whichever admits first leaves the other one alive,
    able to submit and pump against an authority it does not share. The lease is
    what makes the claim true rather than aspirational.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    first = _service(state, workspace, provider=_Provider(tag="bind1"))
    try:
        marker = json.loads(
            (state / AUTHORITY_MARKER_NAME).read_text(encoding="utf-8"),
        )
        assert marker["marker_version"] == AUTHORITY_MARKER_VERSION
        assert marker["authority"] == first._execution_authority()
        # Secret-free and path-free, like the record fingerprint it mirrors.
        rendered = json.dumps(marker)
        assert str(state) not in rendered and str(workspace) not in rendered
        # No job has ever existed here, and the second authority is still
        # refused - because a live holder, not a history, is what binds a root.
        with pytest.raises(CodingAuthorityConflict):
            _stranger(state, workspace)
    finally:
        first.close()


@needs_host
def test_several_same_authority_services_coexist_on_one_root(
    tmp_path: Path,
) -> None:
    """Shared holders, so peers on one queue are the normal case."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    peers = [
        _service(state, workspace, provider=_Provider(tag="peer{}".format(index)))
        for index in range(3)
    ]
    try:
        assert all(peer._authority_fd >= 0 for peer in peers)
        receipt = peers[0].submit(_TENANT, "key-1", _request(workspace))
        assert _wait(peers[2], receipt.job_id).state is CodingJobState.COMPLETED
    finally:
        for peer in peers:
            peer.close()


@needs_host
def test_rotation_is_refused_while_an_old_service_is_still_live(
    tmp_path: Path,
) -> None:
    """Terminal jobs are not enough; the old authority must actually be gone."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    owner = _service(state, workspace, provider=_Provider(tag="rot1"))
    try:
        receipt = owner.submit(_TENANT, "key-1", _request(workspace))
        assert _wait(owner, receipt.job_id).state is CodingJobState.COMPLETED
        # Every job is terminal, and rotation is *still* refused: the old
        # service is alive and holds its share of the lease.
        with pytest.raises(CodingAuthorityConflict):
            _stranger(state, workspace)
        marker = json.loads(
            (state / AUTHORITY_MARKER_NAME).read_text(encoding="utf-8"),
        )
        assert marker["authority"] == owner._execution_authority()
    finally:
        owner.close()
    # Old service gone, all work terminal: rotation succeeds and rebinds.
    rotated = _stranger(state, workspace)
    try:
        assert rotated.implementation_backend == "some-other-backend"
        marker = json.loads(
            (state / AUTHORITY_MARKER_NAME).read_text(encoding="utf-8"),
        )
        assert marker["authority"] == rotated._execution_authority()
    finally:
        rotated.close()


@needs_host
def test_rotation_is_refused_while_work_is_open_even_with_no_live_service(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    with _resource_held(state, workspace):
        owner = _service(state, workspace, provider=_Provider(tag="open1"))
        owner.submit(_TENANT, "key-1", _request(workspace))
        owner.close()
        # Nobody is alive, so the exclusive lock is available - and rotation is
        # still refused, because the old authority's work is still open.
        with pytest.raises(CodingAuthorityConflict):
            _stranger(state, workspace)


@needs_host
def test_a_refused_startup_never_rewrites_a_lost_marker(tmp_path: Path) -> None:
    """Validation happens before the marker is written, not after.

    Writing first meant a stranger could win the exclusive lock, replace a lost
    marker with its own, and only *then* fail on the open job - leaving the root
    bound to an authority whose construction had failed and locking out the
    worker that was actually correct.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    with _resource_held(state, workspace):
        owner = _service(state, workspace, provider=_Provider(tag="lost"))
        receipt = owner.submit(_TENANT, "key-1", _request(workspace))
        owner_authority = owner._execution_authority()
        owner.close()
        # The marker is lost, but an open job still carries the old authority.
        (state / AUTHORITY_MARKER_NAME).unlink()
        assert not (state / AUTHORITY_MARKER_NAME).exists()

        with pytest.raises(CodingAuthorityConflict):
            _stranger(state, workspace)
        # Refused, and it did not mint a marker on the way out.
        assert not (state / AUTHORITY_MARKER_NAME).exists()

        # The correct worker is not locked out: it rebinds the root itself.
        heir = _service(state, workspace, provider=_Provider(tag="heir2"))
        try:
            marker = json.loads(
                (state / AUTHORITY_MARKER_NAME).read_text(encoding="utf-8"),
            )
            assert marker["authority"] == owner_authority
            assert heir.get(_TENANT, receipt.job_id).state is CodingJobState.QUEUED
        finally:
            heir.close()


@needs_host
@pytest.mark.parametrize(
    "body",
    [
        "{not json",
        json.dumps({"marker_version": "wrong", "authority": {}}),
        json.dumps({"marker_version": AUTHORITY_MARKER_VERSION}),
        json.dumps({"marker_version": AUTHORITY_MARKER_VERSION, "authority": 5}),
        json.dumps({
            "marker_version": AUTHORITY_MARKER_VERSION,
            "authority": {},
            "extra": 1,
        }),
    ],
)
def test_a_malformed_marker_is_a_refusal_never_an_absence(
    tmp_path: Path, body: str,
) -> None:
    """Damaged state is not permission to rebind the root."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    _service(state, workspace, provider=_Provider(tag="seed")).close()
    marker = state / AUTHORITY_MARKER_NAME
    marker.write_text(body, encoding="utf-8")

    with pytest.raises(CodingAuthorityConflict):
        _service(state, workspace, provider=_Provider(tag="after"))
    # Byte-identical: a refusal repairs nothing and overwrites nothing.
    assert marker.read_text(encoding="utf-8") == body


@needs_host
def test_a_marker_that_is_not_a_regular_file_is_refused(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    _service(state, workspace, provider=_Provider(tag="seed2")).close()
    marker = state / AUTHORITY_MARKER_NAME
    marker.unlink()
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("{}", encoding="utf-8")
    marker.symlink_to(elsewhere)

    with pytest.raises(CodingAuthorityConflict):
        _service(state, workspace, provider=_Provider(tag="after2"))
    # Refused rather than followed: the link and its target are untouched.
    assert marker.is_symlink()
    assert elsewhere.read_text(encoding="utf-8") == "{}"


@needs_host
def test_an_oversized_marker_is_refused_without_being_parsed(
    tmp_path: Path,
) -> None:
    """A bounded reader refuses a body it will not read, rather than reading it."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    _service(state, workspace, provider=_Provider(tag="size")).close()
    marker = state / AUTHORITY_MARKER_NAME
    # Structurally valid JSON, and far too large to be this marker.
    body = json.dumps({
        "marker_version": AUTHORITY_MARKER_VERSION,
        "authority": {"padding": "x" * (MAX_AUTHORITY_MARKER_BYTES * 2)},
    })
    assert len(body.encode()) > MAX_AUTHORITY_MARKER_BYTES
    marker.write_text(body, encoding="utf-8")

    with pytest.raises(CodingAuthorityConflict):
        _service(state, workspace, provider=_Provider(tag="size2"))
    assert marker.read_text(encoding="utf-8") == body


@needs_host
def test_the_marker_is_read_through_one_descriptor_not_a_name(
    tmp_path: Path,
) -> None:
    """Every question is asked of the descriptor, so no name can be swapped.

    An `lstat` followed by a separate open checks one file and reads another
    whenever the name is replaced in between. These are the shapes that
    substitution produces, and each must be refused rather than followed.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    service = _service(state, workspace, provider=_Provider(tag="fd"))
    marker = state / AUTHORITY_MARKER_NAME
    try:
        good = marker.read_text(encoding="utf-8")
        # A regular file reads normally through the descriptor.
        assert service._read_authority_marker() == service._execution_authority()

        # Replaced by a symbolic link to an otherwise valid marker: refused by
        # `O_NOFOLLOW` at open time, so the target is never even read.
        decoy = tmp_path / "decoy.json"
        decoy.write_text(good, encoding="utf-8")
        marker.unlink()
        marker.symlink_to(decoy)
        with pytest.raises(CodingAuthorityConflict):
            service._read_authority_marker()
        assert marker.is_symlink()

        # Replaced by a directory: refused by the `fstat` on that descriptor.
        marker.unlink()
        marker.mkdir()
        with pytest.raises(CodingAuthorityConflict):
            service._read_authority_marker()
        marker.rmdir()

        # Absent is still absent, and is the only thing that reads as `None`.
        assert service._read_authority_marker() is None
    finally:
        service.close()


@needs_host
def test_an_unreadable_job_record_refuses_startup_and_rotation(
    tmp_path: Path,
) -> None:
    """An unreadable record is not evidence that its job finished."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    owner = _service(state, workspace, provider=_Provider(tag="corrupt"))
    try:
        receipt = owner.submit(_TENANT, "key-1", _request(workspace))
        assert _wait(owner, receipt.job_id).state is CodingJobState.COMPLETED
    finally:
        owner.close()
    marker_before = (state / AUTHORITY_MARKER_NAME).read_text(encoding="utf-8")
    record_path = (
        state / "tenants" / CodingService._tenant_ref(_TENANT)
        / "jobs" / (receipt.job_id + ".json")
    )
    record_path.write_text("{ truncated", encoding="utf-8")

    # Same authority: refused, because the record cannot be shown to be closed.
    with pytest.raises(CodingAuthorityConflict):
        _service(state, workspace, provider=_Provider(tag="same"))
    # Rotation: refused for the same reason, and the marker survives intact.
    with pytest.raises(CodingAuthorityConflict):
        _stranger(state, workspace)
    assert (state / AUTHORITY_MARKER_NAME).read_text(
        encoding="utf-8",
    ) == marker_before


@needs_host
@pytest.mark.parametrize("failure", ["close_status", "shutdown", "job_leases"])
def test_a_failing_teardown_step_still_releases_both_leases(
    tmp_path: Path, failure: str,
) -> None:
    """No teardown step may keep a stopped service holding the root.

    The `finally` used to sit *inside* the teardown, after the executor drain
    and the job-lease release, so a failure in either of those earlier steps
    leaked both root descriptors and locked the state root against every later
    service - a lock-out invisible in the error the caller actually saw.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    owner = _service(state, workspace, provider=_Provider(tag="teardown"))

    def _explode(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("teardown step is broken")

    if failure == "close_status":
        owner._close_status = _explode  # type: ignore[method-assign]
        owner.close()
    elif failure == "shutdown":
        owner._executor.shutdown = _explode  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            owner.close()
    else:
        owner._acquire_job_lease("job_" + "f" * 24)
        owner._release_job_lease = _explode  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            owner.close()
    assert owner._authority_fd == -1
    # Both leases came back despite the injected failure, so another authority
    # can rotate a root whose every job is terminal.
    rotated = _stranger(state, workspace)
    try:
        marker = json.loads(
            (state / AUTHORITY_MARKER_NAME).read_text(encoding="utf-8"),
        )
        assert marker["authority"] == rotated._execution_authority()
    finally:
        rotated.close()


def test_a_host_without_flock_refuses_rather_than_pretending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No inter-process lock, no isolation claim - and therefore no service."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    import flyto_ai.coding.service as service_module

    monkeypatch.setattr(service_module, "fcntl", None)
    with pytest.raises(CodingAuthorityUnavailable) as refusal:
        _service(tmp_path / "state", workspace, provider=_Provider(tag="nolock"))
    assert refusal.value.code == "execution_authority_unavailable"
    assert refusal.value.retryable is False
    # Nothing was bound, because nothing could be.
    assert not (tmp_path / "state" / AUTHORITY_MARKER_NAME).exists()


@needs_host
def test_a_released_descriptor_recovers_the_root_without_a_ttl(
    tmp_path: Path,
) -> None:
    """A crash is a closed descriptor, and the kernel is the only clock."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    crashed = _service(state, workspace, provider=_Provider(tag="crash"))
    receipt = crashed.submit(_TENANT, "key-1", _request(workspace))
    assert _wait(crashed, receipt.job_id).state is CodingJobState.COMPLETED
    # Simulate the process dying: the descriptor goes, nothing else runs.
    assert crashed._authority_fd >= 0
    crashed._release_state_root_authority()
    try:
        # No TTL elapsed, no heartbeat was missed; the lease is simply free.
        rotated = _stranger(state, workspace)
        rotated.close()
    finally:
        crashed.close()


@needs_host
def test_a_refused_startup_changes_nothing_it_touched(tmp_path: Path) -> None:
    """A service that must not run must also not have run."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    with _resource_held(state, workspace):
        owner = _service(
            state, workspace, provider=_Provider(tag="intact"),
            require_codex_audit=True,
        )
        try:
            receipt = owner.submit(_TENANT, "key-1", _request(workspace))
            projection = CodingMissionProjection.from_mapping(receipt.mission or {})
            claim = owner._workspace_claim_path(str(workspace))
            before_item = _store(state).get_work_item(projection.work_item_id)
            before_record = json.loads((
                state / "tenants" / CodingService._tenant_ref(_TENANT)
                / "jobs" / (receipt.job_id + ".json")
            ).read_text(encoding="utf-8"))
            before_marker = (state / AUTHORITY_MARKER_NAME).read_text(encoding="utf-8")

            for _ in range(3):
                with pytest.raises(CodingAuthorityConflict):
                    _stranger(state, workspace, require_codex_audit=True)

            after_item = _store(state).get_work_item(projection.work_item_id)
            assert after_item.attempts == before_item.attempts == 0
            assert after_item.fence == before_item.fence == 0
            assert after_item.status == STATUS_READY
            assert claim.exists()
            assert json.loads((
                state / "tenants" / CodingService._tenant_ref(_TENANT)
                / "jobs" / (receipt.job_id + ".json")
            ).read_text(encoding="utf-8")) == before_record
            assert (state / AUTHORITY_MARKER_NAME).read_text(
                encoding="utf-8",
            ) == before_marker
        finally:
            owner.close()


@needs_host
def test_a_nested_route_change_is_incompatible_but_a_new_build_is_not(
    tmp_path: Path,
) -> None:
    """The fingerprint hashes the whole validated policy, not its top level."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    service = _service(state, workspace)
    try:
        base = service._execution_authority()
        # A build id is not part of the fingerprint at all.
        service.build_id = service.build_id + "-reloaded"
        assert service._execution_authority() == base
        assert service._may_execute({"execution_authority": base}) is True

        limits = SimpleNamespace()

        @dataclasses.dataclass(frozen=True)
        class _Limits:
            max_seconds: int = 30
            max_files: int = 10

        @dataclasses.dataclass(frozen=True)
        class _Lane:
            enabled: bool = True
            limits: _Limits = dataclasses.field(default_factory=_Limits)

        @dataclasses.dataclass(frozen=True)
        class _Route:
            strict: bool = True
            indexer: _Lane = dataclasses.field(default_factory=_Lane)
            blueprint: _Lane = dataclasses.field(default_factory=_Lane)

        del limits
        outer = CodingService._policy_digest(_Route())
        # A change buried two levels down must move the digest. Hashing only the
        # top level made an Indexer, Blueprint or RouteLimits change invisible.
        nested = CodingService._policy_digest(
            _Route(indexer=_Lane(limits=_Limits(max_seconds=31))),
        )
        deeper = CodingService._policy_digest(
            _Route(blueprint=_Lane(enabled=False)),
        )
        assert len({outer, nested, deeper}) == 3
        assert len(outer) == 32
        assert CodingService._policy_digest(None) == "none"
    finally:
        service.close()


@needs_host
def test_a_lane_that_runs_a_different_binary_is_a_different_authority() -> None:
    """Execution identity survives the digest; it is not normalized away.

    An earlier version folded every path-shaped string to a placeholder, so two
    capability policies invoking `/opt/indexer-v1` and `/opt/indexer-v2` hashed
    identically and were allowed to share one state root while running different
    binaries. A state root and its `flock` are host-local, so cross-host
    pathname stability was never a reason to erase that.
    """

    @dataclasses.dataclass(frozen=True)
    class _Capability:
        name: str = "indexer"
        argv: tuple = ("/opt/indexer-v1/bin/indexer", "--strict")

    @dataclasses.dataclass(frozen=True)
    class _Lane:
        enabled: bool = True
        capability: _Capability = dataclasses.field(default_factory=_Capability)

    @dataclasses.dataclass(frozen=True)
    class _Route:
        strict: bool = True
        indexer: _Lane = dataclasses.field(default_factory=_Lane)

    baseline = CodingService._policy_digest(_Route())
    # A different checkout of the same tool, differing *only* in the path.
    other_checkout = CodingService._policy_digest(_Route(indexer=_Lane(
        capability=_Capability(argv=("/opt/indexer-v2/bin/indexer", "--strict")),
    )))
    # A different executable entirely, nested two levels down.
    other_binary = CodingService._policy_digest(_Route(indexer=_Lane(
        capability=_Capability(argv=("/opt/indexer-v1/bin/other", "--strict")),
    )))
    # And a relative path, which the old rule also folded away.
    relative = CodingService._policy_digest(_Route(indexer=_Lane(
        capability=_Capability(argv=("bin/indexer", "--strict")),
    )))
    assert len({baseline, other_checkout, other_binary, relative}) == 4
    # Still bounded, and still only a digest.
    for digest in (baseline, other_checkout, other_binary, relative):
        assert len(digest) == 32
        assert all(character in "0123456789abcdef" for character in digest)
        assert "/opt" not in digest and "indexer" not in digest


@needs_host
def test_a_path_only_route_change_refuses_a_shared_state_root(
    tmp_path: Path,
) -> None:
    """The collision, proved end to end against a real state root."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"

    @dataclasses.dataclass(frozen=True)
    class _Lane:
        argv: tuple = ("/opt/indexer-v1/bin/indexer",)

    owner = _service(state, workspace, provider=_Provider(tag="pathauth"))
    try:
        # Two policies that differ only in a path now fingerprint differently,
        # so a service carrying the second cannot adopt the first's records.
        first = dict(owner._execution_authority())
        second = dict(
            first, route=CodingService._policy_digest(
                _Lane(argv=("/opt/indexer-v2/bin/indexer",)),
            ),
        )
        assert first["route"] != second["route"]
        assert owner._may_execute({"execution_authority": first}) is True
        assert owner._may_execute({"execution_authority": second}) is False

        # A build id still changes nothing: it is not in the fingerprint.
        owner.build_id = owner.build_id + "-reloaded"
        assert dict(owner._execution_authority()) == first
        assert owner._may_execute({"execution_authority": first}) is True

        # The marker on disk stays bounded and carries only digests.
        marker_text = (state / AUTHORITY_MARKER_NAME).read_text(encoding="utf-8")
        assert len(marker_text.encode()) <= MAX_AUTHORITY_MARKER_BYTES
        assert "/opt" not in marker_text
        assert str(workspace) not in marker_text and str(state) not in marker_text
        marker = json.loads(marker_text)
        assert len(marker["authority"]["route"]) in (4, 32)
        assert len(marker["authority"]["emergency"]) in (4, 32)
    finally:
        owner.close()


@needs_host
def test_a_pre_upgrade_record_is_migrated_or_terminalized_never_requeued(
    tmp_path: Path,
) -> None:
    """A record with no fingerprint must not circle the queue forever."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    with _resource_held(state, workspace):
        owner = _service(state, workspace, provider=_Provider(tag="pre"))
        receipt = owner.submit(_TENANT, "key-1", _request(workspace))
        path = (
            state / "tenants" / CodingService._tenant_ref(_TENANT)
            / "jobs" / (receipt.job_id + ".json")
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        record.pop("execution_authority")
        path.write_text(json.dumps(record), encoding="utf-8")
        owner.close()

        # No implementer has run, so nothing contradicts adoption: migrate.
        adopter = _service(state, workspace, provider=_Provider(tag="adopt"))
        migrated = json.loads(path.read_text(encoding="utf-8"))
        assert migrated["execution_authority"] == adopter._execution_authority()
        assert migrated["state"] == CodingJobState.QUEUED.value
    try:
        assert _wait(adopter, receipt.job_id).state is CodingJobState.COMPLETED
    finally:
        adopter.close()

    # A queued record that has already entered an implementer is *not* the
    # no-execution case, whatever its backend field says.
    other = tmp_path / "state2"
    with _resource_held(other, workspace):
        owner = _service(other, workspace, provider=_Provider(tag="pre2"))
        stranded = owner.submit(_TENANT, "key-2", _request(workspace))
        path = (
            other / "tenants" / CodingService._tenant_ref(_TENANT)
            / "jobs" / (stranded.job_id + ".json")
        )
        record = json.loads(path.read_text(encoding="utf-8"))
        record.pop("execution_authority")
        record["implementer_started"] = True
        path.write_text(json.dumps(record), encoding="utf-8")
        owner.close()

        settler = _service(other, workspace, provider=_Provider(tag="settle"))
        try:
            after = settler.get(_TENANT, stranded.job_id)
            assert after.state is CodingJobState.FAILED
            assert after.failure_code == EXECUTION_AUTHORITY_UNBOUND
            assert after.landable is False
        finally:
            settler.close()


@needs_host
def test_a_v0_executing_record_is_never_adopted_on_an_empty_backend(
    tmp_path: Path,
) -> None:
    """`implementation_backend` is written on outcome, so absence proves nothing.

    A `running` record with no backend may already have entered a provider. It
    is never stamped with this service's authority; it is either left to its
    live lease holder or, once the lease is provably free, terminalized with its
    accounting closed.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    runtime = CodingMissionRuntime(state, worker="w-v0")
    tenant_ref = CodingService._tenant_ref(_TENANT)
    placed = runtime.admit(
        tenant_ref=tenant_ref, job_id="job_" + "c" * 24,
        workspace_sha256="c" * 64, envelope=None, message="v0 running",
    )
    jobs = state / "tenants" / tenant_ref / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    path = jobs / ("job_" + "c" * 24 + ".json")
    now = time.time()
    path.write_text(json.dumps({
        "job_id": "job_" + "c" * 24,
        "state": CodingJobState.RUNNING.value,
        "submitted_at": now,
        "updated_at": now,
        "working_dir": str(workspace),
        "implementation_backend": "",
        "mission": placed.projection.to_mapping(),
    }), encoding="utf-8")

    # A live holder of the job lease keeps it. The lock is taken directly,
    # before any service exists, so nothing has had a chance to settle it.
    import fcntl

    lease_dir = state / "locks" / "jobs"
    lease_dir.mkdir(parents=True, exist_ok=True)
    lease_fd = os.open(
        lease_dir / ("job_" + "c" * 24 + ".lock"), os.O_CREAT | os.O_RDWR, 0o600,
    )
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    marker_before = (
        (state / AUTHORITY_MARKER_NAME).read_text(encoding="utf-8")
        if (state / AUTHORITY_MARKER_NAME).exists() else None
    )
    try:
        # A live round nobody can attribute. Starting beside it would be
        # exactly the unattributable execution the fingerprint exists to
        # prevent, so the new service refuses instead of coexisting.
        with pytest.raises(CodingAuthorityConflict):
            _service(state, workspace, provider=_Provider(tag="peer"))
        record = json.loads(path.read_text(encoding="utf-8"))
        # Untouched: not stolen, not stamped, not terminalized.
        assert "execution_authority" not in record
        assert record["state"] == CodingJobState.RUNNING.value
        item = _store(state).get_work_item(placed.work_item_id)
        assert item.status == STATUS_READY and item.attempts == 0
        after = (
            (state / AUTHORITY_MARKER_NAME).read_text(encoding="utf-8")
            if (state / AUTHORITY_MARKER_NAME).exists() else None
        )
        assert after == marker_before
    finally:
        fcntl.flock(lease_fd, fcntl.LOCK_UN)
        os.close(lease_fd)

    # Lease provably free: terminalized with a stable code, item accounted.
    settler = _service(state, workspace, provider=_Provider(tag="settle2"))
    try:
        after = settler.get(_TENANT, "job_" + "c" * 24)
        assert after.state is CodingJobState.FAILED
        assert after.failure_code == EXECUTION_AUTHORITY_UNBOUND
        record = json.loads(path.read_text(encoding="utf-8"))
        assert "execution_authority" not in record
    finally:
        settler.close()


@needs_host
def test_a_v0_awaiting_record_accepts_but_never_reworks(tmp_path: Path) -> None:
    """Accept describes a hashed revision; rework would adopt an unknown route."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    for index, verdict in enumerate((
        CodingAuditVerdict.ACCEPT, CodingAuditVerdict.REWORK,
    )):
        state = tmp_path / "state{}".format(index)
        service = _service(state, workspace, require_codex_audit=True)
        try:
            receipt = service.submit(_TENANT, "key-1", _request(workspace))
            ready = _wait(service, receipt.job_id)
            assert ready.state is CodingJobState.AWAITING_CODEX_AUDIT
            path = (
                state / "tenants" / CodingService._tenant_ref(_TENANT)
                / "jobs" / (receipt.job_id + ".json")
            )
        finally:
            service.close()
        record = json.loads(path.read_text(encoding="utf-8"))
        record.pop("execution_authority")
        path.write_text(json.dumps(record), encoding="utf-8")

        heir = _service(state, workspace, require_codex_audit=True)
        try:
            # Start-up leaves an awaiting record exactly as it found it.
            assert heir.get(_TENANT, receipt.job_id).state is (
                CodingJobState.AWAITING_CODEX_AUDIT
            )
            assert "execution_authority" not in json.loads(
                path.read_text(encoding="utf-8"),
            )
            findings = () if verdict is CodingAuditVerdict.ACCEPT else (
                CodingAuditFinding(
                    code="needs_more", message="do more work",
                    severity=CodingAuditSeverity.MAJOR,
                ),
            )
            if verdict is CodingAuditVerdict.ACCEPT:
                accepted = heir.audit(
                    _TENANT, receipt.job_id,
                    ready.implementation_revision_sha256, verdict, findings,
                )
                assert accepted.state is CodingJobState.CODEX_ACCEPTED
            else:
                with pytest.raises(CodingAuthorityConflict):
                    heir.audit(
                        _TENANT, receipt.job_id,
                        ready.implementation_revision_sha256, verdict, findings,
                    )
                settled = heir.get(_TENANT, receipt.job_id)
                assert settled.state is CodingJobState.FAILED
                assert settled.failure_code == EXECUTION_AUTHORITY_UNBOUND
                assert settled.rework_count == 0
        finally:
            heir.close()


def test_the_authority_conflict_is_public_package_surface() -> None:
    """Callers branch on this by type, so it binds to the package, not a module."""

    import flyto_ai.coding as coding
    from flyto_ai.coding.service import CodingAuthorityConflict as internal

    assert "CodingAuthorityConflict" in coding.__all__
    assert coding.CodingAuthorityConflict is internal
    assert issubclass(coding.CodingAuthorityConflict, coding.CodingServiceError)
    assert coding.CodingAuthorityConflict("x").code == "execution_authority_conflict"
    assert coding.CodingAuthorityConflict("x").retryable is False


@needs_host
def test_a_changed_build_id_does_not_strand_semantically_identical_work(
    tmp_path: Path,
) -> None:
    """Compatibility is semantic, not a build stamp.

    A hot reload or an ordinary restart mints a new build id. Binding execution
    to it would strand a queued job that an otherwise identical worker can run
    perfectly well - the fingerprint's own failure mode, inverted.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    with _resource_held(state, workspace):
        submitter = _service(state, workspace, provider=_Provider(tag="old-build"))
        try:
            receipt = submitter.submit(_TENANT, "key-1", _request(workspace))
            assert submitter.get(_TENANT, receipt.job_id).state is CodingJobState.QUEUED
            admitted_build = submitter.build_id
            record = json.loads((
                state / "tenants" / CodingService._tenant_ref(_TENANT)
                / "jobs" / (receipt.job_id + ".json")
            ).read_text(encoding="utf-8"))
            # The fingerprint never carried the build id in the first place.
            assert "build_id" not in record["execution_authority"]
        finally:
            submitter.close()

        worker = _service(state, workspace, provider=_Provider(tag="new-build"))
        # A different build, same semantics, and therefore still compatible.
        worker.build_id = admitted_build + "-reloaded"
        assert worker.build_id != admitted_build
        assert worker._may_execute(record) is True
    try:
        settled = _wait(worker, receipt.job_id)
        assert settled.state is CodingJobState.COMPLETED
        item = _items(state)[
            CodingMissionProjection.from_mapping(settled.mission or {}).work_item_id
        ]
        assert item.attempts == 1
    finally:
        worker.close()


@needs_host
def test_a_peer_cannot_account_work_while_admission_is_still_committing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ready mission item is not proof that its private record is orphaned.

    Mission placement intentionally precedes the private job and round
    envelopes. A peer may dispatch in that bounded window, but the admitting
    service still holds the job lease. The peer must therefore requeue the item
    without a closure, then run it normally after admission commits. Its
    attempt and fence still record the real scheduling collision.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    submitter = _service(state, workspace, provider=_Provider(tag="submitter"))
    peer = _service(state, workspace, provider=_Provider(tag="peer"))
    original_admit = submitter._admit_mission
    observed: dict[str, str] = {}

    def dispatch_between_mission_and_record(*args, **kwargs):
        admission = original_admit(*args, **kwargs)
        observed["work_item_id"] = admission.work_item_id
        assert peer._dispatch_once() == _DISPATCH_FOREIGN
        item = _store(state).get_work_item(admission.work_item_id)
        assert item.status == STATUS_READY
        assert item.attempts == 1 and item.fence == 1
        return admission

    monkeypatch.setattr(
        submitter, "_admit_mission", dispatch_between_mission_and_record,
    )
    try:
        receipt = submitter.submit(_TENANT, "key-1", _request(workspace))
        assert observed["work_item_id"] == (
            CodingMissionProjection.from_mapping(
                receipt.mission or {},
            ).work_item_id
        )
        settled = _wait(peer, receipt.job_id)
        assert settled.state is CodingJobState.COMPLETED
        item = _store(state).get_work_item(observed["work_item_id"])
        assert item.status == STATUS_CLOSED
        assert item.attempts == 2
    finally:
        submitter.close()
        peer.close()


@needs_host
def test_admission_releases_its_lease_before_any_pump_can_dispatch(
    tmp_path: Path,
) -> None:
    """Ordering, asserted from the durable state a pump would have to read.

    Submitting the pump before releasing the lease let this instance's own
    worker dispatch its own item, refuse it as leased, and requeue it - burning
    an attempt on its own job. `submit` must therefore return with the lease
    already gone, and the item must still be untouched at `ready`.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    with _resource_held(state, workspace):
        service = _service(state, workspace, provider=_Provider(tag="order"))
        try:
            receipt = service.submit(_TENANT, "key-1", _request(workspace))
            # The lease is not held by anybody the moment submit returns...
            assert receipt.job_id not in service._job_leases
            assert service._claim_round(receipt.job_id) is True
            service._release_round(receipt.job_id)
            # ...and no pump has burnt an attempt on it.
            projection = CodingMissionProjection.from_mapping(receipt.mission or {})
            item = _store(state).get_work_item(projection.work_item_id)
            assert item.status == STATUS_READY
            assert item.attempts == 0 and item.fence == 0
        finally:
            pass
    try:
        settled = _wait(service, receipt.job_id)
        assert settled.state is CodingJobState.COMPLETED
        item = _store(state).get_work_item(projection.work_item_id)
        assert item.attempts == 1
    finally:
        service.close()


@needs_host
def test_dispatch_waits_behind_a_live_admission_before_accounting_absence(
    tmp_path: Path,
) -> None:
    """A placed item is not orphaned until its admission lease is provably free."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    tenant_ref = CodingService._tenant_ref(_TENANT)
    job_id = "job_" + "b" * 24
    placed = CodingMissionRuntime(state, worker="admitter").admit(
        tenant_ref=tenant_ref, job_id=job_id, workspace_sha256="b" * 64,
        envelope=None, message="admission still committing",
    )

    import fcntl

    lease_dir = state / "locks" / "jobs"
    lease_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lease_dir / (job_id + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    service = _service(state, workspace, provider=_Provider(tag="barrier"))
    try:
        assert service._dispatch_once() == "foreign"
        collided = _store(state).get_work_item(placed.work_item_id)
        assert collided.status == STATUS_READY
        assert collided.attempts == 1 and collided.closure is None

        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        descriptor = -1
        assert service._dispatch_once() == "ran"
        accounted = _store(state).get_work_item(placed.work_item_id)
        assert accounted.status == STATUS_CLOSED
        assert accounted.disposition == DISPOSITION_BLOCKED
        assert accounted.attempts == 2
    finally:
        if descriptor >= 0:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        service.close()


@needs_host
def test_missing_round_envelope_is_accounted_only_after_the_round_lease(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    provider = _Provider(tag="missing-envelope")
    service = _service(state, workspace, provider=provider)
    service._schedule_pump = lambda: None  # type: ignore[method-assign]
    try:
        receipt = service.submit(_TENANT, "key-1", _request(workspace))
        projection = CodingMissionProjection.from_mapping(receipt.mission or {})
        service._discard_path(
            service._round_path(CodingService._tenant_ref(_TENANT), projection.work_item_id),
        )
        assert service._dispatch_once() == "ran"
        settled = service.get(_TENANT, receipt.job_id)
        assert settled.state is CodingJobState.FAILED
        assert settled.failure_code == "round_envelope_unbound"
        item = _store(state).get_work_item(projection.work_item_id)
        assert item.status == STATUS_CLOSED and item.disposition == DISPOSITION_BLOCKED
        assert provider.rounds == 0
    finally:
        service.close()


@needs_host
def test_a_stale_round_finally_cannot_release_a_reacquired_job_lease(
    tmp_path: Path,
) -> None:
    """A terminal publish and its round ``finally`` cannot ABA a new owner.

    Terminal settlement deliberately releases the execution lease before the
    worker future returns. Another operation may then acquire the same job
    lease while the old round is still unwinding. The old round must release
    only the token it acquired, never the replacement lease.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    service = _service(state, workspace, provider=_Provider(tag="aba"))
    job_id = "job_" + "a" * 24
    try:
        assert service._claim_round(job_id) is True
        old_token = service._round_lease_tokens[(job_id, threading.get_ident())]

        # Mirrors the release performed by a terminal record publication.
        service._release_job_lease(job_id)
        assert service._acquire_job_lease(job_id) is True
        replacement_token = service._job_lease_tokens[job_id]
        assert replacement_token is not old_token

        # Mirrors the old worker's later ``finally: _release_round(...)``.
        service._release_round(job_id)
        assert service._job_lease_tokens[job_id] is replacement_token
        assert job_id in service._job_leases
    finally:
        service._release_job_lease(job_id)
        service.close()


@needs_host
def test_a_live_running_round_cannot_be_stolen_by_another_instance(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    runner = _service(state, workspace, provider=_Provider(delay=0.6, tag="run"))
    other = _service(state, workspace, provider=_Provider(tag="other"))
    try:
        receipt = runner.submit(_TENANT, "key-1", _request(workspace))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if runner.get(_TENANT, receipt.job_id).state is CodingJobState.RUNNING:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("the round never reached running")
        # A round is genuinely executing, so its lease is genuinely held.
        assert other._claim_round(receipt.job_id) is False
        settled = _wait(runner, receipt.job_id)
        assert settled.state is CodingJobState.COMPLETED
        item = _items(state)[
            CodingMissionProjection.from_mapping(settled.mission or {}).work_item_id
        ]
        assert item.attempts == 1
    finally:
        runner.close()
        other.close()


@needs_host
def test_a_different_startup_authority_never_executes_another_worker_job(
    tmp_path: Path,
) -> None:
    """Shared durable state, unshared authority: requeued, never redirected."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    with _resource_held(state, workspace):
        owner = _service(state, workspace, provider=_Provider(tag="owner"))
        receipt = owner.submit(_TENANT, "key-1", _request(workspace))
        record = json.loads((
            state / "tenants" / CodingService._tenant_ref(_TENANT)
            / "jobs" / (receipt.job_id + ".json")
        ).read_text(encoding="utf-8"))
        authority = record["execution_authority"]
        assert authority["backend"] == "native"
        assert authority["audit_required"] is False
        # No secret, no credential, no absolute path in the fingerprint.
        rendered = json.dumps(authority)
        assert str(workspace) not in rendered and str(state) not in rendered

        # A service that would build a different implementer cannot even start
        # against this root while the job is live, so it never reaches a pump.
        with pytest.raises(CodingAuthorityConflict):
            _stranger(state, workspace)
        assert owner._may_execute(record) is True
        # An unfingerprinted or malformed record fails closed both ways.
        assert owner._may_execute({}) is False
        assert owner._may_execute({"execution_authority": {"backend": "native"}}) is False
        assert owner._may_execute(
            {"execution_authority": dict(authority, audit_required=True)},
        ) is False
    try:
        settled = _wait(owner, receipt.job_id)
        assert settled.state is CodingJobState.COMPLETED
        # The compatible owner ran it; the stranger's provider never did.
        assert (workspace / "notes-owner.txt").exists()
        assert not (workspace / "notes-stranger.txt").exists()
    finally:
        owner.close()


# --------------------------------------------------------------------------
# restart
# --------------------------------------------------------------------------


@needs_host
@pytest.mark.parametrize("item_state", ["closed", "missing"])
def test_restart_terminalizes_a_queued_job_whose_mission_item_cannot_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, item_state: str,
) -> None:
    """Closed/absent queue authority releases the old job without replay."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    provider = _Provider(tag="must-not-replay")
    owner = _service(
        state, workspace, provider=provider, require_codex_audit=True,
    )
    owner._schedule_pump = lambda: None  # type: ignore[method-assign]
    receipt = owner.submit(_TENANT, "old", _request(workspace))
    projection = CodingMissionProjection.from_mapping(receipt.mission or {})
    job_path = (
        state / "tenants" / CodingService._tenant_ref(_TENANT)
        / "jobs" / (receipt.job_id + ".json")
    )
    if item_state == "closed":
        with owner._mission.dispatch() as work:
            assert work is not None and work.work_item_id == projection.work_item_id
            owner._mission.close_accounted(
                work,
                tenant_ref=work.tenant_ref,
                job_id=work.job_id,
                mission_id=work.mission_id,
                work_item_id=work.work_item_id,
                disposition=DISPOSITION_BLOCKED,
                rationale="the hostile fixture closed the queued authority",
                risk="the queued implementation must not be replayed",
                evidence_refs=("fixture-closed",),
            )
    owner.close()

    original_work_item = CodingMissionRuntime.work_item
    if item_state == "missing":
        monkeypatch.setattr(CodingMissionRuntime, "work_item", lambda self, item_id: None)
    restarted = _service(
        state, workspace, provider=provider, require_codex_audit=True,
    )
    if item_state == "missing":
        monkeypatch.setattr(CodingMissionRuntime, "work_item", original_work_item)
    try:
        settled = restarted.get(_TENANT, receipt.job_id)
        assert settled.state is CodingJobState.FAILED
        assert settled.failure_code == MISSION_ITEM_UNAVAILABLE
        failed_projection = CodingMissionProjection.from_mapping(settled.mission or {})
        assert failed_projection.status == MISSION_STATUS_CLOSED
        assert provider.rounds == 0
        assert not restarted._resume_path(
            CodingService._tenant_ref(_TENANT), receipt.job_id,
        ).exists()
        assert not restarted._workspace_claim_path(str(workspace)).exists()
        assert restarted._workspace_root_authority is None

        # The exact old claim is gone: a replacement admission in the same
        # workspace can acquire a fresh claim without replaying the old round.
        # Exercise the wired queue, not merely the claim writer: an identical
        # admission is an idempotent replay of the replacement, and its one
        # Mission item reaches the provider exactly once.
        restarted._schedule_pump = lambda: None  # type: ignore[method-assign]
        replacement = restarted.submit(_TENANT, "replacement", _request(workspace))
        replay = restarted.submit(_TENANT, "replacement", _request(workspace))
        assert replacement.job_id != receipt.job_id
        assert replay.job_id == replacement.job_id
        claim = json.loads(
            restarted._workspace_claim_path(str(workspace)).read_text(encoding="utf-8"),
        )
        assert claim["job_id"] == replacement.job_id
        assert restarted._workspace_root_authority is not None
        assert restarted._workspace_root_authority.held_digests == [
            restarted._workspace_digest(str(workspace)),
        ]
        assert json.loads(job_path.read_text(encoding="utf-8"))["state"] == "failed"

        assert restarted._dispatch_once() == "ran"
        if item_state == "missing":
            # The hostile lookup made the item absent only to reconciliation;
            # its real ready row remains in MissionStore. The ordinary wired
            # dispatcher accounts that now-terminal old row without invoking
            # the provider, then the replacement is next in queue.
            accounted_old_item = _store(state).get_work_item(
                projection.work_item_id,
            )
            assert accounted_old_item.status == STATUS_CLOSED
            assert accounted_old_item.disposition == DISPOSITION_DEFERRED
            assert provider.rounds == 0
            assert restarted._dispatch_once() == "ran"
        replacement_settled = _wait(restarted, replacement.job_id)
        assert replacement_settled.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert provider.rounds == 1
        replacement_projection = CodingMissionProjection.from_mapping(
            replacement_settled.mission or {},
        )
        replacement_item = _store(state).get_work_item(
            replacement_projection.work_item_id,
        )
        assert replacement_item.status == STATUS_CLOSED
        assert replacement_item.attempts == 1
        assert restarted._dispatch_once() != "ran"
        assert provider.rounds == 1
        assert restarted.get(_TENANT, receipt.job_id).state is CodingJobState.FAILED
    finally:
        restarted.close()


@needs_host
def test_a_restart_reclaims_only_leases_it_can_prove_are_free(
    tmp_path: Path,
) -> None:
    """Reclaim is evidence-based, and an interrupted item is accounted for."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    runtime = CodingMissionRuntime(state, worker="w-restart")
    tenant_ref = CodingService._tenant_ref(_TENANT)
    placed = runtime.admit(
        tenant_ref=tenant_ref, job_id="job_" + "e" * 24,
        workspace_sha256="e" * 64, envelope=None, message="interrupted",
    )
    # A job record exactly as a crashed process would have left it.
    jobs = state / "tenants" / tenant_ref / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    now = time.time()
    # Stamped with this configuration's own authority, so the record is an
    # *interrupted* round rather than a pre-fingerprint one - the two settle
    # under deliberately different codes.
    probe = _service(state, workspace)
    authority = probe._execution_authority()
    probe.close()
    (jobs / ("job_" + "e" * 24 + ".json")).write_text(json.dumps({
        "job_id": "job_" + "e" * 24,
        "state": CodingJobState.RUNNING.value,
        "submitted_at": now,
        "updated_at": now,
        "working_dir": str(workspace),
        "execution_authority": authority,
        "mission": placed.projection.to_mapping(),
    }), encoding="utf-8")
    # Dispatch it once and leave the block without closing, exactly as an
    # interrupted round does: the item returns to the queue, keeping its
    # attempt, and no lease survives to be stolen from anybody.
    with runtime.dispatch() as work:
        assert work is not None
        assert runtime.store.get_work_item(work.work_item_id).status == STATUS_DISPATCHED
        # While that lease is live, nothing may reclaim it - not on age, not on
        # silence, not on a heartbeat that never arrived.
        with pytest.raises(MissionRouteError):
            CodingMissionRuntime(state, worker="w-other").reclaim(work.work_item_id)
    assert runtime.store.get_work_item(placed.work_item_id).status == STATUS_READY

    service = _service(state, workspace)
    try:
        receipt = service.get(_TENANT, "job_" + "e" * 24)
        # The interrupted job is failed closed exactly as the service requires.
        assert receipt.state is CodingJobState.FAILED
        assert receipt.failure_code == "service_restarted"
        # And its work item is accounted for through the ordinary store-ordered
        # route: a live job submitted afterwards drives the queue, and the
        # orphaned item closes with the whole accounting rather than running.
        live = service.submit(_TENANT, "key-live", _request(workspace))
        _wait(service, live.job_id)
        deadline = time.monotonic() + 15
        item = runtime.store.get_work_item(placed.work_item_id)
        while time.monotonic() < deadline and item.status != STATUS_CLOSED:
            time.sleep(0.05)
            item = runtime.store.get_work_item(placed.work_item_id)
        assert item.status == STATUS_CLOSED
        # Never fixed: work that did not run delivered nothing.
        assert item.disposition in (DISPOSITION_DEFERRED, DISPOSITION_BLOCKED)
        assert item.closure is not None and item.closure.rationale
        assert item.closure.risk and item.closure.owner
        assert item.closure.evidence_refs and item.closure.revisit_at
    finally:
        service.close()


@needs_host
def test_an_awaiting_audit_job_survives_a_restart_with_its_mission_open(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    service = _service(state, workspace, require_codex_audit=True)
    try:
        receipt = service.submit(_TENANT, "key-1", _request(workspace))
        ready = _wait(service, receipt.job_id)
        assert ready.state is CodingJobState.AWAITING_CODEX_AUDIT
    finally:
        service.close()
    restarted = _service(state, workspace, require_codex_audit=True)
    try:
        after = restarted.get(_TENANT, receipt.job_id)
        assert after.state is CodingJobState.AWAITING_CODEX_AUDIT
        projection = CodingMissionProjection.from_mapping(after.mission or {})
        assert projection.mission_status == MISSION_OPEN
        accepted = restarted.audit(
            _TENANT, receipt.job_id, after.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
    finally:
        restarted.close()


# --------------------------------------------------------------------------
# projection: fleet versus owner
# --------------------------------------------------------------------------


@needs_host
def test_the_fleet_view_is_snapshot_only_and_carries_no_secret(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    service = _service(state, workspace)
    try:
        receipt = service.submit(
            _TENANT, "key-1",
            _request(workspace, message="use token {}".format(_SECRET)),
        )
        _wait(service, receipt.job_id)
        fleet = json.dumps(service.mission_fleet())
        assert _SECRET not in fleet
        assert str(workspace) not in fleet
        assert "use token" not in fleet
        # No prose, no coordinate, no evidence value, no worker identity.
        for banned in ("objective", "desired_result", "statement", "rationale",
                       "risk", "owner", "evidence", "worker", "working_dir",
                       "location", "coordinates"):
            assert banned not in fleet, banned
        payload = service.mission_fleet()
        assert payload["available"] is True
        assert payload["missions"] and payload["work_items"]
    finally:
        service.close()


@needs_host
def test_full_mission_context_is_owner_only(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    service = _service(state, workspace)
    try:
        receipt = service.submit(
            _TENANT, "key-1", _request(workspace, message="private objective text"),
        )
        _wait(service, receipt.job_id)
        owned = service.mission_context(_TENANT, receipt.job_id)
        assert owned["objective"] == "private objective text"
        assert owned["acceptance_criteria"]
        # Another tenant cannot reach the same job at all, and the projection
        # identifiers alone are not a way in.
        from flyto_ai.coding.service import CodingJobNotFound

        with pytest.raises(CodingJobNotFound):
            service.mission_context("tenant-other", receipt.job_id)
    finally:
        service.close()


@needs_host
def test_a_public_receipt_publishes_only_the_closed_mission_field_set(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace)
    state = tmp_path / "state"
    service = _service(state, workspace)
    try:
        receipt = service.submit(
            _TENANT, "key-1",
            _request(workspace, message="secret {}".format(_SECRET)),
        )
        settled = _wait(service, receipt.job_id)
        body = receipt_to_mapping(settled)
        assert set(body["mission"]) == MISSION_PROJECTION_FIELDS
        rendered = json.dumps(body["mission"])
        assert _SECRET not in rendered
        assert str(workspace) not in rendered
        assert "secret" not in rendered
    finally:
        service.close()


@needs_host
def test_the_public_mcp_inventory_is_unchanged_by_the_lifecycle(
    tmp_path: Path,
) -> None:
    from flyto_ai.coding.mcp_server import CodingMCPServer

    tools = CodingMCPServer._tools()
    assert [tool["name"] for tool in tools] == [
        "flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit",
    ]
    assert not any("mission" in tool["name"] for tool in tools)


# --------------------------------------------------------------------------
# refusals reach the receipt vocabulary
# --------------------------------------------------------------------------


def test_every_mission_refusal_carries_a_machine_code() -> None:
    from flyto_ai.coding.mission_runtime import (
        MissionCapacityRefused,
        MissionConflictRefused,
        MissionCorruptRefused,
        MissionDependencyRefused,
        MissionStaleFenceRefused,
        MissionUnsupportedRefused,
    )

    codes = set()
    for kind in (
        MissionCapacityRefused, MissionConflictRefused, MissionCorruptRefused,
        MissionDependencyRefused, MissionStaleFenceRefused,
        MissionUnsupportedRefused, MissionAuthorityRefused,
    ):
        refusal = MissionRouteRefused(kind("refused"))
        assert refusal.failure_phase == "mission"
        assert refusal.code.startswith("mission_")
        assert isinstance(refusal.details["retryable"], bool)
        codes.add(refusal.code)
    assert len(codes) == 7
    assert MissionRouteRefused(MissionCapacityRefused("full")).retryable is True
    assert MissionRouteRefused(MissionAuthorityRefused("no")).retryable is False


@needs_host
def test_an_accounted_closure_is_never_fixed(tmp_path: Path) -> None:
    state = tmp_path / "state"
    runtime = CodingMissionRuntime(state, worker="w-close")
    tenant = "a" * 64
    runtime.admit(
        tenant_ref=tenant, job_id="job_" + "1" * 24,
        workspace_sha256="1" * 64, envelope=None, message="one",
    )
    with runtime.dispatch() as work:
        assert work is not None
        with pytest.raises(MissionAuthorityRefused):
            runtime.close_accounted(
                work, tenant_ref=tenant, job_id=work.job_id,
                mission_id=work.mission_id, work_item_id=work.work_item_id,
                disposition=DISPOSITION_FIXED,
                rationale="r", risk="k", evidence_refs=("e",),
            )
        runtime.close_accounted(
            work, tenant_ref=tenant, job_id=work.job_id,
            mission_id=work.mission_id, work_item_id=work.work_item_id,
            disposition=DISPOSITION_BLOCKED,
            rationale="the round could not run", risk="the objective is not reached",
            evidence_refs=("evidence-1",),
        )
    item = runtime.store.get_work_item(
        list(_items(state))[0],
    )
    assert item.disposition == DISPOSITION_BLOCKED
    assert item.closure is not None and item.closure.revisit_at


@needs_host
def test_a_closure_the_kernel_refuses_never_half_lands(tmp_path: Path) -> None:
    """A `Closure` missing its accounting is refused before anything is written."""

    with pytest.raises(Exception):
        Closure(disposition=DISPOSITION_BLOCKED, rationale="only a rationale")
    with pytest.raises(Exception):
        Closure(disposition=DISPOSITION_FIXED, revisit_at=int(time.time()) + 10)
