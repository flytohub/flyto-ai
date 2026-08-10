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
import json
import sys
import threading
import time
from pathlib import Path
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
    CodingService,
    MissionRouteRefused,
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
_SECRET = "sk-live-do-not-leak-0123456789"


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


class _Provider:
    """Deterministic provider; every effect goes through the real tool boundary."""

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
        if self.delay:
            await asyncio.sleep(self.delay)
        for path, content in (
            ("result.txt", "verified\n"),
            ("notes-{}.txt".format(self.tag), "{} round {}\n".format(self.tag, index)),
        ):
            outcome = await kwargs["dispatch_fn"](
                "coding_write_file",
                {"path": path, "content": content, "overwrite": True},
            )
            assert outcome["ok"]
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
    # Distinct attributable bytes per job, so "one provider round each" is a
    # statement about scheduling rather than about two fixtures colliding.
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
        # Exactly one provider round each: the handoff never duplicated work.
        assert first_provider.rounds == 1
        assert second_provider.rounds == 1
        # Both jobs really did land their own attributable bytes.
        assert (workspace / "notes-one.txt").read_text() == "one round 1\n"
        assert (workspace / "notes-two.txt").read_text() == "two round 1\n"
        items = _items(state)
        assert len(items) == 2
        # Every item left the queue explicitly, and each was really dispatched.
        assert all(item.status == STATUS_CLOSED for item in items.values())
        assert all(item.disposition for item in items.values())
        assert all(item.attempts >= 1 for item in items.values())
        # Exactly two rounds of provider work across both instances - one per
        # job. `attempts` is deliberately *not* asserted to be one: a pump that
        # is offered a job another instance has leased requeues it untouched,
        # which is a correct non-execution and still costs an attempt. The
        # round counts above are what "no duplicate provider execution" means.
        assert sum(
            provider.rounds for provider in (first_provider, second_provider)
        ) == 2
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
# restart
# --------------------------------------------------------------------------


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
    (jobs / ("job_" + "e" * 24 + ".json")).write_text(json.dumps({
        "job_id": "job_" + "e" * 24,
        "state": CodingJobState.RUNNING.value,
        "submitted_at": now,
        "updated_at": now,
        "working_dir": str(workspace),
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
