"""Job-lifetime worktree ownership across concurrent Codex frontends.

Every Codex conversation runs its own `code-mcp` process against one shared
state root. These tests drive two real `CodingService` instances over that
shared root — the same shape the supervisor produces — because the failures
this module exists to prevent only appear between processes: an audit gap that
another job edits, and a rework that lands on a worker which never held the
original session in memory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from flyto_ai.coding.contracts import CodingAuditVerdict, CodingJobState
from flyto_ai.coding.mcp_server import CodingMCPServer
from flyto_ai.coding.service import (
    WORKSPACE_CLAIM_VERSION,
    AbandonStateConflict,
    CodingJobNotFound,
    CodingService,
    WorkspaceBusy,
    WorkspaceClaimUnresolved,
    error_details,
)

from tests.test_coding_service import (
    ReworkingProvider,
    _audited_service,
    _awaiting,
    _blocker,
    _request,
    _service,
    _wait,
)


def _workspace(root: Path, name: str) -> Path:
    """Create one worktree with the repository check the harness expects."""

    workspace = root / name
    (workspace / ".flyto").mkdir(parents=True)
    (workspace / "result.txt").write_text("verified\n", encoding="utf-8")
    return workspace


def _paired_services(tmp_path: Path, workspace: Path, **kwargs):
    """Two audited services over one state root, as two MCP workers would be."""

    first = _audited_service(tmp_path, workspace, **kwargs)
    second = _audited_service(tmp_path, workspace, **kwargs)
    return first, second


def _claim_path(service: CodingService, workspace: Path) -> Path:
    return service._workspace_claim_path(str(workspace))


# --------------------------------------------------------------------------
# Contention and parallelism
# --------------------------------------------------------------------------


def test_two_workers_on_one_worktree_fail_fast_and_name_the_owner(
    tmp_path: Path,
) -> None:
    """The audit gap is exclusive: a second frontend is refused, not queued."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first, second = _paired_services(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(first, "tenant-audit", "own-001", workspace)

        with pytest.raises(WorkspaceBusy) as busy:
            second.submit("tenant-audit", "own-002", _request(workspace))
        assert busy.value.code == "workspace_busy"
        assert busy.value.owner_job_id == owner.job_id
        # The owning job id is the only context published, and it is an opaque
        # host-minted token rather than a path or any prompt material.
        assert error_details(busy.value) == {"owner_job_id": owner.job_id}
        assert str(workspace) not in json.dumps(error_details(busy.value))

        # The refusal changed nothing: the owner is still exactly auditable.
        still = first.get("tenant-audit", owner.job_id)
        assert still.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert still.implementation_revision_sha256 == owner.implementation_revision_sha256
    finally:
        second.close()
        first.close()


def test_mcp_publishes_the_busy_owner_as_bounded_structured_error_details(
    tmp_path: Path,
) -> None:
    """A Codex frontend learns which job to audit without a new tool."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first, second = _paired_services(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(first, "tenant-audit", "own-001", workspace)
        response = CodingMCPServer(second, "tenant-audit").handle({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "flyto_coding_submit",
                "arguments": {
                    "idempotency_key": "own-002",
                    "request": {"message": "second", "working_dir": str(workspace)},
                },
            },
        })
        structured = response["result"]["structuredContent"]
        assert structured["ok"] is False
        assert structured["error"] == "workspace_busy"
        assert structured["details"] == {"owner_job_id": owner.job_id}
        assert response["result"]["isError"] is True
    finally:
        second.close()
        first.close()


def test_different_worktrees_stay_parallel_across_workers(tmp_path: Path) -> None:
    """Cross-repository parallelism is the property the claim must not break."""

    alpha = _workspace(tmp_path, "alpha")
    beta = _workspace(tmp_path, "beta")
    first = _audited_service(tmp_path, alpha, provider=ReworkingProvider())
    second = _audited_service(tmp_path, beta, provider=ReworkingProvider())
    try:
        one = _awaiting(first, "tenant-audit", "par-001", alpha)
        two = _awaiting(second, "tenant-audit", "par-002", beta)
        assert one.job_id != two.job_id
        assert one.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert two.state is CodingJobState.AWAITING_CODEX_AUDIT
        # Two live claims, one per worktree, neither blocking the other.
        assert _claim_path(first, alpha).exists()
        assert _claim_path(second, beta).exists()
    finally:
        second.close()
        first.close()


def test_an_idempotent_retry_replays_the_receipt_without_a_second_claim(
    tmp_path: Path,
) -> None:
    """A retry is a read of the existing job, so it must not contend at all."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first, second = _paired_services(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(first, "tenant-audit", "idem-001", workspace)
        claim = json.loads(_claim_path(first, workspace).read_text(encoding="utf-8"))

        # The same key from the *other* worker replays rather than colliding.
        replay = second.submit("tenant-audit", "idem-001", _request(workspace))
        assert replay.job_id == owner.job_id
        assert replay.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert json.loads(
            _claim_path(first, workspace).read_text(encoding="utf-8"),
        ) == claim
    finally:
        second.close()
        first.close()


# --------------------------------------------------------------------------
# Cross-worker rework
# --------------------------------------------------------------------------


def test_rework_through_a_non_owner_worker_resumes_the_same_session(
    tmp_path: Path,
) -> None:
    """The whole point: the audit may land on a worker that never ran the job."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    implementer = ReworkingProvider()
    auditor_side = ReworkingProvider()
    first = _audited_service(tmp_path, workspace, provider=implementer)
    second = _audited_service(tmp_path, workspace, provider=auditor_side)
    try:
        owner = _awaiting(first, "tenant-audit", "cross-001", workspace)
        # The implementing worker is gone, exactly as a closed Codex tab.
        first.close()

        # The harness numbers each provider's rounds from its own counter, so
        # offset the replacement worker to keep its writes distinct from the
        # bytes the first worker already left. Identical content would produce
        # no attributable change and trigger a retry, which would say nothing
        # about the property under test.
        auditor_side.rounds = 10

        queued = second.audit(
            "tenant-audit", owner.job_id, owner.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, (_blocker(),),
        )
        assert queued.state is CodingJobState.REWORK_QUEUED

        reworked = _wait(second, "tenant-audit", owner.job_id)
        assert reworked.state is CodingJobState.AWAITING_CODEX_AUDIT
        # The exact same Claude session continues; a fresh one would be a new id.
        assert reworked.implementation_session_id == owner.implementation_session_id
        assert reworked.thread_id == owner.thread_id
        assert reworked.job_id == owner.job_id
        assert reworked.rework_count == 1
        # The replacement worker really did run exactly one round, and it is
        # the round that carried the typed audit findings.
        assert auditor_side.rounds == 11
        assert implementer.rounds == 1
        assert "cover the audited path" in auditor_side.prompts[-1]
        assert (
            reworked.implementation_revision_sha256
            != owner.implementation_revision_sha256
        )
    finally:
        second.close()
        first.close()


def test_the_durable_envelope_never_restores_startup_authority(
    tmp_path: Path,
) -> None:
    """Authority comes from the running process, never from a stored request."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "env-001", workspace)
        tenant_ref = service._tenant_ref("tenant-audit")
        envelope = json.loads(
            service._resume_path(tenant_ref, owner.job_id).read_text(encoding="utf-8"),
        )
        assert envelope["envelope_version"] == "flyto.coding-resume-envelope.v1"
        assert envelope["job_id"] == owner.job_id
        assert envelope["session_bound"] == owner.implementation_session_id
        for forbidden in (
            "approval_policy", "sandbox_mode", "config_path",
            "command_sandbox_image", "checks", "capabilities", "provider",
        ):
            assert forbidden not in envelope

        # The digest is stored, not recomputed: redaction rewrites prose, so a
        # recomputed hash would never match and rework would die silently.
        record = json.loads(
            (
                service.state_root / "tenants" / tenant_ref / "jobs"
                / (owner.job_id + ".json")
            ).read_text(encoding="utf-8"),
        )
        assert envelope["request_sha256"] == record["request_sha256"]
        restored = service._read_resume_envelope(tenant_ref, owner.job_id, record)
        assert restored is not None
        assert restored.working_dir == str(workspace)
    finally:
        service.close()


def test_an_envelope_bound_to_another_session_is_refused(tmp_path: Path) -> None:
    """A mismatched binding must not silently start a new Claude session."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "bind-001", workspace)
        tenant_ref = service._tenant_ref("tenant-audit")
        path = service._resume_path(tenant_ref, owner.job_id)
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["session_bound"] = "some-other-session"
        path.write_text(json.dumps(envelope), encoding="utf-8")

        record = json.loads(
            (
                service.state_root / "tenants" / tenant_ref / "jobs"
                / (owner.job_id + ".json")
            ).read_text(encoding="utf-8"),
        )
        assert service._read_resume_envelope(tenant_ref, owner.job_id, record) is None
    finally:
        service.close()


def test_rework_does_not_deadlock_on_the_claim_it_already_holds(
    tmp_path: Path,
) -> None:
    """A job must never queue behind its own worktree hold."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ReworkingProvider()
    service = _audited_service(tmp_path, workspace, provider=provider, max_rework_rounds=3)
    try:
        owner = _awaiting(service, "tenant-audit", "loop-001", workspace)
        current = owner
        for round_index in range(1, 3):
            service.audit(
                "tenant-audit", owner.job_id,
                current.implementation_revision_sha256,
                CodingAuditVerdict.REWORK, (_blocker(),),
            )
            current = _wait(service, "tenant-audit", owner.job_id)
            assert current.state is CodingJobState.AWAITING_CODEX_AUDIT
            assert current.rework_count == round_index
            claim = json.loads(
                _claim_path(service, workspace).read_text(encoding="utf-8"),
            )
            # One continuous hold by one job across every round.
            assert claim["job_id"] == owner.job_id
            assert claim["claimed_at"] <= claim["updated_at"]

        accepted = service.audit(
            "tenant-audit", owner.job_id, current.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
        assert accepted.landable is True
        assert not _claim_path(service, workspace).exists()
    finally:
        service.close()


def test_the_claim_survives_the_audit_gap_and_releases_on_accept(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first, second = _paired_services(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(first, "tenant-audit", "gap-001", workspace)
        assert _claim_path(first, workspace).exists()

        first.audit(
            "tenant-audit", owner.job_id, owner.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert not _claim_path(first, workspace).exists()

        # Only now may the next Codex task take the same worktree.
        follow_up = second.submit("tenant-audit", "gap-002", _request(workspace))
        assert follow_up.state is CodingJobState.QUEUED
        _wait(second, "tenant-audit", follow_up.job_id)
    finally:
        second.close()
        first.close()


# --------------------------------------------------------------------------
# Stale, corrupt, and orphaned claims
# --------------------------------------------------------------------------


def test_a_settled_owner_releases_its_claim_on_the_next_startup(
    tmp_path: Path,
) -> None:
    """A claim whose owner is provably terminal is swept; that is safe."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    tenant_ref = service._tenant_ref("tenant-audit")
    try:
        owner = _awaiting(service, "tenant-audit", "stale-001", workspace)
        record_path = (
            service.state_root / "tenants" / tenant_ref / "jobs"
            / (owner.job_id + ".json")
        )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["state"] = CodingJobState.FAILED.value
        record["landable"] = False
        record_path.write_text(json.dumps(record), encoding="utf-8")
    finally:
        service.close()

    successor = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        assert not _claim_path(successor, workspace).exists()
        fresh = successor.submit("tenant-audit", "stale-002", _request(workspace))
        assert fresh.state is CodingJobState.QUEUED
        _wait(successor, "tenant-audit", fresh.job_id)
    finally:
        successor.close()


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(b"{ not json", id="unparseable"),
        pytest.param(
            json.dumps({"claim_version": "flyto.coding-workspace-claim.v99"}).encode(),
            id="unknown_version",
        ),
        pytest.param(
            json.dumps({
                "claim_version": WORKSPACE_CLAIM_VERSION,
                "job_id": "job_" + "a" * 24,
                "tenant_ref": "f" * 64,
                "workspace_sha256": "0" * 64,
                "state": "queued",
                "instance_id": "i" * 24,
                "process_id": 1,
                "claimed_at": 1.0,
                "updated_at": 1.0,
                "unexpected": True,
            }).encode(),
            id="extended_shape",
        ),
    ],
)
def test_an_unevaluable_claim_fails_closed_and_is_never_discarded(
    tmp_path: Path, corrupt: bytes,
) -> None:
    """Absence of authority must be proven, never assumed.

    Deleting a claim the service cannot read would turn "I do not know whether
    a job owns this tree" into "nobody owns it", which is exactly the
    concurrent-edit hazard the claim exists to stop.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        path = _claim_path(service, workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(corrupt)

        with pytest.raises(WorkspaceClaimUnresolved) as unresolved:
            service.submit("tenant-audit", "corrupt-001", _request(workspace))
        assert unresolved.value.code == "workspace_claim_unresolved"
        # Still present: only a host operator may clear it.
        assert path.read_bytes() == corrupt
    finally:
        service.close()

    # A restart must not "fix" it either; the sweep only removes proven-settled.
    successor = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        assert _claim_path(successor, workspace).read_bytes() == corrupt
        with pytest.raises(WorkspaceClaimUnresolved):
            successor.submit("tenant-audit", "corrupt-002", _request(workspace))
    finally:
        successor.close()


def test_a_claim_naming_an_unknown_job_is_unresolved_not_free(
    tmp_path: Path,
) -> None:
    """A missing owner record is ambiguous, so it must not release the tree."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        path = _claim_path(service, workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        missing = "job_" + "b" * 24
        path.write_text(json.dumps({
            "claim_version": WORKSPACE_CLAIM_VERSION,
            "job_id": missing,
            "tenant_ref": service._tenant_ref("tenant-audit"),
            "workspace_sha256": service._workspace_digest(str(workspace)),
            "state": "awaiting_codex_audit",
            "instance_id": "a" * 24,
            "process_id": 999999,
            "claimed_at": 1.0,
            "updated_at": 1.0,
        }), encoding="utf-8")

        assert service._workspace_authority(str(workspace)) == ("unresolved", missing)
        with pytest.raises(WorkspaceClaimUnresolved):
            service.submit("tenant-audit", "orphan-001", _request(workspace))
        assert path.exists()
    finally:
        service.close()


def _valid_claim(service: CodingService, workspace: Path, job_id: str) -> dict:
    return {
        "claim_version": WORKSPACE_CLAIM_VERSION,
        "job_id": job_id,
        "tenant_ref": service._tenant_ref("tenant-audit"),
        "workspace_sha256": service._workspace_digest(str(workspace)),
        "state": "awaiting_codex_audit",
        "instance_id": "a" * 24,
        "process_id": 4321,
        "claimed_at": 1.0,
        "updated_at": 2.0,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda claim: claim.pop("workspace_sha256"), id="missing_digest"),
        pytest.param(lambda claim: claim.pop("state"), id="missing_state"),
        pytest.param(lambda claim: claim.pop("claimed_at"), id="missing_claimed_at"),
        pytest.param(lambda claim: claim.pop("instance_id"), id="missing_instance"),
        pytest.param(
            lambda claim: claim.update({"workspace_sha256": "0" * 64}),
            id="mismatched_workspace_digest",
        ),
        pytest.param(
            lambda claim: claim.update({"state": "codex_accepted"}),
            id="state_outside_claim_owned",
        ),
        pytest.param(
            lambda claim: claim.update({"process_id": -1}), id="negative_pid",
        ),
        pytest.param(
            lambda claim: claim.update({"claimed_at": "soon"}), id="non_numeric_time",
        ),
    ],
)
def test_a_partially_bound_claim_is_unresolved_never_free(
    tmp_path: Path, mutate,
) -> None:
    """Missing fields must fail closed exactly like unknown ones.

    A half-written claim proves nothing about ownership. Accepting it because
    it merely lacks a key — rather than carrying an extra one — would be the
    fail-open this binding exists to prevent.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "bind-100", workspace)
        claim = _valid_claim(service, workspace, owner.job_id)
        mutate(claim)
        path = _claim_path(service, workspace)
        path.write_text(json.dumps(claim), encoding="utf-8")

        status, _owner_id = service._workspace_authority(str(workspace))
        assert status == "unresolved"
        with pytest.raises(WorkspaceClaimUnresolved):
            service.submit("tenant-audit", "bind-101", _request(workspace))
        # Preserved for a host operator, never silently repaired.
        assert json.loads(path.read_text(encoding="utf-8")) == claim
    finally:
        service.close()


def test_a_claim_pointing_at_an_unrelated_terminal_record_is_unresolved(
    tmp_path: Path,
) -> None:
    """The owner record must bind back to this exact job and worktree.

    Without that binding a well-formed claim could name a settled job for some
    other tree and read as `free`, releasing a worktree whose ownership was
    never actually evaluated.
    """

    alpha = _workspace(tmp_path, "alpha")
    beta = _workspace(tmp_path, "beta")
    owner_service = _audited_service(tmp_path, beta, provider=ReworkingProvider())
    try:
        # A real job on `beta`, driven to a terminal state.
        other = _awaiting(owner_service, "tenant-audit", "unrelated-001", beta)
        owner_service.audit(
            "tenant-audit", other.job_id, other.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert owner_service.get("tenant-audit", other.job_id).state is (
            CodingJobState.CODEX_ACCEPTED
        )
    finally:
        owner_service.close()

    service = _audited_service(tmp_path, alpha, provider=ReworkingProvider())
    try:
        # Point alpha's claim at beta's terminal record. Its digest says alpha,
        # so the claim itself is well formed for this tree.
        path = _claim_path(service, alpha)
        path.parent.mkdir(parents=True, exist_ok=True)
        claim = _valid_claim(service, alpha, other.job_id)
        path.write_text(json.dumps(claim), encoding="utf-8")

        status, owner_id = service._workspace_authority(str(alpha))
        assert (status, owner_id) == ("unresolved", other.job_id)
        with pytest.raises(WorkspaceClaimUnresolved):
            service.submit("tenant-audit", "unrelated-002", _request(alpha))
        # The startup sweep must preserve it too.
        service._sweep_workspace_claims()
        assert path.exists()
    finally:
        service.close()


def test_a_record_whose_own_workspace_fields_disagree_is_unresolved(
    tmp_path: Path,
) -> None:
    """A record must agree with itself before it can answer for a claim."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "bind-200", workspace)
        tenant_ref = service._tenant_ref("tenant-audit")
        record_path = (
            service.state_root / "tenants" / tenant_ref / "jobs"
            / (owner.job_id + ".json")
        )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["workspace_sha256"] = "1" * 64
        record_path.write_text(json.dumps(record), encoding="utf-8")

        status, owner_id = service._workspace_authority(str(workspace))
        assert (status, owner_id) == ("unresolved", owner.job_id)

        # And a record naming a different worktree cannot answer either.
        record["workspace_sha256"] = hashlib.sha256(
            str(tmp_path / "elsewhere").encode(),
        ).hexdigest()
        record["working_dir"] = str(tmp_path / "elsewhere")
        record_path.write_text(json.dumps(record), encoding="utf-8")
        assert service._workspace_authority(str(workspace))[0] == "unresolved"
    finally:
        service.close()


@pytest.mark.parametrize(
    "state",
    [
        pytest.param(None, id="missing_state"),
        pytest.param("", id="empty_state"),
        pytest.param("retired", id="unknown_state"),
        pytest.param(7, id="wrong_typed_state"),
    ],
)
def test_a_record_without_a_known_state_is_unresolved_never_free(
    tmp_path: Path, state,
) -> None:
    """Absence of a valid terminal state is not proof that ownership settled."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "state-100", workspace)
        tenant_ref = service._tenant_ref("tenant-audit")
        record_path = (
            service.state_root / "tenants" / tenant_ref / "jobs"
            / (owner.job_id + ".json")
        )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if state is None:
            record.pop("state")
        else:
            record["state"] = state
        record_path.write_text(json.dumps(record), encoding="utf-8")

        assert service._workspace_authority(str(workspace)) == (
            "unresolved", owner.job_id,
        )
        with pytest.raises(WorkspaceClaimUnresolved):
            service.submit("tenant-audit", "state-101", _request(workspace))
    finally:
        service.close()


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf")],
)
def test_a_non_finite_claim_timestamp_is_unresolved(
    tmp_path: Path, value: float,
) -> None:
    """`NaN` fails every comparison and `Infinity` passes them all."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "finite-100", workspace)
        claim = _valid_claim(service, workspace, owner.job_id)
        claim["updated_at"] = value
        path = _claim_path(service, workspace)
        # `json.dumps` emits the non-finite literals `json.loads` accepts back.
        path.write_text(json.dumps(claim), encoding="utf-8")

        assert service._workspace_authority(str(workspace))[0] == "unresolved"
        with pytest.raises(WorkspaceClaimUnresolved):
            service.submit("tenant-audit", "finite-101", _request(workspace))
    finally:
        service.close()


def _steal_claim(service: CodingService, workspace: Path, owner_job_id: str) -> str:
    """Hand this worktree's claim to a real competing job record."""

    tenant_ref = service._tenant_ref("tenant-audit")
    jobs = service.state_root / "tenants" / tenant_ref / "jobs"
    intruder = "job_" + "f" * 24
    rival = json.loads((jobs / (owner_job_id + ".json")).read_text(encoding="utf-8"))
    rival["job_id"] = intruder
    rival["state"] = CodingJobState.AWAITING_CODEX_AUDIT.value
    (jobs / (intruder + ".json")).write_text(json.dumps(rival), encoding="utf-8")
    claim = json.loads(_claim_path(service, workspace).read_text(encoding="utf-8"))
    claim["job_id"] = intruder
    _claim_path(service, workspace).write_text(json.dumps(claim), encoding="utf-8")
    return intruder


def test_accept_is_refused_while_a_foreign_claim_owns_the_worktree(
    tmp_path: Path,
) -> None:
    """A landable record must never coexist with someone else's ownership."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "acc-foreign", workspace)
        intruder = _steal_claim(service, workspace, owner.job_id)

        with pytest.raises(WorkspaceBusy) as busy:
            service.audit(
                "tenant-audit", owner.job_id, owner.implementation_revision_sha256,
                CodingAuditVerdict.ACCEPT, (),
            )
        assert busy.value.owner_job_id == intruder

        still = service.get("tenant-audit", owner.job_id)
        assert still.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert still.landable is False
        assert still.audit_count == 0
        # The foreign claim is untouched.
        assert json.loads(
            _claim_path(service, workspace).read_text(encoding="utf-8"),
        )["job_id"] == intruder
    finally:
        service.close()


def test_accept_is_refused_while_the_claim_cannot_be_evaluated(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "acc-unres", workspace)
        _claim_path(service, workspace).write_bytes(b"{ corrupt")

        with pytest.raises(WorkspaceClaimUnresolved):
            service.audit(
                "tenant-audit", owner.job_id, owner.implementation_revision_sha256,
                CodingAuditVerdict.ACCEPT, (),
            )
        still = service.get("tenant-audit", owner.job_id)
        assert still.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert still.landable is False
        assert still.audit_count == 0
        assert _claim_path(service, workspace).read_bytes() == b"{ corrupt"
    finally:
        service.close()


@pytest.mark.parametrize(
    "verdict",
    [
        pytest.param(CodingAuditVerdict.ACCEPT, id="accept"),
        pytest.param(CodingAuditVerdict.REWORK, id="rework"),
    ],
)
def test_a_missing_claim_refuses_both_verdicts_and_is_not_recreated(
    tmp_path: Path, verdict: CodingAuditVerdict,
) -> None:
    """A vanished claim is not proof of uninterrupted ownership.

    During the missing interval another Codex could have taken this worktree,
    edited files outside this job's attributable set, settled, and released.
    Recomputing only this job's files would not see that, so reasserting from
    `free` would manufacture continuity that never existed.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ReworkingProvider()
    service = _audited_service(tmp_path, workspace, provider=provider)
    try:
        owner = _awaiting(service, "tenant-audit", "acc-free", workspace)
        _claim_path(service, workspace).unlink()
        assert service._workspace_authority(str(workspace)) == ("free", "")

        findings = () if verdict is CodingAuditVerdict.ACCEPT else (_blocker(),)
        with pytest.raises(WorkspaceClaimUnresolved):
            service.audit(
                "tenant-audit", owner.job_id,
                owner.implementation_revision_sha256, verdict, findings,
            )

        still = service.get("tenant-audit", owner.job_id)
        assert still.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert still.landable is False
        assert still.audit_count == 0
        assert still.rework_count == 0
        assert provider.rounds == 1
        # Refusing must not quietly re-create the hold it could not prove.
        assert not _claim_path(service, workspace).exists()
    finally:
        service.close()


def test_a_brand_new_submit_still_creates_its_first_claim(tmp_path: Path) -> None:
    """Only submit may claim a free worktree, and it still must."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        assert service._workspace_authority(str(workspace)) == ("free", "")
        queued = service.submit("tenant-audit", "fresh-001", _request(workspace))
        claim = json.loads(_claim_path(service, workspace).read_text(encoding="utf-8"))
        assert claim["job_id"] == queued.job_id
        assert claim["tenant_ref"] == service._tenant_ref("tenant-audit")
        assert _wait(service, "tenant-audit", queued.job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
    finally:
        service.close()


def test_host_abandon_is_the_spillway_for_an_orphaned_claimless_job(
    tmp_path: Path,
) -> None:
    """Fail-closed needs an explicit way out, and abandon is the only one."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "spill-001", workspace)
        _claim_path(service, workspace).unlink()

        abandoned = service.abandon("tenant-audit", owner.job_id)
        assert abandoned.state is CodingJobState.FAILED
        assert abandoned.failure_code == "job_abandoned"
        assert abandoned.landable is False

        fresh = service.submit("tenant-audit", "spill-002", _request(workspace))
        assert fresh.state is CodingJobState.QUEUED
        assert _wait(service, "tenant-audit", fresh.job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
    finally:
        service.close()


def test_a_same_job_id_claim_from_another_tenant_is_never_taken_over(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "tenant-001", workspace)
        path = _claim_path(service, workspace)
        claim = json.loads(path.read_text(encoding="utf-8"))
        claim["tenant_ref"] = hashlib.sha256(b"tenant-other").hexdigest()
        path.write_text(json.dumps(claim), encoding="utf-8")

        with pytest.raises(WorkspaceClaimUnresolved):
            service.audit(
                "tenant-audit", owner.job_id,
                owner.implementation_revision_sha256,
                CodingAuditVerdict.ACCEPT, (),
            )
        # The other namespace's claim is left exactly as it was.
        assert json.loads(path.read_text(encoding="utf-8")) == claim
        assert service.get("tenant-audit", owner.job_id).landable is False
    finally:
        service.close()


def test_a_record_naming_a_different_job_cannot_answer_for_a_claim(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "bind-300", workspace)
        tenant_ref = service._tenant_ref("tenant-audit")
        record_path = (
            service.state_root / "tenants" / tenant_ref / "jobs"
            / (owner.job_id + ".json")
        )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["job_id"] = "job_" + "e" * 24
        record_path.write_text(json.dumps(record), encoding="utf-8")

        assert service._workspace_authority(str(workspace))[0] == "unresolved"
    finally:
        service.close()


def test_rework_still_succeeds_while_the_valid_claim_is_present(
    tmp_path: Path,
) -> None:
    """The fail-closed rule must not break the ordinary continuous case."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ReworkingProvider()
    service = _audited_service(tmp_path, workspace, provider=provider)
    try:
        owner = _awaiting(service, "tenant-audit", "reacq-001", workspace)
        before = json.loads(_claim_path(service, workspace).read_text(encoding="utf-8"))

        service.audit(
            "tenant-audit", owner.job_id, owner.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, (_blocker(),),
        )
        reworked = _wait(service, "tenant-audit", owner.job_id)
        assert reworked.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert reworked.implementation_session_id == owner.implementation_session_id
        after = json.loads(_claim_path(service, workspace).read_text(encoding="utf-8"))
        # One continuous hold: same job, same original acquisition time.
        assert after["job_id"] == owner.job_id
        assert after["claimed_at"] == before["claimed_at"]
    finally:
        service.close()


def test_a_foreign_claim_stops_a_round_before_it_becomes_audit_ready(
    tmp_path: Path,
) -> None:
    """A round that cannot prove ownership must not reach the audit gap."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = ReworkingProvider()
    service = _audited_service(tmp_path, workspace, provider=provider)
    try:
        owner = _awaiting(service, "tenant-audit", "foreign-001", workspace)
        tenant_ref = service._tenant_ref("tenant-audit")
        jobs = service.state_root / "tenants" / tenant_ref / "jobs"

        # A genuine competing job: a real record naming this same worktree and
        # sitting in a claim-owned state, with the claim handed to it.
        intruder = "job_" + "f" * 24
        rival = json.loads((jobs / (owner.job_id + ".json")).read_text(encoding="utf-8"))
        rival["job_id"] = intruder
        rival["state"] = CodingJobState.AWAITING_CODEX_AUDIT.value
        (jobs / (intruder + ".json")).write_text(json.dumps(rival), encoding="utf-8")

        stolen = json.loads(_claim_path(service, workspace).read_text(encoding="utf-8"))
        stolen["job_id"] = intruder
        _claim_path(service, workspace).write_text(json.dumps(stolen), encoding="utf-8")
        assert service._workspace_authority(str(workspace)) == ("held", intruder)

        # The transition is refused before any round is scheduled, so the job
        # never re-enters a claim-owned state it cannot justify.
        with pytest.raises(WorkspaceBusy) as busy:
            service.audit(
                "tenant-audit", owner.job_id, owner.implementation_revision_sha256,
                CodingAuditVerdict.REWORK, (_blocker(),),
            )
        assert busy.value.owner_job_id == intruder
        assert provider.rounds == 1

        still = service.get("tenant-audit", owner.job_id)
        assert still.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert still.landable is False
        assert still.audit_count == 0
        # The foreign claim was never deleted by the job that lost the race.
        assert json.loads(
            _claim_path(service, workspace).read_text(encoding="utf-8"),
        )["job_id"] == intruder
        # The refused round handed its lease back rather than looking busy.
        assert service._acquire_job_lease(owner.job_id)
        service._release_job_lease(owner.job_id)
    finally:
        service.close()


def test_an_unresolved_claim_stops_a_round_and_is_left_for_repair(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "unres-001", workspace)
        _claim_path(service, workspace).write_bytes(b"{ corrupt")

        with pytest.raises(WorkspaceClaimUnresolved):
            service.audit(
                "tenant-audit", owner.job_id, owner.implementation_revision_sha256,
                CodingAuditVerdict.REWORK, (_blocker(),),
            )
        still = service.get("tenant-audit", owner.job_id)
        assert still.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert still.landable is False
        assert still.audit_count == 0
        # Preserved verbatim for a host operator.
        assert _claim_path(service, workspace).read_bytes() == b"{ corrupt"
        assert service._acquire_job_lease(owner.job_id)
        service._release_job_lease(owner.job_id)
    finally:
        service.close()


def test_a_claim_write_failure_cannot_open_an_unclaimed_audit_gap(
    tmp_path: Path,
) -> None:
    """If the hold cannot be persisted, the job must not become auditable."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    original = service._reassert_workspace_claim
    calls = {"n": 0}

    def failing_hold(tenant_ref, job_id, ws, state):
        calls["n"] += 1
        if state == CodingJobState.AWAITING_CODEX_AUDIT.value:
            raise OSError("claim store is unwritable")
        return original(tenant_ref, job_id, ws, state)

    try:
        service._reassert_workspace_claim = failing_hold
        queued = service.submit("tenant-audit", "wfail-001", _request(workspace))
        settled = _wait(service, "tenant-audit", queued.job_id)
        assert calls["n"] > 0
        # The record never advanced into the audit gap.
        assert settled.state is CodingJobState.FAILED
        assert settled.landable is False
    finally:
        service._reassert_workspace_claim = original
        service.close()


def test_host_repair_clears_only_an_unresolved_claim(tmp_path: Path) -> None:
    """The repair valve is explicit, and refuses while a live job owns the tree."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "repair-001", workspace)
        # A live owner is never cleared by repair; audit or abandon it instead.
        with pytest.raises(WorkspaceBusy) as busy:
            service.repair_workspace_claim(str(workspace))
        assert busy.value.owner_job_id == owner.job_id

        service.abandon("tenant-audit", owner.job_id)
        assert service.repair_workspace_claim(str(workspace)) == {
            "repaired": False, "status": "free", "owner_job_id": "",
        }

        path = _claim_path(service, workspace)
        path.write_bytes(b"{ corrupt")
        report = service.repair_workspace_claim(str(workspace))
        assert report["repaired"] is True and report["status"] == "unresolved"
        assert not path.exists()

        recovered = service.submit("tenant-audit", "repair-002", _request(workspace))
        assert recovered.state is CodingJobState.QUEUED
        _wait(service, "tenant-audit", recovered.job_id)
    finally:
        service.close()


# --------------------------------------------------------------------------
# The release valve
# --------------------------------------------------------------------------


def test_abandon_releases_the_worktree_and_can_never_land_a_job(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first, second = _paired_services(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(first, "tenant-audit", "aband-001", workspace)
        tenant_ref = first._tenant_ref("tenant-audit")

        abandoned = second.abandon("tenant-audit", owner.job_id)
        assert abandoned.state is CodingJobState.FAILED
        assert abandoned.failure_code == "job_abandoned"
        assert abandoned.landable is False
        assert not _claim_path(second, workspace).exists()
        assert not second._resume_path(tenant_ref, owner.job_id).exists()

        # Abandoning is terminal: it can never be audited into landability.
        with pytest.raises(Exception) as conflict:
            second.audit(
                "tenant-audit", owner.job_id, owner.implementation_revision_sha256,
                CodingAuditVerdict.ACCEPT, (),
            )
        assert getattr(conflict.value, "code", "") == "audit_state_conflict"
        assert first.get("tenant-audit", owner.job_id).landable is False

        # And the worktree is genuinely reusable.
        follow_up = second.submit("tenant-audit", "aband-002", _request(workspace))
        assert follow_up.state is CodingJobState.QUEUED
        _wait(second, "tenant-audit", follow_up.job_id)
    finally:
        second.close()
        first.close()


def test_abandon_refuses_every_state_that_is_not_audit_ready(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "aband-003", workspace)
        service.audit(
            "tenant-audit", owner.job_id, owner.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        # Accepted work is not an orphan and must stay exactly as audited.
        with pytest.raises(AbandonStateConflict):
            service.abandon("tenant-audit", owner.job_id)
        assert service.get("tenant-audit", owner.job_id).landable is True

        with pytest.raises(CodingJobNotFound):
            service.abandon("tenant-audit", "job_" + "c" * 24)
    finally:
        service.close()


def test_abandon_is_not_reachable_through_the_public_mcp_inventory(
    tmp_path: Path,
) -> None:
    """The audited route keeps exactly three tools; the valve is host-only."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        server = CodingMCPServer(service, "tenant-audit")
        listed = server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })
        names = [tool["name"] for tool in listed["result"]["tools"]]
        assert names == [
            "flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit",
        ]
        for forbidden in ("flyto_coding_abandon", "flyto_coding_release"):
            assert forbidden not in names
            response = server.handle({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": forbidden, "arguments": {}},
            })
            assert response["result"]["structuredContent"]["ok"] is False
    finally:
        service.close()


# --------------------------------------------------------------------------
# The legacy non-audited flow keeps its own contract
# --------------------------------------------------------------------------


def test_a_legacy_service_still_serializes_instead_of_refusing(
    tmp_path: Path,
) -> None:
    """No audit gap means no job-lifetime claim, so the old flow is unchanged."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _service(tmp_path, workspace)
    try:
        one = service.submit(
            "tenant-a", "legacy-001", _request(workspace, message="one", require_changes=False),
        )
        two = service.submit(
            "tenant-a", "legacy-002", _request(workspace, message="two", require_changes=False),
        )
        assert _wait(service, "tenant-a", one.job_id).state is CodingJobState.COMPLETED
        assert _wait(service, "tenant-a", two.job_id).state is CodingJobState.COMPLETED
        assert not _claim_path(service, workspace).exists()
    finally:
        service.close()


def test_a_legacy_service_still_honours_an_audited_job_claim(
    tmp_path: Path,
) -> None:
    """Otherwise a legacy worker could edit a tree mid-audit."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    audited = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    legacy = _service(tmp_path, workspace)
    try:
        owner = _awaiting(audited, "tenant-audit", "mixed-001", workspace)
        with pytest.raises(WorkspaceBusy) as busy:
            legacy.submit(
                "tenant-a", "mixed-002",
                _request(workspace, require_changes=False),
            )
        assert busy.value.owner_job_id == owner.job_id
    finally:
        legacy.close()
        audited.close()


def test_the_audited_route_stays_claude_only_with_no_fallback() -> None:
    """Ownership work must not have loosened implementer selection."""

    from flyto_ai.agents.claude_code import DEFAULT_CLAUDE_MODEL, ClaudeCodeAgent

    assert DEFAULT_CLAUDE_MODEL == "claude-opus-5"

    class _Unset:
        model = ""

    # An unset or unusable configuration resolves to the pinned model. There is
    # deliberately no fallback chain and no substitution of another backend.
    assert ClaudeCodeAgent.resolve_model(_Unset()) == "claude-opus-5"
