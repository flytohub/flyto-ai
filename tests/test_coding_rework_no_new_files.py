# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""A rework round that changes nothing new is not automatically a dead end.

This is the production failure. A rework round re-snapshots the worktree, so it
reports only what *that* round touched. When the auditor's finding is about a
file the previous round already changed, the correct repair frequently rewrites
that same file with the same bytes - and the honest answer to "what is newly
attributable here" is nothing at all. The service used to read that as "the
implementer produced nothing", terminalize the job, discard the resume envelope
and drop the session, so Codex lost a loop it was in the middle of and the
still-valid cumulative revision went with it.

Zero new files is only a dead end when the *cumulative* binding can no longer
be proven. Every test below is about where that line sits: the positive case
proves the loop survives, and each negative proves one binding is load-bearing
rather than decorative. The negatives are deliberately unit-level against
`_cumulative_attribution`, because driving the whole service eight times would
test the harness, not the boundary.
"""
import json
import sys
import time
from pathlib import Path

import pytest

from flyto_ai.coding.contracts import (
    ApprovalPolicy,
    audit_findings_sha256,
    CodingAuditFinding,
    CodingAuditSeverity,
    CodingAuditVerdict,
    CodingJobState,
    CodingTaskRequest,
    CodingTaskResult,
    SandboxMode,
)
from flyto_ai.coding.service import (
    REWORK_LIMIT_FAILURE_CODE,
    AuditStateConflict,
    CodingJobNotFound,
    CodingService,
    ReworkLimitReached,
    receipt_to_mapping,
)

_SETTLED = {
    CodingJobState.COMPLETED,
    CodingJobState.FAILED,
    CodingJobState.CODEX_ACCEPTED,
    CodingJobState.AWAITING_CODEX_AUDIT,
}
_SESSION = "sdk-session-stable"


class StableRewriteBackend:
    """Writes the *same bytes* every round, so only round one is attributable.

    This is the whole reproduction: a real backend repairing a file it already
    repaired produces an identical worktree, and a content-hash snapshot
    correctly reports no new attributable file.
    """

    def __init__(self, workspace: Path, *, session: str = _SESSION) -> None:
        self.workspace = workspace
        self.session = session
        self.requests: list = []

    async def run(self, request):
        self.requests.append(request)
        from flyto_ai.agents.models import CodeTaskResponse

        (self.workspace / "result.txt").write_text("verified\n")
        return CodeTaskResponse(
            # The *model* finished cleanly. The round is blocked by the host's
            # own required check, which is what makes the failure auditable
            # rather than a provider dead end.
            ok=True,
            message="applied the same repair again",
            session_id="local-evidence-{}".format(len(self.requests)),
            attempts=1,
            claude_session_id=self.session,
            claude_num_turns=3,
            claude_usage={"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0, "ok": True},
        )


def _workspace(tmp_path: Path) -> Path:
    """A worktree whose required check always fails, so rounds stay blocked."""

    workspace = tmp_path / "rework-workspace"
    workspace.mkdir(parents=True)
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir(exist_ok=True)
    config.write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: always_fails\n"
        "    argv: {}\n".format(json.dumps([sys.executable, "-c", "raise SystemExit(3)"]))
    )
    return workspace


def _passing_workspace(tmp_path: Path) -> Path:
    """A worktree whose required check proves a clean rework can close."""

    workspace = tmp_path / "passing-rework-workspace"
    workspace.mkdir(parents=True)
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir(exist_ok=True)
    config.write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: always_passes\n"
        "    argv: {}\n".format(json.dumps([sys.executable, "-c", "raise SystemExit(0)"]))
    )
    return workspace


def _service(tmp_path: Path, workspace: Path, backend, *, max_rework_rounds: int = 3):
    from flyto_ai.agents.claude_code import ClaudeCodingAgent

    return CodingService(
        lambda store: ClaudeCodingAgent(store, agent=backend),
        state_root=str(tmp_path / "rework-state"),
        workspace_roots=(str(workspace),),
        max_workers=2,
        max_queued=8,
        require_codex_audit=True,
        implementation_backend="claude",
        max_rework_rounds=max_rework_rounds,
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        approval_policy=ApprovalPolicy.NEVER,
    )


def _wait(service, tenant, job_id, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        receipt = service.get(tenant, job_id)
        if receipt.state in _SETTLED:
            return receipt
        time.sleep(0.02)
    raise AssertionError("coding job did not settle")


def _result(failure_code: str, *, thread_id: str = _SESSION) -> CodingTaskResult:
    """A minimal failed round result; only the fields the proof reads matter."""

    return CodingTaskResult(
        ok=False,
        status="failed",
        attempts=1,
        thread_id=thread_id,
        message="",
        failure_code=failure_code,
    )


def _finding():
    return CodingAuditFinding(
        code="needs_repair",
        message="the required check still fails",
        severity=CodingAuditSeverity.BLOCKER,
        evidence_ref="check.always_fails",
    )


# --------------------------------------------------------------------------
# the production case
# --------------------------------------------------------------------------


def test_a_rework_round_with_no_new_files_stays_auditable(tmp_path):
    """The exact failure: repair the same file again, keep the audit loop."""

    workspace = _workspace(tmp_path)
    backend = StableRewriteBackend(workspace)
    service = _service(tmp_path, workspace, backend)
    try:
        first = _wait(service, "tenant", service.submit(
            "tenant", "rework-key", CodingTaskRequest(
                message="repair the result", working_dir=str(workspace),
            ),
        ).job_id)

        # Round one really did change a file, and is blocked but auditable.
        assert first.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert first.implementation_session_id == _SESSION
        assert first.implementation_blockers
        assert first.landable is False
        original_files = tuple(service.get("tenant", first.job_id).result.files_changed or ())
        del original_files  # the receipt's own cumulative set is asserted below
        record_files = json.loads(
            (service.state_root / "tenants" / service._tenant_ref("tenant")
             / "jobs" / (first.job_id + ".json")).read_text(encoding="utf-8"),
        )["implementation_files"]
        assert "result.txt" in record_files

        # Codex orders rework. The backend rewrites the identical bytes, so
        # this round attributes nothing new.
        service.audit(
            "tenant", first.job_id, first.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, [_finding()],
        )
        second = _wait(service, "tenant", first.job_id)
        assert len(backend.requests) == 2
        # The rework round resumed the exact same backend session.
        assert backend.requests[1].sdk_session_id == _SESSION

        # This is the assertion the bug used to fail: still auditable.
        assert second.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert second.failure_code is None or second.state is not CodingJobState.FAILED
        assert second.implementation_session_id == _SESSION, "the session was dropped"
        assert second.implementation_revision_sha256
        assert second.implementation_blockers, "a blocked round must say why"
        assert second.landable is False
        assert second.rework_count == 1
        assert receipt_to_mapping(second)["job_terminal"] is False

        # The revision is recomputed from the cumulative set, not inherited.
        record = json.loads(
            (service.state_root / "tenants" / service._tenant_ref("tenant")
             / "jobs" / (first.job_id + ".json")).read_text(encoding="utf-8"),
        )
        assert "result.txt" in record["implementation_files"]
        assert record["implementation_revision_sha256"] == service._revision_digest(
            str(workspace), record["implementation_files"],
        )

        # And it is still reworkable: the loop did not merely survive once.
        service.audit(
            "tenant", first.job_id, second.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, [_finding()],
        )
        third = _wait(service, "tenant", first.job_id)
        assert third.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert third.rework_count == 2
        assert third.implementation_session_id == _SESSION
    finally:
        service.close()


def test_verified_no_change_rework_closes_against_the_cumulative_revision(tmp_path):
    """A clean recheck reuses proof; it never invents a second diff."""

    workspace = _passing_workspace(tmp_path)
    backend = StableRewriteBackend(workspace)
    service = _service(tmp_path, workspace, backend)
    try:
        first = _wait(service, "tenant", service.submit(
            "tenant", "clean-rework-key", CodingTaskRequest(
                message="repair the result", working_dir=str(workspace),
            ),
        ).job_id)
        assert first.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert first.result is not None and first.result.files_changed == ["result.txt"]

        service.audit(
            "tenant", first.job_id, first.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, [_finding()],
        )
        second = _wait(service, "tenant", first.job_id)

        assert second.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert second.result is not None and second.result.ok is True
        assert second.result.files_changed == ["result.txt"]
        assert second.result.failure_code is None
        assert second.implementation_revision_sha256 == first.implementation_revision_sha256
        assert second.implementation_blockers == ()

        accepted = service.audit(
            "tenant", second.job_id, second.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, [],
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
        assert accepted.landable is True
    finally:
        service.close()


def test_acceptance_stays_impossible_while_blockers_remain(tmp_path):
    """Surviving rework must not soften the accept gate."""

    from flyto_ai.coding.service import AuditBlockersUnresolved

    workspace = _workspace(tmp_path)
    service = _service(tmp_path, workspace, StableRewriteBackend(workspace))
    try:
        receipt = _wait(service, "tenant", service.submit(
            "tenant", "accept-key", CodingTaskRequest(
                message="repair", working_dir=str(workspace),
            ),
        ).job_id)
        assert receipt.implementation_blockers
        with pytest.raises(AuditBlockersUnresolved):
            service.audit(
                "tenant", receipt.job_id, receipt.implementation_revision_sha256,
                CodingAuditVerdict.ACCEPT, [],
            )
    finally:
        service.close()


# --------------------------------------------------------------------------
# the rework ceiling settles the job instead of bouncing the caller
# --------------------------------------------------------------------------


def test_the_rework_limit_terminalizes_releases_and_closes_the_loop(tmp_path):
    workspace = _workspace(tmp_path)
    backend = StableRewriteBackend(workspace)
    service = _service(tmp_path, workspace, backend, max_rework_rounds=1)
    try:
        first = _wait(service, "tenant", service.submit(
            "tenant", "limit-key", CodingTaskRequest(
                message="repair", working_dir=str(workspace),
            ),
        ).job_id)
        service.audit(
            "tenant", first.job_id, first.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, [_finding()],
        )
        second = _wait(service, "tenant", first.job_id)
        assert second.rework_count == 1

        # The budget is spent. This audit settles the job rather than bouncing.
        with pytest.raises(ReworkLimitReached):
            service.audit(
                "tenant", first.job_id, second.implementation_revision_sha256,
                CodingAuditVerdict.REWORK, [_finding()],
            )

        settled = service.get("tenant", first.job_id)
        assert settled.state is CodingJobState.FAILED
        assert settled.failure_code == REWORK_LIMIT_FAILURE_CODE
        assert settled.landable is False
        assert receipt_to_mapping(settled)["job_terminal"] is True

        # Bounded history is kept: an operator can still see how far it got.
        assert settled.implementation_session_id == _SESSION
        assert settled.implementation_revision_sha256
        assert settled.implementation_blockers

        # Resume authority is gone and the worktree is free again.
        tenant_ref = service._tenant_ref("tenant")
        assert not service._resume_path(tenant_ref, first.job_id).exists()
        assert not service._workspace_claim_path(str(workspace)).exists()

        # And no further verdict can move it.
        for verdict in (CodingAuditVerdict.REWORK, CodingAuditVerdict.ACCEPT):
            with pytest.raises((AuditStateConflict, CodingJobNotFound, ReworkLimitReached)):
                service.audit(
                    "tenant", first.job_id, settled.implementation_revision_sha256,
                    verdict, [_finding()] if verdict is CodingAuditVerdict.REWORK else [],
                )
    finally:
        service.close()


# --------------------------------------------------------------------------
# every binding the proof rests on is load-bearing
# --------------------------------------------------------------------------


@pytest.fixture()
def bound(tmp_path):
    """A service with one job genuinely bound to a cumulative revision."""

    workspace = _workspace(tmp_path)
    backend = StableRewriteBackend(workspace)
    service = _service(tmp_path, workspace, backend)
    receipt = _wait(service, "tenant", service.submit(
        "tenant", "bound-key", CodingTaskRequest(
            message="repair", working_dir=str(workspace),
        ),
    ).job_id)
    assert receipt.state is CodingJobState.AWAITING_CODEX_AUDIT
    tenant_ref = service._tenant_ref("tenant")
    path = (
        service.state_root / "tenants" / tenant_ref / "jobs" / (receipt.job_id + ".json")
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    request = CodingTaskRequest(message="repair", working_dir=str(workspace))
    result = _result("verification_failed")
    try:
        yield service, tenant_ref, receipt.job_id, record, request, result, workspace, path
    finally:
        service.close()


def _prove(service, tenant_ref, job_id, record, request, result):
    return service._cumulative_attribution(record, result, request, tenant_ref, job_id)


def test_the_bound_job_proves(bound):
    service, tenant_ref, job_id, record, request, result, _ws, _p = bound
    assert _prove(service, tenant_ref, job_id, record, request, result) == ("result.txt",)


def test_a_fresh_job_with_no_files_proves_nothing(bound):
    service, tenant_ref, job_id, record, request, result, _ws, _p = bound
    for empty in ({}, dict(record, implementation_files=[])):
        assert _prove(service, tenant_ref, job_id, empty, request, result) == ()


@pytest.mark.parametrize("session", ["", "sdk-other-session", "host-provisional-1"])
def test_a_different_or_provisional_session_proves_nothing(bound, session):
    service, tenant_ref, job_id, record, request, result, _ws, _p = bound
    moved = _result("verification_failed", thread_id=session)
    assert _prove(service, tenant_ref, job_id, record, request, moved) == ()
    # ...and a record that never bound a session cannot be rescued by one.
    assert _prove(
        service, tenant_ref, job_id,
        dict(record, implementation_session_id=""), request, result,
    ) == ()


def test_a_missing_or_tampered_resume_envelope_proves_nothing(bound):
    service, tenant_ref, job_id, record, request, result, _ws, _p = bound
    envelope = service._resume_path(tenant_ref, job_id)

    original = envelope.read_text(encoding="utf-8")
    envelope.write_text(
        json.dumps(dict(json.loads(original), session_bound="sdk-somebody-else")),
        encoding="utf-8",
    )
    assert _prove(service, tenant_ref, job_id, record, request, result) == ()

    envelope.write_text(original, encoding="utf-8")
    assert _prove(service, tenant_ref, job_id, record, request, result) == ("result.txt",)

    envelope.unlink()
    assert _prove(service, tenant_ref, job_id, record, request, result) == ()


def test_a_missing_or_foreign_workspace_claim_proves_nothing(bound):
    service, tenant_ref, job_id, record, request, result, workspace, _p = bound
    claim_path = service._workspace_claim_path(str(workspace))

    original = claim_path.read_text(encoding="utf-8")
    claim_path.write_text(
        json.dumps(dict(json.loads(original), job_id="job_" + "f0" * 12)),
        encoding="utf-8",
    )
    assert _prove(service, tenant_ref, job_id, record, request, result) == ()

    claim_path.write_text(original, encoding="utf-8")
    claim_path.unlink()
    assert _prove(service, tenant_ref, job_id, record, request, result) == ()


def test_a_foreign_tenant_namespace_proves_nothing(bound):
    service, tenant_ref, job_id, record, request, result, _ws, _p = bound
    assert _prove(service, "tenant_somebody_else", job_id, record, request, result) == ()


def test_a_different_workspace_proves_nothing(bound, tmp_path):
    service, tenant_ref, job_id, record, request, result, _ws, _p = bound
    other = tmp_path / "elsewhere"
    other.mkdir()
    elsewhere = CodingTaskRequest(message="repair", working_dir=str(other))
    assert _prove(service, tenant_ref, job_id, record, elsewhere, result) == ()
    # A record that points somewhere else is refused from the other direction.
    assert _prove(
        service, tenant_ref, job_id,
        dict(record, working_dir=str(other)), request, result,
    ) == ()


def test_a_missing_or_unbound_revision_proves_nothing(bound):
    service, tenant_ref, job_id, record, request, result, _ws, _p = bound
    for bad in ("", "not-a-digest", "ab" * 10):
        assert _prove(
            service, tenant_ref, job_id,
            dict(record, implementation_revision_sha256=bad), request, result,
        ) == ()


def test_an_empty_oversized_or_mutated_cumulative_set_proves_nothing(bound):
    service, tenant_ref, job_id, record, request, result, _ws, _p = bound
    from flyto_ai.coding.service import MAX_ATTRIBUTABLE_FILES

    for files in (
        [],
        "result.txt",                                   # not a list
        ["result.txt", "result.txt"],                   # duplicated
        [1234],                                         # re-typed
        ["a{}.txt".format(index) for index in range(MAX_ATTRIBUTABLE_FILES + 1)],
    ):
        assert _prove(
            service, tenant_ref, job_id,
            dict(record, implementation_files=files), request, result,
        ) == (), files


def test_a_deleted_symlinked_or_escaping_path_proves_nothing(bound, tmp_path):
    service, tenant_ref, job_id, record, request, result, workspace, _p = bound

    for escaping in ("../outside.txt", "/etc/passwd", "nested/../../out.txt"):
        assert _prove(
            service, tenant_ref, job_id,
            dict(record, implementation_files=[escaping]), request, result,
        ) == (), escaping

    target = workspace / "result.txt"
    target.unlink()
    assert _prove(service, tenant_ref, job_id, record, request, result) == ()

    outside = tmp_path / "outside.txt"
    outside.write_text("elsewhere\n")
    target.symlink_to(outside)
    assert _prove(service, tenant_ref, job_id, record, request, result) == ()


@pytest.mark.parametrize(
    "code",
    [
        "provider_failed",
        "provider_auth_failed",
        "provider_quota_exhausted",
        "provider_capacity_unavailable",
        "provider_policy_refused",
        "snapshot_failed",
        "session_binding_failed",
        "invalid_config",
        "",
    ],
)
def test_a_non_auditable_failure_is_never_held_open(bound, code):
    """The proof only decides attribution; the failure vocabulary still gates.

    A provable cumulative set must not turn an infrastructure or provider
    failure into a rework loop - those are exactly the outcomes the closed
    auditable vocabulary exists to keep terminal.
    """

    service, _tenant_ref, _job_id, _record, _request, _bound_result, _ws, _p = bound
    failed = _result(code)
    assert service._auditable_failure(
        failed, None, None, True, cumulative=("result.txt",),
    ) is False


def test_an_auditable_failure_with_a_proven_set_is_held_open(bound):
    service, _tenant_ref, _job_id, _record, _request, _bound_result, _ws, _p = bound
    for code in ("verification_failed", "turn_limit_exceeded"):
        held = _result(code)
        assert service._auditable_failure(
            held, None, None, True, cumulative=("result.txt",),
        ) is True
        # Without the proof, the same round is a dead end.
        assert service._auditable_failure(held, None, None, True) is False


def test_mutated_cumulative_bytes_are_never_laundered_into_continuity(bound):
    """The stored digest must be re-proven against the bytes, not just present.

    Checking only that a 64-hex revision was recorded proves the job was once
    bound to something. It says nothing about what is in the tree *now*. If a
    file changed between rounds - by anything, attributable or not - reusing
    the cumulative set would re-sign foreign bytes under the same session and
    hand an auditor a "continuous" revision that never existed.
    """

    service, tenant_ref, job_id, record, request, result, workspace, _p = bound

    # Baseline: the binding holds while the bytes are the ones that were hashed.
    assert _prove(service, tenant_ref, job_id, record, request, result) == ("result.txt",)

    # Nothing about the *bindings* changes here: same file list, same session,
    # same claim, same envelope, same workspace. Only the content moves.
    target = workspace / "result.txt"
    original = target.read_bytes()
    target.write_bytes(original + b"a line nobody attributed\n")

    assert _prove(service, tenant_ref, job_id, record, request, result) == ()

    # Restoring the exact bytes restores the binding, so the check is really
    # about content rather than about mtime or any other incidental signal.
    target.write_bytes(original)
    assert _prove(service, tenant_ref, job_id, record, request, result) == ("result.txt",)


def test_a_stale_stored_revision_is_refused_even_though_it_is_well_formed(bound):
    service, tenant_ref, job_id, record, request, result, _ws, _p = bound
    stale = dict(record, implementation_revision_sha256="ab" * 32)
    assert _prove(service, tenant_ref, job_id, stale, request, result) == ()


def test_the_rework_limit_records_the_audit_that_settled_it(tmp_path):
    """The count and the evidence must describe the same verdict."""

    workspace = _workspace(tmp_path)
    service = _service(tmp_path, workspace, StableRewriteBackend(workspace),
                       max_rework_rounds=1)
    try:
        first = _wait(service, "tenant", service.submit(
            "tenant", "digest-key", CodingTaskRequest(
                message="repair", working_dir=str(workspace),
            ),
        ).job_id)
        service.audit(
            "tenant", first.job_id, first.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, [_finding()],
        )
        second = _wait(service, "tenant", first.job_id)
        first_digest = service.get("tenant", first.job_id).audit_findings_sha256

        final_findings = [CodingAuditFinding(
            code="still_broken",
            message="the last verdict before the ceiling",
            severity=CodingAuditSeverity.BLOCKER,
            evidence_ref="check.always_fails",
        )]
        with pytest.raises(ReworkLimitReached):
            service.audit(
                "tenant", first.job_id, second.implementation_revision_sha256,
                CodingAuditVerdict.REWORK, final_findings,
            )

        settled = service.get("tenant", first.job_id)
        expected = audit_findings_sha256(final_findings)
        assert settled.audit_findings_sha256 == expected
        assert settled.audit_findings_sha256 != first_digest
        assert settled.audit_count == 2
        assert settled.failure_code == REWORK_LIMIT_FAILURE_CODE
    finally:
        service.close()


def test_settlement_is_idempotent_and_releases_only_its_own_claim(tmp_path):
    workspace = _workspace(tmp_path)
    service = _service(tmp_path, workspace, StableRewriteBackend(workspace),
                       max_rework_rounds=1)
    try:
        first = _wait(service, "tenant", service.submit(
            "tenant", "idem-key", CodingTaskRequest(
                message="repair", working_dir=str(workspace),
            ),
        ).job_id)
        service.audit(
            "tenant", first.job_id, first.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, [_finding()],
        )
        second = _wait(service, "tenant", first.job_id)
        with pytest.raises(ReworkLimitReached):
            service.audit(
                "tenant", first.job_id, second.implementation_revision_sha256,
                CodingAuditVerdict.REWORK, [_finding()],
            )
        tenant_ref = service._tenant_ref("tenant")
        record = json.loads(
            (service.state_root / "tenants" / tenant_ref / "jobs"
             / (first.job_id + ".json")).read_text(encoding="utf-8"),
        )

        # A later job legitimately takes the freed worktree.
        claim_path = service._workspace_claim_path(str(workspace))
        assert not claim_path.exists()
        service._create_workspace_claim(
            tenant_ref, "job_" + "b9" * 12, str(workspace), "queued",
        )
        assert claim_path.exists()

        # Replaying the settlement must not release the newcomer's claim.
        service._settle_at_rework_limit(
            service.state_root / "tenants" / tenant_ref / "jobs" / (first.job_id + ".json"),
            tenant_ref, first.job_id, record, 2, record["audit_findings_sha256"],
        )
        assert claim_path.exists(), "a replayed settlement released a foreign claim"
    finally:
        service.close()
