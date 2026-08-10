# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""A bounded stop that keeps its session is only half of a closed loop.

Round 7 made a `$5` ceiling stop tell the truth: the round happened, the session
existed, the turns and usage were real.  What it could not do was *use* any of
that.  The session survived inside one returned object and then the job went
terminal, so the only way to carry on was to hand somebody a session id and hope
they resumed it against the right tree.  That is not a continuation, it is a
suggestion.

Two things are being closed here, and they are separate on purpose.

*Identity, at the moment it exists.*  The provider announces its session in the
SDK `System`/`init` message, before the model has touched anything.  Binding it
there - rather than when the whole agent call returns - is what makes a round
that dies six minutes later still attributable.  A host that learns the session
only on the way out has no session at all for exactly the rounds that need one.

*Permission, bound to bytes.*  Holding a session id proves nothing, so a resume
is granted by a durable, tenant-local, single-use authority that binds the
tenant, the backend, the exact session, the workspace, the exact attributable
bytes, the verification contract and a monotonic generation.  Every one of those
is re-proven before the provider is contacted.  Resuming a real session against
a tree that has moved underneath it would produce a model editing a file it
believes it already wrote - which is worse than refusing.

What this module deliberately does not test is a real Claude account.  The SDK
message shapes are the real classes and the seams are the production ones, but
whether a live `error_max_budget_usd` round can genuinely be resumed by session
id is an account-level fact no local test can establish.
"""
import asyncio
import contextlib
import itertools
import dataclasses
import json
import os
import stat
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from flyto_ai.agents.claude_code import (
    ProviderSessionBindingFailed,
    signal_provider_session,
    signal_provider_start,
)
from flyto_ai.agents.models import CodeTaskResponse
from flyto_ai.coding.continuation import (
    _AUTHORITY_FIELDS,
    CONTINUABLE_STOP_CODES,
    CONTINUATION_CODES,
    CONTINUATION_CONTRACT_CHANGED,
    CONTINUATION_CONTRACT_UNPINNED,
    CONTINUATION_POLICY_CHANGED,
    CONTINUATION_REVISION_MISMATCH,
    CONTINUATION_SESSION_INVALID,
    CONTINUATION_UNAVAILABLE,
    CONTINUATION_WORKSPACE_MISMATCH,
    MAX_CONTINUATION_GENERATION,
    STATE_CLAIMED,
    STATE_OPEN,
    STATE_SETTLED,
    ContinuationAuthority,
    ContinuationConflict,
    ContinuationCorrupt,
    ContinuationStore,
    DEFAULT_SNAPSHOT_POLICY,
    JournalEntry,
    SnapshotPolicy,
    SnapshotPolicyInvalid,
    check_transition,
    WorkspaceUnobservable,
    is_continuable_session,
    read_journal,
    session_ref,
    workspace_manifest_digest,
)
from flyto_ai.coding.preflight import FAILURE_PHASE_PREFLIGHT
from flyto_ai.coding.checks import read_project_contract
from flyto_ai.coding.contracts import (
    ApprovalPolicy,
    CodingAuditFinding,
    CodingAuditSeverity,
    CodingAuditVerdict,
    CodingJobState,
    CodingTaskRequest,
    ContractSnapshot,
    SandboxMode,
)
from flyto_ai.coding import service as service_module
from flyto_ai.coding.service import (
    AUTHORITY_MARKER_NAME,
    CodingAuthorityConflict,
    CodingService,
    ContinuationRefused,
    IdempotencyConflict,
    VerificationContractChanged,
    VerificationRequired,
    SessionBindingFailed,
    receipt_to_mapping,
)
from flyto_ai.coding.store import ThreadStore, bind_provider_session, mark_provider_start

_SESSION = "3f2b0c18-9d4a-4e77-9f61-2c5a7b0e1d43"
_OTHER_SESSION = "8ac41d55-2b7e-41f0-8c19-6d0f4a2e7b91"
_BUDGET = "provider_job_budget_exhausted"
#: How many independent OS processes contend for one transition. More than
#: two, because the interesting failure is not "a second process" but "every
#: process after the winner".
_RACERS = 8
_SETTLED = {
    CodingJobState.COMPLETED,
    CodingJobState.FAILED,
    CodingJobState.CODEX_ACCEPTED,
    CodingJobState.AWAITING_CODEX_AUDIT,
}


# ──────────────────────────────────────────────────────────────────────
# production-shaped fixtures
# ──────────────────────────────────────────────────────────────────────


class SegmentBackend:
    """Stands in for `ClaudeCodeAgent`, at exactly the seams a real one uses.

    It is deliberately *not* a simulated service: the real `ClaudeCodingAgent`
    arms `on_provider_start` and `on_provider_session` on this object, and this
    object calls the same two module-level seam functions the SDK path calls.
    Everything else - snapshots, attribution, check execution, session
    validation - is the production adapter's own work, unmocked.
    """

    def __init__(self, workspace: Path, plan, *, session: str = _SESSION) -> None:
        self.workspace = workspace
        self.session = session
        self.plan = list(plan)
        self.requests: list = []
        #: Every session id this backend was actually asked to resume, in order.
        self.resumed: list = []

    async def run(self, request):
        self.requests.append(request)
        self.resumed.append(request.sdk_session_id)
        # The provider boundary, then the session the provider established.
        # Both before any edit, exactly as the SDK path orders them.
        signal_provider_start(self)
        signal_provider_session(self, self.session)
        outcome = self.plan.pop(0) if self.plan else "ok"
        (self.workspace / "feature.py").write_text(
            "# segment {}\n".format(len(self.requests)), encoding="utf-8",
        )
        if outcome == "budget":
            return CodeTaskResponse(
                ok=False,
                message="stopped at the configured ceiling",
                session_id="local-evidence-{}".format(len(self.requests)),
                attempts=1,
                claude_session_id=self.session,
                claude_num_turns=37,
                claude_usage={"input_tokens": 120, "output_tokens": 45},
                provider_failure_code=_BUDGET,
            )
        return CodeTaskResponse(
            ok=True,
            message="finished the change",
            session_id="local-evidence-{}".format(len(self.requests)),
            attempts=1,
            claude_session_id=self.session,
            claude_num_turns=4,
            claude_usage={"input_tokens": 10, "output_tokens": 5},
        )


def _workspace(tmp_path: Path, name: str = "workspace") -> Path:
    workspace = tmp_path / name
    workspace.mkdir(parents=True, exist_ok=True)
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir(exist_ok=True)
    config.write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: declared\n"
        "    argv: {}\n"
        "    required: true\n".format(json.dumps([sys.executable, "--version"])),
        encoding="utf-8",
    )
    # An ordinary unrelated file: never attributed to any round, and therefore
    # invisible to the revision digest. The manifest still has to see it.
    (workspace / "README.md").write_text("# project\n", encoding="utf-8")
    return workspace


def _service(
    tmp_path: Path,
    workspace: Path,
    backend: SegmentBackend,
    *,
    state_dir: str = "continuation-state",
    implementation_backend: str = "claude",
) -> CodingService:
    from flyto_ai.agents.claude_code import ClaudeCodingAgent

    return CodingService(
        lambda store: ClaudeCodingAgent(store, agent=backend),
        state_root=str(tmp_path / state_dir),
        workspace_roots=(str(workspace),),
        max_workers=1,
        max_queued=8,
        require_codex_audit=True,
        implementation_backend=implementation_backend,
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        approval_policy=ApprovalPolicy.NEVER,
    )


def _wait(service, tenant, job_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        receipt = service.get(tenant, job_id)
        if receipt.state in _SETTLED:
            return receipt
        time.sleep(0.02)
    raise AssertionError("coding job did not settle")


def _request(workspace: Path, **overrides) -> CodingTaskRequest:
    fields = {"message": "implement the feature", "working_dir": str(workspace)}
    fields.update(overrides)
    return CodingTaskRequest(**fields)


def _authority_of(service, tenant, session=_SESSION):
    return service._continuation.load(service._tenant_ref(tenant), session)


def _stopped_job(tmp_path, *, plan=("budget", "ok"), tenant="t"):
    """The live incident, reproduced: one job that stops at its ceiling."""

    workspace = _workspace(tmp_path)
    backend = SegmentBackend(workspace, plan)
    service = _service(tmp_path, workspace, backend)
    first = service.submit(tenant, "segment-1", _request(workspace))
    stopped = _wait(service, tenant, first.job_id)
    return service, backend, workspace, stopped


# ──────────────────────────────────────────────────────────────────────
# 1. the init-time seam: identity, before anything else happens
# ──────────────────────────────────────────────────────────────────────


def test_the_sdk_init_message_is_where_the_session_is_bound(tmp_path, monkeypatch):
    """Driven through the real `_run_claude_code` and the real SDK classes.

    The ordering claim is the point: the binding must land before the assistant
    turn, not after the call returns. A stand-in message class would skip the
    production `isinstance` branch entirely and prove nothing.
    """

    import claude_agent_sdk as sdk
    from flyto_ai.agents import claude_code as cc
    from flyto_ai.agents.evidence import EvidenceCollector
    from flyto_ai.agents.models import CodeTaskRequest

    timeline: list = []
    init = sdk.SystemMessage(subtype="init", data={"session_id": _SESSION})
    assistant = sdk.AssistantMessage(
        content=[sdk.TextBlock(text="editing")], model="claude",
    )

    async def fake_query(prompt=None, options=None):
        yield init
        # Recorded *after* the host has consumed the init message, so the
        # ordering assertion below is about when binding really happened.
        timeline.append("assistant-turn")
        yield assistant

    monkeypatch.setattr(sdk, "query", fake_query)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    agent = cc.ClaudeCodeAgent.__new__(cc.ClaudeCodeAgent)
    agent._cc = types.SimpleNamespace(
        model="", permission_mode="acceptEdits", max_turns=10,
        verification_timeout=30, system_prompt="", allowed_tools=None,
        max_budget_usd=5.0,
    )
    bound: list = []

    def record(session):
        timeline.append("bound")
        bound.append(session)

    agent.on_provider_session = record
    result = asyncio.run(agent._run_claude_code(
        request=CodeTaskRequest(
            message="do the work", working_dir=str(workspace),
            max_fix_attempts=1, max_turns=10,
            service_mode=True, service_edit_authority=True,
        ),
        indexer_context="", feedback="", session_id=None, max_budget=5.0,
        max_turns=10, evidence=EvidenceCollector("s", base_dir=str(tmp_path / "e")),
        on_stream=None,
    ))
    assert bound == [_SESSION]
    assert result["session_id"] == _SESSION
    # Bound at the init message, so the assistant turn that follows already
    # happened inside an identity the host had written down. Binding on the way
    # out would put these two the other way round - and would bind nothing at
    # all for the rounds that never come back.
    assert timeline == ["bound", "assistant-turn"]


def test_an_identical_init_is_idempotent_and_a_second_identity_is_refused():
    agent = types.SimpleNamespace()
    bound: list = []
    agent.on_provider_session = bound.append

    signal_provider_session(agent, _SESSION)
    signal_provider_session(agent, _SESSION)
    assert bound == [_SESSION], "a reconnect into the same session changes nothing"

    with pytest.raises(ProviderSessionBindingFailed):
        signal_provider_session(agent, _OTHER_SESSION)
    assert bound == [_SESSION], "a moved session never becomes the bound one"


@pytest.mark.parametrize(
    "identity",
    ["", "host-abcdef", "route-abcdef", "not a session", "x" * 200, None, 7, True],
)
def test_an_unsafe_or_provisional_identity_is_never_bound(identity):
    """A host-minted placeholder is the host's own invention, not a session."""

    agent = types.SimpleNamespace()
    bound: list = []
    agent.on_provider_session = bound.append
    with pytest.raises(ProviderSessionBindingFailed):
        signal_provider_session(agent, identity)
    assert bound == []


def test_a_host_that_cannot_record_the_session_stops_the_round():
    """Running unowned is the state this seam exists to remove."""

    agent = types.SimpleNamespace()

    def refuse(_session):
        raise OSError("state root is unwritable")

    agent.on_provider_session = refuse
    with pytest.raises(ProviderSessionBindingFailed):
        signal_provider_session(agent, _SESSION)


def test_a_backend_with_no_hook_installed_is_unaffected():
    signal_provider_session(types.SimpleNamespace(), _SESSION)


def test_the_store_seam_is_optional_and_carries_the_identity_verbatim(tmp_path):
    """The host-side half of the seam, over a real `ThreadStore`.

    Optional in the same way the start marker is: a store with no hook is a
    legacy caller, not an error. And it never interprets the id - the backend
    owns what a session is, the host owns what to do about it.
    """

    store = ThreadStore(str(tmp_path / "threads"))
    # No hook installed: a legacy caller is simply unaffected.
    bind_provider_session(store, _SESSION)

    seen: list = []
    started: list = []
    store.on_provider_session = seen.append
    store.on_provider_start = lambda: started.append("start")
    # Two hooks, two facts. The start marker fires without an identity, and
    # binding an identity does not imply anything about when the round began.
    mark_provider_start(store)
    assert (started, seen) == (["start"], [])
    bind_provider_session(store, _SESSION)
    assert seen == [_SESSION]

    def refuse(_session):
        raise SessionBindingFailed("the record cannot be written")

    store.on_provider_session = refuse
    with pytest.raises(SessionBindingFailed):
        bind_provider_session(store, _SESSION)


def test_the_binding_is_durable_before_the_round_returns(tmp_path):
    """A crash injected *after* binding still leaves the session on the record."""

    workspace = _workspace(tmp_path)

    class _CrashAfterBinding(SegmentBackend):
        async def run(self, request):
            self.requests.append(request)
            signal_provider_start(self)
            signal_provider_session(self, self.session)
            raise RuntimeError("worker died mid-round")

    backend = _CrashAfterBinding(workspace, [])
    service = _service(tmp_path, workspace, backend)
    try:
        queued = service.submit("t", "crash-after", _request(workspace))
        failed = _wait(service, "t", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        # The identity survived a round that never returned a result at all.
        assert failed.implementation_session_id == _SESSION
        assert failed.implementer_started is True
        # ...and a failure the host cannot classify leaves nothing resumable.
        assert failed.continuation_available is False
        assert _authority_of(service, "t") is None
    finally:
        service.close(wait=True)


def test_a_crash_before_binding_leaves_no_invented_session(tmp_path):
    workspace = _workspace(tmp_path)

    class _CrashBeforeBinding(SegmentBackend):
        async def run(self, request):
            self.requests.append(request)
            signal_provider_start(self)
            raise RuntimeError("worker died before init")

    backend = _CrashBeforeBinding(workspace, [])
    service = _service(tmp_path, workspace, backend)
    try:
        queued = service.submit("t", "crash-before", _request(workspace))
        failed = _wait(service, "t", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.implementation_session_id == ""
        assert failed.continuation_available is False
    finally:
        service.close(wait=True)


def test_the_service_refuses_a_session_that_contradicts_the_bound_one(tmp_path):
    """A backend that moves a bound round into another conversation fails closed."""

    workspace = _workspace(tmp_path)
    service = _service(tmp_path, workspace, SegmentBackend(workspace, []))
    try:
        queued = service.submit("t", "conflict", _request(workspace))
        _wait(service, "t", queued.job_id)
        tenant_ref = service._tenant_ref("t")
        path = service._tenant_dir(tenant_ref) / "jobs" / (queued.job_id + ".json")
        # The job already bound `_SESSION`; a second, different identity is a
        # boundary violation rather than a reconnect.
        # A real descriptor: the record write below releases the lease for
        # real, and a sentinel would blow up inside `flock`.
        service._job_leases[queued.job_id] = os.open(os.devnull, os.O_RDONLY)
        try:
            with pytest.raises(SessionBindingFailed):
                service._bind_provider_session(
                    path, tenant_ref, queued.job_id, str(workspace), _OTHER_SESSION,
                )
            # The same identity a second time is free.
            service._bind_provider_session(
                path, tenant_ref, queued.job_id, str(workspace), _SESSION,
            )
        finally:
            service._job_leases.pop(queued.job_id, None)
    finally:
        service.close(wait=True)


def test_binding_requires_this_worker_to_still_hold_the_job_lease(tmp_path):
    """Without the lease another process owns the round, so this one may not bind."""

    workspace = _workspace(tmp_path)
    service = _service(tmp_path, workspace, SegmentBackend(workspace, []))
    try:
        queued = service.submit("t", "lease", _request(workspace))
        _wait(service, "t", queued.job_id)
        tenant_ref = service._tenant_ref("t")
        path = service._tenant_dir(tenant_ref) / "jobs" / (queued.job_id + ".json")
        assert queued.job_id not in service._job_leases
        with pytest.raises(SessionBindingFailed):
            service._bind_provider_session(
                path, tenant_ref, queued.job_id, str(workspace), _SESSION,
            )
    finally:
        service.close(wait=True)


# ──────────────────────────────────────────────────────────────────────
# 2. the stopped job publishes exactly one bound authority
# ──────────────────────────────────────────────────────────────────────


def test_a_budget_stop_is_terminal_unaudited_and_still_continuable(tmp_path):
    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        assert stopped.state is CodingJobState.FAILED
        assert stopped.failure_code == _BUDGET
        assert stopped.landable is False
        assert stopped.audit_count == 0
        # Round 7's facts survive into the durable record, not just the result.
        assert stopped.implementer_started is True
        assert stopped.implementation_session_id == _SESSION
        assert stopped.result.attempts >= 1
        assert stopped.result.rounds_used == 37
        assert stopped.implementation_revision_sha256

        authority = _authority_of(service, "t")
        assert authority is not None
        assert authority.state == STATE_OPEN
        assert authority.generation == 1
        assert authority.session_id == _SESSION
        assert authority.backend == "claude"
        assert authority.job_id == authority.origin_job_id == stopped.job_id
        assert authority.working_dir == str(workspace)
        assert authority.revision_sha256 == stopped.implementation_revision_sha256
        assert authority.failure_code in CONTINUABLE_STOP_CODES
        assert authority.authorized_config_sha256
        # Exactly one authority, and it is the only file in the partition.
        partition = (
            service.state_root / "tenants" / service._tenant_ref("t") / "continuation"
        )
        assert [p.name for p in partition.glob("*.json")] == [
            session_ref(_SESSION) + ".json",
        ]
    finally:
        service.close(wait=True)


def test_a_failure_the_host_cannot_name_never_becomes_continuable(tmp_path):
    """Only the closed bounded-stop vocabulary produces an authority."""

    workspace = _workspace(tmp_path)

    class _UnknownStop(SegmentBackend):
        async def run(self, request):
            response = await super().run(request)
            return CodeTaskResponse(
                ok=False, message=response.message, session_id=response.session_id,
                attempts=1, claude_session_id=self.session, claude_num_turns=3,
                provider_failure_code="something_nobody_has_seen",
            )

    backend = _UnknownStop(workspace, ["budget"])
    service = _service(tmp_path, workspace, backend)
    try:
        queued = service.submit("t", "unknown", _request(workspace))
        failed = _wait(service, "t", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == "provider_failed"
        assert _authority_of(service, "t") is None
        assert failed.continuation_available is False
    finally:
        service.close(wait=True)


# ──────────────────────────────────────────────────────────────────────
# 3. the second job: one explicit resume, one provider call, same session
# ──────────────────────────────────────────────────────────────────────


def test_a_second_submit_resumes_the_exact_session_and_reaches_audit(tmp_path):
    """The whole closed loop, over the public submit/get/audit surface."""

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        assert backend.resumed == [None], "the first segment opened a session"

        resumed = service.submit(
            "t", "segment-2",
            _request(workspace, thread_id=_SESSION, resume=True),
        )
        second = _wait(service, "t", resumed.job_id)

        assert second.job_id != stopped.job_id
        assert second.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert second.implementation_session_id == _SESSION
        # Exactly two provider invocations, and the second entered the first's
        # session rather than opening a new one.
        assert len(backend.requests) == 2
        assert backend.resumed == [None, _SESSION]

        # The authority settles exactly once, on reaching an auditable revision.
        authority = _authority_of(service, "t")
        assert authority is not None and authority.state == STATE_SETTLED
        assert service.get("t", resumed.job_id).continuation_available is False

        # Exact-revision audit still governs the accept.
        from flyto_ai.coding.service import RevisionMismatch

        with pytest.raises(RevisionMismatch):
            service.audit(
                "t", resumed.job_id, "0" * 64, CodingAuditVerdict.ACCEPT, (),
            )
        accepted = service.audit(
            "t", resumed.job_id, second.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
        assert accepted.landable is True
    finally:
        service.close(wait=True)


def test_one_authority_cannot_be_spent_twice(tmp_path):
    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        first = service.submit(
            "t", "segment-2", _request(workspace, thread_id=_SESSION, resume=True),
        )
        _wait(service, "t", first.job_id)
        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "segment-3",
                _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_UNAVAILABLE
        assert len(backend.requests) == 2, "a refused resume never reaches a provider"
    finally:
        service.close(wait=True)


def test_a_second_bounded_stop_rotates_the_same_session_forward(tmp_path):
    """Monotonic, and never automatic: each segment is a separate submit."""

    service, backend, workspace, stopped = _stopped_job(
        tmp_path, plan=("budget", "budget", "ok"),
    )
    try:
        second = service.submit(
            "t", "segment-2", _request(workspace, thread_id=_SESSION, resume=True),
        )
        stopped_again = _wait(service, "t", second.job_id)
        assert stopped_again.failure_code == _BUDGET
        assert stopped_again.landable is False

        rotated = _authority_of(service, "t")
        assert rotated.generation == 2
        assert rotated.state == STATE_OPEN
        assert rotated.session_id == _SESSION
        assert rotated.job_id == second.job_id
        assert rotated.origin_job_id == stopped.job_id
        # The bytes moved with it, so a stale generation cannot be replayed.
        assert rotated.revision_sha256 == stopped_again.implementation_revision_sha256

        assert service.get("t", second.job_id).continuation_generation == 2
        assert service.get("t", second.job_id).continuation_available is True
        assert len(backend.requests) == 2, "rotation never spends a segment itself"

        third = service.submit(
            "t", "segment-3", _request(workspace, thread_id=_SESSION, resume=True),
        )
        done = _wait(service, "t", third.job_id)
        assert done.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert backend.resumed == [None, _SESSION, _SESSION]
        assert _authority_of(service, "t").state == STATE_SETTLED
    finally:
        service.close(wait=True)


def test_same_job_codex_rework_stays_a_separate_mechanism(tmp_path):
    """A reworkable round produces no continuation authority at all."""

    workspace = _workspace(tmp_path)
    backend = SegmentBackend(workspace, ["ok", "ok"])
    service = _service(tmp_path, workspace, backend)
    try:
        queued = service.submit("t", "rework", _request(workspace))
        ready = _wait(service, "t", queued.job_id)
        assert ready.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert _authority_of(service, "t") is None

        reworked = service.audit(
            "t", queued.job_id, ready.implementation_revision_sha256,
            CodingAuditVerdict.REWORK,
            (CodingAuditFinding(
                code="style", severity=CodingAuditSeverity.MAJOR,
                message="tighten the boundary",
            ),),
        )
        assert reworked.state in {
            CodingJobState.REWORK_QUEUED, CodingJobState.REWORK_RUNNING,
        }
        settled = _wait(service, "t", queued.job_id)
        # The rework resumed the same session, in the same job, with no
        # continuation authority anywhere near it.
        assert settled.implementation_session_id == _SESSION
        assert backend.resumed == [None, _SESSION]
        assert _authority_of(service, "t") is None
        assert settled.continuation_available is False
    finally:
        service.close(wait=True)


# ──────────────────────────────────────────────────────────────────────
# 4. byte drift: every case refuses before the provider
# ──────────────────────────────────────────────────────────────────────


def _drift_modify(workspace):
    (workspace / "feature.py").write_text("# tampered\n", encoding="utf-8")


def _drift_delete(workspace):
    (workspace / "feature.py").unlink()


def _drift_chmod(workspace):
    target = workspace / "feature.py"
    target.chmod(target.stat().st_mode | stat.S_IXUSR)


def _drift_symlink(workspace):
    target = workspace / "feature.py"
    other = workspace / "elsewhere.py"
    other.write_text("# segment 1\n", encoding="utf-8")
    target.unlink()
    target.symlink_to(other)


def _drift_truncate(workspace):
    (workspace / "feature.py").write_text("", encoding="utf-8")


def _drift_add_unrelated(workspace):
    """The probe that got through: a path the stopped round never touched.

    Nobody attributed `intruder.py` to the round, so the attributable-revision
    digest cannot see it. A resumed model would nevertheless be looking right
    at it.
    """

    (workspace / "intruder.py").write_text("# not mine\n", encoding="utf-8")


def _drift_add_nested(workspace):
    nested = workspace / "pkg" / "deep"
    nested.mkdir(parents=True)
    (nested / "planted.py").write_text("# planted\n", encoding="utf-8")


def _drift_delete_unrelated(workspace):
    (workspace / "README.md").unlink()


def _drift_retype_directory(workspace):
    target = workspace / "feature.py"
    target.unlink()
    target.mkdir()


def _drift_add_special(workspace):
    os.mkfifo(workspace / "pipe")


@pytest.mark.parametrize(
    "drift",
    [
        _drift_modify, _drift_delete, _drift_chmod, _drift_symlink,
        _drift_truncate, _drift_add_unrelated, _drift_add_nested,
        _drift_delete_unrelated, _drift_retype_directory, _drift_add_special,
    ],
    ids=[
        "modify", "delete", "chmod", "symlink-swap", "truncate",
        "add-unrelated", "add-nested", "delete-unrelated", "file-to-directory",
        "add-fifo",
    ],
)
def test_any_byte_drift_refuses_before_the_provider_is_contacted(tmp_path, drift):
    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        drift(workspace)
        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "segment-2",
                _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_REVISION_MISMATCH
        assert len(backend.requests) == 1, "no provider call was made"
        # Refused in preflight: no second job exists to poll or clean up.
        assert excinfo.value.failure_phase == "preflight"
        assert excinfo.value.retryable is False
        # The authority is untouched, so restoring the bytes restores the offer.
        assert _authority_of(service, "t").state == STATE_OPEN
    finally:
        service.close(wait=True)


def test_restoring_the_exact_bytes_restores_the_same_continuation(tmp_path):
    """Refusal is about the bytes, not about a tripped one-way flag."""

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        original = (workspace / "feature.py").read_bytes()
        _drift_modify(workspace)
        with pytest.raises(ContinuationRefused):
            service.submit(
                "t", "drifted", _request(workspace, thread_id=_SESSION, resume=True),
            )
        (workspace / "feature.py").write_bytes(original)
        resumed = service.submit(
            "t", "restored", _request(workspace, thread_id=_SESSION, resume=True),
        )
        assert _wait(service, "t", resumed.job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
        assert backend.resumed == [None, _SESSION]
    finally:
        service.close(wait=True)


def test_a_changed_verification_contract_refuses_the_continuation(tmp_path):
    """A post-stop contract edit is workspace drift, and is refused as such.

    `.flyto/coding.yaml` lives *inside* the workspace and is not one of the three
    excluded version-control names, so rewriting it after the stop changes the
    exact tree this authority promised a later segment it would find. The
    full-manifest gate therefore refuses first, before the pin is consulted at
    all, which is why the stable code here is `continuation_revision_mismatch`
    and not a contract-specific one.

    That ordering is the point of the pin, not a hole in it. The pin is no longer
    what refuses a changed file - comparing the current file against the
    pre-stop digest is exactly what made "the stopped round edited its own
    contract" unsatisfiable - it is what lets the round that *did* stop keep
    executing the contract it was admitted under. The contract-specific refusals
    are reserved for a pin that cannot be recovered or is not the one this
    authority binds; `test_a_tampered_pin_is_refused_before_any_provider_call`
    and `test_an_unrecoverable_pin_is_terminal_and_says_so` cover those.
    """

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        config = workspace / ".flyto" / "coding.yaml"
        config.write_text(
            "version: flyto.coding-config.v1\n"
            "checks:\n"
            "  - name: renamed\n"
            "    argv: {}\n"
            "    required: true\n".format(json.dumps([sys.executable, "--version"])),
            encoding="utf-8",
        )
        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "segment-2",
                _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_REVISION_MISMATCH
        assert excinfo.value.failure_phase == "preflight"
        assert len(backend.requests) == 1
        # Nothing was consumed, so restoring the exact bytes restores the offer -
        # the same property `test_restoring_the_exact_bytes...` proves for source.
        assert _authority_of(service, "t").state == STATE_OPEN
    finally:
        service.close(wait=True)


def test_a_different_workspace_cannot_borrow_the_authority(tmp_path):
    service, backend, workspace, stopped = _stopped_job(tmp_path)
    other = _workspace(tmp_path, "other-workspace")
    try:
        service.workspace_roots = tuple(service.workspace_roots) + (
            Path(other).resolve(),
        )
        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "segment-2", _request(other, thread_id=_SESSION, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_WORKSPACE_MISMATCH
        assert len(backend.requests) == 1
    finally:
        service.close(wait=True)


def test_another_backend_cannot_even_start_beside_this_session(tmp_path):
    """A different implementer is refused a step earlier than it used to be.

    This once proved that a `codex` service constructed beside a live `claude`
    one was stopped at `submit` by the continuation authority's backend guard.
    The state-root authority lease now refuses it at construction: the
    implementer is part of the startup authority, and a root with live work
    belongs to exactly one.

    The original invariant holds in its stronger form. No foreign backend enters
    the session, nothing about the stopped round moves, and - the half that
    matters most - the *correct* service can still continue it.
    """

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        marker = service.state_root / AUTHORITY_MARKER_NAME
        marker_before = marker.read_text(encoding="utf-8")
        record_path = _job_record_path(service, "t", stopped.job_id)
        record_before = record_path.read_text(encoding="utf-8")

        with pytest.raises(CodingAuthorityConflict):
            _service(tmp_path, workspace, backend, implementation_backend="codex")

        # The refused service reached no provider, consumed no generation, and
        # rewrote neither the marker nor the stopped round's record.
        assert len(backend.requests) == 1
        assert _authority_of(service, "t").state == STATE_OPEN
        assert marker.read_text(encoding="utf-8") == marker_before
        assert record_path.read_text(encoding="utf-8") == record_before

        # And the authority is still spendable by the backend that owns it.
        resumed = service.submit(
            "t", "segment-2", _request(workspace, thread_id=_SESSION, resume=True),
        )
        assert _wait(service, "t", resumed.job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
        assert backend.resumed == [None, _SESSION]
    finally:
        service.close(wait=True)


@pytest.mark.parametrize("identity", ["host-4c1d9f", "route-4c1d9f"])
def test_a_provisional_session_is_refused_without_any_lookup(tmp_path, identity):
    """A host-minted placeholder is refused on shape, so it probes nothing."""

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "bad-id", _request(workspace, thread_id=identity, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_SESSION_INVALID
        assert len(backend.requests) == 1
    finally:
        service.close(wait=True)


@pytest.mark.parametrize("identity", ["not a session", "x" * 200, "with/slash"])
def test_a_malformed_session_never_reaches_continuation_at_all(tmp_path, identity):
    """The request contract refuses it first, which is earlier still."""

    workspace = _workspace(tmp_path)
    with pytest.raises(ValueError):
        _request(workspace, thread_id=identity, resume=True)


# ──────────────────────────────────────────────────────────────────────
# 5. non-disclosure
# ──────────────────────────────────────────────────────────────────────


def test_a_cross_tenant_guess_is_indistinguishable_from_an_absent_session(tmp_path):
    """The two callers below must not be able to tell each other's world apart."""

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        # Tenant `b` guesses tenant `t`'s real, live session id.
        with pytest.raises(ContinuationRefused) as guessed:
            service.submit(
                "b", "guess", _request(workspace, thread_id=_SESSION, resume=True),
            )
        # ...and asks for a session that has never existed anywhere.
        with pytest.raises(ContinuationRefused) as absent:
            service.submit(
                "b", "absent",
                _request(workspace, thread_id=_OTHER_SESSION, resume=True),
            )
        assert guessed.value.code == absent.value.code == CONTINUATION_UNAVAILABLE
        assert guessed.value.details == absent.value.details
        assert str(guessed.value) == str(absent.value)
        # The other tenant's live authority was neither read nor disturbed.
        assert _authority_of(service, "t").state == STATE_OPEN
        assert len(backend.requests) == 1
    finally:
        service.close(wait=True)


def test_every_refusal_projects_only_bounded_closed_vocabulary(tmp_path):
    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        _drift_modify(workspace)
        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "drift", _request(workspace, thread_id=_SESSION, resume=True),
            )
        from flyto_ai.coding.service import error_details

        rendered = json.dumps(error_details(excinfo.value)) + str(excinfo.value)
        assert excinfo.value.code in CONTINUATION_CODES
        for forbidden in (str(workspace), _SESSION, "feature.py", str(tmp_path)):
            assert forbidden not in rendered, forbidden
    finally:
        service.close(wait=True)


# ──────────────────────────────────────────────────────────────────────
# 6. one owner only
# ──────────────────────────────────────────────────────────────────────


def test_two_services_racing_one_session_produce_exactly_one_owner(tmp_path):
    """Two independent `CodingService` instances over one shared state root.

    Both hold real cross-process file locks, so this is the production
    single-owner mechanism rather than a thread-level imitation.
    """

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    rival = _service(tmp_path, workspace, backend)
    try:
        granted, refused = [], []
        for name, instance in (("a", service), ("b", rival), ("c", service)):
            try:
                granted.append(instance.submit(
                    "t", "race-" + name,
                    _request(workspace, thread_id=_SESSION, resume=True),
                ))
            except ContinuationRefused as exc:
                refused.append(exc)
        assert len(granted) == 1, "exactly one submit may consume a generation"
        assert len(refused) == 2
        assert {exc.code for exc in refused} == {CONTINUATION_UNAVAILABLE}
        _wait(service, "t", granted[0].job_id)
        # One extra provider call in total: the losers never reached one.
        assert len(backend.requests) == 2
    finally:
        rival.close(wait=True)
        service.close(wait=True)


def test_distinct_sessions_and_workspaces_are_not_serialized_by_the_race(tmp_path):
    """One busy authority must not become a global lock on continuation."""

    first_ws = _workspace(tmp_path, "ws-one")
    second_ws = _workspace(tmp_path, "ws-two")
    left = SegmentBackend(first_ws, ["budget", "ok"], session=_SESSION)
    right = SegmentBackend(second_ws, ["budget", "ok"], session=_OTHER_SESSION)
    from flyto_ai.agents.claude_code import ClaudeCodingAgent

    #: Which backend the next round runs on. Two independent conversations in
    #: two independent worktrees, sharing one service and one state root.
    active = {"backend": left}
    service = CodingService(
        lambda store: ClaudeCodingAgent(store, agent=active["backend"]),
        state_root=str(tmp_path / "twin-state"),
        workspace_roots=(str(first_ws), str(second_ws)),
        max_workers=2, max_queued=8, require_codex_audit=True,
        implementation_backend="claude",
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        approval_policy=ApprovalPolicy.NEVER,
    )
    try:
        one = service.submit("t", "left-1", _request(first_ws))
        _wait(service, "t", one.job_id)
        active["backend"] = right
        two = service.submit("t", "right-1", _request(second_ws))
        _wait(service, "t", two.job_id)

        assert _authority_of(service, "t", _SESSION).generation == 1
        assert _authority_of(service, "t", _OTHER_SESSION).generation == 1

        active["backend"] = left
        resumed_left = service.submit(
            "t", "left-2", _request(first_ws, thread_id=_SESSION, resume=True),
        )
        assert _wait(service, "t", resumed_left.job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
        active["backend"] = right
        resumed_right = service.submit(
            "t", "right-2",
            _request(second_ws, thread_id=_OTHER_SESSION, resume=True),
        )
        assert _wait(service, "t", resumed_right.job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
        # Neither continuation was blocked by the other, and each entered its
        # own conversation.
        assert left.resumed == [None, _SESSION]
        assert right.resumed == [None, _OTHER_SESSION]
    finally:
        service.close(wait=True)


# ──────────────────────────────────────────────────────────────────────
# 7. tamper, stale generations, and crash reconciliation
# ──────────────────────────────────────────────────────────────────────


def _journal_path(service, tenant, session=_SESSION):
    return service._continuation.journal_path(service._tenant_ref(tenant), session)


def _authority_path(service, tenant, session=_SESSION):
    return service._continuation.path(service._tenant_ref(tenant), session)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda raw: {**raw, "revision_sha256": "0" * 64},
        lambda raw: {**raw, "generation": 9999},
        lambda raw: {**raw, "authority_version": "flyto.coding-continuation.v0"},
        lambda raw: {**raw, "tenant_ref": "0" * 64},
        lambda raw: {k: v for k, v in raw.items() if k != "record_sha256"},
        lambda raw: {**raw, "record_sha256": "f" * 64},
        lambda raw: {**raw, "state": "definitely-open"},
        lambda raw: {**raw, "files": []},
    ],
    ids=[
        "revision", "generation", "version", "tenant", "no-digest", "bad-digest",
        "state", "empty-files",
    ],
)
def test_a_tampered_or_truncated_authority_is_simply_unavailable(tmp_path, corrupt):
    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        path = _authority_path(service, "t")
        path.write_text(
            json.dumps(corrupt(json.loads(path.read_text()))), encoding="utf-8",
        )
        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "tampered",
                _request(workspace, thread_id=_SESSION, resume=True),
            )
        # Identical to absent: a caller learns nothing about what was there.
        assert excinfo.value.code == CONTINUATION_UNAVAILABLE
        assert len(backend.requests) == 1
    finally:
        service.close(wait=True)


def test_a_truncated_file_is_unreadable_rather_than_partially_trusted(tmp_path):
    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        path = _authority_path(service, "t")
        path.write_text(path.read_text()[: len(path.read_text()) // 2], encoding="utf-8")
        assert _authority_of(service, "t") is None
        with pytest.raises(ContinuationRefused):
            service.submit(
                "t", "truncated",
                _request(workspace, thread_id=_SESSION, resume=True),
            )
    finally:
        service.close(wait=True)


def test_a_replayed_older_generation_cannot_be_reinstated(tmp_path):
    """The audit probe, in full: old authority bytes *and* old workspace bytes.

    The earlier version of this test only restored the record, so it passed for
    the wrong reason - the tree happened to have moved on, and the revision
    check caught it. That proved nothing about replay. Restoring the workspace
    too removes every other defence and leaves exactly one question: can a
    perfectly valid, correctly self-signed generation-1 record be put back?

    It cannot, because nothing a rewriter controls says what generation this
    session is at. The append-only journal does, and its tail names generation
    two.
    """

    service, backend, workspace, stopped = _stopped_job(
        tmp_path, plan=("budget", "budget", "ok"),
    )
    try:
        path = _authority_path(service, "t")
        generation_one_record = path.read_bytes()
        generation_one_bytes = (workspace / "feature.py").read_bytes()

        second = service.submit(
            "t", "segment-2", _request(workspace, thread_id=_SESSION, resume=True),
        )
        _wait(service, "t", second.job_id)
        assert _authority_of(service, "t").generation == 2
        assert len(backend.requests) == 2

        # Restore *both* halves of the world generation 1 described.
        (workspace / "feature.py").write_bytes(generation_one_bytes)
        path.write_bytes(generation_one_record)
        # The body is still perfectly valid on its own terms...
        assert ContinuationAuthority.from_mapping(
            json.loads(path.read_text()),
        ).generation == 1
        # ...and it is still not the record this session is at.
        assert _authority_of(service, "t") is None

        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "replayed",
                _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_UNAVAILABLE
        assert len(backend.requests) == 2, "no provider call followed the replay"
    finally:
        service.close(wait=True)


@pytest.mark.parametrize("consume", ["claim", "settle"])
def test_an_old_open_record_cannot_be_replayed_over_a_consumed_one(
    tmp_path, consume,
):
    """Replay after CLAIMED and after SETTLED are both double-spend attempts."""

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    tenant_ref = service._tenant_ref("t")
    try:
        path = _authority_path(service, "t")
        open_record = path.read_bytes()
        opened = _authority_of(service, "t")
        store = service._continuation
        if consume == "claim":
            store.commit(opened, opened.claimed("job_" + "9" * 24, time.time()))
        else:
            store.commit(opened, opened.settled(time.time()))

        path.write_bytes(open_record)
        assert store.load(tenant_ref, _SESSION) is None
        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "double-spend",
                _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_UNAVAILABLE
        assert len(backend.requests) == 1
    finally:
        service.close(wait=True)


def test_the_journal_is_the_only_monotonic_source(tmp_path):
    """Every structural rule of the chain, at the parser, with no service."""

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        journal = _journal_path(service, "t")
        lines = journal.read_bytes().splitlines(keepends=True)
        assert len(lines) == 1
        entries = read_journal(b"".join(lines))
        assert entries[0].sequence == 1 and entries[0].generation == 1
        assert entries[0].previous_entry_sha256 == ""

        # A truncated final append is refused, not silently discarded: reading
        # it as "the state before" is how a crash would reopen a transition.
        with pytest.raises(ContinuationCorrupt):
            read_journal(lines[0][:-4])
        # An unknown key is not a forward-compatible extension.
        broken = json.loads(lines[0])
        broken["grant_scope"] = "all-workspaces"
        with pytest.raises(ContinuationCorrupt):
            read_journal(json.dumps(broken).encode() + b"\n")
        # A chain that does not start at one, or does not link, is refused.
        with pytest.raises(ContinuationCorrupt):
            read_journal(lines[0] + lines[0])

        # The two rules below are isolated deliberately. Each forged entry is
        # legal in every *other* respect - correct version, legal state
        # transition, non-decreasing generation, valid self-digest - so the
        # only thing standing between it and acceptance is the rule named.
        tail = entries[-1]
        skipped = JournalEntry(
            sequence=tail.sequence + 2,
            generation=tail.generation,
            state=STATE_SETTLED,
            authority_sha256="a" * 64,
            previous_entry_sha256=tail.entry_digest(),
            recorded_at=tail.recorded_at + 1,
        )
        with pytest.raises(ContinuationCorrupt):
            read_journal(lines[0] + skipped.to_line())

        unchained = JournalEntry(
            sequence=tail.sequence + 1,
            generation=tail.generation,
            state=STATE_SETTLED,
            authority_sha256="a" * 64,
            previous_entry_sha256="b" * 64,
            recorded_at=tail.recorded_at + 1,
        )
        with pytest.raises(ContinuationCorrupt):
            read_journal(lines[0] + unchained.to_line())

        # ...and the same entry, correctly sequenced and correctly chained, is
        # accepted. Without this the two probes above could pass for any reason.
        linked = JournalEntry(
            sequence=tail.sequence + 1,
            generation=tail.generation,
            state=STATE_SETTLED,
            authority_sha256="a" * 64,
            previous_entry_sha256=tail.entry_digest(),
            recorded_at=tail.recorded_at + 1,
        )
        assert len(read_journal(lines[0] + linked.to_line())) == 2
    finally:
        service.close(wait=True)


def test_an_authority_moved_into_another_tenant_partition_binds_nothing(tmp_path):
    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        source = _authority_path(service, "t")
        target = _authority_path(service, "b")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(), encoding="utf-8")
        source.unlink()
        assert _authority_of(service, "b") is None
        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "b", "stolen", _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_UNAVAILABLE
    finally:
        service.close(wait=True)


def test_an_authority_renamed_onto_another_session_key_binds_nothing(tmp_path):
    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        source = _authority_path(service, "t")
        target = _authority_path(service, "t", _OTHER_SESSION)
        target.write_text(source.read_text(), encoding="utf-8")
        assert _authority_of(service, "t", _OTHER_SESSION) is None
    finally:
        service.close(wait=True)


def test_an_abandoned_origin_leaves_nothing_to_continue(tmp_path):
    """The operator walked away from the job, so its session is finished."""

    workspace = _workspace(tmp_path)
    backend = SegmentBackend(workspace, ["ok"])
    service = _service(tmp_path, workspace, backend)
    try:
        first = service.submit("t", "abandon-1", _request(workspace))
        ready = _wait(service, "t", first.job_id)
        assert ready.state is CodingJobState.AWAITING_CODEX_AUDIT
        # Plant a live authority for this job's session, as a bounded stop would.
        tenant_ref = service._tenant_ref("t")
        service._continuation.create(ContinuationAuthority(
            tenant_ref=tenant_ref, backend="claude", session_id=_SESSION,
            job_id=first.job_id, origin_job_id=first.job_id,
            working_dir=str(workspace),
            workspace_sha256=service._workspace_digest(str(workspace)),
            revision_sha256=ready.implementation_revision_sha256,
            files=("feature.py",),
            authorized_config_sha256=service._read_json(
                service._tenant_dir(tenant_ref) / "jobs" / (first.job_id + ".json"),
            )["authorized_config_sha256"],
            contract_snapshot_sha256=service._read_json(
                service._tenant_dir(tenant_ref) / "jobs" / (first.job_id + ".json"),
            )["contract_snapshot_sha256"],
            workspace_manifest_sha256=workspace_manifest_digest(str(workspace)),
            snapshot_policy_sha256=service.snapshot_policy.identity(),
            request_sha256="0" * 64, failure_code=_BUDGET,
            generation=1, sequence=1,
        ))
        service._update_record(
            service._tenant_dir(tenant_ref) / "jobs" / (first.job_id + ".json"),
            continuation_session_id=_SESSION, continuation_generation=1,
        )
        service.abandon("t", first.job_id)
        assert _authority_of(service, "t").state == STATE_SETTLED
        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "abandon-2",
                _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_UNAVAILABLE
    finally:
        service.close(wait=True)


def test_a_claim_whose_worker_died_is_reconciled_rather_than_pinned(tmp_path):
    """A crash between claim and outcome must not leave a permanent claim.

    Nor may it reopen: an authority whose claimant's fate is unknown could
    already have been spent, and reopening it is how a second worker would make
    a duplicate provider call into the same session.
    """

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    tenant_ref = service._tenant_ref("t")
    try:
        authority = _authority_of(service, "t")
        # Exactly the durable shape a process killed just after claiming leaves:
        # the journal records the claim, and no outcome ever follows it.
        service._continuation.commit(
            authority, authority.claimed("job_" + "0" * 24, time.time()),
        )
        assert _authority_of(service, "t").state == STATE_CLAIMED
    finally:
        service.close(wait=True)

    restarted = _service(tmp_path, workspace, backend)
    try:
        settled = restarted._continuation.load(tenant_ref, _SESSION)
        assert settled.state == STATE_SETTLED, "no stuck claim survives a restart"
        with pytest.raises(ContinuationRefused) as excinfo:
            restarted.submit(
                "t", "after-restart",
                _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_UNAVAILABLE
        assert len(backend.requests) == 1, "restart never re-enters the session"
    finally:
        restarted.close(wait=True)


def test_a_live_claim_survives_a_concurrent_instance_starting_up(tmp_path):
    """Reconciliation must not steal an authority a running job still owns."""

    service, backend, workspace, stopped = _stopped_job(
        tmp_path, plan=("budget", "budget", "ok"),
    )
    try:
        second = service.submit(
            "t", "segment-2", _request(workspace, thread_id=_SESSION, resume=True),
        )
        _wait(service, "t", second.job_id)
        # Generation 2 is open and owned by a settled job; a fresh instance
        # starting against the same state root must leave it alone.
        assert _authority_of(service, "t").generation == 2
        sibling = _service(tmp_path, workspace, backend)
        try:
            live = _authority_of(sibling, "t")
            assert live.state == STATE_OPEN and live.generation == 2
        finally:
            sibling.close(wait=True)
    finally:
        service.close(wait=True)


def test_a_failed_authority_write_never_makes_a_terminal_job_look_resumable(
    tmp_path, monkeypatch,
):
    """A crash *before* the authority exists leaves an honest terminal job."""

    workspace = _workspace(tmp_path)
    backend = SegmentBackend(workspace, ["budget"])
    service = _service(tmp_path, workspace, backend)
    try:
        monkeypatch.setattr(
            ContinuationStore, "create",
            lambda self, authority: (_ for _ in ()).throw(OSError("disk full")),
        )
        queued = service.submit("t", "no-authority", _request(workspace))
        failed = _wait(service, "t", queued.job_id)
        monkeypatch.undo()
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == _BUDGET
        assert failed.continuation_available is False
        assert failed.continuation_generation == 0
        assert _authority_of(service, "t") is None
        with pytest.raises(ContinuationRefused):
            service.submit(
                "t", "hopeful", _request(workspace, thread_id=_SESSION, resume=True),
            )
    finally:
        service.close(wait=True)


def test_a_generation_ceiling_stops_the_chain_instead_of_extending_it(
    tmp_path, monkeypatch,
):
    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        # The real branch, with a smaller ceiling. Rotating thirty-two times
        # would test the loop, not the bound.
        monkeypatch.setattr(service_module, "MAX_CONTINUATION_GENERATION", 1)
        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "too-far", _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_UNAVAILABLE
        assert len(backend.requests) == 1
    finally:
        service.close(wait=True)


# ──────────────────────────────────────────────────────────────────────
# 8. the public projection and the unchanged facade
# ──────────────────────────────────────────────────────────────────────


def test_the_public_receipt_is_bounded_and_reveals_no_authority(tmp_path):
    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        payload = receipt_to_mapping(service.get("t", stopped.job_id))
        assert payload["continuation_available"] is True
        assert payload["continuation_generation"] == 1
        assert payload["failure_code"] == _BUDGET
        assert payload["required_actions"] == ["adjust_coding_job_budget"]
        assert payload["failure_phase"] == "provider"
        assert payload["landable"] is False
        assert payload["implementation_session_id"] == _SESSION

        rendered = json.dumps(payload)
        # The authority record, its file name and the canonical workspace path
        # are private. The session is public because a caller must send it back.
        for forbidden in (
            session_ref(_SESSION), str(workspace), str(service.state_root),
            "continuation_authority", "claimed_by_job_id", "origin_job_id",
            "authorized_config_sha256", "workspace_sha256",
        ):
            assert forbidden not in rendered, forbidden
        # Exactly two continuation fields cross, and no more.
        assert sorted(k for k in payload if k.startswith("continuation")) == [
            "continuation_available", "continuation_generation",
        ]
    finally:
        service.close(wait=True)


def test_the_mcp_surface_is_still_exactly_three_tools_and_old_payloads_work(tmp_path):
    from flyto_ai.coding.mcp_server import CodingMCPServer

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        server = CodingMCPServer(service, tenant_id="t")
        listed = server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })
        names = [tool["name"] for tool in listed["result"]["tools"]]
        assert names == [
            "flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit",
        ]
        # The continuation request is the *existing* payload, unchanged.
        schema = listed["result"]["tools"][0]["inputSchema"]
        request_schema = schema["properties"]["request"]["properties"]
        assert "thread_id" in request_schema and "resume" in request_schema

        called = server.handle({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {
                "name": "flyto_coding_submit",
                "arguments": {
                    "idempotency_key": "segment-2",
                    "request": {
                        "message": "continue the feature",
                        "working_dir": str(workspace),
                        "thread_id": _SESSION,
                        "resume": True,
                    },
                },
            },
        })
        job = called["result"]["structuredContent"]["job"]
        assert called["result"]["isError"] is False
        assert _wait(service, "t", job["job_id"]).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
        assert backend.resumed == [None, _SESSION]

        # A refusal crosses the same facade as bounded, typed data.
        refused = server.handle({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "flyto_coding_submit",
                "arguments": {
                    "idempotency_key": "segment-3",
                    "request": {
                        "message": "continue again",
                        "working_dir": str(workspace),
                        "thread_id": _SESSION,
                        "resume": True,
                    },
                },
            },
        })
        payload = refused["result"]["structuredContent"]
        assert payload["ok"] is False
        assert payload["error"] == CONTINUATION_UNAVAILABLE
        assert payload["details"]["failure_phase"] == "preflight"
    finally:
        service.close(wait=True)


def test_an_ordinary_job_reports_no_continuation_at_all(tmp_path):
    """The additive fields must be invisible to every caller that never resumes."""

    workspace = _workspace(tmp_path)
    backend = SegmentBackend(workspace, ["ok"])
    service = _service(tmp_path, workspace, backend)
    try:
        queued = service.submit("t", "plain", _request(workspace))
        ready = _wait(service, "t", queued.job_id)
        assert ready.continuation_available is False
        assert ready.continuation_generation == 0
        assert backend.resumed == [None]
    finally:
        service.close(wait=True)


# ──────────────────────────────────────────────────────────────────────
# the authority record itself
# ──────────────────────────────────────────────────────────────────────


_NOW = 1_760_000_000.0


def _authority(**overrides) -> ContinuationAuthority:
    fields = dict(
        tenant_ref="a" * 64, backend="claude", session_id=_SESSION,
        job_id="job_" + "1" * 24, origin_job_id="job_" + "1" * 24,
        working_dir="/tmp/ws", workspace_sha256="b" * 64,
        revision_sha256="c" * 64, workspace_manifest_sha256="0" * 64,
        snapshot_policy_sha256=DEFAULT_SNAPSHOT_POLICY.identity(),
        files=("feature.py",),
        authorized_config_sha256="d" * 64,
        contract_snapshot_sha256="f" * 64,
        request_sha256="e" * 64,
        failure_code=_BUDGET, generation=1, sequence=1,
        created_at=_NOW, updated_at=_NOW,
    )
    fields.update(overrides)
    return ContinuationAuthority(**fields)


def test_the_record_digest_covers_every_persisted_field_without_exception():
    """Any stored-but-unhashed field is a field an editor may change for free."""

    authority = _authority()
    mapping = authority.to_mapping()
    assert ContinuationAuthority.from_mapping(mapping) == authority
    for field in _AUTHORITY_FIELDS:
        mutated = dict(mapping)
        current = mutated[field]
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            mutated[field] = (
                ["other.py"] if field == "files"
                else "settled" if field == "state"
                else "/tmp/elsewhere" if field == "working_dir"
                else "job_" + "7" * 24 if field.endswith("job_id")
                else "9" * 64
            )
        else:
            mutated[field] = current + 1
        if mutated[field] == current:
            raise AssertionError("mutation for {} was inert".format(field))
        with pytest.raises(ContinuationCorrupt):
            ContinuationAuthority.from_mapping(mutated)


def test_an_unknown_or_missing_authority_key_is_refused(tmp_path):
    """The probe that got through: an extra key nobody hashed, so nobody saw."""

    mapping = _authority().to_mapping()
    injected = dict(mapping)
    injected["grant_scope"] = "all-workspaces"
    with pytest.raises(ContinuationCorrupt):
        ContinuationAuthority.from_mapping(injected)

    for missing in ("state", "sequence", "workspace_manifest_sha256", "record_sha256"):
        trimmed = {k: v for k, v in mapping.items() if k != missing}
        with pytest.raises(ContinuationCorrupt):
            ContinuationAuthority.from_mapping(trimmed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation", True), ("sequence", True), ("generation", 1.0),
        ("updated_at", -1.0), ("created_at", 10.0),
        ("files", ["b.py", "a.py"]), ("files", ["a.py", "a.py"]), ("files", []),
        ("files", ["/etc/passwd"]), ("working_dir", "relative/path"),
        ("session_id", "host-1"), ("state", "definitely-open"),
        ("generation", 0), ("sequence", 0),
        ("generation", MAX_CONTINUATION_GENERATION + 1),
    ],
)
def test_non_canonical_authority_shapes_are_refused(field, value):
    """Bool-as-number, impossible time, unsorted or duplicated paths, all refused.

    Every probe here is *re-signed* with the bad value in place, so the record
    is entirely self-consistent and the integrity digest has nothing to say
    about it. What refuses it is the parser, which is the point: a digest only
    proves nobody edited the record after it was written, never that what was
    written was ever a legal shape.
    """

    forged = dataclasses.replace(_authority(), **{field: value})
    mapping = forged.to_mapping()
    assert mapping["record_sha256"] == forged.content_digest(), "probe was not re-signed"
    assert mapping[field] == value or field == "files"
    with pytest.raises(ContinuationCorrupt):
        ContinuationAuthority.from_mapping(mapping)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_timestamp_cannot_even_be_signed(value):
    """Refused twice over: canonical JSON rejects it, and so would the parser."""

    forged = dataclasses.replace(_authority(), created_at=value)
    with pytest.raises(ValueError):
        forged.to_mapping()
    mapping = _authority().to_mapping()
    mapping["created_at"] = value
    with pytest.raises(ContinuationCorrupt):
        ContinuationAuthority.from_mapping(mapping)


def test_rotation_moves_the_bytes_and_the_generation_but_never_the_session():
    first = _authority()
    second = first.rotated(
        job_id="job_" + "2" * 24, revision_sha256="f" * 64,
        workspace_manifest_sha256="e" * 64,
        files=("feature.py", "extra.py"), failure_code=_BUDGET, now=_NOW + 1,
    )
    assert second.generation == first.generation + 1
    assert second.sequence == first.sequence + 1
    assert second.session_id == first.session_id
    assert second.origin_job_id == first.origin_job_id
    assert second.state == STATE_OPEN and second.claimed_by_job_id == ""
    assert second.revision_sha256 == "f" * 64
    assert second.workspace_manifest_sha256 == "e" * 64
    assert second.files == ("extra.py", "feature.py")


def test_a_provisional_session_can_never_key_an_authority():
    assert is_continuable_session(_SESSION) is True
    for identity in ("host-1", "route-1", "", "x" * 200, None, True):
        assert is_continuable_session(identity) is False
        with pytest.raises(ValueError):
            session_ref(identity)


def test_the_store_partitions_by_tenant_and_never_writes_the_session_in_a_name(
    tmp_path,
):
    store = ContinuationStore(tmp_path)
    stamped = store.create(_authority())
    path = store.path(stamped.tenant_ref, _SESSION)
    journal = store.journal_path(stamped.tenant_ref, _SESSION)
    assert _SESSION not in str(path) and _SESSION not in str(journal)
    assert path.parent.parent.name == stamped.tenant_ref
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(journal.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert store.load("f" * 64, _SESSION) is None, "another tenant sees nothing"
    assert store.load(stamped.tenant_ref, _SESSION) == stamped
    assert store.open_authority(stamped.tenant_ref, _SESSION) is not None
    store.commit(stamped, stamped.settled(stamped.updated_at + 2))
    assert store.open_authority(stamped.tenant_ref, _SESSION) is None


def test_a_second_create_for_one_session_is_refused(tmp_path):
    """Only the journal may say a session exists, and only once."""

    store = ContinuationStore(tmp_path)
    store.create(_authority())
    with pytest.raises(ContinuationConflict):
        store.create(_authority())


def test_a_commit_against_a_stale_read_is_refused(tmp_path):
    """The compare-and-swap that makes two processes safe, at unit level."""

    store = ContinuationStore(tmp_path)
    opened = store.create(_authority())
    store.commit(opened, opened.claimed("job_" + "2" * 24, opened.updated_at + 1))
    # A second caller still holding the pre-claim read.
    with pytest.raises(ContinuationConflict):
        store.commit(opened, opened.claimed("job_" + "3" * 24, opened.updated_at + 2))


def test_a_settled_authority_can_never_transition_again(tmp_path):
    store = ContinuationStore(tmp_path)
    opened = store.create(_authority())
    settled = store.commit(opened, opened.settled(opened.updated_at + 1))
    for attempt in (
        settled.claimed("job_" + "3" * 24, settled.updated_at + 1),
        settled.rotated(
            job_id="job_" + "3" * 24, revision_sha256="f" * 64,
            workspace_manifest_sha256="e" * 64, files=("a.py",),
            failure_code=_BUDGET, now=settled.updated_at + 1,
        ),
    ):
        with pytest.raises(ContinuationConflict):
            store.commit(settled, attempt)


# ──────────────────────────────────────────────────────────────────────
# the workspace manifest, on its own
# ──────────────────────────────────────────────────────────────────────


def _tree(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.py").write_text("alpha\n", encoding="utf-8")
    (root / "pkg").mkdir()
    (root / "pkg" / "b.py").write_text("beta\n", encoding="utf-8")
    return root


def test_the_manifest_is_deterministic_and_observes_everything(tmp_path):
    root = _tree(tmp_path / "tree")
    baseline = workspace_manifest_digest(str(root))
    assert baseline == workspace_manifest_digest(str(root)), "not deterministic"

    changes = [
        ("new top-level file", lambda: (root / "c.py").write_text("c\n")),
        ("new nested file", lambda: (root / "pkg" / "d.py").write_text("d\n")),
        ("new empty directory", lambda: (root / "empty").mkdir()),
        ("changed bytes", lambda: (root / "a.py").write_text("changed\n")),
        ("executable bit", lambda: (root / "a.py").chmod(
            (root / "a.py").stat().st_mode | stat.S_IXUSR,
        )),
        ("deleted file", lambda: (root / "a.py").unlink()),
        ("deleted directory", lambda: (
            (root / "pkg" / "b.py").unlink(), (root / "pkg" / "d.py").unlink(),
            (root / "pkg").rmdir(),
        )),
    ]
    seen = {baseline}
    for label, apply_change in changes:
        apply_change()
        digest = workspace_manifest_digest(str(root))
        assert digest not in seen, "{} did not move the manifest".format(label)
        seen.add(digest)


def test_two_files_swapping_names_change_the_manifest(tmp_path):
    """A rename pair keeps every byte and every count, and is still drift."""

    root = tmp_path / "swap"
    root.mkdir()
    (root / "one").write_text("first\n", encoding="utf-8")
    (root / "two").write_text("second\n", encoding="utf-8")
    before = workspace_manifest_digest(str(root))
    (root / "one").rename(root / "scratch")
    (root / "two").rename(root / "one")
    (root / "scratch").rename(root / "two")
    assert workspace_manifest_digest(str(root)) != before


@pytest.mark.parametrize(
    "plant",
    [
        lambda root: os.mkfifo(root / "pipe"),
        lambda root: os.mkfifo(root / "pkg" / "nested-pipe"),
    ],
    ids=["fifo", "nested-fifo"],
)
def test_an_undescribable_entry_refuses_the_whole_manifest(tmp_path, plant):
    """A device, socket or FIFO has no content this snapshot could describe."""

    root = _tree(tmp_path / "tree")
    workspace_manifest_digest(str(root))
    plant(root)
    with pytest.raises(WorkspaceUnobservable):
        workspace_manifest_digest(str(root))


@pytest.mark.parametrize(
    "plant",
    [
        lambda root: (root / "link").symlink_to("a.py"),
        lambda root: (root / "pkg" / "link").symlink_to("/etc/passwd"),
        lambda root: (root / "dirlink").symlink_to("pkg"),
        lambda root: (root / "broken").symlink_to("nowhere-at-all"),
    ],
    ids=["file-link", "absolute-link", "directory-link", "dangling-link"],
)
def test_a_symlink_is_recorded_as_itself_and_never_followed(tmp_path, plant):
    """Real `.venv` and `node_modules` trees are full of these.

    Refusing them outright - which the previous version did - meant no real
    repository could ever be snapshotted, so no real repository could ever be
    continued. They are recorded as link objects instead: the path and where it
    points, never what it points at.
    """

    root = _tree(tmp_path / "tree")
    before = workspace_manifest_digest(str(root))
    plant(root)
    after = workspace_manifest_digest(str(root))
    assert after != before, "a new link is workspace drift"
    # Deterministic, and still not followed: the digest does not change when
    # the *target's* content changes through a path outside the tree.
    assert after == workspace_manifest_digest(str(root))


def test_retargeting_a_link_moves_the_digest(tmp_path):
    """The target is part of what the model sees, so it is part of the snapshot."""

    root = _tree(tmp_path / "tree")
    (root / "link").symlink_to("a.py")
    before = workspace_manifest_digest(str(root))
    (root / "link").unlink()
    (root / "link").symlink_to("pkg/b.py")
    assert workspace_manifest_digest(str(root)) != before


def test_a_link_and_a_file_with_the_same_text_are_not_confused(tmp_path):
    """Domain separation: a link to `x` must not hash like a file containing `x`."""

    linked = _tree(tmp_path / "linked")
    (linked / "entry").symlink_to("a.py")
    plain = _tree(tmp_path / "plain")
    (plain / "entry").write_text("a.py", encoding="utf-8")
    assert workspace_manifest_digest(str(linked)) != workspace_manifest_digest(str(plain))


def test_a_link_whose_target_exceeds_the_bound_is_refused(tmp_path, monkeypatch):
    from flyto_ai.coding import continuation as continuation_module

    root = _tree(tmp_path / "tree")
    (root / "link").symlink_to("a.py")
    monkeypatch.setattr(continuation_module, "MAX_MANIFEST_LINK_BYTES", 2)
    with pytest.raises(WorkspaceUnobservable):
        workspace_manifest_digest(str(root))


def test_only_version_control_metadata_is_excluded_and_it_is_named(tmp_path):
    """An exclusion is a blind spot, so there are exactly three and they are listed."""

    from flyto_ai.coding.continuation import MANIFEST_EXCLUDED_DIRECTORIES

    assert MANIFEST_EXCLUDED_DIRECTORIES == (".git", ".hg", ".svn")
    root = _tree(tmp_path / "tree")
    before = workspace_manifest_digest(str(root))
    (root / ".git").mkdir()
    (root / ".git" / "index").write_text("machine state\n", encoding="utf-8")
    assert workspace_manifest_digest(str(root)) == before
    # Everything else that looks like metadata is still observed: build output,
    # caches and dot-directories are all places a file can hide.
    for hidden in (".flyto", "node_modules", "__pycache__", "dist", ".venv"):
        (root / hidden).mkdir()
        (root / hidden / "x").write_text("x\n", encoding="utf-8")
        assert workspace_manifest_digest(str(root)) != before, hidden
        before = workspace_manifest_digest(str(root))


def test_the_manifest_does_not_require_version_control_to_exist(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "only.txt").write_text("no repository here\n", encoding="utf-8")
    assert len(workspace_manifest_digest(str(plain))) == 64


def test_the_manifest_refuses_a_tree_past_its_bounds(tmp_path, monkeypatch):
    from flyto_ai.coding import continuation as continuation_module

    root = _tree(tmp_path / "tree")
    monkeypatch.setattr(continuation_module, "MAX_MANIFEST_ENTRIES", 2)
    with pytest.raises(WorkspaceUnobservable):
        workspace_manifest_digest(str(root))
    monkeypatch.undo()

    monkeypatch.setattr(continuation_module, "MAX_MANIFEST_FILE_BYTES", 2)
    with pytest.raises(WorkspaceUnobservable):
        workspace_manifest_digest(str(root))
    monkeypatch.undo()

    monkeypatch.setattr(continuation_module, "MAX_MANIFEST_DEPTH", 0)
    with pytest.raises(WorkspaceUnobservable):
        workspace_manifest_digest(str(root))


def test_a_workspace_the_host_cannot_describe_is_offered_no_continuation(tmp_path):
    """An unobservable tree stays terminal and truthful rather than resumable."""

    from flyto_ai.agents.claude_code import ClaudeCodingAgent

    class _LinkingBackend(SegmentBackend):
        async def run(self, request):
            response = await super().run(request)
            # A FIFO has no content the snapshot can describe, so the tree
            # stops being bindable. A symlink would be fine - real workspaces
            # are full of them - and is covered separately.
            os.mkfifo(self.workspace / "unreadable")
            return response

    workspace = _workspace(tmp_path)
    backend = _LinkingBackend(workspace, ["budget"])
    service = CodingService(
        lambda store: ClaudeCodingAgent(store, agent=backend),
        state_root=str(tmp_path / "unobservable-state"),
        workspace_roots=(str(workspace),),
        max_workers=1, max_queued=8, require_codex_audit=True,
        implementation_backend="claude",
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        approval_policy=ApprovalPolicy.NEVER,
    )
    try:
        queued = service.submit("t", "unobservable", _request(workspace))
        failed = _wait(service, "t", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == _BUDGET
        assert failed.continuation_available is False
        assert _authority_of(service, "t") is None
    finally:
        service.close(wait=True)


# ──────────────────────────────────────────────────────────────────────
# the state filesystem boundary
# ──────────────────────────────────────────────────────────────────────


def test_a_symlinked_continuation_directory_is_never_followed(tmp_path):
    """An attacker's link must not become this host's state directory."""

    state_root = tmp_path / "state"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    tenant_ref = "a" * 64
    (state_root / "tenants" / tenant_ref).mkdir(parents=True)
    (state_root / "tenants" / tenant_ref / "continuation").symlink_to(elsewhere)

    store = ContinuationStore(state_root)
    with pytest.raises(OSError):
        store.create(_authority(tenant_ref=tenant_ref))
    assert list(elsewhere.iterdir()) == [], "nothing was written through the link"
    assert store.load(tenant_ref, _SESSION) is None


def test_a_symlinked_session_file_is_never_followed(tmp_path):
    state_root = tmp_path / "state"
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    store = ContinuationStore(state_root)
    stamped = store.create(_authority())
    path = store.path(stamped.tenant_ref, _SESSION)
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ContinuationCorrupt):
        store._read(stamped.tenant_ref, session_ref(_SESSION) + ".json")
    assert store.load(stamped.tenant_ref, _SESSION) is None
    assert target.read_text() == "{}", "the link target was not written through"


def test_a_truncated_journal_or_authority_is_unusable(tmp_path):
    store = ContinuationStore(tmp_path)
    stamped = store.create(_authority())
    tenant_ref = stamped.tenant_ref
    journal = store.journal_path(tenant_ref, _SESSION)
    authority = store.path(tenant_ref, _SESSION)
    whole_journal = journal.read_bytes()
    whole_authority = authority.read_bytes()

    journal.write_bytes(whole_journal[:-6])
    assert store.load(tenant_ref, _SESSION) is None
    journal.write_bytes(whole_journal)
    assert store.load(tenant_ref, _SESSION) is not None

    authority.write_bytes(whole_authority[: len(whole_authority) // 2])
    assert store.load(tenant_ref, _SESSION) is None
    # A missing body is unavailable, never "the journal says so, close enough".
    authority.unlink()
    assert store.load(tenant_ref, _SESSION) is None


def test_an_interruption_at_any_persistence_boundary_fails_closed(tmp_path):
    """Each half of a transition, alone, must read as unavailable.

    The journal is appended first on purpose. Losing the second half costs
    availability; losing them the other way round would leave a body no
    transition ever recorded, which is the shape of a forgery.
    """

    store = ContinuationStore(tmp_path)
    stamped = store.create(_authority())
    tenant_ref = stamped.tenant_ref
    journal = store.journal_path(tenant_ref, _SESSION)
    authority = store.path(tenant_ref, _SESSION)
    opened_journal = journal.read_bytes()
    opened_body = authority.read_bytes()

    claimed = store.commit(
        stamped, stamped.claimed("job_" + "2" * 24, stamped.updated_at + 1),
    )
    claimed_journal = journal.read_bytes()
    claimed_body = authority.read_bytes()

    # Interrupted after the journal append, before the body was replaced.
    authority.write_bytes(opened_body)
    assert store.load(tenant_ref, _SESSION) is None
    # Interrupted before the append but somehow after the body: also refused.
    journal.write_bytes(opened_journal)
    authority.write_bytes(claimed_body)
    assert store.load(tenant_ref, _SESSION) is None
    # Both halves present and agreeing is the only readable state.
    journal.write_bytes(claimed_journal)
    authority.write_bytes(claimed_body)
    assert store.load(tenant_ref, _SESSION) == claimed


# ──────────────────────────────────────────────────────────────────────
# a real multiprocess race
# ──────────────────────────────────────────────────────────────────────


def _claim_in_a_separate_process(
    state_root, tenant_ref, session, job_id, barrier, outcome_path,
):
    """Runs in a fresh interpreter: no shared object, no shared lock, no threads.

    The outcome is written to its own file rather than sent over a queue,
    because the only shared state this test is entitled to assume is the one
    the mechanism itself uses: a directory.
    """

    from flyto_ai.coding.continuation import ContinuationStore

    store = ContinuationStore(Path(state_root))
    opened = store.open_authority(tenant_ref, session)
    # Every process reads the same open authority *before* any of them writes,
    # so the race is on the transition rather than on the read.
    barrier.wait(timeout=60)
    if opened is None:
        result = "unavailable"
    else:
        try:
            store.commit(opened, opened.claimed(job_id, time.time()))
        except Exception:
            result = "unavailable"
        else:
            result = "claimed"
    Path(outcome_path).write_text(result, encoding="utf-8")


def test_separate_os_processes_racing_one_transition_have_one_winner(tmp_path):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    state_root = tmp_path / "race-state"
    outcomes_dir = tmp_path / "outcomes"
    outcomes_dir.mkdir()
    store = ContinuationStore(state_root)
    stamped = store.create(_authority())
    tenant_ref = stamped.tenant_ref

    barrier = context.Barrier(_RACERS)
    workers = []
    for index in range(_RACERS):
        outcome_path = outcomes_dir / "{}.txt".format(index)
        workers.append((outcome_path, context.Process(
            target=_claim_in_a_separate_process,
            args=(
                str(state_root), tenant_ref, _SESSION,
                "job_" + "{:024x}".format(index), barrier, str(outcome_path),
            ),
        )))
    for _, worker in workers:
        worker.start()
    for _, worker in workers:
        worker.join(timeout=120)
        assert worker.exitcode == 0, "a racing process crashed"
    outcomes = [path.read_text(encoding="utf-8") for path, _ in workers]

    assert len(outcomes) == _RACERS
    assert outcomes.count("claimed") == 1, outcomes
    assert outcomes.count("unavailable") == _RACERS - 1, outcomes

    # The durable record agrees with exactly one winner, and the journal holds
    # exactly the two transitions that really happened.
    final = store.load(tenant_ref, _SESSION)
    assert final is not None and final.state == STATE_CLAIMED
    entries = store.journal(tenant_ref, _SESSION)
    assert [entry.sequence for entry in entries] == [1, 2]
    assert [entry.state for entry in entries] == [STATE_OPEN, STATE_CLAIMED]
    assert entries[-1].authority_sha256 == final.content_digest()


def test_a_reconciling_restart_never_reopens_a_consumed_transition(tmp_path):
    """Crash recovery may settle forward. It may never rewind the chain."""

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    tenant_ref = service._tenant_ref("t")
    state_root = Path(service.state_root)
    try:
        authority = _authority_of(service, "t")
        service._continuation.commit(
            authority, authority.claimed("job_" + "0" * 24, time.time()),
        )
    finally:
        service.close(wait=True)

    for _ in range(3):
        restarted = _service(tmp_path, workspace, backend)
        try:
            settled = restarted._continuation.load(tenant_ref, _SESSION)
            assert settled.state == STATE_SETTLED
        finally:
            restarted.close(wait=True)

    # Reconciliation ran three times and appended exactly one settle.
    states = [
        entry.state
        for entry in ContinuationStore(state_root).journal(tenant_ref, _SESSION)
    ]
    assert states == [STATE_OPEN, STATE_CLAIMED, STATE_SETTLED]
    assert len(backend.requests) == 1


# ──────────────────────────────────────────────────────────────────────
# round 11: the five reproduced blockers
# ──────────────────────────────────────────────────────────────────────


def test_only_root_version_control_state_is_excluded(tmp_path):
    """A nested directory merely named `.git` is ordinary workspace content."""

    root = _tree(tmp_path / "tree")
    (root / ".git").mkdir()
    (root / ".git" / "index").write_text("machine state\n", encoding="utf-8")
    baseline = workspace_manifest_digest(str(root))
    (root / ".git" / "index").write_text("rewritten by tooling\n", encoding="utf-8")
    assert workspace_manifest_digest(str(root)) == baseline

    # ...but a `.git` that is not the root's is observed like anything else.
    nested = root / "pkg" / ".git"
    nested.mkdir()
    (nested / "sample").write_text("a vendored fixture\n", encoding="utf-8")
    assert workspace_manifest_digest(str(root)) != baseline


def test_a_symlinked_workspace_root_is_refused(tmp_path):
    """Blocker 2. `Path.resolve()` followed the link and snapshotted elsewhere."""

    real = _tree(tmp_path / "real")
    link = tmp_path / "link-to-real"
    link.symlink_to(real)
    assert len(workspace_manifest_digest(str(real))) == 64
    with pytest.raises(WorkspaceUnobservable):
        workspace_manifest_digest(str(link))


def test_a_symlinked_ancestor_of_the_workspace_is_refused(tmp_path):
    """The whole ancestry is checked, not only the final component."""

    real = _tree(tmp_path / "outer" / "inner")
    bridge = tmp_path / "bridge"
    bridge.symlink_to(tmp_path / "outer")
    assert len(workspace_manifest_digest(str(real))) == 64
    with pytest.raises(WorkspaceUnobservable):
        workspace_manifest_digest(str(bridge / "inner"))


def test_a_relative_workspace_path_is_refused(tmp_path):
    with pytest.raises(WorkspaceUnobservable):
        workspace_manifest_digest("relative/path")
    with pytest.raises(WorkspaceUnobservable):
        workspace_manifest_digest("")


def test_a_directory_swapped_for_a_link_mid_walk_is_detected(tmp_path, monkeypatch):
    """Blocker 2, deterministically: no sleeping, no timing.

    The swap is performed by the instrumented `scandir` itself, at exactly the
    moment between listing a directory and descending into it. That is the only
    window a check-then-open design leaves, so it is the window the test opens.
    """

    from flyto_ai.coding import continuation as continuation_module

    root = _tree(tmp_path / "tree")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "planted.py").write_text("# not yours\n", encoding="utf-8")

    real_scandir = os.scandir
    swapped = {"done": False}

    def scandir_then_swap(target):
        result = real_scandir(target)
        if not swapped["done"]:
            swapped["done"] = True
            # `pkg` has been listed as a directory; replace it with a link
            # before the walk can descend.
            import shutil

            shutil.rmtree(root / "pkg")
            (root / "pkg").symlink_to(elsewhere)
        return result

    monkeypatch.setattr(continuation_module.os, "scandir", scandir_then_swap)
    with pytest.raises(WorkspaceUnobservable):
        workspace_manifest_digest(str(root))


def test_a_state_root_behind_a_symlinked_ancestor_is_refused(tmp_path):
    """Blocker 3. `mkdir(parents=True)` created state on the far side of a link."""

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    bridge = tmp_path / "bridge"
    bridge.symlink_to(elsewhere)

    store = ContinuationStore(bridge / "state")
    with pytest.raises((OSError, ContinuationCorrupt)):
        store.create(_authority())
    assert list(elsewhere.iterdir()) == [], "state was written through the link"


def test_state_directories_and_files_are_created_private(tmp_path):
    store = ContinuationStore(tmp_path / "state")
    stamped = store.create(_authority())
    body = store.path(stamped.tenant_ref, _SESSION)
    journal = store.journal_path(stamped.tenant_ref, _SESSION)
    assert stat.S_IMODE(body.stat().st_mode) == 0o600
    assert stat.S_IMODE(journal.stat().st_mode) == 0o600
    for directory in (body.parent, body.parent.parent, body.parent.parent.parent):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_a_group_readable_state_file_is_refused(tmp_path):
    store = ContinuationStore(tmp_path / "state")
    stamped = store.create(_authority())
    store.path(stamped.tenant_ref, _SESSION).chmod(0o640)
    assert store.load(stamped.tenant_ref, _SESSION) is None


def test_a_hard_linked_state_file_is_refused(tmp_path):
    """Another name for the same inode is another owner for the same bytes."""

    store = ContinuationStore(tmp_path / "state")
    stamped = store.create(_authority())
    body = store.path(stamped.tenant_ref, _SESSION)
    os.link(body, body.parent / "attacker-owned-name")
    assert store.load(stamped.tenant_ref, _SESSION) is None

    journal = store.journal_path(stamped.tenant_ref, _SESSION)
    os.link(journal, journal.parent / "attacker-owned-journal")
    with pytest.raises(ContinuationCorrupt):
        store.commit(stamped, stamped.claimed("job_" + "2" * 24, stamped.updated_at + 1))


def test_a_predictable_temporary_name_cannot_be_pre_created(tmp_path):
    """Blocker: `O_CREAT|O_TRUNC` on a PID-derived name reuses attacker residue."""

    store = ContinuationStore(tmp_path / "state")
    stamped = store.create(_authority())
    directory = store.path(stamped.tenant_ref, _SESSION).parent
    name = session_ref(_SESSION) + ".json"
    residue = directory / ".{}.{}.tmp".format(name, os.getpid())
    residue.write_text("planted", encoding="utf-8")

    store.commit(stamped, stamped.claimed("job_" + "2" * 24, stamped.updated_at + 1))
    # The transition succeeded without ever touching the guessable name.
    assert residue.read_text(encoding="utf-8") == "planted"
    assert store.load(stamped.tenant_ref, _SESSION).state == STATE_CLAIMED
    # And no temporary file was left behind.
    assert sorted(p.name for p in directory.iterdir()) == sorted([
        name, session_ref(_SESSION) + ".journal", residue.name,
    ])


def test_a_state_file_that_grows_while_it_is_read_is_refused(tmp_path, monkeypatch):
    """The bound is enforced while reading, not only from the first stat."""

    from flyto_ai.coding import continuation as continuation_module

    store = ContinuationStore(tmp_path / "state")
    stamped = store.create(_authority())
    monkeypatch.setattr(continuation_module, "MAX_STATE_FILE_BYTES", 8)
    assert store.load(stamped.tenant_ref, _SESSION) is None


@pytest.mark.parametrize(
    "boundary",
    ["directory-create", "journal-fsync", "body-fsync", "final-directory-fsync"],
)
def test_a_durability_failure_is_never_reported_as_a_successful_grant(
    tmp_path, monkeypatch, boundary,
):
    """Blocker 4. An undurable write is a failed write.

    The previous version swallowed directory-fsync failures on the grounds that
    some filesystems refuse them. But the caller has already been told the
    authority exists; after a power loss the journal or the body can be missing
    and the continuation regresses. Losing availability is safe. Reporting a
    durable grant that is not durable is not.
    """

    from flyto_ai.coding import continuation as continuation_module

    store = ContinuationStore(tmp_path / "state")
    real_fsync = os.fsync
    calls = {"n": 0}

    def failing_fsync(handle):
        info = os.fstat(handle)
        is_directory = stat.S_ISDIR(info.st_mode)
        calls["n"] += 1
        if boundary == "directory-create" and is_directory and calls["n"] <= 1:
            raise OSError(5, "injected")
        if boundary == "journal-fsync" and not is_directory:
            raise OSError(5, "injected")
        if boundary == "body-fsync" and not is_directory and calls["n"] > 2:
            raise OSError(5, "injected")
        if boundary == "final-directory-fsync" and is_directory:
            raise OSError(5, "injected")
        return real_fsync(handle)

    monkeypatch.setattr(continuation_module.os, "fsync", failing_fsync)
    with pytest.raises((ContinuationCorrupt, OSError)):
        store.create(_authority())
    monkeypatch.undo()

    # Whatever landed on disk, it is never a usable authority.
    assert store.load("a" * 64, _SESSION) is None
    assert store.open_authority("a" * 64, _SESSION) is None


def test_an_illegal_generation_jump_is_refused_at_commit_and_at_parse(tmp_path):
    """Blocker 5. A claim consumes a segment; it does not mint the next one."""

    store = ContinuationStore(tmp_path / "state")
    opened = store.create(_authority())

    forged = dataclasses.replace(
        opened,
        state=STATE_CLAIMED,
        claimed_by_job_id="job_" + "2" * 24,
        generation=opened.generation + 1,
        sequence=opened.sequence + 1,
        updated_at=opened.updated_at + 1,
    )
    # Entirely self-consistent: correctly re-signed, legal state pair, higher
    # generation. Only the exact transition table refuses it.
    assert forged.to_mapping()["record_sha256"] == forged.content_digest()
    with pytest.raises(ContinuationConflict):
        store.commit(opened, forged)
    with pytest.raises(ContinuationCorrupt):
        check_transition(opened, forged)

    # The same refusal on the way back in, from a forged journal.
    entries = store.journal(opened.tenant_ref, _SESSION)
    tail = entries[-1]
    jumped = JournalEntry(
        sequence=tail.sequence + 1,
        generation=tail.generation + 1,
        state=STATE_CLAIMED,
        authority_sha256=forged.content_digest(),
        previous_entry_sha256=tail.entry_digest(),
        recorded_at=tail.recorded_at + 1,
    )
    raw = store.journal_path(opened.tenant_ref, _SESSION).read_bytes()
    with pytest.raises(ContinuationCorrupt):
        read_journal(raw + jumped.to_line())

    # ...and the legal claim, at the same generation, is accepted.
    legal = dataclasses.replace(forged, generation=opened.generation)
    assert store.commit(opened, legal).state == STATE_CLAIMED


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_ref", "f" * 64),
        ("backend", "codex"),
        ("session_id", _OTHER_SESSION),
        ("origin_job_id", "job_" + "9" * 24),
        ("working_dir", "/tmp/somewhere-else"),
        ("workspace_sha256", "1" * 64),
        ("authorized_config_sha256", "2" * 64),
        ("request_sha256", "3" * 64),
    ],
)
def test_a_transition_may_not_change_what_the_authority_is(tmp_path, field, value):
    """Identity is not a field a transition gets to edit."""

    store = ContinuationStore(tmp_path / "state")
    opened = store.create(_authority())
    forged = dataclasses.replace(
        opened,
        state=STATE_CLAIMED,
        claimed_by_job_id="job_" + "2" * 24,
        sequence=opened.sequence + 1,
        updated_at=opened.updated_at + 1,
        **{field: value},
    )
    with pytest.raises(ContinuationConflict):
        store.commit(opened, forged)


def test_a_claim_may_not_quietly_rewrite_the_segment_it_claims(tmp_path):
    store = ContinuationStore(tmp_path / "state")
    opened = store.create(_authority())
    for field, value in (
        ("revision_sha256", "7" * 64),
        ("workspace_manifest_sha256", "8" * 64),
        ("files", ("other.py",)),
        ("job_id", "job_" + "5" * 24),
    ):
        forged = dataclasses.replace(
            opened,
            state=STATE_CLAIMED,
            claimed_by_job_id="job_" + "2" * 24,
            sequence=opened.sequence + 1,
            updated_at=opened.updated_at + 1,
            **{field: value},
        )
        with pytest.raises(ContinuationConflict):
            store.commit(opened, forged)


@pytest.mark.parametrize(
    "forge",
    [
        lambda a: dataclasses.replace(
            a, state=STATE_OPEN, claimed_by_job_id="job_" + "2" * 24,
            sequence=a.sequence + 1, updated_at=a.updated_at + 1,
        ),
        lambda a: dataclasses.replace(
            a, state=STATE_CLAIMED, claimed_by_job_id="",
            sequence=a.sequence + 1, updated_at=a.updated_at + 1,
        ),
        lambda a: dataclasses.replace(
            a, state=STATE_CLAIMED, claimed_by_job_id="job_" + "2" * 24,
            sequence=a.sequence + 1, updated_at=a.updated_at - 100,
        ),
    ],
    ids=["open-with-claimant", "claimed-without-claimant", "backwards-in-time"],
)
def test_state_semantics_are_enforced_on_every_transition(tmp_path, forge):
    store = ContinuationStore(tmp_path / "state")
    opened = store.create(_authority())
    with pytest.raises(ContinuationConflict):
        store.commit(opened, forge(opened))


def test_an_unrecognized_stop_code_cannot_be_carried_by_an_authority():
    """An authority exists because of a bounded stop, and only because of one."""

    forged = dataclasses.replace(_authority(), failure_code="something_else")
    with pytest.raises(ContinuationCorrupt):
        ContinuationAuthority.from_mapping(forged.to_mapping())


def test_a_directory_swapped_for_another_directory_mid_walk_is_detected(
    tmp_path, monkeypatch,
):
    """`O_NOFOLLOW` refuses a link. It has nothing to say about a real directory.

    Renaming a *different* real directory into a listed name passes every
    open-time check, so the only thing that catches it is comparing the inode
    the listing described against the inode the descriptor actually opened.

    The swap is timed to happen after the containing directory has already been
    enumerated *and* re-checked, so the parent-mutation guard cannot be what
    refuses it. Entries sort `a.py`, `pkg`, `zzz`: the swap of `zzz` is
    performed while `pkg` is being listed, which is strictly after the root's
    own before/after comparison has passed.
    """

    from flyto_ai.coding import continuation as continuation_module

    root = _tree(tmp_path / "tree")
    (root / "zzz").mkdir()
    (root / "zzz" / "original.py").write_text("# original\n", encoding="utf-8")
    impostor = tmp_path / "impostor"
    impostor.mkdir()
    (impostor / "planted.py").write_text("# not yours\n", encoding="utf-8")

    real_scandir = os.scandir
    seen = {"calls": 0}

    def scandir_then_swap(target):
        result = real_scandir(target)
        seen["calls"] += 1
        if seen["calls"] == 2:
            import shutil

            shutil.rmtree(root / "zzz")
            impostor.rename(root / "zzz")
        return result

    monkeypatch.setattr(continuation_module.os, "scandir", scandir_then_swap)
    with pytest.raises(WorkspaceUnobservable):
        workspace_manifest_digest(str(root))


def test_a_state_file_that_grows_during_the_read_itself_is_refused(
    tmp_path, monkeypatch,
):
    """The in-loop bound, isolated from the initial size check.

    A file that is small when it is stat-ed and then grows underneath the
    reader would otherwise be buffered without limit, so the bound has to be
    re-checked against what has actually been read.
    """

    from flyto_ai.coding import continuation as continuation_module

    store = ContinuationStore(tmp_path / "state")
    stamped = store.create(_authority())
    bound = continuation_module.MAX_STATE_FILE_BYTES
    chunk = 1024 * 1024
    served = {"n": 0}

    def endless_read(handle, size):
        # A file that keeps producing bytes after it was stat-ed as small.
        # Twice the bound is available; a reader that only compares sizes at
        # the end would buffer all of it first.
        if served["n"] * chunk >= bound * 2:
            return b""
        served["n"] += 1
        return b"x" * chunk

    monkeypatch.setattr(continuation_module.os, "read", endless_read)
    assert store.load(stamped.tenant_ref, _SESSION) is None
    # The load-bearing assertion is not that it refused - a later size
    # comparison would also refuse - but that it stopped buffering as soon as
    # the bound was passed, rather than after reading everything on offer.
    assert served["n"] <= bound // chunk + 1, served["n"]


def test_the_journal_lock_is_held_across_the_read_and_the_append(tmp_path, monkeypatch):
    """Deterministic, single-process proof that the critical section is locked.

    The multiprocess race is the real evidence, but it is probabilistic: the
    compare-and-swap against the journal tail already refuses most losers on
    its own, so removing the lock only sometimes corrupts a run. What the lock
    uniquely provides is that the tail read and the append cannot be
    interleaved at all - and that can be observed directly.

    `flock` is held per open-file-description, so a second `os.open` of the
    same journal contends exactly as another process would, even from here.
    """

    import errno as errno_module
    import fcntl

    store = ContinuationStore(tmp_path / "state")
    opened = store.create(_authority())
    journal = store.journal_path(opened.tenant_ref, _SESSION)
    observed = {}
    real_check = ContinuationStore._check_transition

    def probe(previous, updated, entries, tail):
        handle = os.open(journal, os.O_RDWR)
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                observed["refused"] = exc.errno
            else:
                observed["refused"] = None
                fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)
        return real_check(previous, updated, entries, tail)

    monkeypatch.setattr(ContinuationStore, "_check_transition", staticmethod(probe))
    store.commit(opened, opened.claimed("job_" + "2" * 24, opened.updated_at + 1))

    assert observed["refused"] in (
        errno_module.EWOULDBLOCK, errno_module.EAGAIN, errno_module.EACCES,
    ), "the journal was not locked while its tail was being decided"


# ──────────────────────────────────────────────────────────────────────
# round 12: portable real-shaped trees, in place of host repository scans
# ──────────────────────────────────────────────────────────────────────


def _real_shaped_tree(root: Path, *, packages: int = 40, per_package: int = 25) -> Path:
    """A tree with the properties that mattered about the real repositories.

    Depth, breadth, a dependency-shaped subtree full of symlinks, a large-ish
    file, and a control-plane runtime directory. Everything a host scan proved
    is proved here, in a directory this test owns and in bounded time.
    """

    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# project\n", encoding="utf-8")
    (root / "big.bin").write_bytes(b"\0" * (2 * 1024 * 1024))
    for index in range(packages):
        package = root / "src" / "pkg{:03d}".format(index) / "nested"
        package.mkdir(parents=True)
        for item in range(per_package):
            (package / "m{:03d}.py".format(item)).write_text(
                "value = {}\n".format(index * 1000 + item), encoding="utf-8",
            )
    # A dependency tree, with the link density that made the previous
    # symlink-refusing version unable to snapshot anything real.
    modules = root / "node_modules" / "dep"
    modules.mkdir(parents=True)
    (modules / "index.js").write_text("module.exports = 1;\n", encoding="utf-8")
    (root / "node_modules" / ".bin").mkdir()
    (root / "node_modules" / ".bin" / "dep").symlink_to("../dep/index.js")
    (root / "venv-link").symlink_to("src")
    # A deep chain, well inside the depth bound.
    deep = root
    for level in range(12):
        deep = deep / "d{}".format(level)
        deep.mkdir()
    (deep / "leaf.txt").write_text("bottom\n", encoding="utf-8")
    # Control-plane runtime state, of the shape that never stops moving.
    runtime = root / ".flyto-index"
    runtime.mkdir()
    (runtime / "task-runs.sqlite").write_bytes(b"live database v1")
    return root


@pytest.fixture(scope="module")
def real_shaped(tmp_path_factory):
    return _real_shaped_tree(tmp_path_factory.mktemp("shaped") / "workspace")


def test_the_snapshot_succeeds_and_is_stable_on_a_real_shaped_tree(real_shaped):
    """The property the host scans were standing in for.

    Roughly a thousand files, a thousand directories, symlinks, a two-megabyte
    blob and twelve levels of nesting - all of which the previous bounds and
    the previous symlink refusal would have rejected.
    """

    first = workspace_manifest_digest(str(real_shaped))
    second = workspace_manifest_digest(str(real_shaped))
    assert len(first) == 64
    assert first == second, "the snapshot is not deterministic"


@pytest.mark.parametrize(
    "change",
    [
        lambda root: (root / "src" / "planted.py").write_text("# new\n"),
        lambda root: (root / "node_modules" / "dep" / "extra.js").write_text("x\n"),
        lambda root: (root / "README.md").write_text("# changed\n"),
        lambda root: (root / "README.md").unlink(),
        lambda root: (root / "README.md").chmod(0o755),
        lambda root: (
            (root / "venv-link").unlink(), (root / "venv-link").symlink_to("node_modules"),
        ),
        lambda root: (
            (root / "README.md").unlink(), (root / "README.md").mkdir(),
        ),
    ],
    ids=["add", "add-in-dependency-tree", "modify", "delete", "chmod",
         "retarget-link", "retype"],
)
def test_source_drift_of_every_kind_moves_the_snapshot(tmp_path, change):
    root = _real_shaped_tree(tmp_path / "workspace", packages=3, per_package=3)
    before = workspace_manifest_digest(str(root))
    change(root)
    assert workspace_manifest_digest(str(root)) != before


# ──────────────────────────────────────────────────────────────────────
# round 12, blocker 2: source authority and control-plane runtime state
# ──────────────────────────────────────────────────────────────────────


_STRICT_POLICY = SnapshotPolicy(
    runtime_state_names=(".flyto-index",),
    rationale="host-owned Indexer pre/post gates revalidate this tree",
)


def test_the_default_projection_has_no_blind_spot(tmp_path):
    """Everything that is not version control is source, including caches."""

    assert DEFAULT_SNAPSHOT_POLICY.runtime_state_names == ()
    root = _real_shaped_tree(tmp_path / "workspace", packages=2, per_package=2)
    before = workspace_manifest_digest(str(root))
    # The very directory a policy might classify is observed by default.
    (root / ".flyto-index" / "task-runs.sqlite").write_bytes(b"live database v2")
    assert workspace_manifest_digest(str(root)) != before


def test_an_explicit_policy_stops_control_plane_churn_moving_the_source_digest(
    tmp_path,
):
    """The production failure, and the whole reason a policy exists.

    Another Indexer rewrites `.flyto-index/task-runs.sqlite` continuously, so a
    whole-tree digest of a real product never repeats and continuation is
    refused forever - for a reason that has nothing to do with the source.
    """

    root = _real_shaped_tree(tmp_path / "workspace", packages=2, per_package=2)
    before = workspace_manifest_digest(str(root), _STRICT_POLICY)
    for revision in range(3):
        (root / ".flyto-index" / "task-runs.sqlite").write_bytes(
            b"live database v%d" % revision,
        )
        (root / ".flyto-index" / "run-{}.json".format(revision)).write_text("{}")
    assert workspace_manifest_digest(str(root), _STRICT_POLICY) == before
    # ...and the same churn still moves the default projection.
    assert workspace_manifest_digest(str(root)) != workspace_manifest_digest(
        str(root), _STRICT_POLICY,
    )


def test_a_classified_directory_is_still_present_or_absent(tmp_path):
    """Its contents are somebody else's business. Its existence is not."""

    root = _real_shaped_tree(tmp_path / "workspace", packages=2, per_package=2)
    before = workspace_manifest_digest(str(root), _STRICT_POLICY)
    import shutil

    shutil.rmtree(root / ".flyto-index")
    assert workspace_manifest_digest(str(root), _STRICT_POLICY) != before


def test_classification_is_exactly_root_relative(tmp_path):
    """A nested namesake is ordinary workspace content, not runtime state."""

    root = _real_shaped_tree(tmp_path / "workspace", packages=2, per_package=2)
    nested = root / "src" / ".flyto-index"
    nested.mkdir()
    (nested / "fixture.json").write_text("{}", encoding="utf-8")
    before = workspace_manifest_digest(str(root), _STRICT_POLICY)
    (nested / "fixture.json").write_text('{"changed": true}', encoding="utf-8")
    assert workspace_manifest_digest(str(root), _STRICT_POLICY) != before


def test_source_drift_is_still_detected_under_a_policy(tmp_path):
    root = _real_shaped_tree(tmp_path / "workspace", packages=2, per_package=2)
    before = workspace_manifest_digest(str(root), _STRICT_POLICY)
    (root / "src" / "planted.py").write_text("# unattributed\n", encoding="utf-8")
    assert workspace_manifest_digest(str(root), _STRICT_POLICY) != before


def test_the_projection_is_part_of_what_is_digested(tmp_path):
    """Two policies never produce the same digest for the same bytes."""

    root = _real_shaped_tree(tmp_path / "workspace", packages=2, per_package=2)
    import shutil

    shutil.rmtree(root / ".flyto-index")
    # Even with nothing classified present, the identities differ.
    assert workspace_manifest_digest(str(root)) != workspace_manifest_digest(
        str(root), _STRICT_POLICY,
    )


@pytest.mark.parametrize(
    "names",
    [
        ("../escape",), ("src/nested",), (".",), ("..",), ("",),
        (".git",), ("a" * 200,), ("b", "a"), ("a", "a"),
        (".flyto-index", "node_modules", ".venv", "dist", "build"),
    ],
    ids=["traversal", "separator", "dot", "dotdot", "empty", "version-control",
         "overlong", "unsorted", "duplicate", "too-many"],
)
def test_a_malformed_or_greedy_policy_is_refused(names):
    """A policy is a short list of named exceptions or it is not a policy."""

    with pytest.raises(SnapshotPolicyInvalid):
        SnapshotPolicy(runtime_state_names=names)


def test_a_policy_identity_covers_its_stated_reason():
    """Re-purposing an exclusion silently is a policy change."""

    first = SnapshotPolicy(runtime_state_names=(".flyto-index",), rationale="gates")
    second = SnapshotPolicy(runtime_state_names=(".flyto-index",), rationale="other")
    assert first.identity() != second.identity()


def test_only_a_strict_indexer_backed_route_may_classify_runtime_state(tmp_path):
    """The justification is the gate, not the intention.

    Two of the four combinations below cannot even be constructed:
    `CodingRoutePolicy` already refuses a strict route without a required
    Indexer. That is the invariant this classification leans on, so it is
    asserted rather than assumed - and the defensive branches are exercised
    with duck-typed stand-ins, because no real policy can reach them.
    """

    from flyto_ai.coding.contracts import CapabilitySpec
    from flyto_ai.coding.route import CodingRoutePolicy

    workspace = _workspace(tmp_path)
    plain = _service(tmp_path, workspace, SegmentBackend(workspace, []))
    try:
        assert plain.snapshot_policy == DEFAULT_SNAPSHOT_POLICY
    finally:
        plain.close(wait=True)

    def _spec(required, name="flyto-indexer"):
        return CapabilitySpec(
            name=name, argv=("python3", "-m", "flyto_ai.mcp_server"),
            required=required, required_tools=("task",), allowed_tools=("task",),
            tool_permissions=(("task", "read_only"),),
        )

    def _route(**overrides):
        fields = {
            "strict": True,
            "indexer": _spec(True),
            "blueprint": _spec(True, "flyto-blueprint"),
            "core_enabled": True,
        }
        fields.update(overrides)
        return CodingRoutePolicy(**fields)

    # The route type itself guarantees a strict route has a required Indexer.
    for indexer in (None, _spec(False)):
        with pytest.raises(ValueError):
            _route(indexer=indexer)

    # A non-strict route, even with a required Indexer, classifies nothing:
    # the gates only run on the strict route.
    assert CodingService._startup_snapshot_policy(
        _route(strict=False),
    ) == DEFAULT_SNAPSHOT_POLICY
    assert CodingService._startup_snapshot_policy(None) == DEFAULT_SNAPSHOT_POLICY
    # Defensive branches, unreachable through the real policy type.
    for route in (
        types.SimpleNamespace(strict=True, indexer=None),
        types.SimpleNamespace(strict=True, indexer=types.SimpleNamespace(required=False)),
        types.SimpleNamespace(strict=False, indexer=types.SimpleNamespace(required=True)),
    ):
        assert CodingService._startup_snapshot_policy(route) == DEFAULT_SNAPSHOT_POLICY

    granted = CodingService._startup_snapshot_policy(
        _route(),
    )
    assert granted.runtime_state_names == (".flyto-index",)
    assert granted.identity() != DEFAULT_SNAPSHOT_POLICY.identity()


def test_an_authority_granted_under_another_projection_is_refused(tmp_path):
    """Policy drift refuses before the provider is contacted."""

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        authority = _authority_of(service, "t")
        assert authority.snapshot_policy_sha256 == DEFAULT_SNAPSHOT_POLICY.identity()
        # The same service, restarted with a projection that would observe the
        # tree differently. Nothing about the bytes changed.
        service.snapshot_policy = _STRICT_POLICY
        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "policy-drift",
                _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_POLICY_CHANGED
        assert len(backend.requests) == 1, "no provider call followed the refusal"
    finally:
        service.close(wait=True)


def test_the_projection_is_frozen_for_the_life_of_a_session(tmp_path):
    """A rotation may not quietly re-observe the tree a different way."""

    store = ContinuationStore(tmp_path / "state")
    opened = store.create(_authority())
    forged = dataclasses.replace(
        opened, state=STATE_CLAIMED, claimed_by_job_id="job_" + "2" * 24,
        sequence=opened.sequence + 1, updated_at=opened.updated_at + 1,
        snapshot_policy_sha256=_STRICT_POLICY.identity(),
    )
    with pytest.raises(ContinuationConflict):
        store.commit(opened, forged)


# ──────────────────────────────────────────────────────────────────────
# round 12, blocker 4: the service must refuse a linked state ancestry
# ──────────────────────────────────────────────────────────────────────


def test_the_service_refuses_a_state_root_behind_a_symlinked_ancestor(tmp_path):
    """`resolve()` + `mkdir(parents=True)` wrote through the link the store rejects."""

    workspace = _workspace(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    bridge = tmp_path / "bridge"
    bridge.symlink_to(elsewhere)

    with pytest.raises((ContinuationCorrupt, OSError, ValueError)):
        _service(
            tmp_path, workspace, SegmentBackend(workspace, []),
            state_dir="bridge/state",
        )
    assert list(elsewhere.iterdir()) == [], "state was created through the link"


def test_the_service_still_creates_a_missing_private_state_root(tmp_path):
    workspace = _workspace(tmp_path)
    service = _service(
        tmp_path, workspace, SegmentBackend(workspace, []),
        state_dir="fresh/deep/state",
    )
    try:
        root = Path(service.state_root)
        assert root.is_dir()
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert stat.S_IMODE(root.parent.stat().st_mode) == 0o700
    finally:
        service.close(wait=True)


def test_the_service_refuses_a_state_root_that_is_itself_a_symlink(tmp_path):
    workspace = _workspace(tmp_path)
    elsewhere = tmp_path / "target"
    elsewhere.mkdir()
    link = tmp_path / "state-link"
    link.symlink_to(elsewhere)
    with pytest.raises((ContinuationCorrupt, OSError, ValueError)):
        _service(
            tmp_path, workspace, SegmentBackend(workspace, []),
            state_dir="state-link",
        )
    assert list(elsewhere.iterdir()) == []


# ──────────────────────────────────────────────────────────────────────
# round 12, blocker 1: no repository scan under the global state guard
# ──────────────────────────────────────────────────────────────────────


def _two_workspace_service(tmp_path, backend):
    from flyto_ai.agents.claude_code import ClaudeCodingAgent

    first = _workspace(tmp_path, "ws-a")
    second = _workspace(tmp_path, "ws-b")
    service = CodingService(
        lambda store: ClaudeCodingAgent(store, agent=backend),
        state_root=str(tmp_path / "phased-state"),
        workspace_roots=(str(first), str(second)),
        max_workers=2, max_queued=8, require_codex_audit=True,
        implementation_backend="claude",
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        approval_policy=ApprovalPolicy.NEVER,
    )
    return service, first, second


@contextlib.contextmanager
def _paused_observation(monkeypatch, workspace):
    """Hold one workspace's repository observation open, deterministically.

    The pause is inside `preflight_repository`, which is the first expensive
    thing phase one does, so the caller is parked exactly where a
    twenty-second `flyto-code` snapshot would park it. No sleeping: the test
    waits on events the paused thread sets.
    """

    entered = threading.Event()
    release = threading.Event()
    real = service_module.preflight_repository

    def paused(target, *args, **kwargs):
        if str(target) == str(workspace):
            entered.set()
            assert release.wait(60), "the paused observation was never released"
        return real(target, *args, **kwargs)

    monkeypatch.setattr(service_module, "preflight_repository", paused)
    try:
        yield entered, release
    finally:
        release.set()


def test_an_unrelated_workspace_is_not_blocked_by_a_slow_observation(
    tmp_path, monkeypatch,
):
    """The production complaint: one big repository stalled every other Codex.

    Preflight and the workspace snapshot used to run under the global state
    guard, so a submit for an unrelated tenant and an unrelated workspace could
    not even be admitted until the scan finished. The assertion is a state one:
    while A is provably still inside its observation, B has a durable job
    record and an idempotency record of its own.
    """

    backend = SegmentBackend(_workspace(tmp_path, "ws-a"), ["ok", "ok"])
    service, first, second = _two_workspace_service(tmp_path, backend)
    outcome = {}
    try:
        with _paused_observation(monkeypatch, first) as (entered, release):
            waiter = threading.Thread(
                target=lambda: outcome.setdefault(
                    "a", service.submit("t", "slow-a", _request(first)),
                ),
                daemon=True,
            )
            waiter.start()
            assert entered.wait(60), "the slow observation never started"

            # A is parked inside phase one and has committed nothing.
            assert not release.is_set()
            tenant_dir = service._tenant_dir(service._tenant_ref("t"))
            assert list((tenant_dir / "jobs").glob("*.json")) == []

            # B, an unrelated workspace, is admitted to completion anyway.
            receipt = service.submit("t", "fast-b", _request(second))
            assert receipt.job_id
            assert (tenant_dir / "jobs" / (receipt.job_id + ".json")).is_file()
            # ...and A is *still* parked, so B genuinely overtook it.
            assert not release.is_set()
            assert entered.is_set()

            release.set()
            waiter.join(60)
        assert not waiter.is_alive()
        assert outcome["a"].job_id != receipt.job_id
    finally:
        service.close(wait=True)


def test_an_idempotent_replay_is_exact_while_an_observation_is_in_flight(
    tmp_path, monkeypatch,
):
    """A replay must not queue behind somebody else's repository scan."""

    backend = SegmentBackend(_workspace(tmp_path, "ws-a"), ["ok", "ok", "ok"])
    service, first, second = _two_workspace_service(tmp_path, backend)
    try:
        established = service.submit("t", "established", _request(second))
        with _paused_observation(monkeypatch, first) as (entered, release):
            waiter = threading.Thread(
                target=lambda: service.submit("t", "slow-a", _request(first)),
                daemon=True,
            )
            waiter.start()
            assert entered.wait(60)

            replay = service.submit("t", "established", _request(second))
            assert replay.job_id == established.job_id
            assert not release.is_set(), "the replay waited for the scan"

            # A different request under the same key is still a conflict.
            with pytest.raises(IdempotencyConflict):
                service.submit("t", "established", _request(first))

            release.set()
            waiter.join(60)
    finally:
        service.close(wait=True)


def test_two_admissions_for_one_workspace_cannot_both_observe_and_claim(
    tmp_path, monkeypatch,
):
    """Same-workspace exclusivity survives the phase split.

    The admission lock is per workspace, so the second submit for the *same*
    tree does not begin its observation until the first has finished claiming.
    """

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    observations = []
    real_observe = CodingService._observe_continuation

    # Observation no longer takes an authorized digest: a continuation derives no
    # authority from the current contract file, so there is nothing to hand it.
    def counting(self, tenant_ref, request):
        observations.append(request.working_dir)
        return real_observe(self, tenant_ref, request)

    monkeypatch.setattr(CodingService, "_observe_continuation", counting)
    try:
        results = []
        errors = []

        def resume(key):
            try:
                results.append(service.submit(
                    "t", key, _request(workspace, thread_id=_SESSION, resume=True),
                ))
            except ContinuationRefused as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=resume, args=("race-{}".format(index),))
            for index in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
            assert not thread.is_alive()

        assert len(results) == 1, "one authority was claimed twice"
        assert len(errors) == 2
        assert {exc.code for exc in errors} == {CONTINUATION_UNAVAILABLE}
        # Serialized, not interleaved: every observation really ran.
        assert len(observations) == 3
        _wait(service, "t", results[0].job_id)
        assert len(backend.requests) == 2, "a refused admission reached a provider"
    finally:
        service.close(wait=True)


def test_admission_never_holds_the_state_guard_while_it_observes(
    tmp_path, monkeypatch,
):
    """Stated as an invariant, not inferred from a stopwatch.

    The guard is re-entrant per thread, so the observation phase can be asked
    directly whether this thread is inside it. Depth zero is the whole claim of
    the phase split, and it is also the reason the lock order
    (admission -> guard) has no cycle to deadlock on.
    """

    workspace = _workspace(tmp_path)
    backend = SegmentBackend(workspace, ["ok"])
    service = _service(tmp_path, workspace, backend)
    depths = {}
    real_preflight = service_module.preflight_repository
    real_read = service_module.read_project_contract
    real_observe = CodingService._observe_continuation

    seen = []
    reads = []

    def record_preflight(target, *args, **kwargs):
        seen.append(service._state_lock_depth)
        return real_preflight(target, *args, **kwargs)

    def record_read(workspace_arg, *args, **kwargs):
        reads.append(service._state_lock_depth)
        return real_read(workspace_arg, *args, **kwargs)

    def record_observe(self, tenant_ref, request):
        depths["observe"] = self._state_lock_depth
        return real_observe(self, tenant_ref, request)

    monkeypatch.setattr(service_module, "preflight_repository", record_preflight)
    monkeypatch.setattr(service_module, "read_project_contract", record_read)
    monkeypatch.setattr(CodingService, "_observe_continuation", record_observe)
    try:
        service.submit("t", "depth", _request(workspace))
        # The observation phase - preflight, the contract read that *pins* it by
        # value, and the workspace snapshot - runs with the global guard
        # released. Preflight is therefore paid exactly once, outside the guard.
        assert depths == {"observe": 0}
        assert seen == [0], seen
        # What still runs inside the guard is the bounded re-read of one small
        # file: pin at depth 0, drift re-check at depth 1. That is the whole
        # time-of-check/time-of-use repair, and it is deliberately not a walk
        # and deliberately not another preflight.
        assert reads == [0, 1], reads
    finally:
        service.close(wait=True)


def test_the_multiprocess_claim_still_grants_exactly_one_owner_after_the_split(
    tmp_path,
):
    """The phase split must not have loosened the durable compare-and-swap."""

    import multiprocessing

    context = multiprocessing.get_context("spawn")
    state_root = tmp_path / "race-state"
    outcomes_dir = tmp_path / "outcomes"
    outcomes_dir.mkdir()
    store = ContinuationStore(state_root)
    stamped = store.create(_authority())

    barrier = context.Barrier(_RACERS)
    workers = []
    for index in range(_RACERS):
        outcome_path = outcomes_dir / "{}.txt".format(index)
        workers.append((outcome_path, context.Process(
            target=_claim_in_a_separate_process,
            args=(
                str(state_root), stamped.tenant_ref, _SESSION,
                "job_" + "{:024x}".format(index), barrier, str(outcome_path),
            ),
        )))
    for _, worker in workers:
        worker.start()
    for _, worker in workers:
        worker.join(120)
        assert worker.exitcode == 0
    outcomes = [path.read_text(encoding="utf-8") for path, _ in workers]
    assert outcomes.count("claimed") == 1, outcomes
    assert outcomes.count("unavailable") == _RACERS - 1, outcomes


# ──────────────────────────────────────────────────────────────────────
# round 13: the contract may not move between observing and committing
# ──────────────────────────────────────────────────────────────────────


_CONTRACT_REVISION = itertools.count()


def _rewrite_contract(workspace, name=None):
    """A different, entirely valid contract for the same repository."""

    # A fresh name every time: two swaps that produce identical bytes would
    # not be a contract change at all, and the test would pass vacuously.
    name = name or "renamed{}".format(next(_CONTRACT_REVISION))
    (workspace / ".flyto" / "coding.yaml").write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: {}\n"
        "    argv: {}\n"
        "    required: true\n".format(name, json.dumps([sys.executable, "--version"])),
        encoding="utf-8",
    )


def _side_effects(service, tenant, workspace):
    """Everything a refused admission must not have left behind."""

    tenant_ref = service._tenant_ref(tenant)
    tenant_dir = service._tenant_dir(tenant_ref)
    claim = service._workspace_claim_path(str(workspace))
    # The bounded runtime status this instance publishes. These fixture
    # services are isolated - one instance, one state root, no other worker -
    # so an unchanged file here really does mean the refused request published
    # nothing, rather than meaning somebody else happened not to write.
    status_path = service._status.instance_path()
    return {
        "status": status_path.read_bytes() if status_path.is_file() else None,
        "jobs": sorted(p.name for p in (tenant_dir / "jobs").glob("*.json")),
        "idempotency": sorted(
            p.name for p in (tenant_dir / "idempotency").glob("*.json")
        ),
        "resume": sorted(
            p.name for p in (tenant_dir / "resume").glob("*.json")
        ) if (tenant_dir / "resume").is_dir() else [],
        "leases": sorted(service._job_leases),
        "claim": claim.is_file(),
    }


@contextlib.contextmanager
def _contract_swapped_between_phases(monkeypatch, workspace):
    """Replace the contract at exactly the phase-one/phase-two boundary.

    Deterministic by construction: the swap is performed by the instrumented
    guarded transition on its way *in*, so it always lands after phase one has
    finished observing - and, for a new job, after it has pinned the contract by
    value - and before the bounded re-read inside the state guard. No sleeping
    and no threads.

    The instrumented seam is `_commit_admission` rather than
    `_observe_continuation` on purpose. Phase one now both observes and pins, so
    a swap landing before the pin would simply be pinned instead of detected -
    a legitimate but entirely different scenario, and not the time-of-check race
    this covers.
    """

    real_commit = CodingService._commit_admission

    def swap_then_commit(self, *args, **kwargs):
        _rewrite_contract(workspace)
        return real_commit(self, *args, **kwargs)

    monkeypatch.setattr(CodingService, "_commit_admission", swap_then_commit)
    try:
        yield
    finally:
        # Restored on the way out, so a retry *after* the race is an ordinary
        # admission rather than another swap. Leaving the patch installed would
        # make "retryable" untestable: every retry would race again.
        monkeypatch.setattr(CodingService, "_commit_admission", real_commit)


def test_a_contract_swapped_during_admission_refuses_before_any_effect(
    tmp_path, monkeypatch,
):
    """The ordinary submit. Nothing durable may survive the refusal."""

    workspace = _workspace(tmp_path)
    backend = SegmentBackend(workspace, ["ok"])
    service = _service(tmp_path, workspace, backend)
    try:
        before = _side_effects(service, "t", workspace)
        with _contract_swapped_between_phases(monkeypatch, workspace):
            with pytest.raises(VerificationContractChanged) as excinfo:
                service.submit("t", "swapped", _request(workspace))
        assert excinfo.value.code == "verification_contract_changed"
        assert excinfo.value.failure_phase == FAILURE_PHASE_PREFLIGHT
        # Retryable: an ordering accident, not a repository somebody must fix.
        assert excinfo.value.retryable is True
        assert backend.requests == [], "a provider was contacted"
        assert _side_effects(service, "t", workspace) == before

        # The same request, admitted afresh against the contract that is now
        # actually there, is ordinary work.
        receipt = service.submit("t", "swapped-retry", _request(workspace))
        assert _wait(service, "t", receipt.job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
    finally:
        service.close(wait=True)


def test_a_contract_swapped_during_a_continuation_leaves_the_authority_open(
    tmp_path, monkeypatch,
):
    """The continuation submit. The authority must stay unconsumed."""

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        opened = _authority_of(service, "t")
        assert opened.state == STATE_OPEN
        before = _side_effects(service, "t", workspace)

        with _contract_swapped_between_phases(monkeypatch, workspace):
            with pytest.raises(VerificationContractChanged):
                service.submit(
                    "t", "swapped-resume",
                    _request(workspace, thread_id=_SESSION, resume=True),
                )
        assert len(backend.requests) == 1, "a provider was contacted"

        # Untouched: same generation, same sequence, still open.
        after = _authority_of(service, "t")
        assert after == opened
        assert after.state == STATE_OPEN
        assert [entry.sequence for entry in
                service._continuation.journal(service._tenant_ref("t"), _SESSION)] == [1]
        assert _side_effects(service, "t", workspace) == before

        # Retried against the now-current contract, the existing rule applies:
        # the swap rewrote a file inside the observed workspace, so this is drift
        # from the exact stopped revision rather than a silent re-licensing.
        with pytest.raises(ContinuationRefused) as refused:
            service.submit(
                "t", "resume-after-swap",
                _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert refused.value.code == CONTINUATION_REVISION_MISMATCH
        assert _authority_of(service, "t").state == STATE_OPEN
        assert len(backend.requests) == 1
    finally:
        service.close(wait=True)


def test_a_newly_unreadable_contract_keeps_its_own_preflight_refusal(
    tmp_path, monkeypatch,
):
    """"Changed" and "no longer honourable" are different answers."""

    workspace = _workspace(tmp_path)
    backend = SegmentBackend(workspace, ["ok"])
    service = _service(tmp_path, workspace, backend)
    real_observe = CodingService._observe_continuation

    def observe_then_break(self, tenant_ref, request):
        # For an ordinary new job the observation returns before the contract has
        # been read at all, so preflight meets a workspace that has genuinely
        # lost its contract and answers with its own refusal rather than with
        # "changed".
        observed = real_observe(self, tenant_ref, request)
        (workspace / ".flyto" / "coding.yaml").unlink()
        return observed

    monkeypatch.setattr(CodingService, "_observe_continuation", observe_then_break)
    try:
        with pytest.raises(VerificationRequired) as excinfo:
            service.submit("t", "broken", _request(workspace))
        assert not isinstance(excinfo.value, VerificationContractChanged)
        assert backend.requests == []
    finally:
        service.close(wait=True)


def test_an_unchanged_continuation_costs_exactly_two_snapshots_and_admits(
    tmp_path, monkeypatch,
):
    """A continuation re-proves the whole tree outside the guard, never inside.

    A continuation pays for exactly two whole-tree walks and no more: one to
    observe the authority, and one to re-prove it at the phase boundary, before
    the authority is consumed. Both run outside the global state guard, which is
    exactly what the recorded depths of zero assert. A third walk would mean a
    repository scan had reached the guarded transition, which is the thing that
    must never happen. The only re-check that does run inside the guard is
    `_observed_config_digest`, a bounded read of one small contract file, which
    is not a snapshot at all.
    """

    snapshots = []
    depths = []
    holder = []
    real_digest = service_module.workspace_manifest_digest

    def counting(working_dir, policy=None):
        snapshots.append(str(working_dir))
        # The guard is re-entrant per thread, so it can simply be asked whether
        # this walk is happening inside it. Zero is the whole claim.
        if holder:
            depths.append(holder[0]._state_lock_depth)
        return real_digest(working_dir, policy)

    monkeypatch.setattr(service_module, "workspace_manifest_digest", counting)

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    holder.append(service)
    try:
        snapshots.clear()
        depths.clear()
        resumed = service.submit(
            "t", "unchanged", _request(workspace, thread_id=_SESSION, resume=True),
        )
        assert snapshots == [str(workspace), str(workspace)], (
            "a continuation must observe once and re-prove once"
        )
        assert depths == [0, 0], "a tree was walked under the global state guard"
        assert _wait(service, "t", resumed.job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
    finally:
        service.close(wait=True)


def test_the_contract_recheck_does_not_serialize_unrelated_workspaces(
    tmp_path, monkeypatch,
):
    """The re-read is inside the guard, so it has to stay bounded."""

    backend = SegmentBackend(_workspace(tmp_path, "ws-a"), ["ok", "ok"])
    service, first, second = _two_workspace_service(tmp_path, backend)
    try:
        with _paused_observation(monkeypatch, first) as (entered, release):
            waiter = threading.Thread(
                target=lambda: service.submit("t", "slow-a", _request(first)),
                daemon=True,
            )
            waiter.start()
            assert entered.wait(60)
            receipt = service.submit("t", "fast-b", _request(second))
            assert receipt.job_id
            assert not release.is_set()
            release.set()
            waiter.join(60)
        assert not waiter.is_alive()
    finally:
        service.close(wait=True)


def test_a_contract_race_is_reported_as_retryable_with_nothing_to_fix(tmp_path, monkeypatch):
    """A scheduler must not treat an ordering accident as a dead end.

    Every other `VerificationRequired` means somebody has to change a file
    before the request can ever work. This one means two things happened in the
    wrong order, so it carries no required action and says so.
    """

    from flyto_ai.coding.mcp_server import CodingMCPServer
    from flyto_ai.coding.service import error_details

    workspace = _workspace(tmp_path)
    backend = SegmentBackend(workspace, ["ok"])
    service = _service(tmp_path, workspace, backend)
    try:
        with _contract_swapped_between_phases(monkeypatch, workspace):
            with pytest.raises(VerificationContractChanged) as excinfo:
                service.submit("t", "retryable", _request(workspace))
        error = excinfo.value
        assert error.retryable is True
        assert error.required_actions == ()
        assert error.failure_phase == FAILURE_PHASE_PREFLIGHT
        # ...and it is the *only* verification refusal that is retryable, so the
        # distinction cannot be lost by widening the base class.
        assert VerificationRequired.retryable is False
        details = error_details(error)
        assert details["retryable"] is True
        assert "required_actions" not in details

        # The same typed facts cross the public MCP facade unchanged.
        server = CodingMCPServer(service, tenant_id="t")
        with _contract_swapped_between_phases(monkeypatch, workspace):
            response = server.handle({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {
                    "name": "flyto_coding_submit",
                    "arguments": {
                        "idempotency_key": "retryable-mcp",
                        "request": {
                            "message": "implement the feature",
                            "working_dir": str(workspace),
                        },
                    },
                },
            })
        payload = response["result"]["structuredContent"]
        assert payload["ok"] is False
        assert payload["error"] == "verification_contract_changed"
        assert payload["details"]["retryable"] is True
        assert payload["details"]["failure_phase"] == FAILURE_PHASE_PREFLIGHT
        assert "required_actions" not in payload["details"]
        assert backend.requests == []

        # Retrying against the contract that is now actually there succeeds.
        receipt = service.submit("t", "retryable-retry", _request(workspace))
        assert _wait(service, "t", receipt.job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
    finally:
        service.close(wait=True)


def test_a_continuation_contract_race_is_retryable_then_meets_the_authority_rule(
    tmp_path, monkeypatch,
):
    """Two steps, two different answers, and both are the honest one.

    The race itself is retryable: nothing is wrong yet. The *next* admission is
    not, because by then the swap has really rewritten a file inside the exact
    tree the authority promised, and continuing into a workspace that moved is
    the thing this refuses.
    """

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        with _contract_swapped_between_phases(monkeypatch, workspace):
            with pytest.raises(VerificationContractChanged) as race:
                service.submit(
                    "t", "resume-race",
                    _request(workspace, thread_id=_SESSION, resume=True),
                )
        assert race.value.retryable is True
        assert race.value.required_actions == ()
        assert _authority_of(service, "t").state == STATE_OPEN

        with pytest.raises(ContinuationRefused) as settled:
            service.submit(
                "t", "resume-after-race",
                _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert settled.value.code == CONTINUATION_REVISION_MISMATCH
        assert settled.value.retryable is False
        assert _authority_of(service, "t").state == STATE_OPEN
        assert len(backend.requests) == 1
    finally:
        service.close(wait=True)


@contextlib.contextmanager
def _source_rewritten_between_phases(monkeypatch, path, replacement):
    """Rewrite one ordinary source file at exactly the phase-one/two boundary.

    Deterministic by construction, and for the same reason the contract swap
    above is: the write is performed by the instrumented guarded transition on
    its way *in*, so it always lands after `_observe_continuation` has proven
    the whole tree against the authority and before anything is consumed. No
    sleeping and no threads.

    The file is deliberately not `.flyto/coding.yaml`. The guarded re-check only
    ever re-read the contract, so an ordinary tracked source file rewritten in
    this window satisfied every gate that existed and still reached the claim,
    the worktree and the provider.
    """

    real_commit = CodingService._commit_admission

    def rewrite_then_commit(self, *args, **kwargs):
        path.write_bytes(replacement)
        return real_commit(self, *args, **kwargs)

    monkeypatch.setattr(CodingService, "_commit_admission", rewrite_then_commit)
    try:
        yield
    finally:
        # Restored on the way out, so a later admission is an ordinary one
        # rather than another race. Otherwise "the authority survived" would be
        # untestable: every retry would rewrite the tree again.
        monkeypatch.setattr(CodingService, "_commit_admission", real_commit)


@pytest.mark.parametrize(
    "relative",
    ["README.md", "feature.py"],
    ids=["unattributed-source", "attributed-source"],
)
def test_source_rewritten_at_the_commit_seam_refuses_before_anything_is_spent(
    tmp_path, monkeypatch, relative,
):
    """The window between observing a tree and consuming its authority.

    Phase one proved the whole workspace; the gate inside the state guard then
    re-proved only the contract file. A non-contract file changed at exactly
    that boundary was therefore invisible - and a continuation admitted on a
    stale observation resumes a real provider session against a tree the model
    never saw, which is the one thing this mechanism exists to prevent.

    Both parametrisations matter and fail differently without the repair. The
    unattributed file is invisible to the attributable-revision digest and is
    caught only by the whole-tree manifest; the attributed one is the file the
    stopped round was actually credited with.
    """

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        target = workspace / relative
        original = target.read_bytes()
        opened = _authority_of(service, "t")
        assert opened.state == STATE_OPEN
        before = _side_effects(service, "t", workspace)
        contract_before = (workspace / ".flyto" / "coding.yaml").read_bytes()

        with _source_rewritten_between_phases(
            monkeypatch, target, b"# rewritten under the admission lock\n",
        ):
            with pytest.raises(ContinuationRefused) as excinfo:
                service.submit(
                    "t", "seam-drift",
                    _request(workspace, thread_id=_SESSION, resume=True),
                )

        # The existing, precise answer: the tree moved. Not a contract race, and
        # not a vague "unavailable" that hides which invariant was violated.
        assert excinfo.value.code == CONTINUATION_REVISION_MISMATCH
        assert not isinstance(excinfo.value, VerificationContractChanged)
        assert excinfo.value.failure_phase == "preflight"
        assert excinfo.value.retryable is False
        # The contract really was untouched, so nothing but the source drift
        # could have produced that refusal.
        assert (workspace / ".flyto" / "coding.yaml").read_bytes() == contract_before

        # No provider was contacted: the only request on record is the stopped
        # round's own.
        assert len(backend.requests) == 1, "a provider was contacted"
        assert backend.resumed == [None], backend.resumed

        # The authority was not consumed. Same generation, same sequence, still
        # open, and the journal never grew a second entry.
        after = _authority_of(service, "t")
        assert after == opened
        assert after.state == STATE_OPEN
        assert [entry.sequence for entry in
                service._continuation.journal(service._tenant_ref("t"), _SESSION)] == [1]

        # No job, no lease, no worktree claim, no idempotency record, no resume
        # envelope, no published status.
        assert _side_effects(service, "t", workspace) == before

        # And because nothing was spent, restoring the exact bytes restores the
        # offer: the same authority still admits, and reaches audit.
        target.write_bytes(original)
        resumed = service.submit(
            "t", "seam-drift-retry",
            _request(workspace, thread_id=_SESSION, resume=True),
        )
        assert _wait(service, "t", resumed.job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
        assert backend.resumed == [None, _SESSION]
        # Spent exactly once, and only by the admission that actually happened.
        assert _authority_of(service, "t").state == STATE_SETTLED
    finally:
        service.close(wait=True)


# ──────────────────────────────────────────────────────────────────────
# round 14: the pin *is* the verifier, so it is bound, private, and never
# re-derived from the tree the model just edited
#
# Everything below is an end-to-end assertion against the real service, the
# real durable records and the real authority journal. Nothing stubs the pin
# itself: a test that constructed its own snapshot would prove the dataclass
# works and prove nothing about what the host authorized.
# ──────────────────────────────────────────────────────────────────────


def _job_record_path(service, tenant, job_id):
    return (
        service._tenant_dir(service._tenant_ref(tenant)) / "jobs" / (job_id + ".json")
    )


def _job_record(service, tenant, job_id):
    return service._read_json(_job_record_path(service, tenant, job_id))


def test_the_pin_is_captured_before_the_first_provider_call_and_kept_private(
    tmp_path, monkeypatch,
):
    """Timing is the guarantee: authorized before any provider edit could exist.

    Asserted as an order of observed events rather than inferred from where the
    call happens to sit in the source, because "before the provider" is the only
    reason a snapshot can be trusted as the pre-edit contract at all.
    """

    workspace = _workspace(tmp_path)
    backend = SegmentBackend(workspace, ["ok"])
    service = _service(tmp_path, workspace, backend)
    order = []
    real_pin = CodingService._pin_verified_contract
    real_run = SegmentBackend.run

    def recording_pin(self, target):
        pinned = real_pin(self, target)
        order.append(("pin", pinned.identity()))
        return pinned

    async def recording_run(self, request):
        order.append(("provider", ""))
        return await real_run(self, request)

    monkeypatch.setattr(CodingService, "_pin_verified_contract", recording_pin)
    monkeypatch.setattr(SegmentBackend, "run", recording_run)
    try:
        queued = service.submit("t", "pinned", _request(workspace))
        ready = _wait(service, "t", queued.job_id)
        assert ready.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert [event for event, _ in order] == ["pin", "provider"]

        record = _job_record(service, "t", queued.job_id)
        pinned = ContractSnapshot.from_mapping(record["contract_snapshot"])
        # By value, not by digest. The declared checks themselves are in private
        # state, which is what makes a later round executable without a re-read.
        assert [check.name for check in pinned.checks] == ["declared"]
        assert pinned.has_required_check()
        # Content-addressed, and the record stores the address it must reproduce.
        assert pinned.identity() == record["contract_snapshot_sha256"]
        assert pinned.identity() == order[0][1]
        # One document, named two ways: the snapshot's source digest and the
        # job's contract authority may never disagree.
        assert pinned.config_sha256 == record["authorized_config_sha256"]
        assert pinned.config_sha256 == read_project_contract(str(workspace)).digest

        # Private tenant state, not an artifact: 0600, no group, no world.
        mode = stat.S_IMODE(
            _job_record_path(service, "t", queued.job_id).stat().st_mode,
        )
        assert mode == 0o600
    finally:
        service.close(wait=True)


def test_a_restart_restores_the_pin_from_private_state_not_from_the_file(tmp_path):
    """A fresh process reapplies what was recorded, whatever the tree now says."""

    workspace = _workspace(tmp_path)
    backend = SegmentBackend(workspace, ["ok"])
    service = _service(tmp_path, workspace, backend)
    try:
        queued = service.submit("t", "restart-pin", _request(workspace))
        assert _wait(service, "t", queued.job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
        bound = _job_record(service, "t", queued.job_id)["contract_snapshot_sha256"]
    finally:
        service.close(wait=True)

    # The contract on disk is now something no reader would accept. A round that
    # re-read it would die with `invalid_config`; a round that reapplies the pin
    # cannot even observe it.
    (workspace / ".flyto" / "coding.yaml").write_text(":\n  not: [a\n", encoding="utf-8")

    restarted = _service(tmp_path, workspace, backend)
    try:
        record = _job_record(restarted, "t", queued.job_id)
        restored = restarted._record_pinned_contract(record)
        assert restored is not None, "a restart could not restore its own pin"
        assert restored.identity() == bound
        assert [check.name for check in restored.checks] == ["declared"]
        assert restored.has_required_check()
        # ...and the current file is provably unreadable, so nothing about the
        # restored contract could have come from it.
        assert restarted._observed_config_digest(str(workspace)) == ""
        assert restored.config_sha256 != ""
    finally:
        restarted.close(wait=True)


def test_the_pin_is_only_restored_after_every_authority_check_has_passed(
    tmp_path, monkeypatch,
):
    """Ordering as a state assertion: no refusal above may consult the pin.

    Tenant, session shape, backend, workspace, snapshot policy, the attributable
    revision and the full manifest are all answered first. Only a caller that
    has passed every one of them reaches the pin - and then the compare-and-swap
    still has to succeed before the round is real.
    """

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    restores = []
    real_restore = CodingService._restore_pinned_contract

    def recording(self, tenant_ref, authority):
        restores.append(authority.session_id)
        return real_restore(self, tenant_ref, authority)

    monkeypatch.setattr(CodingService, "_restore_pinned_contract", recording)
    try:
        # session shape: refused with no lookup at all.
        with pytest.raises(ContinuationRefused) as invalid:
            service.submit(
                "t", "order-1",
                _request(workspace, thread_id="host-4c1d9f", resume=True),
            )
        assert invalid.value.code == CONTINUATION_SESSION_INVALID

        # tenant: another partition's guess is indistinguishable from absent.
        with pytest.raises(ContinuationRefused) as foreign:
            service.submit(
                "u", "order-2", _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert foreign.value.code == CONTINUATION_UNAVAILABLE

        # backend: now refused a step earlier still. A different implementer is
        # a different startup authority, so it never constructs against this
        # root at all - and therefore never reaches the pin either.
        with pytest.raises(CodingAuthorityConflict):
            _service(tmp_path, workspace, backend, implementation_backend="codex")

        # workspace.
        other = _workspace(tmp_path, "order-workspace")
        service.workspace_roots = tuple(service.workspace_roots) + (
            Path(other).resolve(),
        )
        with pytest.raises(ContinuationRefused) as wrong_workspace:
            service.submit(
                "t", "order-4", _request(other, thread_id=_SESSION, resume=True),
            )
        assert wrong_workspace.value.code == CONTINUATION_WORKSPACE_MISMATCH

        # the full manifest: a path nobody attributed to the stopped round.
        _drift_add_unrelated(workspace)
        with pytest.raises(ContinuationRefused) as drifted:
            service.submit(
                "t", "order-5", _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert drifted.value.code == CONTINUATION_REVISION_MISMATCH

        # Not one of those refusals learned anything from the pin.
        assert restores == []
        assert len(backend.requests) == 1
        assert _authority_of(service, "t").state == STATE_OPEN

        (workspace / "intruder.py").unlink()
        resumed = service.submit(
            "t", "order-6", _request(workspace, thread_id=_SESSION, resume=True),
        )
        assert _wait(service, "t", resumed.job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
        # Exactly once, for this session, and only after all of the above passed.
        assert restores == [_SESSION]
        assert backend.resumed == [None, _SESSION]
    finally:
        service.close(wait=True)


def test_a_tampered_pin_is_refused_before_any_provider_call(tmp_path):
    """Stored, well-formed, and simply not the contract this authority binds.

    The weakening is the exact escalation the mechanism exists to stop: keep
    every check, drop the requirement. The authority carries only the snapshot's
    content address, so the edited record can no longer reproduce it.
    """

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        path = _job_record_path(service, "t", stopped.job_id)
        record = service._read_json(path)
        bound = record["contract_snapshot_sha256"]
        weakened = dict(record["contract_snapshot"])
        weakened["checks"] = [
            dict(check, required=False) for check in weakened["checks"]
        ]
        assert weakened != record["contract_snapshot"], "the fixture changed nothing"
        # Still a perfectly valid snapshot - that is the whole difficulty.
        assert ContractSnapshot.from_mapping(weakened).identity() != bound
        service._update_record(path, contract_snapshot=weakened)
        # The bound address is deliberately left alone: an attacker who could
        # rewrite it too would be rewriting the journal, which is separately
        # digest-chained.
        assert service._read_json(path)["contract_snapshot_sha256"] == bound
        assert _authority_of(service, "t").contract_snapshot_sha256 == bound

        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "tampered", _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_CONTRACT_CHANGED
        assert excinfo.value.failure_phase == "preflight"
        assert excinfo.value.retryable is False
        assert len(backend.requests) == 1, "a provider was contacted"
        assert _authority_of(service, "t").state == STATE_OPEN
    finally:
        service.close(wait=True)


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {"config_sha256": "not-a-digest", "checks": []},
        {"config_sha256": "a" * 64, "checks": [{"name": "declared"}]},
        {"config_sha256": "a" * 64, "surprise": 1},
    ],
    ids=["absent", "unaddressable", "ungrammatical-check", "unknown-key"],
)
def test_an_unrecoverable_pin_is_terminal_and_says_so(tmp_path, snapshot):
    """Nothing is repaired, defaulted, or inferred into existence.

    A snapshot that cannot go back through the contract grammar is not weakened
    into a usable one and not silently re-read from disk. The refusal is the
    distinct terminal code, because no retry can help and the only honest route
    forward is a fresh job that re-reads and re-pins from scratch.
    """

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        service._update_record(
            _job_record_path(service, "t", stopped.job_id),
            contract_snapshot=snapshot,
        )
        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "unpinned", _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_CONTRACT_UNPINNED
        assert excinfo.value.code in CONTINUATION_CODES
        assert excinfo.value.failure_phase == "preflight"
        assert excinfo.value.retryable is False
        assert len(backend.requests) == 1, "a provider was contacted"
        # Left open rather than consumed: the operator loses nothing they could
        # still have used, and a restored record restores the offer.
        assert _authority_of(service, "t").state == STATE_OPEN
    finally:
        service.close(wait=True)


def test_a_missing_origin_record_leaves_nothing_to_restore(tmp_path):
    """The legacy shape, reached the only way a live host can reach it.

    A session whose origin job record is gone records no verifier anywhere, so
    it is exactly as unpinnable as a pre-pinning authority - and gets the same
    terminal code rather than a guess.
    """

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        _job_record_path(service, "t", stopped.job_id).unlink()
        with pytest.raises(ContinuationRefused) as excinfo:
            service.submit(
                "t", "no-origin", _request(workspace, thread_id=_SESSION, resume=True),
            )
        assert excinfo.value.code == CONTINUATION_CONTRACT_UNPINNED
        assert len(backend.requests) == 1
    finally:
        service.close(wait=True)


def test_no_pin_ever_crosses_the_public_receipt_or_the_three_tool_surface(tmp_path):
    """The verifier is host state. A caller may not read it or supply one."""

    from flyto_ai.coding.mcp_server import CodingMCPServer

    service, backend, workspace, stopped = _stopped_job(tmp_path)
    try:
        record = _job_record(service, "t", stopped.job_id)
        identity = record["contract_snapshot_sha256"]
        config_digest = record["authorized_config_sha256"]
        rendered = json.dumps(receipt_to_mapping(service.get("t", stopped.job_id)))
        for forbidden in (
            "contract_snapshot", "pinned_contract", "authorized_config_sha256",
            identity, config_digest,
        ):
            assert forbidden not in rendered, forbidden

        # The bounded runtime status document this instance publishes.
        status_path = service._status.instance_path()
        if status_path.is_file():
            published = status_path.read_text(encoding="utf-8")
            for forbidden in ("contract_snapshot", identity, config_digest):
                assert forbidden not in published, forbidden

        server = CodingMCPServer(service, tenant_id="t")
        listed = server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })
        assert [tool["name"] for tool in listed["result"]["tools"]] == [
            "flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit",
        ]
        request_schema = (
            listed["result"]["tools"][0]["inputSchema"]["properties"]["request"]
        )
        assert "pinned_contract" not in request_schema["properties"]
        assert "authorized_config_sha256" not in request_schema["properties"]

        # ...and a caller that sends one anyway is refused by the decoder, not
        # quietly ignored: choosing your own verifier must never be silent.
        for field, value in (
            ("pinned_contract", record["contract_snapshot"]),
            ("authorized_config_sha256", config_digest),
        ):
            refused = server.handle({
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "flyto_coding_submit",
                    "arguments": {
                        "idempotency_key": "byo-verifier",
                        "request": {
                            "message": "implement the feature",
                            "working_dir": str(workspace),
                            field: value,
                        },
                    },
                },
            })
            payload = refused["result"]["structuredContent"]
            assert payload["ok"] is False, field
        assert len(backend.requests) == 1
    finally:
        service.close(wait=True)
