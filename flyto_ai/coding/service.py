# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Tenant-scoped durable job service for the native Flyto2 coding agent."""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Protocol, Sequence, Tuple

from flyto_ai.coding.checks import read_project_contract
from flyto_ai.coding.contracts import (
    ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,
    ContractSnapshot,
    FAILURE_PHASE_VERIFICATION,
    JOB_FAILURE_ACTIONS,
    MAX_AUDIT_ROUNDS,
    TERMINAL_CODING_JOB_STATES,
    MAX_IMPLEMENTATION_BLOCKERS,
    MISSION_COMPLETED,
    MISSION_DISPOSITION_FIXED,
    MISSION_OPEN,
    MISSION_PROJECTION_FIELDS,
    MISSION_STATUS_CLOSED,
    MISSION_STATUS_DISPATCHED,
    MISSION_STATUS_READY,
    ApprovalPolicy,
    CapabilityStatus,
    CheckResult,
    CodingAuditFinding,
    CodingAuditVerdict,
    CodingJobReceipt,
    CodingJobState,
    CodingMissionEnvelope,
    CodingMissionProjection,
    CodingTaskRequest,
    CodingTaskResult,
    SandboxMode,
    safe_blockers,
    audit_findings_sha256,
    require_revision_sha256,
    validate_audit_submission,
)
from flyto_ai.coding.emergency import (
    EmergencyAuthorityError,
    EmergencyAuthorityReceipt,
    EmergencyCircuitBreaker,
    EmergencyOverflowPolicy,
    EmergencyTrigger,
    classify_overflow_trigger,
)
from flyto_ai.coding.continuation import (
    CONTINUABLE_STOP_CODES,
    CONTINUATION_BACKEND_MISMATCH,
    CONTINUATION_CODES,
    CONTINUATION_CONTRACT_CHANGED,
    CONTINUATION_CONTRACT_UNPINNED,
    CONTINUATION_POLICY_CHANGED,
    CONTINUATION_REVISION_MISMATCH,
    CONTINUATION_SESSION_INVALID,
    CONTINUATION_UNAVAILABLE,
    CONTINUATION_WORKSPACE_MISMATCH,
    MAX_CONTINUATION_GENERATION,
    PROVISIONAL_SESSION_PREFIXES,
    STATE_CLAIMED,
    STATE_SETTLED,
    ContinuationAuthority,
    ContinuationConflict,
    ContinuationCorrupt,
    ContinuationStore,
    DEFAULT_SNAPSHOT_POLICY,
    SnapshotPolicy,
    WorkspaceUnobservable,
    is_continuable_session,
    secure_directory,
    workspace_manifest_digest,
)
from flyto_ai.coding.mission_runtime import (
    CRITERION_AUDIT,
    CRITERION_CHECKS,
    CRITERION_REVISION,
    DISPOSITION_BLOCKED,
    DISPOSITION_DEFERRED,
    CodingMissionRuntime,
    DispatchedWork,
    MissionAdmission,
    MissionAuthorityRefused,
    MissionConflictRefused,
    MissionHeartbeat,
    MissionRouteError,
    worker_identity,
)
from flyto_ai.coding.route import (
    CodingRoutePolicy,
    CodingRouteReceipt,
    route_failure_point,
    route_thread_id,
)
from flyto_ai.coding.route_status import (
    STATUS_FAILURE_CODES,
    CodingRouteStatus,
    RouteStatusPublisher,
    current_service_build_id,
    route_mode,
    route_progress,
    service_build_id,
    service_version,
)
from flyto_ai.coding.preflight import (
    FAILURE_PHASE_PREFLIGHT,
    PREFLIGHT_ACTIONS,
    CODE_CAPABILITY_UNAVAILABLE,
    CODE_VERIFICATION_TOOL_MISSING,
    CODE_VERIFICATION_CONTRACT_INVALID,
    preflight_repository,
)
from flyto_ai.coding.store import ThreadStore, redact_evidence


try:  # pragma: no cover - Windows fallback is exercised by static review.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^job_[a-f0-9]{24}$")
_TENANT_REF = re.compile(r"^[a-f0-9]{64}$")
#: Structured error context is a closed vocabulary of short opaque tokens.
#: Paths, prose, and anything unbounded fail these patterns and are dropped.
_DETAIL_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
_DETAIL_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MAX_ERROR_DETAIL_FIELDS = 8
#: At most this many tokens survive from any one allowlisted list field.
_MAX_ERROR_DETAIL_TOKENS = 8
#: The only string tokens a list-valued detail may contain. Keeping this a
#: closed set is what stops `required_actions` from becoming a prose channel.
_PROJECTABLE_TOKENS = frozenset(PREFLIGHT_ACTIONS) | frozenset(JOB_FAILURE_ACTIONS)
#: Detail keys whose lists carry contract identifiers rather than closed
#: allowlist tokens. Named here, and nowhere else, so adding one is a decision
#: rather than a side effect of raising an error with a new field.
_IDENTIFIER_DETAIL_KEYS = frozenset({"verification_blockers"})
_BACKEND_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ALLOWED_REQUEST_FIELDS = frozenset({
    "message", "working_dir", "thread_id", "resume", "max_attempts",
    "max_rounds", "require_changes",
})
#: What a public payload may *decode*. Deliberately wider than
#: `_ALLOWED_REQUEST_FIELDS`, which is also the resume-envelope field list: the
#: mission envelope is a contract this slice accepts and projects, not durable
#: session authority to be replayed into a later round.
_DECODABLE_REQUEST_FIELDS = _ALLOWED_REQUEST_FIELDS | {"mission"}
# The revision binds only the attributable change set. Version control
# internals, credential files, evidence, and unrelated workspace content are
# never read by this digest.
_PROTECTED_REVISION_PARTS = frozenset({".git", ".hg", ".svn", ".ssh"})
_REVISION_DOMAIN = b"flyto.coding-revision.v1\n"
_REVISION_CHUNK_BYTES = 1024 * 1024
MAX_ATTRIBUTABLE_FILES = 512
MAX_REVISION_FILE_BYTES = 8 * 1024 * 1024
MAX_REVISION_TOTAL_BYTES = 128 * 1024 * 1024
MAX_REWORK_FEEDBACK_CHARS = 60_000
MAX_REWORK_MESSAGE_CHARS = 180_000
#: Thread ids a host mints when no implementation session exists yet. `route-`
#: comes from `flyto_ai.coding.route`, `host-` from the Claude adapter's
#: `HOST_THREAD_PREFIX`. Neither is a resumable implementation session, so
#: neither may ever be recorded as one. A test binds this to both sources.
PROVISIONAL_THREAD_PREFIXES = PROVISIONAL_SESSION_PREFIXES
#: The post-work lane is the only place a host lane refuses a round *because of
#: the implementation itself* rather than because a lane could not be trusted.
#: These two codes mean exactly "the implementer's own round did not close its
#: source-controlled checks" — real, attributable work that needs another
#: round. Every other lane failure is an infrastructure or safety refusal and
#: stays terminal. This is deliberately a closed two-item set: widening it
#: would let a lane that could not prove its own evidence look reworkable.
IMPLEMENTATION_BLOCKER_LANE = "indexer_post"
IMPLEMENTATION_BLOCKER_ROUTE_CODES = frozenset({
    "implementation_not_successful", "required_checks_failed",
})
#: The only implementation outcomes a host will hold a job open for, and a
#: deliberately closed set. Both are statements about *this round's own work*:
#: a recognized resumable provider stop that consumed its round budget, and a
#: required host verification that really ran and really failed. Everything
#: else — an unrecognized provider error, a session that would not bind, a
#: workspace boundary violation, an unreadable config, any safety or
#: infrastructure refusal — stays terminal, even when a buggy backend also
#: reports changed files. Widening this set is how an unknown failure would
#: quietly become resumable, so it is enumerated rather than inferred.
AUDITABLE_IMPLEMENTATION_FAILURE_CODES = frozenset({
    "turn_limit_exceeded", "verification_failed",
})
#: Stable classification for a failed round the host cannot name more precisely.
#: A blocker list is never empty, so a reader can always tell "blocked" from
#: "clean", even when nothing more specific survived the bounds.
GENERIC_IMPLEMENTATION_BLOCKER = "implementation_incomplete"
#: Stable settlement code for a job that used every configured repair round.
#: Distinct from any implementer or provider failure: nothing went wrong in the
#: round, the job simply ran out of the budget the host gave it.
REWORK_LIMIT_FAILURE_CODE = "rework_limit_reached"
#: Blocker codes share the audit finding vocabulary: short, stable, opaque.
_BLOCKER_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{1,63}$")
#: How one round was executed. Persisted before the implementer is invoked, so
#: an in-flight or crashed overflow round is never read back as strict.
EXECUTION_MODE_STRICT = "strict"
EXECUTION_MODE_EMERGENCY = "emergency"
#: Closed schema tokens for the two durable records this module owns beside the
#: job record. A file that does not name its exact version is not read.
WORKSPACE_CLAIM_VERSION = "flyto.coding-workspace-claim.v1"
RESUME_ENVELOPE_VERSION = "flyto.coding-resume-envelope.v1"
#: Closed schema token for the private binding between one mission work item and
#: the exact round it stands for. Private tenant state, never projected.
ROUND_ENVELOPE_VERSION = "flyto.coding-round-envelope.v1"
_ROUND_ENVELOPE_FIELDS = frozenset({
    "envelope_version", "work_item_id", "job_id", "rework", "message", "created_at",
})
#: The kernel mints work item ids as a prefix and exactly twelve decimal digits.
#: Matched here before one is ever used to build a path.
_WORK_ITEM_ID = re.compile(r"^w-[0-9]{12}$")
#: Closed schema token for the private record of which Indexer contract this
#: job's root task is authorized against. It lives in the job record, which is
#: 0600 and never projected into a receipt, a prompt, a log or an audit body.
PLAN_AUTHORITY_VERSION = "flyto.coding-plan-authority.v1"
_PLAN_AUTHORITY_DOMAIN = b"flyto.coding-plan-authority.v1\n"
#: A plan is a bounded contract, not a payload. Anything larger than this is not
#: something this host will carry forward across rounds.
MAX_PLAN_AUTHORITY_BYTES = 256 * 1024
#: Exactly the keys a workspace claim may contain. The raw worktree path is
#: deliberately absent: the claim is keyed by its digest, and the owning job
#: record already holds the path under the same 0600 state root.
_WORKSPACE_CLAIM_FIELDS = frozenset({
    "claim_version", "job_id", "tenant_ref", "workspace_sha256", "state",
    "instance_id", "process_id", "claimed_at", "updated_at",
})
#: Exactly the keys a resume envelope may contain: the bounded public request
#: fields plus the bindings that make it usable for one job's rework and for
#: nothing else. Startup authority fields are never among them.
_RESUME_ENVELOPE_FIELDS = frozenset(_ALLOWED_REQUEST_FIELDS) | {
    "envelope_version", "job_id", "request_sha256", "session_bound", "created_at",
}


class CodingServiceError(RuntimeError):
    """Base error with a stable, non-sensitive service code.

    Three typed attributes travel with every error, not just the ones a
    subclass bothered to annotate, because a caller deciding what to do next
    needs them for *all* failures:

    `failure_phase`
        How far the job got before this refusal. `preflight` is the strongest
        statement available - it means no session was opened, no worktree claim
        was taken, and no job exists to poll.
    `retryable`
        Whether repeating the identical request could ever succeed without
        somebody changing something first. Capacity is retryable; a missing
        verification contract is not.
    `required_actions`
        Bounded tokens from a closed allowlist naming the work that would clear
        this. Empty when the caller has nothing to do.

    They are folded into `details` so they cross the MCP and HTTP facades
    through the one existing bounded projection, rather than needing each
    transport to learn about them separately.
    """

    code = "service_error"
    failure_phase = "service"
    retryable = False
    required_actions: Tuple[str, ...] = ()

    @property
    def context(self) -> Dict[str, Any]:
        """Subclass-specific bounded context; empty by default.

        Paths, prose, and credentials never appear here.
        """
        return {}

    @property
    def details(self) -> Dict[str, Any]:
        """The typed envelope plus whatever bounded context a subclass added."""

        payload: Dict[str, Any] = {
            "failure_phase": self.failure_phase,
            "retryable": bool(self.retryable),
        }
        if self.required_actions:
            payload["required_actions"] = tuple(self.required_actions)
        payload.update(self.context)
        return payload


class CodingServiceBusy(CodingServiceError):
    """The service cannot take more work right now; the request itself is fine."""

    code = "service_busy"
    failure_phase = "capacity"
    retryable = True


class CodingCapacityUnavailable(CodingServiceBusy):
    """The bounded job queue is saturated.

    Distinct from the base `service_busy` so a caller can tell "this instance is
    at its configured concurrency ceiling, back off and retry" from any other
    transient refusal. It stays a `CodingServiceBusy` subclass so existing
    handlers and the 429 mapping keep working unchanged.
    """

    code = "capacity_unavailable"


class VerificationRequired(CodingServiceError):
    """This repository has not declared how a change to it must be verified.

    Raised from preflight, so nothing was created and nothing needs cleaning
    up. Never retryable: an identical resubmission fails identically until the
    repository adds or completes its contract.
    """

    code = "verification_required"
    failure_phase = FAILURE_PHASE_PREFLIGHT
    retryable = False

    def __init__(self, message: str, required_actions: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.required_actions = tuple(required_actions)


class VerificationContractInvalid(VerificationRequired):
    """A verification contract exists but cannot be honoured as written."""

    code = "verification_contract_invalid"


class VerificationContractChanged(VerificationRequired):
    """The contract moved between observing this repository and committing to it.

    Distinct from `verification_contract_invalid`: nothing is wrong with the new
    contract. It is simply not the one this submit measured the workspace
    against, and a job pinned for its whole life to a digest that no longer
    describes the file on disk is a job nobody can audit honestly.

    Refused before a job, a lease, a worktree claim, a continuation claim or a
    session exists, so the caller may simply resubmit and be admitted against
    the contract that is actually there.

    Retryable, unlike every other `VerificationRequired`. Those mean "somebody
    must fix something before this can ever work"; this one means "two things
    happened in the wrong order". Nothing is broken and there is no required
    action, so reporting it as terminal would stop a scheduler on a race that
    resolves itself on the next attempt. A continuation retry is retryable in
    exactly the same way - it just then meets the ordinary
    `continuation_contract_changed` rule, because an authority granted under
    the old contract cannot be relicensed under a new one.
    """

    code = "verification_contract_changed"
    retryable = True


class VerificationToolMissing(VerificationRequired):
    """A required check is declared correctly and its program is not installed.

    Three neighbouring refusals, three different people: `verification_required`
    is for whoever owns the repository's contract, `capability_unavailable` is
    for whoever provisions the capability bridge, and this one is for whoever
    provisions the host's tooling. Collapsing them would leave every recipient
    reading prose to find out whether the work is theirs.

    Carries the names of the checks that cannot run. They are contract
    identifiers, already validated as safe tokens, so they say *which* tool to
    install without the refusal becoming a channel for paths or argv.
    """

    code = "verification_tool_missing"

    def __init__(
        self,
        message: str,
        required_actions: Sequence[str] = (),
        blockers: Sequence[str] = (),
    ) -> None:
        super().__init__(message, required_actions)
        self.blockers = tuple(blockers)

    @property
    def context(self) -> Dict[str, Any]:
        if not self.blockers:
            return {}
        return {"verification_blockers": tuple(self.blockers)}


class CapabilityUnavailable(VerificationRequired):
    """A required capability cannot be attached on this host.

    The contract is fine; the environment is not. Kept distinct so an operator
    installing a missing tool is never sent to edit a correct contract.
    """

    code = "capability_unavailable"


class CodingServiceReloadRequired(CodingServiceError):
    """The source tree changed after this long-lived service imported it."""

    code = "service_reload_required"


class CodingJobNotFound(CodingServiceError):
    code = "job_not_found"


class IdempotencyConflict(CodingServiceError):
    code = "idempotency_conflict"


class WorkspaceDenied(CodingServiceError):
    code = "workspace_denied"


class WorkspaceBusy(CodingServiceError):
    """Another job owns this worktree until its audit loop closes.

    Ownership spans the whole job, not one implementation round, so a second
    Codex frontend cannot edit a tree between an implementation and the exact
    revision audit that binds it. The owning job id is safe to publish: it is
    an opaque host-minted token, never a path or a prompt.
    """

    code = "workspace_busy"
    failure_phase = "workspace"

    def __init__(self, message: str, owner_job_id: str = "") -> None:
        super().__init__(message)
        self.owner_job_id = str(owner_job_id or "")

    @property
    def context(self) -> Dict[str, Any]:
        return {"owner_job_id": self.owner_job_id} if self.owner_job_id else {}


class WorkspaceClaimUnresolved(CodingServiceError):
    """A worktree carries a claim whose authority cannot be evaluated.

    This is deliberately distinct from `workspace_busy`. Busy means "a named
    live job owns this tree"; unresolved means "something claims this tree and
    the service cannot prove otherwise". Both refuse the edit, but only the
    second needs a host operator to look at the state root, because no job
    transition will ever clear it on its own.
    """

    code = "workspace_claim_unresolved"
    failure_phase = "workspace"

    def __init__(self, message: str, owner_job_id: str = "") -> None:
        super().__init__(message)
        self.owner_job_id = str(owner_job_id or "")

    @property
    def context(self) -> Dict[str, Any]:
        return {"owner_job_id": self.owner_job_id} if self.owner_job_id else {}


class AbandonStateConflict(CodingServiceError):
    """Only an audit-ready job may be abandoned, and only into a failure."""

    code = "abandon_state_conflict"


class AuditNotEnabled(CodingServiceError):
    code = "audit_not_enabled"


class AuditStateConflict(CodingServiceError):
    code = "audit_state_conflict"


class RevisionMismatch(CodingServiceError):
    code = "revision_mismatch"


class RevisionUnavailable(CodingServiceError):
    code = "revision_unavailable"


class SessionBindingFailed(CodingServiceError):
    code = "session_binding_failed"


class ContinuationRefused(CodingServiceError):
    """A resume request was not granted, and says as little as it safely can.

    Refused in `submit`, before a job, a lease, a claim or a session exists, so
    `preflight` is the truthful phase. Never retryable on its own: every code
    here needs somebody to change something first - restore the bytes, restore
    the contract, or accept that this session is finished.

    The default code is deliberately the collapsed one. A caller learns a
    specific reason only after it has already proven it owns the authority; up
    to that point every distinguishable answer would be an existence oracle for
    somebody else's session.
    """

    code = CONTINUATION_UNAVAILABLE
    failure_phase = "preflight"
    retryable = False

    def __init__(self, message: str, code: str = CONTINUATION_UNAVAILABLE) -> None:
        super().__init__(message)
        # Bounded by the module's own closed set, so a future caller cannot
        # smuggle prose into a facade through this field.
        self.code = code if code in CONTINUATION_CODES else CONTINUATION_UNAVAILABLE


#: Every way a job's durable Indexer plan authority, or the cumulative scope
#: it governs, can fail to be provable. Closed by construction: each is a stable
#: token a caller may branch on, and none of them ever carries contract content,
#: a path, a session, or provider prose.
PLAN_AUTHORITY_CODES: Tuple[str, ...] = (
    "plan_authority_unavailable",
    "plan_authority_unsealable",
    "cumulative_scope_unproven",
    "cumulative_scope_unbounded",
    "cumulative_session_unproven",
    "cumulative_workspace_unproven",
    "cumulative_claim_unproven",
    "cumulative_resume_unproven",
    "cumulative_revision_mismatch",
)


class PlanAuthorityUnprovable(CodingServiceError):
    """This job cannot prove what its root task is authorized to change.

    Lifecycle, stated precisely because the phase depends on it: this is
    *pre-implementer*, not submit-preflight. The job already exists, it already
    holds a durable worktree claim, admission and capability startup already
    happened. What has not happened is the implementer call, so no session was
    opened, no provider budget was spent and nothing was produced to audit.
    `preflight` is therefore the wrong phase - that one promises no job and no
    claim - and `verification` is the truthful one: the host could not verify
    the authority this round would have run under.

    Not retryable. Every member of the closed set means a durable fact is
    missing, stale, replayed or contradicted, so an identical rework of this
    job cannot repair it. The way forward is a fresh job planned against the
    authority that actually exists now, which is exactly what
    `resubmit_against_current_contract` names.

    The code is the whole message. The contract itself, the paths it names and
    the session it belongs to stay in the private job record.
    """

    code = "plan_authority_unavailable"
    failure_phase = FAILURE_PHASE_VERIFICATION
    retryable = False
    required_actions = (ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,)

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code if code in PLAN_AUTHORITY_CODES else "plan_authority_unavailable"


class ReworkLimitReached(CodingServiceError):
    code = "rework_limit_reached"


class ReworkNotResumable(CodingServiceError):
    code = "rework_not_resumable"


class RouteEvidenceMissing(CodingServiceError):
    code = "route_evidence_missing"


class AuditBlockersUnresolved(CodingServiceError):
    """An auditable revision still carries host blockers, so it cannot land.

    This is deliberately not `audit_state_conflict`: the job *is* auditable and
    the auditor *is* authorized. Only `accept` is refused, and only while the
    blockers stand. `rework` remains available and is the way through.
    """

    code = "audit_blockers_unresolved"


class EmergencyAuthorityMissing(CodingServiceError):
    """An emergency-bound job cannot be served without its authority contract."""

    code = "emergency_authority_missing"


#: States whose persisted route evidence must still hold when read back.
class _RoundSettlement:
    """One round's mission closure: performed once, published with its state.

    The invariant this type exists to hold is a publication order, not a lock.
    A round used to write `failed` or `awaiting_codex_audit` and only then close
    its work item, which left a real window in which a poller saw a terminal job
    whose item this process still held dispatched - and, worse, in which an
    auditor could accept that job and schedule a repair child before the parent
    item settled, so the late closure would then rewrite the record's projection
    onto the wrong work item.

    So the closure happens *before* the state is published and returns the
    record change instead of writing it. Every publishing path folds that change
    into the same `_update_record`, which makes "this job is terminal" and "this
    round's item is owner-closed" one durable fact rather than two.

    Calling it twice is a no-op that replays the first answer: a backstop in the
    worker's `finally` can therefore always run without risking a second close
    or a second projection move.
    """

    __slots__ = ("_service", "_work", "_tenant_ref", "_job_id", "_settled", "_changes")

    def __init__(
        self,
        service: "CodingService",
        work: Optional["DispatchedWork"],
        tenant_ref: str,
        job_id: str,
    ) -> None:
        self._service = service
        self._work = work
        self._tenant_ref = tenant_ref
        self._job_id = job_id
        #: A round with no dispatch handle - a legacy direct call - has nothing
        #: to settle and is born settled, so every call site stays uniform.
        self._settled = work is None
        self._changes: Dict[str, Any] = {}

    @property
    def settled(self) -> bool:
        return self._settled

    @property
    def work_item_id(self) -> str:
        return "" if self._work is None else self._work.work_item_id

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
        work = self._work
        assert work is not None  # `_settled` is True from birth when it is None
        self._changes = self._service._close_round_item(
            work,
            self._tenant_ref,
            self._job_id,
            revision=revision,
            files=files,
            state=state,
            failure_code=failure_code,
        )
        return dict(self._changes)


class MissionRouteRefused(CodingServiceError):
    """The mission lane refused this job, under the kernel's own stable code.

    One class, many codes. `capacity`, `conflict`, `stale fence`, `dependency`,
    `corruption`, `authority` and `unsupported host` are already a closed,
    machine-readable vocabulary in `flyto_ai.coding.mission_runtime`, so this
    adopts that code rather than inventing a parallel one - a caller branching
    on `code` sees exactly what the kernel decided, and `retryable` travels with
    it so backing off is never a guess.
    """

    failure_phase = "mission"

    def __init__(self, exc: MissionRouteError) -> None:
        super().__init__(str(exc))
        self.code = getattr(exc, "code", "mission_unavailable")
        self.retryable = bool(getattr(exc, "retryable", False))


#: How long a pump waits for work the store is holding behind a resource
#: conflict before giving the worker thread back. Bounded on purpose: a pump
#: that waited forever would be a worker a permanently blocked worktree could
#: consume. The wait backs off from `_PUMP_POLL_SECONDS` to
#: `_PUMP_POLL_CEILING`, because each attempt rewrites the durable store.
_PUMP_WAIT_SECONDS = 120.0
_PUMP_POLL_SECONDS = 0.02
_PUMP_POLL_CEILING = 0.2
#: How many times one pump will accept work whose round another process has
#: leased before concluding that process owns it. Requeueing is free; retrying
#: forever would burn a fencing token per attempt for nothing.
_PUMP_MAX_FOREIGN = 3

#: What one dispatch attempt did. A pump treats all three differently, so they
#: are named rather than encoded as a bare boolean.
_DISPATCH_RAN = "ran"
_DISPATCH_IDLE = "idle"
_DISPATCH_FOREIGN = "foreign"

_ROUTE_EVIDENCE_STATES = frozenset({
    CodingJobState.AWAITING_CODEX_AUDIT.value,
    CodingJobState.REWORK_QUEUED.value,
    CodingJobState.REWORK_RUNNING.value,
    CodingJobState.CODEX_ACCEPTED.value,
})

_INTERRUPTED_JOB_STATES = frozenset({
    CodingJobState.QUEUED.value,
    CodingJobState.RUNNING.value,
    CodingJobState.REWORK_QUEUED.value,
    CodingJobState.REWORK_RUNNING.value,
})

#: States during which one job owns its worktree exclusively. This is a strict
#: superset of `_INTERRUPTED_JOB_STATES`: it also covers the audit gap, where
#: no round is executing but the recorded revision must still describe the
#: files an auditor will read. Releasing at the end of a round is exactly what
#: let a competing job invalidate an audit before it happened.
_CLAIM_OWNED_STATES = _INTERRUPTED_JOB_STATES | {
    CodingJobState.AWAITING_CODEX_AUDIT.value,
}

#: The one terminal vocabulary, projected to the raw strings a durable record
#: stores. Derived from `TERMINAL_CODING_JOB_STATES` so a new terminal state
#: can never be added in one place and forgotten in the other.
_TERMINAL_STATE_VALUES = frozenset(
    state.value for state in TERMINAL_CODING_JOB_STATES
)


def route_blocks_implementation(route: "CodingRouteReceipt") -> bool:
    """Whether a failed route stopped only on the implementation's own result.

    Every host lane ran and was trusted; the post-work lane simply refused to
    certify a round whose own required checks did not pass. That is a statement
    about the change, not about the route, so the round stays auditable for
    rework. It is never a passed route and never makes anything landable.
    """

    if not isinstance(route, CodingRouteReceipt) or route.ok or not route.strict:
        return False
    lane, _action, code = route_failure_point(route)
    return lane == IMPLEMENTATION_BLOCKER_LANE and code in IMPLEMENTATION_BLOCKER_ROUTE_CODES


def recorded_blockers(record: Mapping[str, Any]) -> Tuple[str, ...]:
    """Read one record's blocker list, dropping anything outside the bounds."""

    stored = record.get("implementation_blockers")
    if not isinstance(stored, (list, tuple)):
        return ()
    return tuple(
        item for item in stored
        if isinstance(item, str)
        and not isinstance(item, bool)
        and _BLOCKER_CODE_RE.fullmatch(item)
    )[:MAX_IMPLEMENTATION_BLOCKERS]


class _RoundProgress:
    """What one execution round actually did, tracked by the host itself.

    `begin()` runs immediately before the selected implementer is invoked,
    never because a job entered `running`. It writes the durable record first
    and the in-memory flag second, so a process that dies while the model is
    working still leaves a job record that says the implementer started.
    """

    def __init__(self, on_start: Optional[Callable[[], None]] = None) -> None:
        self.implementer_started = False
        #: The exact ordered attributable set this round handed to the proof
        #: lanes, recorded by the route seam. The durable revision must bind
        #: this same tuple or the round is refused: a set the Indexer validated
        #: and a different set an auditor is offered are not the same evidence.
        self.route_scope: Tuple[str, ...] = ()
        self.emergency = False
        self.trigger: Optional[EmergencyTrigger] = None
        self._on_start = on_start

    def begin(self) -> None:
        """Record the start of one implementer invocation, exactly once."""
        if self.implementer_started:
            return
        if self._on_start is not None:
            self._on_start()
        self.implementer_started = True


def _arm_start_marker(store: Any, progress: Optional[Any]) -> None:
    """Let the adapter say when a provider attempt really begins.

    The marker used to be written just before the adapter was entered, which
    made it a statement about this service rather than about the run: an adapter
    that refused before ever contacting a provider - an unhonourable contract, a
    verification tool that is not installed - was durably recorded as having
    started. The hook moves the signal to the first real provider or session
    call, where it is both true and still early enough to survive a worker that
    dies inside that call.
    """

    if progress is None:
        return
    try:
        store.on_provider_start = progress.begin
    except Exception as exc:
        # Failing closed, because the alternative was worse than useless: a
        # store that cannot carry the callback used to be handled by declaring
        # the provider started, which is the exact false record this seam
        # exists to remove. The production store is host-owned and supports
        # this; anything that does not is a wiring error, not a run.
        raise RuntimeError(
            "the implementation store cannot carry the provider start signal",
        ) from exc


def _arm_session_binding(store: Any, binder: Optional[Callable[[str], None]]) -> None:
    """Let the adapter say which session the provider actually established.

    Separate from `_arm_start_marker` because the two facts are separate. The
    start marker answers "was an implementer invoked", and it is true before any
    identity exists. This answers "which conversation is this round", and it is
    only true once a backend has said so.

    Binding at that moment - rather than when the whole agent call returns - is
    what makes a bounded stop continuable at all. A round that is killed, or
    that dies against a ceiling six minutes in, has already had its session
    written down under the job that owns the worktree.

    Fails closed for the same reason the start marker does: a store that cannot
    carry the callback is a wiring error, and running a round whose identity
    nobody records is exactly the state being removed.
    """

    if binder is None:
        return
    try:
        store.on_provider_session = binder
    except Exception as exc:
        raise RuntimeError(
            "the implementation store cannot carry the provider session signal",
        ) from exc


def _reconcile_start_marker(progress: Optional[Any], result: Any) -> None:
    """Record a start the adapter proved but did not signal.

    Belt and braces for an implementer that predates the hook: a result showing
    a real attempt means a provider ran, whoever failed to say so.
    """

    if progress is None or getattr(progress, "implementer_started", False):
        return
    if int(getattr(result, "attempts", 0) or 0) >= 1:
        progress.begin()


class CodingImplementer(Protocol):
    """One implementation round, whichever backend the host selected.

    The native `FlytoCodingAgent` and the optional Claude adapter are explicit
    peers at this boundary; neither is privileged and neither is a fallback for
    the other.
    """

    async def run(self, request: CodingTaskRequest) -> CodingTaskResult:
        ...  # pragma: no cover - structural protocol


AgentFactory = Callable[[ThreadStore], CodingImplementer]


def request_from_mapping(value: Mapping[str, Any]) -> CodingTaskRequest:
    """Decode the public service request; provider and tenant fields are forbidden."""

    if not isinstance(value, Mapping):
        raise ValueError("coding request must be an object")
    unknown = set(value) - _DECODABLE_REQUEST_FIELDS
    if unknown:
        raise ValueError("unsupported coding request fields: {}".format(", ".join(sorted(unknown))))
    # Absent and explicitly null are different payloads and are answered
    # differently. The published schema types `mission` as an object, so a
    # caller that sent the key sent something the schema does not describe;
    # silently reading it as "no mission" would let a client believe it had
    # named one and get a job that ignored it. Both transports decode here, so
    # MCP and HTTP cannot drift on this.
    if "mission" in value:
        mission_value = value["mission"]
        if mission_value is None:
            raise ValueError("mission must be an object; omit the key to send no mission")
        mission = CodingMissionEnvelope.from_mapping(mission_value)
    else:
        mission = None
    return CodingTaskRequest(
        message=str(value.get("message", "")),
        working_dir=str(value.get("working_dir", "")),
        thread_id=str(value["thread_id"]) if value.get("thread_id") is not None else None,
        resume=bool(value.get("resume", False)),
        max_attempts=int(value.get("max_attempts", 3)),
        max_rounds=int(value.get("max_rounds", 30)),
        require_changes=bool(value.get("require_changes", True)),
        mission=mission,
    )


def error_details(exc: BaseException) -> Dict[str, Any]:
    """Project one error's structured context, or nothing at all.

    Only short, closed-vocabulary scalars cross this boundary, so a facade can
    publish "which job owns that worktree" without turning a tool result or an
    HTTP body into an unbounded log channel. Prose, paths, floats, and nested
    payloads are dropped rather than truncated.
    """

    details = getattr(exc, "details", None)
    if not isinstance(details, Mapping):
        return {}
    projected: Dict[str, Any] = {}
    for key, value in sorted(details.items())[:_MAX_ERROR_DETAIL_FIELDS]:
        if not isinstance(key, str) or not _DETAIL_KEY_RE.fullmatch(key):
            continue
        if isinstance(value, bool) or isinstance(value, int):
            projected[key] = value
        elif isinstance(value, str) and _DETAIL_VALUE_RE.fullmatch(value):
            projected[key] = value
        elif isinstance(value, (list, tuple)):
            # Two bounded list shapes, and no third. Most keys may carry only
            # tokens from a closed allowlist, which is what lets
            # `required_actions` reach a caller as data it can branch on rather
            # than as prose; anything not on the allowlist is dropped, not
            # truncated, so this can never become an open string channel.
            #
            # `_IDENTIFIER_DETAIL_KEYS` is the single exception, and it is
            # narrow by construction: the values are contract identifiers whose
            # shape the config parser already enforced, they are matched against
            # the same strict pattern as any scalar detail, and the key itself
            # has to be one this module named in advance. A caller learns which
            # declared thing blocked it; it never learns a path, an argv or a
            # message.
            if key in _IDENTIFIER_DETAIL_KEYS:
                allowed = [
                    item for item in value
                    if isinstance(item, str) and _DETAIL_VALUE_RE.fullmatch(item)
                ]
            else:
                allowed = [
                    item for item in value
                    if isinstance(item, str) and item in _PROJECTABLE_TOKENS
                ]
            if allowed:
                projected[key] = sorted(set(allowed))[:_MAX_ERROR_DETAIL_TOKENS]
    return projected


def receipt_to_mapping(receipt: CodingJobReceipt) -> Dict[str, Any]:
    """Return a JSON-safe, secret-redacted public receipt."""

    projected = dataclasses.asdict(receipt)
    # Derived, not stored: `asdict` sees dataclass fields only, so the terminal
    # signal is injected here to keep one source of truth in the receipt.
    projected["job_terminal"] = receipt.job_terminal
    phase, retryable, actions = receipt.failure_semantics
    projected["failure_phase"] = phase
    projected["retryable"] = retryable
    projected["required_actions"] = list(actions)
    result = projected.get("result")
    if isinstance(result, dict):
        result.pop("evidence_path", None)
        for check in result.get("checks", []):
            if isinstance(check, dict):
                check.pop("output_preview", None)
    mission = projected.get("mission")
    if isinstance(mission, dict):
        # Belt and braces over the receipt's own revalidation: re-project
        # through the closed field set, so no key can ride out of this facade
        # that `CodingMissionProjection` does not publish.
        projected["mission"] = {
            key: mission[key] for key in sorted(MISSION_PROJECTION_FIELDS) if key in mission
        }
    return redact_evidence(projected)


class CodingService:
    """Bounded asynchronous facade; provider credentials never enter a job."""

    def __init__(
        self,
        agent_factory: AgentFactory,
        *,
        state_root: str,
        workspace_roots: Sequence[str],
        max_workers: int = 2,
        max_queued: int = 100,
        approval_policy: ApprovalPolicy = ApprovalPolicy.NEVER,
        sandbox_mode: SandboxMode = SandboxMode.WORKSPACE_WRITE,
        config_path: str = ".flyto/coding.yaml",
        sandbox_image: str = "python:3.12-slim",
        require_codex_audit: bool = False,
        implementation_backend: str = "native",
        attachable_capability_kinds: Optional[Sequence[str]] = None,
        max_rework_rounds: int = 3,
        route_policy: Optional[CodingRoutePolicy] = None,
        emergency_policy: Optional[EmergencyOverflowPolicy] = None,
    ) -> None:
        if not 1 <= max_workers <= 16:
            raise ValueError("max_workers must be between 1 and 16")
        if not max_workers <= max_queued <= 1000:
            raise ValueError("max_queued must be between max_workers and 1000")
        roots = tuple(Path(path).expanduser().resolve() for path in workspace_roots)
        if not roots or any(not path.is_dir() for path in roots):
            raise ValueError("workspace_roots must contain existing directories")
        self.agent_factory = agent_factory
        # Not `.resolve()` + `mkdir(parents=True)`. That pair follows links
        # twice - once to decide where to write, once to write - and the store
        # below is built to refuse exactly what it would have followed. The
        # refusal has to happen here, before any directory exists, or the
        # service creates state on the far side of a link and only then hands
        # a store that would have said no.
        self.state_root = secure_directory(Path(state_root))
        self.workspace_roots = roots
        self.max_queued = max_queued
        self.approval_policy = ApprovalPolicy(approval_policy)
        self.sandbox_mode = SandboxMode(sandbox_mode)
        self.config_path = str(config_path)
        self.sandbox_image = str(sandbox_image)
        # Audit authority, implementer identity, and the rework ceiling are
        # startup decisions. No job payload can reach them.
        if not isinstance(require_codex_audit, bool):
            raise ValueError("require_codex_audit must be a boolean")
        self.require_codex_audit = require_codex_audit
        if not isinstance(implementation_backend, str) or not _BACKEND_ID.fullmatch(
            implementation_backend,
        ):
            raise ValueError("implementation_backend must be a safe non-empty identifier")
        self.implementation_backend = implementation_backend
        # What the *selected* implementer can bridge, declared by whoever wired
        # it. `None` means unproven and is treated as "nothing", so a contract
        # that requires a capability is refused at preflight rather than
        # accepted here and refused later inside the implementer. Backends
        # publish their own answer as `attachable_capability_kinds`.
        self.attachable_capability_kinds = frozenset(attachable_capability_kinds or ())
        if isinstance(max_rework_rounds, bool) or not isinstance(max_rework_rounds, int):
            raise ValueError("max_rework_rounds must be an integer")
        if not 1 <= max_rework_rounds < MAX_AUDIT_ROUNDS:
            raise ValueError(
                "max_rework_rounds must be between 1 and {}".format(MAX_AUDIT_ROUNDS - 1),
            )
        self.max_rework_rounds = max_rework_rounds
        if route_policy is not None and not isinstance(route_policy, CodingRoutePolicy):
            raise ValueError("route_policy must be a CodingRoutePolicy")
        # Host-owned lane authority is a startup decision like every other
        # field above. No job payload can enable, detach, or relax it.
        self.route_policy = route_policy
        if emergency_policy is not None and not isinstance(
            emergency_policy, EmergencyOverflowPolicy,
        ):
            raise ValueError("emergency_policy must be an EmergencyOverflowPolicy")
        # Emergency overflow authority is startup-only too, and it is granted
        # for one named implementer. A policy naming a different backend than
        # the one selected above is a configuration error, not a silent
        # redirect of work to an unintended agent.
        self.emergency_policy = emergency_policy or EmergencyOverflowPolicy()
        if self.emergency_policy.enabled and not self.emergency_policy.applies_to(
            self.implementation_backend,
        ):
            raise ValueError(
                "emergency_policy backend must match the selected implementation backend",
            )
        self._breaker = EmergencyCircuitBreaker(self.emergency_policy)
        # Tenant-partitioned continuation authorities. Reads and writes always
        # go through the authenticated caller's own partition, which is what
        # makes a guessed session from another tenant indistinguishable from a
        # session that never existed.
        self._continuation = ContinuationStore(self.state_root)
        # Which parts of a workspace this host calls source. The strict public
        # route is the only configuration that may classify anything as
        # control-plane runtime state, because it is the only one whose
        # mandatory Indexer pre/post gates independently revalidate that tree
        # and record the result in the route receipt. Everything else - a
        # legacy library service, a non-strict route, a route without an
        # Indexer - gets the default policy and observes the whole tree.
        self.snapshot_policy = self._startup_snapshot_policy(route_policy)
        # A fresh instance is a fresh, closed circuit with a new opaque id and
        # the build digest of the sources this process actually loaded.
        self.instance_id = uuid.uuid4().hex[:24]
        self.build_id = service_build_id()
        self.service_version = service_version()
        self.started_at = time.time()
        self._status_published = 0
        self._status_failures = 0
        self._status_failure_code = ""
        self._last_status_record: Dict[str, Any] = {}
        # Reuse the request contract's path/image validation without persisting
        # a synthetic request or accepting those authority fields remotely.
        config_parts = Path(self.config_path).parts
        if Path(self.config_path).is_absolute() or "\x00" in self.config_path or ".." in config_parts:
            raise ValueError("config_path must be relative")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:-]*", self.sandbox_image):
            raise ValueError("sandbox_image is invalid")
        self._lock = threading.RLock()
        self._state_lock_depth = 0
        self._workspace_locks: Dict[str, threading.Lock] = {}
        #: Per-workspace admission locks, distinct from the per-round locks
        #: above so a submit never queues behind a running model round.
        self._admission_locks: Dict[str, threading.Lock] = {}
        self._job_leases: Dict[str, int] = {}
        # In-memory resume context is only the fast path. The durable, redacted
        # envelope beside each job record is what makes rework possible from a
        # worker that never implemented it, and it is bound to one exact
        # implementation session so it can continue that session but never
        # start a new one.
        self._resume: Dict[Tuple[str, str], CodingTaskRequest] = {}
        self._pending: set[Future[Any]] = set()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="flyto-coding")
        self._closed = False
        # The durable, cross-process mission queue. Constructing it creates
        # nothing, so an unsupported host is discovered by asking rather than by
        # leaving half a store behind; admission is where the refusal lands.
        self._mission = CodingMissionRuntime(
            self.state_root, worker=worker_identity(self.instance_id),
        )
        #: How many work items start-up reconciliation returned to the queue.
        #: Each one earns exactly one pump, so accounting for them stays on the
        #: ordinary store-ordered route.
        self._reclaimed = 0
        try:
            self._lock_fd = os.open(
                self.state_root / ".service.lock", os.O_CREAT | os.O_RDWR, 0o600,
            )
        except OSError:
            # Never leave worker threads behind when construction fails.
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise
        try:
            (self.state_root / "locks" / "jobs").mkdir(
                parents=True, exist_ok=True, mode=0o700,
            )
            (self.state_root / "locks" / "workspaces").mkdir(
                parents=True, exist_ok=True, mode=0o700,
            )
            self._status = RouteStatusPublisher(
                self.state_root,
                instance_id=self.instance_id,
                build_id=self.build_id,
                version=self.service_version,
                started_at=self.started_at,
            )
            with self._state_guard():
                self._publish_status({})
                self._reconcile_interrupted_jobs()
        except BaseException:
            self.close(wait=False)
            raise

    def submit(
        self,
        tenant_id: str,
        idempotency_key: str,
        request: CodingTaskRequest,
    ) -> CodingJobReceipt:
        """Create or reuse one tenant-owned job and schedule it exactly once."""

        tenant_ref = self._tenant_ref(tenant_id)
        if not _SAFE_ID.fullmatch(idempotency_key):
            raise ValueError("idempotency_key must be a safe identifier")
        self._assert_workspace(request.working_dir)
        request = self._with_startup_authority(request)
        request_digest = self._request_digest(request)
        tenant_dir = self._tenant_dir(tenant_ref)
        idempotency_path = tenant_dir / "idempotency" / (hashlib.sha256(idempotency_key.encode()).hexdigest() + ".json")

        # Phase 0, unlocked: an idempotent replay must not pay for a repository
        # scan it does not need. This read is advisory - the authoritative one
        # happens under the guard below - but it is exact when it hits, because
        # an idempotency record is written once and never rewritten.
        replay = self._replayed_receipt(
            tenant_ref, tenant_dir, idempotency_path, request_digest,
        )
        if replay is not None:
            return replay

        # Phase 1, workspace-scoped: everything expensive. Reading the
        # verification contract and snapshotting the workspace can take tens of
        # seconds on a large repository, and doing that under the global state
        # guard put every other tenant, every other workspace and every other
        # Codex behind whichever submit happened to be scanning. The lock here
        # is per workspace, so two submits for the *same* tree still cannot both
        # observe-then-claim, and an unrelated tree is not delayed at all.
        #
        # Lock order is workspace-admission -> state guard, everywhere, and the
        # admission lock is never taken while the state guard or a round's
        # workspace lock is held. There is therefore no cycle to deadlock on.
        with self._admission_lock(request.working_dir):
            # Observation first, and deliberately so. A continuation does not
            # derive its contract from the tree it is re-entering: the stopped
            # round may itself have edited `.flyto/coding.yaml`, and that edit
            # is part of the exact revision a continuation must find unchanged.
            # Reading the current file to decide a continuation's authority is
            # what made the two invariants unsatisfiable at once.
            #
            # This claims nothing, so nothing is consumed if the transition
            # below is refused.
            observed = self._observe_continuation(tenant_ref, request)
            if observed is None:
                # An ordinary new job. One read decides feasibility *and* pins,
                # by value, the contract this job is authorized against for its
                # whole life, rework rounds included.
                pinned = self._pin_verified_contract(request.working_dir)
                observed_config = pinned.config_sha256
            else:
                # A continuation inherits the origin's pin, recovered from
                # private tenant state and re-proved against the identity the
                # authority itself binds. The current file is not consulted for
                # authority at all - only for drift, below.
                pinned = self._restore_pinned_contract(tenant_ref, observed)
                observed_config = self._observed_config_digest(request.working_dir)
            authorized_config = pinned.config_sha256
            request = dataclasses.replace(
                request,
                authorized_config_sha256=authorized_config,
                pinned_contract=pinned,
            )
            return self._commit_admission(
                tenant_ref, tenant_dir, idempotency_path, request,
                request_digest, authorized_config, observed, pinned,
                observed_config,
            )

    def _pin_verified_contract(self, workspace: str) -> ContractSnapshot:
        """Prove this repository can be verified, and pin what it declared.

        Two reads, bound together by a digest rather than by hope. Preflight
        performs its own read and returns the digest it decided from; this then
        reads the document again to capture the checks, capabilities and actions
        by value, and refuses unless that second read hashes to exactly what
        preflight approved. A contract swapped between the two is therefore a
        refusal, not a snapshot of one document wearing another's verdict.

        The snapshot is taken here, at submit, before any implementer has been
        constructed and long before a provider has been contacted. That timing
        is the whole guarantee: everything this job later executes was
        authorized before the first provider edit could exist.
        """

        authorized = self._require_verifiable_repository(workspace)
        try:
            contract = read_project_contract(workspace, self.config_path)
        except (OSError, ValueError) as exc:
            raise VerificationContractChanged(
                "the repository verification contract changed during admission",
            ) from exc
        if contract.digest != authorized:
            raise VerificationContractChanged(
                "the repository verification contract changed during admission",
            )
        return contract.snapshot()

    @staticmethod
    def _record_pinned_contract(
        record: Mapping[str, Any],
    ) -> Optional[ContractSnapshot]:
        """Rebuild a job's pin from its own durable record, or refuse to guess.

        Revalidating rather than trusting: the stored mapping goes back through
        the full contract grammar, and the identity the record also stored must
        still match. A record whose snapshot was edited in place therefore
        yields `None` rather than a weakened contract, and `None` fails closed
        everywhere it is consumed - a round with no pin and a non-empty
        `authorized_config_sha256` falls back to the historical digest gate,
        which refuses outright when the file no longer matches.
        """

        stored = record.get("contract_snapshot")
        if stored is None:
            return None
        try:
            pinned = ContractSnapshot.from_mapping(stored)
        except (ValueError, TypeError):
            return None
        expected = str(record.get("contract_snapshot_sha256") or "")
        if not expected or pinned.identity() != expected:
            return None
        return pinned

    def _observed_config_digest(self, workspace: str) -> str:
        """The current contract file's digest, or "" when it has none to give.

        Never raises. This is a drift probe, not an authority decision: a
        continuation's authority comes from its pin, and whether the file on
        disk currently parses is a fact about the tree the model produced. The
        empty string is a real observation ("unreadable or absent"), and two
        empty observations compare equal, so an unparseable contract that stays
        unparseable is correctly seen as *not* having drifted.
        """

        try:
            return read_project_contract(workspace, self.config_path).digest
        except (OSError, ValueError):
            return ""

    def _restore_pinned_contract(
        self, tenant_ref: str, authority: ContinuationAuthority,
    ) -> ContractSnapshot:
        """Recover the origin job's pin, and prove it is the one bound here.

        The snapshot lives in the origin job's private record; the authority
        carries only its content address. Rebuilding goes back through the full
        validating constructor, so a hand-edited state file cannot introduce a
        check, capability or action shape the contract grammar would refuse -
        and the identity comparison then rejects any snapshot that is
        well-formed but simply not the one this session was granted under.

        Reached only after `_observe_continuation` has proven tenant, session,
        backend, workspace, policy, revision and manifest. A caller that has not
        proven all of that never learns anything from here.
        """

        record: Mapping[str, Any]
        try:
            record = self._read_json(
                self._tenant_dir(tenant_ref, create=False)
                / "jobs" / (authority.origin_job_id + ".json"),
            )
        except (OSError, ValueError):
            raise ContinuationRefused(
                "the pinned verification contract can no longer be recovered",
                CONTINUATION_CONTRACT_UNPINNED,
            ) from None
        try:
            pinned = ContractSnapshot.from_mapping(record.get("contract_snapshot"))
        except (ValueError, TypeError):
            raise ContinuationRefused(
                "the pinned verification contract can no longer be recovered",
                CONTINUATION_CONTRACT_UNPINNED,
            ) from None
        if pinned.identity() != authority.contract_snapshot_sha256:
            # Stored, well-formed, and not what this authority was granted
            # under. Tampering and a mismatched restore are the same refusal.
            raise ContinuationRefused(
                "the pinned verification contract does not match this authority",
                CONTINUATION_CONTRACT_CHANGED,
            )
        return pinned

    def _replayed_receipt(
        self,
        tenant_ref: str,
        tenant_dir: Path,
        idempotency_path: Path,
        request_digest: str,
    ) -> Optional[CodingJobReceipt]:
        """Return the receipt an idempotency key already names, or `None`."""

        try:
            reference = self._read_json(idempotency_path)
        except (FileNotFoundError, NotADirectoryError):
            return None
        except (OSError, ValueError):
            # Unreadable here is not decisive; the authoritative check under
            # the guard will raise if it is really broken.
            return None
        if reference.get("request_sha256") != request_digest:
            raise IdempotencyConflict("idempotency key was already used for another request")
        referenced = str(reference.get("job_id", ""))
        if not _JOB_ID.fullmatch(referenced):
            raise CodingServiceError("idempotency record is invalid")
        try:
            record = self._read_json(tenant_dir / "jobs" / (referenced + ".json"))
        except (OSError, ValueError):
            return None
        return self._public_receipt(tenant_ref, record)

    def _commit_admission(
        self,
        tenant_ref: str,
        tenant_dir: Path,
        idempotency_path: Path,
        request: CodingTaskRequest,
        request_digest: str,
        authorized_config: str,
        observed: Optional[ContinuationAuthority],
        pinned: ContractSnapshot,
        observed_config: str,
    ) -> CodingJobReceipt:
        """Re-prove the stopped tree, then take the short guarded transition.

        Not everything here is cheap. For a continuation this method performs
        the *second* whole-workspace proof, and that walk is the expensive part
        of admission. It deliberately runs outside the global state guard,
        immediately before the guarded block below, so the scan costs this
        workspace time without putting any other tenant or any unrelated
        workspace behind it. The caller's per-workspace admission lock is what
        keeps a competing submit for *this* tree from interleaving.

        What is cheap is the guarded transition itself. The caller has already
        paid for the phase-one observation, and inside the guard only a handful
        of bounded reads and writes run - including the re-read of one small
        contract file - so the global guard is held for microseconds rather
        than for the length of a scan.
        """

        if observed is not None:
            # The phase boundary itself was a time-of-check window. Phase one
            # proved the *whole* tree against the authority; the gate inside the
            # guard below then re-proved only `.flyto/coding.yaml`. An ordinary
            # tracked source file rewritten in between satisfied both and still
            # reached the claim, the worktree and the provider - which is
            # precisely the "resumed model editing a file it believes it already
            # wrote" this mechanism exists to refuse.
            #
            # Re-proving here rather than inside the guard is deliberate. This
            # walks the workspace, and a scan under the global guard would put
            # every other tenant and every unrelated workspace behind it. The
            # admission lock the caller already holds is per workspace and is
            # still held, so no competing submit for *this* tree can interleave,
            # and no unrelated tree is delayed at all.
            #
            # It runs before `_claim_continuation`, before the job lease, the
            # worktree claim, the durable record, the idempotency record and the
            # resume envelope, so a refusal leaves the authority open and
            # nothing to clean up.
            #
            # Deferred, though, when the contract file is what moved: that is
            # already the guarded gate's question, and it answers it as a
            # retryable ordering accident rather than as terminal tree drift.
            # Re-proving the tree first would relabel a race nobody can fix.
            if self._observed_config_digest(request.working_dir) == observed_config:
                self._prove_stopped_tree(observed)

        with self._state_guard():
            if self._closed:
                raise CodingServiceError("coding service is closed")
            # Authoritative replay check. A competing submit with the same key
            # may have landed while this one was scanning.
            replay = self._replayed_receipt(
                tenant_ref, tenant_dir, idempotency_path, request_digest,
            )
            if replay is not None:
                return replay
            if current_service_build_id() != self.build_id:
                # Never begin a fresh job with modules imported from a
                # different build than the files an auditor will inspect.
                # An idempotent retry above remains readable, and existing jobs
                # may still be polled/audited so their exact session can close.
                raise CodingServiceReloadRequired(
                    "coding service source changed; reload the MCP worker",
                )
            if len(self._pending) >= self.max_queued:
                raise CodingCapacityUnavailable("coding job queue is full")
            # The last authority to re-prove, and the reason it is re-proven
            # here rather than trusted from phase one: the contract read and
            # the workspace snapshot happen outside the global guard, so a
            # repository whose `.flyto/coding.yaml` is replaced in between
            # would otherwise pin this job to a digest that no longer matches
            # the file. This is a bounded re-read of one small file, not the
            # snapshot, so the guard stays short.
            #
            # It runs before the continuation claim, the lease, the worktree
            # claim and every durable record, so a change here leaves nothing
            # behind: no job to poll, no claim to release, no authority
            # consumed. An unreadable or unhonourable contract keeps its own
            # precise preflight refusal; a *different but valid* one is a
            # distinct answer, because the caller has nothing to fix.
            #
            # What is re-proved is that the contract *file* is still the one
            # phase one observed - not that it still matches this job's pinned
            # authority. For a new job those are the same digest. For a
            # continuation they are deliberately different: the pin is the
            # pre-edit contract and the file may legitimately be the post-edit
            # one, so comparing against the pin here would re-introduce exactly
            # the deadlock. Drift of the file during admission is still caught,
            # which is what this gate was always for.
            if self._observed_config_digest(request.working_dir) != observed_config:
                raise VerificationContractChanged(
                    "the repository verification contract changed during admission",
                )
            if observed is None and not observed_config:
                # A new job must have a readable contract; `_pin_verified_contract`
                # already proved that, so an empty digest here is a race.
                raise VerificationContractChanged(
                    "the repository verification contract changed during admission",
                )
            now = time.time()
            job_id = "job_{}".format(uuid.uuid4().hex[:24])
            # The claim is a compare-and-swap against the durable journal tail,
            # so an authority that moved while this submit was scanning is
            # refused here rather than double-spent. Observation was advisory;
            # this is the decision.
            granted = self._claim_continuation(tenant_ref, job_id, observed)
            if not self._acquire_job_lease(job_id):
                raise CodingServiceBusy("coding job lease is already held")
            record = {
                "job_id": job_id,
                "state": CodingJobState.QUEUED.value,
                "submitted_at": now,
                "updated_at": now,
                "request_sha256": request_digest,
                "workspace_sha256": hashlib.sha256(request.working_dir.encode()).hexdigest(),
                "working_dir": request.working_dir,
                "thread_id": "",
                "evidence_sha256": "",
                "result": None,
                "failure_code": None,
                "implementation_backend": "",
                "implementation_session_id": "",
                "implementation_revision_sha256": "",
                "implementation_files": [],
                # Host-tracked truth about whether the implementer was ever
                # invoked for this job. A queued job has not started one.
                "implementer_started": False,
                "execution_mode": EXECUTION_MODE_STRICT,
                "emergency_trigger": None,
                "emergency_authority": None,
                "audit_count": 0,
                "rework_count": 0,
                "audit_findings_sha256": "",
                # Host-owned contract authority for this job's whole life. A
                # later round re-applies exactly this, never the current file.
                "authorized_config_sha256": authorized_config,
                # The contract itself, by value, in private tenant state. This
                # record is 0600 and is never projected into a receipt, a
                # status document, a prompt, a log or an audit body: only
                # bounded, already-validated contract data lives here - no
                # workspace path, no message, no secret. It is what a rework
                # round and a restart restore, and what a continuation must
                # reproduce by identity.
                "contract_snapshot": pinned.to_mapping(),
                "contract_snapshot_sha256": pinned.identity(),
                "landable": False,
                "implementation_blockers": [],
                # Bounded continuation binding. Empty for an ordinary job; for a
                # continuation it names the exact session this job was admitted
                # to re-enter and the generation it consumed.
                "continuation_session_id": granted.session_id if granted else "",
                "continuation_generation": granted.generation if granted else 0,
                "continuation_origin_job_id": granted.origin_job_id if granted else "",
                # Bounded, secret-safe mission coordinates. Filled in below,
                # from the work item admission actually placed.
                "mission": None,
            }
            try:
                # Every job gets a mission. A caller that named one has its
                # immutable contract honoured and validated; a caller that named
                # none gets the coding adapter's synthesized contract, which is
                # built here and never inside the workload-neutral kernel.
                #
                # This runs before the worktree claim and before any durable
                # job record, so a refusal leaves no record to poll. A crash
                # *after* it leaves one ready work item whose job never came into
                # existence; the dispatch pump accounts for that item explicitly
                # rather than leaving it in the queue forever.
                admission = self._admit_mission(tenant_ref, job_id, request)
                record["mission"] = admission.projection.to_mapping()
                # Exclusive worktree ownership is taken here, once an
                # idempotent replay has already been ruled out, and it is held
                # for the whole job rather than for one round. Everything after
                # this point can therefore assume no competing job will edit
                # the tree before this one is audited.
                #
                # Only an audited job takes a claim, because only an audited
                # job has a gap between "the implementer finished" and "an
                # auditor read the tree" to protect. A legacy direct-library
                # service still honours a claim someone else holds; its own
                # rounds remain serialized by the per-round workspace lock,
                # which is the behaviour that flow has always had.
                if self.require_codex_audit:
                    self._create_workspace_claim(
                        tenant_ref, job_id, request.working_dir, record["state"],
                    )
                else:
                    self._assert_workspace_available(job_id, request.working_dir)
                self._write_json(tenant_dir / "jobs" / (job_id + ".json"), record)
                self._publish_status(record)
                self._write_json(
                    idempotency_path,
                    {"job_id": job_id, "request_sha256": request_digest},
                )
                self._resume[(tenant_ref, job_id)] = request
                self._write_resume_envelope(
                    tenant_ref, job_id, request, request_digest,
                )
                # What one dispatch needs to reconstruct this exact round, in
                # private tenant state. A first round replays the resume
                # envelope's own message, so nothing is duplicated here.
                self._write_round_envelope(
                    tenant_ref, job_id, admission.work_item_id, rework=False,
                )
                # Not `self._run_job`. The store decides which work item runs
                # next, across every process sharing this state root, so what is
                # queued here is a pump: it asks the store, and runs whatever the
                # store chose. Per-submit executor timing therefore cannot
                # reorder the queue.
                future = self._executor.submit(self._pump_dispatch)
            except BaseException:
                self._release_workspace_claim(job_id, request.working_dir)
                self._discard_resume(tenant_ref, job_id)
                # A job that never came into existence never consumed a
                # generation. Returning the authority to `open` is safe because
                # this still runs inside the guard that claimed it, so no other
                # process could have observed the claim in between.
                self._revert_continuation_claim(tenant_ref, granted, job_id)
                self._release_job_lease(job_id)
                raise
            self._pending.add(future)
            # The lease is deliberately *not* released by this callback any
            # more. A pump may run a different job than the one whose submit
            # queued it, so the round that actually executes a job is the only
            # thing that may hand back that job's lease.
            future.add_done_callback(self._forget_future)
            return self._public_receipt(tenant_ref, record)

    def _require_verifiable_repository(self, workspace: str) -> str:
        """Refuse, before anything exists, a repository that cannot be verified.

        The three outcomes are deliberately three different exception types
        rather than one with a variable message: "you have no contract", "your
        contract is wrong" and "this host is missing a tool you require" are
        resolved by different people doing different work, and a caller that
        branches on `code` must be able to tell them apart without reading
        prose.
        """

        outcome = preflight_repository(
            workspace,
            self.config_path,
            attachable_capability_kinds=self.attachable_capability_kinds,
        )
        if outcome.ok:
            return outcome.config_sha256
        if outcome.code == CODE_VERIFICATION_CONTRACT_INVALID:
            raise VerificationContractInvalid(
                "repository verification contract cannot be honoured",
                outcome.required_actions,
            )
        if outcome.code == CODE_CAPABILITY_UNAVAILABLE:
            raise CapabilityUnavailable(
                "a required capability cannot be attached on this host",
                outcome.required_actions,
            )
        if outcome.code == CODE_VERIFICATION_TOOL_MISSING:
            raise VerificationToolMissing(
                "a required verification tool is not installed on this host",
                outcome.required_actions,
                outcome.blockers,
            )
        raise VerificationRequired(
            "repository has not declared how a change must be verified",
            outcome.required_actions,
        )

    def _reassert_audit_claim(
        self, tenant_ref: str, record: Mapping[str, Any],
    ) -> None:
        """Prove the awaiting job still owns its worktree; the caller holds the guard.

        Raises `WorkspaceBusy` or `WorkspaceClaimUnresolved` without touching
        the record, the audit count, or the offending claim, so a refused audit
        leaves the job exactly as auditable as it was.
        """

        if not self.require_codex_audit:
            return
        workspace = str(record.get("working_dir") or "")
        job_id = str(record.get("job_id") or "")
        if not _JOB_ID.fullmatch(job_id) or not workspace:
            # A record that cannot even name its own job or worktree cannot
            # prove ownership of one. Returning here would let an unidentified
            # record be audited without any claim at all.
            raise WorkspaceClaimUnresolved(
                "the coding job record cannot identify its own workspace claim",
            )
        self._reassert_workspace_claim(
            tenant_ref, job_id, workspace, str(record.get("state")),
        )

    def _with_startup_authority(self, request: CodingTaskRequest) -> CodingTaskRequest:
        """Overwrite every authority field from this process's startup config.

        Applied to a submitted request and again to a rework request rebuilt
        from a durable envelope. Authority is never carried by a payload and
        never restored from disk, so a stored request cannot outlive, widen, or
        contradict the policy the running service was started with.
        """

        return dataclasses.replace(
            request,
            # Never carried by a payload: the service sets the job's contract
            # authority explicitly after preflight has read the contract, and
            # re-applies it from the durable record on every rework round.
            authorized_config_sha256="",
            # Never carried by a payload either, and for the same reason: the
            # pin *is* the verifier, so a request that could supply one could
            # choose the checks it is graded against. The service re-applies it
            # from the durable job record on every round.
            pinned_contract=None,
            approval_policy=self.approval_policy,
            sandbox_mode=self.sandbox_mode,
            checks=(),
            capabilities=(),
            config_path=self.config_path,
            command_sandbox_image=self.sandbox_image,
        )

    @staticmethod
    def _startup_snapshot_policy(route_policy: Any) -> SnapshotPolicy:
        """Decide, once, which projection this host is entitled to use.

        Only a strict route with a required Indexer capability may classify a
        directory as control-plane runtime state, and the reason is specific:
        that route runs a mandatory Indexer gate before any model edit and
        again after the source-controlled checks, and records both in the route
        receipt. The classified tree is therefore still validated - by the lane
        that owns it - rather than merely unobserved.

        Any other configuration gets the default policy, which observes every
        non-VCS entry. That includes a service whose route was configured
        without an Indexer: the justification is the gate, not the intention.
        """

        indexer = getattr(route_policy, "indexer", None)
        if (
            route_policy is None
            or not getattr(route_policy, "strict", False)
            or indexer is None
            or not getattr(indexer, "required", False)
        ):
            return DEFAULT_SNAPSHOT_POLICY
        return SnapshotPolicy(
            runtime_state_names=(".flyto-index",),
            rationale="host-owned Indexer pre/post gates revalidate this tree",
        )

    @contextmanager
    def _admission_lock(self, workspace: str) -> Iterator[None]:
        """Serialize observe-then-claim for one workspace, and only that one.

        Deliberately a different lock from the per-round workspace lock. This
        one is held only across admission, so a submit never waits behind a
        model round, and a running round never waits behind a scan.
        """

        resolved = str(Path(workspace).resolve())
        with self._lock:
            local = self._admission_locks.setdefault(resolved, threading.Lock())
        digest = hashlib.sha256(resolved.encode()).hexdigest()
        directory = self.state_root / "locks" / "admission"
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = directory / (digest + ".lock")
        with local:
            handle = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                os.close(handle)

    def _observe_continuation(
        self,
        tenant_ref: str,
        request: CodingTaskRequest,
    ) -> Optional[ContinuationAuthority]:
        """Decide whether this submit may re-enter an existing backend session.

        Returns the claimed authority, or `None` when the request is an ordinary
        new job. Raises `ContinuationRefused` when a continuation was asked for
        and cannot be granted.

        Single ownership rests on the durable transition journal, not on this
        process holding a lock. The claim is a compare-and-swap against the
        journal tail, so a second OS process that loaded the same open
        authority - even one that never shared this interpreter or this
        in-process guard - finds the tail already moved and is refused. The
        state guard around this call is an optimisation, not the invariant.

        Ordering is a disclosure decision. Everything that could reveal whether
        somebody else's authority exists is answered first, with one code.
        Only after this caller has proven it holds an open authority in its own
        partition do the specific mismatches become visible - and by then they
        describe the caller's own workspace and contract, not a secret.
        """

        if not request.resume:
            return None
        session = str(request.thread_id or "")
        if not session:
            # `resume` with no thread id is the historical in-process rework
            # shape, which never crosses this path.
            return None
        if not is_continuable_session(session):
            # A host-minted provisional id, an oversized id, or an id outside
            # the accepted shape. Refused without any lookup at all, so it
            # cannot be used to probe for anything.
            raise ContinuationRefused(
                "the requested session cannot be continued",
                CONTINUATION_SESSION_INVALID,
            )
        authority = self._continuation.open_authority(tenant_ref, session)
        if authority is None:
            # Absent, another tenant's, already consumed, settled, superseded,
            # corrupt, truncated, replayed, or claimed by a racing process.
            # Exactly one answer for all of them - with one exception, and only
            # because it is not a secret: an authority this tenant stored under
            # the pre-pinning schema names no contract, so no continuation of it
            # could ever prove which verifier the stopped round ran. That is the
            # caller's own record, it is terminal, and no retry changes it, so
            # saying which of the two it is costs nothing and saves an operator
            # from retrying something that can never succeed.
            if self._continuation.is_unpinned_legacy(tenant_ref, session):
                raise ContinuationRefused(
                    "this continuation authority predates contract pinning",
                    CONTINUATION_CONTRACT_UNPINNED,
                )
            raise ContinuationRefused("no continuation authority is available")
        if not authority.contract_snapshot_sha256:
            # Defensive: a v3 record must bind a snapshot. Reaching here means a
            # record was written by something that skipped the binding, and an
            # unbound authority is never continued on a guess.
            raise ContinuationRefused(
                "this continuation authority binds no verification contract",
                CONTINUATION_CONTRACT_UNPINNED,
            )
        if authority.generation >= MAX_CONTINUATION_GENERATION:
            raise ContinuationRefused("no continuation authority is available")
        if authority.backend != self.implementation_backend:
            # This host would call a different provider, so "the same session"
            # is not a thing it could enter.
            raise ContinuationRefused(
                "the continuation authority names another implementation backend",
                CONTINUATION_BACKEND_MISMATCH,
            )
        if (
            authority.working_dir != request.working_dir
            or authority.workspace_sha256 != hashlib.sha256(
                request.working_dir.encode(),
            ).hexdigest()
        ):
            raise ContinuationRefused(
                "the continuation authority was taken in another workspace",
                CONTINUATION_WORKSPACE_MISMATCH,
            )
        if authority.snapshot_policy_sha256 != self.snapshot_policy.identity():
            # The authority was granted under a different projection of this
            # workspace. Continuing would mean re-proving a snapshot that never
            # observed the same things - which is what a policy change, an
            # added exclusion, or a strict-route authority replayed on a
            # non-strict service all look like.
            raise ContinuationRefused(
                "the workspace snapshot policy changed since the stop",
                CONTINUATION_POLICY_CHANGED,
            )
        # There is deliberately no comparison against the *current* contract
        # file here any more. The stopped round may have edited it, and that
        # edit is part of the exact revision the manifest check below demands be
        # unchanged - so requiring the file to still hash to the pre-edit
        # authority made the two invariants unsatisfiable together, and the only
        # ways out were to re-read (letting the edit authorize itself) or to
        # refuse forever. The authority instead binds the snapshot it was
        # granted under, and `_restore_pinned_contract` re-proves it by identity
        # after everything below has passed. Pinning is not weakened: it moved
        # from "the file still hashes to X" to "the contract still *is* X".
        #
        # A contract file edited *after* the stop is still refused, and audibly:
        # `.flyto/coding.yaml` is inside the workspace and is not one of the three
        # excluded version-control names, so it is part of the manifest below and
        # arrives as `CONTINUATION_REVISION_MISMATCH` rather than as a
        # contract-specific code. That is the honest answer - the tree moved -
        # and it is why the contract-specific codes are reserved for a pin that
        # cannot be recovered or re-proved.
        self._prove_stopped_tree(authority)
        return authority

    def _prove_stopped_tree(self, authority: ContinuationAuthority) -> None:
        """Prove the workspace is still, byte for byte, the tree that stopped.

        Factored out because it is asked twice, and must answer identically both
        times: once when the authority is observed, and again at the phase
        boundary immediately before that authority is consumed. One
        implementation means the seam re-proof can never drift into a weaker
        question than the observation it is re-proving.
        """

        try:
            revision = self._revision_digest(authority.working_dir, authority.files)
            # The whole tree, not only what the stopped round was credited
            # with. An audit probe added an unrelated `intruder.py` between
            # segments and was admitted, because a digest of the attributable
            # set cannot see a path nobody attributed. A resumed model would
            # then be reasoning about a workspace it had never seen.
            manifest = workspace_manifest_digest(
                authority.working_dir, self.snapshot_policy,
            )
        except (CodingServiceError, WorkspaceUnobservable, OSError):
            # Unreadable, replaced by a directory, symlink-swapped, escaped, a
            # special file, or past a manifest bound. None of that is the tree
            # that stopped, and none of it is describable exactly enough to
            # continue into.
            raise ContinuationRefused(
                "the workspace no longer matches the stopped revision",
                CONTINUATION_REVISION_MISMATCH,
            ) from None
        if (
            revision != authority.revision_sha256
            or manifest != authority.workspace_manifest_sha256
        ):
            # Modified, added, deleted, chmod'ed, re-typed or newly-appeared
            # bytes. All of them land here, and they land before the provider
            # is contacted.
            raise ContinuationRefused(
                "the workspace no longer matches the stopped revision",
                CONTINUATION_REVISION_MISMATCH,
            )

    def _claim_continuation(
        self,
        tenant_ref: str,
        job_id: str,
        observed: Optional[ContinuationAuthority],
    ) -> Optional[ContinuationAuthority]:
        """Consume exactly the authority that was observed, or refuse.

        The compare-and-swap is against the durable journal tail, not against
        what this process last read, so a second OS process that observed the
        same open authority while this one was scanning cannot also claim it.
        """

        if observed is None:
            return None
        try:
            return self._continuation.commit(
                observed, observed.claimed(job_id, time.time()),
            )
        except (ContinuationConflict, ContinuationCorrupt, OSError, ValueError):
            # Another process consumed this exact transition first, or the
            # journal stopped being readable. Both are "you may not have it",
            # and neither reveals which.
            raise ContinuationRefused(
                "no continuation authority is available",
            ) from None

    def _revert_continuation_claim(
        self,
        tenant_ref: str,
        granted: Optional[ContinuationAuthority],
        job_id: str,
    ) -> None:
        """Settle a claim whose job failed to come into existence.

        Never reopened. A consumed transition is consumed: the journal has
        already recorded it, and rewinding a hash chain would be the one edit
        that turns this mechanism back into something replayable. The operator
        loses a continuation they had not yet started, which is strictly safer
        than a generation two processes could both believe they own.
        """

        if granted is None:
            return
        try:
            stored = self._continuation.load(tenant_ref, granted.session_id)
            if (
                stored is None
                or stored.state != STATE_CLAIMED
                or stored.claimed_by_job_id != job_id
                or stored.sequence != granted.sequence
            ):
                # Somebody else's state now. Never rewrite it.
                return
            self._continuation.commit(stored, stored.settled(time.time()))
        except (ContinuationConflict, ContinuationCorrupt, OSError, ValueError):
            # The submit is already failing. A stuck `claimed` record is safe:
            # it refuses further continuation rather than granting one.
            pass

    def get(self, tenant_id: str, job_id: str) -> CodingJobReceipt:
        """Read a job only from the authenticated tenant namespace."""

        if not _JOB_ID.fullmatch(job_id):
            raise CodingJobNotFound("coding job does not exist")
        tenant_ref = self._tenant_ref(tenant_id)
        path = self._tenant_dir(tenant_ref, create=False) / "jobs" / (job_id + ".json")
        with self._state_guard():
            try:
                record = self._read_json(path)
            except FileNotFoundError as exc:
                raise CodingJobNotFound("coding job does not exist") from exc
            # Reading an audit-ready, accepted, or landable record on a strict
            # service revalidates its route evidence, so a job whose persisted
            # proof was removed or edited can never read back as landable.
            if (
                str(record.get("state")) in _ROUTE_EVIDENCE_STATES
                or record.get("landable") is True
            ):
                self._require_execution_authority(record)
            return self._public_receipt(tenant_ref, record)

    def audit(
        self,
        tenant_id: str,
        job_id: str,
        implementation_revision_sha256: str,
        verdict: CodingAuditVerdict,
        findings: Sequence[CodingAuditFinding],
    ) -> CodingJobReceipt:
        """Apply one authenticated audit decision to one exact revision.

        `accept` records landability as evidence only. Nothing in this service
        stages, commits, pushes, or publishes anything.
        """

        tenant_ref = self._tenant_ref(tenant_id)
        if not _JOB_ID.fullmatch(job_id):
            raise CodingJobNotFound("coding job does not exist")
        claimed = require_revision_sha256(
            implementation_revision_sha256, "implementation_revision_sha256",
        )
        verdict, findings = validate_audit_submission(verdict, findings)
        if not self.require_codex_audit:
            raise AuditNotEnabled("this coding service does not require an audit")
        path = self._tenant_dir(tenant_ref, create=False) / "jobs" / (job_id + ".json")
        with self._state_guard():
            if self._closed:
                raise CodingServiceError("coding service is closed")
            try:
                record = self._read_json(path)
            except FileNotFoundError as exc:
                raise CodingJobNotFound("coding job does not exist") from exc
            if str(record.get("state")) != CodingJobState.AWAITING_CODEX_AUDIT.value:
                raise AuditStateConflict("coding job is not awaiting an audit")
            # Belt and braces over the publication order. `awaiting_codex_audit`
            # is written in the same record update as this round's owner-closed
            # projection, so an open one here means the record was edited or an
            # older build wrote it. Either way an audit must not act: accepting
            # would land a revision whose work item is still dispatched, and
            # reworking would place a repair child under a parent that has not
            # settled, which is exactly how one audit forks the graph.
            settled = self._record_projection(record)
            if settled is not None and settled.status != MISSION_STATUS_CLOSED:
                raise AuditStateConflict(
                    "this round's mission work item has not settled yet",
                )
            stored = str(record.get("implementation_revision_sha256") or "")
            if stored != claimed:
                raise RevisionMismatch("audit does not bind the recorded implementation revision")
            # Prove this job still owns its worktree before reading the tree or
            # deciding anything. A verdict is a statement about files only this
            # job was allowed to touch, so both `accept` and `rework` require
            # the exact existing, fully bound tenant+job+workspace claim — a
            # foreign or unevaluable one stops either verdict, or a landable
            # record could coexist with someone else's ownership.
            #
            # An absent claim is `workspace_claim_unresolved`, never reacquired.
            # Only the original submit may create a claim from `free`. For a job
            # that is already awaiting audit, `free` proves nothing: during the
            # unobservable gap another job could have taken this worktree,
            # edited files outside this job's attributable set, settled, and
            # released. Reacquiring here would manufacture continuity that was
            # never established, and the revision recomputed below would not
            # detect it. The host spillway is abandon or repair.
            self._reassert_audit_claim(tenant_ref, record)
            # Recompute from the live workspace so an edit landed after the
            # implementation cannot inherit an earlier audit decision.
            if self._stored_revision(record) != stored:
                raise RevisionMismatch("the implementation revision changed since it was submitted")
            self._require_execution_authority(record)
            blockers = recorded_blockers(record)
            if blockers and verdict is CodingAuditVerdict.ACCEPT:
                # The host already observed why this revision is not landable.
                # An auditor may disagree about quality, but cannot overrule a
                # recorded host fact by accepting around it. Rework is still
                # available and is the only way these blockers clear. Refused
                # before the audit round is counted, so a mistaken accept costs
                # the job nothing.
                raise AuditBlockersUnresolved(
                    "this implementation still carries unresolved host blockers",
                )
            digest = audit_findings_sha256(findings)
            audit_count = int(record.get("audit_count", 0)) + 1
            if audit_count > MAX_AUDIT_ROUNDS:
                raise ReworkLimitReached("coding job exhausted its audit rounds")
            if verdict is CodingAuditVerdict.ACCEPT:
                self._update_record_locked(
                    path,
                    state=CodingJobState.CODEX_ACCEPTED.value,
                    audit_count=audit_count,
                    audit_findings_sha256=digest,
                    landable=True,
                    failure_code=None,
                )
                self._discard_resume(tenant_ref, job_id)
                # The accepted revision is what completes the mission, and only
                # on the main axis: a side item's accept closes that branch and
                # returns to its ancestor without ever claiming the objective
                # was reached.
                self._accept_mission(
                    tenant_ref, job_id, self._read_json(path), path,
                )
                return self._public_receipt(tenant_ref, self._read_json(path))
            return self._schedule_rework(
                path, tenant_ref, job_id, record, findings, digest, audit_count,
            )

    def abandon(self, tenant_id: str, job_id: str) -> CodingJobReceipt:
        """Fail one audit-ready job closed so its worktree can be reused.

        This is the operator's only release valve for a job whose auditor went
        away, and it is deliberately the weakest operation in the service: it
        can move `awaiting_codex_audit` to `failed` and nothing else. It never
        accepts, never lands, never stages or commits, and never lets a round
        skip an audit — abandoning is strictly worse for the caller than
        auditing, so it can never be used to route around the audit gate.
        """

        tenant_ref = self._tenant_ref(tenant_id)
        if not _JOB_ID.fullmatch(job_id):
            raise CodingJobNotFound("coding job does not exist")
        path = self._tenant_dir(tenant_ref, create=False) / "jobs" / (job_id + ".json")
        with self._state_guard():
            try:
                record = self._read_json(path)
            except FileNotFoundError as exc:
                raise CodingJobNotFound("coding job does not exist") from exc
            if str(record.get("state")) != CodingJobState.AWAITING_CODEX_AUDIT.value:
                raise AbandonStateConflict("only an audit-ready coding job can be abandoned")
            # The lease proves no live round is mid-flight. Abandoning under a
            # running implementer would release a worktree that is still being
            # written, which is exactly the race this change exists to close.
            if not self._acquire_job_lease(job_id):
                raise CodingServiceBusy("coding job is already being executed")
            try:
                self._update_record_locked(
                    path,
                    state=CodingJobState.FAILED.value,
                    failure_code="job_abandoned",
                    landable=False,
                )
            finally:
                self._release_job_lease(job_id)
            self._discard_resume(tenant_ref, job_id)
            self._release_workspace_claim(job_id, str(record.get("working_dir") or ""))
            # An abandoned predecessor leaves nothing to continue. Settling here
            # is what makes a later resume of its session refuse rather than
            # re-enter a conversation whose operator has walked away from it.
            self._settle_continuation(tenant_ref, job_id, path)
            return self._public_receipt(tenant_ref, self._read_json(path))

    def repair_workspace_claim(self, workspace: str) -> Dict[str, Any]:
        """Clear an unevaluable workspace claim on explicit host authority.

        This is the only way an `unresolved` claim is ever removed, and it is
        deliberately manual: the service refuses to guess, so a human decides
        that no audit is pending on that tree. A claim held by a live job is
        never touched here — that job must be audited or abandoned, which keeps
        the release valve from becoming a way to edit a tree mid-audit.
        """

        self._assert_workspace(workspace)
        with self._state_guard():
            status, owner_job_id = self._workspace_authority(workspace)
            if status == "held":
                raise WorkspaceBusy(
                    "a live coding job owns this workspace; audit or abandon it first",
                    owner_job_id,
                )
            if status == "free":
                return {"repaired": False, "status": status, "owner_job_id": owner_job_id}
            self._discard_path(self._workspace_claim_path(workspace))
            return {"repaired": True, "status": status, "owner_job_id": owner_job_id}

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # The state-root lease must outlive active worker writes. `wait=False`
        # cancels jobs that have not started but still drains running jobs.
        self._executor.shutdown(wait=True, cancel_futures=not wait)
        with self._lock:
            for job_id in tuple(self._job_leases):
                self._release_job_lease(job_id)
        try:
            with self._state_guard():
                # A graceful shutdown says so, keeping every fact it already
                # published. A crashed instance keeps its last `active` row,
                # which is why a reader also consults the pid and timestamp.
                self._close_status()
        except OSError:
            pass
        os.close(self._lock_fd)

    def _schedule_rework(
        self,
        path: Path,
        tenant_ref: str,
        job_id: str,
        record: Mapping[str, Any],
        findings: Sequence[CodingAuditFinding],
        digest: str,
        audit_count: int,
    ) -> CodingJobReceipt:
        """Queue one same-session repair round; the caller holds the lock."""

        rework_count = int(record.get("rework_count", 0)) + 1
        if rework_count > self.max_rework_rounds:
            # The ceiling is a settlement, not a bounce. Leaving the job in
            # `awaiting_codex_audit` kept its worktree claim held forever by a
            # job no verdict could ever move again, so the tree stayed locked
            # against every other job and the terminal condition was invisible
            # to a caller polling the receipt.
            self._settle_at_rework_limit(
                path, tenant_ref, job_id, record, audit_count, digest,
            )
            raise ReworkLimitReached("coding job exhausted its rework rounds")
        session = str(record.get("implementation_session_id") or "")
        # An audit may legitimately arrive at a different worker than the one
        # that implemented the job: every Codex frontend runs its own stdio
        # process against the same state root. The in-memory context is a fast
        # path; the durable envelope is what makes the loop closeable from any
        # live worker, and it can only ever continue this job's exact session.
        original = self._resume.get((tenant_ref, job_id))
        if original is None:
            original = self._read_resume_envelope(tenant_ref, job_id, record)
        if not session or original is None:
            raise ReworkNotResumable("the implementation session cannot be resumed")
        request = dataclasses.replace(
            self._with_startup_authority(dataclasses.replace(
                original,
                message=self._rework_message(original.message, findings),
                thread_id=session,
                resume=True,
            )),
            # The authority this job was granted at submit, restored from the
            # durable record. Recomputing it from the current file is exactly
            # the escalation this closes: a model that edited the contract in
            # round one would otherwise have the edit authorized in round two.
            authorized_config_sha256=str(record.get("authorized_config_sha256") or ""),
            # The contract itself, restored from the same durable record rather
            # than re-read. This is what lets a job whose implementation edited
            # `.flyto/coding.yaml` still reach rework: the round runs the checks
            # the job was admitted under, not the ones it just wrote for itself.
            pinned_contract=self._record_pinned_contract(record),
        )
        if len(self._pending) >= self.max_queued:
            raise CodingServiceBusy("coding job queue is full")
        # Claim the job before queueing. A concurrent audit then observes a
        # non-awaiting state and cannot schedule a duplicate rework round.
        if not self._acquire_job_lease(job_id):
            raise CodingServiceBusy("coding job is already being executed")
        try:
            # One repair child, under the same mission, in the repair lane, with
            # an explicit parent and a return edge that points at the main axis.
            # The operation key names this job and this repair round, so a retry
            # - here or in a second process reading the same record - reconciles
            # to the child that already exists instead of forking the graph.
            repair = self._submit_repair(tenant_ref, job_id, record, rework_count)
            self._write_round_envelope(
                tenant_ref, job_id, repair.work_item_id, rework=True,
                message=request.message,
            )
            self._update_record_locked(
                path,
                state=CodingJobState.REWORK_QUEUED.value,
                audit_count=audit_count,
                rework_count=rework_count,
                audit_findings_sha256=digest,
                landable=False,
                failure_code=None,
                mission=repair.projection.to_mapping(),
            )
        except BaseException:
            # The transition was refused — most often because this job can no
            # longer prove it owns the worktree. Hand back the lease so the job
            # stays exactly as auditable as it was, rather than looking busy to
            # every later caller.
            self._release_job_lease(job_id)
            raise
        try:
            future = self._executor.submit(self._pump_dispatch)
        except RuntimeError as exc:
            self._update_record_locked(
                path,
                state=CodingJobState.FAILED.value,
                failure_code="service_rework_not_scheduled",
            )
            self._release_job_lease(job_id)
            raise CodingServiceError("coding rework could not be scheduled") from exc
        self._pending.add(future)
        future.add_done_callback(self._forget_future)
        return self._public_receipt(tenant_ref, self._read_json(path))

    def _settle_at_rework_limit(
        self,
        path: Path,
        tenant_ref: str,
        job_id: str,
        record: Mapping[str, Any],
        audit_count: int,
        audit_findings_sha256: str,
    ) -> None:
        """Terminalize a job that has used every repair round it was given.

        Explicitly, and in one place, so the three things that must happen
        together cannot drift apart:

        * the record becomes `failed` under a stable code, which makes
          `job_terminal` true and makes any later audit or rework impossible
          because both require an awaiting state;
        * the resume authority is discarded, so the session can never be
          reopened by a worker that reads the envelope later;
        * the exact worktree claim is released, so the tree this job has
          finished with stops blocking every other job.

        The bounded historical evidence - session, cumulative files, revision,
        blockers - is deliberately *kept*. It is what an operator reads to see
        what was attempted and how far it got; none of it makes the job
        landable, because `landable` requires an accepted verdict this job can
        no longer receive.
        """

        self._update_record_locked(
            path,
            state=CodingJobState.FAILED.value,
            audit_count=audit_count,
            # The count and the evidence must describe the same verdict.
            # Persisting the incremented count beside the *previous* audit's
            # digest left a record that read as "N audits, and here is the
            # findings hash of audit N-1".
            audit_findings_sha256=audit_findings_sha256,
            failure_code=REWORK_LIMIT_FAILURE_CODE,
            landable=False,
        )
        # Both are idempotent by construction: discarding an absent envelope
        # and releasing a claim this job no longer holds are both no-ops, so a
        # retried settlement cannot release somebody else's later claim.
        self._discard_resume(tenant_ref, job_id)
        self._release_workspace_claim(job_id, str(record.get("working_dir") or ""))

    @staticmethod
    def _rework_message(original: str, findings: Sequence[CodingAuditFinding]) -> str:
        """Render bounded, deterministic feedback from typed findings only."""

        lines = ["Audit verdict: rework. Resolve every finding below in this same thread."]
        for index, finding in enumerate(sorted(
            findings, key=lambda item: (item.code, item.evidence_ref),
        ), start=1):
            reference = " (ref: {})".format(finding.evidence_ref) if finding.evidence_ref else ""
            lines.append("{}. [{}] {}: {}{}".format(
                index, finding.severity.value, finding.code, finding.message, reference,
            ))
        feedback = "\n".join(lines)[:MAX_REWORK_FEEDBACK_CHARS]
        remaining = max(0, MAX_REWORK_MESSAGE_CHARS - len(feedback) - 32)
        return "{}\n\nOriginal task:\n{}".format(feedback, original[:remaining])

    def _run_job(
        self,
        tenant_ref: str,
        job_id: str,
        request: CodingTaskRequest,
        *,
        rework: bool = False,
        work: Optional[DispatchedWork] = None,
    ) -> None:
        path = self._tenant_dir(tenant_ref) / "jobs" / (job_id + ".json")
        workspace_lock = self._workspace_lock(request.working_dir)
        progress = _RoundProgress(
            on_start=lambda: self._mark_implementer_started(path),
        )
        # Liveness while the implementer runs, and again at every phase
        # boundary. Observability only: no lease moves, no fencing token is
        # burned, and nothing about this dispatch's authority depends on a
        # heartbeat arriving - which is exactly why no worker is ever stolen
        # from because one stopped.
        pulse = MissionHeartbeat(work.handle) if work is not None else None
        # The round's mission closure, performed exactly once and *published
        # with* the state it justifies. Every terminal or audit-ready record
        # this round writes folds the closed item's projection into the same
        # write, so no reader and no auditor can ever observe a settled job
        # whose work item this process still holds open.
        settle = _RoundSettlement(self, work, tenant_ref, job_id)
        with workspace_lock:
            try:
                self._update_record(path, state=(
                    CodingJobState.REWORK_RUNNING if rework else CodingJobState.RUNNING
                ).value)
                if pulse is not None:
                    # The receipt's mission projection follows the work item's
                    # real lifecycle, so a reader can tell "queued behind a busy
                    # worktree" from "running" without inferring either.
                    self._advance_projection(path, status=MISSION_STATUS_DISPATCHED)
                    pulse.beat()
                    pulse.start()
                store = ThreadStore(str(self._tenant_dir(tenant_ref) / "threads"))
                # Armed once, for every lane this round might take. The strict
                # route, the emergency lane and the unrouted legacy call all
                # share this store, so none of them can run a provider whose
                # session the host would not write down.
                _arm_session_binding(store, self._session_binder(
                    path, tenant_ref, job_id, request.working_dir,
                ))
                result, route, authority = asyncio.run(
                    self._run_round(
                        store, request, job_id, path, progress, rework=rework,
                    ),
                )
                if pulse is not None:
                    pulse.beat()
                self._record_outcome(
                    path, tenant_ref, job_id, request, result, store, rework, route,
                    progress=progress, authority=authority, settle=settle,
                )
            except CodingServiceError as exc:
                self._fail_job(
                    path, tenant_ref, job_id, exc.code, exc, progress, settle=settle,
                )
            except BaseException as exc:  # noqa: BLE001 - a worker never leaks
                # Every exit from a worker must leave a terminal record. An
                # unobserved future exception would strand the job `running`
                # forever, which is exactly what a fail-closed service must
                # never do.
                self._fail_job(
                    path, tenant_ref, job_id, "service_execution_failed", exc,
                    progress, settle=settle,
                )
                if not isinstance(exc, Exception):
                    raise
            finally:
                # Stopping the pulse first keeps a heartbeat from racing the
                # closure. The settlement below is a *backstop* only: every
                # publishing path above has already taken it, and taking it
                # twice is a no-op. It exists for the paths that never reached a
                # publish at all - an unwritable record, a killed transition -
                # where leaving the item dispatched would strand the queue.
                if pulse is not None:
                    pulse.stop()
                if not settle.settled:
                    changes = settle(state="round_unpublished")
                    if changes:
                        try:
                            self._update_record(path, **changes)
                        except (OSError, ValueError):
                            pass

    # -- mission lane ---------------------------------------------------

    def _admit_mission(
        self, tenant_ref: str, job_id: str, request: CodingTaskRequest,
    ) -> "MissionAdmission":
        """Create or validate this job's mission and place its one work item."""

        try:
            return self._mission.admit(
                tenant_ref=tenant_ref,
                job_id=job_id,
                workspace_sha256=self._workspace_digest(request.working_dir),
                envelope=request.mission,
                message=request.message,
            )
        except MissionRouteError as exc:
            raise MissionRouteRefused(exc) from exc

    def _submit_repair(
        self,
        tenant_ref: str,
        job_id: str,
        record: Mapping[str, Any],
        rework_count: int,
    ) -> "MissionAdmission":
        """Place one repair child for this job's next round."""

        projection = self._record_projection(record)
        if projection is None:
            raise MissionRouteRefused(
                MissionAuthorityRefused("this job has no mission work item to repair"),
            )
        try:
            return self._mission.submit_repair(
                tenant_ref=tenant_ref,
                job_id=job_id,
                workspace_sha256=self._workspace_digest(
                    str(record.get("working_dir") or ""),
                ),
                projection=projection,
                round_index=int(rework_count),
            )
        except MissionRouteError as exc:
            raise MissionRouteRefused(exc) from exc

    @staticmethod
    def _record_projection(
        record: Mapping[str, Any],
    ) -> Optional[CodingMissionProjection]:
        """Rebuild a job's mission projection, revalidating it on the way out.

        A stored mapping goes back through the closed decoder, so a record
        edited in place yields `None` rather than a weakened projection - and
        `None` fails closed everywhere it is consumed.
        """

        stored = record.get("mission")
        if not isinstance(stored, Mapping):
            return None
        try:
            return CodingMissionProjection.from_mapping(stored)
        except (ValueError, TypeError):
            return None

    def _pump_dispatch(self) -> None:
        """Run exactly one store-selected work item, whichever job owns it.

        The handoff this loop exists for is the cross-instance one. Two jobs on
        one worktree cannot run at the same time - the mission store holds that
        worktree as an exclusive resource - so the second job's pump finds every
        candidate conflicted and must not simply hand its worker back: the
        worktree's claim is released in *another* process, and no in-process
        callback will ever fire to say so. Waiting on the store's own two
        numbers is what makes the release observable across instances.

        It is bounded in three independent ways, so it can never poll forever:

        * it stops the moment the store reports no item in `ready`;
        * it stops once nothing is running anywhere and the queue has stopped
          moving, because then the conflict is not going to resolve itself;
        * it stops at a hard deadline regardless.

        The wait also backs off, because every dispatch attempt rewrites the
        durable store. A tight retry against a busy worktree would cost more
        than the round it is waiting for.
        """

        deadline = time.monotonic() + _PUMP_WAIT_SECONDS
        delay = _PUMP_POLL_SECONDS
        foreign = 0
        while not self._closed:
            outcome = self._dispatch_once()
            if outcome == _DISPATCH_RAN:
                # This process just released a resource claim. Anything that was
                # waiting on it inside *this* instance is woken immediately
                # rather than after a poll tick.
                self._prime_pump()
                return
            # Whether this instance still holds an admission lease of its own.
            # It is the difference between "somebody else's queue is busy" and
            # "my own job has not had its turn yet", and only the second is a
            # reason for this pump to keep waiting.
            owed = self._owns_queued_work()
            if outcome == _DISPATCH_FOREIGN:
                # The store offered work whose job another process has leased.
                # It was requeued untouched - never stolen, never run twice -
                # and that process has its own pump for it. Hammering it would
                # burn a fencing token per attempt for nothing, so back off to
                # the ceiling; and when this instance owes no round of its own,
                # give the worker back rather than shadowing another process.
                delay = _PUMP_POLL_CEILING
                if not owed:
                    foreign += 1
                    if foreign >= _PUMP_MAX_FOREIGN:
                        return
            ready, dispatched = self._mission.queue_state()
            if not ready or time.monotonic() >= deadline:
                return
            if not dispatched and not owed:
                # Ready work nobody is running that this pump could not take,
                # and nothing of this instance's own waiting behind it. Nothing
                # here is going to release anything, so waiting would be waiting
                # on an event that cannot happen.
                return
            time.sleep(delay)
            delay = min(_PUMP_POLL_CEILING, delay * 2)

    def _owns_queued_work(self) -> bool:
        """Whether this instance still holds a lease for a round it admitted.

        Admission takes the job lease and the round that runs the job hands it
        back, so a non-empty lease table means this instance has work of its own
        that has not had its turn. That is the one thing that justifies a pump
        waiting behind another process's running round: its own job is next.
        """

        with self._lock:
            return bool(self._job_leases)

    def _prime_pump(self) -> None:
        """Queue one more pump when the shared queue still holds ready work."""

        with self._lock:
            if self._closed or len(self._pending) >= self.max_queued:
                return
            if not self._mission.ready_work():
                return
            try:
                future = self._executor.submit(self._pump_dispatch)
            except RuntimeError:
                return
            self._pending.add(future)
        future.add_done_callback(self._forget_future)

    def _dispatch_once(self) -> str:
        """Take the next work item the store chose and run its owning job.

        Answers which of three things happened, because a pump treats them
        differently: this call ran or accounted for an item, the store had
        nothing to give, or the item belongs to a round another process holds.
        """

        try:
            with self._mission.dispatch() as work:
                if work is None:
                    return _DISPATCH_IDLE
                return self._run_dispatched(work)
        except MissionRouteError:
            # A refusal from the kernel is never a reason to spin: the pump
            # gives its worker back and the next submit will try again.
            return _DISPATCH_IDLE

    def _run_dispatched(self, work: "DispatchedWork") -> str:
        """Resolve one dispatched work item back to its private owner and run it.

        Three refusals happen before anything is executed, and each of them is a
        case where running would be worse than not running:

        * the owning record cannot be read, or is already terminal. The item is
          closed with full accounting rather than left in the queue, which is
          what makes a restart's failed-closed jobs explicitly accounted for.
        * the round envelope does not name this work item. Something else placed
          it, so this process has no proof of what it was for.
        * the job lease is held elsewhere. Another process owns this round; the
          item is requeued untouched, never stolen and never duplicated.
        """

        tenant_ref, job_id = work.tenant_ref, work.job_id
        if not _JOB_ID.fullmatch(job_id) or not _TENANT_REF.fullmatch(tenant_ref):
            self._account_unrunnable(work, "mission_coordinates_unresolvable")
            return _DISPATCH_RAN
        path = self._tenant_dir(tenant_ref, create=False) / "jobs" / (job_id + ".json")
        try:
            record = self._read_json(path)
        except (OSError, ValueError):
            self._account_unrunnable(work, "job_record_unreadable")
            return _DISPATCH_RAN
        state = str(record.get("state") or "")
        if state in _TERMINAL_STATE_VALUES or state not in (
            CodingJobState.QUEUED.value, CodingJobState.REWORK_QUEUED.value,
        ):
            self._account_unrunnable(work, "job_not_runnable")
            return _DISPATCH_RAN
        round_envelope = self._read_round_envelope(tenant_ref, work.work_item_id)
        if round_envelope is None or str(round_envelope.get("job_id") or "") != job_id:
            self._account_unrunnable(work, "round_envelope_unbound")
            return _DISPATCH_RAN
        rework = bool(round_envelope.get("rework"))
        request = self._reconstruct_request(tenant_ref, job_id, record, round_envelope)
        if request is None:
            self._account_unrunnable(work, "request_unreconstructable")
            return _DISPATCH_RAN
        if not self._claim_round(job_id):
            # Held by another process. Leaving the `with` block without closing
            # requeues the item, so the owner keeps its work and nothing is
            # duplicated behind its back.
            return _DISPATCH_FOREIGN
        try:
            self._run_job(tenant_ref, job_id, request, rework=rework, work=work)
        finally:
            self._release_round(job_id)
        return _DISPATCH_RAN

    def _reconstruct_request(
        self,
        tenant_ref: str,
        job_id: str,
        record: Mapping[str, Any],
        round_envelope: Mapping[str, Any],
    ) -> Optional[CodingTaskRequest]:
        """Rebuild the private request one dispatched round must execute.

        Host authority is never replayed from a durable envelope. The bounded
        public request comes back from this job's own resume envelope, the
        startup authority is re-imposed by *this* service, and the pinned
        contract and its digest are restored from the job record - exactly as a
        rework round already did, so a second process cannot widen what an
        implementer is authorized to do.
        """

        request_sha256 = str(record.get("request_sha256") or "")
        original = self._resume.get((tenant_ref, job_id))
        if original is None:
            original = self._load_resume_request(tenant_ref, job_id, request_sha256)
        if original is None:
            return None
        rework = bool(round_envelope.get("rework"))
        if rework:
            session = str(record.get("implementation_session_id") or "")
            message = str(round_envelope.get("message") or "")
            if not session or not message:
                return None
            original = dataclasses.replace(
                original, message=message, thread_id=session, resume=True,
            )
        try:
            return dataclasses.replace(
                self._with_startup_authority(original),
                authorized_config_sha256=str(
                    record.get("authorized_config_sha256") or "",
                ),
                pinned_contract=self._record_pinned_contract(record),
            )
        except (ValueError, TypeError):
            return None

    def _account_unrunnable(self, work: "DispatchedWork", reason: str) -> None:
        """Close one work item nobody can run, with the whole accounting.

        Accounted rather than silently dropped, and never fixed: work that never
        ran did not deliver anything. This is the path an interrupted job's item
        takes after a restart failed it closed, which is what keeps a reclaimed
        item from circling the queue forever.

        A job that is simply no longer runnable - already terminal, already
        settled - is `deferred`: the work was not refused, it was overtaken, and
        a fresh submission is the way back. Anything this host could not even
        resolve is `blocked`, because something has to be repaired before that
        item could ever run. Both carry the whole accounting either way.
        """

        deferred = reason == "job_not_runnable"
        try:
            self._mission.close_accounted(
                work,
                tenant_ref=work.tenant_ref,
                job_id=work.job_id,
                mission_id=work.mission_id,
                work_item_id=work.work_item_id,
                disposition=(
                    DISPOSITION_DEFERRED if deferred else DISPOSITION_BLOCKED
                ),
                rationale=(
                    "the host could not run this work item: {}".format(reason)
                ),
                risk=(
                    "the mission's objective is not reached and this workspace's "
                    "next round has to be submitted again"
                ),
                evidence_refs=("reason-{}".format(reason), "job-{}".format(work.job_id)),
            )
        except MissionRouteError:
            return

    def _close_round_item(
        self,
        work: "DispatchedWork",
        tenant_ref: str,
        job_id: str,
        *,
        revision: str,
        files: Sequence[str],
        state: str,
        failure_code: str,
    ) -> Dict[str, Any]:
        """Owner-close this round's work item and return the record change.

        Nothing is written here. The caller folds the returned mapping into the
        *same* `_update_record` that publishes the round's terminal or
        audit-ready state, so the two become durable together: a poller that can
        see `failed` or `awaiting_codex_audit` already sees the closed item's
        projection, and an auditor that can act on that state is acting after
        this process gave up its execution authority over the item.

        `fixed` is reachable only when the host itself has an attributable,
        auditable revision to point at - a digest it computed and the exact file
        set it computed it over - and both are passed in by the round rather
        than re-read, because at this point they are not on disk yet. Everything
        else is a pre-audit terminal failure and closes `blocked` with the whole
        accounting, because a round that stopped without producing an auditable
        revision is precisely the outcome a silent close would hide.
        """

        try:
            if revision and files:
                self._mission.close_fixed(
                    work,
                    tenant_ref=tenant_ref,
                    job_id=job_id,
                    mission_id=work.mission_id,
                    work_item_id=work.work_item_id,
                )
                disposition = MISSION_DISPOSITION_FIXED
            else:
                self._mission.close_accounted(
                    work,
                    tenant_ref=tenant_ref,
                    job_id=job_id,
                    mission_id=work.mission_id,
                    work_item_id=work.work_item_id,
                    disposition=DISPOSITION_BLOCKED,
                    rationale=(
                        "this round produced no attributable auditable revision;"
                        " the job settled as {}".format(state or "unknown")
                    ),
                    risk=(
                        "the mission's main axis is not proven and no audit can "
                        "accept this workspace's current state"
                    ),
                    evidence_refs=(
                        "job-{}".format(job_id),
                        "state-{}".format(state or "unknown"),
                        "code-{}".format(failure_code or "none"),
                    ),
                )
                disposition = DISPOSITION_BLOCKED
        except MissionRouteError:
            # The item could not be closed - a stale fence, a corrupt store. The
            # projection is deliberately left describing the dispatch, because
            # advancing it would publish a closure that did not happen.
            return {}
        return self._projection_change(
            work, status=MISSION_STATUS_CLOSED, disposition=disposition,
        )

    def _projection_change(
        self,
        work: "DispatchedWork",
        *,
        status: str,
        disposition: str = "",
    ) -> Dict[str, Any]:
        """Build the record change that moves *this round's* projection.

        The identity check is the whole point. A repair child may already have
        replaced the record's projection by the time a late closure lands, and
        advancing it then would publish this round's disposition onto the next
        round's work item. When the record has moved on, this round's closure
        has nothing to say about it and says nothing.
        """

        try:
            record = self._read_json(
                self._tenant_dir(work.tenant_ref, create=False)
                / "jobs" / (work.job_id + ".json"),
            )
        except (OSError, ValueError):
            return {}
        stored = record.get("mission")
        if not isinstance(stored, Mapping):
            return {}
        if str(stored.get("work_item_id") or "") != work.work_item_id:
            return {}
        try:
            return {
                "mission": CodingMissionRuntime.advance(
                    stored, status=status, disposition=disposition,
                ),
            }
        except (ValueError, TypeError):
            return {}

    def _advance_projection(
        self,
        path: Path,
        *,
        status: str,
        disposition: str = "",
        mission_status: Optional[str] = None,
        returned_to_main_axis: Optional[bool] = None,
    ) -> None:
        """Move the record's stored projection, or leave it exactly as it was."""

        try:
            record = self._read_json(path)
        except (OSError, ValueError):
            return
        stored = record.get("mission")
        if not isinstance(stored, Mapping):
            return
        try:
            advanced = CodingMissionRuntime.advance(
                stored,
                status=status,
                disposition=disposition,
                mission_status=mission_status,
                returned_to_main_axis=returned_to_main_axis,
            )
        except (ValueError, TypeError):
            return
        try:
            self._update_record(path, mission=advanced)
        except (OSError, ValueError):
            return

    def _accept_mission(
        self, tenant_ref: str, job_id: str, record: Mapping[str, Any], path: Path,
    ) -> None:
        """Apply one accepted audit to this job's mission.

        A root accept completes the mission - but only once every work item of
        that mission is closed, and only with evidence for exactly the criteria
        the mission itself declared. A side item's accept closes nothing further
        and completes nothing: it records that control returned to the ancestor
        the item named, and the main axis keeps whatever state it had.
        """

        projection = self._record_projection(record)
        if projection is None:
            return
        evidence = {
            CRITERION_REVISION: str(
                record.get("implementation_revision_sha256") or "",
            ),
            CRITERION_CHECKS: str(record.get("authorized_config_sha256") or ""),
            CRITERION_AUDIT: str(record.get("audit_findings_sha256") or ""),
        }
        try:
            mission = self._mission.complete(
                tenant_ref=tenant_ref,
                job_id=job_id,
                mission_id=projection.mission_id,
                work_item_id=projection.work_item_id,
                evidence={key: value for key, value in evidence.items() if value},
            )
        except MissionRouteError:
            return
        self._advance_projection(
            path,
            status=MISSION_STATUS_CLOSED,
            disposition=MISSION_DISPOSITION_FIXED,
            mission_status=(
                MISSION_COMPLETED if mission is not None else MISSION_OPEN
            ),
            returned_to_main_axis=None if projection.is_root else True,
        )

    # -- mission observability -----------------------------------------

    def mission_fleet(self, *, limit: int = 50) -> Dict[str, Any]:
        """A bounded, snapshot-only, secret-free view of every mission here.

        Observable, never actionable, and never conversational. Nothing in this
        payload is accepted as authority by any operation on this service, and
        nothing in it is ever placed in a request, a prompt or a thread: another
        Codex job is something an operator can see, not something a model can be
        told about.
        """

        return self._mission.fleet(limit=limit)

    def mission_context(self, tenant_id: str, job_id: str) -> Dict[str, Any]:
        """Full mission context for one job, for its owning tenant only."""

        tenant_ref = self._tenant_ref(tenant_id)
        if not _JOB_ID.fullmatch(job_id):
            raise CodingJobNotFound("coding job does not exist")
        path = self._tenant_dir(tenant_ref, create=False) / "jobs" / (job_id + ".json")
        try:
            record = self._read_json(path)
        except (OSError, ValueError) as exc:
            raise CodingJobNotFound("coding job does not exist") from exc
        projection = self._record_projection(record)
        if projection is None:
            raise CodingJobNotFound("coding job has no mission")
        try:
            return self._mission.context(
                tenant_ref=tenant_ref,
                job_id=job_id,
                work_item_id=projection.work_item_id,
            )
        except MissionRouteError as exc:
            raise MissionRouteRefused(exc) from exc

    # -- round envelopes ------------------------------------------------

    def _round_path(self, tenant_ref: str, work_item_id: str) -> Path:
        if not _WORK_ITEM_ID.fullmatch(str(work_item_id)):
            raise CodingServiceError("coding round envelope id is invalid")
        return (
            self.state_root / "tenants" / tenant_ref / "rounds"
            / (work_item_id + ".json")
        )

    def _write_round_envelope(
        self,
        tenant_ref: str,
        job_id: str,
        work_item_id: str,
        *,
        rework: bool,
        message: str = "",
    ) -> None:
        """Bind one placed work item to the exact round it stands for.

        Private tenant state, 0600, and never projected anywhere. A first round
        replays this job's resume envelope, so it stores no message at all; a
        repair round stores the bounded audit feedback the host rendered, which
        is the one thing a second process could not otherwise reconstruct.
        """

        self._write_json(self._round_path(tenant_ref, work_item_id), {
            "envelope_version": ROUND_ENVELOPE_VERSION,
            "work_item_id": work_item_id,
            "job_id": job_id,
            "rework": bool(rework),
            "message": str(message or "")[:MAX_REWORK_MESSAGE_CHARS] if rework else "",
            "created_at": time.time(),
        })

    def _read_round_envelope(
        self, tenant_ref: str, work_item_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Read one round envelope, or refuse a file that is not exactly one."""

        try:
            envelope = self._read_json(self._round_path(tenant_ref, work_item_id))
        except (OSError, ValueError, CodingServiceError):
            return None
        if (
            envelope.get("envelope_version") != ROUND_ENVELOPE_VERSION
            or set(envelope) - _ROUND_ENVELOPE_FIELDS
            or str(envelope.get("work_item_id") or "") != work_item_id
            or not isinstance(envelope.get("rework"), bool)
        ):
            return None
        return envelope

    # -- round leases ---------------------------------------------------

    def _claim_round(self, job_id: str) -> bool:
        """Own this job's round, inheriting a lease this process already holds.

        Admission takes the lease so nothing else can start the job in between;
        the round that actually runs it inherits that lease rather than
        deadlocking against it, and hands it back when the round ends. A lease
        held by *another* process is never taken, which is what stops two
        services from running one job.
        """

        with self._lock:
            if job_id in self._job_leases:
                return True
            return self._acquire_job_lease(job_id)

    def _release_round(self, job_id: str) -> None:
        with self._lock:
            self._release_job_lease(job_id)

    async def _run_round(
        self,
        store: ThreadStore,
        request: CodingTaskRequest,
        job_id: str,
        path: Path,
        progress: _RoundProgress,
        *,
        rework: bool = False,
    ) -> "tuple[CodingTaskResult, Optional[CodingRouteReceipt], Optional[EmergencyAuthorityReceipt]]":
        """Run the strict route, then the emergency lane only if it is earned.

        A job already bound to emergency authority stays there: its rework
        continues in the same session on the same authority instead of
        silently returning to a route that is still broken.
        """

        record = self._read_json(path)
        # The submitted request digest is a job-lifetime fact. A rework round
        # rewrites the message, so the authority must bind the original digest
        # the record already carries, never a per-round recomputation.
        request_sha256 = str(record.get("request_sha256") or "")
        bound = self._bound_emergency_authority(record, job_id, rework=rework)
        if bound is not None:
            result, authority = await self._emergency_round(
                store, request, job_id, request_sha256, path, progress,
                bound.trigger(), mode="emergency_rework",
            )
            return result, None, authority

        rework_binding = None
        if rework:
            # A rework snapshots only this round, but the audit binds the whole
            # job's attributable revision.  Keep the durable record available
            # to the implementer boundary so a clean no-op verification can be
            # promoted only after that cumulative binding is re-proved.
            rework_binding = (record, path.parent.parent.name, job_id)
        result, route = await self._implement(
            store, request, progress, rework_binding=rework_binding,
            job_binding=(path, path.parent.parent.name, job_id),
        )
        trigger = self._overflow_trigger(route, result, progress, path)
        if trigger is None:
            return result, route, None
        if not self._breaker.record_infrastructure_failure():
            # The breaker has not tripped yet. This round stays fail-closed,
            # exactly as it would without any emergency authority configured.
            return result, route, None
        emergency, authority = await self._emergency_round(
            store, request, job_id, request_sha256, path, progress, trigger,
            mode="emergency",
        )
        return emergency, None, authority

    def _mark_implementer_started(self, path: Path) -> None:
        """Durably record that the implementer is about to be invoked.

        This runs before the call, not after it returns, so a crashed or killed
        process still leaves evidence that a model round was in flight.

        On the audited public route the record also names the implementer at
        this point. A legacy audit-disabled library service keeps its
        documented shape, where `implementation_backend` is bound only by an
        audit; its configured backend is still published in runtime status.
        """

        changes: Dict[str, Any] = {"implementer_started": True}
        if self.require_codex_audit:
            changes["implementation_backend"] = self.implementation_backend
        self._update_record(path, **changes)

    def _session_binder(
        self, path: Path, tenant_ref: str, job_id: str, workspace: str,
    ) -> Callable[[str], None]:
        """Build this round's durable session-binding callback."""

        def bind(session_id: str) -> None:
            self._bind_provider_session(path, tenant_ref, job_id, workspace, session_id)

        return bind

    def _bind_provider_session(
        self,
        path: Path,
        tenant_ref: str,
        job_id: str,
        workspace: str,
        session_id: str,
    ) -> None:
        """Persist the session the provider just established, or refuse the round.

        Three bindings, and any of them failing stops the round before the model
        can do anything the host would then be unable to attribute:

        * the id must be a real backend session. A host-minted `host-`/`route-`
          provisional id is a placeholder this service invented, and promoting
          one into an implementation session would manufacture a continuable
          identity out of nothing;
        * this worker must still hold the job's execution lease. Without it
          another process owns the round, and a session written from here would
          name a conversation this job no longer controls;
        * the record must not already name a *different* session. A backend that
          moves a bound round into another conversation is a boundary violation,
          not a reconnect.

        Repeating the identical id is a no-op, so an SDK that re-announces its
        init - or a resumed round that lands in exactly the session it asked for
        - costs nothing and changes nothing.

        The write itself goes through `_update_record_locked`, which reasserts
        the worktree claim for a claim-owned state. So a job that has lost its
        exclusive hold on the tree cannot bind a session to it either.
        """

        if not is_continuable_session(session_id):
            raise SessionBindingFailed("the provider session identity is unusable")
        with self._state_guard():
            if job_id not in self._job_leases:
                raise SessionBindingFailed(
                    "this worker no longer holds the coding job execution lease",
                )
            record = self._read_json(path)
            recorded = str(record.get("implementation_session_id") or "")
            if recorded == session_id:
                return
            if recorded:
                raise SessionBindingFailed(
                    "the provider established a different implementation session",
                )
            changes: Dict[str, Any] = {
                "implementation_session_id": session_id,
                # A provider that has named a session was necessarily entered.
                "implementer_started": True,
            }
            if self.require_codex_audit:
                changes["implementation_backend"] = self.implementation_backend
            self._update_record_locked(path, **changes)

    @staticmethod
    def _plan_authority_digest(
        job_id: str, request_sha256: str, workspace_sha256: str, contract: Mapping[str, Any],
    ) -> str:
        payload = json.dumps(
            {
                "version": PLAN_AUTHORITY_VERSION,
                "job_id": job_id,
                "request_sha256": request_sha256,
                "workspace_sha256": workspace_sha256,
                "contract": contract,
            },
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False, default=str,
        )
        digest = hashlib.sha256()
        digest.update(_PLAN_AUTHORITY_DOMAIN)
        digest.update(payload.encode("utf-8"))
        return digest.hexdigest()

    def _persist_plan_authority(
        self, path: Path, record: Mapping[str, Any], contract: Mapping[str, Any],
    ) -> None:
        """Record which contract this round was authorized against.

        Called only after a genuine pre-lane success, so a lane that refused
        never leaves behind authority a later round could amend. Bounded and
        integrity-protected: a contract this host cannot carry exactly is not
        carried at all, and a later round then fails closed rather than
        amending something nobody proved.
        """

        if not isinstance(contract, Mapping) or not contract:
            # A pre-lane that "succeeded" without returning a contract has not
            # proven what this round is authorized to change, and a later round
            # would have nothing exact to amend. Refuse now rather than defer
            # the defect to whichever rework discovers it.
            raise PlanAuthorityUnprovable("plan_authority_unsealable")
        job_id = str(record.get("job_id") or "")
        request_sha256 = str(record.get("request_sha256") or "")
        workspace_sha256 = str(record.get("workspace_sha256") or "")
        try:
            body = json.dumps(
                contract, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False, default=str,
            )
        except (TypeError, ValueError):
            # Not canonically serializable, so not something this host can
            # carry forward byte-exactly across a restart.
            raise PlanAuthorityUnprovable("plan_authority_unsealable") from None
        if len(body.encode("utf-8")) > MAX_PLAN_AUTHORITY_BYTES:
            raise PlanAuthorityUnprovable("plan_authority_unsealable")
        stored = json.loads(body)
        self._update_record(path, indexer_plan_authority={
            "version": PLAN_AUTHORITY_VERSION,
            "job_id": job_id,
            "request_sha256": request_sha256,
            "workspace_sha256": workspace_sha256,
            "contract": stored,
            "contract_sha256": self._plan_authority_digest(
                job_id, request_sha256, workspace_sha256, stored,
            ),
            "recorded_at": time.time(),
        })

    def _load_plan_authority(
        self, record: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Return the exact parent contract this job may amend, or `None`.

        Every binding is re-proven from the durable record: schema version, the
        job it belongs to, the root request it was planned for, the workspace it
        was planned in, and its own content digest. A replayed envelope from
        another job, a stale one from a different request, or a hand-edited
        contract all fail here - before the caller can ask an Indexer to amend
        something nobody authorized.
        """

        envelope = record.get("indexer_plan_authority")
        if not isinstance(envelope, Mapping):
            return None
        if str(envelope.get("version") or "") != PLAN_AUTHORITY_VERSION:
            return None
        contract = envelope.get("contract")
        if not isinstance(contract, Mapping) or not contract:
            return None
        job_id = str(record.get("job_id") or "")
        request_sha256 = str(record.get("request_sha256") or "")
        workspace_sha256 = str(record.get("workspace_sha256") or "")
        if (
            str(envelope.get("job_id") or "") != job_id
            or str(envelope.get("request_sha256") or "") != request_sha256
            or str(envelope.get("workspace_sha256") or "") != workspace_sha256
        ):
            return None
        expected = self._plan_authority_digest(
            job_id, request_sha256, workspace_sha256, contract,
        )
        if str(envelope.get("contract_sha256") or "") != expected:
            return None
        return dict(contract)

    def _prove_prior_scope(
        self,
        record: Mapping[str, Any],
        request: CodingTaskRequest,
        tenant_ref: str,
        job_id: str,
    ) -> Tuple[str, ...]:
        """Re-prove what this job already owns, *before* the implementer edits.

        Timing is the whole point. The stored revision describes the tree as it
        was when the previous round closed, so it can only be re-proven while
        that is still true. Asking the same question after a resumed model has
        edited the workspace would compare the old digest against new bytes and
        refuse every legitimate rework.

        Every binding a cumulative audit will later rely on is asserted here:
        a bounded, canonical, duplicate-free prior set; a real backend session
        the record already names; a durable resume envelope sealed to that
        session; this tenant and job's live worktree claim; the same workspace
        path; and the exact bytes the recorded revision digest describes.
        Anything unproven raises before a provider is contacted.
        """

        prior_raw = record.get("implementation_files")
        if prior_raw in (None, []):
            return ()
        if not isinstance(prior_raw, (list, tuple)):
            raise PlanAuthorityUnprovable("cumulative_scope_unproven")
        prior = [
            str(item) for item in prior_raw
            if isinstance(item, str) and not isinstance(item, bool)
        ]
        if len(prior) != len(prior_raw) or len(set(prior)) != len(prior):
            raise PlanAuthorityUnprovable("cumulative_scope_unproven")
        if len(prior) > MAX_ATTRIBUTABLE_FILES:
            raise PlanAuthorityUnprovable("cumulative_scope_unbounded")

        session = str(record.get("implementation_session_id") or "")
        if not session or session.startswith(PROVISIONAL_THREAD_PREFIXES):
            raise PlanAuthorityUnprovable("cumulative_session_unproven")
        if not _SHA256_RE.fullmatch(
            str(record.get("implementation_revision_sha256") or ""),
        ):
            raise PlanAuthorityUnprovable("cumulative_scope_unproven")
        if str(record.get("working_dir") or "") != request.working_dir:
            raise PlanAuthorityUnprovable("cumulative_workspace_unproven")
        try:
            self._require_owned_claim(tenant_ref, job_id, request.working_dir)
        except CodingServiceError:
            raise PlanAuthorityUnprovable("cumulative_claim_unproven") from None
        if self._load_resume_request(
            tenant_ref, job_id, str(record.get("request_sha256") or ""),
            session_bound=session,
        ) is None:
            raise PlanAuthorityUnprovable("cumulative_resume_unproven")
        try:
            current = self._revision_digest(request.working_dir, sorted(prior))
        except CodingServiceError:
            raise PlanAuthorityUnprovable("cumulative_revision_mismatch") from None
        if current != str(record.get("implementation_revision_sha256") or ""):
            raise PlanAuthorityUnprovable("cumulative_revision_mismatch")
        return tuple(sorted(prior))

    def _cumulative_route_scope(
        self,
        prior: Sequence[str],
        session_bound: str,
        result: CodingTaskResult,
        request: CodingTaskRequest,
    ) -> Tuple[str, ...]:
        """Union the proven prior scope with this round's host snapshot.

        The expensive proofs already happened before the round. What is left is
        the part that can only be known afterwards: that the round stayed in the
        session it was bound to, and that the union is still a bounded,
        canonical, in-workspace set. This is the exact ordered set the service
        will hash into the revision an auditor signs, so it is also exactly what
        the Indexer is asked to validate.
        """

        from flyto_ai.coding.route import CodingRouteError, RouteLane

        def refuse(code: str) -> "CodingRouteError":
            return CodingRouteError(code, RouteLane.INDEXER_POST)

        if prior:
            session = str(getattr(result, "thread_id", "") or "")
            if not session or session != session_bound:
                raise refuse("cumulative_session_unproven")
        union = sorted({
            str(item) for item in (getattr(result, "files_changed", ()) or ())
        } | {str(item) for item in prior})
        if not union or len(union) > MAX_ATTRIBUTABLE_FILES:
            raise refuse("cumulative_scope_unbounded")
        root = Path(request.working_dir).resolve()
        for relative in union:
            try:
                # Canonical, inside the workspace, not a link: identical to what
                # the revision digest will demand, asserted before the lanes
                # rather than after them.
                self._revision_target(root, relative)
            except CodingServiceError:
                raise refuse("cumulative_scope_unsafe") from None
        return tuple(union)

    def _bound_emergency_authority(
        self, record: Mapping[str, Any], job_id: str, *, rework: bool,
    ) -> Optional[EmergencyAuthorityReceipt]:
        """Return the authority that may continue this exact rework round.

        This runs *before* any implementer invocation, so every check here is a
        precondition for bypassing the strict route, not a later audit. A
        sealed receipt copied from another job cannot start an implementation:
        it must belong to this job, this original request, this service's
        implementer, and a round the service itself scheduled as rework.
        """

        stored = record.get("emergency_authority")
        if not isinstance(stored, dict):
            return None
        if not rework:
            # An initial round has no legitimate prior authority. A record that
            # carries one was either transplanted or replayed, so nothing runs.
            raise EmergencyAuthorityMissing(
                "an initial round cannot carry emergency authority",
            )
        if not self.emergency_policy.enabled:
            # A record can name emergency authority; only startup can grant it.
            raise EmergencyAuthorityMissing(
                "this service does not carry emergency overflow authority",
            )
        try:
            authority = EmergencyAuthorityReceipt.from_mapping(stored)
        except EmergencyAuthorityError as exc:
            raise EmergencyAuthorityMissing("emergency authority is invalid") from exc
        if not authority.sealed:
            raise EmergencyAuthorityMissing("emergency authority is not bound to a round")
        if authority.implementer_backend != self.implementation_backend:
            raise EmergencyAuthorityMissing(
                "emergency authority names a different implementer",
            )
        if authority.job_id != job_id:
            raise EmergencyAuthorityMissing("emergency authority belongs to another job")
        if authority.request_sha256 != str(record.get("request_sha256") or ""):
            raise EmergencyAuthorityMissing(
                "emergency authority does not bind this request",
            )
        if authority.session_id != str(record.get("implementation_session_id") or ""):
            raise EmergencyAuthorityMissing(
                "emergency authority does not bind this implementation session",
            )
        # The recorded trigger must still be one the policy would classify as
        # infrastructure; a rewritten lane or code re-enters classification.
        if classify_overflow_trigger(
            authority.trigger_lane, authority.trigger_action, authority.trigger_code,
        ) is None:
            raise EmergencyAuthorityMissing(
                "emergency authority does not name an infrastructure trigger",
            )
        return authority

    def _overflow_trigger(
        self,
        route: Optional["CodingRouteReceipt"],
        result: CodingTaskResult,
        progress: _RoundProgress,
        path: Path,
    ) -> Optional[EmergencyTrigger]:
        """Classify one failed round as broken infrastructure, or refuse.

        Every guard here is a reason *not* to open the lane. The route must
        have failed, the implementer must never have been invoked, no
        attributable change may exist, and the failure must be one of the
        positively classified infrastructure codes in a pre-implementer lane.
        """

        if not self.emergency_policy.applies_to(self.implementation_backend):
            return None
        if route is None or route.ok:
            return None
        if progress.implementer_started or getattr(result, "ok", False):
            # An implementer that already ran must never run twice.
            return None
        if list(getattr(result, "files_changed", ()) or ()):
            # A workspace edit already exists, so this is no longer a
            # pre-implementer infrastructure failure.
            return None
        try:
            if self._read_json(path).get("implementer_started") is True:
                # The durable record outranks in-memory progress. If a start
                # was ever recorded for this job, no second implementation may
                # run, even when the in-process flag was lost.
                return None
        except (OSError, ValueError):
            return None
        lane, action, code = route_failure_point(route)
        return classify_overflow_trigger(lane, action, code)

    async def _emergency_round(
        self,
        store: ThreadStore,
        request: CodingTaskRequest,
        job_id: str,
        request_sha256: str,
        path: Path,
        progress: _RoundProgress,
        trigger: EmergencyTrigger,
        *,
        mode: str,
    ) -> "tuple[CodingTaskResult, Optional[EmergencyAuthorityReceipt]]":
        """Run the startup-selected implementer directly under open-circuit authority.

        This is the same backend the operator already chose, with the same
        source-controlled checks and the same exact-revision binding. It skips
        only the lanes whose infrastructure is provably unreachable, and it
        still ends at an independent Codex audit.

        The overflow attempt becomes durable *before* the model is called, so a
        round that is still running — or that died mid-flight — is never read
        back as an ordinary strict round.
        """

        if mode == "emergency_rework":
            # A later process may not have tripped its own counter yet, but the
            # persisted authority already proved a real infrastructure trigger
            # for this job. Continuing is not a new decision.
            self._breaker.force_open()
        circuit = self._breaker.note_activation()
        progress.emergency = True
        progress.trigger = trigger
        # Truthful ordering: the durable record names the overflow lane and its
        # trigger before the model is called, never after it returns.
        self._update_record(path, execution_mode=EXECUTION_MODE_EMERGENCY, **{
            "emergency_trigger": {
                "lane": trigger.lane,
                "action": trigger.action,
                "code": trigger.code,
            },
        })
        _arm_start_marker(store, progress)
        result = await self.agent_factory(store).run(request)
        _reconcile_start_marker(progress, result)
        required = [
            item for item in (getattr(result, "checks", ()) or ())
            if getattr(item, "required", False)
        ]
        enforced = bool(required) and all(
            getattr(item, "passed", False) for item in required
        )
        if getattr(result, "ok", False) and not enforced:
            # The emergency lane keeps the source-controlled checks. A round
            # that produced no passing required check has no proof and is not
            # an emergency success.
            result = dataclasses.replace(
                result, ok=False, status="failed",
                failure_code="emergency_checks_missing",
            )
        authority = EmergencyAuthorityReceipt(
            mode=mode,
            circuit_state=circuit,
            trigger_lane=trigger.lane,
            trigger_action=trigger.action,
            trigger_code=trigger.code,
            implementer_backend=self.implementation_backend,
            instance_id=self.instance_id,
            build_id=self.build_id,
            job_id=job_id,
            request_sha256=request_sha256,
            implementer_started=True,
            checks_enforced=enforced,
        )
        return result, authority

    async def _implement(
        self,
        store: ThreadStore,
        request: CodingTaskRequest,
        progress: Optional[_RoundProgress] = None,
        *,
        rework_binding: Optional[Tuple[Mapping[str, Any], str, str]] = None,
        job_binding: Optional[Tuple[Path, str, str]] = None,
    ) -> "tuple[CodingTaskResult, Optional[CodingRouteReceipt]]":
        """Run the selected implementer, wrapped by the host-owned route.

        Without a strict startup policy this is exactly the historical direct
        call, so existing library callers keep their behavior. The public
        `code-mcp` / `code-serve` builders always enable the strict route.
        """

        policy = self.route_policy
        if policy is None or not policy.strict:
            _arm_start_marker(store, progress)
            unrouted = await self.agent_factory(store).run(request)
            _reconcile_start_marker(progress, unrouted)
            if rework_binding is not None:
                unrouted = self._promote_verified_cumulative_no_change(
                    unrouted, request, *rework_binding,
                )
            return unrouted, None

        from flyto_ai.coding.capabilities import CapabilityManager
        from flyto_ai.coding.route import CodingRouteOrchestrator

        specs = tuple(
            spec for spec in (policy.indexer, policy.blueprint) if spec is not None
        )
        # The host lanes need the least authority that actually permits the
        # real calls: Indexer `task`/`verify` persist state and reindex.
        manager = CapabilityManager(request.working_dir, "workspace_write")
        try:
            try:
                statuses = await manager.start(specs)
            except Exception:
                # A capability that cannot even be launched is unavailable
                # infrastructure, not a service crash. Which one it was is not
                # determinable here, so the mandatory first lane is reported.
                return (
                    self._route_unavailable(request),
                    self._route_unavailable_receipt(policy),
                )
            missing = [
                status.name for status in statuses
                if status.required and not status.available
            ]
            if missing or not manager.required_available:
                # A stale or unavailable capability must never reach an
                # auditable state, so the round fails before the model can
                # edit. The lane names the provider that was actually missing.
                lane = self._unavailable_lane(policy, missing)
                return (
                    self._route_unavailable(request, lane=lane),
                    self._route_unavailable_receipt(policy, lane=lane),
                )
            orchestrator = CodingRouteOrchestrator(
                policy,
                capability_dispatch=self._lane_dispatcher(manager, specs),
                core_dispatch=self._core_dispatcher() if policy.core_enabled else None,
            )

            async def implement(
                bound_request: CodingTaskRequest, projection: str,
            ) -> CodingTaskResult:
                effective = bound_request
                if projection:
                    effective = dataclasses.replace(
                        bound_request,
                        message="{}\n\n{}".format(bound_request.message, projection),
                    )
                # The pre-lanes have passed. The start itself is recorded
                # by the adapter, at the first real provider or session call,
                # so an adapter that refuses before contacting one is not
                # written down as having started.
                _arm_start_marker(store, progress)
                routed = await self.agent_factory(store).run(effective)
                _reconcile_start_marker(progress, routed)
                if rework_binding is not None:
                    routed = self._promote_verified_cumulative_no_change(
                        routed, bound_request, *rework_binding,
                    )
                return routed

            parent_contract = None
            on_pre_contract = None
            cumulative_scope = None
            if job_binding is not None:
                job_path, job_tenant_ref, bound_job_id = job_binding
                bound_record = self._read_json(job_path)
                # Only a rework amends. A first round has no parent and its
                # Indexer request is unchanged from what it has always been.
                prior_scope: Tuple[str, ...] = ()
                session_bound = ""
                if rework_binding is not None:
                    parent_contract = self._load_plan_authority(bound_record)
                    if parent_contract is None:
                        raise PlanAuthorityUnprovable("plan_authority_unavailable")
                    # Before the resumed implementer touches anything, while
                    # the recorded revision still describes the tree.
                    prior_scope = self._prove_prior_scope(
                        bound_record, request, job_tenant_ref, bound_job_id,
                    )
                    session_bound = str(
                        bound_record.get("implementation_session_id") or "",
                    )

                def on_pre_contract(contract, _path=job_path):
                    self._persist_plan_authority(
                        _path, self._read_json(_path), contract,
                    )

                def cumulative_scope(
                    round_result, _prior=prior_scope, _session=session_bound,
                ):
                    scope = self._cumulative_route_scope(
                        _prior, _session, round_result, request,
                    )
                    if progress is not None:
                        progress.route_scope = scope
                    return scope

            outcome = await orchestrator.run(
                request, implement,
                parent_contract=parent_contract,
                on_pre_contract=on_pre_contract,
                cumulative_scope=cumulative_scope,
            )
        finally:
            closed = True
            try:
                await manager.close()
            except Exception:
                closed = False
        if not closed:
            # A capability that will not shut down cleanly leaves an
            # unaccounted process. That is never a passed route.
            return (
                self._route_unavailable(request, "route_capability_close_failed"),
                self._route_unavailable_receipt(policy, "capability_close_failed"),
            )
        return outcome

    def _promote_verified_cumulative_no_change(
        self,
        result: CodingTaskResult,
        request: CodingTaskRequest,
        record: Mapping[str, Any],
        tenant_ref: str,
        job_id: str,
    ) -> CodingTaskResult:
        """Close a clean no-op rework against the already audited revision.

        ``require_changes`` is a job invariant, not a demand to manufacture a
        fresh diff after every audit.  The provider adapter reports
        ``no_changes`` when a resumed round leaves the tree untouched even if
        the job already owns a real attributable revision.  Promote only that
        exact host-generated outcome, only with passing required checks, and
        only after every cumulative session/claim/envelope/content binding is
        re-proved.  The promoted ``files_changed`` is therefore the job's
        cumulative attributable set consumed by the post-route validator, not
        a claim that this round rewrote those files.
        """

        if result.ok or self._implementation_failure_code(result) != "no_changes":
            return result
        required = [
            item for item in (getattr(result, "checks", ()) or ())
            if getattr(item, "required", False)
        ]
        if not required or any(not getattr(item, "passed", False) for item in required):
            return result
        cumulative = self._cumulative_attribution(
            record, result, request, tenant_ref, job_id,
        )
        if not cumulative:
            return result
        return dataclasses.replace(
            result,
            ok=True,
            status="completed",
            files_changed=list(cumulative),
            failure_code=None,
            message=result.message or "Coding rework verified the cumulative revision.",
        )

    @staticmethod
    def _lane_dispatcher(manager: Any, specs: Sequence[Any]) -> Any:
        """Map a bare lane tool name onto its provider-scoped capability tool."""

        from flyto_ai.coding.mcp_catalog import provider_tool_name

        async def dispatch(tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
            for spec in specs:
                scoped = provider_tool_name(spec.name, tool)
                if any(item.get("name") == scoped for item in manager.definitions):
                    return await manager.dispatch(scoped, dict(arguments))
            return {"ok": False, "error": "unknown route tool"}

        return dispatch

    @staticmethod
    def _core_dispatcher() -> Any:
        """Route Core validation through the one supported adapter boundary."""

        from flyto_ai.coding.route import CORE_ALLOWED_TOOLS

        async def dispatch(tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
            if tool not in CORE_ALLOWED_TOOLS:
                return {"ok": False, "error": "core tool is not allowlisted"}
            from flyto_ai.tools.core_tools import dispatch_core_tool

            return await dispatch_core_tool(tool, dict(arguments))

        return dispatch

    @staticmethod
    def _unavailable_lane(policy: Any, missing: Sequence[str]) -> str:
        """Name the lane whose provider was actually unavailable at startup.

        A Blueprint that will not launch is a Blueprint failure. Reporting it
        as `indexer_pre` because that lane runs first would send an operator to
        the wrong process.
        """
        from flyto_ai.coding.route import RouteLane

        names = {str(item) for item in missing}
        indexer = getattr(policy.indexer, "name", "") if policy.indexer else ""
        blueprint = getattr(policy.blueprint, "name", "") if policy.blueprint else ""
        if blueprint and blueprint in names and indexer not in names:
            return RouteLane.BLUEPRINT.value
        return RouteLane.INDEXER_PRE.value

    @staticmethod
    def _route_unavailable(
        request: CodingTaskRequest,
        code: str = "route_capability_unavailable",
        *,
        lane: str = "indexer_pre",
    ) -> CodingTaskResult:
        return CodingTaskResult(
            ok=False,
            message="coding route lane {} failed: {}".format(lane, code),
            thread_id=route_thread_id(request.thread_id or request.working_dir),
            attempts=0,
            status="failed",
            evidence_path="",
            failure_code=code,
        )

    @staticmethod
    def _route_unavailable_receipt(
        policy: Any,
        code: str = "capability_unavailable",
        *,
        lane: str = "indexer_pre",
    ) -> "CodingRouteReceipt":
        from flyto_ai.coding.route import (
            CodingRouteReceipt,
            RouteLaneReceipt,
            RouteLaneStatus,
        )

        return CodingRouteReceipt(
            strict=True, ok=False, failure_code=code,
            lanes=(RouteLaneReceipt(
                lane=lane, required=True,
                status=RouteLaneStatus.FAILED, reason_code=code,
            ),),
        )

    def _record_outcome(
        self,
        path: Path,
        tenant_ref: str,
        job_id: str,
        request: CodingTaskRequest,
        result: CodingTaskResult,
        store: ThreadStore,
        rework: bool,
        route: Optional["CodingRouteReceipt"] = None,
        *,
        progress: Optional[_RoundProgress] = None,
        authority: Optional[EmergencyAuthorityReceipt] = None,
        settle: Optional["_RoundSettlement"] = None,
    ) -> None:
        """Move one finished implementation round into its durable state.

        Every exit from here that publishes a terminal or audit-ready state
        settles the round's mission work item first and writes the resulting
        projection in the same record update, so the published state and the
        owner-closed item are never observable apart.
        """

        if settle is None:
            settle = _RoundSettlement(self, None, tenant_ref, job_id)

        started = progress.implementer_started if progress is not None else False
        result_record = dataclasses.asdict(result)
        result_record["evidence_path"] = ""
        outcome: Dict[str, Any] = {
            # The two authorities are alternatives. A round that took the
            # emergency lane records no route receipt, so a failed strict
            # receipt can never be mistaken for a passed one.
            "route_receipt": route.to_mapping() if route is not None else None,
            "emergency_authority": (
                authority.to_mapping() if authority is not None else None
            ),
            "thread_id": result.thread_id,
            "evidence_sha256": store.digest(result.thread_id),
            "result": result_record,
            "failure_code": result.failure_code,
            "implementer_started": started,
            # Re-validated on the way in: whatever an adapter put here, only
            # safe contract identifiers become durable.
            "verification_blockers": list(
                safe_blockers(getattr(result, "verification_blockers", ()))
            ),
        }
        if started and self.require_codex_audit:
            outcome["implementation_backend"] = self.implementation_backend
        blockers: Tuple[str, ...] = ()
        cumulative: Tuple[str, ...] = ()
        if rework and not result.ok and not (result.files_changed or ()):
            # Only asked on the exact path the production failure took: a
            # rework round that changed nothing new. Everywhere else the
            # round's own files are the evidence and this proof is not
            # consulted at all.
            cumulative = self._cumulative_attribution(
                self._read_json(path), result, request, tenant_ref, job_id,
            )
        if not result.ok:
            if not self._auditable_failure(
                result, route, authority, started, cumulative,
            ):
                # The route seam already proved this round's cumulative scope
                # before the proof lanes ran, so a lane that refused afterwards
                # does not narrow what the round owns. If that proof never
                # succeeded the terminal evidence stays this round's own.
                proof = self._failed_round_proof(
                    request, result, started, outcome,
                    tuple(getattr(progress, "route_scope", ()) or ()),
                )
                self._discard_resume(tenant_ref, job_id)
                # Terminal and non-landable either way. The only question left
                # is whether this exact session may be carried forward, and it
                # is answered from host-owned proof, never from the round's
                # word: a recognized bounded stop, a real session, and a
                # revision the host itself just hashed.
                #
                # Settled *before* the terminal state is published, so a reader
                # that sees `failed` already sees the final continuation answer.
                # Publishing first would expose a window in which a job looks
                # finished while its authority is still mid-rotation.
                self._close_continuation(
                    tenant_ref, job_id, path,
                    self._implementation_failure_code(result), proof,
                )
                self._update_record(
                    path, state=CodingJobState.FAILED.value, landable=False,
                    **settle(
                        state=CodingJobState.FAILED.value,
                        failure_code=self._implementation_failure_code(result),
                    ),
                    **proof,
                )
                return
            # A real, attributable, resumable implementation that is not
            # landable. It falls through to the same auditable tail as a
            # successful round, which still fails closed on its own if the
            # session, change set, or revision cannot be bound exactly.
            blockers = self._implementation_blockers(result, route)
        if not self.require_codex_audit:
            self._discard_resume(tenant_ref, job_id)
            self._update_record(
                path, state=CodingJobState.COMPLETED.value,
                **settle(
                    state=CodingJobState.COMPLETED.value,
                    failure_code=str(result.failure_code or ""),
                ),
                **outcome,
            )
            return
        session = str(result.thread_id or "")
        if not session:
            raise SessionBindingFailed("the implementation session id is missing")
        if blockers and session.startswith(PROVISIONAL_THREAD_PREFIXES):
            # A provisional host thread proves nothing about an implementation
            # session and can never be resumed, so a round that would need
            # rework to continue has nothing to continue. Fail closed rather
            # than mint an unusable audit loop.
            raise SessionBindingFailed(
                "a blocked round has no resumable implementation session",
            )
        files = {str(item) for item in result.files_changed}
        if rework:
            record = self._read_json(path)
            recorded = str(record.get("implementation_session_id") or "")
            if recorded and recorded != session:
                raise SessionBindingFailed("rework left the original implementation session")
            # The agent re-snapshots per round, so a rework round only reports
            # what it touched. The audited revision must stay cumulative or an
            # untouched earlier file could change without invalidating it.
            files |= {str(item) for item in (record.get("implementation_files") or [])}
        files = sorted(files)
        validated_scope = tuple(getattr(progress, "route_scope", ()) or ())
        if validated_scope and tuple(files) != validated_scope:
            # Equality, not inclusion. The whole point of validating a
            # cumulative scope is that it is the scope an auditor is later
            # offered; a record that reconstructs a different order or a
            # different membership has quietly broken that link.
            raise RevisionUnavailable(
                "the audited change set is not the validated cumulative scope",
            )
        if not files:
            raise RevisionUnavailable("no attributable implementation change to audit")
        if len(files) > MAX_ATTRIBUTABLE_FILES:
            raise RevisionUnavailable("the attributable change set is outside the revision bound")
        revision = self._revision_digest(request.working_dir, files)
        if authority is not None:
            # Seal the authority to this exact round only once the session and
            # revision exist. An unsealed authority can never authorize a
            # verdict, and a sealed one cannot be moved to another job.
            outcome["emergency_authority"] = authority.seal(
                job_id=job_id,
                request_sha256=str(self._read_json(path).get("request_sha256") or ""),
                session_id=session,
                revision_sha256=revision,
            ).to_mapping()
        # An audited round always names its implementer, whether or not the
        # in-memory progress object survived the round.
        outcome["implementation_backend"] = self.implementation_backend
        outcome["implementer_started"] = True
        # Seal the resume envelope to the session this round actually produced,
        # before the job becomes auditable. Only now can another worker prove
        # that replaying it continues this session instead of starting one.
        with self._state_guard():
            self._seal_resume_envelope(
                tenant_ref,
                job_id,
                request,
                str(self._read_json(path).get("request_sha256") or ""),
                session,
            )
        # Reaching an auditable revision is where a continuation ends. The
        # ordinary same-session audit and rework rules own the job from here,
        # so the authority settles exactly once and can never be spent again -
        # and it settles before the job becomes visibly auditable, so no reader
        # ever sees an audit-ready job still advertising a live resume.
        self._settle_continuation(tenant_ref, job_id, path)
        # The item closes `fixed` here, on this round's own revision and file
        # set, and the closed projection is published in the same write that
        # makes the job auditable. An auditor therefore never observes an
        # audit-ready job whose work item is still dispatched, which is what
        # made an accept-then-rework able to fork the graph.
        settled = settle(
            revision=revision,
            files=files,
            state=CodingJobState.AWAITING_CODEX_AUDIT.value,
        )
        self._update_record(
            path,
            state=CodingJobState.AWAITING_CODEX_AUDIT.value,
            implementation_session_id=session,
            **settled,
            implementation_revision_sha256=revision,
            implementation_files=files,
            working_dir=request.working_dir,
            landable=False,
            # A round that closed cleanly clears whatever the previous round
            # was blocked on, so a successful rework becomes acceptable without
            # any separate operator action.
            implementation_blockers=list(blockers),
            **outcome,
        )

    def _close_continuation(
        self,
        tenant_ref: str,
        job_id: str,
        path: Path,
        failure_code: str,
        proof: Mapping[str, Any],
    ) -> None:
        """Open, rotate, or settle this job's continuation authority.

        Called on the terminal path only. A round that became auditable keeps
        the existing same-session rework loop instead, which is why a Codex
        rework and a budget continuation can never be confused: they are
        produced by mutually exclusive branches of the same decision.

        Creating one requires all four host-owned facts at once - an audited
        service, a recognized bounded stop, a real backend session, and a
        revision this host hashed from the tree it still owns. Anything less
        settles instead, so a failure the host cannot describe never leaves a
        resumable record behind.
        """

        session = str(proof.get("implementation_session_id") or "")
        revision = str(proof.get("implementation_revision_sha256") or "")
        files = tuple(str(item) for item in (proof.get("implementation_files") or ()))
        record = self._read_json(path)
        continuable = (
            self.require_codex_audit
            and failure_code in CONTINUABLE_STOP_CODES
            and is_continuable_session(session)
            and bool(_SHA256_RE.fullmatch(revision))
            and bool(files)
            and len(files) <= MAX_ATTRIBUTABLE_FILES
            # A session whose pin cannot be recovered and re-proved is not
            # offered forward at all. Writing an authority that no continuation
            # could ever satisfy would advertise something untrue on the
            # receipt; declining to write one keeps the job terminal and honest.
            and self._record_pinned_contract(record) is not None
        )
        if continuable:
            try:
                # The tree as it stands at the stop, in full. This is the state
                # a later segment must find unchanged, so it is measured here
                # rather than reconstructed from the attributable set.
                manifest = workspace_manifest_digest(
                    str(record.get("working_dir") or ""), self.snapshot_policy,
                )
            except (WorkspaceUnobservable, OSError):
                # A workspace this host cannot describe exactly cannot be
                # promised to a later segment. The job stays terminal and
                # truthful; it simply offers nothing.
                continuable = False
        if not continuable:
            self._settle_continuation(tenant_ref, job_id, path)
            return
        with self._state_guard():
            existing = self._continuation.load(tenant_ref, session)
            now = time.time()
            if existing is None:
                authority = ContinuationAuthority(
                    tenant_ref=tenant_ref,
                    backend=self.implementation_backend,
                    session_id=session,
                    job_id=job_id,
                    origin_job_id=job_id,
                    working_dir=str(record.get("working_dir") or ""),
                    workspace_sha256=str(record.get("workspace_sha256") or ""),
                    revision_sha256=revision,
                    workspace_manifest_sha256=manifest,
                    snapshot_policy_sha256=self.snapshot_policy.identity(),
                    files=tuple(sorted(set(files))),
                    authorized_config_sha256=str(
                        record.get("authorized_config_sha256") or "",
                    ),
                    # Bind the contract this session ran under, by identity. The
                    # snapshot stays in this job's private record; the authority
                    # carries only its address, so a later segment has to
                    # produce a snapshot that still hashes to it.
                    contract_snapshot_sha256=str(
                        record.get("contract_snapshot_sha256") or "",
                    ),
                    request_sha256=str(record.get("request_sha256") or ""),
                    failure_code=failure_code,
                    generation=1,
                    sequence=1,
                )
                try:
                    self._continuation.create(authority)
                except (ContinuationConflict, ContinuationCorrupt, OSError, ValueError):
                    # The job is already terminal and truthful. A continuation
                    # that could not be written is simply not offered.
                    return
                self._update_record_locked(
                    path,
                    continuation_session_id=session,
                    continuation_generation=1,
                    continuation_origin_job_id=job_id,
                )
                return
            if existing.claimed_by_job_id != job_id or existing.state != STATE_CLAIMED:
                # This job is not the one that consumed the live authority, so
                # it has no standing to move it. Never rewrite somebody else's
                # generation, and never resurrect a settled one.
                return
            try:
                if existing.generation >= MAX_CONTINUATION_GENERATION:
                    self._continuation.commit(existing, existing.settled(now))
                    return
                rotated = self._continuation.commit(existing, existing.rotated(
                    job_id=job_id,
                    revision_sha256=revision,
                    workspace_manifest_sha256=manifest,
                    files=files,
                    failure_code=failure_code,
                    now=now,
                ))
            except (ContinuationConflict, ContinuationCorrupt, OSError, ValueError):
                return
            self._update_record_locked(
                path,
                continuation_session_id=session,
                continuation_generation=rotated.generation,
                continuation_origin_job_id=rotated.origin_job_id,
            )

    def _settle_continuation(
        self, tenant_ref: str, job_id: str, path: Optional[Path] = None,
    ) -> None:
        """Close this job's continuation authority, exactly once and only its own.

        Idempotent, so the crash-reconciliation path, a terminal worker exit, an
        abandon, and a normal settlement can all call it without double-closing
        anything. A record it does not own is left untouched: settling another
        job's authority would silently destroy a live continuation.
        """

        try:
            record = self._read_json(path) if path is not None else {}
        except (OSError, ValueError):
            return
        session = str(record.get("continuation_session_id") or "")
        if not is_continuable_session(session):
            return
        with self._state_guard():
            authority = self._continuation.load(tenant_ref, session)
            if authority is None or authority.state == STATE_SETTLED:
                return
            if job_id not in {authority.job_id, authority.claimed_by_job_id}:
                return
            try:
                self._continuation.commit(authority, authority.settled(time.time()))
            except (ContinuationConflict, ContinuationCorrupt, OSError, ValueError):
                return

    def _cumulative_attribution(
        self,
        record: Optional[Mapping[str, Any]],
        result: CodingTaskResult,
        request: CodingTaskRequest,
        tenant_ref: str,
        job_id: str,
    ) -> Tuple[str, ...]:
        """Prove a rework round is still bound to a real earlier revision.

        A rework round snapshots the worktree itself, so it reports only what
        *this* round touched. When the auditor's finding is about a file the
        previous round already changed, the correct repair frequently rewrites
        that same file - and the honest answer to "what is newly attributable
        here" is nothing at all. Terminalizing on that is the production bug:
        a real, still-auditable revision was thrown away, its session was
        discarded, and Codex lost the loop it was in the middle of.

        Zero new files is therefore not, by itself, a dead end. It is only a
        dead end when the *cumulative* binding cannot still be proven. Every
        binding below is proven from durable host state, never from the
        model's word, and any one of them failing returns `()`:

        * the record is already audit-bound - it names a session, a revision
          and a bounded, non-empty attributable set;
        * this round resumed that exact session, and it is a real backend
          session rather than a provisional host id;
        * the workspace is the same one the recorded change was measured in;
        * this exact tenant+job still holds the live worktree claim, so nobody
          else has edited the tree since;
        * the resume envelope still exists, still matches this job's request
          digest, and is still sealed to that same session;
        * every cumulative path still resolves, canonically and without
          following a link, to a regular file inside the workspace.

        Returns the cumulative set on success so the caller re-derives the
        revision from exactly what was proven, never from a stored digest.
        """

        if not record:
            return ()
        recorded_session = str(record.get("implementation_session_id") or "")
        session = str(getattr(result, "thread_id", "") or "")
        if (
            not recorded_session
            or recorded_session != session
            or session.startswith(PROVISIONAL_THREAD_PREFIXES)
        ):
            return ()
        if not _SHA256_RE.fullmatch(str(record.get("implementation_revision_sha256") or "")):
            return ()

        stored = record.get("implementation_files")
        if not isinstance(stored, (list, tuple)):
            return ()
        files = sorted({
            str(item) for item in stored
            if isinstance(item, str) and not isinstance(item, bool)
        })
        if not files or len(files) != len(stored) or len(files) > MAX_ATTRIBUTABLE_FILES:
            # A mutated set - duplicated, re-typed, or grown past the bound -
            # is not the set that was audited.
            return ()

        workspace = str(record.get("working_dir") or "")
        if not workspace or workspace != request.working_dir:
            return ()

        try:
            # Still ours, and still ours *in this tenant*.
            self._require_owned_claim(tenant_ref, job_id, workspace)
        except CodingServiceError:
            return ()

        if self._load_resume_request(
            tenant_ref,
            job_id,
            str(record.get("request_sha256") or ""),
            session_bound=session,
        ) is None:
            return ()

        root = Path(workspace).resolve()
        for relative in files:
            try:
                target = self._revision_target(root, relative)
            except CodingServiceError:
                return ()
            # `_revision_digest` treats a deleted path as a recorded absence,
            # which is right for a round that really did delete something. It
            # is not right here: the cumulative set is being reused as the sole
            # evidence, so every member has to still be a real file.
            if target.is_symlink() or not target.is_file():
                return ()

        # The load-bearing check, and the one that was missing: the *bytes*
        # must still hash to the revision an auditor already saw. Checking only
        # that a digest was recorded proves the job was once bound to
        # something; it says nothing about what is in the tree now. Without
        # this, a round that changed `result.txt` outside its own attribution -
        # or anything else that edited the tree between rounds - would be
        # laundered into a "continuous" revision and re-signed under the same
        # session, which is precisely the continuity claim an audit relies on.
        try:
            current = self._revision_digest(workspace, files)
        except CodingServiceError:
            return ()
        if current != str(record.get("implementation_revision_sha256") or ""):
            return ()
        return tuple(files)

    def _auditable_failure(
        self,
        result: CodingTaskResult,
        route: Optional["CodingRouteReceipt"],
        authority: Optional[EmergencyAuthorityReceipt],
        started: bool,
        cumulative: Tuple[str, ...] = (),
    ) -> bool:
        """Whether a failed round is real work awaiting rework, not a dead end.

        This is the whole difference between "the implementer never produced
        anything an auditor could read" and "the implementer produced a real,
        attributable, resumable change that is not good enough yet". Every
        condition here is a precondition for the *weaker* of the two outcomes:
        the round still cannot land, cannot be accepted, and cannot skip an
        audit. It can only be sent back to its own session.

        Provider-neutral by construction: nothing here names a backend, a
        model, a stop reason, or a check. It reads the host's own snapshots,
        the host's own lane receipt, and nothing else.
        """

        if not self.require_codex_audit or not started:
            # Without an audit loop there is no rework path to hold the job
            # open for, and a round that never invoked an implementer has
            # produced nothing to rework.
            return False
        if self._implementation_failure_code(result) not in (
            AUDITABLE_IMPLEMENTATION_FAILURE_CODES
        ):
            # The closed vocabulary decides this, not the shape of the result.
            # A backend that reports an unrecognized failure *and* changed
            # files is describing something the host cannot reason about, and
            # an unreasonable failure is never resumable.
            return False
        if authority is not None:
            # The emergency lane is deliberately all-or-nothing: its authority
            # is only honoured when the required checks really passed. A
            # partial overflow round stays terminal rather than inventing a
            # rework loop on already-degraded authority.
            return False
        files = tuple(str(item) for item in (getattr(result, "files_changed", ()) or ()))
        if len(files) > MAX_ATTRIBUTABLE_FILES:
            # More than the revision bound can describe means no exact revision
            # an auditor could bind.
            return False
        if not files and not cumulative:
            # Nothing new *and* nothing still provable from earlier rounds. A
            # fresh job in this position has produced nothing; a rework round
            # in this position has lost its bindings. Either way there is no
            # revision to audit.
            return False
        session = str(getattr(result, "thread_id", "") or "")
        if not session or session.startswith(PROVISIONAL_THREAD_PREFIXES):
            # Rework must resume this exact conversation. A host-minted
            # provisional id cannot, so there is nothing to reopen.
            return False
        if route is None:
            # No strict route wrapped this round, so the implementer's own
            # result is the only evidence and it reported a real change.
            return True
        # A strict route that failed anywhere except on the implementation's
        # own result is an infrastructure or safety refusal, and stays terminal.
        return route_blocks_implementation(route)

    @staticmethod
    def _implementation_failure_code(result: CodingTaskResult) -> str:
        """Return what the implementer said, never what a lane said about it.

        A host lane that refuses a round rewrites `failure_code` to name itself
        and carries the implementer's own classification alongside. Reading the
        lane's code here would classify every wrapped round identically, which
        is exactly how an unknown provider failure would look reworkable.
        """

        wrapped = str(getattr(result, "implementation_failure_code", "") or "")
        return wrapped or str(getattr(result, "failure_code", "") or "")

    @classmethod
    def _implementation_blockers(
        cls, result: CodingTaskResult, route: Optional["CodingRouteReceipt"],
    ) -> Tuple[str, ...]:
        """Derive the bounded, stable reasons this revision cannot be accepted.

        Every code comes from a host-owned fact: the implementer's own failure
        classification, the names of required checks the host ran and watched
        fail, and the route's own lane failure code. Model prose, provider
        exception text, command output, and paths have no representation here.
        """

        codes = set()
        # Both facts, when both exist: what the implementer reported, and what
        # the lane that refused the round reported about it.
        for failure in (
            str(getattr(result, "implementation_failure_code", "") or ""),
            str(getattr(result, "failure_code", "") or ""),
        ):
            if _BLOCKER_CODE_RE.fullmatch(failure):
                codes.add(failure)
        for check in getattr(result, "checks", ()) or ():
            if not getattr(check, "required", False) or getattr(check, "passed", False):
                continue
            name = "check.{}".format(getattr(check, "name", ""))[:64]
            if _BLOCKER_CODE_RE.fullmatch(name):
                codes.add(name)
        if route is not None and not route.ok:
            lane_code = "route.{}".format(route.failure_code)[:64]
            if _BLOCKER_CODE_RE.fullmatch(lane_code):
                codes.add(lane_code)
        if not codes:
            codes.add(GENERIC_IMPLEMENTATION_BLOCKER)
        return tuple(sorted(codes))[:MAX_IMPLEMENTATION_BLOCKERS]

    def _failed_round_proof(
        self,
        request: CodingTaskRequest,
        result: CodingTaskResult,
        started: bool,
        outcome: Mapping[str, Any],
        scope: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Keep bounded proof that an implementation ran, without landability.

        A round that failed in Core validation or Indexer post-work still
        produced real work. Retaining the session, attributable files, and the
        revision digest lets a later reader tell "the model never ran" from
        "the model ran and the proof lane refused it". None of this makes the
        job auditable: the state stays terminal and non-landable.

        `scope` is the cumulative set the route seam already proved for this
        round - the same tuple `_cumulative_route_scope` returned and recorded
        on the round's progress. A rework round only reports what it touched,
        so deriving the terminal evidence from `files_changed` alone would drop
        every file earlier rounds opened and describe a narrower change set
        than the one this job actually owns. When that proof exists the
        evidence binds it exactly; when it does not, nothing is carried over
        and this round's own snapshot remains the only claim made.
        """

        record = dict(outcome)
        if not started:
            return record
        session = str(result.thread_id or "")
        if session and not session.startswith(PROVISIONAL_THREAD_PREFIXES):
            # A provisional host thread proves nothing about an implementation
            # session, so it is never promoted into one.
            record["implementation_session_id"] = session
        proven = [str(item) for item in (scope or ())]
        # The proven union in the exact order it was validated in, or else this
        # round's own snapshot. Never a reconstruction from the stored record:
        # a prior scope that was not re-proven before the round is not evidence.
        files = proven or sorted({str(item) for item in (result.files_changed or ())})
        if not files or len(files) > MAX_ATTRIBUTABLE_FILES:
            return record
        record["implementation_files"] = files
        record["working_dir"] = request.working_dir
        try:
            record["implementation_revision_sha256"] = self._revision_digest(
                request.working_dir, files,
            )
        except (CodingServiceError, OSError):
            # A revision that cannot be hashed safely is simply not recorded.
            # The failure is already terminal; this is evidence, not a gate.
            pass
        return record

    def _fail_job(
        self,
        path: Path,
        tenant_ref: str,
        job_id: str,
        code: str,
        exc: BaseException,
        progress: Optional[_RoundProgress] = None,
        *,
        settle: Optional["_RoundSettlement"] = None,
    ) -> None:
        """Force one terminal fail-closed record for any worker exit.

        The round's work item is owner-closed before this record is published,
        and its projection travels in the same write, so `failed` is never
        observable while this process still holds the item dispatched.
        """

        self._discard_resume(tenant_ref, job_id)
        changes: Dict[str, Any] = {
            "state": CodingJobState.FAILED.value,
            "failure_code": code,
            "landable": False,
            "error": str(redact_evidence(str(exc)))[:1000],
        }
        if settle is not None:
            changes.update(
                settle(state=CodingJobState.FAILED.value, failure_code=code),
            )
        if progress is not None and progress.implementer_started:
            changes["implementer_started"] = True
            if self.require_codex_audit:
                changes["implementation_backend"] = self.implementation_backend
        # Any worker exit that lands here is a failure the host could not
        # classify as a bounded stop, so whatever continuation this job was
        # holding is closed rather than left looking resumable - and closed
        # before the terminal record is published, so the two are never
        # observable out of order.
        self._settle_continuation(tenant_ref, job_id, path)
        try:
            self._update_record(path, **changes)
        except (OSError, ValueError):
            # The record is unwritable. There is nothing further this worker
            # can durably say, and raising here would only lose the original
            # failure inside an unobserved future.
            pass

    def status_health(self) -> Dict[str, Any]:
        """Report whether the diagnostic recorder itself is working.

        Status publication is deliberately non-fatal to a real job, but a
        silently broken recorder would be indistinguishable from a quiet
        service. These bounded counters make that difference inspectable.
        """
        with self._lock:
            return {
                "instance_id": self.instance_id,
                "build_id": self.build_id,
                "published": self._status_published,
                "failures": self._status_failures,
                "last_failure_code": self._status_failure_code,
            }

    def _close_status(self) -> None:
        """Republish the last known facts with a `closed` lifecycle.

        A graceful shutdown changes only the lifecycle and the update time. It
        must not erase which job ran, where it stopped, or whether the
        implementer started — that is exactly what a later reader needs.
        """
        with self._lock:
            last = dict(self._last_status_record)
        self._publish_status(last, lifecycle="closed")

    def _publish_status(
        self, record: Mapping[str, Any], *, lifecycle: str = "active",
    ) -> None:
        """Refresh this instance's bounded runtime status. Never authoritative.

        The caller already holds the cross-process state guard. Only closed,
        bounded facts cross into the status file: the task message, error text,
        workspace path, and file list stay in the per-job record, which remains
        the single source of authority.
        """
        lane, action = route_progress(record)
        try:
            status = CodingRouteStatus(
                lifecycle=lifecycle,
                instance_id=self.instance_id,
                build_id=self.build_id,
                service_version=self.service_version,
                process_id=os.getpid(),
                started_at=self.started_at,
                updated_at=time.time(),
                implementation_backend=self.implementation_backend,
                emergency_enabled=self.emergency_policy.enabled,
                circuit_state=self._breaker.state,
                emergency_activations=self._breaker.activations,
                job_id=str(record.get("job_id") or ""),
                state=str(record.get("state") or ""),
                mode=route_mode(record),
                lane=lane,
                action=action,
                failure_code=str(record.get("failure_code") or ""),
                implementer_started=record.get("implementer_started") is True,
                implementation_session_id=str(
                    record.get("implementation_session_id") or "",
                ),
                implementation_revision_sha256=str(
                    record.get("implementation_revision_sha256") or "",
                ),
                audit_count=int(record.get("audit_count") or 0),
                rework_count=int(record.get("rework_count") or 0),
                landable=record.get("landable") is True,
                job_terminal=str(record.get("state") or "") in _TERMINAL_STATE_VALUES,
                publish_failures=self._status_failures,
                last_publish_failure_code=self._status_failure_code,
            )
            self._status.publish(status)
        except (OSError, ValueError) as exc:
            # Status is a diagnostic pointer. A write that cannot complete must
            # never fail a real job — but it is counted with a stable code
            # rather than swallowed, and this path never publishes again while
            # reporting its own failure.
            self._status_failures += 1
            self._status_failure_code = (
                STATUS_FAILURE_CODES[0] if isinstance(exc, OSError)
                else STATUS_FAILURE_CODES[1]
            )
            if self._status_failures == 1:
                # One bounded, secret-free line so a permanently unwritable
                # state root is visible without inspecting counters.
                print(
                    "flyto coding status recorder unavailable: {}".format(
                        self._status_failure_code,
                    ),
                    file=sys.stderr,
                )
            return
        self._status_published += 1
        with self._lock:
            self._last_status_record = dict(record)

    def _require_execution_authority(self, record: Mapping[str, Any]) -> None:
        """Accept exactly one valid execution authority for this exact round.

        A strict public round is auditable only with its own proof, and there
        are exactly two shapes of proof:

        - a digest-valid, passed, strict `CodingRouteReceipt`; or
        - a digest-valid emergency receipt that this service is configured to
          honour, sealed to this job, request, session, and revision, whose
          required source-controlled checks really passed.

        Missing, mixed, transplanted, tampered, disabled, wrong-backend,
        failed-check, and ordinary failed-route evidence all fail closed. A
        failed strict route is never rewritten as a passed one.
        """
        policy = self.route_policy
        if policy is None or not policy.strict:
            return
        route_stored = record.get("route_receipt")
        emergency_stored = record.get("emergency_authority")
        has_route = isinstance(route_stored, dict)
        has_emergency = isinstance(emergency_stored, dict)
        if has_route and has_emergency:
            # Two authorities are never additive. One of them must be a
            # fabrication, so neither is trusted.
            raise RouteEvidenceMissing(
                "this round claims both route and emergency authority",
            )
        if has_emergency:
            self._require_emergency_authority(record, emergency_stored)
            return
        if not has_route:
            raise RouteEvidenceMissing("this round has no coding route evidence")
        try:
            route = CodingRouteReceipt.from_mapping(route_stored)
        except ValueError as exc:
            raise RouteEvidenceMissing("coding route evidence is invalid") from exc
        if not route.strict:
            raise RouteEvidenceMissing("coding route evidence is not a passed strict route")
        if not route.ok and not (
            # The single exception, and it grants strictly less than a pass: a
            # strict route whose every lane ran and was trusted, which stopped
            # only because the implementation it wrapped did not close its own
            # required checks, and whose record still says so. Such a job may
            # be read and reworked. It may never be landable or accepted, and
            # the accept path refuses it independently of this check.
            route_blocks_implementation(route)
            and recorded_blockers(record)
            and record.get("landable") is not True
            and str(record.get("state")) != CodingJobState.CODEX_ACCEPTED.value
        ):
            raise RouteEvidenceMissing("coding route evidence is not a passed strict route")
        if record.get("implementer_started") is not True:
            raise RouteEvidenceMissing("this round never started an implementer")

    def _require_emergency_authority(
        self, record: Mapping[str, Any], stored: Mapping[str, Any],
    ) -> None:
        """Validate one emergency authority against this service and this job."""

        if not self.emergency_policy.enabled:
            # Only startup grants this authority. A record that names it on a
            # service without it is evidence of tampering, not permission.
            raise EmergencyAuthorityMissing(
                "this service does not carry emergency overflow authority",
            )
        try:
            authority = EmergencyAuthorityReceipt.from_mapping(stored)
        except EmergencyAuthorityError as exc:
            raise EmergencyAuthorityMissing("emergency authority is invalid") from exc
        if not authority.sealed:
            raise EmergencyAuthorityMissing("emergency authority is not bound to a round")
        if authority.implementer_backend != self.implementation_backend:
            raise EmergencyAuthorityMissing(
                "emergency authority names a different implementer",
            )
        if not authority.checks_enforced or not authority.implementer_started:
            raise EmergencyAuthorityMissing(
                "emergency authority lacks a completed checked implementation",
            )
        # Binding: a receipt lifted from another job's record cannot describe
        # this job's id, request, session, and revision at the same time.
        for field_name, recorded in (
            ("job_id", str(record.get("job_id") or "")),
            ("request_sha256", str(record.get("request_sha256") or "")),
            ("session_id", str(record.get("implementation_session_id") or "")),
            ("revision_sha256", str(record.get("implementation_revision_sha256") or "")),
        ):
            if getattr(authority, field_name) != recorded:
                raise EmergencyAuthorityMissing(
                    "emergency authority does not bind this round",
                )
        if record.get("implementer_started") is not True:
            raise EmergencyAuthorityMissing("this round never started an implementer")

    def _stored_revision(self, record: Mapping[str, Any]) -> str:
        """Recompute the digest of a persisted attributable change set."""

        files = record.get("implementation_files")
        if not isinstance(files, list) or not files:
            raise RevisionUnavailable("the attributable change set is unreadable")
        working_dir = str(record.get("working_dir") or "")
        if not working_dir:
            raise RevisionUnavailable("the implementation workspace is unavailable")
        # Startup authority is re-evaluated before any live read. A service
        # restarted with a narrower allowlist must neither hash nor accept a
        # workspace it is no longer configured to serve.
        self._assert_workspace(working_dir)
        return self._revision_digest(working_dir, [str(item) for item in files])

    @classmethod
    def _revision_digest(cls, working_dir: str, files: Sequence[str]) -> str:
        """Hash the exact current bytes of one bounded attributable change set."""

        root = Path(working_dir).resolve() if working_dir else None
        if root is None or not root.is_dir():
            raise RevisionUnavailable("the implementation workspace is unavailable")
        entries = sorted(set(files))
        if not entries or len(entries) > MAX_ATTRIBUTABLE_FILES:
            raise RevisionUnavailable("the attributable change set is outside the revision bound")
        digest = hashlib.sha256()
        digest.update(_REVISION_DOMAIN)
        total = 0
        for relative in entries:
            target = cls._revision_target(root, relative)
            entry = cls._revision_entry(target, MAX_REVISION_TOTAL_BYTES - total)
            if entry is None:
                # Deletion is part of the revision, not a missing observation.
                digest.update("{}\0absent\n".format(relative).encode("utf-8"))
                continue
            mode, content, size = entry
            total += size
            digest.update("{}\0present\0{}\0{}\n".format(
                relative, mode, content,
            ).encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _revision_entry(
        target: Path, remaining: int,
    ) -> Optional[Tuple[str, str, int]]:
        """Read one file's type, mode, and bytes through a single descriptor.

        Type, size, mode, and content all come from the same open descriptor,
        so a pathname or inode swapped during hashing is detected instead of
        silently mixing two files into one revision. `O_NOFOLLOW` refuses a
        final-component symlink; on platforms without it the lstat/fstat
        identity comparison still fails closed for the same substitution.
        """

        try:
            before = os.lstat(target)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RevisionUnavailable("an attributable path cannot be read") from exc
        if not stat.S_ISREG(before.st_mode):
            raise RevisionUnavailable("an attributable path is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            handle = os.open(target, flags)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise RevisionUnavailable("an attributable path cannot be opened safely") from exc
        try:
            opened = os.fstat(handle)
            if not stat.S_ISREG(opened.st_mode) or (
                (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            ):
                raise RevisionUnavailable("an attributable path changed while it was read")
            if opened.st_size > MAX_REVISION_FILE_BYTES or opened.st_size > remaining:
                raise RevisionUnavailable("attributable content exceeds the revision bound")
            content = hashlib.sha256()
            read = 0
            while True:
                chunk = os.read(handle, _REVISION_CHUNK_BYTES)
                if not chunk:
                    break
                read += len(chunk)
                if read > MAX_REVISION_FILE_BYTES or read > remaining:
                    raise RevisionUnavailable("attributable content exceeds the revision bound")
                content.update(chunk)
            after = os.fstat(handle)
            if read != after.st_size or (
                (after.st_dev, after.st_ino, after.st_size, after.st_mode, after.st_mtime_ns)
                != (
                    opened.st_dev, opened.st_ino, opened.st_size,
                    opened.st_mode, opened.st_mtime_ns,
                )
            ):
                raise RevisionUnavailable("an attributable path changed while it was read")
            return "x" if opened.st_mode & 0o111 else "-", content.hexdigest(), read
        finally:
            os.close(handle)

    @staticmethod
    def _revision_target(root: Path, relative: str) -> Path:
        """Resolve one attributable path or fail closed."""

        if (
            not relative
            or len(relative) > 1024
            or "\x00" in relative
            or "\\" in relative
            or relative.startswith(("/", "~"))
        ):
            raise RevisionUnavailable("an attributable path is not a safe relative path")
        parts = PurePosixPath(relative).parts
        if not parts or any(
            part in {"", ".", ".."} or part in _PROTECTED_REVISION_PARTS or part.startswith(".env")
            for part in parts
        ):
            raise RevisionUnavailable("an attributable path is not a safe relative path")
        target = root.joinpath(*parts)
        if target.is_symlink():
            raise RevisionUnavailable("an attributable path is a symlink")
        resolved = Path(os.path.realpath(target))
        if resolved != root and root not in resolved.parents:
            raise RevisionUnavailable("an attributable path escapes the workspace")
        return target

    def _forget_future(self, future: Future[Any], job_id: str = "") -> None:
        with self._lock:
            self._pending.discard(future)
            if job_id:
                self._release_job_lease(job_id)

    def _assert_workspace(self, workspace: str) -> None:
        target = Path(workspace).resolve()
        if not any(target == root or root in target.parents for root in self.workspace_roots):
            raise WorkspaceDenied("working_dir is outside the configured workspace roots")

    @contextmanager
    def _workspace_lock(self, workspace: str) -> Iterator[None]:
        """Serialize edits to one workspace across threads and MCP processes."""

        resolved = str(Path(workspace).resolve())
        with self._lock:
            local = self._workspace_locks.setdefault(resolved, threading.Lock())
        digest = hashlib.sha256(resolved.encode()).hexdigest()
        path = self.state_root / "locks" / "workspaces" / (digest + ".lock")
        with local:
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    @staticmethod
    def _workspace_digest(workspace: str) -> str:
        return hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()

    def _workspace_claim_path(self, workspace: str) -> Path:
        return self.state_root / "locks" / "workspaces" / (
            self._workspace_digest(workspace) + ".owner.json"
        )

    def _workspace_authority(self, workspace: str) -> Tuple[str, str]:
        """Decide who may edit one worktree: `(status, owner_job_id)`.

        There are exactly three answers, and the third is the reason this
        method exists:

        - `free`: no claim, or a claim whose owning record proves the job has
          settled. Editing is safe.
        - `held`: a claim whose owning record is still in a claim-owned state.
          Only that job may edit.
        - `unresolved`: a claim exists but its authority cannot be evaluated —
          corrupt JSON, an unknown version or shape, an unreadable file, or an
          owning record that cannot be read. Absence of authority is *not*
          proven, so this must never be treated as `free`.

        Deleting an unresolved claim would convert "I cannot tell whether a job
        owns this tree" into "nobody owns this tree", which is precisely the
        concurrent-edit hazard the claim exists to prevent. Such a claim is
        left in place and only a host operator can clear it.
        """

        path = self._workspace_claim_path(workspace)
        if not path.exists():
            return ("free", "")
        try:
            claim = self._read_json(path)
        except FileNotFoundError:
            # Swept between the check and the read; nothing owns the tree.
            return ("free", "")
        except (OSError, ValueError, json.JSONDecodeError):
            return ("unresolved", "")
        digest = self._workspace_digest(workspace)
        if not self._claim_is_well_formed(claim, digest):
            return ("unresolved", "")
        job_id = str(claim["job_id"])
        tenant_ref = str(claim["tenant_ref"])
        owner = self.state_root / "tenants" / tenant_ref / "jobs" / (job_id + ".json")
        try:
            record = self._read_json(owner)
        except FileNotFoundError:
            # The claim names a job this state root has no record of. That is
            # unresolved, not free: a record can be missing because a tenant
            # directory was moved or partially restored, and guessing "free"
            # would hand the tree to a competing job.
            return ("unresolved", job_id)
        except (OSError, ValueError, json.JSONDecodeError):
            return ("unresolved", job_id)
        # The record must bind back to this exact claim and this exact
        # worktree. Without that, a well-formed claim could name an unrelated
        # settled job and read as `free`, which would release a tree whose
        # ownership was never actually evaluated.
        if not self._record_binds_claim(record, job_id, digest):
            return ("unresolved", job_id)
        # A record only settles ownership by naming a state this service
        # actually defines. A missing, unknown, or wrong-typed state is not
        # evidence that the job finished — treating "not claim-owned" as free
        # would release a worktree on the strength of an unreadable field.
        state = record.get("state")
        if isinstance(state, bool) or not isinstance(state, str):
            return ("unresolved", job_id)
        try:
            CodingJobState(state)
        except ValueError:
            return ("unresolved", job_id)
        if state in _CLAIM_OWNED_STATES:
            return ("held", job_id)
        return ("free", job_id)

    @staticmethod
    def _claim_is_well_formed(claim: Mapping[str, Any], digest: str) -> bool:
        """Require the exact claim shape, bounded values, and this worktree.

        Missing keys are rejected as firmly as unknown ones: a partially
        written claim proves nothing about ownership, and treating it as
        absent would be exactly the fail-open this guards against.
        """

        if set(claim) != _WORKSPACE_CLAIM_FIELDS:
            return False
        if claim.get("claim_version") != WORKSPACE_CLAIM_VERSION:
            return False
        if not _JOB_ID.fullmatch(str(claim.get("job_id") or "")):
            return False
        if not _TENANT_REF.fullmatch(str(claim.get("tenant_ref") or "")):
            return False
        # A claim is keyed by workspace digest, so a claim whose recorded
        # digest disagrees with the tree being queried is not about this tree.
        if str(claim.get("workspace_sha256") or "") != digest:
            return False
        if str(claim.get("state") or "") not in _CLAIM_OWNED_STATES:
            return False
        if not _SAFE_ID.fullmatch(str(claim.get("instance_id") or "")):
            return False
        process_id = claim.get("process_id")
        if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id < 0:
            return False
        for name in ("claimed_at", "updated_at"):
            value = claim.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            # "Bounded" has to mean finite. `NaN` fails every comparison and
            # `Infinity` passes them all, so neither is a usable timestamp.
            if not math.isfinite(value) or value < 0:
                return False
        return True

    def _record_binds_claim(
        self, record: Mapping[str, Any], job_id: str, digest: str,
    ) -> bool:
        """Require the owner record to name this job and this exact worktree."""

        if str(record.get("job_id") or "") != job_id:
            return False
        if str(record.get("workspace_sha256") or "") != hashlib.sha256(
            str(record.get("working_dir") or "").encode(),
        ).hexdigest():
            # The record's own two spellings of its workspace must agree before
            # either is trusted to answer for a claim.
            return False
        working_dir = str(record.get("working_dir") or "")
        if not working_dir:
            return False
        try:
            return self._workspace_digest(working_dir) == digest
        except (OSError, ValueError, RuntimeError):
            return False

    def _assert_workspace_available(self, job_id: str, workspace: str) -> str:
        """Refuse a worktree owned by another job or by an unevaluable claim.

        Every job runs this check, including a legacy service that takes no
        claim of its own. Honouring a claim is what protects an audit gap;
        taking one is only needed by a job that will have such a gap.
        """

        status, owner_job_id = self._workspace_authority(workspace)
        if status == "unresolved":
            raise WorkspaceClaimUnresolved(
                "a coding workspace claim cannot be evaluated and needs host repair",
                owner_job_id,
            )
        if status == "held" and owner_job_id != job_id:
            raise WorkspaceBusy(
                "another coding job owns this workspace until its audit closes",
                owner_job_id,
            )
        return status

    def _create_workspace_claim(
        self, tenant_ref: str, job_id: str, workspace: str, state: str,
    ) -> None:
        """Take a first hold for a job that does not exist yet.

        This is the only path that may turn `free` into ownership, and it runs
        exactly once per job: inside `submit`, under the state guard, before
        the job record is published. At that moment `free` genuinely means "no
        job owns this tree", because this job has no history to be continuous
        with.
        """

        self._assert_workspace_available(job_id, workspace)
        now = time.time()
        self._write_claim(tenant_ref, job_id, workspace, state, claimed_at=now, now=now)

    def _reassert_workspace_claim(
        self, tenant_ref: str, job_id: str, workspace: str, state: str,
    ) -> None:
        """Require an existing hold this exact job already owns, and restate it.

        A live job may never recreate a claim from `free`. If the claim for an
        existing queued, running, awaiting, or rework job has disappeared, the
        service cannot prove that no other Codex acquired this worktree, edited
        unrelated files, settled, and released in the gap. Recomputing only
        *this* job's attributable files would not detect that, so reacquiring
        here would manufacture continuity that was never established.

        Missing is therefore `unresolved`, exactly like corrupt: it needs a
        host decision (`code-release --abandon-job` or `--repair-workspace`),
        not an automatic repair.
        """

        claim = self._require_owned_claim(tenant_ref, job_id, workspace)
        self._write_claim(
            tenant_ref, job_id, workspace, state,
            claimed_at=float(claim["claimed_at"]), now=time.time(),
        )

    def _require_owned_claim(
        self, tenant_ref: str, job_id: str, workspace: str,
    ) -> Mapping[str, Any]:
        """Return the live, fully bound claim this exact tenant+job owns."""

        status, owner_job_id = self._workspace_authority(workspace)
        if status == "unresolved":
            raise WorkspaceClaimUnresolved(
                "a coding workspace claim cannot be evaluated and needs host repair",
                owner_job_id,
            )
        if status == "free":
            raise WorkspaceClaimUnresolved(
                "this coding job no longer holds a workspace claim it can prove",
                job_id,
            )
        if owner_job_id != job_id:
            raise WorkspaceBusy(
                "another coding job owns this workspace until its audit closes",
                owner_job_id,
            )
        claim = self._read_json(self._workspace_claim_path(workspace))
        if str(claim.get("tenant_ref") or "") != tenant_ref:
            # Same job id, different tenant namespace. Never overwrite that as
            # this tenant's hold, and do not confirm the other tenant's job.
            raise WorkspaceClaimUnresolved(
                "a coding workspace claim belongs to another tenant namespace",
            )
        return claim

    def _write_claim(
        self,
        tenant_ref: str,
        job_id: str,
        workspace: str,
        state: str,
        *,
        claimed_at: float,
        now: float,
    ) -> None:
        self._write_json(self._workspace_claim_path(workspace), {
            "claim_version": WORKSPACE_CLAIM_VERSION,
            "job_id": job_id,
            "tenant_ref": tenant_ref,
            "workspace_sha256": self._workspace_digest(workspace),
            "state": str(state),
            "instance_id": self.instance_id,
            "process_id": os.getpid(),
            "claimed_at": claimed_at,
            "updated_at": now,
        })

    def _release_workspace_claim(self, job_id: str, workspace: str) -> None:
        """Drop this job's hold, never another job's and never an unreadable one."""

        if not workspace:
            return
        path = self._workspace_claim_path(workspace)
        try:
            claim = self._read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return
        # A malformed or foreign claim is somebody else's problem to repair.
        # Releasing one here would let a settling job clear ownership it never
        # actually held.
        if not self._claim_is_well_formed(claim, self._workspace_digest(workspace)):
            return
        if str(claim["job_id"]) == job_id:
            self._discard_path(path)

    def _sweep_workspace_claims(self) -> None:
        """Drop only the claims whose owning record proves the job has settled.

        A claim that cannot be evaluated survives the sweep on purpose. Startup
        is exactly when a half-written state root is most likely, and clearing
        an unreadable claim here would silently reopen a worktree whose audit
        may still be pending.
        """

        directory = self.state_root / "locks" / "workspaces"
        if not directory.is_dir():
            return
        for path in directory.glob("*.owner.json"):
            try:
                claim = self._read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            digest = str(claim.get("workspace_sha256") or "") if isinstance(
                claim, Mapping,
            ) else ""
            # The sweep answers the same question as `_workspace_authority`, so
            # it must apply the same bindings. A claim it cannot fully evaluate
            # is left exactly where it is.
            if not self._claim_is_well_formed(claim, digest):
                continue
            if path.name != digest + ".owner.json":
                # The filename is the lookup key, so a claim stored under a
                # different digest than it declares is not evaluable here.
                continue
            job_id = str(claim["job_id"])
            owner = (
                self.state_root / "tenants" / str(claim["tenant_ref"])
                / "jobs" / (job_id + ".json")
            )
            try:
                record = self._read_json(owner)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not self._record_binds_claim(record, job_id, digest):
                continue
            if str(record.get("state")) not in _CLAIM_OWNED_STATES:
                self._discard_path(path)

    def _resume_path(self, tenant_ref: str, job_id: str) -> Path:
        if not _JOB_ID.fullmatch(job_id):
            raise CodingServiceError("coding resume envelope id is invalid")
        return self.state_root / "tenants" / tenant_ref / "resume" / (job_id + ".json")

    def _write_resume_envelope(
        self,
        tenant_ref: str,
        job_id: str,
        request: CodingTaskRequest,
        request_sha256: str,
        session_bound: str = "",
    ) -> None:
        """Persist only what one job's rework may replay, and nothing more.

        Startup authority — approval policy, sandbox mode, config path, sandbox
        image, checks, capabilities — is deliberately absent. A rework round
        re-imposes those from the running service, so a stale envelope can never
        widen the authority a later process grants an implementer.
        """

        payload: Dict[str, Any] = {
            "envelope_version": RESUME_ENVELOPE_VERSION,
            "job_id": job_id,
            "request_sha256": request_sha256,
            "session_bound": str(session_bound or ""),
            "created_at": time.time(),
        }
        for field in sorted(_ALLOWED_REQUEST_FIELDS):
            payload[field] = getattr(request, field)
        # `thread_id` stays exactly `None` when absent rather than becoming an
        # empty string: the request contract accepts a safe identifier or
        # nothing at all, and `""` is neither.
        payload["thread_id"] = request.thread_id or None
        self._write_json(self._resume_path(tenant_ref, job_id), payload)

    def _load_resume_request(
        self,
        tenant_ref: str,
        job_id: str,
        request_sha256: str,
        session_bound: Optional[str] = None,
    ) -> Optional[CodingTaskRequest]:
        """Rebuild one job's original request from its durable envelope.

        Two bindings always hold: the envelope names this job, and it carries
        the digest the job record already recorded. The stored digest is
        compared rather than recomputed, because the persisted prose has passed
        through redaction and no longer hashes to the original. `session_bound`
        adds the third binding when a caller demands one exact session; passing
        `None` reads the envelope for resealing, never for execution.
        """

        try:
            envelope = self._read_json(self._resume_path(tenant_ref, job_id))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if (
            envelope.get("envelope_version") != RESUME_ENVELOPE_VERSION
            or set(envelope) - _RESUME_ENVELOPE_FIELDS
            or str(envelope.get("job_id") or "") != job_id
            or str(envelope.get("request_sha256") or "") != str(request_sha256 or "")
            or (
                session_bound is not None
                and str(envelope.get("session_bound") or "") != session_bound
            )
        ):
            return None
        try:
            return request_from_mapping({
                field: envelope[field]
                for field in _ALLOWED_REQUEST_FIELDS
                if field in envelope
            })
        except (ValueError, TypeError):
            return None

    def _read_resume_envelope(
        self, tenant_ref: str, job_id: str, record: Mapping[str, Any],
    ) -> Optional[CodingTaskRequest]:
        """Return the request a rework may replay into this job's exact session."""

        session = str(record.get("implementation_session_id") or "")
        if not session:
            return None
        return self._load_resume_request(
            tenant_ref,
            job_id,
            str(record.get("request_sha256") or ""),
            session_bound=session,
        )

    def _seal_resume_envelope(
        self,
        tenant_ref: str,
        job_id: str,
        request: CodingTaskRequest,
        request_sha256: str,
        session: str,
    ) -> None:
        """Bind the stored request to the session this round actually produced.

        The stored request wins over the caller's, because a rework round's
        request carries audit feedback prepended to the task. Resealing it
        would compound that feedback on every round; the envelope must keep
        describing the original task for the life of the job.
        """

        stored = self._load_resume_request(tenant_ref, job_id, request_sha256)
        self._write_resume_envelope(
            tenant_ref, job_id, stored or request, request_sha256,
            session_bound=session,
        )

    def _discard_resume(self, tenant_ref: str, job_id: str) -> None:
        """Forget one job's resume context in memory and on disk together."""

        self._resume.pop((tenant_ref, job_id), None)
        try:
            self._discard_path(self._resume_path(tenant_ref, job_id))
        except CodingServiceError:
            return

    @staticmethod
    def _discard_path(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            return

    @contextmanager
    def _state_guard(self) -> Iterator[None]:
        """Protect short state mutations without owning the root for process life."""

        with self._lock:
            outermost = self._state_lock_depth == 0
            if outermost and fcntl is not None:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            self._state_lock_depth += 1
            try:
                yield
            finally:
                self._state_lock_depth -= 1
                if outermost and fcntl is not None:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _job_lease_path(self, job_id: str) -> Path:
        if not _JOB_ID.fullmatch(job_id):
            raise CodingServiceError("coding job lease id is invalid")
        return self.state_root / "locks" / "jobs" / (job_id + ".lock")

    def _acquire_job_lease(self, job_id: str) -> bool:
        """Claim one execution round; leases are released automatically on crash."""

        if job_id in self._job_leases:
            return False
        fd = os.open(self._job_lease_path(job_id), os.O_CREAT | os.O_RDWR, 0o600)
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                return False
        self._job_leases[job_id] = fd
        return True

    def _release_job_lease(self, job_id: str) -> None:
        fd = self._job_leases.pop(job_id, None)
        if fd is None:
            return
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    @staticmethod
    def _tenant_ref(tenant_id: str) -> str:
        if not _SAFE_ID.fullmatch(tenant_id):
            raise ValueError("tenant_id must be a safe identifier")
        return hashlib.sha256(tenant_id.encode()).hexdigest()

    def _tenant_dir(self, tenant_ref: str, *, create: bool = True) -> Path:
        path = self.state_root / "tenants" / tenant_ref
        if create:
            (path / "jobs").mkdir(parents=True, exist_ok=True, mode=0o700)
            (path / "idempotency").mkdir(exist_ok=True, mode=0o700)
        return path

    @staticmethod
    def _request_digest(request: CodingTaskRequest) -> str:
        """Identity of what the *caller* asked for, host authority excluded.

        `authorized_config_sha256` is decided by the service, not the caller,
        and it is decided after this digest has already been used to look up an
        idempotency record. Hashing it would make an identical resubmission
        collide with its own earlier record whenever the repository contract
        had changed in between - an idempotent replay would raise a conflict
        instead of returning the receipt it was replaying.
        """

        fields = dataclasses.asdict(request)
        fields.pop("authorized_config_sha256", None)
        # An absent mission is dropped rather than hashed as `null`. A caller
        # that never named one must keep digesting to exactly the value it did
        # before this field existed, or every stored idempotency record from
        # before the envelope would stop matching its own replay.
        if fields.get("mission") is None:
            fields.pop("mission", None)
        payload = json.dumps(
            fields, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), default=lambda value: value.value,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("coding service record must be an object")
        return value

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = json.dumps(redact_evidence(dict(value)), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        fd, temporary = tempfile.mkstemp(prefix=".job-", suffix=".tmp", dir=str(path.parent))
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def _update_record(self, path: Path, **changes: Any) -> None:
        with self._state_guard():
            self._update_record_locked(path, **changes)

    def _update_record_locked(self, path: Path, **changes: Any) -> None:
        """Update one record while the caller owns the cross-process guard."""

        record = self._read_json(path)
        record.update(changes)
        record["updated_at"] = time.time()
        job_id = str(record.get("job_id") or "")
        state = str(record.get("state"))
        workspace = str(record.get("working_dir") or "")
        claims = bool(job_id and workspace and self.require_codex_audit)
        # Ownership is asserted *before* the record is published, inside the
        # guard the caller already holds. A claim-owned state is a promise that
        # this job exclusively owns the worktree, so a transition that cannot
        # prove that promise must fail rather than become visible. Logging it
        # and continuing would let a round reach `awaiting_codex_audit` with no
        # valid claim, which is the concurrent-edit window this exists to shut.
        if claims and state in _CLAIM_OWNED_STATES:
            # `.../tenants/<tenant_ref>/jobs/<job_id>.json`. This is a
            # reassertion, never a creation: only `submit` may claim a free
            # worktree, so a live job whose claim vanished fails closed instead
            # of silently taking the tree back.
            self._reassert_workspace_claim(
                path.parent.parent.name, job_id, workspace, state,
            )
        self._write_json(path, record)
        self._publish_status(record)
        # A settled record publishes first, then releases. Release only ever
        # removes this job's own valid claim; a foreign or unresolved one is
        # left for a host operator.
        if claims and state not in _CLAIM_OWNED_STATES:
            self._release_workspace_claim(job_id, workspace)
        # Publish the settled record before releasing its execution lease.
        # A second process therefore cannot observe an auditable/terminal
        # state while the previous round still appears to own the job.
        if state not in _INTERRUPTED_JOB_STATES:
            self._release_job_lease(job_id)

    @staticmethod
    def _decode_result(value: Any) -> Optional[CodingTaskResult]:
        if not isinstance(value, dict):
            return None
        data = dict(value)
        data["checks"] = [CheckResult(**item) for item in data.get("checks", [])]
        statuses = []
        for item in data.get("capabilities", []):
            item = dict(item)
            item["tools"] = tuple(item.get("tools", ()))
            item["missing_tools"] = tuple(item.get("missing_tools", ()))
            statuses.append(CapabilityStatus(**item))
        data["capabilities"] = statuses
        return CodingTaskResult(**data)

    @staticmethod
    def _bounded_generation(value: Any) -> int:
        """A generation is a small counter or it is nothing."""

        if isinstance(value, bool) or not isinstance(value, int):
            return 0
        if value < 0 or value > MAX_CONTINUATION_GENERATION:
            return 0
        return value

    def _public_receipt(
        self, tenant_ref: str, record: Mapping[str, Any],
    ) -> CodingJobReceipt:
        """Project one job, including whether anything is left to continue.

        `continuation_available` is answered from the authority itself rather
        than from a flag on the record, because the record cannot know when a
        successor consumed it. A stale `True` would invite a caller to resume
        something already spent, which is exactly the wrong direction to be
        wrong in. The authority record, its generation history, and the
        canonical workspace path stay private; only the boolean crosses.
        """

        receipt = self._receipt(record)
        session = str(record.get("continuation_session_id") or "")
        if not is_continuable_session(session):
            return receipt
        try:
            authority = self._continuation.open_authority(tenant_ref, session)
        except (OSError, ValueError):
            return receipt
        if authority is None:
            return receipt
        return dataclasses.replace(
            receipt,
            continuation_available=authority.generation < MAX_CONTINUATION_GENERATION,
            continuation_generation=authority.generation,
        )

    @classmethod
    def _receipt(cls, record: Mapping[str, Any]) -> CodingJobReceipt:
        return CodingJobReceipt(
            continuation_generation=cls._bounded_generation(
                record.get("continuation_generation"),
            ),
            job_id=str(record["job_id"]),
            state=CodingJobState(str(record["state"])),
            submitted_at=float(record["submitted_at"]),
            updated_at=float(record["updated_at"]),
            thread_id=str(record.get("thread_id") or ""),
            evidence_sha256=str(record.get("evidence_sha256") or ""),
            result=cls._decode_result(record.get("result")),
            failure_code=str(record["failure_code"]) if record.get("failure_code") else None,
            implementation_backend=str(record.get("implementation_backend") or ""),
            implementation_session_id=str(record.get("implementation_session_id") or ""),
            implementation_revision_sha256=str(record.get("implementation_revision_sha256") or ""),
            audit_count=int(record.get("audit_count") or 0),
            rework_count=int(record.get("rework_count") or 0),
            audit_findings_sha256=str(record.get("audit_findings_sha256") or ""),
            landable=record.get("landable") is True,
            implementer_started=record.get("implementer_started") is True,
            implementation_blockers=recorded_blockers(record),
            # Re-validated on the way out too. A record written by an older
            # build, or edited, cannot put anything but an identifier here.
            verification_blockers=safe_blockers(record.get("verification_blockers")),
            route_receipt=(
                record["route_receipt"]
                if isinstance(record.get("route_receipt"), dict) else None
            ),
            emergency_authority=(
                record["emergency_authority"]
                if isinstance(record.get("emergency_authority"), dict) else None
            ),
            # Only the bounded, secret-safe projection crosses. The receipt
            # revalidates it through the closed field set on the way in, so a
            # record edited in place cannot publish a mission field this
            # contract does not name - and prose, coordinates, evidence values,
            # worker identity and worktree paths have no field to ride out on.
            mission=(
                record["mission"]
                if isinstance(record.get("mission"), dict) else None
            ),
        )

    def _reconcile_interrupted_jobs(self) -> None:
        tenants = self.state_root / "tenants"
        if tenants.is_dir():
            for path in tenants.glob("*/jobs/job_*.json"):
                try:
                    record = self._read_json(path)
                    # An awaiting-audit job survives a restart; in-flight work
                    # does not, but another live MCP process may still own its
                    # job lease.
                    if record.get("state") in _INTERRUPTED_JOB_STATES:
                        job_id = str(record.get("job_id") or "")
                        if not self._acquire_job_lease(job_id):
                            continue
                        try:
                            record.update({
                                "state": CodingJobState.FAILED.value,
                                "updated_at": time.time(),
                                "landable": False,
                                "failure_code": "service_restarted",
                            })
                            self._write_json(path, record)
                            self._publish_status(record)
                            # `.../tenants/<tenant_ref>/jobs/<job_id>.json`
                            self._discard_resume(path.parent.parent.name, job_id)
                            self._reclaim_mission_item(path, record)
                        finally:
                            self._release_job_lease(job_id)
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
        # Claims outlive the process that took them on purpose, so a crash
        # cannot expose a worktree whose audit has not happened yet. Sweeping
        # afterwards keeps that from becoming a permanent pin, but only for
        # ownership this pass can actually evaluate: a claim is dropped only
        # when it is well formed, bound to its own worktree, and its owning
        # record binds back and proves the job settled — including the records
        # this pass just failed closed. A missing, unreadable, unbound, or
        # otherwise unresolved claim is deliberately preserved for a host
        # operator, because discarding it would reopen a worktree whose audit
        # may still be pending.
        self._sweep_workspace_claims()
        self._reconcile_continuation_claims()
        # One pump per item this pass returned to the queue, so a restart's
        # accounting happens through the ordinary store-ordered route rather
        # than through a second, privileged path that could close work the
        # scheduler still believes is running.
        for _ in range(self._reclaimed):
            self._prime_pump()
        self._reclaimed = 0

    def _reclaim_mission_item(self, path: Path, record: Mapping[str, Any]) -> None:
        """Return one interrupted job's work item to the queue, on proof.

        Proof, never age. The kernel reclaims a dispatch exactly when it can
        take that item's execution lease itself, which is impossible while any
        live process holds it - so a worker that is slow, paused, or simply on
        the other side of a restart is never stolen from, and no heartbeat gap
        or TTL is consulted at any point. A live holder answers
        `MissionConflictRefused` and this pass leaves the item exactly where it
        found it.

        A reclaimed item goes back to `ready`, and its accounting happens when a
        pump next offers it: the owning record is terminal by then, so it closes
        deferred with the full rationale rather than running a job that already
        failed closed.
        """

        projection = self._record_projection(record)
        if projection is None:
            return
        item = self._mission.work_item(projection.work_item_id)
        if item is None or item.status != MISSION_STATUS_DISPATCHED:
            return
        try:
            reclaimed = self._mission.reclaim(projection.work_item_id)
        except MissionConflictRefused:
            # A live worker still holds this lease. It keeps it.
            return
        except MissionRouteError:
            return
        if not reclaimed:
            return
        self._advance_projection(path, status=MISSION_STATUS_READY)
        self._reclaimed += 1

    def _reconcile_continuation_claims(self) -> None:
        """Resolve authorities whose claiming job died before it could settle.

        A crash between "the authority is claimed" and "the job recorded an
        outcome" is the one window where a continuation could be pinned forever.
        The pin is deliberate while a job might still be live -- releasing it
        early is how two workers would enter the same session -- so this pass
        only acts once the claiming job has provably stopped, and it decides
        from the claimant's own durable record rather than from a timer:

        * the record is gone, or names another workspace, or cannot be read:
          the authority is settled. Nothing can prove what that job did, and a
          continuation nobody can attribute is not one this host will grant.
        * the record settled without producing a newer generation: settled, for
          the same reason its own terminal path would have settled it.
        * the record is still in an in-flight state after a restart: that state
          was already failed closed above, so it settles here too.

        Reopening is never done. An authority that returns to `open` after an
        unexplained death would let a second worker re-enter a session whose
        first attempt may have already spent it, which is exactly the duplicate
        provider call this whole mechanism exists to prevent. The operator's
        route back is a fresh job, and `continuation_available` reports `False`
        truthfully rather than offering a resume that might double-spend.
        """

        root = self.state_root / "tenants"
        if not root.is_dir():
            return
        for path in root.glob("*/continuation/*.json"):
            tenant_ref = path.parent.parent.name
            try:
                # The file name is a digest, so the session can only come from
                # the body. That body is then re-read through the store, which
                # subjects this pass to the same journal agreement every other
                # reader gets: a replayed authority resolves to `None` here and
                # is left exactly where it is.
                claimed_session = str(
                    json.loads(path.read_text(encoding="utf-8")).get("session_id") or "",
                )
                if not is_continuable_session(claimed_session):
                    continue
                authority = self._continuation.load(tenant_ref, claimed_session)
            except (ContinuationCorrupt, OSError, ValueError, AttributeError):
                continue
            if authority is None or authority.state != STATE_CLAIMED:
                continue
            claimant = str(authority.claimed_by_job_id or "")
            if not _JOB_ID.fullmatch(claimant):
                self._force_settle(authority)
                continue
            record_path = root / tenant_ref / "jobs" / (claimant + ".json")
            try:
                record = self._read_json(record_path)
            except (OSError, ValueError):
                self._force_settle(authority)
                continue
            if str(record.get("state")) in _CLAIM_OWNED_STATES:
                # Still, as far as this pass can tell, a live or audit-ready
                # job. Its own settlement path owns the authority.
                continue
            self._force_settle(authority)

    def _force_settle(self, authority: ContinuationAuthority) -> None:
        """Settle forward. Reconciliation never rewinds the journal."""

        try:
            self._continuation.commit(authority, authority.settled(time.time()))
        except (ContinuationConflict, ContinuationCorrupt, OSError, ValueError):
            return
