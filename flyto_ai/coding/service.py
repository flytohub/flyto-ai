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

from flyto_ai.coding.contracts import (
    MAX_AUDIT_ROUNDS,
    ApprovalPolicy,
    CapabilityStatus,
    CheckResult,
    CodingAuditFinding,
    CodingAuditVerdict,
    CodingJobReceipt,
    CodingJobState,
    CodingTaskRequest,
    CodingTaskResult,
    SandboxMode,
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
from flyto_ai.coding.route import (
    ROUTE_THREAD_PREFIX,
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
from flyto_ai.coding.store import ThreadStore, redact_evidence


try:  # pragma: no cover - Windows fallback is exercised by static review.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_JOB_ID = re.compile(r"^job_[a-f0-9]{24}$")
_TENANT_REF = re.compile(r"^[a-f0-9]{64}$")
#: Structured error context is a closed vocabulary of short opaque tokens.
#: Paths, prose, and anything unbounded fail these patterns and are dropped.
_DETAIL_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
_DETAIL_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MAX_ERROR_DETAIL_FIELDS = 8
_BACKEND_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ALLOWED_REQUEST_FIELDS = frozenset({
    "message", "working_dir", "thread_id", "resume", "max_attempts",
    "max_rounds", "require_changes",
})
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
PROVISIONAL_THREAD_PREFIXES = (ROUTE_THREAD_PREFIX, "host-")
#: How one round was executed. Persisted before the implementer is invoked, so
#: an in-flight or crashed overflow round is never read back as strict.
EXECUTION_MODE_STRICT = "strict"
EXECUTION_MODE_EMERGENCY = "emergency"
#: Closed schema tokens for the two durable records this module owns beside the
#: job record. A file that does not name its exact version is not read.
WORKSPACE_CLAIM_VERSION = "flyto.coding-workspace-claim.v1"
RESUME_ENVELOPE_VERSION = "flyto.coding-resume-envelope.v1"
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
    """Base error with a stable, non-sensitive service code."""

    code = "service_error"

    @property
    def details(self) -> Dict[str, Any]:
        """Bounded, closed-schema context a facade may publish beside `code`.

        The default is deliberately empty: an error is a stable code first, and
        only a subclass that has something safe and actionable to add may fill
        this in. Paths, prose, and credentials never appear here.
        """
        return {}


class CodingServiceBusy(CodingServiceError):
    code = "service_busy"


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

    def __init__(self, message: str, owner_job_id: str = "") -> None:
        super().__init__(message)
        self.owner_job_id = str(owner_job_id or "")

    @property
    def details(self) -> Dict[str, Any]:
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

    def __init__(self, message: str, owner_job_id: str = "") -> None:
        super().__init__(message)
        self.owner_job_id = str(owner_job_id or "")

    @property
    def details(self) -> Dict[str, Any]:
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


class ReworkLimitReached(CodingServiceError):
    code = "rework_limit_reached"


class ReworkNotResumable(CodingServiceError):
    code = "rework_not_resumable"


class RouteEvidenceMissing(CodingServiceError):
    code = "route_evidence_missing"


class EmergencyAuthorityMissing(CodingServiceError):
    """An emergency-bound job cannot be served without its authority contract."""

    code = "emergency_authority_missing"


#: States whose persisted route evidence must still hold when read back.
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


class _RoundProgress:
    """What one execution round actually did, tracked by the host itself.

    `begin()` runs immediately before the selected implementer is invoked,
    never because a job entered `running`. It writes the durable record first
    and the in-memory flag second, so a process that dies while the model is
    working still leaves a job record that says the implementer started.
    """

    def __init__(self, on_start: Optional[Callable[[], None]] = None) -> None:
        self.implementer_started = False
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
    unknown = set(value) - _ALLOWED_REQUEST_FIELDS
    if unknown:
        raise ValueError("unsupported coding request fields: {}".format(", ".join(sorted(unknown))))
    return CodingTaskRequest(
        message=str(value.get("message", "")),
        working_dir=str(value.get("working_dir", "")),
        thread_id=str(value["thread_id"]) if value.get("thread_id") is not None else None,
        resume=bool(value.get("resume", False)),
        max_attempts=int(value.get("max_attempts", 3)),
        max_rounds=int(value.get("max_rounds", 30)),
        require_changes=bool(value.get("require_changes", True)),
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
    return projected


def receipt_to_mapping(receipt: CodingJobReceipt) -> Dict[str, Any]:
    """Return a JSON-safe, secret-redacted public receipt."""

    projected = dataclasses.asdict(receipt)
    result = projected.get("result")
    if isinstance(result, dict):
        result.pop("evidence_path", None)
        for check in result.get("checks", []):
            if isinstance(check, dict):
                check.pop("output_preview", None)
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
        self.state_root = Path(state_root).expanduser().resolve()
        self.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
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
        with self._state_guard():
            if self._closed:
                raise CodingServiceError("coding service is closed")
            if idempotency_path.exists():
                reference = self._read_json(idempotency_path)
                if reference.get("request_sha256") != request_digest:
                    raise IdempotencyConflict("idempotency key was already used for another request")
                referenced = str(reference.get("job_id", ""))
                if not _JOB_ID.fullmatch(referenced):
                    raise CodingServiceError("idempotency record is invalid")
                return self._receipt(
                    self._read_json(tenant_dir / "jobs" / (referenced + ".json")),
                )
            if current_service_build_id() != self.build_id:
                # Never begin a fresh job with modules imported from a
                # different build than the files an auditor will inspect.
                # An idempotent retry above remains readable, and existing jobs
                # may still be polled/audited so their exact session can close.
                raise CodingServiceReloadRequired(
                    "coding service source changed; reload the MCP worker",
                )
            if len(self._pending) >= self.max_queued:
                raise CodingServiceBusy("coding job queue is full")
            now = time.time()
            job_id = "job_{}".format(uuid.uuid4().hex[:24])
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
                "landable": False,
            }
            try:
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
                future = self._executor.submit(self._run_job, tenant_ref, job_id, request)
            except BaseException:
                self._release_workspace_claim(job_id, request.working_dir)
                self._discard_resume(tenant_ref, job_id)
                self._release_job_lease(job_id)
                raise
            self._pending.add(future)
            future.add_done_callback(
                lambda completed, claimed=job_id: self._forget_future(completed, claimed),
            )
            return self._receipt(record)

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
            approval_policy=self.approval_policy,
            sandbox_mode=self.sandbox_mode,
            checks=(),
            capabilities=(),
            config_path=self.config_path,
            command_sandbox_image=self.sandbox_image,
        )

    def get(self, tenant_id: str, job_id: str) -> CodingJobReceipt:
        """Read a job only from the authenticated tenant namespace."""

        if not _JOB_ID.fullmatch(job_id):
            raise CodingJobNotFound("coding job does not exist")
        path = self._tenant_dir(self._tenant_ref(tenant_id), create=False) / "jobs" / (job_id + ".json")
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
            return self._receipt(record)

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
                return self._receipt(self._read_json(path))
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
            return self._receipt(self._read_json(path))

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
        request = self._with_startup_authority(dataclasses.replace(
            original,
            message=self._rework_message(original.message, findings),
            thread_id=session,
            resume=True,
        ))
        if len(self._pending) >= self.max_queued:
            raise CodingServiceBusy("coding job queue is full")
        # Claim the job before queueing. A concurrent audit then observes a
        # non-awaiting state and cannot schedule a duplicate rework round.
        if not self._acquire_job_lease(job_id):
            raise CodingServiceBusy("coding job is already being executed")
        try:
            self._update_record_locked(
                path,
                state=CodingJobState.REWORK_QUEUED.value,
                audit_count=audit_count,
                rework_count=rework_count,
                audit_findings_sha256=digest,
                landable=False,
                failure_code=None,
            )
        except BaseException:
            # The transition was refused — most often because this job can no
            # longer prove it owns the worktree. Hand back the lease so the job
            # stays exactly as auditable as it was, rather than looking busy to
            # every later caller.
            self._release_job_lease(job_id)
            raise
        try:
            future = self._executor.submit(self._run_job, tenant_ref, job_id, request, rework=True)
        except RuntimeError as exc:
            self._update_record_locked(
                path,
                state=CodingJobState.FAILED.value,
                failure_code="service_rework_not_scheduled",
            )
            self._release_job_lease(job_id)
            raise CodingServiceError("coding rework could not be scheduled") from exc
        self._pending.add(future)
        future.add_done_callback(
            lambda completed, claimed=job_id: self._forget_future(completed, claimed),
        )
        return self._receipt(self._read_json(path))

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
    ) -> None:
        path = self._tenant_dir(tenant_ref) / "jobs" / (job_id + ".json")
        workspace_lock = self._workspace_lock(request.working_dir)
        progress = _RoundProgress(
            on_start=lambda: self._mark_implementer_started(path),
        )
        with workspace_lock:
            try:
                self._update_record(path, state=(
                    CodingJobState.REWORK_RUNNING if rework else CodingJobState.RUNNING
                ).value)
                store = ThreadStore(str(self._tenant_dir(tenant_ref) / "threads"))
                result, route, authority = asyncio.run(
                    self._run_round(
                        store, request, job_id, path, progress, rework=rework,
                    ),
                )
                self._record_outcome(
                    path, tenant_ref, job_id, request, result, store, rework, route,
                    progress=progress, authority=authority,
                )
            except CodingServiceError as exc:
                self._fail_job(path, tenant_ref, job_id, exc.code, exc, progress)
            except BaseException as exc:  # noqa: BLE001 - a worker never leaks
                # Every exit from a worker must leave a terminal record. An
                # unobserved future exception would strand the job `running`
                # forever, which is exactly what a fail-closed service must
                # never do.
                self._fail_job(
                    path, tenant_ref, job_id, "service_execution_failed", exc, progress,
                )
                if not isinstance(exc, Exception):
                    raise

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

        result, route = await self._implement(store, request, progress)
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
        progress.begin()
        result = await self.agent_factory(store).run(request)
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
    ) -> "tuple[CodingTaskResult, Optional[CodingRouteReceipt]]":
        """Run the selected implementer, wrapped by the host-owned route.

        Without a strict startup policy this is exactly the historical direct
        call, so existing library callers keep their behavior. The public
        `code-mcp` / `code-serve` builders always enable the strict route.
        """

        policy = self.route_policy
        if policy is None or not policy.strict:
            if progress is not None:
                progress.begin()
            return await self.agent_factory(store).run(request), None

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
                if progress is not None:
                    # The host records the start durably here, between the
                    # pre-lanes passing and the model actually being called.
                    progress.begin()
                return await self.agent_factory(store).run(effective)

            outcome = await orchestrator.run(request, implement)
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
    ) -> None:
        """Move one finished implementation round into its durable state."""

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
        }
        if started and self.require_codex_audit:
            outcome["implementation_backend"] = self.implementation_backend
        if not result.ok:
            self._discard_resume(tenant_ref, job_id)
            self._update_record(
                path, state=CodingJobState.FAILED.value, landable=False,
                **self._failed_round_proof(request, result, started, outcome),
            )
            return
        if not self.require_codex_audit:
            self._discard_resume(tenant_ref, job_id)
            self._update_record(path, state=CodingJobState.COMPLETED.value, **outcome)
            return
        session = str(result.thread_id or "")
        if not session:
            raise SessionBindingFailed("the implementation session id is missing")
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
        self._update_record(
            path,
            state=CodingJobState.AWAITING_CODEX_AUDIT.value,
            implementation_session_id=session,
            implementation_revision_sha256=revision,
            implementation_files=files,
            working_dir=request.working_dir,
            landable=False,
            **outcome,
        )

    def _failed_round_proof(
        self,
        request: CodingTaskRequest,
        result: CodingTaskResult,
        started: bool,
        outcome: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Keep bounded proof that an implementation ran, without landability.

        A round that failed in Core validation or Indexer post-work still
        produced real work. Retaining the session, attributable files, and the
        revision digest lets a later reader tell "the model never ran" from
        "the model ran and the proof lane refused it". None of this makes the
        job auditable: the state stays terminal and non-landable.
        """

        record = dict(outcome)
        if not started:
            return record
        session = str(result.thread_id or "")
        if session and not session.startswith(PROVISIONAL_THREAD_PREFIXES):
            # A provisional host thread proves nothing about an implementation
            # session, so it is never promoted into one.
            record["implementation_session_id"] = session
        files = sorted({str(item) for item in (result.files_changed or ())})
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
    ) -> None:
        """Force one terminal fail-closed record for any worker exit."""

        self._discard_resume(tenant_ref, job_id)
        changes: Dict[str, Any] = {
            "state": CodingJobState.FAILED.value,
            "failure_code": code,
            "landable": False,
            "error": str(redact_evidence(str(exc)))[:1000],
        }
        if progress is not None and progress.implementer_started:
            changes["implementer_started"] = True
            if self.require_codex_audit:
                changes["implementation_backend"] = self.implementation_backend
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
        if not route.strict or not route.ok:
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
        payload = json.dumps(
            dataclasses.asdict(request), ensure_ascii=False, sort_keys=True,
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

    @classmethod
    def _receipt(cls, record: Mapping[str, Any]) -> CodingJobReceipt:
        return CodingJobReceipt(
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
            route_receipt=(
                record["route_receipt"]
                if isinstance(record.get("route_receipt"), dict) else None
            ),
            emergency_authority=(
                record["emergency_authority"]
                if isinstance(record.get("emergency_authority"), dict) else None
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
