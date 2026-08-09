# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Tenant-scoped durable job service for the native Flyto2 coding agent."""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import re
import stat
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
from flyto_ai.coding.route import CodingRoutePolicy, CodingRouteReceipt, route_thread_id
from flyto_ai.coding.store import ThreadStore, redact_evidence


try:  # pragma: no cover - Windows fallback is exercised by static review.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_JOB_ID = re.compile(r"^job_[a-f0-9]{24}$")
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


class CodingServiceError(RuntimeError):
    """Base error with a stable, non-sensitive service code."""

    code = "service_error"


class CodingServiceBusy(CodingServiceError):
    code = "service_busy"


class CodingJobNotFound(CodingServiceError):
    code = "job_not_found"


class IdempotencyConflict(CodingServiceError):
    code = "idempotency_conflict"


class WorkspaceDenied(CodingServiceError):
    code = "workspace_denied"


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
        # Resume context is process-local on purpose: the task prompt is not
        # persisted, so a restart cannot silently start a new session.
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
            with self._state_guard():
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
        request = dataclasses.replace(
            request,
            approval_policy=self.approval_policy,
            sandbox_mode=self.sandbox_mode,
            checks=(),
            capabilities=(),
            config_path=self.config_path,
            command_sandbox_image=self.sandbox_image,
        )
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
                "audit_count": 0,
                "rework_count": 0,
                "audit_findings_sha256": "",
                "landable": False,
            }
            try:
                self._write_json(tenant_dir / "jobs" / (job_id + ".json"), record)
                self._write_json(
                    idempotency_path,
                    {"job_id": job_id, "request_sha256": request_digest},
                )
                self._resume[(tenant_ref, job_id)] = request
                future = self._executor.submit(self._run_job, tenant_ref, job_id, request)
            except BaseException:
                self._release_job_lease(job_id)
                raise
            self._pending.add(future)
            future.add_done_callback(
                lambda completed, claimed=job_id: self._forget_future(completed, claimed),
            )
            return self._receipt(record)

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
                self._require_route_evidence(record)
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
            # Recompute from the live workspace so an edit landed after the
            # implementation cannot inherit an earlier audit decision.
            if self._stored_revision(record) != stored:
                raise RevisionMismatch("the implementation revision changed since it was submitted")
            self._require_route_evidence(record)
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
                self._resume.pop((tenant_ref, job_id), None)
                return self._receipt(self._read_json(path))
            return self._schedule_rework(
                path, tenant_ref, job_id, record, findings, digest, audit_count,
            )

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
        original = self._resume.get((tenant_ref, job_id))
        if not session or original is None:
            raise ReworkNotResumable("the implementation session cannot be resumed")
        request = dataclasses.replace(
            original,
            message=self._rework_message(original.message, findings),
            thread_id=session,
            resume=True,
        )
        if len(self._pending) >= self.max_queued:
            raise CodingServiceBusy("coding job queue is full")
        # Claim the job before queueing. A concurrent audit then observes a
        # non-awaiting state and cannot schedule a duplicate rework round.
        if not self._acquire_job_lease(job_id):
            raise CodingServiceBusy("coding job is already being executed")
        self._update_record_locked(
            path,
            state=CodingJobState.REWORK_QUEUED.value,
            audit_count=audit_count,
            rework_count=rework_count,
            audit_findings_sha256=digest,
            landable=False,
            failure_code=None,
        )
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
        with workspace_lock:
            self._update_record(path, state=(
                CodingJobState.REWORK_RUNNING if rework else CodingJobState.RUNNING
            ).value)
            try:
                store = ThreadStore(str(self._tenant_dir(tenant_ref) / "threads"))
                result, route = asyncio.run(self._implement(store, request))
                self._record_outcome(
                    path, tenant_ref, job_id, request, result, store, rework, route,
                )
            except CodingServiceError as exc:
                self._fail_job(path, tenant_ref, job_id, exc.code, exc)
            except Exception as exc:
                self._fail_job(path, tenant_ref, job_id, "service_execution_failed", exc)

    async def _implement(
        self, store: ThreadStore, request: CodingTaskRequest,
    ) -> "tuple[CodingTaskResult, Optional[CodingRouteReceipt]]":
        """Run the selected implementer, wrapped by the host-owned route.

        Without a strict startup policy this is exactly the historical direct
        call, so existing library callers keep their behavior. The public
        `code-mcp` / `code-serve` builders always enable the strict route.
        """

        policy = self.route_policy
        if policy is None or not policy.strict:
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
                # A capability that cannot even be launched is an unavailable
                # Indexer, not a service crash.
                return self._route_unavailable(request), self._route_unavailable_receipt(policy)
            missing = [status.name for status in statuses if status.required and not status.available]
            if missing or not manager.required_available:
                # A stale or unavailable Indexer must never reach an auditable
                # state, so the round fails before the model can edit.
                return self._route_unavailable(request), self._route_unavailable_receipt(policy)
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
    def _route_unavailable(
        request: CodingTaskRequest, code: str = "route_capability_unavailable",
    ) -> CodingTaskResult:
        return CodingTaskResult(
            ok=False,
            message="coding route lane indexer_pre failed: {}".format(code),
            thread_id=route_thread_id(request.thread_id or request.working_dir),
            attempts=0,
            status="failed",
            evidence_path="",
            failure_code=code,
        )

    @staticmethod
    def _route_unavailable_receipt(
        policy: Any, code: str = "capability_unavailable",
    ) -> "CodingRouteReceipt":
        from flyto_ai.coding.route import (
            CodingRouteReceipt,
            RouteLane,
            RouteLaneReceipt,
            RouteLaneStatus,
        )

        return CodingRouteReceipt(
            strict=True, ok=False, failure_code=code,
            lanes=(RouteLaneReceipt(
                lane=RouteLane.INDEXER_PRE.value, required=True,
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
    ) -> None:
        """Move one finished implementation round into its durable state."""

        result_record = dataclasses.asdict(result)
        result_record["evidence_path"] = ""
        outcome: Dict[str, Any] = {
            "route_receipt": route.to_mapping() if route is not None else None,
            "thread_id": result.thread_id,
            "evidence_sha256": store.digest(result.thread_id),
            "result": result_record,
            "failure_code": result.failure_code,
        }
        if not result.ok:
            self._resume.pop((tenant_ref, job_id), None)
            self._update_record(path, state=CodingJobState.FAILED.value, **outcome)
            return
        if not self.require_codex_audit:
            self._resume.pop((tenant_ref, job_id), None)
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
        self._update_record(
            path,
            state=CodingJobState.AWAITING_CODEX_AUDIT.value,
            implementation_backend=self.implementation_backend,
            implementation_session_id=session,
            implementation_revision_sha256=self._revision_digest(request.working_dir, files),
            implementation_files=files,
            working_dir=request.working_dir,
            landable=False,
            **outcome,
        )

    def _fail_job(
        self, path: Path, tenant_ref: str, job_id: str, code: str, exc: Exception,
    ) -> None:
        self._resume.pop((tenant_ref, job_id), None)
        self._update_record(
            path,
            state=CodingJobState.FAILED.value,
            failure_code=code,
            error=str(redact_evidence(str(exc)))[:1000],
        )

    def _require_route_evidence(self, record: Mapping[str, Any]) -> None:
        """A strict public round is auditable only with its own route proof.

        Deleting or editing `route_receipt` in the persisted job JSON cannot
        produce an acceptance: the evidence must be present, valid, strict,
        and successful for this exact implementation round.
        """
        policy = self.route_policy
        if policy is None or not policy.strict:
            return
        stored = record.get("route_receipt")
        if not isinstance(stored, dict):
            raise RouteEvidenceMissing("this round has no coding route evidence")
        try:
            route = CodingRouteReceipt.from_mapping(stored)
        except ValueError as exc:
            raise RouteEvidenceMissing("coding route evidence is invalid") from exc
        if not route.strict or not route.ok:
            raise RouteEvidenceMissing("coding route evidence is not a passed strict route")

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
        self._write_json(path, record)
        # Publish the settled record before releasing its execution lease.
        # A second process therefore cannot observe an auditable/terminal
        # state while the previous round still appears to own the job.
        if str(record.get("state")) not in _INTERRUPTED_JOB_STATES:
            self._release_job_lease(str(record.get("job_id") or ""))

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
            route_receipt=(
                record["route_receipt"]
                if isinstance(record.get("route_receipt"), dict) else None
            ),
        )

    def _reconcile_interrupted_jobs(self) -> None:
        tenants = self.state_root / "tenants"
        if not tenants.is_dir():
            return
        for path in tenants.glob("*/jobs/job_*.json"):
            try:
                record = self._read_json(path)
                # An awaiting-audit job survives a restart; in-flight work does
                # not, but another live MCP process may still own its job lease.
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
                    finally:
                        self._release_job_lease(job_id)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
