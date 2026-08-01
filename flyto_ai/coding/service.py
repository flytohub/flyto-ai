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
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from flyto_ai.coding.agent import FlytoCodingAgent
from flyto_ai.coding.contracts import (
    ApprovalPolicy,
    CapabilityStatus,
    CheckResult,
    CodingJobReceipt,
    CodingJobState,
    CodingTaskRequest,
    CodingTaskResult,
    SandboxMode,
)
from flyto_ai.coding.store import ThreadStore, redact_evidence


try:  # pragma: no cover - Windows fallback is exercised by static review.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_JOB_ID = re.compile(r"^job_[a-f0-9]{24}$")
_ALLOWED_REQUEST_FIELDS = frozenset({
    "message", "working_dir", "thread_id", "resume", "max_attempts",
    "max_rounds", "require_changes",
})


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


AgentFactory = Callable[[ThreadStore], FlytoCodingAgent]


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
        # Reuse the request contract's path/image validation without persisting
        # a synthetic request or accepting those authority fields remotely.
        config_parts = Path(self.config_path).parts
        if Path(self.config_path).is_absolute() or "\x00" in self.config_path or ".." in config_parts:
            raise ValueError("config_path must be relative")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:-]*", self.sandbox_image):
            raise ValueError("sandbox_image is invalid")
        self._lock = threading.RLock()
        self._workspace_locks: Dict[str, threading.Lock] = {}
        self._pending: set[Future[Any]] = set()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="flyto-coding")
        self._closed = False
        self._lock_fd = os.open(self.state_root / ".service.lock", os.O_CREAT | os.O_RDWR, 0o600)
        if fcntl is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(self._lock_fd)
                raise CodingServiceBusy("coding state root is already served") from exc
        self._reconcile_interrupted_jobs()

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
        with self._lock:
            if self._closed:
                raise CodingServiceError("coding service is closed")
            if idempotency_path.exists():
                reference = self._read_json(idempotency_path)
                if reference.get("request_sha256") != request_digest:
                    raise IdempotencyConflict("idempotency key was already used for another request")
                return self.get(tenant_id, str(reference.get("job_id", "")))
            if len(self._pending) >= self.max_queued:
                raise CodingServiceBusy("coding job queue is full")
            now = time.time()
            job_id = "job_{}".format(uuid.uuid4().hex[:24])
            record = {
                "job_id": job_id,
                "state": CodingJobState.QUEUED.value,
                "submitted_at": now,
                "updated_at": now,
                "request_sha256": request_digest,
                "workspace_sha256": hashlib.sha256(request.working_dir.encode()).hexdigest(),
                "thread_id": "",
                "evidence_sha256": "",
                "result": None,
                "failure_code": None,
            }
            self._write_json(tenant_dir / "jobs" / (job_id + ".json"), record)
            self._write_json(idempotency_path, {"job_id": job_id, "request_sha256": request_digest})
            future = self._executor.submit(self._run_job, tenant_ref, job_id, request)
            self._pending.add(future)
            future.add_done_callback(self._forget_future)
            return self._receipt(record)

    def get(self, tenant_id: str, job_id: str) -> CodingJobReceipt:
        """Read a job only from the authenticated tenant namespace."""

        if not _JOB_ID.fullmatch(job_id):
            raise CodingJobNotFound("coding job does not exist")
        path = self._tenant_dir(self._tenant_ref(tenant_id), create=False) / "jobs" / (job_id + ".json")
        try:
            record = self._read_json(path)
        except FileNotFoundError as exc:
            raise CodingJobNotFound("coding job does not exist") from exc
        return self._receipt(record)

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # The state-root lease must outlive active worker writes. `wait=False`
        # cancels jobs that have not started but still drains running jobs.
        self._executor.shutdown(wait=True, cancel_futures=not wait)
        if fcntl is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)

    def _run_job(self, tenant_ref: str, job_id: str, request: CodingTaskRequest) -> None:
        path = self._tenant_dir(tenant_ref) / "jobs" / (job_id + ".json")
        workspace_lock = self._workspace_lock(request.working_dir)
        with workspace_lock:
            self._update_record(path, state=CodingJobState.RUNNING.value)
            try:
                store = ThreadStore(str(self._tenant_dir(tenant_ref) / "threads"))
                result = asyncio.run(self.agent_factory(store).run(request))
                state = CodingJobState.COMPLETED if result.ok else CodingJobState.FAILED
                result_record = dataclasses.asdict(result)
                result_record["evidence_path"] = ""
                self._update_record(
                    path,
                    state=state.value,
                    thread_id=result.thread_id,
                    evidence_sha256=store.digest(result.thread_id),
                    result=result_record,
                    failure_code=result.failure_code,
                )
            except Exception as exc:
                self._update_record(
                    path,
                    state=CodingJobState.FAILED.value,
                    failure_code="service_execution_failed",
                    error=str(redact_evidence(str(exc)))[:1000],
                )

    def _forget_future(self, future: Future[Any]) -> None:
        with self._lock:
            self._pending.discard(future)

    def _assert_workspace(self, workspace: str) -> None:
        target = Path(workspace).resolve()
        if not any(target == root or root in target.parents for root in self.workspace_roots):
            raise WorkspaceDenied("working_dir is outside the configured workspace roots")

    def _workspace_lock(self, workspace: str) -> threading.Lock:
        with self._lock:
            return self._workspace_locks.setdefault(str(Path(workspace).resolve()), threading.Lock())

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
        with self._lock:
            record = self._read_json(path)
            record.update(changes)
            record["updated_at"] = time.time()
            self._write_json(path, record)

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
        )

    def _reconcile_interrupted_jobs(self) -> None:
        tenants = self.state_root / "tenants"
        if not tenants.is_dir():
            return
        for path in tenants.glob("*/jobs/job_*.json"):
            try:
                record = self._read_json(path)
                if record.get("state") in {CodingJobState.QUEUED.value, CodingJobState.RUNNING.value}:
                    record.update({
                        "state": CodingJobState.FAILED.value,
                        "updated_at": time.time(),
                        "failure_code": "service_restarted",
                    })
                    self._write_json(path, record)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
