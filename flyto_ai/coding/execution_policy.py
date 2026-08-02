# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Bounded resource, sandbox, secret, and human-approval admission policy."""
from __future__ import annotations

import asyncio
import inspect
import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Tuple

from flyto_ai.coding.store import redact_evidence
from flyto_ai.permissions import PermissionLevel
from flyto_ai.redaction import is_sensitive_key


EXECUTION_POLICY_VERSION = "flyto.capability-execution-policy.v1"
_SAFE_SENSITIVE_KEYS = frozenset({
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "token_budget",
    "total_tokens",
})
_PATH_ARGUMENT_KEYS = frozenset({
    "cwd", "directory", "file_path", "output_path", "working_dir", "workspace",
})


@dataclass(frozen=True)
class ExecutionLimits:
    """Hard limits for one manager lifecycle; all values are locally enforced."""

    max_calls: int = 10_000
    max_failures: int = 1_000
    max_elapsed_seconds: int = 86_400
    max_concurrency: int = 32
    queue_timeout_seconds: int = 30
    approval_timeout_seconds: int = 300
    max_argument_bytes: int = 1024 * 1024
    max_argument_depth: int = 32
    max_argument_nodes: int = 10_000
    max_result_bytes: int = 1024 * 1024
    max_result_depth: int = 32
    max_result_nodes: int = 10_000

    def __post_init__(self) -> None:
        bounded = {
            "max_calls": (self.max_calls, 1, 1_000_000),
            "max_failures": (self.max_failures, 1, 1_000_000),
            "max_elapsed_seconds": (self.max_elapsed_seconds, 1, 7 * 86_400),
            "max_concurrency": (self.max_concurrency, 1, 1_024),
            "queue_timeout_seconds": (self.queue_timeout_seconds, 1, 3_600),
            "approval_timeout_seconds": (self.approval_timeout_seconds, 1, 3_600),
            "max_argument_bytes": (self.max_argument_bytes, 1, 16 * 1024 * 1024),
            "max_argument_depth": (self.max_argument_depth, 1, 128),
            "max_argument_nodes": (self.max_argument_nodes, 1, 1_000_000),
            "max_result_bytes": (self.max_result_bytes, 1, 16 * 1024 * 1024),
            "max_result_depth": (self.max_result_depth, 1, 128),
            "max_result_nodes": (self.max_result_nodes, 1, 1_000_000),
        }
        for name, (value, lower, upper) in bounded.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("{} must be an integer".format(name))
            if not lower <= value <= upper:
                raise ValueError("{} is outside the supported range".format(name))


@dataclass(frozen=True)
class ExecutionPolicy:
    """Replaceable host policy layered after permission evaluation."""

    limits: ExecutionLimits = ExecutionLimits()
    approval_level: Optional[PermissionLevel] = None
    reject_sensitive_arguments: bool = True
    allowed_sensitive_keys: Tuple[str, ...] = ()
    allow_outside_workspace_paths: bool = False
    workspace_path_keys: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.limits, ExecutionLimits):
            raise ValueError("limits must be an ExecutionLimits value")
        if self.approval_level is not None and not isinstance(
            self.approval_level, PermissionLevel,
        ):
            raise ValueError("approval_level must be a PermissionLevel or None")
        if not isinstance(self.reject_sensitive_arguments, bool):
            raise ValueError("reject_sensitive_arguments must be a boolean")
        if not isinstance(self.allow_outside_workspace_paths, bool):
            raise ValueError("allow_outside_workspace_paths must be a boolean")
        if any(not isinstance(key, str) or not key for key in self.allowed_sensitive_keys):
            raise ValueError("allowed_sensitive_keys must contain non-empty strings")
        if len(set(self.allowed_sensitive_keys)) != len(self.allowed_sensitive_keys):
            raise ValueError("allowed_sensitive_keys contains duplicates")
        if any(not isinstance(key, str) or not key for key in self.workspace_path_keys):
            raise ValueError("workspace_path_keys must contain non-empty strings")
        normalized_path_keys = [key.lower() for key in self.workspace_path_keys]
        if len(set(normalized_path_keys)) != len(normalized_path_keys):
            raise ValueError("workspace_path_keys contains duplicates")


@dataclass(frozen=True)
class ApprovalRequest:
    """Redacted human-decision envelope; raw arguments never leave the host gate."""

    request_id: str
    provider_name: str
    remote_name: str
    required_level: PermissionLevel
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    approver_ref: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.approved, bool):
            raise ValueError("approval decision must contain a boolean")
        for name, value, limit in (
            ("approver_ref", self.approver_ref, 256),
            ("reason", self.reason, 1000),
        ):
            if not isinstance(value, str) or len(value) > limit:
                raise ValueError("{} must be a bounded string".format(name))


ApprovalResolver = Callable[
    [ApprovalRequest], ApprovalDecision | Awaitable[ApprovalDecision]
]


@dataclass(frozen=True)
class ExecutionCompletion:
    """Policy result after one admitted call releases its concurrency lease."""

    allowed: bool
    result_bytes: int
    error: str = ""


@dataclass
class ExecutionAdmission:
    """One admission decision and, when allowed, an exactly-once lease."""

    allowed: bool
    call_id: str
    error: str = ""
    policy_code: str = "allow"
    approval: Optional[ApprovalDecision] = None
    _controller: Optional["ExecutionPolicyController"] = None
    _finished: bool = False

    async def finish(self, ok: bool, result: Any) -> ExecutionCompletion:
        """Record completion and release capacity exactly once."""
        if not self.allowed or self._controller is None:
            return ExecutionCompletion(False, 0, self.error or "call was not admitted")
        if self._finished:
            raise RuntimeError("execution admission was already finished")
        self._finished = True
        return await self._controller._finish(ok, result)

    def denial_payload(self) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": self.error,
            "policy_outcome": "block",
            "policy_code": self.policy_code,
            "call_id": self.call_id,
        }


class ExecutionPolicyController:
    """Atomically enforce lifecycle budgets before capability dispatch."""

    def __init__(
        self,
        workspace: str,
        policy: Optional[ExecutionPolicy] = None,
        *,
        approval_resolver: Optional[ApprovalResolver] = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("execution policy workspace must be a directory")
        if policy is not None and not isinstance(policy, ExecutionPolicy):
            raise ValueError("policy must be an ExecutionPolicy value")
        if approval_resolver is not None and not callable(approval_resolver):
            raise ValueError("approval_resolver must be callable")
        self.policy = policy or ExecutionPolicy()
        self._approval_resolver = approval_resolver
        self._started_at = time.monotonic()
        self._calls = 0
        self._failures = 0
        self._active = 0
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.policy.limits.max_concurrency)

    async def admit(
        self,
        provider_name: str,
        remote_name: str,
        required_level: PermissionLevel,
        arguments: Mapping[str, Any],
    ) -> ExecutionAdmission:
        """Validate one request, obtain approval, and acquire bounded capacity."""
        call_id = "call_{}".format(uuid.uuid4().hex[:20])
        if not isinstance(provider_name, str) or not provider_name:
            return self._deny(call_id, "invalid_tool", "provider tool name is invalid")
        if not isinstance(remote_name, str) or not remote_name:
            return self._deny(call_id, "invalid_tool", "remote tool name is invalid")
        if not isinstance(required_level, PermissionLevel):
            return self._deny(call_id, "invalid_permission", "required permission is invalid")
        if not isinstance(arguments, Mapping):
            return self._deny(call_id, "invalid_arguments", "arguments must be an object")
        shape_error = _json_shape_error(
            arguments,
            max_depth=self.policy.limits.max_argument_depth,
            max_nodes=self.policy.limits.max_argument_nodes,
        )
        if shape_error:
            return self._deny(call_id, "invalid_arguments", shape_error)
        try:
            argument_bytes = _json_bytes(arguments)
        except (TypeError, ValueError, RecursionError):
            return self._deny(call_id, "invalid_arguments", "arguments must be finite JSON")
        if argument_bytes > self.policy.limits.max_argument_bytes:
            return self._deny(call_id, "argument_budget", "arguments exceed the byte budget")
        sensitive = self._sensitive_argument(arguments)
        if sensitive:
            return self._deny(
                call_id, "secret_argument",
                "sensitive argument key is not allowed: {}".format(sensitive),
            )
        outside = self._outside_workspace_path(arguments)
        if outside:
            return self._deny(
                call_id, "sandbox_path",
                "path argument escapes the workspace: {}".format(outside),
            )
        budget_error = await self._budget_error()
        if budget_error:
            return self._deny(call_id, budget_error[0], budget_error[1])

        approval = await self._approval(
            call_id, provider_name, remote_name, required_level, arguments,
        )
        if approval is not None and not approval.approved:
            return ExecutionAdmission(
                allowed=False,
                call_id=call_id,
                error=approval.reason or "human approval was denied",
                policy_code="approval_denied",
                approval=approval,
            )

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.policy.limits.queue_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return self._deny(call_id, "concurrency_timeout", "execution queue timed out")

        async with self._lock:
            budget_error = self._budget_error_unlocked()
            if budget_error:
                self._semaphore.release()
                return self._deny(call_id, budget_error[0], budget_error[1])
            self._calls += 1
            self._active += 1
        return ExecutionAdmission(
            allowed=True,
            call_id=call_id,
            approval=approval,
            _controller=self,
        )

    async def snapshot(self) -> Dict[str, Any]:
        """Return bounded, secret-free counters for trace/evidence reporting."""
        async with self._lock:
            return {
                "contract_version": EXECUTION_POLICY_VERSION,
                "calls": self._calls,
                "failures": self._failures,
                "active": self._active,
                "elapsed_ms": int((time.monotonic() - self._started_at) * 1000),
                "limits": {
                    "max_calls": self.policy.limits.max_calls,
                    "max_failures": self.policy.limits.max_failures,
                    "max_concurrency": self.policy.limits.max_concurrency,
                    "max_argument_bytes": self.policy.limits.max_argument_bytes,
                    "max_result_bytes": self.policy.limits.max_result_bytes,
                },
            }

    async def _approval(
        self,
        call_id: str,
        provider_name: str,
        remote_name: str,
        required_level: PermissionLevel,
        arguments: Mapping[str, Any],
    ) -> Optional[ApprovalDecision]:
        threshold = self.policy.approval_level
        if threshold is None or required_level < threshold:
            return None
        if self._approval_resolver is None:
            return ApprovalDecision(False, reason="human approval is required but unavailable")
        request = ApprovalRequest(
            request_id=call_id,
            provider_name=provider_name,
            remote_name=remote_name,
            required_level=required_level,
            arguments=redact_evidence(dict(arguments)),
        )
        started = time.monotonic()
        timeout = self.policy.limits.approval_timeout_seconds
        try:
            decision = await asyncio.wait_for(
                asyncio.to_thread(self._approval_resolver, request),
                timeout=timeout,
            )
            if inspect.isawaitable(decision):
                remaining = max(0.001, timeout - (time.monotonic() - started))
                decision = await asyncio.wait_for(
                    decision,
                    timeout=remaining,
                )
        except asyncio.TimeoutError:
            return ApprovalDecision(False, reason="approval resolver timed out")
        except Exception:
            return ApprovalDecision(False, reason="approval resolver failed")
        if not isinstance(decision, ApprovalDecision):
            return ApprovalDecision(False, reason="approval resolver returned an invalid decision")
        return decision

    async def _budget_error(self) -> Optional[Tuple[str, str]]:
        async with self._lock:
            return self._budget_error_unlocked()

    def _budget_error_unlocked(self) -> Optional[Tuple[str, str]]:
        limits = self.policy.limits
        if time.monotonic() - self._started_at > limits.max_elapsed_seconds:
            return "elapsed_budget", "execution elapsed-time budget is exhausted"
        if self._calls >= limits.max_calls:
            return "call_budget", "execution call budget is exhausted"
        if self._failures >= limits.max_failures:
            return "failure_budget", "execution failure budget is exhausted"
        return None

    async def _finish(self, ok: bool, result: Any) -> ExecutionCompletion:
        error = "" if isinstance(ok, bool) else "completion status must be a boolean"
        shape_error = _json_shape_error(
            result,
            max_depth=self.policy.limits.max_result_depth,
            max_nodes=self.policy.limits.max_result_nodes,
        )
        error = error or shape_error
        try:
            result_bytes = 0 if error else _json_bytes(result)
        except (TypeError, ValueError, RecursionError):
            result_bytes = 0
            error = "capability result must be finite JSON"
        if result_bytes > self.policy.limits.max_result_bytes:
            error = "capability result exceeds the byte budget"
        allowed = not error
        async with self._lock:
            self._active = max(0, self._active - 1)
            if not ok or not allowed:
                self._failures += 1
        self._semaphore.release()
        return ExecutionCompletion(allowed=allowed, result_bytes=result_bytes, error=error)

    def _sensitive_argument(self, value: Any, prefix: str = "") -> str:
        if not self.policy.reject_sensitive_arguments:
            return ""
        allowed = _SAFE_SENSITIVE_KEYS | {
            key.lower() for key in self.policy.allowed_sensitive_keys
        }
        if isinstance(value, Mapping):
            for key, item in value.items():
                name = str(key)
                path = "{}.{}".format(prefix, name).strip(".")
                if is_sensitive_key(name) and name.lower() not in allowed:
                    return path
                found = self._sensitive_argument(item, path)
                if found:
                    return found
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                found = self._sensitive_argument(item, "{}[{}]".format(prefix, index))
                if found:
                    return found
        return ""

    def _outside_workspace_path(self, value: Any, prefix: str = "") -> str:
        if self.policy.allow_outside_workspace_paths:
            return ""
        path_keys = _PATH_ARGUMENT_KEYS | {
            key.lower() for key in self.policy.workspace_path_keys
        }
        if isinstance(value, Mapping):
            for key, item in value.items():
                name = str(key)
                path = "{}.{}".format(prefix, name).strip(".")
                if name.lower() in path_keys and isinstance(item, str):
                    if "\x00" in item:
                        return path
                    candidate = Path(item).expanduser()
                    if not candidate.is_absolute():
                        candidate = self.workspace / candidate
                    resolved = candidate.resolve(strict=False)
                    if resolved != self.workspace and self.workspace not in resolved.parents:
                        return path
                found = self._outside_workspace_path(item, path)
                if found:
                    return found
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                found = self._outside_workspace_path(item, "{}[{}]".format(prefix, index))
                if found:
                    return found
        return ""

    @staticmethod
    def _deny(call_id: str, code: str, error: str) -> ExecutionAdmission:
        return ExecutionAdmission(
            allowed=False,
            call_id=call_id,
            error=error,
            policy_code=code,
        )


def _json_bytes(value: Any) -> int:
    return len(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8"))


def _json_shape_error(value: Any, *, max_depth: int, max_nodes: int) -> str:
    """Validate JSON structure iteratively before recursive serialization."""
    stack = [(value, 0)]
    seen = set()
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            return "JSON value exceeds the structural node budget"
        if depth > max_depth:
            return "JSON value exceeds the structural depth budget"
        if item is None or isinstance(item, (bool, int, str)):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                return "JSON value contains a non-finite number"
            continue
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in seen:
                return "JSON value contains a cyclic or repeated container"
            seen.add(identity)
            if any(not isinstance(key, str) for key in item):
                return "JSON object keys must be strings"
            stack.extend((child, depth + 1) for child in item.values())
            continue
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in seen:
                return "JSON value contains a cyclic or repeated container"
            seen.add(identity)
            stack.extend((child, depth + 1) for child in item)
            continue
        return "JSON value contains an unsupported type"
    return ""


__all__ = [
    "ApprovalDecision",
    "ApprovalRequest",
    "ApprovalResolver",
    "EXECUTION_POLICY_VERSION",
    "ExecutionAdmission",
    "ExecutionCompletion",
    "ExecutionLimits",
    "ExecutionPolicy",
    "ExecutionPolicyController",
]
