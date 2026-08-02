# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Content-addressed redacted capability traces, replay, and outcome feedback."""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from flyto_ai.coding.store import redact_evidence


EXECUTION_TRACE_VERSION = "flyto.capability-trace.v1"
ReplayDispatch = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]
TraceNormalizer = Callable[[Any], Any]


class OutcomeFeedbackSink(Protocol):
    """Host-owned bridge to Blueprint or another learning backend."""

    def __call__(self, payload: Mapping[str, Any]) -> Any: ...


@dataclass(frozen=True)
class ExecutionTraceEvent:
    sequence: int
    provider_name: str
    remote_name: str
    required_permission: str
    arguments: Mapping[str, Any]
    dispatched: bool
    replay_safe: bool
    ok: bool
    policy_code: str
    result: Any
    previous_hash: str
    event_hash: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "provider_name": self.provider_name,
            "remote_name": self.remote_name,
            "required_permission": self.required_permission,
            "arguments": _thaw_json(self.arguments),
            "dispatched": self.dispatched,
            "replay_safe": self.replay_safe,
            "ok": self.ok,
            "policy_code": self.policy_code,
            "result": _thaw_json(self.result),
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
        }


@dataclass(frozen=True)
class ReplayMismatch:
    sequence: int
    provider_name: str
    reason: str
    expected_hash: str
    actual_hash: str


@dataclass(frozen=True)
class ExecutionReplayReport:
    trace_fingerprint: str
    attempted: int
    matched: int
    skipped: int
    mismatches: Tuple[ReplayMismatch, ...]

    @property
    def ok(self) -> bool:
        return self.attempted > 0 and not self.mismatches and self.matched == self.attempted

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": EXECUTION_TRACE_VERSION,
            "trace_fingerprint": self.trace_fingerprint,
            "attempted": self.attempted,
            "matched": self.matched,
            "skipped": self.skipped,
            "ok": self.ok,
            "mismatches": [
                {
                    "sequence": item.sequence,
                    "provider_name": item.provider_name,
                    "reason": item.reason,
                    "expected_hash": item.expected_hash,
                    "actual_hash": item.actual_hash,
                }
                for item in self.mismatches
            ],
        }


@dataclass(frozen=True)
class OutcomeFeedbackReceipt:
    blueprint_id: str
    execution_id: str
    success: bool
    trace_fingerprint: str
    sink_result: Any


class ExecutionTraceLedger:
    """Bounded append-only hash chain that stores only redacted JSON evidence."""

    def __init__(self, *, max_events: int = 10_000) -> None:
        if isinstance(max_events, bool) or not isinstance(max_events, int):
            raise ValueError("max_events must be an integer")
        if not 1 <= max_events <= 100_000:
            raise ValueError("max_events is outside the supported range")
        self._max_events = max_events
        self._events: List[ExecutionTraceEvent] = []

    @property
    def events(self) -> Tuple[ExecutionTraceEvent, ...]:
        return tuple(self._events)

    @property
    def fingerprint(self) -> str:
        if not self._events:
            return hashlib.sha256(EXECUTION_TRACE_VERSION.encode()).hexdigest()
        return self._events[-1].event_hash

    def record(
        self,
        *,
        provider_name: str,
        remote_name: str,
        required_permission: str,
        arguments: Mapping[str, Any],
        dispatched: bool,
        ok: bool,
        policy_code: str,
        result: Any,
    ) -> ExecutionTraceEvent:
        """Append one deterministic event after recursively redacting evidence."""
        if len(self._events) >= self._max_events:
            raise RuntimeError("execution trace event budget is exhausted")
        for name, value in {
            "provider_name": provider_name,
            "remote_name": remote_name,
            "required_permission": required_permission,
            "policy_code": policy_code,
        }.items():
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError("{} must be a bounded string".format(name))
        if not isinstance(arguments, Mapping):
            raise ValueError("trace arguments must be an object")
        if not isinstance(dispatched, bool) or not isinstance(ok, bool):
            raise ValueError("trace dispatched and ok fields must be booleans")
        if required_permission not in {"read_only", "workspace_write", "danger_full"}:
            raise ValueError("required_permission is unsupported")
        raw_arguments = dict(arguments)
        try:
            raw_argument_bytes = _canonical_bytes(raw_arguments)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("trace arguments must be finite JSON") from exc
        safe_arguments = redact_evidence(raw_arguments)
        safe_result = redact_evidence(result)
        try:
            replay_safe = raw_argument_bytes == _canonical_bytes(safe_arguments)
        except (TypeError, ValueError, RecursionError):
            replay_safe = False
        frozen_arguments = _freeze_json(safe_arguments)
        frozen_result = _freeze_json(safe_result)
        sequence = len(self._events) + 1
        previous_hash = self.fingerprint
        payload = {
            "contract_version": EXECUTION_TRACE_VERSION,
            "sequence": sequence,
            "provider_name": provider_name,
            "remote_name": remote_name,
            "required_permission": required_permission,
            "arguments": safe_arguments,
            "dispatched": dispatched,
            "replay_safe": replay_safe,
            "ok": ok,
            "policy_code": policy_code,
            "result": safe_result,
        }
        event_hash = hashlib.sha256(
            previous_hash.encode("ascii") + _canonical_bytes(payload),
        ).hexdigest()
        event = ExecutionTraceEvent(
            sequence=sequence,
            provider_name=provider_name,
            remote_name=remote_name,
            required_permission=required_permission,
            arguments=frozen_arguments,
            dispatched=dispatched,
            replay_safe=replay_safe,
            ok=ok,
            policy_code=policy_code,
            result=frozen_result,
            previous_hash=previous_hash,
            event_hash=event_hash,
        )
        self._events.append(event)
        return event

    def export(self) -> Dict[str, Any]:
        return {
            "contract_version": EXECUTION_TRACE_VERSION,
            "fingerprint": self.fingerprint,
            "event_count": len(self._events),
            "events": [event.as_dict() for event in self._events],
        }

    def verify_chain(self) -> bool:
        previous_hash = hashlib.sha256(EXECUTION_TRACE_VERSION.encode()).hexdigest()
        for event in self._events:
            payload = {
                "contract_version": EXECUTION_TRACE_VERSION,
                "sequence": event.sequence,
                "provider_name": event.provider_name,
                "remote_name": event.remote_name,
                "required_permission": event.required_permission,
                "arguments": event.arguments,
                "dispatched": event.dispatched,
                "replay_safe": event.replay_safe,
                "ok": event.ok,
                "policy_code": event.policy_code,
                "result": event.result,
            }
            expected = hashlib.sha256(
                previous_hash.encode("ascii") + _canonical_bytes(payload),
            ).hexdigest()
            if event.previous_hash != previous_hash or event.event_hash != expected:
                return False
            previous_hash = event.event_hash
        return True

    async def replay(
        self,
        dispatch: ReplayDispatch,
        *,
        normalizers: Optional[Mapping[str, TraceNormalizer]] = None,
        allowed_permissions: Sequence[str] = ("read_only",),
    ) -> ExecutionReplayReport:
        """Replay safe events under an explicit permission ceiling."""
        if not callable(dispatch):
            raise ValueError("replay dispatch must be callable")
        normalizers = dict(normalizers or {})
        if any(not isinstance(name, str) or not callable(fn) for name, fn in normalizers.items()):
            raise ValueError("replay normalizers must map provider names to callables")
        allowed_permissions = tuple(allowed_permissions)
        supported_permissions = {"read_only", "workspace_write", "danger_full"}
        if (
            not allowed_permissions
            or any(level not in supported_permissions for level in allowed_permissions)
            or len(set(allowed_permissions)) != len(allowed_permissions)
        ):
            raise ValueError("replay permissions must contain unique supported levels")
        events = tuple(self._events)
        trace_fingerprint = (
            events[-1].event_hash
            if events
            else hashlib.sha256(EXECUTION_TRACE_VERSION.encode()).hexdigest()
        )
        mismatches: List[ReplayMismatch] = []
        attempted = 0
        matched = 0
        skipped = 0
        for event in events:
            if (
                not event.dispatched
                or not event.replay_safe
                or event.required_permission not in allowed_permissions
            ):
                skipped += 1
                continue
            attempted += 1
            try:
                expected = _normalized_result(
                    event.provider_name, _thaw_json(event.result), normalizers,
                )
                expected_hash = hashlib.sha256(_canonical_bytes(expected)).hexdigest()
            except Exception as exc:
                error_hash = hashlib.sha256(
                    _canonical_bytes({"error_type": type(exc).__name__}),
                ).hexdigest()
                mismatches.append(ReplayMismatch(
                    sequence=event.sequence,
                    provider_name=event.provider_name,
                    reason="normalizer_failed",
                    expected_hash=error_hash,
                    actual_hash=error_hash,
                ))
                continue
            try:
                actual_raw = await dispatch(
                    event.provider_name, _thaw_json(event.arguments),
                )
            except Exception as exc:
                actual_hash = hashlib.sha256(
                    _canonical_bytes({"error_type": type(exc).__name__}),
                ).hexdigest()
                reason = "dispatch_failed"
            else:
                try:
                    actual = _normalized_result(
                        event.provider_name, redact_evidence(actual_raw), normalizers,
                    )
                    actual_hash = hashlib.sha256(_canonical_bytes(actual)).hexdigest()
                    reason = "result_mismatch"
                except Exception as exc:
                    actual_hash = hashlib.sha256(
                        _canonical_bytes({"error_type": type(exc).__name__}),
                    ).hexdigest()
                    reason = "normalizer_failed"
            if actual_hash == expected_hash:
                matched += 1
            else:
                mismatches.append(ReplayMismatch(
                    sequence=event.sequence,
                    provider_name=event.provider_name,
                    reason=reason,
                    expected_hash=expected_hash,
                    actual_hash=actual_hash,
                ))
        return ExecutionReplayReport(
            trace_fingerprint=trace_fingerprint,
            attempted=attempted,
            matched=matched,
            skipped=skipped,
            mismatches=tuple(mismatches),
        )

    async def publish_blueprint_outcome(
        self,
        blueprint_id: str,
        replay: ExecutionReplayReport,
        sink: OutcomeFeedbackSink,
        *,
        timeout_seconds: int = 30,
    ) -> OutcomeFeedbackReceipt:
        """Publish one idempotency-ready, evidence-bound learning outcome."""
        if not isinstance(blueprint_id, str) or not blueprint_id or len(blueprint_id) > 128:
            raise ValueError("blueprint_id must be a bounded string")
        if replay.trace_fingerprint != self.fingerprint:
            raise ValueError("replay report belongs to a different execution trace")
        if not callable(sink):
            raise ValueError("outcome feedback sink must be callable")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 300
        ):
            raise ValueError("outcome feedback timeout is outside the supported range")
        execution_id = "trace_{}".format(self.fingerprint[:24])
        payload = {
            "blueprint_id": blueprint_id,
            "success": replay.ok,
            "execution_id": execution_id,
            "evidence": {
                "contract_version": EXECUTION_TRACE_VERSION,
                "trace_fingerprint": self.fingerprint,
                "event_count": len(self._events),
                "replay": replay.as_dict(),
            },
        }
        started = time.monotonic()
        try:
            sink_result = await asyncio.wait_for(
                asyncio.to_thread(sink, payload),
                timeout=timeout_seconds,
            )
            if inspect.isawaitable(sink_result):
                remaining = max(0.001, timeout_seconds - (time.monotonic() - started))
                sink_result = await asyncio.wait_for(sink_result, timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("outcome feedback sink timed out") from exc
        except Exception as exc:
            raise RuntimeError("outcome feedback sink failed") from exc
        return OutcomeFeedbackReceipt(
            blueprint_id=blueprint_id,
            execution_id=execution_id,
            success=replay.ok,
            trace_fingerprint=self.fingerprint,
            sink_result=redact_evidence(sink_result),
        )


def _normalized_result(
    provider_name: str,
    result: Any,
    normalizers: Mapping[str, TraceNormalizer],
) -> Any:
    normalizer = normalizers.get(provider_name)
    return normalizer(result) if normalizer else result


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _thaw_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): _freeze_json(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "EXECUTION_TRACE_VERSION",
    "ExecutionReplayReport",
    "ExecutionTraceEvent",
    "ExecutionTraceLedger",
    "OutcomeFeedbackReceipt",
    "OutcomeFeedbackSink",
    "ReplayDispatch",
    "ReplayMismatch",
    "TraceNormalizer",
]
