# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Durable, fail-closed admission host for prepared execution sessions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import signal
import socket
import struct
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from flyto_ai.execution_session import ExecutionAuthority, prepare_execution_session
from flyto_ai.scheduler import ScheduledTask, Scheduler, TaskSchedule
from flyto_ai.scheduler.tasks import ScheduleType

EXECUTION_SESSION_HOST_INSTRUCTION_VERSION = "flyto.execution-session-host-instruction.v1"
EXECUTION_SESSION_HOST_RECEIPT_VERSION = "flyto.execution-session-host-receipt.v1"
_TASK_PREFIX = "execution-session-"
_DIGEST_PREFIX = "sha256:"
_DIGEST_LENGTH = len(_DIGEST_PREFIX) + 64
_MAX_RECEIPT_BYTES = 16_384
_MAX_CONNECTOR_FRAME_BYTES = 131_072
_CONNECTOR_RESULT_KEYS = {"ok", "message", "error", "cost_usd"}
_RECONCILE_INTERVAL_SECONDS = 0.01
_RECONCILE_GRACE_SECONDS = 0.5

ExecutionSessionCallback = Callable[
    [Mapping[str, Any]], Awaitable[Mapping[str, Any]]
]


class ExecutionSessionHostError(ValueError):
    """Raised when durable admission cannot be proven safe and unambiguous."""


def _task_id(session_id: str) -> str:
    """Map the exact, already-validated session identifier to a safe token."""
    return _TASK_PREFIX + hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _instruction(prepared: Mapping[str, Any]) -> dict[str, Any]:
    attestations = prepared.get("attestations")
    planning_input = prepared.get("planning_input")
    if not isinstance(attestations, Mapping) or not isinstance(planning_input, Mapping):
        raise ExecutionSessionHostError("prepared session has an invalid shape")
    session_id = planning_input.get("session_id")
    digests = {
        "request": attestations.get("request"),
        "authority": attestations.get("authority"),
        "route": attestations.get("route"),
        "overall": prepared.get("overall_digest"),
    }
    if not isinstance(session_id, str) or any(
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or not value.startswith(_DIGEST_PREFIX)
        for value in digests.values()
    ):
        raise ExecutionSessionHostError("prepared session has invalid correlation")
    return {
        "contract_version": EXECUTION_SESSION_HOST_INSTRUCTION_VERSION,
        "session": {"session_id": session_id, "task_id": _task_id(session_id)},
        "digests": digests,
    }


def _encode(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if len(encoded.encode("utf-8")) > _MAX_RECEIPT_BYTES:
        raise ExecutionSessionHostError("admission data exceeds its durable bound")
    return encoded


async def _admission_executor(_instruction_json: str) -> dict[str, Any]:
    return {"ok": False, "error": "execution_not_connected"}


def _detached_prepared(prepared: Mapping[str, Any]) -> Mapping[str, Any]:
    """Give a connector an isolated plain-JSON snapshot, never host containers."""
    try:
        encoded = json.dumps(
            prepared,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        detached = json.loads(encoded)
    except (RecursionError, TypeError, ValueError, OverflowError) as exc:
        raise ExecutionSessionHostError("prepared session is not detached JSON") from exc
    if not isinstance(detached, dict):
        raise ExecutionSessionHostError("prepared session has an invalid shape")
    return detached


def _remaining_seconds(prepared: Mapping[str, Any], now_ms: int) -> float:
    planning_input = prepared.get("planning_input")
    activation = (
        planning_input.get("activation")
        if isinstance(planning_input, Mapping)
        else None
    )
    expires_at_ms = (
        activation.get("expires_at_ms") if isinstance(activation, Mapping) else None
    )
    if (
        isinstance(expires_at_ms, bool)
        or not isinstance(expires_at_ms, int)
        or expires_at_ms <= now_ms
    ):
        raise ExecutionSessionHostError("prepared session has an invalid deadline")
    return (expires_at_ms - now_ms) / 1000.0


async def _run_connector_child(
    connector: ExecutionSessionCallback, prepared: Mapping[str, Any]
) -> str:
    """Run and reduce connector output inside the disposable child process."""
    try:
        raw = await connector(prepared)
    except BaseException:
        return "exception"
    if type(raw) is not dict or set(raw) != _CONNECTOR_RESULT_KEYS:
        return "invalid"
    ok = raw.get("ok")
    message = raw.get("message")
    error = raw.get("error")
    cost = raw.get("cost_usd")
    if type(ok) is not bool or type(message) is not str:
        return "invalid"
    if type(cost) not in (int, float) or cost != 0.0:
        return "invalid"
    if ok is True:
        return "success" if message == "" and error is None else "invalid"
    return (
        "failure"
        if message == "" and error == "execution_connector_failed"
        else "invalid"
    )


def _recv_exact(channel: socket.socket, size: int) -> bytes:
    """Receive one bounded worker frame or fail without partial interpretation."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = channel.recv(remaining)
        if not chunk:
            raise EOFError("connector worker channel closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _connector_worker_main(
    connector: ExecutionSessionCallback,
    channel: socket.socket,
) -> None:
    """Pre-established child entry point with a bounded one-shot protocol."""
    try:
        os.setsid()
        channel.sendall(b"R")
        size = struct.unpack("!I", _recv_exact(channel, 4))[0]
        if size > _MAX_CONNECTOR_FRAME_BYTES:
            raise ValueError("connector worker frame exceeds its bound")
        envelope = json.loads(_recv_exact(channel, size))
        if not isinstance(envelope, dict):
            raise ValueError("connector worker frame is invalid")
        deadline = envelope.get("deadline")
        prepared = envelope.get("prepared")
        if type(deadline) is not float or not isinstance(prepared, dict):
            raise ValueError("connector worker frame is invalid")
        if time.monotonic() >= deadline:
            channel.sendall(b"T")
            return
        outcome = asyncio.run(_run_connector_child(connector, prepared))
        channel.sendall(
            {
                "success": b"S",
                "failure": b"F",
                "invalid": b"I",
                "exception": b"E",
            }.get(outcome, b"I")
        )
    except BaseException:
        try:
            channel.sendall(b"E")
        except BaseException:
            pass
    finally:
        channel.close()


def _stop_connector_process(process: multiprocessing.Process) -> None:
    """Enforce bounded child termination before any caller can observe closure."""
    if process.pid is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (PermissionError, ProcessLookupError):
            pass
        if process.is_alive():
            process.kill()
    process.join(timeout=_RECONCILE_GRACE_SECONDS)
    if process.is_alive():
        raise ExecutionSessionHostError("connector isolation could not be terminated")


class ExecutionSessionConnector:
    """One-shot trusted connector in a process established before admission."""

    def __init__(self, connector: ExecutionSessionCallback) -> None:
        if not callable(connector):
            raise ExecutionSessionHostError("connector callback must be callable")
        try:
            context = multiprocessing.get_context("fork")
        except ValueError as exc:
            raise ExecutionSessionHostError(
                "trusted connectors require enforceable process isolation"
            ) from exc
        parent, child = socket.socketpair()
        parent.setblocking(False)
        self._channel = parent
        self._process = context.Process(
            target=_connector_worker_main,
            args=(connector, child),
            daemon=False,
        )
        self._closed = False
        try:
            self._process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()

    def close(self) -> None:
        """Forcibly terminate and reap the owned worker exactly once."""
        if self._closed:
            return
        self._closed = True
        try:
            _stop_connector_process(self._process)
        finally:
            self._channel.close()

    async def _receive(self, deadline: float, *, maximum: int) -> bytes | None:
        value = bytearray()
        while len(value) < maximum:
            if asyncio.get_running_loop().time() >= deadline:
                return None
            try:
                chunk = self._channel.recv(maximum - len(value))
            except BlockingIOError:
                if not self._process.is_alive():
                    return bytes(value)
                await asyncio.sleep(_RECONCILE_INTERVAL_SECONDS)
                continue
            if not chunk:
                return bytes(value)
            value.extend(chunk)
            return bytes(value)
        return bytes(value)

    async def _send(self, value: bytes, deadline: float) -> bool:
        remaining = memoryview(value)
        while remaining:
            if asyncio.get_running_loop().time() >= deadline:
                return False
            try:
                sent = self._channel.send(remaining)
            except BlockingIOError:
                if not self._process.is_alive():
                    return False
                await asyncio.sleep(_RECONCILE_INTERVAL_SECONDS)
                continue
            remaining = remaining[sent:]
        return True

    async def invoke(
        self, prepared: Mapping[str, Any], activation_deadline: float
    ) -> str:
        """Activate the ready worker within the caller's absolute deadline."""
        if self._closed:
            raise ExecutionSessionHostError("connector isolation is closed")
        ready = await self._receive(activation_deadline, maximum=1)
        if ready is None:
            return "timeout"
        if ready != b"R":
            return "exception"
        remaining_seconds = max(
            0.0, activation_deadline - asyncio.get_running_loop().time()
        )
        if remaining_seconds == 0.0:
            return "timeout"
        envelope = json.dumps(
            {
                "deadline": activation_deadline,
                "prepared": _detached_prepared(prepared),
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(envelope) > _MAX_CONNECTOR_FRAME_BYTES:
            raise ExecutionSessionHostError("connector input exceeds its bound")
        frame = struct.pack("!I", len(envelope)) + envelope
        if not await self._send(frame, activation_deadline):
            return "timeout"
        outcome = await self._receive(activation_deadline, maximum=1)
        if outcome is None:
            return "timeout"
        return {
            b"S": "success",
            b"F": "failure",
            b"I": "invalid",
            b"E": "exception",
            b"T": "timeout",
        }.get(outcome, "invalid")


def _connector_executor(
    connector: ExecutionSessionConnector,
    prepared: Mapping[str, Any],
    activation_deadline: float,
) -> Callable[[str], Awaitable[dict[str, Any]]]:
    """Adapt one trusted in-memory connector to the Scheduler's exact boundary."""
    connector_input = _detached_prepared(prepared)

    async def execute(_instruction_json: str) -> dict[str, Any]:
        # Each possible invocation gets a fresh copy. The durable Scheduler fence
        # remains the sole authority deciding whether invocation happens at all.
        remaining_seconds = max(
            0.0, activation_deadline - asyncio.get_running_loop().time()
        )
        if remaining_seconds == 0.0:
            return {
                "ok": False,
                "message": "",
                "error": "execution_connector_timeout",
                "cost_usd": 0.0,
            }
        try:
            outcome = await connector.invoke(connector_input, activation_deadline)
        finally:
            connector.close()
        if outcome == "timeout":
            return {
                "ok": False,
                "message": "",
                "error": "execution_connector_timeout",
                "cost_usd": 0.0,
            }
        if outcome == "invalid":
            raise ExecutionSessionHostError("connector result has an invalid shape")
        if outcome == "exception":
            raise ExecutionSessionHostError("connector execution failed")
        if outcome == "success":
            return {"ok": True, "message": "", "error": None, "cost_usd": 0.0}
        if outcome != "failure":
            raise ExecutionSessionHostError("connector result has an invalid outcome")
        return {
            "ok": False,
            "message": "",
            "error": "execution_connector_failed",
            "cost_usd": 0.0,
        }

    return execute


def _receipt(
    instruction: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    if set(result) != {"slot", "ok", "cost_usd", "message", "error", "evidence_ref"}:
        raise ExecutionSessionHostError("durable occurrence result has an invalid shape")
    ok = result.get("ok")
    error = result.get("error")
    if (
        type(ok) is not bool
        or result.get("message") != ""
        or result.get("cost_usd") != 0.0
        or (ok is True and error is not None)
        or (
            ok is False
            and error
            not in {
                "execution_not_connected",
                "execution_connector_failed",
                "execution_connector_timeout",
                "execution_outcome_unknown",
                "executor_failed",
            }
        )
    ):
        raise ExecutionSessionHostError("durable occurrence has an invalid outcome")
    receipt = {
        "contract_version": EXECUTION_SESSION_HOST_RECEIPT_VERSION,
        "session": dict(instruction["session"]),
        "digests": dict(instruction["digests"]),
        "status": "connected" if ok else "blocked",
        "result": dict(result),
    }
    # Round-trip through strict JSON to return containers detached from scheduler state.
    return json.loads(_encode(receipt))


async def admit_execution_session(
    state_root: os.PathLike[str] | str,
    untrusted_request: Mapping[str, Any],
    trusted_manifests: Sequence[Mapping[str, Any]],
    authority: ExecutionAuthority,
    now_ms: int,
    *,
    trusted_blueprints: Sequence[Mapping[str, Any]] = (),
    limit: int = 8,
    trusted_connector: ExecutionSessionConnector | None = None,
) -> Mapping[str, Any]:
    """Prepare and durably admit one provider-neutral execution occurrence."""
    if state_root is None or isinstance(state_root, bool):
        raise ExecutionSessionHostError("state_root must select durable state")
    try:
        state_path = os.fspath(state_root)
    except TypeError as exc:
        raise ExecutionSessionHostError("state_root must be a filesystem path") from exc
    if not state_path:
        raise ExecutionSessionHostError("state_root must select durable state")
    if trusted_connector is not None and not isinstance(
        trusted_connector, ExecutionSessionConnector
    ):
        raise ExecutionSessionHostError(
            "trusted_connector must be a pre-established connector"
        )

    loop = asyncio.get_running_loop()
    admission_started = loop.time()
    try:
        prepared = prepare_execution_session(
            untrusted_request,
            trusted_manifests,
            authority,
            now_ms,
            trusted_blueprints=trusted_blueprints,
            limit=limit,
        )
        instruction = _instruction(prepared)
        activation_deadline = admission_started + _remaining_seconds(prepared, now_ms)
        reconciliation_deadline = activation_deadline + _RECONCILE_GRACE_SECONDS
        instruction_json = _encode(instruction)
        task_id = instruction["session"]["task_id"]
        executor = (
            _admission_executor
            if trusted_connector is None
            else _connector_executor(trusted_connector, prepared, activation_deadline)
        )
        scheduler = Scheduler(executor=executor, state_root=state_root)

        existing = scheduler.get_task(task_id)
        if existing is None:
            task = ScheduledTask(
                task_id=task_id,
                name="Execution session admission",
                instruction=instruction_json,
                schedule=TaskSchedule(type=ScheduleType.ONE_SHOT, run_at=0),
                tags=["execution-session-admission"],
            )
            try:
                scheduler.add_task(task)
            except ValueError:
                # A concurrent admission may have won the durable unique-key race.
                existing = scheduler.get_task(task_id)
                if existing is None:
                    raise
        if existing is not None and existing.instruction != instruction_json:
            raise ExecutionSessionHostError(
                "session admission conflicts with durable state"
            )
    except BaseException:
        if trusted_connector is not None:
            trusted_connector.close()
        raise

    try:
        await scheduler.run_once()
        persisted = scheduler.get_task(task_id)
        if persisted is None or persisted.instruction != instruction_json:
            raise ExecutionSessionHostError("session admission durable state is unresolved")
        while True:
            summary = scheduler.summary()
            matching = [item for item in summary["tasks"] if item["task_id"] == task_id]
            if len(matching) != 1:
                raise ExecutionSessionHostError("session admission occurrence is unresolved")
            if len(matching[0]["results"]) == 1:
                break
            if matching[0]["results"]:
                raise ExecutionSessionHostError("session admission occurrence is unresolved")
            # Another host may own the durable fence. Its live dispatch lease makes
            # this pass a no-op. If that owner exits after connector entry, the
            # persisted fence lets this already-waiting duplicate reconcile the
            # occurrence to unknown without entering its own connector.
            if asyncio.get_running_loop().time() >= reconciliation_deadline:
                raise ExecutionSessionHostError("session admission reconciliation timed out")
            await asyncio.sleep(_RECONCILE_INTERVAL_SECONDS)
            await scheduler.run_once()
        return _receipt(instruction, matching[0]["results"][0])
    finally:
        if trusted_connector is not None:
            trusted_connector.close()


__all__ = [
    "EXECUTION_SESSION_HOST_INSTRUCTION_VERSION",
    "EXECUTION_SESSION_HOST_RECEIPT_VERSION",
    "ExecutionSessionConnector",
    "ExecutionSessionHostError",
    "admit_execution_session",
]
