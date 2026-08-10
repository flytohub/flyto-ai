# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Stable stdio supervisor for hot-reloading the coding MCP worker.

Codex keeps local MCP commands alive for the lifetime of a loaded task. Python
modules imported by such a process do not change when their source files are
repaired. This supervisor owns the client stdio connection and delegates to a
replaceable ``code-mcp`` child. When the coding source digest changes, it
restarts only that child at a safe job boundary and replays the MCP handshake.

The proxy never retries a request whose delivery is uncertain and never
replaces a worker while a job *this connection submitted* is non-terminal. New
submissions are failed closed with ``service_reload_pending`` until that
exact-session job reaches a terminal state. Jobs merely observed through the
tenant-visible ``flyto_coding_get`` and ``flyto_coding_audit`` tools belong to
other supervisors and never hold this worker back.

Every read from the worker is deadlined. This service only schedules or
inspects background work, so a call that has not answered within seconds is a
wedged worker, not a slow one. Waiting forever would hang the Codex frontend
and leave the shared state-root locks held by a process that will never write
again, so the deadline ends with a bounded protocol error and a terminated
worker rather than with silence.
"""
from __future__ import annotations

import hashlib
import json
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Mapping, Optional, Sequence

from flyto_ai.coding.route_status import current_service_build_id


MAX_SUPERVISOR_MESSAGE_BYTES = 256 * 1024
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 5.0
#: Submit, get, and audit all return as soon as the service has recorded a
#: durable decision; none of them waits for an implementation round. A read
#: that misses this bound describes a worker that is not going to answer.
WORKER_RESPONSE_TIMEOUT_SECONDS = 30.0
#: The handshake is a pure in-process reply, so it is held to the same bound.
WORKER_HANDSHAKE_TIMEOUT_SECONDS = 30.0
TERMINAL_JOB_STATES = frozenset({"completed", "failed", "codex_accepted"})
#: Mirrors the `--state-dir` default of `code-mcp` in `flyto_ai.cli`. It is a
#: literal rather than an import so this module does not drag the CLI into
#: every supervisor process; `tests/test_coding_mcp_supervisor.py` asserts the
#: two stay equal, so a change to either is caught rather than silently split.
DEFAULT_CODING_STATE_DIR = "~/.flyto/coding-service"
_JOB_ID_RE = re.compile(r"^job_[a-f0-9]{24}$")
_SAFE_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class CodingMCPWorkerUnavailable(RuntimeError):
    """The replaceable worker could not serve a request deterministically."""


class CodingMCPWorkerTimeout(CodingMCPWorkerUnavailable):
    """The worker did not answer within this service's bounded deadline."""


class _WorkerChannel:
    """One worker process plus the thread that drains its stdout.

    A dedicated reader thread is used instead of polling the pipe directly so
    the deadline behaves identically on every platform this package supports,
    rather than only where a subprocess pipe happens to be selectable.
    """

    def __init__(self, argv: Sequence[str]) -> None:
        self.process: subprocess.Popen[bytes] = subprocess.Popen(
            tuple(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        self.lines: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        stdout = self.process.stdout
        if stdout is None:
            self.lines.put(None)
            return
        try:
            while True:
                line = stdout.readline(MAX_SUPERVISOR_MESSAGE_BYTES + 1)
                if not line:
                    break
                self.lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            # `None` is end-of-stream, so a waiting request learns the worker
            # died instead of sitting out its whole deadline.
            self.lines.put(None)

    def write(self, raw: bytes) -> None:
        stdin = self.process.stdin
        if stdin is None:
            raise CodingMCPWorkerUnavailable("worker pipes are unavailable")
        stdin.write(raw)
        stdin.flush()

    def read(self, timeout: float) -> bytes:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as exc:
            raise CodingMCPWorkerTimeout("worker response exceeded its deadline") from exc
        if line is None:
            raise CodingMCPWorkerUnavailable("worker closed its output stream")
        return line

    def stop(self, *, graceful: bool = True) -> None:
        """End the worker, then release its pipes and reader deterministically.

        Shutdown and release are separate because a worker that exits cleanly
        and one that has to be killed must both leave nothing behind. Releasing
        in a `finally` is what guarantees that: an early return on the happy
        path used to leak the stdout pipe and strand the reader thread.
        """

        try:
            self._end_process(graceful=graceful)
        finally:
            self._release()

    def _end_process(self, *, graceful: bool) -> None:
        """Close, then terminate, then kill. A wedged worker still holds locks.

        `graceful=False` skips straight to termination. A worker that already
        missed its deadline has proven it is not reading its input, so waiting
        another shutdown timeout for a clean exit only delays the release of
        the state-root locks it is still holding.
        """

        process = self.process
        if graceful:
            try:
                if process.stdin is not None:
                    process.stdin.close()
                process.wait(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
                return
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        process.terminate()
        try:
            process.wait(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            try:
                # Even the post-kill reap is bounded. An unreapable process is
                # a host-level fault, and blocking the client's stdio loop on
                # it forever would break the same no-hang contract the response
                # deadline exists to keep.
                process.wait(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _release(self) -> None:
        """Join the reader and close both pipes, without ever blocking forever.

        The join runs first and is bounded: the process has normally exited, so
        the reader's blocking `readline` has returned end-of-stream and the
        thread is finishing. Closing stdout before joining would instead race a
        live reader.

        If that first join times out the reader is still blocked on a pipe, so
        the pipes are closed to unblock it — a closed stream reads as
        end-of-stream — and only the remaining cleanup budget is spent on a
        final join. Total time here is bounded by twice the shutdown timeout.
        """

        self._reader.join(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe is None:
                continue
            try:
                pipe.close()
            except OSError:
                pass
        if self._reader.is_alive():
            self._reader.join(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)


def durable_job_state_reader(
    state_dir: str, tenant_id: str,
) -> Optional[Callable[[str], Optional[str]]]:
    """Build a reader for one tenant's durable job states, or return `None`.

    The supervisor is not the coding service and must not act like one: this
    reads exactly one bounded job record per known job id, by exact path, and
    returns only its state string. It never lists a directory, never parses a
    request, and never writes.
    """

    if not state_dir or not _SAFE_TENANT_RE.fullmatch(tenant_id or ""):
        return None
    tenant_ref = hashlib.sha256(tenant_id.encode()).hexdigest()
    jobs = Path(state_dir).expanduser().resolve() / "tenants" / tenant_ref / "jobs"

    def read_state(job_id: str) -> Optional[str]:
        if not _JOB_ID_RE.fullmatch(job_id):
            return None
        try:
            raw = (jobs / (job_id + ".json")).read_bytes()
        except OSError:
            return None
        if len(raw) > MAX_SUPERVISOR_MESSAGE_BYTES:
            return None
        try:
            record = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError, ValueError):
            return None
        state = record.get("state") if isinstance(record, Mapping) else None
        return state if isinstance(state, str) else None

    return read_state


class CodingMCPWorkerSupervisor:
    """Proxy one MCP connection to a safely replaceable coding worker."""

    def __init__(
        self,
        worker_argv: Sequence[str],
        *,
        stdin: Optional[BinaryIO] = None,
        stdout: Optional[BinaryIO] = None,
        build_id_provider: Callable[[], str] = current_service_build_id,
        job_state_reader: Optional[Callable[[str], Optional[str]]] = None,
        response_timeout: float = WORKER_RESPONSE_TIMEOUT_SECONDS,
        handshake_timeout: float = WORKER_HANDSHAKE_TIMEOUT_SECONDS,
    ) -> None:
        if not worker_argv or not all(isinstance(item, str) and item for item in worker_argv):
            raise ValueError("worker argv must contain non-empty strings")
        if not 0 < response_timeout <= WORKER_RESPONSE_TIMEOUT_SECONDS:
            raise ValueError("response_timeout must be a positive bounded number of seconds")
        if not 0 < handshake_timeout <= WORKER_HANDSHAKE_TIMEOUT_SECONDS:
            raise ValueError("handshake_timeout must be a positive bounded number of seconds")
        self.worker_argv = tuple(worker_argv)
        self.stdin = stdin or sys.stdin.buffer
        self.stdout = stdout or sys.stdout.buffer
        self._build_id_provider = build_id_provider
        self._job_state_reader = job_state_reader
        self.response_timeout = float(response_timeout)
        self.handshake_timeout = float(handshake_timeout)
        self._channel: Optional[_WorkerChannel] = None
        self._worker_build_id = ""
        self._initialize_request: Optional[Dict[str, Any]] = None
        self._initialized_notification: Optional[Dict[str, Any]] = None
        self._initialized = False
        self._active_jobs: Dict[str, str] = {}
        self.reload_count = 0
        self.timeout_count = 0

    @property
    def worker_pid(self) -> int:
        return int(self._channel.process.pid) if self._channel is not None else 0

    def serve(self) -> None:
        """Serve bounded newline-delimited MCP messages until client EOF."""

        try:
            for raw in self.stdin:
                response = self.handle_line(raw)
                if response is not None:
                    self.stdout.write(response)
                    self.stdout.flush()
        finally:
            self.close()

    def handle_line(self, raw: bytes) -> Optional[bytes]:
        """Handle one client message; exposed separately for deterministic tests."""

        if len(raw) > MAX_SUPERVISOR_MESSAGE_BYTES:
            return self._protocol_error(None, -32600, "request exceeds message limit")
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError
        except (UnicodeError, json.JSONDecodeError, ValueError):
            return self._protocol_error(None, -32700, "parse error")

        method = request.get("method")
        if method == "initialize" and request.get("id") is not None:
            self._initialize_request = dict(request)

        try:
            source_changed = self._ensure_worker()
            if source_changed and self._active_jobs and self._is_submit(request):
                return self._tool_error(request.get("id"), "service_reload_pending")
            response = self._exchange(raw, expect_response=request.get("id") is not None)
        except CodingMCPWorkerTimeout:
            # The worker owes a response it will never send. Terminating it
            # releases the state-root locks it still holds and lets the next
            # request start a worker that reconciles the interrupted job
            # truthfully. This request is not retried: its delivery is
            # uncertain, and a caller recovers a submitted job by replaying the
            # same idempotency key, which the supervisor must never do for them.
            self.timeout_count += 1
            self._stop_worker(graceful=False)
            return self._protocol_error(
                request.get("id"), -32603, "coding worker exceeded its deadline",
            )
        except (OSError, CodingMCPWorkerUnavailable):
            # A broken pipe, a malformed frame, or an oversized response all
            # mean this worker's stream is no longer trustworthy: the next
            # request could read a leftover frame and answer the wrong caller.
            # Release it for the same reason a timeout does, and for the same
            # reason do not resend — delivery of this request is uncertain.
            self._stop_worker(graceful=False)
            return self._protocol_error(request.get("id"), -32603, "coding worker unavailable")

        if response is not None:
            self._observe_job(response, request)
            if method == "initialize":
                self._initialized = True
        if method == "notifications/initialized":
            # Cache only after forwarding it. If source changed between the
            # initialize response and this notification, _ensure_worker replays
            # initialize and this request completes that new handshake once.
            self._initialized_notification = dict(request)
        return response

    def close(self) -> None:
        """Close the current worker without touching any other MCP instance."""

        self._stop_worker()

    def _ensure_worker(self) -> bool:
        current_build = self._build_id_provider()
        channel = self._channel
        if channel is None or channel.process.poll() is not None:
            self._channel = None
            self._start_worker(current_build, replay=self._initialized)
            return False
        if current_build == self._worker_build_id:
            return False
        # Only a job that is still non-terminal in durable state can hold a
        # worker back. A client that stopped polling must not pin a stale
        # in-memory entry and block every later submission forever.
        self._forget_settled_jobs()
        if self._active_jobs:
            return True
        self._restart_worker(current_build)
        return False

    def _forget_settled_jobs(self) -> None:
        """Drop tracked jobs the durable record already reports as terminal."""

        reader = self._job_state_reader
        if reader is None or not self._active_jobs:
            return
        for job_id in tuple(self._active_jobs):
            state = reader(job_id)
            if state is not None and state in TERMINAL_JOB_STATES:
                self._active_jobs.pop(job_id, None)

    def _start_worker(self, build_id: str, *, replay: bool) -> None:
        self._channel = _WorkerChannel(self.worker_argv)
        self._worker_build_id = build_id
        if replay:
            self._replay_handshake()

    def _restart_worker(self, build_id: str) -> None:
        self._stop_worker()
        self._start_worker(build_id, replay=self._initialized)
        self.reload_count += 1

    def _stop_worker(self, *, graceful: bool = True) -> None:
        channel = self._channel
        self._channel = None
        self._worker_build_id = ""
        if channel is not None:
            channel.stop(graceful=graceful)

    def _replay_handshake(self) -> None:
        if self._initialize_request is None:
            return
        replay = dict(self._initialize_request)
        replay["id"] = "flyto-supervisor-initialize"
        encoded = json.dumps(replay, ensure_ascii=False, separators=(",", ":")).encode()
        response = self._exchange(
            encoded + b"\n", expect_response=True, timeout=self.handshake_timeout,
        )
        if response is None:
            raise CodingMCPWorkerUnavailable("worker initialization returned no response")
        try:
            value = json.loads(response)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CodingMCPWorkerUnavailable("worker initialization was malformed") from exc
        if not isinstance(value, Mapping) or "error" in value:
            raise CodingMCPWorkerUnavailable("worker initialization failed")
        if self._initialized_notification is not None:
            notification = json.dumps(
                self._initialized_notification,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode() + b"\n"
            self._exchange(notification, expect_response=False)

    def _exchange(
        self,
        raw: bytes,
        *,
        expect_response: bool,
        timeout: Optional[float] = None,
    ) -> Optional[bytes]:
        channel = self._channel
        if channel is None:
            raise CodingMCPWorkerUnavailable("worker pipes are unavailable")
        try:
            channel.write(raw)
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise CodingMCPWorkerUnavailable("worker transport failed") from exc
        if not expect_response:
            return None
        response = channel.read(
            self.response_timeout if timeout is None else timeout,
        )
        if len(response) > MAX_SUPERVISOR_MESSAGE_BYTES:
            raise CodingMCPWorkerUnavailable("worker response is missing or oversized")
        return response

    def _observe_job(self, raw: bytes, request: Mapping[str, Any]) -> None:
        """Update tracked jobs from one response, bounded by its own request.

        Ownership is decided by the request a response answers, never by the
        response alone. `flyto_coding_get` and `flyto_coding_audit` are
        tenant-visible: they answer truthfully about jobs a *different*
        supervisor submitted, on a different workspace, over a different
        connection. Registering a non-terminal job seen through them would pin
        this connection's worker to work it does not own, so every submission
        here would be refused with `service_reload_pending` until somebody
        else's job settled.

        Only a successful `flyto_coding_submit` on this connection registers a
        job. `_active_jobs` is therefore exactly the set of jobs submitted
        locally and still non-terminal: a terminal observation may clear an
        entry — including through `get` or `audit`, which is how a caller
        reports its own job settled — but no observation can ever create one.
        An idempotent submit replay names the same locally submitted job, so it
        re-registers it and tracking survives the replay.
        """

        observed = self._observed_job(raw, request)
        if observed is None:
            return
        job_id, state = observed
        if state in TERMINAL_JOB_STATES:
            # Clears only what is already tracked; a foreign job is not.
            self._active_jobs.pop(job_id, None)
            return
        if self._tool_name(request) != "flyto_coding_submit":
            return
        self._active_jobs[job_id] = state

    @staticmethod
    def _observed_job(
        raw: bytes, request: Mapping[str, Any],
    ) -> Optional[tuple[str, str]]:
        """Return the `(job_id, state)` this response truthfully reports.

        Anything short of a successful, correctly addressed, well-formed job
        receipt returns `None` and therefore changes no tracked state. The id
        check keeps the observation bound to the request that caused it, and
        the job-id pattern check keeps a tracked job reconcilable: an id the
        durable reader would refuse could never be released again.
        """

        try:
            response = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError, ValueError):
            return None
        if not isinstance(response, Mapping) or "error" in response:
            return None
        if response.get("id") != request.get("id"):
            return None
        result = response.get("result")
        if not isinstance(result, Mapping) or result.get("isError"):
            return None
        structured = result.get("structuredContent")
        if not isinstance(structured, Mapping) or structured.get("ok") is not True:
            return None
        job = structured.get("job")
        if not isinstance(job, Mapping):
            return None
        job_id = job.get("job_id")
        state = job.get("state")
        if not isinstance(job_id, str) or not isinstance(state, str):
            return None
        if not _JOB_ID_RE.fullmatch(job_id):
            return None
        return job_id, state

    @staticmethod
    def _tool_name(request: Mapping[str, Any]) -> str:
        if request.get("method") != "tools/call":
            return ""
        params = request.get("params")
        if not isinstance(params, Mapping):
            return ""
        name = params.get("name")
        return name if isinstance(name, str) else ""

    @classmethod
    def _is_submit(cls, request: Mapping[str, Any]) -> bool:
        return cls._tool_name(request) == "flyto_coding_submit"

    @staticmethod
    def _protocol_error(request_id: Any, code: int, message: str) -> bytes:
        return json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }, separators=(",", ":")).encode() + b"\n"

    @staticmethod
    def _tool_error(request_id: Any, code: str) -> bytes:
        payload = {"ok": False, "error": code}
        return json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "isError": True,
                "structuredContent": payload,
            },
        }, separators=(",", ":")).encode() + b"\n"


def worker_argv_from_process(argv: Sequence[str]) -> tuple[str, ...]:
    """Replace the supervisor subcommand with an isolated worker command."""

    try:
        command_index = tuple(argv).index("code-mcp-supervisor")
    except ValueError as exc:
        raise ValueError("code-mcp-supervisor command is missing") from exc
    return (
        sys.executable,
        "-m",
        "flyto_ai.cli",
        "code-mcp",
        *tuple(argv)[command_index + 1 :],
    )


def _option_value(argv: Sequence[str], name: str) -> str:
    """Read one `--flag value` or `--flag=value` pair without argparse.

    The supervisor deliberately does not re-parse the worker's full command
    line: it needs two values to locate durable job records, and inventing a
    second parser would be a second place for authority to drift.
    """

    items = tuple(argv)
    for index, item in enumerate(items):
        if item == name and index + 1 < len(items):
            return items[index + 1]
        if item.startswith(name + "="):
            return item[len(name) + 1 :]
    return ""


def supervisor_from_argv(argv: Sequence[str]) -> CodingMCPWorkerSupervisor:
    """Build the supervisor one CLI invocation implies.

    `--state-dir` is optional on `code-mcp`, so an omitted flag means the CLI
    default rather than "no durable state". Reading it as empty would silently
    disable reconciliation from durable job records — the very mechanism that
    stops a client which stopped polling from pinning reloads forever.
    """

    return CodingMCPWorkerSupervisor(
        worker_argv_from_process(argv),
        job_state_reader=durable_job_state_reader(
            _option_value(argv, "--state-dir") or DEFAULT_CODING_STATE_DIR,
            _option_value(argv, "--tenant"),
        ),
    )


def serve_supervised_stdio(argv: Sequence[str]) -> None:
    """Run the stable supervisor for the current CLI invocation."""

    supervisor_from_argv(argv).serve()
