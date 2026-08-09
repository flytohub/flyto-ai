# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Stable stdio supervisor for hot-reloading the coding MCP worker.

Codex keeps local MCP commands alive for the lifetime of a loaded task. Python
modules imported by such a process do not change when their source files are
repaired. This supervisor owns the client stdio connection and delegates to a
replaceable ``code-mcp`` child. When the coding source digest changes, it
restarts only that child at a safe job boundary and replays the MCP handshake.

The proxy never retries a request whose delivery is uncertain and never
replaces a worker while a known job is non-terminal. New submissions are
failed closed with ``service_reload_pending`` until the existing exact-session
job reaches a terminal state.
"""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Any, BinaryIO, Callable, Dict, Mapping, Optional, Sequence

from flyto_ai.coding.route_status import current_service_build_id


MAX_SUPERVISOR_MESSAGE_BYTES = 256 * 1024
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 5.0
TERMINAL_JOB_STATES = frozenset({"completed", "failed", "codex_accepted"})


class CodingMCPWorkerUnavailable(RuntimeError):
    """The replaceable worker could not serve a request deterministically."""


class CodingMCPWorkerSupervisor:
    """Proxy one MCP connection to a safely replaceable coding worker."""

    def __init__(
        self,
        worker_argv: Sequence[str],
        *,
        stdin: Optional[BinaryIO] = None,
        stdout: Optional[BinaryIO] = None,
        build_id_provider: Callable[[], str] = current_service_build_id,
    ) -> None:
        if not worker_argv or not all(isinstance(item, str) and item for item in worker_argv):
            raise ValueError("worker argv must contain non-empty strings")
        self.worker_argv = tuple(worker_argv)
        self.stdin = stdin or sys.stdin.buffer
        self.stdout = stdout or sys.stdout.buffer
        self._build_id_provider = build_id_provider
        self._worker: Optional[subprocess.Popen[bytes]] = None
        self._worker_build_id = ""
        self._initialize_request: Optional[Dict[str, Any]] = None
        self._initialized_notification: Optional[Dict[str, Any]] = None
        self._initialized = False
        self._active_jobs: Dict[str, str] = {}
        self.reload_count = 0

    @property
    def worker_pid(self) -> int:
        return int(self._worker.pid) if self._worker is not None else 0

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
        except (OSError, CodingMCPWorkerUnavailable):
            return self._protocol_error(request.get("id"), -32603, "coding worker unavailable")

        if response is not None:
            self._observe_job(response)
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
        if self._worker is None or self._worker.poll() is not None:
            self._start_worker(current_build, replay=self._initialized)
            return False
        if current_build == self._worker_build_id:
            return False
        if self._active_jobs:
            return True
        self._restart_worker(current_build)
        return False

    def _start_worker(self, build_id: str, *, replay: bool) -> None:
        self._worker = subprocess.Popen(
            self.worker_argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        self._worker_build_id = build_id
        if replay:
            self._replay_handshake()

    def _restart_worker(self, build_id: str) -> None:
        self._stop_worker()
        self._start_worker(build_id, replay=self._initialized)
        self.reload_count += 1

    def _stop_worker(self) -> None:
        worker = self._worker
        self._worker = None
        self._worker_build_id = ""
        if worker is None:
            return
        try:
            if worker.stdin is not None:
                worker.stdin.close()
            worker.wait(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
        except (OSError, subprocess.TimeoutExpired):
            worker.terminate()
            try:
                worker.wait(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                worker.kill()
                worker.wait()

    def _replay_handshake(self) -> None:
        if self._initialize_request is None:
            return
        replay = dict(self._initialize_request)
        replay["id"] = "flyto-supervisor-initialize"
        encoded = json.dumps(replay, ensure_ascii=False, separators=(",", ":")).encode()
        response = self._exchange(encoded + b"\n", expect_response=True)
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

    def _exchange(self, raw: bytes, *, expect_response: bool) -> Optional[bytes]:
        worker = self._worker
        if worker is None or worker.stdin is None or worker.stdout is None:
            raise CodingMCPWorkerUnavailable("worker pipes are unavailable")
        try:
            worker.stdin.write(raw)
            worker.stdin.flush()
            if not expect_response:
                return None
            response = worker.stdout.readline(MAX_SUPERVISOR_MESSAGE_BYTES + 1)
        except (BrokenPipeError, OSError) as exc:
            raise CodingMCPWorkerUnavailable("worker transport failed") from exc
        if not response or len(response) > MAX_SUPERVISOR_MESSAGE_BYTES:
            raise CodingMCPWorkerUnavailable("worker response is missing or oversized")
        return response

    def _observe_job(self, raw: bytes) -> None:
        try:
            response = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError):
            return
        if not isinstance(response, Mapping):
            return
        result = response.get("result")
        if not isinstance(result, Mapping):
            return
        structured = result.get("structuredContent")
        if not isinstance(structured, Mapping):
            return
        job = structured.get("job")
        if not isinstance(job, Mapping):
            return
        job_id = job.get("job_id")
        state = job.get("state")
        if not isinstance(job_id, str) or not isinstance(state, str):
            return
        if state in TERMINAL_JOB_STATES:
            self._active_jobs.pop(job_id, None)
        else:
            self._active_jobs[job_id] = state

    @staticmethod
    def _is_submit(request: Mapping[str, Any]) -> bool:
        if request.get("method") != "tools/call":
            return False
        params = request.get("params")
        return isinstance(params, Mapping) and params.get("name") == "flyto_coding_submit"

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


def serve_supervised_stdio(argv: Sequence[str]) -> None:
    """Run the stable supervisor for the current CLI invocation."""

    CodingMCPWorkerSupervisor(worker_argv_from_process(argv)).serve()
