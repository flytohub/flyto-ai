# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Isolated subprocess and bounded JSON-RPC transport for MCP stdio."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from flyto_ai.coding.contracts import CapabilitySpec


MAX_MCP_MESSAGE_BYTES = 1024 * 1024
MAX_MCP_INFLIGHT_REQUESTS = 32


class CapabilityTimeout(RuntimeError):
    """One bounded capability request exceeded its configured deadline.

    The exception *type* is the classification. A caller must never parse the
    message text to learn that a call timed out: the host maps this exact type
    to one stable machine code, so route evidence can distinguish transport
    exhaustion from a real domain refusal.
    """


def capability_failure_code(exc: BaseException) -> str:
    """Classify one dispatch failure into a closed, stable machine code.

    An unrecognised failure returns an empty string on purpose. A code is
    evidence, and inventing one for an unclassified error would be a claim the
    transport cannot support.
    """
    if isinstance(exc, CapabilityTimeout):
        return "timeout"
    return ""


def encode_mcp_message(payload: Dict[str, Any]) -> bytes:
    """Encode one newline-delimited JSON-RPC message under the byte limit."""
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"),
    ).encode() + b"\n"
    if len(encoded) > MAX_MCP_MESSAGE_BYTES:
        raise RuntimeError("capability request exceeds the message limit")
    return encoded


def decode_mcp_message(raw: bytes) -> Dict[str, Any]:
    """Decode one bounded JSON object and reject all other wire shapes."""
    if len(raw) > MAX_MCP_MESSAGE_BYTES:
        raise RuntimeError("capability response exceeds the message limit")
    try:
        decoded = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("capability returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("capability response must be an object")
    return decoded


def isolated_capability_environment(
    spec: CapabilitySpec,
    runtime_home: str,
) -> Dict[str, str]:
    """Build the minimal child environment from an explicit passthrough list."""
    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "TERM", "TMPDIR")
        if key in os.environ
    }
    for name in spec.env_passthrough:
        if name in os.environ:
            env[name] = os.environ[name]
    env["HOME"] = runtime_home
    return env


async def _close_process_stdin(process: asyncio.subprocess.Process) -> None:
    """Close the writer while its event loop is alive and bound the drain."""
    writer = process.stdin
    if writer is None or writer.is_closing():
        return
    writer.close()
    try:
        await asyncio.wait_for(writer.wait_closed(), timeout=0.25)
    except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError):
        pass


async def _wait_for_process_exit(process: asyncio.subprocess.Process) -> None:
    """Prefer EOF shutdown, then terminate and kill within fixed bounds."""
    if process.returncode is not None:
        await process.wait()
        return
    try:
        await asyncio.wait_for(asyncio.shield(process.wait()), timeout=0.25)
        return
    except asyncio.TimeoutError:
        pass
    try:
        process.terminate()
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await asyncio.wait_for(asyncio.shield(process.wait()), timeout=2)
        return
    except asyncio.TimeoutError:
        pass
    try:
        process.kill()
    except ProcessLookupError:
        pass
    await process.wait()


class McpJsonRpcTransport:
    """One-shot MCP process with bounded concurrent request correlation."""

    def __init__(self, spec: CapabilitySpec, workspace: str) -> None:
        self.spec = spec
        self.workspace = str(Path(workspace).resolve())
        self.process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending: Dict[int, asyncio.Future[Dict[str, Any]]] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._runtime_home: Optional[tempfile.TemporaryDirectory] = None
        self._write_lock: Optional[asyncio.Lock] = None
        self._inflight: Optional[asyncio.Semaphore] = None
        self._started = False
        self._closing = False

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("capability transport can only be started once")
        self._started = True
        executable = shutil.which(self.spec.argv[0])
        if not executable:
            raise RuntimeError("capability executable is not installed")
        self._runtime_home = tempfile.TemporaryDirectory(prefix="flyto-capability-home-")
        env = isolated_capability_environment(self.spec, self._runtime_home.name)
        try:
            self.process = await asyncio.create_subprocess_exec(
                executable,
                *self.spec.argv[1:],
                cwd=self.workspace,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=MAX_MCP_MESSAGE_BYTES + 1,
            )
        except Exception:
            self._runtime_home.cleanup()
            self._runtime_home = None
            raise
        self._write_lock = asyncio.Lock()
        self._inflight = asyncio.Semaphore(MAX_MCP_INFLIGHT_REQUESTS)
        self._reader_task = asyncio.create_task(self._read_responses())
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send one bounded request and await only its correlated response."""
        self._validate_call(method, params)
        if not self.process or not self._inflight or not self._reader_task:
            raise RuntimeError("capability is not running")
        async with self._inflight:
            self._request_id += 1
            request_id = self._request_id
            future = asyncio.get_running_loop().create_future()
            self._pending[request_id] = future
            try:
                await self._send({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                })
                try:
                    return await asyncio.wait_for(
                        asyncio.shield(future), timeout=self.spec.timeout_seconds,
                    )
                except asyncio.TimeoutError as exc:
                    await self._notify_cancelled(request_id, "request timed out")
                    raise CapabilityTimeout("capability request timed out") from exc
                except asyncio.CancelledError:
                    await self._notify_cancelled(request_id, "caller cancelled")
                    raise
            finally:
                self._pending.pop(request_id, None)
                if not future.done():
                    future.cancel()

    async def notify(self, method: str, params: Dict[str, Any]) -> None:
        self._validate_call(method, params)
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def close(self) -> None:
        self._closing = True
        self._fail_pending(RuntimeError("capability transport closed"))
        process, self.process = self.process, None
        if process:
            await _close_process_stdin(process)
            await _wait_for_process_exit(process)
        if self._reader_task:
            if not self._reader_task.done():
                self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)
            self._reader_task = None
        if self._stderr_task:
            self._stderr_task.cancel()
            await asyncio.gather(self._stderr_task, return_exceptions=True)
            self._stderr_task = None
        if self._runtime_home:
            self._runtime_home.cleanup()
            self._runtime_home = None

    @staticmethod
    def _validate_call(method: str, params: Dict[str, Any]) -> None:
        if not isinstance(method, str) or not method or len(method) > 256:
            raise ValueError("capability method must be a bounded string")
        if not isinstance(params, dict):
            raise ValueError("capability params must be an object")

    async def _send(self, payload: Dict[str, Any]) -> None:
        if self._closing or not self.process or not self.process.stdin:
            raise RuntimeError("capability is not running")
        if not self._write_lock:
            raise RuntimeError("capability transport is not initialized")
        try:
            async with self._write_lock:
                self.process.stdin.write(encode_mcp_message(payload))
                await self.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise RuntimeError("capability closed stdin") from exc

    async def _read(self) -> Dict[str, Any]:
        if not self.process or not self.process.stdout:
            raise RuntimeError("capability is not running")
        try:
            raw = await self.process.stdout.readline()
        except ValueError as exc:
            raise RuntimeError("capability response exceeds the message limit") from exc
        if not raw:
            raise RuntimeError("capability closed stdout")
        return decode_mcp_message(raw)

    async def _read_responses(self) -> None:
        """Own stdout exclusively and route each response to one pending call."""
        try:
            while True:
                message = await self._read()
                if message.get("jsonrpc") != "2.0":
                    raise RuntimeError("capability returned an invalid JSON-RPC version")
                if "id" not in message:
                    continue
                response_id = message["id"]
                if isinstance(response_id, bool) or not isinstance(response_id, int):
                    raise RuntimeError("capability returned an invalid response id")
                future = self._pending.get(response_id)
                if future is None or future.done():
                    continue
                if "error" in message:
                    future.set_exception(RuntimeError(
                        "capability request failed: {}".format(
                            str(message["error"])[:1000],
                        ),
                    ))
                    continue
                result = message.get("result", {})
                if not isinstance(result, dict):
                    future.set_exception(
                        RuntimeError("capability result must be an object"),
                    )
                    continue
                future.set_result(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closing:
                self._fail_pending(exc)

    async def _notify_cancelled(self, request_id: int, reason: str) -> None:
        """Best-effort MCP cancellation; local pending state remains authoritative."""
        try:
            await self.notify(
                "notifications/cancelled",
                {"requestId": request_id, "reason": reason},
            )
        except (RuntimeError, ValueError):
            pass

    def _fail_pending(self, error: Exception) -> None:
        """Fail every active request exactly once after transport loss/closure."""
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(error)

    async def _drain_stderr(self) -> None:
        if not self.process or not self.process.stderr:
            return
        while True:
            chunk = await self.process.stderr.read(4096)
            if not chunk:
                return
