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
    """One-shot isolated MCP process with correlated request/response framing."""

    def __init__(self, spec: CapabilitySpec, workspace: str) -> None:
        self.spec = spec
        self.workspace = str(Path(workspace).resolve())
        self.process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._stderr_task: Optional[asyncio.Task] = None
        self._runtime_home: Optional[tempfile.TemporaryDirectory] = None
        self._started = False

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
            )
        except Exception:
            self._runtime_home.cleanup()
            self._runtime_home = None
            raise
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send one request and ignore unrelated notifications/responses."""
        self._request_id += 1
        request_id = self._request_id
        await self._send({
            "jsonrpc": "2.0", "id": request_id, "method": method, "params": params,
        })
        deadline = asyncio.get_running_loop().time() + self.spec.timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise RuntimeError("capability request timed out")
            message = await asyncio.wait_for(self._read(), timeout=remaining)
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(
                    "capability request failed: {}".format(str(message["error"])[:1000]),
                )
            result = message.get("result", {})
            if not isinstance(result, dict):
                raise RuntimeError("capability result must be an object")
            return result

    async def notify(self, method: str, params: Dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def close(self) -> None:
        process, self.process = self.process, None
        if process:
            await _close_process_stdin(process)
            await _wait_for_process_exit(process)
        if self._stderr_task:
            self._stderr_task.cancel()
            await asyncio.gather(self._stderr_task, return_exceptions=True)
            self._stderr_task = None
        if self._runtime_home:
            self._runtime_home.cleanup()
            self._runtime_home = None

    async def _send(self, payload: Dict[str, Any]) -> None:
        if not self.process or not self.process.stdin:
            raise RuntimeError("capability is not running")
        self.process.stdin.write(encode_mcp_message(payload))
        await self.process.stdin.drain()

    async def _read(self) -> Dict[str, Any]:
        if not self.process or not self.process.stdout:
            raise RuntimeError("capability is not running")
        raw = await self.process.stdout.readline()
        if not raw:
            raise RuntimeError("capability closed stdout")
        return decode_mcp_message(raw)

    async def _drain_stderr(self) -> None:
        if not self.process or not self.process.stderr:
            return
        total = 0
        while total < MAX_MCP_MESSAGE_BYTES:
            chunk = await self.process.stderr.read(4096)
            if not chunk:
                return
            total += len(chunk)
