# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Detachable command and MCP-stdio capability adapters."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from flyto_ai.coding.contracts import CapabilitySpec, CapabilityStatus
from flyto_ai.coding.store import redact_evidence


MAX_MCP_MESSAGE_BYTES = 1024 * 1024
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def _provider_tool_name(capability: str, tool_name: str) -> str:
    base = "cap_{}_{}".format(
        _SAFE_NAME.sub("_", capability), _SAFE_NAME.sub("_", tool_name),
    ).strip("_")
    if len(base) <= 64:
        return base
    digest = hashlib.sha256(base.encode()).hexdigest()[:10]
    return "{}_{}".format(base[:53], digest)


class McpStdioSession:
    """Minimal bounded MCP client for capability discovery and tool calls."""

    def __init__(self, spec: CapabilitySpec, workspace: str) -> None:
        self.spec = spec
        self.workspace = str(Path(workspace).resolve())
        self.process: Optional[asyncio.subprocess.Process] = None
        self.tools: List[Dict[str, Any]] = []
        self.remote_tool_names: List[str] = []
        self.negotiated_protocol_version = ""
        self.server_name = ""
        self._tool_map: Dict[str, str] = {}
        self._request_id = 0
        self._stderr_task: Optional[asyncio.Task] = None
        self._runtime_home: Optional[tempfile.TemporaryDirectory] = None

    async def start(self) -> None:
        executable = shutil.which(self.spec.argv[0])
        if not executable:
            raise RuntimeError("capability executable is not installed")
        env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TERM", "TMPDIR") if key in os.environ}
        for name in self.spec.env_passthrough:
            if name in os.environ:
                env[name] = os.environ[name]
        self._runtime_home = tempfile.TemporaryDirectory(prefix="flyto-capability-home-")
        env["HOME"] = self._runtime_home.name
        self.process = await asyncio.create_subprocess_exec(
            executable, *self.spec.argv[1:], cwd=self.workspace, env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        try:
            initialized = await self._request("initialize", {
                "protocolVersion": self.spec.protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "flyto-ai", "version": "1"},
            })
            negotiated = initialized.get("protocolVersion")
            if negotiated != self.spec.protocol_version:
                raise RuntimeError(
                    "capability negotiated unsupported MCP protocol version: {}".format(
                        str(negotiated)[:100],
                    )
                )
            self.negotiated_protocol_version = str(negotiated)
            server_info = initialized.get("serverInfo", {})
            if isinstance(server_info, dict) and isinstance(server_info.get("name"), str):
                self.server_name = server_info["name"][:128]
            await self._notify("notifications/initialized", {})
            listed = await self._request("tools/list", {})
        except Exception:
            await self.close()
            raise
        raw_tools = listed.get("tools", []) if isinstance(listed, dict) else []
        if not isinstance(raw_tools, list) or len(raw_tools) > 2000:
            await self.close()
            raise RuntimeError("capability returned an invalid tool catalog")
        definitions: List[Dict[str, Any]] = []
        remote_tool_names: List[str] = []
        for item in raw_tools:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            provider_name = _provider_tool_name(self.spec.name, item["name"])
            schema = item.get("inputSchema") if isinstance(item.get("inputSchema"), dict) else {
                "type": "object", "properties": {},
            }
            definitions.append({
                "name": provider_name,
                "description": "[{}] {}".format(self.spec.name, str(item.get("description", ""))[:2000]),
                "inputSchema": schema,
            })
            self._tool_map[provider_name] = item["name"]
            remote_tool_names.append(item["name"])
        missing = sorted(set(self.spec.required_tools) - set(remote_tool_names))
        self.remote_tool_names = sorted(remote_tool_names)
        if missing:
            await self.close()
            raise RuntimeError("capability is missing required tools: {}".format(", ".join(missing)))
        self.tools = definitions

    async def dispatch(self, provider_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        remote_name = self._tool_map.get(provider_name)
        if not remote_name:
            return {"ok": False, "error": "unknown capability tool"}
        result = await self._request("tools/call", {"name": remote_name, "arguments": arguments})
        return {"ok": not bool(result.get("isError")) if isinstance(result, dict) else False,
                "capability": self.spec.name, "tool": remote_name,
                "result": redact_evidence(result)}

    async def close(self) -> None:
        process, self.process = self.process, None
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if self._stderr_task:
            self._stderr_task.cancel()
            await asyncio.gather(self._stderr_task, return_exceptions=True)
            self._stderr_task = None
        if self._runtime_home:
            self._runtime_home.cleanup()
            self._runtime_home = None

    async def _request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._request_id += 1
        request_id = self._request_id
        await self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        deadline = asyncio.get_running_loop().time() + self.spec.timeout_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise RuntimeError("capability request timed out")
            message = await asyncio.wait_for(self._read(), timeout=remaining)
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError("capability request failed: {}".format(str(message["error"])[:1000]))
            result = message.get("result", {})
            if not isinstance(result, dict):
                raise RuntimeError("capability result must be an object")
            return result

    async def _notify(self, method: str, params: Dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _send(self, payload: Dict[str, Any]) -> None:
        if not self.process or not self.process.stdin:
            raise RuntimeError("capability is not running")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        if len(encoded) > MAX_MCP_MESSAGE_BYTES:
            raise RuntimeError("capability request exceeds the message limit")
        self.process.stdin.write(encoded)
        await self.process.stdin.drain()

    async def _read(self) -> Dict[str, Any]:
        if not self.process or not self.process.stdout:
            raise RuntimeError("capability is not running")
        raw = await self.process.stdout.readline()
        if not raw:
            raise RuntimeError("capability closed stdout")
        if len(raw) > MAX_MCP_MESSAGE_BYTES:
            raise RuntimeError("capability response exceeds the message limit")
        try:
            decoded = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("capability returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("capability response must be an object")
        return decoded

    async def _drain_stderr(self) -> None:
        if not self.process or not self.process.stderr:
            return
        total = 0
        while total < MAX_MCP_MESSAGE_BYTES:
            chunk = await self.process.stderr.read(4096)
            if not chunk:
                return
            total += len(chunk)


class CapabilityManager:
    """Start only configured adapters and fail closed for required failures."""

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self.sessions: List[McpStdioSession] = []
        self.statuses: List[CapabilityStatus] = []
        self._dispatch: Dict[str, McpStdioSession] = {}

    @property
    def definitions(self) -> List[Dict[str, Any]]:
        return [definition for session in self.sessions for definition in session.tools]

    @property
    def required_available(self) -> bool:
        return all(status.available for status in self.statuses if status.required)

    async def start(self, specs: Sequence[CapabilitySpec]) -> List[CapabilityStatus]:
        for spec in specs:
            if spec.kind == "command":
                available = shutil.which(spec.argv[0]) is not None
                self.statuses.append(CapabilityStatus(
                    name=spec.name, available=available, required=spec.required,
                    kind=spec.kind, contract_version=spec.contract_version,
                    error=None if available else "capability executable is not installed",
                ))
                continue
            session = McpStdioSession(spec, self.workspace)
            try:
                await session.start()
            except Exception as exc:
                missing = tuple(sorted(set(spec.required_tools) - set(session.remote_tool_names)))
                self.statuses.append(CapabilityStatus(
                    name=spec.name, available=False, required=spec.required,
                    kind=spec.kind, contract_version=spec.contract_version,
                    negotiated_protocol_version=session.negotiated_protocol_version,
                    server_name=session.server_name,
                    tool_count=len(session.remote_tool_names),
                    tools=tuple(session.remote_tool_names),
                    missing_tools=missing,
                    error=str(exc)[:1000],
                ))
                continue
            self.sessions.append(session)
            for definition in session.tools:
                self._dispatch[definition["name"]] = session
            self.statuses.append(CapabilityStatus(
                name=spec.name, available=True, required=spec.required,
                kind=spec.kind, contract_version=spec.contract_version,
                negotiated_protocol_version=session.negotiated_protocol_version,
                server_name=session.server_name,
                tool_count=len(session.tools),
                tools=tuple(session.remote_tool_names),
            ))
        return list(self.statuses)

    async def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        session = self._dispatch.get(name)
        if not session:
            return {"ok": False, "error": "unknown capability tool"}
        try:
            return await session.dispatch(name, arguments)
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:1000], "capability_failed": True}

    async def close(self) -> None:
        await asyncio.gather(*(session.close() for session in self.sessions), return_exceptions=True)
        self.sessions.clear()
        self._dispatch.clear()
