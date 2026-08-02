# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Compatibility facade and lifecycle manager for detachable capabilities."""
from __future__ import annotations

import asyncio
import shutil
from typing import Any, Dict, List, Sequence

from flyto_ai.coding.contracts import CapabilitySpec, CapabilityStatus
from flyto_ai.coding.mcp_session import McpStdioSession
from flyto_ai.coding.mcp_transport import MAX_MCP_MESSAGE_BYTES
from flyto_ai.coding.permissions import CapabilityPermissionGate
from flyto_ai.coding.tool_registry import CapabilityToolRegistry
from flyto_ai.permissions import PermissionLevel


__all__ = [
    "CapabilityManager",
    "MAX_MCP_MESSAGE_BYTES",
    "McpStdioSession",
]


class CapabilityManager:
    """Coordinate adapter lifecycle, registry, permission gate, and dispatch."""

    def __init__(
        self,
        workspace: str,
        permission_level: PermissionLevel | str = PermissionLevel.WORKSPACE_WRITE,
    ) -> None:
        self.workspace = workspace
        self._permission_gate = CapabilityPermissionGate(permission_level)
        self.permission_level = self._permission_gate.runtime_level
        self._registry = CapabilityToolRegistry()
        self.sessions: List[McpStdioSession] = []
        self.statuses: List[CapabilityStatus] = []
        self._started = False

    @property
    def definitions(self) -> List[Dict[str, Any]]:
        return self._registry.definitions

    @property
    def tools(self) -> List[Dict[str, Any]]:
        """Expose attached definitions through the generic ToolExecutor contract."""
        return self.definitions

    @property
    def permission_overrides(self) -> Dict[str, PermissionLevel]:
        """Return provider-name permission metadata for the outer Agent gate."""
        return self._registry.permission_overrides

    @property
    def required_available(self) -> bool:
        return all(status.available for status in self.statuses if status.required)

    async def start(self, specs: Sequence[CapabilitySpec]) -> List[CapabilityStatus]:
        if self._started:
            raise RuntimeError("capability manager can only be started once")
        self._started = True
        try:
            for spec in specs:
                if spec.kind == "command":
                    available = shutil.which(spec.argv[0]) is not None
                    self.statuses.append(CapabilityStatus(
                        name=spec.name,
                        available=available,
                        required=spec.required,
                        kind=spec.kind,
                        contract_version=spec.contract_version,
                        error=None if available else "capability executable is not installed",
                    ))
                    continue
                session = McpStdioSession(spec, self.workspace)
                try:
                    await session.start()
                except Exception as exc:
                    missing = tuple(
                        sorted(set(spec.required_tools) - set(session.remote_tool_names))
                    )
                    self.statuses.append(CapabilityStatus(
                        name=spec.name,
                        available=False,
                        required=spec.required,
                        kind=spec.kind,
                        contract_version=spec.contract_version,
                        negotiated_protocol_version=session.negotiated_protocol_version,
                        server_name=session.server_name,
                        tool_count=len(session.remote_tool_names),
                        tools=tuple(session.remote_tool_names),
                        missing_tools=missing,
                        error=str(exc)[:1000],
                    ))
                    continue
                try:
                    self._registry.register_session(session, spec)
                except Exception:
                    await session.close()
                    raise
                self.sessions.append(session)
                self.statuses.append(CapabilityStatus(
                    name=spec.name,
                    available=True,
                    required=spec.required,
                    kind=spec.kind,
                    contract_version=spec.contract_version,
                    negotiated_protocol_version=session.negotiated_protocol_version,
                    server_name=session.server_name,
                    tool_count=len(session.tools),
                    tools=tuple(session.remote_tool_names),
                ))
        except Exception:
            await self.close()
            raise
        return list(self.statuses)

    async def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        entry = self._registry.resolve(name)
        if entry is None:
            return {"ok": False, "error": "unknown capability tool"}
        try:
            evaluation = self._permission_gate.evaluate(
                entry.provider_name,
                entry.remote_name,
                entry.required_level,
                arguments,
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc)[:1000],
                "policy_outcome": "block",
                "capability_failed": True,
            }
        if not evaluation.decision.allowed:
            return evaluation.denial_payload()
        try:
            return await entry.session.dispatch(name, arguments)
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:1000], "capability_failed": True}

    async def close(self) -> None:
        await asyncio.gather(
            *(session.close() for session in self.sessions),
            return_exceptions=True,
        )
        self.sessions.clear()
        self._registry.clear()
