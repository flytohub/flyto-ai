# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Compatibility facade and lifecycle manager for detachable capabilities."""
from __future__ import annotations

import asyncio
import shutil
from typing import Any, Dict, List, Mapping, Optional, Sequence

from flyto_ai.coding.contracts import CapabilitySpec, CapabilityStatus
from flyto_ai.coding.execution_policy import (
    ApprovalResolver,
    ExecutionPolicy,
    ExecutionPolicyController,
)
from flyto_ai.coding.execution_trace import (
    ExecutionReplayReport,
    ExecutionTraceLedger,
    OutcomeFeedbackReceipt,
    OutcomeFeedbackSink,
    TraceNormalizer,
)
from flyto_ai.coding.mcp_session import McpStdioSession
from flyto_ai.coding.mcp_transport import MAX_MCP_MESSAGE_BYTES
from flyto_ai.coding.permissions import (
    CapabilityPermissionGate,
    PermissionRiskResolver,
)
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
        *,
        execution_policy: Optional[ExecutionPolicy] = None,
        approval_resolver: Optional[ApprovalResolver] = None,
        trace_ledger: Optional[ExecutionTraceLedger] = None,
        risk_resolvers: Optional[Mapping[str, PermissionRiskResolver]] = None,
    ) -> None:
        self.workspace = workspace
        self._permission_gate = CapabilityPermissionGate(
            permission_level,
            risk_resolvers=risk_resolvers,
        )
        self.permission_level = self._permission_gate.runtime_level
        self._execution_policy = ExecutionPolicyController(
            workspace,
            execution_policy,
            approval_resolver=approval_resolver,
        )
        self._trace = trace_ledger or ExecutionTraceLedger()
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

    @property
    def execution_trace(self) -> Dict[str, Any]:
        """Return detached, redacted, hash-chained execution evidence."""
        return self._trace.export()

    @property
    def execution_trace_valid(self) -> bool:
        """Verify the complete execution evidence chain without mutating it."""
        return self._trace.verify_chain()

    async def execution_policy_snapshot(self) -> Dict[str, Any]:
        """Return the current bounded execution counters and limits."""
        return await self._execution_policy.snapshot()

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
                        sorted(set(spec.required_tools) - set(session.observed_tool_names))
                    )
                    self.statuses.append(CapabilityStatus(
                        name=spec.name,
                        available=False,
                        required=spec.required,
                        kind=spec.kind,
                        contract_version=spec.contract_version,
                        negotiated_protocol_version=session.negotiated_protocol_version,
                        server_name=session.server_name,
                        tool_count=len(session.observed_tool_names),
                        tools=session.observed_tool_names,
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
        return await self._dispatch(name, arguments, record_trace=True)

    async def _dispatch(
        self,
        name: str,
        arguments: Dict[str, Any],
        *,
        record_trace: bool,
    ) -> Dict[str, Any]:
        if not isinstance(name, str) or not name:
            return {"ok": False, "error": "unknown capability tool"}
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
        except Exception:
            result = {
                "ok": False,
                "error": "capability permission evaluation failed",
                "policy_outcome": "block",
                "policy_code": "permission_error",
                "capability_failed": True,
            }
            return self._with_trace(
                entry, arguments, False, result, "permission_error", record_trace,
            )
        if not evaluation.decision.allowed:
            result = evaluation.denial_payload()
            result["policy_code"] = "permission_denied"
            return self._with_trace(
                entry, arguments, False, result, "permission_denied", record_trace,
                required_level=evaluation.required_level,
            )

        admission = await self._execution_policy.admit(
            entry.provider_name,
            entry.remote_name,
            evaluation.required_level,
            arguments,
        )
        if not admission.allowed:
            result = admission.denial_payload()
            return self._with_trace(
                entry, arguments, False, result, admission.policy_code, record_trace,
                required_level=evaluation.required_level,
            )

        cancelled = False
        try:
            result = await entry.session.dispatch(name, arguments)
            if not isinstance(result, dict):
                result = {
                    "ok": False,
                    "error": "capability returned a non-object result",
                    "capability_failed": True,
                }
        except asyncio.CancelledError:
            cancelled = True
            result = {
                "ok": False,
                "error": "capability dispatch was cancelled",
                "capability_failed": True,
            }
        except Exception as exc:
            result = {
                "ok": False,
                "error": str(exc)[:1000],
                "capability_failed": True,
            }

        completion = await asyncio.shield(
            admission.finish(result.get("ok") is True, result),
        )
        policy_code = "allow"
        if not completion.allowed:
            policy_code = "result_budget"
            result = {
                "ok": False,
                "error": completion.error,
                "policy_outcome": "block",
                "policy_code": policy_code,
                "capability_failed": True,
            }
        traced = self._with_trace(
            entry,
            arguments,
            True,
            result,
            "cancelled" if cancelled else policy_code,
            record_trace,
            required_level=evaluation.required_level,
        )
        if cancelled:
            raise asyncio.CancelledError
        return traced

    async def record_policy_denial(
        self,
        name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        *,
        policy_code: str,
    ) -> Dict[str, Any]:
        """Record a denial produced by an outer Agent policy boundary."""
        entry = self._registry.resolve(name) if isinstance(name, str) else None
        if entry is None:
            return result
        try:
            required = self._permission_gate.required_level(
                entry.remote_name,
                entry.required_level,
                arguments,
            )
        except Exception:
            required = entry.required_level
        return self._with_trace(
            entry,
            arguments,
            False,
            result,
            policy_code,
            True,
            required_level=required,
        )

    async def replay_execution_trace(
        self,
        *,
        normalizers: Optional[Mapping[str, TraceNormalizer]] = None,
        allowed_permissions: Sequence[str] = ("read_only",),
    ) -> ExecutionReplayReport:
        """Replay a fixed trace snapshot under an explicit permission ceiling."""
        return await self._trace.replay(
            lambda name, arguments: self._dispatch(
                name, arguments, record_trace=False,
            ),
            normalizers=normalizers,
            allowed_permissions=allowed_permissions,
        )

    async def publish_blueprint_outcome(
        self,
        blueprint_id: str,
        replay: ExecutionReplayReport,
        sink: OutcomeFeedbackSink,
        *,
        timeout_seconds: int = 30,
    ) -> OutcomeFeedbackReceipt:
        """Publish one replay-bound outcome through a host-owned sink."""
        return await self._trace.publish_blueprint_outcome(
            blueprint_id, replay, sink, timeout_seconds=timeout_seconds,
        )

    def _with_trace(
        self,
        entry,
        arguments: Mapping[str, Any],
        dispatched: bool,
        result: Dict[str, Any],
        policy_code: str,
        record_trace: bool,
        *,
        required_level: Optional[PermissionLevel] = None,
    ) -> Dict[str, Any]:
        if not record_trace:
            return result
        trace_result = {
            key: value for key, value in result.items() if key != "call_id"
        }
        try:
            self._trace.record(
                provider_name=entry.provider_name,
                remote_name=entry.remote_name,
                required_permission=(
                    entry.required_level if required_level is None else required_level
                ).name.lower(),
                arguments=arguments,
                dispatched=dispatched,
                ok=result.get("ok") is True,
                policy_code=policy_code,
                result=trace_result,
            )
        except Exception:
            return {
                "ok": False,
                "error": "capability execution trace is unavailable",
                "policy_outcome": "block",
                "policy_code": "trace_unavailable",
                "capability_failed": True,
            }
        return result

    async def close(self) -> None:
        await asyncio.gather(
            *(session.close() for session in self.sessions),
            return_exceptions=True,
        )
        self.sessions.clear()
        self._registry.clear()
