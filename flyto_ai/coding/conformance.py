# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Reusable conformance contract for any detachable capability adapter."""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from flyto_ai.coding.capabilities import CapabilityManager
from flyto_ai.coding.contracts import CapabilitySpec
from flyto_ai.coding.mcp_catalog import provider_tool_name
from flyto_ai.coding.store import redact_evidence
from flyto_ai.permissions import PermissionLevel


ADAPTER_CONFORMANCE_VERSION = "flyto.adapter-conformance.v1"
ResultVerifier = Callable[[Mapping[str, Any]], bool | Awaitable[bool]]


class ManagerFactory(Protocol):
    def __call__(
        self, workspace: str, permission_level: PermissionLevel | str,
    ) -> CapabilityManager: ...


@dataclass(frozen=True)
class AdapterConformanceCase:
    """One named remote-tool probe with an optional domain-owned verifier."""

    name: str
    remote_tool: str
    arguments: Mapping[str, Any]
    expected_ok: bool = True
    expected_dispatched: bool = True
    verifier: Optional[ResultVerifier] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or len(self.name) > 128:
            raise ValueError("conformance case name must be a bounded string")
        if not isinstance(self.remote_tool, str) or not self.remote_tool:
            raise ValueError("conformance remote_tool must be a non-empty string")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("conformance arguments must be an object")
        if not isinstance(self.expected_ok, bool):
            raise ValueError("conformance expected_ok must be a boolean")
        if not isinstance(self.expected_dispatched, bool):
            raise ValueError("conformance expected_dispatched must be a boolean")
        if self.verifier is not None and not callable(self.verifier):
            raise ValueError("conformance verifier must be callable")
        try:
            canonical_arguments = json.loads(json.dumps(
                dict(self.arguments),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ))
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("conformance arguments must be finite JSON") from exc
        object.__setattr__(
            self,
            "arguments",
            _freeze_json(canonical_arguments),
        )


@dataclass(frozen=True)
class AdapterConformanceCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class AdapterConformanceReport:
    capability: str
    checks: Tuple[AdapterConformanceCheck, ...]
    fingerprint: str
    trace_fingerprint: str = ""
    trace_event_count: int = 0
    policy_calls: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": ADAPTER_CONFORMANCE_VERSION,
            "capability": self.capability,
            "ok": self.ok,
            "fingerprint": self.fingerprint,
            "trace_fingerprint": self.trace_fingerprint,
            "trace_event_count": self.trace_event_count,
            "policy_calls": self.policy_calls,
            "checks": [dataclasses.asdict(check) for check in self.checks],
        }


async def run_adapter_conformance(
    workspace: str,
    spec: CapabilitySpec,
    cases: Sequence[AdapterConformanceCase],
    *,
    permission_level: PermissionLevel | str = PermissionLevel.READ_ONLY,
    manager_factory: ManagerFactory = CapabilityManager,
) -> AdapterConformanceReport:
    """Negotiate, scope, execute, verify, and close one adapter end to end."""
    if not isinstance(spec, CapabilitySpec):
        raise TypeError("adapter conformance requires a CapabilitySpec")
    if spec.kind != "mcp-stdio":
        raise ValueError("adapter conformance currently requires an MCP stdio adapter")
    try:
        cases = tuple(cases)
    except TypeError as exc:
        raise TypeError("adapter conformance cases have an invalid type") from exc
    if not cases or len(cases) > 256:
        raise ValueError("adapter conformance requires between 1 and 256 cases")
    if any(not isinstance(case, AdapterConformanceCase) for case in cases):
        raise TypeError("adapter conformance cases have an invalid type")
    case_names = tuple(case.name for case in cases)
    if len(set(case_names)) != len(case_names):
        raise ValueError("adapter conformance case names must be unique")
    if not callable(manager_factory):
        raise ValueError("manager_factory must be callable")

    checks = []
    allowed = set(spec.allowed_tools)
    classified = {name for name, _level in spec.tool_permissions}
    checks.append(AdapterConformanceCheck(
        "allowed_tools_declared",
        bool(allowed),
        "{} allowed tools".format(len(allowed)),
    ))
    checks.append(AdapterConformanceCheck(
        "permissions_exhaustive",
        classified == allowed,
        "classified={} allowed={}".format(len(classified), len(allowed)),
    ))
    checks.append(AdapterConformanceCheck(
        "cases_scoped",
        all(case.remote_tool in allowed for case in cases),
        "every case must target allowed_tools",
    ))
    checks.append(AdapterConformanceCheck(
        "tools_covered",
        {case.remote_tool for case in cases} == allowed,
        "every allowed tool must have a conformance case",
    ))

    manager = manager_factory(workspace, permission_level)
    trace_fingerprint = ""
    trace_event_count = 0
    policy_calls = 0
    try:
        status_spec = dataclasses.replace(spec, required=True)
        statuses = await manager.start((status_spec,))
        status = statuses[0] if statuses else None
        checks.append(AdapterConformanceCheck(
            "negotiated",
            bool(status and status.available),
            "" if status is None else str(status.error or ""),
        ))
        checks.append(AdapterConformanceCheck(
            "protocol_exact",
            bool(status and status.negotiated_protocol_version == spec.protocol_version),
            "" if status is None else status.negotiated_protocol_version,
        ))
        checks.append(AdapterConformanceCheck(
            "catalog_exact",
            bool(status and set(status.tools) == allowed),
            "" if status is None else ",".join(status.tools),
        ))
        provider_names = {definition.get("name") for definition in manager.tools}
        checks.append(AdapterConformanceCheck(
            "provider_names_exact",
            provider_names == {
                provider_tool_name(spec.name, remote_name) for remote_name in allowed
            },
            "{} provider tools".format(len(provider_names)),
        ))

        for case in cases:
            provider_name = provider_tool_name(spec.name, case.remote_tool)
            event_count = manager.execution_trace["event_count"]
            result = await manager.dispatch(
                provider_name, _thaw_json(case.arguments),
            )
            events = manager.execution_trace["events"]
            event = events[-1] if len(events) == event_count + 1 else None
            observed_dispatched = None if event is None else event["dispatched"]
            passed = (
                result.get("ok") is case.expected_ok
                and observed_dispatched is case.expected_dispatched
            )
            detail = "expected ok={}/dispatched={} got ok={}/dispatched={}".format(
                case.expected_ok,
                case.expected_dispatched,
                result.get("ok"),
                observed_dispatched,
            )
            if passed and case.verifier is not None:
                verified = case.verifier(result)
                if inspect.isawaitable(verified):
                    verified = await verified
                passed = verified is True
                detail = "domain verifier {}".format("passed" if passed else "failed")
            checks.append(AdapterConformanceCheck(
                "case:{}".format(case.name), passed, detail,
            ))
        trace = manager.execution_trace
        snapshot = await manager.execution_policy_snapshot()
        trace_fingerprint = str(trace["fingerprint"])
        trace_event_count = int(trace["event_count"])
        policy_calls = int(snapshot["calls"])
        checks.append(AdapterConformanceCheck(
            "evidence_chain",
            manager.execution_trace_valid and trace_event_count == len(cases),
            "{} trace events".format(trace_event_count),
        ))
        checks.append(AdapterConformanceCheck(
            "policy_leases_released",
            snapshot["active"] == 0,
            "{} active calls".format(snapshot["active"]),
        ))
    except Exception as exc:
        checks.append(AdapterConformanceCheck(
            "runtime_exception", False, type(exc).__name__,
        ))
    finally:
        await manager.close()
        await manager.close()
        checks.append(AdapterConformanceCheck(
            "closed", not manager.tools and not manager.sessions,
            "manager registry and sessions must be empty",
        ))

    checks = [
        AdapterConformanceCheck(
            check.name,
            check.passed,
            str(redact_evidence(check.detail)),
        )
        for check in checks
    ]
    safe_checks = [redact_evidence(dataclasses.asdict(check)) for check in checks]
    safe_cases = [
        {
            "name": case.name,
            "remote_tool": case.remote_tool,
            "arguments": redact_evidence(_thaw_json(case.arguments)),
            "expected_ok": case.expected_ok,
            "expected_dispatched": case.expected_dispatched,
            "has_verifier": case.verifier is not None,
        }
        for case in cases
    ]
    fingerprint = hashlib.sha256(json.dumps(
        {
            "contract_version": ADAPTER_CONFORMANCE_VERSION,
            "capability": spec.name,
            "contract_version_under_test": spec.contract_version,
            "protocol_version": spec.protocol_version,
            "allowed_tools": sorted(spec.allowed_tools),
            "tool_permissions": sorted(spec.tool_permissions),
            "cases": safe_cases,
            "checks": safe_checks,
            "trace_fingerprint": trace_fingerprint,
            "trace_event_count": trace_event_count,
            "policy_calls": policy_calls,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return AdapterConformanceReport(
        capability=spec.name,
        checks=tuple(checks),
        fingerprint=fingerprint,
        trace_fingerprint=trace_fingerprint,
        trace_event_count=trace_event_count,
        policy_calls=policy_calls,
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({
            key: _freeze_json(item) for key, item in value.items()
        })
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "ADAPTER_CONFORMANCE_VERSION",
    "AdapterConformanceCase",
    "AdapterConformanceCheck",
    "AdapterConformanceReport",
    "ManagerFactory",
    "ResultVerifier",
    "run_adapter_conformance",
]
