# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Atomic runtime permission evaluation for composed capability tools."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping

from flyto_ai.permissions import (
    PermissionDecision,
    PermissionEnforcer,
    PermissionLevel,
)


PermissionRiskResolver = Callable[[Mapping[str, Any]], PermissionLevel]


def coerce_permission_level(value: PermissionLevel | str) -> PermissionLevel:
    """Normalize one host-selected ceiling without accepting unknown tiers."""
    if isinstance(value, PermissionLevel):
        return value
    if not isinstance(value, str):
        raise ValueError("unknown capability permission level")
    try:
        return PermissionLevel[value.upper()]
    except KeyError as exc:
        raise ValueError("unknown capability permission level") from exc


def _execute_module_risk(arguments: Mapping[str, Any]) -> PermissionLevel:
    """Preserve Core's argument-sensitive module classification."""
    return PermissionEnforcer().required_level("execute_module", dict(arguments))


DEFAULT_PERMISSION_RISK_RESOLVERS: Mapping[str, PermissionRiskResolver] = MappingProxyType({
    "execute_module": _execute_module_risk,
})


@dataclass(frozen=True)
class CapabilityPermissionEvaluation:
    """Complete evidence for one composed tool permission decision."""

    decision: PermissionDecision
    required_level: PermissionLevel
    runtime_level: PermissionLevel

    def denial_payload(self) -> Dict[str, Any]:
        """Return the stable fail-closed response used by runtime dispatch."""
        return {
            "ok": False,
            "error": self.decision.reason,
            "policy_outcome": self.decision.outcome.value,
            "required_permission": self.required_level.name.lower(),
            "runtime_permission": self.runtime_level.name.lower(),
        }


class CapabilityPermissionGate:
    """Evaluate declared and argument-sensitive risk under one host ceiling."""

    def __init__(
        self,
        runtime_level: PermissionLevel | str = PermissionLevel.WORKSPACE_WRITE,
        *,
        risk_resolvers: Mapping[str, PermissionRiskResolver] | None = None,
    ) -> None:
        self._runtime_level = coerce_permission_level(runtime_level)
        self._risk_resolvers = dict(
            DEFAULT_PERMISSION_RISK_RESOLVERS
            if risk_resolvers is None
            else risk_resolvers
        )
        if any(
            not isinstance(name, str) or not name or not callable(resolver)
            for name, resolver in self._risk_resolvers.items()
        ):
            raise ValueError("capability risk resolvers must map tool names to callables")

    @property
    def runtime_level(self) -> PermissionLevel:
        return self._runtime_level

    def required_level(
        self,
        remote_name: str,
        declared_level: PermissionLevel,
        arguments: Mapping[str, Any],
    ) -> PermissionLevel:
        """Resolve call risk monotonically: dynamic policy may only escalate."""
        resolver = self._risk_resolvers.get(remote_name)
        if resolver is None:
            return declared_level
        dynamic_level = resolver(arguments)
        if not isinstance(dynamic_level, PermissionLevel):
            raise ValueError("capability risk resolver returned an invalid permission level")
        return max(declared_level, dynamic_level)

    def evaluate(
        self,
        provider_name: str,
        remote_name: str,
        declared_level: PermissionLevel,
        arguments: Mapping[str, Any],
    ) -> CapabilityPermissionEvaluation:
        """Evaluate one exact call and retain both required and runtime tiers."""
        required = self.required_level(remote_name, declared_level, arguments)
        decision = PermissionEnforcer(
            self._runtime_level,
            overrides={provider_name: required},
        ).check(provider_name, dict(arguments))
        return CapabilityPermissionEvaluation(
            decision=decision,
            required_level=required,
            runtime_level=self._runtime_level,
        )
