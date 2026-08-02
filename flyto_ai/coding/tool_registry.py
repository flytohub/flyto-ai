# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Transactional registry for provider-safe composed capability tools."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Protocol

from flyto_ai.coding.contracts import CapabilitySpec
from flyto_ai.permissions import PermissionLevel


class CapabilityToolSession(Protocol):
    """Minimal session surface required by the registry and manager."""

    tools: List[Dict[str, Any]]

    def remote_tool_name(self, provider_name: str) -> Optional[str]: ...

    async def dispatch(
        self, provider_name: str, arguments: Dict[str, Any],
    ) -> Dict[str, Any]: ...


def _freeze_definition(value: Any) -> Any:
    """Recursively freeze negotiated JSON metadata stored by the registry."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_definition(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_definition(item) for item in value)
    return deepcopy(value)


def _thaw_definition(value: Any) -> Any:
    """Return a detached JSON-shaped copy for provider-facing definitions."""
    if isinstance(value, Mapping):
        return {key: _thaw_definition(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_definition(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class RegisteredCapabilityTool:
    """Immutable routing and permission metadata for one negotiated tool."""

    provider_name: str
    remote_name: str
    required_level: PermissionLevel
    definition: Mapping[str, Any]
    session: CapabilityToolSession


class CapabilityToolRegistry:
    """Register complete sessions atomically and reject provider-name collisions."""

    def __init__(self) -> None:
        self._entries: Dict[str, RegisteredCapabilityTool] = {}

    @property
    def definitions(self) -> List[Dict[str, Any]]:
        return [_thaw_definition(entry.definition) for entry in self._entries.values()]

    @property
    def permission_overrides(self) -> Dict[str, PermissionLevel]:
        return {
            name: entry.required_level
            for name, entry in self._entries.items()
        }

    def register_session(
        self,
        session: CapabilityToolSession,
        spec: CapabilitySpec,
    ) -> None:
        """Validate every definition before committing any entry."""
        declared_permissions = dict(spec.tool_permissions)
        pending: Dict[str, RegisteredCapabilityTool] = {}
        for definition in session.tools:
            if not isinstance(definition, Mapping):
                raise RuntimeError("capability tool definition must be an object")
            provider_name = definition.get("name")
            if not isinstance(provider_name, str) or not provider_name:
                raise RuntimeError("capability tool definition has no provider name")
            remote_name = session.remote_tool_name(provider_name)
            if remote_name is None:
                raise RuntimeError("capability tool mapping is incomplete")
            if not isinstance(definition.get("inputSchema"), Mapping):
                raise RuntimeError("capability tool definition has no input schema")
            if provider_name in self._entries or provider_name in pending:
                raise RuntimeError(
                    "capability provider tool name collision: {}".format(provider_name),
                )
            declared = declared_permissions.get(remote_name, "workspace_write")
            try:
                required_level = PermissionLevel[declared.upper()]
            except (AttributeError, KeyError) as exc:
                raise RuntimeError("capability tool permission is invalid") from exc
            pending[provider_name] = RegisteredCapabilityTool(
                provider_name=provider_name,
                remote_name=remote_name,
                required_level=required_level,
                definition=_freeze_definition(definition),
                session=session,
            )
        self._entries.update(pending)

    def resolve(self, provider_name: str) -> RegisteredCapabilityTool | None:
        return self._entries.get(provider_name)

    def clear(self) -> None:
        self._entries.clear()
