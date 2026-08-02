# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Bounded manifest parsing and normalized identity for agent-stack profiles."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import yaml

from flyto_ai.coding.contracts import CapabilitySpec


AGENT_STACK_CONTRACT_VERSION = "flyto.agent-stack.v1"
AGENT_STACK_POLICY_VERSION = "flyto.agent-stack.v2"
SUPPORTED_AGENT_STACK_MANIFEST_VERSIONS = frozenset({
    AGENT_STACK_CONTRACT_VERSION,
    AGENT_STACK_POLICY_VERSION,
})
MAX_AGENT_STACK_MANIFEST_BYTES = 256 * 1024
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MANIFEST_KEYS = frozenset({"version", "profile", "capabilities"})
_CAPABILITY_KEYS = frozenset({field.name for field in dataclasses.fields(CapabilitySpec)})


@dataclass(frozen=True)
class AgentStackManifest:
    """Validated, source-controlled composition for one agent domain profile."""

    profile: str
    capabilities: Tuple[CapabilitySpec, ...]
    manifest_fingerprint: str
    contract_version: str = AGENT_STACK_CONTRACT_VERSION


def validate_profile_name(profile: str) -> str:
    """Validate one bounded profile identity and return it unchanged."""
    if not isinstance(profile, str) or not _PROFILE_NAME.fullmatch(profile):
        raise ValueError("agent stack profile must be a safe identifier")
    return profile


def validate_contract_version(contract_version: str) -> str:
    """Accept only implemented agent-stack contract versions."""
    if contract_version not in SUPPORTED_AGENT_STACK_MANIFEST_VERSIONS:
        raise ValueError("unsupported agent stack manifest version")
    return contract_version


def compose_capability_stack(*groups: Sequence[CapabilitySpec]) -> Tuple[CapabilitySpec, ...]:
    """Compose arbitrary capability groups without encoding their task domain."""
    capabilities = tuple(spec for group in groups for spec in group)
    if not capabilities:
        raise ValueError("agent stack must contain at least one capability")
    if len(capabilities) > 64:
        raise ValueError("agent stack cannot exceed 64 capabilities")
    if any(not isinstance(spec, CapabilitySpec) for spec in capabilities):
        raise TypeError("agent stack accepts only CapabilitySpec values")
    names = tuple(spec.name for spec in capabilities)
    if len(set(names)) != len(names):
        raise ValueError("agent stack capabilities contain duplicate names")
    return capabilities


def canonical_manifest_fingerprint(
    contract_version: str,
    profile: str,
    capabilities: Sequence[CapabilitySpec],
) -> str:
    """Hash the normalized policy-bearing profile, never raw YAML formatting."""
    normalized = {
        "version": contract_version,
        "profile": profile,
        "capabilities": [dataclasses.asdict(spec) for spec in capabilities],
    }
    return hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


def _validate_manifest_capability_mapping(index: int, item: Mapping) -> None:
    string_fields = ("name", "kind", "contract_version", "protocol_version")
    list_fields = ("argv", "required_tools", "allowed_tools", "env_passthrough")
    for field in string_fields:
        if field in item and not isinstance(item[field], str):
            raise ValueError(
                "capability {} field {} must be a string".format(index, field),
            )
    if "required" in item and not isinstance(item["required"], bool):
        raise ValueError("capability {} field required must be a boolean".format(index))
    if "timeout_seconds" in item and (
        isinstance(item["timeout_seconds"], bool)
        or not isinstance(item["timeout_seconds"], int)
    ):
        raise ValueError("capability {} field timeout_seconds must be an integer".format(index))
    for field in list_fields:
        if field not in item:
            continue
        value = item[field]
        if not isinstance(value, list) or any(not isinstance(member, str) for member in value):
            raise ValueError(
                "capability {} field {} must be an array of strings".format(index, field),
            )
    tool_permissions = item.get("tool_permissions")
    if tool_permissions is not None and (
        not isinstance(tool_permissions, Mapping)
        or any(
            not isinstance(name, str) or not isinstance(level, str)
            for name, level in tool_permissions.items()
        )
    ):
        raise ValueError(
            "capability {} field tool_permissions must be an object of strings".format(index),
        )


def _resolve_manifest_file(workspace: str, manifest_path: str) -> Path:
    """Resolve one regular manifest file without allowing workspace escape."""
    root = Path(workspace).expanduser().resolve(strict=True)
    requested = Path(manifest_path)
    if requested.is_absolute():
        raise ValueError("agent stack manifest path must be relative to the workspace")
    try:
        path = (root / requested).resolve(strict=True)
    except OSError as exc:
        raise ValueError("agent stack manifest must be a regular file") from exc
    if root != path and root not in path.parents:
        raise ValueError("agent stack manifest must remain inside the workspace")
    if not path.is_file():
        raise ValueError("agent stack manifest must be a regular file")
    return path


def _decode_manifest_file(path: Path) -> Mapping:
    """Read and decode one bounded UTF-8 YAML object."""
    raw = path.read_bytes()
    if len(raw) > MAX_AGENT_STACK_MANIFEST_BYTES:
        raise ValueError("agent stack manifest exceeds the 256 KiB limit")
    try:
        value = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("agent stack manifest is not valid UTF-8 YAML") from exc
    if not isinstance(value, Mapping):
        raise ValueError("agent stack manifest must be a YAML object")
    unknown = set(value) - _MANIFEST_KEYS
    if unknown:
        raise ValueError("unknown agent stack manifest fields: {}".format(
            ", ".join(sorted(unknown)),
        ))
    return value


def _manifest_header(value: Mapping) -> Tuple[str, str]:
    """Validate and return the versioned profile identity."""
    contract_version = validate_contract_version(value.get("version"))
    profile = validate_profile_name(value.get("profile"))
    return contract_version, profile


def _validate_v2_tool_policy(spec: CapabilitySpec) -> None:
    """Require an exhaustive tool classification for a v2 MCP capability."""
    classified = {name for name, _level in spec.tool_permissions}
    allowed = set(spec.allowed_tools)
    if classified == allowed:
        return
    missing = sorted(allowed - classified)
    extra = sorted(classified - allowed)
    details = []
    if missing:
        details.append("missing {}".format(", ".join(missing)))
    if extra:
        details.append("extra {}".format(", ".join(extra)))
    raise ValueError(
        "v2 MCP capability {} must classify every allowed tool ({})".format(
            spec.name, "; ".join(details),
        )
    )


def _parse_manifest_capability(
    index: int,
    item: object,
    contract_version: str,
) -> CapabilitySpec:
    """Parse one capability independently from manifest I/O and composition."""
    if not isinstance(item, Mapping):
        raise ValueError("agent stack capability {} must be an object".format(index))
    unknown_capability_fields = set(item) - _CAPABILITY_KEYS
    if unknown_capability_fields:
        raise ValueError(
            "unknown capability fields at index {}: {}".format(
                index, ", ".join(sorted(unknown_capability_fields)),
            )
        )
    _validate_manifest_capability_mapping(index, item)
    spec = CapabilitySpec.from_mapping(item)
    if spec.kind == "mcp-stdio" and ("allowed_tools" not in item or not spec.allowed_tools):
        raise ValueError(
            "manifest MCP capability {} must declare a non-empty allowed_tools list".format(
                spec.name,
            )
        )
    if spec.kind == "mcp-stdio" and contract_version == AGENT_STACK_POLICY_VERSION:
        _validate_v2_tool_policy(spec)
    return spec


def _manifest_capabilities(value: Mapping, contract_version: str) -> Tuple[CapabilitySpec, ...]:
    """Parse and compose the bounded capability array from one document."""
    raw_capabilities = value.get("capabilities")
    if not isinstance(raw_capabilities, Sequence) or isinstance(
        raw_capabilities, (str, bytes),
    ):
        raise ValueError("agent stack capabilities must be a YAML array")
    return compose_capability_stack(
        tuple(
            _parse_manifest_capability(index, item, contract_version)
            for index, item in enumerate(raw_capabilities)
        )
    )


def load_agent_stack_manifest(
    workspace: str,
    manifest_path: str = ".flyto/agent-stack.yaml",
) -> AgentStackManifest:
    """Load a bounded workspace-local stack profile and fail closed on tool scope."""
    value = _decode_manifest_file(_resolve_manifest_file(workspace, manifest_path))
    contract_version, profile = _manifest_header(value)
    capabilities = _manifest_capabilities(value, contract_version)
    return AgentStackManifest(
        profile=profile,
        capabilities=capabilities,
        manifest_fingerprint=canonical_manifest_fingerprint(
            contract_version, profile, capabilities,
        ),
        contract_version=contract_version,
    )
