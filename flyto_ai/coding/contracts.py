# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Versioned contracts for the provider-neutral Flyto2 coding control plane."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


CONTRACT_VERSION = "flyto.coding.v1"
CONFIG_VERSION = "flyto.coding-config.v1"
SERVICE_CONTRACT_VERSION = "flyto.coding-service.v1"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_PASSTHROUGH_ENV_RE = re.compile(r"^FLYTO_[A-Z0-9_]{1,120}$")
TOOL_PERMISSION_LEVELS = frozenset({"read_only", "workspace_write", "danger_full"})


def _mapping_string(
    value: Mapping[str, Any], field_name: str, default: str,
) -> str:
    """Read one string field without converting numbers, booleans, or nulls."""
    result = value.get(field_name, default)
    if not isinstance(result, str):
        raise ValueError("{} must be a string".format(field_name))
    return result


def _mapping_bool(
    value: Mapping[str, Any], field_name: str, default: bool,
) -> bool:
    """Read one boolean field without Python truthiness coercion."""
    result = value.get(field_name, default)
    if not isinstance(result, bool):
        raise ValueError("{} must be a boolean".format(field_name))
    return result


def _mapping_int(
    value: Mapping[str, Any], field_name: str, default: int,
) -> int:
    """Read one integer field while rejecting booleans and string numerals."""
    result = value.get(field_name, default)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError("{} must be an integer".format(field_name))
    return result


def _require_string_array(value: Any, field_name: str) -> Tuple[str, ...]:
    """Normalize one JSON/YAML string array without coercing unsafe values."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("{} must be a JSON/YAML array".format(field_name))
    if any(not isinstance(item, str) for item in value):
        raise ValueError("{} must contain only strings".format(field_name))
    return tuple(value)


def _validate_argv(argv: Sequence[str], field_name: str) -> None:
    """Enforce the shared bounded argv contract used at process boundaries."""
    if not argv or len(argv) > 32:
        raise ValueError("{} must contain between 1 and 32 items".format(field_name))
    if any(not isinstance(arg, str) or not arg or len(arg) > 4096 for arg in argv):
        raise ValueError("{} contains an invalid item".format(field_name))


def _validate_tool_names(field_name: str, names: Sequence[str]) -> None:
    """Validate a bounded, unique tuple of provider-neutral tool names."""
    if len(names) > 100:
        raise ValueError("{} cannot exceed 100 items".format(field_name))
    if any(not isinstance(name, str) or not _NAME_RE.fullmatch(name) for name in names):
        raise ValueError("{} contains an invalid name".format(field_name))
    if len(set(names)) != len(names):
        raise ValueError("{} contains duplicates".format(field_name))


def _normalize_tool_permissions(value: Any) -> Tuple[Tuple[str, str], ...]:
    """Convert one permission object into a deterministic immutable policy."""
    if not isinstance(value, Mapping):
        raise ValueError("capability tool_permissions must be a JSON/YAML object")
    if any(
        not isinstance(name, str) or not isinstance(level, str)
        for name, level in value.items()
    ):
        raise ValueError("capability tool_permissions keys and values must be strings")
    return tuple(sorted(value.items()))


def _validate_tool_permissions(
    permissions: Sequence[Tuple[str, str]],
    allowed_tools: Sequence[str],
) -> None:
    """Validate permission pairs independently from profile-version policy."""
    if len(permissions) > 100:
        raise ValueError("capability tool_permissions cannot exceed 100 items")
    if any(not isinstance(item, (list, tuple)) or len(item) != 2 for item in permissions):
        raise ValueError("capability tool_permissions entries must be name/level pairs")
    permission_names = tuple(item[0] for item in permissions)
    if any(
        not isinstance(name, str) or not _NAME_RE.fullmatch(name)
        for name in permission_names
    ):
        raise ValueError("capability tool_permissions contains an invalid tool name")
    if len(set(permission_names)) != len(permission_names):
        raise ValueError("capability tool_permissions contains duplicate tool names")
    if any(
        not isinstance(item[1], str) or item[1] not in TOOL_PERMISSION_LEVELS
        for item in permissions
    ):
        raise ValueError("capability tool_permissions contains an invalid permission level")
    if allowed_tools and not set(permission_names).issubset(allowed_tools):
        raise ValueError("capability tool_permissions must refer only to allowed_tools")


class ApprovalPolicy(str, Enum):
    """Whether a host may pause for operations outside the sandbox policy."""

    NEVER = "never"
    ON_REQUEST = "on-request"
    ON_FAILURE = "on-failure"
    ALWAYS = "always"


class SandboxMode(str, Enum):
    """Native workspace authority. There is deliberately no unrestricted mode."""

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"


class CodingJobState(str, Enum):
    """Durable state exposed by the detachable coding service."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class CheckSpec:
    """One real, argv-only verification command."""

    name: str
    argv: Tuple[str, ...]
    timeout_seconds: int = 120
    required: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.fullmatch(self.name):
            raise ValueError("check name must be a safe identifier")
        _validate_argv(self.argv, "check argv")
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, int,
        ):
            raise ValueError("check timeout_seconds must be an integer")
        if not 1 <= self.timeout_seconds <= 900:
            raise ValueError("check timeout_seconds must be between 1 and 900")
        if not isinstance(self.required, bool):
            raise ValueError("check required must be a boolean")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CheckSpec":
        if not isinstance(value, Mapping):
            raise ValueError("check must be a JSON/YAML object")
        argv = _require_string_array(value.get("argv"), "check argv")
        return cls(
            name=_mapping_string(value, "name", ""),
            argv=argv,
            timeout_seconds=_mapping_int(value, "timeout_seconds", 120),
            required=_mapping_bool(value, "required", True),
        )


@dataclass(frozen=True)
class CapabilitySpec:
    """A detachable external capability exposed through an argv boundary."""

    name: str
    argv: Tuple[str, ...]
    required: bool = False
    kind: str = "mcp-stdio"
    contract_version: str = ""
    protocol_version: str = "2025-06-18"
    required_tools: Tuple[str, ...] = ()
    allowed_tools: Tuple[str, ...] = ()
    tool_permissions: Tuple[Tuple[str, str], ...] = ()
    env_passthrough: Tuple[str, ...] = ()
    timeout_seconds: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.fullmatch(self.name):
            raise ValueError("capability name must be a safe identifier")
        if not isinstance(self.required, bool):
            raise ValueError("capability required must be a boolean")
        if not isinstance(self.kind, str) or self.kind not in {"mcp-stdio", "command"}:
            raise ValueError("capability kind must be mcp-stdio or command")
        _validate_argv(self.argv, "capability argv")
        if not isinstance(self.contract_version, str):
            raise ValueError("capability contract_version must be a string")
        if not isinstance(self.protocol_version, str):
            raise ValueError("capability protocol_version must be a string")
        if self.kind == "mcp-stdio" and not self.protocol_version:
            raise ValueError("MCP capability protocol_version is required")
        _validate_tool_names("capability required_tools", self.required_tools)
        _validate_tool_names("capability allowed_tools", self.allowed_tools)
        if self.allowed_tools and not set(self.required_tools).issubset(self.allowed_tools):
            raise ValueError("capability required_tools must be included in allowed_tools")
        _validate_tool_permissions(self.tool_permissions, self.allowed_tools)
        if self.kind == "command" and self.tool_permissions:
            raise ValueError("command capabilities cannot declare tool_permissions")
        if len(self.env_passthrough) > 32:
            raise ValueError("capability env_passthrough cannot exceed 32 items")
        if any(not _PASSTHROUGH_ENV_RE.fullmatch(name) for name in self.env_passthrough):
            raise ValueError("capability env_passthrough accepts only explicit FLYTO_* names")
        if len(set(self.env_passthrough)) != len(self.env_passthrough):
            raise ValueError("capability env_passthrough contains duplicates")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, int):
            raise ValueError("capability timeout_seconds must be an integer")
        if not 1 <= self.timeout_seconds <= 60:
            raise ValueError("capability timeout_seconds must be between 1 and 60")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CapabilitySpec":
        if not isinstance(value, Mapping):
            raise ValueError("capability must be a JSON/YAML object")
        argv = _require_string_array(value.get("argv"), "capability argv")
        required_tools = _require_string_array(
            value.get("required_tools", ()), "capability required_tools",
        )
        allowed_tools = _require_string_array(
            value.get("allowed_tools", ()), "capability allowed_tools",
        )
        tool_permissions = _normalize_tool_permissions(value.get("tool_permissions", {}))
        env_passthrough = _require_string_array(
            value.get("env_passthrough", ()), "capability env_passthrough",
        )
        return cls(
            name=_mapping_string(value, "name", ""),
            argv=argv,
            required=_mapping_bool(value, "required", False),
            kind=_mapping_string(value, "kind", "mcp-stdio"),
            contract_version=_mapping_string(value, "contract_version", ""),
            protocol_version=_mapping_string(
                value, "protocol_version", "2025-06-18",
            ),
            required_tools=required_tools,
            allowed_tools=allowed_tools,
            tool_permissions=tool_permissions,
            env_passthrough=env_passthrough,
            timeout_seconds=_mapping_int(value, "timeout_seconds", 10),
        )


@dataclass
class CodingTaskRequest:
    """Provider-neutral request used by every coding backend."""

    message: str
    working_dir: str
    thread_id: Optional[str] = None
    resume: bool = False
    approval_policy: ApprovalPolicy = ApprovalPolicy.NEVER
    sandbox_mode: SandboxMode = SandboxMode.WORKSPACE_WRITE
    checks: Tuple[CheckSpec, ...] = ()
    capabilities: Tuple[CapabilitySpec, ...] = ()
    max_attempts: int = 3
    max_rounds: int = 30
    require_changes: bool = True
    config_path: str = ".flyto/coding.yaml"
    command_sandbox_image: str = "python:3.12-slim"

    def __post_init__(self) -> None:
        self.message = self.message.strip()
        if not self.message or len(self.message) > 200_000:
            raise ValueError("message must contain between 1 and 200000 characters")
        root = Path(self.working_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("working_dir must be an existing directory")
        self.working_dir = str(root)
        if self.thread_id is not None and not _NAME_RE.fullmatch(self.thread_id):
            raise ValueError("thread_id must be a safe identifier")
        if self.resume and not self.thread_id:
            raise ValueError("resume requires thread_id")
        if not 1 <= self.max_attempts <= 5:
            raise ValueError("max_attempts must be between 1 and 5")
        if not 1 <= self.max_rounds <= 100:
            raise ValueError("max_rounds must be between 1 and 100")
        if (
            not self.command_sandbox_image
            or len(self.command_sandbox_image) > 255
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:-]*", self.command_sandbox_image)
        ):
            raise ValueError("command_sandbox_image is invalid")
        self.approval_policy = ApprovalPolicy(self.approval_policy)
        self.sandbox_mode = SandboxMode(self.sandbox_mode)
        self.checks = tuple(self.checks)
        self.capabilities = tuple(self.capabilities)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    required: bool
    exit_code: Optional[int]
    duration_ms: int
    output_sha256: str
    output_preview: str = ""
    error: Optional[str] = None


@dataclass(frozen=True)
class CapabilityStatus:
    name: str
    available: bool
    required: bool
    kind: str
    contract_version: str = ""
    negotiated_protocol_version: str = ""
    server_name: str = ""
    tool_count: int = 0
    tools: Tuple[str, ...] = ()
    missing_tools: Tuple[str, ...] = ()
    error: Optional[str] = None


@dataclass
class CodingTaskResult:
    ok: bool
    message: str
    thread_id: str
    attempts: int
    status: str
    contract_version: str = CONTRACT_VERSION
    files_changed: List[str] = field(default_factory=list)
    checks: List[CheckResult] = field(default_factory=list)
    capabilities: List[CapabilityStatus] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    rounds_used: int = 0
    evidence_path: str = ""
    failure_code: Optional[str] = None
    command_sandbox: str = ""


@dataclass(frozen=True)
class CodingJobReceipt:
    """Secret-free snapshot returned by HTTP and MCP service facades."""

    job_id: str
    state: CodingJobState
    submitted_at: float
    updated_at: float
    service_contract_version: str = SERVICE_CONTRACT_VERSION
    thread_id: str = ""
    evidence_sha256: str = ""
    result: Optional[CodingTaskResult] = None
    failure_code: Optional[str] = None
