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
        if not _NAME_RE.fullmatch(self.name):
            raise ValueError("check name must be a safe identifier")
        if not self.argv or len(self.argv) > 32:
            raise ValueError("check argv must contain between 1 and 32 items")
        if any(not isinstance(arg, str) or not arg or len(arg) > 4096 for arg in self.argv):
            raise ValueError("check argv contains an invalid item")
        if not 1 <= self.timeout_seconds <= 900:
            raise ValueError("check timeout_seconds must be between 1 and 900")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CheckSpec":
        argv = value.get("argv")
        if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)):
            raise ValueError("check argv must be a JSON/YAML array")
        return cls(
            name=str(value.get("name", "")),
            argv=tuple(str(item) for item in argv),
            timeout_seconds=int(value.get("timeout_seconds", 120)),
            required=bool(value.get("required", True)),
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
    env_passthrough: Tuple[str, ...] = ()
    timeout_seconds: int = 10

    def __post_init__(self) -> None:
        if not _NAME_RE.fullmatch(self.name):
            raise ValueError("capability name must be a safe identifier")
        if self.kind not in {"mcp-stdio", "command"}:
            raise ValueError("capability kind must be mcp-stdio or command")
        if not self.argv or len(self.argv) > 32:
            raise ValueError("capability argv must contain between 1 and 32 items")
        if self.kind == "mcp-stdio" and not self.protocol_version:
            raise ValueError("MCP capability protocol_version is required")
        if len(self.required_tools) > 100:
            raise ValueError("capability required_tools cannot exceed 100 items")
        if any(not _NAME_RE.fullmatch(name) for name in self.required_tools):
            raise ValueError("capability required_tools contains an invalid name")
        if len(set(self.required_tools)) != len(self.required_tools):
            raise ValueError("capability required_tools contains duplicates")
        if len(self.env_passthrough) > 32:
            raise ValueError("capability env_passthrough cannot exceed 32 items")
        if any(not _PASSTHROUGH_ENV_RE.fullmatch(name) for name in self.env_passthrough):
            raise ValueError("capability env_passthrough accepts only explicit FLYTO_* names")
        if len(set(self.env_passthrough)) != len(self.env_passthrough):
            raise ValueError("capability env_passthrough contains duplicates")
        if not 1 <= self.timeout_seconds <= 60:
            raise ValueError("capability timeout_seconds must be between 1 and 60")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CapabilitySpec":
        argv = value.get("argv")
        if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)):
            raise ValueError("capability argv must be a JSON/YAML array")
        required_tools = value.get("required_tools", ())
        env_passthrough = value.get("env_passthrough", ())
        if not isinstance(required_tools, Sequence) or isinstance(required_tools, (str, bytes)):
            raise ValueError("capability required_tools must be a JSON/YAML array")
        if not isinstance(env_passthrough, Sequence) or isinstance(env_passthrough, (str, bytes)):
            raise ValueError("capability env_passthrough must be a JSON/YAML array")
        return cls(
            name=str(value.get("name", "")),
            argv=tuple(str(item) for item in argv),
            required=bool(value.get("required", False)),
            kind=str(value.get("kind", "mcp-stdio")),
            contract_version=str(value.get("contract_version", "")),
            protocol_version=str(value.get("protocol_version", "2025-06-18")),
            required_tools=tuple(str(item) for item in required_tools),
            env_passthrough=tuple(str(item) for item in env_passthrough),
            timeout_seconds=int(value.get("timeout_seconds", 10)),
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
