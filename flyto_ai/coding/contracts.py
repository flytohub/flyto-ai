# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Versioned contracts for the provider-neutral Flyto2 coding control plane."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


CONTRACT_VERSION = "flyto.coding.v1"
CONFIG_VERSION = "flyto.coding-config.v1"
# v2 adds the audit states, verdict, finding contract, and the receipt fields
# that bind one audit to one implementation revision. The provider-neutral
# task contract above is deliberately unchanged.
SERVICE_CONTRACT_VERSION = "flyto.coding-service.v2"
LEGACY_SERVICE_CONTRACT_VERSIONS = ("flyto.coding-service.v1",)
SUPPORTED_SERVICE_CONTRACT_VERSIONS = (
    LEGACY_SERVICE_CONTRACT_VERSIONS + (SERVICE_CONTRACT_VERSION,)
)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_PASSTHROUGH_ENV_RE = re.compile(r"^FLYTO_[A-Z0-9_]{1,120}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_AUDIT_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{1,63}$")
_AUDIT_EVIDENCE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:#/@-]{0,255}$")
# C0, DEL, and C1. Findings are machine-readable feedback, not log transport.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
TOOL_PERMISSION_LEVELS = frozenset({"read_only", "workspace_write", "danger_full"})
MAX_AUDIT_FINDINGS = 50
MAX_AUDIT_MESSAGE_CHARS = 2000
MAX_AUDIT_EVIDENCE_REF_CHARS = 256
MAX_IMPLEMENTATION_SESSION_ID_CHARS = 128
MAX_AUDIT_ROUNDS = 100


def require_revision_sha256(value: Any, field_name: str) -> str:
    """Validate one exact lowercase 64-hex digest without coercing anything."""
    if isinstance(value, bool) or not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError("{} must be a lowercase 64-character sha256 digest".format(field_name))
    return value


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
    """Durable state exposed by the detachable coding service.

    The four original states keep their exact public JSON values. The audit
    states are additive and name their actor explicitly, so a reader cannot
    confuse "the implementer finished" with "Codex accepted the change".
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_CODEX_AUDIT = "awaiting_codex_audit"
    REWORK_QUEUED = "rework_queued"
    REWORK_RUNNING = "rework_running"
    CODEX_ACCEPTED = "codex_accepted"


#: States that must already bind one exact implementation revision.
AUDIT_BOUND_CODING_JOB_STATES = frozenset({
    CodingJobState.AWAITING_CODEX_AUDIT,
    CodingJobState.REWORK_QUEUED,
    CodingJobState.REWORK_RUNNING,
    CodingJobState.CODEX_ACCEPTED,
})
#: States only reachable after Codex has recorded at least one audit round.
AUDITED_CODING_JOB_STATES = frozenset({
    CodingJobState.REWORK_QUEUED,
    CodingJobState.REWORK_RUNNING,
    CodingJobState.CODEX_ACCEPTED,
})
#: States that never change again without a new job.
TERMINAL_CODING_JOB_STATES = frozenset({
    CodingJobState.COMPLETED,
    CodingJobState.FAILED,
    CodingJobState.CODEX_ACCEPTED,
})


class CodingAuditVerdict(str, Enum):
    """The authenticated auditor's binding decision on one revision."""

    ACCEPT = "accept"
    REWORK = "rework"

    @property
    def requires_findings(self) -> bool:
        """`rework` must explain itself; `accept` must not carry findings."""
        return self is CodingAuditVerdict.REWORK


class CodingAuditSeverity(str, Enum):
    """How strongly one finding argues against landing the revision."""

    BLOCKER = "blocker"
    MAJOR = "major"
    MINOR = "minor"


_AUDIT_FINDING_FIELDS = frozenset({"code", "severity", "message", "evidence_ref"})


@dataclass(frozen=True)
class CodingAuditFinding:
    """One bounded, machine-addressable reason a revision needs rework.

    This is deterministic feedback for the implementer, not an evidence
    channel: it carries a stable code, a severity, a bounded message, and an
    optional bounded reference. Raw logs, check output, credentials, and
    arbitrary nested payloads have no representation here.
    """

    code: str
    severity: CodingAuditSeverity
    message: str
    evidence_ref: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.code, bool) or not isinstance(self.code, str):
            raise ValueError("audit finding code must be a string")
        if not _AUDIT_CODE_RE.fullmatch(self.code):
            raise ValueError("audit finding code must be a stable safe identifier")
        if isinstance(self.severity, bool) or not isinstance(self.severity, str):
            raise ValueError("audit finding severity must be a string")
        try:
            object.__setattr__(self, "severity", CodingAuditSeverity(self.severity))
        except ValueError as exc:
            raise ValueError("audit finding severity must be blocker, major, or minor") from exc
        if isinstance(self.message, bool) or not isinstance(self.message, str):
            raise ValueError("audit finding message must be a string")
        if not 1 <= len(self.message) <= MAX_AUDIT_MESSAGE_CHARS:
            raise ValueError(
                "audit finding message must contain between 1 and {} characters".format(
                    MAX_AUDIT_MESSAGE_CHARS,
                ),
            )
        if not self.message.strip():
            raise ValueError("audit finding message must contain visible text")
        if _CONTROL_CHARS_RE.search(self.message):
            raise ValueError("audit finding message cannot contain control characters")
        if isinstance(self.evidence_ref, bool) or not isinstance(self.evidence_ref, str):
            raise ValueError("audit finding evidence_ref must be a string")
        if self.evidence_ref:
            if len(self.evidence_ref) > MAX_AUDIT_EVIDENCE_REF_CHARS:
                raise ValueError(
                    "audit finding evidence_ref cannot exceed {} characters".format(
                        MAX_AUDIT_EVIDENCE_REF_CHARS,
                    ),
                )
            if not _AUDIT_EVIDENCE_REF_RE.fullmatch(self.evidence_ref):
                raise ValueError("audit finding evidence_ref is not a bounded safe reference")
            if ".." in self.evidence_ref:
                raise ValueError("audit finding evidence_ref cannot traverse paths")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CodingAuditFinding":
        if not isinstance(value, Mapping):
            raise ValueError("audit finding must be a JSON object")
        unknown = set(value) - _AUDIT_FINDING_FIELDS
        if unknown:
            raise ValueError("unsupported audit finding fields: {}".format(
                ", ".join(sorted(str(name) for name in unknown)),
            ))
        return cls(
            code=_mapping_string(value, "code", ""),
            severity=_mapping_string(value, "severity", ""),
            message=_mapping_string(value, "message", ""),
            evidence_ref=_mapping_string(value, "evidence_ref", ""),
        )

    def to_mapping(self) -> Dict[str, str]:
        """Return the canonical JSON projection used for hashing and transport."""
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "evidence_ref": self.evidence_ref,
        }


def audit_findings_sha256(findings: Sequence[CodingAuditFinding]) -> str:
    """Digest one audit round's findings so a receipt can bind them exactly."""
    items = tuple(findings)
    if len(items) > MAX_AUDIT_FINDINGS:
        raise ValueError("audit findings cannot exceed {} items".format(MAX_AUDIT_FINDINGS))
    if any(not isinstance(item, CodingAuditFinding) for item in items):
        raise ValueError("audit findings must be CodingAuditFinding values")
    payload = json.dumps(
        [item.to_mapping() for item in items],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_audit_submission(
    verdict: CodingAuditVerdict,
    findings: Sequence[CodingAuditFinding],
) -> Tuple[CodingAuditVerdict, Tuple[CodingAuditFinding, ...]]:
    """Bind a verdict to coherent findings.

    Transports validate shape; the authenticated service calls this to reject
    an unexplained `rework` or an `accept` that still lists objections. The
    implementer never reaches this function on its own behalf.
    """
    verdict = CodingAuditVerdict(verdict)
    items = tuple(findings)
    if any(not isinstance(item, CodingAuditFinding) for item in items):
        raise ValueError("audit findings must be CodingAuditFinding values")
    if len(items) > MAX_AUDIT_FINDINGS:
        raise ValueError("audit findings cannot exceed {} items".format(MAX_AUDIT_FINDINGS))
    keys = tuple((item.code, item.evidence_ref) for item in items)
    if len(set(keys)) != len(keys):
        raise ValueError("audit findings contain duplicate code/evidence_ref pairs")
    if verdict.requires_findings and not items:
        raise ValueError("a rework verdict requires at least one audit finding")
    if not verdict.requires_findings and items:
        raise ValueError("an accept verdict cannot carry audit findings")
    return verdict, items


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
    """Secret-free snapshot returned by HTTP and MCP service facades.

    The audit fields are additive and default to the legacy values, so a
    caller that constructs a pre-audit receipt keeps working. `landable` is
    eligibility evidence about an audited revision; nothing in this package
    turns it into a commit, push, or publish.
    """

    job_id: str
    state: CodingJobState
    submitted_at: float
    updated_at: float
    service_contract_version: str = SERVICE_CONTRACT_VERSION
    thread_id: str = ""
    evidence_sha256: str = ""
    result: Optional[CodingTaskResult] = None
    failure_code: Optional[str] = None
    implementation_backend: str = ""
    implementation_session_id: str = ""
    implementation_revision_sha256: str = ""
    audit_count: int = 0
    rework_count: int = 0
    audit_findings_sha256: str = ""
    landable: bool = False
    #: Whether the selected implementer was actually invoked for this job.
    #: Recorded by the host immediately before the call, so a round that failed
    #: in a pre-implementer lane reports `False` truthfully.
    implementer_started: bool = False
    #: Secret-free proof of which host-owned lanes ran around the round.
    #: Absent for a legacy direct-library service that has no strict route.
    route_receipt: Optional[Dict[str, Any]] = None
    #: Digest-validated proof that this round ran on the host-owned emergency
    #: overflow lane instead of the strict route. Present only for a job the
    #: breaker actually diverted; never a substitute for a strict receipt.
    emergency_authority: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", CodingJobState(self.state))
        if not isinstance(self.implementation_backend, str):
            raise ValueError("implementation_backend must be a string")
        if self.implementation_backend and not _NAME_RE.fullmatch(self.implementation_backend):
            raise ValueError("implementation_backend must be a safe identifier")
        # The session id is opaque to this contract: it is bounded and free of
        # control characters, but never parsed or given meaning here.
        if isinstance(self.implementation_session_id, bool) or not isinstance(
            self.implementation_session_id, str,
        ):
            raise ValueError("implementation_session_id must be a string")
        if (
            len(self.implementation_session_id) > MAX_IMPLEMENTATION_SESSION_ID_CHARS
            or _CONTROL_CHARS_RE.search(self.implementation_session_id)
        ):
            raise ValueError("implementation_session_id must be a bounded opaque token")
        for digest_field in ("implementation_revision_sha256", "audit_findings_sha256"):
            digest = getattr(self, digest_field)
            if isinstance(digest, bool) or not isinstance(digest, str):
                raise ValueError("{} must be a string".format(digest_field))
            if digest:
                require_revision_sha256(digest, digest_field)
        for count_field in ("audit_count", "rework_count"):
            count = getattr(self, count_field)
            if isinstance(count, bool) or not isinstance(count, int):
                raise ValueError("{} must be an integer".format(count_field))
            if not 0 <= count <= MAX_AUDIT_ROUNDS:
                raise ValueError("{} must be between 0 and {}".format(count_field, MAX_AUDIT_ROUNDS))
        if self.rework_count > self.audit_count:
            raise ValueError("rework_count cannot exceed audit_count")
        if bool(self.audit_count) != bool(self.audit_findings_sha256):
            raise ValueError("audit_count and audit_findings_sha256 must be recorded together")
        if self.state in AUDIT_BOUND_CODING_JOB_STATES:
            # Phase 2 proves that rework continues the same implementation
            # session, so the identity must already be bound to the revision.
            if not self.implementation_revision_sha256:
                raise ValueError(
                    "{} requires implementation_revision_sha256".format(self.state.value),
                )
            if not self.implementation_backend:
                raise ValueError("{} requires implementation_backend".format(self.state.value))
            if not self.implementation_session_id:
                raise ValueError("{} requires implementation_session_id".format(self.state.value))
        if self.state in AUDITED_CODING_JOB_STATES and self.audit_count < 1:
            raise ValueError("{} requires at least one recorded audit".format(self.state.value))
        if not isinstance(self.landable, bool):
            raise ValueError("landable must be a boolean")
        if not isinstance(self.implementer_started, bool):
            raise ValueError("implementer_started must be a boolean")
        # Acceptance and landability are the same public fact seen from two
        # sides. Neither is an action: nothing here stages, commits, or pushes.
        if self.landable and self.state is not CodingJobState.CODEX_ACCEPTED:
            raise ValueError("only a Codex-accepted receipt may be landable")
        if self.state is CodingJobState.CODEX_ACCEPTED and not self.landable:
            raise ValueError("a Codex-accepted receipt must be landable")
        if self.route_receipt is not None:
            if not isinstance(self.route_receipt, Mapping):
                raise ValueError("route_receipt must be a JSON object")
            # Revalidate the persisted lane evidence, so a tampered or
            # truncated record cannot present itself as a passed route.
            from flyto_ai.coding.route import CodingRouteReceipt

            route = CodingRouteReceipt.from_mapping(self.route_receipt)
            if self.landable and not route.ok:
                raise ValueError("a failed coding route cannot produce a landable receipt")
            if self.landable and not route.strict:
                raise ValueError(
                    "a landable receipt requires strict coding route evidence",
                )
            object.__setattr__(self, "route_receipt", route.to_mapping())
        if self.emergency_authority is not None:
            if not isinstance(self.emergency_authority, Mapping):
                raise ValueError("emergency_authority must be a JSON object")
            # The two authorities are alternatives, never a blend. A round that
            # claimed both would let a failed strict route borrow emergency
            # landability, or the reverse.
            if self.route_receipt is not None:
                raise ValueError(
                    "a receipt cannot carry both route and emergency authority",
                )
            from flyto_ai.coding.emergency import EmergencyAuthorityReceipt

            authority = EmergencyAuthorityReceipt.from_mapping(self.emergency_authority)
            if self.landable and not authority.sealed:
                raise ValueError(
                    "a landable emergency receipt must bind its exact round",
                )
            if self.landable and not authority.checks_enforced:
                raise ValueError(
                    "a landable emergency receipt requires passed required checks",
                )
            # A sealed authority must describe *this* receipt. Comparing the
            # binding to the public fields keeps a serialized receipt internally
            # consistent even outside the service that produced it.
            for bound, public, label in (
                (authority.job_id, self.job_id, "job"),
                (
                    authority.session_id, self.implementation_session_id,
                    "implementation session",
                ),
                (
                    authority.revision_sha256, self.implementation_revision_sha256,
                    "implementation revision",
                ),
            ):
                if bound and bound != public:
                    raise ValueError(
                        "emergency authority does not bind this receipt's {}".format(
                            label,
                        ),
                    )
            if (
                authority.implementer_backend
                and self.implementation_backend
                and authority.implementer_backend != self.implementation_backend
            ):
                raise ValueError(
                    "emergency authority names a different implementation backend",
                )
            object.__setattr__(self, "emergency_authority", authority.to_mapping())
