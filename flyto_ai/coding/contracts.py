# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Versioned contracts for the provider-neutral Flyto2 coding control plane."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# The generic mission kernel owns the vocabulary and the bounds; this layer owns
# only the coding-side envelope that names one. Importing the constants rather
# than restating them is what stops a request this contract accepts from being a
# mission `MissionStore` would refuse. Nothing here imports the scheduler, the
# store, or any durable state - see the mission section below.
from flyto_ai.orchestration.mission_control import (
    DISPOSITIONS as MISSION_DISPOSITIONS,
    LANE_PRIMARY as MISSION_LANE_PRIMARY,
    LANES as MISSION_LANES,
    MAX_ACCEPTANCE_CRITERIA as MISSION_MAX_ACCEPTANCE_CRITERIA,
    MAX_DEPENDENCIES as MISSION_MAX_DEPENDENCIES,
    MAX_FIELD_CHARS as MISSION_MAX_FIELD_CHARS,
    MAX_PRIORITY as MISSION_MAX_PRIORITY,
    MAX_TEXT_CHARS as MISSION_MAX_TEXT_CHARS,
    DISPOSITION_FIXED as MISSION_DISPOSITION_FIXED,
    MISSION_COMPLETED,
    MISSION_OPEN,
    STATUS_CLOSED as MISSION_STATUS_CLOSED,
    STATUS_DISPATCHED as MISSION_STATUS_DISPATCHED,
    STATUS_READY as MISSION_STATUS_READY,
    AcceptanceCriterion,
    MissionError,
)


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
_CONFIG_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
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
MAX_IMPLEMENTATION_BLOCKERS = 16
#: How many check names any durable or public projection will carry.
MAX_VERIFICATION_BLOCKERS = 8


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
    REWORK_ROUTE_BLOCKED = "rework_route_blocked"
    CODEX_ACCEPTED = "codex_accepted"


#: States that must already bind one exact implementation revision.
AUDIT_BOUND_CODING_JOB_STATES = frozenset({
    CodingJobState.AWAITING_CODEX_AUDIT,
    CodingJobState.REWORK_QUEUED,
    CodingJobState.REWORK_RUNNING,
    CodingJobState.REWORK_ROUTE_BLOCKED,
    CodingJobState.CODEX_ACCEPTED,
})
#: States only reachable after Codex has recorded at least one audit round.
AUDITED_CODING_JOB_STATES = frozenset({
    CodingJobState.REWORK_QUEUED,
    CodingJobState.REWORK_RUNNING,
    CodingJobState.REWORK_ROUTE_BLOCKED,
    CodingJobState.CODEX_ACCEPTED,
})
#: How far a job got before it failed. A closed set, because a caller decides
#: what to do next from the phase, not from prose.
FAILURE_PHASE_PREFLIGHT = "preflight"
FAILURE_PHASE_PROVIDER = "provider"
FAILURE_PHASE_VERIFICATION = "verification"
FAILURE_PHASE_IMPLEMENTATION = "implementation"
FAILURE_PHASE_WORKSPACE = "workspace"
FAILURE_PHASE_CAPACITY = "capacity"
FAILURE_PHASE_SERVICE = "service"

#: Bounded tokens naming work a human must do before a retry could succeed.
#: Separate from the preflight allowlist because these are provider-account
#: actions, not repository-contract actions.
ACTION_REFRESH_PROVIDER_CREDENTIALS = "refresh_provider_credentials"
ACTION_RESTORE_PROVIDER_QUOTA = "restore_provider_quota"
ACTION_REVISE_REQUEST_FOR_PROVIDER_POLICY = "revise_request_for_provider_policy"
#: The contract moved under a running job. Nothing is broken and nothing is
#: retryable: the job was authorized against a document that no longer exists,
#: and only a fresh submission can be authorized against the current one.
ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT = "resubmit_against_current_contract"
#: Retry the same repair round after a host-owned route lane is restored.  This
#: is not a provider continuation and consumes a separate, one-shot host retry.
ACTION_RETRY_REWORK_ROUTE = "retry_rework_route"

#: A round refused because the repository contract changed after the job was
#: authorized. Deliberately *not* a provider code: no model was called, and
#: blaming one hides a substitution attempt behind a transport-shaped failure.
VERIFICATION_CONTRACT_CHANGED = "verification_contract_changed"

#: This host cannot isolate a repository-declared action, so no round that
#: could invoke one may start. Not a provider fault and not a contract fault:
#: the deployment is missing the boundary the action surface requires.
ACTION_SANDBOX_UNAVAILABLE = "action_sandbox_unavailable"
#: Install or start the container runtime and pull the pinned action image.
ACTION_PROVISION_ACTION_SANDBOX = "provision_action_sandbox"

#: Install the declared verification tool this host is missing. The same token
#: the submit-time preflight uses, defined once here so the early refusal and
#: the late one can never drift into two different words for one job of work.
ACTION_INSTALL_VERIFICATION_TOOL = "install_verification_tool"
#: Raise or divide the *configured* per-job spend ceiling. Deliberately not the
#: quota action: quota is capacity the provider account does not have, and the
#: fix is a purchase somebody else makes. This ceiling is a number this host was
#: told to enforce, and the fix is a decision the operator makes about how much
#: one job may spend - or about splitting the work into bounded segments.
ACTION_ADJUST_CODING_JOB_BUDGET = "adjust_coding_job_budget"

JOB_FAILURE_ACTIONS: Tuple[str, ...] = (
    ACTION_ADJUST_CODING_JOB_BUDGET,
    ACTION_INSTALL_VERIFICATION_TOOL,
    ACTION_REFRESH_PROVIDER_CREDENTIALS,
    ACTION_RESTORE_PROVIDER_QUOTA,
    ACTION_PROVISION_ACTION_SANDBOX,
    ACTION_RETRY_REWORK_ROUTE,
    ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,
    ACTION_REVISE_REQUEST_FOR_PROVIDER_POLICY,
)

#: `failure_code` -> (phase, retryable, required actions).
#:
#: Retryable means precisely "repeating the identical request could succeed
#: with nothing else changed". Only transient provider capacity qualifies.
#: Auth, quota and policy are terminal until somebody acts, so they carry an
#: action token instead; an unrecognized failure is never guessed to be
#: transient and stays conservative.
JOB_FAILURE_SEMANTICS: Dict[str, Tuple[str, bool, Tuple[str, ...]]] = {
    "rework_route_blocked": (
        FAILURE_PHASE_VERIFICATION, False, (ACTION_RETRY_REWORK_ROUTE,),
    ),
    "rework_route_recovery_exhausted": (
        FAILURE_PHASE_VERIFICATION, False, (),
    ),
    "provider_capacity_unavailable": (FAILURE_PHASE_PROVIDER, True, ()),
    "provider_auth_failed": (
        FAILURE_PHASE_PROVIDER, False, (ACTION_REFRESH_PROVIDER_CREDENTIALS,),
    ),
    "provider_quota_exhausted": (
        FAILURE_PHASE_PROVIDER, False, (ACTION_RESTORE_PROVIDER_QUOTA,),
    ),
    "provider_policy_refused": (
        FAILURE_PHASE_PROVIDER, False, (ACTION_REVISE_REQUEST_FOR_PROVIDER_POLICY,),
    ),
    "provider_failed": (FAILURE_PHASE_PROVIDER, False, ()),
    # Durable Indexer plan authority, and the cumulative scope it governs.
    # `verification` rather than `preflight`: the job exists and already holds a
    # worktree claim by the time these are raised, and the base contract says
    # `preflight` means neither does. Terminal for this job - an identical
    # rework cannot restore a durable fact that is missing or contradicted - so
    # the honest recovery is a fresh job against the authority that exists now.
    "plan_authority_unavailable": (
        FAILURE_PHASE_VERIFICATION, False, (ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,),
    ),
    "plan_authority_unsealable": (
        FAILURE_PHASE_VERIFICATION, False, (ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,),
    ),
    "cumulative_scope_unproven": (
        FAILURE_PHASE_VERIFICATION, False, (ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,),
    ),
    "cumulative_scope_unbounded": (
        FAILURE_PHASE_VERIFICATION, False, (ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,),
    ),
    "cumulative_scope_unsafe": (
        FAILURE_PHASE_VERIFICATION, False, (ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,),
    ),
    "cumulative_session_unproven": (
        FAILURE_PHASE_VERIFICATION, False, (ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,),
    ),
    "cumulative_workspace_unproven": (
        FAILURE_PHASE_WORKSPACE, False, (ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,),
    ),
    "cumulative_claim_unproven": (
        FAILURE_PHASE_WORKSPACE, False, (ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,),
    ),
    "cumulative_resume_unproven": (
        FAILURE_PHASE_VERIFICATION, False, (ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,),
    ),
    "cumulative_revision_mismatch": (
        FAILURE_PHASE_VERIFICATION, False, (ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,),
    ),
    "turn_limit_exceeded": (FAILURE_PHASE_PROVIDER, False, ()),
    "verification_required": (FAILURE_PHASE_PREFLIGHT, False, ()),
    # The configured per-job ceiling was reached. A provider-phase stop, never
    # retryable unchanged - repeating the identical request spends the identical
    # ceiling and stops in the identical place - and answered by an operator
    # decision about the budget rather than by a purchase.
    "provider_job_budget_exhausted": (
        FAILURE_PHASE_PROVIDER, False, (ACTION_ADJUST_CODING_JOB_BUDGET,),
    ),
    # A required check exists and cannot be launched. Preflight normally
    # refuses this before a job exists; reaching it from an adapter means the
    # tool went away between submit and the round, and it is still the same
    # answer to the same person - not a verdict on the change.
    "verification_tool_missing": (
        FAILURE_PHASE_PREFLIGHT, False, (ACTION_INSTALL_VERIFICATION_TOOL,),
    ),
    "verification_failed": (FAILURE_PHASE_VERIFICATION, False, ()),
    "invalid_config": (FAILURE_PHASE_PREFLIGHT, False, ()),
    "required_capability_unavailable": (FAILURE_PHASE_PREFLIGHT, False, ()),
    "snapshot_failed": (FAILURE_PHASE_WORKSPACE, False, ()),
    "session_binding_failed": (FAILURE_PHASE_IMPLEMENTATION, False, ()),
    "no_changes": (FAILURE_PHASE_IMPLEMENTATION, False, ()),
    # Not a failure of the round: the job exhausted the repair budget the host
    # configured for it. Never retryable -- the identical request would settle
    # identically -- and there is no caller action that reopens a settled job.
    "rework_limit_reached": (FAILURE_PHASE_IMPLEMENTATION, False, ()),
    VERIFICATION_CONTRACT_CHANGED: (
        FAILURE_PHASE_PREFLIGHT, False, (ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,),
    ),
    ACTION_SANDBOX_UNAVAILABLE: (
        FAILURE_PHASE_PREFLIGHT, False, (ACTION_PROVISION_ACTION_SANDBOX,),
    ),
}


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
    """One real, argv-only verification command.

    ``proof_kinds`` is an optional pinned semantic claim about a successful
    required check. It lets a strict post-work lane consume repository-owned
    verification without importing the unreviewed worktree into the coding
    service process. The service still runs the command and supplies the
    :class:`CheckResult`; provider prose can never manufacture the evidence.
    """

    name: str
    argv: Tuple[str, ...]
    timeout_seconds: int = 120
    required: bool = True
    proof_kinds: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.fullmatch(self.name):
            raise ValueError("check name must be a safe identifier")
        _validate_argv(self.argv, "check argv")
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, int,
        ):
            raise ValueError("check timeout_seconds must be an integer")
        if not 1 <= self.timeout_seconds <= 1_800:
            raise ValueError("check timeout_seconds must be between 1 and 1800")
        if not isinstance(self.required, bool):
            raise ValueError("check required must be a boolean")
        object.__setattr__(self, "proof_kinds", tuple(self.proof_kinds))
        if len(self.proof_kinds) > 8:
            raise ValueError("check proof_kinds cannot exceed 8 items")
        if any(
            not isinstance(item, str) or not _NAME_RE.fullmatch(item)
            for item in self.proof_kinds
        ):
            raise ValueError("check proof_kinds must contain safe identifiers")
        if len(set(self.proof_kinds)) != len(self.proof_kinds):
            raise ValueError("check proof_kinds contains duplicates")
        if self.proof_kinds and not self.required:
            raise ValueError("a proof check must be required")

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
            proof_kinds=_require_string_array(
                value.get("proof_kinds", ()), "check proof_kinds",
            ),
        )

    def to_mapping(self) -> Dict[str, Any]:
        """Project to the pinned shape without invalidating legacy pins."""

        value: Dict[str, Any] = {
            "name": self.name,
            "argv": list(self.argv),
            "timeout_seconds": self.timeout_seconds,
            "required": self.required,
        }
        # Previous snapshots had no such key. Omitting the empty additive field
        # keeps their canonical identity stable when a new reader loads them.
        if self.proof_kinds:
            value["proof_kinds"] = list(self.proof_kinds)
        return value


def _validate_action_arguments(argv: Sequence[str]) -> None:
    """Keep a reviewed action's *arguments* pointed inside the workspace.

    What this is: a guard against a declared action accidentally naming a path
    outside the tree it runs in - `--config /etc/something`, `~/.aws/creds`,
    `../../other-repo`. Those are almost always a mistake in review, and a
    mistake that is cheap to refuse.

    What this is emphatically *not*: a sandbox. `argv[0]` is a reviewed
    executable and it can do anything the account can do, including reading
    every path refused here. The real containment properties are elsewhere -
    the caller cannot supply arguments at all, and the declaration is bound to
    a job-lifetime config digest, so what runs is what was committed and
    reviewed. This only stops the declaration itself from pointing outward by
    accident.

    `argv[0]` is exempt: naming an interpreter by absolute path is normal and
    is the reviewed decision.
    """

    for argument in argv[1:]:
        if argument.startswith("~"):
            raise ValueError("action arguments must not reference a home directory")
        if argument.startswith("/") and "/" in argument[1:]:
            raise ValueError("action arguments must not be absolute paths")
        parts = argument.replace("\\", "/").split("/")
        if ".." in parts:
            raise ValueError("action arguments must not traverse out of the workspace")


@dataclass(frozen=True)
class ProjectActionSpec:
    """One repository-declared deterministic command, invocable only by name.

    This is the narrow answer to "the implementer needs to run a real project
    command" that does not become "the implementer gets a shell". Everything
    that decides *what runs* is source-controlled: the name, the exact argv,
    the timeout, and an optional subdirectory. A caller supplies the name and
    nothing else, so there is no argument to inject into, no interpolation to
    escape from, and no environment to expand.

    Deliberately not a check. A check is verification the host runs and trusts;
    an action is work the implementer may ask for. Keeping them separate types
    in separate contract keys is what stops a passing action from ever being
    read as a passing check.
    """

    name: str
    argv: Tuple[str, ...]
    timeout_seconds: int = 120
    #: Optional workspace-relative subdirectory. Absolute paths, traversal and
    #: any non-canonical spelling are refused here; containment against the
    #: real workspace is proven again at launch, when the path exists.
    working_subdir: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.fullmatch(self.name):
            raise ValueError("action name must be a safe identifier")
        _validate_argv(self.argv, "action argv")
        _validate_action_arguments(self.argv)
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, int,
        ):
            raise ValueError("action timeout_seconds must be an integer")
        if not 1 <= self.timeout_seconds <= 900:
            raise ValueError("action timeout_seconds must be between 1 and 900")
        if not isinstance(self.description, str) or len(self.description) > 200:
            raise ValueError("action description must be a bounded string")
        if any(not char.isprintable() for char in self.description):
            raise ValueError("action description must not contain control characters")
        if not isinstance(self.working_subdir, str):
            raise ValueError("action working_subdir must be a string")
        if self.working_subdir:
            candidate = self.working_subdir
            if (
                len(candidate) > 1024
                or candidate.startswith(("/", "~"))
                or "\\" in candidate
                or "\x00" in candidate
            ):
                raise ValueError("action working_subdir must be a safe relative path")
            parts = PurePosixPath(candidate).parts
            if not parts or any(part in {"", ".", ".."} for part in parts):
                raise ValueError("action working_subdir must be a safe relative path")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProjectActionSpec":
        if not isinstance(value, Mapping):
            raise ValueError("action must be a JSON/YAML object")
        unknown = set(value) - {
            "name", "argv", "timeout_seconds", "working_subdir", "description",
        }
        if unknown:
            # Fail closed: an unrecognized key may be somebody's idea of an
            # escape hatch, and silently ignoring it would grant it.
            raise ValueError(
                "action contains unsupported keys: {}".format(", ".join(sorted(unknown))),
            )
        return cls(
            name=_mapping_string(value, "name", ""),
            argv=_require_string_array(value.get("argv"), "action argv"),
            timeout_seconds=_mapping_int(value, "timeout_seconds", 120),
            working_subdir=_mapping_string(value, "working_subdir", ""),
            description=_mapping_string(value, "description", ""),
        )


@dataclass(frozen=True)
class ProjectActionResult:
    """What one declared action did. Never evidence that anything is verified."""

    name: str
    ok: bool
    exit_code: Optional[int]
    duration_ms: int
    stdout: str = ""
    stderr: str = ""
    truncated: bool = False
    timed_out: bool = False
    error: str = ""


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
        if not 1 <= self.timeout_seconds <= 900:
            raise ValueError("capability timeout_seconds must be between 1 and 900")

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


#: Domain separation for the pinned-contract identity, so a snapshot digest can
#: never be mistaken for a config-byte digest, a revision digest, or an audit
#: findings digest.
_CONTRACT_SNAPSHOT_DOMAIN = b"flyto.coding-contract-snapshot.v1"
#: The same bounds the source-controlled reader enforces, restated here because
#: a snapshot may also be rebuilt from stored state that never passed through
#: that reader. Anything beyond them is refused rather than truncated.
MAX_SNAPSHOT_CHECKS = 32
MAX_SNAPSHOT_CAPABILITIES = 16
MAX_SNAPSHOT_ACTIONS = 16


@dataclass(frozen=True)
class ContractSnapshot:
    """The verification contract a job is authorized against, by value.

    Pinning a *digest* was enough to notice that `.flyto/coding.yaml` changed;
    it was never enough to keep executing the contract that was authorized. An
    implementation that edits its own contract leaves the host holding a digest
    of bytes that are no longer on disk, so the only remaining answers were
    "re-read the file" - which lets the edit authorize itself - or "refuse",
    which makes the change impossible to finish.

    This carries the checks, capabilities and actions themselves, captured from
    the exact bytes preflight validated before the first provider edit. Later
    rounds execute *this*, never the current file, so a model that weakens,
    replaces, or breaks its own contract still faces the checks it started
    under, and can still reach audit.

    `config_sha256` remains the digest of those exact bytes - it is what an
    auditor uses to find the document this snapshot came from - and
    :meth:`identity` is the content address of the snapshot itself, which is
    what stored state is bound to so tampering is detectable.
    """

    checks: Tuple[CheckSpec, ...] = ()
    capabilities: Tuple[CapabilitySpec, ...] = ()
    actions: Tuple["ProjectActionSpec", ...] = ()
    config_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "actions", tuple(self.actions))
        if len(self.checks) > MAX_SNAPSHOT_CHECKS:
            raise ValueError("contract snapshot exceeds its check bound")
        if len(self.capabilities) > MAX_SNAPSHOT_CAPABILITIES:
            raise ValueError("contract snapshot exceeds its capability bound")
        if len(self.actions) > MAX_SNAPSHOT_ACTIONS:
            raise ValueError("contract snapshot exceeds its action bound")
        if any(not isinstance(item, CheckSpec) for item in self.checks):
            raise ValueError("contract snapshot checks must be CheckSpec values")
        if any(not isinstance(item, CapabilitySpec) for item in self.capabilities):
            raise ValueError("contract snapshot capabilities must be CapabilitySpec values")
        if any(not isinstance(item, ProjectActionSpec) for item in self.actions):
            raise ValueError("contract snapshot actions must be ProjectActionSpec values")
        names = [item.name for item in self.actions]
        if len(set(names)) != len(names):
            raise ValueError("contract snapshot actions contain duplicate names")
        proof_kinds = [
            kind for check in self.checks for kind in check.proof_kinds
        ]
        if len(set(proof_kinds)) != len(proof_kinds):
            raise ValueError("contract snapshot proof_kinds must be unique")
        if not isinstance(self.config_sha256, str) or not _CONFIG_DIGEST_RE.fullmatch(
            self.config_sha256,
        ):
            # A snapshot with no addressable source document is not a pin. There
            # is deliberately no "" case: an empty digest is what an unpinned
            # request already looks like, and the two must not be confusable.
            raise ValueError("contract snapshot config_sha256 must be a sha256 digest")

    def to_mapping(self) -> Dict[str, Any]:
        """Project to bounded JSON, in exactly the shape the reader accepts."""

        return {
            "config_sha256": self.config_sha256,
            "checks": [item.to_mapping() for item in self.checks],
            "capabilities": [
                {
                    "name": item.name,
                    "argv": list(item.argv),
                    "required": item.required,
                    "kind": item.kind,
                    "contract_version": item.contract_version,
                    "protocol_version": item.protocol_version,
                    "required_tools": list(item.required_tools),
                    "allowed_tools": list(item.allowed_tools),
                    # A mapping, not a pair list: `from_mapping` normalizes only
                    # objects, so emitting pairs would make this unreadable.
                    "tool_permissions": {name: level for name, level in item.tool_permissions},
                    "env_passthrough": list(item.env_passthrough),
                    "timeout_seconds": item.timeout_seconds,
                }
                for item in self.capabilities
            ],
            "actions": [
                {
                    "name": item.name,
                    "argv": list(item.argv),
                    "timeout_seconds": item.timeout_seconds,
                    "working_subdir": item.working_subdir,
                    "description": item.description,
                }
                for item in self.actions
            ],
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "ContractSnapshot":
        """Rebuild a snapshot from stored state, revalidating every spec.

        Nothing is repaired and nothing is defaulted into existence: each entry
        goes back through the same constructor the source-controlled reader
        uses, so a hand-edited state file cannot introduce a check, capability
        or action shape the contract grammar would have refused.
        """

        if not isinstance(value, Mapping):
            raise ValueError("contract snapshot must be an object")
        unknown = set(value) - {"config_sha256", "checks", "capabilities", "actions"}
        if unknown:
            raise ValueError("contract snapshot contains unsupported keys")
        raw_checks = value.get("checks", [])
        raw_capabilities = value.get("capabilities", [])
        raw_actions = value.get("actions", [])
        for raw in (raw_checks, raw_capabilities, raw_actions):
            if not isinstance(raw, (list, tuple)):
                raise ValueError("contract snapshot entries must be arrays")
        return cls(
            checks=tuple(CheckSpec.from_mapping(item) for item in raw_checks),
            capabilities=tuple(
                CapabilitySpec.from_mapping(item) for item in raw_capabilities
            ),
            actions=tuple(ProjectActionSpec.from_mapping(item) for item in raw_actions),
            config_sha256=_mapping_string(value, "config_sha256", ""),
        )

    def identity(self) -> str:
        """Content address of the pinned contract itself."""

        digest = hashlib.sha256()
        digest.update(_CONTRACT_SNAPSHOT_DOMAIN)
        digest.update(
            json.dumps(
                self.to_mapping(), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ).encode("utf-8"),
        )
        return digest.hexdigest()

    def has_required_check(self) -> bool:
        return any(check.required for check in self.checks)


# --------------------------------------------------------------------------
# mission envelope
#
# A coding job may optionally name the mission it serves. This section is the
# whole of that contract: bounded immutable value types, one digest, and two
# projections. It deliberately contains no store, no scheduler, no lifecycle,
# and no clock - naming a mission is not joining one, and this slice does not
# create, dispatch, or close anything.
#
# Nothing here is domain-specific. `scope`, the criteria statements, and the
# lineage identifiers are opaque to this layer exactly as they are opaque to the
# kernel; no repository, product, project count, or provider is named.
# --------------------------------------------------------------------------

#: The three durable work-item statuses, in the kernel's own vocabulary.
MISSION_WORK_STATUSES: Tuple[str, ...] = (
    MISSION_STATUS_READY, MISSION_STATUS_DISPATCHED, MISSION_STATUS_CLOSED,
)
#: The two mission lifecycle statuses. Deliberately a separate vocabulary from
#: the work-item statuses above: "this work item is closed" and "the mission is
#: complete" are different claims, and a consumer that had to infer the second
#: from the first would be guessing at exactly the point it matters most.
MISSION_STATUSES: Tuple[str, ...] = (MISSION_OPEN, MISSION_COMPLETED)
#: Exactly the identifier shape the kernel mints: a prefix and twelve decimal
#: digits. Published as a pattern so the MCP and HTTP schemas can refuse a
#: malformed lineage id before it reaches any decoder.
MISSION_ID_PATTERN = "^m-[0-9]{12}$"
WORK_ITEM_ID_PATTERN = "^w-[0-9]{12}$"
_MISSION_ID_RE = re.compile(MISSION_ID_PATTERN)
_WORK_ITEM_ID_RE = re.compile(WORK_ITEM_ID_PATTERN)
#: Domain separation, so a main-axis digest can never be confused with an audit
#: findings digest, a config digest, or a revision digest.
_MISSION_AXIS_DOMAIN = b"flyto.coding-mission-axis.v1"

#: Closed field sets. Every one of them is applied on decode, because an
#: unrecognized key is at best a typo and at worst somebody's escape hatch.
MISSION_ENVELOPE_FIELDS = frozenset({
    "scope", "objective", "desired_result", "acceptance_criteria", "priority",
    "lane", "mission_id", "parent_id", "return_to_id", "depends_on_ids",
})
MISSION_CRITERION_FIELDS = frozenset({"id", "statement"})
#: The *only* mission keys a fleet or public receipt may carry. Objective and
#: desired-result prose, closure rationale, risk, owner, evidence refs, and the
#: workspace path have no member here and no way to acquire one.
MISSION_PROJECTION_FIELDS = frozenset({
    "mission_id", "scope", "work_item_id", "main_axis_sha256", "criteria_ids",
    "lane", "priority", "status", "disposition", "parent_id", "return_to_id",
    "returned_to_main_axis", "mission_status",
})


def _mission_token(value: Any, field_name: str) -> str:
    """Validate one bounded, printable, whitespace-free mission identifier.

    Mirrors the kernel's own token rule against the kernel's own bound, but
    raises `ValueError` rather than `MissionRejected`: this is a transport
    contract, and every facade above it already answers `ValueError` with a
    400-class refusal instead of a 500.
    """

    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError("{} must be a string".format(field_name))
    if not value or len(value) > MISSION_MAX_FIELD_CHARS:
        raise ValueError("{} must contain between 1 and {} characters".format(
            field_name, MISSION_MAX_FIELD_CHARS,
        ))
    if not value.isprintable() or any(char.isspace() for char in value):
        raise ValueError("{} must be printable and free of whitespace".format(field_name))
    return value


def _mission_text(value: Any, field_name: str) -> str:
    """Validate one bounded, printable, single-line mission prose field."""

    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError("{} must be a string".format(field_name))
    if len(value) > MISSION_MAX_TEXT_CHARS:
        raise ValueError("{} cannot exceed {} characters".format(
            field_name, MISSION_MAX_TEXT_CHARS,
        ))
    if not value.isprintable():
        raise ValueError("{} must not contain control characters".format(field_name))
    stripped = value.strip()
    if not stripped:
        raise ValueError("{} must contain visible text".format(field_name))
    return stripped


def _mission_identifier(value: Any, field_name: str, pattern: "re.Pattern[str]") -> str:
    """Insist on an identifier the kernel could actually have minted."""

    if isinstance(value, bool) or not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError("{} is not a well-formed mission identifier".format(field_name))
    return value


def _mission_optional_identifier(
    value: Any, field_name: str, pattern: "re.Pattern[str]",
) -> Optional[str]:
    if value is None:
        return None
    return _mission_identifier(value, field_name, pattern)


def _mission_optional_field(value: Mapping[str, Any], field_name: str) -> Optional[str]:
    """Read one optional string without turning a number or `False` into text."""

    item = value.get(field_name)
    if item is None:
        return None
    if isinstance(item, bool) or not isinstance(item, str):
        raise ValueError("mission {} must be a string".format(field_name))
    return item


def _mission_criterion_from_mapping(value: Any) -> AcceptanceCriterion:
    """Decode one acceptance criterion into the kernel's own value type."""

    if not isinstance(value, Mapping):
        raise ValueError("mission acceptance criterion must be a JSON object")
    unknown = set(value) - MISSION_CRITERION_FIELDS
    if unknown:
        raise ValueError("unsupported mission acceptance criterion fields: {}".format(
            ", ".join(sorted(str(name) for name in unknown)),
        ))
    identifier = _mission_token(value.get("id"), "mission acceptance criterion id")
    statement = _mission_text(
        value.get("statement"), "mission acceptance criterion statement",
    )
    try:
        return AcceptanceCriterion(identifier, statement)
    except MissionError as exc:
        # The kernel is the authority on its own value type. Its refusal is
        # translated, never swallowed, and never widened into a server error.
        raise ValueError("mission acceptance criterion is invalid") from exc


def mission_axis_sha256(
    scope: str,
    objective: str,
    desired_result: str,
    acceptance_criteria: Sequence[AcceptanceCriterion],
) -> str:
    """Digest the immutable main axis: what is being attempted, and its proof.

    Length-prefixed so two distinct field tuples cannot collide by joining, and
    domain-separated so the value cannot be replayed as any other digest in this
    contract. This is what lets a public receipt bind one exact mission contract
    without republishing a single word of its prose.
    """

    accumulator = hashlib.sha256()
    accumulator.update(_MISSION_AXIS_DOMAIN)
    parts: List[str] = [scope, objective, desired_result]
    for criterion in acceptance_criteria:
        parts.append(criterion.id)
        parts.append(criterion.statement)
    for part in parts:
        raw = part.encode("utf-8")
        accumulator.update(b"\n")
        accumulator.update(str(len(raw)).encode("ascii"))
        accumulator.update(b":")
        accumulator.update(raw)
    return accumulator.hexdigest()


@dataclass(frozen=True)
class CodingMissionEnvelope:
    """The mission one coding job serves, named on the request.

    Immutable by construction, because the main axis is immutable by design: a
    mission that turned out to be the wrong mission is a different mission, not
    an edited one. The envelope carries the contract (`scope`, `objective`,
    `desired_result`, `acceptance_criteria`), the queue position (`lane`,
    `priority`), and optional lineage into an existing mission graph.

    Lineage follows the kernel's rule exactly. An envelope with neither
    `parent_id` nor `return_to_id` is a root: it runs in the primary lane and
    depends on nothing, because it is the first work item of its mission. Any
    other envelope is a side item and must name both the parent it descends from
    and the ancestor it hands control back to, inside a mission that already
    exists. Whether the return target really is an ancestor is the store's
    decision, not this contract's; this layer refuses only what it can prove
    from the envelope alone.
    """

    scope: str
    objective: str
    desired_result: str
    acceptance_criteria: Tuple[AcceptanceCriterion, ...]
    priority: int = 0
    lane: str = MISSION_LANE_PRIMARY
    mission_id: Optional[str] = None
    parent_id: Optional[str] = None
    return_to_id: Optional[str] = None
    depends_on_ids: Tuple[str, ...] = ()

    @property
    def is_root(self) -> bool:
        """Whether this envelope names the main axis rather than a side branch."""
        return self.parent_id is None and self.return_to_id is None

    @property
    def criteria_ids(self) -> Tuple[str, ...]:
        return tuple(criterion.id for criterion in self.acceptance_criteria)

    @property
    def main_axis_sha256(self) -> str:
        """Content address of the immutable contract this envelope states."""
        return mission_axis_sha256(
            self.scope, self.objective, self.desired_result, self.acceptance_criteria,
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _mission_token(self.scope, "mission scope"))
        object.__setattr__(
            self, "objective", _mission_text(self.objective, "mission objective"),
        )
        object.__setattr__(
            self,
            "desired_result",
            _mission_text(self.desired_result, "mission desired_result"),
        )

        criteria = self.acceptance_criteria
        if isinstance(criteria, (str, bytes)) or not isinstance(criteria, Sequence):
            raise ValueError("mission acceptance_criteria must be a JSON array")
        criteria = tuple(criteria)
        if not 1 <= len(criteria) <= MISSION_MAX_ACCEPTANCE_CRITERIA:
            raise ValueError(
                "mission acceptance_criteria must contain between 1 and {} items".format(
                    MISSION_MAX_ACCEPTANCE_CRITERIA,
                ),
            )
        if any(not isinstance(item, AcceptanceCriterion) for item in criteria):
            raise ValueError("mission acceptance_criteria must be AcceptanceCriterion values")
        identifiers = tuple(item.id for item in criteria)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("mission acceptance_criteria contains duplicate ids")
        object.__setattr__(self, "acceptance_criteria", criteria)

        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("mission priority must be an integer")
        if not 0 <= self.priority <= MISSION_MAX_PRIORITY:
            raise ValueError("mission priority must be between 0 and {}".format(
                MISSION_MAX_PRIORITY,
            ))
        if (
            isinstance(self.lane, bool)
            or not isinstance(self.lane, str)
            or self.lane not in MISSION_LANES
        ):
            raise ValueError("mission lane must be one of {}".format(
                ", ".join(MISSION_LANES),
            ))

        object.__setattr__(self, "mission_id", _mission_optional_identifier(
            self.mission_id, "mission mission_id", _MISSION_ID_RE,
        ))
        object.__setattr__(self, "parent_id", _mission_optional_identifier(
            self.parent_id, "mission parent_id", _WORK_ITEM_ID_RE,
        ))
        object.__setattr__(self, "return_to_id", _mission_optional_identifier(
            self.return_to_id, "mission return_to_id", _WORK_ITEM_ID_RE,
        ))

        dependencies = self.depends_on_ids
        if isinstance(dependencies, (str, bytes)) or not isinstance(dependencies, Sequence):
            raise ValueError("mission depends_on_ids must be a JSON array")
        dependencies = tuple(dependencies)
        if len(dependencies) > MISSION_MAX_DEPENDENCIES:
            raise ValueError("mission depends_on_ids cannot exceed {} items".format(
                MISSION_MAX_DEPENDENCIES,
            ))
        for item in dependencies:
            _mission_identifier(item, "mission depends_on_ids entry", _WORK_ITEM_ID_RE)
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("mission depends_on_ids contains duplicates")
        object.__setattr__(self, "depends_on_ids", dependencies)

        # Lineage. Half a lineage is the dangerous case: a side item that names
        # a parent but no route home is precisely the work that never returns.
        if (self.parent_id is None) != (self.return_to_id is None):
            raise ValueError(
                "a side work item requires both parent_id and return_to_id",
            )
        if self.is_root:
            if self.lane != MISSION_LANE_PRIMARY:
                raise ValueError("the root work item runs in the primary lane")
            if dependencies:
                raise ValueError("the root work item cannot depend on earlier work")
        elif self.mission_id is None:
            raise ValueError("a side work item requires the mission it descends from")

    @classmethod
    def from_mapping(cls, value: Any) -> "CodingMissionEnvelope":
        """Decode one mission envelope strictly; unknown keys are refused."""

        if not isinstance(value, Mapping):
            raise ValueError("mission must be a JSON object")
        unknown = set(value) - MISSION_ENVELOPE_FIELDS
        if unknown:
            raise ValueError("unsupported mission fields: {}".format(
                ", ".join(sorted(str(name) for name in unknown)),
            ))
        raw_criteria = value.get("acceptance_criteria", ())
        if isinstance(raw_criteria, (str, bytes)) or not isinstance(raw_criteria, Sequence):
            raise ValueError("mission acceptance_criteria must be a JSON array")
        if len(raw_criteria) > MISSION_MAX_ACCEPTANCE_CRITERIA:
            raise ValueError(
                "mission acceptance_criteria cannot exceed {} items".format(
                    MISSION_MAX_ACCEPTANCE_CRITERIA,
                ),
            )
        return cls(
            scope=_mapping_string(value, "scope", ""),
            objective=_mapping_string(value, "objective", ""),
            desired_result=_mapping_string(value, "desired_result", ""),
            acceptance_criteria=tuple(
                _mission_criterion_from_mapping(item) for item in raw_criteria
            ),
            priority=_mapping_int(value, "priority", 0),
            lane=_mapping_string(value, "lane", MISSION_LANE_PRIMARY),
            mission_id=_mission_optional_field(value, "mission_id"),
            parent_id=_mission_optional_field(value, "parent_id"),
            return_to_id=_mission_optional_field(value, "return_to_id"),
            depends_on_ids=_require_string_array(
                value.get("depends_on_ids", ()), "mission depends_on_ids",
            ),
        )

    def to_mapping(self) -> Dict[str, Any]:
        """Round-trip projection of the *request* side, prose included.

        This is not the receipt projection. A request already carries the
        caller's own prose, so nothing is hidden from the caller who wrote it;
        :class:`CodingMissionProjection` is what crosses a fleet or public
        boundary, and it has no prose field at all.
        """

        return {
            "scope": self.scope,
            "objective": self.objective,
            "desired_result": self.desired_result,
            "acceptance_criteria": [
                {"id": criterion.id, "statement": criterion.statement}
                for criterion in self.acceptance_criteria
            ],
            "priority": self.priority,
            "lane": self.lane,
            "mission_id": self.mission_id,
            "parent_id": self.parent_id,
            "return_to_id": self.return_to_id,
            "depends_on_ids": list(self.depends_on_ids),
        }

    def projection(
        self,
        *,
        mission_id: str,
        work_item_id: str,
        status: str,
        mission_status: str = MISSION_OPEN,
        disposition: str = "",
        returned_to_main_axis: bool = False,
    ) -> "CodingMissionProjection":
        """Build the receipt-side projection of this envelope's placed work item.

        The two identifiers and both statuses are supplied by whoever actually
        placed the work, because this contract cannot mint or observe them and
        must not pretend to. Scope, lane, priority, the criteria ids, and the
        main-axis digest come from the envelope itself, so a projection can never
        describe a different contract from the one that was submitted.
        """

        return CodingMissionProjection(
            mission_id=mission_id,
            scope=self.scope,
            work_item_id=work_item_id,
            mission_status=mission_status,
            main_axis_sha256=self.main_axis_sha256,
            criteria_ids=self.criteria_ids,
            lane=self.lane,
            priority=self.priority,
            status=status,
            disposition=disposition,
            parent_id=self.parent_id or "",
            return_to_id=self.return_to_id or "",
            returned_to_main_axis=returned_to_main_axis,
        )


@dataclass(frozen=True)
class CodingMissionProjection:
    """Secret-safe mission facts for a fleet or public coding receipt.

    Every field is an identifier, a digest, a closed-vocabulary token, a bounded
    integer, or a boolean. There is deliberately no field for the objective, the
    desired result, a criterion statement, a closure rationale, a risk, an
    owner, an evidence ref, or a workspace path - not a redacted one, not an
    optional one. A reader learns *which* mission and *where* the work stands;
    it never learns what the mission says.

    `scope` is the exception that proves the rule: it is the kernel's own
    grouping token - bounded, printable, whitespace-free, and never prose - so a
    fleet view can group missions by scope without reading a single word of any
    mission's contract.

    `mission_status` and `status` are two different facts and are carried
    separately. One work item closing says nothing about whether the immutable
    main axis was reached: the kernel completes a mission only when every work
    item is closed *and* the root closed as fixed. A consumer forced to infer
    completion from a single work item's status would routinely infer it wrongly.
    """

    mission_id: str
    scope: str
    work_item_id: str
    main_axis_sha256: str
    criteria_ids: Tuple[str, ...]
    lane: str
    priority: int
    status: str
    #: The mission lifecycle, not this work item's. Open until the kernel has
    #: accepted the whole mission as complete.
    mission_status: str = MISSION_OPEN
    disposition: str = ""
    parent_id: str = ""
    return_to_id: str = ""
    #: Whether control has come back to the main axis. Only a closed side item
    #: can have returned; a root item is already on the axis and never claims it.
    returned_to_main_axis: bool = False

    @property
    def is_root(self) -> bool:
        return not self.parent_id and not self.return_to_id

    def __post_init__(self) -> None:
        _mission_identifier(self.mission_id, "mission mission_id", _MISSION_ID_RE)
        _mission_token(self.scope, "mission scope")
        _mission_identifier(self.work_item_id, "mission work_item_id", _WORK_ITEM_ID_RE)
        require_revision_sha256(self.main_axis_sha256, "mission main_axis_sha256")

        identifiers = self.criteria_ids
        if isinstance(identifiers, (str, bytes)) or not isinstance(identifiers, Sequence):
            raise ValueError("mission criteria_ids must be a JSON array")
        identifiers = tuple(identifiers)
        if not 1 <= len(identifiers) <= MISSION_MAX_ACCEPTANCE_CRITERIA:
            raise ValueError(
                "mission criteria_ids must contain between 1 and {} items".format(
                    MISSION_MAX_ACCEPTANCE_CRITERIA,
                ),
            )
        for item in identifiers:
            _mission_token(item, "mission criteria_ids entry")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("mission criteria_ids contains duplicates")
        object.__setattr__(self, "criteria_ids", identifiers)

        if (
            isinstance(self.lane, bool)
            or not isinstance(self.lane, str)
            or self.lane not in MISSION_LANES
        ):
            raise ValueError("mission lane must be one of {}".format(
                ", ".join(MISSION_LANES),
            ))
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("mission priority must be an integer")
        if not 0 <= self.priority <= MISSION_MAX_PRIORITY:
            raise ValueError("mission priority must be between 0 and {}".format(
                MISSION_MAX_PRIORITY,
            ))
        if (
            isinstance(self.status, bool)
            or not isinstance(self.status, str)
            or self.status not in MISSION_WORK_STATUSES
        ):
            raise ValueError("mission status must be one of {}".format(
                ", ".join(MISSION_WORK_STATUSES),
            ))

        if (
            isinstance(self.mission_status, bool)
            or not isinstance(self.mission_status, str)
            or self.mission_status not in MISSION_STATUSES
        ):
            raise ValueError("mission mission_status must be one of {}".format(
                ", ".join(MISSION_STATUSES),
            ))

        if isinstance(self.disposition, bool) or not isinstance(self.disposition, str):
            raise ValueError("mission disposition must be a string")
        if self.disposition and self.disposition not in MISSION_DISPOSITIONS:
            raise ValueError("mission disposition must be one of {}".format(
                ", ".join(MISSION_DISPOSITIONS),
            ))
        # A closed work item left the queue some particular way, and an open one
        # has not left it at all. Either half of that pair on its own is a claim
        # the kernel could not have produced.
        if bool(self.disposition) != (self.status == MISSION_STATUS_CLOSED):
            raise ValueError(
                "mission disposition and closed status must be recorded together",
            )

        for name in ("parent_id", "return_to_id"):
            item = getattr(self, name)
            if isinstance(item, bool) or not isinstance(item, str):
                raise ValueError("mission {} must be a string".format(name))
            if item:
                _mission_identifier(item, "mission {}".format(name), _WORK_ITEM_ID_RE)
        if bool(self.parent_id) != bool(self.return_to_id):
            raise ValueError(
                "a side work item requires both parent_id and return_to_id",
            )

        # The kernel completes a mission only once every work item is closed and
        # the root closed as fixed. Both halves are provable from this
        # projection alone, so a receipt can never claim a completed mission
        # while also describing work that is still open - or a main axis that
        # was deferred or blocked.
        if self.mission_status == MISSION_COMPLETED:
            if self.status != MISSION_STATUS_CLOSED:
                raise ValueError(
                    "a completed mission has no open work items",
                )
            if self.is_root and self.disposition != MISSION_DISPOSITION_FIXED:
                raise ValueError(
                    "a mission completes only on a fixed main work item",
                )

        if not isinstance(self.returned_to_main_axis, bool):
            raise ValueError("mission returned_to_main_axis must be a boolean")
        if self.returned_to_main_axis:
            if self.is_root:
                raise ValueError(
                    "a root work item is already on the main axis",
                )
            if self.status != MISSION_STATUS_CLOSED:
                raise ValueError(
                    "control returns to the main axis only when the side item closes",
                )

    @classmethod
    def from_mapping(cls, value: Any) -> "CodingMissionProjection":
        if not isinstance(value, Mapping):
            raise ValueError("mission must be a JSON object")
        unknown = set(value) - MISSION_PROJECTION_FIELDS
        if unknown:
            raise ValueError("unsupported mission fields: {}".format(
                ", ".join(sorted(str(name) for name in unknown)),
            ))
        identifiers = value.get("criteria_ids", ())
        if isinstance(identifiers, (str, bytes)) or not isinstance(identifiers, Sequence):
            raise ValueError("mission criteria_ids must be a JSON array")
        if len(identifiers) > MISSION_MAX_ACCEPTANCE_CRITERIA:
            raise ValueError("mission criteria_ids cannot exceed {} items".format(
                MISSION_MAX_ACCEPTANCE_CRITERIA,
            ))
        return cls(
            mission_id=_mapping_string(value, "mission_id", ""),
            scope=_mapping_string(value, "scope", ""),
            work_item_id=_mapping_string(value, "work_item_id", ""),
            mission_status=_mapping_string(value, "mission_status", MISSION_OPEN),
            main_axis_sha256=_mapping_string(value, "main_axis_sha256", ""),
            criteria_ids=_require_string_array(identifiers, "mission criteria_ids"),
            lane=_mapping_string(value, "lane", ""),
            priority=_mapping_int(value, "priority", 0),
            status=_mapping_string(value, "status", ""),
            disposition=_mapping_string(value, "disposition", ""),
            parent_id=_mapping_string(value, "parent_id", ""),
            return_to_id=_mapping_string(value, "return_to_id", ""),
            returned_to_main_axis=_mapping_bool(value, "returned_to_main_axis", False),
        )

    def to_mapping(self) -> Dict[str, Any]:
        """Canonical JSON projection; exactly `MISSION_PROJECTION_FIELDS`."""

        return {
            "mission_id": self.mission_id,
            "scope": self.scope,
            "work_item_id": self.work_item_id,
            "mission_status": self.mission_status,
            "main_axis_sha256": self.main_axis_sha256,
            "criteria_ids": list(self.criteria_ids),
            "lane": self.lane,
            "priority": self.priority,
            "status": self.status,
            "disposition": self.disposition,
            "parent_id": self.parent_id,
            "return_to_id": self.return_to_id,
            "returned_to_main_axis": self.returned_to_main_axis,
        }


@dataclass
class CodingTaskRequest:
    """Provider-neutral request used by every coding backend."""

    message: str
    working_dir: str
    #: Repositories this job may touch, as an atomic lease set.  Empty means
    #: the host derives the nearest Git boundary from ``working_dir`` before
    #: admission; it never means "all configured workspaces".
    repository_roots: Tuple[str, ...] = ()
    #: Optional safe label for the submitting client/task.  Observability only:
    #: it grants no authority and is never put into an implementer prompt.
    owner_ref: Optional[str] = None
    thread_id: Optional[str] = None
    resume: bool = False
    #: Explicitly repeat one audited repair round whose host-owned route failed
    #: before the provider began.  This is an action on the existing same-key
    #: job, never provider input and never permission to create a fresh job.
    retry_rework_route: bool = False
    approval_policy: ApprovalPolicy = ApprovalPolicy.NEVER
    sandbox_mode: SandboxMode = SandboxMode.WORKSPACE_WRITE
    checks: Tuple[CheckSpec, ...] = ()
    capabilities: Tuple[CapabilitySpec, ...] = ()
    max_attempts: int = 3
    # Keep the public coding request aligned with the Claude adapter's bounded
    # default. A transport-level 30 here silently overrode the adapter's 100
    # and stranded complete implementations as turn_limit_exceeded.
    max_rounds: int = 100
    require_changes: bool = True
    config_path: str = ".flyto/coding.yaml"
    command_sandbox_image: str = "python:3.12-slim"
    #: SHA-256 of the repository contract this job was authorized against,
    #: established once by the service at submit and reapplied from startup
    #: authority on every rework round. A remote payload can never set it - the
    #: service overwrites it - so a model that edits `.flyto/coding.yaml`
    #: mid-job cannot have the edit authorize itself in a later round.
    authorized_config_sha256: str = ""
    #: The contract this round must execute, by value, captured by the host from
    #: the exact bytes preflight validated before the first provider edit. A
    #: remote payload can never set it - the service overwrites it - and when it
    #: is present the implementer executes it instead of re-reading the file, so
    #: an implementation that edits its own `.flyto/coding.yaml` still faces the
    #: checks it started under rather than becoming impossible to finish.
    pinned_contract: Optional["ContractSnapshot"] = None
    #: The mission this job serves, when the caller named one. Optional and
    #: additive: a request without it is exactly the request it always was, and
    #: nothing in this slice reads it to schedule, dispatch, or close anything.
    mission: Optional["CodingMissionEnvelope"] = None

    def __post_init__(self) -> None:
        self.message = self.message.strip()
        if not self.message or len(self.message) > 200_000:
            raise ValueError("message must contain between 1 and 200000 characters")
        root = Path(self.working_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("working_dir must be an existing directory")
        self.working_dir = str(root)
        if len(self.repository_roots) > 16:
            raise ValueError("repository_roots cannot exceed 16 items")
        repositories = []
        for value in self.repository_roots:
            if not isinstance(value, str):
                raise ValueError("repository_roots must contain paths")
            repository = Path(value).expanduser().resolve()
            if not repository.is_dir():
                raise ValueError("repository_roots must contain existing directories")
            repositories.append(str(repository))
        if len(set(repositories)) != len(repositories):
            raise ValueError("repository_roots must not contain duplicates")
        self.repository_roots = tuple(repositories)
        if self.owner_ref is not None and not _NAME_RE.fullmatch(self.owner_ref):
            raise ValueError("owner_ref must be a safe identifier")
        if self.thread_id is not None and not _NAME_RE.fullmatch(self.thread_id):
            raise ValueError("thread_id must be a safe identifier")
        if self.resume and not self.thread_id:
            raise ValueError("resume requires thread_id")
        if not isinstance(self.retry_rework_route, bool):
            raise ValueError("retry_rework_route must be a boolean")
        if self.retry_rework_route and self.resume is not True:
            raise ValueError("retry_rework_route requires resume")
        if not isinstance(self.authorized_config_sha256, str) or (
            self.authorized_config_sha256
            and not _CONFIG_DIGEST_RE.fullmatch(self.authorized_config_sha256)
        ):
            raise ValueError("authorized_config_sha256 must be a sha256 digest or empty")
        if self.pinned_contract is not None:
            if not isinstance(self.pinned_contract, ContractSnapshot):
                # A raw mapping is refused rather than coerced, for the same
                # reason the mission envelope is: decoding is the host's job and
                # it applies a closed, revalidating field set.
                raise ValueError("pinned_contract must be a ContractSnapshot")
            if (
                self.authorized_config_sha256
                and self.pinned_contract.config_sha256 != self.authorized_config_sha256
            ):
                # The two must name one document. A request that pinned one
                # contract and claimed authority over another would let an
                # auditor and an implementer describe different bytes.
                raise ValueError(
                    "pinned_contract must match authorized_config_sha256",
                )
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
        if self.mission is not None and not isinstance(self.mission, CodingMissionEnvelope):
            # A raw mapping is refused rather than coerced: decoding is the
            # transport's job, and it applies a closed field set this
            # constructor would silently skip.
            raise ValueError("mission must be a CodingMissionEnvelope")
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


def safe_blockers(values: Any) -> Tuple[str, ...]:
    """Bound and validate check names for a durable or public projection.

    Refuses rather than truncates. A value that is not already a safe contract
    identifier is dropped entirely, so a path, an argv fragment, a message or an
    environment value cannot arrive here shortened into something that looks
    like a name. Deduplicated and ordered so two equivalent failures project
    identically.
    """

    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, set)):
        return ()
    kept = []
    for value in values:
        if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
            continue
        if value not in kept:
            kept.append(value)
    return tuple(sorted(kept)[:MAX_VERIFICATION_BLOCKERS])


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
    #: When a host lane rewrites this result, the failure code the implementer
    #: itself reported. `failure_code` then names the lane that refused the
    #: round, which is a different fact. Empty for an unwrapped result, so a
    #: caller that never saw a lane is unaffected.
    implementation_failure_code: str = ""
    #: Which declared checks could not be launched, by name. Contract
    #: identifiers only - never a path, an argv or a message - so the reason a
    #: round could not be verified can cross a public boundary as data.
    verification_blockers: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.verification_blockers = safe_blockers(self.verification_blockers)


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
    #: Stable host-derived reasons this implementation is real, attributable,
    #: and still not landable. A non-empty list means "audited rework is the
    #: only way forward": the auditor may order rework, but may not accept.
    #: Empty for every ordinary round, so a pre-blocker caller is unaffected.
    implementation_blockers: Tuple[str, ...] = ()
    #: Declared checks that could not be launched for this job, by name. Safe
    #: contract identifiers only; empty unless verification could not run.
    verification_blockers: Tuple[str, ...] = ()
    #: Which continuation segment this job is, counted by the host. `0` for
    #: every ordinary job. Non-zero says only "this session has been carried
    #: forward N times" — never where the authority is, what it binds, or which
    #: workspace path it names.
    continuation_generation: int = 0
    #: Whether an unconsumed continuation authority exists for this job's own
    #: session, in this job's own tenant. A bounded, typed answer to "is there
    #: anything left to resume", and the only other continuation state a caller
    #: sees. Never true for a job whose session was never established.
    continuation_available: bool = False
    #: Secret-free proof of which host-owned lanes ran around the round.
    #: Absent for a legacy direct-library service that has no strict route.
    route_receipt: Optional[Dict[str, Any]] = None
    #: Digest-validated proof that this round ran on the host-owned emergency
    #: overflow lane instead of the strict route. Present only for a job the
    #: breaker actually diverted; never a substitute for a strict receipt.
    emergency_authority: Optional[Dict[str, Any]] = None
    #: Bounded, secret-safe mission coordinates for this job, when it serves a
    #: mission. Stored as the canonical projection mapping - like
    #: `route_receipt` - so the durable record and the public body are the same
    #: closed field set, revalidated on every construction.
    mission: Optional[Dict[str, Any]] = None

    @property
    def job_terminal(self) -> bool:
        """Whether this *job* will never change state again without a new job.

        Deliberately derived from `state` rather than stored, so it cannot
        drift from the state machine, and deliberately read from the single
        existing `TERMINAL_CODING_JOB_STATES` vocabulary rather than a second
        list that would have to be kept in step.

        This is about the job, not about the service process. A live instance
        can hold nothing but terminal jobs, and a closing instance can still
        own a job that is merely awaiting audit; conflating the two is how a
        settled job came to look active in runtime status.
        """
        return self.state in TERMINAL_CODING_JOB_STATES

    @property
    def failure_semantics(self) -> Tuple[str, bool, Tuple[str, ...]]:
        """Phase, retryability and required actions for this job's outcome.

        Derived from `failure_code` for the same reason `job_terminal` is
        derived from `state`: a stored copy is a second source of truth that
        will eventually disagree. A job that has not failed reports no phase,
        is not retryable, and needs nothing.

        An unrecognized code is reported conservatively - service phase, not
        retryable, no action - rather than guessed into a friendlier category.
        Guessing is how a terminal billing failure becomes a retry loop.
        """

        code = self.failure_code or ""
        if not code:
            return "", False, ()
        if code in JOB_FAILURE_SEMANTICS:
            return JOB_FAILURE_SEMANTICS[code]

        # A strict route rewrites `failure_code` to name the lane that refused
        # the round (`route_implementation_not_successful`) and carries the
        # implementer's own classification alongside it. Reading only the outer
        # code reported every wrapped provider failure as generic, so a caller
        # holding a receipt for an exhausted quota was told "service, not
        # retryable, nothing to do" - true of the lane, useless about the
        # cause.
        #
        # The inner code is consulted only when it is already in the closed
        # table. It is a host-owned classification produced from a fixed marker
        # allowlist, never model prose, and an unrecognized value falls through
        # to the same conservative answer as before rather than being trusted.
        wrapped = ""
        result = self.result
        if result is not None:
            wrapped = str(getattr(result, "implementation_failure_code", "") or "")
        if wrapped in JOB_FAILURE_SEMANTICS:
            return JOB_FAILURE_SEMANTICS[wrapped]
        return FAILURE_PHASE_SERVICE, False, ()

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
        blockers = self.implementation_blockers
        if isinstance(blockers, (str, bytes)) or not isinstance(blockers, Sequence):
            raise ValueError("implementation_blockers must be a JSON array")
        blockers = tuple(blockers)
        if len(blockers) > MAX_IMPLEMENTATION_BLOCKERS:
            raise ValueError(
                "implementation_blockers cannot exceed {} items".format(
                    MAX_IMPLEMENTATION_BLOCKERS,
                ),
            )
        if any(
            isinstance(item, bool)
            or not isinstance(item, str)
            or not _AUDIT_CODE_RE.fullmatch(item)
            for item in blockers
        ):
            raise ValueError("implementation_blockers must be stable safe identifiers")
        if len(set(blockers)) != len(blockers):
            raise ValueError("implementation_blockers contains duplicates")
        object.__setattr__(self, "implementation_blockers", blockers)
        # An unresolved blocker and an accepted, landable revision are directly
        # contradictory claims. A serialized receipt cannot make both.
        if blockers and (self.landable or self.state is CodingJobState.CODEX_ACCEPTED):
            raise ValueError(
                "an accepted receipt cannot carry unresolved implementation blockers",
            )
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
        if self.mission is not None:
            if not isinstance(self.mission, Mapping):
                raise ValueError("mission must be a JSON object")
            # Revalidate and re-canonicalize, so a hand-built or tampered record
            # cannot smuggle an extra key - or a word of prose - into a receipt
            # by presenting it as an already-projected mission.
            object.__setattr__(
                self, "mission", CodingMissionProjection.from_mapping(self.mission).to_mapping(),
            )
