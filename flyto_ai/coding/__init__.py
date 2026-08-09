# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Provider-neutral, detachable coding-agent control plane."""

from importlib import import_module

from flyto_ai.coding.agent import FlytoCodingAgent
from flyto_ai.coding.capabilities import CapabilityManager, McpStdioSession
from flyto_ai.coding.checks import CheckRunner, load_project_config
from flyto_ai.coding.conformance import (
    AdapterConformanceCase,
    AdapterConformanceReport,
    run_adapter_conformance,
)
from flyto_ai.coding.contracts import (
    AUDIT_BOUND_CODING_JOB_STATES,
    AUDITED_CODING_JOB_STATES,
    CONFIG_VERSION,
    CONTRACT_VERSION,
    MAX_AUDIT_EVIDENCE_REF_CHARS,
    MAX_AUDIT_FINDINGS,
    MAX_AUDIT_MESSAGE_CHARS,
    MAX_AUDIT_ROUNDS,
    SERVICE_CONTRACT_VERSION,
    SUPPORTED_SERVICE_CONTRACT_VERSIONS,
    TERMINAL_CODING_JOB_STATES,
    TOOL_PERMISSION_LEVELS,
    ApprovalPolicy,
    CapabilitySpec,
    CapabilityStatus,
    CheckResult,
    CheckSpec,
    CodingAuditFinding,
    CodingAuditSeverity,
    CodingAuditVerdict,
    CodingJobReceipt,
    CodingJobState,
    CodingTaskRequest,
    CodingTaskResult,
    SandboxMode,
    audit_findings_sha256,
    require_revision_sha256,
    validate_audit_submission,
)
# Audit callers must be able to handle every fail-closed decision by type.
# `CodingJobNotFound` and `WorkspaceDenied` are included because `audit()`
# itself raises them; job submission internals stay unexported.
from flyto_ai.coding.service import (
    AuditNotEnabled,
    AuditStateConflict,
    CodingJobNotFound,
    CodingServiceError,
    CodingServiceReloadRequired,
    RevisionMismatch,
    RevisionUnavailable,
    ReworkLimitReached,
    ReworkNotResumable,
    SessionBindingFailed,
    WorkspaceDenied,
    receipt_to_mapping,
)
from flyto_ai.coding.store import ThreadStore
from flyto_ai.coding.execution_policy import (
    ApprovalDecision,
    ApprovalRequest,
    ExecutionLimits,
    ExecutionPolicy,
    ExecutionPolicyController,
)
from flyto_ai.coding.execution_trace import (
    ExecutionReplayReport,
    ExecutionTraceLedger,
    OutcomeFeedbackReceipt,
)
from flyto_ai.coding.scenario_matrix import (
    AdapterScenario,
    ScenarioMatrixReport,
    run_scenario_matrix,
)
from flyto_ai.coding.workspace import WorkspaceTools, WorkspaceViolation

_STACK_EXPORTS = frozenset({
    "AGENT_STACK_CONTRACT_VERSION",
    "AGENT_STACK_POLICY_VERSION",
    "AgentStackManifest",
    "DEFAULT_COMPONENTS",
    "SUPPORTED_AGENT_STACK_MANIFEST_VERSIONS",
    "build_agent_stack_capabilities",
    "compose_capability_stack",
    "load_agent_stack_manifest",
    "probe_capability_stack",
    "probe_agent_stack",
})


def __getattr__(name: str):
    """Load stack helpers lazily so the stack module remains warning-free as a CLI."""
    if name in _STACK_EXPORTS:
        return getattr(import_module("flyto_ai.coding.stack"), name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))

__all__ = [
    "AdapterConformanceCase",
    "AdapterConformanceReport",
    "AdapterScenario",
    "ApprovalPolicy",
    "ApprovalDecision",
    "ApprovalRequest",
    "AGENT_STACK_CONTRACT_VERSION",
    "AGENT_STACK_POLICY_VERSION",
    "AgentStackManifest",
    "AUDIT_BOUND_CODING_JOB_STATES",
    "AUDITED_CODING_JOB_STATES",
    "AuditNotEnabled",
    "AuditStateConflict",
    "CapabilityManager",
    "CapabilitySpec",
    "CapabilityStatus",
    "CheckResult",
    "CheckRunner",
    "CheckSpec",
    "CodingAuditFinding",
    "CodingAuditSeverity",
    "CodingAuditVerdict",
    "CodingJobNotFound",
    "CodingJobReceipt",
    "CodingJobState",
    "CodingServiceError",
    "CodingServiceReloadRequired",
    "CodingTaskRequest",
    "CodingTaskResult",
    "CONFIG_VERSION",
    "CONTRACT_VERSION",
    "DEFAULT_COMPONENTS",
    "ExecutionLimits",
    "ExecutionPolicy",
    "ExecutionPolicyController",
    "ExecutionReplayReport",
    "ExecutionTraceLedger",
    "FlytoCodingAgent",
    "MAX_AUDIT_EVIDENCE_REF_CHARS",
    "MAX_AUDIT_FINDINGS",
    "MAX_AUDIT_MESSAGE_CHARS",
    "MAX_AUDIT_ROUNDS",
    "McpStdioSession",
    "OutcomeFeedbackReceipt",
    "RevisionMismatch",
    "RevisionUnavailable",
    "ReworkLimitReached",
    "ReworkNotResumable",
    "SandboxMode",
    "ScenarioMatrixReport",
    "SERVICE_CONTRACT_VERSION",
    "SessionBindingFailed",
    "SUPPORTED_AGENT_STACK_MANIFEST_VERSIONS",
    "SUPPORTED_SERVICE_CONTRACT_VERSIONS",
    "TERMINAL_CODING_JOB_STATES",
    "ThreadStore",
    "TOOL_PERMISSION_LEVELS",
    "WorkspaceDenied",
    "WorkspaceTools",
    "WorkspaceViolation",
    "audit_findings_sha256",
    "build_agent_stack_capabilities",
    "compose_capability_stack",
    "load_project_config",
    "load_agent_stack_manifest",
    "probe_capability_stack",
    "probe_agent_stack",
    "receipt_to_mapping",
    "require_revision_sha256",
    "run_adapter_conformance",
    "run_scenario_matrix",
    "validate_audit_submission",
]
