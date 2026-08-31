# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Provider-neutral coding control plane with lazy public exports."""
from __future__ import annotations

from importlib import import_module
from typing import Any


_GROUPS = {
    "flyto_ai.coding.agent": ("FlytoCodingAgent",),
    "flyto_ai.coding.capabilities": ("CapabilityManager", "McpStdioSession"),
    "flyto_ai.coding.checks": ("CheckRunner", "load_project_config"),
    "flyto_ai.coding.conformance": (
        "AdapterConformanceCase", "AdapterConformanceReport", "run_adapter_conformance",
    ),
    "flyto_ai.coding.contracts": (
        "AUDIT_BOUND_CODING_JOB_STATES", "AUDITED_CODING_JOB_STATES",
        "CONFIG_VERSION", "CONTRACT_VERSION", "MAX_AUDIT_EVIDENCE_REF_CHARS",
        "MAX_AUDIT_FINDINGS", "MAX_AUDIT_MESSAGE_CHARS", "MAX_AUDIT_ROUNDS",
        "MISSION_COMPLETED", "MISSION_DISPOSITION_FIXED", "MISSION_DISPOSITIONS",
        "MISSION_ID_PATTERN", "MISSION_OPEN", "MISSION_STATUS_CLOSED",
        "MISSION_STATUS_DISPATCHED", "MISSION_STATUS_READY", "MISSION_LANE_PRIMARY",
        "MISSION_LANES", "MISSION_MAX_ACCEPTANCE_CRITERIA",
        "MISSION_MAX_DEPENDENCIES", "MISSION_MAX_FIELD_CHARS", "MISSION_MAX_PRIORITY",
        "MISSION_MAX_TEXT_CHARS", "MISSION_PROJECTION_FIELDS", "MISSION_STATUSES",
        "MISSION_WORK_STATUSES", "SERVICE_CONTRACT_VERSION",
        "SUPPORTED_SERVICE_CONTRACT_VERSIONS", "TERMINAL_CODING_JOB_STATES",
        "TOOL_PERMISSION_LEVELS", "ApprovalPolicy", "CapabilitySpec",
        "CapabilityStatus", "CheckResult", "CheckSpec", "CodingAuditFinding",
        "CodingAuditSeverity", "CodingAuditVerdict", "CodingJobReceipt",
        "CodingJobState", "CodingMissionEnvelope", "CodingMissionProjection",
        "CodingTaskRequest", "CodingTaskResult", "SandboxMode", "WORK_ITEM_ID_PATTERN",
        "audit_findings_sha256", "mission_axis_sha256", "require_revision_sha256",
        "validate_audit_submission",
    ),
    "flyto_ai.coding.service": (
        "AbandonStateConflict", "AuditBlockersUnresolved", "AuditNotEnabled",
        "AuditStateConflict", "CodingAuthorityConflict", "CodingAuthorityUnavailable",
        "CodingJobNotFound", "CodingServiceError", "CodingServiceReloadRequired",
        "RevisionMismatch", "RevisionUnavailable", "ReworkLimitReached",
        "ReworkNotResumable", "SessionBindingFailed", "WorkspaceBusy",
        "WorkspaceClaimUnresolved", "WorkspaceDenied", "receipt_to_mapping",
    ),
    "flyto_ai.coding.store": ("ThreadStore",),
    "flyto_ai.coding.execution_policy": (
        "ApprovalDecision", "ApprovalRequest", "ExecutionLimits", "ExecutionPolicy",
        "ExecutionPolicyController",
    ),
    "flyto_ai.coding.execution_trace": (
        "ExecutionReplayReport", "ExecutionTraceLedger", "OutcomeFeedbackReceipt",
    ),
    "flyto_ai.coding.scenario_matrix": (
        "AdapterScenario", "ScenarioMatrixReport", "run_scenario_matrix",
    ),
    "flyto_ai.coding.workspace": ("WorkspaceTools", "WorkspaceViolation"),
    "flyto_ai.coding.stack": (
        "AGENT_STACK_CONTRACT_VERSION", "AGENT_STACK_POLICY_VERSION",
        "AgentStackManifest", "DEFAULT_COMPONENTS",
        "SUPPORTED_AGENT_STACK_MANIFEST_VERSIONS", "build_agent_stack_capabilities",
        "compose_capability_stack", "load_agent_stack_manifest",
        "probe_capability_stack", "probe_agent_stack",
    ),
}
_EXPORTS = {
    name: module
    for module, names in _GROUPS.items()
    for name in names
}
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
    "AbandonStateConflict",
    "AuditBlockersUnresolved",
    "AuditNotEnabled",
    "AuditStateConflict",
    "CodingAuthorityConflict",
    "CodingAuthorityUnavailable",
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
    "CodingMissionEnvelope",
    "CodingMissionProjection",
    "mission_axis_sha256",
    "MISSION_COMPLETED",
    "MISSION_DISPOSITION_FIXED",
    "MISSION_DISPOSITIONS",
    "MISSION_ID_PATTERN",
    "MISSION_OPEN",
    "MISSION_STATUS_CLOSED",
    "MISSION_STATUS_DISPATCHED",
    "MISSION_STATUS_READY",
    "MISSION_LANE_PRIMARY",
    "MISSION_LANES",
    "MISSION_MAX_ACCEPTANCE_CRITERIA",
    "MISSION_MAX_DEPENDENCIES",
    "MISSION_MAX_FIELD_CHARS",
    "MISSION_MAX_PRIORITY",
    "MISSION_MAX_TEXT_CHARS",
    "MISSION_PROJECTION_FIELDS",
    "MISSION_STATUSES",
    "MISSION_WORK_STATUSES",
    "WORK_ITEM_ID_PATTERN",
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
    "WorkspaceBusy",
    "WorkspaceClaimUnresolved",
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


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
