# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Provider-neutral, detachable coding-agent control plane."""

from importlib import import_module

from flyto_ai.coding.agent import FlytoCodingAgent
from flyto_ai.coding.capabilities import CapabilityManager, McpStdioSession
from flyto_ai.coding.checks import CheckRunner, load_project_config
from flyto_ai.coding.contracts import (
    CONFIG_VERSION,
    CONTRACT_VERSION,
    TOOL_PERMISSION_LEVELS,
    ApprovalPolicy,
    CapabilitySpec,
    CapabilityStatus,
    CheckResult,
    CheckSpec,
    CodingTaskRequest,
    CodingTaskResult,
    SandboxMode,
)
from flyto_ai.coding.store import ThreadStore
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
    "ApprovalPolicy",
    "AGENT_STACK_CONTRACT_VERSION",
    "AGENT_STACK_POLICY_VERSION",
    "AgentStackManifest",
    "CapabilityManager",
    "CapabilitySpec",
    "CapabilityStatus",
    "CheckResult",
    "CheckRunner",
    "CheckSpec",
    "CodingTaskRequest",
    "CodingTaskResult",
    "CONFIG_VERSION",
    "CONTRACT_VERSION",
    "DEFAULT_COMPONENTS",
    "FlytoCodingAgent",
    "McpStdioSession",
    "SandboxMode",
    "SUPPORTED_AGENT_STACK_MANIFEST_VERSIONS",
    "ThreadStore",
    "TOOL_PERMISSION_LEVELS",
    "WorkspaceTools",
    "WorkspaceViolation",
    "build_agent_stack_capabilities",
    "compose_capability_stack",
    "load_project_config",
    "load_agent_stack_manifest",
    "probe_capability_stack",
    "probe_agent_stack",
]
