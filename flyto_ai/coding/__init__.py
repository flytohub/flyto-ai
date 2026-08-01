# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Provider-neutral, detachable coding-agent control plane."""

from flyto_ai.coding.agent import FlytoCodingAgent
from flyto_ai.coding.capabilities import CapabilityManager, McpStdioSession
from flyto_ai.coding.checks import CheckRunner, load_project_config
from flyto_ai.coding.contracts import (
    CONFIG_VERSION,
    CONTRACT_VERSION,
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

__all__ = [
    "ApprovalPolicy",
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
    "FlytoCodingAgent",
    "McpStdioSession",
    "SandboxMode",
    "ThreadStore",
    "WorkspaceTools",
    "WorkspaceViolation",
    "load_project_config",
]
