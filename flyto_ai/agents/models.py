# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Compatibility models for the optional Claude SDK coding backend.

The provider-neutral public contracts live in :mod:`flyto_ai.coding`.  These
models intentionally remain small so existing integrations can keep selecting
the Claude SDK backend without making it the default Flyto coding runtime.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


MAX_SDK_SESSION_ID_CHARS = 128
#: An SDK session id is opaque. It is bounded and free of separators so it can
#: also be used as a durable thread identifier, but it is never parsed.
_SDK_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def is_safe_sdk_session_id(value: Any) -> bool:
    """Check one opaque Claude SDK session identity without interpreting it."""
    return (
        not isinstance(value, bool)
        and isinstance(value, str)
        and bool(_SDK_SESSION_ID_RE.fullmatch(value))
    )


@dataclass
class CodeTaskRequest:
    """User request for the optional Claude SDK backend."""
    message: str
    working_dir: str
    verification_recipe: Optional[str] = None
    verification_args: Dict[str, Any] = field(default_factory=dict)
    reference_image: Optional[str] = None
    max_fix_attempts: int = 3
    max_budget_usd: float = 5.0
    max_turns: int = 30
    # Internal service-mode state. Direct CLI and legacy callers never set
    # these, so every existing constructor keeps its exact behavior.
    sdk_session_id: Optional[str] = None
    service_mode: bool = False
    #: Whether the startup sandbox/approval authority permits model edits.
    #: Only the in-process adapter sets this; no remote payload can reach it.
    service_edit_authority: bool = True

    def __post_init__(self) -> None:
        if self.sdk_session_id is not None and not is_safe_sdk_session_id(self.sdk_session_id):
            raise ValueError("sdk_session_id must be a bounded opaque identifier")
        if not isinstance(self.service_mode, bool):
            raise ValueError("service_mode must be a boolean")
        if not isinstance(self.service_edit_authority, bool):
            raise ValueError("service_edit_authority must be a boolean")


@dataclass
class VerificationResult:
    """Outcome of a single verification run."""
    passed: bool
    recipe_name: str
    screenshot_path: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    comparison_summary: Optional[str] = None
    duration_ms: int = 0
    error: Optional[str] = None


@dataclass
class EvidenceRecord:
    """Single audit entry for the evidence trail."""
    timestamp: float
    phase: str          # "context" | "coding" | "verification" | "feedback"
    action: str         # "indexer_query" | "tool_approved" | "tool_denied" | ...
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeTaskResponse:
    """Final result returned by ClaudeCodeAgent.run()."""
    ok: bool
    message: str
    session_id: str
    attempts: int
    verification_results: List[VerificationResult] = field(default_factory=list)
    evidence: List[EvidenceRecord] = field(default_factory=list)
    files_changed: List[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    # Claude SDK return values
    claude_session_id: Optional[str] = None
    claude_num_turns: int = 0
    claude_duration_ms: int = 0
    claude_usage: Optional[Dict[str, Any]] = None
