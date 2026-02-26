# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Data models for Claude Code Agent orchestration."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CodeTaskRequest:
    """User request for the Claude Code Agent."""
    message: str
    working_dir: str
    verification_recipe: Optional[str] = None
    verification_args: Dict[str, Any] = field(default_factory=dict)
    reference_image: Optional[str] = None
    max_fix_attempts: int = 3
    max_budget_usd: float = 5.0
    max_turns: int = 30


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
