# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Data models for the prompt evolution system."""
import hashlib
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """A single evaluation test case."""

    id: str
    category: str  # browser, automation, edge_case, language, failure
    user_input: str
    expected_behavior: str
    forbidden_behavior: str = ""
    golden_answer: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    weight: float = 1.0

    # Mock tool results — so eval can run without real API/browser
    mock_execution_results: List[Dict[str, Any]] = Field(default_factory=list)
    mock_tool_calls: List[Dict[str, Any]] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    """Detailed score for one eval case against one candidate."""

    case_id: str
    task_score: float = 0.0       # 0-5
    compliance_score: float = 0.0  # 0-5
    ux_score: float = 0.0         # 0-5
    penalties: float = 0.0
    total_score: float = 0.0
    notes: List[str] = Field(default_factory=list)
    passed: bool = True


class CandidateScore(BaseModel):
    """Aggregate score for a prompt candidate across all eval cases."""

    candidate_id: str
    scores: List[ScoreBreakdown] = Field(default_factory=list)
    task_avg: float = 0.0
    compliance_avg: float = 0.0
    ux_avg: float = 0.0
    penalty_total: float = 0.0
    weighted_total: float = 0.0
    stability: float = 5.0  # 0-5, higher = more stable
    pass_rate: float = 0.0
    eval_count: int = 0


class PromptBlock(BaseModel):
    """A modular prompt block (Prompt DNA unit)."""

    name: str
    content: str
    category: str  # policy, behavior, gate, override
    priority: int = 50  # ordering priority (lower = earlier)


class PromptCandidate(BaseModel):
    """A candidate system prompt for evaluation."""

    id: str
    parent_id: str = ""
    mutation_type: str = "baseline"
    mutation_desc: str = ""
    blocks: List[PromptBlock] = Field(default_factory=list)
    generation: int = 0

    def render(self, **kwargs) -> str:
        """Render the full prompt from blocks, ordered by priority."""
        sorted_blocks = sorted(self.blocks, key=lambda b: b.priority)
        parts = [b.content for b in sorted_blocks]
        prompt = "\n\n".join(parts)
        for k, v in kwargs.items():
            prompt = prompt.replace("{" + k + "}", str(v))
        return prompt

    def content_hash(self) -> str:
        """Short hash of the rendered content for dedup."""
        rendered = self.render()
        return hashlib.sha256(rendered.encode()).hexdigest()[:12]


class EvolutionConfig(BaseModel):
    """Configuration for the evolution loop."""

    population_size: int = 5
    generations: int = 3
    elite_count: int = 1
    novelty_count: int = 1

    # Scoring weights (must sum to 1.0)
    task_weight: float = 0.45
    compliance_weight: float = 0.35
    ux_weight: float = 0.20

    # Penalties (added as negative offsets to total)
    penalty_hallucination: float = -20.0
    penalty_rule_violation: float = -30.0
    penalty_format_error: float = -15.0
    penalty_task_failure: float = -50.0

    # Eval settings
    stability_runs: int = 1       # repeated runs per case
    use_llm_judge: bool = False
    judge_model: str = "gpt-4o-mini"
    quick_screen_subset: int = 10  # cases for round A

    # Stop criteria
    stop_no_improve: int = 2
    target_score: float = 90.0

    # Prompt rendering
    mode: str = "execute"
    module_count: int = 300


class GenerationResult(BaseModel):
    """Results from one generation of evolution."""

    generation: int
    timestamp: float = Field(default_factory=time.time)
    candidates: List[CandidateScore] = Field(default_factory=list)
    best_id: str = ""
    best_score: float = 0.0
    improvement: float = 0.0  # vs previous generation
    regressions: List[str] = Field(default_factory=list)


class EvolutionReport(BaseModel):
    """Full report from an evolution run."""

    config: EvolutionConfig
    generations: List[GenerationResult] = Field(default_factory=list)
    best_candidate_id: str = ""
    best_score: float = 0.0
    total_eval_runs: int = 0
    total_duration_s: float = 0.0
    regression_cases: List[str] = Field(default_factory=list)
