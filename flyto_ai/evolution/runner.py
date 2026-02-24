# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Eval runner — run eval cases against prompt candidates.

Supports two execution modes:
  1. Mock mode (default) — uses predetermined tool results from eval cases.
     Fast, deterministic, no API key needed for tool dispatch.
  2. Live mode — runs through the real agent with actual tool dispatch.
     Requires API key and flyto-core installed.

Both modes require an LLM API key for the chat response generation.
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from flyto_ai.evolution.models import (
    CandidateScore, EvalCase, EvolutionConfig, PromptCandidate, ScoreBreakdown,
)
from flyto_ai.evolution.scorer import score_response, score_with_llm_judge

logger = logging.getLogger(__name__)

# Default eval cases directory
_EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "eval"


def load_eval_cases(path: Optional[str] = None) -> List[EvalCase]:
    """Load eval cases from a YAML file.

    Default path: <project_root>/eval/cases.yaml
    """
    if path is None:
        path = str(_EVAL_DIR / "cases.yaml")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("Eval cases not found: {}".format(path))

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cases = []
    for item in data.get("cases", []):
        cases.append(EvalCase(**item))

    return cases


def load_rubric(path: Optional[str] = None) -> EvolutionConfig:
    """Load scoring rubric/config from YAML. Falls back to defaults."""
    if path is None:
        path = str(_EVAL_DIR / "rubric.yaml")

    p = Path(path)
    if not p.exists():
        return EvolutionConfig()

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return EvolutionConfig(**data)


async def eval_candidate(
    candidate: PromptCandidate,
    cases: List[EvalCase],
    config: Optional[EvolutionConfig] = None,
    provider=None,
    subset: Optional[int] = None,
) -> CandidateScore:
    """Evaluate a prompt candidate against all eval cases.

    Parameters
    ----------
    candidate : PromptCandidate
        The prompt to evaluate.
    cases : list of EvalCase
        Test cases to run.
    config : EvolutionConfig, optional
        Scoring configuration.
    provider : LLMProvider, optional
        LLM provider for generating responses. If None, uses mock responses.
    subset : int, optional
        Only run first N cases (for quick screening).
    """
    cfg = config or EvolutionConfig()
    eval_cases = cases[:subset] if subset else cases

    scores: List[ScoreBreakdown] = []
    system_prompt = candidate.render(module_count=cfg.module_count)

    for case in eval_cases:
        try:
            score = await _eval_single_case(
                case, system_prompt, cfg, provider,
            )
            scores.append(score)
        except Exception as e:
            logger.warning("Eval failed for case %s: %s", case.id, e)
            scores.append(ScoreBreakdown(
                case_id=case.id,
                task_score=0.0,
                compliance_score=0.0,
                ux_score=0.0,
                penalties=-10.0,
                total_score=0.0,
                notes=["eval error: {}".format(str(e)[:100])],
                passed=False,
            ))

    return _aggregate_scores(candidate.id, scores, cfg)


async def _eval_single_case(
    case: EvalCase,
    system_prompt: str,
    config: EvolutionConfig,
    provider=None,
) -> ScoreBreakdown:
    """Evaluate a single case — get LLM response, then score it."""
    execution_results = case.mock_execution_results
    tool_calls = case.mock_tool_calls

    if provider is not None:
        # Generate response using the LLM with mock tool dispatch
        response = await _generate_response(
            case, system_prompt, execution_results, tool_calls, provider, config,
        )
    else:
        # No provider — use a placeholder response for testing
        response = "(no LLM provider — scoring mock only)"

    # Score the response
    if config.use_llm_judge and provider is not None:
        return await score_with_llm_judge(
            case, response, execution_results, provider,
        )

    return score_response(case, response, execution_results, config)


async def _generate_response(
    case: EvalCase,
    system_prompt: str,
    execution_results: List[Dict[str, Any]],
    tool_calls: List[Dict[str, Any]],
    provider,
    config: EvolutionConfig,
) -> str:
    """Generate an LLM response for an eval case using mock tool dispatch.

    The mock dispatch returns predetermined results from the eval case,
    simulating what would happen with real tools.
    """
    messages = [{"role": "user", "content": case.user_input}]

    # Build mock dispatch that returns predetermined results
    mock_results = list(execution_results)
    call_index = [0]

    async def mock_dispatch(func_name: str, func_args: dict) -> dict:
        if func_name == "execute_module" and call_index[0] < len(mock_results):
            result = mock_results[call_index[0]]
            call_index[0] += 1
            return result
        # For non-execute tools, return generic success
        return {"ok": True, "data": "mock result for {}".format(func_name)}

    # Build mock tool defs from tool_calls
    tool_defs = _build_mock_tool_defs(tool_calls)

    try:
        content, _, _, _ = await provider.chat(
            messages, system_prompt, tool_defs, mock_dispatch,
            max_rounds=config.stability_runs * 5,
        )
        return content or ""
    except Exception as e:
        logger.warning("Response generation failed: %s", e)
        return "(generation failed: {})".format(str(e)[:100])


def _build_mock_tool_defs(tool_calls: List[Dict[str, Any]]) -> List[Dict]:
    """Build minimal tool definitions for mock dispatch."""
    # Always include the standard flyto-ai tools
    return [
        {
            "name": "search_modules",
            "description": "Search for flyto-core modules by keyword.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_module_info",
            "description": "Get module specification.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "module_id": {"type": "string"},
                },
                "required": ["module_id"],
            },
        },
        {
            "name": "execute_module",
            "description": "Execute a flyto-core module.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "module_id": {"type": "string"},
                    "params": {"type": "object"},
                    "context": {"type": "object"},
                },
                "required": ["module_id", "params"],
            },
        },
        {
            "name": "validate_params",
            "description": "Validate module parameters.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "module_id": {"type": "string"},
                    "params": {"type": "object"},
                },
                "required": ["module_id", "params"],
            },
        },
    ]


def _aggregate_scores(
    candidate_id: str,
    scores: List[ScoreBreakdown],
    config: EvolutionConfig,
) -> CandidateScore:
    """Aggregate individual case scores into a candidate score."""
    if not scores:
        return CandidateScore(candidate_id=candidate_id)

    n = len(scores)
    task_avg = sum(s.task_score for s in scores) / n
    compliance_avg = sum(s.compliance_score for s in scores) / n
    ux_avg = sum(s.ux_score for s in scores) / n
    penalty_total = sum(s.penalties for s in scores)
    pass_count = sum(1 for s in scores if s.passed)

    weighted_total = (
        config.task_weight * (task_avg / 5.0 * 100)
        + config.compliance_weight * (compliance_avg / 5.0 * 100)
        + config.ux_weight * (ux_avg / 5.0 * 100)
        + penalty_total
    )
    weighted_total = max(0.0, min(100.0, weighted_total))

    # Stability: variance of total scores across cases
    if n > 1:
        totals = [s.total_score for s in scores]
        mean = sum(totals) / n
        variance = sum((t - mean) ** 2 for t in totals) / n
        # Convert to 0-5 scale (lower variance = higher stability)
        stability = max(0.0, 5.0 - (variance ** 0.5) / 10.0)
    else:
        stability = 5.0

    return CandidateScore(
        candidate_id=candidate_id,
        scores=scores,
        task_avg=round(task_avg, 2),
        compliance_avg=round(compliance_avg, 2),
        ux_avg=round(ux_avg, 2),
        penalty_total=round(penalty_total, 2),
        weighted_total=round(weighted_total, 2),
        stability=round(stability, 2),
        pass_rate=round(pass_count / n, 2) if n else 0.0,
        eval_count=n,
    )


def format_score_report(
    candidate: PromptCandidate,
    score: CandidateScore,
) -> str:
    """Format a human-readable score report for a candidate."""
    lines = [
        "Candidate: {} (gen {}, {})".format(
            candidate.id, candidate.generation, candidate.mutation_type,
        ),
        "  Score: {:.1f}/100  Pass rate: {:.0%}  Stability: {:.1f}/5".format(
            score.weighted_total, score.pass_rate, score.stability,
        ),
        "  Task: {:.1f}/5  Compliance: {:.1f}/5  UX: {:.1f}/5  Penalties: {:.0f}".format(
            score.task_avg, score.compliance_avg, score.ux_avg, score.penalty_total,
        ),
    ]

    # Show failed cases
    failed = [s for s in score.scores if not s.passed]
    if failed:
        lines.append("  Failed cases ({}/{}):".format(len(failed), score.eval_count))
        for s in failed[:5]:
            notes_str = "; ".join(s.notes[:2]) if s.notes else ""
            lines.append("    - {} (task={:.1f}, comp={:.1f}) {}".format(
                s.case_id, s.task_score, s.compliance_score, notes_str,
            ))

    return "\n".join(lines)
