# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Multi-layer scoring engine for prompt evaluation.

Three scoring rounds:
  A) Task Success — did the response accomplish the goal?
  B) Compliance — did it follow prompt rules?
  C) UX Quality — is it clear, concise, and useful?

Two modes:
  - Rule-based (default) — pattern matching, no API needed
  - LLM judge (optional) — uses a model to evaluate semantics
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

from flyto_ai.evolution.models import EvalCase, EvolutionConfig, ScoreBreakdown

logger = logging.getLogger(__name__)


def score_response(
    case: EvalCase,
    response: str,
    execution_results: List[Dict[str, Any]],
    config: Optional[EvolutionConfig] = None,
) -> ScoreBreakdown:
    """Score a response against an eval case using rule-based checks.

    Returns a ScoreBreakdown with task/compliance/ux scores + penalties.
    """
    cfg = config or EvolutionConfig()
    notes: List[str] = []
    penalties = 0.0

    task_score = _score_task(case, response, execution_results, notes)
    compliance_score = _score_compliance(case, response, execution_results, notes)
    ux_score = _score_ux(response, notes)

    # Apply penalties
    penalties += _check_hallucination(case, response, execution_results, cfg, notes)
    penalties += _check_forbidden(case, response, cfg, notes)

    total = (
        cfg.task_weight * (task_score / 5.0 * 100)
        + cfg.compliance_weight * (compliance_score / 5.0 * 100)
        + cfg.ux_weight * (ux_score / 5.0 * 100)
        + penalties
    )
    total = max(0.0, min(100.0, total))

    passed = task_score >= 2.0 and compliance_score >= 2.0 and penalties > -30

    return ScoreBreakdown(
        case_id=case.id,
        task_score=round(task_score, 2),
        compliance_score=round(compliance_score, 2),
        ux_score=round(ux_score, 2),
        penalties=round(penalties, 2),
        total_score=round(total, 2),
        notes=notes,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Round A: Task Success (0-5)
# ---------------------------------------------------------------------------

def _score_task(
    case: EvalCase,
    response: str,
    execution_results: List[Dict[str, Any]],
    notes: List[str],
) -> float:
    """Score task success based on expected behavior keywords."""
    if not case.expected_behavior:
        return 3.0  # neutral if no expected behavior defined

    score = 5.0
    keywords = _extract_check_phrases(case.expected_behavior)

    if not keywords:
        return 3.0

    matched = 0
    for kw in keywords:
        if _phrase_in_text(kw, response):
            matched += 1

    match_rate = matched / len(keywords) if keywords else 0

    # All executions failed but response claims success → 0
    all_failed = (
        execution_results
        and all(not r.get("ok", False) for r in execution_results)
    )
    if all_failed:
        success_patterns = ["成功", "完成", "已經", "找到", "搜尋結果", "以下是",
                            "successfully", "completed", "found", "here are"]
        claims_success = any(p in response for p in success_patterns)
        if claims_success:
            notes.append("task: claims success but all executions failed")
            return 0.0

    if match_rate >= 0.8:
        score = 5.0
    elif match_rate >= 0.6:
        score = 4.0
    elif match_rate >= 0.4:
        score = 3.0
    elif match_rate >= 0.2:
        score = 2.0
    else:
        score = 1.0
        notes.append("task: low keyword match ({:.0%})".format(match_rate))

    return score


# ---------------------------------------------------------------------------
# Round B: Compliance (0-5)
# ---------------------------------------------------------------------------

def _score_compliance(
    case: EvalCase,
    response: str,
    execution_results: List[Dict[str, Any]],
    notes: List[str],
) -> float:
    """Score rule compliance."""
    score = 5.0
    deductions = 0.0

    # Check: forbidden behavior patterns
    if case.forbidden_behavior:
        forbidden_kws = _extract_check_phrases(case.forbidden_behavior)
        for kw in forbidden_kws:
            if _phrase_in_text(kw, response):
                deductions += 1.5
                notes.append("compliance: forbidden pattern found: {}".format(kw[:30]))

    # Check: all executions failed → response must acknowledge error
    all_failed = (
        execution_results
        and all(not r.get("ok", False) for r in execution_results)
    )
    if all_failed:
        error_patterns = ["失敗", "錯誤", "無法", "error", "fail", "unable", "could not"]
        mentions_error = any(p in response.lower() for p in error_patterns)
        if not mentions_error:
            deductions += 2.0
            notes.append("compliance: no error acknowledgment when all tools failed")

    # Check: response language matches input language
    lang_mismatch = _check_language_mismatch(case.user_input, response)
    if lang_mismatch:
        deductions += 1.0
        notes.append("compliance: language mismatch ({})".format(lang_mismatch))

    score = max(0.0, score - deductions)
    return score


# ---------------------------------------------------------------------------
# Round C: UX Quality (0-5)
# ---------------------------------------------------------------------------

def _score_ux(response: str, notes: List[str]) -> float:
    """Score UX quality: clarity, conciseness, structure."""
    score = 5.0

    # Too short (< 20 chars) = probably useless
    if len(response.strip()) < 20:
        score -= 2.0
        notes.append("ux: response too short")

    # Too long (> 3000 chars) = probably verbose
    if len(response) > 3000:
        score -= 1.0
        notes.append("ux: response too long ({} chars)".format(len(response)))

    # Very long (> 5000 chars) = definitely too verbose
    if len(response) > 5000:
        score -= 1.0
        notes.append("ux: excessively long")

    # Has structure (headers, lists, code blocks) = good
    has_structure = bool(
        re.search(r'^[#\-\*\d]', response, re.MULTILINE)
        or '```' in response
    )
    if not has_structure and len(response) > 200:
        score -= 0.5
        notes.append("ux: no structure in long response")

    # Apology essays (over-apologizing)
    apology_patterns = ["i apologize", "i'm sorry", "抱歉", "對不起", "很遺憾"]
    apology_count = sum(1 for p in apology_patterns if p in response.lower())
    if apology_count >= 2:
        score -= 1.0
        notes.append("ux: excessive apologies")

    return max(0.0, score)


# ---------------------------------------------------------------------------
# Penalty checks
# ---------------------------------------------------------------------------

def _check_hallucination(
    case: EvalCase,
    response: str,
    execution_results: List[Dict[str, Any]],
    config: EvolutionConfig,
    notes: List[str],
) -> float:
    """Detect hallucinated results — claiming data that wasn't in tool results."""
    penalty = 0.0

    all_failed = (
        execution_results
        and all(not r.get("ok", False) for r in execution_results)
    )

    if all_failed:
        # Check if response contains fabricated URLs (not from any tool result)
        urls_in_response = re.findall(r'https?://[^\s\)]+', response)
        if urls_in_response:
            # Check if any URL came from tool results
            tool_text = json.dumps(execution_results, ensure_ascii=False, default=str)
            fabricated = [u for u in urls_in_response if u not in tool_text]
            if fabricated:
                penalty += config.penalty_hallucination
                notes.append("hallucination: fabricated {} URL(s)".format(len(fabricated)))

        # Check if response presents "search results" or structured data
        data_patterns = [
            r"標題\s*[:：]", r"連結\s*[:：]", r"title\s*:", r"link\s*:",
            r"搜尋結果", r"search results",
        ]
        for pat in data_patterns:
            if re.search(pat, response, re.IGNORECASE):
                penalty += config.penalty_hallucination
                notes.append("hallucination: fabricated data pattern")
                break

    return penalty


def _check_forbidden(
    case: EvalCase,
    response: str,
    config: EvolutionConfig,
    notes: List[str],
) -> float:
    """Apply rule violation penalties for forbidden behaviors."""
    if not case.forbidden_behavior:
        return 0.0

    penalty = 0.0
    forbidden_kws = _extract_check_phrases(case.forbidden_behavior)
    critical_violations = 0
    for kw in forbidden_kws:
        if _phrase_in_text(kw, response):
            critical_violations += 1

    if critical_violations > 0:
        penalty += config.penalty_rule_violation
        notes.append("forbidden: {} violation(s)".format(critical_violations))

    return penalty


# ---------------------------------------------------------------------------
# LLM Judge (optional, for deeper evaluation)
# ---------------------------------------------------------------------------

async def score_with_llm_judge(
    case: EvalCase,
    response: str,
    execution_results: List[Dict[str, Any]],
    provider,
) -> ScoreBreakdown:
    """Use an LLM as judge for semantic evaluation.

    Falls back to rule-based scoring if LLM call fails.
    """
    judge_prompt = _build_judge_prompt(case, response, execution_results)
    messages = [{"role": "user", "content": judge_prompt}]

    system = (
        "You are an impartial evaluation judge. Score the AI response precisely. "
        "Output ONLY valid JSON with no other text."
    )

    try:
        content, _, _, _ = await provider.chat(
            messages, system, tools=[], dispatch_fn=_noop_dispatch, max_rounds=1,
        )
        scores = _parse_judge_output(content)
        return ScoreBreakdown(
            case_id=case.id,
            task_score=scores.get("task", 3.0),
            compliance_score=scores.get("compliance", 3.0),
            ux_score=scores.get("ux", 3.0),
            penalties=scores.get("penalties", 0.0),
            total_score=scores.get("total", 50.0),
            notes=["llm_judge: {}".format(scores.get("notes", ""))],
            passed=scores.get("task", 0) >= 2 and scores.get("compliance", 0) >= 2,
        )
    except Exception as e:
        logger.warning("LLM judge failed, falling back to rule-based: %s", e)
        return score_response(case, response, execution_results)


async def _noop_dispatch(name: str, args: dict) -> dict:
    return {"ok": False, "error": "judge has no tools"}


def _build_judge_prompt(
    case: EvalCase,
    response: str,
    execution_results: List[Dict[str, Any]],
) -> str:
    exec_summary = json.dumps(execution_results[:5], ensure_ascii=False, default=str)[:500]
    return (
        "Evaluate the following AI response.\n\n"
        "## User Request\n{}\n\n"
        "## Expected Behavior\n{}\n\n"
        "## Forbidden Behavior\n{}\n\n"
        "## Tool Execution Results\n{}\n\n"
        "## AI Response\n{}\n\n"
        "## Scoring (0-5 each)\n"
        "1. task: Did it accomplish the goal? (0=wrong, 5=perfect)\n"
        "2. compliance: Did it follow rules? (0=violated, 5=perfect)\n"
        "3. ux: Is it clear and useful? (0=terrible, 5=excellent)\n"
        "4. penalties: Negative score for hallucination/fabrication (0 or negative)\n\n"
        "Output JSON: {{\"task\": N, \"compliance\": N, \"ux\": N, \"penalties\": N, \"notes\": \"...\"}}"
    ).format(
        case.user_input,
        case.expected_behavior,
        case.forbidden_behavior or "(none)",
        exec_summary,
        response[:1500],
    )


def _parse_judge_output(content: str) -> Dict[str, Any]:
    """Parse JSON from LLM judge output."""
    # Try to find JSON in the response
    m = re.search(r'\{[^}]+\}', content)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"task": 3.0, "compliance": 3.0, "ux": 3.0, "penalties": 0.0, "notes": "parse failed"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_check_phrases(text: str) -> List[str]:
    """Extract meaningful check keywords from expected/forbidden behavior text.

    Splits on commas, semicolons, '、', and spaces to get individual keywords.
    For CJK text, also splits long phrases into 2-3 char segments.
    """
    # Split by common delimiters
    parts = re.split(r'[,;、\n]', text)
    keywords = []
    for p in parts:
        p = p.strip().strip('-').strip('*').strip()
        if len(p) < 2:
            continue
        # For CJK phrases, extract key sub-phrases (2+ CJK chars)
        cjk_chars = [c for c in p if '\u4e00' <= c <= '\u9fff']
        if len(cjk_chars) >= 4:
            # Split long CJK phrases into overlapping 2-char segments
            cjk_str = ''.join(cjk_chars)
            for i in range(0, len(cjk_str) - 1, 2):
                seg = cjk_str[i:i+2]
                if seg not in keywords:
                    keywords.append(seg)
        elif len(cjk_chars) >= 2:
            keywords.append(''.join(cjk_chars))
        else:
            # ASCII: split by spaces and take significant words
            words = p.split()
            for w in words:
                w = w.strip('.,;:!?()"\'')
                if len(w) >= 3:
                    keywords.append(w)
    return keywords


def _phrase_in_text(phrase: str, text: str) -> bool:
    """Check if a phrase/keyword appears in text (case-insensitive for ASCII)."""
    if any('\u4e00' <= c <= '\u9fff' for c in phrase):
        return phrase in text
    return phrase.lower() in text.lower()


def _check_language_mismatch(user_input: str, response: str) -> str:
    """Check if response language matches user input language.

    Returns empty string if OK, or description of mismatch.
    """
    # Simple heuristic: if user input is mostly CJK, response should have CJK
    cjk_in_input = sum(1 for c in user_input if '\u4e00' <= c <= '\u9fff')
    cjk_ratio_input = cjk_in_input / max(len(user_input), 1)

    if cjk_ratio_input > 0.3:
        # User wrote Chinese → response should have Chinese
        # Exclude code blocks from language check
        text_only = re.sub(r'```[\s\S]*?```', '', response)
        cjk_in_text = sum(1 for c in text_only if '\u4e00' <= c <= '\u9fff')
        total_text = len(text_only.strip())
        if total_text > 20 and cjk_in_text / max(total_text, 1) < 0.05:
            return "Chinese input but response is mostly English"

    return ""
