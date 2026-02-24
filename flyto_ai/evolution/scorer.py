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

def _check_success_claim(response: str, success_patterns: List[str],
                         negation_prefixes: List[str]) -> bool:
    """Check if response claims success, ignoring negated patterns.

    'Module not found' contains 'found' but is negated → not a success claim.
    """
    text_lower = response.lower()
    for p in success_patterns:
        p_lower = p if any('\u4e00' <= c <= '\u9fff' for c in p) else p.lower()
        if p_lower not in text_lower:
            continue
        idx = text_lower.index(p_lower)
        before = text_lower[max(0, idx - 5):idx]
        if any(neg in before for neg in negation_prefixes):
            continue  # negated usage, e.g. "not found"
        return True
    return False


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
        negation_prefixes = ["not ", "no ", "未", "沒", "不", "無法"]
        claims_success = _check_success_claim(response, success_patterns, negation_prefixes)
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

    # Check: forbidden behavior patterns (whole-phrase matching to reduce false positives)
    if case.forbidden_behavior:
        forbidden_phrases = _split_forbidden_phrases(case.forbidden_behavior)
        for phrase in forbidden_phrases:
            if _phrase_in_text(phrase, response):
                deductions += 1.5
                notes.append("compliance: forbidden pattern found: {}".format(phrase[:30]))

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
    forbidden_phrases = _split_forbidden_phrases(case.forbidden_behavior)
    critical_violations = 0
    for phrase in forbidden_phrases:
        if _phrase_in_text(phrase, response):
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

    Splits on commas, semicolons, '、', and newlines first,
    then extracts CJK clusters and ASCII words from each part.
    CJK clusters are groups of consecutive CJK characters — never crosses
    non-CJK boundaries (avoids producing gibberish like '從提' from '從 browser 提取').
    """
    parts = re.split(r'[,;、\n]', text)
    keywords = []
    for p in parts:
        p = p.strip().strip('-').strip('*').strip()
        if len(p) < 2:
            continue
        # Find CJK clusters (consecutive CJK characters)
        cjk_clusters = re.findall(r'[\u4e00-\u9fff]+', p)
        for cluster in cjk_clusters:
            if len(cluster) < 2:
                continue  # single CJK char, not useful
            if len(cluster) <= 3:
                # Short cluster (2-3 chars): keep whole
                if cluster not in keywords:
                    keywords.append(cluster)
            else:
                # Longer cluster: split into 2-char non-overlapping segments
                for i in range(0, len(cluster) - 1, 2):
                    seg = cluster[i:i + 2]
                    if seg not in keywords:
                        keywords.append(seg)
        # ASCII words (3+ chars)
        ascii_words = re.findall(r'[a-zA-Z0-9_.]+', p)
        for w in ascii_words:
            w = w.strip('.,;:!?()"\'')
            if len(w) >= 3 and w not in keywords:
                keywords.append(w)
    return keywords


def _split_forbidden_phrases(text: str) -> List[str]:
    """Split forbidden behavior text into check phrases.

    Always keeps phrases whole (comma-separated) to avoid false positives
    from matching common English words like "using", "module", "search".
    """
    parts = re.split(r'[,;\n]', text)
    phrases = []
    for p in parts:
        p = p.strip().strip('-').strip('*').strip()
        if len(p) < 3:
            continue
        phrases.append(p)
    return phrases


_NEGATION_WORDS = frozenset([
    "cannot", "can't", "not", "never", "refuse", "unable", "won't",
    "don't", "doesn't", "didn't", "shouldn't", "wouldn't",
])


def _phrase_in_text(phrase: str, text: str) -> bool:
    """Check if a phrase/keyword appears in text.

    For CJK text: exact substring match.
    For ASCII short phrases (1-2 words): exact substring match.
    For ASCII longer phrases (3+ words): fuzzy match — >=50% word overlap,
    BUT only if no negation word appears immediately before the first matched word.
    """
    if any('\u4e00' <= c <= '\u9fff' for c in phrase):
        return phrase in text

    phrase_lower = phrase.lower()
    text_lower = text.lower()

    # Exact substring match first
    if phrase_lower in text_lower:
        return True

    # For multi-word phrases, check word overlap (exclude stop words)
    _stop = {"the", "and", "for", "with", "from", "that", "this", "all", "any", "but"}
    words = [w for w in phrase_lower.split() if len(w) >= 3 and w not in _stop]
    if len(words) < 3:
        return False  # short phrases need exact match

    matched_words = [w for w in words if w in text_lower]
    if len(matched_words) / len(words) < 0.5:
        return False

    # The first word (typically the action verb) must be present
    if words[0] not in text_lower:
        return False

    # Negation check: if the first matched word is preceded by negation, skip
    first_match = matched_words[0]
    idx = text_lower.find(first_match)
    if idx > 0:
        before = text_lower[max(0, idx - 20):idx].strip()
        before_words = before.split()
        if before_words and before_words[-1] in _NEGATION_WORDS:
            return False

    return True


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
