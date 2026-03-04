# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for cost tracking and budget enforcement."""
import pytest

from flyto_ai.cost import (
    CostTracker,
    BudgetExceededError,
    estimate_cost,
    _match_model_cost,
)


# --- estimate_cost tests ---

def test_estimate_cost_known_model():
    cost = estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.15 + 0.60, abs=0.01)


def test_estimate_cost_prefix_match():
    # "claude-sonnet-4-5-20250929" should match "claude-sonnet-4-5" prefix
    cost = estimate_cost("claude-sonnet-4-5-20250929", 1000, 500)
    assert cost > 0


def test_estimate_cost_unknown_model():
    cost = estimate_cost("unknown-model-v99", 1000, 500)
    assert cost == 0.0


def test_estimate_cost_empty_model():
    cost = estimate_cost("", 1000, 500)
    assert cost == 0.0


def test_estimate_cost_cache_read():
    # Cache read at 10% of input rate
    cost_no_cache = estimate_cost("gpt-4o-mini", 1000, 500)
    cost_with_cache = estimate_cost("gpt-4o-mini", 1000, 500, cache_read_tokens=1000)
    assert cost_with_cache > cost_no_cache


def test_estimate_cost_local_model_free():
    cost = estimate_cost("llama3.2", 1_000_000, 1_000_000)
    assert cost == 0.0


# --- _match_model_cost tests ---

def test_match_exact():
    rate = _match_model_cost("gpt-4o")
    assert rate == (2.50, 10.0)


def test_match_prefix():
    rate = _match_model_cost("claude-sonnet-4-5-20250929")
    assert rate[0] > 0


def test_match_longest_prefix():
    # "claude-haiku-4-5" should match over "claude-haiku"
    rate = _match_model_cost("claude-haiku-4-5-20251001")
    assert rate == (0.80, 4.0)


# --- CostTracker tests ---

def test_tracker_basic_recording():
    tracker = CostTracker()
    record = tracker.record("gpt-4o-mini", "openai", 1000, 500)
    assert record.prompt_tokens == 1000
    assert record.completion_tokens == 500
    assert record.estimated_cost_usd > 0
    assert tracker.call_count == 1
    assert tracker.session_total_usd > 0


def test_tracker_accumulates():
    tracker = CostTracker()
    tracker.record("gpt-4o-mini", "openai", 1000, 500)
    tracker.record("gpt-4o-mini", "openai", 2000, 1000)
    assert tracker.call_count == 2
    assert tracker.total_prompt_tokens == 3000
    assert tracker.total_completion_tokens == 1500


def test_tracker_blueprint_zero_cost():
    tracker = CostTracker()
    record = tracker.record("gpt-4o-mini", "openai", 1000, 500, is_blueprint_replay=True)
    assert record.estimated_cost_usd == 0.0
    assert tracker.session_total_usd == 0.0
    assert tracker.blueprint_savings_usd > 0


def test_tracker_session_budget():
    tracker = CostTracker(session_budget_usd=0.0001)
    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.record("gpt-4o", "openai", 100_000, 50_000)
    assert exc_info.value.current > 0
    assert exc_info.value.limit == 0.0001


def test_tracker_global_budget():
    tracker = CostTracker(global_budget_usd=0.0001)
    with pytest.raises(BudgetExceededError):
        tracker.record("gpt-4o", "openai", 100_000, 50_000)


def test_tracker_no_budget_no_error():
    tracker = CostTracker()
    # Should not raise even with large usage
    tracker.record("gpt-4o", "openai", 1_000_000, 1_000_000)
    assert tracker.session_total_usd > 0


def test_tracker_reset_session():
    tracker = CostTracker()
    tracker.record("gpt-4o-mini", "openai", 1000, 500)
    assert tracker.session_total_usd > 0
    tracker.reset_session()
    assert tracker.session_total_usd == 0.0
    assert tracker.global_total_usd > 0  # global persists


def test_tracker_summary():
    tracker = CostTracker(session_budget_usd=10.0)
    tracker.record("gpt-4o-mini", "openai", 1000, 500)
    summary = tracker.summary()
    assert "session_total_usd" in summary
    assert "global_total_usd" in summary
    assert "blueprint_savings_usd" in summary
    assert "call_count" in summary
    assert summary["session_budget_usd"] == 10.0
