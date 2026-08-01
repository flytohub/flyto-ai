"""Contract tests for the real native-coding benchmark harness."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_native_coding.py"
SPEC = importlib.util.spec_from_file_location("benchmark_native_coding", SCRIPT)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_tier_sequence_distributes_101_cases_without_loss():
    tiers = benchmark.tier_sequence(101)

    assert len(tiers) == 101
    assert tiers.count("standard") == 34
    assert tiers.count("intermediate") == 34
    assert tiers.count("advanced") == 33


def test_case_specs_are_distinct_and_increase_depth():
    standard = benchmark.case_spec(1, "standard")
    intermediate = benchmark.case_spec(35, "intermediate")
    advanced = benchmark.case_spec(69, "advanced")

    assert standard["depth"] < intermediate["depth"] < advanced["depth"]
    assert standard["allowed_changes"] == ["logic.py"]
    assert intermediate["allowed_changes"] == ["service.py"]
    assert advanced["allowed_changes"] == ["records/service.py"]
    assert len({standard["case_id"], intermediate["case_id"], advanced["case_id"]}) == 3
    for item in (standard, intermediate, advanced):
        assert any(path.startswith("test") for path in item["files"])


def _case(case_id, tier, success=True, attempts=1):
    return {
        "case_id": case_id,
        "tier": tier,
        "success": success,
        "attempts": attempts,
        "hidden_retries": 0,
    }


def test_summary_requires_count_distinctness_tiers_and_single_attempts():
    cases = []
    for index, tier in enumerate(benchmark.tier_sequence(101), start=1):
        cases.append(_case(f"case-{index}", tier))

    passing = benchmark.summarize(cases, 0.90)
    assert passing["gate_pass"] is True
    assert passing["success_rate"] == 1.0

    duplicate = [dict(item) for item in cases]
    duplicate[-1]["case_id"] = duplicate[0]["case_id"]
    assert benchmark.summarize(duplicate, 0.90)["gate_pass"] is False

    repaired = [dict(item) for item in cases]
    repaired[0]["attempts"] = 2
    assert benchmark.summarize(repaired, 0.90)["gate_pass"] is True

    hidden_retry = [dict(item) for item in cases]
    hidden_retry[0]["hidden_retries"] = 1
    assert benchmark.summarize(hidden_retry, 0.90)["gate_pass"] is False

    weak_tier = [dict(item) for item in cases]
    advanced_seen = 0
    for item in weak_tier:
        if item["tier"] == "advanced" and advanced_seen < 4:
            item["success"] = False
            advanced_seen += 1
    assert benchmark.summarize(weak_tier, 0.90)["gate_pass"] is False


def test_canonical_digest_is_order_independent_and_change_sensitive():
    assert benchmark.canonical_digest({"a": 1, "b": 2}) == benchmark.canonical_digest(
        {"b": 2, "a": 1}
    )
    assert benchmark.canonical_digest({"a": 1}) != benchmark.canonical_digest({"a": 2})


def test_benchmark_thread_ids_are_fresh_for_interrupted_resume():
    first = benchmark.benchmark_thread_id("native-standard-001")
    second = benchmark.benchmark_thread_id("native-standard-001")

    assert first.startswith("native_standard_001_")
    assert second.startswith("native_standard_001_")
    assert first != second


def test_benchmark_settings_bind_transport_thinking_repair_and_token_budgets():
    settings = benchmark.benchmark_settings(
        Namespace(max_agent_attempts=3, max_tokens=4096)
    )

    assert settings == {
        "max_agent_attempts": 3,
        "max_completion_tokens": 4096,
        "transport": "native-/api/chat",
        "think": False,
    }
