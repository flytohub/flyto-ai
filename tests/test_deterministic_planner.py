"""Regression tests for benchmark-discovered deterministic routing gaps."""

import pytest

from flyto_ai.intelligence.planner import classify_tool_intent


@pytest.mark.parametrize(
    "message",
    [
        "Reuse a verified Blueprint to fetch an API response and save JSON.",
        "Convert the supplied CSV rows to JSON using a verified procedure.",
        "Import an unsigned community Blueprint and quarantine it.",
        "Reuse a high-scoring Blueprint from a different repository.",
    ],
)
def test_explicit_blueprint_and_transform_verbs_are_actions(message):
    decision = classify_tool_intent(message)

    assert decision.mode == "action"
    assert decision.tool_eligible is True
    assert "action_verb_en" in decision.signals


@pytest.mark.parametrize(
    "message",
    [
        "MCPもBlueprintも使わないでください。",
        "Blueprintを使用しないでください。",
        "ツールを使わないで説明してください。",
    ],
)
def test_japanese_no_tool_requests_expose_a_negation_signal(message):
    decision = classify_tool_intent(message)

    assert decision.mode == "answer_only"
    assert decision.tool_eligible is False
    assert decision.reason == "explicit_no_tool_request"
    assert decision.signals == ("no_tool",)
