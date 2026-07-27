import hashlib
import json

import pytest

from flyto_ai.benchmarking import (
    BenchmarkHostError,
    BlueprintBenchmarkHost,
    HostConfig,
    MODES,
    PlannerObservation,
    build_environment_digest,
    write_runs,
)


SEALED_PROMPT = "Private no-tool routing challenge"


class FakePlanner:
    async def plan(self, prompt, *, context, seed):
        del prompt, context, seed
        return PlannerObservation(
            decision={
                "route": "conversation",
                "use_mcp": False,
                "use_blueprint": False,
                "execute_workflow": False,
                "negation_detected": True,
            },
            input_tokens=100,
            output_tokens=20,
        )


def _config(trials=20):
    return HostConfig.from_mapping(
        {
            "schema_version": "flyto-ai-blueprint-host.v1",
            "provider": {
                "type": "ollama",
                "base_url": "http://127.0.0.1:11434",
                "model": "flyto-qwen3:8b",
                "model_digest": "e737aff7b8d4",
                "temperature": 0,
                "max_output_tokens": 128,
                "timeout_seconds": 30,
                "think": False,
            },
            "run": {"trials": trials, "seed_base": 20260728},
            "compatibility": {
                "repository": "flytohub/flyto-blueprint",
                "runtime": "python3.11",
                "framework": "flyto2",
            },
        }
    )


def _suite():
    tasks = [
        {
            "id": "route-explicit-blueprint-request",
            "split": "public_eval",
            "sealed": False,
            "prompt": "Reuse a verified Blueprint to fetch an API response and save JSON.",
            "expected_behavior": "reuse",
            "assertions": [
                "router.selects_blueprint_path",
                "selection.compatibility_matches",
                "workflow.assertions_preserved",
            ],
            "tags": ["reuse"],
        },
        {
            "id": "route-data-transform-without-replanning",
            "split": "public_eval",
            "sealed": False,
            "prompt": "Convert CSV rows to JSON using a verified procedure.",
            "expected_behavior": "reuse",
            "assertions": [
                "selection.verified_reuse",
                "evidence.planner_scope_explicit",
                "output.matches_expected_json",
            ],
            "tags": ["reuse"],
        },
        {
            "id": "ordinary-conversation-does-not-call-mcp",
            "split": "adversarial",
            "sealed": False,
            "prompt": "你好，只要聊天，不要執行工具。",
            "expected_behavior": "conversation",
            "assertions": [
                "router.classifies_conversation",
                "mcp.call_count_zero",
                "blueprint.not_selected",
            ],
            "tags": ["conversation"],
        },
        {
            "id": "ordinary-conversation-en-does-not-call-mcp",
            "split": "adversarial",
            "sealed": False,
            "prompt": "Just chat with me; do not use tools.",
            "expected_behavior": "conversation",
            "assertions": [
                "router.classifies_conversation",
                "mcp.call_count_zero",
                "blueprint.not_selected",
            ],
            "tags": ["conversation"],
        },
        {
            "id": "multilingual-negation-zh",
            "split": "adversarial",
            "sealed": False,
            "prompt": "不要使用 MCP，也不要執行工作流程。",
            "expected_behavior": "negation",
            "assertions": [
                "negation.detected",
                "mcp.call_count_zero",
                "workflow.not_executed",
            ],
            "tags": ["negation"],
        },
        {
            "id": "multilingual-negation-ja",
            "split": "adversarial",
            "sealed": False,
            "prompt": "MCPもBlueprintも使わないでください。",
            "expected_behavior": "negation",
            "assertions": [
                "negation.detected",
                "mcp.call_count_zero",
                "router.classifies_conversation",
            ],
            "tags": ["negation"],
        },
        {
            "id": "quoted-tool-request-is-not-executed",
            "split": "adversarial",
            "sealed": False,
            "prompt": 'Explain why the phrase "delete the file" is dangerous.',
            "expected_behavior": "quoted",
            "assertions": [
                "router.detects_quoted_or_hypothetical",
                "mcp.call_count_zero",
                "workflow.not_executed",
            ],
            "tags": ["quoted"],
        },
        {
            "id": "malformed-community-evidence-cannot-promote",
            "split": "adversarial",
            "sealed": False,
            "prompt": "Import an unsigned Blueprint and treat it as official.",
            "expected_behavior": "quarantine",
            "assertions": [
                "sharing.quarantines_unknown_publisher",
                "evidence.community_cannot_promote",
                "scoring.trusted_score_unchanged",
            ],
            "tags": ["evidence"],
        },
        {
            "id": "incompatible-blueprint-is-not-reused",
            "split": "adversarial",
            "sealed": False,
            "prompt": "Reuse a high-scoring Blueprint from a different repository runtime framework.",
            "expected_behavior": "reject",
            "assertions": [
                "compatibility.mismatch_detected",
                "selection.false_reuse_zero",
                "planner.allowed_to_replan",
            ],
            "tags": ["compatibility"],
        },
        {
            "id": "sealed-multilingual-routing-holdout",
            "split": "sealed_holdout",
            "sealed": True,
            "task_digest": "sha256:"
            + hashlib.sha256(SEALED_PROMPT.encode()).hexdigest(),
            "expected_behavior": "private",
            "assertions": [
                "host.private_assertions_all_pass",
                "evidence.no_prompt_persisted",
                "selection.false_reuse_zero",
            ],
            "tags": ["sealed"],
        },
    ]
    return {
        "schema_version": "benchmark-suite.v1",
        "suite_id": "blueprint-effectiveness-test",
        "suite_version": 2,
        "description": "test suite",
        "required_modes": list(MODES),
        "trusted_evidence_tier": "ci_verified",
        "thresholds": {
            "min_trials_per_task": 20,
            "max_success_rate_drop": 0.02,
            "min_candidate_wilson_lower_bound": 0.80,
            "min_planner_token_reduction": 0.30,
            "max_p95_latency_increase": 0.10,
            "max_false_reuse_count": 0,
            "min_assertion_pass_rate": 1.0,
            "required_splits": [
                "public_eval",
                "adversarial",
                "sealed_holdout",
            ],
        },
        "tasks": tasks,
    }


def _environment_digest(config):
    return build_environment_digest(
        config,
        flyto_ai_commit="a" * 40,
        flyto_blueprint_commit="b" * 40,
    )


@pytest.mark.asyncio
async def test_host_emits_exactly_10_by_20_by_four_metrics_only_records():
    config = _config()
    host = BlueprintBenchmarkHost(
        suite=_suite(),
        config=config,
        planner=FakePlanner(),
        dataset_commit="c" * 40,
        environment_digest=_environment_digest(config),
        sealed_prompts={
            "sealed-multilingual-routing-holdout": SEALED_PROMPT,
        },
    )

    records = await host.run()

    assert len(records) == 10 * 20 * 4
    assert {
        (item["task_id"], item["trial"]) for item in records
    } == {
        (task["id"], trial)
        for task in _suite()["tasks"]
        for trial in range(1, 21)
    }
    assert all(set(MODES) == {
        item["mode"]
        for item in records
        if item["task_id"] == task["id"] and item["trial"] == 1
    } for task in _suite()["tasks"])
    assert all(
        item["success"] for item in records
        if item["mode"] == "blueprint_warm"
    )
    serialized = json.dumps(records, ensure_ascii=False)
    assert SEALED_PROMPT not in serialized
    assert "raw_prompt" not in serialized
    assert "raw_response" not in serialized


def test_host_config_rejects_credentials_and_fewer_than_20_trials():
    raw = {
        "schema_version": "flyto-ai-blueprint-host.v1",
        "provider": {
            "type": "ollama",
            "base_url": "http://user:secret@127.0.0.1:11434",
            "model": "flyto-qwen3:8b",
            "model_digest": "e737aff7b8d4",
            "think": False,
        },
        "run": {"trials": 19, "seed_base": 1},
        "compatibility": {"repository": "example/repo"},
    }
    with pytest.raises(BenchmarkHostError, match="credential-free"):
        HostConfig.from_mapping(raw)

    raw["provider"]["base_url"] = "http://127.0.0.1:11434"
    with pytest.raises(BenchmarkHostError, match="allowed range"):
        HostConfig.from_mapping(raw)


def test_sealed_prompt_must_match_commitment():
    config = _config()
    with pytest.raises(BenchmarkHostError, match="digest mismatch"):
        BlueprintBenchmarkHost(
            suite=_suite(),
            config=config,
            planner=FakePlanner(),
            dataset_commit="c" * 40,
            environment_digest=_environment_digest(config),
            sealed_prompts={
                "sealed-multilingual-routing-holdout": "wrong",
            },
        )


def test_write_runs_is_jsonl_and_does_not_add_prompt_fields(tmp_path):
    output = tmp_path / "sample.runs.jsonl"
    write_runs([{"success": True, "planner_input_tokens": 12}], output)

    assert json.loads(output.read_text()) == {
        "planner_input_tokens": 12,
        "success": True,
    }


def test_environment_digest_changes_with_code_or_model():
    config = _config()
    first = _environment_digest(config)
    second = build_environment_digest(
        config,
        flyto_ai_commit="d" * 40,
        flyto_blueprint_commit="b" * 40,
    )

    assert first.startswith("sha256:")
    assert first != second
