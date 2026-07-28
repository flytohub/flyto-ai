import asyncio
from pathlib import Path

import pytest

from flyto_ai.benchmarking import BenchmarkHostError
from flyto_ai.benchmark_matrix import load_matrix
from flyto_ai.benchmarking_v3 import (
    HostIdentity,
    RealBlueprintBenchmarkHost,
    RealWorkloadExecutor,
    _reference_workflow,
    build_v3_environment_digest,
    load_v3_host_config,
)
from flyto_blueprint.benchmark import load_suite


ROOT = Path(__file__).resolve().parents[1]
BLUEPRINT_ROOT = ROOT.parent / "flyto-blueprint"
CONFIG_PATH = (
    BLUEPRINT_ROOT / "benchmarks/templates/host-run-v3-template.yaml"
)
SUITE_PATH = (
    BLUEPRINT_ROOT
    / "benchmarks/suites/blueprint-effectiveness-v3.yaml"
)
MODEL_DIGEST = "f" * 64
COMMIT = "a" * 40
SEALED_PROMPT = (
    "請不要呼叫任何工具，也不要搜尋 Blueprint；"
    "只用繁體中文說明「重複使用」的意思。"
)


def _config():
    return load_v3_host_config(
        CONFIG_PATH,
        model="qwen3:0.6b",
        model_digest=MODEL_DIGEST,
    )


def test_real_coding_browser_and_api_workloads_use_os_paths():
    executor = RealWorkloadExecutor(object())

    async def run():
        observations = []
        for task_id in (
            "real-coding-change-and-test",
            "real-browser-fetch-and-extract",
            "real-api-fetch-and-persist",
        ):
            observations.append(
                await executor.execute(
                    task_id,
                    _reference_workflow(task_id)["steps"],
                    seed=1,
                )
            )
        return observations

    try:
        observations = asyncio.run(run())
    finally:
        executor.close()

    assert all(item.executed for item in observations)
    assert all(item.success for item in observations)
    assert all(item.digest.startswith("sha256:") for item in observations)
    assert [item.tool_calls for item in observations] == [2, 1, 2]


def test_v3_environment_binds_full_host_identity():
    config = _config()
    local = HostIdentity(
        host_id="local-apple-silicon",
        hardware_family="apple-silicon",
        runner_kind="local",
    )
    independent = HostIdentity(
        host_id="github-linux-x86-64",
        hardware_family="linux-x86-64",
        runner_kind="independent_ci",
    )

    local_digest = build_v3_environment_digest(
        config,
        flyto_ai_commit=COMMIT,
        flyto_blueprint_commit=COMMIT,
        host_identity=local,
    )
    independent_digest = build_v3_environment_digest(
        config,
        flyto_ai_commit=COMMIT,
        flyto_blueprint_commit=COMMIT,
        host_identity=independent,
    )

    assert local_digest.startswith("sha256:")
    assert independent_digest.startswith("sha256:")
    assert local_digest != independent_digest


def test_v3_host_rejects_unbound_provenance():
    suite = load_suite(SUITE_PATH)
    sealed_prompts = {
        "sealed-multilingual-routing-holdout-v3": SEALED_PROMPT
    }

    with pytest.raises(BenchmarkHostError, match="dataset_commit"):
        RealBlueprintBenchmarkHost(
            suite=suite,
            config=_config(),
            model=object(),
            dataset_commit="not-a-commit",
            environment_digest="sha256:" + ("b" * 64),
            run_id="invalid-provenance",
            host_identity=HostIdentity(
                host_id="local-apple-silicon",
                hardware_family="apple-silicon",
                runner_kind="local",
            ),
            sealed_prompts=sealed_prompts,
        )


def test_v3_host_template_never_contains_secret_holdout_prompt():
    raw = CONFIG_PATH.read_text(encoding="utf-8")

    assert SEALED_PROMPT not in raw
    assert "api_key" not in raw.lower()
    assert "password" not in raw.lower()


def test_real_model_matrix_has_history_and_three_model_families():
    matrix = load_matrix(
        ROOT / "benchmarks/blueprint-v3-model-matrix.yaml"
    )
    families = {entry["model"].split(":", 1)[0] for entry in matrix}

    assert len(matrix) == 4
    assert families == {"qwen3", "llama3.2", "gemma3"}
    assert sum(entry["model"] == "qwen3:0.6b" for entry in matrix) == 2
