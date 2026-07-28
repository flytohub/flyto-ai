# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Run a pinned multi-model Blueprint benchmark matrix without mocks."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import yaml

from flyto_ai.benchmarking import BenchmarkHostError
from flyto_ai.benchmarking_v3 import main as run_v3_host

MATRIX_SCHEMA = "flyto-ai-blueprint-matrix.v1"
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{12,64}$")


def load_matrix(path: str | Path) -> list[dict]:
    """Load a strict matrix whose model entries are bound to Ollama digests."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise BenchmarkHostError("benchmark matrix must be an object")
    unknown = set(raw) - {"schema_version", "runs"}
    if unknown:
        raise BenchmarkHostError(
            "benchmark matrix has unknown field(s): {}".format(
                ", ".join(sorted(unknown))
            )
        )
    if raw.get("schema_version") != MATRIX_SCHEMA:
        raise BenchmarkHostError(
            "matrix schema_version must be {}".format(MATRIX_SCHEMA)
        )
    runs = raw.get("runs")
    if not isinstance(runs, list) or not runs:
        raise BenchmarkHostError("matrix runs must be a non-empty list")
    validated = []
    seen = set()
    for index, entry in enumerate(runs):
        if not isinstance(entry, Mapping):
            raise BenchmarkHostError(
                "matrix run {} must be an object".format(index)
            )
        unknown_entry = set(entry) - {"run_id", "model", "model_digest"}
        if unknown_entry:
            raise BenchmarkHostError(
                "matrix run {} has unknown field(s): {}".format(
                    index,
                    ", ".join(sorted(unknown_entry)),
                )
            )
        run_id = str(entry.get("run_id", ""))
        model = str(entry.get("model", ""))
        digest = str(entry.get("model_digest", ""))
        if not _ID_RE.fullmatch(run_id):
            raise BenchmarkHostError(
                "matrix run {} has invalid run_id".format(index)
            )
        if run_id in seen:
            raise BenchmarkHostError(
                "matrix run_id '{}' is duplicated".format(run_id)
            )
        if not _MODEL_RE.fullmatch(model):
            raise BenchmarkHostError(
                "matrix run {} has invalid model".format(index)
            )
        if not _DIGEST_RE.fullmatch(digest):
            raise BenchmarkHostError(
                "matrix run {} has invalid model_digest".format(index)
            )
        seen.add(run_id)
        validated.append(
            {
                "run_id": run_id,
                "model": model,
                "model_digest": digest,
            }
        )
    return validated


def run_matrix(
    *,
    matrix_path: str | Path,
    suite_path: str | Path,
    host_config_path: str | Path,
    output_dir: str | Path,
    dataset_commit: str,
    flyto_ai_commit: str,
    flyto_blueprint_commit: str,
    sealed_prompt_env: str,
) -> dict:
    """Execute every matrix entry and build one deterministic scorecard each."""
    from flyto_blueprint.benchmark import (
        build_scorecard,
        load_runs,
        load_suite,
        write_scorecard,
    )

    matrix = load_matrix(matrix_path)
    suite = load_suite(suite_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    completed = []
    for entry in matrix:
        raw_path = destination / "{}.runs.jsonl".format(entry["run_id"])
        scorecard_path = destination / "{}.scorecard.json".format(
            entry["run_id"]
        )
        if raw_path.exists() or scorecard_path.exists():
            raise BenchmarkHostError(
                "matrix output already exists for '{}'".format(
                    entry["run_id"]
                )
            )
        exit_code = run_v3_host(
            [
                "--suite",
                str(suite_path),
                "--config",
                str(host_config_path),
                "--dataset-commit",
                dataset_commit,
                "--flyto-ai-commit",
                flyto_ai_commit,
                "--flyto-blueprint-commit",
                flyto_blueprint_commit,
                "--run-id",
                entry["run_id"],
                "--model",
                entry["model"],
                "--model-digest",
                entry["model_digest"],
                "--output",
                str(raw_path),
                "--sealed-prompt-env",
                sealed_prompt_env,
            ]
        )
        if exit_code != 0:
            raise BenchmarkHostError(
                "matrix run '{}' failed".format(entry["run_id"])
            )
        scorecard = build_scorecard(suite, load_runs(raw_path))
        write_scorecard(scorecard, scorecard_path)
        if scorecard["proof_status"] != "verified":
            raise BenchmarkHostError(
                "matrix run '{}' regressed".format(entry["run_id"])
            )
        completed.append(
            {
                "run_id": entry["run_id"],
                "model": entry["model"],
                "raw_path": str(raw_path),
                "scorecard_path": str(scorecard_path),
                "scorecard_digest": scorecard["scorecard_digest"],
            }
        )
    return {
        "schema_version": MATRIX_SCHEMA,
        "status": "verified",
        "run_count": len(completed),
        "runs": completed,
        "raw_prompts_persisted": 0,
        "raw_responses_persisted": 0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI for repeatable local multi-model runs."""
    parser = argparse.ArgumentParser(
        description="Run the real multi-model Blueprint v3 matrix."
    )
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--host-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-commit", required=True)
    parser.add_argument("--flyto-ai-commit", required=True)
    parser.add_argument("--flyto-blueprint-commit", required=True)
    parser.add_argument(
        "--sealed-prompt-env",
        default="FLYTO_BENCHMARK_SEALED_PROMPT",
    )
    args = parser.parse_args(argv)
    try:
        result = run_matrix(
            matrix_path=args.matrix,
            suite_path=args.suite,
            host_config_path=args.host_config,
            output_dir=args.output_dir,
            dataset_commit=args.dataset_commit,
            flyto_ai_commit=args.flyto_ai_commit,
            flyto_blueprint_commit=args.flyto_blueprint_commit,
            sealed_prompt_env=args.sealed_prompt_env,
        )
    except (BenchmarkHostError, OSError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0
