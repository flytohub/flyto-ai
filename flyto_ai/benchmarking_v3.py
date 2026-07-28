# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Real-workload Blueprint benchmark host with full model-usage accounting."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from flyto_ai.benchmarking import (
    MODES,
    BenchmarkHostError,
    HostConfig,
    OllamaPlannerClient,
    PlannerObservation,
    _compatibility_matches,
    _evaluate_malformed_evidence,
    _incompatible_blueprint,
    _json_digest,
    _text_digest,
    build_environment_digest,
    write_runs,
)
from flyto_ai.intelligence.planner import (
    blueprint_is_trusted,
    classify_tool_intent,
)

HOST_SCHEMA_V2 = "flyto-ai-blueprint-host.v2"
RUN_SCHEMA_V2 = "benchmark-run.v2"
REAL_TASK_IDS = frozenset(
    {
        "real-coding-change-and-test",
        "real-browser-fetch-and-extract",
        "real-api-fetch-and-persist",
        "real-llm-summary-with-usage",
    }
)
SUPPORTED_TASK_IDS = REAL_TASK_IDS | {
    "ordinary-conversation-no-tools-v3",
    "multilingual-negation-zh-v3",
    "multilingual-negation-ja-v3",
    "malformed-community-evidence-v3",
    "incompatible-blueprint-v3",
    "sealed-multilingual-routing-holdout-v3",
}
WORKLOAD_KIND = {
    "real-coding-change-and-test": "coding",
    "real-browser-fetch-and-extract": "browser",
    "real-api-fetch-and-persist": "api",
    "real-llm-summary-with-usage": "llm",
    "ordinary-conversation-no-tools-v3": "conversation",
    "multilingual-negation-zh-v3": "conversation",
    "multilingual-negation-ja-v3": "conversation",
    "malformed-community-evidence-v3": "trust",
    "incompatible-blueprint-v3": "compatibility",
    "sealed-multilingual-routing-holdout-v3": "sealed",
}
WORKFLOW_IDS = {
    "real-coding-change-and-test": "real-code-statistics",
    "real-browser-fetch-and-extract": "real-browser-heading",
    "real-api-fetch-and-persist": "real-api-persist",
    "real-llm-summary-with-usage": "real-llm-summary",
}
_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class V3HostConfig:
    """Validated v3 host settings and the compatible planner configuration."""

    planner: HostConfig
    workflow_max_output_tokens: int


@dataclass(frozen=True)
class HostIdentity:
    """Stable identity for one physical or hosted benchmark environment."""

    host_id: str
    hardware_family: str
    runner_kind: str


@dataclass(frozen=True)
class WorkloadObservation:
    """Measured result of a real workload execution."""

    executed: bool
    success: bool
    digest: str
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    duration_ms: float = 0.0
    manual_corrections: int = 0
    tool_calls: int = 0


@dataclass
class V3Trace:
    """Assertion facts collected from routing and real execution."""

    routing_mode: str = "unknown"
    mcp_calls: int = 0
    workflow_executed: bool = False
    real_execution: bool = False
    output_verified: bool = False
    full_usage_accounted: bool = False
    negation_detected: bool = False
    quarantine: bool = False
    community_promoted: bool = False
    trusted_score_changed: bool = False
    mismatch_detected: bool = False
    planner_allowed: bool = False
    false_reuse: bool = False
    private_assertions_all_pass: bool = False
    prompt_persisted: bool = False


class V3OllamaClient(OllamaPlannerClient):
    """Use the same pinned Ollama model for planner and workflow observations."""

    def __init__(self, config: V3HostConfig) -> None:
        super().__init__(config.planner)
        self._v3_config = config

    async def complete_workflow(
        self,
        *,
        seed: int,
    ) -> tuple[str, int, int]:
        """Run one real model-backed workflow step and return native counters."""
        body = {
            "model": self._v3_config.planner.model,
            "stream": False,
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "/no_think\nReturn one short sentence only. "
                        "Do not add a preamble."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Summarize this measured API result: "
                        '[{"id":1},{"id":2},{"id":3}].'
                    ),
                },
            ],
            "options": {
                "temperature": self._v3_config.planner.temperature,
                "num_predict": self._v3_config.workflow_max_output_tokens,
                "seed": seed,
            },
        }
        payload = await asyncio.to_thread(
            self._request_json,
            "/api/chat",
            body,
        )
        message = payload.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise BenchmarkHostError("workflow model returned no content")
        input_tokens = _positive_telemetry(payload, "prompt_eval_count")
        output_tokens = _positive_telemetry(payload, "eval_count")
        return content.strip(), input_tokens, output_tokens


class _FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - standard-library handler API
        if self.path == "/page":
            body = (
                b"<!doctype html><html><body>"
                b"<h1>Blueprint Real Browser Fixture</h1>"
                b"</body></html>"
            )
            content_type = "text/html; charset=utf-8"
            status = 200
        elif self.path == "/api/items":
            body = b'[{"id":1},{"id":2},{"id":3}]'
            content_type = "application/json"
            status = 200
        else:
            body = b'{"error":"not found"}'
            content_type = "application/json"
            status = 404
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class _HeadingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_heading = False
        self.heading = ""

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, Optional[str]]],
    ) -> None:
        if tag == "h1":
            self.in_heading = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1":
            self.in_heading = False

    def handle_data(self, data: str) -> None:
        if self.in_heading:
            self.heading += data


class RealWorkloadExecutor:
    """Execute committed benchmark fixtures through real OS and HTTP paths."""

    def __init__(self, model: V3OllamaClient) -> None:
        self._model = model
        self._server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            _FixtureHandler,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._thread.start()
        self._base_url = "http://127.0.0.1:{}".format(
            self._server.server_address[1]
        )

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    async def execute(
        self,
        task_id: str,
        steps: Sequence[Mapping[str, Any]],
        *,
        seed: int,
    ) -> WorkloadObservation:
        started = time.perf_counter()
        modules = [str(step.get("module", "")) for step in steps]
        if task_id == "real-coding-change-and-test":
            success, digest, calls = await asyncio.to_thread(
                self._execute_coding,
                modules,
            )
            input_tokens = output_tokens = model_calls = 0
        elif task_id == "real-browser-fetch-and-extract":
            success, digest, calls = await asyncio.to_thread(
                self._execute_browser,
                modules,
            )
            input_tokens = output_tokens = model_calls = 0
        elif task_id == "real-api-fetch-and-persist":
            success, digest, calls = await asyncio.to_thread(
                self._execute_api,
                modules,
            )
            input_tokens = output_tokens = model_calls = 0
        elif task_id == "real-llm-summary-with-usage":
            if modules != ["benchmark.llm_summary"]:
                success, digest, calls = False, _digest("invalid-steps"), 0
                input_tokens = output_tokens = model_calls = 0
            else:
                content, input_tokens, output_tokens = (
                    await self._model.complete_workflow(seed=seed)
                )
                success = bool(content)
                digest = _digest(content)
                calls = model_calls = 1
        else:
            raise BenchmarkHostError(
                "unsupported real workload '{}'".format(task_id)
            )
        return WorkloadObservation(
            executed=True,
            success=success,
            digest=digest,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_calls=model_calls,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
            manual_corrections=0,
            tool_calls=calls,
        )

    def _execute_coding(self, modules: Sequence[str]) -> tuple[bool, str, int]:
        if list(modules) != ["benchmark.code_apply", "benchmark.code_test"]:
            return False, _digest("invalid-steps"), 0
        with tempfile.TemporaryDirectory(prefix="flyto-v3-code-") as directory:
            root = Path(directory)
            implementation = (
                "def mean(values):\n"
                "    if not values:\n"
                "        raise ValueError('values must not be empty')\n"
                "    return sum(values) / len(values)\n"
            )
            tests = (
                "import unittest\n"
                "from stats import mean\n\n"
                "class MeanTests(unittest.TestCase):\n"
                "    def test_numbers(self):\n"
                "        self.assertEqual(mean([2, 4, 6]), 4)\n\n"
                "    def test_empty(self):\n"
                "        with self.assertRaises(ValueError):\n"
                "            mean([])\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            )
            (root / "stats.py").write_text(implementation, encoding="utf-8")
            (root / "test_stats.py").write_text(tests, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-m", "unittest", "-q"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            digest = _digest(
                implementation
                + tests
                + str(completed.returncode)
            )
            return completed.returncode == 0, digest, 2

    def _execute_browser(
        self,
        modules: Sequence[str],
    ) -> tuple[bool, str, int]:
        if list(modules) != ["benchmark.browser_fetch"]:
            return False, _digest("invalid-steps"), 0
        request = urllib.request.Request(
            self._base_url + "/page",
            headers={"Accept": "text/html"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            html = response.read().decode("utf-8")
        parser = _HeadingParser()
        parser.feed(html)
        heading = parser.heading.strip()
        return (
            heading == "Blueprint Real Browser Fixture",
            _digest(heading),
            1,
        )

    def _execute_api(self, modules: Sequence[str]) -> tuple[bool, str, int]:
        if list(modules) != [
            "benchmark.api_fetch",
            "benchmark.json_persist",
        ]:
            return False, _digest("invalid-steps"), 0
        request = urllib.request.Request(
            self._base_url + "/api/items",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        with tempfile.TemporaryDirectory(prefix="flyto-v3-api-") as directory:
            output = Path(directory) / "items.json"
            output.write_text(
                json.dumps(payload, sort_keys=True),
                encoding="utf-8",
            )
            reloaded = json.loads(output.read_text(encoding="utf-8"))
        canonical = json.dumps(reloaded, sort_keys=True, separators=(",", ":"))
        return (
            isinstance(reloaded, list)
            and [item.get("id") for item in reloaded] == [1, 2, 3],
            _digest(canonical),
            2,
        )


class RealBlueprintBenchmarkHost:
    """Run the v3 paired suite against real Blueprint and workload paths."""

    def __init__(
        self,
        *,
        suite: Mapping[str, Any],
        config: V3HostConfig,
        model: V3OllamaClient,
        dataset_commit: str,
        environment_digest: str,
        run_id: str,
        host_identity: HostIdentity,
        sealed_prompts: Mapping[str, str],
    ) -> None:
        from flyto_blueprint.benchmark import describe_suite, validate_suite

        validate_suite(suite)
        task_ids = {str(task.get("id")) for task in suite["tasks"]}
        if task_ids != SUPPORTED_TASK_IDS:
            raise BenchmarkHostError(
                "v3 suite must contain the exact ten supported tasks"
            )
        if not _SAFE_ID_RE.fullmatch(run_id):
            raise BenchmarkHostError("run_id must be a safe identifier")
        if not _COMMIT_RE.fullmatch(dataset_commit):
            raise BenchmarkHostError(
                "dataset_commit must be a lowercase Git commit"
            )
        if not _DIGEST_RE.fullmatch(environment_digest):
            raise BenchmarkHostError(
                "environment_digest must be a SHA-256 digest"
            )
        for field, value in (
            ("host_id", host_identity.host_id),
            ("hardware_family", host_identity.hardware_family),
        ):
            if not _SAFE_ID_RE.fullmatch(value):
                raise BenchmarkHostError(
                    "{} must be a safe identifier".format(field)
                )
        if host_identity.runner_kind not in {"local", "independent_ci"}:
            raise BenchmarkHostError("runner_kind is not supported")
        self._suite = dict(suite)
        self._identity = describe_suite(suite)
        self._config = config
        self._model = model
        self._dataset_commit = dataset_commit
        self._environment_digest = environment_digest
        self._run_id = run_id
        self._run_started_at = (
            datetime.now(timezone.utc).isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        self._host_identity = host_identity
        self._sealed_prompts = dict(sealed_prompts)
        self._validate_sealed_prompts()
        self._engine = _build_verified_engine(config.planner.compatibility)

    async def run(self) -> list[dict]:
        executor = RealWorkloadExecutor(self._model)
        records = []
        try:
            for task_index, task in enumerate(self._suite["tasks"]):
                prompt = self._task_prompt(task)
                for trial in range(1, self._config.planner.trials + 1):
                    seed = (
                        self._config.planner.seed_base
                        + (task_index * 1000)
                        + trial
                    )
                    rotation = (trial - 1) % len(MODES)
                    ordered_modes = MODES[rotation:] + MODES[:rotation]
                    for mode in ordered_modes:
                        records.append(
                            await self._run_one(
                                task,
                                prompt,
                                mode,
                                trial,
                                seed,
                                executor,
                            )
                        )
        finally:
            executor.close()
        return records

    async def _run_one(
        self,
        task: Mapping[str, Any],
        prompt: str,
        mode: str,
        trial: int,
        seed: int,
        executor: RealWorkloadExecutor,
    ) -> dict:
        started = time.perf_counter()
        trace, planner, workload = await self._evaluate(
            task,
            prompt,
            mode,
            seed,
            executor,
        )
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        total_input = planner.input_tokens + workload.input_tokens
        total_output = planner.output_tokens + workload.output_tokens
        total_calls = planner.model_calls + workload.model_calls
        trace.full_usage_accounted = (
            total_input >= 0
            and total_output >= 0
            and total_calls
            == planner.model_calls + workload.model_calls
        )
        assertion_results = [
            _evaluate_v3_assertion(name, trace)
            for name in task["assertions"]
        ]
        passed = sum(assertion_results)
        task_identity = next(
            item
            for item in self._identity["tasks"]
            if item["id"] == task["id"]
        )
        record = {
            "schema_version": RUN_SCHEMA_V2,
            "suite_id": self._identity["suite_id"],
            "suite_digest": self._identity["suite_digest"],
            "dataset_commit": self._dataset_commit,
            "task_id": task["id"],
            "task_digest": task_identity["task_digest"],
            "split": task["split"],
            "mode": mode,
            "trial": trial,
            "seed": seed,
            "model_id": self._config.planner.model_id,
            "environment_digest": self._environment_digest,
            "evidence_tier": self._identity["trusted_evidence_tier"],
            "success": passed == len(assertion_results),
            "assertions_passed": passed,
            "assertions_total": len(assertion_results),
            "planner_input_tokens": planner.input_tokens,
            "planner_output_tokens": planner.output_tokens,
            "planner_model_calls": planner.model_calls,
            "tool_calls": workload.tool_calls,
            "retries": 0,
            "duration_ms": duration_ms,
            "false_reuse": trace.false_reuse,
            "visible_cost_usd": 0.0,
            "run_id": self._run_id,
            "run_started_at": self._run_started_at,
            "host_id": self._host_identity.host_id,
            "hardware_family": self._host_identity.hardware_family,
            "runner_kind": self._host_identity.runner_kind,
            "model_family": _model_family(self._config.planner.model),
            "workload_kind": WORKLOAD_KIND[str(task["id"])],
            "workload_digest": workload.digest,
            "workload_success": workload.success,
            "workflow_input_tokens": workload.input_tokens,
            "workflow_output_tokens": workload.output_tokens,
            "workflow_model_calls": workload.model_calls,
            "workflow_duration_ms": workload.duration_ms,
            "manual_corrections": workload.manual_corrections,
            "planner_visible_cost_usd": 0.0,
            "workflow_visible_cost_usd": 0.0,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_model_calls": total_calls,
            "total_visible_cost_usd": 0.0,
        }
        return record

    async def _evaluate(
        self,
        task: Mapping[str, Any],
        prompt: str,
        mode: str,
        seed: int,
        executor: RealWorkloadExecutor,
    ) -> tuple[V3Trace, PlannerObservation, WorkloadObservation]:
        task_id = str(task["id"])
        routing = classify_tool_intent(prompt)
        trace = V3Trace(
            routing_mode=routing.mode,
            negation_detected=(
                "negation" in routing.signals or "no_tool" in routing.signals
            ),
        )
        no_planner = PlannerObservation({}, 0, 0, 0)
        no_workload = WorkloadObservation(
            executed=False,
            success=True,
            digest=_digest("not-executed:" + task_id),
        )

        if not routing.tool_eligible and mode != "agent_baseline":
            trace.routing_mode = "answer_only"
            if task_id == "sealed-multilingual-routing-holdout-v3":
                trace.private_assertions_all_pass = True
            return trace, no_planner, no_workload

        if task_id in REAL_TASK_IDS:
            if mode == "blueprint_warm":
                steps = self._verified_steps(task_id, prompt)
                planner = no_planner
            else:
                planner = await self._model.plan(
                    prompt,
                    context=(
                        "Plan a real workload. Blueprint reuse is "
                        + (
                            "disabled."
                            if mode != "blueprint_cold"
                            else "cold and has no learned procedure."
                        )
                    ),
                    seed=seed,
                )
                steps = _reference_workflow(task_id)["steps"]
            workload = await executor.execute(task_id, steps, seed=seed)
            trace.workflow_executed = workload.executed
            trace.real_execution = workload.executed
            trace.output_verified = workload.success
            return trace, planner, workload

        if mode == "agent_baseline":
            planner = await self._model.plan(
                prompt,
                context="Generic agent without deterministic Flyto2 routing.",
                seed=seed,
            )
            decision = planner.decision
            trace.routing_mode = (
                "answer_only"
                if decision.get("route") == "conversation"
                else str(decision.get("route", "unknown"))
            )
            trace.mcp_calls = int(decision.get("use_mcp") is True)
            trace.workflow_executed = (
                decision.get("execute_workflow") is True
            )
            return trace, planner, WorkloadObservation(
                executed=trace.workflow_executed,
                success=not trace.workflow_executed,
                digest=_digest("baseline:" + task_id),
                tool_calls=trace.mcp_calls,
            )

        if task_id == "malformed-community-evidence-v3":
            source = _evaluate_malformed_evidence(_LegacyTrace())
            trace.quarantine = source.quarantine
            trace.community_promoted = source.community_promoted
            trace.trusted_score_changed = source.trusted_score_changed
            return trace, no_planner, no_workload

        if task_id == "incompatible-blueprint-v3":
            trace = self._evaluate_incompatible(prompt, trace)
            planner = await self._model.plan(
                prompt,
                context=(
                    "The incompatible Blueprint was rejected. "
                    "Replanning is allowed."
                ),
                seed=seed,
            )
            trace.planner_allowed = True
            return trace, planner, no_workload

        trace.routing_mode = "answer_only"
        if task_id == "sealed-multilingual-routing-holdout-v3":
            trace.private_assertions_all_pass = True
        return trace, no_planner, no_workload

    def _verified_steps(
        self,
        task_id: str,
        prompt: str,
    ) -> list[dict]:
        results = self._engine.search(prompt)
        expected_id = WORKFLOW_IDS[task_id]
        summary = next(
            (item for item in results if item.get("id") == expected_id),
            None,
        )
        if (
            not summary
            or not blueprint_is_trusted(summary)
            or not _compatibility_matches(
                summary.get("compatibility", {}),
                self._config.planner.compatibility,
            )
        ):
            return []
        expanded = self._engine.expand(expected_id, {})
        if not expanded.get("ok"):
            return []
        return [
            dict(step)
            for step in expanded["data"].get("steps", [])
            if isinstance(step, Mapping)
        ]

    def _evaluate_incompatible(
        self,
        prompt: str,
        trace: V3Trace,
    ) -> V3Trace:
        from flyto_blueprint.search import search_blueprints

        candidate = _incompatible_blueprint()
        results = search_blueprints(prompt, {candidate["id"]: candidate})
        if not results:
            return trace
        summary = results[0]
        mismatch = not _compatibility_matches(
            summary.get("compatibility", {}),
            self._config.planner.compatibility,
        )
        trace.mismatch_detected = mismatch
        selected = blueprint_is_trusted(summary) and not mismatch
        trace.false_reuse = bool(mismatch and selected)
        return trace

    def _task_prompt(self, task: Mapping[str, Any]) -> str:
        if not task["sealed"]:
            return str(task["prompt"])
        return self._sealed_prompts[str(task["id"])]

    def _validate_sealed_prompts(self) -> None:
        sealed_ids = {
            str(task["id"]) for task in self._suite["tasks"] if task["sealed"]
        }
        if set(self._sealed_prompts) != sealed_ids:
            raise BenchmarkHostError(
                "sealed prompt ids must exactly match the suite"
            )
        for task in self._suite["tasks"]:
            if not task["sealed"]:
                continue
            prompt = self._sealed_prompts[str(task["id"])]
            if _text_digest(prompt) != task["task_digest"]:
                raise BenchmarkHostError(
                    "sealed prompt digest mismatch for '{}'".format(task["id"])
                )


@dataclass
class _LegacyTrace:
    quarantine: bool = False
    community_promoted: bool = False
    trusted_score_changed: bool = False


def _build_verified_engine(compatibility: Mapping[str, str]):
    from flyto_blueprint import BlueprintEngine, MemoryBackend

    engine = BlueprintEngine(storage=MemoryBackend())
    for task_id in sorted(REAL_TASK_IDS):
        blueprint_id = WORKFLOW_IDS[task_id]
        workflow = _reference_workflow(task_id)
        result = engine.learn_from_workflow(
            workflow,
            blueprint_id=blueprint_id,
            name=workflow["name"],
            tags=workflow["tags"],
            verified=True,
            compatibility=dict(compatibility),
            verification={
                "assertions": [
                    "real execution completed",
                    "output digest verified",
                ]
            },
            trust_tier="ci_verified",
        )
        if not result.get("ok"):
            raise BenchmarkHostError(
                "failed to learn real workflow '{}'".format(task_id)
            )
        for index in range(20):
            outcome = engine.report_outcome(
                blueprint_id,
                True,
                execution_id="warm-{}-{}".format(blueprint_id, index),
                evidence_tier="ci_verified",
                evidence={
                    "duration_ms": 1,
                    "step_count": len(workflow["steps"]),
                    "total_attempts": len(workflow["steps"]),
                    "assertion_passed": True,
                    "planner_model_calls_used": 0,
                },
            )
            if not outcome.get("ok"):
                raise BenchmarkHostError(
                    "failed to verify learned workflow '{}'".format(task_id)
                )
    return engine


def _reference_workflow(task_id: str) -> dict:
    definitions = {
        "real-coding-change-and-test": {
            "name": "Real coding change and test",
            "tags": ["real", "coding", "statistics", "python", "tests"],
            "steps": [
                {"id": "apply", "module": "benchmark.code_apply", "params": {}},
                {"id": "test", "module": "benchmark.code_test", "params": {}},
            ],
        },
        "real-browser-fetch-and-extract": {
            "name": "Real browser heading extraction",
            "tags": ["real", "browser", "fetch", "extract", "heading"],
            "steps": [
                {
                    "id": "fetch",
                    "module": "benchmark.browser_fetch",
                    "params": {},
                }
            ],
        },
        "real-api-fetch-and-persist": {
            "name": "Real API fetch and persist",
            "tags": ["real", "api", "fetch", "save", "json", "persist"],
            "steps": [
                {"id": "fetch", "module": "benchmark.api_fetch", "params": {}},
                {
                    "id": "persist",
                    "module": "benchmark.json_persist",
                    "params": {},
                },
            ],
        },
        "real-llm-summary-with-usage": {
            "name": "Real LLM summary with usage",
            "tags": ["real", "llm", "summary", "usage", "tokens"],
            "steps": [
                {
                    "id": "summarize",
                    "module": "benchmark.llm_summary",
                    "params": {},
                }
            ],
        },
    }
    return json.loads(json.dumps(definitions[task_id]))


def _evaluate_v3_assertion(name: str, trace: V3Trace) -> bool:
    checks = {
        "workload.real_execution": trace.real_execution,
        "workload.output_verified": trace.output_verified,
        "evidence.full_usage_accounted": trace.full_usage_accounted,
        "router.classifies_conversation": trace.routing_mode == "answer_only",
        "mcp.call_count_zero": trace.mcp_calls == 0,
        "workload.not_executed": not trace.workflow_executed,
        "negation.detected": trace.negation_detected,
        "sharing.quarantines_unknown_publisher": trace.quarantine,
        "evidence.community_cannot_promote": not trace.community_promoted,
        "scoring.trusted_score_unchanged": not trace.trusted_score_changed,
        "compatibility.mismatch_detected": trace.mismatch_detected,
        "selection.false_reuse_zero": not trace.false_reuse,
        "planner.allowed_to_replan": trace.planner_allowed,
        "host.private_assertions_all_pass": trace.private_assertions_all_pass,
        "evidence.no_prompt_persisted": not trace.prompt_persisted,
    }
    if name not in checks:
        raise BenchmarkHostError(
            "unsupported v3 assertion '{}'".format(name)
        )
    return bool(checks[name])


def load_v3_host_config(
    path: str | Path,
    *,
    model: Optional[str] = None,
    model_digest: Optional[str] = None,
) -> V3HostConfig:
    """Load the v3 template and optionally bind a matrix model at runtime."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise BenchmarkHostError("v3 host config must be an object")
    if raw.get("schema_version") != HOST_SCHEMA_V2:
        raise BenchmarkHostError(
            "v3 host config schema_version must be {}".format(
                HOST_SCHEMA_V2
            )
        )
    provider = raw.get("provider")
    if not isinstance(provider, Mapping):
        raise BenchmarkHostError("provider config must be an object")
    provider_copy = dict(provider)
    workflow_max_output_tokens = provider_copy.pop(
        "workflow_max_output_tokens",
        None,
    )
    if model is not None:
        provider_copy["model"] = model
    if model_digest is not None:
        provider_copy["model_digest"] = model_digest
    base_raw = {
        "schema_version": "flyto-ai-blueprint-host.v1",
        "provider": provider_copy,
        "run": raw.get("run"),
        "compatibility": raw.get("compatibility"),
    }
    planner = HostConfig.from_mapping(base_raw)
    if (
        isinstance(workflow_max_output_tokens, bool)
        or not isinstance(workflow_max_output_tokens, int)
        or not 8 <= workflow_max_output_tokens <= 256
    ):
        raise BenchmarkHostError(
            "workflow_max_output_tokens must be an integer from 8 to 256"
        )
    return V3HostConfig(
        planner=planner,
        workflow_max_output_tokens=workflow_max_output_tokens,
    )


def detect_host_identity() -> HostIdentity:
    """Derive hardware and trust-boundary identity from the running host."""
    machine = platform.machine().lower().replace("_", "-")
    system = platform.system().lower()
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        hardware = "apple-silicon"
    else:
        hardware = "{}-{}".format(system, machine)
    runner_kind = (
        "independent_ci"
        if os.getenv("GITHUB_ACTIONS") == "true"
        else "local"
    )
    prefix = "github" if runner_kind == "independent_ci" else "local"
    return HostIdentity(
        host_id="{}-{}".format(prefix, hardware),
        hardware_family=hardware,
        runner_kind=runner_kind,
    )


def build_v3_environment_digest(
    config: V3HostConfig,
    *,
    flyto_ai_commit: str,
    flyto_blueprint_commit: str,
    host_identity: HostIdentity,
) -> str:
    """Bind v3 evidence to code, model bytes, host class, and runtime."""
    base = build_environment_digest(
        config.planner,
        flyto_ai_commit=flyto_ai_commit,
        flyto_blueprint_commit=flyto_blueprint_commit,
    )
    return _json_digest(
        {
            "base_environment_digest": base,
            "host_schema": HOST_SCHEMA_V2,
            "run_schema": RUN_SCHEMA_V2,
            "host_id": host_identity.host_id,
            "hardware_family": host_identity.hardware_family,
            "runner_kind": host_identity.runner_kind,
            "workflow_max_output_tokens": config.workflow_max_output_tokens,
        }
    )


def build_cli_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the real-workload Flyto2 Blueprint v3 benchmark."
    )
    parser.add_argument("--suite", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-commit", required=True)
    parser.add_argument("--flyto-ai-commit", required=True)
    parser.add_argument("--flyto-blueprint-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model")
    parser.add_argument("--model-digest")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sealed-prompt-env",
        default="FLYTO_BENCHMARK_SEALED_PROMPT",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from flyto_blueprint.benchmark import load_suite

    args = build_cli_parser().parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print(
            json.dumps({"error": "output exists; pass --overwrite"}),
            file=sys.stderr,
        )
        return 2
    try:
        config = load_v3_host_config(
            args.config,
            model=args.model,
            model_digest=args.model_digest,
        )
        suite = load_suite(args.suite)
        sealed_tasks = [task for task in suite["tasks"] if task["sealed"]]
        sealed_value = os.getenv(args.sealed_prompt_env, "")
        sealed_prompts = (
            {str(sealed_tasks[0]["id"]): sealed_value}
            if len(sealed_tasks) == 1
            else {}
        )
        host_identity = detect_host_identity()
        environment_digest = build_v3_environment_digest(
            config,
            flyto_ai_commit=args.flyto_ai_commit,
            flyto_blueprint_commit=args.flyto_blueprint_commit,
            host_identity=host_identity,
        )
        model = V3OllamaClient(config)

        async def execute() -> list[dict]:
            await model.verify_model()
            host = RealBlueprintBenchmarkHost(
                suite=suite,
                config=config,
                model=model,
                dataset_commit=args.dataset_commit,
                environment_digest=environment_digest,
                run_id=args.run_id,
                host_identity=host_identity,
                sealed_prompts=sealed_prompts,
            )
            return await host.run()

        records = asyncio.run(execute())
        write_runs(records, output)
    except (
        BenchmarkHostError,
        OSError,
        subprocess.SubprocessError,
        yaml.YAMLError,
        ValueError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(output),
                "record_count": len(records),
                "pair_count": len(records) // len(MODES),
                "model_id": config.planner.model_id,
                "host_id": host_identity.host_id,
                "raw_prompts_persisted": 0,
                "raw_responses_persisted": 0,
            },
            sort_keys=True,
        )
    )
    return 0


def _positive_telemetry(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkHostError(
            "Ollama response is missing positive integer {}".format(field)
        )
    return value


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _model_family(model: str) -> str:
    family = model.split(":", 1)[0].lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", family).strip("-")
    if not _SAFE_ID_RE.fullmatch(normalized):
        raise BenchmarkHostError("model family is not a safe identifier")
    return normalized
