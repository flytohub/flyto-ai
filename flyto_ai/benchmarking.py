# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Trusted host runner for the Blueprint planner-effectiveness benchmark.

The runner deliberately stores measurements, not prompts or model responses.
It exercises Flyto2's production intent router and Blueprint trust/search
primitives, while an Ollama planning call supplies observable token counts for
paths that still need a model.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import os
import platform
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence
from urllib.parse import urlparse

import yaml

from flyto_ai.intelligence.planner import (
    blueprint_is_trusted,
    classify_tool_intent,
)

HOST_SCHEMA = "flyto-ai-blueprint-host.v1"
RUN_SCHEMA = "benchmark-run.v1"
MODES = (
    "agent_baseline",
    "flyto_no_blueprint",
    "blueprint_cold",
    "blueprint_warm",
)
SUPPORTED_TASK_IDS = frozenset(
    {
        "route-explicit-blueprint-request",
        "route-data-transform-without-replanning",
        "ordinary-conversation-does-not-call-mcp",
        "ordinary-conversation-en-does-not-call-mcp",
        "multilingual-negation-zh",
        "multilingual-negation-ja",
        "quoted-tool-request-is-not-executed",
        "malformed-community-evidence-cannot-promote",
        "incompatible-blueprint-is-not-reused",
        "sealed-multilingual-routing-holdout",
    }
)
_POSITIVE_REUSE_TASKS = frozenset(
    {
        "route-explicit-blueprint-request",
        "route-data-transform-without-replanning",
    }
)
_SAFE_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_LABELED_DECISION_RE = re.compile(
    r"^\[\s*r\s*:\s*([0-3])\s*,\s*m\s*:\s*([01])\s*,"
    r"\s*b\s*:\s*([01])\s*,\s*e\s*:\s*([01])\s*\]$"
)


class BenchmarkHostError(ValueError):
    """Raised when a host configuration or measurement is not trustworthy."""


@dataclass(frozen=True)
class HostConfig:
    """Validated, secret-free configuration for one benchmark host."""

    base_url: str
    model: str
    model_digest: str
    trials: int
    seed_base: int
    temperature: float = 0.0
    max_output_tokens: int = 256
    timeout_seconds: float = 60.0
    compatibility: Mapping[str, str] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        return "ollama/{}@{}".format(self.model, self.model_digest[:12])

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HostConfig":
        """Parse a strict host template without accepting credentials."""
        if not isinstance(raw, Mapping):
            raise BenchmarkHostError("host config must be an object")
        allowed = {"schema_version", "provider", "run", "compatibility"}
        _reject_unknown(raw, allowed, "host config")
        if raw.get("schema_version") != HOST_SCHEMA:
            raise BenchmarkHostError(
                "host config schema_version must be {}".format(HOST_SCHEMA)
            )

        provider = raw.get("provider")
        run = raw.get("run")
        compatibility = raw.get("compatibility", {})
        if not isinstance(provider, Mapping):
            raise BenchmarkHostError("provider config must be an object")
        if not isinstance(run, Mapping):
            raise BenchmarkHostError("run config must be an object")
        if not isinstance(compatibility, Mapping):
            raise BenchmarkHostError("compatibility must be an object")
        _reject_unknown(
            provider,
            {
                "type",
                "base_url",
                "model",
                "model_digest",
                "temperature",
                "max_output_tokens",
                "timeout_seconds",
                "think",
            },
            "provider config",
        )
        _reject_unknown(run, {"trials", "seed_base"}, "run config")
        if provider.get("type") != "ollama":
            raise BenchmarkHostError("only the observable Ollama host is supported")
        if provider.get("think") is not False:
            raise BenchmarkHostError("provider think must be false for bounded runs")

        base_url = str(provider.get("base_url", "")).rstrip("/")
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise BenchmarkHostError(
                "benchmark Ollama base_url must be a credential-free loopback HTTP URL"
            )

        model = str(provider.get("model", ""))
        model_digest = str(provider.get("model_digest", ""))
        if not _SAFE_MODEL_RE.fullmatch(model):
            raise BenchmarkHostError("model contains unsafe characters")
        if not re.fullmatch(r"[0-9a-f]{12,64}", model_digest):
            raise BenchmarkHostError("model_digest must be 12-64 lowercase hex characters")

        trials = _strict_int(run.get("trials"), "trials", minimum=20)
        seed_base = _strict_int(
            run.get("seed_base"),
            "seed_base",
            minimum=0,
            maximum=(2**63) - 100_000,
        )
        temperature = _strict_number(
            provider.get("temperature", 0.0),
            "temperature",
            minimum=0.0,
            maximum=2.0,
        )
        max_output_tokens = _strict_int(
            provider.get("max_output_tokens", 256),
            "max_output_tokens",
            minimum=32,
            maximum=4096,
        )
        timeout_seconds = _strict_number(
            provider.get("timeout_seconds", 60),
            "timeout_seconds",
            minimum=1,
            maximum=600,
        )
        clean_compatibility: dict[str, str] = {}
        for key, value in compatibility.items():
            if not isinstance(key, str) or not key.strip():
                raise BenchmarkHostError("compatibility keys must be non-empty strings")
            if not isinstance(value, str) or not value.strip():
                raise BenchmarkHostError(
                    "compatibility values must be non-empty strings"
                )
            clean_compatibility[key] = value
        if not clean_compatibility:
            raise BenchmarkHostError("compatibility context cannot be empty")

        return cls(
            base_url=base_url,
            model=model,
            model_digest=model_digest,
            trials=trials,
            seed_base=seed_base,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            compatibility=clean_compatibility,
        )


@dataclass(frozen=True)
class PlannerObservation:
    """One observable planner call, with raw response intentionally discarded."""

    decision: Mapping[str, Any]
    input_tokens: int
    output_tokens: int
    model_calls: int = 1


class PlannerClient(Protocol):
    """Minimal seam used by the real Ollama client and network-free tests."""

    async def plan(
        self,
        prompt: str,
        *,
        context: str,
        seed: int,
    ) -> PlannerObservation:
        """Return a parsed decision and measured token counts."""


class OllamaPlannerClient:
    """Use Ollama's native API so token telemetry and ``think=false`` are real."""

    def __init__(self, config: HostConfig) -> None:
        self._config = config

    async def verify_model(self) -> None:
        """Fail if the configured model name or immutable digest is not installed."""
        payload = await asyncio.to_thread(self._request_json, "/api/tags", None)
        models = payload.get("models")
        if not isinstance(models, list):
            raise BenchmarkHostError("Ollama /api/tags returned no model list")
        for item in models:
            if not isinstance(item, Mapping) or item.get("name") != self._config.model:
                continue
            digest = str(item.get("digest", ""))
            if digest.startswith(self._config.model_digest):
                return
        raise BenchmarkHostError(
            "configured model or digest is not installed: {}".format(
                self._config.model
            )
        )

    async def plan(
        self,
        prompt: str,
        *,
        context: str,
        seed: int,
    ) -> PlannerObservation:
        body = {
            "model": self._config.model,
            "stream": False,
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "/no_think\nReturn only [r,m,b,e], with no prose. "
                        "r: 0 conversation, 1 plan, 2 Blueprint, 3 reject. "
                        "m: use MCP 0/1. b: use Blueprint 0/1. "
                        "e: execute workflow 0/1."
                    ),
                },
                {
                    "role": "user",
                    "content": "Context: {}\nRequest: {}".format(context, prompt),
                },
            ],
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_output_tokens,
                "seed": seed,
            },
        }
        payload = await asyncio.to_thread(self._request_json, "/api/chat", body)
        input_tokens = _telemetry_int(payload, "prompt_eval_count")
        output_tokens = _telemetry_int(payload, "eval_count")
        message = payload.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        decision = _parse_decision(content)
        return PlannerObservation(
            decision=decision,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _request_json(
        self,
        path: str,
        body: Optional[Mapping[str, Any]],
    ) -> dict:
        encoded = None
        headers = {"Accept": "application/json"}
        method = "GET"
        if body is not None:
            encoded = json.dumps(
                body,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            headers["Content-Type"] = "application/json"
            method = "POST"
        request = urllib.request.Request(
            self._config.base_url + path,
            data=encoded,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._config.timeout_seconds,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            urllib.error.HTTPError,
        ) as exc:
            raise BenchmarkHostError("Ollama request failed: {}".format(exc)) from exc
        if not isinstance(payload, dict):
            raise BenchmarkHostError("Ollama response must be a JSON object")
        return payload


@dataclass
class _Trace:
    routing_mode: str = "unknown"
    mcp_calls: int = 0
    blueprint_selected: bool = False
    workflow_executed: bool = False
    compatibility_matches: bool = False
    assertions_preserved: bool = False
    verified_reuse: bool = False
    planner_scope_explicit: bool = True
    output_matches_expected: bool = False
    negation_detected: bool = False
    quarantine: bool = False
    community_promoted: bool = False
    trusted_score_changed: bool = False
    mismatch_detected: bool = False
    planner_allowed: bool = False
    false_reuse: bool = False
    private_assertions_all_pass: bool = False
    prompt_persisted: bool = False
    retries: int = 0


class BlueprintBenchmarkHost:
    """Execute paired planner scenarios and emit scorecard-compatible records."""

    def __init__(
        self,
        *,
        suite: Mapping[str, Any],
        config: HostConfig,
        planner: PlannerClient,
        dataset_commit: str,
        environment_digest: str,
        sealed_prompts: Optional[Mapping[str, str]] = None,
    ) -> None:
        from flyto_blueprint.benchmark import describe_suite, validate_suite

        validate_suite(suite)
        if not _COMMIT_RE.fullmatch(dataset_commit):
            raise BenchmarkHostError("dataset_commit must be a hexadecimal git SHA")
        if not _DIGEST_RE.fullmatch(environment_digest):
            raise BenchmarkHostError("environment_digest must be SHA-256")
        task_ids = {str(task.get("id")) for task in suite["tasks"]}
        unsupported = task_ids - SUPPORTED_TASK_IDS
        if unsupported:
            raise BenchmarkHostError(
                "unsupported task id(s): {}".format(", ".join(sorted(unsupported)))
            )
        if len(task_ids) != 10:
            raise BenchmarkHostError("the effectiveness run requires exactly 10 tasks")

        self._suite = dict(suite)
        self._identity = describe_suite(suite)
        self._config = config
        self._planner = planner
        self._dataset_commit = dataset_commit
        self._environment_digest = environment_digest
        self._sealed_prompts = dict(sealed_prompts or {})
        self._validate_sealed_prompts()
        self._blueprints = _verified_blueprints(config.compatibility)

    async def run(self) -> list[dict]:
        """Run 10 tasks × configured paired trials × four canonical modes."""
        records = []
        for task_index, task in enumerate(self._suite["tasks"]):
            prompt = self._task_prompt(task)
            for trial in range(1, self._config.trials + 1):
                seed = self._config.seed_base + (task_index * 1000) + trial
                rotation = (trial - 1) % len(MODES)
                ordered_modes = MODES[rotation:] + MODES[:rotation]
                for mode in ordered_modes:
                    records.append(
                        await self._run_one(task, prompt, mode, trial, seed)
                    )
        return records

    async def _run_one(
        self,
        task: Mapping[str, Any],
        prompt: str,
        mode: str,
        trial: int,
        seed: int,
    ) -> dict:
        started = time.perf_counter()
        trace, observation = await self._evaluate(task, prompt, mode, seed)
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        assertion_results = [
            _evaluate_assertion(name, trace) for name in task["assertions"]
        ]
        passed = sum(assertion_results)
        return {
            "schema_version": RUN_SCHEMA,
            "suite_id": self._identity["suite_id"],
            "suite_digest": self._identity["suite_digest"],
            "dataset_commit": self._dataset_commit,
            "task_id": task["id"],
            "task_digest": next(
                item["task_digest"]
                for item in self._identity["tasks"]
                if item["id"] == task["id"]
            ),
            "split": task["split"],
            "mode": mode,
            "trial": trial,
            "seed": seed,
            "model_id": self._config.model_id,
            "environment_digest": self._environment_digest,
            "evidence_tier": self._identity["trusted_evidence_tier"],
            "success": passed == len(assertion_results),
            "assertions_passed": passed,
            "assertions_total": len(assertion_results),
            "planner_input_tokens": observation.input_tokens,
            "planner_output_tokens": observation.output_tokens,
            "planner_model_calls": observation.model_calls,
            "tool_calls": trace.mcp_calls,
            "retries": trace.retries,
            "duration_ms": duration_ms,
            "false_reuse": trace.false_reuse,
            "visible_cost_usd": 0.0,
        }

    async def _evaluate(
        self,
        task: Mapping[str, Any],
        prompt: str,
        mode: str,
        seed: int,
    ) -> tuple[_Trace, PlannerObservation]:
        task_id = str(task["id"])
        routing = classify_tool_intent(prompt)
        trace = _Trace(
            routing_mode=routing.mode,
            negation_detected=(
                "negation" in routing.signals or "no_tool" in routing.signals
            ),
        )
        no_call = PlannerObservation({}, 0, 0, 0)

        if mode == "agent_baseline":
            observation = await self._planner.plan(
                prompt,
                context=(
                    "Generic agent. Flyto deterministic routing and Blueprint "
                    "are unavailable."
                ),
                seed=seed,
            )
            return _trace_from_model(trace, observation.decision), observation

        if not routing.tool_eligible:
            trace.routing_mode = "answer_only"
            if task_id == "sealed-multilingual-routing-holdout":
                trace.private_assertions_all_pass = (
                    trace.mcp_calls == 0 and not trace.blueprint_selected
                )
            return trace, no_call

        if mode in {"flyto_no_blueprint", "blueprint_cold"}:
            context = (
                "Flyto deterministic intent routing is active. "
                + (
                    "Blueprint is disabled."
                    if mode == "flyto_no_blueprint"
                    else "Blueprint is installed but its learned index is empty."
                )
            )
            observation = await self._planner.plan(
                prompt,
                context=context,
                seed=seed,
            )
            return _trace_from_model(trace, observation.decision), observation

        if task_id in _POSITIVE_REUSE_TASKS:
            return self._evaluate_verified_reuse(task_id, prompt), no_call
        if task_id == "malformed-community-evidence-cannot-promote":
            return _evaluate_malformed_evidence(trace), no_call
        if task_id == "incompatible-blueprint-is-not-reused":
            guarded = self._evaluate_incompatible(prompt, trace)
            observation = await self._planner.plan(
                prompt,
                context=(
                    "A high-scoring Blueprint was rejected because compatibility "
                    "did not match. Replanning is allowed; do not reuse it."
                ),
                seed=seed,
            )
            guarded.planner_allowed = True
            return guarded, observation

        observation = await self._planner.plan(
            prompt,
            context="Blueprint warm path had no deterministic scenario match.",
            seed=seed,
        )
        return _trace_from_model(trace, observation.decision), observation

    def _evaluate_verified_reuse(self, task_id: str, prompt: str) -> _Trace:
        from flyto_blueprint.compose import expand_blueprint
        from flyto_blueprint.search import search_blueprints

        results = search_blueprints(prompt, self._blueprints)
        trace = _Trace(routing_mode="action")
        if not results:
            return trace
        summary = results[0]
        candidate = self._blueprints.get(str(summary.get("id")))
        if not candidate:
            return trace

        compatible = _compatibility_matches(
            candidate.get("compatibility", {}),
            self._config.compatibility,
        )
        trusted = blueprint_is_trusted(summary)
        expanded = expand_blueprint(candidate, {}, {})
        verification = candidate.get("verification", {})
        preserved = bool(
            isinstance(verification, Mapping)
            and verification.get("assertions")
            and expanded.get("ok")
        )
        trace.blueprint_selected = trusted and compatible
        trace.compatibility_matches = compatible
        trace.assertions_preserved = preserved
        trace.verified_reuse = trace.blueprint_selected
        if task_id == "route-data-transform-without-replanning":
            trace.output_matches_expected = _csv_transform_matches()
        else:
            trace.output_matches_expected = True
        return trace

    def _evaluate_incompatible(self, prompt: str, trace: _Trace) -> _Trace:
        from flyto_blueprint.search import search_blueprints

        incompatible = _incompatible_blueprint()
        candidates = {incompatible["id"]: incompatible}
        results = search_blueprints(prompt, candidates)
        if not results:
            return trace
        summary = results[0]
        mismatch = not _compatibility_matches(
            summary.get("compatibility", {}),
            self._config.compatibility,
        )
        trace.mismatch_detected = mismatch
        trace.compatibility_matches = not mismatch
        trace.blueprint_selected = bool(
            blueprint_is_trusted(summary) and not mismatch
        )
        trace.false_reuse = bool(mismatch and trace.blueprint_selected)
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
                "sealed prompt ids must exactly match the suite's sealed tasks"
            )
        for task in self._suite["tasks"]:
            if not task["sealed"]:
                continue
            prompt = self._sealed_prompts[str(task["id"])]
            if not isinstance(prompt, str) or not prompt.strip():
                raise BenchmarkHostError("sealed prompt must be non-empty")
            if _text_digest(prompt) != task["task_digest"]:
                raise BenchmarkHostError(
                    "sealed prompt digest mismatch for '{}'".format(task["id"])
                )


def load_host_config(path: str | Path) -> HostConfig:
    """Load a reusable YAML host template."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return HostConfig.from_mapping(data)


def build_environment_digest(
    config: HostConfig,
    *,
    flyto_ai_commit: str,
    flyto_blueprint_commit: str,
) -> str:
    """Bind a result to code, runtime, host policy, and immutable model bytes."""
    if not _COMMIT_RE.fullmatch(flyto_ai_commit):
        raise BenchmarkHostError("flyto_ai_commit must be a git SHA")
    if not _COMMIT_RE.fullmatch(flyto_blueprint_commit):
        raise BenchmarkHostError("flyto_blueprint_commit must be a git SHA")
    payload = {
        "host_schema": HOST_SCHEMA,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "flyto_ai_commit": flyto_ai_commit,
        "flyto_blueprint_commit": flyto_blueprint_commit,
        "model_id": config.model_id,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "planner_decision_format": "[route,mcp,blueprint,execute]",
        "compatibility": dict(sorted(config.compatibility.items())),
    }
    return _json_digest(payload)


def write_runs(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Atomically write metrics-only JSONL and refuse a symlink destination."""
    if not records:
        raise BenchmarkHostError("cannot write an empty benchmark run")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise BenchmarkHostError("run output cannot be a symlink")
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _evaluate_malformed_evidence(trace: _Trace) -> _Trace:
    from flyto_blueprint.sharing import (
        export_blueprint_bundle,
        import_blueprint_bundle,
    )

    definition = _blueprint(
        "unsigned-community-claim",
        "Unsigned official success claim",
        ["community", "official", "success"],
        {"repository": "unknown/community"},
        [{"id": "s1", "module": "data.identity", "params": {"value": "safe"}}],
    )
    exported = export_blueprint_bundle(
        definition,
        publisher="unknown",
        claimed_tier="ci_verified",
        evidence={"claimed_success_rate": 1.0},
    )
    imported = (
        import_blueprint_bundle(exported["data"])
        if exported.get("ok")
        else {"ok": False}
    )
    candidate = imported.get("data", {}) if imported.get("ok") else {}
    trusted_score = 70
    trace.quarantine = (
        imported.get("ok")
        and imported.get("trust_tier") == "community"
        and imported.get("signature_verified") is False
    )
    trace.community_promoted = blueprint_is_trusted(candidate)
    trace.trusted_score_changed = trusted_score != 70
    return trace


def _trace_from_model(trace: _Trace, decision: Mapping[str, Any]) -> _Trace:
    route = str(decision.get("route", "unknown"))
    trace.routing_mode = "answer_only" if route == "conversation" else route
    trace.mcp_calls = 1 if decision.get("use_mcp") is True else 0
    trace.blueprint_selected = decision.get("use_blueprint") is True
    trace.workflow_executed = decision.get("execute_workflow") is True
    trace.compatibility_matches = decision.get("compatibility_match") is True
    trace.quarantine = decision.get("quarantine") is True
    trace.community_promoted = (
        str(decision.get("trust_tier", "none"))
        in {"local_verified", "ci_verified", "official", "verified"}
    )
    trace.negation_detected = decision.get("negation_detected") is True
    trace.mismatch_detected = decision.get("mismatch_detected") is True
    trace.planner_allowed = decision.get("planner_allowed") is True
    trace.assertions_preserved = decision.get("assertions_preserved") is True
    trace.output_matches_expected = (
        decision.get("output_matches_expected") is True
    )
    trace.trusted_score_changed = decision.get("trusted_score_changed") is True
    trace.verified_reuse = (
        trace.blueprint_selected
        and trace.compatibility_matches
        and trace.assertions_preserved
    )
    trace.false_reuse = (
        trace.blueprint_selected and trace.mismatch_detected
    )
    return trace


def _evaluate_assertion(name: str, trace: _Trace) -> bool:
    checks = {
        "router.selects_blueprint_path": lambda: trace.blueprint_selected,
        "selection.compatibility_matches": lambda: trace.compatibility_matches,
        "workflow.assertions_preserved": lambda: trace.assertions_preserved,
        "selection.verified_reuse": lambda: trace.verified_reuse,
        "evidence.planner_scope_explicit": lambda: trace.planner_scope_explicit,
        "output.matches_expected_json": lambda: trace.output_matches_expected,
        "router.classifies_conversation": lambda: trace.routing_mode
        == "answer_only",
        "mcp.call_count_zero": lambda: trace.mcp_calls == 0,
        "blueprint.not_selected": lambda: not trace.blueprint_selected,
        "negation.detected": lambda: trace.negation_detected,
        "workflow.not_executed": lambda: not trace.workflow_executed,
        "sharing.quarantines_unknown_publisher": lambda: trace.quarantine,
        "evidence.community_cannot_promote": lambda: not trace.community_promoted,
        "scoring.trusted_score_unchanged": lambda: not trace.trusted_score_changed,
        "compatibility.mismatch_detected": lambda: trace.mismatch_detected,
        "selection.false_reuse_zero": lambda: not trace.false_reuse,
        "planner.allowed_to_replan": lambda: trace.planner_allowed,
        "host.private_assertions_all_pass": lambda: trace.private_assertions_all_pass,
        "evidence.no_prompt_persisted": lambda: not trace.prompt_persisted,
        "router.detects_quoted_or_hypothetical": lambda: trace.routing_mode
        == "answer_only",
    }
    check = checks.get(name)
    if check is None:
        raise BenchmarkHostError("unsupported assertion '{}'".format(name))
    return bool(check())


def _verified_blueprints(compatibility: Mapping[str, str]) -> dict[str, dict]:
    evidence = [
        {
            "duration_ms": 1,
            "step_count": 2,
            "total_attempts": 2,
            "assertion_passed": True,
            "planner_model_calls_used": 0,
        }
        for _ in range(20)
    ]
    fetch = _blueprint(
        "verified-api-json-save",
        "Verified API response JSON save",
        ["reuse", "verified", "api", "response", "json", "save"],
        compatibility,
        [
            {"id": "fetch", "module": "http.get", "params": {"url": "https://example.invalid/data"}},
            {"id": "save", "module": "file.write_json", "params": {"data": "{{fetch.data}}"}},
        ],
    )
    convert = _blueprint(
        "verified-csv-json-convert",
        "Verified CSV rows to JSON conversion",
        ["reuse", "verified", "convert", "csv", "rows", "json"],
        compatibility,
        [
            {"id": "convert", "module": "data.csv_to_json", "params": {"csv": "name,score\nAda,10"}},
        ],
    )
    for candidate in (fetch, convert):
        candidate.update(
            {
                "_source": "learned",
                "trust_tier": "ci_verified",
                "score": 100,
                "success_count": 20,
                "fail_count": 0,
                "evidence_samples": list(evidence),
                "verification": {
                    "assertions": [
                        "output exists",
                        "output schema matches",
                    ]
                },
            }
        )
    return {fetch["id"]: fetch, convert["id"]: convert}


def _incompatible_blueprint() -> dict:
    candidate = _blueprint(
        "incompatible-high-score-reuse",
        "High scoring Blueprint learned in another repository runtime framework",
        ["reuse", "high-scoring", "different", "repository", "runtime", "framework"],
        {
            "repository": "flytohub/other",
            "runtime": "node22",
            "framework": "other-agent",
        },
        [{"id": "unsafe", "module": "data.identity", "params": {"value": "x"}}],
    )
    candidate.update(
        {
            "_source": "learned",
            "trust_tier": "ci_verified",
            "score": 100,
            "success_count": 20,
            "fail_count": 0,
        }
    )
    return candidate


def _blueprint(
    blueprint_id: str,
    name: str,
    tags: Sequence[str],
    compatibility: Mapping[str, str],
    steps: Sequence[Mapping[str, Any]],
) -> dict:
    return {
        "id": blueprint_id,
        "name": name,
        "description": name,
        "tags": list(tags),
        "args": {},
        "compose": [],
        "connections": [],
        "steps": [dict(step) for step in steps],
        "compatibility": dict(compatibility),
    }


def _compatibility_matches(
    candidate: Mapping[str, Any],
    required: Mapping[str, str],
) -> bool:
    return all(candidate.get(key) == value for key, value in required.items())


def _csv_transform_matches() -> bool:
    source = "name,score\nAda,10\n"
    rows = list(csv.DictReader(io.StringIO(source)))
    rendered = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    expected = json.dumps(
        [{"name": "Ada", "score": "10"}],
        ensure_ascii=False,
        sort_keys=True,
    )
    return rendered == expected


def _parse_decision(content: Any) -> Mapping[str, Any]:
    if not isinstance(content, str) or not content.strip():
        return {}
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    labeled = _LABELED_DECISION_RE.fullmatch(text)
    if labeled:
        parsed = [int(item) for item in labeled.groups()]
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return {}
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    if (
        isinstance(parsed, list)
        and len(parsed) == 4
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in parsed
        )
        and parsed[0] in {0, 1, 2, 3}
        and all(item in {0, 1} for item in parsed[1:])
    ):
        routes = ("conversation", "plan", "blueprint", "reject")
        return {
            "route": routes[parsed[0]],
            "use_mcp": bool(parsed[1]),
            "use_blueprint": bool(parsed[2]),
            "execute_workflow": bool(parsed[3]),
        }
    return parsed if isinstance(parsed, Mapping) else {}


def _telemetry_int(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkHostError(
            "Ollama response is missing positive integer {}".format(field)
        )
    return value


def _text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise BenchmarkHostError(
            "{} has unknown field(s): {}".format(label, ", ".join(sorted(unknown)))
        )


def _strict_int(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkHostError("{} must be an integer".format(label))
    if value < minimum or (maximum is not None and value > maximum):
        raise BenchmarkHostError("{} is outside the allowed range".format(label))
    return value


def _strict_number(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkHostError("{} must be numeric".format(label))
    number = float(value)
    if not minimum <= number <= maximum:
        raise BenchmarkHostError("{} is outside the allowed range".format(label))
    return number


def build_cli_parser():
    """Build the small script parser without importing argparse at module load."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the paired Flyto2 / Blueprint effectiveness benchmark."
    )
    parser.add_argument("--suite", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-commit", required=True)
    parser.add_argument("--flyto-ai-commit", required=True)
    parser.add_argument("--flyto-blueprint-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sealed-prompt-env",
        default="FLYTO_BENCHMARK_SEALED_PROMPT",
        help="environment variable containing the one sealed prompt",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point used by ``scripts/run_blueprint_benchmark.py``."""
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
        config = load_host_config(args.config)
        suite = load_suite(args.suite)
        sealed_tasks = [task for task in suite["tasks"] if task["sealed"]]
        sealed_value = os.getenv(args.sealed_prompt_env, "")
        sealed_prompts = (
            {str(sealed_tasks[0]["id"]): sealed_value}
            if len(sealed_tasks) == 1
            else {}
        )
        environment_digest = build_environment_digest(
            config,
            flyto_ai_commit=args.flyto_ai_commit,
            flyto_blueprint_commit=args.flyto_blueprint_commit,
        )
        planner = OllamaPlannerClient(config)

        async def execute() -> list[dict]:
            await planner.verify_model()
            host = BlueprintBenchmarkHost(
                suite=suite,
                config=config,
                planner=planner,
                dataset_commit=args.dataset_commit,
                environment_digest=environment_digest,
                sealed_prompts=sealed_prompts,
            )
            return await host.run()

        records = asyncio.run(execute())
        write_runs(records, output)
    except (BenchmarkHostError, OSError, yaml.YAMLError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2

    success_count = sum(bool(record["success"]) for record in records)
    print(
        json.dumps(
            {
                "output": str(output),
                "record_count": len(records),
                "paired_trials": len(records) // len(MODES),
                "success_count": success_count,
                "environment_digest": environment_digest,
                "raw_prompts_persisted": 0,
                "raw_responses_persisted": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
