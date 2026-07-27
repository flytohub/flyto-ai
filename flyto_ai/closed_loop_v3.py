# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Model-agnostic contracts for planning, recovery, routing, and evaluation."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Protocol, Sequence

PLAN_IR_VERSION = "flyto.plan-ir.v1"
CHECKPOINT_VERSION = "flyto.checkpoint.v1"
BENCHMARK_VERSION = "flyto.benchmark.v1"
_STEP_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_-]*)(?=[.\[])")
_HIGH_RISK_WORDS = frozenset({
    "architecture",
    "audit",
    "debug",
    "deploy",
    "fix",
    "implement",
    "migration",
    "payment",
    "production",
    "refactor",
    "security",
})
_FRONTIER_MODEL_MARKERS = (
    "claude-opus",
    "claude-sonnet",
    "codex",
    "gpt-5",
    "o1",
    "o3",
    "o4",
)
_BALANCED_MODEL_MARKERS = (
    "claude-haiku",
    "gpt-4",
    "qwen2.5:14b",
    "qwen3",
)


def stable_hash(value: Any) -> str:
    """Hash a JSON-compatible value deterministically."""
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:{}".format(hashlib.sha256(canonical.encode("utf-8")).hexdigest())


def _collect_step_refs(value: Any) -> List[str]:
    refs: List[str] = []
    if isinstance(value, str):
        refs.extend(match.group(1) for match in _STEP_REF_RE.finditer(value))
    elif isinstance(value, dict):
        for nested in value.values():
            refs.extend(_collect_step_refs(nested))
    elif isinstance(value, list):
        for nested in value:
            refs.extend(_collect_step_refs(nested))
    return refs


@dataclass(frozen=True)
class PlanStep:
    """One immutable logical step in a compiled execution plan."""

    step_id: str
    module_id: str
    params: Dict[str, Any]
    contract: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return the runtime-compatible step representation."""
        result = {
            "id": self.step_id,
            "module": self.module_id,
            "params": copy.deepcopy(self.params),
        }
        result.update(copy.deepcopy(self.contract))
        return result


@dataclass(frozen=True)
class PlanIR:
    """Typed, hash-addressed plan that must pass a gate before execution."""

    blueprint_id: str
    steps: Sequence[PlanStep]
    workflow_hash: str
    version: str = PLAN_IR_VERSION

    @classmethod
    def compile(
        cls,
        blueprint_id: str,
        steps: Sequence[Mapping[str, Any]],
    ) -> "PlanIR":
        """Compile loose workflow dictionaries into a stable Plan IR."""
        compiled: List[PlanStep] = []
        raw_steps = list(steps) if isinstance(steps, (list, tuple)) else []
        for index, raw in enumerate(raw_steps, start=1):
            step = dict(raw) if isinstance(raw, Mapping) else {}
            step_id = str(step.pop("id", "") or "step_{}".format(index))
            module_id = str(
                step.pop("module", "") or step.pop("module_id", "") or "",
            )
            params = step.pop("params", {})
            if not isinstance(params, dict):
                params = {"__invalid_params__": params}
            compiled.append(PlanStep(
                step_id=step_id,
                module_id=module_id,
                params=copy.deepcopy(params),
                contract=copy.deepcopy(step),
            ))
        normalized = [item.to_dict() for item in compiled]
        return cls(
            blueprint_id=blueprint_id,
            steps=tuple(compiled),
            workflow_hash=stable_hash(normalized),
        )

    def gate(self) -> List[Dict[str, Any]]:
        """Return structural errors; an empty list means execution is allowed."""
        errors: List[Dict[str, Any]] = []
        if not self.steps:
            return [{"step_id": None, "error": "Plan IR has no executable steps"}]

        seen: set[str] = set()
        for item in self.steps:
            step_errors = []
            if item.step_id in seen:
                step_errors.append("Duplicate step id: {}".format(item.step_id))
            if not item.module_id:
                step_errors.append("Step module is required")
            if "__invalid_params__" in item.params:
                step_errors.append("Step params must be an object")
            invalid_refs = sorted({
                ref for ref in _collect_step_refs(item.params) if ref not in seen
            })
            if invalid_refs:
                step_errors.append(
                    "Missing or forward step reference(s): {}".format(
                        ", ".join(invalid_refs),
                    ),
                )
            errors.extend({
                "step_id": item.step_id,
                "module_id": item.module_id,
                "error": error,
            } for error in step_errors)
            seen.add(item.step_id)
        return errors

    def to_steps(self) -> List[Dict[str, Any]]:
        """Return isolated runtime dictionaries."""
        return [item.to_dict() for item in self.steps]


class CheckpointStore(Protocol):
    """Persistence contract used by the Blueprint executor."""

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load one checkpoint."""

    def save(self, key: str, value: Dict[str, Any]) -> None:
        """Atomically save one checkpoint."""

    def delete(self, key: str) -> None:
        """Delete one checkpoint if it exists."""


class MemoryCheckpointStore:
    """Process-local checkpoint store useful for embedding and tests."""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        value = self._data.get(key)
        return copy.deepcopy(value) if value is not None else None

    def save(self, key: str, value: Dict[str, Any]) -> None:
        self._data[key] = copy.deepcopy(value)

    def delete(self, key: str) -> None:
        self._data.pop(key, None)


class JsonCheckpointStore:
    """Atomic 0600 JSON checkpoints.

    Step results may contain sensitive values because downstream references need
    exact outputs to resume. Enable this store only in a trusted local directory.
    """

    def __init__(self, base_dir: str) -> None:
        self._base_dir = Path(base_dir).expanduser()
        self._base_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            self._base_dir.chmod(0o700)
        except OSError:
            pass

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._base_dir / "{}.json".format(digest)

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path(key)
        if not path.exists():
            return None
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("Checkpoint path must be a regular file")
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None

    def save(self, key: str, value: Dict[str, Any]) -> None:
        path = self._path(key)
        temp_path = path.with_name(
            "{}.{}.tmp".format(path.name, uuid.uuid4().hex),
        )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(temp_path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            path.chmod(0o600)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists() and not path.is_symlink():
            path.unlink()


def checkpoint_key(blueprint_id: str, workflow_hash: str) -> str:
    """Return the stable key shared by failed and resumed executions."""
    return "{}:{}".format(blueprint_id, workflow_hash)


@dataclass(frozen=True)
class RepairDecision:
    """A bounded strategy change for one failed logical step."""

    module_id: str
    params: Dict[str, Any]
    reason: str
    retry: Optional[Dict[str, Any]] = None
    assertions: Optional[List[Dict[str, Any]]] = None


RepairFn = Callable[
    [Dict[str, Any], Dict[str, Any], Dict[str, Any]],
    Awaitable[Optional[RepairDecision]],
]


def repair_from_result(
    current_module: str,
    current_params: Dict[str, Any],
    failure: Dict[str, Any],
) -> Optional[RepairDecision]:
    """Consume a structured repair hint emitted by Core or middleware."""
    raw = failure.get("repair") if isinstance(failure, dict) else None
    if not isinstance(raw, dict):
        return None
    module_id = raw.get("module_id", current_module)
    params = raw.get("params", current_params)
    if not isinstance(module_id, str) or not module_id:
        return None
    if not isinstance(params, dict):
        return None
    if module_id == current_module and params == current_params:
        return None
    return RepairDecision(
        module_id=module_id,
        params=copy.deepcopy(params),
        reason=str(raw.get("reason") or "structured runtime repair hint"),
        retry=copy.deepcopy(raw.get("retry")) if isinstance(raw.get("retry"), dict) else None,
        assertions=(
            copy.deepcopy(raw.get("assertions"))
            if isinstance(raw.get("assertions"), list)
            else None
        ),
    )


@dataclass(frozen=True)
class ModelCandidate:
    """One configured model that can be selected by capability."""

    provider: str
    model: str
    tier: int
    cost_rank: int

    @property
    def label(self) -> str:
        return "{}:{}".format(self.provider, self.model or "default")

    @classmethod
    def from_name(
        cls,
        provider: str,
        model: str,
        cost_rank: int,
    ) -> "ModelCandidate":
        lowered = model.lower()
        if any(marker in lowered for marker in _FRONTIER_MODEL_MARKERS):
            tier = 3
        elif any(marker in lowered for marker in _BALANCED_MODEL_MARKERS):
            tier = 2
        else:
            tier = 1
        return cls(provider=provider, model=model, tier=tier, cost_rank=cost_rank)


@dataclass(frozen=True)
class ModelRoute:
    """Auditable model-routing decision."""

    mode: str
    required_tier: int
    reason: str
    provider: str = ""
    model: str = ""
    candidate_label: str = ""
    degraded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "required_tier": self.required_tier,
            "reason": self.reason,
            "provider": self.provider,
            "model": self.model,
            "candidate_label": self.candidate_label,
            "degraded": self.degraded,
        }


class CapabilityModelRouter:
    """Route cheap work locally and escalate novel/high-risk work."""

    def route(
        self,
        message: str,
        candidates: Sequence[ModelCandidate],
        *,
        deterministic_available: bool = False,
        prior_failure: bool = False,
        plan_steps: int = 0,
    ) -> ModelRoute:
        if deterministic_available:
            return ModelRoute(
                mode="deterministic",
                required_tier=0,
                reason="verified Blueprint or deterministic plan available",
                model="deterministic",
                candidate_label="deterministic",
            )

        lowered = message.lower()
        high_risk = prior_failure or plan_steps > 8 or any(
            word in lowered for word in _HIGH_RISK_WORDS
        )
        if high_risk:
            required_tier = 3
            reason = (
                "prior failure requires escalation"
                if prior_failure
                else "novel or high-risk task requires frontier reasoning"
            )
        elif len(message) > 180 or plan_steps > 3:
            required_tier = 2
            reason = "multi-step task requires balanced reasoning"
        else:
            required_tier = 1
            reason = "bounded low-risk task"

        ordered = sorted(candidates, key=lambda item: (item.cost_rank, -item.tier))
        eligible = [item for item in ordered if item.tier >= required_tier]
        degraded = False
        if eligible:
            selected = eligible[0]
        elif ordered:
            selected = max(ordered, key=lambda item: (item.tier, -item.cost_rank))
            degraded = True
            reason = "{}; no configured model meets tier {}".format(
                reason,
                required_tier,
            )
        else:
            return ModelRoute(
                mode="unavailable",
                required_tier=required_tier,
                reason="no model candidates configured",
                degraded=True,
            )
        return ModelRoute(
            mode="llm",
            required_tier=required_tier,
            reason=reason,
            provider=selected.provider,
            model=selected.model,
            candidate_label=selected.label,
            degraded=degraded,
        )


@dataclass(frozen=True)
class BenchmarkCase:
    """One model-independent agent benchmark case."""

    name: str
    message: str
    expected_ok: bool = True
    mode: str = "execute"


def _response_field(response: Any, name: str, default: Any) -> Any:
    if isinstance(response, dict):
        return response.get(name, default)
    return getattr(response, name, default)


def _cost_value(response: Any) -> float:
    cost = _response_field(response, "cost", None)
    if hasattr(cost, "model_dump"):
        cost = cost.model_dump()
    if not isinstance(cost, dict):
        return 0.0
    return float(cost.get("session_total_usd") or cost.get("total_usd") or 0.0)


async def run_model_benchmark(
    agents: Mapping[str, Any],
    cases: Sequence[BenchmarkCase],
) -> Dict[str, Any]:
    """Run the same closed-loop cases across agent/model implementations."""
    models: Dict[str, Any] = {}
    for label, agent_or_factory in agents.items():
        case_results = []
        for case in cases:
            agent = agent_or_factory() if callable(agent_or_factory) else agent_or_factory
            started = time.monotonic()
            try:
                response = await agent.chat(case.message, mode=case.mode)
                ok = bool(_response_field(response, "ok", False))
                error = _response_field(response, "error", None)
            except Exception as exc:
                response = {}
                ok = False
                error = str(exc) or type(exc).__name__
            duration_ms = int((time.monotonic() - started) * 1000)
            executions = _response_field(response, "execution_results", []) or []
            retries = sum(
                max(int(item.get("attempt_count", 1)) - 1, 0)
                for item in executions
                if isinstance(item, dict)
            )
            side_effects = sum(
                1 for item in executions
                if isinstance(item, dict) and item.get("executed", True)
            )
            assertions = [
                assertion
                for item in executions
                if isinstance(item, dict)
                for assertion in item.get("assertions", [])
                if isinstance(assertion, dict)
            ]
            case_results.append({
                "case": case.name,
                "ok": ok,
                "expected_ok": case.expected_ok,
                "passed": ok == case.expected_ok,
                "duration_ms": duration_ms,
                "rounds": int(_response_field(response, "rounds_used", 0) or 0),
                "retries": retries,
                "side_effects": side_effects,
                "assertions_passed": sum(1 for item in assertions if item.get("ok")),
                "assertions_total": len(assertions),
                "cost_usd": _cost_value(response),
                "error": error,
            })

        total = len(case_results)
        passed = sum(1 for item in case_results if item["passed"])
        models[label] = {
            "cases": case_results,
            "summary": {
                "passed": passed,
                "total": total,
                "success_rate": passed / total if total else 0.0,
                "duration_ms": sum(item["duration_ms"] for item in case_results),
                "rounds": sum(item["rounds"] for item in case_results),
                "retries": sum(item["retries"] for item in case_results),
                "side_effects": sum(item["side_effects"] for item in case_results),
                "cost_usd": sum(item["cost_usd"] for item in case_results),
            },
        }
    return {
        "version": BENCHMARK_VERSION,
        "case_count": len(cases),
        "models": models,
    }


@dataclass(frozen=True)
class DistillationDecision:
    """Eligibility result for turning one success into a verified Blueprint."""

    eligible: bool
    reason: str
    workflow: Optional[Dict[str, Any]] = None
    evidence_count: int = 0


def evaluate_distillation(
    tool_calls: Sequence[Dict[str, Any]],
    execution_results: Sequence[Dict[str, Any]],
    user_message: str,
    *,
    min_steps: int = 3,
) -> DistillationDecision:
    """Allow learning only from successful, runtime-verified executions."""
    if any(item.get("function") == "use_blueprint" for item in tool_calls):
        return DistillationDecision(False, "existing Blueprint was reused")
    if len(execution_results) < min_steps:
        return DistillationDecision(False, "not enough execution steps")

    steps = []
    evidence_count = 0
    for index, result in enumerate(execution_results, start=1):
        module_id = result.get("module_id", "")
        if not module_id or not result.get("ok") or result.get("executed") is False:
            return DistillationDecision(False, "execution contains an unverified step")

        validation = result.get("validation")
        assertions = result.get("assertions", [])
        validation_ok = isinstance(validation, dict) and (
            validation.get("valid") is True or validation.get("ok") is True
        )
        assertions_ok = bool(assertions) and all(
            isinstance(item, dict) and item.get("ok") is True
            for item in assertions
        )
        # Provider tool logs predate explicit validation evidence. Their ``ok``
        # field is still accepted, while v3 records must carry stronger proof.
        is_v3_record = "executed" in result or "validation" in result
        if is_v3_record and not (validation_ok or assertions_ok):
            return DistillationDecision(False, "v3 step lacks validation evidence")
        evidence_count += int(validation_ok) + sum(
            1 for item in assertions
            if isinstance(item, dict) and item.get("ok") is True
        )

        step = {
            "id": str(result.get("step_id") or "step_{}".format(index)),
            "module": module_id,
            "params": copy.deepcopy(
                result.get("arguments", {}).get("params", {}),
            ),
        }
        if assertions_ok:
            step["assertions"] = [
                {
                    "path": item.get("path", ""),
                    "op": item.get("op", "equals"),
                    "value": copy.deepcopy(item.get("expected")),
                }
                for item in assertions
            ]
        steps.append(step)

    return DistillationDecision(
        eligible=True,
        reason="all steps completed with runtime evidence",
        workflow={
            "name": user_message[:80],
            "description": "Distilled from verified closed-loop execution",
            "steps": steps,
            "distillation": {
                "version": PLAN_IR_VERSION,
                "evidence_count": evidence_count,
            },
        },
        evidence_count=evidence_count,
    )
