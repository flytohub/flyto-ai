# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Domain-neutral aggregation for adapter conformance scenarios."""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

from flyto_ai.coding.conformance import (
    AdapterConformanceCase,
    AdapterConformanceReport,
    ManagerFactory,
    run_adapter_conformance,
)
from flyto_ai.coding.contracts import CapabilitySpec
from flyto_ai.coding.capabilities import CapabilityManager
from flyto_ai.permissions import PermissionLevel


SCENARIO_MATRIX_VERSION = "flyto.adapter-scenario-matrix.v1"


@dataclass(frozen=True)
class AdapterScenario:
    """One replaceable domain fixture; the runner has no domain-specific branches."""

    scenario_id: str
    domain: str
    spec: CapabilitySpec
    cases: Tuple[AdapterConformanceCase, ...]

    def __post_init__(self) -> None:
        for name, value in {"scenario_id": self.scenario_id, "domain": self.domain}.items():
            if not isinstance(value, str) or not value or len(value) > 128:
                raise ValueError("{} must be a bounded string".format(name))
        if not isinstance(self.spec, CapabilitySpec):
            raise TypeError("scenario spec must be a CapabilitySpec")
        try:
            cases = tuple(self.cases)
        except TypeError as exc:
            raise ValueError("scenario cases must contain conformance cases") from exc
        if not cases or any(
            not isinstance(case, AdapterConformanceCase) for case in cases
        ):
            raise ValueError("scenario cases must contain conformance cases")
        object.__setattr__(self, "cases", cases)


@dataclass(frozen=True)
class AdapterScenarioResult:
    scenario_id: str
    domain: str
    report: AdapterConformanceReport


@dataclass(frozen=True)
class ScenarioMatrixReport:
    results: Tuple[AdapterScenarioResult, ...]
    fingerprint: str

    @property
    def ok(self) -> bool:
        return bool(self.results) and all(result.report.ok for result in self.results)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": SCENARIO_MATRIX_VERSION,
            "ok": self.ok,
            "fingerprint": self.fingerprint,
            "results": [
                {
                    "scenario_id": result.scenario_id,
                    "domain": result.domain,
                    "report": result.report.as_dict(),
                }
                for result in self.results
            ],
        }


async def run_scenario_matrix(
    workspace: str,
    scenarios: Sequence[AdapterScenario],
    *,
    permission_level: PermissionLevel | str = PermissionLevel.READ_ONLY,
    max_concurrency: int = 4,
    manager_factory: ManagerFactory = CapabilityManager,
) -> ScenarioMatrixReport:
    """Run independent adapter suites with bounded cross-scenario concurrency."""
    scenarios = tuple(scenarios)
    if not scenarios or len(scenarios) > 64:
        raise ValueError("scenario matrix requires between 1 and 64 scenarios")
    if any(not isinstance(item, AdapterScenario) for item in scenarios):
        raise TypeError("scenario matrix contains an invalid scenario")
    ids = tuple(item.scenario_id for item in scenarios)
    if len(set(ids)) != len(ids):
        raise ValueError("scenario matrix contains duplicate scenario ids")
    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int):
        raise ValueError("max_concurrency must be an integer")
    if not 1 <= max_concurrency <= 32:
        raise ValueError("max_concurrency is outside the supported range")
    if not callable(manager_factory):
        raise ValueError("manager_factory must be callable")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(item: AdapterScenario) -> AdapterScenarioResult:
        async with semaphore:
            report = await run_adapter_conformance(
                workspace,
                item.spec,
                item.cases,
                permission_level=permission_level,
                manager_factory=manager_factory,
            )
            return AdapterScenarioResult(item.scenario_id, item.domain, report)

    results = tuple(await asyncio.gather(*(run(item) for item in scenarios)))
    fingerprint = hashlib.sha256(json.dumps(
        {
            "contract_version": SCENARIO_MATRIX_VERSION,
            "results": [
                {
                    "scenario_id": result.scenario_id,
                    "domain": result.domain,
                    "ok": result.report.ok,
                    "fingerprint": result.report.fingerprint,
                }
                for result in results
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return ScenarioMatrixReport(results=results, fingerprint=fingerprint)


__all__ = [
    "AdapterScenario",
    "AdapterScenarioResult",
    "SCENARIO_MATRIX_VERSION",
    "ScenarioMatrixReport",
    "run_scenario_matrix",
]
