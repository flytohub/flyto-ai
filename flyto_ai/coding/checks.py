# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Source-controlled configuration and real verification checks."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import yaml

from flyto_ai.coding.contracts import (
    CONFIG_VERSION,
    CapabilitySpec,
    CheckResult,
    CheckSpec,
)
from flyto_ai.coding.workspace import WorkspaceTools


MAX_CONFIG_BYTES = 256 * 1024


def load_project_config(
    workspace: str,
    relative_path: str = ".flyto/coding.yaml",
) -> Tuple[Tuple[CheckSpec, ...], Tuple[CapabilitySpec, ...]]:
    """Load one bounded YAML contract without environment expansion."""

    root = Path(workspace).resolve(strict=True)
    raw = Path(relative_path)
    if raw.is_absolute():
        raise ValueError("coding config path must be relative")
    path = (root / raw).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("coding config path escapes the workspace") from exc
    if not path.exists():
        return (), ()
    if not path.is_file() or path.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError("coding config is not a bounded file")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("coding config must be an object")
    if loaded.get("version") != CONFIG_VERSION:
        raise ValueError("coding config version is unsupported")
    unknown = set(loaded) - {"version", "checks", "capabilities"}
    if unknown:
        raise ValueError("coding config contains unsupported keys: {}".format(", ".join(sorted(unknown))))
    raw_checks = loaded.get("checks", [])
    raw_capabilities = loaded.get("capabilities", [])
    if not isinstance(raw_checks, list) or len(raw_checks) > 32:
        raise ValueError("coding config checks must be an array of at most 32 items")
    if not isinstance(raw_capabilities, list) or len(raw_capabilities) > 16:
        raise ValueError("coding config capabilities must be an array of at most 16 items")
    checks = tuple(CheckSpec.from_mapping(item) for item in raw_checks if isinstance(item, dict))
    capabilities = tuple(
        CapabilitySpec.from_mapping(item) for item in raw_capabilities if isinstance(item, dict)
    )
    if len(checks) != len(raw_checks) or len(capabilities) != len(raw_capabilities):
        raise ValueError("coding config entries must be objects")
    return checks, capabilities


class CheckRunner:
    """Execute declared checks and retain content-addressed evidence."""

    def __init__(self, workspace_tools: WorkspaceTools) -> None:
        self.workspace_tools = workspace_tools

    async def run(self, checks: Sequence[CheckSpec]) -> List[CheckResult]:
        results: List[CheckResult] = []
        for check in checks:
            started = time.monotonic()
            try:
                raw = await self.workspace_tools.run_check(check.argv, check.timeout_seconds)
                output = str(raw.get("output", ""))
                results.append(CheckResult(
                    name=check.name,
                    passed=bool(raw.get("ok")),
                    required=check.required,
                    exit_code=raw.get("exit_code"),
                    duration_ms=int(raw.get("duration_ms", (time.monotonic() - started) * 1000)),
                    output_sha256=str(raw.get("output_sha256") or hashlib.sha256(output.encode()).hexdigest()),
                    output_preview=output[-4000:],
                    error=None if raw.get("ok") else ("timed out" if raw.get("timed_out") else "non-zero exit"),
                ))
            except Exception as exc:
                message = str(exc)[:1000]
                results.append(CheckResult(
                    name=check.name,
                    passed=False,
                    required=check.required,
                    exit_code=None,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    output_sha256=hashlib.sha256(message.encode()).hexdigest(),
                    error=message,
                ))
        return results

    @staticmethod
    def passed(results: Sequence[CheckResult]) -> bool:
        return bool(results) and all(result.passed for result in results if result.required)
