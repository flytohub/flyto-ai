#!/usr/bin/env python3
"""Run 101+ no-mock ordinary-development cases through FlytoCodingAgent."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from uuid import uuid4

from flyto_ai.coding import CheckSpec, CodingTaskRequest, FlytoCodingAgent, ThreadStore
from flyto_ai.providers.ollama import OllamaProvider


DEFAULT_CASES = 101
DEFAULT_RATE = 0.90
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_AGENT_ATTEMPTS = 3
OLLAMA_TRANSPORT = "native-/api/chat"
OLLAMA_THINK = False
TIERS = ("standard", "intermediate", "advanced")


class BenchmarkFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 101+ real Ollama coding-agent cases with real checks."
    )
    parser.add_argument("--cases", type=int, default=DEFAULT_CASES)
    parser.add_argument("--minimum-rate", type=float, default=DEFAULT_RATE)
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--case-timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--max-agent-attempts",
        type=int,
        default=DEFAULT_MAX_AGENT_ATTEMPTS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/benchmarks/native-coding"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--keep-workspaces", action="store_true")
    args = parser.parse_args()
    if args.cases < DEFAULT_CASES:
        parser.error(f"--cases must be at least {DEFAULT_CASES}")
    if not 0.0 < args.minimum_rate <= 1.0:
        parser.error("--minimum-rate must be in (0, 1]")
    if not 10.0 <= args.case_timeout <= 900.0:
        parser.error("--case-timeout must be between 10 and 900 seconds")
    if not 512 <= args.max_tokens <= 32768:
        parser.error("--max-tokens must be between 512 and 32768")
    if not 1 <= args.max_agent_attempts <= 5:
        parser.error("--max-agent-attempts must be between 1 and 5")
    return args


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def benchmark_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Return the execution settings that must stay fixed across a run."""
    return {
        "max_agent_attempts": args.max_agent_attempts,
        "max_completion_tokens": args.max_tokens,
        "transport": OLLAMA_TRANSPORT,
        "think": OLLAMA_THINK,
    }


def tier_sequence(count: int) -> list[str]:
    base, remainder = divmod(count, len(TIERS))
    sizes = [base + (1 if index < remainder else 0) for index in range(len(TIERS))]
    return [tier for tier, size in zip(TIERS, sizes) for _ in range(size)]


def _standard_case(index: int) -> dict[str, Any]:
    offset = (index % 17) + 2
    name = f"apply_offset_{index:03d}"
    files = {
        "logic.py": (
            f"def {name}(value):\n"
            f"    \"\"\"Return value increased by exactly {offset}.\"\"\"\n"
            f"    return value - {offset}\n"
        ),
        "test_logic.py": (
            "import unittest\n\n"
            f"from logic import {name}\n\n\n"
            "class LogicTest(unittest.TestCase):\n"
            "    def test_positive_zero_and_negative_inputs(self):\n"
            f"        self.assertEqual({name}(10), {10 + offset})\n"
            f"        self.assertEqual({name}(0), {offset})\n"
            f"        self.assertEqual({name}(-5), {-5 + offset})\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
    }
    return {
        "depth": 1,
        "files": files,
        "allowed_changes": ["logic.py"],
        "message": (
            "Fix the failing offset implementation so the existing tests pass. "
            "Inspect the workspace, edit only logic.py, preserve the public function "
            "name, and rely on the required host check for proof."
        ),
    }


def _intermediate_case(index: int) -> dict[str, Any]:
    name = f"summarize_values_{index:03d}"
    values = [index % 9 + 1, 8, 8, 3]
    unique = sorted(set(values))
    average = round(sum(unique) / len(unique), 2)
    files = {
        "service.py": (
            f"def {name}(values):\n"
            "    \"\"\"Return ascending unique values plus count, total and 2dp average.\"\"\"\n"
            "    if not values:\n"
            "        return {'values': [], 'count': 0, 'total': 0, 'average': 0.0}\n"
            "    ordered = sorted(set(values), reverse=True)\n"
            "    total = sum(ordered)\n"
            "    return {\n"
            "        'values': ordered,\n"
            "        'count': len(values),\n"
            "        'total': total,\n"
            "        'average': total // len(ordered),\n"
            "    }\n"
        ),
        "test_service.py": (
            "import unittest\n\n"
            f"from service import {name}\n\n\n"
            "class ServiceTest(unittest.TestCase):\n"
            "    def test_summary_contract(self):\n"
            f"        self.assertEqual({name}({values!r}), {{\n"
            f"            'values': {unique!r},\n"
            f"            'count': {len(unique)},\n"
            f"            'total': {sum(unique)},\n"
            f"            'average': {average!r},\n"
            "        })\n\n"
            "    def test_empty_values(self):\n"
            f"        self.assertEqual({name}([]), {{\n"
            "            'values': [], 'count': 0, 'total': 0, 'average': 0.0,\n"
            "        })\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
    }
    return {
        "depth": 3,
        "files": files,
        "allowed_changes": ["service.py"],
        "message": (
            "Repair the summary implementation to match its docstring and all existing "
            "tests. Inspect both files, edit only service.py, do not weaken tests, and "
            "use the required host check as the acceptance proof."
        ),
    }


def _advanced_case(index: int) -> dict[str, Any]:
    name = f"rank_active_records_{index:03d}"
    bonus = (index % 7) + 1
    files = {
        "records/__init__.py": "",
        "records/models.py": (
            "from dataclasses import dataclass\n\n\n"
            "@dataclass(frozen=True)\n"
            "class Record:\n"
            "    name: str\n"
            "    score: int\n"
            "    active: bool = True\n"
        ),
        "records/service.py": (
            "from .models import Record\n\n\n"
            f"def {name}(records, bonus):\n"
            "    \"\"\"Rank active records by capped score descending, then name ascending.\"\"\"\n"
            "    selected = [record for record in records if not record.active]\n"
            "    ranked = [\n"
            "        (record.name, min(100, record.score + bonus))\n"
            "        for record in selected\n"
            "    ]\n"
            "    return sorted(ranked, key=lambda item: (item[1], item[0]))\n"
        ),
        "test_records.py": (
            "import unittest\n\n"
            "from records.models import Record\n"
            f"from records.service import {name}\n\n\n"
            "class RecordServiceTest(unittest.TestCase):\n"
            "    def test_filters_caps_and_ranks(self):\n"
            "        records = [\n"
            "            Record('delta', 99, True),\n"
            "            Record('alpha', 90, True),\n"
            "            Record('charlie', 90, True),\n"
            "            Record('ignored', 100, False),\n"
            "        ]\n"
            f"        self.assertEqual({name}(records, {bonus}), [\n"
            f"            ('delta', {min(100, 99 + bonus)}),\n"
            f"            ('alpha', {min(100, 90 + bonus)}),\n"
            f"            ('charlie', {min(100, 90 + bonus)}),\n"
            "        ])\n\n"
            "    def test_empty_input(self):\n"
            f"        self.assertEqual({name}([], {bonus}), [])\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
    }
    return {
        "depth": 6,
        "files": files,
        "allowed_changes": ["records/service.py"],
        "message": (
            "Fix the record ranking service so its documented filtering, score cap, "
            "descending score order, and deterministic name tie-break all satisfy the "
            "existing tests. Inspect the package, edit only records/service.py, and "
            "do not modify tests or models."
        ),
    }


def case_spec(index: int, tier: str) -> dict[str, Any]:
    builders = {
        "standard": _standard_case,
        "intermediate": _intermediate_case,
        "advanced": _advanced_case,
    }
    try:
        spec = builders[tier](index)
    except KeyError as exc:
        raise ValueError(f"unknown tier: {tier}") from exc
    return {"case_id": f"native-{tier}-{index:03d}", "tier": tier, **spec}


def benchmark_thread_id(case_id: str) -> str:
    """Return a fresh thread id so interrupted checkpoints resume safely."""
    return f"{case_id.replace('-', '_')}_{uuid4().hex}"


def write_fixture(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def ollama_model_info(base_url: str, model: str) -> dict[str, Any]:
    tags_url = base_url.rstrip("/")
    if tags_url.endswith("/v1"):
        tags_url = tags_url[:-3]
    with urlopen(f"{tags_url}/api/tags", timeout=10) as response:
        payload = json.load(response)
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise BenchmarkFailure("Ollama model inventory is unavailable")
    for item in models:
        if isinstance(item, dict) and item.get("name") == model:
            return {
                "name": model,
                "digest": str(item.get("digest") or ""),
                "size": int(item.get("size") or 0),
            }
    raise BenchmarkFailure(f"Ollama model is not installed: {model}")


def summarize(cases: list[dict[str, Any]], minimum_rate: float) -> dict[str, Any]:
    tiers: dict[str, Any] = {}
    for tier in TIERS:
        selected = [case for case in cases if case["tier"] == tier]
        successful = sum(1 for case in selected if case["success"])
        rate = successful / len(selected) if selected else 0.0
        tiers[tier] = {
            "total": len(selected),
            "successful": successful,
            "failed": len(selected) - successful,
            "success_rate": round(rate, 6),
            "gate_pass": bool(selected) and rate >= minimum_rate,
        }
    successful = sum(1 for case in cases if case["success"])
    rate = successful / len(cases) if cases else 0.0
    return {
        "total": len(cases),
        "successful": successful,
        "failed": len(cases) - successful,
        "success_rate": round(rate, 6),
        "minimum_count": DEFAULT_CASES,
        "minimum_rate": minimum_rate,
        "all_distinct": len({case["case_id"] for case in cases}) == len(cases),
        "no_hidden_retries": all(case["hidden_retries"] == 0 for case in cases),
        "tiers": tiers,
        "gate_pass": (
            len(cases) >= DEFAULT_CASES
            and rate >= minimum_rate
            and all(item["gate_pass"] for item in tiers.values())
            and len({case["case_id"] for case in cases}) == len(cases)
            and all(case["hidden_retries"] == 0 for case in cases)
        ),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


async def run_case(
    *,
    agent: FlytoCodingAgent,
    workspace: Path,
    spec: dict[str, Any],
    case_timeout: float,
    max_agent_attempts: int,
) -> dict[str, Any]:
    write_fixture(workspace, spec["files"])
    started = time.monotonic()
    error = None
    result = None
    try:
        result = await asyncio.wait_for(
            agent.run(
                CodingTaskRequest(
                    message=spec["message"],
                    working_dir=str(workspace),
                    thread_id=benchmark_thread_id(spec["case_id"]),
                    checks=(
                        CheckSpec(
                            name="unittest",
                            argv=("python", "-m", "unittest", "-q"),
                            timeout_seconds=60,
                        ),
                    ),
                    max_attempts=max_agent_attempts,
                    max_rounds=8,
                )
            ),
            timeout=case_timeout,
        )
        changed = sorted(result.files_changed)
        expected = sorted(spec["allowed_changes"])
        if not result.ok:
            raise BenchmarkFailure(result.failure_code or result.message)
        if changed != expected:
            raise BenchmarkFailure(
                f"changed paths {changed!r} did not match allowed paths {expected!r}"
            )
        if not result.checks or not all(check.passed for check in result.checks):
            raise BenchmarkFailure("required real checks did not pass")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:800]
    evidence_sha = ""
    if result and result.evidence_path and Path(result.evidence_path).is_file():
        evidence_sha = hashlib.sha256(Path(result.evidence_path).read_bytes()).hexdigest()
    return {
        "case_id": spec["case_id"],
        "tier": spec["tier"],
        "depth": spec["depth"],
        "success": error is None,
        "attempts": result.attempts if result else 1,
        "hidden_retries": 0,
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "rounds": result.rounds_used if result else 0,
        "files_changed": sorted(result.files_changed) if result else [],
        "check_sha256": (
            result.checks[0].output_sha256 if result and result.checks else ""
        ),
        "evidence_sha256": evidence_sha,
        "input_sha256": canonical_digest(
            {"message": spec["message"], "files": spec["files"]}
        ),
        "error": error,
    }


async def async_main(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_info = ollama_model_info(args.base_url, args.model)
    run_id = uuid4().hex
    checkpoint = output_dir / f"native-coding-in-progress-{run_id}.json"
    existing_cases: list[dict[str, Any]] = []
    settings = benchmark_settings(args)
    if args.resume:
        resume = json.loads(args.resume.read_text(encoding="utf-8"))
        if (
            resume.get("model") != model_info
            or resume.get("requested_cases") != args.cases
            or resume.get("benchmark_settings") != settings
        ):
            raise BenchmarkFailure("resume checkpoint does not match this benchmark")
        existing_cases = list(resume.get("cases") or [])
        checkpoint = args.resume.resolve()
        run_id = str(resume.get("run_id") or run_id)
    completed_ids = {case["case_id"] for case in existing_cases}
    evidence_root = output_dir / "evidence" / run_id
    evidence_root.mkdir(parents=True, exist_ok=True)
    provider = OllamaProvider(
        model=args.model,
        base_url=args.base_url,
        temperature=0.0,
        max_tokens=args.max_tokens,
        think=OLLAMA_THINK,
    )
    agent = FlytoCodingAgent(provider, store=ThreadStore(str(evidence_root)))
    workspace_root = Path(tempfile.mkdtemp(prefix="flyto-native-coding-"))
    started = time.monotonic()
    try:
        cases = list(existing_cases)
        for ordinal, tier in enumerate(tier_sequence(args.cases), start=1):
            spec = case_spec(ordinal, tier)
            if spec["case_id"] in completed_ids:
                continue
            workspace = workspace_root / spec["case_id"]
            workspace.mkdir(parents=True)
            case = await run_case(
                agent=agent,
                workspace=workspace,
                spec=spec,
                case_timeout=args.case_timeout,
                max_agent_attempts=args.max_agent_attempts,
            )
            cases.append(case)
            atomic_json(
                checkpoint,
                {
                    "schema": "flyto.native-coding-benchmark.checkpoint.v1",
                    "run_id": run_id,
                    "model": model_info,
                    "requested_cases": args.cases,
                    "benchmark_settings": settings,
                    "cases": cases,
                },
            )
            print(
                f"[{len(cases):03d}/{args.cases}] {case['case_id']} "
                f"{'PASS' if case['success'] else 'FAIL'} "
                f"{case['duration_ms'] / 1000:.1f}s",
                flush=True,
            )
        cases.sort(key=lambda case: case["case_id"])
        family = summarize(cases, args.minimum_rate)
        report = {
            "schema": "flyto.native-coding-benchmark.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "family": "ordinary_development",
            "no_mocks": True,
            "provider": "ollama",
            "transport": OLLAMA_TRANSPORT,
            "think": OLLAMA_THINK,
            "model": model_info,
            "agent_contract": "flyto.coding.v1",
            "real_checks": ["python -m unittest -q"],
            "max_agent_attempts": args.max_agent_attempts,
            "max_completion_tokens": args.max_tokens,
            "attempts_reported_per_case": True,
            "requested_cases": args.cases,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "summary": family,
            "cases": cases,
            "ok": family["gate_pass"],
        }
        report["report_sha256"] = canonical_digest(report)
        destination = output_dir / f"native-coding-benchmark-{report['report_sha256']}.json"
        atomic_json(destination, report)
        checkpoint.unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "ok": report["ok"],
                    "report": str(destination),
                    "report_sha256": report["report_sha256"],
                    "summary": report["summary"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if report["ok"] else 1
    finally:
        if args.keep_workspaces:
            print(f"workspaces={workspace_root}")
        else:
            shutil.rmtree(workspace_root, ignore_errors=True)


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(async_main(args))
    except (BenchmarkFailure, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
