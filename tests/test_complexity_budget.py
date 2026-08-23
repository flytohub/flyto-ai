# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""A ratchet on module size and parameter count. It may only tighten.

This repository's own quality rules ask for atomic units, functions that stay
simple, and modules that get split. The code does not meet them in places:
`coding/service.py` is 8,227 lines with 184 methods, and `CodingService.__init__`
takes 17 parameters. Nothing enforced those rules, so the numbers drifted upward
one reasonable-looking commit at a time, and no single commit was ever the one
that made it bad.

A threshold alone cannot fix that: set it at today's worst and it permits every
future file to be that bad, set it at the target and the suite is red until a
multi-week refactor lands. So this records the debt instead. Every file over
`FILE_LINE_BUDGET` and every function over `PARAMETER_BUDGET` is listed in
`complexity_baseline.json` with its current number, and the rules are:

* a file or function not in the baseline may not exceed the budget at all
* one that is in the baseline may not exceed its recorded number

New code is therefore held to the rule immediately, while existing debt is
visible, counted, and can only be paid down. `scripts/update_complexity_baseline.py`
refuses to raise a recorded number, so the baseline cannot be quietly re-cut
around a regression.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "flyto_ai"
BASELINE = Path(__file__).resolve().parent / "complexity_baseline.json"

FILE_LINE_BUDGET = 800
PARAMETER_BUDGET = 8


def measure(package: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return every file over the line budget and function over the parameter budget."""
    files: Dict[str, int] = {}
    functions: Dict[str, int] = {}
    for path in sorted(package.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT).as_posix()
        line_count = len(text.splitlines())
        if line_count > FILE_LINE_BUDGET:
            files[relative] = line_count
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            # `self`/`cls` are counted: a method with fifteen collaborators is
            # not made simpler by the first one being implicit.
            count = len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
            if count > PARAMETER_BUDGET:
                functions[f"{relative}::{node.name}"] = count
    return files, functions


def _load_baseline() -> Tuple[Dict[str, int], Dict[str, int]]:
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    return data["files"], data["functions"]


def _regressions(current: Dict[str, int], baseline: Dict[str, int], budget: int, unit: str):
    problems = []
    for name, value in sorted(current.items()):
        allowed = baseline.get(name)
        if allowed is None:
            problems.append(
                f"{name}: {value} {unit} exceeds the budget of {budget} and is not "
                "recorded debt. Split it, or record it deliberately with "
                "scripts/update_complexity_baseline.py --accept-new."
            )
        elif value > allowed:
            problems.append(
                f"{name}: {value} {unit}, worse than the recorded {allowed}. "
                "The baseline only moves down."
            )
    return problems


def test_no_module_or_signature_grows_past_its_recorded_debt() -> None:
    files, functions = measure(PACKAGE)
    baseline_files, baseline_functions = _load_baseline()

    problems = _regressions(files, baseline_files, FILE_LINE_BUDGET, "lines")
    problems += _regressions(
        functions, baseline_functions, PARAMETER_BUDGET, "parameters"
    )
    assert not problems, "\n".join(problems)


def test_baseline_records_no_entry_that_is_already_clean() -> None:
    """A baseline that outlives its debt stops describing anything.

    Without this, an entry paid down to zero stays in the file forever and the
    ratchet silently allows it to grow back to the recorded number.
    """
    files, functions = measure(PACKAGE)
    baseline_files, baseline_functions = _load_baseline()

    stale = [name for name in baseline_files if name not in files]
    stale += [name for name in baseline_functions if name not in functions]
    assert not stale, (
        "these are no longer over budget and must be dropped from "
        f"complexity_baseline.json: {', '.join(sorted(stale))}"
    )
