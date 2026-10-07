# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Every workflow job that runs this repository's tests installs the package.

`tests/conftest.py` imports `flyto_ai`, so pytest cannot even collect a single
test file -- however self-contained that file is -- unless the package's
declared base dependencies are installed. The advisory-freshness job said
"install pytest only" and was red on every scheduled run from the day it
landed, with `ModuleNotFoundError: No module named 'pydantic'` before the gate
it exists to run was ever reached. The dependency set lives in
`pyproject.toml`; a job gets it by installing this checkout, never by
restating packages.
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

_RUNS_PYTEST = re.compile(r"(?:^|\s)(?:python -m )?pytest(?:\s|$)")
# `pip install` of this checkout: `.`, `-e .`, `".[dev]"`, or the checkout's
# directory when the workflow uses `actions/checkout` with `path: flyto-ai`.
_INSTALLS_SELF = re.compile(
    r"pip install\b[^\n]*?(?:\s|^)(?:-e\s+)?"
    r"[\"']?(?:\.|\./flyto-ai)(?:\[[^\]]*\])?[\"']?(?=\s|$)",
)


def _jobs():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job_id, job in (document.get("jobs") or {}).items():
            yield path.name, job_id, job.get("steps") or []


def _pytest_jobs_without_self_install():
    missing = []
    for workflow, job_id, steps in _jobs():
        installed = False
        for step in steps:
            run = step.get("run") or ""
            if _INSTALLS_SELF.search(run):
                installed = True
            if _RUNS_PYTEST.search(run) and not installed:
                missing.append("{}:{}".format(workflow, job_id))
                break
    return missing


def test_every_job_running_pytest_installs_the_package_first() -> None:
    assert _pytest_jobs_without_self_install() == []


def test_the_rule_sees_at_least_ci_and_advisory_freshness() -> None:
    # Guard the guard: if the parser stopped recognising pytest steps, the
    # assertion above would pass vacuously.
    seen = {
        workflow
        for workflow, _job, steps in _jobs()
        for step in steps
        if _RUNS_PYTEST.search(step.get("run") or "")
    }
    assert {"ci.yml", "advisory-freshness.yml"} <= seen


def test_the_rule_rejects_a_pytest_only_install() -> None:
    assert not _INSTALLS_SELF.search("python -m pip install --upgrade pip pytest")
    assert _INSTALLS_SELF.search("python -m pip install --upgrade pip ./flyto-ai pytest")
    assert _INSTALLS_SELF.search('.venv/bin/python -m pip install -e ".[dev]"')
    assert not _INSTALLS_SELF.search("pip install -e ../flyto-core")
