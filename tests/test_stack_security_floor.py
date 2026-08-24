# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Every declared `flyto-core` floor must clear every published Core advisory.

Core already computes the number this needs. `flyto-core/security/advisories.json`
is the machine-readable manifest its `SECURITY_STATUS.md` is generated from, and
the lowest version at which every advisory is fixed falls straight out of it.
Until now that number lived only in generated prose, so nothing downstream could
read it, and both this package and Blueprint carried floors — `>=2.16.1` and
`>=2.12.0` — that predated all 33 of them. Neither was a decision; both were
just never revisited, and no check could have said so.

This repository is the only place in the stack that can check it. `stack-lock.json`
makes CI check out Blueprint, Core and the Indexer beside this one at pinned
revisions, so the whole trio is on disk here and nowhere else. The test therefore
covers Blueprint's floors too: the gap it closes is not per-repository.

Skips rather than passes when the sibling checkout is absent, which is the normal
state of a bare `pip install -e .` clone. A skip says "not checked here"; a pass
would say "checked and fine", and that is the exact substitution this whole class
of defect was made of.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
ADVISORIES = WORKSPACE / "flyto-core" / "security" / "advisories.json"
# Repository name -> the pyproject whose flyto-core floors must hold.
DEPENDENT_PYPROJECTS = {
    "flyto-ai": Path(__file__).resolve().parents[1] / "pyproject.toml",
    "flyto-blueprint": WORKSPACE / "flyto-blueprint" / "pyproject.toml",
}
# `flyto-core`, optionally with extras, then a `>=` floor. A requirement that
# pins some other way (`==`, `~=`, no specifier) is reported rather than parsed
# loosely: an unrecognised form here means the floor is not being checked, and
# silence about that is how the original gap survived.
REQUIREMENT = re.compile(r"""["']flyto-core(?P<extras>\[[^\]]*\])?(?P<spec>[^"']*)["']""")
FLOOR = re.compile(r"^>=\s*(?P<version>\d+(?:\.\d+)*)$")
DOMAIN_CAPABILITY_FLOOR = "2.31.0"


def _version_key(version: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in version.split(".") if part.isdigit())


def _fully_patched_from() -> str:
    """Lowest Core version at which every published advisory is fixed."""
    import json

    advisories = json.loads(ADVISORIES.read_text(encoding="utf-8"))
    patched = {
        item["patched"].lstrip(">= ").strip()
        for item in advisories
        if item.get("patched") not in (None, "-", "")
    }
    assert patched, "advisories.json declares no patched versions"
    return max(patched, key=_version_key)


def _core_requirements(pyproject: Path) -> List[str]:
    return [
        match.group("spec").strip()
        for match in REQUIREMENT.finditer(pyproject.read_text(encoding="utf-8"))
    ]


def _present_dependents() -> Dict[str, Path]:
    return {name: path for name, path in DEPENDENT_PYPROJECTS.items() if path.exists()}


@pytest.fixture(scope="module")
def floor() -> str:
    if not ADVISORIES.exists():
        pytest.skip(
            "flyto-core is not checked out beside this repository; "
            "the Core advisory floor cannot be read"
        )
    return _fully_patched_from()


def test_every_declared_core_floor_clears_every_advisory(floor: str) -> None:
    dependents = _present_dependents()
    assert "flyto-ai" in dependents, "this repository's own pyproject must be readable"

    failures = []
    checked = 0
    for name, pyproject in dependents.items():
        for spec in _core_requirements(pyproject):
            matched = FLOOR.match(spec)
            if matched is None:
                failures.append(
                    f"{name}: flyto-core requirement {spec!r} has no `>=` floor, "
                    "so no advisory floor is being enforced for it"
                )
                continue
            checked += 1
            declared = matched.group("version")
            if _version_key(declared) < _version_key(floor):
                failures.append(
                    f"{name}: flyto-core floor >={declared} predates {floor}, the "
                    "lowest version clearing every published Core advisory"
                )

    assert checked, "no flyto-core requirement was found to check"
    assert not failures, "\n".join(failures)


def test_the_floor_is_read_from_core_rather_than_restated(floor: str) -> None:
    """The number must come from Core's manifest, not a copy kept here.

    A constant in this repository would be a second source of truth that goes
    stale the moment Core publishes advisory 34 — the same shape as every other
    finding this file exists because of.
    """
    assert ADVISORIES.is_file()
    assert _version_key(floor) >= (2, 28, 1)


def test_browser_full_and_dev_cannot_resolve_below_domain_capability_floor() -> None:
    """Core surfaces keep their exact extras and include the three solvers."""
    import tomllib

    pyproject = DEPENDENT_PYPROJECTS["flyto-ai"]
    extras = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    expected_core_extras = {
        "browser": {"browser"},
        "full": {"browser"},
        "dev": {"browser", "api"},
    }
    for extra, expected_extras in expected_core_extras.items():
        requirements = [
            requirement
            for requirement in extras[extra]
            if requirement.startswith("flyto-core")
        ]
        assert len(requirements) == 1, f"{extra} must declare exactly one Core floor"
        matched = REQUIREMENT.fullmatch(f'"{requirements[0]}"')
        assert matched is not None
        declared_extras = {
            item.strip()
            for item in (matched.group("extras") or "")[1:-1].split(",")
            if item.strip()
        }
        assert declared_extras == expected_extras, (
            f"{extra} must declare exactly Core extras {sorted(expected_extras)}"
        )
        floor_match = FLOOR.fullmatch(matched.group("spec").strip())
        assert floor_match is not None
        assert _version_key(floor_match.group("version")) >= _version_key(
            DOMAIN_CAPABILITY_FLOOR
        )
