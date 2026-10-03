# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Generic planning names no provider and filters by neutral resource fields."""

from __future__ import annotations

import ast
from pathlib import Path

from flyto_ai.capability_router import route_capabilities

ROOT = Path(__file__).resolve().parents[1] / "flyto_ai"
LEGACY_LAB_MODULES = {"robotics_planning.py", "robotics_planner_server.py"}


def _manifest(name: str, **extra: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "manifest_contract": "flyto.capability-manifest.v1",
        "canonical_id": f"lift.{name}@1",
        "runtime_name": name,
        "name": name,
        "version": "1.0.0",
        "domain": "lift",
        "description": f"Atomic {name} capability.",
        "tags": [],
        "aliases": [],
        "control_class": "motion",
        "required_observations": [],
        "required_resources": [],
        "required_permissions": [],
        "positive_examples": [],
        "negative_examples": [],
        "intent_ids": [],
        "affordances": [],
        "effects": [],
        "handled_events": [],
    }
    manifest.update(extra)
    return manifest


def _reasons(route: dict, name: str) -> list[str]:
    for item in route["excluded"]:
        if item["runtime_name"] == name:
            return list(item["reasons"])
    return []


def _candidate_names(route: dict) -> set[str]:
    return {item["runtime_name"] for item in route["candidates"]}


def test_neutral_resource_compatibility_filters_the_catalog():
    catalog = [
        _manifest("move_to_floor", compatible_resources=["lift-a"]),
        _manifest("open_doors", compatible_resources=["*"]),
    ]
    route = route_capabilities(
        "move to floor and open doors",
        catalog,
        context={"resource_model": "lift-b"},
    )
    assert "move_to_floor" not in _candidate_names(route)
    assert "resource_incompatible" in _reasons(route, "move_to_floor")
    assert "open_doors" in _candidate_names(route)


def test_legacy_lab_fields_are_read_as_aliases():
    catalog = [_manifest("move_to_floor", compatible_robots=["lift-a"])]
    route = route_capabilities(
        "move to floor", catalog, context={"robot_model": "lift-b"}
    )
    assert "resource_incompatible" in _reasons(route, "move_to_floor")

    allowed = route_capabilities(
        "move to floor", catalog, context={"robot_model": "lift-a"}
    )
    assert "move_to_floor" in _candidate_names(allowed)


def test_neutral_field_wins_over_legacy_alias():
    catalog = [
        _manifest(
            "move_to_floor",
            compatible_resources=["lift-a"],
            compatible_robots=["*"],
        )
    ]
    route = route_capabilities(
        "move to floor", catalog, context={"resource_model": "lift-b"}
    )
    assert "resource_incompatible" in _reasons(route, "move_to_floor")


def test_unsourced_manifest_is_never_attributed_by_prefix():
    # A provider-looking prefix must not turn into a named provider source.
    manifest = _manifest("move", canonical_id="robotics.motion.move@1")
    route = route_capabilities("move", [manifest])
    sources = {item["source"] for item in route["candidates"]}
    assert sources == {"external"}


def test_explicit_source_is_kept():
    manifest = _manifest("move", source="vendor.lift")
    route = route_capabilities("move", [manifest])
    assert {item["source"] for item in route["candidates"]} == {
        "vendor.lift"
    }


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(("." * node.level) + node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def test_no_generic_module_imports_the_legacy_lab_planner():
    offenders = []
    for path in ROOT.rglob("*.py"):
        if path.name in LEGACY_LAB_MODULES:
            continue
        for name in _imports(path):
            if name.endswith("robotics_planning") or name.endswith(
                "robotics_planner_server"
            ):
                offenders.append(f"{path.relative_to(ROOT)} -> {name}")
    assert offenders == []


def test_router_names_no_provider():
    source = (ROOT / "capability_router.py").read_text(encoding="utf-8")
    assert "flyto-robotics" not in source
    assert '"robot_incompatible"' not in source


def test_structured_provider_is_reexported_for_released_callers():
    from flyto_ai.robotics_planning import StructuredJsonProvider as legacy
    from flyto_ai.structured_provider import StructuredJsonProvider

    assert legacy is StructuredJsonProvider
