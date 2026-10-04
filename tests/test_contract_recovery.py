# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Recovery guidance comes only from declared capability contracts."""

from __future__ import annotations

import pytest

from flyto_ai import contract_recovery
from flyto_ai.contract_recovery import (
    GUIDANCE_VERSION,
    core_supports_recovery,
    declared_recovery,
    recovery_guidance,
    render_recovery_guidance,
)


def _contract(**extra):
    contract = {
        "schema": "flyto.capability-contract.v1",
        "actuates": True,
        "safety_class": "movement",
        "requires_safe_stop": True,
        "cancellable": True,
        "idempotent": False,
        "effects": [],
        "requires": [],
        "evidence": [],
    }
    contract.update(extra)
    return contract


def _module(capability, module_id=None, contract=None, **extra):
    module = {
        "module_id": module_id or capability,
        "provides_capability": capability,
        "contract": contract,
    }
    module.update(extra)
    return module


RECOVERY = {"substitutes": ["lift.reroute", "lift.wait"], "context": "door.state"}


def test_declared_recovery_is_normalized():
    declaration = declared_recovery(_contract(recovery=RECOVERY))
    assert declaration is not None
    assert declaration.substitutes == ("lift.reroute", "lift.wait")
    assert declaration.context == "door.state"
    assert declaration.to_dict() == {
        "substitutes": ["lift.reroute", "lift.wait"],
        "context": "door.state",
    }


def test_recovery_context_alias_and_duplicates():
    declaration = declared_recovery(
        _contract(
            recovery={
                "substitutes": ["lift.wait", "lift.wait"],
                "recovery_context": "door.state",
            }
        )
    )
    assert declaration.substitutes == ("lift.wait",)
    assert declaration.context == "door.state"


@pytest.mark.parametrize(
    "recovery",
    [
        None,
        "lift.wait",
        {"substitutes": "lift.wait"},
        {"substitutes": ["Not An Id"]},
        {"substitutes": [1]},
        {"substitutes": ["lift.wait"], "context": "Bad Context"},
        {"substitutes": [f"lift.s{i}" for i in range(17)]},
        {"substitutes": []},
        {},
    ],
)
def test_malformed_or_empty_declaration_fails_closed(recovery):
    contract = _contract() if recovery is None else _contract(recovery=recovery)
    assert declared_recovery(contract) is None


@pytest.mark.parametrize(
    "broken",
    [
        {"recovery": RECOVERY},
        {"actuates": "yes", "safety_class": "movement", "requires_safe_stop": True,
         "cancellable": True, "recovery": RECOVERY},
        {"actuates": True, "safety_class": "unknown", "requires_safe_stop": True,
         "cancellable": True, "recovery": RECOVERY},
    ],
)
def test_recovery_from_an_unreadable_contract_is_not_trusted(broken):
    assert declared_recovery(broken) is None
    guidance = recovery_guidance(
        "lift.move",
        [_module("lift.move", contract=broken), _module("lift.reroute", contract=_contract())],
    )
    assert guidance["source"] == "none"
    assert guidance["substitutes"] == []


def test_non_mapping_contract():
    assert declared_recovery(None) is None
    assert declared_recovery(["recovery"]) is None


def test_guidance_offers_only_installed_declared_substitutes():
    grade = {"level": "DANGER_FULL", "approval": "required"}
    candidates = [
        _module("lift.move", "vendor.lift.move", _contract(recovery=RECOVERY)),
        _module("lift.reroute", "vendor.lift.reroute", _contract(), grade=grade,
                plugin="vendor-lift", tool_name="vendor-lift__lift_reroute"),
        # Installed but not declared: never offered.
        _module("lift.override_brakes", "vendor.lift.override", _contract()),
    ]
    guidance = recovery_guidance("lift.move", candidates)
    assert guidance["version"] == GUIDANCE_VERSION
    assert guidance["source"] == "contract"
    assert guidance["context"] == "door.state"
    assert guidance["substitutes"] == [
        {
            "capability": "lift.reroute",
            "module_id": "vendor.lift.reroute",
            "plugin": "vendor-lift",
            "tool_name": "vendor-lift__lift_reroute",
            "grade": grade,
        }
    ]
    assert guidance["unavailable"] == ["lift.wait"]


def test_failed_capability_is_never_its_own_substitute():
    recovery = {"substitutes": ["lift.move"]}
    candidates = [_module("lift.move", contract=_contract(recovery=recovery))]
    guidance = recovery_guidance("lift.move", candidates)
    assert guidance["source"] == "contract"
    assert guidance["substitutes"] == []
    assert guidance["unavailable"] == []


def test_no_declaration_yields_empty_guidance():
    candidates = [
        _module("lift.move", contract=_contract()),
        _module("lift.reroute", contract=_contract()),
    ]
    guidance = recovery_guidance("lift.move", candidates)
    assert guidance["source"] == "none"
    assert guidance["substitutes"] == []
    text = render_recovery_guidance(guidance)
    assert "declares no recovery" in text
    assert "Do not invent" in text


def test_unknown_failed_capability():
    guidance = recovery_guidance("lift.unknown", [_module("lift.move")])
    assert guidance["source"] == "none"


def test_render_lists_substitutes_and_context():
    candidates = [
        _module("lift.move", contract=_contract(recovery=RECOVERY)),
        _module("lift.reroute", "vendor.lift.reroute", _contract()),
    ]
    text = render_recovery_guidance(recovery_guidance("lift.move", candidates))
    assert "lift.reroute (vendor.lift.reroute)" in text
    assert "own approval" in text
    assert "'door.state'" in text


def test_render_without_installed_substitute():
    recovery = {"substitutes": ["lift.wait"]}
    candidates = [_module("lift.move", contract=_contract(recovery=recovery))]
    text = render_recovery_guidance(recovery_guidance("lift.move", candidates))
    assert "No declared substitute is installed here." in text


def test_core_support_feature_detection(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("core.capability_contract")
    fake._REQUIRED_KEYS = frozenset({"actuates"})
    fake._OPTIONAL_KEYS = frozenset({"schema", "recovery"})
    monkeypatch.setitem(sys.modules, "core.capability_contract", fake)
    core_pkg = sys.modules.get("core")
    if core_pkg is not None:
        monkeypatch.setattr(core_pkg, "capability_contract", fake, raising=False)
    monkeypatch.setattr(contract_recovery, "_core_support", None)
    assert core_supports_recovery() is True

    fake._OPTIONAL_KEYS = frozenset({"schema"})
    monkeypatch.setattr(contract_recovery, "_core_support", None)
    assert core_supports_recovery() is False


def test_core_support_absent_core(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "core.capability_contract", None)
    monkeypatch.setattr(contract_recovery, "_core_support", None)
    assert core_supports_recovery() is False


def test_pack_groups_carry_declared_recovery_into_entries_and_tools():
    from flyto_ai.tools.pack_tools import build_pack_tools, get_pack_capability_groups

    move = {
        "module_id": "vendor.lift.move",
        "provides_capability": "lift.move",
        "plugin": "vendor-lift",
        "params_schema": {},
        "contract": _contract(actuates=False, safety_class="controlled",
                              requires_safe_stop=False, recovery=RECOVERY),
    }
    reroute = {
        "module_id": "vendor.lift.reroute",
        "provides_capability": "lift.reroute",
        "plugin": "vendor-lift",
        "params_schema": {},
        "contract": _contract(actuates=False, safety_class="controlled",
                              requires_safe_stop=False),
    }
    infos = {item["module_id"]: item for item in (move, reroute)}
    manifest = {
        "plugins": [{"id": "vendor-lift", "version": "1.0.0",
                     "module_ids": list(infos)}],
        "capabilities": [],
    }
    result = get_pack_capability_groups(
        manifest=manifest, module_info=infos.get, plugin_modules=lambda _p: None
    )
    modules = {m["provides_capability"]: m for m in result["groups"][0]["modules"]}
    assert modules["lift.move"]["recovery"] == {
        "substitutes": ["lift.reroute", "lift.wait"], "context": "door.state",
    }
    assert modules["lift.reroute"]["recovery"] is None

    guidance = recovery_guidance("lift.move", result["groups"][0]["modules"])
    assert [item["module_id"] for item in guidance["substitutes"]] == [
        "vendor.lift.reroute"
    ]

    tools = {t["name"]: t for t in build_pack_tools(result["groups"])["tools"]}
    move_tool = next(t for n, t in tools.items() if n.endswith("lift_move"))
    assert "may be replaced only by: lift.reroute, lift.wait" in move_tool["description"]
