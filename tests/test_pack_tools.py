"""Per-package capability groups and pack tools."""

import importlib.util

import pytest

from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
from flyto_ai.tools.pack_tools import (
    PACK_TOOL_NAME_PATTERN,
    bind_resource_capabilities,
    build_pack_tools,
    get_pack_capability_groups,
    get_pack_tool_catalog,
    pack_tool_name,
    params_schema_to_json_schema,
    resolve_pack_tool_call,
)


def _contract(**overrides):
    contract = {
        "schema": "flyto.capability-contract.v1",
        "actuates": True,
        "safety_class": "movement",
        "requires_safe_stop": True,
        "cancellable": True,
        "idempotent": False,
    }
    contract.update(overrides)
    return contract


_DISTANCE = {"distance_m": {"type": "number", "min": 0.05, "max": 2.0, "required": True,
                            "description": "Distance in metres"}}

_MODULES = {
    # Two packages both provide motion.advance, for two different resources.
    "alpha.advance": {"plugin": "alpha-pack", "provides_capability": "motion.advance",
                      "params_schema": _DISTANCE, "contract": _contract()},
    "alpha.halt": {"plugin": "alpha-pack", "provides_capability": "motion.halt", "params_schema": {},
                   "contract": _contract(safety_class="controlled", requires_safe_stop=False,
                                         cancellable=False)},
    "alpha.observe": {"plugin": "alpha-pack", "provides_capability": "vision.observe", "params_schema": {},
                      "contract": _contract(actuates=False, safety_class="read_only",
                                            requires_safe_stop=False)},
    "beta.advance": {"plugin": "beta-pack", "provides_capability": "motion.advance",
                     "params_schema": {"distance_m": {"type": "number", "max": 0.5}},
                     "contract": None},
    "beta.helper": {"plugin": "beta-pack", "provides_capability": "", "params_schema": {}, "contract": None},
    # A module that claims a package it does not belong to in the manifest.
    "beta.spoof": {"plugin": "alpha-pack", "provides_capability": "motion.rotate", "params_schema": {},
                   "contract": _contract()},
    "browser.goto": {"plugin": "", "provides_capability": "", "params_schema": {}, "contract": None},
}

_MANIFEST = {
    "modules": sorted(_MODULES),
    "capabilities": [
        {"capability": "motion.advance", "providers": ["alpha.advance", "beta.advance"]},
        {"capability": "motion.halt", "providers": ["alpha.halt"]},
        {"capability": "vision.observe", "providers": ["alpha.observe"]},
    ],
    "plugins": [
        {"id": "beta-pack", "version": "0.2.0", "module_count": 3,
         "module_ids": ["beta.advance", "beta.helper", "beta.spoof"]},
        {"id": "alpha-pack", "version": "1.0.0", "module_count": 3, "description": "mobile base",
         "module_ids": ["alpha.advance", "alpha.halt", "alpha.observe"]},
    ],
}


def _groups(manifest=_MANIFEST):
    return get_pack_capability_groups(manifest=manifest, module_info=_MODULES.get)


def test_groups_are_per_package_with_per_module_contracts():
    result = _groups()
    assert result["ok"] is True
    assert [group["plugin"] for group in result["groups"]] == ["alpha-pack", "beta-pack"]
    alpha, beta = result["groups"]
    assert alpha["description"] == "mobile base"
    assert alpha["version"] == "1.0.0"
    assert [m["module_id"] for m in alpha["modules"]] == ["alpha.advance", "alpha.halt", "alpha.observe"]
    advance = alpha["modules"][0]
    assert set(advance) >= {"module_id", "provides_capability", "params_schema", "contract"}
    assert advance["contract"]["safety_class"] == "movement"
    assert advance["params_schema"] == _DISTANCE
    # Ownership comes from Core's registry, not from the manifest listing.
    assert [m["module_id"] for m in beta["modules"]] == ["beta.advance", "beta.helper"]


def test_core_modules_are_never_grouped():
    modules = [m["module_id"] for g in _groups()["groups"] for m in g["modules"]]
    assert "browser.goto" not in modules


def test_module_ids_fall_back_to_capability_providers():
    manifest = dict(_MANIFEST)
    manifest["plugins"] = [{"id": "alpha-pack", "version": "1.0.0", "module_count": 3}]
    result = _groups(manifest)
    assert [m["module_id"] for m in result["groups"][0]["modules"]] == [
        "alpha.advance", "alpha.halt", "alpha.observe",
    ]


def test_old_core_without_plugins_yields_no_groups():
    assert _groups({"modules": ["browser.goto"]})["groups"] == []
    assert get_pack_capability_groups(manifest_reader=lambda: None)["ok"] is False


def test_two_packs_providing_the_same_capability_bind_to_their_own_module():
    built = build_pack_tools(_groups()["groups"])
    index = built["index"]
    assert built["collisions"] == []
    assert index["alpha-pack__motion_advance"]["module_id"] == "alpha.advance"
    assert index["beta-pack__motion_advance"]["module_id"] == "beta.advance"

    alpha_resource = bind_resource_capabilities(index, ["motion.advance", "motion.halt"], pack="alpha-pack")
    assert [b["tool_name"] for b in alpha_resource["bound"]] == [
        "alpha-pack__motion_advance", "alpha-pack__motion_halt",
    ]
    beta_resource = bind_resource_capabilities(index, ["motion.advance", "motion.halt"], pack="beta-pack")
    assert [b["module_id"] for b in beta_resource["bound"]] == ["beta.advance"]
    assert beta_resource["missing"] == ["motion.halt"]

    unscoped = bind_resource_capabilities(index, ["motion.advance", "vision.observe"])
    assert unscoped["ambiguous"] == [{"capability": "motion.advance", "plugins": ["alpha-pack", "beta-pack"]}]
    assert [b["capability"] for b in unscoped["bound"]] == ["vision.observe"]

    alpha_call = resolve_pack_tool_call(
        "alpha-pack__motion_advance", {"resource_id": "r-1", "arguments": {"distance_m": 0.3}}, index,
    )
    beta_call = resolve_pack_tool_call(
        "beta-pack__motion_advance", {"resource_id": "r-2", "arguments": {"distance_m": 0.3}}, index,
    )
    assert (alpha_call["plugin"], alpha_call["module_id"], alpha_call["resource_id"]) == (
        "alpha-pack", "alpha.advance", "r-1")
    assert (beta_call["plugin"], beta_call["module_id"], beta_call["resource_id"]) == (
        "beta-pack", "beta.advance", "r-2")


def test_pack_tools_are_graded_by_contract():
    built = build_pack_tools(_groups()["groups"])
    index = built["index"]
    assert index["alpha-pack__motion_advance"]["permission_level"] == "DANGER_FULL"
    assert index["alpha-pack__motion_advance"]["risk_level"] == 4
    # No contract from a non-core package: fail closed.
    assert index["beta-pack__motion_advance"]["permission_level"] == "DANGER_FULL"
    assert index["beta-pack__motion_advance"]["actuating"] is True
    assert index["alpha-pack__vision_observe"]["permission_level"] == "READ_ONLY"
    # The stop keeps its immediate path.
    halt = index["alpha-pack__motion_halt"]
    assert halt["immediate"] is True and halt["permission_level"] == "READ_ONLY"
    tool = next(t for t in built["tools"] if t["name"] == "alpha-pack__motion_halt")
    assert tool["_meta"]["flyto2/immediate"] is True


def test_permission_overrides_make_actuating_pack_tools_ask():
    catalog = get_pack_tool_catalog(manifest=_MANIFEST, module_info=_MODULES.get)
    overrides = catalog["permission_overrides"]
    assert overrides["alpha-pack__motion_advance"] == PermissionLevel.DANGER_FULL
    enforcer = PermissionEnforcer(PermissionLevel.WORKSPACE_WRITE, overrides=overrides)
    advance = enforcer.check_route("alpha-pack__motion_advance", {"resource_id": "r"}, "action")
    assert advance.allowed is False
    # A stop is available even on an ambiguous turn.
    halt = enforcer.check_route("alpha-pack__motion_halt", {"resource_id": "r"}, "ambiguous")
    assert halt.allowed is True


def test_tool_shape_takes_resource_and_arguments():
    built = build_pack_tools(_groups()["groups"])
    tool = next(t for t in built["tools"] if t["name"] == "alpha-pack__motion_advance")
    schema = tool["inputSchema"]
    assert schema["required"] == ["resource_id"]
    arguments = schema["properties"]["arguments"]
    assert arguments["properties"]["distance_m"] == {
        "type": "number", "description": "Distance in metres", "minimum": 0.05, "maximum": 2.0,
    }
    assert arguments["required"] == ["distance_m"]
    assert tool["_meta"]["flyto2/source"] == {
        "type": "module_pack", "plugin": "alpha-pack", "module_id": "alpha.advance",
        "capability": "motion.advance",
    }
    # Modules without a declared capability get no tool.
    assert not any("helper" in t["name"] for t in built["tools"])


def test_call_resolution_refuses_rather_than_repairs():
    index = build_pack_tools(_groups()["groups"])["index"]
    # Out-of-bounds values pass through untouched: the host refuses, never clamps.
    call = resolve_pack_tool_call(
        "alpha-pack__motion_advance", {"resource_id": "r", "arguments": {"distance_m": 9.0}}, index,
    )
    assert call["ok"] is True and call["arguments"] == {"distance_m": 9.0}
    assert resolve_pack_tool_call("nope__x", {"resource_id": "r"}, index)["ok"] is False
    assert resolve_pack_tool_call("alpha-pack__motion_advance", {"arguments": {}}, index)["ok"] is False
    assert resolve_pack_tool_call(
        "alpha-pack__motion_advance", {"resource_id": "r", "arguments": []}, index,
    )["ok"] is False
    assert resolve_pack_tool_call(
        "alpha-pack__motion_advance", {"resource_id": "r", "speed": 1}, index,
    )["ok"] is False


def test_names_are_bounded_and_collisions_are_withheld():
    assert pack_tool_name("flyto-modules-robotics", "motion.advance") == "flyto-modules-robotics__motion_advance"
    long_name = pack_tool_name("p" * 80, "very.long.capability.identifier.name")
    assert len(long_name) <= 64 and PACK_TOOL_NAME_PATTERN.fullmatch(long_name)
    assert long_name != pack_tool_name("p" * 81, "very.long.capability.identifier.name")
    assert "__" not in pack_tool_name("a__b", "x").split("__", 1)[0]

    groups = [
        {"plugin": "dup", "modules": [
            {"module_id": "dup.one", "provides_capability": "thing.do", "contract": _contract()},
            {"module_id": "dup.two", "provides_capability": "thing_do", "contract": _contract()},
            {"module_id": "dup.three", "provides_capability": "thing.other", "contract": _contract()},
        ]},
        {"plugin": "pack.x", "modules": [
            {"module_id": "x.a", "provides_capability": "go", "contract": _contract()}]},
        {"plugin": "pack_x", "modules": [
            {"module_id": "y.a", "provides_capability": "go", "contract": _contract()}]},
    ]
    built = build_pack_tools(groups)
    assert sorted(built["index"]) == ["dup__thing_other"]
    assert sorted(c["name"] for c in built["collisions"]) == ["dup__thing_do", "pack_x__go"]


def test_params_schema_projection_ignores_unknown_shapes():
    assert params_schema_to_json_schema(None) == {
        "type": "object", "properties": {}, "additionalProperties": False,
    }
    schema = params_schema_to_json_schema({"mode": {"type": "select", "options": ["a", "b"]}, 3: {}})
    assert schema["properties"] == {"mode": {"enum": ["a", "b"]}}


@pytest.mark.skipif(
    importlib.util.find_spec("core.capability_contract") is None,
    reason="installed flyto-core predates capability contracts",
)
def test_real_core_registry_groups_contracted_pack_modules():
    from core.modules.registry import ModuleRegistry
    from core.modules.registry.core import PluginInfo

    ModuleRegistry.get_plugins()  # finish discovery before borrowing the registry

    class _Module:
        pass

    def _register(plugin, module_id, capability, contract):
        ModuleRegistry._loading_plugin = plugin
        try:
            ModuleRegistry.register(module_id, _Module, {
                "module_id": module_id, "provides_capability": capability,
                "params_schema": {}, "contract": contract,
            })
        finally:
            ModuleRegistry._loading_plugin = ""

    added = ["zzalpha.advance", "zzbeta.advance"]
    try:
        _register("zz-alpha", "zzalpha.advance", "zzmotion.advance", _contract())
        _register("zz-beta", "zzbeta.advance", "zzmotion.advance", _contract(safety_class="dangerous"))
        for name in ("zz-alpha", "zz-beta"):
            ModuleRegistry._plugins[name] = PluginInfo(
                name=name, version="0.0.1", module_count=1, entry_point="x:y",
            )
        ModuleRegistry._bump_generation()

        catalog = get_pack_tool_catalog()
        assert catalog["ok"] is True
        groups = {g["plugin"]: g for g in catalog["groups"]}
        assert [m["module_id"] for m in groups["zz-alpha"]["modules"]] == ["zzalpha.advance"]
        assert groups["zz-beta"]["modules"][0]["contract"]["safety_class"] == "dangerous"
        index = catalog["index"]
        assert index["zz-alpha__zzmotion_advance"]["risk_level"] == 4
        assert index["zz-beta__zzmotion_advance"]["risk_level"] == 5
        assert PermissionEnforcer().required_level(
            "execute_module", {"module_id": "zzalpha.advance"},
        ) == PermissionLevel.DANGER_FULL
    finally:
        for module_id in added:
            ModuleRegistry.unregister(module_id)
        for name in ("zz-alpha", "zz-beta"):
            ModuleRegistry._plugins.pop(name, None)
        ModuleRegistry._bump_generation()
