# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Tests for the flyto-core MCP contract exposed by flyto-ai."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from flyto_ai.mcp_client import McpClientManager
from flyto_ai.mcp_server import (
    CLIENT_CAPABILITIES_META_KEY,
    MODERN_PROTOCOL_VERSION,
    PROTOCOL_VERSION_META_KEY,
)
from flyto_ai.tools import core_tools
from flyto_ai.providers.base import dispatch_and_log_tool


def _fake_handler(*, validation_result=None, execute_result=None):
    calls = {"execute": 0, "validate": 0}

    def list_modules(category=None):
        return {"categories": [{"name": "browser", "count": 2}, {"name": "string", "count": 1}]}

    def validate_params(module_id, params):
        calls["validate"] += 1
        if validation_result is not None:
            return validation_result
        return {"valid": True, "errors": []}

    async def execute_module(module_id, params, context, browser_sessions):
        calls["execute"] += 1
        return execute_result or {"ok": True, "module_id": module_id, "data": params}

    return {
        "TOOLS": [
            {
                "name": "list_modules",
                "description": "List modules",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "execute_module",
                "description": "Execute module",
                "inputSchema": {"type": "object", "properties": {"module_id": {"type": "string"}}},
            },
        ],
        "list_modules": list_modules,
        "validate_params": validate_params,
        "execute_module": execute_module,
        "_calls": calls,
    }


def _capability_manifest(**overrides):
    """A schema-valid installed-capability manifest in Core's real wire shape.

    This mirrors `core.capability_manifest.get_capability_manifest`: the
    contract and digest are stamped under `schema` and `hash`, `modules` is a
    list of bare module-id strings, each `capabilities` record is a
    `capability` plus the `providers` that supply it, and each `plugins` record
    carries `id`, `version`, and `module_count`.
    """
    manifest = {
        "schema": core_tools.CORE_CAPABILITY_MANIFEST_CONTRACT,
        "hash": "a" * 64,
        "module_count": 2,
        "capability_count": 1,
        "plugin_count": 1,
        "modules": ["browser.launch", "string.uppercase"],
        "capabilities": [
            {
                "capability": "capability.browse@1",
                "providers": ["browser.launch"],
            },
        ],
        "plugins": [
            {"id": "core-browser", "version": "2.26.11", "module_count": 1},
        ],
    }
    manifest.update(overrides)
    return manifest


def _with_core_manifest(monkeypatch, reader):
    monkeypatch.setattr(core_tools, "_get_core_capability_manifest_fn", lambda: reader)


def test_installed_module_ids_none_only_for_absent_or_old_core(monkeypatch):
    _with_core_manifest(monkeypatch, None)

    assert core_tools.get_core_installed_module_ids() is None


def test_installed_module_ids_exclude_capability_and_plugin_ids(monkeypatch):
    _with_core_manifest(monkeypatch, lambda: _capability_manifest())

    module_ids = core_tools.get_core_installed_module_ids()

    assert module_ids == frozenset({"browser.launch", "string.uppercase"})
    assert isinstance(module_ids, frozenset)
    # A capability names what a module provides and a plugin names who ships
    # it. Neither is executable, so neither may reach a Blueprint engine.
    assert "capability.browse@1" not in module_ids
    assert "core-browser" not in module_ids


def test_valid_empty_manifest_is_empty_frozenset_not_none(monkeypatch):
    _with_core_manifest(monkeypatch, lambda: _capability_manifest(
        module_count=0,
        capability_count=0,
        plugin_count=0,
        modules=[],
        capabilities=[],
        plugins=[],
    ))

    assert core_tools.get_core_installed_module_ids() == frozenset()


@pytest.mark.parametrize(
    "manifest",
    [
        pytest.param("not-a-mapping", id="not_a_mapping"),
        pytest.param({}, id="no_schema"),
        pytest.param(_capability_manifest(schema="flyto-core.other.v9"), id="wrong_schema"),
        pytest.param(_capability_manifest(hash=""), id="missing_hash"),
        pytest.param(_capability_manifest(hash="not-a-digest"), id="malformed_hash"),
        pytest.param(_capability_manifest(module_count=99), id="module_count_mismatch"),
        pytest.param(_capability_manifest(capability_count=99), id="capability_count_mismatch"),
        pytest.param(_capability_manifest(plugin_count=99), id="plugin_count_mismatch"),
        pytest.param(_capability_manifest(modules="browser.launch"), id="modules_not_a_list"),
        pytest.param(
            _capability_manifest(
                module_count=1,
                modules=[{"module_id": "browser.launch"}],
                capabilities=[],
                capability_count=0,
            ),
            id="modules_are_records_not_strings",
        ),
        pytest.param(
            _capability_manifest(
                capabilities=[
                    {
                        "capability": "capability.browse@1",
                        "providers": ["ghost.module"],
                    },
                ],
            ),
            id="capability_claims_uninstalled_provider",
        ),
        pytest.param(
            _capability_manifest(
                capabilities=[
                    {"capability": "capability.browse@1", "providers": []},
                ],
            ),
            id="capability_without_providers",
        ),
        pytest.param(
            _capability_manifest(
                capabilities=[{"capability_id": "capability.browse@1"}],
            ),
            id="capability_uses_legacy_key",
        ),
        pytest.param(
            _capability_manifest(plugins=[{"plugin": "core-browser"}]),
            id="plugin_uses_legacy_key",
        ),
        pytest.param(
            _capability_manifest(plugins=[{"id": "core-browser", "module_count": 1}]),
            id="plugin_without_version",
        ),
        pytest.param(
            _capability_manifest(
                plugins=[{"id": "core-browser", "version": "2.26.11"}],
            ),
            id="plugin_without_module_count",
        ),
        pytest.param(
            _capability_manifest(
                module_count=1,
                modules=["not a safe id"],
                capabilities=[],
                capability_count=0,
            ),
            id="unsafe_module_identity",
        ),
        pytest.param(
            _capability_manifest(
                module_count=1,
                modules=["browser.launch", "browser.launch"],
            ),
            id="duplicate_module_identity",
        ),
        pytest.param(_capability_manifest(ok=False), id="core_reports_failure"),
    ],
)
def test_malformed_manifest_from_new_core_is_empty_frozenset(monkeypatch, manifest):
    _with_core_manifest(monkeypatch, lambda: manifest)

    assert core_tools.get_core_installed_module_ids() == frozenset()


def test_failing_new_core_reader_is_empty_frozenset(monkeypatch):
    def boom():
        raise RuntimeError("core registry unavailable")

    _with_core_manifest(monkeypatch, boom)

    assert core_tools.get_core_installed_module_ids() == frozenset()


def test_public_manifest_reports_installed_capability_provenance(monkeypatch):
    handler = _fake_handler()
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: handler)
    _with_core_manifest(monkeypatch, lambda: _capability_manifest())

    manifest = core_tools.get_core_capability_manifest()
    installed = manifest["installed_capabilities"]

    assert installed["supported"] is True
    assert installed["status"] == "ok"
    assert installed["contract"] == core_tools.CORE_CAPABILITY_MANIFEST_CONTRACT
    assert installed["manifest_hash"] == "a" * 64
    assert installed["module_count"] == 2
    assert installed["capability_count"] == 1
    assert installed["plugin_count"] == 1
    # Identities stay host side; only counts and Core's digest are published.
    assert "capability_ids" not in installed
    assert "module_ids" not in installed


def test_public_manifest_marks_old_core_unsupported(monkeypatch):
    handler = _fake_handler()
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: handler)
    _with_core_manifest(monkeypatch, None)

    installed = core_tools.get_core_capability_manifest()["installed_capabilities"]

    assert installed["supported"] is False
    assert installed["status"] == "unsupported_core"
    assert installed["capability_count"] == 0


def test_public_manifest_marks_malformed_core_invalid(monkeypatch):
    handler = _fake_handler()
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: handler)
    _with_core_manifest(monkeypatch, lambda: {"schema": "wrong"})

    installed = core_tools.get_core_capability_manifest()["installed_capabilities"]

    assert installed["supported"] is True
    assert installed["status"] == "invalid"
    assert installed["capability_count"] == 0


@pytest.fixture
def real_core_manifest():
    """Bind the *installed* Core's manifest reader, bypassing every fixture.

    Fixtures encode what the host believes Core emits, so a fixture can agree
    with a wrong belief forever. This resolves `core.capability_manifest`
    itself, resets the module-level bind cache around the test, and restores it
    afterwards so no later test inherits a real binding.
    """
    saved = (
        core_tools._cached_capability_manifest_contract,
        core_tools._capability_manifest_contract_checked,
    )
    core_tools._cached_capability_manifest_contract = None
    core_tools._capability_manifest_contract_checked = False
    try:
        contract = core_tools._get_core_capability_manifest_contract()
        if contract is None:
            pytest.skip("flyto-core is absent or predates the capability manifest")
        try:
            raw = contract[0]()
        except Exception as e:  # pragma: no cover - depends on local install
            pytest.skip("installed flyto-core cannot report a manifest: {}".format(e))
        yield raw
    finally:
        (
            core_tools._cached_capability_manifest_contract,
            core_tools._capability_manifest_contract_checked,
        ) = saved


def test_real_installed_core_manifest_passes_host_validation(real_core_manifest):
    """A real, non-empty Core must never validate down to an empty set.

    This is the regression that fixtures cannot catch: every host-side key
    name, entry shape, cross-check, and count rule is exercised against the
    manifest the installed Core actually emits. If Core reports modules and
    this host reports none, the bridge is broken even though every fixture
    still passes.
    """
    declared = real_core_manifest.get("module_count")
    assert isinstance(declared, int) and not isinstance(declared, bool)
    if declared == 0:
        pytest.skip("installed flyto-core reports no modules")

    module_ids = core_tools.get_core_installed_module_ids()

    assert module_ids is not None, "a manifest-capable Core must not report None"
    assert module_ids, (
        "installed flyto-core declares {} modules but host validation "
        "produced an empty set".format(declared)
    )
    assert len(module_ids) == declared


def test_real_installed_core_provenance_counts_are_exact(real_core_manifest):
    if not real_core_manifest.get("module_count"):
        pytest.skip("installed flyto-core reports no modules")

    _module_ids, summary = core_tools._read_core_installed_module_ids()

    assert summary["status"] == "ok"
    assert summary["supported"] is True
    assert summary["module_count"] == real_core_manifest["module_count"]
    assert summary["capability_count"] == real_core_manifest["capability_count"]
    assert summary["plugin_count"] == real_core_manifest["plugin_count"]
    assert summary["manifest_hash"] == real_core_manifest["hash"].strip().lower()


def test_real_installed_core_module_ids_exclude_capabilities_and_plugins(
    real_core_manifest,
):
    if not real_core_manifest.get("module_count"):
        pytest.skip("installed flyto-core reports no modules")

    module_ids = core_tools.get_core_installed_module_ids()
    declared_modules = frozenset(real_core_manifest["modules"])
    capability_ids = {
        entry["capability"] for entry in real_core_manifest["capabilities"]
    }
    plugin_ids = {entry["id"] for entry in real_core_manifest["plugins"]}

    assert module_ids == declared_modules
    assert not module_ids & (capability_ids - declared_modules)
    assert not module_ids & (plugin_ids - declared_modules)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query,module_id,capability_id,intent,affordance",
    [
        (
            "solve.rigid-transform-3d transform.point-3d data.compute-only domain.solve.requested",
            "math.rigid_transform_3d",
            "domain.solve.rigid-transform-3d",
            "solve.rigid-transform-3d",
            "transform.point-3d",
        ),
        (
            "solve.constant-acceleration-kinematics compute.position-velocity data.compute-only domain.solve.requested",
            "physics.kinematics_constant_acceleration",
            "domain.solve.constant-acceleration-kinematics",
            "solve.constant-acceleration-kinematics",
            "compute.position-velocity",
        ),
        (
            "solve.ideal-dilution compute.stock-diluent-volume data.compute-only domain.solve.requested",
            "chemistry.ideal_dilution",
            "domain.solve.ideal-dilution",
            "solve.ideal-dilution",
            "compute.stock-diluent-volume",
        ),
    ],
)
async def test_real_core_bridge_exposes_exact_nested_solver_semantics(
    query, module_id, capability_id, intent, affordance
):
    result = await core_tools.dispatch_core_tool(
        "search_modules", {"query": query, "limit": 100}
    )
    hit = next(item for item in result["results"] if item["module_id"] == module_id)
    assert hit["provides_capability"] == capability_id
    assert hit["semantics"] == {
        "intent_ids": [intent],
        "affordances": [affordance],
        "effects": ["data.compute-only"],
        "handled_events": ["domain.solve.requested"],
    }


def test_core_tool_defs_include_manifest_and_metadata(monkeypatch):
    handler = _fake_handler()
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: handler)

    tools = core_tools.get_core_tool_defs()
    names = {t["name"] for t in tools}

    assert "get_core_capability_manifest" in names
    execute_tool = next(t for t in tools if t["name"] == "execute_module")
    assert execute_tool["metadata"]["source"] == "flyto-core"
    assert execute_tool["metadata"]["contract_version"] == core_tools.CORE_MCP_CONTRACT_VERSION
    assert execute_tool["metadata"]["approval_policy"] == "module_category_runtime"
    assert execute_tool["annotations"]["destructiveHint"] is True


def test_core_manifest_has_fingerprint_categories_and_approval_model(monkeypatch):
    handler = _fake_handler()
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: handler)

    manifest = core_tools.get_core_capability_manifest(include_tools=True, include_categories=True)

    assert manifest["ok"] is True
    assert manifest["source"] == "flyto-core"
    assert manifest["contract_version"] == core_tools.CORE_MCP_CONTRACT_VERSION
    assert manifest["tool_count"] == 3
    assert len(manifest["tool_fingerprint"]) == 16
    assert manifest["module_count"] == 3
    assert "runtime secrets only" in manifest["approval_model"]["sensitive_inputs"]
    assert any(t["name"] == "execute_module" for t in manifest["tools"])


@pytest.mark.asyncio
async def test_manifest_tool_dispatch(monkeypatch):
    handler = _fake_handler()
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: handler)

    result = await core_tools.dispatch_core_tool(
        "get_core_capability_manifest",
        {"include_tools": False, "include_categories": False},
    )

    assert result["ok"] is True
    assert "tools" not in result
    assert "categories" not in result


@pytest.mark.asyncio
async def test_execute_module_validates_params_before_execution(monkeypatch):
    handler = _fake_handler(validation_result={"valid": False, "errors": ["url is required"]})
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: handler)

    result = await core_tools.dispatch_core_tool(
        "execute_module",
        {"module_id": "browser.goto", "params": {}},
    )

    assert result["ok"] is False
    assert result["params_valid"] is False
    assert "url is required" in result["error"]
    assert handler["_calls"]["validate"] == 1
    assert handler["_calls"]["execute"] == 0


@pytest.mark.asyncio
async def test_execute_module_runs_after_successful_validation(monkeypatch):
    handler = _fake_handler(validation_result={"valid": True, "errors": []})
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: handler)

    result = await core_tools.dispatch_core_tool(
        "execute_module",
        {"module_id": "browser.goto", "params": {"url": "https://example.com"}},
    )

    assert result["ok"] is True
    assert handler["_calls"]["validate"] == 1
    assert handler["_calls"]["execute"] == 1


@pytest.mark.asyncio
async def test_provider_log_entry_carries_core_mcp_evidence():
    async def dispatch(name, args):
        return {"ok": True, "data": {"title": "Example"}}

    _result_str, log_entry, _images = await dispatch_and_log_tool(
        "execute_module",
        {"module_id": "browser.extract", "params": {"selector": "h1"}},
        dispatch,
        round_num=0,
    )

    assert log_entry["mcp"]["source"] == "flyto-core"
    assert log_entry["mcp"]["contract_version"] == core_tools.CORE_MCP_CONTRACT_VERSION
    assert log_entry["mcp"]["module_id"] == "browser.extract"
    assert log_entry["mcp"]["ok"] is True


def test_mcp_client_builds_modern_request_metadata():
    manager = McpClientManager(["unused"])
    params = manager._request_params(
        {"name": "search"},
        modern=True,
    )

    assert params["name"] == "search"
    assert (
        params["_meta"][PROTOCOL_VERSION_META_KEY]
        == MODERN_PROTOCOL_VERSION
    )
    assert params["_meta"][CLIENT_CAPABILITIES_META_KEY] == {}


@pytest.mark.asyncio
async def test_mcp_client_prefers_modern_discovery(monkeypatch):
    manager = McpClientManager(["modern-server"])
    send_request = AsyncMock(side_effect=[
        {"supportedVersions": [MODERN_PROTOCOL_VERSION]},
        {"tools": [{"name": "search"}]},
    ])
    notify = AsyncMock()

    async def spawn(*args, **kwargs):
        return object()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(manager, "_send_request", send_request)
    monkeypatch.setattr(manager, "_send_notification", notify)

    assert await manager.connect() is True
    assert manager._modern_protocol is True
    assert send_request.await_args_list[0].args == ("server/discover", {})
    assert send_request.await_args_list[0].kwargs == {"modern": True}
    assert send_request.await_args_list[1].args == ("tools/list", {})
    notify.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_client_falls_back_to_legacy_handshake(monkeypatch):
    manager = McpClientManager(["legacy-server"])
    send_request = AsyncMock(side_effect=[
        None,
        {"protocolVersion": "2025-11-25"},
        {"tools": []},
    ])
    notify = AsyncMock()

    async def spawn(*args, **kwargs):
        return object()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(manager, "_send_request", send_request)
    monkeypatch.setattr(manager, "_send_notification", notify)

    assert await manager.connect() is True
    assert manager._modern_protocol is False
    assert send_request.await_args_list[1].args[0] == "initialize"
    assert send_request.await_args_list[1].kwargs == {"modern": False}
    notify.assert_awaited_once_with("notifications/initialized")
