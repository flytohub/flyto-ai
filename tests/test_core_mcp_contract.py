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
