# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Tests for flyto-ai MCP server protocol version negotiation.

Regression: prevents the same bug as flyto-core/issues/16 — clients on
older MCP protocol versions must not be rejected.
"""
import pytest

from flyto_ai.mcp_server import (
    CLIENT_CAPABILITIES_META_KEY,
    CLIENT_INFO_META_KEY,
    LEGACY_PROTOCOL_VERSIONS,
    MCPServer,
    MODERN_PROTOCOL_VERSION,
    PROTOCOL_VERSION_META_KEY,
    SUPPORTED_PROTOCOL_VERSIONS,
    build_initialize_response,
    negotiate_protocol_version,
)

MODERN_META = {
    PROTOCOL_VERSION_META_KEY: MODERN_PROTOCOL_VERSION,
    CLIENT_CAPABILITIES_META_KEY: {},
    CLIENT_INFO_META_KEY: {"name": "test", "version": "1.0"},
}


def modern_request(request_id, method, params=None):
    request_params = dict(params or {})
    request_params["_meta"] = dict(MODERN_META)
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": request_params,
    }


@pytest.mark.parametrize("client_version", list(SUPPORTED_PROTOCOL_VERSIONS))
def test_supported_version_is_echoed(client_version):
    assert negotiate_protocol_version(client_version) == client_version


@pytest.mark.parametrize("bad", ["1999-01-01", "", None])
def test_unsupported_version_falls_back_to_server_preferred(bad):
    assert negotiate_protocol_version(bad) == SUPPORTED_PROTOCOL_VERSIONS[0]


def test_build_initialize_response_shape():
    resp = build_initialize_response("2025-06-18")
    assert resp["protocolVersion"] == "2025-06-18"
    assert resp["serverInfo"]["name"] == "flyto-ai"
    assert "tools" in resp["capabilities"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_version",
    ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"],
)
async def test_initialize_handler_negotiates(client_version):
    server = MCPServer()
    resp = await server.handle({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": client_version,
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"},
        },
    })
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == client_version


@pytest.mark.asyncio
async def test_initialize_handler_unsupported_falls_back():
    server = MCPServer()
    resp = await server.handle({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "1999-01-01", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    })
    assert resp["result"]["protocolVersion"] == LEGACY_PROTOCOL_VERSIONS[0]


@pytest.mark.asyncio
async def test_initialize_handler_missing_version():
    server = MCPServer()
    resp = await server.handle({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    })
    assert resp["result"]["protocolVersion"] == LEGACY_PROTOCOL_VERSIONS[0]


@pytest.mark.asyncio
async def test_modern_discovery_is_stateless_and_cacheable():
    server = MCPServer()
    response = await server.handle(modern_request(2, "server/discover"))
    result = response["result"]

    assert result["supportedVersions"][0] == MODERN_PROTOCOL_VERSION
    assert result["resultType"] == "complete"
    assert result["ttlMs"] == 60_000
    assert result["cacheScope"] == "public"
    assert (
        result["_meta"]["io.modelcontextprotocol/serverInfo"]["name"]
        == "flyto-ai"
    )


@pytest.mark.asyncio
async def test_modern_tools_list_has_required_result_fields():
    server = MCPServer()
    server._registry = {}
    response = await server.handle(modern_request(3, "tools/list"))
    result = response["result"]

    assert [tool["name"] for tool in result["tools"]] == ["chat"]
    assert result["resultType"] == "complete"
    assert result["ttlMs"] == 60_000
    assert result["cacheScope"] == "public"


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["initialize", "ping", "logging/setLevel"])
async def test_modern_removed_methods_are_rejected(method):
    response = await MCPServer().handle(modern_request(4, method))
    assert response["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_modern_request_requires_client_capabilities():
    request = modern_request(5, "tools/list")
    del request["params"]["_meta"][CLIENT_CAPABILITIES_META_KEY]
    response = await MCPServer().handle(request)
    assert response["error"]["code"] == -32602


@pytest.mark.asyncio
async def test_unsupported_modern_version_returns_supported_versions():
    request = modern_request(6, "tools/list")
    request["params"]["_meta"][PROTOCOL_VERSION_META_KEY] = "1900-01-01"
    response = await MCPServer().handle(request)
    assert response["error"]["code"] == -32022
    assert MODERN_PROTOCOL_VERSION in response["error"]["data"]["supported"]
