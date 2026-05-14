# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for flyto-ai MCP server protocol version negotiation.

Regression: prevents the same bug as flyto-core/issues/16 — clients on
older MCP protocol versions must not be rejected.
"""
import pytest

from flyto_ai.mcp_server import (
    MCPServer,
    SUPPORTED_PROTOCOL_VERSIONS,
    build_initialize_response,
    negotiate_protocol_version,
)


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
    assert resp["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[0]


@pytest.mark.asyncio
async def test_initialize_handler_missing_version():
    server = MCPServer()
    resp = await server.handle({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
    })
    assert resp["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[0]
