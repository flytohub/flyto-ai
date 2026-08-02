# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Unit and integration tests for atomic MCP transport/catalog/session layers."""
from __future__ import annotations

import asyncio
import gc
import sys

import pytest

from flyto_ai.coding.contracts import CapabilitySpec
from flyto_ai.coding.mcp_catalog import (
    build_mcp_tool_catalog,
    catalog_tool_names,
    mcp_domain_status,
    provider_tool_name,
)
from flyto_ai.coding.mcp_session import McpStdioSession
from flyto_ai.coding.mcp_transport import (
    MAX_MCP_MESSAGE_BYTES,
    decode_mcp_message,
    encode_mcp_message,
    isolated_capability_environment,
)


def test_mcp_wire_codec_round_trips_unicode_object():
    payload = {"jsonrpc": "2.0", "id": 1, "params": {"text": "閉環"}}
    encoded = encode_mcp_message(payload)
    assert encoded.endswith(b"\n")
    assert decode_mcp_message(encoded) == payload


def test_mcp_wire_codec_rejects_oversize_invalid_and_non_object_messages():
    with pytest.raises(RuntimeError, match="request exceeds"):
        encode_mcp_message({"value": "x" * MAX_MCP_MESSAGE_BYTES})
    with pytest.raises(RuntimeError, match="response exceeds"):
        decode_mcp_message(b"x" * (MAX_MCP_MESSAGE_BYTES + 1))
    with pytest.raises(RuntimeError, match="invalid JSON"):
        decode_mcp_message(b"not-json\n")
    with pytest.raises(RuntimeError, match="must be an object"):
        decode_mcp_message(b"[]\n")


def test_capability_environment_passes_only_base_and_declared_flyto_names(monkeypatch):
    monkeypatch.setenv("FLYTO_ALLOWED", "visible")
    monkeypatch.setenv("UNSCOPED_SECRET", "hidden")
    spec = CapabilitySpec(
        name="isolated",
        argv=("server",),
        env_passthrough=("FLYTO_ALLOWED",),
    )
    env = isolated_capability_environment(spec, "/isolated/home")
    assert env["HOME"] == "/isolated/home"
    assert env["FLYTO_ALLOWED"] == "visible"
    assert "UNSCOPED_SECRET" not in env


def test_provider_tool_name_is_safe_bounded_and_deterministic():
    assert provider_tool_name("robot.planner", "move/arm") == "cap_robot_planner_move_arm"
    long_name = provider_tool_name("domain" * 20, "operation" * 20)
    assert len(long_name) == 64
    assert long_name == provider_tool_name("domain" * 20, "operation" * 20)
    assert long_name != provider_tool_name("domain" * 20, "different" * 20)


@pytest.mark.parametrize(
    ("result", "ok", "error"),
    [
        (None, False, "must be an object"),
        ({"isError": True}, False, "MCP error"),
        ({"structuredContent": {"ok": False, "error": "denied"}}, False, "denied"),
        ({"structuredContent": {"status": "failed"}}, False, "failed"),
        ({"content": [{"text": '{"ok": false, "message": "nested"}'}]}, False, "nested"),
        ({"content": [{"text": "untrusted prose says failed"}]}, True, None),
        ({"structuredContent": {"ok": True}}, True, None),
    ],
)
def test_mcp_domain_status_uses_only_machine_readable_evidence(result, ok, error):
    actual_ok, actual_error = mcp_domain_status(result)
    assert actual_ok is ok
    if error is None:
        assert actual_error is None
    else:
        assert error in actual_error


def _catalog_spec(*, required=("observe",), allowed=("observe",)):
    return CapabilitySpec(
        name="robot",
        argv=("server",),
        required_tools=required,
        allowed_tools=allowed,
    )


def test_catalog_exposes_only_selected_tools_with_reversible_names():
    raw = [
        {"name": "observe", "description": "read", "inputSchema": {"type": "object"}},
        {"name": "move", "description": "write", "inputSchema": {"type": "object"}},
    ]
    catalog = build_mcp_tool_catalog(_catalog_spec(), raw)
    assert catalog.remote_names == ("observe",)
    assert catalog.definitions[0]["name"] == "cap_robot_observe"
    assert catalog.tool_map == {"cap_robot_observe": "observe"}
    assert catalog_tool_names(raw) == ("move", "observe")


def test_catalog_fails_closed_on_shape_required_and_allowed_drift():
    with pytest.raises(RuntimeError, match="invalid tool catalog"):
        build_mcp_tool_catalog(_catalog_spec(), {})
    with pytest.raises(RuntimeError, match="invalid tool catalog"):
        build_mcp_tool_catalog(_catalog_spec(), [{}] * 2001)
    with pytest.raises(RuntimeError, match="missing required tools"):
        build_mcp_tool_catalog(_catalog_spec(), [{"name": "move"}])
    spec = _catalog_spec(required=(), allowed=("observe", "move"))
    with pytest.raises(RuntimeError, match="missing allowed tools"):
        build_mcp_tool_catalog(spec, [{"name": "observe"}])


def test_catalog_defaults_invalid_schema_and_keeps_first_duplicate_definition():
    raw = [
        {"name": "observe", "description": "first", "inputSchema": []},
        {"name": "observe", "description": "second", "inputSchema": {"type": "string"}},
        {"invalid": True},
    ]
    catalog = build_mcp_tool_catalog(_catalog_spec(), raw)
    assert catalog.definitions[0]["description"] == "[robot] first"
    assert catalog.definitions[0]["inputSchema"] == {
        "type": "object", "properties": {},
    }


def _write_correlated_mcp_server(path):
    source = (
        "import json, sys\n"
        "for line in sys.stdin:\n"
        " msg=json.loads(line)\n"
        " if 'id' not in msg: continue\n"
        " method=msg.get('method')\n"
        " if method=='initialize': result={'protocolVersion':'2025-06-18','capabilities':{},'serverInfo':{'name':'atomic-fixture','version':'1'}}\n"
        " elif method=='tools/list': result={'tools':[{'name':'observe','description':'read','inputSchema':{'type':'object'}}]}\n"
        " elif method=='tools/call':\n"
        "  print(json.dumps({'jsonrpc':'2.0','method':'progress','params':{}}),flush=True)\n"
        "  result={'structuredContent':{'ok':True,'arguments':msg['params'].get('arguments',{})}}\n"
        " else: result={}\n"
        " print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':result}),flush=True)\n"
    )
    path.write_text(source)


@pytest.mark.asyncio
async def test_real_session_correlates_responses_and_closes_transport(tmp_path):
    server = tmp_path / "atomic_server.py"
    _write_correlated_mcp_server(server)
    session = McpStdioSession(
        CapabilitySpec(
            name="atomic",
            argv=(sys.executable, str(server)),
            required_tools=("observe",),
            allowed_tools=("observe",),
        ),
        str(tmp_path),
    )
    await session.start()
    assert session.server_name == "atomic-fixture"
    assert session.remote_tool_names == ["observe"]
    result = await session.dispatch("cap_atomic_observe", {"value": 7})
    assert result["ok"] is True
    assert result["result"]["structuredContent"]["arguments"] == {"value": 7}
    await session.close()
    await session.close()
    assert session.process is None
    with pytest.raises(RuntimeError, match="started once"):
        await session.start()


@pytest.mark.asyncio
async def test_session_closes_process_when_catalog_contract_fails(tmp_path):
    server = tmp_path / "atomic_server.py"
    _write_correlated_mcp_server(server)
    session = McpStdioSession(
        CapabilitySpec(
            name="atomic",
            argv=(sys.executable, str(server)),
            required_tools=("missing",),
            allowed_tools=("missing",),
        ),
        str(tmp_path),
    )
    with pytest.raises(RuntimeError, match="missing required tools"):
        await session.start()
    assert session.process is None


@pytest.mark.asyncio
async def test_repeated_real_session_lifecycle_releases_every_subprocess_transport(tmp_path):
    server = tmp_path / "atomic_server.py"
    _write_correlated_mcp_server(server)
    spec = CapabilitySpec(
        name="atomic",
        argv=(sys.executable, str(server)),
        required_tools=("observe",),
        allowed_tools=("observe",),
    )
    for index in range(12):
        session = McpStdioSession(spec, str(tmp_path))
        await session.start()
        result = await session.dispatch("cap_atomic_observe", {"index": index})
        assert result["ok"] is True
        await session.close()
        assert session.process is None
    del session
    gc.collect()
    await asyncio.sleep(0)
