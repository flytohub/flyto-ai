# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Unit and integration tests for atomic MCP transport/catalog/session layers."""
from __future__ import annotations

import asyncio
import gc
import json
import random
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
        build_mcp_tool_catalog(
            _catalog_spec(), [{"name": "move", "inputSchema": {"type": "object"}}],
        )
    spec = _catalog_spec(required=(), allowed=("observe", "move"))
    with pytest.raises(RuntimeError, match="missing allowed tools"):
        build_mcp_tool_catalog(
            spec, [{"name": "observe", "inputSchema": {"type": "object"}}],
        )


def test_catalog_rejects_invalid_schema_duplicate_and_malformed_entries():
    raw = [
        {"name": "observe", "description": "first", "inputSchema": []},
        {"name": "observe", "description": "second", "inputSchema": {"type": "string"}},
        {"invalid": True},
    ]
    with pytest.raises(RuntimeError, match="inputSchema must be an object"):
        build_mcp_tool_catalog(_catalog_spec(), raw)
    duplicate = [
        {"name": "observe", "inputSchema": {"type": "object"}},
        {"name": "observe", "inputSchema": {"type": "object"}},
    ]
    with pytest.raises(RuntimeError, match="duplicate names"):
        build_mcp_tool_catalog(_catalog_spec(), duplicate)
    with pytest.raises(RuntimeError, match="entry 0 has an invalid name"):
        build_mcp_tool_catalog(_catalog_spec(), [{"name": "bad name", "inputSchema": {}}])
    with pytest.raises(RuntimeError, match="inputSchema type must be object"):
        build_mcp_tool_catalog(
            _catalog_spec(), [{"name": "observe", "inputSchema": {"type": "string"}}],
        )


def test_catalog_definitions_are_detached_from_remote_schema():
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
    raw = [{"name": "observe", "inputSchema": schema}]
    catalog = build_mcp_tool_catalog(_catalog_spec(), raw)
    schema["properties"]["value"]["type"] = "string"
    assert catalog.definitions[0]["inputSchema"]["properties"]["value"]["type"] == "integer"


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
    assert session.tools == []
    assert session.remote_tool_names == []
    assert session.observed_tool_names == ("observe",)
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
    assert session.remote_tool_names == []
    assert session.observed_tool_names == ("observe",)


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


def _write_resilience_mcp_server(path):
    source = (
        "import json, sys\n"
        "pending=[]\n"
        "for line in sys.stdin:\n"
        " msg=json.loads(line)\n"
        " if 'id' not in msg: continue\n"
        " method=msg.get('method')\n"
        " if method=='initialize': result={'protocolVersion':'2025-06-18','capabilities':{},'serverInfo':{'name':'resilience','version':'1'}}\n"
        " elif method=='tools/list': result={'tools':[{'name':'observe','inputSchema':{'type':'object'}}]}\n"
        " elif method=='tools/call':\n"
        "  args=msg['params'].get('arguments',{})\n"
        "  if args.get('hold'): continue\n"
        "  if args.get('reverse'):\n"
        "   pending.append(msg)\n"
        "   if len(pending)<2: continue\n"
        "   for item in reversed(pending): print(json.dumps({'jsonrpc':'2.0','id':item['id'],'result':{'structuredContent':{'ok':True,'value':item['params']['arguments']['value']}}}),flush=True)\n"
        "   pending=[]\n"
        "   continue\n"
        "  result={'structuredContent':{'ok':True,'value':args.get('value')}}\n"
        " else: result={}\n"
        " print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':result}),flush=True)\n"
    )
    path.write_text(source)


@pytest.mark.asyncio
async def test_transport_correlates_out_of_order_concurrent_responses(tmp_path):
    server = tmp_path / "resilience_server.py"
    _write_resilience_mcp_server(server)
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
    first, second = await asyncio.gather(
        session.dispatch("cap_atomic_observe", {"reverse": True, "value": 1}),
        session.dispatch("cap_atomic_observe", {"reverse": True, "value": 2}),
    )
    assert first["result"]["structuredContent"]["value"] == 1
    assert second["result"]["structuredContent"]["value"] == 2
    await session.close()


@pytest.mark.asyncio
async def test_cancelled_request_does_not_poison_later_dispatch(tmp_path):
    server = tmp_path / "resilience_server.py"
    _write_resilience_mcp_server(server)
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
    held = asyncio.create_task(
        session.dispatch("cap_atomic_observe", {"hold": True}),
    )
    await asyncio.sleep(0.02)
    held.cancel()
    with pytest.raises(asyncio.CancelledError):
        await held
    result = await session.dispatch("cap_atomic_observe", {"value": 9})
    assert result["result"]["structuredContent"]["value"] == 9
    await session.close()


@pytest.mark.asyncio
async def test_timed_out_request_does_not_poison_later_dispatch(tmp_path):
    server = tmp_path / "resilience_server.py"
    _write_resilience_mcp_server(server)
    session = McpStdioSession(
        CapabilitySpec(
            name="atomic",
            argv=(sys.executable, str(server)),
            required_tools=("observe",),
            allowed_tools=("observe",),
            timeout_seconds=1,
        ),
        str(tmp_path),
    )
    await session.start()
    with pytest.raises(RuntimeError, match="request timed out"):
        await session.dispatch("cap_atomic_observe", {"hold": True})
    result = await session.dispatch("cap_atomic_observe", {"value": 11})
    assert result["result"]["structuredContent"]["value"] == 11
    await session.close()


@pytest.mark.asyncio
async def test_bounded_concurrent_dispatch_soak_preserves_every_result(tmp_path):
    server = tmp_path / "resilience_server.py"
    _write_resilience_mcp_server(server)
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
    results = await asyncio.gather(*(
        session.dispatch("cap_atomic_observe", {"value": index})
        for index in range(256)
    ))
    assert [item["result"]["structuredContent"]["value"] for item in results] == list(
        range(256),
    )
    await session.close()


@pytest.mark.asyncio
async def test_transport_fails_all_pending_calls_when_child_crashes(tmp_path):
    server = tmp_path / "crash_server.py"
    server.write_text(
        "import json, os, sys\n"
        "for line in sys.stdin:\n"
        " msg=json.loads(line)\n"
        " if 'id' not in msg: continue\n"
        " if msg.get('method')=='initialize': result={'protocolVersion':'2025-06-18','serverInfo':{}}\n"
        " elif msg.get('method')=='tools/list': result={'tools':[{'name':'observe','inputSchema':{'type':'object'}}]}\n"
        " else: os._exit(7)\n"
        " print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':result}),flush=True)\n"
    )
    session = McpStdioSession(
        CapabilitySpec(
            name="atomic", argv=(sys.executable, str(server)),
            required_tools=("observe",), allowed_tools=("observe",),
        ),
        str(tmp_path),
    )
    await session.start()
    calls = [
        asyncio.create_task(session.dispatch("cap_atomic_observe", {"value": index}))
        for index in range(8)
    ]
    results = await asyncio.gather(*calls, return_exceptions=True)
    assert all(isinstance(result, RuntimeError) for result in results)
    assert all("closed" in str(result) for result in results)
    await session.close()


def _write_invalid_wire_server(path, call_source):
    path.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        " msg=json.loads(line)\n"
        " if 'id' not in msg: continue\n"
        " if msg.get('method')=='initialize': result={'protocolVersion':'2025-06-18','serverInfo':{}}\n"
        " elif msg.get('method')=='tools/list': result={'tools':[{'name':'observe','inputSchema':{'type':'object'}}]}\n"
        f" else: {call_source}\n"
        " if msg.get('method')!='tools/call': print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':result}),flush=True)\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call_source", "message"),
    [
        ("print('{broken',flush=True)", "invalid JSON"),
        (
            "print(json.dumps({'jsonrpc':'1.0','id':msg['id'],'result':{}}),flush=True)",
            "invalid JSON-RPC version",
        ),
        (
            "print(json.dumps({'jsonrpc':'2.0','id':'wrong','result':{}}),flush=True)",
            "invalid response id",
        ),
        (
            "print('x'*(1024*1024+1),flush=True)",
            "response exceeds the message limit",
        ),
    ],
)
async def test_real_transport_rejects_malformed_wire_responses(
    tmp_path, call_source, message,
):
    server = tmp_path / "invalid_wire_server.py"
    _write_invalid_wire_server(server, call_source)
    session = McpStdioSession(
        CapabilitySpec(
            name="atomic",
            argv=(sys.executable, str(server)),
            required_tools=("observe",),
            allowed_tools=("observe",),
            timeout_seconds=2,
        ),
        str(tmp_path),
    )
    await session.start()
    with pytest.raises(RuntimeError, match=message):
        await session.dispatch("cap_atomic_observe", {})
    await session.close()


@pytest.mark.asyncio
async def test_stderr_drain_remains_live_beyond_message_limit(tmp_path):
    server = tmp_path / "stderr_server.py"
    _write_invalid_wire_server(
        server,
        "sys.stderr.write('x'*(1024*1024+8192)); sys.stderr.flush(); "
        "print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':{'structuredContent':{'ok':True}}}),flush=True)",
    )
    session = McpStdioSession(
        CapabilitySpec(
            name="atomic",
            argv=(sys.executable, str(server)),
            required_tools=("observe",),
            allowed_tools=("observe",),
            timeout_seconds=5,
        ),
        str(tmp_path),
    )
    await session.start()
    assert (await session.dispatch("cap_atomic_observe", {}))["ok"] is True
    assert (await session.dispatch("cap_atomic_observe", {}))["ok"] is True
    await session.close()


def test_wire_codec_bounded_property_matrix_is_deterministic():
    rng = random.Random(20260802)
    for request_id in range(256):
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "params": {
                "integer": rng.randint(-(2**31), 2**31 - 1),
                "flag": bool(rng.getrandbits(1)),
                "text": "".join(chr(32 + rng.randrange(95)) for _ in range(rng.randrange(128))),
                "items": [rng.randrange(1000) for _ in range(rng.randrange(12))],
            },
        }
        assert decode_mcp_message(encode_mcp_message(payload)) == payload
        assert json.loads(encode_mcp_message(payload)) == payload
