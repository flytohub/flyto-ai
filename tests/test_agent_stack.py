# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Composition tests for the detachable Flyto2 full-agent stack."""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from flyto_ai.coding.capabilities import CapabilityManager
from flyto_ai.coding.contracts import CapabilitySpec
from flyto_ai.coding.stack import (
    AGENT_STACK_CONTRACT_VERSION,
    DEFAULT_COMPONENTS,
    build_agent_stack_capabilities,
    compose_capability_stack,
    load_agent_stack_manifest,
    probe_capability_stack,
    probe_agent_stack,
)
from flyto_ai.coding import build_agent_stack_capabilities as public_stack_builder
from flyto_ai.protocols import ToolExecutor


def test_full_stack_is_split_into_least_privilege_components():
    specs = build_agent_stack_capabilities(python_executable="python")
    assert tuple(spec.name for spec in specs) == DEFAULT_COMPONENTS
    assert all(spec.required for spec in specs)
    assert all(spec.required_tools == spec.allowed_tools for spec in specs)

    by_name = {spec.name: spec for spec in specs}
    assert by_name["flyto-page-inspector"].allowed_tools == ("inspect_page",)
    assert "chat" not in by_name["flyto-blueprint"].allowed_tools
    assert "execute_module" not in by_name["flyto-blueprint"].allowed_tools
    assert "inspect_page" not in by_name["flyto-core"].allowed_tools
    assert by_name["flyto-indexer"].argv == (
        "python", "-m", "flyto_indexer.mcp_server",
    )
    assert by_name["flyto-core"].argv == ("python", "-m", "core.mcp_server")


def test_stack_components_can_be_detached_or_made_optional():
    selected = build_agent_stack_capabilities(
        ("flyto-indexer", "flyto-core"),
        python_executable="python",
    )
    assert all(spec.required for spec in selected)

    specs = build_agent_stack_capabilities(
        ("flyto-indexer", "flyto-core"),
        required_components=("flyto-indexer",),
        python_executable="python",
    )
    assert [spec.name for spec in specs] == ["flyto-indexer", "flyto-core"]
    assert specs[0].required is True
    assert specs[1].required is False


def test_stack_builder_is_available_from_the_public_coding_api():
    assert public_stack_builder is build_agent_stack_capabilities


def test_generic_stack_composes_arbitrary_domain_capabilities():
    robotics = CapabilitySpec(name="robot-planner", argv=("robot-planner",), kind="command")
    operations = CapabilitySpec(name="ticket-router", argv=("ticket-router",), kind="command")
    assert compose_capability_stack((robotics,), (operations,)) == (robotics, operations)
    with pytest.raises(ValueError, match="duplicate names"):
        compose_capability_stack((robotics, robotics))


def test_capability_manager_satisfies_generic_agent_tool_executor(tmp_path):
    manager = CapabilityManager(str(tmp_path))
    assert isinstance(manager, ToolExecutor)
    assert manager.tools == []


def _write_manifest(path, capabilities, *, profile="robotics-lab", extra=""):
    path.write_text(
        "version: flyto.agent-stack.v1\n"
        "profile: {}\n".format(profile)
        + extra
        + "capabilities:\n"
        + "".join("  - {}\n".format(json.dumps(item)) for item in capabilities)
    )


def _write_profile_mcp_server(path):
    path.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    msg=json.loads(line)\n"
        "    if 'id' not in msg: continue\n"
        "    method=msg.get('method')\n"
        "    if method=='initialize': result={'protocolVersion':'2025-06-18','capabilities':{},'serverInfo':{'name':'domain-fixture','version':'1'}}\n"
        "    elif method=='tools/list': result={'tools':[{'name':'plan_motion','inputSchema':{'type':'object'}}]}\n"
        "    elif method=='tools/call': result={'content':[{'type':'text','text':'planned'}]}\n"
        "    else: result={}\n"
        "    print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':result}), flush=True)\n"
    )


def test_manifest_loads_and_attests_arbitrary_profile(tmp_path):
    server = tmp_path / "profile_server.py"
    manifest_path = tmp_path / "agent-stack.yaml"
    _write_profile_mcp_server(server)
    _write_manifest(manifest_path, [{
        "name": "robot-planner",
        "argv": [sys.executable, str(server)],
        "required": True,
        "contract_version": "example.robotics.v1",
        "required_tools": ["plan_motion"],
        "allowed_tools": ["plan_motion"],
    }])

    manifest = load_agent_stack_manifest(str(tmp_path), "agent-stack.yaml")
    assert manifest.profile == "robotics-lab"
    assert manifest.capabilities[0].name == "robot-planner"
    assert len(manifest.manifest_fingerprint) == 64

    import asyncio

    result = asyncio.run(probe_capability_stack(
        str(tmp_path), manifest.capabilities,
        profile=manifest.profile,
        manifest_fingerprint=manifest.manifest_fingerprint,
    ))
    assert result["ok"] is True
    assert result["profile"] == "robotics-lab"
    assert result["manifest_fingerprint"] == manifest.manifest_fingerprint
    assert result["components"][0]["tools"] == ("plan_motion",)


def test_manifest_fails_closed_on_scope_schema_and_path_escape(tmp_path):
    manifest_path = tmp_path / "agent-stack.yaml"
    base = {"name": "unsafe", "argv": ["unsafe-server"], "required": True}
    _write_manifest(manifest_path, [base])
    with pytest.raises(ValueError, match="allowed_tools"):
        load_agent_stack_manifest(str(tmp_path), "agent-stack.yaml")

    scoped = dict(base, allowed_tools=["inspect"])
    _write_manifest(manifest_path, [scoped], extra="unknown: true\n")
    with pytest.raises(ValueError, match="unknown agent stack"):
        load_agent_stack_manifest(str(tmp_path), "agent-stack.yaml")

    invalid_type = dict(scoped, required="false")
    _write_manifest(manifest_path, [invalid_type])
    with pytest.raises(ValueError, match="required must be a boolean"):
        load_agent_stack_manifest(str(tmp_path), "agent-stack.yaml")

    _write_manifest(manifest_path, [scoped, scoped])
    with pytest.raises(ValueError, match="duplicate names"):
        load_agent_stack_manifest(str(tmp_path), "agent-stack.yaml")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="inside the workspace"):
        load_agent_stack_manifest(str(workspace), "../agent-stack.yaml")


def test_stack_rejects_unknown_duplicate_and_unselected_required_components():
    with pytest.raises(ValueError, match="unknown agent stack"):
        build_agent_stack_capabilities(("unknown",), required_components=())
    with pytest.raises(ValueError, match="duplicates"):
        build_agent_stack_capabilities(
            ("flyto-core", "flyto-core"), required_components=("flyto-core",),
        )
    with pytest.raises(ValueError, match="were not selected"):
        build_agent_stack_capabilities(
            ("flyto-core",), required_components=("flyto-indexer",),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("flyto_indexer") is None,
    reason="flyto-indexer is an independently installed stack component",
)
def test_real_full_stack_preflight_negotiates_isolated_tool_surfaces(tmp_path):
    import asyncio

    result = asyncio.run(probe_agent_stack(str(tmp_path)))
    assert result["contract_version"] == AGENT_STACK_CONTRACT_VERSION
    assert result["ok"] is True
    assert len(result["composition_fingerprint"]) == 64
    components = {item["name"]: item for item in result["components"]}
    assert tuple(components) == DEFAULT_COMPONENTS
    assert components["flyto-page-inspector"]["tools"] == ("inspect_page",)
    assert "chat" not in components["flyto-blueprint"]["tools"]
    assert all(item["negotiated_protocol_version"] == "2025-06-18" for item in components.values())
