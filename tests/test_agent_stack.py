# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Composition tests for the detachable Flyto2 full-agent stack."""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from flyto_ai.agent import Agent
from flyto_ai.coding.capabilities import CapabilityManager
from flyto_ai.coding.contracts import CapabilitySpec, CheckSpec
from flyto_ai.coding.stack import (
    AGENT_STACK_CONTRACT_VERSION,
    AGENT_STACK_POLICY_VERSION,
    DEFAULT_COMPONENTS,
    MAX_AGENT_STACK_MANIFEST_BYTES,
    build_agent_stack_capabilities,
    compose_capability_stack,
    load_agent_stack_manifest,
    probe_capability_stack,
    probe_agent_stack,
)
from flyto_ai.coding import build_agent_stack_capabilities as public_stack_builder
from flyto_ai.coding.stack_manifest import (
    compose_capability_stack as manifest_compose_capability_stack,
    load_agent_stack_manifest as manifest_load_agent_stack_manifest,
)
from flyto_ai.coding.stack_presets import (
    build_agent_stack_capabilities as preset_build_agent_stack_capabilities,
)
from flyto_ai.coding.stack_probe import (
    probe_capability_stack as runtime_probe_capability_stack,
)
from flyto_ai.config import AgentConfig
from flyto_ai.permissions import PermissionLevel
from flyto_ai.protocols import ToolExecutor


def test_full_stack_is_split_into_least_privilege_components():
    specs = build_agent_stack_capabilities(python_executable="python")
    assert tuple(spec.name for spec in specs) == DEFAULT_COMPONENTS
    assert all(spec.required for spec in specs)
    assert all(spec.required_tools == spec.allowed_tools for spec in specs)
    assert all(
        {name for name, _level in spec.tool_permissions} == set(spec.allowed_tools)
        for spec in specs
    )

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


def test_stack_facade_preserves_atomic_module_identities():
    assert compose_capability_stack is manifest_compose_capability_stack
    assert load_agent_stack_manifest is manifest_load_agent_stack_manifest
    assert build_agent_stack_capabilities is preset_build_agent_stack_capabilities
    assert probe_capability_stack is runtime_probe_capability_stack


def test_generic_stack_composes_arbitrary_domain_capabilities():
    robotics = CapabilitySpec(name="robot-planner", argv=("robot-planner",), kind="command")
    operations = CapabilitySpec(name="ticket-router", argv=("ticket-router",), kind="command")
    assert compose_capability_stack((robotics,), (operations,)) == (robotics, operations)
    with pytest.raises(ValueError, match="duplicate names"):
        compose_capability_stack((robotics, robotics))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"argv": ["runner", 1]}, "argv must contain only strings"),
        ({"required_tools": ["inspect", 1]}, "required_tools must contain only strings"),
        ({"allowed_tools": ["inspect", 1]}, "allowed_tools must contain only strings"),
        ({"env_passthrough": ["FLYTO_TOKEN", 1]}, "env_passthrough must contain only strings"),
        ({"tool_permissions": {"inspect": 1}}, "keys and values must be strings"),
    ],
)
def test_capability_mapping_rejects_non_string_boundary_values(override, message):
    value = {"name": "boundary", "argv": ["runner"], "kind": "command"}
    value.update(override)
    with pytest.raises(ValueError, match=message):
        CapabilitySpec.from_mapping(value)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"name": 7}, "name must be a string"),
        ({"required": 1}, "required must be a boolean"),
        ({"kind": True}, "kind must be a string"),
        ({"contract_version": 1}, "contract_version must be a string"),
        ({"protocol_version": None}, "protocol_version must be a string"),
        ({"timeout_seconds": "10"}, "timeout_seconds must be an integer"),
        ({"timeout_seconds": True}, "timeout_seconds must be an integer"),
    ],
)
def test_capability_mapping_rejects_scalar_coercion(override, message):
    value = {"name": "boundary", "argv": ["runner"]}
    value.update(override)
    with pytest.raises(ValueError, match=message):
        CapabilitySpec.from_mapping(value)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"name": 7}, "name must be a string"),
        ({"required": 1}, "required must be a boolean"),
        ({"timeout_seconds": "120"}, "timeout_seconds must be an integer"),
        ({"timeout_seconds": False}, "timeout_seconds must be an integer"),
    ],
)
def test_check_mapping_rejects_scalar_coercion(override, message):
    value = {"name": "lint", "argv": ["ruff", "check", "."]}
    value.update(override)
    with pytest.raises(ValueError, match=message):
        CheckSpec.from_mapping(value)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"required": 1}, "required must be a boolean"),
        ({"contract_version": 1}, "contract_version must be a string"),
        ({"protocol_version": 1}, "protocol_version must be a string"),
        ({"timeout_seconds": True}, "timeout_seconds must be an integer"),
        ({"argv": ("runner", 1)}, "argv contains an invalid item"),
    ],
)
def test_capability_contract_rejects_invalid_direct_python_values(override, message):
    value = {"name": "boundary", "argv": ("runner",)}
    value.update(override)
    with pytest.raises(ValueError, match=message):
        CapabilitySpec(**value)


def test_capability_manager_satisfies_generic_agent_tool_executor(tmp_path):
    manager = CapabilityManager(str(tmp_path))
    assert isinstance(manager, ToolExecutor)
    assert manager.tools == []


def _write_manifest(
    path, capabilities, *, profile="robotics-lab",
    version=AGENT_STACK_CONTRACT_VERSION, extra="",
):
    path.write_text(
        "version: {}\n".format(version)
        + "profile: {}\n".format(profile)
        + extra
        + "capabilities:\n"
        + "".join("  - {}\n".format(json.dumps(item)) for item in capabilities)
    )


def _write_profile_mcp_server(path, tool_name="plan_motion"):
    source = (
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    msg=json.loads(line)\n"
        "    if 'id' not in msg: continue\n"
        "    method=msg.get('method')\n"
        "    if method=='initialize': result={'protocolVersion':'2025-06-18','capabilities':{},'serverInfo':{'name':'domain-fixture','version':'1'}}\n"
        "    elif method=='tools/list': result={'tools':[{'name':'__TOOL__','inputSchema':{'type':'object'}}]}\n"
        "    elif method=='tools/call': result={'content':[{'type':'text','text':'planned'}]}\n"
        "    else: result={}\n"
        "    print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':result}), flush=True)\n"
    )
    path.write_text(source.replace("__TOOL__", tool_name))


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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"\xff", "not valid UTF-8 YAML"),
        (b"- not-an-object\n", "must be a YAML object"),
        (b"version: unknown\nprofile: test\ncapabilities: []\n", "unsupported"),
        (
            b"version: flyto.agent-stack.v1\nprofile: unsafe/name\ncapabilities: []\n",
            "safe identifier",
        ),
        (
            b"version: flyto.agent-stack.v1\nprofile: test\ncapabilities: {}\n",
            "must be a YAML array",
        ),
        (
            b"version: flyto.agent-stack.v1\nprofile: test\ncapabilities:\n  - nope\n",
            "must be an object",
        ),
    ],
)
def test_manifest_decoder_rejects_each_invalid_document_boundary(tmp_path, payload, message):
    manifest_path = tmp_path / "agent-stack.yaml"
    manifest_path.write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        load_agent_stack_manifest(str(tmp_path), "agent-stack.yaml")


def test_manifest_file_boundary_rejects_absolute_missing_directory_and_oversize(tmp_path):
    manifest_path = tmp_path / "agent-stack.yaml"
    with pytest.raises(ValueError, match="must be relative"):
        load_agent_stack_manifest(str(tmp_path), str(manifest_path))
    with pytest.raises(ValueError, match="regular file"):
        load_agent_stack_manifest(str(tmp_path), "missing.yaml")
    directory = tmp_path / "profile"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        load_agent_stack_manifest(str(tmp_path), "profile")
    manifest_path.write_bytes(b"x" * (MAX_AGENT_STACK_MANIFEST_BYTES + 1))
    with pytest.raises(ValueError, match="256 KiB"):
        load_agent_stack_manifest(str(tmp_path), "agent-stack.yaml")


def test_manifest_fingerprint_is_deterministic_and_profile_sensitive(tmp_path):
    manifest_path = tmp_path / "agent-stack.yaml"
    capability = {
        "name": "observer",
        "argv": ["observer"],
        "allowed_tools": ["inspect"],
    }
    _write_manifest(manifest_path, [capability], profile="profile-a")
    first = load_agent_stack_manifest(str(tmp_path), "agent-stack.yaml")
    _write_manifest(manifest_path, [capability], profile="profile-a")
    second = load_agent_stack_manifest(str(tmp_path), "agent-stack.yaml")
    _write_manifest(manifest_path, [capability], profile="profile-b")
    changed = load_agent_stack_manifest(str(tmp_path), "agent-stack.yaml")
    assert first.manifest_fingerprint == second.manifest_fingerprint
    assert changed.manifest_fingerprint != first.manifest_fingerprint


def test_v2_manifest_requires_exhaustive_tool_permission_classification(tmp_path):
    manifest_path = tmp_path / "agent-stack.yaml"
    capability = {
        "name": "mission-control",
        "argv": ["mission-control"],
        "allowed_tools": ["observe", "move", "safe_stop"],
        "tool_permissions": {
            "observe": "read_only",
            "move": "danger_full",
            "safe_stop": "workspace_write",
        },
    }
    _write_manifest(
        manifest_path, [capability], version=AGENT_STACK_POLICY_VERSION,
    )
    manifest = load_agent_stack_manifest(str(tmp_path), "agent-stack.yaml")
    assert manifest.contract_version == AGENT_STACK_POLICY_VERSION
    assert dict(manifest.capabilities[0].tool_permissions) == {
        "move": "danger_full",
        "observe": "read_only",
        "safe_stop": "workspace_write",
    }

    incomplete = dict(capability, tool_permissions={"observe": "read_only"})
    _write_manifest(
        manifest_path, [incomplete], version=AGENT_STACK_POLICY_VERSION,
    )
    with pytest.raises(ValueError, match="must classify every allowed tool"):
        load_agent_stack_manifest(str(tmp_path), "agent-stack.yaml")

    invalid = dict(capability, tool_permissions={
        "observe": "read_only",
        "move": "unrestricted",
        "safe_stop": "workspace_write",
    })
    _write_manifest(
        manifest_path, [invalid], version=AGENT_STACK_POLICY_VERSION,
    )
    with pytest.raises(ValueError, match="invalid permission level"):
        load_agent_stack_manifest(str(tmp_path), "agent-stack.yaml")


def test_runtime_permission_ceiling_is_enforced_inside_manager_and_agent(tmp_path):
    import asyncio

    server = tmp_path / "policy_server.py"
    _write_profile_mcp_server(server)

    async def read_only_scenario():
        read_spec = CapabilitySpec(
            name="observer",
            argv=(sys.executable, str(server)),
            required=True,
            allowed_tools=("plan_motion",),
            tool_permissions=(("plan_motion", "read_only"),),
        )
        manager = CapabilityManager(str(tmp_path), PermissionLevel.READ_ONLY)
        await manager.start((read_spec,))
        provider_name = manager.tools[0]["name"]
        agent = Agent(
            AgentConfig(
                provider="ollama",
                enable_memory=False,
                enable_transcript=False,
                enable_injection_detection=False,
                enable_pro=False,
                enable_deterministic=False,
                enable_model_routing=False,
                permission_level="read_only",
            ),
            api_client=object(),
            tool_executor=manager,
        )
        outer_decision = agent._permission_enforcer.check(provider_name, {})
        result = await manager.dispatch(provider_name, {})
        await manager.close()
        return outer_decision, result

    outer_decision, result = asyncio.run(read_only_scenario())
    assert outer_decision.allowed is True
    assert result["ok"] is True

    async def blocked_scenario():
        danger_spec = CapabilitySpec(
            name="actuator",
            argv=(sys.executable, str(server)),
            required=True,
            allowed_tools=("plan_motion",),
            tool_permissions=(("plan_motion", "danger_full"),),
        )
        manager = CapabilityManager(str(tmp_path), PermissionLevel.WORKSPACE_WRITE)
        await manager.start((danger_spec,))
        result = await manager.dispatch(manager.tools[0]["name"], {})
        await manager.close()
        return result

    blocked = asyncio.run(blocked_scenario())
    assert blocked["ok"] is False
    assert blocked["policy_outcome"] == "require_confirmation"
    assert blocked["required_permission"] == "danger_full"


def test_core_execute_module_keeps_argument_sensitive_danger_gate(tmp_path):
    import asyncio

    server = tmp_path / "core_policy_server.py"
    _write_profile_mcp_server(server, "execute_module")

    async def scenario():
        manager = CapabilityManager(str(tmp_path), PermissionLevel.WORKSPACE_WRITE)
        await manager.start((CapabilitySpec(
            name="core-runtime",
            argv=(sys.executable, str(server)),
            required=True,
            allowed_tools=("execute_module",),
            tool_permissions=(("execute_module", "workspace_write"),),
        ),))
        provider_name = manager.tools[0]["name"]
        safe = await manager.dispatch(provider_name, {"module_id": "string.uppercase"})
        danger = await manager.dispatch(provider_name, {"module_id": "shell.run"})
        await manager.close()
        return safe, danger

    safe, danger = asyncio.run(scenario())
    assert safe["ok"] is True
    assert danger["ok"] is False
    assert danger["required_permission"] == "danger_full"


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
    assert result["composition_fingerprint"] == (
        "648c821f1c2a6d462a8b9afce3e8a575366aa4c952b9887f8a3717637e56854f"
    )
    components = {item["name"]: item for item in result["components"]}
    assert tuple(components) == DEFAULT_COMPONENTS
    assert components["flyto-page-inspector"]["tools"] == ("inspect_page",)
    assert "chat" not in components["flyto-blueprint"]["tools"]
    assert all(item["negotiated_protocol_version"] == "2025-06-18" for item in components.values())
