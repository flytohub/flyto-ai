# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Atomic and closed-loop tests for composed capability runtime policy."""
from __future__ import annotations

import asyncio
import sys

import pytest

from flyto_ai.agent import Agent, _bind_tool_executor
from flyto_ai.coding.capabilities import CapabilityManager
from flyto_ai.coding.contracts import CapabilitySpec
from flyto_ai.coding.execution_policy import (
    ApprovalDecision,
    ExecutionLimits,
    ExecutionPolicy,
)
from flyto_ai.coding.execution_trace import ExecutionTraceLedger
from flyto_ai.coding.permissions import (
    CapabilityPermissionGate,
    coerce_permission_level,
)
from flyto_ai.coding.tool_registry import CapabilityToolRegistry
from flyto_ai.config import AgentConfig
from flyto_ai.permissions import PermissionLevel


class _FakeSession:
    def __init__(self, definitions, mapping):
        self.tools = definitions
        self._mapping = mapping
        self.calls = []

    def remote_tool_name(self, provider_name):
        return self._mapping.get(provider_name)

    async def dispatch(self, provider_name, arguments):
        self.calls.append((provider_name, arguments))
        return {"ok": True}


def _spec(name: str, tools, permissions=()):
    return CapabilitySpec(
        name=name,
        argv=("server",),
        allowed_tools=tuple(tools),
        tool_permissions=tuple(permissions),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (PermissionLevel.READ_ONLY, PermissionLevel.READ_ONLY),
        ("read_only", PermissionLevel.READ_ONLY),
        ("WORKSPACE_WRITE", PermissionLevel.WORKSPACE_WRITE),
        ("danger_full", PermissionLevel.DANGER_FULL),
    ],
)
def test_permission_level_coercion_is_explicit(value, expected):
    assert coerce_permission_level(value) == expected


@pytest.mark.parametrize("value", ["root", "", 1, None])
def test_permission_level_coercion_rejects_unknown_authority(value):
    with pytest.raises(ValueError, match="unknown capability permission"):
        coerce_permission_level(value)


def test_permission_gate_never_allows_dynamic_policy_to_lower_declared_risk():
    gate = CapabilityPermissionGate(
        PermissionLevel.WORKSPACE_WRITE,
        risk_resolvers={"move": lambda _arguments: PermissionLevel.READ_ONLY},
    )
    evaluation = gate.evaluate(
        "cap_robot_move",
        "move",
        PermissionLevel.DANGER_FULL,
        {},
    )
    assert evaluation.required_level == PermissionLevel.DANGER_FULL
    assert evaluation.decision.allowed is False
    assert evaluation.denial_payload()["policy_outcome"] == "require_confirmation"


def test_permission_gate_supports_domain_specific_argument_risk_resolvers():
    def movement_risk(arguments):
        if arguments.get("physical"):
            return PermissionLevel.DANGER_FULL
        return PermissionLevel.WORKSPACE_WRITE

    gate = CapabilityPermissionGate(
        PermissionLevel.WORKSPACE_WRITE,
        risk_resolvers={"move": movement_risk},
    )
    simulated = gate.evaluate(
        "cap_robot_move", "move", PermissionLevel.READ_ONLY, {"physical": False},
    )
    physical = gate.evaluate(
        "cap_robot_move", "move", PermissionLevel.READ_ONLY, {"physical": True},
    )
    assert simulated.decision.allowed is True
    assert simulated.required_level == PermissionLevel.WORKSPACE_WRITE
    assert physical.decision.allowed is False
    assert physical.required_level == PermissionLevel.DANGER_FULL


def test_permission_gate_preserves_core_argument_sensitive_escalation():
    gate = CapabilityPermissionGate(PermissionLevel.WORKSPACE_WRITE)
    safe = gate.evaluate(
        "cap_core_execute_module",
        "execute_module",
        PermissionLevel.WORKSPACE_WRITE,
        {"module_id": "string.uppercase"},
    )
    danger = gate.evaluate(
        "cap_core_execute_module",
        "execute_module",
        PermissionLevel.WORKSPACE_WRITE,
        {"module_id": "shell.run"},
    )
    assert safe.decision.allowed is True
    assert danger.decision.allowed is False
    assert danger.required_level == PermissionLevel.DANGER_FULL


def test_permission_gate_rejects_invalid_resolver_output():
    gate = CapabilityPermissionGate(
        risk_resolvers={"move": lambda _arguments: "danger_full"},
    )
    with pytest.raises(ValueError, match="invalid permission level"):
        gate.evaluate(
            "cap_robot_move", "move", PermissionLevel.READ_ONLY, {},
        )


def test_tool_registry_commits_complete_session_atomically():
    registry = CapabilityToolRegistry()
    first = _FakeSession(
        [{"name": "cap_existing", "inputSchema": {"type": "object"}}],
        {"cap_existing": "observe"},
    )
    registry.register_session(
        first,
        _spec("existing", ("observe",), (("observe", "read_only"),)),
    )
    conflicting = _FakeSession(
        [
            {"name": "cap_fresh", "inputSchema": {"type": "object"}},
            {"name": "cap_existing", "inputSchema": {"type": "object"}},
        ],
        {"cap_fresh": "move", "cap_existing": "stop"},
    )
    with pytest.raises(RuntimeError, match="collision"):
        registry.register_session(
            conflicting,
            _spec(
                "conflicting",
                ("move", "stop"),
                (("move", "workspace_write"), ("stop", "danger_full")),
            ),
        )
    assert registry.resolve("cap_existing") is not None
    assert registry.resolve("cap_fresh") is None
    assert len(registry.definitions) == 1


def test_tool_registry_returns_copies_and_clears_all_runtime_metadata():
    registry = CapabilityToolRegistry()
    session = _FakeSession(
        [{"name": "cap_observe", "inputSchema": {"type": "object"}}],
        {"cap_observe": "observe"},
    )
    registry.register_session(
        session,
        _spec("observer", ("observe",), (("observe", "read_only"),)),
    )
    definitions = registry.definitions
    definitions[0]["name"] = "mutated"
    definitions[0]["inputSchema"]["type"] = "string"
    overrides = registry.permission_overrides
    overrides["cap_observe"] = PermissionLevel.DANGER_FULL
    entry = registry.resolve("cap_observe")
    assert entry.provider_name == "cap_observe"
    assert entry.definition["inputSchema"]["type"] == "object"
    with pytest.raises(TypeError):
        entry.definition["inputSchema"]["type"] = "string"
    assert registry.permission_overrides == {
        "cap_observe": PermissionLevel.READ_ONLY,
    }
    registry.clear()
    assert registry.definitions == []
    assert registry.permission_overrides == {}


def test_tool_registry_rejects_incomplete_session_mapping():
    registry = CapabilityToolRegistry()
    session = _FakeSession([{"name": "cap_unmapped"}], {})
    with pytest.raises(RuntimeError, match="mapping is incomplete"):
        registry.register_session(session, _spec("broken", ("observe",)))
    assert registry.definitions == []


class _Executor:
    def __init__(self, tools, overrides):
        self.tools = tools
        self.permission_overrides = overrides

    async def dispatch(self, _name, _arguments):
        return {"ok": True}


def test_agent_executor_binding_is_independently_validated():
    executor = _Executor(
        [{"name": "observe"}],
        {"observe": PermissionLevel.READ_ONLY},
    )
    tools, dispatch, overrides = _bind_tool_executor(executor, None, None)
    assert tools == [{"name": "observe"}]
    assert callable(dispatch)
    assert overrides == {"observe": PermissionLevel.READ_ONLY}
    with pytest.raises(ValueError, match="tools must be a list"):
        _bind_tool_executor(_Executor((), {}), None, None)
    with pytest.raises(ValueError, match="must map names"):
        _bind_tool_executor(_Executor([], {"observe": "read_only"}), None, None)


@pytest.mark.asyncio
async def test_manager_has_single_start_and_idempotent_close_lifecycle(tmp_path):
    manager = CapabilityManager(str(tmp_path))
    command = CapabilitySpec(
        name="python-runtime", argv=(sys.executable,), kind="command",
    )
    statuses = await manager.start((command,))
    assert statuses[0].available is True
    with pytest.raises(RuntimeError, match="started once"):
        await manager.start((command,))
    await manager.close()
    await manager.close()
    assert manager.tools == []
    assert manager.permission_overrides == {}


def _write_policy_server(path):
    source = (
        "import json, sys\n"
        "tools=['observe','move','execute_module','fail']\n"
        "for line in sys.stdin:\n"
        " msg=json.loads(line)\n"
        " if 'id' not in msg: continue\n"
        " method=msg.get('method')\n"
        " if method=='initialize': result={'protocolVersion':'2025-06-18','capabilities':{},'serverInfo':{'name':'policy-fixture','version':'1'}}\n"
        " elif method=='tools/list': result={'tools':[{'name':name,'inputSchema':{'type':'object'}} for name in tools]}\n"
        " elif method=='tools/call':\n"
        "  name=msg['params']['name']; args=msg['params'].get('arguments',{})\n"
        "  result={'structuredContent':({'ok':False,'error':'domain failed'} if name=='fail' else {'ok':True,'tool':name,'arguments':args})}\n"
        " else: result={}\n"
        " print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':result}),flush=True)\n"
    )
    path.write_text(source)


def _agent_config(permission_level="workspace_write"):
    return AgentConfig(
        provider="ollama",
        enable_memory=False,
        enable_transcript=False,
        enable_injection_detection=False,
        enable_pro=False,
        enable_deterministic=False,
        enable_model_routing=False,
        permission_level=permission_level,
    )


@pytest.mark.asyncio
async def test_manager_rolls_back_all_sessions_on_provider_name_collision(tmp_path):
    server = tmp_path / "policy_server.py"
    _write_policy_server(server)
    permissions = (("observe", "read_only"),)
    first = CapabilitySpec(
        name="domain.one",
        argv=(sys.executable, str(server)),
        allowed_tools=("observe",),
        tool_permissions=permissions,
    )
    second = CapabilitySpec(
        name="domain_one",
        argv=(sys.executable, str(server)),
        allowed_tools=("observe",),
        tool_permissions=permissions,
    )
    manager = CapabilityManager(str(tmp_path))
    with pytest.raises(RuntimeError, match="provider tool name collision"):
        await manager.start((first, second))
    assert manager.sessions == []
    assert manager.tools == []
    assert manager.permission_overrides == {}
    await manager.close()


@pytest.mark.asyncio
async def test_real_composed_runtime_closes_manifest_to_evidence_loop(tmp_path):
    server = tmp_path / "policy_server.py"
    _write_policy_server(server)
    spec = CapabilitySpec(
        name="domain",
        argv=(sys.executable, str(server)),
        required=True,
        allowed_tools=("observe", "move", "execute_module", "fail"),
        tool_permissions=(
            ("execute_module", "workspace_write"),
            ("fail", "read_only"),
            ("move", "danger_full"),
            ("observe", "read_only"),
        ),
    )
    manager = CapabilityManager(str(tmp_path), PermissionLevel.WORKSPACE_WRITE)
    await manager.start((spec,))
    agent = Agent(
        _agent_config(),
        api_client=object(),
        tool_executor=manager,
    )
    dispatch = agent._make_safe_dispatch(execute_blueprints=False)
    names = {item["name"].rsplit("_", 1)[-1]: item["name"] for item in manager.tools}
    try:
        observed = await dispatch(names["observe"], {"subject": "sample"})
        outer_block = await dispatch(names["move"], {"physical": True})
        inner_block = await manager.dispatch(names["move"], {"physical": True})
        safe_module = await dispatch(
            names["module"], {"module_id": "string.uppercase"},
        )
        dynamic_block = await dispatch(
            names["module"], {"module_id": "shell.run"},
        )
        domain_failure = await dispatch(names["fail"], {})
        trace_before_replay = manager.execution_trace
        policy_before_replay = await manager.execution_policy_snapshot()
        replay = await manager.replay_execution_trace(
            allowed_permissions=("read_only", "workspace_write"),
        )
        trace_after_replay = manager.execution_trace
        published = []

        async def outcome_sink(payload):
            published.append(payload)
            return {"ok": True, "api_key": "must-redact"}

        receipt = await manager.publish_blueprint_outcome(
            "bp_closed_loop", replay, outcome_sink,
        )
    finally:
        await manager.close()

    assert observed["ok"] is True
    assert observed["result"]["structuredContent"]["arguments"] == {
        "subject": "sample",
    }
    assert outer_block["policy_outcome"] == "require_confirmation"
    assert inner_block["policy_outcome"] == "require_confirmation"
    assert safe_module["ok"] is True
    assert dynamic_block["policy_outcome"] == "require_confirmation"
    assert domain_failure["ok"] is False
    assert domain_failure["error"] == "domain failed"
    assert trace_before_replay["event_count"] == 6
    assert trace_before_replay == trace_after_replay
    assert {event["policy_code"] for event in trace_before_replay["events"]} >= {
        "agent_permission", "permission_denied", "allow",
    }
    assert policy_before_replay["calls"] == 3
    assert policy_before_replay["failures"] == 1
    assert replay.ok is True
    assert replay.attempted == 3
    assert replay.skipped == 3
    assert receipt.success is True
    assert receipt.sink_result["api_key"] == "***"
    assert published[0]["evidence"]["trace_fingerprint"] == replay.trace_fingerprint
    assert manager.tools == []
    assert (await manager.dispatch(names["observe"], {}))["ok"] is False


@pytest.mark.asyncio
async def test_manager_execution_policy_is_in_the_real_dispatch_path(tmp_path):
    server = tmp_path / "policy_server.py"
    _write_policy_server(server)
    spec = CapabilitySpec(
        name="bounded",
        argv=(sys.executable, str(server)),
        allowed_tools=("observe",),
        tool_permissions=(("observe", "read_only"),),
    )
    manager = CapabilityManager(
        str(tmp_path),
        execution_policy=ExecutionPolicy(
            limits=ExecutionLimits(max_calls=1, max_result_bytes=512),
        ),
    )
    await manager.start((spec,))
    name = manager.tools[0]["name"]
    try:
        secret = await manager.dispatch(name, {"api_key": "never-send"})
        first = await manager.dispatch(name, {"subject": "safe"})
        exhausted = await manager.dispatch(name, {})
        snapshot = await manager.execution_policy_snapshot()
    finally:
        await manager.close()

    assert secret["policy_code"] == "secret_argument"
    assert "never-send" not in str(manager.execution_trace)
    assert first["ok"] is True
    assert exhausted["policy_code"] == "call_budget"
    assert snapshot["calls"] == 1
    assert [event["dispatched"] for event in manager.execution_trace["events"]] == [
        False, True, False,
    ]


@pytest.mark.asyncio
async def test_manager_supports_replaceable_risk_approval_and_result_atoms(tmp_path):
    server = tmp_path / "policy_server.py"
    _write_policy_server(server)
    spec = CapabilitySpec(
        name="replaceable",
        argv=(sys.executable, str(server)),
        allowed_tools=("observe",),
        tool_permissions=(("observe", "read_only"),),
    )
    denied = CapabilityManager(
        str(tmp_path),
        PermissionLevel.WORKSPACE_WRITE,
        risk_resolvers={"observe": lambda _args: PermissionLevel.DANGER_FULL},
    )
    await denied.start((spec,))
    denied_result = await denied.dispatch(denied.tools[0]["name"], {})
    await denied.close()

    approvals = []

    async def approve(request):
        approvals.append(request)
        return ApprovalDecision(True, approver_ref="operator:test")

    approved = CapabilityManager(
        str(tmp_path),
        PermissionLevel.DANGER_FULL,
        execution_policy=ExecutionPolicy(
            limits=ExecutionLimits(max_result_bytes=64),
            approval_level=PermissionLevel.DANGER_FULL,
        ),
        approval_resolver=approve,
        risk_resolvers={"observe": lambda _args: PermissionLevel.DANGER_FULL},
    )
    await approved.start((spec,))
    result = await approved.dispatch(
        approved.tools[0]["name"], {"subject": "x" * 80},
    )
    await approved.close()

    assert denied_result["policy_code"] == "permission_denied"
    assert result["policy_code"] == "result_budget"
    assert approvals[0].required_level == PermissionLevel.DANGER_FULL
    assert approved.execution_trace["events"][0]["policy_code"] == "result_budget"


@pytest.mark.asyncio
async def test_manager_cancellation_releases_lease_and_records_evidence(tmp_path):
    class SlowSession(_FakeSession):
        def __init__(self):
            super().__init__(
                [{"name": "cap_slow_wait", "inputSchema": {"type": "object"}}],
                {"cap_slow_wait": "wait"},
            )
            self.entered = asyncio.Event()

        async def dispatch(self, provider_name, arguments):
            self.entered.set()
            await asyncio.Event().wait()

    session = SlowSession()
    manager = CapabilityManager(str(tmp_path))
    manager._registry.register_session(
        session,
        _spec("slow", ("wait",), (("wait", "read_only"),)),
    )
    task = asyncio.create_task(manager.dispatch("cap_slow_wait", {}))
    await session.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    snapshot = await manager.execution_policy_snapshot()
    assert snapshot["active"] == 0
    assert snapshot["failures"] == 1
    assert manager.execution_trace["events"][0]["policy_code"] == "cancelled"


@pytest.mark.asyncio
async def test_manager_fails_closed_when_trace_budget_is_exhausted(tmp_path):
    session = _FakeSession(
        [{"name": "cap_trace_observe", "inputSchema": {"type": "object"}}],
        {"cap_trace_observe": "observe"},
    )
    manager = CapabilityManager(
        str(tmp_path), trace_ledger=ExecutionTraceLedger(max_events=1),
    )
    manager._registry.register_session(
        session,
        _spec("trace", ("observe",), (("observe", "read_only"),)),
    )
    assert (await manager.dispatch("cap_trace_observe", {}))["ok"] is True
    blocked = await manager.dispatch("cap_trace_observe", {})
    assert blocked["policy_code"] == "trace_unavailable"
