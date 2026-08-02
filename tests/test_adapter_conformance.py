# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Reusable adapter conformance and cross-domain scenario matrix tests."""
from __future__ import annotations

import sys

import pytest

from flyto_ai.coding.conformance import (
    AdapterConformanceCase,
    run_adapter_conformance,
)
from flyto_ai.coding.contracts import CapabilitySpec
from flyto_ai.coding.scenario_matrix import AdapterScenario, run_scenario_matrix
from flyto_ai.permissions import PermissionLevel


def _write_domain_adapter(path):
    path.write_text(
        "import json, sys\n"
        "domain=sys.argv[1]\n"
        "tools=sys.argv[2].split(',')\n"
        "for line in sys.stdin:\n"
        " msg=json.loads(line)\n"
        " if 'id' not in msg: continue\n"
        " method=msg.get('method')\n"
        " if method=='initialize': result={'protocolVersion':'2025-06-18','serverInfo':{'name':domain}}\n"
        " elif method=='tools/list': result={'tools':[{'name':name,'inputSchema':{'type':'object'}} for name in tools]}\n"
        " elif method=='tools/call':\n"
        "  args=msg['params'].get('arguments',{})\n"
        "  result={'structuredContent':{'ok':not args.get('fail',False),'domain':domain,'tool':msg['params']['name'],'arguments':args}}\n"
        " else: result={}\n"
        " print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':result}),flush=True)\n"
    )


def _spec(server, name, tools, level="workspace_write"):
    return CapabilitySpec(
        name=name,
        argv=(sys.executable, str(server), name, ",".join(tools)),
        required=True,
        required_tools=tuple(tools),
        allowed_tools=tuple(tools),
        tool_permissions=tuple((tool, level) for tool in tools),
    )


def _domain_verifier(domain, tool):
    def verify(result):
        evidence = result["result"]["structuredContent"]
        return evidence["domain"] == domain and evidence["tool"] == tool

    return verify


@pytest.mark.asyncio
async def test_adapter_conformance_closes_contract_runtime_domain_and_lifecycle(tmp_path):
    server = tmp_path / "domain_adapter.py"
    _write_domain_adapter(server)
    spec = _spec(server, "workflow", ("plan", "execute"))
    report = await run_adapter_conformance(
        str(tmp_path),
        spec,
        (
            AdapterConformanceCase(
                "plan", "plan", {"task": "fixture"},
                verifier=_domain_verifier("workflow", "plan"),
            ),
            AdapterConformanceCase(
                "domain-failure", "execute", {"fail": True}, expected_ok=False,
            ),
        ),
        permission_level=PermissionLevel.DANGER_FULL,
    )
    assert report.ok is True
    assert len(report.fingerprint) == 64
    assert len(report.trace_fingerprint) == 64
    assert report.trace_event_count == 2
    assert report.policy_calls == 2
    assert {check.name for check in report.checks} >= {
        "permissions_exhaustive", "catalog_exact", "case:plan", "closed",
        "evidence_chain", "policy_leases_released", "tools_covered",
    }


@pytest.mark.asyncio
async def test_adapter_conformance_fails_when_permission_contract_is_incomplete(tmp_path):
    server = tmp_path / "domain_adapter.py"
    _write_domain_adapter(server)
    spec = CapabilitySpec(
        name="incomplete",
        argv=(sys.executable, str(server), "incomplete", "observe"),
        required_tools=("observe",),
        allowed_tools=("observe",),
    )
    report = await run_adapter_conformance(
        str(tmp_path), spec, (AdapterConformanceCase("observe", "observe", {}),),
    )
    assert report.ok is False
    assert next(
        check for check in report.checks if check.name == "permissions_exhaustive"
    ).passed is False
    assert report.as_dict()["ok"] is False


@pytest.mark.asyncio
async def test_domain_neutral_matrix_closes_workflow_page_robotics_and_security_lab(tmp_path):
    server = tmp_path / "domain_adapter.py"
    _write_domain_adapter(server)
    definitions = (
        ("workflow", "general-workflow", "execute", {"task": "triage"}, "workspace_write"),
        ("page", "page-inspection", "inspect", {"fixture": "local-page"}, "read_only"),
        ("robotics", "robotics-simulation", "simulate", {"world": "fixture"}, "workspace_write"),
        (
            "security",
            "authorized-security-lab",
            "probe",
            {"scope_attested": True, "target": "fixture.invalid"},
            "danger_full",
        ),
    )
    scenarios = tuple(
        AdapterScenario(
            scenario_id=scenario_id,
            domain=domain,
            spec=_spec(server, domain, (tool,), level),
            cases=(AdapterConformanceCase(
                "closed-loop",
                tool,
                arguments,
                verifier=_domain_verifier(domain, tool),
            ),),
        )
        for scenario_id, domain, tool, arguments, level in definitions
    )
    report = await run_scenario_matrix(
        str(tmp_path),
        scenarios,
        max_concurrency=2,
        permission_level=PermissionLevel.DANGER_FULL,
    )
    assert report.ok is True
    assert len(report.results) == 4
    assert {result.domain for result in report.results} == {
        "general-workflow",
        "page-inspection",
        "robotics-simulation",
        "authorized-security-lab",
    }
    assert len(report.fingerprint) == 64
    assert report.as_dict()["contract_version"] == "flyto.adapter-scenario-matrix.v1"


def test_scenario_matrix_contract_rejects_duplicate_ids():
    spec = CapabilitySpec(
        name="fixture",
        argv=("fixture",),
        required_tools=("observe",),
        allowed_tools=("observe",),
        tool_permissions=(("observe", "read_only"),),
    )
    scenario = AdapterScenario(
        "duplicate", "fixture", spec,
        (AdapterConformanceCase("observe", "observe", {}),),
    )
    with pytest.raises(ValueError, match="duplicate scenario ids"):
        # Contract validation runs before the manager factory is used.
        import asyncio
        asyncio.run(run_scenario_matrix(".", (scenario, scenario)))


def test_scenario_and_matrix_reject_invalid_contract_boundaries():
    spec = CapabilitySpec(
        name="fixture",
        argv=("fixture",),
        allowed_tools=("observe",),
        tool_permissions=(("observe", "read_only"),),
    )
    case = AdapterConformanceCase("observe", "observe", {})
    with pytest.raises(ValueError, match="scenario_id"):
        AdapterScenario("", "fixture", spec, (case,))
    with pytest.raises(TypeError, match="CapabilitySpec"):
        AdapterScenario("fixture", "fixture", object(), (case,))
    with pytest.raises(ValueError, match="cases"):
        AdapterScenario("fixture", "fixture", spec, None)

    import asyncio
    with pytest.raises(ValueError, match="between 1 and 64"):
        asyncio.run(run_scenario_matrix(".", ()))
    with pytest.raises(TypeError, match="invalid scenario"):
        asyncio.run(run_scenario_matrix(".", (object(),)))
    scenario = AdapterScenario("fixture", "fixture", spec, (case,))
    with pytest.raises(ValueError, match="must be an integer"):
        asyncio.run(run_scenario_matrix(".", (scenario,), max_concurrency=True))
    with pytest.raises(ValueError, match="outside"):
        asyncio.run(run_scenario_matrix(".", (scenario,), max_concurrency=33))
    with pytest.raises(ValueError, match="must be callable"):
        asyncio.run(run_scenario_matrix(".", (scenario,), manager_factory=None))


def test_conformance_cases_are_immutable_json_and_uniquely_named(tmp_path):
    arguments = {"nested": {"items": [1, 2]}}
    case = AdapterConformanceCase("observe", "observe", arguments)
    arguments["nested"]["items"].append(3)
    assert case.arguments["nested"]["items"] == (1, 2)
    with pytest.raises(TypeError):
        case.arguments["nested"]["changed"] = True

    server = tmp_path / "domain_adapter.py"
    _write_domain_adapter(server)
    spec = _spec(server, "unique", ("observe",))
    with pytest.raises(ValueError, match="must be unique"):
        import asyncio
        asyncio.run(run_adapter_conformance(str(tmp_path), spec, (case, case)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": ""}, "case name"),
        ({"remote_tool": ""}, "remote_tool"),
        ({"arguments": []}, "arguments must"),
        ({"expected_ok": 1}, "expected_ok"),
        ({"expected_dispatched": 1}, "expected_dispatched"),
        ({"verifier": True}, "verifier"),
        ({"arguments": {"value": float("nan")}}, "finite JSON"),
    ],
)
def test_conformance_case_contract_rejects_invalid_values(kwargs, message):
    values = {"name": "case", "remote_tool": "observe", "arguments": {}}
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        AdapterConformanceCase(**values)


def test_conformance_runner_rejects_invalid_suite_boundaries(tmp_path):
    case = AdapterConformanceCase("observe", "observe", {})
    command = CapabilitySpec(name="command", argv=(sys.executable,), kind="command")
    import asyncio
    with pytest.raises(TypeError, match="CapabilitySpec"):
        asyncio.run(run_adapter_conformance(str(tmp_path), object(), (case,)))
    with pytest.raises(ValueError, match="MCP stdio"):
        asyncio.run(run_adapter_conformance(str(tmp_path), command, (case,)))

    spec = CapabilitySpec(
        name="fixture",
        argv=(sys.executable,),
        allowed_tools=("observe",),
        tool_permissions=(("observe", "read_only"),),
    )
    with pytest.raises(ValueError, match="between 1 and 256"):
        asyncio.run(run_adapter_conformance(str(tmp_path), spec, ()))
    with pytest.raises(TypeError, match="invalid type"):
        asyncio.run(run_adapter_conformance(str(tmp_path), spec, (object(),)))
    with pytest.raises(TypeError, match="invalid type"):
        asyncio.run(run_adapter_conformance(str(tmp_path), spec, None))
    with pytest.raises(ValueError, match="must be callable"):
        asyncio.run(run_adapter_conformance(
            str(tmp_path), spec, (case,), manager_factory=None,
        ))


@pytest.mark.asyncio
async def test_conformance_supports_async_verifier_and_bounds_verifier_failure(tmp_path):
    server = tmp_path / "domain_adapter.py"
    _write_domain_adapter(server)
    spec = _spec(server, "async", ("observe",), level="read_only")

    async def verify(result):
        return result["result"]["structuredContent"]["arguments"] == {
            "items": [1, 2],
        }

    passing = await run_adapter_conformance(
        str(tmp_path),
        spec,
        (AdapterConformanceCase(
            "async-verifier", "observe", {"items": [1, 2]}, verifier=verify,
        ),),
    )
    assert passing.ok is True

    def failed(_result):
        raise RuntimeError("private-verifier-detail")

    bounded = await run_adapter_conformance(
        str(tmp_path),
        spec,
        (AdapterConformanceCase(
            "failed-verifier", "observe", {}, verifier=failed,
        ),),
    )
    assert bounded.ok is False
    runtime = next(check for check in bounded.checks if check.name == "runtime_exception")
    assert runtime.detail == "RuntimeError"
    assert "private-verifier-detail" not in str(bounded.as_dict())


@pytest.mark.asyncio
async def test_conformance_defaults_to_read_only_and_cannot_mistake_denial_for_failure(tmp_path):
    server = tmp_path / "domain_adapter.py"
    _write_domain_adapter(server)
    spec = _spec(server, "danger", ("act",), level="danger_full")
    report = await run_adapter_conformance(
        str(tmp_path),
        spec,
        (AdapterConformanceCase("denied", "act", {}, expected_ok=False),),
    )
    case_check = next(check for check in report.checks if check.name == "case:denied")
    assert case_check.passed is False
    assert "dispatched=False" in case_check.detail

    policy_case = AdapterConformanceCase(
        "expected-policy-denial",
        "act",
        {},
        expected_ok=False,
        expected_dispatched=False,
    )
    policy_report = await run_adapter_conformance(str(tmp_path), spec, (policy_case,))
    assert policy_report.ok is True
    assert policy_report.policy_calls == 0
