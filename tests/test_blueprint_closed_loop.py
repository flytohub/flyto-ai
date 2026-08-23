"""Closed-loop coverage for Blueprint → Agent safety → Core → feedback."""
import json

import pytest

from flyto_ai.agent import Agent
from flyto_ai.assistant.router import feedback
from flyto_ai.blueprint_loop import execute_blueprint_loop
from flyto_ai.closed_loop_v3 import (
    BenchmarkCase,
    CapabilityModelRouter,
    JsonCheckpointStore,
    ModelCandidate,
    PlanIR,
    evaluate_distillation,
    run_model_benchmark,
)
from flyto_ai.config import AgentConfig
from flyto_ai.intelligence.planner import extract_intent
from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
from flyto_ai.providers.base import dispatch_and_log_tool
from flyto_ai.providers.failover import ProviderChain
from flyto_ai.tools import blueprint_tools
from flyto_ai.tools.blueprint_tools import dispatch_blueprint_tool
from flyto_ai.execution_verification import build_execution_verification_receipt


def _bare_agent(dispatch, permission_level="workspace_write"):
    """Construct only the Agent state needed by the safe/deterministic paths."""
    agent = Agent.__new__(Agent)
    agent._config = AgentConfig(
        provider="ollama",
        enable_injection_detection=False,
        enable_memory=False,
        enable_pro=False,
        enable_transcript=False,
        permission_level=permission_level,
    )
    agent._provider = object()
    agent._dispatch_fn = dispatch
    agent._policies = None
    agent._permission_enforcer = PermissionEnforcer(
        level=PermissionLevel[permission_level.upper()],
    )
    agent._assistant = None
    agent._hooks = None
    agent._pro = None
    agent._tools = [{"name": "use_blueprint"}]
    agent._trusted_blueprint_resolver = lambda blueprint_id: {
        "id": blueprint_id,
        "trust_tier": "local_verified",
    }
    agent._cost_tracker = None
    agent._session_id = "closed-loop-test"
    return agent


def _bound_outcome(arguments, **extra):
    return {
        "ok": True,
        "blueprint_id": arguments["blueprint_id"],
        "execution_id": arguments["execution_id"],
        "evidence_tier": "local_verified",
        **extra,
    }


@pytest.mark.asyncio
async def test_blueprint_loop_validates_executes_and_reports_once():
    calls = []

    async def dispatch(name, arguments):
        calls.append((name, arguments))
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            return {
                "status": "success",
                "data": {"value": arguments["params"]["text"].upper()},
            }
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments, score=75)
        raise AssertionError(name)

    result = await execute_blueprint_loop(
        blueprint_id="bp_upper",
        steps=[
            {"module": "string.uppercase", "params": {"text": "one"}},
            {"module": "string.uppercase", "params": {"text": "two"}},
        ],
        dispatch=dispatch,
    )

    assert [name for name, _ in calls] == [
        "validate_params",
        "validate_params",
        "execute_module",
        "execute_module",
        "report_blueprint_outcome",
    ]
    assert result["ok"] is True
    assert result["closed_loop_ok"] is True
    assert result["outcome_reported"] is True
    assert result["evidence"]["step_count"] == 2
    assert result["evidence"]["passed_steps"] == 2
    assert calls[-1][1]["execution_id"] == result["execution_id"]
    runtime_evidence = calls[-1][1]["_execution_evidence"]
    assert runtime_evidence["step_count"] == 2
    assert runtime_evidence["total_attempts"] == 2
    assert runtime_evidence["assertion_passed"] is True
    assert runtime_evidence["selection_mode"] == "model_selected"
    assert "planner_model_calls_used" not in runtime_evidence
    assert "model_calls_used" not in runtime_evidence
    assert "model_call_scope" not in runtime_evidence


@pytest.mark.asyncio
async def test_identity_free_ok_response_does_not_close_loop():
    async def dispatch(name, arguments):
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            return {"ok": True}
        if name == "report_blueprint_outcome":
            return {"ok": True}
        raise AssertionError(name)

    result = await execute_blueprint_loop(
        "identity_required",
        [{"module": "string.uppercase", "params": {"text": "safe"}}],
        dispatch,
    )

    assert result["ok"] is True
    assert result["outcome_reported"] is False
    assert result["closed_loop_ok"] is False


@pytest.mark.asyncio
async def test_through_tool_identity_free_trusted_response_fails_closed(monkeypatch):
    class IdentityFreeEngine:
        def report_outcome(self, **_arguments):
            return {"ok": True, "evidence_tier": "local_verified"}

    monkeypatch.setattr("flyto_blueprint.get_engine", lambda: IdentityFreeEngine())

    async def dispatch(name, arguments):
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            return {"ok": True}
        return await dispatch_blueprint_tool(name, arguments)

    result = await execute_blueprint_loop(
        "identity_required_through_tool",
        [{"module": "string.uppercase", "params": {"text": "safe"}}],
        dispatch,
    )

    assert result["outcome_reported"] is False
    assert result["closed_loop_ok"] is False


@pytest.mark.asyncio
async def test_unsafe_step_receipt_falls_back_to_community_and_keeps_checkpoint(
    monkeypatch, tmp_path,
):
    class CommunityEchoEngine:
        def __init__(self):
            self.calls = []

        def report_outcome(self, **arguments):
            self.calls.append(arguments)
            return {
                "ok": True,
                "blueprint_id": arguments["blueprint_id"],
                "execution_id": arguments["execution_id"],
                "evidence_tier": arguments["evidence_tier"],
            }

    engine = CommunityEchoEngine()
    monkeypatch.setattr("flyto_blueprint.get_engine", lambda: engine)

    async def dispatch(name, arguments):
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            return {"ok": True}
        return await dispatch_blueprint_tool(name, arguments)

    result = await execute_blueprint_loop(
        "community_fallback",
        [{
            "id": "unsafe step id",
            "module": "string.uppercase",
            "params": {"text": "safe"},
        }],
        dispatch,
        checkpoint_store=JsonCheckpointStore(str(tmp_path)),
    )

    assert result["ok"] is True
    assert result["outcome"]["evidence_tier"] == "community"
    assert result["outcome_reported"] is False
    assert result["closed_loop_ok"] is False
    assert result["evidence"]["checkpoint_cleared"] is False
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert engine.calls[0]["evidence_tier"] == "community"
    assert "verification" not in engine.calls[0]


@pytest.mark.asyncio
async def test_blueprint_loop_stops_on_validation_failure_and_reports_failure():
    calls = []

    async def dispatch(name, arguments):
        calls.append((name, arguments))
        if name == "validate_params":
            return {"valid": False, "errors": ["text is required"]}
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments)
        raise AssertionError("Execution must stop after failed validation")

    result = await execute_blueprint_loop(
        blueprint_id="bp_invalid",
        steps=[{"module": "string.uppercase", "params": {}}],
        dispatch=dispatch,
    )

    assert [name for name, _ in calls] == [
        "validate_params",
        "report_blueprint_outcome",
    ]
    assert result["ok"] is False
    assert result["closed_loop_ok"] is False
    assert result["evidence"]["validation_passed"] is False
    assert result["evidence"]["failed_module"] == "string.uppercase"
    assert calls[-1][1]["success"] is False


@pytest.mark.asyncio
async def test_whole_workflow_preflight_prevents_partial_side_effects():
    calls = []

    async def dispatch(name, arguments):
        calls.append((name, arguments))
        if name == "validate_params":
            return {
                "valid": arguments["module_id"] != "string.reverse",
                "errors": ["invalid reverse params"],
            }
        if name == "execute_module":
            raise AssertionError("Preflight failure must prevent every side effect")
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments)
        raise AssertionError(name)

    result = await execute_blueprint_loop(
        blueprint_id="bp_preflight",
        steps=[
            {
                "id": "upper",
                "module": "string.uppercase",
                "params": {"text": "hello"},
            },
            {
                "id": "reverse",
                "module": "string.reverse",
                "params": {"text": "hello"},
            },
        ],
        dispatch=dispatch,
    )

    assert [name for name, _ in calls] == [
        "validate_params",
        "validate_params",
        "report_blueprint_outcome",
    ]
    assert result["ok"] is False
    assert result["evidence"]["preflight_passed"] is False
    assert result["evidence"]["side_effects_started"] is False
    assert result["evidence"]["executed_steps"] == 0
    assert result["evidence"]["failed_step_id"] == "reverse"


@pytest.mark.asyncio
async def test_blueprint_loop_resolves_core_step_outputs_and_asserts_result():
    calls = []

    async def dispatch(name, arguments):
        calls.append((name, arguments))
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            if arguments["module_id"] == "string.uppercase":
                return {"status": "success", "data": {"result": "HELLO"}}
            assert arguments["params"] == {"text": "HELLO"}
            return {"status": "success", "data": {"result": "OLLEH"}}
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments)
        raise AssertionError(name)

    result = await execute_blueprint_loop(
        blueprint_id="bp_dataflow",
        steps=[
            {
                "id": "upper",
                "module": "string.uppercase",
                "params": {"text": "hello"},
            },
            {
                "id": "reverse",
                "module": "string.reverse",
                "params": {"text": "${upper.result}"},
                "assertions": {
                    "path": "data.result",
                    "op": "equals",
                    "value": "OLLEH",
                },
            },
        ],
        dispatch=dispatch,
    )

    assert [name for name, _ in calls] == [
        "validate_params",
        "execute_module",
        "validate_params",
        "execute_module",
        "report_blueprint_outcome",
    ]
    assert result["closed_loop_ok"] is True
    assert result["evidence"]["deferred_validation_steps"] == ["reverse"]
    assert result["evidence"]["assertion_passed"] is True
    assert result["evidence"]["completed_step_ids"] == ["upper", "reverse"]
    assert result["executions"][1]["assertions"][0]["ok"] is True


@pytest.mark.asyncio
async def test_blueprint_loop_retries_failed_result_with_bounded_policy():
    execute_count = 0

    async def dispatch(name, arguments):
        nonlocal execute_count
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            execute_count += 1
            if execute_count < 3:
                return {"ok": False, "error": "transient"}
            return {"status": "success", "data": {"result": "done"}}
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments)
        raise AssertionError(name)

    result = await execute_blueprint_loop(
        blueprint_id="bp_retry",
        steps=[{
            "id": "unstable",
            "module": "string.uppercase",
            "params": {"text": "hello"},
            "retry": {"count": 2, "delay_ms": 0, "backoff": "exponential"},
        }],
        dispatch=dispatch,
    )

    assert result["closed_loop_ok"] is True
    assert execute_count == 3
    assert result["evidence"]["total_attempts"] == 3
    assert result["executions"][0]["attempt_count"] == 3
    assert [item["ok"] for item in result["executions"][0]["attempts"]] == [
        False,
        False,
        True,
    ]


@pytest.mark.asyncio
async def test_assertion_failure_fails_outcome_and_emits_resume_point():
    report_args = None

    async def dispatch(name, arguments):
        nonlocal report_args
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            return {"status": "success", "data": {"result": "unexpected"}}
        if name == "report_blueprint_outcome":
            report_args = arguments
            return _bound_outcome(arguments)
        raise AssertionError(name)

    result = await execute_blueprint_loop(
        blueprint_id="bp_assertion",
        steps=[{
            "id": "verify_output",
            "module": "string.uppercase",
            "params": {"text": "hello"},
            "assert": {
                "path": "data.result",
                "op": "equals",
                "value": "HELLO",
            },
        }],
        dispatch=dispatch,
    )

    assert result["ok"] is False
    assert result["evidence"]["failed_phase"] == "assertion"
    assert result["evidence"]["assertion_passed"] is False
    assert result["evidence"]["resume_from_step_id"] == "verify_output"
    assert result["executions"][0]["assertions"][0]["ok"] is False
    assert report_args["success"] is False


@pytest.mark.asyncio
async def test_forward_reference_fails_structural_preflight():
    calls = []

    async def dispatch(name, arguments):
        calls.append((name, arguments))
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments)
        raise AssertionError("Structural failure must not reach Core")

    result = await execute_blueprint_loop(
        blueprint_id="bp_forward_ref",
        steps=[
            {
                "id": "first",
                "module": "string.uppercase",
                "params": {"text": "${later.result}"},
            },
            {
                "id": "later",
                "module": "string.reverse",
                "params": {"text": "hello"},
            },
        ],
        dispatch=dispatch,
    )

    assert [name for name, _ in calls] == ["report_blueprint_outcome"]
    assert result["evidence"]["preflight_passed"] is False
    assert "forward step reference" in (
        result["evidence"]["preflight_errors"][0]["error"]
    )


@pytest.mark.asyncio
async def test_agent_auto_executes_safe_blueprint_through_nested_dispatch():
    calls = []

    async def raw_dispatch(name, arguments):
        calls.append((name, arguments))
        if name == "use_blueprint":
            return {
                "ok": True,
                "blueprint_id": "bp_safe",
                "steps": [
                    {"module": "string.uppercase", "params": {"text": "hello"}},
                ],
            }
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            return {"status": "success", "data": {"value": "HELLO"}}
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments)
        raise AssertionError(name)

    result = await _bare_agent(raw_dispatch)._make_safe_dispatch()(
        "use_blueprint",
        {"blueprint_id": "bp_safe", "args": {"text": "hello"}},
    )

    assert [name for name, _ in calls] == [
        "use_blueprint",
        "validate_params",
        "execute_module",
        "report_blueprint_outcome",
    ]
    assert result["closed_loop_ok"] is True
    assert result["executions"][0]["module_id"] == "string.uppercase"


@pytest.mark.asyncio
async def test_nested_blueprint_execution_cannot_bypass_agent_permission():
    calls = []

    async def raw_dispatch(name, arguments):
        calls.append((name, arguments))
        if name == "use_blueprint":
            return {
                "ok": True,
                "blueprint_id": "bp_danger",
                "steps": [
                    {"module": "shell.run", "params": {"command": "whoami"}},
                ],
            }
        if name == "validate_params":
            return {"valid": True}
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments)
        if name == "execute_module":
            raise AssertionError("Dangerous module reached the raw dispatcher")
        raise AssertionError(name)

    result = await _bare_agent(raw_dispatch)._make_safe_dispatch()(
        "use_blueprint",
        {"blueprint_id": "bp_danger", "args": {}},
    )

    assert result["ok"] is False
    assert result["closed_loop_ok"] is False
    assert result["evidence"]["failed_module"] == "shell.run"
    assert not any(name == "execute_module" for name, _ in calls)
    report_args = next(args for name, args in calls if name == "report_blueprint_outcome")
    assert report_args["success"] is False


@pytest.mark.asyncio
async def test_agent_preflights_later_danger_before_safe_step_side_effect():
    calls = []

    async def raw_dispatch(name, arguments):
        calls.append((name, arguments))
        if name == "use_blueprint":
            return {
                "ok": True,
                "blueprint_id": "bp_late_danger",
                "steps": [
                    {
                        "id": "safe_first",
                        "module": "string.uppercase",
                        "params": {"text": "hello"},
                    },
                    {
                        "id": "danger_later",
                        "module": "shell.run",
                        "params": {"command": "whoami"},
                    },
                ],
            }
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments)
        if name in {"validate_params", "execute_module"}:
            raise AssertionError("Access preflight must happen before Core calls")
        raise AssertionError(name)

    result = await _bare_agent(raw_dispatch)._make_safe_dispatch()(
        "use_blueprint",
        {"blueprint_id": "bp_late_danger", "args": {}},
    )

    assert [name for name, _ in calls] == [
        "use_blueprint",
        "report_blueprint_outcome",
    ]
    assert result["evidence"]["preflight_passed"] is False
    assert result["evidence"]["side_effects_started"] is False
    assert result["evidence"]["failed_step_id"] == "danger_later"


@pytest.mark.asyncio
async def test_provider_log_preserves_nested_execution_evidence():
    async def dispatch(_name, _arguments):
        return {
            "ok": True,
            "closed_loop_ok": True,
            "blueprint_id": "bp_logged",
            "execution_id": "bp_execution_1",
            "outcome_reported": True,
            "evidence": {"step_count": 1, "passed_steps": 1},
            "executions": [
                {
                    "function": "execute_module",
                    "module_id": "string.uppercase",
                    "ok": True,
                },
            ],
            "results": [{"data": "model-visible raw result"}],
        }

    result_str, log_entry, images = await dispatch_and_log_tool(
        "use_blueprint",
        {"blueprint_id": "bp_logged", "args": {}},
        dispatch,
        round_num=0,
    )

    assert json.loads(result_str)["results"]
    assert images == []
    assert log_entry["ok"] is True
    assert log_entry["execution_id"] == "bp_execution_1"
    assert log_entry["evidence"]["passed_steps"] == 1
    assert log_entry["executions"][0]["module_id"] == "string.uppercase"
    assert "results" not in log_entry


def test_planner_matches_blueprint_summary_and_extracts_explicit_args(monkeypatch):
    class FakeEngine:
        def search(self, _message):
            return [{
                "id": "learned_copy",
                "score": 70,
                "use_count": 2,
                "trust_tier": "local_verified",
                "evidence_card": {
                    "sample_count": 3,
                    "success_count": 3,
                    "success_rate": 1.0,
                },
                "args": {
                    "text": {"type": "string", "required": True},
                    "count": {"type": "integer", "required": True},
                },
            }]

    import flyto_blueprint

    monkeypatch.setattr(flyto_blueprint, "get_engine", lambda: FakeEngine())

    result = extract_intent("repeat text=Flyto count=3")

    assert result == {
        "intent": "blueprint",
        "blueprint_id": "learned_copy",
        "args": {"text": "Flyto", "count": 3},
        "selection_evidence": {
            "trust_tier": "local_verified",
            "sample_count": 3,
            "success_count": 3,
            "success_rate": 1.0,
        },
    }


@pytest.mark.asyncio
async def test_deterministic_blueprint_reuse_is_zero_planner_model_calls(monkeypatch):
    calls = []

    async def raw_dispatch(name, arguments):
        calls.append((name, arguments))
        if name == "use_blueprint":
            return {
                "ok": True,
                "blueprint_id": "learned_copy",
                "steps": [
                    {"module": "string.uppercase", "params": {"text": "hello"}},
                ],
            }
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            return {"status": "success", "data": {"value": "HELLO"}}
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments)
        raise AssertionError(name)

    from flyto_ai.intelligence import planner

    monkeypatch.setattr(
        planner,
        "extract_intent",
        lambda _message: {
            "intent": "blueprint",
            "blueprint_id": "learned_copy",
            "args": {"text": "hello"},
        },
    )

    response = await _bare_agent(raw_dispatch)._try_deterministic(
        "repeat text=hello",
        on_tool_call=None,
        on_stream=None,
        dispatch_wrapper=None,
    )

    assert response is not None
    assert response.ok is True
    assert response.model == "deterministic"
    assert response.rounds_used == 0
    assert response.execution_results[0]["module_id"] == "string.uppercase"
    assert response.tool_calls[0]["outcome_reported"] is True
    assert response.tool_calls[0]["evidence"]["selection_mode"] == "deterministic"
    assert response.tool_calls[0]["evidence"]["planner_model_calls_used"] == 0
    assert "model_calls_used" not in response.tool_calls[0]["evidence"]
    assert response.tool_calls[0]["evidence"]["model_call_scope"] == "planner"
    report_arguments = calls[-1][1]
    assert report_arguments["_execution_evidence"]["selection_mode"] == "deterministic"
    assert "planner_model_calls_used" not in report_arguments["_execution_evidence"]
    assert "model_calls_used" not in report_arguments["_execution_evidence"]
    assert "model_call_scope" not in report_arguments["_execution_evidence"]
    assert [name for name, _ in calls] == [
        "use_blueprint",
        "validate_params",
        "execute_module",
        "report_blueprint_outcome",
    ]


@pytest.mark.asyncio
async def test_planner_call_scope_does_not_claim_llm_step_is_token_free():
    calls = []

    async def dispatch(name, arguments):
        calls.append((name, arguments))
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            return {"status": "success", "data": {"value": "mocked"}}
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments)
        raise AssertionError(name)

    result = await execute_blueprint_loop(
        blueprint_id="model-backed-step",
        steps=[
            {
                "module": "llm.chat",
                "params": {"prompt": "This call is mocked in the test."},
            },
        ],
        dispatch=dispatch,
        selection_mode="deterministic",
    )

    runtime_evidence = calls[-1][1]["_execution_evidence"]
    assert result["ok"] is True
    assert "planner_model_calls_used" not in runtime_evidence
    assert "model_call_scope" not in runtime_evidence
    assert "model_calls_used" not in runtime_evidence
    assert "workflow_model_calls_used" not in runtime_evidence


def test_feedback_reused_blueprint_is_idempotent_and_not_relearned(monkeypatch):
    class FakeEngine:
        def __init__(self):
            self.reported = []
            self.learned = []

        def report_outcome(self, blueprint_id, success, execution_id=""):
            self.reported.append((blueprint_id, success, execution_id))
            return {"ok": True}

        def learn_from_execution(self, **kwargs):
            self.learned.append(kwargs)

    engine = FakeEngine()
    import flyto_blueprint

    monkeypatch.setattr(flyto_blueprint, "get_engine", lambda: engine)

    feedback(
        tool_calls=[{
            "function": "use_blueprint",
            "arguments": {"blueprint_id": "learned_copy"},
            "execution_id": "bp_execution_1",
        }],
        execution_results=[{
            "module_id": "string.uppercase",
            "arguments": {"params": {"text": "hello"}},
            "ok": True,
        }],
        user_message="repeat text=hello",
    )

    assert engine.reported == [("learned_copy", True, "bp_execution_1")]
    assert engine.learned == []


def test_feedback_does_not_verify_blueprint_without_execution_evidence(
    monkeypatch,
):
    class FakeEngine:
        def __init__(self):
            self.reported = []

        def report_outcome(self, *args, **kwargs):
            self.reported.append((args, kwargs))

    engine = FakeEngine()
    import flyto_blueprint

    monkeypatch.setattr(flyto_blueprint, "get_engine", lambda: engine)

    feedback(
        tool_calls=[{
            "function": "use_blueprint",
            "arguments": {"blueprint_id": "selected_only"},
            "execution_id": "bp_no_execution",
        }],
        execution_results=[],
        user_message="select but do not execute",
    )

    assert engine.reported == []


@pytest.mark.asyncio
async def test_blueprint_tool_forwards_execution_id(monkeypatch):
    class FakeEngine:
        def __init__(self):
            self.reported = None

        def report_outcome(
            self,
            blueprint_id,
            success,
            execution_id="",
            evidence_tier="local_verified",
            evidence=None,
            verification=None,
        ):
            self.reported = (
                blueprint_id,
                success,
                execution_id,
                evidence_tier,
                evidence,
            )
            return {"ok": True}

    engine = FakeEngine()
    import flyto_blueprint

    monkeypatch.setattr(flyto_blueprint, "get_engine", lambda: engine)

    result = await dispatch_blueprint_tool(
        "report_blueprint_outcome",
        {
            "blueprint_id": "learned_copy",
            "success": True,
            "execution_id": "bp_execution_1",
        },
    )

    assert result == {"ok": True}
    assert engine.reported == (
        "learned_copy",
        True,
        "bp_execution_1",
        "community",
        None,
    )


@pytest.mark.asyncio
async def test_closed_loop_capability_marks_only_host_execution_verified(
    monkeypatch,
):
    class FakeEngine:
        def __init__(self):
            self.reported = []

        def report_outcome(
            self,
            blueprint_id,
            success,
            execution_id="",
            evidence_tier="local_verified",
            evidence=None,
            verification=None,
        ):
            self.reported.append({
                "blueprint_id": blueprint_id,
                "success": success,
                "execution_id": execution_id,
                "evidence_tier": evidence_tier,
                "evidence": evidence,
                "verification": verification,
            })
            return {
                "ok": True,
                "blueprint_id": blueprint_id,
                "evidence_tier": evidence_tier,
            }

    engine = FakeEngine()
    import flyto_blueprint

    monkeypatch.setattr(flyto_blueprint, "get_engine", lambda: engine)

    async def dispatch(name, arguments):
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            return {"status": "success", "data": {"result": "HELLO"}}
        return await dispatch_blueprint_tool(name, arguments)

    result = await execute_blueprint_loop(
        blueprint_id="learned_copy",
        steps=[{
            "module": "string.uppercase",
            "params": {"text": "hello"},
        }],
        dispatch=dispatch,
    )
    forged = await dispatch_blueprint_tool(
        "report_blueprint_outcome",
        {
            "blueprint_id": "learned_copy",
            "success": True,
            "execution_id": "forged",
            "_evidence_capability": "flyto-ai.closed-loop-verified",
            "_execution_evidence": {
                "model_calls_used": 0,
                "selection_mode": "deterministic",
            },
        },
    )

    assert result["closed_loop_ok"] is True
    assert result["outcome"]["evidence_tier"] == "local_verified"
    assert forged["evidence_tier"] == "community"
    assert [item["evidence_tier"] for item in engine.reported] == [
        "local_verified",
        "community",
    ]
    assert engine.reported[0]["evidence"]["step_count"] == 1
    assert engine.reported[0]["evidence"]["selection_mode"] == "model_selected"
    assert engine.reported[0]["verification"]["evidence"]["outcome_success"] is True
    assert engine.reported[1]["evidence"] is None


@pytest.mark.asyncio
async def test_real_blueprint_engine_trusted_outcome_is_bound_and_idempotent(
    monkeypatch,
):
    from flyto_blueprint import BlueprintEngine, MemoryBackend

    engine = BlueprintEngine(storage=MemoryBackend())
    workflow = {
        "name": "real trusted loop",
        "steps": [
            {"id": "one", "module": "string.uppercase", "params": {}},
            {"id": "two", "module": "string.lowercase", "params": {}},
            {"id": "three", "module": "string.reverse", "params": {}},
        ],
    }
    learning_receipt = build_execution_verification_receipt(
        "learning:real-loop",
        {
            "steps": [
                {"step_id": "one", "module_id": "string.uppercase"},
                {"step_id": "two", "module_id": "string.lowercase"},
                {"step_id": "three", "module_id": "string.reverse"},
            ],
            "checks": {"fixture_valid": True},
            "counts": {"step_count": 3},
            "structural_digest": "sha256:" + "a" * 64,
        },
    )
    learned = engine.learn_from_execution(workflow, verification=learning_receipt)
    blueprint_id = learned["data"]["id"]
    monkeypatch.setattr("flyto_blueprint.get_engine", lambda: engine)
    reports = []

    async def dispatch(name, arguments):
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            return {"ok": True, "data": {"result": "SAFE"}}
        reports.append(arguments)
        return await dispatch_blueprint_tool(name, arguments)

    before = engine._blueprints[blueprint_id]["score"]
    result = await execute_blueprint_loop(
        blueprint_id,
        workflow["steps"],
        dispatch,
    )
    trusted = engine._blueprints[blueprint_id]
    assert result["closed_loop_ok"] is True
    assert trusted["score"] == before + 5
    assert len(trusted["evidence_samples"]) == 1

    duplicate = await dispatch_blueprint_tool(
        "report_blueprint_outcome", reports[0],
    )
    assert duplicate["skipped"] == "already_reported"
    assert engine._blueprints[blueprint_id]["score"] == before + 5

    tampered = dict(reports[0])
    tampered["_verification_receipt"] = dict(reports[0]["_verification_receipt"])
    tampered["_verification_receipt"]["evidence_sha256"] = "0" * 64
    await dispatch_blueprint_tool("report_blueprint_outcome", tampered)
    forged = dict(reports[0])
    forged["execution_id"] = "copied_dictionary"
    forged["_evidence_capability"] = "flyto-ai.closed-loop-verified"
    await dispatch_blueprint_tool("report_blueprint_outcome", forged)
    assert engine._blueprints[blueprint_id]["score"] == before + 5
    assert len(engine._blueprints[blueprint_id]["evidence_samples"]) == 1


@pytest.mark.asyncio
async def test_blueprint_tool_dispatches_portable_bundles_without_host_keys(
    monkeypatch,
):
    class FakeEngine:
        def __init__(self):
            self.calls = []

        def export_blueprint(self, blueprint_id, publisher=""):
            self.calls.append(("export", blueprint_id, publisher))
            return {"ok": True, "bundle": {"blueprint": {"id": blueprint_id}}}

        def import_blueprint(self, bundle):
            self.calls.append(("import", bundle))
            return {"ok": True, "trust_tier": "community"}

    engine = FakeEngine()
    import flyto_blueprint

    monkeypatch.setattr(flyto_blueprint, "get_engine", lambda: engine)

    exported = await dispatch_blueprint_tool(
        "export_blueprint",
        {"blueprint_id": "portable", "publisher": "team-a"},
    )
    imported = await dispatch_blueprint_tool(
        "import_blueprint",
        {"bundle": exported["bundle"]},
    )

    assert imported == {"ok": True, "trust_tier": "community"}
    assert engine.calls == [
        ("export", "portable", "team-a"),
        ("import", {"blueprint": {"id": "portable"}}),
    ]


@pytest.mark.asyncio
async def test_blueprint_tool_preserves_expanded_execution_contract(monkeypatch):
    class FakeEngine:
        def expand(self, blueprint_id, args):
            assert blueprint_id == "learned_dataflow"
            assert args == {"text": "Flyto"}
            return {
                "ok": True,
                "data": {
                    "steps": [
                        {
                            "id": "upper",
                            "module": "string.uppercase",
                            "params": {"text": "Flyto"},
                            "retry": {"count": 1, "delay_ms": 0},
                            "assertions": {
                                "path": "data.result",
                                "op": "equals",
                                "value": "FLYTO",
                            },
                        },
                        {
                            "id": "reverse",
                            "module": "string.reverse",
                            "params": {"text": "${upper.result}"},
                        },
                    ],
                },
            }

    import flyto_blueprint

    monkeypatch.setattr(flyto_blueprint, "get_engine", lambda: FakeEngine())

    result = await dispatch_blueprint_tool(
        "use_blueprint",
        {
            "blueprint_id": "learned_dataflow",
            "args": {"text": "Flyto"},
        },
    )

    assert result["ok"] is True
    assert result["steps"] == [
        {
            "id": "upper",
            "module": "string.uppercase",
            "params": {"text": "Flyto"},
            "retry": {"count": 1, "delay_ms": 0},
            "assertions": {
                "path": "data.result",
                "op": "equals",
                "value": "FLYTO",
            },
        },
        {
            "id": "reverse",
            "module": "string.reverse",
            "params": {"text": "${upper.result}"},
        },
    ]


@pytest.mark.asyncio
async def test_real_core_dataflow_and_assertion_integration():
    pytest.importorskip("core")
    from flyto_ai.tools.core_tools import dispatch_core_tool

    async def dispatch(name, arguments):
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments)
        return await dispatch_core_tool(name, arguments)

    result = await execute_blueprint_loop(
        blueprint_id="bp_real_core_dataflow",
        steps=[
            {
                "id": "upper",
                "module": "string.uppercase",
                "params": {"text": "Flyto"},
            },
            {
                "id": "reverse",
                "module": "string.reverse",
                "params": {"text": "${upper.result}"},
                "assertions": {
                    "path": "data.result",
                    "op": "equals",
                    "value": "OTYLF",
                },
            },
        ],
        dispatch=dispatch,
    )

    assert result["closed_loop_ok"] is True
    assert result["evidence"]["executor_version"] == "blueprint-loop.v3"
    assert result["evidence"]["plan_ir_version"] == "flyto.plan-ir.v1"
    assert result["evidence"]["workflow_hash"].startswith("sha256:")
    assert result["evidence"]["completed_step_ids"] == ["upper", "reverse"]


def test_plan_ir_is_stable_and_fails_forward_references():
    steps = [
        {
            "id": "first",
            "module": "string.uppercase",
            "params": {"text": "${later.result}"},
        },
        {
            "id": "later",
            "module": "string.reverse",
            "params": {"text": "Flyto"},
        },
    ]

    first = PlanIR.compile("bp_plan", steps)
    second = PlanIR.compile("bp_plan", steps)

    assert first.workflow_hash == second.workflow_hash
    assert first.version == "flyto.plan-ir.v1"
    assert "forward step reference" in first.gate()[0]["error"].lower()


@pytest.mark.asyncio
async def test_checkpoint_resumes_without_replaying_completed_side_effect(tmp_path):
    store = JsonCheckpointStore(str(tmp_path))
    phase = "fail"
    execute_counts = {"upper": 0, "reverse": 0}

    async def dispatch(name, arguments):
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            if arguments["module_id"] == "string.uppercase":
                execute_counts["upper"] += 1
                return {"ok": True, "data": {"result": "FLYTO"}}
            execute_counts["reverse"] += 1
            if phase == "fail":
                return {"ok": False, "error": "temporary upstream failure"}
            assert arguments["params"]["text"] == "FLYTO"
            return {"ok": True, "data": {"result": "OTYLF"}}
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments)
        raise AssertionError(name)

    steps = [
        {
            "id": "upper",
            "module": "string.uppercase",
            "params": {"text": "Flyto"},
        },
        {
            "id": "reverse",
            "module": "string.reverse",
            "params": {"text": "${upper.result}"},
        },
    ]
    failed = await execute_blueprint_loop(
        "bp_resume",
        steps,
        dispatch,
        checkpoint_store=store,
        max_repairs=0,
    )

    assert failed["ok"] is False
    checkpoint_files = list(tmp_path.glob("*.json"))
    assert len(checkpoint_files) == 1
    assert checkpoint_files[0].stat().st_mode & 0o777 == 0o600

    phase = "success"
    resumed = await execute_blueprint_loop(
        "bp_resume",
        steps,
        dispatch,
        checkpoint_store=store,
        max_repairs=0,
    )

    assert resumed["closed_loop_ok"] is True
    assert execute_counts == {"upper": 1, "reverse": 2}
    assert resumed["evidence"]["checkpoint_loaded"] is True
    assert resumed["evidence"]["resumed_step_ids"] == ["upper"]
    assert resumed["evidence"]["checkpoint_cleared"] is True
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.asyncio
async def test_structured_repair_changes_strategy_before_outcome():
    calls = []

    async def dispatch(name, arguments):
        calls.append((name, arguments))
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            if arguments["module_id"] == "string.primary":
                return {
                    "ok": False,
                    "error": "unsupported strategy",
                    "repair": {
                        "module_id": "string.fallback",
                        "params": {"text": "fixed"},
                        "reason": "switch to compatible module",
                    },
                }
            return {"ok": True, "data": {"result": "FIXED"}}
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments)
        raise AssertionError(name)

    result = await execute_blueprint_loop(
        "bp_repair",
        [{
            "id": "repairable",
            "module": "string.primary",
            "params": {"text": "broken"},
        }],
        dispatch,
    )

    assert result["closed_loop_ok"] is True
    assert result["evidence"]["repair_count"] == 1
    assert result["evidence"]["repairs"][0]["to_module"] == "string.fallback"
    assert result["executions"][0]["source_module_id"] == "string.primary"
    assert result["executions"][0]["module_id"] == "string.fallback"
    assert result["evidence"]["total_attempts"] == 2


@pytest.mark.asyncio
async def test_malformed_repair_contract_is_bounded_failure():
    async def dispatch(name, arguments):
        if name == "validate_params":
            return {"valid": True}
        if name == "execute_module":
            return {
                "ok": False,
                "repair": {
                    "module_id": "string.fallback",
                    "params": {"text": "fixed"},
                    "retry": {"count": 999},
                },
            }
        if name == "report_blueprint_outcome":
            return _bound_outcome(arguments)
        raise AssertionError(name)

    result = await execute_blueprint_loop(
        "bp_invalid_repair",
        [{
            "id": "repairable",
            "module": "string.primary",
            "params": {"text": "broken"},
        }],
        dispatch,
    )

    assert result["ok"] is False
    assert result["evidence"]["repairs"][0]["phase"] == "contract"
    assert "Invalid repair contract" in result["results"][0]["result"]["error"]


def test_capability_router_escalates_high_risk_and_uses_deterministic_first():
    candidates = [
        ModelCandidate.from_name("ollama", "qwen2.5:7b", 0),
        ModelCandidate.from_name("openai", "gpt-5-codex", 2),
    ]
    router = CapabilityModelRouter()

    deterministic = router.route(
        "repeat the verified workflow",
        candidates,
        deterministic_available=True,
    )
    escalated = router.route(
        "implement security migration and refactor production",
        candidates,
    )

    assert deterministic.mode == "deterministic"
    assert escalated.candidate_label == "openai:gpt-5-codex"
    assert escalated.required_tier == 3
    assert escalated.degraded is False


def test_provider_chain_applies_capability_route_without_breaking_failover():
    chain = ProviderChain(
        object(),
        [object()],
        ["ollama:qwen2.5:7b", "openai:gpt-5-codex"],
    )

    assert chain.prefer_provider("openai:gpt-5-codex") is True
    assert chain.active_provider_name == "openai:gpt-5-codex"
    assert chain._build_try_order() == [1, 0]
    assert chain.prefer_provider("missing:model") is False
    assert chain.active_provider_name == "openai:gpt-5-codex"


def test_agent_capability_route_activates_selected_provider():
    agent = Agent.__new__(Agent)
    agent._config = AgentConfig(enable_model_routing=True)
    agent._provider = ProviderChain(
        object(),
        [object()],
        ["ollama:qwen2.5:7b", "openai:gpt-5-codex"],
    )
    agent._model_router = CapabilityModelRouter()
    agent._model_candidates = [
        ModelCandidate.from_name("ollama", "qwen2.5:7b", 0),
        ModelCandidate.from_name("openai", "gpt-5-codex", 2),
    ]
    agent._last_model_route = None

    route = agent._select_model_route(
        "implement a production security migration",
    )

    assert route.candidate_label == "openai:gpt-5-codex"
    assert route.degraded is False
    assert agent._provider.active_provider_name == "openai:gpt-5-codex"


@pytest.mark.asyncio
async def test_middleware_forwards_configured_distillation_threshold(monkeypatch):
    from flyto_ai.assistant import middleware as middleware_module

    captured = {}

    def fake_feedback(tool_calls, execution_results, user_message, *, min_steps):
        captured.update({
            "tool_calls": tool_calls,
            "execution_results": execution_results,
            "user_message": user_message,
            "min_steps": min_steps,
        })

    monkeypatch.setattr(middleware_module.router, "feedback", fake_feedback)
    monkeypatch.setattr(middleware_module.router, "init_storage", lambda: None)
    assistant = middleware_module.AssistantMiddleware(
        distillation_min_steps=7,
    )

    await assistant.post_process(
        [],
        [{"ok": True, "module_id": "string.uppercase"}],
        "verified workflow",
    )

    assert captured["min_steps"] == 7
    assert captured["user_message"] == "verified workflow"


@pytest.mark.asyncio
async def test_cross_model_benchmark_reports_success_cost_and_side_effects():
    class FakeAgent:
        def __init__(self, ok):
            self.ok = ok

        async def chat(self, _message, mode):
            assert mode == "execute"
            return {
                "ok": self.ok,
                "rounds_used": 0 if self.ok else 2,
                "cost": {"session_total_usd": 0.01 if self.ok else 0.02},
                "execution_results": [{
                    "ok": self.ok,
                    "executed": True,
                    "attempt_count": 2,
                    "assertions": [{"ok": self.ok}],
                }],
            }

    report = await run_model_benchmark(
        {
            "strong": lambda: FakeAgent(True),
            "weak": lambda: FakeAgent(False),
        },
        [BenchmarkCase("closed-loop", "run proof")],
    )

    assert report["version"] == "flyto.benchmark.v1"
    assert report["models"]["strong"]["summary"]["success_rate"] == 1.0
    assert report["models"]["strong"]["summary"]["retries"] == 1
    assert report["models"]["strong"]["summary"]["side_effects"] == 1
    assert report["models"]["weak"]["summary"]["success_rate"] == 0.0


def test_distillation_requires_runtime_evidence_and_preserves_assertions():
    results = []
    for index, module_id in enumerate(
        ["string.uppercase", "string.reverse", "string.lowercase"],
        start=1,
    ):
        results.append({
            "module_id": module_id,
            "step_id": "s{}".format(index),
            "ok": True,
            "executed": True,
            "arguments": {"params": {"text": "Flyto"}},
            "validation": {"valid": True},
            "assertions": [{
                "path": "data.result",
                "op": "truthy",
                "ok": True,
                "expected": None,
            }],
        })

    decision = evaluate_distillation([], results, "verified text flow")
    rejected = evaluate_distillation(
        [],
        [{**item, "validation": {"valid": False}, "assertions": []} for item in results],
        "unverified flow",
    )

    assert decision.eligible is True
    assert decision.workflow["steps"][0]["assertions"][0]["op"] == "truthy"
    assert decision.workflow["distillation"]["evidence_count"] == 6
    assert rejected.eligible is False


class _CapabilityAwareEngine:
    """A Blueprint engine that accepts the host-derived module-id set."""

    def __init__(self):
        self.calls = []

    def list_blueprints(self, available_module_ids=None):
        self.calls.append(("list_blueprints", available_module_ids))
        return [{"id": "learned_copy"}]

    def search(self, query, available_module_ids=None):
        self.calls.append(("search", query, available_module_ids))
        return [{"id": "learned_copy"}]

    def expand(self, blueprint_id, args, available_module_ids=None):
        self.calls.append(("expand", blueprint_id, available_module_ids))
        return {
            "ok": True,
            "data": {
                "steps": [{"module": "string.uppercase", "params": args}],
            },
        }


class _LegacyEngine:
    """A Blueprint engine predating capability filtering."""

    def __init__(self):
        self.calls = []

    def list_blueprints(self):
        self.calls.append("list_blueprints")
        return []

    def search(self, _message):
        self.calls.append("search")
        return []

    def expand(self, blueprint_id, args):
        self.calls.append("expand")
        return {"ok": True, "data": {"steps": [{"module": "string.uppercase", "params": args}]}}


def _install_engine(monkeypatch, engine, module_ids):
    import flyto_blueprint

    monkeypatch.setattr(flyto_blueprint, "get_engine", lambda: engine)
    monkeypatch.setattr(
        blueprint_tools,
        "get_core_installed_module_ids",
        lambda: module_ids,
    )


@pytest.mark.asyncio
async def test_blueprint_discovery_uses_host_derived_module_ids(monkeypatch):
    engine = _CapabilityAwareEngine()
    installed = frozenset({"string.uppercase", "browser.launch"})
    _install_engine(monkeypatch, engine, installed)

    await dispatch_blueprint_tool("list_blueprints", {})
    await dispatch_blueprint_tool("list_blueprints", {"query": "copy text"})
    await dispatch_blueprint_tool(
        "use_blueprint",
        {"blueprint_id": "learned_copy", "args": {"text": "Flyto"}},
    )

    assert engine.calls == [
        ("list_blueprints", installed),
        ("search", "copy text", installed),
        ("expand", "learned_copy", installed),
    ]


@pytest.mark.asyncio
async def test_model_supplied_availability_never_reaches_the_engine(monkeypatch):
    engine = _CapabilityAwareEngine()
    installed = frozenset({"string.uppercase"})
    _install_engine(monkeypatch, engine, installed)

    await dispatch_blueprint_tool(
        "list_blueprints",
        {
            "query": "copy text",
            "available_module_ids": ["shell.run", "docker.run"],
            "available_capabilities": ["shell.run", "docker.run"],
            "capabilities": ["ssh.connect"],
        },
    )
    await dispatch_blueprint_tool(
        "use_blueprint",
        {
            "blueprint_id": "learned_copy",
            "args": {"text": "Flyto"},
            "installed_capabilities": ["shell.run"],
            "module_ids": ["shell.run"],
        },
    )

    assert engine.calls == [
        ("search", "copy text", installed),
        ("expand", "learned_copy", installed),
    ]


@pytest.mark.asyncio
async def test_absent_or_old_core_leaves_blueprint_filtering_untouched(monkeypatch):
    engine = _CapabilityAwareEngine()
    _install_engine(monkeypatch, engine, None)

    await dispatch_blueprint_tool("list_blueprints", {})
    await dispatch_blueprint_tool(
        "use_blueprint",
        {"blueprint_id": "learned_copy", "args": {}},
    )

    assert engine.calls == [
        ("list_blueprints", None),
        ("expand", "learned_copy", None),
    ]


@pytest.mark.asyncio
async def test_empty_capability_set_is_forwarded_not_dropped(monkeypatch):
    engine = _CapabilityAwareEngine()
    _install_engine(monkeypatch, engine, frozenset())

    await dispatch_blueprint_tool("list_blueprints", {})

    assert engine.calls == [("list_blueprints", frozenset())]


@pytest.mark.asyncio
async def test_legacy_engine_is_called_without_the_module_id_keyword(monkeypatch):
    engine = _LegacyEngine()
    _install_engine(monkeypatch, engine, frozenset({"string.uppercase"}))

    await dispatch_blueprint_tool("list_blueprints", {})
    await dispatch_blueprint_tool("list_blueprints", {"query": "copy"})
    result = await dispatch_blueprint_tool(
        "use_blueprint",
        {"blueprint_id": "learned_copy", "args": {"text": "Flyto"}},
    )

    assert engine.calls == ["list_blueprints", "search", "expand"]
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_module_discovery_failure_narrows_instead_of_widening(monkeypatch):
    engine = _CapabilityAwareEngine()
    import flyto_blueprint

    def boom():
        raise RuntimeError("core bridge exploded")

    monkeypatch.setattr(flyto_blueprint, "get_engine", lambda: engine)
    monkeypatch.setattr(blueprint_tools, "get_core_installed_module_ids", boom)

    await dispatch_blueprint_tool("list_blueprints", {})

    assert engine.calls == [("list_blueprints", frozenset())]


def test_blueprint_tool_schema_hides_every_availability_argument(monkeypatch):
    tool_defs = [{
        "name": "list_blueprints",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "available_module_ids": {"type": "array"},
                "available_capabilities": {"type": "array"},
            },
            "required": ["query", "available_module_ids", "available_capabilities"],
        },
    }]
    import flyto_blueprint.tools as blueprint_tool_defs

    monkeypatch.setattr(
        blueprint_tool_defs, "get_blueprint_tools", lambda: tool_defs,
    )

    published = blueprint_tools.get_blueprint_tool_defs()

    properties = published[0]["inputSchema"]["properties"]
    assert "available_module_ids" not in properties
    assert "available_capabilities" not in properties
    assert published[0]["inputSchema"]["required"] == ["query"]
    # The upstream definition is not mutated in place.
    assert "available_module_ids" in tool_defs[0]["inputSchema"]["properties"]
