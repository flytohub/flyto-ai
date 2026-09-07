"""Actual Core schemas remain usable after repeated invalid actor arguments."""

from unittest.mock import AsyncMock

import pytest

from flyto_ai.assistant.middleware import AssistantMiddleware, _result_ok
from flyto_ai.assistant.resilience import SnapshotGuard
from flyto_ai.assistant.safety import CircuitBreaker
from flyto_ai.tools import core_tools


@pytest.mark.asyncio
async def test_three_invalid_calls_return_core_contract_then_corrected_call_executes(monkeypatch):
    from core.mcp_handler import get_module_info, validate_params

    execute = AsyncMock(return_value={"status": "success", "message": "Typed"})
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: {
        "execute_module": execute, "validate_params": validate_params, "get_module_info": get_module_info,
    })
    middleware = AssistantMiddleware()
    middleware._apply_blueprint_guard = AsyncMock(return_value=None)
    middleware._apply_snapshot_guard = AsyncMock(return_value=None)
    dispatch = middleware.wrap(core_tools.dispatch_core_tool)
    bad = {"module_id": "browser.type", "params": {
        "type_method": "id", "target": "", "text": "test-input",
    }}
    for _ in range(3):
        result = await dispatch("execute_module", bad)
        assert result["params_valid"] is False
        assert result["params_schema"] == get_module_info(module_id="browser.type")["params_schema"]
        assert "type_method" in result["params_schema"]
    blocked = await dispatch("execute_module", bad)
    assert blocked["ok"] is False
    assert "corrected call" in blocked["error"]
    execute.assert_not_awaited()
    good = {"module_id": "browser.type", "params": {
        "type_method": "id", "target": "username", "text": "test-input",
    }}
    # Core success without an ok field or data field must not trip either counter.
    for _ in range(5):
        result = await dispatch("execute_module", good)
        assert result["status"] == "success"
    assert execute.await_count == 5


def test_executed_failures_and_explicit_empty_observations_still_trip():
    failures = CircuitBreaker()
    for _ in range(3):
        failures.record_result("browser.click", False, {"ok": False, "status": "success"}, {"target": "x"})
    assert failures.is_tripped("browser.click", {"target": "different"})
    assert _result_ok({"ok": False, "status": "success"}) is False
    empty = CircuitBreaker()
    for _ in range(3):
        empty.record_result("browser.extract", True, {"status": "success", "data": []})
    assert empty.is_tripped("browser.extract")


@pytest.mark.asyncio
async def test_status_success_snapshot_is_observed_before_interaction():
    middleware = AssistantMiddleware()
    guard = SnapshotGuard()
    dispatch = AsyncMock(return_value={"status": "success", "data": {"text": "Observed page"}})
    args = {"module_id": "browser.click", "params": {"click_method": "id", "target": "submit"}}
    result = await middleware._apply_snapshot_guard("execute_module", args, guard, dispatch)
    assert result["_auto_snapshot"] is True
    assert result["ok"] is False
    assert result["action_executed"] is False
    assert not guard.needs_snapshot("execute_module", args)
    dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_click_exposes_its_own_canonical_method_schema(monkeypatch):
    from core.mcp_handler import get_module_info, validate_params

    execute = AsyncMock()
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: {
        "execute_module": execute, "validate_params": validate_params, "get_module_info": get_module_info,
    })
    result = await core_tools.dispatch_core_tool("execute_module", {
        "module_id": "browser.click", "params": {"click_method": "id", "target": ""},
    })
    assert result["params_valid"] is False
    assert "click_method" in result["params_schema"]
    assert "target" in result["params_schema"]
    execute.assert_not_awaited()
