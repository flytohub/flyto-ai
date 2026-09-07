"""Ad-hoc references fail before typing; explicit bindings use Core semantics."""

from unittest.mock import AsyncMock

import pytest

from flyto_ai.tools import core_tools
from flyto_ai.tools.browser_scope import browser_session_scope
from flyto_ai.tools.parameter_references import (
    UnresolvedParameterReference,
    resolve_module_params,
)


@pytest.mark.parametrize("value", ["${params.password}", "${{ params.password }}", "{{params.password}}"])
@pytest.mark.asyncio
async def test_unbound_password_never_reaches_core_and_can_be_corrected(monkeypatch, value):
    execute = AsyncMock(return_value={"status": "success"})
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: {"execute_module": execute})
    async with browser_session_scope("input-owner"):
        result = await core_tools.dispatch_core_tool("execute_module", {
            "module_id": "browser.type", "params": {"sensitive_text": value},
        })
        execute.assert_not_awaited()
        assert result["ok"] is False
        assert result["error_code"] == "unresolved_parameter_reference"
        assert result["retryable"] is True
        assert value not in result["error"]
        fixed = await core_tools.dispatch_core_tool("execute_module", {
            "module_id": "browser.type", "params": {"sensitive_text": "test-authorized-input"},
        })
    assert fixed["status"] == "success"
    execute.assert_awaited_once()
    assert execute.await_args.kwargs["params"] == {"sensitive_text": "test-authorized-input"}


@pytest.mark.parametrize("syntax", ["${params.password}", "${{params.password}}", "{{ params.password }}"])
def test_explicit_input_scope_resolves_through_real_core(syntax):
    original = {"sensitive_text": syntax, "literal": "unchanged"}
    context = {"params": {"password": "test-bound-input"}}
    assert resolve_module_params(original, context) == {
        "sensitive_text": "test-bound-input", "literal": "unchanged",
    }
    assert original["sensitive_text"] == syntax


def test_nested_step_output_keeps_core_types():
    params = {"body": {"records": "${read.data.records}", "url": "https://example.test/${read.data.key}"}}
    context = {"read": {"data": {"records": [{"count": 2}], "key": "observed"}}}
    resolved = resolve_module_params(params, context)
    assert resolved == {"body": {"records": [{"count": 2}], "url": "https://example.test/observed"}}


def test_model_cannot_resolve_ambient_service_secrets(monkeypatch):
    monkeypatch.setenv("SDK_BINDING_TEST_SECRET", "test-private-process-value")
    with pytest.raises(UnresolvedParameterReference):
        resolve_module_params({"sensitive_text": "${env.SDK_BINDING_TEST_SECRET}"},
                              {"env": {"SDK_BINDING_TEST_SECRET": "model-shaped-map"}})


def test_missing_nested_binding_is_refused():
    with pytest.raises(UnresolvedParameterReference):
        resolve_module_params({"sensitive_text": "{{params.missing}}"}, {"params": {}})


def test_runtime_handles_are_not_variable_data_or_stringified():
    class Handle:
        def __str__(self):
            raise AssertionError("Runtime handle was inspected")

    handle = Handle()
    with pytest.raises(UnresolvedParameterReference):
        resolve_module_params({"text": "${browser}"}, {"browser": handle})
    assert resolve_module_params({"browser": handle, "text": "literal"}, None)["browser"] is handle
    assert resolve_module_params({"text": "${rows.items[1]}"},
                                 {"rows": {"items": [handle, "second"]}}) == {"text": "second"}


@pytest.mark.asyncio
async def test_bad_launch_parameters_do_not_close_an_existing_browser(monkeypatch):
    execute = AsyncMock()
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: {"execute_module": execute})
    async with browser_session_scope("preserved") as scope:
        scope.sessions["already-active"] = object()
        result = await core_tools.dispatch_core_tool("execute_module", {
            "module_id": "browser.launch", "params": {"channel": "${params.channel}"},
        })
        execute.assert_not_awaited()
        assert result["ok"] is False
        assert "already-active" in scope.sessions
        # This test owns no real browser; leave a successful close receipt to
        # the adapter fixture instead of claiming a physical resource existed.
        execute.return_value = {"status": "success"}
    execute.assert_awaited_once()
    assert execute.await_args.kwargs["module_id"] == "browser.close"
