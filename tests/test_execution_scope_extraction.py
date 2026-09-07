"""Existing Core facade overrides remain authoritative after helper extraction."""

from unittest.mock import AsyncMock

import pytest

from flyto_ai.tools import core_tools


def test_public_clear_and_status_use_the_current_facade_registry(monkeypatch):
    legacy = {"foreign": object()}
    selected = {"owned": object()}
    resets = []
    monkeypatch.setattr(core_tools, "_browser_sessions", legacy)
    monkeypatch.setattr(core_tools, "_active_browser_sessions", lambda: selected)
    monkeypatch.setattr(core_tools, "_set_browser_retry_state", lambda *state: resets.append(state))

    assert "exactly one browser" in core_tools.get_browser_status()
    core_tools.clear_browser_sessions()
    assert selected == {}
    assert "foreign" in legacy
    assert resets == [(False, "", 0)]
    assert core_tools.get_browser_status() == ""


def test_unscoped_retry_state_uses_replaceable_core_module_globals(monkeypatch):
    monkeypatch.setattr(core_tools, "_browser_launch_failed", True)
    monkeypatch.setattr(core_tools, "_browser_launch_error", "fixture-failure")
    monkeypatch.setattr(core_tools, "_goto_consecutive_fails", 2)
    assert core_tools._browser_retry_state() == (True, "fixture-failure", 2)
    core_tools._set_browser_retry_state(False, "", 1)
    assert (core_tools._browser_launch_failed, core_tools._browser_launch_error,
            core_tools._goto_consecutive_fails) == (False, "", 1)


@pytest.mark.asyncio
async def test_dispatch_still_obeys_the_core_facade_validation_override(monkeypatch):
    execute = AsyncMock()
    handler = {"execute_module": execute}
    refusal = {"ok": False, "params_valid": False, "error": "fixture-validation-refusal"}
    seen = []

    def validate(active_handler, module_id, params):
        seen.append((active_handler, module_id, params))
        return refusal

    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: handler)
    monkeypatch.setattr(core_tools, "_validate_execute_module_args", validate)
    result = await core_tools.dispatch_core_tool("execute_module", {
        "module_id": "string.upper", "params": {"text": "fixture"},
    })
    assert result == refusal
    assert seen == [(handler, "string.upper", {"text": "fixture"})]
    execute.assert_not_awaited()
