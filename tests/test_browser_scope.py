"""Concurrent browser ownership and terminal cleanup through the Core adapter."""

import asyncio

import pytest

from flyto_ai.tools import core_tools
from flyto_ai.tools.browser_scope import (
    BrowserCleanupError,
    browser_session_scope,
    current_browser_scope,
)


@pytest.fixture
def browser_runtime(monkeypatch):
    calls = []
    sequence = []

    async def execute(module_id, params, context, browser_sessions):
        owner = current_browser_scope().owner_id if current_browser_scope() else "legacy"
        if module_id == "browser.launch":
            sid = owner + "-" + str(len(sequence))
            sequence.append(sid)
            browser_sessions[sid] = object()
            calls.append(("open", sid))
            return {"status": "success", "session_id": sid}
        sid = (context or {}).get("browser_session")
        if sid is None and len(browser_sessions) == 1:
            sid = next(iter(browser_sessions))
        if sid not in browser_sessions:
            return {"ok": False, "error": "Browser session is not owned by this caller"}
        if module_id == "browser.close":
            await asyncio.sleep(0)
            calls.append(("close", sid))
            browser_sessions.pop(sid)
        return {"status": "success", "session_id": sid}

    handler = {"execute_module": execute}
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: handler)
    monkeypatch.setattr(core_tools, "_browser_sessions", {"foreign": object()})
    monkeypatch.setattr(core_tools, "_browser_launch_failed", False)
    monkeypatch.setattr(core_tools, "_browser_launch_error", "")
    monkeypatch.setattr(core_tools, "_goto_consecutive_fails", 0)
    return calls, handler


async def module(name, **extra):
    return await core_tools.dispatch_core_tool("execute_module", {
        "module_id": name, "params": {}, **extra,
    })


@pytest.mark.asyncio
async def test_concurrent_turns_and_relaunch_keep_foreign_browsers(browser_runtime):
    calls, _ = browser_runtime

    async def turn(owner):
        async with browser_session_scope(owner) as scope:
            first = await module("browser.launch")
            await asyncio.sleep(0)
            denied = await module("browser.goto", context={"browser_session": "foreign"})
            assert denied["ok"] is False
            assert first["session_id"] in scope.sessions
            assert core_tools.get_browser_status()
            await module("browser.launch")
        assert scope.receipt()["status"] == "closed"
        assert set(scope.closed_session_ids) == scope.owned_session_ids
        assert len(scope.closed_session_ids) == len(scope.owned_session_ids)
        return scope

    scopes = await asyncio.gather(turn("one"), turn("two"))
    assert scopes[0].owned_session_ids.isdisjoint(scopes[1].owned_session_ids)
    assert set(core_tools._browser_sessions) == {"foreign"}
    assert ("close", "foreign") not in calls


@pytest.mark.asyncio
async def test_nested_scope_restores_outer_live_browser(browser_runtime):
    async with browser_session_scope("outer") as outer:
        first = await module("browser.launch")
        async with browser_session_scope("inner") as inner:
            await module("browser.launch")
        assert inner.closed
        assert current_browser_scope() is outer
        assert first["session_id"] in outer.sessions
    assert current_browser_scope() is None


@pytest.mark.asyncio
async def test_retry_breaker_does_not_poison_another_scope(browser_runtime):
    async with browser_session_scope("failed"):
        core_tools._set_browser_retry_state(True, "Launch failed", 0)
        async with browser_session_scope("healthy"):
            result = await module("browser.launch")
            assert result["status"] == "success"
        assert core_tools._browser_retry_state()[0] is True
    assert core_tools._browser_retry_state()[0] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["error", "cancel"])
async def test_error_and_cancellation_close_owned_sessions(browser_runtime, ending):
    scopes = []
    started = asyncio.Event()

    async def turn():
        async with browser_session_scope("interrupted") as scope:
            scopes.append(scope)
            await module("browser.launch")
            started.set()
            if ending == "error":
                raise ValueError("Task could not finish")
            await asyncio.Event().wait()

    task = asyncio.create_task(turn())
    await started.wait()
    if ending == "cancel":
        task.cancel()
    with pytest.raises(ValueError if ending == "error" else asyncio.CancelledError):
        await task
    assert scopes[0].closed
    assert set(scopes[0].closed_session_ids) == scopes[0].owned_session_ids
    assert scopes[0].sessions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["unacknowledged", "timeout"])
async def test_cleanup_failure_is_explicit_and_never_relaunches(browser_runtime, failure):
    calls, handler = browser_runtime
    normal = handler["execute_module"]

    async def execute(module_id, **kwargs):
        if module_id == "browser.close":
            if failure == "timeout":
                await asyncio.Event().wait()
            return {"ok": False, "status": "success", "error": "Close was not acknowledged"}
        return await normal(module_id=module_id, **kwargs)

    handler["execute_module"] = execute
    with pytest.raises(BrowserCleanupError):
        async with browser_session_scope("failed-close", cleanup_timeout=0.01) as scope:
            await module("browser.launch")
    assert scope.receipt()["status"] == "failed"
    assert scope.sessions
    assert not scope.closed_session_ids
    assert len([call for call in calls if call[0] == "open"]) == 1


@pytest.mark.asyncio
async def test_model_close_is_recorded_and_not_repeated_at_exit(browser_runtime):
    calls, _ = browser_runtime
    async with browser_session_scope("explicit") as scope:
        await module("browser.launch")
        await module("browser.close")
    assert set(scope.closed_session_ids) == scope.owned_session_ids
    assert len([call for call in calls if call[0] == "close"]) == 1


@pytest.mark.asyncio
async def test_closed_scope_cannot_spawn_late_background_browsers(browser_runtime):
    resume = asyncio.Event()

    async def late():
        await resume.wait()
        return await module("browser.launch")

    async with browser_session_scope("finished"):
        task = asyncio.create_task(late())
    resume.set()
    result = await task
    assert result["ok"] is False
    assert "scope is closed" in result["error"]
