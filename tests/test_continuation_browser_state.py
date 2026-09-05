"""An admitted correction reuses prepared middleware and its owned browser."""
import json
from unittest.mock import AsyncMock

import pytest

from flyto_ai import Agent, AgentConfig
from flyto_ai.assistant import router
from flyto_ai.assistant.middleware import AssistantMiddleware
from flyto_ai.tools import core_tools
from flyto_ai.tools.browser_scope import browser_session_scope, current_browser_scope

GOAL = "Open the calendar and read its exact meeting titles."


class ObservingProvider:
    def __init__(self):
        self.prompts = []
        self.observed_sessions = []

    async def chat(self, messages, system_prompt, tools, dispatch_fn,
                   max_rounds=30, on_stream=None, tool_choice=None):
        turn = len(self.prompts)
        self.prompts.append(system_prompt)
        actions = ["browser.snapshot"] if turn == 1 else ["browser.launch", "browser.snapshot"]
        logs = []
        for module in actions:
            args = {"module_id": module, "params": {}}
            result = await dispatch_fn("execute_module", args)
            self.observed_sessions.append(result.get("session_id"))
            logs.append({"function": "execute_module", "module_id": module,
                         "arguments": args, "ok": result.get("status") == "success",
                         "result_preview": json.dumps(result)})
        return "Current page observed.", logs, 1, {}


def make_agent():
    agent = Agent(config=AgentConfig(
        provider="openai", api_key="test", enable_deterministic=False,
        enable_memory=False, enable_pro=False, enable_transcript=False,
        enable_injection_detection=False,
    ), system_prompt="Host-owned computer goal prompt.")
    agent._provider = ObservingProvider()
    agent._tools = [{"name": "execute_module"}]
    agent._dispatch_fn = core_tools.dispatch_core_tool
    agent._emit_audit = lambda *args: None
    middleware = AssistantMiddleware()
    middleware._apply_blueprint_guard = AsyncMock(return_value=None)
    middleware.post_process = AsyncMock(return_value=None)
    agent._assistant = middleware
    return agent


@pytest.mark.asyncio
async def test_continuation_reuses_preparation_and_owned_browser_then_new_goal_resets(monkeypatch):
    operations = []

    async def execute(module_id, params, context, browser_sessions):
        scope = current_browser_scope()
        if module_id == "browser.launch":
            session = scope.owner_id + "-session"
            browser_sessions[session] = {"authenticated": True}
        else:
            session = (context or {}).get("browser_session") or next(iter(browser_sessions))
        operations.append((module_id, session))
        result = {"status": "success", "session_id": session,
                  "data": {"authenticated": browser_sessions[session]["authenticated"]}}
        if module_id == "browser.close":
            browser_sessions.pop(session)
        return result

    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: {
        "execute_module": execute, "validate_params": lambda **kwargs: {"valid": True},
    })
    monkeypatch.setattr(core_tools, "_browser_sessions", {"foreign": object()})
    agent = make_agent()
    async with browser_session_scope("first") as first:
        await agent.chat(GOAL)
        prepared = agent._execution_assisted_dispatch[3]
        await agent.continue_execution(message="Current calendar evidence is incomplete", goal=GOAL)
        assert agent._execution_assisted_dispatch[3] is prepared
        assert agent._assistant._apply_blueprint_guard.await_count == 1
        assert list(first.sessions) == ["first-session"]
        assert "BROWSER IS ALREADY RUNNING" in agent._provider.prompts[1]
        assert "Omit context.browser_session" in agent._provider.prompts[1]
        assert agent._provider.observed_sessions == ["first-session"] * 3
    assert first.closed_session_ids == ["first-session"]
    async with browser_session_scope("second"):
        await agent.chat(GOAL)
        assert agent._execution_assisted_dispatch[3] is not prepared
        assert agent._assistant._apply_blueprint_guard.await_count == 2
    assert operations.count(("browser.launch", "first-session")) == 1
    assert set(core_tools._browser_sessions) == {"foreign"}


@pytest.mark.asyncio
async def test_custom_prompt_never_borrows_other_scope_or_legacy_browser(monkeypatch):
    monkeypatch.setattr(core_tools, "_browser_sessions", {"foreign": object()})
    agent = make_agent()
    prompt, _ = await agent._build_prompt(GOAL, "execute", True, None, None)
    assert "BROWSER IS ALREADY RUNNING" not in prompt
    async with browser_session_scope("outer") as outer:
        outer.sessions["own"] = object()
        try:
            prompt, _ = await agent._build_prompt(GOAL, "execute", True, None, None)
            assert "BROWSER IS ALREADY RUNNING" in prompt
            async with browser_session_scope("inner"):
                prompt, _ = await agent._build_prompt(GOAL, "execute", True, None, None)
                assert "BROWSER IS ALREADY RUNNING" not in prompt
        finally:
            # This is a registry-only unit probe, with no Core process to close.
            outer.sessions.clear()


@pytest.mark.asyncio
async def test_blueprint_guidance_is_not_a_successful_requested_action():
    async def discover(name, arguments):
        assert name == "list_blueprints"
        return {"blueprints": [{
            "id": "verified-login", "score": 100, "use_count": 4,
            "trust_tier": "ci_verified", "args": {},
            "evidence_card": {"sample_count": 4, "success_count": 4, "success_rate": 1.0},
        }]}

    redirected = await router.guard("execute_module", {"module_id": "browser.launch"}, GOAL, discover)
    assert redirected["_blueprint_redirect"] is True
    assert redirected["ok"] is False
    assert redirected["action_executed"] is False
    assert redirected["status"] == "guidance_required"
