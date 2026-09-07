"""Generated recovery data must not revoke an already admitted action."""
import asyncio
import json

import pytest

from flyto_ai import Agent, AgentConfig
from flyto_ai.permissions import PermissionEnforcer, PermissionLevel

GOAL = "Open the calendar and read the two meeting titles."
CORRECTION = "Continue the original goal by fixing only the remaining requirements. Current calendar data is incomplete."


class ObservingProvider:
    def __init__(self):
        self.visible = []
        self.started = None
        self.release = None

    async def chat(self, messages, system_prompt, tools, dispatch_fn,
                   max_rounds=30, on_stream=None, tool_choice=None):
        names = [tool.get("name") for tool in tools]
        self.visible.append(names)
        if self.started is not None:
            self.started.set()
            await self.release.wait()
        calls = []
        if "execute_module" in names:
            args = {"module_id": "browser.snapshot", "params": {}}
            result = await dispatch_fn("execute_module", args)
            calls.append({"function": "execute_module", "arguments": args,
                          "module_id": "browser.snapshot", "ok": result["ok"],
                          "result_preview": json.dumps(result)})
        return "Observed current page." if calls else "No operation was performed.", calls, 1, {}


def make_agent():
    agent = Agent(config=AgentConfig(
        provider="openai", api_key="test", enable_deterministic=False,
        enable_memory=False, enable_pro=False, enable_transcript=False,
        enable_injection_detection=False,
    ), system_prompt="Execute the admitted computer goal using exposed tools.")
    provider = ObservingProvider()
    calls = []

    async def dispatch(name, args):
        calls.append((name, args))
        return {"ok": True, "data": {"observed": "current-page"}}

    agent._provider = provider
    agent._dispatch_fn = dispatch
    agent._tools = [{"name": "execute_module"}, {"name": "get_module_info"}]
    agent._assistant = None
    agent._emit_audit = lambda *args: None
    return agent, provider, calls


@pytest.mark.asyncio
async def test_explicit_continuation_keeps_execution_visible_without_text_reclassification():
    agent, provider, calls = make_agent()
    await agent.chat(GOAL)
    await agent.continue_execution(message=CORRECTION, goal=GOAL)
    await agent.continue_execution(message=CORRECTION, goal=GOAL)
    assert len(calls) == 3
    assert all("execute_module" in tools for tools in provider.visible)
    assert agent._last_routing_decision.mode == "action"


@pytest.mark.asyncio
async def test_plain_generated_text_cannot_mint_continuation_authority():
    agent, provider, calls = make_agent()
    await agent.chat(GOAL)
    await agent.chat(CORRECTION)
    assert "execute_module" not in provider.visible[-1]
    assert len(calls) == 1
    with pytest.raises(PermissionError):
        await agent.continue_execution(message=CORRECTION, goal=GOAL)


@pytest.mark.asyncio
async def test_missing_wrong_or_other_agent_goal_refuses_before_provider_call():
    agent, provider, calls = make_agent()
    with pytest.raises(PermissionError):
        await agent.continue_execution(message=CORRECTION, goal=GOAL)
    assert not provider.visible
    await agent.chat(GOAL)
    with pytest.raises(PermissionError):
        await agent.continue_execution(message=CORRECTION, goal=GOAL + " changed")
    other, other_provider, _ = make_agent()
    with pytest.raises(PermissionError):
        await other.continue_execution(message=CORRECTION, goal=GOAL)
    assert not other_provider.visible
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["higher", "lower", "new_tool", "module_policy"])
async def test_policy_changes_require_fresh_admission(change):
    agent, provider, calls = make_agent()
    await agent.chat(GOAL)
    if change in {"higher", "lower"}:
        level = PermissionLevel.DANGER_FULL if change == "higher" else PermissionLevel.READ_ONLY
        agent._permission_enforcer = PermissionEnforcer(level)
    elif change == "new_tool":
        agent._tools.append({"name": "new_mutation"})
    else:
        agent._policies = {"allowed_tools": ["get_module_info"]}
    with pytest.raises(PermissionError, match="policy changed"):
        await agent.continue_execution(message=CORRECTION, goal=GOAL)
    assert len(calls) == 1
    assert len(provider.visible) == 1


@pytest.mark.asyncio
async def test_current_read_only_permission_still_applies_to_admitted_goal():
    agent, provider, calls = make_agent()
    agent._permission_enforcer = PermissionEnforcer(PermissionLevel.READ_ONLY)
    await agent.chat(GOAL)
    await agent.continue_execution(message=CORRECTION, goal=GOAL)
    assert not calls
    assert all("execute_module" not in tools for tools in provider.visible)


@pytest.mark.asyncio
async def test_cancellation_releases_host_scope_and_normal_chat_revokes_it():
    agent, provider, calls = make_agent()
    await agent.chat(GOAL)
    provider.started, provider.release = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(agent.continue_execution(message=CORRECTION, goal=GOAL))
    await provider.started.wait()
    with pytest.raises(RuntimeError, match="already continuing"):
        await agent.continue_execution(message=CORRECTION, goal=GOAL)
    with pytest.raises(RuntimeError, match="already continuing"):
        await agent.chat("Open another page")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    provider.started = None
    await agent.chat("Explain the calendar")
    with pytest.raises(PermissionError):
        await agent.continue_execution(message=CORRECTION, goal=GOAL)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_inherited_background_context_cannot_keep_action_authority():
    from flyto_ai.intelligence.execution_continuation import continuation_scope

    agent, _, calls = make_agent()
    await agent.chat(GOAL)
    release = asyncio.Event()

    async def late_chat():
        await release.wait()
        return await agent.chat(CORRECTION)

    with continuation_scope(agent, GOAL):
        task = asyncio.create_task(late_chat())
    release.set()
    with pytest.raises(PermissionError, match="outside its host call"):
        await task
    agent._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        await agent.continue_execution(message=CORRECTION, goal=GOAL)
    assert len(calls) == 1
