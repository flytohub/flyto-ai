"""Typed host goals retain action authority without reclassifying conversation."""

import asyncio
import hashlib
import json

import pytest

from flyto_ai import AgentConfig
from flyto_ai.intelligence.planner import classify_tool_intent
from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
from test_execution_continuation import make_agent

GOAL = "查看行事曆"


@pytest.mark.asyncio
async def test_host_start_and_continuation_preserve_dispatch_wrapper_behavior():
    agent, _, calls = make_agent()
    wrapped_calls = []

    def wrap(dispatch):
        async def wrapped(name, args):
            wrapped_calls.append((name, args))
            return await dispatch(name, args)
        return wrapped

    try:
        await agent.start_execution(GOAL, dispatch_wrapper=wrap)
        await agent.continue_execution("Observe remaining evidence.", goal=GOAL, dispatch_wrapper=wrap)
        assert len(calls) == 2
        assert wrapped_calls == calls
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_typed_goal_can_start_and_continue_on_exact_agent_and_goal():
    agent, provider, calls = make_agent()
    other, other_provider, _ = make_agent()
    try:
        assert not classify_tool_intent(GOAL).tool_eligible
        await agent.start_execution(GOAL)
        assert agent._execution_admission.goal_sha256 == hashlib.sha256(GOAL.encode()).hexdigest()
        await agent.continue_execution("Observe remaining evidence.", goal=GOAL)
        assert len(calls) == 2
        assert all("execute_module" in row for row in provider.visible)
        with pytest.raises(PermissionError, match="matching action admission"):
            await agent.continue_execution("Continue", goal=GOAL + " changed")
        with pytest.raises(PermissionError, match="matching action admission"):
            await other.continue_execution("Continue", goal=GOAL)
        assert not other_provider.visible
    finally:
        await agent.close()
        await other.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    GOAL, "請說明如何查看行事曆", "不要查看行事曆", "『查看行事曆』這句話是什麼意思？",
])
async def test_normal_chat_cannot_mint_typed_host_admission(message):
    agent, _, calls = make_agent()
    try:
        await agent.chat(message)
        with pytest.raises(PermissionError, match="matching action admission"):
            await agent.continue_execution("Continue", goal=message)
        assert not calls
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_task_authority_keeps_permission_ceiling_and_rejects_policy_change():
    agent, provider, calls = make_agent()
    agent._permission_enforcer = PermissionEnforcer(PermissionLevel.READ_ONLY)
    try:
        await agent.start_execution(GOAL)
        await agent.continue_execution("Continue", goal=GOAL)
        assert not calls and all("execute_module" not in row for row in provider.visible)
        agent._permission_enforcer = PermissionEnforcer(PermissionLevel.DANGER_FULL)
        with pytest.raises(PermissionError, match="policy changed"):
            await agent.continue_execution("Continue", goal=GOAL)
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_host_admission_cannot_override_a_tool_policy_denial():
    agent, _, calls = make_agent()
    agent._policies = {"allowed_tools": ["get_module_info"]}
    try:
        await agent.start_execution(GOAL)
        await agent.continue_execution("Continue", goal=GOAL)
        assert not calls
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_chat_preparation_already_in_flight_cannot_be_replaced_by_host_start():
    agent, _, calls = make_agent()
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare():
        entered.set()
        await release.wait()

    agent._init_memory = prepare
    pending = asyncio.create_task(agent.chat("Explain the calendar"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        with pytest.raises(RuntimeError):
            await agent.start_execution(GOAL)
        release.set()
        await pending
        assert not calls and agent._execution_admission is None
    finally:
        release.set()
        await agent.close()


@pytest.mark.asyncio
async def test_failed_start_discards_prior_admission_and_releases_its_lock():
    agent, _, _ = make_agent()
    original_init = agent._init_memory

    async def unavailable():
        raise ValueError("Preparation failed")

    try:
        await agent.start_execution(GOAL)
        agent._init_memory = unavailable
        with pytest.raises(ValueError, match="Preparation failed"):
            await agent.start_execution("Another assigned goal")
        assert agent._execution_admission is None
        with pytest.raises(PermissionError):
            await agent.continue_execution("Continue", goal=GOAL)
        agent._init_memory = original_init
        await agent.start_execution(GOAL)
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_inflight_start_is_exclusive_and_cancellation_releases_its_scope():
    agent, provider, _ = make_agent()
    provider.started, provider.release = asyncio.Event(), asyncio.Event()
    pending = asyncio.create_task(agent.start_execution(GOAL))
    try:
        await asyncio.wait_for(provider.started.wait(), 2)
        for invoke in (
            lambda: agent.start_execution(GOAL),
            lambda: agent.chat("Open another page"),
            lambda: agent.continue_execution("Continue", goal=GOAL),
        ):
            with pytest.raises(RuntimeError):
                await invoke()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert agent._execution_admission is None
        provider.started = None
        await agent.chat("Explain the calendar")
        with pytest.raises(PermissionError):
            await agent.continue_execution("Continue", goal=GOAL)
    finally:
        if not pending.done():
            pending.cancel()
        await agent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["policy", "background", "nested", "nested_same"])
async def test_preparation_cannot_change_policy_or_reuse_host_context(change):
    agent, provider, calls = make_agent()
    original_init = agent._init_memory
    entered = False

    async def prepare():
        nonlocal entered
        if entered:
            return await original_init()
        entered = True
        if change == "policy":
            agent._policies = {"blocked_tools": ["execute_module"]}
        elif change == "background":
            with pytest.raises((PermissionError, RuntimeError)):
                await asyncio.create_task(agent.chat(GOAL))
        else:
            with pytest.raises((PermissionError, RuntimeError)):
                await agent.chat(GOAL if change == "nested_same" else "Open another page")

    agent._init_memory = prepare
    try:
        if change == "policy":
            with pytest.raises(PermissionError, match="policy changed"):
                await agent.start_execution(GOAL)
            assert not provider.visible and not calls
        else:
            await agent.start_execution(GOAL)
            assert len(provider.visible) == len(calls) == 1
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_host_start_options_cannot_be_supplied_as_chat_flags_or_change_mode():
    agent, provider, calls = make_agent()
    try:
        with pytest.raises(TypeError):
            await agent.chat(GOAL, host_admitted=True)
        with pytest.raises(TypeError):
            await agent.start_execution(GOAL, mode="forge")
        assert not provider.visible and not calls
        assert all(tool.get("name") != "start_execution" for tool in agent._tools)
    finally:
        await agent.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["claude_cli", "local_ai"])
@pytest.mark.parametrize("typed", [False, True])
async def test_external_transport_is_not_reset_by_rejected_interleaving(source, typed):
    from unittest.mock import Mock
    from flyto_ai.cli_runtime import CliAgent, CliRuntimeConfig
    from flyto_ai.local_runtime import LocalModelAgent

    entered, release = asyncio.Event(), asyncio.Event()

    async def completion(**_request):
        entered.set()
        await release.wait()
        return '{"content":"Observed response.","tool_calls":[]}'

    config = AgentConfig(provider=source, model="", enable_transcript=False)
    options = {"config": config, "completion_fn": completion, "tools": []}
    agent = (CliAgent(cli=CliRuntimeConfig(source=source), **options) if source == "claude_cli"
             else LocalModelAgent(**options))
    reset = Mock(wraps=agent.cli_runtime.reset)
    agent.cli_runtime.reset = reset
    pending = asyncio.create_task(agent.start_execution(GOAL) if typed else agent.chat("Explain the calendar"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        for invoke in (lambda: agent.start_execution(GOAL), lambda: agent.chat(GOAL),
                       lambda: agent.continue_execution("Continue", goal=GOAL)):
            with pytest.raises(RuntimeError):
                await invoke()
        assert reset.call_count == 1 and agent.cli_runtime.continuation is False
        release.set()
        await pending
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await agent.close()


@pytest.mark.asyncio
async def test_typed_chinese_goal_reads_real_core_file_without_widening_permissions(tmp_path, monkeypatch):
    from flyto_ai.tools.core_tools import dispatch_core_tool

    monkeypatch.chdir(tmp_path)
    content = "Calendar observation: meeting at 10:30.\n"
    (tmp_path / "calendar.txt").write_text(content)
    agent, _, _ = make_agent()
    observed = []

    class CoreProvider:
        async def chat(self, messages, system_prompt, tools, dispatch_fn, max_rounds=30, **kwargs):
            result = await dispatch_fn("execute_module", {
                "module_id": "file.read", "params": {"path": "calendar.txt"},
            })
            observed.append(result)
            return "Observed file.", [], 1, {}

    agent._provider = CoreProvider()
    agent._dispatch_fn = dispatch_core_tool
    try:
        await agent.start_execution("查看行事曆檔案的原始內容")
        assert observed[0].get("ok") is True
        assert content in json.dumps(observed[0], ensure_ascii=False).replace("\\n", "\n")
        assert (tmp_path / "calendar.txt").read_text() == content
        assert agent._permission_enforcer.level == PermissionLevel.WORKSPACE_WRITE
    finally:
        await agent.close()
