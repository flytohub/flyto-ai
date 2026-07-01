# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for sub-agent orchestration system."""
import pytest

from flyto_ai.orchestration.policies import OrchestrationPolicy
from flyto_ai.orchestration.sub_agent import SubAgent, SubAgentStatus
from flyto_ai.orchestration.orchestrator import AgentOrchestrator


# --- Policy tests ---

def test_policy_defaults():
    p = OrchestrationPolicy()
    assert p.max_depth == 3
    assert p.default_timeout == 300
    assert p.max_concurrent == 5
    assert p.cascade_kill is True


def test_policy_depth_tools():
    p = OrchestrationPolicy()
    # Root (depth 0): all tools
    assert p.allowed_tools_at_depth(0) is None
    # Depth 1: restricted
    tools_1 = p.allowed_tools_at_depth(1)
    assert "execute_module" in tools_1
    assert "inspect_page" in tools_1
    # Depth 2: more restricted
    tools_2 = p.allowed_tools_at_depth(2)
    assert "execute_module" in tools_2
    assert "inspect_page" not in tools_2
    # Depth 3: most restricted
    tools_3 = p.allowed_tools_at_depth(3)
    assert len(tools_3) < len(tools_2)


def test_policy_can_spawn():
    p = OrchestrationPolicy(max_depth=3)
    assert p.can_spawn_at_depth(0) is True
    assert p.can_spawn_at_depth(1) is True
    assert p.can_spawn_at_depth(2) is True
    assert p.can_spawn_at_depth(3) is False
    assert p.can_spawn_at_depth(4) is False


def test_policy_max_rounds():
    p = OrchestrationPolicy(max_tool_rounds_per_depth=10)
    assert p.max_rounds_at_depth(0) == 30
    assert p.max_rounds_at_depth(1) == 10
    assert p.max_rounds_at_depth(2) == 8
    assert p.max_rounds_at_depth(3) >= 5


# --- SubAgent tests ---

def test_sub_agent_init():
    agent = SubAgent(
        task="test task",
        parent_session_id="parent-123",
        depth=1,
        timeout=60,
    )
    assert agent.task == "test task"
    assert agent.depth == 1
    assert agent.status == SubAgentStatus.PENDING
    assert len(agent.run_id) == 12


def test_sub_agent_cancel():
    agent = SubAgent(task="test", parent_session_id="p1")
    agent.cancel()
    assert agent.status == SubAgentStatus.CANCELLED


# --- Orchestrator tests ---

def test_orchestrator_init():
    orch = AgentOrchestrator(parent_session_id="test-session")
    assert orch.active_count == 0
    assert len(orch.all_agents) == 0


@pytest.mark.asyncio
async def test_orchestrator_max_depth_reject():
    policy = OrchestrationPolicy(max_depth=2)
    orch = AgentOrchestrator(parent_session_id="test", policy=policy)

    with pytest.raises(RuntimeError, match="max depth"):
        await orch.spawn("task", depth=2)  # depth >= max_depth


@pytest.mark.asyncio
async def test_orchestrator_max_concurrent_reject():
    policy = OrchestrationPolicy(max_concurrent=1)
    orch = AgentOrchestrator(parent_session_id="test", policy=policy)

    # Mock: manually add a running agent
    from flyto_ai.orchestration.sub_agent import SubAgent
    fake = SubAgent(task="running", parent_session_id="test")
    fake.status = SubAgentStatus.RUNNING
    orch._agents[fake.run_id] = fake

    with pytest.raises(RuntimeError, match="concurrent"):
        await orch.spawn("second task", depth=1)


def test_orchestrator_summary():
    orch = AgentOrchestrator(parent_session_id="test")
    summary = orch.summary()
    assert summary["total"] == 0
    assert summary["active"] == 0


def test_orchestrator_cancel_all():
    orch = AgentOrchestrator(parent_session_id="test")
    count = orch.cancel_all()
    assert count == 0
