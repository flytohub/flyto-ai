"""Host-owned continuation of an action already admitted on this Agent."""
import asyncio
import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass

from flyto_ai.intelligence.confirmation import (
    ToolIntentDecision,
    route_with_confirmation,
)

_ACTIVE = ContextVar("flyto_execution_continuation", default=None)


@dataclass(frozen=True)
class ExecutionAdmission:
    goal_sha256: str
    decision: ToolIntentDecision
    policy: object


def _policy(agent):
    enforcer = agent._permission_enforcer
    return (enforcer.level, enforcer._overrides, agent._policies, agent._tools)


def route_chat(agent, message, history, mode):
    """Classify ordinary input; only a validated host scope inherits admission."""
    active = _ACTIVE.get()
    if active is not None and active[0] is agent:
        admission = active[1]
        if active[2] is not asyncio.current_task() or not getattr(agent, "_continuation_active", False):
            raise PermissionError("Execution continuation is unavailable outside its host call")
        if admission is not getattr(agent, "_execution_admission", None):
            raise PermissionError("The execution admission is no longer current")
        if admission.policy != _policy(agent):
            raise PermissionError("Execution policy changed; a new action admission is required")
        return admission.decision
    if getattr(agent, "_continuation_active", False):
        raise RuntimeError("This Agent is already continuing an execution")
    decision = (
        route_with_confirmation(message, history) if mode == "execute"
        else ToolIntentDecision("action", 1.0, "explicit_non_execute_mode", (mode,))
    )
    agent._execution_admission = (
        ExecutionAdmission(
            hashlib.sha256(message.encode()).hexdigest(), decision, deepcopy(_policy(agent)),
        ) if mode == "execute" and decision.tool_eligible else None
    )
    return decision


@contextmanager
def continuation_scope(agent, goal):
    """Grant one host call the existing goal admission, never a text-based bypass."""
    admission = getattr(agent, "_execution_admission", None)
    if getattr(agent, "_closed", False):
        raise RuntimeError("agent is closed")
    if (
        not isinstance(admission, ExecutionAdmission) or not isinstance(goal, str)
        or admission.goal_sha256 != hashlib.sha256(goal.encode()).hexdigest()
    ):
        raise PermissionError("No matching action admission exists on this Agent")
    if admission.policy != _policy(agent):
        raise PermissionError("Execution policy changed; a new action admission is required")
    if getattr(agent, "_continuation_active", False):
        raise RuntimeError("This Agent is already continuing an execution")
    agent._continuation_active = True
    token = _ACTIVE.set((agent, admission, asyncio.current_task()))
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        agent._continuation_active = False
