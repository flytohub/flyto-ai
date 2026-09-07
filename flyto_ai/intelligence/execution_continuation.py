"""Host-owned continuation of an action already admitted on this Agent."""
import asyncio
import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List

from flyto_ai.permissions import PermissionLevel

from flyto_ai.intelligence.confirmation import (
    ToolIntentDecision,
    route_with_confirmation,
)

_ACTIVE = ContextVar("flyto_execution_continuation", default=None)
_STARTING = ContextVar("flyto_execution_start", default=None)


@dataclass(frozen=True)
class ExecutionAdmission:
    goal_sha256: str
    decision: ToolIntentDecision
    policy: object


@dataclass
class _HostStart:
    agent: object
    task: object
    goal_sha256: str
    policy: object
    claimed: bool = False
    consumed: bool = False


def _policy(agent):
    enforcer = agent._permission_enforcer
    return (
        enforcer.level, enforcer._overrides, enforcer.workspace_root,
        agent._policies, agent._tools,
    )


def check_chat_available(agent):
    """Reject interleaving before a facade resets transport or awaits preparation."""
    if getattr(agent, "_chat_active_task", None) is not None:
        if getattr(agent, "_continuation_active", False):
            raise RuntimeError("This Agent is already continuing an execution")
        raise RuntimeError("This Agent already has an active chat turn")
    if getattr(agent, "_execution_start_active", False):
        start = _STARTING.get()
        if start is None or start.agent is not agent or start.task is not asyncio.current_task() or start.claimed:
            raise RuntimeError("This Agent is already starting an execution")
    if getattr(agent, "_continuation_active", False):
        active = _ACTIVE.get()
        if active is None or active[0] is not agent or active[2] is not asyncio.current_task():
            raise RuntimeError("This Agent is already continuing an execution")


def guard_chat_turn(chat):
    """Give one native chat ownership before its first await, including subclasses."""
    @wraps(chat)
    async def guarded(agent, *args, **kwargs):
        check_chat_available(agent)
        start = _STARTING.get()
        if getattr(agent, "_execution_start_active", False):
            message = kwargs.get("message", args[0] if args else None)
            mode = kwargs.get("mode", args[3] if len(args) > 3 else "execute")
            if not isinstance(message, str) or hashlib.sha256(message.encode()).hexdigest() != start.goal_sha256 or mode != "execute":
                raise PermissionError("The host execution goal does not match its chat turn")
            start.claimed = True
        agent._chat_active_task = asyncio.current_task()
        try:
            return await chat(agent, *args, **kwargs)
        finally:
            agent._chat_active_task = None
    return guarded


async def start_execution(agent, goal, **chat_options):
    """Start a Python-host-authorized task without treating it as conversation.

    Only the host selects this method. No tool, HTTP flag or prompt text can
    enter its scope; it grants no tool permissions beyond the captured policy.
    """
    allowed = {"history", "template_context", "on_tool_call", "on_stream", "dispatch_wrapper"}
    if set(chat_options) - allowed:
        raise TypeError("Unsupported host execution chat option")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("A host execution requires a nonempty goal")
    if getattr(agent, "_closed", False):
        raise RuntimeError("agent is closed")
    if any(getattr(agent, name, False) for name in ("_chat_active_task", "_execution_start_active", "_continuation_active")):
        raise RuntimeError("This Agent already has an active execution")
    start = _HostStart(agent, asyncio.current_task(), hashlib.sha256(goal.encode()).hexdigest(), deepcopy(_policy(agent)))
    agent._execution_admission = agent._execution_assisted_dispatch = None
    agent._execution_start_active = True
    token = _STARTING.set(start)
    try:
        return await agent.chat(message=goal, mode="execute", **chat_options)
    except BaseException:
        agent._execution_admission = agent._execution_assisted_dispatch = None
        raise
    finally:
        _STARTING.reset(token)
        agent._execution_start_active = False


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
    if getattr(agent, "_execution_start_active", False):
        start = _STARTING.get()
        if start is None or start.agent is not agent or start.task is not asyncio.current_task() or not start.claimed or start.consumed:
            raise PermissionError("Host execution admission is unavailable outside its start call")
        if start.policy != _policy(agent):
            raise PermissionError("Execution policy changed during host task preparation")
        start.consumed = True
        decision = ToolIntentDecision("action", 1.0, "host_admitted_goal", ("typed_task",))
    else:
        decision = (
            route_with_confirmation(message, history) if mode == "execute"
            else ToolIntentDecision("action", 1.0, "explicit_non_execute_mode", (mode,))
        )
    agent._execution_admission = (
        ExecutionAdmission(
            hashlib.sha256(message.encode()).hexdigest(), decision, deepcopy(_policy(agent)),
        ) if mode == "execute" and decision.tool_eligible else None
    )
    agent._execution_assisted_dispatch = None
    return decision


def assisted_dispatch(agent, base_dispatch, user_message):
    """Keep preparation, observations and retry state within one admitted goal."""
    active = _ACTIVE.get()
    if active is not None and active[0] is agent:
        cached = getattr(agent, "_execution_assisted_dispatch", None)
        if (
            cached is None or cached[0] is not active[1]
            or cached[1] is not base_dispatch or cached[2] is not agent._assistant
        ):
            raise PermissionError("The admitted execution dispatcher is no longer available")
        return cached[3]
    wrapped = agent._assistant.wrap(base_dispatch, user_message)
    admission = getattr(agent, "_execution_admission", None)
    if admission is not None:
        agent._execution_assisted_dispatch = (admission, base_dispatch, agent._assistant, wrapped)
    return wrapped


@contextmanager
def continuation_scope(agent, goal):
    """Grant one host call the existing goal admission, never a text-based bypass."""
    admission = getattr(agent, "_execution_admission", None)
    if getattr(agent, "_closed", False):
        raise RuntimeError("agent is closed")
    if getattr(agent, "_continuation_active", False):
        raise RuntimeError("This Agent is already continuing an execution")
    if getattr(agent, "_chat_active_task", None) is not None or getattr(agent, "_execution_start_active", False):
        raise RuntimeError("This Agent already has an active execution")
    if (
        not isinstance(admission, ExecutionAdmission) or not isinstance(goal, str)
        or admission.goal_sha256 != hashlib.sha256(goal.encode()).hexdigest()
    ):
        raise PermissionError("No matching action admission exists on this Agent")
    if admission.policy != _policy(agent):
        raise PermissionError("Execution policy changed; a new action admission is required")
    agent._continuation_active = True
    token = _ACTIVE.set((agent, admission, asyncio.current_task()))
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        agent._continuation_active = False


def ensure_routing_state(agent) -> None:
    """Support lightweight test agents created without ``__init__``."""
    if not hasattr(agent, "_preferred_language"):
        agent._preferred_language = None
    if not hasattr(agent, "_last_routing_decision"):
        agent._last_routing_decision = None
    if not hasattr(agent, "_routing_metrics"):
        agent._routing_metrics = {
            "turns": 0,
            "answer_only_turns": 0,
            "ambiguous_turns": 0,
            "action_turns": 0,
            "tool_calls_attempted": 0,
            "tool_calls_executed": 0,
            "tool_calls_blocked": 0,
        }


def record_routing_decision(
    agent,
    decision: ToolIntentDecision,
) -> None:
    agent._ensure_routing_state()
    agent._last_routing_decision = decision
    agent._routing_metrics["turns"] += 1
    key = "{}_turns".format(decision.mode)
    if key in agent._routing_metrics:
        agent._routing_metrics[key] += 1


def tools_for_route(
    agent,
    decision: ToolIntentDecision,
    mode: str,
) -> List[Dict]:
    """Expose the smallest schema set justified by this turn."""
    tools = list(agent._tools or [])
    if mode != "execute":
        return tools
    if decision.mode == "answer_only":
        return []

    enforcer = agent._permission_enforcer
    maximum = (
        PermissionLevel.READ_ONLY
        if decision.mode == "ambiguous"
        else enforcer.level
    )
    return [
        tool for tool in tools
        if enforcer.required_level(agent._tool_name(tool), {}) <= maximum
    ]
