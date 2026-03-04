# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""SubAgent — isolated agent session with timeout and tool restrictions."""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SubAgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SubAgentResult:
    """Result from a sub-agent execution."""
    run_id: str
    status: SubAgentStatus
    message: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    execution_results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: int = 0
    usage: Optional[Dict[str, int]] = None


class SubAgent:
    """An isolated agent session with its own context and timeout.

    Sub-agents:
    - Have their own conversation history (isolated from parent)
    - Have restricted tool access based on depth
    - Have independent timeout enforcement
    - Report results back to the parent when done
    """

    def __init__(
        self,
        task: str,
        parent_session_id: str,
        depth: int = 1,
        timeout: int = 300,
        allowed_tools: Optional[frozenset] = None,
        max_tool_rounds: int = 10,
        config=None,
    ) -> None:
        self.run_id = uuid.uuid4().hex[:12]
        self.task = task
        self.parent_session_id = parent_session_id
        self.depth = depth
        self.timeout = timeout
        self.allowed_tools = allowed_tools
        self.max_tool_rounds = max_tool_rounds
        self.config = config

        self.status = SubAgentStatus.PENDING
        self.result: Optional[SubAgentResult] = None
        self._task_handle: Optional[asyncio.Task] = None
        self._cancel_event = asyncio.Event()

    async def run(self) -> SubAgentResult:
        """Execute the sub-agent task with timeout enforcement."""
        self.status = SubAgentStatus.RUNNING
        t0 = time.monotonic()

        try:
            result = await asyncio.wait_for(
                self._execute(),
                timeout=self.timeout,
            )
            self.status = SubAgentStatus.COMPLETED
            duration = int((time.monotonic() - t0) * 1000)
            self.result = SubAgentResult(
                run_id=self.run_id,
                status=SubAgentStatus.COMPLETED,
                message=result.get("message", ""),
                tool_calls=result.get("tool_calls", []),
                execution_results=result.get("execution_results", []),
                duration_ms=duration,
                usage=result.get("usage"),
            )
        except asyncio.TimeoutError:
            self.status = SubAgentStatus.TIMEOUT
            duration = int((time.monotonic() - t0) * 1000)
            self.result = SubAgentResult(
                run_id=self.run_id,
                status=SubAgentStatus.TIMEOUT,
                error="Sub-agent timed out after {}s".format(self.timeout),
                duration_ms=duration,
            )
            logger.warning("Sub-agent %s timed out after %ds", self.run_id, self.timeout)
        except asyncio.CancelledError:
            self.status = SubAgentStatus.CANCELLED
            duration = int((time.monotonic() - t0) * 1000)
            self.result = SubAgentResult(
                run_id=self.run_id,
                status=SubAgentStatus.CANCELLED,
                error="Sub-agent cancelled (parent stopped)",
                duration_ms=duration,
            )
        except Exception as e:
            self.status = SubAgentStatus.FAILED
            duration = int((time.monotonic() - t0) * 1000)
            self.result = SubAgentResult(
                run_id=self.run_id,
                status=SubAgentStatus.FAILED,
                error=str(e)[:500],
                duration_ms=duration,
            )
            logger.warning("Sub-agent %s failed: %s", self.run_id, str(e)[:200])

        return self.result

    async def _execute(self) -> Dict[str, Any]:
        """Internal execution — creates a child Agent and runs the task."""
        from flyto_ai.agent import Agent
        from flyto_ai.config import AgentConfig

        # Create child config (inherits parent, but with reduced rounds)
        child_config = self.config or AgentConfig.from_env()
        child_config.max_tool_rounds = self.max_tool_rounds
        child_config.enable_transcript = False  # sub-agents don't create transcripts

        # Create child agent
        child = Agent(config=child_config)

        # Apply tool restrictions
        if self.allowed_tools is not None:
            child._tools = [
                t for t in child._tools
                if t.get("name", "") in self.allowed_tools
            ]

        # Run the task
        response = await child.chat(
            message=self.task,
            mode="execute",
        )

        return {
            "message": response.message,
            "tool_calls": response.tool_calls,
            "execution_results": response.execution_results,
            "usage": response.usage.model_dump() if response.usage else None,
        }

    def cancel(self) -> None:
        """Cancel this sub-agent."""
        self._cancel_event.set()
        if self._task_handle and not self._task_handle.done():
            self._task_handle.cancel()
        self.status = SubAgentStatus.CANCELLED
