# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""AgentOrchestrator — spawn, monitor, and collect sub-agent results."""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from flyto_ai.orchestration.policies import OrchestrationPolicy
from flyto_ai.orchestration.sub_agent import SubAgent, SubAgentResult, SubAgentStatus

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Manages sub-agent lifecycle: spawn, monitor, collect, and cascade kill.

    Usage::

        orch = AgentOrchestrator(parent_session_id="abc123")
        run_id = await orch.spawn("Search for Python tutorials", depth=1)
        result = await orch.wait(run_id)
        # or
        results = await orch.wait_all()
    """

    def __init__(
        self,
        parent_session_id: str,
        policy: Optional[OrchestrationPolicy] = None,
        config=None,
    ) -> None:
        self._parent_session_id = parent_session_id
        self._policy = policy or OrchestrationPolicy()
        self._config = config
        self._agents: Dict[str, SubAgent] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    @property
    def active_count(self) -> int:
        """Number of currently running sub-agents."""
        return sum(
            1 for a in self._agents.values()
            if a.status == SubAgentStatus.RUNNING
        )

    @property
    def all_agents(self) -> List[SubAgent]:
        """All sub-agents (including completed)."""
        return list(self._agents.values())

    async def spawn(
        self,
        task: str,
        depth: int = 1,
        timeout: Optional[int] = None,
    ) -> str:
        """Spawn a new sub-agent and return its run_id.

        Raises
        ------
        RuntimeError
            If max depth or max concurrent limit is exceeded.
        """
        if not self._policy.can_spawn_at_depth(depth):
            raise RuntimeError(
                "Cannot spawn sub-agent: max depth {} exceeded (current depth: {})".format(
                    self._policy.max_depth, depth
                )
            )

        if self.active_count >= self._policy.max_concurrent:
            raise RuntimeError(
                "Cannot spawn sub-agent: max concurrent limit {} reached".format(
                    self._policy.max_concurrent
                )
            )

        agent = SubAgent(
            task=task,
            parent_session_id=self._parent_session_id,
            depth=depth,
            timeout=timeout or self._policy.default_timeout,
            allowed_tools=self._policy.allowed_tools_at_depth(depth),
            max_tool_rounds=self._policy.max_rounds_at_depth(depth),
            config=self._config,
        )
        self._agents[agent.run_id] = agent

        # Launch as background task
        task_handle = asyncio.create_task(agent.run())
        self._tasks[agent.run_id] = task_handle
        agent._task_handle = task_handle

        logger.info(
            "Spawned sub-agent %s (depth=%d, timeout=%ds): %s",
            agent.run_id, depth, agent.timeout, task[:80],
        )

        return agent.run_id

    async def wait(self, run_id: str) -> SubAgentResult:
        """Wait for a specific sub-agent to complete and return its result."""
        if run_id not in self._tasks:
            raise KeyError("Unknown sub-agent: {}".format(run_id))

        await self._tasks[run_id]
        return self._agents[run_id].result

    async def wait_all(self) -> List[SubAgentResult]:
        """Wait for all sub-agents to complete and return their results."""
        if not self._tasks:
            return []

        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        return [a.result for a in self._agents.values() if a.result]

    def cancel(self, run_id: str) -> None:
        """Cancel a specific sub-agent."""
        if run_id in self._agents:
            self._agents[run_id].cancel()
            logger.info("Cancelled sub-agent %s", run_id)

    def cancel_all(self) -> int:
        """Cancel all running sub-agents (cascade kill).

        Returns the number of agents cancelled.
        """
        count = 0
        for agent in self._agents.values():
            if agent.status == SubAgentStatus.RUNNING:
                agent.cancel()
                count += 1
        if count:
            logger.info("Cascade kill: cancelled %d sub-agents", count)
        return count

    def get_status(self, run_id: str) -> Optional[SubAgentStatus]:
        """Get the status of a sub-agent."""
        agent = self._agents.get(run_id)
        return agent.status if agent else None

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all sub-agents."""
        return {
            "total": len(self._agents),
            "active": self.active_count,
            "agents": [
                {
                    "run_id": a.run_id,
                    "task": a.task[:80],
                    "depth": a.depth,
                    "status": a.status.value,
                    "duration_ms": a.result.duration_ms if a.result else 0,
                }
                for a in self._agents.values()
            ],
        }
