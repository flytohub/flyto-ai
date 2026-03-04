# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Orchestration policies — depth limits, tool access control, timeouts."""
from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Set

# Default allowed tools per depth level.
# Deeper agents get fewer tools (security-first).
_TOOLS_BY_DEPTH = {
    0: None,  # root agent: all tools allowed
    1: frozenset({
        "list_modules", "search_modules", "get_module_info", "get_module_examples",
        "execute_module", "validate_params",
        "inspect_page",
        "list_blueprints", "use_blueprint",
    }),
    2: frozenset({
        "list_modules", "search_modules", "get_module_info", "get_module_examples",
        "execute_module", "validate_params",
    }),
    3: frozenset({
        "search_modules", "get_module_info",
        "execute_module", "validate_params",
    }),
}


@dataclass(frozen=True)
class OrchestrationPolicy:
    """Policy governing sub-agent behavior.

    Attributes
    ----------
    max_depth : int
        Maximum nesting depth (0 = root, 3 = deepest sub-agent).
        Default 3 is safer than OpenClaw's 5.
    default_timeout : int
        Per sub-agent timeout in seconds.
    max_concurrent : int
        Maximum concurrent sub-agents from a single parent.
    cascade_kill : bool
        If True, stopping a parent kills all its children.
    max_tool_rounds_per_depth : int
        Sub-agents get fewer tool rounds at deeper levels.
    """
    max_depth: int = 3
    default_timeout: int = 300
    max_concurrent: int = 5
    cascade_kill: bool = True
    max_tool_rounds_per_depth: int = 10

    def allowed_tools_at_depth(self, depth: int) -> Optional[FrozenSet[str]]:
        """Return the tool allowlist for a given depth.

        Returns None if all tools are allowed (depth 0).
        """
        if depth <= 0:
            return None
        # Clamp to deepest defined level
        level = min(depth, max(_TOOLS_BY_DEPTH.keys()))
        return _TOOLS_BY_DEPTH.get(level, _TOOLS_BY_DEPTH[3])

    def max_rounds_at_depth(self, depth: int) -> int:
        """Return max tool rounds for a given depth."""
        if depth <= 0:
            return 30  # root default
        # Reduce rounds as depth increases
        return max(5, self.max_tool_rounds_per_depth - (depth - 1) * 2)

    def can_spawn_at_depth(self, depth: int) -> bool:
        """Check if spawning is allowed at the given depth."""
        return depth < self.max_depth
