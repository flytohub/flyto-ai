# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Pre-gather codebase context from flyto-indexer before spawning Claude Code.

Uses the same lazy-import pattern as core_tools.py to avoid hard dependency.
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Cached indexer handle (None = not checked, False = unavailable)
_cached_registry: Any = None
_registry_checked = False


def _get_registry() -> Any:
    """Lazily import flyto-indexer ToolRegistry. Cached after first call."""
    global _cached_registry, _registry_checked
    if _registry_checked:
        return _cached_registry
    _registry_checked = True
    try:
        from flyto_indexer.tool_registry import ToolRegistry
        _cached_registry = ToolRegistry()
        logger.debug("flyto-indexer ToolRegistry loaded")
    except ImportError:
        _cached_registry = None
        logger.debug("flyto-indexer not installed — indexer context disabled")
    return _cached_registry


async def gather_context(message: str, working_dir: str) -> str:
    """Gather impact analysis and references from the indexer.

    Returns a formatted string to inject into the system prompt.
    Returns empty string if indexer is unavailable or no useful data found.
    """
    registry = _get_registry()
    if registry is None:
        return ""

    sections = []
    try:
        projects = await _call(registry, "list_projects", {})
        if projects and isinstance(projects, dict):
            names = list(projects.get("projects", {}).keys())
            if names:
                sections.append("**Indexed projects**: {}".format(", ".join(names)))

        health = await _call(registry, "code_health_score", {"project": ""})
        if health and isinstance(health, dict):
            grade = health.get("grade", "")
            score = health.get("score", "")
            if grade:
                sections.append("**Code health**: {} ({}/100)".format(grade, score))

    except Exception as e:
        logger.debug("Indexer context gathering failed: %s", e)

    return "\n".join(sections)


async def _call(registry: Any, tool_name: str, args: Dict[str, Any]) -> Optional[Dict]:
    """Call an indexer tool, returning None on failure."""
    try:
        handler = getattr(registry, "dispatch", None)
        if handler is None:
            return None
        result = await handler(tool_name, args)
        if isinstance(result, dict) and result.get("ok") is False:
            return None
        return result
    except Exception as e:
        logger.debug("Indexer call %s failed: %s", tool_name, e)
        return None
