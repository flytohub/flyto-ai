# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""flyto-ai — Natural language automation agent with lazy public exports."""

from importlib import import_module
from typing import Any

from flyto_ai.agent import Agent
from flyto_ai.config import AgentConfig
from flyto_ai.package_metadata import package_version


__version__ = package_version()
_EXPORTS = {
    "Agent": ("flyto_ai.agent", "Agent"),
    "AgentConfig": ("flyto_ai.config", "AgentConfig"),
    "ApiClient": ("flyto_ai.protocols", "ApiClient"),
    "ToolExecutor": ("flyto_ai.protocols", "ToolExecutor"),
    "PermissionLevel": ("flyto_ai.permissions", "PermissionLevel"),
    "PermissionEnforcer": ("flyto_ai.permissions", "PermissionEnforcer"),
    "ChatMessage": ("flyto_ai.models", "ChatMessage"),
    "ChatRequest": ("flyto_ai.models", "ChatRequest"),
    "ChatResponse": ("flyto_ai.models", "ChatResponse"),
    "StreamEvent": ("flyto_ai.models", "StreamEvent"),
    "StreamEventType": ("flyto_ai.models", "StreamEventType"),
    "UsageStats": ("flyto_ai.models", "UsageStats"),
    "ClaudeCodeAgent": ("flyto_ai.agents.claude_code", "ClaudeCodeAgent"),
}
__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError("module 'flyto_ai' has no attribute '{}'".format(name))
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def create_agent(
    provider: str = "",
    api_key: str = "",
    model: str = "",
    **kwargs,
) -> Agent:
    """Convenience factory for creating an Agent."""

    config = AgentConfig(provider=provider, api_key=api_key, model=model, **kwargs)
    return Agent(config=config)
