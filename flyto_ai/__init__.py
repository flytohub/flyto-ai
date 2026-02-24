# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""flyto-ai — Natural language automation agent."""
from flyto_ai.agent import Agent
from flyto_ai.config import AgentConfig
from flyto_ai.models import ChatMessage, ChatRequest, ChatResponse, StreamEvent, StreamEventType, UsageStats

__version__ = "0.9.7"
__all__ = [
    "Agent", "AgentConfig",
    "ChatMessage", "ChatRequest", "ChatResponse",
    "StreamEvent", "StreamEventType", "UsageStats",
    "__version__",
]


def create_agent(
    provider: str = "",
    api_key: str = "",
    model: str = "",
    **kwargs,
) -> Agent:
    """Convenience factory for creating an Agent."""
    config = AgentConfig(provider=provider, api_key=api_key, model=model, **kwargs)
    return Agent(config=config)
