# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Abstract LLM provider interface."""
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict, List, Tuple


# Type alias for the dispatch function
DispatchFn = Callable[[str, dict], Coroutine[Any, Any, dict]]


class LLMProvider(ABC):
    """Abstract base class for LLM providers (OpenAI, Anthropic, etc.)."""

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict],
        dispatch_fn: DispatchFn,
        max_rounds: int = 15,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Run a chat loop with function calling.

        Returns (final_message, tool_call_log).
        """
