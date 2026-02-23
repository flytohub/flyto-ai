# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Abstract LLM provider interface."""
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from flyto_ai.models import StreamCallback

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
        on_stream: Optional[StreamCallback] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Run a chat loop with function calling.

        Parameters
        ----------
        on_stream : callable, optional
            ``on_stream(StreamEvent)`` — called for each streaming event.
            When set, providers should enable streaming for LLM responses.
            When None, behaviour is unchanged (non-streaming).

        Returns (final_message, tool_call_log).
        """
