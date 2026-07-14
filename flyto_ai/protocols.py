# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Runtime protocols — decouple agent loop from concrete provider/tool implementations.

Inspired by claw-code's ``ConversationRuntime<C: ApiClient, T: ToolExecutor>`` pattern.
Existing classes already satisfy these protocols:
  - ``LLMProvider`` (providers/base.py) → ``ApiClient``
  - ``ToolRegistry`` (tools/registry.py) → ``ToolExecutor``
"""
from __future__ import annotations

from typing import Any, Callable, Coroutine, Dict, List, Optional, Protocol, Tuple, runtime_checkable

# Re-export the canonical type alias so consumers don't need providers.base
DispatchFn = Callable[[str, dict], Coroutine[Any, Any, dict]]

# StreamCallback is Callable[[StreamEvent], None] — defined here to avoid
# importing flyto_ai.models (which requires pydantic) at module level.
StreamCallback = Optional[Callable[..., None]]


@runtime_checkable
class ApiClient(Protocol):
    """Protocol for LLM API clients.

    Any object with a compatible ``chat`` async method satisfies this protocol.
    The existing ``LLMProvider`` ABC is already structurally compatible.
    """

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict],
        dispatch_fn: DispatchFn,
        max_rounds: int = 30,
        on_stream: Optional[StreamCallback] = None,
    ) -> Tuple[str, List[Dict[str, Any]], int, Dict[str, int]]:
        """Run a chat loop with function calling.

        Returns ``(final_message, tool_call_log, rounds_used, usage_dict)``.
        """
        ...


@runtime_checkable
class ToolExecutor(Protocol):
    """Protocol for tool dispatch backends.

    Any object with ``tools`` (property) and ``dispatch`` (async method)
    satisfies this protocol.  ``ToolRegistry`` is already compatible.
    """

    @property
    def tools(self) -> List[Dict]:
        """All registered tool definitions in MCP format."""
        ...

    async def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool call by name. Returns a result dict."""
        ...
