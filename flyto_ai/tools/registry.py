# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tool registry — aggregates MCP tools, AI-exclusive tools, and blueprint tools."""
import contextvars
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)

# Type alias for async tool handlers
ToolHandler = Callable[..., Coroutine[Any, Any, Dict[str, Any]]]

# Recursion depth limit for dispatch
MAX_DISPATCH_DEPTH = 10
_dispatch_depth: contextvars.ContextVar[int] = contextvars.ContextVar("dispatch_depth", default=0)


class ToolRegistry:
    """Collects tool definitions and their handlers for LLM function calling.

    Tools are in MCP format: ``{name, description, inputSchema}``.
    """

    def __init__(self) -> None:
        self._tools: List[Dict] = []
        self._handlers: Dict[str, ToolHandler] = {}

    def register(self, tool_def: Dict, handler: ToolHandler) -> None:
        """Register a tool definition with its async handler."""
        name = tool_def["name"]
        self._tools.append(tool_def)
        self._handlers[name] = handler

    def register_many(self, tool_defs: List[Dict], handler: ToolHandler) -> None:
        """Register multiple tools that share a dispatcher handler."""
        for tool_def in tool_defs:
            self._tools.append(tool_def)
            self._handlers[tool_def["name"]] = handler

    @property
    def tools(self) -> List[Dict]:
        """All registered tool definitions in MCP format."""
        return list(self._tools)

    async def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool call to its handler (with recursion depth limit)."""
        depth = _dispatch_depth.get()
        if depth >= MAX_DISPATCH_DEPTH:
            return {"ok": False, "error": "Dispatch recursion limit ({}) exceeded".format(MAX_DISPATCH_DEPTH)}

        handler = self._handlers.get(name)
        if handler is None:
            return {"ok": False, "error": "Unknown tool: {}".format(name)}

        token = _dispatch_depth.set(depth + 1)
        try:
            return await handler(name, arguments)
        except Exception as e:
            logger.warning("Tool call failed (%s): %s", name, e)
            return {"ok": False, "error": str(e)}
        finally:
            _dispatch_depth.reset(token)

    def to_openai_format(self) -> List[Dict]:
        """Convert all tools to OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"],
                },
            }
            for t in self._tools
        ]

    def to_anthropic_format(self) -> List[Dict]:
        """Convert all tools to Anthropic tool use format."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["inputSchema"],
            }
            for t in self._tools
        ]
