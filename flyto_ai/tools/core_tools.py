# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""flyto-core MCP tool bridge — lazily imports core handler."""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Shared browser session store across tool calls within a chat session
_browser_sessions: Dict[str, Any] = {}


def _get_mcp_handler():
    """Lazily import mcp_handler to avoid circular imports.

    Returns None if flyto-core is not installed.
    """
    try:
        from core.mcp_handler import (
            TOOLS,
            list_modules,
            search_modules,
            get_module_info,
            get_module_examples,
            execute_module,
            validate_params,
        )
        return {
            "TOOLS": TOOLS,
            "list_modules": list_modules,
            "search_modules": search_modules,
            "get_module_info": get_module_info,
            "get_module_examples": get_module_examples,
            "execute_module": execute_module,
            "validate_params": validate_params,
        }
    except ImportError:
        return None


def get_core_tool_defs():
    """Return flyto-core MCP tool definitions (empty list if not installed)."""
    handler = _get_mcp_handler()
    return handler["TOOLS"] if handler else []


async def dispatch_core_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool call to the flyto-core MCP handler."""
    handler = _get_mcp_handler()
    if handler is None:
        return {"ok": False, "error": "flyto-core not installed. Run: pip install flyto-core"}

    if name == "list_modules":
        return handler["list_modules"](category=arguments.get("category"))

    elif name == "search_modules":
        return handler["search_modules"](
            query=arguments.get("query", ""),
            category=arguments.get("category"),
            limit=arguments.get("limit", 20),
        )

    elif name == "get_module_info":
        return handler["get_module_info"](module_id=arguments.get("module_id", ""))

    elif name == "get_module_examples":
        return handler["get_module_examples"](module_id=arguments.get("module_id", ""))

    elif name == "execute_module":
        return await handler["execute_module"](
            module_id=arguments.get("module_id", ""),
            params=arguments.get("params", {}),
            context=arguments.get("context"),
            browser_sessions=_browser_sessions,
        )

    elif name == "validate_params":
        return handler["validate_params"](
            module_id=arguments.get("module_id", ""),
            params=arguments.get("params", {}),
        )

    return {"ok": False, "error": "Unknown core tool: {}".format(name)}
