# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""flyto-core MCP tool bridge — lazily imports core handler."""
import logging
import threading
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Shared browser session store across tool calls within a chat session.
# Keys are browser session IDs from browser.launch results.
# Cleared via clear_browser_sessions() between independent chat sessions.
_browser_sessions: Dict[str, Any] = {}
_browser_sessions_lock = threading.Lock()


def clear_browser_sessions() -> None:
    """Clear the shared browser session store (call between independent chats)."""
    with _browser_sessions_lock:
        _browser_sessions.clear()


_cached_handler = None
_handler_checked = False


def _get_mcp_handler():
    """Lazily import mcp_handler to avoid circular imports.

    Returns None if flyto-core is not installed. Result is cached.
    """
    global _cached_handler, _handler_checked
    if _handler_checked:
        return _cached_handler
    _handler_checked = True
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
        _cached_handler = {
            "TOOLS": TOOLS,
            "list_modules": list_modules,
            "search_modules": search_modules,
            "get_module_info": get_module_info,
            "get_module_examples": get_module_examples,
            "execute_module": execute_module,
            "validate_params": validate_params,
        }
    except ImportError:
        _cached_handler = None
    return _cached_handler


def get_core_tool_defs():
    """Return flyto-core MCP tool definitions (empty list if not installed)."""
    handler = _get_mcp_handler()
    return handler["TOOLS"] if handler else []


# ---------------------------------------------------------------------------
# Browser retry — transient error detection + smart retry at dispatch level
# ---------------------------------------------------------------------------

_TRANSIENT_PATTERNS = [
    "timeout", "timed out", "target closed", "session closed",
    "navigation failed", "browser disconnected",
    "execution context was destroyed", "connection refused",
    "net::err_", "page crashed",
]

# Patterns that indicate the browser session itself is dead (needs relaunch)
_SESSION_DEAD_PATTERNS = [
    "target closed", "session closed", "browser disconnected",
    "browser has been closed", "browser.close",
]


def _is_transient_error(error_msg: str) -> bool:
    """Check if an error message indicates a transient browser failure."""
    lower = error_msg.lower()
    return any(p in lower for p in _TRANSIENT_PATTERNS)


def _is_session_dead(error_msg: str) -> bool:
    """Check if an error message indicates the browser session is dead."""
    lower = error_msg.lower()
    return any(p in lower for p in _SESSION_DEAD_PATTERNS)


async def _relaunch_browser() -> Dict[str, Any]:
    """Attempt to relaunch a fresh browser session."""
    handler = _get_mcp_handler()
    if handler is None:
        return {"ok": False, "error": "flyto-core not installed"}
    try:
        result = await handler["execute_module"](
            module_id="browser.launch",
            params={},
            context=None,
            browser_sessions=_browser_sessions,
        )
        return result
    except Exception as e:
        return {"ok": False, "error": "Relaunch failed: {}".format(e)}


async def dispatch_core_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool call to the flyto-core MCP handler.

    For browser.* modules, transient errors trigger one automatic retry.
    If the session is dead, a fresh browser.launch is attempted before retrying.
    """
    result = await _dispatch_core_tool_inner(name, arguments)

    # Smart retry — only for browser module execute_module calls
    if (
        name == "execute_module"
        and isinstance(result, dict)
        and not result.get("ok", True)
    ):
        module_id = arguments.get("module_id", "")
        error_msg = str(result.get("error", ""))

        if module_id.startswith("browser.") and _is_transient_error(error_msg):
            logger.info("Browser transient error on %s, retrying: %s", module_id, error_msg[:100])

            # Session dead → relaunch first
            if _is_session_dead(error_msg):
                relaunch = await _relaunch_browser()
                if not relaunch.get("ok", False):
                    relaunch_err = relaunch.get("error", "unknown")
                    logger.warning("Browser relaunch failed: %s", relaunch_err)
                    return {
                        "ok": False,
                        "error": "Browser session dead ({}). Relaunch also failed: {}".format(
                            error_msg[:100], relaunch_err,
                        ),
                    }

            # Retry once
            result = await _dispatch_core_tool_inner(name, arguments)

    return result


_sandbox_mgr = None


def set_sandbox_manager(mgr) -> None:
    """Set the sandbox manager for sandboxed module execution."""
    global _sandbox_mgr
    _sandbox_mgr = mgr


async def _dispatch_core_tool_inner(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Core dispatch logic (no retry)."""
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
        module_id = arguments.get("module_id", "")
        # Sandbox: route dangerous categories to Docker container
        if _sandbox_mgr and _sandbox_mgr.needs_sandbox(module_id):
            return await _sandbox_mgr.execute(
                module_id, arguments.get("params", {}), arguments.get("context"),
            )
        return await handler["execute_module"](
            module_id=module_id,
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
