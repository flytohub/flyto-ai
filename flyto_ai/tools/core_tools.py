# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""flyto-core MCP tool bridge — lazily imports core handler."""
import hashlib
import importlib.metadata
import logging
import re
import threading
from copy import deepcopy
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CORE_MCP_CONTRACT_VERSION = "flyto-core-mcp.v1"

# Shared browser session store across tool calls within a chat session.
# Keys are browser session IDs from browser.launch results.
# Cleared via clear_browser_sessions() between independent chat sessions.
_browser_sessions: Dict[str, Any] = {}
_browser_sessions_lock = threading.Lock()

# Browser cascade breaker: when browser.launch fails, skip subsequent browser.* calls.
# Reset on each new browser.launch attempt.
_browser_launch_failed: bool = False
_browser_launch_error: str = ""

# Goto circuit breaker: after N consecutive goto failures, return a non-retryable error.
_goto_consecutive_fails: int = 0
_GOTO_MAX_FAILS: int = 3

_READ_ONLY_CORE_TOOLS = frozenset({
    "list_modules",
    "search_modules",
    "get_module_info",
    "get_module_examples",
    "validate_params",
    "list_recipes",
    "get_core_capability_manifest",
})

_EXECUTION_CORE_TOOLS = frozenset({"execute_module", "run_recipe"})

_DANGER_MODULE_CATEGORIES = frozenset({
    "shell", "process", "docker", "k8s", "ssh", "network", "port", "dns",
    "file", "path", "env", "git",
})

CORE_CAPABILITY_MANIFEST_TOOL = {
    "name": "get_core_capability_manifest",
    "description": (
        "Return the flyto-core MCP capability manifest for agents and cloud UIs: "
        "contract version, installed core version, tool fingerprint, recipes support, "
        "module categories, and risk/approval metadata for callable tools."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "include_tools": {
                "type": "boolean",
                "description": "Include per-tool metadata in the response.",
                "default": True,
            },
            "include_categories": {
                "type": "boolean",
                "description": "Include flyto-core module category counts when available.",
                "default": True,
            },
        },
    },
}


def clear_browser_sessions() -> None:
    """Clear the shared browser session store (call between independent chats)."""
    global _browser_launch_failed, _browser_launch_error, _goto_consecutive_fails
    with _browser_sessions_lock:
        _browser_sessions.clear()
    _browser_launch_failed = False
    _browser_launch_error = ""
    _goto_consecutive_fails = 0


def get_browser_status() -> str:
    """Get a prompt hint about browser state for the LLM.

    Returns empty string if no browser running, or an instruction
    telling the LLM to reuse the existing browser.
    """
    with _browser_sessions_lock:
        if not _browser_sessions:
            return ""
        return (
            "BROWSER IS ALREADY RUNNING. Do NOT call browser.launch again. "
            "Continue using browser.goto / browser.snapshot / browser.click directly."
        )


def _is_ok(result: Dict[str, Any]) -> bool:
    """Check if a module result indicates success.

    flyto-core modules return {"status": "success"} without an "ok" field.
    Normalize: ok=true OR status=="success" → success.
    """
    if result.get("ok"):
        return True
    return result.get("status") == "success"


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
        # Recipe support is optional (flyto-core >= 2.15.0)
        try:
            from core.mcp_handler import list_recipes, run_recipe
            _cached_handler["list_recipes"] = list_recipes
            _cached_handler["run_recipe"] = run_recipe
        except ImportError:
            pass
    except ImportError:
        _cached_handler = None
    return _cached_handler


def _core_package_version() -> str:
    """Return installed flyto-core package version when available."""
    try:
        return importlib.metadata.version("flyto-core")
    except importlib.metadata.PackageNotFoundError:
        return ""
    except Exception:
        return ""


def _risk_for_tool(name: str) -> str:
    if name in _READ_ONLY_CORE_TOOLS:
        return "read_only"
    if name in _EXECUTION_CORE_TOOLS:
        return "workspace_write"
    return "workspace_write"


def _approval_policy_for_tool(name: str) -> str:
    if name == "execute_module":
        return "module_category_runtime"
    if name == "run_recipe":
        return "recipe_runtime"
    return "none"


def _tool_annotations(name: str) -> Dict[str, bool]:
    read_only = name in _READ_ONLY_CORE_TOOLS
    return {
        "readOnlyHint": read_only,
        "destructiveHint": not read_only,
        "idempotentHint": read_only,
    }


def _enrich_core_tool_def(tool_def: Dict[str, Any]) -> Dict[str, Any]:
    """Attach agent-facing MCP metadata without changing the callable schema."""
    tool = deepcopy(tool_def)
    name = tool.get("name", "")
    tool.setdefault("description", "")
    tool.setdefault("inputSchema", {"type": "object", "properties": {}})
    tool["annotations"] = _tool_annotations(name)
    tool["metadata"] = {
        "source": "flyto-core",
        "contract_version": CORE_MCP_CONTRACT_VERSION,
        "risk_level": _risk_for_tool(name),
        "approval_policy": _approval_policy_for_tool(name),
        "evidence_fields": [
            "run_id",
            "tool_name",
            "module_id",
            "recipe_name",
            "params_valid",
            "ok",
            "duration_ms",
        ],
    }
    return tool


def _manifest_fingerprint(tools: List[Dict[str, Any]]) -> str:
    payload = [
        {
            "name": t.get("name", ""),
            "inputSchema": t.get("inputSchema", {}),
            "metadata": t.get("metadata", {}),
        }
        for t in tools
    ]
    raw = json_dumps(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def json_dumps(value: Any) -> str:
    """Stable JSON helper kept local to avoid importing heavier utilities."""
    import json
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _core_tool_defs_from_handler(handler: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = [_enrich_core_tool_def(t) for t in handler.get("TOOLS", [])]
    tools.append(_enrich_core_tool_def(CORE_CAPABILITY_MANIFEST_TOOL))
    return tools


def get_core_tool_defs():
    """Return flyto-core MCP tool definitions (empty list if not installed)."""
    handler = _get_mcp_handler()
    return _core_tool_defs_from_handler(handler) if handler else []


def get_core_capability_manifest(
    include_tools: bool = True,
    include_categories: bool = True,
) -> Dict[str, Any]:
    """Build the flyto-core MCP manifest used by agents and cloud diagnostics."""
    handler = _get_mcp_handler()
    if handler is None:
        return {
            "ok": False,
            "source": "flyto-core",
            "contract_version": CORE_MCP_CONTRACT_VERSION,
            "error": "flyto-core not installed. Run: pip install flyto-core",
        }

    tools = _core_tool_defs_from_handler(handler)
    manifest: Dict[str, Any] = {
        "ok": True,
        "source": "flyto-core",
        "contract_version": CORE_MCP_CONTRACT_VERSION,
        "core_version": _core_package_version(),
        "tool_count": len(tools),
        "tool_fingerprint": _manifest_fingerprint(tools),
        "recipes_available": bool(handler.get("list_recipes") and handler.get("run_recipe")),
        "approval_model": {
            "execute_module": "module category decides runtime approval",
            "run_recipe": "recipe content decides runtime approval",
            "sensitive_inputs": "runtime secrets only; never request credentials through MCP elicitation",
        },
    }

    if include_tools:
        manifest["tools"] = [
            {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema", {}),
                "annotations": t.get("annotations", {}),
                "metadata": t.get("metadata", {}),
            }
            for t in tools
        ]

    if include_categories:
        try:
            categories = handler["list_modules"](category=None).get("categories", [])
            manifest["categories"] = categories
            manifest["module_count"] = sum(c.get("count", 0) for c in categories)
        except Exception as e:
            manifest["categories_error"] = str(e)

    return manifest


# ---------------------------------------------------------------------------
# Browser retry — transient error detection + smart retry at dispatch level
# ---------------------------------------------------------------------------

# Import shared error classification from flyto-core (canonical source)
try:
    from core.modules.atomic.llm._resilience import (
        is_transient_error as _is_transient_error,
        is_session_dead as _is_session_dead,
    )
except ImportError:
    # Fallback: inline patterns if flyto-core is too old
    _TRANSIENT_PATTERNS = [
        "timeout", "timed out", "target closed", "session closed",
        "navigation failed", "browser disconnected",
        "execution context was destroyed", "connection refused",
        "net::err_", "page crashed",
    ]
    _SESSION_DEAD_PATTERNS = [
        "target closed", "session closed", "browser disconnected",
        "browser has been closed", "browser.close",
    ]

    def _is_transient_error(error_msg: str) -> bool:
        lower = error_msg.lower()
        return any(p in lower for p in _TRANSIENT_PATTERNS)

    def _is_session_dead(error_msg: str) -> bool:
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
        and not _is_ok(result)
    ):
        module_id = arguments.get("module_id", "")
        error_msg = str(result.get("error", ""))

        if module_id.startswith("browser.") and _is_transient_error(error_msg):
            logger.info("Browser transient error on %s, retrying: %s", module_id, error_msg[:100])

            # Session dead → relaunch first
            if _is_session_dead(error_msg):
                relaunch = await _relaunch_browser()
                if not _is_ok(relaunch):
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


# ---------------------------------------------------------------------------
# search_modules guardrail — detect web search misuse
# ---------------------------------------------------------------------------

# Module-like pattern: dot notation (browser.launch, string.upper)
_MODULE_QUERY_RE = re.compile(r"[a-z][a-z0-9_]*\.[a-z]", re.IGNORECASE)

# Automation-related keywords that signal a legitimate module search
_AUTOMATION_KEYWORDS = frozenset([
    "click", "type", "extract", "screenshot", "resize", "convert", "send",
    "email", "file", "image", "api", "http", "json", "csv", "pdf", "database",
    "sql", "scrape", "download", "upload", "parse", "format", "encode", "decode",
    "compress", "encrypt", "hash", "wait", "scroll", "select", "form", "login",
    "notify", "slack", "telegram", "webhook", "string", "array", "datetime",
    "evaluate", "snapshot", "launch", "goto", "navigate", "browser", "fill",
    "submit", "button", "input", "checkbox", "dropdown", "module", "workflow",
])


def _looks_like_module_query(query: str) -> bool:
    """Check if a query looks like it's searching for an automation module.

    Returns True if the query contains module-like patterns (dot notation)
    or automation-related keywords.
    """
    q = query.strip()
    if not q:
        return False

    # Dot notation like "browser.launch" → definitely module search
    if _MODULE_QUERY_RE.search(q):
        return True

    # Contains automation keywords → module search
    q_lower = q.lower()
    return any(kw in q_lower for kw in _AUTOMATION_KEYWORDS)


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

    elif name == "get_core_capability_manifest":
        return get_core_capability_manifest(
            include_tools=arguments.get("include_tools", True),
            include_categories=arguments.get("include_categories", True),
        )

    elif name == "search_modules":
        query = arguments.get("query", "")
        result = handler["search_modules"](
            query=query,
            category=arguments.get("category"),
            limit=arguments.get("limit", 20),
        )
        # Guardrail: if no results and query doesn't look like a module search, hint browser
        if isinstance(result, dict) and result.get("total", 0) == 0 and not _looks_like_module_query(query):
            result["web_search_hint"] = (
                "No modules match this query. This looks like a web search request. "
                "Use Browser Protocol instead: "
                "execute_module('browser.launch') → "
                "execute_module('browser.goto', {url: 'https://www.google.com/search?q=...'}) → "
                "execute_module('browser.snapshot') to read the results."
            )
            logger.info("search_modules guardrail: query looks like web search: %s", query[:50])
        return result

    elif name == "get_module_info":
        return handler["get_module_info"](module_id=arguments.get("module_id", ""))

    elif name == "get_module_examples":
        return handler["get_module_examples"](module_id=arguments.get("module_id", ""))

    elif name == "execute_module":
        global _browser_launch_failed, _browser_launch_error, _goto_consecutive_fails
        module_id = arguments.get("module_id", "")

        # Browser cascade breaker: if browser.launch already failed,
        # skip all subsequent browser.* calls immediately.
        if module_id.startswith("browser.") and module_id != "browser.launch" and _browser_launch_failed:
            return {
                "ok": False,
                "error": "Skipped: browser.launch failed earlier ({}). Fix browser.launch first.".format(
                    _browser_launch_error[:100],
                ),
            }

        # Goto circuit breaker: stop retrying after N consecutive failures
        if module_id == "browser.goto" and _goto_consecutive_fails >= _GOTO_MAX_FAILS:
            return {
                "ok": False,
                "error": (
                    "STOP: browser.goto has failed {} times consecutively. "
                    "Do NOT call browser.goto again. "
                    "Try browser.goto with a Google search URL instead: "
                    "https://www.google.com/search?q=YOUR+QUERY"
                ).format(_goto_consecutive_fails),
            }

        # On new browser.launch: close existing sessions to avoid
        # "Multiple browser sessions active" errors.
        if module_id == "browser.launch":
            _browser_launch_failed = False
            _browser_launch_error = ""
            _goto_consecutive_fails = 0
            if _browser_sessions:
                for sid in list(_browser_sessions.keys()):
                    try:
                        await handler["execute_module"](
                            module_id="browser.close",
                            params={},
                            context={"browser_session": sid},
                            browser_sessions=_browser_sessions,
                        )
                    except Exception:
                        pass
                with _browser_sessions_lock:
                    _browser_sessions.clear()

        validation = _validate_execute_module_args(handler, module_id, arguments.get("params", {}))
        if validation is not None:
            return validation

        # Sandbox: route dangerous categories to Docker container
        if _sandbox_mgr and _sandbox_mgr.needs_sandbox(module_id):
            return await _sandbox_mgr.execute(
                module_id, arguments.get("params", {}), arguments.get("context"),
            )
        result = await handler["execute_module"](
            module_id=module_id,
            params=arguments.get("params", {}),
            context=arguments.get("context"),
            browser_sessions=_browser_sessions,
        )

        # Track browser.launch failure for cascade breaker
        if module_id == "browser.launch" and isinstance(result, dict) and not _is_ok(result):
            _browser_launch_failed = True
            _browser_launch_error = str(result.get("error", "unknown error"))

        # Track goto failures for circuit breaker
        if module_id == "browser.goto":
            if isinstance(result, dict) and _is_ok(result):
                _goto_consecutive_fails = 0  # reset on success
            else:
                _goto_consecutive_fails += 1

        return result

    elif name == "validate_params":
        return handler["validate_params"](
            module_id=arguments.get("module_id", ""),
            params=arguments.get("params", {}),
        )

    elif name == "list_recipes":
        fn = handler.get("list_recipes")
        if not fn:
            return {"ok": False, "error": "Recipe support requires flyto-core >= 2.15.0"}
        return fn()

    elif name == "run_recipe":
        fn = handler.get("run_recipe")
        if not fn:
            return {"ok": False, "error": "Recipe support requires flyto-core >= 2.15.0"}
        return await fn(
            recipe_name=arguments.get("recipe_name", ""),
            args=arguments.get("args", {}),
            browser_sessions=_browser_sessions,
        )

    return {"ok": False, "error": "Unknown core tool: {}".format(name)}


def _validate_execute_module_args(
    handler: Dict[str, Any],
    module_id: str,
    params: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Validate module params before execution when flyto-core exposes validation."""
    validate = handler.get("validate_params")
    if not validate or not module_id:
        return None
    try:
        result = validate(module_id=module_id, params=params or {})
    except Exception as e:
        return {
            "ok": False,
            "error": "flyto-core validate_params failed before execute_module: {}".format(e),
            "module_id": module_id,
            "params_valid": False,
        }

    if isinstance(result, dict):
        valid = result.get("valid")
        ok = result.get("ok")
        errors = result.get("errors") or result.get("error")
        if valid is False or ok is False:
            return {
                "ok": False,
                "error": "Invalid params for {}: {}".format(module_id, errors or "schema validation failed"),
                "module_id": module_id,
                "params_valid": False,
                "validation": result,
            }
    return None
