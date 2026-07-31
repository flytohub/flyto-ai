# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""MCP server — JSON-RPC 2.0 STDIO transport.

Exposes flyto-core's runtime-discovered registry modules plus a meta ``chat`` tool to Claude Desktop,
ChatGPT, VSCode, and other MCP-compatible hosts.

No external MCP library required — raw JSON-RPC over stdin/stdout,
identical pattern to flyto-core and flyto-indexer.
"""
import asyncio
import json
import logging
import sys
from typing import Any, Dict, Optional

from flyto_ai import __version__

logger = logging.getLogger(__name__)

# MCP 2026-07-28 is stateless and selected through metadata on every request.
# Older revisions retain their initialize handshake.
MODERN_PROTOCOL_VERSION = "2026-07-28"
LEGACY_PROTOCOL_VERSIONS = (
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
SUPPORTED_PROTOCOL_VERSIONS = (
    MODERN_PROTOCOL_VERSION,
    *LEGACY_PROTOCOL_VERSIONS,
)
PROTOCOL_VERSION_META_KEY = "io.modelcontextprotocol/protocolVersion"
CLIENT_CAPABILITIES_META_KEY = "io.modelcontextprotocol/clientCapabilities"
CLIENT_INFO_META_KEY = "io.modelcontextprotocol/clientInfo"
SERVER_INFO_META_KEY = "io.modelcontextprotocol/serverInfo"
DISCOVERY_TTL_MS = 60_000
STATIC_LIST_TTL_MS = 60_000


def _server_info() -> Dict[str, Any]:
    return {
        "name": "flyto-ai",
        "title": "Flyto2 AI Automation Agent",
        "version": __version__,
        "description": (
            "Turn natural-language requests into validated Flyto2 automation "
            "tool calls."
        ),
        "websiteUrl": "https://github.com/flytohub/flyto-ai",
    }


def _server_capabilities() -> Dict[str, Any]:
    return {"tools": {"listChanged": False}}

SERVER_CAPABILITIES = {
    "capabilities": _server_capabilities(),
    "serverInfo": _server_info(),
}


def negotiate_protocol_version(client_version: Optional[str]) -> str:
    """Select a supported version, preferring the latest revision."""
    if client_version and client_version in SUPPORTED_PROTOCOL_VERSIONS:
        return client_version
    return SUPPORTED_PROTOCOL_VERSIONS[0]


def negotiate_legacy_protocol_version(client_version: Optional[str]) -> str:
    """Select a handshake revision without crossing into stateless MCP."""
    if client_version and client_version in LEGACY_PROTOCOL_VERSIONS:
        return client_version
    return LEGACY_PROTOCOL_VERSIONS[0]


def build_initialize_response(client_version: Optional[str]) -> Dict:
    return {
        "protocolVersion": negotiate_legacy_protocol_version(client_version),
        **SERVER_CAPABILITIES,
    }

# Meta-tool: lets external AI (Claude Desktop) ask flyto-ai agent
# to execute a full workflow via natural language.
CHAT_TOOL = {
    "name": "chat",
    "description": (
        "Send a natural language message to the flyto-ai agent. "
        "The agent will plan and execute automation workflows using "
        "runtime-discovered registry modules "
        "(browser, file, image, API, database, etc.). "
        "Returns the agent's response, tool calls made, and execution results."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "What you want to automate (natural language).",
            },
            "mode": {
                "type": "string",
                "enum": ["execute", "yaml"],
                "description": "execute = run modules directly; yaml = only generate workflow YAML.",
                "default": "execute",
            },
        },
        "required": ["message"],
    },
}


def _make_error(
    req_id: Any,
    code: int,
    message: str,
    data: Optional[Dict[str, Any]] = None,
) -> Dict:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": error}


def build_modern_result(
    result: Dict[str, Any],
    *,
    server_info: Dict[str, Any],
    ttl_ms: Optional[int] = None,
    cache_scope: Optional[str] = None,
) -> Dict[str, Any]:
    """Add the fields required on successful MCP 2026-07-28 results."""
    modern_result = dict(result)
    modern_result.setdefault("resultType", "complete")
    metadata = modern_result.get("_meta")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata.setdefault(SERVER_INFO_META_KEY, server_info)
    modern_result["_meta"] = metadata
    if ttl_ms is not None and cache_scope is not None:
        modern_result["ttlMs"] = ttl_ms
        modern_result["cacheScope"] = cache_scope
    return modern_result


def request_protocol_era(
    req_id: Any,
    method: str,
    params: Any,
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Classify a request as modern or legacy and validate modern metadata."""
    if not isinstance(params, dict):
        if method == "server/discover":
            return None, _make_error(req_id, -32602, "params must be an object")
        return "legacy", None

    metadata = params.get("_meta")
    has_version = (
        isinstance(metadata, dict)
        and PROTOCOL_VERSION_META_KEY in metadata
    )
    if not has_version:
        if method == "server/discover":
            return None, _make_error(
                req_id,
                -32602,
                "Missing required request metadata: {}".format(
                    PROTOCOL_VERSION_META_KEY,
                ),
            )
        return "legacy", None

    requested = metadata.get(PROTOCOL_VERSION_META_KEY)
    if not isinstance(requested, str):
        return None, _make_error(
            req_id,
            -32602,
            "{} must be a string".format(PROTOCOL_VERSION_META_KEY),
        )
    if requested != MODERN_PROTOCOL_VERSION:
        return None, _make_error(
            req_id,
            -32022,
            "Unsupported protocol version",
            {
                "requested": requested,
                "supported": list(SUPPORTED_PROTOCOL_VERSIONS),
            },
        )
    if not isinstance(metadata.get(CLIENT_CAPABILITIES_META_KEY), dict):
        return None, _make_error(
            req_id,
            -32602,
            "Missing or invalid request metadata: {}".format(
                CLIENT_CAPABILITIES_META_KEY,
            ),
        )
    client_info = metadata.get(CLIENT_INFO_META_KEY)
    if client_info is not None and (
        not isinstance(client_info, dict)
        or not isinstance(client_info.get("name"), str)
        or not isinstance(client_info.get("version"), str)
    ):
        return None, _make_error(
            req_id,
            -32602,
            "Invalid request metadata: {}".format(CLIENT_INFO_META_KEY),
        )
    return "modern", None


def _make_result(
    req_id: Any,
    result: Dict[str, Any],
    *,
    modern: bool = False,
    ttl_ms: Optional[int] = None,
    cache_scope: Optional[str] = None,
) -> Dict:
    if modern:
        result = build_modern_result(
            result,
            server_info=_server_info(),
            ttl_ms=ttl_ms,
            cache_scope=cache_scope,
        )
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


class MCPServer:
    """Stateless MCP server — processes one JSON-RPC request at a time."""

    def __init__(self) -> None:
        self._agent = None
        self._registry = None

    def _ensure_agent(self):
        if self._agent is not None:
            return
        from flyto_ai import Agent, AgentConfig
        config = AgentConfig.from_env()
        self._agent = Agent(config=config)

    def _ensure_registry(self):
        if self._registry is not None:
            return
        self._ensure_agent()
        # Build tool list from agent's registered tools
        self._registry = {t["name"]: t for t in self._agent.tools}

    async def handle(self, request: Dict) -> Optional[Dict]:
        """Handle a single JSON-RPC request. Returns response dict or None for notifications."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})
        era, protocol_error = request_protocol_era(req_id, method, params)
        if protocol_error is not None:
            return protocol_error
        modern = era == "modern"

        if method == "initialize" and not modern:
            client_version = params.get("protocolVersion") if isinstance(params, dict) else None
            return _make_result(req_id, build_initialize_response(client_version))

        if method == "server/discover" and modern:
            return _make_result(
                req_id,
                {
                    "supportedVersions": list(SUPPORTED_PROTOCOL_VERSIONS),
                    "capabilities": _server_capabilities(),
                    "instructions": (
                        "Use chat for natural-language automation or call "
                        "runtime-discovered Flyto2 tools directly."
                    ),
                },
                modern=True,
                ttl_ms=DISCOVERY_TTL_MS,
                cache_scope="public",
            )

        if modern and method in {"initialize", "ping", "logging/setLevel"}:
            return _make_error(req_id, -32601, "Method not found: {}".format(method))

        if method.startswith("notifications/"):
            return None  # notification, no response

        if method == "ping":
            return _make_result(req_id, {})

        if method == "tools/list":
            self._ensure_registry()
            tools = list(self._registry.values()) + [CHAT_TOOL]
            return _make_result(
                req_id,
                {"tools": tools},
                modern=modern,
                ttl_ms=STATIC_LIST_TTL_MS,
                cache_scope="public",
            )

        if method == "tools/call":
            if not isinstance(params, dict):
                return _make_error(req_id, -32602, "params must be an object")
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            if not isinstance(tool_name, str) or not isinstance(arguments, dict):
                return _make_error(
                    req_id,
                    -32602,
                    "Tool name must be a string and arguments must be an object",
                )
            return await self._handle_tool_call(
                req_id,
                tool_name,
                arguments,
                modern=modern,
            )

        return _make_error(req_id, -32601, "Method not found: {}".format(method))

    async def _handle_tool_call(
        self,
        req_id: Any,
        name: str,
        arguments: Dict,
        *,
        modern: bool,
    ) -> Dict:
        self._ensure_agent()

        # Meta-tool: chat
        if name == "chat":
            message = arguments.get("message", "")
            mode = arguments.get("mode", "execute")
            if not message:
                return _make_error(req_id, -32602, "message is required")

            result = await self._agent.chat(message, mode=mode)
            content = result.message
            if result.execution_results:
                executed = [er.get("module_id", "") for er in result.execution_results]
                content += "\n\nExecuted modules: {}".format(", ".join(executed))

            return _make_result(
                req_id,
                {"content": [{"type": "text", "text": content}]},
                modern=modern,
            )

        # Regular tool dispatch
        self._ensure_registry()
        dispatch = self._agent.dispatch_fn
        if name not in self._registry and dispatch:
            result = await dispatch(name, arguments)
            text = json.dumps(result, ensure_ascii=False, default=str)
            return _make_result(
                req_id,
                {"content": [{"type": "text", "text": text}]},
                modern=modern,
            )

        if not dispatch:
            return _make_error(req_id, -32602, "No tools available")

        result = await dispatch(name, arguments)
        text = json.dumps(result, ensure_ascii=False, default=str)
        return _make_result(
            req_id,
            {"content": [{"type": "text", "text": text}]},
            modern=modern,
        )


async def async_main():
    """STDIO MCP server loop — read JSON-RPC from stdin, write to stdout."""
    server = MCPServer()

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    # Use stdout for JSON-RPC responses
    write_transport, write_protocol = await asyncio.get_event_loop().connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout,
    )
    writer = asyncio.StreamWriter(write_transport, write_protocol, reader, asyncio.get_event_loop())

    while True:
        line = await reader.readline()
        if not line:
            break  # EOF

        line_str = line.decode("utf-8", errors="replace").strip()
        if not line_str:
            continue

        try:
            request = json.loads(line_str)
        except json.JSONDecodeError:
            resp = _make_error(None, -32700, "Parse error")
            writer.write((json.dumps(resp) + "\n").encode("utf-8"))
            await writer.drain()
            continue

        try:
            response = await server.handle(request)
        except Exception as e:
            logger.error("MCP handler error: %s", e)
            response = _make_error(request.get("id"), -32603, str(e))

        if response is not None:
            writer.write((json.dumps(response, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
            await writer.drain()


def main():
    """Entry point for ``flyto-ai-mcp`` and ``flyto-ai mcp``."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
