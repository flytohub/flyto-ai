# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""MCP server — JSON-RPC 2.0 STDIO transport.

Exposes flyto-ai's 412 modules + a meta ``chat`` tool to Claude Desktop,
ChatGPT, VSCode, and other MCP-compatible hosts.

No external MCP library required — raw JSON-RPC over stdin/stdout,
identical pattern to flyto-core and flyto-indexer.
"""
import asyncio
import json
import logging
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SERVER_INFO = {
    "protocolVersion": "2024-11-05",
    "capabilities": {"tools": {"listChanged": False}},
    "serverInfo": {
        "name": "flyto-ai",
        "version": "0.9.3",
    },
}

# Meta-tool: lets external AI (Claude Desktop) ask flyto-ai agent
# to execute a full workflow via natural language.
CHAT_TOOL = {
    "name": "chat",
    "description": (
        "Send a natural language message to the flyto-ai agent. "
        "The agent will plan and execute automation workflows using 412 modules "
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


def _make_error(req_id: Any, code: int, message: str) -> Dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _make_result(req_id: Any, result: Any) -> Dict:
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

        if method == "initialize":
            return _make_result(req_id, SERVER_INFO)

        elif method == "notifications/initialized":
            return None  # notification, no response

        elif method == "ping":
            return _make_result(req_id, {})

        elif method == "tools/list":
            self._ensure_registry()
            tools = list(self._registry.values()) + [CHAT_TOOL]
            return _make_result(req_id, {"tools": tools})

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            return await self._handle_tool_call(req_id, tool_name, arguments)

        else:
            return _make_error(req_id, -32601, "Method not found: {}".format(method))

    async def _handle_tool_call(self, req_id: Any, name: str, arguments: Dict) -> Dict:
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

            return _make_result(req_id, {
                "content": [{"type": "text", "text": content}],
            })

        # Regular tool dispatch
        self._ensure_registry()
        dispatch = self._agent.dispatch_fn
        if name not in self._registry and dispatch:
            result = await dispatch(name, arguments)
            text = json.dumps(result, ensure_ascii=False, default=str)
            return _make_result(req_id, {
                "content": [{"type": "text", "text": text}],
            })

        if not dispatch:
            return _make_error(req_id, -32602, "No tools available")

        result = await dispatch(name, arguments)
        text = json.dumps(result, ensure_ascii=False, default=str)
        return _make_result(req_id, {
            "content": [{"type": "text", "text": text}],
        })


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
