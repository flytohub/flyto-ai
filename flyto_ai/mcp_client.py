# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""MCP client lifecycle state machine — managed connection to MCP servers.

Inspired by claw-code's ``McpServerManager`` + ``McpLifecycleValidator``
with phase tracking, reconnection, and graceful degradation.

Usage::

    manager = McpClientManager(
        server_cmd=["flyto-index", "mcp", "--stdio"],
        name="flyto-indexer",
    )
    await manager.connect()

    result = await manager.call_tool("search", {"query": "auth"})

    # If the server crashes, manager enters DEGRADED state
    # and attempts automatic reconnection.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class McpConnectionState(Enum):
    """Lifecycle phases for an MCP server connection."""
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"         # partial — some tools unavailable
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"


@dataclass
class McpToolInfo:
    """Metadata for a single MCP tool."""
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)


class McpClientManager:
    """Lifecycle-managed MCP client connection over stdio.

    Manages spawn → initialize → list_tools → call_tool, with automatic
    reconnection and degraded-mode fallback.
    """

    def __init__(
        self,
        server_cmd: List[str],
        name: str = "mcp-server",
        max_reconnect_attempts: int = 3,
        reconnect_delay_seconds: float = 1.0,
    ) -> None:
        self._cmd = server_cmd
        self._name = name
        self._max_reconnect = max_reconnect_attempts
        self._reconnect_delay = reconnect_delay_seconds

        self._state = McpConnectionState.DISCONNECTED
        self._process: Optional[asyncio.subprocess.Process] = None
        self._tools: List[McpToolInfo] = []
        self._request_id: int = 0
        self._reconnect_count: int = 0

    @property
    def state(self) -> McpConnectionState:
        return self._state

    @property
    def tools(self) -> List[McpToolInfo]:
        return list(self._tools)

    @property
    def is_available(self) -> bool:
        return self._state in (McpConnectionState.READY, McpConnectionState.DEGRADED)

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def connect(self) -> bool:
        """Spawn the MCP server process and initialize the connection.

        Returns True if connection succeeded.
        """
        self._state = McpConnectionState.INITIALIZING
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Send initialize request
            init_result = await self._send_request("initialize", {
                "protocolVersion": "2025-11-25",
                "clientInfo": {"name": "flyto-ai", "version": "0.12.0"},
                "capabilities": {},
            })

            if init_result is None:
                self._state = McpConnectionState.DISCONNECTED
                return False

            # Send initialized notification
            await self._send_notification("notifications/initialized")

            # List available tools
            tools_result = await self._send_request("tools/list", {})
            if tools_result and "tools" in tools_result:
                self._tools = [
                    McpToolInfo(
                        name=t["name"],
                        description=t.get("description", ""),
                        input_schema=t.get("inputSchema", {}),
                    )
                    for t in tools_result["tools"]
                ]
                logger.info("MCP '%s': connected with %d tools", self._name, len(self._tools))

            self._state = McpConnectionState.READY
            self._reconnect_count = 0
            return True

        except Exception as e:
            logger.warning("MCP '%s' connect failed: %s", self._name, e)
            self._state = McpConnectionState.DISCONNECTED
            return False

    async def reconnect(self) -> bool:
        """Attempt to reconnect to the MCP server."""
        if self._reconnect_count >= self._max_reconnect:
            logger.warning("MCP '%s': max reconnect attempts (%d) reached",
                           self._name, self._max_reconnect)
            self._state = McpConnectionState.DISCONNECTED
            return False

        self._state = McpConnectionState.RECONNECTING
        self._reconnect_count += 1
        await self.disconnect()

        await asyncio.sleep(self._reconnect_delay * self._reconnect_count)
        return await self.connect()

    async def disconnect(self) -> None:
        """Terminate the MCP server process."""
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass
            self._process = None
        self._state = McpConnectionState.DISCONNECTED
        self._tools = []

    async def call_tool(self, name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call an MCP tool. Attempts reconnection on failure."""
        if not self.is_available:
            return {"ok": False, "error": "MCP server '{}' is {}".format(
                self._name, self._state.value)}

        try:
            result = await self._send_request("tools/call", {
                "name": name,
                "arguments": arguments or {},
            })
            if result is None:
                # Server may have crashed
                self._state = McpConnectionState.DEGRADED
                if await self.reconnect():
                    # Retry once after reconnect
                    result = await self._send_request("tools/call", {
                        "name": name,
                        "arguments": arguments or {},
                    })
                if result is None:
                    return {"ok": False, "error": "MCP tool call failed after reconnect"}
            return result
        except Exception as e:
            self._state = McpConnectionState.DEGRADED
            return {"ok": False, "error": str(e)}

    async def _send_request(self, method: str, params: Dict) -> Optional[Dict]:
        """Send a JSON-RPC request and wait for matching response.

        Skips notifications and non-matching responses (by request ID).
        """
        if not self._process or not self._process.stdin or not self._process.stdout:
            return None

        req_id = self._next_id()
        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }
        line = json.dumps(request) + "\n"

        try:
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()

            # Read lines until we get a response matching our request ID
            deadline = asyncio.get_event_loop().time() + 30.0
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    logger.warning("MCP '%s' request timed out", self._name)
                    return None

                response_line = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=remaining,
                )
                if not response_line:
                    return None

                try:
                    response = json.loads(response_line)
                except json.JSONDecodeError:
                    continue  # skip non-JSON lines

                # Skip notifications (no "id" field)
                if "id" not in response:
                    continue

                # Skip responses for other request IDs
                if response.get("id") != req_id:
                    continue

                if "error" in response:
                    logger.warning("MCP '%s' error: %s", self._name, response["error"])
                    return None
                return response.get("result")

        except (asyncio.TimeoutError, OSError) as e:
            logger.warning("MCP '%s' request failed: %s", self._name, e)
            return None

    async def _send_notification(self, method: str, params: Optional[Dict] = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return
        notification = {"jsonrpc": "2.0", "method": method}
        if params:
            notification["params"] = params
        line = json.dumps(notification) + "\n"
        try:
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
        except OSError:
            pass
