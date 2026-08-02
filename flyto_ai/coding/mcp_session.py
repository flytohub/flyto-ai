# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""MCP negotiation and tool-call orchestration over atomic transport/catalog."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from flyto_ai.coding.contracts import CapabilitySpec
from flyto_ai.coding.mcp_catalog import (
    build_mcp_tool_catalog,
    catalog_tool_names,
    mcp_domain_status,
)
from flyto_ai.coding.mcp_transport import McpJsonRpcTransport
from flyto_ai.coding.store import redact_evidence


class McpStdioSession:
    """Negotiate one bounded MCP capability and dispatch its scoped tools."""

    def __init__(self, spec: CapabilitySpec, workspace: str) -> None:
        self.spec = spec
        self.workspace = workspace
        self.transport = McpJsonRpcTransport(spec, workspace)
        self.tools: List[Dict[str, Any]] = []
        self.remote_tool_names: List[str] = []
        self.observed_tool_names: Tuple[str, ...] = ()
        self.negotiated_protocol_version = ""
        self.server_name = ""
        self._tool_map: Dict[str, str] = {}
        self._started = False
        self._closed = False

    @property
    def process(self):
        """Preserve read-only access to the underlying process for compatibility."""
        return self.transport.process

    async def start(self) -> None:
        if self._started or self._closed:
            raise RuntimeError("capability session can only be started once")
        self._started = True
        await self.transport.start()
        try:
            initialized = await self.transport.request("initialize", {
                "protocolVersion": self.spec.protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "flyto-ai", "version": "1"},
            })
            negotiated = initialized.get("protocolVersion")
            if negotiated != self.spec.protocol_version:
                raise RuntimeError(
                    "capability negotiated unsupported MCP protocol version: {}".format(
                        str(negotiated)[:100],
                    )
                )
            self.negotiated_protocol_version = str(negotiated)
            server_info = initialized.get("serverInfo", {})
            if not isinstance(server_info, dict):
                raise RuntimeError("capability returned invalid serverInfo")
            server_name = server_info.get("name", "")
            if not isinstance(server_name, str):
                raise RuntimeError("capability returned invalid server name")
            self.server_name = server_name[:128]
            await self.transport.notify("notifications/initialized", {})
            listed = await self.transport.request("tools/list", {})
            raw_tools = listed.get("tools", []) if isinstance(listed, dict) else []
            self.remote_tool_names = list(catalog_tool_names(raw_tools))
            self.observed_tool_names = tuple(self.remote_tool_names)
            catalog = build_mcp_tool_catalog(self.spec, raw_tools)
        except Exception:
            await self.close()
            raise
        self.tools = [dict(definition) for definition in catalog.definitions]
        self._tool_map = dict(catalog.tool_map)
        self.remote_tool_names = list(catalog.remote_names)
        self.observed_tool_names = tuple(catalog.remote_names)

    async def dispatch(self, provider_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self._started or self._closed:
            return {"ok": False, "error": "capability session is not running"}
        if not isinstance(arguments, dict):
            return {"ok": False, "error": "capability arguments must be an object"}
        remote_name = self._tool_map.get(provider_name)
        if not remote_name:
            return {"ok": False, "error": "unknown capability tool"}
        result = await self.transport.request(
            "tools/call", {"name": remote_name, "arguments": arguments},
        )
        ok, error = mcp_domain_status(result)
        response = {
            "ok": ok,
            "capability": self.spec.name,
            "tool": remote_name,
            "result": redact_evidence(result),
        }
        if error:
            response["error"] = error
        return response

    def remote_tool_name(self, provider_name: str) -> Optional[str]:
        """Resolve a provider-safe name back to the negotiated MCP tool name."""
        return self._tool_map.get(provider_name)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.transport.close()
        self.tools = []
        self.remote_tool_names = []
        self._tool_map.clear()
