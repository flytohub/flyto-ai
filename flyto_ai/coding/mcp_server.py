# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Detachable JSON-RPC/MCP stdio facade for the Flyto2 coding service."""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, Mapping, Optional

from flyto_ai.coding.service import (
    CodingJobNotFound,
    CodingService,
    CodingServiceError,
    request_from_mapping,
    receipt_to_mapping,
)


MCP_PROTOCOL_VERSION = "2025-06-18"
MAX_MESSAGE_BYTES = 256 * 1024


class CodingMCPServer:
    """Expose submit/status tools for one startup-configured tenant."""

    def __init__(self, service: CodingService, tenant_id: str) -> None:
        self.service = service
        self.tenant_id = tenant_id

    def handle(self, request: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        request_id = request.get("id")
        method = request.get("method")
        if request_id is None:
            return None
        if request.get("jsonrpc") != "2.0" or not isinstance(method, str):
            return self._error(request_id, -32600, "invalid request")
        params = request.get("params", {})
        if not isinstance(params, Mapping):
            return self._error(request_id, -32602, "params must be an object")
        try:
            if method == "initialize":
                requested = str(params.get("protocolVersion", ""))
                if requested != MCP_PROTOCOL_VERSION:
                    return self._error(request_id, -32602, "unsupported MCP protocol version")
                return self._result(request_id, {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "flyto-coding", "version": "1"},
                })
            if method == "tools/list":
                return self._result(request_id, {"tools": self._tools()})
            if method == "tools/call":
                return self._result(request_id, self._call(params))
            return self._error(request_id, -32601, "method not found")
        except (ValueError, CodingServiceError) as exc:
            code = exc.code if isinstance(exc, CodingServiceError) else "invalid_request"
            return self._result(request_id, {
                "content": [{"type": "text", "text": json.dumps({"ok": False, "error": code})}],
                "isError": True,
                "structuredContent": {"ok": False, "error": code},
            })

    def _call(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise ValueError("tool arguments must be an object")
        if name == "flyto_coding_submit":
            request_value = arguments.get("request")
            if not isinstance(request_value, Mapping):
                raise ValueError("request must be an object")
            receipt = self.service.submit(
                self.tenant_id,
                str(arguments.get("idempotency_key", "")),
                request_from_mapping(request_value),
            )
        elif name == "flyto_coding_get":
            try:
                receipt = self.service.get(self.tenant_id, str(arguments.get("job_id", "")))
            except CodingJobNotFound:
                raise
        else:
            raise ValueError("unknown coding tool")
        payload = {"ok": True, "job": receipt_to_mapping(receipt)}
        return {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "isError": False,
            "structuredContent": payload,
        }

    @staticmethod
    def _tools() -> list[Dict[str, Any]]:
        return [
            {
                "name": "flyto_coding_submit",
                "description": "Submit a provider-neutral coding job to the configured Flyto2 tenant.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["idempotency_key", "request"],
                    "properties": {
                        "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 128},
                        "request": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["message", "working_dir"],
                            "properties": {
                                "message": {"type": "string", "minLength": 1, "maxLength": 200000},
                                "working_dir": {"type": "string", "minLength": 1, "maxLength": 4096},
                                "thread_id": {"type": "string", "minLength": 1, "maxLength": 64},
                                "resume": {"type": "boolean"},
                                "max_attempts": {"type": "integer", "minimum": 1, "maximum": 5},
                                "max_rounds": {"type": "integer", "minimum": 1, "maximum": 100},
                                "require_changes": {"type": "boolean"},
                            },
                        },
                    },
                },
            },
            {
                "name": "flyto_coding_get",
                "description": "Read a coding job owned by the configured Flyto2 tenant.",
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["job_id"],
                    "properties": {"job_id": {"type": "string", "pattern": "^job_[a-f0-9]{24}$"}},
                },
            },
        ]

    @staticmethod
    def _result(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def serve_stdio(server: CodingMCPServer) -> None:
    """Process bounded newline-delimited MCP messages until stdin closes."""

    for raw in sys.stdin.buffer:
        if len(raw) > MAX_MESSAGE_BYTES:
            response = CodingMCPServer._error(None, -32600, "request exceeds message limit")
        else:
            try:
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError
                response = server.handle(value)
            except (UnicodeError, json.JSONDecodeError, ValueError):
                response = CodingMCPServer._error(None, -32700, "parse error")
        if response is not None:
            encoded = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
            sys.stdout.buffer.write(encoded)
            sys.stdout.buffer.flush()
