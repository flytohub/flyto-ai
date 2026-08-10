# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Detachable JSON-RPC/MCP stdio facade for the Flyto2 coding service."""
from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, Mapping, Optional

from flyto_ai.coding.contracts import (
    MAX_AUDIT_EVIDENCE_REF_CHARS,
    MAX_AUDIT_FINDINGS,
    MAX_AUDIT_MESSAGE_CHARS,
    MISSION_ID_PATTERN,
    MISSION_LANES,
    MISSION_MAX_ACCEPTANCE_CRITERIA,
    MISSION_MAX_DEPENDENCIES,
    MISSION_MAX_FIELD_CHARS,
    MISSION_MAX_PRIORITY,
    MISSION_MAX_TEXT_CHARS,
    WORK_ITEM_ID_PATTERN,
    CodingAuditFinding,
    CodingAuditSeverity,
    CodingAuditVerdict,
    require_revision_sha256,
)
from flyto_ai.coding.service import (
    CodingJobNotFound,
    CodingService,
    CodingServiceError,
    error_details,
    request_from_mapping,
    receipt_to_mapping,
)


MCP_PROTOCOL_VERSION = "2025-06-18"
#: Advanced from "1" when the audit tool and audit states joined this surface.
#: The protocol version and every existing tool schema stay compatible.
CODING_MCP_SERVER_VERSION = "2"
MAX_INSTRUCTIONS_CHARS = 512
#: Self-contained description of the host-owned loop. It states what the
#: caller must do and what this server will not do; it deliberately does not
#: claim the server can prove which principal is auditing.
CODING_MCP_INSTRUCTIONS = (
    "Coding: use flyto_coding_submit; poll flyto_coding_get to "
    "awaiting_codex_audit or terminal. failed is terminal/non-landable. At "
    "awaiting_codex_audit independently inspect/test the workspace, then "
    "audit exact implementation_revision_sha256 with flyto_coding_audit. "
    "rework sends typed findings to the same job/session; poll and re-audit. "
    "Only accept is landable. The verdict is from the host-authenticated "
    "auditor; server cannot prove caller identity. Never stages, commits, "
    "pushes, publishes, or deploys."
)
MAX_MESSAGE_BYTES = 256 * 1024
_JOB_ID_PATTERN = "^job_[a-f0-9]{24}$"
_SHA256_PATTERN = "^[a-f0-9]{64}$"
_AUDIT_CODE_PATTERN = "^[A-Za-z][A-Za-z0-9_.-]{1,63}$"
_JOB_ID_RE = re.compile(_JOB_ID_PATTERN)
_AUDIT_ARGUMENT_FIELDS = frozenset({
    "job_id", "implementation_revision_sha256", "verdict", "findings",
})
#: The optional mission envelope, declared strictly. Bounds and vocabularies are
#: read from the contract layer, which reads them from the generic mission
#: kernel, so a schema this facade publishes cannot drift from what the decoder
#: will accept. `additionalProperties: false` at both levels; the decoder
#: enforces the same closed field sets again, because a host may not validate.
_MISSION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scope", "objective", "desired_result", "acceptance_criteria"],
    "properties": {
        "scope": {
            "type": "string", "minLength": 1, "maxLength": MISSION_MAX_FIELD_CHARS,
        },
        "objective": {
            "type": "string", "minLength": 1, "maxLength": MISSION_MAX_TEXT_CHARS,
        },
        "desired_result": {
            "type": "string", "minLength": 1, "maxLength": MISSION_MAX_TEXT_CHARS,
        },
        "acceptance_criteria": {
            "type": "array",
            "minItems": 1,
            "maxItems": MISSION_MAX_ACCEPTANCE_CRITERIA,
            # The decoder is stricter still - it refuses two criteria that share
            # an id even when their statements differ - but a schema that stayed
            # silent about duplicates would advertise a payload the decoder
            # rejects.
            "uniqueItems": True,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "statement"],
                "properties": {
                    "id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MISSION_MAX_FIELD_CHARS,
                    },
                    "statement": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MISSION_MAX_TEXT_CHARS,
                    },
                },
            },
        },
        "priority": {"type": "integer", "minimum": 0, "maximum": MISSION_MAX_PRIORITY},
        "lane": {"type": "string", "enum": list(MISSION_LANES)},
        "mission_id": {"type": "string", "pattern": MISSION_ID_PATTERN},
        "parent_id": {"type": "string", "pattern": WORK_ITEM_ID_PATTERN},
        "return_to_id": {"type": "string", "pattern": WORK_ITEM_ID_PATTERN},
        "depends_on_ids": {
            "type": "array",
            "maxItems": MISSION_MAX_DEPENDENCIES,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": WORK_ITEM_ID_PATTERN},
        },
    },
}


def _tool_string(arguments: Mapping[str, Any], field_name: str) -> str:
    """Read one tool argument as a string without truthiness or str() coercion."""
    value = arguments.get(field_name)
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError("{} must be a string".format(field_name))
    return value


def _tool_job_id(arguments: Mapping[str, Any]) -> str:
    """Enforce the declared job id pattern at runtime, not only in the schema."""
    job_id = _tool_string(arguments, "job_id")
    if not _JOB_ID_RE.fullmatch(job_id):
        raise ValueError("job_id must match the published job identifier pattern")
    return job_id


def _reject_unknown_arguments(arguments: Mapping[str, Any], allowed: frozenset) -> None:
    """Apply `additionalProperties: false` at runtime; hosts may not validate."""
    unknown = set(arguments) - allowed
    if unknown:
        raise ValueError("unsupported tool arguments: {}".format(
            ", ".join(sorted(str(name) for name in unknown)),
        ))


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
                    "serverInfo": {
                        "name": "flyto-coding", "version": CODING_MCP_SERVER_VERSION,
                    },
                    "instructions": CODING_MCP_INSTRUCTIONS,
                })
            if method == "tools/list":
                return self._result(request_id, {"tools": self._tools()})
            if method == "tools/call":
                return self._result(request_id, self._call(params))
            return self._error(request_id, -32601, "method not found")
        except (ValueError, CodingServiceError) as exc:
            code = exc.code if isinstance(exc, CodingServiceError) else "invalid_request"
            payload: Dict[str, Any] = {"ok": False, "error": code}
            # A subclass may attach bounded, closed-schema context. This stays
            # additive: `ok` and `error` keep their exact shape, and a caller
            # that ignores `details` behaves exactly as it did before.
            details = error_details(exc)
            if details:
                payload["details"] = details
            return self._result(request_id, {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "isError": True,
                "structuredContent": payload,
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
        elif name == "flyto_coding_audit":
            _reject_unknown_arguments(arguments, _AUDIT_ARGUMENT_FIELDS)
            findings = arguments.get("findings")
            if not isinstance(findings, list):
                raise ValueError("findings must be an array")
            if len(findings) > MAX_AUDIT_FINDINGS:
                raise ValueError("findings cannot exceed {} items".format(MAX_AUDIT_FINDINGS))
            # Transport validates shape only. Whether this revision is
            # acceptable, and whether the verdict and findings are coherent,
            # is decided by the authenticated tenant-bound service.
            receipt = self.service.audit(
                self.tenant_id,
                _tool_job_id(arguments),
                require_revision_sha256(
                    arguments.get("implementation_revision_sha256"),
                    "implementation_revision_sha256",
                ),
                CodingAuditVerdict(_tool_string(arguments, "verdict")),
                tuple(CodingAuditFinding.from_mapping(item) for item in findings),
            )
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
                                # Optional and additive. Naming a mission does
                                # not add a tool, a lane, or an authority: the
                                # inventory stays exactly submit/get/audit.
                                "mission": _MISSION_SCHEMA,
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
                    "properties": {"job_id": {"type": "string", "pattern": _JOB_ID_PATTERN}},
                },
            },
            {
                "name": "flyto_coding_audit",
                "description": (
                    "Record the authenticated auditor's verdict on one exact implementation "
                    "revision. accept marks the job landable; rework must list findings and "
                    "continues the same job and thread. The implementer cannot call this."
                ),
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "job_id", "implementation_revision_sha256", "verdict", "findings",
                    ],
                    "properties": {
                        "job_id": {"type": "string", "pattern": _JOB_ID_PATTERN},
                        "implementation_revision_sha256": {
                            "type": "string", "pattern": _SHA256_PATTERN,
                        },
                        "verdict": {
                            "type": "string",
                            "enum": [item.value for item in CodingAuditVerdict],
                        },
                        "findings": {
                            "type": "array",
                            "maxItems": MAX_AUDIT_FINDINGS,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["code", "severity", "message"],
                                "properties": {
                                    "code": {"type": "string", "pattern": _AUDIT_CODE_PATTERN},
                                    "severity": {
                                        "type": "string",
                                        "enum": [item.value for item in CodingAuditSeverity],
                                    },
                                    "message": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": MAX_AUDIT_MESSAGE_CHARS,
                                    },
                                    "evidence_ref": {
                                        "type": "string",
                                        "maxLength": MAX_AUDIT_EVIDENCE_REF_CHARS,
                                    },
                                },
                            },
                        },
                    },
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
