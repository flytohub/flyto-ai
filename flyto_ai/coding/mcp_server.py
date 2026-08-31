# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Detachable JSON-RPC/MCP stdio facade for the Flyto2 coding service."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
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
    CodingService,
    CodingServiceError,
    error_details,
    request_from_mapping,
    receipt_to_mapping,
)


MCP_PROTOCOL_VERSION = "2025-06-18"
#: Advanced from "2" when conditional bounded reads and typed next actions
#: joined ``flyto_coding_get``. The protocol version and public job receipt
#: remain compatible; an observation sibling appears only when a caller opts
#: into an explicit detail or conditional-read argument.
CODING_MCP_SERVER_VERSION = "3"
MAX_INSTRUCTIONS_CHARS = 512
#: Self-contained description of the host-owned loop. It states what the
#: caller must do and what this server will not do; it deliberately does not
#: claim the server can prove which principal is auditing.
CODING_MCP_INSTRUCTIONS = (
    "flyto_coding_submit starts work. flyto_coding_get detail returns "
    "observation.next_action/change_token; to wait, use detail=summary, resend "
    "the token, wait_ms<=20000. At awaiting_codex_audit independently inspect/test, "
    "then use flyto_coding_audit on exact "
    "implementation_revision_sha256. Rework keeps the same job/session; "
    "repeat. Only accept is landable; failed is terminal/non-landable. "
    "Verdicts are host-authenticated; server cannot prove caller identity. "
    "Never stages, commits, pushes, publishes, or deploys."
)
MAX_MESSAGE_BYTES = 256 * 1024
#: A long-poll stays ten seconds below the supervisor's 30-second response
#: deadline. That reserve covers the initial/final tenant-bound reads and JSON
#: serialization on a contended host; it is an invariant in the supervisor
#: regression suite, not an operator-tunable escape hatch.
MAX_GET_WAIT_MS = 20_000
GET_WAIT_POLL_INTERVAL_SECONDS = 0.25
GET_WAIT_RETRY_AFTER_MS = 250
MAX_GET_PROGRESS_AGE_MS = 2_147_483_647
_monotonic = time.monotonic
_sleep = time.sleep
_wall_time = time.time
#: Only the public, already-redacted receipt is hashed. The domain prefix keeps
#: this cursor distinct from implementation revisions, route receipts, config
#: digests, evidence digests, and every other 64-character value in the route.
_GET_CHANGE_TOKEN_DOMAIN = b"flyto.coding-get-change.v1\n"
_JOB_ID_PATTERN = "^job_[a-f0-9]{24}$"
_SHA256_PATTERN = "^[a-f0-9]{64}$"
_AUDIT_CODE_PATTERN = "^[A-Za-z][A-Za-z0-9_.-]{1,63}$"
_JOB_ID_RE = re.compile(_JOB_ID_PATTERN)
_SHA256_RE = re.compile(_SHA256_PATTERN)
_GET_ARGUMENT_FIELDS = frozenset({
    "job_id", "detail", "after_change_token", "wait_ms",
})
_GET_DETAILS = ("summary", "full")
#: The compact projection is intentionally an allowlist, not "the full receipt
#: minus large-looking fields". New receipt fields therefore stay out until a
#: review decides they are safe and useful for polling.
_GET_SUMMARY_FIELDS = (
    "service_contract_version",
    "job_id",
    "state",
    "submitted_at",
    "updated_at",
    "job_terminal",
    "landable",
    "implementer_started",
    "implementation_revision_sha256",
    "audit_count",
    "rework_count",
    "implementation_blockers",
    "verification_blockers",
    "failure_code",
    "failure_phase",
    "retryable",
    "required_actions",
)
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


def _tool_get_wait_ms(arguments: Mapping[str, Any]) -> int:
    """Decode the optional get wait without accepting booleans or coercion."""

    value = arguments.get("wait_ms", 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("wait_ms must be an integer")
    if not 0 <= value <= MAX_GET_WAIT_MS:
        raise ValueError("wait_ms must be between 0 and {}".format(MAX_GET_WAIT_MS))
    return value


def _tool_change_token(arguments: Mapping[str, Any]) -> str:
    """Decode one optional lower-case SHA-256 conditional-read cursor."""

    if "after_change_token" not in arguments:
        return ""
    value = _tool_string(arguments, "after_change_token")
    if not _SHA256_RE.fullmatch(value):
        raise ValueError("after_change_token must be a lower-case SHA-256 token")
    return value


def _tool_get_detail(arguments: Mapping[str, Any]) -> str:
    """Decode the optional compact/full projection choice without coercion."""

    if "detail" not in arguments:
        return "full"
    value = _tool_string(arguments, "detail")
    if value not in _GET_DETAILS:
        raise ValueError("detail must be summary or full")
    return value


def _get_job_projection(receipt: Any, detail: str) -> Dict[str, Any]:
    """Keep the historical full receipt or return the fixed polling subset."""

    if detail == "full":
        return receipt_to_mapping(receipt)
    phase, retryable, actions = receipt.failure_semantics
    # Do not construct and redact the full nested receipt merely to discard it:
    # an audit-ready result can approach the transport ceiling. Every selected
    # scalar/list comes from the already-validated receipt contract, and this
    # explicit mapping makes adding a receipt field fail closed by default.
    summary: Dict[str, Any] = {
        "service_contract_version": receipt.service_contract_version,
        "job_id": receipt.job_id,
        "state": receipt.state.value,
        "submitted_at": receipt.submitted_at,
        "updated_at": receipt.updated_at,
        "job_terminal": receipt.job_terminal,
        "landable": receipt.landable,
        "implementer_started": receipt.implementer_started,
        "implementation_revision_sha256": receipt.implementation_revision_sha256,
        "audit_count": receipt.audit_count,
        "rework_count": receipt.rework_count,
        "implementation_blockers": list(receipt.implementation_blockers),
        "verification_blockers": list(receipt.verification_blockers),
        "failure_code": receipt.failure_code,
        "failure_phase": phase,
        "retryable": retryable,
        "required_actions": list(actions),
    }
    return {field: summary[field] for field in _GET_SUMMARY_FIELDS}


def _receipt_change_token(job: Mapping[str, Any], detail: str) -> str:
    """Bind a cursor to exactly one secret-redacted public job projection."""

    encoded = json.dumps(
        job,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_GET_CHANGE_TOKEN_DOMAIN)
    digest.update(detail.encode("ascii") + b"\n")
    digest.update(encoded)
    return digest.hexdigest()


def _get_next_action(receipt: Any) -> str:
    """Project one closed action token from the existing receipt state."""

    state = receipt.state
    if state.value in {"queued", "running", "rework_queued", "rework_running"}:
        return "wait"
    if state.value == "awaiting_codex_audit":
        return "audit_revision"
    if state.value == "rework_route_blocked":
        return "retry_rework_route"
    if state.value == "codex_accepted":
        return "land_accepted_revision"
    _phase, retryable, required_actions = receipt.failure_semantics
    if retryable:
        return "retry_same_request"
    if required_actions:
        return "resolve_required_actions"
    return "stop_non_landable"


def _get_observation(
    receipt: Any,
    job: Mapping[str, Any],
    *,
    detail: str,
    after_change_token: str,
    waited_ms: int,
    timed_out: bool,
) -> Dict[str, Any]:
    """Build the fixed, path-free conditional-read metadata projection."""

    next_action = _get_next_action(receipt)
    change_token = _receipt_change_token(job, detail)
    return {
        "detail": detail,
        "change_token": change_token,
        "changed": not after_change_token or change_token != after_change_token,
        "timed_out": timed_out,
        "waited_ms": waited_ms,
        "retry_after_ms": GET_WAIT_RETRY_AFTER_MS if timed_out else 0,
        "recommended_wait_ms": MAX_GET_WAIT_MS if next_action == "wait" else 0,
        "progress_age_ms": min(
            MAX_GET_PROGRESS_AGE_MS,
            max(0, int((_wall_time() - receipt.updated_at) * 1000)),
        ),
        "next_action": next_action,
    }


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
            return self._call_get(arguments)
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

    def _call_get(self, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        """Read once or conditionally wait for one tenant-bound job change.

        A call containing only ``job_id`` remains the exact historical full
        response. Callers that explicitly choose ``summary`` use the service's
        lock-free durable peek, avoiding the shared coordination guard and
        omitting the large result,
        route, mission, and evidence projections while retaining every field a
        polling state machine needs. A full read remains mandatory before an
        independent audit.
        """

        _reject_unknown_arguments(arguments, _GET_ARGUMENT_FIELDS)
        job_id = _tool_job_id(arguments)
        detail = _tool_get_detail(arguments)
        wait_ms = _tool_get_wait_ms(arguments)
        after_change_token = _tool_change_token(arguments)
        if wait_ms and not after_change_token:
            raise ValueError("after_change_token is required when wait_ms is positive")
        if wait_ms and detail != "summary":
            raise ValueError("positive wait_ms requires detail=summary")

        get_receipt = self.service.get
        if detail == "summary":
            summary_get = getattr(self.service, "get_summary", None)
            if callable(summary_get):
                get_receipt = summary_get

        receipt = get_receipt(self.tenant_id, job_id)
        job = _get_job_projection(receipt, detail)
        if set(arguments) == {"job_id"}:
            payload = {"ok": True, "job": job}
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(payload, ensure_ascii=False),
                }],
                "isError": False,
                "structuredContent": payload,
            }
        change_token = _receipt_change_token(job, detail)
        waited_ms = 0
        timed_out = False
        wait_started = _monotonic()

        # Only background-owned states can make progress without a caller
        # action. Audit-ready, blocked, accepted, and terminal receipts return
        # immediately even when a stale client asks to wait, preventing a
        # pointless 20-second delay exactly when the caller should act.
        if (
            wait_ms
            and change_token == after_change_token
            and _get_next_action(receipt) == "wait"
        ):
            deadline = wait_started + wait_ms / 1000.0
            while change_token == after_change_token:
                remaining = deadline - _monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                _sleep(min(GET_WAIT_POLL_INTERVAL_SECONDS, remaining))
                receipt = get_receipt(self.tenant_id, job_id)
                job = _get_job_projection(receipt, detail)
                change_token = _receipt_change_token(job, detail)
            waited_ms = min(
                wait_ms,
                max(0, int((_monotonic() - wait_started) * 1000)),
            )

        observation = _get_observation(
            receipt,
            job,
            detail=detail,
            after_change_token=after_change_token,
            waited_ms=waited_ms,
            timed_out=timed_out,
        )
        payload = {"ok": True, "job": job, "observation": observation}
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
                                "repository_roots": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 1,
                                    "uniqueItems": True,
                                    "items": {
                                        "type": "string", "minLength": 1, "maxLength": 4096,
                                    },
                                },
                                "owner_ref": {
                                    "type": "string",
                                    "pattern": r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$",
                                },
                                "thread_id": {"type": "string", "minLength": 1, "maxLength": 64},
                                "resume": {"type": "boolean"},
                                "retry_rework_route": {"type": "boolean"},
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
                "description": (
                    "Read a tenant-owned coding job. Default detail=full preserves the "
                    "historical receipt for audit. detail=summary is the compact polling "
                    "projection. Reuse observation.change_token as after_change_token with "
                    "a wait_ms up to 20000 to wait only while background work can progress."
                ),
                "inputSchema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["job_id"],
                    "properties": {
                        "job_id": {"type": "string", "pattern": _JOB_ID_PATTERN},
                        "detail": {"type": "string", "enum": list(_GET_DETAILS)},
                        "after_change_token": {
                            "type": "string", "pattern": _SHA256_PATTERN,
                        },
                        "wait_ms": {
                            "type": "integer", "minimum": 0, "maximum": MAX_GET_WAIT_MS,
                        },
                    },
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
