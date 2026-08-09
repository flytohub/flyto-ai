# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Authenticated loopback HTTP facade for :mod:`flyto_ai.coding.service`."""
from __future__ import annotations

import hashlib
import hmac
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Tuple
from urllib.parse import urlsplit

from flyto_ai.coding.contracts import (
    MAX_AUDIT_FINDINGS,
    CodingAuditFinding,
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


MAX_REQUEST_BYTES = 256 * 1024
_JOB_ID_RE = re.compile(r"^job_[a-f0-9]{24}$")
_AUDIT_FIELDS = frozenset({"implementation_revision_sha256", "verdict", "findings"})
#: Stale revisions, wrong state, exhausted rework, and a worktree owned by
#: another live job are conflicts; missing startup authority is a policy
#: denial. Unlisted codes stay 403.
_AUDIT_STATUS = {
    "workspace_busy": 409,
    "abandon_state_conflict": 409,
    "revision_mismatch": 409,
    "revision_unavailable": 409,
    "audit_state_conflict": 409,
    "rework_limit_reached": 409,
    "rework_not_resumable": 409,
    "session_binding_failed": 409,
    "idempotency_conflict": 409,
    "service_busy": 429,
}


class CodingHTTPServer(ThreadingHTTPServer):
    """One configured tenant and provider boundary; requests cannot override it."""

    daemon_threads = True

    def __init__(self, address: Tuple[str, int], service: CodingService, tenant_id: str, auth_token: str) -> None:
        if not auth_token or len(auth_token) < 16:
            raise ValueError("auth_token must contain at least 16 characters")
        self.coding_service = service
        self.tenant_id = tenant_id
        self.auth_token_sha256 = hashlib.sha256(auth_token.encode()).digest()
        super().__init__(address, CodingHTTPHandler)


class CodingHTTPHandler(BaseHTTPRequestHandler):
    """Serve health, submit, and status without logging request material."""

    server: CodingHTTPServer

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/healthz":
            self._json(200, {"ok": True, "service": "flyto-coding"})
            return
        if not self._authorized():
            self._json(401, {"ok": False, "error": "unauthorized"})
            return
        prefix = "/v1/coding/jobs/"
        if not path.startswith(prefix) or "/" in path[len(prefix):]:
            self._json(404, {"ok": False, "error": "not_found"})
            return
        try:
            receipt = self.server.coding_service.get(self.server.tenant_id, path[len(prefix):])
        except CodingJobNotFound:
            self._json(404, {"ok": False, "error": "job_not_found"})
            return
        self._json(200, {"ok": True, "job": receipt_to_mapping(receipt)})

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path != "/v1/coding/jobs":
            job_id = self._audit_target(path)
            if job_id is None:
                self._json(404, {"ok": False, "error": "not_found"})
            else:
                self._audit(job_id)
            return
        if not self._authorized():
            self._json(401, {"ok": False, "error": "unauthorized"})
            return
        idempotency_key = self.headers.get("Idempotency-Key", "")
        try:
            body = self._read_json()
            request = request_from_mapping(body)
            receipt = self.server.coding_service.submit(
                self.server.tenant_id, idempotency_key, request,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"ok": False, "error": "invalid_request", "message": str(exc)[:500]})
            return
        except CodingServiceError as exc:
            status = _AUDIT_STATUS.get(exc.code, 403)
            payload = {"ok": False, "error": exc.code}
            # Same bounded projection the MCP facade applies, so a caller sees
            # the owning job of a busy worktree over either transport.
            details = error_details(exc)
            if details:
                payload["details"] = details
            self._json(status, payload)
            return
        self._json(202, {"ok": True, "job": receipt_to_mapping(receipt)})

    @staticmethod
    def _audit_target(path: str) -> Any:
        """Return the job id for the audit route, or None when unmatched."""
        prefix, suffix = "/v1/coding/jobs/", "/audit"
        if not path.startswith(prefix) or not path.endswith(suffix):
            return None
        job_id = path[len(prefix):-len(suffix)]
        return job_id if _JOB_ID_RE.fullmatch(job_id) else None

    def _audit(self, job_id: str) -> None:
        """Forward one authenticated audit decision to the tenant-bound service.

        The transport validates shape only. Acceptance, revision binding, and
        rework limits remain the service's decision.
        """
        if not self._authorized():
            self._json(401, {"ok": False, "error": "unauthorized"})
            return
        try:
            body = self._read_json()
            unknown = set(body) - _AUDIT_FIELDS
            if unknown:
                # Backend selection and audit authority are startup-only.
                raise ValueError("unsupported audit fields")
            findings = body.get("findings")
            if not isinstance(findings, list) or len(findings) > MAX_AUDIT_FINDINGS:
                raise ValueError("findings must be a bounded array")
            verdict = body.get("verdict")
            if isinstance(verdict, bool) or not isinstance(verdict, str):
                raise ValueError("verdict must be a string")
            receipt = self.server.coding_service.audit(
                self.server.tenant_id,
                job_id,
                require_revision_sha256(
                    body.get("implementation_revision_sha256"),
                    "implementation_revision_sha256",
                ),
                CodingAuditVerdict(verdict),
                tuple(CodingAuditFinding.from_mapping(item) for item in findings),
            )
        except CodingJobNotFound:
            self._json(404, {"ok": False, "error": "job_not_found"})
            return
        except (ValueError, json.JSONDecodeError):
            # Never echo audit payload material back to the caller.
            self._json(400, {"ok": False, "error": "invalid_request"})
            return
        except CodingServiceError as exc:
            self._json(_AUDIT_STATUS.get(exc.code, 403), {"ok": False, "error": exc.code})
            return
        self._json(200, {"ok": True, "job": receipt_to_mapping(receipt)})

    def _authorized(self) -> bool:
        provided = self.headers.get("Authorization", "")
        if not provided.startswith("Bearer "):
            return False
        return hmac.compare_digest(
            hashlib.sha256(provided[7:].encode()).digest(),
            self.server.auth_token_sha256,
        )

    def _read_json(self) -> Dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if not 1 <= size <= MAX_REQUEST_BYTES:
            raise ValueError("request body size is invalid")
        raw = self.rfile.read(size)
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        return value

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)


def build_http_server(
    service: CodingService,
    *,
    tenant_id: str,
    auth_token: str,
    host: str = "127.0.0.1",
    port: int = 0,
) -> CodingHTTPServer:
    """Build a loopback-only server; public TLS belongs to Flyto2 Cloud."""

    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("coding HTTP server is loopback-only")
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    return CodingHTTPServer((host, port), service, tenant_id, auth_token)
