# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Authenticated loopback HTTP facade for :mod:`flyto_ai.coding.service`."""
from __future__ import annotations

import hashlib
import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Tuple
from urllib.parse import urlsplit

from flyto_ai.coding.service import (
    CodingJobNotFound,
    CodingService,
    CodingServiceError,
    request_from_mapping,
    receipt_to_mapping,
)


MAX_REQUEST_BYTES = 256 * 1024


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
        if urlsplit(self.path).path != "/v1/coding/jobs":
            self._json(404, {"ok": False, "error": "not_found"})
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
            status = 409 if exc.code == "idempotency_conflict" else 429 if exc.code == "service_busy" else 403
            self._json(status, {"ok": False, "error": exc.code})
            return
        self._json(202, {"ok": True, "job": receipt_to_mapping(receipt)})

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
