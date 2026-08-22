# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Loopback HTTP boundary for the Flyto2 structured robotics planner."""

from __future__ import annotations

import argparse
import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Awaitable, Callable, Mapping

from flyto_ai.capability_router import (
    CapabilityRoutingError,
    prepare_planner_request,
)

from flyto_ai.providers.ollama import (
    OllamaProvider,
    OllamaStructuredOutputError,
)
from flyto_ai.robotics_planning import (
    MAX_REQUEST_BYTES,
    RoboticsPlanningError,
    RoboticsPlanningService,
)


PlannerCallable = Callable[[object], Any]
PlannerPreparer = Callable[..., Awaitable[dict[str, Any]]]


class GovernedRoboticsPlanner:
    """Apply Flyto2 discovery before any robotics provider can see a request.

    The Robotics caller supplies its local, executable capability catalog.  AI
    narrows that catalog through the trusted Blueprint/Core discovery bridges;
    it never widens the caller's execution authority.  Keeping this composition
    in the host adapter means a production entry point cannot accidentally call
    the model-facing service with an unprepared request.
    """

    def __init__(
        self,
        service: RoboticsPlanningService,
        *,
        prepare: PlannerPreparer = prepare_planner_request,
    ) -> None:
        self._service = service
        self._prepare = prepare

    async def plan(self, raw_request: object) -> dict[str, Any]:
        if not isinstance(raw_request, Mapping):
            raise RoboticsPlanningError("planner request must be an object")
        try:
            prepared = await self._prepare(
                raw_request,
                require_goal_frame=True,
                require_discovery=True,
            )
        except CapabilityRoutingError as exc:
            raise RoboticsPlanningError(
                f"capability routing rejected the planner request: {exc}"
            ) from exc
        return await self._service.plan(prepared)


class RoboticsPlannerHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying one injected asynchronous planner function."""

    def __init__(
        self,
        server_address: tuple[str, int],
        planner: PlannerCallable,
    ) -> None:
        super().__init__(server_address, RoboticsPlannerHandler)
        self.planner = planner


class RoboticsPlannerHandler(BaseHTTPRequestHandler):
    """Serve health and one versioned planning endpoint without request logs."""

    server: RoboticsPlannerHTTPServer

    def log_message(self, format: str, *args: object) -> None:
        """Avoid leaking natural-language mission goals into access logs."""

    def _json(self, status: int, payload: object) -> None:
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(404, {"error": "not_found"})
            return
        self._json(
            200,
            {
                "status": "ok",
                "service": "flyto-ai-robotics-planner",
                "planning_mode": "live_llm",
            },
        )

    def do_POST(self) -> None:
        if self.path != "/v1/robotics/plan":
            self._json(404, {"error": "not_found"})
            return
        length_value = self.headers.get("Content-Length", "")
        try:
            length = int(length_value)
        except ValueError:
            self._json(400, {"error": "invalid_content_length"})
            return
        if not 1 <= length <= MAX_REQUEST_BYTES:
            self._json(413, {"error": "request_too_large"})
            return
        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            self._json(400, {"error": "invalid_json"})
            return
        try:
            result = asyncio.run(self.server.planner(payload))
        except RoboticsPlanningError as exc:
            self._json(
                422,
                {
                    "error": "plan_rejected",
                    "detail": str(exc)[:1000],
                },
            )
            return
        except OllamaStructuredOutputError as exc:
            self._json(
                502,
                {
                    "error": "provider_unavailable",
                    "detail": str(exc)[:1000],
                },
            )
            return
        except Exception:
            self._json(502, {"error": "provider_unavailable"})
            return
        self._json(200, result)


def build_server(
    *,
    host: str,
    port: int,
    model: str,
    timeout_seconds: float,
) -> RoboticsPlannerHTTPServer:
    """Create the default loopback server backed by a real Ollama model."""

    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("robotics planner binds to loopback only")
    provider = OllamaProvider(
        model=model,
        temperature=0.0,
        max_tokens=2500,
    )
    service = RoboticsPlanningService(
        provider,
        provider_name="flyto-ai",
        model=model.replace(":", "-"),
        timeout_seconds=timeout_seconds,
    )
    governed = GovernedRoboticsPlanner(service)
    return RoboticsPlannerHTTPServer((host, port), governed.plan)


def main() -> None:
    """Run the local structured planner until interrupted."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--model", default="flyto-qwen3:8b")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    server = build_server(
        host=args.host,
        port=args.port,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
