# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from flyto_ai.robotics_planner_server import RoboticsPlannerHTTPServer
from flyto_ai.robotics_planning import RoboticsPlanningError


def request_json(url: str, payload: object) -> tuple[int, dict]:
    raw = json.dumps(payload).encode()
    request = Request(
        url,
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=2) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_http_boundary_serves_health_success_and_rejection() -> None:
    async def planner(payload):
        if payload.get("goal") == "reject":
            raise RoboticsPlanningError("unsafe plan")
        return {"contract_version": "test", "plan": {"goal": payload["goal"]}}

    server = RoboticsPlannerHTTPServer(("127.0.0.1", 0), planner)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(f"{base}/health", timeout=2) as response:
            health = json.loads(response.read())
        status, result = request_json(
            f"{base}/v1/robotics/plan",
            {"goal": "ok"},
        )
        rejected_status, rejected = request_json(
            f"{base}/v1/robotics/plan",
            {"goal": "reject"},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert health["planning_mode"] == "live_llm"
    assert status == 200
    assert result["plan"]["goal"] == "ok"
    assert rejected_status == 422
    assert rejected == {"error": "plan_rejected", "detail": "unsafe plan"}


def test_server_rejects_non_json_without_calling_planner() -> None:
    calls = []

    async def planner(payload):
        calls.append(payload)
        return {}

    server = RoboticsPlannerHTTPServer(("127.0.0.1", 0), planner)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    request = Request(
        f"http://127.0.0.1:{server.server_port}/v1/robotics/plan",
        data=b"not-json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with pytest.raises(HTTPError) as caught:
            urlopen(request, timeout=2)
        payload = json.loads(caught.value.read())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert caught.value.code == 400
    assert payload == {"error": "invalid_json"}
    assert calls == []
