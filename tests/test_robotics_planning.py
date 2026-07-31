# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import copy
import json

import pytest

from flyto_ai.robotics_planning import (
    ATTESTATION_CONTRACT,
    RESPONSE_CONTRACT,
    RoboticsPlanningError,
    RoboticsPlanningService,
)


def request_payload() -> dict:
    return {
        "planner_contract": "flyto.robotics.planner-request.v1",
        "instructions": "Use only shortlisted capabilities and stop safely.",
        "goal": "Go through one available branch and ask for approval.",
        "goal_frame": None,
        "robot_id": "robot.test",
        "capability_route": {
            "registry_snapshot": "a" * 64,
            "confidence": 1.0,
            "needs_clarification": False,
            "candidates": [
                {"runtime_name": "navigate_to_location"},
                {"runtime_name": "ask_human"},
                {"runtime_name": "resume"},
                {"runtime_name": "safe_stop"},
            ],
        },
        "capabilities": [
            {
                "runtime_name": "navigate_to_location",
                "arguments": [
                    {
                        "name": "location_id",
                        "type": "string",
                        "required": True,
                    }
                ],
            },
            {
                "runtime_name": "ask_human",
                "arguments": [
                    {
                        "name": "approval_id",
                        "type": "string",
                        "required": True,
                    },
                    {
                        "name": "prompt_key",
                        "type": "string",
                        "required": True,
                    },
                ],
            },
            {
                "runtime_name": "resume",
                "arguments": [
                    {
                        "name": "approval_id",
                        "type": "string",
                        "required": True,
                    }
                ],
            },
            {
                "runtime_name": "safe_stop",
                "arguments": [
                    {
                        "name": "seconds",
                        "type": "number",
                        "required": False,
                        "minimum": 0.0,
                        "maximum": 300.0,
                    }
                ],
            },
        ],
        "observations": {
            "semantic_map": {
                "locations": [
                    {"location_id": "route.orange.1"},
                    {"location_id": "route.orange.2"},
                    {"location_id": "route.yellow.1"},
                ]
            },
            "route_candidates": [
                {
                    "route_id": "route.orange",
                    "location_ids": ["route.orange.1", "route.orange.2"],
                    "score": 10,
                    "reason_codes": ["healthy"],
                },
                {
                    "route_id": "route.yellow",
                    "location_ids": ["route.yellow.1"],
                    "score": 8,
                    "reason_codes": ["shorter"],
                },
            ],
        },
    }


def valid_plan() -> dict:
    goal = request_payload()["goal"]
    return {
        "contract_version": "flyto.robotics.plan.v1",
        "plan_id": "plan.test.1",
        "robot_id": "robot.test",
        "goal": goal,
        "generated_by": {
            "kind": "llm",
            "provider": "flyto-ai",
            "model": "test-model",
        },
        "steps": [
            {
                "step_id": "orange.1",
                "capability": "navigate_to_location",
                "arguments": {"location_id": "route.orange.1"},
                "timeout_seconds": 30.0,
                "on_failure": "request_replan",
            },
            {
                "step_id": "orange.2",
                "capability": "navigate_to_location",
                "arguments": {"location_id": "route.orange.2"},
                "timeout_seconds": 30.0,
                "on_failure": "request_replan",
            },
            {
                "step_id": "approval.ask",
                "capability": "ask_human",
                "arguments": {
                    "approval_id": "delivery.test",
                    "prompt_key": "confirm.delivery",
                },
                "timeout_seconds": 30.0,
                "on_failure": "abort",
            },
            {
                "step_id": "approval.resume",
                "capability": "resume",
                "arguments": {"approval_id": "delivery.test"},
                "timeout_seconds": 5.0,
                "on_failure": "abort",
            },
            {
                "step_id": "stop",
                "capability": "safe_stop",
                "arguments": {"seconds": 1.0},
                "timeout_seconds": 2.0,
                "on_failure": "abort",
            },
        ],
    }


class FakeStructuredProvider:
    def __init__(self, plans: list[dict]) -> None:
        self._plans = list(plans)
        self.calls = []

    async def complete_json_schema(
        self,
        *,
        messages,
        schema,
        timeout_seconds=120.0,
    ):
        self.calls.append(
            {
                "messages": copy.deepcopy(messages),
                "schema": copy.deepcopy(schema),
                "timeout_seconds": timeout_seconds,
            }
        )
        plan = self._plans.pop(0)
        return {
            "model": "test-model",
            "created_at": "2026-07-30T00:00:00Z",
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 20,
            "total_duration": 1_000_000,
            "message": {"content": json.dumps(plan)},
        }


@pytest.mark.asyncio
async def test_live_plan_repairs_once_and_returns_hashed_attestation() -> None:
    unsafe = valid_plan()
    unsafe["steps"] = [
        unsafe["steps"][0],
        unsafe["steps"][2],
        unsafe["steps"][1],
        unsafe["steps"][-1],
    ]
    provider = FakeStructuredProvider([unsafe, valid_plan()])
    service = RoboticsPlanningService(
        provider,
        provider_name="flyto-ai",
        model="test-model",
    )

    result = await service.plan(request_payload())

    assert result["contract_version"] == RESPONSE_CONTRACT
    assert result["attestation"]["contract_version"] == ATTESTATION_CONTRACT
    assert result["attestation"]["mode"] == "live_llm"
    assert result["attestation"]["attempt_count"] == 2
    assert result["attestation"]["attempts"][0]["validation"]["passed"] is False
    assert result["attestation"]["attempts"][1]["validation"]["passed"] is True
    assert result["attestation"]["selected_route_id"] == "route.orange"
    assert len(result["attestation"]["snapshot"]) == 64
    assert len(result["attestation"]["request_sha256"]) == 64
    assert len(result["attestation"]["plan_sha256"]) == 64
    assert "independent validator rejected" in provider.calls[1]["messages"][-1][
        "content"
    ].lower()
    first_schema = provider.calls[0]["schema"]
    route_variants = first_schema["properties"]["steps"]["oneOf"]
    orange_route = next(
        variant
        for variant in route_variants
        if variant["prefixItems"][0]["properties"]["arguments"]["properties"][
            "location_id"
        ].get("const")
        == "route.orange.1"
    )
    assert [
        step["properties"]["capability"]["const"]
        for step in orange_route["prefixItems"]
    ] == [
        "navigate_to_location",
        "navigate_to_location",
        "ask_human",
        "resume",
        "safe_stop",
    ]
    assert orange_route["minItems"] == orange_route["maxItems"] == 5


@pytest.mark.asyncio
async def test_plan_fails_closed_after_bounded_repair() -> None:
    invalid = valid_plan()
    invalid["steps"][0]["arguments"]["linear_x"] = 1.0
    provider = FakeStructuredProvider([invalid, invalid])
    service = RoboticsPlanningService(
        provider,
        provider_name="flyto-ai",
        model="test-model",
    )

    with pytest.raises(RoboticsPlanningError, match="all structured"):
        await service.plan(request_payload())

    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_request_rejects_capability_catalog_outside_shortlist() -> None:
    payload = request_payload()
    payload["capabilities"].append(
        {"runtime_name": "raw_motor", "arguments": []}
    )
    provider = FakeStructuredProvider([valid_plan()])
    service = RoboticsPlanningService(
        provider,
        provider_name="flyto-ai",
        model="test-model",
    )

    with pytest.raises(
        RoboticsPlanningError,
        match="outside the routed shortlist",
    ):
        await service.plan(payload)

    assert provider.calls == []
