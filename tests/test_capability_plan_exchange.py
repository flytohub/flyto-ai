# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""The planner wire contract flyto-ai shares with flyto-robotics.

``tests/fixtures/capability-plan-exchange.v1.json`` is byte-identical to the
copy in flyto-robotics' ``tests/fixtures/``: a request robotics really builds
and a plan robotics accepts unchanged. Both repos pin its digest, so the two
sides cannot drift apart one PR at a time.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from flyto_ai.capability_router import CapabilityRoutingError, prepare_planner_request
from flyto_ai.planner_contract import (
    PLANNER_REQUEST_CONTRACT,
    RETIRED_PLANNER_REQUEST_CONTRACTS,
)
from flyto_ai.robotics_planning import (
    PLAN_CONTRACT,
    RoboticsPlanningError,
    build_plan_schema,
    validate_plan,
    validate_request,
)

FIXTURE = Path(__file__).parent / "fixtures" / "capability-plan-exchange.v1.json"
# Must equal EXCHANGE_FIXTURE_SHA256 in flyto-robotics tests/test_capability_plan_exchange.py.
EXCHANGE_FIXTURE_SHA256 = (
    "a891da42ce4f5af2fb57abc441185de8f8e511be20cec810dcfc4f293beae877"
)
SIBLING = (
    Path(__file__).resolve().parents[2]
    / "flyto-robotics"
    / "tests"
    / "fixtures"
    / "capability-plan-exchange.v1.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_is_the_pinned_shared_copy() -> None:
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == EXCHANGE_FIXTURE_SHA256


def test_fixture_matches_a_sibling_robotics_checkout() -> None:
    if not SIBLING.is_file():
        pytest.skip("no flyto-robotics checkout beside this one")
    assert SIBLING.read_bytes() == FIXTURE.read_bytes()


def test_planner_accepts_what_robotics_sends() -> None:
    request = _fixture()["request"]
    assert request["planner_contract"] == PLANNER_REQUEST_CONTRACT
    validated = validate_request(request)
    assert validated.payload["resource_id"] == request["resource_id"]
    source = _fixture()["plan"]["generated_by"]
    schema = build_plan_schema(
        validated, provider_name=source["provider"], model=source["model"]
    )
    assert schema["properties"]["contract_version"]["const"] == PLAN_CONTRACT
    assert schema["properties"]["resource_id"]["const"] == request["resource_id"]
    assert "robot_id" not in schema["properties"]


def test_planner_emits_a_plan_robotics_accepts() -> None:
    fixture = _fixture()
    plan = fixture["plan"]
    assert plan["contract_version"] == PLAN_CONTRACT == "flyto.capability-plan.v1"
    normalized, _ = validate_plan(
        plan,
        validate_request(fixture["request"]),
        provider_name=plan["generated_by"]["provider"],
        model=plan["generated_by"]["model"],
    )
    assert normalized == plan


def _retired_request() -> dict:
    request = copy.deepcopy(_fixture()["request"])
    request["planner_contract"] = "flyto.robotics.planner-request.v1"
    request["robot_id"] = request.pop("resource_id")
    return request


def test_retired_request_contract_is_refused_by_name() -> None:
    assert "flyto.robotics.planner-request.v1" in RETIRED_PLANNER_REQUEST_CONTRACTS
    with pytest.raises(RoboticsPlanningError, match="is retired: robot_id became"):
        validate_request(_retired_request())


@pytest.mark.asyncio
async def test_router_refuses_retired_request_before_any_discovery() -> None:
    async def never_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("discovery must not run for a retired contract")

    with pytest.raises(CapabilityRoutingError, match="upgrade flyto-robotics"):
        await prepare_planner_request(
            _retired_request(),
            core_dispatch=never_called,
            blueprint_search=never_called,
        )


def test_unknown_request_contract_names_the_current_one() -> None:
    request = copy.deepcopy(_fixture()["request"])
    request["planner_contract"] = "flyto.robotics.planner-request.v9"
    with pytest.raises(RoboticsPlanningError, match=PLANNER_REQUEST_CONTRACT):
        validate_request(request)
