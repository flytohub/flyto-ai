# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Which lab planner request contract is accepted, and which are refused.

The lab planner (``flyto_ai.robotics_planning`` behind
``flyto_ai.robotics_planner_server``) takes one request contract. Versions it
no longer takes are declared here with what replaced them, so a stale caller is
told how to recover instead of failing on a field it never sent.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

# v2 names the commanded equipment ``resource_id`` and asks for a
# ``flyto.capability-plan.v1`` plan.
PLANNER_REQUEST_CONTRACT = "flyto.robotics.planner-request.v2"

# No compatibility window: no released Desktop reaches the lab planner (Flyto
# Cloud imports neither it nor flyto-robotics' planner client), and the only v1
# callers are flyto-robotics lab tools up to 0.6.6, run against a loopback
# planner started beside them. Remove an entry once no supported flyto-robotics
# release can send that contract.
RETIRED_PLANNER_REQUEST_CONTRACTS: Mapping[str, str] = MappingProxyType(
    {
        "flyto.robotics.planner-request.v1": (
            "robot_id became resource_id and plans became "
            "flyto.capability-plan.v1; upgrade flyto-robotics to 0.7.0 or later"
        ),
    }
)


def planner_contract_refusal(value: object) -> str | None:
    """Return why ``value`` is not the current request contract, else None."""
    if value == PLANNER_REQUEST_CONTRACT:
        return None
    if isinstance(value, str) and value in RETIRED_PLANNER_REQUEST_CONTRACTS:
        return (
            f"planner_contract {value} is retired: "
            f"{RETIRED_PLANNER_REQUEST_CONTRACTS[value]}"
        )
    return f"planner_contract must be {PLANNER_REQUEST_CONTRACT}"


def require_planner_contract(value: object, error: type[Exception]) -> None:
    """Raise ``error`` with the refusal reason unless ``value`` is current."""
    refusal = planner_contract_refusal(value)
    if refusal is not None:
        raise error(refusal)
