# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Hostile tests for the optional coding mission envelope and its projection.

This slice adds a contract only. Nothing here creates, dispatches, or closes a
mission, and nothing here asserts that a `MissionStore` was touched - wiring the
lifecycle is the next isolated job. What is tested is exactly what shipped: the
value types, the strict decoders, the receipt projection's secret-safety, the
unchanged public tool inventory, and that a caller who never names a mission is
unaffected.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from flyto_ai.coding.contracts import (
    MISSION_LANE_PRIMARY,
    MISSION_MAX_ACCEPTANCE_CRITERIA,
    MISSION_MAX_DEPENDENCIES,
    MISSION_MAX_PRIORITY,
    MISSION_COMPLETED,
    MISSION_OPEN,
    MISSION_PROJECTION_FIELDS,
    MISSION_STATUS_CLOSED,
    MISSION_STATUS_READY,
    CodingJobReceipt,
    CodingJobState,
    CodingMissionEnvelope,
    CodingMissionProjection,
    CodingTaskRequest,
    mission_axis_sha256,
)
from flyto_ai.coding.service import receipt_to_mapping, request_from_mapping
from flyto_ai.orchestration.mission_control import AcceptanceCriterion


MISSION_ID = "m-000000000001"
PARENT_ID = "w-000000000001"
RETURN_ID = "w-000000000002"
WORK_ID = "w-000000000003"


def _criterion(identifier: str = "c1") -> AcceptanceCriterion:
    return AcceptanceCriterion(identifier, "the declared checks pass")


def _envelope(**overrides: object) -> CodingMissionEnvelope:
    fields: dict = {
        "scope": "scope-token",
        "objective": "reach the stated objective",
        "desired_result": "the objective is demonstrably reached",
        "acceptance_criteria": (_criterion(),),
    }
    fields.update(overrides)
    return CodingMissionEnvelope(**fields)  # type: ignore[arg-type]


def _mission_payload(**overrides: object) -> dict:
    payload: dict = {
        "scope": "scope-token",
        "objective": "reach the stated objective",
        "desired_result": "the objective is demonstrably reached",
        "acceptance_criteria": [{"id": "c1", "statement": "the declared checks pass"}],
    }
    payload.update(overrides)
    return payload


def _projection(**overrides: object) -> CodingMissionProjection:
    fields: dict = {
        "mission_id": MISSION_ID,
        "scope": "scope-token",
        "work_item_id": WORK_ID,
        "main_axis_sha256": _envelope().main_axis_sha256,
        "criteria_ids": ("c1",),
        "lane": MISSION_LANE_PRIMARY,
        "priority": 0,
        "status": MISSION_STATUS_READY,
    }
    fields.update(overrides)
    return CodingMissionProjection(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# dataclass validation
# --------------------------------------------------------------------------


def test_a_root_envelope_is_accepted_and_addresses_its_own_axis() -> None:
    envelope = _envelope()
    assert envelope.is_root
    assert envelope.criteria_ids == ("c1",)
    assert envelope.main_axis_sha256 == mission_axis_sha256(
        envelope.scope,
        envelope.objective,
        envelope.desired_result,
        envelope.acceptance_criteria,
    )
    assert len(envelope.main_axis_sha256) == 64


def test_the_main_axis_digest_is_domain_separated_and_unambiguous() -> None:
    """Distinct field tuples cannot collide by joining, and prose is bound."""

    base = _envelope()
    moved = _envelope(scope="scope-toke", objective="nreach the stated objective")
    assert base.main_axis_sha256 != moved.main_axis_sha256
    assert base.main_axis_sha256 != _envelope(
        acceptance_criteria=(_criterion(), _criterion("c2")),
    ).main_axis_sha256


@pytest.mark.parametrize(
    "overrides",
    [
        {"scope": "has whitespace"},
        {"scope": ""},
        {"scope": "line\nbreak"},
        {"objective": "   "},
        {"objective": "controlchar"},
        {"desired_result": "x" * 2001},
        {"priority": True},
        {"priority": -1},
        {"priority": MISSION_MAX_PRIORITY + 1},
        {"lane": "made-up-lane"},
        {"mission_id": "m-1"},
        {"mission_id": MISSION_ID + "0"},
        {"acceptance_criteria": ()},
        {"acceptance_criteria": (_criterion(), _criterion())},
        {"acceptance_criteria": ({"id": "c1", "statement": "s"},)},
        {"acceptance_criteria": "c1"},
    ],
)
def test_the_envelope_refuses_malformed_values(overrides: dict) -> None:
    with pytest.raises(ValueError):
        _envelope(**overrides)


def test_a_boolean_priority_is_not_an_integer() -> None:
    """`True` is an `int` in Python and is never a priority here."""

    with pytest.raises(ValueError):
        _envelope(priority=False)


@pytest.mark.parametrize(
    "overrides",
    [
        # Half a lineage: a parent with no route home, or the reverse.
        {"mission_id": MISSION_ID, "parent_id": PARENT_ID},
        {"mission_id": MISSION_ID, "return_to_id": RETURN_ID},
        # A side item with no mission to descend from.
        {"parent_id": PARENT_ID, "return_to_id": RETURN_ID},
        # A root item cannot sit in the repair lane...
        {"lane": "repair"},
        # ...and cannot depend on work that precedes the first work item.
        {"depends_on_ids": (PARENT_ID,)},
    ],
)
def test_the_envelope_refuses_illegal_lineage(overrides: dict) -> None:
    with pytest.raises(ValueError):
        _envelope(**overrides)


def test_a_side_envelope_may_take_the_repair_lane_and_dependencies() -> None:
    envelope = _envelope(
        mission_id=MISSION_ID,
        parent_id=PARENT_ID,
        return_to_id=RETURN_ID,
        lane="repair",
        priority=MISSION_MAX_PRIORITY,
        depends_on_ids=(WORK_ID,),
    )
    assert not envelope.is_root
    assert envelope.lane == "repair"
    assert envelope.depends_on_ids == (WORK_ID,)


@pytest.mark.parametrize(
    "dependencies",
    [
        (WORK_ID, WORK_ID),
        ("w-1",),
        ("not-an-id",),
        tuple("w-{:012d}".format(index) for index in range(MISSION_MAX_DEPENDENCIES + 1)),
    ],
)
def test_the_envelope_refuses_bad_dependencies(dependencies: tuple) -> None:
    with pytest.raises(ValueError):
        _envelope(
            mission_id=MISSION_ID,
            parent_id=PARENT_ID,
            return_to_id=RETURN_ID,
            depends_on_ids=dependencies,
        )


def test_too_many_acceptance_criteria_are_refused() -> None:
    criteria = tuple(
        _criterion("c{}".format(index))
        for index in range(MISSION_MAX_ACCEPTANCE_CRITERIA + 1)
    )
    with pytest.raises(ValueError):
        _envelope(acceptance_criteria=criteria)


def test_the_envelope_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _envelope().scope = "other"  # type: ignore[misc]


# --------------------------------------------------------------------------
# mapping / schema decode
# --------------------------------------------------------------------------


def test_the_envelope_round_trips_through_its_mapping() -> None:
    envelope = _envelope(
        mission_id=MISSION_ID,
        parent_id=PARENT_ID,
        return_to_id=RETURN_ID,
        priority=7,
        depends_on_ids=(WORK_ID,),
    )
    assert CodingMissionEnvelope.from_mapping(envelope.to_mapping()) == envelope
    # The projection is JSON, not Python objects.
    json.dumps(envelope.to_mapping())


@pytest.mark.parametrize(
    "overrides",
    [
        {"unknown_field": "x"},
        {"acceptance_criteria": [{"id": "c1", "statement": "s", "owner": "someone"}]},
        {"acceptance_criteria": [{"id": "c1"}]},
        {"acceptance_criteria": "c1"},
        {"acceptance_criteria": [["c1", "s"]]},
        {"priority": True},
        {"priority": "3"},
        {"priority": 3.0},
        {"lane": 1},
        {"mission_id": 5},
        {"parent_id": True},
        {"depends_on_ids": WORK_ID},
        {"depends_on_ids": [1]},
    ],
)
def test_the_envelope_decoder_is_strict(overrides: dict) -> None:
    with pytest.raises(ValueError):
        CodingMissionEnvelope.from_mapping(_mission_payload(**overrides))


def test_the_envelope_decoder_refuses_a_non_object() -> None:
    for value in ("mission", 3, [], None):
        with pytest.raises(ValueError):
            CodingMissionEnvelope.from_mapping(value)


def test_too_many_decoded_criteria_are_refused_before_construction() -> None:
    payload = _mission_payload(acceptance_criteria=[
        {"id": "c{}".format(index), "statement": "s"}
        for index in range(MISSION_MAX_ACCEPTANCE_CRITERIA + 1)
    ])
    with pytest.raises(ValueError):
        CodingMissionEnvelope.from_mapping(payload)


# --------------------------------------------------------------------------
# receipt projection
# --------------------------------------------------------------------------


def test_the_projection_publishes_exactly_its_closed_field_set() -> None:
    projection = _projection()
    assert set(projection.to_mapping()) == MISSION_PROJECTION_FIELDS
    assert CodingMissionProjection.from_mapping(projection.to_mapping()) == projection


def test_the_projection_reports_both_statuses_and_a_grouping_scope() -> None:
    """A consumer groups by scope and reads completion without inferring it."""

    body = _projection().to_mapping()
    assert body["scope"] == "scope-token"
    assert body["mission_status"] == MISSION_OPEN
    # The work item and the mission are two different clocks.
    assert body["status"] == MISSION_STATUS_READY
    done = _projection(
        status=MISSION_STATUS_CLOSED,
        disposition="fixed",
        mission_status=MISSION_COMPLETED,
    )
    assert done.to_mapping()["mission_status"] == MISSION_COMPLETED


@pytest.mark.parametrize(
    "overrides",
    [
        # A completed mission has no open work items...
        {"mission_status": MISSION_COMPLETED},
        {"mission_status": MISSION_COMPLETED, "status": MISSION_STATUS_READY},
        # ...and never completes on a deferred or blocked main axis.
        {
            "mission_status": MISSION_COMPLETED,
            "status": MISSION_STATUS_CLOSED,
            "disposition": "deferred",
        },
        {
            "mission_status": MISSION_COMPLETED,
            "status": MISSION_STATUS_CLOSED,
            "disposition": "blocked",
        },
        {"mission_status": "made-up-status"},
        {"mission_status": True},
        {"scope": "has whitespace"},
        {"scope": ""},
    ],
)
def test_the_projection_refuses_incoherent_mission_lifecycle(overrides: dict) -> None:
    with pytest.raises(ValueError):
        _projection(**overrides)


def test_a_completed_mission_may_carry_a_closed_side_item() -> None:
    """The fixed-root rule binds the main axis, not every branch."""

    projection = _projection(
        parent_id=PARENT_ID,
        return_to_id=RETURN_ID,
        status=MISSION_STATUS_CLOSED,
        disposition="deferred",
        mission_status=MISSION_COMPLETED,
        returned_to_main_axis=True,
    )
    assert not projection.is_root
    assert projection.mission_status == MISSION_COMPLETED


def test_the_projection_carries_no_prose_and_no_workspace() -> None:
    """The whole point: identity and position, never content."""

    forbidden = {
        "objective", "desired_result", "statement", "acceptance_criteria",
        "rationale", "risk", "owner", "evidence_refs", "workspace",
        "working_dir",
    }
    assert not (MISSION_PROJECTION_FIELDS & forbidden)
    body = json.dumps(_projection().to_mapping())
    assert "reach the stated objective" not in body
    assert "the objective is demonstrably reached" not in body


def test_the_envelope_builds_a_faithful_projection() -> None:
    envelope = _envelope(
        mission_id=MISSION_ID,
        parent_id=PARENT_ID,
        return_to_id=RETURN_ID,
        lane="repair",
        priority=4,
    )
    projection = envelope.projection(
        mission_id=MISSION_ID,
        work_item_id=WORK_ID,
        status=MISSION_STATUS_CLOSED,
        disposition="fixed",
        returned_to_main_axis=True,
    )
    assert projection.scope == envelope.scope
    assert projection.mission_status == MISSION_OPEN
    assert projection.main_axis_sha256 == envelope.main_axis_sha256
    assert projection.criteria_ids == envelope.criteria_ids
    assert projection.lane == "repair"
    assert projection.priority == 4
    assert projection.returned_to_main_axis is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"mission_id": WORK_ID},
        {"work_item_id": MISSION_ID},
        {"main_axis_sha256": "not-a-digest"},
        {"main_axis_sha256": ""},
        {"criteria_ids": ()},
        {"criteria_ids": ("c1", "c1")},
        {"lane": "made-up-lane"},
        {"priority": True},
        {"status": "made-up-status"},
        # A closed item must say how it left; an open one must not.
        {"disposition": "fixed"},
        {"status": MISSION_STATUS_CLOSED},
        {"status": MISSION_STATUS_CLOSED, "disposition": "made-up"},
        # Half a lineage.
        {"parent_id": PARENT_ID},
        {"return_to_id": RETURN_ID},
        # A root item is already on the axis; an open side item has not returned.
        {"returned_to_main_axis": True},
        {
            "parent_id": PARENT_ID,
            "return_to_id": RETURN_ID,
            "returned_to_main_axis": True,
        },
    ],
)
def test_the_projection_refuses_incoherent_state(overrides: dict) -> None:
    with pytest.raises(ValueError):
        _projection(**overrides)


def test_the_projection_decoder_refuses_unknown_fields() -> None:
    payload = _projection().to_mapping()
    payload["objective"] = "smuggled prose"
    with pytest.raises(ValueError):
        CodingMissionProjection.from_mapping(payload)


# --------------------------------------------------------------------------
# request / receipt integration
# --------------------------------------------------------------------------


def test_a_legacy_request_is_unchanged(tmp_path: Path) -> None:
    request = request_from_mapping({
        "message": "do the work", "working_dir": str(tmp_path),
    })
    assert request.mission is None
    assert dataclasses.asdict(request)["mission"] is None


def test_a_request_may_carry_a_mission(tmp_path: Path) -> None:
    request = request_from_mapping({
        "message": "do the work",
        "working_dir": str(tmp_path),
        "mission": _mission_payload(),
    })
    assert isinstance(request.mission, CodingMissionEnvelope)
    assert request.mission.scope == "scope-token"


def test_a_request_refuses_a_raw_mission_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CodingTaskRequest(
            message="do the work",
            working_dir=str(tmp_path),
            mission=_mission_payload(),  # type: ignore[arg-type]
        )


def test_a_bad_mission_fails_the_whole_request(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        request_from_mapping({
            "message": "do the work",
            "working_dir": str(tmp_path),
            "mission": _mission_payload(unknown_field="x"),
        })


def _receipt(**overrides: object) -> CodingJobReceipt:
    fields: dict = {
        "job_id": "job_" + "a" * 24,
        "state": CodingJobState.QUEUED,
        "submitted_at": 1.0,
        "updated_at": 2.0,
    }
    fields.update(overrides)
    return CodingJobReceipt(**fields)  # type: ignore[arg-type]


def test_a_legacy_receipt_still_constructs_and_projects() -> None:
    body = receipt_to_mapping(_receipt())
    assert body["mission"] is None
    assert body["job_id"].startswith("job_")


def test_a_receipt_revalidates_and_canonicalizes_its_mission() -> None:
    receipt = _receipt(mission=_projection().to_mapping())
    assert set(receipt.mission or {}) == MISSION_PROJECTION_FIELDS
    body = receipt_to_mapping(receipt)
    assert set(body["mission"]) == MISSION_PROJECTION_FIELDS
    assert body["mission"]["mission_id"] == MISSION_ID
    json.dumps(body)


def test_a_receipt_refuses_a_smuggled_mission_field() -> None:
    payload = _projection().to_mapping()
    payload["objective"] = "smuggled prose"
    with pytest.raises(ValueError):
        _receipt(mission=payload)


def test_a_receipt_refuses_an_incoherent_mission() -> None:
    payload = _projection().to_mapping()
    payload["main_axis_sha256"] = "0" * 63
    with pytest.raises(ValueError):
        _receipt(mission=payload)


def test_a_receipt_refuses_a_non_object_mission() -> None:
    with pytest.raises(ValueError):
        _receipt(mission="m-000000000001")


# --------------------------------------------------------------------------
# public MCP surface
# --------------------------------------------------------------------------


def test_the_mission_envelope_adds_no_mcp_tool() -> None:
    """The audited public inventory stays exactly submit/get/audit."""

    from flyto_ai.coding.mcp_server import CodingMCPServer

    tools = CodingMCPServer._tools()
    assert [tool["name"] for tool in tools] == [
        "flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit",
    ]
    assert len(tools) == 3
    assert not any("mission" in tool["name"] for tool in tools)


def test_the_submit_schema_declares_the_mission_strictly() -> None:
    from flyto_ai.coding.mcp_server import CodingMCPServer

    submit = CodingMCPServer._tools()[0]
    request = submit["inputSchema"]["properties"]["request"]
    mission = request["properties"]["mission"]
    # Optional and additive: the required set is untouched.
    assert request["required"] == ["message", "working_dir"]
    assert mission["additionalProperties"] is False
    assert mission["required"] == [
        "scope", "objective", "desired_result", "acceptance_criteria",
    ]
    assert mission["properties"]["acceptance_criteria"]["items"][
        "additionalProperties"
    ] is False
    assert mission["properties"]["lane"]["enum"] == ["repair", "primary"]
    assert mission["properties"]["priority"]["maximum"] == MISSION_MAX_PRIORITY
    # Every declared bound is a real bound, and the schema is JSON.
    json.dumps(submit)


def test_the_published_schema_matches_what_the_decoder_accepts() -> None:
    """A schema that advertised a shape the decoder refuses would be a lie."""

    from flyto_ai.coding.mcp_server import CodingMCPServer

    mission = (
        CodingMCPServer._tools()[0]["inputSchema"]["properties"]["request"]
        ["properties"]["mission"]
    )
    assert set(mission["properties"]) == set(_mission_payload()) | {
        "priority", "lane", "mission_id", "parent_id", "return_to_id",
        "depends_on_ids",
    }
    # And the decoder accepts a payload built to exactly that schema.
    decoded = CodingMissionEnvelope.from_mapping(_mission_payload(
        priority=3,
        lane=MISSION_LANE_PRIMARY,
    ))
    assert decoded.priority == 3


def test_the_schema_declares_the_uniqueness_the_decoder_enforces() -> None:
    from flyto_ai.coding.mcp_server import CodingMCPServer

    mission = (
        CodingMCPServer._tools()[0]["inputSchema"]["properties"]["request"]
        ["properties"]["mission"]
    )
    assert mission["properties"]["acceptance_criteria"]["uniqueItems"] is True
    assert mission["properties"]["depends_on_ids"]["uniqueItems"] is True


def test_an_explicitly_null_mission_is_refused_not_ignored(tmp_path: Path) -> None:
    """Absent and null are different payloads; the schema types this an object.

    Reading `null` as "no mission" would hand a caller who believed it had named
    one a job that quietly ignored it. Both transports decode here, so MCP and
    HTTP cannot disagree about it.
    """

    with pytest.raises(ValueError):
        request_from_mapping({
            "message": "do the work",
            "working_dir": str(tmp_path),
            "mission": None,
        })
    # Omitting the key remains the supported way to send no mission.
    assert request_from_mapping({
        "message": "do the work", "working_dir": str(tmp_path),
    }).mission is None


@pytest.mark.parametrize(
    "mission_value",
    ["", 0, False, [], "null"],
)
def test_a_non_object_mission_is_refused(tmp_path: Path, mission_value: object) -> None:
    with pytest.raises(ValueError):
        request_from_mapping({
            "message": "do the work",
            "working_dir": str(tmp_path),
            "mission": mission_value,
        })


# --------------------------------------------------------------------------
# public package surface
# --------------------------------------------------------------------------


def test_the_mission_contract_is_importable_from_the_package() -> None:
    """Callers bind to `flyto_ai.coding`, never to an internal module path."""

    import flyto_ai.coding as coding

    exported = (
        "CodingMissionEnvelope",
        "CodingMissionProjection",
        "mission_axis_sha256",
        "MISSION_COMPLETED",
        "MISSION_DISPOSITION_FIXED",
        "MISSION_DISPOSITIONS",
        "MISSION_ID_PATTERN",
        "MISSION_LANE_PRIMARY",
        "MISSION_LANES",
        "MISSION_MAX_ACCEPTANCE_CRITERIA",
        "MISSION_MAX_DEPENDENCIES",
        "MISSION_MAX_FIELD_CHARS",
        "MISSION_MAX_PRIORITY",
        "MISSION_MAX_TEXT_CHARS",
        "MISSION_OPEN",
        "MISSION_PROJECTION_FIELDS",
        "MISSION_STATUS_CLOSED",
        "MISSION_STATUS_DISPATCHED",
        "MISSION_STATUS_READY",
        "MISSION_STATUSES",
        "MISSION_WORK_STATUSES",
        "WORK_ITEM_ID_PATTERN",
    )
    for name in exported:
        assert name in coding.__all__, name
        assert hasattr(coding, name), name
    # The package-level types are the same objects, not a parallel definition.
    assert coding.CodingMissionEnvelope is CodingMissionEnvelope
    assert coding.CodingMissionProjection is CodingMissionProjection


def test_the_package_surface_builds_a_mission_without_internal_imports() -> None:
    import flyto_ai.coding as coding

    envelope = coding.CodingMissionEnvelope(
        scope="scope-token",
        objective="reach the stated objective",
        desired_result="the objective is demonstrably reached",
        acceptance_criteria=(AcceptanceCriterion("c1", "the declared checks pass"),),
        lane=coding.MISSION_LANE_PRIMARY,
    )
    projection = envelope.projection(
        mission_id=MISSION_ID,
        work_item_id=WORK_ID,
        status=coding.MISSION_STATUS_CLOSED,
        disposition=coding.MISSION_DISPOSITION_FIXED,
        mission_status=coding.MISSION_COMPLETED,
    )
    assert set(projection.to_mapping()) == coding.MISSION_PROJECTION_FIELDS
