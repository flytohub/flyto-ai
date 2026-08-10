# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""A settled job must never read as an active one.

Two lifecycles were being conflated. `lifecycle` describes the *service
process*: whether this instance is still accepting and running work. A job's
terminality describes the *job*: whether its state can still change. They are
independent in both directions - a live instance can hold nothing but finished
jobs, and an instance that has closed can leave a job sitting in
`awaiting_codex_audit` for an auditor who has not arrived yet - so a reader that
infers one from the other is wrong roughly half the time.

`job_terminal` is derived from `state` rather than stored, and derived from the
one existing `TERMINAL_CODING_JOB_STATES` vocabulary, so the two can never
disagree and a new terminal state cannot be added in one place and forgotten in
the other.
"""
import dataclasses

import pytest

from flyto_ai.coding.contracts import (
    TERMINAL_CODING_JOB_STATES,
    CodingJobReceipt,
    CodingJobState,
)
from flyto_ai.coding.route_status import CodingRouteStatus
from flyto_ai.coding.service import receipt_to_mapping

_REVISION = "c1" * 32
#: States whose receipts refuse to exist without implementation evidence.
_BOUND_STATES = {
    CodingJobState.AWAITING_CODEX_AUDIT,
    CodingJobState.REWORK_QUEUED,
    CodingJobState.REWORK_RUNNING,
    CodingJobState.CODEX_ACCEPTED,
}
#: Of those, the ones that only exist after an auditor ordered rework.
_REWORK_STATES = {CodingJobState.REWORK_QUEUED, CodingJobState.REWORK_RUNNING}


def _receipt(state: CodingJobState) -> CodingJobReceipt:
    bound = {}
    if state in _BOUND_STATES:
        bound = {
            "implementation_revision_sha256": _REVISION,
            "implementation_session_id": "sdk-session-1",
            "implementation_backend": "native",
            # Every state past the first audit must show one, and a reworked
            # job must show the rework it came from.
            "audit_count": 0 if state is CodingJobState.AWAITING_CODEX_AUDIT else 1,
            "audit_findings_sha256": (
                "" if state is CodingJobState.AWAITING_CODEX_AUDIT else "d4" * 32
            ),
            "rework_count": 1 if state in _REWORK_STATES else 0,
            # An accepted receipt is landable by definition; every other state
            # is forbidden from claiming it.
            "landable": state is CodingJobState.CODEX_ACCEPTED,
        }
    return CodingJobReceipt(
        job_id="job_" + "a1b2c3d4" * 3,
        state=state,
        submitted_at=1.0,
        updated_at=2.0,
        **bound,
    )


@pytest.mark.parametrize(
    "state",
    [
        CodingJobState.QUEUED,
        CodingJobState.RUNNING,
        CodingJobState.AWAITING_CODEX_AUDIT,
        CodingJobState.REWORK_QUEUED,
        CodingJobState.REWORK_RUNNING,
    ],
)
def test_an_unsettled_job_is_not_terminal(state):
    receipt = _receipt(state)
    assert receipt.job_terminal is False
    assert receipt_to_mapping(receipt)["job_terminal"] is False


@pytest.mark.parametrize(
    "state",
    [
        CodingJobState.COMPLETED,
        CodingJobState.FAILED,
        CodingJobState.CODEX_ACCEPTED,
    ],
)
def test_a_settled_job_is_terminal(state):
    receipt = _receipt(state)
    assert receipt.job_terminal is True
    assert receipt_to_mapping(receipt)["job_terminal"] is True


def test_terminality_is_read_from_the_one_existing_vocabulary():
    """No second list: every state agrees with `TERMINAL_CODING_JOB_STATES`."""

    for state in CodingJobState:
        try:
            receipt = _receipt(state)
        except ValueError:  # pragma: no cover - a state we cannot construct
            continue
        assert receipt.job_terminal is (state in TERMINAL_CODING_JOB_STATES), state


def test_job_terminal_is_derived_and_cannot_be_stored_out_of_step():
    """It is a property, not a field, so nothing can set it to a lie."""

    fields = {field.name for field in dataclasses.fields(CodingJobReceipt)}
    assert "job_terminal" not in fields

    with pytest.raises(TypeError):
        CodingJobReceipt(
            job_id="job_x", state=CodingJobState.QUEUED,
            submitted_at=1.0, updated_at=2.0, job_terminal=True,
        )


def test_service_lifecycle_and_job_terminality_are_independent():
    """All four combinations are representable; neither implies the other."""

    for lifecycle in ("active", "closed"):
        for terminal in (False, True):
            status = CodingRouteStatus(
                instance_id="i" * 8,
                build_id="b" * 8,
                lifecycle=lifecycle,
                job_id="job_" + "a1b2c3d4" * 3,
                state=(
                    CodingJobState.FAILED.value if terminal
                    else CodingJobState.RUNNING.value
                ),
                job_terminal=terminal,
            )
            assert status.lifecycle == lifecycle
            assert status.job_terminal is terminal

    # The combination that used to be unrepresentable is the important one: a
    # live instance whose only job has already settled.
    live_but_done = CodingRouteStatus(
        instance_id="i" * 8, build_id="b" * 8,
        lifecycle="active", state=CodingJobState.COMPLETED.value, job_terminal=True,
    )
    assert live_but_done.lifecycle == "active"
    assert live_but_done.job_terminal is True


def test_status_defaults_to_not_terminal_for_a_row_with_no_job():
    status = CodingRouteStatus(instance_id="i" * 8, build_id="b" * 8)
    assert status.job_id == ""
    assert status.job_terminal is False
