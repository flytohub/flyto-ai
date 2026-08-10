# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Provider failures are named from a closed allowlist, or not named at all.

Two properties matter more than the individual codes.  First, the raw provider
text is a *classification input only*: it is matched in memory and dropped, so
no test here may ever find provider prose on the far side of the boundary.
Second, an unrecognized failure stays the conservative `provider_failed` rather
than being guessed into a friendlier category - guessing is how a terminal
billing failure would quietly become a retry loop.
"""
import pathlib
import re

import pytest

from flyto_ai.agents import claude_code
from flyto_ai.agents.claude_code import (
    DEFAULT_CLAUDE_MODEL,
    MAX_PROVIDER_ERROR_SCAN_CHARS,
    PROVIDER_FAILURE_MARKERS,
    RESUMABLE_PROVIDER_FAILURE_CODES,
    RETRYABLE_PROVIDER_FAILURE_CODES,
    provider_failure_code,
)
from flyto_ai.coding.contracts import (
    CodingJobReceipt,
    CodingJobState,
    CodingTaskResult,
)
from flyto_ai.coding.service import receipt_to_mapping


@pytest.mark.parametrize(
    "text,expected",
    [
        ("error_max_turns", "turn_limit_exceeded"),
        ("Claude reached maximum number of turns", "turn_limit_exceeded"),
        ("authentication_error: invalid x-api-key", "provider_auth_failed"),
        ("Could not resolve authentication method", "provider_auth_failed"),
        ("permission_error while calling the API", "provider_auth_failed"),
        ("Your credit balance is too low to continue", "provider_quota_exhausted"),
        ("monthly quota exceeded for this org", "provider_quota_exhausted"),
        ("rate_limit_error: slow down", "provider_capacity_unavailable"),
        ("overloaded_error", "provider_capacity_unavailable"),
        ("blocked by content policy", "provider_policy_refused"),
        ("stop_reason: refusal", "provider_policy_refused"),
    ],
)
def test_recognized_conditions_get_their_own_stable_code(text, expected):
    assert provider_failure_code(RuntimeError(text)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "sdk transport failure",
        "connection reset by peer",
        "wrote 529 bytes then gave up",
        "the auditor issued a refusal",
        "",
        "something nobody has seen before",
    ],
)
def test_an_unrecognized_failure_is_never_guessed_into_a_category(text):
    assert provider_failure_code(RuntimeError(text)) == "provider_failed"


def test_the_code_never_carries_provider_text():
    """The message is an input to classification and nothing else."""

    secret = (
        "authentication_error: key "
        + "sk-"
        + "ant-not-a-real-key-9f8e7d for acme-corp"
    )
    code = provider_failure_code(RuntimeError(secret))

    assert code == "provider_auth_failed"
    assert "sk-ant" not in code
    assert "acme-corp" not in code
    assert code.replace("_", "").isalpha()


def test_classification_is_bounded_and_survives_a_hostile_exception():
    class Hostile(Exception):
        def __str__(self):
            raise ValueError("this exception refuses to be read")

    assert provider_failure_code(Hostile()) == "provider_failed"

    # A marker hidden past the scan bound is not found, and that is the safe
    # direction to fail: unrecognized, never mislabelled.
    padded = "x" * (MAX_PROVIDER_ERROR_SCAN_CHARS + 50) + "rate_limit_error"
    assert provider_failure_code(RuntimeError(padded)) == "provider_failed"


def test_only_transient_capacity_is_retryable_and_only_bounded_stops_resume():
    """Terminal categories must not leak into either narrow set.

    The resumable set is the *bounded stops*: conditions that end a round after
    a real session exists because a host-configured ceiling was reached. The
    configured spend ceiling joined the turn ceiling there, and it belongs for
    the same reason - the work happened, the session is real, and only the bound
    ran out. Still exact: a set that grows by accident is how a terminal
    condition quietly becomes resumable.
    """

    assert RETRYABLE_PROVIDER_FAILURE_CODES == {"provider_capacity_unavailable"}
    assert RESUMABLE_PROVIDER_FAILURE_CODES == {
        "turn_limit_exceeded", "provider_job_budget_exhausted",
    }
    # A configured job ceiling is resumable; an account quota is not.
    assert "provider_quota_exhausted" not in RESUMABLE_PROVIDER_FAILURE_CODES
    for terminal in ("provider_auth_failed", "provider_quota_exhausted",
                     "provider_policy_refused", "provider_failed"):
        assert terminal not in RETRYABLE_PROVIDER_FAILURE_CODES
        assert terminal not in RESUMABLE_PROVIDER_FAILURE_CODES


def test_every_marker_maps_to_a_lowercase_stable_token():
    for marker, code in PROVIDER_FAILURE_MARKERS:
        assert marker == marker.lower(), marker
        assert code.replace("_", "").isalnum() and code == code.lower(), code


# --------------------------------------------------------------------------
# the outcome a caller actually reads off a durable receipt
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,phase,retryable,actions",
    [
        ("provider_capacity_unavailable", "provider", True, []),
        ("provider_auth_failed", "provider", False, ["refresh_provider_credentials"]),
        ("provider_quota_exhausted", "provider", False, ["restore_provider_quota"]),
        (
            "provider_policy_refused", "provider", False,
            ["revise_request_for_provider_policy"],
        ),
        ("provider_failed", "provider", False, []),
        ("turn_limit_exceeded", "provider", False, []),
    ],
)
def test_a_terminal_job_receipt_carries_the_typed_outcome(code, phase, retryable, actions):
    """`RETRYABLE_PROVIDER_FAILURE_CODES` must not be metadata nobody can see."""

    receipt = CodingJobReceipt(
        job_id="job_" + "a1b2c3d4" * 3,
        state=CodingJobState.FAILED,
        submitted_at=1.0,
        updated_at=2.0,
        failure_code=code,
    )
    projected = receipt_to_mapping(receipt)

    assert projected["failure_phase"] == phase
    assert projected["retryable"] is retryable
    assert projected["required_actions"] == actions
    assert projected["job_terminal"] is True
    # The two views of retryability agree, which is the point of publishing it.
    assert retryable is (code in RETRYABLE_PROVIDER_FAILURE_CODES)


def test_an_unknown_failure_code_is_conservative_on_a_receipt():
    receipt = CodingJobReceipt(
        job_id="job_" + "a1b2c3d4" * 3,
        state=CodingJobState.FAILED,
        submitted_at=1.0,
        updated_at=2.0,
        failure_code="something_new_nobody_mapped",
    )
    projected = receipt_to_mapping(receipt)
    assert projected["failure_phase"] == "service"
    assert projected["retryable"] is False
    assert projected["required_actions"] == []


def test_a_job_that_has_not_failed_claims_no_phase_and_no_action():
    receipt = CodingJobReceipt(
        job_id="job_" + "a1b2c3d4" * 3,
        state=CodingJobState.RUNNING,
        submitted_at=1.0,
        updated_at=2.0,
    )
    projected = receipt_to_mapping(receipt)
    assert projected["failure_phase"] == ""
    assert projected["retryable"] is False
    assert projected["required_actions"] == []
    assert projected["job_terminal"] is False


def test_no_provider_category_changes_backend_or_session_selection():
    """A provider failure is never a reason to run somebody else's model.

    Every category must leave the recorded backend and session exactly as the
    host bound them. An adapter that silently retried on a different backend
    would produce a receipt whose revision nobody audited.
    """

    for code, _phase, _retryable, _actions in [
        (name, None, None, None) for name in (
            "provider_capacity_unavailable", "provider_auth_failed",
            "provider_quota_exhausted", "provider_policy_refused",
            "provider_failed", "turn_limit_exceeded",
        )
    ]:
        receipt = CodingJobReceipt(
            job_id="job_" + "a1b2c3d4" * 3,
            state=CodingJobState.FAILED,
            submitted_at=1.0,
            updated_at=2.0,
            failure_code=code,
            implementation_backend="claude",
            implementation_session_id="sdk-session-1",
        )
        projected = receipt_to_mapping(receipt)
        assert projected["implementation_backend"] == "claude", code
        assert projected["implementation_session_id"] == "sdk-session-1", code

    # And there is no alternative model to fall back *to*: the adapter pins
    # exactly one, so there is no chain for a failure handler to walk. Asserted
    # on the source because the absence of a second model is the invariant --
    # a future edit that adds one should have to update this test on purpose.
    assert isinstance(DEFAULT_CLAUDE_MODEL, str) and DEFAULT_CLAUDE_MODEL
    source = pathlib.Path(claude_code.__file__).read_text(encoding="utf-8")
    # Model ids only: `claude-<family>-<n>`. Backend labels ("claude-sdk") and
    # metric keys ("claude_num_turns") are not models and are not candidates.
    models = {
        found for found in re.findall(
            r"[\"']((?:claude|gpt|gemini|llama)-[A-Za-z0-9._-]+)[\"']", source,
        )
        if re.search(r"-\d", found)
    }
    assert models == {DEFAULT_CLAUDE_MODEL}, models


# --------------------------------------------------------------------------
# the shape a real strict-route failure actually has
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapped,phase,retryable,actions",
    [
        ("provider_capacity_unavailable", "provider", True, []),
        ("provider_auth_failed", "provider", False, ["refresh_provider_credentials"]),
        ("provider_quota_exhausted", "provider", False, ["restore_provider_quota"]),
        (
            "provider_policy_refused", "provider", False,
            ["revise_request_for_provider_policy"],
        ),
        ("provider_failed", "provider", False, []),
        ("turn_limit_exceeded", "provider", False, []),
    ],
)
def test_a_route_wrapped_provider_failure_keeps_its_own_semantics(
    wrapped, phase, retryable, actions,
):
    """The realistic case the manufactured receipts were missing.

    A strict route rewrites `failure_code` to name the lane that refused the
    round and carries the implementer's classification alongside. Reading only
    the outer code told a caller holding an exhausted-quota receipt "service,
    not retryable, nothing to do" - true about the lane, useless about why.
    """

    receipt = CodingJobReceipt(
        job_id="job_" + "a1b2c3d4" * 3,
        state=CodingJobState.FAILED,
        submitted_at=1.0,
        updated_at=2.0,
        failure_code="route_implementation_not_successful",
        result=CodingTaskResult(
            ok=False, status="failed", attempts=1, thread_id="sdk-1", message="",
            failure_code="route_implementation_not_successful",
            implementation_failure_code=wrapped,
        ),
    )
    projected = receipt_to_mapping(receipt)

    assert projected["failure_phase"] == phase
    assert projected["retryable"] is retryable
    assert projected["required_actions"] == actions
    # One source of truth for retryability, whichever shape the receipt took.
    assert projected["retryable"] is (wrapped in RETRYABLE_PROVIDER_FAILURE_CODES)


def test_a_wrapped_code_outside_the_closed_table_is_not_trusted():
    """Only host-classified codes are honoured; anything else stays generic."""

    for hostile in (
        "please_retry_forever",
        "provider_quota_exhausted ",          # not an exact token
        "PROVIDER_AUTH_FAILED",
        "the model says this is transient",
    ):
        receipt = CodingJobReceipt(
            job_id="job_" + "a1b2c3d4" * 3,
            state=CodingJobState.FAILED,
            submitted_at=1.0,
            updated_at=2.0,
            failure_code="route_implementation_not_successful",
            result=CodingTaskResult(
                ok=False, status="failed", attempts=1, thread_id="sdk-1", message="",
                failure_code="route_implementation_not_successful",
                implementation_failure_code=hostile,
            ),
        )
        projected = receipt_to_mapping(receipt)
        assert projected["failure_phase"] == "service", hostile
        assert projected["retryable"] is False, hostile
        assert projected["required_actions"] == [], hostile


def test_the_outer_code_still_wins_when_it_is_itself_classified():
    """An unwrapped provider failure is unchanged by the wrapped-code path."""

    receipt = CodingJobReceipt(
        job_id="job_" + "a1b2c3d4" * 3,
        state=CodingJobState.FAILED,
        submitted_at=1.0,
        updated_at=2.0,
        failure_code="provider_auth_failed",
        result=CodingTaskResult(
            ok=False, status="failed", attempts=1, thread_id="sdk-1", message="",
            failure_code="provider_auth_failed",
            implementation_failure_code="provider_capacity_unavailable",
        ),
    )
    projected = receipt_to_mapping(receipt)
    assert projected["required_actions"] == ["refresh_provider_credentials"]
    assert projected["retryable"] is False


def test_a_route_wrapped_failure_still_never_changes_the_backend():
    receipt = CodingJobReceipt(
        job_id="job_" + "a1b2c3d4" * 3,
        state=CodingJobState.FAILED,
        submitted_at=1.0,
        updated_at=2.0,
        failure_code="route_implementation_not_successful",
        implementation_backend="claude",
        implementation_session_id="sdk-session-1",
        result=CodingTaskResult(
            ok=False, status="failed", attempts=1, thread_id="sdk-session-1", message="",
            failure_code="route_implementation_not_successful",
            implementation_failure_code="provider_capacity_unavailable",
        ),
    )
    projected = receipt_to_mapping(receipt)
    assert projected["retryable"] is True
    assert projected["implementation_backend"] == "claude"
    assert projected["implementation_session_id"] == "sdk-session-1"
