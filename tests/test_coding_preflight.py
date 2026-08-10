# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Repository verification feasibility, decided before a job can exist.

Every case here is about the same question asked at the earliest possible
moment: *can a change to this repository be verified at all?*  The point of
moving it in front of the Indexer lane, the worktree claim and the implementer
session is that a "no" costs nothing and leaves nothing behind, so these tests
assert on the absence of side effects as much as on the code that came back.
"""
import sys
from pathlib import Path

import pytest

from flyto_ai.coding.preflight import (
    ACTION_ADD_REPOSITORY_VERIFICATION_CONTRACT,
    ACTION_DECLARE_REQUIRED_VERIFICATION_CHECK,
    ACTION_FIX_REPOSITORY_VERIFICATION_CONTRACT,
    ACTION_INSTALL_REQUIRED_CAPABILITY,
    CODE_CAPABILITY_UNAVAILABLE,
    CODE_VERIFICATION_CONTRACT_INVALID,
    CODE_VERIFICATION_REQUIRED,
    FAILURE_PHASE_PREFLIGHT,
    PREFLIGHT_ACTIONS,
    PreflightOutcome,
    preflight_repository,
)
from flyto_ai.coding.service import (
    CapabilityUnavailable,
    CodingServiceError,
    VerificationContractInvalid,
    VerificationRequired,
    error_details,
)

_VALID = """version: flyto.coding-config.v1
checks:
  - name: declared
    argv: [python, --version]
    timeout_seconds: 30
    required: true
"""


def _write(workspace: Path, text: str) -> Path:
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(text, encoding="utf-8")
    return config


# --------------------------------------------------------------------------
# the decision itself
# --------------------------------------------------------------------------


def test_a_declared_contract_passes_and_carries_its_checks(tmp_path):
    _write(tmp_path, _VALID)
    outcome = preflight_repository(str(tmp_path))

    assert outcome.ok is True
    assert outcome.code == ""
    assert outcome.required_actions == ()
    assert [check.name for check in outcome.checks] == ["declared"]


def test_no_contract_at_all_is_verification_required(tmp_path):
    outcome = preflight_repository(str(tmp_path))

    assert outcome.ok is False
    assert outcome.code == CODE_VERIFICATION_REQUIRED
    assert outcome.required_actions == (ACTION_ADD_REPOSITORY_VERIFICATION_CONTRACT,)


def test_a_contract_with_no_required_check_is_verification_required(tmp_path):
    """Declared but toothless is still undeclared, and says so more precisely."""

    _write(tmp_path, _VALID.replace("required: true", "required: false"))
    outcome = preflight_repository(str(tmp_path))

    assert outcome.code == CODE_VERIFICATION_REQUIRED
    # The narrower action, so "add a contract" stays honest for repositories
    # that genuinely have none.
    assert outcome.required_actions == (ACTION_DECLARE_REQUIRED_VERIFICATION_CHECK,)


@pytest.mark.parametrize(
    "text",
    [
        "version: flyto.coding-config.v1\nchecks: [",          # not YAML
        "just a string",                                        # not an object
        "version: some.other.contract.v9\nchecks: []\n",        # unknown version
        "version: flyto.coding-config.v1\nsurprise: 1\n",       # unsupported key
        _VALID.replace("timeout_seconds: 30", "timeout_seconds: 99999"),
        _VALID.replace("name: declared", "name: 'not a safe id'"),
    ],
)
def test_an_unhonourable_contract_is_distinct_from_a_missing_one(tmp_path, text):
    """"You wrote none" and "yours is wrong" are different jobs for different people."""

    _write(tmp_path, text)
    outcome = preflight_repository(str(tmp_path))

    assert outcome.ok is False
    assert outcome.code == CODE_VERIFICATION_CONTRACT_INVALID
    assert outcome.required_actions == (ACTION_FIX_REPOSITORY_VERIFICATION_CONTRACT,)


def test_a_required_capability_that_cannot_launch_is_its_own_code(tmp_path):
    """The contract is fine; the host is not. Never send that to a contract editor."""

    _write(tmp_path, _VALID + """capabilities:
  - name: absent_tool
    argv: [flyto-tool-that-is-not-installed-anywhere]
    required: true
""")
    outcome = preflight_repository(str(tmp_path))

    assert outcome.code == CODE_CAPABILITY_UNAVAILABLE
    assert outcome.required_actions == (ACTION_INSTALL_REQUIRED_CAPABILITY,)


def test_a_backend_with_no_bridge_makes_a_resolvable_capability_infeasible(tmp_path):
    """Installed is not attachable, and preflight must not confuse the two.

    This is the exact gap that made preflight a lie: `argv[0]` resolved, so
    preflight said yes, and then the selected backend refused the very same
    contract because it has no capability bridge at all.
    """

    _write(tmp_path, _VALID + """capabilities:
  - name: resolvable
    argv: [{}]
    required: true
""".format(sys.executable))

    # Resolvable, but the backend declares it can bridge nothing.
    assert preflight_repository(
        str(tmp_path), attachable_capability_kinds=frozenset(),
    ).code == CODE_CAPABILITY_UNAVAILABLE

    # Unproven is treated exactly like "nothing": assuming otherwise is what
    # produced a green light in front of a guaranteed refusal.
    assert preflight_repository(str(tmp_path)).code == CODE_CAPABILITY_UNAVAILABLE

    # A backend that really does bridge this kind passes.
    assert preflight_repository(
        str(tmp_path), attachable_capability_kinds=frozenset({"mcp-stdio"}),
    ).ok is True


def test_the_claude_backend_declares_the_bridge_it_actually_has(tmp_path):
    """Preflight's refusal and the adapter's refusal must have the same cause.

    The adapter refuses every required capability the moment it reads the
    contract. Preflight is only honest if it reaches that same verdict first,
    from the backend's own declaration rather than from a hardcoded name.
    """

    from flyto_ai.agents.claude_code import ClaudeCodingAgent
    from flyto_ai.coding.agent import FlytoCodingAgent

    assert ClaudeCodingAgent.attachable_capability_kinds == frozenset()
    # The native implementer drives a real capability manager, so it declares
    # the kinds it can genuinely bridge. The two backends must not be equal, or
    # the declaration is not carrying any information.
    assert FlytoCodingAgent.attachable_capability_kinds
    assert (
        FlytoCodingAgent.attachable_capability_kinds
        != ClaudeCodingAgent.attachable_capability_kinds
    )

    _write(tmp_path, _VALID + """capabilities:
  - name: needs_bridge
    argv: [{}]
    required: true
""".format(sys.executable))

    outcome = preflight_repository(
        str(tmp_path),
        attachable_capability_kinds=ClaudeCodingAgent.attachable_capability_kinds,
    )
    assert outcome.ok is False
    assert outcome.code == CODE_CAPABILITY_UNAVAILABLE
    assert outcome.required_actions == (ACTION_INSTALL_REQUIRED_CAPABILITY,)


def test_an_optional_missing_capability_does_not_refuse(tmp_path):
    _write(tmp_path, _VALID + """capabilities:
  - name: absent_tool
    argv: [flyto-tool-that-is-not-installed-anywhere]
    required: false
""")
    assert preflight_repository(str(tmp_path)).ok is True


def test_preflight_reads_and_creates_nothing(tmp_path):
    """A refusal must not be a side effect; it is only an answer."""

    before = sorted(path.name for path in tmp_path.iterdir())
    assert preflight_repository(str(tmp_path)).ok is False
    assert sorted(path.name for path in tmp_path.iterdir()) == before


# --------------------------------------------------------------------------
# the typed envelope callers actually observe
# --------------------------------------------------------------------------


def test_every_refusal_names_a_closed_allowlisted_action():
    for action in PREFLIGHT_ACTIONS:
        assert action.islower()
        assert " " not in action

    with pytest.raises(ValueError):
        PreflightOutcome(ok=False, code="made_up_code", required_actions=("x",))
    with pytest.raises(ValueError):
        PreflightOutcome(
            ok=False, code=CODE_VERIFICATION_REQUIRED, required_actions=("do_something",),
        )
    with pytest.raises(ValueError):
        # A refusal that names no action is unactionable, so it is not a
        # refusal this module is willing to produce.
        PreflightOutcome(ok=False, code=CODE_VERIFICATION_REQUIRED)
    with pytest.raises(ValueError):
        PreflightOutcome(ok=True, code=CODE_VERIFICATION_REQUIRED)


@pytest.mark.parametrize(
    "error,code,actions",
    [
        (
            VerificationRequired("x", (ACTION_ADD_REPOSITORY_VERIFICATION_CONTRACT,)),
            "verification_required",
            [ACTION_ADD_REPOSITORY_VERIFICATION_CONTRACT],
        ),
        (
            VerificationContractInvalid("x", (ACTION_FIX_REPOSITORY_VERIFICATION_CONTRACT,)),
            "verification_contract_invalid",
            [ACTION_FIX_REPOSITORY_VERIFICATION_CONTRACT],
        ),
        (
            CapabilityUnavailable("x", (ACTION_INSTALL_REQUIRED_CAPABILITY,)),
            "capability_unavailable",
            [ACTION_INSTALL_REQUIRED_CAPABILITY],
        ),
    ],
)
def test_preflight_errors_project_a_bounded_typed_envelope(error, code, actions):
    assert error.code == code
    assert isinstance(error, CodingServiceError)
    assert error_details(error) == {
        "failure_phase": FAILURE_PHASE_PREFLIGHT,
        "retryable": False,
        "required_actions": actions,
    }


def test_the_action_list_can_never_become_a_prose_channel():
    """Only allowlisted tokens survive the projection; the rest are dropped."""

    smuggled = VerificationRequired("x", (ACTION_ADD_REPOSITORY_VERIFICATION_CONTRACT,))
    smuggled.required_actions = (
        ACTION_ADD_REPOSITORY_VERIFICATION_CONTRACT,
        "/Users/someone/secret/path",
        "here is a whole sentence of model prose",
    )
    assert error_details(smuggled)["required_actions"] == [
        ACTION_ADD_REPOSITORY_VERIFICATION_CONTRACT,
    ]
