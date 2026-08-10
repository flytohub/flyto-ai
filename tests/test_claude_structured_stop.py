# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""A bounded provider stop is a real round, not a round that never happened.

Job job_4ad75034fe97417f93f39f67 ran the repaired compound pre lane, passed all
four scoped gates, spent about six minutes inside Claude, and was then recorded
as ``provider_failed`` with ``attempts=0``, ``rounds=0``, no session and no
required action.  Everything about that record is false except the failure.

The cause is a category error about where the provider says why it stopped.
The SDK emits an ``is_error`` ``ResultMessage`` carrying the subtype, the
session, the turn count and the usage, and *then* exits non-zero.  The host was
classifying the trailing exception's text - a rendering of the same event, which
can be absent, translated or reworded - and knew only ``error_max_turns``.  The
configured ``$5`` ceiling produced ``error_max_budget_usd``, matched nothing, and
so the session was dropped and the work stranded as unauditable.

Two distinctions this module exists to hold:

*A configured ceiling is not an account quota.*  Quota is capacity the provider
account does not have and somebody fixes with a purchase.  This is a number the
operator told this host to enforce, and the answer is a decision about how much
one job may spend.  Same terminality, different person, different work - so a
different code and a different required action.

*Bounded is not empty.*  A stop after a real session began keeps the session,
the attempt, the turns and the usage.  It still claims nothing: no checks, no
verification, no auditability, no landability.
"""
import asyncio
import types

import pytest

from flyto_ai.agents.claude_code import (
    RESUMABLE_PROVIDER_FAILURE_CODES,
    STRUCTURED_STOP_SUBTYPES,
    bounded_turns,
    bounded_usage,
    provider_failure_code,
    structured_stop_code,
)
from flyto_ai.coding.contracts import (
    ACTION_ADJUST_CODING_JOB_BUDGET,
    JOB_FAILURE_SEMANTICS,
)

_BUDGET = "provider_job_budget_exhausted"
_TURNS = "turn_limit_exceeded"
_SESSION = "3f2b0c18-9d4a-4e77-9f61-2c5a7b0e1d43"


def _fields(**overrides):
    fields = {
        "is_error": True,
        "subtype": "error_max_budget_usd",
        "session_id": _SESSION,
        "num_turns": 37,
        "usage": {"input_tokens": 120, "output_tokens": 45},
        "total_cost_usd": 5.0,
        "duration_ms": 361_000,
        "result": "",
    }
    fields.update(overrides)
    return fields


def _result_message(**overrides):
    """A duck-typed stand-in, for the classifier's own unit tests."""

    return types.SimpleNamespace(**_fields(**overrides))


def _sdk_result(**overrides):
    """The real SDK class, so the production ``isinstance`` branch is entered.

    Using a look-alike here would have been the whole bug in miniature: the
    production code dispatches on type, so a stand-in silently skips the branch
    under test and the test passes for the wrong reason.
    """

    import claude_agent_sdk as sdk

    fields = _fields(**overrides)
    return sdk.ResultMessage(
        subtype=fields["subtype"],
        duration_ms=fields["duration_ms"],
        duration_api_ms=0,
        is_error=fields["is_error"],
        num_turns=fields["num_turns"],
        session_id=fields["session_id"],
        total_cost_usd=fields["total_cost_usd"],
        usage=fields["usage"],
        result=fields["result"],
    )


# --------------------------------------------------------------------------
# the structured signal itself
# --------------------------------------------------------------------------


def test_the_budget_subtype_is_recognised_and_is_not_quota():
    assert structured_stop_code(_result_message()) == _BUDGET
    assert STRUCTURED_STOP_SUBTYPES["error_max_budget_usd"] == _BUDGET
    assert STRUCTURED_STOP_SUBTYPES["error_max_turns"] == _TURNS
    # The two conditions are answered by different people doing different work.
    assert _BUDGET != "provider_quota_exhausted"


def test_the_turn_subtype_keeps_its_existing_meaning():
    assert structured_stop_code(_result_message(subtype="error_max_turns")) == _TURNS


@pytest.mark.parametrize(
    "message",
    [
        _result_message(is_error=False),
        _result_message(is_error="yes"),          # truthy is not an assertion
        _result_message(is_error=1),
        _result_message(subtype="error_unknown_future_thing"),
        _result_message(subtype=""),
        _result_message(subtype=None),
        _result_message(subtype=123),
        _result_message(subtype="a" * 200),
        types.SimpleNamespace(),
    ],
)
def test_an_unrecognised_or_hostile_stop_is_not_guessed(message):
    """Closed by construction: an unknown subtype gets no category at all."""

    assert structured_stop_code(message) == ""


def test_a_hostile_attribute_cannot_become_a_category():
    class _Hostile:
        @property
        def is_error(self):
            raise RuntimeError("boom")

    assert structured_stop_code(_Hostile()) == ""


@pytest.mark.parametrize(
    "value,expected",
    [(37, 37), (0, 0), (-1, 0), (True, 0), ("37", 0), (None, 0), (10 ** 9, 0), (1.5, 0)],
)
def test_turn_counts_are_bounded_integers_or_the_host_fallback(value, expected):
    assert bounded_turns(value) == expected


def test_usage_keeps_only_safe_integer_counters():
    assert bounded_usage({"input_tokens": 5, "output_tokens": 7}) == {
        "input_tokens": 5, "output_tokens": 7,
    }
    # Prose, floats, hostile names and absurd magnitudes are dropped whole.
    assert bounded_usage({
        "model": "claude-x", "cost_usd": 5.0, "bad key!": 1,
        "input_tokens": 10 ** 12, "ok_tokens": 3,
    }) == {"ok_tokens": 3}
    assert bounded_usage(None) == {}
    assert bounded_usage("input_tokens=5") == {}


def test_the_budget_stop_keeps_its_session_and_is_resumable():
    assert _BUDGET in RESUMABLE_PROVIDER_FAILURE_CODES
    assert _TURNS in RESUMABLE_PROVIDER_FAILURE_CODES


def test_the_exception_text_path_still_works_as_a_fallback():
    """Kept, but no longer the only signal - and the budget text is known now."""

    assert provider_failure_code(Exception("error_max_budget_usd")) == _BUDGET
    assert provider_failure_code(Exception("Reached maximum budget")) == _BUDGET
    assert provider_failure_code(Exception("error_max_turns")) == _TURNS
    assert provider_failure_code(Exception("credit balance is too low")) == (
        "provider_quota_exhausted"
    )
    assert provider_failure_code(Exception("something nobody has seen")) == "provider_failed"


# --------------------------------------------------------------------------
# the production stream, end to end through _run_claude_code
# --------------------------------------------------------------------------


def _drive(monkeypatch, tmp_path, messages, raises=None, max_turns=100):
    """Run the real `_run_claude_code` over a production-shaped message stream."""

    import claude_agent_sdk
    from flyto_ai.agents import claude_code as cc
    from flyto_ai.agents.models import CodeTaskRequest

    async def fake_query(prompt=None, options=None):
        for message in messages:
            yield message
        if raises is not None:
            raise raises

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    agent = cc.ClaudeCodeAgent.__new__(cc.ClaudeCodeAgent)
    agent._cc = types.SimpleNamespace(
        model="", permission_mode="acceptEdits", max_turns=max_turns,
        verification_timeout=30, system_prompt="", allowed_tools=None,
        max_budget_usd=5.0,
    )
    request = CodeTaskRequest(
        message="do the work", working_dir=str(workspace),
        max_fix_attempts=1, max_turns=max_turns,
        service_mode=True, service_edit_authority=True,
    )
    from flyto_ai.agents.evidence import EvidenceCollector

    evidence = EvidenceCollector("test-session", base_dir=str(tmp_path / "evidence"))
    return asyncio.run(agent._run_claude_code(
        request=request, indexer_context="", feedback="", session_id=None,
        max_budget=5.0, max_turns=max_turns, evidence=evidence, on_stream=None,
    ))


def _stream(subtype="error_max_budget_usd"):
    import claude_agent_sdk as sdk

    init = sdk.SystemMessage(subtype="init", data={"session_id": _SESSION})
    assistant = sdk.AssistantMessage(
        content=[sdk.TextBlock(text="working on it")], model="claude",
    )
    return [init, assistant, _sdk_result(subtype=subtype)]


@pytest.mark.parametrize(
    "subtype,code",
    [("error_max_budget_usd", _BUDGET), ("error_max_turns", _TURNS)],
)
def test_a_structured_stop_preserves_the_whole_round(tmp_path, monkeypatch, subtype, code):
    """The exact live shape: init, work, structured stop, then the SDK's exit."""

    secret = "sk-live-do-not-log-me AND /private/path/to/prompt"
    result = _drive(
        monkeypatch, tmp_path, _stream(subtype),
        raises=Exception("ProcessError: command failed {}".format(secret)),
    )
    monkeypatch.undo()

    assert result["incomplete_code"] == code
    # The session the provider established really survives the trailing failure.
    assert result["session_id"] == _SESSION
    assert result["num_turns"] >= 1
    assert result["usage"] == {"input_tokens": 120, "output_tokens": 45}
    # No provider prose, path or credential crosses the boundary.
    rendered = repr(result)
    for forbidden in ("sk-live", "/private/path", "ProcessError", "command failed"):
        assert forbidden not in rendered, forbidden


def test_a_turn_stop_still_reports_the_host_ceiling(tmp_path, monkeypatch):
    """Previously accepted behaviour, unchanged: the host's own bound is the truth."""

    messages = _stream("error_max_turns")
    messages[-1] = _sdk_result(subtype="error_max_turns", num_turns=0)
    result = _drive(
        monkeypatch, tmp_path, messages,
        raises=Exception("error_max_turns"), max_turns=100,
    )
    monkeypatch.undo()

    assert result["incomplete_code"] == _TURNS
    assert result["num_turns"] == 100


def test_a_stop_before_any_init_never_fabricates_a_session(tmp_path, monkeypatch):
    """Nothing was established, so nothing may be claimed or resumed."""

    with pytest.raises(Exception):
        _drive(
            monkeypatch, tmp_path, [],
            raises=Exception("error_max_budget_usd"),
        )
    monkeypatch.undo()


def test_an_unknown_subtype_with_a_hostile_exception_stays_terminal(tmp_path, monkeypatch):
    """No structured category and no recognised text: the round is not recovered."""

    messages = _stream("error_something_new")
    with pytest.raises(Exception):
        _drive(
            monkeypatch, tmp_path, messages,
            raises=Exception("totally unrecognised provider explosion"),
        )
    monkeypatch.undo()


def test_an_unsafe_session_is_never_resumed(tmp_path, monkeypatch):
    """A structured stop is not enough; the session must also be a safe one."""

    import claude_agent_sdk as sdk

    unsafe = "not a safe session id"
    init = sdk.SystemMessage(subtype="init", data={"session_id": unsafe})
    messages = [init, _sdk_result(session_id=unsafe)]
    with pytest.raises(Exception):
        _drive(
            monkeypatch, tmp_path, messages,
            raises=Exception("error_max_budget_usd"),
        )
    monkeypatch.undo()


def test_malformed_numeric_and_usage_fields_do_not_invent_evidence(tmp_path, monkeypatch):
    messages = _stream()
    messages[-1] = _sdk_result(
        num_turns="lots", usage={"input_tokens": "many", "model": "claude"},
    )
    result = _drive(
        monkeypatch, tmp_path, messages, raises=Exception("error_max_budget_usd"),
    )
    monkeypatch.undo()

    assert result["incomplete_code"] == _BUDGET
    assert result["num_turns"] == 0          # not guessed, not carried through
    assert bounded_usage(result["usage"]) == {}


# --------------------------------------------------------------------------
# what a caller is told
# --------------------------------------------------------------------------


def test_the_public_semantics_separate_a_job_budget_from_an_account_quota():
    budget = JOB_FAILURE_SEMANTICS[_BUDGET]
    quota = JOB_FAILURE_SEMANTICS["provider_quota_exhausted"]

    assert budget[0] == "provider" and budget[1] is False
    assert budget[2] == (ACTION_ADJUST_CODING_JOB_BUDGET,)
    # Same phase and terminality; different work, so different actions.
    assert quota[1] is False
    assert budget[2] != quota[2]
    # Not retryable, and no fallback backend is ever implied by either.
    assert budget[1] is False
