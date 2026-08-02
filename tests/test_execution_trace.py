# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Content-addressed replay and Blueprint feedback tests."""
from __future__ import annotations

import dataclasses

import pytest

from flyto_ai.coding.execution_trace import ExecutionTraceLedger


def _record(
    ledger, *, value=1, ok=True, dispatched=True, result=None,
    include_secret=False,
):
    arguments = {"value": value}
    if include_secret:
        arguments["api_key"] = "must-redact"
    return ledger.record(
        provider_name="cap_fixture_observe",
        remote_name="observe",
        required_permission="read_only",
        arguments=arguments,
        dispatched=dispatched,
        ok=ok,
        policy_code="allow" if dispatched else "approval_denied",
        result=result if result is not None else {"ok": ok, "value": value},
    )


def test_trace_hash_chain_is_deterministic_redacted_and_tamper_evident():
    first = ExecutionTraceLedger()
    second = ExecutionTraceLedger()
    event = _record(first, include_secret=True)
    _record(second, include_secret=True)
    assert event.arguments["api_key"] == "***"
    assert "must-redact" not in str(first.export())
    exported = first.export()
    exported["events"][0]["arguments"]["api_key"] = "mutated"
    assert first.export()["events"][0]["arguments"]["api_key"] == "***"
    with pytest.raises(TypeError):
        event.arguments["api_key"] = "mutated"
    assert first.fingerprint == second.fingerprint
    assert first.verify_chain() is True

    object.__setattr__(event, "result", {"ok": False})
    assert first.verify_chain() is False


def test_trace_event_budget_and_contract_validation_fail_closed():
    ledger = ExecutionTraceLedger(max_events=1)
    _record(ledger)
    with pytest.raises(RuntimeError, match="event budget"):
        _record(ledger, value=2)
    with pytest.raises(ValueError, match="provider_name"):
        ExecutionTraceLedger().record(
            provider_name="",
            remote_name="observe",
            required_permission="read_only",
            arguments={},
            dispatched=True,
            ok=True,
            policy_code="allow",
            result={},
        )
    with pytest.raises(ValueError, match="required_permission"):
        ExecutionTraceLedger().record(
            provider_name="cap_fixture_observe",
            remote_name="observe",
            required_permission="root",
            arguments={},
            dispatched=True,
            ok=True,
            policy_code="allow",
            result={},
        )
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="finite JSON"):
        ExecutionTraceLedger().record(
            provider_name="cap_fixture_observe",
            remote_name="observe",
            required_permission="read_only",
            arguments=cyclic,
            dispatched=True,
            ok=True,
            policy_code="allow",
            result={},
        )


@pytest.mark.asyncio
async def test_replay_contract_and_normalizer_failures_are_bounded():
    ledger = ExecutionTraceLedger()
    _record(ledger)
    with pytest.raises(ValueError, match="dispatch"):
        await ledger.replay(None)
    with pytest.raises(ValueError, match="normalizers"):
        await ledger.replay(lambda _name, _args: None, normalizers={"cap": None})
    with pytest.raises(ValueError, match="permissions"):
        await ledger.replay(lambda _name, _args: None, allowed_permissions=())

    async def dispatch(_name, _arguments):
        return {"ok": True, "value": 1}

    def broken(_result):
        raise RuntimeError("secret-normalizer-detail")

    report = await ledger.replay(
        dispatch, normalizers={"cap_fixture_observe": broken},
    )
    assert report.ok is False
    assert report.mismatches[0].reason == "normalizer_failed"
    assert "secret-normalizer-detail" not in str(report.as_dict())


@pytest.mark.asyncio
async def test_replay_matches_dispatched_events_and_skips_policy_denials():
    ledger = ExecutionTraceLedger()
    _record(ledger, value=1)
    _record(ledger, value=2, dispatched=False, ok=False, result={"ok": False})

    async def dispatch(_name, arguments):
        return {"ok": True, "value": arguments["value"]}

    report = await ledger.replay(dispatch)
    assert report.ok is True
    assert report.attempted == 1
    assert report.matched == 1
    assert report.skipped == 1


@pytest.mark.asyncio
async def test_replay_uses_a_fixed_snapshot_when_dispatch_appends_new_evidence():
    ledger = ExecutionTraceLedger()
    first = _record(ledger, value=1)

    async def dispatch(_name, arguments):
        _record(ledger, value=arguments["value"] + 1)
        return {"ok": True, "value": arguments["value"]}

    report = await ledger.replay(dispatch)
    assert report.ok is True
    assert report.attempted == 1
    assert report.trace_fingerprint == first.event_hash
    assert len(ledger.events) == 2


@pytest.mark.asyncio
async def test_replay_reports_mismatch_dispatch_failure_and_supports_normalizer():
    ledger = ExecutionTraceLedger()
    _record(ledger, result={"ok": True, "value": 1, "request_id": "original"})

    async def changed(_name, _arguments):
        return {"ok": True, "value": 1, "request_id": "new"}

    mismatch = await ledger.replay(changed)
    assert mismatch.ok is False
    assert mismatch.mismatches[0].reason == "result_mismatch"

    def remove_request_id(result):
        return {key: value for key, value in result.items() if key != "request_id"}

    matched = await ledger.replay(
        changed,
        normalizers={"cap_fixture_observe": remove_request_id},
    )
    assert matched.ok is True

    async def failed(_name, _arguments):
        raise RuntimeError("transient")

    failure = await ledger.replay(failed)
    assert failure.mismatches[0].reason == "dispatch_failed"


@pytest.mark.asyncio
async def test_replay_skips_redacted_and_non_read_only_calls_by_default():
    ledger = ExecutionTraceLedger()
    ledger.record(
        provider_name="cap_fixture_write",
        remote_name="write",
        required_permission="workspace_write",
        arguments={"value": 1},
        dispatched=True,
        ok=True,
        policy_code="allow",
        result={"ok": True},
    )
    _record(ledger, value=2, include_secret=True)
    calls = []

    async def dispatch(name, arguments):
        calls.append((name, arguments))
        if name == "cap_fixture_write":
            return {"ok": True}
        return {"ok": True, "value": arguments.get("value")}

    default = await ledger.replay(dispatch)
    assert default.attempted == 0
    assert default.skipped == 2
    assert calls == []

    explicit = await ledger.replay(
        dispatch, allowed_permissions=("workspace_write",),
    )
    assert explicit.attempted == 1
    assert explicit.matched == 1
    assert calls == [("cap_fixture_write", {"value": 1})]


@pytest.mark.asyncio
async def test_blueprint_feedback_is_trace_bound_redacted_and_idempotency_ready():
    ledger = ExecutionTraceLedger()
    _record(ledger)

    async def dispatch(_name, arguments):
        return {"ok": True, "value": arguments["value"]}

    replay = await ledger.replay(dispatch)
    published = []

    async def sink(payload):
        published.append(payload)
        return {"ok": True, "access_token": "never-store"}

    first = await ledger.publish_blueprint_outcome("bp_fixture", replay, sink)
    second = await ledger.publish_blueprint_outcome("bp_fixture", replay, sink)
    assert first.success is True
    assert first.execution_id == second.execution_id
    assert first.execution_id == "trace_{}".format(ledger.fingerprint[:24])
    assert first.sink_result["access_token"] == "***"
    assert published[0]["evidence"]["trace_fingerprint"] == ledger.fingerprint
    assert published[0]["evidence"]["replay"]["ok"] is True

    unrelated = dataclasses.replace(replay, trace_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="different execution trace"):
        await ledger.publish_blueprint_outcome("bp_fixture", unrelated, sink)

    sync_receipt = await ledger.publish_blueprint_outcome(
        "bp_fixture", replay, lambda _payload: {"ok": True},
    )
    assert sync_receipt.sink_result == {"ok": True}
    with pytest.raises(ValueError, match="blueprint_id"):
        await ledger.publish_blueprint_outcome("", replay, sink)
    with pytest.raises(ValueError, match="must be callable"):
        await ledger.publish_blueprint_outcome("bp_fixture", replay, None)
    with pytest.raises(ValueError, match="timeout"):
        await ledger.publish_blueprint_outcome(
            "bp_fixture", replay, sink, timeout_seconds=True,
        )

    def failed_sink(_payload):
        raise RuntimeError("secret-sink-detail")

    with pytest.raises(RuntimeError, match="sink failed") as failure:
        await ledger.publish_blueprint_outcome("bp_fixture", replay, failed_sink)
    assert "secret-sink-detail" not in str(failure.value)
