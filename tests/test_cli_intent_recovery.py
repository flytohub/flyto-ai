"""Malformed inference may be corrected once; it never authorizes a batch."""

import asyncio
import json

import pytest

from flyto_ai.cli_runtime.contracts import INTENT_SCHEMA, MAX_CALLS, CliRuntimeConfig, CliRuntimeError, checked_intent
from flyto_ai.cli_runtime.transport import CliTransport


TOOLS = [{"name": "read_fixture", "inputSchema": {"type": "object"}}]


def intent(*calls, content=""):
    return {"content": content, "tool_calls": list(calls)}


def call(arguments, name="read_fixture"):
    return {"name": name, "arguments_json": arguments}


async def drive(responses, *, rounds=6, timeout=5):
    pending = iter(responses)
    prompts, dispatched = [], []

    async def complete(**request):
        prompts.append(json.loads(request["prompt"]))
        answer = next(pending)
        if isinstance(answer, Exception):
            raise answer
        return json.dumps(answer)

    async def dispatch(name, arguments):
        dispatched.append((name, arguments))
        return {"ok": True, "observed": arguments}

    transport = CliTransport(CliRuntimeConfig("codex_cli", timeout_seconds=timeout), completion_fn=complete)
    try:
        result = await transport.chat([{"role": "user", "content": "Read current fixture"}], "Host authority", TOOLS, dispatch, max_rounds=rounds)
    finally:
        await transport.close()
    return transport, result, prompts, dispatched


def test_wire_schema_matches_checker_bounds():
    properties = INTENT_SCHEMA["properties"]
    assert properties["content"]["maxLength"] == 50_000
    assert properties["tool_calls"]["maxItems"] == MAX_CALLS
    assert properties["tool_calls"]["items"]["properties"]["arguments_json"]["maxLength"] == 65_536


@pytest.mark.parametrize(("value", "reason"), [
    ({"content": "", "tool_calls": [], "receipt": "private-value"}, "intent_shape"),
    (intent(content=1), "content_bounds"),
    ({"content": "", "tool_calls": {}}, "calls_shape"),
    (intent(*[call("{}") for _ in range(MAX_CALLS + 1)]), "call_limit"),
    (intent({"name": "read_fixture", "arguments_json": "{}", "extra": "private-value"}), "call_shape"),
    (intent(call({})), "arguments_bounds"),
    (intent(call('{"value":"private-value"')), "arguments_json"),
    (intent(call('{"value":1,"value":2}')), "arguments_json"),
    (intent(call("[]")), "arguments_object"),
])
def test_validation_reasons_are_fixed_and_do_not_echo_output(value, reason):
    with pytest.raises(CliRuntimeError) as failure:
        checked_intent(value, {"read_fixture"})
    assert failure.value.code == "cli_invalid_output"
    assert getattr(failure.value, "reason", None) == reason
    assert str(failure.value) == "cli_invalid_output"
    assert "private-value" not in repr(failure.value)


@pytest.mark.asyncio
async def test_invalid_batch_has_zero_effects_then_one_correction_uses_prior_observations(caplog):
    malformed = intent(call('{"row":"one"}'), call('{"private-value":'))
    transport, result, prompts, dispatched = await drive([
        intent(call('{"row":"one"}')), malformed,
        intent(call('{"row":"two"}')), intent(content="Observed current rows"),
    ])
    assert transport.last_error is None
    assert result[0] == "Observed current rows" and result[2] == 4
    assert dispatched == [("read_fixture", {"row": "one"}), ("read_fixture", {"row": "two"})]
    correction = prompts[2]["messages"]
    assert any(message.get("role") == "tool" and "one" in message["content"] for message in correction)
    assert "arguments_json" in correction[-1]["content"]
    assert "private-value" not in json.dumps(correction)
    assert "arguments_json" in caplog.text and "private-value" not in caplog.text


@pytest.mark.asyncio
async def test_only_one_correction_is_allowed_and_failed_batch_is_never_dispatched():
    invalid = intent(call("[]"))
    transport, _, prompts, dispatched = await drive([invalid, invalid, intent(content="Must not be reached")])
    assert transport.last_error == "cli_invalid_output"
    assert len(prompts) == 2 and dispatched == []


@pytest.mark.asyncio
async def test_correction_cannot_exceed_existing_round_budget():
    transport, _, prompts, dispatched = await drive([intent(call("[]")), intent(content="Must not be reached")], rounds=1)
    assert transport.last_error == "cli_round_budget_exhausted"
    assert len(prompts) == 1 and dispatched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["cli_native_action_refused", "cli_native_tools_exposed", "cli_session_changed", "cli_quota_exhausted", "cli_auth_required", "cli_invalid_output"])
async def test_transport_authority_and_provider_failures_are_not_format_retries(code):
    transport, _, prompts, dispatched = await drive([CliRuntimeError(code), intent(content="Must not be reached")])
    assert transport.last_error == code
    assert len(prompts) == 1 and dispatched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("valid_first", ["{}", "[]", '{"private-value":'])
async def test_unknown_tool_in_batch_is_never_retried_or_partially_dispatched(valid_first):
    transport, _, prompts, dispatched = await drive([intent(call(valid_first), call("{}", name="unauthorized")), intent(content="Must not be reached")])
    assert transport.last_error == "cli_tool_not_available"
    assert len(prompts) == 1 and dispatched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("envelope", [
    intent(call("{}", name="unauthorized"), content=1),
    {**intent(call("{}", name="unauthorized")), "extra": "private-value"},
    {"tool_calls": [call("{}", name="unauthorized")]},
])
async def test_invalid_envelope_cannot_turn_unknown_tool_into_format_retry(envelope):
    transport, _, prompts, dispatched = await drive([envelope, intent(content="Must not be reached")])
    assert transport.last_error == "cli_tool_not_available"
    assert len(prompts) == 1 and dispatched == []


@pytest.mark.asyncio
async def test_cancel_during_correction_cannot_dispatch_late_response():
    entered = asyncio.Event()
    calls, dispatched = [], []

    async def complete(**request):
        calls.append(request)
        if len(calls) == 1:
            return json.dumps(intent(call("[]")))
        entered.set()
        await asyncio.Event().wait()

    async def dispatch(*args):
        dispatched.append(args)

    transport = CliTransport(CliRuntimeConfig("codex_cli"), completion_fn=complete)
    task = asyncio.create_task(transport.chat([{"role": "user", "content": "Read"}], "", TOOLS, dispatch))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await transport.close()
    assert dispatched == [] and len(calls) == 2


@pytest.mark.asyncio
async def test_correction_uses_remaining_deadline_and_counts_both_inferences(monkeypatch):
    from types import SimpleNamespace
    import flyto_ai.cli_runtime.transport as runtime

    clock = [100.0]
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    real_wait_for = asyncio.wait_for
    timeouts, dispatched = [], []

    async def bounded_wait(awaitable, timeout):
        timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(runtime.asyncio, "wait_for", bounded_wait)
    responses = iter([
        (intent(call("[]")), {"input_tokens": 3}, 3.0),
        (intent(call("{}")), {"input_tokens": 7}, 2.1),
    ])

    async def infer(_prompt):
        value, usage, duration = next(responses)
        clock[0] += duration
        return value, usage

    async def dispatch(*args):
        dispatched.append(args)

    transport = CliTransport(CliRuntimeConfig("codex_cli", timeout_seconds=5), completion_fn=infer)
    monkeypatch.setattr(transport, "_infer", infer)
    try:
        result = await transport.chat([{"role": "user", "content": "Read"}], "", TOOLS, dispatch)
    finally:
        await transport.close()
    assert timeouts == [5.0, 2.0]
    assert result[2:] == (2, {"input_tokens": 10})
    assert transport.last_error == "cli_timeout" and dispatched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds", [1, 30])
async def test_expired_original_deadline_cannot_start_correction(monkeypatch, rounds):
    from types import SimpleNamespace
    import flyto_ai.cli_runtime.transport as runtime

    clock, requests, dispatched = [100.0], [], []
    monkeypatch.setattr(runtime, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    async def complete(**request):
        requests.append(request)
        clock[0] += 6.0
        return json.dumps(intent(call("[]")))

    async def dispatch(*args):
        dispatched.append(args)

    transport = CliTransport(CliRuntimeConfig("codex_cli", timeout_seconds=5), completion_fn=complete)
    try:
        await transport.chat([{"role": "user", "content": "Read"}], "", TOOLS, dispatch, max_rounds=rounds)
    finally:
        await transport.close()
    assert transport.last_error == "cli_timeout"
    assert len(requests) == 1 and dispatched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("second_invalid", [False, True])
async def test_admitted_slice_continuation_preserves_one_correction_and_actual_actions(second_invalid):
    from flyto_ai.cli_runtime import CliAgent
    from flyto_ai.config import AgentConfig

    def read(name):
        return call(json.dumps({"module_id": "file.read", "params": {"path": name}}), name="execute_module")

    invalid = call('{"private-value":', name="execute_module")
    # The entire invalid batch, including its valid first proposal, stays inert.
    responses = ([intent(read("one")), intent(invalid), intent(read("two")), intent(read("three")),
                  intent(invalid), intent(content="Must not be reached")]
                 if second_invalid else
                 [intent(read("one")), intent(read("two")), intent(read("three")),
                  intent(read("four"), invalid), intent(read("four")), intent(content="Observed four records")])
    answers = iter(responses)
    prompts, dispatched = [], []

    async def complete(**request):
        prompts.append(json.loads(request["prompt"]))
        return json.dumps(next(answers))

    async def dispatch(name, arguments):
        dispatched.append(arguments["params"]["path"])
        return {"ok": True, "data": {"observed": dispatched[-1]}}

    agent = CliAgent(
        AgentConfig(max_tool_rounds=4, enable_memory=False, enable_pro=False, enable_transcript=False),
        cli=CliRuntimeConfig("codex_cli"), completion_fn=complete,
        tools=[{"name": "execute_module", "inputSchema": {"type": "object"}}], dispatch_fn=dispatch,
        policies={"allowed_tools": ["execute_module"], "allowed_categories": ["file"]},
    )
    agent._assistant = None
    goal = "Read the four local workspace records."
    try:
        first = await agent.start_execution(goal)
        assert first.error == "cli_round_budget_exhausted"
        assert first.rounds_used == 4
        assert dispatched == ["one", "two", "three"]
        assert len(prompts) == 4
        agent.config.max_tool_rounds = 2
        second = await agent.continue_execution("Continue from observed records.", goal=goal)
        if second_invalid:
            assert second.error == "cli_invalid_output"
            assert second.rounds_used == 1
            assert dispatched == ["one", "two", "three"]
        else:
            assert second.ok
            assert second.rounds_used == 2
            assert dispatched == ["one", "two", "three", "four"]
            assert any(message.get("role") == "user" and "arguments_json" in str(message.get("content"))
                       for message in prompts[4]["messages"])
        assert first.rounds_used + second.rounds_used <= 6
        assert "private-value" not in json.dumps(prompts)
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_successful_host_call_does_not_renew_format_correction_allowance():
    transport, result, prompts, dispatched = await drive([
        intent(call("[]")), intent(call('{"row":"one"}')),
        intent(call("[]")), intent(content="Must not be reached"),
    ])
    assert transport.last_error == "cli_invalid_output"
    assert result[0] is None and result[2] == 3
    assert len(prompts) == 3 and dispatched == [("read_fixture", {"row": "one"})]


@pytest.mark.asyncio
async def test_closed_transport_cancels_correction_without_dispatch():
    entered = asyncio.Event()
    requests, dispatched = [], []

    async def complete(**request):
        requests.append(request)
        if len(requests) == 1:
            return json.dumps(intent(call("[]")))
        entered.set()
        await asyncio.Event().wait()

    async def dispatch(*args):
        dispatched.append(args)

    transport = CliTransport(CliRuntimeConfig("codex_cli"), completion_fn=complete)
    task = asyncio.create_task(transport.chat([{"role": "user", "content": "Read"}], "", TOOLS, dispatch))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await transport.close()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await transport.close()
    assert len(requests) == 2 and dispatched == []
