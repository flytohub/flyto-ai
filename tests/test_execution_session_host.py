from __future__ import annotations

import asyncio
import inspect
import json
import multiprocessing
import os
import threading
import time
from pathlib import Path

import pytest

from flyto_ai.execution_session import (
    EXECUTION_SESSION_REQUEST_VERSION,
    ExecutionAuthority,
    _MAX_TIMESTAMP_MS,
)
from flyto_ai.execution_session_host import (
    ExecutionSessionConnector,
    ExecutionSessionHostError,
    admit_execution_session,
)
from flyto_ai.scheduler import Scheduler


class _EmptyPathLike:
    def __fspath__(self) -> str:
        return ""


def _authority() -> ExecutionAuthority:
    return ExecutionAuthority(
        tenant_id="tenant.a",
        principal_id="principal.secret",
        verified=True,
        allowed_sources=("catalog.alpha",),
        allowed_domains=("data.workflow",),
        granted_permissions=("records.write",),
        enabled_capabilities=("save_response",),
    )


def _manifest() -> dict[str, object]:
    return {
        "manifest_contract": "flyto.capability-manifest.v1",
        "canonical_id": "data.workflow.save_response@1",
        "runtime_name": "save_response",
        "name": "save_response",
        "version": "1.0.0",
        "source": "catalog.alpha",
        "domain": "data.workflow",
        "description": "Save an already supplied response.",
        "control_class": "data",
        "required_permissions": ["records.write"],
        "intent_ids": ["response.save"],
        "affordances": ["record.write"],
        "effects": ["response.persisted"],
    }


def _request(session_id: str = "session.1") -> dict[str, object]:
    return {
        "contract_version": EXECUTION_SESSION_REQUEST_VERSION,
        "session_id": session_id,
        "space": {
            "space_id": "space.lobby",
            "display_name": "Lobby",
            "wake_words": [],
            "active_timeout_ms": 30_000,
        },
        "activation": {
            "source": "typed",
            "observed_wake_word": None,
            "activated_at_ms": 10_000,
            "expires_at_ms": 40_000,
        },
        "goal": {
            "text": "TOP SECRET goal: save the supplied response",
            "frame": {
                "contract_version": "flyto.goal-frame.v1",
                "intent_ids": ["response.save"],
                "required_affordances": ["record.write"],
                "desired_effects": ["response.persisted"],
                "trigger_events": [],
                "constraints": [],
            },
        },
    }


def _short_request(session_id: str = "session.short") -> dict[str, object]:
    request = _request(session_id)
    request["space"]["active_timeout_ms"] = 250  # type: ignore[index]
    request["activation"] = {
        "source": "typed",
        "observed_wake_word": None,
        "activated_at_ms": 19_990,
        "expires_at_ms": 20_240,
    }
    return request


async def _admit(root: Path, request: dict[str, object] | None = None):
    return await admit_execution_session(
        root, request or _request(), [_manifest()], _authority(), 20_000
    )


def _success() -> dict[str, object]:
    return {"ok": True, "message": "", "error": None, "cost_usd": 0.0}


def _failure() -> dict[str, object]:
    return {
        "ok": False,
        "message": "",
        "error": "execution_connector_failed",
        "cost_usd": 0.0,
    }


async def _wait_process_event(event, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not event.is_set():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("connector process did not signal in time")
        await asyncio.sleep(0.01)


def _live_connector_children() -> list[multiprocessing.Process]:
    return [child for child in multiprocessing.active_children() if child.is_alive()]


def _isolated(callback) -> ExecutionSessionConnector:
    return ExecutionSessionConnector(callback)


def _connector_async_tasks() -> list[asyncio.Task[object]]:
    return [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and "ExecutionSessionConnector" in task.get_coro().__qualname__
    ]


@pytest.mark.asyncio
async def test_stable_identity_truthful_block_and_secret_minimized_definition(tmp_path) -> None:
    receipt = await _admit(tmp_path)
    task = Scheduler(state_root=tmp_path).list_tasks()[0]
    definition = json.loads(task.instruction)

    assert receipt["status"] == "blocked"
    assert receipt["result"]["ok"] is False
    assert receipt["result"]["error"] == "execution_not_connected"
    assert receipt["session"]["session_id"] == "session.1"
    assert receipt["session"]["task_id"] == task.task_id
    assert task.task_id.startswith("execution-session-")
    assert set(definition) == {"contract_version", "session", "digests"}
    persisted = task.instruction.lower()
    for secret in ("top secret", "principal.secret", "save_response", "records.write"):
        assert secret not in persisted


@pytest.mark.asyncio
async def test_restart_and_duplicate_return_same_receipt_without_replay(tmp_path) -> None:
    first = await _admit(tmp_path)
    second = await _admit(tmp_path)
    assert second == first
    summary = Scheduler(state_root=tmp_path).summary()
    assert summary["task_count"] == 1
    assert summary["tasks"][0]["run_count"] == 1


@pytest.mark.asyncio
async def test_trusted_connector_connects_once_with_scheduler_evidence(tmp_path) -> None:
    calls = multiprocessing.get_context("fork").Value("i", 0)
    seen_path = tmp_path / "connector-input.json"

    async def connector(prepared):
        with calls.get_lock():
            calls.value += 1
        seen_path.write_text(json.dumps(prepared), encoding="utf-8")
        return _success()

    first = await admit_execution_session(
        tmp_path,
        _request(),
        [_manifest()],
        _authority(),
        20_000,
        trusted_connector=_isolated(connector),
    )
    second = await admit_execution_session(
        tmp_path,
        _request(),
        [_manifest()],
        _authority(),
        20_000,
        trusted_connector=_isolated(connector),
    )

    assert first == second
    assert first["status"] == "connected"
    assert first["result"]["ok"] is True
    assert first["result"]["evidence_ref"].startswith("sched-result-")
    seen = json.loads(seen_path.read_text(encoding="utf-8"))
    assert seen["planning_input"]["goal"]["text"].startswith("TOP SECRET")
    assert calls.value == 1


@pytest.mark.asyncio
async def test_connector_input_and_output_mutation_cannot_change_state(tmp_path) -> None:
    returned = _success()

    async def connector(prepared):
        prepared["planning_input"]["goal"]["text"] = "mutated"
        returned_result = returned
        return returned_result

    receipt = await admit_execution_session(
        tmp_path,
        _request(),
        [_manifest()],
        _authority(),
        20_000,
        trusted_connector=_isolated(connector),
    )
    returned["ok"] = False
    returned["error"] = "secret provider prose"
    assert (await _admit(tmp_path)) == receipt
    durable = json.dumps(Scheduler(state_root=tmp_path).summary(), sort_keys=True)
    assert "mutated" not in durable
    assert "secret provider prose" not in durable


@pytest.mark.asyncio
async def test_concurrent_duplicate_connector_admission_invokes_at_most_once(tmp_path) -> None:
    context = multiprocessing.get_context("fork")
    calls = context.Value("i", 0)
    entered = context.Event()

    async def connector(_prepared):
        with calls.get_lock():
            calls.value += 1
        entered.set()
        await asyncio.sleep(0.05)
        return _success()

    async def connect():
        return await admit_execution_session(
            tmp_path,
            _request(),
            [_manifest()],
            _authority(),
            20_000,
            trusted_connector=_isolated(connector),
        )

    first, second = await asyncio.gather(connect(), connect())
    assert entered.is_set()
    assert first == second
    assert calls.value == 1


@pytest.mark.asyncio
async def test_waiting_duplicate_recovers_cancelled_owner_without_replay(
    tmp_path, monkeypatch
) -> None:
    context = multiprocessing.get_context("fork")
    calls = context.Value("i", 0)
    connector_entered = context.Event()
    duplicate_dispatched = asyncio.Event()
    run_once_calls = 0
    original_run_once = Scheduler.run_once

    async def observed_run_once(self):
        nonlocal run_once_calls
        run_once_calls += 1
        if run_once_calls >= 2:
            duplicate_dispatched.set()
        return await original_run_once(self)

    monkeypatch.setattr(Scheduler, "run_once", observed_run_once)

    async def connector(_prepared):
        with calls.get_lock():
            calls.value += 1
        connector_entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def connect():
        return await admit_execution_session(
            tmp_path,
            _request(),
            [_manifest()],
            _authority(),
            20_000,
            trusted_connector=_isolated(connector),
        )

    owner = asyncio.create_task(connect())
    await _wait_process_event(connector_entered)
    duplicate = asyncio.create_task(connect())
    await duplicate_dispatched.wait()

    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    receipt = await asyncio.wait_for(duplicate, timeout=2.0)

    assert receipt["status"] == "blocked"
    assert receipt["result"]["error"] == "execution_outcome_unknown"
    assert calls.value == 1

    async def must_not_run(_prepared):
        raise AssertionError("connector replayed after persisted closure")

    restarted = await admit_execution_session(
        tmp_path,
        _request(),
        [_manifest()],
        _authority(),
        20_000,
        trusted_connector=_isolated(must_not_run),
    )
    assert restarted == receipt


@pytest.mark.asyncio
async def test_connector_deadline_converges_owner_duplicate_and_restart(
    tmp_path, monkeypatch
) -> None:
    context = multiprocessing.get_context("fork")
    calls = context.Value("i", 0)
    run_once_calls = 0
    connector_entered = context.Event()
    post_receipt_side_effect = tmp_path / "late-side-effect"
    original_run_once = Scheduler.run_once

    async def observed_run_once(self):
        nonlocal run_once_calls
        run_once_calls += 1
        return await original_run_once(self)

    monkeypatch.setattr(Scheduler, "run_once", observed_run_once)

    async def never_returns(_prepared):
        with calls.get_lock():
            calls.value += 1
        connector_entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            while True:
                post_receipt_side_effect.write_text("live", encoding="utf-8")
                await asyncio.sleep(0)

    async def connect(connector=never_returns):
        return await admit_execution_session(
            tmp_path,
            _short_request(),
            [_manifest()],
            _authority(),
            20_000,
            trusted_connector=_isolated(connector),
        )

    owner = asyncio.create_task(connect())
    await _wait_process_event(connector_entered)
    duplicate = asyncio.create_task(connect())
    owner_receipt = await asyncio.wait_for(asyncio.shield(owner), timeout=1.0)
    duplicate_receipt = await asyncio.wait_for(asyncio.shield(duplicate), timeout=1.0)

    assert owner_receipt == duplicate_receipt
    assert owner_receipt["status"] == "blocked"
    assert owner_receipt["result"]["error"] == "execution_connector_timeout"
    assert calls.value == 1
    assert run_once_calls < 20
    assert _live_connector_children() == []
    marker_mtime = post_receipt_side_effect.stat().st_mtime_ns if post_receipt_side_effect.exists() else None
    await asyncio.sleep(0.05)
    assert (post_receipt_side_effect.stat().st_mtime_ns if post_receipt_side_effect.exists() else None) == marker_mtime

    durable = json.dumps(Scheduler(state_root=tmp_path).summary(), sort_keys=True).lower()
    for content in (
        "top secret",
        "principal.secret",
        "save_response",
        "never_returns",
        "trusted_connector",
    ):
        assert content not in durable

    async def must_not_run(_prepared):
        raise AssertionError("timed-out connector replayed")

    restarted = await connect(must_not_run)
    assert restarted == owner_receipt
    assert calls.value == 1


@pytest.mark.asyncio
async def test_delayed_executor_entry_uses_absolute_activation_deadline(
    tmp_path, monkeypatch
) -> None:
    calls = 0
    first = True
    original_run_once = Scheduler.run_once

    async def delayed_run_once(self):
        nonlocal first
        if first:
            first = False
            await asyncio.sleep(0.26)
        return await original_run_once(self)

    monkeypatch.setattr(Scheduler, "run_once", delayed_run_once)

    async def connector(_prepared):
        nonlocal calls
        calls += 1
        return _success()

    receipt = await admit_execution_session(
        tmp_path,
        _short_request("session.delayed"),
        [_manifest()],
        _authority(),
        20_000,
        trusted_connector=_isolated(connector),
    )
    assert receipt["result"]["error"] == "execution_connector_timeout"
    assert calls == 0


@pytest.mark.asyncio
async def test_worker_start_stall_times_out_with_zero_live_work(
    tmp_path, monkeypatch
) -> None:
    import flyto_ai.execution_session_host as host

    context = multiprocessing.get_context("fork")
    worker_started = context.Event()
    late_effect = tmp_path / "start-timeout-late-effect"
    initial_children = {child.pid for child in multiprocessing.active_children()}
    initial_threads = {thread.ident for thread in threading.enumerate()}

    def stalled_start(_callback, channel):
        os.setsid()
        worker_started.set()
        time.sleep(2.0)
        late_effect.write_text("late", encoding="utf-8")
        channel.close()

    monkeypatch.setattr(host, "_connector_worker_main", stalled_start)

    async def must_not_enter(_prepared):
        raise AssertionError("connector entered before worker readiness")

    handle = ExecutionSessionConnector(must_not_enter)
    await _wait_process_event(worker_started)
    receipt = await admit_execution_session(
        tmp_path,
        _short_request("session.start-timeout"),
        [_manifest()],
        _authority(),
        20_000,
        trusted_connector=handle,
    )

    assert receipt["result"]["error"] == "execution_connector_timeout"
    assert {
        child.pid for child in multiprocessing.active_children()
    } == initial_children
    assert {thread.ident for thread in threading.enumerate()} == initial_threads
    assert _connector_async_tasks() == []
    assert not late_effect.exists()
    await asyncio.sleep(0.05)
    assert not late_effect.exists()


@pytest.mark.asyncio
async def test_owner_cancel_during_worker_start_stall_reaps_before_return(
    tmp_path, monkeypatch
) -> None:
    import flyto_ai.execution_session_host as host

    context = multiprocessing.get_context("fork")
    worker_started = context.Event()
    late_effect = tmp_path / "start-cancel-late-effect"
    initial_children = {child.pid for child in multiprocessing.active_children()}
    initial_threads = {thread.ident for thread in threading.enumerate()}

    def stalled_start(_callback, channel):
        os.setsid()
        worker_started.set()
        time.sleep(2.0)
        late_effect.write_text("late", encoding="utf-8")
        channel.close()

    monkeypatch.setattr(host, "_connector_worker_main", stalled_start)

    async def must_not_enter(_prepared):
        raise AssertionError("connector entered before worker readiness")

    admission = asyncio.create_task(
        admit_execution_session(
            tmp_path,
            _request("session.start-cancel"),
            [_manifest()],
            _authority(),
            20_000,
            trusted_connector=ExecutionSessionConnector(must_not_enter),
        )
    )
    await _wait_process_event(worker_started)
    admission.cancel()
    with pytest.raises(asyncio.CancelledError):
        await admission

    assert {
        child.pid for child in multiprocessing.active_children()
    } == initial_children
    assert {thread.ident for thread in threading.enumerate()} == initial_threads
    assert _connector_async_tasks() == []
    assert not late_effect.exists()
    await asyncio.sleep(0.05)
    assert not late_effect.exists()


@pytest.mark.asyncio
async def test_validation_failure_closes_preestablished_worker(tmp_path) -> None:
    initial_children = {child.pid for child in multiprocessing.active_children()}

    async def must_not_enter(_prepared):
        raise AssertionError("invalid admission invoked connector")

    invalid = _request("session.invalid-with-worker")
    invalid["device_command"] = {"move": True}
    with pytest.raises(ValueError):
        await admit_execution_session(
            tmp_path,
            invalid,
            [_manifest()],
            _authority(),
            20_000,
            trusted_connector=ExecutionSessionConnector(must_not_enter),
        )

    assert {
        child.pid for child in multiprocessing.active_children()
    } == initial_children
    assert _connector_async_tasks() == []


@pytest.mark.asyncio
async def test_owner_cancel_terminates_resistant_connector_before_return(
    tmp_path,
) -> None:
    context = multiprocessing.get_context("fork")
    calls = context.Value("i", 0)
    entered = context.Event()
    late_effect = tmp_path / "cancel-late-effect"

    async def resistant(_prepared):
        with calls.get_lock():
            calls.value += 1
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            while True:
                late_effect.write_text("live", encoding="utf-8")
                await asyncio.sleep(0)

    owner = asyncio.create_task(
        admit_execution_session(
            tmp_path,
            _request("session.resistant"),
            [_manifest()],
            _authority(),
            20_000,
            trusted_connector=_isolated(resistant),
        )
    )
    await _wait_process_event(entered)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner

    assert _live_connector_children() == []
    marker_mtime = late_effect.stat().st_mtime_ns if late_effect.exists() else None
    await asyncio.sleep(0.05)
    assert (late_effect.stat().st_mtime_ns if late_effect.exists() else None) == marker_mtime

    async def must_not_run(_prepared):
        raise AssertionError("connector replayed")

    recovered = await admit_execution_session(
        tmp_path,
        _request("session.resistant"),
        [_manifest()],
        _authority(),
        20_000,
        trusted_connector=_isolated(must_not_run),
    )
    assert recovered["result"]["error"] == "execution_outcome_unknown"
    restarted = await admit_execution_session(
        tmp_path,
        _request("session.resistant"),
        [_manifest()],
        _authority(),
        20_000,
        trusted_connector=_isolated(must_not_run),
    )
    assert restarted == recovered
    assert calls.value == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        {"ok": 1, "message": "", "error": None, "cost_usd": 0.0},
        {"ok": True, "message": "provider prose", "error": None, "cost_usd": 0.0},
        {"ok": True, "message": "", "error": None, "cost_usd": 1.0},
        {"ok": True, "message": "", "error": None, "cost_usd": 0.0, "extra": 1},
        {"ok": False, "message": "secret", "error": "secret", "cost_usd": 0.0},
    ],
)
async def test_hostile_connector_output_closes_stable_without_content(tmp_path, output) -> None:
    async def connector(_prepared):
        return output

    receipt = await admit_execution_session(
        tmp_path,
        _request(),
        [_manifest()],
        _authority(),
        20_000,
        trusted_connector=_isolated(connector),
    )
    assert receipt["status"] == "blocked"
    assert receipt["result"]["error"] == "executor_failed"
    assert "secret" not in json.dumps(receipt)


@pytest.mark.asyncio
async def test_connector_failure_and_exception_are_content_free_and_not_replayed(tmp_path) -> None:
    calls = multiprocessing.get_context("fork").Value("i", 0)

    async def failed(_prepared):
        with calls.get_lock():
            calls.value += 1
        return _failure()

    receipt = await admit_execution_session(
        tmp_path, _request(), [_manifest()], _authority(), 20_000, trusted_connector=_isolated(failed)
    )
    assert receipt["result"]["error"] == "execution_connector_failed"
    assert (await _admit(tmp_path)) == receipt
    assert calls.value == 1

    async def crashed(_prepared):
        raise RuntimeError("provider secret")

    other = await admit_execution_session(
        tmp_path,
        _request("session.crash"),
        [_manifest()],
        _authority(),
        20_000,
        trusted_connector=_isolated(crashed),
    )
    assert other["result"]["error"] == "executor_failed"
    assert "provider secret" not in json.dumps(other)


@pytest.mark.asyncio
async def test_cancelled_connector_recovers_unknown_without_replay(tmp_path) -> None:
    context = multiprocessing.get_context("fork")
    calls = context.Value("i", 0)
    entered = context.Event()

    async def connector(_prepared):
        with calls.get_lock():
            calls.value += 1
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    admission = asyncio.create_task(
        admit_execution_session(
            tmp_path,
            _request(),
            [_manifest()],
            _authority(),
            20_000,
            trusted_connector=_isolated(connector),
        )
    )
    await _wait_process_event(entered)
    admission.cancel()
    with pytest.raises(asyncio.CancelledError):
        await admission

    async def must_not_run(_prepared):
        raise AssertionError("connector replayed")

    recovered = await admit_execution_session(
        tmp_path,
        _request(),
        [_manifest()],
        _authority(),
        20_000,
        trusted_connector=_isolated(must_not_run),
    )
    assert recovered["result"]["error"] == "execution_outcome_unknown"
    assert calls.value == 1


@pytest.mark.asyncio
async def test_connector_is_not_request_authority_or_durable_content(tmp_path) -> None:
    hostile = _request()
    hostile["trusted_connector"] = "untrusted"
    with pytest.raises(ValueError):
        await _admit(tmp_path, hostile)

    async def named_secret_connector(_prepared):
        return _success()

    await admit_execution_session(
        tmp_path,
        _request(),
        [_manifest()],
        _authority(),
        20_000,
        trusted_connector=_isolated(named_secret_connector),
    )
    catalog = json.dumps(Scheduler(state_root=tmp_path).summary(), sort_keys=True).lower()
    definition = Scheduler(state_root=tmp_path).list_tasks()[0].instruction.lower()
    forbidden = (
        "top secret",
        "principal.secret",
        "save_response",
        "records.write",
        "named_secret_connector",
        "trusted_connector",
        "manifest",
        "credential",
    )
    assert all(value not in catalog for value in forbidden)
    assert all(value not in definition for value in forbidden)


@pytest.mark.asyncio
async def test_same_session_with_different_attestation_fails_closed(tmp_path) -> None:
    await _admit(tmp_path)
    changed = _request()
    changed["space"]["display_name"] = "Changed"  # type: ignore[index]
    with pytest.raises(ExecutionSessionHostError, match="conflicts"):
        await _admit(tmp_path, changed)
    assert Scheduler(state_root=tmp_path).summary()["tasks"][0]["run_count"] == 1


@pytest.mark.asyncio
async def test_deterministic_task_id_collision_fails_without_second_occurrence(
    tmp_path, monkeypatch
) -> None:
    import flyto_ai.execution_session_host as host

    monkeypatch.setattr(host, "_task_id", lambda _session_id: "execution-session-collision")
    first = await _admit(tmp_path, _request("session.first"))
    with pytest.raises(ExecutionSessionHostError, match="conflicts"):
        await _admit(tmp_path, _request("session.second"))

    summary = Scheduler(state_root=tmp_path).summary()
    assert first["session"]["session_id"] == "session.first"
    assert summary["task_count"] == 1
    assert summary["tasks"][0]["run_count"] == 1


@pytest.mark.asyncio
async def test_entered_occurrence_restart_projects_unknown_without_replay(
    tmp_path, monkeypatch
) -> None:
    import flyto_ai.execution_session_host as host

    calls = 0
    entered = asyncio.Event()
    original_executor = host._admission_executor

    async def interrupted_executor(_instruction: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(host, "_admission_executor", interrupted_executor)
    admission = asyncio.create_task(_admit(tmp_path))
    await entered.wait()
    admission.cancel()
    with pytest.raises(asyncio.CancelledError):
        await admission

    monkeypatch.setattr(host, "_admission_executor", original_executor)
    receipt = await _admit(tmp_path)
    assert receipt["status"] == "blocked"
    assert receipt["result"]["ok"] is False
    assert receipt["result"]["error"] == "execution_outcome_unknown"
    assert calls == 1
    summary = Scheduler(state_root=tmp_path).summary()
    assert summary["task_count"] == 1
    assert summary["tasks"][0]["run_count"] == 1


@pytest.mark.asyncio
async def test_valid_timestamp_ceiling_uses_immediate_scheduler_boundary(tmp_path) -> None:
    request = _request()
    request["activation"] = {
        "source": "typed",
        "observed_wake_word": None,
        "activated_at_ms": _MAX_TIMESTAMP_MS - 1_000,
        "expires_at_ms": _MAX_TIMESTAMP_MS,
    }
    receipt = await admit_execution_session(
        tmp_path,
        request,
        [_manifest()],
        _authority(),
        _MAX_TIMESTAMP_MS - 1,
    )
    task = Scheduler(state_root=tmp_path).list_tasks()[0]
    assert receipt["result"]["error"] == "execution_not_connected"
    assert task.schedule.run_at > 0


@pytest.mark.asyncio
async def test_hostile_unknown_and_oversized_data_are_rejected(tmp_path) -> None:
    hostile = _request()
    hostile["device_command"] = {"move": True}
    with pytest.raises(ValueError):
        await _admit(tmp_path, hostile)
    oversized = _request()
    oversized["goal"]["text"] = "x" * 4_001  # type: ignore[index]
    with pytest.raises(ValueError):
        await _admit(tmp_path, oversized)
    assert not (tmp_path / "scheduler-catalog").exists()


@pytest.mark.asyncio
async def test_state_root_is_mandatory_and_receipt_is_detached_json(tmp_path) -> None:
    with pytest.raises(ExecutionSessionHostError, match="durable"):
        await admit_execution_session(None, _request(), [_manifest()], _authority(), 20_000)  # type: ignore[arg-type]
    receipt = await _admit(tmp_path)
    detached = json.loads(json.dumps(receipt))
    detached["digests"]["request"] = "changed"
    assert (await _admit(tmp_path))["digests"]["request"] != "changed"


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_path", ["", b"", _EmptyPathLike()])
async def test_empty_state_root_creates_no_scheduler_catalog(
    tmp_path, monkeypatch, empty_path
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ExecutionSessionHostError, match="durable"):
        await admit_execution_session(
            empty_path, _request(), [_manifest()], _authority(), 20_000
        )
    assert not (tmp_path / "scheduler-catalog").exists()


def test_source_imports_prove_no_cloud_core_or_device_execution() -> None:
    import flyto_ai.execution_session_host as host

    source = inspect.getsource(host).lower()
    forbidden = ("flyto_cloud", "core_tools", "robot", "device", "browser", "credential")
    assert all(token not in source for token in forbidden)
    assert "prepare_execution_session(" in source
    assert "scheduler(" in source
