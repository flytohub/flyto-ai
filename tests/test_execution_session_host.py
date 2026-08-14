from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path

import pytest

from flyto_ai.execution_session import (
    EXECUTION_SESSION_REQUEST_VERSION,
    ExecutionAuthority,
    _MAX_TIMESTAMP_MS,
)
from flyto_ai.execution_session_host import (
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


async def _admit(root: Path, request: dict[str, object] | None = None):
    return await admit_execution_session(
        root, request or _request(), [_manifest()], _authority(), 20_000
    )


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
