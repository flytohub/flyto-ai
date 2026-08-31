"""Compact conditional polling contract for ``flyto_coding_get``."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from flyto_ai.coding.contracts import (
    CodingJobReceipt,
    CodingJobState,
    CodingTaskResult,
    audit_findings_sha256,
)
from flyto_ai.coding.mcp_server import (
    CODING_MCP_SERVER_VERSION,
    GET_WAIT_RETRY_AFTER_MS,
    MAX_GET_WAIT_MS,
    CodingMCPServer,
    _GET_ARGUMENT_FIELDS,
    _GET_SUMMARY_FIELDS,
)
from flyto_ai.coding.mcp_supervisor import WORKER_RESPONSE_TIMEOUT_SECONDS
from flyto_ai.coding.service import CodingJobNotFound, receipt_to_mapping
from tests.test_coding_service import _declare_verification, _service


_JOB_ID = "job_" + "a" * 24
_REVISION = "b" * 64


def _receipt(
    state: CodingJobState = CodingJobState.QUEUED,
    *,
    updated_at: float = 90.0,
    failure_code: str | None = None,
    result: CodingTaskResult | None = None,
) -> CodingJobReceipt:
    bound = state in {
        CodingJobState.AWAITING_CODEX_AUDIT,
        CodingJobState.REWORK_QUEUED,
        CodingJobState.REWORK_RUNNING,
        CodingJobState.REWORK_ROUTE_BLOCKED,
        CodingJobState.CODEX_ACCEPTED,
    }
    audited = state in {
        CodingJobState.REWORK_QUEUED,
        CodingJobState.REWORK_RUNNING,
        CodingJobState.REWORK_ROUTE_BLOCKED,
        CodingJobState.CODEX_ACCEPTED,
    }
    accepted = state is CodingJobState.CODEX_ACCEPTED
    return CodingJobReceipt(
        job_id=_JOB_ID,
        state=state,
        submitted_at=1.0,
        updated_at=updated_at,
        result=result,
        failure_code=failure_code,
        implementation_backend="codex" if bound else "",
        implementation_session_id="opaque-session" if bound else "",
        implementation_revision_sha256=_REVISION if bound else "",
        audit_count=1 if audited else 0,
        rework_count=(0 if accepted else 1) if audited else 0,
        audit_findings_sha256=audit_findings_sha256(()) if audited else "",
        landable=accepted,
    )


class ScriptedGetService:
    """Tenant-recording fake with independent full and summary read scripts."""

    def __init__(
        self,
        receipts: list[CodingJobReceipt],
        *,
        summary_receipts: list[CodingJobReceipt] | None = None,
    ) -> None:
        self.receipts = receipts
        self.summary_receipts = summary_receipts or receipts
        self.full_calls: list[tuple[str, str]] = []
        self.summary_calls: list[tuple[str, str]] = []

    @staticmethod
    def _next(values: list[CodingJobReceipt], count: int) -> CodingJobReceipt:
        return values[min(count, len(values) - 1)]

    def get(self, tenant_id: str, job_id: str) -> CodingJobReceipt:
        receipt = self._next(self.receipts, len(self.full_calls))
        self.full_calls.append((tenant_id, job_id))
        return receipt

    def get_summary(self, tenant_id: str, job_id: str) -> CodingJobReceipt:
        receipt = self._next(self.summary_receipts, len(self.summary_calls))
        self.summary_calls.append((tenant_id, job_id))
        return receipt


def _call(server: CodingMCPServer, **arguments):
    return server.handle({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "flyto_coding_get",
            "arguments": {"job_id": _JOB_ID, **arguments},
        },
    })["result"]


class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def _install_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr("flyto_ai.coding.mcp_server._monotonic", clock.now)
    monkeypatch.setattr("flyto_ai.coding.mcp_server._wall_time", clock.now)
    monkeypatch.setattr("flyto_ai.coding.mcp_server._sleep", clock.sleep)
    return clock


def test_get_schema_is_closed_bounded_and_keeps_exactly_three_tools() -> None:
    tools = {item["name"]: item for item in CodingMCPServer._tools()}
    assert set(tools) == {
        "flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit",
    }
    schema = tools["flyto_coding_get"]["inputSchema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["job_id"]
    assert set(schema["properties"]) == set(_GET_ARGUMENT_FIELDS)
    assert schema["properties"]["detail"]["enum"] == ["summary", "full"]
    assert schema["properties"]["after_change_token"]["pattern"] == "^[a-f0-9]{64}$"
    assert schema["properties"]["wait_ms"] == {
        "type": "integer", "minimum": 0, "maximum": MAX_GET_WAIT_MS,
    }
    assert CODING_MCP_SERVER_VERSION == "3"
    assert MAX_GET_WAIT_MS / 1000 <= WORKER_RESPONSE_TIMEOUT_SECONDS - 10


def test_job_id_only_get_preserves_the_exact_legacy_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    receipt = _receipt()
    service = ScriptedGetService([receipt])
    result = _call(CodingMCPServer(service, "tenant-a"))
    payload = result["structuredContent"]

    assert result["isError"] is False
    assert payload == {"ok": True, "job": receipt_to_mapping(receipt)}
    assert set(payload) == {"ok", "job"}
    assert service.full_calls == [("tenant-a", _JOB_ID)]
    assert service.summary_calls == []
    assert json.loads(result["content"][0]["text"]) == payload


def test_explicit_full_get_preserves_job_and_adds_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    receipt = _receipt()
    service = ScriptedGetService([receipt])
    result = _call(CodingMCPServer(service, "tenant-a"), detail="full")
    payload = result["structuredContent"]

    assert result["isError"] is False
    assert payload["job"] == receipt_to_mapping(receipt)
    assert payload["observation"] == {
        "detail": "full",
        "change_token": payload["observation"]["change_token"],
        "changed": True,
        "timed_out": False,
        "waited_ms": 0,
        "retry_after_ms": 0,
        "recommended_wait_ms": MAX_GET_WAIT_MS,
        "progress_age_ms": 10_000,
        "next_action": "wait",
    }
    assert len(payload["observation"]["change_token"]) == 64
    assert service.full_calls == [("tenant-a", _JOB_ID)]
    assert service.summary_calls == []
    assert json.loads(result["content"][0]["text"]) == payload


def test_summary_is_an_allowlisted_compact_polling_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_clock(monkeypatch)
    large = CodingTaskResult(
        ok=False,
        message="provider detail " * 2_000,
        thread_id="private-thread-shape",
        attempts=1,
        status="running",
        files_changed=["src/file_{:03d}.py".format(index) for index in range(64)],
    )
    receipt = _receipt(CodingJobState.RUNNING, result=large)
    service = ScriptedGetService([receipt])
    server = CodingMCPServer(service, "tenant-a")
    full = _call(server, detail="full")["structuredContent"]
    summary = _call(server, detail="summary")["structuredContent"]

    assert tuple(summary["job"]) == _GET_SUMMARY_FIELDS
    assert summary["job"] == {
        field: full["job"][field] for field in _GET_SUMMARY_FIELDS
    }
    assert summary["job"]["state"] == "running"
    assert summary["job"]["job_terminal"] is False
    assert summary["job"]["landable"] is False
    assert summary["job"]["failure_phase"] == ""
    assert summary["job"]["required_actions"] == []
    assert "result" not in summary["job"]
    assert "route_receipt" not in summary["job"]
    assert "mission" not in summary["job"]
    assert summary["observation"]["detail"] == "summary"
    assert summary["observation"]["change_token"] != full["observation"]["change_token"]
    full_bytes = len(json.dumps(full, ensure_ascii=False, separators=(",", ":")).encode())
    summary_bytes = len(json.dumps(summary, ensure_ascii=False, separators=(",", ":")).encode())
    assert summary_bytes < full_bytes // 10
    assert service.summary_calls == [("tenant-a", _JOB_ID)]


def test_summary_long_poll_returns_on_change_and_tokens_are_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch)
    queued = _receipt(CodingJobState.QUEUED)
    running = _receipt(CodingJobState.RUNNING, updated_at=100.25)
    service = ScriptedGetService(
        [queued], summary_receipts=[queued, queued, running],
    )
    server = CodingMCPServer(service, "tenant-a")
    first = _call(server, detail="summary")["structuredContent"]
    token = first["observation"]["change_token"]
    changed = _call(
        server,
        detail="summary",
        after_change_token=token,
        wait_ms=1_000,
    )["structuredContent"]

    assert changed["job"]["state"] == "running"
    assert changed["observation"]["change_token"] != token
    assert changed["observation"]["changed"] is True
    assert changed["observation"]["timed_out"] is False
    assert changed["observation"]["waited_ms"] == 250
    assert changed["observation"]["retry_after_ms"] == 0
    assert changed["observation"]["next_action"] == "wait"
    assert clock.value == 100.25
    assert service.summary_calls == [
        ("tenant-a", _JOB_ID),
        ("tenant-a", _JOB_ID),
        ("tenant-a", _JOB_ID),
    ]


def test_unchanged_long_poll_times_out_with_bounded_retry_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _install_clock(monkeypatch)
    queued = _receipt()
    service = ScriptedGetService([queued], summary_receipts=[queued])
    server = CodingMCPServer(service, "tenant-a")
    token = _call(server, detail="summary")["structuredContent"]["observation"][
        "change_token"
    ]
    unchanged = _call(
        server,
        detail="summary",
        after_change_token=token,
        wait_ms=500,
    )["structuredContent"]

    assert unchanged["observation"] == {
        "detail": "summary",
        "change_token": token,
        "changed": False,
        "timed_out": True,
        "waited_ms": 500,
        "retry_after_ms": GET_WAIT_RETRY_AFTER_MS,
        "recommended_wait_ms": MAX_GET_WAIT_MS,
        "progress_age_ms": 10_500,
        "next_action": "wait",
    }
    assert clock.value == 100.5


@pytest.mark.parametrize(("receipt", "action"), [
    (_receipt(CodingJobState.AWAITING_CODEX_AUDIT), "audit_revision"),
    (
        _receipt(
            CodingJobState.REWORK_ROUTE_BLOCKED,
            failure_code="rework_route_blocked",
        ),
        "retry_rework_route",
    ),
    (_receipt(CodingJobState.CODEX_ACCEPTED), "land_accepted_revision"),
    (
        _receipt(CodingJobState.FAILED, failure_code="provider_capacity_unavailable"),
        "retry_same_request",
    ),
    (
        _receipt(CodingJobState.FAILED, failure_code="provider_auth_failed"),
        "resolve_required_actions",
    ),
    (_receipt(CodingJobState.FAILED, failure_code="unknown_failure"), "stop_non_landable"),
    (_receipt(CodingJobState.COMPLETED), "stop_non_landable"),
])
def test_actionable_and_terminal_states_never_wait(
    monkeypatch: pytest.MonkeyPatch,
    receipt: CodingJobReceipt,
    action: str,
) -> None:
    clock = _install_clock(monkeypatch)
    service = ScriptedGetService([receipt])
    server = CodingMCPServer(service, "tenant-a")
    first = _call(server, detail="summary")["structuredContent"]
    second = _call(
        server,
        detail="summary",
        after_change_token=first["observation"]["change_token"],
        wait_ms=MAX_GET_WAIT_MS,
    )["structuredContent"]

    assert second["observation"]["next_action"] == action
    assert second["observation"]["waited_ms"] == 0
    assert second["observation"]["timed_out"] is False
    assert second["observation"]["recommended_wait_ms"] == 0
    assert clock.value == 100.0


@pytest.mark.parametrize("arguments", [
    {"job_id": True},
    {"job_id": "job_" + "z" * 24},
    {"job_id": _JOB_ID, "detail": True},
    {"job_id": _JOB_ID, "detail": "compact"},
    {"job_id": _JOB_ID, "wait_ms": True},
    {"job_id": _JOB_ID, "wait_ms": -1},
    {"job_id": _JOB_ID, "wait_ms": MAX_GET_WAIT_MS + 1},
    {"job_id": _JOB_ID, "wait_ms": 1},
    {"job_id": _JOB_ID, "wait_ms": 1, "after_change_token": "a" * 64},
    {"job_id": _JOB_ID, "after_change_token": True},
    {"job_id": _JOB_ID, "after_change_token": "B" * 64},
    {"job_id": _JOB_ID, "after_change_token": "a" * 63},
    {"job_id": _JOB_ID, "workspace": "/must/not/be/accepted"},
])
def test_get_runtime_rejects_every_malformed_or_undeclared_argument(
    arguments: dict,
) -> None:
    service = ScriptedGetService([_receipt()])
    response = CodingMCPServer(service, "tenant-a").handle({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        "params": {"name": "flyto_coding_get", "arguments": arguments},
    })["result"]
    assert response["isError"] is True
    assert response["structuredContent"] == {"ok": False, "error": "invalid_request"}
    assert service.full_calls == []
    assert service.summary_calls == []


def test_lock_free_summary_peek_stays_tenant_isolated_and_avoids_coordination_wait(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    service = _service(tmp_path, workspace)
    queued = service.submit("tenant-a", "get-summary-lock", _request_for(workspace))
    entered = threading.Event()
    release = threading.Event()
    full_done = threading.Event()

    def hold_guard() -> None:
        with service._state_guard():
            entered.set()
            assert release.wait(timeout=5)

    def full_read() -> None:
        service.get("tenant-a", queued.job_id)
        full_done.set()

    holder = threading.Thread(target=hold_guard)
    holder.start()
    try:
        assert entered.wait(timeout=5)
        summary = service.get_summary("tenant-a", queued.job_id)
        assert summary.job_id == queued.job_id
        with pytest.raises(CodingJobNotFound):
            service.get_summary("tenant-b", queued.job_id)

        blocked = threading.Thread(target=full_read)
        blocked.start()
        assert not full_done.wait(timeout=0.05)
        release.set()
        assert full_done.wait(timeout=5)
        blocked.join(timeout=5)
    finally:
        release.set()
        holder.join(timeout=5)
        service.close(wait=True)


def _request_for(workspace: Path):
    """Build the normal service request without broadening production schema."""

    from tests.test_coding_service import _request

    return _request(workspace)
