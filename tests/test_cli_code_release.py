"""The host-only release valve, exercised through the real CLI entry point.

`code-release` is the one way to retire an orphaned audit or clear a claim the
service refuses to evaluate. It is deliberately not an MCP tool, so these tests
drive `flyto_ai.cli.main` exactly as an operator would and assert both what it
does and what it must never be able to do.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from flyto_ai.coding.contracts import (
    CodingAuditVerdict,
    CodingJobState,
    CodingMissionProjection,
)
from flyto_ai.coding.mcp_server import CodingMCPServer
from flyto_ai.coding.mission_runtime import CodingMissionRuntime
from flyto_ai.coding.service import AbandonStateConflict
from flyto_ai.orchestration.mission_control import (
    DISPOSITION_BLOCKED,
    STATUS_CLOSED,
    STATUS_DISPATCHED,
)
from tests.test_coding_service import (
    ReworkingProvider,
    _audited_service,
    _awaiting,
    _blocker,
    _request,
)
from tests.test_coding_workspace_ownership import _claim_path


def _run(monkeypatch, capsys, *args):
    """Invoke the real CLI and return `(exit_code, stdout, stderr)`."""

    import flyto_ai.cli as cli

    monkeypatch.setattr("sys.argv", ["flyto-ai", "code-release", *args])
    code = 0
    try:
        cli.main()
    except SystemExit as exit_signal:
        code = int(exit_signal.code or 0)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _orphaned(tmp_path: Path):
    """One audit-ready job whose worker is gone, plus its state/workspace."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "cli-001", workspace)
    finally:
        service.close()
    return owner, workspace, str(tmp_path / "service-state")


def test_abandon_reports_failed_and_non_landable_json(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    owner, workspace, state_dir = _orphaned(tmp_path)
    code, out, _err = _run(
        monkeypatch, capsys,
        "--tenant", "tenant-audit",
        "--workspace-root", str(workspace),
        "--state-dir", state_dir,
        "--abandon-job", owner.job_id,
        "--json",
    )
    assert code == 0
    report = json.loads(out)
    assert report == {
        "operation": "abandon_job",
        "job_id": owner.job_id,
        "state": CodingJobState.FAILED.value,
        "failure_code": "job_abandoned",
        "landable": False,
    }

    # The durable record agrees, and the worktree is released.
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        settled = service.get("tenant-audit", owner.job_id)
        assert settled.state is CodingJobState.FAILED
        assert settled.landable is False
        assert not _claim_path(service, workspace).exists()
        # Reusable: a fresh job may now take the same worktree.
        assert service.submit(
            "tenant-audit", "cli-002", _request(workspace),
        ).state is CodingJobState.QUEUED
    finally:
        service.close()


def test_abandon_reconciles_an_orphaned_audit_dispatch_and_is_idempotent(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """The historical failed-job/open-item split releases the repository."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    owner = _awaiting(service, "tenant-audit", "stale-audit", workspace)
    for future in tuple(service._pending):
        future.result(timeout=5)
    monkeypatch.setattr(service, "_schedule_pump", lambda: None)
    repair = service.audit(
        "tenant-audit", owner.job_id, owner.implementation_revision_sha256,
        CodingAuditVerdict.REWORK, (_blocker(),),
    )
    projection = CodingMissionProjection.from_mapping(repair.mission or {})
    assert projection.lane == "repair"

    # Stand in for the process dying after the repair item was dispatched but
    # before its audit-ready settlement closed the item.
    script = (
        "import os\n"
        "from flyto_ai.coding.mission_runtime import CodingMissionRuntime\n"
        f"runtime = CodingMissionRuntime({str(service.state_root)!r}, "
        "worker='crashed-auditor')\n"
        f"item = runtime.store.get_work_item({projection.work_item_id!r})\n"
        "with runtime.store.dispatch_expected(\n"
        "    operation='cmd-crashed-audit', worker=runtime.worker,\n"
        f"    work_item_id={projection.work_item_id!r},\n"
        "    expected_attempt=item.attempts + 1,\n"
        ") as work:\n"
        f"    assert work is not None and work.work_item_id == "
        f"{projection.work_item_id!r}\n"
        "    os._exit(0)\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    item = CodingMissionRuntime._persisted_work_item(
        service.state_root, projection.work_item_id,
    )
    assert item is not None and item.status == STATUS_DISPATCHED
    record_path = (
        service.state_root / "tenants" / service._tenant_ref("tenant-audit")
        / "jobs" / (owner.job_id + ".json")
    )
    service._advance_projection(record_path, status=STATUS_DISPATCHED)
    service._update_record(
        record_path,
        state=CodingJobState.AWAITING_CODEX_AUDIT.value,
        landable=False,
    )
    service.close()

    args = (
        "--tenant", "tenant-audit",
        "--workspace-root", str(workspace),
        "--state-dir", str(tmp_path / "service-state"),
        "--abandon-job", owner.job_id,
        "--json",
    )
    code, _out, err = _run(monkeypatch, capsys, *args)
    assert code == 0, err
    settled_item = CodingMissionRuntime._persisted_work_item(
        tmp_path / "service-state", projection.work_item_id,
    )
    assert settled_item is not None
    assert settled_item.status == STATUS_CLOSED
    assert settled_item.disposition == DISPOSITION_BLOCKED

    # The exact historical terminal shape is a safe no-op on a retry.
    code, _out, err = _run(monkeypatch, capsys, *args)
    assert code == 0, err

    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        receipt = service.get("tenant-audit", owner.job_id)
        assert receipt.state is CodingJobState.FAILED
        assert receipt.failure_code == "job_abandoned"
        assert receipt.landable is False
        assert receipt.mission is not None
        assert receipt.mission["status"] == STATUS_CLOSED
        assert receipt.mission["disposition"] == DISPOSITION_BLOCKED
        assert not _claim_path(service, workspace).exists()
        fresh = service.submit("tenant-audit", "after-stale-audit", _request(workspace))
        assert fresh.job_id != owner.job_id
    finally:
        service.close()


def test_abandon_does_not_reconcile_an_unrelated_failed_job(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    owner, workspace, state_dir = _orphaned(tmp_path)
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        path = (
            service.state_root / "tenants" / service._tenant_ref("tenant-audit")
            / "jobs" / (owner.job_id + ".json")
        )
        service._update_record(
            path, state=CodingJobState.FAILED.value,
            failure_code="service_execution_failed", landable=False,
        )
    finally:
        service.close()
    code, _out, err = _run(
        monkeypatch, capsys,
        "--tenant", "tenant-audit", "--workspace-root", str(workspace),
        "--state-dir", state_dir, "--abandon-job", owner.job_id, "--json",
    )
    assert code == 2
    assert "abandon_state_conflict" in err


def test_abandon_retires_only_kernel_accounted_queued_work(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """A pre-fix admission race is recoverable without weakening live work.

    The mission kernel has already closed the exact item blocked, so no worker
    can run it. The durable job record is intentionally left queued to model an
    older service that published it after the item was accounted. The host
    valve may fail that proof-bound record closed and release its worktree.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    monkeypatch.setattr(service, "_schedule_pump", lambda: None)
    owner = service.submit("tenant-audit", "cli-stranded", _request(workspace))
    projection = owner.mission or {}
    with pytest.raises(AbandonStateConflict):
        service.abandon("tenant-audit", owner.job_id)
    with service._mission.dispatch() as work:
        assert work is not None
        assert work.work_item_id == projection["work_item_id"]
        service._account_unrunnable(work, "job_record_unreadable")
    assert service.get("tenant-audit", owner.job_id).state is CodingJobState.QUEUED
    try:
        # The service that admitted the work is deliberately still alive. The
        # release must rely on the exact closed mission item and free job lease,
        # not on stopping every unrelated supervisor sharing the state root.
        code, out, _err = _run(
            monkeypatch, capsys,
            "--tenant", "tenant-audit",
            "--workspace-root", str(workspace),
            "--state-dir", str(tmp_path / "service-state"),
            "--abandon-job", owner.job_id,
            "--json",
        )
        assert code == 0
        assert json.loads(out) == {
            "operation": "abandon_job",
            "job_id": owner.job_id,
            "state": CodingJobState.FAILED.value,
            "failure_code": "job_abandoned",
            "landable": False,
        }
        assert service.get("tenant-audit", owner.job_id).state is CodingJobState.FAILED
        assert not _claim_path(service, workspace).exists()
    finally:
        service.close()

    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        assert not _claim_path(service, workspace).exists()
        fresh = service.submit("tenant-audit", "cli-after-stranded", _request(workspace))
        assert fresh.state is CodingJobState.QUEUED
    finally:
        service.close()


def test_repair_clears_a_corrupt_claim_and_refuses_a_live_one(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    owner, workspace, state_dir = _orphaned(tmp_path)
    common = (
        "--tenant", "tenant-audit",
        "--workspace-root", str(workspace),
        "--state-dir", state_dir,
    )

    # A live owner is never cleared by repair; it must be audited or abandoned.
    code, _out, err = _run(
        monkeypatch, capsys, *common, "--repair-workspace", str(workspace), "--json",
    )
    assert code == 2
    assert "workspace_busy" in err

    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        service.abandon("tenant-audit", owner.job_id)
        _claim_path(service, workspace).write_bytes(b"{ corrupt")
    finally:
        service.close()

    code, out, _err = _run(
        monkeypatch, capsys, *common, "--repair-workspace", str(workspace), "--json",
    )
    assert code == 0
    report = json.loads(out)
    assert report["operation"] == "repair_workspace"
    assert report["repaired"] is True
    assert report["status"] == "unresolved"

    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        assert not _claim_path(service, workspace).exists()
    finally:
        service.close()


@pytest.mark.parametrize(
    "target",
    [
        pytest.param(("--abandon-job", "not-a-job-id"), id="malformed_job_id"),
        pytest.param(
            ("--abandon-job", "job_" + "9" * 24), id="unknown_job",
        ),
        pytest.param(
            ("--repair-workspace", "/definitely/outside/any/root"),
            id="workspace_outside_roots",
        ),
    ],
)
def test_invalid_input_exits_cleanly_with_a_stable_code(
    tmp_path: Path, monkeypatch, capsys, target,
) -> None:
    """A bad request is a bounded stable code on stderr, never a traceback."""

    _owner, workspace, state_dir = _orphaned(tmp_path)
    code, out, err = _run(
        monkeypatch, capsys,
        "--tenant", "tenant-audit",
        "--workspace-root", str(workspace),
        "--state-dir", state_dir,
        *target,
    )
    assert code == 2
    assert "Traceback" not in err and "Traceback" not in out
    assert err.strip()


def test_an_invalid_tenant_exits_bounded_instead_of_tracebacking(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """Contract rejections of operator input are usage errors, not crashes."""

    owner, workspace, state_dir = _orphaned(tmp_path)
    code, out, err = _run(
        monkeypatch, capsys,
        "--tenant", "../bad",
        "--workspace-root", str(workspace),
        "--state-dir", state_dir,
        "--abandon-job", owner.job_id,
    )
    assert code == 2
    assert "invalid_request" in err
    assert "Traceback" not in err and "Traceback" not in out


def test_abandon_never_invokes_an_implementer(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """The valve retires durable state; it must not be able to run a round."""

    owner, workspace, state_dir = _orphaned(tmp_path)
    started = []
    import flyto_ai.coding.service as service_module

    original = service_module.CodingService._run_job

    def tracked(self, *args, **kwargs):
        started.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(service_module.CodingService, "_run_job", tracked)
    code, _out, _err = _run(
        monkeypatch, capsys,
        "--tenant", "tenant-audit",
        "--workspace-root", str(workspace),
        "--state-dir", state_dir,
        "--abandon-job", owner.job_id,
    )
    assert code == 0
    assert started == []


def test_the_release_valve_adds_no_mcp_tool(tmp_path: Path) -> None:
    """The audited inventory stays exactly submit/get/audit."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        listed = CodingMCPServer(service, "tenant-audit").handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })
        names = [tool["name"] for tool in listed["result"]["tools"]]
        assert names == [
            "flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit",
        ]
        assert not any(
            token in name
            for name in names
            for token in ("abandon", "release", "repair")
        )
    finally:
        service.close()
