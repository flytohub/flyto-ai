# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Host-only unified task window."""
import json
import subprocess
from pathlib import Path

from flyto_ai.coding.contracts import CodingJobState, CodingTaskRequest
from flyto_ai.coding.task_window import TASK_WINDOW_SCHEMA, read_task_window

from tests.test_coding_service import ReworkingProvider, _audited_service, _wait
from tests.test_coding_workspace_ownership import _declare_verification, _workspace


def test_task_window_joins_mission_queue_repo_set_and_audit_without_secrets(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path, "flyto-code")
    _declare_verification(workspace)
    subprocess.run(["git", "-C", str(workspace), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(workspace), "add", "."], check=True)
    subprocess.run([
        "git", "-C", str(workspace), "-c", "user.name=Flyto Test",
        "-c", "user.email=flyto@example.invalid", "commit", "-qm", "fixture",
    ], check=True)
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        request = CodingTaskRequest(
            message="private main axis with secret-token-value",
            working_dir=str(workspace),
            owner_ref="codex-019ff12f",
        )
        queued = service.submit("tenant-audit", "window-001", request)
        receipt = _wait(service, "tenant-audit", queued.job_id)
        assert receipt.state is CodingJobState.AWAITING_CODEX_AUDIT

        report = read_task_window(service.state_root)
        encoded = json.dumps(report, sort_keys=True)
        assert report["schema"] == TASK_WINDOW_SCHEMA
        assert report["tasks"][0]["owner_ref"] == "codex-019ff12f"
        assert report["tasks"][0]["repository_digests"]
        assert report["tasks"][0]["implementation_session_bound"] is True
        assert report["tasks"][0]["main_axis_sha256"]
        assert "private main axis" not in encoded
        assert "secret-token-value" not in encoded
        assert str(workspace) not in encoded
        assert receipt.implementation_session_id not in encoded
        assert "working_dir" not in encoded
    finally:
        service.close()


def test_empty_task_window_is_read_only_and_bounded(tmp_path: Path) -> None:
    state = tmp_path / "missing"
    report = read_task_window(state, limit=1)
    assert report["tasks"] == []
    assert report["missions"] == []
    assert not state.exists()
