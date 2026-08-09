"""Worktree ownership across real OS processes, not just two objects.

The rest of the ownership suite builds two `CodingService` instances inside one
interpreter. That covers the durable state machine but shares a process, so it
cannot prove the property that actually matters in production: a Codex frontend
exits, and a *different* interpreter picks the job up from the shared state
root alone.

This module runs a genuine second interpreter. It uses a stub implementer
written to a temporary file — no network, no sockets, and no Claude API — so it
stays deterministic and self-contained while still exercising the real service.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from flyto_ai.coding.contracts import CodingJobState

REPO_ROOT = Path(__file__).resolve().parent.parent
#: Every child call is bounded so a hung interpreter fails the test instead of
#: hanging the suite.
CHILD_TIMEOUT_SECONDS = 120

_DRIVER = r'''
import json
import sys

from flyto_ai.coding import FlytoCodingAgent
from flyto_ai.coding.contracts import (
    CodingAuditFinding,
    CodingAuditSeverity,
    CodingAuditVerdict,
    CodingJobState,
)
from flyto_ai.coding.service import CodingService

STATE_ROOT, WORKSPACE, ACTION = sys.argv[1], sys.argv[2], sys.argv[3]
ROUND_TAG = sys.argv[4] if len(sys.argv) > 4 else "1"


class StubProvider:
    """Writes the files the repository check expects. No network, no SDK."""

    async def chat(self, **kwargs):
        for path, content in (
            ("result.txt", "verified\n"),
            ("notes.txt", "round {}\n".format(ROUND_TAG)),
        ):
            outcome = await kwargs["dispatch_fn"](
                "coding_write_file",
                {"path": path, "content": content, "overwrite": True},
            )
            assert outcome["ok"], outcome
        return "done", [{"function": "coding_write_file", "ok": True}], 1, {"total_tokens": 1}


def build():
    return CodingService(
        lambda store: FlytoCodingAgent(StubProvider(), store=store),
        state_root=STATE_ROOT,
        workspace_roots=(WORKSPACE,),
        max_workers=2,
        max_queued=8,
        require_codex_audit=True,
        max_rework_rounds=3,
    )


def settle(service, job_id, timeout=90):
    import time

    deadline = time.monotonic() + timeout
    terminal = {
        CodingJobState.AWAITING_CODEX_AUDIT,
        CodingJobState.COMPLETED,
        CodingJobState.FAILED,
        CodingJobState.CODEX_ACCEPTED,
    }
    while time.monotonic() < deadline:
        receipt = service.get("tenant-audit", job_id)
        if receipt.state in terminal:
            return receipt
        time.sleep(0.02)
    raise AssertionError("job did not settle")


def emit(payload):
    sys.stdout.write("FLYTO_RESULT " + json.dumps(payload) + "\n")


service = build()
try:
    if ACTION == "implement":
        from flyto_ai.coding.service import request_from_mapping

        queued = service.submit("tenant-audit", "mp-001", request_from_mapping({
            "message": "write verified result", "working_dir": WORKSPACE,
        }))
        receipt = settle(service, queued.job_id)
        emit({
            "job_id": receipt.job_id,
            "state": receipt.state.value,
            "session": receipt.implementation_session_id,
            "revision": receipt.implementation_revision_sha256,
        })
    elif ACTION == "submit_should_fail":
        from flyto_ai.coding.service import request_from_mapping

        try:
            service.submit("tenant-audit", "mp-002", request_from_mapping({
                "message": "competing task", "working_dir": WORKSPACE,
            }))
            emit({"error": "", "owner_job_id": ""})
        except Exception as exc:
            emit({
                "error": getattr(exc, "code", type(exc).__name__),
                "owner_job_id": getattr(exc, "owner_job_id", ""),
            })
    elif ACTION == "rework":
        job_id, revision = sys.argv[5], sys.argv[6]
        service.audit(
            "tenant-audit", job_id, revision, CodingAuditVerdict.REWORK,
            (CodingAuditFinding(
                "needs_more", CodingAuditSeverity.BLOCKER, "cover the audited path",
            ),),
        )
        receipt = settle(service, job_id)
        emit({
            "job_id": receipt.job_id,
            "state": receipt.state.value,
            "session": receipt.implementation_session_id,
            "revision": receipt.implementation_revision_sha256,
            "rework_count": receipt.rework_count,
        })
    else:
        raise SystemExit("unknown action")
finally:
    service.close()
'''


def _driver(tmp_path: Path) -> Path:
    path = tmp_path / "driver.py"
    path.write_text(_DRIVER, encoding="utf-8")
    return path


def _run(driver: Path, *args: str) -> dict:
    """Run one bounded child interpreter and return its single result payload."""

    completed = subprocess.run(
        (sys.executable, "-W", "error", str(driver), *args),
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert completed.returncode == 0, (
        "child failed\nstdout:\n{}\nstderr:\n{}".format(
            completed.stdout, completed.stderr,
        )
    )
    for line in completed.stdout.splitlines():
        if line.startswith("FLYTO_RESULT "):
            return json.loads(line[len("FLYTO_RESULT "):])
    raise AssertionError("child emitted no result: {}".format(completed.stdout))


def test_ownership_and_exact_session_survive_a_real_owner_process_exit(
    tmp_path: Path,
) -> None:
    """Three separate interpreters, one shared state root.

    Proves the production shape: the implementing process exits entirely, the
    worktree stays owned, and a different process resumes the same session.
    """

    workspace = tmp_path / "workspace"
    (workspace / ".flyto").mkdir(parents=True)
    (workspace / ".flyto" / "coding.yaml").write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: real_file_check\n"
        "    argv: {}\n".format(json.dumps([
            sys.executable,
            "-c",
            "from pathlib import Path; "
            "assert Path('result.txt').read_text() == 'verified\\n'",
        ])),
        encoding="utf-8",
    )
    state_root = str(tmp_path / "service-state")
    driver = _driver(tmp_path)

    # Process 1 implements, reaches the audit gap, then exits completely.
    owner = _run(driver, state_root, str(workspace), "implement")
    assert owner["state"] == CodingJobState.AWAITING_CODEX_AUDIT.value
    assert owner["session"]

    # The claim outlived the process that took it.
    claims = list((Path(state_root) / "locks" / "workspaces").glob("*.owner.json"))
    assert len(claims) == 1
    assert json.loads(claims[0].read_text(encoding="utf-8"))["job_id"] == owner["job_id"]

    # Process 2 is a competing Codex frontend on the same worktree.
    refused = _run(driver, state_root, str(workspace), "submit_should_fail")
    assert refused["error"] == "workspace_busy"
    assert refused["owner_job_id"] == owner["job_id"]

    # Process 3 audits and reworks. It never held the session in memory.
    reworked = _run(
        driver, state_root, str(workspace), "rework", "2",
        owner["job_id"], owner["revision"],
    )
    assert reworked["job_id"] == owner["job_id"]
    assert reworked["state"] == CodingJobState.AWAITING_CODEX_AUDIT.value
    assert reworked["rework_count"] == 1
    # The exact prior session continued; a fresh one would carry a new id.
    assert reworked["session"] == owner["session"]
    assert reworked["revision"] != owner["revision"]
