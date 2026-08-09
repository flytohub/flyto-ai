import hashlib
import json
import sys
import time
from pathlib import Path

import pytest
import subprocess

from flyto_ai.coding.mcp_supervisor import (
    DEFAULT_CODING_STATE_DIR,
    WORKER_SHUTDOWN_TIMEOUT_SECONDS,
    WORKER_HANDSHAKE_TIMEOUT_SECONDS,
    WORKER_RESPONSE_TIMEOUT_SECONDS,
    CodingMCPWorkerSupervisor,
    _option_value,
    durable_job_state_reader,
    supervisor_from_argv,
    worker_argv_from_process,
)


_WORKER = r"""
import json
import pathlib
import sys

version = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
for raw in sys.stdin.buffer:
    request = json.loads(raw)
    request_id = request.get("id")
    if request_id is None:
        continue
    method = request.get("method")
    if method == "initialize":
        result = {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "fake", "version": version},
        }
    elif method == "tools/list":
        result = {"tools": [{"name": version}]}
    elif method == "tools/call":
        params = request.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments", {})
        if name == "flyto_coding_submit":
            job_id = "job_" + "a" * 24
            state = "queued"
        elif name == "flyto_coding_get":
            job_id = arguments.get("job_id", "job_" + "a" * 24)
            state = "failed"
        else:
            job_id = arguments.get("job_id", "job_" + "a" * 24)
            state = "codex_accepted"
        payload = {"ok": True, "job": {"job_id": job_id, "state": state}}
        result = {
            "content": [{"type": "text", "text": json.dumps(payload)}],
            "isError": False,
            "structuredContent": payload,
        }
    else:
        result = {}
    response = {"jsonrpc": "2.0", "id": request_id, "result": result}
    sys.stdout.buffer.write(json.dumps(response).encode() + b"\n")
    sys.stdout.buffer.flush()
"""


#: A worker that accepts a request and then never answers, so the supervisor's
#: deadline is the only thing that can end the exchange.
_WEDGED_WORKER = r"""
import sys
import time

for raw in sys.stdin.buffer:
    time.sleep(600)
"""


def _line(request):
    return json.dumps(request, separators=(",", ":")).encode() + b"\n"


def _value(raw):
    assert raw is not None
    return json.loads(raw)


def _initialize(supervisor):
    response = supervisor.handle_line(_line({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18"},
    }))
    assert _value(response)["result"]["serverInfo"]["name"] == "fake"
    assert supervisor.handle_line(_line({
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    })) is None


def _supervisor(tmp_path, build):
    script = tmp_path / "worker.py"
    marker = tmp_path / "version.txt"
    script.write_text(_WORKER, encoding="utf-8")
    marker.write_text(build[0], encoding="utf-8")
    supervisor = CodingMCPWorkerSupervisor(
        (sys.executable, "-u", str(script), str(marker)),
        build_id_provider=lambda: build[0],
    )
    return supervisor, marker


def test_worker_argv_replaces_only_the_supervisor_subcommand():
    argv = (
        "flyto-ai", "code-mcp-supervisor", "--tenant", "local-codex",
        "--workspace-root", "/workspace",
    )
    worker = worker_argv_from_process(argv)
    assert worker[:4] == (sys.executable, "-m", "flyto_ai.cli", "code-mcp")
    assert worker[4:] == argv[2:]


def test_a_safe_source_change_reloads_only_the_worker_and_replays_handshake(tmp_path):
    build = ["build-one"]
    supervisor, marker = _supervisor(tmp_path, build)
    try:
        _initialize(supervisor)
        first_pid = supervisor.worker_pid
        first = _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        })))
        assert first["result"]["tools"][0]["name"] == "build-one"

        marker.write_text("build-two", encoding="utf-8")
        build[0] = "build-two"
        second = _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {},
        })))
        assert second["result"]["tools"][0]["name"] == "build-two"
        assert supervisor.worker_pid != first_pid
        assert supervisor.reload_count == 1
    finally:
        supervisor.close()


def test_source_change_preserves_active_job_and_blocks_only_new_submissions(tmp_path):
    build = ["build-one"]
    supervisor, marker = _supervisor(tmp_path, build)
    job_id = "job_" + "a" * 24
    try:
        _initialize(supervisor)
        original_pid = supervisor.worker_pid
        queued = _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "flyto_coding_submit", "arguments": {}},
        })))
        assert queued["result"]["structuredContent"]["job"]["state"] == "queued"

        marker.write_text("build-two", encoding="utf-8")
        build[0] = "build-two"
        blocked = _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "flyto_coding_submit", "arguments": {}},
        })))
        assert blocked["result"]["structuredContent"] == {
            "ok": False, "error": "service_reload_pending",
        }
        assert supervisor.worker_pid == original_pid

        failed = _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "flyto_coding_get", "arguments": {"job_id": job_id},
            },
        })))
        assert failed["result"]["structuredContent"]["job"]["state"] == "failed"

        listed = _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0", "id": 5, "method": "tools/list", "params": {},
        })))
        assert listed["result"]["tools"][0]["name"] == "build-two"
        assert supervisor.worker_pid != original_pid
        assert supervisor.reload_count == 1
    finally:
        supervisor.close()


def _job_record(state_dir: Path, tenant: str, job_id: str, state: str) -> Path:
    """Write one durable job record exactly where the service would put it."""

    tenant_ref = hashlib.sha256(tenant.encode()).hexdigest()
    jobs = state_dir / "tenants" / tenant_ref / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    path = jobs / (job_id + ".json")
    path.write_text(json.dumps({"job_id": job_id, "state": state}), encoding="utf-8")
    return path


def test_read_deadlines_are_bounded_to_thirty_seconds() -> None:
    """Submit, get, and audit all return promptly; a longer wait is a wedge."""

    assert WORKER_RESPONSE_TIMEOUT_SECONDS <= 30.0
    assert WORKER_HANDSHAKE_TIMEOUT_SECONDS <= 30.0
    with pytest.raises(ValueError):
        CodingMCPWorkerSupervisor(("true",), response_timeout=31.0)
    with pytest.raises(ValueError):
        CodingMCPWorkerSupervisor(("true",), handshake_timeout=0)


def test_the_supervisor_state_dir_default_matches_the_cli() -> None:
    """A drifted default would silently disable durable self-healing."""

    import flyto_ai.cli as cli

    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert '"--state-dir", default="{}"'.format(DEFAULT_CODING_STATE_DIR) in source


def test_a_wedged_worker_is_terminated_and_the_request_is_not_retried(
    tmp_path,
) -> None:
    """A read that misses its deadline ends bounded, and delivery is uncertain.

    The supervisor must not resend: the worker may already have created the
    job. Recovery belongs to the caller, replaying the same idempotency key.
    """

    script = tmp_path / "wedged.py"
    script.write_text(_WEDGED_WORKER, encoding="utf-8")
    supervisor = CodingMCPWorkerSupervisor(
        (sys.executable, "-u", str(script)),
        build_id_provider=lambda: "build-one",
        response_timeout=0.4,
        handshake_timeout=0.4,
    )
    try:
        started = time.monotonic()
        response = _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "flyto_coding_submit", "arguments": {}},
        })))
        elapsed = time.monotonic() - started

        assert response["error"]["code"] == -32603
        assert "deadline" in response["error"]["message"]
        assert response["id"] == 2
        assert elapsed < 30
        assert supervisor.timeout_count == 1
        # The wedged child is gone, so the state-root locks it held are freed.
        assert supervisor.worker_pid == 0
    finally:
        supervisor.close()


def test_a_later_request_starts_a_fresh_worker_after_a_timeout(tmp_path) -> None:
    """Recovery is a new worker, which reconciles interrupted jobs truthfully."""

    wedged = tmp_path / "wedged.py"
    wedged.write_text(_WEDGED_WORKER, encoding="utf-8")
    supervisor = CodingMCPWorkerSupervisor(
        (sys.executable, "-u", str(wedged)),
        build_id_provider=lambda: "build-one",
        response_timeout=0.4,
        handshake_timeout=0.4,
    )
    try:
        first = _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })))
        assert first["error"]["code"] == -32603
        assert supervisor.worker_pid == 0

        # A later call is served by a brand-new child rather than the corpse.
        second = _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        })))
        assert second["error"]["code"] == -32603
        assert supervisor.timeout_count == 2
    finally:
        supervisor.close()


def test_a_stale_active_job_reloads_once_durable_state_is_terminal(
    tmp_path,
) -> None:
    """A client that stops polling must not block reloads forever."""

    build = ["build-one"]
    supervisor, marker = _supervisor(tmp_path, build)
    state_dir = tmp_path / "state"
    tenant = "local-codex"
    job_id = "job_" + "a" * 24
    _job_record(state_dir, tenant, job_id, "running")
    supervisor._job_state_reader = durable_job_state_reader(str(state_dir), tenant)
    try:
        _initialize(supervisor)
        original_pid = supervisor.worker_pid
        supervisor.handle_line(_line({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "flyto_coding_submit", "arguments": {}},
        }))
        assert job_id in supervisor._active_jobs

        # Source changes while the job is genuinely still running: the worker
        # is preserved and only new submissions are refused.
        marker.write_text("build-two", encoding="utf-8")
        build[0] = "build-two"
        blocked = _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "flyto_coding_submit", "arguments": {}},
        })))
        assert blocked["result"]["structuredContent"]["error"] == "service_reload_pending"
        assert supervisor.worker_pid == original_pid

        # The client never polls again, but the job really did settle. The
        # durable record — not the in-memory set — releases the reload.
        _job_record(state_dir, tenant, job_id, "codex_accepted")
        listed = _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": {},
        })))
        assert listed["result"]["tools"][0]["name"] == "build-two"
        assert supervisor.worker_pid != original_pid
        assert supervisor.reload_count == 1
        assert supervisor._active_jobs == {}
    finally:
        supervisor.close()


def test_self_healing_needs_every_tracked_job_to_be_terminal(tmp_path) -> None:
    """Multiple concurrent jobs: one settled job must not release the worker."""

    build = ["build-one"]
    supervisor, marker = _supervisor(tmp_path, build)
    state_dir = tmp_path / "state"
    tenant = "local-codex"
    settled = "job_" + "a" * 24
    live = "job_" + "b" * 24
    _job_record(state_dir, tenant, settled, "failed")
    _job_record(state_dir, tenant, live, "awaiting_codex_audit")
    supervisor._job_state_reader = durable_job_state_reader(str(state_dir), tenant)
    supervisor._active_jobs = {settled: "running", live: "running"}
    try:
        _initialize(supervisor)
        original_pid = supervisor.worker_pid
        marker.write_text("build-two", encoding="utf-8")
        build[0] = "build-two"

        blocked = _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "flyto_coding_submit", "arguments": {}},
        })))
        assert blocked["result"]["structuredContent"]["error"] == "service_reload_pending"
        # The settled job was dropped; the audit-bound one still holds the worker.
        assert supervisor._active_jobs == {live: "running"}
        assert supervisor.worker_pid == original_pid

        _job_record(state_dir, tenant, live, "codex_accepted")
        _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {},
        })))
        assert supervisor.worker_pid != original_pid
        assert supervisor._active_jobs == {}
    finally:
        supervisor.close()


def test_the_durable_reader_is_bounded_and_refuses_unsafe_identifiers(
    tmp_path,
) -> None:
    state_dir = tmp_path / "state"
    tenant = "local-codex"
    job_id = "job_" + "c" * 24
    _job_record(state_dir, tenant, job_id, "queued")
    reader = durable_job_state_reader(str(state_dir), tenant)
    assert reader is not None
    assert reader(job_id) == "queued"
    # Traversal, wrong shape, and unknown ids all read as "unknown", never as
    # a state that could release a worker.
    assert reader("../../etc/passwd") is None
    assert reader("job_zz") is None
    assert reader("job_" + "d" * 24) is None
    # An unusable configuration disables self-healing rather than guessing.
    assert durable_job_state_reader("", tenant) is None
    assert durable_job_state_reader(str(state_dir), "../evil") is None


def _assert_released(channel) -> None:
    """A stopped channel leaves no open pipe and no live reader thread."""

    assert channel.process.poll() is not None, "worker process is still running"
    assert not channel._reader.is_alive(), "stdout reader thread is still alive"
    for name in ("stdin", "stdout"):
        pipe = getattr(channel.process, name)
        assert pipe is not None and pipe.closed, "worker {} is still open".format(name)


def test_a_graceful_reload_releases_the_replaced_worker(tmp_path) -> None:
    """Hot reload must not leak the pipes or reader of the worker it replaces."""

    build = ["build-one"]
    supervisor, marker = _supervisor(tmp_path, build)
    try:
        _initialize(supervisor)
        replaced = supervisor._channel
        marker.write_text("build-two", encoding="utf-8")
        build[0] = "build-two"
        _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        })))
        assert supervisor._channel is not replaced
        _assert_released(replaced)
    finally:
        supervisor.close()


def test_close_releases_the_current_worker(tmp_path) -> None:
    build = ["build-one"]
    supervisor, _marker = _supervisor(tmp_path, build)
    _initialize(supervisor)
    channel = supervisor._channel
    supervisor.close()
    assert supervisor._channel is None
    _assert_released(channel)


def test_a_timeout_releases_the_wedged_worker(tmp_path) -> None:
    """Terminating a wedged worker must also reclaim its pipes and reader."""

    script = tmp_path / "wedged.py"
    script.write_text(_WEDGED_WORKER, encoding="utf-8")
    supervisor = CodingMCPWorkerSupervisor(
        (sys.executable, "-u", str(script)),
        build_id_provider=lambda: "build-one",
        response_timeout=0.4,
        handshake_timeout=0.4,
    )
    try:
        supervisor._ensure_worker()
        channel = supervisor._channel
        response = _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })))
        assert response["error"]["code"] == -32603
        assert supervisor.timeout_count == 1
        _assert_released(channel)
    finally:
        supervisor.close()


#: A live worker that answers with a frame no client may act on. Its stream is
#: desynchronized, so the supervisor must release it rather than keep it.
_MALFORMED_WORKER = r"""
import sys

for raw in sys.stdin.buffer:
    sys.stdout.buffer.write(b"x" * (256 * 1024 + 8) + b"\n")
    sys.stdout.buffer.flush()
"""


def test_an_oversized_live_worker_is_released_and_replaced(tmp_path) -> None:
    """A desynchronized stream is released, and the request is not retried.

    Keeping such a worker would let the next request read a leftover frame and
    answer the wrong caller.
    """

    script = tmp_path / "malformed.py"
    script.write_text(_MALFORMED_WORKER, encoding="utf-8")
    supervisor = CodingMCPWorkerSupervisor(
        (sys.executable, "-u", str(script)),
        build_id_provider=lambda: "build-one",
        response_timeout=2.0,
        handshake_timeout=2.0,
    )
    try:
        supervisor._ensure_worker()
        channel = supervisor._channel
        first = _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })))
        assert first["error"]["code"] == -32603
        assert supervisor._channel is None
        _assert_released(channel)

        # A later request is served by a brand-new worker.
        second_channel_before = supervisor._channel
        _value(supervisor.handle_line(_line({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        })))
        assert second_channel_before is None
        assert supervisor.timeout_count == 0
    finally:
        supervisor.close()


def test_the_post_kill_reap_is_bounded(tmp_path) -> None:
    """An unreapable process must not block the client's stdio loop forever."""

    script = tmp_path / "wedged.py"
    script.write_text(_WEDGED_WORKER, encoding="utf-8")
    supervisor = CodingMCPWorkerSupervisor(
        (sys.executable, "-u", str(script)),
        build_id_provider=lambda: "build-one",
        response_timeout=0.3,
        handshake_timeout=0.3,
    )
    supervisor._ensure_worker()
    channel = supervisor._channel
    waits = []

    class _Unreapable:
        """Never exits, so every wait must time out."""

        def __init__(self, real):
            self._real = real
            self.pid = real.pid

        def poll(self):
            return None

        def wait(self, timeout=None):
            waits.append(timeout)
            raise subprocess.TimeoutExpired(cmd="worker", timeout=timeout)

        def terminate(self):
            return None

        def kill(self):
            return None

        def __getattr__(self, name):
            return getattr(self._real, name)

    real = channel.process
    try:
        channel.process = _Unreapable(real)
        started = time.monotonic()
        channel._end_process(graceful=False)
        elapsed = time.monotonic() - started
        # terminate-wait and kill-wait, both bounded; never an unbounded wait().
        assert waits and all(value is not None for value in waits), waits
        assert elapsed < 3 * WORKER_SHUTDOWN_TIMEOUT_SECONDS
    finally:
        channel.process = real
        supervisor.close()


def test_release_closes_pipes_and_reaps_a_reader_that_outlives_one_join(
    tmp_path,
) -> None:
    """A first join that times out must not leave a reader or pipe alive."""

    build = ["build-one"]
    supervisor, _marker = _supervisor(tmp_path, build)
    _initialize(supervisor)
    channel = supervisor._channel
    joins = []
    real_join = channel._reader.join

    def counting_join(timeout=None):
        joins.append(timeout)
        if len(joins) == 1:
            # Simulate a reader still blocked on the pipe at the first attempt.
            return None
        return real_join(timeout)

    try:
        channel._reader.join = counting_join
        channel.stop()
        assert len(joins) >= 1
        assert all(value is not None for value in joins), joins
        for name in ("stdin", "stdout"):
            assert getattr(channel.process, name).closed
        assert not channel._reader.is_alive()
    finally:
        channel._reader.join = real_join
        supervisor.close()


def test_an_omitted_state_dir_still_enables_durable_reconciliation() -> None:
    """An absent flag means the CLI default, not "no durable state"."""

    argv = (
        "flyto-ai", "code-mcp-supervisor", "--tenant", "local-codex",
        "--workspace-root", "/workspace",
    )
    assert _option_value(argv, "--state-dir") == ""
    supervisor = supervisor_from_argv(argv)
    try:
        # Reconciliation is wired up rather than silently disabled.
        assert supervisor._job_state_reader is not None
        assert supervisor._job_state_reader("job_" + "a" * 24) is None
    finally:
        supervisor.close()

    expected = durable_job_state_reader(DEFAULT_CODING_STATE_DIR, "local-codex")
    assert expected is not None


def test_option_values_are_read_from_either_flag_spelling() -> None:
    argv = (
        "flyto-ai", "code-mcp-supervisor", "--tenant", "local-codex",
        "--state-dir=/srv/state", "--workspace-root", "/workspace",
    )
    assert _option_value(argv, "--tenant") == "local-codex"
    assert _option_value(argv, "--state-dir") == "/srv/state"
    assert _option_value(argv, "--missing") == ""
