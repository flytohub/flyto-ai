import json
import sys

from flyto_ai.coding.mcp_supervisor import (
    CodingMCPWorkerSupervisor,
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
