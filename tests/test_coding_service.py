from __future__ import annotations

import asyncio
import dataclasses
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from flyto_ai.coding import (
    CapabilityManager,
    CapabilitySpec,
    CheckSpec,
    CodingTaskRequest,
    FlytoCodingAgent,
)
from flyto_ai.coding.contracts import CodingJobState
from flyto_ai.coding.http_server import build_http_server
from flyto_ai.coding.mcp_server import CodingMCPServer, MCP_PROTOCOL_VERSION
from flyto_ai.coding.service import (
    CodingJobNotFound,
    CodingService,
    IdempotencyConflict,
    request_from_mapping,
    receipt_to_mapping,
)

TEST_BEARER_TOKEN = "unit-test-bearer-token"


class RealToolProvider:
    """Small deterministic provider harness; all effects use the real tool boundary."""

    active = 0
    max_active = 0
    lock = threading.Lock()

    def __init__(self, content: str = "verified\n", delay: float = 0.0) -> None:
        self.content = content
        self.delay = delay

    async def chat(self, **kwargs):
        with self.lock:
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            result = await kwargs["dispatch_fn"](
                "coding_write_file", {
                    "path": "result.txt", "content": self.content, "overwrite": True,
                },
            )
            assert result["ok"]
            return "done", [{"function": "coding_write_file", "ok": True}], 1, {"total_tokens": 1}
        finally:
            with self.lock:
                type(self).active -= 1


def _request(
    workspace: Path,
    *,
    message: str = "write verified result",
    require_changes: bool = True,
) -> CodingTaskRequest:
    return CodingTaskRequest(
        message=message,
        working_dir=str(workspace),
        require_changes=require_changes,
    )


def _wait(service: CodingService, tenant: str, job_id: str, timeout: float = 10) -> object:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        receipt = service.get(tenant, job_id)
        if receipt.state in {CodingJobState.COMPLETED, CodingJobState.FAILED}:
            return receipt
        time.sleep(0.02)
    raise AssertionError("coding job did not finish")


def _service(tmp_path: Path, workspace: Path, *, delay: float = 0.0) -> CodingService:
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir(exist_ok=True)
    config.write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: real_file_check\n"
        "    argv: {}\n".format(json.dumps([
            sys.executable,
            "-c",
            "from pathlib import Path; assert Path('result.txt').read_text() == 'verified\\n'",
        ]))
    )
    return CodingService(
        lambda store: FlytoCodingAgent(RealToolProvider(delay=delay), store=store),
        state_root=str(tmp_path / "service-state"),
        workspace_roots=(str(workspace),),
        max_workers=2,
        max_queued=8,
    )


def test_service_runs_real_tools_checks_idempotently_and_isolates_tenants(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _service(tmp_path, workspace)
    try:
        untrusted_request = _request(workspace)
        untrusted_request.checks = (CheckSpec(
            "unsafe_remote_check", (sys.executable, "-c", "raise SystemExit(99)"),
        ),)
        queued = service.submit("tenant-a", "request-001", untrusted_request)
        duplicate = service.submit("tenant-a", "request-001", _request(workspace))
        assert duplicate.job_id == queued.job_id
        with pytest.raises(IdempotencyConflict):
            service.submit("tenant-a", "request-001", _request(workspace, message="different"))
        with pytest.raises(CodingJobNotFound):
            service.get("tenant-b", queued.job_id)

        completed = _wait(service, "tenant-a", queued.job_id)
        assert completed.state is CodingJobState.COMPLETED
        assert completed.result is not None and completed.result.ok
        assert completed.result.checks[0].passed
        assert completed.result.checks[0].name == "real_file_check"
        assert len(completed.evidence_sha256) == 64
        public = receipt_to_mapping(completed)
        assert "evidence_path" not in public["result"]
        assert "output_preview" not in public["result"]["checks"][0]
        assert (workspace / "result.txt").read_text() == "verified\n"
    finally:
        service.close()

    persisted = _service(tmp_path, workspace)
    try:
        assert persisted.get("tenant-a", queued.job_id).state is CodingJobState.COMPLETED
    finally:
        persisted.close()


def test_service_serializes_jobs_that_share_a_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    RealToolProvider.active = 0
    RealToolProvider.max_active = 0
    service = _service(tmp_path, workspace, delay=0.1)
    try:
        first = service.submit(
            "tenant-a", "parallel-001", _request(workspace, message="first", require_changes=False),
        )
        second = service.submit(
            "tenant-a", "parallel-002", _request(workspace, message="second", require_changes=False),
        )
        assert _wait(service, "tenant-a", first.job_id).state is CodingJobState.COMPLETED
        assert _wait(service, "tenant-a", second.job_id).state is CodingJobState.COMPLETED
        assert RealToolProvider.max_active == 1
    finally:
        service.close()


def test_http_server_requires_auth_rejects_provider_fields_and_runs_job(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _service(tmp_path, workspace)
    server = build_http_server(
        service, tenant_id="tenant-http", auth_token=TEST_BEARER_TOKEN, port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = "http://127.0.0.1:{}/v1/coding/jobs".format(server.server_address[1])
    try:
        body = json.dumps({"message": "task", "working_dir": str(workspace), "api_key": "forbidden"}).encode()
        request = Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json", "Idempotency-Key": "http-001",
        })
        with pytest.raises(HTTPError) as unauthorized:
            urlopen(request)
        assert unauthorized.value.code == 401

        request.add_header("Authorization", f"Bearer {TEST_BEARER_TOKEN}")
        with pytest.raises(HTTPError) as invalid:
            urlopen(request)
        assert invalid.value.code == 400

        body = json.dumps({
            "message": "task",
            "working_dir": str(workspace),
        }).encode()
        request = Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Idempotency-Key": "http-002",
            "Authorization": f"Bearer {TEST_BEARER_TOKEN}",
        })
        response = json.loads(urlopen(request).read())
        completed = _wait(service, "tenant-http", response["job"]["job_id"])
        assert completed.state is CodingJobState.COMPLETED
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        service.close()


def test_mcp_negotiates_protocol_and_exposes_tenant_bound_tools(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _service(tmp_path, workspace)
    server = CodingMCPServer(service, "tenant-mcp")
    try:
        rejected = server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "old"},
        })
        assert rejected and rejected["error"]["code"] == -32602
        initialized = server.handle({
            "jsonrpc": "2.0", "id": 2, "method": "initialize",
            "params": {"protocolVersion": MCP_PROTOCOL_VERSION},
        })
        assert initialized and initialized["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
        listed = server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}})
        assert {item["name"] for item in listed["result"]["tools"]} == {
            "flyto_coding_submit", "flyto_coding_get",
        }
    finally:
        service.close()


def test_capability_preflight_uses_real_negotiation_and_required_tool_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = tmp_path / "capability_server.py"
    fixture.write_text(
        "import json, os, sys\n"
        "ready=(os.environ.get('FLYTO_TEST_CAPABILITY_TOKEN')=='runtime-secret' and 'AWS_SECRET_ACCESS_KEY' not in os.environ)\n"
        "for line in sys.stdin:\n"
        " msg=json.loads(line)\n"
        " if 'id' not in msg: continue\n"
        " if msg['method']=='initialize': out={'protocolVersion':'2025-06-18','capabilities':{},'serverInfo':{'name':'real-fixture','version':'1'}}\n"
        " elif msg['method']=='tools/list': out={'tools':([{'name':'context','inputSchema':{'type':'object'}}] if ready else [])}\n"
        " else: out={}\n"
        " print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':out}), flush=True)\n"
    )

    monkeypatch.setenv("FLYTO_TEST_CAPABILITY_TOKEN", "runtime-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-cross-boundary")

    async def scenario(required_tool: str):
        manager = CapabilityManager(str(tmp_path))
        try:
            return (await manager.start((CapabilitySpec(
                name="real-capability",
                argv=(sys.executable, str(fixture)),
                required=True,
                required_tools=(required_tool,),
                env_passthrough=("FLYTO_TEST_CAPABILITY_TOKEN",),
            ),)))[0]
        finally:
            await manager.close()

    available = asyncio.run(scenario("context"))
    assert available.available
    assert available.negotiated_protocol_version == "2025-06-18"
    assert available.server_name == "real-fixture"
    assert available.tools == ("context",)
    assert "runtime-secret" not in json.dumps(dataclasses.asdict(available))
    unavailable = asyncio.run(scenario("impact"))
    assert not unavailable.available
    assert unavailable.missing_tools == ("impact",)


def test_code_mcp_cli_runs_as_a_real_stdio_process(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "flyto_ai.cli",
            "code-mcp",
            "--tenant",
            "tenant-cli",
            "--workspace-root",
            str(workspace),
            "--state-dir",
            str(tmp_path / "mcp-state"),
            "--provider",
            "ollama",
        ],
        cwd=str(Path(__file__).parents[1]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    requests = "\n".join([
        json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": MCP_PROTOCOL_VERSION},
        }),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
        "",
    ])
    stdout, stderr = process.communicate(requests, timeout=10)
    assert process.returncode == 0, stderr
    responses = [json.loads(line) for line in stdout.splitlines()]
    assert responses[0]["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert {tool["name"] for tool in responses[1]["result"]["tools"]} == {
        "flyto_coding_submit", "flyto_coding_get",
    }


def test_service_mapping_forbids_authority_fields_and_capability_contract_is_typed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="unsupported coding request fields"):
        request_from_mapping({"message": "task", "working_dir": str(workspace), "provider": "openai"})
    with pytest.raises(ValueError, match="unsupported coding request fields"):
        request_from_mapping({
            "message": "task", "working_dir": str(workspace),
            "checks": [{"name": "unsafe", "argv": ["sh"]}],
        })
    capability = CapabilitySpec.from_mapping({
        "name": "core",
        "argv": ["flyto-core", "mcp"],
        "required_tools": ["browser_navigate"],
    })
    assert capability.required_tools == ("browser_navigate",)
    with pytest.raises(ValueError, match="only explicit FLYTO"):
        CapabilitySpec(name="unsafe", argv=("tool",), env_passthrough=("AWS_SECRET_ACCESS_KEY",))
    with pytest.raises(ValueError, match="required_tools must be"):
        CapabilitySpec.from_mapping({
            "name": "invalid", "argv": ["tool"], "required_tools": "context",
        })
