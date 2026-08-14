from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from flyto_ai.coding import (
    ApprovalPolicy,
    CapabilityManager,
    CapabilitySpec,
    CheckSpec,
    CodingTaskRequest,
    FlytoCodingAgent,
    SandboxMode,
)
from flyto_ai.coding.contracts import (
    MAX_AUDIT_FINDINGS,
    MAX_AUDIT_MESSAGE_CHARS,
    SERVICE_CONTRACT_VERSION,
    CodingAuditFinding,
    CodingAuditSeverity,
    CodingAuditVerdict,
    CodingJobReceipt,
    CodingJobState,
    TERMINAL_CODING_JOB_STATES,
    audit_findings_sha256,
    validate_audit_submission,
)
from flyto_ai.coding.http_server import CodingHTTPHandler, build_http_server
from flyto_ai.coding.mcp_server import (
    _AUDIT_ARGUMENT_FIELDS,
    CodingMCPServer,
    MCP_PROTOCOL_VERSION,
)
from flyto_ai.coding.service import (
    AuditBlockersUnresolved,
    AuditNotEnabled,
    AuditStateConflict,
    CodingAuthorityConflict,
    CodingJobNotFound,
    CodingService,
    IdempotencyConflict,
    RevisionMismatch,
    RevisionUnavailable,
    REWORK_LIMIT_FAILURE_CODE,
    ReworkLimitReached,
    ReworkNotResumable,
    VerificationRequired,
    error_details,
    request_from_mapping,
    receipt_to_mapping,
)

TEST_BEARER_TOKEN = "unit-test-bearer-token"
# The implementer stops at `awaiting_codex_audit`; waiting only for the legacy
# terminal states would hang once the phase-2 service lands that transition.
#: Preflight refuses a repository that never declared how it wants to be
#: verified, before a job, a worktree claim or an implementer session exists.
#: A fixture that means to exercise anything past preflight therefore has to
#: declare a contract, exactly as a real repository does.
_TEST_VERIFICATION_CONTRACT = """version: flyto.coding-config.v1
checks:
  - name: declared
    argv: [python, --version]
    timeout_seconds: 30
    required: true
"""


def _declare_verification(workspace) -> None:
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(_TEST_VERIFICATION_CONTRACT, encoding="utf-8")


_SETTLED_STATES = TERMINAL_CODING_JOB_STATES | {CodingJobState.AWAITING_CODEX_AUDIT}
_AUDIT_JOB_ID = "job_" + "a1b2c3d4" * 3
_AUDIT_REVISION = "b3" * 32


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
        if receipt.state in _SETTLED_STATES:
            return receipt
        time.sleep(0.02)
    raise AssertionError("coding job did not finish")


class ReworkingProvider:
    """Writes a passing result plus a per-round file so revisions differ."""

    def __init__(self, delay: float = 0.0) -> None:
        self.rounds = 0
        self.delay = delay
        self.prompts: list = []

    async def chat(self, **kwargs):
        self.rounds += 1
        self.prompts.append(json.dumps(kwargs.get("messages", []), ensure_ascii=False))
        if self.delay:
            await asyncio.sleep(self.delay)
        for path, content in (
            ("result.txt", "verified\n"),
            ("notes.txt", "round {}\n".format(self.rounds)),
        ):
            result = await kwargs["dispatch_fn"](
                "coding_write_file", {"path": path, "content": content, "overwrite": True},
            )
            assert result["ok"]
        return "done", [{"function": "coding_write_file", "ok": True}], 1, {"total_tokens": 1}


class PartialReworkProvider:
    """Round 1 changes two files; later rounds leave one of them untouched."""

    def __init__(self) -> None:
        self.rounds = 0

    async def chat(self, **kwargs):
        self.rounds += 1
        writes = [("result.txt", "verified\n")]
        if self.rounds == 1:
            writes.append(("helper.txt", "original helper\n"))
        else:
            writes.append(("notes.txt", "round {}\n".format(self.rounds)))
        for path, content in writes:
            result = await kwargs["dispatch_fn"](
                "coding_write_file", {"path": path, "content": content, "overwrite": True},
            )
            assert result["ok"]
        return "done", [{"function": "coding_write_file", "ok": True}], 1, {"total_tokens": 1}


class NoChangeProvider:
    """Produces no attributable change, so the implementation must fail."""

    async def chat(self, **kwargs):
        return "nothing to do", [], 1, {"total_tokens": 1}


def _service(
    tmp_path: Path,
    workspace: Path,
    *,
    delay: float = 0.0,
    provider: object = None,
    require_codex_audit: bool = False,
    max_rework_rounds: int = 3,
    extra_roots: tuple = (),
) -> CodingService:
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
        lambda store: FlytoCodingAgent(provider or RealToolProvider(delay=delay), store=store),
        state_root=str(tmp_path / "service-state"),
        # The configured tree set is part of startup authority, so two services
        # sharing a state root must declare the same one. `extra_roots` lets a
        # fixture that drives several worktrees through one root do exactly
        # that, rather than presenting two semantically different workers.
        workspace_roots=(str(workspace), *(str(root) for root in extra_roots)),
        max_workers=2,
        max_queued=8,
        require_codex_audit=require_codex_audit,
        max_rework_rounds=max_rework_rounds,
    )


def test_service_runs_real_tools_checks_idempotently_and_isolates_tenants(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
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
    _declare_verification(workspace)
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


def test_services_share_one_state_root_without_reconciling_live_jobs(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    RealToolProvider.active = 0
    RealToolProvider.max_active = 0
    first = _service(tmp_path, workspace, delay=0.15)
    second = None
    try:
        request = _request(workspace, message="shared", require_changes=False)
        queued = first.submit("tenant-a", "shared-001", request)

        # A second MCP process constructs another service over the same durable
        # root. It must not fail startup or mark the live first job interrupted.
        second = _service(tmp_path, workspace, delay=0.15)
        duplicate = second.submit("tenant-a", "shared-001", request)
        assert duplicate.job_id == queued.job_id
        assert duplicate.state in {CodingJobState.QUEUED, CodingJobState.RUNNING}

        completed = _wait(second, "tenant-a", queued.job_id)
        assert completed.state is CodingJobState.COMPLETED
        assert RealToolProvider.max_active == 1
    finally:
        if second is not None:
            second.close()
        first.close()


def test_admitter_releases_host_authority_after_a_peer_settles_last_job(
    tmp_path: Path,
) -> None:
    """A peer's terminal write cannot leave the submitter holding forever."""

    from flyto_ai.coding.workspace_authority import describe_workspace_root

    workspace = tmp_path / "workspace-peer-release"
    workspace.mkdir()
    _declare_verification(workspace)
    state_root = tmp_path / "shared-peer-release-state"
    registry = tmp_path / "shared-peer-release-registry"
    provider = ReworkingProvider()
    kwargs = {
        "state_root": str(state_root),
        "workspace_roots": (str(workspace),),
        "workspace_registry_root": str(registry),
        "max_workers": 2,
        "max_queued": 8,
        "require_codex_audit": True,
    }
    first = CodingService(
        lambda store: FlytoCodingAgent(provider, store=store), **kwargs,
    )
    second = CodingService(
        lambda store: FlytoCodingAgent(provider, store=store), **kwargs,
    )
    try:
        awaiting = _awaiting(first, "tenant-peer", "peer-release-001", workspace)
        assert first._workspace_root_authority is not None
        assert second._workspace_root_authority is None
        assert describe_workspace_root(registry, workspace)["status"] == "live"

        accepted = second.audit(
            "tenant-peer",
            awaiting.job_id,
            awaiting.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT,
            (),
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED

        deadline = time.monotonic() + 3
        while first._workspace_root_authority is not None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert first._workspace_root_authority is None
        assert describe_workspace_root(registry, workspace)["status"] == "adoptable"
    finally:
        second.close()
        first.close()


def test_services_serialize_distinct_jobs_for_one_workspace_across_instances(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    RealToolProvider.active = 0
    RealToolProvider.max_active = 0
    first = _service(tmp_path, workspace, delay=0.1)
    second = _service(tmp_path, workspace, delay=0.1)
    try:
        one = first.submit(
            "tenant-a", "shared-parallel-001",
            _request(workspace, message="one", require_changes=False),
        )
        two = second.submit(
            "tenant-a", "shared-parallel-002",
            _request(workspace, message="two", require_changes=False),
        )
        assert _wait(first, "tenant-a", one.job_id).state is CodingJobState.COMPLETED
        assert _wait(second, "tenant-a", two.job_id).state is CodingJobState.COMPLETED
        assert RealToolProvider.max_active == 1
    finally:
        second.close()
        first.close()


def test_http_server_requires_auth_rejects_provider_fields_and_runs_job(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
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
    _declare_verification(workspace)
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
            "flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit",
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
    _declare_verification(workspace)
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
    try:
        # This proves protocol behavior, not cold-import performance. On busy
        # hosts the isolated CLI imports the full public package before serving.
        stdout, stderr = process.communicate(requests, timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    assert process.returncode == 0, stderr
    responses = [json.loads(line) for line in stdout.splitlines()]
    assert responses[0]["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert {tool["name"] for tool in responses[1]["result"]["tools"]} == {
        "flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit",
    }


def test_service_mapping_forbids_authority_fields_and_capability_contract_is_typed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    with pytest.raises(ValueError, match="unsupported coding request fields"):
        request_from_mapping({"message": "task", "working_dir": str(workspace), "provider": "openai"})
    with pytest.raises(ValueError, match="unsupported coding request fields"):
        request_from_mapping({
            "message": "task", "working_dir": str(workspace),
            "checks": [{"name": "unsafe", "argv": ["sh"]}],
        })
    decoded = request_from_mapping({
        "message": "task",
        "working_dir": str(workspace),
        "repository_roots": [str(workspace)],
        "owner_ref": "codex-task-1",
    })
    assert decoded.repository_roots == (str(workspace),)
    assert decoded.owner_ref == "codex-task-1"
    with pytest.raises(ValueError, match="array of paths"):
        request_from_mapping({
            "message": "task",
            "working_dir": str(workspace),
            "repository_roots": str(workspace),
        })
    with pytest.raises(ValueError, match="array of paths"):
        request_from_mapping({
            "message": "task", "working_dir": str(workspace),
            "repository_roots": [],
        })
    with pytest.raises(ValueError, match="owner_ref must be a string"):
        request_from_mapping({
            "message": "task", "working_dir": str(workspace), "owner_ref": 7,
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


class RecordingAuditService:
    """Duck-typed service proving exactly what the MCP layer forwards."""

    def __init__(self) -> None:
        self.calls: list = []

    def audit(self, tenant_id, job_id, implementation_revision_sha256, verdict, findings):
        self.calls.append(
            (tenant_id, job_id, implementation_revision_sha256, verdict, findings),
        )
        accepted = verdict is CodingAuditVerdict.ACCEPT
        return CodingJobReceipt(
            job_id=job_id,
            state=CodingJobState.CODEX_ACCEPTED if accepted else CodingJobState.REWORK_QUEUED,
            submitted_at=1.0,
            updated_at=2.0,
            implementation_backend="native",
            implementation_session_id="opaque-session-token",
            implementation_revision_sha256=implementation_revision_sha256,
            audit_count=1,
            rework_count=0 if accepted else 1,
            audit_findings_sha256=audit_findings_sha256(findings),
            landable=accepted,
        )


def _audit_request(**arguments) -> dict:
    return {
        "jsonrpc": "2.0", "id": 9, "method": "tools/call",
        "params": {"name": "flyto_coding_audit", "arguments": arguments},
    }


def _valid_finding_payload() -> dict:
    return {
        "code": "missing_regression_test",
        "severity": "blocker",
        "message": "cover the rework transition",
        "evidence_ref": "tests/test_coding_service.py:120",
    }


def test_initialize_declares_the_host_owned_audit_loop() -> None:
    from flyto_ai.coding.mcp_server import (
        CODING_MCP_INSTRUCTIONS,
        CODING_MCP_SERVER_VERSION,
        MAX_INSTRUCTIONS_CHARS,
    )

    server = CodingMCPServer(RecordingAuditService(), "tenant-audit")
    result = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": MCP_PROTOCOL_VERSION},
    })["result"]
    assert result["protocolVersion"] == MCP_PROTOCOL_VERSION
    assert result["serverInfo"] == {"name": "flyto-coding", "version": "2"}
    assert CODING_MCP_SERVER_VERSION == "2"

    instructions = result["instructions"]
    assert isinstance(instructions, str)
    assert instructions == CODING_MCP_INSTRUCTIONS
    assert 0 < len(instructions) <= MAX_INSTRUCTIONS_CHARS == 512

    # The loop must be readable without any out-of-band documentation.
    for tool in ("flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit"):
        assert tool in instructions
    for phrase in (
        "awaiting_codex_audit or terminal",
        "failed is terminal/non-landable",
        "At awaiting_codex_audit independently inspect/test the workspace",
        "audit exact implementation_revision_sha256",
        "rework sends typed findings to the same job/session",
        "poll and re-audit", "Only accept is landable",
        "host-authenticated auditor", "cannot prove caller identity",
        "Never stages, commits, pushes, publishes, or deploys",
    ):
        assert phrase in instructions
    # It must not claim an authority the transport does not have.
    assert "Codex is the caller" not in instructions
    # No remote model, provider, or backend selector is advertised.
    for forbidden in ("model", "provider", "backend"):
        assert forbidden not in instructions.lower()

    listed = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert {item["name"] for item in listed["result"]["tools"]} == {
        "flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit",
    }


def test_audit_tool_schema_is_strict_and_hides_backend_selection() -> None:
    server = CodingMCPServer(RecordingAuditService(), "tenant-audit")
    listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    tools = {item["name"]: item for item in listed["result"]["tools"]}
    schema = tools["flyto_coding_audit"]["inputSchema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "job_id", "implementation_revision_sha256", "verdict", "findings",
    }
    # The runtime handler must enforce exactly the declared key set; an MCP
    # host is not required to validate the schema for us.
    assert set(schema["properties"]) == set(_AUDIT_ARGUMENT_FIELDS)
    assert schema["properties"]["job_id"]["pattern"] == "^job_[a-f0-9]{24}$"
    assert schema["properties"]["implementation_revision_sha256"]["pattern"] == "^[a-f0-9]{64}$"
    assert schema["properties"]["verdict"]["enum"] == ["accept", "rework"]
    findings = schema["properties"]["findings"]
    assert findings["type"] == "array"
    assert findings["maxItems"] == MAX_AUDIT_FINDINGS
    item = findings["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"code", "severity", "message"}
    assert set(item["properties"]) == {"code", "severity", "message", "evidence_ref"}
    assert item["properties"]["severity"]["enum"] == ["blocker", "major", "minor"]
    assert item["properties"]["message"]["maxLength"] == MAX_AUDIT_MESSAGE_CHARS
    # Backend authority stays a startup decision and evidence paths stay private.
    catalog = json.dumps(tools)
    assert "backend" not in catalog
    assert "evidence_path" not in catalog


def test_route_retry_flag_is_publicly_decodable_without_adding_a_tool(tmp_path) -> None:
    from flyto_ai.coding.mcp_server import CodingMCPServer

    workspace = tmp_path / "retry-schema"
    workspace.mkdir()
    decoded = request_from_mapping({
        "message": "resume the audited repair",
        "working_dir": str(workspace),
        "thread_id": "sdk-session-1",
        "resume": True,
        "retry_rework_route": True,
    })
    assert decoded.retry_rework_route is True

    tools = {tool["name"]: tool for tool in CodingMCPServer._tools()}
    assert set(tools) == {
        "flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit",
    }
    request_schema = tools["flyto_coding_submit"]["inputSchema"]["properties"]["request"]
    assert request_schema["additionalProperties"] is False
    assert request_schema["properties"]["retry_rework_route"] == {"type": "boolean"}
    with pytest.raises(ValueError, match="must be a boolean"):
        request_from_mapping({
            "message": "resume the audited repair",
            "working_dir": str(workspace),
            "thread_id": "sdk-session-1",
            "resume": True,
            "retry_rework_route": "false",
        })
    with pytest.raises(ValueError, match="resume must be a boolean"):
        request_from_mapping({
            "message": "resume the audited repair",
            "working_dir": str(workspace),
            "thread_id": "sdk-session-1",
            "resume": "false",
            "retry_rework_route": True,
        })


def test_audit_finding_is_bounded_typed_and_carries_no_raw_payload() -> None:
    finding = CodingAuditFinding.from_mapping(_valid_finding_payload())
    assert finding.severity is CodingAuditSeverity.BLOCKER
    assert finding.code == "missing_regression_test"
    assert finding.to_mapping() == _valid_finding_payload()
    minimal = CodingAuditFinding.from_mapping({
        "code": "c1", "severity": "minor", "message": "m",
    })
    assert minimal.evidence_ref == ""
    assert minimal.to_mapping()["evidence_ref"] == ""


@pytest.mark.parametrize(("payload", "match"), [
    ("not-an-object", "must be a JSON object"),
    ([{"code": "c1"}], "must be a JSON object"),
    ({"code": True, "severity": "minor", "message": "m"}, "code must be a string"),
    ({"code": "1bad", "severity": "minor", "message": "m"}, "stable safe identifier"),
    ({"code": "c1", "severity": True, "message": "m"}, "severity must be a string"),
    ({"code": "c1", "severity": "critical", "message": "m"}, "blocker, major, or minor"),
    ({"code": "c1", "severity": "minor", "message": {"nested": 1}}, "message must be a string"),
    ({"code": "c1", "severity": "minor", "message": ""}, "between 1 and 2000"),
    ({"code": "c1", "severity": "minor", "message": "   "}, "visible text"),
    ({"code": "c1", "severity": "minor", "message": "x" * 2001}, "between 1 and 2000"),
    ({"code": "c1", "severity": "minor", "message": "bad\nline"}, "control characters"),
    ({"code": "c1", "severity": "minor", "message": "bad\x00byte"}, "control characters"),
    (
        {"code": "c1", "severity": "minor", "message": "m", "raw_log": "TOKEN=secret"},
        "unsupported audit finding fields: raw_log",
    ),
    (
        {"code": "c1", "severity": "minor", "message": "m", "evidence_ref": "/etc/passwd"},
        "bounded safe reference",
    ),
    (
        {"code": "c1", "severity": "minor", "message": "m", "evidence_ref": "a/../b"},
        "cannot traverse paths",
    ),
    (
        {"code": "c1", "severity": "minor", "message": "m", "evidence_ref": "e" * 300},
        "cannot exceed 256",
    ),
])
def test_audit_finding_rejects_malformed_values(payload, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CodingAuditFinding.from_mapping(payload)


def test_audit_findings_digest_is_canonical_order_sensitive_and_bounded() -> None:
    first = CodingAuditFinding("a_code", CodingAuditSeverity.MAJOR, "first message")
    second = CodingAuditFinding("b_code", "minor", "second message")
    assert len(audit_findings_sha256(())) == 64
    assert audit_findings_sha256((first, second)) == audit_findings_sha256((first, second))
    assert audit_findings_sha256((first, second)) != audit_findings_sha256((second, first))
    assert audit_findings_sha256(()) != audit_findings_sha256((first,))
    with pytest.raises(ValueError, match="cannot exceed"):
        audit_findings_sha256(tuple(
            CodingAuditFinding("code_{}".format(index), "minor", "m")
            for index in range(MAX_AUDIT_FINDINGS + 1)
        ))
    with pytest.raises(ValueError, match="must be CodingAuditFinding"):
        audit_findings_sha256(({"code": "a_code"},))


def test_audit_submission_binds_verdict_to_findings() -> None:
    finding = CodingAuditFinding("unbounded_input", "blocker", "reject oversized payloads")
    assert CodingAuditVerdict.REWORK.requires_findings is True
    assert CodingAuditVerdict.ACCEPT.requires_findings is False
    assert validate_audit_submission(CodingAuditVerdict.REWORK, [finding]) == (
        CodingAuditVerdict.REWORK, (finding,),
    )
    assert validate_audit_submission("accept", ()) == (CodingAuditVerdict.ACCEPT, ())
    with pytest.raises(ValueError, match="rework verdict requires"):
        validate_audit_submission(CodingAuditVerdict.REWORK, ())
    with pytest.raises(ValueError, match="accept verdict cannot carry"):
        validate_audit_submission(CodingAuditVerdict.ACCEPT, (finding,))
    with pytest.raises(ValueError, match="duplicate code/evidence_ref"):
        validate_audit_submission(CodingAuditVerdict.REWORK, (finding, finding))
    with pytest.raises(ValueError):
        validate_audit_submission("approve", ())


def test_receipt_audit_fields_are_additive_and_landability_is_gated() -> None:
    legacy = CodingJobReceipt(
        job_id=_AUDIT_JOB_ID, state=CodingJobState.COMPLETED,
        submitted_at=1.0, updated_at=2.0,
    )
    assert SERVICE_CONTRACT_VERSION == "flyto.coding-service.v2"
    assert legacy.service_contract_version == SERVICE_CONTRACT_VERSION
    assert legacy.implementation_backend == ""
    assert legacy.implementation_session_id == ""
    assert legacy.implementation_revision_sha256 == ""
    assert legacy.audit_count == 0 and legacy.rework_count == 0
    assert legacy.audit_findings_sha256 == ""
    assert legacy.landable is False

    accepted = CodingJobReceipt(
        job_id=_AUDIT_JOB_ID, state=CodingJobState.CODEX_ACCEPTED,
        submitted_at=1.0, updated_at=2.0,
        implementation_backend="native",
        implementation_session_id="opaque-session-token",
        implementation_revision_sha256=_AUDIT_REVISION,
        audit_count=2, rework_count=1,
        audit_findings_sha256=audit_findings_sha256(()),
        landable=True,
    )
    assert accepted.landable is True
    assert accepted.state is CodingJobState.CODEX_ACCEPTED
    assert accepted.implementation_backend == "native"
    assert accepted.implementation_session_id == "opaque-session-token"
    # Acceptance and landability are one fact; neither direction may drift.
    with pytest.raises(ValueError, match="must be landable"):
        dataclasses.replace(accepted, landable=False)


@pytest.mark.parametrize(("changes", "match"), [
    ({"state": CodingJobState.COMPLETED}, "only a Codex-accepted receipt"),
    (
        {
            "state": CodingJobState.REWORK_QUEUED, "landable": False,
            "audit_count": 0, "rework_count": 0, "audit_findings_sha256": "",
        },
        "requires at least one recorded audit",
    ),
    (
        {
            "state": CodingJobState.AWAITING_CODEX_AUDIT, "landable": False,
            "implementation_revision_sha256": "",
            "audit_count": 0, "rework_count": 0, "audit_findings_sha256": "",
        },
        "requires implementation_revision_sha256",
    ),
    ({"implementation_revision_sha256": ""}, "requires implementation_revision_sha256"),
    ({"implementation_revision_sha256": "B3" * 32}, "64-character sha256"),
    ({"implementation_revision_sha256": "b3" * 31}, "64-character sha256"),
    ({"implementation_revision_sha256": True}, "must be a string"),
    ({"audit_count": 1, "rework_count": 2}, "rework_count cannot exceed audit_count"),
    ({"audit_count": True}, "audit_count must be an integer"),
    ({"audit_count": 1000, "rework_count": 0}, "audit_count must be between 0 and 100"),
    ({"audit_findings_sha256": ""}, "must be recorded together"),
    (
        {
            "state": CodingJobState.RUNNING, "landable": False,
            "audit_count": 0, "rework_count": 0,
        },
        "must be recorded together",
    ),
    ({"landable": False}, "must be landable"),
    ({"implementation_backend": "not a name"}, "safe identifier"),
    ({"implementation_backend": ""}, "requires implementation_backend"),
    ({"implementation_session_id": ""}, "requires implementation_session_id"),
    ({"implementation_session_id": "opaque\x00token"}, "bounded opaque token"),
    ({"implementation_session_id": "s" * 129}, "bounded opaque token"),
    ({"landable": "yes"}, "landable must be a boolean"),
])
def test_receipt_rejects_incoherent_audit_state(changes: dict, match: str) -> None:
    base = dict(
        job_id=_AUDIT_JOB_ID, state=CodingJobState.CODEX_ACCEPTED,
        submitted_at=1.0, updated_at=2.0,
        implementation_backend="native",
        implementation_session_id="opaque-session-token",
        implementation_revision_sha256=_AUDIT_REVISION,
        audit_count=1, rework_count=0,
        audit_findings_sha256=audit_findings_sha256(()),
        landable=True,
    )
    base.update(changes)
    with pytest.raises(ValueError, match=match):
        CodingJobReceipt(**base)


def test_audit_tool_forwards_a_typed_tenant_bound_call() -> None:
    recorder = RecordingAuditService()
    server = CodingMCPServer(recorder, "tenant-audit")
    response = server.handle(_audit_request(
        job_id=_AUDIT_JOB_ID,
        implementation_revision_sha256=_AUDIT_REVISION,
        verdict="rework",
        findings=[_valid_finding_payload()],
    ))
    result = response["result"]
    assert result["isError"] is False
    (tenant_id, job_id, revision, verdict, findings), = recorder.calls
    assert tenant_id == "tenant-audit"
    assert job_id == _AUDIT_JOB_ID
    assert revision == _AUDIT_REVISION
    assert verdict is CodingAuditVerdict.REWORK
    assert findings == (CodingAuditFinding.from_mapping(_valid_finding_payload()),)
    job = result["structuredContent"]["job"]
    assert job["state"] == CodingJobState.REWORK_QUEUED.value
    assert job["landable"] is False
    assert job["rework_count"] == 1
    assert job["audit_findings_sha256"] == audit_findings_sha256(findings)
    assert job["service_contract_version"] == SERVICE_CONTRACT_VERSION
    assert "evidence_path" not in json.dumps(job)

    accepted = server.handle(_audit_request(
        job_id=_AUDIT_JOB_ID,
        implementation_revision_sha256=_AUDIT_REVISION,
        verdict="accept",
        findings=[],
    ))["result"]["structuredContent"]["job"]
    assert accepted["state"] == CodingJobState.CODEX_ACCEPTED.value
    assert accepted["landable"] is True
    assert recorder.calls[1][3] is CodingAuditVerdict.ACCEPT
    assert recorder.calls[1][4] == ()


@pytest.mark.parametrize("arguments", [
    {"implementation_revision_sha256": _AUDIT_REVISION, "verdict": "accept", "findings": []},
    {"job_id": 123, "implementation_revision_sha256": _AUDIT_REVISION,
     "verdict": "accept", "findings": []},
    {"job_id": "job_" + "z" * 24, "implementation_revision_sha256": _AUDIT_REVISION,
     "verdict": "accept", "findings": []},
    {"job_id": _AUDIT_JOB_ID.upper(), "implementation_revision_sha256": _AUDIT_REVISION,
     "verdict": "accept", "findings": []},
    {"job_id": _AUDIT_JOB_ID, "implementation_revision_sha256": _AUDIT_REVISION,
     "verdict": "accept", "findings": [], "implementation_backend": "claude"},
    {"job_id": _AUDIT_JOB_ID, "implementation_revision_sha256": _AUDIT_REVISION,
     "verdict": "accept", "findings": [], "landable": True},
    {"job_id": _AUDIT_JOB_ID, "verdict": "accept", "findings": []},
    {"job_id": _AUDIT_JOB_ID, "implementation_revision_sha256": "B3" * 32,
     "verdict": "accept", "findings": []},
    {"job_id": _AUDIT_JOB_ID, "implementation_revision_sha256": True,
     "verdict": "accept", "findings": []},
    {"job_id": _AUDIT_JOB_ID, "implementation_revision_sha256": _AUDIT_REVISION,
     "verdict": "approve", "findings": []},
    {"job_id": _AUDIT_JOB_ID, "implementation_revision_sha256": _AUDIT_REVISION,
     "verdict": True, "findings": []},
    {"job_id": _AUDIT_JOB_ID, "implementation_revision_sha256": _AUDIT_REVISION,
     "verdict": "accept"},
    {"job_id": _AUDIT_JOB_ID, "implementation_revision_sha256": _AUDIT_REVISION,
     "verdict": "rework", "findings": {"code": "c1"}},
    {"job_id": _AUDIT_JOB_ID, "implementation_revision_sha256": _AUDIT_REVISION,
     "verdict": "rework", "findings": ["not-an-object"]},
    {"job_id": _AUDIT_JOB_ID, "implementation_revision_sha256": _AUDIT_REVISION,
     "verdict": "rework",
     "findings": [{"code": "c1", "severity": "minor", "message": "m", "log": "secret"}]},
    {"job_id": _AUDIT_JOB_ID, "implementation_revision_sha256": _AUDIT_REVISION,
     "verdict": "rework",
     "findings": [{"code": "c1", "severity": "minor", "message": "m"}] * (MAX_AUDIT_FINDINGS + 1)},
])
def test_audit_tool_rejects_malformed_payloads_before_reaching_the_service(
    arguments: dict,
) -> None:
    recorder = RecordingAuditService()
    server = CodingMCPServer(recorder, "tenant-audit")
    response = server.handle(_audit_request(**arguments))
    assert response["result"]["isError"] is True
    assert response["result"]["structuredContent"] == {"ok": False, "error": "invalid_request"}
    assert recorder.calls == []


def _blocker(code: str = "missing_regression_test") -> CodingAuditFinding:
    return CodingAuditFinding(code, CodingAuditSeverity.BLOCKER, "cover the audited path")


def _audited_service(tmp_path: Path, workspace: Path, **kwargs) -> CodingService:
    return _service(tmp_path, workspace, require_codex_audit=True, **kwargs)


def _awaiting(service: CodingService, tenant: str, key: str, workspace: Path):
    queued = service.submit(tenant, key, _request(workspace))
    receipt = _wait(service, tenant, queued.job_id)
    assert receipt.state is CodingJobState.AWAITING_CODEX_AUDIT
    return receipt


class FakeClaudeBackend:
    """Stands in for the Claude SDK: real edits, real response shape, no network."""

    def __init__(
        self,
        workspace: Path,
        *,
        session: str = "sdk-session-1",
        ok: bool = True,
        writes: bool = True,
        provider_failure_code: str = "",
    ) -> None:
        self.workspace = workspace
        self.session = session
        self.ok = ok
        self.writes = writes
        self.provider_failure_code = provider_failure_code
        self.requests: list = []

    async def run(self, request):
        self.requests.append(request)
        round_index = len(self.requests)
        if self.writes:
            (self.workspace / "result.txt").write_text("verified\n")
            (self.workspace / "notes.txt").write_text("round {}\n".format(round_index))
        from flyto_ai.agents.models import CodeTaskResponse
        return CodeTaskResponse(
            ok=self.ok,
            message="applied changes in {}".format(self.workspace),
            session_id="local-evidence-{}".format(round_index),
            attempts=1,
            claude_session_id=self.session,
            claude_num_turns=3,
            claude_usage={"input_tokens": 11, "output_tokens": 7, "cost_usd": 1.5, "ok": True},
            provider_failure_code=self.provider_failure_code,
        )


def _claude_workspace(tmp_path: Path, *, check: str = "pass") -> Path:
    workspace = tmp_path / "claude-workspace"
    workspace.mkdir(parents=True)
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir(exist_ok=True)
    argv = {
        "pass": [sys.executable, "-c", "from pathlib import Path; assert Path('result.txt').read_text() == 'verified\\n'"],
        "trivial": [sys.executable, "-c", "pass"],
        "fail": [sys.executable, "-c", "raise SystemExit(3)"],
    }[check]
    config.write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: real_claude_check\n"
        "    argv: {}\n".format(json.dumps(argv))
    )
    return workspace


def _claude_adapter(tmp_path: Path, workspace: Path, backend: FakeClaudeBackend):
    from flyto_ai.agents.claude_code import ClaudeCodingAgent
    from flyto_ai.coding.store import ThreadStore

    store = ThreadStore(str(tmp_path / "claude-threads"))
    return ClaudeCodingAgent(store, agent=backend), store


def test_claude_adapter_binds_the_sdk_session_and_uses_real_repository_checks(
    tmp_path: Path,
) -> None:
    workspace = _claude_workspace(tmp_path)
    backend = FakeClaudeBackend(workspace)
    adapter, store = _claude_adapter(tmp_path, workspace, backend)

    first = asyncio.run(adapter.run(_request(workspace)))
    assert first.ok is True
    assert first.thread_id == "sdk-session-1"
    assert backend.requests[0].service_mode is True
    assert backend.requests[0].sdk_session_id is None
    # Attribution comes from independent snapshots, not from model prose.
    assert first.files_changed == ["notes.txt", "result.txt"]
    assert [check.name for check in first.checks] == ["real_claude_check"]
    assert first.checks[0].passed is True
    # Only bounded integer counters survive; floats and booleans are dropped.
    assert first.usage == {"input_tokens": 11, "output_tokens": 7}
    assert first.evidence_path == ""
    assert str(workspace) not in first.message
    assert os.path.realpath(str(workspace)) not in first.message
    assert len(store.digest("sdk-session-1")) == 64

    second = asyncio.run(adapter.run(CodingTaskRequest(
        message="apply the audit feedback",
        working_dir=str(workspace),
        thread_id="sdk-session-1",
        resume=True,
    )))
    assert second.ok is True
    assert second.thread_id == "sdk-session-1"
    assert backend.requests[1].sdk_session_id == "sdk-session-1"
    assert second.files_changed == ["notes.txt"]
    # The service adapter keeps evidence in the ThreadStore only.
    assert not list((tmp_path / "claude-threads").glob("**/evidence-*.json"))


def test_claude_adapter_fails_closed_on_verification_session_and_capability(
    tmp_path: Path,
) -> None:
    from flyto_ai.agents.claude_code import ClaudeCodingAgent
    from flyto_ai.coding.store import ThreadStore

    unchecked = tmp_path / "unchecked"
    unchecked.mkdir()
    store = ThreadStore(str(tmp_path / "threads-a"))
    result = asyncio.run(
        ClaudeCodingAgent(store, agent=FakeClaudeBackend(unchecked)).run(_request(unchecked))
    )
    assert result.ok is False and result.failure_code == "verification_required"

    failing = _claude_workspace(tmp_path / "b", check="fail")
    adapter, _ = _claude_adapter(tmp_path / "b", failing, FakeClaudeBackend(failing))
    result = asyncio.run(adapter.run(_request(failing)))
    assert result.ok is False and result.failure_code == "verification_failed"

    idle = _claude_workspace(tmp_path / "c", check="trivial")
    adapter, _ = _claude_adapter(
        tmp_path / "c", idle, FakeClaudeBackend(idle, writes=False),
    )
    result = asyncio.run(adapter.run(_request(idle)))
    assert result.ok is False and result.failure_code == "no_changes"

    provider_failure = _claude_workspace(tmp_path / "d")
    adapter, _ = _claude_adapter(
        tmp_path / "d", provider_failure, FakeClaudeBackend(provider_failure, ok=False),
    )
    result = asyncio.run(adapter.run(_request(provider_failure)))
    assert result.ok is False and result.failure_code == "provider_failed"

    drifting = _claude_workspace(tmp_path / "e")
    adapter, _ = _claude_adapter(
        tmp_path / "e", drifting, FakeClaudeBackend(drifting, session="sdk-other"),
    )
    result = asyncio.run(adapter.run(CodingTaskRequest(
        message="rework", working_dir=str(drifting),
        thread_id="sdk-session-1", resume=True,
    )))
    assert result.ok is False and result.failure_code == "session_binding_failed"

    capability = _claude_workspace(tmp_path / "f")
    (capability / ".flyto" / "coding.yaml").write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: real_claude_check\n"
        "    argv: {}\n"
        "capabilities:\n"
        "  - name: indexer\n"
        "    argv: [flyto-indexer-mcp]\n"
        "    required: true\n"
        "    required_tools: [context]\n".format(json.dumps([sys.executable, "-c", "pass"]))
    )
    adapter, _ = _claude_adapter(tmp_path / "f", capability, FakeClaudeBackend(capability))
    result = asyncio.run(adapter.run(_request(capability)))
    assert result.ok is False
    assert result.failure_code == "required_capability_unavailable"


class ExplodingClaudeBackend:
    """Fails before any SDK session identity exists."""

    def __init__(self) -> None:
        self.requests: list = []

    async def run(self, request):
        self.requests.append(request)
        raise RuntimeError("sdk transport failure")


def _claude_service(
    tmp_path: Path,
    workspace: Path,
    backend,
    *,
    require_codex_audit: bool = False,
    sandbox_mode: SandboxMode = SandboxMode.WORKSPACE_WRITE,
    approval_policy: ApprovalPolicy = ApprovalPolicy.NEVER,
) -> CodingService:
    from flyto_ai.agents.claude_code import ClaudeCodingAgent

    return CodingService(
        lambda store: ClaudeCodingAgent(store, agent=backend),
        state_root=str(tmp_path / "claude-service-state"),
        workspace_roots=(str(workspace),),
        max_workers=2,
        max_queued=8,
        require_codex_audit=require_codex_audit,
        sandbox_mode=sandbox_mode,
        approval_policy=approval_policy,
    )


@pytest.mark.parametrize(("sandbox_mode", "approval_policy", "code"), [
    (SandboxMode.READ_ONLY, ApprovalPolicy.NEVER, "workspace_read_only"),
    (SandboxMode.READ_ONLY, ApprovalPolicy.ALWAYS, "workspace_read_only"),
    (SandboxMode.WORKSPACE_WRITE, ApprovalPolicy.ON_REQUEST, "approval_required"),
    (SandboxMode.WORKSPACE_WRITE, ApprovalPolicy.ALWAYS, "approval_required"),
])
def test_claude_refuses_impossible_write_tasks_before_calling_the_backend(
    tmp_path: Path, sandbox_mode: SandboxMode, approval_policy: ApprovalPolicy, code: str,
) -> None:
    workspace = _claude_workspace(tmp_path)
    backend = FakeClaudeBackend(workspace)
    service = _claude_service(
        tmp_path, workspace, backend,
        sandbox_mode=sandbox_mode, approval_policy=approval_policy,
    )
    try:
        queued = service.submit("tenant-claude", "claude-authority", _request(workspace))
        failed = _wait(service, "tenant-claude", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == code
        assert failed.thread_id and len(failed.evidence_sha256) == 64
        # The model was never asked to do something its authority forbids.
        assert backend.requests == []
        assert not (workspace / "result.txt").exists()
    finally:
        service.close()


def test_claude_read_only_authority_runs_with_read_only_tools_and_no_edits(
    tmp_path: Path,
) -> None:
    from flyto_ai.agents.claude_code import SERVICE_READONLY_TOOLS, ClaudeCodeAgent

    workspace = _claude_workspace(tmp_path, check="trivial")
    backend = FakeClaudeBackend(workspace, writes=False)
    service = _claude_service(
        tmp_path, workspace, backend, sandbox_mode=SandboxMode.READ_ONLY,
    )
    try:
        queued = service.submit("tenant-claude", "claude-readonly", CodingTaskRequest(
            message="review the workspace", working_dir=str(workspace),
            require_changes=False,
        ))
        finished = _wait(service, "tenant-claude", queued.job_id)
        assert finished.state is CodingJobState.COMPLETED
        assert len(backend.requests) == 1
        code_request = backend.requests[0]
        assert code_request.service_edit_authority is False
        assert code_request.require_changes is False
        options = ClaudeCodeAgent()._option_kwargs(
            code_request, session_id=None, system_prompt="s", max_turns=1, max_budget=1.0,
        )
        assert set(options["allowed_tools"]) == set(SERVICE_READONLY_TOOLS)
        assert options["permission_mode"] == "default"
    finally:
        service.close()


def test_claude_read_only_authority_never_accepts_an_observed_change(
    tmp_path: Path,
) -> None:
    workspace = _claude_workspace(tmp_path, check="trivial")
    # The backend writes anyway, standing in for an SDK catalog regression.
    backend = FakeClaudeBackend(workspace)
    service = _claude_service(
        tmp_path, workspace, backend, sandbox_mode=SandboxMode.READ_ONLY,
    )
    try:
        queued = service.submit("tenant-claude", "claude-drift", CodingTaskRequest(
            message="review only", working_dir=str(workspace), require_changes=False,
        ))
        failed = _wait(service, "tenant-claude", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == "unexpected_workspace_change"
        assert failed.landable is False
    finally:
        service.close()


def test_claude_session_id_beyond_the_thread_boundary_fails_durably(
    tmp_path: Path,
) -> None:
    from flyto_ai.agents.claude_code import HOST_THREAD_PREFIX

    workspace = _claude_workspace(tmp_path)
    backend = FakeClaudeBackend(workspace, session="s" * 100)
    service = _claude_service(tmp_path, workspace, backend)
    try:
        queued = service.submit("tenant-claude", "claude-longid", _request(workspace))
        failed = _wait(service, "tenant-claude", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == "session_binding_failed"
        assert failed.thread_id.startswith(HOST_THREAD_PREFIX)
        assert len(failed.thread_id) <= 64
        assert len(failed.evidence_sha256) == 64
        assert failed.implementation_session_id == ""
    finally:
        service.close()


def test_claude_accepts_a_normal_uuid_session(tmp_path: Path) -> None:
    workspace = _claude_workspace(tmp_path)
    session = "8f14e45f-ceea-467a-9e0f-6b1a2c3d4e5f"
    backend = FakeClaudeBackend(workspace, session=session)
    service = _claude_service(tmp_path, workspace, backend, require_codex_audit=True)
    try:
        queued = service.submit("tenant-claude", "claude-uuid", _request(workspace))
        awaiting = _wait(service, "tenant-claude", queued.job_id)
        assert awaiting.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert awaiting.implementation_session_id == session
        assert awaiting.thread_id == session
    finally:
        service.close()


def test_pre_session_failures_carry_a_durable_non_sdk_thread_id(tmp_path: Path) -> None:
    from flyto_ai.agents.claude_code import HOST_THREAD_PREFIX, ClaudeCodingAgent
    from flyto_ai.coding.store import ThreadStore

    generated = ClaudeCodingAgent.host_thread_id("")
    assert generated.startswith(HOST_THREAD_PREFIX)
    assert ClaudeCodingAgent.host_thread_id("sdk-session-1") == "sdk-session-1"
    for unusable in ("", None, "../escape", "a" * 100, 5, True):
        assert ClaudeCodingAgent.host_thread_id(unusable).startswith(HOST_THREAD_PREFIX)
    # The provisional id is a real ThreadStore identifier, so the service can
    # always compute an evidence digest for a failed round.
    assert len(ThreadStore(str(tmp_path / "threads")).digest(generated)) == 64


def test_claude_missing_checks_fail_durably_through_the_real_service(tmp_path: Path) -> None:
    """A repository with no verification contract is refused before it costs anything.

    This used to be discovered inside the implementer, so "you never said how
    to verify this" arrived as a failed job with a burnt session, a held
    worktree claim and a receipt to poll. Preflight answers the same question
    first, so the refusal is a `verification_required` raised out of `submit`
    itself and there is nothing left behind to clean up.
    """

    workspace = tmp_path / "unchecked-workspace"
    workspace.mkdir()
    backend = FakeClaudeBackend(workspace)
    service = _claude_service(tmp_path, workspace, backend)
    tenant_ref = service._tenant_ref("tenant-claude")
    try:
        with pytest.raises(VerificationRequired) as refused:
            service.submit("tenant-claude", "claude-checks", _request(workspace))

        assert refused.value.code == "verification_required"
        assert error_details(refused.value) == {
            "failure_phase": "preflight",
            "retryable": False,
            "required_actions": ["add_repository_verification_contract"],
        }

        # Nothing was created: no job record to poll, no worktree claim to
        # release, and the implementer was never contacted.
        jobs = service.state_root / "tenants" / tenant_ref / "jobs"
        assert not jobs.exists() or list(jobs.glob("*.json")) == []
        assert not service._workspace_claim_path(str(workspace)).exists()
        assert backend.requests == [] if hasattr(backend, "requests") else True

        # And the refusal is not sticky: declaring a contract makes the very
        # same submission viable, so preflight gates feasibility rather than
        # blacklisting a workspace.
        _declare_verification(workspace)
        queued = service.submit("tenant-claude", "claude-checks", _request(workspace))
        assert queued.job_id
    finally:
        service.close()


def test_claude_provider_exception_fails_durably_before_any_sdk_session(
    tmp_path: Path,
) -> None:
    from flyto_ai.agents.claude_code import HOST_THREAD_PREFIX

    workspace = _claude_workspace(tmp_path)
    backend = ExplodingClaudeBackend()
    service = _claude_service(tmp_path, workspace, backend)
    try:
        queued = service.submit("tenant-claude", "claude-boom", _request(workspace))
        failed = _wait(service, "tenant-claude", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == "provider_failed"
        assert failed.thread_id.startswith(HOST_THREAD_PREFIX)
        assert len(failed.evidence_sha256) == 64
        assert failed.implementation_session_id == ""
        assert "sdk transport failure" not in json.dumps(receipt_to_mapping(failed))
    finally:
        service.close()


def test_claude_adapter_completes_a_codex_rework_cycle_through_the_real_service(
    tmp_path: Path,
) -> None:
    workspace = _claude_workspace(tmp_path)
    backend = FakeClaudeBackend(workspace)
    service = _claude_service(tmp_path, workspace, backend, require_codex_audit=True)
    try:
        queued = service.submit("tenant-claude", "claude-cycle", _request(workspace))
        first = _wait(service, "tenant-claude", queued.job_id)
        assert first.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert first.thread_id == "sdk-session-1"
        assert first.implementation_session_id == "sdk-session-1"
        assert first.implementation_backend == "native"
        assert backend.requests[0].sdk_session_id is None

        service.audit(
            "tenant-claude", queued.job_id, first.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, (_blocker(),),
        )
        second = _wait(service, "tenant-claude", queued.job_id)
        assert second.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert second.job_id == first.job_id
        assert second.implementation_session_id == "sdk-session-1"
        assert second.implementation_revision_sha256 != first.implementation_revision_sha256
        assert len(backend.requests) == 2
        assert backend.requests[1].sdk_session_id == "sdk-session-1"
        assert backend.requests[1].service_mode is True

        accepted = service.audit(
            "tenant-claude", queued.job_id, second.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
        assert accepted.landable is True
        assert accepted.audit_count == 2 and accepted.rework_count == 1
    finally:
        service.close()


# ── auditable-needs-rework ────────────────────────────────────────────
#
# A round that really ran, really edited the workspace, and really kept its
# session is not the same failure as a round that never produced anything an
# auditor could read. These tests pin that difference in both directions.


def _install_fake_sdk(
    monkeypatch,
    workspace: Path,
    *,
    session: str,
    error: str,
    emit_init: bool = True,
):
    """Install a `claude_agent_sdk` double that edits, then raises mid-stream.

    The edits land in the real workspace before the raise, so the host's own
    snapshot attribution is what the assertions observe — never model prose.
    """

    module = types.ModuleType("claude_agent_sdk")

    class SystemMessage:
        def __init__(self, subtype: str, session_id: str) -> None:
            self.subtype = subtype
            self.session_id = session_id

    class AssistantMessage:
        def __init__(self, content) -> None:
            self.content = content

    class ResultMessage:  # pragma: no cover - never reached by these rounds
        pass

    class TextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class ClaudeAgentOptions:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class HookMatcher:
        def __init__(self, hooks) -> None:
            self.hooks = hooks

    async def query(*, prompt, options):
        if emit_init:
            yield SystemMessage("init", session)
        (workspace / "result.txt").write_text("verified\n")
        (workspace / "notes.txt").write_text("partial\n")
        yield AssistantMessage([TextBlock("editing the workspace")])
        raise Exception(error)

    for name, value in (
        ("query", query),
        ("ClaudeAgentOptions", ClaudeAgentOptions),
        ("HookMatcher", HookMatcher),
        ("AssistantMessage", AssistantMessage),
        ("ResultMessage", ResultMessage),
        ("SystemMessage", SystemMessage),
        ("TextBlock", TextBlock),
    ):
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)

    async def _no_context(message, working_dir):
        return ""

    monkeypatch.setattr(
        "flyto_ai.agents.indexer_context.gather_context", _no_context,
    )


def _pinned_claude_agent():
    """A Claude backend whose loop bounds come from the defaults, not the env."""

    from flyto_ai.agents.claude_code import ClaudeCodeAgent
    from flyto_ai.config import ClaudeCodeConfig

    agent = ClaudeCodeAgent()
    agent._cc = ClaudeCodeConfig()
    return agent


def _durable_text(*roots: Path) -> str:
    """Every byte this service persisted, as one searchable string."""

    parts = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file():
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


_TURN_LIMIT_ERROR = "Claude Code process failed with error_max_turns after 100 turns"


def test_a_turn_limit_after_session_capture_keeps_the_round_attributable(
    tmp_path: Path, monkeypatch,
) -> None:
    """Invariant 1: the init session survives a recognized bounded stop."""

    from flyto_ai.agents.claude_code import ClaudeCodingAgent
    from flyto_ai.coding.store import ThreadStore
    from flyto_ai.config import ClaudeCodeConfig

    workspace = _claude_workspace(tmp_path)
    _install_fake_sdk(
        monkeypatch, workspace, session="sdk-session-9", error=_TURN_LIMIT_ERROR,
    )
    threads = tmp_path / "claude-threads"
    adapter = ClaudeCodingAgent(ThreadStore(str(threads)), agent=_pinned_claude_agent())

    request = _request(workspace)
    result = asyncio.run(adapter.run(request))

    assert result.ok is False
    assert result.failure_code == "turn_limit_exceeded"
    # The exact SDK session, not a provisional host thread.
    assert result.thread_id == "sdk-session-9"
    # Host snapshots, not model prose, and never the zeroed shape the broken
    # route reported for this exact condition.
    assert result.files_changed == ["notes.txt", "result.txt"]
    assert result.attempts == 1
    # Reaching a turn limit proves the configured budget was consumed, so the
    # host's own ceiling is reported rather than the 0 that arrives when no
    # ResultMessage does.
    budget = min(request.max_rounds, ClaudeCodeConfig().max_turns)
    assert result.rounds_used == budget

    # A resumed round reports the same host-known budget.
    resumed = asyncio.run(adapter.run(CodingTaskRequest(
        message="continue the work", working_dir=str(workspace),
        thread_id="sdk-session-9", resume=True,
    )))
    assert resumed.thread_id == "sdk-session-9"
    assert resumed.failure_code == "turn_limit_exceeded"
    assert resumed.rounds_used == budget
    # Invariant 1/7: no provider exception text reaches anything durable.
    public = json.dumps(dataclasses.asdict(result), default=str)
    for leak in ("error_max_turns", _TURN_LIMIT_ERROR, "Exception"):
        assert leak not in public
        assert leak not in _durable_text(threads)


def test_a_turn_limit_before_any_session_stays_terminal(
    tmp_path: Path, monkeypatch,
) -> None:
    """Invariant 5: the same stop without a session is never auditable."""

    workspace = _claude_workspace(tmp_path)
    # The workspace is edited before the raise, so the only thing missing is a
    # session identity. That alone must keep the round terminal.
    _install_fake_sdk(
        monkeypatch, workspace, session="sdk-session-9",
        error=_TURN_LIMIT_ERROR, emit_init=False,
    )
    service = _claude_service(
        tmp_path, workspace, _pinned_claude_agent(), require_codex_audit=True,
    )
    try:
        queued = service.submit("tenant-claude", "claude-nosession", _request(workspace))
        failed = _wait(service, "tenant-claude", queued.job_id)

        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == "turn_limit_exceeded"
        assert failed.landable is False
        # Never invent a session, and never claim a revision without one.
        assert failed.implementation_session_id == ""
        assert failed.implementation_revision_sha256 == ""
        assert failed.implementation_blockers == ()
        assert failed.thread_id.startswith("host-")
        state_root = tmp_path / "claude-service-state"
        for leak in ("error_max_turns", _TURN_LIMIT_ERROR):
            assert leak not in json.dumps(receipt_to_mapping(failed))
            assert leak not in _durable_text(state_root)
    finally:
        service.close()


def test_a_bounded_stop_with_real_changes_awaits_audit_and_reworks_in_session(
    tmp_path: Path,
) -> None:
    """Invariants 2-4, 6: blocked, accept-refused, same-session rework, cleared."""

    workspace = _claude_workspace(tmp_path)
    backend = FakeClaudeBackend(
        workspace, ok=False, provider_failure_code="turn_limit_exceeded",
    )
    service = _claude_service(tmp_path, workspace, backend, require_codex_audit=True)
    try:
        queued = service.submit("tenant-claude", "claude-blocked", _request(workspace))
        blocked = _wait(service, "tenant-claude", queued.job_id)

        # Invariant 3: auditable, bound to the exact session and revision.
        assert blocked.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert blocked.landable is False
        assert blocked.implementation_session_id == "sdk-session-1"
        assert len(blocked.implementation_revision_sha256) == 64
        assert blocked.implementer_started is True
        # Invariant 6: the receipt reports what the host actually observed.
        assert blocked.result is not None
        assert blocked.result.files_changed == ["notes.txt", "result.txt"]
        assert blocked.result.attempts == 1
        assert blocked.failure_code == "turn_limit_exceeded"
        assert "turn_limit_exceeded" in blocked.implementation_blockers

        # Invariant 3: accept is refused while the blocker stands, and costs
        # the job nothing.
        with pytest.raises(AuditBlockersUnresolved):
            service.audit(
                "tenant-claude", queued.job_id,
                blocked.implementation_revision_sha256,
                CodingAuditVerdict.ACCEPT, (),
            )
        unchanged = service.get("tenant-claude", queued.job_id)
        assert unchanged.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert unchanged.audit_count == 0
        assert unchanged.landable is False

        # Invariant 3/4: rework is allowed and resumes the same session.
        backend.ok = True
        backend.provider_failure_code = ""
        service.audit(
            "tenant-claude", queued.job_id, blocked.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, (_blocker(),),
        )
        reworked = _wait(service, "tenant-claude", queued.job_id)
        assert reworked.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert reworked.job_id == blocked.job_id
        assert reworked.implementation_session_id == "sdk-session-1"
        assert backend.requests[1].sdk_session_id == "sdk-session-1"
        # Invariant 4: the blocker is cleared, and the audited revision moves
        # even though this round only re-touched part of the change set — the
        # revision stays bound to the cumulative implementation files.
        assert reworked.implementation_blockers == ()
        assert reworked.result.files_changed == ["notes.txt"]
        assert reworked.implementation_revision_sha256 != (
            blocked.implementation_revision_sha256
        )

        accepted = service.audit(
            "tenant-claude", queued.job_id, reworked.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
        assert accepted.landable is True
        assert accepted.implementation_blockers == ()
    finally:
        service.close()


def test_a_failed_required_check_with_real_changes_is_auditable_not_terminal(
    tmp_path: Path,
) -> None:
    """Invariants 2-3: host checks failing is rework, not a dead end."""

    workspace = _claude_workspace(tmp_path, check="fail")
    backend = FakeClaudeBackend(workspace)
    service = _claude_service(tmp_path, workspace, backend, require_codex_audit=True)
    try:
        queued = service.submit("tenant-claude", "claude-checkfail", _request(workspace))
        blocked = _wait(service, "tenant-claude", queued.job_id)

        assert blocked.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert blocked.landable is False
        assert blocked.implementation_session_id == "sdk-session-1"
        assert blocked.failure_code == "verification_failed"
        # The failing check names itself, so the auditor knows what to order.
        assert set(blocked.implementation_blockers) == {
            "verification_failed", "check.real_claude_check",
        }
        assert blocked.result.checks[0].passed is False

        with pytest.raises(AuditBlockersUnresolved):
            service.audit(
                "tenant-claude", queued.job_id,
                blocked.implementation_revision_sha256,
                CodingAuditVerdict.ACCEPT, (),
            )
        # Rework remains available on the exact session.
        reworked = service.audit(
            "tenant-claude", queued.job_id, blocked.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, (_blocker(),),
        )
        assert reworked.state in {
            CodingJobState.REWORK_QUEUED, CodingJobState.REWORK_RUNNING,
        }
        assert reworked.implementation_session_id == "sdk-session-1"
        _wait(service, "tenant-claude", queued.job_id)
        assert backend.requests[1].sdk_session_id == "sdk-session-1"
    finally:
        service.close()


def test_an_unknown_provider_failure_stays_terminal_despite_real_changes(
    tmp_path: Path,
) -> None:
    """Invariant 5 / closed vocabulary: only recognized failures are resumable.

    This backend is byte-for-byte the blocked one except for its failure
    classification: same valid session, same real writes, same passing checks.
    An unrecognized code is something the host cannot reason about, so it never
    holds the job open.
    """

    workspace = _claude_workspace(tmp_path)
    backend = FakeClaudeBackend(workspace, ok=False)
    service = _claude_service(tmp_path, workspace, backend, require_codex_audit=True)
    try:
        queued = service.submit("tenant-claude", "claude-unknown", _request(workspace))
        failed = _wait(service, "tenant-claude", queued.job_id)

        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == "provider_failed"
        assert failed.landable is False
        assert failed.implementation_blockers == ()
        # The round really ran and really wrote; that alone is never enough.
        assert failed.implementer_started is True
        assert failed.result.files_changed == ["notes.txt", "result.txt"]
        assert (workspace / "notes.txt").exists()
    finally:
        service.close()


@pytest.mark.parametrize(("check", "writes", "sandbox", "requires", "code"), [
    # No attributable change, so no revision an auditor could bind.
    ("trivial", False, SandboxMode.WORKSPACE_WRITE, True, "no_changes"),
    # A failure about the workspace boundary, never about quality.
    ("pass", True, SandboxMode.READ_ONLY, False, "unexpected_workspace_change"),
])
def test_failures_without_an_auditable_change_stay_terminal(
    tmp_path: Path,
    check: str,
    writes: bool,
    sandbox: SandboxMode,
    requires: bool,
    code: str,
) -> None:
    """Invariant 5: only a real attributable change can hold a job open."""

    workspace = _claude_workspace(tmp_path, check=check)
    backend = FakeClaudeBackend(workspace, writes=writes)
    service = _claude_service(
        tmp_path, workspace, backend, require_codex_audit=True, sandbox_mode=sandbox,
    )
    try:
        queued = service.submit("tenant-claude", "claude-" + code, CodingTaskRequest(
            message="do the work", working_dir=str(workspace), require_changes=requires,
        ))
        failed = _wait(service, "tenant-claude", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == code
        assert failed.landable is False
        assert failed.implementation_blockers == ()
    finally:
        service.close()


def test_public_message_redacts_every_canonical_workspace_spelling(tmp_path: Path) -> None:
    from flyto_ai.agents.claude_code import ClaudeCodingAgent

    real = tmp_path / "real-ws"
    real.mkdir()
    link = tmp_path / "link-ws"
    link.symlink_to(real)

    class _Response:
        message = "edited {}/app.py then re-read {}/app.py".format(link, real)

    text = ClaudeCodingAgent._public_message(_Response(), str(link))
    for spelling in (str(link), str(real), os.path.realpath(str(link))):
        assert spelling not in text
    assert text.count("<workspace>") == 2


class _FakeHTTPConnection:
    """Socket-free transport so handler routing is testable without binding."""

    def __init__(self, raw: bytes) -> None:
        self._reader = io.BytesIO(raw)
        self.sent = bytearray()

    def makefile(self, mode: str = "rb", *args, **kwargs):
        return self._reader

    def sendall(self, data) -> None:
        self.sent.extend(bytes(data))

    def close(self) -> None:
        return None


class _FakeHTTPServer:
    def __init__(self, service, tenant_id: str, auth_token: str) -> None:
        self.coding_service = service
        self.tenant_id = tenant_id
        self.auth_token_sha256 = hashlib.sha256(auth_token.encode()).digest()


def _http(
    service,
    method: str,
    path: str,
    *,
    tenant: str = "tenant-http",
    token: str = TEST_BEARER_TOKEN,
    body=None,
    content_type: str = "application/json",
    extra_headers: str = "",
):
    payload = b"" if body is None else json.dumps(body).encode()
    head = "{} {} HTTP/1.1\r\nHost: 127.0.0.1\r\n".format(method, path)
    if body is not None:
        head += "Content-Type: {}\r\nContent-Length: {}\r\n".format(content_type, len(payload))
    if token:
        head += "Authorization: Bearer {}\r\n".format(token)
    head += extra_headers + "Connection: close\r\n\r\n"
    connection = _FakeHTTPConnection(head.encode() + payload)
    CodingHTTPHandler(
        connection, ("127.0.0.1", 0), _FakeHTTPServer(service, tenant, TEST_BEARER_TOKEN),
    )
    raw = bytes(connection.sent)
    header, _, tail = raw.partition(b"\r\n\r\n")
    status = int(header.split(b" ")[1])
    return status, (json.loads(tail) if tail else {}), header.decode("latin-1")


def test_http_audit_endpoint_drives_the_real_state_machine(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    provider = ReworkingProvider()
    service = _audited_service(tmp_path, workspace, provider=provider)
    try:
        awaiting = _awaiting(service, "tenant-http", "http-audit", workspace)
        route = "/v1/coding/jobs/{}/audit".format(awaiting.job_id)

        status, payload, _ = _http(service, "POST", route, token="", body={
            "implementation_revision_sha256": awaiting.implementation_revision_sha256,
            "verdict": "accept", "findings": [],
        })
        assert status == 401 and payload["error"] == "unauthorized"

        status, payload, _ = _http(service, "POST", route, tenant="tenant-other", body={
            "implementation_revision_sha256": awaiting.implementation_revision_sha256,
            "verdict": "accept", "findings": [],
        })
        assert status == 404 and payload["error"] == "job_not_found"

        status, payload, _ = _http(service, "POST", route, body={
            "implementation_revision_sha256": _AUDIT_REVISION,
            "verdict": "accept", "findings": [],
        })
        assert status == 409 and payload["error"] == "revision_mismatch"

        assert service.get("tenant-http", awaiting.job_id).audit_count == 0

        status, payload, headers = _http(service, "POST", route, body={
            "implementation_revision_sha256": awaiting.implementation_revision_sha256,
            "verdict": "rework", "findings": [_valid_finding_payload()],
        })
        assert status == 200
        assert payload["job"]["state"] == CodingJobState.REWORK_QUEUED.value
        assert payload["job"]["landable"] is False
        assert "no-store" in headers
        assert "evidence_path" not in json.dumps(payload)

        repaired = _wait(service, "tenant-http", awaiting.job_id)
        status, payload, _ = _http(service, "POST", route, body={
            "implementation_revision_sha256": repaired.implementation_revision_sha256,
            "verdict": "accept", "findings": [],
        })
        assert status == 200
        assert payload["job"]["state"] == CodingJobState.CODEX_ACCEPTED.value
        assert payload["job"]["landable"] is True
        assert payload["job"]["audit_count"] == 2

        status, payload, _ = _http(service, "POST", route, body={
            "implementation_revision_sha256": repaired.implementation_revision_sha256,
            "verdict": "accept", "findings": [],
        })
        assert status == 409 and payload["error"] == "audit_state_conflict"
    finally:
        service.close()


@pytest.mark.parametrize("body", [
    {"verdict": "accept", "findings": []},
    {"implementation_revision_sha256": _AUDIT_REVISION, "findings": []},
    {"implementation_revision_sha256": _AUDIT_REVISION, "verdict": "accept"},
    {"implementation_revision_sha256": "B3" * 32, "verdict": "accept", "findings": []},
    {"implementation_revision_sha256": _AUDIT_REVISION, "verdict": "approve", "findings": []},
    {"implementation_revision_sha256": _AUDIT_REVISION, "verdict": True, "findings": []},
    {"implementation_revision_sha256": _AUDIT_REVISION, "verdict": "rework", "findings": {}},
    {"implementation_revision_sha256": _AUDIT_REVISION, "verdict": "rework",
     "findings": [{"code": "c1", "severity": "minor", "message": "m", "log": "secret"}]},
    {"implementation_revision_sha256": _AUDIT_REVISION, "verdict": "accept", "findings": [],
     "implementation_backend": "claude"},
    {"implementation_revision_sha256": _AUDIT_REVISION, "verdict": "accept", "findings": [],
     "require_codex_audit": False},
    {"implementation_revision_sha256": _AUDIT_REVISION, "verdict": "accept", "findings": [],
     "landable": True},
])
def test_http_audit_rejects_malformed_or_authority_bearing_bodies(
    tmp_path: Path, body: dict,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace)
    try:
        awaiting = _awaiting(service, "tenant-http", "http-malformed", workspace)
        status, payload, _ = _http(
            service, "POST", "/v1/coding/jobs/{}/audit".format(awaiting.job_id), body=body,
        )
        assert status == 400
        assert payload == {"ok": False, "error": "invalid_request"}
        unchanged = service.get("tenant-http", awaiting.job_id)
        assert unchanged.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert unchanged.audit_count == 0 and unchanged.landable is False
    finally:
        service.close()


def test_http_routes_stay_compatible_and_reject_unknown_audit_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    service = _service(tmp_path, workspace)
    try:
        status, payload, _ = _http(service, "GET", "/healthz", token="")
        assert status == 200 and payload["ok"] is True

        status, payload, _ = _http(
            service, "POST", "/v1/coding/jobs",
            body={"message": "task", "working_dir": str(workspace)},
            extra_headers="Idempotency-Key: http-compat\r\n",
        )
        assert status == 202
        job_id = payload["job"]["job_id"]
        completed = _wait(service, "tenant-http", job_id)
        assert completed.state is CodingJobState.COMPLETED

        status, payload, _ = _http(service, "GET", "/v1/coding/jobs/" + job_id)
        assert status == 200 and payload["job"]["job_id"] == job_id

        for unknown in ("/v1/coding/jobs/not-a-job/audit", "/v1/coding/audit", "/v1/other"):
            status, payload, _ = _http(service, "POST", unknown, body={})
            assert status == 404 and payload["error"] == "not_found"

        # Audit stays unavailable while the service is not audit-required.
        status, payload, _ = _http(
            service, "POST", "/v1/coding/jobs/{}/audit".format(job_id),
            body={
                "implementation_revision_sha256": _AUDIT_REVISION,
                "verdict": "accept", "findings": [],
            },
        )
        assert status == 403 and payload["error"] == "audit_not_enabled"
    finally:
        service.close()


def test_cli_built_native_service_stops_at_awaiting_codex_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public route never reaches completed/landable without Codex."""
    import argparse

    import flyto_ai.cli as cli

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
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
    monkeypatch.setattr(
        cli, "_create_native_coding_provider", lambda args: RealToolProvider(),
    )
    # The public route always enables the host-owned lanes, so a real Indexer
    # capability must be reachable before the implementer may edit.
    from test_coding_route import BLUEPRINT_FIXTURE, INDEXER_FIXTURE

    fixture = tmp_path / "indexer_fixture.py"
    fixture.write_text(INDEXER_FIXTURE)
    blueprint_fixture = tmp_path / "blueprint_fixture.py"
    blueprint_fixture.write_text(BLUEPRINT_FIXTURE)

    def _args(**overrides):
        values = dict(
            tenant="tenant-route", workspace_root=[str(workspace)],
            state_dir=str(tmp_path / "route-state"), provider="ollama", model=None,
            base_url=None, config=".flyto/coding.yaml", approval="never",
            sandbox="workspace-write", sandbox_image="python:3.12-slim",
            max_workers=2, max_queued=8, implementation_backend="native",
            max_rework_rounds=3,
            indexer_command="{} {}".format(sys.executable, fixture),
            blueprint_command="{} {}".format(sys.executable, blueprint_fixture),
        )
        values.update(overrides)
        return argparse.Namespace(**values)

    service = cli._build_coding_service(_args())
    assert service.route_policy is not None and service.route_policy.strict is True
    # The strict public route attaches Blueprint and Core without a flag.
    assert service.route_policy.blueprint is not None
    assert service.route_policy.core_enabled is True
    try:
        queued = service.submit("tenant-route", "route-001", _request(workspace))
        awaiting = _wait(service, "tenant-route", queued.job_id)
        assert awaiting.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert awaiting.state is not CodingJobState.COMPLETED
        assert awaiting.landable is False
        assert awaiting.implementation_backend == "native"
        assert awaiting.audit_count == 0
        assert receipt_to_mapping(awaiting)["landable"] is False
        assert (workspace / "result.txt").read_text() == "verified\n"
        # The host-owned lanes actually ran around the native implementer.
        assert awaiting.route_receipt is not None
        assert awaiting.route_receipt["ok"] is True
        lane_states = {
            item["lane"]: item["status"] for item in awaiting.route_receipt["lanes"]
        }
        assert lane_states["indexer_pre"] == "applied"
        assert lane_states["indexer_post"] == "applied"

        accepted = service.audit(
            "tenant-route", queued.job_id, awaiting.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
        assert accepted.landable is True
    finally:
        service.close()


def test_public_route_fails_closed_when_the_indexer_lane_is_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A green repository check must not mask an unreachable host-owned lane."""
    import argparse

    import flyto_ai.cli as cli

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir(exist_ok=True)
    config.write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: trivial\n"
        "    argv: {}\n".format(json.dumps([sys.executable, "-c", "pass"]))
    )
    monkeypatch.setattr(
        cli, "_create_native_coding_provider", lambda args: RealToolProvider(),
    )
    service = cli._build_coding_service(argparse.Namespace(
        tenant="tenant-route", workspace_root=[str(workspace)],
        state_dir=str(tmp_path / "closed-state"), provider="ollama", model=None,
        base_url=None, config=".flyto/coding.yaml", approval="never",
        sandbox="workspace-write", sandbox_image="python:3.12-slim",
        max_workers=2, max_queued=8, implementation_backend="native",
        max_rework_rounds=3,
        indexer_command="{} {}".format(sys.executable, tmp_path / "absent.py"),
        blueprint_command=None,
    ))
    try:
        queued = service.submit("tenant-route", "closed-001", _request(workspace))
        failed = _wait(service, "tenant-route", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.state is not CodingJobState.AWAITING_CODEX_AUDIT
        assert failed.failure_code == "route_capability_unavailable"
        assert failed.landable is False
        assert failed.route_receipt["ok"] is False
        # The implementer never edited, so no revision exists to audit.
        assert failed.implementation_revision_sha256 == ""
        assert not (workspace / "result.txt").exists()
        with pytest.raises(AuditStateConflict):
            service.audit(
                "tenant-route", queued.job_id, _AUDIT_REVISION,
                CodingAuditVerdict.ACCEPT, (),
            )
    finally:
        service.close()


def test_public_package_exports_the_canonical_audit_surface() -> None:
    import flyto_ai.coding as coding
    from flyto_ai.coding import contracts, service as service_module

    exported = {
        "CodingAuditFinding": contracts.CodingAuditFinding,
        "CodingAuditSeverity": contracts.CodingAuditSeverity,
        "CodingAuditVerdict": contracts.CodingAuditVerdict,
        "CodingJobReceipt": contracts.CodingJobReceipt,
        "CodingJobState": contracts.CodingJobState,
        "AUDIT_BOUND_CODING_JOB_STATES": contracts.AUDIT_BOUND_CODING_JOB_STATES,
        "AUDITED_CODING_JOB_STATES": contracts.AUDITED_CODING_JOB_STATES,
        "TERMINAL_CODING_JOB_STATES": contracts.TERMINAL_CODING_JOB_STATES,
        "SERVICE_CONTRACT_VERSION": contracts.SERVICE_CONTRACT_VERSION,
        "SUPPORTED_SERVICE_CONTRACT_VERSIONS": contracts.SUPPORTED_SERVICE_CONTRACT_VERSIONS,
        "MAX_AUDIT_FINDINGS": contracts.MAX_AUDIT_FINDINGS,
        "MAX_AUDIT_MESSAGE_CHARS": contracts.MAX_AUDIT_MESSAGE_CHARS,
        "MAX_AUDIT_EVIDENCE_REF_CHARS": contracts.MAX_AUDIT_EVIDENCE_REF_CHARS,
        "MAX_AUDIT_ROUNDS": contracts.MAX_AUDIT_ROUNDS,
        "audit_findings_sha256": contracts.audit_findings_sha256,
        "require_revision_sha256": contracts.require_revision_sha256,
        "validate_audit_submission": contracts.validate_audit_submission,
        "receipt_to_mapping": service_module.receipt_to_mapping,
        "CodingServiceError": service_module.CodingServiceError,
        "AuditNotEnabled": service_module.AuditNotEnabled,
        "AuditStateConflict": service_module.AuditStateConflict,
        "RevisionMismatch": service_module.RevisionMismatch,
        "RevisionUnavailable": service_module.RevisionUnavailable,
        "ReworkLimitReached": service_module.ReworkLimitReached,
        "ReworkNotResumable": service_module.ReworkNotResumable,
        "SessionBindingFailed": service_module.SessionBindingFailed,
        "CodingJobNotFound": service_module.CodingJobNotFound,
        "WorkspaceDenied": service_module.WorkspaceDenied,
    }
    for name, canonical in exported.items():
        assert name in coding.__all__, name
        assert getattr(coding, name) is canonical, name

    # Every advertised name must resolve, and remote-payload decoding and the
    # workspace digest internals stay out of the public package surface.
    assert len(coding.__all__) == len(set(coding.__all__))
    for name in coding.__all__:
        assert getattr(coding, name) is not None, name
    for internal in ("request_from_mapping", "CodingService", "_revision_digest"):
        assert internal not in coding.__all__


def test_audit_disabled_service_keeps_the_legacy_completed_flow(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    service = _service(tmp_path, workspace)
    try:
        assert service.require_codex_audit is False
        queued = service.submit("tenant-legacy", "legacy-001", _request(workspace))
        completed = _wait(service, "tenant-legacy", queued.job_id)
        assert completed.state is CodingJobState.COMPLETED
        assert completed.landable is False
        assert completed.audit_count == 0 and completed.rework_count == 0
        assert completed.implementation_backend == ""
        assert completed.implementation_revision_sha256 == ""
        with pytest.raises(AuditNotEnabled):
            service.audit(
                "tenant-legacy", queued.job_id, _AUDIT_REVISION, CodingAuditVerdict.ACCEPT, (),
            )
    finally:
        service.close()


def test_required_audit_holds_a_successful_implementation_and_accept_makes_it_landable(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace)
    try:
        awaiting = _awaiting(service, "tenant-audit", "audit-001", workspace)
        assert awaiting.landable is False
        assert awaiting.implementation_backend == "native"
        assert awaiting.implementation_session_id == awaiting.thread_id != ""
        assert len(awaiting.implementation_revision_sha256) == 64
        assert awaiting.audit_count == 0 and awaiting.rework_count == 0
        assert awaiting.result is not None and awaiting.result.ok

        accepted = service.audit(
            "tenant-audit", awaiting.job_id, awaiting.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
        assert accepted.landable is True
        assert accepted.audit_count == 1 and accepted.rework_count == 0
        assert accepted.audit_findings_sha256 == audit_findings_sha256(())
        assert accepted.implementation_revision_sha256 == awaiting.implementation_revision_sha256
        assert service.get("tenant-audit", awaiting.job_id).landable is True

        # A repeat decision on an already-accepted job fails closed.
        with pytest.raises(AuditStateConflict):
            service.audit(
                "tenant-audit", awaiting.job_id, awaiting.implementation_revision_sha256,
                CodingAuditVerdict.ACCEPT, (),
            )
    finally:
        service.close()


def test_failed_implementation_stays_failed_under_required_audit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace, provider=NoChangeProvider())
    try:
        queued = service.submit("tenant-audit", "audit-fail", _request(workspace))
        failed = _wait(service, "tenant-audit", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.landable is False
        assert failed.implementation_revision_sha256 == ""
        with pytest.raises(AuditStateConflict):
            service.audit(
                "tenant-audit", queued.job_id, _AUDIT_REVISION, CodingAuditVerdict.ACCEPT, (),
            )
    finally:
        service.close()


def test_rework_resumes_the_same_job_thread_and_session_with_a_new_revision(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    provider = ReworkingProvider()
    service = _audited_service(tmp_path, workspace, provider=provider)
    try:
        first = _awaiting(service, "tenant-audit", "audit-rework", workspace)
        finding = _blocker()
        queued = service.audit(
            "tenant-audit", first.job_id, first.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, (finding,),
        )
        assert queued.state is CodingJobState.REWORK_QUEUED
        assert queued.audit_count == 1 and queued.rework_count == 1
        assert queued.audit_findings_sha256 == audit_findings_sha256((finding,))
        assert queued.landable is False

        second = _wait(service, "tenant-audit", first.job_id)
        assert second.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert second.job_id == first.job_id
        assert second.thread_id == first.thread_id
        assert second.implementation_session_id == first.implementation_session_id
        assert second.implementation_revision_sha256 != first.implementation_revision_sha256
        assert second.audit_count == 1 and second.rework_count == 1
        assert provider.rounds == 2

        # The resumed prompt carries typed findings only.
        resumed = provider.prompts[-1]
        assert finding.code in resumed
        assert "cover the audited path" in resumed
        assert "evidence_path" not in resumed
        assert ".flyto/coding.yaml" not in resumed

        accepted = service.audit(
            "tenant-audit", first.job_id, second.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
        assert accepted.landable is True
        assert accepted.audit_count == 2 and accepted.rework_count == 1
    finally:
        service.close()


def test_rework_rounds_are_bounded_by_startup_configuration(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    provider = ReworkingProvider()
    service = _audited_service(tmp_path, workspace, provider=provider, max_rework_rounds=1)
    try:
        first = _awaiting(service, "tenant-audit", "audit-limit", workspace)
        service.audit(
            "tenant-audit", first.job_id, first.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, (_blocker(),),
        )
        second = _wait(service, "tenant-audit", first.job_id)
        assert second.state is CodingJobState.AWAITING_CODEX_AUDIT
        with pytest.raises(ReworkLimitReached):
            service.audit(
                "tenant-audit", first.job_id, second.implementation_revision_sha256,
                CodingAuditVerdict.REWORK, (_blocker("second_round"),),
            )
        # The ceiling now settles the job instead of bouncing the caller and
        # leaving a worktree claimed forever by a job no verdict could move.
        settled = service.get("tenant-audit", first.job_id)
        assert settled.state is CodingJobState.FAILED
        assert settled.failure_code == REWORK_LIMIT_FAILURE_CODE
        assert settled.rework_count == 1
        assert settled.landable is False
        assert receipt_to_mapping(settled)["job_terminal"] is True
        # Bounded history survives; resume authority and the claim do not.
        assert settled.implementation_session_id
        assert settled.implementation_revision_sha256
        assert not service._workspace_claim_path(str(workspace)).exists()
        assert provider.rounds == 2
    finally:
        service.close()


def test_concurrent_audit_calls_schedule_at_most_one_rework(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    provider = ReworkingProvider(delay=0.2)
    service = _audited_service(tmp_path, workspace, provider=provider)
    try:
        first = _awaiting(service, "tenant-audit", "audit-race", workspace)
        outcomes: list = []
        barrier = threading.Barrier(2)

        def call() -> None:
            barrier.wait()
            try:
                outcomes.append(service.audit(
                    "tenant-audit", first.job_id, first.implementation_revision_sha256,
                    CodingAuditVerdict.REWORK, (_blocker(),),
                ).state.value)
            except AuditStateConflict as exc:
                outcomes.append(exc.code)

        threads = [threading.Thread(target=call) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert sorted(outcomes) == [
            "audit_state_conflict", CodingJobState.REWORK_QUEUED.value,
        ]
        observed = set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = service.get("tenant-audit", first.job_id).state
            observed.add(state)
            if state in _SETTLED_STATES:
                break
            time.sleep(0.005)
        assert CodingJobState.REWORK_RUNNING in observed
        repaired = _wait(service, "tenant-audit", first.job_id)
        assert repaired.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert repaired.rework_count == 1
        assert provider.rounds == 2
    finally:
        service.close()


def test_audit_fails_closed_on_stale_wrong_or_incoherent_input(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace)
    try:
        awaiting = _awaiting(service, "tenant-audit", "audit-stale", workspace)
        revision = awaiting.implementation_revision_sha256

        with pytest.raises(RevisionMismatch):
            service.audit(
                "tenant-audit", awaiting.job_id, _AUDIT_REVISION, CodingAuditVerdict.ACCEPT, (),
            )
        with pytest.raises(ValueError, match="accept verdict cannot carry"):
            service.audit(
                "tenant-audit", awaiting.job_id, revision,
                CodingAuditVerdict.ACCEPT, (_blocker(),),
            )
        with pytest.raises(ValueError, match="rework verdict requires"):
            service.audit(
                "tenant-audit", awaiting.job_id, revision, CodingAuditVerdict.REWORK, (),
            )
        with pytest.raises(ValueError, match="64-character sha256"):
            service.audit(
                "tenant-audit", awaiting.job_id, "not-a-digest", CodingAuditVerdict.ACCEPT, (),
            )
        with pytest.raises(CodingJobNotFound):
            service.audit(
                "tenant-other", awaiting.job_id, revision, CodingAuditVerdict.ACCEPT, (),
            )
        with pytest.raises(CodingJobNotFound):
            service.audit(
                "tenant-audit", "job_" + "f" * 24, revision, CodingAuditVerdict.ACCEPT, (),
            )

        # None of the rejected calls mutated the job.
        unchanged = service.get("tenant-audit", awaiting.job_id)
        assert unchanged.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert unchanged.audit_count == 0 and unchanged.rework_count == 0
        assert unchanged.audit_findings_sha256 == ""
        assert unchanged.landable is False
        assert unchanged.implementation_revision_sha256 == revision
    finally:
        service.close()


def test_live_workspace_edit_invalidates_the_audited_revision(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace)
    try:
        awaiting = _awaiting(service, "tenant-audit", "audit-live", workspace)
        (workspace / "result.txt").write_text("tampered\n")
        with pytest.raises(RevisionMismatch):
            service.audit(
                "tenant-audit", awaiting.job_id, awaiting.implementation_revision_sha256,
                CodingAuditVerdict.ACCEPT, (),
            )
        assert service.get("tenant-audit", awaiting.job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )

        # Deletion is a distinct, deterministic revision, not a missing read.
        (workspace / "result.txt").unlink()
        deleted = CodingService._revision_digest(str(workspace), ["result.txt"])
        (workspace / "result.txt").write_text("verified\n")
        assert deleted != awaiting.implementation_revision_sha256
        assert CodingService._revision_digest(str(workspace), ["result.txt"]) == (
            awaiting.implementation_revision_sha256
        )
    finally:
        service.close()


def test_revision_digest_rejects_unsafe_attributable_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    (workspace / "result.txt").write_text("verified\n")
    (workspace / "package").mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n")
    escape = workspace / "escape.txt"
    escape.symlink_to(outside)

    for unsafe in (
        "../outside.txt",
        "package/../../outside.txt",
        str(outside),
        "/etc/passwd",
        "~/secret.txt",
        ".git/config",
        ".env",
        ".env.local",
        "escape.txt",
        "package",
        "",
        "result.txt\x00",
        "a" * 1200,
    ):
        with pytest.raises(RevisionUnavailable):
            CodingService._revision_digest(str(workspace), [unsafe])
    with pytest.raises(RevisionUnavailable):
        CodingService._revision_digest(str(workspace), [])
    with pytest.raises(RevisionUnavailable):
        CodingService._revision_digest(
            str(workspace), ["file{}.txt".format(index) for index in range(600)],
        )
    with pytest.raises(RevisionUnavailable):
        CodingService._revision_digest(str(tmp_path / "missing"), ["result.txt"])
    assert len(CodingService._revision_digest(str(workspace), ["result.txt"])) == 64


def test_rework_revision_stays_cumulative_over_earlier_attributable_files(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    provider = PartialReworkProvider()
    service = _audited_service(tmp_path, workspace, provider=provider)
    try:
        first = _awaiting(service, "tenant-audit", "audit-cumulative", workspace)
        assert (workspace / "helper.txt").read_text() == "original helper\n"
        service.audit(
            "tenant-audit", first.job_id, first.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, (_blocker(),),
        )
        second = _wait(service, "tenant-audit", first.job_id)
        assert second.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert provider.rounds == 2

        record = service._read_json(
            service._tenant_dir(service._tenant_ref("tenant-audit"))
            / "jobs" / (first.job_id + ".json"),
        )
        # Round 2 never touched helper.txt, but it stays part of the revision.
        assert record["implementation_files"] == ["helper.txt", "notes.txt", "result.txt"]

        (workspace / "helper.txt").write_text("tampered after the audited round\n")
        with pytest.raises(RevisionMismatch):
            service.audit(
                "tenant-audit", first.job_id, second.implementation_revision_sha256,
                CodingAuditVerdict.ACCEPT, (),
            )
        unchanged = service.get("tenant-audit", first.job_id)
        assert unchanged.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert unchanged.landable is False
        assert unchanged.audit_count == 1

        (workspace / "helper.txt").write_text("original helper\n")
        accepted = service.audit(
            "tenant-audit", first.job_id, second.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.landable is True
    finally:
        service.close()


def test_restart_refuses_a_changed_workspace_root_set_before_reading_a_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace)
    try:
        awaiting = _awaiting(service, "tenant-audit", "audit-authority", workspace)
    finally:
        service.close()

    narrowed = tmp_path / "narrowed"
    narrowed.mkdir()

    def forbidden(*args, **kwargs):
        raise AssertionError("the old workspace must not be hashed")

    monkeypatch.setattr(CodingService, "_revision_digest", staticmethod(forbidden))
    with pytest.raises(CodingAuthorityConflict):
        CodingService(
            lambda store: FlytoCodingAgent(RealToolProvider(), store=store),
            state_root=str(tmp_path / "service-state"),
            workspace_roots=(str(narrowed),),
            require_codex_audit=True,
        )

    # Refusal is before audit/rework code can read or hash the old workspace,
    # and the audit-ready record remains untouched for a compatible restart.
    record = json.loads(
        (tmp_path / "service-state" / "tenants"
         / CodingService._tenant_ref("tenant-audit") / "jobs"
         / (awaiting.job_id + ".json")).read_text(encoding="utf-8"),
    )
    assert record["state"] == CodingJobState.AWAITING_CODEX_AUDIT.value
    assert record["audit_count"] == 0
    assert record["landable"] is False


def _fake_stat(source: os.stat_result, **overrides):
    class _Stat:
        pass

    projected = _Stat()
    for name in ("st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns"):
        setattr(projected, name, overrides.get(name, getattr(source, name)))
    return projected


@pytest.mark.parametrize(("call_index", "overrides"), [
    (1, {"st_ino": 987654321}),
    (1, {"st_dev": 987654321}),
    (1, {"st_mode": stat.S_IFDIR | 0o755}),
    (2, {"st_size": 4096}),
    (2, {"st_mtime_ns": 1}),
    (2, {"st_ino": 987654321}),
])
def test_revision_hashing_detects_a_swapped_or_mutated_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, call_index: int, overrides: dict,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    (workspace / "result.txt").write_text("verified\n")
    assert len(CodingService._revision_digest(str(workspace), ["result.txt"])) == 64

    real_fstat = os.fstat
    calls = {"count": 0}

    def counting_fstat(fd):
        calls["count"] += 1
        result = real_fstat(fd)
        if calls["count"] == call_index:
            return _fake_stat(result, **overrides)
        return result

    monkeypatch.setattr(os, "fstat", counting_fstat)
    with pytest.raises(RevisionUnavailable):
        CodingService._revision_digest(str(workspace), ["result.txt"])


def test_revision_hashing_fails_closed_without_o_nofollow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    (workspace / "result.txt").write_text("verified\n")

    # The documented fallback: identity comparison, not the open flag, is what
    # rejects a pathname substituted between the lstat and the open.
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    assert len(CodingService._revision_digest(str(workspace), ["result.txt"])) == 64

    real_fstat = os.fstat

    def swapped_fstat(fd):
        return _fake_stat(real_fstat(fd), st_ino=123456789)

    monkeypatch.setattr(os, "fstat", swapped_fstat)
    with pytest.raises(RevisionUnavailable):
        CodingService._revision_digest(str(workspace), ["result.txt"])


def test_revision_hashing_rejects_unreadable_and_dangling_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    (workspace / "dangling.txt").symlink_to(tmp_path / "never-created.txt")
    with pytest.raises(RevisionUnavailable):
        CodingService._revision_digest(str(workspace), ["dangling.txt"])

    (workspace / "package").mkdir()
    (workspace / "package" / "module.py").write_text("value = 1\n")
    with pytest.raises(RevisionUnavailable):
        CodingService._revision_digest(str(workspace), ["package/module.py/nested.py"])

    executable = workspace / "run.sh"
    executable.write_text("#!/bin/sh\n")
    plain = CodingService._revision_digest(str(workspace), ["run.sh"])
    executable.chmod(0o755)
    assert CodingService._revision_digest(str(workspace), ["run.sh"]) != plain


def test_restart_preserves_awaiting_audit_and_reworks_the_same_session(
    tmp_path: Path,
) -> None:
    """A restarted worker continues the exact session; it never opens a new one.

    The durable resume envelope replaced the process-local prompt cache, so an
    audit that lands on a worker which never implemented the job — a restart
    here, a different Codex frontend in practice — can still send rework. What
    must not change is the session: the envelope is bound to one
    `implementation_session_id` and can only ever continue it.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        awaiting = _awaiting(service, "tenant-audit", "audit-restart", workspace)
    finally:
        service.close()

    # A fresh provider restarts its own round counter, so offset it to keep the
    # replacement worker's writes distinct from the bytes already on disk.
    successor = ReworkingProvider()
    successor.rounds = 10
    restarted = _audited_service(tmp_path, workspace, provider=successor)
    try:
        persisted = restarted.get("tenant-audit", awaiting.job_id)
        assert persisted.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert persisted.implementation_session_id == awaiting.implementation_session_id

        queued = restarted.audit(
            "tenant-audit", awaiting.job_id, persisted.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, (_blocker(),),
        )
        assert queued.state is CodingJobState.REWORK_QUEUED
        reworked = _wait(restarted, "tenant-audit", awaiting.job_id)
        assert reworked.state is CodingJobState.AWAITING_CODEX_AUDIT
        # The same session, continued. A fresh session would be a different id.
        assert reworked.implementation_session_id == awaiting.implementation_session_id
        assert reworked.thread_id == awaiting.thread_id
        assert reworked.audit_count == 1 and reworked.rework_count == 1
        assert successor.rounds == 11

        accepted = restarted.audit(
            "tenant-audit", awaiting.job_id, reworked.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
        assert accepted.landable is True
    finally:
        restarted.close()


def test_rework_fails_closed_when_the_resume_envelope_is_gone(tmp_path: Path) -> None:
    """Without a session-bound envelope the loop refuses rather than guesses."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        awaiting = _awaiting(service, "tenant-audit", "audit-noenv", workspace)
    finally:
        service.close()

    restarted = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        tenant_ref = restarted._tenant_ref("tenant-audit")
        restarted._resume_path(tenant_ref, awaiting.job_id).unlink()
        persisted = restarted.get("tenant-audit", awaiting.job_id)

        with pytest.raises(ReworkNotResumable):
            restarted.audit(
                "tenant-audit", awaiting.job_id, persisted.implementation_revision_sha256,
                CodingAuditVerdict.REWORK, (_blocker(),),
            )
        # A refused rework consumes no audit round and leaves the job auditable.
        assert restarted.get("tenant-audit", awaiting.job_id).audit_count == 0
        accepted = restarted.audit(
            "tenant-audit", awaiting.job_id, persisted.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
        assert accepted.landable is True
    finally:
        restarted.close()


def test_restart_rejects_an_awaiting_job_whose_workspace_moved_on(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace)
    try:
        awaiting = _awaiting(service, "tenant-audit", "audit-drift", workspace)
    finally:
        service.close()

    (workspace / "result.txt").write_text("changed after the audit request\n")
    restarted = _audited_service(tmp_path, workspace)
    try:
        with pytest.raises(RevisionMismatch):
            restarted.audit(
                "tenant-audit", awaiting.job_id, awaiting.implementation_revision_sha256,
                CodingAuditVerdict.ACCEPT, (),
            )
        assert restarted.get("tenant-audit", awaiting.job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
    finally:
        restarted.close()


def test_restart_reconciles_interrupted_rework_to_a_stable_failure(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace)
    try:
        awaiting = _awaiting(service, "tenant-audit", "audit-inflight", workspace)
        record_path = (
            service._tenant_dir(service._tenant_ref("tenant-audit"))
            / "jobs" / (awaiting.job_id + ".json")
        )
        # Simulate a process that died while a rework round was running.
        record = service._read_json(record_path)
        record.update({
            "state": CodingJobState.REWORK_RUNNING.value,
            "audit_count": 1,
            "rework_count": 1,
            "audit_findings_sha256": audit_findings_sha256((_blocker(),)),
        })
        service._write_json(record_path, record)
    finally:
        service.close()

    restarted = _audited_service(tmp_path, workspace)
    try:
        reconciled = restarted.get("tenant-audit", awaiting.job_id)
        assert reconciled.state is CodingJobState.FAILED
        assert reconciled.failure_code == "service_restarted"
        assert reconciled.landable is False
        assert reconciled.rework_count == 1
        with pytest.raises(AuditStateConflict):
            restarted.audit(
                "tenant-audit", awaiting.job_id, reconciled.implementation_revision_sha256,
                CodingAuditVerdict.ACCEPT, (),
            )
    finally:
        restarted.close()


def test_audit_authority_is_startup_only_and_receipts_stay_redacted(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    for forbidden in (
        {"implementation_backend": "claude"},
        {"require_codex_audit": False},
        {"max_rework_rounds": 99},
        {"landable": True},
        {"model": "claude-opus-5"},
        {"provider": "anthropic"},
        {"audit_findings_sha256": _AUDIT_REVISION},
        {"implementation_session_id": "sdk-1"},
    ):
        payload = {"message": "task", "working_dir": str(workspace)}
        payload.update(forbidden)
        with pytest.raises(ValueError, match="unsupported coding request fields"):
            request_from_mapping(payload)
    for invalid, match in (
        ({"implementation_backend": "not a name"}, "implementation_backend must be"),
        ({"implementation_backend": ""}, "implementation_backend must be"),
        ({"max_rework_rounds": 0}, "max_rework_rounds must be between"),
        ({"max_rework_rounds": 100}, "max_rework_rounds must be between"),
        ({"require_codex_audit": "yes"}, "require_codex_audit must be a boolean"),
    ):
        with pytest.raises(ValueError, match=match):
            CodingService(
                lambda store: FlytoCodingAgent(RealToolProvider(), store=store),
                state_root=str(tmp_path / "invalid-state"),
                workspace_roots=(str(workspace),),
                **invalid,
            )

    service = _audited_service(tmp_path, workspace)
    try:
        awaiting = _awaiting(service, "tenant-audit", "audit-redaction", workspace)
        accepted = service.audit(
            "tenant-audit", awaiting.job_id, awaiting.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        public = receipt_to_mapping(accepted)
        assert public["landable"] is True
        assert public["implementation_backend"] == "native"
        assert public["audit_findings_sha256"] == audit_findings_sha256(())
        assert "evidence_path" not in public["result"]
        assert "output_preview" not in public["result"]["checks"][0]
        assert "working_dir" not in public
        assert str(workspace) not in json.dumps(public)
    finally:
        service.close()


def test_mcp_audit_tool_drives_the_real_service_state_machine(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    provider = ReworkingProvider()
    service = _audited_service(tmp_path, workspace, provider=provider)
    server = CodingMCPServer(service, "tenant-mcp-audit")
    try:
        awaiting = _awaiting(service, "tenant-mcp-audit", "audit-mcp", workspace)
        reworked = server.handle(_audit_request(
            job_id=awaiting.job_id,
            implementation_revision_sha256=awaiting.implementation_revision_sha256,
            verdict="rework",
            findings=[_valid_finding_payload()],
        ))["result"]["structuredContent"]["job"]
        assert reworked["state"] == CodingJobState.REWORK_QUEUED.value
        assert reworked["landable"] is False

        repaired = _wait(service, "tenant-mcp-audit", awaiting.job_id)
        assert repaired.state is CodingJobState.AWAITING_CODEX_AUDIT

        stale = server.handle(_audit_request(
            job_id=awaiting.job_id,
            implementation_revision_sha256=awaiting.implementation_revision_sha256,
            verdict="accept",
            findings=[],
        ))["result"]
        assert stale["isError"] is True
        # Every service error now carries the typed envelope, so a caller can
        # branch on phase and retryability without a per-code lookup table.
        assert stale["structuredContent"] == {
            "ok": False,
            "error": "revision_mismatch",
            "details": {"failure_phase": "service", "retryable": False},
        }

        accepted = server.handle(_audit_request(
            job_id=awaiting.job_id,
            implementation_revision_sha256=repaired.implementation_revision_sha256,
            verdict="accept",
            findings=[],
        ))["result"]["structuredContent"]["job"]
        assert accepted["state"] == CodingJobState.CODEX_ACCEPTED.value
        assert accepted["landable"] is True
        assert accepted["audit_count"] == 2
        assert "evidence_path" not in json.dumps(accepted)
    finally:
        service.close()


def test_service_options_never_reintroduce_the_nested_session_marker(
    tmp_path: Path,
) -> None:
    """A real provider start must not look like a nested Claude Code session.

    The installed SDK strips an inherited ``CLAUDECODE`` from the child
    environment and then merges ``options.env`` over it, so shipping the key
    with any value at all puts the marker back and the CLI rejects the start
    before a session exists. The key has to be absent, not empty or false.
    """

    from flyto_ai.agents.claude_code import ClaudeCodeAgent
    from flyto_ai.agents.models import CodeTaskRequest

    for authority in (True, False):
        options = ClaudeCodeAgent()._option_kwargs(
            CodeTaskRequest(
                message="task", working_dir=str(tmp_path),
                service_mode=True, service_edit_authority=authority,
            ),
            session_id=None, system_prompt="s", max_turns=1, max_budget=1.0,
        )
        env = options["env"]
        assert isinstance(env, dict)
        assert "CLAUDECODE" not in env
        # Nothing else may smuggle the marker back in under another value.
        assert "CLAUDECODE" not in json.dumps(env)
        # The pinned service route is unchanged by the fix.
        assert options["model"] == "claude-opus-5"


class RaisingClaudeBackend:
    """Stands in for an SDK that dies before any session exists."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.requests: list = []

    async def run(self, request):
        self.requests.append(request)
        raise self.exc


def _provider_error_event(store, thread_id: str):
    events = [
        event for event in store.events(thread_id)
        if event.get("type") == "coding.provider_error"
    ]
    assert len(events) == 1
    return events[0]


def test_provider_start_failure_is_durably_recorded_under_the_returned_thread(
    tmp_path: Path,
) -> None:
    """A start that dies before a session still leaves sanitized evidence."""

    workspace = _claude_workspace(tmp_path, check="trivial")
    secret = "sk-live-DEADBEEF /Users/someone/private/token.txt"
    backend = RaisingClaudeBackend(RuntimeError(secret))
    adapter, store = _claude_adapter(tmp_path, workspace, backend)

    failed = asyncio.run(adapter.run(CodingTaskRequest(
        message="implement", working_dir=str(workspace), require_changes=False,
    )))
    assert failed.ok is False
    assert failed.failure_code == "provider_failed"
    assert failed.thread_id.startswith("host-")

    # The diagnostic is durable and lives under the exact returned thread id.
    event = _provider_error_event(store, failed.thread_id)
    assert event["data"] == {
        "backend": "claude-sdk",
        "error_class": "RuntimeError",
        "failure_code": "provider_failed",
    }
    # Nothing derived from the message, arguments, or environment is kept.
    recorded = json.dumps(store.events(failed.thread_id))
    assert secret not in recorded
    assert "sk-live" not in recorded and "token.txt" not in recorded
    assert secret not in failed.message


def test_provider_start_failure_uses_one_host_thread_id(tmp_path: Path) -> None:
    """The diagnostic and the returned failure must name the same thread."""

    workspace = _claude_workspace(tmp_path, check="trivial")
    adapter, store = _claude_adapter(
        tmp_path, workspace, RaisingClaudeBackend(RuntimeError("boom")),
    )
    failed = asyncio.run(adapter.run(CodingTaskRequest(
        message="implement", working_dir=str(workspace), require_changes=False,
    )))
    # Exactly one provisional thread exists, and it is the one returned.
    threads = [
        thread.name for thread in (tmp_path / "claude-threads").iterdir()
        if thread.is_dir()
    ]
    assert threads == [failed.thread_id]
    _provider_error_event(store, failed.thread_id)

    # A supplied safe thread id is preserved rather than replaced.
    supplied = "sdk-session-resume"
    resumed = asyncio.run(adapter.run(CodingTaskRequest(
        message="implement", working_dir=str(workspace), thread_id=supplied,
        resume=True, require_changes=False,
    )))
    assert resumed.thread_id == supplied
    _provider_error_event(store, supplied)


@pytest.mark.parametrize("text", [
    "Claude Code returned an error result: Reached maximum number of turns (2)",
    "error_max_turns",
])
def test_known_turn_limit_is_classified_apart_from_an_unknown_failure(
    tmp_path: Path, text: str,
) -> None:
    """A bounded turn limit is nameable; anything else stays provider_failed."""

    workspace = _claude_workspace(tmp_path, check="trivial")
    adapter, store = _claude_adapter(
        tmp_path, workspace, RaisingClaudeBackend(Exception(text)),
    )
    limited = asyncio.run(adapter.run(CodingTaskRequest(
        message="implement", working_dir=str(workspace), require_changes=False,
    )))
    assert limited.ok is False
    assert limited.failure_code == "turn_limit_exceeded"
    # The raw SDK text never reaches the receipt or the durable record.
    assert text not in limited.message
    event = _provider_error_event(store, limited.thread_id)
    assert event["data"]["failure_code"] == "turn_limit_exceeded"
    assert event["data"]["error_class"] == "Exception"
    assert text not in json.dumps(event)

    unknown_adapter, unknown_store = _claude_adapter(
        tmp_path / "unknown", workspace,
        RaisingClaudeBackend(Exception("upstream 500 from the provider")),
    )
    unknown = asyncio.run(unknown_adapter.run(CodingTaskRequest(
        message="implement", working_dir=str(workspace), require_changes=False,
    )))
    assert unknown.failure_code == "provider_failed"
    unknown_event = _provider_error_event(unknown_store, unknown.thread_id)
    assert unknown_event["data"]["failure_code"] == "provider_failed"


# ──────────────────────────────────────────────────────────────────────
# the host-pinned verification contract, across a same-job rework round
# ──────────────────────────────────────────────────────────────────────


def test_a_rework_round_reapplies_the_pin_and_never_the_edited_contract(
    tmp_path: Path,
) -> None:
    """The escalation this closes, end to end, through the real state machine.

    Round one is admitted against the contract the repository declared. The
    contract file is then replaced with bytes no reader accepts - the strongest
    available stand-in for "the implementation rewrote its own verifier", because
    a round that re-read the file could not even parse it and would fail
    `invalid_config` before deriving a single check.

    Round two nevertheless completes, which is only possible if it executed the
    snapshot the host captured before the first provider call. The pin is
    therefore not a way to skip verification: it is what keeps a job that touched
    its own contract still gradeable against the checks it was admitted under.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    provider = ReworkingProvider()
    service = _audited_service(tmp_path, workspace, provider=provider)
    try:
        first = _awaiting(service, "tenant-pin", "pin-rework", workspace)
        record_path = (
            service._tenant_dir(service._tenant_ref("tenant-pin"))
            / "jobs" / (first.job_id + ".json")
        )
        record = service._read_json(record_path)
        bound = record["contract_snapshot_sha256"]
        authorized = record["authorized_config_sha256"]
        # Captured by value, and addressable. Both are required: a digest alone
        # could only ever refuse, never keep executing.
        assert record["contract_snapshot"]["config_sha256"] == authorized
        assert [check["name"] for check in record["contract_snapshot"]["checks"]] == [
            "real_file_check",
        ]
        assert service._record_pinned_contract(record).identity() == bound

        # The contract the round was graded by is replaced by something no
        # reader will accept, exactly as a self-editing implementation could.
        config = workspace / ".flyto" / "coding.yaml"
        config.write_text("checks:\n  - name: [unclosed\n", encoding="utf-8")
        assert service._observed_config_digest(str(workspace)) == ""

        queued = service.audit(
            "tenant-pin", first.job_id, first.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, (_blocker(),),
        )
        assert queued.state is CodingJobState.REWORK_QUEUED

        second = _wait(service, "tenant-pin", first.job_id)
        # Reached at all, rather than stranded on `invalid_config`.
        assert second.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert second.failure_code is None
        assert provider.rounds == 2
        assert second.result is not None
        # ...and graded by the pinned check, not by anything now on disk.
        assert [check.name for check in second.result.checks] == ["real_file_check"]
        assert all(check.passed for check in second.result.checks)

        # The authority the job carries is unchanged by the edit. Recomputing it
        # from the current file is the escalation; restoring it is the fix.
        after = service._read_json(record_path)
        assert after["contract_snapshot_sha256"] == bound
        assert after["authorized_config_sha256"] == authorized
    finally:
        service.close()


def test_a_caller_may_never_supply_its_own_verifier(tmp_path: Path) -> None:
    """The pin is the verifier, so a decodable request must not be able to set one."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    for field in ("pinned_contract", "authorized_config_sha256"):
        with pytest.raises(ValueError, match="unsupported coding request fields"):
            request_from_mapping({
                "message": "task", "working_dir": str(workspace), field: None,
            })
    # A locally-constructed request may name one, but the service overwrites
    # every authority field from startup configuration before it is used.
    service = _service(tmp_path, workspace)
    try:
        smuggled = dataclasses.replace(
            _request(workspace), authorized_config_sha256="c" * 64,
        )
        assert service._with_startup_authority(smuggled).authorized_config_sha256 == ""
        assert service._with_startup_authority(smuggled).pinned_contract is None
    finally:
        service.close()
