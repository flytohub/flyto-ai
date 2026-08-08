from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import stat
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
from flyto_ai.coding.http_server import build_http_server
from flyto_ai.coding.mcp_server import (
    _AUDIT_ARGUMENT_FIELDS,
    CodingMCPServer,
    MCP_PROTOCOL_VERSION,
)
from flyto_ai.coding.service import (
    AuditNotEnabled,
    AuditStateConflict,
    CodingJobNotFound,
    CodingService,
    IdempotencyConflict,
    RevisionMismatch,
    RevisionUnavailable,
    ReworkLimitReached,
    ReworkNotResumable,
    WorkspaceDenied,
    request_from_mapping,
    receipt_to_mapping,
)

TEST_BEARER_TOKEN = "unit-test-bearer-token"
# The implementer stops at `awaiting_codex_audit`; waiting only for the legacy
# terminal states would hang once the phase-2 service lands that transition.
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
        workspace_roots=(str(workspace),),
        max_workers=2,
        max_queued=8,
        require_codex_audit=require_codex_audit,
        max_rework_rounds=max_rework_rounds,
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


def test_audit_disabled_service_keeps_the_legacy_completed_flow(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
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
        unchanged = service.get("tenant-audit", first.job_id)
        assert unchanged.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert unchanged.rework_count == 1
        assert provider.rounds == 2
    finally:
        service.close()


def test_concurrent_audit_calls_schedule_at_most_one_rework(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
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


def test_restart_revalidates_the_persisted_workspace_before_reading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace)
    try:
        awaiting = _awaiting(service, "tenant-audit", "audit-authority", workspace)
    finally:
        service.close()

    narrowed = tmp_path / "narrowed"
    narrowed.mkdir()
    restarted = CodingService(
        lambda store: FlytoCodingAgent(RealToolProvider(), store=store),
        state_root=str(tmp_path / "service-state"),
        workspace_roots=(str(narrowed),),
        require_codex_audit=True,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("the old workspace must not be hashed")

    monkeypatch.setattr(CodingService, "_revision_digest", staticmethod(forbidden))
    try:
        with pytest.raises(WorkspaceDenied):
            restarted.audit(
                "tenant-audit", awaiting.job_id, awaiting.implementation_revision_sha256,
                CodingAuditVerdict.ACCEPT, (),
            )
        with pytest.raises(WorkspaceDenied):
            restarted.audit(
                "tenant-audit", awaiting.job_id, awaiting.implementation_revision_sha256,
                CodingAuditVerdict.REWORK, (_blocker(),),
            )
        still_awaiting = restarted.get("tenant-audit", awaiting.job_id)
        assert still_awaiting.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert still_awaiting.audit_count == 0
        assert still_awaiting.landable is False
    finally:
        restarted.close()


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


def test_restart_preserves_awaiting_audit_and_fails_closed_on_rework(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _audited_service(tmp_path, workspace)
    try:
        awaiting = _awaiting(service, "tenant-audit", "audit-restart", workspace)
    finally:
        service.close()

    restarted = _audited_service(tmp_path, workspace)
    try:
        persisted = restarted.get("tenant-audit", awaiting.job_id)
        assert persisted.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert persisted.implementation_session_id == awaiting.implementation_session_id
        # The task prompt is not persisted, so a new session is never started.
        with pytest.raises(ReworkNotResumable):
            restarted.audit(
                "tenant-audit", awaiting.job_id, persisted.implementation_revision_sha256,
                CodingAuditVerdict.REWORK, (_blocker(),),
            )
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
    for forbidden in (
        {"implementation_backend": "claude"},
        {"require_codex_audit": False},
        {"max_rework_rounds": 99},
        {"landable": True},
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
        assert stale["structuredContent"] == {"ok": False, "error": "revision_mismatch"}

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
