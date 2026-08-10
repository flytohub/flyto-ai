# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""A verifier that cannot start is a host defect, all the way out to the caller.

Submit-time preflight already refuses a repository whose required check names a
program this host does not have.  What this module is about is everything that
happens when the tool disappears *after* that answer was given - between submit
and the adapter, or between the provider and verification - because the refusal
has to survive those races with its meaning intact.

The failure it exists to prevent is a category error: verification could not
run, and the round gets blamed for it.  ``route_implementation_not_successful``
and ``service_execution_failed`` both describe the change, so both send whoever
reads them to look at code that was never the problem, while the actual work -
install a tool - goes unnamed.

Three things are therefore proven at each boundary: the code stays
``verification_tool_missing`` with the phase, retryability and action that
belong to it; the blocker names cross as identifiers and nothing else crosses at
all; and a round whose verification never ran does not become auditable or
landable.  The facade tests drive the real production handlers rather than a
second implementation of them, because a projection that is only correct in a
test double is not a projection.
"""
import asyncio
import json
import stat
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from flyto_ai.agents.models import CodeTaskResponse
from flyto_ai.coding.checks import VerificationToolUnavailable
from flyto_ai.coding.contracts import (
    ACTION_INSTALL_VERIFICATION_TOOL,
    TERMINAL_CODING_JOB_STATES,
    JOB_FAILURE_SEMANTICS,
    CodingJobReceipt,
    CodingJobState,
    CodingTaskResult,
    safe_blockers,
)
from flyto_ai.coding.service import (
    CodingServiceError,
    VerificationToolMissing,
    error_details,
    receipt_to_mapping,
)

_ABSENT = "definitely-not-installed-xyz"
_CODE = "verification_tool_missing"
_SESSION = "sess-0123456789abcdef"


def _contract(root: Path, argv, name="unit"):
    (root / ".flyto").mkdir(parents=True, exist_ok=True)
    rendered = ", ".join('"{}"'.format(part) for part in argv)
    (root / ".flyto" / "coding.yaml").write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: {}\n"
        "    argv: [{}]\n"
        "    required: true\n".format(name, rendered),
        encoding="utf-8",
    )
    return str(root)


# --------------------------------------------------------------------------
# the typed vocabulary itself
# --------------------------------------------------------------------------


def test_the_failure_code_carries_preflight_semantics():
    """One table, so the early refusal and the late one say the same thing."""

    phase, retryable, actions = JOB_FAILURE_SEMANTICS[_CODE]
    assert phase == "preflight"
    assert retryable is False
    assert actions == (ACTION_INSTALL_VERIFICATION_TOOL,)


@pytest.mark.parametrize(
    "hostile",
    [
        ["/usr/local/bin/pytest"],
        ["../../etc/passwd"],
        ["rm -rf /"],
        ["AWS_SECRET_ACCESS_KEY=abc123"],
        ["unit; cat /etc/shadow"],
        ["a" * 400],
        ["check\nname"],
        "unit",
        [{"name": "unit"}, None, 7],
    ],
)
def test_blocker_shaped_input_is_refused_not_truncated(hostile):
    """Dropped whole, never shortened into something that looks like a name.

    Truncation is the dangerous option: half a path is still a path, and it
    still reads like an identifier once it is short enough.
    """

    assert safe_blockers(hostile) == ()


def test_valid_blockers_are_bounded_and_deduplicated():
    assert safe_blockers(["unit", "unit", "lint"]) == ("lint", "unit")
    assert len(safe_blockers(["check_{}".format(i) for i in range(50)])) == 8


# --------------------------------------------------------------------------
# both adapters, both timings
# --------------------------------------------------------------------------


def _native_agent(tmp_path):
    from flyto_ai.coding.agent import FlytoCodingAgent
    from flyto_ai.coding.store import ThreadStore

    class _Provider:
        def __init__(self):
            self.calls = 0

        async def chat(self, **kwargs):
            self.calls += 1
            return "done", [], 1, {}

    provider = _Provider()
    agent = FlytoCodingAgent(provider, store=ThreadStore(str(tmp_path / "store")))
    return agent, provider


def test_native_adapter_refuses_before_any_provider_call(tmp_path):
    """Submit said yes; by the time the adapter ran, the tool had gone."""

    from flyto_ai.coding.contracts import CodingTaskRequest

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _contract(workspace, [_ABSENT])
    agent, provider = _native_agent(tmp_path)

    result = asyncio.run(agent.run(CodingTaskRequest(
        message="change something", working_dir=str(workspace),
    )))

    assert result.ok is False
    assert result.failure_code == _CODE
    assert result.attempts == 0
    assert result.rounds_used == 0
    assert result.verification_blockers == ("unit",)
    # No provider was contacted, so nothing about a model can be blamed.
    assert provider.calls == 0
    assert _ABSENT not in result.message


def test_native_adapter_survives_the_late_race(tmp_path, monkeypatch):
    """The provider really ran; the verifier vanished before it could be used."""

    from flyto_ai.coding.contracts import CodingTaskRequest
    from flyto_ai.coding import agent as agent_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _contract(workspace, ["python3", "--version"])
    agent, provider = _native_agent(tmp_path)

    async def vanished(self, checks):
        raise VerificationToolUnavailable(("unit",))

    monkeypatch.setattr(agent_module.CheckRunner, "run", vanished)
    result = asyncio.run(agent.run(CodingTaskRequest(
        message="change something", working_dir=str(workspace),
    )))
    monkeypatch.undo()

    assert result.failure_code == _CODE
    assert result.verification_blockers == ("unit",)
    # Honest about what did happen: a provider attempt really occurred.
    assert result.attempts >= 1
    assert provider.calls >= 1
    # And with no verification there is no verdict, so nothing landable.
    assert result.ok is False
    assert not result.checks


# --------------------------------------------------------------------------
# the optional backend, driven for real
# --------------------------------------------------------------------------


class _Backend:
    """A controlled stand-in for the real SDK backend.

    It answers the one question these tests are about - what the adapter does
    around the provider - without a paid session. It can refuse before the
    provider like the real nested boundary does, signal the host-owned start
    callback at exactly the point the real backend signals it, and return a real
    ``CodeTaskResponse``.
    """

    def __init__(self, *, raises=None, fail_before_signal=False, turns=3):
        self.requests = []
        self.on_provider_start = None
        self.signalled = 0
        self._raises = raises
        self._fail_before_signal = fail_before_signal
        self._turns = turns

    async def run(self, request):
        self.requests.append(request)
        if self._raises is not None:
            # The real backend raises these while assembling the session, before
            # any provider interaction - so before the start signal.
            raise self._raises
        if self._fail_before_signal:
            raise RuntimeError("died assembling the session")
        # The seam the production backend uses, at the moment it uses it.
        from flyto_ai.agents.claude_code import signal_provider_start

        signal_provider_start(self)
        self.signalled += 1
        return CodeTaskResponse(
            ok=True,
            message="done",
            session_id="local-1",
            attempts=1,
            claude_session_id=_SESSION,
            claude_num_turns=self._turns,
            claude_usage={"input_tokens": 11, "output_tokens": 22},
        )


def _claude_workspace(tmp_path, argv, name="unit"):
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    _contract(workspace, argv, name=name)
    return workspace


def _claude_request(workspace):
    from flyto_ai.coding.contracts import ApprovalPolicy, CodingTaskRequest, SandboxMode

    return CodingTaskRequest(
        message="do the work",
        working_dir=str(workspace),
        approval_policy=ApprovalPolicy.NEVER,
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
    )


def _claude_agent(tmp_path, backend):
    from flyto_ai.agents.claude_code import ClaudeCodingAgent
    from flyto_ai.coding.store import ThreadStore

    return ClaudeCodingAgent(ThreadStore(str(tmp_path / "threads")), agent=backend)


def test_the_native_marker_is_written_only_by_provider_work(tmp_path):
    """The native peer signals at `provider.chat`, not on entering the adapter.

    Both halves matter: a refusal that never reaches a provider must leave the
    marker false, and a real call must set it before the provider is entered so
    a crash inside it is still recorded.
    """

    from flyto_ai.coding.contracts import CodingTaskRequest

    # Half one: pre-provider refusal signals nothing.
    refused = tmp_path / "refused"
    refused.mkdir()
    _contract(refused, [_ABSENT])
    agent, provider = _native_agent(tmp_path / "a")
    signals = []
    agent.store.on_provider_start = lambda: signals.append("started")
    asyncio.run(agent.run(CodingTaskRequest(
        message="m", working_dir=str(refused),
    )))
    assert provider.calls == 0
    assert signals == []

    # Half two: a real provider call signals exactly once, before the call.
    running = tmp_path / "running"
    running.mkdir()
    _contract(running, [sys.executable, "-c", "pass"])
    agent, provider = _native_agent(tmp_path / "b")
    order = []
    agent.store.on_provider_start = lambda: order.append("marked")

    async def chat(**kwargs):
        order.append("provider")
        return "done", [], 1, {}

    provider.chat = chat
    asyncio.run(agent.run(CodingTaskRequest(
        message="m", working_dir=str(running),
    )))
    assert order[:2] == ["marked", "provider"], order


def test_claude_adapter_refuses_before_any_session(tmp_path):
    """Case 1: resolvable at submit, gone by the time the adapter ran."""

    workspace = _claude_workspace(tmp_path, [_ABSENT])
    backend = _Backend()
    result = asyncio.run(_claude_agent(tmp_path, backend).run(_claude_request(workspace)))

    assert result.failure_code == _CODE
    assert result.verification_blockers == ("unit",)
    assert result.attempts == 0
    assert result.rounds_used == 0
    assert result.usage == {}
    # The backend was never entered, so there is no session and no fallback.
    assert backend.requests == []
    assert backend.signalled == 0
    assert _ABSENT not in (result.message or "")


def test_claude_adapter_late_race_keeps_every_number_honest(tmp_path, monkeypatch):
    """Case 2: the session really ran; the verifier vanished before it was used."""

    from flyto_ai.coding import checks as checks_module

    workspace = _claude_workspace(tmp_path, [sys.executable, "-c", "pass"])
    backend = _Backend(turns=3)

    async def vanished(self, checks):
        raise VerificationToolUnavailable(("unit",))

    monkeypatch.setattr(checks_module.CheckRunner, "run", vanished)
    result = asyncio.run(_claude_agent(tmp_path, backend).run(_claude_request(workspace)))
    monkeypatch.undo()

    assert result.failure_code == _CODE
    assert result.verification_blockers == ("unit",)
    assert result.ok is False
    # Honest about the work that really happened.
    assert result.attempts == 1
    assert result.rounds_used == 3
    assert result.usage == {"input_tokens": 11, "output_tokens": 22}
    assert backend.signalled == 1
    # No verdict exists, so nothing is verified and nothing may land.
    assert not result.checks


@pytest.mark.parametrize("failure", ["sandbox", "bridge"])
def test_a_pre_session_refusal_is_not_a_started_session(tmp_path, failure):
    """Case 3: the deterministic boundary keeps its own code and starts nothing."""

    from flyto_ai.agents.claude_code import (
        ActionSandboxMissing,
        ProjectActionBridgeUnavailable,
    )

    raised = (
        ActionSandboxMissing("no sandbox")
        if failure == "sandbox"
        else ProjectActionBridgeUnavailable("contract changed")
    )
    workspace = _claude_workspace(tmp_path, [sys.executable, "-c", "pass"])
    backend = _Backend(raises=raised)
    agent = _claude_agent(tmp_path, backend)
    # Watch the durable marker itself, not merely the backend's own counter:
    # what must stay false is the host's record that an implementer started.
    started = []
    agent.store.on_provider_start = lambda: started.append("started")
    result = asyncio.run(agent.run(_claude_request(workspace)))

    assert result.attempts == 0
    assert result.rounds_used == 0
    # Its own typed code: not a verification failure and not a provider failure.
    assert result.failure_code not in (_CODE, "provider_failed")
    assert result.failure_code
    # The backend was entered but never reached its provider boundary, and the
    # host was never told a session began.
    assert backend.requests and backend.signalled == 0
    assert started == [], started


def test_the_start_signal_fires_at_the_provider_and_not_before(tmp_path):
    """Case 4: the marker is written by provider work, and only by provider work."""

    workspace = _claude_workspace(tmp_path, [sys.executable, "-c", "pass"])

    # Fault *before* the signal: nothing may be recorded as started.
    early = _Backend(fail_before_signal=True)
    asyncio.run(_claude_agent(tmp_path, early).run(_claude_request(workspace)))
    assert early.signalled == 0

    # And the signal really is the seam the production backend uses.
    from flyto_ai.agents.claude_code import signal_provider_start

    written = []
    crashing = types.SimpleNamespace(on_provider_start=lambda: written.append("durable"))

    def provider_work():
        signal_provider_start(crashing)
        raise KeyboardInterrupt("killed inside the provider")

    with pytest.raises(KeyboardInterrupt):
        provider_work()
    assert written == ["durable"]
    # Idempotent across resumes and retries.
    signal_provider_start(crashing)
    assert written == ["durable"]


def test_the_real_backend_signals_immediately_before_query(tmp_path, monkeypatch):
    """The production ``_run_claude_code``, stopped at its own SDK boundary.

    The fake backend elsewhere in this module proves what the *adapter* does
    around a provider; it cannot prove where the real backend puts the signal,
    because it never runs that code. Here the real method runs and the SDK
    ``query`` is replaced by an async iterator, so the ordering is observed
    rather than asserted about the source: everything deterministic happens,
    then the marker, then the first message is awaited.
    """

    from flyto_ai.agents import claude_code as cc
    from flyto_ai.agents.models import CodeTaskRequest

    order = []

    async def fake_query(prompt=None, options=None):
        order.append("query")
        if False:  # pragma: no cover - an empty async generator
            yield None

    # The production method imports the SDK inside itself, so the stub has to
    # go on the SDK module - patching the adapter's namespace would silently do
    # nothing and let the real client run.
    import claude_agent_sdk

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _contract(workspace, [sys.executable, "-c", "pass"])

    agent = cc.ClaudeCodeAgent.__new__(cc.ClaudeCodeAgent)
    agent.on_provider_start = lambda: order.append("marked")
    agent._cc = types.SimpleNamespace(
        model="", permission_mode="acceptEdits", max_turns=4,
        verification_timeout=30, system_prompt="", allowed_tools=None,
        max_budget_usd=1.0,
    )

    request = CodeTaskRequest(
        message="do the work",
        working_dir=str(workspace),
        max_fix_attempts=1,
        max_turns=2,
        service_mode=True,
        service_edit_authority=True,
    )
    try:
        asyncio.run(agent._run_claude_code(
            request=request, indexer_context="", feedback="", session_id=None,
            max_budget=1.0, max_turns=2,
            evidence=None,
            on_stream=None,
        ))
    except Exception:
        # The method may still refuse afterwards; the ordering is the subject.
        pass
    monkeypatch.undo()

    # Both happened, and in the only order that is honest: every deterministic
    # precondition, then the durable marker, then the first provider message.
    assert order[:2] == ["marked", "query"], order


def test_the_backend_signals_through_the_hook_the_adapter_arms(tmp_path):
    """The adapter arms the seam; the backend is what decides when it fires."""

    workspace = _claude_workspace(tmp_path, [sys.executable, "-c", "pass"])
    backend = _Backend()
    agent = _claude_agent(tmp_path, backend)
    asyncio.run(agent.run(_claude_request(workspace)))

    assert callable(backend.on_provider_start)
    assert backend.signalled == 1


# --------------------------------------------------------------------------
# the durable start marker
# --------------------------------------------------------------------------


def test_the_start_marker_is_not_written_by_entering_an_adapter():
    """Recording a start that never happened makes a clean refusal look partial."""

    from flyto_ai.coding.store import mark_provider_start

    started = []
    store = types.SimpleNamespace(on_provider_start=lambda: started.append(1))

    assert started == []          # arming alone signals nothing
    mark_provider_start(store)
    assert started == [1]
    mark_provider_start(store)    # idempotent across retries and resumes
    assert started == [1]


def test_the_start_marker_survives_a_crash_inside_the_provider_call():
    """Durable *before* the call, so a worker that dies mid-call still shows it."""

    from flyto_ai.coding.store import mark_provider_start

    written = []
    store = types.SimpleNamespace(on_provider_start=lambda: written.append("durable"))

    def provider_call():
        mark_provider_start(store)
        raise KeyboardInterrupt("worker killed mid-call")

    with pytest.raises(KeyboardInterrupt):
        provider_call()

    assert written == ["durable"]


def test_a_store_without_a_hook_is_unaffected():
    from flyto_ai.coding.store import mark_provider_start

    mark_provider_start(types.SimpleNamespace())  # no hook: no error, no signal


def test_the_service_reconciles_a_start_it_was_not_told_about():
    """An adapter that proves an attempt but never signalled is still recorded."""

    from flyto_ai.coding.service import _reconcile_start_marker

    class _Progress:
        def __init__(self):
            self.implementer_started = False

        def begin(self):
            self.implementer_started = True

    progress = _Progress()
    _reconcile_start_marker(progress, CodingTaskResult(
        ok=False, message="", thread_id="t", attempts=0, status="failed",
        failure_code=_CODE,
    ))
    assert progress.implementer_started is False

    _reconcile_start_marker(progress, CodingTaskResult(
        ok=False, message="", thread_id="t", attempts=1, status="failed",
    ))
    assert progress.implementer_started is True


# --------------------------------------------------------------------------
# the real public facades
# --------------------------------------------------------------------------


def _forbidden(rendered):
    """Nothing that identifies a host path, a command, or a credential.

    Deliberately not a ban on the substring "token": bounded integer usage
    counters are named ``input_tokens``/``output_tokens`` and are exactly the
    honest numbers a late-race result has to keep. What must never appear is a
    filesystem path, an argv fragment, exception prose, or anything shaped like
    a credential.
    """

    lowered = rendered.lower()
    for marker in (
        _ABSENT, "/usr/", "/bin/", "argv", "path=", "traceback", "site-packages",
        ".flyto/coding.yaml", "api_key", "apikey", "bearer ", "password",
        "secret", "sk-", "authorization",
    ):
        assert marker not in lowered, marker


def test_the_http_facade_projects_the_typed_refusal(tmp_path):
    """The production handler's own error branch, driven directly.

    A TCP listener is not required to prove a projection, and this sandbox
    forbids binding one; what matters is that the bytes come out of the real
    ``do_POST`` error path and the real serializer, not a copy of them.
    """

    from flyto_ai.coding import http_server as http_module

    error = VerificationToolMissing(
        "a required verification tool is not installed on this host",
        (ACTION_INSTALL_VERIFICATION_TOOL,),
        ("unit",),
    )

    sent = {}

    class _Handler(http_module.CodingHTTPHandler):
        def __init__(self):  # bypass BaseHTTPRequestHandler's socket setup
            self.path = "/v1/coding/jobs"
            self.headers = {}

        def _authorized(self):
            return True

        def _read_json(self):
            return {"message": "m", "working_dir": str(tmp_path)}

        def _json(self, status, payload):
            sent["status"] = status
            sent["payload"] = payload

    handler = _Handler()
    handler.server = types.SimpleNamespace(
        tenant_id="tenant",
        coding_service=types.SimpleNamespace(
            submit=lambda *args, **kwargs: (_ for _ in ()).throw(error),
        ),
    )
    # The real production request handler, on its real error branch.
    http_module.CodingHTTPHandler.do_POST(handler)

    payload = sent["payload"]
    assert payload["ok"] is False
    assert payload["error"] == _CODE
    details = payload["details"]
    assert details["failure_phase"] == "preflight"
    assert details["retryable"] is False
    assert details["required_actions"] == [ACTION_INSTALL_VERIFICATION_TOOL]
    assert details["verification_blockers"] == ["unit"]
    _forbidden(json.dumps(payload))


def test_the_mcp_facade_projects_the_typed_refusal(tmp_path):
    """The production dispatch method, on its real CodingServiceError branch."""

    from flyto_ai.coding import mcp_server as mcp_module

    error = VerificationToolMissing(
        "a required verification tool is not installed on this host",
        (ACTION_INSTALL_VERIFICATION_TOOL,),
        ("unit", "lint"),
    )

    server = mcp_module.CodingMCPServer.__new__(mcp_module.CodingMCPServer)

    def _call(params):
        raise error

    server._call = _call
    response = server.handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "coding_submit", "arguments": {}},
    })

    body = response["result"]
    assert body["isError"] is True
    payload = body["structuredContent"]
    assert payload["ok"] is False
    assert payload["error"] == _CODE
    details = payload["details"]
    assert details["failure_phase"] == "preflight"
    assert details["retryable"] is False
    assert details["required_actions"] == [ACTION_INSTALL_VERIFICATION_TOOL]
    assert details["verification_blockers"] == ["lint", "unit"]
    _forbidden(json.dumps(response))


def test_the_receipt_projection_carries_blockers_and_nothing_else():
    """The other public shape: a finished job's receipt, not an error body."""

    receipt = CodingJobReceipt(
        job_id="job-1",
        state=CodingJobState.FAILED,
        submitted_at=1.0,
        updated_at=2.0,
        failure_code=_CODE,
        verification_blockers=("unit",),
    )
    projected = receipt_to_mapping(receipt)

    assert projected["failure_code"] == _CODE
    assert projected["failure_phase"] == "preflight"
    assert projected["retryable"] is False
    assert projected["required_actions"] == [ACTION_INSTALL_VERIFICATION_TOOL]
    assert projected["verification_blockers"] == ["unit"]
    assert projected["landable"] is False
    _forbidden(json.dumps(projected, default=str))


def test_the_receipt_reader_carries_record_blockers():
    """The durable record is where a finished job's blockers come from.

    Re-validated on the way out as well as on the way in, so a record written by
    an older build - or edited by hand - still cannot put anything but an
    identifier in front of a caller.
    """

    from flyto_ai.coding.service import CodingService

    receipt = CodingService._receipt({
        "job_id": "job-1",
        "state": CodingJobState.FAILED.value,
        "submitted_at": 1.0,
        "updated_at": 2.0,
        "failure_code": _CODE,
        "verification_blockers": ["unit", "/usr/bin/pytest", "lint"],
    })

    # The identifiers survive; the path is dropped rather than shortened.
    assert receipt.verification_blockers == ("lint", "unit")
    assert receipt_to_mapping(receipt)["verification_blockers"] == ["lint", "unit"]


def test_submit_time_refusal_does_no_work_at_all(tmp_path):
    """Zero job record, zero claim, zero route call, zero session, zero provider."""

    from flyto_ai.coding.service import CodingService

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _contract(workspace, [_ABSENT])

    built = []

    def _never(store):
        built.append(store)
        raise AssertionError("no implementer may be constructed for a refusal")

    service = CodingService(
        _never,
        state_root=str(tmp_path / "state"),
        workspace_roots=(str(workspace),),
        max_workers=1,
        max_queued=2,
    )

    baseline_state = sorted(
        p.name for p in (tmp_path / "state").rglob("*") if p.is_file()
    )

    with pytest.raises(CodingServiceError) as excinfo:
        service._require_verifiable_repository(str(workspace))

    assert excinfo.value.code == _CODE
    assert built == []
    assert not getattr(service, "_resume", {})
    assert not getattr(service, "_pending", set())
    # The service's own lock and empty index exist from construction; what must
    # not exist is any job. Compared against the state the constructor left, so
    # the assertion is about the refusal rather than about start-up.
    state_root = tmp_path / "state"
    after = sorted(p.name for p in state_root.rglob("*") if p.is_file())
    assert after == sorted(baseline_state), (after, baseline_state)
    assert not any("job" in name for name in after)

    details = error_details(excinfo.value)
    assert details["verification_blockers"] == ["unit"]
    _forbidden(json.dumps(details))


# --------------------------------------------------------------------------
# the durable job seam: submit -> worker -> record -> receipt -> facade
# --------------------------------------------------------------------------

_SETTLED = TERMINAL_CODING_JOB_STATES | {CodingJobState.AWAITING_CODEX_AUDIT}


def _worktree_is_free(service, workspace):
    """No live claim remains on the worktree.

    Read from the durable claim file the production release path writes, so
    this is the same fact a second job would consult - not an in-memory guess.
    """

    path = service._workspace_claim_path(str(workspace))
    if not path.exists():
        return True
    try:
        claim = service._read_json(path)
    except Exception:
        return False
    return not claim or not claim.get("job_id")


def _settle(service, tenant, job_id, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        receipt = service.get(tenant, job_id)
        if receipt.state in _SETTLED:
            return receipt
        time.sleep(0.02)
    raise AssertionError("the job never reached a terminal state")


def _vanishing_tool(tmp_path):
    """A real executable that exists for preflight and is removed afterwards.

    The pinned contract bytes never change, which is the point: the race being
    reproduced is a host that changed underneath an authorized job, not a
    repository that rewrote its own contract mid-flight.
    """

    tool = tmp_path / "bin" / "declared-verifier"
    tool.parent.mkdir(parents=True, exist_ok=True)
    tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return tool


def _audited_service(tmp_path, workspace, factory):
    """The production service as startup configures it for an audited Claude job.

    ``require_codex_audit=True`` is not decoration: it is what makes ``submit``
    take a durable workspace claim at all. Constructing the service without it -
    which an earlier version of this file did - meant no claim ever existed, so
    every assertion about releasing one was vacuously true and a mutation that
    disabled the release could not be caught. The backend identifier is the same
    safe public token startup passes.
    """

    from flyto_ai.coding.service import CodingService

    return CodingService(
        factory,
        state_root=str(tmp_path / "state"),
        workspace_roots=(str(workspace),),
        max_workers=1,
        max_queued=4,
        require_codex_audit=True,
        implementation_backend="claude",
    )


class _PausingClaudeBackend:
    """A controlled ``ClaudeCodeAgent`` whose round can be held open.

    The pause is the point. It lets the test observe production state at the one
    instant that matters - after ``submit`` has durably created the claim and the
    execution lease, and before the adapter has done anything - instead of racing
    an unlink against a worker and hoping.
    """

    def __init__(self, *, released, reached, turns=3, signal=True):
        self.requests = []
        self.on_provider_start = None
        self.signalled = 0
        self._released = released
        self._reached = reached
        self._turns = turns
        self._signal = signal

    async def run(self, request):
        self.requests.append(request)
        if self._signal:
            from flyto_ai.agents.claude_code import signal_provider_start

            signal_provider_start(self)
            self.signalled += 1
        return CodeTaskResponse(
            ok=True,
            message="done",
            session_id="local-1",
            attempts=1,
            claude_session_id=_SESSION,
            claude_num_turns=self._turns,
            claude_usage={"input_tokens": 11, "output_tokens": 22},
        )


def _claude_factory(tmp_path, backend, *, released, reached):
    """Build the real adapter, pausing first so the test can inspect the host.

    The factory runs inside the worker after the job, its claim and its lease
    exist, and before the adapter is entered - which is exactly the window the
    race lives in.
    """

    def factory(store):
        from flyto_ai.agents.claude_code import ClaudeCodingAgent

        reached.set()
        assert released.wait(timeout=20), "the worker was never released"
        return ClaudeCodingAgent(store, agent=backend)

    return factory


def _claim(service, workspace):
    path = service._workspace_claim_path(str(workspace))
    if not path.exists():
        return None
    return service._read_json(path)


def _assert_held(service, workspace, job_id):
    """Production state really shows this job holding the tree and the lease."""

    claim = _claim(service, workspace)
    assert claim, "no workspace claim was ever created"
    assert claim.get("job_id") == job_id, claim
    assert claim.get("claim_version"), claim
    # The production authority agrees, not just the file on disk.
    service._reassert_workspace_claim(
        service._tenant_ref("tenant"), job_id, str(workspace), claim.get("state") or "running",
    )
    # The execution lease is held by this service process.
    assert job_id in service._job_leases, "the job execution lease is not held"
    assert service._job_lease_path(job_id).exists()


def _assert_settled(service, workspace, job_id):
    """Both holds are gone, so the next job is not blocked by a dead one."""

    claim = _claim(service, workspace)
    assert not claim or not claim.get("job_id"), claim
    assert job_id not in service._job_leases, "the execution lease outlived the job"


def _audited_request(workspace):
    from flyto_ai.coding.contracts import (
        ApprovalPolicy,
        CodingTaskRequest,
        SandboxMode,
    )

    return CodingTaskRequest(
        message="do the work",
        working_dir=str(workspace),
        approval_policy=ApprovalPolicy.NEVER,
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
    )


def test_the_pre_provider_race_closes_the_job_durably(tmp_path):
    """Submit says yes, the tool disappears, and the audited job closes truthfully.

    Every assertion is read back from production state - the claim file, the
    lease registry, the durable record through the real receipt - because that
    is what an operator and an auditor actually see.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    tool = _vanishing_tool(tmp_path)
    _contract(workspace, [str(tool)])

    released, reached = threading.Event(), threading.Event()
    backend = _PausingClaudeBackend(released=released, reached=reached)
    service = _audited_service(
        tmp_path, workspace,
        _claude_factory(tmp_path, backend, released=released, reached=reached),
    )
    try:
        request = _audited_request(workspace)
        submitted = service.submit("tenant", "idem-1", request)
        assert reached.wait(timeout=20), "the worker never started the job"

        # The hold really exists at this instant, on both the tree and the job.
        _assert_held(service, workspace, submitted.job_id)
        # And it is enforced: a competing submission cannot take the worktree.
        with pytest.raises(CodingServiceError) as busy:
            service.submit("tenant", "idem-competing", request)
        assert busy.value.code == "workspace_busy"

        # The authorized verifier disappears; the contract bytes do not change.
        tool.unlink()
        released.set()
        receipt = _settle(service, "tenant", submitted.job_id)

        assert receipt.state is CodingJobState.FAILED
        assert receipt.job_terminal is True
        assert receipt.failure_code == _CODE
        phase, retryable, actions = receipt.failure_semantics
        assert (phase, retryable) == ("preflight", False)
        assert actions == (ACTION_INSTALL_VERIFICATION_TOOL,)
        assert receipt.verification_blockers == ("unit",)
        assert receipt.implementer_started is False
        assert receipt.landable is False
        assert receipt.implementation_revision_sha256 == ""
        assert receipt.result is not None
        assert receipt.result.attempts == 0
        assert receipt.result.rounds_used == 0
        # The selected Claude backend was never entered, and nothing else ran
        # in its place: this service has no native factory at all.
        assert backend.requests == []
        assert backend.signalled == 0

        # Both holds are released by the terminal transition.
        _assert_settled(service, workspace, submitted.job_id)

        # Replay is idempotent and keeps the same terminal evidence.
        again = service.get("tenant", submitted.job_id)
        assert (again.state, again.failure_code) == (receipt.state, receipt.failure_code)
        assert again.verification_blockers == receipt.verification_blockers
        assert service.submit("tenant", "idem-1", request).job_id == submitted.job_id

        # And the worktree is genuinely free for the next job.
        tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
        released.set()
        follow_on = service.submit("tenant", "idem-follow", request)
        assert follow_on.job_id != submitted.job_id
        _settle(service, "tenant", follow_on.job_id)

        projected = receipt_to_mapping(receipt)
        assert projected["failure_code"] == _CODE
        assert projected["failure_phase"] == "preflight"
        assert projected["retryable"] is False
        assert projected["required_actions"] == [ACTION_INSTALL_VERIFICATION_TOOL]
        assert projected["verification_blockers"] == ["unit"]
        assert projected["landable"] is False
        _forbidden(json.dumps(projected, default=str))
    finally:
        released.set()
        service.close()


def test_the_post_provider_race_closes_the_job_durably(tmp_path, monkeypatch):
    """The Claude session really ran, and the round still may not land."""

    from flyto_ai.coding import checks as checks_module

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _contract(workspace, [sys.executable, "-c", "pass"])

    released, reached = threading.Event(), threading.Event()
    backend = _PausingClaudeBackend(released=released, reached=reached, turns=3)
    service = _audited_service(
        tmp_path, workspace,
        _claude_factory(tmp_path, backend, released=released, reached=reached),
    )

    async def vanished(self, checks):
        raise VerificationToolUnavailable(("unit",))

    try:
        request = _audited_request(workspace)
        monkeypatch.setattr(checks_module.CheckRunner, "run", vanished)
        submitted = service.submit("tenant", "idem-2", request)
        assert reached.wait(timeout=20), "the worker never started the job"

        _assert_held(service, workspace, submitted.job_id)
        with pytest.raises(CodingServiceError) as busy:
            service.submit("tenant", "idem-competing", request)
        assert busy.value.code == "workspace_busy"

        released.set()
        receipt = _settle(service, "tenant", submitted.job_id)
        monkeypatch.undo()

        assert receipt.state is CodingJobState.FAILED
        assert receipt.job_terminal is True
        assert receipt.failure_code == _CODE
        assert receipt.verification_blockers == ("unit",)
        # The session really happened, and the record says so honestly.
        assert backend.requests and backend.signalled == 1
        assert receipt.implementer_started is True
        assert receipt.result is not None
        assert receipt.result.attempts == 1
        assert receipt.result.rounds_used == 3
        assert receipt.result.usage == {"input_tokens": 11, "output_tokens": 22}
        # Nothing verified it, so there is no revision and nothing to accept.
        assert receipt.implementation_revision_sha256 == ""
        assert receipt.landable is False
        assert not receipt.result.checks

        _assert_settled(service, workspace, submitted.job_id)

        again = service.get("tenant", submitted.job_id)
        assert (again.state, again.failure_code) == (receipt.state, receipt.failure_code)
        assert again.result.rounds_used == receipt.result.rounds_used

        released.set()
        follow_on = service.submit("tenant", "idem-follow-2", request)
        assert follow_on.job_id != submitted.job_id
        _settle(service, "tenant", follow_on.job_id)

        projected = receipt_to_mapping(receipt)
        assert projected["failure_phase"] == "preflight"
        assert projected["verification_blockers"] == ["unit"]
        assert projected["landable"] is False
        _forbidden(json.dumps(projected, default=str))
    finally:
        released.set()
        service.close()
