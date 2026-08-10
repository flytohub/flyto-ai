# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""The job's contract authority must survive the trip into the SDK layer.

The service captured `authorized_config_sha256` correctly and the action bridge
enforced it correctly, and the whole thing was still bypassable: the adapter
built its `CodeTaskRequest` without carrying the field, so the SDK layer saw the
empty default and re-authorized whatever `.flyto/coding.yaml` happened to say at
that moment. Two green components, one broken seam - which is why every test
here asserts on the *boundary* rather than on either side of it.

The second half is what the gate is worth. A round whose contract changed must
refuse *before* the provider is contacted, on both backends, whether or not the
repository declares actions and whether or not the round may edit. Each negative
test therefore records provider invocations and asserts the count is zero.
"""
import asyncio
import json
import sys
import textwrap
from pathlib import Path

import pytest

from flyto_ai.agents.claude_code import (
    ClaudeCodingAgent,
    ProjectActionBridgeUnavailable,
)
from flyto_ai.agents.models import CodeTaskResponse
from flyto_ai.coding.checks import config_digest
from flyto_ai.coding.contracts import (
    VERIFICATION_CONTRACT_CHANGED,
    ApprovalPolicy,
    CodingTaskRequest,
    SandboxMode,
)
from flyto_ai.coding.store import ThreadStore

_SESSION = "sdk-session-authority"


@pytest.fixture(autouse=True)
def action_sandbox_available(monkeypatch):
    """Isolate these tests from sandbox availability.

    They are about *contract authority*, so the action boundary must not be the
    thing that fails first: a host without a container runtime would otherwise
    report `action_sandbox_unavailable` and hide whether the digest gate works
    at all. The precedence itself (digest before sandbox) is asserted in
    `test_coding_action_sandbox.py`.
    """

    from flyto_ai.coding.actions import ProjectActionExecutor

    def resolve(self):
        # A backend is only usable once it has resolved an immutable image
        # identity, so a stand-in must supply one too.
        self._image_id = "sha256:" + "1a" * 32
        return "docker"

    monkeypatch.setattr(ProjectActionExecutor, "_detect_backend", resolve)
    original = ProjectActionExecutor.__init__

    def patched(self, workspace, config_path=".flyto/coding.yaml",
                *, sandbox_image="flyto-test-action-image:pinned"):
        original(self, workspace, config_path,
                 sandbox_image=sandbox_image or "flyto-test-action-image:pinned")

    monkeypatch.setattr(ProjectActionExecutor, "__init__", patched)



class RecordingClaudeAgent:
    """Stands in for `ClaudeCodeAgent`, recording exactly what it was handed.

    It is the assertion target for the seam: whatever the adapter fails to put
    on the request simply is not here.
    """

    def __init__(self, workspace: Path, *, writes: bool = True) -> None:
        self.workspace = workspace
        self.writes = writes
        self.requests: list = []

    async def run(self, request):
        self.requests.append(request)
        if self.writes:
            (self.workspace / "result.txt").write_text("verified\n")
        return CodeTaskResponse(
            ok=True,
            message="done",
            session_id="local-{}".format(len(self.requests)),
            attempts=1,
            claude_session_id=_SESSION,
            claude_num_turns=1,
            claude_usage={"input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0, "ok": True},
        )


def _contract(workspace: Path, body: str) -> Path:
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("version: flyto.coding-config.v1\n" + textwrap.dedent(body))
    return config


_TRIVIAL_CHECK = """checks:
  - name: trivial
    argv: {argv}
    required: true
"""


def _workspace(tmp_path: Path, *, actions: str = "", marker: str = "one") -> Path:
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    body = _TRIVIAL_CHECK.format(
        argv=json.dumps([sys.executable, "-c", "pass  # {}".format(marker)]),
    )
    _contract(workspace, body + actions)
    return workspace


def _request(workspace: Path, digest: str, **overrides) -> CodingTaskRequest:
    fields = {
        "message": "do the work",
        "working_dir": str(workspace),
        "approval_policy": ApprovalPolicy.NEVER,
        "sandbox_mode": SandboxMode.WORKSPACE_WRITE,
        "authorized_config_sha256": digest,
    }
    fields.update(overrides)
    return CodingTaskRequest(**fields)


def _run(agent, request):
    return asyncio.run(agent.run(request))


# --------------------------------------------------------------------------
# the seam: what actually reaches the SDK layer
# --------------------------------------------------------------------------


def test_the_adapter_carries_the_config_path_and_job_digest(tmp_path):
    """The exact bug: both host-owned fields must arrive, with exact values."""

    workspace = _workspace(tmp_path)
    digest = config_digest(str(workspace))
    backend = RecordingClaudeAgent(workspace)
    agent = ClaudeCodingAgent(ThreadStore(str(tmp_path / "threads")), agent=backend)

    result = _run(agent, _request(workspace, digest))
    assert result.failure_code != VERIFICATION_CONTRACT_CHANGED
    assert len(backend.requests) == 1

    carried = backend.requests[0]
    assert carried.authorized_config_sha256 == digest, "the job digest was dropped"
    assert carried.config_path == ".flyto/coding.yaml"
    assert carried.service_mode is True
    assert carried.service_edit_authority is True


def test_a_non_default_config_path_is_carried_too(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "custom").mkdir(parents=True)
    (workspace / "custom" / "contract.yaml").write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n  - name: trivial\n    argv: {}\n    required: true\n".format(
            json.dumps([sys.executable, "-c", "pass"]),
        ),
    )
    relative = "custom/contract.yaml"
    digest = config_digest(str(workspace), relative)
    backend = RecordingClaudeAgent(workspace)
    agent = ClaudeCodingAgent(ThreadStore(str(tmp_path / "threads")), agent=backend)

    _run(agent, _request(workspace, digest, config_path=relative))

    assert backend.requests[0].config_path == relative
    assert backend.requests[0].authorized_config_sha256 == digest


def test_an_unauthorized_job_carries_an_empty_digest_not_a_forged_one(tmp_path):
    """No authority established means none is invented at the seam."""

    workspace = _workspace(tmp_path)
    backend = RecordingClaudeAgent(workspace)
    agent = ClaudeCodingAgent(ThreadStore(str(tmp_path / "threads")), agent=backend)

    _run(agent, _request(workspace, ""))
    assert backend.requests[0].authorized_config_sha256 == ""


# --------------------------------------------------------------------------
# a changed contract refuses before the provider is contacted
# --------------------------------------------------------------------------


@pytest.mark.parametrize("declares_actions", [False, True])
@pytest.mark.parametrize("edit_authority", [True, False])
def test_a_changed_contract_refuses_before_any_provider_call(
    tmp_path, declares_actions, edit_authority,
):
    """Round one edits the contract; round two must not reach the model.

    Parametrized across both axes the review named: a repository with no
    declared actions is gated by the same rule as one with them, and a
    read-only round is gated too - the contract decides verification, and
    verification is not something only writable rounds care about.
    """

    actions = ""
    if declares_actions:
        actions = "actions:\n  - {}\n".format(json.dumps({
            "name": "regenerate",
            "argv": [sys.executable, "-c", "pass"],
            "timeout_seconds": 30,
        }))
    workspace = _workspace(tmp_path, actions=actions)
    job_digest = config_digest(str(workspace))

    backend = RecordingClaudeAgent(workspace, writes=edit_authority)
    agent = ClaudeCodingAgent(ThreadStore(str(tmp_path / "threads")), agent=backend)

    sandbox = (
        SandboxMode.WORKSPACE_WRITE if edit_authority else SandboxMode.READ_ONLY
    )
    first = _run(agent, _request(
        workspace, job_digest, sandbox_mode=sandbox, require_changes=edit_authority,
    ))
    assert first.failure_code != VERIFICATION_CONTRACT_CHANGED
    invocations_after_round_one = len(backend.requests)
    assert invocations_after_round_one == 1

    # The contract moves under the running job.
    _workspace(tmp_path, actions=actions, marker="two")
    assert config_digest(str(workspace)) != job_digest

    second = _run(agent, _request(
        workspace, job_digest, sandbox_mode=sandbox, require_changes=edit_authority,
    ))

    assert second.ok is False
    assert second.failure_code == VERIFICATION_CONTRACT_CHANGED
    # The whole point: the model was never contacted for round two.
    assert len(backend.requests) == invocations_after_round_one, (
        "the provider was invoked despite a contract change"
    )


def test_a_changed_contract_cannot_weaken_the_required_checks(tmp_path):
    """Deleting the required check is exactly the escalation being blocked."""

    workspace = _workspace(tmp_path)
    job_digest = config_digest(str(workspace))
    backend = RecordingClaudeAgent(workspace)
    agent = ClaudeCodingAgent(ThreadStore(str(tmp_path / "threads")), agent=backend)

    # Round one: the contract still requires a check.
    _run(agent, _request(workspace, job_digest))
    assert len(backend.requests) == 1

    # Round two would like there to be no required checks at all.
    _contract(workspace, """
        checks:
          - name: toothless
            argv: {}
            required: false
        """.format(json.dumps([sys.executable, "-c", "pass"])))

    result = _run(agent, _request(workspace, job_digest))

    # Refused as a contract change, *not* as `verification_required`: the
    # difference matters because the second would read as "this repository
    # declares nothing", rewarding the edit with a softer diagnosis.
    assert result.failure_code == VERIFICATION_CONTRACT_CHANGED
    assert len(backend.requests) == 1


def test_the_native_backend_is_gated_by_the_same_rule(tmp_path):
    """Provider-neutral: the rule lives in the round, not in one adapter."""

    from flyto_ai.coding.agent import FlytoCodingAgent

    workspace = _workspace(tmp_path)
    job_digest = config_digest(str(workspace))

    calls = []

    class RecordingProvider:
        async def chat(self, **kwargs):
            calls.append(kwargs)
            raise AssertionError("the provider must not be reached")

    agent = FlytoCodingAgent(
        RecordingProvider(), store=ThreadStore(str(tmp_path / "threads")),
    )
    _workspace(tmp_path, marker="changed")
    assert config_digest(str(workspace)) != job_digest

    result = asyncio.run(agent.run(_request(workspace, job_digest)))

    assert result.ok is False
    assert result.failure_code == VERIFICATION_CONTRACT_CHANGED
    assert calls == [], "the native provider was invoked despite a contract change"


# --------------------------------------------------------------------------
# a bridge failure is a control-plane refusal, never a provider failure
# --------------------------------------------------------------------------


def test_a_bridge_failure_is_not_reported_as_a_provider_failure(tmp_path):
    """`provider_failed` for a contract substitution blamed the wrong party."""

    workspace = _workspace(tmp_path, actions="actions:\n  - {}\n".format(json.dumps({
        "name": "regenerate",
        "argv": [sys.executable, "-c", "pass"],
        "timeout_seconds": 30,
    })))
    digest = config_digest(str(workspace))

    class BridgeFailingAgent(RecordingClaudeAgent):
        async def run(self, request):
            self.requests.append(request)
            raise ProjectActionBridgeUnavailable(
                "the repository contract changed after this job was authorized",
            )

    backend = BridgeFailingAgent(workspace)
    agent = ClaudeCodingAgent(ThreadStore(str(tmp_path / "threads")), agent=backend)

    result = _run(agent, _request(workspace, digest))

    assert result.ok is False
    assert result.failure_code == VERIFICATION_CONTRACT_CHANGED
    assert result.failure_code != "provider_failed"

    # No prose, path, argv or environment value crosses the boundary.
    rendered = json.dumps(
        {"message": result.message, "failure_code": result.failure_code},
    )
    for leak in (str(tmp_path), str(workspace), sys.executable, "regenerate",
                 "PATH", "authorized"):
        assert leak not in rendered, leak


def test_the_contract_changed_code_has_typed_non_provider_semantics():
    from flyto_ai.coding.contracts import (
        ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,
        CodingJobReceipt,
        CodingJobState,
    )
    from flyto_ai.coding.service import receipt_to_mapping

    receipt = CodingJobReceipt(
        job_id="job_" + "a1b2c3d4" * 3,
        state=CodingJobState.FAILED,
        submitted_at=1.0,
        updated_at=2.0,
        failure_code=VERIFICATION_CONTRACT_CHANGED,
    )
    projected = receipt_to_mapping(receipt)

    # Preflight phase, because that is what it is: a feasibility fact the host
    # established before the model, arriving late only because the file moved.
    assert projected["failure_phase"] == "preflight"
    assert projected["failure_phase"] != "provider"
    assert projected["retryable"] is False
    assert projected["required_actions"] == [ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT]
    assert projected["job_terminal"] is True


def test_a_contract_change_is_not_auditable_and_never_opens_rework():
    """It is not real attributable work, so it must not hold a job open."""

    from flyto_ai.coding.service import AUDITABLE_IMPLEMENTATION_FAILURE_CODES

    assert VERIFICATION_CONTRACT_CHANGED not in AUDITABLE_IMPLEMENTATION_FAILURE_CODES
