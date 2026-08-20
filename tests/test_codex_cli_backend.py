from __future__ import annotations

import argparse
import asyncio
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from flyto_ai.agents.codex_cli import CodexCliCodingAgent
from flyto_ai.coding import CodingTaskRequest
from flyto_ai.coding.store import ThreadStore


SESSION = "019ff5bd-cf74-7d81-aa3c-26825ce18e14"


def _fake_codex(tmp_path: Path) -> Path:
    executable = tmp_path / "codex-fake"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys, time\n"
        "prompt = sys.stdin.read()\n"
        "root = pathlib.Path.cwd()\n"
        "(root / 'codex-invocation.json').write_text(json.dumps({\n"
        "    'argv': sys.argv[1:], 'prompt': prompt,\n"
        "}, sort_keys=True), encoding='utf-8')\n"
        "print(json.dumps({'type': 'thread.started', 'thread_id': "
        + repr(SESSION)
        + "}), flush=True)\n"
        "if 'INVALID_OUTPUT' in prompt:\n"
        "    print('not-json', flush=True)\n"
        "    print(json.dumps({'type': 'turn.completed', 'usage': {}}), flush=True)\n"
        "    raise SystemExit(0)\n"
        "if 'BIG_OUTPUT' in prompt:\n"
        "    print(json.dumps({'type': 'item.completed', 'item': {\n"
        "        'type': 'agent_message', 'text': 'x' * 512,\n"
        "    }}), flush=True)\n"
        "if 'TIMEOUT' in prompt:\n"
        "    time.sleep(2)\n"
        "if 'AUTH_FAIL' in prompt:\n"
        "    print('unauthorized: not logged in', file=sys.stderr, flush=True)\n"
        "    raise SystemExit(1)\n"
        "if 'EXIT_BEFORE_COMPLETE' in prompt:\n"
        "    raise SystemExit(1)\n"
        "(root / 'result.txt').write_text('verified\\n', encoding='utf-8')\n"
        "print(json.dumps({'type': 'turn.started'}), flush=True)\n"
        "print(json.dumps({'type': 'item.completed', 'item': {\n"
        "    'type': 'agent_message', 'text': 'implemented in ' + str(root),\n"
        "}}), flush=True)\n"
        "print(json.dumps({'type': 'turn.completed', 'usage': {\n"
        "    'input_tokens': 12, 'output_tokens': 3, 'bad-value': 'drop',\n"
        "}}), flush=True)\n"
        "if 'EXIT_AFTER_COMPLETE' in prompt:\n"
        "    raise SystemExit(1)\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def _workspace(tmp_path: Path, *, required_capability: bool = False) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir()
    capability = ""
    if required_capability:
        capability = (
            "capabilities:\n"
            "  - name: unavailable\n"
            "    kind: command\n"
            "    argv: [python, --version]\n"
            "    required: true\n"
        )
    config.write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: result\n"
        "    argv:\n"
        "      - " + json.dumps(sys.executable) + "\n"
        "      - -c\n"
        "      - \"from pathlib import Path; assert Path('result.txt').read_text() == 'verified\\\\n'\"\n"
        "    required: true\n"
        + capability,
        encoding="utf-8",
    )
    return workspace


def _agent(tmp_path: Path, executable: Path) -> tuple[CodexCliCodingAgent, ThreadStore]:
    store = ThreadStore(str(tmp_path / "threads"))
    return (
        CodexCliCodingAgent(
            store,
            executable=str(executable),
            model="gpt-5.6-sol",
        ),
        store,
    )


def test_codex_cli_round_is_sandboxed_host_verified_and_attributed(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    workspace = _workspace(tmp_path)
    agent, store = _agent(tmp_path, executable)
    starts: list[str] = []
    sessions: list[str] = []
    store.on_provider_start = lambda: starts.append("started")
    store.on_provider_session = sessions.append

    result = asyncio.run(agent.run(CodingTaskRequest(
        message="implement the bounded change",
        working_dir=str(workspace),
    )))

    assert result.ok is True
    assert result.thread_id == SESSION
    assert starts == ["started"]
    assert sessions == [SESSION]
    assert result.files_changed == ["codex-invocation.json", "result.txt"]
    assert result.checks and result.checks[0].passed is True
    assert result.usage == {"input_tokens": 12, "output_tokens": 3}
    assert str(workspace) not in result.message
    assert "<workspace>" in result.message

    invocation = json.loads((workspace / "codex-invocation.json").read_text())
    argv = invocation["argv"]
    assert "--ignore-user-config" in argv
    assert "--ignore-rules" in argv
    assert "--strict-config" in argv
    assert 'web_search="disabled"' in argv
    assert all(
        feature in argv
        for feature in (
            "plugins", "apps", "standalone_web_search", "search_tool",
            "browser_use", "computer_use", "multi_agent", "enable_fanout",
        )
    )
    assert "--sandbox" in argv and "workspace-write" in argv
    assert "--cd" in argv and str(workspace) in argv
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert "implement the bounded change" not in argv
    assert "implement the bounded change" in invocation["prompt"]
    assert "Never call flyto_coding" in invocation["prompt"]


def test_codex_cli_rework_resumes_the_exact_provider_session(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    workspace = _workspace(tmp_path)
    agent, _store = _agent(tmp_path, executable)
    first = asyncio.run(agent.run(CodingTaskRequest(
        message="first round",
        working_dir=str(workspace),
    )))
    assert first.ok is True

    resumed = asyncio.run(agent.run(CodingTaskRequest(
        message="apply audit findings",
        working_dir=str(workspace),
        thread_id=SESSION,
        resume=True,
    )))
    assert resumed.ok is True
    assert resumed.thread_id == SESSION
    argv = json.loads(
        (workspace / "codex-invocation.json").read_text(encoding="utf-8"),
    )["argv"]
    assert argv[0] == "exec" or "resume" in argv
    resume_index = argv.index("resume")
    assert argv.index("--sandbox") < resume_index
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"
    assert argv.index("--cd") < resume_index
    assert argv[argv.index("--cd") + 1] == str(workspace)
    assert SESSION in argv
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv


def test_codex_cli_initial_and_resume_argv_are_not_ephemeral(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    workspace = _workspace(tmp_path)
    agent, _store = _agent(tmp_path, executable)

    first = asyncio.run(agent.run(CodingTaskRequest(
        message="first round",
        working_dir=str(workspace),
    )))
    assert first.ok is True
    initial_argv = json.loads(
        (workspace / "codex-invocation.json").read_text(encoding="utf-8"),
    )["argv"]

    resumed = asyncio.run(agent.run(CodingTaskRequest(
        message="apply audit findings",
        working_dir=str(workspace),
        thread_id=SESSION,
        resume=True,
    )))
    assert resumed.ok is True
    resume_argv = json.loads(
        (workspace / "codex-invocation.json").read_text(encoding="utf-8"),
    )["argv"]

    assert "--ephemeral" not in initial_argv
    assert "--ephemeral" not in resume_argv


def test_required_capability_refuses_before_codex_process_start(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    workspace = _workspace(tmp_path, required_capability=True)
    agent, store = _agent(tmp_path, executable)
    starts: list[str] = []
    store.on_provider_start = lambda: starts.append("started")

    result = asyncio.run(agent.run(CodingTaskRequest(
        message="must not start",
        working_dir=str(workspace),
    )))

    assert result.ok is False
    assert result.failure_code == "required_capability_unavailable"
    assert starts == []
    assert not (workspace / "codex-invocation.json").exists()


def test_codex_provider_failure_is_closed_and_sanitized(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    workspace = _workspace(tmp_path)
    agent, _store = _agent(tmp_path, executable)

    result = asyncio.run(agent.run(CodingTaskRequest(
        message="AUTH_FAIL",
        working_dir=str(workspace),
    )))

    assert result.ok is False
    assert result.failure_code == "provider_auth_failed"
    assert "unauthorized" not in result.message.lower()
    assert result.thread_id == SESSION


def test_nonzero_exit_after_completed_turn_keeps_host_verified_result(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    workspace = _workspace(tmp_path)
    agent, store = _agent(tmp_path, executable)

    result = asyncio.run(agent.run(CodingTaskRequest(
        message="EXIT_AFTER_COMPLETE",
        working_dir=str(workspace),
    )))

    assert result.ok is True
    assert result.failure_code is None
    assert result.checks and result.checks[0].passed is True
    events = [
        json.loads(line)
        for line in Path(store.evidence_path(SESSION)).read_text().splitlines()
    ]
    round_event = next(event for event in events if event["type"] == "coding.round")
    assert round_event["data"]["provider_exit_code"] == 1
    assert round_event["data"]["turn_completed"] is True
    assert round_event["data"]["completed_with_nonzero_exit"] is True


def test_nonzero_exit_without_completed_turn_stays_failed(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    workspace = _workspace(tmp_path)
    agent, _store = _agent(tmp_path, executable)

    result = asyncio.run(agent.run(CodingTaskRequest(
        message="EXIT_BEFORE_COMPLETE",
        working_dir=str(workspace),
    )))

    assert result.ok is False
    assert result.failure_code == "provider_failed"


def test_resume_refuses_a_changed_provider_session(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    workspace = _workspace(tmp_path)
    agent, _store = _agent(tmp_path, executable)

    result = asyncio.run(agent.run(CodingTaskRequest(
        message="resume mismatch",
        working_dir=str(workspace),
        thread_id="different-session",
        resume=True,
    )))

    assert result.ok is False
    assert result.failure_code == "session_binding_failed"


def test_read_only_child_mutation_is_detected_from_host_snapshot(tmp_path: Path) -> None:
    from flyto_ai.coding import SandboxMode

    executable = _fake_codex(tmp_path)
    workspace = _workspace(tmp_path)
    agent, _store = _agent(tmp_path, executable)

    result = asyncio.run(agent.run(CodingTaskRequest(
        message="attempt read-only mutation",
        working_dir=str(workspace),
        sandbox_mode=SandboxMode.READ_ONLY,
        require_changes=False,
    )))

    assert result.ok is False
    assert result.failure_code == "unexpected_workspace_change"


def test_invalid_or_oversized_jsonl_never_becomes_success(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import flyto_ai.agents.codex_cli as codex_cli

    executable = _fake_codex(tmp_path)
    for index, (message, limit) in enumerate((
        ("INVALID_OUTPUT", codex_cli.MAX_STREAM_BYTES),
        ("BIG_OUTPUT", 128),
    )):
        case = tmp_path / "case-{}".format(index)
        case.mkdir()
        workspace = _workspace(case)
        agent, _store = _agent(case, executable)
        monkeypatch.setattr(codex_cli, "MAX_STREAM_BYTES", limit)
        result = asyncio.run(agent.run(CodingTaskRequest(
            message=message,
            working_dir=str(workspace),
        )))
        assert result.ok is False
        assert result.failure_code == "provider_failed"


def test_provider_session_hook_failure_is_terminal(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    workspace = _workspace(tmp_path)
    agent, store = _agent(tmp_path, executable)

    def refuse(_session: str) -> None:
        raise RuntimeError("host refused binding")

    store.on_provider_session = refuse
    result = asyncio.run(agent.run(CodingTaskRequest(
        message="binding must fail closed",
        working_dir=str(workspace),
    )))
    assert result.ok is False
    assert result.failure_code == "session_binding_failed"


def test_host_timeout_kills_the_codex_process_group(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    workspace = _workspace(tmp_path)
    agent, _store = _agent(tmp_path, executable)
    # Constructor bounds production configuration to at least 30 seconds. The
    # focused test narrows the already-validated instance to keep the proof fast.
    agent.timeout_seconds = 0.5

    result = asyncio.run(agent.run(CodingTaskRequest(
        message="TIMEOUT",
        working_dir=str(workspace),
    )))
    assert result.ok is False
    assert result.failure_code == "provider_failed"


def test_codex_backend_does_not_inherit_provider_or_ci_secrets(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-cross")
    environment = CodexCliCodingAgent._environment()
    assert "OPENAI_API_KEY" not in environment
    assert "GITHUB_TOKEN" not in environment
    assert environment.get("HOME") == os.environ.get("HOME")


def _service_args(tmp_path: Path, executable: Path) -> argparse.Namespace:
    return argparse.Namespace(
        tenant="tenant-codex",
        workspace_root=[str(tmp_path)],
        state_dir=str(tmp_path / "state"),
        provider=None,
        model="gpt-5.6-sol",
        base_url=None,
        codex_command=str(executable),
        config=".flyto/coding.yaml",
        approval="never",
        sandbox="workspace-write",
        sandbox_image="python:3.12-slim",
        max_workers=1,
        max_queued=4,
        implementation_backend="codex",
        max_rework_rounds=3,
        emergency_overflow_backend=None,
        emergency_overflow_threshold=1,
        indexer_command=None,
        blueprint_command=None,
    )


def test_cli_selects_codex_once_and_keeps_independent_audit(tmp_path: Path, monkeypatch) -> None:
    import flyto_ai.cli as cli

    executable = _fake_codex(tmp_path)

    def forbidden(_args):
        raise AssertionError("codex backend must not resolve a native provider")

    monkeypatch.setattr(cli, "_create_native_coding_provider", forbidden)
    service = cli._build_coding_service(_service_args(tmp_path, executable))
    try:
        assert service.require_codex_audit is True
        assert service.implementation_backend == "codex"
        agent = service.agent_factory(ThreadStore(str(tmp_path / "thread-store")))
        assert isinstance(agent, CodexCliCodingAgent)
        assert agent.model == "gpt-5.6-sol"
    finally:
        service.close()


@pytest.mark.parametrize("model", [None, "", "bad model", "x" * 129])
def test_codex_backend_refuses_missing_or_unbounded_model(
    tmp_path: Path,
    model: object,
) -> None:
    executable = _fake_codex(tmp_path)
    with pytest.raises(ValueError, match="bounded --model"):
        CodexCliCodingAgent(
            ThreadStore(str(tmp_path / "threads")),
            executable=str(executable),
            model=model,  # type: ignore[arg-type]
        )
