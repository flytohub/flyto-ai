# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Closed-loop tests for the native provider-neutral coding control plane."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from flyto_ai.coding import (
    ApprovalPolicy,
    CapabilityManager,
    CapabilitySpec,
    CheckRunner,
    CheckSpec,
    CodingTaskRequest,
    FlytoCodingAgent,
    SandboxMode,
    ThreadStore,
    WorkspaceTools,
    WorkspaceViolation,
    load_project_config,
)
from flyto_ai.coding.store import redact_evidence
from flyto_ai.providers.base import LLMProvider


class ScriptedProvider(LLMProvider):
    """Deterministic provider contract double; filesystem/checks remain real."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = 0

    async def chat(
        self, messages, system_prompt, tools, dispatch_fn, max_rounds=30, on_stream=None,
    ):
        action = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        logs = []
        for name, args in action:
            result = await dispatch_fn(name, args)
            logs.append({"function": name, "ok": result.get("ok")})
        return "native flyto attempt {}".format(self.calls), logs, len(action), {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        }


def run(coro):
    return asyncio.run(coro)


def test_contract_is_versioned_and_resume_requires_thread(tmp_path):
    with pytest.raises(ValueError, match="resume requires"):
        CodingTaskRequest(message="fix", working_dir=str(tmp_path), resume=True)
    request = CodingTaskRequest(message="fix", working_dir=str(tmp_path))
    assert request.sandbox_mode == SandboxMode.WORKSPACE_WRITE
    assert request.approval_policy == ApprovalPolicy.NEVER
    assert request.max_rounds == 100


def test_thread_store_is_append_only_redacted_and_workspace_bound(tmp_path):
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    workspace.mkdir()
    store = ThreadStore(str(state))
    metadata = store.create(str(workspace), "thread_one")
    store.append("thread_one", "conversation.message", {
        "role": "user", "content": "api_key=super-secret-value Bearer abcdefghijklmnop",
    })
    raw = Path(store.evidence_path("thread_one")).read_text()
    assert "super-secret-value" not in raw
    assert "abcdefghijklmnop" not in raw
    assert "***" in raw
    assert store.load("thread_one", str(workspace))["thread_id"] == metadata["thread_id"]
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="different workspace"):
        store.load("thread_one", str(other))


def test_evidence_preserves_usage_counts_but_not_credentials():
    projected = redact_evidence({
        "prompt_tokens": 123,
        "completion_tokens": 45,
        "total_tokens": 168,
        "token": "credential-token-value",
        "access_token": "credential-access-value",
        "password": "credential-password-value",
        "nested": {"output_tokens": 9, "api_key": "credential-key-value"},
    })
    assert projected["prompt_tokens"] == 123
    assert projected["completion_tokens"] == 45
    assert projected["total_tokens"] == 168
    assert projected["nested"]["output_tokens"] == 9
    assert projected["token"] == "***"
    assert projected["access_token"] == "***"
    assert projected["password"] == "***"
    assert projected["nested"]["api_key"] == "***"


def test_workspace_blocks_traversal_absolute_and_symlink_escape(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    tools = WorkspaceTools(str(workspace))
    with pytest.raises(WorkspaceViolation):
        tools.resolve("../outside/secret.txt")
    with pytest.raises(WorkspaceViolation):
        tools.resolve(str(outside / "secret.txt"))
    with pytest.raises(WorkspaceViolation):
        tools.resolve("escape/secret.txt")


def test_workspace_exact_replace_and_snapshot_preserve_existing_dirty_state(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("value = 1\n")
    tools = WorkspaceTools(str(tmp_path))
    before = tools.snapshot()
    result = tools.replace_text("app.py", "value = 1", "value = 2")
    after = tools.snapshot()
    assert result["ok"] is True
    assert path.read_text() == "value = 2\n"
    assert tools.changed_since(before, after) == ["app.py"]
    with pytest.raises(WorkspaceViolation, match="exactly once"):
        tools.replace_text("app.py", "missing", "x")


def test_workspace_search_contract_is_literal_and_guides_empty_results(
    tmp_path, monkeypatch,
):
    """The literal-search contract, decided by argv rather than by the host.

    Only the process boundary is faked. The argv under assertion is the one
    production builds, so the fake cannot pass while `--fixed-strings` or the
    credential globs are missing, and the ripgrep precondition is still
    exercised against the real code path.
    """

    from flyto_ai.coding import workspace as workspace_module

    (tmp_path / "service.py").write_text("'count': len(ordered),\n")
    tools = WorkspaceTools(str(tmp_path))
    search_tool = next(
        definition
        for definition in tools.definitions
        if definition["name"] == "coding_search"
    )

    observed = []
    monkeypatch.setattr(
        workspace_module.shutil, "which", lambda name: "/usr/bin/{}".format(name),
    )

    async def fake_run_process(self, argv, timeout_seconds, *, model_command=False):
        observed.append(list(argv))
        assert argv[0] == "rg"
        assert "--fixed-strings" in argv, "coding_search must stay a literal search"
        for glob in ("!.env", "!**/.git/**", "!**/.ssh/**", "!**/.aws/**"):
            assert glob in argv, "coding_search must keep its credential globs"
        separator = argv.index("--")
        query = argv[separator + 1]
        target = Path(argv[separator + 2])
        # A literal matcher, exactly as `--fixed-strings` promises.
        lines = [
            "{}:{}:{}".format(target.name, number, text)
            for number, text in enumerate(target.read_text().splitlines(), start=1)
            if query in text
        ]
        return {
            "ok": bool(lines), "exit_code": 0 if lines else 1, "timed_out": False,
            "output": "\n".join(lines),
        }

    monkeypatch.setattr(WorkspaceTools, "_run_process", fake_run_process)

    assert "literal" in search_tool["description"].lower()
    missing = run(tools.search(r"count: \\d+", "service.py"))
    assert missing == {
        "ok": True,
        "matches": [],
        "truncated": False,
        "query_mode": "literal",
        "next_action": (
            "No literal matches. Read the current file before retrying an edit; "
            "verification output may contain runtime values that are not source text."
        ),
    }
    found = run(tools.search("len(ordered)", "service.py"))
    assert found["query_mode"] == "literal"
    assert len(found["matches"]) == 1
    assert "next_action" not in found
    assert observed and all("--fixed-strings" in argv for argv in observed)

    # Faking the boundary must not soften the precondition in front of it.
    monkeypatch.setattr(workspace_module.shutil, "which", lambda name: None)
    with pytest.raises(WorkspaceViolation, match="ripgrep"):
        run(tools.search("len(ordered)", "service.py"))


def test_read_only_and_approval_policies_fail_closed(tmp_path):
    path = tmp_path / "app.py"
    path.write_text("x = 1\n")
    read_only = WorkspaceTools(str(tmp_path), sandbox_mode=SandboxMode.READ_ONLY)
    with pytest.raises(WorkspaceViolation, match="read-only"):
        read_only.replace_text("app.py", "1", "2")
    approval = WorkspaceTools(str(tmp_path), approval_policy=ApprovalPolicy.ON_REQUEST)
    result = run(approval.run([sys.executable, "-c", "print('ok')"], 10))
    assert result["approval_required"] is True
    with pytest.raises(WorkspaceViolation, match="approval"):
        approval.replace_text("app.py", "1", "2")


def test_workspace_file_tools_hide_credentials_and_vcs_internals(tmp_path):
    (tmp_path / ".env").write_text("API_KEY=must-not-be-read")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("private")
    (tmp_path / "public.txt").write_text("safe")
    tools = WorkspaceTools(str(tmp_path))

    listed = tools.list_files()
    assert listed["files"] == ["public.txt"]
    for protected in (".env", ".git/config"):
        with pytest.raises(WorkspaceViolation, match="protected"):
            tools.read_file(protected)
        with pytest.raises(WorkspaceViolation, match="protected"):
            tools.write_file(protected, "changed", overwrite=True)


def test_workspace_runs_real_argv_without_shell_and_redacts_output(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    if not tools.command_sandbox_backend:
        pytest.skip("functional OS command sandbox backend is unavailable")
    result = run(tools.run([
        sys.executable, "-c", "print('password=hunter-two')",
    ], 10))
    assert result["ok"] is True
    assert "hunter-two" not in result["output"]
    with pytest.raises(WorkspaceViolation, match="shells may only"):
        run(tools.run(["sh", "-c", "echo unsafe"], 10))


def test_docker_sandbox_masks_protected_files_with_an_unreadable_inode(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("API_KEY=must-not-be-mounted")
    runtime_home = tmp_path / "runtime"
    runtime_home.mkdir()
    tools = WorkspaceTools(str(workspace))
    tools.command_sandbox_backend = "docker"
    monkeypatch.setattr(tools, "_docker_workspace", lambda _runtime_home: (workspace, False))

    command = tools._sandbox_command([sys.executable, "-c", "print('ok')"], str(runtime_home))

    denied_file = runtime_home / "blocked-file"
    assert denied_file.exists()
    assert denied_file.stat().st_mode & 0o777 == 0
    assert "src=/dev/null" not in command
    assert any(
        "src={},dst=/workspace/.env,readonly".format(denied_file) in item
        for item in command
    )


def test_model_command_os_sandbox_denies_host_read_and_all_workspace_writes(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("host-secret")
    (workspace / "source.txt").write_text("workspace-readable")
    (workspace / ".env").write_text("API_KEY=workspace-secret")
    tools = WorkspaceTools(str(workspace))
    if not tools.command_sandbox_backend:
        pytest.skip("functional OS command sandbox backend is unavailable")

    readable = run(tools.run([
        sys.executable, "-c", "from pathlib import Path; print(Path('source.txt').read_text())",
    ], 10))
    assert readable["ok"] is True
    assert "workspace-readable" in readable["output"]
    assert readable["sandbox_backend"] in {"docker", "bwrap"}

    protected_read = run(tools.run([
        sys.executable, "-c", "from pathlib import Path; print(Path('.env').read_text())",
    ], 10))
    assert protected_read["ok"] is False
    assert "workspace-secret" not in protected_read["output"]

    network = run(tools.run([
        sys.executable, "-c",
        "import socket; assert [name for _, name in socket.if_nameindex()] == ['lo']",
    ], 10))
    assert network["ok"] is True

    host_read = run(tools.run([
        sys.executable, "-c", "import sys; from pathlib import Path; print(Path(sys.argv[1]).read_text())",
        str(outside),
    ], 10))
    assert host_read["ok"] is False
    assert "host-secret" not in host_read["output"]

    workspace_write = run(tools.run([
        sys.executable, "-c", "from pathlib import Path; Path('mutated.txt').write_text('bad')",
    ], 10))
    assert workspace_write["ok"] is False
    assert not (workspace / "mutated.txt").exists()


def test_source_controlled_check_is_distinct_trusted_command_lane(tmp_path):
    tools = WorkspaceTools(str(tmp_path))
    result = run(tools.run_check([
        sys.executable, "-c", "from pathlib import Path; Path('check-artifact').write_text('verified')",
    ], 10))
    assert result["ok"] is True
    assert (tmp_path / "check-artifact").read_text() == "verified"


def test_source_controlled_config_and_real_check_runner(tmp_path):
    config = tmp_path / ".flyto" / "coding.yaml"
    config.parent.mkdir()
    config.write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: python-smoke\n"
        "    argv: [python, -c, \"print('verified')\"]\n"
        "capabilities:\n"
        "  - name: optional-tool\n"
        "    kind: command\n"
        "    argv: [python]\n"
        "    required_tools: [context]\n"
        "    allowed_tools: [context, impact]\n"
    )
    checks, capabilities = load_project_config(str(tmp_path))
    assert checks[0].name == "python-smoke"
    assert capabilities[0].required is False
    assert capabilities[0].required_tools == ("context",)
    assert capabilities[0].allowed_tools == ("context", "impact")
    results = run(CheckRunner(WorkspaceTools(str(tmp_path))).run(checks))
    assert CheckRunner.passed(results) is True
    assert results[0].output_sha256


def _write_mcp_server(path: Path) -> None:
    path.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    msg=json.loads(line)\n"
        "    if 'id' not in msg: continue\n"
        "    method=msg.get('method')\n"
        "    if method=='initialize': result={'protocolVersion':'2025-06-18','capabilities':{},'serverInfo':{'name':'fixture','version':'1'}}\n"
        "    elif method=='tools/list': result={'tools':[{'name':'context','description':'real fixture','inputSchema':{'type':'object','properties':{}}}]}\n"
        "    elif method=='tools/call': result={'content':[{'type':'text','text':'real context'}]}\n"
        "    else: result={}\n"
        "    print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':result}), flush=True)\n"
    )


def test_real_mcp_stdio_discovery_and_call(tmp_path):
    server = tmp_path / "mcp_fixture.py"
    _write_mcp_server(server)
    manager = CapabilityManager(str(tmp_path))
    async def scenario():
        statuses = await manager.start([CapabilitySpec(
            name="indexer", argv=(sys.executable, str(server)), required=True,
        )])
        definition = manager.definitions[0]
        result = await manager.dispatch(definition["name"], {})
        await manager.close()
        return statuses, result

    statuses, result = run(scenario())
    assert statuses[0].available is True
    assert statuses[0].tool_count == 1
    assert result["ok"] is True
    assert result["result"]["content"][0]["text"] == "real context"


def test_mcp_allowed_tools_expose_only_the_selected_surface(tmp_path):
    server = tmp_path / "mcp_fixture.py"
    server.write_text(
        "import json, sys\n"
        "tools=[{'name':'context','description':'read','inputSchema':{'type':'object'}},"
        "{'name':'mutate','description':'write','inputSchema':{'type':'object'}}]\n"
        "for line in sys.stdin:\n"
        "    msg=json.loads(line)\n"
        "    if 'id' not in msg: continue\n"
        "    method=msg.get('method')\n"
        "    if method=='initialize': result={'protocolVersion':'2025-06-18','capabilities':{},'serverInfo':{'name':'fixture','version':'1'}}\n"
        "    elif method=='tools/list': result={'tools':tools}\n"
        "    elif method=='tools/call': result={'content':[{'type':'text','text':msg['params']['name']}]}\n"
        "    else: result={}\n"
        "    print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':result}), flush=True)\n"
    )
    manager = CapabilityManager(str(tmp_path))

    async def scenario():
        statuses = await manager.start([CapabilitySpec(
            name="indexer-context",
            argv=(sys.executable, str(server)),
            required=True,
            required_tools=("context",),
            allowed_tools=("context",),
        )])
        definitions = manager.definitions
        result = await manager.dispatch(definitions[0]["name"], {})
        hidden = await manager.dispatch("cap_indexer-context_mutate", {})
        await manager.close()
        return statuses, definitions, result, hidden

    statuses, definitions, result, hidden = run(scenario())
    assert statuses[0].available is True
    assert statuses[0].tools == ("context",)
    assert statuses[0].tool_count == 1
    assert [definition["name"] for definition in definitions] == [
        "cap_indexer-context_context",
    ]
    assert result["ok"] is True
    assert hidden == {"ok": False, "error": "unknown capability tool"}


def test_mcp_allowed_tools_fail_closed_when_catalog_does_not_match(tmp_path):
    server = tmp_path / "mcp_fixture.py"
    _write_mcp_server(server)
    manager = CapabilityManager(str(tmp_path))

    async def scenario():
        statuses = await manager.start([CapabilitySpec(
            name="page-inspector",
            argv=(sys.executable, str(server)),
            required=True,
            allowed_tools=("inspect_page",),
        )])
        await manager.close()
        return statuses

    statuses = run(scenario())
    assert statuses[0].available is False
    assert statuses[0].tools == ("context",)
    assert "missing allowed tools: inspect_page" in str(statuses[0].error)


@pytest.mark.parametrize("structured", [False, True])
def test_mcp_dispatch_propagates_nested_domain_failure(tmp_path, structured):
    server = tmp_path / "mcp_fixture.py"
    result_expr = (
        "{'structuredContent':{'ok':False,'error':'browser failed'},'isError':False}"
        if structured
        else "{'content':[{'type':'text','text':json.dumps({'ok':False,'error':'browser failed'})}]}"
    )
    server.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    msg=json.loads(line)\n"
        "    if 'id' not in msg: continue\n"
        "    method=msg.get('method')\n"
        "    if method=='initialize': result={'protocolVersion':'2025-06-18','capabilities':{},'serverInfo':{'name':'fixture','version':'1'}}\n"
        "    elif method=='tools/list': result={'tools':[{'name':'inspect_page','inputSchema':{'type':'object'}}]}\n"
        "    elif method=='tools/call': result=" + result_expr + "\n"
        "    else: result={}\n"
        "    print(json.dumps({'jsonrpc':'2.0','id':msg['id'],'result':result}), flush=True)\n"
    )
    manager = CapabilityManager(str(tmp_path))

    async def scenario():
        await manager.start([CapabilitySpec(
            name="page-inspector",
            argv=(sys.executable, str(server)),
            required=True,
            required_tools=("inspect_page",),
            allowed_tools=("inspect_page",),
        )])
        result = await manager.dispatch(manager.definitions[0]["name"], {})
        await manager.close()
        return result

    result = run(scenario())
    assert result["ok"] is False
    assert result["error"] == "browser failed"
    assert result["capability"] == "page-inspector"
    assert result["tool"] == "inspect_page"


def test_required_tools_must_be_inside_the_allowlist():
    with pytest.raises(ValueError, match="included in allowed_tools"):
        CapabilitySpec(
            name="bad-profile",
            argv=("python",),
            required_tools=("impact",),
            allowed_tools=("context",),
        )


def test_required_missing_capability_fails_before_provider(tmp_path):
    provider = ScriptedProvider([[('coding_write_file', {
        'path': 'result.txt', 'content': 'ok', 'overwrite': False,
    })]])
    store = ThreadStore(str(tmp_path / "state"))
    agent = FlytoCodingAgent(provider, store=store)
    request = CodingTaskRequest(
        message="write result", working_dir=str(tmp_path),
        checks=(CheckSpec("smoke", (sys.executable, "-c", "print('ok')")),),
        capabilities=(CapabilitySpec("missing", ("definitely-not-installed-flyto",), required=True, kind="command"),),
    )
    result = run(agent.run(request))
    assert result.ok is False
    assert result.failure_code == "required_capability_unavailable"
    assert provider.calls == 0


def test_no_verification_fails_before_provider_or_write(tmp_path):
    provider = ScriptedProvider([[('coding_write_file', {
        'path': 'result.txt', 'content': 'ok', 'overwrite': False,
    })]])
    agent = FlytoCodingAgent(provider, store=ThreadStore(str(tmp_path / "state")))
    result = run(agent.run(CodingTaskRequest(message="write result", working_dir=str(tmp_path))))
    assert result.failure_code == "verification_required"
    assert provider.calls == 0
    assert not (tmp_path / "result.txt").exists()


def test_native_agent_real_write_check_evidence_and_usage(tmp_path):
    provider = ScriptedProvider([[('coding_write_file', {
        'path': 'result.txt', 'content': 'closed-loop\n', 'overwrite': False,
    })]])
    store = ThreadStore(str(tmp_path / "state"))
    agent = FlytoCodingAgent(provider, store=store)
    check = CheckSpec("content", (
        sys.executable, "-c",
        "from pathlib import Path; assert Path('result.txt').read_text() == 'closed-loop\\n'",
    ))
    result = run(agent.run(CodingTaskRequest(
        message="write verified result", working_dir=str(tmp_path), checks=(check,),
    )))
    assert result.ok is True
    assert result.files_changed == ["result.txt"]
    assert result.checks[0].passed is True
    assert result.usage["total_tokens"] == 15
    events = store.events(result.thread_id)
    assert {event["type"] for event in events} >= {
        "thread.created", "tool.completed", "provider.completed", "verification.completed",
    }


def test_native_agent_repairs_after_real_check_failure(tmp_path):
    provider = ScriptedProvider([
        [('coding_write_file', {'path': 'answer.txt', 'content': 'wrong', 'overwrite': False})],
        [('coding_write_file', {'path': 'answer.txt', 'content': 'correct', 'overwrite': True})],
    ])
    agent = FlytoCodingAgent(provider, store=ThreadStore(str(tmp_path / "state")))
    check = CheckSpec("answer", (
        sys.executable, "-c", "from pathlib import Path; assert Path('answer.txt').read_text() == 'correct'",
    ))
    result = run(agent.run(CodingTaskRequest(
        message="write the answer", working_dir=str(tmp_path), checks=(check,), max_attempts=2,
    )))
    assert result.ok is True
    assert result.attempts == 2
    assert provider.calls == 2
    assert (tmp_path / "answer.txt").read_text() == "correct"


def test_passing_check_without_attributable_change_is_not_success(tmp_path):
    provider = ScriptedProvider([[]])
    agent = FlytoCodingAgent(provider, store=ThreadStore(str(tmp_path / "state")))
    check = CheckSpec("always", (sys.executable, "-c", "print('ok')"))
    result = run(agent.run(CodingTaskRequest(
        message="make a change", working_dir=str(tmp_path), checks=(check,), max_attempts=1,
    )))
    assert result.ok is False
    assert result.failure_code == "no_changes"
