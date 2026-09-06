"""Real subprocess protocol tests; no model, account or provider network calls."""

import asyncio
import base64
import json
import os
import textwrap
from pathlib import Path

import pytest

from flyto_ai import Agent, AgentConfig
from flyto_ai.cli_runtime import (
    CliAgent,
    CliRuntimeConfig,
    CliRuntimeError,
    cli_environment,
    complete_json,
    inspect_cli_runtime,
    required_cli_flags,
    resolve_cli_executable,
)
from flyto_ai.cli_runtime.contracts import checked_intent
from flyto_ai.cli_runtime.events import EventReader
from flyto_ai.cli_runtime.process import ProcessRunner


def binary(tmp_path, program):
    path = tmp_path / "local-cli"
    flags = " ".join(required_cli_flags("claude_cli"))
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(f'''\
        import json,sys,os,time
        if '--version' in sys.argv:
            print('2.1.258 (Claude Code)');sys.exit(0)
        if '--help' in sys.argv:
            print({flags!r});sys.exit(0)
        request=json.loads(sys.stdin.readline())
        envelope=json.loads(request['message']['content'][0]['text'])
        def emit(value):
            print(json.dumps(value),flush=True)
        def finish(value):
            emit({{'type':'result','subtype':'success','is_error':False,
                  'session_id':'test-session','structured_output':value,
                  'usage':{{'input_tokens':10,'output_tokens':5}}}})
        emit({{'type':'system','subtype':'init','session_id':'test-session','tools':[]}})
        ''') + textwrap.dedent(program))
    path.chmod(0o700)
    return str(path)


def make_agent(command, dispatch, *, permission="workspace_write", tools=None):
    config = AgentConfig(enable_memory=False, enable_pro=False, enable_transcript=False,
                         enable_injection_detection=False, permission_level=permission,
                         max_tool_rounds=3)
    agent = CliAgent(config, cli=CliRuntimeConfig("claude_cli", command=command),
                     tools=tools or [{"name": "execute_module", "inputSchema": {"type": "object"}}],
                     dispatch_fn=dispatch, system_prompt="Complete the assigned local task with exposed tools.",
                     policies={"allowed_tools": ["execute_module"], "allowed_categories": ["file"]})
    agent._assistant = None
    return agent


@pytest.mark.asyncio
async def test_real_cli_subprocess_proposes_but_host_alone_writes_and_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "result.txt"
    command = binary(tmp_path, f'''
        prior=[m for m in envelope['messages'] if m['role']=='tool']
        if not prior:
            finish({{'content':'', 'tool_calls':[{{'name':'execute_module','arguments_json':json.dumps({{'module_id':'file.write','params':{{'path':{str(target)!r},'content':'observed-local-value'}}}})}}]}})
        else:
            assert 'host-receipt' in prior[-1]['content']
            finish({{'content':'The host wrote the requested note.','tool_calls':[]}})
    ''')
    calls, events = [], []
    async def dispatch(name, args):
        calls.append((name, args))
        target.write_text(args["params"]["content"])
        return {"ok": True, "data": {"receipt": "host-receipt"}}
    agent = make_agent(command, dispatch)
    try:
        result = await agent.chat("Write a note in the current workspace.", on_stream=events.append)
    finally:
        await agent.close()
    assert result.ok and result.provider == "claude_cli"
    assert target.read_text() == "observed-local-value"
    assert len(calls) == 1
    assert result.execution_results[0]["function"] == "execute_module"
    assert "host-receipt" in result.execution_results[0]["result_preview"]
    assert any(event.type.value == "tool_end" for event in events)
    assert result.usage.total_tokens == 30
    assert not agent.cli_runtime.runner._processes


@pytest.mark.asyncio
async def test_read_only_policy_cannot_be_overridden_by_cli_intent(tmp_path):
    command = binary(tmp_path, '''
        finish({'content':'', 'tool_calls':[{'name':'execute_module','arguments_json':'{}'}]})
    ''')
    calls = []
    async def dispatch(*args):
        calls.append(args)
        return {"ok": True}
    agent = make_agent(command, dispatch, permission="read_only")
    try:
        result = await agent.chat("Write a note in this workspace.")
    finally:
        await agent.close()
    assert not result.ok
    assert result.error == "cli_tool_not_available"
    assert calls == []


@pytest.mark.asyncio
async def test_toolless_complete_json_never_dispatches_and_disables_native_tools(tmp_path):
    command = binary(tmp_path, '''
        assert not envelope['tools']
        assert sys.argv[sys.argv.index('--tools')+1]==''
        assert json.loads(sys.argv[sys.argv.index('--mcp-config')+1])=={'mcpServers':{}}
        assert '--safe-mode' in sys.argv and '--restricted' in sys.argv
        assert '--no-session-persistence' in sys.argv
        assert '--bare' not in sys.argv
        assert 'OPENAI_API_KEY' not in os.environ
        finish({'complete':False,'remaining':['Read the requested record']})
    ''')
    value = await complete_json(CliRuntimeConfig("claude_cli", command=command), prompt="Review observations.",
                                schema={"type": "object", "properties": {"complete": {"type": "boolean"}}})
    assert json.loads(value)["complete"] is False


@pytest.mark.asyncio
async def test_partial_observations_survive_cli_provider_failure_without_replaying(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    command = binary(tmp_path, '''
        if any(m['role']=='tool' for m in envelope['messages']):
            emit({'type':'result','subtype':'error_during_execution','is_error':True,'result':'quota exceeded'})
        else:
            finish({'content':'','tool_calls':[{'name':'execute_module','arguments_json':json.dumps({'module_id':'file.read','params':{'path':'result.txt'}})}]})
    ''')
    calls = []
    async def dispatch(*args):
        calls.append(args)
        return {"ok": True, "data": {"observed": "existing record"}}
    agent = make_agent(command, dispatch)
    try:
        result = await agent.chat("Read the workspace file and summarize it.")
    finally:
        await agent.close()
    assert not result.ok and result.error == "cli_quota_exhausted"
    assert len(calls) == 1 and len(result.execution_results) == 1
    assert "existing record" in result.execution_results[0]["result_preview"]


@pytest.mark.asyncio
async def test_same_admission_continuation_keeps_host_tool_observations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    command = binary(tmp_path, '''
        seen=[m for m in envelope['messages'] if m['role']=='tool']
        if seen:
            assert 'observed-past-action' in seen[0]['content']
            finish({'content':'The prior observation is available.','tool_calls':[]})
        else:
            finish({'content':'','tool_calls':[{'name':'execute_module','arguments_json':json.dumps({'module_id':'file.read','params':{'path':'result.txt'}})}]})
    ''')
    calls = []
    async def dispatch(*args):
        calls.append(args)
        return {"ok": True, "data": "observed-past-action"}
    agent = make_agent(command, dispatch)
    goal = "Read the workspace note."
    try:
        first = await agent.chat(goal)
        second = await agent.continue_execution("Review the earlier observation.", goal=goal)
        with pytest.raises(PermissionError):
            await agent.continue_execution("Continue", goal="Another goal")
    finally:
        await agent.close()
    assert first.ok and second.ok and len(calls) == 1


@pytest.mark.asyncio
async def test_observed_image_is_forwarded_in_memory_not_read_from_model_path(tmp_path):
    encoded = base64.b64encode(b"\x89PNG\r\n\x1a\nfixture-image").decode()
    command = binary(tmp_path, f'''
        image=request['message']['content'][1]
        assert image['type']=='image' and image['source']['data']=={encoded!r}
        assert not os.listdir(os.getcwd())
        finish({{'complete':False}})
    ''')
    runner = ProcessRunner(CliRuntimeConfig("claude_cli", command=command))
    try:
        result, _ = await runner.infer('{"messages":[],"tools":[]}', {"type": "object"},
                                       [{"media_type": "image/png", "base64": encoded}])
    finally:
        await runner.close()
    assert result == {"complete": False}


@pytest.mark.asyncio
async def test_cancel_kills_owned_subprocess_group_and_prevents_late_side_effect(tmp_path):
    marker = tmp_path / "late-side-effect"
    command = binary(tmp_path, f'''
        import subprocess
        subprocess.Popen([sys.executable,'-c',"import time,pathlib;time.sleep(1);pathlib.Path({str(marker)!r}).write_text('late')"])
        time.sleep(30)
    ''')
    runner = ProcessRunner(CliRuntimeConfig("claude_cli", command=command))
    pending = asyncio.create_task(runner.infer('{"messages":[],"tools":[]}', {"type": "object"}))
    for _ in range(100):
        if runner._supported and runner._processes:
            await asyncio.sleep(0.05)
            break
        await asyncio.sleep(0.01)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    await runner.close()
    await asyncio.sleep(1.1)
    assert not marker.exists() and not runner._processes
    with pytest.raises(CliRuntimeError, match="cli_closed"):
        await runner.infer('{}', {"type": "object"})


@pytest.mark.asyncio
async def test_timeout_is_bounded_and_does_not_retry_provider(tmp_path):
    command = binary(tmp_path, "time.sleep(30)\n")
    with pytest.raises(CliRuntimeError, match="cli_timeout"):
        await complete_json(CliRuntimeConfig("claude_cli", command=command, timeout_seconds=0.15),
                            prompt="Return JSON.", schema={"type": "object"})


@pytest.mark.asyncio
async def test_native_action_event_aborts_without_accepting_later_success(tmp_path):
    command = binary(tmp_path, '''
        emit({'type':'assistant','message':{'content':[{'type':'tool_use','name':'Bash','input':{}}]}})
        finish({'content':'Done','tool_calls':[]})
    ''')
    with pytest.raises(CliRuntimeError, match="cli_native_action_refused"):
        await complete_json(CliRuntimeConfig("claude_cli", command=command), prompt="Return JSON.", schema={"type": "object"})


@pytest.mark.asyncio
async def test_installed_cli_without_required_protocol_is_not_ready(tmp_path):
    command = binary(tmp_path, "raise AssertionError('Inference must not launch')\n")
    text = Path(command).read_text().replace(" ".join(required_cli_flags("claude_cli")), " ".join(required_cli_flags("codex_cli")))
    Path(command).write_text(text)
    status = await inspect_cli_runtime(CliRuntimeConfig("codex_cli", command=command))
    assert status["installed"] and not status["supported"]
    assert status["reason_code"] == "cli_incomplete_output"


@pytest.mark.asyncio
async def test_missing_binary_fails_without_api_fallback(tmp_path):
    status = await inspect_cli_runtime(CliRuntimeConfig("claude_cli", command=str(tmp_path / "absent")))
    assert not status["installed"] and status["reason_code"] == "cli_not_found"
    with pytest.raises(CliRuntimeError, match="cli_not_found"):
        await complete_json(CliRuntimeConfig("claude_cli", command=str(tmp_path / "absent")), prompt="Return JSON.", schema={"type": "object"})


@pytest.mark.parametrize("system,source,executable,expected", [
    ("Darwin", "codex_cli", True, "/Applications/ChatGPT.app/Contents/Resources/codex"),
    ("Linux", "codex_cli", True, None),
    ("Darwin", "claude_cli", True, None),
    ("Darwin", "codex_cli", False, None),
])
def test_official_bundle_discovery_is_mac_only_and_requires_executable(monkeypatch, system, source, executable, expected):
    from flyto_ai.cli_runtime import process
    monkeypatch.setattr(process.shutil, "which", lambda _: None)
    monkeypatch.setattr(process.platform, "system", lambda: system)
    monkeypatch.setattr(process.Path, "is_file", lambda path: str(path) == "/Applications/ChatGPT.app/Contents/Resources/codex")
    monkeypatch.setattr(process.os, "access", lambda _path, _mode: executable)
    assert resolve_cli_executable(source) == expected
    assert ProcessRunner(CliRuntimeConfig(source)).executable == expected


def test_path_selection_and_explicit_missing_command_never_fall_back(monkeypatch):
    from flyto_ai.cli_runtime import process
    monkeypatch.setattr(process.shutil, "which", lambda name: "/trusted/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(process.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(process.Path, "is_file", lambda _path: (_ for _ in ()).throw(AssertionError("No bundle lookup permitted")))
    assert resolve_cli_executable("codex_cli") == "/trusted/bin/codex"
    assert resolve_cli_executable("codex_cli", "/missing/explicit-codex") is None


@pytest.mark.asyncio
async def test_native_agent_still_requires_its_api_key():
    agent = Agent(AgentConfig(provider="openai", enable_memory=False, enable_pro=False, enable_transcript=False))
    try:
        result = await agent.chat("Hello")
    finally:
        await agent.close()
    assert not result.ok and result.error == "no_api_key"


@pytest.mark.parametrize("intent", [
    {"content": "Done", "tool_calls": [], "execution_id": "claimed"},
    {"content": "", "tool_calls": [{"name": "unknown", "arguments_json": "{}"}]},
    {"content": "", "tool_calls": [{"name": "execute_module", "arguments_json": '{"x":1,"x":2}'}]},
    {"content": "", "tool_calls": [{"name": "execute_module", "arguments_json": "[]"}]},
])
def test_host_validates_entire_intent_before_any_dispatch(intent):
    with pytest.raises(CliRuntimeError):
        checked_intent(intent, {"execute_module"})


def test_environment_does_not_inherit_api_keys_or_provider_tokens(monkeypatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "GITHUB_TOKEN", "HTTP_PROXY"):
        monkeypatch.setenv(name, "must-not-cross-boundary")
    environment = cli_environment()
    assert set(environment) <= {"HOME", "USER", "LOGNAME", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "PATH", "TMPDIR", "LANG", "LC_ALL", "TERM", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    assert environment.get("HOME") == os.environ.get("HOME")
    assert environment.get("USER") == os.environ.get("USER")
    assert environment.get("LOGNAME") == os.environ.get("LOGNAME")


def test_cli_turn_completion_alone_is_not_structured_goal_result():
    reader = EventReader("codex_cli")
    reader.read(b'{"type":"thread.started","thread_id":"t1"}')
    reader.read(b'{"type":"turn.completed","usage":{}}')
    with pytest.raises(CliRuntimeError, match="cli_incomplete_output"):
        reader.result()


@pytest.mark.asyncio
async def test_delegated_inference_never_probes_local_cli_and_still_dispatches_in_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from flyto_ai.cli_runtime import transport
    def forbidden(*args, **kwargs):
        raise AssertionError('Delegated inference must not inspect a local binary')
    monkeypatch.setattr(transport, 'ProcessRunner', forbidden)
    observed = []
    async def complete(*, prompt, schema, system_prompt):
        envelope = json.loads(prompt)
        assert schema['required'] == ['content', 'tool_calls'] and system_prompt
        if any(item['role'] == 'tool' for item in envelope['messages']):
            return json.dumps({'content': 'Observed both local steps.', 'tool_calls': []})
        return json.dumps({'content': '', 'tool_calls': [
            {'name': 'execute_module', 'arguments_json': json.dumps({'module_id': 'file.read', 'params': {'path': name}})}
            for name in ['first.txt', 'second.txt']]})
    async def dispatch(name, args):
        observed.append(args['params']['path'])
        if len(observed) == 2:
            assert observed[0] == 'first.txt'
        return {'ok': True, 'data': {'actual_observation': observed[-1]}}
    agent = CliAgent(AgentConfig(enable_memory=False, enable_pro=False, enable_transcript=False),
                     cli=CliRuntimeConfig('codex_cli'), completion_fn=complete,
                     tools=[{'name': 'execute_module', 'inputSchema': {'type': 'object'}}], dispatch_fn=dispatch,
                     policies={'allowed_tools': ['execute_module'], 'allowed_categories': ['file']})
    agent._assistant = None
    try:
        result = await agent.chat('Read the two local workspace files in order.')
    finally:
        await agent.close()
    assert result.ok and observed == ['first.txt', 'second.txt']
    assert result.provider == 'codex_cli' and len(result.execution_results) == 2
    assert result.usage is None  # Missing delegated usage is not fabricated as zero.


@pytest.mark.asyncio
async def test_close_cancels_delegated_inference_before_late_host_dispatch():
    entered, cleaned = asyncio.Event(), asyncio.Event()
    async def complete(**kwargs):
        entered.set()
        try:
            await asyncio.sleep(30)
        finally:
            cleaned.set()
    calls = []
    async def dispatch(*args):
        calls.append(args)
        return {'ok': True}
    agent = CliAgent(AgentConfig(enable_memory=False, enable_pro=False, enable_transcript=False),
                     cli=CliRuntimeConfig('codex_cli'), completion_fn=complete,
                     tools=[{'name': 'execute_module', 'inputSchema': {'type': 'object'}}], dispatch_fn=dispatch)
    agent._assistant = None
    task = asyncio.create_task(agent.chat('Read the local workspace file.'))
    await asyncio.wait_for(entered.wait(), 1)
    await agent.close()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set() and calls == []


@pytest.mark.asyncio
async def test_terminal_cli_cannot_leave_detached_child_side_effect(tmp_path):
    marker = tmp_path / 'late-success-side-effect'
    command = binary(tmp_path, f'''
        import subprocess
        subprocess.Popen([sys.executable,'-c',"import time,pathlib;time.sleep(1);pathlib.Path({str(marker)!r}).write_text('late')"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        finish({{'ok':True}})
    ''')
    result = await complete_json(CliRuntimeConfig('claude_cli', command=command), prompt='Return JSON.', schema={'type':'object'})
    assert json.loads(result) == {'ok': True}
    await asyncio.sleep(1.1)
    assert not marker.exists()
