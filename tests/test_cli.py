# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Tests for CLI entry point."""
import json
import os
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from flyto_ai.cli import main, _post_webhook


def test_version_output(capsys):
    """flyto-ai version prints version string and logo."""
    sys.argv = ["flyto-ai", "version"]
    main()
    out = capsys.readouterr().out
    from flyto_ai import __version__
    assert "v{}".format(__version__) in out
    assert "___" in out  # ASCII art present


def test_version_shows_deps_status(capsys):
    """Version command shows optional dependency status."""
    sys.argv = ["flyto-ai", "version"]
    main()
    out = capsys.readouterr().out
    # Should mention at least one optional dep
    assert "openai" in out.lower()
    assert "anthropic" in out.lower()


def test_chat_no_provider_shows_error(capsys):
    """Chat without API key shows error."""
    sys.argv = ["flyto-ai", "chat", "hello"]
    import os
    # Clear any env vars that might provide keys
    env_backup = {}
    for key in ["FLYTO_AI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "FLYTO_AI_PROVIDER"]:
        env_backup[key] = os.environ.pop(key, None)
    try:
        try:
            main()
        except SystemExit:
            pass  # Expected — exits with code 1
        err = capsys.readouterr().err
        assert "Error" in err or "No API key" in err
    finally:
        for key, val in env_backup.items():
            if val is not None:
                os.environ[key] = val


def test_chat_args_parsing():
    """Chat subcommand parses message, webhook, and flags."""
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    chat_p = sub.add_parser("chat")
    chat_p.add_argument("message", nargs="+")
    chat_p.add_argument("--provider", "-p")
    chat_p.add_argument("--model", "-m")
    chat_p.add_argument("--webhook", "-w")

    args = parser.parse_args(["chat", "scrape", "example.com", "-p", "ollama", "-w", "https://hook.site/test"])
    assert args.command == "chat"
    assert args.message == ["scrape", "example.com"]
    assert args.provider == "ollama"
    assert args.webhook == "https://hook.site/test"


def test_help_no_crash(capsys):
    """Running with no args prints help without crashing."""
    sys.argv = ["flyto-ai"]
    main()  # Should print help, not crash
    out = capsys.readouterr().out
    assert "automation" in out.lower() or "usage" in out.lower()


def test_help_shows_serve(capsys):
    """Help mentions serve subcommand and chat has --plan flag."""
    sys.argv = ["flyto-ai", "-h"]
    try:
        main()
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "serve" in out

    # Verify --plan flag exists on chat subcommand
    sys.argv = ["flyto-ai", "chat", "-h"]
    try:
        main()
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--plan" in out


def test_webhook_post():
    """_post_webhook sends JSON POST to target URL."""
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            received["body"] = json.loads(self.rfile.read(length))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass  # suppress

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.handle_request, daemon=True)
    t.start()

    from flyto_ai.models import ChatResponse
    result = ChatResponse(ok=True, message="test workflow", session_id="")
    _post_webhook("http://127.0.0.1:{}".format(port), result)

    t.join(timeout=5)
    server.server_close()

    assert received.get("body", {}).get("ok") is True
    assert received["body"]["message"] == "test workflow"


def test_serve_args_parsing():
    """Serve subcommand parses host, port, provider."""
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    serve_p = sub.add_parser("serve")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=7411)
    serve_p.add_argument("--provider", "-p")

    args = parser.parse_args(["serve", "--port", "8080", "-p", "ollama"])
    assert args.command == "serve"
    assert args.port == 8080
    assert args.provider == "ollama"
    assert args.host == "0.0.0.0"


# ──────────────────────────────────────────────────────────────────────
# Startup implementer selection for the public coding route
# ──────────────────────────────────────────────────────────────────────

def _service_args(tmp_path, **overrides):
    import argparse
    values = dict(
        tenant="tenant-cli",
        workspace_root=[str(tmp_path)],
        state_dir=str(tmp_path / "state"),
        provider="ollama",
        model=None,
        base_url=None,
        config=".flyto/coding.yaml",
        approval="never",
        sandbox="workspace-write",
        sandbox_image="python:3.12-slim",
        max_workers=2,
        max_queued=100,
        implementation_backend="native",
        max_rework_rounds=3,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class _StubProvider:
    async def chat(self, **kwargs):  # pragma: no cover - never called here
        raise AssertionError("no provider call belongs in a startup test")


def test_native_selection_builds_audited_native_service(tmp_path, monkeypatch):
    import flyto_ai.cli as cli
    from flyto_ai.coding import FlytoCodingAgent
    from flyto_ai.coding.store import ThreadStore

    calls = []
    monkeypatch.setattr(
        cli, "_create_native_coding_provider",
        lambda args: calls.append(args) or _StubProvider(),
    )
    service = cli._build_coding_service(
        _service_args(tmp_path, max_rework_rounds=5),
    )
    try:
        assert service.require_codex_audit is True
        assert service.implementation_backend == "native"
        assert service.max_rework_rounds == 5
        # Startup validation runs once before any job is scheduled.
        assert len(calls) == 1
        agent = service.agent_factory(ThreadStore(str(tmp_path / "threads")))
        assert isinstance(agent, FlytoCodingAgent)
        assert len(calls) == 2
    finally:
        service.close()


def test_claude_selection_builds_audited_adapter_without_native_provider(
    tmp_path, monkeypatch,
):
    import types
    import flyto_ai.cli as cli
    from flyto_ai.agents.claude_code import ClaudeCodingAgent
    from flyto_ai.coding.store import ThreadStore

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", types.ModuleType("claude_agent_sdk"))

    def _forbidden(args):
        raise AssertionError("the claude backend must not create a native provider")

    monkeypatch.setattr(cli, "_create_native_coding_provider", _forbidden)
    service = cli._build_coding_service(
        _service_args(tmp_path, implementation_backend="claude", max_rework_rounds=2),
    )
    try:
        assert service.require_codex_audit is True
        assert service.implementation_backend == "claude"
        assert service.max_rework_rounds == 2
        first = service.agent_factory(ThreadStore(str(tmp_path / "threads-a")))
        second = service.agent_factory(ThreadStore(str(tmp_path / "threads-b")))
        assert isinstance(first, ClaudeCodingAgent)
        assert isinstance(second, ClaudeCodingAgent)
        # One read-only startup configuration, so authority cannot drift.
        assert first.agent._config is second.agent._config
        assert first.agent.resolve_model(first.agent._cc) == "claude-opus-5"
    finally:
        service.close()


def test_claude_service_options_stay_pinned_to_opus_5(tmp_path, monkeypatch):
    import types
    import flyto_ai.cli as cli
    from flyto_ai.agents.models import CodeTaskRequest
    from flyto_ai.coding.store import ThreadStore

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", types.ModuleType("claude_agent_sdk"))
    monkeypatch.setenv("FLYTO_AI_CC_MODEL", "claude-sonnet-4-6")
    service = cli._build_coding_service(
        _service_args(tmp_path, implementation_backend="claude"),
    )
    try:
        agent = service.agent_factory(ThreadStore(str(tmp_path / "threads"))).agent
        assert agent._cc.model == "claude-sonnet-4-6"
        options = agent._option_kwargs(
            CodeTaskRequest(
                message="task", working_dir=str(tmp_path), service_mode=True,
            ),
            session_id=None, system_prompt="s", max_turns=1, max_budget=1.0,
        )
        assert options["model"] == "claude-opus-5"
    finally:
        service.close()


def test_unavailable_or_invalid_startup_selection_fails_closed(tmp_path, monkeypatch):
    import pytest
    import flyto_ai.cli as cli

    monkeypatch.setattr(cli, "_create_native_coding_provider", lambda args: _StubProvider())
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    with pytest.raises(ValueError, match="requires the optional Claude"):
        cli._build_coding_service(
            _service_args(tmp_path, implementation_backend="claude"),
        )
    with pytest.raises(ValueError, match="invalid --implementation-backend"):
        cli._build_coding_service(
            _service_args(tmp_path, implementation_backend="codex"),
        )
    for invalid in (0, 100, "3", True):
        with pytest.raises(ValueError, match="max_rework_rounds"):
            cli._build_coding_service(_service_args(tmp_path, max_rework_rounds=invalid))
    # Nothing was left holding the durable state lease.
    service = cli._build_coding_service(_service_args(tmp_path))
    service.close()


def test_environment_backend_default_is_bounded(monkeypatch):
    import pytest
    import flyto_ai.cli as cli

    monkeypatch.delenv(cli.CODING_BACKEND_ENV, raising=False)
    assert cli._default_coding_backend() == "native"
    for value in ("native", "claude"):
        monkeypatch.setenv(cli.CODING_BACKEND_ENV, value)
        assert cli._default_coding_backend() == value
    monkeypatch.setenv(cli.CODING_BACKEND_ENV, "  claude  ")
    assert cli._default_coding_backend() == "claude"
    for invalid in ("codex", "gpt", "Native", "native,claude"):
        monkeypatch.setenv(cli.CODING_BACKEND_ENV, invalid)
        with pytest.raises(SystemExit, match=cli.CODING_BACKEND_ENV):
            cli._default_coding_backend()


def test_both_public_commands_expose_the_startup_options():
    import subprocess
    from pathlib import Path

    for command in ("code-mcp", "code-serve"):
        completed = subprocess.run(
            [sys.executable, "-m", "flyto_ai.cli", command, "--help"],
            cwd=str(Path(__file__).parents[1]),
            capture_output=True, text=True, timeout=60, check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--implementation-backend" in completed.stdout
        assert "--max-rework-rounds" in completed.stdout
        assert "native" in completed.stdout and "claude" in completed.stdout
        # There is no switch that disables the Codex audit requirement.
        assert "--require-codex-audit" not in completed.stdout
        assert "--no-audit" not in completed.stdout


def test_invalid_backend_env_only_affects_the_coding_service_commands(tmp_path, monkeypatch):
    import pytest
    import flyto_ai.cli as cli

    monkeypatch.setenv(cli.CODING_BACKEND_ENV, "codex")
    # Building the global parser must not evaluate the service-only default.
    monkeypatch.setattr(sys, "argv", ["flyto-ai", "version"])
    cli.main()

    monkeypatch.setattr(cli, "_create_native_coding_provider", lambda args: _StubProvider())
    with pytest.raises(SystemExit, match=cli.CODING_BACKEND_ENV):
        cli._build_coding_service(
            _service_args(tmp_path, implementation_backend=None),
        )
    # An explicit CLI choice still overrides a broken environment default.
    service = cli._build_coding_service(
        _service_args(tmp_path, implementation_backend="native"),
    )
    try:
        assert service.implementation_backend == "native"
    finally:
        service.close()


def test_unrelated_commands_survive_an_invalid_backend_env(tmp_path):
    import subprocess
    from pathlib import Path

    env = dict(os.environ, FLYTO_AI_CODING_BACKEND="codex")
    root = str(Path(__file__).parents[1])
    for argv in (["version"], ["--help"], ["code-mcp", "--help"]):
        completed = subprocess.run(
            [sys.executable, "-m", "flyto_ai.cli", *argv],
            cwd=root, env=env, capture_output=True, text=True, timeout=60, check=False,
        )
        assert completed.returncode == 0, (argv, completed.stderr)

    started = subprocess.run(
        [
            sys.executable, "-m", "flyto_ai.cli", "code-mcp",
            "--tenant", "tenant-env", "--workspace-root", str(tmp_path),
            "--state-dir", str(tmp_path / "state"), "--provider", "ollama",
        ],
        cwd=root, env=env, input="", capture_output=True, text=True,
        timeout=120, check=False,
    )
    assert started.returncode != 0
    assert "FLYTO_AI_CODING_BACKEND" in (started.stderr + started.stdout)


def test_claude_route_reads_no_native_provider_configuration(tmp_path, monkeypatch):
    import types
    import pytest
    import flyto_ai.cli as cli
    from flyto_ai.agents.models import CodeTaskRequest
    from flyto_ai.coding.store import ThreadStore
    from flyto_ai.config import AgentConfig

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", types.ModuleType("claude_agent_sdk"))
    monkeypatch.setattr(
        AgentConfig, "from_env",
        classmethod(lambda cls: pytest.fail("the claude route must not resolve native config")),
    )
    monkeypatch.setattr(
        cli, "_create_native_coding_provider",
        lambda args: pytest.fail("the claude route must not create a native provider"),
    )
    for native in ("FLYTO_AI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(native, "must-not-be-read")
    monkeypatch.setenv("FLYTO_AI_PROVIDER", "openai")
    monkeypatch.setenv("FLYTO_AI_BASE_URL", "https://must-not-be-read.invalid")
    monkeypatch.setenv("FLYTO_AI_CC_MAX_BUDGET", "7.5")
    monkeypatch.setenv("FLYTO_AI_CC_MAX_TURNS", "11")
    monkeypatch.setenv("FLYTO_AI_CC_MODEL", "claude-sonnet-4-6")

    service = cli._build_coding_service(
        _service_args(tmp_path, implementation_backend="claude"),
    )
    try:
        agent = service.agent_factory(ThreadStore(str(tmp_path / "threads"))).agent
        # Bounded Claude settings are honored...
        assert agent._cc.max_budget_usd == 7.5
        assert agent._cc.max_turns == 11
        assert agent._cc.model == "claude-sonnet-4-6"
        # ...while no native credential or provider setting was resolved.
        assert agent._config.api_key == ""
        assert agent._config.base_url in (None, "")
        assert "must-not-be-read" not in repr(agent._config)
        options = agent._option_kwargs(
            CodeTaskRequest(message="t", working_dir=str(tmp_path), service_mode=True),
            session_id=None, system_prompt="s", max_turns=1, max_budget=1.0,
        )
        assert options["model"] == "claude-opus-5"
    finally:
        service.close()
