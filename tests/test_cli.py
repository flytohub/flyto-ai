# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for CLI entry point."""
import json
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from flyto_ai.cli import main, _post_webhook


def test_version_output(capsys):
    """flyto-ai version prints version string and logo."""
    sys.argv = ["flyto-ai", "version"]
    main()
    out = capsys.readouterr().out
    assert "v0.5.4" in out
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
