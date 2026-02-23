# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""flyto-ai CLI — natural language automation from the terminal."""
import argparse
import asyncio
import json
import sys
from urllib.request import Request, urlopen
from urllib.error import URLError


def main():
    parser = argparse.ArgumentParser(
        prog="flyto-ai",
        description="AI agent that turns natural language into executable automation workflows.",
    )
    sub = parser.add_subparsers(dest="command")

    # flyto-ai chat "scrape example.com"
    chat_p = sub.add_parser("chat", help="Send a message to the agent")
    chat_p.add_argument("message", nargs="+", help="What you want to automate")
    chat_p.add_argument("--provider", "-p", help="LLM provider (openai, anthropic, ollama)")
    chat_p.add_argument("--model", "-m", help="Model name")
    chat_p.add_argument("--api-key", "-k", help="API key (or use env vars)")
    chat_p.add_argument("--json", action="store_true", help="Output raw JSON")
    chat_p.add_argument("--plan", action="store_true", help="Only generate YAML workflow (don't execute)")
    chat_p.add_argument("--webhook", "-w", help="POST result to this webhook URL")

    # flyto-ai version
    sub.add_parser("version", help="Show version and optional dependency status")

    # flyto-ai blueprints
    bp_p = sub.add_parser("blueprints", help="List and export learned blueprints")
    bp_p.add_argument("--export", action="store_true", help="Export top blueprints as YAML")
    bp_p.add_argument("--min-score", type=int, default=0, help="Minimum score to show (default: 0)")

    # flyto-ai serve
    serve_p = sub.add_parser("serve", help="Start HTTP server for webhook triggers")
    serve_p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_p.add_argument("--port", type=int, default=7411, help="Bind port (default: 7411)")
    serve_p.add_argument("--provider", "-p", help="LLM provider")
    serve_p.add_argument("--model", "-m", help="Model name")
    serve_p.add_argument("--api-key", "-k", help="API key (or use env vars)")

    # flyto-ai (interactive mode, no subcommand)
    sub.add_parser("interactive", help="Start interactive chat (default when no args)")

    args = parser.parse_args()

    if args.command == "version":
        _cmd_version()
    elif args.command == "blueprints":
        _cmd_blueprints(args)
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "chat":
        _cmd_chat(args)
    elif args.command == "interactive" or (args.command is None and sys.stdin.isatty()):
        _cmd_interactive(args)
    elif args.command is None:
        parser.print_help()
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # Shortcut: flyto-ai "scrape example.com" → treat as chat
        args.message = sys.argv[1:]
        args.provider = None
        args.model = None
        args.api_key = None
        args.json = False
        args.plan = False
        args.webhook = None
        _cmd_chat(args)
    else:
        parser.print_help()


_LOGO_LINES = [
    r"  _____ _       _        ____       _    ___  ",
    r" |  ___| |_   _| |_ ___ |___ \     / \  |_ _| ",
    r" | |_  | | | | | __/ _ \  __) |   / _ \  | |  ",
    r" |  _| | | |_| | || (_) |/ __/   / ___ \ | |  ",
    r" |_|   |_|\__, |\__\___/|_____|  /_/   \_\___| ",
    r"           |___/                                ",
]

# 256-color gradient: cyan → blue → purple
_GRADIENT = ["\033[38;5;87m", "\033[38;5;81m", "\033[38;5;75m",
             "\033[38;5;69m", "\033[38;5;63m", "\033[38;5;99m"]
_RESET = "\033[0m"
_DIM = "\033[90m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"

# UI helpers
import re as _re
import shutil as _shutil
_ANSI_RE = _re.compile(r'\033\[[0-9;]*m')

# Purple gradient for input box (light → deep)
_P1 = "\033[38;5;147m"  # light lavender — top
_P2 = "\033[38;5;141m"  # medium purple — input │
_P3 = "\033[38;5;135m"  # deeper purple — status │
_P4 = "\033[38;5;99m"   # deep purple — bottom


def _term_width():
    return _shutil.get_terminal_size((80, 24)).columns


def _cmd_version():
    from flyto_ai import __version__

    print()
    for i, line in enumerate(_LOGO_LINES):
        color = _GRADIENT[i % len(_GRADIENT)]
        print("{}{}{}".format(color, line, _RESET))

    print()
    print("  {}{}v{}{}  {}412 batteries included{}".format(
        _BOLD, _CYAN, __version__, _RESET, _DIM, _RESET,
    ))
    print()

    # Show optional deps status with versions
    deps = [
        ("openai", "openai", "openai"),
        ("anthropic", "anthropic", "anthropic"),
        ("flyto-core", "core", "flyto-core"),
        ("flyto-blueprint", "flyto_blueprint", "flyto-blueprint"),
    ]
    for label, module, pkg_name in deps:
        try:
            mod = __import__(module)
            ver = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
            if not ver:
                ver = _get_pkg_version(pkg_name)
            if ver:
                print("  {}\u2714{} {} {}{}{}".format(_GREEN, _RESET, label, _DIM, ver, _RESET))
            else:
                print("  {}\u2714{} {}".format(_GREEN, _RESET, label))
        except ImportError:
            print("  {}\u2718 {}{}".format(_DIM, label, _RESET))
    print()


def _get_pkg_version(pkg_name):
    """Get package version from importlib.metadata."""
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return None


def _cmd_blueprints(args):
    try:
        from flyto_blueprint import get_engine
        from flyto_blueprint.storage.sqlite import SQLiteBackend
    except ImportError:
        print("{}flyto-blueprint not installed. Run: pip install flyto-ai[blueprint]{}".format(
            _DIM, _RESET,
        ), file=sys.stderr)
        sys.exit(1)

    engine = get_engine(storage=SQLiteBackend())
    all_bp = engine.list_blueprints()

    learned = [bp for bp in all_bp if bp.get("_source") == "learned"]
    builtins = [bp for bp in all_bp if bp.get("_source") != "learned"]

    if args.min_score:
        learned = [bp for bp in learned if bp.get("score", 0) >= args.min_score]

    if args.export:
        _export_blueprints(learned)
        return

    # Summary
    print()
    print("  {}Blueprints{}".format(_BOLD, _RESET))
    print("  builtin: {}{}{}  learned: {}{}{}".format(
        _CYAN, len(builtins), _RESET, _GREEN, len(learned), _RESET,
    ))
    print()

    if not learned:
        print("  {}No learned blueprints yet.{}".format(_DIM, _RESET))
        print("  {}Use the agent to build workflows — good ones auto-save as blueprints.{}".format(
            _DIM, _RESET,
        ))
        print()
        return

    # Table
    print("  {}{:<30} {:>5}  {:>4}/{:<4}  {}{}".format(
        _DIM, "NAME", "SCORE", "OK", "FAIL", "TAGS", _RESET,
    ))
    print("  {}{}{}".format(_DIM, "-" * 65, _RESET))
    for bp in sorted(learned, key=lambda b: b.get("score", 0), reverse=True):
        name = bp.get("name", bp.get("id", "?"))[:30]
        score = bp.get("score", 0)
        ok = bp.get("success_count", 0)
        fail = bp.get("fail_count", 0)
        tags = ", ".join(bp.get("tags", [])[:3])
        score_color = _GREEN if score >= 50 else (_YELLOW if score >= 20 else _DIM)
        print("  {:<30} {}{:>5}{}  {:>4}/{:<4}  {}{}{}".format(
            name, score_color, score, _RESET, ok, fail, _DIM, tags, _RESET,
        ))
    print()
    print("  {}Tip: flyto-ai blueprints --export > my-blueprints.yaml{}".format(_DIM, _RESET))
    print()


def _export_blueprints(blueprints):
    """Export learned blueprints as YAML to stdout."""
    import yaml

    if not blueprints:
        print("# No learned blueprints to export.", file=sys.stderr)
        sys.exit(0)

    export = []
    for bp in sorted(blueprints, key=lambda b: b.get("score", 0), reverse=True):
        clean = {
            "id": bp.get("id"),
            "name": bp.get("name"),
            "description": bp.get("description", ""),
            "tags": bp.get("tags", []),
            "score": bp.get("score", 0),
            "success_count": bp.get("success_count", 0),
            "steps": bp.get("steps", []),
        }
        if bp.get("params"):
            clean["params"] = bp["params"]
        export.append(clean)

    print("# Exported from flyto-ai — {} learned blueprint(s)".format(len(export)))
    print("# Submit as PR to https://github.com/flytohub/flyto-blueprint")
    print(yaml.dump(export, allow_unicode=True, sort_keys=False, default_flow_style=False))


def _cmd_interactive(args):
    """Interactive chat REPL — like Claude Code but for automation workflows."""
    from flyto_ai import Agent, AgentConfig, __version__

    # Init agent
    config = AgentConfig.from_env()
    provider = getattr(args, "provider", None)
    model = getattr(args, "model", None)
    api_key = getattr(args, "api_key", None)
    if provider:
        config.provider = provider
    if model:
        config.model = model
    if api_key:
        config.api_key = api_key

    agent = Agent(config=config)
    history = []
    tool_count = len(agent._tools) if agent._tools else 0

    # ── Welcome ────────────────────────────────────────────────
    print()
    for i, line in enumerate(_LOGO_LINES):
        color = _GRADIENT[i % len(_GRADIENT)]
        print("  {}{}{}".format(color, line, _RESET))
    print()
    print("  {}{}v{}{}  {}Interactive Mode{}".format(
        _BOLD, _CYAN, __version__, _RESET, _DIM, _RESET,
    ))
    print("  {}Provider: {}{}{}  Model: {}{}{}  Tools: {}{}{}".format(
        _DIM, _RESET, config.provider or "openai", _DIM,
        _RESET, config.resolved_model, _DIM,
        _RESET, tool_count, _RESET,
    ))
    print()

    # readline support — up/down arrow history recall
    try:
        import readline
        import os as _os
        _history_dir = _os.path.expanduser("~/.flyto-ai")
        _os.makedirs(_history_dir, exist_ok=True)
        _history_file = _os.path.join(_history_dir, "history")
        try:
            readline.read_history_file(_history_file)
        except FileNotFoundError:
            pass
        readline.set_history_length(500)
        readline.parse_and_bind("tab: complete")
    except ImportError:
        _history_file = None

    def _save_history():
        if _history_file:
            try:
                readline.write_history_file(_history_file)
            except Exception:
                pass

    mode = "execute"

    def _status_text():
        """Build colored status content."""
        sep = " {}\u00b7{} ".format(_DIM, _RESET)
        if mode == "execute":
            mode_part = "{}\u23f5\u23f5 execute{}".format(_GREEN, _RESET)
        else:
            mode_part = "{}\u25b7\u25b7 plan-only{}".format(_YELLOW, _RESET)
        provider_part = "{}/{}".format(config.provider or "openai", config.resolved_model)
        tools_part = "{}{} tools{}".format(_YELLOW, tool_count, _RESET)
        parts = [mode_part, provider_part, tools_part]
        if history:
            parts.append("{}{} msgs{}".format(_DIM, len(history) // 2, _RESET))
        return sep.join(parts)

    def _draw_box():
        """Draw input box + status. Returns after cursor is positioned for input."""
        w = _term_width()
        # Layout: top → (blank) → bottom → status, then cursor back to blank
        sys.stdout.write("{}╭{}╮{}\n".format(_P1, "─" * (w - 2), _RESET))
        sys.stdout.write("\n")  # blank line for input
        sys.stdout.write("{}╰{}╯{}\n".format(_P4, "─" * (w - 2), _RESET))
        sys.stdout.write("  {}\n".format(_status_text()))
        sys.stdout.write("\033[3A\r")  # cursor back up to blank input line
        sys.stdout.flush()

    def _erase_box(user_text=""):
        """After Enter: erase box frame, leave user text as chat message."""
        # Cursor is on bottom border line (C). Lines:
        # A(top)  B(input)  C(bottom=cursor)  D(status)
        sys.stdout.write("\033[2A\r\033[K")          # → A, clear top border
        if user_text:
            sys.stdout.write("{}❯{} {}\n".format(    # rewrite A as user msg
                _CYAN, _RESET, user_text))
            sys.stdout.write("\033[K")                # clear B (old input)
        else:
            sys.stdout.write("\n\033[K")              # A blank, clear B
        sys.stdout.write("\n\033[K")                  # → C, clear bottom
        sys.stdout.write("\n\033[K")                  # → D, clear status
        sys.stdout.flush()

    # Persistent event loop — keeps browser sessions alive across messages
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
      while True:
        # ── Draw input box ────────────────────────────────────
        _draw_box()
        try:
            user_input = input(
                "{}│{} {}❯{} ".format(_P2, _RESET, _CYAN, _RESET),
            ).strip()
        except (EOFError, KeyboardInterrupt):
            _erase_box()
            print("{}Bye!{}".format(_DIM, _RESET))
            _save_history()
            break

        # Erase box, keep user text as chat message (or blank if empty)
        _erase_box(user_input)

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            if cmd in ("/exit", "/quit", "/q"):
                print("{}Bye!{}".format(_DIM, _RESET))
                _save_history()
                break
            elif cmd in ("/clear", "/reset"):
                history.clear()
                print("{}Conversation cleared.{}".format(_DIM, _RESET))
                continue
            elif cmd == "/mode":
                mode = "yaml" if mode == "execute" else "execute"
                label = "execute (run modules)" if mode == "execute" else "plan-only (YAML output)"
                print("{}Switched to: {}{}".format(_DIM, label, _RESET))
                continue
            elif cmd == "/help":
                print()
                print("  {}/clear{}   — Reset conversation".format(_CYAN, _RESET))
                print("  {}/mode{}    — Toggle execute / plan-only".format(_CYAN, _RESET))
                print("  {}/history{} — Show message count".format(_CYAN, _RESET))
                print("  {}/version{} — Show version info".format(_CYAN, _RESET))
                print("  {}/exit{}    — Quit".format(_CYAN, _RESET))
                print()
                continue
            elif cmd == "/version":
                _cmd_version()
                continue
            else:
                print("{}Unknown command: {}{}".format(_DIM, cmd, _RESET))
                continue

        # ── Response ──────────────────────────────────────────

        # Live tool progress
        _tool_count = [0]

        def _on_tool_call(func_name, func_args):
            _tool_count[0] += 1
            if func_name == "execute_module":
                label = func_args.get("module_id", func_name)
            else:
                label = func_name
            sys.stdout.write("\r\033[K")
            sys.stdout.write(
                "  {}\u25cb {}{}".format(_DIM, label, _RESET),
            )
            sys.stdout.flush()

        sys.stdout.write(
            "  {}\u25cb Thinking...{}".format(_DIM, _RESET),
        )
        sys.stdout.flush()

        result = loop.run_until_complete(agent.chat(
            user_input, history=history, mode=mode,
            on_tool_call=_on_tool_call,
        ))

        # Clear progress line
        sys.stdout.write("\r\033[K")

        if result.ok:
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result.message})

            # Show executed modules as compact list
            if result.execution_results:
                for er in result.execution_results:
                    mid = er.get("module_id", "")
                    ok = er.get("ok", True)
                    icon = "{}\u2713{}".format(_GREEN, _RESET) if ok else "{}\u2717{}".format(_YELLOW, _RESET)
                    print("  {} {}".format(icon, mid))
                print()

            # Show response (truncate verbose error dumps)
            msg_lines = result.message.split("\n")
            if len(msg_lines) > 20:
                for line in msg_lines[:5]:
                    print("  {}".format(line))
                print("  {}... ({} lines truncated){}".format(
                    _DIM, len(msg_lines) - 5, _RESET))
            else:
                for line in msg_lines:
                    print("  {}".format(line))

            meta_parts = []
            if result.execution_results:
                meta_parts.append("{} executed".format(len(result.execution_results)))
            if result.tool_calls:
                meta_parts.append("{} tool calls".format(len(result.tool_calls)))
            if meta_parts:
                print()
                print("  {}{}{}".format(_DIM, " \u00b7 ".join(meta_parts), _RESET))
        else:
            err_msg = result.error or result.message
            # Truncate long error messages
            err_lines = err_msg.split("\n")
            if len(err_lines) > 3:
                err_msg = "\n".join(err_lines[:3]) + "\n..."
            print("  {}Error: {}{}".format(_YELLOW, err_msg, _RESET))

        print()
    finally:
        loop.close()


def _cmd_chat(args):
    from flyto_ai import Agent, AgentConfig

    config = AgentConfig.from_env()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.api_key:
        config.api_key = args.api_key

    agent = Agent(config=config)
    message = " ".join(args.message)
    mode = "yaml" if getattr(args, "plan", False) else "execute"
    result = asyncio.run(agent.chat(message, mode=mode))

    if args.json:
        print(result.model_dump_json(indent=2))
    elif result.ok:
        print(result.message)
    else:
        print("Error: {}".format(result.error or result.message), file=sys.stderr)
        sys.exit(1)

    # Webhook: POST result
    if args.webhook:
        _post_webhook(args.webhook, result)


def _post_webhook(url, result):
    """POST chat result to a webhook URL."""
    payload = json.dumps(result.model_dump(), ensure_ascii=False, default=str).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        resp = urlopen(req, timeout=10)
        print("{}Webhook: {} {}{}".format(_DIM, resp.status, url, _RESET), file=sys.stderr)
    except URLError as e:
        print("{}Webhook failed: {} — {}{}".format(_YELLOW, url, e, _RESET), file=sys.stderr)


def _cmd_serve(args):
    """Start a lightweight HTTP server that accepts POST /chat to trigger the agent."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from flyto_ai import Agent, AgentConfig

    config = AgentConfig.from_env()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.api_key:
        config.api_key = args.api_key

    agent = Agent(config=config)

    # Persistent event loop — keeps browser sessions alive across requests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path not in ("/chat", "/api/chat"):
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            message = body.get("message", "")
            if not message:
                self._json_response(400, {"ok": False, "error": "Missing 'message' field"})
                return

            result = loop.run_until_complete(agent.chat(
                message,
                history=body.get("history"),
                template_context=body.get("template_context"),
                mode=body.get("mode", "execute"),
            ))
            self._json_response(200, result.model_dump())

        def do_GET(self):
            if self.path in ("/health", "/"):
                self._json_response(200, {"ok": True, "status": "ready"})
            else:
                self.send_error(404)

        def _json_response(self, code, data):
            payload = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *a):
            # Colorized request log
            sys.stderr.write("  {}[{}]{} {}\n".format(
                _DIM, self.log_date_time_string(), _RESET, fmt % a,
            ))

    server = HTTPServer((args.host, args.port), Handler)
    print()
    print("  {}{}Flyto2 AI Server{}".format(_BOLD, _CYAN, _RESET))
    print("  Listening on {}http://{}:{}{}".format(_GREEN, args.host, args.port, _RESET))
    print()
    print("  {}POST /chat{}  {}{\"message\": \"scrape example.com\"}{}".format(
        _BOLD, _RESET, _DIM, _RESET,
    ))
    print("  {}GET  /health{}".format(_BOLD, _RESET))
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  {}Shutting down.{}".format(_DIM, _RESET))
        server.server_close()


if __name__ == "__main__":
    main()
