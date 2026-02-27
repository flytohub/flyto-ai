# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""flyto-ai CLI — natural language automation from the terminal."""
import argparse
import asyncio
import importlib.resources
import json
import sys
import time
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
    chat_p.add_argument("--max-rounds", type=int, help="Max tool call rounds (default: from config)")
    chat_p.add_argument("--webhook", "-w", help="POST result to this webhook URL")
    chat_p.add_argument("--no-memory", action="store_true", help="Disable memory system")
    chat_p.add_argument("--sandbox", action="store_true", help="Enable Docker sandbox for dangerous modules")

    # flyto-ai version
    sub.add_parser("version", help="Show version and optional dependency status")

    # flyto-ai blueprints
    bp_p = sub.add_parser("blueprints", help="List and export learned blueprints")
    bp_p.add_argument("--export", action="store_true", help="Export top blueprints as YAML")
    bp_p.add_argument("--min-score", type=int, default=0, help="Minimum score to show (default: 0)")

    # flyto-ai serve
    serve_p = sub.add_parser("serve", help="Start HTTP server for webhook triggers")
    serve_p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve_p.add_argument("--port", type=int, default=7411, help="Bind port (default: 7411)")
    serve_p.add_argument("--provider", "-p", help="LLM provider")
    serve_p.add_argument("--model", "-m", help="Model name")
    serve_p.add_argument("--api-key", "-k", help="API key (or use env vars)")
    serve_p.add_argument("--dir", default=None, help="Working directory for /claude commands")

    # flyto-ai mcp
    sub.add_parser("mcp", help="Start MCP server (JSON-RPC 2.0 over STDIO)")

    # flyto-ai prompt-lab
    lab_p = sub.add_parser("prompt-lab", help="Prompt evolution system — eval, evolve, report")
    lab_sub = lab_p.add_subparsers(dest="lab_command")
    lab_eval = lab_sub.add_parser("eval", help="Evaluate current prompt against test cases")
    lab_eval.add_argument("--cases", help="Path to eval cases YAML")
    lab_eval.add_argument("--rubric", help="Path to rubric YAML")
    lab_eval.add_argument("--json", action="store_true", help="Output JSON")
    lab_evolve = lab_sub.add_parser("evolve", help="Run evolution loop")
    lab_evolve.add_argument("--cases", help="Path to eval cases YAML")
    lab_evolve.add_argument("--rubric", help="Path to rubric YAML")
    lab_evolve.add_argument("--generations", type=int, help="Max generations")
    lab_evolve.add_argument("--population", type=int, help="Candidates per generation")
    lab_evolve.add_argument("--output", help="Output directory for results")
    lab_evolve.add_argument("--provider", "-p", help="LLM provider")
    lab_evolve.add_argument("--api-key", "-k", help="API key")
    lab_sub.add_parser("cases", help="List all eval cases")
    lab_sub.add_parser("report", help="Show latest evolution report")

    # flyto-ai code "fix login page"
    code_p = sub.add_parser("code", help="AI-driven coding with Claude Code Agent + verification")
    code_p.add_argument("message", nargs="+", help="What you want to build/fix")
    code_p.add_argument("--dir", "-d", default=".", help="Project working directory (default: .)")
    code_p.add_argument("--verify", help="Verification recipe name (e.g. 'screenshot')")
    code_p.add_argument("--verify-args", help="Recipe args as JSON string")
    code_p.add_argument("--reference", help="Reference image path for visual comparison")
    code_p.add_argument("--max-attempts", type=int, default=3, help="Max fix attempts (default: 3)")
    code_p.add_argument("--budget", type=float, default=5.0, help="Max budget in USD (default: 5.0)")
    code_p.add_argument("--max-turns", type=int, default=30, help="Max Claude Code turns (default: 30)")
    code_p.add_argument("--json", action="store_true", help="Output raw JSON")

    # flyto-ai (interactive mode, no subcommand)
    sub.add_parser("interactive", help="Start interactive chat (default when no args)")

    args = parser.parse_args()

    if args.command == "version":
        _cmd_version()
    elif args.command == "blueprints":
        _cmd_blueprints(args)
    elif args.command == "serve":
        _cmd_serve(args)
    elif args.command == "mcp":
        _cmd_mcp()
    elif args.command == "prompt-lab":
        _cmd_prompt_lab(args)
    elif args.command == "code":
        _cmd_code(args)
    elif args.command == "chat":
        _cmd_chat(args)
    elif args.command == "interactive" or (args.command is None and sys.stdin.isatty()):
        _cmd_interactive(args)
    elif args.command is None and not sys.stdin.isatty() and not sys.stdin.closed:
        # Pipe mode: echo "scrape example.com" | flyto-ai
        try:
            _cmd_pipe(args)
        except OSError:
            parser.print_help()
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


def _cmd_code(args):
    """Run Claude Code Agent with optional verification loop."""
    import os

    # Early check: claude-agent-sdk installed?
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        print("\033[31mError:\033[0m claude-agent-sdk is required for the 'code' command.")
        print("Install with: \033[1mpip install flyto-ai[agent]\033[0m")
        sys.exit(1)

    message = " ".join(args.message)
    working_dir = os.path.abspath(args.dir)

    verify_args = {}
    if args.verify_args:
        verify_args = json.loads(args.verify_args)

    from flyto_ai.config import AgentConfig
    from flyto_ai.agents.claude_code import ClaudeCodeAgent
    from flyto_ai.agents.models import CodeTaskRequest

    config = AgentConfig.from_env()
    agent = ClaudeCodeAgent(config=config)

    request = CodeTaskRequest(
        message=message,
        working_dir=working_dir,
        verification_recipe=args.verify or None,
        verification_args=verify_args,
        reference_image=args.reference or None,
        max_fix_attempts=args.max_attempts,
        max_budget_usd=args.budget,
        max_turns=args.max_turns,
    )

    # Streaming callback
    def _on_stream(event):
        if getattr(args, "json", False):
            return
        etype = event.get("type", "")
        if etype == "phase_start":
            phase = event.get("phase", "")
            attempt = event.get("attempt", "")
            label = phase
            if attempt:
                label = "{} (attempt {})".format(phase, attempt)
            sys.stdout.write("\033[36m▶ {}\033[0m\n".format(label))
            sys.stdout.flush()
        elif etype == "phase_end":
            phase = event.get("phase", "")
            sys.stdout.write("\033[36m◀ {} done\033[0m\n".format(phase))
            sys.stdout.flush()
        elif etype == "token":
            sys.stdout.write(event.get("content", ""))
            sys.stdout.flush()
        elif etype == "verification_result":
            passed = event.get("passed", False)
            icon = "\033[32m✓\033[0m" if passed else "\033[31m✗\033[0m"
            summary = event.get("summary") or event.get("error") or ""
            sys.stdout.write("\n{} Verification: {}\n".format(icon, summary or ("passed" if passed else "failed")))
            sys.stdout.flush()

    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(agent.run(request, on_stream=_on_stream))
    finally:
        loop.close()

    if getattr(args, "json", False):
        import dataclasses
        out = dataclasses.asdict(result)
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    else:
        icon = "\033[32m✓\033[0m" if result.ok else "\033[31m✗\033[0m"
        print("\n{} {} (attempts: {}, cost: ${:.4f})".format(
            icon, result.message, result.attempts, result.total_cost_usd,
        ))
        if result.files_changed:
            print("  Files changed: {}".format(", ".join(result.files_changed)))
        # Claude SDK details
        if result.claude_session_id:
            dur_s = result.claude_duration_ms / 1000 if result.claude_duration_ms else 0
            print("  Claude: session={}, turns={}, duration={:.1f}s".format(
                result.claude_session_id, result.claude_num_turns, dur_s,
            ))
        if result.claude_usage:
            u = result.claude_usage
            print("  Usage: input={}, output={}, cache_read={}, cache_create={}".format(
                u.get("input_tokens", 0), u.get("output_tokens", 0),
                u.get("cache_read_input_tokens", 0), u.get("cache_creation_input_tokens", 0),
            ))

    sys.exit(0 if result.ok else 1)


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

# Rich Markdown rendering (optional)
_rich_console = None
try:
    from rich.console import Console as _RichConsole
    from rich.markdown import Markdown as _RichMarkdown
    _rich_console = _RichConsole()
except ImportError:
    pass


def _print_markdown(text: str) -> None:
    """Render markdown with rich if available, otherwise plain print."""
    if _rich_console:
        _rich_console.print(_RichMarkdown(text))
    else:
        for line in text.split("\n"):
            print("  {}".format(line))


# UI helpers
import re as _re
import shutil as _shutil
_ANSI_RE = _re.compile(r'\033\[[0-9;]*m')

# ---------------------------------------------------------------------------
# Shared server helpers — rate limiter, auth, CORS
# ---------------------------------------------------------------------------

_RATE_LIMIT = 10     # requests per window
_RATE_BURST = 3      # max burst
_RATE_WINDOW = 60    # seconds
_RATE_STALE = 600    # cleanup entries older than 10 min


class _RateLimiter:
    """Token bucket rate limiter per IP with periodic cleanup."""

    def __init__(self):
        self._buckets = {}  # ip -> [tokens, last_refill_time]

    def check(self, ip: str) -> bool:
        now = time.monotonic()
        if ip not in self._buckets:
            self._buckets[ip] = [_RATE_BURST - 1, now]
            return True
        bucket = self._buckets[ip]
        elapsed = now - bucket[1]
        refill = elapsed * (_RATE_LIMIT / _RATE_WINDOW)
        bucket[0] = min(_RATE_BURST, bucket[0] + refill)
        bucket[1] = now
        if bucket[0] < 1:
            return False
        bucket[0] -= 1
        return True

    def cleanup(self):
        """Remove stale entries to prevent memory leak."""
        now = time.monotonic()
        stale = [ip for ip, (_, ts) in self._buckets.items() if now - ts > _RATE_STALE]
        for ip in stale:
            del self._buckets[ip]


# Server API key auth (optional — set FLYTO_AI_SERVER_KEY to enable)
import os as _os
_SERVER_KEY = _os.getenv("FLYTO_AI_SERVER_KEY", "")

# CORS allowed origins (set FLYTO_AI_CORS_ORIGINS=http://localhost:3000,https://app.flyto.dev)
_CORS_ORIGINS_RAW = _os.getenv("FLYTO_AI_CORS_ORIGINS", "")
_CORS_ORIGINS = frozenset(o.strip() for o in _CORS_ORIGINS_RAW.split(",") if o.strip()) if _CORS_ORIGINS_RAW else None

# Telegram Bot gateway (set TELEGRAM_BOT_TOKEN to enable /telegram webhook)
_TG_TOKEN = _os.getenv("TELEGRAM_BOT_TOKEN", "")
_TG_ALLOWED_RAW = _os.getenv("TELEGRAM_ALLOWED_CHATS", "")
_TG_ALLOWED_CHATS = frozenset(int(c.strip()) for c in _TG_ALLOWED_RAW.split(",") if c.strip()) if _TG_ALLOWED_RAW else frozenset()


async def _tg_send(token: str, chat_id: int, text: str):
    """Send a message via Telegram Bot API."""
    import aiohttp
    url = "https://api.telegram.org/bot{}/sendMessage".format(token)
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    # Retry without parse_mode in case Markdown is invalid
                    payload.pop("parse_mode", None)
                    async with session.post(url, json=payload) as _:
                        pass
    except Exception:
        pass  # Best-effort — don't crash the webhook


_TG_HELP_TEXT = (
    "Available commands:\n"
    "\n"
    "  (plain text) — flyto-ai agent (automation)\n"
    "  /claude <msg> — Claude Code (read/write code)\n"
    "  /yaml — list learned blueprints\n"
    "  /blueprint — same as /yaml\n"
    "  /help — show this message"
)


def _tg_list_blueprints() -> str:
    """List top blueprints from flyto-blueprint engine."""
    try:
        from flyto_blueprint import get_engine
        engine = get_engine()
        bps = engine.list_blueprints()
    except Exception as e:
        return "Error loading blueprints: {}".format(e)

    if not bps:
        return "No blueprints yet."

    lines = []
    sorted_bps = sorted(bps, key=lambda b: b.get("score", 0), reverse=True)[:10]
    for bp in sorted_bps:
        name = bp.get("name", "?")
        score = bp.get("score", 0)
        lines.append("  {} (score: {})".format(name, score))
    return "Blueprints:\n" + "\n".join(lines)


async def _tg_run_claude(msg: str, working_dir: str, config) -> str:
    """Run a task via ClaudeCodeAgent and return formatted reply."""
    try:
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        from flyto_ai.agents.models import CodeTaskRequest

        cc_agent = ClaudeCodeAgent(config=config)
        request = CodeTaskRequest(message=msg, working_dir=working_dir)
        result = await cc_agent.run(request)

        parts = []
        if result.ok:
            parts.append("Done")
        else:
            parts.append("Failed")
        if result.message:
            parts.append(result.message)
        if result.files_changed:
            parts.append("Files: " + ", ".join(result.files_changed))
        return "\n".join(parts)
    except Exception as e:
        return "Claude error: {}".format(e)


def _check_server_auth(auth_header: str) -> bool:
    """Check Bearer token against FLYTO_AI_SERVER_KEY. Returns True if auth passes."""
    if not _SERVER_KEY:
        return True  # no key configured = open access
    if not auth_header:
        return False
    if auth_header.startswith("Bearer "):
        return auth_header[7:] == _SERVER_KEY
    return auth_header == _SERVER_KEY


def _get_cors_origin(request_origin: str) -> str:
    """Return allowed CORS origin. Uses whitelist if configured, else '*'."""
    if _CORS_ORIGINS is None:
        return "*"
    # Sanitize to prevent HTTP response splitting (CWE-113)
    sanitized = request_origin.replace("\r", "").replace("\n", "")
    if sanitized in _CORS_ORIGINS:
        return sanitized
    return ""


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


def _cmd_prompt_lab(args):
    """Prompt evolution system — eval, evolve, report, cases."""
    lab_cmd = getattr(args, "lab_command", None)

    if lab_cmd == "eval":
        _prompt_lab_eval(args)
    elif lab_cmd == "evolve":
        _prompt_lab_evolve(args)
    elif lab_cmd == "cases":
        _prompt_lab_cases(args)
    elif lab_cmd == "report":
        _prompt_lab_report(args)
    else:
        print()
        print("  {}{}Prompt Lab{} — System Prompt Evolution".format(_BOLD, _CYAN, _RESET))
        print()
        print("  {}eval{}     Evaluate current prompt against test cases".format(_CYAN, _RESET))
        print("  {}evolve{}   Run evolution loop (generate → eval → select)".format(_CYAN, _RESET))
        print("  {}cases{}    List all eval test cases".format(_CYAN, _RESET))
        print("  {}report{}   Show latest evolution report".format(_CYAN, _RESET))
        print()
        print("  {}Example: flyto-ai prompt-lab eval{}".format(_DIM, _RESET))
        print()


def _prompt_lab_eval(args):
    """Run eval on current baseline prompt (rule-based, no API key needed)."""
    from flyto_ai.evolution.blocks import get_baseline_candidate
    from flyto_ai.evolution.runner import load_eval_cases, load_rubric, format_score_report, _generate_mock_response
    from flyto_ai.evolution.scorer import score_response

    cases = load_eval_cases(getattr(args, "cases", None))
    config = load_rubric(getattr(args, "rubric", None))
    baseline = get_baseline_candidate(config.mode)

    print()
    print("  {}{}Prompt Lab — Eval{} ".format(_BOLD, _CYAN, _RESET))
    print("  {}Baseline prompt · {} cases · rule-based scoring{}".format(
        _DIM, len(cases), _RESET,
    ))
    print()

    passed = 0
    failed = 0
    total_score = 0.0
    failures = []

    for case in cases:
        response = _generate_mock_response(case)
        score = score_response(case, response, case.mock_execution_results, config)
        total_score += score.total_score

        if score.passed:
            passed += 1
            icon = "{}\u2713{}".format(_GREEN, _RESET)
        else:
            failed += 1
            icon = "{}\u2717{}".format(_YELLOW, _RESET)
            failures.append((case, score))

        notes_str = ""
        if score.notes:
            notes_str = "  {}{}{}".format(_DIM, "; ".join(score.notes[:2]), _RESET)
        print("  {} {:<30} {:.0f}/100{}".format(
            icon, case.id[:30], score.total_score, notes_str,
        ))

    avg = total_score / len(cases) if cases else 0
    print()
    print("  {}Results: {} passed · {} failed · avg {:.1f}/100{}".format(
        _BOLD, passed, failed, avg, _RESET,
    ))

    if failures:
        print()
        print("  {}Failed cases:{}".format(_YELLOW, _RESET))
        for case, score in failures[:5]:
            notes = "; ".join(score.notes[:2]) if score.notes else ""
            print("    {} — task={:.1f} comp={:.1f} pen={:.0f}  {}{}{}".format(
                case.id, score.task_score, score.compliance_score,
                score.penalties, _DIM, notes, _RESET,
            ))

    if getattr(args, "json", False):
        import json as _json
        data = {
            "passed": passed, "failed": failed, "avg_score": round(avg, 1),
            "cases": len(cases),
        }
        print()
        print(_json.dumps(data, indent=2))

    print()


def _prompt_lab_evolve(args):
    """Run the evolution loop."""
    from flyto_ai.evolution.loop import EvolutionLoop, format_evolution_report
    from flyto_ai.evolution.runner import load_rubric

    config = load_rubric(getattr(args, "rubric", None))
    if getattr(args, "generations", None):
        config.generations = args.generations
    if getattr(args, "population", None):
        config.population_size = args.population

    # Optional LLM provider for live eval
    provider = None
    api_key = getattr(args, "api_key", None)
    prov_name = getattr(args, "provider", None)
    if api_key or prov_name:
        from flyto_ai.config import AgentConfig
        from flyto_ai.providers import create_provider
        cfg = AgentConfig.from_env()
        if api_key:
            cfg.api_key = api_key
        if prov_name:
            cfg.provider = prov_name
        provider = create_provider(
            cfg.provider or "openai",
            api_key=cfg.api_key,
            model=cfg.resolved_model,
        )

    output = getattr(args, "output", None) or "eval/results"

    print()
    print("  {}{}Prompt Lab — Evolution{} ".format(_BOLD, _CYAN, _RESET))
    print("  {}generations={} · population={} · {}{}".format(
        _DIM, config.generations, config.population_size,
        "LLM eval" if provider else "rule-based eval", _RESET,
    ))
    print()

    loop_obj = EvolutionLoop(
        config=config, provider=provider,
        eval_cases_path=getattr(args, "cases", None),
        output_dir=output,
    )

    def _on_progress(gen, cid, score):
        print("  {}gen {} · {} · {:.1f}{}".format(_DIM, gen, cid[:20], score, _RESET))

    report = asyncio.run(loop_obj.run(on_progress=_on_progress))

    print()
    print(format_evolution_report(report))
    print()
    print("  {}Results saved to: {}{}".format(_DIM, output, _RESET))
    print()


def _prompt_lab_cases(args):
    """List all eval test cases."""
    from flyto_ai.evolution.runner import load_eval_cases

    cases = load_eval_cases()

    print()
    print("  {}{}Prompt Lab — {} Eval Cases{}".format(_BOLD, _CYAN, len(cases), _RESET))
    print()

    by_category = {}
    for c in cases:
        by_category.setdefault(c.category, []).append(c)

    for cat, cat_cases in sorted(by_category.items()):
        print("  {}{}{}  ({})".format(_BOLD, cat, _RESET, len(cat_cases)))
        for c in cat_cases:
            tags = " ".join("[{}]".format(t) for t in c.tags[:3])
            weight_str = " {}w={:.0f}{}".format(_YELLOW, c.weight, _RESET) if c.weight > 1 else ""
            print("    {} {}{}{}".format(c.id, _DIM, tags, _RESET) + weight_str)
        print()


def _prompt_lab_report(args):
    """Show the latest evolution report."""
    from pathlib import Path

    results_dir = Path("eval/results")
    report_file = results_dir / "report.json"

    if not report_file.exists():
        print("  {}No evolution report found. Run: flyto-ai prompt-lab evolve{}".format(
            _DIM, _RESET,
        ))
        return

    import json as _json
    from flyto_ai.evolution.models import EvolutionReport
    from flyto_ai.evolution.loop import format_evolution_report

    with open(report_file, "r", encoding="utf-8") as f:
        data = _json.load(f)

    report = EvolutionReport(**data)
    print()
    print(format_evolution_report(report))
    print()


def _cmd_mcp():
    """Start MCP server (JSON-RPC 2.0 over STDIO)."""
    from flyto_ai.mcp_server import main as mcp_main
    mcp_main()


def _handle_memory_cmd(subcmd, args_list, agent, loop):
    """Handle /memory subcommands: list, search <query>, clear."""
    store = agent.memory_store
    if not store:
        print("  {}Memory not enabled.{}".format(_DIM, _RESET))
        return

    if subcmd == "list":
        sessions = loop.run_until_complete(store.list_sessions())
        if not sessions:
            print("  {}No memory sessions.{}".format(_DIM, _RESET))
            return
        print()
        for s in sessions:
            print("  {}{}{}  {} msgs  {}last: {:.0f}s ago{}".format(
                _CYAN, s["session_id"], _RESET,
                s["message_count"],
                _DIM, __import__("time").time() - s["updated_at"], _RESET,
            ))
        print()

    elif subcmd == "search" and args_list:
        query = " ".join(args_list)
        search = agent.memory_search
        if not search:
            print("  {}Memory search not available.{}".format(_DIM, _RESET))
            return
        results = loop.run_until_complete(search.search(query, top_k=5))
        if not results:
            print("  {}No results for: {}{}".format(_DIM, query, _RESET))
            return
        print()
        for r in results:
            content = r["content"][:100].replace("\n", " ")
            print("  {:.4f}  {}{}{}".format(r["score"], _DIM, content, _RESET))
        print()

    elif subcmd == "clear":
        sessions = loop.run_until_complete(store.list_sessions())
        for s in sessions:
            loop.run_until_complete(store.delete_session(s["session_id"]))
        print("  {}Memory cleared ({} sessions).{}".format(_DIM, len(sessions), _RESET))

    else:
        print("  {}Usage: /memory [list|search <query>|clear]{}".format(_DIM, _RESET))


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
    tool_count = len(agent.tools)

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
            elif cmd == "/history":
                msg_count = len(history) // 2
                print("{}  {} message(s) in history{}".format(_DIM, msg_count, _RESET))
                if history:
                    for i in range(0, len(history), 2):
                        user_msg = history[i]["content"][:60] if i < len(history) else ""
                        print("  {}{}. {}{}".format(_DIM, i // 2 + 1, user_msg, _RESET))
                continue
            elif cmd == "/memory":
                parts = user_input.split()
                subcmd = parts[1] if len(parts) > 1 else "list"
                _handle_memory_cmd(subcmd, parts[2:] if len(parts) > 2 else [], agent, loop)
                continue
            elif cmd == "/help":
                print()
                print("  {}/clear{}   — Reset conversation".format(_CYAN, _RESET))
                print("  {}/mode{}    — Toggle execute / plan-only".format(_CYAN, _RESET))
                print("  {}/history{} — Show conversation history".format(_CYAN, _RESET))
                print("  {}/memory{}  — List, search, or clear memory".format(_CYAN, _RESET))
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

        # Streaming state
        _streamed_any = [False]  # did we receive at least one TOKEN?
        _stream_buf = []         # collect streamed tokens for dedup

        def _on_stream(event):
            from flyto_ai.models import StreamEventType
            if event.type == StreamEventType.TOKEN:
                if not _streamed_any[0]:
                    # First token replaces "Thinking..."
                    sys.stdout.write("\r\033[K  ")
                    _streamed_any[0] = True
                sys.stdout.write(event.content)
                sys.stdout.flush()
                _stream_buf.append(event.content)
            elif event.type == StreamEventType.TOOL_START:
                label = event.tool_args.get("module_id", event.tool_name) if event.tool_args else event.tool_name
                if event.tool_name == "execute_module" and event.tool_args:
                    label = event.tool_args.get("module_id", event.tool_name)
                else:
                    label = event.tool_name or ""
                # Clear current line and show tool indicator
                sys.stdout.write("\r\033[K")
                sys.stdout.write(
                    "  {}\u25cb {}{}".format(_DIM, label, _RESET),
                )
                sys.stdout.flush()
            elif event.type == StreamEventType.DONE:
                if _streamed_any[0]:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

        # Live tool progress (fallback when streaming not emitting TOOL_START)
        def _on_tool_call(func_name, func_args):
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
            on_stream=_on_stream,
        ))

        # Clear progress line (if no streaming happened)
        if not _streamed_any[0]:
            sys.stdout.write("\r\033[K")

        if result.ok:
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": result.message})
            # Keep last 40 messages (20 exchanges) to avoid unbounded growth
            if len(history) > 40:
                history[:] = history[-40:]

            # Show executed modules as compact list
            if result.execution_results:
                for er in result.execution_results:
                    mid = er.get("module_id", "")
                    ok = er.get("ok", True)
                    icon = "{}\u2713{}".format(_GREEN, _RESET) if ok else "{}\u2717{}".format(_YELLOW, _RESET)
                    if ok:
                        print("  {} {}".format(icon, mid))
                    else:
                        err_preview = er.get("result_preview", "")
                        # Try to extract error message from result_preview JSON
                        err_msg = ""
                        if err_preview:
                            try:
                                err_data = json.loads(err_preview) if err_preview.startswith("{") else {}
                                err_msg = err_data.get("error", "")
                            except Exception:
                                pass
                        if err_msg:
                            print("  {} {}  {}{}{}".format(icon, mid, _DIM, err_msg[:80], _RESET))
                        else:
                            print("  {} {}".format(icon, mid))
                print()

            # Show response — skip if already streamed
            if not _streamed_any[0]:
                _print_markdown(result.message)

            meta_parts = []
            if result.execution_results:
                meta_parts.append("{} executed".format(len(result.execution_results)))
            if result.tool_calls:
                meta_parts.append("{} tool calls".format(len(result.tool_calls)))
            if result.usage and result.usage.total_tokens > 0:
                meta_parts.append("{} tokens".format(result.usage.total_tokens))
            if result.rounds_used > 0:
                meta_parts.append("{} rounds".format(result.rounds_used))
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


def _cmd_pipe(args):
    """Handle piped stdin: echo "scrape example.com" | flyto-ai"""
    from flyto_ai import Agent, AgentConfig

    message = sys.stdin.read().strip()
    if not message:
        return

    config = AgentConfig.from_env()
    agent = Agent(config=config)
    result = asyncio.run(agent.chat(message, mode="execute"))

    if result.ok:
        print(result.message)
    else:
        print("Error: {}".format(result.error or result.message), file=sys.stderr)
        sys.exit(1)


def _cmd_chat(args):
    from flyto_ai import Agent, AgentConfig

    config = AgentConfig.from_env()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.api_key:
        config.api_key = args.api_key
    if getattr(args, "max_rounds", None):
        config.max_tool_rounds = args.max_rounds
    if getattr(args, "no_memory", False):
        config.enable_memory = False
    if getattr(args, "sandbox", False):
        config.enable_sandbox = True

    agent = Agent(config=config)
    message = " ".join(args.message)
    mode = "yaml" if getattr(args, "plan", False) else "execute"

    # Streaming callback for non-JSON output
    _streamed = [False]

    def _on_stream(event):
        if args.json:
            return  # JSON mode: no streaming output
        from flyto_ai.models import StreamEventType
        if event.type == StreamEventType.TOKEN:
            sys.stdout.write(event.content)
            sys.stdout.flush()
            _streamed[0] = True
        elif event.type == StreamEventType.TOOL_START:
            label = event.tool_name or ""
            if event.tool_name == "execute_module" and event.tool_args:
                label = event.tool_args.get("module_id", label)
            sys.stderr.write("{}\u25cb {}{}\n".format(_DIM, label, _RESET))
        elif event.type == StreamEventType.DONE:
            if _streamed[0]:
                sys.stdout.write("\n")
                sys.stdout.flush()

    result = asyncio.run(agent.chat(message, mode=mode, on_stream=_on_stream))

    if args.json:
        print(result.model_dump_json(indent=2))
    elif result.ok:
        if not _streamed[0]:
            print(result.message)
        # Show token usage + rounds as meta line
        meta_parts = []
        if result.usage and result.usage.total_tokens > 0:
            meta_parts.append("{} tokens".format(result.usage.total_tokens))
        if result.rounds_used > 0:
            meta_parts.append("{} rounds".format(result.rounds_used))
        if meta_parts:
            print("{}{}{}".format(_DIM, " | ".join(meta_parts), _RESET), file=sys.stderr)
    else:
        print("Error: {}".format(result.error or result.message), file=sys.stderr)
        sys.exit(1)

    # Webhook: POST result
    if args.webhook:
        _post_webhook(args.webhook, result)


def _post_webhook(url, result):
    """POST chat result to a webhook URL (user-specified, not SSRF-protected)."""
    payload = json.dumps(result.model_dump(), ensure_ascii=False, default=str).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        resp = urlopen(req, timeout=10)
        print("{}Webhook: {} {}{}".format(_DIM, resp.status, url, _RESET), file=sys.stderr)
    except (URLError, ValueError, OSError) as e:
        print("{}Webhook failed: {} — {}{}".format(_YELLOW, url, e, _RESET), file=sys.stderr)


def _cmd_serve(args):
    """Start HTTP server. Uses aiohttp if installed, otherwise stdlib fallback."""
    try:
        import aiohttp  # noqa: F401
        _cmd_serve_aiohttp(args)
    except ImportError:
        _cmd_serve_stdlib(args)


def _cmd_serve_aiohttp(args):
    """Async HTTP server using aiohttp — native async, supports concurrent requests."""
    import aiohttp.web
    from aiohttp import web
    from flyto_ai import Agent, AgentConfig
    from flyto_ai.models import ChatRequest

    config = AgentConfig.from_env()
    if args.provider:
        config.provider = args.provider
    if args.model:
        config.model = args.model
    if args.api_key:
        config.api_key = args.api_key

    agent = Agent(config=config)

    MAX_BODY_SIZE = 1_000_000
    _request_count = [0]
    _CLEANUP_INTERVAL = 50
    _limiter = _RateLimiter()

    def _cleanup_old(request):
        _request_count[0] += 1
        if _request_count[0] % _CLEANUP_INTERVAL == 0:
            _limiter.cleanup()
            try:
                from flyto_ai.tools.core_tools import clear_browser_sessions
                clear_browser_sessions()
            except Exception:
                pass

    # --- CORS middleware ---
    @web.middleware
    async def cors_middleware(request, handler):
        if request.method == "OPTIONS":
            resp = web.Response()
        else:
            resp = await handler(request)
        origin = _get_cors_origin(request.headers.get("Origin", ""))
        if origin:
            resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return resp

    # --- Serve demo.html ---
    async def handle_demo(request):
        html_path = importlib.resources.files("flyto_ai").joinpath("static/demo.html")
        return web.Response(
            text=html_path.read_text("utf-8"),
            content_type="text/html",
        )

    # --- Chat (batch) ---
    async def handle_chat(request):
        _cleanup_old(request)
        if not _check_server_auth(request.headers.get("Authorization", "")):
            return web.json_response(
                {"ok": False, "error": "Unauthorized. Set Authorization: Bearer <key>"}, status=401,
            )
        ip = request.remote or "unknown"
        if not _limiter.check(ip):
            return web.json_response(
                {"ok": False, "error": "Rate limited — try again in a moment"}, status=429,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"ok": False, "error": "Invalid JSON body"}, status=400,
            )

        try:
            req = ChatRequest.model_validate(body)
        except Exception as e:
            return web.json_response(
                {"ok": False, "error": "Invalid request: {}".format(e)}, status=400,
            )

        history_dicts = None
        if req.history:
            history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

        result = await agent.chat(
            req.message,
            history=history_dicts,
            template_context=req.template_context,
            mode=body.get("mode", "execute"),
        )
        return web.json_response(result.model_dump(), dumps=lambda x: json.dumps(x, ensure_ascii=False, default=str))

    # --- Chat SSE streaming ---
    async def handle_chat_stream(request):
        _cleanup_old(request)
        if not _check_server_auth(request.headers.get("Authorization", "")):
            return web.json_response(
                {"ok": False, "error": "Unauthorized. Set Authorization: Bearer <key>"}, status=401,
            )
        ip = request.remote or "unknown"
        if not _limiter.check(ip):
            return web.json_response(
                {"ok": False, "error": "Rate limited — try again in a moment"}, status=429,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response(
                {"ok": False, "error": "Invalid JSON body"}, status=400,
            )

        try:
            req = ChatRequest.model_validate(body)
        except Exception as e:
            return web.json_response(
                {"ok": False, "error": "Invalid request: {}".format(e)}, status=400,
            )

        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        })
        await response.prepare(request)

        # Queue bridges sync on_stream callback → async SSE writer
        queue = asyncio.Queue()

        def on_stream(event):
            data = json.dumps({
                "type": event.type.value,
                "content": event.content,
                "tool_name": event.tool_name,
                "tool_args": event.tool_args,
                "tool_result": event.tool_result,
            }, ensure_ascii=False, default=str)
            queue.put_nowait("event: {}\ndata: {}\n\n".format(event.type.value, data))

        async def sse_writer():
            while True:
                sse = await queue.get()
                if sse is None:
                    break
                await response.write(sse.encode("utf-8"))

        writer_task = asyncio.ensure_future(sse_writer())

        history_dicts = None
        if req.history:
            history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

        result = await agent.chat(
            req.message,
            history=history_dicts,
            template_context=req.template_context,
            mode=body.get("mode", "execute"),
            on_stream=on_stream,
        )

        # Signal writer to stop, then send final result
        queue.put_nowait(None)
        await writer_task

        result_data = json.dumps(result.model_dump(), ensure_ascii=False, default=str)
        await response.write("event: result\ndata: {}\n\n".format(result_data).encode("utf-8"))
        await response.write_eof()
        return response

    async def handle_health(request):
        return web.json_response({"ok": True, "status": "ready"})

    # --- Telegram Bot webhook ---
    _claude_working_dir = getattr(args, "dir", None) or _os.getcwd()

    async def _tg_process(chat_id: int, text: str):
        """Background task: route command and send result to Telegram."""
        try:
            stripped = text.strip()
            if stripped == "/help":
                reply = _TG_HELP_TEXT
            elif stripped in ("/yaml", "/blueprint"):
                reply = _tg_list_blueprints()
            elif stripped.startswith("/claude "):
                msg = stripped[8:].strip()
                if not msg:
                    reply = "Usage: /claude <message>"
                else:
                    reply = await _tg_run_claude(msg, _claude_working_dir, config)
            elif stripped == "/claude":
                reply = "Usage: /claude <message>"
            else:
                # Default — flyto-ai agent
                result = await agent.chat(text, mode="execute")
                reply = result.message or "Done."
        except Exception as e:
            reply = "Error: {}".format(e)

        if len(reply) > 4000:
            reply = reply[:4000] + "\n\n... (truncated)"

        await _tg_send(_TG_TOKEN, chat_id, reply)

    async def handle_telegram(request):
        """Receive Telegram Update, spawn background task, return immediately."""
        if not _TG_TOKEN:
            return web.json_response(
                {"ok": False, "error": "TELEGRAM_BOT_TOKEN not set"}, status=503,
            )

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": True})

        msg = body.get("message") or body.get("edited_message") or {}
        text = msg.get("text", "")
        chat_id = msg.get("chat", {}).get("id")

        if not text or not chat_id:
            return web.json_response({"ok": True})

        # Whitelist check
        if _TG_ALLOWED_CHATS and chat_id not in _TG_ALLOWED_CHATS:
            return web.json_response({"ok": True})

        # Send "processing" indicator and spawn background task
        await _tg_send(_TG_TOKEN, chat_id, "\u23f3 Processing...")
        asyncio.ensure_future(_tg_process(chat_id, text))

        # Return immediately so Telegram doesn't timeout
        return web.json_response({"ok": True})

    app = web.Application(client_max_size=MAX_BODY_SIZE, middlewares=[cors_middleware])
    app.router.add_get("/", handle_demo)
    app.router.add_get("/demo", handle_demo)
    app.router.add_post("/chat", handle_chat)
    app.router.add_post("/api/chat", handle_chat)
    app.router.add_post("/chat/stream", handle_chat_stream)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/telegram", handle_telegram)

    print()
    print("  {}{}Flyto2 AI Server{}  {}(aiohttp){}".format(_BOLD, _CYAN, _RESET, _DIM, _RESET))
    print("  Listening on {}http://{}:{}{}".format(_GREEN, args.host, args.port, _RESET))
    print()
    print("  {}GET  /{}            Demo page".format(_BOLD, _RESET))
    print("  {}POST /chat{}        Chat API".format(_BOLD, _RESET))
    print("  {}POST /chat/stream{} Streaming SSE".format(_BOLD, _RESET))
    print("  {}GET  /health{}      Health check".format(_BOLD, _RESET))
    if _TG_TOKEN:
        print("  {}POST /telegram{}   Telegram Bot webhook".format(_BOLD, _RESET))
    print()

    web.run_app(app, host=args.host, port=args.port, print=lambda *a: None)


def _cmd_serve_stdlib(args):
    """Synchronous HTTP server using stdlib — fallback when aiohttp not installed."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from flyto_ai import Agent, AgentConfig
    from flyto_ai.models import ChatRequest

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

    MAX_BODY_SIZE = 1_000_000  # 1 MB
    _request_count = [0]
    _CLEANUP_INTERVAL = 50  # run session cleanup every N requests
    _limiter = _RateLimiter()

    from flyto_ai.tools.core_tools import clear_browser_sessions

    # Load demo HTML once
    _demo_html = importlib.resources.files("flyto_ai").joinpath("static/demo.html").read_text("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            # Periodic cleanup of stale browser sessions and rate limiter
            _request_count[0] += 1
            if _request_count[0] % _CLEANUP_INTERVAL == 0:
                _limiter.cleanup()
                clear_browser_sessions()

            # Auth check
            if not _check_server_auth(self.headers.get("Authorization", "")):
                self._json_response(401, {"ok": False, "error": "Unauthorized. Set Authorization: Bearer <key>"})
                return

            # Rate limit
            ip = self.client_address[0]
            if not _limiter.check(ip):
                self._json_response(429, {"ok": False, "error": "Rate limited — try again in a moment"})
                return

            if self.path not in ("/chat", "/api/chat", "/chat/stream"):
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_BODY_SIZE:
                self._json_response(400, {"ok": False, "error": "Invalid Content-Length"})
                return
            try:
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, ValueError):
                self._json_response(400, {"ok": False, "error": "Invalid JSON body"})
                return

            try:
                req = ChatRequest.model_validate(body)
            except Exception as e:
                self._json_response(400, {"ok": False, "error": "Invalid request: {}".format(e)})
                return

            history_dicts = None
            if req.history:
                history_dicts = [{"role": m.role, "content": m.content} for m in req.history]

            # SSE streaming for /chat/stream
            if self.path == "/chat/stream":
                self._handle_stream(req, body, history_dicts)
                return

            result = loop.run_until_complete(agent.chat(
                req.message,
                history=history_dicts,
                template_context=req.template_context,
                mode=body.get("mode", "execute"),
            ))
            self._json_response(200, result.model_dump())

        def _handle_stream(self, req, body, history_dicts):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            origin = _get_cors_origin(self.headers.get("Origin", ""))
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
            self.end_headers()

            def on_stream(event):
                data = json.dumps({
                    "type": event.type.value,
                    "content": event.content,
                    "tool_name": event.tool_name,
                    "tool_args": event.tool_args,
                    "tool_result": event.tool_result,
                }, ensure_ascii=False, default=str)
                sse = "event: {}\ndata: {}\n\n".format(event.type.value, data)
                try:
                    self.wfile.write(sse.encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass

            result = loop.run_until_complete(agent.chat(
                req.message,
                history=history_dicts,
                template_context=req.template_context,
                mode=body.get("mode", "execute"),
                on_stream=on_stream,
            ))

            result_data = json.dumps(result.model_dump(), ensure_ascii=False, default=str)
            try:
                self.wfile.write("event: result\ndata: {}\n\n".format(result_data).encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

        def do_GET(self):
            if self.path == "/health":
                self._json_response(200, {"ok": True, "status": "ready"})
            elif self.path in ("/", "/demo"):
                self._html_response(_demo_html)
            else:
                self.send_error(404)

        def do_OPTIONS(self):
            self.send_response(200)
            origin = _get_cors_origin(self.headers.get("Origin", ""))
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _json_response(self, code, data):
            payload = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            origin = _get_cors_origin(self.headers.get("Origin", ""))
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
            self.end_headers()
            self.wfile.write(payload)

        def _html_response(self, html):
            payload = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
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
    print("  {}{}Flyto2 AI Server{}  {}(stdlib){}".format(_BOLD, _CYAN, _RESET, _DIM, _RESET))
    print("  Listening on {}http://{}:{}{}".format(_GREEN, args.host, args.port, _RESET))
    print()
    print("  {}GET  /{}            Demo page".format(_BOLD, _RESET))
    print("  {}POST /chat{}        Chat API".format(_BOLD, _RESET))
    print("  {}POST /chat/stream{} Streaming SSE".format(_BOLD, _RESET))
    print("  {}GET  /health{}      Health check".format(_BOLD, _RESET))
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  {}Shutting down.{}".format(_DIM, _RESET))
        server.server_close()
    finally:
        loop.close()


if __name__ == "__main__":
    main()
