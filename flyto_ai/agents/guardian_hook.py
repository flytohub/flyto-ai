# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""PreToolUse safety hook for Claude Agent SDK.

Inspects tool calls before execution and blocks dangerous operations.
Designed for the Claude Agent SDK hooks interface — independent of
flyto-ai's own policies.py (different data shape).
"""
import logging
import os
from typing import Any, Dict, Set

logger = logging.getLogger(__name__)

# ── Blocked bash patterns (substring match, case-insensitive) ──
BLOCKED_BASH = [
    "rm -rf /",
    "sudo rm",
    "mkfs",
    "> /dev/",
    "curl | sh",
    "curl |sh",
    "wget | sh",
    "wget |sh",
    "chmod 777 /",
    "dd if=",
    ":(){:|:&};:",
    "shutdown",
    "reboot",
    "kill -9 1",
    "npm publish",
    "pip upload",
    "twine upload",
    "git push --force",
    "git push -f",
]

# ── Blocked file path fragments (case-insensitive) ──
BLOCKED_PATHS = [
    ".env",
    "credentials",
    "service-account",
    ".git/config",
    ".ssh/",
    "id_rsa",
    "id_ed25519",
    ".aws/",
    ".kube/config",
    "secrets.yaml",
    "secrets.json",
]

# ── Allowed file extensions for Edit/Write ──
ALLOWED_EXTENSIONS: Set[str] = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".vue", ".html", ".css", ".scss",
    ".json", ".yaml", ".yml", ".md", ".txt", ".toml", ".cfg", ".ini",
    ".sh", ".bash", ".zsh", ".sql", ".graphql", ".proto",
    ".xml", ".svg", ".csv", ".env.example", ".gitignore",
    ".dockerfile", ".dockerignore", ".editorconfig",
    ".rs", ".go", ".java", ".kt", ".swift", ".c", ".cpp", ".h",
}


def _is_path_blocked(path: str) -> bool:
    lower = path.lower()
    return any(p in lower for p in BLOCKED_PATHS)


def _is_extension_allowed(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    if not ext:
        base = os.path.basename(path.lower())
        return base in {"dockerfile", "makefile", "gemfile", "rakefile", "procfile"}
    return ext in ALLOWED_EXTENSIONS


def _check_bash(command: str) -> None:
    lower = command.lower()
    for pattern in BLOCKED_BASH:
        if pattern in lower:
            raise GuardianBlocked("Bash blocked: matched '{}' in command".format(pattern))


def _check_file_tool(tool_name: str, args: Dict[str, Any]) -> None:
    path = args.get("file_path", "") or args.get("path", "")
    if not path:
        return
    if _is_path_blocked(path):
        raise GuardianBlocked("{} blocked: sensitive path '{}'".format(tool_name, path))
    if tool_name in ("Edit", "Write") and not _is_extension_allowed(path):
        raise GuardianBlocked(
            "{} blocked: extension not in allowlist for '{}'".format(tool_name, path)
        )


class GuardianBlocked(Exception):
    """Raised when guardian blocks a tool call."""


async def guardian_pre_hook(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_use_id: str = "",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Claude Agent SDK PreToolUse hook.

    Returns empty dict to approve, raises GuardianBlocked to deny.
    """
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        _check_bash(command)

    elif tool_name in ("Edit", "Write", "Read"):
        _check_file_tool(tool_name, tool_input)

    logger.debug("Guardian approved: %s (id=%s)", tool_name, tool_use_id)
    return {}
