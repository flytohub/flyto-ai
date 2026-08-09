# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""PreToolUse safety hook for Claude Agent SDK.

Inspects tool calls before execution and blocks dangerous operations.
Designed for the Claude Agent SDK hooks interface — independent of
flyto-ai's own policies.py (different data shape).
"""
import logging
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

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


# ── Repository landing, publish, and deploy denials (service mode) ──
# Structured program/sub-command matching. A single fragile substring is not
# enough: `git   push`, `git -C dir push`, and `FOO=1 sudo git push` are the
# same act of landing code.
_DENIED_SUBCOMMANDS: Dict[str, Set[str]] = {
    "git": {
        "add", "commit", "push", "tag", "merge", "rebase", "reset", "clean",
        "restore", "rm", "mv", "revert", "cherry-pick", "am", "apply", "stash",
        "checkout", "switch", "branch", "remote", "submodule", "filter-branch",
    },
    "gh": {"pr", "release", "repo", "workflow"},
    "hub": {"push", "merge"},
    "npm": {"publish", "deploy"},
    "pnpm": {"publish"},
    "yarn": {"publish"},
    "pip": {"upload"},
    "twine": {"upload"},
    "poetry": {"publish"},
    "cargo": {"publish"},
    "gem": {"push"},
    "docker": {"push"},
    "podman": {"push"},
    "kubectl": {"apply", "create", "delete", "patch", "replace", "rollout"},
    "helm": {"install", "upgrade", "rollback", "uninstall"},
    "terraform": {"apply", "destroy"},
    "pulumi": {"up", "destroy"},
    "serverless": {"deploy"},
    "sls": {"deploy"},
    "vercel": {"deploy"},
    "netlify": {"deploy"},
    "fly": {"deploy"},
    "flyctl": {"deploy"},
    "eb": {"deploy"},
    "heroku": {"deploy"},
}
#: Programs whose mere invocation lands or publishes something.
_DENIED_PROGRAMS = frozenset({"git-push", "git-receive-pack"})
#: Leading wrappers that do not change what is ultimately executed.
_COMMAND_WRAPPERS = frozenset({"sudo", "env", "command", "nohup", "time", "nice", "doas"})
#: Options that consume the following token, so it is not the sub-command.
_OPTIONS_WITH_VALUE: Dict[str, Set[str]] = {"git": {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}}
_SEGMENT_SPLIT_RE = re.compile(r"(?:\|\||&&|[;\n|&])")
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _is_path_blocked(path: str) -> bool:
    lower = path.lower()
    return any(p in lower for p in BLOCKED_PATHS)


def _split_segments(command: str) -> List[List[str]]:
    """Split one command line into token lists without executing anything."""
    segments: List[List[str]] = []
    for raw in _SEGMENT_SPLIT_RE.split(command):
        piece = raw.strip()
        if not piece:
            continue
        try:
            tokens = shlex.split(piece, comments=True)
        except ValueError:
            # Unbalanced quoting is not a reason to trust the command.
            tokens = piece.split()
        if tokens:
            segments.append(tokens)
    return segments


def _program_and_subcommand(tokens: Iterable[str]) -> "tuple[str, str]":
    """Resolve the effective program and its first sub-command token."""
    remaining = [token for token in tokens]
    while remaining and (
        _ASSIGNMENT_RE.match(remaining[0]) or os.path.basename(remaining[0]) in _COMMAND_WRAPPERS
    ):
        remaining.pop(0)
    if not remaining:
        return "", ""
    program = os.path.basename(remaining[0]).lower()
    if program.endswith(".exe"):
        program = program[:-4]
    index = 1
    valued = _OPTIONS_WITH_VALUE.get(program, set())
    while index < len(remaining):
        token = remaining[index]
        if not token.startswith("-"):
            break
        if token in valued:
            index += 2
            continue
        index += 1
    subcommand = remaining[index].lower() if index < len(remaining) else ""
    return program, subcommand


def _check_service_command(command: str) -> None:
    """Deny repository landing, publishing, and deployment in service mode."""
    for tokens in _split_segments(command):
        program, subcommand = _program_and_subcommand(tokens)
        if not program:
            continue
        if program in _DENIED_PROGRAMS:
            raise GuardianBlocked("Bash blocked: repository landing is not permitted in service mode")
        denied = _DENIED_SUBCOMMANDS.get(program)
        if denied and subcommand in denied:
            raise GuardianBlocked(
                "Bash blocked: '{} {}' is not permitted in service mode".format(program, subcommand),
            )


#: Tools that mutate workspace content and therefore need edit authority.
_MUTATING_TOOLS = frozenset({"Edit", "Write", "NotebookEdit", "MultiEdit"})
#: Tools whose optional `path` is a search root, not a file to write.
_SEARCH_TOOLS = frozenset({"Glob", "Grep"})
#: Search tools that return file contents rather than only file names.
_CONTENT_SEARCH_TOOLS = frozenset({"Grep"})


def _resolve_workspace_path(
    tool_name: str, path: str, workspace: str, *, allow_symlink: bool = False,
) -> None:
    """Require one tool path to resolve inside the exact supplied workspace.

    The caller-supplied spelling is not the authority. After resolution the
    sensitive-path policy is applied again, and a final-component symlink is
    refused outright: a link named `notes.py` says nothing about its target.
    """
    root = Path(os.path.realpath(os.path.expanduser(workspace)))
    candidate = Path(os.path.expanduser(path))
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = Path(os.path.realpath(candidate))
    if resolved != root and root not in resolved.parents:
        raise GuardianBlocked("{} blocked: path is outside the run workspace".format(tool_name))
    if candidate.is_symlink() and not allow_symlink:
        raise GuardianBlocked("{} blocked: symlinked path is not permitted".format(tool_name))
    if _is_path_blocked(str(resolved)):
        raise GuardianBlocked("{} blocked: sensitive resolved path".format(tool_name))


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


def _check_file_tool(
    tool_name: str, args: Dict[str, Any], workspace: Optional[str] = None,
) -> None:
    path = args.get("file_path", "") or args.get("path", "")
    if not isinstance(path, str) or not path:
        return
    # Errors never echo the path: it can itself carry credential material.
    if _is_path_blocked(path):
        raise GuardianBlocked("{} blocked: sensitive path".format(tool_name))
    if tool_name in ("Edit", "Write") and not _is_extension_allowed(path):
        raise GuardianBlocked("{} blocked: extension not in allowlist".format(tool_name))
    if workspace:
        _resolve_workspace_path(tool_name, path, workspace)


class GuardianBlocked(Exception):
    """Raised when guardian blocks a tool call."""


async def guardian_pre_hook(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_use_id: str = "",
    *,
    workspace: Optional[str] = None,
    service_mode: bool = False,
    edit_authority: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Claude Agent SDK PreToolUse hook.

    Returns empty dict to approve, raises GuardianBlocked to deny. `workspace`
    confines file and search tools to one run root; `service_mode` additionally
    denies repository landing, publishing, and deployment; `edit_authority`
    reflects the host's startup sandbox/approval decision. Service mode already
    shapes the tool catalog, so these checks are defense in depth against a
    future catalog regression.
    """
    if not edit_authority and tool_name in _MUTATING_TOOLS:
        raise GuardianBlocked(
            "{} blocked: this run has no workspace write authority".format(tool_name),
        )

    if tool_name in _SEARCH_TOOLS:
        if service_mode and tool_name in _CONTENT_SEARCH_TOOLS:
            # Directory-wide Grep can return bytes from a protected result
            # path even when its root is safe. Limit service-mode content
            # search to one explicit regular file, which is the same boundary
            # `Read` can authorize independently.
            root = tool_input.get("path", "")
            if not isinstance(root, str):
                raise GuardianBlocked(
                    "{} blocked: search path must be a string".format(tool_name),
                )
            if not workspace or not root:
                raise GuardianBlocked(
                    "{} blocked: service content search requires one explicit file".format(
                        tool_name,
                    ),
                )
            _resolve_workspace_path(tool_name, root, workspace)
            candidate = Path(os.path.expanduser(root))
            if not candidate.is_absolute():
                candidate = Path(workspace) / candidate
            resolved = Path(os.path.realpath(candidate))
            if not resolved.is_file():
                raise GuardianBlocked(
                    "{} blocked: service content search requires one regular file".format(
                        tool_name,
                    ),
                )
        # An omitted search root means the SDK cwd, which is already confined.
        root = tool_input.get("path", "")
        if not isinstance(root, str):
            raise GuardianBlocked("{} blocked: search path must be a string".format(tool_name))
        if workspace and root:
            _resolve_workspace_path(tool_name, root, workspace, allow_symlink=True)

    elif tool_name == "Bash":
        command = tool_input.get("command", "")
        if not isinstance(command, str):
            raise GuardianBlocked("Bash blocked: command must be a string")
        _check_bash(command)
        if service_mode:
            _check_service_command(command)

    elif tool_name in ("Edit", "Write", "Read", "NotebookEdit", "MultiEdit"):
        _check_file_tool(tool_name, tool_input, workspace)

    logger.debug("Guardian approved: %s (id=%s)", tool_name, tool_use_id)
    return {}
