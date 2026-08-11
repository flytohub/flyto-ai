# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Workspace-confined tools for the native Flyto2 coding backend."""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from flyto_ai.coding.contracts import ApprovalPolicy, SandboxMode
from flyto_ai.coding.store import redact_evidence


MAX_FILE_BYTES = 512 * 1024
MAX_TOOL_OUTPUT_BYTES = 128 * 1024
MAX_SNAPSHOT_FILES = 50_000
_BLOCKED_EXECUTABLES = {
    "chmod", "chown", "dd", "doas", "kill", "mkfs", "mount", "nc",
    "netcat", "pkill", "rm", "rmdir", "rsync", "scp", "ssh", "su",
    "sudo", "umount", "unlink",
}
_SHELLS = {"sh", "bash", "zsh", "fish"}
_PROTECTED_DIRECTORY_NAMES = {
    ".aws", ".git", ".gnupg", ".ssh", ".terraform", ".vault",
}
_PROTECTED_FILE_NAMES = {
    ".env", ".netrc", "credentials.json", "id_dsa", "id_ed25519",
    "id_rsa", "secrets.json",
}
_DANGEROUS_ARG_PATTERNS = (
    re.compile(r"^/(?:$|Users(?:/|$)|home(?:/|$)|etc(?:/|$)|var(?:/|$))"),
    re.compile(r"^~(?:/|$)"),
)


#: Where a containerised boundary always sees the workspace.
CONTAINER_WORKSPACE = "/workspace"


def resolve_executable(program: str, workspace: Optional[str] = None) -> Optional[str]:
    """Answer, once, whether an argv[0] can actually be launched on this host.

    This exists to be *shared*. Preflight has to give the same answer as the
    runner or it is not a preflight at all - it would be a second opinion, and
    the two would drift the first time either changed. So there is one function,
    the runner calls it on the path that really executes, and anything that
    wants to know in advance calls the same one.

    A plain name is searched on ``PATH``. An absolute path is checked directly.
    A relative path containing a separator is resolved from the repository
    workspace, because the real check process runs with that workspace as its
    cwd; resolving it from the long-lived MCP service cwd makes feasibility
    depend on where the supervisor happened to be launched. ``None`` means the
    argv could not be launched; it never means "probably fine".

    Deliberately does not run anything. A verification command that exits
    non-zero may be describing the very defect the task was opened to fix, so
    executing it here would refuse exactly the work that most needs doing.
    """

    if not isinstance(program, str) or not program:
        return None
    if workspace and not os.path.isabs(program) and os.path.dirname(program):
        try:
            program = str(Path(workspace).expanduser().resolve() / program)
        except (OSError, RuntimeError, ValueError):
            return None
    return shutil.which(program)


def container_hardening_argv(docker: str, cidfile: str) -> List[str]:
    """The isolation flags every containerised execution boundary shares.

    Factored so the read-only model-command sandbox and the read/write project
    action sandbox cannot drift into two different security postures. The
    *only* difference between them is the workspace mount mode and the uid that
    owns writes; everything that constrains the container - no network,
    read-only root filesystem, dropped capabilities, no new privileges, bounded
    pids/memory/cpu - is defined once, here.

    Nothing from the host environment is forwarded. In particular the Docker
    socket is never mounted: a container that could reach the daemon could
    start an unconstrained sibling and the whole boundary would be decorative.
    """

    return [
        docker, "run", "--rm",
        "--cidfile", cidfile,
        # No network at all. A declared action regenerates derived files; it
        # has no reason to reach a registry, a package index or an exfil host.
        "--network", "none",
        "--read-only",
        "--pids-limit", "128", "--memory", "1g", "--cpus", "2",
        "--security-opt", "no-new-privileges", "--cap-drop", "ALL",
    ]


def container_runtime_argv() -> List[str]:
    """Writable scratch and a deterministic in-container environment.

    `HOME` points at a tmpfs path that exists only for this container, so an
    action resolving `~/.ssh`, `~/.aws` or `~/.gitconfig` finds nothing. The
    host's real home is never mounted, so there is nothing to find.
    """

    return [
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=128m,mode=1777",
        "--workdir", CONTAINER_WORKSPACE,
        "--env", "HOME=/tmp/home",
        "--env", "TMPDIR=/tmp",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
    ]


def container_workspace_mount_argv(source: Path, *, writable: bool) -> List[str]:
    """Bind exactly one directory, and nothing else, at the workspace path.

    `writable` is the entire read/write difference. A read-only boundary must
    never gain write access by sharing this helper, so the flag is explicit at
    every call site rather than defaulted.
    """

    mount = "type=bind,src={},dst={}".format(source, CONTAINER_WORKSPACE)
    if not writable:
        mount += ",readonly"
    return ["--mount", mount]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class WorkspaceViolation(RuntimeError):
    pass


class WorkspaceTools:
    """Bounded local tools; every path resolves under exactly one workspace."""

    def __init__(
        self,
        root: str,
        *,
        sandbox_mode: SandboxMode = SandboxMode.WORKSPACE_WRITE,
        approval_policy: ApprovalPolicy = ApprovalPolicy.NEVER,
        sandbox_image: str = "python:3.12-slim",
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("workspace root must be a directory")
        self.sandbox_mode = SandboxMode(sandbox_mode)
        self.approval_policy = ApprovalPolicy(approval_policy)
        self.sandbox_image = sandbox_image
        if any(character in str(self.root) for character in ("\n", "\r", ",")):
            raise ValueError("workspace path is incompatible with command sandboxing")
        self.directly_changed: Set[str] = set()
        self._docker_host = ""
        self.command_sandbox_backend = self._detect_command_sandbox()

    @property
    def definitions(self) -> List[Dict[str, Any]]:
        return [
            self._tool("coding_list_files", "List bounded workspace files.", {
                "type": "object", "properties": {
                    "path": {"type": "string", "default": "."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 200},
                }, "additionalProperties": False,
            }),
            self._tool("coding_read_file", "Read a UTF-8 text file inside the workspace.", {
                "type": "object", "properties": {"path": {"type": "string"}},
                "required": ["path"], "additionalProperties": False,
            }),
            self._tool(
                "coding_search",
                "Search workspace text for one literal fixed string; returns bounded matches. "
                "This is not a regular-expression search.",
                {
                "type": "object", "properties": {
                    "query": {"type": "string"}, "path": {"type": "string", "default": "."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                }, "required": ["query"], "additionalProperties": False,
                },
            ),
            self._tool("coding_replace_text", "Replace one exact unique text occurrence atomically.", {
                "type": "object", "properties": {
                    "path": {"type": "string"}, "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                }, "required": ["path", "old_text", "new_text"], "additionalProperties": False,
            }),
            self._tool("coding_write_file", "Create or replace one UTF-8 workspace file atomically.", {
                "type": "object", "properties": {
                    "path": {"type": "string"}, "content": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": False},
                }, "required": ["path", "content"], "additionalProperties": False,
            }),
            self._tool("coding_run", "Run an argv-only development command without a shell.", {
                "type": "object", "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 32},
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300, "default": 120},
                }, "required": ["argv"], "additionalProperties": False,
            }),
            self._tool("coding_git_diff", "Read the current bounded git status and diff summary.", {
                "type": "object", "properties": {}, "additionalProperties": False,
            }),
        ]

    async def dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name == "coding_list_files":
                return self.list_files(str(args.get("path", ".")), int(args.get("limit", 200)))
            if name == "coding_read_file":
                return self.read_file(str(args["path"]))
            if name == "coding_search":
                return await self.search(
                    str(args["query"]), str(args.get("path", ".")), int(args.get("limit", 100)),
                )
            if name == "coding_replace_text":
                return self.replace_text(str(args["path"]), str(args["old_text"]), str(args["new_text"]))
            if name == "coding_write_file":
                return self.write_file(
                    str(args["path"]), str(args["content"]), bool(args.get("overwrite", False)),
                )
            if name == "coding_run":
                argv = args.get("argv")
                if not isinstance(argv, list):
                    raise WorkspaceViolation("argv must be an array")
                return await self.run(argv, int(args.get("timeout_seconds", 120)))
            if name == "coding_git_diff":
                return await self.git_diff()
            return {"ok": False, "error": "unknown coding tool"}
        except (KeyError, OSError, UnicodeError, ValueError, WorkspaceViolation) as exc:
            return {"ok": False, "error": str(exc)[:1000], "error_type": type(exc).__name__}

    def resolve(self, raw_path: str, *, allow_missing: bool = False) -> Path:
        if not raw_path or "\x00" in raw_path:
            raise WorkspaceViolation("path is empty or invalid")
        supplied = Path(raw_path)
        if supplied.is_absolute():
            raise WorkspaceViolation("absolute paths are not allowed")
        if self._is_protected_relative(supplied):
            raise WorkspaceViolation("path is protected by the coding security policy")
        candidate = (self.root / supplied).resolve(strict=not allow_missing)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceViolation("path escapes the workspace") from exc
        return candidate

    def list_files(self, raw_path: str = ".", limit: int = 200) -> Dict[str, Any]:
        target = self.resolve(raw_path)
        if not target.is_dir():
            raise WorkspaceViolation("list path must be a directory")
        limit = max(1, min(limit, 500))
        files: List[str] = []
        for current, dirnames, filenames in os.walk(target):
            dirnames[:] = sorted(
                directory for directory in dirnames
                if directory not in {".flyto-index", "node_modules", ".venv"}
                and not self._is_protected_relative(
                    (Path(current) / directory).relative_to(self.root)
                )
            )
            for filename in sorted(filenames):
                path = Path(current) / filename
                if self._is_protected_relative(path.relative_to(self.root)):
                    continue
                files.append(path.relative_to(self.root).as_posix())
                if len(files) >= limit:
                    return {"ok": True, "files": files, "truncated": True}
        return {"ok": True, "files": files, "truncated": False}

    def read_file(self, raw_path: str) -> Dict[str, Any]:
        path = self.resolve(raw_path)
        if not path.is_file():
            raise WorkspaceViolation("path must be a file")
        data = path.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            raise WorkspaceViolation("file exceeds the read limit")
        if b"\x00" in data:
            raise WorkspaceViolation("binary files are not supported")
        return {"ok": True, "path": path.relative_to(self.root).as_posix(), "content": data.decode("utf-8")}

    async def search(self, query: str, raw_path: str = ".", limit: int = 100) -> Dict[str, Any]:
        if not query or len(query) > 4096:
            raise WorkspaceViolation("query is empty or too large")
        target = self.resolve(raw_path)
        if not shutil.which("rg"):
            raise WorkspaceViolation("ripgrep is required for coding_search")
        result = await self._run_process(
            [
                "rg", "--line-number", "--color", "never", "--fixed-strings",
                "--glob", "!.env", "--glob", "!.env.*",
                "--glob", "!**/.git/**", "--glob", "!**/.ssh/**",
                "--glob", "!**/.aws/**", "--glob", "!**/.gnupg/**",
                "--glob", "!**/credentials.json", "--glob", "!**/secrets.json",
                "--", query, str(target),
            ],
            timeout_seconds=30,
        )
        lines = result["output"].splitlines()[: max(1, min(limit, 500))]
        response: Dict[str, Any] = {
            "ok": result["exit_code"] in {0, 1},
            "matches": lines,
            "truncated": len(lines) >= limit,
            "query_mode": "literal",
        }
        if not lines:
            response["next_action"] = (
                "No literal matches. Read the current file before retrying an edit; "
                "verification output may contain runtime values that are not source text."
            )
        return response

    def replace_text(self, raw_path: str, old_text: str, new_text: str) -> Dict[str, Any]:
        self._require_write()
        if not old_text:
            raise WorkspaceViolation("old_text must not be empty")
        path = self.resolve(raw_path)
        data = path.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            raise WorkspaceViolation("file exceeds the edit limit")
        text = data.decode("utf-8")
        count = text.count(old_text)
        if count != 1:
            raise WorkspaceViolation("old_text must occur exactly once; found {}".format(count))
        updated = text.replace(old_text, new_text, 1)
        if len(updated.encode("utf-8")) > MAX_FILE_BYTES:
            raise WorkspaceViolation("edited file exceeds the size limit")
        self._atomic_write(path, updated)
        relative = path.relative_to(self.root).as_posix()
        self.directly_changed.add(relative)
        return {"ok": True, "path": relative, "sha256": _sha256(updated.encode("utf-8"))}

    def write_file(self, raw_path: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
        self._require_write()
        path = self.resolve(raw_path, allow_missing=True)
        if path.exists() and not overwrite:
            raise WorkspaceViolation("file exists; use exact replacement or set overwrite")
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise WorkspaceViolation("content exceeds the write limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = path.parent.resolve(strict=True)
        try:
            resolved_parent.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceViolation("parent path escapes the workspace") from exc
        self._atomic_write(path, content)
        relative = path.relative_to(self.root).as_posix()
        self.directly_changed.add(relative)
        return {"ok": True, "path": relative, "sha256": _sha256(content.encode("utf-8"))}

    async def run(self, argv: Sequence[str], timeout_seconds: int = 120) -> Dict[str, Any]:
        self._validate_argv(argv)
        if self.approval_policy in {ApprovalPolicy.ON_REQUEST, ApprovalPolicy.ALWAYS}:
            return {"ok": False, "approval_required": True, "error": "host approval is required for command execution"}
        if not self.command_sandbox_backend:
            return {
                "ok": False,
                "error": "no supported OS command sandbox is available",
                "sandbox_backend": "",
            }
        return await self._run_process(
            list(argv), timeout_seconds=max(1, min(timeout_seconds, 300)),
            model_command=True,
        )

    async def run_check(self, argv: Sequence[str], timeout_seconds: int) -> Dict[str, Any]:
        """Run a predeclared verification command regardless of interactive approval."""
        self._validate_argv(argv)
        return await self._run_process(list(argv), timeout_seconds=timeout_seconds)

    async def git_diff(self) -> Dict[str, Any]:
        if not (self.root / ".git").exists():
            return {"ok": True, "is_git": False, "status": "", "diff_stat": ""}
        status = await self._run_process(["git", "status", "--short"], timeout_seconds=20)
        stat = await self._run_process(["git", "diff", "--stat"], timeout_seconds=20)
        return {
            "ok": status["ok"] and stat["ok"], "is_git": True,
            "status": status["output"], "diff_stat": stat["output"],
        }

    def snapshot(self) -> Dict[str, str]:
        """Hash bounded source state so pre-existing dirty files are not misattributed."""
        paths: Iterable[Path]
        if (self.root / ".git").exists() and shutil.which("git"):
            import subprocess
            completed = subprocess.run(
                ["git", "ls-files", "-co", "--exclude-standard", "-z"],
                cwd=str(self.root), capture_output=True, check=False, timeout=30,
            )
            raw_paths = completed.stdout.decode("utf-8", errors="replace").split("\x00")
            paths = (self.root / item for item in raw_paths if item)
        else:
            paths = (path for path in self.root.rglob("*") if path.is_file())
        result: Dict[str, str] = {}
        for index, path in enumerate(paths):
            if index >= MAX_SNAPSHOT_FILES:
                raise WorkspaceViolation("workspace snapshot exceeds the file limit")
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(self.root)
                if resolved.stat().st_size <= 10 * 1024 * 1024:
                    result[resolved.relative_to(self.root).as_posix()] = _sha256(resolved.read_bytes())
            except (OSError, ValueError):
                continue
        return result

    @staticmethod
    def changed_since(before: Mapping[str, str], after: Mapping[str, str]) -> List[str]:
        return sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))

    def _require_write(self) -> None:
        if self.sandbox_mode != SandboxMode.WORKSPACE_WRITE:
            raise WorkspaceViolation("sandbox is read-only")
        if self.approval_policy in {ApprovalPolicy.ON_REQUEST, ApprovalPolicy.ALWAYS}:
            raise WorkspaceViolation("host approval is required for file writes")

    def _validate_argv(self, argv: Sequence[str]) -> None:
        if not argv or len(argv) > 32 or any(not isinstance(item, str) or not item or len(item) > 4096 for item in argv):
            raise WorkspaceViolation("argv must contain between 1 and 32 bounded strings")
        executable = Path(argv[0]).name.lower()
        if executable in _BLOCKED_EXECUTABLES:
            raise WorkspaceViolation("executable is blocked by coding policy")
        if executable in _SHELLS and (len(argv) < 2 or argv[1].startswith("-")):
            raise WorkspaceViolation("shells may only execute a workspace script without flags")
        for item in argv[1:]:
            if any(pattern.search(item) for pattern in _DANGEROUS_ARG_PATTERNS):
                raise WorkspaceViolation("command argument targets a protected path")

    async def _run_process(
        self, argv: List[str], timeout_seconds: int, *, model_command: bool = False,
    ) -> Dict[str, Any]:
        started = time.monotonic()
        # The single resolver, on the path that actually launches the process.
        # Preflight calls this same function, which is what makes its answer
        # binding rather than advisory.
        executable = resolve_executable(argv[0], str(self.root))
        if not executable:
            raise WorkspaceViolation("executable is not installed")
        env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TERM", "TMPDIR") if key in os.environ}
        runtime_parent = None
        if model_command and self.command_sandbox_backend == "docker":
            runtime_parent_path = Path.home() / ".flyto" / "coding-runtime"
            runtime_parent_path.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                runtime_parent_path.chmod(0o700)
            except OSError:
                pass
            runtime_parent = str(runtime_parent_path)
        with tempfile.TemporaryDirectory(
            prefix="flyto-coding-home-", dir=runtime_parent,
        ) as runtime_home:
            env["HOME"] = runtime_home
            runtime_tmp = Path(runtime_home) / "tmp"
            runtime_tmp.mkdir(mode=0o700)
            env["TMPDIR"] = str(runtime_tmp)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            if model_command and self.command_sandbox_backend == "docker" and self._docker_host:
                env["DOCKER_HOST"] = self._docker_host
            command = [executable, *argv[1:]]
            if model_command:
                command = self._sandbox_command(command, runtime_home)
            process = await asyncio.create_subprocess_exec(
                *command, cwd=str(self.root), env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                output, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                if model_command and self.command_sandbox_backend == "docker":
                    await self._cleanup_docker_container(runtime_home)
                try:
                    output, _ = await asyncio.wait_for(process.communicate(), timeout=2)
                except asyncio.TimeoutError:
                    output = b""
                return {
                    "ok": False, "exit_code": None, "timed_out": True,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "output": "command timed out",
                    "sandbox_backend": self.command_sandbox_backend if model_command else "trusted-check",
                }
        truncated = len(output) > MAX_TOOL_OUTPUT_BYTES
        bounded = output[:MAX_TOOL_OUTPUT_BYTES].decode("utf-8", errors="replace")
        if truncated:
            bounded += "\n...[output truncated]"
        bounded = str(redact_evidence(bounded))
        return {
            "ok": process.returncode == 0, "exit_code": process.returncode,
            "timed_out": False, "duration_ms": int((time.monotonic() - started) * 1000),
            "output": bounded, "output_sha256": _sha256(output), "truncated": truncated,
            "sandbox_backend": self.command_sandbox_backend if model_command else "trusted-check",
        }

    def _sandbox_command(self, command: List[str], runtime_home: str) -> List[str]:
        if self.command_sandbox_backend == "docker":
            cidfile = str(Path(runtime_home) / "container.cid")
            docker_workspace, staged = self._docker_workspace(runtime_home)
            container_command = list(command)
            executable_name = Path(container_command[0]).name
            if executable_name.startswith("python"):
                container_command[0] = "python"
            elif Path(container_command[0]).is_absolute():
                container_command[0] = executable_name
            wrapped = [
                *container_hardening_argv(shutil.which("docker") or "docker", cidfile),
                # Model commands run as `nobody` and see the tree read-only.
                # Project actions differ only in these two respects; every
                # other constraint above is shared, by construction.
                "--user", "65534:65534",
                *container_workspace_mount_argv(docker_workspace, writable=False),
                *container_runtime_argv(),
            ]
            protected_paths = [] if staged else self._protected_existing_paths()
            denied_file: Path | None = None
            if any(not path.is_dir() for path in protected_paths):
                denied_file = Path(runtime_home) / "blocked-file"
                denied_file.touch(mode=0o000)
                denied_file.chmod(0o000)
            for path in protected_paths:
                relative = path.relative_to(self.root).as_posix()
                target = "/workspace/{}".format(relative)
                if path.is_dir():
                    wrapped.extend(["--mount", "type=tmpfs,dst={},tmpfs-mode=000".format(target)])
                else:
                    assert denied_file is not None
                    wrapped.extend([
                        "--mount", "type=bind,src={},dst={},readonly".format(
                            denied_file, target,
                        ),
                    ])
            return [*wrapped, self.sandbox_image, *container_command]
        if self.command_sandbox_backend == "bwrap":
            wrapped = [
                shutil.which("bwrap") or "bwrap", "--die-with-parent",
                "--new-session", "--unshare-net", "--ro-bind", "/", "/",
                "--bind", runtime_home, runtime_home,
                "--ro-bind", str(self.root), str(self.root),
                "--chdir", str(self.root), "--setenv", "HOME", runtime_home,
                "--setenv", "TMPDIR", str(Path(runtime_home) / "tmp"),
            ]
            for path in self._protected_existing_paths():
                if path.is_dir():
                    wrapped.extend(["--tmpfs", str(path)])
                else:
                    wrapped.extend(["--ro-bind", "/dev/null", str(path)])
            return [*wrapped, "--", *command]
        raise WorkspaceViolation("no supported OS command sandbox is available")

    def _docker_workspace(self, runtime_home: str) -> tuple[Path, bool]:
        """Stage macOS paths that the local Docker VM cannot bind directly."""
        if sys.platform != "darwin" or str(self.root).startswith("/Users/"):
            return self.root, False
        staged = Path(runtime_home) / "workspace"

        def ignore(current: str, names: List[str]) -> Set[str]:
            current_path = Path(current)
            try:
                base = current_path.relative_to(self.root)
            except ValueError:
                base = Path(".")
            return {
                name for name in names
                if self._is_protected_relative(base / name)
                or name in {".flyto-index"}
            }

        shutil.copytree(self.root, staged, symlinks=True, ignore=ignore)
        for current, dirnames, filenames in os.walk(staged):
            current_path = Path(current)
            current_path.chmod(current_path.stat().st_mode | 0o055)
            for filename in filenames:
                path = current_path / filename
                if not path.is_symlink():
                    path.chmod(path.stat().st_mode | 0o044)
        return staged, True

    async def _cleanup_docker_container(self, runtime_home: str) -> None:
        cidfile = Path(runtime_home) / "container.cid"
        try:
            container_id = cidfile.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if not re.fullmatch(r"[a-f0-9]{12,64}", container_id):
            return
        process = await asyncio.create_subprocess_exec(
            shutil.which("docker") or "docker", "rm", "-f", container_id,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()

    def _protected_existing_paths(self) -> List[Path]:
        protected: List[Path] = []
        for current, dirnames, filenames in os.walk(self.root):
            retained = []
            for dirname in dirnames:
                path = Path(current) / dirname
                if self._is_protected_relative(path.relative_to(self.root)):
                    protected.append(path)
                else:
                    retained.append(dirname)
            dirnames[:] = retained
            for filename in filenames:
                path = Path(current) / filename
                if self._is_protected_relative(path.relative_to(self.root)):
                    protected.append(path)
            if len(protected) >= 256:
                break
        return sorted(protected, key=lambda item: (len(item.parts), str(item)))

    @staticmethod
    def _is_protected_relative(path: Path) -> bool:
        parts = [part.casefold() for part in path.parts]
        if any(part in _PROTECTED_DIRECTORY_NAMES for part in parts):
            return True
        if not parts:
            return False
        filename = parts[-1]
        return (
            filename in _PROTECTED_FILE_NAMES
            or filename.startswith(".env.")
            or filename.endswith(".pem")
            or filename.endswith(".key")
        )

    def _detect_command_sandbox(self) -> str:
        if sys.platform.startswith("linux") and shutil.which("bwrap"):
            return "bwrap"
        if shutil.which("docker"):
            try:
                context = subprocess.run(
                    ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, timeout=5, check=False, text=True,
                )
                result = subprocess.run(
                    ["docker", "image", "inspect", self.sandbox_image],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, timeout=5, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return ""
            if result.returncode == 0:
                candidate = context.stdout.strip() if context.returncode == 0 else ""
                self._docker_host = candidate or os.environ.get("DOCKER_HOST", "")
                return "docker"
        return ""

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
        fd, raw_temp = tempfile.mkstemp(prefix=".flyto-edit-", dir=str(path.parent))
        try:
            os.fchmod(fd, mode)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(raw_temp, path)
        except Exception:
            try:
                os.unlink(raw_temp)
            except OSError:
                pass
            raise

    @staticmethod
    def _tool(name: str, description: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        return {"name": name, "description": description, "inputSchema": schema}
