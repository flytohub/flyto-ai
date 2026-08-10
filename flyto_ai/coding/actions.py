# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Run repository-declared commands by name, and nothing else.

An implementer sometimes needs a real project command to run - regenerate a
derived file, rebuild an index - and the obvious way to allow that is the one
that must never be taken: give the model a shell. This module is the narrow
alternative.

The split that makes it safe is *who decides what runs*. The repository decides,
in source control: the name, the exact argv, the timeout, the optional
subdirectory. The caller decides only *which declared name* to invoke. There is
no argument to inject into, no string to interpolate, no environment to expand,
and no way to express a command that was not reviewed and committed.

Three properties are worth stating because they are easy to lose:

*The contract is re-read at launch.* An authorization names a config digest; if
the file has changed since, the invocation is refused. Otherwise a model that
can edit the workspace could authorize a harmless action, rewrite the
declaration, and have the host run the rewrite.

*The process is isolated at the syscall.* ``shell=False`` so no metacharacter
is ever interpreted, an explicit minimal environment so nothing leaks in from
the host, and ``start_new_session`` so a timeout can kill the whole process
group rather than orphan children that keep running.

*An action is not a check.* Nothing here returns verification evidence. Actions
and checks are separate contract keys, separate types and separate code paths,
and the host re-runs its required checks afterward regardless of what an action
reported. A green action proves a command exited zero; it proves nothing about
the change.
"""
from __future__ import annotations

import errno
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from flyto_ai.coding.checks import config_digest, load_project_actions
from flyto_ai.coding.contracts import (
    ContractSnapshot,
    ProjectActionResult,
    ProjectActionSpec,
)
from flyto_ai.coding.workspace import (
    CONTAINER_WORKSPACE,
    WorkspaceTools,
    container_hardening_argv,
    container_runtime_argv,
    container_workspace_mount_argv,
)

__all__ = [
    "ACTION_SANDBOX_UNAVAILABLE",
    "MAX_ACTION_OUTPUT_BYTES",
    "MAX_ACTION_OUTPUT_CHARS",
    "ActionSandboxUnavailable",
    "ProjectActionError",
    "ProjectActionExecutor",
    "UndeclaredAction",
]

#: Stable code for "this host cannot isolate a declared action". Distinct from
#: every provider and contract code: nothing is wrong with the model, the
#: request or the repository - the host simply cannot offer the boundary the
#: action surface is only safe behind.
ACTION_SANDBOX_UNAVAILABLE = "action_sandbox_unavailable"
_NO_SANDBOX = (
    "no OS isolation boundary is available for repository-declared actions"
)
#: A container image content identity, and nothing looser. A tag is mutable and
#: therefore is never an identity; a short id is ambiguous and therefore is
#: never accepted.
_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
#: A container id as Docker writes it into a cidfile.
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{12,64}$")

#: A read-only, host-produced list is the only Git-derived information an
#: action receives.  The repository's `.git` directory stays masked; the
#: image's narrow `git` shim serves this file only for the exact `ls-files`
#: command used by deterministic documentation generators.
_TRACKED_FILES_TARGET = "/run/flyto-action/tracked-files"
_MAX_TRACKED_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_TRACKED_MANIFEST_FILES = 50_000

#: Per-stream output bound. Enough to diagnose, far too little to exfiltrate a
#: repository through a tool result.
MAX_ACTION_OUTPUT_CHARS = 16000
#: The byte ceiling actually held in memory per stream while draining. Kept
#: above the character bound because one character can be several bytes.
MAX_ACTION_OUTPUT_BYTES = MAX_ACTION_OUTPUT_CHARS * 4
#: Read size. Small enough that a single read cannot itself be the memory
#: problem, large enough that draining a big stream is not syscall-bound.
_READ_CHUNK_BYTES = 65536
#: System directories, in a fixed order. The host's own `PATH` is never
#: inherited: it routinely contains a project-local `node_modules/.bin` or a
#: directory the workspace itself can write, and resolving a reviewed
#: executable through one of those would let the repository decide which binary
#: "python" means.
_SYSTEM_PATH_DIRS = ("/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin")


def _fixed_path() -> str:
    """The deterministic search path an action resolves its executable on.

    System directories come first and always win. The running interpreter's
    own directory is appended last, so a contract that declares `python`
    resolves to the interpreter this service is actually running rather than
    failing outright in a virtualenv deployment - but it can never shadow a
    system binary, which matters because a virtualenv's `bin` may sit inside
    the very repository the model is editing.

    Either way nothing is inherited from the host's `PATH`.
    """

    # Not `realpath`: a virtualenv's `bin/python` is a symlink into the base
    # installation, and that base directory frequently contains only
    # `python3.11`. The configured directory is the one that has the names a
    # contract actually writes.
    interpreter_dir = os.path.dirname(os.path.abspath(sys.executable))
    ordered = list(_SYSTEM_PATH_DIRS)
    if interpreter_dir and interpreter_dir not in ordered:
        # Appended, never prepended. A virtualenv's `bin` can live inside the
        # repository the model is editing, so it must never be able to shadow a
        # system binary - it only supplies names the system does not have.
        ordered.append(interpreter_dir)
    return os.pathsep.join(ordered)


def _fixed_executable(name: str) -> str:
    """Resolve a host utility only from the deterministic system path."""

    for directory in _fixed_path().split(os.pathsep):
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return ""
#: The complete environment an action ever sees. `HOME` is deliberately absent:
#: it is the doorway to ~/.ssh, ~/.aws, ~/.config, ~/.gitconfig and the shell
#: history, and a reviewed command has no business locating any of them through
#: an inherited variable. So are XDG_*, SSH_*, cloud and git configuration,
#: PYTHONPATH, VIRTUAL_ENV, and every provider credential.
#:
#: This is explicit system compatibility rather than ambient inheritance: a
#: command that genuinely needs a variable must have it added here, in review.
_STATIC_ENV = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONUNBUFFERED": "1",
    # Never write bytecode into a workspace an auditor is about to read.
    "PYTHONDONTWRITEBYTECODE": "1",
}
#: Windows cannot start a process without it; still not inherited anywhere else.
_WINDOWS_REQUIRED = ("SYSTEMROOT",)

# A file named ``credentials.json`` is normally secret material and remains
# protected everywhere.  One established public-source exception exists:
# flyto-i18n stores UI copy in ``locales/<product>/<locale>/credentials.json``.
# The action bridge may expose that catalog only while it is the exact regular,
# single-link file already staged in Git and while its bounded JSON shape proves
# it is a locale catalog.  A model cannot turn an arbitrary credentials file
# into an exception by editing it in the worktree.
_PUBLIC_LOCALE_CATALOG_BYTES = 64 * 1024
_PUBLIC_LOCALE_CATALOG_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _protected_paths(root: Path) -> List[Path]:
    """Paths inside the workspace that must never be visible to an action.

    Reuses the workspace policy rather than restating it, so `.git`, `.ssh`,
    `.env`, key material and the rest cannot drift between the read-only model
    boundary and this one.
    """

    # A bounded, read-only walk; `WorkspaceTools` owns the policy itself.
    probe = WorkspaceTools.__new__(WorkspaceTools)
    probe.root = root
    return [
        path for path in probe._protected_existing_paths()
        if not _is_unchanged_public_locale_catalog(root, path)
    ]


def _is_unchanged_public_locale_catalog(root: Path, path: Path) -> bool:
    """Recognise one non-secret use of the otherwise protected filename.

    The exception is intentionally narrower than "tracked files are safe".
    Real credentials are sometimes committed by mistake; tracking alone must
    never reveal them to repository code.  This requires the exact locale path,
    a strict translation-document schema, and byte-for-byte agreement with the
    Git index.  Any ambiguity falls back to masking.
    """

    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    parts = relative.parts
    if (
        len(parts) != 4
        or parts[0] != "locales"
        or parts[3].casefold() != "credentials.json"
        or not all(_PUBLIC_LOCALE_CATALOG_PART.fullmatch(part) for part in parts[1:3])
    ):
        return False
    try:
        if path.is_symlink() or any(
            (root.joinpath(*parts[:index])).is_symlink()
            for index in range(1, len(parts))
        ):
            return False
        info = path.stat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size > _PUBLIC_LOCALE_CATALOG_BYTES
        ):
            return False
    except OSError:
        return False

    git = _fixed_executable("git")
    if not git or not (root / ".git").exists():
        return False
    common = {
        "cwd": str(root),
        "env": ProjectActionExecutor._environment(),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "timeout": 10,
        "check": False,
    }
    try:
        tracked = subprocess.run(  # noqa: S603 - fixed Git query, literal path
            [git, "--literal-pathspecs", "ls-files", "--error-unmatch", "--", relative.as_posix()],
            **common,
        )
        unchanged = subprocess.run(  # noqa: S603 - fixed Git query, literal path
            [git, "--literal-pathspecs", "diff", "--quiet", "--no-ext-diff", "--", relative.as_posix()],
            **common,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if tracked.returncode != 0 or unchanged.returncode != 0:
        return False

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(document, dict) or set(document) != {
        "$schema", "locale", "category", "version", "translations",
    }:
        return False
    translations = document.get("translations")
    return bool(
        document.get("$schema") == "../../../schema/locale.schema.json"
        and document.get("locale") == parts[2]
        and document.get("category") == "{}.credentials".format(parts[1])
        and isinstance(document.get("version"), str)
        and isinstance(translations, dict)
        and 0 < len(translations) <= 256
        and all(
            isinstance(key, str)
            and key.startswith("credentials.")
            and isinstance(value, str)
            and len(value) <= 1000
            for key, value in translations.items()
        )
    )


def _container_command(argv: Sequence[str]) -> List[str]:
    """Re-point a host-spelled executable at the container image.

    A contract naturally declares the interpreter it uses on a developer
    machine. Inside the image that absolute path does not exist, so the leading
    element is reduced to its name and resolved on the image's own `PATH`. Only
    `argv[0]` is touched; every argument stays exactly as reviewed.
    """

    command = list(argv)
    if not command:  # pragma: no cover - the schema forbids an empty argv
        return command
    name = Path(command[0]).name
    if name.startswith("python"):
        command[0] = "python3"
    elif Path(command[0]).is_absolute():
        command[0] = name
    return command


class ProjectActionError(RuntimeError):
    """An action could not be run as declared."""

    code = "project_action_failed"


class UndeclaredAction(ProjectActionError):
    """The requested name is not in the source-controlled contract."""

    code = "project_action_undeclared"


class ActionSandboxUnavailable(ProjectActionError):
    """No isolation boundary exists, so no action may run at all.

    Raised at construction, so a caller finds out before a session is opened
    rather than at the moment a model asks for an action. There is deliberately
    no host-execution fallback: running repository-controlled code directly on
    the host is the failure this whole boundary exists to prevent, and
    "temporarily" doing it when the sandbox is missing would make the guarantee
    conditional on an operator's configuration rather than on the design.
    """

    code = ACTION_SANDBOX_UNAVAILABLE


class ProjectActionExecutor:
    """Host-owned runner for one workspace's declared actions."""

    def __init__(
        self,
        workspace: str,
        config_path: str = ".flyto/coding.yaml",
        *,
        sandbox_image: str,
        pinned_contract: Optional[ContractSnapshot] = None,
    ) -> None:
        # `strict=True`: a workspace that does not exist cannot be canonical,
        # and every containment check below is relative to this resolved root.
        self._root = Path(workspace).resolve(strict=True)
        self._config_path = config_path
        # The host-authorized action surface, by value. When present the file is
        # never read here: the surface a round may run is the one validated
        # before the first provider edit, so a session cannot widen its own
        # action authority by rewriting the contract it is running under, and a
        # legitimate contract edit does not strand the tool either.
        if pinned_contract is not None and not isinstance(pinned_contract, ContractSnapshot):
            raise ValueError("pinned_contract must be a ContractSnapshot")
        self._pinned = pinned_contract
        if not isinstance(sandbox_image, str) or not sandbox_image.strip():
            raise ActionSandboxUnavailable(_NO_SANDBOX)
        # Startup authority only. The image is the action's entire root
        # filesystem, so letting a request, a contract or a model choose it
        # would hand over the boundary along with the thing being bounded.
        self._image = sandbox_image
        #: The immutable content identity the probe resolved. Execution uses
        #: only this; `self._image` is a *locator* and is never launched.
        self._image_id = ""
        self._runtime: Optional[Path] = None
        self._cidfile: Optional[Path] = None
        self._docker = shutil.which("docker") or ""
        # Probing at construction is what makes the refusal *pre-session*.
        self._backend = self._detect_backend()
        if self._backend != "docker":
            raise ActionSandboxUnavailable(_NO_SANDBOX)
        if not _IMAGE_ID_RE.fullmatch(self._image_id):
            # A backend without a resolved identity is not a usable backend.
            raise ActionSandboxUnavailable(_NO_SANDBOX)

    def _detect_backend(self) -> str:
        """Resolve the startup image to one immutable identity, or refuse.

        Proving that a *tag* exists proves nothing about what will run: a local
        tag is mutable, so a probe that approved `image:pinned` and a launch
        that spells `image:pinned` can be two different root filesystems if the
        tag was retargeted in between. That is the same authority-substitution
        shape as the contract digest bug, one layer down.

        So the probe resolves the tag to its content identity and keeps only
        that. Anything ambiguous - no output, several lines, a value that is not
        a strict `sha256:<64 hex>` - is refused rather than guessed at, because
        a partially understood identity is not an identity.
        """

        if not self._docker:
            return ""
        try:
            probe = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [
                    self._docker, "image", "inspect",
                    "--format", "{{.Id}}", self._image,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
                text=True,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if probe.returncode != 0:
            return ""
        lines = [line.strip() for line in (probe.stdout or "").splitlines()]
        resolved = [line for line in lines if line]
        if len(resolved) != 1 or not _IMAGE_ID_RE.fullmatch(resolved[0]):
            # Several matches, empty output, or a shape this host does not
            # recognise. None of those name one thing to run.
            return ""
        self._image_id = resolved[0]
        return "docker"

    @property
    def root(self) -> Path:
        return self._root

    def _surface(self) -> Tuple[Tuple[ProjectActionSpec, ...], str]:
        """The action surface this executor is authorized to run, and its digest.

        One accessor for both, so the pair can never describe two documents.
        """

        if self._pinned is not None:
            return self._pinned.actions, self._pinned.config_sha256
        return load_project_actions(str(self._root), self._config_path)

    def declared(self) -> Tuple[ProjectActionSpec, ...]:
        """The authorized declared surface: the pin, or a fresh read."""

        actions, _digest = self._surface()
        return actions

    def digest(self) -> str:
        """Content address of the contract, for binding an authorization to it."""

        if self._pinned is not None:
            return self._pinned.config_sha256
        return config_digest(str(self._root), self._config_path)

    def run(
        self,
        name: str,
        *,
        expected_config_sha256: str = "",
    ) -> ProjectActionResult:
        """Run one declared action by name, or refuse.

        `expected_config_sha256` binds this invocation to the exact contract the
        caller was authorized against. Supplying it is what makes the call
        TOCTOU-safe; omitting it still re-reads and re-validates the contract,
        but cannot detect a substitution that happened in between.
        """

        if not isinstance(name, str) or not name:
            raise UndeclaredAction("an action name is required")
        actions, digest = self._surface()
        if expected_config_sha256 and expected_config_sha256 != digest:
            raise ProjectActionError(
                "the verification contract changed since this action was authorized",
            )
        selected = next((item for item in actions if item.name == name), None)
        if selected is None:
            # Deliberately does not echo the requested name: it came from a
            # model, and a refusal message is not a place to reflect one.
            raise UndeclaredAction("no such declared project action")
        return self._launch(selected)

    # -- internals --------------------------------------------------------

    def _resolve_cwd(self, action: ProjectActionSpec) -> Path:
        """Prove the working directory is a real directory inside the workspace.

        The spec already refused absolute paths and traversal as *text*. This
        is the second half, and the half that matters: the path is resolved on
        the real filesystem, refused if any component is a symbolic link, and
        checked for containment against the canonical root - so a link planted
        inside the workspace cannot walk out of it.
        """

        if not action.working_subdir:
            return self._root
        target = self._root.joinpath(*Path(action.working_subdir).parts)
        probe = self._root
        for part in Path(action.working_subdir).parts:
            probe = probe / part
            if probe.is_symlink():
                raise ProjectActionError("the action working directory is a symlink")
        resolved = Path(os.path.realpath(target))
        if resolved != self._root and self._root not in resolved.parents:
            raise ProjectActionError("the action working directory escapes the workspace")
        if not resolved.is_dir():
            raise ProjectActionError("the action working directory does not exist")
        return resolved

    def _sandboxed_argv(
        self, action: ProjectActionSpec, cwd: Path,
    ) -> Tuple[list, Path]:
        """Wrap one declared action in the host-owned isolation boundary.

        Returns the full launch argv and the cidfile path used to reap the
        container. The image is the one pinned at service startup; nothing on
        the request, the contract or the model's side of the boundary can name
        a different one, because an attacker-chosen image is an attacker-chosen
        root filesystem.
        """

        if self._backend != "docker":  # pragma: no cover - guarded at construction
            raise ActionSandboxUnavailable(_NO_SANDBOX)

        # Docker Desktop/Colima do not necessarily share the host's system
        # temporary directory with their VM. A private sibling of the claimed
        # workspace is on the same shared filesystem, remains outside the
        # writable workspace bind, and is exposed to the container only through
        # the single read-only manifest mount below.
        runtime = Path(tempfile.mkdtemp(
            prefix=".flyto-action-", dir=str(self._root.parent),
        ))
        self._runtime = runtime
        cidfile = runtime / "container.cid"
        try:
            tracked_manifest = self._tracked_files_manifest(runtime)
        except BaseException:
            shutil.rmtree(runtime, ignore_errors=True)
            self._runtime = None
            raise

        relative = cwd.relative_to(self._root).as_posix()
        workdir = CONTAINER_WORKSPACE
        if relative and relative != ".":
            workdir = "{}/{}".format(CONTAINER_WORKSPACE, relative)

        argv = [
            *container_hardening_argv(self._docker, str(cidfile)),
            # Container network isolation does not constrain the *daemon*.
            # Without this, an image that vanished between probe and launch
            # becomes an implicit registry pull - a network action, performed by
            # a privileged process, fetching content nobody authorized.
            "--pull=never",
            # The workspace is writable because regenerating a derived file is
            # the point. It is also the *only* writable thing: the container
            # root is read-only, and no other host path is bound.
            *container_workspace_mount_argv(self._root, writable=True),
            *container_runtime_argv(),
            "--mount",
            "type=bind,src={},dst={},readonly".format(
                tracked_manifest, _TRACKED_FILES_TARGET,
            ),
        ]
        if workdir != CONTAINER_WORKSPACE:
            argv.extend(["--workdir", workdir])
        # Writes must land as files this job's revision digest can attribute,
        # so the container runs as the host identity that owns the worktree
        # rather than as `nobody` - which could not write at all.
        if os.name == "posix":
            argv.extend(["--user", "{}:{}".format(os.getuid(), os.getgid())])
        argv.extend(self._masking_argv(runtime))
        # The resolved identity, never the tag. Retargeting the tag after the
        # probe cannot change what this launches.
        argv.append(self._image_id)
        argv.extend(_container_command(action.argv))
        return argv, cidfile

    def _tracked_files_manifest(self, runtime: Path) -> Path:
        """Export path names without exposing repository control metadata.

        Some established generators use ``git ls-files`` to select source
        inputs. Mounting `.git` would also expose history, remotes and helper
        configuration to repository code, so the trusted host runs one fixed
        read-only Git query instead. The container receives only a bounded,
        validated newline list through a read-only file mount.

        Non-Git workspaces get an empty list. A workspace that Git recognises
        but cannot enumerate is refused before its action can mutate files.
        """

        manifest = runtime / "tracked-files"
        manifest.write_text("", encoding="utf-8")
        git = _fixed_executable("git")
        if not git or not (self._root / ".git").exists():
            return manifest

        probe = subprocess.run(  # noqa: S603 - trusted binary, fixed argv
            [git, "-C", str(self._root), "rev-parse", "--is-inside-work-tree"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
            env=self._environment(),
        )
        if probe.returncode != 0 or probe.stdout.strip() != b"true":
            return manifest

        raw = runtime / "tracked-files.raw"
        try:
            with raw.open("wb") as output:
                listed = subprocess.run(  # noqa: S603 - trusted binary, fixed argv
                    [
                        git, "-C", str(self._root), "ls-files", "-z",
                        "--cached", "--others", "--exclude-standard",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                    check=False,
                    env=self._environment(),
                )
            if listed.returncode != 0:
                raise ProjectActionError(
                    "the tracked file manifest could not be produced",
                )
            if raw.stat().st_size > _MAX_TRACKED_MANIFEST_BYTES:
                raise ProjectActionError("the tracked file manifest is too large")
            encoded_paths = raw.read_bytes().split(b"\0")
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProjectActionError(
                "the tracked file manifest could not be produced",
            ) from exc
        finally:
            raw.unlink(missing_ok=True)

        if encoded_paths and encoded_paths[-1] == b"":
            encoded_paths.pop()
        if len(encoded_paths) > _MAX_TRACKED_MANIFEST_FILES:
            raise ProjectActionError("the tracked file manifest has too many files")

        safe_paths: List[str] = []
        for encoded in encoded_paths:
            try:
                relative = encoded.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise ProjectActionError(
                    "the tracked file manifest contains an invalid path",
                ) from exc
            path = Path(relative)
            if (
                not relative
                or path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
                or any(ord(character) < 32 for character in relative)
            ):
                raise ProjectActionError(
                    "the tracked file manifest contains an invalid path",
                )
            if WorkspaceTools._is_protected_relative(path):
                continue
            safe_paths.append(path.as_posix())

        manifest.write_text(
            "".join("{}\n".format(path) for path in sorted(safe_paths)),
            encoding="utf-8",
        )
        return manifest

    def _masking_argv(self, runtime: Path) -> list:
        """Hide `.git`, credentials and every other protected path from the action.

        A writable workspace mount would otherwise expose `.git` - history,
        remotes, and any credential helper configuration - to code the
        implementer just wrote. Directories are covered with an empty tmpfs and
        files with an empty read-only bind, which is the same masking the
        read-only model-command boundary applies. The placeholder must remain
        stat-able by Docker Desktop/Colima's VM; read-only emptiness, rather
        than host mode bits, is what prevents the real contents being exposed.
        """

        masked: list = []
        protected = _protected_paths(self._root)
        denied: Path | None = None
        if any(not path.is_dir() for path in protected):
            denied = runtime / "blocked-file"
            denied.touch(mode=0o444)
            denied.chmod(0o444)
        for path in protected:
            target = "{}/{}".format(
                CONTAINER_WORKSPACE, path.relative_to(self._root).as_posix(),
            )
            if path.is_dir():
                masked.extend([
                    "--mount", "type=tmpfs,dst={},tmpfs-mode=000".format(target),
                ])
            else:
                assert denied is not None
                masked.extend([
                    "--mount",
                    "type=bind,src={},dst={},readonly".format(denied, target),
                ])
        return masked

    def _reap_container(self) -> None:
        """Remove the container even when the client process was killed first.

        Killing the `docker run` client does not necessarily stop the container
        it started, so a timeout that only killed the process group would leave
        a container still writing into the worktree the host is about to hash.
        """

        cidfile = getattr(self, "_cidfile", None)
        if cidfile is None:
            return
        try:
            container_id = Path(cidfile).read_text(encoding="utf-8").strip()
        except OSError:
            container_id = ""
        # Only an id this executor's own private cidfile produced, and only if
        # it is shaped like one. A foreign or malformed value must never reach
        # `docker rm -f`: removing somebody else's container would be a far
        # worse outcome than leaking one of ours.
        if _CONTAINER_ID_RE.fullmatch(container_id):
            try:
                subprocess.run(  # noqa: S603 - fixed argv, no shell
                    [self._docker, "rm", "-f", container_id],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
                pass
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            shutil.rmtree(runtime, ignore_errors=True)
            self._runtime = None
        self._cidfile = None

    @staticmethod
    def _environment() -> Dict[str, str]:
        """The whole environment, built from constants rather than inherited."""

        env = dict(_STATIC_ENV)
        env["PATH"] = _fixed_path()
        if os.name != "posix":  # pragma: no cover - exercised on Windows only
            for key in _WINDOWS_REQUIRED:
                if key in os.environ:
                    env[key] = os.environ[key]
        return env

    def _launch(self, action: ProjectActionSpec) -> ProjectActionResult:
        cwd = self._resolve_cwd(action)
        started = time.monotonic()
        # Never `action.argv` directly. The declaration is digest-bound, but the
        # *code it runs* lives in a worktree the implementer can edit, so a
        # pinned `python scripts/generate.py` is a pinned instruction to execute
        # whatever that script says a moment later. Isolation is what closes
        # that, not the pin.
        argv, cidfile = self._sandboxed_argv(action, cwd)
        self._cidfile = cidfile
        try:
            process = subprocess.Popen(  # noqa: S603 - argv-direct, never a shell
                argv,
                cwd=str(self._root),
                env=self._environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                shell=False,
                # Its own process group, so a timeout kills the children too
                # instead of leaving them holding the workspace.
                start_new_session=True,
                # Bytes, not text: decoding is done once at the end, on a
                # already-bounded buffer, so invalid UTF-8 cannot desynchronize
                # an incremental decoder mid-stream.
                text=False,
            )
        except (OSError, ValueError) as exc:
            self._reap_container()
            return ProjectActionResult(
                name=action.name, ok=False, exit_code=None,
                duration_ms=int((time.monotonic() - started) * 1000),
                # Never the raw exception: it embeds absolute host paths.
                error=_launch_error(exc),
            )
        except BaseException:
            # Cancellation included: an orphaned container would keep writing
            # into the worktree after the round it belonged to had ended.
            self._reap_container()
            raise

        out_sink = _BoundedSink()
        err_sink = _BoundedSink()
        readers = [
            threading.Thread(
                target=_drain, args=(process.stdout, out_sink), daemon=True,
            ),
            threading.Thread(
                target=_drain, args=(process.stderr, err_sink), daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()

        # Everything from here on is inside `finally`, because once `Popen` has
        # returned there is a live container and only two acceptable outcomes:
        # it is reaped, or the process exits. `KeyboardInterrupt`, a
        # cancellation propagated through the bridge, or any other
        # `BaseException` raised out of `wait()` previously walked straight past
        # the reap and left a container writing into the claimed worktree after
        # its round had ended.
        timed_out = False
        try:
            try:
                process.wait(timeout=action.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate_group(process)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:  # pragma: no cover - unkillable
                    pass
            except BaseException:
                # Cancellation is not an action outcome. The group is killed and
                # the container removed, then the exception continues on its way
                # untranslated - a caller cancelling a round must not receive
                # something that reads like a result.
                self._terminate_group(process)
                raise
        finally:
            # The pipes close when every writer in the group exits, so joining
            # is bounded. A descendant that survived the kill would hold its end
            # open, so the join is bounded rather than trusted - including on
            # the exceptional path, where an unbounded join would hang the
            # cancellation it is supposed to be honouring.
            for reader in readers:
                reader.join(timeout=10)
            # Idempotent, and scoped to this executor's private cidfile.
            self._reap_container()
        duration_ms = int((time.monotonic() - started) * 1000)

        return ProjectActionResult(
            name=action.name,
            ok=(not timed_out and process.returncode == 0),
            exit_code=process.returncode,
            duration_ms=duration_ms,
            stdout=out_sink.text(),
            stderr=err_sink.text(),
            truncated=out_sink.truncated or err_sink.truncated,
            timed_out=timed_out,
            error=(
                "timed out" if timed_out
                else ("" if process.returncode == 0 else "non-zero exit")
            ),
        )

    @staticmethod
    def _terminate_group(process: subprocess.Popen) -> None:
        """Kill the whole group, and keep killing until the group is gone.

        The leader exiting on SIGTERM is not the end of the story: descendants
        it started are in the same process group and can outlive it, still
        writing into the worktree the host is about to hash. So the group is
        signalled, the leader is reaped, and the group is signalled again with
        SIGKILL - the second signal is what actually reaches a survivor that
        ignored or outlived the first.
        """

        try:
            group = os.getpgid(process.pid)
        except (OSError, AttributeError):  # pragma: no cover - platform without pgid
            try:
                process.kill()
            except OSError:
                pass
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                # The whole group is already gone.
                return
            except OSError:  # pragma: no cover - permission or platform refusal
                break
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                continue
        # Final sweep: the leader may have exited on the first signal while a
        # descendant kept running, in which case `wait` returned early and the
        # group still needs the hard kill.
        try:
            os.killpg(group, signal.SIGKILL)
        except (OSError, AttributeError):
            pass


class _BoundedSink:
    """Keep at most the declared bound of a stream, and count what was dropped.

    A bound applied *after* `communicate()` is not a bound at all: the pipe is
    drained into an unbounded list first, so a command that writes a gigabyte
    exhausts the host before anything is clipped. This keeps a fixed-size tail
    instead, so peak memory is the bound regardless of how much is produced,
    and there is no temporary file for the volume to move into.

    A tail rather than a head, because the end of a failing command's output is
    where the reason usually is.
    """

    __slots__ = ("_chunks", "_size", "seen", "truncated")

    def __init__(self) -> None:
        self._chunks: Deque[bytes] = deque()
        self._size = 0
        #: Total bytes ever observed, whether or not they were retained. This
        #: is the high-water mark a test can assert against.
        self.seen = 0
        self.truncated = False

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.seen += len(chunk)
        if len(chunk) > MAX_ACTION_OUTPUT_BYTES:
            # One oversized read is clipped before it is ever retained.
            chunk = chunk[-MAX_ACTION_OUTPUT_BYTES:]
            self.truncated = True
        self._chunks.append(chunk)
        self._size += len(chunk)
        while self._size > MAX_ACTION_OUTPUT_BYTES:
            oldest = self._chunks.popleft()
            drop = min(len(oldest), self._size - MAX_ACTION_OUTPUT_BYTES)
            if drop < len(oldest):
                self._chunks.appendleft(oldest[drop:])
            self._size -= drop
            self.truncated = True

    @property
    def retained(self) -> int:
        return self._size

    def text(self) -> str:
        # Decoded once, at the end, on bounded bytes: invalid UTF-8 becomes
        # replacement characters instead of raising or corrupting a stream.
        decoded = b"".join(self._chunks).decode("utf-8", errors="replace")
        if len(decoded) > MAX_ACTION_OUTPUT_CHARS:
            return decoded[-MAX_ACTION_OUTPUT_CHARS:]
        return decoded


def _drain(pipe, sink: _BoundedSink) -> None:
    """Read one pipe to EOF in fixed-size chunks, retaining only the bound.

    Both pipes are drained by their own thread, which is what makes a large
    stderr impossible to deadlock against a large stdout: neither writer can
    ever block on a full pipe waiting for a reader that is busy elsewhere.
    """

    if pipe is None:  # pragma: no cover - both pipes are always requested
        return
    try:
        while True:
            chunk = pipe.read(_READ_CHUNK_BYTES)
            if not chunk:
                return
            sink.feed(chunk)
    except (OSError, ValueError):  # pragma: no cover - pipe torn down by a kill
        return
    finally:
        try:
            pipe.close()
        except OSError:  # pragma: no cover
            pass


def _launch_error(exc: BaseException) -> str:
    """A stable reason a process could not start, with no host path in it.

    `FileNotFoundError` and friends stringify to the absolute path they tried,
    which would put the host's directory layout into a tool result a model
    reads. The errno name is the whole of what a caller needs.
    """

    name = getattr(type(exc), "__name__", "OSError")
    errno_name = errno.errorcode.get(getattr(exc, "errno", None) or 0, "")
    return "the action could not be started ({})".format(errno_name or name)


def _sanitized(text: str) -> str:
    """Strip anything that is not printable text from a model-visible string.

    A description reaches a tool catalog, so a control character in it is an
    injection into whatever renders that catalog: newlines that forge a second
    tool entry, escape sequences that rewrite a terminal. Dropped rather than
    escaped, because nothing legitimate needs them.
    """

    return "".join(
        char for char in text
        if char.isprintable() and char not in {"\u2028", "\u2029"}
    )[:200]


def action_catalog(actions: Sequence[ProjectActionSpec]) -> Tuple[Mapping[str, str], ...]:
    """The bounded description an implementer may see: names and purpose only.

    Never the argv. A model that can read the exact command is one prompt away
    from being asked to reproduce it somewhere less supervised, and it does not
    need it: it invokes by name.
    """

    return tuple(
        {"name": action.name, "description": _sanitized(action.description)}
        for action in actions
    )
