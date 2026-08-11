# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Source-controlled configuration and real verification checks."""
from __future__ import annotations

import hashlib
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import yaml

from flyto_ai.coding.contracts import (
    CONFIG_VERSION,
    CapabilitySpec,
    CheckResult,
    CheckSpec,
    ContractSnapshot,
    ProjectActionSpec,
)
from flyto_ai.coding.workspace import WorkspaceTools, resolve_executable


MAX_CONFIG_BYTES = 256 * 1024
#: A repository may declare a small, reviewable set of named actions.
MAX_PROJECT_ACTIONS = 16


#: Read one byte past the bound so "at the bound" is distinguishable from
#: "over the bound" without loading whatever else is in the file.
_MAX_CONFIG_READ = MAX_CONFIG_BYTES + 1
_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


@dataclass(frozen=True)
class ProjectContract:
    """One repository contract, parsed once from one exact read.

    Checks, capabilities and actions all come from this single object so no two
    consumers can ever parse different bytes: preflight cannot approve one
    document while an action executor authorizes against another.
    """

    checks: Tuple[CheckSpec, ...] = ()
    capabilities: Tuple[CapabilitySpec, ...] = ()
    actions: Tuple[ProjectActionSpec, ...] = ()
    #: SHA-256 of the exact bytes parsed, or "" when no contract exists.
    digest: str = ""
    present: bool = False

    def snapshot(self) -> ContractSnapshot:
        """Pin this contract by value, for a job that must keep executing it.

        Only ever called on a contract that has just been read and validated,
        which is why the digest is required rather than defaulted: a snapshot
        that cannot name the document it came from is not auditable.
        """

        return ContractSnapshot(
            checks=self.checks,
            capabilities=self.capabilities,
            actions=self.actions,
            config_sha256=self.digest,
        )


def contract_from_snapshot(snapshot: ContractSnapshot) -> ProjectContract:
    """View a pinned snapshot as the contract a round executes.

    The one adapter between "what the host authorized" and "what this round
    runs", so no consumer has to learn two shapes. `present` is `True` because
    a pin only exists for a contract that really was read and validated - the
    absence case never produces one.
    """

    return ProjectContract(
        checks=snapshot.checks,
        capabilities=snapshot.capabilities,
        actions=snapshot.actions,
        digest=snapshot.config_sha256,
        present=True,
    )


def round_contract(
    workspace: str,
    relative_path: str,
    *,
    pinned: Optional[ContractSnapshot] = None,
) -> ProjectContract:
    """The contract one implementation round must derive everything from.

    A pinned snapshot wins outright and the file is not read at all. That is the
    whole fix: re-reading is what let an implementation's own edit authorize
    itself, and refusing on a digest mismatch is what made a legitimate contract
    change impossible to finish. Executing the pin does neither.

    With no pin this is exactly the historical read, so every caller that has
    not been handed one behaves as it always did.
    """

    if pinned is not None:
        return contract_from_snapshot(pinned)
    return read_project_contract(workspace, relative_path)


#: Every no-follow primitive the descriptor walk depends on. A host missing any
#: of them cannot be served the guarantee this reader advertises, so it is
#: refused rather than quietly downgraded to a pathname walk.
_REQUIRED_OPEN_FLAGS = ("O_NOFOLLOW", "O_DIRECTORY", "O_CLOEXEC")


def _require_nofollow_support() -> None:
    if any(not hasattr(os, name) for name in _REQUIRED_OPEN_FLAGS):
        raise ValueError(
            "this host cannot open the coding config without following links",
        )


def _open_contract_descriptor(workspace: str, relative_path: str) -> int:
    """Walk to the contract by descriptor, so no component can be swapped.

    A pathname walk cannot be made safe. Checking ``is_symlink()`` on each
    component and then opening the assembled path is two separate resolutions of
    the same name: between them any parent directory can be replaced with a
    symbolic link, and the open follows the replacement. The check described the
    directory that *was* there, not the one that gets opened.

    So each component is opened relative to the descriptor of the one before it,
    with ``O_NOFOLLOW`` and ``O_DIRECTORY``. A directory swapped after it has
    been opened is irrelevant: the descriptor still refers to the original
    inode, and the next component resolves inside *that*, not inside whatever
    now bears the name. Containment stops being a string comparison and becomes
    a property of the walk itself.

    Returns an open descriptor for the final candidate; the caller owns closing
    it. Raises ``FileNotFoundError`` when the contract is simply absent.
    """

    _require_nofollow_support()
    raw = Path(relative_path)
    if raw.is_absolute() or not raw.parts or any(
        part in {"", ".", ".."} for part in raw.parts
    ):
        raise ValueError("coding config path must be a safe relative path")

    root = Path(workspace).resolve(strict=True)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        parent = os.open(root, directory_flags)
    except OSError as exc:
        raise ValueError("the coding workspace root cannot be opened safely") from exc

    try:
        for part in raw.parts[:-1]:
            try:
                nested = os.open(part, directory_flags, dir_fd=parent)
            except FileNotFoundError:
                raise
            except OSError as exc:
                # ELOOP for a symlinked component, ENOTDIR for a file where a
                # directory was expected.
                raise ValueError(
                    "coding config path crosses a symbolic link or non-directory",
                ) from exc
            os.close(parent)
            parent = nested
        try:
            return os.open(raw.parts[-1], _OPEN_FLAGS, dir_fd=parent)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ValueError("coding config cannot be opened safely") from exc
    finally:
        os.close(parent)


def read_project_contract(
    workspace: str,
    relative_path: str = ".flyto/coding.yaml",
) -> ProjectContract:
    """Parse one contract from one bounded, substitution-checked read.

    The read is the security boundary, so it is done against a descriptor
    rather than a path: opened ``O_NOFOLLOW`` so a symlinked final component is
    refused outright, ``fstat``-ed to insist on a regular file, read to at most
    one byte past the bound, and ``fstat``-ed again afterwards. If the identity
    or size changed between those two stats, something replaced or grew the
    file mid-read and the contract is refused rather than parsed.

    ``O_NONBLOCK`` is not an optimisation: a FIFO planted at the contract path
    would otherwise block the opening thread forever.
    """

    try:
        descriptor = _open_contract_descriptor(workspace, relative_path)
    except FileNotFoundError:
        return ProjectContract()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("coding config is not a regular file")
        if before.st_size > MAX_CONFIG_BYTES:
            raise ValueError("coding config exceeds the size bound")
        chunks, remaining = [], _MAX_CONFIG_READ
        while remaining > 0:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > MAX_CONFIG_BYTES:
            raise ValueError("coding config exceeds the size bound")
        after = os.fstat(descriptor)
        if (
            after.st_ino != before.st_ino
            or after.st_dev != before.st_dev
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise ValueError("coding config changed while it was being read")
    finally:
        os.close(descriptor)

    if not raw.strip():
        return ProjectContract(digest=hashlib.sha256(raw).hexdigest(), present=True)
    try:
        loaded = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("coding config is not readable YAML") from exc
    if not isinstance(loaded, dict):
        raise ValueError("coding config must be an object")
    if loaded.get("version") != CONFIG_VERSION:
        raise ValueError("coding config version is unsupported")
    unknown = set(loaded) - {"version", "checks", "capabilities", "actions"}
    if unknown:
        raise ValueError(
            "coding config contains unsupported keys: {}".format(", ".join(sorted(unknown))),
        )

    raw_checks = loaded.get("checks", [])
    raw_capabilities = loaded.get("capabilities", [])
    raw_actions = loaded.get("actions", [])
    if not isinstance(raw_checks, list) or len(raw_checks) > 32:
        raise ValueError("coding config checks must be an array of at most 32 items")
    if not isinstance(raw_capabilities, list) or len(raw_capabilities) > 16:
        raise ValueError("coding config capabilities must be an array of at most 16 items")
    if not isinstance(raw_actions, list) or len(raw_actions) > MAX_PROJECT_ACTIONS:
        raise ValueError(
            "coding config actions must be an array of at most {} items".format(
                MAX_PROJECT_ACTIONS,
            ),
        )
    checks = tuple(CheckSpec.from_mapping(item) for item in raw_checks if isinstance(item, dict))
    capabilities = tuple(
        CapabilitySpec.from_mapping(item) for item in raw_capabilities if isinstance(item, dict)
    )
    if len(checks) != len(raw_checks) or len(capabilities) != len(raw_capabilities):
        raise ValueError("coding config entries must be objects")
    proof_kinds = [kind for check in checks for kind in check.proof_kinds]
    if len(set(proof_kinds)) != len(proof_kinds):
        raise ValueError("coding config proof_kinds must be unique")
    actions = tuple(ProjectActionSpec.from_mapping(item) for item in raw_actions)
    names = [action.name for action in actions]
    if len(set(names)) != len(names):
        raise ValueError("coding config actions contain duplicate names")
    return ProjectContract(
        checks=checks,
        capabilities=capabilities,
        actions=actions,
        digest=hashlib.sha256(raw).hexdigest(),
        present=True,
    )


def load_project_config(
    workspace: str,
    relative_path: str = ".flyto/coding.yaml",
) -> Tuple[Tuple[CheckSpec, ...], Tuple[CapabilitySpec, ...]]:
    """Load one bounded YAML contract without environment expansion.

    Kept at its original two-tuple signature so every existing caller is
    unaffected. It now delegates to `read_project_contract`, so a malformed
    `actions:` block fails this call too - a contract is valid as a whole or
    not at all, and letting checks parse while actions quietly failed is how
    preflight would pass a repository whose action surface is broken.
    """

    contract = read_project_contract(workspace, relative_path)
    return contract.checks, contract.capabilities


def config_digest(workspace: str, relative_path: str = ".flyto/coding.yaml") -> str:
    """Content address of the source-controlled contract, or "" when absent."""

    return read_project_contract(workspace, relative_path).digest


def load_project_actions(
    workspace: str,
    relative_path: str = ".flyto/coding.yaml",
) -> Tuple[Tuple[ProjectActionSpec, ...], str]:
    """The declared action surface and the digest of the exact bytes it came from.

    One read produces both, so the pair can never describe two different
    documents - the race the previous two-call version had.
    """

    contract = read_project_contract(workspace, relative_path)
    return contract.actions, contract.digest


class VerificationToolUnavailable(RuntimeError):
    """A required check cannot be launched, so no verification verdict exists.

    Carries the affected check names - contract identifiers, already validated
    as safe tokens - and nothing else. No path, no argv, no exception text.
    """

    def __init__(self, blockers: Sequence[str]) -> None:
        super().__init__("required verification tool is not installed")
        self.blockers = tuple(blockers)


def unlaunchable_required_checks(
    checks: Sequence[CheckSpec], workspace: Optional[str] = None,
) -> Tuple[str, ...]:
    """Required checks whose program cannot be launched, by name.

    Delegates to the one resolver the runner itself uses, so this can never
    answer differently from the process that actually starts the command.
    """

    blocked: list = []
    for check in checks:
        if not getattr(check, "required", False) or not check.argv:
            continue
        if (
            resolve_executable(check.argv[0], workspace) is None
            and check.name not in blocked
        ):
            blocked.append(check.name)
    return tuple(blocked)


class CheckRunner:
    """Execute declared checks and retain content-addressed evidence."""

    def __init__(self, workspace_tools: WorkspaceTools) -> None:
        self.workspace_tools = workspace_tools

    async def run(self, checks: Sequence[CheckSpec]) -> List[CheckResult]:
        """Run declared checks, refusing outright if a required one cannot start.

        Preflight should have caught this long before a session existed, so
        reaching it here means a contract race or an invariant bug. It is still
        worth failing precisely: running the rest and reporting "verification
        did not pass" would describe the change, when the change was never the
        problem, and the operator would go looking in the wrong place. The typed
        refusal names the checks instead, which is the same answer preflight
        would have given and the same work it would have asked for.
        """

        unlaunchable = unlaunchable_required_checks(
            checks, str(self.workspace_tools.root),
        )
        if unlaunchable:
            raise VerificationToolUnavailable(unlaunchable)
        results: List[CheckResult] = []
        for check in checks:
            started = time.monotonic()
            try:
                raw = await self.workspace_tools.run_check(check.argv, check.timeout_seconds)
                output = str(raw.get("output", ""))
                results.append(CheckResult(
                    name=check.name,
                    passed=bool(raw.get("ok")),
                    required=check.required,
                    exit_code=raw.get("exit_code"),
                    duration_ms=int(raw.get("duration_ms", (time.monotonic() - started) * 1000)),
                    output_sha256=str(raw.get("output_sha256") or hashlib.sha256(output.encode()).hexdigest()),
                    output_preview=output[-4000:],
                    error=None if raw.get("ok") else ("timed out" if raw.get("timed_out") else "non-zero exit"),
                ))
            except Exception as exc:
                message = str(exc)[:1000]
                results.append(CheckResult(
                    name=check.name,
                    passed=False,
                    required=check.required,
                    exit_code=None,
                    duration_ms=int((time.monotonic() - started) * 1000),
                    output_sha256=hashlib.sha256(message.encode()).hexdigest(),
                    error=message,
                ))
        return results

    @staticmethod
    def passed(results: Sequence[CheckResult]) -> bool:
        return bool(results) and all(result.passed for result in results if result.required)
