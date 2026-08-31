# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Fail-closed scope binding for strict validation-only coding jobs."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, Tuple

from flyto_ai.coding.errors import CodingServiceError, PlanAuthorityUnprovable


class ValidationRequest(Protocol):
    """The small request projection validation scope needs."""

    require_changes: bool
    working_dir: str


RevisionTarget = Callable[[Path, str], Path]
RevisionDigest = Callable[[str, Sequence[str]], str]


def planned_validation_scope(
    contract: Mapping[str, Any],
    request: ValidationRequest,
    *,
    max_files: int,
    revision_target: RevisionTarget,
    revision_digest: RevisionDigest,
) -> Tuple[Tuple[str, ...], str]:
    """Bind a validation-only plan to its exact existing Git candidate.

    A fresh implementation owns only bytes changed after its baseline. A
    validation-only recovery instead owns the sealed Indexer ledger, but only
    when every allowed path is already a real dirty path and the complete set
    can be revision-bound before the implementer runs.
    """

    if request.require_changes is not False:
        return (), ""
    ledger = contract.get("intent_ledger")
    allowed = ledger.get("allowed_paths") if isinstance(ledger, Mapping) else None
    if not isinstance(allowed, (list, tuple)) or not allowed:
        raise PlanAuthorityUnprovable("cumulative_scope_unproven")
    scope = [
        str(item) for item in allowed
        if isinstance(item, str) and not isinstance(item, bool)
    ]
    if (
        len(scope) != len(allowed)
        or len(scope) != len(set(scope))
        or len(scope) > max_files
    ):
        raise PlanAuthorityUnprovable("cumulative_scope_unbounded")
    root = Path(request.working_dir).resolve()
    for relative in scope:
        try:
            revision_target(root, relative)
        except CodingServiceError:
            raise PlanAuthorityUnprovable("cumulative_scope_unsafe") from None
    if not set(scope).issubset(_git_dirty_paths(root, max_files=max_files)):
        raise PlanAuthorityUnprovable("cumulative_scope_unproven")
    ordered = tuple(sorted(scope))
    try:
        revision = revision_digest(request.working_dir, ordered)
    except CodingServiceError:
        raise PlanAuthorityUnprovable("cumulative_scope_unproven") from None
    return ordered, revision


def _git_dirty_paths(root: Path, *, max_files: int) -> frozenset[str]:
    """Return one bounded, hook-free Git dirty-path inventory."""

    commands = (
        (
            "git", "-C", str(root), "diff", "--no-ext-diff", "--no-renames",
            "--name-only", "-z", "HEAD", "--",
        ),
        (
            "git", "-C", str(root), "ls-files", "--others",
            "--exclude-standard", "-z", "--",
        ),
    )
    paths: set[str] = set()
    total = 0
    for command in commands:
        try:
            completed = subprocess.run(
                command, capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise PlanAuthorityUnprovable("cumulative_scope_unproven") from None
        total += len(completed.stdout)
        if completed.returncode != 0 or total > 4 * 1024 * 1024:
            raise PlanAuthorityUnprovable("cumulative_scope_unproven")
        try:
            decoded = completed.stdout.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise PlanAuthorityUnprovable("cumulative_scope_unsafe") from None
        paths.update(item for item in decoded.split("\x00") if item)
        if len(paths) > max_files:
            raise PlanAuthorityUnprovable("cumulative_scope_unbounded")
    return frozenset(paths)
