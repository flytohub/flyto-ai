# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Host-global workspace-root brokering.

On 2026-08-11 one service used a `coding-service` state root and another used
`coding-service-<suffix>`, both configured with the same workspace root, and
two sessions edited one checkout concurrently. Each service was internally
consistent: workspace claims live *under* a state root, so neither could see
the other. These tests pin the layer that sits above both.

Every lock here is a real `flock`. The multi-process cases use real
subprocesses, because a same-process lock would prove nothing about the
invariant being claimed.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from flyto_ai.coding.service import CodingService
from flyto_ai.coding.workspace_authority import (
    WORKSPACE_AUTHORITY_VERSION,
    WorkspaceAuthorityConflict,
    WorkspaceAuthorityUnavailable,
    WorkspaceRootAuthority,
    canonical_workspace_root,
    describe_workspace_root,
    state_root_has_open_work,
    workspace_digest,
)

pytest.importorskip("fcntl")


#: Joins the registry, reports success, then blocks until told to stop. Holding
#: the lease in a *separate process* is what makes the contention real.
_HOLDER = r"""
import sys
from flyto_ai.coding.workspace_authority import WorkspaceRootAuthority

authority = WorkspaceRootAuthority(sys.argv[1])
try:
    authority.join(state_root=sys.argv[2], workspace_roots=(sys.argv[3],))
except Exception as exc:
    sys.stdout.write("refused:" + type(exc).__name__ + "\n")
    sys.stdout.flush()
    raise SystemExit(3)
sys.stdout.write("held\n")
sys.stdout.flush()
sys.stdin.readline()
authority.release()
"""


def _registry(tmp_path: Path) -> Path:
    return Path(os.path.realpath(tmp_path)) / "registry"


def _tree(tmp_path: Path, name: str) -> Path:
    path = Path(os.path.realpath(tmp_path)) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _spawn_holder(registry: Path, state_root: Path, workspace: Path):
    script = Path(os.path.realpath(registry.parent)) / "holder.py"
    script.write_text(_HOLDER, encoding="utf-8")
    process = subprocess.Popen(
        (sys.executable, "-u", str(script), str(registry), str(state_root), str(workspace)),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert process.stdout is not None
    return process, process.stdout.readline()


def _stop(process) -> None:
    try:
        if process.poll() is None:
            assert process.stdin is not None
            process.stdin.write(b"\n")
            process.stdin.flush()
            process.wait(timeout=30)
    finally:
        if process.poll() is None:  # pragma: no cover - defensive
            process.kill()
            process.wait(timeout=30)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()


# --- 1. same identity, many processes --------------------------------------


def test_one_state_root_hosts_many_live_processes(tmp_path: Path) -> None:
    """Many Codex clients share one queue, so they must share one tree."""

    registry = _registry(tmp_path)
    state_root = _tree(tmp_path, "state")
    workspace = _tree(tmp_path, "workspace")

    first, line = _spawn_holder(registry, state_root, workspace)
    try:
        assert line == b"held\n"
        second, second_line = _spawn_holder(registry, state_root, workspace)
        try:
            assert second_line == b"held\n"
            # And a third, in this process, joins the same shared lease.
            local = WorkspaceRootAuthority(registry)
            local.join(state_root=state_root, workspace_roots=(workspace,))
            assert local.held_digests == [workspace_digest(workspace)]
            local.release()
        finally:
            _stop(second)
    finally:
        _stop(first)


# --- 2. two state roots, one tree ------------------------------------------


def test_a_second_state_root_cannot_take_a_live_tree(tmp_path: Path) -> None:
    """The exact 2026-08-11 collision, refused."""

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    first_root = _tree(tmp_path, "coding-service")
    second_root = _tree(tmp_path, "coding-service-suffix")

    holder, line = _spawn_holder(registry, first_root, workspace)
    try:
        assert line == b"held\n"
        intruder = WorkspaceRootAuthority(registry)
        with pytest.raises(WorkspaceAuthorityConflict) as excinfo:
            intruder.join(state_root=second_root, workspace_roots=(workspace,))
        assert excinfo.value.code == "workspace_authority_conflict"
        assert intruder.held_digests == []
    finally:
        _stop(holder)


# --- 3. disjoint trees ------------------------------------------------------


def test_disjoint_workspaces_proceed_independently(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    holder, line = _spawn_holder(
        registry, _tree(tmp_path, "state-a"), _tree(tmp_path, "workspace-a"),
    )
    try:
        assert line == b"held\n"
        other = WorkspaceRootAuthority(registry)
        other.join(
            state_root=_tree(tmp_path, "state-b"),
            workspace_roots=(_tree(tmp_path, "workspace-b"),),
        )
        assert len(other.held_digests) == 1
        other.release()
    finally:
        _stop(holder)


# --- 4. crash releases the lease, durable identity still blocks -------------


def test_a_crash_frees_the_lease_but_open_work_still_blocks_adoption(
    tmp_path: Path,
) -> None:
    """The kernel releases the lock; the recorded identity keeps the tree.

    A crash must not hand a tree to whoever restarts first while the previous
    owner still has unresolved work — that is how an audit-pending job would
    be silently orphaned.
    """

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    owner_root = _tree(tmp_path, "owner-state")
    other_root = _tree(tmp_path, "other-state")

    holder, line = _spawn_holder(registry, owner_root, workspace)
    assert line == b"held\n"
    holder.kill()
    holder.wait(timeout=30)
    for stream in (holder.stdin, holder.stdout):
        if stream is not None:
            stream.close()

    # The lease is free: the same state root re-joins immediately.
    resumed = WorkspaceRootAuthority(registry)
    resumed.join(state_root=owner_root, workspace_roots=(workspace,))
    resumed.release()

    # Now give the dead owner unresolved durable work.
    jobs = owner_root / "tenants" / "tenant-a" / "jobs"
    jobs.mkdir(parents=True)
    (jobs / "job_a.json").write_text(
        json.dumps({"state": "awaiting_codex_audit"}), encoding="utf-8",
    )
    assert state_root_has_open_work(owner_root) is True

    intruder = WorkspaceRootAuthority(registry)
    with pytest.raises(WorkspaceAuthorityConflict):
        intruder.join(state_root=other_root, workspace_roots=(workspace,))
    assert intruder.held_digests == []


# --- 5. bounded recovery ----------------------------------------------------


def test_a_terminal_owner_is_recovered_without_editing_json(tmp_path: Path) -> None:
    """Once the previous owner's work is finished, adoption is automatic.

    This is the operator recovery path: finish or retire the old work — with
    the host release valve if an audit is stranded — and the next start takes
    the tree. No hand-edited registry file, and no flag that skips the check.
    """

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    owner_root = _tree(tmp_path, "owner-state")
    other_root = _tree(tmp_path, "other-state")

    first = WorkspaceRootAuthority(registry)
    first.join(state_root=owner_root, workspace_roots=(workspace,))
    first.release()

    jobs = owner_root / "tenants" / "tenant-a" / "jobs"
    jobs.mkdir(parents=True)
    record = jobs / "job_a.json"
    record.write_text(json.dumps({"state": "awaiting_codex_audit"}), encoding="utf-8")

    successor = WorkspaceRootAuthority(registry)
    with pytest.raises(WorkspaceAuthorityConflict):
        successor.join(state_root=other_root, workspace_roots=(workspace,))

    # The valve's outcome for an abandoned job, reached without touching the
    # registry: the job becomes terminal.
    record.write_text(
        json.dumps({"state": "failed", "failure_code": "job_abandoned"}),
        encoding="utf-8",
    )
    assert state_root_has_open_work(owner_root) is False

    successor.join(state_root=other_root, workspace_roots=(workspace,))
    try:
        assert successor.held_digests == [workspace_digest(workspace)]
        entry = json.loads(
            (registry / "{}.json".format(workspace_digest(workspace)))
            .read_text(encoding="utf-8"),
        )
        assert entry["version"] == WORKSPACE_AUTHORITY_VERSION
        assert entry["state_root"] == str(other_root)
    finally:
        successor.release()


def test_inactive_legacy_parent_does_not_serialize_new_repo_set_children(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    parent = _tree(tmp_path, "flytohub")
    alpha = _tree(parent, "flyto-code")
    beta = _tree(parent, "flyto-engine")
    owner_root = _tree(tmp_path, "owner-state")
    other_root = _tree(tmp_path, "other-state")

    historical = WorkspaceRootAuthority(registry)
    historical.join(state_root=owner_root, workspace_roots=(parent,))
    historical.release()

    jobs = owner_root / "tenants" / "tenant-a" / "jobs"
    jobs.mkdir(parents=True)
    (jobs / "job_a.json").write_text(json.dumps({
        "job_id": "job_" + "a" * 24,
        "state": "running",
        "working_dir": str(alpha),
        "repository_roots": [str(alpha)],
        "repository_digests": [workspace_digest(alpha)],
    }), encoding="utf-8")
    current = WorkspaceRootAuthority(registry)
    current.join(state_root=owner_root, workspace_roots=(alpha,))

    parallel = WorkspaceRootAuthority(registry)
    try:
        parallel.join(state_root=other_root, workspace_roots=(beta,))
        assert parallel.held_digests == [workspace_digest(beta)]
    finally:
        parallel.release()
        current.release()


def test_incremental_repo_set_failure_keeps_old_holds_and_takes_no_partial_set(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    alpha = _tree(tmp_path, "alpha")
    beta = _tree(tmp_path, "beta")
    blocked = _tree(tmp_path, "blocked")
    owner_root = _tree(tmp_path, "owner-state")
    foreign_root = _tree(tmp_path, "foreign-state")

    owner = WorkspaceRootAuthority(registry)
    owner.join(state_root=owner_root, workspace_roots=(alpha,))
    foreign = WorkspaceRootAuthority(registry)
    foreign.join(state_root=foreign_root, workspace_roots=(blocked,))
    try:
        with pytest.raises(WorkspaceAuthorityConflict):
            owner.join(
                state_root=owner_root, workspace_roots=(beta, blocked),
            )
        assert owner.held_digests == [workspace_digest(alpha)]
        report = describe_workspace_root(registry, beta)
        assert report["status"] == "adoptable"
        assert any(
            item["relationship"] == "exact" and item["status"] == "adoptable"
            for item in report["owners"]
        )
    finally:
        foreign.release()
        owner.release()


def test_a_surviving_workspace_claim_keeps_legacy_work_fail_closed(
    tmp_path: Path,
) -> None:
    """Migration must not make audit-pending work look free."""

    owner_root = _tree(tmp_path, "owner-state")
    claims = owner_root / "locks" / "workspaces"
    claims.mkdir(parents=True)
    (claims / "some-claim.owner.json").write_text("{}", encoding="utf-8")

    assert state_root_has_open_work(owner_root) is True


def test_a_persistent_workspace_lock_is_not_open_work(tmp_path: Path) -> None:
    """An unlocked flock rendezvous file carries no durable ownership."""

    owner_root = _tree(tmp_path, "owner-state")
    claims = owner_root / "locks" / "workspaces"
    claims.mkdir(parents=True)
    (claims / "workspace.lock").write_text("", encoding="utf-8")

    assert state_root_has_open_work(owner_root) is False


# --- 6. aliases and races ---------------------------------------------------


def test_a_symlinked_alias_contends_for_the_same_entry(tmp_path: Path) -> None:
    """Naming the tree through a link is still the same tree."""

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    alias = Path(os.path.realpath(tmp_path)) / "alias"
    alias.symlink_to(workspace, target_is_directory=True)

    assert canonical_workspace_root(alias) == canonical_workspace_root(workspace)
    assert workspace_digest(alias) == workspace_digest(workspace)

    holder, line = _spawn_holder(registry, _tree(tmp_path, "state-a"), workspace)
    try:
        assert line == b"held\n"
        intruder = WorkspaceRootAuthority(registry)
        with pytest.raises(WorkspaceAuthorityConflict):
            intruder.join(
                state_root=_tree(tmp_path, "state-b"), workspace_roots=(alias,),
            )
    finally:
        _stop(holder)


def test_a_relative_and_dotted_alias_contend_for_the_same_entry(
    tmp_path: Path,
) -> None:
    """`..`, `.` and a trailing separator are not new trees."""

    workspace = _tree(tmp_path, "workspace")
    dotted = workspace / "." / ".." / "workspace"
    assert workspace_digest(dotted) == workspace_digest(workspace)
    assert workspace_digest(str(workspace) + "/") == workspace_digest(workspace)


def test_a_child_of_a_live_owned_tree_is_refused(tmp_path: Path) -> None:
    """Editing a subdirectory of somebody's tree is editing their tree."""

    registry = _registry(tmp_path)
    parent = _tree(tmp_path, "workspace")
    child = _tree(tmp_path, "workspace/nested")

    holder, line = _spawn_holder(registry, _tree(tmp_path, "state-a"), parent)
    try:
        assert line == b"held\n"
        intruder = WorkspaceRootAuthority(registry)
        with pytest.raises(WorkspaceAuthorityConflict):
            intruder.join(
                state_root=_tree(tmp_path, "state-b"), workspace_roots=(child,),
            )
    finally:
        _stop(holder)


def test_a_parent_of_a_live_owned_child_is_refused(tmp_path: Path) -> None:
    """The other direction of the same collision, and equally refused.

    A service on `<tree>` and a service on `<tree>/child` write the same
    files. Deciding this requires a registry scan, which is only sound under
    the registry-wide join lock — so this is the test that the lock exists and
    is actually used.
    """

    registry = _registry(tmp_path)
    parent = _tree(tmp_path, "workspace")
    child = _tree(tmp_path, "workspace/nested")

    holder, line = _spawn_holder(registry, _tree(tmp_path, "state-a"), child)
    try:
        assert line == b"held\n"
        outer = WorkspaceRootAuthority(registry)
        with pytest.raises(WorkspaceAuthorityConflict) as excinfo:
            outer.join(
                state_root=_tree(tmp_path, "state-b"), workspace_roots=(parent,),
            )
        assert excinfo.value.code == "workspace_authority_conflict"
        assert outer.held_digests == []
    finally:
        _stop(holder)


def test_a_crashed_child_owner_with_open_work_blocks_the_parent(
    tmp_path: Path,
) -> None:
    """Nested trees are not adoptable on easier terms than identical ones."""

    registry = _registry(tmp_path)
    parent = _tree(tmp_path, "workspace")
    child = _tree(tmp_path, "workspace/nested")
    child_root = _tree(tmp_path, "child-state")

    holder, line = _spawn_holder(registry, child_root, child)
    assert line == b"held\n"
    holder.kill()
    holder.wait(timeout=30)
    for stream in (holder.stdin, holder.stdout):
        if stream is not None:
            stream.close()

    jobs = child_root / "tenants" / "tenant-a" / "jobs"
    jobs.mkdir(parents=True)
    (jobs / "job_a.json").write_text(
        json.dumps({"state": "awaiting_codex_audit"}), encoding="utf-8",
    )

    outer = WorkspaceRootAuthority(registry)
    with pytest.raises(WorkspaceAuthorityConflict):
        outer.join(
            state_root=_tree(tmp_path, "state-b"), workspace_roots=(parent,),
        )
    assert outer.held_digests == []


def test_a_terminal_child_owner_lets_the_parent_proceed(tmp_path: Path) -> None:
    """Fail-closed, not permanently closed: finished work releases the tree."""

    registry = _registry(tmp_path)
    parent = _tree(tmp_path, "workspace")
    child = _tree(tmp_path, "workspace/nested")

    first = WorkspaceRootAuthority(registry)
    first.join(state_root=_tree(tmp_path, "child-state"), workspace_roots=(child,))
    first.release()

    outer = WorkspaceRootAuthority(registry)
    outer.join(state_root=_tree(tmp_path, "state-b"), workspace_roots=(parent,))
    try:
        assert outer.held_digests == [workspace_digest(parent)]
    finally:
        outer.release()


def test_a_concurrent_parent_child_start_race_has_one_winner(
    tmp_path: Path,
) -> None:
    """Both directions decided atomically under a real concurrent start.

    Half the processes claim the parent and half the nested child. Exactly one
    may win, whichever it is: the registry-wide lock serialises the decision,
    so there is no interleaving in which a parent and a child both believe
    they own their tree.
    """

    registry = _registry(tmp_path)
    parent = _tree(tmp_path, "workspace")
    child = _tree(tmp_path, "workspace/nested")
    script = Path(os.path.realpath(tmp_path)) / "holder.py"
    script.write_text(_HOLDER, encoding="utf-8")

    processes = [
        subprocess.Popen(
            (
                sys.executable, "-u", str(script), str(registry),
                str(_tree(tmp_path, "state-{}".format(index))),
                str(parent if index % 2 else child),
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        for index in range(6)
    ]
    try:
        lines = []
        for process in processes:
            assert process.stdout is not None
            lines.append(process.stdout.readline())
        assert lines.count(b"held\n") == 1
        assert all(
            line.startswith(b"refused:") for line in lines if line != b"held\n"
        )
    finally:
        for process in processes:
            _stop(process)


def test_a_concurrent_start_race_has_exactly_one_winner(tmp_path: Path) -> None:
    """Two different state roots starting at once cannot both win."""

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    script = Path(os.path.realpath(tmp_path)) / "holder.py"
    script.write_text(_HOLDER, encoding="utf-8")

    processes = [
        subprocess.Popen(
            (
                sys.executable, "-u", str(script), str(registry),
                str(_tree(tmp_path, "state-{}".format(index))), str(workspace),
            ),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        for index in range(6)
    ]
    try:
        lines = []
        for process in processes:
            assert process.stdout is not None
            lines.append(process.stdout.readline())
        assert lines.count(b"held\n") == 1
        assert all(
            line.startswith(b"refused:") for line in lines if line != b"held\n"
        )
    finally:
        for process in processes:
            _stop(process)


# --- 7. refusal precedes every side effect ----------------------------------


def _service_kwargs(state_root: Path, workspace: Path, registry: Path) -> dict:
    return {
        "state_root": str(state_root),
        "workspace_roots": (str(workspace),),
        "workspace_registry_root": str(registry),
    }


def test_an_idle_second_service_coexists_and_never_steals_ownership(
    tmp_path: Path,
) -> None:
    """Host-global ownership is demand scoped, not taken at construction.

    A live holder owns the tree. A second, *idle* service on a different state
    root used to be refused at construction because it eagerly joined; now it
    constructs fine, takes no ownership, and does not disturb the holder. The
    conflict is deferred to the moment that idle service actually tries to
    admit work on the contended tree -- see the service-level demand-scoping
    regressions.
    """

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    holder, line = _spawn_holder(registry, _tree(tmp_path, "state-a"), workspace)
    try:
        assert line == b"held\n"
        intruder_root = Path(os.path.realpath(tmp_path)) / "state-b"

        # An idle service never constructs a provider, so this factory proves
        # provider construction did not happen either.
        def _never(store):  # pragma: no cover - must never be called
            raise AssertionError("an idle service constructed a provider")

        service = CodingService(
            _never, **_service_kwargs(intruder_root, workspace, registry),
        )
        try:
            # Idle: it owns nothing at all.
            assert service._workspace_root_authority is None
            # The holder is still the sole, undisturbed owner.
            report = describe_workspace_root(registry, workspace)
            assert report["status"] == "live"
        finally:
            service.close()
    finally:
        _stop(holder)


def test_an_idle_service_owns_no_tree(tmp_path: Path) -> None:
    """Constructing an idle service leaves its trees unowned/unregistered."""

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    state_root = Path(os.path.realpath(tmp_path)) / "state"

    service = CodingService(
        lambda store: None, **_service_kwargs(state_root, workspace, registry),
    )
    try:
        # No durable non-terminal work, so no host-global hold at all.
        assert service._workspace_root_authority is None
        assert describe_workspace_root(registry, workspace)["status"] == (
            "unregistered"
        )
    finally:
        service.close()
    assert service._workspace_root_authority is None


# --- 8. inventory -----------------------------------------------------------


def test_a_changed_registry_is_a_different_execution_authority(
    tmp_path: Path,
) -> None:
    """Two registries cannot certify each other's work.

    Same state root, same trees, different host-global registry: each would
    broker against a registry the other cannot see. The state-root authority
    must therefore refuse to call them the same worker.
    """

    workspace = _tree(tmp_path, "workspace")
    state_root = Path(os.path.realpath(tmp_path)) / "state"
    first = CodingService(
        lambda store: None,
        **_service_kwargs(state_root, workspace, _registry(tmp_path)),
    )
    try:
        left = first._execution_authority()
    finally:
        first.close()

    second = CodingService(
        lambda store: None,
        **_service_kwargs(
            state_root, workspace, Path(os.path.realpath(tmp_path)) / "other-registry",
        ),
    )
    try:
        right = second._execution_authority()
    finally:
        second.close()

    assert left["workspace_registry"] != right["workspace_registry"]
    assert left != right
    # A digest, never a path.
    assert len(left["workspace_registry"]) == 64
    assert str(tmp_path) not in json.dumps(left)


def test_a_changed_root_set_is_a_different_execution_authority(
    tmp_path: Path,
) -> None:
    """Covering one tree is not the same worker as covering two."""

    registry = _registry(tmp_path)
    one = _tree(tmp_path, "workspace")
    two = _tree(tmp_path, "workspace-two")
    state_root = Path(os.path.realpath(tmp_path)) / "state"

    narrow = CodingService(
        lambda store: None, **_service_kwargs(state_root, one, registry),
    )
    try:
        left = narrow._execution_authority()
    finally:
        narrow.close()

    wide = CodingService(
        lambda store: None,
        state_root=str(state_root),
        workspace_roots=(str(one), str(two)),
        workspace_registry_root=str(registry),
    )
    try:
        right = wide._execution_authority()
    finally:
        wide.close()

    assert left["workspace_roots"] != right["workspace_roots"]
    assert len(left["workspace_roots"]) == 64
    assert str(tmp_path) not in json.dumps(right)


def test_an_alias_is_the_same_execution_authority(tmp_path: Path) -> None:
    """Normalisation is preserved: a link is not a different configuration."""

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    alias = Path(os.path.realpath(tmp_path)) / "alias"
    alias.symlink_to(workspace, target_is_directory=True)
    state_root = Path(os.path.realpath(tmp_path)) / "state"

    direct = CodingService(
        lambda store: None, **_service_kwargs(state_root, workspace, registry),
    )
    try:
        left = direct._execution_authority()
    finally:
        direct.close()

    linked = CodingService(
        lambda store: None, **_service_kwargs(state_root, alias, registry),
    )
    try:
        right = linked._execution_authority()
    finally:
        linked.close()

    assert left["workspace_roots"] == right["workspace_roots"]
    assert left == right


# --- adversarial registry ---------------------------------------------------


def test_a_symlinked_registry_entry_is_refused(tmp_path: Path) -> None:
    """A record that is really a link somewhere else is not our record."""

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    seed = WorkspaceRootAuthority(registry)
    seed.join(state_root=_tree(tmp_path, "state-a"), workspace_roots=(workspace,))
    seed.release()

    entry = registry / "{}.json".format(workspace_digest(workspace))
    elsewhere = Path(os.path.realpath(tmp_path)) / "planted.json"
    elsewhere.write_text(
        json.dumps({"version": WORKSPACE_AUTHORITY_VERSION, "state_root": "/x"}),
        encoding="utf-8",
    )
    entry.unlink()
    entry.symlink_to(elsewhere)

    with pytest.raises(WorkspaceAuthorityUnavailable):
        WorkspaceRootAuthority(registry).join(
            state_root=_tree(tmp_path, "state-b"), workspace_roots=(workspace,),
        )


def test_a_group_readable_lease_is_refused(tmp_path: Path) -> None:
    """A lock somebody else can replace is not a lock."""

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    seed = WorkspaceRootAuthority(registry)
    seed.join(state_root=_tree(tmp_path, "state-a"), workspace_roots=(workspace,))
    seed.release()

    (registry / "{}.lock".format(workspace_digest(workspace))).chmod(0o664)

    with pytest.raises(WorkspaceAuthorityUnavailable):
        WorkspaceRootAuthority(registry).join(
            state_root=_tree(tmp_path, "state-b"), workspace_roots=(workspace,),
        )


def test_a_malformed_entry_fails_closed(tmp_path: Path) -> None:
    """Damaged state is refused, never read as "nobody owns this"."""

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    seed = WorkspaceRootAuthority(registry)
    seed.join(state_root=_tree(tmp_path, "state-a"), workspace_roots=(workspace,))
    seed.release()

    (registry / "{}.json".format(workspace_digest(workspace))).write_text(
        "{not json", encoding="utf-8",
    )

    with pytest.raises(WorkspaceAuthorityUnavailable):
        WorkspaceRootAuthority(registry).join(
            state_root=_tree(tmp_path, "state-b"), workspace_roots=(workspace,),
        )


def test_a_probe_that_cannot_answer_never_reads_as_free(
    tmp_path: Path, monkeypatch,
) -> None:
    """Only real contention proves a holder; a broken probe fails closed."""

    import errno as _errno

    from flyto_ai.coding import workspace_authority as module

    registry = _registry(tmp_path)
    parent = _tree(tmp_path, "workspace")
    child = _tree(tmp_path, "workspace/nested")
    seed = WorkspaceRootAuthority(registry)
    seed.join(state_root=_tree(tmp_path, "child-state"), workspace_roots=(child,))
    seed.release()

    real = module.fcntl.flock

    def _broken(descriptor, operation):
        if operation & module.fcntl.LOCK_NB:
            raise OSError(_errno.EIO, "io")
        return real(descriptor, operation)

    monkeypatch.setattr(module.fcntl, "flock", _broken)
    with pytest.raises(WorkspaceAuthorityUnavailable):
        WorkspaceRootAuthority(registry).join(
            state_root=_tree(tmp_path, "state-b"), workspace_roots=(parent,),
        )


# --- operator recovery ------------------------------------------------------


def test_the_operator_report_distinguishes_every_recovery_state(
    tmp_path: Path,
) -> None:
    """One read-only command answers all four cases, with no JSON to edit."""

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    owner_root = _tree(tmp_path, "owner-state")

    assert describe_workspace_root(registry, workspace)["status"] == "unregistered"

    holder, line = _spawn_holder(registry, owner_root, workspace)
    try:
        assert line == b"held\n"
        live = describe_workspace_root(registry, workspace)
        assert live["status"] == "live"
        assert live["state_root"] == str(owner_root)
        assert live["workspace_digest"] == workspace_digest(workspace)
    finally:
        _stop(holder)

    assert describe_workspace_root(registry, workspace)["status"] == "adoptable"

    jobs = owner_root / "tenants" / "tenant-a" / "jobs"
    jobs.mkdir(parents=True)
    (jobs / "job_a.json").write_text(
        json.dumps({"state": "awaiting_codex_audit"}), encoding="utf-8",
    )
    stranded = describe_workspace_root(registry, workspace)
    assert stranded["status"] == "crashed_with_open_work"
    assert stranded["state_root"] == str(owner_root)


def test_the_supervisor_reason_for_a_workspace_conflict_is_distinct() -> None:
    """A workspace conflict is not the state-root refusal, and says so."""

    from flyto_ai.coding.mcp_supervisor import (
        WORKER_AUTHORITY_EXIT_CODE,
        WORKER_AUTHORITY_REASON,
        WORKER_WORKSPACE_EXIT_CODE,
        WORKER_WORKSPACE_REASON,
    )

    assert WORKER_WORKSPACE_EXIT_CODE != WORKER_AUTHORITY_EXIT_CODE
    assert WORKER_WORKSPACE_REASON != WORKER_AUTHORITY_REASON
    assert "code-workspace-status" in WORKER_WORKSPACE_REASON
    # The remedy for the *other* failure must not be suggested here.
    assert "code-release" not in WORKER_WORKSPACE_REASON
    assert "/" not in WORKER_WORKSPACE_REASON
    assert len(WORKER_WORKSPACE_REASON) <= 200


def test_a_broken_exclusive_lock_on_the_join_path_is_never_contention(
    tmp_path: Path, monkeypatch,
) -> None:
    """`EIO` on `LOCK_EX` must not fall through to the shared branch.

    Falling through would join a tree whose ownership was never established —
    the service would believe it owned a tree nobody had checked.
    """

    import errno as _errno

    from flyto_ai.coding import workspace_authority as module

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    real = module.fcntl.flock

    def _broken(descriptor, operation):
        if operation == (module.fcntl.LOCK_EX | module.fcntl.LOCK_NB):
            raise OSError(_errno.EIO, "io")
        return real(descriptor, operation)

    monkeypatch.setattr(module.fcntl, "flock", _broken)
    authority = WorkspaceRootAuthority(registry)
    with pytest.raises(WorkspaceAuthorityUnavailable):
        authority.join(
            state_root=_tree(tmp_path, "state-a"), workspace_roots=(workspace,),
        )
    assert authority.held_digests == []


@pytest.mark.parametrize("name", ["ENOLCK", "ENOTSUP", "EBADF"])
def test_other_non_contention_errnos_on_join_fail_closed(
    tmp_path: Path, monkeypatch, name: str,
) -> None:
    import errno as _errno

    from flyto_ai.coding import workspace_authority as module

    code = getattr(_errno, name, None)
    if code is None or code in module._CONTENDED_ERRNOS:
        pytest.skip("{} is unavailable or aliases contention".format(name))

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    real = module.fcntl.flock

    def _broken(descriptor, operation):
        if operation == (module.fcntl.LOCK_EX | module.fcntl.LOCK_NB):
            raise OSError(code, name)
        return real(descriptor, operation)

    monkeypatch.setattr(module.fcntl, "flock", _broken)
    with pytest.raises(WorkspaceAuthorityUnavailable):
        WorkspaceRootAuthority(registry).join(
            state_root=_tree(tmp_path, "state-a"), workspace_roots=(workspace,),
        )


# --- bounded coordination lock ---------------------------------------------


_COORDINATION_HOLDER = r"""
import sys
import fcntl, os
from flyto_ai.coding.workspace_authority import (
    REGISTRY_LOCK_NAME, WorkspaceRootAuthority,
)

authority = WorkspaceRootAuthority(sys.argv[1])
directory = authority._registry_directory()
descriptor = os.open(str(directory / REGISTRY_LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(descriptor, fcntl.LOCK_EX)
sys.stdout.write("held\n")
sys.stdout.flush()
sys.stdin.readline()
"""


def _spawn_coordination_holder(registry: Path):
    script = Path(os.path.realpath(registry.parent)) / "coordination.py"
    script.write_text(_COORDINATION_HOLDER, encoding="utf-8")
    process = subprocess.Popen(
        (sys.executable, "-u", str(script), str(registry)),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert process.stdout is not None
    return process, process.stdout.readline()


def test_a_wedged_coordination_holder_cannot_hang_a_join(tmp_path: Path) -> None:
    """One stuck process must not become a host-wide outage."""

    from flyto_ai.coding.workspace_authority import (
        REGISTRY_LOCK_TIMEOUT_SECONDS,
        WorkspaceAuthorityBusy,
    )

    registry = _registry(tmp_path)
    holder, line = _spawn_coordination_holder(registry)
    try:
        assert line == b"held\n"
        started = time.monotonic()
        with pytest.raises(WorkspaceAuthorityBusy) as excinfo:
            WorkspaceRootAuthority(registry).join(
                state_root=_tree(tmp_path, "state-a"),
                workspace_roots=(_tree(tmp_path, "workspace"),),
            )
        elapsed = time.monotonic() - started
        assert excinfo.value.code == "workspace_authority_busy"
        assert elapsed < REGISTRY_LOCK_TIMEOUT_SECONDS * 3

        # The read-only report is bounded by the same deadline.
        started = time.monotonic()
        with pytest.raises(WorkspaceAuthorityBusy):
            describe_workspace_root(registry, _tree(tmp_path, "workspace"))
        assert time.monotonic() - started < REGISTRY_LOCK_TIMEOUT_SECONDS * 3
    finally:
        _stop(holder)


def test_a_crashed_coordination_holder_releases_the_lock(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    holder, line = _spawn_coordination_holder(registry)
    assert line == b"held\n"
    holder.kill()
    holder.wait(timeout=30)
    for stream in (holder.stdin, holder.stdout):
        if stream is not None:
            stream.close()

    authority = WorkspaceRootAuthority(registry)
    authority.join(
        state_root=_tree(tmp_path, "state-a"),
        workspace_roots=(_tree(tmp_path, "workspace"),),
    )
    authority.release()


# --- durable write ----------------------------------------------------------


def test_a_failed_directory_fsync_never_reports_success(
    tmp_path: Path, monkeypatch,
) -> None:
    """An un-durable authority record must not be reported as written."""

    import errno as _errno

    from flyto_ai.coding import workspace_authority as module

    registry = _registry(tmp_path)
    real = os.fsync

    def _broken(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(_errno.EIO, "io")
        return real(descriptor)

    monkeypatch.setattr(module.os, "fsync", _broken)
    authority = WorkspaceRootAuthority(registry)
    with pytest.raises(WorkspaceAuthorityUnavailable):
        authority.join(
            state_root=_tree(tmp_path, "state-a"),
            workspace_roots=(_tree(tmp_path, "workspace"),),
        )
    assert authority.held_digests == []


def test_a_failed_replace_is_bounded_and_leaves_no_temp_file(
    tmp_path: Path, monkeypatch,
) -> None:
    import errno as _errno

    from flyto_ai.coding import workspace_authority as module

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")

    def _broken(source, target):
        raise OSError(_errno.EIO, "io")

    monkeypatch.setattr(module.os, "replace", _broken)
    with pytest.raises(WorkspaceAuthorityUnavailable):
        WorkspaceRootAuthority(registry).join(
            state_root=_tree(tmp_path, "state-a"), workspace_roots=(workspace,),
        )
    assert not list(registry.glob(".entry-*"))


# --- overlap-aware operator report -----------------------------------------


def test_the_report_finds_an_ancestor_owner(tmp_path: Path) -> None:
    """Asking about a child must name the parent that would refuse it."""

    registry = _registry(tmp_path)
    parent = _tree(tmp_path, "workspace")
    child = _tree(tmp_path, "workspace/nested")
    owner_root = _tree(tmp_path, "owner-state")

    holder, line = _spawn_holder(registry, owner_root, parent)
    try:
        assert line == b"held\n"
        report = describe_workspace_root(registry, child)
        assert report["status"] == "live"
        assert report["state_root"] == str(owner_root)
        assert [item["relationship"] for item in report["owners"]] == [
            "owner_is_ancestor",
        ]
        assert report["owners"][0]["workspace_digest"] == workspace_digest(parent)
    finally:
        _stop(holder)


def test_the_report_finds_a_descendant_owner(tmp_path: Path) -> None:
    """And the reverse direction, which is the one that used to read empty."""

    registry = _registry(tmp_path)
    parent = _tree(tmp_path, "workspace")
    child = _tree(tmp_path, "workspace/nested")
    owner_root = _tree(tmp_path, "owner-state")

    holder, line = _spawn_holder(registry, owner_root, child)
    try:
        assert line == b"held\n"
        report = describe_workspace_root(registry, parent)
        assert report["status"] == "live"
        assert report["state_root"] == str(owner_root)
        assert [item["relationship"] for item in report["owners"]] == [
            "owner_is_descendant",
        ]
    finally:
        _stop(holder)


def test_the_report_lists_multiple_overlaps_deterministically(
    tmp_path: Path,
) -> None:
    """Exact first, then ancestor, then descendant, and stable across runs."""

    registry = _registry(tmp_path)
    parent = _tree(tmp_path, "workspace")
    child = _tree(tmp_path, "workspace/nested")
    grandchild = _tree(tmp_path, "workspace/nested/deeper")

    for index, root in enumerate((parent, child, grandchild)):
        seed = WorkspaceRootAuthority(registry)
        seed.join(
            state_root=_tree(tmp_path, "state-{}".format(index)),
            workspace_roots=(root,),
        )
        seed.release()

    report = describe_workspace_root(registry, child)
    assert [item["relationship"] for item in report["owners"]] == [
        "exact", "owner_is_ancestor", "owner_is_descendant",
    ]
    assert report["status"] == "adoptable"
    assert describe_workspace_root(registry, child) == report


def test_the_report_headline_is_the_most_blocking_overlap(tmp_path: Path) -> None:
    """An adoptable exact entry must not mask a live parent."""

    registry = _registry(tmp_path)
    parent = _tree(tmp_path, "workspace")
    child = _tree(tmp_path, "workspace/nested")

    seed = WorkspaceRootAuthority(registry)
    seed.join(state_root=_tree(tmp_path, "child-state"), workspace_roots=(child,))
    seed.release()

    holder, line = _spawn_holder(registry, _tree(tmp_path, "parent-state"), parent)
    try:
        assert line == b"held\n"
        report = describe_workspace_root(registry, child)
        assert report["status"] == "live"
        assert {item["relationship"] for item in report["owners"]} == {
            "exact", "owner_is_ancestor",
        }
    finally:
        _stop(holder)


def test_the_report_fails_closed_on_a_malformed_entry(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    seed = WorkspaceRootAuthority(registry)
    seed.join(state_root=_tree(tmp_path, "state-a"), workspace_roots=(workspace,))
    seed.release()

    (registry / "{}.json".format(workspace_digest(workspace))).write_text(
        "{not json", encoding="utf-8",
    )
    with pytest.raises(WorkspaceAuthorityUnavailable):
        describe_workspace_root(registry, workspace)


def test_the_report_takes_no_lasting_lease(tmp_path: Path) -> None:
    """Reporting must never become a way to own a tree."""

    registry = _registry(tmp_path)
    workspace = _tree(tmp_path, "workspace")
    seed = WorkspaceRootAuthority(registry)
    seed.join(state_root=_tree(tmp_path, "state-a"), workspace_roots=(workspace,))
    seed.release()

    assert describe_workspace_root(registry, workspace)["status"] == "adoptable"
    # A real service still starts immediately afterwards.
    later = WorkspaceRootAuthority(registry)
    later.join(state_root=_tree(tmp_path, "state-b"), workspace_roots=(workspace,))
    try:
        assert later.held_digests == [workspace_digest(workspace)]
    finally:
        later.release()


def _classify(monkeypatch, error) -> int:
    """Run `_cmd_code_mcp`'s classification and return its exit status."""

    from flyto_ai import cli

    def _explode(args):
        raise error

    monkeypatch.setattr(cli, "_build_coding_service", _explode)
    with pytest.raises(SystemExit) as excinfo:
        cli._cmd_code_mcp(argparse.Namespace(tenant="t"))
    return excinfo.value.code


def test_the_three_workspace_refusals_get_three_exit_statuses(
    monkeypatch, capsys,
) -> None:
    """Conflict, busy, and registry fault are not the same condition."""

    from flyto_ai.coding.mcp_supervisor import (
        WORKER_AUTHORITY_EXIT_CODE,
        WORKER_WORKSPACE_BUSY_EXIT_CODE,
        WORKER_WORKSPACE_EXIT_CODE,
        WORKER_WORKSPACE_REGISTRY_EXIT_CODE,
    )
    from flyto_ai.coding.service import (
        CodingWorkspaceAuthorityBusy,
        CodingWorkspaceAuthorityConflict,
        CodingWorkspaceAuthorityUnavailable,
    )

    cases = (
        (CodingWorkspaceAuthorityConflict("x"), WORKER_WORKSPACE_EXIT_CODE),
        (CodingWorkspaceAuthorityBusy("x"), WORKER_WORKSPACE_BUSY_EXIT_CODE),
        (
            CodingWorkspaceAuthorityUnavailable("x"),
            WORKER_WORKSPACE_REGISTRY_EXIT_CODE,
        ),
    )
    statuses = set()
    for error, expected in cases:
        assert _classify(monkeypatch, error) == expected
        statuses.add(expected)
        # stderr carries the stable code only, never the message.
        assert error.code in capsys.readouterr().err

    assert len(statuses) == 3
    assert WORKER_AUTHORITY_EXIT_CODE not in statuses


def test_only_the_busy_workspace_refusal_is_retryable() -> None:
    from flyto_ai.coding.service import (
        CodingWorkspaceAuthorityBusy,
        CodingWorkspaceAuthorityConflict,
        CodingWorkspaceAuthorityUnavailable,
    )

    assert CodingWorkspaceAuthorityBusy.retryable is True
    assert CodingWorkspaceAuthorityConflict.retryable is False
    assert CodingWorkspaceAuthorityUnavailable.retryable is False
    assert len({
        CodingWorkspaceAuthorityBusy.code,
        CodingWorkspaceAuthorityConflict.code,
        CodingWorkspaceAuthorityUnavailable.code,
    }) == 3


@pytest.mark.parametrize("status_name", [
    "WORKER_WORKSPACE_EXIT_CODE",
    "WORKER_WORKSPACE_BUSY_EXIT_CODE",
    "WORKER_WORKSPACE_REGISTRY_EXIT_CODE",
])
def test_the_supervisor_reports_each_refusal_end_to_end(
    tmp_path: Path, status_name: str,
) -> None:
    """A real worker exiting with each status yields its own fixed reason."""

    from flyto_ai.coding import mcp_supervisor as module
    from flyto_ai.coding.mcp_supervisor import CodingMCPWorkerSupervisor

    status = getattr(module, status_name)
    script = Path(os.path.realpath(tmp_path)) / "refuse.py"
    script.write_text("import sys\nsys.exit({})\n".format(status), encoding="utf-8")
    supervisor = CodingMCPWorkerSupervisor(
        (sys.executable, "-u", str(script)), build_id_provider=lambda: "build",
    )
    try:
        raw = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        }, separators=(",", ":")).encode() + b"\n"
        value = json.loads(supervisor.handle_line(raw))
    finally:
        supervisor.close()

    expected = {
        "WORKER_WORKSPACE_EXIT_CODE": module.WORKER_WORKSPACE_REASON,
        "WORKER_WORKSPACE_BUSY_EXIT_CODE": module.WORKER_WORKSPACE_BUSY_REASON,
        "WORKER_WORKSPACE_REGISTRY_EXIT_CODE": module.WORKER_WORKSPACE_REGISTRY_REASON,
    }[status_name]
    message = value["error"]["message"]
    assert value["error"]["code"] == -32603
    assert message == expected
    assert "\n" not in message and len(message) <= 200
    assert str(tmp_path) not in message


def test_the_busy_and_registry_reasons_give_no_false_ownership_advice() -> None:
    """Neither may send an operator after an owner that does not exist."""

    from flyto_ai.coding import mcp_supervisor as module

    for reason in (
        module.WORKER_WORKSPACE_BUSY_REASON,
        module.WORKER_WORKSPACE_REGISTRY_REASON,
    ):
        assert "owns" not in reason
        assert "code-release" not in reason
        assert "/" not in reason.replace("code-workspace-status", "")
    assert "retry" in module.WORKER_WORKSPACE_BUSY_REASON
    assert "code-workspace-status" in module.WORKER_WORKSPACE_REGISTRY_REASON
    assert len({
        module.WORKER_WORKSPACE_REASON,
        module.WORKER_WORKSPACE_BUSY_REASON,
        module.WORKER_WORKSPACE_REGISTRY_REASON,
    }) == 3


def test_the_status_command_prints_every_overlapping_owner(
    tmp_path: Path, capsys,
) -> None:
    """One command must show the whole incident, not just its worst part."""

    from flyto_ai import cli

    registry = _registry(tmp_path)
    parent = _tree(tmp_path, "workspace")
    child = _tree(tmp_path, "workspace/nested")
    grandchild = _tree(tmp_path, "workspace/nested/deeper")
    for index, root in enumerate((parent, child, grandchild)):
        seed = WorkspaceRootAuthority(registry)
        seed.join(
            state_root=_tree(tmp_path, "state-{}".format(index)),
            workspace_roots=(root,),
        )
        seed.release()

    cli._cmd_code_workspace_status(argparse.Namespace(
        workspace=str(child), registry_root=str(registry), json=False,
    ))
    out = capsys.readouterr().out

    assert "overlapping owners (3)" in out
    # Deterministic order, and each relationship actually shown.
    assert out.index("exact") < out.index("owner_is_ancestor")
    assert out.index("owner_is_ancestor") < out.index("owner_is_descendant")
    for index in range(3):
        assert str(_tree(tmp_path, "state-{}".format(index))) in out


def test_the_public_mcp_inventory_is_still_exactly_three() -> None:
    from flyto_ai.coding.mcp_server import CodingMCPServer

    assert tuple(tool["name"] for tool in CodingMCPServer._tools()) == (
        "flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit",
    )


def test_the_registry_location_is_neutral_and_outside_worktrees(
    tmp_path, monkeypatch,
) -> None:
    """No product or repository name appears in the protocol's default path.

    The suite-wide override is removed here on purpose: the property under
    test is the *default* the protocol ships with, not the private location a
    test run happens to be pointed at. `HOME` is pinned to a neutral directory
    for the same reason -- the property belongs to the suffix the protocol
    chooses, so a developer whose home path happens to spell one of these
    tokens must neither fail this test nor pass it for the wrong reason.
    """

    from flyto_ai.coding import workspace_authority

    monkeypatch.delenv(workspace_authority.WORKSPACE_REGISTRY_ENV, raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    home = tmp_path / "neutral-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    root = workspace_authority.default_registry_root()
    assert root.is_relative_to(home), "the default must stay under the user's home"
    suffix = root.relative_to(home).as_posix().lower()
    for token in ("flyto", "flytohub", "claude", "codex"):
        assert token not in suffix
