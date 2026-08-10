# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Durable authority to continue one bounded provider stop, and nothing else.

A round that stops at a configured ceiling is real work: a session existed, the
model edited the tree, and the only thing missing is more budget.  Carrying that
forward safely is not "remember the session id" -- an unbound resume is a
request to re-enter someone's conversation against whatever the tree happens to
contain now.  So continuation is an explicit, single-use, tenant-local authority.
Holding the id proves nothing.

Three independent audit probes broke the first version of this file, and each
one is a design property here rather than a patch:

*A digest of the files the model touched is not a digest of the workspace.*
The first version re-hashed only the attributable change set, so an unrelated
new file added between segments was invisible: the model would resume believing
it knew the tree.  `workspace_manifest_digest` observes every entry - added,
deleted, re-typed, re-moded - and it is what admission re-proves.

*A record that vouches for itself cannot detect its own replay.*  A digest
stored inside the record it protects is recomputable by whoever rewrote the
record, so restoring an older-but-valid authority body, together with the bytes
it described, was accepted.  The monotonic truth therefore lives outside the
replaceable file, in an append-only hash-chained journal whose tail is the only
thing allowed to say what generation and state this session is at.

*A schema that ignores unknown keys is not a schema.*  Extra fields survived
into a loaded authority because the digest covered only the fields the parser
happened to read.  Both records are now exact: unknown keys, missing keys,
bool-as-number, non-finite numbers and non-canonical shapes are all refused.

Everything here is provider-neutral and repository-neutral.  It knows about
sessions, tenants, directories and bytes; it knows nothing about Claude, Git, or
any particular project.
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import stat
import time
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from flyto_ai.coding.route import ROUTE_THREAD_PREFIX

#: Closed schema tokens. A file that does not name exactly these is not read.
CONTINUATION_AUTHORITY_VERSION = "flyto.coding-continuation-authority.v3"
#: The schema that existed before an authority bound the contract it was granted
#: under. A v2 record is not upgradable: nothing in it says which contract the
#: stopped round executed, and inventing one would be exactly the silent
#: re-authorization this version exists to prevent. It is recognized only so the
#: refusal can be precise, and it is never read for anything else.
LEGACY_UNPINNED_AUTHORITY_VERSION = "flyto.coding-continuation-authority.v2"
CONTINUATION_JOURNAL_VERSION = "flyto.coding-continuation-journal.v1"
_AUTHORITY_DOMAIN = b"flyto.coding-continuation-authority.v3\n"
_JOURNAL_DOMAIN = b"flyto.coding-continuation-journal.v1\n"
_MANIFEST_DOMAIN = b"flyto.coding-workspace-manifest.v1\n"

#: Session id shapes a host mints for itself when no provider session exists.
#: Neither is a backend session, so neither may ever be continued. Shared with
#: `flyto_ai.coding.service` so there is one definition of "not a real session".
PROVISIONAL_SESSION_PREFIXES: Tuple[str, ...] = (ROUTE_THREAD_PREFIX, "host-")

_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID_RE = re.compile(r"^job_[a-f0-9]{24}$")
_BACKEND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{1,63}$")

STATE_OPEN = "open"
STATE_CLAIMED = "claimed"
STATE_SETTLED = "settled"
_STATES = frozenset({STATE_OPEN, STATE_CLAIMED, STATE_SETTLED})

#: The only state changes this module will ever record, and - just as
#: load-bearing - exactly what each one may do to the generation.
#:
#:      (from)          (to)                  generation
#:      none        ->  open                  = 1
#:      open(g)     ->  claimed(g)            unchanged
#:      open(g)     ->  settled(g)            unchanged
#:      claimed(g)  ->  open(g+1)             exactly one more  (a rotation)
#:      claimed(g)  ->  settled(g)            unchanged
#:      settled     ->  nothing
#:
#: An audit probe committed a self-consistent `open(1) -> claimed(2)` because
#: the old rules asked only for a legal state pair and a non-decreasing
#: generation. Claiming is not a rotation: letting it advance the generation
#: lets one claim consume a segment and mint the next one in a single step.
_LEGAL_TRANSITIONS = {
    None: frozenset({STATE_OPEN}),
    STATE_OPEN: frozenset({STATE_CLAIMED, STATE_SETTLED}),
    STATE_CLAIMED: frozenset({STATE_OPEN, STATE_SETTLED}),
    STATE_SETTLED: frozenset(),
}
#: Exactly what the generation must be after each transition, as a delta.
_TRANSITION_GENERATION_DELTA = {
    (STATE_OPEN, STATE_CLAIMED): 0,
    (STATE_OPEN, STATE_SETTLED): 0,
    (STATE_CLAIMED, STATE_SETTLED): 0,
    # Returning to `open` is only ever a rotation, and a rotation is one
    # generation. Anything else would be a reopen of a spent segment.
    (STATE_CLAIMED, STATE_OPEN): 1,
}
#: Fields no transition may ever touch. These are the identity of the
#: continuation: who it belongs to, which conversation it is, which tree and
#: which contract it was granted against, and when it began. A transition that
#: could change any of them would not be a transition, it would be a new grant
#: wearing an old journal.
_TRANSITION_INVARIANT_FIELDS: Tuple[str, ...] = (
    "tenant_ref", "backend", "session_id", "origin_job_id", "working_dir",
    "workspace_sha256", "authorized_config_sha256",
    # The pinned contract is identity too. A rotation that could swap it would
    # carry the session forward under a verifier the origin never authorized.
    "contract_snapshot_sha256", "request_sha256",
    # The projection is identity, not payload. A rotation that changed it would
    # be continuing a different observation of the same directory.
    "snapshot_policy_sha256", "created_at",
)

#: How many times one session may be carried forward before a human decides.
#: A continuation never starts itself, so this is not a spend loop; it is the
#: point past which "just one more segment" stops being an engineering answer.
MAX_CONTINUATION_GENERATION = 32
#: The attributable set an authority may bind, matching the revision bound.
MAX_CONTINUATION_FILES = 512
#: Journal length. Each generation costs at most an open, a claim and a
#: settle/rotate, so this bounds the file without bounding useful work.
MAX_CONTINUATION_TRANSITIONS = MAX_CONTINUATION_GENERATION * 4 + 8
#: A recorded time must be a real one. Before 2001 or after 2200 is a corrupted
#: or forged record, not a clock skew.
_MIN_TIMESTAMP = 978_307_200.0
_MAX_TIMESTAMP = 7_258_118_400.0

# ── workspace manifest bounds ────────────────────────────────────────
#: Finite, but sized from the workspaces this control plane actually serves
#: rather than from a fixture. Measured on the three live products:
#:
#:   flyto-ai       18,138 files    2,010 dirs    3 links    0.82 GB
#:   flyto-engine    3,010 files      280 dirs    0 links    0.40 GB
#:   flyto-code    114,035 files    9,707 dirs   44 links    1.14 GB
#:
#: The largest single file observed was 279 MB and the deepest tree was 11
#: levels. The bounds below leave roughly 3x headroom on every axis, because a
#: bound that a real repository can reach is a bound that turns into "this
#: product can never be continued".
#:
#: Exceeding one is still refused rather than truncated: a partial manifest
#: would silently stop observing exactly the paths an attacker would choose.
MAX_MANIFEST_ENTRIES = 400_000
MAX_MANIFEST_DEPTH = 128
MAX_MANIFEST_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_MANIFEST_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
#: The largest an authority body or a whole journal may be. Both are small,
#: bounded records; anything approaching this is not one of ours.
MAX_STATE_FILE_BYTES = 4 * 1024 * 1024
#: A symlink target is recorded, not followed, so it only has to be bounded.
MAX_MANIFEST_LINK_BYTES = 4096
_MANIFEST_CHUNK_BYTES = 4 * 1024 * 1024
#: The only names excluded, the only place they are excluded, and the entire
#: justification: these are version-control machine state at the *root* of the
#: workspace, rewritten by tooling rather than by a model edit, so including
#: them would make an unrelated `git gc` look like workspace drift.
#:
#: Nothing else is excluded. Not `.venv`, not `node_modules`, not
#: `.flyto-index`, not build output, not caches, not dot-directories. Those are
#: precisely the trees where a second agent could change what the model reads
#: without touching anything a narrower snapshot would notice, and an excluded
#: directory is a blind spot by definition. They are large, which is a cost
#: paid in the bounds above rather than in coverage.
#:
#: The match is root-relative on purpose: a nested directory that merely
#: happens to be named `.git` - a fixture, a template, a vendored sample - is
#: ordinary workspace content and is observed like anything else.
MANIFEST_EXCLUDED_DIRECTORIES: Tuple[str, ...] = (".git", ".hg", ".svn")

#: The one outward answer for every condition that would otherwise confirm an
#: authority exists: absent, another tenant's, already consumed, settled,
#: superseded, corrupt, truncated, replayed, or lost to a concurrent claimer.
CONTINUATION_UNAVAILABLE = "continuation_unavailable"
#: The request itself is not continuable. Refused before any lookup.
CONTINUATION_SESSION_INVALID = "continuation_session_invalid"
#: Reached only by a caller that has already proven it owns the authority.
CONTINUATION_BACKEND_MISMATCH = "continuation_backend_mismatch"
CONTINUATION_WORKSPACE_MISMATCH = "continuation_workspace_mismatch"
#: The pinned contract this authority binds could be read but is not the one it
#: was granted under. Reserved for exactly that: a *changed contract file* is not
#: this code, because the file lives inside the observed workspace and therefore
#: lands on `CONTINUATION_REVISION_MISMATCH` below, before the pin is consulted
#: at all. Keeping the two apart is what lets an operator tell "your tree moved"
#: from "your recorded verifier no longer reproduces its own address".
CONTINUATION_CONTRACT_CHANGED = "continuation_contract_changed"
CONTINUATION_REVISION_MISMATCH = "continuation_revision_mismatch"
#: The projection this host would snapshot under is not the one the
#: authority was granted under. Reached only after ownership is proven.
CONTINUATION_POLICY_CHANGED = "continuation_snapshot_policy_changed"
#: The stored authority predates contract pinning, so nothing records which
#: verifier the stopped round actually ran. Distinct from every other refusal on
#: purpose: the operator has nothing to restore and nothing to fix, and no
#: retry can help. The only honest route forward is a fresh job, which re-reads
#: and re-pins the contract from scratch. Never auto-upgraded - inferring a
#: snapshot for a segment nobody measured would be the silent re-authorization
#: this whole mechanism exists to prevent.
CONTINUATION_CONTRACT_UNPINNED = "continuation_contract_unpinned"
#: The one action that resolves it, from the closed preflight vocabulary shape.
CONTINUATION_UNPINNED_ACTION = "submit_a_new_coding_job"

CONTINUATION_CODES = frozenset({
    CONTINUATION_CONTRACT_UNPINNED,
    CONTINUATION_UNAVAILABLE,
    CONTINUATION_SESSION_INVALID,
    CONTINUATION_BACKEND_MISMATCH,
    CONTINUATION_WORKSPACE_MISMATCH,
    CONTINUATION_CONTRACT_CHANGED,
    CONTINUATION_REVISION_MISMATCH,
    CONTINUATION_POLICY_CHANGED,
})

#: The bounded provider stops a host is willing to carry forward, and a closed
#: set. Both mean "this round consumed a ceiling the host itself configured".
CONTINUABLE_STOP_CODES = frozenset({
    "provider_job_budget_exhausted", "turn_limit_exceeded",
})


class ContinuationCorrupt(ValueError):
    """A stored record cannot be read as exactly what it claims to be."""


class ContinuationConflict(RuntimeError):
    """The durable high-water mark moved under a transition being committed."""


class WorkspaceUnobservable(ValueError):
    """The workspace cannot be described exactly, so it cannot be bound."""


def is_continuable_session(value: Any) -> bool:
    """Whether an id could name a real backend session this host may continue."""

    return (
        not isinstance(value, bool)
        and isinstance(value, str)
        and bool(_SESSION_RE.fullmatch(value))
        and not value.startswith(PROVISIONAL_SESSION_PREFIXES)
    )


def session_ref(session_id: str) -> str:
    """Key an authority by its session without writing the session in a name."""

    if not is_continuable_session(session_id):
        raise ValueError("continuation session id is not continuable")
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# the snapshot policy
# ──────────────────────────────────────────────────────────────────────

#: Closed schema token for the projection an authority was granted under.
SNAPSHOT_POLICY_VERSION = "flyto.coding-snapshot-policy.v1"
_POLICY_DOMAIN = b"flyto.coding-snapshot-policy.v1\n"
#: At most this many names may be classified as control-plane runtime state.
#: A policy is a list of exceptions; a long list is not a policy.
MAX_RUNTIME_STATE_NAMES = 4
#: A classified name is a single root-relative directory name, nothing else.
#: No globs, no separators, no traversal, no dot entries.
_RUNTIME_STATE_NAME_RE = re.compile(r"^[A-Za-z0-9.][A-Za-z0-9_.-]{0,63}$")
#: `.` and `..` match the pattern above and are not names.
_RUNTIME_STATE_RESERVED = frozenset({".", ".."})


class SnapshotPolicyInvalid(ValueError):
    """A projection this host will not snapshot under."""


@dataclass(frozen=True)
class SnapshotPolicy:
    """Which parts of a workspace are source, and which are somebody's runtime.

    A workspace snapshot exists to answer "is this the tree the model stopped
    in". That question has one honest complication: a control plane may keep
    its own live database *inside* the tree it serves. `flyto-code` carries a
    `.flyto-index/task-runs.sqlite` that its Indexer rewrites continuously, so a
    whole-tree digest of that repository never repeats and continuation is
    refused forever - not because the source moved, but because a different
    component was doing its job.

    The fix is not a blanket ignore list. `node_modules`, `.venv`, build output
    and caches stay fully observed, because those are exactly where a second
    agent could change what the model reads. What may be classified is a
    *named, root-relative* directory that a host-owned lane independently
    revalidates, and the classification is itself digest-bound into the
    authority: a policy that changes, grows, or is used on a route without the
    lane that justifies it refuses continuation before provider contact.

    The default is the empty policy - everything non-VCS is source - so a caller
    that says nothing gets the strictest behaviour.
    """

    #: Exact root-relative directory names owned by the control plane rather
    #: than by the model. Stored sorted and unique so the identity is canonical.
    runtime_state_names: Tuple[str, ...] = ()
    #: Free-form, bounded justification recorded in the identity. Two policies
    #: that exclude the same name for different stated reasons are different
    #: policies, so a silent re-purposing is a policy change.
    rationale: str = ""

    def __post_init__(self) -> None:
        names = tuple(self.runtime_state_names)
        if len(names) > MAX_RUNTIME_STATE_NAMES:
            raise SnapshotPolicyInvalid("too many runtime-state names")
        for name in names:
            if (
                not isinstance(name, str)
                or name in _RUNTIME_STATE_RESERVED
                or not _RUNTIME_STATE_NAME_RE.fullmatch(name)
            ):
                raise SnapshotPolicyInvalid("a runtime-state name is not a plain name")
            if name in MANIFEST_EXCLUDED_DIRECTORIES:
                raise SnapshotPolicyInvalid(
                    "version-control state is already excluded and is not a policy choice",
                )
        if list(names) != sorted(set(names)):
            raise SnapshotPolicyInvalid("runtime-state names are not canonical")
        if len(self.rationale) > 200:
            raise SnapshotPolicyInvalid("a policy rationale is not bounded")

    def identity(self) -> str:
        """The digest an authority binds. Any change here is a policy change."""

        payload = json.dumps(
            {
                "policy_version": SNAPSHOT_POLICY_VERSION,
                "runtime_state_names": list(self.runtime_state_names),
                "rationale": self.rationale,
            },
            ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
        digest = hashlib.sha256()
        digest.update(_POLICY_DOMAIN)
        digest.update(payload.encode("utf-8"))
        return digest.hexdigest()

    def classifies(self, name: str, *, at_root: bool) -> bool:
        """Only an exact root-relative name. A nested namesake stays source."""

        return at_root and name in self.runtime_state_names


#: Everything that is not version control is source. The default, and what a
#: generic directory, a non-strict route, or a caller that says nothing gets.
DEFAULT_SNAPSHOT_POLICY = SnapshotPolicy()


# ──────────────────────────────────────────────────────────────────────
# the workspace manifest
# ──────────────────────────────────────────────────────────────────────


def workspace_manifest_digest(
    working_dir: str, policy: Optional[SnapshotPolicy] = None,
) -> str:
    """Digest every observable entry of one workspace, deterministically.

    This answers "is the tree the model stopped in still the tree in front of
    me", and it is deliberately a different question from the
    attributable-revision digest. That one binds what a round changed, which is
    what an auditor signs. This one binds what is *there*, which is what a
    resumed model will be looking at. A file nobody attributed to the round is
    still a file the model did not write and does not know about.

    Two properties that a pathname-based walk cannot provide, and that an audit
    probe broke the previous version on:

    *The root is never resolved.* `Path.resolve()` follows a symlinked
    workspace root, so a caller pointed at a link would snapshot whatever the
    link happened to reference. The absolute path is instead opened one
    component at a time with ``O_DIRECTORY | O_NOFOLLOW``, so a link anywhere
    in the configured path is a refusal rather than a redirection.

    *No directory is ever re-opened by name.* Every descent, stat, read and
    readlink is relative to a directory descriptor this walk already holds and
    already checked. A directory swapped for a symlink between the check and
    the open has nothing to swap: there is no second lookup.

    Symlinks inside the tree are recorded as link objects - path plus target,
    never followed. A real `.venv` or `node_modules` is full of them, and
    refusing them outright is what made the previous version unable to snapshot
    any real repository. Their targets are part of what the model sees, so a
    retargeted link moves the digest.

    Everything the walk cannot describe exactly still raises: a device node, a
    socket, an unreadable entry, a path that changes type or bytes while it is
    read, a directory mutated during its own enumeration, a tree past any
    bound. A skipped entry is a hole exactly where an attacker would put one.
    """

    if not working_dir or not working_dir.startswith("/"):
        raise WorkspaceUnobservable("the workspace path is not absolute")
    policy = DEFAULT_SNAPSHOT_POLICY if policy is None else policy
    if not isinstance(policy, SnapshotPolicy):
        raise WorkspaceUnobservable("the snapshot policy is not a policy")
    root_fd = _open_path_without_following(working_dir)
    try:
        pinned = os.fstat(root_fd)
        digest = hashlib.sha256()
        digest.update(_MANIFEST_DOMAIN)
        # The projection is part of what is being digested. Two snapshots taken
        # under different policies are different observations of the tree and
        # must never compare equal, whatever the bytes happened to be.
        digest.update("\0policy\0{}\n".format(policy.identity()).encode("utf-8"))
        counters = _ManifestCounters()
        _digest_directory(
            root_fd, "", digest, counters, depth=0, at_root=True, policy=policy,
        )
        # The configured path must still lead to the directory that was
        # actually walked. Without this a root swapped after the descriptor was
        # taken would produce a digest for a tree the caller never named.
        _assert_same_directory(working_dir, pinned)
        digest.update("\0entries\0{}\n".format(counters.entries).encode("utf-8"))
        return digest.hexdigest()
    finally:
        os.close(root_fd)


class _ManifestCounters:
    """Bounds carried through the walk, checked as they are consumed."""

    __slots__ = ("entries", "total")

    def __init__(self) -> None:
        self.entries = 0
        self.total = 0

    def count(self, size: int = 0) -> None:
        self.entries += 1
        self.total += size
        if self.entries > MAX_MANIFEST_ENTRIES:
            raise WorkspaceUnobservable("the workspace exceeds the manifest entry bound")
        if self.total > MAX_MANIFEST_TOTAL_BYTES:
            raise WorkspaceUnobservable("the workspace exceeds the manifest size bound")


def _open_path_without_following(path: str) -> int:
    """Open an absolute directory path one component at a time, never following.

    The whole ancestry is checked, not just the final component: a link three
    levels up redirects a snapshot just as effectively as a link on the leaf.
    """

    handle = os.open("/", os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC)
    try:
        for part in PurePosixPath(path).parts[1:]:
            if part in ("", ".", ".."):
                raise WorkspaceUnobservable("the workspace path is not canonical")
            nested = os.open(
                part, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
                dir_fd=handle,
            )
            os.close(handle)
            handle = nested
        return handle
    except OSError as exc:
        os.close(handle)
        if exc.errno in (errno.ELOOP, errno.EMLINK, errno.ENOTDIR):
            # `O_DIRECTORY | O_NOFOLLOW` on a symlink reports `ELOOP` on Linux
            # and `ENOTDIR` on macOS, and a plain file reports `ENOTDIR` on
            # both. All three mean the same thing to a caller: this path does
            # not lead to a directory this host is willing to walk.
            raise WorkspaceUnobservable(
                "the workspace path is not a directory or traverses a symlink",
            ) from exc
        raise WorkspaceUnobservable("the workspace is unavailable") from exc
    except BaseException:
        os.close(handle)
        raise


def _assert_same_directory(path: str, pinned: os.stat_result) -> None:
    handle = _open_path_without_following(path)
    try:
        now = os.fstat(handle)
    finally:
        os.close(handle)
    if (now.st_dev, now.st_ino) != (pinned.st_dev, pinned.st_ino):
        raise WorkspaceUnobservable("the workspace root changed while it was read")


def _digest_directory(
    dir_fd: int,
    prefix: str,
    digest: "hashlib._Hash",
    counters: _ManifestCounters,
    *,
    depth: int,
    policy: SnapshotPolicy,
    at_root: bool = False,
) -> None:
    """Feed one directory, in sorted order, descending through descriptors only."""

    if depth > MAX_MANIFEST_DEPTH:
        raise WorkspaceUnobservable("the workspace exceeds the manifest depth bound")
    before = os.fstat(dir_fd)
    try:
        with os.scandir(dir_fd) as scan:
            # Each entry is stat-ed *now*, inside the enumeration window, not
            # lazily when the loop below reaches it. `DirEntry.stat()` is lazy
            # on macOS, so leaving it to the loop put the check for a late
            # sibling arbitrarily far from its open - and a directory renamed
            # into that name in between would be stat-ed after the swap and
            # match itself. Materialising here is what makes the identity
            # comparison at open time mean anything.
            children = sorted(
                ((item.name, item.stat(follow_symlinks=False)) for item in scan),
                key=lambda pair: pair[0],
            )
    except OSError as exc:
        raise WorkspaceUnobservable("a workspace directory cannot be read") from exc
    after = os.fstat(dir_fd)
    if (
        (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_nlink)
        != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_nlink)
    ):
        # An entry was inserted, removed or renamed while this directory was
        # being listed. The resulting digest would describe a tree that never
        # existed at any single instant.
        raise WorkspaceUnobservable("a workspace directory changed while it was read")

    for name, info in children:
        if "\x00" in name or "/" in name:
            raise WorkspaceUnobservable("a workspace path is not describable")
        relative = "{}/{}".format(prefix, name) if prefix else name
        mode = info.st_mode
        if stat.S_ISLNK(mode):
            _digest_link(dir_fd, name, relative, info, digest, counters)
        elif stat.S_ISDIR(mode):
            if at_root and name in MANIFEST_EXCLUDED_DIRECTORIES:
                continue
            if policy.classifies(name, at_root=at_root):
                # Classified as control-plane runtime state. Its *presence* is
                # still recorded - removing it is drift - but its contents are
                # another component's business, revalidated by that component's
                # own host-owned lane rather than by this digest.
                counters.count()
                digest.update("{}\0runtime-state\n".format(relative).encode("utf-8"))
                continue
            counters.count()
            digest.update("{}\0dir\n".format(relative).encode("utf-8"))
            _digest_child_directory(
                dir_fd, name, relative, info, digest, counters, depth + 1, policy,
            )
        elif stat.S_ISREG(mode):
            _digest_file(dir_fd, name, relative, info, digest, counters)
        else:
            raise WorkspaceUnobservable("a workspace entry is not a regular file")


def _digest_child_directory(
    dir_fd: int,
    name: str,
    relative: str,
    seen: os.stat_result,
    digest: "hashlib._Hash",
    counters: _ManifestCounters,
    depth: int,
    policy: SnapshotPolicy,
) -> None:
    try:
        child = os.open(
            name, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
            dir_fd=dir_fd,
        )
    except OSError as exc:
        # `ELOOP`/`ENOTDIR` here is a directory that became a link or a file
        # between the listing and the descent. That is the swap this design
        # exists to detect, not a directory to skip.
        raise WorkspaceUnobservable("a workspace directory changed while it was read") from exc
    try:
        opened = os.fstat(child)
        if (opened.st_dev, opened.st_ino) != (seen.st_dev, seen.st_ino):
            raise WorkspaceUnobservable("a workspace directory changed while it was read")
        _digest_directory(
            child, relative, digest, counters, depth=depth, policy=policy,
        )
    finally:
        os.close(child)


def _digest_link(
    dir_fd: int,
    name: str,
    relative: str,
    info: os.stat_result,
    digest: "hashlib._Hash",
    counters: _ManifestCounters,
) -> None:
    """Record a link as itself: where it is, and where it points."""

    try:
        target = os.readlink(name, dir_fd=dir_fd)
    except OSError as exc:
        raise WorkspaceUnobservable("a workspace symlink cannot be read") from exc
    if len(target.encode("utf-8", "surrogateescape")) > MAX_MANIFEST_LINK_BYTES:
        raise WorkspaceUnobservable("a workspace symlink target exceeds its bound")
    counters.count()
    # The target is hashed rather than embedded so one absurd name cannot
    # dominate the manifest, and it is domain-separated from file content so a
    # link can never collide with a file holding the same text.
    target_digest = hashlib.sha256(
        b"link\0" + target.encode("utf-8", "surrogateescape"),
    ).hexdigest()
    digest.update("{}\0link\0{}\n".format(relative, target_digest).encode("utf-8"))


def _digest_file(
    dir_fd: int,
    name: str,
    relative: str,
    seen: os.stat_result,
    digest: "hashlib._Hash",
    counters: _ManifestCounters,
) -> None:
    """Stream one file through a single no-follow descriptor, or fail closed."""

    try:
        handle = os.open(
            name, os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC, dir_fd=dir_fd,
        )
    except OSError as exc:
        raise WorkspaceUnobservable("a workspace file cannot be opened safely") from exc
    try:
        opened = os.fstat(handle)
        if not stat.S_ISREG(opened.st_mode):
            raise WorkspaceUnobservable("a workspace entry changed type while it was read")
        if (opened.st_dev, opened.st_ino) != (seen.st_dev, seen.st_ino):
            raise WorkspaceUnobservable("a workspace file changed while it was read")
        if opened.st_size > MAX_MANIFEST_FILE_BYTES:
            raise WorkspaceUnobservable("a workspace file exceeds the manifest bound")
        content = hashlib.sha256()
        read = 0
        while True:
            chunk = os.read(handle, _MANIFEST_CHUNK_BYTES)
            if not chunk:
                break
            read += len(chunk)
            if read > MAX_MANIFEST_FILE_BYTES:
                # Bounded *while* reading, not only from the initial size: a
                # file that grows after it is stat-ed must not be buffered or
                # streamed without limit.
                raise WorkspaceUnobservable("a workspace file exceeds the manifest bound")
            content.update(chunk)
        after = os.fstat(handle)
        if read != after.st_size or (
            (after.st_dev, after.st_ino, after.st_size, after.st_mode, after.st_mtime_ns)
            != (
                opened.st_dev, opened.st_ino, opened.st_size,
                opened.st_mode, opened.st_mtime_ns,
            )
        ):
            raise WorkspaceUnobservable("a workspace file changed while it was read")
        counters.count(read)
        digest.update("{}\0file\0{}\0{}\0{}\n".format(
            relative, "x" if opened.st_mode & 0o111 else "-", read,
            content.hexdigest(),
        ).encode("utf-8"))
    finally:
        os.close(handle)


# ──────────────────────────────────────────────────────────────────────
# exact record parsing
# ──────────────────────────────────────────────────────────────────────


def _exact_keys(value: Any, expected: frozenset, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContinuationCorrupt("{} is not an object".format(what))
    present = set(value)
    if present != expected:
        # Missing and unknown are the same refusal. A record with an extra key
        # is a record somebody else's code wrote, and a record with a missing
        # key is one this code cannot fully bind. Neither is actionable.
        raise ContinuationCorrupt("{} does not match its exact schema".format(what))
    return value


def _text(value: Mapping[str, Any], key: str, pattern: "re.Pattern[str]") -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or isinstance(raw, bool) or not pattern.fullmatch(raw):
        raise ContinuationCorrupt("a stored field is unreadable")
    return raw


def _counter(value: Any, *, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        # `True` is an `int` in Python and would otherwise read as `1`.
        raise ContinuationCorrupt("a stored counter is not an integer")
    if value < low or value > high:
        raise ContinuationCorrupt("a stored counter is outside its bound")
    return value


def _timestamp(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContinuationCorrupt("a stored timestamp is not a number")
    number = float(value)
    if not math.isfinite(number) or number < _MIN_TIMESTAMP or number > _MAX_TIMESTAMP:
        raise ContinuationCorrupt("a stored timestamp is not a real time")
    return number


def _session(value: Any) -> str:
    if not is_continuable_session(value):
        raise ContinuationCorrupt("a stored session is not continuable")
    return str(value)


def _workspace_path(value: Any) -> str:
    if (
        not isinstance(value, str)
        or isinstance(value, bool)
        or not value
        or len(value) > 4096
        or "\x00" in value
        or not value.startswith("/")
    ):
        raise ContinuationCorrupt("a stored workspace path is unreadable")
    return value


def _optional_job_id(value: Any) -> str:
    if value == "":
        return ""
    if not isinstance(value, str) or not _JOB_ID_RE.fullmatch(value):
        raise ContinuationCorrupt("a stored claimant is unreadable")
    return value


def _attributable(value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ContinuationCorrupt("a stored attributable set is unreadable")
    if len(value) > MAX_CONTINUATION_FILES:
        raise ContinuationCorrupt("a stored attributable set exceeds its bound")
    for item in value:
        if not isinstance(item, str) or isinstance(item, bool) or not item:
            raise ContinuationCorrupt("a stored attributable path is unreadable")
        if len(item) > 1024 or "\x00" in item or item.startswith("/"):
            raise ContinuationCorrupt("a stored attributable path is unsafe")
    if list(value) != sorted(set(value)):
        # Canonical shape only: a duplicated or reordered list hashes
        # differently from the one that was granted.
        raise ContinuationCorrupt("a stored attributable set is not canonical")
    return tuple(value)


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


# ──────────────────────────────────────────────────────────────────────
# the authority record
# ──────────────────────────────────────────────────────────────────────

#: Every field an authority persists, and the exact set a stored record must
#: carry. Adding one is a schema change that invalidates every existing record,
#: which is the intended cost.
_AUTHORITY_FIELDS: Tuple[str, ...] = (
    "tenant_ref", "backend", "session_id", "job_id", "origin_job_id",
    "working_dir", "workspace_sha256", "revision_sha256",
    "workspace_manifest_sha256", "snapshot_policy_sha256", "files",
    "authorized_config_sha256", "contract_snapshot_sha256",
    "request_sha256", "failure_code", "generation", "sequence", "state",
    "claimed_by_job_id", "created_at", "updated_at",
)
_AUTHORITY_KEYS = frozenset(_AUTHORITY_FIELDS) | {"authority_version", "record_sha256"}


@dataclass(frozen=True)
class ContinuationAuthority:
    """One tenant-local, single-use permission to re-enter one exact session."""

    tenant_ref: str
    backend: str
    session_id: str
    job_id: str
    origin_job_id: str
    working_dir: str
    workspace_sha256: str
    revision_sha256: str
    #: The whole tree, not just what was attributed. See the module docstring:
    #: an unrelated new file is drift the attributable digest cannot see.
    workspace_manifest_sha256: str
    #: The projection the manifest above was taken under. Frozen for the whole
    #: life of the session: a later segment that would observe the tree
    #: differently is not continuing the same observation.
    snapshot_policy_sha256: str
    files: Tuple[str, ...]
    authorized_config_sha256: str
    #: Content address of the canonical contract snapshot the stopped round was
    #: authorized under. The snapshot itself lives in the origin job's private
    #: record; binding only its identity here keeps the journal small and makes
    #: the pair tamper-evident, because a later segment must produce a snapshot
    #: that still hashes to this. An authority can therefore neither carry a
    #: contract forward silently nor accept a rewritten one.
    contract_snapshot_sha256: str
    request_sha256: str
    failure_code: str
    generation: int
    #: This authority's position in its session's durable transition journal.
    #: The journal, not this number, is authoritative; carrying it here is what
    #: lets a load detect an authority body that does not belong to the tail.
    sequence: int
    state: str = STATE_OPEN
    claimed_by_job_id: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def content_digest(self) -> str:
        """Hash every persisted field, with no exceptions.

        `updated_at` is included precisely because the first version excluded
        it: any field that is stored but unhashed is a field an editor may
        change for free.
        """

        payload = {"authority_version": CONTINUATION_AUTHORITY_VERSION}
        for name in _AUTHORITY_FIELDS:
            value = getattr(self, name)
            payload[name] = list(value) if name == "files" else value
        digest = hashlib.sha256()
        digest.update(_AUTHORITY_DOMAIN)
        digest.update(_canonical(payload))
        return digest.hexdigest()

    def to_mapping(self) -> Dict[str, Any]:
        mapping: Dict[str, Any] = {
            "authority_version": CONTINUATION_AUTHORITY_VERSION,
        }
        for name in _AUTHORITY_FIELDS:
            value = getattr(self, name)
            mapping[name] = list(value) if name == "files" else value
        mapping["record_sha256"] = self.content_digest()
        return mapping

    @classmethod
    def from_mapping(cls, value: Any) -> "ContinuationAuthority":
        """Rebuild an authority, or refuse. Never repair, never guess a field."""

        record = _exact_keys(value, _AUTHORITY_KEYS, "continuation authority")
        if str(record.get("authority_version")) != CONTINUATION_AUTHORITY_VERSION:
            raise ContinuationCorrupt("continuation authority version is unsupported")
        state = str(record.get("state") or "")
        if state not in _STATES:
            raise ContinuationCorrupt("continuation authority state is unknown")
        authority = cls(
            tenant_ref=_text(record, "tenant_ref", _SHA256_RE),
            backend=_text(record, "backend", _BACKEND_RE),
            session_id=_session(record.get("session_id")),
            job_id=_text(record, "job_id", _JOB_ID_RE),
            origin_job_id=_text(record, "origin_job_id", _JOB_ID_RE),
            working_dir=_workspace_path(record.get("working_dir")),
            workspace_sha256=_text(record, "workspace_sha256", _SHA256_RE),
            revision_sha256=_text(record, "revision_sha256", _SHA256_RE),
            workspace_manifest_sha256=_text(
                record, "workspace_manifest_sha256", _SHA256_RE,
            ),
            snapshot_policy_sha256=_text(record, "snapshot_policy_sha256", _SHA256_RE),
            files=_attributable(record.get("files")),
            authorized_config_sha256=_text(record, "authorized_config_sha256", _SHA256_RE),
            contract_snapshot_sha256=_text(
                record, "contract_snapshot_sha256", _SHA256_RE,
            ),
            request_sha256=_text(record, "request_sha256", _SHA256_RE),
            failure_code=_text(record, "failure_code", _CODE_RE),
            generation=_counter(
                record.get("generation"), low=1, high=MAX_CONTINUATION_GENERATION,
            ),
            sequence=_counter(
                record.get("sequence"), low=1, high=MAX_CONTINUATION_TRANSITIONS,
            ),
            state=state,
            claimed_by_job_id=_optional_job_id(record.get("claimed_by_job_id")),
            created_at=_timestamp(record.get("created_at")),
            updated_at=_timestamp(record.get("updated_at")),
        )
        stored = record.get("record_sha256")
        if not isinstance(stored, str) or not _SHA256_RE.fullmatch(stored):
            raise ContinuationCorrupt("continuation authority digest is unreadable")
        if stored != authority.content_digest():
            raise ContinuationCorrupt("continuation authority integrity check failed")
        # Shape is not the same thing as sense. A record can be perfectly typed,
        # perfectly signed, and still describe a state that no transition could
        # have produced - an open offer that already names a consumer, a claim
        # with no claimant, a stop code this host never emits.
        assert_authority_semantics(authority)
        return authority

    # -- transitions ---------------------------------------------------
    # Each returns the *next* record. None of them writes anything: only
    # `ContinuationStore.commit` may, and only against the journal tail.

    def claimed(self, job_id: str, now: float) -> "ContinuationAuthority":
        return replace(
            self, state=STATE_CLAIMED, claimed_by_job_id=job_id,
            sequence=self.sequence + 1, updated_at=now,
        )

    def settled(self, now: float) -> "ContinuationAuthority":
        return replace(
            self, state=STATE_SETTLED, sequence=self.sequence + 1, updated_at=now,
        )

    def rotated(
        self,
        *,
        job_id: str,
        revision_sha256: str,
        workspace_manifest_sha256: str,
        files: Sequence[str],
        failure_code: str,
        now: float,
    ) -> "ContinuationAuthority":
        """Carry the same session forward one generation, never sideways."""

        return replace(
            self,
            job_id=job_id,
            revision_sha256=revision_sha256,
            workspace_manifest_sha256=workspace_manifest_sha256,
            files=tuple(sorted({str(item) for item in files})),
            failure_code=failure_code,
            generation=self.generation + 1,
            sequence=self.sequence + 1,
            state=STATE_OPEN,
            claimed_by_job_id="",
            updated_at=now,
        )


def assert_authority_semantics(authority: "ContinuationAuthority") -> None:
    """State-dependent rules a well-formed record must satisfy on its own."""

    if authority.state == STATE_CLAIMED and not authority.claimed_by_job_id:
        raise ContinuationCorrupt("a claimed continuation names no claimant")
    if authority.state == STATE_OPEN and authority.claimed_by_job_id:
        # An offer that already names a consumer is either a forgery or a
        # half-applied edit. A *settled* record, by contrast, legitimately
        # remembers whoever consumed it - that is the historical fact it
        # exists to record.
        raise ContinuationCorrupt("an open continuation already names a claimant")
    if authority.created_at > authority.updated_at:
        raise ContinuationCorrupt("a continuation was updated before it was created")
    if authority.failure_code not in CONTINUABLE_STOP_CODES:
        # The authority exists because of a recognized bounded stop. A record
        # naming any other reason was not produced by that path.
        raise ContinuationCorrupt("a continuation names an unrecognized stop")


def check_transition(
    previous: "ContinuationAuthority", updated: "ContinuationAuthority",
) -> None:
    """Enforce the exact transition table, including what may not change."""

    allowed = _LEGAL_TRANSITIONS.get(previous.state, frozenset())
    if updated.state not in allowed:
        raise ContinuationCorrupt("a continuation transition is not legal")
    delta = _TRANSITION_GENERATION_DELTA[(previous.state, updated.state)]
    if updated.generation != previous.generation + delta:
        raise ContinuationCorrupt("a continuation transition moved the generation illegally")
    if updated.generation > MAX_CONTINUATION_GENERATION:
        raise ContinuationCorrupt("a continuation generation is outside its bound")
    for field in _TRANSITION_INVARIANT_FIELDS:
        if getattr(previous, field) != getattr(updated, field):
            raise ContinuationCorrupt("a continuation transition changed a bound field")
    if updated.updated_at < previous.updated_at:
        raise ContinuationCorrupt("a continuation transition moved backwards in time")
    if updated.state == STATE_CLAIMED:
        # A claim consumes exactly the segment it found. Everything describing
        # that segment must survive it untouched.
        for field in ("job_id", "revision_sha256", "workspace_manifest_sha256",
                      "files", "failure_code"):
            if getattr(previous, field) != getattr(updated, field):
                raise ContinuationCorrupt("a claim changed what it was claiming")
    if updated.state == STATE_SETTLED:
        for field in ("job_id", "revision_sha256", "workspace_manifest_sha256",
                      "files", "failure_code", "claimed_by_job_id"):
            if getattr(previous, field) != getattr(updated, field):
                raise ContinuationCorrupt("a settlement changed what it was settling")
    assert_authority_semantics(updated)


# ──────────────────────────────────────────────────────────────────────
# the transition journal
# ──────────────────────────────────────────────────────────────────────

_JOURNAL_FIELDS: Tuple[str, ...] = (
    "sequence", "generation", "state", "authority_sha256", "previous_entry_sha256",
    "recorded_at",
)
_JOURNAL_KEYS = frozenset(_JOURNAL_FIELDS) | {"journal_version", "entry_sha256"}


@dataclass(frozen=True)
class JournalEntry:
    """One irreversible step, hash-chained to the step before it."""

    sequence: int
    generation: int
    state: str
    authority_sha256: str
    previous_entry_sha256: str
    recorded_at: float

    def entry_digest(self) -> str:
        payload = {"journal_version": CONTINUATION_JOURNAL_VERSION}
        for name in _JOURNAL_FIELDS:
            payload[name] = getattr(self, name)
        digest = hashlib.sha256()
        digest.update(_JOURNAL_DOMAIN)
        digest.update(_canonical(payload))
        return digest.hexdigest()

    def to_line(self) -> bytes:
        payload = {"journal_version": CONTINUATION_JOURNAL_VERSION}
        for name in _JOURNAL_FIELDS:
            payload[name] = getattr(self, name)
        payload["entry_sha256"] = self.entry_digest()
        return _canonical(payload) + b"\n"

    @classmethod
    def from_mapping(cls, value: Any) -> "JournalEntry":
        record = _exact_keys(value, _JOURNAL_KEYS, "continuation journal entry")
        if str(record.get("journal_version")) != CONTINUATION_JOURNAL_VERSION:
            raise ContinuationCorrupt("continuation journal version is unsupported")
        state = str(record.get("state") or "")
        if state not in _STATES:
            raise ContinuationCorrupt("continuation journal state is unknown")
        previous = record.get("previous_entry_sha256")
        if previous != "" and (
            not isinstance(previous, str) or not _SHA256_RE.fullmatch(previous)
        ):
            raise ContinuationCorrupt("continuation journal chain is unreadable")
        entry = cls(
            sequence=_counter(
                record.get("sequence"), low=1, high=MAX_CONTINUATION_TRANSITIONS,
            ),
            generation=_counter(
                record.get("generation"), low=1, high=MAX_CONTINUATION_GENERATION,
            ),
            state=state,
            authority_sha256=_text(record, "authority_sha256", _SHA256_RE),
            previous_entry_sha256=str(previous),
            recorded_at=_timestamp(record.get("recorded_at")),
        )
        stored = record.get("entry_sha256")
        if not isinstance(stored, str) or stored != entry.entry_digest():
            raise ContinuationCorrupt("continuation journal integrity check failed")
        return entry


def read_journal(raw: bytes) -> Tuple[JournalEntry, ...]:
    """Parse a whole journal, or refuse the whole journal.

    Every structural rule is checked here rather than at use time, so a caller
    that gets a tail back knows the chain behind it is intact: strictly
    increasing sequence from one, unbroken hash chain, non-decreasing
    generation, and only transitions this module can produce.

    A trailing partial line - the shape an interrupted append leaves - refuses
    the journal rather than being discarded. Silently dropping it would let a
    crash mid-transition read back as the state before it, which is the
    difference between "unavailable" and "reopened".
    """

    if not raw:
        return ()
    if not raw.endswith(b"\n"):
        raise ContinuationCorrupt("continuation journal is truncated")
    entries = []
    previous: Optional[JournalEntry] = None
    for line in raw.split(b"\n"):
        if not line:
            continue
        try:
            value = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ContinuationCorrupt("continuation journal is unreadable") from exc
        entry = JournalEntry.from_mapping(value)
        if previous is None:
            if entry.sequence != 1 or entry.previous_entry_sha256 != "":
                raise ContinuationCorrupt("continuation journal does not start cleanly")
            if entry.state != STATE_OPEN or entry.generation != 1:
                raise ContinuationCorrupt("continuation journal does not start at an open first generation")
        else:
            if entry.sequence != previous.sequence + 1:
                raise ContinuationCorrupt("continuation journal sequence is not monotonic")
            if entry.previous_entry_sha256 != previous.entry_digest():
                raise ContinuationCorrupt("continuation journal chain is broken")
            if entry.state not in _LEGAL_TRANSITIONS[previous.state]:
                raise ContinuationCorrupt("continuation journal records an impossible transition")
            # The exact table, not merely "did not go backwards". A claim that
            # advances the generation would consume one segment and mint the
            # next in a single step, which is how a self-consistent forged
            # `open(1) -> claimed(2)` was accepted.
            expected = previous.generation + _TRANSITION_GENERATION_DELTA[
                (previous.state, entry.state)
            ]
            if entry.generation != expected:
                raise ContinuationCorrupt(
                    "continuation journal moved the generation illegally",
                )
            if entry.recorded_at < previous.recorded_at:
                raise ContinuationCorrupt("continuation journal moved backwards in time")
        entries.append(entry)
        previous = entry
    if len(entries) > MAX_CONTINUATION_TRANSITIONS:
        raise ContinuationCorrupt("continuation journal exceeds its transition bound")
    return tuple(entries)


# ──────────────────────────────────────────────────────────────────────
# secure, descriptor-relative storage
# ──────────────────────────────────────────────────────────────────────

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)

try:  # pragma: no cover - the Windows branch is exercised by static review.
    from fcntl import LOCK_EX as _LOCK_EX, flock as _flock
except ImportError:  # pragma: no cover
    _flock = None
    _LOCK_EX = 0


class ContinuationStore:
    """Tenant-partitioned, symlink-refusing storage for continuation state.

    Partitioning is the non-disclosure mechanism: a lookup happens inside the
    *authenticated caller's* tenant directory, so a session belonging to
    somebody else is not "denied", it is simply not there.

    Every path component is opened with `O_NOFOLLOW | O_DIRECTORY` from the
    state root downwards and every file is opened relative to that descriptor,
    so a directory or file replaced by a symlink between the check and the open
    is refused rather than followed out of the state root.
    """

    def __init__(self, state_root: Path) -> None:
        self.state_root = Path(state_root)
        if not (_O_NOFOLLOW and _O_DIRECTORY):
            # Fail closed rather than silently degrade to a followable open.
            raise RuntimeError(
                "this platform cannot open continuation state without following links",
            )

    # -- path handling -------------------------------------------------

    def path(self, tenant_ref: str, session_id: str) -> Path:
        """The authority's location. For diagnostics and tests, never for I/O."""

        return self._directory_path(tenant_ref) / (session_ref(session_id) + ".json")

    def journal_path(self, tenant_ref: str, session_id: str) -> Path:
        return self._directory_path(tenant_ref) / (session_ref(session_id) + ".journal")

    def _directory_path(self, tenant_ref: str) -> Path:
        if not _SHA256_RE.fullmatch(tenant_ref):
            raise ValueError("continuation tenant reference is invalid")
        return self.state_root / "tenants" / tenant_ref / "continuation"

    def _open_directory(self, tenant_ref: str, *, create: bool) -> int:
        """Walk the *whole* absolute state path without following one link.

        The previous version called ``state_root.mkdir(parents=True)`` and then
        applied ``O_NOFOLLOW`` only from the root downwards, so a symlink in an
        ancestor of the configured root was followed and state was written on
        the other side of it. An audit probe did exactly that.

        Every component from ``/`` is now opened, and created when missing,
        relative to the descriptor of the component above it. A link anywhere
        in the ancestry has no lookup to intercept.
        """

        if not _SHA256_RE.fullmatch(tenant_ref):
            raise ValueError("continuation tenant reference is invalid")
        root = str(self.state_root)
        if not root.startswith("/"):
            raise ValueError("continuation state root is not absolute")
        parts = list(PurePosixPath(root).parts[1:]) + [
            "tenants", tenant_ref, "continuation",
        ]
        handle = os.open("/", os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC)
        try:
            for part in parts:
                if part in ("", ".", ".."):
                    raise ValueError("continuation state path is not canonical")
                if create:
                    try:
                        os.mkdir(part, 0o700, dir_fd=handle)
                        _fsync_directory(handle)
                    except FileExistsError:
                        pass
                nested = os.open(
                    part, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
                    dir_fd=handle,
                )
                os.close(handle)
                handle = nested
            _assert_private_directory(handle)
            return handle
        except BaseException:
            os.close(handle)
            raise

    def _read(self, tenant_ref: str, name: str) -> bytes:
        """Read one state file, bounded while reading and pinned to one inode."""

        try:
            directory = self._open_directory(tenant_ref, create=False)
        except (OSError, ValueError):
            return b""
        try:
            handle = os.open(
                name, os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC, dir_fd=directory,
            )
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.EMLINK}:
                # The final component is a symlink. Refusing to read it is the
                # point; an attacker's link must not become this host's state.
                raise ContinuationCorrupt("continuation state is not a regular file")
            return b""
        finally:
            os.close(directory)
        try:
            opened = _assert_private_file(handle)
            if opened.st_size > MAX_STATE_FILE_BYTES:
                raise ContinuationCorrupt("continuation state exceeds its bound")
            chunks = []
            total = 0
            while True:
                chunk = os.read(handle, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_STATE_FILE_BYTES:
                    # Bounded *while* reading. A file that grows after its
                    # first stat must not be buffered without limit.
                    raise ContinuationCorrupt("continuation state exceeds its bound")
                chunks.append(chunk)
            after = os.fstat(handle)
            if total != after.st_size or (
                (after.st_dev, after.st_ino, after.st_size, after.st_mode,
                 after.st_mtime_ns)
                != (
                    opened.st_dev, opened.st_ino, opened.st_size,
                    opened.st_mode, opened.st_mtime_ns,
                )
            ):
                raise ContinuationCorrupt("continuation state changed while it was read")
            return b"".join(chunks)
        finally:
            os.close(handle)

    # -- reading -------------------------------------------------------

    def journal(self, tenant_ref: str, session_id: str) -> Tuple[JournalEntry, ...]:
        try:
            raw = self._read(tenant_ref, session_ref(session_id) + ".journal")
        except (ContinuationCorrupt, ValueError):
            raise
        return read_journal(raw)

    def load(self, tenant_ref: str, session_id: str) -> Optional[ContinuationAuthority]:
        """Return the authority, or `None` for absent *and* for unusable.

        The two are the same answer on purpose. A corrupt, replayed or
        journal-contradicting authority is not a different outcome a caller may
        learn about; it is an authority nobody can act on, and telling them
        apart would confirm one existed.

        The journal tail is the only source of "which generation and state this
        session is at". The JSON body may only *describe* that tail.
        """

        try:
            entries = self.journal(tenant_ref, session_id)
        except (ContinuationCorrupt, ValueError, OSError):
            return None
        if not entries:
            return None
        tail = entries[-1]
        try:
            raw = self._read(tenant_ref, session_ref(session_id) + ".json")
        except (ContinuationCorrupt, ValueError, OSError):
            return None
        if not raw:
            return None
        try:
            authority = ContinuationAuthority.from_mapping(json.loads(raw.decode("utf-8")))
        except (ContinuationCorrupt, ValueError, UnicodeDecodeError):
            return None
        if authority.tenant_ref != tenant_ref or authority.session_id != session_id:
            # A record moved into another tenant's directory, or renamed onto a
            # different session's key, binds neither.
            return None
        if (
            authority.content_digest() != tail.authority_sha256
            or authority.sequence != tail.sequence
            or authority.generation != tail.generation
            or authority.state != tail.state
        ):
            # This is the anti-replay check. An older authority body is
            # perfectly self-consistent - it was validly signed once - so its
            # own digest proves nothing. Only the append-only tail can say
            # which body is current, and a body that is not the tail's body is
            # a replay no matter how well formed it is.
            return None
        return authority

    def open_authority(
        self, tenant_ref: str, session_id: str,
    ) -> Optional[ContinuationAuthority]:
        """Return the authority only while it is still unconsumed."""

        authority = self.load(tenant_ref, session_id)
        if authority is None or authority.state != STATE_OPEN:
            return None
        return authority

    def is_unpinned_legacy(self, tenant_ref: str, session_id: str) -> bool:
        """Whether this tenant's stored session is a pre-pinning v2 record.

        Deliberately narrow. It answers one question - "is the reason you got
        nothing back that the record predates contract pinning?" - and it
        answers it only inside the caller's own tenant partition, keyed by the
        session the caller already named. A record belonging to another tenant,
        another session, or no recognizable schema reads as `False`, so this
        can never become an existence oracle for somebody else's continuation.

        Nothing here is trusted for authority. The record is not rebuilt, not
        repaired and not upgraded; the only fact extracted is a version string,
        and the only use of it is to make the refusal precise.
        """

        try:
            raw = self._read(tenant_ref, session_ref(session_id) + ".json")
        except (ContinuationCorrupt, ValueError, OSError):
            return False
        if not raw:
            return False
        try:
            record = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return False
        if not isinstance(record, dict):
            # `json.loads` produces exactly `dict` for an object, so this is the
            # precise test rather than an abstract one.
            return False
        return (
            str(record.get("authority_version") or "")
            == LEGACY_UNPINNED_AUTHORITY_VERSION
            and str(record.get("tenant_ref") or "") == tenant_ref
            and str(record.get("session_id") or "") == session_id
        )

    # -- writing -------------------------------------------------------

    def create(self, authority: ContinuationAuthority) -> ContinuationAuthority:
        """Record the first transition for a session that has none."""

        if authority.sequence != 1 or authority.generation != 1:
            raise ContinuationConflict("a first continuation must be generation one")
        if authority.state != STATE_OPEN:
            raise ContinuationConflict("a first continuation must be open")
        now = time.time()
        stamped = replace(authority, created_at=now, updated_at=now)
        return self._commit(None, stamped)

    def commit(
        self, previous: ContinuationAuthority, updated: ContinuationAuthority,
    ) -> ContinuationAuthority:
        """Advance a session by one transition, or refuse.

        Compare-and-swap against the durable tail, not against whatever the
        caller last read. Two processes that both loaded the same open
        authority will both arrive here; only the one whose `previous` still
        matches the tail may proceed, and the other raises.
        """

        return self._commit(previous, updated)

    def _commit(
        self,
        previous: Optional[ContinuationAuthority],
        updated: ContinuationAuthority,
    ) -> ContinuationAuthority:
        """Read the tail, decide, and append - all while holding the journal.

        The lock is on the journal file itself rather than on any service
        object, because the processes this has to be safe against do not share
        a service object, an interpreter or a machine-local convention. Two
        Codex workers that both loaded the same open authority will both arrive
        here; the second one re-reads the tail *inside* the lock, finds it has
        moved, and is refused. Without the lock both would append and the
        journal would end with two entries claiming the same sequence -
        fail-closed, but fail-closed for everybody, which is an outage rather
        than an answer.
        """

        tenant_ref, session_id = updated.tenant_ref, updated.session_id
        directory = self._open_directory(tenant_ref, create=True)
        try:
            handle = os.open(
                session_ref(session_id) + ".journal",
                os.O_RDWR | os.O_CREAT | _O_NOFOLLOW | _O_CLOEXEC,
                0o600,
                dir_fd=directory,
            )
            try:
                info = os.fstat(handle)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    # A hard link is another name for this file somebody else
                    # may own, and a lock on it protects nothing.
                    raise ContinuationCorrupt(
                        "the continuation journal is not a private file",
                    )
                _lock_exclusive(handle)
                entries = read_journal(_read_all(handle))
                tail = entries[-1] if entries else None
                self._check_transition(previous, updated, entries, tail)
                entry = JournalEntry(
                    sequence=updated.sequence,
                    generation=updated.generation,
                    state=updated.state,
                    authority_sha256=updated.content_digest(),
                    previous_entry_sha256=(
                        tail.entry_digest() if tail is not None else ""
                    ),
                    recorded_at=updated.updated_at,
                )
                line = entry.to_line()
                os.lseek(handle, 0, os.SEEK_END)
                if os.write(handle, line) != len(line):
                    raise ContinuationCorrupt(
                        "the continuation journal could not be appended",
                    )
                os.fsync(handle)
                _fsync_directory(directory)
                # Journal first, then the body, and both under the lock. A
                # crash between them leaves a tail the body does not match, and
                # `load` refuses that: availability is lost, which is safe. The
                # other order would leave a body ahead of the tail, and a body
                # nobody recorded is exactly a forged one.
                self._replace_json(
                    directory, session_ref(session_id) + ".json", updated,
                )
            finally:
                os.close(handle)
        finally:
            os.close(directory)
        return updated

    @staticmethod
    def _check_transition(
        previous: Optional[ContinuationAuthority],
        updated: ContinuationAuthority,
        entries: Sequence[JournalEntry],
        tail: Optional[JournalEntry],
    ) -> None:
        """Everything that must be true for this step to be the next one.

        Deliberately checked here *and* again when the journal is parsed. A
        rule enforced only on the way in is a rule that anybody who can write
        the file gets to skip; a rule enforced only on the way out cannot stop
        this process from recording nonsense in the first place.
        """

        if previous is None:
            if tail is not None:
                raise ContinuationConflict("this continuation session already exists")
        else:
            if tail is None or tail.authority_sha256 != previous.content_digest():
                raise ContinuationConflict("the continuation high-water mark moved")
            if updated.sequence != tail.sequence + 1:
                raise ContinuationConflict("a continuation transition is not the next one")
            try:
                check_transition(previous, updated)
            except ContinuationCorrupt as exc:
                raise ContinuationConflict(str(exc)) from None
        assert_authority_semantics(updated)
        if len(entries) + 1 > MAX_CONTINUATION_TRANSITIONS:
            raise ContinuationConflict("this continuation session has no transitions left")

    @staticmethod
    def _replace_json(
        directory: int, name: str, authority: ContinuationAuthority,
    ) -> None:
        """Write the body to a fresh private file, then rename it into place.

        The temporary name is unpredictable and created with ``O_EXCL``. A
        PID-derived name is guessable by anyone who can list the directory, and
        `O_CREAT | O_TRUNC` on a guessed name will happily reuse an attacker's
        hard link or leftover residue - so the write would land in a file this
        process does not own and the rename would publish it.
        """

        temporary = ".{}.{}.tmp".format(name, os.urandom(16).hex())
        handle = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        try:
            _assert_private_file(handle)
            payload = _canonical(authority.to_mapping())
            if os.write(handle, payload) != len(payload):
                raise ContinuationCorrupt("the continuation authority could not be written")
            os.fsync(handle)
        except BaseException:
            os.close(handle)
            try:
                os.unlink(temporary, dir_fd=directory)
            except OSError:
                pass
            raise
        else:
            os.close(handle)
        os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        _fsync_directory(directory)


def _read_all(handle: int) -> bytes:
    os.lseek(handle, 0, os.SEEK_SET)
    chunks = []
    total = 0
    while True:
        chunk = os.read(handle, 1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_STATE_FILE_BYTES:
            raise ContinuationCorrupt("continuation state exceeds its bound")
        chunks.append(chunk)
    return b"".join(chunks)


def _assert_private_file(handle: int) -> os.stat_result:
    """Refuse a state file this process does not privately own.

    Three separate conditions, and each has been a real escalation somewhere:
    a non-regular file (a device or FIFO the reader would block on), a file
    with more than one link (another name for the same inode that somebody else
    controls), and group- or world-accessible permissions.
    """

    info = os.fstat(handle)
    if not stat.S_ISREG(info.st_mode):
        raise ContinuationCorrupt("continuation state is not a regular file")
    if info.st_nlink != 1:
        raise ContinuationCorrupt("continuation state is not a private file")
    if info.st_uid != os.geteuid():
        raise ContinuationCorrupt("continuation state belongs to another account")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ContinuationCorrupt("continuation state is not privately permissioned")
    return info


def _assert_private_directory(handle: int) -> os.stat_result:
    info = os.fstat(handle)
    if not stat.S_ISDIR(info.st_mode):
        raise ContinuationCorrupt("continuation state is not a directory")
    if info.st_uid != os.geteuid():
        raise ContinuationCorrupt("continuation state belongs to another account")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise ContinuationCorrupt("continuation state is not privately permissioned")
    return info


def _lock_exclusive(handle: int) -> None:
    """Serialize transitions across processes, or refuse to transition at all."""

    if _flock is None:
        # No advisory locking means no way to make a compare-and-swap atomic
        # against another process. Refusing is the only honest option: the
        # alternative is a mechanism that only looks single-owner.
        raise ContinuationCorrupt(
            "this platform cannot serialize continuation transitions",
        )
    _flock(handle, _LOCK_EX)


def _fsync_directory(handle: int) -> None:
    """Make a rename, create or unlink durable, or refuse to claim it is.

    The previous version swallowed the error, on the reasoning that some
    filesystems refuse to fsync a directory descriptor and failing here would
    turn a durability nicety into an outage. That reasoning is wrong for this
    record. Without a durable directory entry, a power loss can leave the
    journal or the authority body missing or reverted - and the caller has
    already been told the transition succeeded. A continuation that regresses
    after a crash is precisely the double-spend this whole mechanism exists to
    prevent, so an undurable write is a failed write.

    Losing availability here is safe: the transition is either fully recorded
    or unusable, and `load` refuses a half-written pair.
    """

    try:
        os.fsync(handle)
    except OSError as exc:
        raise ContinuationCorrupt(
            "a continuation state directory could not be made durable",
        ) from exc


def secure_directory(path: Path, *, create: bool = True) -> Path:
    """Open - and optionally create - an absolute directory path, link-free.

    Every component from ``/`` is opened relative to the descriptor of the one
    above it with ``O_DIRECTORY | O_NOFOLLOW``, and missing components are
    created private through that same descriptor. A symlink anywhere in the
    ancestry has no lookup to intercept, so nothing is ever created or written
    on the far side of one.

    This is deliberately not `Path.resolve()` followed by `mkdir(parents=True)`.
    That pair follows links twice - once to decide where to write and once to
    write - and an audit probe used exactly that to place state outside the
    configured root.

    Returns the same path it was given, unresolved, so callers keep configuring
    the host in the terms they wrote down.
    """

    absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
    if not str(absolute).startswith("/"):
        raise ContinuationCorrupt("a state path is not absolute")
    handle = os.open("/", os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC)
    try:
        for part in PurePosixPath(str(absolute)).parts[1:]:
            if part in ("", ".", ".."):
                raise ContinuationCorrupt("a state path is not canonical")
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=handle)
                    _fsync_directory(handle)
                except FileExistsError:
                    pass
            try:
                nested = os.open(
                    part, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
                    dir_fd=handle,
                )
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.EMLINK, errno.ENOTDIR):
                    raise ContinuationCorrupt(
                        "a state path component is a symlink or not a directory",
                    ) from exc
                raise
            os.close(handle)
            handle = nested
        _assert_private_directory(handle)
        return absolute
    finally:
        os.close(handle)
