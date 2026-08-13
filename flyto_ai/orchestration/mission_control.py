# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Durable, multi-process scheduling of *missions* over arbitrary work.

This kernel is deliberately provider- and domain-neutral.  It knows only that
somebody declared a *mission* - an objective, the result they want, and the
criteria that would prove they got it - and that reaching it takes a rooted
graph of *work items*, each of which may need exclusive use of some named
*resources* while it runs.  It has no vocabulary of its own: no enumerated
project kinds, no blessed repository names, no branch on what a caller's
coordinate happens to spell.  A coding adapter, a pentest adapter, a robot
fleet and a workflow engine all submit the same shapes.

Six separations carry the design.

*Contract versus progress.*  The objective, the desired result and the
acceptance criteria are fixed at :meth:`MissionStore.create_mission` and there
is no API that edits them - not a setter, not a merge, not an "amend".  A
mission that turned out to be the wrong mission is a new mission.  Everything
that moves lives on the work items instead.

*Graph versus queue.*  Work items form a cycle-free DAG rooted at exactly one
main item.  Any side item names both the ``parent_id`` it descends from and the
``return_to_id`` it must hand control back to, and that return target has to be
an ancestor - the path home always points at the root, never sideways or down.
The queue is only an ordering over the ready items of that graph.

*Record versus authority.*  A work item marked ``dispatched`` is evidence that
a dispatch happened, never proof that its worker is still alive.  Authority is
the OS ``flock`` execution lease the dispatch handle holds: it exists while the
holding process exists, and it cannot be forged by writing a row.  Heartbeats
are observability - they move a counter and a timestamp and nothing else.  No
lease is ever stolen because a clock ran out; :meth:`MissionStore.reclaim` takes
one only when the kernel can *prove* the lease is free by taking it.

*Fence versus liveness.*  Every dispatch burns a durable, globally monotonic
fencing token.  A worker mutation carries the token it was dispatched under, so
a worker that was slow, paused or partitioned cannot write into the era that
replaced it: it gets :class:`MissionStaleFence`, a typed and stable answer, not
a silent overwrite.

*Closure versus silence.*  A work item leaves the queue as ``fixed``,
``deferred`` or ``blocked`` and in no other way.  Deferring or blocking is
allowed - pretending is not - so both demand a rationale, a risk, evidence
refs, a named owner and a revisit time.  A mission completes only with evidence
for every acceptance criterion, with every work item explicitly closed, and
with every return edge still resolving to an ancestor.

*Store versus object.*  Capacity, counters, the fencing token and the fairness
rotation live in the durable store, not on a Python object, so two processes
share one queue depth and one token sequence.  Constructing a
:class:`MissionStore` with a capacity that disagrees with the durable one is a
rejected contract, not a second opinion.

*Name versus file.*  Nothing here is opened by pathname twice.  The configured
root is walked one component at a time with ``O_NOFOLLOW``, so a symbolic link
anywhere along it - not merely at the end - is refused rather than followed, and
every file below is opened relative to the directory descriptor that walk
produced.  The database is read *through* that descriptor and materialised with
``sqlite3.deserialize``, and published by writing a complete, fsynced file and
renaming it into place.  SQLite is never handed the pathname, which is the only
version of this that is a guarantee rather than a hope: a check made before and
after ``sqlite3.connect`` cannot see a decoy that exists only during the call,
and this kernel would rather not need the check.

Everything else follows from failing closed.  Selection is deterministic - the
least recently served scope first, then the repair lane *within* that scope,
then priority, then age - so no producer can monopolise dispatch by submitting
more, and none can do it by relabelling its work as urgent either.  A dispatch
claims *all* of a work item's canonically sorted resources inside one
transaction or claims none of them, and the repair lane uses that identical
path, so a repair cannot cut in front of a conflict.  A lease becomes authority
only after the transaction that took it is durable; every earlier failure
unlocks it.  Storage is private to the account that owns it: every directory
this kernel creates is :data:`DIRECTORY_MODE` and every file
:data:`FILE_MODE`.  An existing store is *validated*, never repaired in
passing - a missing table, index, setting or counter is
:class:`MissionCorrupt`, because recreating a lost ``fence`` row would reissue
a fencing token that had already been spent.  Durable bytes that do not parse,
or that carry an unknown schema version, an unknown lane, an unknown status or
a cyclic graph, raise :class:`MissionCorrupt` instead of being guessed at.

What this module does **not** claim is worth stating exactly, because a
security property described loosely is worse than one described narrowly.  The
store directory is owner-only, so every actor able to write inside it runs as
the account this kernel runs as, and nothing here is cryptographic: an actor
willing to rewrite owner-owned files at will can construct any store it likes,
and no check made from inside that same account can tell that apart from a
legitimate backup restore.  A substituted *foreign* database is refused - the
identity in ``store.id``, read through the same verified descriptor, will not
match the one recorded inside it - but an *edited copy* of this store's own
database is not detected and is not meant to be.

What is guaranteed, against the ordinary non-adversarial cases that actually
happen - two agent processes racing, a directory moved by a tidy-up script, a
half-finished initialisation, an I/O error - is narrower and firm:

* The bytes a transaction reads are the bytes of the file it opened and
  verified, and the work it commits lands in the directory it walked to.
* Two concurrent callers never both succeed while writing into different
  directories, and never lose each other's updates.  Exclusion is held on the
  store directory itself, which the holder already has open, so there is no
  lock pathname to unlink and no second inode to acquire.
* A caller is told it succeeded only if the configured path still names the
  store it published into - checked before publication and again before the
  result is returned.  Otherwise it gets :class:`MissionDisplaced` (nothing was
  written) or :class:`MissionIndeterminate` (the rename happened, but
  durability or reachability could not be established; the operation must not
  be retried blindly, because creating a mission twice is not creating it once).
* A store is bootstrapped only when the directory holds no durable residue at
  all.  An initialisation that failed removes only what it created; one that
  died leaves evidence behind, and that evidence is corruption rather than an
  invitation to mint a second identity over the first.
* Nothing publishes over durable state it has not validated in full, so damage
  cannot be carried forward under the signature of unrelated traffic.

*Attempt versus effect.*  Every mutation that publishes carries a caller-chosen
``operation`` key, and the receipt for it is written in the same transaction as
the effect it describes.  Publication being one atomic rename, a receipt exists
exactly when its effect does - so a call interrupted by
:class:`MissionIndeterminate` is answerable rather than ambiguous.  Presenting
the same key with the same payload reconciles to the original effect, with its
original ids, timestamps and counters; a dispatch additionally hands back
authority over the *same* work item at the *same* fencing token, so an
interrupted dispatch cannot strand work that the scheduler believes is running.
The same key with a different payload is :class:`MissionOperationConflict`, not
a second opinion.  Receipts are bounded by :data:`MAX_OPERATIONS` and released
only by :meth:`MissionStore.acknowledge_operation` - never on a timer, because
a receipt nobody has collected is the only record of whether an interrupted
call took effect.

*Supported host versus every host.*  This kernel is an optional capability of
the package it ships in, not a floor under it.  It requires two primitives, and
where they are absent it says so and does nothing: every ``MissionStore``
operation - read or write - raises :class:`MissionUnsupported` before touching
the filesystem, so an unsupported host is left byte-for-byte as it was found.
There is no reduced backend, because a reduced backend would be a store that
makes these promises without keeping them.  :func:`inspect_host` reports the
answer without raising, which is what lets a service refuse work at start-up
instead of discovering the problem mid-mutation, and what lets a test suite skip
precisely the behaviour that needs the primitives while still proving the
boundary itself on every host.

Two host primitives are required rather than emulated.  Without a working
``flock`` no mutation runs at all, and without ``sqlite3.Connection.serialize``
and ``deserialize`` - CPython 3.11 and later - a store cannot be bound to the
file this kernel verified, so both raise :class:`MissionUnsupported` instead of
quietly serving a weaker version under the same promises.
:func:`inspect_host` answers both questions without touching a filesystem, so a
service can refuse work at start-up rather than at its first mutation.

Durable data is secret-free by contract.  Only validated, printable, bounded
identifiers and caller-supplied prose reach the database.  **Callers must not
pass credentials, API keys, bearer tokens, cookies or personal data in any
field.**  Snapshots go further and carry no free text at all - no objective, no
rationale, no evidence ref, no coordinate - so an observability surface cannot
leak what a caller put in a mission body.
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import sqlite3
import stat
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

try:  # pragma: no cover - import shape differs only on non-POSIX hosts
    import fcntl
except ImportError:  # pragma: no cover - Windows and friends
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "DEFAULT_QUEUE_CAPACITY",
    "DEFAULT_SNAPSHOT_LIMIT",
    "DIRECTORY_MODE",
    "DISPOSITIONS",
    "DISPOSITION_BLOCKED",
    "DISPOSITION_DEFERRED",
    "DISPOSITION_FIXED",
    "FILE_MODE",
    "LANES",
    "LANE_PRIMARY",
    "LANE_REPAIR",
    "DEPENDENCY_SATISFYING_DISPOSITIONS",
    "MAX_ACCEPTANCE_CRITERIA",
    "MAX_DAG_DEPTH",
    "MAX_DATABASE_BYTES",
    "MAX_DEPENDENCIES",
    "MAX_DISPATCH_CANDIDATES",
    "MAX_EVIDENCE_REFS",
    "MAX_ROTATION_SCOPES",
    "MAX_FIELD_CHARS",
    "MAX_OPERATIONS",
    "MAX_PRIORITY",
    "MAX_QUEUE_CAPACITY",
    "MAX_RESOURCES",
    "MAX_REVISIT_HORIZON_SECONDS",
    "MAX_SEQUENCE",
    "MAX_SNAPSHOT_ITEMS",
    "MAX_TEXT_CHARS",
    "MAX_WORK_ITEMS_PER_MISSION",
    "OPERATION_ABANDON_WORK_ITEM",
    "OPERATION_CLOSE_WORK_ITEM",
    "OPERATION_COMPLETE_MISSION",
    "OPERATION_CREATE_MISSION",
    "OPERATION_DISPATCH",
    "OPERATION_KINDS",
    "OPERATION_RECLAIM_WORK_ITEM",
    "OPERATION_SUBMIT_WORK_ITEM",
    "MISSION_COMPLETED",
    "MISSION_OPEN",
    "SCHEMA_VERSION",
    "STATUS_CLOSED",
    "STATUS_DISPATCHED",
    "STATUS_READY",
    "AcceptanceCriterion",
    "Closure",
    "DispatchHandle",
    "Mission",
    "MissionCapacityExceeded",
    "MissionConflict",
    "MissionCorrupt",
    "MissionDisplaced",
    "MissionError",
    "MissionHostFailure",
    "MissionIndeterminate",
    "MissionMetrics",
    "MissionOperationConflict",
    "MissionOperationSettled",
    "MissionRejected",
    "MissionResource",
    "MissionSnapshot",
    "MissionStaleFence",
    "MissionStore",
    "MissionSummary",
    "MissionUnauthorized",
    "HostCapabilities",
    "MissionUnsupported",
    "inspect_host",
    "WorkCoordinates",
    "WorkItem",
    "WorkItemSummary",
]

#: Durable layout version.  A store stamped with anything else is not ours.
SCHEMA_VERSION = 1

#: Mode for every directory this kernel creates or adopts: owner-only.
DIRECTORY_MODE = 0o700
#: Mode for every file this kernel creates: owner-only.
FILE_MODE = 0o600

#: Bounds.  Every one of them is enforced on the way in, so a durable record
#: can never be larger than the reader that will have to parse it back.
MAX_FIELD_CHARS = 256
MAX_TEXT_CHARS = 2000
MAX_ACCEPTANCE_CRITERIA = 32
MAX_EVIDENCE_REFS = 16
MAX_RESOURCES = 16
MAX_DEPENDENCIES = 16
MAX_PRIORITY = 1000
#: Upper bound for every durable counter, sequence and fencing token.  Chosen
#: so a value stays exactly representable after a round trip through anything
#: that speaks JSON numbers, which a receipt or an audit export eventually will.
MAX_SEQUENCE = 2**53
MAX_DAG_DEPTH = 64
#: How many distinct scopes the durable fairness rotation will carry.  A store
#: that has accumulated more has stopped being a bounded scheduling surface.
MAX_ROTATION_SCOPES = 10_000
MAX_WORK_ITEMS_PER_MISSION = 1024
#: The largest queue this implementation will accept, and the reason for it.
#:
#: Publication rewrites the whole database, so cost grows with the store rather
#: than with the change.  A thousand work items was measured end to end - about
#: 27s of submissions, with the final publication taking 47ms - so a thousand is
#: the largest size for which this design has evidence rather than optimism.
#: Declaring a hundred thousand would be advertising a service envelope nobody
#: has run, on an O(n) write.  Raising it is an architecture decision (an
#: incremental durability format), not a constant.
MAX_QUEUE_CAPACITY = 1_000
DEFAULT_QUEUE_CAPACITY = 256

#: How many operation receipts a store retains.  Bounded like everything else;
#: see :meth:`MissionStore.acknowledge_operation` for how they are released.
MAX_OPERATIONS = 512
#: How far ahead of closure a revisit time may be pointed: one year.
MAX_REVISIT_HORIZON_SECONDS = 365 * 24 * 60 * 60
#: How many ready work items one dispatch will consider before giving up.
#: Reaching it is counted in :attr:`MissionMetrics.scan_truncations` rather
#: than being swallowed - a silent cap reads as "nothing was runnable".
MAX_DISPATCH_CANDIDATES = 64
#: Snapshot bounds.  Observability never streams a whole store.
MAX_SNAPSHOT_ITEMS = 200
DEFAULT_SNAPSHOT_LIMIT = 50

#: Which queue a work item waits in.  Repair work is preferred over primary
#: work, and is otherwise identical - including in how it claims resources.
Lane = Literal["repair", "primary"]
LANE_REPAIR: Lane = "repair"
LANE_PRIMARY: Lane = "primary"
LANES: Tuple[Lane, ...] = (LANE_REPAIR, LANE_PRIMARY)

#: The only three ways a work item may leave the queue.
Disposition = Literal["fixed", "deferred", "blocked"]
DISPOSITION_FIXED: Disposition = "fixed"
DISPOSITION_DEFERRED: Disposition = "deferred"
DISPOSITION_BLOCKED: Disposition = "blocked"
DISPOSITIONS: Tuple[Disposition, ...] = (
    DISPOSITION_FIXED,
    DISPOSITION_DEFERRED,
    DISPOSITION_BLOCKED,
)
#: Closing as deferred or blocked is legitimate; closing quietly is not.
_ACCOUNTED_DISPOSITIONS: Tuple[Disposition, ...] = (
    DISPOSITION_DEFERRED,
    DISPOSITION_BLOCKED,
)
#: The dispositions that let something which *depended* on this work run.
#:
#: Only ``fixed`` does.  A dependency that was deferred or blocked did not
#: deliver what its dependents were waiting for, so a dependent never becomes
#: runnable on the strength of it - it is closed through
#: :meth:`MissionStore.abandon_unrunnable_work_item` with the same full
#: accounting instead.  That is also why a dependent may not itself close as
#: ``fixed``: claiming to have fixed work that rests on a blocked dependency is
#: precisely the fiction this kernel exists to make unrepresentable.
DEPENDENCY_SATISFYING_DISPOSITIONS: Tuple[Disposition, ...] = (DISPOSITION_FIXED,)

WorkStatus = Literal["ready", "dispatched", "closed"]
STATUS_READY: WorkStatus = "ready"
STATUS_DISPATCHED: WorkStatus = "dispatched"
STATUS_CLOSED: WorkStatus = "closed"
_STATUSES: Tuple[WorkStatus, ...] = (STATUS_READY, STATUS_DISPATCHED, STATUS_CLOSED)

MissionStatus = Literal["open", "completed"]
MISSION_OPEN: MissionStatus = "open"
MISSION_COMPLETED: MissionStatus = "completed"
_MISSION_STATUSES: Tuple[MissionStatus, ...] = (MISSION_OPEN, MISSION_COMPLETED)

#: The whole database is read into memory and published as one atomic file,
#: so it has to be bounded like anything else this kernel reads.
MAX_DATABASE_BYTES = 64 * 1024 * 1024

_DOMAIN = b"flyto-ai/mission-control/1"
_DIGEST_CHARS = 64
_HEX_DIGITS = frozenset("0123456789abcdef")
_STORE_DIRNAME = "mission-control"
_DB_NAME = "missions.db"
_LEASE_DIRNAME = "leases"
_STAGED_NAME = ".missions.db.staged"
_STORE_ID_NAME = "store.id"

#: Exactly the durable settings a v1 store carries.  Not "at least".
#: The mutations that publish, and therefore the ones that need a receipt.
OPERATION_CREATE_MISSION = "create-mission"
OPERATION_SUBMIT_WORK_ITEM = "submit-work-item"
OPERATION_DISPATCH = "dispatch"
OPERATION_CLOSE_WORK_ITEM = "close-work-item"
OPERATION_ABANDON_WORK_ITEM = "abandon-work-item"
OPERATION_COMPLETE_MISSION = "complete-mission"
OPERATION_RECLAIM_WORK_ITEM = "reclaim-work-item"
OPERATION_KINDS: Tuple[str, ...] = (
    OPERATION_CREATE_MISSION,
    OPERATION_SUBMIT_WORK_ITEM,
    OPERATION_DISPATCH,
    OPERATION_CLOSE_WORK_ITEM,
    OPERATION_ABANDON_WORK_ITEM,
    OPERATION_COMPLETE_MISSION,
    OPERATION_RECLAIM_WORK_ITEM,
)
#: Which result field each kind records, and which table it must point into.
_OPERATION_SUBJECT = {
    OPERATION_CREATE_MISSION: "mission_id",
    OPERATION_SUBMIT_WORK_ITEM: "work_item_id",
    OPERATION_DISPATCH: "work_item_id",
    OPERATION_CLOSE_WORK_ITEM: "work_item_id",
    OPERATION_ABANDON_WORK_ITEM: "work_item_id",
    OPERATION_COMPLETE_MISSION: "mission_id",
    OPERATION_RECLAIM_WORK_ITEM: "work_item_id",
}

_META_KEYS = frozenset(
    {"schema_version", "queue_capacity", "store_id", "dispatch_cursor"}
)

_COUNTERS = (
    "fence",
    "mission_seq",
    "work_seq",
    "submit_seq",
    "dispatch_seq",
    "dispatches",
    "capacity_rejects",
    "conflicts",
    "stale_fence_rejects",
    "scan_truncations",
    "cursor_sweeps",
    "operation_seq",
    "lease_rejects",
    "abandoned",
    "missions_created",
    "missions_completed",
    "closed_fixed",
    "closed_deferred",
    "closed_blocked",
    "latency_ms_total",
    "latency_ms_max",
    "latency_samples",
)
_CLOSED_COUNTER = {
    DISPOSITION_FIXED: "closed_fixed",
    DISPOSITION_DEFERRED: "closed_deferred",
    DISPOSITION_BLOCKED: "closed_blocked",
}

_NO_INTERPROCESS_LOCK = (
    "the host has no supported inter-process lock, so this mission store cannot "
    "mutate state safely"
)
_NOT_A_REAL_DIRECTORY = (
    "the mission store path is a symbolic link or not a directory, so it is "
    "refused rather than followed outside the configured store"
)
_NOT_A_REAL_FILE = (
    "the mission store file is a symbolic link or not a regular file, so it is "
    "refused rather than followed outside the configured store"
)
_NO_PRIVATE_DIRECTORY = (
    "the mission store directory could not be made private, so nothing was written"
)
_NO_PRIVATE_FILE = "the mission store file could not be made private, so nothing was written"


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class MissionError(Exception):
    """Base class for every failure this kernel raises."""


class MissionRejected(MissionError):
    """A caller-supplied value or contract is outside the accepted bounds."""


class MissionCapacityExceeded(MissionError):
    """The durable, store-wide queue is full.  Back off; nothing was queued."""


class MissionConflict(MissionError):
    """A resource, a lease or a lifecycle transition is held by somebody else."""


class MissionStaleFence(MissionError):
    """The caller presented a fencing token that a later dispatch replaced."""


class MissionUnauthorized(MissionError):
    """The caller does not hold live execution authority for this work item.

    A fencing token proves *when* a caller was dispatched; it does not prove
    that it is still the one running.  Authority is the open, ``flock``-held
    lease descriptor this process took at dispatch, and nothing that is visible
    in a snapshot - an id, a fence, a worker name - can stand in for it.
    """


class MissionOperationConflict(MissionError):
    """One operation key was reused for a different operation.

    A key identifies *one logical mutation*.  Presenting it again with a
    different kind, or with a different payload, is not a retry - it is two
    different intentions wearing one name, and answering it with either effect
    would be a guess.  Reusing a key deliberately to "update" an operation is
    the same mistake wearing a plan.
    """


class MissionOperationSettled(MissionError):
    """The operation happened, and its effect can no longer be handed back.

    Raised when a dispatch is retried after its era ended - the work was closed,
    or requeued and taken by somebody else.  The recorded outcome is available on
    :attr:`result` so a caller can reconcile rather than guess, but no authority
    is granted, because there is none left to grant.
    """

    def __init__(self, message: str, result: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.result = dict(result)


class MissionDisplaced(MissionError):
    """The configured path stopped naming the store this operation was bound to.

    Raised *before* anything was published, so nothing was written: the caller's
    work was refused rather than committed into a store that is no longer the
    one the pathname resolves to.  Two callers must never both be told they
    succeeded while writing into different directories that share a name.
    """


class MissionIndeterminate(MissionError):
    """The mutation may or may not have taken effect, and nothing may assume.

    Reserved for the window after the atomic rename: the new bytes can already
    be visible to a reader, but this call could not establish that they will
    survive a crash, or could no longer prove the directory it published into is
    the one the configured path names.  Success is forbidden and so is an
    automatic retry - the operation is not known to be idempotent, and repeating
    a mission or a dispatch that did land would duplicate real work.
    """


class MissionHostFailure(MissionError):
    """A host primitive failed in a way this kernel refuses to interpret.

    Reserved for the answers that are neither success nor a clean "somebody
    else holds it": ``EBADF``, ``EIO``, ``ENOTSUP``, a permission error, a
    descriptor that stopped resolving.  Guessing that any of those means the
    resource is free is exactly how two workers end up holding one lease.
    """


class MissionCorrupt(MissionError):
    """Durable state could not be trusted, so the operation failed closed."""


class MissionUnsupported(MissionError):
    """The host cannot supply a primitive this kernel's contract rests on."""


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def _check_token(value: object, label: str) -> str:
    """Validate one durable identifier.

    Accepts a bounded, printable, whitespace-free string.  That rules out
    control characters, tabs and newlines, and any other non-printable code
    point that could smuggle structure into a record, a log line or a path.  It
    cannot, and does not try to, tell a safe identifier from a secret - see the
    module docstring.
    """

    if not isinstance(value, str):
        raise MissionRejected(f"{label} must be a string")
    if not value:
        raise MissionRejected(f"{label} must not be empty")
    if len(value) > MAX_FIELD_CHARS:
        raise MissionRejected(f"{label} exceeds {MAX_FIELD_CHARS} characters")
    if not value.isprintable():
        raise MissionRejected(f"{label} contains a non-printable character")
    if any(char.isspace() for char in value):
        raise MissionRejected(f"{label} contains whitespace")
    return value


def _check_text(value: object, label: str) -> str:
    """Validate one durable prose field: bounded, printable, single-line.

    Interior spaces are fine - this is prose - but control characters are not,
    so a stored rationale can never inject a line into anything that renders it.
    """

    if not isinstance(value, str):
        raise MissionRejected(f"{label} must be a string")
    stripped = value.strip()
    if not stripped:
        raise MissionRejected(f"{label} must not be empty")
    if len(value) > MAX_TEXT_CHARS:
        raise MissionRejected(f"{label} exceeds {MAX_TEXT_CHARS} characters")
    if not value.isprintable():
        raise MissionRejected(f"{label} contains a non-printable character")
    return stripped


def _check_int(value: object, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MissionRejected(f"{label} must be an integer")
    if value < low or value > high:
        raise MissionRejected(f"{label} is out of bounds")
    return value


#: Ids this kernel mints are exactly a prefix and twelve decimal digits.  They
#: are checked on the way *out* of storage as well as on the way in, because an
#: id is the join key for lineage, dependencies, claims and leases: an id that
#: is merely "a token" would let damaged state address rows this kernel never
#: wrote.
_ID_DIGITS = 12
_MISSION_PREFIX = "m-"
_WORK_PREFIX = "w-"


def _check_identifier(value: object, label: str, prefix: str) -> str:
    token = _check_token(value, label)
    body = token[len(prefix) :]
    if not token.startswith(prefix) or len(body) != _ID_DIGITS or not body.isdigit():
        raise MissionRejected(f"{label} is not a well-formed identifier")
    return token


def _check_bool(value: object, label: str) -> bool:
    """Insist on a real ``bool``.  Truthiness is not a contract."""

    if not isinstance(value, bool):
        raise MissionRejected(f"{label} must be a bool")
    return value


def _digest(*parts: str) -> str:
    """Length-prefixed digest so distinct tuples cannot collide by joining."""

    accumulator = hashlib.sha256()
    accumulator.update(_DOMAIN)
    for part in parts:
        raw = part.encode("utf-8")
        accumulator.update(b"\n")
        accumulator.update(str(len(raw)).encode("ascii"))
        accumulator.update(b":")
        accumulator.update(raw)
    return accumulator.hexdigest()


# --------------------------------------------------------------------------
# value types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MissionResource:
    """An arbitrary bounded resource coordinate: namespace, kind, identity.

    A work item may not run while another running work item holds any resource
    it names.  What a resource *means* - a git worktree, a lab bench, a robot
    arm, a target host, a database - is entirely the caller's business.
    """

    namespace: str
    kind: str
    identity: str

    def __post_init__(self) -> None:
        _check_token(self.namespace, "resource namespace")
        _check_token(self.kind, "resource kind")
        _check_token(self.identity, "resource identity")

    @property
    def digest(self) -> str:
        """Content address of this resource; the only form claimed on disk."""

        return _digest(self.namespace, self.kind, self.identity)

    @property
    def sort_key(self) -> Tuple[str, str, str]:
        """Canonical order, so every dispatcher claims in the same sequence."""

        return (self.namespace, self.kind, self.identity)


@dataclass(frozen=True)
class WorkCoordinates:
    """Where a work item does its work, in the caller's own vocabulary.

    ``project`` / ``repository`` / ``location`` are three opaque bounded
    tokens.  A coding adapter reads them as a product, a git repository and a
    path; a robot fleet reads them as a site, a cell and a fixture.  This kernel
    reads them as three strings and never branches on their contents.
    """

    project: str
    repository: str
    location: str

    def __post_init__(self) -> None:
        _check_token(self.project, "coordinate project")
        _check_token(self.repository, "coordinate repository")
        _check_token(self.location, "coordinate location")


@dataclass(frozen=True)
class AcceptanceCriterion:
    """One immutable, identified statement of what would prove the mission done."""

    id: str
    statement: str

    def __post_init__(self) -> None:
        _check_token(self.id, "acceptance criterion id")
        object.__setattr__(self, "statement", _check_text(self.statement, "acceptance statement"))


@dataclass(frozen=True)
class Closure:
    """Why a work item left the queue, and what it left behind.

    ``fixed`` is the only disposition that needs nothing further: the work was
    done.  ``deferred`` and ``blocked`` are legitimate outcomes and are exactly
    the ones a silent skip would hide behind, so both require the whole
    accounting - a rationale, the risk of leaving it, evidence refs, a named
    owner and a revisit time.  ``revisit_at`` is epoch seconds and is
    meaningless for ``fixed``, which therefore refuses it.

    The clock is deliberately not consulted here: this type is pure, and
    :meth:`DispatchHandle.close` is what checks that a revisit time
    points at the future.
    """

    disposition: Disposition
    rationale: Optional[str] = None
    risk: Optional[str] = None
    evidence_refs: Tuple[str, ...] = ()
    owner: Optional[str] = None
    revisit_at: Optional[int] = None

    def __post_init__(self) -> None:
        if self.disposition not in DISPOSITIONS:
            raise MissionRejected(
                "a work item may close only as " + ", ".join(DISPOSITIONS)
            )
        refs = tuple(self.evidence_refs or ())
        if len(refs) > MAX_EVIDENCE_REFS:
            raise MissionRejected(f"evidence refs exceed {MAX_EVIDENCE_REFS} entries")
        for ref in refs:
            _check_token(ref, "evidence ref")
        if len(set(refs)) != len(refs):
            raise MissionRejected("evidence refs must be distinct")
        object.__setattr__(self, "evidence_refs", refs)

        if self.rationale is not None:
            object.__setattr__(self, "rationale", _check_text(self.rationale, "rationale"))
        if self.risk is not None:
            object.__setattr__(self, "risk", _check_text(self.risk, "risk"))
        if self.owner is not None:
            _check_token(self.owner, "closure owner")
        if self.revisit_at is not None:
            _check_int(self.revisit_at, "revisit_at", 1, 2**63 - 1)

        if self.disposition == DISPOSITION_FIXED:
            if self.revisit_at is not None:
                raise MissionRejected("a fixed work item has nothing to revisit")
            return

        missing = [
            name
            for name, value in (
                ("rationale", self.rationale),
                ("risk", self.risk),
                ("owner", self.owner),
                ("revisit_at", self.revisit_at),
            )
            if value is None
        ]
        if not refs:
            missing.append("evidence_refs")
        if missing:
            raise MissionRejected(
                f"a {self.disposition} work item requires " + ", ".join(sorted(missing))
            )

    def as_payload(self) -> Dict[str, Any]:
        return {
            "disposition": self.disposition,
            "rationale": self.rationale,
            "risk": self.risk,
            "evidence_refs": list(self.evidence_refs),
            "owner": self.owner,
            "revisit_at": self.revisit_at,
        }


@dataclass(frozen=True)
class Mission:
    """One immutable contract, plus the mutable status of reaching it.

    ``objective``, ``desired_result`` and ``acceptance_criteria`` are settled at
    creation.  The dataclass is frozen and the store exposes no operation that
    rewrites them, so the only honest way to change what is being attempted is
    to create a different mission.
    """

    mission_id: str
    scope: str
    objective: str
    desired_result: str
    acceptance_criteria: Tuple[AcceptanceCriterion, ...]
    status: MissionStatus
    created_at: float
    completed_at: Optional[float] = None
    acceptance_evidence: Tuple[Tuple[str, str], ...] = ()

    @property
    def criteria_ids(self) -> Tuple[str, ...]:
        return tuple(criterion.id for criterion in self.acceptance_criteria)


@dataclass(frozen=True)
class WorkItem:
    """One node of a mission's rooted, cycle-free work graph."""

    work_item_id: str
    mission_id: str
    scope: str
    lane: Lane
    priority: int
    is_root: bool
    parent_id: Optional[str]
    return_to_id: Optional[str]
    coordinates: WorkCoordinates
    resources: Tuple[MissionResource, ...]
    depends_on_ids: Tuple[str, ...]
    status: WorkStatus
    fence: int
    attempts: int
    submit_seq: int
    submitted_at: float
    dispatched_at: Optional[float] = None
    worker: Optional[str] = None
    heartbeats: int = 0
    last_heartbeat_at: Optional[float] = None
    closure: Optional[Closure] = None

    @property
    def disposition(self) -> Optional[Disposition]:
        return self.closure.disposition if self.closure is not None else None


@dataclass(frozen=True)
class WorkItemSummary:
    """A secret-free projection of one work item, for observability only.

    Carries no prose and no coordinates: an operator dashboard should be able
    to render a whole store without ever holding what a caller wrote into a
    rationale, an evidence ref or a repository path.
    """

    work_item_id: str
    mission_id: str
    scope: str
    lane: Lane
    priority: int
    is_root: bool
    parent_id: Optional[str]
    return_to_id: Optional[str]
    status: WorkStatus
    disposition: Optional[Disposition]
    fence: int
    attempts: int
    heartbeats: int
    resource_count: int
    dependency_count: int


@dataclass(frozen=True)
class MissionSummary:
    """A secret-free projection of one mission: identity, shape and progress."""

    mission_id: str
    scope: str
    status: MissionStatus
    criteria_ids: Tuple[str, ...]
    work_items: int
    closed_work_items: int


@dataclass(frozen=True)
class MissionMetrics:
    """Bounded, secret-free counters for the whole store."""

    queue_capacity: int
    queue_depth: int
    dispatched: int
    closed_fixed: int
    closed_deferred: int
    closed_blocked: int
    missions_open: int
    missions_completed: int
    dispatches: int
    capacity_rejects: int
    conflicts: int
    stale_fence_rejects: int
    scan_truncations: int
    cursor_sweeps: int
    operations_retained: int
    lease_rejects: int
    abandoned: int
    submit_to_dispatch_ms_total: int
    submit_to_dispatch_ms_max: int
    submit_to_dispatch_samples: int

    @property
    def submit_to_dispatch_ms_mean(self) -> float:
        if not self.submit_to_dispatch_samples:
            return 0.0
        return self.submit_to_dispatch_ms_total / self.submit_to_dispatch_samples


@dataclass(frozen=True)
class MissionSnapshot:
    """A bounded read-only view.  ``truncated`` is never left implicit."""

    missions: Tuple[MissionSummary, ...]
    work_items: Tuple[WorkItemSummary, ...]
    truncated: bool
    metrics: MissionMetrics


# --------------------------------------------------------------------------
# host primitives
# --------------------------------------------------------------------------

_POSIX_MODES = os.name == "posix"


class _LocalLock:
    """One process-local mutex plus a count of who still needs it alive."""

    __slots__ = ("lock", "holders")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.holders = 0


#: Process-local mutexes, keyed by lock path and reference-counted by the
#: threads holding or waiting on them, so the registry stays proportional to
#: live threads rather than to every store this process has ever opened.
_LOCAL_LOCKS: Dict[str, _LocalLock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


def _interprocess_locking_supported() -> bool:
    """Whether this host can exclude other *processes*, not merely other threads.

    Probed through the module attribute rather than cached at import time, so a
    host that has to be treated as unsupported can be represented faithfully in
    a test.
    """

    names = ("flock", "LOCK_EX", "LOCK_NB", "LOCK_UN")
    return all(getattr(fcntl, name, None) is not None for name in names)


def _require_interprocess_lock() -> None:
    if not _interprocess_locking_supported():
        raise MissionUnsupported(_NO_INTERPROCESS_LOCK)


@dataclass(frozen=True)
class HostCapabilities:
    """What this host can and cannot supply, answered without touching a store.

    A service that accepts work it cannot durably schedule has already failed;
    this is how it finds out first.  Nothing here creates a directory, opens a
    store, or changes a mode - it is a question about the interpreter and the
    platform, not about any particular store.
    """

    interprocess_locking: bool
    database_binding: bool

    @property
    def supported(self) -> bool:
        return self.interprocess_locking and self.database_binding

    @property
    def missing(self) -> Tuple[str, ...]:
        absent = []
        if not self.interprocess_locking:
            absent.append("interprocess-locking")
        if not self.database_binding:
            absent.append("database-binding")
        return tuple(absent)


def _require_supported_host() -> None:
    """Refuse, before anything is opened, on a host that cannot keep the contract.

    Placement is the whole point.  Discovering the missing primitive part-way
    through - once the store directory has been created and moded - leaves a
    directory that looks like a store on a host that can never serve one, and
    the next operator to find it has to work out whether it holds anything.  A
    capability this kernel does not have is answered before it has touched the
    filesystem at all, so an unsupported host is left exactly as it was found.
    """

    capabilities = inspect_host()
    if capabilities.supported:
        return
    raise MissionUnsupported(
        "this host cannot support a mission store: "
        + ", ".join(capabilities.missing)
        + "; nothing was created.  There is deliberately no lesser backend to"
        " fall back to - the guarantees this store makes are the reason it"
        " requires these primitives"
    )


def inspect_host() -> HostCapabilities:
    """Report the primitives this kernel refuses to run without.

    Read-only and side-effect free by construction, so a service can call it
    during start-up - before it advertises itself, and before any store exists -
    and reject work up front rather than at the first mutation.
    """

    return HostCapabilities(
        interprocess_locking=_interprocess_locking_supported(),
        database_binding=_database_binding_supported(),
    )


def _database_binding_supported() -> bool:
    """Whether a database can be opened from *bytes* rather than from a name.

    This is the primitive the storage boundary rests on, so it is probed rather
    than assumed - and probed through a function, so a host that has to be
    treated as unsupported can be represented faithfully in a test.  It arrived
    in CPython 3.11; on anything older this kernel refuses to run rather than
    reintroducing the pathname it exists to avoid.
    """

    return all(
        callable(getattr(sqlite3.Connection, name, None))
        for name in ("serialize", "deserialize")
    )


_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
_NOFOLLOW_DIRECTORY_FLAGS = _DIRECTORY_FLAGS | getattr(os, "O_NOFOLLOW", 0)


def _open_component(parent: int, name: str, *, create: bool) -> Tuple[int, bool]:
    """Open one directory component *relative to its parent*, never through a link.

    ``O_NOFOLLOW`` makes a symbolic-link component fail rather than resolve, and
    ``dir_fd`` means the lookup happens in the directory this kernel already
    holds open rather than by re-walking a pathname a bystander could have
    rearranged underneath it.

    Returns the descriptor and whether *this call* created the directory, which
    is the only thing that entitles a caller to set its mode.
    """

    try:
        return os.open(name, _NOFOLLOW_DIRECTORY_FLAGS, dir_fd=parent), False
    except FileNotFoundError:
        if not create:
            raise
    except OSError as exc:
        raise MissionError(f"{_NOT_A_REAL_DIRECTORY}: {name}: {exc}") from exc
    created = True
    try:
        os.mkdir(name, DIRECTORY_MODE, dir_fd=parent)
    except FileExistsError:  # pragma: no cover - lost a race with another store
        created = False
    except OSError as exc:
        raise MissionError(f"the mission store root is unusable: {name}: {exc}") from exc
    try:
        return os.open(name, _NOFOLLOW_DIRECTORY_FLAGS, dir_fd=parent), created
    except OSError as exc:
        raise MissionError(f"{_NOT_A_REAL_DIRECTORY}: {name}: {exc}") from exc


def _open_configured_root(path: Path, *, create: bool) -> int:
    """Walk the configured root one component at a time, refusing every link.

    Validating only the final component is not enough, and this is the whole
    reason the walk exists: ``mkdir(parents=True)`` and every pathname-based
    call after it resolve symbolic links in *ancestor* directories silently, so
    a configured root of ``alias/nested`` where ``alias`` is a link would be
    established, chmodded and written wherever the link led - outside the
    directory the caller named, with nothing having been refused.

    Nothing here re-modes a directory it did not create.  The ancestors belong
    to the caller, not to this kernel; they are checked for identity, and only
    the store directory below them has its mode enforced.
    """

    parts = path.parts
    if not parts or not path.is_absolute():  # pragma: no cover - always absolute
        raise MissionError("the mission store root must be an absolute path")
    try:
        descriptor = os.open(parts[0], _DIRECTORY_FLAGS)
    except OSError as exc:
        raise MissionError(f"the mission store root is unusable: {exc}") from exc
    for name in parts[1:]:
        try:
            child, _ = _open_component(descriptor, name, create=create)
        except BaseException:
            # ``FileNotFoundError`` travels on deliberately: a component that is
            # simply not there means there is no store here yet, which a
            # read-only caller must be able to tell apart from a component that
            # is there and is not safe to walk through.
            os.close(descriptor)
            raise
        os.close(descriptor)
        descriptor = child
    return descriptor


def _settle_mode(descriptor: int, name: str, mode: int, *, created: bool) -> None:
    """Give a path this kernel just created its exact mode, or refuse an old one.

    Two different situations that look alike and are not.  A path *this call*
    created may have been masked by the umask, so its mode is set and verified
    here - that is completing our own work.  A path that was already there is
    somebody else's decision, and silently re-moding it is a write performed by
    an operation that may well have been a read, on an artifact whose history
    this kernel does not know.

    For an existing path the property checked is therefore the one that actually
    matters - nobody outside this account can reach it - and not "the bits are
    the ones we would have chosen".  A store deliberately made read-only (0500
    and 0400: an audit copy, a read-only mount, an observer) satisfies the
    contract completely and stays usable.  Anything readable or writable by
    group or other does not, and is refused with its mode exactly as it was
    found.
    """

    if not _POSIX_MODES:  # pragma: no cover - POSIX mode bits are meaningless here
        return
    current = stat.S_IMODE(os.fstat(descriptor).st_mode)
    if not created:
        if current & 0o077:
            raise MissionCorrupt(
                f"{name} is mode {current:04o}, which grants access outside this"
                " account; a mission store refuses such state rather than silently"
                " re-moding a file it did not create"
            )
        return
    if current == mode:
        return
    try:
        os.fchmod(descriptor, mode)
    except OSError as exc:
        raise MissionError(f"{_NO_PRIVATE_FILE}: {name}: {exc}") from exc
    if stat.S_IMODE(os.fstat(descriptor).st_mode) != mode:
        raise MissionError(f"{_NO_PRIVATE_FILE}: {name} did not keep the mode")


def _secure_child_directory(parent: int, name: str, *, create: bool) -> int:
    """Open or create one owner-only directory inside an already-held directory."""

    descriptor, created = _open_component(parent, name, create=create)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):  # pragma: no cover
            raise MissionError(f"{_NOT_A_REAL_DIRECTORY}: {name}")
        _settle_mode(descriptor, name, DIRECTORY_MODE, created=created)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _secure_open_at(parent: int, name: str, *, create: bool) -> int:
    """Open one owner-only regular file inside an already-held directory.

    ``create=False`` is a *read*: the descriptor is ``O_RDONLY`` and no mode is
    ever changed through it.  A read that quietly repaired permissions would be
    a write that the caller did not ask for and cannot see, and it would make a
    genuinely read-only store - a mount, an observer, an auditor's copy -
    impossible to use at all.
    """

    common = getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    created = False
    descriptor = None
    if create:
        # ``O_EXCL`` is the only way to *know* the file is ours rather than to
        # guess from its size or link count, and knowing is what decides whether
        # this call is entitled to set a mode at all.
        try:
            descriptor = os.open(
                name, common | os.O_RDWR | os.O_CREAT | os.O_EXCL, FILE_MODE, dir_fd=parent
            )
            created = True
        except FileExistsError:
            descriptor = None
        except OSError as exc:
            raise MissionCorrupt(f"{_NOT_A_REAL_FILE}: {name}: {exc}") from exc
    if descriptor is None:
        try:
            descriptor = os.open(
                name, common | (os.O_RDWR if create else os.O_RDONLY), dir_fd=parent
            )
        except FileNotFoundError:
            raise
        except OSError as exc:
            # ELOOP for a symbolic link, ENOTDIR, EACCES: a durable path that is
            # not the plain file this kernel wrote is damaged state.
            raise MissionCorrupt(f"{_NOT_A_REAL_FILE}: {name}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise MissionCorrupt(f"{_NOT_A_REAL_FILE}: {name}")
        _settle_mode(descriptor, name, FILE_MODE, created=created)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_all(descriptor: int, limit: int) -> bytes:
    """Read a whole file *through the descriptor*, one byte past the bound."""

    chunks = []
    remaining = limit + 1
    while remaining > 0:
        chunk = os.read(descriptor, min(remaining, 1 << 20))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@contextmanager
def _local_lock(key: str) -> Iterator[None]:
    """Take the process-local mutex for one key, and retire it when unused."""

    with _LOCAL_LOCKS_GUARD:
        entry = _LOCAL_LOCKS.get(key)
        if entry is None:
            entry = _LocalLock()
            _LOCAL_LOCKS[key] = entry
        entry.holders += 1
    try:
        with entry.lock:
            yield
    finally:
        with _LOCAL_LOCKS_GUARD:
            entry.holders -= 1
            if entry.holders <= 0 and _LOCAL_LOCKS.get(key) is entry:
                del _LOCAL_LOCKS[key]


#: ``flock`` answers that mean "this host cannot lock this object at all",
#: as opposed to "somebody else holds it" or "something went wrong".
_UNLOCKABLE = frozenset(
    value
    for value in (
        getattr(errno, name, None)
        for name in ("ENOTSUP", "EOPNOTSUPP", "EINVAL", "ENOLCK")
    )
    if value is not None
)


@contextmanager
def _exclusive_directory(store: "_Store") -> Iterator[None]:
    """Serialize the store on the *directory itself*, not on a file inside it.

    A lock file is a pathname, and a pathname can be unlinked and recreated by
    anybody who can write the directory.  Two processes then lock two different
    inodes, both believe they hold the store, both read the same snapshot and
    the second publication silently discards the first - a lost update that no
    amount of care inside the transaction can prevent, because the mutual
    exclusion never happened.

    The store directory has no such weakness *for the caller that is already
    holding it open*: this kernel walked to it and holds a descriptor, and that
    descriptor keeps naming the same inode however the pathname is rearranged.
    Locking it therefore locks the thing the transaction is actually bound to.
    Replacing the directory at the pathname is still possible - it is answered
    at the publication boundary by :meth:`MissionStore._verify_configured_path`,
    not here.
    """

    _require_interprocess_lock()
    with _local_lock(f"{store.identity.dev}:{store.identity.ino}"):
        try:
            fcntl.flock(store.fd, fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno in _UNLOCKABLE:
                raise MissionUnsupported(
                    f"{_NO_INTERPROCESS_LOCK}: the store directory cannot be locked: {exc}"
                ) from exc
            raise MissionHostFailure(
                f"the mission store directory could not be locked: {exc}"
            ) from exc
        try:
            yield
        finally:
            try:
                fcntl.flock(store.fd, fcntl.LOCK_UN)
            except OSError:  # pragma: no cover - unlock of a lock never taken
                pass


#: The only two errnos that mean "somebody else holds this lock".  Everything
#: else ``flock`` can answer is a host failure, and treating a host failure as
#: "free" is how two workers end up holding one lease.
_WOULD_BLOCK = frozenset(
    value
    for value in (getattr(errno, name, None) for name in ("EAGAIN", "EWOULDBLOCK"))
    if value is not None
)


def _try_lease(lease_dir_fd: int, name: str) -> Optional[int]:
    """Take a non-blocking exclusive ``flock``, or report that somebody holds it.

    ``flock`` locks the open file *description*, so two descriptors conflict even
    inside one process.  That is what makes the lease honest evidence of a live
    holder rather than a per-process convention, and it is why
    :meth:`MissionStore.reclaim` can prove a lease is free instead of guessing
    from a clock.

    ``None`` means, and only ever means, ``EAGAIN``/``EWOULDBLOCK``: a live
    holder.  ``EBADF``, ``EIO``, ``ENOTSUP``, ``EACCES`` and every other answer
    raise :class:`MissionHostFailure`, because none of them is evidence that the
    lease is available and a scheduler that shrugs at them will hand the same
    work to two workers.
    """

    _require_interprocess_lock()
    descriptor = _secure_open_at(lease_dir_fd, name, create=True)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(descriptor)
        if exc.errno in _WOULD_BLOCK:
            return None
        raise MissionHostFailure(
            f"the execution lease {name} could not be tested: {exc}"
        ) from exc
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@dataclass(frozen=True)
class _Store:
    """One open store directory: the descriptor, and what it is bound to."""

    fd: int
    root_identity: "_Identity"
    identity: "_Identity"


@dataclass(frozen=True)
class _Txn:
    """One open transaction: the connection, and the directory it belongs to.

    The descriptor travels with the connection because every authority check
    inside a transaction - a lease file, a store identity - has to be made
    against the same directory this transaction will publish into, not against
    a pathname that could now mean something else.
    """

    conn: "sqlite3.Connection"
    store_fd: int


@dataclass(frozen=True)
class _Identity:
    """Which file or directory a path named at the moment it was inspected."""

    dev: int
    ino: int

    @classmethod
    def of(cls, info: os.stat_result) -> "_Identity":
        return cls(info.st_dev, info.st_ino)


class _Lease:
    """One held execution lease: the descriptor, and what it was open on.

    The descriptor alone is not enough to trust later.  A file descriptor number
    can be closed and reissued to something else entirely, and the lease *path*
    can be unlinked and recreated by anybody who can write the store, which
    would leave this process holding a lock on an orphaned inode while a second
    dispatcher happily locks the replacement.  Recording the identity at the
    moment the lock was taken is what makes both of those detectable.
    """

    __slots__ = ("work_item_id", "name", "fd", "identity", "fence", "released")

    def __init__(self, work_item_id: str, name: str, fd: int, fence: int) -> None:
        self.work_item_id = work_item_id
        self.name = name
        self.fd = fd
        self.identity = _Identity.of(os.fstat(fd))
        self.fence = fence
        self.released = False


def _drop_lease(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError:  # pragma: no cover - unlock of a lock never taken
        pass
    finally:
        os.close(descriptor)


# --------------------------------------------------------------------------
# durable decoding
# --------------------------------------------------------------------------

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS meta ("
    " key TEXT PRIMARY KEY,"
    " value TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS counters ("
    " name TEXT PRIMARY KEY,"
    " value INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS missions ("
    " mission_id TEXT PRIMARY KEY,"
    " scope TEXT NOT NULL,"
    " objective TEXT NOT NULL,"
    " desired_result TEXT NOT NULL,"
    " criteria TEXT NOT NULL,"
    " status TEXT NOT NULL,"
    " created_at REAL NOT NULL,"
    " completed_at REAL,"
    " evidence TEXT)",
    "CREATE TABLE IF NOT EXISTS work_items ("
    " work_item_id TEXT PRIMARY KEY,"
    " mission_id TEXT NOT NULL REFERENCES missions(mission_id),"
    " scope TEXT NOT NULL,"
    " lane TEXT NOT NULL,"
    " priority INTEGER NOT NULL,"
    " is_root INTEGER NOT NULL,"
    " parent_id TEXT,"
    " return_to_id TEXT,"
    " project TEXT NOT NULL,"
    " repository TEXT NOT NULL,"
    " location TEXT NOT NULL,"
    " resources TEXT NOT NULL,"
    " status TEXT NOT NULL,"
    " fence INTEGER NOT NULL,"
    " attempts INTEGER NOT NULL,"
    " submit_seq INTEGER NOT NULL,"
    " submitted_at REAL NOT NULL,"
    " dispatched_at REAL,"
    " worker TEXT,"
    " heartbeats INTEGER NOT NULL,"
    " last_heartbeat_at REAL,"
    " disposition TEXT,"
    " closure TEXT)",
    # Execution dependencies, deliberately *not* lineage.  A parent says where a
    # side issue came from and where it hands control back; a dependency says
    # what has to be delivered before this work can start at all.  Conflating
    # them makes one of the two unexpressible.
    "CREATE TABLE IF NOT EXISTS dependencies ("
    " work_item_id TEXT NOT NULL REFERENCES work_items(work_item_id),"
    " depends_on_id TEXT NOT NULL REFERENCES work_items(work_item_id),"
    " PRIMARY KEY (work_item_id, depends_on_id))",
    "CREATE INDEX IF NOT EXISTS dependencies_reverse ON dependencies(depends_on_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS work_items_one_root"
    " ON work_items(mission_id) WHERE is_root = 1",
    "CREATE INDEX IF NOT EXISTS work_items_ready ON work_items(status, mission_id)",
    "CREATE TABLE IF NOT EXISTS claims ("
    " resource TEXT PRIMARY KEY,"
    " work_item_id TEXT NOT NULL)",
    # Operation receipts.  Written in the same transaction as the effect they
    # describe, so - publication being one atomic rename - a receipt exists
    # exactly when its effect does.  That is what makes an interrupted call
    # answerable instead of ambiguous.
    "CREATE TABLE IF NOT EXISTS operations ("
    " operation_key TEXT PRIMARY KEY,"
    " kind TEXT NOT NULL,"
    " payload TEXT NOT NULL,"
    " result TEXT NOT NULL,"
    " recorded_at REAL NOT NULL,"
    " sequence INTEGER NOT NULL UNIQUE,"
    " acknowledged INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS operations_release ON operations(acknowledged, sequence)",
    "CREATE TABLE IF NOT EXISTS rotation ("
    " scope TEXT PRIMARY KEY,"
    " last_dispatch_seq INTEGER NOT NULL)",
)

#: A work item is runnable only when every dependency it declared has closed
#: with a disposition that actually delivered.  Written as ``NOT EXISTS`` over
#: the dependency table with ``IFNULL``, because ``disposition <> 'fixed'`` is
#: *null* - and therefore not true - for an open dependency, which would have
#: read as "no blocker" and made the whole check ornamental.
_READY_PREDICATE = (
    " w.status = 'ready' AND m.status = 'open'"
    " AND NOT EXISTS ("
    "  SELECT 1 FROM dependencies d"
    "  LEFT JOIN work_items p ON p.work_item_id = d.depends_on_id"
    "  WHERE d.work_item_id = w.work_item_id"
    "  AND (p.work_item_id IS NULL OR p.status <> 'closed'"
    "       OR IFNULL(p.disposition, '') <> 'fixed'))"
)

#: Deterministic selection, fairness first.
#:
#: The scope served longest ago outranks *everything*, including the repair
#: lane.  Lane preference is real but it is a preference *within* the scope that
#: fairness selected, because a lane-first order lets one tenant that can
#: replenish repair work forever hold every other tenant's primary work off the
#: queue indefinitely - starvation dressed up as urgency.  Below that: priority,
#: then age, then id, so the order is total and reproducible.
_SELECT_READY = (
    "SELECT w.* FROM work_items w"
    " JOIN missions m ON m.mission_id = w.mission_id"
    " LEFT JOIN rotation r ON r.scope = w.scope"
    " WHERE" + _READY_PREDICATE + " ORDER BY"
    " COALESCE(r.last_dispatch_seq, 0) ASC,"
    " (CASE w.lane WHEN 'repair' THEN 0 ELSE 1 END) ASC,"
    " w.priority DESC,"
    " w.submit_seq ASC,"
    " w.work_item_id ASC"
    " LIMIT ?"
)

#: The starvation escape hatch for the bounded window above.
#:
#: The preferred order is a *ranking*, so a full window of mutually conflicting
#: candidates would be re-examined on every call forever and a runnable item
#: ranked below them would never be reached.  This sweep walks ready work in id
#: order from a durable cursor instead, so every item is eventually considered
#: no matter how it ranks.  Bounded per call, and the cursor is durable, so the
#: guarantee survives a restart.
_SELECT_SWEEP = (
    "SELECT w.* FROM work_items w"
    " JOIN missions m ON m.mission_id = w.mission_id"
    " WHERE" + _READY_PREDICATE + " AND w.work_item_id > ?"
    " ORDER BY w.work_item_id ASC LIMIT ?"
)


_SCHEMA_SHAPE: Optional[frozenset] = None


def _expected_schema_objects() -> frozenset:
    """Every ``sqlite_master`` row a freshly bootstrapped v1 store has.

    Derived by applying :data:`_SCHEMA` to an empty in-memory database rather
    than by listing names in a constant, so the implicit ``sqlite_autoindex_*``
    entries SQLite creates for the declared primary keys are covered exactly and
    cannot drift away from the schema they come from.
    """

    global _SCHEMA_SHAPE
    if _SCHEMA_SHAPE is None:
        probe = sqlite3.connect(":memory:")
        try:
            for statement in _SCHEMA:
                probe.execute(statement)
            _SCHEMA_SHAPE = frozenset(
                (str(kind), str(name))
                for kind, name in probe.execute(
                    "SELECT type, name FROM sqlite_master"
                ).fetchall()
            )
        finally:
            probe.close()
    return _SCHEMA_SHAPE


def _decode_json(raw: object, label: str) -> Any:
    if not isinstance(raw, str):
        raise MissionCorrupt(f"the stored {label} is malformed")
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise MissionCorrupt(f"the stored {label} is malformed") from exc


def _decode_resources(raw: object) -> Tuple[MissionResource, ...]:
    payload = _decode_json(raw, "resource list")
    if not isinstance(payload, list) or len(payload) > MAX_RESOURCES:
        raise MissionCorrupt("the stored resource list is malformed")
    resources = []
    for entry in payload:
        if not isinstance(entry, list) or len(entry) != 3:
            raise MissionCorrupt("the stored resource list is malformed")
        try:
            resources.append(MissionResource(entry[0], entry[1], entry[2]))
        except MissionRejected as exc:
            raise MissionCorrupt("the stored resource list is malformed") from exc
    if tuple(resources) != _canonical_resources(resources):
        raise MissionCorrupt("the stored resource list is not canonically ordered")
    return tuple(resources)


def _decode_closure(raw: object) -> Optional[Closure]:
    if raw is None:
        return None
    payload = _decode_json(raw, "closure")
    if not isinstance(payload, dict):
        raise MissionCorrupt("the stored closure is malformed")
    try:
        return Closure(
            disposition=payload.get("disposition"),
            rationale=payload.get("rationale"),
            risk=payload.get("risk"),
            evidence_refs=tuple(payload.get("evidence_refs") or ()),
            owner=payload.get("owner"),
            revisit_at=payload.get("revisit_at"),
        )
    except (MissionRejected, TypeError) as exc:
        raise MissionCorrupt("the stored closure is malformed") from exc


def _decode_criteria(raw: object) -> Tuple[AcceptanceCriterion, ...]:
    payload = _decode_json(raw, "acceptance criteria")
    if not isinstance(payload, list) or not payload:
        raise MissionCorrupt("the stored acceptance criteria are malformed")
    if len(payload) > MAX_ACCEPTANCE_CRITERIA:
        raise MissionCorrupt("the stored acceptance criteria exceed the bound")
    criteria = []
    for entry in payload:
        if not isinstance(entry, list) or len(entry) != 2:
            raise MissionCorrupt("the stored acceptance criteria are malformed")
        try:
            criteria.append(AcceptanceCriterion(entry[0], entry[1]))
        except MissionRejected as exc:
            raise MissionCorrupt("the stored acceptance criteria are malformed") from exc
    if len({criterion.id for criterion in criteria}) != len(criteria):
        raise MissionCorrupt("the stored acceptance criteria repeat an id")
    return tuple(criteria)


def _canonical_resources(
    resources: Iterable[MissionResource],
) -> Tuple[MissionResource, ...]:
    """Deduplicate and sort, so every dispatcher claims in the same order.

    A stable claim order is not cosmetic: two dispatchers that claimed the same
    pair of resources in opposite orders would be able to deadlock each other
    inside one transaction, and a store that accepted duplicates would conflict
    with itself.
    """

    unique = {resource.sort_key: resource for resource in resources}
    return tuple(unique[key] for key in sorted(unique))


def _decode(value: object, label: str, check, *args) -> Any:
    """Run one *inbound* validator over *outbound* bytes.

    Durable state gets exactly the same scrutiny as a caller argument; the only
    difference is which error it raises.  A record that would have been refused
    on the way in is corruption on the way out, never something to repair
    silently or to pass along because it happens to be the right Python type.
    """

    try:
        return check(value, label, *args)
    except MissionRejected as exc:
        raise MissionCorrupt(f"the stored {label} is malformed: {exc}") from exc


def _decode_time(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MissionCorrupt(f"the stored {label} is malformed")
    moment = float(value)
    if not math.isfinite(moment) or moment < 0:
        raise MissionCorrupt(f"the stored {label} is not a usable timestamp")
    return moment


def _decode_evidence(
    raw: object, criteria: Tuple[AcceptanceCriterion, ...]
) -> Tuple[Tuple[str, str], ...]:
    """Acceptance evidence is only evidence if it still matches the contract."""

    payload = _decode_json(raw, "acceptance evidence")
    if not isinstance(payload, dict):
        raise MissionCorrupt("the stored acceptance evidence is malformed")
    evidence = {}
    for key, value in payload.items():
        criterion = _decode(key, "acceptance criterion id", _check_token)
        evidence[criterion] = _decode(value, "acceptance evidence ref", _check_token)
    if set(evidence) != {criterion.id for criterion in criteria}:
        raise MissionCorrupt(
            "the stored acceptance evidence does not match the acceptance criteria"
        )
    return tuple(sorted(evidence.items()))


def _row_to_mission(row: sqlite3.Row) -> Mission:
    status = row["status"]
    if status not in _MISSION_STATUSES:
        raise MissionCorrupt("the stored mission status is unknown")
    criteria = _decode_criteria(row["criteria"])
    completed_at = row["completed_at"]
    if (status == MISSION_COMPLETED) != (completed_at is not None):
        raise MissionCorrupt("the stored mission disagrees with its own completion time")
    if (status == MISSION_COMPLETED) != (row["evidence"] is not None):
        raise MissionCorrupt("the stored mission disagrees with its own acceptance evidence")
    return Mission(
        mission_id=_decode(row["mission_id"], "mission id", _check_identifier, _MISSION_PREFIX),
        scope=_decode(row["scope"], "mission scope", _check_token),
        objective=_decode(row["objective"], "objective", _check_text),
        desired_result=_decode(row["desired_result"], "desired result", _check_text),
        acceptance_criteria=criteria,
        status=status,
        created_at=_decode_time(row["created_at"], "mission creation time"),
        completed_at=(
            None if completed_at is None else _decode_time(completed_at, "completion time")
        ),
        acceptance_evidence=(
            () if row["evidence"] is None else _decode_evidence(row["evidence"], criteria)
        ),
    )


def _decode_operation_result(raw: object, kind: str) -> Dict[str, Any]:
    """Parse one receipt's recorded outcome, refusing any shape but its own."""

    payload = _decode_json(raw, "operation result")
    if not isinstance(payload, dict):
        raise MissionCorrupt("the stored operation result is malformed")
    subject = _OPERATION_SUBJECT[kind]
    prefix = _MISSION_PREFIX if subject == "mission_id" else _WORK_PREFIX
    expected = {subject}
    if kind == OPERATION_DISPATCH:
        expected = {subject, "fence"}
    if set(payload) != expected:
        raise MissionCorrupt("the stored operation result does not match its kind")
    result: Dict[str, Any] = {
        subject: _decode(payload[subject], "operation subject", _check_identifier, prefix)
    }
    if kind == OPERATION_DISPATCH:
        result["fence"] = _decode(payload["fence"], "operation fence", _check_int, 1, MAX_SEQUENCE)
    return result


def _decode_optional_id(value: object, label: str) -> Optional[str]:
    if value is None:
        return None
    return _decode(value, label, _check_identifier, _WORK_PREFIX)


def _row_to_item(row: sqlite3.Row, depends_on_ids: Tuple[str, ...]) -> WorkItem:
    """Rebuild one work item, refusing every field that is not exactly right.

    ``depends_on_ids`` is passed in rather than looked up here so that no call
    site can quietly reconstruct an item with its execution dependencies
    dropped - an item that looks dependency-free is an item the scheduler would
    run too early.
    """

    lane = row["lane"]
    if lane not in LANES:
        raise MissionCorrupt("the stored work-item lane is unknown")
    status = row["status"]
    if status not in _STATUSES:
        raise MissionCorrupt("the stored work-item status is unknown")

    raw_root = row["is_root"]
    if isinstance(raw_root, bool) or not isinstance(raw_root, int) or raw_root not in (0, 1):
        raise MissionCorrupt("the stored work-item root flag is malformed")
    is_root = bool(raw_root)

    parent_id = _decode_optional_id(row["parent_id"], "parent id")
    return_to_id = _decode_optional_id(row["return_to_id"], "return_to id")
    if is_root != (parent_id is None) or is_root != (return_to_id is None):
        raise MissionCorrupt("the stored work item disagrees with its own lineage")

    closure = _decode_closure(row["closure"])
    if (status == STATUS_CLOSED) != (closure is not None):
        raise MissionCorrupt("the stored work item disagrees with its own closure")
    disposition = row["disposition"]
    if disposition != (None if closure is None else closure.disposition):
        raise MissionCorrupt("the stored work item disagrees with its own disposition")

    worker = row["worker"]
    if (status == STATUS_DISPATCHED) != (worker is not None):
        raise MissionCorrupt("the stored work item disagrees with its own worker")
    dispatched_at = row["dispatched_at"]
    if (status == STATUS_DISPATCHED) != (dispatched_at is not None):
        raise MissionCorrupt("the stored work item disagrees with its own dispatch time")

    # A fencing token is burned exactly once per dispatch, so carrying a token
    # and having been dispatched are the same fact recorded twice.  Work closed
    # by :meth:`MissionStore.abandon_unrunnable_work_item` never ran, and
    # correctly carries neither.
    fence = _decode(row["fence"], "fence", _check_int, 0, MAX_SEQUENCE)
    attempts = _decode(row["attempts"], "attempts", _check_int, 0, MAX_SEQUENCE)
    if (fence == 0) != (attempts == 0):
        raise MissionCorrupt("the stored work item disagrees with its own dispatch history")
    if status == STATUS_DISPATCHED and attempts < 1:
        raise MissionCorrupt("a dispatched work item was never dispatched")

    return WorkItem(
        work_item_id=_decode(row["work_item_id"], "work item id", _check_identifier, _WORK_PREFIX),
        mission_id=_decode(row["mission_id"], "mission id", _check_identifier, _MISSION_PREFIX),
        scope=_decode(row["scope"], "work-item scope", _check_token),
        lane=lane,
        priority=_decode(row["priority"], "priority", _check_int, 0, MAX_PRIORITY),
        is_root=is_root,
        parent_id=parent_id,
        return_to_id=return_to_id,
        coordinates=_decode_coordinates(row),
        resources=_decode_resources(row["resources"]),
        depends_on_ids=depends_on_ids,
        status=status,
        fence=fence,
        attempts=attempts,
        submit_seq=_decode(row["submit_seq"], "submit sequence", _check_int, 1, MAX_SEQUENCE),
        submitted_at=_decode_time(row["submitted_at"], "submission time"),
        dispatched_at=(
            None if dispatched_at is None else _decode_time(dispatched_at, "dispatch time")
        ),
        worker=None if worker is None else _decode(worker, "worker", _check_token),
        heartbeats=_decode(row["heartbeats"], "heartbeats", _check_int, 0, MAX_SEQUENCE),
        last_heartbeat_at=(
            None
            if row["last_heartbeat_at"] is None
            else _decode_time(row["last_heartbeat_at"], "heartbeat time")
        ),
        closure=closure,
    )


def _decode_coordinates(row: sqlite3.Row) -> WorkCoordinates:
    try:
        return WorkCoordinates(row["project"], row["repository"], row["location"])
    except MissionRejected as exc:
        raise MissionCorrupt("the stored work-item coordinates are malformed") from exc


def _chain_to_root(items: Mapping[str, WorkItem], start: str) -> Tuple[str, ...]:
    """Walk parent edges from ``start`` to the root, refusing anything that is not a DAG.

    Returns the chain including ``start`` itself, so ``chain[1:]`` is exactly the
    set of ids a return edge is allowed to point at.  A repeated node, a missing
    parent, a non-root item with no parent, or a chain deeper than
    :data:`MAX_DAG_DEPTH` is durable state this kernel will not reason over.
    """

    chain: list = []
    seen = set()
    current: Optional[str] = start
    while current is not None:
        if current in seen:
            raise MissionCorrupt("the work-item graph contains a cycle")
        if len(chain) > MAX_DAG_DEPTH:
            raise MissionCorrupt("the work-item graph is deeper than the bound")
        seen.add(current)
        item = items.get(current)
        if item is None:
            raise MissionCorrupt("the work-item graph references a missing item")
        chain.append(current)
        if item.is_root:
            return tuple(chain)
        if item.parent_id is None:
            raise MissionCorrupt("a non-root work item has no parent")
        current = item.parent_id
    raise MissionCorrupt("the work-item graph has no root")  # pragma: no cover - defensive


def _validate_dependency_graph(items: Mapping[str, WorkItem]) -> None:
    """Insist the declared execution dependencies are a DAG inside one mission.

    Submission cannot normally create a cycle - a dependency has to exist
    already, so the edge always points at an older item - which is exactly why
    this check matters: a cycle in durable state did not come from the API, and
    the scheduler would otherwise express it as work that is silently never
    runnable, forever, with nothing to point at.

    Bounded by construction: each node is finished once, and the traversal is
    iterative, so a deep or wide graph costs time rather than stack.
    """

    state: Dict[str, int] = {}
    for start in sorted(items):
        if state.get(start):
            continue
        # (node, has_been_expanded); the second visit is the post-order one.
        stack = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                state[node] = 2
                continue
            if state.get(node) == 2:
                continue
            if state.get(node) == 1:
                raise MissionCorrupt("the work-item dependencies contain a cycle")
            item = items.get(node)
            if item is None:
                raise MissionCorrupt(
                    "a work item depends on something outside its own mission"
                )
            state[node] = 1
            stack.append((node, True))
            for dependency in item.depends_on_ids:
                if dependency == node:
                    raise MissionCorrupt("a work item depends on itself")
                if state.get(dependency) == 1:
                    raise MissionCorrupt("the work-item dependencies contain a cycle")
                if state.get(dependency) != 2:
                    stack.append((dependency, False))


# --------------------------------------------------------------------------
# dispatch handle
# --------------------------------------------------------------------------


class DispatchHandle:
    """Execution authority for exactly one work item, for as long as it is held.

    The handle owns an open, ``flock``-held lease descriptor.  That descriptor -
    not the row in the database, and certainly not a timestamp - is what makes
    this process the one entitled to run the item.  :meth:`heartbeat` moves
    observability counters and touches nothing else; there is no API on this
    type or on the store that hands the lease to another worker, and none that
    lets a heartbeat lapse into a release.
    """

    __slots__ = ("_store", "_item", "_lease", "_released", "_closed")

    def __init__(self, store: "MissionStore", item: WorkItem, lease: _Lease) -> None:
        self._store = store
        self._item = item
        self._lease = lease
        self._released = False
        self._closed = False

    # -- identity -------------------------------------------------------

    @property
    def work_item(self) -> WorkItem:
        return self._item

    @property
    def work_item_id(self) -> str:
        return self._item.work_item_id

    @property
    def mission_id(self) -> str:
        return self._item.mission_id

    @property
    def scope(self) -> str:
        return self._item.scope

    @property
    def lane(self) -> Lane:
        return self._item.lane

    @property
    def priority(self) -> int:
        return self._item.priority

    @property
    def coordinates(self) -> WorkCoordinates:
        return self._item.coordinates

    @property
    def resources(self) -> Tuple[MissionResource, ...]:
        return self._item.resources

    @property
    def fence(self) -> int:
        """The token this dispatch burned.  Every mutation must present it."""

        return self._item.fence

    @property
    def worker(self) -> str:
        return self._item.worker or ""

    @property
    def closed(self) -> bool:
        return self._closed

    # -- authority ------------------------------------------------------

    @property
    def depends_on_ids(self) -> Tuple[str, ...]:
        return self._item.depends_on_ids

    def heartbeat(self) -> int:
        """Record liveness.  Observability only; authority is untouched.

        Returns the new heartbeat count.  Nothing about the lease, the fence,
        the status or the resource claims moves, and no other worker becomes
        able to take this item because a heartbeat did or did not arrive.  The
        lease is re-verified first, so a heartbeat is also never a way to write
        into an era this process no longer owns.
        """

        if self._released:
            raise MissionUnauthorized("the dispatch lease has already been released")
        return self._store._heartbeat(self._lease)

    def close(self, closure: Closure, *, operation: str) -> WorkItem:
        """Close this work item, on the authority of this live lease.

        Reachable only from the handle that holds the lease.  There is no store
        method that closes work from an id and a fencing token, because both of
        those are visible in a snapshot and neither is evidence that the caller
        is still the one running the work.
        """

        if self._released:
            raise MissionUnauthorized("the dispatch lease has already been released")
        item = self._store._close(self._lease, closure, _check_token(operation, "operation"))
        self._closed = True
        return item

    def _release(self) -> None:
        """Give the lease back, requeuing the item if it was never closed.

        An item that ends its dispatch unclosed goes back to ``ready`` with its
        resource claims dropped; it keeps its fencing token, so the *next*
        dispatch is what makes this handle's token stale.
        """

        if self._released:
            return
        self._released = True
        try:
            if not self._closed:
                self._store._requeue(self._lease)
        finally:
            self._store._retire_lease(self._lease)


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------


class MissionStore:
    """Durable, multi-process mission scheduling rooted at one directory.

    Every mutating method takes a single store-wide ``flock`` and runs inside one
    SQLite transaction, so two processes see one queue, one fencing sequence and
    one fairness rotation.  Constructing a store creates nothing: the first
    mutating call establishes the store directory, the database and the lease
    directory as owner-only, refusing any of them that is a symbolic link, and
    fails closed rather than storing missions somewhere readable.  Read-only
    :meth:`snapshot` and :meth:`metrics` create nothing and lock nothing, so they
    remain usable against a store this process may only read.

    ``queue_capacity`` is a property of the *store*.  The first initialisation
    records it durably; a later :class:`MissionStore` that asks for a different
    one is rejected rather than silently given its own idea of full.
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        *,
        queue_capacity: Optional[int] = None,
    ) -> None:
        if queue_capacity is not None:
            _check_int(queue_capacity, "queue_capacity", 1, MAX_QUEUE_CAPACITY)
        self._requested_capacity = queue_capacity
        # Lexical, never ``resolve()``.  ``resolve`` follows the caller's root
        # through a symbolic link *before* anything gets to inspect it, so a
        # store pointed at an alias would be silently established wherever the
        # alias led - outside the directory the caller actually configured, and
        # re-moded and written there.  The configured path is kept exactly as
        # given (absolute and lexically normalised only) so that
        # :meth:`_prepare_root` can ``lstat`` the real final component and
        # refuse it.
        self._configured_root = Path(os.path.abspath(os.fspath(root)))
        self._root = self._configured_root / _STORE_DIRNAME
        self._db = self._root / _DB_NAME
        self._lease_root = self._root / _LEASE_DIRNAME
        #: Leases this *process* is holding, keyed by work item.  A dispatch
        #: handle is authorised by identity against this registry, so a handle
        #: assembled somewhere else - in another process, out of snapshot data -
        #: has nothing to be authorised against.
        self._leases: Dict[str, _Lease] = {}

    @property
    def root(self) -> Path:
        return self._root

    @property
    def configured_root(self) -> Path:
        """The directory the caller named, lexically, without following links."""

        return self._configured_root

    # -- missions -------------------------------------------------------

    def create_mission(
        self,
        *,
        operation: str,
        scope: str,
        objective: str,
        desired_result: str,
        acceptance_criteria: Sequence[Union[AcceptanceCriterion, Tuple[str, str]]],
    ) -> Mission:
        """Record one immutable mission contract.

        The three contract fields are validated here and never accepted again.
        There is deliberately no ``update_mission``: a mission whose objective
        was wrong is a different mission, and rewriting it after work items have
        already closed against it would make every closure unauditable.
        """

        key = _check_token(operation, "operation")
        _check_token(scope, "scope")
        objective = _check_text(objective, "objective")
        desired_result = _check_text(desired_result, "desired result")
        criteria = self._normalise_criteria(acceptance_criteria)
        payload = self._payload_digest(
            OPERATION_CREATE_MISSION,
            scope,
            objective,
            desired_result,
            [[item.id, item.statement] for item in criteria],
        )
        now = time.time()

        recalled = None
        with self._mutate() as txn:
            conn = txn.conn
            recalled = self._recall(conn, key, OPERATION_CREATE_MISSION, payload)
            if recalled is not None:
                mission_id = recalled["mission_id"]
            else:
                mission_id = self._create_mission_row(
                    conn, key, payload, scope, objective, desired_result, criteria, now
                )
        if recalled is not None:
            # The original effect, with its original id and timestamps, read back
            # from durable state rather than reconstructed from this call.
            return self.get_mission(mission_id)
        return Mission(
            mission_id=mission_id,
            scope=scope,
            objective=objective,
            desired_result=desired_result,
            acceptance_criteria=criteria,
            status=MISSION_OPEN,
            created_at=now,
        )

    def _create_mission_row(
        self,
        conn: sqlite3.Connection,
        key: str,
        payload: str,
        scope: str,
        objective: str,
        desired_result: str,
        criteria: Tuple[AcceptanceCriterion, ...],
        now: float,
    ) -> str:
            mission_id = _MISSION_PREFIX + "%012d" % self._bump(conn, "mission_seq")
            self._bump(conn, "missions_created")
            conn.execute(
                "INSERT INTO missions"
                " (mission_id, scope, objective, desired_result, criteria, status,"
                "  created_at, completed_at, evidence)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                (
                    mission_id,
                    scope,
                    objective,
                    desired_result,
                    json.dumps(
                        [[item.id, item.statement] for item in criteria],
                        separators=(",", ":"),
                    ),
                    MISSION_OPEN,
                    now,
                ),
            )
            self._record(
                conn, key, OPERATION_CREATE_MISSION, payload, {"mission_id": mission_id}
            )
            return mission_id

    def get_mission(self, mission_id: str) -> Mission:
        _check_token(mission_id, "mission id")
        with self._read() as txn:
            conn = None if txn is None else txn.conn
            if conn is None:
                raise MissionRejected("no such mission")
            row = conn.execute(
                "SELECT * FROM missions WHERE mission_id = ?", (mission_id,)
            ).fetchone()
        if row is None:
            raise MissionRejected("no such mission")
        return _row_to_mission(row)

    def complete_mission(
        self, mission_id: str, evidence: Mapping[str, str], *, operation: str
    ) -> Mission:
        """Close a mission, but only against the whole contract.

        Three things have to hold at once, and each of them is the thing some
        other design quietly skips: every acceptance criterion has evidence and
        no evidence is offered for a criterion that was never agreed; every work
        item is explicitly closed as fixed, deferred or blocked; and the graph is
        still a rooted DAG in which every side item's return edge resolves to one
        of its own ancestors.  Anything else raises rather than completing.
        """

        _check_token(mission_id, "mission id")
        key = _check_token(operation, "operation")
        if not isinstance(evidence, Mapping):
            raise MissionRejected("acceptance evidence must be a mapping")
        supplied = {}
        for criterion, value in evidence.items():
            _check_token(criterion, "acceptance criterion id")
            supplied[criterion] = _check_token(value, "acceptance evidence ref")

        payload = self._payload_digest(
            OPERATION_COMPLETE_MISSION, mission_id, sorted(supplied.items())
        )
        now = time.time()
        recalled = None
        with self._mutate() as txn:
            conn = txn.conn
            recalled = self._recall(conn, key, OPERATION_COMPLETE_MISSION, payload)
            if recalled is not None:
                mission = self._load_mission(conn, mission_id)
            else:
                mission = self._load_mission(conn, mission_id)
                self._complete_mission_row(conn, key, payload, mission, supplied, now)
        if recalled is not None:
            return self.get_mission(mission_id)
        return Mission(
            mission_id=mission.mission_id,
            scope=mission.scope,
            objective=mission.objective,
            desired_result=mission.desired_result,
            acceptance_criteria=mission.acceptance_criteria,
            status=MISSION_COMPLETED,
            created_at=mission.created_at,
            completed_at=now,
            acceptance_evidence=tuple(sorted(supplied.items())),
        )

    def _complete_mission_row(
        self,
        conn: sqlite3.Connection,
        key: str,
        payload: str,
        mission: Mission,
        supplied: Dict[str, str],
        now: float,
    ) -> None:
            mission_id = mission.mission_id
            if mission.status != MISSION_OPEN:
                raise MissionRejected("the mission is already completed")

            expected = set(mission.criteria_ids)
            if set(supplied) != expected:
                missing = sorted(expected - set(supplied))
                unknown = sorted(set(supplied) - expected)
                raise MissionRejected(
                    "acceptance evidence must cover exactly the acceptance criteria"
                    f" (missing: {missing}, unknown: {unknown})"
                )

            items = self._load_items(conn, mission_id)
            if not items:
                raise MissionRejected("a mission with no work items cannot be completed")
            open_items = sorted(
                item.work_item_id for item in items.values() if item.status != STATUS_CLOSED
            )
            if open_items:
                raise MissionRejected(
                    f"every work item must be explicitly closed first: {open_items}"
                )
            # The main axis is not one of several equal outcomes.  Side
            # issues may legitimately end deferred or blocked with full
            # accounting, but a mission whose *root* was deferred or blocked did
            # not reach its objective, and recording that as completed would be
            # the largest silent skip this kernel could possibly permit.
            root = next(item for item in items.values() if item.is_root)
            if root.disposition != DISPOSITION_FIXED:
                raise MissionRejected(
                    f"the main work item closed as {root.disposition};"
                    " a mission completes only on a fixed root"
                )
            self._validate_graph(items)

            conn.execute(
                "UPDATE missions SET status = ?, completed_at = ?, evidence = ?"
                " WHERE mission_id = ?",
                (
                    MISSION_COMPLETED,
                    now,
                    json.dumps(supplied, sort_keys=True, separators=(",", ":")),
                    mission_id,
                ),
            )
            self._bump(conn, "missions_completed")
            self._record(
                conn, key, OPERATION_COMPLETE_MISSION, payload, {"mission_id": mission_id}
            )

    # -- work items -----------------------------------------------------

    def submit_work_item(
        self,
        mission_id: str,
        *,
        operation: str,
        coordinates: WorkCoordinates,
        resources: Sequence[MissionResource] = (),
        lane: Lane = LANE_PRIMARY,
        priority: int = 0,
        root: bool = False,
        parent_id: Optional[str] = None,
        return_to_id: Optional[str] = None,
        depends_on_ids: Sequence[str] = (),
    ) -> WorkItem:
        """Add one node to a mission's work graph.

        Two different edges are recorded here and they must not be confused.

        *Lineage* is ``parent_id`` / ``return_to_id``: where a side issue came
        from and which ancestor it hands control back to.  The first item of a
        mission is its root - no parent, no return edge, primary lane - and
        every later item must name both, with the return target being the parent
        or one of its ancestors.  A return edge that points sideways or
        downwards is not a route home and is rejected.

        *Dependencies* are ``depends_on_ids``: what must be **delivered** before
        this work can start.  They are bounded, immutable once submitted, must
        name work items of this same mission, and are what the scheduler
        actually gates readiness on - a dependency that has not closed as
        ``fixed`` keeps this item off the queue entirely.  Lineage gates
        nothing; a parent and a child are frequently runnable at the same time.
        """

        _check_token(mission_id, "mission id")
        key = _check_token(operation, "operation")
        _check_bool(root, "root")
        dependencies = self._normalise_dependencies(depends_on_ids)
        if not isinstance(coordinates, WorkCoordinates):
            raise MissionRejected("coordinates must be a WorkCoordinates")
        if lane not in LANES:
            raise MissionRejected("lane must be one of " + ", ".join(LANES))
        _check_int(priority, "priority", 0, MAX_PRIORITY)
        if not isinstance(resources, (list, tuple)):
            raise MissionRejected("resources must be a sequence")
        if len(resources) > MAX_RESOURCES:
            raise MissionRejected(f"resources exceed {MAX_RESOURCES} entries")
        for resource in resources:
            if not isinstance(resource, MissionResource):
                raise MissionRejected("every resource must be a MissionResource")
        canonical = _canonical_resources(resources)
        if root:
            if parent_id is not None or return_to_id is not None:
                raise MissionRejected("the root work item has no parent and no return edge")
            if lane != LANE_PRIMARY:
                raise MissionRejected("the root work item runs in the primary lane")
        else:
            if parent_id is None or return_to_id is None:
                raise MissionRejected(
                    "a side work item requires both parent_id and return_to_id"
                )
            _check_token(parent_id, "parent id")
            _check_token(return_to_id, "return_to id")

        payload = self._payload_digest(
            OPERATION_SUBMIT_WORK_ITEM,
            mission_id,
            [coordinates.project, coordinates.repository, coordinates.location],
            [list(resource.sort_key) for resource in canonical],
            lane,
            priority,
            root,
            parent_id,
            return_to_id,
            list(dependencies),
        )
        now = time.time()
        rejected = False
        recalled = None
        with self._mutate() as txn:
            conn = txn.conn
            recalled = self._recall(conn, key, OPERATION_SUBMIT_WORK_ITEM, payload)
            if recalled is not None:
                work_item_id = recalled["work_item_id"]
                mission = self._load_mission(conn, mission_id)
                submit_seq = 0
                rejected = False
            else:
                mission = self._load_mission(conn, mission_id)
                if mission.status != MISSION_OPEN:
                    raise MissionRejected("the mission is already completed")

                depth = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM work_items WHERE status != ?", (STATUS_CLOSED,)
                    ).fetchone()[0]
                )
                if depth >= self._capacity(conn):
                    self._bump(conn, "capacity_rejects")
                    rejected = True
                else:
                    work_item_id, submit_seq = self._submit_work_item_row(
                        conn, key, payload, mission, now, root, lane, priority,
                        parent_id, return_to_id, coordinates, canonical, dependencies,
                    )
        if rejected:
            raise MissionCapacityExceeded(
                "the mission queue is at its durable capacity; nothing was queued"
            )
        if recalled is not None:
            return self.get_work_item(work_item_id)
        return WorkItem(
            work_item_id=work_item_id,
            mission_id=mission_id,
            scope=mission.scope,
            lane=lane,
            priority=priority,
            is_root=root,
            parent_id=parent_id,
            return_to_id=return_to_id,
            coordinates=coordinates,
            resources=canonical,
            depends_on_ids=dependencies,
            status=STATUS_READY,
            fence=0,
            attempts=0,
            submit_seq=submit_seq,
            submitted_at=now,
        )

    def _submit_work_item_row(
        self,
        conn: sqlite3.Connection,
        key: str,
        payload: str,
        mission: Mission,
        now: float,
        root: bool,
        lane: Lane,
        priority: int,
        parent_id: Optional[str],
        return_to_id: Optional[str],
        coordinates: WorkCoordinates,
        canonical: Tuple[MissionResource, ...],
        dependencies: Tuple[str, ...],
    ) -> Tuple[str, int]:
                mission_id = mission.mission_id
                items = self._load_items(conn, mission_id)
                if len(items) >= MAX_WORK_ITEMS_PER_MISSION:
                    raise MissionRejected(
                        f"a mission holds at most {MAX_WORK_ITEMS_PER_MISSION} work items"
                    )
                if root:
                    if items:
                        raise MissionRejected("the mission already has a root work item")
                else:
                    if not items:
                        raise MissionRejected("a mission needs its root work item first")
                    parent = items.get(parent_id or "")
                    if parent is None:
                        raise MissionRejected("parent_id names no work item of this mission")
                    chain = _chain_to_root(items, parent.work_item_id)
                    if return_to_id not in chain:
                        raise MissionRejected(
                            "return_to_id must be the parent or one of its ancestors"
                        )

                for dependency in dependencies:
                    if dependency not in items:
                        raise MissionRejected(
                            "depends_on_ids must name work items of this same mission"
                        )
                # The existing graph is walked before this item joins it, so a
                # cycle that damaged state introduced is refused here rather
                # than becoming a scheduling deadlock nobody can explain.
                _validate_dependency_graph(items)

                work_item_id = _WORK_PREFIX + "%012d" % self._bump(conn, "work_seq")
                submit_seq = self._bump(conn, "submit_seq")
                conn.execute(
                    "INSERT INTO work_items"
                    " (work_item_id, mission_id, scope, lane, priority, is_root, parent_id,"
                    "  return_to_id, project, repository, location, resources, status, fence,"
                    "  attempts, submit_seq, submitted_at, dispatched_at, worker, heartbeats,"
                    "  last_heartbeat_at, disposition, closure)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?,"
                    "         NULL, NULL, 0, NULL, NULL, NULL)",
                    (
                        work_item_id,
                        mission_id,
                        mission.scope,
                        lane,
                        priority,
                        1 if root else 0,
                        parent_id,
                        return_to_id,
                        coordinates.project,
                        coordinates.repository,
                        coordinates.location,
                        json.dumps(
                            [list(resource.sort_key) for resource in canonical],
                            separators=(",", ":"),
                        ),
                        STATUS_READY,
                        submit_seq,
                        now,
                    ),
                )
                for dependency in dependencies:
                    conn.execute(
                        "INSERT INTO dependencies (work_item_id, depends_on_id)"
                        " VALUES (?, ?)",
                        (work_item_id, dependency),
                    )
                self._record(
                    conn,
                    key,
                    OPERATION_SUBMIT_WORK_ITEM,
                    payload,
                    {"work_item_id": work_item_id},
                )
                return work_item_id, submit_seq

    def get_work_item(self, work_item_id: str) -> WorkItem:
        _check_token(work_item_id, "work item id")
        with self._read() as txn:
            conn = None if txn is None else txn.conn
            if conn is None:
                raise MissionRejected("no such work item")
            return self._load_item(conn, work_item_id)

    # -- dispatch -------------------------------------------------------

    @contextmanager
    def dispatch(self, *, operation: str, worker: str) -> Iterator[Optional[DispatchHandle]]:
        """Claim the next runnable work item, or yield ``None`` if there is none.

        Selection is deterministic and starvation-resistant: the scope served
        longest ago first, then the repair lane within that scope, then
        priority, then age.  A
        candidate is taken only if *all* of its canonically sorted resources can
        be claimed inside one transaction and its execution lease can be taken;
        otherwise the whole attempt for that candidate is rolled back - never
        half-claimed - and the next candidate is considered.  The repair lane
        runs this identical path, so preferring repair work reorders the queue
        without letting it walk through a conflict.

        The yielded handle owns the lease for the duration of the ``with`` block.
        Leaving the block without :meth:`DispatchHandle.close` puts the item back
        on the queue; a later dispatch then burns a higher fencing token, which
        is what makes this one stale.

        ``operation`` names this dispatch, durably.  If the call is interrupted
        after it published - the answer :class:`MissionIndeterminate` exists for
        exactly that - presenting the same key again reacquires authority over
        the *same* work item at the *same* fencing token, rather than dispatching
        something else or reporting an idle queue over work that is durably
        recorded as running.  A key already spent on a dispatch whose era has
        ended raises :class:`MissionOperationSettled`; one whose lease is still
        held by a live process raises :class:`MissionConflict`.
        """

        _check_token(worker, "worker")
        key = _check_token(operation, "operation")
        handle = self._dispatch_once(key, worker)
        try:
            yield handle
        finally:
            if handle is not None:
                handle._release()

    @contextmanager
    def dispatch_expected(
        self, *, operation: str, worker: str, work_item_id: str, expected_attempt: int
    ) -> Iterator[Optional[DispatchHandle]]:
        """Claim exactly one expected ready item and attempt era.

        Unlike :meth:`dispatch`, this contract never falls through to another
        queue item if the advisory candidate moved. The exact identity and its
        attempt era are part of the durable idempotency payload, so the
        same operation can recover only the dispatch it named.
        """

        _check_token(worker, "worker")
        _check_token(work_item_id, "work item id")
        if isinstance(expected_attempt, bool) or not isinstance(expected_attempt, int) \
                or expected_attempt < 1:
            raise MissionRejected("expected attempt is invalid")
        key = _check_token(operation, "operation")
        handle = self._dispatch_expected_once(
            key, worker, work_item_id, expected_attempt
        )
        try:
            yield handle
        finally:
            if handle is not None:
                handle._release()

    def _dispatch_expected_once(
        self, key: str, worker: str, work_item_id: str, expected_attempt: int
    ) -> Optional[DispatchHandle]:
        payload = self._payload_digest(
            OPERATION_DISPATCH, worker, work_item_id, expected_attempt
        )
        pending: Optional[Tuple[WorkItem, int, int]] = None
        recalled = None
        try:
            with self._mutate() as txn:
                conn = txn.conn
                recalled = self._recall(conn, key, OPERATION_DISPATCH, payload)
                if recalled is None:
                    rows = conn.execute(_SELECT_READY, (1,)).fetchall()
                    row = rows[0] if rows and rows[0]["work_item_id"] == work_item_id else None
                    if row is not None:
                        item = self._item_from_row(conn, row)
                        if item.status == STATUS_READY \
                                and item.attempts + 1 == expected_attempt:
                            pending = self._try_candidate(
                                txn, row, worker, time.time(), key, payload
                            )
        except BaseException:
            if pending is not None:
                _drop_lease(pending[2])
            raise

        if recalled is not None:
            if recalled["work_item_id"] != work_item_id:
                raise MissionCorrupt("targeted dispatch receipt names another item")
            return self._recover_dispatch(key, worker, recalled)
        if pending is None:
            return None
        item, fence, descriptor = pending
        try:
            lease = _Lease(
                item.work_item_id, self._lease_name(item.work_item_id), descriptor, fence
            )
        except BaseException:  # pragma: no cover - fstat on a descriptor we hold
            _drop_lease(descriptor)
            raise
        self._leases[item.work_item_id] = lease
        return DispatchHandle(self, item, lease)

    def _recover_dispatch(
        self, key: str, worker: str, result: Dict[str, Any]
    ) -> DispatchHandle:
        """Hand back authority over the dispatch this key already performed.

        The receipt says which work item was taken and under which token.  What
        it cannot say is whether anybody is still running it, so that is
        established the only honest way available - by trying to take the lease.
        Taking it proves no live process holds one; failing to take it proves one
        does, and the original holder keeps its authority rather than having a
        duplicate minted beside it.

        Nothing is re-counted here.  The dispatch already happened: its fencing
        token, its attempt, its resource claims and its rotation position are all
        durable, and issuing them again would be inventing a second dispatch to
        paper over an ambiguous first one.
        """

        work_item_id = result["work_item_id"]
        with self._mutate() as txn:
            item = self._load_item(txn.conn, work_item_id)
            if item.status != STATUS_DISPATCHED or item.fence != result["fence"]:
                raise MissionOperationSettled(
                    f"dispatch {key!r} took {work_item_id} at fence {result['fence']},"
                    " and that era has ended; the work is no longer this caller's to run",
                    result,
                )
            if item.worker != worker:
                raise MissionOperationConflict(
                    f"dispatch {key!r} was taken by worker {item.worker!r}, not {worker!r}"
                )
            descriptor = self._take_lease(txn.store_fd, work_item_id)
            if descriptor is None:
                raise MissionConflict(
                    f"dispatch {key!r} is still held by a live worker; its authority has"
                    " not lapsed and will not be duplicated"
                )
        try:
            lease = _Lease(
                work_item_id, self._lease_name(work_item_id), descriptor, item.fence
            )
        except BaseException:  # pragma: no cover - fstat on a descriptor we hold
            _drop_lease(descriptor)
            raise
        self._leases[work_item_id] = lease
        return DispatchHandle(self, item, lease)

    def _dispatch_once(self, key: str, worker: str) -> Optional[DispatchHandle]:
        """Select, claim and commit one dispatch, and only then hand out authority.

        The ordering here is the whole point.  A lease that is registered while
        the transaction is still open becomes a ghost the moment that
        transaction rolls back: the durable record says ``ready`` and any worker
        may take it, while this process still holds the descriptor and the
        registry entry that make it look dispatched - occupancy with nothing
        behind it, held until the process exits.  So the descriptor is carried
        as a *pending* claim with no authority attached, and it is either
        promoted to a lease after the commit succeeded or unlocked and closed on
        every single failure path before that.
        """

        now = time.time()
        payload = self._payload_digest(OPERATION_DISPATCH, worker)
        pending: Optional[Tuple[WorkItem, int, int]] = None
        recalled = None
        try:
            with self._mutate() as txn:
                conn = txn.conn
                recalled = self._recall(conn, key, OPERATION_DISPATCH, payload)
                if recalled is not None:
                    rows = ()
                else:
                    rows = conn.execute(
                        _SELECT_READY, (MAX_DISPATCH_CANDIDATES,)
                    ).fetchall()
                for row in rows:
                    pending = self._try_candidate(txn, row, worker, now, key)
                    if pending is not None:
                        break

                if recalled is None and pending is None and len(rows) >= MAX_DISPATCH_CANDIDATES:
                    # A full window of candidates that all conflicted would
                    # otherwise be rescanned in the same order forever, so a
                    # runnable item ranked below them would never be reached.
                    # Sweep by id from a durable cursor instead, and advance the
                    # cursor whether or not this sweep wins, so successive calls
                    # provably reach every ready item.
                    self._bump(conn, "scan_truncations")
                    self._bump(conn, "cursor_sweeps")
                    cursor = self._meta_get(conn, "dispatch_cursor", "")
                    swept = conn.execute(
                        _SELECT_SWEEP, (cursor, MAX_DISPATCH_CANDIDATES)
                    ).fetchall()
                    self._meta_set(
                        conn,
                        "dispatch_cursor",
                        (
                            ""
                            if len(swept) < MAX_DISPATCH_CANDIDATES
                            else str(swept[-1]["work_item_id"])
                        ),
                    )
                    for row in swept:
                        pending = self._try_candidate(txn, row, worker, now, key)
                        if pending is not None:
                            break
        except BaseException:
            if pending is not None:
                _drop_lease(pending[2])
            raise

        if recalled is not None:
            # A second visit to a dispatch that already took effect: recover it
            # rather than looking for different work under the same name.
            return self._recover_dispatch(key, worker, recalled)
        if pending is None:
            return None
        item, fence, descriptor = pending
        try:
            lease = _Lease(item.work_item_id, self._lease_name(item.work_item_id), descriptor, fence)
        except BaseException:  # pragma: no cover - fstat on a descriptor we hold
            _drop_lease(descriptor)
            raise
        self._leases[item.work_item_id] = lease
        return DispatchHandle(self, item, lease)

    def _try_candidate(
        self, txn: _Txn, row: sqlite3.Row, worker: str, now: float, key: str = "",
        payload: Optional[str] = None,
    ) -> Optional[Tuple[WorkItem, int, int]]:
        """Claim one candidate completely, or leave the store exactly as found.

        Returns the pending claim - the dispatched item, its fencing token and
        the still-locked lease descriptor - rather than a handle.  Authority is
        not this function's to grant: the transaction it runs in has not
        committed yet.
        """

        conn = txn.conn
        item = self._item_from_row(conn, row)
        # The readiness SQL is a filter, not the authority.  A dependency row
        # that damaged state left dangling would drop out of the join and make
        # this item look free to run, so what actually gates the dispatch is
        # re-checked here against decoded records.
        for dependency in item.depends_on_ids:
            blocker = self._load_item(conn, dependency)
            if blocker.status != STATUS_CLOSED:
                raise MissionCorrupt(
                    "a work item was offered for dispatch with an open dependency"
                )
            if blocker.disposition not in DEPENDENCY_SATISFYING_DISPOSITIONS:
                raise MissionCorrupt(
                    "a work item was offered for dispatch with an unsatisfied dependency"
                )

        conn.execute("SAVEPOINT try_claim")
        try:
            for resource in item.resources:
                conn.execute(
                    "INSERT INTO claims (resource, work_item_id) VALUES (?, ?)",
                    (resource.digest, item.work_item_id),
                )
        except sqlite3.IntegrityError:
            # All or nothing: the partial claims from this attempt go away with
            # the savepoint, so a loser leaves no trace.
            conn.execute("ROLLBACK TO try_claim")
            conn.execute("RELEASE try_claim")
            self._bump(conn, "conflicts")
            return None

        # The repair lane reaches this line by the same route as everything
        # else: a preference in the ordering, never a way past a live claim.
        descriptor = self._take_lease(txn.store_fd, item.work_item_id)
        if descriptor is None:
            conn.execute("ROLLBACK TO try_claim")
            conn.execute("RELEASE try_claim")
            self._bump(conn, "conflicts")
            return None

        try:
            fence = self._bump(conn, "fence")
            dispatch_seq = self._bump(conn, "dispatch_seq")
            self._bump(conn, "dispatches")
            conn.execute(
                "UPDATE work_items SET status = ?, fence = ?, attempts = attempts + 1,"
                " worker = ?, dispatched_at = ? WHERE work_item_id = ?",
                (STATUS_DISPATCHED, fence, worker, now, item.work_item_id),
            )
            conn.execute(
                "INSERT INTO rotation (scope, last_dispatch_seq) VALUES (?, ?)"
                " ON CONFLICT(scope) DO UPDATE SET last_dispatch_seq = excluded"
                ".last_dispatch_seq",
                (item.scope, dispatch_seq),
            )
            self._record_latency(conn, now - item.submitted_at)
            self._record(
                conn,
                key,
                OPERATION_DISPATCH,
                payload or self._payload_digest(OPERATION_DISPATCH, worker),
                {"work_item_id": item.work_item_id, "fence": fence},
            )
            conn.execute("RELEASE try_claim")
        except BaseException:
            _drop_lease(descriptor)
            raise

        dispatched = replace(
            item,
            status=STATUS_DISPATCHED,
            fence=fence,
            attempts=item.attempts + 1,
            dispatched_at=now,
            worker=worker,
        )
        return dispatched, fence, descriptor

    def abandon_unrunnable_work_item(
        self, work_item_id: str, closure: Closure, *, operation: str
    ) -> WorkItem:
        """Close a ready item that a failed dependency put permanently out of reach.

        This is the one closure that does not come from a dispatch, and it
        exists so that strict dependency gating cannot become a deadlock.  An
        item whose dependency closed ``deferred`` or ``blocked`` will never be
        offered for dispatch, so it could never be closed by a worker, so its
        mission could never complete - and a scheduler with an unreachable state
        invites exactly the silent skip this kernel refuses.

        It is not a bypass.  The precondition is objectively checkable and is
        checked here: the item must still be ``ready``, never dispatched, and at
        least one of its declared dependencies must be closed with a disposition
        that did not deliver.  It is a host operation, so no execution lease is
        involved - there is no worker to impersonate.  And the closure must be
        ``deferred`` or ``blocked`` with the full accounting: work that never ran
        cannot be recorded as fixed.
        """

        _check_token(work_item_id, "work item id")
        key = _check_token(operation, "operation")
        if not isinstance(closure, Closure):
            raise MissionRejected("closure must be a Closure")
        if closure.disposition == DISPOSITION_FIXED:
            raise MissionRejected("work that never ran cannot be closed as fixed")
        self._check_closure(closure)
        payload = self._payload_digest(
            OPERATION_ABANDON_WORK_ITEM, work_item_id, closure.as_payload()
        )

        with self._mutate() as txn:
            conn = txn.conn
            if self._recall(conn, key, OPERATION_ABANDON_WORK_ITEM, payload) is not None:
                return self.get_work_item(work_item_id)
            item = self._load_item(conn, work_item_id)
            if item.status != STATUS_READY or item.attempts:
                raise MissionConflict(
                    "only a ready work item that was never dispatched can be abandoned"
                )
            unsatisfied = [
                dependency
                for dependency in item.depends_on_ids
                if self._load_item(conn, dependency).disposition
                not in DEPENDENCY_SATISFYING_DISPOSITIONS
            ]
            if not unsatisfied:
                raise MissionRejected(
                    "this work item is not blocked by an unsatisfied dependency"
                )
            self._write_closure(conn, work_item_id, closure)
            self._bump(conn, "abandoned")
            self._record(
                conn,
                key,
                OPERATION_ABANDON_WORK_ITEM,
                payload,
                {"work_item_id": work_item_id},
            )
        return self.get_work_item(work_item_id)

    # -- worker mutations, reachable only through a live lease ----------

    def _close(self, lease: _Lease, closure: Closure, key: str) -> WorkItem:
        """Close the work item this lease authorises.

        A ``deferred`` or ``blocked`` closure has to point its revisit time at
        the future: a revisit that already happened is a silent skip wearing a
        timestamp.  A ``fixed`` closure additionally has to be *possible* - an
        item resting on a dependency that was deferred or blocked did not get
        what it was waiting for, and may not claim otherwise.
        """

        self._check_closure(closure)
        payload = self._payload_digest(
            OPERATION_CLOSE_WORK_ITEM, lease.work_item_id, lease.fence, closure.as_payload()
        )
        refusal: Optional[MissionError] = None
        with self._mutate() as txn:
            conn = txn.conn
            if self._recall(conn, key, OPERATION_CLOSE_WORK_ITEM, payload) is not None:
                return self.get_work_item(lease.work_item_id)
            item = self._authorize(txn, lease)
            if isinstance(item, MissionError):
                refusal = item
            elif closure.disposition == DISPOSITION_FIXED:
                unsatisfied = sorted(
                    dependency
                    for dependency in item.depends_on_ids
                    if self._load_item(conn, dependency).disposition
                    not in DEPENDENCY_SATISFYING_DISPOSITIONS
                )
                if unsatisfied:
                    raise MissionRejected(
                        "a work item whose dependencies did not deliver cannot close as"
                        f" fixed: {unsatisfied}"
                    )
                self._write_closure(conn, lease.work_item_id, closure)
                self._record(
                    conn,
                    key,
                    OPERATION_CLOSE_WORK_ITEM,
                    payload,
                    {"work_item_id": lease.work_item_id},
                )
            else:
                self._write_closure(conn, lease.work_item_id, closure)
                self._record(
                    conn,
                    key,
                    OPERATION_CLOSE_WORK_ITEM,
                    payload,
                    {"work_item_id": lease.work_item_id},
                )
        if refusal is not None:
            raise refusal
        return self.get_work_item(lease.work_item_id)

    def _check_closure(self, closure: Closure) -> None:
        if not isinstance(closure, Closure):
            raise MissionRejected("closure must be a Closure")
        if closure.disposition not in _ACCOUNTED_DISPOSITIONS:
            return
        revisit = closure.revisit_at
        assert revisit is not None  # guaranteed by Closure.__post_init__
        now = time.time()
        if revisit <= now:
            raise MissionRejected("revisit_at must point at the future")
        if revisit > now + MAX_REVISIT_HORIZON_SECONDS:
            raise MissionRejected("revisit_at is further out than the revisit horizon")

    def _write_closure(
        self, conn: sqlite3.Connection, work_item_id: str, closure: Closure
    ) -> None:
        conn.execute(
            "UPDATE work_items SET status = ?, disposition = ?, closure = ?, worker = NULL,"
            " dispatched_at = NULL WHERE work_item_id = ?",
            (
                STATUS_CLOSED,
                closure.disposition,
                json.dumps(closure.as_payload(), sort_keys=True, separators=(",", ":")),
                work_item_id,
            ),
        )
        conn.execute("DELETE FROM claims WHERE work_item_id = ?", (work_item_id,))
        self._bump(conn, _CLOSED_COUNTER[closure.disposition])

    def _authorize(self, txn: _Txn, lease: _Lease) -> Union[WorkItem, MissionError]:
        """Prove the caller still holds live execution authority, or refuse.

        Four separate things have to hold, and each of them is a real way the
        others can be true while the caller is not the owner:

        *This process took the lease.*  The lease object must be the very one
        this store handed out and still has registered.  A handle assembled from
        snapshot data - in this process or another - has nothing registered and
        gets no further, which is what makes an id plus a fence useless as
        authority.

        *The descriptor is still open on the same file.*  A descriptor number
        that was closed can be reissued to something else entirely, so the
        recorded device and inode are compared, not merely the number.

        *The lease path still names that file.*  Anybody who can write the store
        could unlink the lease and recreate it, leaving this process locking an
        orphaned inode while a second dispatcher locks the replacement.

        *The durable record still belongs to this era.*  Only then does the
        fencing token get to speak, and a stale one is rejected and counted.

        The refusal is *returned*, not raised.  Raising out of the transaction
        would roll back the very counter that records the refusal, so a store
        under attack would show a clean ``lease_rejects`` of zero - the one
        number an operator would look at.  Callers commit, then raise.
        """

        conn, store_fd = txn.conn, txn.store_fd
        registered = self._leases.get(lease.work_item_id)
        if lease.released or registered is not lease:
            return self._reject_lease(conn, "the execution lease is not held by this process")
        try:
            live = _Identity.of(os.fstat(lease.fd))
        except OSError as exc:
            return self._reject_lease(
                conn, f"the execution lease descriptor is no longer usable: {exc}"
            )
        if live != lease.identity:
            return self._reject_lease(conn, "the execution lease descriptor was reissued")
        try:
            on_disk = self._lease_identity(store_fd, lease.name)
        except MissionCorrupt as exc:
            return self._reject_lease(conn, f"the execution lease file is unusable: {exc}")
        if on_disk != lease.identity:
            return self._reject_lease(conn, "the execution lease file was replaced")

        item = self._load_item(conn, lease.work_item_id)
        if item.status != STATUS_DISPATCHED:
            return self._reject_lease(conn, "the work item is no longer dispatched")
        if item.fence != lease.fence:
            self._bump(conn, "stale_fence_rejects")
            return MissionStaleFence(
                "the presented fencing token was replaced by a later dispatch"
            )
        return item

    def _reject_lease(self, conn: sqlite3.Connection, reason: str) -> MissionUnauthorized:
        self._bump(conn, "lease_rejects")
        return MissionUnauthorized(reason)

    def _heartbeat(self, lease: _Lease) -> int:
        refusal: Optional[MissionError] = None
        count = 0
        with self._mutate() as txn:
            conn = txn.conn
            item = self._authorize(txn, lease)
            if isinstance(item, MissionError):
                refusal = item
            else:
                count = item.heartbeats + 1
                conn.execute(
                    "UPDATE work_items SET heartbeats = ?, last_heartbeat_at = ?"
                    " WHERE work_item_id = ?",
                    (count, time.time(), lease.work_item_id),
                )
        if refusal is not None:
            raise refusal
        return count

    def _requeue(self, lease: _Lease) -> None:
        """Put an unclosed item back, but only if this lease still owns its era."""

        with self._mutate() as txn:
            conn = txn.conn
            if isinstance(self._authorize(txn, lease), MissionError):
                # Somebody else already owns this item, or it is already closed,
                # or this lease was revoked.  Releasing a lease must never
                # disturb another era.
                return
            conn.execute(
                "UPDATE work_items SET status = ?, worker = NULL, dispatched_at = NULL"
                " WHERE work_item_id = ?",
                (STATUS_READY, lease.work_item_id),
            )
            conn.execute(
                "DELETE FROM claims WHERE work_item_id = ?", (lease.work_item_id,)
            )

    def _retire_lease(self, lease: _Lease) -> None:
        if self._leases.get(lease.work_item_id) is lease:
            del self._leases[lease.work_item_id]
        lease.released = True
        _drop_lease(lease.fd)

    def reclaim(self, work_item_id: str, *, operation: str) -> bool:
        """Requeue a dispatched item whose lease can be *proved* free.

        This is the only way a dispatch ends without its worker, and it is
        evidence-based rather than time-based: the lease is reclaimed exactly
        when this process can take the ``flock`` itself, which is impossible
        while any live process still holds it.  No TTL, no grace period, no
        guess from a heartbeat that stopped arriving.
        """

        _check_token(work_item_id, "work item id")
        key = _check_token(operation, "operation")
        payload = self._payload_digest(OPERATION_RECLAIM_WORK_ITEM, work_item_id)
        with self._mutate() as txn:
            conn = txn.conn
            if self._recall(conn, key, OPERATION_RECLAIM_WORK_ITEM, payload) is not None:
                return True
            item = self._load_item(conn, work_item_id)
            if item.status != STATUS_DISPATCHED:
                return False
            lease = self._take_lease(txn.store_fd, work_item_id)
            if lease is None:
                raise MissionConflict("the execution lease is still held by a live worker")
            try:
                conn.execute(
                    "UPDATE work_items SET status = ?, worker = NULL, dispatched_at = NULL"
                    " WHERE work_item_id = ?",
                    (STATUS_READY, work_item_id),
                )
                conn.execute("DELETE FROM claims WHERE work_item_id = ?", (work_item_id,))
                self._record(
                    conn,
                    key,
                    OPERATION_RECLAIM_WORK_ITEM,
                    payload,
                    {"work_item_id": work_item_id},
                )
            finally:
                _drop_lease(lease)
        return True

    def is_claimed(self, resource: MissionResource) -> bool:
        """Whether some running work item currently holds this resource."""

        if not isinstance(resource, MissionResource):
            raise MissionRejected("resource must be a MissionResource")
        with self._read() as txn:
            conn = None if txn is None else txn.conn
            if conn is None:
                return False
            row = conn.execute(
                "SELECT 1 FROM claims WHERE resource = ?", (resource.digest,)
            ).fetchone()
        return row is not None

    # -- observability --------------------------------------------------

    def snapshot(
        self, *, mission_id: Optional[str] = None, limit: int = DEFAULT_SNAPSHOT_LIMIT
    ) -> MissionSnapshot:
        """A bounded, secret-free, read-only view of the store.

        Carries identities, shape, status and counters, and no free text at all -
        no objective, no rationale, no evidence ref, no coordinate - so this
        surface cannot leak whatever a caller put in a mission body.  Truncation
        is reported rather than implied.
        """

        _check_int(limit, "limit", 1, MAX_SNAPSHOT_ITEMS)
        if mission_id is not None:
            _check_token(mission_id, "mission id")

        with self._read() as txn:
            conn = None if txn is None else txn.conn
            if conn is None:
                return MissionSnapshot((), (), False, self._empty_metrics())
            if mission_id is None:
                mission_rows = conn.execute(
                    "SELECT * FROM missions ORDER BY mission_id LIMIT ?", (limit + 1,)
                ).fetchall()
                item_rows = conn.execute(
                    "SELECT * FROM work_items ORDER BY work_item_id LIMIT ?", (limit + 1,)
                ).fetchall()
            else:
                mission_rows = conn.execute(
                    "SELECT * FROM missions WHERE mission_id = ?", (mission_id,)
                ).fetchall()
                item_rows = conn.execute(
                    "SELECT * FROM work_items WHERE mission_id = ?"
                    " ORDER BY work_item_id LIMIT ?",
                    (mission_id, limit + 1),
                ).fetchall()
            truncated = len(mission_rows) > limit or len(item_rows) > limit
            mission_rows = mission_rows[:limit]
            item_rows = item_rows[:limit]

            missions = []
            for row in mission_rows:
                mission = _row_to_mission(row)
                total, closed = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(status = ?), 0) FROM work_items"
                    " WHERE mission_id = ?",
                    (STATUS_CLOSED, mission.mission_id),
                ).fetchone()
                missions.append(
                    MissionSummary(
                        mission_id=mission.mission_id,
                        scope=mission.scope,
                        status=mission.status,
                        criteria_ids=mission.criteria_ids,
                        work_items=int(total),
                        closed_work_items=int(closed),
                    )
                )
            items = []
            for row in item_rows:
                item = self._item_from_row(conn, row)
                items.append(
                    WorkItemSummary(
                        work_item_id=item.work_item_id,
                        mission_id=item.mission_id,
                        scope=item.scope,
                        lane=item.lane,
                        priority=item.priority,
                        is_root=item.is_root,
                        parent_id=item.parent_id,
                        return_to_id=item.return_to_id,
                        status=item.status,
                        disposition=item.disposition,
                        fence=item.fence,
                        attempts=item.attempts,
                        heartbeats=item.heartbeats,
                        resource_count=len(item.resources),
                        dependency_count=len(item.depends_on_ids),
                    )
                )
            metrics = self._metrics(conn)
        return MissionSnapshot(tuple(missions), tuple(items), truncated, metrics)

    def scheduler_order(self, *, limit: int = DEFAULT_SNAPSHOT_LIMIT) -> Tuple[str, ...]:
        """Read the current preferred ready-work order without dispatching.

        This executes the same fairness/lane/priority query as dispatch but
        takes no lease, advances no fence, rotates no scope and writes no
        receipt.  It is therefore an observability hint, not authority and not
        a reservation: a concurrent dispatcher may change the order as soon as
        this snapshot returns.
        """

        _check_int(limit, "limit", 1, MAX_SNAPSHOT_ITEMS)
        with self._read() as txn:
            conn = None if txn is None else txn.conn
            if conn is None:
                return ()
            rows = conn.execute(_SELECT_READY, (limit,)).fetchall()
        return tuple(str(row["work_item_id"]) for row in rows)

    def metrics(self) -> MissionMetrics:
        """Bounded, secret-free counters for the whole store."""

        with self._read() as txn:
            conn = None if txn is None else txn.conn
            if conn is None:
                return self._empty_metrics()
            return self._metrics(conn)

    # -- operation receipts ---------------------------------------------

    def acknowledge_operation(self, operation: str) -> bool:
        """Release one receipt: the caller has the outcome and can forget it.

        Retention is the honest half of idempotency.  Receipts are what make an
        interrupted call answerable, so they cannot be discarded on a timer or
        by age - only a caller saying "I have this outcome" makes one safe to
        forget.  Until then it occupies part of :data:`MAX_OPERATIONS`, and a
        store whose callers never acknowledge will eventually refuse new
        operations rather than quietly dropping the evidence somebody may still
        be about to ask for.

        Returns ``False`` when the key is unknown, which is what a caller sees
        after the receipt has already been released.
        """

        key = _check_token(operation, "operation")
        released = False
        with self._mutate() as txn:
            row = txn.conn.execute(
                "SELECT acknowledged FROM operations WHERE operation_key = ?", (key,)
            ).fetchone()
            if row is not None:
                released = True
                if not row["acknowledged"]:
                    txn.conn.execute(
                        "UPDATE operations SET acknowledged = 1 WHERE operation_key = ?",
                        (key,),
                    )
        return released

    @staticmethod
    def _payload_digest(kind: str, *parts: object) -> str:
        """A canonical fingerprint of what the caller asked for.

        Two calls under one key are the same call only if they asked for the
        same thing, so the arguments are folded into a digest rather than
        compared field by field - the digest is bounded, order-stable and
        carries none of the caller's text into durable storage.
        """

        rendered = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
        return _digest(kind, rendered)

    def _recall(
        self, conn: sqlite3.Connection, key: str, kind: str, payload: str
    ) -> Optional[Dict[str, Any]]:
        """Find this operation's receipt, or refuse a key that means something else."""

        row = conn.execute(
            "SELECT kind, payload, result FROM operations WHERE operation_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        if row["kind"] != kind:
            raise MissionOperationConflict(
                f"operation {key!r} already names a {row['kind']} operation, not a {kind}"
            )
        if row["payload"] != payload:
            raise MissionOperationConflict(
                f"operation {key!r} was recorded with a different payload; a key"
                " identifies one logical operation, not a slot to overwrite"
            )
        return _decode_operation_result(row["result"], kind)

    def _record(
        self,
        conn: sqlite3.Connection,
        key: str,
        kind: str,
        payload: str,
        result: Dict[str, Any],
    ) -> None:
        """Write the receipt in the same transaction as the effect it describes."""

        self._release_capacity(conn)
        conn.execute(
            "INSERT INTO operations"
            " (operation_key, kind, payload, result, recorded_at, sequence, acknowledged)"
            " VALUES (?, ?, ?, ?, ?, ?, 0)",
            (
                key,
                kind,
                payload,
                json.dumps(result, sort_keys=True, separators=(",", ":")),
                time.time(),
                self._bump(conn, "operation_seq"),
            ),
        )

    def _release_capacity(self, conn: sqlite3.Connection) -> None:
        """Make room by discarding acknowledged receipts, oldest first.

        Never an unacknowledged one.  A receipt that nobody has collected is the
        only record of whether an interrupted call took effect, and reclaiming
        space by destroying it would trade a bounded table for an unanswerable
        question.
        """

        held = int(conn.execute("SELECT COUNT(*) FROM operations").fetchone()[0])
        if held < MAX_OPERATIONS:
            return
        surplus = held - MAX_OPERATIONS + 1
        conn.execute(
            "DELETE FROM operations WHERE operation_key IN ("
            " SELECT operation_key FROM operations WHERE acknowledged = 1"
            " ORDER BY sequence LIMIT ?)",
            (surplus,),
        )
        if int(conn.execute("SELECT COUNT(*) FROM operations").fetchone()[0]) >= MAX_OPERATIONS:
            raise MissionCapacityExceeded(
                f"the store is holding {MAX_OPERATIONS} unacknowledged operation receipts;"
                " acknowledge the outcomes you already have before starting more work"
            )

    # -- internals: lifecycle helpers -----------------------------------

    @staticmethod
    def _lease_name(work_item_id: str) -> str:
        return _digest(work_item_id) + ".lease"

    def _lease_path(self, work_item_id: str) -> Path:
        """Where the lease file lives.  For diagnostics; never used to open it."""

        return self._lease_root / self._lease_name(work_item_id)

    @contextmanager
    def _lease_directory(self, store_fd: int) -> Iterator[int]:
        descriptor = _secure_child_directory(store_fd, _LEASE_DIRNAME, create=True)
        try:
            yield descriptor
        finally:
            os.close(descriptor)

    def _take_lease(self, store_fd: int, work_item_id: str) -> Optional[int]:
        with self._lease_directory(store_fd) as lease_fd:
            return _try_lease(lease_fd, self._lease_name(work_item_id))

    def _lease_identity(self, store_fd: int, name: str) -> _Identity:
        with self._lease_directory(store_fd) as lease_fd:
            try:
                info = os.stat(name, dir_fd=lease_fd, follow_symlinks=False)
            except OSError as exc:
                raise MissionCorrupt(f"the execution lease {name} is unusable: {exc}") from exc
            if not stat.S_ISREG(info.st_mode):
                raise MissionCorrupt(f"{_NOT_A_REAL_FILE}: {name}")
            return _Identity.of(info)

    @staticmethod
    def _normalise_dependencies(depends_on_ids: Sequence[str]) -> Tuple[str, ...]:
        """Validate, deduplicate and canonically order declared dependencies.

        Canonical order matters for the same reason it matters for resources:
        two callers that declared the same dependency set in different orders
        must produce byte-identical durable state, or an audit cannot compare
        them.
        """

        if not isinstance(depends_on_ids, (list, tuple)):
            raise MissionRejected("depends_on_ids must be a sequence")
        if len(depends_on_ids) > MAX_DEPENDENCIES:
            raise MissionRejected(f"depends_on_ids exceed {MAX_DEPENDENCIES} entries")
        return tuple(
            sorted(
                {
                    _check_identifier(entry, "depends_on id", _WORK_PREFIX)
                    for entry in depends_on_ids
                }
            )
        )

    def _dependencies_of(self, conn: sqlite3.Connection, work_item_id: str) -> Tuple[str, ...]:
        rows = conn.execute(
            "SELECT depends_on_id FROM dependencies WHERE work_item_id = ?"
            " ORDER BY depends_on_id LIMIT ?",
            (work_item_id, MAX_DEPENDENCIES + 1),
        ).fetchall()
        if len(rows) > MAX_DEPENDENCIES:
            raise MissionCorrupt("a work item declares more dependencies than the bound")
        dependencies = tuple(
            _decode(row["depends_on_id"], "depends_on id", _check_identifier, _WORK_PREFIX)
            for row in rows
        )
        for dependency in dependencies:
            # A dangling dependency edge is the dangerous kind: it drops out of
            # the readiness join, so the item it gates would look free to run.
            if conn.execute(
                "SELECT 1 FROM work_items WHERE work_item_id = ?", (dependency,)
            ).fetchone() is None:
                raise MissionCorrupt("a work item depends on something that does not exist")
        return dependencies

    def _item_from_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> WorkItem:
        return _row_to_item(row, self._dependencies_of(conn, row["work_item_id"]))

    @staticmethod
    def _normalise_criteria(
        criteria: Sequence[Union[AcceptanceCriterion, Tuple[str, str]]],
    ) -> Tuple[AcceptanceCriterion, ...]:
        if not isinstance(criteria, (list, tuple)):
            raise MissionRejected("acceptance criteria must be a sequence")
        if not criteria:
            raise MissionRejected("a mission needs at least one acceptance criterion")
        if len(criteria) > MAX_ACCEPTANCE_CRITERIA:
            raise MissionRejected(
                f"acceptance criteria exceed {MAX_ACCEPTANCE_CRITERIA} entries"
            )
        normalised = []
        for entry in criteria:
            if isinstance(entry, AcceptanceCriterion):
                normalised.append(entry)
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                normalised.append(AcceptanceCriterion(entry[0], entry[1]))
            else:
                raise MissionRejected(
                    "each acceptance criterion is an AcceptanceCriterion or an (id, statement)"
                )
        ids = [item.id for item in normalised]
        if len(set(ids)) != len(ids):
            raise MissionRejected("acceptance criterion ids must be distinct")
        return tuple(normalised)

    @staticmethod
    def _validate_graph(items: Mapping[str, WorkItem]) -> None:
        """Insist the mission is still one rooted DAG with real return edges."""

        roots = [item for item in items.values() if item.is_root]
        if len(roots) != 1:
            raise MissionCorrupt("a mission has exactly one root work item")
        for item in items.values():
            if item.is_root:
                if item.parent_id is not None or item.return_to_id is not None:
                    raise MissionCorrupt("the root work item carries a parent or return edge")
                continue
            chain = _chain_to_root(items, item.work_item_id)
            if item.return_to_id not in chain[1:]:
                raise MissionCorrupt(
                    "a side work item does not return toward the root work item"
                )
        # Lineage and dependencies are separate graphs; both have to hold.
        _validate_dependency_graph(items)

    # -- internals: storage ---------------------------------------------

    def _capacity(self, conn: sqlite3.Connection) -> int:
        try:
            capacity = int(self._meta_get(conn, "queue_capacity"))
        except (TypeError, ValueError) as exc:
            raise MissionCorrupt("the durable queue capacity is malformed") from exc
        if capacity < 1 or capacity > MAX_QUEUE_CAPACITY:
            raise MissionCorrupt("the durable queue capacity is out of bounds")
        return capacity

    @staticmethod
    def _bump(conn: sqlite3.Connection, name: str) -> int:
        conn.execute("UPDATE counters SET value = value + 1 WHERE name = ?", (name,))
        row = conn.execute("SELECT value FROM counters WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise MissionCorrupt(f"the durable counter {name} is missing")
        return int(row[0])

    @staticmethod
    def _record_latency(conn: sqlite3.Connection, seconds: float) -> None:
        millis = max(0, int(seconds * 1000))
        conn.execute(
            "UPDATE counters SET value = value + ? WHERE name = 'latency_ms_total'", (millis,)
        )
        conn.execute("UPDATE counters SET value = value + 1 WHERE name = 'latency_samples'")
        conn.execute(
            "UPDATE counters SET value = MAX(value, ?) WHERE name = 'latency_ms_max'", (millis,)
        )

    def _load_mission(self, conn: sqlite3.Connection, mission_id: str) -> Mission:
        row = conn.execute(
            "SELECT * FROM missions WHERE mission_id = ?", (mission_id,)
        ).fetchone()
        if row is None:
            raise MissionRejected("no such mission")
        return _row_to_mission(row)

    def _load_item(self, conn: sqlite3.Connection, work_item_id: str) -> WorkItem:
        row = conn.execute(
            "SELECT * FROM work_items WHERE work_item_id = ?", (work_item_id,)
        ).fetchone()
        if row is None:
            raise MissionRejected("no such work item")
        return self._item_from_row(conn, row)

    def _load_items(
        self, conn: sqlite3.Connection, mission_id: str
    ) -> Dict[str, WorkItem]:
        rows = conn.execute(
            "SELECT * FROM work_items WHERE mission_id = ? ORDER BY submit_seq LIMIT ?",
            (mission_id, MAX_WORK_ITEMS_PER_MISSION + 1),
        ).fetchall()
        if len(rows) > MAX_WORK_ITEMS_PER_MISSION:
            raise MissionCorrupt("a mission holds more work items than the bound allows")
        return {row["work_item_id"]: self._item_from_row(conn, row) for row in rows}

    @staticmethod
    def _meta_get(conn: sqlite3.Connection, key: str, default: Optional[str] = None) -> str:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            if default is None:
                raise MissionCorrupt(f"the durable store setting {key} is missing")
            return default
        if not isinstance(row[0], str):
            raise MissionCorrupt(f"the durable store setting {key} is malformed")
        return row[0]

    @staticmethod
    def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _counters(self, conn: sqlite3.Connection) -> Dict[str, int]:
        """Read every durable counter, insisting that every one of them is there.

        Defaulting a missing counter to zero would turn damaged state into a
        confident, wrong metric - a store that lost its ``conflicts`` row would
        report a perfectly healthy scheduler.  A counter that is missing,
        negative, out of bounds or not an integer is corruption.
        """

        rows = conn.execute("SELECT name, value FROM counters").fetchall()
        counters: Dict[str, int] = {}
        for name, value in rows:
            if not isinstance(name, str) or name not in _COUNTERS:
                raise MissionCorrupt("the durable counters carry an unknown name")
            if isinstance(value, bool) or not isinstance(value, int):
                raise MissionCorrupt(f"the durable counter {name} is malformed")
            if value < 0 or value > MAX_SEQUENCE:
                raise MissionCorrupt(f"the durable counter {name} is out of bounds")
            counters[name] = value
        missing = sorted(set(_COUNTERS) - set(counters))
        if missing:
            raise MissionCorrupt(f"the durable counters are incomplete: {missing}")
        return counters

    def _verify_claims(self, conn: sqlite3.Connection, capacity: int) -> None:
        """Every claim must be a real digest held by a really-dispatched item."""

        bound = capacity * MAX_RESOURCES
        rows = conn.execute(
            "SELECT resource, work_item_id FROM claims LIMIT ?", (bound + 1,)
        ).fetchall()
        if len(rows) > bound:
            raise MissionCorrupt("the store holds more resource claims than the bound allows")
        for resource, work_item_id in rows:
            if not isinstance(resource, str) or len(resource) != _DIGEST_CHARS:
                raise MissionCorrupt("a stored resource claim is malformed")
            if set(resource) - _HEX_DIGITS:
                raise MissionCorrupt("a stored resource claim is not a resource digest")
            owner = _decode(work_item_id, "claim work item id", _check_identifier, _WORK_PREFIX)
            row = conn.execute(
                "SELECT status FROM work_items WHERE work_item_id = ?", (owner,)
            ).fetchone()
            if row is None or row["status"] != STATUS_DISPATCHED:
                raise MissionCorrupt("a resource claim outlives the dispatch that took it")

    def _verify_rotation(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT scope, last_dispatch_seq FROM rotation LIMIT ?", (MAX_ROTATION_SCOPES + 1,)
        ).fetchall()
        if len(rows) > MAX_ROTATION_SCOPES:
            raise MissionCorrupt("the fairness rotation carries more scopes than the bound")
        for scope, seq in rows:
            _decode(scope, "rotation scope", _check_token)
            _decode(seq, "rotation sequence", _check_int, 1, MAX_SEQUENCE)

    def _metrics(self, conn: sqlite3.Connection) -> MissionMetrics:
        # Every path that reaches a connection has already run
        # :meth:`_validate_durable_state`, so this only projects state that has
        # been proven consistent rather than re-proving it.
        counters = self._counters(conn)
        capacity = self._capacity(conn)
        queue_depth = int(
            conn.execute(
                "SELECT COUNT(*) FROM work_items WHERE status = ?", (STATUS_READY,)
            ).fetchone()[0]
        )
        dispatched = int(
            conn.execute(
                "SELECT COUNT(*) FROM work_items WHERE status = ?", (STATUS_DISPATCHED,)
            ).fetchone()[0]
        )
        created = counters["missions_created"]
        completed = counters["missions_completed"]
        if completed > created:
            raise MissionCorrupt("more missions completed than were ever created")
        return MissionMetrics(
            queue_capacity=capacity,
            queue_depth=queue_depth,
            dispatched=dispatched,
            closed_fixed=counters["closed_fixed"],
            closed_deferred=counters["closed_deferred"],
            closed_blocked=counters["closed_blocked"],
            missions_open=created - completed,
            missions_completed=completed,
            dispatches=counters["dispatches"],
            capacity_rejects=counters["capacity_rejects"],
            conflicts=counters["conflicts"],
            stale_fence_rejects=counters["stale_fence_rejects"],
            scan_truncations=counters["scan_truncations"],
            cursor_sweeps=counters["cursor_sweeps"],
            operations_retained=int(
                conn.execute("SELECT COUNT(*) FROM operations").fetchone()[0]
            ),
            lease_rejects=counters["lease_rejects"],
            abandoned=counters["abandoned"],
            submit_to_dispatch_ms_total=counters["latency_ms_total"],
            submit_to_dispatch_ms_max=counters["latency_ms_max"],
            submit_to_dispatch_samples=counters["latency_samples"],
        )

    def _empty_metrics(self) -> MissionMetrics:
        """What a store that has never been written reports: zeros, not guesses."""

        return MissionMetrics(
            queue_capacity=self._requested_capacity or DEFAULT_QUEUE_CAPACITY,
            queue_depth=0,
            dispatched=0,
            closed_fixed=0,
            closed_deferred=0,
            closed_blocked=0,
            missions_open=0,
            missions_completed=0,
            dispatches=0,
            capacity_rejects=0,
            conflicts=0,
            stale_fence_rejects=0,
            scan_truncations=0,
            cursor_sweeps=0,
            operations_retained=0,
            lease_rejects=0,
            abandoned=0,
            submit_to_dispatch_ms_total=0,
            submit_to_dispatch_ms_max=0,
            submit_to_dispatch_samples=0,
        )

    @contextmanager
    def _open_store(self, *, create: bool) -> Iterator[Optional[_Store]]:
        """Hold the store directory open, having walked to it without links.

        Every subsequent file operation is performed *relative to this
        descriptor*, so nothing this kernel touches can be redirected by
        rearranging a pathname after the walk.  The identities of the configured
        root and of the store directory are captured here so that the
        publication boundary can ask a question the descriptor cannot answer on
        its own: does the configured *path* still name this store?
        ``create=False`` creates nothing; a store that is not there yields
        ``None``.
        """

        try:
            root_fd = _open_configured_root(self._configured_root, create=create)
        except FileNotFoundError:
            if create:  # pragma: no cover - create=True makes the components
                raise
            yield None
            return
        try:
            root_identity = _Identity.of(os.fstat(root_fd))
            try:
                store_fd = _secure_child_directory(root_fd, _STORE_DIRNAME, create=create)
            except FileNotFoundError:
                yield None
                return
        finally:
            os.close(root_fd)
        try:
            yield _Store(store_fd, root_identity, _Identity.of(os.fstat(store_fd)))
        finally:
            os.close(store_fd)

    def _verify_configured_path(self, store: _Store) -> None:
        """Ask whether the configured path still names the store we are bound to.

        Descriptor-relative I/O guarantees that this transaction reads and
        writes one specific directory.  It cannot, by itself, guarantee that the
        directory is still the one the caller's path refers to - and a caller
        told "created mission m-2" when its configured path now leads to an
        empty directory has been told something false.  Worse, the next call
        would bootstrap a second store there and neither universe would know
        about the other.

        So the lexical path is walked again, without following links, and both
        identities are compared.  A mismatch is refused rather than blessed.
        """

        try:
            root_fd = _open_configured_root(self._configured_root, create=False)
        except (FileNotFoundError, MissionError) as exc:
            raise MissionDisplaced(
                f"the configured mission store path no longer resolves: {exc}"
            ) from exc
        try:
            if _Identity.of(os.fstat(root_fd)) != store.root_identity:
                raise MissionDisplaced(
                    "the configured root directory was replaced while it was in use"
                )
            try:
                store_fd = _secure_child_directory(root_fd, _STORE_DIRNAME, create=False)
            except FileNotFoundError as exc:
                raise MissionDisplaced(
                    "the configured path no longer holds a mission store"
                ) from exc
            try:
                if _Identity.of(os.fstat(store_fd)) != store.identity:
                    raise MissionDisplaced(
                        "the mission store directory was replaced while it was in use"
                    )
            finally:
                os.close(store_fd)
        finally:
            os.close(root_fd)

    def _load_database(self, store_fd: int) -> Optional[bytes]:
        """Read the database through a descriptor this kernel opened and checked.

        This is the difference between a detection story and a guarantee.  A
        pathname handed to ``sqlite3.connect`` is resolved again by SQLite, so a
        decoy installed for the length of that call and then withdrawn is served
        to the caller while every ``lstat`` before and after sees the right
        inode - a swap this kernel would never notice.  Bytes read through the
        descriptor come from the file that was opened, and no amount of
        rearranging the name afterwards changes that.
        """

        try:
            descriptor = _secure_open_at(store_fd, _DB_NAME, create=False)
        except FileNotFoundError:
            return None
        try:
            data = _read_all(descriptor, MAX_DATABASE_BYTES)
        except OSError as exc:
            raise MissionCorrupt(f"the mission database is unreadable: {exc}") from exc
        finally:
            os.close(descriptor)
        if len(data) > MAX_DATABASE_BYTES:
            raise MissionCorrupt("the mission database exceeds the database bound")
        if not data:
            raise MissionCorrupt("the mission database is empty")
        return data

    @staticmethod
    def _has_store_identity(store_fd: int) -> bool:
        try:
            descriptor = _secure_open_at(store_fd, _STORE_ID_NAME, create=False)
        except FileNotFoundError:
            return False
        os.close(descriptor)
        return True

    def _load_store_id(self, store_fd: int) -> str:
        """The identity of *this* store, kept beside the database, not inside it.

        Exactly sixty-four lowercase hexadecimal bytes and nothing else - no
        trailing newline, no surrounding whitespace, no upper case.  A file this
        kernel wrote has that shape exactly; anything else was written by
        something else, and stretching the format to accommodate it would be
        accommodating precisely the thing this file exists to detect.
        """

        try:
            descriptor = _secure_open_at(store_fd, _STORE_ID_NAME, create=False)
        except FileNotFoundError as exc:
            raise MissionCorrupt("the mission store identity is missing") from exc
        try:
            raw = _read_all(descriptor, _DIGEST_CHARS)
        finally:
            os.close(descriptor)
        if len(raw) != _DIGEST_CHARS:
            raise MissionCorrupt("the mission store identity is not the expected length")
        identity = raw.decode("ascii", "replace")
        if set(identity) - _HEX_DIGITS:
            raise MissionCorrupt("the mission store identity is malformed")
        return identity

    @staticmethod
    def _materialise(data: Optional[bytes]) -> sqlite3.Connection:
        """Build a connection over bytes, never over a name.

        ``deserialize`` is what makes the descriptor binding real, and a host
        whose SQLite cannot do it is refused rather than quietly served the
        pathname-based version with the same promises attached.  Bytes that are
        not a database surface here as :class:`MissionCorrupt`; a raw
        ``sqlite3`` error is an implementation detail no caller should have to
        catch.
        """

        if not _database_binding_supported():
            raise MissionUnsupported(
                "this host's sqlite3 cannot open a database from bytes, so a mission"
                " store cannot be bound to the file this kernel verified"
            )
        conn = sqlite3.connect(":memory:", isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            if data is not None:
                conn.deserialize(data)
            conn.execute("PRAGMA foreign_keys = ON")
            # Force SQLite to parse the header and schema *now*, so malformed
            # bytes become a typed answer here rather than an untyped one from
            # whichever query happens to touch the file first.
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlite3.DatabaseError as exc:
            conn.close()
            raise MissionCorrupt(f"the mission database is unusable: {exc}") from exc
        except BaseException:
            conn.close()
            raise
        return conn

    def _publish(self, store_fd: int, conn: sqlite3.Connection) -> None:
        """Replace the whole database atomically, or change nothing at all.

        There is no journal file and no partial write to observe: a reader
        either opens the previous database or the next one, both of which are
        complete.

        The two failure windows are deliberately given different types, because
        they mean different things to a caller.  Anything that goes wrong before
        the rename leaves the previous database in place and is a clean
        :class:`MissionHostFailure`.  A directory ``fsync`` that fails *after*
        the rename is not: the new bytes may already be visible, and the only
        honest report is :class:`MissionIndeterminate`.  Swallowing that error
        would be claiming crash durability on the strength of an I/O error that
        said the opposite.
        """

        data = conn.serialize()
        if data is None or len(data) > MAX_DATABASE_BYTES:  # pragma: no cover - bounded
            raise MissionRejected("the mission database exceeds the database bound")
        try:
            os.unlink(_STAGED_NAME, dir_fd=store_fd)
        except FileNotFoundError:
            pass
        flags = (
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(_STAGED_NAME, flags, FILE_MODE, dir_fd=store_fd)
        except OSError as exc:
            raise MissionHostFailure(f"the mission store could not be staged: {exc}") from exc
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException as exc:
            try:
                os.unlink(_STAGED_NAME, dir_fd=store_fd)
            except OSError:  # pragma: no cover - best-effort cleanup
                pass
            if isinstance(exc, OSError):
                raise MissionHostFailure(
                    f"the mission store could not be written: {exc}"
                ) from exc
            raise
        try:
            os.replace(_STAGED_NAME, _DB_NAME, src_dir_fd=store_fd, dst_dir_fd=store_fd)
        except OSError as exc:
            try:
                os.unlink(_STAGED_NAME, dir_fd=store_fd)
            except OSError:  # pragma: no cover - best-effort cleanup
                pass
            raise MissionHostFailure(
                f"the mission store could not be published: {exc}"
            ) from exc
        try:
            os.fsync(store_fd)
        except OSError as exc:
            raise MissionIndeterminate(
                "the mission store was renamed into place but its directory could not be"
                f" synced, so the change may be visible without being durable: {exc}"
            ) from exc

    @contextmanager
    def _mutate(self) -> Iterator["_Txn"]:
        """One store-wide lock, one transaction, one atomic publication.

        The order of the last three steps is the contract: validate the whole
        durable state before touching it, verify the configured path still names
        this store, publish, and verify once more before the caller is allowed
        to believe any of it.
        """

        _require_supported_host()
        with self._open_store(create=True) as store:
            assert store is not None  # create=True either opens or raises
            with _exclusive_directory(store):
                data = self._load_database(store.fd)
                if data is None and self._has_store_identity(store.fd):
                    # Bootstrap residue: an earlier attempt wrote the identity
                    # and then failed or died before publishing the database.
                    # Reinventing the identity here would quietly fork the
                    # store, so this is damage for a host to look at.
                    raise MissionCorrupt(
                        "the mission store holds an identity but no database, so an"
                        " earlier initialisation did not complete"
                    )
                conn = self._materialise(data)
                bootstrapped = False
                published = False
                try:
                    try:
                        conn.execute("BEGIN IMMEDIATE")
                        if data is None:
                            self._bootstrap(conn, store.fd)
                            bootstrapped = True
                        self._validate_durable_state(conn, store.fd)
                        yield _Txn(conn, store.fd)
                        conn.execute("COMMIT")
                    except BaseException:
                        try:
                            conn.execute("ROLLBACK")
                        except sqlite3.DatabaseError:  # pragma: no cover
                            pass
                        raise
                    self._verify_configured_path(store)
                    try:
                        self._publish(store.fd, conn)
                    except MissionIndeterminate:
                        # The rename happened; only the proof of durability did
                        # not.  Whatever this call created is now part of a
                        # publication somebody may already be reading, so the
                        # rollback below must not touch it.
                        published = True
                        raise
                    published = True
                    try:
                        self._verify_configured_path(store)
                    except MissionDisplaced as exc:
                        raise MissionIndeterminate(
                            "the mission store was published into the directory this"
                            " transaction was bound to, but the configured path stopped"
                            f" naming it, so this result cannot be reported: {exc}"
                        ) from exc
                except sqlite3.DatabaseError as exc:
                    raise MissionCorrupt(f"the mission database is unusable: {exc}") from exc
                except BaseException:
                    if bootstrapped and not published:
                        # Only this attempt's own residue, and only when nothing
                        # was published: a crash leaves the identity behind on
                        # purpose, so the next open can see it.
                        try:
                            os.unlink(_STORE_ID_NAME, dir_fd=store.fd)
                        except OSError:  # pragma: no cover - best-effort rollback
                            pass
                    raise
                finally:
                    conn.close()

    @contextmanager
    def _read(self) -> Iterator[Optional["_Txn"]]:
        """Read without creating, locking or re-moding anything.

        An unsupported host is refused here too, and for a reason worth stating:
        a store that cannot be opened is not the same thing as a store that is
        empty.  Reporting "queue depth 0, no missions, nothing amiss" from a host
        that could never have read a store anyway would be a confident answer
        assembled out of an inability to look - the same mistake as defaulting a
        missing counter to zero, and this module refuses it in both places.  One
        rule covers reads and mutations alike: :func:`inspect_host` is how a
        caller asks without being refused.
        """

        _require_supported_host()
        with self._open_store(create=False) as store:
            if store is None:
                yield None
                return
            data = self._load_database(store.fd)
            if data is None:
                if self._has_store_identity(store.fd):
                    raise MissionCorrupt(
                        "the mission store holds an identity but no database, so an"
                        " earlier initialisation did not complete"
                    )
                yield None
                return
            conn = self._materialise(data)
            try:
                self._validate_durable_state(conn, store.fd)
                yield _Txn(conn, store.fd)
            except sqlite3.DatabaseError as exc:
                raise MissionCorrupt(f"the mission database is unusable: {exc}") from exc
            finally:
                conn.close()

    def _bootstrap(self, conn: sqlite3.Connection, store_fd: int) -> None:
        """Create the schema, but only for a store that genuinely has none.

        Bootstrap and validation are deliberately separate.  Creating anything
        that is merely *missing* would turn every kind of damage into a silent
        migration: a store that lost its ``fence`` counter would have it
        recreated at zero and would then reissue a fencing token it had already
        issued, which is the one thing a fencing token exists to prevent.
        """

        for statement in _SCHEMA:
            conn.execute(statement)
        for name in _COUNTERS:
            conn.execute("INSERT INTO counters (name, value) VALUES (?, 0)", (name,))
        store_id = _digest(os.urandom(32).hex())
        self._meta_set(conn, "schema_version", str(SCHEMA_VERSION))
        self._meta_set(conn, "queue_capacity", str(self._configured_capacity()))
        self._meta_set(conn, "store_id", store_id)
        self._meta_set(conn, "dispatch_cursor", "")
        descriptor = _secure_open_at(store_fd, _STORE_ID_NAME, create=True)
        try:
            os.ftruncate(descriptor, 0)
            os.write(descriptor, store_id.encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _validate_schema(self, conn: sqlite3.Connection, store_fd: int) -> None:
        """Insist an existing store is *exactly* v1.  Never repair it in passing.

        Exactly, not "at least": an unknown table, view, trigger, index or
        durable setting is state this kernel did not write, cannot reason about
        and would otherwise carry forward on every publication - a foreign
        schema riding along inside a store that reports itself healthy.  There
        is no migration path here by design; a v1 store either is one or is
        corrupt.  The expected shape is derived from the schema statements
        themselves, so the implicit indexes SQLite creates are covered without
        anyone having to remember to list them.
        """

        try:
            present = frozenset(
                (str(row["type"]), str(row["name"]))
                for row in conn.execute("SELECT type, name FROM sqlite_master").fetchall()
            )
        except sqlite3.DatabaseError as exc:
            raise MissionCorrupt(f"the mission database has no schema: {exc}") from exc
        expected = _expected_schema_objects()
        missing = sorted(name for _, name in expected - present)
        if missing:
            raise MissionCorrupt(f"the mission database is missing {missing}")
        unknown = sorted(f"{kind} {name}" for kind, name in present - expected)
        if unknown:
            raise MissionCorrupt(f"the mission database carries unknown objects {unknown}")

        keys = frozenset(
            str(row["key"]) for row in conn.execute("SELECT key FROM meta").fetchall()
        )
        if keys != _META_KEYS:
            absent = sorted(_META_KEYS - keys)
            extra = sorted(keys - _META_KEYS)
            raise MissionCorrupt(
                f"the mission database settings are wrong (missing: {absent},"
                f" unknown: {extra})"
            )

        version = self._meta_get(conn, "schema_version")
        if version != str(SCHEMA_VERSION):
            raise MissionCorrupt(
                f"the mission database is version {version!r}, not {SCHEMA_VERSION}"
            )
        # The identity lives in two places on purpose.  Inside the database it
        # travels with a copy; beside it, in a file this kernel opened through
        # the same verified descriptor, it does not.  A foreign database left at
        # the pathname therefore fails to be this store rather than being read
        # as though it were.
        if self._meta_get(conn, "store_id") != self._load_store_id(store_fd):
            raise MissionCorrupt(
                "the database at the store pathname belongs to a different store"
            )
        durable = self._capacity(conn)
        if self._requested_capacity is not None and self._requested_capacity != durable:
            raise MissionRejected(
                "the durable queue capacity is "
                f"{durable}, not the requested {self._requested_capacity};"
                " capacity belongs to the store, not to one process"
            )

    def _validate_durable_state(self, conn: sqlite3.Connection, store_fd: int) -> None:
        """Validate the *whole* store before anything is allowed to build on it.

        Decoding every row is not enough and never was.  What makes a scheduler
        wrong is rarely a malformed field; it is two tables that no longer agree
        - a resource still claimed by nobody, a completed mission whose root
        never ran, a fairness rotation that forgot a tenant.  Every one of those
        is *derivable* from the rows themselves, so every one of them is checked
        here, and anything that cannot be derived (how many conflicts were seen,
        how many callers were rate-limited) is bounded rather than invented.

        Whole-file publication already costs O(database size) and the database
        is bounded at :data:`MAX_DATABASE_BYTES`, so a complete pass is finite
        and roughly free next to the write it guards.
        """

        self._validate_schema(conn, store_fd)
        counters = self._counters(conn)
        capacity = self._capacity(conn)
        self._verify_referential_integrity(conn, capacity)
        missions = self._validate_missions(conn, counters)
        edges = self._load_all_dependencies(conn, capacity)
        grouped, items = self._validate_work_items(conn, counters, missions, edges)
        self._validate_dependencies(edges, items)
        self._validate_operations(conn, counters, missions, items)
        self._verify_claims(conn, items, capacity)
        self._verify_rotation(conn, counters, items)
        self._validate_cursor(conn, items)
        for mission_id, mission_items in grouped.items():
            self._validate_graph(mission_items)
            if missions[mission_id].status == MISSION_COMPLETED:
                self._validate_completed(missions[mission_id], mission_items)
        for mission in missions.values():
            if mission.status == MISSION_COMPLETED and mission.mission_id not in grouped:
                raise MissionCorrupt(
                    "a completed mission holds no work items, so nothing was ever done"
                )

    @staticmethod
    def _verify_referential_integrity(conn: sqlite3.Connection, capacity: int) -> None:
        """Ask SQLite itself whether any declared reference is dangling.

        Row-by-row checks only ever see the rows they thought to look at: a
        ``dependencies`` entry whose *source* is a work item that does not exist
        is reachable from neither the work items nor the missions, so it stays
        invisible to every traversal that starts from real data.  The foreign
        keys are declared; this makes the database prove they hold.
        """

        bound = capacity * (MAX_DEPENDENCIES + MAX_RESOURCES) + 1
        rows = conn.execute(
            "SELECT * FROM pragma_foreign_key_check LIMIT ?", (bound,)
        ).fetchall()
        if rows:
            raise MissionCorrupt(
                f"the mission database holds {len(rows)} dangling reference(s), the first"
                f" in table {rows[0][0]!r}"
            )

    @staticmethod
    def _validate_missions(
        conn: sqlite3.Connection, counters: Dict[str, int]
    ) -> Dict[str, Mission]:
        missions: Dict[str, Mission] = {}
        for row in conn.execute("SELECT * FROM missions ORDER BY mission_id").fetchall():
            mission = _row_to_mission(row)
            missions[mission.mission_id] = mission
        completed = sum(
            1 for mission in missions.values() if mission.status == MISSION_COMPLETED
        )
        for label, expected in (
            ("mission_seq", len(missions)),
            ("missions_created", len(missions)),
            ("missions_completed", completed),
        ):
            if counters[label] != expected:
                raise MissionCorrupt(
                    f"the durable counter {label} disagrees with the stored missions"
                )
        # Ids are minted from a counter, one at a time, and never removed, so
        # the set of them is not merely "the right size" - it is exactly the
        # first N.  A renumbered or reused id would otherwise pass every
        # per-row check while pointing lineage and evidence at the wrong work.
        expected_ids = {
            _MISSION_PREFIX + "%012d" % index for index in range(1, len(missions) + 1)
        }
        if set(missions) != expected_ids:
            raise MissionCorrupt("the stored mission ids are not the ones that were minted")
        return missions

    @staticmethod
    def _load_all_dependencies(
        conn: sqlite3.Connection, capacity: int
    ) -> Dict[str, Tuple[str, ...]]:
        """Read every dependency edge in one query.

        Decoding a work item on its own asks the database for that item's
        dependencies, and asking once per row turns a whole-store pass into a
        query per item plus a query per edge.  The full pass reads the table once
        instead; the per-item path stays available for the single-item callers
        that genuinely want one row.
        """

        bound = capacity * MAX_DEPENDENCIES + 1
        rows = conn.execute(
            "SELECT work_item_id, depends_on_id FROM dependencies"
            " ORDER BY work_item_id, depends_on_id LIMIT ?",
            (bound,),
        ).fetchall()
        if len(rows) >= bound:
            raise MissionCorrupt("the store holds more dependencies than the bound allows")
        edges: Dict[str, list] = {}
        for source, target in rows:
            edges.setdefault(
                _decode(source, "depends_on source", _check_identifier, _WORK_PREFIX), []
            ).append(_decode(target, "depends_on id", _check_identifier, _WORK_PREFIX))
        for source, targets in edges.items():
            if len(targets) > MAX_DEPENDENCIES:
                raise MissionCorrupt("a work item declares more dependencies than the bound")
        return {source: tuple(targets) for source, targets in edges.items()}

    def _validate_work_items(
        self,
        conn: sqlite3.Connection,
        counters: Dict[str, int],
        missions: Dict[str, Mission],
        edges: Dict[str, Tuple[str, ...]],
    ) -> Tuple[Dict[str, Dict[str, WorkItem]], Dict[str, WorkItem]]:
        grouped: Dict[str, Dict[str, WorkItem]] = {}
        items: Dict[str, WorkItem] = {}
        closed = {disposition: 0 for disposition in DISPOSITIONS}
        submit_seqs = set()
        fences = []
        attempts = 0
        abandoned = 0
        for row in conn.execute("SELECT * FROM work_items ORDER BY work_item_id").fetchall():
            item = _row_to_item(row, edges.get(str(row["work_item_id"]), ()))
            if item.mission_id not in missions:
                raise MissionCorrupt("a work item belongs to no stored mission")
            if item.scope != missions[item.mission_id].scope:
                raise MissionCorrupt("a work item disagrees with its mission's scope")
            if item.fence > counters["fence"]:
                raise MissionCorrupt("a work item carries a fencing token never issued")
            if item.attempts > counters["dispatches"]:
                raise MissionCorrupt("a work item records dispatches that never happened")
            grouped.setdefault(item.mission_id, {})[item.work_item_id] = item
            items[item.work_item_id] = item
            submit_seqs.add(item.submit_seq)
            attempts += item.attempts
            if item.fence:
                fences.append(item.fence)
            if item.closure is not None:
                closed[item.closure.disposition] += 1
                if not item.attempts:
                    # Closed without ever having run: only the host abandon path
                    # can produce this, and only with an accounted disposition.
                    abandoned += 1
                    if item.closure.disposition == DISPOSITION_FIXED:
                        raise MissionCorrupt(
                            "a work item that never ran is recorded as fixed"
                        )

        total = len(items)
        expected_ids = {_WORK_PREFIX + "%012d" % index for index in range(1, total + 1)}
        if set(items) != expected_ids:
            raise MissionCorrupt("the stored work item ids are not the ones that were minted")
        if submit_seqs != set(range(1, total + 1)):
            raise MissionCorrupt("the stored submission sequence is not exactly 1..N")
        # One token per dispatch, one dispatch per attempt: these four counters
        # and the attempts recorded on the rows are five views of one number.
        for label, expected in (
            ("work_seq", total),
            ("submit_seq", total),
            ("closed_fixed", closed[DISPOSITION_FIXED]),
            ("closed_deferred", closed[DISPOSITION_DEFERRED]),
            ("closed_blocked", closed[DISPOSITION_BLOCKED]),
            ("abandoned", abandoned),
            ("dispatches", attempts),
            ("dispatch_seq", attempts),
            ("fence", attempts),
            ("latency_samples", attempts),
        ):
            if counters[label] != expected:
                raise MissionCorrupt(
                    f"the durable counter {label} disagrees with the stored work items"
                )
        if len(set(fences)) != len(fences):
            raise MissionCorrupt("two work items carry the same fencing token")
        self._validate_latency(counters)
        return grouped, items

    @staticmethod
    def _validate_latency(counters: Dict[str, int]) -> None:
        samples = counters["latency_samples"]
        total = counters["latency_ms_total"]
        largest = counters["latency_ms_max"]
        if samples == 0:
            if total or largest:
                raise MissionCorrupt("latency was recorded for dispatches that never happened")
        elif largest > total:
            raise MissionCorrupt("the largest recorded latency exceeds the total")

    @staticmethod
    def _validate_dependencies(
        edges: Dict[str, Tuple[str, ...]], items: Dict[str, WorkItem]
    ) -> None:
        for source, targets in edges.items():
            if source not in items:
                raise MissionCorrupt("a dependency names a work item that is not stored")
            for target in targets:
                if target not in items:
                    raise MissionCorrupt("a dependency names a work item that is not stored")
                if items[source].mission_id != items[target].mission_id:
                    raise MissionCorrupt("a dependency crosses missions")

    @staticmethod
    def _validate_operations(
        conn: sqlite3.Connection,
        counters: Dict[str, int],
        missions: Dict[str, Mission],
        items: Dict[str, WorkItem],
    ) -> None:
        """Receipts are durable state, so they are held to the durable standard.

        A receipt is what an interrupted caller is told to trust, which makes a
        wrong one worse than none: it would answer "yes, that happened" about
        work that did not, or point at an id nothing else knows about.  Shape,
        bounds, kind, sequence and the subject it names are all checked, and the
        recorded outcome must still agree with the state it claims to describe.
        """

        rows = conn.execute(
            "SELECT operation_key, kind, payload, result, recorded_at, sequence,"
            " acknowledged FROM operations ORDER BY sequence LIMIT ?",
            (MAX_OPERATIONS + 1,),
        ).fetchall()
        if len(rows) > MAX_OPERATIONS:
            raise MissionCorrupt("the store holds more operation receipts than the bound")
        sequences = set()
        for row in rows:
            _decode(row["operation_key"], "operation key", _check_token)
            kind = row["kind"]
            if kind not in OPERATION_KINDS:
                raise MissionCorrupt("a stored operation receipt has an unknown kind")
            payload = row["payload"]
            if (
                not isinstance(payload, str)
                or len(payload) != _DIGEST_CHARS
                or set(payload) - _HEX_DIGITS
            ):
                raise MissionCorrupt("a stored operation payload is not a digest")
            _decode_time(row["recorded_at"], "operation time")
            acknowledged = row["acknowledged"]
            if isinstance(acknowledged, bool) or acknowledged not in (0, 1):
                raise MissionCorrupt("a stored operation acknowledgement is malformed")
            sequence = _decode(
                row["sequence"], "operation sequence", _check_int, 1, counters["operation_seq"]
            )
            if sequence in sequences:  # pragma: no cover - UNIQUE covers it
                raise MissionCorrupt("two operation receipts share a sequence")
            sequences.add(sequence)

            result = _decode_operation_result(row["result"], kind)
            subject = _OPERATION_SUBJECT[kind]
            known = missions if subject == "mission_id" else items
            if result[subject] not in known:
                raise MissionCorrupt(
                    "an operation receipt names something the store does not hold"
                )
            if kind == OPERATION_DISPATCH and result["fence"] > counters["fence"]:
                raise MissionCorrupt(
                    "an operation receipt records a fencing token never issued"
                )
            if kind == OPERATION_COMPLETE_MISSION and (
                missions[result[subject]].status != MISSION_COMPLETED
            ):
                raise MissionCorrupt(
                    "an operation receipt records a completion that did not happen"
                )

    @staticmethod
    def _verify_claims(
        conn: sqlite3.Connection, items: Dict[str, WorkItem], capacity: int
    ) -> None:
        """The claims table must be *exactly* what the dispatched items declare.

        Checking each row on its own leaves the two failures that matter
        unreachable.  A claim that was deleted lets a second work item be
        dispatched onto a resource somebody is still holding - the exclusion is
        simply gone, and every individual surviving row still looks perfect.  A
        claim that was added blocks work for a resource nobody declared.  Only
        an exact bijection between "resources the dispatched items say they
        hold" and "rows in the claims table" rules both out.
        """

        bound = capacity * MAX_RESOURCES
        rows = conn.execute(
            "SELECT resource, work_item_id FROM claims LIMIT ?", (bound + 1,)
        ).fetchall()
        if len(rows) > bound:
            raise MissionCorrupt("the store holds more resource claims than the bound allows")
        recorded = set()
        for resource, work_item_id in rows:
            if not isinstance(resource, str) or len(resource) != _DIGEST_CHARS:
                raise MissionCorrupt("a stored resource claim is malformed")
            if set(resource) - _HEX_DIGITS:
                raise MissionCorrupt("a stored resource claim is not a resource digest")
            owner = _decode(work_item_id, "claim work item id", _check_identifier, _WORK_PREFIX)
            recorded.add((resource, owner))
        if len(recorded) != len(rows):  # pragma: no cover - resource is the primary key
            raise MissionCorrupt("the claims table repeats a resource")

        expected = {
            (resource.digest, item.work_item_id)
            for item in items.values()
            if item.status == STATUS_DISPATCHED
            for resource in item.resources
        }
        if recorded != expected:
            missing = len(expected - recorded)
            extra = len(recorded - expected)
            raise MissionCorrupt(
                "the resource claims do not match the dispatched work items"
                f" ({missing} missing, {extra} unaccounted)"
            )

    @staticmethod
    def _verify_rotation(
        conn: sqlite3.Connection, counters: Dict[str, int], items: Dict[str, WorkItem]
    ) -> None:
        rows = conn.execute(
            "SELECT scope, last_dispatch_seq FROM rotation LIMIT ?", (MAX_ROTATION_SCOPES + 1,)
        ).fetchall()
        if len(rows) > MAX_ROTATION_SCOPES:
            raise MissionCorrupt("the fairness rotation carries more scopes than the bound")
        sequences = []
        scopes = set()
        for scope, seq in rows:
            scopes.add(_decode(scope, "rotation scope", _check_token))
            sequences.append(
                _decode(seq, "rotation sequence", _check_int, 1, counters["dispatch_seq"])
            )
        if len(sequences) != len(set(sequences)):
            raise MissionCorrupt("two scopes share a dispatch position in the rotation")
        # The rotation is what stops one tenant monopolising dispatch, so a
        # forgotten scope is not cosmetic: it reads as a tenant that has never
        # been served and jumps the queue for as long as the loss goes unnoticed.
        served = {item.scope for item in items.values() if item.attempts}
        if scopes != served:
            raise MissionCorrupt(
                "the fairness rotation does not match the scopes that have been served"
            )

    def _validate_cursor(self, conn: sqlite3.Connection, items: Dict[str, WorkItem]) -> None:
        cursor = self._meta_get(conn, "dispatch_cursor")
        if not cursor:
            return
        identifier = _decode(cursor, "dispatch cursor", _check_identifier, _WORK_PREFIX)
        if identifier not in items:
            raise MissionCorrupt("the dispatch cursor names a work item that is not stored")

    @staticmethod
    def _validate_completed(mission: Mission, items: Dict[str, WorkItem]) -> None:
        """A completed mission has to look like one, not merely say so.

        The status, the completion time and the evidence are three fields a
        tamper can set in one statement.  What it cannot fake without doing the
        work is the state of the work items: every one closed, exactly one root,
        and that root ``fixed``.  Side issues may legitimately be deferred or
        blocked with their full accounting - the main axis may not.
        """

        open_items = sorted(
            item.work_item_id for item in items.values() if item.status != STATUS_CLOSED
        )
        if open_items:
            raise MissionCorrupt(
                f"a completed mission still holds open work items: {open_items}"
            )
        roots = [item for item in items.values() if item.is_root]
        if len(roots) != 1:  # pragma: no cover - _validate_graph gets here first
            raise MissionCorrupt("a completed mission does not have exactly one root")
        if roots[0].disposition != DISPOSITION_FIXED:
            raise MissionCorrupt(
                f"a completed mission has a {roots[0].disposition} root, so its objective"
                " was never reached"
            )

    def _configured_capacity(self) -> int:
        return self._requested_capacity or DEFAULT_QUEUE_CAPACITY
