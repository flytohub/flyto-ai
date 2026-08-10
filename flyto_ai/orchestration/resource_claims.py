# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Durable, multi-process claims over arbitrary named resources.

This kernel is deliberately domain-neutral.  It knows only that a *claim*
binds an *owner identity* to a *resource*, and that the owner carries some
mutable *lifecycle state* while it holds that claim.  It has no vocabulary of
its own: no sanctioned namespaces, no enumerated kinds, no list of blessed
lifecycle tokens, and no branch on what a caller's name happens to spell.
Callers name their own namespace, kind and identity, choose their own state
tokens, and supply an :class:`OwnerAuthority` that resolves the claim against
whatever state authoritatively owns the truth in their domain.

Two separations carry the whole design:

*Identity versus lifecycle.*  :class:`OwnerRef` is the identity - a scope and
an id - and it is the only thing ``reassert`` and ``release`` compare.  The
lifecycle token passed alongside it is mutable metadata: one owner may move
through queued, running, awaiting-review or any other caller-chosen token
without ever becoming a different owner.  A *different* identity conflicts, no
matter which token it presents.

*Record versus truth.*  A stored record is evidence that somebody claimed the
resource; it is never proof that they still hold it.  :meth:`inspect` reports
what is written down.  :meth:`resolve` additionally consults the caller's
authority, which is what makes a genuinely missing owner detectable instead of
being guessed at from a damaged file.

Everything else follows from failing closed:

* Lookup is content-addressed.  Only the SHA-256 digest of the resource triple
  reaches the filesystem, so a claim path leaks no caller vocabulary.
* Records use one exact, bounded schema.  Malformed bytes, an unknown version,
  a record bound to another resource, transition evidence that lags its own
  sequence counter, an unreadable file - all resolve to ``unresolved``, and an
  ``unresolved`` record is never deleted automatically.  Reading one is bounded
  before it is parsed: the claim path is opened with ``O_NOFOLLOW``, refused
  unless it is a regular file, and read only one byte past
  :data:`MAX_RECORD_BYTES`, so neither a planted symbolic link nor an enormous
  file can make a reader leave the store or load what it is about to reject.
* An authority that raises, or answers ``unknown``, is ambiguous.  Ambiguity
  never releases a claim; it needs an explicit host repair.
* Exclusion is atomic across processes.  A claim is published by an exclusive
  hard link over fully fsynced content, guarded by a POSIX ``flock`` and by a
  process-local lock, so a reader never sees a partial write and two processes
  never both win.
* Storage is private to the account that owns it, and stays inside the store
  it was pointed at.  Every directory this kernel creates is created at
  :data:`DIRECTORY_MODE`, and a directory it finds already standing is
  tightened to that mode *before* the operation is allowed to touch anything
  inside it - an inherited group- or world-writable claim root would otherwise
  let a bystander delete a live claim or substitute a record of their own.
  Claim and lock files are :data:`FILE_MODE`, and a lock file found at a wider
  mode is tightened through its own descriptor before it is locked.  A root,
  shard or lock path that turns out to be a symbolic link is refused rather
  than followed, so nothing outside the configured store is ever re-moded or
  written.  If the identity or the mode of any of them cannot be established
  and verified, the operation fails closed with :class:`ResourceClaimError`
  rather than storing a claim somewhere readable.
* A host that cannot supply those primitives is refused rather than served a
  guarantee this kernel cannot keep.  Without a working inter-process lock
  every mutating operation raises :class:`ResourceClaimError`, because a
  process-local lock alone would let a second process win the same claim.
  Without atomic hard-link publication :meth:`ResourceClaimStore.acquire`
  raises :class:`ResourceClaimError` and leaves no claim behind; bytes are
  never copied into the final path, so no reader ever observes a partial
  record.  Read-only :meth:`ResourceClaimStore.inspect` and
  :meth:`ResourceClaimStore.resolve` keep working on such a host: reporting
  what is written down needs no exclusion, creates no directory and changes no
  mode, so it is also the one path that stays usable on a store the caller can
  only read.
* There is no daemon, no heartbeat and no TTL.  Time is never consulted, so
  nothing is ever guessed from a clock.

Durable data is secret-free by contract.  Only validated, printable, bounded
identifiers and non-sensitive lifecycle tokens are persisted.  **Callers must
not pass credentials, API keys, bearer tokens, cookies or personal data in any
field.**  Pass opaque handles or digests of those values instead; this module
stores what it is given and cannot tell a tenant digest from a password.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Optional, Protocol, Sequence, Tuple, Union

try:  # pragma: no cover - import shape differs only on non-POSIX hosts
    import fcntl
except ImportError:  # pragma: no cover - Windows and friends
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "DIRECTORY_MODE",
    "FILE_MODE",
    "MAX_FIELD_CHARS",
    "MAX_RECORD_BYTES",
    "MAX_SEQUENCE",
    "MAX_TRANSITIONS",
    "RECORD_VERSION",
    "VERDICTS",
    "VERDICT_HELD",
    "VERDICT_MISSING",
    "VERDICT_RELEASED",
    "VERDICT_UNKNOWN",
    "ClaimInspection",
    "ClaimOutcome",
    "ClaimRecord",
    "ClaimResolution",
    "ClaimStatus",
    "ClaimTransition",
    "OwnerAuthority",
    "OwnerRef",
    "OwnerVerdict",
    "ResolvedStatus",
    "ResourceClaimConflict",
    "ResourceClaimError",
    "ResourceClaimRejected",
    "ResourceClaimStore",
    "ResourceClaimUnresolved",
    "ResourceRef",
]

RECORD_VERSION = 2
MAX_FIELD_CHARS = 256
MAX_RECORD_BYTES = 4096
MAX_SEQUENCE = 2**32
MAX_TRANSITIONS = 8

#: Mode for every directory this kernel creates or adopts: owner-only.
DIRECTORY_MODE = 0o700
#: Mode for every file this kernel creates: owner-only.
FILE_MODE = 0o600

_DOMAIN = b"flyto-ai/resource-claim/1"
_DIGEST_CHARS = 64
_HEX = frozenset("0123456789abcdef")
_RECORD_KEYS = frozenset(
    {"version", "binding", "owner_scope", "owner_id", "state", "sequence", "transitions"}
)
_TRANSITION_KEYS = frozenset({"sequence", "state"})

#: What is written down about a resource, with no authority consulted.
ClaimStatus = Literal["free", "held", "unresolved"]

#: What the caller's authority says about the owner named in a record.
OwnerVerdict = Literal["held", "released", "missing", "unknown"]

#: What is written down, reconciled against the caller's authority.
ResolvedStatus = Literal["free", "held", "released", "missing", "unresolved"]

VERDICT_HELD: OwnerVerdict = "held"
VERDICT_RELEASED: OwnerVerdict = "released"
VERDICT_MISSING: OwnerVerdict = "missing"
VERDICT_UNKNOWN: OwnerVerdict = "unknown"
VERDICTS: Tuple[OwnerVerdict, ...] = (
    VERDICT_HELD,
    VERDICT_RELEASED,
    VERDICT_MISSING,
    VERDICT_UNKNOWN,
)


class ResourceClaimError(Exception):
    """Base class for every failure this kernel raises."""


class ResourceClaimRejected(ResourceClaimError):
    """A caller-supplied value is outside the accepted bounds."""


class ResourceClaimConflict(ResourceClaimError):
    """The resource is claimed by an identity other than the calling one."""


class ResourceClaimUnresolved(ResourceClaimError):
    """A stored claim could not be resolved, so the operation failed closed."""


def _check_token(value: object, label: str) -> str:
    """Validate one durable field.

    Accepts a bounded, printable, whitespace-free string.  That rules out
    control characters, tabs and newlines, line and paragraph separators, and
    any other non-printable code point that could smuggle structure into a
    record, a log line or a path, and it keeps every stored value a single
    unambiguous token with no leading, trailing or interior padding to argue
    about.  It cannot, and does not try to, tell a safe identifier from a
    secret - see the module docstring.
    """

    if not isinstance(value, str):
        raise ResourceClaimRejected(f"{label} must be a string")
    if not value:
        raise ResourceClaimRejected(f"{label} must not be empty")
    if len(value) > MAX_FIELD_CHARS:
        raise ResourceClaimRejected(f"{label} exceeds {MAX_FIELD_CHARS} characters")
    if not value.isprintable():
        raise ResourceClaimRejected(f"{label} contains a non-printable character")
    if any(char.isspace() for char in value):
        raise ResourceClaimRejected(f"{label} contains whitespace")
    return value


def _check_sequence(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResourceClaimRejected(f"{label} must be an integer")
    if value < 1 or value > MAX_SEQUENCE:
        raise ResourceClaimRejected(f"{label} is out of bounds")
    return value


def _digest(*parts: str) -> str:
    """Length-prefixed digest so distinct triples cannot collide by joining."""

    accumulator = hashlib.sha256()
    accumulator.update(_DOMAIN)
    for part in parts:
        raw = part.encode("utf-8")
        accumulator.update(b"\n")
        accumulator.update(str(len(raw)).encode("ascii"))
        accumulator.update(b":")
        accumulator.update(raw)
    return accumulator.hexdigest()


@dataclass(frozen=True)
class ResourceRef:
    """An arbitrary bounded resource coordinate: namespace, kind, identity."""

    namespace: str
    kind: str
    identity: str

    def __post_init__(self) -> None:
        _check_token(self.namespace, "namespace")
        _check_token(self.kind, "kind")
        _check_token(self.identity, "identity")

    @property
    def digest(self) -> str:
        """Content address of this resource; the only form written to disk."""

        return _digest(self.namespace, self.kind, self.identity)


@dataclass(frozen=True)
class OwnerRef:
    """*Who* holds a claim, and nothing about *how* they are getting on.

    Identity is exactly ``(scope, id)``.  Lifecycle state is deliberately not a
    field here: it is passed alongside an :class:`OwnerRef` to
    :meth:`ResourceClaimStore.acquire` and :meth:`ResourceClaimStore.reassert`,
    stored as mutable metadata, and never compared when deciding whether a
    caller is the owner on record.
    """

    scope: str
    id: str

    def __post_init__(self) -> None:
        _check_token(self.scope, "owner scope")
        _check_token(self.id, "owner id")

    @property
    def digest(self) -> str:
        """Content address of this identity, for callers that want one."""

        return _digest(self.scope, self.id)


@dataclass(frozen=True)
class ClaimTransition:
    """One bounded step of lifecycle evidence: a sequence and a state token."""

    sequence: int
    state: str

    def __post_init__(self) -> None:
        _check_sequence(self.sequence, "transition sequence")
        _check_token(self.state, "transition state")


@dataclass(frozen=True)
class ClaimRecord:
    """One well-formed claim, exactly as stored."""

    binding: str
    owner: OwnerRef
    state: str
    sequence: int
    transitions: Tuple[ClaimTransition, ...]
    version: int = RECORD_VERSION


@dataclass(frozen=True)
class ClaimInspection:
    """What is written down about one resource, with no authority consulted."""

    status: ClaimStatus
    reason: str
    record: Optional[ClaimRecord] = None


@dataclass(frozen=True)
class ClaimResolution:
    """What is written down, reconciled against the caller's authority."""

    status: ResolvedStatus
    reason: str
    record: Optional[ClaimRecord] = None
    verdict: Optional[OwnerVerdict] = None

    @property
    def reclaimable(self) -> bool:
        """True only when the authority positively proved the owner released."""

        return self.status == "released"


@dataclass(frozen=True)
class ClaimOutcome:
    """The result of a host operation: did it change anything, and why not."""

    applied: bool
    status: ResolvedStatus
    reason: str
    record: Optional[ClaimRecord] = None


class OwnerAuthority(Protocol):
    """Caller-supplied resolver for whether an owner still holds its claim.

    Called with the owner identity on record and the lifecycle state recorded
    alongside it.  It must return one of :data:`VERDICTS`:

    ``"held"``
        The domain's authoritative state says this owner still holds it.
    ``"released"``
        The domain's authoritative state proves this owner let it go.  This is
        the only verdict that permits :meth:`ResourceClaimStore.sweep` to
        release a claim.
    ``"missing"``
        The owner is genuinely absent from the authoritative state - an
        orphan.  Fails closed; only an explicit repair clears it.
    ``"unknown"``
        The authority cannot decide.  Fails closed.

    Raising, or returning anything else, is treated as an authority error and
    also fails closed.  There is no answer that lets an unresolved or ambiguous
    claim be taken automatically.
    """

    def __call__(self, owner: OwnerRef, state: str) -> OwnerVerdict:  # pragma: no cover
        ...


class _LocalLock:
    """One process-local mutex plus a count of who still needs it alive."""

    __slots__ = ("lock", "holders")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.holders = 0


#: Process-local mutexes, keyed by lock path, held **only while in use**.
#:
#: A plain cache keyed by path would grow once per distinct resource and never
#: shrink, so a long-lived process that touches many resources would leak one
#: mutex each - unbounded in the only dimension that is unbounded here.  Each
#: entry is instead reference-counted by the threads that are holding or
#: waiting on it and dropped when the last one leaves, which bounds the
#: registry by live threads rather than by resources ever seen.
#:
#: Exclusion is not weakened: a thread registers itself under the guard
#: *before* it blocks, so an entry can never be evicted while anybody still
#: wants it, and every contender for one resource therefore takes the very same
#: mutex.  Different resources still have different keys and never contend.
_LOCAL_LOCKS: Dict[str, _LocalLock] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()
# ``O_NOFOLLOW`` refuses a final path component that is a symbolic link, so a
# lock file planted as a link to somewhere else is never opened at all.
_OPEN_FLAGS = (
    os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)
#: Read-only, and equally unwilling to follow a link out of the store.
#:
#: ``O_NONBLOCK`` is not an optimisation here, it is the whole defence against
#: a named pipe planted at a claim path: a plain ``O_RDONLY`` open of a FIFO
#: blocks until somebody opens the write end, so a reader would hang forever
#: instead of refusing the entry.  On a regular file the flag has no effect.
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
#: One byte past the bound is all a reader ever needs: it distinguishes "at the
#: bound" from "over the bound" without loading whatever else is in the file.
_MAX_READ_BYTES = MAX_RECORD_BYTES + 1
_POSIX_MODES = os.name == "posix"

_NO_INTERPROCESS_LOCK = (
    "the host has no supported inter-process lock, so this claim store cannot "
    "mutate a claim safely"
)
_NO_ATOMIC_PUBLICATION = (
    "the claim store filesystem cannot publish a claim atomically, so no claim "
    "was written"
)
_NO_PRIVATE_DIRECTORY = (
    "the claim store directory could not be made private, so no claim was "
    "written"
)
_NOT_A_REAL_DIRECTORY = (
    "the claim store path is a symbolic link or not a directory, so it is "
    "refused rather than followed outside the configured store"
)
_NOT_A_REAL_RECORD = (
    "the claim record path is a symbolic link or not a regular file, so it is "
    "refused rather than followed outside the configured store"
)
_NO_PRIVATE_LOCK = (
    "the claim store lock file could not be made private, so no claim was "
    "written"
)
_UNUSABLE_LOCK = "the claim store lock file could not be opened"


def _own_directory(path: Path) -> os.stat_result:
    """``lstat`` one path and insist it really is a directory, not a link to one.

    ``lstat``, never ``stat``: the whole point is to see a symbolic link *as* a
    symbolic link.  ``Path.mkdir(exist_ok=True)`` resolves the link before
    asking whether a directory is already there, so a ``resource-claims`` entry
    planted as a link to somewhere else would otherwise be adopted silently -
    and then chmodded and written through, re-moding a directory outside the
    configured store and putting claims where the caller never pointed.  A path
    this kernel cannot confirm is a real directory it owns is refused instead of
    followed.
    """

    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ResourceClaimError(f"the claim root is unusable: {exc}") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ResourceClaimError(f"{_NOT_A_REAL_DIRECTORY}: {path}")
    return info


def _secure_directory(path: Path) -> None:
    """Create or adopt one directory, private to this account, or fail closed.

    ``mkdir`` alone is not enough for three reasons.  Its mode is masked by the
    process umask, so a permissive umask would silently widen a directory this
    kernel believes it created privately.  ``exist_ok`` says nothing about the
    mode of a directory that was already there - one inherited from an
    installer, an archive extraction or an earlier revision of this module can
    be group- or world-writable, which would let a bystander unlink a live
    claim or drop a record of their own into the shard.  And ``exist_ok``
    accepts a symbolic link to a directory as though it were the directory, so
    the identity of the path has to be checked before its mode is touched.

    The order matters and is the point of this function: create, then confirm
    by ``lstat`` that the path is a real directory, and only then tighten the
    mode.  Nothing is chmodded and nothing is written until the path has been
    proven to be inside the configured store.  The mode is re-read from a
    second ``lstat`` afterwards, so a ``chmod`` that quietly does nothing, or a
    link swapped in behind it, is caught rather than trusted.

    Anything that goes wrong - the path is a link or not a directory, the mode
    cannot be read, the chmod is refused, or the mode does not take - raises
    :class:`ResourceClaimError`.  Storing a claim in a directory whose identity
    or permissions could not be established is exactly what this fails closed
    to avoid.
    """

    try:
        path.mkdir(mode=DIRECTORY_MODE, parents=True, exist_ok=True)
    except OSError as exc:
        raise ResourceClaimError(f"the claim root is unusable: {exc}") from exc
    # Before any chmod, and before anything is written inside.
    info = _own_directory(path)
    if not _POSIX_MODES:  # pragma: no cover - POSIX mode bits are meaningless here
        return
    if stat.S_IMODE(info.st_mode) == DIRECTORY_MODE:
        return
    try:
        os.chmod(path, DIRECTORY_MODE)
    except OSError as exc:
        raise ResourceClaimError(f"{_NO_PRIVATE_DIRECTORY}: {exc}") from exc
    if stat.S_IMODE(_own_directory(path).st_mode) != DIRECTORY_MODE:
        raise ResourceClaimError(f"{_NO_PRIVATE_DIRECTORY}: {path} did not keep the mode")


def _interprocess_locking_supported() -> bool:
    """Whether this host can exclude other *processes*, not merely other threads.

    A process-local :class:`threading.Lock` says nothing about a second
    interpreter, so the only honest answer is whether a real advisory file lock
    is available.  Probed through the module attribute rather than cached at
    import time, so a host that has to be treated as unsupported can be
    represented faithfully in a test.
    """

    return all(getattr(fcntl, name, None) is not None for name in ("flock", "LOCK_EX", "LOCK_UN"))


def _require_interprocess_lock() -> None:
    """Fail closed before touching the store on a host that cannot exclude."""

    if not _interprocess_locking_supported():
        raise ResourceClaimError(_NO_INTERPROCESS_LOCK)


def _secure_open_lock(lock_path: Path) -> int:
    """Open one lock file, private to this account, or fail closed.

    The mode argument to ``os.open`` applies only when the file is *created*,
    so a lock file left behind at a wider mode - by an earlier revision of this
    module, by an umask that no longer applies, or by a bystander who got there
    first - would keep that mode forever and stay writable by whoever can reach
    it.  Holding a claim behind a lock anyone can truncate or replace is not
    exclusion, so the mode is settled before ``flock`` is attempted and before
    any record is touched.

    Both the open and the tightening work on the file itself rather than on its
    name: ``O_NOFOLLOW`` refuses a symlinked lock path outright, and ``fchmod``
    and ``fstat`` address the open descriptor, so nothing here can be redirected
    by swapping the path between two calls.
    """

    try:
        descriptor = os.open(lock_path, _OPEN_FLAGS, FILE_MODE)
    except OSError as exc:
        raise ResourceClaimError(f"{_UNUSABLE_LOCK}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ResourceClaimError(f"{_UNUSABLE_LOCK}: {lock_path} is not a regular file")
        if _POSIX_MODES and stat.S_IMODE(info.st_mode) != FILE_MODE:
            try:
                os.fchmod(descriptor, FILE_MODE)
            except OSError as exc:
                raise ResourceClaimError(f"{_NO_PRIVATE_LOCK}: {exc}") from exc
            if stat.S_IMODE(os.fstat(descriptor).st_mode) != FILE_MODE:
                raise ResourceClaimError(
                    f"{_NO_PRIVATE_LOCK}: {lock_path} did not keep the mode"
                )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def _local_lock(key: str) -> Iterator[None]:
    """Take the process-local mutex for one key, and retire it when unused.

    Registration happens under the guard *before* the mutex is awaited, so an
    entry is never evicted out from under a waiter and every contender for one
    key provably takes the same object.  The last one out removes it, which is
    what keeps the registry proportional to live threads instead of to the
    number of distinct resources this process has ever touched.
    """

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
            # The identity check matters: a racing thread may already have
            # retired this entry and installed a fresh one under the same key.
            if entry.holders <= 0 and _LOCAL_LOCKS.get(key) is entry:
                del _LOCAL_LOCKS[key]


@contextmanager
def _exclusive(lock_path: Path) -> Iterator[None]:
    """Serialize one resource across threads *and* processes, or refuse outright.

    Both locks are mandatory.  There is no degraded mode in which only the
    process-local lock is taken: that would advertise exclusion this kernel
    cannot deliver, and two processes could then both publish a claim.
    """

    _require_interprocess_lock()
    with _local_lock(str(lock_path)):
        # Private before locked: see :func:`_secure_open_lock`.
        descriptor = _secure_open_lock(lock_path)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except OSError as exc:
                raise ResourceClaimError(f"{_NO_INTERPROCESS_LOCK}: {exc}") from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:  # pragma: no cover - unlock of a lock never taken
                pass
            os.close(descriptor)


def _encode(record: ClaimRecord) -> bytes:
    payload = {
        "version": record.version,
        "binding": record.binding,
        "owner_scope": record.owner.scope,
        "owner_id": record.owner.id,
        "state": record.state,
        "sequence": record.sequence,
        "transitions": [
            {"sequence": step.sequence, "state": step.state} for step in record.transitions
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_RECORD_BYTES:
        raise ResourceClaimRejected("the claim record exceeds the record bound")
    return raw


def _unresolved(reason: str) -> ClaimInspection:
    return ClaimInspection("unresolved", reason)


def _read_bounded(descriptor: int) -> bytes:
    """Read at most :data:`_MAX_READ_BYTES`, looping only over short reads.

    ``os.read`` may return fewer bytes than asked for without being at EOF, so
    a single call cannot be trusted to have seen the whole bounded prefix.  The
    loop is bounded by the byte budget, not by the size of the file: it stops
    as soon as the budget is spent or the file ends, so an enormous file costs
    the same as a small one.
    """

    chunks = []
    remaining = _MAX_READ_BYTES
    while remaining > 0:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode_transitions(
    payload: object, state: str, sequence: int
) -> Union[Tuple[ClaimTransition, ...], ClaimInspection]:
    """Parse and cross-check the bounded lifecycle evidence on a record."""

    if not isinstance(payload, list):
        return _unresolved("the claim record transitions are malformed")
    if not payload or len(payload) > MAX_TRANSITIONS:
        return _unresolved("the claim record transitions are out of bounds")

    steps = []
    previous = 0
    for entry in payload:
        if not isinstance(entry, dict) or set(entry) != _TRANSITION_KEYS:
            return _unresolved("the claim record transitions are malformed")
        try:
            step = ClaimTransition(sequence=entry["sequence"], state=entry["state"])
        except ResourceClaimRejected:
            return _unresolved("the claim record transitions are malformed")
        if step.sequence <= previous:
            return _unresolved("the claim record transitions are not monotonic")
        if step.sequence > sequence:
            return _unresolved("the claim record transitions run ahead of the sequence")
        previous = step.sequence
        steps.append(step)

    latest = steps[-1]
    if latest.sequence != sequence or latest.state != state:
        # The counter moved without the evidence moving with it: the record was
        # written by an older or interrupted writer and cannot be trusted.
        return _unresolved("the claim record transition evidence is stale")
    return tuple(steps)


def _decode(raw: bytes, expected_binding: str) -> ClaimInspection:
    """Parse one stored record, failing closed on anything unexpected."""

    if len(raw) > MAX_RECORD_BYTES:
        return _unresolved("the claim record exceeds the record bound")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return _unresolved("the claim record is malformed")
    if not isinstance(payload, dict):
        return _unresolved("the claim record is malformed")
    if set(payload) != _RECORD_KEYS:
        return _unresolved("the claim record does not match the schema")

    version = payload["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        return _unresolved("the claim record version is malformed")
    if version != RECORD_VERSION:
        return _unresolved("the claim record version is unknown")

    binding = payload["binding"]
    if not isinstance(binding, str) or len(binding) != _DIGEST_CHARS or not set(binding) <= _HEX:
        return _unresolved("the claim record binding is malformed")
    if binding != expected_binding:
        return _unresolved("the claim record is bound to another resource")

    try:
        sequence = _check_sequence(payload["sequence"], "sequence")
    except ResourceClaimRejected:
        return _unresolved("the claim record sequence is malformed")

    try:
        owner = OwnerRef(scope=payload["owner_scope"], id=payload["owner_id"])
    except ResourceClaimRejected:
        return _unresolved("the claim record owner identity is missing or malformed")

    try:
        state = _check_token(payload["state"], "state")
    except ResourceClaimRejected:
        return _unresolved("the claim record state is missing or malformed")

    steps = _decode_transitions(payload["transitions"], state, sequence)
    if isinstance(steps, ClaimInspection):
        return steps

    record = ClaimRecord(
        binding=binding,
        owner=owner,
        state=state,
        sequence=sequence,
        transitions=steps,
        version=version,
    )
    return ClaimInspection("held", "a claim is recorded", record)


def _extend(
    steps: Sequence[ClaimTransition], step: ClaimTransition
) -> Tuple[ClaimTransition, ...]:
    """Append one step of evidence, keeping only the most recent window."""

    return (tuple(steps) + (step,))[-MAX_TRANSITIONS:]


class ResourceClaimStore:
    """Durable, multi-process claims rooted at one directory.

    Every method is safe to call from several processes at once, and every
    method touches exactly one resource, so no operation scans the store or
    does unbounded work.  Nothing here expires: a claim outlives the process
    that took it until its exact owner identity releases it, an authority
    proves the owner released it, or a host explicitly repairs the record.

    Constructing a store creates nothing.  The first mutating call establishes
    the root, the shard and the lock file as owner-only, refusing any of the
    three that is a symbolic link, tightening any that was already standing at
    a wider mode, and failing closed if it cannot.  :meth:`inspect` and
    :meth:`resolve` never create, re-mode or lock anything, so they remain
    usable against a store this process may only read.
    """

    def __init__(self, root: Union[str, os.PathLike]) -> None:
        self._root = Path(os.fspath(root)).resolve() / "resource-claims"

    @property
    def root(self) -> Path:
        return self._root

    # -- lookup ---------------------------------------------------------

    def inspect(self, resource: ResourceRef) -> ClaimInspection:
        """Report what is written down: ``free``, ``held`` or ``unresolved``.

        ``held`` here means only that a well-formed record exists.  Use
        :meth:`resolve` when you need to know whether the owner still lives.
        """

        return self._read(resource.digest)

    def resolve(self, resource: ResourceRef, authority: OwnerAuthority) -> ClaimResolution:
        """Reconcile the stored record against the caller's authority.

        This is the path that makes a missing owner genuinely detectable: a
        record can be perfectly well-formed and still name an owner that the
        authoritative state has never heard of.  Ambiguity - an ``unknown``
        verdict, a bad return value, or a resolver that raises - resolves to
        ``unresolved`` and never authorises a takeover.
        """

        found = self._read(resource.digest)
        if found.status != "held" or found.record is None:
            return ClaimResolution(found.status, found.reason, found.record)
        return self._reconcile(found.record, authority)

    # -- ownership ------------------------------------------------------

    def acquire(self, resource: ResourceRef, owner: OwnerRef, state: str) -> ClaimRecord:
        """Take an unheld resource, or fail.  Atomic against other processes.

        Never takes over an existing record, however stale or damaged it looks.
        Use :meth:`sweep` or :meth:`repair` to clear one deliberately first.
        """

        _check_token(state, "state")
        digest = resource.digest
        with self._guard(digest):
            found = self._read(digest)
            if found.status == "unresolved":
                raise ResourceClaimUnresolved(found.reason)
            if found.status == "held":
                raise ResourceClaimConflict("the resource is already claimed")
            record = ClaimRecord(
                binding=digest,
                owner=owner,
                state=state,
                sequence=1,
                transitions=(ClaimTransition(1, state),),
            )
            self._publish(digest, record, exclusive=True)
            return record

    def reassert(self, resource: ResourceRef, owner: OwnerRef, state: str) -> ClaimRecord:
        """Restate a claim for the identity already on record.

        The same owner may present a different lifecycle state on every call -
        that is the point.  The sequence advances and the new state is appended
        to the bounded transition evidence.  A different identity conflicts.
        """

        _check_token(state, "state")
        digest = resource.digest
        with self._guard(digest):
            record = self._require_owned(digest, owner)
            if record.sequence >= MAX_SEQUENCE:
                raise ResourceClaimRejected("the claim sequence is exhausted")
            sequence = record.sequence + 1
            renewed = ClaimRecord(
                binding=digest,
                owner=owner,
                state=state,
                sequence=sequence,
                transitions=_extend(record.transitions, ClaimTransition(sequence, state)),
            )
            self._publish(digest, renewed, exclusive=False)
            return renewed

    def release(self, resource: ResourceRef, owner: OwnerRef) -> bool:
        """Drop a claim held by this exact identity.  False when already free.

        Lifecycle state is irrelevant here: an owner that has moved on from the
        state it acquired under still releases its own claim.
        """

        digest = resource.digest
        with self._guard(digest):
            found = self._read(digest)
            if found.status == "unresolved":
                raise ResourceClaimUnresolved(found.reason)
            if found.status == "free" or found.record is None:
                return False
            if found.record.owner != owner:
                raise ResourceClaimConflict("the claim is held by another owner")
            return self._discard(digest)

    # -- host operations ------------------------------------------------

    def sweep(self, resource: ResourceRef, authority: OwnerAuthority) -> ClaimOutcome:
        """Release one claim, only when the authority proves it was released.

        Bounded by construction: one resource, one authority call, no scan.
        Every other resolution - held, missing, unknown, an authority error, an
        unresolved record - leaves the claim exactly where it is.
        """

        digest = resource.digest
        with self._guard(digest):
            found = self._read(digest)
            if found.status != "held" or found.record is None:
                return ClaimOutcome(False, found.status, found.reason, found.record)
            resolution = self._reconcile(found.record, authority)
            if resolution.status != "released":
                return ClaimOutcome(False, resolution.status, resolution.reason, found.record)
            self._discard(digest)
            return ClaimOutcome(True, "free", "the owner released the claim", found.record)

    def repair(
        self, resource: ResourceRef, authority: Optional[OwnerAuthority] = None
    ) -> ClaimOutcome:
        """Explicitly discard a record a host has decided is not a live claim.

        With no authority this clears only an ``unresolved`` record - a record
        that cannot be parsed cannot be attributed to anyone.  With an
        authority it will additionally clear a well-formed but orphaned record
        whose owner the authority reports ``missing``.  It refuses a held
        claim, and it refuses an ambiguous one: an ``unknown`` verdict or a
        resolver that raises leaves the record untouched, because a host
        repairing an ambiguous claim is indistinguishable from stealing it.
        """

        digest = resource.digest
        with self._guard(digest):
            found = self._read(digest)
            if found.status == "free":
                return ClaimOutcome(False, "free", found.reason)
            if found.status == "unresolved":
                return ClaimOutcome(self._discard(digest), "free", found.reason)

            record = found.record
            assert record is not None
            if authority is None:
                return ClaimOutcome(
                    False, "held", "a well-formed claim needs an authority to repair", record
                )
            resolution = self._reconcile(record, authority)
            if resolution.status != "missing":
                return ClaimOutcome(False, resolution.status, resolution.reason, record)
            self._discard(digest)
            return ClaimOutcome(True, "free", "the claim owner is missing", record)

    # -- internals ------------------------------------------------------

    @staticmethod
    def _reconcile(record: ClaimRecord, authority: OwnerAuthority) -> ClaimResolution:
        try:
            verdict: Any = authority(record.owner, record.state)
        except Exception:
            return ClaimResolution("unresolved", "the owner authority failed", record)
        if verdict not in VERDICTS:
            return ClaimResolution(
                "unresolved", "the owner authority returned an unusable verdict", record
            )
        if verdict == VERDICT_HELD:
            return ClaimResolution("held", "the owner still holds the claim", record, verdict)
        if verdict == VERDICT_RELEASED:
            return ClaimResolution("released", "the owner released the claim", record, verdict)
        if verdict == VERDICT_MISSING:
            return ClaimResolution("missing", "the claim owner is missing", record, verdict)
        return ClaimResolution(
            "unresolved", "the owner authority could not decide", record, VERDICT_UNKNOWN
        )

    def _shard(self, digest: str) -> Path:
        return self._root / digest[:2]

    def _claim_path(self, digest: str) -> Path:
        return self._shard(digest) / (digest + ".claim.json")

    def _lock_path(self, digest: str) -> Path:
        return self._shard(digest) / (digest + ".lock")

    @contextmanager
    def _guard(self, digest: str) -> Iterator[None]:
        # Checked before anything is created, so an unsupported host is left
        # exactly as it was found.
        _require_interprocess_lock()
        # Root first, then shard: an overly permissive root would otherwise let
        # a bystander replace the shard beneath a correctly moded one.  Both
        # are settled before the lock file - itself a file inside the shard -
        # is opened, so nothing this operation creates is ever momentarily
        # exposed in a wider directory.
        _secure_directory(self._root)
        _secure_directory(self._shard(digest))
        with _exclusive(self._lock_path(digest)):
            yield

    def _read(self, digest: str) -> ClaimInspection:
        """Read one record without following a link and without reading far.

        ``Path.read_bytes`` was wrong on both counts.  It follows a symbolic
        link, so a claim path planted as a link would have been read - and,
        worse, reported as a legitimate record - from wherever it pointed,
        outside the store the caller configured.  And it reads to EOF, so a
        multi-gigabyte file dropped at a claim path would have been pulled into
        memory in full only to be rejected by the record bound afterwards.

        ``O_NOFOLLOW`` refuses the link at the syscall, ``fstat`` on the open
        descriptor refuses anything that is not a regular file (a directory, a
        FIFO, a device), and the read stops one byte past
        :data:`MAX_RECORD_BYTES` - enough to tell "at the bound" from "over the
        bound" and not one byte more.  A missing file is still simply ``free``;
        anything present but unfit is ``unresolved`` and is never deleted
        automatically.
        """

        try:
            descriptor = os.open(self._claim_path(digest), _READ_FLAGS)
        except FileNotFoundError:
            return ClaimInspection("free", "no claim is recorded")
        except OSError:
            # ELOOP for a symlink, ENOTDIR for a shard that is not a directory,
            # EACCES, and anything else the host refuses.
            return _unresolved(_NOT_A_REAL_RECORD)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                return _unresolved(_NOT_A_REAL_RECORD)
            raw = _read_bounded(descriptor)
        except OSError:
            return _unresolved("the claim record is unreadable")
        finally:
            os.close(descriptor)
        return _decode(raw, digest)

    def _require_owned(self, digest: str, owner: OwnerRef) -> ClaimRecord:
        found = self._read(digest)
        if found.status == "unresolved":
            raise ResourceClaimUnresolved(found.reason)
        if found.status == "free" or found.record is None:
            raise ResourceClaimConflict("the resource is not claimed")
        if found.record.owner != owner:
            raise ResourceClaimConflict("the claim is held by another owner")
        return found.record

    def _publish(self, digest: str, record: ClaimRecord, *, exclusive: bool) -> None:
        """Stage a whole record, fsync it, then expose it in one atomic step.

        Nothing is ever written through the final claim path.  Whichever
        publication step is used, it either exposes the complete fsynced record
        or fails, and the staged file is removed on the way out either way, so a
        failed publication leaves neither a partial record nor a final claim.
        """

        raw = _encode(record)
        path = self._claim_path(digest)
        descriptor, staged = tempfile.mkstemp(dir=str(self._shard(digest)), suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(staged, FILE_MODE)
            if exclusive:
                self._link(staged, path)
            else:
                try:
                    os.replace(staged, path)
                except OSError as exc:
                    raise ResourceClaimError(f"{_NO_ATOMIC_PUBLICATION}: {exc}") from exc
                staged = ""
        finally:
            if staged:
                try:
                    os.unlink(staged)
                except OSError:  # pragma: no cover - best-effort cleanup
                    pass
        self._sync_shard(digest)

    @staticmethod
    def _link(staged: str, path: Path) -> None:
        """Publish already-complete content atomically, or publish nothing at all.

        The hard link is the whole guarantee: the final path either does not
        exist or names content that was fully written and fsynced before the
        link was made, so a reader can never observe a partial record.  There is
        deliberately no copy-the-bytes fallback for a filesystem that cannot
        hard-link - opening the final path and writing into it would expose
        exactly the torn record this design exists to prevent, and would race a
        second writer that the ``O_EXCL`` check had already let through.  Such a
        host fails closed instead, leaving no claim for the caller to mistake
        for a successful one.
        """

        try:
            os.link(staged, path)
        except FileExistsError as exc:
            raise ResourceClaimConflict("the resource is already claimed") from exc
        except OSError as exc:
            raise ResourceClaimError(f"{_NO_ATOMIC_PUBLICATION}: {exc}") from exc

    def _sync_shard(self, digest: str) -> None:
        try:
            descriptor = os.open(str(self._shard(digest)), os.O_RDONLY)
        except OSError:  # pragma: no cover - platforms without directory fds
            return
        try:
            os.fsync(descriptor)
        except OSError:  # pragma: no cover - filesystems without directory fsync
            pass
        finally:
            os.close(descriptor)

    def _discard(self, digest: str) -> bool:
        try:
            self._claim_path(digest).unlink()
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise ResourceClaimError(f"the claim record could not be removed: {exc}") from exc
        self._sync_shard(digest)
        return True
