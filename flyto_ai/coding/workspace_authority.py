# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Host-global authority over a workspace root, independent of any state root.

A coding state root brokers the *jobs* inside it. It cannot broker a directory
tree, because the claim files that record "this worktree is being edited" live
under the state root itself: point two services at two different state roots
and each one keeps a private, perfectly consistent opinion about the same tree.

That is not hypothetical. On 2026-08-11 one service used a
`coding-service` state root and another used a `coding-service-<suffix>` root,
both configured with the same workspace root, and two sessions edited the same
checkout concurrently. Neither service was wrong about its own state; there was
simply nothing above them that owned the tree.

This module is that missing layer. It is deliberately small and looks like the
state-root startup authority it sits beside, because the invariant is the same
shape:

* every live service holds a **shared** `flock` on one registry entry per
  canonical workspace root, so many processes sharing a state root coexist;
* a newcomer first tries the **exclusive** lock. Getting it proves nobody is
  alive on that tree, and only then may the recorded identity be written or
  rotated;
* failing to get it proves somebody is alive, so the recorded identity is
  authoritative and a different state root is refused.

Liveness is the kernel's answer, never a timestamp, so a crash releases the
lease with no heartbeat and no TTL. Durable identity is what stops the crash
from becoming a free-for-all: an incompatible state root may only take a tree
over once the previous owner's work is *finished*, never merely because the
previous owner died.

Nothing here knows what a job is, which provider runs it, or what product owns
the tree. It takes paths and a callable that answers "does this state root
still have open work", so it stays reusable and project-neutral.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - platform dependent
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

from flyto_ai.coding.continuation import ContinuationCorrupt, secure_directory


#: Bumped only when the on-disk entry schema changes incompatibly.
WORKSPACE_AUTHORITY_VERSION = "coding.workspace-authority.v1"
#: Startup-only override, read once when a registry is constructed. It exists so
#: isolated tests get a private registry; it is never read from a job payload,
#: and a running service cannot be redirected by changing it.
WORKSPACE_REGISTRY_ENV = "CODING_WORKSPACE_AUTHORITY_ROOT"
#: Deliberately neutral: no product, repository, or vendor name appears in the
#: protocol or in the default location. The registry brokers directories, and a
#: directory does not belong to a product.
_DEFAULT_REGISTRY_DIRNAME = "coding-workspace-authority"
#: An entry is tiny. Anything larger is treated as damaged rather than parsed.
MAX_ENTRY_BYTES = 64 * 1024
#: Serialises the whole join transaction so overlap is decided atomically.
REGISTRY_LOCK_NAME = ".registry.lock"
#: Strict upper bound on waiting for the registry-wide lock. A join holds it
#: for a few reads and one small write, so anything approaching this means the
#: holder is wedged, and a bounded refusal beats an unbounded hang.
REGISTRY_LOCK_TIMEOUT_SECONDS = 5.0
#: Retry granularity. Bounded by the deadline above, never by a retry count.
REGISTRY_LOCK_POLL_SECONDS = 0.02
#: The operator report stays bounded like every other surface here.
MAX_REPORTED_OWNERS = 32
#: Most blocking first, so the headline status is the one that would actually
#: refuse a start.
_STATUS_ORDER = {"live": 0, "crashed_with_open_work": 1, "adoptable": 2}
#: Deterministic ordering for the reported overlap list.
_RELATIONSHIP_ORDER = {"exact": 0, "owner_is_ancestor": 1, "owner_is_descendant": 2}
_DIGEST_RE = re.compile(r"^[a-f0-9]{64}$")
#: The only errnos that prove another process holds a lock. POSIX allows either
#: `EWOULDBLOCK`/`EAGAIN` or `EACCES` for a contended non-blocking lock, and on
#: Linux the first two are the same value. Anything else means the probe
#: failed, which is never evidence that a tree is free.
_CONTENDED_ERRNOS = frozenset({errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES})


def _paths_overlap(left: Path, right: Path) -> bool:
    """Whether two canonical trees can write the same file.

    True when they are equal or when either contains the other. Compared on
    resolved path *parts*, so `/a/bc` is not treated as living inside `/a/b`
    the way a string prefix test would.
    """

    first, second = left.parts, right.parts
    shortest = min(len(first), len(second))
    return first[:shortest] == second[:shortest]


class WorkspaceAuthorityError(Exception):
    """Base class carrying a stable, non-sensitive code.

    Deliberately not a `CodingServiceError`: this layer must not import the
    service it protects. `flyto_ai.coding.service` maps these onto its own
    bounded error surface, which is what reaches a client.
    """

    code = "workspace_authority_error"

    def __init__(self, message: str, *, workspace_digest: str = "") -> None:
        super().__init__(message)
        #: Identifies *which* tree refused without naming it. A digest is safe
        #: to log or surface; the path is not.
        self.workspace_digest = workspace_digest


class WorkspaceAuthorityConflict(WorkspaceAuthorityError):
    """Another coding state root owns this workspace tree."""

    code = "workspace_authority_conflict"


class WorkspaceAuthorityUnavailable(WorkspaceAuthorityError):
    """The registry cannot answer, so ownership cannot be established."""

    code = "workspace_authority_unavailable"


class WorkspaceAuthorityBusy(WorkspaceAuthorityError):
    """Another process holds the registry-wide lock past the deadline.

    Distinct from unavailable because it is transient and retryable by the
    caller: the registry is intact, somebody else is simply using it. It is
    raised rather than waited out so one wedged holder cannot hang every start
    on the host.
    """

    code = "workspace_authority_busy"


def default_registry_root() -> Path:
    """Where the registry lives when a host does not choose for itself.

    Stable across state roots by construction -- it is derived from the user's
    state directory, not from any state root -- and deliberately outside every
    product worktree, so a claim file can never land in a checkout that a job is
    about to edit, or in a repository whose dirty state someone is auditing.
    """

    override = os.environ.get(WORKSPACE_REGISTRY_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_STATE_HOME", "").strip()
    root = Path(base).expanduser() if base else Path.home() / ".local" / "state"
    return root / _DEFAULT_REGISTRY_DIRNAME


def canonical_workspace_root(path: Any) -> Path:
    """Resolve one workspace root to the identity the registry brokers.

    `resolve()` collapses `..`, `.`, and every symlink, so two configurations
    that name the same tree by different routes -- a link, a relative path, a
    trailing slash -- produce the same identity and therefore contend for the
    same entry. Alias-by-symlink was the obvious way to walk around a broker
    keyed on the configured string.
    """

    return Path(os.path.abspath(os.path.expanduser(str(path)))).resolve()


def workspace_digest(path: Any) -> str:
    """A stable, path-free identifier for one canonical workspace root."""

    canonical = str(canonical_workspace_root(path))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def state_root_open_workspace_roots(
    state_root: Any,
) -> Optional[Tuple[Path, ...]]:
    """Return the canonical trees durable work still needs.

    ``None`` means the state could not be attributed safely.  Callers must then
    retain or refuse the broad configured boundary; an unreadable record must
    never be converted into permission to release a lease.  Records written
    before repo-set leases existed fall back to their one ``working_dir`` and
    therefore retain the exact conservative behaviour they had before.
    """

    root = Path(os.path.abspath(os.path.expanduser(str(state_root))))
    if not root.is_dir():
        return ()
    terminal = {"completed", "failed", "codex_accepted"}
    required: Dict[str, Path] = {}
    records: Dict[str, Optional[Tuple[Path, ...]]] = {}
    tenants = root / "tenants"
    try:
        tenant_dirs = sorted(tenants.iterdir()) if tenants.is_dir() else []
    except OSError:
        return None
    for tenant in tenant_dirs:
        jobs = tenant / "jobs"
        try:
            entries = sorted(jobs.glob("*.json")) if jobs.is_dir() else []
        except OSError:
            return None
        for entry in entries:
            try:
                record = json.loads(entry.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            if not isinstance(record, dict):
                return None
            job_id = str(record.get("job_id") or "")
            if not job_id:
                if str(record.get("state")) in terminal:
                    # Historical terminal fixtures/records can predate opaque
                    # job ids.  They carry no live claim to attribute and do
                    # not prevent adoption merely because they are old.
                    continue
                return None
            if job_id in records:
                return None
            values = record.get("repository_roots")
            if values is None:
                values = [record.get("working_dir")]
            if (
                not isinstance(values, list)
                or not 1 <= len(values) <= 16
                or any(not isinstance(value, str) or not value for value in values)
            ):
                workspace_roots = None
            else:
                workspace_roots = tuple(
                    sorted({canonical_workspace_root(value) for value in values})
                )
                if record.get("repository_roots") is not None:
                    digests = record.get("repository_digests")
                    if (
                        not isinstance(digests, list)
                        or digests != [workspace_digest(value) for value in workspace_roots]
                    ):
                        workspace_roots = None
            records[job_id] = workspace_roots
            if str(record.get("state")) not in terminal:
                if workspace_roots is None:
                    return None
                for workspace in workspace_roots:
                    required[str(workspace)] = workspace

    # A surviving worktree owner claim is open work too.  Bind it back to the
    # owning job record so a stale exact claim does not broaden one unrelated
    # repository into a parent-wide lock.  A missing/malformed binding remains
    # ambiguous and therefore fail-closed.
    claims = root / "locks" / "workspaces"
    try:
        owner_paths = sorted(claims.glob("*.owner.json")) if claims.is_dir() else []
    except OSError:
        return None
    for claim in owner_paths:
        try:
            value = json.loads(claim.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(value, dict):
            return None
        workspace_roots = records.get(str(value.get("job_id") or ""))
        if workspace_roots is None:
            return None
        for workspace in workspace_roots:
            required[str(workspace)] = workspace
    return tuple(required[key] for key in sorted(required))


def state_root_has_open_work(
    state_root: Any, workspace_root: Optional[Any] = None,
) -> bool:
    """Whether a state root still holds work that must not be orphaned.

    Fail-closed in every ambiguous direction: an unreadable record, a malformed
    record, or an unreadable directory all count as open. The question being
    answered is "may an incompatible state root take this tree over", and the
    only safe default is no.

    Open means either a non-terminal job record or any surviving workspace
    owner claim. Persistent ``*.lock`` files are only flock rendezvous points;
    they do not encode ownership and must not make every previously used state
    root impossible to retire. A legacy owner claim from before this layer
    existed still keeps the tree locked to its own state root, which is what
    stops migration from making audit-pending work look free.
    """

    required = state_root_open_workspace_roots(state_root)
    if required is None:
        return True
    if workspace_root is None:
        return bool(required)
    target = canonical_workspace_root(workspace_root)
    if target in required:
        return True

    # A v1 job written before repository sets acquired a configured ancestor
    # but persisted only ``working_dir``.  Preserve that conservative ancestor
    # relationship during migration.  New records name their exact repo set,
    # so an inactive historical parent entry does not keep serialising two
    # unrelated child repos merely because another new child job is open.
    root = Path(os.path.abspath(os.path.expanduser(str(state_root))))
    terminal = {"completed", "failed", "codex_accepted"}
    claimed_jobs = set()
    claims = root / "locks" / "workspaces"
    try:
        for claim in sorted(claims.glob("*.owner.json")) if claims.is_dir() else ():
            value = json.loads(claim.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not isinstance(value.get("job_id"), str):
                return True
            claimed_jobs.add(value["job_id"])
        tenants = root / "tenants"
        for tenant in sorted(tenants.iterdir()) if tenants.is_dir() else ():
            jobs = tenant / "jobs"
            for entry in sorted(jobs.glob("*.json")) if jobs.is_dir() else ():
                record = json.loads(entry.read_text(encoding="utf-8"))
                if not isinstance(record, dict):
                    return True
                if "repository_roots" in record:
                    continue
                if (
                    str(record.get("state")) in terminal
                    and str(record.get("job_id") or "") not in claimed_jobs
                ):
                    continue
                working = record.get("working_dir")
                if not isinstance(working, str) or not working:
                    return True
                legacy = canonical_workspace_root(working)
                if target == legacy or target in legacy.parents:
                    return True
    except (OSError, ValueError):
        return True
    return False


class WorkspaceRootAuthority:
    """One process's joined authority over a set of canonical workspace roots.

    Holds one shared `flock` per root for the life of the process. Releasing is
    idempotent and closes every descriptor, so a partially joined set never
    leaves half a hold behind.
    """

    def __init__(self, registry_root: Optional[Any] = None) -> None:
        self.registry_root = (
            default_registry_root() if registry_root is None
            else Path(registry_root).expanduser()
        )
        self._descriptors: Dict[str, int] = {}

    # -- lifecycle -------------------------------------------------------

    def join(
        self,
        *,
        state_root: Any,
        workspace_roots: Sequence[Any],
        has_open_work: Callable[[str, Optional[str]], bool] = state_root_has_open_work,
    ) -> None:
        """Take authority over every root, or take none and refuse.

        Roots are joined in canonical sorted order. Two processes starting at
        the same moment on the same overlapping set therefore request them in
        the same order, so they cannot each hold half of the other's set and
        deadlock: one wins the first contested entry and the other refuses.
        """

        if fcntl is None:
            raise WorkspaceAuthorityUnavailable(
                "this host has no inter-process lock, so a workspace root"
                " cannot be bound to one coding state root",
            )
        owner = str(Path(os.path.abspath(os.path.expanduser(str(state_root)))))
        canonical = sorted({canonical_workspace_root(item) for item in workspace_roots})
        if not canonical:
            return
        directory = self._registry_directory()
        coordination = -1
        held_before = set(self._descriptors)
        try:
            # One registry-wide lock, taken first and held for the whole join.
            #
            # Per-entry locks alone cannot decide overlap: deciding whether a
            # *descendant* of my tree is owned means reading entries I do not
            # hold, and between that read and my write another process can take
            # one. Serialising the whole transaction is what makes both
            # directions atomic under a concurrent start, and it is cheap
            # because a join is a few reads and one small write.
            #
            # Lock order is always registry -> entry, never the reverse, so two
            # joiners cannot hold half of each other's set.
            coordination = self._open_coordination_lock(directory)
            self._refuse_overlapping_owner(
                directory, owner, canonical, has_open_work,
            )
            for root in canonical:
                if workspace_digest(root) in held_before:
                    continue
                self._join_one(directory, owner, root, has_open_work)
        except BaseException:
            # All or nothing for this incremental request.  Existing holds are
            # not collateral damage when one new repository conflicts.
            self.release_digests(set(self._descriptors) - held_before)
            raise
        finally:
            self._release_coordination_lock(coordination)

    def _release_coordination_lock(self, descriptor: int) -> None:
        """Give the registry-wide lock back. Safe on `-1` and on a failure."""

        if descriptor == -1:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(descriptor)
        except OSError:
            pass

    def release(self) -> None:
        """Drop every hold. Safe to call twice, and on a failed join."""

        self.release_digests(set(self._descriptors))

    def release_digests(self, digests: Sequence[str]) -> None:
        """Drop a selected set of holds; unknown digests are harmless."""

        for digest in tuple(sorted(set(digests))):
            descriptor = self._descriptors.pop(digest, None)
            if descriptor is None:
                continue
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(descriptor)
            except OSError:
                pass

    def retain(self, workspace_roots: Sequence[Any]) -> None:
        """Release every hold not present in the desired canonical repo set."""

        desired = {workspace_digest(root) for root in workspace_roots}
        self.release_digests(set(self._descriptors) - desired)

    @property
    def held_digests(self) -> List[str]:
        return sorted(self._descriptors)

    # -- protocol --------------------------------------------------------

    def _registry_directory(self) -> Path:
        """Create or open the private registry, refusing links on the way."""

        try:
            return secure_directory(self.registry_root, create=True)
        except (ContinuationCorrupt, OSError) as exc:
            raise WorkspaceAuthorityUnavailable(
                "the workspace authority registry is not a safe private"
                " directory, so workspace ownership cannot be established",
            ) from exc

    def _join_one(
        self,
        directory: Path,
        owner: str,
        root: Path,
        has_open_work: Callable[[str, Optional[str]], bool],
    ) -> None:
        digest = workspace_digest(root)
        descriptor = self._open_entry_lock(directory, digest)
        try:
            exclusive = True
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                # Only real contention may be read as "somebody else holds
                # this". An `EIO`, `ENOLCK`, `ENOTSUP` or `EBADF` here says the
                # lock attempt itself failed, and falling through to the shared
                # branch would quietly join a tree whose ownership was never
                # established -- exactly the silent-degradation this layer
                # exists to prevent.
                if exc.errno not in _CONTENDED_ERRNOS:
                    raise WorkspaceAuthorityUnavailable(
                        "the workspace authority lease could not be taken",
                        workspace_digest=digest,
                    ) from exc
                exclusive = False
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except OSError as exc:
                    raise WorkspaceAuthorityUnavailable(
                        "the workspace authority lease is unavailable",
                        workspace_digest=digest,
                    ) from exc
            recorded = self._read_entry(directory, digest)
            if not exclusive:
                # Somebody is alive on this tree right now.
                if recorded is None:
                    raise WorkspaceAuthorityConflict(
                        "a live coding service holds this workspace root and"
                        " records no owning state root; stop it before starting",
                        workspace_digest=digest,
                    )
                if recorded.get("state_root") != owner:
                    raise WorkspaceAuthorityConflict(
                        "this workspace root is held by a live coding service"
                        " using a different state root",
                        workspace_digest=digest,
                    )
            else:
                # Nobody is alive. Identity still decides whether this process
                # may take over, because a crash must not hand the tree to
                # whoever restarts first while the previous owner's work stands.
                if recorded is not None and recorded.get("state_root") != owner:
                    previous = str(recorded.get("state_root") or "")
                    if not previous or has_open_work(previous, str(root)):
                        raise WorkspaceAuthorityConflict(
                            "this workspace root belongs to another coding state"
                            " root that still has unresolved work",
                            workspace_digest=digest,
                        )
                if recorded is None or recorded.get("state_root") != owner:
                    self._write_entry(directory, digest, owner, root)
                # Downgrade so the next process sharing this state root joins.
                fcntl.flock(descriptor, fcntl.LOCK_SH)
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptors[digest] = descriptor

    def _open_coordination_lock(self, directory: Path) -> int:
        """Take the registry-wide join lock within a strict upper bound.

        Never a blocking `flock`. A live but wedged holder would then hang
        every start and every status read on the host, turning one stuck
        process into a total outage of a mechanism whose whole job is to keep
        services honest about each other.

        Instead: retry non-blocking on real contention until a fixed deadline,
        then give a bounded `busy` answer. Total sleep is bounded by
        construction -- `REGISTRY_LOCK_TIMEOUT_SECONDS` -- and every non-
        contention errno fails immediately as unavailable rather than being
        retried, because retrying an `EIO` only delays the same refusal.
        """

        path = directory / REGISTRY_LOCK_NAME
        descriptor = self._open_private_file(path, "the workspace authority registry lock")
        deadline = time.monotonic() + REGISTRY_LOCK_TIMEOUT_SECONDS
        try:
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return descriptor
                except OSError as exc:
                    if exc.errno not in _CONTENDED_ERRNOS:
                        raise WorkspaceAuthorityUnavailable(
                            "the workspace authority registry lock could not be taken",
                        ) from exc
                    if time.monotonic() >= deadline:
                        raise WorkspaceAuthorityBusy(
                            "the workspace authority registry is busy; another"
                            " process is joining or reporting on it",
                        ) from exc
                time.sleep(REGISTRY_LOCK_POLL_SECONDS)
        except BaseException:
            os.close(descriptor)
            raise

    def _refuse_overlapping_owner(
        self,
        directory: Path,
        owner: str,
        canonical: Sequence[Path],
        has_open_work: Callable[[str, Optional[str]], bool],
    ) -> None:
        """Refuse any registered tree that overlaps one of mine.

        Overlap is symmetric and both directions are real concurrent editing:
        a service on `<tree>` and a service on `<tree>/child` write the same
        files. Equality is handled by the per-entry step; this covers strict
        ancestors and strict descendants in one registry scan, which is only
        sound because the caller holds the registry-wide lock.

        A registered tree owned by somebody else refuses when its owner is
        alive, and also when its owner is merely crashed but still has
        unresolved work -- the same rule the per-entry step applies, so a
        nested tree cannot be adopted on easier terms than an identical one.
        """

        mine = {str(root) for root in canonical}
        for digest, recorded in self._scan_entries(directory):
            registered = recorded.get("workspace_root")
            previous = str(recorded.get("state_root") or "")
            if not isinstance(registered, str) or not registered:
                raise WorkspaceAuthorityUnavailable(
                    "the workspace authority entry is malformed",
                    workspace_digest=digest,
                )
            if registered in mine or previous == owner:
                continue
            if not any(_paths_overlap(Path(registered), root) for root in canonical):
                continue
            if self._entry_is_live(directory, digest):
                raise WorkspaceAuthorityConflict(
                    "a live coding service using a different state root owns an"
                    " overlapping workspace root",
                    workspace_digest=digest,
                )
            if not previous or has_open_work(previous, registered):
                raise WorkspaceAuthorityConflict(
                    "an overlapping workspace root belongs to another coding"
                    " state root that still has unresolved work",
                    workspace_digest=digest,
                )

    def _scan_entries(self, directory: Path):
        """Every readable registry entry as `(digest, record)`.

        A damaged or unreadable entry raises rather than being skipped: a scan
        that quietly ignored what it could not parse would report "no overlap"
        for a tree somebody owns.
        """

        try:
            names = sorted(directory.glob("*.json"))
        except OSError as exc:
            raise WorkspaceAuthorityUnavailable(
                "the workspace authority registry could not be read",
            ) from exc
        entries = []
        for path in names:
            digest = path.name[: -len(".json")]
            if not _DIGEST_RE.fullmatch(digest):
                # Not one of ours. A stray file must not be parsed as an entry,
                # and must not silently suppress the scan either.
                continue
            recorded = self._read_entry(directory, digest)
            if recorded is not None:
                entries.append((digest, recorded))
        return entries

    def _entry_is_live(self, directory: Path, digest: str) -> bool:
        """Whether somebody currently holds this entry's lease.

        Only genuine lock contention proves a holder. Every other failure --
        an unsupported filesystem, a bad descriptor, an I/O error -- means the
        probe could not answer, and an unanswered probe is never allowed to
        read as *free*: it raises, so the join fails closed rather than
        stepping onto a tree it could not check.

        A missing lock file is genuinely not held. It is the one absence that
        is safe, because an entry's lock is created before its record.
        """

        path = directory / "{}.lock".format(digest)
        try:
            descriptor = self._open_private_file(
                path, "the workspace authority lease", create=False, digest=digest,
            )
        except FileNotFoundError:
            return False
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in _CONTENDED_ERRNOS:
                    return True
                raise WorkspaceAuthorityUnavailable(
                    "the workspace authority lease could not be probed",
                    workspace_digest=digest,
                ) from exc
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            return False
        finally:
            os.close(descriptor)

    def _open_private_file(
        self,
        path: Path,
        description: str,
        *,
        create: bool = True,
        digest: str = "",
    ) -> int:
        """Open one registry file, refusing anything that is not our own.

        `O_NOFOLLOW` refuses a symlink at the final component and the
        descriptor is then `fstat`-ed, so the checks are made against the file
        actually held rather than a name that could be swapped in between. A
        non-regular file, or one readable by anyone else, is refused rather
        than used: a lock somebody else can replace is not a lock.
        """

        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        if create:
            flags |= os.O_CREAT
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileNotFoundError:
            if not create:
                raise
            raise WorkspaceAuthorityUnavailable(
                "{} could not be opened".format(description),
                workspace_digest=digest,
            ) from None
        except OSError as exc:
            raise WorkspaceAuthorityUnavailable(
                "{} could not be opened".format(description),
                workspace_digest=digest,
            ) from exc
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise WorkspaceAuthorityUnavailable(
                    "{} is not a regular file".format(description),
                    workspace_digest=digest,
                )
            if info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise WorkspaceAuthorityUnavailable(
                    "{} is not privately permissioned".format(description),
                    workspace_digest=digest,
                )
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    def _open_entry_lock(self, directory: Path, digest: str) -> int:
        return self._open_private_file(
            directory / "{}.lock".format(digest),
            "the workspace authority lease",
            digest=digest,
        )

    def _entry_path(self, directory: Path, digest: str) -> Path:
        return directory / "{}.json".format(digest)

    def _read_entry(self, directory: Path, digest: str) -> Optional[Dict[str, Any]]:
        """Read one entry, or None when there is none.

        A damaged entry raises rather than reading as absent: treating it as
        "no owner" would let whoever started next overwrite it, which is the
        one thing a recorded identity exists to prevent.
        """

        path = self._entry_path(directory, digest)
        try:
            descriptor = self._open_private_file(
                path, "the workspace authority entry", create=False, digest=digest,
            )
        except FileNotFoundError:
            return None
        try:
            if os.fstat(descriptor).st_size > MAX_ENTRY_BYTES:
                raise WorkspaceAuthorityUnavailable(
                    "the workspace authority entry is oversized",
                    workspace_digest=digest,
                )
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                descriptor = -1  # the context manager owns it now
                value = json.loads(stream.read(MAX_ENTRY_BYTES + 1))
        except (OSError, ValueError) as exc:
            raise WorkspaceAuthorityUnavailable(
                "the workspace authority entry is unreadable",
                workspace_digest=digest,
            ) from exc
        finally:
            if descriptor != -1:
                os.close(descriptor)
        if (
            not isinstance(value, dict)
            or value.get("version") != WORKSPACE_AUTHORITY_VERSION
            or not isinstance(value.get("state_root"), str)
            or not value.get("state_root")
        ):
            raise WorkspaceAuthorityUnavailable(
                "the workspace authority entry is malformed",
                workspace_digest=digest,
            )
        return value

    def _write_entry(
        self, directory: Path, digest: str, owner: str, root: Path,
    ) -> None:
        """Record the owning state root atomically, privately, and durably.

        The caller holds the exclusive lock, which is what makes this safe.

        Durability is the point of the directory fsync at the end. Without it
        a host crash can lose the *rename* while keeping the file contents, so
        the tree would come back apparently unowned and the next incompatible
        state root would adopt it -- precisely the takeover the recorded
        identity exists to prevent. This function therefore never returns
        successfully unless the record is on stable storage.
        """

        if not _DIGEST_RE.fullmatch(digest):  # pragma: no cover - internal
            raise WorkspaceAuthorityUnavailable("workspace digest is malformed")
        payload = {
            "version": WORKSPACE_AUTHORITY_VERSION,
            "state_root": owner,
            # Kept for the operator recovery path: a human clearing an entry
            # needs to know which tree it was. It never reaches a client.
            "workspace_root": str(root),
        }
        target = self._entry_path(directory, digest)
        handle, temporary = tempfile.mkstemp(dir=str(directory), prefix=".entry-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        except OSError as exc:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise WorkspaceAuthorityUnavailable(
                "the workspace authority entry could not be written",
                workspace_digest=digest,
            ) from exc
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        # The rename itself has to reach stable storage. Opened with
        # `O_DIRECTORY | O_NOFOLLOW` on the directory we already validated,
        # rather than by reopening a path that could have been swapped.
        try:
            handle = os.open(
                str(directory),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise WorkspaceAuthorityUnavailable(
                "the workspace authority registry could not be synced",
                workspace_digest=digest,
            ) from exc
        try:
            os.fsync(handle)
        except OSError as exc:
            raise WorkspaceAuthorityUnavailable(
                "the workspace authority entry could not be made durable",
                workspace_digest=digest,
            ) from exc
        finally:
            os.close(handle)


def describe_workspace_root(
    registry_root: Optional[Any],
    workspace_root: Any,
    *,
    has_open_work: Callable[[str, Optional[str]], bool] = state_root_has_open_work,
) -> Dict[str, Any]:
    """Read-only ownership report for one tree, for an operator to act on.

    Answers the three questions a refusal raises and nothing else: who owns
    this tree, are they still running, and can I take it over. `status` is one
    of:

    ``unregistered``
        Nobody has ever claimed it. The next start takes it.
    ``live``
        A process is holding the lease right now. Stop it.
    ``crashed_with_open_work``
        The owner is gone but left non-terminal jobs or workspace owner
        claims. Finish or retire them against *that* state root; the release
        valve is the subtractive way to retire a stranded audit.
    ``adoptable``
        The owner is gone and left nothing unresolved. The next start adopts
        it automatically, with no file to edit by hand.

    `state_root` is included because an operator has to know which root to go
    and clean up, and this runs locally under their own authority. It is never
    routed to an MCP client.
    """

    authority = WorkspaceRootAuthority(registry_root)
    canonical = canonical_workspace_root(workspace_root)
    digest = workspace_digest(canonical)
    report: Dict[str, Any] = {
        "workspace_digest": digest,
        "status": "unregistered",
        "state_root": "",
        "registry_root": str(authority.registry_root),
        "owners": [],
    }
    directory = authority._registry_directory()
    # The same registry-wide lock the join takes, for the same reason: a scan
    # that raced a join could report an owner that has just gone away, or miss
    # one that has just arrived. It is released immediately -- this reads, and
    # never takes a long-lived ownership lease on anything.
    coordination = authority._open_coordination_lock(directory)
    try:
        owners = []
        for entry_digest, recorded in authority._scan_entries(directory):
            registered = recorded.get("workspace_root")
            if not isinstance(registered, str) or not registered:
                raise WorkspaceAuthorityUnavailable(
                    "the workspace authority entry is malformed",
                    workspace_digest=entry_digest,
                )
            registered_path = Path(registered)
            if not _paths_overlap(registered_path, canonical):
                continue
            state_root = str(recorded.get("state_root") or "")
            if authority._entry_is_live(directory, entry_digest):
                status = "live"
            elif not state_root or has_open_work(state_root, registered):
                status = "crashed_with_open_work"
            else:
                status = "adoptable"
            owners.append({
                "workspace_digest": entry_digest,
                "relationship": _relationship(registered_path, canonical),
                "status": status,
                "state_root": state_root,
            })
    finally:
        authority._release_coordination_lock(coordination)

    # Deterministic: exact first, then ancestors, then descendants, then by
    # digest, so two runs against one registry print the same thing and an
    # operator reading a bug report sees what the reporter saw.
    owners.sort(key=lambda item: (
        _RELATIONSHIP_ORDER.get(item["relationship"], 9), item["workspace_digest"],
    ))
    report["owners"] = owners[:MAX_REPORTED_OWNERS]
    if not owners:
        return report
    # The headline is the most blocking overlap, not merely the exact one: an
    # exact entry that is adoptable while a parent is live must not read as
    # "adoptable", because starting here would still be refused.
    blocking = min(owners, key=lambda item: _STATUS_ORDER.get(item["status"], 9))
    report["status"] = blocking["status"]
    report["state_root"] = blocking["state_root"]
    return report


def _relationship(registered: Path, requested: Path) -> str:
    """How a registered tree overlaps the one being asked about."""

    if registered == requested:
        return "exact"
    if len(registered.parts) < len(requested.parts):
        return "owner_is_ancestor"
    return "owner_is_descendant"
