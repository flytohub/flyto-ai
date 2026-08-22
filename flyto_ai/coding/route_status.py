# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Bounded durable runtime status for the coding control plane.

Per-job JSON stays authoritative. This is a compact pointer that answers two
operational questions a future Codex cannot otherwise answer after a restart:
*where did the round stop* and *did the implementer really start*.

Many Codex conversations share one state root, each with its own long-lived
`code-mcp` process, so a single latest-writer file would let an old process
overwrite a newer one's status. Instead each service instance owns one file
named by its own opaque instance id, and a small shared index lists the
instances:

```text
<state root>/status/index.json                  bounded instance index
<state root>/status/instance-<id>.json          one instance's latest status
```

Every record carries the instance id, an immutable build digest of the
installed coding control plane, and the process start time, so a reader can
tell a repaired build from an old build that is still live, and either from a
stale instance that is no longer running.

The schema is closed. It carries ids, states, counters, and stable codes only:
no task message, error text, working directory, file list, source content,
environment, command line, or credential has a representation here.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import tempfile
import time

try:  # pragma: no cover - platform dependent
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from flyto_ai.coding.contracts import (
    MAX_IMPLEMENTATION_SESSION_ID_CHARS,
    CodingJobState,
)
from flyto_ai.coding.route import CodingRouteReceipt, route_failure_point


ROUTE_STATUS_CONTRACT_VERSION = "flyto.coding-route-status.v1"
ROUTE_STATUS_DIRNAME = "status"
ROUTE_STATUS_INDEX_FILENAME = "index.json"
#: How many instances the shared index retains. Older entries and their files
#: are removed deterministically, so the directory cannot grow without bound.
MAX_STATUS_INSTANCES = 32
#: An instance that has not published for this long is stale and collectable.
#: Seven days comfortably outlives a long Codex conversation while keeping the
#: directory small on a machine that runs many of them.
STATUS_INSTANCE_TTL_SECONDS = 7 * 24 * 3600
#: The index is read back by every publisher, so it is bounded like any other
#: untrusted input. An index above this size is discarded, never republished.
MAX_STATUS_INDEX_BYTES = 256 * 1024
#: Route lanes and job modes are stable vocabularies, not free text.
ROUTE_STATUS_MODES = ("strict", "emergency")
#: `active` while the instance is serving; `closed` after a graceful shutdown.
#: A crashed instance keeps its last `active` row, which is why liveness is
#: decided by the per-instance lease below rather than by this field alone.
ROUTE_STATUS_LIFECYCLES = ("active", "closed")
#: Each publisher holds an exclusive `flock` on its own lease file for the whole
#: life of the process. The kernel drops that lock when the owning process dies
#: for any reason, including `SIGKILL` and a panic, so an uncontended lease is
#: positive proof the instance is gone rather than an inference from its pid.
#: A recorded pid can be reused by an unrelated process (the 2026-08-11 incident
#: reused one for `cloudphotod`), so a pid probe alone can never prove liveness.
ROUTE_STATUS_LEASE_SUFFIX = ".lease"
#: The only errnos that prove another owner holds the lease. POSIX allows
#: either `EWOULDBLOCK`/`EAGAIN` or `EACCES` for a contended non-blocking lock,
#: and on Linux `EWOULDBLOCK` and `EAGAIN` are the same value. Anything else --
#: `ENOTSUP`/`EOPNOTSUPP` on a filesystem without `flock`, `EIO`, `EBADF`,
#: `ENOLCK` -- means the probe failed, not that somebody is holding it.
_LEASE_CONTENDED_ERRNOS = frozenset({
    errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES,
})
#: Stable codes for a recorder that could not publish. Exception class names
#: are not a contract, so they never reach a persisted or reported field.
STATUS_FAILURE_CODES = ("status_write_failed", "status_validation_failed")
_ID_RE = re.compile(r"^[a-z0-9]{8,64}$")
_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_ACTION_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,63}$")
_LANE_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
_JOB_ID_RE = re.compile(r"^job_[a-f0-9]{24}$")
_BACKEND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")

_BUILD_ID: Optional[str] = None


def _service_source_paths() -> Tuple[Path, ...]:
    """Return the bounded source set imported by a coding worker.

    This file is intentionally part of its own inventory. Expanding the source
    set therefore also changes the digest seen by an already-running
    supervisor, allowing its next safe worker to load the expanded inventory.
    """

    package_root = Path(__file__).resolve().parents[1]
    candidates = list((package_root / "coding").glob("*.py"))
    candidates.extend((package_root / "providers").glob("*.py"))
    candidates.extend((
        package_root / "cli.py",
        package_root / "config.py",
        package_root / "agents" / "claude_code.py",
        package_root / "agents" / "codex_cli.py",
        package_root / "tools" / "core_tools.py",
    ))
    return tuple(sorted(path for path in candidates if path.is_file()))


def current_service_build_id() -> str:
    """Compute the coding control plane digest from the files on disk now.

    Unlike :func:`service_build_id`, this value is intentionally not cached.
    Long-lived MCP processes use it at request boundaries to detect that their
    already-imported Python modules no longer match the repaired source tree.
    """

    digest = hashlib.sha256()
    digest.update(b"flyto.coding-build.v2\n")
    try:
        from flyto_ai.package_metadata import package_version

        digest.update(package_version().encode("utf-8"))
    except Exception:  # pragma: no cover - metadata is best effort only
        digest.update(b"0+unknown")
    try:
        package_root = Path(__file__).resolve().parents[1]
        for path in _service_source_paths():
            digest.update(str(path.relative_to(package_root)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    except OSError:  # pragma: no cover - unreadable source tree
        pass
    return digest.hexdigest()[:32]


def service_build_id() -> str:
    """Return the immutable startup digest of this coding control plane.

    The digest covers the coding package and its bounded startup adapter/config
    dependencies, so a repaired build publishes a different id than the build
    a still-running old process loaded. When the sources cannot be read the
    package version alone is used; the result is still stable, just coarser,
    and is never fabricated per run.
    """
    global _BUILD_ID
    if _BUILD_ID is not None:
        return _BUILD_ID
    _BUILD_ID = current_service_build_id()
    return _BUILD_ID


def service_version() -> str:
    """Return the human-readable package version that owns this instance."""
    try:
        from flyto_ai.package_metadata import package_version

        return str(package_version())[:64]
    except Exception:  # pragma: no cover - metadata is best effort only
        return "0+unknown"


def _bounded_text(value: Any, pattern: "re.Pattern[str]", limit: int = 128) -> str:
    """Accept only text matching one closed pattern; anything else becomes ''."""
    if isinstance(value, bool) or not isinstance(value, str):
        return ""
    if len(value) > limit or not pattern.fullmatch(value):
        return ""
    return value


@dataclass(frozen=True)
class CodingRouteStatus:
    """One instance's latest bounded status. Closed schema, no free text."""

    instance_id: str
    build_id: str
    contract_version: str = ROUTE_STATUS_CONTRACT_VERSION
    service_version: str = ""
    process_id: int = 0
    started_at: float = 0.0
    updated_at: float = 0.0
    lifecycle: str = "active"
    implementation_backend: str = ""
    emergency_enabled: bool = False
    circuit_state: str = "closed"
    emergency_activations: int = 0
    job_id: str = ""
    state: str = ""
    mode: str = "strict"
    lane: str = ""
    action: str = ""
    failure_code: str = ""
    implementer_started: bool = False
    implementation_session_id: str = ""
    implementation_revision_sha256: str = ""
    audit_count: int = 0
    rework_count: int = 0
    landable: bool = False
    #: Whether the *job* named by `job_id` is settled. Independent of
    #: `lifecycle`, which describes this service process: an active instance
    #: may report a terminal job, and a closed instance may have left one
    #: awaiting audit.
    job_terminal: bool = False
    #: Bounded self-report from the recorder itself, so a reader can tell a
    #: quiet service from a broken one.
    publish_failures: int = 0
    last_publish_failure_code: str = ""

    def __post_init__(self) -> None:
        if self.contract_version != ROUTE_STATUS_CONTRACT_VERSION:
            raise ValueError("unsupported coding route status contract version")
        for name in ("instance_id", "build_id"):
            if not _ID_RE.fullmatch(getattr(self, name) or ""):
                raise ValueError("route status {} must be an opaque token".format(name))
        if self.state and self.state not in {item.value for item in CodingJobState}:
            raise ValueError("route status state is unknown")
        if self.mode not in ROUTE_STATUS_MODES:
            raise ValueError("route status mode is unknown")
        if self.lifecycle not in ROUTE_STATUS_LIFECYCLES:
            raise ValueError("route status lifecycle is unknown")
        if self.circuit_state not in ("closed", "open"):
            raise ValueError("route status circuit_state is unknown")
        for name in ("emergency_enabled", "implementer_started", "landable"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError("route status {} must be a boolean".format(name))
        if self.last_publish_failure_code and (
            self.last_publish_failure_code not in STATUS_FAILURE_CODES
        ):
            raise ValueError("route status last_publish_failure_code is unknown")
        for name in (
            "process_id", "audit_count", "rework_count", "emergency_activations",
            "publish_failures",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "route status {} must be a non-negative integer".format(name),
                )
        for name in ("started_at", "updated_at"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError("route status {} must be a timestamp".format(name))
            object.__setattr__(self, name, float(value))
        # Every remaining string is narrowed to its closed vocabulary rather
        # than trusted, so a malformed caller value degrades to empty instead
        # of persisting something unbounded.
        object.__setattr__(self, "service_version", str(self.service_version or "")[:64])
        if _CONTROL_CHARS_RE.search(self.service_version):
            raise ValueError("route status service_version must be printable")
        object.__setattr__(
            self, "implementation_backend",
            _bounded_text(self.implementation_backend, _BACKEND_RE, 64),
        )
        object.__setattr__(self, "job_id", _bounded_text(self.job_id, _JOB_ID_RE, 32))
        object.__setattr__(self, "lane", _bounded_text(self.lane, _LANE_RE, 32))
        object.__setattr__(self, "action", _bounded_text(self.action, _ACTION_RE, 64))
        object.__setattr__(
            self, "failure_code", _bounded_text(self.failure_code, _CODE_RE, 64),
        )
        object.__setattr__(
            self, "implementation_revision_sha256",
            _bounded_text(self.implementation_revision_sha256, _SHA256_RE, 64),
        )
        session = self.implementation_session_id
        if (
            isinstance(session, bool)
            or not isinstance(session, str)
            or len(session) > MAX_IMPLEMENTATION_SESSION_ID_CHARS
            or _CONTROL_CHARS_RE.search(session)
        ):
            session = ""
        object.__setattr__(self, "implementation_session_id", session)
        if self.landable and not self.implementer_started:
            raise ValueError("a landable round must record a started implementer")

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "instance_id": self.instance_id,
            "build_id": self.build_id,
            "service_version": self.service_version,
            "process_id": self.process_id,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "lifecycle": self.lifecycle,
            "implementation_backend": self.implementation_backend,
            "emergency_enabled": self.emergency_enabled,
            "circuit_state": self.circuit_state,
            "emergency_activations": self.emergency_activations,
            "job_id": self.job_id,
            "state": self.state,
            "mode": self.mode,
            "lane": self.lane,
            "action": self.action,
            "failure_code": self.failure_code,
            "implementer_started": self.implementer_started,
            "implementation_session_id": self.implementation_session_id,
            "implementation_revision_sha256": self.implementation_revision_sha256,
            "audit_count": self.audit_count,
            "rework_count": self.rework_count,
            "landable": self.landable,
            "publish_failures": self.publish_failures,
            "last_publish_failure_code": self.last_publish_failure_code,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CodingRouteStatus":
        """Read one persisted status back; unknown fields fail closed."""
        if not isinstance(value, Mapping):
            raise ValueError("route status must be an object")
        known = set(cls.__dataclass_fields__)
        unknown = set(value) - known
        if unknown:
            raise ValueError("unsupported route status fields")
        return cls(**{key: value[key] for key in known if key in value})

    def index_entry(self) -> Dict[str, Any]:
        """Project the compact row the shared index keeps for this instance."""
        return {
            "instance_id": self.instance_id,
            "build_id": self.build_id,
            "service_version": self.service_version,
            "process_id": self.process_id,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "lifecycle": self.lifecycle,
            "job_id": self.job_id,
            "state": self.state,
            "mode": self.mode,
            "circuit_state": self.circuit_state,
            "implementer_started": self.implementer_started,
        }


#: The index row is its own closed schema. Every row read back from disk is
#: re-projected through it, so an unknown, malformed, or oversized field is
#: dropped instead of being retained and republished.
_INDEX_ROW_SCHEMA: Dict[str, str] = {
    "instance_id": "id",
    "build_id": "id",
    "service_version": "version",
    "process_id": "count",
    "started_at": "time",
    "updated_at": "time",
    "lifecycle": "lifecycle",
    "job_id": "job",
    "state": "state",
    "mode": "mode",
    "circuit_state": "circuit",
    "implementer_started": "flag",
}


def project_index_row(value: Any) -> Optional[Dict[str, Any]]:
    """Return one validated index row, or `None` when it cannot be trusted.

    Rows come from a shared file that other processes write, so they are
    untrusted input. A row without a usable instance id is discarded outright;
    every other field degrades to its neutral value rather than propagating.
    """
    if not isinstance(value, Mapping):
        return None
    instance_id = value.get("instance_id")
    if not isinstance(instance_id, str) or not _ID_RE.fullmatch(instance_id):
        return None
    row: Dict[str, Any] = {}
    for name, kind in _INDEX_ROW_SCHEMA.items():
        item = value.get(name)
        if kind == "id":
            row[name] = _bounded_text(item, _ID_RE, 64)
        elif kind == "version":
            row[name] = (
                str(item)[:64]
                if isinstance(item, str) and not _CONTROL_CHARS_RE.search(item[:64])
                else ""
            )
        elif kind == "count":
            row[name] = (
                item if isinstance(item, int)
                and not isinstance(item, bool) and 0 <= item <= 2 ** 31 else 0
            )
        elif kind == "time":
            row[name] = (
                float(item) if isinstance(item, (int, float))
                and not isinstance(item, bool) and item >= 0 else 0.0
            )
        elif kind == "lifecycle":
            row[name] = item if item in ROUTE_STATUS_LIFECYCLES else "active"
        elif kind == "job":
            row[name] = _bounded_text(item, _JOB_ID_RE, 32)
        elif kind == "state":
            row[name] = (
                item if item in {entry.value for entry in CodingJobState} else ""
            )
        elif kind == "mode":
            row[name] = item if item in ROUTE_STATUS_MODES else "strict"
        elif kind == "circuit":
            row[name] = item if item in ("closed", "open") else "closed"
        else:
            row[name] = item is True
    return row


def process_alive(process_id: int) -> Optional[bool]:
    """Best-effort local liveness for one recorded pid.

    `None` means undecidable, not alive. This answers "does *some* process hold
    this pid", which is strictly weaker than "is that process the instance that
    recorded it": pids are reused. It is retained only as a negative
    corroborator — a false here is still a true absence — and must never be the
    sole basis for reporting an instance alive. Use `lease_alive` for that.
    """
    if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id <= 0:
        return None
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def lease_alive(lease_path: Path) -> Optional[bool]:
    """Decide liveness from an instance's lease file, immune to pid reuse.

    A live publisher holds `LOCK_EX` on this file for its whole life, so:

    * acquiring `LOCK_EX | LOCK_NB` here proves *nobody* holds it — the owner
      exited or crashed, and the kernel released it. Returns `False`.
    * failing to acquire it *with a contention errno* proves someone still
      holds it. Returns `True`.
    * `None` means undecidable — no `flock` on this platform, the file cannot
      be opened, or the lock failed for any non-contention reason such as
      `ENOTSUP` on a filesystem without `flock`. Never reported as alive.

    The probe never creates the file: a missing lease is a publisher that never
    started one, which is undecidable rather than dead, so an instance written
    by an older build is not silently declared gone.
    """
    if fcntl is None:
        # Without `flock` there is no crash-released proof of any kind. Claiming
        # liveness from a pid alone is exactly the reuse bug this replaces.
        return None
    try:
        descriptor = os.open(str(lease_path), os.O_RDWR)
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            # Only lock contention proves a holder. Every other errno means the
            # probe itself failed — an unsupported filesystem, a bad
            # descriptor, an I/O error — and reports undecidable. Treating
            # those as `True` would let a filesystem that cannot lock at all
            # report every dead instance as alive, which is the same class of
            # false positive as the pid reuse this replaces.
            return True if exc.errno in _LEASE_CONTENDED_ERRNOS else None
        # Uncontended: release immediately so this probe never becomes the
        # thing that keeps a dead instance looking alive to the next reader.
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        return False
    finally:
        os.close(descriptor)


def lease_collectable(lease_path: Path) -> bool:
    """Whether an instance's files may be removed by deterministic pruning.

    Distinct from `lease_alive`, which conflates two different `None`s. Here
    they must be told apart:

    * the lease file does not exist -- nothing holds a lock and there is no
      liveness proof to destroy, so the row is collectable. This is how an
      instance published by a build older than the lease is retired.
    * the lease exists but the probe could not decide (`ENOTSUP`, `EIO`, a
      permission failure) -- removing it could strip a live instance's only
      proof and let a second live instance create a fresh inode at the same
      path, where both would hold uncontended locks. Not collectable.
    """
    if not lease_path.exists():
        return True
    return lease_alive(lease_path) is False


def route_progress(record: Mapping[str, Any]) -> Tuple[str, str]:
    """Derive `(lane, action)` for one job record without trusting prose.

    An overflow round reports the infrastructure failure that opened the lane,
    which the host persists before the implementer is invoked. Only then does a
    strict round read its own route receipt. A record with neither reports
    empty strings rather than a guess.
    """
    trigger = record.get("emergency_trigger")
    if isinstance(trigger, Mapping):
        lane = trigger.get("lane")
        action = trigger.get("action")
        return (
            lane if isinstance(lane, str) else "",
            action if isinstance(action, str) else "",
        )
    authority = record.get("emergency_authority")
    if isinstance(authority, Mapping):
        lane = authority.get("trigger_lane")
        action = authority.get("trigger_action")
        return (
            lane if isinstance(lane, str) else "",
            action if isinstance(action, str) else "",
        )
    stored = record.get("route_receipt")
    if isinstance(stored, Mapping):
        try:
            lane, action, _code = route_failure_point(
                CodingRouteReceipt.from_mapping(stored),
            )
            return lane, action
        except ValueError:
            return "", ""
    return "", ""


def route_mode(record: Mapping[str, Any]) -> str:
    """Return the execution mode this record proves, defaulting to strict.

    The persisted `execution_mode` is written before an overflow invocation, so
    it is authoritative even while the round is still running. A completed
    authority or a persisted trigger both confirm the same thing.
    """
    if record.get("execution_mode") == "emergency":
        return "emergency"
    if isinstance(record.get("emergency_authority"), Mapping):
        return "emergency"
    if isinstance(record.get("emergency_trigger"), Mapping):
        return "emergency"
    return "strict"


class RouteStatusPublisher:
    """Own one instance's status file and its row in the bounded shared index.

    The caller must already hold the service's cross-process state guard: this
    class does the atomic 0600 writes and the deterministic pruning, not the
    locking policy.
    """

    def __init__(
        self,
        state_root: Path,
        *,
        instance_id: str,
        started_at: Optional[float] = None,
        build_id: str = "",
        version: str = "",
        process_id: Optional[int] = None,
    ) -> None:
        if not _ID_RE.fullmatch(instance_id or ""):
            raise ValueError("route status instance_id must be an opaque token")
        self.root = Path(state_root) / ROUTE_STATUS_DIRNAME
        self.instance_id = instance_id
        self.build_id = build_id or service_build_id()
        self.service_version = version or service_version()
        self.process_id = int(os.getpid() if process_id is None else process_id)
        self.started_at = float(time.time() if started_at is None else started_at)
        self._lease_fd: Optional[int] = None

    @property
    def index_path(self) -> Path:
        return self.root / ROUTE_STATUS_INDEX_FILENAME

    def instance_path(self, instance_id: str = "") -> Path:
        return self.root / "instance-{}.json".format(instance_id or self.instance_id)

    def lease_path(self, instance_id: str = "") -> Path:
        return self.instance_path(instance_id).with_suffix(ROUTE_STATUS_LEASE_SUFFIX)

    def acquire_lease(self) -> bool:
        """Take this instance's crash-released liveness lease.

        Held for the life of the process and released by the kernel however it
        dies, which is what lets a later reader distinguish a running instance
        from a crashed one without trusting a reusable pid.

        Returns `False` when no lease could be taken (no `flock`, or the file
        could not be opened). That degrades liveness to *undecidable*, never to
        a false *alive*, so it is not fatal to publishing status.
        """
        if self._lease_fd is not None:
            return True
        if fcntl is None:
            return False
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            descriptor = os.open(
                str(self.lease_path()), os.O_RDWR | os.O_CREAT, 0o600,
            )
        except OSError:
            return False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Another live process already owns this instance id. Do not hold a
            # descriptor we did not lock.
            os.close(descriptor)
            return False
        self._lease_fd = descriptor
        return True

    def release_lease(self) -> None:
        """Drop the liveness lease on graceful shutdown.

        Closing the descriptor releases the `flock`, so a reader sees this
        instance as not alive immediately rather than at the retention window.
        A crash reaches the same state without running this.
        """
        descriptor, self._lease_fd = self._lease_fd, None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except (OSError, AttributeError):
            pass
        try:
            os.close(descriptor)
        except OSError:
            pass

    def publish(self, status: CodingRouteStatus) -> CodingRouteStatus:
        """Write this instance's status, then refresh the bounded index."""
        if status.instance_id != self.instance_id:
            raise ValueError("route status belongs to another instance")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _atomic_write_json(self.instance_path(), status.to_mapping())
        self._refresh_index(status)
        return status

    def read_index(self) -> Dict[str, Any]:
        """Read the shared index through its closed schema.

        An unreadable, oversized, wrong-version, or malformed index reads as
        empty, and every surviving row is re-projected. Nothing untrusted is
        carried forward into the next write.
        """
        empty = {"contract_version": ROUTE_STATUS_CONTRACT_VERSION, "instances": []}
        try:
            if self.index_path.stat().st_size > MAX_STATUS_INDEX_BYTES:
                return empty
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return empty
        if not isinstance(value, dict) or not isinstance(value.get("instances"), list):
            return empty
        if value.get("contract_version") != ROUTE_STATUS_CONTRACT_VERSION:
            return empty
        rows: List[Dict[str, Any]] = []
        seen: set = set()
        for item in value["instances"][: MAX_STATUS_INSTANCES * 4]:
            row = project_index_row(item)
            if row is None or row["instance_id"] in seen:
                continue
            seen.add(row["instance_id"])
            rows.append(row)
        return {
            "contract_version": ROUTE_STATUS_CONTRACT_VERSION,
            "updated_at": _row_timestamp(value),
            "instances": rows,
        }

    def read_instance(self, instance_id: str) -> Optional[CodingRouteStatus]:
        """Read one instance's status file through the closed schema.

        An unreadable, oversized, or malformed file returns `None` rather than
        propagating whatever happened to be on disk.
        """
        if not _ID_RE.fullmatch(instance_id or ""):
            return None
        path = self.instance_path(instance_id)
        try:
            if path.stat().st_size > MAX_STATUS_INDEX_BYTES:
                return None
            return CodingRouteStatus.from_mapping(
                json.loads(path.read_text(encoding="utf-8")),
            )
        except (OSError, ValueError, TypeError):
            return None

    def inspect(self, *, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """Return every known instance annotated for local inspection.

        The compact index supplies the inventory; each instance's own file
        supplies the richer bounded facts when it is still readable. `stale` is
        true when either the retention window elapsed or the instance build no
        longer matches this reader's on-disk build. `alive` is a best-effort pid
        probe and may be `None`. A process started before this schema existed
        publishes no row at all, so an inventory here describes the instances
        that speak this schema and nothing else.
        """
        moment = time.time() if now is None else now
        annotated = []
        for row in self.read_index()["instances"]:
            entry = dict(row)
            detailed = self.read_instance(str(row.get("instance_id", "")))
            if detailed is not None:
                entry.update(detailed.to_mapping())
            age_stale = (
                moment - _row_timestamp(row) > STATUS_INSTANCE_TTL_SECONDS
            )
            build_stale = entry.get("build_id") != self.build_id
            alive = self._instance_alive(entry)
            entry["age_stale"] = age_stale
            entry["build_stale"] = build_stale
            entry["stale"] = age_stale or build_stale
            entry["reload_required"] = (
                build_stale
                and entry.get("lifecycle") == "active"
                and alive is not False
            )
            entry["alive"] = alive
            entry["current"] = row.get("instance_id") == self.instance_id
            annotated.append(entry)
        return annotated

    def _instance_alive(self, entry: Mapping[str, Any]) -> Optional[bool]:
        """Decide one row's liveness without ever trusting a bare pid.

        Order matters, and each step is a proof rather than a hint:

        1. `lifecycle == "closed"` is a durable record that the instance shut
           down and republished on the way out. A closed row is never alive, so
           no probe can resurrect it after its pid is reused.
        2. This reader's own row is alive while it holds its own lease. A
           read-only reader (`code-status`) holds none, so it falls through to
           the same lease probe as any other row rather than answering from an
           instance id that merely happens to match.
        3. Otherwise the crash-released lease decides.
        4. If the lease is undecidable, a pid probe may only *lower* the answer
           to `False`. It can never raise it to `True`, because that is the
           reuse bug: an unrelated process inheriting the pid would look alive.
        """
        if entry.get("lifecycle") == "closed":
            return False
        instance_id = str(entry.get("instance_id", ""))
        if instance_id == self.instance_id and self._lease_fd is not None:
            return True
        if not _ID_RE.fullmatch(instance_id):
            # Never fall through to `lease_path("")`, which would probe this
            # reader's own lease and report a foreign row as alive.
            return None
        leased = lease_alive(self.lease_path(instance_id))
        if leased is not None:
            return leased
        return False if process_alive(entry.get("process_id", 0)) is False else None

    def _refresh_index(self, status: CodingRouteStatus) -> None:
        """Replace this instance's row, then prune stale and excess instances."""
        index = self.read_index()
        rows: List[Dict[str, Any]] = [
            row for row in index["instances"]
            if row.get("instance_id") != self.instance_id
        ]
        rows.append(status.index_entry())
        now = status.updated_at or time.time()
        fresh = [
            row for row in rows
            if row.get("instance_id") == self.instance_id
            or now - _row_timestamp(row) <= STATUS_INSTANCE_TTL_SECONDS
        ]
        fresh.sort(key=_row_timestamp, reverse=True)
        kept = fresh[:MAX_STATUS_INSTANCES]
        kept_ids = {str(row.get("instance_id")) for row in kept}
        for instance_id in {str(row.get("instance_id")) for row in rows} - kept_ids:
            if instance_id == self.instance_id or not _ID_RE.fullmatch(instance_id):
                continue
            lease = self.lease_path(instance_id)
            # Collect an instance only once its lease proves it is gone. A
            # quiet process can fall out of the index on age or capacity while
            # still running; unlinking its lease would strip the only proof it
            # is alive and let a later publisher create a new inode at the same
            # path, so two live instances would each hold an uncontended lock.
            # `None` is undecidable and is treated as live, which costs one
            # retained file and never costs a false liveness answer.
            if not lease_collectable(lease):
                continue
            for path in (self.instance_path(instance_id), lease):
                try:
                    path.unlink()
                except OSError:
                    # A file another process already collected is not an error.
                    pass
        _atomic_write_json(self.index_path, {
            "contract_version": ROUTE_STATUS_CONTRACT_VERSION,
            "updated_at": now,
            "instances": kept,
        })


def _row_timestamp(row: Mapping[str, Any]) -> float:
    value = row.get("updated_at")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace one status file atomically at mode 0600."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    handle, temporary = tempfile.mkstemp(
        prefix=".status-", suffix=".tmp", dir=str(path.parent),
    )
    try:
        os.fchmod(handle, 0o600)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
