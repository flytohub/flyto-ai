# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Deterministic, host-only health monitoring for the coding control plane.

The watchdog is deliberately outside every Codex and implementation session.
It reads the same bounded status and task-window projections operators already
use, records a compact transition log, and can publish one secret-free dead-man
heartbeat to a GitHub Actions repository variable.  No model is invoked and no
job, claim, audit, source tree, or coding-service record is ever mutated.
"""
from __future__ import annotations

import hashlib
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - platform dependent
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

from flyto_ai.coding.route_status import (
    MAX_STATUS_INDEX_BYTES,
    ROUTE_STATUS_CONTRACT_VERSION,
    ROUTE_STATUS_DIRNAME,
    ROUTE_STATUS_INDEX_FILENAME,
    RouteStatusPublisher,
    service_build_id,
)
# `TaskWindowCorrupt` is deliberately not imported: it derives from
# `RuntimeError`, so naming it in an `except` tuple beside `RuntimeError` is
# unreachable duplication that only suggests a distinction that does not exist.
from flyto_ai.coding.task_window import read_task_window


WATCHDOG_SCHEMA = "flyto.coding-watchdog.v1"
WATCHDOG_HEARTBEAT_SCHEMA = "flyto.coding-watchdog-heartbeat.v1"
DEFAULT_HEALTH_DIR = "~/.flyto/health/coding"
DEFAULT_STUCK_SECONDS = 3_600
DEFAULT_ORPHAN_GRACE_SECONDS = 180
DEFAULT_GITHUB_HEARTBEAT_INTERVAL = 300
DEFAULT_GITHUB_VARIABLE = "FLYTO_CODING_HEARTBEAT"
DEFAULT_LAUNCH_INTERVAL = 60
MAX_HISTORY_BYTES = 256 * 1024
MAX_HISTORY_ARCHIVES = 4
MAX_LATEST_BYTES = 128 * 1024
# GitHub caps one Actions variable at 48 KB. The heartbeat projection is two
# orders of magnitude smaller, so exceeding this means the projection itself
# regressed and must fail locally rather than be truncated by the API.
MAX_GITHUB_VARIABLE_BYTES = 48 * 1024
MIN_STUCK_SECONDS = 60
MIN_GITHUB_HEARTBEAT_INTERVAL = 60
MIN_LAUNCH_INTERVAL = 30
# Every health record this watchdog opens by name is opened `O_NOFOLLOW`. The
# health directory is created `0o700`, but an operator may legitimately point
# `--health-dir` at a pre-existing world-writable parent such as `/tmp`, and a
# symlink planted there would otherwise redirect an append or a read to a file
# this process owns but does not intend to touch. `_atomic_write` needs no such
# flag: `os.replace` overwrites the symlink itself rather than its target.
# `getattr` because the constant is POSIX-only; a `0` there is inert, and the
# recorder already refuses to run without `fcntl` on such a platform.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$")
_VARIABLE = re.compile(r"^[A-Z][A-Z0-9_]{0,99}$")
_EXECUTION_STATES = frozenset({"queued", "running", "rework_queued", "rework_running"})
_AUDIT_STATES = frozenset({"awaiting_codex_audit"})


class WatchdogError(RuntimeError):
    """A stable host-side watchdog operation failed."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _bounded_int(value: Any, *, minimum: int, maximum: int = 31_536_000) -> Optional[int]:
    """Return an in-range plain integer, or None for anything else.

    `bool` is rejected explicitly: `True` is an `int` in Python, and letting a
    flag arrive where a threshold belongs would silently install an agent that
    polls every second.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if minimum <= value <= maximum else None


def _validate_observation_options(
    *,
    stuck_seconds: Any,
    orphan_grace_seconds: Any,
    github_repository: Any,
    github_variable: Any,
    github_heartbeat_interval: Any,
) -> None:
    """Reject every option the observing run path would later reject.

    Both `run_watchdog_once` and `launch_agent_definition` call this, so an
    installed LaunchAgent can never carry a configuration that fails on every
    wake. Validation must live in one place; two copies drift and the drift is
    only visible once the watchdog has already gone silent.
    """

    if _bounded_int(stuck_seconds, minimum=MIN_STUCK_SECONDS) is None:
        raise WatchdogError("watchdog_stuck_seconds_invalid")
    if _bounded_int(orphan_grace_seconds, minimum=0) is None:
        raise WatchdogError("watchdog_orphan_grace_invalid")
    if _bounded_int(
        github_heartbeat_interval, minimum=MIN_GITHUB_HEARTBEAT_INTERVAL,
    ) is None:
        raise WatchdogError("github_heartbeat_interval_invalid")
    if not isinstance(github_repository, str) or not isinstance(github_variable, str):
        raise WatchdogError("github_heartbeat_config_invalid")
    # The variable name is validated even with no repository configured: an
    # unusable name must fail at install time, not on the first turn that
    # actually has something unhealthy to report.
    if not _VARIABLE.fullmatch(github_variable):
        raise WatchdogError("github_heartbeat_config_invalid")
    if github_repository and not _REPOSITORY.fullmatch(github_repository):
        raise WatchdogError("github_heartbeat_config_invalid")


def _resolve_root(value: Any) -> str:
    """Normalise one root to a single absolute, symlink-free spelling.

    `realpath` rather than `abspath` because every downstream decision — the
    overlap guard, the LaunchAgent label, the tree the observer reads — must
    describe the same directory whatever path was typed to reach it. A purely
    lexical `abspath` lets `--health-dir /tmp/link` name the coding-service
    tree through a symlink, and lets install and uninstall derive two different
    labels for one state root.

    Non-strict on purpose: a health directory that does not exist yet resolves
    to itself, so the first run is checked by exactly the rules every later run
    is checked by.
    """

    return os.path.realpath(os.path.abspath(os.path.expanduser(str(value))))


def _resolve_disjoint_roots(state_root: Any, health_root: Any) -> Tuple[str, str]:
    """Resolve both roots and forbid either containing the other.

    The observer must not be able to observe its own writes. A health directory
    inside the state root would make every heartbeat mutate the tree the
    watchdog reads — self-triggering transitions at best, and at worst a
    non-AI observer writing into durable coding-service state it is explicitly
    forbidden to touch. The reverse nesting is equally wrong: the state root
    would inherit the health directory's ownership and rotation.

    Both roots are resolved through `_resolve_root` first: a lexical comparison
    of unresolved paths is not a containment check, and a symlinked health
    directory would otherwise walk straight through this guard.
    """

    state = _resolve_root(state_root)
    health = _resolve_root(health_root)
    if state == health:
        raise WatchdogError("watchdog_paths_overlap")
    try:
        shared = os.path.commonpath([state, health])
    except ValueError:  # pragma: no cover - separate drives cannot overlap
        return state, health
    if shared in (state, health):
        raise WatchdogError("watchdog_paths_overlap")
    return state, health


def _check(code: str, status: str, count: int = 0) -> Dict[str, Any]:
    if status not in {"pass", "warn", "fail"}:
        raise ValueError("watchdog check status is invalid")
    return {"code": code, "status": status, "count": max(0, int(count))}


def _level(checks: Sequence[Mapping[str, Any]]) -> str:
    statuses = {str(item.get("status", "")) for item in checks}
    if "fail" in statuses:
        return "critical"
    if "warn" in statuses:
        return "degraded"
    return "healthy"


def _fingerprint(value: Mapping[str, Any]) -> str:
    projected = {
        "health": value.get("health"),
        "reader_build_id": value.get("reader_build_id"),
        "checks": value.get("checks"),
        "counts": value.get("counts"),
    }
    payload = json.dumps(
        projected, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_watchdog(
    instances: Sequence[Mapping[str, Any]],
    window: Mapping[str, Any],
    *,
    reader_build_id: str,
    observed_at: Optional[float] = None,
    stuck_seconds: int = DEFAULT_STUCK_SECONDS,
    orphan_grace_seconds: int = DEFAULT_ORPHAN_GRACE_SECONDS,
    state_status: str = "pass",
) -> Dict[str, Any]:
    """Evaluate already-bounded status projections without exposing identities."""

    if stuck_seconds < MIN_STUCK_SECONDS or orphan_grace_seconds < 0:
        raise ValueError("watchdog thresholds are invalid")
    now = float(time.time() if observed_at is None else observed_at)
    checks: List[Dict[str, Any]] = [_check("state_readable", state_status)]

    live = [
        row for row in instances
        if row.get("lifecycle") == "active" and row.get("alive") is True
    ]
    undecidable = [
        row for row in instances
        if row.get("lifecycle") == "active" and row.get("alive") is None
    ]
    live_current = [row for row in live if row.get("build_id") == reader_build_id]
    live_stale = [row for row in live if row.get("build_id") != reader_build_id]
    active_by_job = {
        str(row.get("job_id"))
        for row in live
        if isinstance(row.get("job_id"), str) and row.get("job_id")
    }
    undecidable_by_job = {
        str(row.get("job_id"))
        for row in undecidable
        if isinstance(row.get("job_id"), str) and row.get("job_id")
    }

    tasks = [item for item in window.get("tasks", []) if isinstance(item, Mapping)]
    executing = [item for item in tasks if item.get("state") in _EXECUTION_STATES]
    awaiting_audit = [item for item in tasks if item.get("state") in _AUDIT_STATES]
    orphaned = 0
    liveness_unknown = 0
    stalled = 0
    for task in executing:
        updated_at = task.get("updated_at", 0)
        age = now - float(updated_at) if isinstance(updated_at, (int, float)) else now
        job_id = str(task.get("job_id", ""))
        if job_id in active_by_job:
            if age > stuck_seconds:
                stalled += 1
            continue
        if job_id in undecidable_by_job:
            if age > orphan_grace_seconds:
                liveness_unknown += 1
            continue
        if age > orphan_grace_seconds:
            orphaned += 1

    audit_overdue = sum(
        1
        for task in awaiting_audit
        if isinstance(task.get("updated_at"), (int, float))
        and now - float(task["updated_at"]) > stuck_seconds
    )
    publish_failures = sum(
        int(row.get("publish_failures", 0) or 0)
        for row in live
        if isinstance(row.get("publish_failures", 0), int)
    )
    emergency = sum(
        1
        for row in live
        if row.get("mode") == "emergency" or row.get("circuit_state") == "open"
    )

    checks.extend([
        _check(
            "execution_liveness",
            "fail" if orphaned else ("warn" if liveness_unknown else "pass"),
            orphaned or liveness_unknown,
        ),
        _check("execution_progress", "warn" if stalled else "pass", stalled),
        _check("codex_audit_backlog", "warn" if audit_overdue else "pass", audit_overdue),
        _check("rolling_build_reload", "warn" if live_stale else "pass", len(live_stale)),
        _check("status_recorder", "warn" if publish_failures else "pass", publish_failures),
        _check("emergency_spillway", "warn" if emergency else "pass", emergency),
    ])
    # A host can be intentionally idle with no Codex process. That is healthy
    # when no execution is stranded; requiring an always-on model process would
    # waste quota and turn a watchdog into another control plane.
    route_observed = bool(instances) or bool(tasks) or bool(window.get("available"))
    checks.append(_check("route_observed", "pass" if route_observed else "warn"))
    checks.sort(key=lambda item: item["code"])

    counts = {
        "instances": len(instances),
        "live_instances": len(live),
        "live_current_build": len(live_current),
        "live_stale_build": len(live_stale),
        "executing_tasks": len(executing),
        "awaiting_codex_audit": len(awaiting_audit),
        "orphaned_tasks": orphaned,
        "stalled_tasks": stalled,
    }
    report: Dict[str, Any] = {
        "schema": WATCHDOG_SCHEMA,
        "observed_at": int(now),
        "health": _level(checks),
        "reader_build_id": reader_build_id,
        "reason_codes": [item["code"] for item in checks if item["status"] != "pass"],
        "counts": counts,
        "checks": checks,
    }
    report["fingerprint"] = _fingerprint(report)
    return report


def _validate_status_index(state_root: Path) -> str:
    path = state_root / ROUTE_STATUS_DIRNAME / ROUTE_STATUS_INDEX_FILENAME
    if not state_root.exists():
        return "warn"
    if not state_root.is_dir() or not os.access(state_root, os.R_OK):
        return "fail"
    if not path.exists():
        return "pass"
    try:
        # The publisher owns this file, so its own bound is the authority. The
        # watchdog's smaller `MAX_LATEST_BYTES` describes records the watchdog
        # writes; applying it here would report a large-but-valid index as a
        # `state_readable` failure and manufacture an incident.
        if path.stat().st_size > MAX_STATUS_INDEX_BYTES:
            return "fail"
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "fail"
    if (
        not isinstance(value, dict)
        or value.get("contract_version") != ROUTE_STATUS_CONTRACT_VERSION
        or not isinstance(value.get("instances"), list)
    ):
        return "fail"
    return "pass"


def collect_watchdog_snapshot(
    state_root: Any,
    *,
    observed_at: Optional[float] = None,
    stuck_seconds: int = DEFAULT_STUCK_SECONDS,
    orphan_grace_seconds: int = DEFAULT_ORPHAN_GRACE_SECONDS,
) -> Dict[str, Any]:
    """Collect one read-only, secret-free health snapshot."""

    root = Path(_resolve_root(state_root))
    build_id = service_build_id()
    state_status = _validate_status_index(root)
    reader = RouteStatusPublisher(root, instance_id="0" * 24, build_id=build_id)
    instances = reader.inspect(now=observed_at)
    try:
        window = read_task_window(root, limit=200)
    except (OSError, ValueError, RuntimeError):
        window = {"available": False, "tasks": []}
        state_status = "fail"
    return evaluate_watchdog(
        instances,
        window,
        reader_build_id=build_id,
        observed_at=observed_at,
        stuck_seconds=stuck_seconds,
        orphan_grace_seconds=orphan_grace_seconds,
        state_status=state_status,
    )


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = json.dumps(
        dict(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=".watchdog-", suffix=".tmp", dir=str(path.parent),
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> Dict[str, Any]:
    """Read one bounded health record, or `{}` for anything else.

    The descriptor is opened once and then measured and read through that same
    descriptor. Checking `is_symlink()` and `stat()` by name first and reading
    by name afterwards left a window in which the record could be swapped for a
    symlink between the check and the read, which is the whole attack the
    symlink check was added to stop. `O_NOFOLLOW` refuses the symlink at open
    time instead, and `fstat` then describes exactly the file that will be read.
    """

    try:
        descriptor = os.open(str(path), os.O_RDONLY | _O_NOFOLLOW)
    except OSError:
        return {}
    try:
        if os.fstat(descriptor).st_size > MAX_LATEST_BYTES:
            return {}
        # `os.read` is permitted to return a short read, so it is drained rather
        # than called once, and the cap is re-applied to the drained total: the
        # size that `fstat` reported is a fact about the past, not a promise
        # about a file another process may still be appending to.
        chunks: List[bytes] = []
        drained = 0
        while drained <= MAX_LATEST_BYTES:
            chunk = os.read(descriptor, 65_536)
            if not chunk:
                break
            drained += len(chunk)
            chunks.append(chunk)
        if drained > MAX_LATEST_BYTES:
            return {}
        raw = b"".join(chunks)
    except OSError:
        return {}
    finally:
        os.close(descriptor)
    try:
        # `UnicodeDecodeError` is not named beside `ValueError`: it derives from
        # it, and spelling both suggests a distinction that does not exist.
        value = json.loads(raw.decode("utf-8"))
    except ValueError:
        return {}
    return dict(value) if isinstance(value, dict) else {}


class WatchdogRecorder:
    """Write latest health every run and append history only on transitions."""

    def __init__(self, health_root: Any) -> None:
        self.root = Path(_resolve_root(health_root))
        self.latest_path = self.root / "latest.json"
        self.history_path = self.root / "history.jsonl"
        self.github_state_path = self.root / "github.json"
        self.lock_path = self.root / "watchdog.lock"
        self._lock_fd: Optional[int] = None

    def __enter__(self) -> "WatchdogRecorder":
        if fcntl is None:
            raise WatchdogError("watchdog_lock_unavailable")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Opening the lock and taking it are different failures: an unopenable
        # health directory is not another watchdog, and reporting it as one
        # would hide a broken host behind a benign code.
        try:
            descriptor = os.open(
                str(self.lock_path), os.O_RDWR | os.O_CREAT | _O_NOFOLLOW, 0o600,
            )
        except OSError as exc:
            # A symlinked lock file fails closed here rather than placing this
            # watchdog's exclusive lock on a file somewhere else, where it would
            # neither exclude a second watchdog nor be found by one.
            raise WatchdogError("watchdog_lock_unavailable") from exc
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(descriptor)
            raise WatchdogError("watchdog_already_running") from exc
        self._lock_fd = descriptor
        return self

    def __exit__(self, *_exc: Any) -> None:
        descriptor, self._lock_fd = self._lock_fd, None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)

    def previous(self) -> Dict[str, Any]:
        return _read_json(self.latest_path)

    def record(self, report: Mapping[str, Any]) -> Tuple[Dict[str, Any], bool]:
        previous = self.previous()
        value = dict(report)
        value["previous_health"] = str(previous.get("health", ""))
        changed = previous.get("fingerprint") != value.get("fingerprint")
        value["transition"] = changed
        _atomic_write(self.latest_path, value)
        if changed:
            self._append_history(value)
        return value, changed

    def github_due(self, report: Mapping[str, Any], interval_seconds: int) -> bool:
        state = _read_json(self.github_state_path)
        last_sent = state.get("last_sent_at", 0)
        return (
            state.get("fingerprint") != report.get("fingerprint")
            or not isinstance(last_sent, (int, float))
            or int(report.get("observed_at", 0)) - float(last_sent) >= interval_seconds
        )

    def mark_github_sent(self, report: Mapping[str, Any]) -> None:
        _atomic_write(self.github_state_path, {
            "schema": WATCHDOG_HEARTBEAT_SCHEMA,
            "last_sent_at": int(report.get("observed_at", 0)),
            "fingerprint": str(report.get("fingerprint", "")),
        })

    def _append_history(self, value: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = (
            json.dumps(dict(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        # `lstat`, not `stat`: a symlinked history file must be measured as the
        # link it is, so rotation decides on this record rather than on whatever
        # the link points at. The append below then refuses the link outright.
        try:
            size = os.lstat(str(self.history_path)).st_size
        except FileNotFoundError:
            size = 0
        if size + len(payload) > MAX_HISTORY_BYTES:
            self._rotate_history()
        try:
            descriptor = os.open(
                str(self.history_path),
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | _O_NOFOLLOW,
                0o600,
            )
        except OSError as exc:
            # `latest.json` is already durable at this point, so the turn's
            # contract is met; a symlinked or otherwise unappendable history
            # file is a tampered health directory and gets its own stable code
            # rather than a bare OSError the caller cannot classify.
            raise WatchdogError("watchdog_history_unwritable") from exc
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _rotate_history(self) -> None:
        oldest = self.history_path.with_name(
            "{}.{}".format(self.history_path.name, MAX_HISTORY_ARCHIVES),
        )
        try:
            oldest.unlink()
        except FileNotFoundError:
            pass
        # `lexists` throughout: a dangling symlink is still a name that has to be
        # rotated out of the way, and `exists()` would report it absent and then
        # leave it in place for the next append to refuse forever.
        for number in range(MAX_HISTORY_ARCHIVES - 1, 0, -1):
            source = self.history_path.with_name("{}.{}".format(self.history_path.name, number))
            target = self.history_path.with_name("{}.{}".format(self.history_path.name, number + 1))
            if os.path.lexists(source):
                os.replace(source, target)
        if os.path.lexists(self.history_path):
            os.replace(self.history_path, self.history_path.with_name(self.history_path.name + ".1"))


def github_heartbeat_payload(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Project local health to the intentionally tiny public heartbeat schema."""

    return {
        "schema": WATCHDOG_HEARTBEAT_SCHEMA,
        "observed_at": int(report.get("observed_at", 0)),
        "health": str(report.get("health", "critical")),
        "fingerprint": str(report.get("fingerprint", "")),
        "reader_build_id": str(report.get("reader_build_id", "")),
        "reason_codes": list(report.get("reason_codes", []))[:16],
    }


def _run_gh(argv: Sequence[str]) -> Any:
    """Run one bounded `gh` call, converting every failure to a stable code.

    A hung or missing `gh` must never escape as an arbitrary exception: the
    caller is inside the one turn that still has to record health, and an
    unhandled `TimeoutExpired` there would leave the watchdog silent exactly
    when the remote dead-man switch is the only remaining witness.
    """

    try:
        return subprocess.run(
            list(argv), capture_output=True, text=True, timeout=20, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WatchdogError("github_heartbeat_failed") from exc


def publish_github_heartbeat(
    repository: str,
    variable: str,
    report: Mapping[str, Any],
    *,
    gh_command: Optional[str] = None,
) -> None:
    """Create or update one repository variable using the authenticated gh CLI."""

    if (
        not isinstance(repository, str)
        or not isinstance(variable, str)
        or not _REPOSITORY.fullmatch(repository)
        or not _VARIABLE.fullmatch(variable)
    ):
        raise WatchdogError("github_heartbeat_config_invalid")
    executable = gh_command or shutil.which("gh")
    if not executable:
        raise WatchdogError("github_cli_unavailable")
    payload = json.dumps(
        github_heartbeat_payload(report), ensure_ascii=True, sort_keys=True,
        separators=(",", ":"),
    )
    if len(payload.encode("utf-8")) > MAX_GITHUB_VARIABLE_BYTES:
        raise WatchdogError("github_heartbeat_payload_too_large")
    # `gh variable set` is upsert: it creates a missing variable and updates an
    # existing one in a single call. The previous PATCH-then-parse-404-then-POST
    # pair duplicated the same intent, doubled the window in which a hung `gh`
    # could stall the only turn that still has to record health, and decided
    # control flow by string-matching an error message the CLI never promised.
    result = _run_gh(
        [executable, "variable", "set", variable,
         "--repo", repository, "--body", payload],
    )
    if result.returncode != 0:
        raise WatchdogError("github_heartbeat_failed")


def notify_health_transition(previous: str, current: str) -> None:
    """Send one best-effort macOS notification only when health changes."""

    if sys.platform != "darwin" or previous == current:
        return
    message = "Coding route recovered" if current == "healthy" else "Coding route is {}".format(current)
    try:
        subprocess.run(
            ["/usr/bin/osascript", "-e",
             'display notification "{}" with title "Flyto watchdog"'.format(message)],
            capture_output=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def run_watchdog_once(
    *,
    state_root: Any,
    health_root: Any,
    stuck_seconds: int = DEFAULT_STUCK_SECONDS,
    orphan_grace_seconds: int = DEFAULT_ORPHAN_GRACE_SECONDS,
    github_repository: str = "",
    github_variable: str = DEFAULT_GITHUB_VARIABLE,
    github_heartbeat_interval: int = DEFAULT_GITHUB_HEARTBEAT_INTERVAL,
    notify: bool = False,
    observed_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Observe, optionally publish, and durably record one watchdog turn."""

    _validate_observation_options(
        stuck_seconds=stuck_seconds,
        orphan_grace_seconds=orphan_grace_seconds,
        github_repository=github_repository,
        github_variable=github_variable,
        github_heartbeat_interval=github_heartbeat_interval,
    )
    state_root, health_root = _resolve_disjoint_roots(state_root, health_root)
    with WatchdogRecorder(health_root) as recorder:
        report = collect_watchdog_snapshot(
            state_root,
            observed_at=observed_at,
            stuck_seconds=stuck_seconds,
            orphan_grace_seconds=orphan_grace_seconds,
        )
        github_state = "disabled"
        if github_repository and recorder.github_due(report, github_heartbeat_interval):
            try:
                publish_github_heartbeat(github_repository, github_variable, report)
                recorder.mark_github_sent(report)
                github_state = "sent"
            except (WatchdogError, OSError) as exc:
                # `OSError` is caught beside `WatchdogError` because
                # `mark_github_sent` writes `github.json`. Letting that escape
                # would abandon the turn *after* the heartbeat had already been
                # published, so the remote switch would read healthy while the
                # local record — the thing this turn exists to write — was never
                # stored. Losing only the send-interval bookkeeping is safe: the
                # next turn simply republishes an unchanged heartbeat.
                github_state = (
                    exc.code if isinstance(exc, WatchdogError)
                    else "github_state_unrecordable"
                )
                report["checks"].append(_check("github_heartbeat", "warn", 1))
                report["checks"].sort(key=lambda item: item["code"])
                report["health"] = _level(report["checks"])
                report["reason_codes"] = [
                    item["code"] for item in report["checks"] if item["status"] != "pass"
                ]
                report["fingerprint"] = _fingerprint(report)
        elif github_repository:
            github_state = "not_due"
        report["github"] = github_state
        stored, _changed = recorder.record(report)
        if notify:
            notify_health_transition(stored["previous_health"], stored["health"])
        return stored


def launch_agent_label(state_root: Any) -> str:
    """Derive the per-state-root agent label from the resolved state root.

    `_resolve_root` is what makes install and uninstall agree: the label has to
    be a property of the directory, not of the spelling the operator happened
    to type, or `--uninstall` computes a different label and silently removes
    nothing while the agent keeps waking.
    """

    digest = hashlib.sha256(_resolve_root(state_root).encode("utf-8")).hexdigest()[:12]
    return "com.flyto2.coding-watchdog.{}".format(digest)


def launch_agent_definition(
    *,
    state_root: Any,
    health_root: Any,
    interval_seconds: int = DEFAULT_LAUNCH_INTERVAL,
    stuck_seconds: int = DEFAULT_STUCK_SECONDS,
    orphan_grace_seconds: int = DEFAULT_ORPHAN_GRACE_SECONDS,
    github_repository: str = "",
    github_variable: str = DEFAULT_GITHUB_VARIABLE,
    github_heartbeat_interval: int = DEFAULT_GITHUB_HEARTBEAT_INTERVAL,
    notify: bool = True,
    executable: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the bounded launchd property list without credentials."""

    if _bounded_int(interval_seconds, minimum=MIN_LAUNCH_INTERVAL) is None:
        raise WatchdogError("watchdog_interval_invalid")
    # Every value below is baked into a plist that wakes unattended. Any option
    # `run_watchdog_once` would reject must be rejected here too, or the install
    # succeeds and then fails on every single wake with nobody watching.
    _validate_observation_options(
        stuck_seconds=stuck_seconds,
        orphan_grace_seconds=orphan_grace_seconds,
        github_repository=github_repository,
        github_variable=github_variable,
        github_heartbeat_interval=github_heartbeat_interval,
    )
    state, health = _resolve_disjoint_roots(state_root, health_root)
    arguments = [
        executable or sys.executable, "-m", "flyto_ai", "code-watchdog",
        "--state-dir", state, "--health-dir", health,
        "--stuck-seconds", str(stuck_seconds),
        "--orphan-grace-seconds", str(orphan_grace_seconds),
        "--github-heartbeat-interval", str(github_heartbeat_interval),
    ]
    if github_repository:
        arguments.extend(["--github-repository", github_repository])
        arguments.extend(["--github-variable", github_variable])
    if notify:
        arguments.append("--notify")
    return {
        "Label": launch_agent_label(state),
        "ProgramArguments": arguments,
        "RunAtLoad": True,
        "StartInterval": int(interval_seconds),
        "ProcessType": "Background",
        # launchd starts with a minimal PATH, and `gh` is normally a Homebrew or
        # user-local install. Derive the user prefix from the real home rather
        # than assuming `/Users/<login>`, which is wrong for any relocated or
        # network home directory.
        "EnvironmentVariables": {
            "PATH": os.pathsep.join((
                str(Path.home() / ".local" / "bin"),
                "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin",
            )),
        },
        "StandardOutPath": "/dev/null",
        "StandardErrorPath": "/dev/null",
    }


def install_launch_agent(**options: Any) -> Dict[str, Any]:
    """Install and bootstrap the per-state-root macOS LaunchAgent."""

    if sys.platform != "darwin":
        raise WatchdogError("watchdog_launchd_unsupported")
    definition = launch_agent_definition(**options)
    agents = Path.home() / "Library" / "LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    path = agents / "{}.plist".format(definition["Label"])
    descriptor, temporary = tempfile.mkstemp(prefix=".watchdog-", suffix=".plist", dir=str(agents))
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            plistlib.dump(definition, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    domain = "gui/{}".format(os.getuid())
    subprocess.run(
        ["/bin/launchctl", "bootout", domain, str(path)],
        capture_output=True, check=False,
    )
    loaded = subprocess.run(
        ["/bin/launchctl", "bootstrap", domain, str(path)],
        capture_output=True, check=False,
    )
    if loaded.returncode != 0:
        raise WatchdogError("watchdog_launchd_bootstrap_failed")
    return {"label": definition["Label"], "plist": str(path), "loaded": True}


def uninstall_launch_agent(state_root: Any) -> Dict[str, Any]:
    """Unload and remove the per-state-root macOS LaunchAgent."""

    if sys.platform != "darwin":
        raise WatchdogError("watchdog_launchd_unsupported")
    label = launch_agent_label(state_root)
    path = Path.home() / "Library" / "LaunchAgents" / "{}.plist".format(label)
    subprocess.run(
        ["/bin/launchctl", "bootout", "gui/{}".format(os.getuid()), str(path)],
        capture_output=True, check=False,
    )
    removed = path.exists()
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return {"label": label, "plist": str(path), "removed": removed}
