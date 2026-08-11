# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Liveness proofs for the 2026-08-11 startup-authority rotation incident.

During that incident `code-status` reported a historical, already-closed row as
alive because the recorded pid had been reused by an unrelated process
(`cloudphotod`). Liveness was a bare `os.kill(pid, 0)` probe, which answers
"does some process hold this pid" rather than "is that process the instance
that recorded it".

These tests pin the four properties that make the answer trustworthy:

1. a real publisher holding its lease reads as alive;
2. a crashed publisher reads as not alive, because the kernel released the
   lease without the process running any cleanup;
3. a `closed` row is never alive, whatever any probe says;
4. an unrelated process that reuses the recorded pid does not make the row
   alive.

Plus the supervisor's bounded authority reason, which is what made the same
incident unreadable from the client: every symptom arrived as a generic
`-32603 coding worker unavailable`.
"""
import errno
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace

import pytest

from flyto_ai.coding import route_status

from flyto_ai.coding.mcp_supervisor import (
    WORKER_AUTHORITY_EXIT_CODE,
    WORKER_AUTHORITY_REASON,
    CodingMCPWorkerSupervisor,
)
from flyto_ai.coding.route_status import (
    ROUTE_STATUS_CONTRACT_VERSION,
    STATUS_INSTANCE_TTL_SECONDS,
    CodingRouteStatus,
    RouteStatusPublisher,
    lease_alive,
    service_build_id,
)


fcntl = pytest.importorskip("fcntl")


#: Takes the same lease a real publisher takes, announces it, then blocks
#: forever. Killing it proves the kernel releases the lease with no cleanup
#: code of ours running — which is exactly what a crashed `code-mcp` does.
_LEASE_HOLDER = r"""
import fcntl
import os
import sys
import time

descriptor = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
sys.stdout.write("held\n")
sys.stdout.flush()
time.sleep(600)
"""


def _instance_id(char):
    return char * 24


def _publisher(root, instance_id):
    return RouteStatusPublisher(
        root,
        instance_id=instance_id,
        build_id=service_build_id(),
        version="test",
    )


def _status(instance_id, publisher, *, lifecycle="active", process_id=None):
    return CodingRouteStatus(
        contract_version=ROUTE_STATUS_CONTRACT_VERSION,
        instance_id=instance_id,
        build_id=publisher.build_id,
        service_version=publisher.service_version,
        process_id=publisher.process_id if process_id is None else process_id,
        started_at=publisher.started_at,
        updated_at=time.time(),
        lifecycle=lifecycle,
    )


def _row(publisher, instance_id):
    for entry in publisher.inspect():
        if entry.get("instance_id") == instance_id:
            return entry
    raise AssertionError("instance {} is absent from the index".format(instance_id))


def test_live_publisher_with_a_held_lease_reads_alive(tmp_path):
    """A running instance is alive, and says so to an independent reader."""
    instance_id = _instance_id("a")
    publisher = _publisher(tmp_path, instance_id)
    assert publisher.acquire_lease() is True
    publisher.publish(_status(instance_id, publisher))

    # A second publisher object is a genuinely independent reader: it shares no
    # descriptor with the first and must decide from the lease on disk.
    reader = _publisher(tmp_path, _instance_id("b"))
    assert _row(reader, instance_id)["alive"] is True
    assert lease_alive(publisher.lease_path(instance_id)) is True

    publisher.release_lease()


def test_a_crashed_publisher_reads_not_alive_without_any_cleanup(tmp_path):
    """SIGKILL releases the lease. No shutdown path of ours gets to run."""
    instance_id = _instance_id("c")
    publisher = _publisher(tmp_path, instance_id)
    publisher.publish(_status(instance_id, publisher))
    lease = publisher.lease_path(instance_id)

    script = tmp_path / "holder.py"
    script.write_text(_LEASE_HOLDER, encoding="utf-8")
    holder = subprocess.Popen(
        (sys.executable, "-u", str(script), str(lease)),
        stdout=subprocess.PIPE,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline() == b"held\n"
        # While that process lives, the row is alive.
        reader = _publisher(tmp_path, _instance_id("d"))
        assert _row(reader, instance_id)["alive"] is True

        holder.kill()
        holder.wait(timeout=30)
    finally:
        if holder.poll() is None:  # pragma: no cover - defensive
            holder.kill()
        if holder.stdout is not None:
            holder.stdout.close()

    reader = _publisher(tmp_path, _instance_id("d"))
    assert _row(reader, instance_id)["alive"] is False
    assert lease_alive(lease) is False


def test_a_closed_row_is_never_alive_even_holding_its_own_lease(tmp_path):
    """`lifecycle=closed` is durable and outranks every probe.

    The lease is deliberately still held here. A closed row must read as not
    alive anyway, so a reused pid can never resurrect a historical row.
    """
    instance_id = _instance_id("e")
    publisher = _publisher(tmp_path, instance_id)
    assert publisher.acquire_lease() is True
    publisher.publish(_status(instance_id, publisher, lifecycle="closed"))

    reader = _publisher(tmp_path, _instance_id("f"))
    row = _row(reader, instance_id)
    assert row["lifecycle"] == "closed"
    assert row["alive"] is False
    # The publisher's own view of its own row agrees.
    assert _row(publisher, instance_id)["alive"] is False

    publisher.release_lease()


def test_an_unrelated_process_reusing_the_recorded_pid_is_not_alive(tmp_path):
    """The exact 2026-08-11 false positive: a live pid, a dead instance.

    The row records this test process's own pid, which is unambiguously a live
    pid — `os.kill(pid, 0)` succeeds. No lease is held for the instance, so the
    recorded instance is gone and the row must not read as alive.
    """
    instance_id = _instance_id("g")
    publisher = _publisher(tmp_path, instance_id)
    publisher.publish(
        _status(instance_id, publisher, process_id=os.getpid()),
    )
    # Establish the lease file exists but is unheld, as it would be after the
    # owner died and the kernel released it.
    publisher.lease_path(instance_id).touch(mode=0o600)

    reader = _publisher(tmp_path, _instance_id("h"))
    row = _row(reader, instance_id)
    assert row["process_id"] == os.getpid()
    os.kill(os.getpid(), 0)  # the recorded pid really is live
    assert row["alive"] is False


def test_a_released_lease_reports_not_alive_after_graceful_close(tmp_path):
    """Graceful shutdown reaches the same state a crash does."""
    instance_id = _instance_id("i")
    publisher = _publisher(tmp_path, instance_id)
    assert publisher.acquire_lease() is True
    publisher.publish(_status(instance_id, publisher))
    publisher.release_lease()

    reader = _publisher(tmp_path, _instance_id("j"))
    assert _row(reader, instance_id)["alive"] is False


def test_a_missing_lease_is_undecidable_rather_than_alive(tmp_path):
    """An instance from a build that published no lease is never claimed alive."""
    instance_id = _instance_id("k")
    publisher = _publisher(tmp_path, instance_id)
    publisher.publish(_status(instance_id, publisher, process_id=os.getpid()))
    assert not publisher.lease_path(instance_id).exists()

    reader = _publisher(tmp_path, _instance_id("l"))
    assert _row(reader, instance_id)["alive"] is None


#: Refuses at the state-root authority exactly the way `code-mcp` does.
_AUTHORITY_REFUSING_WORKER = r"""
import sys
sys.exit({exit_code})
""".format(exit_code=WORKER_AUTHORITY_EXIT_CODE)


#: Dies for an ordinary reason, which must keep the generic reason.
_ORDINARY_DYING_WORKER = r"""
import sys
sys.exit(1)
"""


def _run_one_request(tmp_path, source, name):
    script = tmp_path / name
    script.write_text(source, encoding="utf-8")
    supervisor = CodingMCPWorkerSupervisor(
        (sys.executable, "-u", str(script)),
        build_id_provider=lambda: "build",
    )
    try:
        raw = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        }, separators=(",", ":")).encode() + b"\n"
        return json.loads(supervisor.handle_line(raw))
    finally:
        supervisor.close()


def test_supervisor_reports_a_bounded_authority_reason(tmp_path):
    """An authority refusal is actionable, not a generic transport failure."""
    value = _run_one_request(
        tmp_path, _AUTHORITY_REFUSING_WORKER, "refuse.py",
    )
    assert value["error"]["code"] == -32603
    assert value["error"]["message"] == WORKER_AUTHORITY_REASON
    assert "code-status" in value["error"]["message"]
    assert "code-release" in value["error"]["message"]


def test_the_authority_reason_leaks_nothing(tmp_path):
    """A fixed sentence selected by exit code cannot carry worker state."""
    value = _run_one_request(
        tmp_path, _AUTHORITY_REFUSING_WORKER, "refuse.py",
    )
    message = value["error"]["message"]
    assert str(tmp_path) not in message
    assert "/" not in message.replace("code-status", "").replace("code-release", "")
    assert "\n" not in message and len(message) <= 200


def test_an_ordinary_worker_death_keeps_the_generic_reason(tmp_path):
    """Only exit code 78 earns the specific reason."""
    value = _run_one_request(tmp_path, _ORDINARY_DYING_WORKER, "die.py")
    assert value["error"]["code"] == -32603
    assert value["error"]["message"] == "coding worker unavailable"


def test_public_mcp_inventory_is_unchanged():
    """The repair adds no tool. The public surface stays exactly three."""
    from flyto_ai.coding.mcp_server import CodingMCPServer

    assert tuple(tool["name"] for tool in CodingMCPServer._tools()) == (
        "flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit",
    )


def test_signal_module_is_available_for_the_crash_proof():
    """Guards the crash test's premise on platforms without SIGKILL."""
    assert hasattr(signal, "SIGKILL")


@pytest.mark.parametrize("code", sorted({errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES}))
def test_only_contention_errnos_prove_a_holder(tmp_path, monkeypatch, code):
    """A contended lock is the one positive liveness answer."""
    lease = tmp_path / "instance-{}.lease".format(_instance_id("m"))
    lease.touch(mode=0o600)

    def _refuse(descriptor, operation):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(route_status.fcntl, "flock", _refuse)
    assert lease_alive(lease) is True


@pytest.mark.parametrize("name", ["ENOTSUP", "EOPNOTSUPP", "EIO", "EBADF", "ENOLCK"])
def test_a_broken_or_unsupporting_filesystem_is_undecidable(tmp_path, monkeypatch, name):
    """`flock` failing for any non-contention reason must never read alive.

    A filesystem that cannot lock at all would otherwise report *every* dead
    instance as alive — the same class of false positive as the pid reuse this
    replaces, but applied to every row at once.
    """
    code = getattr(errno, name, None)
    if code is None or code in {errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES}:
        pytest.skip("{} is unavailable or aliases a contention errno".format(name))
    lease = tmp_path / "instance-{}.lease".format(_instance_id("n"))
    lease.touch(mode=0o600)

    def _fail(descriptor, operation):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(route_status.fcntl, "flock", _fail)
    assert lease_alive(lease) is None


def test_an_undecidable_lease_is_never_reported_alive(tmp_path, monkeypatch):
    """The undecidable errno reaches `inspect()` as `None`, not `True`."""
    instance_id = _instance_id("o")
    publisher = _publisher(tmp_path, instance_id)
    publisher.publish(_status(instance_id, publisher, process_id=os.getpid()))
    publisher.lease_path(instance_id).touch(mode=0o600)

    reader = _publisher(tmp_path, _instance_id("p"))

    def _fail(descriptor, operation):
        raise OSError(errno.EIO, "io")

    monkeypatch.setattr(route_status.fcntl, "flock", _fail)
    assert _row(reader, instance_id)["alive"] is None


def test_pruning_never_unlinks_a_held_lease(tmp_path):
    """A quiet but live instance keeps the lease that proves it is alive.

    It falls out of the index on age, which is correct — but unlinking its
    lease would strip the proof and let a later publisher create a fresh inode
    at the same path, where two live instances would each hold an uncontended
    lock and both read as alive.
    """
    quiet_id = _instance_id("q")
    quiet = _publisher(tmp_path, quiet_id)
    assert quiet.acquire_lease() is True
    stale_moment = time.time() - (STATUS_INSTANCE_TTL_SECONDS * 2)
    quiet.publish(replace(_status(quiet_id, quiet), updated_at=stale_moment))

    # A second publisher's refresh prunes the aged row.
    other = _publisher(tmp_path, _instance_id("r"))
    assert other.acquire_lease() is True
    other.publish(_status(_instance_id("r"), other))

    assert quiet.lease_path(quiet_id).exists()
    assert lease_alive(quiet.lease_path(quiet_id)) is True

    quiet.release_lease()
    other.release_lease()


def test_pruning_still_collects_an_unheld_lease(tmp_path):
    """Cleanup is not disabled — a genuinely dead instance is still collected."""
    dead_id = _instance_id("s")
    dead = _publisher(tmp_path, dead_id)
    assert dead.acquire_lease() is True
    dead.publish(replace(
        _status(dead_id, dead),
        updated_at=time.time() - (STATUS_INSTANCE_TTL_SECONDS * 2),
    ))
    dead.release_lease()  # the instance exits

    other = _publisher(tmp_path, _instance_id("t"))
    other.publish(_status(_instance_id("t"), other))

    assert not dead.lease_path(dead_id).exists()
    assert not dead.instance_path(dead_id).exists()
