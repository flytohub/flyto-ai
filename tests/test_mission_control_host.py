# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""The mission-store host contract, proven on every interpreter in the matrix.

``flyto_ai.orchestration.mission_control`` is an *optional capability* of this
package rather than a floor under it.  It needs two host primitives - a working
inter-process ``flock`` and a ``sqlite3`` that can open a database from bytes
(``Connection.serialize``/``deserialize``, CPython 3.11 and later) - and it will
not pretend to work without them, because every guarantee it makes rests on
exactly those two things.  The package as a whole still supports the Python
floor it declares; this module is what keeps that statement honest.

So this file, unlike the behavioural suite beside it, is never skipped.  On a
supported interpreter it proves the boundary is reported truthfully and does not
get in the way; on an unsupported one it proves the far more important half:
that a refusal arrives *before* anything is created, that it is typed, that it
covers reads as well as writes, and that no quieter, less careful backend is
substituted behind the caller's back.

Everything here is deliberately written to Python 3.10 syntax and runtime, and
the unsupported host is simulated with a scoped ``monkeypatch`` double rather
than by comparing ``sys.version_info`` - the question is whether *this*
interpreter has the primitives, and a version number is only ever a proxy for
that.
"""
import os
import time

import pytest

from flyto_ai.orchestration import mission_control
from flyto_ai.orchestration.mission_control import (
    HostCapabilities,
    MissionStore,
    MissionUnsupported,
    inspect_host,
)

#: Filesystem calls that would leave a trace.  A preflight that touches any of
#: them has already broken the promise it exists to make.
_MUTATING_CALLS = (
    "open",
    "fchmod",
    "chmod",
    "mkdir",
    "makedirs",
    "unlink",
    "rmdir",
    "replace",
    "rename",
    "ftruncate",
    "write",
    "fsync",
    "truncate",
)


def _watch_filesystem(monkeypatch):
    """Record every filesystem call the module makes, and let them through."""

    seen = []

    def spy(name, original):
        def call(*args, **kwargs):
            seen.append(name)
            return original(*args, **kwargs)

        return call

    for name in _MUTATING_CALLS:
        monkeypatch.setattr(mission_control.os, name, spy(name, getattr(os, name)))
    return seen


def _pretend_unsupported(monkeypatch, *, locking=True, binding=True):
    """A scoped double for a host missing one or both primitives.

    Scoped, and only over this module's own capability probes: no global
    interpreter state is altered, and ``monkeypatch`` puts everything back.
    """

    monkeypatch.setattr(
        mission_control, "_interprocess_locking_supported", lambda: locking
    )
    monkeypatch.setattr(
        mission_control, "_database_binding_supported", lambda: binding
    )


# --------------------------------------------------------------------------
# the preflight answers truthfully, and answers without doing anything
# --------------------------------------------------------------------------


def test_inspect_host_reports_both_primitives():
    capabilities = inspect_host()
    assert isinstance(capabilities, HostCapabilities)
    assert isinstance(capabilities.interprocess_locking, bool)
    assert isinstance(capabilities.database_binding, bool)
    assert capabilities.supported == (
        capabilities.interprocess_locking and capabilities.database_binding
    )
    assert capabilities.supported == (capabilities.missing == ())


def test_inspect_host_agrees_with_the_actual_primitives():
    """Truthful, not merely self-consistent: check it against the real things."""

    capabilities = inspect_host()
    assert capabilities.database_binding == (
        callable(getattr(mission_control.sqlite3.Connection, "serialize", None))
        and callable(getattr(mission_control.sqlite3.Connection, "deserialize", None))
    )
    expected_locking = all(
        getattr(mission_control.fcntl, name, None) is not None
        for name in ("flock", "LOCK_EX", "LOCK_NB", "LOCK_UN")
    )
    assert capabilities.interprocess_locking == expected_locking


def test_inspect_host_touches_no_filesystem(monkeypatch, tmp_path):
    seen = _watch_filesystem(monkeypatch)
    inspect_host()
    monkeypatch.undo()

    assert seen == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "locking,binding,missing",
    [
        (False, True, ("interprocess-locking",)),
        (True, False, ("database-binding",)),
        (False, False, ("interprocess-locking", "database-binding")),
    ],
)
def test_inspect_host_names_what_is_missing(monkeypatch, locking, binding, missing):
    _pretend_unsupported(monkeypatch, locking=locking, binding=binding)
    capabilities = inspect_host()

    assert capabilities.supported is False
    assert capabilities.missing == missing


# --------------------------------------------------------------------------
# an unsupported host is refused, typed, and left exactly as it was found
# --------------------------------------------------------------------------


def _blocked_closure():
    """A fully accounted closure, so the argument checks are not what refuses."""

    return mission_control.Closure(
        disposition="blocked",
        rationale="the host cannot support a mission store",
        risk="the work is not scheduled anywhere",
        evidence_refs=("evidence-1",),
        owner="owner-a",
        revisit_at=int(time.time()) + 3600,
    )


def _every_public_operation(store):
    """One call per public entry point, enough to reach the boundary."""

    def create():
        store.create_mission(
            operation="op-1",
            scope="scope-a",
            objective="an objective",
            desired_result="a desired result",
            acceptance_criteria=[("crit-0", "demonstrated")],
        )

    def submit():
        store.submit_work_item(
            "m-000000000001",
            operation="op-2",
            coordinates=mission_control.WorkCoordinates("p", "r", "l"),
            root=True,
        )

    def dispatch():
        with store.dispatch(operation="op-3", worker="worker-1"):
            pass  # pragma: no cover - the dispatch never yields

    return {
        "create_mission": create,
        "submit_work_item": submit,
        "dispatch": dispatch,
        "complete_mission": lambda: store.complete_mission(
            "m-000000000001", {"crit-0": "e"}, operation="op-4"
        ),
        "abandon": lambda: store.abandon_unrunnable_work_item(
            "w-000000000001", _blocked_closure(), operation="op-5"
        ),
        "reclaim": lambda: store.reclaim("w-000000000001", operation="op-6"),
        "acknowledge_operation": lambda: store.acknowledge_operation("op-1"),
        "metrics": store.metrics,
        "snapshot": store.snapshot,
        "get_mission": lambda: store.get_mission("m-000000000001"),
        "get_work_item": lambda: store.get_work_item("w-000000000001"),
        "is_claimed": lambda: store.is_claimed(
            mission_control.MissionResource("ns", "kind", "identity")
        ),
    }


@pytest.mark.parametrize(
    "locking,binding",
    [(False, True), (True, False), (False, False)],
)
def test_every_operation_is_refused_before_anything_is_created(
    monkeypatch, tmp_path, locking, binding
):
    """Reads and writes alike, and the parent directory stays byte-for-byte empty.

    The placement is the substance of this test.  A refusal that arrives after
    the store directory has been created and moded leaves an empty store behind
    on a host that can never serve one - and the next person to find it has to
    work out whether it contains anything.  Nothing is created, so there is
    nothing to explain.
    """

    parent = tmp_path / "workspace"
    parent.mkdir()
    store = MissionStore(parent / "store")
    _pretend_unsupported(monkeypatch, locking=locking, binding=binding)

    for name, call in _every_public_operation(store).items():
        with pytest.raises(MissionUnsupported) as excinfo:
            call()
        assert "nothing was created" in str(excinfo.value), name

    monkeypatch.undo()
    # Byte for byte: not the store directory, not a lock file, not a stray
    # sidecar, not even the configured root.
    assert list(parent.iterdir()) == []
    assert not (parent / "store").exists()


def test_the_refusal_names_the_missing_primitive(monkeypatch, tmp_path):
    _pretend_unsupported(monkeypatch, binding=False)
    with pytest.raises(MissionUnsupported) as excinfo:
        MissionStore(tmp_path).metrics()

    message = str(excinfo.value)
    assert "database-binding" in message
    assert "interprocess-locking" not in message
    # And it says, in as many words, that there is no lesser mode on offer.
    assert "fall back" in message


def test_an_unsupported_host_issues_no_filesystem_calls_at_all(monkeypatch, tmp_path):
    parent = tmp_path / "workspace"
    parent.mkdir()
    store = MissionStore(parent / "store")
    _pretend_unsupported(monkeypatch, binding=False)
    seen = _watch_filesystem(monkeypatch)

    for call in (
        store.metrics,
        store.snapshot,
        lambda: store.create_mission(
            operation="op-1",
            scope="scope-a",
            objective="an objective",
            desired_result="a desired result",
            acceptance_criteria=[("crit-0", "demonstrated")],
        ),
    ):
        with pytest.raises(MissionUnsupported):
            call()

    monkeypatch.undo()
    assert seen == []
    assert list(parent.iterdir()) == []


def test_constructing_a_store_on_an_unsupported_host_is_harmless(monkeypatch, tmp_path):
    """Construction is not an operation, and must not become one.

    A service that builds its objects at import time and only later discovers it
    cannot use them should still find the disk untouched.
    """

    parent = tmp_path / "workspace"
    parent.mkdir()
    _pretend_unsupported(monkeypatch, locking=False, binding=False)

    store = MissionStore(parent / "store", queue_capacity=8)
    assert store.configured_root == parent / "store"
    assert store.root == parent / "store" / "mission-control"

    monkeypatch.undo()
    assert list(parent.iterdir()) == []


def test_no_quieter_backend_is_substituted(monkeypatch, tmp_path):
    """A missing primitive is refused; it is never routed around.

    The failure mode this rules out is the tempting one: notice that
    ``deserialize`` is unavailable, quietly hand the pathname to SQLite instead,
    and carry on with a store that still advertises a descriptor-bound storage
    boundary it no longer has.
    """

    parent = tmp_path / "workspace"
    parent.mkdir()
    store = MissionStore(parent / "store")
    _pretend_unsupported(monkeypatch, binding=False)

    connected = []
    real_connect = mission_control.sqlite3.connect

    def watching(*args, **kwargs):  # pragma: no cover - must never be reached
        connected.append(args[0] if args else None)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(mission_control.sqlite3, "connect", watching)
    with pytest.raises(MissionUnsupported):
        store.create_mission(
            operation="op-1",
            scope="scope-a",
            objective="an objective",
            desired_result="a desired result",
            acceptance_criteria=[("crit-0", "demonstrated")],
        )
    monkeypatch.undo()

    assert connected == []
    assert list(parent.iterdir()) == []


# --------------------------------------------------------------------------
# on a supported host the boundary stays out of the way
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not inspect_host().supported,
    reason="needs the primitives it is checking do not get in the way of",
)
def test_a_supported_host_is_not_refused(tmp_path):
    store = MissionStore(tmp_path)
    assert store.metrics().missions_open == 0
    mission = store.create_mission(
        operation="op-1",
        scope="scope-a",
        objective="an objective",
        desired_result="a desired result",
        acceptance_criteria=[("crit-0", "demonstrated")],
    )
    assert store.get_mission(mission.mission_id).mission_id == mission.mission_id
