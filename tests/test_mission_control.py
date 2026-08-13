# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Tests for the provider-neutral durable mission scheduling kernel.

Every scope, coordinate, resource and evidence ref used here is a made-up
bounded token, so no test can accidentally encode a domain vocabulary or depend
on a literal product name.  The properties the suite exists to pin down are the
ones the kernel is built around: the mission contract is immutable, the work
graph is a rooted DAG with return edges that actually point home, a work item
leaves the queue only with an accounted disposition, durable state is shared
across independent processes rather than owned by a Python object, dispatch is
deterministic and starvation-resistant, execution authority is an OS lease and
never a timestamp, and every ambiguous or damaged answer fails closed.

The host-capability tests substitute the primitive, never the kernel: they
replace the ``fcntl`` module the store reaches for, plant a real symbolic link,
or write junk into the durable database, and then assert on the store's public
behaviour.  The contention test uses two real ``multiprocessing.Process``
workers against one store directory, because a threading test would prove
nothing about the inter-process lock that the whole exclusion contract rests on.
Mode assertions are exact rather than masked, and are skipped off POSIX, where
the bits do not mean what they say.
"""
import dataclasses
import errno
import fcntl
import inspect
import itertools
import json
import multiprocessing
import os
import queue
import sqlite3
import stat
import time
import types
from pathlib import Path

import pytest

from flyto_ai.orchestration import mission_control
from flyto_ai.orchestration.mission_control import (
    DEFAULT_SNAPSHOT_LIMIT,
    DIRECTORY_MODE,
    DISPOSITION_BLOCKED,
    DISPOSITION_DEFERRED,
    DISPOSITION_FIXED,
    FILE_MODE,
    LANE_PRIMARY,
    LANE_REPAIR,
    MAX_SNAPSHOT_ITEMS,
    MAX_DEPENDENCIES,
    MAX_DISPATCH_CANDIDATES,
    MAX_PRIORITY,
    MISSION_COMPLETED,
    MISSION_OPEN,
    STATUS_CLOSED,
    STATUS_DISPATCHED,
    STATUS_READY,
    AcceptanceCriterion,
    Closure,
    MissionCapacityExceeded,
    MissionConflict,
    MissionCorrupt,
    MissionDisplaced,
    MissionError,
    MissionHostFailure,
    MissionIndeterminate,
    MissionOperationConflict,
    MissionOperationSettled,
    MissionRejected,
    MissionResource,
    MissionStaleFence,
    MissionStore,
    MissionUnauthorized,
    MissionUnsupported,
    WorkCoordinates,
)

_POSIX = os.name == "posix"
_OPERATIONS = itertools.count()

#: Everything below exercises real store behaviour, which needs both primitives
#: this kernel refuses to emulate.  The support boundary itself is *not* skipped
#: with them - it lives in ``test_mission_control_host.py``, which runs
#: everywhere, so the one thing an unsupported host must still get right is the
#: one thing that is always proven.
#:
#: Probed, never compared against a version number: the question is whether this
#: interpreter has the primitives, and a version is only ever a proxy for that.
_HOST = mission_control.inspect_host()
pytestmark = pytest.mark.skipif(
    not _HOST.supported,
    reason=(
        "mission store behaviour needs "
        + ", ".join(_HOST.missing or ("-",))
        + "; the host contract is covered by tests/test_mission_control_host.py"
    ),
)


def _key(prefix: str = "op") -> str:
    """A fresh operation identity, for calls whose retry story is not the point."""

    return f"{prefix}-{next(_OPERATIONS):08d}"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _coords(name: str = "alpha") -> WorkCoordinates:
    return WorkCoordinates(
        project=f"project-{name}", repository=f"repo-{name}", location=f"path-{name}"
    )


def _resource(name: str) -> MissionResource:
    return MissionResource(namespace="ns-test", kind="kind-test", identity=f"res-{name}")


def _fixed() -> Closure:
    return Closure(disposition=DISPOSITION_FIXED, evidence_refs=("evidence-fixed",))


def _deferred(owner: str = "owner-a") -> Closure:
    return Closure(
        disposition=DISPOSITION_DEFERRED,
        rationale="the upstream contract is not published yet",
        risk="the gap stays open until the contract lands",
        evidence_refs=("evidence-deferred",),
        owner=owner,
        revisit_at=int(time.time()) + 3600,
    )


def _blocked(owner: str = "owner-b") -> Closure:
    return Closure(
        disposition=DISPOSITION_BLOCKED,
        rationale="the target environment refuses the operation",
        risk="the objective cannot be reached without an operator",
        evidence_refs=("evidence-blocked",),
        owner=owner,
        revisit_at=int(time.time()) + 7200,
    )


def _mission(store: MissionStore, *, scope: str = "scope-a", criteria: int = 1):
    return store.create_mission(
        operation=_key("mission"),
        scope=scope,
        objective="reach the stated outcome without unrecorded skips",
        desired_result="the outcome holds and every step is accounted for",
        acceptance_criteria=[
            (f"crit-{index}", f"statement {index}") for index in range(criteria)
        ],
    )


def _root(store: MissionStore, mission_id: str, **kwargs):
    kwargs.setdefault("coordinates", _coords())
    kwargs.setdefault("operation", _key("root"))
    return store.submit_work_item(mission_id, root=True, **kwargs)


def _side(store: MissionStore, mission_id: str, parent, return_to=None, **kwargs):
    kwargs.setdefault("coordinates", _coords("side"))
    kwargs.setdefault("operation", _key("side"))
    return store.submit_work_item(
        mission_id,
        parent_id=parent.work_item_id,
        return_to_id=(return_to or parent).work_item_id,
        **kwargs,
    )


def _run_one(store: MissionStore, closure: Closure = None, worker: str = "worker-1"):
    """Dispatch the next runnable item and close it, returning its id."""

    with store.dispatch(operation=_key('dispatch'), worker=worker) as handle:
        if handle is None:
            return None
        work_item_id = handle.work_item_id
        handle.close(closure or _fixed(), operation=_key('close'))
    return work_item_id


def _sql(store: MissionStore, statement: str, params=()) -> int:
    """Reach behind the kernel and damage durable state on purpose.

    Returns the number of rows the statement touched.  A corruption test that
    does not check this can pass for the wrong reason: a ``WHERE`` clause that
    matches nothing changes the file (SQLite bumps its change counter either
    way) while leaving the state perfectly valid, and the assertion that the
    store now fails closed would then be proving something else entirely.
    """

    conn = sqlite3.connect(str(store.root / mission_control._DB_NAME))
    try:
        touched = conn.execute(statement, params).rowcount
        conn.commit()
    finally:
        conn.close()
    return touched


def _counter(store: MissionStore, name: str) -> int:
    return _query(store, "SELECT value FROM counters WHERE name = ?", (name,))[0][0]


# --------------------------------------------------------------------------
# 1. the mission contract is immutable
# --------------------------------------------------------------------------


def test_mission_contract_is_frozen_and_has_no_edit_path(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)

    for field in ("objective", "desired_result", "acceptance_criteria"):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(mission, field, "something else")

    # There is no amend/update/set path on the public surface either: an
    # objective that changes after work closed against it is unauditable.
    surface = [name for name in dir(store) if not name.startswith("_")]
    assert not [
        name
        for name in surface
        if "objective" in name or "amend" in name or name.startswith(("update", "set_"))
    ]


def test_mission_contract_is_unchanged_by_a_whole_lifecycle(tmp_path):
    store = MissionStore(tmp_path)
    criteria = [("crit-a", "the outcome is demonstrated")]
    mission = store.create_mission(
        scope="scope-a",
        objective="the original objective",
        desired_result="the original desired result",
        acceptance_criteria=criteria,
    operation=_key('mission'))
    # Mutating the caller's list afterwards must not reach the store.
    criteria.append(("crit-b", "a criterion nobody agreed to"))

    root = _root(store, mission.mission_id)
    _run_one(store)
    completed = store.complete_mission(mission.mission_id, {"crit-a": "evidence-a"}, operation=_key('complete'))

    reread = store.get_mission(mission.mission_id)
    assert reread.objective == "the original objective"
    assert reread.desired_result == "the original desired result"
    assert reread.criteria_ids == ("crit-a",)
    assert completed.status == MISSION_COMPLETED
    assert completed.acceptance_evidence == (("crit-a", "evidence-a"),)
    assert store.get_work_item(root.work_item_id).status == STATUS_CLOSED


# --------------------------------------------------------------------------
# 2. the work graph is a rooted, cycle-free DAG with real return edges
# --------------------------------------------------------------------------


def test_side_work_item_requires_parent_and_return_edge(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)

    with pytest.raises(MissionRejected):
        store.submit_work_item(mission.mission_id, coordinates=_coords(), operation=_key('submit'))

    root = _root(store, mission.mission_id)

    with pytest.raises(MissionRejected):
        store.submit_work_item(
            mission.mission_id, coordinates=_coords(), parent_id=root.work_item_id
        , operation=_key('submit'))
    with pytest.raises(MissionRejected):
        store.submit_work_item(
            mission.mission_id, coordinates=_coords(), return_to_id=root.work_item_id
        , operation=_key('submit'))
    with pytest.raises(MissionRejected):
        store.submit_work_item(
            mission.mission_id,
            coordinates=_coords(),
            parent_id="w-000000000999",
            return_to_id=root.work_item_id,
        operation=_key('submit'))
    with pytest.raises(MissionRejected):
        _root(store, mission.mission_id)

    side = _side(store, mission.mission_id, root)
    assert side.parent_id == root.work_item_id
    assert side.return_to_id == root.work_item_id
    assert not side.is_root


def test_return_edge_must_point_toward_the_root(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)
    left = _side(store, mission.mission_id, root)
    right = _side(store, mission.mission_id, root)

    # A sibling is not on the path home.
    with pytest.raises(MissionRejected):
        _side(store, mission.mission_id, right, return_to=left)

    # The parent and the grandparent both are.
    deep = _side(store, mission.mission_id, right, return_to=root)
    assert deep.return_to_id == root.work_item_id
    assert _side(store, mission.mission_id, deep, return_to=right).return_to_id == (
        right.work_item_id
    )


def test_an_injected_parent_cycle_is_refused_as_corrupt(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)
    first = _side(store, mission.mission_id, root)
    second = _side(store, mission.mission_id, first)

    _sql(
        store,
        "UPDATE work_items SET parent_id = ? WHERE work_item_id = ?",
        (second.work_item_id, first.work_item_id),
    )

    with pytest.raises(MissionCorrupt):
        _side(store, mission.mission_id, second)


def test_completion_refuses_a_return_edge_that_stopped_pointing_home(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)
    side = _side(store, mission.mission_id, root)
    _run_one(store)
    _run_one(store)

    _sql(
        store,
        "UPDATE work_items SET return_to_id = ? WHERE work_item_id = ?",
        (side.work_item_id, side.work_item_id),
    )
    with pytest.raises(MissionCorrupt):
        store.complete_mission(mission.mission_id, {"crit-0": "evidence-a"}, operation=_key('complete'))


def test_mission_completes_only_with_evidence_and_closed_work(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store, criteria=2)
    root = _root(store, mission.mission_id)
    _side(store, mission.mission_id, root)

    # Nothing closed yet.
    with pytest.raises(MissionRejected):
        store.complete_mission(mission.mission_id, {"crit-0": "e-0", "crit-1": "e-1"}, operation=_key('complete'))

    _run_one(store)
    _run_one(store)

    with pytest.raises(MissionRejected):  # missing evidence
        store.complete_mission(mission.mission_id, {"crit-0": "e-0"}, operation=_key('complete'))
    with pytest.raises(MissionRejected):  # evidence for a criterion nobody agreed
        store.complete_mission(
            mission.mission_id, {"crit-0": "e-0", "crit-1": "e-1", "crit-9": "e-9"}
        , operation=_key('complete'))

    completed = store.complete_mission(mission.mission_id, {"crit-0": "e-0", "crit-1": "e-1"}, operation=_key('complete'))
    assert completed.status == MISSION_COMPLETED
    with pytest.raises(MissionRejected):
        store.complete_mission(mission.mission_id, {"crit-0": "e-0", "crit-1": "e-1"}, operation=_key('complete'))


# --------------------------------------------------------------------------
# 3. no silent skip: deferred and blocked carry their whole accounting
# --------------------------------------------------------------------------


def test_a_work_item_closes_only_with_a_known_disposition(tmp_path):
    with pytest.raises(MissionRejected):
        Closure(disposition="skipped")
    with pytest.raises(MissionRejected):
        Closure(disposition="wontfix")


@pytest.mark.parametrize("disposition", [DISPOSITION_DEFERRED, DISPOSITION_BLOCKED])
def test_deferred_and_blocked_require_full_accounting(disposition):
    complete = dict(
        rationale="the upstream contract is not published yet",
        risk="the gap stays open until the contract lands",
        evidence_refs=("evidence-1",),
        owner="owner-a",
        revisit_at=int(time.time()) + 3600,
    )
    assert Closure(disposition=disposition, **complete).disposition == disposition

    for omitted in ("rationale", "risk", "evidence_refs", "owner", "revisit_at"):
        partial = dict(complete)
        partial[omitted] = () if omitted == "evidence_refs" else None
        with pytest.raises(MissionRejected) as excinfo:
            Closure(disposition=disposition, **partial)
        assert omitted in str(excinfo.value)


def test_fixed_has_nothing_to_revisit(tmp_path):
    with pytest.raises(MissionRejected):
        Closure(disposition=DISPOSITION_FIXED, revisit_at=int(time.time()) + 60)


def test_revisit_time_must_point_at_the_future(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    _root(store, mission.mission_id)

    stale_revisit = Closure(
        disposition=DISPOSITION_BLOCKED,
        rationale="the target environment refuses the operation",
        risk="the objective is unreachable until an operator intervenes",
        evidence_refs=("evidence-1",),
        owner="owner-a",
        revisit_at=1,
    )
    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as handle:
        with pytest.raises(MissionRejected):
            handle.close(stale_revisit, operation=_key('close'))
        handle.close(_deferred(), operation=_key('close'))

    item = store.get_work_item(handle.work_item_id)
    assert item.status == STATUS_CLOSED
    assert item.disposition == DISPOSITION_DEFERRED
    assert item.closure.owner == "owner-a"
    assert item.closure.evidence_refs == ("evidence-deferred",)
    assert item.closure.revisit_at > time.time()


# --------------------------------------------------------------------------
# 4. capacity and durability belong to the store, not to a Python object
# --------------------------------------------------------------------------


def test_queue_capacity_is_global_and_survives_restart(tmp_path):
    store = MissionStore(tmp_path, queue_capacity=2)
    mission = _mission(store)
    root = _root(store, mission.mission_id)
    _side(store, mission.mission_id, root)

    with pytest.raises(MissionCapacityExceeded):
        _side(store, mission.mission_id, root)
    assert store.metrics().capacity_rejects == 1

    # A different Python object over the same directory shares the same queue.
    restarted = MissionStore(tmp_path)
    assert restarted.get_mission(mission.mission_id).objective == mission.objective
    assert restarted.metrics().queue_depth == 2
    with pytest.raises(MissionCapacityExceeded):
        _side(restarted, mission.mission_id, root)
    assert restarted.metrics().capacity_rejects == 2

    # And capacity is not a per-object opinion.
    with pytest.raises(MissionRejected):
        _side(MissionStore(tmp_path, queue_capacity=99), mission.mission_id, root)

    _run_one(restarted)
    assert MissionStore(tmp_path).metrics().queue_depth == 1
    _side(MissionStore(tmp_path), mission.mission_id, root)


def test_durable_state_is_owner_only_on_disk(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    _root(store, mission.mission_id)
    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as handle:
        assert handle is not None
        lease_dir = store.root / "leases"
        leases = list(lease_dir.iterdir())

        if not _POSIX:  # pragma: no cover - POSIX mode bits are meaningless here
            return
        assert stat.S_IMODE(store.root.stat().st_mode) == DIRECTORY_MODE
        assert stat.S_IMODE(lease_dir.stat().st_mode) == DIRECTORY_MODE
        assert stat.S_IMODE((store.root / "missions.db").stat().st_mode) == FILE_MODE
        assert stat.S_IMODE((store.root / "store.id").stat().st_mode) == FILE_MODE
        assert leases and all(stat.S_IMODE(p.stat().st_mode) == FILE_MODE for p in leases)
        # There is deliberately no lock *file*: a lock pathname can be unlinked
        # and recreated by anyone who can write the directory, which is how two
        # processes end up locking two inodes and losing each other's writes.
        assert not (store.root / "missions.lock").exists()


# --------------------------------------------------------------------------
# 5. deterministic order, repair preference and anti-starvation fairness
# --------------------------------------------------------------------------


def test_scheduler_order_is_the_same_read_only_preference_as_dispatch(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    lower = _root(store, mission.mission_id, priority=1)
    higher = _side(store, mission.mission_id, lower, priority=9)
    before = store.metrics()

    assert store.scheduler_order() == (higher.work_item_id, lower.work_item_id)
    assert store.metrics() == before


def test_targeted_dispatch_never_falls_through_and_recovers_same_receipt(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    expected = _root(store, mission.mission_id, priority=1)
    displaced = _side(store, mission.mission_id, expected, priority=9)
    with store.dispatch_expected(
        operation=_key("displaced-candidate"),
        worker="worker-1",
        work_item_id=expected.work_item_id,
        expected_attempt=expected.attempts + 1,
    ) as missing:
        assert missing is None
    assert store.get_work_item(expected.work_item_id).status == STATUS_READY
    assert store.get_work_item(displaced.work_item_id).status == STATUS_READY

    operation = _key("targeted-dispatch")
    with store.dispatch_expected(
        operation=operation,
        worker="worker-1",
        work_item_id=displaced.work_item_id,
        expected_attempt=displaced.attempts + 1,
    ) as handle:
        assert handle is not None
        assert handle.work_item_id == displaced.work_item_id
        fence = handle.fence
        # Stand in for a hard crash: release only the OS lease, without requeue.
        handle._released = True
        handle._lease.released = True
        mission_control._drop_lease(handle._lease.fd)
        store._leases.pop(handle.work_item_id)

    restarted = MissionStore(tmp_path)
    with restarted.dispatch_expected(
        operation=operation,
        worker="worker-1",
        work_item_id=displaced.work_item_id,
        expected_attempt=displaced.attempts + 1,
    ) as recovered:
        assert recovered is not None
        assert recovered.work_item_id == displaced.work_item_id
        assert recovered.fence == fence
        recovered.close(_fixed(), operation=_key("targeted-close"))

    assert restarted.get_work_item(expected.work_item_id).status == STATUS_READY


def test_dispatch_order_is_priority_then_age(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id, priority=0)
    older = _side(store, mission.mission_id, root, priority=5)
    newer = _side(store, mission.mission_id, root, priority=5)
    urgent = _side(store, mission.mission_id, root, priority=9)

    order = [_run_one(store) for _ in range(4)]
    assert order == [
        urgent.work_item_id,
        older.work_item_id,
        newer.work_item_id,
        root.work_item_id,
    ]
    assert _run_one(store) is None


def test_repair_lane_is_preferred_over_higher_priority_primary_work(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id, priority=9)
    repair = _side(store, mission.mission_id, root, priority=0, lane=LANE_REPAIR)

    assert _run_one(store) == repair.work_item_id
    assert _run_one(store) == root.work_item_id


def test_fairness_rotates_scopes_so_one_producer_cannot_monopolise(tmp_path):
    store = MissionStore(tmp_path)
    loud = _mission(store, scope="scope-loud")
    quiet = _mission(store, scope="scope-quiet")

    loud_root = _root(store, loud.mission_id, priority=9)
    for _ in range(3):
        _side(store, loud.mission_id, loud_root, priority=9)
    _root(store, quiet.mission_id, priority=1)

    scopes = []
    while True:
        with store.dispatch(operation=_key('dispatch'), worker="worker-1") as handle:
            if handle is None:
                break
            scopes.append(handle.scope)
            handle.close(_fixed(), operation=_key('close'))

    assert len(scopes) == 5
    # The quiet, lower-priority tenant is served second, not last: the loud
    # tenant cannot push it behind four items by submitting more.
    assert scopes[:2] == ["scope-loud", "scope-quiet"]
    assert scopes[2:] == ["scope-loud"] * 3


# --------------------------------------------------------------------------
# 6. resource exclusion across real processes, and the repair lane inside it
# --------------------------------------------------------------------------


def _reap(processes, *queues) -> None:
    """Terminate, join and drain, whatever happened above.

    A test that fails while a child still holds the store lock would otherwise
    leave that child alive, and pytest would wait on the queue's feeder thread
    at exit - a hang that looks like a slow test rather than a broken one, which
    is exactly the way an assertion failure gets mistaken for a timeout.
    """

    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=30)
    for channel in queues:
        channel.close()
        channel.join_thread()


def _contend(root: str, index: int, start, hold, results) -> None:
    """Dispatch once from an independent process and hold whatever it won."""

    store = MissionStore(root)
    start.wait(timeout=120)
    with store.dispatch(operation=f"contend-{index}", worker="worker-contend") as handle:
        results.put("" if handle is None else handle.work_item_id)
        hold.wait(timeout=120)


def test_two_processes_contending_for_one_resource_produce_one_winner(tmp_path):
    store = MissionStore(tmp_path)
    shared = _resource("shared")
    private = {}
    for name in ("a", "b"):
        mission = _mission(store, scope=f"scope-{name}")
        own = _resource(name)
        item = _root(
            store, mission.mission_id, coordinates=_coords(name), resources=[shared, own]
        )
        private[item.work_item_id] = own

    context = multiprocessing.get_context("spawn")
    start = context.Barrier(2)
    hold = context.Barrier(3)
    results = context.Queue()
    processes = [
        context.Process(target=_contend, args=(str(tmp_path), index, start, hold, results))
        for index in range(2)
    ]
    for process in processes:
        process.start()

    try:
        outcomes = [results.get(timeout=60) for _ in range(2)]
        winners = [outcome for outcome in outcomes if outcome]

        assert len(winners) == 1, outcomes
        winner = winners[0]
        assert store.get_work_item(winner).status == STATUS_DISPATCHED

        # The contested resource is held exactly once, and the loser left no
        # partial claim behind on the resource nobody else wanted.
        assert store.is_claimed(shared)
        assert store.is_claimed(private[winner])
        for item_id, resource in private.items():
            if item_id != winner:
                assert not store.is_claimed(resource)
                assert store.get_work_item(item_id).status == STATUS_READY
        assert store.metrics().conflicts >= 1
        assert store.metrics().dispatches == 1
    finally:
        try:
            hold.wait(timeout=60)
            for process in processes:
                process.join(timeout=60)
        finally:
            _reap(processes, results)

    assert [process.exitcode for process in processes] == [0, 0]
    # Both leases are gone now, so the unclaimed item is runnable again.
    assert not store.is_claimed(shared)


def test_repair_work_cannot_bypass_a_resource_conflict(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    contested = _resource("contested")
    root = _root(store, mission.mission_id, resources=[contested])

    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as held:
        assert held.work_item_id == root.work_item_id
        # The repair is raised while the primary work is still running, and it
        # wants the very same resource.  Preferring the repair lane reorders the
        # queue; it does not let a repair walk through a live claim.
        repair = _side(
            store,
            mission.mission_id,
            root,
            lane=LANE_REPAIR,
            priority=9,
            resources=[contested],
        )
        before = store.metrics().conflicts
        with store.dispatch(operation=_key('dispatch'), worker="worker-2") as blocked:
            assert blocked is None
        assert store.metrics().conflicts == before + 1
        assert store.get_work_item(repair.work_item_id).status == STATUS_READY
        held.close(_fixed(), operation=_key('close'))

    # Once the conflict clears, the preferred lane runs immediately.
    assert _run_one(store) == repair.work_item_id


def test_a_dispatch_claims_every_resource_or_none(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    shared, only = _resource("shared"), _resource("only")
    root = _root(store, mission.mission_id, resources=[shared])
    blocked = _side(store, mission.mission_id, root, resources=[only, shared])

    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as handle:
        assert handle.work_item_id == root.work_item_id
        assert handle.resources == (shared,)
        with store.dispatch(operation=_key('dispatch'), worker="worker-2") as loser:
            assert loser is None
        assert not store.is_claimed(only)
        assert store.get_work_item(blocked.work_item_id).status == STATUS_READY


# --------------------------------------------------------------------------
# 7. fencing tokens and lease authority
# --------------------------------------------------------------------------


def test_a_stale_fence_is_rejected_even_from_the_handle_that_holds_the_lease(tmp_path):
    """A live lease is necessary for a mutation.  It is not sufficient.

    The fence answers a different question from the lease: not "are you holding
    the lock" but "is the era you were dispatched into still the current one".
    Durable state is advanced here to stand in for the sequence a crashed
    worker, a reclaim and a re-dispatch produce, because that is precisely the
    situation in which a handle can still be holding a descriptor while the row
    it refers to has moved on.
    """

    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)

    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as handle:
        assert handle.fence >= 1
        # A consistent later era: exactly what one further dispatch of this item
        # would have left behind - a new token on the row, one more attempt, and
        # every counter that moves per dispatch moved with them.  Rewriting only
        # the row would be corruption, and would be refused as such long before
        # the fencing token ever got a chance to speak.
        _sql(store, "UPDATE work_items SET fence = fence + 1, attempts = attempts + 1")
        _sql(
            store,
            "UPDATE counters SET value = value + 1 WHERE name IN"
            " ('fence', 'dispatch_seq', 'dispatches', 'latency_samples')",
        )

        with pytest.raises(MissionStaleFence):
            handle.close(_fixed(), operation=_key('close'))
        with pytest.raises(MissionStaleFence):
            handle.heartbeat()

        assert store.metrics().stale_fence_rejects == 2
        # Nothing was written into the newer era.
        assert store.get_work_item(root.work_item_id).status == STATUS_DISPATCHED
        assert store.get_work_item(root.work_item_id).disposition is None


def test_a_later_dispatch_ends_the_previous_handles_authority(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)

    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as first:
        stale = first.fence
        # Leaving without closing requeues the item; the token stays behind.

    assert store.get_work_item(root.work_item_id).status == STATUS_READY

    with store.dispatch(operation=_key('dispatch'), worker="worker-2") as second:
        assert second.fence > stale
        # The first handle cannot reach past its own release, by construction:
        # it no longer holds a lease at all.
        with pytest.raises(MissionUnauthorized):
            first.close(_fixed(), operation=_key('close'))
        with pytest.raises(MissionUnauthorized):
            first.heartbeat()
        assert store.get_work_item(root.work_item_id).fence == second.fence
        second.close(_fixed(), operation=_key('close'))

    assert store.get_work_item(root.work_item_id).disposition == DISPOSITION_FIXED


def test_a_heartbeat_never_transfers_or_releases_authority(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id, resources=[_resource("held")])

    # There is no store-level heartbeat and no store-level close: a worker
    # mutation is reachable only from the handle that holds the lease.
    assert not hasattr(store, "heartbeat")
    assert not hasattr(store, "close_work_item")

    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as handle:
        fence = handle.fence
        assert handle.heartbeat() == 1
        assert handle.heartbeat() == 2

        item = store.get_work_item(root.work_item_id)
        assert item.status == STATUS_DISPATCHED
        assert item.fence == fence
        assert item.worker == "worker-1"
        assert item.heartbeats == 2

        # Nobody else can take the item, and the lease cannot be reclaimed
        # while it is provably held - there is no TTL to wait out.
        with store.dispatch(operation=_key('dispatch'), worker="worker-2") as other:
            assert other is None
        with pytest.raises(MissionConflict):
            store.reclaim(root.work_item_id, operation=_key('reclaim'))

        # Heartbeating changed nothing that decides ownership.
        assert store.get_work_item(root.work_item_id).fence == fence
        assert store.get_work_item(root.work_item_id).worker == "worker-1"
        assert store.is_claimed(_resource("held"))
        handle.close(_fixed(), operation=_key('close'))

    with pytest.raises(MissionUnauthorized):
        handle.heartbeat()


def test_a_lease_is_reclaimed_only_when_it_is_provably_free(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id, resources=[_resource("held")])

    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as handle:
        with pytest.raises(MissionConflict):
            store.reclaim(handle.work_item_id, operation=_key('reclaim'))

    # After the holder let go the item is already back on the queue, so there
    # is nothing left to reclaim.
    assert store.reclaim(root.work_item_id, operation=_key('reclaim')) is False
    assert store.get_work_item(root.work_item_id).status == STATUS_READY


# --------------------------------------------------------------------------
# 8. fail closed: corrupt records, symlinks, unsupported locking
# --------------------------------------------------------------------------


def test_a_database_that_is_not_a_database_fails_closed(tmp_path):
    store = MissionStore(tmp_path)
    _mission(store)
    (store.root / mission_control._DB_NAME).write_bytes(b"this is not a database")

    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        _mission(store)


def test_an_unknown_lane_or_schema_version_fails_closed(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    _root(store, mission.mission_id)

    _sql(store, "UPDATE work_items SET lane = 'sideways'")
    with pytest.raises(MissionCorrupt):
        store.snapshot()
    with pytest.raises(MissionCorrupt):
        _run_one(store)

    _sql(store, "UPDATE work_items SET lane = 'primary'")
    _sql(store, "UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    with pytest.raises(MissionCorrupt):
        store.snapshot()
    with pytest.raises(MissionCorrupt):
        _mission(store)


def test_a_symlinked_store_root_is_refused_rather_than_followed(tmp_path):
    if not _POSIX:  # pragma: no cover - no symlink semantics to speak of
        pytest.skip("POSIX only")
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    home = tmp_path / "home"
    home.mkdir()
    (home / "mission-control").symlink_to(outside, target_is_directory=True)

    store = MissionStore(home)
    with pytest.raises(MissionError):
        _mission(store)

    # The target was neither re-moded nor written into.
    assert list(outside.iterdir()) == []
    assert stat.S_IMODE(outside.stat().st_mode) == 0o755


def test_a_host_without_flock_is_refused_rather_than_served(tmp_path, monkeypatch):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    _root(store, mission.mission_id)

    monkeypatch.setattr(mission_control, "fcntl", types.SimpleNamespace())
    with pytest.raises(MissionUnsupported):
        _mission(store)
    with pytest.raises(MissionUnsupported):
        with store.dispatch(operation=_key('dispatch'), worker="worker-1"):
            pass  # pragma: no cover - the dispatch never yields

    # One rule, reads included.  "Nothing here" and "I could not look" are
    # different answers, and a host with no exclusion primitive is giving the
    # second one; reporting a queue depth from it would be a confident number
    # assembled out of an inability to read.
    with pytest.raises(MissionUnsupported):
        store.metrics()
    with pytest.raises(MissionUnsupported):
        store.snapshot()
    with pytest.raises(MissionUnsupported):
        store.get_mission(mission.mission_id)

    # And the refusal changed nothing: the store is intact once the primitive
    # is back.
    monkeypatch.undo()
    assert store.metrics().queue_depth == 1
    assert store.snapshot().missions[0].mission_id == mission.mission_id


# --------------------------------------------------------------------------
# 9. bounded, secret-free observability
# --------------------------------------------------------------------------


def test_snapshots_are_bounded_and_carry_no_free_text(tmp_path):
    marker = "do-not-log-me-4f3a"
    store = MissionStore(tmp_path)
    mission = store.create_mission(
        scope="scope-a",
        objective=f"reach the outcome {marker}",
        desired_result=f"the outcome {marker} holds",
        acceptance_criteria=[("crit-0", f"the {marker} outcome is demonstrated")],
    operation=_key('mission'))
    root = store.submit_work_item(
        mission.mission_id,
        root=True,
        coordinates=WorkCoordinates(f"project-{marker}", f"repo-{marker}", f"path-{marker}"),
        resources=[MissionResource("ns-test", "kind-test", f"res-{marker}")],
    operation=_key('submit'))
    for _ in range(3):
        _side(store, mission.mission_id, root)

    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as handle:
        handle.close(
            Closure(
                disposition=DISPOSITION_BLOCKED,
                rationale=f"blocked by {marker}",
                risk=f"the {marker} gap stays open",
                evidence_refs=(f"evidence-{marker}",),
                owner=f"owner-{marker}",
                revisit_at=int(time.time()) + 3600,
            )
        , operation=_key('close'))

    full = store.snapshot()
    assert marker not in repr(full)
    assert marker not in repr(store.metrics())
    assert full.truncated is False
    assert len(full.work_items) == 4
    assert full.missions[0].criteria_ids == ("crit-0",)
    assert full.missions[0].work_items == 4
    assert full.missions[0].closed_work_items == 1

    summary = next(item for item in full.work_items if item.work_item_id == root.work_item_id)
    assert summary.disposition == DISPOSITION_BLOCKED
    assert summary.status == STATUS_CLOSED
    assert summary.resource_count == 1
    assert summary.is_root and summary.parent_id is None

    bounded = store.snapshot(limit=2)
    assert bounded.truncated is True
    assert len(bounded.work_items) == 2

    scoped = store.snapshot(mission_id=mission.mission_id, limit=1)
    assert scoped.truncated is True
    assert len(scoped.missions) == 1

    for bad in (0, -1, MAX_SNAPSHOT_ITEMS + 1, "many"):
        with pytest.raises(MissionRejected):
            store.snapshot(limit=bad)
    assert DEFAULT_SNAPSHOT_LIMIT <= MAX_SNAPSHOT_ITEMS


def test_metrics_account_for_every_outcome(tmp_path):
    store = MissionStore(tmp_path, queue_capacity=4)
    empty = store.metrics()
    assert empty.queue_depth == 0 and empty.dispatches == 0

    mission = _mission(store)
    root = _root(store, mission.mission_id)
    _side(store, mission.mission_id, root)
    _side(store, mission.mission_id, root)
    _side(store, mission.mission_id, root)
    with pytest.raises(MissionCapacityExceeded):
        _side(store, mission.mission_id, root)

    assert store.metrics().queue_depth == 4
    _run_one(store, _fixed())
    _run_one(store, _deferred())
    _run_one(store, _blocked())

    metrics = store.metrics()
    assert metrics.queue_capacity == 4
    assert metrics.queue_depth == 1
    assert metrics.dispatched == 0
    assert (metrics.closed_fixed, metrics.closed_deferred, metrics.closed_blocked) == (1, 1, 1)
    assert metrics.dispatches == 3
    assert metrics.capacity_rejects == 1
    assert metrics.submit_to_dispatch_samples == 3
    assert metrics.submit_to_dispatch_ms_total >= 0
    assert metrics.submit_to_dispatch_ms_max >= 0
    assert metrics.submit_to_dispatch_ms_mean >= 0
    assert metrics.missions_open == 1 and metrics.missions_completed == 0

    _run_one(store)
    store.complete_mission(mission.mission_id, {"crit-0": "evidence-a"}, operation=_key('complete'))
    final = store.metrics()
    assert final.missions_completed == 1
    assert final.queue_depth == 0
    assert final.scan_truncations == 0


def test_public_surface_is_small_and_typed(tmp_path):
    exported = set(mission_control.__all__)
    assert exported <= set(dir(mission_control))
    for name in ("MissionStore", "DispatchHandle", "Closure", "MissionStaleFence"):
        assert name in exported
    # Every failure mode the caller has to branch on is a typed subclass.
    for error in (
        MissionRejected,
        MissionCapacityExceeded,
        MissionConflict,
        MissionStaleFence,
        MissionCorrupt,
        MissionUnsupported,
    ):
        assert issubclass(error, MissionError)
    assert MissionStore(tmp_path).root == Path(tmp_path).resolve() / "mission-control"
    assert LANE_PRIMARY != LANE_REPAIR
    assert STATUS_READY != STATUS_DISPATCHED != STATUS_CLOSED
    assert AcceptanceCriterion("crit-0", "text").id == "crit-0"


def test_no_public_store_method_accepts_a_fencing_token(tmp_path):
    """The shape of the API is the first line of the authority argument.

    A fence is snapshot-visible.  If any public method took one, holding a lease
    would stop being what authorises a mutation, and every other defence in this
    module would be arguing about a door that was already open.
    """

    store = MissionStore(tmp_path)
    for name in dir(store):
        if name.startswith("_"):
            continue
        member = getattr(store, name)
        if not callable(member):
            continue
        parameters = set(inspect.signature(member).parameters)
        assert "fence" not in parameters, name
        assert "lease" not in parameters, name


# --------------------------------------------------------------------------
# 10. execution authority is a live lease, not a visible fencing token
# --------------------------------------------------------------------------


def _hold_lease(root: str, release, results) -> None:
    """Dispatch in an independent process and keep the lease held."""

    store = MissionStore(root)
    with store.dispatch(operation="holder-dispatch", worker="holder") as handle:
        results.put((handle.work_item_id, handle.fence))
        release.wait(timeout=120)


def _forge_authority(root: str, work_item_id: str, fence: int, results) -> None:
    """From another process, try every route to close work we do not hold."""

    store = MissionStore(root)
    outcomes = {"public_close": hasattr(store, "close_work_item")}

    # Everything an observer can legitimately see about the live dispatch.
    item = store.get_work_item(work_item_id)
    outcomes["sees_fence"] = item.fence == fence

    # A handle assembled by hand out of exactly that visible data, over a real
    # descriptor on the real lease file.
    path = store._lease_path(work_item_id)
    descriptor = os.open(path, os.O_RDWR)
    try:
        lease = mission_control._Lease(work_item_id, path, descriptor, fence)
        handle = mission_control.DispatchHandle(store, item, lease)
        for label, call in (
            ("close", lambda: handle.close(_fixed(), operation=_key('close'))),
            ("heartbeat", handle.heartbeat),
        ):
            try:
                call()
                outcomes[label] = "accepted"
            except MissionUnauthorized:
                outcomes[label] = "rejected"
            except MissionError as exc:  # pragma: no cover - any other refusal
                outcomes[label] = f"error:{type(exc).__name__}"
    finally:
        os.close(descriptor)

    try:
        store.reclaim(work_item_id, operation=_key('reclaim'))
        outcomes["reclaim"] = "accepted"
    except MissionConflict:
        outcomes["reclaim"] = "rejected"
    results.put(outcomes)


def test_another_process_holding_the_fence_cannot_close_live_work(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id, resources=[_resource("held")])

    context = multiprocessing.get_context("spawn")
    release = context.Event()
    results = context.Queue()
    holder = context.Process(target=_hold_lease, args=(str(tmp_path), release, results))
    started = [holder]
    holder.start()
    try:
        work_item_id, fence = results.get(timeout=60)
        assert work_item_id == root.work_item_id

        forger = context.Process(
            target=_forge_authority, args=(str(tmp_path), work_item_id, fence, results)
        )
        started.append(forger)
        forger.start()
        outcomes = results.get(timeout=60)
        forger.join(timeout=60)
        assert forger.exitcode == 0

        assert outcomes == {
            "public_close": False,
            "sees_fence": True,
            "close": "rejected",
            "heartbeat": "rejected",
            "reclaim": "rejected",
        }
        # The holder is untouched: same era, same worker, still holding.
        live = store.get_work_item(work_item_id)
        assert live.status == STATUS_DISPATCHED
        assert live.fence == fence
        assert live.worker == "holder"
        assert live.disposition is None
        assert store.is_claimed(_resource("held"))
        assert store.metrics().lease_rejects >= 2
    finally:
        try:
            release.set()
            holder.join(timeout=60)
        finally:
            _reap(started, results)
    assert holder.exitcode == 0


def test_a_replaced_lease_file_revokes_authority(tmp_path):
    """Holding a lock on an orphaned inode is not holding the lease."""

    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)

    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as handle:
        path = store._lease_path(handle.work_item_id)
        original = path.stat().st_ino
        path.unlink()
        os.close(os.open(path, os.O_CREAT | os.O_RDWR, 0o600))
        assert path.stat().st_ino != original

        with pytest.raises(MissionUnauthorized):
            handle.close(_fixed(), operation=_key('close'))
        with pytest.raises(MissionUnauthorized):
            handle.heartbeat()

    # Releasing a revoked lease must not requeue somebody else's work either,
    # so the item is left exactly as it was for an explicit host repair.
    assert store.get_work_item(root.work_item_id).status == STATUS_DISPATCHED
    assert store.metrics().lease_rejects >= 2
    assert store.reclaim(root.work_item_id, operation=_key('reclaim')) is True
    assert store.get_work_item(root.work_item_id).status == STATUS_READY


def test_a_closed_lease_descriptor_revokes_authority(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    _root(store, mission.mission_id)

    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as handle:
        os.close(handle._lease.fd)
        # Re-point the recorded descriptor at a different open file, which is
        # exactly what a reissued descriptor number looks like.
        replacement = os.open(store.root / "store.id", os.O_RDONLY)
        try:
            os.dup2(replacement, handle._lease.fd)
            with pytest.raises(MissionUnauthorized):
                handle.close(_fixed(), operation=_key('close'))
        finally:
            os.close(replacement)


# --------------------------------------------------------------------------
# 11. the configured root, and the identity of what the store paths name
# --------------------------------------------------------------------------


def test_a_symlinked_configured_root_is_refused_rather_than_followed(tmp_path):
    if not _POSIX:  # pragma: no cover - no symlink semantics to speak of
        pytest.skip("POSIX only")
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)

    store = MissionStore(alias)
    # The configured root is kept lexically, so the alias is still visible as
    # the alias rather than having been resolved away before validation.
    assert store.configured_root == Path(alias)
    with pytest.raises(MissionError):
        _mission(store)

    assert list(outside.iterdir()) == []
    assert not (outside / "mission-control").exists()
    assert stat.S_IMODE(outside.stat().st_mode) == 0o755


def test_a_symlinked_database_is_refused(tmp_path):
    if not _POSIX:  # pragma: no cover - no symlink semantics to speak of
        pytest.skip("POSIX only")
    store = MissionStore(tmp_path)
    _mission(store)
    database = store.root / mission_control._DB_NAME
    elsewhere = tmp_path / "elsewhere.db"

    database.unlink()
    database.symlink_to(elsewhere)
    with pytest.raises(MissionError):
        _mission(store)
    with pytest.raises(MissionCorrupt):
        store.metrics()
    assert not elsewhere.exists()


def _decoy_database(tmp_path) -> bytes:
    """A perfectly valid mission database that belongs to a different store."""

    other = MissionStore(tmp_path / "other")
    other.create_mission(
        scope="scope-decoy",
        objective="DECOY",
        desired_result="DECOY",
        acceptance_criteria=[("crit-0", "DECOY")],
    operation=_key('mission'))
    return (other.root / mission_control._DB_NAME).read_bytes()


def test_a_database_swapped_back_during_a_read_is_never_the_one_read(tmp_path, monkeypatch):
    """The exact replace-open-restore sequence, defeated by construction.

    An identity check before and after the open cannot see this attack: the
    pathname holds the right inode at both moments and the decoy only exists in
    between.  What defeats it is not a better check - it is never handing a
    pathname to SQLite at all.  The bytes come from a descriptor this kernel
    opened with ``O_NOFOLLOW`` relative to a directory it walked without
    following a link, so re-pointing the name afterwards has nothing left to
    affect.
    """

    store = MissionStore(tmp_path)
    store.create_mission(
        scope="scope-a",
        objective="ORIGINAL",
        desired_result="the original desired result",
        acceptance_criteria=[("crit-0", "the original statement")],
    operation=_key('mission'))
    decoy_bytes = _decoy_database(tmp_path)
    database = store.root / mission_control._DB_NAME
    aside = store.root / "aside.db"
    decoy = store.root / "decoy.db"
    decoy.write_bytes(decoy_bytes)

    real_read = mission_control._read_all
    real_materialise = MissionStore._materialise
    state = {"swapped": False, "restored": False}

    def swapping(descriptor, limit):
        # The descriptor is already open on the real database.  Install the
        # decoy at the pathname and leave it there for the whole load.
        if not state["swapped"]:
            state["swapped"] = True
            os.rename(database, aside)
            os.rename(decoy, database)
        return real_read(descriptor, limit)

    def restoring(*args, **kwargs):
        # Withdraw the decoy before the load finishes, so an identity check made
        # either side of the window sees the right inode both times.
        if state["swapped"] and not state["restored"]:
            state["restored"] = True
            os.rename(database, decoy)
            os.rename(aside, database)
        return real_materialise(*args[1:], **kwargs)

    monkeypatch.setattr(mission_control, "_read_all", swapping)
    monkeypatch.setattr(MissionStore, "_materialise", restoring)
    mission = store.get_mission("m-000000000001")
    monkeypatch.undo()

    assert state["swapped"] and state["restored"]
    assert mission.objective == "ORIGINAL"
    # And the store is untouched afterwards.
    assert store.get_mission("m-000000000001").objective == "ORIGINAL"
    assert database.read_bytes() != decoy_bytes


def test_a_foreign_database_left_at_the_pathname_is_refused(tmp_path):
    """Being a valid database is not the same as being *this* store's database."""

    store = MissionStore(tmp_path)
    _mission(store)
    database = store.root / mission_control._DB_NAME
    original = database.read_bytes()

    database.write_bytes(_decoy_database(tmp_path))
    with pytest.raises(MissionCorrupt):
        store.get_mission("m-000000000001")
    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        _mission(store)

    # The sidecar identity is what refuses it, and it is not repaired in
    # passing: restoring the real database restores the store.
    database.write_bytes(original)
    assert store.get_mission("m-000000000001").objective.startswith("reach the")


def test_a_missing_store_identity_is_not_reinvented(tmp_path):
    store = MissionStore(tmp_path)
    _mission(store)
    (store.root / "store.id").unlink()

    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        _mission(store)
    assert not (store.root / "store.id").exists()


def _displace_store(store: MissionStore, tmp_path, monkeypatch, seam: str):
    """Rename the bound store directory away, leaving an empty one at the name."""

    moved = tmp_path / "moved"
    original = getattr(MissionStore, seam)
    state = {"displaced": False}

    def displacing(self, *args, **kwargs):
        if not state["displaced"]:
            state["displaced"] = True
            if seam == "_publish":
                # Publish first, so the displacement is only discoverable
                # *after* the bytes have been renamed into place.
                original(self, *args, **kwargs)
            os.rename(store.root, moved)
            os.mkdir(store.root, mode=0o700)
            if seam == "_publish":
                return None
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MissionStore, seam, displacing)
    return moved, state


def test_a_displaced_store_directory_refuses_before_publishing(tmp_path, monkeypatch):
    """Descriptor binding keeps a transaction coherent.  It cannot make it correct.

    Renaming the bound directory away and leaving an empty one at the configured
    name would otherwise let this call commit happily into a directory the
    caller's path no longer reaches, report success, and leave the next call to
    bootstrap a second store at that path - two universes, each convinced it is
    the store, neither aware of the other.  So the configured path is walked
    again and compared against the bound identities before anything is
    published, and the call is refused rather than blessed.
    """

    store = MissionStore(tmp_path)
    first = _mission(store)
    before = (store.root / mission_control._DB_NAME).read_bytes()
    moved, state = _displace_store(store, tmp_path, monkeypatch, "_load_database")

    with pytest.raises(MissionDisplaced):
        _mission(store)
    monkeypatch.undo()
    assert state["displaced"]

    # Nothing was published into the bound directory: refused, not committed.
    assert (moved / mission_control._DB_NAME).read_bytes() == before
    # And this one call forked no second store into the impostor directory.
    assert list(store.root.iterdir()) == []

    relocated = MissionStore(tmp_path / "relocated")
    relocated.configured_root.mkdir()
    os.rmdir(store.root)
    os.rename(moved, relocated.root)
    assert relocated.get_mission(first.mission_id).mission_id == first.mission_id
    assert relocated.metrics().missions_open == 1


def test_a_displacement_discovered_after_publication_is_indeterminate(tmp_path, monkeypatch):
    """Once the rename has happened there is no honest way to say yes *or* no."""

    store = MissionStore(tmp_path)
    _mission(store)
    moved, state = _displace_store(store, tmp_path, monkeypatch, "_publish")

    with pytest.raises(MissionIndeterminate):
        _mission(store)
    monkeypatch.undo()
    assert state["displaced"]

    # The bytes did land in the directory this transaction was bound to, which
    # is precisely why the caller must not be told the call failed either.
    connection = sqlite3.connect(str(moved / mission_control._DB_NAME))
    try:
        recorded = [
            row[0]
            for row in connection.execute("SELECT mission_id FROM missions ORDER BY mission_id")
        ]
    finally:
        connection.close()
    assert recorded == ["m-000000000001", "m-000000000002"]


# --------------------------------------------------------------------------
# 12. only "somebody else holds it" is a held lease
# --------------------------------------------------------------------------


def _flock_raising(number: int):
    """Fail only the non-blocking lease probe, leaving the store lock alone."""

    real = fcntl.flock
    probe = fcntl.LOCK_EX | fcntl.LOCK_NB

    def flocking(descriptor, operation):
        if operation == probe:
            raise OSError(number, os.strerror(number))
        return real(descriptor, operation)

    return flocking


def test_a_would_block_answer_is_the_only_held_lease(tmp_path, monkeypatch):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)

    monkeypatch.setattr(mission_control.fcntl, "flock", _flock_raising(errno.EAGAIN))
    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as handle:
        assert handle is None
    assert store.metrics().conflicts == 1
    assert store.get_work_item(root.work_item_id).status == STATUS_READY


@pytest.mark.parametrize(
    "number",
    [
        errno.EBADF,
        errno.EIO,
        getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
        errno.EACCES,
        errno.EPERM,
        errno.EINVAL,
    ],
)
def test_every_other_flock_failure_fails_closed(tmp_path, monkeypatch, number):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)

    monkeypatch.setattr(mission_control.fcntl, "flock", _flock_raising(number))
    with pytest.raises(MissionHostFailure):
        with store.dispatch(operation=_key('dispatch'), worker="worker-1"):
            pass  # pragma: no cover - the dispatch never yields

    monkeypatch.undo()
    # Nothing was dispatched and nothing was claimed on the way out.
    assert store.get_work_item(root.work_item_id).status == STATUS_READY
    assert store.metrics().dispatches == 0


# --------------------------------------------------------------------------
# 13. a full window of conflicts cannot hide a runnable item forever
# --------------------------------------------------------------------------


def test_the_bounded_scan_eventually_reaches_a_candidate_below_the_window(tmp_path):
    """Ranking is bounded; reachability is not allowed to be.

    The preferred order is a ranking, so a window full of mutually conflicting
    candidates would be re-examined in the same order on every call and an item
    ranked below them would never be looked at again.  The durable cursor is
    what turns "bounded per call" into "everything eventually", and it has to
    survive a restart to mean anything.
    """

    store = MissionStore(tmp_path)
    mission = _mission(store)
    contested = _resource("contested")
    holder = _root(store, mission.mission_id, resources=[contested])
    for _ in range(MAX_DISPATCH_CANDIDATES):
        _side(store, mission.mission_id, holder, resources=[contested])
    # Submitted last, so it ranks below every conflicting candidate.
    reachable = _side(store, mission.mission_id, holder)

    with store.dispatch(operation="holder-dispatch", worker="holder") as held:
        assert held.work_item_id == holder.work_item_id

        # One window's worth of candidates, all blocked by the held resource.
        with store.dispatch(operation=_key('dispatch'), worker="worker-1") as blocked:
            assert blocked is None
        after_first = store.metrics()
        assert after_first.scan_truncations == 1
        assert after_first.cursor_sweeps == 1
        assert after_first.conflicts == 2 * MAX_DISPATCH_CANDIDATES

        # The cursor advanced, and it is durable: a brand new store object over
        # the same directory picks the sweep up where the last one stopped.
        with MissionStore(tmp_path).dispatch(operation=_key('dispatch'), worker="worker-2") as reached:
            assert reached is not None
            assert reached.work_item_id == reachable.work_item_id
            reached.close(_fixed(), operation=_key('close'))

    assert store.metrics().cursor_sweeps == 2


# --------------------------------------------------------------------------
# 14. dependencies are a real scheduling graph, not lineage under another name
# --------------------------------------------------------------------------


def test_dependencies_gate_readiness_and_outrank_priority(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)
    groundwork = _side(store, mission.mission_id, root, priority=0)
    urgent = _side(
        store,
        mission.mission_id,
        root,
        priority=MAX_PRIORITY,
        depends_on_ids=[groundwork.work_item_id],
    )
    assert urgent.depends_on_ids == (groundwork.work_item_id,)
    # Lineage and dependency are different edges and are recorded separately.
    assert urgent.parent_id == root.work_item_id
    assert root.depends_on_ids == ()

    # The urgent item outranks everything on priority and is still not offered.
    assert _run_one(store) == root.work_item_id
    assert _run_one(store) == groundwork.work_item_id
    assert _run_one(store) == urgent.work_item_id


def test_a_dependency_that_did_not_deliver_keeps_its_dependents_off_the_queue(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)
    groundwork = _side(store, mission.mission_id, root)
    dependent = _side(
        store, mission.mission_id, root, depends_on_ids=[groundwork.work_item_id]
    )

    assert _run_one(store) == root.work_item_id
    assert _run_one(store, _deferred()) == groundwork.work_item_id

    # Deferred is an honest outcome, but it did not deliver, so nothing that
    # was waiting on it becomes runnable.
    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as nothing:
        assert nothing is None
    assert store.get_work_item(dependent.work_item_id).status == STATUS_READY

    # It cannot be quietly dropped, and it cannot be called fixed either.
    with pytest.raises(MissionRejected):
        store.abandon_unrunnable_work_item(dependent.work_item_id, _fixed(), operation=_key('abandon'))
    # Nor is this a general-purpose closure with the lease filed off: work that
    # already ran is refused, and so is work that is merely waiting its turn.
    with pytest.raises(MissionConflict):
        store.abandon_unrunnable_work_item(root.work_item_id, _blocked(), operation=_key('abandon'))
    idle = _mission(store, scope="scope-idle")
    idle_root = _root(store, idle.mission_id)
    with pytest.raises(MissionRejected):
        store.abandon_unrunnable_work_item(idle_root.work_item_id, _blocked(), operation=_key('abandon'))

    closed = store.abandon_unrunnable_work_item(dependent.work_item_id, _blocked(), operation=_key('abandon'))
    assert closed.disposition == DISPOSITION_BLOCKED
    assert closed.closure.owner == "owner-b"
    assert store.metrics().abandoned == 1
    assert store.metrics().closed_blocked == 1


def test_a_fixed_closure_is_refused_when_a_dependency_did_not_deliver(tmp_path):
    """Enforced at the closure, not only at the gate.

    Dispatch already refuses to offer work whose dependencies did not deliver,
    so reaching this rule means the durable record changed underneath a running
    worker.  It is still checked, because "my dependency is blocked but I fixed
    it anyway" is exactly the claim this kernel must not be able to record.
    """

    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)
    groundwork = _side(store, mission.mission_id, root)
    dependent = _side(
        store, mission.mission_id, root, depends_on_ids=[groundwork.work_item_id]
    )

    _run_one(store)  # root
    _run_one(store)  # groundwork, fixed, so the dependent becomes runnable

    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as handle:
        assert handle.work_item_id == dependent.work_item_id
        # The dependency's outcome is rewritten mid-flight, and rewritten
        # *consistently* - row and durable counters together - so that what the
        # close path meets is a coherent store in which the dependency simply
        # did not deliver, not a store that is merely damaged.
        _sql(
            store,
            "UPDATE work_items SET disposition = ?, closure = ? WHERE work_item_id = ?",
            (
                DISPOSITION_DEFERRED,
                json.dumps(
                    _deferred().as_payload(), sort_keys=True, separators=(",", ":")
                ),
                groundwork.work_item_id,
            ),
        )
        _sql(store, "UPDATE counters SET value = value - 1 WHERE name = 'closed_fixed'")
        _sql(store, "UPDATE counters SET value = value + 1 WHERE name = 'closed_deferred'")
        with pytest.raises(MissionRejected):
            handle.close(_fixed(), operation=_key('close'))
        # The honest outcomes remain available.
        handle.close(_blocked(), operation=_key('close'))

    assert store.get_work_item(dependent.work_item_id).disposition == DISPOSITION_BLOCKED


def test_dependency_declarations_are_validated(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    other = _mission(store, scope="scope-other")
    root = _root(store, mission.mission_id)
    foreign_root = _root(store, other.mission_id)

    for bad in (
        ["w-000000000999"],  # names nothing
        [foreign_root.work_item_id],  # another mission's work
        ["not-an-id"],  # not an identifier at all
        [root.work_item_id] * (MAX_DEPENDENCIES + 1),  # over the bound
        "w-000000000001",  # not a sequence
    ):
        with pytest.raises(MissionRejected):
            _side(store, mission.mission_id, root, depends_on_ids=bad)

    # Duplicates are canonicalised rather than stored twice.
    duplicated = _side(
        store, mission.mission_id, root, depends_on_ids=[root.work_item_id] * 3
    )
    assert duplicated.depends_on_ids == (root.work_item_id,)


@pytest.mark.parametrize("self_edge", [True, False])
def test_an_injected_dependency_cycle_is_refused(tmp_path, self_edge):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)
    first = _side(store, mission.mission_id, root)
    second = _side(store, mission.mission_id, root, depends_on_ids=[first.work_item_id])

    _sql(
        store,
        "INSERT INTO dependencies (work_item_id, depends_on_id) VALUES (?, ?)",
        (
            (first.work_item_id, first.work_item_id)
            if self_edge
            else (first.work_item_id, second.work_item_id)
        ),
    )

    with pytest.raises(MissionCorrupt):
        _side(store, mission.mission_id, root)


# --------------------------------------------------------------------------
# 15. the main axis has to actually be fixed
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "closure_factory,disposition",
    [(_deferred, DISPOSITION_DEFERRED), (_blocked, DISPOSITION_BLOCKED)],
)
def test_a_mission_does_not_complete_on_a_root_that_was_not_fixed(
    tmp_path, closure_factory, disposition
):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)
    _side(store, mission.mission_id, root)

    assert _run_one(store, closure_factory()) == root.work_item_id
    _run_one(store, _fixed())

    assert store.get_work_item(root.work_item_id).disposition == disposition
    with pytest.raises(MissionRejected) as excinfo:
        store.complete_mission(mission.mission_id, {"crit-0": "evidence-a"}, operation=_key('complete'))
    assert disposition in str(excinfo.value)
    assert store.get_mission(mission.mission_id).status == MISSION_OPEN
    assert store.metrics().missions_completed == 0


def test_side_issues_may_close_any_accounted_way_once_the_root_is_fixed(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)
    _side(store, mission.mission_id, root)
    _side(store, mission.mission_id, root)

    assert _run_one(store, _fixed()) == root.work_item_id
    _run_one(store, _deferred())
    _run_one(store, _blocked())

    completed = store.complete_mission(mission.mission_id, {"crit-0": "evidence-a"}, operation=_key('complete'))
    assert completed.status == MISSION_COMPLETED
    assert store.metrics().missions_completed == 1
    assert store.metrics().missions_open == 0


# --------------------------------------------------------------------------
# 16. repair work cannot be replenished into a monopoly
# --------------------------------------------------------------------------


def test_continuous_repair_replenishment_cannot_starve_a_quiet_scope(tmp_path):
    """Lane preference is a preference *within* a scope, never above fairness.

    A tenant that can always produce more repair work would otherwise hold every
    other tenant's primary work off the queue for as long as it kept producing -
    starvation with an urgent-sounding name.
    """

    store = MissionStore(tmp_path)
    loud = _mission(store, scope="scope-loud")
    quiet = _mission(store, scope="scope-quiet")
    loud_root = _root(store, loud.mission_id)
    _side(store, loud.mission_id, loud_root, lane=LANE_REPAIR, priority=MAX_PRIORITY)
    quiet_root = _root(store, quiet.mission_id)

    served = []
    for _ in range(4):
        with store.dispatch(operation=_key('dispatch'), worker="worker-1") as handle:
            assert handle is not None
            served.append((handle.scope, handle.lane))
            handle.close(_fixed(), operation=_key('close'))
        # The loud scope replenishes its repair queue after every dispatch.
        _side(
            store, loud.mission_id, loud_root, lane=LANE_REPAIR, priority=MAX_PRIORITY
        )

    # Repair still wins the first pick, because no scope had been served yet.
    assert served[0] == ("scope-loud", LANE_REPAIR)
    # And then fairness takes over: the quiet scope's ordinary primary work is
    # served second, not after an unbounded run of replenished repair items.
    assert served[1] == ("scope-quiet", LANE_PRIMARY)
    assert store.get_work_item(quiet_root.work_item_id).status == STATUS_CLOSED
    assert [scope for scope, _ in served].count("scope-quiet") == 1


# --------------------------------------------------------------------------
# 17. every durable field is validated on the way back out
# --------------------------------------------------------------------------

_DIGEST = "a" * 64
_MISSION = "m-000000000001"
_ROOT = "w-000000000001"
_SIDE = "w-000000000002"

#: Each entry damages exactly one durable field in a way that some *inbound*
#: check would have refused.  None of them is a Python-level type error, so
#: none of them would be caught by anything except deliberate validation.
_CORRUPTIONS = [
    ("counter removed", "DELETE FROM counters WHERE name = 'conflicts'", ()),
    ("counter negative", "UPDATE counters SET value = -1 WHERE name = 'dispatches'", ()),
    ("counter unknown", "INSERT INTO counters (name, value) VALUES ('mystery', 0)", ()),
    ("counter not an integer", "UPDATE counters SET value = 'lots' WHERE name = 'fence'", ()),
    ("counter out of bounds", "UPDATE counters SET value = ? WHERE name = 'fence'", (2**60,)),
    ("capacity zero", "UPDATE meta SET value = '0' WHERE key = 'queue_capacity'", ()),
    ("capacity malformed", "UPDATE meta SET value = 'plenty' WHERE key = 'queue_capacity'", ()),
    ("capacity missing", "DELETE FROM meta WHERE key = 'queue_capacity'", ()),
    ("schema version unknown", "UPDATE meta SET value = '99' WHERE key = 'schema_version'", ()),
    (
        "claim outlives its dispatch",
        "INSERT INTO claims (resource, work_item_id) VALUES (?, ?)",
        (_DIGEST, _SIDE),
    ),
    (
        "claim is not a digest",
        "INSERT INTO claims (resource, work_item_id) VALUES ('zz', ?)",
        (_SIDE,),
    ),
    (
        "claim owner is malformed",
        "INSERT INTO claims (resource, work_item_id) VALUES (?, 'nope')",
        (_DIGEST,),
    ),
    ("rotation sequence is zero", "UPDATE rotation SET last_dispatch_seq = 0", ()),
    ("rotation scope is malformed", "UPDATE rotation SET scope = 'has space'", ()),
    ("timestamp is not finite", "UPDATE work_items SET submitted_at = 9e999", ()),
    ("timestamp is negative", "UPDATE work_items SET submitted_at = -1", ()),
    ("timestamp is not a number", "UPDATE work_items SET submitted_at = 'noon'", ()),
    ("priority is out of bounds", "UPDATE work_items SET priority = 100000", ()),
    ("root flag is not boolean", "UPDATE work_items SET is_root = 2 WHERE is_root = 1", ()),
    (
        "identifier is malformed",
        "UPDATE work_items SET mission_id = 'nope' WHERE work_item_id = ?",
        (_SIDE,),
    ),
    (
        "lineage disagrees with the root flag",
        "UPDATE work_items SET parent_id = NULL WHERE work_item_id = ?",
        (_SIDE,),
    ),
    (
        "disposition disagrees with the closure",
        "UPDATE work_items SET disposition = 'deferred' WHERE work_item_id = ?",
        (_ROOT,),
    ),
    (
        "closed work has no closure",
        "UPDATE work_items SET closure = NULL WHERE work_item_id = ?",
        (_ROOT,),
    ),
    (
        "a fencing token without a dispatch",
        "UPDATE work_items SET fence = 7 WHERE work_item_id = ?",
        (_SIDE,),
    ),
    (
        "a worker without a dispatch",
        "UPDATE work_items SET worker = 'ghost' WHERE work_item_id = ?",
        (_SIDE,),
    ),
    (
        "resources are not canonically ordered",
        "UPDATE work_items SET resources = ? WHERE work_item_id = ?",
        ('[["ns-test","kind-test","res-b"],["ns-test","kind-test","res-a"]]', _SIDE),
    ),
    (
        "acceptance criteria repeat an id",
        "UPDATE missions SET criteria = ?",
        ('[["crit-0","one"],["crit-0","two"]]',),
    ),
    ("an open mission has a completion time", "UPDATE missions SET completed_at = 1.0", ()),
    ("an open mission has acceptance evidence", "UPDATE missions SET evidence = '{}'", ()),
    (
        "a dependency that does not exist",
        "INSERT INTO dependencies (work_item_id, depends_on_id) VALUES (?, 'w-000000000999')",
        (_SIDE,),
    ),
]


def _populated(tmp_path) -> MissionStore:
    """A store with one closed root, one ready side issue and one dispatch."""

    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)
    _side(store, mission.mission_id, root)
    assert _run_one(store, _fixed()) == root.work_item_id
    assert (mission.mission_id, root.work_item_id) == (_MISSION, _ROOT)
    return store


def _read_everything(store: MissionStore) -> None:
    store.metrics()
    store.snapshot()
    store.get_mission(_MISSION)
    store.get_work_item(_ROOT)
    store.get_work_item(_SIDE)


def test_the_populated_store_reads_cleanly_before_it_is_damaged(tmp_path):
    """The battery below is only evidence if the undamaged store passes it."""

    _read_everything(_populated(tmp_path))


@pytest.mark.parametrize(
    "statement,params",
    [(statement, params) for _, statement, params in _CORRUPTIONS],
    ids=[label for label, _, _ in _CORRUPTIONS],
)
def test_damaged_durable_state_fails_closed(tmp_path, statement, params):
    store = _populated(tmp_path)
    _sql(store, statement, params)
    with pytest.raises(MissionCorrupt):
        _read_everything(store)


def _query(store: MissionStore, statement: str, params=()) -> list:
    connection = sqlite3.connect(str(store.root / mission_control._DB_NAME))
    try:
        return connection.execute(statement, params).fetchall()
    finally:
        connection.close()


# --------------------------------------------------------------------------
# 18. an existing store is validated, never quietly migrated back to life
# --------------------------------------------------------------------------


def test_a_lost_fence_counter_is_never_resurrected_at_zero(tmp_path):
    """Recreating a missing counter would reissue a fencing token already spent.

    ``INSERT OR IGNORE`` on every mutation looks like harmless idempotence and
    is the opposite: a store that lost its ``fence`` row would have it recreated
    at zero and would then hand out fence ``1`` to a second worker while the
    first is still running under fence ``1``.  Bootstrap belongs to an empty
    store; for an existing one, missing state is damage.
    """

    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)
    assert _run_one(store) == root.work_item_id
    side = _side(store, mission.mission_id, root)
    assert store.get_work_item(root.work_item_id).fence == 1
    assert store.metrics().dispatches == 1

    _sql(store, "DELETE FROM counters WHERE name = 'fence'")

    with pytest.raises(MissionCorrupt):
        with store.dispatch(operation=_key('dispatch'), worker="worker-1"):
            pass  # pragma: no cover - the dispatch never yields
    with pytest.raises(MissionCorrupt):
        _side(store, mission.mission_id, root)
    with pytest.raises(MissionCorrupt):
        store.metrics()

    # Nothing was dispatched, nothing was claimed, and above all nothing was
    # recreated: the counter is still gone.
    assert _query(store, "SELECT name FROM counters WHERE name = 'fence'") == []
    assert _query(store, "SELECT COUNT(*) FROM claims")[0][0] == 0
    assert _query(
        store, "SELECT status, fence FROM work_items WHERE work_item_id = ?",
        (side.work_item_id,),
    ) == [(STATUS_READY, 0)]


@pytest.mark.parametrize(
    "statement,vanished",
    [
        ("DROP TABLE claims", "claims"),
        ("DROP TABLE dependencies", "dependencies"),
        ("DROP INDEX work_items_ready", "work_items_ready"),
        ("DROP INDEX work_items_one_root", "work_items_one_root"),
        ("DROP TABLE rotation", "rotation"),
    ],
)
def test_a_partial_schema_is_corruption_not_a_migration(tmp_path, statement, vanished):
    store = _populated(tmp_path)
    _sql(store, statement)

    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        _mission(store)

    assert _query(
        store, "SELECT name FROM sqlite_master WHERE name = ?", (vanished,)
    ) == []


@pytest.mark.parametrize("key", ["schema_version", "queue_capacity", "store_id"])
def test_a_missing_durable_setting_is_corruption(tmp_path, key):
    store = _populated(tmp_path)
    _sql(store, "DELETE FROM meta WHERE key = ?", (key,))

    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        _mission(store)
    assert _query(store, "SELECT key FROM meta WHERE key = ?", (key,)) == []


# --------------------------------------------------------------------------
# 19. a lease is authority only once the dispatch that took it is durable
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seam", ["_publish", "_record_latency"])
def test_a_dispatch_that_fails_before_it_is_durable_leaves_no_ghost_lease(
    tmp_path, monkeypatch, seam
):
    """A registered lease with no committed dispatch is occupancy with nothing behind it.

    The durable record would say ``ready`` - so any worker may take the item -
    while this process still holds the descriptor and the registry entry that
    make it look taken, for as long as the process lives.  Both seams here fail
    after the lease has been locked: one inside the transaction, one at the
    moment the transaction would have become durable.
    """

    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id, resources=[_resource("held")])

    def failing(*args, **kwargs):
        raise MissionCorrupt("injected failure before the dispatch is durable")

    monkeypatch.setattr(MissionStore, seam, failing)
    with pytest.raises(MissionCorrupt):
        with store.dispatch(operation=_key('dispatch'), worker="worker-1"):
            pass  # pragma: no cover - the dispatch never yields
    monkeypatch.undo()

    # No ghost: no registry entry, no claim, no dispatch, still ready.
    assert store._leases == {}
    assert store.get_work_item(root.work_item_id).status == STATUS_READY
    assert store.get_work_item(root.work_item_id).fence == 0
    assert not store.is_claimed(_resource("held"))
    assert store.metrics().dispatches == 0

    # And the lease file is genuinely unlocked, so the item dispatches normally.
    with store.dispatch(operation=_key('dispatch'), worker="worker-2") as handle:
        assert handle is not None
        assert handle.work_item_id == root.work_item_id
        assert handle.fence == 1
        handle.close(_fixed(), operation=_key('close'))
    assert store._leases == {}


def test_a_released_handle_leaves_no_registry_entry(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    _root(store, mission.mission_id)

    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as handle:
        assert list(store._leases) == [handle.work_item_id]
    assert store._leases == {}


# --------------------------------------------------------------------------
# 20. no symbolic link is followed on the way to the configured root
# --------------------------------------------------------------------------


def test_a_symbolic_link_in_an_ancestor_of_the_configured_root_is_refused(tmp_path):
    """Checking only the final component leaves the whole path above it unchecked.

    ``mkdir(parents=True)`` resolves an ancestor link silently, so a configured
    root of ``alias/nested`` would have been created, chmodded and written at
    the far end of ``alias`` with nothing ever refused.
    """

    if not _POSIX:  # pragma: no cover - no symlink semantics to speak of
        pytest.skip("POSIX only")
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o755)
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)

    store = MissionStore(alias / "nested")
    with pytest.raises(MissionError):
        _mission(store)
    # Reads refuse too, rather than reporting an empty store that is really a
    # store somewhere this kernel declined to look.
    with pytest.raises(MissionError):
        store.metrics()

    assert list(outside.iterdir()) == []
    assert not (outside / "nested").exists()
    assert stat.S_IMODE(outside.stat().st_mode) == 0o755


def test_a_host_that_cannot_bind_a_database_to_bytes_is_refused(tmp_path, monkeypatch):
    """The storage boundary rests on this primitive, so its absence is refused.

    Falling back to handing SQLite the pathname would reintroduce exactly the
    swap-back window this design exists to close, while still carrying the
    promises that only the descriptor-bound version can keep.
    """

    store = MissionStore(tmp_path)
    _mission(store)
    monkeypatch.setattr(mission_control, "_database_binding_supported", lambda: False)

    with pytest.raises(MissionUnsupported):
        _mission(store)
    with pytest.raises(MissionUnsupported):
        store.metrics()


def test_a_deep_configured_root_without_links_is_created_privately(tmp_path):
    """The walk still has to be able to build an ordinary nested root."""

    store = MissionStore(tmp_path / "a" / "b" / "c")
    mission = _mission(store)
    assert store.get_mission(mission.mission_id).mission_id == mission.mission_id
    assert (tmp_path / "a" / "b" / "c" / "mission-control").is_dir()
    if _POSIX:
        assert stat.S_IMODE(store.root.stat().st_mode) == DIRECTORY_MODE


# --------------------------------------------------------------------------
# 21. bootstrap is all-or-nothing, and its residue is never reinvented
# --------------------------------------------------------------------------


def test_a_failed_bootstrap_publication_leaves_nothing_behind(tmp_path, monkeypatch):
    """A bootstrap that could not publish must not leave half a store.

    The identity is written beside the database, so an attempt that dies between
    the two leaves a directory that looks like a store to a human and like an
    empty directory to a naive check.  Treating that as new would mint a second
    identity over the first - the same store forked in place.
    """

    store = MissionStore(tmp_path)

    def failing(*args, **kwargs):
        raise MissionHostFailure("injected publication failure")

    monkeypatch.setattr(MissionStore, "_publish", failing)
    with pytest.raises(MissionHostFailure):
        _mission(store)
    monkeypatch.undo()

    # This attempt removed only what this attempt created.
    assert not (store.root / "store.id").exists()
    assert not (store.root / mission_control._DB_NAME).exists()

    # A genuinely absent store may still bootstrap, exactly once.
    assert _mission(store).mission_id == "m-000000000001"
    assert (store.root / "store.id").exists()


def test_a_bootstrap_residue_left_by_a_crash_is_not_reinvented(tmp_path):
    """The filesystem state a crash in that window leaves, and what it must mean."""

    store = MissionStore(tmp_path)
    _mission(store)
    identity = (store.root / "store.id").read_bytes()
    # Exactly what a crash between the identity write and the publication
    # leaves behind: the sidecar, and no database.
    (store.root / mission_control._DB_NAME).unlink()

    with pytest.raises(MissionCorrupt):
        _mission(store)
    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        store.snapshot()

    # Still the original identity: not reinvented, not overwritten, and no
    # second store quietly created on top of it.
    assert (store.root / "store.id").read_bytes() == identity
    assert not (store.root / mission_control._DB_NAME).exists()


# --------------------------------------------------------------------------
# 22. exclusion cannot be bypassed by replacing a lock pathname
# --------------------------------------------------------------------------


def _replace_lock_pathname(root: str) -> None:
    """The classic bypass: unlink the lock pathname and put a fresh inode there.

    Done while another process is *inside* its critical section, which is the
    only moment at which it proves anything.
    """

    lock = Path(root) / "mission-control" / "missions.lock"
    try:
        os.unlink(lock)
    except FileNotFoundError:
        pass
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(descriptor)


def _hold_transaction(root: str, entered, release, results) -> None:
    """Sit inside a real store transaction until told to leave."""

    store = MissionStore(root)
    try:
        with store._mutate():
            entered.set()
            release.wait(timeout=60)
    except BaseException as exc:
        results.put({"who": "holder", "made": None, "error": f"{type(exc).__name__}: {exc}"})
        raise
    results.put({"who": "holder", "made": None, "error": None})


def _create_one(root: str, results) -> None:
    store = MissionStore(root)
    try:
        made = store.create_mission(
            scope="scope-concurrent",
            objective="the second writer's objective",
            desired_result="the outcome holds",
            acceptance_criteria=[("crit-0", "demonstrated")],
            operation="second-writer",
        ).mission_id
    except BaseException as exc:
        results.put({"who": "writer", "made": None, "error": f"{type(exc).__name__}: {exc}"})
        raise
    results.put({"who": "writer", "made": made, "error": None})


def test_replacing_the_lock_pathname_cannot_cause_a_lost_update(tmp_path):
    """Two real processes, with the lock pathname replaced mid-critical-section.

    A lock *file* is a name, and a name can be unlinked and recreated by anyone
    who can write the directory: the second process then locks a different inode
    while the first is still inside, both read the same snapshot, and the second
    publication silently discards the first.  Exclusion is held on the store
    directory itself - which the holder already has open - so there is no second
    inode to acquire and the attack is inert rather than merely unlikely.

    The timing here is deterministic rather than hopeful: the pathname is
    replaced only once the holder has signalled that it is inside its
    transaction, so a bypass would be taken every single time, not sometimes.
    """

    store = MissionStore(tmp_path)
    _mission(store)

    context = multiprocessing.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    results = context.Queue()
    holder = context.Process(target=_hold_transaction, args=(str(tmp_path), entered, release, results))
    writer = context.Process(target=_create_one, args=(str(tmp_path), results))
    started = [holder, writer]

    try:
        holder.start()
        assert entered.wait(timeout=60), "the holder never entered its transaction"

        # The bypass, executed at the only moment it could ever work.
        _replace_lock_pathname(str(tmp_path))

        writer.start()
        # The writer must make no progress at all while the holder is inside.
        with pytest.raises(queue.Empty):
            results.get(timeout=5)

        release.set()
        reported = {report["who"]: report for report in (results.get(timeout=60) for _ in started)}
        for process in started:
            process.join(timeout=60)

        assert [report["error"] for report in reported.values()] == [None, None]
        assert [process.exitcode for process in started] == [0, 0]
    finally:
        release.set()
        _reap(started, results)

    # One durable result per reported success: the writer's mission is the
    # second one, and the first was not lost behind it.
    assert reported["writer"]["made"] == "m-000000000002"
    durable = sorted(row[0] for row in _query(store, "SELECT mission_id FROM missions"))
    assert durable == ["m-000000000001", "m-000000000002"]
    assert store.metrics().missions_open == 2
    # And the replaced pathname was never load-bearing in the first place.
    assert (store.root / "missions.lock").exists()


def _create_concurrently(root: str, count: int, churn: bool, barrier, results) -> None:
    store = MissionStore(root)
    barrier.wait(timeout=60)
    made = []
    try:
        for index in range(count):
            if churn:
                _replace_lock_pathname(root)
            made.append(
                store.create_mission(
                    scope="scope-concurrent",
                    objective=f"objective {index}",
                    desired_result="the outcome holds",
                    acceptance_criteria=[("crit-0", "demonstrated")],
                    operation=f"concurrent-{int(churn)}-{index}",
                ).mission_id
            )
    except BaseException as exc:  # reported, never swallowed: a child that dies
        results.put({"made": made, "error": f"{type(exc).__name__}: {exc}"})
        raise
    results.put({"made": made, "error": None})


def test_concurrent_writers_never_lose_an_update(tmp_path):
    """Throughput case: many missions from two processes, all of them durable."""

    store = MissionStore(tmp_path)
    _mission(store)

    per_process = 3
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_create_concurrently,
            args=(str(tmp_path), per_process, churn, barrier, results),
        )
        for churn in (False, True)
    ]
    for process in processes:
        process.start()
    try:
        reported = [results.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=60)
        failures = [report["error"] for report in reported if report["error"]]
        assert not failures, failures
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        _reap(processes, results)

    announced = sorted(mission_id for report in reported for mission_id in report["made"])
    assert len(announced) == 2 * per_process
    assert len(set(announced)) == len(announced)
    durable = sorted(row[0] for row in _query(store, "SELECT mission_id FROM missions"))
    assert durable == sorted([_MISSION] + announced)
    assert durable == [f"m-{index:012d}" for index in range(1, 2 * per_process + 2)]
    assert store.metrics().missions_open == 2 * per_process + 1


# --------------------------------------------------------------------------
# 23. an I/O error is never evidence of durability
# --------------------------------------------------------------------------


def test_a_directory_fsync_failure_is_never_reported_as_success(tmp_path, monkeypatch):
    """After the rename, "it probably worked" is not an answer this may give.

    The bytes may already be visible to a reader, so the call cannot claim
    failure; the directory entry was never synced, so it cannot claim durability
    either.  The only honest report is that the outcome is unknown - and it must
    not be retried for the caller, because creating a mission twice is not the
    same as creating it once.
    """

    store = MissionStore(tmp_path)
    _mission(store)
    real_fsync = os.fsync
    state = {"directory_syncs": 0}

    def fsyncing(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            state["directory_syncs"] += 1
            raise OSError(errno.EIO, "injected directory sync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(mission_control.os, "fsync", fsyncing)
    with pytest.raises(MissionIndeterminate):
        _mission(store)
    monkeypatch.undo()

    assert state["directory_syncs"] == 1
    # Indeterminate is not a synonym for "failed": the rename did happen.
    assert store.metrics().missions_open == 2


# --------------------------------------------------------------------------
# 24. the durable schema is exactly v1, not "at least" v1
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement,residue",
    [
        ("CREATE TABLE foreign_state (a TEXT)", "foreign_state"),
        ("CREATE VIEW foreign_view AS SELECT 1 AS a", "foreign_view"),
        (
            "CREATE TRIGGER foreign_trigger AFTER INSERT ON missions"
            " BEGIN SELECT 1; END",
            "foreign_trigger",
        ),
        ("CREATE INDEX foreign_index ON missions(scope)", "foreign_index"),
    ],
)
def test_an_unknown_schema_object_is_refused(tmp_path, statement, residue):
    """State this kernel did not write is state it cannot reason about.

    Accepting it means republishing it on every mutation - a foreign schema
    riding along inside a store that reports itself healthy, blessed by a kernel
    that never looked at it.
    """

    store = _populated(tmp_path)
    _sql(store, statement)

    with pytest.raises(MissionCorrupt) as excinfo:
        store.metrics()
    assert residue in str(excinfo.value)
    with pytest.raises(MissionCorrupt):
        _mission(store)
    # Refused, not quietly dropped: repairing the store is a host's decision.
    assert _query(store, "SELECT name FROM sqlite_master WHERE name = ?", (residue,))


def test_an_unknown_durable_setting_is_refused(tmp_path):
    store = _populated(tmp_path)
    _sql(store, "INSERT INTO meta (key, value) VALUES ('unknown-setting', 'x')")

    with pytest.raises(MissionCorrupt) as excinfo:
        store.metrics()
    assert "unknown-setting" in str(excinfo.value)
    with pytest.raises(MissionCorrupt):
        _mission(store)
    assert _query(store, "SELECT key FROM meta WHERE key = 'unknown-setting'")


# --------------------------------------------------------------------------
# 25. nothing publishes over durable state it has not validated
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement,params",
    [
        (
            "INSERT INTO claims (resource, work_item_id) VALUES ('not-a-digest', ?)",
            ("w-000000000999",),
        ),
        (
            "INSERT INTO claims (resource, work_item_id) VALUES (?, 'w-000000000999')",
            (_DIGEST,),
        ),
        ("UPDATE work_items SET priority = 99999 WHERE work_item_id = ?", (_SIDE,)),
        ("UPDATE work_items SET submitted_at = 9e999 WHERE work_item_id = ?", (_SIDE,)),
        ("UPDATE counters SET value = value + 3 WHERE name = 'work_seq'", ()),
    ],
)
def test_an_unrelated_mutation_will_not_publish_over_corruption(tmp_path, statement, params):
    """Validating only the rows an operation touches signs for the rest anyway.

    ``create_mission`` reads no claim and no work item, so a store carrying a
    malformed claim would keep being republished by unrelated traffic, each
    mutation putting its name to state it never looked at, until something
    incidental finally noticed.
    """

    store = _populated(tmp_path)
    _sql(store, statement, params)
    database = store.root / mission_control._DB_NAME
    before = database.read_bytes()

    with pytest.raises(MissionCorrupt):
        _mission(store)
    with pytest.raises(MissionCorrupt):
        _root(store, _MISSION)
    with pytest.raises(MissionCorrupt):
        with store.dispatch(operation=_key('dispatch'), worker="worker-1"):
            pass  # pragma: no cover - the dispatch never yields

    # Failed closed *and* left the bytes alone.
    assert database.read_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [
        b"not a database at all",
        b"SQLite format 3\x00" + b"\x00" * 512,
        b"SQLite format 3\x00" + bytes(range(256)) * 4,
    ],
)
def test_malformed_database_bytes_surface_as_a_typed_error(tmp_path, payload):
    """A caller should never have to catch ``sqlite3.DatabaseError`` from this."""

    store = MissionStore(tmp_path)
    _mission(store)
    (store.root / mission_control._DB_NAME).write_bytes(payload)

    for call in (
        store.metrics,
        store.snapshot,
        lambda: store.get_mission(_MISSION),
        lambda: _mission(store),
    ):
        with pytest.raises(MissionCorrupt):
            call()


# --------------------------------------------------------------------------
# 26. an indeterminate result must leave the store reconcilable
# --------------------------------------------------------------------------


def _failing_directory_fsync(monkeypatch, state):
    real_fsync = os.fsync

    def fsyncing(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            state["directory_syncs"] += 1
            raise OSError(errno.EIO, "injected directory sync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr(mission_control.os, "fsync", fsyncing)


def test_an_indeterminate_bootstrap_keeps_the_identity_it_published(tmp_path, monkeypatch):
    """Rollback must not delete the identity of a store that is already visible.

    The very first mutation both bootstraps and publishes.  If the directory
    sync fails afterwards the rename has already happened, so the database is on
    disk and a reader can already see it - but a rollback that reasons "the call
    failed, so undo the bootstrap" removes ``store.id`` and leaves behind a
    database that no longer belongs to any store.  That turns a recoverable
    indeterminate result into permanent corruption, and it does so at the one
    moment there is nothing to fall back on.
    """

    store = MissionStore(tmp_path)
    state = {"directory_syncs": 0}
    _failing_directory_fsync(monkeypatch, state)

    with pytest.raises(MissionIndeterminate):
        _mission(store)
    monkeypatch.undo()
    assert state["directory_syncs"] == 1

    # Both artifacts survived, and they still match each other.
    database = store.root / mission_control._DB_NAME
    sidecar = store.root / "store.id"
    assert database.exists() and sidecar.exists()

    # So the store reconciles: the publication that did land is readable, and
    # ordinary work continues from it rather than from a second universe.
    assert store.get_mission("m-000000000001").objective.startswith("reach the")
    assert store.metrics().missions_open == 1
    assert _mission(store).mission_id == "m-000000000002"
    assert MissionStore(tmp_path).metrics().missions_open == 2


# --------------------------------------------------------------------------
# 27. the claims table is exactly what the dispatched work items declare
# --------------------------------------------------------------------------


def test_a_deleted_claim_cannot_free_a_resource_that_is_still_held(tmp_path):
    """The exclusion record is not advisory, and its absence is not permission.

    Every surviving claim row can look perfect while the one that mattered has
    been removed, and a scheduler that checks rows individually will happily
    dispatch a second work item onto a resource whose first holder is still
    running.  What rules it out is requiring the claims table to be exactly the
    set of resources the dispatched items declare - no more and no fewer.
    """

    store = MissionStore(tmp_path)
    mission = _mission(store)
    contested = _resource("contested")
    root = _root(store, mission.mission_id, resources=[contested])
    rival = _side(store, mission.mission_id, root, resources=[contested])
    database = store.root / mission_control._DB_NAME

    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as held:
        assert held.work_item_id == root.work_item_id
        assert store.is_claimed(contested)

        # The exclusion record is deleted while the first lease is still live.
        _sql(store, "DELETE FROM claims")
        before = database.read_bytes()

        # No second dispatch of the same resource, and no unrelated mutation
        # either: the store is refused until the record tells the truth again.
        with pytest.raises(MissionCorrupt):
            with store.dispatch(operation=_key('dispatch'), worker="worker-2"):
                pass  # pragma: no cover - the dispatch never yields
        with pytest.raises(MissionCorrupt):
            _mission(store)
        with pytest.raises(MissionCorrupt):
            store.is_claimed(contested)
        with pytest.raises(MissionCorrupt):
            held.heartbeat()
        assert database.read_bytes() == before

        # The first lease is still the live one, and the item is still recorded
        # as dispatched: nothing was released by deleting a row.
        assert _query(
            store, "SELECT status, worker FROM work_items WHERE work_item_id = ?",
            (root.work_item_id,),
        ) == [(STATUS_DISPATCHED, "worker-1")]

        # Restore the truth and exclusion works exactly as before: the rival
        # cannot run while the resource is genuinely held.
        _sql(
            store,
            "INSERT INTO claims (resource, work_item_id) VALUES (?, ?)",
            (contested.digest, root.work_item_id),
        )
        with store.dispatch(operation=_key('dispatch'), worker="worker-3") as blocked:
            assert blocked is None
        assert store.get_work_item(rival.work_item_id).status == STATUS_READY


def test_an_extra_claim_for_an_undeclared_resource_is_refused(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id, resources=[_resource("declared")])
    database = store.root / mission_control._DB_NAME

    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as held:
        assert held is not None
        # A syntactically perfect claim, owned by a genuinely dispatched item,
        # for a resource that item never declared.
        _sql(
            store,
            "INSERT INTO claims (resource, work_item_id) VALUES (?, ?)",
            (_resource("undeclared").digest, root.work_item_id),
        )
        before = database.read_bytes()
        with pytest.raises(MissionCorrupt):
            _mission(store)
        with pytest.raises(MissionCorrupt):
            store.metrics()
        assert database.read_bytes() == before

        # Withdraw the surplus and the store is exactly itself again, which is
        # what proves the refusal was about this row and nothing else.
        _sql(store, "DELETE FROM claims WHERE resource = ?", (_resource("undeclared").digest,))
        assert store.metrics().dispatched == 1
        held.close(_fixed(), operation=_key('close'))


def test_a_claim_owned_by_the_wrong_dispatched_item_is_refused(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id, resources=[_resource("first")])
    other = _mission(store, scope="scope-b")
    _root(store, other.mission_id, resources=[_resource("second")])

    with store.dispatch(operation=_key('dispatch'), worker="worker-1") as first:
        with store.dispatch(operation=_key('dispatch'), worker="worker-2") as second:
            assert first is not None and second is not None
            # Both are dispatched and both hold a claim; the ownership is swapped.
            _sql(
                store,
                "UPDATE claims SET work_item_id = ? WHERE work_item_id = ?",
                (second.work_item_id, root.work_item_id),
            )
            with pytest.raises(MissionCorrupt):
                store.metrics()
            with pytest.raises(MissionCorrupt):
                _mission(store)

            _sql(
                store,
                "UPDATE claims SET work_item_id = ? WHERE resource = ?",
                (root.work_item_id, _resource("first").digest),
            )
            assert store.metrics().dispatched == 2
            first.close(_fixed(), operation=_key('close'))
            second.close(_fixed(), operation=_key('close'))


# --------------------------------------------------------------------------
# 28. a completed mission has to look like one
# --------------------------------------------------------------------------


def _forge_completion(store: MissionStore, mission_id: str) -> None:
    """Set every field a tamper can set in one statement, and nothing it cannot."""

    _sql(
        store,
        "UPDATE missions SET status = 'completed', completed_at = 1.0, evidence = ?"
        " WHERE mission_id = ?",
        (json.dumps({"crit-0": "evidence-a"}, separators=(",", ":")), mission_id),
    )
    _sql(store, "UPDATE counters SET value = value + 1 WHERE name = 'missions_completed'")


def test_a_completed_mission_with_open_work_is_refused(tmp_path):
    """Status, completion time and evidence are three fields.  The work is not."""

    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)
    database = store.root / mission_control._DB_NAME

    _forge_completion(store, mission.mission_id)
    before = database.read_bytes()

    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        store.get_work_item(root.work_item_id)
    with pytest.raises(MissionCorrupt):
        _mission(store)
    assert database.read_bytes() == before


def test_a_completed_mission_with_no_work_at_all_is_refused(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    _forge_completion(store, mission.mission_id)

    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        _mission(store)


@pytest.mark.parametrize("closure_factory", [_deferred, _blocked])
def test_a_completed_mission_whose_root_was_not_fixed_is_refused(tmp_path, closure_factory):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    _root(store, mission.mission_id)
    _run_one(store, closure_factory())

    _forge_completion(store, mission.mission_id)
    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        _mission(store)


# --------------------------------------------------------------------------
# 29. a reference nothing points at is still a reference
# --------------------------------------------------------------------------


def test_an_orphan_dependency_row_is_refused(tmp_path):
    """A dependency whose *source* does not exist is reachable from no traversal.

    Walking dependencies out of the stored work items can only ever find rows
    whose source is a stored work item, so a row pointing *from* a work item
    that was never stored is invisible to every check built that way.  Asking
    SQLite to verify its own declared foreign keys is what closes that gap.
    """

    store = _populated(tmp_path)
    database = store.root / mission_control._DB_NAME
    # The tampering connection has foreign keys off, exactly as an external
    # editor would, so the row lands despite the declared reference.
    _sql(
        store,
        "INSERT INTO dependencies (work_item_id, depends_on_id) VALUES ('w-000000000999', ?)",
        (_ROOT,),
    )
    before = database.read_bytes()

    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        _mission(store)
    with pytest.raises(MissionCorrupt):
        store.get_work_item(_ROOT)
    assert database.read_bytes() == before
    # Refused, not silently dropped.
    assert _query(store, "SELECT COUNT(*) FROM dependencies")[0][0] == 1


# --------------------------------------------------------------------------
# 30. every number that can be re-derived from the rows is re-derived
# --------------------------------------------------------------------------

_DRIFTS = [
    (
        "dispatch count inflated",
        "UPDATE counters SET value = value + 10 WHERE name IN"
        " ('dispatches', 'dispatch_seq', 'fence', 'latency_samples')",
        (),
    ),
    ("rotation forgotten", "DELETE FROM rotation", ()),
    (
        "rotation invented",
        "INSERT INTO rotation (scope, last_dispatch_seq) VALUES ('scope-ghost', 1)",
        (),
    ),
    ("rotation ahead of dispatch", "UPDATE rotation SET last_dispatch_seq = 99", ()),
    (
        "mission ids renumbered",
        "UPDATE missions SET mission_id = 'm-000000000007'",
        (),
    ),
    (
        "work ids renumbered",
        "UPDATE work_items SET work_item_id = 'w-000000000009' WHERE work_item_id = ?",
        (_SIDE,),
    ),
    (
        "submit sequence duplicated",
        "UPDATE work_items SET submit_seq = 1 WHERE work_item_id = ?",
        (_SIDE,),
    ),
    (
        "dispatch cursor names nothing",
        "UPDATE meta SET value = 'w-000000000999' WHERE key = 'dispatch_cursor'",
        (),
    ),
    (
        "dispatch cursor is not an id",
        "UPDATE meta SET value = 'somewhere' WHERE key = 'dispatch_cursor'",
        (),
    ),
    ("abandoned count invented", "UPDATE counters SET value = 3 WHERE name = 'abandoned'", ()),
    (
        "attempts disagree with dispatches",
        "UPDATE work_items SET attempts = attempts + 1 WHERE work_item_id = ?",
        (_ROOT,),
    ),
    (
        # Derived, never a literal.  ``_populated`` has one real dispatch, so the
        # recorded maximum *is* the recorded total, and any constant is either
        # impossible or perfectly valid depending on how long that dispatch
        # happened to take - which is a property of the machine, not of the
        # store.  One millisecond above the stored total is impossible on every
        # machine, because a maximum can never exceed a sum of non-negative
        # samples.
        "largest latency exceeds the total",
        "UPDATE counters SET value ="
        " (SELECT value FROM counters WHERE name = 'latency_ms_total') + 1"
        " WHERE name = 'latency_ms_max'",
        (),
    ),
]


@pytest.mark.parametrize(
    "statement,params",
    [(statement, params) for _, statement, params in _DRIFTS],
    ids=[label for label, _, _ in _DRIFTS],
)
def test_durable_state_that_drifts_from_the_rows_is_refused(tmp_path, statement, params):
    """Counters an operator reads are only worth reading if they cannot drift.

    Every number here is derivable from the stored rows - how many dispatches
    happened, which scopes have been served, which ids were minted - so a
    confident wrong answer is always avoidable, and an unverifiable one is never
    published.
    """

    store = _populated(tmp_path)
    database = store.root / mission_control._DB_NAME
    assert _sql(store, statement, params) > 0, "the corruption matched no rows"
    before = database.read_bytes()

    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        _mission(store)
    assert database.read_bytes() == before


def test_a_latency_maximum_above_the_total_is_refused(tmp_path):
    """The one latency relation that is impossible regardless of the clock.

    Every recorded latency is non-negative, so the largest single sample can
    never exceed their sum.  That is what makes this checkable without knowing
    anything about how fast the machine is - and it is why the damage below is
    derived from the stored total rather than written as a constant.  A constant
    would be impossible on a quick machine and perfectly valid on a slow one,
    and the test would then pass or fail on suite load rather than on behaviour.
    """

    store = _populated(tmp_path)
    samples = _counter(store, "latency_samples")
    total = _counter(store, "latency_ms_total")
    largest = _counter(store, "latency_ms_max")

    # The store is healthy first, and healthy in the way that matters here.
    assert samples == 1
    assert largest <= total
    store.metrics()

    assert (
        _sql(
            store,
            "UPDATE counters SET value ="
            " (SELECT value FROM counters WHERE name = 'latency_ms_total') + 1"
            " WHERE name = 'latency_ms_max'",
        )
        == 1
    )
    # Proof that the row really became impossible, before any public call: the
    # maximum is now strictly greater than the sum it is drawn from.
    damaged = _counter(store, "latency_ms_max")
    assert damaged == total + 1
    assert damaged > _counter(store, "latency_ms_total")

    database = store.root / mission_control._DB_NAME
    before = database.read_bytes()
    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        _mission(store)
    assert database.read_bytes() == before


def test_operational_counters_stay_bounded_rather_than_invented(tmp_path):
    """Not every counter is derivable, and those are not pretended to be.

    How many callers were rejected for capacity, or lost a resource race, is
    genuinely unrecoverable from the surviving rows.  Those are bounded and
    checked for shape, and no equality is invented for them.
    """

    store = _populated(tmp_path)
    for name in ("capacity_rejects", "conflicts", "stale_fence_rejects", "cursor_sweeps"):
        _sql(store, "UPDATE counters SET value = value + 4 WHERE name = ?", (name,))
    assert store.metrics().conflicts >= 4
    assert _mission(store).mission_id == "m-000000000002"

    # Bounded, though: a negative or absurd value is still refused.
    _sql(store, "UPDATE counters SET value = -1 WHERE name = 'conflicts'")
    with pytest.raises(MissionCorrupt):
        store.metrics()


def test_abandoning_work_keeps_the_counters_derivable(tmp_path):
    """The invariants have to hold for the honest paths too, not just the tampered ones."""

    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)
    groundwork = _side(store, mission.mission_id, root)
    dependent = _side(
        store, mission.mission_id, root, depends_on_ids=[groundwork.work_item_id]
    )

    _run_one(store)
    _run_one(store, _deferred())
    store.abandon_unrunnable_work_item(dependent.work_item_id, _blocked(), operation=_key('abandon'))

    metrics = store.metrics()
    assert metrics.abandoned == 1
    assert metrics.dispatches == 2
    assert (metrics.closed_fixed, metrics.closed_deferred, metrics.closed_blocked) == (1, 1, 1)
    assert store.snapshot().truncated is False


# --------------------------------------------------------------------------
# 31. a read is a read
# --------------------------------------------------------------------------

_WRITE_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND | getattr(os, "O_EXCL", 0)
)
_MUTATING_CALLS = (
    "fchmod", "chmod", "mkdir", "unlink", "rmdir", "replace", "rename",
    "ftruncate", "write", "fsync", "truncate",
)


def test_read_only_operations_issue_no_write_capable_filesystem_calls(tmp_path, monkeypatch):
    """"Without creating, locking or re-moding anything" has to be literally true.

    A read that quietly repairs a mode is a write the caller never asked for and
    cannot see, and a read that opens ``O_RDWR`` cannot run against a read-only
    mount, an audit copy or an observer account at all - which is most of the
    reason to have a read-only path in the first place.
    """

    store = _populated(tmp_path)
    real_open = os.open
    opened = []
    mutating = []

    def spying_open(path, flags, mode=0o777, *, dir_fd=None):
        opened.append(flags)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def spy(name, original):
        def call(*args, **kwargs):
            mutating.append(name)
            return original(*args, **kwargs)

        return call

    monkeypatch.setattr(mission_control.os, "open", spying_open)
    for name in _MUTATING_CALLS:
        monkeypatch.setattr(mission_control.os, name, spy(name, getattr(os, name)))

    store.metrics()
    store.snapshot()
    store.get_mission(_MISSION)
    store.get_work_item(_ROOT)
    store.is_claimed(_resource("nothing"))
    monkeypatch.undo()

    assert not mutating, mutating
    assert opened, "the read path opened nothing at all"
    assert all(not (flags & _WRITE_FLAGS) for flags in opened), [oct(f) for f in opened]


def test_a_read_only_store_is_readable(tmp_path):
    """0500 and 0400 are private to this account, so they are honoured, not fought."""

    if not _POSIX:  # pragma: no cover - POSIX mode bits are meaningless here
        pytest.skip("POSIX only")
    store = _populated(tmp_path)
    database = store.root / mission_control._DB_NAME
    sidecar = store.root / "store.id"
    modes = (0o500, 0o400, 0o400)
    for path, mode in zip((store.root, database, sidecar), modes):
        os.chmod(path, mode)
    try:
        assert store.metrics().missions_open == 1
        assert store.get_mission(_MISSION).mission_id == _MISSION
        assert store.snapshot().work_items
        # And the modes are exactly as the operator left them.
        assert tuple(
            stat.S_IMODE(path.stat().st_mode)
            for path in (store.root, database, sidecar)
        ) == modes
    finally:
        os.chmod(store.root, 0o700)
        os.chmod(database, 0o600)
        os.chmod(sidecar, 0o600)


@pytest.mark.parametrize(
    "directory_mode,file_mode",
    [(0o755, 0o600), (0o700, 0o644), (0o770, 0o600), (0o700, 0o666)],
)
def test_a_group_readable_store_is_refused_without_being_repaired(
    tmp_path, directory_mode, file_mode
):
    if not _POSIX:  # pragma: no cover - POSIX mode bits are meaningless here
        pytest.skip("POSIX only")
    store = _populated(tmp_path)
    database = store.root / mission_control._DB_NAME
    os.chmod(store.root, directory_mode)
    os.chmod(database, file_mode)
    try:
        with pytest.raises(MissionCorrupt):
            store.metrics()
        with pytest.raises(MissionCorrupt):
            _mission(store)
        # Refused with the modes untouched: a read does not re-mode anything,
        # and neither does a mutation that never got to write.
        assert stat.S_IMODE(store.root.stat().st_mode) == directory_mode
        assert stat.S_IMODE(database.stat().st_mode) == file_mode
    finally:
        os.chmod(store.root, 0o700)
        os.chmod(database, 0o600)


@pytest.mark.parametrize(
    "identity",
    [
        b"a" * 63,
        b"a" * 65,
        b"a" * 64 + b"\n",
        b"A" * 64,
        (b"a" * 63) + b"g",
        b"",
    ],
)
def test_the_store_identity_shape_is_exact(tmp_path, identity):
    """A sidecar that is not sixty-four hex bytes is not an identity at all."""

    store = _populated(tmp_path)
    sidecar = store.root / "store.id"
    sidecar.write_bytes(identity)
    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        _mission(store)


@pytest.mark.parametrize("suffix", [b"\n", b" ", b"\r\n", b"\t"])
def test_the_real_identity_with_trailing_whitespace_is_still_refused(tmp_path, suffix):
    """The shape rule, isolated from the value rule.

    Every other identity case would also be refused for *not matching* the value
    recorded inside the database, which would leave the shape rule untested.
    Here the value is exactly right and only the framing is wrong, so nothing
    but the format check can be doing the work - and accommodating a stray
    newline is precisely the sort of helpfulness that turns a fingerprint into a
    suggestion.
    """

    store = _populated(tmp_path)
    sidecar = store.root / "store.id"
    identity = sidecar.read_bytes()
    assert len(identity) == 64 and identity.decode("ascii").islower()

    sidecar.write_bytes(identity + suffix)
    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        _mission(store)

    # And with the exact bytes back, the store is simply itself again.
    sidecar.write_bytes(identity)
    assert store.metrics().missions_open == 1


# --------------------------------------------------------------------------
# 32. an interrupted mutation is answerable, not merely ambiguous
# --------------------------------------------------------------------------


def _indeterminate(monkeypatch):
    """Fail the directory sync *after* the rename, exactly where it hurts."""

    state = {"directory_syncs": 0}
    _failing_directory_fsync(monkeypatch, state)
    return state


def test_an_indeterminate_dispatch_is_recovered_by_its_own_operation(tmp_path, monkeypatch):
    """The dispatch that nobody received is the worst kind of ambiguity.

    The row says dispatched, the resource is claimed, the counters have moved -
    and no caller ever got a handle, so nothing on earth is going to close it.
    Without a durable identity for the attempt, the only way out is an operator
    noticing.  With one, the same operation reclaims *the same* work item at
    *the same* fencing token, and the second attempt is a recovery rather than a
    new dispatch.
    """

    store = MissionStore(tmp_path)
    mission = _mission(store)
    contested = _resource("contested")
    root = _root(store, mission.mission_id, resources=[contested])

    state = _indeterminate(monkeypatch)
    with pytest.raises(MissionIndeterminate):
        with store.dispatch(operation="attempt-1", worker="worker-1"):
            pass  # pragma: no cover - the dispatch never yields
    monkeypatch.undo()
    assert state["directory_syncs"] == 1

    # The durable scheduler believes somebody is executing.  Nobody is.
    stranded = store.get_work_item(root.work_item_id)
    assert stranded.status == STATUS_DISPATCHED
    assert store.is_claimed(contested)
    assert store.metrics().dispatches == 1

    # A *fresh* store object, as a restarted service would have, recovers it.
    recovered = MissionStore(tmp_path)
    with recovered.dispatch(operation="attempt-1", worker="worker-1") as handle:
        assert handle is not None
        assert handle.work_item_id == root.work_item_id
        assert handle.fence == stranded.fence
        handle.close(_fixed(), operation="close-1")

    # Recovered, not re-dispatched: one dispatch, one token, one claim released.
    after = recovered.metrics()
    assert after.dispatches == 1
    assert after.dispatched == 0
    assert not recovered.is_claimed(contested)
    assert recovered.get_work_item(root.work_item_id).disposition == DISPOSITION_FIXED


def _recover_in_child(root: str, key: str, results) -> None:
    store = MissionStore(root)
    try:
        with store.dispatch(operation=key, worker="worker-1") as handle:
            results.put(
                {"work_item_id": handle.work_item_id, "fence": handle.fence, "error": None}
            )
            handle.close(_fixed(), operation=key + "-close")
    except BaseException as exc:
        results.put({"work_item_id": None, "fence": None, "error": f"{type(exc).__name__}: {exc}"})
        raise


def test_an_indeterminate_dispatch_is_recovered_by_another_process(tmp_path, monkeypatch):
    """The recovering caller is usually not the process that was interrupted."""

    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id, resources=[_resource("contested")])

    _indeterminate(monkeypatch)
    with pytest.raises(MissionIndeterminate):
        with store.dispatch(operation="attempt-1", worker="worker-1"):
            pass  # pragma: no cover - the dispatch never yields
    monkeypatch.undo()
    stranded = store.get_work_item(root.work_item_id)

    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    child = context.Process(
        target=_recover_in_child, args=(str(tmp_path), "attempt-1", results)
    )
    child.start()
    try:
        report = results.get(timeout=60)
        child.join(timeout=60)
        assert report["error"] is None, report["error"]
        assert child.exitcode == 0
    finally:
        _reap([child], results)

    assert report["work_item_id"] == root.work_item_id
    assert report["fence"] == stranded.fence
    assert store.metrics().dispatches == 1
    assert store.get_work_item(root.work_item_id).status == STATUS_CLOSED


def test_a_dispatch_key_cannot_be_reused_by_a_different_worker(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    _root(store, mission.mission_id)

    with store.dispatch(operation="attempt-1", worker="worker-1") as handle:
        assert handle is not None
        handle.close(_fixed(), operation="close-1")

    with pytest.raises(MissionOperationConflict):
        with store.dispatch(operation="attempt-1", worker="worker-2"):
            pass  # pragma: no cover - the dispatch never yields


def test_recovering_a_dispatch_whose_lease_is_live_is_refused(tmp_path):
    """Recovery hands back authority.  It never mints a second copy of it."""

    store = MissionStore(tmp_path)
    mission = _mission(store)
    _root(store, mission.mission_id)

    with store.dispatch(operation="attempt-1", worker="worker-1") as held:
        assert held is not None
        with pytest.raises(MissionConflict):
            with store.dispatch(operation="attempt-1", worker="worker-1"):
                pass  # pragma: no cover - the dispatch never yields
        # The original is untouched and still authoritative.
        assert store.get_work_item(held.work_item_id).fence == held.fence
        held.close(_fixed(), operation="close-1")


def test_recovering_a_dispatch_whose_era_ended_is_terminal(tmp_path):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    root = _root(store, mission.mission_id)

    with store.dispatch(operation="attempt-1", worker="worker-1") as handle:
        fence = handle.fence
        handle.close(_fixed(), operation="close-1")

    with pytest.raises(MissionOperationSettled) as excinfo:
        with store.dispatch(operation="attempt-1", worker="worker-1"):
            pass  # pragma: no cover - the dispatch never yields
    assert excinfo.value.result == {"work_item_id": root.work_item_id, "fence": fence}


@pytest.mark.parametrize("kind", ["create", "submit", "complete", "close", "abandon"])
def test_an_interrupted_mutation_reconciles_to_its_original_effect(tmp_path, monkeypatch, kind):
    """Same key, same payload: the original ids, counters and timestamps.

    Not a second mission that looks like the first, and not a second closure
    counted twice - the effect that already happened, read back from durable
    state.
    """

    store = MissionStore(tmp_path)
    if kind == "create":
        state = _indeterminate(monkeypatch)
        with pytest.raises(MissionIndeterminate):
            store.create_mission(
                operation="op-1",
                scope="scope-a",
                objective="an objective",
                desired_result="a desired result",
                acceptance_criteria=[("crit-0", "demonstrated")],
            )
        monkeypatch.undo()
        assert state["directory_syncs"] == 1
        again = MissionStore(tmp_path).create_mission(
            operation="op-1",
            scope="scope-a",
            objective="an objective",
            desired_result="a desired result",
            acceptance_criteria=[("crit-0", "demonstrated")],
        )
        assert again.mission_id == "m-000000000001"
        assert store.metrics().missions_open == 1
        return

    mission = _mission(store)
    if kind == "submit":
        _indeterminate(monkeypatch)
        with pytest.raises(MissionIndeterminate):
            store.submit_work_item(
                mission.mission_id, operation="op-1", coordinates=_coords(), root=True
            )
        monkeypatch.undo()
        again = MissionStore(tmp_path).submit_work_item(
            mission.mission_id, operation="op-1", coordinates=_coords(), root=True
        )
        assert again.work_item_id == "w-000000000001"
        assert store.metrics().queue_depth == 1
        return

    root = _root(store, mission.mission_id)
    if kind == "close":
        with store.dispatch(operation="d-1", worker="worker-1") as handle:
            _indeterminate(monkeypatch)
            with pytest.raises(MissionIndeterminate):
                handle.close(_fixed(), operation="op-1")
            monkeypatch.undo()
            # Same key, same payload: reconciled, and counted exactly once.
            settled = handle.close(_fixed(), operation="op-1")
            assert settled.disposition == DISPOSITION_FIXED
        assert store.metrics().closed_fixed == 1
        return

    if kind == "abandon":
        groundwork = _side(store, mission.mission_id, root)
        dependent = _side(
            store, mission.mission_id, root, depends_on_ids=[groundwork.work_item_id]
        )
        _run_one(store)
        _run_one(store, _deferred())
        _indeterminate(monkeypatch)
        with pytest.raises(MissionIndeterminate):
            store.abandon_unrunnable_work_item(
                dependent.work_item_id, _blocked(), operation="op-1"
            )
        monkeypatch.undo()
        MissionStore(tmp_path).abandon_unrunnable_work_item(
            dependent.work_item_id, _blocked(), operation="op-1"
        )
        assert store.metrics().abandoned == 1
        assert store.metrics().closed_blocked == 1
        return

    _run_one(store)
    _indeterminate(monkeypatch)
    with pytest.raises(MissionIndeterminate):
        store.complete_mission(mission.mission_id, {"crit-0": "e-0"}, operation="op-1")
    monkeypatch.undo()
    again = MissionStore(tmp_path).complete_mission(
        mission.mission_id, {"crit-0": "e-0"}, operation="op-1"
    )
    assert again.status == MISSION_COMPLETED
    assert store.metrics().missions_completed == 1


def test_one_key_cannot_name_two_different_operations(tmp_path):
    store = MissionStore(tmp_path)
    mission = store.create_mission(
        operation="shared-key",
        scope="scope-a",
        objective="an objective",
        desired_result="a desired result",
        acceptance_criteria=[("crit-0", "demonstrated")],
    )

    # Same key, different payload.
    with pytest.raises(MissionOperationConflict):
        store.create_mission(
            operation="shared-key",
            scope="scope-a",
            objective="a different objective",
            desired_result="a desired result",
            acceptance_criteria=[("crit-0", "demonstrated")],
        )
    # Same key, different kind.
    with pytest.raises(MissionOperationConflict):
        store.submit_work_item(
            mission.mission_id, operation="shared-key", coordinates=_coords(), root=True
        )
    # And the original is still exactly itself.
    assert store.get_mission(mission.mission_id).objective == "an objective"
    assert store.metrics().missions_open == 1


# --------------------------------------------------------------------------
# 33. receipts are durable state, and they are bounded
# --------------------------------------------------------------------------


def test_receipts_are_released_only_by_acknowledgement(tmp_path, monkeypatch):
    """Retention is the honest half of idempotency.

    A receipt is the only record of whether an interrupted call took effect, so
    it cannot be reclaimed on a timer or by age.  The store fills, refuses more
    work, and says why - it does not quietly discard the evidence somebody may
    be about to ask for.
    """

    monkeypatch.setattr(mission_control, "MAX_OPERATIONS", 3)
    store = MissionStore(tmp_path)
    for index in range(3):
        store.create_mission(
            operation=f"op-{index}",
            scope="scope-a",
            objective=f"objective {index}",
            desired_result="a desired result",
            acceptance_criteria=[("crit-0", "demonstrated")],
        )
    assert store.metrics().operations_retained == 3

    with pytest.raises(MissionCapacityExceeded):
        store.create_mission(
            operation="op-3",
            scope="scope-a",
            objective="objective 3",
            desired_result="a desired result",
            acceptance_criteria=[("crit-0", "demonstrated")],
        )
    # Nothing was created behind the refusal.
    assert store.metrics().missions_open == 3

    assert store.acknowledge_operation("op-0") is True
    assert store.acknowledge_operation("op-0") is True  # idempotent
    store.create_mission(
        operation="op-3",
        scope="scope-a",
        objective="objective 3",
        desired_result="a desired result",
        acceptance_criteria=[("crit-0", "demonstrated")],
    )
    assert store.metrics().missions_open == 4
    # The acknowledged receipt was the one released; the others are intact, so
    # their outcomes are still answerable.
    assert store.acknowledge_operation("op-0") is False
    for index in (1, 2):
        assert store.acknowledge_operation(f"op-{index}") is True


_RECEIPT_CORRUPTIONS = [
    (
        "unknown kind",
        "UPDATE operations SET kind = 'invent-something'",
        (),
    ),
    (
        "payload is not a digest",
        "UPDATE operations SET payload = 'nope'",
        (),
    ),
    (
        "result names nothing",
        "UPDATE operations SET result = ?",
        ('{"mission_id":"m-000000000009"}',),
    ),
    (
        "result shape is wrong",
        "UPDATE operations SET result = ?",
        ('{"work_item_id":"w-000000000001"}',),
    ),
    (
        "sequence beyond the counter",
        "UPDATE operations SET sequence = 999",
        (),
    ),
    (
        "acknowledgement is malformed",
        "UPDATE operations SET acknowledged = 7",
        (),
    ),
    (
        "orphan receipt",
        "INSERT INTO operations (operation_key, kind, payload, result, recorded_at,"
        " sequence, acknowledged) VALUES ('ghost', 'create-mission', ?, ?, 1.0, 900, 0)",
        ("b" * 64, '{"mission_id":"m-000000000009"}'),
    ),
]


@pytest.mark.parametrize(
    "statement,params",
    [(statement, params) for _, statement, params in _RECEIPT_CORRUPTIONS],
    ids=[label for label, _, _ in _RECEIPT_CORRUPTIONS],
)
def test_damaged_receipts_fail_closed(tmp_path, statement, params):
    store = MissionStore(tmp_path)
    mission = _mission(store)
    database = store.root / mission_control._DB_NAME
    _sql(store, "DELETE FROM operations WHERE kind != 'create-mission'")
    _sql(store, statement, params)
    before = database.read_bytes()

    with pytest.raises(MissionCorrupt):
        store.metrics()
    with pytest.raises(MissionCorrupt):
        store.get_mission(mission.mission_id)
    with pytest.raises(MissionCorrupt):
        _mission(store)
    assert database.read_bytes() == before


def test_receipts_leak_nothing_through_observability(tmp_path):
    marker = "do-not-log-me-9c1b"
    store = MissionStore(tmp_path)
    store.create_mission(
        operation=f"key-{marker}",
        scope="scope-a",
        objective=f"objective {marker}",
        desired_result=f"result {marker}",
        acceptance_criteria=[("crit-0", f"criterion {marker}")],
    )
    assert marker not in repr(store.snapshot())
    assert marker not in repr(store.metrics())
    assert store.metrics().operations_retained == 1


# --------------------------------------------------------------------------
# 34. the host answer, and the shape of whole-store validation
# --------------------------------------------------------------------------


def test_host_preflight_is_read_only_and_reports_missing_primitives(tmp_path, monkeypatch):
    """A service has to be able to refuse work before it accepts any."""

    real_open = os.open
    opened = []
    mutating = []

    def spying_open(path, flags, mode=0o777, *, dir_fd=None):
        opened.append(flags)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    def spy(name, original):
        def call(*args, **kwargs):
            mutating.append(name)
            return original(*args, **kwargs)

        return call

    monkeypatch.setattr(mission_control.os, "open", spying_open)
    for name in _MUTATING_CALLS:
        monkeypatch.setattr(mission_control.os, name, spy(name, getattr(os, name)))
    capabilities = mission_control.inspect_host()
    monkeypatch.undo()

    assert capabilities.supported is True
    assert capabilities.missing == ()
    assert not mutating and not opened

    monkeypatch.setattr(mission_control, "fcntl", types.SimpleNamespace())
    unsupported = mission_control.inspect_host()
    monkeypatch.undo()
    assert unsupported.supported is False
    assert unsupported.missing == ("interprocess-locking",)
    assert unsupported.database_binding is True


def test_whole_store_validation_does_not_query_once_per_work_item(tmp_path, monkeypatch):
    """Validation must be complete without being quadratic.

    Decoding each row's dependencies with its own query turns a whole-store pass
    into a query per item; the cost lands on every read and every mutation.  The
    shape is asserted rather than the wall clock, so this cannot go green on a
    fast machine.
    """

    def _count_queries(store, items):
        mission = _mission(store)
        root = _root(store, mission.mission_id)
        for _ in range(items - 1):
            _side(store, mission.mission_id, root)
        statements = []
        real_connect = mission_control.sqlite3.connect

        def counting(*args, **kwargs):
            connection = real_connect(*args, **kwargs)
            connection.set_trace_callback(statements.append)
            return connection

        monkeypatch.setattr(mission_control.sqlite3, "connect", counting)
        store.metrics()
        monkeypatch.undo()
        return len(statements)

    small = _count_queries(MissionStore(tmp_path / "small"), 8)
    large = _count_queries(MissionStore(tmp_path / "large"), 32)
    # Four times the work items must not mean four times the queries.
    assert large - small <= 2, (small, large)


def test_missing_counters_are_never_reported_as_zero(tmp_path):
    """The one failure mode a metric must not have is a confident wrong answer.

    A store that lost its counters would otherwise report a perfectly healthy
    scheduler - no conflicts, no rejects, no stale fences - which is exactly the
    reading an operator would trust.
    """

    store = _populated(tmp_path)
    assert store.metrics().dispatches == 1
    _sql(store, "DELETE FROM counters")
    with pytest.raises(MissionCorrupt):
        store.metrics()
