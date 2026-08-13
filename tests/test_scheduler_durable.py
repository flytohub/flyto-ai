"""Focused contracts for the durable MissionStore-governed scheduler."""
import asyncio
import math
import multiprocessing
import os
import sqlite3
import stat
from datetime import datetime, timezone

import pytest

from flyto_ai.orchestration.mission_control import Closure, DISPOSITION_FIXED
from flyto_ai.scheduler import ScheduledTask, Scheduler, TaskSchedule
from flyto_ai.scheduler.catalog import CatalogError, ScheduleCatalog
from flyto_ai.scheduler.engine import _op
from flyto_ai.scheduler.tasks import ScheduleType


def task(**kwargs):
    values = dict(task_id="durable-task", name="Durable", instruction="bounded instruction")
    values.update(kwargs)
    return ScheduledTask(**values)


def _put_catalog_task(state_root, task_id, start):
    start.wait()
    ScheduleCatalog(state_root).put(task(task_id=task_id))


def _receipt_acknowledged(store, operation):
    with store._read() as txn:
        row = txn.conn.execute(
            "SELECT acknowledged FROM operations WHERE operation_key=?", (operation,)
        ).fetchone()
    assert row is not None
    return bool(row["acknowledged"])


@pytest.mark.asyncio
async def test_durable_restart_and_real_dispatch(tmp_path):
    calls = []

    async def executor(instruction):
        calls.append(instruction)
        await asyncio.sleep(0.3)  # long enough to require an automatic heartbeat
        return {"ok": True, "message": "done", "cost_usd": 0.25}

    first = Scheduler(executor=executor, state_root=tmp_path, check_interval=0.05)
    first.add_task(task(schedule=TaskSchedule(type=ScheduleType.ONE_SHOT)))
    results = await first.run_once()
    assert len(results) == 1 and results[0].ok
    assert calls == ["bounded instruction"]
    occurrence = next(item for item in first._mission_store.snapshot().work_items if not item.is_root)
    assert occurrence.attempts == 1 and occurrence.heartbeats > 0

    restarted = Scheduler(executor=executor, state_root=tmp_path)
    assert restarted.summary()["durable"] is True
    assert restarted.summary()["tasks"][0]["run_count"] == 1
    assert await restarted.run_once() == []


@pytest.mark.asyncio
async def test_recovery_after_executor_side_effect_never_reexecutes(tmp_path):
    calls = 0
    entered = asyncio.Event()

    async def executor(_):
        nonlocal calls
        calls += 1
        entered.set()  # the externally visible side effect
        await asyncio.Event().wait()

    first = Scheduler(executor=executor, state_root=tmp_path)
    first.add_task(task(schedule=TaskSchedule(type=ScheduleType.ONE_SHOT)))
    dispatch = asyncio.create_task(first.run_once())
    await entered.wait()
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
    requeued = next(
        item for item in first._mission_store.snapshot().work_items if not item.is_root
    )
    assert requeued.status == "ready" and requeued.attempts == 1
    assert _receipt_acknowledged(
        first._mission_store, _op("dispatch", requeued.work_item_id, 1)
    )

    async def must_not_run(_):
        raise AssertionError("recovered occurrence was executed twice")

    restarted = Scheduler(executor=must_not_run, state_root=tmp_path)
    result = (await restarted.run_once())[0]
    assert calls == 1
    assert not result.ok and result.error == "execution_outcome_unknown"
    assert result.message == "" and result.cost_usd == 0.0
    assert restarted.summary()["tasks"][0]["run_count"] == 1
    occurrence = next(
        item for item in restarted._mission_store.snapshot().work_items if not item.is_root
    )
    assert occurrence.status == "closed" and occurrence.attempts == 2
    assert await Scheduler(executor=must_not_run, state_root=tmp_path).run_once() == []


@pytest.mark.asyncio
async def test_cancellation_survives_dispatch_acknowledgement_cleanup_error(
    tmp_path, monkeypatch
):
    entered = asyncio.Event()

    async def executor(_):
        entered.set()
        await asyncio.Event().wait()

    scheduler = Scheduler(executor=executor, state_root=tmp_path)
    scheduler.add_task(task(schedule=TaskSchedule(type=ScheduleType.ONE_SHOT)))
    acknowledge = scheduler._mission_store.acknowledge_operation
    fail_cleanup = False

    def failed_acknowledgement(operation):
        if fail_cleanup and operation.startswith("sched-dispatch-"):
            raise RuntimeError("acknowledgement cleanup failed")
        return acknowledge(operation)

    monkeypatch.setattr(
        scheduler._mission_store, "acknowledge_operation", failed_acknowledgement
    )
    dispatch = asyncio.create_task(scheduler.run_once())
    await entered.wait()
    fail_cleanup = True
    dispatch.cancel()

    with pytest.raises(asyncio.CancelledError):
        await dispatch


@pytest.mark.asyncio
async def test_dispatch_acknowledgement_cleanup_error_propagates_without_cancellation(
    tmp_path, monkeypatch
):
    scheduler = Scheduler(state_root=tmp_path)
    scheduled = task(schedule=TaskSchedule(type=ScheduleType.ONE_SHOT))
    scheduler.add_task(scheduled)
    claimed, slot = scheduler._catalog.claim_due(2_000_000_000)[0]
    scheduler._materialize(claimed, slot)

    anchor_id = scheduler._mission_store.scheduler_order(limit=1)[0]
    anchor_op = _op("dispatch", anchor_id, 1)
    with scheduler._mission_store.dispatch_expected(
        operation=anchor_op,
        worker=scheduler._worker,
        work_item_id=anchor_id,
        expected_attempt=1,
    ) as handle:
        handle.close(
            Closure(DISPOSITION_FIXED),
            operation=_op("anchor-close", anchor_id, handle.fence),
        )
    scheduler._mission_store.acknowledge_operation(anchor_op)

    occurrence_id = scheduler._mission_store.scheduler_order(limit=1)[0]
    occurrence_dispatch_op = _op("dispatch", occurrence_id, 1)
    acknowledge = scheduler._mission_store.acknowledge_operation

    def fail_occurrence_dispatch_acknowledgement(operation):
        if operation == occurrence_dispatch_op:
            raise RuntimeError("occurrence dispatch cleanup failed")
        return acknowledge(operation)

    monkeypatch.setattr(scheduler, "get_task", lambda _task_id: None)
    monkeypatch.setattr(
        scheduler._mission_store,
        "acknowledge_operation",
        fail_occurrence_dispatch_acknowledgement,
    )

    with pytest.raises(RuntimeError, match="occurrence dispatch cleanup failed"):
        await scheduler._drain_dispatch()


@pytest.mark.asyncio
async def test_same_fence_recovery_is_closed_unknown_without_executor(tmp_path):
    calls = 0

    async def executor(_):
        nonlocal calls
        calls += 1
        return {"ok": True}

    scheduler = Scheduler(executor=executor, state_root=tmp_path)
    scheduled = task(schedule=TaskSchedule(type=ScheduleType.ONE_SHOT))
    scheduler.add_task(scheduled)
    claimed, slot = scheduler._catalog.claim_due(2_000_000_000)[0]
    scheduler._materialize(claimed, slot)
    with scheduler._catalog.transaction() as conn:
        occurrence = conn.execute("SELECT * FROM occurrences").fetchone()
        conn.execute(
            "UPDATE occurrences SET fence=1 WHERE task_id=? AND slot=?",
            (occurrence["task_id"], occurrence["slot"]),
        )
    result = (await scheduler._drain_dispatch())[0]
    assert calls == 0
    assert result.error == "execution_outcome_unknown"


def test_closed_occurrence_reconciliation_releases_exact_dispatch_receipt(tmp_path):
    scheduler = Scheduler(state_root=tmp_path)
    scheduled = task(schedule=TaskSchedule(type=ScheduleType.ONE_SHOT))
    scheduler.add_task(scheduled)
    claimed, slot = scheduler._catalog.claim_due(2_000_000_000)[0]
    scheduler._materialize(claimed, slot)

    # Close the internal anchor so the occurrence becomes authoritative first.
    anchor_id = scheduler._mission_store.scheduler_order(limit=1)[0]
    anchor_op = _op("dispatch", anchor_id, 1)
    with scheduler._mission_store.dispatch_expected(
        operation=anchor_op,
        worker=scheduler._worker,
        work_item_id=anchor_id,
        expected_attempt=1,
    ) as handle:
        handle.close(Closure(DISPOSITION_FIXED), operation=_op("anchor-close", anchor_id, handle.fence))
    scheduler._mission_store.acknowledge_operation(anchor_op)

    occurrence_id = scheduler._mission_store.scheduler_order(limit=1)[0]
    dispatch_op = _op("dispatch", occurrence_id, 1)
    with scheduler._mission_store.dispatch_expected(
        operation=dispatch_op,
        worker=scheduler._worker,
        work_item_id=occurrence_id,
        expected_attempt=1,
    ) as handle:
        with scheduler._catalog.transaction() as conn:
            conn.execute(
                "UPDATE occurrences SET fence=? WHERE task_id=? AND slot=?",
                (handle.fence, scheduled.task_id, slot),
            )
        handle.close(
            Closure(DISPOSITION_FIXED),
            operation=_op("occurrence-close", scheduled.task_id, slot, handle.fence),
        )

    assert not _receipt_acknowledged(scheduler._mission_store, dispatch_op)
    restarted = Scheduler(state_root=tmp_path)
    restarted._materialize(restarted.get_task(scheduled.task_id), slot)
    assert _receipt_acknowledged(restarted._mission_store, dispatch_op)
    occurrence = restarted._mission_store.get_work_item(occurrence_id)
    assert occurrence.status == "closed" and occurrence.attempts == 1


@pytest.mark.asyncio
async def test_heartbeat_failure_projects_unknown_without_masking_or_retry(tmp_path, monkeypatch):
    calls = 0
    entered = asyncio.Event()

    async def executor(_):
        nonlocal calls
        calls += 1
        entered.set()
        await asyncio.Event().wait()

    async def failed_heartbeat(_handle):
        await entered.wait()
        raise RuntimeError("heartbeat sentinel secret")

    scheduler = Scheduler(executor=executor, state_root=tmp_path)
    monkeypatch.setattr(scheduler, "_heartbeat", failed_heartbeat)
    scheduler.add_task(task(schedule=TaskSchedule(type=ScheduleType.ONE_SHOT)))
    result = (await scheduler.run_once())[0]
    assert calls == 1
    assert result.error == "execution_outcome_unknown"
    assert result.message == "" and result.cost_usd == 0.0
    assert await Scheduler(executor=executor, state_root=tmp_path).run_once() == []
    assert calls == 1


@pytest.mark.asyncio
async def test_heartbeat_failure_cancels_hanging_executor_without_waiting(tmp_path, monkeypatch):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def executor(_):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def failed_heartbeat(_handle):
        await entered.wait()
        raise RuntimeError("heartbeat failed")

    scheduler = Scheduler(executor=executor, state_root=tmp_path)
    monkeypatch.setattr(scheduler, "_heartbeat", failed_heartbeat)
    scheduler.add_task(task(schedule=TaskSchedule(type=ScheduleType.ONE_SHOT)))
    result = (await scheduler.run_once())[0]
    assert cancelled.is_set()
    assert result.error == "execution_outcome_unknown"
    assert result.message == "" and result.cost_usd == 0.0
    assert await Scheduler(executor=executor, state_root=tmp_path).run_once() == []


@pytest.mark.asyncio
async def test_heartbeat_unknown_survives_executor_cleanup_error(tmp_path, monkeypatch):
    entered = asyncio.Event()

    async def executor(_):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            raise RuntimeError("executor cleanup sentinel secret")

    async def returned_heartbeat(_handle):
        await entered.wait()

    scheduler = Scheduler(executor=executor, state_root=tmp_path)
    monkeypatch.setattr(scheduler, "_heartbeat", returned_heartbeat)
    scheduler.add_task(task(schedule=TaskSchedule(type=ScheduleType.ONE_SHOT)))
    result = (await scheduler.run_once())[0]
    assert result.error == "execution_outcome_unknown"
    assert result.message == "" and result.cost_usd == 0.0
    assert "sentinel" not in str(scheduler.summary())
    assert await Scheduler(executor=executor, state_root=tmp_path).run_once() == []


@pytest.mark.asyncio
async def test_heartbeat_return_cancels_hanging_executor_without_waiting(tmp_path, monkeypatch):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def executor(_):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def returned_heartbeat(_handle):
        await entered.wait()

    scheduler = Scheduler(executor=executor, state_root=tmp_path)
    monkeypatch.setattr(scheduler, "_heartbeat", returned_heartbeat)
    scheduler.add_task(task(schedule=TaskSchedule(type=ScheduleType.ONE_SHOT)))
    result = (await scheduler.run_once())[0]
    assert cancelled.is_set()
    assert result.error == "execution_outcome_unknown"
    assert result.message == "" and result.cost_usd == 0.0
    occurrence = next(
        item for item in scheduler._mission_store.snapshot().work_items if not item.is_root
    )
    assert occurrence.status == "closed"
    assert await Scheduler(executor=executor, state_root=tmp_path).run_once() == []


@pytest.mark.asyncio
async def test_background_loop_fail_stops_with_secret_free_code(tmp_path, monkeypatch):
    scheduler = Scheduler(state_root=tmp_path, check_interval=0.01)

    async def broken_run_once():
        raise RuntimeError("loop sentinel secret")

    monkeypatch.setattr(scheduler, "run_once", broken_run_once)
    await scheduler.start()
    await scheduler._loop_task
    summary = scheduler.summary()
    assert summary["running"] is False
    assert summary["error"] == "scheduler_durable_system_failure"
    assert "sentinel" not in str(summary)


@pytest.mark.asyncio
@pytest.mark.parametrize("output,error", [
    ({}, "executor_failed"),
    ({"ok": 1}, "executor_failed"),
    ({"ok": True, "nested": {}}, "executor_failed"),
    ({"ok": True, "cost_usd": math.inf}, "executor_failed"),
])
async def test_untrusted_executor_output_fails_closed(tmp_path, output, error):
    async def executor(_):
        return output

    scheduler = Scheduler(executor=executor, state_root=tmp_path)
    scheduler.add_task(task(schedule=TaskSchedule(type=ScheduleType.ONE_SHOT)))
    result = (await scheduler.run_once())[0]
    assert not result.ok and result.error == error
    item = next(item for item in scheduler._mission_store.snapshot().work_items if not item.is_root)
    assert item.disposition == "blocked"
    assert await Scheduler(executor=executor, state_root=tmp_path).run_once() == []
    with scheduler._catalog.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_over_budget_is_not_success(tmp_path):
    async def executor(_):
        return {"ok": True, "cost_usd": 2.0}

    scheduler = Scheduler(executor=executor, state_root=tmp_path)
    scheduler.add_task(task(budget_usd=1.0, schedule=TaskSchedule(type=ScheduleType.ONE_SHOT)))
    result = (await scheduler.run_once())[0]
    assert not result.ok and result.error == "budget_exceeded"


@pytest.mark.asyncio
async def test_two_schedulers_race_one_slot_one_executor(tmp_path):
    calls = 0

    async def executor(_):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.2)
        return {"ok": True}

    first = Scheduler(executor=executor, state_root=tmp_path, check_interval=0.05)
    first.add_task(task(schedule=TaskSchedule(type=ScheduleType.ONE_SHOT)))
    second = Scheduler(executor=executor, state_root=tmp_path, check_interval=0.05)
    await asyncio.gather(first.run_once(), second.run_once())
    assert calls == 1
    assert first.summary()["tasks"][0]["run_count"] == 1


@pytest.mark.asyncio
async def test_generation_rolls_before_lowered_limit(tmp_path):
    async def executor(_):
        return {"ok": True}

    scheduler = Scheduler(executor=executor, state_root=tmp_path, generation_limit=3)
    scheduler.add_task(task(schedule=TaskSchedule(interval_seconds=1)))
    await scheduler.run_once()
    for _ in range(2):
        await asyncio.sleep(1.05)
        await scheduler.run_once()
    assert len(scheduler._mission_store.snapshot().missions) == 2


def test_disable_remove_and_unreconciled_removal(tmp_path):
    scheduler = Scheduler(state_root=tmp_path)
    scheduler.add_task(task())
    assert scheduler.disable_task("durable-task")
    assert scheduler.list_tasks()[0].enabled is False
    assert scheduler.enable_task("durable-task")
    scheduler._catalog.claim_due(1_800_000_000)
    with pytest.raises(CatalogError, match="unreconciled"):
        scheduler.remove_task("durable-task")


def test_catalog_rejects_symlink_and_unknown_schema(tmp_path):
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(CatalogError, match="symlink"):
        Scheduler(state_root=alias).list_tasks()

    scheduler = Scheduler(state_root=tmp_path / "state")
    scheduler.add_task(task())
    with sqlite3.connect(scheduler._catalog.db) as conn:
        conn.execute("PRAGMA user_version=99")
    with pytest.raises(CatalogError, match="schema"):
        scheduler.list_tasks()


def test_catalog_bootstrap_reopen_and_private_modes(tmp_path):
    state = tmp_path / "new" / "state"
    first = Scheduler(state_root=state)
    first.add_task(task())
    assert Scheduler(state_root=state).list_tasks()[0].task_id == "durable-task"
    assert stat.S_IMODE(state.stat().st_mode) == 0o700
    assert stat.S_IMODE((state / "scheduler-catalog").stat().st_mode) == 0o700
    assert stat.S_IMODE(first._catalog.db.stat().st_mode) == 0o600


@pytest.mark.parametrize("content", [b"", b"SQLite format 3\x00", b"not a sqlite catalog"])
def test_existing_invalid_catalog_is_never_bootstrapped_or_rewritten(tmp_path, content):
    state = tmp_path / "state"
    catalog_dir = state / "scheduler-catalog"
    catalog_dir.mkdir(mode=0o700, parents=True)
    os.chmod(state, 0o700)
    database = catalog_dir / "catalog.sqlite3"
    database.write_bytes(content)
    os.chmod(database, 0o600)

    before = database.read_bytes()
    with pytest.raises(CatalogError):
        ScheduleCatalog(state).rows()
    assert database.read_bytes() == before


def test_existing_failed_integrity_catalog_is_byte_identical(tmp_path):
    state = tmp_path / "state"
    catalog = ScheduleCatalog(state)
    catalog.put(task())
    damaged = bytearray(catalog.db.read_bytes())
    page_size = int.from_bytes(damaged[16:18], "big") or 65_536
    damaged[page_size] = 0  # invalid b-tree page type on the first schema-owned page
    catalog.db.write_bytes(damaged)

    before = catalog.db.read_bytes()
    with pytest.raises(CatalogError):
        ScheduleCatalog(state).rows()
    assert catalog.db.read_bytes() == before


def test_existing_foreign_key_invalid_catalog_is_byte_identical(tmp_path):
    state = tmp_path / "state"
    catalog = ScheduleCatalog(state)
    catalog.put(task())
    with sqlite3.connect(catalog.db) as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "INSERT INTO occurrences VALUES('missing-task',1,'claimed',NULL,NULL,NULL,NULL)"
        )

    before = catalog.db.read_bytes()
    with pytest.raises(CatalogError, match="foreign key"):
        ScheduleCatalog(state).rows()
    assert catalog.db.read_bytes() == before


def test_catalog_rejects_symlinked_ancestor_and_final_file(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    with pytest.raises(CatalogError, match="symlink"):
        Scheduler(state_root=tmp_path / "linked" / "state").list_tasks()

    state = tmp_path / "state"
    catalog_dir = state / "scheduler-catalog"
    catalog_dir.mkdir(mode=0o700, parents=True)
    os.chmod(state, 0o700)
    target = tmp_path / "attacker.sqlite3"
    target.write_bytes(b"not sqlite")
    os.chmod(target, 0o600)
    (catalog_dir / "catalog.sqlite3").symlink_to(target)
    with pytest.raises(CatalogError, match="unsafe"):
        Scheduler(state_root=state).list_tasks()


def test_catalog_rejects_directory_displacement(tmp_path):
    state = tmp_path / "state"
    scheduler = Scheduler(state_root=state)
    scheduler.add_task(task())
    original = state / "scheduler-catalog"
    original.rename(state / "old-catalog")
    original.mkdir(mode=0o700)
    with pytest.raises(CatalogError, match="displaced"):
        scheduler.list_tasks()


def test_catalog_rejects_configured_root_displacement(tmp_path):
    state = tmp_path / "state"
    scheduler = Scheduler(state_root=state)
    scheduler.add_task(task())
    state.rename(tmp_path / "old-state")
    state.mkdir(mode=0o700)
    with pytest.raises(CatalogError, match="configured root.*displaced"):
        scheduler.list_tasks()


def test_catalog_read_rejects_catalog_directory_displacement(tmp_path):
    state = tmp_path / "state"
    catalog = ScheduleCatalog(state)
    catalog.put(task())
    catalog_dir = state / "scheduler-catalog"

    with pytest.raises(CatalogError, match="displaced"):
        with catalog.transaction():
            catalog_dir.rename(state / "detached-catalog")
            catalog_dir.mkdir(mode=0o700)


def test_catalog_read_rejects_final_file_displacement(tmp_path):
    state = tmp_path / "state"
    catalog = ScheduleCatalog(state)
    catalog.put(task())
    attacker = tmp_path / "attacker.sqlite3"
    attacker.write_bytes(b"attacker bytes")
    os.chmod(attacker, 0o600)

    with pytest.raises(CatalogError, match="displaced"):
        with catalog.transaction():
            catalog.db.rename(tmp_path / "original.sqlite3")
            attacker.rename(catalog.db)


def test_catalog_publish_rechecks_configured_binding(tmp_path, monkeypatch):
    state = tmp_path / "state"
    catalog = ScheduleCatalog(state)
    catalog.put(task())
    catalog_dir = state / "scheduler-catalog"
    publish = catalog._publish

    def publish_then_displace(root_fd, data):
        identity = publish(root_fd, data)
        catalog_dir.rename(state / "detached-catalog")
        catalog_dir.mkdir(mode=0o700)
        return identity

    monkeypatch.setattr(catalog, "_publish", publish_then_displace)
    with pytest.raises(CatalogError, match="displaced"):
        catalog.enabled("durable-task", False)


def test_catalog_read_does_not_repair_unsafe_mode(tmp_path):
    scheduler = Scheduler(state_root=tmp_path / "state")
    scheduler.add_task(task())
    os.chmod(scheduler._catalog.db, 0o644)
    with pytest.raises(CatalogError, match="owner-only"):
        scheduler.list_tasks()
    assert stat.S_IMODE(scheduler._catalog.db.stat().st_mode) == 0o644


@pytest.mark.parametrize("statement", [
    "CREATE TABLE foreign_table(value TEXT)",
    "CREATE INDEX foreign_index ON tasks(enabled)",
    "CREATE VIEW foreign_view AS SELECT task_id FROM tasks",
    "CREATE TRIGGER foreign_trigger AFTER UPDATE ON tasks BEGIN SELECT 1; END",
])
def test_catalog_rejects_unknown_schema_objects(tmp_path, statement):
    scheduler = Scheduler(state_root=tmp_path / statement.split()[1])
    scheduler.add_task(task())
    with sqlite3.connect(scheduler._catalog.db) as conn:
        conn.execute(statement)
    with pytest.raises(CatalogError, match="schema"):
        scheduler.list_tasks()


def test_catalog_rejects_unknown_schema_field(tmp_path):
    scheduler = Scheduler(state_root=tmp_path / "state")
    scheduler.add_task(task())
    with sqlite3.connect(scheduler._catalog.db) as conn:
        conn.execute("ALTER TABLE tasks ADD COLUMN foreign_field TEXT")
    with pytest.raises(CatalogError, match="fields"):
        scheduler.list_tasks()


def test_catalog_oversize_refusal_does_not_publish_mutation(tmp_path, monkeypatch):
    import flyto_ai.scheduler.catalog as catalog_module

    scheduler = Scheduler(state_root=tmp_path / "state")
    scheduler.add_task(task())
    original = scheduler._catalog.db.read_bytes()
    monkeypatch.setattr(catalog_module, "MAX_CATALOG_BYTES", len(original) + 1024)
    with pytest.raises(CatalogError, match="exceeds"):
        with scheduler._catalog.transaction() as conn:
            conn.execute("UPDATE tasks SET definition=?", ("x" * 100_000,))
    assert scheduler._catalog.db.read_bytes() == original
    assert Scheduler(state_root=tmp_path / "state").list_tasks()[0].task_id == "durable-task"


def test_catalog_two_process_serialization(tmp_path):
    state = tmp_path / "state"
    context = multiprocessing.get_context("fork")
    start = context.Event()
    processes = [
        context.Process(target=_put_catalog_task, args=(state, f"task-{number}", start))
        for number in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    assert [row["task_id"] for row in ScheduleCatalog(state).rows()] == ["task-0", "task-1"]


@pytest.mark.parametrize("bad", [
    {"name": "n", "instruction": "i", "run_count": 1},
    {"name": "n", "instruction": "i", "enabled": 1},
    {"name": "n", "instruction": "i", "budget_usd": float("nan")},
    {"name": "n", "instruction": "i", "tags": ["x", "x"]},
    {"name": "", "instruction": "i"},
    {"name": " \t", "instruction": "i"},
    {"name": "n", "instruction": "\n  "},
    {"name": "n", "instruction": "i", "schedule": {"type": "interval", "interval_seconds": 0}},
    {"name": "n", "instruction": "i", "schedule": {"type": "cron", "cron": "* * * * *", "run_at": 1}},
])
def test_forged_or_invalid_definition_rejected(bad):
    with pytest.raises(ValueError):
        ScheduledTask.from_dict(bad)


def test_strict_utc_cron_vectors_and_rejections():
    cron = TaskSchedule(type=ScheduleType.CRON, cron_expression="*/15 9-10 * * 1-5")
    monday = datetime(2026, 8, 10, 9, 7, tzinfo=timezone.utc).timestamp()
    assert cron.next_slot(0, now=monday) == datetime(2026, 8, 10, 9, 15, tzinfo=timezone.utc).timestamp()
    for expression in ("@daily", "* * * *", "61 * * * *", "* * * * MON", "*/0 * * * *"):
        with pytest.raises(ValueError):
            TaskSchedule(type=ScheduleType.CRON, cron_expression=expression)


def test_cron_dom_dow_or_wildcard_and_sunday_aliases():
    restricted = TaskSchedule(type=ScheduleType.CRON, cron_expression="0 8 13 * 1")
    sunday = TaskSchedule(type=ScheduleType.CRON, cron_expression="0 8 * * 0,7")
    full_step = TaskSchedule(type=ScheduleType.CRON, cron_expression="0 8 */1 * 0")
    restricted_step = TaskSchedule(type=ScheduleType.CRON, cron_expression="0 8 */2 * 0")
    anchor = datetime(2026, 8, 11, 9, tzinfo=timezone.utc).timestamp()
    assert restricted.next_slot(0, now=anchor) == datetime(
        2026, 8, 13, 8, tzinfo=timezone.utc
    ).timestamp()
    assert sunday.next_slot(0, now=anchor) == datetime(
        2026, 8, 16, 8, tzinfo=timezone.utc
    ).timestamp()
    assert full_step.next_slot(0, now=anchor) == datetime(
        2026, 8, 16, 8, tzinfo=timezone.utc
    ).timestamp()
    assert restricted_step.next_slot(0, now=anchor) == datetime(
        2026, 8, 13, 8, tzinfo=timezone.utc
    ).timestamp()


def test_cron_range_step_and_unsatisfiable_schedule_fail_closed():
    stepped = TaskSchedule(type=ScheduleType.CRON, cron_expression="10-20/5 8 * * *")
    anchor = datetime(2026, 8, 13, 8, 11, tzinfo=timezone.utc).timestamp()
    assert stepped.next_slot(0, now=anchor) == datetime(
        2026, 8, 13, 8, 15, tzinfo=timezone.utc
    ).timestamp()
    with pytest.raises(ValueError, match="unsatisfiable"):
        TaskSchedule(type=ScheduleType.CRON, cron_expression="0 0 31 2 *")
    leap_day = TaskSchedule(type=ScheduleType.CRON, cron_expression="0 0 29 2 *")
    assert leap_day.next_slot(0, now=anchor) == datetime(
        2028, 2, 29, tzinfo=timezone.utc
    ).timestamp()


def test_immediate_one_shot_is_resolved_once_at_catalog_insert(tmp_path):
    catalog = ScheduleCatalog(tmp_path)
    catalog.put(task(schedule=TaskSchedule(type=ScheduleType.ONE_SHOT)), now=1_700_000_000.75)
    stored = catalog.decode_task(catalog.row("durable-task"))
    assert stored.schedule.run_at == 1_700_000_000.75
    assert catalog.claim_due(1_700_000_001) == [(stored, 1_700_000_000)]
    assert catalog.claim_due(1_800_000_000)[0][1] == 1_700_000_000


def test_interval_and_cron_catch_up_claim_only_one_latest_pass_slot(tmp_path):
    catalog = ScheduleCatalog(tmp_path)
    catalog.put(task(task_id="interval", schedule=TaskSchedule(interval_seconds=10)))
    catalog.put(task(task_id="cron", schedule=TaskSchedule(
        type=ScheduleType.CRON, cron_expression="* * * * *"
    )))
    first = catalog.claim_due(960)
    assert [(item.task_id, slot) for item, slot in first] == [
        ("cron", 960), ("interval", 960)
    ]
    claimed = catalog.claim_due(10_000)
    new_slots = [(item.task_id, slot) for item, slot in claimed]
    assert new_slots.count(("interval", 10_000)) == 1
    assert new_slots.count(("cron", 1_020)) == 1
    assert float(catalog.row("interval")["cursor"]) == 10_000
    assert float(catalog.row("cron")["cursor"]) == 10_000


def test_ephemeral_summary_is_truthful():
    assert Scheduler().summary()["guarantees"] == "ephemeral_process_local"
