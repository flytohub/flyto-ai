# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Async scheduler with an optional durable MissionStore-governed path."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Callable, Coroutine, Dict, List, Mapping, Optional

from flyto_ai.orchestration.mission_control import (
    Closure, DISPOSITION_BLOCKED, DISPOSITION_FIXED, MAX_WORK_ITEMS_PER_MISSION,
    MissionConflict, MissionResource, MissionStore, STATUS_CLOSED, STATUS_DISPATCHED,
    WorkCoordinates,
)
from flyto_ai.scheduler.catalog import CatalogError, ScheduleCatalog
from flyto_ai.scheduler.tasks import (
    MAX_BUDGET_USD, MAX_RESULT_TEXT, ScheduledTask, ScheduleType, TaskResult, TaskState,
)

TaskExecutor = Callable[[str], Coroutine[Any, Any, Dict[str, Any]]]
_RESULT_KEYS = {"ok", "message", "error", "cost_usd"}
_CONTROL_FLOW_EXCEPTIONS = (asyncio.CancelledError, KeyboardInterrupt, SystemExit)


def _op(kind: str, *parts: object) -> str:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:32]
    return f"sched-{kind}-{digest}"


def _evidence(task_id: str, slot: int, ok: bool) -> str:
    return "sched-result-" + hashlib.sha256(f"{task_id}\0{slot}\0{int(ok)}".encode()).hexdigest()[:32]


def _execution_id(task_id: str, slot: int) -> str:
    """Return the stable identity for one externally executed occurrence."""
    return _op("execution", task_id, slot)


def _unknown_evidence(task_id: str, slot: int) -> str:
    return "sched-unknown-" + hashlib.sha256(_execution_id(task_id, slot).encode()).hexdigest()[:32]


class Scheduler:
    """Ephemeral scheduler or fail-closed, at-most-once durable adapter.

    A durable execution whose persisted fence proves that executor entry may
    already have happened is never replayed. Recovery closes it as
    ``execution_outcome_unknown``; downstream idempotency is still required
    before exact recovery can be offered.
    """

    def __init__(self, executor: Optional[TaskExecutor] = None, check_interval: float = 10.0,
                 state_root: Optional[os.PathLike[str] | str] = None,
                 *, generation_limit: int = MAX_WORK_ITEMS_PER_MISSION) -> None:
        if isinstance(check_interval, bool) or not isinstance(check_interval, (int, float)) \
                or not math.isfinite(check_interval) or check_interval <= 0:
            raise ValueError("check_interval must be finite and positive")
        if isinstance(generation_limit, bool) or not isinstance(generation_limit, int) \
                or generation_limit < 2 or generation_limit > MAX_WORK_ITEMS_PER_MISSION:
            raise ValueError("generation_limit is invalid")
        self._executor = executor
        self._check_interval = float(check_interval)
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._loop_task: Optional[asyncio.Task[None]] = None
        self._catalog = ScheduleCatalog(state_root) if state_root is not None else None
        self._mission_store = MissionStore(self._catalog.state_root / "scheduler-missions") if self._catalog else None
        self._generation_limit = generation_limit
        self._worker = _op("worker", self._catalog.state_root if self._catalog else os.getpid())
        self._loop_error_code: Optional[str] = None

    @property
    def durable(self) -> bool:
        return self._catalog is not None

    @property
    def task_count(self) -> int:
        return len(self.list_tasks()) if self.durable else len(self._tasks)

    @property
    def running(self) -> bool:
        return self._running

    def add_task(self, task: ScheduledTask) -> str:
        if not isinstance(task, ScheduledTask):
            raise ValueError("task must be a ScheduledTask")
        # Re-parse the canonical boundary so callers cannot smuggle malformed mutable fields.
        clean = ScheduledTask.from_dict(task.to_definition())
        if self._catalog:
            if clean.schedule.type == ScheduleType.INTERVAL and clean.schedule.interval_seconds == 0:
                raise ValueError("durable interval_seconds must be positive")
            self._catalog.put(clean)
        else:
            self._tasks[clean.task_id] = task
        return clean.task_id

    def remove_task(self, task_id: str) -> bool:
        if self._catalog:
            return self._catalog.remove(task_id)
        return self._tasks.pop(task_id, None) is not None

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        if self._catalog:
            row = self._catalog.row(task_id)
            return self._catalog.decode_task(row) if row else None
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[ScheduledTask]:
        if self._catalog:
            return [self._catalog.decode_task(row) for row in self._catalog.rows()]
        return list(self._tasks.values())

    def enable_task(self, task_id: str) -> bool:
        if self._catalog:
            return self._catalog.enabled(task_id, True)
        task = self._tasks.get(task_id)
        if task:
            task.enabled, task.state = True, TaskState.PENDING
            return True
        return False

    def disable_task(self, task_id: str) -> bool:
        if self._catalog:
            return self._catalog.enabled(task_id, False)
        task = self._tasks.get(task_id)
        if task:
            task.enabled, task.state = False, TaskState.DISABLED
            return True
        return False

    async def start(self) -> None:
        if not self._running:
            self._loop_error_code = None
            self._running = True
            self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self.run_once()
            except Exception:
                # Never expose exception text or keep retrying an internal durable failure.
                self._loop_error_code = "scheduler_durable_system_failure"
                self._running = False
                return
            try:
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break

    async def run_once(self) -> List[TaskResult]:
        if not self._catalog:
            results = []
            for task in list(self._tasks.values()):
                if task.is_due:
                    results.append(await self._execute_ephemeral(task))
            return results
        assert self._mission_store is not None
        occurrences = self._catalog.claim_due(time.time())
        results: List[TaskResult] = []
        for task, slot in occurrences:
            self._materialize(task, slot)
            results.extend(await self._drain_dispatch())
        return results

    def _ensure_generation(self, task: ScheduledTask) -> tuple[str, str]:
        assert self._catalog and self._mission_store
        with self._catalog.transaction() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task.task_id,)).fetchone()
            if row is None:
                raise CatalogError("scheduled task disappeared")
            if row["mission_id"] and int(row["item_count"]) < self._generation_limit:
                generation = int(row["generation"])
                root = self._mission_store.get_work_item(str(row["root_id"]))
                self._mission_store.acknowledge_operation(
                    _op("mission-create", task.task_id, generation)
                )
                self._mission_store.acknowledge_operation(
                    _op("anchor-submit", task.task_id, generation)
                )
                if root.status == STATUS_CLOSED:
                    self._mission_store.acknowledge_operation(
                        _op("anchor-close", root.work_item_id, root.fence)
                    )
                    self._mission_store.acknowledge_operation(
                        _op("dispatch", root.work_item_id, root.attempts)
                    )
                return str(row["mission_id"]), str(row["root_id"])
            old_mission = row["mission_id"]
            generation = int(row["generation"]) + 1
        if old_mission:
            complete_op = _op("mission-complete", task.task_id, generation - 1)
            self._mission_store.complete_mission(str(old_mission), {"container": _op("generation", task.task_id, generation - 1)}, operation=complete_op)
            self._mission_store.acknowledge_operation(complete_op)
        create_op = _op("mission-create", task.task_id, generation)
        mission = self._mission_store.create_mission(
            operation=create_op, scope=_op("scope", task.task_id),
            objective="Maintain one bounded scheduled-task occurrence container.",
            desired_result="Every admitted occurrence is explicitly closed.",
            acceptance_criteria=(("container", "The generation container is closed truthfully."),),
        )
        root_op = _op("anchor-submit", task.task_id, generation)
        root = self._mission_store.submit_work_item(
            mission.mission_id, operation=root_op,
            coordinates=WorkCoordinates("scheduler", "catalog", _op("anchor", task.task_id, generation)),
            root=True,
        )
        with self._catalog.transaction() as conn:
            conn.execute("UPDATE tasks SET generation=?,mission_id=?,root_id=?,item_count=1 WHERE task_id=?",
                         (generation, mission.mission_id, root.work_item_id, task.task_id))
        self._mission_store.acknowledge_operation(create_op)
        self._mission_store.acknowledge_operation(root_op)
        return mission.mission_id, root.work_item_id

    def _materialize(self, task: ScheduledTask, slot: int) -> None:
        assert self._catalog and self._mission_store
        with self._catalog.transaction() as conn:
            occurrence = conn.execute("SELECT * FROM occurrences WHERE task_id=? AND slot=?", (task.task_id, slot)).fetchone()
            if occurrence is None or occurrence["state"] == "closed":
                return
            if occurrence["work_item_id"]:
                item = self._mission_store.get_work_item(str(occurrence["work_item_id"]))
                self._mission_store.acknowledge_operation(
                    _op("occurrence-submit", task.task_id, slot)
                )
                if item.status == STATUS_CLOSED:
                    ok = bool(item.closure and item.closure.disposition == DISPOSITION_FIXED)
                    public = {"slot": slot, "ok": ok, "cost_usd": 0.0, "message": "",
                              "error": None if ok else "reconciled_failure",
                              "evidence_ref": _evidence(task.task_id, slot, ok)}
                    conn.execute("UPDATE occurrences SET state='closed',result=? WHERE task_id=? AND slot=?",
                                 (json.dumps(public, sort_keys=True, separators=(",", ":")), task.task_id, slot))
                    self._mission_store.acknowledge_operation(
                        _op("occurrence-close", task.task_id, slot, item.fence)
                    )
                    self._mission_store.acknowledge_operation(
                        _op("dispatch", item.work_item_id, item.attempts)
                    )
                return
        mission_id, root_id = self._ensure_generation(task)
        submit_op = _op("occurrence-submit", task.task_id, slot)
        item = self._mission_store.submit_work_item(
            mission_id, operation=submit_op,
            coordinates=WorkCoordinates("scheduler", "catalog", _op("slot", task.task_id, slot)),
            resources=(MissionResource("scheduler", "task", task.task_id),),
            parent_id=root_id, return_to_id=root_id,
        )
        with self._catalog.transaction() as conn:
            changed = conn.execute(
                "UPDATE occurrences SET state='materialized',mission_id=?,work_item_id=? "
                "WHERE task_id=? AND slot=? AND work_item_id IS NULL",
                (mission_id, item.work_item_id, task.task_id, slot),
            ).rowcount
            if changed:
                conn.execute("UPDATE tasks SET item_count=item_count+1 WHERE task_id=?", (task.task_id,))
        self._mission_store.acknowledge_operation(submit_op)

    async def _drain_dispatch(self) -> List[TaskResult]:
        assert self._catalog and self._mission_store
        results: List[TaskResult] = []
        # One bounded pass; new arrivals wait for the next scheduler tick.
        for _ in range(max(2, self.task_count * 2 + 2)):
            recovery: Optional[str] = None
            with self._catalog.transaction() as conn:
                ids = [row[0] for row in conn.execute(
                    "SELECT root_id FROM tasks WHERE root_id IS NOT NULL UNION "
                    "SELECT work_item_id FROM occurrences WHERE state!='closed' AND work_item_id IS NOT NULL"
                )]
            for item_id in ids:
                if self._mission_store.get_work_item(str(item_id)).status == "dispatched":
                    recovery = str(item_id)
                    break
            order = self._mission_store.scheduler_order(limit=1)
            candidate = recovery or (order[0] if order else None)
            if candidate is None:
                break
            candidate_item = self._mission_store.get_work_item(candidate)
            attempt_era = candidate_item.attempts + int(candidate_item.status != "dispatched")
            dispatch_op = _op("dispatch", candidate, attempt_era)
            try:
                dispatch_context = self._mission_store.dispatch_expected(
                    operation=dispatch_op,
                    worker=self._worker,
                    work_item_id=candidate,
                    expected_attempt=attempt_era,
                )
                handle_context = dispatch_context.__enter__()
            except MissionConflict:
                continue
            control_flow_exc: Optional[BaseException] = None
            try:
                handle = handle_context
                if handle is None:
                    break
                occurrence = None
                with self._catalog.transaction() as conn:
                    occurrence = conn.execute("SELECT * FROM occurrences WHERE work_item_id=?", (handle.work_item_id,)).fetchone()
                    if occurrence:
                        prior_fence = occurrence["fence"]
                        conn.execute("UPDATE occurrences SET fence=? WHERE task_id=? AND slot=?",
                                     (handle.fence, occurrence["task_id"], occurrence["slot"]))
                if occurrence is None:  # fixed internal anchor; never calls the executor
                    anchor_close = _op("anchor-close", handle.work_item_id, handle.fence)
                    handle.close(Closure(DISPOSITION_FIXED), operation=anchor_close)
                    self._mission_store.acknowledge_operation(dispatch_op)
                    self._mission_store.acknowledge_operation(anchor_close)
                    continue
                task = self.get_task(str(occurrence["task_id"]))
                if task is None:
                    raise CatalogError("occurrence has no scheduled definition")
                slot = int(occurrence["slot"])
                if prior_fence is not None:
                    result, close_op = self._close_unknown(task, slot, handle)
                else:
                    result, close_op = await self._execute_with_handle(task, slot, handle)
                results.append(result)
                public = {"slot": slot, "ok": result.ok,
                          "cost_usd": result.cost_usd, "message": result.message,
                          "error": result.error,
                          "evidence_ref": (_unknown_evidence(task.task_id, slot)
                                           if result.error == "execution_outcome_unknown"
                                           else _evidence(task.task_id, slot, result.ok))}
                with self._catalog.transaction() as conn:
                    conn.execute("UPDATE occurrences SET state='closed',result=? WHERE task_id=? AND slot=?",
                                 (json.dumps(public, sort_keys=True, separators=(",", ":")), task.task_id, occurrence["slot"]))
                    if task.schedule.type == ScheduleType.ONE_SHOT:
                        conn.execute("UPDATE tasks SET enabled=0 WHERE task_id=?", (task.task_id,))
                self._mission_store.acknowledge_operation(dispatch_op)
                self._mission_store.acknowledge_operation(close_op)
            except _CONTROL_FLOW_EXCEPTIONS as exc:
                control_flow_exc = exc
                raise
            finally:
                cleanup_exc: Optional[BaseException] = None
                try:
                    dispatch_context.__exit__(*sys.exc_info())
                except BaseException as exc:
                    cleanup_exc = exc
                try:
                    if handle_context is not None:
                        settled = self._mission_store.get_work_item(
                            handle_context.work_item_id
                        )
                        if settled.status != STATUS_DISPATCHED:
                            self._mission_store.acknowledge_operation(dispatch_op)
                except BaseException as exc:
                    if cleanup_exc is None:
                        cleanup_exc = exc
                if cleanup_exc is not None and control_flow_exc is None:
                    raise cleanup_exc
        return results

    async def _execute_with_handle(self, task: ScheduledTask, slot: int, handle: Any) -> tuple[TaskResult, str]:
        started = time.monotonic()
        execution = asyncio.create_task(self._invoke_executor(task))
        pulse = asyncio.create_task(self._heartbeat(handle))
        try:
            done, _ = await asyncio.wait(
                (execution, pulse), return_when=asyncio.FIRST_COMPLETED
            )
            if pulse in done and not execution.done():
                # A heartbeat supervisor is not meant to finish while the
                # executor is live.  Whether it returned or raised, authority
                # over the in-flight side effect is now unknowable.
                pulse.exception()
                execution.cancel()
                try:
                    await execution
                except asyncio.CancelledError:
                    pass
                except Exception:
                    # Cleanup cannot make an already-unknown external outcome
                    # authoritative, and its details are not public evidence.
                    pass
                return self._close_unknown(task, slot, handle, started=started)
            raw = await execution
            ok, message, error, cost = self._validate_output(raw)
            if task.budget_usd is not None and cost > task.budget_usd:
                ok, error = False, "budget_exceeded"
        except BaseException as exc:
            if isinstance(exc, _CONTROL_FLOW_EXCEPTIONS):
                raise
            ok, message, error, cost = False, "", "executor_failed", 0.0
        finally:
            pulse.cancel()
            try:
                await pulse
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            execution.cancel()
            try:
                await execution
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        duration = min(int((time.monotonic() - started) * 1000), 2**31 - 1)
        result = TaskResult(task_id=task.task_id, timestamp=float(slot), ok=ok, message=message,
                            duration_ms=duration, cost_usd=cost, error=error)
        close_op = _op("occurrence-close", task.task_id, slot, handle.fence)
        if ok:
            closure = Closure(DISPOSITION_FIXED, evidence_refs=(_evidence(task.task_id, slot, True),))
        else:
            closure = Closure(DISPOSITION_BLOCKED, rationale="Scheduled execution did not satisfy policy.",
                              risk="The scheduled objective remains unsatisfied.",
                              evidence_refs=(_evidence(task.task_id, slot, False),), owner="scheduler",
                              revisit_at=int(time.time()) + 3600)
        handle.close(closure, operation=close_op)
        return result, close_op

    async def _invoke_executor(self, task: ScheduledTask) -> Dict[str, Any]:
        if self._executor is None:
            raise RuntimeError("executor_unavailable")
        return await self._executor(task.instruction)

    def _close_unknown(self, task: ScheduledTask, slot: int, handle: Any, *,
                       started: Optional[float] = None) -> tuple[TaskResult, str]:
        """Close an occurrence whose executor outcome cannot be proven."""
        duration = 0 if started is None else min(
            int((time.monotonic() - started) * 1000), 2**31 - 1
        )
        result = TaskResult(
            task_id=task.task_id,
            timestamp=float(slot),
            ok=False,
            message="",
            duration_ms=duration,
            cost_usd=0.0,
            error="execution_outcome_unknown",
        )
        close_op = _op("occurrence-close", task.task_id, slot, handle.fence)
        handle.close(
            Closure(
                DISPOSITION_BLOCKED,
                rationale="The durable execution boundary was reached without a provable outcome.",
                risk="The scheduled objective may be incomplete.",
                evidence_refs=(_unknown_evidence(task.task_id, slot),),
                owner="scheduler",
                revisit_at=int(time.time()) + 3600,
            ),
            operation=close_op,
        )
        return result, close_op

    async def _heartbeat(self, handle: Any) -> None:
        while True:
            await asyncio.sleep(min(self._check_interval, 0.25))
            handle.heartbeat()

    @staticmethod
    def _validate_output(raw: Any) -> tuple[bool, str, Optional[str], float]:
        if not isinstance(raw, Mapping) or type(raw) is not dict or set(raw) - _RESULT_KEYS or "ok" not in raw:
            raise ValueError("executor output is not the exact result mapping")
        if type(raw["ok"]) is not bool:
            raise ValueError("executor ok must be an exact bool")
        cost = raw.get("cost_usd", 0.0)
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) \
                or cost < 0 or cost > MAX_BUDGET_USD:
            raise ValueError("executor cost is invalid")
        message, error = raw.get("message", ""), raw.get("error")
        if not isinstance(message, str) or len(message) > MAX_RESULT_TEXT:
            raise ValueError("executor message is invalid")
        if error is not None and (not isinstance(error, str) or len(error) > MAX_RESULT_TEXT):
            raise ValueError("executor error is invalid")
        return raw["ok"], message, error, float(cost)

    async def _execute_ephemeral(self, task: ScheduledTask) -> TaskResult:
        task.state = TaskState.RUNNING
        started = time.monotonic()
        try:
            if self._executor is None:
                raise RuntimeError("No executor configured")
            raw = await self._executor(task.instruction)
            # Preserve legacy defaults only in explicitly non-durable mode.
            ok, cost = raw.get("ok", True), raw.get("cost_usd", 0.0)
            result = TaskResult(task.task_id, ok=ok, message=str(raw.get("message", ""))[:500],
                                duration_ms=int((time.monotonic() - started) * 1000), cost_usd=cost,
                                error=raw.get("error"))
        except Exception as exc:
            result = TaskResult(task.task_id, ok=False,
                                duration_ms=int((time.monotonic() - started) * 1000), error=str(exc)[:500])
        task.state = TaskState.COMPLETED if result.ok else TaskState.FAILED
        task.record_result(result)
        if task.schedule.type == ScheduleType.ONE_SHOT and result.ok:
            task.enabled, task.state = False, TaskState.DISABLED
        return result

    def summary(self) -> Dict[str, Any]:
        tasks = self.list_tasks()
        projected = []
        for task in tasks:
            history = self._catalog.public_results(task.task_id) if self._catalog else []
            projected.append({"task_id": task.task_id, "name": task.name,
                              "state": (TaskState.PENDING if task.enabled else TaskState.DISABLED).value if self.durable else task.state.value,
                              "enabled": task.enabled,
                              "run_count": len(history) if self.durable else task.run_count,
                              "success_rate": (round(sum(int(item["ok"]) for item in history) / len(history), 2) if history else 0.0) if self.durable else round(task.success_rate, 2),
                              "total_cost_usd": round(sum(item["cost_usd"] for item in history), 4) if self.durable else round(task.total_cost_usd, 4),
                              "results": history if self.durable else []})
        return {"running": self._running, "durable": self.durable,
                "guarantees": "mission_store_governed" if self.durable else "ephemeral_process_local",
                "error": self._loop_error_code,
                "task_count": len(tasks), "tasks": projected}
