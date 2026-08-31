import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from flyto_ai.coding.contracts import (
    MISSION_LANE_PRIMARY,
    MISSION_OPEN,
    MISSION_STATUS_READY,
    CodingMissionProjection,
)
from flyto_ai.coding.mission_runtime import CodingMissionRuntime
from flyto_ai.coding.startup_reconciliation import reconcile_interrupted_jobs
from flyto_ai.orchestration import mission_control


def _projection(index: int) -> CodingMissionProjection:
    return CodingMissionProjection(
        mission_id="m-{:012d}".format(index),
        scope="scope-{:04d}".format(index),
        work_item_id="w-{:012d}".format(index),
        main_axis_sha256="a" * 64,
        criteria_ids=("criterion",),
        lane=MISSION_LANE_PRIMARY,
        priority=1,
        status=MISSION_STATUS_READY,
        mission_status=MISSION_OPEN,
    )


def test_terminal_history_uses_one_bounded_mission_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tenant = "b" * 64
    jobs = tmp_path / "tenants" / tenant / "jobs"
    jobs.mkdir(parents=True)
    before = {}
    for index in range(100):
        path = jobs / "job_{:024x}.json".format(index)
        path.write_text(json.dumps({
            "job_id": path.stem,
            "state": "failed",
            "mission": _projection(index).to_mapping(),
        }), encoding="utf-8")
        before[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    observed = []

    def batch(_store, identifiers):
        observed.append(tuple(identifiers))
        return {}

    monkeypatch.setattr(
        "flyto_ai.coding.startup_reconciliation.read_work_items_fail_closed",
        batch,
    )
    service = SimpleNamespace(
        state_root=tmp_path,
        _mission=SimpleNamespace(store=object()),
        _reclaimed=0,
        _read_json=lambda path: json.loads(path.read_text(encoding="utf-8")),
        _record_projection=lambda record: CodingMissionProjection.from_mapping(
            record["mission"],
        ),
        _sweep_workspace_claims=lambda: None,
        _reconcile_continuation_claims=lambda: None,
        _prime_pump=lambda: None,
    )

    reconcile_interrupted_jobs(service)

    assert len(observed) == 1
    assert len(observed[0]) == 100
    assert set(observed[0]) == {
        "w-{:012d}".format(index) for index in range(100)
    }
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in jobs.iterdir()
    }
    assert after == before


@pytest.mark.skipif(
    not mission_control.inspect_host().supported,
    reason="startup batch integration requires durable MissionStore primitives",
)
def test_real_ready_item_is_reconciled_once_with_exact_coordinates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tenant = "c" * 64
    job_id = "job_" + "d" * 24
    runtime = CodingMissionRuntime(tmp_path, worker="startup-test")
    placed = runtime.admit(
        tenant_ref=tenant,
        job_id=job_id,
        workspace_sha256="e" * 64,
        envelope=None,
        message="reconcile one terminal ready item",
    )
    jobs = tmp_path / "tenants" / tenant / "jobs"
    jobs.mkdir(parents=True)
    exact_path = jobs / (job_id + ".json")
    exact_path.write_text(json.dumps({
        "job_id": job_id,
        "state": "failed",
        "mission": placed.projection.to_mapping(),
    }), encoding="utf-8")
    wrong_jobs = tmp_path / "tenants" / ("f" * 64) / "jobs"
    wrong_jobs.mkdir(parents=True)
    wrong_path = wrong_jobs / (job_id + ".json")
    wrong_path.write_bytes(exact_path.read_bytes())

    reads = 0
    original_read = runtime.store._read

    @contextmanager
    def counted_read():
        nonlocal reads
        reads += 1
        with original_read() as transaction:
            yield transaction

    monkeypatch.setattr(runtime.store, "_read", counted_read)
    updated = []
    pumped = []

    def update(path, **changes):
        record = json.loads(path.read_text(encoding="utf-8"))
        record.update(changes)
        path.write_text(json.dumps(record), encoding="utf-8")
        updated.append(path)

    service = SimpleNamespace(
        state_root=tmp_path,
        _mission=runtime,
        _reclaimed=0,
        _read_json=lambda path: json.loads(path.read_text(encoding="utf-8")),
        _record_projection=lambda record: CodingMissionProjection.from_mapping(
            record["mission"],
        ),
        _update_record_locked=update,
        _sweep_workspace_claims=lambda: None,
        _reconcile_continuation_claims=lambda: None,
        _prime_pump=lambda: pumped.append(True),
    )

    reconcile_interrupted_jobs(service)

    assert reads == 1
    assert updated == [exact_path]
    assert pumped == [True]
    assert json.loads(wrong_path.read_text(encoding="utf-8"))["mission"] == (
        placed.projection.to_mapping()
    )
