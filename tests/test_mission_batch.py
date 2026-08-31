from contextlib import contextmanager

import pytest

from flyto_ai.orchestration import mission_control
from flyto_ai.orchestration.mission_batch import (
    MAX_BATCH_WORK_ITEM_LOOKUPS,
    read_work_items,
    read_work_items_fail_closed,
)
from flyto_ai.orchestration.mission_control import (
    MissionRejected,
    MissionStore,
    WorkCoordinates,
)


pytestmark = pytest.mark.skipif(
    not mission_control.inspect_host().supported,
    reason="mission batch lookup requires the durable MissionStore host primitives",
)


def _item(store: MissionStore, index: int):
    mission = store.create_mission(
        operation=f"mission-{index}",
        scope=f"scope-{index}",
        objective="read one exact set from one validated snapshot",
        desired_result="every requested row is projected once",
        acceptance_criteria=(("criterion", "the row remains exact"),),
    )
    return store.submit_work_item(
        mission.mission_id,
        operation=f"item-{index}",
        root=True,
        coordinates=WorkCoordinates(
            project=f"project-{index}",
            repository=f"repository-{index}",
            location=f"location-{index}",
        ),
    )


def test_batch_lookup_validates_the_store_once(tmp_path, monkeypatch) -> None:
    store = MissionStore(tmp_path)
    expected = tuple(_item(store, index) for index in range(12))
    reads = 0
    original = store._read

    @contextmanager
    def counted_read():
        nonlocal reads
        reads += 1
        with original() as transaction:
            yield transaction

    monkeypatch.setattr(store, "_read", counted_read)
    observed = read_work_items(
        store,
        tuple(item.work_item_id for item in reversed(expected)) + ("work-missing",),
    )

    assert reads == 1
    assert set(observed) == {item.work_item_id for item in expected}
    assert all(observed[item.work_item_id] == item for item in expected)


def test_batch_lookup_rejects_ambiguous_input_before_opening_the_store(
    tmp_path,
    monkeypatch,
) -> None:
    store = MissionStore(tmp_path)
    opened = False

    @contextmanager
    def must_not_open():
        nonlocal opened
        opened = True
        yield None

    monkeypatch.setattr(store, "_read", must_not_open)
    with pytest.raises(MissionRejected):
        read_work_items(store, ("work-duplicate", "work-duplicate"))
    with pytest.raises(MissionRejected):
        read_work_items(store, tuple(
            f"work-{index}" for index in range(MAX_BATCH_WORK_ITEM_LOOKUPS + 1)
        ))
    assert read_work_items_fail_closed(store, ("bad id",)) == {}
    assert opened is False
