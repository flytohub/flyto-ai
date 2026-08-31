# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Bounded, snapshot-consistent MissionStore lookup helpers.

The durable store is an atomically published SQLite image. Opening it performs
whole-store validation by design, so callers that need several known rows must
read them from one validated snapshot rather than reopening that image once per
row. This module adds no execution or mutation authority.
"""
from __future__ import annotations

from typing import Dict, Sequence

from flyto_ai.orchestration.mission_control import (
    MissionError,
    MissionRejected,
    MissionStore,
    WorkItem,
    _check_token,
)


MAX_BATCH_WORK_ITEM_LOOKUPS = 4096
_SQL_LOOKUP_CHUNK = 256


def read_work_items(
    store: MissionStore,
    work_item_ids: Sequence[str],
) -> Dict[str, WorkItem]:
    """Read known work items from one fully validated durable snapshot.

    Missing rows are absent from the result. Invalid, duplicate, or oversized
    input is rejected before the store is opened, so a caller never receives a
    partial answer from a malformed candidate set.
    """

    if isinstance(work_item_ids, (str, bytes)):
        raise MissionRejected("work item ids must be a bounded sequence")
    identifiers = tuple(work_item_ids)
    if len(identifiers) > MAX_BATCH_WORK_ITEM_LOOKUPS:
        raise MissionRejected("too many work item ids for one snapshot lookup")
    for value in identifiers:
        _check_token(value, "work item id")
    if len(set(identifiers)) != len(identifiers):
        raise MissionRejected("work item ids must be unique")
    if not identifiers:
        return {}

    with store._read() as txn:
        conn = None if txn is None else txn.conn
        if conn is None:
            return {}
        result: Dict[str, WorkItem] = {}
        for offset in range(0, len(identifiers), _SQL_LOOKUP_CHUNK):
            batch = identifiers[offset : offset + _SQL_LOOKUP_CHUNK]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                "SELECT * FROM work_items WHERE work_item_id IN ({})".format(
                    placeholders,
                ),
                batch,
            ).fetchall()
            for row in rows:
                item = store._item_from_row(conn, row)
                result[item.work_item_id] = item
        return result


def read_work_items_fail_closed(
    store: MissionStore,
    work_item_ids: Sequence[str],
) -> Dict[str, WorkItem]:
    """Return no reconciliation candidates when the snapshot is untrustworthy."""

    try:
        return read_work_items(store, work_item_ids)
    except MissionError:
        return {}
