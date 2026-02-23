# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for SQLiteSessionStore — real SQLite, using tmp_path."""
import asyncio
import os

import pytest
import pytest_asyncio

from flyto_ai.memory.sqlite_store import SQLiteSessionStore


@pytest_asyncio.fixture
async def store(tmp_path):
    """Create a SQLiteSessionStore backed by a temp file."""
    db_path = str(tmp_path / "test_memory.db")
    s = SQLiteSessionStore(db_path=db_path)
    await s.init()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_add_and_get_messages(store):
    """Write messages then read back — verifies basic round-trip."""
    await store.add_message("s1", "user", "hello")
    await store.add_message("s1", "assistant", "hi there")

    msgs = await store.get_messages("s1")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hello"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "hi there"


@pytest.mark.asyncio
async def test_session_auto_created(store):
    """add_message auto-creates session if it doesn't exist."""
    assert not await store.session_exists("auto")
    await store.add_message("auto", "user", "test")
    assert await store.session_exists("auto")


@pytest.mark.asyncio
async def test_session_persistence(tmp_path):
    """Close store, reopen — data survives."""
    db_path = str(tmp_path / "persist.db")

    # Write
    s1 = SQLiteSessionStore(db_path=db_path)
    await s1.init()
    await s1.add_message("s1", "user", "remember me")
    await s1.close()

    # Reopen
    s2 = SQLiteSessionStore(db_path=db_path)
    await s2.init()
    msgs = await s2.get_messages("s1")
    assert len(msgs) == 1
    assert msgs[0]["content"] == "remember me"
    await s2.close()


@pytest.mark.asyncio
async def test_list_sessions(store):
    """Multiple sessions appear in list."""
    await store.add_message("s1", "user", "a")
    await store.add_message("s2", "user", "b")
    await store.add_message("s2", "user", "c")

    sessions = await store.list_sessions()
    assert len(sessions) == 2
    ids = {s["session_id"] for s in sessions}
    assert ids == {"s1", "s2"}
    # s2 has 2 messages
    s2_info = next(s for s in sessions if s["session_id"] == "s2")
    assert s2_info["message_count"] == 2


@pytest.mark.asyncio
async def test_delete_session(store):
    """Delete removes session and all messages."""
    await store.add_message("s1", "user", "hello")
    assert await store.delete_session("s1") is True
    assert not await store.session_exists("s1")
    assert await store.get_messages("s1") == []


@pytest.mark.asyncio
async def test_delete_nonexistent(store):
    """Deleting non-existent session returns False."""
    assert await store.delete_session("nope") is False


@pytest.mark.asyncio
async def test_concurrent_writes(store):
    """Concurrent writes to same session don't crash."""
    async def write(i):
        await store.add_message("s1", "user", "msg {}".format(i))

    await asyncio.gather(*[write(i) for i in range(20)])
    msgs = await store.get_messages("s1")
    assert len(msgs) == 20


@pytest.mark.asyncio
async def test_message_metadata(store):
    """Metadata dict round-trips through JSON."""
    meta = {"tool_calls": [{"name": "search", "args": {}}]}
    await store.add_message("s1", "assistant", "result", metadata=meta)
    msgs = await store.get_messages("s1")
    assert msgs[0]["metadata"] == meta


@pytest.mark.asyncio
async def test_get_messages_with_limit(store):
    """Limit parameter caps returned messages."""
    for i in range(10):
        await store.add_message("s1", "user", "msg {}".format(i))
    msgs = await store.get_messages("s1", limit=3)
    assert len(msgs) == 3
    assert msgs[0]["content"] == "msg 0"


@pytest.mark.asyncio
async def test_get_message_count(store):
    """get_message_count returns accurate count."""
    assert await store.get_message_count("s1") == 0
    await store.add_message("s1", "user", "a")
    await store.add_message("s1", "user", "b")
    assert await store.get_message_count("s1") == 2


@pytest.mark.asyncio
async def test_replace_old_with_summary(store):
    """Summarization replaces old messages, keeps recent ones."""
    for i in range(15):
        await store.add_message("s1", "user", "msg {}".format(i))

    await store.replace_old_with_summary("s1", "Summary of msgs 0-4", keep_recent=10)

    msgs = await store.get_messages("s1")
    assert len(msgs) == 10
    # Oldest remaining should be msg 5
    assert msgs[0]["content"] == "msg 5"

    summaries = await store.get_summaries("s1")
    assert len(summaries) == 1
    assert summaries[0]["content"] == "Summary of msgs 0-4"
    assert summaries[0]["source_count"] == 5
