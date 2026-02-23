# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for SessionStore — TTL, limits, cleanup, locks."""
import time

from flyto_ai.models import ChatMessage
from flyto_ai.session import SessionStore


class TestSessionStore:

    def test_create_and_get(self):
        store = SessionStore()
        store.create("s1", "user1")
        assert store.exists("s1")
        assert store.get_owner("s1") == "user1"

    def test_append_and_get_messages(self):
        store = SessionStore()
        store.create("s1", "user1")
        store.append("s1", ChatMessage(role="user", content="hello"))
        msgs = store.get_messages("s1")
        assert len(msgs) == 1
        assert msgs[0].content == "hello"

    def test_delete(self):
        store = SessionStore()
        store.create("s1", "user1")
        assert store.delete("s1", "user1") is True
        assert store.exists("s1") is False

    def test_delete_wrong_user(self):
        store = SessionStore()
        store.create("s1", "user1")
        assert store.delete("s1", "user2") is False
        assert store.exists("s1") is True

    def test_per_user_limit(self):
        store = SessionStore(max_per_user=2)
        store.create("s1", "user1")
        store.create("s2", "user1")
        store.create("s3", "user1")  # should evict s1
        assert not store.exists("s1")
        assert store.exists("s2")
        assert store.exists("s3")

    def test_max_messages_trimmed(self):
        store = SessionStore()
        store.create("s1", "user1")
        for i in range(25):
            store.append("s1", ChatMessage(role="user", content="msg {}".format(i)))
        assert len(store.get_messages("s1")) == 20

    def test_cleanup_expired(self):
        store = SessionStore(ttl_seconds=0)
        store.create("s1", "user1")
        store.append("s1", ChatMessage(role="user", content="hi"))
        time.sleep(0.01)
        store.cleanup()
        assert not store.exists("s1")

    def test_get_lock(self):
        store = SessionStore()
        lock1 = store.get_lock("s1")
        lock2 = store.get_lock("s1")
        assert lock1 is lock2  # same lock for same session

    def test_nonexistent_session(self):
        store = SessionStore()
        assert store.get_messages("nope") == []
        assert store.get_owner("nope") is None
