# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Memory embeddings must use the operator's provider config and fail quietly.

What the owner met: the chat ran fine on the configured provider key, but the
sidecar log showed ``POST https://api.openai.com/v1/embeddings 401`` after every
turn, because the embedding client was built bare and read a stale
OPENAI_API_KEY from the desktop environment. Memory never wrote and the 401
repeated on every turn.
"""
import logging
from types import SimpleNamespace

import aiosqlite
import openai
import pytest
import pytest_asyncio

from flyto_ai.agent import Agent
from flyto_ai.config import AgentConfig
from flyto_ai.memory import embeddings as embeddings_module
from flyto_ai.memory.bm25 import BM25Index
from flyto_ai.memory.embeddings import EmbeddingStore
from flyto_ai.memory.search import MemorySearch


class _FakeAsyncOpenAI:
    """Stand-in for openai.AsyncOpenAI that records how it was built."""

    constructed = []
    create_calls = []
    fail_with = None

    def __init__(self, **kwargs):
        type(self).constructed.append(kwargs)
        self.embeddings = self

    async def create(self, model, input):
        type(self).create_calls.append((model, input))
        if type(self).fail_with is not None:
            raise type(self).fail_with
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


def _unauthorized():
    # openai's status error reads three things off the response; a plain
    # object carrying them keeps this test free of httpx, which the CI's test
    # environment does not install.
    response = SimpleNamespace(request=None, status_code=401, headers={})
    return openai.AuthenticationError("Incorrect API key provided", response=response, body=None)


@pytest.fixture
def fake_openai(monkeypatch):
    _FakeAsyncOpenAI.constructed = []
    _FakeAsyncOpenAI.create_calls = []
    _FakeAsyncOpenAI.fail_with = None
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeAsyncOpenAI)
    # The once-per-process guard is real state; each test starts a fresh "process".
    monkeypatch.setattr(embeddings_module, "_unavailable_warned", False, raising=False)
    return _FakeAsyncOpenAI


@pytest_asyncio.fixture
async def memory_db(tmp_path):
    db = await aiosqlite.connect(str(tmp_path / "memory.db"))
    yield db
    await db.close()


def _warnings(caplog):
    return [
        r for r in caplog.records
        if r.name == "flyto_ai.memory.embeddings" and r.levelno >= logging.WARNING
    ]


@pytest.mark.asyncio
async def test_embedding_client_is_built_from_configured_provider(fake_openai, tmp_path):
    """The chat used the configured key; embeddings used a bare client and got 401."""
    config = AgentConfig(
        provider="openai",
        api_key="sk-configured" "-by-operator",
        base_url="https://api.openai.com/v1",
        memory_db_path=str(tmp_path / "agent-memory.db"),
        enable_memory=True,
        enable_deterministic=False,
    )
    agent = Agent(config=config)
    await agent._init_memory()

    store = agent.memory_search._embedding
    await store.embed_text("hello")

    assert fake_openai.constructed == [{
        "api_key": "sk-configured" "-by-operator",
        "base_url": "https://api.openai.com/v1",
    }]
    await agent._memory_store._db.close()


@pytest.mark.asyncio
async def test_embeddings_401_is_warned_once_and_chat_continues(fake_openai, memory_db, caplog):
    """A 401 from embeddings appeared in the log on every turn; memory never wrote."""
    fake_openai.fail_with = _unauthorized()
    store = EmbeddingStore(memory_db, api_key="sk-revoked")
    await store.init()
    bm25 = BM25Index(memory_db)
    await bm25.init()
    search = MemorySearch(store, bm25)

    caplog.set_level(logging.DEBUG)
    # Two chat turns: each one searches memory before the reply and indexes after it.
    for turn in range(2):
        results = await search.search("deploy the robot", top_k=3)
        await search.index_content("s1", "User: deploy the robot\nAssistant: turn {}".format(turn))

    assert len(_warnings(caplog)) == 1
    # After the first 401 nothing is sent again — the key is not going to change mid-process.
    assert len(fake_openai.create_calls) == 1
    # Keyword memory still works, so the chat is unchanged.
    assert any("robot" in r["content"] for r in results)
    assert await bm25.count() == 2


@pytest.mark.asyncio
async def test_embeddings_without_any_key_warn_once(monkeypatch, memory_db, caplog):
    """No configured key and no OPENAI_API_KEY in the environment: one warning, chat proceeds."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(embeddings_module, "_unavailable_warned", False, raising=False)
    store = EmbeddingStore(memory_db)
    await store.init()
    bm25 = BM25Index(memory_db)
    await bm25.init()
    search = MemorySearch(store, bm25)

    caplog.set_level(logging.DEBUG)
    for turn in range(2):
        await search.search("anything", top_k=3)
        await search.index_content("s1", "User: anything\nAssistant: turn {}".format(turn))

    assert len(_warnings(caplog)) == 1
    assert await bm25.count() == 2
