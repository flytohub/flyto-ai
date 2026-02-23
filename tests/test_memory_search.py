# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for memory search — embeddings, BM25, and hybrid RRF."""
import os

import pytest
import pytest_asyncio
import aiosqlite

from flyto_ai.memory.bm25 import BM25Index
from flyto_ai.memory.embeddings import EmbeddingStore, cosine_similarity, _pack_vector, _unpack_vector
from flyto_ai.memory.search import MemorySearch, _rrf_merge

_has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))


# ── Unit tests (no API needed) ────────────────────────────────


class TestCosine:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) + 1.0) < 1e-6

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert cosine_similarity(a, b) == 0.0


class TestPackUnpack:
    def test_round_trip(self):
        vec = [1.5, -2.3, 0.0, 4.2]
        blob = _pack_vector(vec)
        result = _unpack_vector(blob)
        for a, b in zip(vec, result):
            assert abs(a - b) < 1e-6


class TestRRFMerge:
    def test_basic_merge(self):
        vec = [("a", "text a", 0.9), ("b", "text b", 0.8)]
        bm25 = [("b", "text b", 5.0), ("c", "text c", 3.0)]
        merged = _rrf_merge(vec, bm25)
        # "b" appears in both → highest RRF score
        assert merged[0][0] == "b"

    def test_single_source(self):
        vec = [("a", "text a", 0.9)]
        merged = _rrf_merge(vec, [])
        assert len(merged) == 1
        assert merged[0][0] == "a"


# ── BM25 tests (real SQLite, no API) ──────────────────────────

@pytest_asyncio.fixture
async def bm25_db(tmp_path):
    db_path = str(tmp_path / "bm25_test.db")
    db = await aiosqlite.connect(db_path)
    idx = BM25Index(db)
    await idx.init()
    yield idx, db
    await db.close()


@pytest.mark.asyncio
async def test_bm25_keyword_search(bm25_db):
    """BM25 ranks keyword-matching documents correctly."""
    idx, _ = bm25_db
    await idx.add_document("d1", "Python is a programming language", session_id="s1")
    await idx.add_document("d2", "JavaScript runs in the browser", session_id="s1")
    await idx.add_document("d3", "Python web frameworks like Django and Flask", session_id="s1")

    results = await idx.search("Python programming")
    assert len(results) >= 2
    # Documents mentioning Python should rank higher
    doc_ids = [r[0] for r in results]
    assert "d1" in doc_ids[:2]
    assert "d3" in doc_ids[:2]


@pytest.mark.asyncio
async def test_bm25_empty_query(bm25_db):
    """Empty query returns no results."""
    idx, _ = bm25_db
    await idx.add_document("d1", "test document")
    results = await idx.search("")
    assert results == []


@pytest.mark.asyncio
async def test_bm25_no_match(bm25_db):
    """Query with no matching terms returns empty."""
    idx, _ = bm25_db
    await idx.add_document("d1", "hello world")
    results = await idx.search("xyzzy foobar")
    assert results == []


@pytest.mark.asyncio
async def test_bm25_count(bm25_db):
    """Count returns number of indexed documents."""
    idx, _ = bm25_db
    assert await idx.count() == 0
    await idx.add_document("d1", "test one")
    await idx.add_document("d2", "test two")
    assert await idx.count() == 2


# ── Embedding tests (need OPENAI_API_KEY) ──────────────────────

@pytest_asyncio.fixture
async def embedding_db(tmp_path):
    db_path = str(tmp_path / "embed_test.db")
    db = await aiosqlite.connect(db_path)
    store = EmbeddingStore(db)
    await store.init()
    yield store, db
    await db.close()


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_openai_key, reason="OPENAI_API_KEY not set")
async def test_embed_and_search(embedding_db):
    """Store embeddings and find the most similar one."""
    store, _ = embedding_db

    texts = [
        "How to cook pasta with tomato sauce",
        "Python web development with Django",
        "Machine learning with neural networks",
        "Italian cuisine and Mediterranean diet",
        "Deep learning for image classification",
    ]
    for i, text in enumerate(texts):
        await store.add("s1", text)

    # Search for cooking-related content
    query_vec = await store.embed_text("cooking Italian food")
    results = await store.search(query_vec, top_k=2)

    assert len(results) >= 2
    # Cooking/Italian related docs should rank highest
    top_contents = [r[1] for r in results]
    assert any("pasta" in c.lower() or "italian" in c.lower() for c in top_contents)


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_openai_key, reason="OPENAI_API_KEY not set")
async def test_embedding_count(embedding_db):
    """Count tracks stored embeddings."""
    store, _ = embedding_db
    assert await store.count() == 0
    await store.add("s1", "test content")
    assert await store.count() == 1


# ── Hybrid search tests ──────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.skipif(not _has_openai_key, reason="OPENAI_API_KEY not set")
async def test_hybrid_rrf(tmp_path):
    """Hybrid search combines vector and BM25 for better results."""
    db_path = str(tmp_path / "hybrid_test.db")
    db = await aiosqlite.connect(db_path)

    emb_store = EmbeddingStore(db)
    await emb_store.init()
    bm25_idx = BM25Index(db)
    await bm25_idx.init()
    search = MemorySearch(emb_store, bm25_idx)

    # Index diverse content
    docs = [
        ("s1", "The weather in Tokyo is sunny today"),
        ("s1", "Python is great for data science and machine learning"),
        ("s1", "I ordered sushi for lunch at the Tokyo restaurant"),
        ("s1", "Installing Python packages with pip"),
        ("s1", "The Tokyo Olympics were held in 2021"),
    ]
    for i, (sid, text) in enumerate(docs):
        await search.index_content(sid, text, doc_id="d{}".format(i))

    results = await search.search("Tokyo weather", top_k=3)
    assert len(results) > 0
    # Tokyo weather doc should be in top results
    top_contents = [r["content"] for r in results]
    assert any("weather" in c.lower() and "tokyo" in c.lower() for c in top_contents)

    await db.close()


@pytest.mark.asyncio
async def test_bm25_only_search(tmp_path):
    """Search works with BM25 only (no API key needed)."""
    db_path = str(tmp_path / "bm25_only.db")
    db = await aiosqlite.connect(db_path)

    # EmbeddingStore that will fail on embed_text (no API key)
    emb_store = EmbeddingStore(db)
    await emb_store.init()
    bm25_idx = BM25Index(db)
    await bm25_idx.init()

    # Index via BM25 only
    await bm25_idx.add_document("d1", "Python web development", session_id="s1")
    await bm25_idx.add_document("d2", "JavaScript frontend framework", session_id="s1")

    search = MemorySearch(emb_store, bm25_idx)
    results = await search.search("Python development", top_k=2)
    # Even without embeddings, BM25 should return results
    assert len(results) >= 1
    assert any("python" in r["content"].lower() for r in results)

    await db.close()
