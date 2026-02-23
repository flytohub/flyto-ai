# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Embedding store — OpenAI embeddings stored as SQLite BLOBs."""
import logging
import math
import struct
import time
from typing import List, Optional, Tuple

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    content TEXT,
    embedding BLOB,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_session ON embeddings(session_id);
"""

# Default dimension for text-embedding-3-small
_DEFAULT_DIM = 1536


def _pack_vector(vec: List[float]) -> bytes:
    """Pack a float vector into bytes."""
    return struct.pack("{}f".format(len(vec)), *vec)


def _unpack_vector(blob: bytes) -> List[float]:
    """Unpack bytes into a float vector."""
    n = len(blob) // 4
    return list(struct.unpack("{}f".format(n), blob))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure Python cosine similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingStore:
    """Store and search embeddings using SQLite + cosine similarity."""

    def __init__(self, db: aiosqlite.Connection, model: str = "text-embedding-3-small") -> None:
        self._db = db
        self._model = model
        self._client = None

    async def init(self) -> None:
        """Create embeddings table."""
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    def _get_client(self):
        """Lazily create OpenAI client."""
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI()
        return self._client

    async def embed_text(self, text: str) -> List[float]:
        """Get embedding vector for text via OpenAI API."""
        client = self._get_client()
        response = await client.embeddings.create(
            model=self._model,
            input=text,
        )
        return response.data[0].embedding

    async def add(self, session_id: str, content: str, embedding: Optional[List[float]] = None) -> None:
        """Add content with its embedding. Computes embedding if not provided."""
        if embedding is None:
            embedding = await self.embed_text(content)
        blob = _pack_vector(embedding)
        now = time.time()
        await self._db.execute(
            "INSERT INTO embeddings (session_id, content, embedding, created_at) VALUES (?, ?, ?, ?)",
            (session_id, content, blob, now),
        )
        await self._db.commit()

    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search for most similar content. Returns list of (session_id, content, score)."""
        cursor = await self._db.execute(
            "SELECT session_id, content, embedding FROM embeddings"
        )
        rows = await cursor.fetchall()

        scored = []
        for session_id, content, blob in rows:
            vec = _unpack_vector(blob)
            score = cosine_similarity(query_embedding, vec)
            scored.append((session_id, content, score))

        scored.sort(key=lambda x: -x[2])
        return scored[:top_k]

    async def count(self) -> int:
        """Return total number of stored embeddings."""
        cursor = await self._db.execute("SELECT COUNT(*) FROM embeddings")
        row = await cursor.fetchone()
        return row[0] if row else 0
