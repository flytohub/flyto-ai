# Copyright 2024 Flyto2
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

# Embeddings are optional: memory falls back to keyword search without them.
# The owner's desktop had no usable OPENAI_API_KEY and saw a 401 logged after
# every single chat turn, so the "embeddings are off" warning is said once per
# process, not once per turn.
_unavailable_warned = False


class EmbeddingsUnavailable(RuntimeError):
    """Raised on every embed call after the store has given up for this process."""


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

    def __init__(
        self,
        db: aiosqlite.Connection,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._db = db
        self._model = model
        # The key and base_url the chat itself is configured with. The client
        # used to be built bare and read OPENAI_API_KEY from the process
        # environment, which on the owner's desktop was absent while the chat
        # ran happily on the operator's configured key — every turn then ended
        # in a 401 from /v1/embeddings. The environment is now only the
        # fallback for an unconfigured key.
        self._api_key = api_key
        self._base_url = base_url
        self._client = None
        self._disabled_reason: Optional[str] = None

    async def init(self) -> None:
        """Create embeddings table."""
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    def _get_client(self):
        """Lazily create OpenAI client from the configured provider settings."""
        if self._client is None:
            import openai
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    def _mark_unavailable(self, exc: Exception) -> None:
        """Degrade to keyword-only memory: warn once per process, stop retrying
        failures that cannot fix themselves (no key at all, key rejected)."""
        global _unavailable_warned
        import openai
        permanent = isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)) or (
            # openai raises the plain base error when no key was found anywhere.
            isinstance(exc, openai.OpenAIError) and not isinstance(exc, openai.APIError)
        )
        if permanent:
            self._disabled_reason = str(exc)
        if not _unavailable_warned:
            _unavailable_warned = True
            logger.warning(
                "Memory embeddings unavailable (%s); memory continues with keyword search only",
                exc,
            )
        else:
            logger.debug("Memory embeddings unavailable: %s", exc)

    async def embed_text(self, text: str) -> List[float]:
        """Get embedding vector for text via OpenAI API."""
        if self._disabled_reason:
            raise EmbeddingsUnavailable(self._disabled_reason)
        try:
            client = self._get_client()
            response = await client.embeddings.create(
                model=self._model,
                input=text,
            )
        except Exception as e:
            self._mark_unavailable(e)
            raise
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
