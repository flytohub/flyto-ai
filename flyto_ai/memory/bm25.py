# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Pure Python BM25 index backed by SQLite."""
import logging
import math
import re
from typing import Dict, List, Tuple

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS bm25_terms (
    term TEXT,
    doc_id TEXT,
    tf REAL,
    PRIMARY KEY (term, doc_id)
);
CREATE TABLE IF NOT EXISTS bm25_docs (
    doc_id TEXT PRIMARY KEY,
    content TEXT,
    session_id TEXT,
    length INTEGER
);
"""

# BM25 parameters
_K1 = 1.2
_B = 0.75

# Simple tokenizer — split on non-alphanumeric, lowercase
_TOKEN_RE = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase terms."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """BM25 full-text search backed by SQLite."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def init(self) -> None:
        """Create BM25 tables."""
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def add_document(self, doc_id: str, text: str, session_id: str = "") -> None:
        """Index a document."""
        tokens = _tokenize(text)
        if not tokens:
            return

        doc_len = len(tokens)

        # Count term frequencies
        tf_counts: Dict[str, int] = {}
        for t in tokens:
            tf_counts[t] = tf_counts.get(t, 0) + 1

        # Store doc
        await self._db.execute(
            "INSERT OR REPLACE INTO bm25_docs (doc_id, content, session_id, length) VALUES (?, ?, ?, ?)",
            (doc_id, text, session_id, doc_len),
        )

        # Store term frequencies (normalized)
        for term, count in tf_counts.items():
            tf = count / doc_len
            await self._db.execute(
                "INSERT OR REPLACE INTO bm25_terms (term, doc_id, tf) VALUES (?, ?, ?)",
                (term, doc_id, tf),
            )
        await self._db.commit()

    async def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Search documents by query. Returns list of (doc_id, content, score)."""
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        # Get corpus stats
        cursor = await self._db.execute("SELECT COUNT(*), AVG(length) FROM bm25_docs")
        row = await cursor.fetchone()
        n_docs = row[0] if row[0] else 0
        avg_dl = row[1] if row[1] else 1.0

        if n_docs == 0:
            return []

        # Score each document
        scores: Dict[str, float] = {}

        for term in set(query_terms):
            # Get document frequency
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM bm25_terms WHERE term = ?", (term,)
            )
            df = (await cursor.fetchone())[0]
            if df == 0:
                continue

            # IDF
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

            # Get all docs with this term
            cursor = await self._db.execute(
                "SELECT t.doc_id, t.tf, d.length FROM bm25_terms t "
                "JOIN bm25_docs d ON t.doc_id = d.doc_id "
                "WHERE t.term = ?",
                (term,),
            )
            for doc_id, tf, doc_len in await cursor.fetchall():
                # BM25 score
                tf_raw = tf * doc_len  # un-normalize
                numerator = tf_raw * (_K1 + 1)
                denominator = tf_raw + _K1 * (1 - _B + _B * doc_len / avg_dl)
                score = idf * numerator / denominator
                scores[doc_id] = scores.get(doc_id, 0) + score

        if not scores:
            return []

        # Fetch content and sort
        results = []
        for doc_id, score in sorted(scores.items(), key=lambda x: -x[1])[:top_k]:
            cursor = await self._db.execute(
                "SELECT content FROM bm25_docs WHERE doc_id = ?", (doc_id,)
            )
            row = await cursor.fetchone()
            content = row[0] if row else ""
            results.append((doc_id, content, score))

        return results

    async def count(self) -> int:
        """Return total number of indexed documents."""
        cursor = await self._db.execute("SELECT COUNT(*) FROM bm25_docs")
        row = await cursor.fetchone()
        return row[0] if row else 0
