# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Hybrid memory search — vector + BM25 with Reciprocal Rank Fusion."""
import hashlib
import logging
from typing import Dict, List, Optional, Tuple

from flyto_ai.memory.bm25 import BM25Index
from flyto_ai.memory.embeddings import EmbeddingStore

logger = logging.getLogger(__name__)


def _rrf_merge(
    vector_results: List[Tuple[str, str, float]],
    bm25_results: List[Tuple[str, str, float]],
    k: int = 60,
) -> List[Tuple[str, str, float]]:
    """Reciprocal Rank Fusion — merge two ranked lists.

    Returns list of (doc_id, content, rrf_score) sorted by score desc.
    """
    scores: Dict[str, float] = {}
    contents: Dict[str, str] = {}

    for rank, (doc_id, content, _) in enumerate(vector_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        contents[doc_id] = content

    for rank, (doc_id, content, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        contents[doc_id] = content

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [(doc_id, contents[doc_id], score) for doc_id, score in ranked]


class MemorySearch:
    """Hybrid search combining vector similarity and BM25 keyword matching."""

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        bm25_index: BM25Index,
    ) -> None:
        self._embedding = embedding_store
        self._bm25 = bm25_index

    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search memory using hybrid vector + BM25 approach.

        Returns list of dicts with keys: session_id, content, score, source.
        """
        results = []

        # Try vector search
        vector_results = []
        try:
            query_vec = await self._embedding.embed_text(query)
            vector_results = await self._embedding.search(query_vec, top_k=top_k * 2)
        except Exception as e:
            logger.debug("Vector search failed: %s", e)

        # BM25 search (always works, no API needed)
        bm25_results = []
        try:
            bm25_results = await self._bm25.search(query, top_k=top_k * 2)
        except Exception as e:
            logger.debug("BM25 search failed: %s", e)

        if not vector_results and not bm25_results:
            return []

        # If only one source available, use it directly
        if vector_results and not bm25_results:
            merged = vector_results[:top_k]
        elif bm25_results and not vector_results:
            merged = bm25_results[:top_k]
        else:
            # Hybrid RRF merge
            merged = _rrf_merge(vector_results, bm25_results)[:top_k]

        for doc_id, content, score in merged:
            results.append({
                "session_id": doc_id.split(":")[0] if ":" in doc_id else doc_id,
                "content": content,
                "score": score,
            })

        return results

    async def index_content(self, session_id: str, content: str, doc_id: Optional[str] = None) -> None:
        """Index content in both vector and BM25 stores."""
        if not content or not content.strip():
            return
        _doc_id = doc_id or "{}:{}".format(session_id, hashlib.sha256(content.encode()).hexdigest()[:16])

        # BM25 — always index (no API needed)
        try:
            await self._bm25.add_document(_doc_id, content, session_id=session_id)
        except Exception as e:
            logger.debug("BM25 indexing failed: %s", e)

        # Vector — may fail if no API key
        try:
            await self._embedding.add(session_id, content)
        except Exception as e:
            logger.debug("Embedding indexing failed: %s", e)
