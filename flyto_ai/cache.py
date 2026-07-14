# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Prompt cache fingerprinting — track cache efficiency for LLM API calls.

Inspired by claw-code's ``PromptCache`` with FNV-1a fingerprinting, TTL-based
hit prediction, and unexpected cache-break alerting.

This is a *client-side* tracker; the actual cache lives server-side (e.g.
Anthropic prompt caching).  We fingerprint the system prompt + tool definitions
to predict whether the server should have a cache hit, then compare against
actual ``cache_creation_input_tokens`` / ``cache_read_input_tokens`` from
the API response.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# FNV-1a constants (64-bit)
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x00000100000001B3
_FNV_MASK = 0xFFFFFFFFFFFFFFFF


def fnv1a_64(data: str) -> int:
    """FNV-1a 64-bit hash of a string."""
    h = _FNV_OFFSET
    for byte in data.encode("utf-8"):
        h ^= byte
        h = (h * _FNV_PRIME) & _FNV_MASK
    return h


@dataclass
class CacheStats:
    """Aggregate cache statistics for a session."""
    total_calls: int = 0
    expected_hits: int = 0
    actual_hits: int = 0
    unexpected_breaks: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0

    def hit_rate(self) -> float:
        """Actual cache hit rate (0.0–1.0)."""
        if self.total_calls == 0:
            return 0.0
        return self.actual_hits / self.total_calls

    def summary(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "expected_hits": self.expected_hits,
            "actual_hits": self.actual_hits,
            "unexpected_breaks": self.unexpected_breaks,
            "hit_rate": round(self.hit_rate(), 3),
            "cache_creation_tokens": self.total_cache_creation_tokens,
            "cache_read_tokens": self.total_cache_read_tokens,
        }


class PromptCache:
    """Client-side prompt cache fingerprint tracker.

    Usage::

        cache = PromptCache(ttl_seconds=30.0)

        # Before LLM call:
        cache.update_fingerprint(system_prompt, tools)
        expect_hit = cache.should_expect_cache_hit()

        # After LLM call:
        cache.record_usage(
            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
        )
    """

    def __init__(self, ttl_seconds: float = 30.0) -> None:
        self._ttl = ttl_seconds
        self._fingerprint: int = 0
        self._last_update: float = 0.0
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def fingerprint(self) -> int:
        return self._fingerprint

    def update_fingerprint(self, system_prompt: str, tools: List[Dict]) -> int:
        """Compute and store fingerprint of system prompt + tools.

        Always updates the TTL timer so repeated identical prompts
        continue to expect cache hits.  Returns the new fingerprint value.
        """
        # Stable serialization: sort tool names for determinism
        tool_names = sorted(t.get("name", "") for t in tools)
        payload = system_prompt + "|" + ",".join(tool_names)
        new_fp = fnv1a_64(payload)

        # Always refresh TTL — even when fingerprint is unchanged
        self._last_update = time.monotonic()
        self._fingerprint = new_fp

        return new_fp

    def should_expect_cache_hit(self) -> bool:
        """Whether a cache hit is expected (prompt unchanged within TTL)."""
        if self._fingerprint == 0:
            return False
        elapsed = time.monotonic() - self._last_update
        return elapsed <= self._ttl

    def record_usage(
        self,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> None:
        """Record actual cache usage from provider response and detect anomalies."""
        self._stats.total_calls += 1
        self._stats.total_cache_creation_tokens += cache_creation_tokens
        self._stats.total_cache_read_tokens += cache_read_tokens

        expected = self.should_expect_cache_hit()
        got_hit = cache_read_tokens > 0

        if expected:
            self._stats.expected_hits += 1

        if got_hit:
            self._stats.actual_hits += 1

        if expected and not got_hit and cache_creation_tokens > 0:
            self._stats.unexpected_breaks += 1
            logger.debug(
                "Unexpected cache break: expected hit but got %d creation tokens "
                "(fingerprint=%016x)",
                cache_creation_tokens, self._fingerprint,
            )

    def reset(self) -> None:
        """Reset fingerprint and stats."""
        self._fingerprint = 0
        self._last_update = 0.0
        self._stats = CacheStats()
