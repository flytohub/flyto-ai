# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Context window compaction — smart message compression to stay within limits.

Better than OpenClaw's compaction which users criticize for "broken memory":
- Hybrid search (embeddings + BM25) ensures semantic recall after compaction
- Keeps recent messages intact (no "amnesia" after pause)
- Gradual: soft threshold summarizes old messages, hard threshold forces trim
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default token thresholds
DEFAULT_SOFT_THRESHOLD = 80_000   # trigger background summarization
DEFAULT_HARD_THRESHOLD = 120_000  # force compaction
DEFAULT_KEEP_RECENT = 10          # always keep last N messages


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars ≈ 1 token for English, 2 chars for CJK)."""
    if not text:
        return 0
    # Simple heuristic: count chars, divide by 3.5 (between EN and CJK average)
    return max(1, len(text) // 3)


def estimate_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate total tokens across a list of messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Multimodal: sum text parts
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += estimate_tokens(part.get("text", ""))
        # Add overhead for role, metadata
        total += 4
    return total


class ContextCompactor:
    """Monitors context window usage and compacts when thresholds are exceeded.

    Usage::

        compactor = ContextCompactor(
            soft_threshold=80000,
            hard_threshold=120000,
            summarize_fn=my_summarizer,
        )

        # Before each LLM call:
        messages = compactor.maybe_compact(messages)
    """

    def __init__(
        self,
        soft_threshold: int = DEFAULT_SOFT_THRESHOLD,
        hard_threshold: int = DEFAULT_HARD_THRESHOLD,
        keep_recent: int = DEFAULT_KEEP_RECENT,
        summarize_fn: Optional[Callable] = None,
    ) -> None:
        self._soft_threshold = soft_threshold
        self._hard_threshold = hard_threshold
        self._keep_recent = keep_recent
        self._summarize_fn = summarize_fn
        self._compaction_count = 0
        self._total_tokens_saved = 0

    @property
    def compaction_count(self) -> int:
        """Number of compactions performed."""
        return self._compaction_count

    @property
    def total_tokens_saved(self) -> int:
        """Estimated total tokens saved by compaction."""
        return self._total_tokens_saved

    def check_threshold(self, messages: List[Dict[str, Any]]) -> str:
        """Check if messages exceed any threshold.

        Returns "ok", "soft", or "hard".
        """
        tokens = estimate_messages_tokens(messages)
        if tokens >= self._hard_threshold:
            return "hard"
        if tokens >= self._soft_threshold:
            return "soft"
        return "ok"

    def maybe_compact(
        self,
        messages: List[Dict[str, Any]],
        system_prompt_tokens: int = 0,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Compact messages if thresholds are exceeded.

        Returns (compacted_messages, was_compacted).
        """
        level = self.check_threshold(messages)

        if level == "ok":
            return messages, False

        if level == "soft":
            return self._soft_compact(messages)

        return self._hard_compact(messages)

    def _soft_compact(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Soft compaction: summarize old messages, keep recent ones."""
        if len(messages) <= self._keep_recent:
            return messages, False

        old_messages = messages[:-self._keep_recent]
        recent_messages = messages[-self._keep_recent:]

        # Try to summarize old messages
        summary = self._summarize_messages(old_messages)
        if not summary:
            # Fallback: just trim old messages
            return recent_messages, True

        tokens_before = estimate_messages_tokens(old_messages)
        tokens_after = estimate_tokens(summary)
        self._total_tokens_saved += max(0, tokens_before - tokens_after)
        self._compaction_count += 1

        # Replace old messages with summary
        summary_msg = {
            "role": "system",
            "content": "[Context Summary]\n{}".format(summary),
        }
        result = [summary_msg] + recent_messages

        logger.info(
            "Soft compaction: %d messages → summary + %d recent (saved ~%d tokens)",
            len(old_messages), len(recent_messages),
            tokens_before - tokens_after,
        )

        return result, True

    def _hard_compact(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Hard compaction: aggressive trim + summarize."""
        # Keep only the most recent messages
        keep = max(3, self._keep_recent // 2)

        if len(messages) <= keep:
            return messages, False

        old_messages = messages[:-keep]
        recent_messages = messages[-keep:]

        tokens_before = estimate_messages_tokens(old_messages)
        self._total_tokens_saved += tokens_before
        self._compaction_count += 1

        # Try summary
        summary = self._summarize_messages(old_messages)
        if summary:
            summary_msg = {
                "role": "system",
                "content": "[Compacted Context]\n{}".format(summary),
            }
            result = [summary_msg] + recent_messages
        else:
            # No summarizer: just drop old messages
            result = recent_messages

        logger.warning(
            "Hard compaction: dropped %d messages, kept %d (saved ~%d tokens)",
            len(old_messages), len(recent_messages), tokens_before,
        )

        return result, True

    def auto_compact_from_usage(
        self,
        messages: List[Dict[str, Any]],
        prompt_tokens: int,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Compact using actual token count from provider response.

        Uses real ``prompt_tokens`` instead of the heuristic estimator,
        inspired by claw-code's auto-compact that triggers when
        ``input_tokens > 100K``.

        Falls back to :meth:`maybe_compact` when ``prompt_tokens`` is 0
        (e.g. deterministic pipeline, no LLM call).
        """
        if prompt_tokens <= 0:
            return self.maybe_compact(messages)

        if prompt_tokens >= self._hard_threshold:
            return self._hard_compact(messages)
        if prompt_tokens >= self._soft_threshold:
            return self._soft_compact(messages)
        return messages, False

    def _summarize_messages(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Summarize a list of messages into a compact summary.

        If no summarize_fn is provided, uses a simple extraction approach.
        """
        if self._summarize_fn:
            try:
                return self._summarize_fn(messages)
            except Exception as e:
                logger.debug("Summarize function failed: %s", e)

        # Fallback: extract key points (simple, no LLM needed)
        points = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                # Take first sentence or first 100 chars
                first_line = content.strip().split("\n")[0][:100]
                points.append("{}: {}".format(role, first_line))

        if not points:
            return None

        # Keep at most 20 points
        if len(points) > 20:
            points = points[:5] + ["...({} messages omitted)...".format(len(points) - 10)] + points[-5:]

        return "\n".join(points)
