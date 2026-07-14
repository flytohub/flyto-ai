# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Tests for context window compaction."""

from flyto_ai.memory.compaction import (
    ContextCompactor,
    estimate_tokens,
    estimate_messages_tokens,
)


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello") > 0
    assert estimate_tokens("hello world " * 100) > estimate_tokens("hello")


def test_estimate_messages_tokens():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there, how can I help?"},
    ]
    tokens = estimate_messages_tokens(msgs)
    assert tokens > 0


def test_estimate_messages_multimodal():
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]},
    ]
    tokens = estimate_messages_tokens(msgs)
    assert tokens > 0


def test_compactor_ok_no_change():
    compactor = ContextCompactor(soft_threshold=10000, hard_threshold=20000)
    msgs = [{"role": "user", "content": "hello"}]
    result, compacted = compactor.maybe_compact(msgs)
    assert compacted is False
    assert result == msgs


def test_compactor_soft_threshold():
    compactor = ContextCompactor(soft_threshold=50, hard_threshold=200, keep_recent=3)
    msgs = [{"role": "user", "content": "x" * 30} for _ in range(10)]
    result, compacted = compactor.maybe_compact(msgs)
    assert compacted is True
    assert len(result) <= 4  # summary + 3 recent


def test_compactor_hard_threshold():
    compactor = ContextCompactor(soft_threshold=50, hard_threshold=100, keep_recent=5)
    msgs = [{"role": "user", "content": "x" * 100} for _ in range(20)]
    result, compacted = compactor.maybe_compact(msgs)
    assert compacted is True
    # Hard compaction keeps fewer messages
    assert len(result) <= 4


def test_compactor_preserves_recent():
    compactor = ContextCompactor(soft_threshold=30, hard_threshold=200, keep_recent=3)
    msgs = [{"role": "user", "content": "msg_{}".format(i)} for i in range(10)]
    result, compacted = compactor.maybe_compact(msgs)
    assert compacted is True
    # Last 3 messages should be preserved
    recent = [m for m in result if m.get("role") == "user"]
    assert any("msg_9" in m["content"] for m in recent)


def test_compactor_tracks_stats():
    compactor = ContextCompactor(soft_threshold=30, hard_threshold=200, keep_recent=3)
    msgs = [{"role": "user", "content": "x" * 20} for _ in range(10)]
    compactor.maybe_compact(msgs)
    assert compactor.compaction_count >= 1
    assert compactor.total_tokens_saved > 0


def test_compactor_with_custom_summarizer():
    def my_summarizer(msgs):
        return "Summary of {} messages".format(len(msgs))

    compactor = ContextCompactor(
        soft_threshold=30, hard_threshold=200,
        keep_recent=3, summarize_fn=my_summarizer,
    )
    msgs = [{"role": "user", "content": "x" * 20} for _ in range(10)]
    result, compacted = compactor.maybe_compact(msgs)
    assert compacted is True
    # Should contain summary message
    summary_msgs = [m for m in result if "Summary of" in m.get("content", "")]
    assert len(summary_msgs) == 1


def test_compactor_check_threshold():
    compactor = ContextCompactor(soft_threshold=50, hard_threshold=200)
    small = [{"role": "user", "content": "hi"}]
    assert compactor.check_threshold(small) == "ok"

    medium = [{"role": "user", "content": "x" * 200}]
    assert compactor.check_threshold(medium) == "soft"

    large = [{"role": "user", "content": "x" * 800}]
    assert compactor.check_threshold(large) == "hard"


def test_compactor_few_messages_no_compact():
    compactor = ContextCompactor(soft_threshold=10, hard_threshold=50, keep_recent=10)
    msgs = [{"role": "user", "content": "x" * 30} for _ in range(5)]
    result, compacted = compactor.maybe_compact(msgs)
    # Too few messages to compact (less than keep_recent)
    assert compacted is False
