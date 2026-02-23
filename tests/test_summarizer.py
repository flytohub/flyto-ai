# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for ConversationSummarizer — real LLM calls."""
import os

import pytest
import pytest_asyncio

from flyto_ai.memory.sqlite_store import SQLiteSessionStore
from flyto_ai.memory.summarizer import ConversationSummarizer

_has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))


def _make_provider():
    """Create a real OpenAI provider for summarization tests."""
    from flyto_ai.providers import create_provider
    return create_provider(
        "openai",
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.3,
        max_tokens=512,
    )


@pytest_asyncio.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "test_summary.db")
    s = SQLiteSessionStore(db_path=db_path)
    await s.init()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_no_summarize_under_threshold(store):
    """Under threshold — nothing happens."""
    summarizer = ConversationSummarizer(provider=None, threshold=20, keep_recent=10)
    for i in range(19):
        await store.add_message("s1", "user", "msg {}".format(i))

    result = await summarizer.maybe_summarize("s1", store)
    assert result is None
    assert await store.get_message_count("s1") == 19


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_openai_key, reason="OPENAI_API_KEY not set")
async def test_summarize_over_threshold(store):
    """Over threshold — old messages replaced with summary."""
    provider = _make_provider()
    summarizer = ConversationSummarizer(provider=provider, threshold=20, keep_recent=10)

    for i in range(25):
        role = "user" if i % 2 == 0 else "assistant"
        await store.add_message("s1", role, "Message number {}".format(i))

    result = await summarizer.maybe_summarize("s1", store)
    assert result is not None
    assert len(result) > 10  # non-trivial summary

    # Should have 10 recent messages left
    msgs = await store.get_messages("s1")
    assert len(msgs) == 10

    # Summary should be stored
    summaries = await store.get_summaries("s1")
    assert len(summaries) == 1
    assert summaries[0]["source_count"] == 15


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_openai_key, reason="OPENAI_API_KEY not set")
async def test_summary_content_quality(store):
    """Summary preserves key information."""
    provider = _make_provider()
    summarizer = ConversationSummarizer(provider=provider, threshold=5, keep_recent=2)

    await store.add_message("s1", "user", "My name is Alice and I live in Tokyo")
    await store.add_message("s1", "assistant", "Nice to meet you Alice! Tokyo is a great city.")
    await store.add_message("s1", "user", "I work as a software engineer at Google")
    await store.add_message("s1", "assistant", "That's a great job! What languages do you use?")
    await store.add_message("s1", "user", "Mostly Python and Go")
    await store.add_message("s1", "assistant", "Both are excellent choices for backend development.")

    result = await summarizer.maybe_summarize("s1", store)
    assert result is not None
    # Summary should mention key facts
    result_lower = result.lower()
    assert "alice" in result_lower or "tokyo" in result_lower or "google" in result_lower
