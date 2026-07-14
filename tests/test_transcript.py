# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Tests for JSONL transcript recording and replay."""
import tempfile
import shutil

import pytest

from flyto_ai.transcript import TranscriptWriter, load_transcript, replay_messages


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_transcript_write_and_load(tmpdir):
    tw = TranscriptWriter("sess-1", transcript_dir=tmpdir)
    tw.record_user("hello")
    tw.record_assistant("hi there", provider="openai", model="gpt-4o-mini")
    tw.close()

    events = load_transcript(str(tw.path))
    assert len(events) == 2
    assert events[0]["type"] == "user"
    assert events[0]["data"]["message"] == "hello"
    assert events[1]["type"] == "assistant"
    assert events[1]["data"]["provider"] == "openai"


def test_transcript_tool_call(tmpdir):
    tw = TranscriptWriter("sess-2", transcript_dir=tmpdir)
    tw.record_tool_call("search_modules", {"query": "email"}, round_num=1)
    tw.record_tool_result("search_modules", {"results": ["email.send"]}, ok=True)
    tw.close()

    events = load_transcript(str(tw.path))
    assert len(events) == 2
    assert events[0]["type"] == "tool_call"
    assert events[0]["data"]["name"] == "search_modules"
    assert events[1]["type"] == "tool_result"


def test_transcript_execution(tmpdir):
    tw = TranscriptWriter("sess-3", transcript_dir=tmpdir)
    tw.record_execution("browser.goto", ok=True, result_preview='{"status": "success"}')
    tw.close()

    events = load_transcript(str(tw.path))
    assert events[0]["type"] == "execution"
    assert events[0]["data"]["ok"] is True


def test_transcript_error(tmpdir):
    tw = TranscriptWriter("sess-4", transcript_dir=tmpdir)
    tw.record_error("connection timeout", context="browser.launch")
    tw.close()

    events = load_transcript(str(tw.path))
    assert events[0]["type"] == "error"
    assert "timeout" in events[0]["data"]["error"]


def test_transcript_meta(tmpdir):
    tw = TranscriptWriter("sess-5", transcript_dir=tmpdir)
    tw.record_meta({"event": "session_start", "provider": "anthropic"})
    tw.close()

    events = load_transcript(str(tw.path))
    assert events[0]["type"] == "meta"
    assert events[0]["data"]["provider"] == "anthropic"


def test_transcript_crash_safe(tmpdir):
    """Each line is independently parseable — corruption of one line doesn't break others."""
    tw = TranscriptWriter("sess-6", transcript_dir=tmpdir)
    tw.record_user("message 1")
    tw.record_user("message 2")
    tw.close()

    # Inject a corrupted line
    with open(str(tw.path), "a") as f:
        f.write("NOT JSON\n")

    # Append more
    tw2 = TranscriptWriter.__new__(TranscriptWriter)
    tw2._session_id = "sess-6"
    tw2._path = tw.path
    tw2._file = None
    tw2.record_user("message 3")
    tw2.close()

    events = load_transcript(str(tw.path))
    assert len(events) == 3  # corrupted line skipped


def test_transcript_replay(tmpdir):
    tw = TranscriptWriter("sess-7", transcript_dir=tmpdir)
    tw.record_user("step 1")
    tw.record_assistant("response 1")
    tw.record_tool_call("test", {})  # should be excluded from replay
    tw.record_user("step 2")
    tw.record_assistant("response 2")
    tw.close()

    events = load_transcript(str(tw.path))
    messages = replay_messages(events)
    assert len(messages) == 4
    assert messages[0] == {"role": "user", "content": "step 1"}
    assert messages[1] == {"role": "assistant", "content": "response 1"}


def test_transcript_large_result_truncation(tmpdir):
    tw = TranscriptWriter("sess-8", transcript_dir=tmpdir)
    large_result = "x" * 5000
    tw.record_tool_result("test", large_result, ok=True)
    tw.close()

    events = load_transcript(str(tw.path))
    result = events[0]["data"]["result"]
    assert len(result) <= 2100  # 2000 + truncation note


def test_load_nonexistent():
    events = load_transcript("/nonexistent/path/file.jsonl")
    assert events == []


def test_transcript_session_id_in_filename(tmpdir):
    tw = TranscriptWriter("my-session-id", transcript_dir=tmpdir)
    assert "my-session-id" in str(tw.path)
    tw.close()
