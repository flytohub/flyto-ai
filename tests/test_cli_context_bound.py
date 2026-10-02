"""The conversation re-sent each round keeps recent tool results and shortens old ones."""

from flyto_ai.cli_runtime.transport import (
    KEEP_RECENT_TOOL_RESULTS,
    OLD_TOOL_RESULT_CHARS,
    TOOL_RESULT_CHARS,
    bounded_context,
)


def _tool(size, name="inspect_page"):
    return {"role": "tool", "tool_name": name, "content": "x" * size}


def test_small_conversations_are_sent_unchanged():
    context = [{"role": "user", "content": "goal"}, _tool(100), {"role": "assistant", "content": "{}"}]
    assert bounded_context(context) == context


def test_old_tool_results_are_shortened_and_recent_ones_kept():
    context = [{"role": "user", "content": "goal"}] + [_tool(10_000) for _ in range(6)]
    bounded = bounded_context(context)
    old, recent = bounded[1:4], bounded[4:]
    assert len(recent) == KEEP_RECENT_TOOL_RESULTS
    assert all(item["content"] == "x" * 10_000 for item in recent)
    for item in old:
        assert item["content"].startswith("x" * OLD_TOOL_RESULT_CHARS)
        assert "8500 more characters" in item["content"]
        assert item["tool_name"] == "inspect_page"


def test_even_a_recent_tool_result_has_a_ceiling_and_the_original_is_not_mutated():
    context = [_tool(TOOL_RESULT_CHARS + 50)]
    bounded = bounded_context(context)
    assert len(bounded[0]["content"]) < TOOL_RESULT_CHARS + 100
    assert len(context[0]["content"]) == TOOL_RESULT_CHARS + 50


def test_user_and_assistant_messages_are_never_cut():
    long = "y" * 50_000
    context = [{"role": "user", "content": long}, {"role": "assistant", "content": long}]
    assert bounded_context(context) == context


def test_only_the_newest_screenshots_are_sent_with_a_turn():
    import asyncio
    import json

    from flyto_ai.cli_runtime.transport import IMAGES_PER_TURN, CliTransport

    sent = []

    async def complete(**kwargs):
        sent.append(kwargs["images"])
        return json.dumps({"content": "done", "tool_calls": []})

    transport = CliTransport(object(), completion_fn=complete)
    transport.image_completion_fn = complete
    transport.images = [{"media_type": "image/png", "base64": str(index)} for index in range(8)]
    asyncio.run(transport._infer("prompt"))
    assert [image["base64"] for image in sent[0]] == [str(index) for index in range(8 - IMAGES_PER_TURN, 8)]
