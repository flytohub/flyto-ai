"""Schema-constrained completion on the OpenAI provider."""

from __future__ import annotations

import pytest

from flyto_ai.providers.openai import OpenAIProvider

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["evidence"],
    "properties": {"evidence": {"type": "array", "items": {"type": "string"}}},
}


class FakeMessage:
    def __init__(self, content, refusal=None):
        self.content = content
        self.refusal = refusal


class FakeChoice:
    def __init__(self, content, *, finish_reason="stop", refusal=None):
        self.message = FakeMessage(content, refusal)
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, choices, model="gpt-4o-mini"):
        self.choices = choices
        self.model = model
        self.usage = type("U", (), {"prompt_tokens": 11, "completion_tokens": 7})()


def provider_with(response, captured=None):
    """A provider whose client returns this, recording what it was sent."""
    p = OpenAIProvider(api_key="test-only-not-a-real-key", model="gpt-4o-mini")

    class Completions:
        async def create(self, **kwargs):
            if captured is not None:
                captured.update(kwargs)
            if isinstance(response, Exception):
                raise response
            return response

    class Chat:
        completions = Completions()

    p._client = type("C", (), {"chat": Chat()})()
    return p


@pytest.mark.asyncio
async def test_a_schema_constrained_reply_is_returned_in_the_shared_shape():
    """Shaped like the other providers so a caller need not know who answered."""
    captured = {}
    p = provider_with(FakeResponse([FakeChoice('{"evidence": ["zone.overview"]}')]), captured)
    out = await p.complete_json_schema(
        messages=[{"role": "user", "content": "check zone 3"}], schema=SCHEMA)
    assert out["message"]["content"] == '{"evidence": ["zone.overview"]}'
    assert out["prompt_tokens"] == 11 and out["completion_tokens"] == 7


@pytest.mark.asyncio
async def test_the_schema_is_sent_strict():
    """Strict structured output removes the three shapes a caller would
    otherwise have to reject: prose, a fence, and an extra field."""
    captured = {}
    p = provider_with(FakeResponse([FakeChoice("{}")]), captured)
    await p.complete_json_schema(messages=[{"role": "user", "content": "x"}], schema=SCHEMA)
    fmt = captured["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == SCHEMA
    assert captured["temperature"] == 0, "a planner should not be creative"


@pytest.mark.asyncio
async def test_a_truncated_reply_is_an_error_not_a_wrong_answer():
    """Truncated JSON would parse as a refusal downstream, which reads as the
    model being wrong rather than the budget being too small."""
    p = provider_with(FakeResponse([FakeChoice('{"evide', finish_reason="length")]))
    with pytest.raises(RuntimeError, match="cut off"):
        await p.complete_json_schema(messages=[{"role": "user", "content": "x"}], schema=SCHEMA)


@pytest.mark.asyncio
async def test_a_model_refusal_is_surfaced_not_swallowed():
    p = provider_with(FakeResponse([FakeChoice(None, refusal="I can't help with that")]))
    with pytest.raises(RuntimeError, match="refused"):
        await p.complete_json_schema(messages=[{"role": "user", "content": "x"}], schema=SCHEMA)


@pytest.mark.asyncio
async def test_no_choices_is_an_error():
    p = provider_with(FakeResponse([]))
    with pytest.raises(RuntimeError, match="no choices"):
        await p.complete_json_schema(messages=[{"role": "user", "content": "x"}], schema=SCHEMA)


@pytest.mark.asyncio
@pytest.mark.parametrize("messages,match", [
    ([], "at least one"),
    ([{"role": "wizard", "content": "x"}], "unsupported role"),
    ([{"role": "user", "content": ""}], "empty or too large"),
])
async def test_malformed_messages_are_refused_before_a_call_is_made(messages, match):
    captured = {}
    p = provider_with(FakeResponse([FakeChoice("{}")]), captured)
    with pytest.raises(ValueError, match=match):
        await p.complete_json_schema(messages=messages, schema=SCHEMA)
    assert captured == {}, "nothing should have been sent"


@pytest.mark.asyncio
async def test_an_absurd_timeout_is_refused():
    p = provider_with(FakeResponse([FakeChoice("{}")]))
    with pytest.raises(ValueError, match="between 1 and 300"):
        await p.complete_json_schema(
            messages=[{"role": "user", "content": "x"}], schema=SCHEMA, timeout_seconds=0.1)


def test_the_provider_satisfies_the_structured_boundary():
    """The point of the method: one interface, not one per vendor.

    Checked structurally rather than with isinstance — StructuredJsonProvider is
    a plain Protocol, and an isinstance check against one raises rather than
    answering, which would look like the provider failing the boundary.
    """
    import inspect

    from flyto_ai.providers.ollama import OllamaProvider
    from flyto_ai.robotics_planning import StructuredJsonProvider

    def names(fn):
        return [p.name for p in inspect.signature(fn).parameters.values()]

    expected = names(StructuredJsonProvider.complete_json_schema)
    for provider in (OpenAIProvider, OllamaProvider):
        assert hasattr(provider, "complete_json_schema"), provider.__name__
        # Parameter names, not the full signature: annotations differ between
        # providers and a caller cannot tell, but a renamed keyword breaks
        # everyone.
        assert names(provider.complete_json_schema) == expected, provider.__name__
        assert inspect.iscoroutinefunction(provider.complete_json_schema)
