# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Tests for provider failover chain."""
import pytest
from unittest.mock import AsyncMock

from flyto_ai.providers.failover import (
    ProviderChain,
    _is_failover_error,
)


class FakeProvider:
    """Fake LLM provider for testing."""

    def __init__(self, name="fake", fail_count=0, error_cls=None, error_msg="error"):
        self.name = name
        self.fail_count = fail_count
        self.call_count = 0
        self.error_cls = error_cls or Exception
        self.error_msg = error_msg

    async def chat(self, messages, system_prompt, tools, dispatch_fn, max_rounds=30, on_stream=None):
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise self.error_cls(self.error_msg)
        return ("response from {}".format(self.name), [], 1, {"prompt_tokens": 10, "completion_tokens": 5})


class RateLimitError(Exception):
    status_code = 429


class ServerError(Exception):
    status_code = 500


class BadRequestError(Exception):
    status_code = 400


# --- _is_failover_error tests ---

def test_failover_on_429():
    assert _is_failover_error(RateLimitError("rate limited"))


def test_failover_on_500():
    assert _is_failover_error(ServerError("internal error"))


def test_no_failover_on_400():
    assert not _is_failover_error(BadRequestError("bad request"))


def test_failover_on_keyword():
    assert _is_failover_error(Exception("service unavailable"))
    assert _is_failover_error(Exception("rate limit exceeded"))
    assert _is_failover_error(Exception("server is overloaded"))


def test_no_failover_on_normal_error():
    assert not _is_failover_error(Exception("invalid API key"))
    assert not _is_failover_error(Exception("model not found"))


# --- ProviderChain tests ---

@pytest.mark.asyncio
async def test_chain_uses_primary_on_success():
    primary = FakeProvider("primary")
    fallback = FakeProvider("fallback")
    chain = ProviderChain(primary, [fallback], ["primary", "fallback"])

    result = await chain.chat([], "sys", [], AsyncMock())
    assert result[0] == "response from primary"
    assert primary.call_count == 1
    assert fallback.call_count == 0


@pytest.mark.asyncio
async def test_chain_fails_over_on_rate_limit():
    primary = FakeProvider("primary", fail_count=99, error_cls=RateLimitError, error_msg="429")
    fallback = FakeProvider("fallback")
    chain = ProviderChain(primary, [fallback], ["primary", "fallback"])

    result = await chain.chat([], "sys", [], AsyncMock())
    assert result[0] == "response from fallback"
    assert primary.call_count == 3  # exhausts retry budget
    assert fallback.call_count == 1


@pytest.mark.asyncio
async def test_chain_retries_before_failover():
    # Primary fails once then succeeds
    primary = FakeProvider("primary", fail_count=1, error_cls=RateLimitError, error_msg="rate limit")
    fallback = FakeProvider("fallback")
    chain = ProviderChain(primary, [fallback], ["primary", "fallback"])

    result = await chain.chat([], "sys", [], AsyncMock())
    assert result[0] == "response from primary"
    assert primary.call_count == 2
    assert fallback.call_count == 0


@pytest.mark.asyncio
async def test_chain_skips_on_non_transient():
    # Non-transient error → skip immediately to fallback
    primary = FakeProvider("primary", fail_count=99, error_cls=BadRequestError, error_msg="invalid key")
    fallback = FakeProvider("fallback")
    chain = ProviderChain(primary, [fallback], ["primary", "fallback"])

    result = await chain.chat([], "sys", [], AsyncMock())
    assert result[0] == "response from fallback"
    assert primary.call_count == 1  # only one attempt for non-transient


@pytest.mark.asyncio
async def test_chain_all_fail():
    primary = FakeProvider("primary", fail_count=99, error_cls=ServerError, error_msg="500")
    fallback = FakeProvider("fallback", fail_count=99, error_cls=ServerError, error_msg="500")
    chain = ProviderChain(primary, [fallback], ["primary", "fallback"])

    with pytest.raises(ServerError):
        await chain.chat([], "sys", [], AsyncMock())


@pytest.mark.asyncio
async def test_chain_session_pinning():
    # After failover to fallback, subsequent calls prefer fallback
    primary = FakeProvider("primary", fail_count=99, error_cls=RateLimitError, error_msg="429")
    fallback = FakeProvider("fallback")
    chain = ProviderChain(primary, [fallback], ["primary", "fallback"])

    await chain.chat([], "sys", [], AsyncMock())
    assert chain.active_provider_name == "fallback"
    assert chain._pinned_index == 1


def test_chain_properties():
    p1 = FakeProvider("a")
    p2 = FakeProvider("b")
    chain = ProviderChain(p1, [p2], ["provider_a", "provider_b"])
    assert chain.provider_count == 2
    assert chain.active_provider_name == "provider_a"
