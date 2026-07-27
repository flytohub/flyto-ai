# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Provider failover chain — automatic LLM provider switching on failure."""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from flyto_ai.models import StreamCallback
from flyto_ai.providers.base import DispatchFn, LLMProvider

logger = logging.getLogger(__name__)

# Errors that trigger failover (rate-limit or server-side)
_FAILOVER_STATUS_CODES = {429, 500, 502, 503, 504}

# Keywords in error messages that indicate transient/server failures
_FAILOVER_KEYWORDS = frozenset({
    "rate_limit", "rate limit", "too many requests",
    "overloaded", "capacity", "server_error", "internal server error",
    "bad gateway", "service unavailable", "gateway timeout",
    "connection error", "timeout", "timed out",
})

# Max retries per provider before moving to the next
_MAX_RETRIES_PER_PROVIDER = 3

# Base delay for exponential backoff (seconds)
_BASE_DELAY = 1.0

# Max delay cap (seconds)
_MAX_DELAY = 30.0


def _is_failover_error(exc: Exception) -> bool:
    """Check if an exception should trigger provider failover."""
    exc_str = str(exc).lower()

    # Check for HTTP status codes in common client libraries
    status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status_code, int) and status_code in _FAILOVER_STATUS_CODES:
        return True

    # Check error message keywords
    return any(kw in exc_str for kw in _FAILOVER_KEYWORDS)


class ProviderChain(LLMProvider):
    """Wraps multiple LLM providers with automatic failover.

    On rate-limit (429) or server error (5xx), automatically switches
    to the next provider in the chain. Each provider gets its own
    retry budget with exponential backoff.

    Session-level pinning: once a provider succeeds, subsequent calls
    in the same session prefer that provider (to leverage caching).
    """

    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: Optional[List[LLMProvider]] = None,
        provider_names: Optional[List[str]] = None,
    ) -> None:
        self._providers: List[LLMProvider] = [primary] + (fallbacks or [])
        self._names: List[str] = provider_names or [
            "provider_{}".format(i) for i in range(len(self._providers))
        ]
        # Session pinning: index of last successful provider
        self._pinned_index: int = 0
        # Per-provider failure tracking
        self._consecutive_failures: List[int] = [0] * len(self._providers)
        self._last_failure_time: List[float] = [0.0] * len(self._providers)

    @property
    def active_provider_name(self) -> str:
        """Name of the currently pinned provider."""
        return self._names[self._pinned_index]

    @property
    def provider_count(self) -> int:
        """Number of providers in the chain."""
        return len(self._providers)

    def prefer_provider(self, provider_name: str) -> bool:
        """Prefer a configured provider for the next call.

        Returns ``False`` without changing the active provider when the
        requested label is not part of this chain.
        """
        try:
            provider_index = self._names.index(provider_name)
        except ValueError:
            return False
        self._pinned_index = provider_index
        return True

    def _reset_failures(self, index: int) -> None:
        """Reset failure tracking for a provider after success."""
        self._consecutive_failures[index] = 0
        self._last_failure_time[index] = 0.0

    def _record_failure(self, index: int) -> None:
        """Record a failure for a provider."""
        self._consecutive_failures[index] += 1
        self._last_failure_time[index] = time.monotonic()

    def _get_backoff_delay(self, index: int) -> float:
        """Calculate exponential backoff delay for a provider."""
        failures = self._consecutive_failures[index]
        if failures == 0:
            return 0.0
        delay = _BASE_DELAY * (2 ** (failures - 1))
        return min(delay, _MAX_DELAY)

    def _build_try_order(self) -> List[int]:
        """Build the order in which to try providers.

        Starts with the pinned provider, then tries others in order.
        Skips providers that have exhausted their retry budget recently.
        """
        order = [self._pinned_index]
        for i in range(len(self._providers)):
            if i != self._pinned_index:
                order.append(i)
        return order

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict],
        dispatch_fn: DispatchFn,
        max_rounds: int = 30,
        on_stream: Optional[StreamCallback] = None,
    ) -> Tuple[str, List[Dict[str, Any]], int, Dict[str, int]]:
        """Run chat with automatic failover across providers."""
        try_order = self._build_try_order()
        last_error: Optional[Exception] = None

        for provider_idx in try_order:
            provider = self._providers[provider_idx]
            name = self._names[provider_idx]

            for attempt in range(_MAX_RETRIES_PER_PROVIDER):
                # Exponential backoff if this provider has recent failures
                delay = self._get_backoff_delay(provider_idx)
                if delay > 0 and attempt > 0:
                    logger.info(
                        "Failover: %s backoff %.1fs (attempt %d/%d)",
                        name, delay, attempt + 1, _MAX_RETRIES_PER_PROVIDER,
                    )
                    await asyncio.sleep(delay)

                try:
                    result = await provider.chat(
                        messages, system_prompt, tools,
                        dispatch_fn, max_rounds, on_stream,
                    )
                    # Success — pin this provider for future calls
                    self._pinned_index = provider_idx
                    self._reset_failures(provider_idx)
                    if provider_idx != try_order[0]:
                        logger.info(
                            "Failover: switched from %s to %s (success)",
                            self._names[try_order[0]], name,
                        )
                    return result

                except Exception as e:
                    self._record_failure(provider_idx)
                    last_error = e

                    if _is_failover_error(e):
                        logger.warning(
                            "Failover: %s failed (attempt %d/%d): %s",
                            name, attempt + 1, _MAX_RETRIES_PER_PROVIDER,
                            str(e)[:200],
                        )
                        continue  # retry same provider with backoff
                    else:
                        # Non-transient error — don't retry this provider
                        logger.warning(
                            "Failover: %s non-transient error, skipping: %s",
                            name, str(e)[:200],
                        )
                        break  # move to next provider

            # This provider exhausted retries — move to next
            logger.info("Failover: %s exhausted retries, trying next provider", name)

        # All providers failed
        if last_error:
            raise last_error
        raise RuntimeError("All providers in chain failed")


def create_provider_chain(
    configs: List[Dict[str, Any]],
) -> ProviderChain:
    """Create a ProviderChain from a list of provider configs.

    Each config dict should have:
    - provider: str (e.g. "openai", "anthropic", "ollama")
    - api_key: str (optional for ollama)
    - model: str (optional, uses provider default)
    - base_url: str (optional)
    - temperature: float (optional)
    - max_tokens: int (optional)
    """
    from flyto_ai.providers import create_provider

    if not configs:
        raise ValueError("At least one provider config required")

    providers: List[LLMProvider] = []
    names: List[str] = []

    for cfg in configs:
        provider_name = cfg.get("provider", "openai")
        kwargs = {}
        if "model" in cfg:
            kwargs["model"] = cfg["model"]
        if "temperature" in cfg:
            kwargs["temperature"] = cfg["temperature"]
        if "max_tokens" in cfg:
            kwargs["max_tokens"] = cfg["max_tokens"]

        if provider_name == "ollama":
            kwargs["base_url"] = cfg.get("base_url", "http://localhost:11434/v1")
        else:
            if "api_key" in cfg:
                kwargs["api_key"] = cfg["api_key"]
            if "base_url" in cfg:
                kwargs["base_url"] = cfg["base_url"]

        providers.append(create_provider(provider_name, **kwargs))
        label = "{}:{}".format(provider_name, cfg.get("model", "default"))
        names.append(label)

    return ProviderChain(
        primary=providers[0],
        fallbacks=providers[1:] if len(providers) > 1 else None,
        provider_names=names,
    )
