# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Token cost tracking and budget enforcement."""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Cost per 1M tokens (USD) — updated 2025-05
# Format: model_prefix → (input_per_1M, output_per_1M)
MODEL_COSTS: Dict[str, tuple] = {
    # Anthropic
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
    "claude-haiku-3-5": (0.80, 4.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-haiku": (0.80, 4.0),
    # OpenAI
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    "o3-mini": (1.10, 4.40),
    # DeepSeek
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
    # Groq (hosted)
    "llama-3": (0.05, 0.08),
    "mixtral": (0.24, 0.24),
    # Local (free)
    "llama3": (0.0, 0.0),
    "qwen": (0.0, 0.0),
    "mistral": (0.0, 0.0),
    "phi": (0.0, 0.0),
}


def _match_model_cost(model: str) -> tuple:
    """Find the best matching cost entry for a model name.

    Tries exact match first, then prefix match (longest wins).
    Returns (input_per_1M, output_per_1M) or (0.0, 0.0) if unknown.
    """
    if not model:
        return (0.0, 0.0)

    lower = model.lower()

    # Exact match
    if lower in MODEL_COSTS:
        return MODEL_COSTS[lower]

    # Prefix match (longest prefix wins)
    best_prefix = ""
    best_cost = (0.0, 0.0)
    for prefix, cost in MODEL_COSTS.items():
        if lower.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
            best_cost = cost

    return best_cost


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
) -> float:
    """Estimate cost in USD for a single LLM call.

    Cache-read tokens are charged at 10% of input rate (Anthropic pricing).
    """
    input_rate, output_rate = _match_model_cost(model)
    input_cost = (prompt_tokens / 1_000_000) * input_rate
    output_cost = (completion_tokens / 1_000_000) * output_rate
    cache_cost = (cache_read_tokens / 1_000_000) * input_rate * 0.1
    return input_cost + output_cost + cache_cost


@dataclass
class CostRecord:
    """A single cost record for one LLM call."""
    timestamp: float
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    estimated_cost_usd: float
    is_blueprint_replay: bool = False


@dataclass
class CostTracker:
    """Tracks token usage and estimated costs across sessions.

    Features:
    - Per-call cost recording
    - Session-level and global budget caps
    - Blueprint replays marked as zero-cost
    - Running totals and savings tracking
    """
    session_budget_usd: Optional[float] = None
    global_budget_usd: Optional[float] = None

    # Internal state
    _records: List[CostRecord] = field(default_factory=list)
    _session_total_usd: float = 0.0
    _global_total_usd: float = 0.0
    _blueprint_savings_usd: float = 0.0
    _total_prompt_tokens: int = 0
    _total_completion_tokens: int = 0

    def record(
        self,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        is_blueprint_replay: bool = False,
    ) -> CostRecord:
        """Record a single LLM call's cost.

        Returns the CostRecord for inspection.
        Raises BudgetExceededError if budget is exceeded.
        """
        cost = estimate_cost(model, prompt_tokens, completion_tokens, cache_read_tokens)

        record = CostRecord(
            timestamp=time.time(),
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=cache_read_tokens,
            estimated_cost_usd=0.0 if is_blueprint_replay else cost,
            is_blueprint_replay=is_blueprint_replay,
        )
        self._records.append(record)

        if is_blueprint_replay:
            self._blueprint_savings_usd += cost
        else:
            self._session_total_usd += cost
            self._global_total_usd += cost

        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        # Budget check (warn but don't block)
        if self.session_budget_usd and self._session_total_usd > self.session_budget_usd:
            logger.warning(
                "Session budget exceeded: $%.4f / $%.2f",
                self._session_total_usd, self.session_budget_usd,
            )
            raise BudgetExceededError(
                "Session budget exceeded: ${:.4f} / ${:.2f}".format(
                    self._session_total_usd, self.session_budget_usd
                ),
                current=self._session_total_usd,
                limit=self.session_budget_usd,
            )

        if self.global_budget_usd and self._global_total_usd > self.global_budget_usd:
            logger.warning(
                "Global budget exceeded: $%.4f / $%.2f",
                self._global_total_usd, self.global_budget_usd,
            )
            raise BudgetExceededError(
                "Global budget exceeded: ${:.4f} / ${:.2f}".format(
                    self._global_total_usd, self.global_budget_usd
                ),
                current=self._global_total_usd,
                limit=self.global_budget_usd,
            )

        return record

    @property
    def session_total_usd(self) -> float:
        """Total cost for the current session."""
        return self._session_total_usd

    @property
    def global_total_usd(self) -> float:
        """Total cost across all sessions."""
        return self._global_total_usd

    @property
    def blueprint_savings_usd(self) -> float:
        """Total savings from blueprint replays."""
        return self._blueprint_savings_usd

    @property
    def total_prompt_tokens(self) -> int:
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._total_completion_tokens

    @property
    def call_count(self) -> int:
        """Total number of LLM calls recorded."""
        return len(self._records)

    def reset_session(self) -> None:
        """Reset session-level counters (keep global)."""
        self._session_total_usd = 0.0
        self._records.clear()

    def summary(self) -> Dict:
        """Return a cost summary dict."""
        return {
            "session_total_usd": round(self._session_total_usd, 6),
            "global_total_usd": round(self._global_total_usd, 6),
            "blueprint_savings_usd": round(self._blueprint_savings_usd, 6),
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "call_count": self.call_count,
            "session_budget_usd": self.session_budget_usd,
            "global_budget_usd": self.global_budget_usd,
        }


class BudgetExceededError(Exception):
    """Raised when a cost budget is exceeded."""

    def __init__(self, message: str, current: float, limit: float):
        super().__init__(message)
        self.current = current
        self.limit = limit
