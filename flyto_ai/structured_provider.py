# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Provider-neutral boundary for schema-constrained JSON completions.

Every bounded planner in flyto-ai (mission interpretation, the legacy lab
planner, and any future contract-driven planner) talks to an LLM through this
one protocol, so no planner needs to know which vendor answered and no generic
module has to import a domain-specific planner to reach it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class StructuredJsonProvider(Protocol):
    """Minimal provider boundary required by bounded structured planners."""

    async def complete_json_schema(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, Any],
        timeout_seconds: float = 120.0,
    ) -> dict[str, Any]:
        """Return a provider-native completion containing message.content."""


__all__ = ["StructuredJsonProvider"]
