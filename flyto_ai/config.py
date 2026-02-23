# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Agent configuration."""
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Function calling quality ratings for known models.
# Used by docs and model selection guidance.
FUNCTION_CALLING_SUPPORT = {
    # Cloud — excellent
    "gpt-4o": "excellent",
    "gpt-4o-mini": "good",
    "claude-sonnet-4-5-20250929": "excellent",
    "claude-haiku-4-5-20251001": "good",
    # Ollama / local — varies
    "qwen2.5:7b": "good",
    "qwen2.5:14b": "good",
    "qwen2.5-coder:7b": "good",
    "llama3.1:8b": "fair",
    "llama3.2": "poor",
    "mistral": "fair",
    "deepseek-r1:8b": "poor",
}


@dataclass
class AgentConfig:
    """Configuration for the AI agent.

    Can be created directly, from a dict, or from environment variables.
    """
    provider: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    base_url: Optional[str] = None
    max_tool_rounds: int = 15
    max_validation_rounds: int = 2

    @classmethod
    def from_dict(cls, data: dict) -> "AgentConfig":
        return cls(
            provider=data.get("provider", ""),
            api_key=data.get("api_key", ""),
            model=data.get("model", ""),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 4096),
            base_url=data.get("base_url") or None,
            max_tool_rounds=data.get("max_tool_rounds", 15),
            max_validation_rounds=data.get("max_validation_rounds", 2),
        )

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create config from environment variables.

        Reads FLYTO_AI_PROVIDER, FLYTO_AI_API_KEY, FLYTO_AI_MODEL, etc.
        Falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY if provider-specific
        key is not set.
        """
        provider = os.getenv("FLYTO_AI_PROVIDER", "")
        api_key = os.getenv("FLYTO_AI_API_KEY", "")

        if not api_key:
            if provider == "openai" or not provider:
                api_key = os.getenv("OPENAI_API_KEY", "")
                if api_key and not provider:
                    provider = "openai"
            if not api_key and provider != "openai":
                api_key = os.getenv("ANTHROPIC_API_KEY", "")
                if api_key and not provider:
                    provider = "anthropic"

        # Ollama: no key needed
        if not api_key and provider == "ollama":
            api_key = "ollama"

        return cls(
            provider=provider,
            api_key=api_key,
            model=os.getenv("FLYTO_AI_MODEL", ""),
            temperature=float(os.getenv("FLYTO_AI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("FLYTO_AI_MAX_TOKENS", "4096")),
            base_url=os.getenv("FLYTO_AI_BASE_URL") or None,
        )

    def __post_init__(self):
        """Validate base_url against SSRF allowlist if set."""
        if self.base_url:
            from flyto_ai.prompt.policies import validate_base_url
            if not validate_base_url(self.base_url):
                logger.warning(
                    "base_url %s not in SSRF allowlist — clearing to prevent misuse",
                    self.base_url,
                )
                self.base_url = None

    @property
    def resolved_model(self) -> str:
        """Return model with sensible defaults per provider."""
        if self.model:
            return self.model
        if self.provider == "anthropic":
            return "claude-sonnet-4-5-20250929"
        if self.provider == "ollama":
            return "llama3.2"
        return "gpt-4o-mini"
