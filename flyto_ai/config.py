# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Agent configuration."""
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

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
class ClaudeCodeConfig:
    """Configuration for the Claude Code Agent mode."""
    max_budget_usd: float = 5.0
    max_turns: int = 30
    max_fix_attempts: int = 3
    allowed_tools: List[str] = field(default_factory=lambda: [
        "Read", "Edit", "Write", "Bash", "Glob", "Grep",
    ])
    verification_timeout: int = 120
    evidence_dir: str = "~/.flyto/evidence"


@dataclass
class FailoverProviderConfig:
    """Configuration for a single provider in the failover chain."""
    provider: str = ""
    api_key: str = ""
    model: str = ""
    base_url: Optional[str] = None


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
    max_tool_rounds: int = 30
    max_validation_rounds: int = 2

    # Memory system
    memory_db_path: str = "~/.flyto/memory.db"
    enable_memory: bool = True
    embedding_model: str = "text-embedding-3-small"

    # Docker sandbox
    enable_sandbox: bool = False
    sandbox_image: str = "flyto-sandbox:latest"
    sandbox_timeout: int = 60

    # Claude Code Agent
    claude_code: ClaudeCodeConfig = field(default_factory=ClaudeCodeConfig)

    # Failover chain
    fallback_providers: List[FailoverProviderConfig] = field(default_factory=list)

    # Cost tracking
    session_budget_usd: Optional[float] = None
    global_budget_usd: Optional[float] = None

    # Transcript
    enable_transcript: bool = True
    transcript_dir: str = "~/.flyto/transcripts"

    # Vault
    vault_path: str = "~/.flyto/vault.enc"
    vault_passphrase: Optional[str] = None
    vault_auto_inject: bool = False

    # Injection detection
    enable_injection_detection: bool = True

    # Browser engine
    browser_engine: str = "chromium"  # "chromium" | "firefox" | "webkit"

    @classmethod
    def from_dict(cls, data: dict) -> "AgentConfig":
        fallbacks = []
        for fb in data.get("fallback_providers", []):
            fallbacks.append(FailoverProviderConfig(
                provider=fb.get("provider", ""),
                api_key=fb.get("api_key", ""),
                model=fb.get("model", ""),
                base_url=fb.get("base_url") or None,
            ))

        return cls(
            provider=data.get("provider", ""),
            api_key=data.get("api_key", ""),
            model=data.get("model", ""),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 4096),
            base_url=data.get("base_url") or None,
            max_tool_rounds=data.get("max_tool_rounds", 30),
            max_validation_rounds=data.get("max_validation_rounds", 2),
            memory_db_path=data.get("memory_db_path", "~/.flyto/memory.db"),
            enable_memory=data.get("enable_memory", True),
            embedding_model=data.get("embedding_model", "text-embedding-3-small"),
            enable_sandbox=data.get("enable_sandbox", False),
            sandbox_image=data.get("sandbox_image", "flyto-sandbox:latest"),
            sandbox_timeout=data.get("sandbox_timeout", 60),
            claude_code=ClaudeCodeConfig(
                **{k: v for k, v in data.get("claude_code", {}).items()
                   if k in ("max_budget_usd", "max_turns", "max_fix_attempts",
                            "allowed_tools", "verification_timeout", "evidence_dir")},
            ),
            fallback_providers=fallbacks,
            session_budget_usd=data.get("session_budget_usd"),
            global_budget_usd=data.get("global_budget_usd"),
            enable_transcript=data.get("enable_transcript", True),
            transcript_dir=data.get("transcript_dir", "~/.flyto/transcripts"),
            vault_path=data.get("vault_path", "~/.flyto/vault.enc"),
            vault_passphrase=data.get("vault_passphrase"),
            vault_auto_inject=data.get("vault_auto_inject", False),
            enable_injection_detection=data.get("enable_injection_detection", True),
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

        # Parse fallback providers from env: FLYTO_AI_FALLBACK_1_PROVIDER, etc.
        fallbacks = []
        for i in range(1, 4):  # support up to 3 fallbacks
            fb_provider = os.getenv("FLYTO_AI_FALLBACK_{}_PROVIDER".format(i), "")
            if fb_provider:
                fallbacks.append(FailoverProviderConfig(
                    provider=fb_provider,
                    api_key=os.getenv("FLYTO_AI_FALLBACK_{}_API_KEY".format(i), ""),
                    model=os.getenv("FLYTO_AI_FALLBACK_{}_MODEL".format(i), ""),
                    base_url=os.getenv("FLYTO_AI_FALLBACK_{}_BASE_URL".format(i)) or None,
                ))

        session_budget = os.getenv("FLYTO_AI_SESSION_BUDGET_USD", "")
        global_budget = os.getenv("FLYTO_AI_GLOBAL_BUDGET_USD", "")

        return cls(
            provider=provider,
            api_key=api_key,
            model=os.getenv("FLYTO_AI_MODEL", ""),
            temperature=float(os.getenv("FLYTO_AI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("FLYTO_AI_MAX_TOKENS", "4096")),
            base_url=os.getenv("FLYTO_AI_BASE_URL") or None,
            max_tool_rounds=int(os.getenv("FLYTO_AI_MAX_TOOL_ROUNDS", "30")),
            memory_db_path=os.getenv("FLYTO_AI_MEMORY_DB", "~/.flyto/memory.db"),
            enable_memory=os.getenv("FLYTO_AI_ENABLE_MEMORY", "true").lower() != "false",
            embedding_model=os.getenv("FLYTO_AI_EMBEDDING_MODEL", "text-embedding-3-small"),
            enable_sandbox=os.getenv("FLYTO_AI_ENABLE_SANDBOX", "false").lower() == "true",
            sandbox_image=os.getenv("FLYTO_AI_SANDBOX_IMAGE", "flyto-sandbox:latest"),
            sandbox_timeout=int(os.getenv("FLYTO_AI_SANDBOX_TIMEOUT", "60")),
            claude_code=ClaudeCodeConfig(
                max_budget_usd=float(os.getenv("FLYTO_AI_CC_MAX_BUDGET", "5.0")),
                max_turns=int(os.getenv("FLYTO_AI_CC_MAX_TURNS", "30")),
                max_fix_attempts=int(os.getenv("FLYTO_AI_CC_MAX_FIX_ATTEMPTS", "3")),
            ),
            fallback_providers=fallbacks,
            session_budget_usd=float(session_budget) if session_budget else None,
            global_budget_usd=float(global_budget) if global_budget else None,
            enable_transcript=os.getenv("FLYTO_AI_ENABLE_TRANSCRIPT", "true").lower() != "false",
            transcript_dir=os.getenv("FLYTO_AI_TRANSCRIPT_DIR", "~/.flyto/transcripts"),
            vault_path=os.getenv("FLYTO_AI_VAULT_PATH", "~/.flyto/vault.enc"),
            vault_passphrase=os.getenv("FLYTO_VAULT_PASSPHRASE") or None,
            vault_auto_inject=os.getenv("FLYTO_AI_VAULT_AUTO_INJECT", "false").lower() == "true",
            enable_injection_detection=os.getenv("FLYTO_AI_ENABLE_INJECTION_DETECTION", "true").lower() != "false",
        )

    def __post_init__(self):
        """Validate config values."""
        # Bounds check temperature and max_tokens
        if self.temperature < 0.0:
            logger.warning("temperature %s < 0, clamping to 0.0", self.temperature)
            self.temperature = 0.0
        elif self.temperature > 2.0:
            logger.warning("temperature %s > 2.0, clamping to 2.0", self.temperature)
            self.temperature = 2.0

        if self.max_tokens < 1:
            logger.warning("max_tokens %s < 1, setting to 1", self.max_tokens)
            self.max_tokens = 1
        elif self.max_tokens > 200_000:
            logger.warning("max_tokens %s > 200000, clamping to 200000", self.max_tokens)
            self.max_tokens = 200_000

        # Validate base_url against SSRF allowlist if set
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
        return "gpt-4o"
