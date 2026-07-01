# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
import os

from flyto_ai.providers.base import LLMProvider

__all__ = [
    "LLMProvider", "PROVIDER_REGISTRY", "create_provider",
    "create_provider_chain", "detect_provider", "MODEL_PREFIX_MAP",
]


# Provider registry — maps provider name to (module_path, class_name, default_kwargs)
PROVIDER_REGISTRY = {
    "openai": {
        "module": "flyto_ai.providers.openai",
        "class": "OpenAIProvider",
    },
    "anthropic": {
        "module": "flyto_ai.providers.anthropic",
        "class": "AnthropicProvider",
    },
    "ollama": {
        "module": "flyto_ai.providers.ollama",
        "class": "OllamaProvider",
    },
}

# Model name prefix → provider (inspired by claw-code's auto-detection)
MODEL_PREFIX_MAP = {
    "claude-": "anthropic",
    "gpt-": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "chatgpt-": "openai",
    "llama": "ollama",
    "qwen": "ollama",
    "mistral": "ollama",
    "phi": "ollama",
    "gemma": "ollama",
    "codellama": "ollama",
    "deepseek": "openai",  # DeepSeek has OpenAI-compat API; ollama if no key
    "command": "openai",   # Cohere via OpenAI-compat
}


def detect_provider(model: str, api_key: str = "") -> str:
    """Auto-detect provider from model name prefix, with env-var fallback.

    Resolution order:
    1. Model name prefix match (e.g. ``claude-`` → ``anthropic``)
    2. API key env-var probe (``ANTHROPIC_API_KEY`` → ``anthropic``, etc.)
    3. Default: ``"openai"``

    Special case: ``deepseek`` models use OpenAI-compat API when
    ``OPENAI_API_KEY`` or ``DEEPSEEK_API_KEY`` is set, else ``ollama``.
    """
    model_lower = model.lower() if model else ""

    # 1. Prefix match (longest prefix first)
    for prefix, provider in sorted(MODEL_PREFIX_MAP.items(), key=lambda x: -len(x[0])):
        if model_lower.startswith(prefix):
            # Special: deepseek → ollama when no API key available
            if prefix == "deepseek" and not api_key:
                if not os.getenv("OPENAI_API_KEY") and not os.getenv("DEEPSEEK_API_KEY"):
                    return "ollama"
            return provider

    # 2. Env-var probe
    if api_key:
        return "openai"  # caller has a key, assume OpenAI-compat
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"

    # 3. Default
    return "openai"


def create_provider(provider_name: str, **kwargs) -> LLMProvider:
    """Create an LLM provider by name using the registry.

    Falls back to OpenAI for unknown provider names.
    """
    import importlib

    entry = PROVIDER_REGISTRY.get(provider_name, PROVIDER_REGISTRY["openai"])
    mod = importlib.import_module(entry["module"])
    cls = getattr(mod, entry["class"])
    return cls(**kwargs)


def create_provider_chain(configs):
    """Create a ProviderChain from a list of provider config dicts.

    See :func:`flyto_ai.providers.failover.create_provider_chain`.
    """
    from flyto_ai.providers.failover import create_provider_chain as _create
    return _create(configs)
