# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
from flyto_ai.providers.base import LLMProvider

__all__ = ["LLMProvider", "PROVIDER_REGISTRY", "create_provider"]


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


def create_provider(provider_name: str, **kwargs) -> LLMProvider:
    """Create an LLM provider by name using the registry.

    Falls back to OpenAI for unknown provider names.
    """
    import importlib

    entry = PROVIDER_REGISTRY.get(provider_name, PROVIDER_REGISTRY["openai"])
    mod = importlib.import_module(entry["module"])
    cls = getattr(mod, entry["class"])
    return cls(**kwargs)
