# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Local Ollama provider (OpenAI-compatible endpoint)."""
from flyto_ai.providers.openai import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    """Ollama provider using the OpenAI-compatible API endpoint."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(
            api_key="ollama",  # Ollama doesn't need a real key
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
        )
