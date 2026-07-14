# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Action Assistant — system-level intelligence layer for the AI agent.

Single entry point: ``AssistantMiddleware`` wraps the raw tool dispatch
and transparently adds blueprint routing, interactive input, selector
healing, and learning.  The Agent only needs one line to integrate:

    dispatch_fn = self._assistant.wrap(self._dispatch_fn)

Architecture::

    User message
        ↓
    AssistantMiddleware.prepare(message)    ← pre-resolve blueprint
        ↓
    LLM decides tool call
        ↓
    AssistantMiddleware.wrap(dispatch)      ← intercept & enhance
        ├── Blueprint guard (redirect to use_blueprint)
        ├── Selector healing (auto-fix broken CSS)
        └── Injection scanning (existing)
        ↓
    Tool result
        ↓
    AssistantMiddleware.post_process(tool_calls, results)
        ├── Blueprint feedback (score +5/-10, auto-learn)
        ├── Pending input detection (ask_user marker)
        └── Preference learning

Submodules:
    router       — Blueprint-first routing + guard
    interactive  — ask_user pending input detection
    resilience   — Selector healing for browser modules
"""
from flyto_ai.assistant.middleware import AssistantMiddleware

__all__ = ["AssistantMiddleware"]
