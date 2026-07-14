# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Shared fixtures for flyto-ai tests."""
import asyncio

import pytest


@pytest.fixture(autouse=True)
def _ensure_legacy_event_loop():
    """Keep legacy unittest-style tests compatible with Python 3.11+."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
