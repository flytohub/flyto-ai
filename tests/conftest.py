# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Shared fixtures for flyto-ai tests."""
import asyncio
import os
import tempfile

import pytest

from flyto_ai.coding.workspace_authority import WORKSPACE_REGISTRY_ENV


@pytest.fixture(autouse=True, scope="session")
def _isolated_workspace_authority_registry():
    """Point every test at a private host-global workspace registry.

    The registry is host-global by design: that is the whole point of it, and
    it is what stops two state roots editing one tree. A test suite must
    therefore never use the real one, or runs would contend with a developer's
    own services and with each other.

    Session-scoped and set through the documented startup-only environment
    override, so subprocesses spawned by the multiprocess tests inherit it and
    take part in the same isolated registry rather than silently falling back
    to the host default.
    """

    with tempfile.TemporaryDirectory(prefix="coding-workspace-authority-") as root:
        previous = os.environ.get(WORKSPACE_REGISTRY_ENV)
        # `realpath`, because the registry walk refuses a symlinked component
        # and the platform temp directory is itself a link on macOS.
        os.environ[WORKSPACE_REGISTRY_ENV] = os.path.join(
            os.path.realpath(root), "registry",
        )
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(WORKSPACE_REGISTRY_ENV, None)
            else:
                os.environ[WORKSPACE_REGISTRY_ENV] = previous


@pytest.fixture(autouse=True)
def _ensure_legacy_event_loop():
    """Keep legacy unittest-style tests compatible with Python 3.11+."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    if not loop.is_closed():
        loop.close()
    asyncio.set_event_loop(None)
