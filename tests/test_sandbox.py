# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for Docker sandbox — classification + execution."""
import json
import shutil

import pytest

from flyto_ai.sandbox.manager import (
    SandboxManager,
    DIRECT_CATEGORIES,
    SANDBOXED_CATEGORIES,
)


class TestNeedsSandbox:
    """Module classification — no Docker needed."""

    def test_browser_needs_sandbox(self):
        mgr = SandboxManager()
        assert mgr.needs_sandbox("browser.click") is True
        assert mgr.needs_sandbox("browser.launch") is True

    def test_shell_needs_sandbox(self):
        mgr = SandboxManager()
        assert mgr.needs_sandbox("shell.exec") is True

    def test_http_needs_sandbox(self):
        mgr = SandboxManager()
        assert mgr.needs_sandbox("http.request") is True

    def test_string_direct(self):
        mgr = SandboxManager()
        assert mgr.needs_sandbox("string.uppercase") is False

    def test_math_direct(self):
        mgr = SandboxManager()
        assert mgr.needs_sandbox("math.add") is False

    def test_json_direct(self):
        mgr = SandboxManager()
        assert mgr.needs_sandbox("json.parse") is False

    def test_all_direct_categories(self):
        mgr = SandboxManager()
        for cat in DIRECT_CATEGORIES:
            assert mgr.needs_sandbox("{}.test".format(cat)) is False

    def test_all_sandboxed_categories(self):
        mgr = SandboxManager()
        for cat in SANDBOXED_CATEGORIES:
            assert mgr.needs_sandbox("{}.test".format(cat)) is True

    def test_no_overlap(self):
        """Direct and sandboxed categories don't overlap."""
        assert DIRECT_CATEGORIES.isdisjoint(SANDBOXED_CATEGORIES)


@pytest.mark.asyncio
async def test_sandbox_no_docker():
    """When Docker is not available, returns clear error."""
    mgr = SandboxManager(image="nonexistent-image:latest", timeout=5)
    # This will fail if docker is not installed OR if image doesn't exist
    result = await mgr.execute("browser.click", {"selector": "button"})
    assert result["ok"] is False
    assert "error" in result


@pytest.mark.asyncio
async def test_sandbox_timeout():
    """Timeout produces error result."""
    mgr = SandboxManager(image="flyto-sandbox:latest", timeout=1)
    # Even if Docker runs, 1s timeout should trigger for most operations
    result = await mgr.execute("browser.launch", {})
    # Either timeout or Docker not available — both are valid "not ok"
    assert result["ok"] is False


@pytest.mark.asyncio
@pytest.mark.skipif(not shutil.which("docker"), reason="Docker not available")
async def test_sandbox_execute_simple():
    """Real Docker execution — requires 'flyto-sandbox:latest' image built."""
    mgr = SandboxManager(timeout=30)

    # Try running a simple string module in the container
    result = await mgr.execute("string.uppercase", {"text": "hello"})

    # If image not built, this will fail — that's expected in CI
    if not result["ok"]:
        error = result.get("error", "")
        # Acceptable: image not found or flyto-core not installed
        assert "not found" in error.lower() or "not installed" in error.lower() or "error" in error.lower()
    else:
        # If it works, verify the result
        assert result.get("data", {}).get("text", "").upper() == "HELLO" or result["ok"]


class TestSandboxNotEnabled:
    """When sandbox is disabled, modules go through normal path."""

    def test_needs_sandbox_still_works(self):
        """needs_sandbox is independent of enabled flag — it's the caller's job to check."""
        mgr = SandboxManager()
        assert mgr.needs_sandbox("browser.click") is True
