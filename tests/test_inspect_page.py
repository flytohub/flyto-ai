# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Focused contract and fallback tests for the inspect_page capability."""

import pytest

from flyto_ai.tools import core_tools
from flyto_ai.tools.inspect_page import INSPECT_PAGE_TOOL, inspect_page


class FakeCoreExecutor:
    """Small Core protocol fake that records browser module calls."""

    def __init__(self, bundled_launch_ok: bool = False) -> None:
        self.bundled_launch_ok = bundled_launch_ok
        self.calls = []

    async def __call__(self, *, module_id, params, browser_sessions):
        self.calls.append((module_id, dict(params)))
        if module_id == "browser.launch":
            if params.get("channel") == "chrome" or self.bundled_launch_ok:
                browser_sessions["default"] = object()
                return {"ok": True}
            return {"ok": False, "error": "bundled Chromium is unavailable"}
        if module_id == "browser.evaluate":
            return {
                "ok": True,
                "data": {"result": {"url": "https://example.com", "elements": []}},
            }
        return {"ok": True}


def _install_core(monkeypatch, executor):
    monkeypatch.setattr(
        core_tools,
        "_get_mcp_handler",
        lambda: {"execute_module": executor},
    )
    monkeypatch.setattr("flyto_ai.prompt.policies.is_safe_url", lambda _url: True)


def test_schema_declares_typed_browser_channel():
    channel = INSPECT_PAGE_TOOL["inputSchema"]["properties"]["browser_channel"]
    assert channel["enum"] == ["auto", "chromium", "chrome", "msedge"]
    assert channel["default"] == "auto"


@pytest.mark.asyncio
async def test_auto_falls_back_to_installed_chrome_and_reports_evidence(monkeypatch):
    executor = FakeCoreExecutor()
    _install_core(monkeypatch, executor)

    result = await inspect_page("https://example.com", wait_ms=0)

    assert result["ok"] is True
    assert result["browser_channel"] == "chrome"
    launches = [params for module, params in executor.calls if module == "browser.launch"]
    assert launches == [{"headless": True}, {"headless": True, "channel": "chrome"}]
    assert ("browser.close", {}) in executor.calls


@pytest.mark.asyncio
async def test_explicit_chromium_fails_closed_without_channel_fallback(monkeypatch):
    executor = FakeCoreExecutor()
    _install_core(monkeypatch, executor)

    result = await inspect_page(
        "https://example.com",
        wait_ms=0,
        browser_channel="chromium",
    )

    assert result["ok"] is False
    assert "bundled Chromium is unavailable" in result["error"]
    launches = [params for module, params in executor.calls if module == "browser.launch"]
    assert launches == [{"headless": True}]


@pytest.mark.asyncio
async def test_unknown_browser_channel_is_rejected_before_core_dispatch(monkeypatch):
    executor = FakeCoreExecutor()
    _install_core(monkeypatch, executor)

    result = await inspect_page("https://example.com", browser_channel="firefox")

    assert result["ok"] is False
    assert "browser_channel must be one of" in result["error"]
    assert executor.calls == []
