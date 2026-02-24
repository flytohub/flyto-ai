# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for browser retry logic (Phase 3)."""
import pytest

from flyto_ai.tools.core_tools import (
    _is_transient_error,
    _is_session_dead,
    _looks_like_module_query,
    dispatch_core_tool,
    _dispatch_core_tool_inner,
)


class TestErrorClassification:
    """Tests for transient/session-dead error classification."""

    def test_timeout_is_transient(self):
        assert _is_transient_error("Navigation timeout of 30000ms exceeded")

    def test_timed_out_is_transient(self):
        assert _is_transient_error("Request timed out after 30s")

    def test_target_closed_is_transient(self):
        assert _is_transient_error("Target closed")

    def test_navigation_failed_is_transient(self):
        assert _is_transient_error("Navigation failed because page crashed")

    def test_context_destroyed_is_transient(self):
        assert _is_transient_error("Execution context was destroyed")

    def test_element_not_found_not_transient(self):
        assert not _is_transient_error("Element not found: #submit-btn")

    def test_invalid_selector_not_transient(self):
        assert not _is_transient_error("Invalid selector: >>css=foo")

    def test_module_not_found_not_transient(self):
        assert not _is_transient_error("Module 'browser.foo' not found")

    def test_target_closed_is_session_dead(self):
        assert _is_session_dead("Target closed")

    def test_session_closed_is_session_dead(self):
        assert _is_session_dead("Session closed")

    def test_browser_disconnected_is_session_dead(self):
        assert _is_session_dead("browser disconnected")

    def test_timeout_is_not_session_dead(self):
        assert not _is_session_dead("Navigation timeout of 30000ms exceeded")


@pytest.mark.asyncio
async def test_transient_error_retried(monkeypatch):
    """browser timeout → retry → succeeds on second attempt."""
    call_count = [0]

    async def mock_execute(module_id, params, context, browser_sessions):
        call_count[0] += 1
        if call_count[0] == 1:
            return {"ok": False, "error": "Navigation timeout of 30000ms exceeded"}
        return {"ok": True, "data": {"text": "Hello"}}

    fake_handler = {
        "execute_module": mock_execute,
        "list_modules": lambda **kw: [],
    }
    monkeypatch.setattr("flyto_ai.tools.core_tools._get_mcp_handler", lambda: fake_handler)

    result = await dispatch_core_tool("execute_module", {
        "module_id": "browser.goto",
        "params": {"url": "https://example.com"},
    })
    assert result["ok"] is True
    assert call_count[0] == 2


@pytest.mark.asyncio
async def test_permanent_error_not_retried(monkeypatch):
    """'element not found' → no retry, return error immediately."""
    call_count = [0]

    async def mock_execute(module_id, params, context, browser_sessions):
        call_count[0] += 1
        return {"ok": False, "error": "Element not found: #submit-btn"}

    fake_handler = {
        "execute_module": mock_execute,
    }
    monkeypatch.setattr("flyto_ai.tools.core_tools._get_mcp_handler", lambda: fake_handler)

    result = await dispatch_core_tool("execute_module", {
        "module_id": "browser.click",
        "params": {"selector": "#submit-btn"},
    })
    assert result["ok"] is False
    assert call_count[0] == 1  # No retry


@pytest.mark.asyncio
async def test_session_dead_relaunches(monkeypatch):
    """'target closed' → relaunch browser → retry → success."""
    calls = []

    async def mock_execute(module_id, params, context, browser_sessions):
        calls.append(module_id)
        if module_id == "browser.launch":
            return {"ok": True, "data": {"session_id": "new-session"}}
        if len(calls) <= 2:
            # First call to browser.extract fails
            return {"ok": False, "error": "Target closed"}
        return {"ok": True, "data": {"text": "Hello"}}

    fake_handler = {
        "execute_module": mock_execute,
    }
    monkeypatch.setattr("flyto_ai.tools.core_tools._get_mcp_handler", lambda: fake_handler)

    result = await dispatch_core_tool("execute_module", {
        "module_id": "browser.extract",
        "params": {"selector": "h1"},
    })
    assert result["ok"] is True
    # Should have: 1) original call, 2) relaunch, 3) retry
    assert "browser.launch" in calls
    assert calls.count("browser.extract") == 2


@pytest.mark.asyncio
async def test_non_browser_not_retried(monkeypatch):
    """string.uppercase error → no retry regardless of error message."""
    call_count = [0]

    async def mock_execute(module_id, params, context, browser_sessions):
        call_count[0] += 1
        return {"ok": False, "error": "timeout processing request"}

    fake_handler = {
        "execute_module": mock_execute,
    }
    monkeypatch.setattr("flyto_ai.tools.core_tools._get_mcp_handler", lambda: fake_handler)

    result = await dispatch_core_tool("execute_module", {
        "module_id": "string.uppercase",
        "params": {"text": "hello"},
    })
    assert result["ok"] is False
    assert call_count[0] == 1  # No retry for non-browser modules


@pytest.mark.asyncio
async def test_relaunch_fail_returns_error(monkeypatch):
    """If relaunch also fails, return the original error."""
    calls = []

    async def mock_execute(module_id, params, context, browser_sessions):
        calls.append(module_id)
        if module_id == "browser.launch":
            return {"ok": False, "error": "Failed to launch browser"}
        return {"ok": False, "error": "Target closed"}

    fake_handler = {
        "execute_module": mock_execute,
    }
    monkeypatch.setattr("flyto_ai.tools.core_tools._get_mcp_handler", lambda: fake_handler)

    result = await dispatch_core_tool("execute_module", {
        "module_id": "browser.goto",
        "params": {"url": "https://example.com"},
    })
    assert result["ok"] is False
    assert "Target closed" in result["error"]
    # Relaunch was attempted
    assert "browser.launch" in calls
    # No retry after failed relaunch — only 1 call to browser.goto
    assert calls.count("browser.goto") == 1


class TestModuleQueryDetection:
    """Tests for _looks_like_module_query — guardrail for web search misuse."""

    def test_dot_notation_is_module(self):
        assert _looks_like_module_query("browser.launch")

    def test_automation_keyword_is_module(self):
        assert _looks_like_module_query("click button")

    def test_resize_image_is_module(self):
        assert _looks_like_module_query("resize image")

    def test_send_email_is_module(self):
        assert _looks_like_module_query("send email notification")

    def test_cjk_person_name_not_module(self):
        assert not _looks_like_module_query("周杰倫")

    def test_cjk_search_query_not_module(self):
        assert not _looks_like_module_query("搜尋周杰倫")

    def test_english_person_not_module(self):
        assert not _looks_like_module_query("Elon Musk")

    def test_news_topic_not_module(self):
        assert not _looks_like_module_query("latest AI news")

    def test_empty_query_not_module(self):
        assert not _looks_like_module_query("")

    def test_mixed_cjk_with_technical_is_module(self):
        assert _looks_like_module_query("screenshot 截圖")

    def test_parse_json_is_module(self):
        assert _looks_like_module_query("parse json")


class TestBrowserCascadeBreaker:
    """Tests for browser cascade breaker — skip browser.* after launch fails."""

    @pytest.mark.asyncio
    async def test_cascade_blocks_goto_after_launch_fail(self, monkeypatch):
        """browser.goto should be short-circuited after browser.launch fails."""
        import flyto_ai.tools.core_tools as ct

        async def mock_execute(module_id, params, context=None, browser_sessions=None):
            if module_id == "browser.launch":
                return {"ok": False, "error": "chromium not installed"}
            return {"ok": True, "data": {}}

        fake_handler = {
            "execute_module": mock_execute,
        }
        monkeypatch.setattr(ct, "_get_mcp_handler", lambda: fake_handler)

        # Reset cascade state
        ct._browser_launch_failed = False
        ct._browser_launch_error = ""

        # browser.launch fails
        result = await ct._dispatch_core_tool_inner(
            "execute_module", {"module_id": "browser.launch", "params": {}},
        )
        assert result["ok"] is False
        assert ct._browser_launch_failed is True

        # browser.goto should be blocked immediately
        result = await ct._dispatch_core_tool_inner(
            "execute_module", {"module_id": "browser.goto", "params": {"url": "https://google.com"}},
        )
        assert result["ok"] is False
        assert "browser.launch failed earlier" in result["error"]

        # browser.snapshot should also be blocked
        result = await ct._dispatch_core_tool_inner(
            "execute_module", {"module_id": "browser.snapshot", "params": {}},
        )
        assert result["ok"] is False
        assert "browser.launch failed earlier" in result["error"]

        # Cleanup
        ct._browser_launch_failed = False
        ct._browser_launch_error = ""

    @pytest.mark.asyncio
    async def test_cascade_resets_on_new_launch(self, monkeypatch):
        """A new browser.launch attempt resets the cascade flag."""
        import flyto_ai.tools.core_tools as ct

        call_count = 0

        async def mock_execute(module_id, params, context=None, browser_sessions=None):
            nonlocal call_count
            call_count += 1
            if module_id == "browser.launch":
                if call_count == 1:
                    return {"ok": False, "error": "first attempt fails"}
                return {"ok": True, "data": {"session_id": "s1"}}
            return {"ok": True, "data": {}}

        fake_handler = {
            "execute_module": mock_execute,
        }
        monkeypatch.setattr(ct, "_get_mcp_handler", lambda: fake_handler)
        ct._browser_launch_failed = False
        ct._browser_launch_error = ""

        # First launch fails
        await ct._dispatch_core_tool_inner(
            "execute_module", {"module_id": "browser.launch", "params": {}},
        )
        assert ct._browser_launch_failed is True

        # Second launch resets and succeeds
        result = await ct._dispatch_core_tool_inner(
            "execute_module", {"module_id": "browser.launch", "params": {}},
        )
        assert result["ok"] is True
        assert ct._browser_launch_failed is False

        # browser.goto should now work
        result = await ct._dispatch_core_tool_inner(
            "execute_module", {"module_id": "browser.goto", "params": {"url": "https://google.com"}},
        )
        assert result["ok"] is True

        # Cleanup
        ct._browser_launch_failed = False
        ct._browser_launch_error = ""

    @pytest.mark.asyncio
    async def test_cascade_does_not_affect_non_browser(self, monkeypatch):
        """Non-browser execute_module calls should not be affected by cascade."""
        import flyto_ai.tools.core_tools as ct

        async def mock_execute(module_id, params, context=None, browser_sessions=None):
            return {"ok": True, "data": {"result": "hello"}}

        fake_handler = {
            "execute_module": mock_execute,
        }
        monkeypatch.setattr(ct, "_get_mcp_handler", lambda: fake_handler)

        # Set cascade flag manually
        ct._browser_launch_failed = True
        ct._browser_launch_error = "chromium not installed"

        # Non-browser module should still work
        result = await ct._dispatch_core_tool_inner(
            "execute_module", {"module_id": "string.uppercase", "params": {"text": "hello"}},
        )
        assert result["ok"] is True

        # Cleanup
        ct._browser_launch_failed = False
        ct._browser_launch_error = ""


@pytest.mark.asyncio
async def test_search_modules_web_guardrail(monkeypatch):
    """search_modules with web search query and 0 results adds hint."""

    def mock_search(query, category=None, limit=20):
        return {"query": query, "total": 0, "results": []}

    fake_handler = {
        "search_modules": mock_search,
    }
    monkeypatch.setattr("flyto_ai.tools.core_tools._get_mcp_handler", lambda: fake_handler)

    result = await _dispatch_core_tool_inner("search_modules", {"query": "周杰倫"})
    assert "web_search_hint" in result
    assert "Browser Protocol" in result["web_search_hint"]


@pytest.mark.asyncio
async def test_search_modules_no_guardrail_for_module_query(monkeypatch):
    """search_modules with module-like query and 0 results does NOT add hint."""

    def mock_search(query, category=None, limit=20):
        return {"query": query, "total": 0, "results": []}

    fake_handler = {
        "search_modules": mock_search,
    }
    monkeypatch.setattr("flyto_ai.tools.core_tools._get_mcp_handler", lambda: fake_handler)

    result = await _dispatch_core_tool_inner("search_modules", {"query": "click button"})
    assert "web_search_hint" not in result
