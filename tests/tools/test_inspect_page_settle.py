# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""inspect_page reads the page when its DOM settles, never after a fixed sleep.

The settle tests drive a real headless Chromium page through the same Core
protocol inspect_page uses, with ``asyncio.sleep`` and ``browser.wait``
patched to fail: the only thing that can end the wait is the page's own
state (load event fired, no request in flight, no busy indicator, controls
present, DOM quiet), or the wait_ms upper bound on a page that never gets
there. The delayed-render tests are the regression for a quiet DOM that is
only a spinner waiting on the network. The reuse tests prove a scoped task
browser is inspected in a new tab instead of a cold launch, including against
Core's real browser.tab module.
"""

import asyncio
import sys
import time

import pytest
import pytest_asyncio

from flyto_ai.tools import core_tools
from flyto_ai.tools import browser_scope
from flyto_ai.tools.browser_scope import BrowserSessionScope
from flyto_ai.tools.inspect_page import INSPECT_PAGE_TOOL, _QUIET_MS, inspect_page

STATIC_PAGE = "<html><body><input id='q' name='q'><button id='go'>Go</button></body></html>"

# Adds a row every 50 ms for 400 ms, then stops; the button only exists at the end.
MUTATES_THEN_STOPS = """
<html><body><ul id='list'></ul><script>
  const started = Date.now();
  const tick = setInterval(() => {
    const li = document.createElement('li');
    li.textContent = 'row ' + (Date.now() - started);
    document.getElementById('list').appendChild(li);
    if (Date.now() - started >= 400) {
      clearInterval(tick);
      const b = document.createElement('button');
      b.id = 'late'; b.textContent = 'Loaded';
      document.body.appendChild(b);
    }
  }, 50);
</script></body></html>
"""

# A spinner with no controls that renders the login form 600 ms later, with
# no DOM change in between: the quiet window passes long before the form.
SPINNER_THEN_FORM = """
<html><body><div id='app'>Loading</div><script>
  setTimeout(() => {
    document.getElementById('app').innerHTML =
      "<input id='email' name='email'><button id='login'>Sign in</button>";
  }, 600);
</script></body></html>
"""

# Navigation is already interactive, so only the progressbar says the page is
# still waiting; the form arrives with a real (routed) API response.
BUSY_UNTIL_API = """
<html><body><a id='home' href='/'>Home</a>
<div id='main'><div role='progressbar' class='spinner'>.</div></div><script>
  fetch('/api/form').then((r) => r.text()).then((html) => {
    document.getElementById('main').innerHTML = html;
  });
</script></body></html>
"""

# No busy indicator at all: a request started after the settle began is the
# only thing that says the page is not done yet.
LATE_REQUEST = """
<html><body><a id='home' href='/'>Home</a><div id='main'></div><script>
  setTimeout(() => {
    fetch('/api/form').then((r) => r.text()).then((html) => {
      document.getElementById('main').innerHTML = html;
    });
  }, 100);
</script></body></html>
"""

# The form is built on the load event, which waits for a slow image.
FORM_ON_LOAD = """
<html><body><a id='home' href='/'>Home</a><img src='/slow.png'><script>
  window.addEventListener('load', () => {
    const input = document.createElement('input');
    input.id = 'email';
    document.body.appendChild(input);
  });
</script></body></html>
"""

FORM_FRAGMENT = "<input id='email' name='email'><button id='login'>Sign in</button>"
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)

# A ticker that never stops mutating the DOM.
NEVER_STOPS = """
<html><body><a id='home' href='/'>Home</a><span id='clock'></span><script>
  setInterval(() => { document.getElementById('clock').textContent = String(Date.now()); }, 30);
</script></body></html>
"""


_REAL_SLEEP = asyncio.sleep


def _forbid_sleep(monkeypatch):
    """Fail any asyncio.sleep issued from inspect_page or Core's browser.wait.

    Playwright's own transport keeps its real sleep, so only a timed wait on
    the inspection path can trip this.
    """
    async def guarded_sleep(delay, *args, **kwargs):
        frame = sys._getframe(1)
        filename = frame.f_code.co_filename.replace("\\", "/")
        if filename.endswith(("flyto_ai/tools/inspect_page.py", "browser/wait.py")):
            raise AssertionError("inspect_page must not sleep; it waits on DOM state")
        return await _REAL_SLEEP(delay, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", guarded_sleep)


def _install_core(monkeypatch, executor):
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: {"execute_module": executor})
    monkeypatch.setattr("flyto_ai.prompt.policies.is_safe_url", lambda _url: True)


class PageBackedCore:
    """Core protocol fake whose evaluate runs on a real Chromium page."""

    def __init__(self, page, html):
        self.page = page
        # None means goto navigates for real, so routed requests can serve
        # the document and its delayed resources.
        self.html = html
        self.calls = []
        self.evaluate_seconds = None

    async def __call__(self, *, module_id, params, browser_sessions, context=None):
        self.calls.append(module_id)
        if module_id == "browser.wait":
            raise AssertionError("browser.wait must not be used to settle the page")
        if module_id == "browser.launch":
            browser_sessions["s1"] = object()
            return {"ok": True}
        if module_id == "browser.goto":
            assert params.get("wait_until") == "domcontentloaded"
            if self.html is None:
                await self.page.goto(params["url"], wait_until="domcontentloaded")
            else:
                await self.page.set_content(self.html, wait_until="domcontentloaded")
            return {"ok": True}
        if module_id == "browser.evaluate":
            started = time.monotonic()
            result = await self.page.evaluate(params["script"])
            self.evaluate_seconds = time.monotonic() - started
            return {"ok": True, "data": {"result": result}}
        return {"ok": True}


@pytest_asyncio.fixture
async def chromium_page():
    playwright_api = pytest.importorskip("playwright.async_api")
    async with playwright_api.async_playwright() as pw:
        try:
            browser = await pw.chromium.launch(headless=True)
        except Exception as exc:  # noqa: BLE001 - no local Chromium means no settle proof here
            pytest.skip("headless Chromium unavailable: {}".format(type(exc).__name__))
        page = await browser.new_page()
        try:
            yield page
        finally:
            await browser.close()


async def _inspect_fixture(monkeypatch, page, html, wait_ms, url="https://example.com"):
    core = PageBackedCore(page, html)
    _install_core(monkeypatch, core)
    _forbid_sleep(monkeypatch)
    result = await inspect_page(url, wait_ms=wait_ms)
    return core, result


async def _serve(page, document, *, delayed=None, delay_s=0.6):
    """Route https://app.test: the document at once, ``delayed`` paths late."""
    delayed = delayed or {}

    async def handle(route):
        path = route.request.url.split("https://app.test", 1)[1]
        if path in delayed:
            body, content_type = delayed[path]
            await asyncio.sleep(delay_s)  # the test's server, not inspect_page
            await route.fulfill(body=body, content_type=content_type)
            return
        await route.fulfill(body=document, content_type="text/html")

    await page.route("https://app.test/**", handle)


def _ids(result):
    return {element.get("id") for element in result["data"]["elements"]}


def test_schema_reads_wait_ms_as_an_upper_bound():
    wait_ms = INSPECT_PAGE_TOOL["inputSchema"]["properties"]["wait_ms"]
    assert wait_ms["default"] == 2000
    assert "Upper bound" in wait_ms["description"]


@pytest.mark.asyncio
async def test_static_page_is_read_after_one_quiet_window(monkeypatch, chromium_page):
    core, result = await _inspect_fixture(monkeypatch, chromium_page, STATIC_PAGE, wait_ms=2000)

    assert result["ok"] is True
    assert "browser.wait" not in core.calls
    ids = {element.get("id") for element in result["data"]["elements"]}
    assert {"q", "go"} <= ids
    assert result["data"]["settle"]["by"] == "quiet"
    # One quiet window, far below the 2000 ms the old fixed wait always paid.
    assert core.evaluate_seconds < 0.3


@pytest.mark.asyncio
async def test_page_that_mutates_then_stops_is_read_when_it_stops(monkeypatch, chromium_page):
    core, result = await _inspect_fixture(monkeypatch, chromium_page, MUTATES_THEN_STOPS, wait_ms=2000)

    assert result["ok"] is True
    settle = result["data"]["settle"]
    assert settle["by"] == "quiet"
    # About 400 ms of mutation plus one quiet window: driven by the last
    # mutation, not by the 2000 ms bound.
    assert 400 <= settle["ms"] < 400 + _QUIET_MS + 300
    assert core.evaluate_seconds < 1.2
    ids = {element.get("id") for element in result["data"]["elements"]}
    assert "late" in ids, "elements rendered by the last mutation must be captured"


@pytest.mark.asyncio
async def test_spinner_without_controls_is_not_mistaken_for_the_page(monkeypatch, chromium_page):
    # The review's probe: the old 200 ms quiet window returned the spinner.
    core, result = await _inspect_fixture(monkeypatch, chromium_page, SPINNER_THEN_FORM, wait_ms=2000)

    assert result["ok"] is True
    assert {"email", "login"} <= _ids(result)
    settle = result["data"]["settle"]
    assert settle["by"] == "quiet", "the form appearing is what ended the wait"
    assert 600 <= settle["ms"] < 600 + _QUIET_MS + 300
    assert "browser.wait" not in core.calls


@pytest.mark.asyncio
async def test_visible_progressbar_holds_the_settle_until_the_api_renders(monkeypatch, chromium_page):
    await _serve(chromium_page, BUSY_UNTIL_API, delayed={"/api/form": (FORM_FRAGMENT, "text/html")})

    _core, result = await _inspect_fixture(
        monkeypatch, chromium_page, None, wait_ms=2000, url="https://app.test/",
    )

    assert result["ok"] is True
    assert {"home", "email", "login"} <= _ids(result)
    settle = result["data"]["settle"]
    assert settle["by"] == "quiet"
    assert settle["ms"] < 600 + _QUIET_MS + 400


@pytest.mark.asyncio
async def test_request_started_while_settling_holds_it_until_it_lands(monkeypatch, chromium_page):
    await _serve(chromium_page, LATE_REQUEST, delayed={"/api/form": (FORM_FRAGMENT, "text/html")})

    _core, result = await _inspect_fixture(
        monkeypatch, chromium_page, None, wait_ms=2000, url="https://app.test/",
    )

    assert result["ok"] is True
    assert {"email", "login"} <= _ids(result), "an in-flight request must not count as settled"
    settle = result["data"]["settle"]
    assert settle["by"] == "quiet"
    assert 600 <= settle["ms"] < 100 + 600 + _QUIET_MS + 400
    # The page's own fetch is handed back once the settle ends.
    assert await chromium_page.evaluate("String(window.fetch).includes('[native code]')")


@pytest.mark.asyncio
async def test_load_event_gates_the_settle(monkeypatch, chromium_page):
    await _serve(chromium_page, FORM_ON_LOAD, delayed={"/slow.png": (_PNG, "image/png")})

    _core, result = await _inspect_fixture(
        monkeypatch, chromium_page, None, wait_ms=2000, url="https://app.test/",
    )

    assert result["ok"] is True
    assert "email" in _ids(result), "the settle must not end before the load event"
    assert result["data"]["settle"]["by"] == "quiet"


@pytest.mark.asyncio
async def test_spinner_that_never_resolves_reports_what_held_it(monkeypatch, chromium_page):
    page = "<html><body><a id='home' href='/'>Home</a><div aria-busy='true'>Loading</div></body></html>"
    _core, result = await _inspect_fixture(monkeypatch, chromium_page, page, wait_ms=600)

    assert result["ok"] is True
    assert result["data"]["settle"]["by"] == "cap"
    assert result["data"]["settle"]["pending"] == "busy"
    assert "home" in _ids(result)


@pytest.mark.asyncio
async def test_page_that_never_settles_returns_elements_at_the_bound(monkeypatch, chromium_page):
    core, result = await _inspect_fixture(monkeypatch, chromium_page, NEVER_STOPS, wait_ms=700)

    assert result["ok"] is True, "reaching wait_ms is a settled page, not an error"
    settle = result["data"]["settle"]
    assert settle["by"] == "cap"
    assert settle["pending"] == "mutating"
    assert 650 <= settle["ms"] < 1000
    assert any(element.get("id") == "home" for element in result["data"]["elements"])


@pytest.mark.asyncio
async def test_wait_ms_zero_skips_the_settle(monkeypatch, chromium_page):
    _core, result = await _inspect_fixture(monkeypatch, chromium_page, NEVER_STOPS, wait_ms=0)

    assert result["ok"] is True
    assert result["data"]["settle"] == {"by": "skipped", "ms": 0}


class RecordingCore:
    """Core protocol fake for the browser-reuse wiring."""

    def __init__(self, *, tab_new_ok=True, tab_new_leaves_tab=False, first_evaluate_navigates=False):
        self.tab_new_ok = tab_new_ok
        # Core's tab.new opens the page and then navigates it; a failed goto
        # leaves that page open.
        self.tab_new_leaves_tab = tab_new_leaves_tab
        self.first_evaluate_navigates = first_evaluate_navigates
        self.tab_count = 2
        self.calls = []

    async def __call__(self, *, module_id, params, browser_sessions, context=None):
        self.calls.append((module_id, dict(params), dict(context or {})))
        if module_id == "browser.wait":
            raise AssertionError("browser.wait must not be used to settle the page")
        if module_id == "browser.launch":
            browser_sessions["cold"] = object()
            return {"ok": True}
        if module_id == "browser.tab":
            action = params["action"]
            if action == "list":
                return {"ok": True, "data": {"tab_count": self.tab_count, "current_index": 1}}
            if action == "new":
                if self.tab_new_leaves_tab:
                    self.tab_count += 1
                    return {"ok": False, "error": "net::ERR_NAME_NOT_RESOLVED at https://erp.example/"}
                if not self.tab_new_ok:
                    return {"ok": False, "error": "Target page, context or browser has been closed"}
                self.tab_count += 1
                return {"ok": True, "data": {"current_index": 2}}
            if action == "close":
                self.tab_count -= 1
            return {"ok": True}
        if module_id == "browser.evaluate":
            if self.first_evaluate_navigates:
                self.first_evaluate_navigates = False
                return {"ok": False, "error": "Execution context was destroyed, most likely because of a navigation"}
            return {"ok": True, "data": {"result": {"url": "https://erp.example/orders", "elements": [{"tag": "a"}]}}}
        return {"ok": True}

    def modules(self):
        return [module for module, _params, _context in self.calls]


@pytest.fixture
def task_scope():
    scope = BrowserSessionScope(owner_id="job-1")
    task_browser = object()
    scope.sessions["workflow-exec-1"] = task_browser
    token = browser_scope._SCOPE.set(scope)
    try:
        yield scope, task_browser
    finally:
        browser_scope._SCOPE.reset(token)


@pytest.mark.asyncio
async def test_task_browser_is_reused_in_a_new_tab_and_restored(monkeypatch, task_scope):
    scope, task_browser = task_scope
    core = RecordingCore()
    _install_core(monkeypatch, core)
    _forbid_sleep(monkeypatch)

    result = await inspect_page("https://erp.example/orders", wait_ms=2000)

    assert result["ok"] is True
    assert result["browser_reused"] is True
    assert result["data"]["url"] == "https://erp.example/orders"
    modules = core.modules()
    assert "browser.launch" not in modules and "browser.close" not in modules
    assert "browser.wait" not in modules
    tab_actions = [params for module, params, _ctx in core.calls if module == "browser.tab"]
    assert tab_actions == [
        {"action": "list"},
        {"action": "new", "url": "https://erp.example/orders"},
        {"action": "close"},
        {"action": "switch", "index": 1},
    ]
    assert all(ctx == {"browser_session": "workflow-exec-1"} for _m, _p, ctx in core.calls)
    # The task's browser stays registered and open for the task's next step.
    assert scope.sessions == {"workflow-exec-1": task_browser}


@pytest.mark.asyncio
async def test_unopenable_task_tab_falls_back_to_a_cold_launch(monkeypatch, task_scope):
    scope, task_browser = task_scope
    core = RecordingCore(tab_new_ok=False)
    _install_core(monkeypatch, core)

    result = await inspect_page("https://example.com", wait_ms=500)

    assert result["ok"] is True
    assert "browser_reused" not in result
    assert "browser.launch" in core.modules() and "browser.close" in core.modules()
    assert scope.sessions == {"workflow-exec-1": task_browser}


@pytest.mark.asyncio
async def test_ambiguous_scope_is_not_reused(monkeypatch, task_scope):
    scope, _task_browser = task_scope
    scope.sessions["second"] = object()
    core = RecordingCore()
    _install_core(monkeypatch, core)

    result = await inspect_page("https://example.com", wait_ms=500)

    assert result["ok"] is True
    assert "browser.tab" not in core.modules()
    assert "browser.launch" in core.modules()


@pytest.mark.asyncio
async def test_redirect_during_settle_reinspects_the_landed_document(monkeypatch):
    core = RecordingCore(first_evaluate_navigates=True)
    _install_core(monkeypatch, core)
    _forbid_sleep(monkeypatch)

    result = await inspect_page("https://example.com", wait_ms=2000)

    assert result["ok"] is True
    assert core.modules().count("browser.evaluate") == 2


@pytest.mark.asyncio
async def test_failed_navigation_closes_its_tab_and_does_not_relaunch(monkeypatch, task_scope):
    scope, task_browser = task_scope
    core = RecordingCore(tab_new_leaves_tab=True)
    _install_core(monkeypatch, core)

    result = await inspect_page("https://erp.example/", wait_ms=500)

    assert result["ok"] is False
    assert "ERR_NAME_NOT_RESOLVED" in result["error"]
    tab_actions = [params for module, params, _ctx in core.calls if module == "browser.tab"]
    assert tab_actions == [
        {"action": "list"},
        {"action": "new", "url": "https://erp.example/"},
        {"action": "list"},
        {"action": "close", "index": 2},
        {"action": "switch", "index": 1},
    ]
    assert core.tab_count == 2, "the tab Core left behind is closed"
    # A cold launch would only repeat the failing navigation.
    assert "browser.launch" not in core.modules()
    assert scope.sessions == {"workflow-exec-1": task_browser}


class _PlaywrightTaskBrowser:
    """The two attributes Core's browser modules read from a task browser."""

    def __init__(self, context, page):
        self._context = context
        self._page = page

    @property
    def page(self):
        return self._page


@pytest.mark.asyncio
async def test_real_core_tab_module_restores_the_operator_page(monkeypatch, chromium_page):
    """Drive Core's own browser.tab and browser.evaluate on a live browser."""
    core_handler = pytest.importorskip("core.mcp_handler")
    core_tab = pytest.importorskip("core.modules.atomic.browser.tab")
    # app.test is a routed test origin with no DNS; the SSRF policy is
    # Core's own concern and is exercised by its suite.
    monkeypatch.setattr(core_tab, "validate_url_with_env_config", lambda url: url)
    # A task browser owns a real context (browser.new_page() gives a closed
    # one that refuses context.new_page()).
    context = await chromium_page.context.browser.new_context()
    chromium_page = await context.new_page()
    await context.route(
        "https://app.test/**",
        lambda route: route.fulfill(body=SPINNER_THEN_FORM, content_type="text/html"),
    )
    await chromium_page.set_content("<html><body><a id='operator'>Signed in</a></body></html>")
    operator_page = chromium_page
    driver = _PlaywrightTaskBrowser(context, operator_page)

    scope = BrowserSessionScope(owner_id="job-1")
    scope.sessions["workflow-exec-1"] = driver
    token = browser_scope._SCOPE.set(scope)
    calls = []

    async def execute(**kwargs):
        calls.append((kwargs["module_id"], kwargs["params"].get("action")))
        assert kwargs["module_id"] not in {"browser.launch", "browser.close", "browser.wait"}
        return await core_handler.execute_module(**kwargs)

    _install_core(monkeypatch, execute)
    _forbid_sleep(monkeypatch)
    try:
        result = await inspect_page("https://app.test/login", wait_ms=2000)
    finally:
        browser_scope._SCOPE.reset(token)

    assert result["ok"] is True, result
    assert result["browser_reused"] is True
    assert {"email", "login"} <= _ids(result)
    assert driver.page is operator_page, "the operator's page is current again"
    assert context.pages == [operator_page], "the inspection tab is closed"
    assert ("browser.tab", "switch") in calls
