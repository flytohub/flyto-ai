# inspect_page settles on DOM state and reuses the task browser

Owner: claude
Branch: claude/sd-k2
Date: 2026-10-03

## What changed

- `flyto_ai/tools/inspect_page.py`: the fixed `browser.wait {duration_ms:
  min(wait_ms, 5000)}` after `browser.goto` is gone. The extraction script now
  settles on a conjunction of page state inside the page: load event fired,
  no fetch/XHR started during the settle still in flight, no visible busy
  indicator (`aria-busy`, progressbar, spinner/skeleton/loading classes), at
  least one interactive element, and then a 200 ms quiet window. DOM
  mutations, finished resources (PerformanceObserver), request start/end and
  the load event restart the window; a quiet window that ends while a signal
  is pending waits for the next state change instead of re-arming on a clock.
  It is bounded by `wait_ms` (still capped at 5000). Reaching the bound
  returns the elements with `settle.by == "cap"` and `settle.pending` naming
  the signal that held it, not an error. The fetch/XHR wrappers are restored
  when the settle ends. `wait_ms = 0` skips the settle. The
  result carries `data.settle = {by, ms}`. A redirect that destroys the
  script's document re-runs the extraction once on the landed page.
- Inside a caller-owned `browser_session_scope` holding exactly one browser (a
  Space task or scoped chat), `inspect_page` opens a new tab in that browser
  (`browser.tab new`), inspects it, closes that tab and switches back to the
  original index. No cold launch or close. If the tab cannot be opened (dead
  browser, older Core without `browser.tab`) it falls back to the cold
  headless launch. If Core opened the tab but its navigation failed, the
  tab-count growth identifies the leaked tab, which is closed by index; the
  original tab is switched back and the navigation error is returned without
  a cold relaunch (which would only repeat the failing goto). A scope with several browsers, or one that is closing, is
  never reused. The scope stays caller authority, never a model parameter.
- Schema: `wait_ms` keeps its name and default (2000); its description now
  reads as an upper bound. The tool description mentions the reuse.
- `tests/tools/test_inspect_page_settle.py` (new), `docs/CODING_CONTROL_PLANE.md`,
  regenerated `docs/reference/`.

## Why

Every inspect_page call paid a flat 2 s (up to 5 s) sleep plus a cold browser
launch and close, and could not see pages behind the operator's login.
`navigator.py`'s `sleep(1)` is intentionally untouched: it is not exposed to
Space tasks.

## Review follow-up (same branch)

The first version settled on DOM quiet alone. A review probe showed a spinner
page that renders its form 600 ms later, with no DOM change in between, came
back as the spinner. The conjunction above fixes that; the probe is now a
regression test.

Known limit: a request the page started *before* the settle began (in-page
JS cannot see an in-flight request it did not wrap) on a page that already
has controls and shows no busy indicator can still settle early. Spinners,
skeletons, `aria-busy`, a page with no controls yet, and chained requests are
all covered.

## Verified

- `pytest tests/tools/test_inspect_page_settle.py tests/test_inspect_page.py
  tests/test_audit_fixes.py tests/test_sprint1_sprint2.py tests/test_browser_scope.py
  tests/test_auto_discover.py tests/test_browser_retry.py tests/test_complexity_budget.py
  tests/test_policies.py tests/test_orchestration.py tests/test_agent_stack.py`:
  259 passed; the settle file alone (16 tests) passed three runs in a row.
  The settle tests run the real script on headless Chromium with
  `asyncio.sleep` (from inspect_page / Core wait) and `browser.wait` set to fail:
  static page < 300 ms; a page mutating for 400 ms settles at ~400 ms + quiet
  window; a no-controls spinner that renders a form at 600 ms, a progressbar
  page waiting on a routed 600 ms API, a page that starts a fetch 100 ms into
  the settle, and a form built on a load event delayed by a slow image all
  return the real form with `settle.by == "quiet"`; a never-settling page
  returns at the bound with `pending`.
- One reuse test drives Core's real `browser.tab` and `browser.evaluate`
  through `core.mcp_handler.execute_module` on a live Chromium context: the
  inspection tab is closed and the operator's page is current again.
- A failed navigation in the task tab closes the leaked tab by index and does
  not cold-launch (Core protocol fake).
- 8 of the 16 tests fail against the first commit's source.
- `ruff check` on touched files and the CI ruff selection: clean.
- `scripts/generate_reference.py` then `--check`: current.
- `flyto-index verify --strict` in the worktree: no FAIL or WARN.

## Not verified

- `task(action='validate')` through the MCP server (the server is pinned to
  another root this session).
- A live Space task against a signed-in ERP page; the reuse path is proven
  with Core's real tab module on a test Chromium context, not a Cloud task
  browser.

## Follow-ups

- Release flyto-ai, bump the flyto-ai pin in flyto-cloud, then a Desktop
  release (flyto_ai is bundled into Desktop).
