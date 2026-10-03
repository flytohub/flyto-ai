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

## Second review follow-up (same branch)

A second review showed the in-page settle could not see a request the page
started before the script was injected (the usual SPA boot fetch), so a page
with a nav link and no spinner settled at ~200 ms without its form. The first
version of this handoff called that a known limit needing a Core hook "not in
Core today"; that was wrong. Core's `browser.goto` has offered
`wait_until='networkidle'`, and its driver exposes the Playwright page.

- Before the in-page settle, `_wait_network_idle` waits on the session's
  Playwright page with `wait_for_load_state("networkidle", timeout=cap)`.
  Playwright tracks every request of the navigation from its start, so the
  boot fetch is seen. It is feature-detected: a driver without a Playwright
  `page` keeps the in-page settle alone (`settle.network == "unobserved"`).
  `browser.goto(wait_until='networkidle')` was not used because Core's
  driver then also waits up to 5 s for 50 characters of body text, which a
  short login page never has, and a goto timeout would also fail the
  navigation itself.
- Both gates share one `wait_ms` bound. A network that never idles is read
  at the bound with `settle.by == "cap"`, `pending == "network"`, and the
  in-page settle is not paid again. `settle.network` and `settle.network_ms`
  are added to the result.
- The cost: Playwright's idle definition is 500 ms with no request, so a
  static page now takes ~500 ms + one 200 ms quiet window instead of ~200 ms.
  Still well under the old fixed 2 s, and it ends on state.
- `BUSY` now also matches `spin` and `loader` classes (Tailwind
  `animate-spin`, `.loader`).
- The reuse path restores tabs in `finally`: it relists and closes every tab
  at or above the pre-inspection count, highest index first, then switches
  back. An exception from Core after `tab new`, and a `window.open` popup the
  page raised, no longer leave tabs in the operator's browser.

## Verified

- `pytest tests/tools/test_inspect_page_settle.py tests/test_inspect_page.py
  tests/test_audit_fixes.py tests/test_sprint1_sprint2.py tests/test_browser_scope.py
  tests/test_auto_discover.py tests/test_browser_retry.py tests/test_complexity_budget.py
  tests/test_policies.py tests/test_orchestration.py tests/test_agent_stack.py`:
  259 passed.
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
- `ruff check` on touched files and the CI ruff selection: clean.
- `scripts/generate_reference.py` then `--check`: current.
- `flyto-index verify --strict` in the worktree: no FAIL or WARN.

## Not verified

- `task(action='validate')` through the MCP server was not run (the server is
  pinned to another root this session). The repo's ruff, targeted pytest and
  `flyto-index verify --strict` were run instead.
- The full flyto-ai pytest suite was not run, only the targeted files above.
- A live Space task against a signed-in ERP page; the reuse path is proven
  with Core's real tab module on a test Chromium context, not a Cloud task
  browser.

## Follow-ups

- Release flyto-ai, bump the flyto-ai pin in flyto-cloud, then a Desktop
  release (flyto_ai is bundled into Desktop).
