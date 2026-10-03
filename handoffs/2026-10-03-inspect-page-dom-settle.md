# inspect_page settles on DOM state and reuses the task browser

Owner: claude
Branch: claude/sd-k2
Date: 2026-10-03

## What changed

- `flyto_ai/tools/inspect_page.py`: the fixed `browser.wait {duration_ms:
  min(wait_ms, 5000)}` after `browser.goto` is gone. The extraction script now
  awaits a MutationObserver quiet window (200 ms) inside the page, bounded by
  `wait_ms` (still capped at 5000). Reaching the bound returns the elements
  with `settle.by == "cap"`, not an error. `wait_ms = 0` skips the settle. The
  result carries `data.settle = {by, ms}`. A redirect that destroys the
  script's document re-runs the extraction once on the landed page.
- Inside a caller-owned `browser_session_scope` holding exactly one browser (a
  Space task or scoped chat), `inspect_page` opens a new tab in that browser
  (`browser.tab new`), inspects it, closes that tab and switches back to the
  original index. No cold launch or close. If the tab cannot be opened (dead
  browser, older Core without `browser.tab`) it falls back to the cold
  headless launch. A scope with several browsers, or one that is closing, is
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

## Verified

- `pytest tests/tools/test_inspect_page_settle.py tests/test_inspect_page.py
  tests/test_audit_fixes.py tests/test_sprint1_sprint2.py tests/test_browser_scope.py
  tests/test_auto_discover.py tests/test_browser_retry.py tests/test_complexity_budget.py`:
  191 passed. The settle tests run the real script on headless Chromium with
  `asyncio.sleep` (from inspect_page / Core wait) and `browser.wait` set to fail:
  static page < 300 ms, a page mutating for 400 ms settles at ~400 ms + quiet
  window and captures the late element, a never-settling page returns at the
  bound with elements.
- The new test file fails to import on the previous source.
- `ruff check` on touched files and the CI ruff selection: clean.
- `scripts/generate_reference.py` then `--check`: current.
- `flyto-index verify --strict` in the worktree: no FAIL or WARN.

## Not verified

- `task(action='validate')` through the MCP server (the server is pinned to
  another root this session).
- A live Space task against a signed-in ERP page; the reuse path is proven
  with a Core protocol fake, not a real task browser.

## Follow-ups

- Release flyto-ai, bump the flyto-ai pin in flyto-cloud, then a Desktop
  release (flyto_ai is bundled into Desktop).
