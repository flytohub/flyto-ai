# Advisory freshness job installs the package it tests

Owner: claude
Branch: claude/advisory-freshness-install
Date: 2026-10-08

## What changed

- `.github/workflows/advisory-freshness.yml` installs `./flyto-ai` (base
  dependencies, no extras) plus pytest instead of "pytest only", and drops the
  header's untrue "green when it lands" claim.
- `tests/test_workflow_test_installs.py` (new): every workflow job that runs
  pytest must `pip install` this checkout first. It fails on the old workflow
  (`advisory-freshness.yml:floors-still-clear-every-advisory`) and passes now.

## Why

The scheduled job was red on every run since it landed: pytest loads
`tests/conftest.py`, which imports `flyto_ai` and therefore pydantic, before the
floor test runs. The dependency set lives in `pyproject.toml`; the job now gets
it by installing the checkout rather than restating packages.

With the install fixed, the gate itself is red, correctly: Core published six
advisories on 2026-09-30 against `< 2.33.0` and both declared floors
(`pyproject.toml`, `flyto-blueprint/pyproject.toml`) still say `>=2.31.1`.

## Verified

- Fresh venv, pytest only, against flyto-core@main and flyto-blueprint@main:
  reproduces `ModuleNotFoundError: No module named 'pydantic'`.
- Same venv plus `pip install ./flyto-ai`: conftest loads; the floor test then
  reports both floors predate 2.33.0 (the real finding).
- `ruff check flyto_ai tests` clean; `check_release_drift.py` PASS;
  `flyto-index verify --strict` exit 0, no FAIL/WARN.
- Full suite locally: see the PR description for the exact count.

## Not verified

- The floor raise itself is not in this change. A flyto-blueprint branch
  `claude/core-floor-2-33` (core extra `>=2.33.0`, version 0.3.2) exists
  locally and passes its checks, but every write to the flyto-blueprint
  repository on GitHub (git push, git/blobs API, contents API) returned HTTP 500
  on 2026-10-08, so it could not be pushed.

## Follow-ups

- Once flyto-blueprint accepts writes: push its branch, merge, then in flyto-ai
  raise the three `flyto-core` floors in `pyproject.toml` to `>=2.33.0` and bump
  `stack-lock.json` (flyto-core -> v2.33.0 `940c36862c14d79b94ba5c9040595de5a551def6`,
  flyto-blueprint -> the merged commit). The advisory-freshness job stays red
  until then, by design.
- Coding watchdog: its schedule was disabled on main in 0054d9b (2026-08-23)
  because `FLYTO_CODING_HEARTBEAT` was never published; issue #38 is closed.
  Re-enable only together with installing the publisher.
