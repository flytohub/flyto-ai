# Advisory freshness job installs the package it tests

Owner: claude
Branch: claude/advisory-freshness-install
Date: 2026-10-08

## What changed

- `.github/workflows/advisory-freshness.yml` installs `./flyto-ai` (base
  dependencies, no extras) plus pytest instead of "pytest only", and drops the
  header's untrue "green when it lands" claim.
- Floors: the three `flyto-core` floors in `pyproject.toml` (`browser`,
  `full`, `dev`) are now `>=2.33.0`; `stack-lock.json` pins flyto-core at
  v2.33.0 (`940c36862c14d79b94ba5c9040595de5a551def6`) and flyto-blueprint at
  `be6811f` (flytohub/flyto-blueprint#23, which raised its own `core` extra to
  `>=2.33.0` and moved to 0.3.2, untagged). The `flyto-blueprint>=0.3.1` floor
  is unchanged: 0.3.2 is not on PyPI, and flyto-ai's own Core floor already
  excludes the vulnerable range whenever both are installed.
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
- After the floor raise, with checkouts at the new lock revisions and again
  with flyto-core at `main`: `tests/test_stack_security_floor.py` passes
  (the exact question the scheduled job asks).

## Not verified

- The full suite was not run locally against Core v2.33.0; the PR's CI is the
  check of the lock bump (2.31.3 -> 2.33.0). Locally, against the previous lock,
  70 tests failed only for host reasons (no bare `python` on PATH outside the
  venv for the coding-route subprocess tests, and the sibling Core checkout at
  `main` for the floor test).
- flyto-blueprint 0.3.2 is not published; no tag was pushed.
- GitHub returned HTTP 500 on every write for several minutes on 2026-10-07
  ~16:55 UTC; it recovered on its own.

## Follow-ups

- Publish flyto-blueprint 0.3.2 (owner: tag `v0.3.2`) before raising flyto-ai's
  `flyto-blueprint` floor to it.
- Coding watchdog: its schedule was disabled on main in 0054d9b (2026-08-23)
  because `FLYTO_CODING_HEARTBEAT` was never published; issue #38 is closed.
  Re-enable only together with installing the publisher.
