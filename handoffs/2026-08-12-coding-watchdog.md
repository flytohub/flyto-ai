# Host-only coding watchdog and remote dead-man switch

- Date: 2026-08-12
- Owner: claude
- Branch: main (uncommitted working tree)
- Status: implemented and reviewed; automated verification NOT run in this
  session (test execution was refused by the local approval gate)
- **Superseded as the authoritative record by
  `2026-08-12-coding-watchdog-hardening.md`**, which consolidates this session
  and the two hardening passes after it. Read that file first; this one is kept
  for the introduction narrative and the original defect list only, and its
  verification section is now out of date.

## What this adds

`flyto-ai code-watchdog` — an independent, non-AI observer of the coding
control plane, plus `.github/workflows/coding-watchdog.yml` as the off-host
dead-man switch.

- `flyto_ai/coding/watchdog.py` (new): evaluation, transition recorder,
  launchd definition, GitHub heartbeat.
- `flyto_ai/cli.py`: `code-watchdog` subcommand and `_cmd_code_watchdog`.
- `tests/test_coding_watchdog.py` (new), `tests/test_cli.py`.
- Docs: `docs/CODING_CONTROL_PLANE.md`, `docs/architecture-map.md`, and the
  regenerated `docs/reference/**` (via the `generate_reference` project action).

The observer reads only the bounded status index and task window. It invokes no
model and has no submit, audit, abandon, repair, commit or push path. It writes
only `~/.flyto/health/coding/{latest.json,history.jsonl,github.json}`, holding
aggregate counts, stable reason codes, the reader build digest and timestamps.
Alert-only: recovery stays with the existing explicit subtractive commands.

## Defects found and fixed during this review

1. **A hung `gh` skipped the health record.** `publish_github_heartbeat` called
   `subprocess.run(timeout=20)` directly, so `TimeoutExpired` (and `OSError`
   for a bad executable) escaped `run_watchdog_once`'s `except WatchdogError`.
   The turn then ended without writing `latest.json` — the local watchdog went
   silent for the same reason the remote one was needed. Now funnelled through
   `_run_gh`, which converts both to `github_heartbeat_failed`.
2. **`--fail-on-unhealthy` failed a successful `--install`/`--uninstall`.**
   Those reports carry no `health` key, so `report.get("health") != "healthy"`
   was always true and a working install exited 1. Now gated on the observe
   path.
3. **LaunchAgent `PATH` was built from `Path.home().name`**, i.e. it assumed
   `/Users/<login>`. Wrong for a relocated or network home, which would leave
   `gh` unfindable. Now derived from the real `Path.home()`.
4. **`--install --github-heartbeat-interval <60`** installed an agent that
   `run_watchdog_once` rejects on every wake. `launch_agent_definition` now
   applies the same bound.
5. **`WatchdogRecorder.__enter__` reported an unopenable lock file as
   `watchdog_already_running`**, hiding a broken host behind a benign code, and
   relied on catching `UnboundLocalError`. Open and lock are now separate, with
   `watchdog_lock_unavailable` for the former.
6. **`_read_json` called `stat()` before `is_symlink()`**, so the size bound was
   applied to the symlink target rather than the record. Reordered.
7. Removed an unused `os` import in the new test module; dropped a redundant
   second `recorder.previous()` read in `run_watchdog_once` in favour of the
   `previous_health` the recorder already computes.

New tests cover 1, 3 and 4.

## Verification status — READ THIS BEFORE TRUSTING THE ABOVE

- `mcp__flyto-actions__run_project_action generate_reference` — ran, exit 0,
  23 generated reference files rewritten. This is the only command that
  actually executed.
- `pytest` and `ruff` — **NOT RUN.** Every attempted invocation
  (`python -m pytest`, `pytest`, `.venv/bin/pytest`, indexer `task validate`)
  was refused by the local approval gate in this session. The fixes above are
  from static review and are unverified at runtime. Run before committing:

  ```bash
  .venv/bin/python -m pytest tests/test_coding_watchdog.py tests/test_cli.py -q
  .venv/bin/python -m ruff check --select E9,F63,F7,F82 flyto_ai tests
  ```

- The macOS `launchctl` install/uninstall path and the GitHub workflow's live
  behaviour against a real Actions variable have never been exercised.

## Not done

- No commit, no push (explicitly out of scope for this session).
- `flyto_ai/agents/guardian_hook.py` and `tests/test_agents.py` were left
  untouched by instruction; their uncommitted `.lock` extension change belongs
  to whoever made it.
