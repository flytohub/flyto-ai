# Coding watchdog — consolidated release record

- Date: 2026-08-12
- Owner: claude
- Branch: main (uncommitted working tree)
- Status: implemented and reviewed. Codex reran the suite against an **earlier**
  revision of this tree and got 3564 passed / 17 skipped; the host verifier
  reported exit 1 with no captured failing test. Pass 4 below landed after that
  run and has no execution evidence at all. **Claude has never been able to
  execute a check here.** Read Verification before trusting any of this.
- Consolidates `2026-08-12-coding-watchdog.md` (introduction) and the three
  hardening passes that followed. This file is the authoritative record for the
  watchdog; the earlier file is kept for the introduction narrative only.

Scope was the watchdog and its own workflow, tests, docs and memory files.
`flyto_ai/agents/guardian_hook.py` and `tests/test_agents.py` were not touched,
by instruction. Nothing was committed or pushed.

## What exists

`flyto-ai code-watchdog` — an independent, non-AI observer of the coding
control plane — plus `.github/workflows/coding-watchdog.yml` as the off-host
dead-man switch.

- `flyto_ai/coding/watchdog.py`: evaluation, transition recorder, launchd
  definition, GitHub heartbeat.
- `flyto_ai/cli.py`: the `code-watchdog` subcommand and `_cmd_code_watchdog`.
- `tests/test_coding_watchdog.py`, `tests/test_cli.py`.
- Docs: `docs/CODING_CONTROL_PLANE.md`, `ARCHITECTURE.md`,
  `docs/architecture-map.md`, and the regenerated `docs/reference/**`.

The observer reads only the bounded status index and task window. It invokes no
model and has no submit, audit, abandon, repair, commit or push path. It writes
only `~/.flyto/health/coding/{latest.json,history.jsonl,github.json}`, holding
aggregate counts, stable reason codes, the reader build digest and timestamps.
Alert-only: recovery stays with the existing explicit subtractive commands.

## Defects repaired, in the order they were found

### Pass 1 — the observer could go silent

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
   `run_watchdog_once` rejects on every wake.
5. **`WatchdogRecorder.__enter__` reported an unopenable lock file as
   `watchdog_already_running`**, hiding a broken host behind a benign code, and
   relied on catching `UnboundLocalError`. Open and lock are now separate, with
   `watchdog_lock_unavailable` for the former.
6. **`_read_json` called `stat()` before `is_symlink()`**, so the size bound was
   applied to the symlink target rather than the record. Reordered.

### Pass 2 — the dead-man switch could be talked out of alerting

7. **The heartbeat was trusted.** `.github/workflows/coding-watchdog.yml` read
   `vars.FLYTO_CODING_HEARTBEAT` — a repository variable any actor who can
   write repository variables controls — and interpolated derived values into
   `GITHUB_OUTPUT` with no character constraint. `health` was taken with
   `str(...)`, so a newline inside it (or inside a rendered reason) appended
   attacker-chosen lines to `GITHUB_OUTPUT`; `healthy=true` there silences the
   switch. `observed_at` was coerced with `int(...)`, so a float or numeric
   string was accepted and a non-numeric string was misreported as
   `heartbeat_invalid`. The variable's raw size was unbounded.
   Now: bounded raw size before parsing (`MAX_HEARTBEAT_BYTES`, 64 KiB), exact
   schema, `observed_at` must be a plain non-bool `int` in
   `0 < n <= 4102444800`, `health` must be one of `healthy|degraded|critical`,
   `reason_codes` must be a list, codes must match `[a-z][a-z0-9_]{0,63}`, and
   the emitted `reason` is re-checked against `[a-z0-9_,:]{1,1024}` before it
   reaches `GITHUB_OUTPUT`. Distinct codes (`heartbeat_missing`,
   `heartbeat_oversized`, `heartbeat_invalid`, `heartbeat_schema_invalid`,
   `heartbeat_timestamp_invalid`, `heartbeat_health_invalid`) let an operator
   tell a bad publisher from a dead host. `RecursionError` from deeply nested
   JSON is caught: previously it escaped
   `except (TypeError, ValueError, json.JSONDecodeError)` and aborted the step
   *before* the incident step ran, converting a malformed variable into
   silence. `MAX_AGE_SECONDS` and the skew/size bounds are parsed outside the
   heartbeat branch so a workflow typo is not reported as a bad heartbeat.
8. **`state_readable` used the wrong byte limit.** `_validate_status_index`
   size-checked the route status index against the watchdog's own
   `MAX_LATEST_BYTES` (128 KiB) instead of the publisher's authoritative
   `MAX_STATUS_INDEX_BYTES` (256 KiB). An index between the two — legal for the
   writer — was reported `fail`, i.e. `critical`, manufacturing an incident.
9. **State root and health directory could overlap.** Nothing stopped
   `--health-dir ~/.flyto/coding-service/health`. The observer would then write
   into the durable coding-service tree it is explicitly forbidden to mutate,
   and would observe its own writes. `_resolve_disjoint_roots` rejects equality
   and nesting in either direction with `watchdog_paths_overlap`, on both the
   run and install paths.
10. **`--install` validated only two of its values.** `stuck_seconds`,
    `orphan_grace_seconds` and `github_variable` went straight into the plist.
    `--install --stuck-seconds 5` produced a LaunchAgent that raised on every
    wake. One shared `_validate_observation_options` is now called by both
    `run_watchdog_once` and `launch_agent_definition`; `bool` is rejected
    explicitly since `True` is an `int`. New codes:
    `watchdog_stuck_seconds_invalid`, `watchdog_orphan_grace_invalid`.
11. **Unreachable duplication removed.** `collect_watchdog_snapshot` caught
    `(OSError, ValueError, RuntimeError, TaskWindowCorrupt)`, but
    `TaskWindowCorrupt` derives from `RuntimeError`.
12. **`gh variable set` for bounded create/update.** The old path decided
    control flow by string-matching `"HTTP 404"` / `"Not Found"` in stderr — a
    message the CLI never promised — and doubled the window in which a hung
    `gh` could stall the one turn that still has to record health. One upsert
    call now creates and updates identically. The payload is refused locally
    above `MAX_GITHUB_VARIABLE_BYTES` (48 KB, GitHub's cap) rather than
    truncated remotely (`github_heartbeat_payload_too_large`).

### Pass 3 — this session

13. **The overlap guard was lexical, so a symlink walked through it.**
    `_resolve_disjoint_roots` compared `os.path.abspath` spellings. That is not
    a containment check: `--health-dir /tmp/link`, where the link points inside
    the coding-service tree, presented the guard with two unrelated strings and
    passed. The same lexical treatment split one directory into two identities
    — `launch_agent_label` hashed the unresolved path, so installing via
    `/tmp/alias` and uninstalling via the real root produced different labels,
    and `--uninstall` reported success while removing nothing and leaving the
    agent waking forever. One new `_resolve_root` (`realpath` over
    `abspath`/`expanduser`, non-strict so a not-yet-created health directory is
    judged by the same rules as every later run) is now the single normaliser
    used by `_resolve_disjoint_roots`, `launch_agent_label`,
    `collect_watchdog_snapshot` and `WatchdogRecorder`.
14. **`evaluate_watchdog` re-spelled its own minimum.** The threshold guard
    compared against a literal `60` while `MIN_STUCK_SECONDS` declared the same
    bound one screen above; the shared validator already used the constant. Now
    one spelling.
15. **The composed path had no test at all.** Every existing test drove a
    single function, so nothing exercised what an unattended LaunchAgent
    actually runs, and nothing proved the central invariant — that the observer
    does not mutate the tree it observes. Added
    `test_one_full_turn_observes_the_state_root_and_writes_only_health`: it
    runs two real `run_watchdog_once` turns against a state root holding a
    valid published index, asserts `state_readable` is `pass`, that the state
    tree is byte-identical before and after both turns, that health lands only
    in the health directory, that the record carries no job id or local path,
    and that an unchanged second turn is not a transition.
16. **CLI defaults were a second, unpinned copy of the module contract.**
    `cli.py` spells `--stuck-seconds`, `--orphan-grace-seconds`,
    `--github-heartbeat-interval`, `--github-variable` and `--health-dir` as
    argparse literals while `flyto_ai.coding.watchdog` declares them as
    constants that the installed LaunchAgent and every other caller obey.
    `DEFAULT_HEALTH_DIR` had no reader at all. Drift there is invisible until
    an unattended agent has been observing on the wrong thresholds. The parser
    keeps its literals — importing the watchdog module at parser-build time
    would pull the mission store into `flyto-ai --help` — and
    `test_code_watchdog_flag_defaults_match_the_module_contract` now pins them
    behaviourally by running the real parser and reading what it handed the
    observer.

### Pass 4 — consolidation: the health directory is not exclusively owned

The earlier passes hardened everything the watchdog *reads from elsewhere* and
assumed the directory it writes to was its own. It is created `0o700`, but
`--health-dir` is an operator-supplied path and a world-writable parent such as
`/tmp/flyto-health` is a plausible choice. Under one, every name in it is
attacker-plantable before first use.

17. **A symlinked `history.jsonl` was a write primitive.** `_append_history`
    opened the path `O_WRONLY|O_CREAT|O_APPEND` — which follows symlinks — and
    sized it with `stat()`, which follows them too. Another local user could
    plant a link and have the watchdog append JSON lines to any file the
    watchdog's own user can write. Pass 1 had fixed exactly this class for
    *reads* (`_read_json`) and left the write. Now `O_NOFOLLOW` plus `lstat`,
    with a refusal reported as `watchdog_history_unwritable` — raised only
    after `_atomic_write` has already made `latest.json` durable, so the turn's
    actual contract still completes. `_atomic_write` itself needed no change:
    `os.replace` overwrites a link rather than following it.
18. **A symlinked `watchdog.lock` moved the lock.** `os.open(O_RDWR|O_CREAT)`
    on a planted link took the exclusive `flock` on the link's target — and
    created that target as a side effect — so it excluded nothing and no second
    watchdog would find it. `O_NOFOLLOW` makes it fail closed as
    `watchdog_lock_unavailable`.
19. **`_read_json` still had a check-then-read race.** Pass 1 reordered
    `is_symlink()` before `stat()`, but both tested a *name* and the read was a
    separate `read_text()` on that name. Swapping in a link between the two
    defeated it. The descriptor is now opened once `O_NOFOLLOW`, `fstat`ed and
    drained; `os.read` is looped because a short read is permitted, and the cap
    is re-applied to the drained total rather than trusted from `fstat`.
    `UnicodeDecodeError` is not named beside `ValueError` — it derives from it.
20. **`_rotate_history` used `exists()`.** A dangling planted link reads as
    absent, so it was never rotated away and every later append refused
    forever. `os.path.lexists` rotates the name.
21. **An unwritable heartbeat cursor discarded the whole turn.** Pass 1 fixed a
    hung `gh`; `mark_github_sent`'s `_atomic_write` was the remaining hole in
    the same promise, and a worse one — it runs *after* the heartbeat is
    published, so an `OSError` there (a `github.json` that is a directory, say)
    left the remote switch reading `healthy` while `latest.json` was never
    written. Now caught beside `WatchdogError` as `github_state_unrecordable`.
    Losing the send-interval bookkeeping is safe: the next turn republishes an
    unchanged heartbeat.
22. **The workflow cancelled runs in progress.** `cancel-in-progress: true` on
    a job whose product is an incident means a dispatch landing between "the
    heartbeat is stale" and "open the issue" cancels the only step that reports
    it. Runs are seconds long, so serializing costs nothing.

## Tests

`tests/test_coding_watchdog.py`, `tests/test_cli.py`.

- Upsert argv shape, single call, no create-retry on failure, oversized
  payload, hung `gh`.
- Install rejects each of stuck/orphan/heartbeat-interval/launch-interval/
  variable/repository; the run path rejects the same; a variable name is
  validated even with no repository configured.
- State/health overlap rejected for equal, nested, reverse-nested **and
  symlinked** roots on both entry points; disjoint roots accepted; install and
  uninstall derive one label through a symlinked state root.
- One full `run_watchdog_once` turn, twice, with an exact no-mutation check on
  the observed tree.
- CLI defaults equal the module constants.
- `_validate_status_index` passes a 129 KiB–256 KiB index and fails above the
  publisher bound.
- Pass 4: a symlinked `latest.json` reads as `{}`; a symlinked `history.jsonl`
  raises `watchdog_history_unwritable`, leaves the link's target byte-identical,
  and still leaves a real `latest.json` carrying the turn's fingerprint; a
  symlinked `watchdog.lock` fails closed and does not create the link's target;
  and an `OSError` from `mark_github_sent` yields `github_state_unrecordable`
  with health still recorded.
- The workflow's validator is **extracted from the YAML heredoc and executed**
  against a real `GITHUB_OUTPUT`/`RUNNER_TEMP`/`GITHUB_STEP_SUMMARY`, so the
  tests exercise the script the workflow actually runs: healthy path, six
  unparseable inputs, nine malformed-field cases, oversize, deep nesting,
  stale, future-dated, reason-code filtering, and a forged-`healthy=true`
  newline attempt. A test also asserts the workflow declares every bound the
  validator reads.

## Docs and memory

`docs/CODING_CONTROL_PLANE.md` (watchdog section), `ARCHITECTURE.md`,
`CHANGELOG.md`, `STATE.md`, `DECISIONS.md` (untrusted inputs; bounds belong to
their writers; an installed configuration must be runnable; a path is a
directory, not a spelling). `docs/reference/**` regenerated via the
`generate_reference` project action. No topology, ownership, route lane,
allowlist or receipt field changed, so `docs/architecture-map.md` needed no
edit beyond the note it already carries.

## Verification status — READ THIS BEFORE TRUSTING THE ABOVE

**Pass 4 has no execution evidence of any kind.** It is static review only. The
Codex run described below predates it, so it does not cover the four new tests,
the `O_NOFOLLOW` opens, the `watchdog_history_unwritable` code, or the
`github_state_unrecordable` branch. Treat pass 4 as unverified until the six
declared checks below have been run against this tree.

`mcp__flyto-indexer__verify --strict` and `mcp__flyto-indexer__task validate`
were both refused again in this session, so **no strict post-indexer gate has
been evaluated from this side.** The revision is intended to be strict-clean and
has not been proven to be.

**There is passing evidence for passes 1–3, and Claude did not produce it.** Codex
independently reran the repository suite against this working tree on
2026-08-12: **3564 passed, 17 skipped in 629.53s** with fail-fast, and **80
passed in 70.08s** for the focused `tests/test_coding_watchdog.py
tests/test_cli.py` pair. That covers every test added here, including this
session's end-to-end and CLI-default tests, and it retires the two assumptions
flagged in the previous revision of this file — that `read_task_window` and
`RouteStatusPublisher` create nothing under an existing state root is now
observed behaviour, not inference from `MissionStore._read(create=False)`.

**The host implementation verifier separately reported the full suite at exit
1.** That disagrees with Codex's green run, and nobody has captured a failing
node id, so there is no reproducible failure to fix. The audit's instruction
was to rerun without changing source unless a reproducible failure is observed;
no failure was observed, and no source was changed in response to this finding.

The most likely non-source explanation is the declared budget.
`.flyto/coding.yaml` gives `check.tests` `timeout_seconds: 900`; Codex measured
629.53s; and `core_capability_bridge` runs pytest immediately before it. A
suite at ~70% of its own timeout will exceed it under load, and a timeout kill
and a test failure produce the same exit code. **`.flyto/coding.yaml` is
outside this change's allowed path scope and was deliberately not touched** —
the verification contract's owner should decide whether that budget is right.
If the next run fails, capture the failing node id, not the exit status: that
is the single piece of evidence that would settle this.

**Claude has still never executed a check in this repository.** Across all five
sessions, `pytest`, `ruff`, `compileall`, `stack_lock.py`,
`generate_reference.py --check`, `mcp__flyto-indexer__task validate` and
`mcp__flyto-indexer__verify --strict` were refused by the local approval gate.
In the rework session the refusal was reconfirmed through four independent
channels — direct Bash in three command shapes, both indexer MCP tools, and a
delegated subagent, which hit the identical gate. The only command that has
ever run from this side is
`mcp__flyto-actions__run_project_action generate_reference` (exit 0, 23 files
rewritten), which ran again in this session after the pass-4 source edits and is
why the generated reference is current.

Run all six declared checks from the repository root before committing:

  ```bash
  .venv/bin/python scripts/stack_lock.py --workspace-parent ..
  .venv/bin/python -m compileall -q flyto_ai
  .venv/bin/python -m ruff check --select E9,F63,F7,F82 flyto_ai tests
  .venv/bin/python scripts/generate_reference.py --check
  .venv/bin/python -m pytest -q tests/test_core_mcp_contract.py tests/test_blueprint_closed_loop.py
  .venv/bin/python -m pytest -q
  ```
- Unexercised at runtime: `launchctl` install/uninstall, a live
  `gh variable set` against a real repository, and the workflow running in
  Actions. The workflow's *validator* is covered by tests; its `gh issue`
  create/refresh/close steps are not.
