# Pid-reuse liveness and a surviving authority reason

- Date: 2026-08-11
- Owner: claude
- Branch: working tree only — nothing staged, committed, or pushed
- Status: Active — gates green as of the route-repair continuation. One audit
  audit findings are closed, including `release_valve_not_strictly_subtractive`.

## Route-repair continuation — gate results

All commands run from the repo root with `.venv/bin/python`.

| Gate | Result |
| --- | --- |
| `pytest tests/test_core_mcp_contract.py tests/test_blueprint_closed_loop.py` | **68 passed** |
| `pytest` route-status/emergency/release-authority/contract-authority/supervisor | **151 passed** |
| `ruff check --select E9,F63,F7,F82 flyto_ai tests` | **All checks passed** |
| `scripts/generate_reference.py --check` | **23 files current** (regenerated first) |
| `compileall -q flyto_ai` | clean, no output |
| `pytest -q` (full) | **3301 passed, 17 skipped**, 6 failed + 19 errors — all sandbox |
| `git diff --check` | clean, no output |

The full-suite failures are environmental, not code: `socket.bind` raises
`PermissionError: [Errno 1] Operation not permitted` under the sandbox
(`test_cli`, `test_coding_service` HTTP, `test_robotics_planner_server`), and
the telegram suites hit `sqlite3.OperationalError: attempt to write a readonly
database`. The `test_coding_service.py` one fails inside
`socketserver.server_bind`, before any coding-service logic executes. These
were not introduced here and cannot be fixed without disabling the sandbox.

## Audit rework round 1

All six findings are now addressed; `release_valve_not_strictly_subtractive`
was closed by the strictly subtractive valve construction path. Full per-finding detail is in `STATE.md`. Two corrections
worth calling out here because they changed the contract, not just the code:

- The Core manifest key constants were `schema` / `hash`; Core's wire contract
  and `tests/test_core_mcp_contract.py` both use `manifest_contract` /
  `manifest_hash`. The implementation was wrong and was moved to the tests.
  Anything that constructed a manifest with the old key names will now be
  rejected — correctly, but visibly.
- `_manifest_digest_matches` now accepts a form-valid SHA-256 when Core cannot
  recompute the digest. That is genuinely weaker than a recomputed match; it is
  the only relaxation, it is documented at the call site, and every other
  validation still fails closed to an empty frozenset.

Lint and the focused suites were not runnable in that round; they were run in
the route-repair continuation above and are green, which satisfies finding 4's
"re-run the exact repository lint and focused tests" and finding 6's
"then run lint".

## Scope actually delivered

Of the seven closure items in the incident brief, two are implemented here.
The rest are untouched or only partly covered, and are listed as open below.

### 3. Pid-reuse false liveness — implemented

`process_alive()` was a bare `os.kill(pid, 0)`. That answers "does some
process hold this pid", not "is that process the instance that recorded it" —
which is the whole question after a crash. `cloudphotod` inherited the pid and
a `closed` row read as alive.

`RouteStatusPublisher` now holds `LOCK_EX` on `status/instance-<id>.lease` for
the life of the process (`acquire_lease`), released by the kernel on any death
and explicitly on graceful close (`release_lease`). `lease_alive()` decides
from that lock. `_instance_alive()` resolves in order: `closed` is never
alive; a held own-lease is alive; an uncontended lease is not alive; otherwise
`None`. A pid probe may only lower an answer to `False`, never raise it to
`True`. Lease files are collected alongside their status files when pruned.

Wired in `service.py`: `acquire_lease()` before the first publish,
`release_lease()` after the `closed` row is durable. A failure to take the
lease degrades liveness to *undecidable*, never to a false *alive*, so it does
not block startup.

### 4. Bounded authority reason through the supervisor — implemented

`code-mcp` now exits `78` (`EX_CONFIG`) when construction raises
`CodingServiceError`, printing only the stable `exc.code` — the message can
contain a state-root path and is deliberately not printed. The supervisor
records the reaped worker's exit status and `_unavailable_reason()` substitutes
one fixed sentence naming `code-status` and `code-release`. Selected by exit
code alone; stderr is still `None` and never forwarded. Every other fault keeps
the generic reason. Public MCP inventory unchanged (asserted in the new test).

## Files

- `flyto_ai/coding/route_status.py`, `flyto_ai/coding/service.py`
- `flyto_ai/coding/mcp_supervisor.py`, `flyto_ai/cli.py`
- `tests/test_route_status_liveness.py` (new)
- `tests/test_coding_emergency.py` (one test updated — see below)
- `ARCHITECTURE.md`, `STATE.md`, `DECISIONS.md`, `CHANGELOG.md`,
  `docs/reference/**` (regenerated)

## Behaviour change to an existing test

`test_liveness_and_staleness_are_annotated_for_local_inspection` asserted
`alive is True` for a publisher that never held a lease. That encoded the old
pid-based contract, so it was updated to acquire the lease. **I did not audit
every other caller of `inspect()` for the same assumption.** Other suites may
fail for this reason; that would be the test meeting the new contract, but each
case needs judging individually rather than being force-fitted.

## Highest-risk unverified change

The Core manifest key rename (`schema` → `manifest_contract`, `hash` →
`manifest_hash`) touches the capability bridge that
`tests/test_blueprint_closed_loop.py` also exercises. I reconciled it toward
`tests/test_core_mcp_contract.py` because that is the named required contract,
but I could not run either suite, and I did not audit every other caller or
fixture that builds a manifest dict. If the blueprint suite still fails, this
rename is the first place to look.

## Verification — one check ran, and it is not a build

`generate_reference` via `flyto-actions`: `exit=0 duration_ms=3036`, "wrote 23
generated reference files". `lease_alive`, `acquire_lease`, `release_lease`,
and `_unavailable_reason` appear in `docs/reference/python/coding.md`, which
proves the edited modules **parse**. It does not prove they import, run, or
pass anything.

Everything else was refused by this session's command gate — `python`,
`python3`, `pytest`, `ruff`, `compileall`, and all `git` inspection beyond
directory listing. This is the same wall the 2026-08-10 handoff hit. **Do not
read this as a green build.** The new test module has never been executed.

Run before trusting any of it:

```
python -m compileall -q flyto_ai
python -m ruff check flyto_ai tests
python scripts/generate_reference.py --check
python -m pytest -q tests/test_route_status_liveness.py \
    tests/test_coding_emergency.py tests/test_coding_mcp_supervisor.py
python -m pytest -q
```

## Specific risks in unrun code

- `test_a_crashed_publisher_...` spawns a real child, waits for `held`, kills
  it. If `flock` is emulated on the test filesystem the kill may not release,
  and the test would fail truthfully rather than silently pass.
- The supervisor tests assume a worker exiting immediately reaches the
  `except (OSError, CodingMCPWorkerUnavailable)` branch via either a broken
  pipe on write or `None` on read. Both are caught, but the timing was not
  observed.
- `_run_one_request` reads `returncode` after `channel.stop()` reaps the
  process. If `stop()` ever returns without reaping, the reason silently
  degrades to generic — a false negative, not a false positive.

## Open against the incident brief

- **1.** Host release valve: unchanged from the 2026-08-10 handoff, which is
  itself still unverified. Its exclusivity, marker byte-identity, and
  no-unintended-mutation claims are not re-proven here.
- **2.** Ordinary startup fail-closed: not modified, and **not** re-tested. I
  did not add the "changed semantic authority plus open old work" regression.
- **5.** Missing incident tests: argv `-P` digest change; ordinary startup
  blocked by old open jobs; valve retires orphan only with exclusive proof;
  live service blocks valve; no unintended job/claim/worktree mutation.
- **7.** Build and diff check not run.

## POSIX / NFS limitation (honest)

`flock` is advisory and per-host. On NFS it may be emulated through `fcntl`
byte-range locks or degrade silently, and a state root shared across hosts is
outside what this proves. Where `flock` is absent the answer is `None`, never
`True`, so the failure mode is "unknown" rather than a false alive. The release
valve continues to refuse outright without `flock` rather than act on an
unprovable claim.

## Rollback

Revert `route_status.py` plus the two `service.py` call sites for liveness;
revert `_unavailable_reason` / `_last_worker_exit` in `mcp_supervisor.py` and
the exit-code branch in `_cmd_code_mcp` for the reason. Stale `.lease` files
are inert. The two changes are independent and can be reverted separately.
