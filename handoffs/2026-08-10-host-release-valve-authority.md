# Host release valve opens a state root without binding it

- Date: 2026-08-10
- Owner: claude
- Branch: main (working tree only — nothing staged, committed, or pushed)
- Status: Active — audit rework applied; **required checks still unrun** (see
  "Verification"). Not ready to close.

## Problem

`flyto-ai code-release --abandon-job` could not perform the one release it
exists for. `_cmd_code_release` constructed a default native/no-route
`CodingService`, so it also constructed that service's *startup authority*.
`_acquire_state_root_authority` then found the recorded marker (in production, a
Claude strict-route host with an emergency overflow grant) different from its
own, took the rotation branch, and `_require_all_jobs_terminal` refused because
of the open `awaiting_codex_audit` job the operator was trying to retire.
`_bind_startup_authority` would have refused for the same reason.

## Change

New construction mode `CodingService.open_host_release_valve(...)`, selected by
the private `_host_release_valve` keyword and used only by the CLI.

- `_acquire_state_root_authority_exclusively()` takes the same lease file with
  `LOCK_EX | LOCK_NB` and never falls back to shared. Any live coding service
  holds the shared lock for its life, so failure is proof one is alive and
  raises `CodingServiceBusy` (`service_busy`). Missing `flock` raises
  `CodingAuthorityUnavailable`, as on the ordinary path.
- The marker is never read, written, rotated, or reproduced, so
  `authority.json` survives byte-for-byte and the strict route still owns the
  root. `_bind_startup_authority` and `_require_all_jobs_terminal` do not run,
  which is what lets one requested `awaiting_codex_audit` job be abandoned while
  other open jobs under the recorded authority are left exactly as found.
- No `RouteStatusPublisher` is built and `_reconcile_interrupted_jobs` does not
  run. `_close_status`/`_publish_status` were already `None`-safe.
- New `HostReleaseValveRefused` (`release_valve_refused`) guards `submit`,
  `audit`, and `_pump_dispatch`. The valve's agent factory
  (`_release_valve_never_implements`) raises, so no implementer exists.
- `abandon` and `repair_workspace_claim` are byte-identical to before: the job
  lease is still taken, only `awaiting_codex_audit` moves to
  `failed`/`job_abandoned` with `landable: false`, and the workspace claim,
  resume envelope, and continuation authority are still released.
- The CLI now catches `CodingServiceError` around construction and prints the
  stable `exc.code`, so a busy root is a bounded code rather than prose.

MCP inventory is untouched: still exactly `flyto_coding_submit`,
`flyto_coding_get`, `flyto_coding_audit`.

## Files

- `flyto_ai/coding/service.py`
- `flyto_ai/cli.py`
- `tests/test_cli_code_release_authority.py` (new)
- `ARCHITECTURE.md`, `STATE.md`, `DECISIONS.md`, `CHANGELOG.md`,
  `handoffs/_registry.md`

## Audit rework, 2026-08-10

An independent audit returned `rework` with four findings. Three are resolved
here; the fourth cannot be closed from this session.

1. `generated_reference_stale` — **resolved.** The declared `generate_reference`
   project action was invoked through `flyto-actions`
   (`exit=0 duration_ms=3924`, "wrote 23 generated reference files"). It
   rewrote exactly the three files the audit named:
   `docs/reference/python/coding.md`, `package-root.md`, `README.md`. The new
   public/internal symbols are now indexed: `HostReleaseValveRefused`,
   `CodingService.open_host_release_valve`,
   `_acquire_state_root_authority_exclusively`, `_refuse_release_valve`,
   `_release_valve_never_implements`. No source changed after regeneration.
2. `test_invalid_enum` — **resolved.** `CodingAuditVerdict` has only `ACCEPT`
   and `REWORK`; the test used a nonexistent `REQUEST_CHANGES`. It now uses
   `REWORK` and the additive-refusal assertion is retained (both verdicts are
   still exercised, because the guard fires ahead of verdict validation).
3. `release_cleanup_evidence` — **resolved.** See below.
4. `verification_required` — **NOT resolved.** See below.

## Cleanup evidence added to the CLI regression

`test_release_retires_one_foreign_authority_job_and_leaves_the_rest` now proves
the cleanup contract directly rather than trusting the reuse of `abandon`:

- Both jobs are asserted **resumable before** the release, so removal is not
  asserted against an empty directory. The envelope is written at admission
  (`_commit_admission`), so an audit-ready job always has one.
- After: the target's durable resume envelope is gone, and its continuation is
  `settled` or `unavailable` — an `open` or `claimed` authority surviving the
  release would surface as itself and fail, rather than being folded into a
  pass.
- The survivor's record, resume envelope, and continuation state are all
  asserted byte-identical / unchanged, and every other stored continuation is
  compared byte-for-byte with the released job's own entry excluded (settling
  it is a write, so a whole-directory comparison would forbid the very cleanup
  being verified).
- The pre-existing marker byte-identity and single-worktree-release assertions
  are retained unchanged.

Honest limitation: this fixture's rounds succeed, so the target job carries no
continuation authority and the continuation assertion is satisfied by
`unavailable` rather than by `settled`. The resume-envelope assertions carry the
real weight. A fixture that strands a *stopped* round would exercise the
`settled` branch; that is worth adding and is not done here.

## Verification — STILL NOT RUN by me

`.flyto/coding.yaml` declares four required checks: `compile`, `lint`,
`generated_reference`, `tests`. In this session the only execution channel that
worked was `mcp__flyto-actions__run_project_action`, which declares exactly one
action (`generate_reference`). Every `python` / `pytest` / `ruff` invocation was
refused by the command gate, and `mcp__flyto-indexer__task` (validate) and
`mcp__flyto-indexer__verify` were both refused for want of a permission grant —
so the strict `indexer_post` gate was not run either.

**Do not read this handoff as a green build.** `generate_reference` is the only
check with a real receipt. `generated_reference --check` should now pass because
the docs were regenerated from current source and nothing was edited afterwards,
but I did not observe it pass. The next agent must run, through the host lane:

```
python -m compileall -q flyto_ai
python -m ruff check --select E9,F63,F7,F82 flyto_ai tests
python scripts/generate_reference.py --check
python -m pytest -q
```

The prior independent run was `3272 passed, 17 skipped, 1 failed in 216.23s`,
the single failure being the enum bug fixed above; the targeted suite was
`14 passed, 1 failed`. Both should now be clean, but that is a prediction, not
evidence.

## Known open questions for the next agent

- The new regression module plants the strict marker and per-record
  `execution_authority` computed from a *real* Claude/strict-route/emergency
  `CodingService` (`_claude_strict_authority`), rather than driving a full
  strict route to produce the job. That keeps the test fast and deterministic
  but means the fixture writes two durable files directly; if the record schema
  gains a field the planting helper does not set, the fixture — not the code —
  is what will need updating.
- `tests/test_cli_code_release_authority.py` imports `_policy` from
  `tests/test_coding_route.py`; that module is heavy, so the new file is not
  cheap to collect.
