# 2026-08-10 Cumulative No-Change Rework

Owner: Codex
Branch: `main`

## Scope

Repair the audited coding route after a correct, verified revision was sent to
same-session rework only to be terminalized because that recheck produced no
new bytes.

## Behavior

- `require_changes` remains mandatory for the job's attributable revision.
- A rework `no_changes` outcome is promoted only with passing required checks
  and a re-proved session, tenant/job claim, resume envelope, cumulative file
  set, and live content digest.
- The cumulative changed paths continue through the normal Indexer post gate
  and exact-revision Codex audit; no lane or audit is bypassed.

## Verification

- `tests/test_coding_rework_no_new_files.py`: `30 passed`.
- Service no-change/rework/audit subset: `92 passed`.
- Coding route suite: `218 passed`.
- Full flyto-ai suite: `2891 passed, 17 skipped`; one non-failing
  `PytestUnhandledThreadExceptionWarning` came from the existing
  `test_pro_integration` aiosqlite teardown after its event loop closed.
- Ruff passed for the changed service and regression test.
