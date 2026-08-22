# Amendment ledger-version interoperability

Owner: Codex
Branch: main
Date: 2026-08-22

## Failure reproduced

An accepted implementation entered audit rework, but the mandatory Indexer
pre-lane rejected the successor before provider start. The persisted parent
used the historical `task-context.v1` intent-ledger label, while the consumer
expected only `intent-ledger.v1`; mixed-intent successor analysis also exposed
the producer's missing root intent mirror.

## Boundary repair

- Parent and successor ledger sections independently accept only the legacy or
  canonical v1 labels during the rolling transition.
- Instruction sections remain strictly `task-context.v1`.
- Unknown versions and every digest, identity, fingerprint, path, chain, and
  amendment-generation mismatch still fail closed.
- Indexer owns canonical emission and immutable intent mirroring; Flyto2 AI
  does not import sibling source or infer a replacement contract.
- `stack-lock.json` pins the shipped Core main and the Indexer producer repair,
  so the required source-controlled preflight cannot validate against stale
  siblings.

## Verification

The pure amendment suite covers legacy-to-legacy, legacy-to-canonical,
canonical-to-canonical, and unknown-version refusal. The exact persisted Cloud
parent contract also validates against a newly generated canonical successor.

The repaired same-job audit rework crossed the legacy/canonical parent-proof
version boundary, then failed closed before provider start with
`route_plan_bound_exceeded`: 36 cumulative steps exceeded the unchanged
32-step ceiling. There is no same-job completion claim.

## Recovery and landing evidence

- Recovery primary job `job_0b90e4cab8e14f5482aec5f6` selected the final
  implementation and all ten governed gates were green.
- Final holistic Cloud job `job_497fc5ee77d948f2b71b26e8` was
  Codex-accepted.
- Follow-up job `job_4f40e4fcb6e54ea387786fe7` was Codex-accepted with
  `landable=true`, `audit_count=1`, and `rework_count=0`.
- Cloud PR <https://github.com/flytohub/flyto-cloud/pull/231> merged by
  protected squash to `main` commit
  `ee8c95678c9a18931890a096ea7c04f6a7295ad0` only after all remote checks
  were green, including Playwright: 136 total, 113 passed, 23 existing skips,
  0 failed, plus Audit Closure.

## Bounded limitation

Broad cumulative audited rework can exceed the route-plan ceiling. Future
repairs should bind active scope to current findings instead of raising or
bypassing the ceiling.
