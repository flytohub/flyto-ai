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
Full repository and live same-job retry evidence is recorded by the closure
run that lands this handoff.
