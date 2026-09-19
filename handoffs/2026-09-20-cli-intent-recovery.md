# Bounded CLI intent recovery

- Owner: Codex
- Branch: `codex/cli-intent-recovery-closure`
- Status: verification in progress; independent review required before merge.

## Scope

Integrates the preserved September 13 CLI format-correction and quota patches
in an isolated checkout based on `9ecb6f35bd18c8aa17934ae5a50e6f5e2a311fbd`.
The original dirty checkout remains unchanged and the nine source files were
archived with a SHA-256 manifest outside the repository.

Only malformed host intent can request one correction. Original elapsed
inference time, round count, usage accounting, earlier observations and tool
authority remain authoritative. Any invalid batch has zero dispatch effects.
Quota service availability does not prove exhausted credits. Diagnostics are
fixed codes/reason enums, never raw provider answers.

## Verification

Pending the focused and full repository checks, strict Indexer gate and task
validation. Cloud owns live CLI learn/reuse/repair acceptance; SDK fixtures do
not establish that product acceptance. No Cloud dependency pin or runtime is
changed by this worktree.

## Rollback

Revert this PR as a unit. The original dirty work and its private archive remain
available. Keep the previous Cloud SDK revision until independent review; do not
replace guarded tool execution with native CLI actions on failure.
