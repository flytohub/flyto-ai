# Demand-scoped workspace authority repair

- Date: 2026-08-12
- Owner: Codex
- Status: verified; final repository commits pending

## Incident

An alternate coding state root completed its last non-landable job but its MCP
worker stayed alive. The host-global broker was process-scoped, so that idle
worker remained the live owner of `/Users/chester/flytohub` and refused every
Codex attached to the normal shared state root. Restarting only reacquired the
same idle hold. During recovery, the old auditor raced a second rework against
worker shutdown; the release valve correctly refused and restart reconciliation
settled the orphaned running state as `failed/service_restarted`.

## Repair

- Idle `CodingService` instances hold no workspace authority.
- Admission acquires before any durable non-terminal job mutation.
- Restart with open work reacquires before reconciliation.
- The last terminal transition releases after claims and leases settle.
- A bounded guarded observer releases worker A when worker B on the same shared
  state root performs the terminal write.
- Parent/child overlap remains symmetric; crashed open work and the release
  valve remain fail-closed; the MCP inventory remains submit/get/audit.

## Deployment model

Many Codex clients should use the same configured state root and queue.
Different state roots may coexist only while idle or on non-overlapping trees.
This is a single-host `flock` design. Multi-host work requires database leases,
not a shared NFS directory.

## Roadmap boundary

Do not add more gates before 2–4 weeks of usage data. Next priorities are the
unified mission board, risk tiers, explicit Claude capacity queueing, SLO/cost
observability, state archival/migrations, and eventual distributed leases.

## Verification

- flyto-ai: compile, fatal Ruff and all 23 generated references pass; full
  pytest is **3496 passed, 17 skipped**.
- The continuation/workspace/demand-scope audit set is **263 passed**; a
  continuation-backed job keeps its claim through the Codex audit loop and
  settles it only at a real terminal boundary.
- The installed-Core extension contract is **91 passed**; the host calls the
  real loader methods, publishes only real result fields, and refuses a
  module-style success alias that conflicts with `ExtensionResult.ok`.
- Demand-scope coverage includes idle startup, admission-before-mutation,
  restart recovery, parent/child overlap, crash authority, release-valve
  isolation, and the cross-process case where a peer settles the last job.
- The live shared checkout reports adoptable after terminal work; an idle
  alternate worker no longer pins `/Users/chester/flytohub`.
- The MCP startup marker was safely rotated only after all 642 historical jobs
  were terminal. The supervisor handshake exposes exactly submit/get/audit.
- Codex task `019ff12f-de18-7830-bcac-d99a79b49c93` resumed and recorded
  `Task 已恢復，可繼續`; no product file or website was changed by that recovery.
