# Bounded CLI intent recovery

- Owner: Codex
- Branch: `codex/cli-intent-recovery-closure`
- Status: local verification passed; independent review required before merge.

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

A subsequent live actor failed with `arguments_json` on the last of four
slice rounds. The pure follow-up `bc8eacc561dbb194441c744d84e3173d76856e45`
is a direct child of `75496e4` and preserves one correction across admitted
continuation. An invalid final-round batch stays inert and returns the existing
slice-budget outcome; its correction spends an ordinary next round only when
the host admits continuation within the remaining total budget. Expired
deadlines remain timeouts, and a second invalid proposal remains terminal.

## Verification

The isolated Python 3.12 environment uses the exact shared Core, Blueprint and
Indexer dependency revisions. Both warning categories from CI are errors.

- Complete SDK suite: 4,559 passed, 15 skipped in 298.52 seconds.
- CLI transport/format cohort: 78 passed; Core MCP cohort: 109 passed.
- The full suite includes five negative OS-sandbox isolation-proof cases.
- Compile, Ruff, generated reference (25 files), release drift, dependency lock,
  package dependency consistency, secret scan and wheel/sdist builds passed.
- CI-pinned strict Indexer: 18 passed; current suppression and environment
  checks: both passed, without warnings or failures.
- Runtime, architecture-memory and reference task validation use disjoint path
  ledgers covering the slice fix and this integration handoff.

The new regression first reproduced four failures, then passed after the
transport fix. It drives real Agent admission and continuation with injected
inference, proves no partial dispatch of an invalid batch or repeated earlier
action, and checks cumulative round accounting, timeout and second-refusal
behavior. It does not claim a live provider emitted a corrected response.

Initial full-suite failures exposed an unavailable `python` command on PATH,
test launchers that cannot use an interpreter path containing spaces, and the
Docker interface-name assumption below. The first two were resolved in the
isolated validation environment; production code was not changed for them.
The final full suite uses an interpreter path without spaces. Earlier failed
run logs remain available with the private acceptance receipts.

The initial pure runtime commit is `75496e4daef843dd591744ea6f7fc4a1d150e4eb`;
the slice correction artifact is `bc8eacc561dbb194441c744d84e3173d76856e45`.
Test portability, configuration documentation and these final receipts are a
separate follow-up commit. Cloud owns live CLI learn/reuse/repair acceptance;
SDK fixtures do not establish that product acceptance. No Cloud dependency pin
or runtime is changed by this worktree.

## Separate test portability maintenance

The full local suite exposed an existing Docker network assertion that required
exactly one interface named `lo`. Docker Desktop's network-none namespace also
contains dormant tunnel devices. A test-only follow-up checks the actual launch
mode, usable outside routes and rejected IPv4/IPv6 connections instead; host-read
and workspace-write denial assertions remain unchanged. Negative fixtures must
reject a routable interface, successful connection or host-network launch.
No sandbox runtime code or restriction is changed.

The current Indexer also found the already-used
`FLYTO_CLOSED_LOOP_MAX_REPAIRS` setting absent from `.env.example`. Its existing
default of 1 and clamp from 0 through 3 are now declared; runtime behavior is
unchanged.

## Rollback

Revert this PR as a unit. The original dirty work and its private archive remain
available. Keep the previous Cloud SDK revision until independent review; do not
replace guarded tool execution with native CLI actions on failure.
