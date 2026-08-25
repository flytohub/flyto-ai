# State

Last updated: 2026-08-25

## Multi-repository audit authority is fail-closed (2026-08-25)

- A live Code + Engine job proved that `repository_roots` granted the Codex
  implementer sibling write authority while source checks, Indexer post-work,
  `implementation_files`, and `implementation_revision_sha256` still described
  only the primary `working_dir`. Engine bytes could therefore be edited but
  were not part of the revision an auditor signed.
- Public submission now accepts one repository root. Cross-repository work uses
  one governed job and exact revision per repository, followed by an independent
  Codex integration audit. Multi-root execution stays closed until all proof
  lanes and the revision/changed-path contract are root-qualified.

## Exact numbered path-list authority (2026-08-25)

- Three audited Code jobs completed their required verification and then
  failed Indexer post-validation because the request's new regression-test
  path was omitted from `intent_ledger.allowed_paths`. The path appeared in a
  numbered, explicitly exact six-path list, but its item description placed it
  outside the 48-character local mutation-verb window.
- The first-round parser now recognizes only a strong exact-set declaration
  followed by a colon and numbered items. It still requires a typed suffix,
  an existing in-root parent, no final symlink, canonical repository-relative
  spelling, positive polarity and the existing 64-target bound. Generic
  inclusion prose and unnumbered lists remain context rather than authority.

## Domain solver and Codex authority implementation (2026-08-24)

- Core-bearing extras now require `>=2.31.0`; the advisory-derived security
  floor remains independent.
- The Codex adapter retains exact `--cd`/`--add-dir` construction for internal
  compatibility, but the public service now refuses multiple repository roots
  before provider startup because those additional roots are not yet covered by
  independent verification and revision authority.
- In the governed stack workspace, required `stack_lock` makes the locked Core
  sibling mandatory and `domain_solver_closed_loop` runs that sibling's source
  semantic contract. Its always-running AI closure exercises the
  installed/imported Core capabilities through all three known-answer and
  rejection cases before Blueprint verified learning. The source-contract test
  skips only for an isolated flyto-ai checkout where the sibling is absent; a
  skip is not presented as execution. The three named baselines verify software
  arithmetic and contracts only, not sensors, hardware, real physical frames,
  laboratory chemistry, medicine, handling, or safety, and make no Physical AI
  or universal-world-model claim.

## The floor gate proved itself, and showed where it was still pinned (2026-08-24)

Within a day of landing, `tests/test_stack_security_floor.py` went red for a real
reason: Core published five further advisories, one critical, affecting
`<= 2.28.1`, and both declared floors still said 2.28.1. That is the derived
number working as designed.

It also exposed the level the design had not covered. CI checks Core out at the
revision `stack-lock.json` pins, so the manifest the gate reads is only as fresh
as the last lock bump — the gate could not have gone red in CI no matter what
Core published. `.github/workflows/advisory-freshness.yml` now asks the same
question against `flyto-core@main`, daily, in a workflow separate from CI so a
pull request never fails for an unrelated repository's publication. The
advisory-derived security minimum is satisfied, and the independently higher
domain-capability floor is `>=2.31.0` as it lands.

## Cross-repository contract repair (2026-08-23)

Four things that were true in the source tree and false everywhere else:

- `stack-lock.json` pinned Blueprint `f3eb62e` and Core `23efc93` while both
  mains had moved on, so the required `stack_lock` check failed locally — no
  audited coding job could start — and CI was integration-testing this package
  against sibling revisions two feature commits stale. The lock now pins the
  current Blueprint and Core revisions.
- `flyto-core[browser]>=2.16.1` accepted every Core release predating all 33
  published advisories, and `flyto-blueprint>=0.1.0` accepted engines with no
  `available_module_ids` parameter, which `tools/blueprint_tools.py` treats as
  "call it the legacy unfiltered way". The module-availability gate was
  therefore off for every PyPI install while reading as enforced here. Floors
  are now `>=2.28.1` and `>=0.3.0`; Blueprint 0.3.0 is prepared but not yet
  tagged, so the second floor does not resolve until that tag is pushed.
- `flyto-product.toml` did not exist here, while Blueprint and Core each shipped
  one disclaiming "intent and provider governance". Layer 1 now declares itself
  and `tests/test_product_contract.py` asserts it exactly.
- The ImportError fallbacks in `assistant/resilience.py` and
  `tools/core_tools.py` were second copies of Core browser constants, and the
  first had already drifted from Core with nothing in the suite touching it.
  Both are now inert, and `tests/test_core_constant_parity.py` fails if a copy
  returns or if a borrowed value stops matching Core.

Verified locally on this checkout: `stack_lock`, `compile`, Ruff
(`E9,F63,F7,F82`), 23 generated reference files current, and the full suite.

## Gates at the boundary, and recorded complexity debt (2026-08-23)

Four checks now hold claims that only a convention held before. `stack-lock.json`
puts Blueprint, Core and the Indexer on disk beside this repository, which makes
this the only place in the stack that can check the stack:

- **Advisory floor.** `tests/test_stack_security_floor.py` derives the lowest
  Core version clearing every published advisory from
  `flyto-core/security/advisories.json` and fails if any declared `flyto-core`
  floor predates it — here and in Blueprint. Read, never restated.
- **Release drift.** `scripts/check_release_drift.py`, a required check in
  `.flyto/coding.yaml` and a CI step, fails when a tag `v<version>` exists and
  the packaged source differs from it. An unreleased version passes; it asks for
  a correct number, not a release.
- **Complexity ratchet.** `tests/test_complexity_budget.py` holds new code to 800
  lines per module and 8 parameters per function, and holds existing debt to
  `tests/complexity_baseline.json`, which may only shrink.
- **Borrowed-value parity.** `tests/test_core_constant_parity.py` fails if a value
  taken from Core stops matching Core or a fallback copy returns.

Recorded debt today: 22 modules over 800 lines and 23 functions over 8
parameters. The largest is `CodingService`, one 6,973-line class inside a
7,774-line module — `coding/errors.py` took the 35 typed failures out of it, but
the class itself is untouched. Splitting it is real work with a real blast
radius on the audited route; the ratchet exists so it cannot get worse in the
meantime.

## Coding control plane is dormant, observed 2026-08-23

Stated rather than left ambiguous, because most of the complexity above serves
this and a reader should not have to guess whether it is live.

A read-only `flyto-ai code-watchdog` against `~/.flyto/coding-service` reports
`health: degraded`, `reason_codes: [codex_audit_backlog, rolling_build_reload]`,
with `awaiting_codex_audit=9`, `live_current_build=0`, `live_stale_build=1` and
32 recorded instances. `jobs.sqlite` has not advanced since 2026-08-10 and no
worker process is running. So the records describe nine jobs waiting on an audit
that nothing is going to perform, and one live instance pinned to a build that
has since been replaced.

This is a state, not an incident: the scheduled watchdog is off precisely so it
stops reporting it as one. Reviving the route means clearing the audit backlog,
reloading the stale-build worker, installing the heartbeat publisher and
restoring the schedule — in one change, in that order.

## Amendment recovery and Cloud landing evidence (2026-08-22)

The repaired same-job audit rework crossed the legacy/canonical parent-proof
version boundary, then failed closed before provider start with
`route_plan_bound_exceeded`: 36 cumulative steps exceeded the unchanged
32-step ceiling. There is therefore no same-job completion claim.

Recovery primary job `job_0b90e4cab8e14f5482aec5f6` selected the final
implementation and all ten governed gates were green. Final holistic Cloud job
`job_497fc5ee77d948f2b71b26e8` was Codex-accepted. Follow-up job
`job_4f40e4fcb6e54ea387786fe7` was Codex-accepted with `landable=true`,
`audit_count=1`, and `rework_count=0`.

Cloud PR <https://github.com/flytohub/flyto-cloud/pull/231> merged by protected
squash to `main` commit `ee8c95678c9a18931890a096ea7c04f6a7295ad0` only after
all remote checks were green, including Playwright: 136 total, 113 passed, 23
existing skips, 0 failed, plus Audit Closure.

The bounded limitation remains: broad cumulative audited rework can exceed the
route-plan ceiling. Future repairs should bind active scope to current findings
instead of raising or bypassing the ceiling.

## Governed Robotics planner entry (2026-08-22)

The loopback Robotics planner server no longer calls the provider-facing
planning service directly. `GovernedRoboticsPlanner` first requires
`flyto.goal-frame.v1` normalization plus Blueprint reuse and Core capability
discovery through `prepare_planner_request`; a routing refusal ends before any
provider call. The Robotics caller's local executable catalog remains the
authority ceiling, and Flyto2 Robotics still revalidates the returned plan
before motion.

Focused planner, routing, planning, and Blueprint closed-loop coverage passes
146 tests. Strict Indexer verification passes 18/18. This is a local contract
closure only: no hardware, deployment, publication, or governed
four-repository stack release is claimed.

## Codex JSONL frame diagnostics and bound (2026-08-20)

The Codex implementation adapter now accepts one valid JSONL event up to 2 MiB
while retaining the separate 8 MiB total stdout ceiling. This closes a proved
false failure where one valid 1,048,577-byte tool-result event appeared inside
a valid 1,448,355-byte completed stream. `coding.round` records content-free
byte counts plus malformed-JSON, malformed-shape, oversized-event,
stream-overflow, invalid-output, and timeout indicators. It never records the
offending frame or provider content.

Focused regression coverage accepts a valid event between 1 MiB and 2 MiB and
continues to reject malformed JSON, a single event over its configured bound,
and a total stream over its independent bound. The source digest inventory is
self-covering, allowing a long-lived supervisor with the older inventory to
observe the inventory expansion at its next safe worker boundary.

## Codex adapter hot-reload coverage (2026-08-20)

`current_service_build_id()` now includes both implementation adapters. A
change under `agents/codex_cli.py` therefore marks an existing supervised
worker stale and causes the ordinary safe-boundary reload before a later job;
previously only the Claude adapter participated in that digest. This restores
the documented source-only rollout behavior without expanding reload,
execution, audit, commit, or deployment authority.

## Codex CLI completion evidence (2026-08-20)

The Codex implementation adapter now treats the bounded protocol terminal
event, not later process teardown, as the model-turn completion authority. A
valid `turn.completed` with no malformed/oversized output and no timeout may
proceed to the existing host snapshots and required checks even when the CLI
then exits non-zero. The `coding.round` evidence retains the exit code and a
`completed_with_nonzero_exit` flag. Missing completion, invalid output,
timeout, failed checks, and absent attributable changes retain their existing
fail-closed results. No Indexer, Blueprint, Core, audit, commit, or deployment
authority changed.

## Digest-bound amendment delta execution (2026-08-14)

The strict Indexer pre-lane now treats an amended task plan as cumulative
evidence rather than cumulative executable work. It recomputes Indexer's
versioned parent digest, validates the exact generation/root/project and
original/added/cumulative path partition, requires all paths to be covered by
resolved targets, and verifies the content-addressed contract id plus every
ancestry id/digest/link/count. It reuses only an exact parent step occurrence;
novel analysis against an original path and every successor gate still run. The
executable delta remains under the normal 32-step ceiling. Pinned Indexer
verification rejects ancestry length eight, so generation seven is the current
compatibility ceiling and generation eight remains fail-closed as a separate
producer/verifier P1.

Both generations' task-profile amendment fields and intent/instruction
fingerprints must exactly mirror their corresponding contract sections. Intent
and ledger description remain continuous, and a successor with a missing or
malformed instruction context fails closed.

The intent-ledger version boundary is rolling-compatible: persisted roots may
carry the historical `task-context.v1` ledger label, while canonical producers
emit `intent-ledger.v1`. Parent and successor are checked independently against
that exact two-value set; instruction context remains strictly
`task-context.v1`, and unknown versions still fail before provider start.

All amendment and ledger paths must use exact canonical repository-relative
POSIX spelling. The accepted cumulative scope is the ordered union of the
authenticated parent ledger, exact audited prior implementation paths and the
current filesystem-validated explicit targets; an omission or invention is a
closed refusal. Existing numeric-suffix literals remain legal, while the host
new-file parser still refuses an unresolved `M1.1` version label.

Indexer envelope failures and ordinary `pass=false` validations now retain only
exact host-registered machine reason/action codes without persisting unknown
tokens, prose, paths, URLs or secrets. New-file extraction rejects a
numeric-only dotted milestone such as `M1.1`, while a typed extension such as
`.7z` remains valid under an explicit mutation instruction. The Engine
parent-plus-successor projection is admitted only through the deterministic
Indexer contract gate; Flyto2 AI neither hard-codes a live step count nor weakens
its proof boundary.

## Pre-provider rework-route recovery (2026-08-14)

Audit rework that fails in Indexer/Blueprint before provider start now remains
nonterminal as `rework_route_blocked`. `flyto_coding_submit` has an explicit
`retry_rework_route` Boolean action; ordinary replay is still read-only. The
action requires the original idempotency key and normalized request, exact
session/revision/audit/mission/plan/worktree/execution proof, and at most one
provider-free route retry. Continuation-origin jobs retain only an exact
claimed-by-current-job authority.

Mission publication checks the persisted child status and acknowledges its
operation only after the owner job record commits. One exact peer-deferred
orphan may receive a deterministic compensation child. A second publication
loss terminalizes as `rework_route_recovery_exhausted`, clears the retry action,
workspace claim and resume envelope, and settles any claimed continuation.
Focused tests cover future and legacy records, ordinary replay, proof drift,
continuation/emergency flow, peer dispatch, build drift, and bounded exhaustion.

## Live-safe orphan retirement (2026-08-14)

`flyto-ai code-release --abandon-job` now opens
`CodingService.open_host_abandon_valve`. It holds the existing authority lease
shared, writes only under the cross-process state guard, and must acquire the
exact target job lease. Audit-ready work may then fail closed; queued or
rework-queued work additionally requires its exact MissionStore item to be
closed blocked/deferred. A live round, ready queued item, unsupported state, or
authority transition refuses without mutation.

`--repair-workspace` still uses the exclusive host release valve and therefore
still refuses while any coding service is alive. Neither valve binds startup
authority, reads or writes `authority.json`, constructs runtime machinery, or
adds an MCP tool. Focused CLI and authority regressions cover live peers,
kernel-closed queued work, target lease contention, survivor preservation,
marker preservation, operation refusal, and descriptor release.

## Project-scoped Indexer plan analysis (2026-08-14)

The coding route now binds every project-aware read-only plan step to the
host-derived workspace project, not only search. Canonical and translated
`impact`, `structure`, and `call_hierarchy` calls receive the current workspace
project. Conflicting project evidence fails closed before analysis or editing.
This closes pre-work `domain_failure` results where an exact symbol from an
isolated worktree was resolved against ambient indexes before the implementer
started.

The tool allowlist, lane ordering, transport bounds, receipt contract, and
fail-closed behavior are unchanged. The complete repository suite, focused
route regressions, generated reference check, and locked cross-stack dependency
check passed before release.
## Governed execution-session connector seam (2026-08-14)

`admit_execution_session` now accepts one keyword-only pre-established,
one-shot `ExecutionSessionConnector` handle. Its trusted async callback receives
a fresh detached prepared-session result in its non-daemon worker process and
runs only through the existing durable Scheduler one-shot occurrence. The
catalog remains content-free: session/task IDs, request/authority/route/overall
digests, and the Scheduler result with its own evidence reference. It stores no
goal, principal, manifests, credentials, callback, or callback identity.

Without a connector the exact prior `execution_not_connected` receipt remains.
Exact zero-cost, empty success closes `connected`; stable empty failure closes
blocked. Invalid output and exceptions are content-free failures. Cancellation
or lost execution authority recovers as `execution_outcome_unknown` and is not
automatically replayed. Connector process start happens during handle
construction, before admission. Admission bounds readiness, nonblocking request
transfer, child-side entry, and result receive to the one absolute deadline; a
worker stalled before readiness cannot block cancellation or timeout. At that
deadline, on owner cancellation, validation failure, exception, or normal
return, the host forcibly terminates that process within the fixed closure grace
and confirms zero live connector work before returning or closing
`execution_connector_timeout`. Delayed executor entry recomputes the remaining
duration and does not call at zero. A duplicate already waiting when its owner
is cancelled runs Scheduler reconciliation only until the same deadline plus
0.5 seconds; it never enters its own connector. No in-process callback task,
daemon thread, or global connector slot survives a return or durable closure.
Duplicate, concurrent, and restarted admission observes the persisted receipt;
changed digests conflict. Connector
input/output mutation cannot change prepared or durable state. Deterministic
start-stall tests cover timeout and owner cancellation with zero remaining
connector task, thread, process, or later side effect. This is a provider
contract only: no Cloud consumer, live account, hardware, or device execution is
claimed.

## Host-only coding watchdog and remote dead-man switch (2026-08-12)

`flyto-ai code-watchdog --state-dir <dir> [--health-dir <dir>] [--json]` is the
independent, non-AI observer of the coding control plane. It reads the same
bounded status index and task window as `code-status` / `code-task-window`,
invokes no model, and cannot submit, audit, abandon, repair, commit or push.

It writes only `~/.flyto/health/coding/`: `latest.json` every run,
`history.jsonl` on transitions (size-rotated, 4 archives), and `github.json` as
the heartbeat cursor. All three hold aggregate counts, stable reason codes, the
reader build digest and timestamps — no prompts, paths, job/session ids,
evidence or credentials. A single `flock` prevents overlapping runs.

`--install` / `--uninstall` manage a per-state-root macOS LaunchAgent (60s
default). `--github-repository OWNER/REPO` publishes the secret-free heartbeat
to a GitHub Actions repository variable via the already-authenticated `gh` CLI;
no token reaches the plist or any health file.
`.github/workflows/coding-watchdog.yml` reads that variable and opens, refreshes
or closes one labelled issue. It is deterministic Actions, not an agentic
workflow, so healthy polling consumes no model quota.

Scheduled polling is disabled as of 2026-08-23. The publisher was never
installed against this repository, so `FLYTO_CODING_HEARTBEAT` was never set and
every 15-minute run failed `heartbeat_missing` -- 400 consecutive red runs in the
visible window, with incident issue #38 open since 2026-08-13. A permanently
firing dead-man switch reports the same thing whether the control plane is dead
or was never wired, so the `schedule` trigger is commented out and only
`workflow_dispatch` remains. Restore the trigger in the same change that runs
`flyto-ai code-watchdog --install --github-repository OWNER/REPO`.

Alert-only in this release: recovery stays with the existing explicit
subtractive commands. The MCP tool inventory is unchanged.

Hardening applied 2026-08-12 (uncommitted): the workflow validates the
untrusted heartbeat variable field by field and re-checks the emitted `reason`
against a single-line allowlist before writing `GITHUB_OUTPUT`, closing a
forged-`healthy=true` path; `state_readable` uses the publisher's
`MAX_STATUS_INDEX_BYTES` rather than the watchdog's record limit; the state
root and health directory must be disjoint (`watchdog_paths_overlap`) and are
now compared and labelled after symlink resolution, so neither a linked
`--health-dir` nor a linked `--state-dir` can defeat the overlap guard or split
one state root into two LaunchAgent labels; `--install` validates every plist
value through the same validator the observing run uses; and the remote
heartbeat is one bounded `gh variable set` upsert with a 48 KB local ceiling.

Consolidation pass 2026-08-12 (uncommitted): the health directory is now
treated as a location the watchdog does not own exclusively, because
`--health-dir` may legitimately be given a world-writable parent. Every record
opened by name — `latest.json`, `github.json`, `history.jsonl`,
`watchdog.lock` — is opened `O_NOFOLLOW`, `_read_json` measures and reads one
descriptor instead of checking a name and then reading it, and rotation uses
`lexists` so a planted link is rotated away rather than left for the next
append. A refused append reports `watchdog_history_unwritable` after
`latest.json` is already durable. A failed heartbeat-cursor write is a
`github_heartbeat` warning (`github_state_unrecordable`) rather than a lost
turn, so the remote switch can never read `healthy` while the local record was
silently skipped. The workflow no longer sets `cancel-in-progress`.

Coverage added 2026-08-12: one test drives a full `run_watchdog_once` turn
against a real state root and asserts the observed tree is byte-identical
afterwards; one asserts the `code-watchdog` parser hands the observer the
module's own default thresholds rather than a drifting second copy; and four
cover the consolidation pass — a symlinked `latest.json`, `history.jsonl` and
`watchdog.lock`, and an unrecordable heartbeat cursor.

Known gaps: LaunchAgent install/uninstall is macOS-only and its `launchctl`
path is not exercised by automated tests; the GitHub workflow's live behaviour
has not been observed against a real repository variable yet, and its
`gh issue` create/refresh/close steps are untested (its heartbeat validator is
extracted from the YAML and executed by the suite).

Verification status — READ BEFORE COMMITTING.

Passing evidence exists, but it was not produced by Claude. Codex independently
reran the repository suite against this working tree on 2026-08-12 and reported
**3564 passed, 17 skipped in 629.53s** with fail-fast, and **80 passed in
70.08s** for the focused `tests/test_coding_watchdog.py tests/test_cli.py`
pair. That is the current best evidence that this revision is green, and it
covers every test added for the watchdog.

The host implementation verifier separately reported the full suite at exit 1.
That result and Codex's green run disagree, and the disagreement is not yet
explained by any observed failing test — no failing node id has been captured
by anyone. The most likely non-source explanation is the declared budget:
`.flyto/coding.yaml` gives `check.tests` `timeout_seconds: 900`, Codex measured
629.53s, and `core_capability_bridge` runs pytest immediately before it. A
suite sitting at ~70% of its own timeout will exceed it under load, and a
timeout kill is indistinguishable from a test failure in the exit code. That
file is outside the change scope for this work and was deliberately not
touched; whoever owns the verification contract should decide whether the
budget is right.

Claude has still never executed a check here. Across all five sessions that
touched this work, `pytest`, `ruff`, `compileall`, `stack_lock.py`,
`generate_reference.py --check`, `mcp__flyto-indexer__verify` and
`mcp__flyto-indexer__task validate` were refused by the local approval gate —
in the last two sessions through direct invocation, through both indexer MCP
tools, and through a subagent, which hit the same gate. The only command that
has ever run from Claude is `run_project_action generate_reference` (exit 0,
23 files rewritten). No strict post-indexer gate has been evaluated from this
side, so the consolidation pass above is static review only and Codex's green
run predates it.

Before committing, run every declared check in `.flyto/coding.yaml` —
`stack_lock`, `compile`, `lint`, `generated_reference`,
`core_capability_bridge`, `tests` — and if `tests` fails, capture the failing
node id rather than the exit code, so the timeout hypothesis above is either
confirmed or replaced by a real defect.

## Repo-set concurrency and unified task window (2026-08-12)

Host-global ownership is now per durable repository set instead of per
configured parent. A normal job infers the nearest Git root; a cross-repository
job can atomically declare up to sixteen non-overlapping Git roots. Private job
state persists canonical roots plus path-free digests, restart reacquires only
open sets, and terminal transitions retain only repositories still needed by
durable work. Pre-set records migrate conservatively from `working_dir`.

`flyto-ai code-task-window --state-dir <dir> [--limit N] [--json]` is the one
host-only view of missions, branches, current scheduler preference, owning
client label, repository claims, implementation presence, failures and
audit/rework counts. Active work sorts before terminal history. It is read-only,
secret-free, bounded, not conversational context, and does not change the
three-tool MCP inventory.

`stack-lock.json` now pins Blueprint, Core and Indexer once. CI reads
checkout refs from that file and verifies all three checked-out HEADs; the
repo-owned coding contract performs the same sibling verification locally.
This replaces duplicated workflow SHA literals.

## Host-only Core extension management (2026-08-12)

`flyto_ai/tools/core_tools.py` gained a generic adapter for Core's extension
surface: `list_core_extensions`, `list_core_extension_kinds`,
`install_core_extension(name, version, upgrade)`,
`uninstall_core_extension(name)`.

It binds the real Core contract in `core.plugin.loader`: `get_plugin_loader`
returns the loader, `EXTENSION_KINDS` declares the kinds as **records** of
`kind` / `prefix` / `entry_point_group`, `normalize_extension_name` folds a
requested name to the identity Core acts on, and `ExtensionResult` carries an
outcome. The loader's methods are `list_extensions`, `install_extension`, and
`uninstall_extension` — not the host's operation names — and the mapping is an
explicit constant rather than `getattr(loader, operation)`.

`list_extensions` returns a **plain list**, not an envelope, so reaching a list
is itself the success; it takes **no kind argument**, so the kind filter is
applied host-side after normalization, against the same bounded token the host
publishes. Loader calls run through `asyncio.to_thread`, including
`get_plugin_loader()` itself, so an install that shells out to a package
installer cannot stall the event loop; kinds read a module constant and need no
thread.

Two drafts of this work were wrong about Core before the tests were run. The
first invented a `core.extensions` module that does not exist. The second used
`install` / `uninstall` and invented `install_enabled` / `restart` / `refresh` /
`rollback` result fields that `ExtensionResult` does not declare. The lesson is
the 2026-08-11 one repeated twice: a bridge written against an imagined Core
validates cleanly against its own fixtures and does nothing against the real
one. What caught both was running the contract tests against the installed
Core.

The adapter holds no kind taxonomy and no name rules of its own. Core's
normalized name and Core's own `code` pass straight through, degrading to a
host code only when Core's value is missing or is not a bounded token. The
envelope publishes Core's real consequence fields under Core's own names —
`previous_version`, `restart_required`, `rolled_back`, `refresh_failed`, each
defaulting to False/empty so an unreported one is never read as "yes" — plus
`install_enabled`, which is the *host's* opt-in state and is not claimed to
come from Core.

Three boundaries are enforced and tested in
`tests/test_core_extension_management.py`:

- Install and uninstall are host-only. They are not MCP tools;
  `get_core_tool_defs` filters any Core tool whose name is an install,
  uninstall, or reinstall verb; and `dispatch_core_tool` refuses such a name
  before it resolves the handler. `list_installed_modules` is deliberately not
  caught by that rule.
- Mutation requires `FLYTO_EXTENSIONS_INSTALL_ENABLED`. The gate is read per
  call and checked before request validation, so an un-opted-in host answers
  `install_disabled` for every input and never reaches Core.
- Every outcome returns one fixed envelope. No pip stdout, stderr, log,
  command line, traceback, or exception text is copied into it, and only the
  exception *type* is logged.

The first 2026-08-12 audit exposed two real-contract failures after 82 passing
tests: the draft called nonexistent `install` / `uninstall` methods and read
invented result flags. The corrected suite now binds the **installed** Core,
requires `list_extensions` / `install_extension` / `uninstall_extension`, and
checks every Core result field the adapter reads, including `kind`. It also
proves an extra module-style `status="success"` key cannot override Core's
explicit `ok=False`, and exercises the disabled mutation gate against the
installed Core without reaching its loader. Final focused result: **91 passed**.

All reference output was regenerated by `scripts/generate_reference.py`; its
`--check` mode reports all **23 generated reference files current**. This
replaces the earlier hand-maintained, environment-unverified draft.

## Demand-scoped workspace authority (2026-08-12)

The host-global broker no longer equates an MCP process lifetime with active
work. An idle service owns no configured tree. Admission acquires authority
before the first durable non-terminal job mutation; restart reacquires before
reconciliation when open work exists; the final terminal transition releases
it. Because any compatible worker on the shared state root may execute and
settle a job, each process that acquired a lease also runs one bounded
idle-release observer under the cross-process state guard. This closes the
case where worker A admitted and worker B settled while A received no later
request.

Current local closure: compile, fatal Ruff, all 23 generated references, and
the full suite pass at **3496 passed, 17 skipped**. The cross-repository seams
also pass on the same host: Blueprint **256 passed** plus build and strict
Indexer 18/18; Core **2785 passed, 11 skipped, 273 deselected** at 63.20%
coverage plus its pinned extension/runtime/registry checks; Indexer fast suite
**2148 passed, 1 skipped, 61 deselected**. Final per-repository commits remain
the only local closure step.

The normal deployment remains one shared state root for many Codex clients.
Different state roots are allowed only when their configured trees do not
overlap, or while the previous root is idle. Parent/child overlap is symmetric
and atomic; neither direction is a documented exception.

### Operational roadmap — observe before adding gates

Run ordinary work for 2–4 weeks and collect evidence before expanding the
architecture. Priorities are a unified mission/branch queue view; automatic
risk tiers; explicit Claude capacity and quota queueing with no silent Codex
fallback; wait/success/rework/false-block/cost SLOs; bounded archival and state
migrations; and database leases before any multi-host deployment. These are
roadmap items, not part of this repair.

## Rework pre-plan now carries the proven cumulative scope (2026-08-11)

Strict post-validation exposed a final `unplanned_diff` even though all five
repository checks passed. The last audit target was present in the amended
Indexer ledger; the missing entries were paths attributed during earlier
rounds but absent from the later audit prose. The service already re-proved
those paths before resuming the implementation session, but only used them for
post-work and the revision digest.

The proven prior tuple now enters the rework pre-plan as well. The route unions
it with the new finding's explicit paths before requesting the same-root
Indexer amendment, preserving the 64-target route bound and leaving first-round
payloads byte-for-byte unchanged. `tests/test_coding_plan_amendment.py` covers
both the route seam and the real service state machine.

## Core manifest bridge realigned to the real Core (2026-08-11)

The 2026-08-11 route-repair round reconciled the Core capability bridge
*toward the fixtures* (`manifest_contract` / `manifest_hash`, record-shaped
`modules`, `capability_id`, `plugin`). Those fixtures were wrong. Core's real
`core.capability_manifest` emits `schema` / `hash`, `modules` as bare module-id
strings, `capabilities` records of `capability` + `providers`, and `plugins`
records of `id` + `version` + `module_count`. The consequence was silent and
total: every real manifest failed validation, so a fully installed Core
reported zero installed modules and Blueprint filtering narrowed to nothing.

- `flyto_ai/tools/core_tools.py` now validates the real shape. Providers are
  cross-checked against the manifest's own `modules`, so a capability cannot
  introduce a module Core did not declare installed; a capability with no
  providers is malformed. Plugin records require all three fields.
- The digest is still recomputed by Core's own `compute_manifest_hash`, paired
  to Core's own reader, and all three declared counts must match exactly.
- `get_core_installed_capabilities` is **removed**. It unioned module and
  capability ids, which is exactly what must not reach a Blueprint engine.
  `flyto_ai/tools/blueprint_tools.py` now passes installed module ids only,
  under `available_module_ids`, and strips every model-facing availability
  argument (including that keyword) from published schemas. Engines predating
  the keyword are still called without it.
- `installed_capabilities` provenance drops `declared_capability_count`; the
  union it disambiguated no longer exists, and `capability_count` is now
  exactly Core's declared count.
- `tests/test_core_mcp_contract.py` fixtures were replaced with the real shape,
  and three tests now bind the *installed* Core directly. They fail if a
  non-empty Core validates down to an empty set — the regression a fixture can
  never catch, because a fixture agrees with whatever the host believes.

**Verified 2026-08-12.** Codex read the installed Core implementation, exercised
the real Core/Blueprint bridge, and ran the full flyto-ai suite: 3496 passed,
17 skipped. Compile, fatal Ruff, all 23 generated references, and the focused
Core contract checks also pass. The refusal record this paragraph replaces is
historical incident evidence, not the current state.

## Host-global workspace root authority (2026-08-11)

A coding state root brokers the jobs inside it and cannot broker a directory
tree, because its workspace claims live under itself. Two services on two state
roots therefore each kept a private, self-consistent opinion about the same
checkout, and two sessions edited one tree concurrently.

`flyto_ai/coding/workspace_authority.py` adds the missing layer: one registry
entry per canonical workspace root, a shared `flock` while a state root has
durable non-terminal work, an exclusive lock required before the recorded
owner may be written or rotated, and durable identity that prevents a crash
from handing open work to whoever restarts first. An ordinary service acquires
at admission before its first job mutation, or at restart before reconciling
existing open work. An idle service does not join. The host release valve never
joins, rotates, or adopts it.

Recovery is bounded and needs no hand-edited JSON: finish or retire the old
owner's work — with `code-release` if an audit is stranded — and the next start
adopts the tree automatically. While any non-terminal job or any surviving
workspace claim remains under the recorded owner, an incompatible state root is
refused, so migration cannot make audit-pending work look free.

Ancestor and descendant overlaps are both refused, atomically under the
registry-wide coordination lock. Real-process tests cover child-after-parent,
parent-after-child, and simultaneous parent/child acquisition.

The registry location is neutral and startup-only: `XDG_STATE_HOME` (or
`~/.local/state`) `/coding-workspace-authority`, overridable through
`CODING_WORKSPACE_AUTHORITY_ROOT` for isolated tests, never from a job payload,
private, symlink-refusing, and outside every product worktree.

## Release valve is strictly subtractive (2026-08-11)

`open_host_release_valve` no longer runs the ordinary constructor. The object
is allocated without `__init__` and `_init_release_valve` runs a fixed order:
validate an existing safe state root with `secure_directory(create=False)`,
require the established furniture (`.service.lock`, `locks/jobs`,
`locks/workspaces`, the authority lease) to already exist, take the authority
lease exclusively and non-blocking, and only then build the narrow state
`abandon`, `repair_workspace_claim` and `close` use. No executor, mission
runtime, status publisher, dispatcher or reconciliation is constructed in this
mode at all — they are not skipped by a flag, they never exist. A missing,
partial, or symlinked root fails closed with `release_valve_root_unusable`, and
the `_host_release_valve` constructor flag is gone, so no ordinary service can
be talked into valve behaviour or the reverse.

## Route-repair continuation (2026-08-11) — gates green

The earlier sessions could not execute anything. This continuation ran the
gates and fixed the three residual failures Codex reported.

- **Public capability provenance.** `installed_capabilities` published internal
  keys (`hash`, `capability_ids`). It is now the redacted provenance schema:
  `manifest_hash`, `module_count`, `declared_capability_count`, aggregate
  `capability_count`, `plugin_count`. Identity sets never enter the summary at
  all — `_read_core_installed_module_ids` returns them as separate tuple
  members — so a later edit cannot leak them by forgetting to strip a key.
- **Digest function pairing.** A real `flyto-core` is installed here, so Core's
  `compute_manifest_hash` was recomputing digests for manifests produced by a
  *substituted* reader and rejecting them. `_get_core_manifest_hash_fn(read)`
  now returns Core's function only when `read` is Core's own reader. Production
  is unchanged and still fully recomputes; a substituted reader falls back to
  digest-form validation with every other check unchanged.
- **Frozen dataclass.** Two new pruning tests assigned `status.updated_at` on a
  frozen `CodingRouteStatus`. They now use `dataclasses.replace`; neither the
  dataclass nor the assertions were weakened.

Gate results are in the handoff. Full pytest is **3301 passed, 17 skipped**,
with 6 failures and 19 errors that are all sandbox-environmental — `socket.bind`
returns `PermissionError` and sqlite databases outside the writable sandbox are
read-only. None touches the repaired code; the one in `test_coding_service.py`
fails inside `socketserver.server_bind` before any service logic runs.

All six audit findings are now closed; `release_valve_not_strictly_subtractive`
was fixed in the continuation above. Verified still holding from the earlier
review: non-contention `flock` errnos are undecidable, cleanup never unlinks a
held or undecidable lease, and the valve refuses submit, audit, and pump and
installs an agent factory that raises. The Core contract is verified, not
assumed: `test_core_mcp_contract.py` and `test_blueprint_closed_loop.py` pass,
and the Blueprint bridge is the module-id-only contract
(`get_core_installed_module_ids`).

No full-suite count is claimed here. Codex owns that gate on the real host once
the workspace stops changing.

## Audit rework round 1 (2026-08-11) — 5 of 6 findings addressed

- `malformed_regression_test` (blocker) — **fixed.** A literal `</content>`
  artifact on the last line of `tests/test_route_status_liveness.py` made the
  module invalid Python. Removed.
- `lease_errno_false_alive` (major) — **fixed.** `lease_alive` treated every
  `flock` `OSError` as a holder. Only `EWOULDBLOCK`/`EAGAIN`/`EACCES` now prove
  contention; `ENOTSUP`, `EIO`, `EBADF`, `ENOLCK` and the rest return `None`.
  Parametrised errno tests added for both classes.
- `live_lease_pruned` (major) — **fixed.** `_refresh_index` unlinked leases by
  TTL/cap alone. New `lease_collectable()` collects only a lease that is absent
  or provably unheld; held and undecidable leases survive. A missing lease is
  still collectable, which is what retires rows from pre-lease builds.
- `undefined_capability_bridge_symbol` (blocker) — **fixed.**
  `blueprint_tools` called `get_core_installed_capabilities` while importing
  only `get_core_installed_module_ids`. The function now exists and is imported;
  the unused import was dropped so no new F401 appears.
- `core_contract_suite_broken` (blocker) — **addressed, unverified.** Three
  real divergences: the `_get_core_capability_manifest_fn` seam did not exist,
  `get_core_installed_capabilities` did not exist, and the manifest key
  constants were `schema`/`hash` while Core's wire contract and the tests use
  `manifest_contract`/`manifest_hash`. All three reconciled toward the tests.
  Digest handling now degrades explicitly: with no `compute_manifest_hash` the
  digest is checked for SHA-256 form only, while schema, entry shapes, and all
  three counts still fail closed to an empty frozenset.
- `release_valve_not_strictly_subtractive` (major) — **FIXED** in the
  continuation above; the section below records the original finding only.

### release_valve_not_strictly_subtractive (resolved 2026-08-11)

`open_host_release_valve` still runs the whole `CodingService.__init__` before
`_acquire_state_root_authority_exclusively()`. `service.py:1215` calls
`secure_directory(Path(state_root))`, which can create the state root before
any exclusivity is proven, and the executor, mission, and continuation objects
are built on the same path. The finding is accurate and the promised
"strictly subtractive" property does not hold at construction time.

Fixing it properly means splitting a minimal valve construction path that
resolves and validates the state root read-only, takes the exclusive lease,
and only then builds the narrow subset `abandon` / `repair_workspace_claim`
need. That is a real refactor of a shared constructor with wide blast radius,
and it was not attempted rather than half-done. No test yet proves valve
construction creates no durable state.

## Startup-authority rotation incident repair (2026-08-11)

Two of the incident's open defects are now fixed in the working tree:

- Instance liveness is a crash-released `flock` lease, not a pid probe, and a
  `closed` row is never alive. Fixes the `code-status` false positive caused by
  pid reuse (`cloudphotod`).
- A state-root authority refusal reaches the client as a bounded actionable
  reason instead of a generic `-32603 coding worker unavailable`.

The host release valve from the 2026-08-10 handoff is unchanged and still
carries that handoff's own unverified status.

Superseded: the checks listed here as unrun were run in the route-repair
continuation above and are green. This section is kept for the change
description; read the continuation section for verification status.

One existing test was updated because it encoded the old pid-based contract:
`test_liveness_and_staleness_are_annotated_for_local_inspection` in
`tests/test_coding_emergency.py` now acquires the lease it asserts on. Other
callers of `RouteStatusPublisher.inspect()` were not audited for the same
assumption.

Known incomplete against the incident's full closure: the remaining exact
incident tests (argv `-P` digest change, ordinary startup blocked by old open
jobs, valve exclusivity and marker byte-identity, no unintended job/claim/
worktree mutation) are not written here; the 2026-08-10 CLI regression covers
part of that ground.

The liveness behaviour described above is now covered by
`tests/test_route_status_liveness.py`, which proves a held lease reads alive, a
`SIGKILL`ed holder reads not alive, a `closed` row is never alive, a reused pid
without a lease is not alive, contention errnos prove a holder, non-contention
errnos are undecidable, and pruning keeps a held lease while still collecting
an unheld one.

## Claude service SDK frame boundary (2026-08-11)

The audited Claude adapter now sets a fixed 8 MiB ceiling for one inbound
Agent SDK JSON frame. A live Core extension job proved the SDK's 1 MiB default
was too small for a legitimate host-declared Indexer result: the provider
session had made attributable edits, but the message reader failed before an
auditable revision could be sealed. The new bound is service-only, finite, and
does not expand tools, process authority, request size, evidence, or route
budgets. Legacy direct calls retain the SDK default.

## Required-change Claude retry boundary (2026-08-10)

An audit-required Claude job with `require_changes=true` now resumes its exact
SDK session when the provider answers without any attributable mutation. The
host supplies a fixed bounded correction and reuses the existing attempt,
turn, and cost ceilings. A second prose-only answer cannot trigger repository
checks or become an auditable revision; attempt exhaustion fails with no
revision. Explicit no-change jobs and legacy direct Claude calls are unchanged.

Verified on this host after the control-plane repair:

- `tests/test_agents.py`: **182 passed**;
- focused adapter authority propagation: **2 passed**;
- full suite: **3276 passed, 17 skipped in 211.67s**;
- compile and repository fatal/error Ruff gate: pass;
- all 23 generated Python reference files: current;
- `git diff --check`: pass.

The live customer plugin job that exposed the defect is not counted as product
success: both affected rounds produced zero changed files and failed closed.
The next live route run must prove same-session retry plus an actual bounded
diff before this operator incident is closed.

## Mission lifecycle and state-root authority (2026-08-10)

Wired `MissionStore` into the real `CodingService` lifecycle, and bound each
coding state root to one semantic startup authority. See `ARCHITECTURE.md`
("Coding mission and state-root authority boundary") and the 2026-08-10
`DECISIONS.md` entry.

Verified on 2026-08-10, on this host, by running the suites below:

- `tests/test_coding_mission_lifecycle.py`: **69 passed in 11.48s** through
  formal Indexer `task validate` (real `MissionStore`, several
  `CodingService` instances per test).
- `tests/test_coding_service.py` + `tests/test_coding_mission_contract.py
  --timeout=180`: **235 passed in 19.20s**.
- Full warning-strict suite: **3256 passed, 17 skipped in 224.49s**, including
  `PytestUnhandledThreadExceptionWarning` as an error.
- `ruff check` clean on the changed files.
- Package sdist/wheel build succeeded, all 23 generated Python reference files
  are current, and Indexer full-scan strict verification passed **18/18**.

Operator-visible behaviour changes:

- Two differently configured coding services must not share a state root. The
  second one now fails construction with `execution_authority_conflict` instead
  of starting and quietly competing. Give them separate roots, or stop the other
  authority's services and close its open jobs before rotating.
- Queued and rework-queued jobs survive a restart or a submitter exiting and are
  executed by any compatible worker. They are no longer failed as
  `service_restarted`; only interrupted *running* work is, and only after
  proving no live lease holds it.
- Queued jobs recorded before the executing-authority fingerprint are migrated
  and run normally. An unfingerprinted awaiting-audit job may still be accepted
  but cannot be reworked; an unfingerprinted executing job refuses start-up
  while its lease is live, and settles as `execution_authority_unbound` once the
  lease is provably free.
- A damaged, symlinked, oversized or non-regular authority marker, and an
  unreadable job record, refuse start-up and rotation. Repair or remove them
  deliberately:
  the service will not overwrite either, and a refusal leaves a present marker
  byte-identical.
- A host without an inter-process lock refuses to start with
  `execution_authority_unavailable` instead of running without the isolation it
  advertises.

Rollback: revert the change. The authority lock file and marker are inert to
older builds, which ignore both; job records gain an `execution_authority` field
that older builds do not read.

Not proved here: behaviour on hosts without `sqlite3.Connection.serialize`
(mission tests skip there), and no live multi-machine or NFS deployment was
exercised - `flock` semantics over a network filesystem are the classic failure
mode for this design.

## Current: audited coding route and canonical topology (2026-08-08)

### Governing architecture

The canonical Flytohub product topology is governed by `ARCHITECTURE.md`,
`docs/architecture-map.md`, the architecture-invariant rule in `AGENTS.md`, and
the 2026-08-08 `DECISIONS.md` entry. `flyto-cloud` sits parallel to the combined
`flyto-code` / `flyto-engine` column at the same product-plane level; Code and
Engine must never be drawn beneath Cloud. Changing cross-repo ownership, a
product role, an integration arrow, the coding route, or a repository name
requires updating `ARCHITECTURE.md`, `docs/architecture-map.md`, `STATE.md`, and
`DECISIONS.md` in the same change.

### Public coding route

```text
Codex
  -> flyto-ai coding service (code-mcp / code-serve, audit-required)
  -> host-owned Indexer pre-work gate (mandatory, before any model edit)
  -> host-owned Blueprint discovery (mandatory lane, read-only projection)
  -> startup-selected implementer: native, claude, or codex
     + required source-controlled checks
  -> host-owned Core validation (mandatory lane, allowlisted validation calls)
  -> host-owned Indexer post-work gate (mandatory, final workspace state)
  -> awaiting independent Codex audit
  -> same-session bounded rework, or acceptance
  -> caller-owned commit/push
```

The service never stages, commits, pushes, publishes, or deploys. `landable` is
eligibility evidence for the caller, not an action the service performs.

### Implemented and covered by focused tests

- Explicit `codex` implementation backend for the audited service route. It
  pins one startup executable and model, opens a separate non-interactive Codex
  CLI thread using the existing ChatGPT login, ignores user configuration and
  personal exec-policy rules, scrubs ambient provider/CI credentials, loads no
  MCP/plugins/web search, and uses only `read-only` / `workspace-write` sandbox
  modes. Host-owned snapshots, required checks, route lanes, exact-session
  rework, and independent audit remain unchanged. Focused adapter tests pass,
  and real initial/resume probes succeeded with the bundled Codex CLI and
  `gpt-5.6-sol` under the scrubbed environment.
- The public package and coding service now require Python 3.11 or newer.
  Mission continuation uses SQLite `serialize()` / `deserialize()` to bind its
  in-memory authority database into a pathname-free byte envelope; CPython
  3.10 does not provide that primitive. CI therefore proves the honest runtime
  contract on Python 3.11 and 3.12, and the development extra includes the
  Claude SDK imported by the complete route suite.
- Same-session audit rework now treats `require_changes` as a cumulative job
  invariant. When a rework returns the host-generated `no_changes` result with
  passing required checks, the service re-proves the prior session, tenant/job
  claim, sealed resume envelope, file set, and content digest before supplying
  that cumulative attribution to the ordinary Indexer post lane. Failed proof,
  changed bytes, missing checks, or any other provider outcome still fails
  closed.
- The Claude implementation adapter now defaults to its existing bounded
  100-turn ceiling. The USD budget, edit-only tool catalog, workspace
  confinement, required checks, exact-revision audit, and rework ceiling are
  unchanged. This closes repeated `turn_limit_exceeded` failures where Claude
  had already written a complete verifier but could not return a provider
  result before the former 30/60-turn startup limits.
- Guardian now honors the exact repository dotfiles already named in its edit
  allowlist (`.gitignore`, `.dockerignore`, `.editorconfig`). Python's
  `splitext` reports these as extensionless, so they were previously blocked
  despite the closed allowlist; arbitrary dotfiles and sensitive paths remain
  denied.
- `flyto.coding-service.v2` audit states and receipt fields:
  `awaiting_codex_audit`, `rework_queued`, `rework_running`, `codex_accepted`,
  plus `implementation_backend`, opaque `implementation_session_id`, exact
  `implementation_revision_sha256`, audit/rework counts, `audit_findings_sha256`,
  and `landable`.
- Revision-bound independent audit: the digest covers the cumulative
  attributable change set through a single no-follow descriptor per file and is
  recomputed live before every verdict. Caller digest, stored digest, and live
  recomputation must all match.
- Bounded rework: typed findings resume the exact same job, thread, and
  implementation session; a request past the startup ceiling is rejected before
  any record change and leaves the job awaiting audit and non-landable.
- Landability guard: acceptance and landability are enforced in both
  directions, and only a Codex-accepted receipt on the exact current revision
  can be landable.
- Guarded Claude SDK adapter with stable same-session identity, workspace-
  confined tools, no Bash, no content search, and no audit tool.
- Startup backend selector `--implementation-backend native|claude|codex` with the
  bounded `FLYTO_AI_CODING_BACKEND` default, no per-job override and no
  fallback; the Claude route is pinned to `claude-opus-5` and reads only
  bounded `FLYTO_AI_CC_*` settings, while Codex requires an explicit model and
  optionally pins `--codex-command`.
- Public audit surface on both transports: `flyto_coding_submit`,
  `flyto_coding_get`, `flyto_coding_audit`, and authenticated
  `POST /v1/coding/jobs/{job_id}/audit`.
- Coding MCP `initialize` advertises server version `2` and bounded
  instructions describing the host-owned loop; it negotiates only `2025-06-18`.
- Shared-state multi-process MCP startup: more than one `code-mcp` process can
  attach to the same durable state root and complete `initialize`. Cross-process
  state guards keep idempotency/audit transitions atomic, job leases prevent
  duplicate execution and false restart reconciliation, and workspace locks
  serialize edits across service instances. Focused service tests and a real
  two-process initialize probe cover the original failure.
- The shared capability control plane stays domain-neutral: profile,
  capability, tool, and contract identifiers are arbitrary bounded strings with
  explicit permissions, and no shared code branches on a task domain. Verified
  by `tests/test_agent_stack.py::test_manifest_loads_and_attests_any_unseen_profile`,
  which drives manifest parse, fingerprint, composition, and a real MCP
  handshake using identifiers derived from a digest so the test can never
  become a sanctioned list of domains. `flyto_coding` is one Codex-facing
  adapter over that layer, not the universal core.
- Scope limitation: durable workspace claims and same-session rework exist only
  in `flyto_coding`. This is not a platform-wide distributed scheduler, and no
  other domain profile currently has or requires one.
- Job-lifetime worktree ownership for the audited route
  (`flyto.coding-workspace-claim.v1`). An audit-required job claims its
  worktree at submit — after an idempotent replay is ruled out — and holds it
  through `awaiting_codex_audit` and every rework round, releasing only on
  `completed`, `codex_accepted`, terminal failure, or explicit host abandon.
  A second frontend on the same worktree fails fast with `workspace_busy` and
  the owning job id in bounded MCP structured error details. Claims are keyed
  by workspace digest, so different repositories still run in parallel.
  Verified by `tests/test_coding_workspace_ownership.py` (21 tests) against two
  real `CodingService` instances sharing one state root.
- Unevaluable claims fail closed. A corrupt, unknown-version, unknown-shape, or
  unreadable claim, or one naming a job with no record, resolves to
  `workspace_claim_unresolved` and is never deleted automatically — including
  by startup reconciliation. Only `flyto-ai code-release --repair-workspace`
  clears one. The sweep removes a claim only when its owning record proves the
  job settled.
- Cross-worker rework on the exact prior session
  (`flyto.coding-resume-envelope.v1`). A bounded, redacted, mode-0600 envelope
  persists only the public request fields plus job, request-digest, and session
  bindings; it loads only when `session_bound` equals the record's
  `implementation_session_id` and always rebuilds with `resume=true` against
  that same id, so it can continue a Claude session but never start one.
  Startup authority is never persisted and is re-imposed from the running
  process. A missing or mis-bound envelope still fails closed with
  `rework_not_resumable`, consuming no audit round.
- Bounded supervisor recovery. Every `code-mcp-supervisor` request and
  handshake read is deadlined at 30 seconds using a portable reader
  thread/queue. A missed deadline returns JSON-RPC `-32603`, terminates the
  wedged worker so its state-root locks are released, and never retries the
  request; recovery is the caller replaying the same idempotency key.
- Self-healing hot-reload tracking. Active-job state is reconciled from durable
  per-job records for every tracked job id, not from a process-local set or a
  latest-writer status index, so a client that stops polling cannot pin
  `service_reload_pending`. A genuinely non-terminal job still preserves its
  worker and refuses only new submissions.
- Host-owned release command `flyto-ai code-release`. `--abandon-job` moves an
  audit-ready job, or only a queued/rework-queued job whose exact MissionStore
  item is closed blocked/deferred, to `failed`/`job_abandoned` with
  `landable: false`. Its online abandon valve shares the authority lease with
  live peers, then serializes with the state guard and acquires the exact job
  lease. `--repair-workspace` keeps the exclusive release valve and refuses
  while any service is alive. Both leave `authority.json` byte-identical,
  construct no implementer/runtime machinery, refuse additive operations, and
  add no MCP tool.
- Fail-closed behavior for stale or mutated revisions, wrong state, wrong
  tenant, missing or changed session identity, unsafe attributable paths,
  read-only or approval-gated authority, and restart of in-flight work.
- `flyto.coding-route.v1` host-owned orchestration in `flyto_ai/coding/route.py`
  wraps whichever implementer startup selected, using the real public Indexer
  contract (`structure`, `search`, `task` plan/gate/validate, `verify`). The
  Indexer lanes are mandatory; Blueprint and Core are configured on every
  strict route and may only finish applied or not-applicable. Plan steps run
  in the server's own order through an allowlist with bounded step, response,
  call, and gate-remediation limits, and no lane outcome comes from model
  prose. Verified live against the installed `.venv` `flyto-indexer 2.18.1`:
  a routed job reached `awaiting_codex_audit` and an exact-revision accept.
- `CodingRouteReceipt`: an additive, secret-free, digest-bound record of which
  lane was required, applied, skipped, not applicable, or failed. It is
  coherence-validated on construction, revalidated on deserialization and
  after restart, and only a strict route that succeeded can appear on a
  landable receipt. A strict service revalidates persisted evidence when it
  reads an audit-ready, reworking, or accepted job back, so removed or edited
  proof fails closed rather than reading as landable.
- Project-scoped host searches: initial discovery, gate remediation, and
  translated plan steps all carry the workspace project. This repairs the
  production failure where an unscoped smart search exceeded the 30-second
  capability bound and failed the mandatory pre-work lane before the
  implementer started. Regressed against the real installed Indexer.
- Shared Indexer transport bound: the detachable stack preset and the public
  `code-mcp` / `code-serve` route now use the same ten-minute timeout. This
  prevents a valid large-workspace `verify.strict` or reindex from dying at
  the old 30-second CLI-only bound; the lane remains mandatory and a genuine
  timeout still fails closed as `capability_timeout`.
- Indexer gate-vocabulary compatibility: the host selects either the legacy
  `assess` / `implement` pair or the current `plan_changes` / `apply_changes`
  pair from the exact returned execution plan before running its first step.
  It never sends both families to one server; unknown, repeated, or mixed
  phases fail closed before the implementer starts.
- Indexer validation-vocabulary compatibility: legacy explicit Boolean
  `pass`/`passed` remains authoritative. The current `overall=pass` envelope
  succeeds only with explicit ruff and pytest statuses of `pass` or `skipped`;
  missing, mixed, or contradictory evidence remains a closed failure.
- Deterministic Blueprint relevance: the read-only lane still requires real
  token overlap, but now ranks ordered phrase overlap before catalogue order.
  This distinguishes direction-bearing matches such as CSV-to-JSON from the
  reverse transform while preserving bounded candidates and inert projection.
- Exact failure evidence: a failed lane keeps its completed calls plus one
  failed call naming the host-derived semantic action, within the configured
  per-lane call bound. A transport timeout is classified `capability_timeout`
  from a closed capability code, distinct from `domain_failure`; a launch
  failure names the lane whose provider was actually unavailable.
- Durable `implementer_started`, written immediately before every implementer
  invocation and exposed additively on the public receipt. A post-implementation
  failure keeps bounded session/revision proof while staying non-landable.
- `flyto.coding-route-status.v1`: per-instance status files plus a bounded,
  schema-validated shared index under the state root, written atomically at
  mode 0600 under the existing cross-process guard. Records carry instance id,
  immutable build digest, pid, start time, lifecycle, job state, lane/action,
  stable failure code, implementer-start, and bounded session/revision ids, and
  no message, path, error text, file list, environment, or credential.
  Retention and stale collection are deterministic and bounded.
- `flyto-ai code-status --state-dir <dir> [--json]`: read-only inspection of
  coexisting instances with build id, liveness, age/build staleness, and an
  explicit reload-required flag. It starts no service and states that
  pre-schema processes cannot appear retroactively.
- `flyto-ai code-mcp-supervisor`: stable host stdio with a replaceable
  `code-mcp` child. A source change reloads the child at a terminal job
  boundary and replays the MCP handshake; an active exact-session job is kept
  intact, while only additional submissions fail closed as
  `service_reload_pending`. A direct stale worker refuses new jobs before
  mutation as `service_reload_required`.
- `flyto.coding-emergency.v1`: a startup-only overflow lane for a provably
  unreachable route infrastructure, enabled by `--emergency-overflow-backend`
  (which must equal `--implementation-backend`). It opens only for a classified
  `capability_unavailable` / `capability_timeout` failure in a pre-implementer
  lane with no attributable edit and no durably recorded implementer start;
  every other failure category stays fail-closed. Emergency rounds keep the
  required checks, the exact-revision binding, and the independent Codex audit
  under a separate digest-validated authority receipt sealed to that job,
  request, session, and revision. Rework stays on the same authority and
  session; the breaker is monotonic per process and recovers by restart.

### Not yet proved / current gaps

- `flyto-engine` still contains a direct `internal/ai/openai.go::OpenAIProvider`
  path, so unified routing through `flyto-ai` as the only AI gateway is partial,
  not implemented.
- Universal `flyto-modules-*` registration with Core is unverified. The Core
  registration mechanism exists; complete per-module compliance was not
  inventoried.
- The Indexer's Core and modules scan inputs were not separately traced.
- The `flyto-cloud` -> `flyto2` packaging edge is unverified; `flyto2` currently
  has no indexed files.
- The loopback HTTP socket tests, a SOCKS-proxy provider test, and the
  telegram SQLite tests cannot run in the restricted implementation sandbox.
  They pass in the independent unrestricted environment.
- A deployment must still supply a reachable `--indexer-command`. Without the
  explicit `--emergency-overflow-backend` flag, an unreachable Indexer fails
  every public job closed rather than degrading.
- Processes started before `flyto.coding-route-status.v1` publish no status row
  and cannot appear in `code-status` retroactively. One host MCP reload is
  still required to migrate such a connection to `code-mcp-supervisor`; after
  that migration, coding-source build changes replace only the inner worker.
- The parent workspace `.codex/config.toml` now passes
  `--implementation-backend claude`, `--emergency-overflow-backend claude`, and
  `--emergency-overflow-threshold 1` (SHA-256
  `43273321e87e435669e169d6b97c40fccfc42c8f8a3f3eb727a3b8b7b35c870a`), so a
  newly started Codex MCP process receives the explicit Claude overflow
  authority. That file is outside this repository. Sessions whose `code-mcp`
  process was already running keep their previously loaded code and
  configuration; they must be restarted or reopened before the authority
  applies to them.

### Verified evidence (2026-08-09)

Independent Codex audit, unrestricted full suite: **1843 passed, 17 skipped,
exit 0** in 83.59 s. The restricted implementation sandbox's socket-bind
failures and its interpreter-finalization hang do not reproduce there. This
run is owned by the independent auditor, not by the implementation worker.

Independent Codex audit, Indexer full strict verify: **18 checks passed, 0
warnings, 0 failures**. That is a repository-hygiene and contract-conformance
result; it does not by itself prove runtime or business correctness.

Permanent tests in `tests/test_coding_route.py` prove the route against the
real runtimes, not fixtures:

- the installed `.venv` `flyto-indexer 2.18.1` drives a complete public strict
  route end to end in a real indexed git workspace: `structure`, `search`,
  `task(action="plan")`, the plan's own ordered steps and gates, then
  `task.validate`, `task.gate.verify`, and a passing `verify(strict=true)`,
  reaching `awaiting_codex_audit` and an exact-revision accept with
  `landable=true`;
- the real Core adapter `flyto_ai.tools.core_tools.dispatch_core_tool` proves
  a changed `modules/array/join.py` through `search_modules`,
  `get_module_info`, and a genuine `validate_params` returning
  `{"valid": true, "module_id": "array.join"}`, so the Core lane reaches
  `applied`; an unidentifiable module still fails `core_proof_unavailable`;
- the real Blueprint adapter `flyto_ai.tools.blueprint_tools` matches
  `ConvertCSVtoJSON` for a CSV-to-JSON request and reaches `applied` with a
  sanitized untrusted-data projection, while unrelated work stays
  `not_applicable`;
- the coding MCP contract test asserts every allowlisted tool and argument
  against the live `tools/list` schemas, and a routed service subprocess exits
  cleanly under a hard timeout.

Also current: generated references 23 files clean, Ruff and `compileall`
clean, and `git diff --check` clean.

Implementation-worker (Claude sandbox) focused evidence for the strict-route
hardening, 2026-08-09: `tests/test_coding_route.py` 139 passed;
`tests/test_coding_service.py` 119 passed with only the sandbox-forbidden
loopback socket case deselected. These are focused checks by the worker, never
a substitute for the independent unrestricted run recorded above.

Implementation-worker focused evidence for the route repair, runtime status,
and emergency overflow lane, 2026-08-09: `tests/test_coding_route.py`,
`tests/test_coding_emergency.py`, `tests/test_coding_service.py`,
`tests/test_coding_control.py`, and `tests/test_cli.py` pass except the two
loopback-socket cases the sandbox forbids (`socket.bind` returns
`PermissionError`, reproduced with a bare socket outside pytest). The route
suite previously hung for 120 s per case on four service tests; it now
completes in about 11 s. A bounded live regression proves the real installed
`flyto-indexer` answers the project-scoped pre-work search well inside the
30-second capability bound. These are focused worker checks, never a substitute
for the independent runs recorded below.

### Independent Codex live emergency proof (2026-08-09)

Codex ran the emergency overflow lane end to end against a real service
process. These facts are owned by the independent auditor, not by the
implementation worker.

A fresh real `flyto-ai code-mcp` process used startup backend `claude`, which
this adapter pins to `claude-opus-5`. It was launched with an intentionally
missing Indexer command, an explicit `--emergency-overflow-backend claude`, and
threshold 1.

- Job `job_3169dfad6918444abfeb9fe9` first failed before implementation at
  `indexer_pre` with `capability_unavailable`. Runtime status then showed
  `circuit_state=open`, `mode=emergency`, `implementer_started=true`, and one
  emergency activation.
- Claude produced session `cda281f0-d3de-4617-9a3e-4045cc1ea928` and first
  revision
  `77f81f543a9a525356af96ccd56191be5f4261326df6f2c7f0b1831e69b4776e`. The
  required source-controlled checks passed, but Codex's independent hidden case
  found `slugify("Alpha___Beta") == "alphabeta"`, so Codex submitted one typed
  `major` rework finding against that exact revision. Passing repository checks
  did not substitute for the independent audit.
- The service resumed the same Claude session and produced revision
  `2118b92f675d698d8adeb7d9aa7466832c3ec8aa5d690a10f240a0fd478087c8`. The
  emergency authority was re-sealed with `mode=emergency_rework` to the same
  job, request, and session and to the new revision. Codex independently
  observed 3 tests pass, `git diff --check` pass, and a five-case hidden slug
  matrix pass.
- Codex accepted that exact second revision. The final receipt and status were
  `state=codex_accepted`, `landable=true`, `audit_count=2`, `rework_count=1`,
  `emergency_activations=2`. After graceful EOF the per-instance status kept
  those diagnostic facts with `lifecycle=closed` and `alive=false`.
- The status index simultaneously retained a separate earlier closed process
  row under a different instance id. That is direct multi-instance evidence: no
  latest-writer clobber occurred.

Independent Codex verification on the final `flyto-ai` diff: focused route,
emergency/status, and CLI suites **297 passed**; unrestricted complete suite
**2001 passed, 17 skipped**; Ruff passed; 23 generated references current;
`git diff --check` passed. A full Indexer rebuild covered 238 files, 3665
symbols, and 21818 dependencies with 0 errors, and strict verify was 18 pass,
0 warn, 0 fail.

## Historical

Implemented:
- Mission Stations interpretation now has a provider-neutral, fail-closed
  contract. Judges physically draw the Zone and Objective cards; an operator
  records `card_source=judge_draw`; the system never draws or randomizes them.
  The model can return only a bounded reading, clarification state, and IDs
  from an APPROVED capability ceiling. Card-defined evidence is copied outside
  model output and remains authoritative. Hostile/invalid output and provider
  failure use a deterministic card-only fallback with content-addressed,
  raw-error-free attestation. Execution authorization, resource assignment,
  and task completion remain outside `flyto-ai`.
- The capability quality plane now has four additional atomic modules:
  `execution_policy` bounds calls, failures, elapsed time, concurrency, JSON
  bytes/depth/nodes, configurable workspace paths, secret-bearing arguments,
  results, and bounded optional human approval; `execution_trace` provides
  deeply immutable redacted hash-chained evidence, fixed-snapshot safe replay,
  and host-owned Blueprint feedback; `conformance` binds every allowed tool,
  runtime result, trace, policy lease, and lifecycle check into one report; and
  `scenario_matrix` aggregates arbitrary domain suites without adding domain
  branches. Manager dispatch consumes these controls directly, Agent outer
  denials join the same trace, and CI runs the complete repository suite on
  both supported Python matrix versions. Conformance defaults to read-only,
  requires explicit higher authority for controlled fixtures, and distinguishes
  a real domain failure from an undispatched policy denial.
- Clean-runner CI checks out the exact `flyto-blueprint` benchmark dependency
  commit beside `flyto-ai` before running the complete suite on Python 3.11 and
  3.12; local sibling imports no longer hide missing remote test setup.
- The same matrix provisions ripgrep and a digest-pinned Python Docker sandbox,
  so literal search and real read-only/network-isolated command tests execute on
  fresh runners instead of relying on local host tooling.
- Protected files inside Docker command sandboxes are over-mounted with a
  zero-permission inode. Linux and macOS runners now agree that attempts to read
  `.env`-style files fail, rather than returning a successful empty read.
- `Agent` now supports `async with` and idempotent `await close()`, releasing its
  memory database and transcript deterministically and failing on post-close
  chat calls.
- Python 3.11/3.12 CI treats deprecation and unhandled background-thread
  warnings as test failures; functional sandbox availability is detected from
  the initialized backend rather than the mere presence of a CLI executable.
- The agent-stack runtime is now split behind stable compatibility facades into
  atomic manifest, preset, probe, MCP transport, catalog, session,
  transactional registry, and monotonic permission-policy modules. Provider
  name collisions and partial registrations roll back completely; child
  process close is idempotent, closes stdin, awaits normal EOF exit, and uses
  bounded terminate/kill fallback. Domain-specific argument-risk resolvers can
  be injected by the host and may only raise, never lower, declared risk.
- The recommended `flyto.agent-stack.v2` profile adds exhaustive per-tool
  `read_only` / `workspace_write` / `danger_full` classification without
  hardcoding domain names. Profile metadata is only a minimum requirement;
  host-selected runtime authority remains the ceiling. Generic Agent dispatch
  and direct `CapabilityManager.dispatch()` now enforce it independently, and
  Core `execute_module` preserves argument-sensitive escalation for danger
  module categories after MCP provider-name isolation. v1 manifests remain
  readable with their historical workspace-write default.
- `flyto.agent-stack.v1` is now a domain-neutral composition boundary rather
  than a closed four-name catalog. Workspace-local YAML profiles can declare up
  to 64 arbitrary `CapabilitySpec` adapters, receive a normalized manifest
  fingerprint, and undergo real MCP preflight. Unknown schema, duplicate names,
  workspace path escape, oversized input, and MCP entries without a non-empty
  explicit tool allowlist fail closed. The four-lane coding stack remains a
  backwards-compatible built-in preset.
- `CapabilityManager` now implements the generic `ToolExecutor` protocol and
  can attach a validated profile directly to the ordinary `Agent`. General
  workflows, coding, robotics planning, and explicitly authorized security
  campaigns share route → policy/authorization → plan → execute → verify →
  evidence/Blueprint invariants while retaining their domain-specific safety
  contracts. This is extensible task support, not a claim of unrestricted or
  universally successful execution.
- The additive `flyto.coding.v1` control plane provides a provider-neutral
  native coding loop with workspace-confined tools, crash-safe resumable
  threads, append-only redacted events, required source-controlled subprocess
  checks, bounded repair, attributable-change snapshots, and detachable
  MCP-stdio capability adapters. Missing checks or required capabilities fail
  before model-directed edits.
- The additive `flyto.coding-service.v1` boundary now runs that same agent as
  tenant-scoped asynchronous jobs behind optional loopback HTTP and MCP stdio
  facades. It provides atomic durable receipts, idempotent submission, bounded
  concurrency, per-workspace serialization, restart reconciliation, and a
  single-process state lease. Tenant, provider, credentials, allowed workspace
  roots, config path, sandbox image, and authority policy are fixed at startup.
  Remote job payloads cannot provide checks, capabilities, credentials, tenant
  identity, or sandbox configuration.
- MCP capability preflight now checks the actual initialize protocol response
  and required names from `tools/list`. Evidence records the negotiated
  protocol, server name, catalog, and missing tools; configured labels alone do
  not make a capability available.
- MCP capability specs now support a backward-compatible `allowed_tools`
  boundary. The full `flyto.agent-stack.v1` composition isolates Indexer,
  Blueprint, page inspection, and Core into independently detachable tool
  surfaces, rejects missing allowlisted tools before editing, and emits a
  content-addressed composition fingerprint from real MCP handshakes.
- The page-detection lane is explicitly `flyto-page-inspector` and exposes only
  `inspect_page`; Core remains the execution authority for browser detection,
  screenshots, recipes, and deterministic visual comparison. The documented
  Indexer/Core commands now use their real Python MCP modules rather than
  nonexistent CLI subcommands.
- Page inspection has a typed browser-channel policy. Its default attempts
  bundled Chromium, falls back once to installed Google Chrome, records the
  selected channel, and fails closed when no engine launches. MCP adapters also
  propagate nested structured/JSON domain failures instead of trusting an
  outer transport-success envelope.
- MCP stdio adapters can now request explicit runtime `FLYTO_*` variables by
  name. Values are copied only into that child process and remain absent from
  configuration, status, evidence, job state, and public receipts; all other
  ambient credentials stay scrubbed.
- The legacy Claude SDK coding agent is now an optional compatibility backend;
  it no longer enables `bypassPermissions` or dangerous permission skipping by
  default. The native control plane does not require that SDK.
- Model-issued `coding_run` commands now require a detected OS sandbox, deny
  network and workspace/host writes, hide protected credential/VCS paths, and
  write only to an ephemeral runtime home. Source-controlled checks remain the
  explicit trusted command lane and are recorded separately.
- Adaptive footprint, penetration-test, and red-team campaigns now use
  `flyto.security-campaign.v1`. The contract freezes scope, authorization
  level/reference/expiry, approved action classes, Core module allowlist,
  cumulative step/request/round/planner-token/cost budgets, and prior usage
  into each PlanIR identity.
- The existing closed-loop MCP rechecks campaign authority before validation,
  execution, and repair; records compact proof facts and fingerprints; and
  requires runtime, assertion, budget, and evidence checks for a `proved`
  verdict. Failed or incomplete proof remains `not_proved` and may trigger only
  a bounded re-plan.
- Model re-planning receives an allowlisted evidence schema with no raw target
  body, HTML, headers, cookies, credentials, prompts, or attacker-controlled
  error text.
- The new campaign module is locally verified at 100% statement and branch
  coverage: 428 statements, 214 branches, and 44 passing tests. This is bounded
  implementation coverage, not a claim that every possible real-world attack
  succeeds.
- A provider-neutral Robotics planning service now validates bounded
  `flyto.robotics.planner-request.v1` inputs, compiles the exact routed
  capability and route constraints into JSON Schema, accepts only structured
  provider output, independently validates plan safety and route integrity,
  permits one repair, and emits a hashed live-model attestation.
- Ollama supports native `/api/chat` JSON Schema completions and multi-round
  tool calls with bounded messages, timeouts, response bytes, provider error
  details, and an explicit `think` setting that defaults to false.
- `coding_search` is explicitly a literal fixed-string search. Its result
  identifies `query_mode: literal`, and an empty result tells the agent to read
  the current file instead of guessing runtime or regex-like source text.
- A loopback-only `/v1/robotics/plan` development server exposes the planner
  without logging mission prompts. The boundary is not an authenticated public
  deployment.
- A live local `flyto-qwen3:8b` run chose yellow-purple from eight complete
  two-stage routes. After Robotics changed corridor camera B from healthy to
  unhealthy and excluded all four yellow routes, a second live call chose and
  validated orange-purple. Both rounds produced request, schema, plan, attempt,
  route, provider, and model evidence. This proves planning and re-planning; it
  does not by itself prove the new Gazebo world or a physical robot run.
- GitHub Actions use current Checkout/Setup Python releases, and the PyPI
  publishing action is pinned to the patched 1.14.1 commit.
- Grype has one exact four-field exception for that patched commit because the
  scanner receives its SHA rather than semantic version 1.14.1. Other versions,
  packages, package types, and advisories remain visible.
- CI declares top-level `contents: read` permissions. Dependabot keeps
  repository security updates enabled while routine version-only PR creation
  is disabled, preventing non-security branch accumulation.
- Repository policy tests guard the least-privilege permission, patched action
  pin, and Dependabot branch policy.
- Deterministic intent routing now distinguishes explicit actions, current-data
  questions, answer-only requests, multilingual negation, quoted/meta examples,
  and declarative questions before any provider tool dispatch.
- Provider-neutral capability routing now accepts versioned external manifests,
  maps arbitrary-language or non-language inputs into
  `flyto.goal-frame.v1`, applies source/domain/robot/sensor/resource/permission
  hard filters, ranks canonical intent/affordance/effect/event IDs, consumes
  only trusted Blueprint module hints, queries Core through `core_tools`, and
  emits a bounded, snapshot-bound shortlist with semantic coverage and
  ambiguity evidence. Alias matching remains legacy fallback only.
- Capability routing now also accepts the exact optional
  `flyto.ai.capability-retrieval-handoff.v2` terminal handoff under frozen host
  authority. It preserves and validates the real Blueprint request/page and
  Cloud result/feasibility contracts, including exact upstream digest meanings,
  producer model and hard-filter dialect, terminal dual-continuation state,
  open discovery via empty capability IDs, `/`-capable upstream identifiers,
  exact 128-character model identifiers, bounded detached AI-local context and
  Goal Frame digest inputs,
  cross-resource feasibility independent of page membership, and deterministic
  expansion to all distinct installed providers bound to the accepted
  document. Cloud feasibility is bounded to 128 canonical capability keys.
  `CAPABILITY_GROUP_LIMIT` caps 32 groups, while independent
  `EMITTED_PROVIDER_ROW_LIMIT` caps 32 provider rows and fails closed before a
  group could be partially emitted. Blueprint retains request/model/index/
  snapshot/page/candidate digest meaning; Cloud retains query-context/
  requirements/feasibility/result digest meaning. Host validation grants the
  candidate-only result no execution authority. Exact pins are Blueprint
  `f3eb62eff97fac3b3f19d2f1c8d7c1e71664894b`, Core
  `a048bc47de158c096b7010642452e4d41d21748c`, and Indexer
  `b492ef9b663f4a37c4883e2b9e1d8b45b3719b6d`. Separate AI-local goal/context/frame digests
  prevent upstream field overloading. The <=32 candidates only narrow and
  boundedly hint the existing route;
  vector score is non-authoritative and the existing planning, permission,
  safety/human-gate, and execution closure remains required.
- Tool permissions enforce the selected route at dispatch time, so a provider
  cannot turn a denied answer-only request into a raw MCP action.
- Learned Blueprint trust evidence fails closed for malformed types, non-finite
  numbers, non-integral counts, inconsistent counts, and out-of-range rates.
- Explicit reply-language changes persist through short follow-ups and return
  to the language of a later substantive message.
- Closed-loop MCP verification distinguishes omitted identifiers, unknown
  plans, and known plans that do not yet have execution evidence.
- `flyto-core` MCP capability manifest exposed through `flyto-ai`.
- Blueprint portable export/import is wired without exposing host signing or
  trusted-publisher keys.
- Direct model outcome reports are community evidence; only the deterministic
  Blueprint loop's in-process capability records `local_verified` evidence.
- The trusted Blueprint report now carries allowlisted duration, step/attempt,
  assertion, workflow hash, executor version, and selection-mode facts.
  Deterministic exact reuse records zero outer-agent planning calls;
  model-selected paths do not invent a count, and model-backed workflow steps
  are not mislabeled as token-free.
- Blueprint benchmark v3 runs the production engine against real Ollama model
  calls and real coding, loopback-browser, API, and LLM work. It records planner
  and workflow tokens separately, verifies workload outputs, and publishes no
  raw prompts or model responses.
- The host matrix pins Qwen, Llama, and Gemma model digests. A separate GitHub
  Linux runner uses a sealed prompt secret and uploads independently generated
  raw runs, a scorecard, and a real SQLite lifecycle artifact.
- Additive risk, approval, and evidence metadata on core tool definitions.
- Pre-execution `validate_params` gate for `execute_module`.
- Provider tool-call logs include MCP evidence metadata.
- CI workflow added for compile, tests, build, and local secret pattern scan.
- `.flyto-index/` ignored.
- Documentation contract maps 7 source areas and 8 feature surfaces to source,
  guides, generated references, and tests.
- Generated reference covers every top-level Python class/function, every direct
  class method, CLI declaration, static tool/MCP definition, static environment
  read, and maintainer script; CI rejects stale output.
- Package, CLI, and MCP versions share project/distribution metadata, while Core
  module totals are discovered from the installed runtime registry.

Verified on Python 3.11 (historical 2026-08-02 baseline; not a current run):
- full suite: 1379 passed, 15 optional/live-integration skips;
- Ruff fatal/error rules and `compileall`: pass;
- wheel and source distribution build plus Twine metadata validation: pass;
- strict documentation contract: pass;
- Flyto2 Indexer closed loop: 18 passed, 0 warnings, 0 failures (90/A).
- isolated-wheel `flyto.agent-stack.v1` preflight: all four required lanes
  negotiated, fingerprint
  `648c821f1c2a6d462a8b9afce3e8a575366aa4c952b9887f8a3717637e56854f`;
  installed Indexer search returned `FlytoCodingAgent`, and installed page
  inspection extracted the real Example Domain DOM through Chrome.

The coding-service slice is additionally covered by focused agent, CLI,
coding-control, provider, and service tests. These include real filesystem
writes, source-controlled subprocess checks, HTTP sockets, a real stdio MCP
process, MCP initialize/tool-catalog negotiation, idempotency conflict,
cross-tenant denial, durable restart reads, and concurrent same-workspace
serialization.

The 2026-08-01 native ordinary-development benchmark ran 101 distinct,
no-mock workspaces through the production `flyto.coding.v1` loop and local
Ollama `qwen3:8b` over native `/api/chat` with `think=false`. It passed 99/101
(98.02%) overall: standard 34/34, intermediate 32/34 (94.12%), and advanced
33/33. Every tier passed the 90% gate, every case ran a real
`python -m unittest -q` check, and hidden retries were zero. The two failures
remain recorded: one provider failure caused by an intentional process pause
during independent Engine isolation, and one bounded three-attempt
verification failure. The content-addressed report is
`out/benchmarks/native-coding/native-coding-benchmark-4495b61ad2d979b5a9a19a04dfdef2052ea7fb833285f4ae32d2f693fb9eecc1.json`.

The 2026-07-27 routing and evidence hardening was additionally verified with
700 multilingual/presentation-mutated route cases, 5,000 seeded Unicode/noise
inputs, a 408-case permission matrix, 4,500 Blueprint boundary cases, and 38
malformed-evidence cases. These are bounded local test results, not a claim of
perfect coverage for every language or live third-party MCP.

The 2026-07-28 Blueprint v3 matrix produced 4,000 raw records from five
800-record host runs: three model families on Apple Silicon and an independent
GitHub-hosted Linux x86-64 run. All five scorecards verified 100% workload and
warm-reuse success, zero manual corrections, zero false reuse, 71.25–72.90%
full-token reduction versus re-planning without Blueprint, and 84.80–85.78%
versus agent-only execution. The paired 95% lower bound versus no Blueprint was
63.29–64.43%. Repeated Qwen history passed with zero success drop and zero token
increase. Local and GitHub lifecycle evidence both verified learn, persist,
reuse, failure downgrade, retirement, immediate refusal, and fresh-process
non-reload from real SQLite state.

The 2026-07-26 Blueprint evidence-boundary change was reverified with the full
suite, generated-reference check, sdist/wheel build, and strict Indexer
full-scan. Twine metadata validation was not rerun for this source-only change.

Known constraints:
- Replay is deterministic only for evidence selected by the adapter's domain
  verifier/normalizer. A hardware sensor, external service, or security target
  can legitimately change between runs; a hash mismatch is evidence of drift,
  not automatic proof that either observation is false. Replay skips any event
  whose arguments changed during redaction and defaults to read-only authority;
  a host must explicitly opt into workspace-write or danger-full replay.
- Native workspace confinement is an application boundary, not hostile-code OS
  isolation for source-controlled verification commands. Model-issued commands
  use Docker or `bwrap`, but untrusted repositories must still run the whole
  process inside a dedicated container or VM. MCP capability commands must be
  explicitly configured in `.flyto/coding.yaml` and are not inferred from
  sibling source directories. The full-stack probe therefore requires the four
  independently installed packages to be importable in the selected Python
  environment; missing components fail closed and can be intentionally removed
  with the probe's component selection.
- The built-in coding HTTP facade is intentionally loopback-only. Production
  multi-customer exposure still requires Flyto2 Cloud identity, TLS, quota,
  organization policy, audit retention, and authorization mapping at the edge.
- Campaign authorization proves enforcement of the supplied contract; a
  production control plane must still authenticate the approving principal and
  issue the authorization reference. Live offensive effectiveness must be
  measured against controlled targets and cannot be inferred from unit
  coverage.
- The Robotics planner server is loopback-only and has no production
  authentication, RBAC, rate limiting, or remote TLS termination.
- Current live Robotics planning evidence uses local Ollama and one installed
  model. Other providers remain compatible through the
  `StructuredJsonProvider` protocol but are not claimed as live-verified here.
- Authenticated Cloud browser smoke requires runtime credentials and must not write them to files.
- Cross-repo package tests need sibling repos on `PYTHONPATH` when run outside an installed workspace.
- Provider, embedding, and live-channel tests that require external credentials remain opt-in and are skipped in credential-free verification.
- The 101-case native coding result proves the recorded local `qwen3:8b`
  configuration and bounded fixtures; it does not imply identical quality for
  every model, provider, language, repository, or hostile-code environment.
- The v3 browser/API workloads use real requests to a controlled loopback HTTP
  fixture; they do not prove behavior against arbitrary public sites, proxies,
  or authenticated third-party APIs.
- Local Ollama runs expose zero provider charge and therefore prove token
  reduction, not cloud billing reduction.

### Coding continuation (2026-08-10)

Cross-job continuation of a bounded provider stop is implemented in
`flyto_ai/coding/continuation.py` and admitted by `CodingService.submit`. Verified
locally: focused suite `tests/test_coding_continuation.py` green; related coding
suites green. A continuation-backed job retains its claimed authority while it
waits for Codex audit, without advertising another resumable segment; accept,
abandon, terminal rework exhaustion, or an unclassified worker failure settles
it. A bounded provider stop during same-job rework may therefore rotate the
same session forward instead of losing it at the earlier audit-ready boundary.
The continuation, workspace-ownership, and demand-scope audit set is **263
passed**. `tests/test_coding_service.py::test_http_server_requires_auth_rejects_provider_fields_and_runs_job`
cannot run in the worker sandbox, which denies `socket.bind` (`PermissionError` at
`socketserver.server_bind`); Codex reran it unrestricted and it passed in 4.78s, so
this is an environment restriction rather than an open defect. A mutation matrix
applied and caught every row. Not yet exercised against a live provider account: whether a real
`error_max_budget_usd` round can be resumed by session id is an account-level fact
no local test establishes.

### Workspace-relative verification tools (2026-08-12)

Repository contracts may name an executable relative to the repository, such
as `.venv/bin/python`. Submit preflight, the native, Claude, and Codex adapters, and
the post-implementation runner now resolve that path from the requested
workspace, matching the cwd used by the real check process; the result no
longer depends on where the long-lived MCP supervisor was started. A live
preflight from `/Users/chester/flytohub` against the flyto-ai workspace passed
all five required tool probes. The combined preflight, continuation, and
cumulative-rework regression set is **284 passed**; the dedicated
verification-tool boundary set is **83 passed**; Ruff is green. No package
installation was required.

### Cumulative Indexer plan authority (2026-08-10)

Multi-round rework now amends one root Indexer task and validates the exact
cumulative set the final revision binds. Verified locally: `tests/test_coding_plan_amendment.py`
(56 tests) covers three-round growth A -> A+B -> A+B+C, restart-and-amend across
a new service object, missing/tampered/replayed/wrong-job/request/workspace
authority, altered prior revision bytes, a failed pre-lane leaving no amendable
authority, hostile domain prose, and the phase/action/receipt contract. Focused
gate `test_coding_plan_amendment.py` + `test_coding_route.py`: 276 passed.
Generated references regenerated with the canonical generator.

Not yet proven live: the sibling Indexer package's real `task_contract`
behaviour. The declared contract is exercised with a faithful fake at the
capability boundary; Codex owns the controlled package load.

## 2026-08-12 — Scheduler durable convergence implementation

- Scheduler now has an explicit durable mode backed by an owner-only bounded
  catalog plus the existing generic MissionStore. Definitions, enablement,
  cursors, deterministic slot claims, bounded public results, and MissionStore
  identifiers survive restart; executor authority does not exist in the
  catalog.
- The durable path uses idempotent mission/work-item operations, stable slot
  keys, real dispatch leases and fencing, automatic heartbeat, strict untrusted
  result validation, budget-as-policy failure, and blocked rather than fixed
  closure for unsuccessful occurrences.
- Cron is a dependency-free strict five-field UTC evaluator. Interval and
  one-shot slots advance from persisted cursors, with at most one missed slot
  admitted per scheduler pass.
- Rollback is source-level: callers can omit `state_root` to retain the clearly
  labelled legacy ephemeral behavior; removing the durable adapter does not
  require or permit weakening MissionStore schema v1.
## 2026-08-13 coding MCP cross-connection rework ownership

- The stable supervisor now treats a successful, correctly addressed
  `flyto_coding_audit` with explicit `verdict=rework` as the mutating start of
  the next implementation round. It pins only the same well-formed job returned
  in `rework_queued` or `rework_running`, preserving that worker across source
  drift until truthful matching terminal observation or durable terminal state.
- Tenant-visible `get`, accept and error observations, malformed/wrong-job
  responses and response-state inference remain non-owning. Unknown tools have
  no observation authority at all and cannot clear a tracked pin with a
  terminal-shaped receipt; only exact submit/get/audit responses are observed.
  Existing submit ownership, polling, reload-pending behavior, and the exact
  three-tool public inventory are unchanged.
