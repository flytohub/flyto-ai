# Decisions

## 2026-08-24 — Domain capability floor and exact repository authority

- Core-bearing `browser`, `full`, and `dev` extras use `>=2.31.0`, independently
  of the advisory-derived security floor, because 2.31.0 first contains the
  three declared deterministic solver baselines.
- Codex receives `working_dir` unchanged as `--cd` and every other normalized,
  leased request root in order as exec-level `--add-dir`. No common parent is
  derived; initial and rework rounds preserve the same authority set.
- The solver contract stops at software arithmetic, a six-field receipt, and
  Blueprint verified learning/outcome. Open source supports interoperability
  and adoption; equations alone are not defensible differentiation or evidence
  of broad domain completeness.

## 2026-08-24 — Freshness is a separate question from correctness, and needs its own schedule

Decision: the advisory floor is checked twice. `tests/test_stack_security_floor.py`
runs in CI against the locked Core revision and answers "are the declared floors
consistent with the Core this change was tested against". A scheduled workflow
runs the same test against `flyto-core@main` and answers "are they still
consistent with what Core has published since".

Why: reading Core's manifest instead of copying a constant removed one kind of
staleness and left another. CI checks Core out at the pinned revision, so the
gate was green by construction regardless of what Core published — and Core
published five advisories, one critical, against `<= 2.28.1` the day after the
floor was set to 2.28.1.

Why two jobs rather than one: a pull request must not fail because an unrelated
repository published an advisory an hour earlier. That would make the gate noise
and the next person would weaken it. Daily is soon enough for a floor bump and
never blocks a merge.

## 2026-08-23 — A claim is checked where it is consumed, not where it is written

Decision: every cross-boundary claim this package makes gets a check that runs
against the boundary, not against the working tree. Concretely: the Core advisory
floor is derived from Core's own manifest at test time, release drift is measured
against the tag the version names, complexity is measured against a committed
register, and values borrowed from Core are compared to Core.

Why: the audit that prompted this found six defects with one shape. The stack
lock pinned revisions both siblings had moved past. The Blueprint floor made a
signature-probed gate unreachable for every PyPI install. The Core floor predated
33 advisories. `requires-python` named a version the package could not install
on. The watchdog watched a heartbeat nobody published. A local wheel build
carried `node_modules` because CI happened to install them one step later. In
every case the source was correct and the check was pointed at the source, so a
green run was true and useless at the same time.

Consequence: these checks can fail for reasons no code change caused — Core
publishing advisory 34 tightens this repository's floor, and a sibling moving
invalidates the lock. That is the intended behaviour, not a defect in the gates.
The alternative is what was there before, which reported nothing.

## 2026-08-23 — Complexity debt is registered and ratcheted, not thresholded

Decision: module size and parameter count are enforced against
`tests/complexity_baseline.json`. New code meets the budget (800 lines, 8
parameters) immediately; recorded entries may only shrink; the updater refuses to
raise a number without an explicit `--accept-new`.

Why: a plain threshold has no honest setting here. At today's worst it licenses
every future module to be 8,000 lines. At the target the suite is red until a
refactor that cannot be done safely in one pass. The register makes the debt
countable and directional instead of aspirational, and makes the day it is paid
down visible in a diff.

Consequence: `CodingService` stays a 6,973-line class for now, and that is
recorded rather than implied. `coding/errors.py` was the first payment — 453
lines of closed failure vocabulary that never touched service state.

## 2026-08-23 — Layer 1 declares its own product role, and the sibling floors are exact

Decision: this repository ships `flyto-product.toml` declaring `flyto-ai` as
layer 1 `intent_governance`, asserted exactly by `tests/test_product_contract.py`,
and its sibling dependency floors name the exact releases the behaviour needs
(`flyto-core[browser]>=2.28.1`, `flyto-blueprint>=0.3.0`).

Why the contract: `flyto-blueprint` and `flyto-core` both declared theirs and
both listed "intent and provider governance" under `does_not_own`. The
responsibility was therefore disclaimed twice and claimed nowhere. A topology
whose layers are each asserted locally still has a hole if one layer never
speaks.

Why the floors: both were nominal. `>=2.16.1` accepted Core releases predating
all 33 published advisories, and nothing in this package checks a Core version
for security, so an environment holding an old Core satisfied the requirement
and ran. `>=0.1.0` was worse than loose — `tools/blueprint_tools.py` probes the
engine signature for `available_module_ids` and treats its absence as legacy
unfiltered behaviour, and no published Blueprint had that parameter, so the
module-availability gate was inert for every install resolved from PyPI while
reading as enforced in the source. A signature-probed feature is only as real
as the floor that guarantees the signature.

Consequence: `[full]` and `[dev]` now require a Core that carries every advisory
fix and a Blueprint that can be gated. Blueprint 0.3.0 must be published before
this floor resolves; that release is prepared in its repository and tagged
separately.

## 2026-08-22 — Ledger version transition is explicit and fail-closed

Decision: validate parent and successor intent-ledger labels independently
against the historical `task-context.v1` producer label and canonical
`intent-ledger.v1`. Keep instruction context pinned to `task-context.v1` and
retain every existing digest, identity, profile mirror, fingerprint, path,
chain, and generation check.

Reason: first-round plans never cross the amendment verifier, so a producer and
consumer schema-label drift remained latent until an actual audited rework.
Rolling compatibility keeps persisted jobs resumable while the producer moves
to the canonical label; accepting arbitrary versions or skipping the parent
proof would weaken the route and was rejected.

## 2026-08-22: Every formal Robotics planner entry performs governed discovery

Decision: the host adapter exposed by `robotics_planner_server.py` must require
goal-frame normalization and Blueprint/Core discovery before invoking
`RoboticsPlanningService`. Discovery failure is a planning refusal and cannot
fall through to the provider.

Reason: the lower-level planning service deliberately accepts an already
routed request so it remains provider-neutral and independently testable. The
server is the composition root; leaving it wired directly to that service made
the documented Blueprint/Core route optional in the one runnable planner
entry. The caller's executable catalog remains the ceiling, so discovery may
narrow authority but never add motor authority.

Boundary: this changes planning admission only. Cloud still owns Task and
assignment authority, Robotics validates and executes, and neither a provider
plan nor a successful action can declare mission completion.

## 2026-08-20: Codex JSONL has separate per-frame and total-stream bounds

Decision: accept one valid Codex CLI JSONL event up to 2 MiB while retaining
the independent 8 MiB bound for the whole stdout stream. Record only bounded,
content-free protocol counters and sizes in `coding.round` evidence.

Reason: a legitimate tool-result event reached 1,048,577 bytes after JSON
encoding and was rejected by the former 1 MiB frame bound even though the
complete 1,448,355-byte stream was valid, completed, and far below its total
ceiling. A 2 MiB event ceiling accepts that proved protocol shape without
turning the 8 MiB stream budget into one unbounded frame.

Boundary: malformed JSON or event shape, an event over 2 MiB, a stream over
8 MiB, timeout, and missing completion still fail closed. Evidence records
counts, byte sizes, and Booleans only; it stores no event body, prompt, path,
error text, or secret. The route-status source inventory remains self-covered
so an inventory expansion changes the digest observed by an older supervisor.

## 2026-08-20: The service build digest covers every implementer adapter

Decision: the coding service build identity includes both
`agents/claude_code.py` and `agents/codex_cli.py`, in addition to the bounded
coding, provider, configuration, and Core-adapter source set it already
covered.

Reason: the MCP supervisor reloads a worker only when this digest changes. A
Codex adapter repair that was absent from the digest left the reader and live
worker claiming the same build while the worker still held the old imported
module. A source repair therefore merged correctly but could not become active
without restarting the host process.

Boundary: this changes reload detection only. It does not alter job ownership,
safe-boundary draining, backend selection, audit authority, or any failure
gate. Regression coverage names both implementer adapters and proves covered
source bytes change the computed build identity.

## 2026-08-20: A completed Codex protocol turn survives teardown failure

Decision: the Codex CLI adapter accepts the protocol's bounded, valid
`turn.completed` event as completion of the implementation turn. A non-zero
process exit that occurs afterward is recorded in `coding.round` evidence but
does not erase that completion before host snapshots and repository checks run.

Reason: the CLI can finish the model turn, write the complete attributable
change, and emit its terminal event before a later teardown path exits
non-zero. Treating both signals as mandatory made a fully produced change with
all host-owned checks green non-landable while retaining no safe diagnostic.
The terminal event is already the structured protocol authority; the process
exit remains useful evidence rather than a second contradictory verdict.

Boundary: missing completion, invalid or oversized JSONL, timeout, session
mismatch, read-only mutation, failed checks, and no-change rounds still fail
closed. This does not let checks replace Indexer post-validation or Codex audit,
and it grants no commit, push, publish, or deployment authority.

## 2026-08-14 — Amendment contracts prove a bounded delta, not a larger execution budget

Indexer amendments restate the parent execution plan before appending successor
work. Counting that cumulative contract as one fresh executable plan caused a
valid 18-step parent plus 16-step amendment to fail before the provider, although
the parent had already completed. Raising the ordinary execution limit would
have accepted unrelated oversized plans and weakened the pre-provider gate.

Flyto2 AI now independently recomputes Indexer's small versioned parent digest
and validates the root, project, generation, parent profile/ledger/instruction
fingerprints, content-addressed contract and parent ids, complete entry-digest
ancestry, exact original/added/cumulative path partition, successor ledger, and
resolved-target coverage against the normalized host project. Parent and
successor profile mirrors, intent continuity, ledger descriptions and both
instruction contexts are mandatory; missing context never degrades to
`not_required`. Exact target
coordinates assign analysis steps to the original or added side, but original
ownership does not prove prior execution: reuse consumes an exact scoped
tool/args/required/purpose multiset occurrence from the completed parent. Novel
original-path analysis and every successor gate run. Each derived delta remains
under the unchanged ordinary limit. The pinned verifier rejects chain length
eight, making generation seven the effective compatibility ceiling; generation
eight closes until the producer/verifier off-by-one is resolved.

The same boundary preserves actionable Indexer failures only when reason and
action values occur in an exact host-owned registry; unknown uppercase tokens,
prose, paths, secrets and mixed-format values remain generic. Numeric-only
suffixes are excluded from new-file inference so a
milestone such as `M1.1` does not widen authority, without rejecting typed
numeric-leading formats such as `.7z`. We rejected fuzzy semantic step matching,
hard-coded live plan lengths, a larger global step budget and importing sibling
Indexer source into Flyto2 AI. The pure compatibility contract lives in the
stdlib-only `flyto_ai.coding.amendment_contract`; route orchestration maps its
content-free failures to lane errors and owns no duplicate schema implementation.

Path authority is exact rather than producer-declared: the cumulative contract
must equal authenticated parent ledger paths plus exact audited prior
implementation paths plus filesystem-validated explicit targets, in stable
order. Every token must already be canonical repository-relative POSIX form.
This rejects omissions, inventions, globs and traversal without treating a
valid existing numeric-suffix filename as an untyped new-file request.

## 2026-08-14 — A pre-provider rework route has one explicit proved retry

A host-owned Indexer/Blueprint outage after Codex ordered rework did not call
the provider, so terminalizing the job destroyed a valid audited session for an
infrastructure failure. We retain the job as `rework_route_blocked` and expose
one action bit, `retry_rework_route`, through the existing submit tool. It is
not a route bypass: admission requires the original key and normalized request,
exact recorded session, unchanged Git/revision/audit/mission/plan and current
execution authority. A continuation journal may coexist only when it is
claimed by this exact job and its session/generation/origin/contract bindings
match the owner record. Ordinary replay remains observational.

MissionStore and the owner record cannot commit atomically. We retain the
deterministic mission operation receipt until the owner record commits,
validate recalled WorkItem identity and status, and permit one compensation
only for the host's exact peer-deferred `job_not_runnable` closure. A second
publication loss is bounded exhaustion: terminalize action-free, settle any
continuation claim, release workspace/resume authority, and leave both mission
children accounted. We rejected hard-coded live job ids, reopening terminal
provider work, unbounded compensation, and a fourth MCP tool. Rollback removes
the request action and blocked-state transition; exhausted records remain
conservative terminal failures.

## 2026-08-14 — Orphan abandonment uses target proof, not global downtime

An unrelated live coding service holding the state-root authority lease does
not prove the target orphan is running. Requiring an exclusive authority lease
for every abandonment therefore deadlocked recovery: a kernel-closed queued
record pinned the supervisor, while stopping all supervisors was the only way
to retire it.

`open_host_abandon_valve` now takes the authority lease shared and may perform
only `abandon`. The write is authorized by four explicit facts: the global
state guard serializes the transition, the exact target job lease is free, the
record is audit-ready or queued/rework-queued, and a queued item is already
closed blocked/deferred in MissionStore. The marker is neither read nor
written. Claim repair has no equivalent exact-target proof, so
`open_host_release_valve` remains exclusive and the online valve refuses it.

We rejected removing the authority lease, weakening queued-state checks, and
making claim repair concurrent. The shared lease prevents authority rotation,
the job lease prevents a live writer race, and the MissionStore disposition
prevents ready work from being discarded. Rollback is the single CLI factory
selection; the exclusive release valve and its tests remain unchanged.

## 2026-08-14 — Every project-aware Indexer plan step inherits host scope

An Indexer task contract may omit `project` from individual read-only plan
steps because the task itself already names the project. The capability process
can see ambient indexes, so the host must carry its scope into canonical and
translated `search`, `impact`, `structure`, and `call_hierarchy` calls. If plan
evidence names a different project, the step is rejected; evidence cannot
change the workspace authority selected by the host.

This is a narrowing, not new authority. It changes no allowlist, tool schema,
lane, gate, timeout, receipt field, or mutation permission. The rejected
alternative was to accept or silently rewrite foreign project evidence, which
would hide a contract violation at the repository boundary.

## 2026-08-13 — Capability search revalidates canonical claims under host verification

- A phase-one Card retains its bounded canonical claim solely as validation
  material. Search projection recomputes the exact claim digest, rebuilds the
  Card under frozen authority, and rejects any field mismatch; no independently
  recomputable Card digest is treated as authority.
- `host_verified` is an explicit frozen-authority fact distinct from capability
  approval and verification and must be exactly true. Completeness additionally
  requires a bounded non-blank source reference. Identifiers never synthesize
  display or semantic content.
- The projection excludes the canonical claim and source reference as well as
  parameters, defaults, payloads, prompts, headers, credentials, tokens,
  secrets, MCP arguments, endpoints, and raw bodies. This remains a contract
  only, with no downstream catalog, retrieval, routing, runtime, or UI.
- Public claim and Card boundaries snapshot an untrusted Mapping once through
  bounded items and thereafter read only the detached exact dict. Duplicate or
  inconsistent entries and arbitrary iterator/getter faults fail with a fixed
  content-free `CapabilityCatalogError`, preventing caller exception leakage.


## 2026-08-12: Codex CLI is an explicit implementation backend, not a route bypass

Claude quota must not decide whether the Flyto2 route exists, but replacing the
route with ad-hoc direct edits would also discard the very evidence the route
was built to protect. We therefore add `codex` as a third startup-selected
implementer behind the existing `CodingService`; it does not become a per-job
fallback and it cannot skip Indexer, Blueprint, Core, checks, or audit.

The implementation process is deliberately not the auditor process. Flyto2
starts a separate non-interactive Codex CLI thread, pins the executable and
model at service startup, binds the structured thread id durably, and resumes
that exact id for typed audit rework. The outer Codex session receives only the
host-derived revision and evidence and remains the sole principal that can
submit an accept/rework verdict. Sharing a model family or ChatGPT account does
not merge those roles or their session state.

The child gets less ambient authority than an ordinary personal Codex run. It
uses `--ignore-user-config` and `--ignore-rules`, has no configured MCP server,
plugin, web search, browser, computer-control, or audit tool, and inherits only
the small runtime environment needed to launch and authenticate the CLI.
Provider and CI credentials are excluded. Its command execution remains inside
Codex's bounded `read-only` / `workspace-write` sandbox, never danger-full.
Flyto2 independently snapshots the workspace before and after, runs the pinned
repository checks, and refuses a changed/missing session, unexpected read-only
write, invalid JSONL, missing verification tool, or required capability.

Selection stays startup-only: `--implementation-backend codex` requires an
explicit bounded `--model`; `--codex-command` may pin the installed executable.
There is no automatic fallback from Claude/native, no remote backend field, and
no audit-disable flag. Rollback is configuration: restart the service with
`native` or `claude`. Existing jobs stay bound to their recorded implementation
backend and session; they are never silently migrated.

## 2026-08-12: The coding watchdog observes; it never acts, and it is not an AI

`code-status` and `code-task-window` make failure inspectable, but a dead or
wedged Codex cannot inspect itself. The observer for that gap is deliberately
the dumbest component in the system.

It is not an AI. A model in the liveness path is a model that can be wrong
about whether the system is alive, can burn Claude/Codex/Copilot/Gemini quota
on healthy polling, and can fail for the same reason the thing it watches
failed. `code-watchdog` therefore evaluates fixed thresholds over the two
bounded projections that already exist and emits stable reason codes. The
GitHub side is an ordinary scheduled workflow, not a GitHub Agentic Workflow,
for the same reason. AI diagnosis may be attached *after* an incident is
opened; it is never what decides that one exists.

It has no authority. The first release is alert-only: no automatic job
abandonment, workspace repair, service restart, audit acceptance, commit, or
push. An observer that can act is a second control plane, and a second control
plane racing the first is how a stuck job becomes a lost one. Operators use the
existing explicit, subtractive recovery commands after reading the reason
codes.

It records aggregates, not identities. Health files carry counts, reason codes,
the reader build digest, and timestamps. Job ids, session ids, repository
paths, prompts, evidence, and credentials never reach `~/.flyto/health/coding/`
and never reach the remote heartbeat, because the heartbeat lands in a *public*
GitHub Actions repository variable. The `gh` CLI supplies its own
already-authenticated credential, so no token is written into the LaunchAgent
plist or any health file.

Liveness is layered, and the outer layer is off-host. The LaunchAgent watches
the coding service; the scheduled GitHub workflow watches the LaunchAgent. Only
the remote witness can observe the case where the local machine dies, which is
exactly the case a local watchdog cannot report. Consequently the local path
must never fail in a way that skips recording: a hung or missing `gh` is
converted to a stable code rather than raised, so one bad heartbeat cannot also
cost the local health record.

An idle host is healthy. Requiring an always-on model process would waste quota
and quietly turn the watchdog into the thing that keeps a Codex alive. Absence
of work is only a fault when work is stranded.

## 2026-08-12: The watchdog's own inputs are untrusted, and its bounds belong to their writers

A monitor is only worth its uptime if it cannot be talked out of alerting. Three
rules follow, and they apply to the watchdog itself rather than to what it
watches.

The public heartbeat is untrusted input. `FLYTO_CODING_HEARTBEAT` is an Actions
repository variable, so anyone able to write repository variables can choose
what the dead-man switch reads. The workflow therefore bounds the raw size
before parsing, requires the exact schema, requires a plain in-range integer
timestamp, allowlists `health` to the three levels the publisher can emit, and
allowlists reason codes. The emitted `reason` is re-checked against a
single-line character allowlist immediately before it reaches `GITHUB_OUTPUT`;
that check is redundant by construction and kept anyway, because the failure it
prevents is a newline forging `healthy=true` and silencing the exact alarm the
workflow exists to raise. A malformed heartbeat is never optimistically read as
healthy, and each rejection carries its own code so an operator can distinguish
a bad publisher from a dead host.

A bound belongs to whoever writes the file. `state_readable` now judges the
route status index by the publisher's `MAX_STATUS_INDEX_BYTES`. A second,
stricter copy of someone else's limit does not add safety; it manufactures
incidents whenever the writer legitimately uses its own headroom, and a monitor
that cries wolf is uninstalled.

An installed configuration must be a runnable configuration. Every value baked
into the LaunchAgent is validated at install time against exactly the bounds
the observing run applies, through one shared validator rather than a second
copy. The worst outcome for a dead-man switch is not a loud rejection at
install; it is a successful-looking install that fails silently on every
unattended wake. The same reasoning forbids the state root and health directory
from overlapping: an observer writing into the tree it observes both mutates
durable coding-service state it has no authority over and triggers on itself.

A path is identified as a directory, never as a spelling. Both roots are
resolved through their symlinks before the overlap comparison, and the resolved
state root is what derives the LaunchAgent label. A lexical `abspath` is not a
containment check — `--health-dir <link-into-state-root>` presents the guard
with two unrelated strings — and it is not an identity either: install
resolving while uninstall does not yields two labels for one state root, so
`--uninstall` reports success, removes nothing, and leaves the agent waking
forever. Resolution is non-strict so a health directory that does not exist yet
is judged by exactly the rules every later run is judged by.

The health directory is not assumed to be exclusively owned. It is created
`0o700`, but `--health-dir` takes an operator-supplied path and a world-writable
parent such as `/tmp` is a plausible choice, so every record the watchdog opens
by name is opened `O_NOFOLLOW` and rotation tests the name with `lexists` rather
than its target. `_atomic_write` needs no flag because `os.replace` overwrites a
link instead of following it; the append, the reads and the lock did need one. A
name-then-read pair is not sufficient either — checking `is_symlink()` and then
reading by name leaves a window in which the record is swapped for a link — so
`_read_json` measures and drains one descriptor.

The turn's contract is the local record, and nothing secondary may cost it. A
hung `gh` was the first way that promise broke; a failed heartbeat-cursor write
was the second, and it is worse, because the heartbeat had already been
published and the remote switch would read `healthy` while the record a human
inspects was never written. Both are now warnings carried inside the record
rather than exceptions that end the turn. The same rule chose the workflow's
`cancel-in-progress: false`: the job's product is an incident, and a run
cancelled between deciding and reporting is silence at the one moment the switch
exists for.

## 2026-08-12: Repository-set leases, one task window, and one stack lock

A configured workspace root is an admission boundary, not a concurrency unit.
Treating `/.../flytohub` as the one leased tree forced unrelated jobs in
`flyto-code` and `flyto-engine` to serialize even though their files do not
overlap. Admission now derives the nearest real Git boundary from
`working_dir`. A genuine cross-repository job may name up to sixteen real,
non-overlapping Git roots; the host-global registry acquires that complete set
under one registry transaction and releases any newly acquired descriptors if
one member refuses. The exact canonical set and its digests are persisted in
the private job record and reacquired from that record after restart. Legacy
records fall back to `working_dir`, while their inactive historical ancestor
entry remains conservative only for legacy work; it cannot serialize unrelated
new child-repository records forever.

Coordination is visible through host-only `flyto-ai code-task-window`. It joins
the generic MissionStore main-axis/branch/order projection with coding job
state, repo digests, implementation-session presence and audit/rework counters.
An optional bounded `owner_ref` labels the submitting Codex/task but grants no
authority. The window carries no prompt, path, evidence, worker identity or
provider session id, starts no service, and is not a fourth MCP tool. Other
Codex tasks can therefore see that work exists without receiving another
task's conversational context.

Finally, `stack-lock.json` is the single dependency revision source for
Blueprint, Core and Indexer. GitHub checkout refs are read from it and the
repository verification contract checks local sibling HEADs against it. A
green local audit and a green CI run can no longer mean different dependency
commits because three SHAs were copied into two places.

We did not add a task taxonomy, automatic Codex implementation fallback,
cross-host lease, or model-visible fleet tool. Multi-host remains a database
lease problem; the local repo-set protocol remains `flock`-based.

## 2026-08-12: Installing Core extensions is host authority, not model authority

An agent that can install software can grant itself capability. So the Core
extension adapter is a host API and never an MCP tool, and the rule is
structural rather than a list of known names: any Core tool whose name is an
install, uninstall, or reinstall verb is withheld from `get_core_tool_defs`
*and* refused by `dispatch_core_tool`. The catalog filter alone would not be a
boundary, because a model or a forwarding client can type a name it never saw.
The word boundary is explicit so read-only reporting (`list_installed_modules`)
stays callable.

Uninstall shares the `FLYTO_EXTENSIONS_INSTALL_ENABLED` opt-in with install:
both change what the installed Core can execute, and a host that may not add
capability may not silently remove it. The gate is checked before the request
is validated, so a disabled host has exactly one observable behaviour.

The envelope is fixed and deliberately poorer than Core's answer. Installer
output is attacker-influenced, unbounded, and frequently carries internal index
URLs and paths; it is not something to render in a cloud UI or hand back to a
model. Only bounded tokens leave: Core's normalized name, Core's own `code`,
and a version. Exception text is dropped even from the host log, where only the
exception type is recorded.

The bridge binds `core.plugin.loader` — the module Core actually owns this
surface in — and not an invented `core.extensions`. Core's shapes are taken as
given: `list_extensions` answers with a plain list and has no kind parameter,
`EXTENSION_KINDS` records carry `kind` / `prefix` / `entry_point_group`,
`install_extension` takes `(name, version, upgrade)` and `uninstall_extension`
takes `(name)`, and `normalize_extension_name` decides identity. The host's
operation names and Core's method names are different words, so the mapping is
an explicit constant; deriving the method from the operation is what made an
earlier revision call an `install` method that does not exist.

Because Core cannot filter a listing, the host does — after normalization, so
the filter matches the bounded token it publishes rather than a raw value.

Where the host must name a field of `ExtensionResult` it uses exactly one name
with no alias fallback, because a look-alike fallback lets a Core rename pass
silently as "field absent". The published set is Core's own:
`previous_version`, `restart_required`, `rolled_back`, `refresh_failed`. We do
not publish an `install_enabled` sourced from Core, because Core does not
report one — that field is the host's opt-in state and says so. A real-contract
test binds the installed `ExtensionResult` and fails, listing Core's actual
declared fields, when a published name is not among them; that test is what
caught two rounds of invented contract in this work.

We kept the adapter generic over whatever kinds Core declares rather than
enumerating today's extension families, and we kept Core as the authority on
name normalization — a host-side rename would make an operator's installed set
unaddressable in Core's own terms. We did not add a bypass flag, did not import
sibling `flyto-core` source, and did not let a malformed Core answer degrade
into a partial list: list and kinds fail closed to `invalid_core_result`.

## 2026-08-12: Workspace authority follows work, not an idle MCP process

The first host-global broker held every configured workspace root for the
entire `CodingService` process lifetime. Codex keeps an MCP worker alive for a
task lifetime, so a custom state root remained the live owner of all of
`flytohub` after its only job was terminal. Other Codex tasks were refused
indefinitely; restarting that idle worker simply reacquired the same hold.

Ownership now starts immediately before the first durable mutation that admits
non-terminal work and survives queued, running, rework, and audit-pending
states. Startup with open durable work reacquires before reconciliation. Once
the state root has no non-terminal job or surviving workspace claim, ownership
is released. A small bounded observer is required because the shared queue
allows worker A to admit and worker B to settle the job; relying only on B's
terminal callback would leave A's process-owned lease held forever. The
observer takes the same cross-process state guard as admission, so it cannot
release between authority acquisition and the first job record.

We kept one shared state root as the normal multi-Codex queue and kept the
configured-root boundary for different state roots. We did not introduce
per-thread state roots, a native/Codex fallback, another public MCP tool, or a
distributed lock. Cross-host operation remains future work and requires a
database lease rather than `flock` or NFS.

## 2026-08-11: Rework planning carries the scope the host already proved

A final strict validation reported `unplanned_diff` after every required check
passed. The amended contract did include the file named by the last audit
finding, but it did not include several files the same job had attributed in
earlier rounds. Audit prose is intentionally narrow and cannot be the sole
source of cumulative plan authority.

Before a rework starts, the service already proves the prior file tuple against
the bound session, resume envelope, worktree claim, workspace and exact
revision digest. That tuple now travels to the Indexer pre-lane, which unions it
with explicit targets from the new finding before requesting the same-root
amendment. The existing finite target bound applies to the union and refuses
instead of truncating. A first-round plan has no prior tuple and its payload is
unchanged. Post-work, persistence and exact-revision audit continue binding the
same cumulative tuple.

## 2026-08-11: The Core manifest contract is Core's, and Blueprint gets module ids only

Two decisions, recorded together because one caused the other.

**The wire shape is read from `core.capability_manifest`, never from a
fixture.** A prior round reconciled the host validator toward the test
fixtures, which had invented `manifest_contract` / `manifest_hash`,
record-shaped `modules`, `capability_id`, and `plugin`. Fixtures and validator
then agreed with each other and disagreed with Core, so every real manifest was
rejected and a fully installed Core reported zero modules with every test
green. The validator now reads `schema` / `hash`, string `modules`,
`capability` + `providers`, and `id` + `version` + `module_count`. A fixture
that disagrees is the thing that is wrong.

**Only module ids cross into Blueprint, under `available_module_ids`.**
`get_core_installed_capabilities` unioned module ids with capability ids so a
step named by capability would still match. That is backwards: a capability id
names what a module provides and a plugin id names who ships it, and neither
can be executed. Handing either to an engine lets it offer a step nothing
installed can run. Both are still validated and counted as provenance. The
function is removed rather than deprecated, because its only caller was the
bridge that must not use it.

Fail-closed direction is unchanged: `None` only for an absent or too-old Core,
an empty frozenset for every failure of a manifest-capable Core.

## 2026-08-11: Instance liveness is a crash-released lease, never a pid

`code-status` reported a historical, already-`closed` instance as alive because
the recorded pid had been reused by an unrelated process (`cloudphotod`). The
probe was `os.kill(pid, 0)`, which answers "does some process hold this pid",
not "is that process the instance that recorded it". Those differ precisely
when it matters — after a crash, which is when the status is read.

Each `RouteStatusPublisher` now holds `LOCK_EX` on its own
`status/instance-<id>.lease` for the life of the process. The kernel releases
that lock however the process dies, including `SIGKILL`, so an uncontended
lease is positive proof the instance is gone rather than an inference. A reader
decides in this order: a `closed` lifecycle is never alive; a held lease is
alive; otherwise an uncontended lease is not alive; otherwise the answer is
`None`. A pid probe may now only *lower* an answer to `False` and can never
raise one to `True`, which is exactly the reuse bug removed.

Honest limitation: `flock` is advisory and per-host. On NFS it may be emulated
via `fcntl` byte locks or silently degrade, and a state root shared across
hosts is outside what this proves. Where `flock` is unavailable the answer is
`None` — undecidable — and never `True`. `code-status` therefore reports
`unknown` rather than inventing liveness, and the host release valve continues
to refuse outright without `flock` rather than acting on an unprovable claim.

## 2026-08-11: A state-root authority refusal survives the supervisor

`code-mcp-supervisor` collapsed every worker fault into
`-32603 coding worker unavailable`. During the rotation incident that hid the
one actionable condition — the state root refusing this build's authority —
behind the same generic string as a broken pipe, so the client could not tell
an operator what to do.

`code-mcp` now exits `78` (`EX_CONFIG`) when construction fails with a
`CodingServiceError`, printing only the stable `exc.code`; the exception
message can carry a state-root path and is deliberately not printed. The
supervisor keeps the reaped worker's exit status — an integer from a closed
set — and substitutes one fixed sentence naming `code-status` and
`code-release`. Worker stderr is still never captured or forwarded, so no path,
prompt, secret, raw error, or job content can travel this channel. Every other
fault keeps the generic reason, and the public MCP inventory stays exactly
`flyto_coding_submit`, `flyto_coding_get`, `flyto_coding_audit`.

## 2026-08-11: The audited Claude route has a finite larger SDK frame ceiling

The Agent SDK defaults one JSON message to 1 MiB. That is smaller than a
legitimate single response from the strict route's host-declared Indexer MCP
server and caused a live attributable implementation session to fail before
revision sealing. Service-mode `ClaudeAgentOptions` therefore sets a fixed
8 MiB `max_buffer_size`.

This is a transport framing bound, not new execution authority. It applies only
to the audited service adapter, remains finite, and does not change allowed
tools, MCP selection, workspace scope, evidence, request, turn, cost, or route
budgets. Legacy direct calls intentionally retain the SDK default so this
production incident does not silently alter their compatibility contract.

## 2026-08-10: A required-change service turn cannot end in prose

`require_changes` is a host-owned job invariant, not merely a final snapshot
check. The Claude compatibility agent previously returned success immediately
when no browser verification recipe was attached, even when the provider used
no repository tool. The service then ran every required subprocess against a
known-empty diff and eventually failed as `no_changes` or, under strict route
projection, `cumulative_scope_unbounded`.

Service mode now treats absence of attributable mutation evidence as bounded
rework inside the exact Claude SDK session. It sends one fixed host-authored
instruction to inspect and edit the requested files, preserving session,
attempt, turn, cost, workspace, route, and startup authority. `Edit` and
`Write` post-hook evidence satisfy the local boundary; a named project action
also qualifies because the outer service independently proves its resulting
snapshot. Final authority still comes only from the host snapshot, required
checks, strict route lanes, exact revision digest, and Codex audit.

We rejected silently resubmitting a new provider conversation: it would lose
the session identity an audit must attribute. We also rejected making this the
default for legacy direct callers or explicit `require_changes=false` jobs,
because inspection-only work legitimately has no mutation.

## 2026-08-10: The release valve opens a state root without binding it

`flyto-ai code-release` built an ordinary `CodingService`, which meant it also
built an ordinary *startup authority*. A host retiring an orphaned job is by
definition not running the strict route that stranded it, so the recorded
authority never matched: `_bind_startup_authority` saw live work under another
authority and rotation demanded every job be terminal — including the one open
`awaiting_codex_audit` job the command existed to retire. The only operation
that could release the root was refused by the state the release was for.

`CodingService.open_host_release_valve` is now a distinct construction mode.

- It takes the state-root authority lease **exclusively** and refuses with
  `service_busy` if any live coding service holds it. Exclusivity proves
  something stronger than agreement — that no service of any authority is alive
  here — so no authority has to be compared, and none is invented.
- It never reads, writes, rotates, or reproduces `authority.json`. The marker
  comes out of the operation byte-for-byte as it went in and the strict route
  still owns the root afterwards.
- `_bind_startup_authority` and `_require_all_jobs_terminal` do not run, so a
  second open job under the same recorded authority is left exactly as found
  and one requested `awaiting_codex_audit` job may still be abandoned.
- It publishes no runtime status, reconciles no interrupted job, and refuses
  `submit`, `audit`, and `_pump_dispatch` with `release_valve_refused`. Its
  agent factory raises, so no implementer can be constructed at all.
- `abandon` and `repair_workspace_claim` are unchanged. The abandon path still
  takes the job lease, still moves only `awaiting_codex_audit` to
  `failed`/`job_abandoned` with `landable: false`, and still releases the
  workspace claim, the resume envelope, and the continuation authority.

We rejected relaxing `_require_all_jobs_terminal` for the audit-ready state:
that would have weakened rotation for every service, not just the valve, and
rotation is what keeps two route semantics off one root. We also rejected
having the valve reproduce the strict authority so it could match the marker —
a host that can synthesize another route's authority can also adopt its work.

The MCP inventory is unchanged at exactly three tools; the valve remains a
local operator command and is not reachable by a model.

## 2026-08-10: Mission authority sets the Python floor at 3.11

The public package and coding service require Python 3.11 or newer. The mission
store deliberately binds its in-memory SQLite authority database into a
pathname-free byte envelope with `sqlite3.Connection.serialize()` and
`deserialize()`. Those primitives are part of the supported CPython surface
from 3.11 onward but are absent on 3.10.

Claiming 3.10 support made a clean runner fail only after entering the mission
control plane. The package metadata and CI matrix now state the real boundary,
CI exercises both Python 3.11 and 3.12, and the development extra installs the
Claude SDK required by the complete route suite. We do not emulate the missing
primitive with temporary files, monkeypatches, or a weaker continuation path;
unsupported hosts remain outside the contract and fail closed.

## 2026-08-10: One semantic startup authority owns an active coding state root

A coding state root is bound by a durable authority lease, not by inference from
its job records. Every compatible live service holds a shared `flock` on
`<state_root>/.authority.lock`; a newcomer must win the exclusive lock before it
may write the bounded, secret-free marker in `<state_root>/authority.json`.
Rotation requires both that no old service is alive and that every job is
terminal. An incompatible service fails construction before status
reconciliation, the workspace-claim sweep, or any pump.

Reason: scanning job records could not establish this. An empty root has no
records, so two incompatible services both constructed, and whichever admitted
work first left the other one running and able to submit and pump against an
authority it did not share. Only a live durable holder makes the claim true.
Refusal at construction also converts "an incompatible worker does not burn
dispatch attempts" from a per-pump budget into an invariant: a service that
never starts is never offered an item.

Liveness is `flock`, never a TTL. A crashed service releases its share when the
kernel closes its descriptor; a paused one is never declared dead.

The fingerprint is a recursive canonical digest of the whole validated startup
policy, including nested Indexer, Blueprint, Core and `RouteLimits` semantics
and every configured string exactly as written. Capability argv and executable
paths are hashed rather than normalized: a state root and its `flock` are
host-local, so two services sharing one are on the same machine, and folding
`/opt/indexer-v1` and `/opt/indexer-v2` together would let two lanes that run
different binaries share a root. Only the digest is persisted, so hashing exact
strings publishes nothing. The fingerprint
deliberately excludes `build_id` - a hot reload changes that without
changing what would execute, and binding to it would strand a queued job a
semantically identical worker could run. Build identity is still enforced at
admission.

Every check runs before any write. Marker validation, active-job validation and
pre-fingerprint settlement all happen under the state guard while the caller
holds the exclusive lock, and the marker is written last. A refused start-up
leaves a present marker byte-identical and never creates a missing one; writing
first let a stranger replace a lost marker, fail on an open job, and lock out
the correct worker. Malformed, unparseable, non-regular and symlinked markers
are refusals rather than absences, and unreadable or state-less job records
refuse both start-up and rotation - neither is evidence that a job finished.

Records predating the fingerprint are adopted only on proof. Queued work with
`implementer_started` false is migrated and runs normally: that flag is written
before the provider call, so it is real evidence no execution began. An empty
`implementation_backend` is *not* such evidence - it is recorded on outcome - so
an executing v0 record is never adopted; if its job lease is held the service
refuses to start beside a round nobody can attribute, and once the lease is
provably free the job is terminalized as `execution_authority_unbound` with its
mission item and worktree claim accounted. Only an unfingerprinted
*awaiting-audit* job is accept-but-not-rework, because a verdict describes a
revision the host already hashed while a new round would adopt an unproven route
policy.

Where the host has no inter-process lock, construction fails with
`CodingAuthorityUnavailable` (`execution_authority_unavailable`) rather than
degrading to a no-op. A service that cannot tell whether another one is alive
cannot claim this isolation, and claiming it falsely is worse than declining the
host.

Operator semantics: `CodingAuthorityConflict` (`execution_authority_conflict`)
and `CodingAuthorityUnavailable` are raised at construction and are not
retryable. Resolve a conflict by stopping the services of the other authority
and closing their open jobs, or by starting with that authority's own
configuration; repair or remove a damaged marker or job record deliberately,
because the service will not overwrite either. Rollback is to revert this
change; the marker and lock file are inert to older builds, which ignore both.

## 2026-08-10: `require_changes` is cumulative across audited rework

`require_changes=true` requires one real attributable job revision; it does not
require every same-session rework round to produce different bytes. A clean
rework may promote the adapter's exact `no_changes` result only when all
required checks pass and the host re-proves the recorded implementation
session, tenant/job worktree claim, sealed resume envelope, cumulative file
set, and current content digest. The cumulative files are then passed through
the ordinary Indexer post validation and exact-revision Codex audit.

Reason: an auditor may correctly request a bounded completion or recheck after
the implementation is already byte-correct. Treating that recheck as a fresh
job discarded a verified revision or forced the model to create meaningless
source churn. Reusing cumulative proof preserves the audit boundary without
inventing a diff or trusting provider prose.

## 2026-08-09: Repository dotfiles need exact basename authority

Guardian may edit only the three repository dotfiles already named in its
closed allowlist: `.gitignore`, `.dockerignore`, and `.editorconfig`.
`os.path.splitext()` treats a leading-dot filename as extensionless, which made
those existing entries ineffective and prevented required generated-index
hygiene from being implemented through the audited route. Matching exact
basenames repairs that contradiction without allowing `.env`, `.bashrc`, or
any other arbitrary dotfile; the sensitive-path gate still runs first.

## 2026-08-09: Claude implementation rounds use the existing 100-turn ceiling

The audited Claude adapter now defaults to 100 turns, which was already the
validated hard ceiling in `CC_MAX_TURNS_CEILING` and in the public coding
request contract. Two fresh, fail-closed robotics verifier jobs exhausted 30
and 60 turns after producing substantial workspace changes, leaving no provider
result or auditable revision. Raising the default to the existing ceiling lets
the provider finish its bounded response; it does not loosen the USD budget,
grant Bash, broaden edit authority, bypass required checks, skip the four host
lanes, or weaken exact-revision Codex audit. Operators may still set a lower
positive value with `FLYTO_AI_CC_MAX_TURNS`.

## 2026-08-09: The capability control plane is domain-neutral; coding is an adapter

The downstream chain is a statement of responsibility and data flow, not a
mandatory synchronous path every task walks. Domains named in requirements —
software development, penetration testing, red-team exercises, robotics,
workflows, ordinary tasks — are **example inputs only**. Encoding them as an
enum, switch, component map, provider rule, or fixed list would make the next
unlisted domain a code change, which is precisely what this layer exists to
avoid.

- Profiles, capabilities, tools, and contract versions are arbitrary bounded
  identifiers validated by grammar, with explicit permissions and contracts.
- The regression that protects this generates its identifiers from a digest
  rather than choosing them, so passing cannot depend on a blessed name. A
  source-text grep test was rejected: it is brittle and would itself become the
  sanctioned list it is meant to prevent.
- The generic negotiation path injects no default component, coding phase, or
  provider, so no domain is forced through a repository-shaped flow.
- Indexer is not assumed to run for every non-code task.

`flyto_coding` and its three MCP tools remain one Codex-facing adapter over this
layer. The audit-required route, durable workspace claim, and same-session
rework stay inside it, because they answer a repository-specific question about
exclusive worktree ownership across an audit gap. The package is deliberately
*not* renamed: public compatibility is worth more than removing the word
"coding" from a module path. This is explicitly not a universal distributed
scheduler, and that limitation is recorded rather than papered over.

## 2026-08-09: Job-lifetime worktree ownership and session-bound cross-worker rework

The owner runs many Codex conversations at once, each starting its own
`code-mcp` worker against one shared state root. Two failures followed from
that topology and are closed here.

- **Ownership must outlive a round, not match one.** The workspace lock was
  released when an implementation round ended, but the job stayed at
  `awaiting_codex_audit` until a human verdict. That interval is exactly when a
  competing frontend could edit the same tree, after which the first job's
  exact-revision audit failed live recomputation and its work was stranded. A
  durable claim now spans the whole job. It is keyed by workspace digest, so
  different repositories keep running in parallel — cross-repo parallelism was a
  requirement, not a side effect.
- **A distributed design was chosen over a shared broker.** A broker would have
  reintroduced the single-owner failure rolled back in
  `handoffs/2026-08-09-multi-process-coding-state.md` ("coding state root is
  already served"), added a daemon with its own crash/auth/reload story, and
  become one wedge point for every repository. It also buys nothing for
  same-session rework: the Claude session lives in the Agent SDK's own on-disk
  store, not in the Python process, so any live worker can resume it given the
  session id. Only the original request was missing.
- **The claim file is an index; the job record is the authority.** Liveness is
  derived from the owning record's state rather than a TTL or heartbeat, which
  removes clock guesswork and makes crash recovery a consequence of reading the
  record rather than a separate mechanism.
- **Unevaluable ownership fails closed and is never auto-cleared.** A corrupt,
  unknown-shape, unreadable, or orphaned claim resolves to `unresolved`, not
  `free`. Deleting it would convert "ownership cannot be evaluated" into
  "nobody owns this tree" — the precise hazard the claim prevents — and startup
  is when a half-written state root is most likely. Only the host operator
  clears one. This is deliberately a distinct code from `workspace_busy`: busy
  names a live owner and resolves itself, unresolved never will.
- **Only audited jobs take a claim.** The claim exists to protect the audit
  gap, and a legacy direct-library service has none. It therefore keeps its
  per-round serialization rather than gaining a new fail-fast rejection, but it
  still honours a claim another job holds, so it can never edit a tree
  mid-audit.
- **The resume envelope may continue a session but never start one.** It
  persists only the public request fields plus job, request-digest, and session
  bindings, loads only when `session_bound` equals the record's
  `implementation_session_id`, and always rebuilds with `resume=true` against
  that id. The stored digest is compared rather than recomputed, because
  redaction rewrites credential-shaped prose and a recomputed hash would never
  match — that would have silently disabled rework instead of failing loudly.
  Startup authority is never persisted and is re-imposed from the running
  process, so a stored request cannot outlive or widen its policy. This
  preserves the original intent of the process-local cache ("a restart must not
  silently start a new session") while removing its process affinity.
- **A missed supervisor deadline terminates rather than retries.** Thirty
  seconds is the bound because submit, get, and audit only schedule or inspect
  background work; a longer wait is a wedged worker, not a slow one, and that
  worker still holds shared state-root locks. The request is never resent: its
  delivery is uncertain and the job may already exist, so recovery belongs to
  the caller replaying an idempotency key. A reader thread and queue were chosen
  over pipe selectors so the deadline behaves identically on every supported
  platform.
- **Active-job tracking reads durable records.** A process-local set could not
  distinguish "the client stopped polling" from "the job is still running", so
  one abandoned entry pinned `service_reload_pending` for the life of a
  frontend. Reconciling each tracked id against its own bounded record fixes
  that without a status index, which is a latest-writer view and cannot answer
  for several concurrent jobs.
- **The release valve is a CLI command, not a fourth MCP tool.** Adding a tool
  would widen the audited public surface and put job retirement within reach of
  a model. `code-release` is strictly subtractive — it can only move
  `awaiting_codex_audit` to `failed`, never accept or land — so it is always
  worse for a caller than auditing and cannot become an audit bypass.

## 2026-08-10: Blueprint reuse preserves direction-bearing phrase order

The Blueprint catalogue may return several candidates with the same token set.
Choosing the first candidate made a request for CSV-to-JSON reuse project a
JSON-to-CSV label when both shared `convert`, `csv`, and `json`.

- The read-only lane still considers at most 20 candidates and still requires
  at least two overlapping normalized tokens.
- Among eligible candidates it ranks overlapping adjacent token pairs first,
  then total token overlap. Catalogue order remains the deterministic final
  tie-break.
- Only the existing inert name-and-digest projection crosses to the
  implementer. No Blueprint steps, prose, or execution authority are exposed.

This is host-owned matching logic, not learned catalogue trust. Rollback is the
bounded score selection and its reverse-transform regression test together.

## 2026-08-10: The execution plan selects one published Indexer gate family

Indexer has published two pre-work gate vocabularies: legacy `assess` /
`implement`, and current `plan_changes` / `apply_changes`. A route fixed to the
legacy pair refused the current server's real plan with
`plan_gate_phase_unknown` before the implementer could start.

The host now derives one complete family from the exact execution plan before
dispatching its first plan step. A plan without gates keeps the legacy default
for backwards compatibility. An unknown phase, a duplicate phase within one
plan scope, or any mix of the two families fails closed. The host never probes
by sending extra gates and never composes a third vocabulary.

The same compatibility boundary covers the published post-validation result.
A present legacy `pass` or `passed` field stays authoritative. When neither is
present, current `overall=pass` evidence is accepted only if both the ruff and
pytest status blocks explicitly say `pass` or `skipped`. A headline without
its component evidence, or any contradictory status, remains failure.

## 2026-08-10: One bounded Indexer timeout for every coding-route entry

Large Flyto2 workspaces can legitimately need more than 30 seconds for the
mandatory post-work `verify.strict`, especially when the Indexer rebuilds its
generated state. The detachable stack preset and the public `code-mcp` /
`code-serve` CLI constructed separate `CapabilitySpec` values, so changing one
left the active public route on the old deadline.

- One named constant now owns the Indexer transport bound, and both route
  constructors use it.
- The bound is ten minutes, within the existing 900-second contract maximum.
  It changes only how long the host waits for one allowlisted Indexer call;
  mandatory lanes, tool permissions, call-count and remediation limits,
  evidence validation, and fail-closed semantics stay the same.
- A real overrun is still classified `capability_timeout` and remains
  non-landable. Green repository checks still cannot replace Indexer post-work
  evidence.

Rollback is one constant change with the same two-constructor regression test.
Do not introduce a per-job override: timeout authority remains startup-owned.

## 2026-08-09: Project-scoped route searches, exact failure evidence, and a startup-only emergency overflow lane

Against the real installed Indexer, the production route policy failed every
ordinary task: `indexer_pre.search` timed out at 30.002 s because the host sent
only `{query}` while the Indexer's smart search fanned out across every indexed
project. The persisted result collapsed this to `route_domain_failure` with an
empty call list, so no reader could tell where the round stopped or that the
implementer had never started.

- Every host-owned Indexer search carries the workspace project. The same query
  with `project` completed in about a second. The 30-second capability bound is
  deliberately unchanged: the query was over-broad, not the deadline too short.
- A lane keeps a bounded call trace, so a failed lane receipt retains its
  completed calls plus one failed call naming the exact host-derived semantic
  action. Transport exhaustion is classified `capability_timeout` from a closed
  machine code the capability adapter reports; the route never parses provider
  prose. Digest validation and fail-closed reads are unchanged.
- `implementer_started` is durable and written immediately before invocation.
  A pre-lane failure reports `false` truthfully; a post-implementation failure
  keeps bounded session/revision proof while remaining non-landable.
- `flyto.coding-route-status.v1` adds per-instance status files plus a bounded
  validated index under the state root. A single latest-writer file was
  rejected because many Codex conversations share one state root and an old
  process would overwrite a newer one's diagnostics. Records are closed,
  bounded, and secret-free; per-job JSON remains the only authority.

Separately, the owner requires that a broken route infrastructure must not
permanently strand all coding. `flyto.coding-emergency.v1` adds a host-owned
circuit breaker and an overflow lane to the *already selected* implementer.

- It is startup authority only (`--emergency-overflow-backend`, which must
  equal `--implementation-backend`). No environment variable, job payload, or
  model output can enable it, and it is disabled unless the flag is present.
- It opens only for a positively classified `capability_unavailable` or
  `capability_timeout` failure in a pre-implementer lane, with no attributable
  edit and no durably recorded implementer start. Every other failure category
  stays fail-closed, including domain refusals, gate denials, stale indexes,
  malformed evidence, failed checks, failed implementations, Core failures,
  Indexer post failures, audit rejections, and rework exhaustion.
- Emergency rounds keep the source-controlled checks, the exact-revision
  binding, and the independent Codex audit, and they never commit or push.
  Acceptance requires a separate digest-validated `EmergencyAuthorityReceipt`
  sealed to that job, request, session, and revision, so a receipt cannot be
  transplanted. `CodingRouteReceipt(strict=True)` is untouched and a failed
  strict route never becomes landable.
- The breaker is monotonic within one process so it cannot oscillate; the
  default threshold is 1 because each Codex conversation is a separate process
  that may only ever see one job. Recovery is a new process: a repaired build
  starts closed and publishes a new build id.

Codex validated this decision live rather than only by test: a real `code-mcp`
process with a deliberately missing Indexer overflowed one job to the pinned
`claude-opus-5` implementer, was sent back once by an independent hidden-case
rework finding despite green repository checks, resumed the same session under
a re-sealed `emergency_rework` authority, and was accepted on that exact second
revision. `STATE.md` records the job, session, revision digests, and counters.

Rollback is configuration: omit `--emergency-overflow-backend` to remove the
lane entirely, which restores the previous fail-closed behavior. Do not make
the overflow lane implicit, do not widen its classified trigger set without a
new dated entry, and do not let it skip the audit.

## 2026-08-09: Coding state roots support multiple MCP processes

Codex starts one stdio MCP process per conversation. Treating the durable state
root as a process-lifetime exclusive lease made the second conversation exit
before its MCP `initialize` response with `coding state root is already served`.
The state root is now explicitly a shared coordination boundary.

- `.service.lock` is held only around short read/modify/write decisions, not
  for the lifetime of a server process.
- Every queued/running execution round owns a crash-released per-job lease.
  A second service may reconcile an interrupted state only when that lease is
  unowned, preventing both duplicate execution and false `service_restarted`.
- Workspace edits remain serialized by a hashed cross-process workspace lock,
  while atomic replacement remains the durable JSON record boundary.
- Tenant hashing, idempotency, exact-revision audit, same-session rework,
  route gates, backend selection, and the rule that only Codex may accept and
  commit are unchanged. This is an operational concurrency correction, not a
  new route, topology edge, fallback, or authority path.

Rollback is to stop additional MCP processes and run one service instance; do
not restore the root-lifetime lock because that reintroduces thread creation
failure. If lease evidence cannot be acquired or validated, keep the affected
job non-landable and fail closed.

## 2026-08-08: Physical judge cards are immutable model input

- The competition judge physically draws one Zone card and one Objective card.
  The operator records the exact pair with `card_source=judge_draw`; Flyto2 AI
  has no draw, shuffle, or random-task behavior.
- The model schema contains only a reading, a clarification decision/key, and
  APPROVED capability IDs. Evidence requirements are repeated outside the
  model-owned object and cannot be removed or expanded by it.
- Invalid JSON, extra fields, raw controls, shortlist escape, missing card
  capabilities, or provider failure uses a deterministic fallback. The
  attestation exposes only bounded reason classes and hashes.
- Interpretation is not authority: Cloud owns plan and resource revisions,
  Robotics owns dispatch/control safety, and the control plane owns evidence
  completion.

Rollback is additive: callers can skip the interpretation service and use the
reviewed card contract directly. Do not roll back by letting a model choose the
cards, evidence contract, live resource, motor command, or completion state.

## 2026-08-08: Host-owned lanes surround the audited coding route

Extends the same-day decision below. Startup-selected implementers were
correct, but the public service still invoked the implementer directly, so the
advertised Indexer / Blueprint / Core chain was not an automatic part of an
audited job. That gap is closed at the service boundary.

- `flyto.coding-route.v1` is a typed, provider-neutral orchestration contract
  in `flyto_ai/coding/route.py`. It is not a Claude prompt convention and does
  not depend on which implementer is selected.
- `code-mcp` and `code-serve` always enable the strict route at startup.
  Direct library `CodingService` construction stays backward compatible with
  no route, and its receipt carries no route evidence rather than a fabricated
  one, so it can never be mistaken for the public audited route.
- The Indexer lane is mandatory before implementation and again after the
  source-controlled checks. Pre-work gathers real workspace context and an
  impact/task plan, executes the returned plan steps in order through an
  allowlist, and must pass its gates before the model may edit. Post-work runs
  strict verification against the final workspace state.
- Model prose never asserts that a lane ran. Every outcome is derived from
  completed allowlisted calls. A missing catalog, failed domain result,
  incomplete required action or gate, malformed evidence, exceeded bound, or
  unavailable Indexer fails closed and never reaches `awaiting_codex_audit`.
- `pass=false` blocks only its own phase and is remediated and re-gated inside
  a bounded loop; exhausting the remediation bound fails the round.
- Blueprint is a host-owned, read-only reuse lane governed by startup policy.
  It passes only a compact content-addressed projection to the implementer and
  never grants workspace or execution authority. `use_blueprint`,
  `save_as_blueprint`, and the export/import tools are outside its allowlist.
  No relevant contract yields a deterministic `not_applicable` outcome.
- Core is a host-owned validation lane, always enabled on the strict route
  and conditional only in outcome, running after implementation.
  Relevance is derived deterministically from the request and the attributable
  changed files. Calls flow through `flyto_ai.tools.core_tools` with a
  validation-only allowlist; `execute_module`, danger-full, and browser
  authority are excluded. Relevant work without an executable proof fails
  closed and is never silently marked passed.
- `CodingRouteReceipt` is an additive, secret-free, machine-checkable record
  of which lane was required, applied, skipped, not applicable, or failed,
  which calls and gates ran, and a content digest. It is coherence-validated
  on construction and revalidated on deserialization, and a failed route can
  never appear on a landable receipt.
- Nothing above weakens the existing audit: the implementer receives no audit
  tool, Claude stays pinned to `claude-opus-5` without Bash or content search,
  selection stays startup-only with no fallback, rework stays bound to the
  same job, thread, and implementation session, and Codex remains the final
  independent authority over an exact `implementation_revision_sha256`.

Rollback is configuration and stays inside the audited route. The only
supported moves are pointing `--indexer-command` or `--blueprint-command` at a
different negotiated server, or stopping the public service, which pauses
host-managed implementation. No flag detaches a lane: all four lanes are
configured on every strict public route, the Indexer lanes are always
mandatory, and Blueprint and Core may resolve only `applied` or
`not_applicable`. Do not roll back by adding a route-bypass flag, by letting a
green repository check stand in for the Indexer post-gate, or by accepting a
model-asserted lane outcome.

## 2026-08-08: One audited coding route with a startup-selected implementer

Supersedes the 2026-08-01 statement that the native `FlytoCodingAgent` is the
only coding-loop implementation. It is now one of three peer
implementers behind the same audited service contract.

- Codex, or whichever principal the host authenticates, is the orchestrator and
  the independent auditor. `flyto-ai` is the single coding route between them
  and the implementer; there is no second path that reaches a landable result.
- The operator selects exactly one implementer at process startup with
  `--implementation-backend native|claude|codex`, or the bounded
  `FLYTO_AI_CODING_BACKEND` default. `native` remains the default. There is no
  per-job backend field, no provider/model auto-routing, and no fallback in
  either direction; an invalid or unavailable selection fails startup.
- Claude service rounds are pinned to `claude-opus-5`. Configuration can vary
  the legacy direct backend's model but can never redirect audited service
  work. The Claude route reads only bounded `FLYTO_AI_CC_*` settings and
  resolves no native provider credential or configuration.
- Codex service rounds pin one installed CLI executable and an explicit model,
  open a separate non-interactive session, ignore personal config/rules, and
  inherit no ambient provider/CI credential. They expose no audit/MCP/plugin/
  web authority; host snapshots and checks remain authoritative.
- An implementer success is never public success. It reaches
  `awaiting_codex_audit` bound to an exact `implementation_revision_sha256`,
  the implementer backend, and an opaque implementation session id.
- A `rework` verdict carries typed bounded findings and resumes the exact same
  job, thread, and implementation session. A changed session fails closed.
  Rework is bounded by the startup `--max-rework-rounds` ceiling.
- Only an `accept` verdict on the exact current revision reaches
  `codex_accepted` and `landable`. Landability is eligibility evidence, never
  an action: nothing in this service stages, commits, pushes, publishes, or
  deploys, and the Claude adapter's guardian denies those command classes.
- The Claude adapter receives only Read/Edit/Write/Glob under write authority
  and Read/Glob otherwise. It never receives Bash or content search, and the
  audit tool is not in its catalog, so an implementer cannot approve itself.
- `code-mcp` and `code-serve` are audit-required unconditionally. No flag or
  environment variable disables that requirement. The MCP `initialize` result
  now advertises server version `2` and bounded instructions describing this
  loop, without claiming the transport can prove the auditing principal.

Rollback is configuration, not code, and it never leaves the audited route.
Select `--implementation-backend native` to detach either external adapter, or lower
`--max-rework-rounds` to tighten the repair ceiling; both keep `code-mcp` and
`code-serve` audit-required. Stopping the service **pauses** Codex-managed
implementation until it is restarted; it does not hand that work to another
path.

`flyto-ai code` and direct Python `CodingService` construction (which keeps
`require_codex_audit=False`) remain for legacy and library compatibility, but
they sit outside the Codex-managed audited route. They cannot produce its
`codex_accepted` receipt or its `landable` evidence, and they are never the
fallback when the service is unavailable.

Do not roll back by adding an audit-disable switch to the public commands, a
per-job backend field, a fallback between implementers, a second route to a
landable receipt, or a landing action inside the service.

## 2026-08-02: Capability quality controls are separate atomic planes

- Keep authority, resource admission, evidence/replay, adapter conformance,
  and scenario aggregation in separate modules. They have different reasons
  to change and must be independently replaceable without rewriting the MCP
  transport, registry, or central manager.
- Enforce call, failure, elapsed-time, concurrency/queue, argument/result
  byte/depth/node, and approval-timeout limits in `execution_policy`. Reject
  non-finite/non-JSON arguments, unapproved secret-shaped keys, and configured
  workspace path escape before a concurrency lease is granted. Ambiguous
  domain fields such as `path` are not assumed to be filesystem paths; hosts
  add their own path keys. Human approval is a host callback receiving redacted
  arguments; a missing, timed-out, failed, or malformed decision fails closed.
- Store capability evidence only through `execution_trace`: a bounded, deeply
  immutable redacted hash chain whose content fingerprint excludes wall-clock
  noise. Agent outer denials and Manager outcomes enter that same evidence
  boundary. Replay freezes its input snapshot, skips redacted arguments,
  permits only read-only calls by default, and requires explicit host opt-in
  for write/danger tiers; optional domain-owned normalizers handle legitimate
  drift. Blueprint feedback is emitted through a host-owned sink with a
  trace-derived stable execution id, never by exposing signing/trust authority
  to the model.
- Make adapter acceptance executable through `run_adapter_conformance()`:
  exhaustive permission classification and allowed-tool case coverage, exact
  protocol/catalog, domain-owned results, trace/policy evidence, and idempotent
  close are one content-bound report. Default test authority is read-only;
  write/danger fixtures opt in explicitly, and cases bind expected dispatch
  state so a denial cannot impersonate a domain failure. Aggregate suites
  through `scenario_matrix`; scenario/domain strings remain metadata, never
  manager routing branches.
- Keep external reality honest. Workflow, page, robotics, and authorized
  security-lab fixtures prove composition semantics and failure containment;
  they do not claim control of unconfigured hardware or authorization against
  third-party systems.
- The complete clean-runner suite checks out the exact Blueprint benchmark
  dependency commit beside `flyto-ai` and installs it explicitly. Local sibling
  availability must not hide a missing CI dependency, and an unpinned moving
  Blueprint branch must not change the proof after a `flyto-ai` commit lands.
- Clean-runner command tests provision ripgrep explicitly and load the Python
  sandbox image from an immutable linux/amd64 digest before assigning the
  runtime-compatible `python:3.12-slim` tag. CI must exercise the real
  fail-closed OS sandbox instead of silently skipping it or depending on a
  mutable preloaded runner image.
- Docker protected-file masking uses an unreadable host inode rather than
  `/dev/null`. The latter hides bytes but still lets a Linux container report a
  successful read; the unreadable bind keeps the cross-platform contract
  fail-closed while protected directories remain zero-permission tmpfs mounts.
- `Agent` owns its lazily opened memory database and transcript writer. It now
  exposes an idempotent async lifecycle and rejects use after close; callers
  should prefer `async with` so SQLite worker threads and evidence files are
  closed before their event loop terminates.
- CI promotes deprecation and unhandled-thread warnings to failures. A green
  run therefore proves lifecycle cleanup instead of merely attaching a warning
  annotation to an otherwise successful test job.

Rollback is layered: detach the policy controller, trace sink, or conformance
runner independently while preserving the existing facade and profile
contracts. Do not roll back by widening tool catalogs, accepting secret
arguments, skipping approval, or trusting unmatched replay evidence.

## 2026-08-02: Agent-stack internals are atomic behind stable facades

- Keep `flyto_ai.coding.stack` as the public composition/CLI facade and
  `flyto_ai.coding.capabilities` as the public session/manager facade. Existing
  imports remain identical while their implementation responsibilities are
  split into independently replaceable modules.
- Give each module one reason to change: `stack_manifest` owns bounded profile
  I/O, schema, composition, and configured fingerprint; `stack_presets` owns
  only the detachable built-in catalog; `stack_probe` owns observed runtime
  attestation; `mcp_transport` owns isolated subprocess and bounded JSON-RPC;
  `mcp_catalog` owns tool naming, scoping, and domain-result normalization;
  `mcp_session` owns handshake and call orchestration; `tool_registry` owns
  transactional provider-name registration; and coding `permissions` owns the
  monotonic runtime permission evaluation.
- Reject partial registry state and provider-name collisions. A failed session
  registration closes the new process, closes previously started sessions,
  and clears all dispatch and permission metadata.
- Keep argument-sensitive risk resolvers host-owned and pluggable. A resolver
  may raise the manifest-declared requirement but can never lower it. Adding a
  robotics, security, data, or operations adapter therefore does not require a
  new task-name branch in `CapabilityManager`.
- Close stdin and await normal child exit before bounded terminate/kill
  escalation. Session and manager close operations are idempotent and leave no
  dispatchable tools or orphaned asyncio subprocess transports.
- Require evidence at four levels: pure boundary tests, real subprocess MCP
  integration, Agent/Manager bypass tests, and the complete repository suite.
  The four-lane observed composition fingerprint remains exactly
  `648c821f1c2a6d462a8b9afce3e8a575366aa4c952b9887f8a3717637e56854f`.

Rollback is one atomic implementation change: revert the internal modules and
facade imports together. Do not roll back by weakening v2 classification,
runtime ceilings, catalog scoping, lifecycle cleanup, or collision rejection;
the stable facades and v1 profile compatibility remove the need for that.

## 2026-08-02: v2 profiles classify authority per tool and enforce it twice

- Keep `flyto.agent-stack.v1` readable for compatibility, with its historical
  workspace-write default for tools lacking policy metadata.
- Make `flyto.agent-stack.v2` the recommended profile contract. Every MCP tool
  in its `allowed_tools` catalog must be classified exactly once as
  `read_only`, `workspace_write`, or `danger_full`; missing, extra, duplicate,
  or unknown classifications fail before process start.
- Treat source-controlled classification as a requirement, not a grant. The
  runtime host independently chooses the `CapabilityManager` permission
  ceiling, and a tool cannot raise that ceiling from YAML or MCP metadata.
- Enforce the effective permission in both the generic `Agent` dispatcher and
  `CapabilityManager.dispatch()`. Direct manager callers therefore cannot
  bypass the Agent gate.
- Preserve argument-sensitive Core checks after MCP provider-name isolation.
  An `execute_module` call classified as workspace-write is escalated to
  danger-full when its actual module category is shell, process, Docker,
  Kubernetes, SSH, network, filesystem, environment, Git, or another existing
  danger category.

Rollback is additive: load an existing v1 manifest or omit `tool_permissions`
from direct `CapabilitySpec` construction. The runtime ceiling and historical
workspace-write default remain; reverting never turns a blocked call into an
implicit danger-full grant.

## 2026-08-02: Agent composition is domain-neutral; authority remains domain-specific

- Keep the shared closed loop independent of task names: normalize intent,
  route installed capabilities, apply policy/authorization, plan, execute,
  verify, and record bounded evidence/Blueprint feedback.
- Keep Indexer, Blueprint, page inspection, and Core as the default coding
  preset, not a hardcoded universal stack. Hosts may load arbitrary
  source-controlled `flyto.agent-stack.v1` profiles or compose
  `CapabilitySpec` groups in Python.
- Make `CapabilityManager` a generic `ToolExecutor` so the same process and
  allowlist boundary can serve `Agent` as well as the coding adapter.
- Require every manifest-loaded MCP capability to declare a non-empty
  `allowed_tools` list. Extensibility does not grant a model the server's full
  discovered catalog.
- Preserve specialized adapters where proof or harm models differ: coding owns
  workspace/check evidence, robotics owns safety and human gates, and security
  campaigns own explicit scope, authorization, expiry, actions, modules, and
  budgets. New domains add a typed contract, guardrail, executor, verifier,
  evidence projection, tests, and rollback notes instead of weakening the
  common boundary.

Rollback is additive: use the built-in coding preset, detach a profile entry,
or stop loading the manifest. Existing domain adapters and public contracts do
not depend on a custom profile.

## 2026-08-02: Full agent composition is tool-allowlisted and detachable

- The provider-neutral `FlytoCodingAgent` remains the owner of the coding loop.
  Indexer, Blueprint, page inspection, and Core attach as four versioned MCP
  capability specs rather than sibling source imports or alternate agents.
- `required_tools` proves compatibility; the new optional `allowed_tools`
  field defines model authority. With an allowlist, every named tool must exist
  and no other discovered server tool is exposed or dispatchable. Omitting it
  preserves the existing full-catalog contract.
- Blueprint and page inspection can use separately started views of the same
  Flyto2 AI MCP implementation. The Blueprint view cannot call `chat`, Core, or
  page inspection; the page view exposes only `inspect_page`.
- `flyto.agent-stack.v1` preflight performs real initialize and `tools/list`
  negotiation and hashes the observed component identities, protocols, and
  exposed tools. It does not invoke a model, navigate a page, or read secrets.
- Page inspection keeps Core as its only browser authority. `auto` tries
  bundled Chromium and then installed Google Chrome and records the chosen
  channel. A successful MCP envelope cannot override nested domain failure.

Rollback is additive: omit any stack component, remove its `allowed_tools`
field to restore the previous full-catalog behavior, or stop using the stack
builder while retaining the underlying `flyto.coding.v1` contract.

## 2026-08-01: Coding service adapters are detachable and tenant-bound

- The native `FlytoCodingAgent` remains the only coding-loop implementation;
  HTTP and MCP are optional facades over a versioned `flyto.coding-service.v1`
  service contract, not alternate agents.
- A service instance resolves its provider, credentials, tenant, workspace
  allowlist, and state root at startup. Job payloads cannot select a tenant or
  carry API keys, bearer tokens, cookies, or provider credentials.
- HTTP jobs require authentication and an idempotency key. Tenant ownership is
  derived from server-side authentication, and job lookups fail closed across
  tenant boundaries. MCP stdio receives the tenant from process configuration.
- Capability configuration declares the MCP protocol version and required tool
  names. Availability is based on the negotiated initialize response and the
  actual `tools/list` catalog; configuration text alone is never proof.
- Authenticated MCP subprocesses may receive only explicitly named `FLYTO_*`
  variables from the runtime environment. Configuration stores names, never
  values; unrelated cloud, source-control, SSH, and provider credentials remain
  absent from the child environment.
- Concurrency is bounded per service and per workspace. Duplicate submissions
  reuse the original durable job; conflicting reuse of an idempotency key is
  rejected.

Rollback is additive: stop or remove the optional HTTP/MCP process and continue
using `flyto-ai code`. No Cloud, Core, Indexer, Blueprint, Engine, or Robotics
repository imports this implementation.

## 2026-08-01: Flyto2 owns the coding loop; vendor agents are adapters

- `flyto.coding.v1` is the stable request/result/evidence contract. The native
  backend uses the selected Flyto2 provider and does not depend on Codex or a
  vendor agent SDK.
- Claude SDK remains a separately selected compatibility adapter and may be
  removed without changing providers, checks, threads, or capability contracts.
- Indexer, Core, and future visual/runtime services attach through explicit
  versioned MCP-stdio entries. Required adapters fail closed; absence never
  grants fallback authority or triggers a sibling source import.
- Model prose is not proof. A run succeeds only after source-controlled real
  commands pass and, for mutating work, snapshot evidence attributes a change
  to the run.
- Native file authority stops at one workspace root and provides no danger-full
  mode. Hostile-code isolation belongs to an outer container or VM.

Rollback is additive: select the compatibility backend or remove the `coding`
package. Existing provider, Blueprint, Core, and Cloud contracts remain intact.

## 2026-07-31: The LLM plans security work; Core remains execution authority

- Footprint, penetration-test, and red-team planning use one versioned
  `flyto.security-campaign.v1` contract.
- Scope, authorization tier and expiry, approved action classes, module
  allowlist, request/round/token/cost budgets, and prior usage are frozen into
  every plan identity.
- All execution still passes through the existing closed-loop MCP and
  `flyto_ai.tools.core_tools.dispatch_core_tool`; there is no security-only
  dispatcher that bypasses Core validation or permissions.
- Model-visible evidence is structurally allowlisted and omits raw target
  content. Failed output is represented by bounded error classes and hashes.
- A campaign can re-plan only within its original authority ceiling and
  cumulative budgets. Missing proof yields `not_proved`, never an inferred
  success.

This lets a real LLM choose and adapt attacks while keeping authorization,
scope, cost, evidence, and the final verdict independently enforceable.

## 2026-07-30: Let the model choose a complete route, not invent waypoints

- Robotics supplies a bounded shortlist, trusted semantic location IDs, and
  complete route candidates after deterministic compatibility, permission,
  resource-health, and dependency filtering.
- Flyto2 AI converts every surviving route into an exact JSON Schema step
  template. The model chooses one candidate and fills bounded arguments; it
  cannot omit an intermediate location or combine parts of different routes.
- Every motion plan must end in `safe_stop`. Human approval and resume IDs must
  pair before later movement. Direct control fields such as `cmd_vel`, wheel
  speed, PWM, motor, shell, and ROS topics are rejected recursively.
- The response attests request, schema, plan, model, provider, attempts, token
  counters, timing, and selected route. Robotics independently verifies and
  executes the same canonical plan bytes.
- Repair is limited to one additional structured completion. If both proposals
  fail, the planner returns no plan.

This retains a visible AI decision at a multi-branch junction without moving
real-time control or safety authority into an LLM.

Rollback is additive: stop the loopback planner, remove the Robotics planner
URL, and continue using existing prevalidated plan inputs. No provider or Core
tool contract needs to change.

## 2026-07-28: Natural language is an adapter, not a routing contract

- Any language, UI, speech, schedule, or sensor event is normalized into
  `flyto.goal-frame.v1`.
- Capability manifests declare canonical intent IDs, affordances, effects, and
  handled events. Exact semantic coverage is the production ranking signal.
- Raw text, aliases, and examples are used only when a legacy caller provides
  no Goal Frame.
- Production callers can require a valid Goal Frame and fail closed before
  catalog discovery.

This prevents the router from accumulating per-language synonym tables and
makes identical meaning produce identical candidates regardless of wording.

## 2026-07-28: Capability catalogs are routed before provider dispatch

- External runtimes publish versioned JSON manifests; Flyto2 does not import
  their source trees.
- Compatibility, permission, domain, and source scope are hard filters.
- Blueprint may boost only module IDs from summaries that pass the existing
  trust/evidence gate.
- Core discovery flows only through `flyto_ai.tools.core_tools` and cannot
  escape an explicit source scope.
- The LLM receives a bounded, snapshot-bound shortlist and ambiguity evidence,
  not the complete catalog.

This keeps selection reproducible as registries grow and prevents a model or an
upstream keyword score from turning an irrelevant module into executable
authority.

## 2026-07-27: Keep security updates, suppress routine dependency branches

- Dependabot security updates remain enabled in repository settings.
- Routine pip and GitHub Actions version-update PRs are disabled with
  `open-pull-requests-limit: 0`.
- A dependency branch is merged when it closes a security alert or has another
  verified product need; lower-bound-only bumps are not merged merely because
  a newer compatible version exists.
- Grype ignores `GHSA-vxmw-7h4f-hqxh` only for the exact pinned
  `pypa/gh-action-pypi-publish` 1.14.1 SHA. The advisory fixes the issue in
  1.13.0, but Syft exposes a pinned Action's SHA as its version, which Grype
  cannot compare semantically. A package-, type-, version-, or advisory-wide
  exception is not allowed.
- Repository policy tests pin the least-privilege CI permission and patched
  publishing action so later edits cannot silently reopen the same alerts.

The project has no dependency lockfile and CI already installs current
compatible releases. Raising minimum versions alone would reduce compatibility
without changing what CI scans or installs.

## 2026-07-26: Prove token reduction at the execution boundary

- Only deterministic exact Blueprint reuse records
  `planner_model_calls_used=0` with `model_call_scope=planner`.
- Blueprint still accepts the old `model_calls_used=0` compatibility field,
  but new Flyto2 AI reports do not emit it. It must not be described as
  workflow-wide zero tokens because an `llm.*` step can still call a model.
- Model-selected paths do not assume whether one or several model calls were
  used.
- The closed loop forwards only allowlisted runtime facts to the Blueprint
  Evidence Card; prompts, params, secrets, and raw results stay out.
- Documentation must describe measured zero re-planning calls, not estimated
  percentage savings or workflow-wide zero tokens.

This makes the “lower token use” claim falsifiable and keeps the evidence
surface small enough to inspect.

## 2026-07-26: Blueprint evidence authority is an in-process capability

- Model-facing outcome reports are always community observations.
- The deterministic Blueprint loop adds a non-serializable object-identity
  capability only after guarded execution; only this path writes
  `local_verified` evidence.
- Model-facing portable exchange cannot supply signing keys or trusted
  publisher mappings; those remain host configuration.

This prevents JSON tool calls from self-promoting shared procedures while
retaining continuous evidence-backed learning.

## 2026-07-26: Closed-loop verification rejects ambiguous plan state

- Omitting both verification identifiers remains a request-shape error.
- An unknown `plan_id` is reported as missing state instead of being folded
  into the request-shape error.
- A known plan without a recorded execution is reported as lacking execution
  evidence, so callers can execute it before retrying verification.

## 2026-07-22: Generate exhaustive implementation references

- Human-authored guides explain behavior, boundaries, and operations; generated references provide exhaustive symbol/CLI/tool/environment inventories.
- Package version comes from `pyproject.toml` in a source tree and installed package metadata in a wheel, preventing CLI/MCP/version drift.
- Core module totals are discovered from the installed registry at runtime; source code does not freeze a fallback count.
- CI validates the documentation manifest and rejects stale generated output.

## 2026-06-21: flyto-core stays the MCP authority

- `flyto-ai` adapts `flyto-core` tools instead of duplicating module metadata.
- The adapter adds metadata and validation but preserves existing tool names and result shapes.
- Cloud should consume `flyto-ai` capability manifests rather than importing `flyto-core` internals.

## 2026-06-21: Agent Builder is not a dependency

- Agent Builder concepts can inform workflow UX, but product code stays code-first and provider-agnostic.
- Durable primitives are MCP, typed tools, traces, evals, guardrails, approvals, and evidence.

## 2026-08-10: Cross-job continuation is a single-use durable authority

A provider round that stops at a configured ceiling (`provider_job_budget_exhausted`
or the bounded turn stop) is real, attributable work. Carrying it forward is an
explicit second `submit` with `resume=true, thread_id=<exact SDK session>`, never
an implicit retry and never a fallback.

- Permission is a tenant-partitioned, single-use **continuation authority** that
  binds tenant, backend, exact session, originating job, workspace identity and
  path, attributable revision, whole-workspace snapshot, snapshot policy,
  authorized verification contract, request digest, stop code, and a monotonic
  generation. Holding a session id proves nothing.
- The monotonic truth lives *outside* the replaceable authority body, in an
  append-only hash-chained **journal**. Its tail is the only thing allowed to say
  which generation and state a session is at, so restoring an older-but-validly-
  signed record is a replay and is refused.
- Transitions are exact: `open(g) -> claimed(g) | settled(g)`,
  `claimed(g) -> open(g+1) | settled(g)`, `settled` terminal, sequence always +1.
  Tenant, backend, session, origin job, workspace, config, request and snapshot
  policy are invariant across every transition.
- Claiming is a compare-and-swap against the journal tail under an exclusive
  `flock`, so many independent Codex processes sharing one state root produce
  exactly one owner. Every loser gets one non-disclosing code.
- Generations are bounded (`MAX_CONTINUATION_GENERATION`). Nothing spends a
  segment automatically and no configured model budget or turn ceiling is raised.

## 2026-08-10: A workspace snapshot has an explicit, digest-bound projection

Continuation re-proves the *whole* workspace before provider contact, because a
digest of only the attributable change set cannot see an unrelated file another
agent added between segments.

- The default projection observes every entry that is not root version-control
  state (`.git`/`.hg`/`.svn`). There is no blanket ignore for `node_modules`,
  `.venv`, build output or caches: those are exactly where undetected input
  change would matter.
- A `SnapshotPolicy` may classify a small number of **exact root-relative**
  directory names as control-plane runtime state. Only the strict public route
  with a required Indexer capability is granted one, and only for `.flyto-index`,
  because that route's mandatory Indexer pre/post gates independently revalidate
  that tree and record the result in the route receipt. Without the gates there is
  no justification, so every other configuration gets the default projection.
- The policy identity is hashed into the manifest digest and frozen into the
  authority. Policy drift, an added exclusion, a malformed policy, or a
  strict-route authority replayed on a non-strict service all refuse before the
  provider is contacted.
- A classified directory's *presence* is still observed; only its contents are
  another component's business. A nested directory with the same name stays
  ordinary source.

Rationale: `flyto-code` carries a live `.flyto-index/task-runs.sqlite` that its
Indexer rewrites continuously. Under a whole-tree digest that repository could
never be continued - for a reason that has nothing to do with its source.

## 2026-08-10: Admission is phased; repository observation never holds the global guard

`CodingService.submit` runs in three phases so one large repository cannot stall
every other tenant and workspace:

1. unlocked idempotent-replay read, so a replay never pays for a scan;
2. per-workspace admission lock, holding the verification-contract read and the
   workspace snapshot - the expensive work;
3. the global state guard, held only for bounded reads and writes: authoritative
   replay, build-drift, capacity, the continuation compare-and-swap, job lease,
   record, workspace claim, idempotency record and executor hand-off.

Lock order is **workspace-admission -> state guard**, always. The admission lock is
never taken while the state guard or a round's workspace lock is held, so there is
no cycle. The admission lock is deliberately distinct from the per-round workspace
lock: a submit never queues behind a running model round, and a round never queues
behind a scan.

## 2026-08-10: A rework is one root task, with cumulative plan authority

Job `job_1be3e31602264f88b617b42a` took three implementation rounds. The third
passed every host check and was still refused by strict Indexer post-work with
`unplanned_diff`, surfacing only as `route_domain_failure`. Two defects with the
same shape produced it, and both are now closed.

- **The pre-lane amends, it does not re-root.** After a genuine pre-lane
  success the exact contract is sealed into a bounded, integrity-protected,
  private envelope in the job record, bound to that job, its root request and
  its workspace. A rework loads and re-proves that envelope and passes it back
  as `task_contract`, so the root task id and objective are preserved and scope
  grows only by typed amendment. With no parent the request is byte-for-byte
  what it always was, so a legacy Indexer sees no new argument.
- **Post-work validates what the audit will bind.** The cumulative attributable
  set is proven *before* the proof lanes run and handed to both Core and the
  Indexer, and `_record_outcome` refuses unless the persisted
  `implementation_files` equal that exact ordered tuple. Validating a narrower
  scope than an auditor is later offered is not validating.
- **Prior authority is re-proven before the implementer edits.** Session,
  resume envelope, worktree claim, workspace path, bounded prior set and the
  exact prior revision digest are all checked while the recorded revision still
  describes the tree. A missing, stale, replayed or tampered parent fails with
  no provider call.
- **A machine identifier is not edit-path authority.** A path that does not yet
  exist needs a real file extension *and* a generic mutation verb adjacent to
  it, so `check.generated_reference` and `pkg/check.some_capability` are
  evidence about a round rather than a request to create a file, while
  `add tests/test_x.py` and `create app/[id]/page.tsx` still work.
- **Rework context is not new mutation authority.** Amendment findings mix edit
  requests with commands, check output, and evidence references. An existing
  path mentioned only in those contextual forms is not added to the amended
  target ledger; a bounded mutation cue in the same clause is required. The
  revision-proven prior scope is always retained, including when the finding
  contributes no new target. First-round targeting and its safety/polarity
  rules are unchanged. Rollback is removal of this amendment-only projection,
  which restores the known over-broad rework behaviour without changing the
  Indexer contract.
  Regeneration is a mutation of its tracked output, but a later bounded
  execution connector (`using`, `via`, `use`, `call`, `calling`, `run`,
  `execute`, `invoke`, `with`, `through`, or bare `by`) is a command boundary
  and does not grant authority over the program. `By running`, `by executing`,
  and `by invoking` remain execution-only. The deliberately bounded positive
  forms `by modifying`, `by editing`, `by updating`, and `by changing` place a
  later mutation cue after that boundary and therefore grant authority to the
  program. An explicit request to modify both output and program still grants
  both. `Include` and evidence language without a mutation cue are
  intentionally contextual rather than authoritative.

## 2026-08-10: Domain diagnostics are validated before they are normalized

A capability's `pass=false` is the one refusal a caller can act on, so its
`reason_codes` and `required_actions` survive into host-owned
`verification_blockers` and the receipt detail. What crosses is bounded and
closed: a value must *already* match a machine-identifier grammar
(`[a-z0-9][a-z0-9_.:-]{1,63}`) before its separators are mapped into the public
blocker grammar.

The order matters more than the grammar. Normalizing first turned
`please open /Users/alice/private token` into
`validation_please_open_users_alice_private_token`, which reads exactly like a
token this host owns. Invalid or overlong values are now dropped whole, never
truncated into validity, and that includes screaming-case values that were
previously rescued.

## 2026-08-10: Plan-authority failures report `verification`, not `preflight`

`preflight` is documented as "no job exists and no claim was taken". A
plan-authority refusal happens inside an admitted job that already holds a
durable worktree claim, after capability startup and before the implementer
call, so it reports `verification` with `retryable=false` and the existing
closed action `resubmit_against_current_contract`. An identical rework cannot
restore a durable fact that is missing or contradicted; a fresh job against
current authority can.

## 2026-08-12: Resolve repository-relative check programs from the workspace

The source-controlled verification contract is interpreted in the repository
whose change it governs. Therefore an argv program containing a relative path
separator is resolved from `request.working_dir`; a plain program name still
uses `PATH`, and an absolute path remains absolute. The same resolver and
workspace argument are used by submit preflight, the native implementation
agent, the Claude implementation adapter, and the real check runner. Using the
service process cwd is forbidden because it makes the same contract pass or
fail solely according to the supervisor launch directory.

## 2026-08-12 — Durable Scheduler converges on MissionStore

- `MissionStore` remains the sole execution scheduler and state machine. The
  Scheduler catalog owns only validated immutable definitions, enabled state,
  deterministic due cursors and slot claims, bounded public result summaries,
  and mission/work-item mappings.
- Durable Scheduler execution has no fallback lane: an occurrence is submitted
  idempotently, dispatched through a real `DispatchHandle`, automatically
  heartbeaten while awaiting the executor, and closed with its live lease and
  fence. Failed or over-budget outcomes close blocked; only policy-valid success
  closes fixed.
- Scheduled tasks use bounded mission generations. Each generation has one
  fixed internal container anchor and bounded occurrence side items, and is
  completed before rollover. The catalog does not own worker, lease, fence,
  running, or executor-success authority.
- `state_root=None` remains the backward-compatible process-local mode and is
  explicitly reported as ephemeral. Supplying `state_root` selects the durable,
  owner-only, multi-process catalog and MissionStore-governed path.

## 2026-08-13 — Execution Sessions validate activation without minting authority

- A governed Execution Session is an exact, versioned and domain-neutral bridge
  from an upstream activation claim to planning. It is not a microphone, wake
  detector, STT layer, provider adapter, Cloud runtime, device runtime, or
  scheduler.
- Space wake words remain bounded and normalized, but this bridge accepts no
  observed wake claim. A display name is not a wake word, and activation never
  authenticates a principal or grants a permission.
- Canonical Spaces may be unnamed and voice-disabled: `display_name` accepts a
  bounded exact-empty string or bounded non-whitespace text, while whitespace-only
  labels fail closed without trimming or collapsing ordinary display text.
  `wake_words` accepts zero through 32 safe,
  normalized-unique values. `active_timeout_ms` is the sole timeout field and a
  strict integer from 1 through 300,000; the activation window is compared to
  it directly in milliseconds, with no seconds alias or unit inference.
- The complete v1 activation-source vocabulary is `typed`, `voice_reviewed`,
  `external_agent`, and `mission_card`. Every source requires
  `observed_wake_word` to be exactly null. `voice_reviewed` records upstream
  review only and claims no wake detector; raw `voice`, `button`, and unknown
  sources fail closed.
- Tenant and principal identity plus allowed source/domain, granted permission,
  and enabled-capability ceilings are accepted only through a verified frozen
  host authority. The untrusted request cannot provide or widen them. Existing
  goal-frame normalization and capability routing remain authoritative, with
  all four ceilings supplied as hard-filter context and no fallback.
- Outputs are detached, canonical, plain JSON containers and are
  principal-minimized. Distinct request/result versions prevent wire-direction
  confusion, while SHA-256 attestations bind canonical request, authority, and
  route projections. A deterministic overall digest excludes only its own field
  and covers the result contract version plus every governed payload field.
  Caller mutation cannot retroactively alter the returned result, which remains
  directly JSON serializable; the host authority itself stays frozen.
- JSON is preflighted iteratively before recursive normalization or hashing.
  The exact ceilings are depth 32; request 4,096 nodes/262,144 bytes; manifests
  500,000 nodes/8,388,608 bytes; Blueprints 500,000 nodes/1,048,576 bytes;
  JSON integers from -9,223,372,036,854,775,807 through
  9,223,372,036,854,775,807; timestamps 0..253,402,300,799,999 ms; and route
  limit 1..32 as a non-boolean integer. Breaches fail closed without truncation.
  Nodes count container and scalar values including the root, but not object
  keys, which remain covered by the byte limit.
- Rollback removes the bridge and stops admission of
  `flyto.execution-session-request.v1`; it does not weaken router filters or reinterpret
  request data as authority. Product topology and integration ownership do not
  change.
## 2026-08-14 — Trusted connectors execute under the durable admission fence

- A production host may transfer one keyword-only, pre-established one-shot
  `ExecutionSessionConnector` handle to `admit_execution_session`; request data
  can neither provide nor name it. Handle construction starts the non-daemon
  worker before admission, so the deadline-bound path never calls
  `Process.start`.
- The connector receives only a detached bounded prepared-session snapshot in
  memory. Durable instruction remains session/task IDs plus the four governed
  digests. Scheduler alone owns the one-shot fence, closure, and evidence ref.
- Accept only the exact four-field Scheduler result. Success is exact `ok=true`,
  empty message, null error, and zero cost. Failure is the one stable empty
  `execution_connector_failed` result. Invalid output and exception details are
  never persisted. A possibly entered but unprovable outcome becomes
  `execution_outcome_unknown` and is never automatically replayed. A concurrent
  duplicate that is already waiting continues Scheduler reconciliation after
  owner cancellation, using the existing fence rather than a connector replay.
- Derive one absolute monotonic deadline from
  `activation.expires_at_ms - now_ms`. Recompute its nonnegative remainder at
  actual executor entry and never invoke at zero. Bound worker readiness,
  nonblocking request transfer, the child-side entry check, and result receive
  to that same deadline. On expiry, owner cancellation, validation failure,
  connector failure, or normal return, forcibly terminate it and confirm process exit
  before returning or publishing the stable empty timeout result. Bound both
  forced cleanup and duplicate reconciliation to the fixed 0.5-second closure
  grace, with a 10 ms sleeping pass interval.
- Transfer ownership of the one-shot handle to admission: it is closed even
  when validation fails or another host already resolved the durable occurrence.
  Do not treat in-loop cancellation as lifecycle proof. There is no detached
  callback task, daemon thread, or global connector slot. If the child cannot be
  proven dead, fail closed without closing a durable receipt. This contract uses
  the host's enforceably terminable process boundary because arbitrary Python
  coroutine code cannot be forcibly stopped in-process.
- Persisted result wins over any later callback, including after restart;
  digest drift is a conflict. No callback preserves the prior exact blocked
  receipt. This decision adds no Cloud or device/runtime claim.

**Rationale.** A host needs a production connection point without creating a
second execution ledger or letting provider identity/content enter durable
authority. Reusing Scheduler's occurrence gives cross-process at-most-once
entry and content-free recovery. Activation expiry is existing validated host
policy and avoids inventing a provider-controlled timeout. Rollback stops
supplying the connector, which restores the original blocked behavior without
rewriting durable records.

## 2026-08-13: A successful cross-connection rework audit owns its worker round

**Decision.** Supervisor ownership is bound to both the request and its
truthful response. A successful local submit owns its non-terminal job. A
successful `flyto_coding_audit` owns a job only when its request explicitly
carried `verdict=rework`, the response is addressed to that request, returns the
same valid job id, and reports `rework_queued` or `rework_running`.

**Rationale.** Rework audit is not observation: it queues or runs the next
implementation round, so the connection that performed it must keep that child
alive through source drift. Reads and other audit outcomes remain observations
and cannot adopt foreign work. Matching terminal evidence clears only an
already tracked pin. This default-deny rule prevents response prose, future
tool names, malformed/error replies, and wrong-job receipts from manufacturing
ownership or terminal-clear authority. Only exact submit/get/audit responses
may observe jobs; get/audit must also return the requested job id. Rollback is
removal of audit-rework registration while retaining
submit tracking and durable terminal reconciliation; that rollback reopens the
confirmed restart gap and is not operationally safe during rework.

## 2026-08-13 — Capability Cards bind semantics to exact host authority

- Phase 1 defines distinct `flyto.capability-claim.v1`,
  `flyto.capability-card.v1`, and `flyto.capability-search.v1` contracts. The
  boundary is provider and domain neutral and calls no existing router.
- Claims own only bounded display/semantic metadata, semantic origin, and a
  nullable source kind/reference. Exact schema validation rejects authority
  injection, unknown fields, unsafe text/types, and raw parameters, credentials,
  tokens, endpoints, bodies, or arbitrary configuration.
- Tenant, Space, catalog id, approval, verification, lifecycle, and binding to
  the exact canonical claim SHA-256 digest come only from frozen host authority.
  Digest mismatch is rejection, not a draft card.
- Completeness requires a non-blank title and summary plus at least one semantic
  id. Source identifiers never synthesize semantic truth. Only complete,
  approved, verified, active, non-retired exact bindings are autonomous-routable.
  Other represented states are audit-visible and non-routable; static derivation
  cannot mint verification.
- Search is a deterministic bounded allowlist projection. Phase 1 has no
  persistence, vector indexing, retrieval/rerank, installation, execution,
  approval/verification service, or UI. Rollback removes these contracts without
  changing Goal Frame, `capability_router`, Execution Session, or topology.

## 2026-08-13 — Retrieval preserves upstream contracts and is never route authority

- Adopt one exact `flyto.ai.capability-retrieval-handoff.v2` handoff and frozen
  `CapabilityRetrievalAuthority`. It preserves the accepted Blueprint
  request/page and Cloud result/feasibility field sets and digest meanings.
  Host verification binds tenant, workspace, Space, model/index/snapshot and
  exact upstream request/context/requirements/result digests. AI-local goal,
  routing context, and Goal Frame use three distinct versioned digest names.
- Admit only a complete terminal page (`top_k` 1..32, equal page size, no input
  or next cursor) whose exact candidates are active, uniquely and
  deterministically ordered, canonically digested, and explicitly
  candidate-only without execution authority.
- Require true Cloud feasibility and exact candidate-resource evidence without
  inventing a co-location rule for requirements satisfied on distinct
  resources or requiring feasibility capabilities on the page. Bound the
  feasibility result to 128 canonical capability keys. Preserve all
  distinct installed providers bound to one accepted capability document,
  ordered by full provider identity; reject duplicate identities and unknown
  documents. Use `CAPABILITY_GROUP_LIMIT` for the 32-group public bound and
  independently use `EMITTED_PROVIDER_ROW_LIMIT` for the 32 emitted provider
  rows. Equal current values do not alias their units or authority. Expand
  every selected group completely and deterministically; if it cannot fit,
  fail closed without a partial group. Rebuild the producer's exact model dialect and active, ACL, risk,
  resource, capability-filter, scope, and candidate bindings. Preserve its
  `/`-capable identifier syntax and field bounds. An empty capability list is
  open discovery; only a nonempty list is an allowlist.
- Bound and detach AI-local routing context and normalized Goal Frame before
  hashing or returning them. Hostile types, depth, nodes, bytes, non-finite
  values, and oversized integers fail through the content-free boundary.
  Blueprint model ID/version are exactly 128 characters maximum; tenant, Space,
  and capability identifiers remain 192.
  Retrieval may narrow the catalog and add at most one relevance point. It may
  not supply semantics, manifests, grants, resource truth, approval,
  verification, parameters, secrets, or execution authority.
- Preserve safety/human-gate candidates and the existing planning, permission,
  and execution closure. Empty retrieval is non-routable; malformed, stale,
  partial, mixed-scope, or infeasible retrieval fails with one content-free
  error. Rollback removes the optional handoff only. The product topology is
  unchanged and vector retrieval remains non-authoritative.
- Lock this producer-compatible decision to Blueprint
  `f3eb62eff97fac3b3f19d2f1c8d7c1e71664894b`, Core
  `a048bc47de158c096b7010642452e4d41d21748c`, and Indexer
  `b492ef9b663f4a37c4883e2b9e1d8b45b3719b6d`. Blueprint owns
  request/model/index/snapshot/page/candidate digest meanings; Cloud owns
  query-context/requirements/feasibility/result meanings. Frozen host
  validation preserves both, but every result remains candidate-only with
  `execution_authority=false`.
