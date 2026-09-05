# Changelog

## 2026-09-06

- Refuse unresolved workflow references before direct module side effects;
  resolve explicit data bindings through Core and keep diagnostics secret-free.

- Added caller-owned browser scopes with concurrent isolation and acknowledged,
  bounded cleanup on success, failure, and cancellation.

## 0.20.0

### Fixed

- In execute mode the assistant no longer says it is running a workflow
  without calling the tool. The forced tool choice fired only for messages
  that looked like a web task by a word list; 「幫我登入kintone」 did not, the
  model replied 「我將執行 "kintone" 工作流程…請稍候。執行中...」 and nothing
  ran, and the polite nudge that followed would have discarded a `kintone`
  call even if it had come. An honesty guard now stands after the nudge:
  runnable tools on the turn, no call made, a reply that reads as a
  commitment — one retry with the tool choice forced where the provider can
  (OpenAI `required`, Anthropic `any`; providers declare
  `supports_forced_tool_choice`), and otherwise, or on a second narration,
  an honest sentence in the operator's language that nothing ran and what
  could. 執行中 / 請稍候 never reach the user without a call.
- Memory embeddings authenticate with the provider the operator configured,
  not with whatever `OPENAI_API_KEY` the process happened to carry. On the
  owner's desktop that was a 401 from `/v1/embeddings` after every turn,
  swallowed at debug level and retried on the next. `EmbeddingStore` takes
  the configured key and base URL; when embeddings are unavailable it says
  so once per process and steps aside, and keyword memory and the chat
  itself are unchanged.

## 0.19.0

### Fixed

- Confirming an action now confirms it. `classify_tool_intent` reads one
  message in isolation, which is right for a safety boundary and wrong for a
  conversation: the assistant asked whether to run a workflow, the operator
  replied "確認", and that classified as small talk — no tools were exposed, so
  the only reply available was to ask again. A bare affirmation is promoted to
  `action` only when the assistant's previous turn actually asked, read off the
  transcript rather than held as state, so a resumed or forked session behaves
  the same. Agreement to a remark stays conversation, and a message that agrees
  and then asks for something else classifies on its own terms.

- The action-verb tables carried no "登入" and no "log in", so the plainest
  instruction this product receives classified as `answer_only`: the assistant
  explained the login instead of performing it, and the caller had to discover
  by trial that a different wording worked. Access and lifecycle verbs are
  added in both the English and CJK tables.

### Changed

- The verb tables and the routing decision move into
  `flyto_ai/intelligence/action_verbs.py` and
  `flyto_ai/intelligence/confirmation.py`. A verb list over ten languages is an
  open set, and the safety boundary should not pay for vocabulary it cannot
  help accumulating — `planner.py` drops from 939 to 868 lines and `agent.py`
  is unchanged in size. Adding a verb is now a change to a data file.

### Unchanged, deliberately

- The ambiguous-turn guard, which exposes only read-only tools when intent is
  unclear. It stops a question turning into a click. A caller whose tool names
  are dynamic should REGISTER them through `ToolExecutor.permission_overrides`,
  which this package already supports, rather than have the guard relaxed.

## 0.18.0

### Changed

- Initialized `code-mcp-supervisor` sessions now serve an exact
  `flyto_coding_get` from one authenticated tenant/job record without starting
  the whole-history coding worker. The bounded no-follow read reuses canonical
  receipt, continuation, route, and emergency validation and writes no state.
  Terminal Mission reconciliation, unsafe storage, future argument shapes,
  incompatible backend/emergency startup selection, and source drift still use
  the canonical worker.
- `code-mcp-supervisor` now gives only a newly spawned worker's first response a
  separate 120-second deadline so complete fail-closed durable-state validation
  can finish on a high-cardinality state root. State-bearing calls are refused
  before initialization, initialize responses are structurally validated, and
  a hot-reloaded worker must replay the exact client-visible initialize and tool
  catalog contracts. Valid initialize errors pass through without replacing
  the last successful replay contract. After startup, submit/get/audit remain
  capped at 30 seconds and an uncertain request is never retried.
- Cold worker startup no longer loads the complete `flyto_ai.coding` facade or
  reopens the MissionStore once per terminal candidate. Coding and compatible
  top-level exports resolve lazily, while `Agent`/`AgentConfig` remain eager to
  preserve the factory's runtime typing contract. Terminal reconciliation
  resolves all known work-item rows from one fully validated bounded snapshot
  while retaining exact coordinate, lease, fence, claim, and recovery checks.
  Durable formats, the three-tool MCP surface, route, and product repositories
  are unchanged.

- Raised the sibling floors to `flyto-core[browser]>=2.31.1` and
  `flyto-blueprint>=0.3.1`.

  The Core floor was `>=2.31.0`, chosen for the three deterministic
  domain-solver capabilities that release added. 2.31.0 is also the one release
  affected by GHSA-wmwj-g59x-c8px (critical): `verify.spec` reached past the
  module allowlist and the dangerous-permission gate to execute arbitrary host
  commands. A capability floor and a security floor were being satisfied by the
  same number, and the number was the vulnerable one. `tests/test_stack_security_floor.py`
  named it without being told — it derives the minimum from Core's own
  `security/advisories.json` rather than restating it, which is why a floor set
  for one reason could be checked against the other.

  Floors rather than notes, for the reason this package already states about
  every other floor: nothing in `flyto_ai` checks a sibling version at runtime,
  so an environment already holding 2.31.0 satisfies a weaker declaration and
  keeps running.

### Released

- This is the release that makes the module-availability gate reachable.
  `tools/blueprint_tools.py` decides whether to pass host-derived
  `available_module_ids` by inspecting the Blueprint engine signature, and an
  engine without the parameter is called exactly as before — unfiltered. Every
  published `flyto-ai` up to 0.17.0 declared `flyto-blueprint>=0.1.0` and so
  resolved to 0.2.2, which has no such parameter: the gate existed only in a
  source checkout. The floor is what makes a signature-probed feature reach an
  install.

## 2026-08-25

- Fail closed on public multi-repository coding jobs whose sibling edits were
  not covered by the primary workspace verifier, Indexer post-lane, changed-path
  ledger, or implementation revision. Cross-repository delivery now requires
  one governed job per repository plus a final integration audit.
- Fix audit rework for large root scopes: the parent contract retains its
  cumulative ledger, while one amendment sends only newly attributed or
  explicitly authorized repair paths to Indexer. The 32-target amendment bound
  no longer rejects a valid root task with more than 32 paths.

## Unreleased

- Close broad-parent audit rework plan exhaustion without raising a ceiling.
  Same-scope findings retain all exact named parent targets; current Indexer
  successors plan only that active set while their ledger stays cumulative.
  The host accepts either exact active coverage or an exact legacy cumulative
  plan during rolling upgrades and rejects every partial or extra set. The
  unchanged executable bound remains 32, and terminal failed jobs are not
  reopened.
- Allow an explicit strict validation-only job to audit an existing dirty
  candidate without manufacturing a rewrite. The sealed Indexer intent ledger
  must name every adopted path, every path must already be a Git working-tree
  change, and the host binds and re-proves the exact revision around the
  provider and required checks. Clean, unplanned, unsafe, oversized, non-Git,
  or concurrently moving candidates remain non-landable. Refresh the stack
  lock to the verified clean Blueprint, Core, and Indexer main revisions used
  by this route repair.
- Strengthen `flyto_coding_get` without adding a fourth tool: a `job_id`-only
  read retains the exact historical `ok` plus full `job` shape, while explicit
  compact summary reads add detail-scoped change tokens, bounded 20-second
  conditional wait, retry/freshness metadata, and closed next actions. Summary
  reads bypass shared coordination contention through one tenant-bound atomic
  durable snapshot and omit the large audit-only
  result/route/mission/evidence/session projections.
- Stabilize the cross-process execution-session test under loaded CI runners,
  and make the interrupted-abandon recovery test replay the same time-bearing
  closure payload instead of accidentally minting a second payload at a
  wall-clock boundary.
- Prepare package version `0.18.1`; `0.18.0` is already published and cannot
  truthfully identify a wheel containing the coding-route repair below.
- Accept a strongly declared `Exact targets:` or
  `Exact targets/authorized changes:` atomic list as first-round path
  authority for its exact mixture of existing and new files. This closes the
  governance loop for generated references, focused tests, and handoff files
  that were explicitly listed but previously omitted solely because they did
  not yet exist. Generic `Targets:` prose, negative clauses, unsafe paths,
  symlinks, malformed or oversized groups, and partial-prefix recovery remain
  fail-closed.
- Repair first-round projection for `add`/`create`/`write`/generation/
  production/`introduce`/`implement` headings that explicitly declare exact
  new files or paths before a colon. Generic report/evidence headings are
  masked context, not mutation authority. An admitted headed statement is an
  ordered atomic group: negative or mixed polarity,
  command/evidence prose, unsafe or non-file spellings, symlink ancestry, and
  statement or remaining global-capacity overflow reject the complete group
  without ordinary-parser prefix recovery. Numbered exact lists and ordinary
  request parsing are unchanged.
- Allow explicitly authorized exact new-file targets beneath missing parent
  directories when their nearest existing ancestor is a real in-workspace
  directory. Authority remains file-exact and refuses every existing symlink
  component, unsafe spelling, non-file suffix, and non-authoritative mention.
- Fixed strict-route intent-ledger projection for an explicitly declared,
  numbered exact path list whose new-file item carries a description longer
  than the local mutation-verb window. Generic inclusion prose remains
  non-authoritative, and all path, polarity, parent, symlink and count bounds
  remain fail-closed.
- Completed exact multi-repository Codex authority: `working_dir` remains
  `--cd`; other normalized leased roots become ordered exec-level `--add-dir`
  options for initial and resumed rounds, with no derived parent.
- Raised every Core-bearing extra to `flyto-core[browser]>=2.31.0`, the first
  release containing the three declared deterministic solver capabilities,
  while retaining the independent advisory-derived security floor.
- Added the required `domain_solver_closed_loop` coding check and documented
  its compute-only, non-safety boundary.

## 0.18.0

### Boundary gates, second round

- `.github/workflows/advisory-freshness.yml` asks the floor question against
  `flyto-core@main` on a daily schedule. The floor gate reads Core's advisory
  manifest rather than a copied constant, so it cannot go stale the way a number
  would — but CI checks Core out at the revision `stack-lock.json` pins, so the
  manifest it reads is only as fresh as the last lock bump. That gap was not
  hypothetical: the floor was set to 2.28.1 and Core published five further
  advisories, one critical, against `<= 2.28.1` the next day while the pinned
  gate stayed green. A separate scheduled workflow rather than a CI step, so a
  pull request never fails because another repository published an advisory an
  hour earlier.

### Changed

- `flyto-core[browser]` floors at `>=2.29.0`, the version clearing those five.
- `docs/OPERATIONS.md` described the pre-0.16.0 base install: it told operators
  the default install ships Core browser modules, Pro contracts and blueprint
  storage. All four moved to extras at 0.16.0. An operator following it would
  have expected browser automation in a default install and found none.
- `CONTRIBUTING.md` said Python 3.10; `requires-python` is `>=3.11`.

Release ordering matters here: `flyto-blueprint 0.3.0` must be on PyPI before
this version's `v0.18.0` tag is pushed. The `blueprint`, `full` and `dev` extras
require `flyto-blueprint>=0.3.0`, and until that release exists those extras do
not resolve.

### Gates for claims that were never checked

Every finding this release answers had the same shape: the working tree was
correct and something outside it was not. Four checks now hold the boundary
instead of a convention.

- `tests/test_stack_security_floor.py` reads
  `flyto-core/security/advisories.json` from the sibling checkout `stack-lock.json`
  guarantees, derives the lowest Core version clearing every published advisory,
  and fails if any declared `flyto-core` floor predates it — in this repository
  *and* in Blueprint, since this is the only place in the stack where all three
  checkouts exist. The number is read, never restated, so advisory 34 tightens
  the floor without anyone remembering to.
- `scripts/check_release_drift.py` (also a required check in `.flyto/coding.yaml`)
  fails when a tag `v<version>` exists and the packaged source at HEAD differs
  from it. It asks for a correct version number, not a release: an unreleased
  version passes.
- `tests/test_complexity_budget.py` puts a ratchet on module size and parameter
  count. Everything over budget today is recorded in
  `tests/complexity_baseline.json` — 22 files over 800 lines, 23 functions over
  8 parameters — and may only shrink; anything new is held to the budget
  immediately. `scripts/update_complexity_baseline.py` refuses to raise a
  recorded number without `--accept-new`, so the register cannot be quietly
  re-cut around a regression.
- `tests/test_core_constant_parity.py` (from the same round) fails if a borrowed
  Core value stops matching Core or if a fallback copy returns.

### Changed

- Moved the 35 typed service failures and `PLAN_AUTHORITY_CODES` out of
  `coding/service.py` into `coding/errors.py`; `service.py` goes from 8,227 to
  7,774 lines. They are a closed vocabulary callers branch on and they touch no
  service state, so sitting beside a 6,973-line class only made both harder to
  read. Everything is re-exported from `flyto_ai.coding.service`, so no import
  path, error identity, `code`, or inheritance edge changed. The 6,973-line
  class itself is untouched and is now the largest single item in the recorded
  complexity debt.

- Declared this package's product role in a repo-root `flyto-product.toml`
  (`flyto.product-contract.v1`, layer 1 `intent_governance`) with an exact test.
  Blueprint and Core already shipped theirs and both disclaimed "intent and
  provider governance", so the layer that owns it was the one layer no file
  claimed.
- Made the sibling dependency floors exact. `flyto-core[browser]` moves to
  `>=2.28.1`, the first Core release clearing every published advisory; the old
  `>=2.16.1` predated all of them and nothing here checks a Core version.
  `flyto-blueprint` moves to `>=0.3.0`, because `tools/blueprint_tools.py`
  probes the engine signature for `available_module_ids` and no published
  Blueprint had that parameter — the module-availability gate was inert for
  every install resolved from PyPI.
- Removed the duplicated Core browser constants behind the ImportError
  fallbacks. `assistant/resilience.py` carried a hand-written copy of Core's
  snapshot and interact module sets that had drifted (missing
  `browser.detect_list` and `browser.readability`, carrying `browser.extract`,
  missing `browser.hover` and `browser.drag`) with no test referencing it, and
  `tools/core_tools.py` carried a copy of Core's transient/session-dead error
  patterns. Every value in both is about `browser.*` modules, which only Core
  can dispatch, so the fallbacks are now empty and inert rather than a second
  source of truth. `tests/test_core_constant_parity.py` asserts the remaining
  borrowed values match Core and that no copy comes back.
- Stopped the coding watchdog's scheduled polling and refreshed the shared stack
  lock. `FLYTO_CODING_HEARTBEAT` was never published for this repository, so
  every 15-minute run failed `heartbeat_missing`: 400 consecutive red runs in
  the visible window and one incident issue open since 2026-08-13. An alarm that
  never stops firing says the same thing whether the control plane died or was
  never wired, so the `schedule` trigger is commented out and `workflow_dispatch`
  remains the live entry. Validation, incident handling and the untrusted-input
  bounds are unchanged; restoring the trigger belongs in the same change that
  installs the publisher. The stack lock now pins the current Blueprint and Core
  revisions, so the required `stack_lock` check and CI's sibling checkouts agree
  with the repositories they claim to verify against.

- Restored audited rework across the Indexer ledger-version transition. The
  amendment boundary accepts both the historical `task-context.v1` ledger
  label and canonical `intent-ledger.v1`, independently for parent and
  successor, while unknown versions and every digest, identity, path,
  fingerprint, and chain mismatch remain fail-closed. The shared stack lock is
  refreshed to the exact Core and Indexer revisions that implement and verify
  this boundary. The repaired same-job audit rework crossed that boundary but
  then failed closed before provider start with `route_plan_bound_exceeded`:
  36 cumulative steps exceeded the unchanged 32-step ceiling, so no same-job
  completion is claimed.

- Recorded the bounded recovery and Cloud landing evidence. Primary recovery
  job `job_0b90e4cab8e14f5482aec5f6` selected the final implementation with all
  ten governed gates green; holistic Cloud job
  `job_497fc5ee77d948f2b71b26e8` was Codex-accepted; and follow-up job
  `job_4f40e4fcb6e54ea387786fe7` was Codex-accepted with `landable=true`,
  `audit_count=1`, and `rework_count=0`. Cloud PR
  <https://github.com/flytohub/flyto-cloud/pull/231> merged by protected squash
  to `main` commit `ee8c95678c9a18931890a096ea7c04f6a7295ad0` only after all
  remote checks were green, including Playwright (136 total, 113 passed, 23
  existing skips, 0 failed) and Audit Closure. Broad cumulative audited rework
  can still exceed the route-plan ceiling; future repairs should bind active
  scope to current findings rather than raise or bypass that ceiling.

- Closed the runnable Robotics planner entry around the existing governed
  capability router. The loopback server now requires goal-frame normalization
  plus Blueprint/Core discovery before model planning, and routing failure
  stops before the provider. Robotics retains final plan and motion authority.

- Raised the Codex CLI adapter's bounded per-event JSONL ceiling from 1 MiB to
  2 MiB after a valid tool-result frame exceeded the former limit. The
  independent 8 MiB total-stream ceiling is unchanged. Secret-free round
  evidence now distinguishes malformed JSON, malformed event shape,
  per-event overflow, total-stream overflow, and timeout, so a closed provider
  failure is diagnosable without retaining provider content.

- Fixed coding-worker hot reload after Codex CLI adapter changes. The bounded
  service build digest now covers both implementer adapters, so a long-lived
  MCP supervisor cannot keep spawning an old Codex adapter after its source is
  repaired.

- Fixed the Codex CLI completion boundary. Once the adapter has received a
  valid `turn.completed` event, with bounded valid JSONL and no timeout, a
  later non-zero process exit is recorded as teardown evidence instead of
  discarding an attributable implementation that still passes every
  host-owned check. Missing completion, malformed or oversized output,
  timeout, authentication failure, failed checks, and no-change rounds remain
  closed failures.

- Made Indexer rework amendments execute a proof-bound delta instead of the
  cumulative parent-plus-successor plan. Flyto2 AI now recomputes the versioned
  parent/content/entry digests and ids, validates complete chain linkage, the
  normalized host project, profile/context fingerprint mirrors, intent and
  objective continuity, canonical path spelling, the authenticated
  parent/audited-prior/explicit-target union, exact path partition and
  resolved-target coverage,
  reuses only an exact completed parent-step multiset occurrence, reruns every
  successor gate, and keeps each delta within the unchanged plan limit. The
  pinned verifier's effective compatibility ceiling is generation seven;
  generation eight fails closed. Both Indexer envelope failures and ordinary
  `pass=false` validations retain only exact host-registered machine codes
  without leaking unknown tokens, prose, paths or secrets. Numeric-only
  milestone suffixes no longer become new-file authority,
  while typed `.7z` targets remain valid. The pure amendment contract is isolated
  in a stdlib-only coding-domain module rather than expanding route orchestration.

- Added proof-bound recovery for audit rework routes that fail in Indexer or
  Blueprint before provider start. The existing submit tool accepts explicit
  `retry_rework_route` without adding a tool; ordinary replay remains read-only,
  same-session/continuation authority is preserved exactly, recalled mission
  children are status-validated, and repeated publication loss closes
  action-free while releasing claims and resume state.

- Split host release into two proof-bounded valves. `code-release
  --abandon-job` may now retire one audit-ready or kernel-closed queued orphan
  while unrelated coding services remain alive; it still requires the target
  state, the exact job lease, and the MissionStore closed disposition before
  writing. `--repair-workspace` remains globally exclusive. Both modes bind no
  startup authority, construct no implementer, preserve `authority.json`, and
  expose no additional MCP tool.

- Fixed the mandatory Indexer pre-work lane for isolated worktrees. Canonical
  and translated plan `impact`, `structure`, and `call_hierarchy` calls now
  inherit the host-owned workspace project just like search. A conflicting
  project in returned plan evidence is rejected before analysis or editing.
- Added a provider-neutral trusted connector seam to governed execution-session
  admission. A host transfers a pre-established one-shot process handle whose
  trusted async callback receives only a detached bounded prepared result in
  memory and executes under the existing durable Scheduler
  one-shot fence. Exact content-free success yields a connected receipt with
  Scheduler evidence; absent, failed, malformed, exceptional, interrupted, and
  unknown outcomes fail closed without persisting connector identity or prose.
  Connector await uses one absolute monotonic activation deadline and recomputes
  remaining time at actual executor entry; zero remaining makes no callback.
  Admission performs no synchronous process start. Worker readiness,
  nonblocking request transfer, child-side entry, and result receive share the
  same deadline. Timeout, owner cancellation, validation failure, exception,
  and normal return forcibly terminate the non-daemon worker within the fixed
  closure grace and confirm zero live connector work before returning or closing
  a stable receipt. Duplicate reconciliation uses the same deadline
  plus that fixed 0.5-second grace and sleeps between passes.
  An already-waiting duplicate reconciles a cancelled owner's fenced occurrence
  to unknown without hanging or invoking another connector. Duplicate and
  restart admission returns the persisted content-free result and never
  automatically replays an unknown external outcome. No Cloud consumer or device
  execution is claimed.

- Repaired the phase-one Capability Card boundary: empty or missing source
  references remain incomplete, host verification is explicit and mandatory,
  and search projection now recomputes the canonical claim digest and rejects
  any tampered claim-derived, identity, lifecycle, trust, version, or schema
  field before emitting its secret-free exact allowlist.
- Hardened Capability claim/Card Mapping ingestion by validating a single
  detached bounded snapshot. Hostile iterators, item views, getters, duplicate
  keys, and malformed entries now fail with stable content-free catalog errors.

- Fixed cross-connection coding MCP rework ownership. The supervisor that
  successfully records an explicit audit `rework` now pins the returned
  matching non-terminal rework job through source drift, even when another
  connection submitted it. Foreign reads, accept/error/terminal observations,
  malformed or mismatched receipts, and response state alone remain non-owning.
  Unknown tools have no observation authority and cannot clear pins with a
  terminal-shaped receipt; matching terminal get/audit observation or durable
  state releases an existing pin for safe hot reload. The public inventory
  remains exactly submit, get, and audit.

- Fixed unified coding-route rework planning so command syntax, check output,
  and evidence references no longer grant new edit authority merely by naming
  an existing path. Explicit mutation instructions still widen an amendment,
  prior host-proven targets are always retained, and first-round target parsing
  remains compatible.
  Tracked outputs explicitly requested for regeneration are recognized, while
  programs named after execution connectors such as `using`, `via`, `use`,
  `call`, `calling`, `run`, `execute`, `invoke`, `with`, `through`, or bare
  `by` remain excluded unless they are themselves explicitly modified.
  `by running` / `by executing` / `by invoking` remain execution-only, while
  `by modifying` / `by editing` / `by updating` / `by changing` explicitly
  authorize the program. Inclusion and evidence language alone do not widen
  scope.

- Added a domain-neutral governed Execution Session validation bridge. It
  fail-closes exact Space/activation/goal-frame claims, accepts no observed
  wake-word claim, takes identity and all routing
  ceilings from a frozen verified host authority, and returns detached bounded
  JSON planning input, a capability route, a hashed principal reference, and
  request/authority/route attestations plus a self-excluding overall digest that
  covers every governed result payload field. It does not implement microphone,
  wake detection, STT, identity provider, Cloud, robotics, device, execution, or
  scheduler runtime behavior, and activation grants no identity or permission.
  Activation sources match the v1 ingress/storage contract exactly:
  `typed`, `voice_reviewed`, `external_agent`, and `mission_card`. All four
  require an exact-null observed wake word; reviewed voice claims only upstream
  review, while raw `voice`, `button`, and unknown sources fail closed.
  Unnamed, voice-disabled Spaces preserve an empty `display_name` and empty
  `wake_words`; whitespace-only display names fail closed, while ordinary
  non-empty display text is preserved without trimming or collapsing. Supplied
  metadata remains bounded and text-safe and never
  becomes identity. The exact `active_timeout_ms` field accepts 1..300000 and
  bounds the activation window directly in integer milliseconds, with no
  seconds alias or conversion.
  Request and result contracts have distinct versions; active expiry is
  exclusive, the configured timeout is capped at 300000 milliseconds, and control,
  format/bidi, zero-width, and surrogate text is rejected.
  Canonical JSON admission now has explicit depth, node, UTF-8 byte, integer,
  and timestamp ceilings before recursion or attestation encoding; route limits
  also reject booleans, coercible strings, and values outside 1..32.

- Added an explicit `codex` implementer to the audited `code-mcp` / `code-serve`
  route. It launches a separate non-interactive Codex CLI session from the
  startup-pinned executable and required model, uses the existing ChatGPT login
  without reading a provider API key, ignores personal configuration and
  exec-policy rules, scrubs ambient provider/CI credentials, and loads no MCP,
  plugin, web-search, browser, computer-control, or audit authority. Structured
  JSONL binds the exact provider session; typed rework resumes only that id.
  Workspace changes still come from host snapshots, all source-controlled
  checks and route lanes still run, and success still stops at independent
  exact-revision Codex audit. There is no per-job backend switch or fallback.

- Added `flyto-ai code-watchdog`, a host-only, non-AI observer for the coding
  control plane. It reads the same bounded projections as `code-status` and
  `code-task-window`, never invokes a model, and has no path to submit, audit,
  abandon, repair, commit, or push. It records aggregate health to
  `~/.flyto/health/coding/`: `latest.json` every run, `history.jsonl` only on a
  transition (size-rotated), and `github.json` as the heartbeat cursor. Records
  carry counts, stable reason codes, the reader build digest, and timestamps —
  never prompts, paths, job or session identifiers, evidence, or credentials.
  An executing task with no provably live owner is `critical` after the orphan
  grace; stalled execution, overdue Codex audit, a stale rolling build, a
  failing status recorder, and emergency spillway are `degraded`. An idle host
  with no Codex process stays `healthy`.
  `--install` / `--uninstall` manage a per-state-root macOS LaunchAgent, and an
  optional `--github-repository` publishes a secret-free heartbeat to a GitHub
  Actions repository variable through the already-authenticated `gh` CLI.
  `.github/workflows/coding-watchdog.yml` is the remote dead-man switch: a
  deterministic scheduled job (no agentic workflow, no model quota) that opens
  or refreshes one labelled issue when the heartbeat is stale or unhealthy and
  closes it on recovery. This first release is alert-only.

- Hardened the coding watchdog before its first release.
  - The dead-man workflow now treats `FLYTO_CODING_HEARTBEAT` as untrusted
    input: bounded raw size, exact schema, plain in-range integer
    `observed_at`, an allowlisted `health` level, and an allowlisted `reason`
    re-checked before it is written to `GITHUB_OUTPUT`. A newline in a
    rendered field could previously have forged `healthy=true` and silenced
    the switch. Each rejection now has its own reason code.
  - `state_readable` judges the status index by the publisher's authoritative
    `MAX_STATUS_INDEX_BYTES` instead of the watchdog's smaller record limit,
    which had reported a large but valid index as a route failure.
  - The state root and the health directory must now be disjoint
    (`watchdog_paths_overlap`), so the observer can neither write into the
    coding-service tree nor observe its own writes. Both roots, and the state
    root that derives the LaunchAgent label, are resolved through their
    symlinks first: a lexical comparison let `--health-dir` reach inside the
    state root through a link, and let `--install` and `--uninstall` compute
    two different labels for one state root.
  - `--install` now validates every value it bakes into the LaunchAgent —
    thresholds, heartbeat interval, repository, and variable name — against
    the bounds the observing run applies, so an install can no longer produce
    an agent that fails on every unattended wake.
  - The remote heartbeat is one bounded `gh variable set` upsert instead of a
    PATCH that parsed its own 404 message to decide whether to POST, and an
    oversized projection fails locally rather than being truncated by GitHub.
  - Every health record the watchdog opens by name is now opened `O_NOFOLLOW`,
    and rotation tests the name rather than its target. `--health-dir` may
    legitimately sit under a world-writable parent, where a planted symlink at
    `history.jsonl` had turned the `O_APPEND` write into a write primitive
    against any file the watchdog's user owns, a planted `latest.json` chose
    what the watchdog read back as its own previous state, and a planted
    `watchdog.lock` placed the exclusive lock somewhere no second watchdog
    would look. A refused history append now reports
    `watchdog_history_unwritable` after `latest.json` is already durable.
  - A failure to record the heartbeat cursor no longer discards the whole turn.
    `mark_github_sent` raising `OSError` after the heartbeat was published had
    left the remote switch reading `healthy` while the local record was never
    written; it is now a `github_heartbeat` warning
    (`github_state_unrecordable`) and the turn still records health.
  - The dead-man workflow no longer cancels a run in progress. Its product is
    an incident, not an artefact, so a dispatch arriving between "the heartbeat
    is stale" and "open the issue" must not cancel the only step that reports
    it.

- Added a host-only adapter for flyto-core extension management:
  `list_core_extensions`, `list_core_extension_kinds`,
  `install_core_extension`, and `uninstall_core_extension` in
  `flyto_ai.tools.core_tools`, bound to `core.plugin.loader`. It is generic over
  the extension kinds Core declares, preserves Core's normalized extension names
  and result codes, and returns one fixed envelope — `code`, `name`, `version`,
  `previous_version`, `install_enabled`, `restart_required`, `rolled_back`,
  `refresh_failed` — that never contains installer (pip) output. Install accepts
  an optional pinned `version` and an `upgrade` flag; listing accepts a kind
  filter, which the host applies itself because Core's `list_extensions` has no
  such parameter.
  Operators must set `FLYTO_EXTENSIONS_INSTALL_ENABLED` to allow install or
  uninstall; listing needs no opt-in. Installation is not exposed to models:
  install-shaped Core tool names are withheld from the MCP tool catalog and
  refused at dispatch.

- Host-global workspace ownership is now demand-scoped to durable non-terminal
  work instead of the lifetime of an MCP worker. An idle worker therefore no
  longer blocks every Codex using another state root after its last job has
  settled. Admission acquires the configured-tree authority before its first
  durable job mutation; restart reacquires it before reconciling open work;
  running, queued, rework, and audit-pending jobs retain it. The last terminal
  transition releases it, and a bounded per-process observer also releases an
  admitting worker's lease when a compatible peer on the same shared state
  root performs that terminal write. Overlapping foreign state roots, crashed
  open work, the subtractive release valve, and the three-tool MCP surface stay
  fail-closed.

- Installed-capability discovery from `flyto-core` now validates the manifest
  Core actually emits. The host read a shape Core never produces — it expected
  `manifest_contract` / `manifest_hash` instead of `schema` / `hash`, `modules`
  as records instead of module-id strings, `capabilities` keyed by
  `capability_id` instead of `capability` plus `providers`, and `plugins` keyed
  by `plugin` instead of `id` / `version` / `module_count`. Every real manifest
  was therefore rejected and a fully installed Core reported zero modules. The
  digest is still recomputed by Core's own function and all three declared
  counts must match exactly. Blueprint now receives installed **module ids
  only**, under `available_module_ids`; capability ids and plugin ids are
  validated and counted as provenance but never handed to an engine, because
  neither is executable. An absent or too-old Core still returns `None` so
  callers leave their filtering alone, and a malformed or failing manifest from
  a manifest-capable Core still returns an empty set rather than an unfiltered
  one. `installed_capabilities` provenance drops `declared_capability_count`;
  `capability_count` is now exactly what Core declared. Rollback: revert
  `flyto_ai/tools/core_tools.py` and `flyto_ai/tools/blueprint_tools.py`
  together — they are one contract.

- `flyto-ai code-status` no longer reports a dead coding instance as alive
  after its process id is reused by an unrelated process. Each service instance
  now holds a crash-released `flock` lease on its own status lease file, and
  liveness is read from that lease instead of from the recorded pid. An
  instance recorded as `closed` is never reported alive. Where `flock` is
  unavailable, liveness reads as unknown rather than alive. Rollback: revert
  `flyto_ai/coding/route_status.py` and the `acquire_lease` / `release_lease`
  calls in `flyto_ai/coding/service.py`; stale `.lease` files are inert and are
  collected with their status files.

- A coding state root that refuses this build's authority now reaches the MCP
  client as a bounded, actionable reason naming `code-status` and
  `code-release`, instead of the generic `-32603 coding worker unavailable`
  that hid it. `code-mcp` exits `78` for that class of refusal and prints only
  its stable error code, never the underlying message, which can contain a
  state-root path. Worker stderr is still never captured or forwarded, and the
  public MCP inventory is unchanged. Rollback: revert the exit-code branch in
  `_cmd_code_mcp` and `_unavailable_reason` in
  `flyto_ai/coding/mcp_supervisor.py`.

- The audited Claude service adapter now uses an explicit finite 8 MiB ceiling
  for one inbound Agent SDK JSON frame. This prevents a legitimate large
  host-declared Indexer result from hitting the SDK's 1 MiB default and
  stranding an attributable implementation session. The change is service-only
  and does not widen tools or route authority; legacy direct calls retain the
  SDK default.

- Audit-required Claude service jobs that set `require_changes=true` no longer
  accept a prose-only provider turn as the end of implementation. When a turn
  produces no attributable `Edit` / `Write` (or bounded project-action)
  evidence, the agent resumes the exact SDK session with a fixed host-authored
  correction while the existing attempt budget remains. Exhaustion returns a
  stable failure and skips the known-useless required-check run. Jobs that
  explicitly allow no change and legacy direct callers keep their one-turn
  behavior; the service adapter carries the host-owned invariant explicitly.

- `flyto-ai code-release` can now retire an orphaned `awaiting_codex_audit` job
  that a *different* startup authority recorded — the case it was built for and
  could not reach. It previously constructed an ordinary service, which bound
  the state root to the operator's own authority; a job left by a strict
  Claude route therefore forced a rotation, and rotation requires every job
  terminal, including the one being retired. The command now opens the root
  through `CodingService.open_host_release_valve`, which takes the authority
  lease exclusively (refusing with `service_busy` while any live coding service
  holds it) and never reads, writes, rotates, or reproduces `authority.json`.
  The marker is byte-for-byte unchanged, other open jobs under that authority
  are untouched, and only the target worktree is released. The valve constructs
  no implementer, publishes no status, reconciles nothing, and refuses
  `submit`, `audit`, and the dispatch pump with `release_valve_refused`. The
  public MCP inventory remains exactly three tools.

- The supported Python floor is now 3.11, matching the SQLite
  `serialize()` / `deserialize()` primitive used by the pathname-free mission
  authority envelope. CI covers Python 3.11 and 3.12, and the development
  extra now includes the Claude SDK imported by the complete route suite.
- The mandatory Indexer lane now uses one shared ten-minute transport bound in
  both the agent-stack preset and the public `code-mcp` / `code-serve` route.
  Large-workspace and concurrent-release strict verification can finish instead
  of being killed by a CLI-only 30-second deadline; every gate remains
  mandatory and genuine timeouts still fail closed as `capability_timeout`.
- The pre-work route now accepts either published Indexer gate family selected
  by the exact plan: legacy `assess` / `implement`, or current `plan_changes` /
  `apply_changes`. Unknown, duplicate, and mixed-family plans still fail before
  any implementation begins. Post validation also understands the current
  `overall=pass` envelope only when its ruff and pytest statuses explicitly
  agree; incomplete or contradictory evidence remains fail-closed.
- Blueprint discovery now ranks ordered phrase overlap before catalogue order,
  so direction-bearing reuse such as CSV-to-JSON no longer selects the reverse
  JSON-to-CSV transform when both candidates share the same token set.
- Every coding job now serves a mission in the durable multi-process mission
  kernel, which is the authority for queue order, repair preference, dependency
  readiness, worktree exclusion and fencing. A caller that names no mission gets
  one synthesized by the coding adapter; the kernel stays workload-neutral.
  Receipts carry only the bounded, secret-free mission projection - no prose, no
  coordinates, no evidence values, no worker identity, no paths. The public MCP
  tool inventory is unchanged.
- A coding state root is now bound to one semantic startup authority by a
  durable, crash-released `flock` lease plus a bounded marker. Services that
  share an authority coexist as peers on one queue; a service that would build a
  different implementer - or run under a different audit requirement, contract
  path, sandbox, approval policy, host lane policy or rework ceiling - fails
  construction with `execution_authority_conflict` before it can reconcile
  status, sweep a workspace claim, or dispatch anything. Rotation is permitted
  once no old service is live and every job is terminal. **Operators running two
  differently configured coding services against one state root must give them
  separate roots.**
- The job lease now covers execution only. It is released once a job's durable
  artifacts exist and before any pump can dispatch, so any compatible worker can
  run the store-selected round and queued work survives its submitter exiting or
  restarting. Queued and rework-queued records are pumped on restart instead of
  being failed as `service_restarted`; only genuinely interrupted running work is
  failed closed, and only after proving no live lease holds it.
- Jobs recorded before the executing-authority fingerprint existed are adopted
  only on proof of no execution: queued work that never reached an implementer
  is migrated and runs normally. An executing record is never adopted - the
  service refuses to start beside a live round it cannot attribute, and settles
  one whose lease is provably free as `execution_authority_unbound` with its
  mission item and worktree claim accounted. An unfingerprinted awaiting-audit
  job may still be accepted by an auditor but may not be reworked, because a new
  round would adopt a route policy it never named.
- A refused start-up now changes nothing: marker and job validation happen
  before the authority marker is written, so a refusal leaves a present marker
  byte-identical and never creates a missing one. The marker is read through a
  single `O_NOFOLLOW`/`O_CLOEXEC` descriptor under a small byte bound and is
  never re-opened by name after a check, so one that is damaged, unparseable,
  symlinked, oversized or not a regular file is a refusal rather than an
  absence; an unreadable job record refuses both start-up and rotation.
- Service teardown now releases both root descriptors under one outer
  `finally`, so a failure draining the executor or releasing a job lease can no
  longer leave a stopped service holding the state root against its successor.
- **A host with no inter-process lock now refuses to start** with
  `execution_authority_unavailable` instead of silently running without the
  isolation it advertises.
- `CodingAuthorityConflict` and `CodingAuthorityUnavailable` are exported from
  `flyto_ai.coding`.

- A successful audit rework no longer has to manufacture a second diff when
  the earlier attributable revision is already correct. The service promotes
  only the host-generated `no_changes` outcome with passing required checks
  after re-proving the same implementation session, tenant/job workspace
  claim, sealed resume envelope, cumulative file set, and live content digest;
  the Indexer post lane then validates that cumulative set normally.
- Guardian now permits edits to the exact repository dotfiles already present
  in its closed allowlist (`.gitignore`, `.dockerignore`, `.editorconfig`).
  Arbitrary dotfiles and all sensitive-path matches remain denied.
- Claude implementation rounds now default to the already-enforced 100-turn
  ceiling. This prevents a complete workspace edit from being discarded as
  `turn_limit_exceeded` merely because an older supervisor started with the
  30-turn default. Cost, tools, workspace confinement, required checks,
  host-owned lanes, exact-revision audit, and rework limits are unchanged.
- One worktree is now owned by one audited coding job for the whole job, not
  for one implementation round. Previously the cross-process workspace lock was
  released when a round finished, so between `awaiting_codex_audit` and the
  Codex verdict another concurrent Codex frontend could edit the same tree; the
  first job's exact-revision audit then failed live recomputation and its work
  was stranded, non-landable and unreworkable. An audit-required job now takes a
  durable `flyto.coding-workspace-claim.v1` claim at submit and holds it across
  `awaiting_codex_audit` and every rework round. A second frontend targeting the
  same worktree fails fast with `workspace_busy` plus the owning job id in
  bounded MCP structured error details, instead of silently invalidating an
  audit. Claims are keyed by workspace digest, so jobs in different repositories
  still run in parallel.
- A workspace claim that cannot be evaluated now fails closed. Corrupt JSON, an
  unknown version, a missing or extra or out-of-range field, a
  `workspace_sha256` that does not match the tree being queried, an unreadable
  file, a claim naming a job with no record, or an owner record that does not
  bind back to that same job and canonical worktree all report
  `workspace_claim_unresolved`. Missing fields fail closed exactly like unknown
  ones: a half-written claim proves nothing about ownership. None of these are
  ever removed automatically, including by startup reconciliation — discarding
  them would turn "ownership is unknown" into "nobody owns this tree". Startup
  sweeps remove a claim only when it is fully bound and its owning record
  proves the job settled.
- Only `submit` may create a workspace claim, and only once, before its job
  record is published. Every later claim-owned transition and both audit
  verdicts reassert a claim this exact tenant, job, and worktree already hold;
  a vanished claim for a live job is `workspace_claim_unresolved` rather than
  something to reacquire. Absence is not proof of uninterrupted ownership —
  another Codex could have taken the worktree during the gap, edited files
  outside this job's attributable set, settled, and released, and recomputing
  only this job's files would never see it. A claim carrying the same job id
  under a different tenant is never taken over. `code-release --abandon-job`
  and `--repair-workspace` remain the only ways out, and neither is an MCP
  tool.
- Claim-owned state transitions are now a gate rather than a log line.
  Ownership is asserted before the record is published, inside the same
  cross-process state guard, so a round can no longer enter `running` or
  `awaiting_codex_audit` without a valid exclusive claim — a claim that was
  stolen, corrupted, or could not be written settles the job `failed` instead
  of opening an unclaimed audit gap. A refused rework hands its execution lease
  back, leaving the job exactly as auditable as before. Release still removes
  only this job's own fully bound claim.
- The supervisor now releases a worker on any unrecoverable exchange, not only
  on a deadline. A broken pipe, malformed frame, or oversized response leaves a
  desynchronized stream whose next read could answer the wrong caller, so the
  worker is terminated and the uncertain request is still never retried. The
  post-kill reap is bounded like every other wait, and pipe cleanup no longer
  leaves a reader thread alive if its first bounded join times out.
- `code-mcp-supervisor` now falls back to the documented `--state-dir` default
  when the flag is omitted. It previously read the absent flag as an empty
  string and silently disabled durable active-job reconciliation.
- Rework can now be sent from any live worker and still continues the exact
  prior Claude session. Resume context was process-local, so an audit arriving
  at a different `code-mcp` process — the normal case when each Codex
  conversation runs its own stdio worker — could never rework. A bounded,
  redacted, mode-0600 `flyto.coding-resume-envelope.v1` record stores only the
  public request fields plus job, request-digest, and session bindings. It
  loads only when its `session_bound` equals the record's
  `implementation_session_id`, and always rebuilds the request with
  `resume=true` against that same id, so it can continue a session but never
  start a fresh one. Startup authority — approval policy, sandbox mode, config
  path, sandbox image, checks, capabilities — is never persisted and is
  re-imposed from the running process. A missing or mis-bound envelope still
  fails closed with `rework_not_resumable` and consumes no audit round.
- `code-mcp-supervisor` can no longer hang a Codex frontend. Every request and
  replayed-handshake read is deadlined at 30 seconds through a portable reader
  thread and queue rather than an unbounded `readline`. A missed deadline
  returns a bounded JSON-RPC `-32603`, terminates the wedged worker so the
  state-root locks it held are released, and never retries the request, whose
  delivery is uncertain; the caller recovers by replaying the same idempotency
  key and the next request starts a fresh worker that reconciles interrupted
  jobs truthfully.
- Hot-reload tracking self-heals from durable job records. A client that stopped
  polling used to leave a stale in-memory entry that blocked every later
  submission with `service_reload_pending` for the life of the frontend. Tracked
  job ids are now reconciled against their bounded per-job records, so a
  genuinely non-terminal job still preserves its worker and refuses only new
  submissions, while a settled one releases the reload without restarting Codex.
- Added the host-owned `flyto-ai code-release` command. `--abandon-job` moves
  only an `awaiting_codex_audit` job to `failed`/`job_abandoned` with
  `landable: false`; `--repair-workspace` clears an unresolved claim and refuses
  while a live job owns the tree. Both are strictly subtractive and neither is
  an MCP tool — the public audited inventory remains exactly
  `flyto_coding_submit`, `flyto_coding_get`, and `flyto_coding_audit`, and the
  implementer still never receives the audit tool. Implementer selection is
  unchanged: Claude remains pinned to `claude-opus-5` with no fallback.
- A legacy non-audited coding service keeps its previous behaviour. It has no
  audit gap, so it takes no job-lifetime claim and its rounds remain serialized
  by the per-round workspace lock — but it now honours a claim held by an
  audited job, so it can never edit a worktree mid-audit.

- Repaired the strict coding route for ordinary tasks. Every host-owned
  Indexer search — initial discovery, gate remediation, and translated plan
  steps — is now scoped to the workspace project. An unscoped smart search
  fanned out over every indexed project and exceeded the 30-second capability
  bound, so the mandatory pre-work lane failed before the configured
  implementer ever started. The capability bound is unchanged.
- Failed route lanes keep their evidence. A failed lane receipt now retains
  every completed call plus one failed call naming the exact host-derived
  action (`structure`, `search`, `task.plan`, `task.gate.<phase>`,
  `task.validate`, `verify.strict`), bounded by the configured per-lane call
  limit. A transport timeout is classified `capability_timeout` from a closed
  machine code rather than collapsed into a generic `domain_failure`, and a
  capability that fails to launch names the lane whose provider was actually
  unavailable instead of always blaming `indexer_pre`.
- Added `flyto.coding-route-status.v1`: bounded durable runtime status under
  the service state root. Each service instance owns
  `status/instance-<id>.json` and shares a validated, byte-bounded
  `status/index.json`, so concurrent `code-mcp` processes never overwrite one
  another. Records carry an opaque instance id, an immutable build digest of
  the loaded coding sources, process id, start time, lifecycle, job state,
  route lane/action, stable failure code, implementer-start truth, and bounded
  session/revision ids — and no message, path, error text, file list,
  environment, or credential. Retention is bounded and stale instances are
  collected deterministically.
- Added `flyto-ai code-status --state-dir <dir> [--json]`, a read-only
  inspection command that lists coexisting service instances with their build
  ids, liveness, and staleness. It states explicitly that processes started
  before this schema publish no row and cannot appear retroactively.
- `implementer_started` is now recorded in the durable job record immediately
  before every real implementer invocation, never inferred from `running`, and
  is exposed additively on the public job receipt. A round that fails after
  implementation keeps its session id, attributable files, and revision digest
  as proof the model ran, while staying terminal and non-landable.
- Added the startup-only `flyto.coding-emergency.v1` overflow lane for a
  provably unreachable route infrastructure, enabled with
  `--emergency-overflow-backend` (which must equal `--implementation-backend`)
  and `--emergency-overflow-threshold` on `code-mcp` and `code-serve`. It
  triggers only for a positively classified `capability_unavailable` /
  `capability_timeout` failure in a pre-implementer lane with no recorded
  implementer start and no attributable edit; a domain refusal, gate denial,
  stale index, malformed evidence, failed check, failed implementation, Core
  failure, Indexer post failure, audit rejection, or rework exhaustion never
  opens it. Emergency rounds call the same startup-selected implementer, keep
  the required source-controlled checks and exact-revision binding, and still
  require an independent Codex audit through a separate digest-validated
  authority receipt sealed to that job, request, session, and revision.
  `CodingRouteReceipt(strict=True)` is unchanged and a failed strict route
  never becomes landable.

- `code-mcp` processes may now share one durable coding state root, so Codex
  can create or resume multiple conversations without the second MCP process
  exiting during `initialize`. Short cross-process state guards preserve
  idempotent submission and atomic records; per-job crash-released leases stop
  duplicate execution and prevent a new process from misclassifying another
  live process's job as interrupted; per-workspace locks still serialize edits.
- The public `code-mcp` / `code-serve` coding service is now a true single
  entry: the new `flyto.coding-route.v1` contract runs host-owned lanes around
  whichever implementer startup selected. The Indexer gate is mandatory before
  any model edit and again after the source-controlled checks; Blueprint reuse
  discovery is a mandatory read-only lane whose outcome is conditional on
  real relevance; Core validation is always enabled on the strict route and
  flows through `flyto_ai.tools.core_tools` with a validation-only allowlist.
  Plan steps run in order through an allowlist under bounded step, response,
  iteration, and gate-remediation limits, and no lane outcome is taken from
  model prose. A missing catalog, failed domain result, malformed evidence,
  incomplete gate, exceeded bound, or unavailable Indexer fails the round
  closed instead of reaching `awaiting_codex_audit`.
- Added an additive secret-free `route_receipt` to the public job receipt
  recording which lane was required, applied, skipped, not applicable, or
  failed, which calls and gates ran, and a content digest. It is validated on
  construction and revalidated on deserialization, and only a strict route
  that succeeded can appear on a landable receipt. A strict service also
  revalidates persisted route evidence when it reads an audit-ready,
  reworking, or accepted job back, including after a restart, so a record
  whose proof was removed or edited fails closed instead of reading as
  landable. Lane success is read from the producing tool's own field with
  fail-closed precedence: a present `pass` or `valid` is authoritative and a
  fallback field can never rescue it.
- Added `--indexer-command` and `--blueprint-command` startup options to both
  public coding commands. They replace a lane's startup command only; no flag
  detaches a lane, and Core validation is always enabled on the strict public
  route. Direct library `CodingService` construction is unchanged and still
  runs no route, so it stays compatible but is not the public audited route.

- Added judge-drawn Mission Station card interpretation with an immutable
  evidence boundary, APPROVED-capability ceiling, strict hostile-output
  validation, deterministic provider fallback, and content-addressed
  attestation. Flyto2 AI does not draw cards, bind resources, authorize motion,
  or decide Task completion.
- Split `flyto-core[browser]`, `flyto-pro-core`, `flyto-blueprint`, and
  `anthropic` out of the unconditional base `dependencies` into `browser`,
  `pro`, `blueprint`, and `anthropic` extras (`full` restores all four
  together). None of them was ever imported at module import time — every
  call site already lazily imports and try/excepts `ImportError` around
  them — so the split changes nothing at runtime for a caller that already
  has them installed. What it fixes is install-time: `flyto-core[browser]`
  alone pulls Playwright plus a Chrome download, which made `pip install
  flyto-ai` impossible inside a slim image that only needs e.g.
  `OpenAIProvider`. Released as 0.16.0, not a patch, because this changes
  what a bare `pip install flyto-ai` gives you: existing consumers of the
  full agent stack (flyto-cloud's worker, the desktop build) must move to
  `flyto-ai[full]` to keep the same install shape.

- `OpenAIProvider.complete_json_schema` drives OpenAI strict structured
  outputs, so a caller that needs a shape gets it enforced upstream rather
  than parsing whatever came back; refusals and truncated replies are
  reported as such instead of being returned as content that fails to parse.
  Released as 0.15.0 rather than a patch because consumers branch on the
  method's presence: flyto-cloud's space planner calls it and checks for it
  by name, since an adapter without it degrades silently to rule-based
  planning, which still produces a plan and therefore looks healthy.
- Added atomic capability execution policy, redacted content-addressed trace,
  fixed-snapshot authority-bounded replay/Blueprint feedback, reusable
  evidence-bound adapter conformance, and a bounded domain-neutral scenario
  matrix. Manager dispatch now enforces the policy/result gates and records
  outcomes while Agent outer denials enter the same deeply immutable trace.
  Conformance defaults to read-only authority and verifies expected dispatch
  state; approval callbacks and outcome sinks have bounded waits and stable
  failure projection.
  Clean-runner CI now installs a pinned sibling Blueprint benchmark fixture
  before the full Python 3.11/3.12 suite.
  It also provisions ripgrep and a digest-pinned Python command sandbox so
  portable search and isolation checks run against real dependencies.
  Protected-file Docker mounts now fail reads consistently across Linux and
  macOS instead of exposing platform-specific `/dev/null` success semantics.
  Added deterministic `Agent` async lifecycle cleanup for SQLite memory and
  transcripts, plus Python 3.12-safe legacy event-loop test isolation.
  CI now fails on deprecation or unhandled-thread warnings and tests actual
  sandbox readiness rather than executable presence.
  Hardened real MCP transport for
  concurrent out-of-order responses, cancellation, timeouts, child crashes,
  malformed/oversized JSON-RPC, sustained stderr, and strict catalog schemas;
  CI now runs the complete suite on Python 3.11 and 3.12.
- Split the policy-bearing agent stack into atomic manifest, preset, probe,
  MCP transport, catalog, session, transactional registry, and runtime
  permission modules behind the existing `stack` and `capabilities` facades.
  Added fail-closed provider-name collision rollback, pluggable monotonic
  argument-risk resolvers, deterministic subprocess cleanup, pure boundary
  tests, real MCP subprocess tests, and full Agent/Manager bypass coverage.
- Added backward-compatible `flyto.agent-stack.v2` profiles with exhaustive
  per-tool read-only, workspace-write, or danger-full classification. The host
  runtime ceiling is enforced inside `CapabilityManager` and again by `Agent`;
  direct manager dispatch cannot bypass policy, and Core module arguments still
  escalate shell/process/container/network/filesystem/Git work to danger-full.
- Generalized `flyto.agent-stack.v1` from a coding-only preset into a bounded,
  source-controlled domain profile. Arbitrary `CapabilitySpec` groups can now
  be composed and preflighted with configuration and runtime fingerprints;
  invalid schema, duplicate names, workspace escape, and unscoped MCP catalogs
  fail closed. The existing four Flyto2 lanes remain the default preset.
- Made `CapabilityManager` implement the generic Agent `ToolExecutor` contract
  and documented the shared route → authority → plan → execute → verify →
  evidence loop across general workflows, coding, robotics, and explicitly
  authorized penetration/red-team campaigns.
- Added the versioned `flyto.agent-stack.v1` composition for independently
  detachable Indexer, Blueprint, page-inspection, and Core MCP lanes, including
  real handshake preflight and a content-addressed composition fingerprint.
- Added backward-compatible per-capability `allowed_tools` enforcement so a
  shared MCP server can expose isolated least-privilege Blueprint and
  `inspect_page` views; unlisted tools are invisible and undispatchable, while
  missing allowlisted tools fail before model-directed edits.
- Corrected the documented Indexer and Core MCP startup commands to their real
  Python modules and documented the complete understand → reuse/plan → inspect
  → execute → verify → evidence/learning Agent line.
- Made page inspection portable across bundled Chromium and installed Chrome,
  with a typed channel selector and selected-channel evidence; nested MCP
  domain failures now propagate instead of appearing as transport success.
- Changed the Ollama agent transport to the native `/api/chat` tool loop so
  local thinking models honor an explicit bounded `think` setting (disabled by
  default), preserve tool-result ordering and token counters, and cannot spend
  their completion budget on reasoning hidden by the OpenAI-compatible route.
- Clarified and hardened `coding_search` as a literal fixed-string contract;
  results now identify the query mode and direct agents to read the current
  file after an empty search instead of repeating regex-like guesses.
- Verified the production native coding loop with 101 distinct no-mock local
  Ollama workspaces: 99/101 overall, 34/34 standard, 32/34 intermediate, and
  33/33 advanced, with real subprocess checks and zero hidden retries. Both
  failures remain in the content-addressed evidence report.
- Added the provider-neutral `flyto.coding.v1` native coding control plane with
  versioned contracts, workspace-confined argv-only tools, persistent resumable
  threads, append-only redacted evidence, mandatory source-controlled real
  checks, bounded repair, attributable-change detection, and detachable
  MCP-stdio capability discovery/tool dispatch.
- Added the detachable `flyto.coding-service.v1` job boundary with tenant-hashed
  durable state, idempotent submission, a bounded queue, per-workspace
  serialization, restart reconciliation, authenticated loopback HTTP, and a
  configured-tenant MCP stdio facade. Provider credentials and tenant selection
  are startup-only and cannot be supplied in job payloads.
- Changed MCP capability preflight to require the negotiated protocol version
  and configured tool names from the real `tools/list` result instead of
  treating a configuration version label as proof of compatibility.
- Added explicit name-only `FLYTO_*` runtime environment passthrough for
  authenticated MCP stdio adapters. This enables detachable Cloud, Engine, and
  Robotics processes without persisting secret values or inheriting unrelated
  host credentials.
- Changed the Claude SDK coding agent into an optional compatibility backend
  and removed its implicit `bypassPermissions` and dangerous permission-skip
  settings. Native coding uses the normal Flyto2 provider stack.
- Added real subprocess, filesystem, symlink-escape, secret-redaction,
  fail-closed preflight, MCP-stdio, verification, repair, and no-change
  regression tests for the new control plane.
- Hardened model-issued coding commands with fail-closed OS sandbox discovery,
  read-only workspace/host access, no network, an ephemeral writable home,
  destructive-command denial, and credential/VCS path protection. Trusted
  source-controlled checks remain a separate verification lane.
- Added the versioned adaptive security campaign loop for footprint,
  penetration-test, and red-team planning. It binds every LLM proposal to
  target scope, expiring authorization, approved action classes, a Core module
  allowlist, cumulative step/request/round/token/cost budgets, proof assertions,
  and a content-addressed plan identity.
- Added runtime rechecks before Core execution and repair, compact proof
  accounting, raw-content-free evidence projection for model re-planning, and
  a `proved`/`not_proved` verifier that cannot turn missing evidence into
  success.
- Added exhaustive branch tests for the new campaign contract and adaptive
  loop, including scope escape, metadata SSRF, private-target policy,
  authorization downgrade/expiry, budget exhaustion, proof omission, prompt
  injection in evidence, failed replanning, and successful bounded repair.
- Added a per-call, control-plane-only authorization signal for generating
  security Blueprints against an explicitly approved non-staging hostname.
  The default remains staging-only and metadata/private-network SSRF checks
  cannot be bypassed by the signal.
- Fixed security campaign budget accounting so `http.batch` consumes the
  number of nested outbound requests at preflight, cost, and evidence
  accounting boundaries instead of counting the whole batch as one request.
- Added a provider-neutral structured Robotics planner with exact atomic
  capability and complete-route schemas, independent safety validation, one
  bounded repair, and tamper-evident request/schema/plan/provider attestation.
- Added native Ollama JSON Schema completions and a loopback-only
  `/v1/robotics/plan` development endpoint for live Physical AI planning.
- Added regression coverage for unsafe controls, shortlist escape, skipped or
  spliced branch locations, approval ordering, bounded repair, request limits,
  loopback binding, and sanitized HTTP errors.
- Added language-neutral `flyto.goal-frame.v1` routing with canonical
  intent/affordance/effect/event coverage, provider-neutral frame requests,
  Unicode-only lexical fallback, and an optional production fail-closed policy
  requiring a Goal Frame.
- Added deterministic-first routing for large external capability manifests,
  including runtime hard filters, deterministic shortlist ranking, trusted
  Blueprint hints, scoped Core discovery, registry snapshots, ambiguity
  evidence, and Robotics planner-request preparation.

- Removed a third-party popularity-tracking image from the README.
- Closed the GitHub security-and-quality backlog: CI now declares read-only
  repository permissions, the vulnerable PyPI publishing action is pinned to
  its patched release, and Checkout/Setup Python pins are current.
- Added an exact Grype exception for the patched PyPI action SHA because Syft
  reports that SHA as the package version instead of its 1.14.1 release. The
  exception matches one advisory, package, package type, and SHA only.
- Changed Dependabot to keep genuine security updates enabled while suppressing
  routine version-only branches that do not change CI's resolved dependencies.
  Added regression tests for these repository security policies.
- Rewrote the README opening around a concrete repeated-work story, plain
  language pain points, scoped token claims, routing/evidence safety, and the
  exact local verification numbers behind those claims.
- Hardened multilingual intent routing for explicit actions, current-data
  questions, negation, quoted/meta examples, and declarative questions. Route
  permissions are now rechecked at dispatch so forged provider calls cannot
  bypass answer-only or confirmation-required decisions.
- Made learned Blueprint trust evidence fail closed for malformed, non-finite,
  non-integral, inconsistent, or out-of-range values.
- Improved explicit reply-language switching and persistence across short
  follow-ups, with regression coverage for multilingual and mixed-language
  conversations.
- Added permanent routing, permission, Blueprint-boundary, malformed-evidence,
  adversarial-provider, presentation-mutation, and seeded Unicode/noise tests.
- Added trusted Blueprint execution evidence for duration, steps, attempts,
  assertions, workflow identity, executor version, and selection mode.
  Deterministic exact reuse now records `planner_model_calls_used=0` with an
  explicit planner scope. Blueprint can accept the old `model_calls_used`
  compatibility field, but new Flyto2 AI reports do not emit it. Model-selected
  paths leave counts unknown instead of fabricating a baseline.
- Rewrote the Blueprint/agent comparison to remove unsupported replay-token
  estimates and explain the Evidence Card proof boundary in plain language.
- Added Blueprint portable export/import dispatch and separated direct model
  outcome reports from host-verified closed-loop evidence with an in-process
  capability boundary. Blueprint selection without module execution evidence
  no longer counts as a verified success.
- Fixed closed-loop MCP verification so a missing identifier, an unknown
  `plan_id`, and a valid plan without execution evidence return distinct
  structured errors.
- Added a documentation contract, feature/API/configuration/operations guides,
  technical whitepaper, and generated references covering every declared
  Python function/class method, CLI option, static tool, environment read, and
  maintainer script.
- Unified package, CLI, and MCP version reporting and changed Core module totals
  to runtime discovery so installed capabilities cannot drift from source text.
- Added version/capability regression tests and a reusable documentation CI gate.
- Prepared a metadata-only PyPI patch release so live registry backlinks,
  project URLs, and runtime-discovered capability wording replace stale
  hard-coded module totals.
- Refactored OpenAI provider chat tool-call dispatch into a shared helper
  pipeline with direct regression tests for text completion, tool dispatch, and
  `ask_user` pause handling.
- Added ruff and flyto-indexer verify steps to the CI release loop.
- Split prompt-evolution mock response generation into category-specific
  helpers with regression tests for adversarial, partial-failure, language, and
  workflow responses.
- Added README usage guidance, `.env.example`, and prompt package docs so
  project documentation passes the local verify gate.
- Added flyto-core MCP capability manifest support.
- Added per-tool MCP metadata for risk, approval policy, and evidence fields.
- Added pre-execution parameter validation for `execute_module`.
- Added MCP evidence metadata to provider tool-call logs.
- Added CI workflow and `.flyto-index/` ignore.
- Added repo memory and workflow handoff scaffold.
- Added `docs/architecture-map.md` so Flyto2 workspace release packets can
  verify `flyto-ai` cross-repo architecture and product-line boundaries.

### Added
- Cross-job continuation of a bounded provider stop: a tenant-partitioned,
  single-use continuation authority with an append-only transition journal, an
  explicit `resume=true, thread_id=<session>` second submit, and two additive
  receipt fields (`continuation_available`, `continuation_generation`). The MCP
  surface is unchanged.
- An explicit, digest-bound workspace snapshot policy. The default observes every
  non-version-control entry; only the strict Indexer-backed route may classify
  `.flyto-index` as control-plane runtime state.

### Changed
- `CodingService.submit` is phased: the verification-contract read and the
  workspace snapshot now run under a per-workspace admission lock instead of the
  global state guard, so one large repository no longer stalls unrelated tenants
  and workspaces.
- The service state root is created component-by-component and refuses a symlinked
  ancestor rather than resolving through it.

### Fixed
- Repository-relative verification programs (for example
  `.venv/bin/python`) are now resolved from the requested workspace in
  preflight, both implementation adapters, and the real check runner, rather
  than from the MCP supervisor's launch directory.
- A multi-round coding rework no longer re-roots its Indexer plan each round.
  The pre-lane amends the exact prior contract, so a later round is not refused
  with `unplanned_diff` for files an earlier round legitimately opened.
- Indexer post-work now validates the exact cumulative attributable set that the
  final revision binds, instead of only the last round's changed files.
- A dotted machine identifier such as `check.generated_reference` is no longer
  parsed out of audit feedback as a request to create a file.

### Added
- Durable, private, integrity-protected plan authority per job, re-proven before
  a resumed implementer edits anything.
- Closed typed failures for unprovable plan authority and cumulative scope,
  reporting `verification`/`workspace` phase with
  `resubmit_against_current_contract`.
- Bounded domain diagnostics: a capability's own `reason_codes` and
  `required_actions` reach `verification_blockers` when they already are machine
  codes, and are dropped whole when they are not.
- Added phase 1 of the provider-neutral Capability Catalog / Skill Registry
  contract: exact semantic claims, frozen host authority, canonical SHA-256
  content binding, detached Capability Cards, and bounded allowlisted search
  projections. Only complete claims with exact approved, verified, active,
  non-retired host bindings are autonomous-routable. This phase adds no storage,
  vector search, retrieval, installation, execution, approval/verification
  service, UI, router call, or runtime-provider integration.
- Replaced the non-landable phase 2 retrieval draft with a frozen host-verified
  `flyto.ai.capability-retrieval-handoff.v2` terminal handoff that preserves the
  exact Blueprint page and Cloud result/feasibility contracts and their digest
  meanings. Distinct versioned AI-local goal, routing-context, and Goal-Frame
  digests avoid overloading upstream names; true cross-resource feasibility and
  exact `candidate_resources` are required without imposing co-location or page
  membership. The producer model and active/ACL/risk/resource/capability filter
  invariants are rebuilt, while one capability candidate retains every
  distinct installed provider bound to its accepted document. Empty capability
  filters retain open-discovery semantics, and upstream `/` identifiers retain
  their exact field bounds. Model ID/version stop at 128 characters, while
  scope/capability IDs remain 192. AI-local routing context and Goal Frame are
  now exact-JSON bounded before digesting or returning, preventing hostile
  objects or recursive/oversized values from leaking raw exceptions. Cloud
  feasibility now accepts at most 128 canonical capability keys.
  `CAPABILITY_GROUP_LIMIT` names the 32-group public bound and independent
  `EMITTED_PROVIDER_ROW_LIMIT` names the 32-provider-row projection bound;
  co-providers expand in stable identity order and overflow has no partial
  output. Blueprint retains request/model/index/snapshot/page/candidate digest
  meanings, Cloud retains query-context/requirements/feasibility/result
  meanings, and host validation grants no execution authority. Final pins are
  Blueprint `f3eb62eff97fac3b3f19d2f1c8d7c1e71664894b`, Core
  `a048bc47de158c096b7010642452e4d41d21748c`, and Indexer
  `b492ef9b663f4a37c4883e2b9e1d8b45b3719b6d`.
  The handoff can narrow and
  boundedly hint already installed providers. It rejects partial pages, stale
  identities, scope/context/model/index/snapshot or hard-filter drift, hostile
  JSON, and authority-shaped results. Vector relevance cannot override Goal
  Frame semantics, safety/human gates, permissions, planning, or execution
  closure.
