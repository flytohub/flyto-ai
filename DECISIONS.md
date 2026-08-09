# Decisions

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
only coding-loop implementation. It is now one of exactly two peer
implementers behind the same audited service contract.

- Codex, or whichever principal the host authenticates, is the orchestrator and
  the independent auditor. `flyto-ai` is the single coding route between them
  and the implementer; there is no second path that reaches a landable result.
- The operator selects exactly one implementer at process startup with
  `--implementation-backend native|claude`, or the bounded
  `FLYTO_AI_CODING_BACKEND` default. `native` remains the default. There is no
  per-job backend field, no provider/model auto-routing, and no fallback in
  either direction; an invalid or unavailable selection fails startup.
- Claude service rounds are pinned to `claude-opus-5`. Configuration can vary
  the legacy direct backend's model but can never redirect audited service
  work. The Claude route reads only bounded `FLYTO_AI_CC_*` settings and
  resolves no native provider credential or configuration.
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
Select `--implementation-backend native` to detach the Claude adapter, or lower
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
