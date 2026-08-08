# State

Last updated: 2026-08-09

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
  -> startup-selected implementer: native or claude
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
- Startup backend selector `--implementation-backend native|claude` with the
  bounded `FLYTO_AI_CODING_BACKEND` default, no per-job override and no
  fallback; the Claude route is pinned to `claude-opus-5` and reads only
  bounded `FLYTO_AI_CC_*` settings.
- Public audit surface on both transports: `flyto_coding_submit`,
  `flyto_coding_get`, `flyto_coding_audit`, and authenticated
  `POST /v1/coding/jobs/{job_id}/audit`.
- Coding MCP `initialize` advertises server version `2` and bounded
  instructions describing the host-owned loop; it negotiates only `2025-06-18`.
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
- A deployment must still supply a reachable `--indexer-command`. An
  unreachable Indexer fails every public job closed rather than degrading.

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
  commit beside `flyto-ai` before running the complete suite on Python 3.10 and
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
- Python 3.10/3.12 CI treats deprecation and unhandled background-thread
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
