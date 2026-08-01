# State

Last updated: 2026-08-01

Implemented:
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

Verified on Python 3.11:
- full suite: 1288 passed, 15 optional/live-integration skips;
- Ruff fatal/error rules and `compileall`: pass;
- wheel and source distribution build plus Twine metadata validation: pass;
- strict documentation contract: pass;
- Flyto2 Indexer closed loop: 18 passed, 0 warnings, 0 failures (90/A).

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
- Native workspace confinement is an application boundary, not hostile-code OS
  isolation for source-controlled verification commands. Model-issued commands
  use Docker or `bwrap`, but untrusted repositories must still run the whole
  process inside a dedicated container or VM. MCP capability commands must be
  explicitly configured in `.flyto/coding.yaml` and are not inferred from
  sibling source directories.
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
