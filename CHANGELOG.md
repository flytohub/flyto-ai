# Changelog

## Unreleased

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
  before the full Python 3.10/3.12 suite.
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
  CI now runs the complete suite on Python 3.10 and 3.12.
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
