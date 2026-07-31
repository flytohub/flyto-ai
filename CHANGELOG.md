# Changelog

## Unreleased

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
