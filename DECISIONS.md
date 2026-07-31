# Decisions

## 2026-07-30: Let the model choose a complete route, not invent waypoints

- Robotics supplies a bounded shortlist, trusted semantic location IDs, and
  complete route candidates after deterministic compatibility, permission,
  resource-health, and dependency filtering.
- Flyto AI converts every surviving route into an exact JSON Schema step
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
