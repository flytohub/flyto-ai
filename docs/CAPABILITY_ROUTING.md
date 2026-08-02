# Capability Routing

Flyto2 routes large capability catalogs with a deterministic-first boundary.
The model is a planner over a verified shortlist; it is not the catalog,
compatibility, permission, or safety authority.

```text
goal in any language, modality, or upstream event format
  → flyto.goal-frame.v1 (canonical intents, affordances, effects, events)
  → source/domain/robot/sensor/resource/permission hard filters
  → exact semantic-frame rank
  → trusted Blueprint module hints
  → scoped Core discovery through core_tools
  → bounded top-k + confidence + ambiguity decision
  → LLM plan
  → runtime schema, permission, and safety validation
```

The language adapter is deliberately separate from the router.
`goal_frame_request()` gives any structured-output provider the installed
ontology vocabulary; `normalize_goal_frame()` rejects unknown fields and unsafe
semantic IDs. The router never branches on a locale and, when a Goal Frame is
present, raw wording does not participate in ranking. This also permits
non-language producers such as UI forms, QR codes, schedules, and sensor events.

`route_capabilities()` accepts only JSON-like manifests. Each manifest should
provide a stable `canonical_id`, executable `runtime_name`, version, domain,
canonical `intent_ids`, `affordances`, `effects`, `handled_events`,
compatibility requirements, safety metadata, and argument contract. Aliases,
tags, and examples remain an optional legacy recall fallback when no Goal
Frame is available; they are not the production semantic authority. The output
uses `flyto.capability-route.v1` and includes semantic coverage, the registry
SHA-256 snapshot, score/reason evidence, exclusions, confidence, and
`needs_clarification`.

`route_with_flyto()` adds two evidence sources:

- Blueprint search summaries can boost installed modules only when the existing
  trust/evidence gate accepts the Blueprint. `module_ids` contains no params or
  execution data.
- Core discovery always flows through
  `flyto_ai.tools.core_tools.dispatch_core_tool`. Discovered Core modules are
  hard-filtered unless `allowed_sources` explicitly includes `flyto-core`.

`prepare_planner_request()` applies this boundary to
`flyto.robotics.planner-request.v1`. It replaces the catalog with the verified
shortlist before provider dispatch and attaches the full routing decision as
evidence. Production robot entry points should set
`require_goal_frame=True`; the default remains backwards compatible with older
planner clients.

Missing semantic coverage or a low-confidence legacy route sets
`needs_clarification=true`. Robot-side policy must then require a human gate or
reject the plan. A provider cannot select a capability that was filtered out
or absent from the exact registry snapshot.

## Stack profiles and routing manifests are separate layers

`flyto.agent-stack.v2` answers “which external capability processes and tools
may this agent instance attach, and what minimum permission does each tool
require?” A source-controlled profile may use arbitrary domain names and MCP
servers, but every MCP entry must carry a non-empty `allowed_tools` list and an
exhaustive `tool_permissions` map. Preflight validates its declared contract
against the server's real catalog before any model receives those tools. v1
profiles remain a compatibility input with workspace-write as the unclassified
default.

The capability manifests consumed by `route_capabilities()` answer the finer
question “which installed action is compatible with this particular goal and
authority envelope?” They describe canonical intent, affordances, effects,
resources, sensors, permissions, schemas, and safety metadata. Keeping process
composition separate from per-goal routing lets one generic Agent host support
many domains without turning a broad catalog into blanket execution authority.
The profile's classification is only a lower-bound requirement; the runtime
permission ceiling is host-owned and enforced again at dispatch.

The composition implementation is also replaceable by layer. Manifest parsing,
built-in presets, runtime probing, MCP transport, catalog normalization,
session negotiation, registry state, and permission evaluation are independent
modules behind stable facades. Domain growth should normally add a
`CapabilitySpec`, routing manifest, typed domain contract, verifier, and—only
when call arguments change harm—a host-owned risk resolver. Risk resolvers may
escalate a declared permission but cannot downgrade it. `CapabilityManager`
therefore remains domain-neutral instead of accumulating robotics, security,
browser, or workflow task branches.

The shared domain-neutral loop is:

```text
goal/event
  → normalized Goal Frame
  → installed-profile and capability hard filters
  → policy / scope / authorization gate
  → domain planner
  → Core or domain executor
  → domain verifier and bounded repair/re-plan
  → redacted evidence, trace, and trusted Blueprint outcome
```

General workflows use `Agent`; software changes add the `FlytoCodingAgent`
workspace/check contract; robotics adds `RoboticsPlanningService` plus robot
safety and human gates; authorized security campaigns add
`run_security_campaign` scope, expiry, action-class, module, and budget gates.
New domains extend these adapters and manifests instead of adding task names to
the router or weakening the common authority boundary.
