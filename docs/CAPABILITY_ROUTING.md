# Capability Routing

Flyto AI routes large capability catalogs with a deterministic-first boundary.
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
