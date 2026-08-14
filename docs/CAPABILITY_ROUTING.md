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
  `flyto_ai.tools.core_tools.dispatch_core_tool`. It adds only **installed,
  registry-declared capability providers** to the routable catalog, and adding a
  provider to the catalog never adds execution authority: the provider still
  passes every hard filter, still has to be selected, and is still validated at
  dispatch.

### Capability identity and provider identity are declared, never derived

`search_modules` results carry two string identity fields that Core normalizes
to a registry value or `""`: `provides_capability`, declared by the module, and
`plugin`, stamped by `ModuleRegistry`. This router consumes both verbatim.

- A Core result becomes a capability manifest only when it carries a module ID
  **and** a non-empty, safe `provides_capability`. `canonical_id` is then
  exactly that declared value.
- The projected manifest keeps `provides_capability` and `plugin`.
- Nothing here reconstructs either identity from `module_id`, `category`, a
  prefix, or a plugin name. `_canonical_id()` does not synthesize a
  `core.module.<module_id>@1` fallback, so a manifest whose `source` is
  `flyto-core` and that declares no explicit safe capability ID is invalid and
  is excluded as `invalid_or_duplicate_identity`.
- An unsafe declared ID is excluded deterministically. It is never stripped,
  repaired, or replaced with an invented one.
- `provides_capability=""` is a legitimate ordinary Core module, not a
  capability. It is simply not projected, so it can neither become an invented
  capability nor appear as a routing exclusion.

A **capability ID** answers "what can be done". A **provider identity** answers
"which installed module would do it" and is the tuple
`(canonical_id, runtime_name, plugin, source)`. Several modules or plugins may
legitimately provide one capability, so only an identical provider identity is
rejected as a duplicate; two distinct runtime module IDs that share a capability
ID both stay routable and both stay distinguishable. Route candidates therefore
carry bounded `plugin` and `source` fields alongside `canonical_id` and
`runtime_name`, and the ranking tiebreak is the full identity, so co-providers
keep a deterministic, auditable order.

The `limit` argument counts canonical capability groups and is bounded by the
public `CAPABILITY_GROUP_LIMIT` (32), not by provider rows. Selecting one
capability expands every distinct exact installed co-provider in full-identity
order. `EMITTED_PROVIDER_ROW_LIMIT` independently caps the final projection at
32 rows; if a complete group would cross that ceiling, routing fails closed and
emits no partial group. The separately named constants prevent a future
group-limit change from silently widening the provider-row boundary.

### Source scope: explicit ceiling, dynamic default

`context.allowed_sources` is an authority ceiling. When a caller supplies it, it
is preserved exactly and discovery never widens it — a discovered `flyto-core`
provider is excluded with `source_out_of_scope` unless the caller listed
`flyto-core`.

When `allowed_sources` is absent, the default scope is the set of sources
present in the supplied manifests, plus `flyto-core` when this round actually
verified at least one registry-declared Core capability provider. Arbitrary Core
search results cannot enter that default scope, because an ordinary module is
never projected into a manifest in the first place.

### Discovery status is a lane outcome, not an empty list

Both discovery lanes report a bounded machine-readable `status` and a bounded
`status_reason` in `discovery_evidence`. The status vocabulary is exactly
`applied`, `not_applicable`, `unavailable`, `failed` and is exported as
`DISCOVERY_STATUSES`. An unreachable or broken lane is therefore no longer
indistinguishable from a legitimate no-match. Reason codes are drawn from a
fixed vocabulary; raw exception text, tracebacks, and bridge internals never
cross this boundary.

| Lane | `applied` | `not_applicable` | `unavailable` | `failed` |
| --- | --- | --- | --- | --- |
| Blueprint | search returned candidates | search returned zero candidates | package or engine absent (`engine_unavailable`) | search raised (`search_failed`), wrong result shape (`invalid_result`), or more than `BLUEPRINT_CANDIDATE_LIMIT` candidates (`result_over_bound`) |
| Core | valid runtime manifest plus verified capability providers (`discovery_matched`) | valid runtime manifest and zero results (`discovery_empty`), or well-formed results that declare no capability (`no_capability_providers`) | explicit `ok=false` (`runtime_unavailable`), absent runtime evidence (`runtime_missing`), or wrong contract (`runtime_contract_mismatch`) | non-boolean `ok` or non-string contract (`runtime_malformed`), or a malformed search response (`search_malformed`) |

`candidate_count` for the Core lane counts **valid projected capability
providers**, not raw search hits. A search that returned only ordinary modules
therefore resolves as `not_applicable` with `candidate_count=0`, which is a
clean lane outcome rather than a failure.

Core runtime evidence is trusted only when the bridge returns the exact
`flyto-core-mcp.v1` contract (`CORE_RUNTIME_CONTRACT`) together with an
explicit boolean `ok=true`. A missing `ok`, a truthy non-boolean `ok`, or a
different contract version is absent or malformed runtime evidence, never a
no-match. The Core search response must be an object with a `results` list
whose every entry carries a safe non-empty `module_id`; a single malformed
entry fails the lane instead of being silently filtered into
`not_applicable`. The same rule covers the identity fields: a present
`provides_capability` or `plugin` that is not a string is a broken bridge and
fails the search contract, because coercing it with `str()` would manufacture an
identity that no registry published.

Blueprint discovery is bounded at `BLUEPRINT_CANDIDATE_LIMIT` (32) candidates
and reads at most one item past that ceiling, so a runaway or streaming bridge
cannot force an unbounded materialization. Because the trust gate consumes only
the single highest-ranked trusted procedure, an over-bound result is evidence of
a broken bridge rather than richer evidence, and it fails closed.

Both lanes stay read-only. Blueprint discovery only re-ranks modules that are
already installed. Core discovery may extend the routable catalog, but only with
installed providers the registry itself declares, and never with execution
authority: a discovered provider still passes the same hard filters, still has
to win selection, and is still schema-, permission-, and safety-validated at
dispatch. Discovery never widens an explicit `allowed_sources`. Core calls
continue to flow through `flyto_ai.tools.core_tools.dispatch_core_tool`, and a
bridge call that raises is re-raised as a bounded `CapabilityRoutingError`
rather than a raw provider or transport exception.

### Planner propagation

`prepare_planner_request()` applies this boundary to
`flyto.robotics.planner-request.v1`. It replaces the catalog with the verified
shortlist before provider dispatch and attaches the full routing decision as
evidence.

The shortlist is resolved against the **combined catalog** the route was
actually decided over — the supplied manifests plus the verified
Core-discovered providers — not by re-filtering the request's original list. A
legitimately discovered provider would otherwise be silently dropped after being
selected. Each selected candidate is matched back by full provider identity
(`canonical_id`, `runtime_name`, `plugin`, **and** `source`), never by runtime
name or capability ID alone, so an unselected or unregistered module that
happens to share a runtime name or a capability ID with a selected provider
cannot inherit its execution authority. Manifests are emitted in route-candidate
order, so two distinct providers of one capability both propagate and keep the
route's deterministic identity order.

Resolution fails closed. Every selected candidate must match **exactly one**
manifest in the combined catalog. A candidate that matches none, or one that
matches more than one same-identity manifest, raises a bounded
`CapabilityRoutingError` carrying only the two counts
(`unresolved=`, `ambiguous=`); no catalog-controlled identity text crosses that
boundary. An unresolved candidate is never silently dropped, because a shortlist
that is quietly narrower than the route it ships with is exactly the mismatch
downstream validation rejects — `robotics_planning` requires `capabilities` to
describe `capability_route.candidates` exactly. An ambiguous candidate fails for
the same reason a duplicate identity is refused during ranking: picking one of
several same-identity manifests would grant execution authority by catalog
order. Callers must therefore de-duplicate provider identities before routing;
supplying a manifest that exactly repeats a discovered Core provider identity is
an ambiguous catalog, not a hint.

Production robot entry points should set `require_goal_frame=True`; the default
remains backwards compatible with older planner clients, including callers whose
environment exposes no discoverable Core capabilities at all.

### Strict discovery mode

`prepare_planner_request(require_discovery=True)` requires both lane statuses to
be in `{applied, not_applicable}`. An `unavailable` or `failed` lane raises a
bounded `CapabilityRoutingError` naming only the two statuses, and it raises
before any provider could run, so a degraded discovery lane cannot silently
become a narrower shortlist that a model then plans against. Production robot
and coding entry points that treat Blueprint and Core as required lanes should
set it alongside `require_goal_frame=True`.

**Compatibility and rollback.** `require_discovery` defaults to `False`, which
preserves the existing library behavior exactly: `route_with_flyto()` is
unchanged apart from the additive `status`/`status_reason` evidence fields and
the typed error wrapping, and existing callers that never read
`discovery_evidence` see no behavior change. Rolling back strict mode is a
call-site change — drop `require_discovery=True` — and needs no contract,
manifest, or catalog migration. Rolling back the contract entirely means
reverting `flyto_ai/capability_router.py`; the evidence fields are additive, so
no persisted route or planner request needs rewriting.

Missing semantic coverage or a low-confidence legacy route sets
`needs_clarification=true`. Robot-side policy must then require a human gate or
reject the plan. A provider cannot select a capability that was filtered out
or absent from the exact registry snapshot.

## Stack profiles and routing manifests are separate layers

## Bounded retrieval handoff

`route_capabilities()` optionally accepts a
`flyto.ai.capability-retrieval-handoff.v2` object plus a frozen
`CapabilityRetrievalAuthority`. This is the only
retrieval-to-route contract. It is provider and domain neutral and imports no
sibling Cloud or Blueprint implementation.

The handoff preserves the accepted Blueprint query and page and the complete
Cloud result and feasibility objects. Their `request_digest`,
`query_context_digest`, `requirements_digest`, `result_digest`, candidate
digests, model/index/snapshot digests, integer scores, fields, and meanings are
not recomputed as AI concepts. The frozen host authority binds those exact
upstream values and tenant/workspace/Space. Separate versioned `goal_digest`,
`routing_context_digest`, and `goal_frame_digest` fields bind AI-local input.
Those AI-local context and normalized Goal Frame inputs are detached through
the same exact-JSON depth, node, byte, finite-number, and integer boundary
before digesting or route projection; hostile values expose only the stable
handoff error. Blueprint model ID and version retain their exact 128-character
producer ceiling, while tenant, Space, and capability IDs retain 192.

Only a terminal full top-k window is admitted: `top_k` is 1..32,
`page_size == top_k`, Blueprint input/next cursor and Cloud continuation are
null, and every layer is candidate-only without execution authority. Cloud
feasibility must be true and its exact `candidate_resources` remain candidate
truth; independent requirements may be satisfied by distinct resources and
are never treated as a co-location demand or required to occur on the returned
page. Candidates retain the exact Blueprint field set and model dialect
(`model_id`, `model_version`, `dimensions`, `model_digest`). Their accepted
document digests resolve to every distinct installed full provider identity,
ordered deterministically; duplicate identities and unknown documents fail
closed.

The boundary rebuilds the upstream active status, nonempty ACL principal/scope,
risk ceiling, resource subset, capability filter, identifier bound, and
candidate-agreement invariants. An empty `capability_ids` list is exact upstream
open discovery; membership is enforced only when the list is nonempty. Upstream
identifiers use Blueprint's `/`-capable syntax with the field-specific 128/192
limits rather than the router-local manifest grammar. Unknown fields, hostile containers,
non-finite or boolean numbers, duplicates, stale/mixed bindings, truncation,
and digest or authority drift fail with only
`invalid capability retrieval handoff`.

The result set may only narrow installed manifests. Its normalized score adds
at most one point after an existing semantic match; it cannot make a
Goal-Frame-zero candidate eligible. Distinct providers of one capability stay
distinct, and final resolution still uses `(canonical_id, runtime_name,
plugin, source)`. Installed safety and human-gate controls remain available,
but retrieval never grants them or any other provider authority. Empty results
are non-routable. Evidence is versioned and digest-only, repeats
`candidate_only=true` and `execution_authority=false`, and states that planning,
permission, and execution closure remain required.

This producer-compatible edge is locked to Blueprint
`f3eb62eff97fac3b3f19d2f1c8d7c1e71664894b`, Core
`a048bc47de158c096b7010642452e4d41d21748c`, and Indexer
`b492ef9b663f4a37c4883e2b9e1d8b45b3719b6d`. Blueprint owns the request,
model, index, snapshot, page, and candidate digest meanings; Cloud owns the
query-context, requirements, feasibility, and complete result meanings. Flyto2
AI validates both against frozen host authority but receives only candidate
evidence, never execution authority.

This edge does not turn a million-row backend into a router input: only the
host-validated <=32-candidate terminal handoff crosses the boundary. Rollback
is removal of the two optional arguments; existing callers and route contracts
remain compatible.

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

Every production adapter should also ship an `AdapterConformanceCase` set.
`run_adapter_conformance()` proves exhaustive permission classification, exact
MCP protocol/catalog negotiation, at least one case for every allowed tool,
domain-owned success and failure evidence, a valid trace, released policy
leases, and complete lifecycle cleanup. Its fingerprint binds the tested
contract, redacted inputs, observed trace, and checks. `run_scenario_matrix()`
aggregates those independent suites under a concurrency bound. The matrix
accepts arbitrary scenario and domain labels; adding a robot, ticketing, data,
or security adapter changes its own spec/cases/verifier, not the shared runner.
Conformance defaults to read-only permission. Tests that intentionally exercise
writes or dangerous actions must select higher test authority explicitly, and
each case states whether a real dispatch is expected so a policy denial cannot
masquerade as the requested domain failure.

Runtime admission remains separate from routing. `ExecutionPolicyController`
applies lifecycle/concurrency, JSON byte/depth/node, secret/path, result, and
bounded optional human-approval checks after permission evaluation but before
dispatch. `ExecutionTraceLedger` then records deeply immutable redacted
hash-chained evidence; the Agent also forwards outer denials to an
evidence-aware executor. Replay takes a fixed snapshot, skips redacted inputs,
and defaults to read-only events. Workspace-write or danger-full replay is an
explicit host decision. A mismatch fails the evidence comparison; only a
host-owned feedback sink may translate the replay report into a trusted
Blueprint outcome.

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
