# Architecture

Runtime flow:

```text
user/cloud/CLI
  -> Agent
  -> provider chat loop
  -> ToolRegistry
  -> flyto_ai.tools.core_tools
  -> flyto-core MCP handler
  -> structured result + evidence metadata
  -> blueprint/eval/trace feedback
```

The reusable control loop is domain-neutral even though each high-risk domain
adds its own contract:

```text
goal / event / sensor input
  -> flyto.goal-frame.v1 normalization
  -> manifest and compatibility routing
  -> policy / scope / authorization gate
  -> domain planner or Agent tool loop
  -> flyto-core or explicit domain executor
  -> domain verification + bounded repair/re-plan
  -> redacted evidence / trace / trusted Blueprint feedback
```

`Agent` owns general workflows; `FlytoCodingAgent` adds workspace, thread,
check, and attributable-change semantics; `RoboticsPlanningService` adds route
integrity and robot safety planning; `run_security_campaign` adds explicit
authorization, scope, action, module, and budget ceilings. New domains attach a
typed adapter at these boundaries instead of adding task-name branches to the
shared loop.

Key boundaries:
- Providers never call `flyto-core` directly.
- Cloud imports `flyto-ai` contracts and dispatchers, not `flyto-core` internals.
- Blueprint learning stores successful tool chains and redacted evidence, not
  secrets. Portable exchange is explicit; Flyto2 AI does not publish bundles by
  itself.
- Model-facing Blueprint outcome calls are community observations. Only the
  deterministic closed-loop executor can attach the in-process capability that
  records `local_verified` evidence.
- Blueprint signing keys and trusted publisher mappings stay host-controlled
  and are never exposed through model tool schemas.
- MCP metadata is additive: existing tool names, schemas, and result shapes stay compatible.

Runtime packages:
- `coding` is the provider-neutral software-development control plane. It owns
  versioned requests/results, workspace-confined argv-only tools, persistent
  resumable threads, append-only redacted trajectories, source-controlled real
  checks, negotiated MCP-stdio capability adapters, and optional tenant-scoped
  HTTP/MCP job facades. It does not import sibling Indexer/Core source trees.
- `assistant` and `intelligence` perform deterministic pre-routing, recovery, selector resolution, and interaction control before provider fallback.
- `providers` normalize model chat, streaming, tool calls, usage, failover, and cost records.
- `tools` owns definitions and handlers; Core registry definitions are discovered lazily instead of copied.
- `memory`, `evolution`, `cache`, `session`, and `transcript` retain bounded learning and evidence state.
- `permissions`, `prompt`, `redaction`, `vault`, `sandbox`, and `agents` enforce execution and data boundaries.
- `channels`, `telegram`, `scheduler`, and `extensions` adapt external events without bypassing the agent/tool contract.

Coding control flow:

```text
CLI / API
  -> flyto_ai.coding.CodingTaskRequest (flyto.coding.v1)
  -> source-controlled checks + required-capability preflight
  -> selected Flyto2 LLMProvider
  -> workspace-confined coding tools + configured MCP-stdio adapters
  -> real check runner
  -> bounded repair loop
  -> CodingTaskResult + append-only JSONL evidence
```

Full-stack capability composition remains additive and process-bound:

```text
FlytoCodingAgent
  -> flyto-indexer (context / impact / task gates)
  -> flyto-blueprint (workflow discovery / reuse / learning)
  -> flyto-page-inspector (real DOM inspection only)
  -> flyto-core (module / recipe execution and visual evidence)
  -> source-controlled checks and bounded repair
```

`flyto_ai.coding.stack` retains that built-in preset and also loads arbitrary
source-controlled agent-stack profiles. `compose_capability_stack()`
has no domain catalog; `CapabilityManager` satisfies the generic
`ToolExecutor` protocol and can attach the resulting tools to `Agent`. Each
lane negotiates its real MCP catalog independently and exposes only
`CapabilitySpec.allowed_tools`; manifest-loaded MCP lanes require an explicit,
non-empty allowlist. The Blueprint and page-inspection lanes may use the same
server implementation without sharing model-visible authority. Removing one
spec detaches that lane without changing the provider, workspace tools, checks,
or result contract. Page inspection continues to flow through `core_tools`;
its portable launch policy tries bundled Chromium and then system Chrome,
records the selected channel, and never bypasses Core.

Policy-bearing `flyto.agent-stack.v2` profiles classify every exposed MCP tool
as read-only, workspace-write, or danger-full. This declaration never grants
authority: `CapabilityManager` receives its ceiling from the runtime host and
checks it at every dispatch, while `Agent` independently applies the same
provider-name overrides in its safe dispatcher. Core module execution also
re-evaluates the concrete `module_id`, preserving danger-category escalation
after MCP tool names have been isolated. Legacy v1 manifests remain accepted
with their historical workspace-write default.

The implementation is deliberately finer-grained than the public API:

```text
flyto_ai.coding.stack (stable facade + CLI)
  -> stack_manifest (bounded I/O, schema, composition, configured fingerprint)
  -> stack_presets  (detachable built-in catalog only)
  -> stack_probe    (runtime negotiation and observed fingerprint)

flyto_ai.coding.capabilities (stable facade + lifecycle coordinator)
  -> mcp_session    (handshake and call orchestration)
     -> mcp_transport (isolated process, bounded JSON-RPC, deterministic close)
     -> mcp_catalog   (scope, provider names, machine-readable result status)
  -> tool_registry  (transactional routing and collision rejection)
  -> permissions    (host ceiling plus monotonic argument-risk resolvers)
```

The ordinary `Agent` separately validates and binds the generic
`ToolExecutor`. A new domain extends `CapabilitySpec`, its profile, and an
optional host-owned argument-risk resolver; it does not add task-name branches
to the manager. Registry updates commit only after the complete session catalog
has validated. Any collision or incomplete mapping closes all affected
sessions and clears runtime dispatch metadata. Transport close first closes
stdin and awaits EOF, then uses bounded terminate/kill escalation, so repeated
attach/detach cycles do not leave orphaned subprocess transports.

Optional service composition:

```text
loopback HTTP (bearer + idempotency) / MCP stdio (configured tenant)
  -> flyto.coding-service.v1
  -> tenant namespace + workspace allowlist + bounded queue
  -> per-workspace serialization
  -> the same FlytoCodingAgent control flow above
  -> durable CodingJobReceipt
```

Provider selection, credentials, tenant identity, workspace roots, and the
state root are startup dependencies. They are not accepted from a job payload.
This makes Cloud, Engine, Robotics, Core, and Indexer consumers replaceable at
the process contract instead of coupling their source trees to `flyto-ai`.

The native backend is the default architecture. `ClaudeCodeAgent` remains a
separate optional adapter and never receives implicit permission bypass. A
capability can be removed from `.flyto/coding.yaml` without changing the
provider, workspace tool, check, or result contracts.

An MCP capability is available only when its initialize response negotiates the
requested protocol and its real `tools/list` includes every required tool.
Configured version labels alone never satisfy preflight. When `allowed_tools`
is configured, every listed tool must exist and only that subset enters the
provider tool catalog; omitted allowlists preserve the prior full-catalog
behavior.
MCP transport success is not treated as domain success: structured content or
a single JSON content block carrying `ok: false` or an error status fails the
capability result before the Agent can trust the evidence.
Authenticated product adapters receive only explicitly named `FLYTO_*` runtime
variables. The source-controlled contract carries names rather than values;
ambient cloud, source-control, SSH, and provider secrets do not cross the
subprocess boundary.

Model-issued commands are a separate trust lane from source-controlled checks.
The former require a detected OS sandbox, deny network and workspace/host
writes, and receive only an ephemeral writable home; the latter are trusted
project verification and may execute repository code. Native file tools reject
VCS internals, credential paths, path traversal, and symlink escape.

Interface surfaces:
- Python consumers import the package facade documented in `docs/API.md`.
- CLI and HTTP/SSE clients use contracts documented in `docs/CLI_AND_MCP.md`.
- MCP hosts negotiate JSON-RPC protocol versions and discover tools at runtime.
- Generated symbol and operator references under `docs/reference/` remain source-derived and are checked in CI.

Core contract:
- `get_core_capability_manifest` reports contract version, installed core version, tool fingerprint, recipes support, module categories, and per-tool risk metadata.
- `execute_module` validates params before execution when `flyto-core` exposes `validate_params`.
- Tool logs include `mcp.source`, `mcp.contract_version`, and module or recipe identity.

## Adaptive security campaign boundary

```text
LLM planner
  -> typed campaign + proposed PlanIR steps
  -> scope / authorization / allowlist / budget gate
  -> existing closed-loop MCP plan
  -> permission gate + flyto-core validation and execution
  -> assertions + compact proof evidence
  -> allowlisted, raw-content-free planner projection
  -> bounded re-plan or verified verdict
```

- The LLM is the decision and prioritization layer; it never becomes execution
  authority. Every round enters through the existing four-tool MCP contract.
- The campaign contract is part of plan identity. Changing target scope,
  authorization, mode, module allowlist, budget, or prior usage changes the
  stored plan hash.
- Active probes, exploit validation, and credential validation require
  progressively stronger authorization. Active steps also require an explicit
  in-scope target and a proof assertion.
- Authorization expiry, scope, module allowlist, request budget, and cost
  budget are checked again at dispatch time, including repaired steps.
- Evidence returned to a subsequent model round is an allowlisted projection
  containing facts and fingerprints, never raw bodies, HTML, headers, cookies,
  credentials, prompts, or attacker-controlled error text.
- A successful Core call alone is not a security verdict. Verification requires
  the runtime closed loop, assertions, budgets, and one proof record per
  executed request; otherwise the verdict is `not_proved`.

## Robotics planning boundary

```text
Flyto2 Robotics routed request
  -> validate request size, shortlist, capabilities, locations, routes
  -> compile provider-native JSON Schema
  -> structured model completion
  -> independent plan/safety/route validation
  -> optional single bounded repair
  -> plan + tamper-evident planning attestation
  -> Flyto2 Robotics final validation and execution
```

- `robotics_planning.py` owns the provider-neutral request, response, and
  attestation boundary. It does not import Flyto2 Robotics source.
- The routed capability shortlist is the authority ceiling. Model output cannot
  introduce a capability or argument field outside that contract.
- Complete route candidates become exact JSON Schema `prefixItems` variants.
  Model choice remains real, while waypoint omission and cross-route splicing
  become structurally invalid.
- Provider validation is not execution authorization. Flyto2 Robotics verifies
  the hashes, route, policy, and executable plan again before movement.
- `robotics_planner_server.py` is a loopback development adapter, not a public
  authenticated service. It suppresses prompt-bearing access logs and bounds
  request, response, timeout, and error detail sizes.
