# Flyto2 coding control plane

`flyto_ai.coding` is a provider-neutral coding loop. The native implementer uses
the configured Flyto2 provider (OpenAI, Anthropic, Ollama, or a compatible
adapter) and requires no vendor agent SDK. An optional Claude adapter is its
peer behind the same contracts. A Codex CLI adapter can use an existing
ChatGPT login in a separate non-interactive implementation session. The
operator selects exactly one backend at startup.

The public `code-mcp` and `code-serve` commands are one audit-required route:
an implementer round ends at `awaiting_codex_audit`, and only an authenticated
host verdict can reach an accepted, landable receipt. See
[Detachable service adapters](#detachable-service-adapters).

## Success contract

A coding run succeeds only when all of these statements are true:

1. every required external capability completed preflight;
2. the provider/tool loop completed without an unhandled failure;
3. every required source-controlled check ran as a real subprocess and passed;
4. when `require_changes` is enabled, the workspace snapshot proves at least
   one change attributable to this run.

Missing checks fail before the provider can edit. A model response never counts
as verification. Check output, events, and tool evidence are bounded and
credential-shaped values are redacted before JSONL persistence.

This is the *implementer* success contract. On the audit-required public route
it produces `awaiting_codex_audit`, not a landable result; the host verdict is
a separate, independent gate.

## Project configuration

Commit `.flyto/coding.yaml` with the codebase:

```yaml
version: flyto.coding-config.v1
checks:
  - name: unit
    argv: [python, -m, pytest, -q]
    timeout_seconds: 300
    required: true
  - name: lint
    argv: [python, -m, ruff, check, .]
    timeout_seconds: 120
    required: true
capabilities:
  - name: flyto-indexer
    kind: mcp-stdio
    argv: [python, -m, flyto_indexer.mcp_server]
    contract_version: flyto-indexer.mcp.v1
    protocol_version: 2025-06-18
    required_tools: [search, impact, task]
    allowed_tools: [search, impact, call_hierarchy, structure, task, verify]
    required: true
  - name: flyto-blueprint
    kind: mcp-stdio
    argv: [python, -m, flyto_ai.mcp_server]
    contract_version: flyto-blueprint.mcp.v1
    protocol_version: 2025-06-18
    required_tools: [list_blueprints, use_blueprint, save_as_blueprint, report_blueprint_outcome]
    allowed_tools: [list_blueprints, use_blueprint, save_as_blueprint, report_blueprint_outcome, export_blueprint, import_blueprint]
    required: true
  - name: flyto-page-inspector
    kind: mcp-stdio
    argv: [python, -m, flyto_ai.mcp_server]
    contract_version: flyto-page-inspector.mcp.v1
    protocol_version: 2025-06-18
    required_tools: [inspect_page]
    allowed_tools: [inspect_page]
    required: true
  - name: flyto-core
    kind: mcp-stdio
    argv: [python, -m, core.mcp_server]
    contract_version: flyto-core.mcp.v1
    protocol_version: 2025-06-18
    required_tools: [execute_module]
    allowed_tools: [list_modules, search_modules, get_module_info, get_module_examples, validate_params, execute_module, list_recipes, run_recipe]
    required: true
```

Configuration is declarative and argv-only. There is no shell expansion,
environment interpolation, or implicit sibling-repository import. Required MCP
servers are initialized and queried with `tools/list`; failure is closed before
editing. The negotiated protocol version and actual tool names must satisfy the
contract; the configured `contract_version` label is evidence metadata, not an
availability claim. `allowed_tools` is also enforced after discovery: tools not
listed there are not shown to the model and cannot be dispatched. When the field
is omitted, the backward-compatible behavior exposes the complete discovered
catalog. Required tools must be a subset of the allowlist. Optional adapters may
be removed independently.

## Built-in coding preset

The built-in Flyto2 coding preset is one control plane with four detachable
capability processes. It is a useful default, not the universe of supported
tasks:

```text
FlytoCodingAgent (provider-neutral owner)
  -> flyto-indexer: context, impact, dependency and task gates
  -> flyto-blueprint: discover/reuse/learn portable workflows
  -> flyto-page-inspector: inspect the real DOM before choosing selectors
  -> flyto-core: validate and execute modules/recipes, screenshots and visual diff
  -> source-controlled checks: accept or reject, then bounded repair
  -> redacted evidence and trusted Blueprint feedback
```

`flyto-page-inspector` is the page-detection component. It is intentionally a
one-tool view over `flyto_ai.mcp_server`; the same server process can provide the
Blueprint view without exposing `chat`, Core execution, or page inspection to
that capability. Core remains the runtime authority underneath page inspection
and provides `browser.detect`, screenshots, and the detachable deterministic
TypeScript visual-diff worker. `inspect_page` accepts a typed
`browser_channel` (`auto`, `chromium`, `chrome`, or `msedge`). The default
tries Core's bundled Chromium and then the installed Google Chrome, reports the
selected channel in its evidence, and still fails closed if neither launches.

Probe the exact installed composition without invoking a model, opening a page,
or reading credentials:

```bash
python -m flyto_ai.coding.stack --workspace . --json
```

The command performs real MCP initialize and `tools/list` handshakes, verifies
every required allowlisted tool, and emits `flyto.agent-stack.v1` evidence with
a content-addressed composition fingerprint. `--components` can remove any
lane; required missing lanes fail closed. The same capability tuple is available
to Python hosts through the stable coding API and plugs directly into the native
Agent request:

```python
from flyto_ai.coding import CodingTaskRequest, build_agent_stack_capabilities

request = CodingTaskRequest(
    message="Implement and verify the requested change",
    working_dir="/workspace/project",
    capabilities=build_agent_stack_capabilities(),
)
```

Passing a component subset makes every selected lane required by default. Use
`required_components` to make a selected lane optional.

## Domain-neutral agent profiles

The same stack contract can compose capabilities for general operations,
robotics, authorized security work, data workflows, or a future domain without
adding another component dictionary to `stack.py`. Commit a bounded profile as
`.flyto/agent-stack.yaml`:

```yaml
version: flyto.agent-stack.v2
profile: field-operations
capabilities:
  - name: mission-control
    kind: mcp-stdio
    argv: [python, -m, example_mission_mcp]
    contract_version: example.mission.v1
    protocol_version: 2025-06-18
    required_tools: [plan, verify]
    allowed_tools: [observe, plan, execute, verify, safe_stop]
    tool_permissions:
      observe: read_only
      plan: workspace_write
      execute: danger_full
      verify: read_only
      safe_stop: workspace_write
    required: true
```

Then preflight the exact source-controlled profile:

```bash
python -m flyto_ai.coding.stack \
  --workspace . \
  --manifest .flyto/agent-stack.yaml \
  --json
```

The loader accepts arbitrary safe capability and profile names, up to 64
capabilities. It rejects unknown fields, duplicate names, manifests outside the
workspace, oversized input, and MCP capabilities without a non-empty explicit
`allowed_tools` list. A v2 profile must also classify every allowed MCP tool
exactly once as `read_only`, `workspace_write`, or `danger_full`; omissions,
extra names, and unknown levels fail before process start. Existing v1 files
remain readable and retain their historical workspace-write default. Its
normalized manifest fingerprint and the real MCP composition fingerprint
provide separate evidence for configured intent and observed runtime state.

Python hosts can use `load_agent_stack_manifest()`,
`compose_capability_stack()`, and `probe_capability_stack()` directly.
`CapabilityManager` implements the generic `ToolExecutor` protocol, so an
initialized manager can attach to the ordinary `flyto_ai.Agent`; it is not tied
to `FlytoCodingAgent`. The latter remains the specialized adapter when a task
needs workspace snapshots, resumable coding threads, real repository checks,
and bounded repair.

Tool classification is not authority. The host constructs `CapabilityManager`
with an independent runtime ceiling (`read_only`, `workspace_write`, or
`danger_full`). Both the Agent's safe dispatcher and the manager's own
`dispatch()` enforce the effective requirement, so calling the manager
directly cannot bypass the outer Agent gate. Core `execute_module` retains its
argument-sensitive check after MCP name isolation: a shell, process, Docker,
Kubernetes, SSH, network, filesystem, environment, or Git module escalates to
danger-full even when the generic tool entry is workspace-write.

### Atomic runtime boundaries

The public imports remain `flyto_ai.coding.stack` and
`flyto_ai.coding.capabilities`, but their internals have one responsibility per
module:

- `stack_manifest` validates bounded workspace-local YAML, typed capabilities,
  v1/v2 policy, and the configured fingerprint.
- `stack_presets` builds only the four detachable built-in lanes.
- `stack_probe` negotiates real catalogs and hashes observed runtime identity.
- `mcp_transport` owns the isolated child environment, byte-bounded JSON-RPC,
  request correlation, timeouts, and deterministic shutdown.
- `mcp_catalog` owns provider-safe names, required/allowed tool scope, schemas,
  and machine-readable nested result status.
- `mcp_session` owns initialize, catalog negotiation, and tool-call
  orchestration over the transport and catalog atoms.
- `tool_registry` commits a complete session catalog transactionally and
  rejects provider-name collisions without leaving partial routes.
- coding `permissions` combines the host ceiling, declared requirement, and
  optional host-owned argument-risk resolver. Dynamic risk is monotonic and
  cannot lower the declared tier.
- `execution_policy` grants a bounded concurrency lease only after lifecycle
  call/failure/time, concurrency/queue, JSON byte/depth/node, secret-bearing
  key, configurable workspace-path, result, and approval-timeout gates pass.
- `execution_trace` stores a deeply immutable redacted content-addressed hash
  chain for Manager execution and Agent outer denials. Fixed-snapshot replay
  skips redacted inputs, defaults to read-only authority, requires explicit
  host opt-in for write/danger tiers, and emits trace-bound outcome feedback
  through a host-owned Blueprint sink.
- `conformance` validates one adapter's exhaustive permission and tool-case
  coverage, handshake, exact catalog, domain results, evidence chain, released
  policy leases, and shutdown; `scenario_matrix` aggregates arbitrary suites
  with bounded concurrency and no domain-name logic. Both default to read-only;
  controlled write/danger fixtures opt in and assert whether each case actually
  crossed the dispatch boundary.

`CapabilityManager` now only coordinates lifecycle, registration, permission
evaluation, and dispatch. Adding a new domain adapter does not require a task
name or tool name branch in that manager. A host can supply a risk resolver for
argument-sensitive operations such as physical actuation; the manifest cannot
install a resolver or raise runtime authority.

The closed-loop test matrix covers each boundary independently and together:

- pure contract, wire-codec, catalog, fingerprint, and permission tests;
- transactional collision and incomplete-mapping rollback tests;
- real MCP subprocess handshake, response correlation, domain-failure, and
  repeated open/dispatch/close tests with unraisable warnings promoted to
  errors;
- out-of-order concurrent response correlation, timeout/cancellation recovery,
  child crash fan-out, malformed/wrong-version/oversized wire responses,
  sustained stderr, deterministic property cases, and YAML alias/depth/node
  amplification boundaries;
- end-to-end conformance scenarios for general workflow, page inspection,
  robotics simulation, and an explicitly authorized inert security lab;
- outer Agent denial plus direct Manager bypass denial, policy admission,
  result budgets, cancellation lease release, trace exhaustion, safe replay,
  and trace-bound Blueprint feedback;
- full routing, Blueprint, coding, robotics, real four-lane preflight, and
  repository regression suites.

“Domain-neutral” does not mean unguarded universal execution. A production
domain still supplies its typed action contract, policy and authorization
checks, executor boundary, verifier, and evidence projection. Flyto2 already
has specialized adapters for general Agent workflows, coding, robotics
planning, and explicitly authorized footprint/pentest/red-team campaigns.
Physical actuation and security actions remain behind their respective safety,
scope, and human-authorization gates.

Hosts replace policy without editing dispatch code:

```python
from flyto_ai.coding import CapabilityManager, ExecutionLimits, ExecutionPolicy

policy = ExecutionPolicy(
    limits=ExecutionLimits(max_calls=500, max_concurrency=8),
    workspace_path_keys=("artifact_path",),
)
manager = CapabilityManager(workspace, execution_policy=policy)

# Read-only and unredacted events only by default.
report = await manager.replay_execution_trace()
# A controlled fixture may opt into additional tiers explicitly.
report = await manager.replay_execution_trace(
    allowed_permissions=("read_only", "workspace_write"),
)
```

Approval callbacks and Blueprint outcome sinks may be synchronous or
asynchronous. The host wait is bounded, callback exceptions are reduced to
stable non-secret errors, and malformed approval decisions fail closed.

Authenticated Flyto2 product adapters opt into runtime variables by name:

```yaml
  - name: flyto-cloud-workflows
    kind: mcp-stdio
    argv: [python, -m, mcp_server]
    protocol_version: 2025-06-18
    env_passthrough: [FLYTO_BACKEND_URL, FLYTO_API_KEY]
    required: true
```

Only explicit uppercase `FLYTO_*` names are accepted. The host supplies their
values at runtime; values never appear in YAML, capability status, evidence,
job records, logs, or tool schemas. The child still receives the small base
runtime environment (`PATH`, locale, terminal, and temp directory). Unrelated
AWS, GitHub, SSH, provider, and operating-system credentials are not inherited.
Removing `env_passthrough` detaches the secret-bearing integration without
changing the coding agent.

## Detachable service adapters

`flyto.coding-service.v2` exposes bounded asynchronous jobs behind one audited
route. `code-mcp` and `code-serve` are two transports over the same service,
not two products:

```text
authenticated loopback HTTP / configured-tenant MCP stdio
  -> CodingService (idempotency, queue, tenant, workspace policy)
  -> exactly one startup-selected implementer
       native  -> FlytoCodingAgent
       claude  -> ClaudeCodingAgent (optional adapter)
       codex   -> CodexCliCodingAgent (logged-in CLI adapter)
  -> real checks + evidence
  -> awaiting_codex_audit
  -> host/auditor verdict
  -> durable secret-redacted receipt
```

### Startup implementer selection

The operator picks the implementer once, when the process starts:

```text
--implementation-backend native|claude|codex      (default: native)
FLYTO_AI_CODING_BACKEND=native|claude|codex       (optional bounded default)
```

There is no per-job backend field, no provider/model auto-routing, and no
fallback among backends. An invalid or unavailable selection fails
startup. `--max-rework-rounds` (default 3) is a process option too; a remote
request cannot override either.

Selecting `claude` requires the optional `flyto-ai[claude-sdk]` extra; startup
fails with an actionable error if the SDK is absent. That route is pinned to
`claude-opus-5` for service work regardless of configuration, reads only
bounded `FLYTO_AI_CC_*` settings, and resolves no native provider credential.
Its tool catalog is Read/Edit/Write/Glob with write authority and Read/Glob
without; it never receives Bash, content search (`Grep`), or the audit tool, so
an implementer cannot approve its own work. Rework continues in the exact same
Claude SDK session; a changed or missing session identity fails closed.

Selecting `codex` requires an explicit bounded `--model`; `--codex-command`
can pin the installed executable. Startup verifies both without making a model
call. Each round runs a separate `codex exec` process with user configuration
and personal exec-policy rules ignored, no configured MCP/plugins/web search,
a scrubbed runtime environment, and only Codex's `read-only` or
`workspace-write` sandbox. Flyto binds the first structured `thread.started`
identifier durably, derives changes from host snapshots, runs the same pinned
checks, and stops at the same independent audit. Rework uses `codex exec
resume` on that exact id. A missing/changed id, malformed or oversized JSONL,
required unattached capability, unavailable check, or unexpected read-only
write fails closed.

### Host-owned route lanes

`flyto.coding-route.v1` (`flyto_ai/coding/route.py`) is a typed, provider-
neutral orchestration contract at the service boundary. Both public commands
enable it at startup; it is not a prompt convention and does not depend on
which implementer was selected.

```text
Indexer pre-work gate   mandatory: workspace context/structure, impact/task
                        planning, ordered plan-step execution, gates
Blueprint discovery     read-only, relevance-checked: a sanitized
                        content-addressed label, or a deterministic
                        not-applicable outcome
  -> selected implementer + required source-controlled checks
Core validation         allowlisted validation/discovery calls through
                        flyto_ai.tools.core_tools, closed by validate_params
Indexer post-work       mandatory: task.validate, task.gate.verify, and
                        verify.strict against the final workspace
```

Rules that hold on every public round:

- The implementer never asserts that a lane ran. Each outcome is derived from
  completed allowlisted calls; plan steps execute in order and a step naming
  anything outside the Indexer allowlist is refused.
- Step count, response bytes and depth, calls per lane, iterations, and gate
  remediations are all bounded. A `pass=false` gate blocks only its own phase
  and is remediated and re-gated inside that bound.
- A missing catalog, failed domain result, malformed evidence, incomplete
  required action or gate, exceeded bound, or unavailable Indexer fails the
  round closed with a stable code. It never reaches `awaiting_codex_audit`.
- Blueprint is read-only. `use_blueprint`, `save_as_blueprint`, and the
  export/import tools are outside its allowlist, so it can never execute a
  workflow or receive workspace authority.
- Core validation excludes `execute_module`, danger-full, and browser
  authority. Relevance is derived from the request and the attributable
  changed files; relevant work without an executable proof fails closed rather
  than being marked passed, and irrelevant work records a reasoned
  not-applicable receipt.
- Source-controlled checks remain the trusted host command lane. They are
  necessary but never substitute for the Indexer post-gate: a green check
  cannot mask a failed lane.

Startup options only *replace* a lane command; they never detach a lane.
`--indexer-command` and `--blueprint-command` default to this interpreter
running `flyto_indexer.mcp_server` and `flyto_ai.mcp_server`. Core validation
is always enabled on the strict route — there is no opt-in flag, because the
real Codex configuration does not pass one.

The pre-work lane calls the real public Indexer surface: `structure`,
`search`, `task(action="plan")`, the returned plan's own steps and gate steps,
then any remaining mandatory gate. A `pass=false` gate is remediated with real
`impact`/`search` evidence and re-gated; a state key needing human or external
authority fails closed instead of being asserted. Post-work refuses outright
if the implementer did not succeed or a required source-controlled check
failed, then runs `task(action="validate", run_tests=false)`,
`task(action="gate", next_phase="verify")`, and `verify(strict=true)`.

Every search this host issues — initial discovery, gate remediation, and a
translated plan step — carries the workspace project. The Indexer's smart
search otherwise fans out across every indexed project and enriches each hit,
which on a multi-project machine exceeded the 30-second capability bound and
failed the mandatory pre-work lane before the implementer could start. The
bound itself is unchanged; the query was narrowed. A plan step that names its
own project keeps it.

### Failure evidence

A lane that stops halfway is not a lane that did nothing. Each lane records a
bounded call trace, so a failed lane receipt keeps every completed call plus
one failed call naming the exact action the host derived:

```json
{"lane": "indexer_pre", "required": true, "status": "failed",
 "reason_code": "capability_timeout",
 "calls": [{"lane": "indexer_pre", "action": "structure", "ok": true,
            "detail_code": "context"},
           {"lane": "indexer_pre", "action": "search", "ok": false,
            "detail_code": "capability_timeout"}]}
```

Actions are host-derived and semantic (`task.plan`, `task.gate.<phase>`,
`task.validate`, `verify.strict`), never raw tool arguments. A transport
timeout is classified `capability_timeout` from a closed machine code the
capability adapter reports, not by parsing an error message, so it stays
distinguishable from `domain_failure` — an Indexer that answered and refused.
A contract failure that dispatched nothing records no call rather than an
invented action. The trace stays inside the configured `max_calls_per_lane`,
and any edit to it breaks the receipt digest.

### Emergency overflow lane

`flyto.coding-emergency.v1` (`flyto_ai/coding/emergency.py`) exists for one
situation: the route infrastructure itself is unreachable, so every job fails
before the implementer runs and the control plane would strand all coding.

It is startup authority, disabled unless an operator passes the flag:

```bash
flyto-ai code-mcp \
  --tenant acme --workspace-root /ABSOLUTE/PATH \
  --implementation-backend claude \
  --emergency-overflow-backend claude \
  --emergency-overflow-threshold 1
```

`--emergency-overflow-backend` must equal `--implementation-backend`, so
granting overflow can never redirect work to an implementer the operator did
not choose. There is no environment variable, job field, or model-reachable
switch. The default threshold is 1 because each Codex conversation runs its
own stdio process and many see only a single job; a higher per-process count
would still strand every one of them. Counters are per instance and are never
shared between processes or builds.

The lane opens only when *all* of these hold, checked before invocation:

- the strict route failed with `capability_unavailable` or `capability_timeout`
- in `indexer_pre` or `blueprint` — a lane that runs before the implementer
- the implementer was never invoked, in memory *and* in the durable record
- no attributable workspace change exists
- this process's breaker has reached its threshold

Everything else stays fail-closed: a domain refusal, blocked gate, stale index
needing remediation, malformed evidence, failed check, failed implementation,
Core validation failure, Indexer post failure, audit rejection, and rework
exhaustion never open it.

An emergency round calls the same startup-selected backend directly, keeps the
required source-controlled checks, binds the same exact revision, and still
ends at `awaiting_codex_audit` for an independent Codex verdict. It never
commits, pushes, or publishes. Before the model is called the host persists
`execution_mode=emergency` with the trigger lane, action, and stable code, so
a round that is still running — or that died mid-flight — is never read back
as an ordinary strict round.

Acceptance requires a separate authority contract. The round records **no**
`route_receipt`; instead it carries a digest-validated
`EmergencyAuthorityReceipt`:

```json
{"contract_version": "flyto.coding-emergency.v1", "mode": "emergency",
 "circuit_state": "open", "trigger_lane": "indexer_pre",
 "trigger_action": "search", "trigger_code": "capability_timeout",
 "implementer_backend": "claude", "instance_id": "...", "build_id": "...",
 "job_id": "job_...", "request_sha256": "...", "session_id": "...",
 "revision_sha256": "...", "implementer_started": true,
 "checks_enforced": true, "audit_required": true, "digest": "..."}
```

One verifier accepts exactly one authority: a digest-valid passed strict
receipt, or a digest-valid emergency receipt this service is configured to
honour, sealed to this job, request, session, and revision, whose required
checks really passed. Missing, mixed, transplanted, tampered, disabled,
wrong-backend, failed-check, and ordinary failed-route evidence all fail
closed. `CodingRouteReceipt(strict=True)` is untouched — a failed strict route
never becomes landable.

Rework after an emergency round stays on the same authority and the same
implementation session. It is legal only for a rework the service itself
scheduled for that same job, and an initial round carrying any pre-existing
authority fails closed before anything runs.

Recovery is a new process. The breaker is monotonic inside one process, so it
cannot oscillate between the strict route and the overflow lane while the
infrastructure flaps. A repaired build starts with a closed circuit and a new
build id; older instances stay visibly old in the status index.

### Runtime status

`flyto.coding-route-status.v1` (`flyto_ai/coding/route_status.py`) answers two
questions a restarted Codex cannot otherwise answer: where the round stopped,
and whether the implementer really started.

```text
<state root>/status/index.json              bounded instance index
<state root>/status/instance-<id>.json      one instance's latest status
```

Many Codex conversations share one state root, each with its own long-lived
`code-mcp` process, so a single latest-writer file would let an old process
overwrite a newer one. Each instance owns a file named by its own opaque
instance id and updates only its own row in the shared index, under the same
cross-process state guard, written atomically at mode 0600.

Every record carries the instance id, an immutable startup build digest over
the coding package and its bounded CLI/config/provider adapter dependencies,
the package version, process id, start
time, and `lifecycle` (`active` / `closed`). A graceful shutdown changes only
the lifecycle and timestamp, preserving the last job id, state, lane, action,
failure code, implementer-start, session, and revision. A crashed instance
keeps its last `active` row, which is why readers also consult the recorded
pid and the retention window.

The schema is closed and bounded. It contains no task message, error text,
working directory, file list, source content, environment, command line, or
credential; an unrecognised value degrades to empty rather than persisting.
The index is read back through the same closed schema under a byte bound, so a
malformed, oversized, duplicated, or unknown-field row is discarded and never
republished. Retention is deterministic: at most 32 instances, and an instance
silent for more than seven days is collected along with its file.

`implementer_started` is written to the durable job record immediately before
every real implementer invocation, never because a job entered `running`, and
it is exposed on the public `CodingJobReceipt` alongside `emergency_authority`
so `flyto_coding_get` and the HTTP receipt show the same truth. A round that
failed *after* implementation keeps its session id, attributable files, and
revision digest as proof that the model ran, while staying terminal,
non-landable, and non-auditable.

Publishing status is deliberately non-fatal to a real job, but a broken
recorder is not silent: failures are counted with a stable code
(`status_write_failed`, `status_validation_failed`), reported through
`CodingService.status_health()`, included in the next successful status
record, and announced once on stderr.

Inspect it read-only, without starting a service:

```bash
flyto-ai code-status --state-dir /ABSOLUTE/PATH/TO/STATE --json
```

It validates every row, annotates age and build staleness, reports whether a
live instance requires reload, probes pid liveness best-effort, and reports the
reader's own build id. A direct `code-mcp` whose startup build no longer matches
disk refuses new jobs with `service_reload_required` before mutation.

For long-lived Codex tasks, use `code-mcp-supervisor`. It owns the stable host
stdio connection and runs `code-mcp` as a replaceable child. With no active job,
a source-build change replaces only that child and replays the MCP handshake.
With a known non-terminal job, it preserves the worker and exact implementation
session, continues status/audit calls, and rejects only a new submission with
`service_reload_pending`; the next request after the job becomes terminal
performs the replacement. It never retries a request whose delivery is
uncertain. A pre-schema process still cannot publish a status row
retroactively; use the host's MCP reload operation once to migrate that old
connection to the supervisor.

### Route receipt

Every routed round carries an additive, secret-free `route_receipt` on the
public job receipt:

```json
{"contract_version": "flyto.coding-route.v1", "strict": true, "ok": true,
 "failure_code": "",
 "lanes": [{"lane": "indexer_pre", "required": true, "status": "applied",
            "reason_code": "completed",
            "calls": [{"lane": "indexer_pre", "action": "structure",
                       "ok": true, "detail_code": "completed"}],
            "gates_passed": ["verify"], "gates_failed": []}],
 "digest": "<sha256 over the lanes>"}
```

It carries no prompt, source, raw MCP output, absolute path, credential, or
model response. Lane coherence is validated on construction and revalidated on
deserialization, so a truncated or edited record fails closed. A strict receipt
cannot claim success without all four canonical lanes present, required, and
finished applied or not-applicable, and only a strict route that succeeded can
appear on a landable job receipt. A strict service revalidates the persisted
evidence whenever it reads an audit-ready, reworking, or accepted job back,
including after a restart, so proof that was removed or edited on disk refuses
the read instead of reading as landable.

### Audit state machine

An implementer success is never public success:

```text
queued -> running -> awaiting_codex_audit
                       |
                       +-- accept  -> codex_accepted  (landable = true)
                       |
                       +-- rework  -> rework_queued -> rework_running
                                        -> awaiting_codex_audit  (same job,
                                           same thread, same implementation
                                           session, new cumulative revision)

terminal, never landable:
  failed                       provider, check, session, or revision failure
  failed (service_restarted)   queued/running/rework work interrupted by restart

rejected without any state change:
  rework_limit_reached         the job stays awaiting_codex_audit on its
                               current exact revision, still non-landable,
                               and no new session is started
```

At `awaiting_codex_audit` the receipt binds `implementation_backend`, an opaque
`implementation_session_id`, and the exact `implementation_revision_sha256` — a
streaming digest over the cumulative attributable change set, recomputed live
before every verdict. The host/auditor independently inspects and tests that
workspace revision, then binds its verdict to that digest. A stale, wrong, or
concurrently mutated revision is rejected without mutating the job.

`require_changes` applies to that cumulative job revision. If a same-session
rework finishes with the adapter's exact `no_changes` result and all required
checks pass, the service may reuse the prior attribution only after re-proving
the session, tenant/job claim, sealed resume envelope, bounded file set, and
live content digest. Those cumulative paths still enter the normal Indexer
post-work validation and a new exact-revision Codex verdict; this is not a
route bypass and does not accept provider prose as evidence.

### Worktree-owned Core contract proof

The Core lane normally validates an installed module through the allowlisted
`search_modules -> get_module_info -> validate_params` sequence. A brand-new
plugin in the implementation worktree is not installed in the coding service
runtime yet. The service must not solve that by adding the worktree to its own
Python path: importing unaudited plugin code into the host would cross the
implementation boundary before Codex has reviewed it.

A repository may instead mark exactly one required verification check with
`proof_kinds: [flyto.core.module-contract.v1]`. The declaration comes from the
repository contract pinned before the provider starts. The implementation
adapter runs that exact command and returns a host-generated `CheckResult`.
Only one matching, required, passing result applies the Core lane; missing,
optional, duplicate, or failed evidence fails closed. The verifier itself is
expected to build and install the candidate wheel in an isolated environment,
then exercise Core discovery, metadata, and parameter validation there.
Provider text and an unpinned check with the same name are never proof.

A rework verdict must carry at least one typed bounded finding (stable code,
severity, message, optional evidence reference); an accept verdict must carry
none. Rework is bounded by `--max-rework-rounds`. Past that ceiling the request
is rejected before the record is touched — the job keeps its current exact
revision `awaiting_codex_audit` and non-landable, and no replacement session is
started. Rework that cannot be resumed in the original session after a restart
fails closed the same way. Only a valid accept on that exact revision can make
the job landable.

`landable` is eligibility evidence, never an action. Nothing in this service
stages, commits, pushes, publishes, or deploys, and the Claude adapter's
guardian denies those command classes as defense in depth.

### Audit surfaces

MCP stdio exposes exactly three tools — `flyto_coding_submit`,
`flyto_coding_get`, and `flyto_coding_audit` — and its `initialize` result
carries bounded instructions describing this loop. HTTP exposes the equivalent
authenticated routes:

```text
POST /v1/coding/jobs                      submit (bearer + Idempotency-Key)
GET  /v1/coding/jobs/{job_id}             poll
POST /v1/coding/jobs/{job_id}/audit       verdict (bearer)
      {"implementation_revision_sha256": "...",
       "verdict": "accept" | "rework",
       "findings": [ ... ]}
```

Neither surface accepts a model, provider, backend, or audit-authority field.
Unknown fields are rejected before the service mutates anything: 404 for an
unknown job, 400 for an invalid shape, 409 for a stale revision / wrong state /
exhausted rework, 429 when busy, 403 for a policy denial. Receipts stay
secret-redacted and never expose an evidence path or raw check output.

Verdicts come from the principal the host authenticates. The transport
validates shape and forwards; it cannot itself prove which principal is
calling, and it never makes the acceptance decision.

The host injects the provider, provider credentials, tenant ID, allowed
workspace roots, and state root when the process starts. A job accepts only the
versioned coding request. Fields such as `provider`, `api_key`, `tenant`, and
`auth_token` are rejected rather than persisted. HTTP requires a bearer token
and `Idempotency-Key`; the built-in server binds only to loopback because public
TLS, identity, quota, and organization policy belong at the Flyto2 Cloud edge.
MCP stdio receives its tenant from process configuration.

The state root may be shared by multiple `code-mcp` workers. This is required
because Codex starts a separate stdio MCP server for each conversation. The
service coordinates them with three bounded lock scopes: a short state guard
for idempotency and audit transitions, a per-job execution lease released by
the operating system after a crash, and a hashed per-workspace lock held only
while an implementation round can edit files. Starting a new MCP process never
fails merely because another conversation is live, and restart reconciliation
skips every job whose lease is still owned.

### Scope of this document

Everything below describes the `flyto_coding` adapter: one Codex-facing profile
over the shared, domain-neutral capability control plane. It is not the
universal core. Profiles for other domains compose arbitrary capability,
tool, and contract identifiers through the same generic contract without
inheriting any of the mechanisms here — no audit-required route, no worktree
claim, no session continuity — because those answer a question specific to
editing a repository. The shared layer never branches on which domain a
profile describes.

### Job-lifetime worktree ownership

Those three scopes each bound one round. An audited job needs one more, because
between "the implementer finished" and "the auditor read the tree" no round is
running and nothing else stops a second Codex frontend from editing the same
worktree — which would invalidate a revision that was never wrong.

An audit-required job therefore takes a durable **workspace claim** at submit,
after an idempotent replay has been ruled out, and holds it across `queued`,
`running`, `awaiting_codex_audit`, `rework_queued`, and `rework_running`. It is
released on `completed`, `codex_accepted`, a terminal failure, or an explicit
host abandon. Rework re-asserts the claim it already holds, so a job never
deadlocks behind itself.

A claim answers only when it is fully bound: the exact field set with valid
bounded values, a `workspace_sha256` equal to the digest of the tree being
queried, and an owner record that names the same job and resolves to the same
canonical worktree. Anything less is `unresolved` — missing keys fail closed
exactly like unknown ones, because a half-written claim proves nothing.

| Situation | Result |
| --- | --- |
| No claim, or a fully bound claim whose owner record proves the job settled | `free` — edit proceeds |
| Fully bound claim whose owner record is in a claim-owned state | `held` — `workspace_busy` |
| Corrupt, unknown version, missing/extra/invalid fields, digest mismatch, unreadable, unknown owner, or an owner record that does not bind back to this job and worktree | `unresolved` — `workspace_claim_unresolved` |

Ownership is asserted *before* a claim-owned state is published, inside the
same cross-process state guard. A transition that cannot prove exclusive
ownership fails rather than becoming visible, so a round can never reach
`awaiting_codex_audit` without a valid claim; the job settles `failed` instead.
Release only ever removes this job's own fully bound claim.

Only `submit` may create a claim from `free`, once, before its job record is
published. Every later transition — and every audit, on **both** verdicts —
*reasserts* an existing claim this exact tenant, job, and worktree already
hold. A claim that has vanished for a live job is `unresolved`, not
reacquirable: during the missing interval another Codex could have taken the
worktree, edited files outside this job's attributable set, settled, and
released, and recomputing only this job's files would not detect it.
Reacquiring would manufacture continuity that was never established. A claim
carrying the same job id under a different tenant is never taken over. The way
out is the host spillway below, never an automatic repair.

`workspace_busy` names the owning job in bounded MCP structured error details
(`{"ok": false, "error": "workspace_busy", "details": {"owner_job_id": "..."}}`),
so an operator knows which job to audit. Only the opaque job id is published;
paths and prompts never are.

An `unresolved` claim is never cleared automatically, including by startup
reconciliation. Deleting a claim the service cannot evaluate would turn
"ownership is unknown" into "nobody owns this tree", which is precisely the
hazard the claim prevents. It requires an explicit host decision.

Different worktrees are unaffected: claims are keyed by workspace digest, so
parallel jobs across repositories keep running concurrently. A legacy
non-audited service takes no claim (it has no audit gap) but still honours one,
so it can never edit a worktree mid-audit.

### Cross-worker rework

An audit may arrive at a worker that never implemented the job — a different
Codex frontend, or a restarted one. A bounded, redacted **resume envelope**
under `tenants/<ref>/resume/<job>.json` (mode 0600) makes that closeable. It
stores only the public request fields plus job, request-digest, and session
bindings, and is loadable only when its `session_bound` equals the record's
`implementation_session_id`. The rebuilt request always carries `resume=true`
and that same session id, so the envelope can continue a Claude session but
never start one. The stored digest is compared rather than recomputed, because
redaction rewrites prose and a recomputed hash would never match.

Startup authority is never persisted. Approval policy, sandbox mode, config
path, sandbox image, checks, and capabilities are re-imposed from the running
process, so a stored request cannot outlive or widen the policy it ran under.

### The host-owned release valve

`flyto-ai code-release` is the only way to retire an orphaned audit or clear an
unresolved claim. It is deliberately not a fourth MCP tool: the audited route
keeps exactly `flyto_coding_submit`, `flyto_coding_get`, and
`flyto_coding_audit`, and nothing reachable by a model may retire a job.

```bash
# Fail an orphaned audit-ready job closed and release its worktree.
flyto-ai code-release --tenant acme --workspace-root /srv/workspaces/acme \
  --abandon-job job_0123456789abcdef01234567

# Clear a claim whose authority cannot be evaluated.
flyto-ai code-release --tenant acme --workspace-root /srv/workspaces/acme \
  --repair-workspace /srv/workspaces/acme/repo
```

Both operations are strictly subtractive. `--abandon-job` moves only
`awaiting_codex_audit` to `failed` with `job_abandoned` and `landable: false`;
it cannot accept, land, or let a round skip its audit, so it is always worse
for the caller than auditing and can never be a bypass. `--repair-workspace`
refuses while a live job owns the tree.

### Bounded supervisor reads

`code-mcp-supervisor` deadlines every worker read at 30 seconds, for both
requests and the replayed handshake. Submit, get, and audit only schedule or
inspect background work, so a longer wait is a wedged worker, not a slow one. A
missed deadline returns a bounded JSON-RPC `-32603` and terminates the worker
so the state-root locks it held are released; the request is never retried,
because its delivery is uncertain and the job may already exist. A caller
recovers by replaying the same idempotency key, which the supervisor must never
do on their behalf. The next request starts a fresh worker, whose startup
reconciliation reports any interrupted job truthfully.

Hot-reload tracking is self-healing from durable job records rather than from a
process-local set alone: a client that stops polling cannot pin
`service_reload_pending` forever. While a tracked job is genuinely non-terminal
the worker is preserved and only new submissions are refused; once every
tracked job reads terminal on disk, the worker is replaced and the handshake
replayed without restarting Codex.

Start either transport without putting a credential on the command line:

```bash
export FLYTO_AI_CODING_SERVER_TOKEN='use-a-runtime-secret-manager'
flyto-ai code-serve \
  --tenant acme \
  --workspace-root /srv/workspaces/acme \
  --implementation-backend native \
  --provider ollama \
  --max-rework-rounds 3 \
  --sandbox-image flyto/coding-python@sha256:REPLACE_WITH_LOCAL_DIGEST

# Claude implementer; needs the optional flyto-ai[claude-sdk] extra.
flyto-ai code-mcp \
  --tenant acme \
  --workspace-root /srv/workspaces/acme \
  --implementation-backend claude \
  --max-rework-rounds 3
```

The HTTP token is read only from the named environment variable. For cloud
providers, use their normal runtime environment variable; service commands do
not expose `--api-key`. The image must already exist locally, and production
configuration should pin its digest. Remote jobs may set only message,
workspace, thread/resume, attempt/round bounds, and whether an attributable
change is required. Checks and capabilities come only from the repo config that
the service loads before the provider round starts.

Jobs are serialized per workspace across service processes and bounded within
each process. Repeating the same idempotency key and request returns the
original job even through a different MCP process; reusing the key for a
different request fails. Tenant directories use one-way tenant references, and
a lookup never scans another tenant. An unleased interrupted queued/running job
becomes `service_restarted` on startup instead of being reported as successful;
a live leased job is left untouched.

## Security boundary

- All file paths resolve beneath one real workspace root; absolute paths,
  `..` escape, and symlink escape are rejected.
- Native mode exposes only `read-only` and `workspace-write`; it has no
  unrestricted filesystem mode.
- Codex CLI mode likewise exposes only `read-only` and `workspace-write`,
  never danger-full. The implementation child ignores personal config/rules,
  receives no Flyto MCP or audit tool, and inherits no provider or CI token.
- Model-issued development commands use `execve`-style argv dispatch inside a
  detected OS sandbox (a pinned local Docker image or `bwrap` on Linux). They can
  read installed tooling and the workspace, but network access and workspace or
  host writes are denied; only an ephemeral runtime home is writable. If no
  supported sandbox backend exists, `coding_run` fails closed.
- Source-controlled checks are the only trusted unsandboxed command path. They
  still receive a credential-scrubbed environment and temporary home, and are
  bounded by argv length, timeout, and output size.
- Privilege/network-transfer/destructive executables and shell command strings
  are denied. Credential files and VCS internals cannot be read, listed, or
  written through native file tools.
- Pre-existing dirty files are hashed before the run so evidence does not
  misattribute unrelated user work.
- Threads use atomic metadata writes and append-only JSONL events. Resume is
  allowed only in the original workspace.
- Coding-service state uses atomic mode-0600 records and a single-process lease;
  receipts survive restart, while in-flight work fails closed. Provider secrets
  remain process configuration and are absent from job files.
- MCP subprocesses inherit no ambient application environment except the
  bounded base set and explicit `FLYTO_*` passthrough names. Secret values are
  neither reflected nor persisted.

The command sandbox is defense in depth, not a complete hostile-repository
boundary: source-controlled checks intentionally execute project code. Run the
whole coding process in a dedicated container or VM for untrusted repositories.

## Connecting an auditing host over MCP

An orchestrator such as Codex reaches this service as a local STDIO MCP server.
Project-scoped Codex configuration lives in `.codex/config.toml` and applies
only in projects you trust. Ask the host to reload MCP configuration after the
initial command change; subsequent coding-source repairs are handled inside the
supervisor without restarting the whole host. The example below is
documentation: it uses placeholders and contains no credential value.

```toml
# .codex/config.toml — project-scoped; trusted projects only.
[mcp_servers.flyto_coding]
# Use an absolute path to the interpreter or console script for this project,
# for example /absolute/path/to/.venv/bin/flyto-ai.
command = "/ABSOLUTE/PATH/TO/.venv/bin/flyto-ai"
args = [
  "code-mcp-supervisor",
  "--tenant", "REPLACE_WITH_TENANT",
  "--workspace-root", "/ABSOLUTE/PATH/TO/WORKSPACE",
  "--state-dir", "/ABSOLUTE/PATH/TO/STATE",
  # Claude implementer; requires the flyto-ai[claude-sdk] extra.
  "--implementation-backend", "claude",
  "--max-rework-rounds", "3",
]
cwd = "/ABSOLUTE/PATH/TO/WORKSPACE"
required = true
enabled_tools = ["flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit"]
startup_timeout_sec = 30
tool_timeout_sec = 900
```

Native alternative — swap the two selection arguments, or drop them and set the
bounded environment default before Codex starts the server:

```toml
args = [
  "code-mcp-supervisor",
  "--tenant", "REPLACE_WITH_TENANT",
  "--workspace-root", "/ABSOLUTE/PATH/TO/WORKSPACE",
  "--implementation-backend", "native",
  "--provider", "ollama",
]
```

```bash
export FLYTO_AI_CODING_BACKEND=native   # or: claude
```

Credentials never belong in this file. Provide them through the runtime
environment or a secret manager for the process that Codex launches.

## Rollback and composition

Rollback is configuration, and it stays inside the audited route:

- Remove a capability entry to detach that MCP server.
- Select `--implementation-backend native` to detach the Claude adapter.
- Lower `--max-rework-rounds` to tighten the repair ceiling.

Each of these keeps `code-mcp` and `code-serve` audit-required. Stopping the
service **pauses** host-managed implementation until it is restarted; it does
not move that work to another path.

`flyto-ai code` and direct Python `CodingService` construction (which keeps
`require_codex_audit=False`) remain supported for legacy and library use, but
they sit outside the host-managed audited route. They cannot produce its
`codex_accepted` state or its `landable` evidence, and they are never the
fallback when the audited service is unavailable.

Removing `flyto_ai.coding` does not alter provider interfaces, Core execution
authority, Blueprint contracts, or the legacy Claude adapter.

## Host-global workspace root authority

A state root brokers the jobs inside it. It cannot broker a directory tree,
because its workspace claims live under itself: point two services at two state
roots and each keeps a private, self-consistent opinion about the same
checkout, and both edit it.

A host-global registry sits above every state root and owns each canonical
repository tree. Configured workspace roots remain the outer admission
boundary, but are not leased as one broad concurrency unit. A normal job leases
the nearest Git boundary containing `working_dir`; a cross-repository request
may declare an atomic set of up to sixteen real, non-overlapping Git roots
inside that boundary. Ownership is demand-scoped: an idle service holds
nothing; admission acquires the whole job set before the first durable
non-terminal mutation, and restart reacquires the exact persisted sets needed
by existing open work. Queued, running, rework, and audit-pending work keep the
ownership. A terminal transition releases that repository while retaining sets
another job still needs. A bounded observer also releases the submitter's
leases when another compatible process on the shared state root performs the
final transition. A newcomer takes each entry exclusively first; getting it
proves nobody is alive, and only then may the recorded owning state root be
written or rotated.
Ancestor and descendant overlap are refused in both directions, decided
atomically under a registry-wide lock with a bounded acquisition deadline.

The registry lives at `$XDG_STATE_HOME/coding-workspace-authority` (or
`~/.local/state/...`), outside every worktree. `CODING_WORKSPACE_AUTHORITY_ROOT`
overrides it at startup only, for isolated tests; no job payload reaches it.

An admission refusal occurs before continuation, job lease, workspace claim,
job/idempotency record, resume envelope, status mutation, or provider contact.
An idle service can therefore remain available without reserving a tree it is
not using.

### Three refusal classes

They are separate because their remedies are, and conflating them sends an
operator after a problem that does not exist.

| Condition | Code | Retryable | Remedy |
| --- | --- | --- | --- |
| Another state root owns an overlapping tree | `workspace_authority_conflict` | no | `code-workspace-status` to identify the owner |
| Registry mid-transaction past the deadline | `workspace_authority_busy` | yes | retry shortly |
| Registry missing, damaged, or unlockable | `workspace_authority_unavailable` | no | `code-workspace-status` to inspect the registry |

Each has its own worker exit status, and the MCP client receives one short
fixed sentence selected by exit code alone — never a path, prompt, raw error,
or job content. The public MCP inventory stays exactly `flyto_coding_submit`,
`flyto_coding_get`, `flyto_coding_audit`.

### Unified task window

```bash
flyto-ai code-task-window --state-dir /ABSOLUTE/PATH/TO/STATE
flyto-ai code-task-window --state-dir /ABSOLUTE/PATH/TO/STATE --limit 100 --json
```

This local read-only command is the shared coordination window for many Codex
frontends. It shows immutable main-axis digests, side/repair return edges,
current scheduler rank, an optional safe `owner_ref`, path-free repository
digests, implementation-session presence, stable failure codes and audit/rework
counters. It does not show the task prompt, objective prose, repository path,
evidence, worker identity or provider session id. Snapshot identifiers grant no
mutation authority and the projection is never injected into another model's
conversation.

Local and CI cross-stack dependency authority comes from
`stack-lock.json`. `scripts/stack_lock.py --workspace-parent ..` proves
the three sibling checkouts equal the manifest; GitHub Actions derives those
same checkout refs from the manifest before running the suite.

### External coding watchdog

`code-status` and `code-task-window` make failures inspectable, but a dead
Codex cannot inspect itself. `code-watchdog` is the independent, deterministic
observer for that gap. It never invokes an LLM and has no path to submit,
audit, abandon, repair, commit or push anything. It only reads the two bounded
host projections above and writes:

```text
~/.flyto/health/coding/latest.json    current secret-free aggregate health
~/.flyto/health/coding/history.jsonl  transition-only, size-rotated history
~/.flyto/health/coding/github.json    last successful remote heartbeat cursor
```

Records contain aggregate counts, stable reason codes, the local reader build
digest and timestamps. They exclude prompts, repository paths, job/session
identifiers, source, commands, evidence, credentials and provider responses.
An executing task with no provably live owner becomes `critical` after the
short orphan grace. Slow live execution, overdue Codex audit, a rolling stale
build, status-recorder failure or emergency-spillway activation is
`degraded`. No live Codex process is healthy when there is no stranded work;
the observer never keeps a model process alive merely to satisfy itself.

Run one observation or install the macOS LaunchAgent:

```bash
flyto-ai code-watchdog --state-dir ~/.flyto/coding-service --json
flyto-ai code-watchdog --install --notify \
  --state-dir ~/.flyto/coding-service \
  --github-repository OWNER/REPOSITORY
flyto-ai code-watchdog --uninstall --state-dir ~/.flyto/coding-service
```

The health directory and the state root must be disjoint. Neither may be the
other or live inside it, and `--install` refuses the same overlap the one-shot
run refuses (`watchdog_paths_overlap`). The observer would otherwise write
health records into the durable coding-service tree it is forbidden to mutate,
and would then observe its own writes as route activity.

Both roots are resolved through their symlinks before that comparison, and the
same resolved state root is what derives the LaunchAgent label. A path is
compared as a directory, never as a spelling: an unresolved comparison lets
`--health-dir` reach inside the state root through a link while the guard sees
two unrelated strings, and lets `--install` and `--uninstall` compute two
different labels for one state root — an uninstall that reports success and
removes nothing while the agent keeps waking.

`--install` validates every value it bakes into the plist — polling interval,
stuck and orphan thresholds, heartbeat interval, repository, and variable name
— against exactly the bounds the observing run applies. An option that the
run path would reject can never be installed, because an agent that fails on
every unattended wake is indistinguishable from no watchdog at all.

The LaunchAgent runs once per minute by default. Unchanged healthy runs update
`latest.json` but do not append history. A GitHub heartbeat is sent at most
once every five minutes unless the fingerprint changes. It is written with a
single `gh variable set` upsert through the already authenticated `gh` CLI, so
a missing and an existing `FLYTO_CODING_HEARTBEAT` take the identical path; no
token is placed in the plist or health files. The projection is refused
locally if it ever exceeds GitHub's 48 KB variable limit rather than being
silently truncated remotely.

`state_readable` judges the status index by the publisher's own
`MAX_STATUS_INDEX_BYTES`, not by the watchdog's smaller record limit. The
writer owns that bound; applying a stricter one here would report a large but
perfectly valid index as a route failure and manufacture an incident.

The health directory is created `0o700`, but `--health-dir` is operator-supplied
and may point under a world-writable parent, so the watchdog does not assume it
owns every name inside it. `latest.json`, `github.json`, `history.jsonl` and
`watchdog.lock` are opened `O_NOFOLLOW`, rotation tests each name with `lexists`
so a planted link is rotated away instead of blocking every later append, and
reads measure and drain a single descriptor rather than checking a name and then
reading it. A refused history append reports `watchdog_history_unwritable`, and
it is raised only after `latest.json` is already durable.

Nothing secondary may cost the turn its local record. A hung `gh`, and a
`github.json` cursor that cannot be written after the heartbeat has already been
published, are both recorded as a `github_heartbeat` warning — the latter as
`github_state_unrecordable` — rather than allowed to end the run. Losing the
send-interval bookkeeping only means the next turn republishes an unchanged
heartbeat; losing the record means the remote switch reads `healthy` while the
local evidence a human would inspect was never written.

`.github/workflows/coding-watchdog.yml` is the external dead-man switch. Every
15 minutes it checks that the heartbeat is healthy and no older than 45
minutes. Failure opens or refreshes one labelled issue and fails the Actions
run; recovery closes that issue. This remote witness catches the case the
local machine or LaunchAgent dies. Runs are serialized but never cancelled: the
job's product is an incident, so a dispatch arriving between "the heartbeat is
stale" and "open the issue" must not cancel the only step that reports it. It uses ordinary deterministic Actions, not
GitHub Agentic Workflows, so healthy polling consumes no Claude, Codex,
Copilot or Gemini quota. AI diagnosis can be added after an incident without
placing an AI inside the liveness path.

The workflow treats the repository variable as untrusted input, because any
actor who can write repository variables can set it. Before anything is
rendered it bounds the raw size, requires an object with the exact heartbeat
schema, requires `observed_at` to be a plain in-range integer, requires
`health` to be one of `healthy`, `degraded`, `critical`, and keeps only reason
codes matching `[a-z][a-z0-9_]*`. The emitted `reason` is then re-checked
against a single-line allowlist before it reaches `GITHUB_OUTPUT`. Without
that last check a newline inside any rendered field would let the variable
append its own `healthy=true` line and silence the dead-man switch it exists
to trip. Each rejection has its own code — `heartbeat_missing`,
`heartbeat_oversized`, `heartbeat_invalid`, `heartbeat_schema_invalid`,
`heartbeat_timestamp_invalid`, `heartbeat_health_invalid`,
`heartbeat_clock_invalid`, `heartbeat_stale` — so an operator can tell a bad
publisher from a dead host. A malformed heartbeat is never optimistically
treated as healthy.

The first release is alert-only. Automatic job abandonment, workspace repair,
service restart, audit acceptance and code mutation are intentionally absent;
operators use the existing explicit, subtractive recovery commands after
reading the stable reason codes.

### Diagnosing and recovering

```bash
# Read-only. Starts no service, joins no authority, mutates nothing.
flyto-ai code-workspace-status --workspace /path/to/tree
flyto-ai code-workspace-status --workspace /path/to/tree --json
```

It reports every bounded overlapping owner with its relationship (`exact`,
`owner_is_ancestor`, `owner_is_descendant`), status, and owning state root, in
deterministic order. The headline status is the most blocking overlap, so an
adoptable exact entry never masks a live parent.

| Status | Meaning | Next step |
| --- | --- | --- |
| `unregistered` | nobody has claimed it | start normally |
| `live` | a process holds the lease now | stop that process |
| `crashed_with_open_work` | owner died leaving non-terminal jobs or claims | finish or retire them under *that* state root |
| `adoptable` | owner gone, nothing unresolved | the next start adopts it |

Recovery never involves editing a registry file. For a stranded audit under the
previous owner, use the subtractive host release valve, which does not join,
rotate, or adopt workspace authority:

```bash
flyto-ai code-release --state-dir <owner state root> --abandon-job <job id>
flyto-ai code-release --state-dir <owner state root> --repair-workspace <tree>
```

Once that state root has no non-terminal job and no surviving workspace owner
claim, the tree reports `adoptable` and the next start takes it.

## Workspace-claim authority and the generic claim kernel

Two claim mechanisms exist in this repository. They are **not** interchangeable,
and exactly one of them is authoritative for a worktree.

`CodingService`'s own workspace claim is the authority. It is the mechanism the
audited route depends on: it binds a worktree to one job for the whole audit
gap, it is understood by `repair_workspace_claim`, `_sweep_workspace_claims`
and the runtime status row, and its semantics are covered by the ownership
suites. Nothing in this section changes it.

`flyto_ai.orchestration.resource_claims.ResourceClaimStore` is a
domain-neutral, multi-process claim kernel: content-addressed records,
authority-aware resolution, no automatic stealing, no TTL, and a strict
fail-closed posture on hosts that cannot supply atomic publication or a real
inter-process lock. It is a general primitive. **It currently has no production
consumer.** It is exported from `flyto_ai.orchestration` and exercised by
`tests/test_resource_claims.py`, and that is the whole of its present role.

It is deliberately not wired into the workspace claim yet. Two independent
stores over one worktree can disagree, and the dangerous direction of that
disagreement is the cheap one to reach: the generic store answers `free` for a
resource the service still owns, and a second job edits a tree whose revision
an auditor is about to read. Introducing that risk to remove a duplication is
the wrong trade while the audited route is the thing being repaired.

The follow-up, when it is taken, is an **adapter, not a replacement**:

- Model the resource as `ResourceRef(namespace=<state-root digest>,
  kind="workspace", identity=<workspace digest>)`, so the kernel never learns a
  path or any caller vocabulary.
- Supply an `OwnerAuthority` that resolves an owning job id against the durable
  job record: `held` for a state in `_CLAIM_OWNED_STATES`, `released` for a
  terminal state, `missing` for a job record that is gone, and `unknown` for a
  record that cannot be read. Ambiguity must stay ambiguous - the kernel already
  refuses to release on anything but a positive `released`.
- Run it in shadow first: acquire and release alongside the existing claim and
  assert the two never disagree, with the service claim still deciding. Promote
  only after the shadow has been silent across the ownership suites.
- Never let a generic-store answer widen access. If the two disagree, the
  worktree is busy.

Until that adapter exists and has been proven, treat any statement that the
claim kernel solves concurrent dispatch as false. It solves the primitive; the
integration is unwritten.

## Cross-job continuation

A bounded provider stop keeps its session, and a second job may re-enter it only
through an explicit `submit` carrying `resume=true` and the exact SDK session.
The MCP surface is unchanged: still exactly `flyto_coding_submit`,
`flyto_coding_get`, `flyto_coding_audit`, and the continuation request uses the
`thread_id`/`resume` fields the submit schema already had.

State lives under the service state root, partitioned by tenant:

    <state_root>/tenants/<tenant_ref>/continuation/<sha256(session)>.json
    <state_root>/tenants/<tenant_ref>/continuation/<sha256(session)>.journal

The `.json` body is the current authority; the `.journal` is an append-only
hash-chained transition log. A body that does not match the journal tail is a
replay and reads as absent. Lookups happen inside the authenticated caller's own
partition, so a guessed session from another tenant is indistinguishable from one
that never existed.

Admission re-proves, before any provider contact: backend, workspace identity and
path, snapshot policy identity, authorized verification contract, the exact
attributable revision, and the whole-workspace snapshot. Refusals are bounded
codes; everything that could confirm somebody else's authority exists collapses to
`continuation_unavailable`.

Public projection is two additive receipt fields, `continuation_available` and
`continuation_generation`. The authority record, its generation history and the
canonical workspace path stay private.

### Threat limit, stated honestly

This is durability and concurrency safety, not a cryptographic signature. An actor
who can rewrite the entire owner-owned state root *and* the workspace together is
not excluded by digests the same account can recompute; what is excluded is
replay, double-spend, silent drift, partial writes, symlink redirection, and the
ordinary consequences of a crash or a race.

## Cumulative plan authority across rework rounds

The strict route keeps one root Indexer task for the whole life of a job.

- After a successful `indexer_pre`, the exact returned contract is sealed into
  `indexer_plan_authority` on the private job record: schema version, owning job,
  root request digest, workspace digest, the contract, and a content digest over
  all of it. It is bounded (`MAX_PLAN_AUTHORITY_BYTES`) and never appears in a
  receipt, a prompt, a log, an error message or an audit body.
- A rework re-proves every binding before the implementer is invoked, then sends
  the contract back as `task_contract` on `task(action="plan")`. Absent parent
  means the argument is omitted entirely.
- The cumulative attributable set is proven before the proof lanes and is the
  same ordered tuple that `task.validate.current_state.changed_paths`, the
  persisted `implementation_files` and `implementation_revision_sha256` all
  bind. Equality is enforced, not inclusion.

Failure codes are closed (`PLAN_AUTHORITY_CODES`) and report `verification` or
`workspace` phase with `resubmit_against_current_contract`. They are terminal
for the job and release its claim.
