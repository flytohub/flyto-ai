# Flyto2 coding control plane

`flyto_ai.coding` is a provider-neutral coding loop. The native implementer uses
the configured Flyto2 provider (OpenAI, Anthropic, Ollama, or a compatible
adapter) and requires no vendor agent SDK. An optional Claude adapter is its
peer behind the same contracts; the operator selects exactly one at startup.

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
  -> real checks + evidence
  -> awaiting_codex_audit
  -> host/auditor verdict
  -> durable secret-redacted receipt
```

### Startup implementer selection

The operator picks the implementer once, when the process starts:

```text
--implementation-backend native|claude      (default: native)
FLYTO_AI_CODING_BACKEND=native|claude       (optional bounded default)
```

There is no per-job backend field, no provider/model auto-routing, and no
fallback in either direction. An invalid or unavailable selection fails
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

Jobs are serialized per workspace and bounded across the service. Repeating the
same idempotency key and request returns the original job; reusing the key for a
different request fails. Tenant directories use one-way tenant references, and
a lookup never scans another tenant. An interrupted queued/running job becomes
`service_restarted` on restart instead of being reported as successful.

## Security boundary

- All file paths resolve beneath one real workspace root; absolute paths,
  `..` escape, and symlink escape are rejected.
- Native mode exposes only `read-only` and `workspace-write`; it has no
  unrestricted filesystem mode.
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
Project-scoped Codex configuration lives in `.codex/config.toml`, applies only
in projects you trust, and needs a Codex restart to take effect. The example
below is documentation: it uses placeholders and contains no credential value.

```toml
# .codex/config.toml — project-scoped; trusted projects only.
[mcp_servers.flyto_coding]
# Use an absolute path to the interpreter or console script for this project,
# for example /absolute/path/to/.venv/bin/flyto-ai.
command = "/ABSOLUTE/PATH/TO/.venv/bin/flyto-ai"
args = [
  "code-mcp",
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
  "code-mcp",
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
