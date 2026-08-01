# Flyto2 coding control plane

`flyto_ai.coding` is a provider-neutral coding loop. It uses the configured
Flyto2 provider (OpenAI, Anthropic, Ollama, or a compatible adapter) and does not
require Codex or Claude Agent SDK. Claude SDK support remains an optional,
detachable compatibility backend.

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

“Domain-neutral” does not mean unguarded universal execution. A production
domain still supplies its typed action contract, policy and authorization
checks, executor boundary, verifier, and evidence projection. Flyto2 already
has specialized adapters for general Agent workflows, coding, robotics
planning, and explicitly authorized footprint/pentest/red-team campaigns.
Physical actuation and security actions remain behind their respective safety,
scope, and human-authorization gates.

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

`flyto.coding-service.v1` exposes the same native loop as asynchronous jobs. It
is an optional composition layer, not a second coding agent:

```text
authenticated loopback HTTP / configured-tenant MCP stdio
  -> CodingService (idempotency, queue, tenant, workspace policy)
  -> FlytoCodingAgent
  -> real checks + evidence
  -> durable secret-redacted receipt
```

The host injects the provider, provider credentials, tenant ID, allowed
workspace roots, and state root when the process starts. A job accepts only the
versioned coding request. Fields such as `provider`, `api_key`, `tenant`, and
`auth_token` are rejected rather than persisted. HTTP requires a bearer token
and `Idempotency-Key`; the built-in server binds only to loopback because public
TLS, identity, quota, and organization policy belong at the Flyto2 Cloud edge.
MCP stdio receives its tenant from process configuration and exposes only
`flyto_coding_submit` and `flyto_coding_get`.

Start either adapter without putting a credential on the command line:

```bash
export FLYTO_AI_CODING_SERVER_TOKEN='use-a-runtime-secret-manager'
flyto-ai code-serve \
  --tenant acme \
  --workspace-root /srv/workspaces/acme \
  --provider ollama \
  --sandbox-image flyto/coding-python@sha256:REPLACE_WITH_LOCAL_DIGEST

flyto-ai code-mcp \
  --tenant acme \
  --workspace-root /srv/workspaces/acme \
  --provider ollama
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

## Rollback and composition

Remove a capability entry to detach that server. Select the optional
`claude-sdk` backend when compatibility is required. Removing
the HTTP/MCP service process restores direct `flyto-ai code` use without a data
migration. Removing `flyto_ai.coding` does not alter provider interfaces, Core
execution authority, Blueprint contracts, or the legacy Claude adapter.
