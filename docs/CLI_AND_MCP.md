# CLI, HTTP, And MCP

The generated [CLI reference](reference/cli.md) is extracted from argparse and
lists every command and option.

## CLI Modes

- `flyto-ai` or `flyto-ai interactive`: interactive chat.
- `flyto-ai chat <message>`: one request with provider/model, YAML-only plan,
  JSON output, memory, sandbox, webhook, and round controls.
- `flyto-ai code <message>`: bounded coding-agent execution with optional Core
  verification recipe, reference image, attempts, budget, and turn limits. This
  is direct legacy/library use; it sits outside the audited service route below
  and cannot produce an accepted, landable receipt.
- `flyto-ai code-serve` and `flyto-ai code-mcp`: the two transports of the one
  audit-required coding service. `flyto-ai code-mcp-supervisor` is the stable
  local MCP facade that safely hot-reloads its inner worker. See
  [Audited coding service](#audited-coding-service).
- `flyto-ai blueprints`: list or export learned workflows.
- `flyto-ai prompt-lab eval|evolve|cases|report`: evaluate and evolve prompts
  without automatically promoting results.
- `flyto-ai serve`: local HTTP/SSE service; default bind is loopback.
- `flyto-ai mcp` or `flyto-ai-mcp`: JSON-RPC MCP over STDIO.
- `flyto-ai version`: package/dependency versions and runtime-discovered module
  count when Core is available.

API keys passed on the command line can appear in shell history/process listings;
environment or secret-manager injection is preferred.

## HTTP Service

The optional `serve` extra uses `aiohttp` when available and a standard-library
fallback otherwise. Server authentication uses `FLYTO_AI_SERVER_KEY`; production
must configure an explicit key and origin allowlist. The service supports health,
chat/streaming, Claude command execution, steering, status, and Telegram webhook
integration as implemented in `flyto_ai.cli`.

## Audited coding service

`flyto-ai code-serve` (authenticated loopback HTTP) and `flyto-ai code-mcp`
(configured-tenant MCP stdio) are two transports over one `CodingService`. Both
are audit-required unconditionally; there is no flag or environment variable
that disables the requirement.

Codex tasks can keep stdio MCP processes alive after the source tree changes.
Configure `code-mcp-supervisor` for that host: it fingerprints the coding source
at each request boundary, preserves any known non-terminal job and its exact
session, rejects only additional submissions with `service_reload_pending`, and
replaces the inner worker automatically once the active job is terminal. A
direct stale `code-mcp` worker independently rejects a new submission with
`service_reload_required` before it creates a job record.

The operator selects the implementer once at startup with
`--implementation-backend native|claude` (default `native`, optional bounded
`FLYTO_AI_CODING_BACKEND` default) plus `--max-rework-rounds` (default 3). There
is no per-job backend selection and no fallback between implementers; an
invalid or unavailable selection fails startup. The `claude` route needs the
optional `flyto-ai[claude-sdk]` extra, is pinned to `claude-opus-5` for service
work, reuses the exact same SDK session across rework, and receives no Bash,
no content search, and no audit tool.

An implementer round ends at `awaiting_codex_audit` bound to an exact
`implementation_revision_sha256`. The authenticated host independently inspects
and tests that workspace revision, then submits a verdict bound to that digest:
`accept` reaches `codex_accepted` with `landable` evidence, while `rework`
returns typed findings to the same job and implementation session for another
bounded round.

A provider or check failure is terminal `failed`. Work interrupted while queued
or running becomes `failed` with `failure_code=service_restarted`. A rework
request past the ceiling is rejected before any record change (HTTP 409 /
`rework_limit_reached`): the job stays at its current exact revision,
`awaiting_codex_audit` and non-landable, and no new session starts. Only a
valid `accept` on that exact revision can make it landable. The service never
stages, commits, pushes, publishes, or deploys; `landable` is evidence, not an
action.

Surfaces:

| Transport | Operation |
| --- | --- |
| MCP tool | `flyto_coding_submit` |
| MCP tool | `flyto_coding_get` (legacy `job_id`-only full receipt; optional compact conditional polling) |
| MCP tool | `flyto_coding_audit` |
| HTTP | `POST /v1/coding/jobs` (bearer + `Idempotency-Key`) |
| HTTP | `GET /v1/coding/jobs/{job_id}` |
| HTTP | `POST /v1/coding/jobs/{job_id}/audit` (bearer) |

Neither surface accepts a model, provider, backend, or audit-authority field.
The MCP `initialize` result advertises server version `3` and bounded
instructions describing this loop. Details, the state machine, and a
project-scoped Codex `.codex/config.toml` example are in
[the coding control plane guide](CODING_CONTROL_PLANE.md).

For polling, first call `flyto_coding_get` with `detail: "summary"`. Its
`observation.change_token` can be sent back as `after_change_token` with
`wait_ms` from 0 through 20,000. The call then waits only while background work
can advance and the compact projection is unchanged. Audit-ready, blocked,
accepted, and terminal jobs return immediately with a typed `next_action`.
Sending only `job_id` still returns the exact historical `ok` plus full `job`
response without an observation sibling. Fetch that full receipt before
independently auditing its exact implementation revision. An observation is
added only after opting in with an explicit detail or conditional-read argument.

## MCP Protocol

This section describes the two general-purpose MCP servers: the general server
and the closed-loop server. It does **not** describe the coding MCP service
above, which is a separate surface that negotiates exactly one protocol
version, `2025-06-18`, and rejects anything else. Do not assume the coding
service supports the stateless 2026 protocol or the legacy handshake range.

Both general MCP servers support the stateless 2026-07-28 protocol and the older
handshake-based revisions from 2024-11-05 through 2025-11-25. Modern hosts can
discover capabilities without opening a sticky protocol session, so reconnects
do not depend on hidden server state. Every modern response identifies the
server, and discovery and tool lists include safe cache guidance.

The general server exposes registered tools plus `chat`. The closed-loop server
keeps large plans and evidence outside the model context while exposing four
bounded tools: `plan`, `execute`, `verify`, and `get_evidence`. Invalid metadata,
unsupported versions, methods, and parameters return structured JSON-RPC
errors.

Hosts should call `tools/list` rather than cache a hard-coded Core module count.
`tools/call` executes through the same registry, validation, permission, retry,
and evidence boundaries used by the agent.

The built-in MCP client tries modern `server/discover` first and automatically
falls back to the legacy initialize handshake when it connects to an older
server.
