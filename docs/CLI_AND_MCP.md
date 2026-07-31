# CLI, HTTP, And MCP

The generated [CLI reference](reference/cli.md) is extracted from argparse and
lists every command and option.

## CLI Modes

- `flyto-ai` or `flyto-ai interactive`: interactive chat.
- `flyto-ai chat <message>`: one request with provider/model, YAML-only plan,
  JSON output, memory, sandbox, webhook, and round controls.
- `flyto-ai code <message>`: bounded coding-agent execution with optional Core
  verification recipe, reference image, attempts, budget, and turn limits.
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

## MCP Protocol

Both MCP servers support the stateless 2026-07-28 protocol and the older
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
