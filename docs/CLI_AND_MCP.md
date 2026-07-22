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

The server supports the protocol versions listed in `SUPPORTED_PROTOCOL_VERSIONS`
and returns its package version from the same metadata source as the CLI. It
advertises tool support, exposes registered tools plus `chat`, returns JSON-RPC
errors for invalid methods/params, and processes newline-delimited STDIO without
requiring an external MCP library.

Hosts should call `tools/list` rather than cache a hard-coded Core module count.
`tools/call` executes through the same registry, validation, permission, retry,
and evidence boundaries used by the agent.

