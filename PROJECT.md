# Project

`flyto-ai` turns natural language into schema-validated Flyto2 automation. It is the agent layer that chooses `flyto-core` MCP modules, executes them through guarded adapters, learns reusable blueprints, and exposes the same contract to Flyto2 Cloud.

Primary users:
- Flyto2 Cloud users invoking AI assistant features.
- Developers using `flyto-ai` CLI or MCP server.
- Agent workflows that need deterministic tool execution instead of generated shell code.

Current priority:
- Keep `flyto-core` MCP integration provider-agnostic, validated, observable, and safe for Cloud UI consumption.
