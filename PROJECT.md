# Project

`flyto-ai` turns natural language into schema-validated Flyto2 automation. It is the agent layer that chooses `flyto-core` MCP modules, executes them through guarded adapters, learns reusable blueprints, and exposes the same contract to Flyto2 Cloud.

Primary users:
- Flyto2 Cloud users invoking AI assistant features.
- Developers using `flyto-ai` CLI or MCP server.
- Agent workflows that need deterministic tool execution instead of generated shell code.

Current priority:
- Keep `flyto-core` MCP integration provider-agnostic, validated, observable, and safe for Cloud UI consumption.

Shipped surfaces:
- Python package facade and provider-agnostic agent runtime.
- Interactive, batch, coding-agent, blueprint, prompt-lab, HTTP/SSE, Telegram, and MCP entrypoints.
- OpenAI-compatible, Anthropic, Ollama, and ordered failover providers.
- Registry-backed Flyto2 Core tools with schema validation, permission checks, retry, evidence, and runtime capability discovery.
- Judge-drawn Mission Station interpretation and structured Robotics planning;
  both are advisory/attested model boundaries, not execution authorization.
- Memory, transcript, blueprint learning, prompt evaluation/evolution, scheduling, channels, extensions, vault, sandbox, and security-workflow generation.

Documentation contract:
- `docs/documentation-manifest.json` maps every source area and product surface to owner documentation and tests.
- `scripts/generate_reference.py` inventories all Python functions/classes/methods plus CLI, tools, environment, and maintainer scripts.
- `.github/workflows/documentation.yml` rejects broken links, stale generated output, missing test paths, retired branding, or non-`flyto2.com` public email domains.
