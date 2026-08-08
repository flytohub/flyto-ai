# Project

`flyto-ai` turns natural language into schema-validated Flyto2 automation. It is the agent layer that chooses `flyto-core` MCP modules, executes them through guarded adapters, learns reusable blueprints, and exposes the same contract to Flyto2 Cloud.

In the governing product topology it is the unified AI gateway/SDK for both `flyto-cloud` and `flyto-engine`: both product columns converge here, and the platform chain below is LLM providers -> `flyto-blueprint` -> `flyto-core`. The canonical topology lives in [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`docs/architecture-map.md`](docs/architecture-map.md). Its standing invariant: `flyto-cloud` and the combined `flyto-code` / `flyto-engine` product column stay parallel at the same product-plane level, and `flyto-code` / `flyto-engine` are never nested inside or drawn beneath Cloud.

Primary users:
- Flyto2 Cloud users invoking AI assistant features.
- Developers using `flyto-ai` CLI or MCP server.
- Agent workflows that need deterministic tool execution instead of generated shell code.
- Codex, or another authenticated orchestrator, driving the audited coding route.

Current priority — one audited coding route:
- Codex reaches implementation only through the `flyto-ai` coding service (`code-mcp` / `code-serve`), which is audit-required.
- The operator selects the implementer once at startup, `native` or `claude`. There is no per-job override and no fallback between them.
- When `claude` is selected, Claude implements; Codex never implements and never approves its own work.
- Codex independently inspects and tests the exact workspace revision, then binds its verdict to that revision digest.
- A rejected revision resumes the same implementation session with typed findings, bounded by a startup rework ceiling.
- Only an accepted exact revision becomes caller-landable. The service itself never stages, commits, pushes, publishes, or deploys — landability is evidence for the caller.

Standing priority:
- Keep `flyto-core` MCP integration provider-agnostic, validated, observable, and safe for Cloud UI consumption, with Core as the execution authority and Blueprint learning bounded to redacted evidence.

Shipped surfaces:
- Python package facade and provider-agnostic agent runtime.
- Interactive, batch, coding-agent, blueprint, prompt-lab, HTTP/SSE, Telegram, and MCP entrypoints.
- The `flyto.coding-service.v2` audited coding service: startup backend selector, MCP and HTTP audit boundary, bounded same-session rework, stable implementation-session identity, and revision-bound landability receipts.
- OpenAI-compatible, Anthropic, Ollama, and ordered failover providers.
- Registry-backed Flyto2 Core tools with schema validation, permission checks, retry, evidence, and runtime capability discovery.
- Memory, transcript, blueprint learning, prompt evaluation/evolution, scheduling, channels, extensions, vault, sandbox, and security-workflow generation.

Current gaps:
- `flyto-engine` still contains a direct OpenAI provider path, so routing every product through this gateway is a migration target rather than a completed state.
- Live Claude-backend MCP execution is not yet verified in the active environment: the optional `claude-agent-sdk` was absent at audit time, so that adapter is covered by in-process fakes only.
- Universal `flyto-modules-*` registration with Core and full four-repository runtime closure are not claimed. See the dated alignment snapshot in [`docs/architecture-map.md`](docs/architecture-map.md) and the evidence detail in [`STATE.md`](STATE.md).

Documentation contract:
- `docs/documentation-manifest.json` maps every source area and product surface to owner documentation and tests.
- `scripts/generate_reference.py` inventories all Python functions/classes/methods plus CLI, tools, environment, and maintainer scripts.
- `.github/workflows/documentation.yml` rejects broken links, stale generated output, missing test paths, retired branding, or non-`flyto2.com` public email domains.
