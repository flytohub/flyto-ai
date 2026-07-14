# Architecture Map

## Core Areas

- `flyto_ai/agent.py`, `flyto_ai/assistant/`, and `flyto_ai/orchestration/`:
  prompt routing, sub-agent coordination, approvals, resilience, and agent
  execution flow.
- `flyto_ai/tools/core_tools.py`: the only supported adapter path from
  `flyto-ai` into `flyto-core` MCP modules and recipes.
- `flyto_ai/providers/`: OpenAI, Anthropic, Ollama, and failover adapters.
  Providers may request tool calls, but they must not call `flyto-core`
  directly.
- `flyto_ai/prompt/`, `flyto_ai/redaction.py`, `flyto_ai/permissions.py`, and
  `flyto_ai/vault.py`: AI governance, prompt-injection guardrails, permission
  checks, redaction, and local secret handling.
- `flyto_ai/memory/`, `flyto_ai/evolution/`, and `flyto_ai/intelligence/`:
  reusable memory, blueprint learning, prompt evolution, scoring, and planning.
- `flyto_ai/mcp_server.py` and `flyto_ai/mcp_client.py`: MCP-compatible server
  and client entry points for external tool/runtime integration.
- `docs/`, `workflows/`, and `handoffs/`: project memory, release process, and
  handoff evidence.

## Cross-Repo Edges

- `flyto-core` is the execution/runtime authority. `flyto-ai` consumes core
  capability manifests, module schemas, `validate_params`, recipes, and
  execution results through `flyto_ai.tools.core_tools`.
- `flyto-cloud` consumes `flyto-ai` assistant, app automation, marketplace,
  workflow, crawler, and template-agent capabilities through stable contracts.
- `flyto-code` and `flyto-engine` consume AI governance, fix reasoning, evidence
  narration, and policy/runtime decisions without importing provider-specific
  code directly.
- `flyto-indexer` verifies source context, impact, security, prompt/audit
  hygiene, and Flyto2 product-line release evidence.
- `flyto-blueprint`, `flyto-pro-core`, and `flyto-pro` provide learning,
  extension, and commercial module capabilities that must remain optional for
  community/open-core surfaces.

## Provider Boundary

```text
Flyto2 Cloud / CLI / MCP client
  -> flyto-ai agent / orchestration
  -> provider adapter
  -> tool-call request
  -> ToolRegistry
  -> flyto_ai.tools.core_tools
  -> flyto-core MCP module or recipe
  -> structured result + evidence metadata
```

- Hosted providers are adapters, not product authorities.
- Provider prompts and responses must pass through redaction, prompt safety,
  permissions, and evidence logging boundaries.
- Local or airgapped deployments must be able to replace hosted providers with
  local endpoints or rules-only operation without changing `flyto-core`.

## Product-Line Role

- Flyto2 Cloud / Apps / Automation: agent app building, crawler automation,
  workflow assistance, template generation, and marketplace flow reasoning.
- Flyto2 Security: AI governance, code/security fix reasoning, evidence
  explanation, redteam consent messaging, and report narrative support.
- Flyto2 Data: future dataset, knowledge-base, vector/search, and data
  governance agent workflows.
- Flyto2 Zero-person Company Agent: operating-system layer for research,
  content, support, sales, development, monitoring, and reporting tasks.
- Flyto2 Big Data / Intelligence: large-scale summarization, trend synthesis,
  threat/brand/GEO visibility analysis, and intelligence report generation.

## Release Invariants

- `flyto-ai` must not duplicate `flyto-core` module schemas or bypass
  `flyto_ai.tools.core_tools`.
- Provider-specific code must not leak into `flyto-cloud`, `flyto-code`, or
  `flyto-engine` product gates.
- Prompt, evidence, memory, and provider logs must not store secrets or
  cross-tenant data.
- Enterprise/airgap mode must have a local-provider or rules-only path and must
  not require external egress by default.
