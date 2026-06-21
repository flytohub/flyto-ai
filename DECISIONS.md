# Decisions

## 2026-06-21: flyto-core stays the MCP authority

- `flyto-ai` adapts `flyto-core` tools instead of duplicating module metadata.
- The adapter adds metadata and validation but preserves existing tool names and result shapes.
- Cloud should consume `flyto-ai` capability manifests rather than importing `flyto-core` internals.

## 2026-06-21: Agent Builder is not a dependency

- Agent Builder concepts can inform workflow UX, but product code stays code-first and provider-agnostic.
- Durable primitives are MCP, typed tools, traces, evals, guardrails, approvals, and evidence.
