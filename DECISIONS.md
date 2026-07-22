# Decisions

## 2026-07-22: Generate exhaustive implementation references

- Human-authored guides explain behavior, boundaries, and operations; generated references provide exhaustive symbol/CLI/tool/environment inventories.
- Package version comes from `pyproject.toml` in a source tree and installed package metadata in a wheel, preventing CLI/MCP/version drift.
- Core module totals are discovered from the installed registry at runtime; source code does not freeze a fallback count.
- CI validates the documentation manifest and rejects stale generated output.

## 2026-06-21: flyto-core stays the MCP authority

- `flyto-ai` adapts `flyto-core` tools instead of duplicating module metadata.
- The adapter adds metadata and validation but preserves existing tool names and result shapes.
- Cloud should consume `flyto-ai` capability manifests rather than importing `flyto-core` internals.

## 2026-06-21: Agent Builder is not a dependency

- Agent Builder concepts can inform workflow UX, but product code stays code-first and provider-agnostic.
- Durable primitives are MCP, typed tools, traces, evals, guardrails, approvals, and evidence.
