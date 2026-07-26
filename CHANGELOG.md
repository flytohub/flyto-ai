# Changelog

## Unreleased

- Fixed closed-loop MCP verification so a missing identifier, an unknown
  `plan_id`, and a valid plan without execution evidence return distinct
  structured errors.
- Added a documentation contract, feature/API/configuration/operations guides,
  technical whitepaper, and generated references covering every declared
  Python function/class method, CLI option, static tool, environment read, and
  maintainer script.
- Unified package, CLI, and MCP version reporting and changed Core module totals
  to runtime discovery so installed capabilities cannot drift from source text.
- Added version/capability regression tests and a reusable documentation CI gate.
- Prepared a metadata-only PyPI patch release so live registry backlinks,
  project URLs, and runtime-discovered capability wording replace stale
  hard-coded module totals.
- Refactored OpenAI provider chat tool-call dispatch into a shared helper
  pipeline with direct regression tests for text completion, tool dispatch, and
  `ask_user` pause handling.
- Added ruff and flyto-indexer verify steps to the CI release loop.
- Split prompt-evolution mock response generation into category-specific
  helpers with regression tests for adversarial, partial-failure, language, and
  workflow responses.
- Added README usage guidance, `.env.example`, and prompt package docs so
  project documentation passes the local verify gate.
- Added flyto-core MCP capability manifest support.
- Added per-tool MCP metadata for risk, approval policy, and evidence fields.
- Added pre-execution parameter validation for `execute_module`.
- Added MCP evidence metadata to provider tool-call logs.
- Added CI workflow and `.flyto-index/` ignore.
- Added repo memory and workflow handoff scaffold.
- Added `docs/architecture-map.md` so Flyto2 workspace release packets can
  verify `flyto-ai` cross-repo architecture and product-line boundaries.
