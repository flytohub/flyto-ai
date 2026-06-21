# flyto-ai Agent Rules

- `flyto-ai` is the agent/runtime adapter between LLM providers, Flyto Cloud, blueprint learning, and `flyto-core` MCP tools.
- Do not store credentials, API keys, cookies, bearer tokens, screenshots containing secrets, or user passwords in source, tests, docs, handoffs, evidence, or generated YAML.
- Treat `flyto-core` as the module/runtime authority. Read its MCP schemas and recipes, but do not modify `flyto-core` from this repo.
- All `flyto-core` execution must flow through `flyto_ai.tools.core_tools` so validation, browser retry, permission checks, and MCP evidence metadata stay consistent.
- New agent capabilities need a closed loop: typed contract, guardrail, evidence/trace, test, docs, and rollback notes.
- Keep `.flyto-index/`, transcripts, evidence, build output, local DBs, and eval results out of commits.
- For deep changes, run flyto-indexer impact/search/verify before finalizing.
