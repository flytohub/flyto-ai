# Architecture

Runtime flow:

```text
user/cloud/CLI
  -> Agent
  -> provider chat loop
  -> ToolRegistry
  -> flyto_ai.tools.core_tools
  -> flyto-core MCP handler
  -> structured result + evidence metadata
  -> blueprint/eval/trace feedback
```

Key boundaries:
- Providers never call `flyto-core` directly.
- Cloud imports `flyto-ai` contracts and dispatchers, not `flyto-core` internals.
- Blueprint learning stores successful tool chains and failure evidence, not secrets.
- MCP metadata is additive: existing tool names, schemas, and result shapes stay compatible.

Core contract:
- `get_core_capability_manifest` reports contract version, installed core version, tool fingerprint, recipes support, module categories, and per-tool risk metadata.
- `execute_module` validates params before execution when `flyto-core` exposes `validate_params`.
- Tool logs include `mcp.source`, `mcp.contract_version`, and module or recipe identity.
