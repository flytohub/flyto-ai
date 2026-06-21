# AI MCP Closed Loop

`flyto-ai` owns the agent-side bridge from provider tool calls to `flyto-core` MCP execution.

## Flow

```text
Agent request
  -> provider tool choice
  -> ToolRegistry
  -> flyto_ai.tools.core_tools
  -> flyto-core MCP handler
  -> validation + execution
  -> provider log entry + evidence metadata
  -> blueprint/eval feedback
```

## Contract

The `get_core_capability_manifest` tool exposes:
- `source`: `flyto-core`
- `contract_version`: `flyto-core-mcp.v1`
- `core_version`: installed package version when available
- `tool_fingerprint`: stable digest for drift checks
- `recipes_available`: recipe support flag
- `approval_model`: runtime approval notes
- `tools`: optional per-tool metadata
- `categories`: optional module category counts

## Safety

- `execute_module` validates params before execution when `flyto-core` exposes `validate_params`.
- Provider logs carry `mcp.source`, `mcp.contract_version`, and module or recipe identity.
- Credentials stay runtime-only and must not be requested through MCP elicitation.
- `flyto-core` remains the runtime authority and is not modified from this repo.

## Verification

```bash
python -m compileall flyto_ai
python -m pytest tests/test_core_mcp_contract.py tests/test_mcp_server.py tests/test_browser_retry.py -q
```
