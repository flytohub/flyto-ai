# The AI → MCP Closed Loop

This is the part that turns “the model said it worked” into “the runtime can
show what ran.”

## Flow

```text
Agent request
  -> deterministic Blueprint match OR provider tool choice
  -> ToolRegistry
  -> flyto_ai.tools.core_tools
  -> flyto-core MCP handler
  -> permission + validation + PlanIR gate
  -> execution + checkpoint/repair/assertions
  -> allowlisted runtime evidence
  -> host-verified Blueprint Evidence Card
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
- A direct model call to `report_blueprint_outcome` records only a deduplicated
  community observation. The Blueprint loop attaches an in-process object
  capability after guarded execution; a JSON tool call cannot forge its object
  identity, so only that path records `local_verified` evidence.
- Selecting or expanding a Blueprint without any module execution evidence
  produces no score update.
- The trusted runtime report sends only duration, step/attempt counts,
  assertion result, workflow hash, executor version, and selection mode.
  Prompts, params, credentials, and raw results are excluded.
- Deterministic exact reuse sets `planner_model_calls_used=0` with
  `model_call_scope=planner`. Model-selected execution leaves the count unknown
  instead of guessing. This proves that the outer agent skipped re-planning; it
  does not claim that an `llm.*` workflow step used zero tokens. Blueprint can
  still read the older `model_calls_used` field for compatibility, but new
  Flyto2 AI reports do not emit it.
- Portable `export_blueprint` and `import_blueprint` calls never accept signing
  keys or trusted-publisher mappings. Unsigned imports are quarantined by
  `flyto-blueprint`.

## What you can measure

The resulting Blueprint Evidence Card shows:

- trusted run count, successes, failures, and observed success rate;
- Wilson 95% lower bound, so a tiny sample does not look more certain than it is;
- retry rate and assertion pass rate;
- duration p50 and p95;
- measured zero-model-call reuse count and rate.

It intentionally does not estimate “tokens saved.” If exact provider token
counters are added later, they can be reported as another measured field.

## Verification

```bash
python -m compileall flyto_ai
python -m pytest tests/test_core_mcp_contract.py tests/test_mcp_server.py tests/test_browser_retry.py tests/test_blueprint_closed_loop.py -q
```
