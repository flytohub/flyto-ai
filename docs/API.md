# Python API Guide

Install `flyto-ai` and import the package facade:

```python
from flyto_ai import Agent, AgentConfig, create_agent

agent = create_agent(provider="openai", model="gpt-4o-mini")
response = await agent.chat("Open example.com and extract the title")
```

The package exports agent/configuration, chat/stream/usage models, permission
controls, provider/tool protocols, and a lazy `ClaudeCodeAgent`. Exact signatures
for every public and internal symbol are in the
[generated reference](reference/python/README.md).

## Main Contracts

- `AgentConfig`: provider/model, limits, memory, sandbox, coding-agent, failover,
  budget, transcript, vault, injection, permission, browser, and Pro flags.
- `Agent`: synchronous/streaming chat orchestration, deterministic plan, provider
  loop, tool validation/execution, recovery, and learning feedback.
- `ChatRequest`/`ChatResponse`: normalized message, mode, result, tool calls,
  workflow, and usage data.
- `StreamEvent`: stable progress/tool/result/error events for CLI, HTTP, Cloud,
  and channels.
- `ApiClient`/`ToolExecutor`: protocols for substitutable provider and tool
  implementations.

## Tools

`ToolRegistry` owns definitions and handlers. Built-ins include user elicitation,
page inspection, website navigation, blueprint operations, and the Flyto2 Core
bridge. Core definitions are loaded lazily and may differ by installed Core
version; use `get_core_capability_manifest` instead of assuming a count or list.

## Compatibility

Public classes/functions and serialized model fields are compatibility surfaces.
Underscore-prefixed symbols are documented for maintenance but may change without
the same compatibility guarantee. Provider-specific raw payloads must not leak
into common model contracts.

## Adaptive security campaigns

`flyto_ai.security.run_security_campaign` accepts a versioned campaign request
and a provider-neutral planner callable. The planner may be a real LLM adapter;
its proposed steps still enter the existing MCP `plan`, `execute`, and `verify`
path before Core can run them.

```python
from flyto_ai.security import run_security_campaign

campaign = {
    "campaign_id": "staging-pentest-2026-07",
    "mode": "pentest",
    "objective": "Validate the approved staging exposure.",
    "target_scope": ["staging.example.com"],
    "authorization": {
        "level": "exploit",
        "reference": "AUTH-2026-0001",
        "expires_at": "2026-08-01T00:00:00Z",
        "approved_actions": ["active_probe", "exploit_validation"],
    },
    "module_allowlist": ["http.request", "security.sqli_probe"],
    "budgets": {
        "max_steps": 10,
        "max_requests": 20,
        "max_rounds": 3,
        "max_planner_tokens": 50000,
        "max_cost_units": 100,
    },
}

result = await run_security_campaign(campaign, planner)
```

The planner receives only the objective, authority ceiling, remaining budgets,
and a bounded `flyto.security-planner-evidence.v1` projection. Raw target
content and secrets are not placed back into the model prompt. Active steps
must name an in-scope target and include assertions. Scope, authorization
expiry, action class, module allowlist, and cumulative budgets are rechecked at
runtime. Consumers must treat only `verified=True` / `verdict="proved"` as
closed; all other results are `not_proved`.

## Structured Robotics planner

`flyto_ai.robotics_planning.RoboticsPlanningService` accepts
`flyto.robotics.planner-request.v1` and returns:

```json
{
  "contract_version": "flyto.ai.robotics-plan-response.v1",
  "plan": {
    "contract_version": "flyto.robotics.plan.v1",
    "generated_by": {
      "kind": "llm",
      "provider": "flyto-ai",
      "model": "flyto-qwen3-8b"
    }
  },
  "attestation": {
    "contract_version": "flyto.ai.robotics-planning-attestation.v1",
    "mode": "live_llm",
    "request_sha256": "...",
    "schema_sha256": "...",
    "plan_sha256": "...",
    "selected_route_id": "orange-purple"
  }
}
```

The caller provides capability argument definitions, the routed shortlist,
semantic location IDs, and optional route candidates. The service enforces:

- request size at most 256 KiB, at most 64 capabilities, 32 routes, and 32
  plan steps;
- exact shortlist/capability parity and bounded argument schemas;
- route candidates as exact, complete step templates when using semantic
  navigation;
- recursively forbidden actuator/control fields;
- unique step IDs, paired `ask_human`/`resume`, and terminal `safe_stop` for
  motion;
- no more than two provider attempts.

Providers implement `StructuredJsonProvider.complete_json_schema`. The native
Ollama adapter is currently live-tested. The loopback HTTP adapter supports
`GET /health` and `POST /v1/robotics/plan`; it is not a remotely authenticated
API.
