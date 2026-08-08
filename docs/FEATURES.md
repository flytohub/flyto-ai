# Feature And Package Map

Flyto2 AI converts natural-language requests into guarded, schema-validated
tool calls and reusable workflows. This map describes shipped source areas;
exact functions and methods, including internal helpers, are listed in the
[generated Python reference](reference/python/README.md).

## Why this is not just another agent loop

| | Model-first agent | Flyto2 AI |
|---|---|---|
| Repeated task | Calls the model again | Tries deterministic Blueprint reuse first |
| Execution trust | Often trusts the generated plan | Preflight, permission, validation, PlanIR, checkpoint, repair, assertions |
| Learning | More chat context | Parameterized workflow plus measured outcomes |
| Reliability view | Read traces by hand | Evidence Card |
| Token claim | Usually estimated | Zero re-planning calls counted only at the measured planner boundary |

The point is not that Flyto2 AI never uses an LLM. It uses one where judgment helps,
then moves stable repeated work out of the model path.

## Agent Runtime

| Package | Responsibility |
|---|---|
| `agent` | provider chat loop, deterministic planning, tool dispatch, validation, memory/blueprint feedback, streaming, and usage aggregation |
| `assistant` | pre-routing, middleware, recovery, choice detection, interaction pauses, safety masking, bounded history, and output tracking |
| `intelligence` | deterministic intent extraction, registry/blueprint planning, execution plans, selector resolution, and optional Pro bridge |
| `models`, `protocols`, `testing` | shared request/response/stream contracts, dependency ports, and test doubles |

The deterministic planner runs before provider fallback when enabled. Provider
responses do not execute arbitrary generated shell/Python; tools are selected
from registered definitions and module parameters are validated before dispatch.

## Providers And Failover

`providers` supplies OpenAI-compatible, Anthropic, Ollama, and ordered failover
adapters behind one async contract. It normalizes content, tool calls, usage,
stream events, truncation, and evidence logging. Custom base URLs pass SSRF
policy validation. Budgets and cost records can stop a session before further
provider spend.

## Flyto2 Core And MCP

- `tools.core_tools` lazily discovers Flyto2 Core MCP tools and enriches them
  with risk, approval, and evidence metadata.
- `execute_module` validates parameters when Core exposes `validate_params` and
  applies browser retry/session recovery at the dispatch boundary.
- capability manifests report installed Core version, tool fingerprint,
  categories, current module count, recipe support, and approval semantics.
- `mcp_server` exposes JSON-RPC 2.0 over STDIO with initialize, tools/list,
  tools/call, and a `chat` meta-tool.
- `mcp_client` manages subprocess lifecycle, negotiation, tool discovery,
  request correlation, timeout, and restart state for external MCP servers.

The [tool/MCP reference](reference/tools-and-mcp.md) lists static definitions;
Core registry modules are intentionally runtime-discovered instead of copied.

## Memory, Learning, And Evaluation

| Package | Responsibility |
|---|---|
| `memory` | SQLite sessions/messages, BM25, optional embeddings, hybrid search, compaction, and summarization |
| `cache`, `session`, `transcript` | prompt result cache, session lifecycle, bounded/rotating replay evidence |
| `evolution` | eval cases, scoring rubric, prompt blocks, mutation/crossover, multi-round evaluation, regression detection, and reports |
| Blueprint integration | saves successful multi-step workflows, reuses repository-compatible patterns, exchanges portable bundles, and feeds trusted/community evidence through `flyto-blueprint` |

Prompt evolution never auto-applies a candidate to the production prompt. It
writes candidates and reports for human review.

Blueprint improvement is evidence-backed workflow learning rather than model
weight training. Direct model outcome reports are community observations and
cannot change the trusted score. Guarded Blueprint execution carries an
in-process capability that records local verified evidence after validation,
permission checks, execution, and assertions. Export/import is explicit;
signing and trusted-publisher configuration remain host-only.

The trusted report includes an allowlisted execution summary: duration, step
count, total attempts, assertion result, workflow hash, executor version, and
selection mode. Only deterministic exact reuse adds
`planner_model_calls_used=0` and `model_call_scope=planner`. Blueprint can read
the older `model_calls_used` field, but new Flyto2 AI reports do not emit it.
Flyto2 Blueprint turns those reports into
sample/success counts, Wilson 95% lower bound, retry/assertion rates, p50/p95
duration, and zero-planner-call counts. This does not count model calls made by
`llm.*` workflow steps. Prompts, parameters, credentials, and raw tool results
are not sent to the Evidence Card.

## Safety And Permissions

- permission levels distinguish read-only, workspace-write, and danger-full
  module classes; approval decisions remain explicit.
- prompt/tool-result injection scanning identifies instruction override,
  exfiltration, and suspicious content before reuse.
- URL policy rejects unsafe provider base URLs and tool destinations.
- vault storage encrypts credentials, injects them only into the process scope,
  and redacts configured values from text.
- optional Docker sandbox runs with no network, memory/CPU limits, timeout, and
  JSON-only input/output.
- Guardian hooks constrain coding-agent paths, extensions, and shell commands.

## Coding Agent And Evidence

`agents` wraps Claude Code-compatible execution with a bounded budget/turn count,
Guardian pre-hook, Flyto2 Indexer context, evidence collection, and a verification
loop. Verification recipes may use Flyto2 Core, and evidence records preserve
attempts/results without storing credentials.

## Channels And Scheduling

- generic channel adapters support Telegram, Slack, Discord, and webhooks;
- the channel router fans out text/file/approval messages with normalized results;
- the Telegram service adds commands, queued jobs, confirmations, steering,
  media/transcription, and Claude CLI bridging;
- scheduler tasks persist one-time/interval/cron-like jobs and dispatch agent
  requests under the same runtime controls.

## Security Test Workflow Generation

`security` maps structured findings into Flyto2 Core YAML for 13 categories:
SQL injection, reflected XSS, auth bypass, SSRF, open redirect, command injection,
path traversal, XXE, SSTI, NoSQL injection, insecure deserialization, mass
assignment, and CRLF injection. Generation validates scheme, blocks metadata and
private targets, and requires staging-like hosts unless an operator explicitly
overrides the production-target guard. Use only on systems you are authorized to
test.

## Extensions And Observability

Extensions declare manifests, load from approved locations, and register hooks.
Shell hooks receive an allowlisted environment and return structured allow/deny
decisions. Telemetry supports memory, JSONL, composite sinks, session traces,
cost/usage records, and redacted audit events.

## Physical Mission Interpretation And Planning

- `mission_interpretation` reads only the Zone and Objective cards physically
  drawn by a judge and recorded by an operator. The model may explain the task,
  request clarification, and select from APPROVED capability IDs; deterministic
  code preserves card evidence and supplies a raw-error-free fallback.
- `robotics_planning` converts a prefiltered Robotics shortlist and complete
  route candidates into a schema-bound plan with independent safety checks and
  attestation.
- Neither surface selects a live resource, grants motor authority, runs a
  controller, or decides whether evidence completes a Task.
