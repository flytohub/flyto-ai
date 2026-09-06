# Local CLI inference runtime

The host chooses `api`, `codex_cli` or `claude_cli`. This SDK implements the two
CLI transports; Cloud owns authenticated source selection, computer binding,
queue leases and goal verification. CLI sign-in remains on its computer.

```python
from flyto_ai import AgentConfig
from flyto_ai.cli_runtime import CliAgent, CliRuntimeConfig, complete_json

cli = CliRuntimeConfig(source="codex_cli", timeout_seconds=45)
agent = CliAgent(
    AgentConfig(provider="codex_cli", model="", max_tool_rounds=8),
    cli=cli, tool_executor=owned_executor, policies=owned_policy,
    system_prompt=supervisor_instructions,
)
try:
    response = await agent.chat(goal)
    # Only after independent review requests a correction to this same goal:
    response = await agent.continue_execution(goal=goal, message=correction)
finally:
    await agent.close()

# A fresh reasoning session has no execution tools or actor conversation.
review_json = await complete_json(
    cli, prompt=observations, schema=review_schema,
    system_prompt=review_instructions,
)
```

`CliAgent` preserves native admission, tool permissions, policies, callbacks,
observations and `ChatResponse`. The caller still owns browser scopes and Core
cleanup. The CLI supplies JSON content and proposed calls; the host validates
the entire intent before dispatching calls sequentially. Receipts come only
from the original host dispatcher. A CLI success is never a goal verdict.

A host can pass `completion_fn(prompt=..., schema=..., system_prompt=...)` that
returns a JSON string. This delegates inference only; local binary discovery
and launching are skipped. The host still performs all tool dispatch. This
initial delegated contract refuses image sidebands rather than silently
losing them; native local sessions support observed PNG/JPEG/WebP attachments.

## Authentication and isolation

`cli_environment()` retains HOME, USER and LOGNAME for official OS keychain
sign-in, optional official CLI home directories and minimal process/runtime
paths. It strips provider keys, token environment variables and unrelated
secrets. It never reads or copies auth-store contents. No CLI setting may be
supplied through a model tool call. `command` is an optional single executable
path for a trusted host/test harness, never a client-selected shell command.
`resolve_cli_executable(source, command=None)` gives execution and status probes
the same host-owned discovery. PATH takes precedence. On macOS only, Codex may
fall back to the executable bundled at
`/Applications/ChatGPT.app/Contents/Resources/codex`. An explicit missing command
never falls back; the discovered binary must still pass protocol checks.

Codex 0.153.4 or newer must support the app-server protocol and strict config.
The runtime disables native integrations, hooks, plugins, notifications and
hosted web search. It reads configuration only in memory, rejects custom
provider routes, disables each inherited MCP, creates an ephemeral thread with
`environments=[]` and `dynamicTools=[]`, and confirms that every MCP is disabled
with no tools/resources before inference. Every turn repeats the empty
environment selection. Native action or approval-request events are refused.
A read-only sandbox alone would not meet this contract.

Claude must support `--tools ''`, safe mode, restricted mode, strict empty MCP,
empty setting sources, no Chrome/slash commands and no session persistence.
Only its non-actuating `StructuredOutput` formatter may appear in startup
inventory. Formatter acknowledgements must match observed formatter call IDs;
other native tool events are refused. The runtime does not use `--bare`, which
would disable normal keychain access.

`inspect_cli_runtime(cli)` returns source, installed, version, supported and a
fixed reason code. It performs bounded official help/version and configuration
checks, not model inference or auth-store inspection. The application performs
its own official sign-in status check. Initial process isolation supports
POSIX; Windows reports `cli_process_isolation_unavailable` until job-object
ownership is implemented. Do not present unsupported runtimes as ready.

## Bounds, cancellation and evidence

Each invocation has an isolated temporary working directory and private process
group. Prompts travel over stdin; model output is bounded in memory. Native
attachments are validated base64 data and never arbitrary model-provided paths.
The process group closes on success, error, timeout or cancellation, including
children that outlive a successful parent. Cancellation is awaited before the
caller receives it. A closed agent cannot spawn or dispatch again.

`complete_json` accepts at most 256 KiB of UTF-8 prompt, 16 KiB of system prompt,
32 KiB of schema and 64 KiB of returned JSON. Inference envelopes have a 1 MB
bound; CLI streams have a 2 MB aggregate limit. A round can propose at most
eight host calls. Timeout is host-selected, bounded to 0.1–300 seconds and must
fit the enclosing job deadline. No transport failure silently retries a model
or a previously completed action. Actual partial tool observations survive.

Errors expose fixed codes, not stdout/stderr or credentials. Examples include
`cli_auth_required`, `cli_quota_exhausted`, `cli_timeout`,
`cli_native_action_refused`, `cli_native_tools_exposed`, and
`cli_nondefault_provider_route`. Missing usage remains unavailable; the runtime
does not manufacture billing totals for delegated inference.

## Verification and limits

The project declares the same Ruff selection as its existing CI command:
`E9,F63,F7,F82`. Ruff 0.16 expanded its upstream defaults, so an unconfigured
local check otherwise differs from CI. This fixes the existing repository
contract; it does not reduce CI checks or rewrite unrelated Agent code.

Tests run real child processes with controlled protocol fixtures through the
native Agent and host dispatcher. They cover sequential dispatch, forbidden
calls, malformed JSON, unavailable runtimes, partial failures, continuation,
images, cancellation, terminal child cleanup and delegated inference. These
fixtures do not claim live model reasoning or production mission completion.

On macOS, official Codex 0.153.4 returned the tiny requested JSON in 5.32 seconds
with four inherited MCPs disabled, no native action event and process cleanup.
Official Claude 2.1.258 returned it in 6.14 seconds with only StructuredOutput
and process cleanup. These are sign-in/transport proofs, not full AI Space or
multi-computer goal acceptance. Cloud must run its own real goal, independent
review and terminal queue reconciliation using these transports.

Rollback: remove the application's opt-in `CliAgent`/`complete_json` selection.
The existing native API Agent keeps its original missing-key behavior.
