# Execution scope helper extraction

Owner: Codex multi_node
Branch: codex/execution-scope-extraction

The owned browser and continuation additions exceeded the existing module size
budgets. This extraction keeps every public Agent method and Core tool facade
entry. Routing state, diagnostics and route-specific tool projection live beside
the existing admission helpers. Browser registry and retry operations live in
browser_scope, reading the current core_tools facade globals on every call.
Core parameter validation lives in a small helper and still consumes only the
handler supplied by core_tools. No execution route, permission, goal admission,
cleanup behavior, provider/tool schema or budget is changed.

Existing scoped browser, retry, continuation, parameter binding and Core MCP
tests exercise the moved behavior. Additional regressions preserve replacement
registries, retry globals and validation overrides at the Core facade; a refused
validation must still prevent the actual handler from running. These are local
adapter checks, not live-model or physical browser acceptance.

Rollback: revert the extraction commit. Public call sites and Cloud adoption do
not need migration. The integration owner runs the exact locked full SDK stack
and CI; no dependency mirror, live service or remote branch is changed here.

The focused adapter and CI MCP cohort passes 151 tests, with deprecation and
unhandled-thread warnings treated as errors. Agent is 1,693 lines under its
unchanged 1,700-line budget; core_tools is 1,764 under 1,777. The complexity
baseline and all permission/tool bounds are unchanged. Compileall and the CI
Ruff profile pass; full locked-stack acceptance belongs to the integration
owner rather than this bounded extraction.
