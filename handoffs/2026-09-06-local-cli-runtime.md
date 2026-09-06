# Local CLI inference runtime

Follow-up: executable discovery is shared publicly with status checks. PATH
remains authoritative; only macOS may fall back to the executable in the known
official ChatGPT app bundle. Explicit commands and non-macOS hosts never use
that fallback. Five regressions cover these selection boundaries.

- Owner: codex
- Branch: codex/local-cli-agent-runtime
- Base: 6759bf80dac2b02c8ed2fbd2eb35ffd49e0ed011
- Status: Reviewable implementation; full SDK CI-equivalent verification pending

## Contract and scope

`flyto_ai.cli_runtime` exposes CliAgent, CliRuntimeConfig, complete_json,
inspect_cli_runtime, cli_environment and required_cli_flags. See
[the runtime contract](../docs/local-cli-runtime.md). Cloud orchestration,
source settings, auth binding and task goal verification remain with Cloud.
Native Agent changes only by extracting its unchanged credential predicate for
the trusted CLI subclass. The official CLI owns sign-in, never host actions.

Both tiny official CLI transports have returned the requested structured JSON
on this Mac, with no native action events and private process cleanup. Codex
uses no execution environments; Claude exposes only its StructuredOutput
formatter. Full model-driven product goals remain separate Cloud acceptance.
Windows and delegated image input are explicitly unavailable, not implied
supported. No remote branch, PR, package release or live service was changed.

## Verification

- Focused Agent, Core MCP and CLI protocol cohort: 159 passed.
- Native CLI tiny transport: Codex 5.32 seconds; Claude 6.14 seconds.
- Exact SDK-stack focused Agent/Core/CLI cohort: 174 passed.
- New CLI and complexity gate: 33 passed; no budget threshold changed.
- CI Ruff selector, complete new-file Ruff, compile, build, release drift and
  generated reference checks passed. Strict Indexer: 18/18 passed.
- Combined task validation runs the 33 tests successfully but remains blocked
  by pre-existing full-style Ruff debt in agent.py. The five-line credential
  predicate change passes the repository CI selector. No rule was weakened.
- An initial complete run in the Cloud archive environment returned 4310
  passed, 89 failed and 21 skipped. Missing SDK development dependencies and
  adjacent Blueprint fixtures account for most failures; this is not a green
  SDK CI claim. Its one new constructor-arity failure was fixed and the exact
  complexity gate now passes. A correctly provisioned full run remains due.

## Rollback

Remove host opt-in CLI selection. API-backed Agent behavior and the canonical
Core dispatcher remain intact; do not change capability grants to compensate
for a missing or unsupported CLI.
