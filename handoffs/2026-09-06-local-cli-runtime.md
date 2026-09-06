# Local CLI inference runtime

Follow-up: executable discovery is shared publicly with status checks. PATH
remains authoritative; only macOS may fall back to the executable in the known
official ChatGPT app bundle. Explicit commands and non-macOS hosts never use
that fallback. Five regressions cover these selection boundaries.

- Owner: codex
- Branch: codex/local-cli-agent-runtime
- Base: 6759bf80dac2b02c8ed2fbd2eb35ffd49e0ed011
- Status: Full SDK suite and CI-aligned combined task validation passed

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
- Ruff 0.16.6 expanded its upstream defaults to 413 rules. The repository had
  no Ruff configuration; 155 Agent diagnostics existed equally on the base and
  changed versions, even with --isolated. Project lint now explicitly selects
  the exact existing CI rules E9,F63,F7,F82, without changing CI or Agent code.
  Combined task validation includes the Agent credential seam and passes Ruff
  plus 40 focused tests under that unchanged CI contract.
- An initial complete run in the Cloud archive environment returned 4310
  passed, 89 failed and 21 skipped. Missing SDK development dependencies and
  adjacent Blueprint fixtures account for most failures; this is not a green
  SDK CI claim. Its one new constructor-arity failure was fixed and the exact
  complexity gate now passes. The correctly installed SDK environment then
  completed with 4403 passed and 17 skipped in 1100.42 seconds (Python 3.12,
  macOS, commit 4e783b5). The later official-bundle discovery change passed its
  38-test cohort and strict/task gates separately.

## Rollback

Remove host opt-in CLI selection. API-backed Agent behavior and the canonical
Core dispatcher remain intact; do not change capability grants to compensate
for a missing or unsupported CLI.
