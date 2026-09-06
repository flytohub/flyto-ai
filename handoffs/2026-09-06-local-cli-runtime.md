# Local CLI inference runtime

## Follow-up: selected models and local inference (2026-09-07)

The same branch adds official Codex model/list metadata, safe manual CLI IDs,
honest external-default metadata and `flyto_ai.local_runtime` for explicitly
selected loopback Ollama/compatible services. Native local inference supports
actual host images; delegated callbacks intentionally refuse unsupported image
sidebands. Host Agent/Core authority and independent goal verification remain
unchanged. See `docs/local-model-runtime.md` for API, errors, limits and rollback.

Verification uses actual child processes and actual loopback HTTP fixtures.
No local model was downloaded, no live model was called in this phase, and no
running user service was changed. Full real-model product acceptance is owned
by Cloud and must not be inferred from these protocol tests.

Follow-up (2026-09-07): computer-local imperative wording now reaches the
existing action classifier. This fixes an observed English read/replace/save/
verify request classified as status-only discovery. A bounded preface grammar
still requires a following imperative verb; it does not grant permissions or
promote quotations, explanations, negation or logs. Verification is a focused
routing and real Agent permission regression, not a model-output substitute.

Follow-up: executable discovery is shared publicly with status checks. PATH
remains authoritative; only macOS may fall back to the executable in the known
official ChatGPT app bundle. Explicit commands and non-macOS hosts never use
that fallback. Five regressions cover these selection boundaries.

- Owner: codex
- Branch: codex/local-cli-agent-runtime
- Base: 6759bf80dac2b02c8ed2fbd2eb35ffd49e0ed011
- Status: Selected-model/local extension complete SDK suite and integrated gates passed; runtime frozen for Cloud acceptance

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

Latest immutable runtime: `2b3abdf84c207ed69202af08a5e2f0379261d3e7`.
The complete SDK suite passed 4476 tests and skipped 17 in 323.25 seconds,
with DeprecationWarning and PytestUnhandledThreadExceptionWarning treated as
errors. It used an isolated SDK editable installation and the existing exact
Core/Blueprint/Indexer stack. The canonical result is
`/tmp/flyto-sdk-model-runtime-installed-full.log`.

Integrated admission/CLI/model/local checks: 326 passed. Model/local task
validation: Ruff and 94 tests passed. CI Core MCP smoke: 109 passed. Strict
Indexer: 18/18. Compile, generated reference, release drift, Ruff and both
distribution builds passed. Codex model/list returned seven current models
using metadata RPCs only. No local model download, live local-model inference,
remote write or user-service modification occurred in this SDK phase.

Earlier base-CLI evidence follows; those counts are historical, not the latest
complete-suite count:

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
