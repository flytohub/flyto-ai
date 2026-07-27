# State

Last updated: 2026-07-27

Implemented:
- Deterministic intent routing now distinguishes explicit actions, current-data
  questions, answer-only requests, multilingual negation, quoted/meta examples,
  and declarative questions before any provider tool dispatch.
- Tool permissions enforce the selected route at dispatch time, so a provider
  cannot turn a denied answer-only request into a raw MCP action.
- Learned Blueprint trust evidence fails closed for malformed types, non-finite
  numbers, non-integral counts, inconsistent counts, and out-of-range rates.
- Explicit reply-language changes persist through short follow-ups and return
  to the language of a later substantive message.
- Closed-loop MCP verification distinguishes omitted identifiers, unknown
  plans, and known plans that do not yet have execution evidence.
- `flyto-core` MCP capability manifest exposed through `flyto-ai`.
- Blueprint portable export/import is wired without exposing host signing or
  trusted-publisher keys.
- Direct model outcome reports are community evidence; only the deterministic
  Blueprint loop's in-process capability records `local_verified` evidence.
- The trusted Blueprint report now carries allowlisted duration, step/attempt,
  assertion, workflow hash, executor version, and selection-mode facts.
  Deterministic exact reuse records zero outer-agent planning calls;
  model-selected paths do not invent a count, and model-backed workflow steps
  are not mislabeled as token-free.
- Additive risk, approval, and evidence metadata on core tool definitions.
- Pre-execution `validate_params` gate for `execute_module`.
- Provider tool-call logs include MCP evidence metadata.
- CI workflow added for compile, tests, build, and local secret pattern scan.
- `.flyto-index/` ignored.
- Documentation contract maps 7 source areas and 8 feature surfaces to source,
  guides, generated references, and tests.
- Generated reference covers every top-level Python class/function, every direct
  class method, CLI declaration, static tool/MCP definition, static environment
  read, and maintainer script; CI rejects stale output.
- Package, CLI, and MCP versions share project/distribution metadata, while Core
  module totals are discovered from the installed runtime registry.

Verified on Python 3.11:
- full suite: 1150 passed, 15 optional/live-integration skips;
- Ruff fatal/error rules and `compileall`: pass;
- wheel and source distribution build plus Twine metadata validation: pass;
- strict documentation contract: pass;
- Flyto2 Indexer closed loop: 17 passed, 0 warnings, 0 failures.

The 2026-07-27 routing and evidence hardening was additionally verified with
700 multilingual/presentation-mutated route cases, 5,000 seeded Unicode/noise
inputs, a 408-case permission matrix, 4,500 Blueprint boundary cases, and 38
malformed-evidence cases. These are bounded local test results, not a claim of
perfect coverage for every language or live third-party MCP.

The 2026-07-26 Blueprint evidence-boundary change was reverified with the full
suite, generated-reference check, sdist/wheel build, and strict Indexer
full-scan. Twine metadata validation was not rerun for this source-only change.

Known constraints:
- Authenticated Cloud browser smoke requires runtime credentials and must not write them to files.
- Cross-repo package tests need sibling repos on `PYTHONPATH` when run outside an installed workspace.
- Provider, embedding, and live-channel tests that require external credentials remain opt-in and are skipped in credential-free verification.
