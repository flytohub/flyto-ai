# State

Last updated: 2026-07-26

Implemented:
- Closed-loop MCP verification distinguishes omitted identifiers, unknown
  plans, and known plans that do not yet have execution evidence.
- `flyto-core` MCP capability manifest exposed through `flyto-ai`.
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
- full suite: 957 passed, 15 optional/live-integration skips;
- Ruff fatal/error rules and `compileall`: pass;
- wheel and source distribution build plus Twine metadata validation: pass;
- strict documentation contract: pass;
- Flyto2 Indexer closed loop: 17 passed, 0 warnings, 0 failures.

Known constraints:
- Authenticated Cloud browser smoke requires runtime credentials and must not write them to files.
- Cross-repo package tests need sibling repos on `PYTHONPATH` when run outside an installed workspace.
- Provider, embedding, and live-channel tests that require external credentials remain opt-in and are skipped in credential-free verification.
