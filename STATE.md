# State

Last updated: 2026-06-21

Implemented:
- `flyto-core` MCP capability manifest exposed through `flyto-ai`.
- Additive risk, approval, and evidence metadata on core tool definitions.
- Pre-execution `validate_params` gate for `execute_module`.
- Provider tool-call logs include MCP evidence metadata.
- CI workflow added for compile, tests, build, and local secret pattern scan.
- `.flyto-index/` ignored.

Known constraints:
- `flyto-core` remains read-only for this work.
- Authenticated Cloud browser smoke requires runtime credentials and must not write them to files.
- Cross-repo package tests need sibling repos on `PYTHONPATH` when run outside an installed workspace.
- Full `python -m pytest` currently has pre-existing failures unrelated to the MCP contract: deterministic pipeline bypassing mocked providers, Python 3.11 event loop assumptions in older tests, and existing prompt/audit drift.
