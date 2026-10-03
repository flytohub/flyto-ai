# Contract-aware grading and pack capability groups

Owner: claude
Branch: claude/pack-capability-grouping
Date: 2026-10-04
Status: Done (pending merge)

## What changed

- `flyto_ai/permissions.py`: `ContractGrade`, `grade_module_contract`,
  `resolve_module_grade`, `core_module_info`, `IMMEDIATE_STOP_CAPABILITIES`,
  `RISK_*` constants. `PermissionEnforcer` takes an optional
  `module_info_resolver`; `execute_module` is raised (never lowered) to the
  contract grade. Actuating or movement/dangerous contracts are `DANGER_FULL`
  (risk 4/5); uncontracted non-core package modules fail closed; a `motion.halt`
  stop contract is `READ_ONLY`, risk 1, `immediate`.
- `flyto_ai/tools/pack_tools.py` (new): `get_pack_capability_groups`,
  `build_pack_tools`, `pack_tool_name`, `pack_tool_permission_overrides`,
  `resolve_pack_tool_call`, `bind_resource_capabilities`,
  `params_schema_to_json_schema`, `get_pack_tool_catalog`.
- `stack-lock.json`: CI Core revision `4a76d3d` -> `82a87fd` (2.35.0).
- `pyproject.toml`: flyto-core floor 2.31.1 -> 2.33.0 (security floor derived
  from Core's advisories at `82a87fd`; first CI run failed on it).
- Tests: `tests/test_contract_grading.py`, `tests/test_pack_tools.py`
  (two packages providing `motion.advance`, collisions, fail-closed grading,
  stop immediacy, and one test against the real Core registry).
- Docs: CHANGELOG, DECISIONS, STATE, regenerated `docs/reference/python/*`.

Not done on purpose: `_DANGER_MODULE_CATEGORIES` in `tools/core_tools.py`
stays, and no planner cleanup (both deferred to Phase C by review).

## Why

Phase A of the module-pack plan: a provider plugs in only through the
`@register_module` contract, so grading must come from the contract, and a
host's resource tier needs each package whole, with per-module contracts
(the manifest `contracts` map collapses two providers of one capability).

## Verified

- Local venv: flyto-core from the archive of `82a87fd` (2.35.0),
  flyto-blueprint `b4228b6`, flyto-indexer `3667604`, Python 3.12.
- New tests: 36 passed.
- Full suite, changed tree: 4530 passed, 76 failed, 32 skipped. Clean
  `origin/main` on the same venv: 4494 passed, 76 failed, 32 skipped. The
  failures are coding-route / benchmark tests that need sibling checkouts and
  the sandbox runtime; the one differing name in each set
  (`test_admission_is_idempotent_and_places_exactly_one_work_item`) passes
  and fails intermittently on rerun.
- CI smoke cohort + new tests with `-W error::DeprecationWarning`: 257 passed.
- `ruff --select E9,F63,F7,F82`, `compileall`, `generate_reference.py --check`,
  `check_release_drift.py`: pass.
- `flyto-index verify . --full-scan --strict`: 21 PASS.

## Not verified

- No Cloud integration: Cloud does not call these APIs yet.
- No real pack (flyto-modules-robotics) installed in the test env; the real
  registry test registers two synthetic plugins.
- The MCP `task(action='validate')` gate was not run (CLI equivalents above).

## Follow-ups

- Cloud AI Space resource tier: use `get_pack_tool_catalog()` on the Desktop,
  merge `permission_overrides` into `SpaceToolExecutor` levels (do not grade
  pack tool names by prefix), route `immediate` bindings to the existing stop
  path, and propose every other call as a Space task step.
- Phase C: remove `_DANGER_MODULE_CATEGORIES` duplicate in `core_tools.py`.
