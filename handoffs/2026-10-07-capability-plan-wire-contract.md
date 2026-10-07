# Capability-plan wire contract landed on both sides

Owner: claude
Branch: claude/capability-plan-contract
Date: 2026-10-07

## What changed

- PR #64 renamed the plan to `flyto.capability-plan.v1` and `robot_id` to
  `resource_id`. This branch also bumps the request contract to
  `flyto.robotics.planner-request.v2` (its field changed) and refuses v1 by name
  from `capability_router.RETIRED_PLANNER_REQUEST_CONTRACTS` through
  `planner_contract_refusal()`, used by both `prepare_planner_request()` and
  `robotics_planning.validate_request()`.
- `tests/fixtures/capability-plan-exchange.v1.json` +
  `tests/test_capability_plan_exchange.py`: byte-identical copy of
  flyto-robotics' fixture, digest pinned on both sides, compared with a sibling
  `../flyto-robotics` checkout when one exists.
- flyto-robotics 0.7.0 (flyto-robotics PR #63) is the matching client.

## Why

flyto-robotics consumes this planner over HTTP; renaming one side alone broke
the lab loop. No compatibility window: flyto-cloud never imports
`flyto_ai.robotics_planning` or flyto-robotics' planner client, so no released
Desktop reaches it.

## Verified

- ruff (CI selection), `scripts/generate_reference.py --check` (regenerated for
  the new public function), `scripts/check_release_drift.py`, full pytest with
  CI warning flags, `flyto-index verify . --full-scan --strict` — see PR.

## Not verified

- No live LLM planning run against flyto-robotics' showcase script.

## Follow-ups

- Drop the v1 entry from `RETIRED_PLANNER_REQUEST_CONTRACTS` once no supported
  flyto-robotics release predates 0.7.0.
