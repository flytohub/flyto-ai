# Governed Robotics planner entry

Owner: codex
Branch: fix/codex-jsonl-diagnostics
Date: 2026-08-22

## What changed

`flyto_ai/robotics_planner_server.py` now composes the formal server entry
through `prepare_planner_request(require_goal_frame=True,
require_discovery=True)` before `RoboticsPlanningService.plan`. Routing refusal
never invokes the provider-facing service. Focused regression tests pin the
order and failure boundary.

## Why

The planning primitives already supported Blueprint/Core capability routing,
but the runnable loopback planner server bypassed it. That made the documented
AI path optional at the composition root.

## Verified

The focused planner/router suite passed 108 tests. Adding the Blueprint
closed-loop suite passed 146 tests. Ruff passed on the touched code, strict
Indexer verification passed 18/18, and the cross-repository contract
smoke accepted one Cloud-generated handoff and one receipt through both edge
and Cloud validators.

## Not verified

No provider network call, deployment, robot motion, publication, or governed
stack release was performed. `scripts/stack_lock.py` still refuses the current
workspace at the pre-existing Core revision mismatch; the lock expects
`ba66727f44bcb15aad2d36a18ef3f2e6b1592bd4` while the clean local Core is
`23efc930b75ce09e50373809c21b0afe5bbcf70e`. The local Indexer revision also
differs from its lock. The lock was not rewritten around uncommitted work.

## Follow-ups

Publish only after the exact multi-repository revisions are represented by a
passing stack lock and normal repository governance.
