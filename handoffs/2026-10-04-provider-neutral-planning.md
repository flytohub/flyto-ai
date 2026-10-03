# Provider-neutral planning and contract-declared recovery

Owner: claude
Branch: claude/ai-planner-cleanup
Date: 2026-10-04

## What changed

- `flyto_ai/capability_router.py`: no provider is inferred from an identifier
  prefix any more (`robotics.` no longer becomes source `flyto-robotics`;
  unsourced non-`core.` manifests are `external`). Resource compatibility uses
  neutral `context.resource_model` / `manifest.compatible_resources`; legacy lab
  `robot_model` / `compatible_robots` are read as aliases and the neutral field
  wins. Exclusion reason `robot_incompatible` -> `resource_incompatible`. Core
  projections carry `compatible_resources: []` (same semantics as before).
- `flyto_ai/contract_recovery.py` (new): reads the optional `recovery` key of a
  registered capability contract (`{"substitutes": [...], "context": "<obs>"}`,
  `recovery_context` alias), fails closed on malformed data, offers only
  installed declared substitutes from the host's candidate set with their own
  grade/module id/tool name, never the failed capability itself.
  `core_supports_recovery()` feature-detects core (display only).
- `flyto_ai/structured_provider.py` (new): the `StructuredJsonProvider`
  boundary. `robotics_planning` re-exports it; `mission_interpretation`,
  the OpenAI provider docstring and tests import the neutral module.
- `robotics_planning.py` / `robotics_planner_server.py` documented as the
  legacy lab protocol kept for released flyto-robotics lab tooling (HTTP
  caller in flyto-robotics `ai_planner.py`). Not deleted.
- `flyto_ai/tools/pack_tools.py` (after PR #59 merged): each pack module entry
  carries `recovery` (declared or `None`), and the tool description names the
  declared substitutes. `recovery_guidance(failed, group["modules"])` works on
  pack group modules directly.
- `flyto_ai/tools/core_tools.py`: removed the unread `_DANGER_MODULE_CATEGORIES`
  duplicate; `flyto_ai.permissions.DANGER_MODULE_CATEGORIES` stays the authority.
- Docs: `docs/CAPABILITY_ROUTING.md` (provider-neutral filters, contract-declared
  recovery), `docs/documentation-manifest.json`, CHANGELOG, regenerated reference.

## Why

Owner architecture: providers plug in only via the `@register_module`
capability contract; flyto-ai planning must not name a provider. Recovery
substitutes must come from the provider's declaration, not from host code.

Rejected: deleting the legacy lab planner (would break the released
flyto-robotics lab planner HTTP client); guessing a provider from `plugin`.

## Verified

See the PR body for exact counts (full pytest, ruff E9/F63/F7/F82,
`generate_reference.py --check`, documentation contract audit,
`flyto-index verify --strict` all passing on the branch).

## Not verified

- flyto-core 2.36 `recovery` field is not merged anywhere yet; the reader
  follows the Phase B plan shape (`substitutes`, `context`). If core lands a
  different key name, `declared_recovery` must follow it.
- No live run against flyto-robotics lab planner; its tests were not run.

## Follow-ups

- Cloud mission recovery should call `recovery_guidance` with the resource's
  approved pack modules once Cloud consumes flyto-ai contract grouping.
- Phase C: move the legacy lab planner into flyto-robotics lab tooling.
