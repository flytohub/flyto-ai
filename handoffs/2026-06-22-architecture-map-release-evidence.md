# Architecture Map Release Evidence

Date: 2026-06-22

## Context

The Flyto2 workspace release packet requires architecture/dependency map
evidence for every core repo. `flyto-ai` had root architecture notes but no
`docs/architecture-map.md`, so the workspace-level architecture deliverable was
correctly marked as P1 `needs_evidence`.

## Change

- Added `docs/architecture-map.md`.
- Documented core areas, cross-repo edges, provider boundaries, product-line
  roles, and release invariants.

## Verification

- `flyto2-release-packet` should now find
  `flyto-ai/docs/architecture-map.md` as architecture evidence.
