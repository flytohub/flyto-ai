# Numbered intent-ledger path authority

Owner: Codex
Branch: main
Date: 2026-08-25

## Failure reproduced

Three governed `flyto-code` jobs passed their repository verification and then
failed Indexer post-validation with `task:unplanned_diff`. The host's root
intent ledger contained five of the six explicitly listed paths and omitted the
new source-contract test because descriptive item text placed its path beyond
the 48-character local mutation-verb window.

## Boundary repair

- First-round projection recognizes a target/intent ledger that explicitly
  declares an exact/only/each repository-relative set followed by a colon and
  numbered items.
- The parser lives in `flyto_ai.coding.path_authority`; `route.py` remains at
  its recorded complexity ceiling rather than raising the baseline.
- Generic inclusion prose, unnumbered lists and negative clauses remain
  non-authoritative. Canonical path, typed suffix, existing parent, final
  symlink, in-root resolution and 64-target checks remain unchanged.
- Audit rework keeps the stricter same-clause mutation projection.

## Verification

- Focused route regressions: 12 passed.
- Complexity budget: 2 passed with no baseline increase.
- Full suite: 4,153 passed, 15 skipped.
- Stack lock, compile, fatal Ruff selection, release drift and all 23 generated
  references passed.
- `flyto-index verify --strict --json`: 18 passed, 0 warnings, 0 failures.

## Follow-up

Restart or let `code-mcp-supervisor` hot-reload at its next safe boundary, then
resubmit the failed Code sidebar job from a clean tree. The replacement job,
not any failed revision, must reach exact-revision Codex acceptance before
landing.
