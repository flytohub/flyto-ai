# Blueprint evidence authority boundary

Date: 2026-07-26

Summary:

- Wired `export_blueprint` and `import_blueprint` through the Flyto AI
  Blueprint dispatcher without exposing signing keys or trusted publishers.
- Direct model-facing `report_blueprint_outcome` calls now record community
  observations.
- The deterministic Blueprint executor attaches an in-process string-subclass
  capability checked by object identity. JSON can copy the text but cannot
  reproduce the object identity, so guarded closed-loop runs alone record
  `local_verified` evidence.
- Assistant feedback skips outcome scoring when a Blueprint was selected but no
  module execution evidence was produced.

Verification:

- `PYTHONPATH=../flyto-blueprint python -m pytest
  tests/test_blueprint_closed_loop.py tests/test_audit_fixes.py -q`
- Run the full suite, Ruff, generated-reference check, and Flyto2 Indexer
  strict verification before release.
