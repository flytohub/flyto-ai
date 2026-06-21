# Handoff: flyto-core MCP Contract

## Summary

`flyto-ai` now exposes a first-class `flyto-core` MCP capability manifest and enriches core tools with risk, approval, and evidence metadata.

## Verification

- `python -m compileall flyto_ai`
- `python -m pytest tests/test_core_mcp_contract.py tests/test_mcp_server.py tests/test_browser_retry.py -q`

## Notes

- `flyto-core` was not modified.
- Credentials were not written to files.
- Cloud can consume this through `/api/ai/tools/manifest`.
- Full `python -m pytest` is not clean yet; current failures are existing suite drift outside the MCP contract path.
