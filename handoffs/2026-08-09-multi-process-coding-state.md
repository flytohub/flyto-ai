# Multi-process coding state handoff

## Closed

- Reproduced the Codex thread-creation failure: a second `code-mcp` process
  sharing the configured state root exited with `coding state root is already
  served` before sending its MCP `initialize` response.
- Replaced the process-lifetime root lease with short cross-process state
  guards, per-job execution leases, and per-workspace edit locks.
- A newly started service reconciles queued/running records only when their job
  lease is unowned. Live work in another Codex conversation is not failed.
- Tenant isolation, idempotency, exact-revision audits, same-session rework,
  route evidence, and caller-owned commit/push authority remain unchanged.

## Proof

```bash
python -m pytest -q tests/test_coding_service.py tests/test_coding_route.py tests/test_cli.py
python -m pytest -q
python -m ruff check flyto_ai/coding/service.py tests/test_coding_service.py
python scripts/generate_reference.py --check
```

The focused service/route/CLI suite passed 323 tests. The complete suite passed
1900 tests with 17 skips. A fresh Flyto2 Indexer rebuild and verify completed all
18 checks with no warnings or failures.

A real local subprocess probe started two `flyto-ai code-mcp` processes with
one temporary state root and the Claude backend. Both independently returned
MCP protocol `2025-06-18` with server name `flyto-coding` while both processes
were still alive.

## Rollback

Temporarily operate one MCP process if a platform lacks process locks. Do not
restore state-root lifetime exclusion; it is incompatible with Codex's
per-conversation stdio lifecycle. Any ambiguous job ownership remains
non-landable and must fail closed.
