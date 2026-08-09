# Strict route repair, runtime status, and emergency overflow

Owner: claude
Branch: main (uncommitted working tree; Codex owns review and landing)
Status: Active — implementation complete; independently verified by Codex,
including a real live emergency round accepted on an exact revision

## Closed

### The production failure

Reproduced against the real installed Indexer with the exact production route
policy: `indexer_pre.search` timed out at 30.002 s. The route sent only
`{query: message[:500]}`, so the Indexer's smart search fanned out across every
indexed project. The same query scoped with `project` completed in about a
second. The persisted result collapsed this to `route_domain_failure` with
`calls=[]`, so nothing recorded where the round stopped or that the implementer
had never started.

- Every host-owned Indexer search now carries the workspace project: initial
  discovery, gate remediation, and translated plan steps. The 30-second
  capability bound is unchanged — the query was over-broad, not the deadline
  too short.
- A failed lane receipt keeps every completed call plus one failed call naming
  the exact host-derived semantic action, bounded by `max_calls_per_lane`.
- A transport timeout is classified `capability_timeout` from a closed machine
  code the capability adapter reports. The route never parses provider prose,
  so it stays distinguishable from `domain_failure`.
- A capability launch failure now names the lane whose provider was actually
  unavailable instead of always reporting `indexer_pre`.

### Truthful implementer start

`implementer_started` is written to the durable job record immediately before
every real implementer invocation, never inferred from `running`, and is
exposed additively on `CodingJobReceipt` (and therefore on both transports). A
round that fails *after* implementation keeps its session id, attributable
files, and revision digest as proof the model ran, while remaining terminal,
non-landable, and non-auditable.

### Runtime status (`flyto.coding-route-status.v1`)

Per-instance `status/instance-<id>.json` plus a validated, byte-bounded shared
`status/index.json`, written atomically at mode 0600 under the existing
cross-process state guard. A single latest-writer file was rejected: many Codex
conversations share one state root and an old process would overwrite a newer
one's diagnostics. `flyto-ai code-status --state-dir <dir> [--json]` reads it
read-only and annotates build id, liveness, and staleness.

### Emergency overflow (`flyto.coding-emergency.v1`)

A startup-only lane for a provably unreachable route infrastructure, enabled
with `--emergency-overflow-backend` (which must equal
`--implementation-backend`) and `--emergency-overflow-threshold` (default 1).
It opens only for a classified `capability_unavailable` / `capability_timeout`
failure in a pre-implementer lane with no attributable edit and no durably
recorded implementer start. Emergency rounds call the same selected backend,
keep the required checks and exact-revision binding, and still require an
independent Codex audit under a separate digest-validated authority receipt
sealed to that job, request, session, and revision.

## Proof

```bash
python -m pytest -q tests/test_coding_route.py
python -m pytest -q tests/test_coding_emergency.py
python -m pytest -q tests/test_coding_service.py tests/test_coding_control.py tests/test_cli.py
python -m ruff check flyto_ai/ tests/ scripts/
python scripts/generate_reference.py --check
python -m pytest -q
git diff --check
```

Results are recorded in `STATE.md`. Two loopback-socket tests
(`test_http_server_requires_auth_rejects_provider_fields_and_runs_job`,
`test_webhook_post`) cannot run in this sandbox: `socket.bind` returns
`PermissionError`, reproduced with a bare socket outside pytest. They are
unrelated to this change and pass in the unrestricted environment.

The four route service tests that previously hung for 120 s each now settle in
seconds; the whole route suite runs in about 11 s.

A bounded live regression
(`test_the_real_indexer_answers_a_project_scoped_search_inside_its_bound`)
starts the real installed `flyto-indexer` and proves the project-scoped
pre-work search completes well inside the 30-second bound.

## Independent Codex closure (owned by the auditor, not by me)

Codex exercised the emergency lane against a real service process. A fresh
`flyto-ai code-mcp` used startup backend `claude`, which this adapter pins to
`claude-opus-5`, launched with an intentionally missing Indexer command,
explicit `--emergency-overflow-backend claude`, and threshold 1.

- Job `job_3169dfad6918444abfeb9fe9` first failed before implementation at
  `indexer_pre` with `capability_unavailable`. Runtime status then showed
  `circuit_state=open`, `mode=emergency`, `implementer_started=true`, and one
  emergency activation.
- Claude produced session `cda281f0-d3de-4617-9a3e-4045cc1ea928` and first
  revision
  `77f81f543a9a525356af96ccd56191be5f4261326df6f2c7f0b1831e69b4776e`. The
  required source-controlled checks passed, yet Codex's independent hidden case
  found `slugify("Alpha___Beta") == "alphabeta"` and submitted one typed
  `major` rework finding against that exact revision. Green repository checks
  did not substitute for the audit.
- The service resumed the same Claude session and produced revision
  `2118b92f675d698d8adeb7d9aa7466832c3ec8aa5d690a10f240a0fd478087c8`, with the
  emergency authority re-sealed as `mode=emergency_rework` to the same job,
  request, and session and to the new revision. Codex independently observed 3
  tests pass, `git diff --check` pass, and a five-case hidden slug matrix pass.
- Codex accepted that exact second revision: `state=codex_accepted`,
  `landable=true`, `audit_count=2`, `rework_count=1`,
  `emergency_activations=2`. After graceful EOF the per-instance status kept
  those diagnostic facts with `lifecycle=closed` and `alive=false`, while the
  index simultaneously retained a separate earlier closed process row under a
  different instance id — direct multi-instance evidence with no latest-writer
  clobber.

Independent verification on the final diff: focused route, emergency/status,
and CLI suites **297 passed**; unrestricted complete suite **2001 passed, 17
skipped**; Ruff passed; 23 generated references current; `git diff --check`
passed; full Indexer rebuild 238 files / 3665 symbols / 21818 dependencies with
0 errors; strict verify 18 pass, 0 warn, 0 fail.

## Residual

- I did not commit, stage, push, or change any parent-workspace configuration.
  Codex has since updated `.codex/config.toml` itself to pass
  `--implementation-backend claude`, `--emergency-overflow-backend claude`, and
  `--emergency-overflow-threshold 1` (SHA-256
  `43273321e87e435669e169d6b97c40fccfc42c8f8a3f3eb727a3b8b7b35c870a`), so a
  newly started Codex MCP process receives the explicit Claude overflow
  authority.
- That configuration does not reach a `code-mcp` process that is already
  running. Such a process keeps its previously loaded code and startup
  arguments: it has no overflow authority, publishes no status row, cannot
  appear in `code-status` retroactively, and still carries the unscoped-search
  bug. Restart or reopen the session to pick all of that up.

## Rollback

Omit `--emergency-overflow-backend` to remove the overflow lane entirely; the
service returns to failing closed on an unreachable Indexer. The project-scoped
search, failure-evidence, and runtime-status changes have no flag and are safe
to keep independently. Deleting the `status/` directory only loses diagnostics;
per-job JSON remains the sole authority.

## For the next agent

Restart any `code-mcp` process that predates this change so it loads the
repair and registers in the status index; see Residual above for why an old
process cannot register itself. New and old instances coexist safely, and
`flyto-ai code-status --state-dir <dir> --json` shows each one's build id,
lifecycle, and liveness so you can tell them apart.
