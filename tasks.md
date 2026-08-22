# Tasks

- [x] Add flyto-core MCP manifest contract.
- [x] Add pre-execution schema validation for core module execution.
- [x] Add MCP metadata to provider tool-call logs.
- [x] Add focused tests for manifest, validation, and evidence logging.
- [x] Restore parent-proof-compatible audited rework across the Indexer intent-ledger version transition.
- [x] Record that repaired same-job rework crossed the legacy/canonical
  parent-proof boundary but failed closed before provider start with
  `route_plan_bound_exceeded` when 36 cumulative steps exceeded the unchanged
  32-step ceiling; no same-job completion is claimed.
- [x] Record recovery primary job `job_0b90e4cab8e14f5482aec5f6` selecting the
  final implementation with all ten governed gates green, holistic Cloud job
  `job_497fc5ee77d948f2b71b26e8` being Codex-accepted, and follow-up job
  `job_4f40e4fcb6e54ea387786fe7` being Codex-accepted with `landable=true`,
  `audit_count=1`, and `rework_count=0`.
- [x] Record Cloud PR <https://github.com/flytohub/flyto-cloud/pull/231>
  protected-squash merge to `main` commit
  `ee8c95678c9a18931890a096ea7c04f6a7295ad0` after all remote checks were
  green, including Playwright (136 total, 113 passed, 23 existing skips, 0
  failed) and Audit Closure.
- [ ] Bind future broad audited repairs to the active scope of current findings
  so cumulative rework stays within the unchanged route-plan ceiling; do not
  raise or bypass the ceiling.
- [x] Add CI and generated index ignore.
- [ ] Add reusable eval datasets for MCP tool selection.
- [ ] Wire Cloud UI diagnostics to `/api/ai/tools/manifest`.
- [x] Split OpenAI provider chat dispatch into test-backed helper pipeline.
- [x] Split prompt-evolution mock response generation into test-backed helpers.
- [ ] Refactor provider chat complexity in small, test-backed steps.
- [x] Repair full-suite pytest drift after deterministic planning and Python 3.11 event loop changes.
- [x] Add an exhaustive source-generated implementation reference and strict documentation contract.
- [x] Eliminate hard-coded runtime module totals from package, CLI, MCP, and demo behavior.
- [x] Align Claude implementation rounds with the existing bounded 100-turn ceiling.
- [x] Make Guardian honor its closed repository-dotfile edit allowlist.
- [x] Let host operators retire a kernel-closed orphan beside unrelated live
  coding services without weakening job leases, MissionStore proof, or claim repair.
# Mission Stations

- [x] Add strict judge-drawn card interpretation and deterministic fallback.
- [x] Keep evidence requirements outside model-owned structured output.
- [ ] Wire the versioned interpretation response into the Cloud Mission Task
  planning adapter without granting resource or execution authority.
