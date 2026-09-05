# Owned browser execution scopes

Owner: Codex ai_space_runtime
Branch: codex/ai-browser-owned-scope
Status: Ready for integrated acceptance after local verification

## Behavior

The host encloses a complete goal in `browser_session_scope(owner_id)`. Core
module validation, permission checks, retry, and evidence still use the existing
adapter. Registry and retry state are isolated using ContextVar. Scope exit
executes browser.close for each owned session, waits through cancellation, and
returns only session IDs and acknowledgement/error metadata. Cleanup is bounded
and raises BrowserCleanupError if any close is uncertain. It never clears or
closes the legacy caller's browser. Late inherited work is refused once closing
starts. The host must keep local recovery and result verification inside this
scope and retain observations before exit.

## Validation

- 118 tests passed: new scope tests plus the CI Core MCP, MCP server, browser
  retry, tool registry, and validation cohort, with warnings treated as errors.
- Scope regressions cover concurrent and nested callers, retry isolation,
  cancellation, action failure, explicit close, foreign-session preservation,
  close failure/timeout, and late background work.
- CI Ruff profile is E9,F63,F7,F82. Indexer validation uses the identical profile
  via a task-local PATH launcher; this avoids mass-changing existing style debt.
- Generated reference, compileall, release drift, strict indexer and task
  validation are required before commit. Evidence: /tmp/flyto-browser-scope-evidence.
- This commit's unit tests substitute the Core handler to exercise adapter
  lifecycle deterministically. They do not claim real-model or physical browser
  acceptance. Cloud's authenticated real-model acceptance is tracked separately.
- The unrelated complete SDK suite, Docker build, and cloud CI are not rerun.

## Rollback

Revert Cloud scope adoption together with this SDK commit. Do not deploy Cloud
scope adoption against an SDK missing browser_session_scope. Legacy unscoped
callers otherwise keep the same public tool contract and behavior.

## Unreleased version correction

The release-drift script compares HEAD, so its pre-commit 0.20.0 pass did not
cover the new code commit. The follow-up declares unreleased 0.20.1 and reruns
that gate after commit. No distribution is published or installed globally.

## Unresolved module parameters

Real UI execution typed an unresolved sensitive_text template into the login
form. Ad-hoc execute_module now rejects unresolved references before side
effects (including before relaunch closes an existing browser). Explicit JSON
context bindings resolve through Core VariableResolver; ambient env.* lookups
are refused, and runtime handles cannot be variable data. Supported ${...},
${{...}}, and {{...}} references are normalized for Core resolution. Diagnostics
contain no input value and tell the actor to repair the current call.

Validation covers actual Core binding semantics, all three placeholder forms,
no-call rejection, successful literal correction, unchanged original parameters,
missing binding, environment-secret isolation, runtime handles, list indexes,
and preservation of an existing browser after invalid launch arguments.
