# Owned browser execution scopes

Owner: Codex ai_space_runtime
Branch: codex/ai-browser-owned-scope
Status: Ready for integrated acceptance after local verification

## Workspace file permission classification

The actual admitted file goal still could not run: the SDK required DANGER_FULL
for every file module before Core received the call. The enforcer now captures
its host working directory and allows only exact file.read/file.write calls
whose literal path resolves inside it under WORKSPACE_WRITE. The captured root
is also part of trusted continuation admission. A changed cwd cannot rebase a
relative path to a different workspace. No per-turn override or blanket tier
elevation is introduced. Unknown file operations and env/path/shell access keep
the original danger requirement; READ_ONLY remains discovery-only.

The SDK classification does not replace Core's schema/path validation or its
environment sandbox. Real Core tests prove a narrower sandbox still refuses a
write, and a symlink changed between admission and Core validation cannot escape.
This is not a claim of atomic filesystem protection against a concurrent swap
after Core's own final path check; that existing Core limitation is unchanged.

The independent cases fail 9 times on the previous SDK and all 26 pass after
repair. The neighboring permission, routing and continuation cohort passes 343
tests. One deterministic provider fixture calls the actual Agent dispatcher and
real Core to read, write and reread isolated temporary files; READ_ONLY leaves
the output absent. The provider fixture is not live-model acceptance, which is
performed separately through the authenticated product UI. The CI Core MCP smoke
adds 109 passing tests. Compileall, CI Ruff, generated reference, release drift,
strict verification (18/18) and task validation (Ruff plus 26 tests) pass.
Exact results are retained in /tmp/flyto-browser-scope-evidence under
the file-permission prefix. Reverting this change restores the earlier blanket
file restriction without altering Core configuration or the agent's tier.

## Natural Chinese local file action admission

The real file goal started with a request to use this computer's tools before
the actual read/replace/write/reread instruction. The anchored verb matcher
missed that imperative, and a status word in the file content incorrectly
selected read-only discovery. A bounded Chinese preface grammar now preserves
the operative clause for all existing intent checks. It does not search later
arbitrary prose for verbs or add a force-action option. File save/replacement
verbs and non-executing questions about file actions are recognized as well.

The independent regression starts with 13 failures and 14 passes on the prior
SDK. It covers natural phrasing variants, quotations, hypothetical examples,
negation and explanation, plus actual Agent tool exposure with normal and
read-only permissions. All 27 pass after repair, together with the surrounding
multilingual routing and admitted-continuation tests (215 total). These tests
use a deterministic provider only to inspect tool policy; they do not stand in
for live model or real filesystem goal acceptance. Four confirmation regressions
and 109 CI Core MCP smoke tests also pass. Compileall, CI Ruff, generated
reference, release drift, strict verification (18/18) and the complete task
validation (Ruff plus 27 tests) pass. Core catalog discovery is checked separately
without executing a file operation: search and exact info expose file.read/write,
but their atomic category differs from the file category used by file.copy.
Exact logs are under /tmp/flyto-browser-scope-evidence with the chinese-action
prefix. Revert this SDK change to roll back admission;
permissions, Core contracts and host continuation APIs are unchanged.

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

## Parameter recovery breaker

The middleware treated Core status=success without ok as failure and considered
missing data/result fields empty; valid browser operations could trip the
breaker. Both interpretations are corrected. Invalid params now have a separate
parameter-fingerprint repetition budget, so three failed validations cannot
ban a corrected invocation. No side-effect-free validation is reported as an
executed operation. The actual Core params_schema accompanies a validation
error. This does not guess selector aliases or override Core's schema.

Regressions use the real Core validator/schema with a counted no-effects handler
to prove three bad arguments, bounded identical repetition, corrected execution,
repeated status-only successes, true execution failure/empty-result limits, and
observation-before-interaction. Active SDK prompts contain no stale concrete
selector-only type/click examples; the rule to follow each module's schema stays
in force. The public signatures add optional params arguments compatibly.

## Default application audit privacy

Real local acceptance exposed credentials copied from user goals and browser
input arguments into INFO logs. Audit emit now projects metadata before any
logger, memory or JSONL sink: goal SHA256/length, usage/counts and operation
identity/status. It excludes arguments, previews, result bodies, runtime handles
and exception messages. Only a closed set of stable error codes is retained.
Provider Tool call logging omits arguments entirely. The authenticated owner
transcript and tool dispatch/stream result are unchanged; this does not erase
pre-existing logs or modify the observed tool result.

Tests verify all three audit sinks, a goal longer than 200 characters, opaque
runtime handles, stable error codes, and unchanged dispatch/owner evidence.
Evidence: audit-* files in /tmp/flyto-browser-scope-evidence. Live goal acceptance
is still tracked separately; these privacy tests do not claim goal completion.

## Trusted continuation of admitted execution

Real review corrections were classified as ambiguous discovery requests, hiding
execute_module while the owned browser remained available. The host now calls
`continue_execution(message=repair, goal=original_goal, history=history,
template_context=context)` after the first ordinary action chat. The SDK checks
the same Agent, exact goal digest, unchanged tools/policy, and a current host
async call. It preserves action routing through the existing guards without
classifying generated recovery prose. It is not a model tool. A new ordinary
chat invalidates prior admission; missing/mismatched admission fails closed.

Policy changes require a new action admission rather than silently widening or
reusing old authority. Cancellation revokes the call scope, and inherited
background context cannot use it. The original goal is not retained as raw text
in the admission. Existing default chat routing and explicit no-action requests
remain unchanged. Cloud must use this entry for corrections, never fall back to
an ordinary fresh chat if it is missing. Revert Cloud adoption with this SDK API.
TASK currently runs one actor chat with tool-loop repair, so it has no outer
generated-correction turn to migrate.

Validation: continuation tests use a deterministic provider to inspect admitted
tools and real SDK dispatch guards, not as live-model acceptance. The separate
UI acceptance must prove actual Core observations, goal verification and cleanup.

## Prepared middleware and browser state during correction

The live browser registry survived corrections, but each Agent.chat recreated
middleware and its first-call blueprint redirect. A redirected browser.launch
was falsely logged as success; the actor then used a redacted session selector
and relaunched its authenticated browser. Custom system prompts also returned
before the owned-browser hint was appended.

The same admitted continuation now reuses its original assisted dispatcher.
Changing the base dispatcher or assistant refuses continuation; an ordinary new
chat clears the cache. Retry counters, snapshot state, prepared guard and current
URL stay with the goal. Blueprint redirects now carry ok=false and
action_executed=false, matching the existing snapshot substitution contract.
Custom prompts append browser state only when an owned scope exists. A sole
browser can be selected by omitting session/context handles; no session from
another scope or legacy global registry is disclosed to the custom prompt.

Regressions prove one browser launch across initial/correction turns, unchanged
authenticated session observation, one preparation guard per admission, fresh
preparation for a new goal, isolated hints, and truthful redirect metadata.
Browser handlers/providers in these unit tests are deterministic doubles. Live
actor verification remains the separate UI acceptance; no model outcome is
changed by this patch. Revert this commit to restore the previous middleware
preparation while retaining the admitted-continuation public API.
