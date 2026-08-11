# Required-change Claude same-session retry

Date: 2026-08-10  
Owner: Codex  
Branch: shared working tree (not committed)

## Incident

Two formal hospital-plugin jobs reached Claude with the complete task in their
sealed resume envelopes, but the provider used no repository tool and returned
after one prose-only turn. The adapter had no browser verification recipe, so
it treated that turn as complete and ran the full repository contract on a
known-empty diff. Both jobs failed closed with no implementation revision.

## Repair

- `CodeTaskRequest` carries the host-owned `require_changes` invariant.
- `ClaudeCodeAgent` detects a service-mode turn with no attributable mutation.
- While the bounded attempt budget remains, it resumes the exact SDK session
  with a fixed instruction to inspect and edit; it never opens a replacement
  conversation.
- Exhaustion returns a stable failed response before required repository checks.
- Explicit no-change service jobs and legacy direct calls retain their prior
  behavior.

## Evidence

- `tests/test_agents.py`: 182 passed.
- Adapter authority propagation: 2 focused tests passed.
- Full suite: 3276 passed, 17 skipped in 211.67s.
- Compile, fatal/error Ruff, generated-reference check, and diff check passed.

## Next action

Restart the sole coding supervisor so it loads this repair, resubmit the narrow
hospital-plugin discovery fix, and require live evidence that the same Claude
session continues after a prose-only turn and produces an actual bounded diff.
Only then run the plugin's complete wheel/Core/Indexer contract and Codex audit.
