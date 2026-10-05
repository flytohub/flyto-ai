# The Claude CLI's formatter reminder is not a native action

Owner: claude
Branch: claude/native-refusal (local, from origin/main 67048bd; not pushed)
Date: 2026-10-05

## What changed

`flyto_ai/cli_runtime/events.py`: `EventReader._claude` accepts a user event
that Claude Code marks `isSynthetic: true` and whose content is only
`{"type": "text", "text": str}` blocks (`_cli_reminder`). Every other user
event keeps the strict rule: each block must be a `tool_result` for a
StructuredOutput call the reader observed, otherwise
`cli_native_action_refused`. Assistant `tool_use` blocks other than
StructuredOutput are still refused.

Tests: `tests/test_cli_codex_runtime.py` (the recorded reminder followed by
the formatter call completes; five near-miss shapes are still refused) and
`tests/test_cli_runtime.py` (a real child process emits the live sequence
through `complete_json`). Doc: `docs/local-cli-runtime.md`. `CHANGELOG.md`.

## Why

Live 2026-10-05 10:22-10:23 local, flyto-cloud task t-52b50c54e6ade276,
AI Space job 3275a2cc-1e50-5d8e-a939-dc2ba0fac639 failed with
`cli_native_action_refused` after two `get_resource_task_result` calls. The
raw stream is not kept (the runtime writes no model reply to disk), so the
event was reproduced against the installed Claude Code 2.1.289 with the
runtime's own argv: when a turn ends in prose instead of calling
StructuredOutput, the CLI emits

    {"type":"user","isSynthetic":true,"message":{"role":"user","content":[
      {"type":"text","text":"[structured-output-enforce] You MUST call the
       StructuredOutput tool to complete this request. Call this tool now."}]}}

and the model then calls the formatter. The reader treated that text block as
an unacknowledged action. The same prompt through the unfixed reader returned
`cli_native_action_refused`; through the fixed reader it returned the
structured answer. The CLI binary contains this enforcement path
(`structured-output-enforce`), so it is a protocol case, not the model trying
a native tool: the init event lists only `StructuredOutput`, and no
non-StructuredOutput `tool_use` was needed to reproduce the code.

Rejected: a host retry of the slice. The failure was a misclassification, so
retrying would have spent a second inference on the same reminder.

## Verified

- `pytest tests/test_cli_codex_runtime.py tests/test_cli_runtime.py`:
  67 passed; with `events.py` reverted to HEAD the two new positive tests fail.
- Live, installed Claude Code 2.1.289, a prompt that makes the first turn end
  in prose: unfixed reader `cli_native_action_refused`, fixed reader
  `{"content":"hello"}`.
- Full suite and ruff: see the registry row.

## Not verified

- That the live job's third inference emitted exactly this reminder: its raw
  stream was never stored. The reproduction is the only CLI path found that
  yields this code without a native `tool_use`.
- No Cloud/Desktop run with this branch. Desktop picks it up only through a
  flyto-ai release and a Desktop build that pins it.

## Follow-ups

- flyto-cloud pins flyto-ai by archive commit (`requirements*.txt`, currently
  b216896). After this merges, bump that pin (archive SHA and sha256) and cut a
  Desktop release; until then AI Space jobs on Claude CLI still fail on the
  reminder. Cloud's own deploy does not run the CLI runtime.
