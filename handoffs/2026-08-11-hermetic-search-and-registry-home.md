# 2026-08-11 Hermetic coding_search boundary and neutral registry HOME

Owner: claude
Branch: unverified — `git` is denied in this session, so the branch was not read.

## Scope

Make two full-suite tests hermetic. Test-only change; no production source,
authority, bridge, or config file was touched, and unrelated dirty changes in
the tree were left alone.

## Behavior

- `tests/test_coding_control.py::test_workspace_search_contract_is_literal_and_guides_empty_results`
  no longer needs a ripgrep install on the host. Only the process boundary is
  faked (`shutil.which` and `WorkspaceTools._run_process`); the argv under
  assertion is still the one `WorkspaceTools.search` builds, so the test fails
  if `--fixed-strings` or the `!.env` / `!**/.git/**` / `!**/.ssh/**` /
  `!**/.aws/**` credential globs are dropped. The fake matches literally, as
  `--fixed-strings` promises. The test then points `which` at `None` and still
  requires `WorkspaceViolation` mentioning ripgrep, so faking the boundary
  cannot soften the precondition in front of it.
- `tests/test_workspace_root_authority.py::test_the_registry_location_is_neutral_and_outside_worktrees`
  pins `HOME` to a neutral temporary directory and asserts token neutrality on
  `default_registry_root().relative_to(home)` — the suffix the protocol
  chooses. Previously the assertion also read the developer's home path, so a
  home directory spelling `flyto`, `claude`, or `codex` would fail the test,
  and any other home passed it for a reason the protocol does not own.

## Verification

Not performed in this session. Every interpreter and VCS invocation was denied
by policy: `python`, `python3`, `/usr/bin/python3 -c "print(1)"`, `pytest`,
`ruff`, `.venv/bin/python …`, and read-only `git status` / `git diff` all
returned `This command requires approval`. A subagent dispatched to run the
commands verbatim reported the same denial. Treat the following as unrun:

- both focused tests
- full `pytest`
- `ruff check`
- `generate_reference` action and its `--check` gate
- diff review

Codex independently reported `3391 passed, 17 skipped` after these two test
edits. That result was not reproduced here and is recorded as Codex's, not as
a lane this session completed.

## Open — generated reference is stale

A concurrent, preserved `flyto_ai/coding/service.py` change left
`docs/reference/python/coding.md` stale. Diagnosed statically:

- One line was inserted at or above `service.py:265`, so every `service.py`
  source link in `coding.md` is short by exactly one. The doc records
  `service.py:265, 316, 324, 336, 353` for `CodingServiceError`,
  `CodingServiceBusy`, `CodingCapacityUnavailable`, `VerificationRequired`,
  and `VerificationContractInvalid`; the source now has them at `266, 317,
  325, 337, 354`. There are 218 such stale references.
- The inventories are *not* stale. `coding.md` says 327 declared symbols and
  `docs/reference/python/README.md` says `32 modules, 327 top-level symbols,
  576 methods`; the 32 `flyto_ai/coding/*.py` modules still sum to 327
  top-level symbols. No symbol was added or removed, so the drift is the
  `#L<lineno>` links rather than the counts.

This was deliberately not hand-patched. Rewriting 218 references by hand
produces a guess at generator output, not generator output, and `--check` —
the lane that actually certifies currency — could not be run to confirm it.
`AGENTS.md` requires a lane outcome to come from a completed allowlisted call.

To close on one exact current revision, run in `/Users/chester/flytohub/flyto-ai`:

1. `.venv/bin/python scripts/generate_reference.py` (allowlisted
   `generate_reference` action in `.flyto/coding.yaml`)
2. `.venv/bin/python scripts/generate_reference.py --check` (required
   `generated_reference` check; the generator does not certify its own output)
3. `.venv/bin/python -m pytest -q`
4. `ruff check .`
5. `git diff` for review

Step 1 is expected to bump the `service.py` links in `coding.md` by one and
leave the inventory counts alone. Any larger delta is real generator output
and should be reported rather than reconciled by hand.

No stage, commit, push, reset, clean, or MCP call was made.
