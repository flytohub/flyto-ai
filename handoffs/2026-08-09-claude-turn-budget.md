# Claude turn-budget handoff

Date: 2026-08-09

## Outcome

- `ClaudeCodeConfig` and its environment fallback now default to the existing
  validated ceiling of 100 turns.
- The parent workspace MCP config also sets
  `FLYTO_AI_CC_MAX_TURNS=100` explicitly.
- This repairs repeated fail-closed `turn_limit_exceeded` outcomes from complex
  single-file verifier jobs. It does not change the $5 default budget, allowed
  tools, workspace boundary, required checks, route lanes, audit, or rework
  ceiling.
- Guardian also now recognizes the exact repository dotfiles already present
  in its edit allowlist. This unblocks `.gitignore` hygiene without granting
  arbitrary dotfile or sensitive-path edits.

## Verification

- Run `pytest -q tests/test_agents.py tests/test_cli.py`.
- Run `ruff check flyto_ai/config.py tests/test_agents.py`.
- Regenerate and check documentation references.
- Trigger a safe supervisor worker reload after the current job is terminal,
  then inspect the next Claude process for `--max-turns 100`.
