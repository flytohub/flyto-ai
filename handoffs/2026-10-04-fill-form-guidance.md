# One-call form filling guidance

Owner: claude
Branch: claude/fill-form-basic
Date: 2026-10-04

## What changed

- `flyto_ai/tools/form_guidance.py` (new): `FILL_FORM_MODULE`,
  `FILL_FORM_GUIDANCE`, `core_provides_module()` (cached, reads Core's
  `get_module_info` through `core_tools._get_mcp_handler`), and
  `execute_module_description()`.
- `core_tools._enrich_core_tool_def`: the `execute_module` description gets the
  guidance appended when Core provides `browser.fill_form` and its own text does
  not already name it (flyto-core 2.38.0 does name it, so this is a no-op there).
  Schema and fingerprint unchanged.
- `browser_scope.get_browser_status()`: when a browser is running and Core
  provides the module, the hint ends with the guidance. Cloud's space-task
  agent appends this hint to its system prompt (`agent.py`), so it reaches the
  demo agent without a flyto-cloud change.
- Kept out of `core_tools.py` because of `tests/test_complexity_budget.py`.

## Why

The demo agent filled the ERP form one tool call per field (~100 s). It only
learned modules from tool descriptions, search and cloud's prompt (which names
browser.type/click for forms).

## Verified

- `tests/test_fill_form_guidance.py` 14 passed (fakes, plus one against the
  installed Core, which here lacks the module and answers False).
- End to end with `PYTHONPATH` = the flyto-core 2.38.0 branch: provides=True,
  execute_module description names fill_form once (not duplicated),
  `resolve_module_grade('browser.fill_form')` = WORKSPACE_WRITE / risk 3.
- Full suite, branch vs clean origin/main on this machine: the only new
  failure was the complexity budget, fixed by the extraction; other failures
  are common to both (local environment).
- ruff (CI selection and full on changed files), reference generation check.

## Not verified

- CI's Core lock (`stack-lock.json`, 4a76d3d) is unchanged, so CI only runs the
  fakes. No real LLM run.

## Follow-ups

- When the Core lock moves past 2.38.0, the real-core test starts asserting True.
