# Implementation Workflow

Implementation rules:
- Prefer additive changes to public tool/provider contracts.
- Keep `flyto-core` access behind `flyto_ai.tools.core_tools`.
- Add tests in the same change as behavior.
- Never write credentials or runtime secrets.

Exit:
- Focused tests pass and changed behavior is documented.
