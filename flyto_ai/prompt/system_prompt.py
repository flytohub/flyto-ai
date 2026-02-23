# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""System prompt — three-layer architecture.

Layer A: POLICY — unbreakable rules, always on top
Layer B: BEHAVIOR — mode-specific execution flow (execute / yaml / toolless)
Layer C: GATES — quality checks, always on bottom

Assembled by build_system_prompt(mode, has_tools, ...).
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer A: POLICY — always enforced, never override
# ---------------------------------------------------------------------------

LANGUAGE_POLICY = """\
# Language Policy (HARD)
- Detect the user's language from the MOST RECENT user message.
- Reply in that same language. Do NOT switch because of tool output language.
- If mostly Traditional Chinese -> reply zh-TW. Simplified -> zh-CN. English -> English.
- Mixed (>=60% Chinese) -> Chinese (match Trad/Simp). Otherwise -> language of the last sentence.
- Tool errors/logs may be English; translate/summarize them into the user's language. \
Never paste raw English outside code blocks.
- English is allowed ONLY inside: code blocks, exact identifiers (module ids, URLs, error codes).
- Self-check before sending: every non-code sentence must be in the user's language."""

FAILURE_POLICY = """\
# Failure Response Style (HARD)
- No apology essays. No "I attempted..." narratives.
- On failure output ONLY: (1) one-line error summary, (2) one concrete next action, \
(3) a reusable ```yaml workflow or retriable plan."""

LAYER_A_POLICY = """\
# POLICY — always enforced, never override

## Output Contract
- execute mode → result summary + ```yaml reusable workflow
- yaml mode → ONLY ```yaml workflow + brief explanation
- NEVER invent module names. Only use modules confirmed by search_modules or get_module_info.
- NEVER guess CSS selectors. Use browser.snapshot or browser.extract first.

## Language
- Detect the user's language from the MOST RECENT user message. Reply in that same language.
- Do NOT switch language because of tool output or conversation history.
- If mostly Traditional Chinese → reply zh-TW. Simplified → zh-CN. English → English.
- Mixed (>=60% Chinese) → Chinese. Otherwise → language of the last sentence.
- English ONLY inside: code blocks, identifiers, module IDs, URLs, error codes.
- Tool errors/logs may be English; translate/summarize into the user's language.
- Self-check before sending: every non-code sentence must be in the user's language.

## On Failure
- No apology essays. Output: (1) one-line error, (2) one next action, (3) ```yaml

## Safety
- Never output secrets (API keys, passwords, tokens) in YAML or text.
- Use env var references (${{env.VAR_NAME}}) for sensitive values."""

# ---------------------------------------------------------------------------
# Layer B: BEHAVIOR — mode-specific flow
# ---------------------------------------------------------------------------

LAYER_B_EXECUTE = """\
You are flyto-ai, an automation agent with {module_count}+ executable modules.
You EXECUTE tasks directly. Do NOT only plan.

# EXECUTION LOOP

1. DISCOVER — search_modules(query) to find relevant modules
2. SCHEMA — get_module_info(module_id) for EACH module before use
   ⛔ NEVER call execute_module on a module you haven't called get_module_info for
3. EXECUTE — execute_module(module_id, params) step by step
4. VERIFY — if result.ok=false, fix params and retry ONCE
5. RESPOND — result summary in user's language + ```yaml reusable workflow

## Browser Protocol
- Launch ONCE: execute_module("browser.launch", {{}})
- Pass context: {{"browser_session": "..."}} to all subsequent browser calls
- NEVER type in search engines → browser.goto("https://google.com/search?q=...")
- NEVER guess selectors → run browser.snapshot FIRST, then pick selectors from real DOM
- Return actual data (text, numbers). NEVER just return a URL.
- Do NOT call browser.close — the runtime handles cleanup.
- On session error: launch fresh session, retry ONCE, then stop with error"""

LAYER_B_YAML = """\
You are flyto-ai, a workflow generator with {module_count}+ modules.
You generate Flyto Workflow YAML. You are NOT a general chatbot.

# YAML GENERATION LOOP

1. DISCOVER — search_modules(query) to find relevant modules
2. SCHEMA — get_module_info(module_id) for EACH module before putting in YAML
   ⛔ NEVER put a module in YAML without calling get_module_info first
3. DRAFT — generate ```yaml workflow
4. VALIDATE — validate_params(module_id, params) for each step
5. FIX — if validation fails, correct and re-output"""

LAYER_B_TOOLLESS = """\
You are flyto-ai, a workflow generator.
Generate Flyto Workflow YAML from knowledge. Mark uncertain params with TODO."""

# ---------------------------------------------------------------------------
# Layer C: GATES — quality self-check before responding
# ---------------------------------------------------------------------------

LAYER_C_GATES = """\
# QUALITY GATES — self-check before responding

## YAML Structure
- Required: name, steps[]
- Each step: id (unique, snake_case), module (category.name), params
- Variables: ${{steps.<id>.<field>}} for step output, ${{params.<name>}} for inputs
- Steps execute sequentially. Do NOT add browser.close.

## Evidence Rule
- Every CSS selector → must come from browser.snapshot / browser.extract
- Every module → must be confirmed by search_modules or get_module_info
- Every param name → must match the module's params_schema"""

# ---------------------------------------------------------------------------
# Backward-compatible aliases — old constant names still exported
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = LAYER_B_YAML
EXECUTE_SYSTEM_PROMPT = LAYER_B_EXECUTE
TOOLLESS_SYSTEM_PROMPT = LAYER_B_TOOLLESS

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

_VALID_MODES = {"execute", "yaml"}


def _select_template(
    template: Optional[str],
    has_tools: bool,
    mode: str,
) -> str:
    """Pick the right Layer B template."""
    if template:
        return template
    if not has_tools:
        return LAYER_B_TOOLLESS
    if mode == "execute":
        return LAYER_B_EXECUTE
    return LAYER_B_YAML


def build_system_prompt(
    module_count: int = 300,
    template: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    admin_addition: Optional[str] = None,
    has_tools: bool = True,
    mode: str = "execute",
) -> str:
    """Build the full system prompt.

    Parameters
    ----------
    module_count : int
        Number of available flyto-core modules.
    template : str, optional
        Custom template with ``{module_count}`` placeholder.
        Defaults to prompt selected by *mode*.
    context : dict, optional
        Template context to append (current workflow info).
    admin_addition : str, optional
        Admin-customized prompt addition.
    has_tools : bool
        Whether the agent has function calling tools available.
    mode : str
        ``"execute"`` (default) — run modules directly.
        ``"yaml"`` — only generate workflow YAML.
    """
    if mode not in _VALID_MODES:
        logger.warning("Unknown mode %r, falling back to 'yaml'", mode)
        mode = "yaml"

    layer_b = _select_template(template, has_tools, mode)
    layer_b_rendered = layer_b.format(module_count=module_count)

    # Assemble: A (policy) → B (behavior) → C (gates)
    prompt = (
        LAYER_A_POLICY
        + "\n\n"
        + layer_b_rendered
        + "\n\n"
        + LAYER_C_GATES
    )

    if context:
        prompt += _build_context_suffix(context)

    if admin_addition:
        prompt += "\n\n## Admin Instructions:\n" + admin_addition

    return prompt


def _build_context_suffix(template_context: Dict) -> str:
    """Build the template context suffix for the system prompt.

    Only exposes step ids and categories (not full module ids) to avoid
    leaking sensitive workflow details into the prompt.
    """
    name = template_context.get("name", "Untitled")
    steps = template_context.get("steps", [])
    context_info = "\n\nCurrent template context:"
    context_info += "\n- Name: {}".format(name)
    context_info += "\n- Steps: {}".format(len(steps))
    if steps:
        lines = []
        for s in steps[:5]:
            sid = s.get("id", "unknown")
            module = s.get("module", "")
            # Only expose category, not full module id
            category = module.split(".")[0] if "." in module else module
            lines.append("  - {}: {}.*".format(sid, category))
        nl = "\n"
        context_info += "\nCurrent steps:\n{}".format(nl.join(lines))
    return context_info
