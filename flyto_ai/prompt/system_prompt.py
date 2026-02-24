# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""System prompt — three-layer architecture.

Layer A: POLICY — unbreakable rules, always on top
Layer B: BEHAVIOR — mode-specific execution flow (execute / yaml / toolless)
Layer C: GATES — quality checks, always on bottom

Assembled by build_system_prompt(mode, has_tools, ...).
"""
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Language detection — deterministic, no LLM involved
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Character ranges for deterministic CJK / Japanese / Korean detection
# ---------------------------------------------------------------------------
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_HIRAGANA_KATAKANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")

_TRAD_CHARS = frozenset(
    "這個們對從與會國為來說學時經過還進當沒動種長點開請問題應該讓機關體書門車東號頭條區結電鄉親發間"
    "幫搜尋資訊買賣圖檔確認設檢視訊號碼據處歷歲歐歸歡歲歎歌歷歸歡歲歎"
    "網絡線練總統編緣績營聯職聽號獲環產畫當發監視覺記訪設許認識試語課請論證變讓負費資質較載辦過運達選遠邊還鄰鑰閃關難電響領頻題類顯風飛驗體點龍"
)

# langdetect code → human-readable label (non-CJK languages)
_LANG_LABELS = {
    "en": "English",
    "fr": "French (fr)",
    "de": "German (de)",
    "es": "Spanish (es)",
    "pt": "Portuguese (pt)",
    "it": "Italian (it)",
    "ru": "Russian (ru)",
    "ar": "Arabic (ar)",
    "th": "Thai (th)",
    "vi": "Vietnamese (vi)",
    "id": "Indonesian (id)",
    "nl": "Dutch (nl)",
    "pl": "Polish (pl)",
    "tr": "Turkish (tr)",
    "uk": "Ukrainian (uk)",
    "hi": "Hindi (hi)",
    "sv": "Swedish (sv)",
    "da": "Danish (da)",
    "fi": "Finnish (fi)",
    "no": "Norwegian (no)",
    "cs": "Czech (cs)",
    "ro": "Romanian (ro)",
    "hu": "Hungarian (hu)",
    "el": "Greek (el)",
    "he": "Hebrew (he)",
    "bg": "Bulgarian (bg)",
    "hr": "Croatian (hr)",
    "sk": "Slovak (sk)",
    "sl": "Slovenian (sl)",
    "sr": "Serbian (sr)",
    "lt": "Lithuanian (lt)",
    "lv": "Latvian (lv)",
    "et": "Estonian (et)",
    "ms": "Malay (ms)",
    "tl": "Filipino (tl)",
    "bn": "Bengali (bn)",
    "ta": "Tamil (ta)",
    "te": "Telugu (te)",
    "mr": "Marathi (mr)",
    "af": "Afrikaans (af)",
    "sw": "Swahili (sw)",
    "ca": "Catalan (ca)",
}


def detect_language(text: str) -> str:
    """Detect reply language from user message text.

    Strategy (hybrid):
    1. CJK / Japanese / Korean — **regex** (deterministic, short-text safe)
    2. Everything else — ``langdetect`` (55+ languages, needs ~20 chars)
    3. Fallback — ``"English"``

    Returns a human-readable label like ``"English"``,
    ``"Traditional Chinese (zh-TW)"``, ``"Japanese (ja)"``, etc.
    """
    stripped = text.strip()
    if not stripped:
        return "English"

    # --- Phase 1: deterministic CJK / JP / KR detection via regex ---
    cjk_count = len(_CJK_RE.findall(stripped))
    jp_count = len(_HIRAGANA_KATAKANA_RE.findall(stripped))
    kr_count = len(_HANGUL_RE.findall(stripped))

    # Japanese: has hiragana/katakana (even if mixed with kanji)
    if jp_count > 0:
        return "Japanese (ja)"

    # Korean: has hangul
    if kr_count > 0:
        return "Korean (ko)"

    # Chinese: significant CJK ratio (>10% of text)
    total = len(stripped)
    if total > 0 and cjk_count / total >= 0.1:
        has_trad = any(c in _TRAD_CHARS for c in stripped)
        return (
            "Traditional Chinese (zh-TW)" if has_trad
            else "Simplified Chinese (zh-CN)"
        )

    # --- Phase 2: non-CJK → langdetect (needs ~20+ chars for accuracy) ---
    # Short Latin text is unreliable in langdetect; fall back to English.
    if len(stripped) < 15:
        return "English"

    try:
        from langdetect import detect
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0  # reproducible results
        code = detect(stripped)
        return _LANG_LABELS.get(code, code.capitalize())
    except Exception:
        return "English"

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

# CRITICAL: INTENT ROUTING — decide FIRST, then act

## search_modules vs browser search — KNOW THE DIFFERENCE
- search_modules() finds **flyto-core automation modules** (e.g. "string.uppercase", "image.resize").
  It does NOT search the web. It does NOT find people, news, lyrics, products, or any real-world info.
- To search the **web** for real-world information → use Browser Protocol (browser.launch → browser.goto Google → browser.snapshot).

## How to classify the user's request:
- User wants **web content** (search a person, topic, product, lyrics, news, weather, etc.) → Browser Protocol. Do NOT call search_modules.
- User wants to **automate a task** (resize image, send email, convert file, scrape a specific URL) → Execution Loop.
- User asks a **general question** you can answer → Answer directly without tools.

## Routing examples:
- "search for Jay Chou" / "find latest AI news" → Browser Protocol (real-world info)
- "resize image to 800x600" / "send an email" → Execution Loop (automation task)
- "what is Python?" → Answer directly (general knowledge)

# ⛔ HARD: FAILURE HANDLING — NEVER FABRICATE RESULTS
- When execute_module returns ok=false → the action **FAILED**. You MUST acknowledge the failure.
- NEVER pretend a failed action succeeded. NEVER fabricate data that wasn't in the tool result.
- If browser.launch fails → STOP. Do NOT call browser.goto or browser.snapshot — they will also fail.
- If all tool calls in a sequence failed → tell the user what went wrong with the actual error message.
- NEVER construct URLs, search results, or page content from your own knowledge when the browser failed.
- When reporting failure: (1) state which module failed, (2) include the error reason, (3) suggest a fix.

# EXECUTION LOOP (for automation tasks only)

1. DISCOVER — search_modules(query) to find relevant modules
2. SCHEMA — get_module_info(module_id) for EACH module before use
   ⛔ NEVER call execute_module on a module you haven't called get_module_info for
3. EXECUTE — execute_module(module_id, params) step by step
4. CHECK — read the result carefully. If ok=false → stop and report the error. Retry ONCE only if the error suggests a fixable param issue.
5. RESPOND — result summary in user's language + ```yaml reusable workflow

## Browser Protocol (for web search / scrape / browse)
1. browser.launch → get a browser session
   ⛔ If ok=false → STOP. Do NOT call goto or snapshot.
2. browser.goto(url) → navigate to the page
   ⛔ If ok=false → STOP. Report the error.
3. browser.snapshot → read page content and find real selectors
4. Extract and summarize actual content FROM the snapshot — not from your knowledge
5. Need to interact (search, login, fill forms)? \
Use selectors FROM step 3: browser.type / browser.click / browser.select → then snapshot again
6. Repeat 3-5 until you have the answer the user actually needs. \
A list of results is not the final answer — click into detail pages for status, price, availability, etc.
- Google shortcut: browser.goto("https://www.google.com/search?q=URL_ENCODED_QUERY")
- ⛔ NEVER guess selectors — only use what browser.snapshot actually shows
- The user CANNOT see the browser. Relay actual content (titles, data, facts). \
NEVER just return a URL. NEVER make up data that wasn't in the tool result.
- Do NOT call browser.close — the runtime handles cleanup.

## API Fallback (when browser is unavailable)
If browser.launch fails (chromium not installed, etc.), try these API alternatives:
1. get_module_info("core.api.google_search") → execute_module("core.api.google_search", {{"query": "..."}})
   Requires GOOGLE_API_KEY + GOOGLE_CSE_ID env vars.
2. get_module_info("core.api.serpapi_search") → execute_module("core.api.serpapi_search", {{"query": "..."}})
   Requires SERPAPI_KEY env var.
3. get_module_info("core.api.http_get") → execute_module("core.api.http_get", {{"url": "..."}})
   No API key needed — direct HTTP GET.
If all methods fail, report clearly which were tried and what went wrong."""

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
    reply_language: Optional[str] = None,
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
    reply_language : str, optional
        Forced reply language (e.g. ``"English"``).
        When set, a hard override is prepended to the prompt.
        Use :func:`detect_language` to compute this from user input.
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

    # Deterministic language override — prepend at very top
    if reply_language:
        prompt = "⛔ REPLY IN {}. All non-code text must be in {}.\n\n".format(
            reply_language, reply_language
        ) + prompt

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
