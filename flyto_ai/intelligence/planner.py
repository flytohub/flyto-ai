# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Deterministic Planner — maps user intent to executable module sequences.

Replaces LLM freestyle module selection with a deterministic pipeline:
1. Intent extraction (1 LLM call, structured output)
2. Recipe matching (zero LLM — Blueprint + Knowledge)
3. Parameter filling (zero LLM — from intent params + defaults)
4. Contract validation (zero LLM — ContractEngine)
5. Execution (zero LLM — sequential module dispatch)

Falls back to LLM freestyle when no recipe matches.
"""
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Common intent → module recipes (built-in, no learning needed)
BUILTIN_RECIPES: Dict[str, List[Dict[str, Any]]] = {
    "open_website": [
        {"module": "browser.launch", "params": {"headless": False, "channel": "chrome"}},
        {"module": "browser.goto", "params_from": ["url"]},
    ],
    "search_on_website": [
        {"module": "browser.snapshot", "params": {}},
        {"module": "browser.type", "params_from": ["search_text"], "selector_from_snapshot": True},
        {"module": "browser.click", "params": {}, "selector_from_snapshot": True, "selector_hint": "search button"},
    ],
    "click_element": [
        {"module": "browser.snapshot", "params": {}},
        {"module": "browser.click", "params_from": ["target"], "selector_from_snapshot": True},
    ],
    "extract_page": [
        {"module": "browser.snapshot", "params": {}},
    ],
    "screenshot": [
        {"module": "browser.screenshot", "params_from": ["output_path"]},
    ],
}



async def extract_intent_llm(message: str, provider) -> Optional[Dict[str, Any]]:
    """Extract structured intent using 1 cheap LLM call. Language-agnostic.

    Returns {"action": "...", "target": "...", "query": "...", "params": {...}}
    or None if the message is a question/conversation (not an action).
    """
    prompt = (
        "Extract the user's ACTION intent as JSON. If it's a question or conversation, return null.\n\n"
        "Possible actions: navigate, search, click, play, screenshot, download, upload, "
        "resize, convert, generate, send, extract, scrape\n\n"
        "For navigate: resolve the site name to a full URL.\n\n"
        "User: {}\n\n"
        "Return ONLY valid JSON (no markdown), example:\n"
        '{{"action":"navigate","url":"https://www.youtube.com","query":""}}\n'
        '{{"action":"search","url":"https://www.youtube.com","query":"周杰倫"}}\n'
        '{{"action":"click","target":"first video"}}\n'
        '{{"action":"resize","target":"image.png","params":{{"width":800,"height":600}}}}\n'
        "null (for questions/conversation)"
    ).format(message)

    try:
        response, _, _, _ = await provider.chat(
            [{"role": "user", "content": prompt}],
            system_prompt="You extract intent as JSON. Return ONLY JSON or null.",
            tools=[], dispatch_fn=None, max_rounds=1,
        )
        if not response or response.strip() == "null":
            return None

        import json
        data = json.loads(response.strip().removeprefix("```json").removesuffix("```").strip())
        if not isinstance(data, dict) or "action" not in data:
            return None
        return data
    except Exception:
        return None


def extract_intent(message: str) -> Optional[Dict[str, Any]]:
    """Synchronous intent extraction — pure data-driven, zero hardcoding.

    Priority:
    1. Explicit URL in message → navigate
    2. Blueprint (learned from past success)
    3. Module registry (395 module schemas)
    4. Returns None → caller uses extract_intent_llm()
    """
    msg = message.strip()

    # 1. Explicit URL → navigate directly
    url_match = re.search(r'(https?://\S+)', msg)
    if url_match:
        return {"intent": "open_website", "url": url_match.group(1), "site": url_match.group(1)}

    # Domain pattern (xxx.com) → navigate
    domain_match = re.search(r'(\w+\.\w{2,}(?:\.\w{2,})?)', msg)
    if domain_match:
        return {"intent": "open_website", "url": "https://" + domain_match.group(1), "site": domain_match.group(1)}

    # 2. Blueprint (learned from past executions)
    blueprint_intent = _match_from_blueprint(msg.lower())
    if blueprint_intent:
        return blueprint_intent

    # 3. Module registry (395 module descriptions)
    registry_intent = _match_from_registry(msg.lower())
    if registry_intent:
        return registry_intent

    # No sync match → caller should try extract_intent_llm()
    return None


def _match_from_blueprint(msg: str) -> Optional[Dict[str, Any]]:
    """Match user message against learned blueprints."""
    try:
        from flyto_blueprint import get_engine
        engine = get_engine()
        results = engine.search(msg)
        if not results:
            return None

        top = results[0]
        if top.get("score", 0) < 50 or top.get("use_count", 0) < 1:
            return None

        steps_data = top.get("steps", [])
        if not steps_data:
            return None

        return {
            "intent": "blueprint",
            "blueprint_id": top.get("id", ""),
            "steps": steps_data,
        }
    except Exception:
        return None


def _match_from_registry(msg: str) -> Optional[Dict[str, Any]]:
    """Match user message to a single module using the module registry.

    Searches module names/descriptions for keyword overlap.
    Returns a single-module intent if found.
    """
    try:
        from core.modules.registry import ModuleRegistry
        all_mods = ModuleRegistry.get_all_metadata()
    except ImportError:
        return None

    if not all_mods:
        return None

    # Keyword search — match module_id and description
    q = msg.lower()
    candidates = []
    for mid, meta in all_mods.items():
        desc = (meta.get("description", "") or "").lower()
        mid_lower = mid.lower()
        score = 0
        for word in q.split():
            if len(word) < 2:
                continue
            if word in mid_lower:
                score += 10
            if word in desc:
                score += 5
        if score > 0:
            candidates.append((mid, meta, score))

    if not candidates:
        return None

    # Sort by score, take the top
    candidates.sort(key=lambda x: x[2], reverse=True)
    top_mid, top_meta, top_score = candidates[0]

    # Require a minimum score to avoid false matches
    if top_score < 10:
        return None

    # Get param schema
    try:
        from core.mcp_handler import get_module_info
        info = get_module_info(module_id=top_mid)
        schema = info.get("params_schema", {}) if info else {}
    except Exception:
        schema = {}

    # Try to extract param values from the message using simple heuristics
    params = _extract_params_from_message(msg, schema)

    return {
        "intent": "single_module",
        "module_id": top_mid,
        "params": params,
        "schema": schema,
    }


def _extract_params_from_message(msg: str, schema: dict) -> dict:
    """Extract parameter values from message text based on schema hints.

    Simple heuristic extraction — not LLM, just pattern matching.
    """
    params = {}

    for param_name, param_def in schema.items():
        ptype = param_def.get("type", "")
        desc = (param_def.get("description", "") or "").lower()

        # Number extraction (width, height, size, count, etc.)
        if ptype in ("integer", "number", "int", "float"):
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', msg)
            if numbers:
                if "width" in param_name or "寬" in param_name:
                    params[param_name] = int(float(numbers[0]))
                    if len(numbers) > 1:
                        # Look for paired height
                        height_key = param_name.replace("width", "height")
                        if height_key in schema:
                            params[height_key] = int(float(numbers[1]))
                elif "height" in param_name or "高" in param_name:
                    if param_name not in params:
                        params[param_name] = int(float(numbers[-1]))
                elif param_name not in params:
                    params[param_name] = int(float(numbers[0])) if ptype in ("integer", "int") else float(numbers[0])

        # URL extraction
        elif "url" in param_name.lower() or "url" in desc:
            url_match = re.search(r'https?://\S+', msg)
            if url_match:
                params[param_name] = url_match.group(0)

        # File path extraction
        elif "path" in param_name.lower() or "file" in param_name.lower():
            path_match = re.search(r'[/~][\w./\-]+\.\w+', msg)
            if path_match:
                params[param_name] = path_match.group(0)

        # Text/string extraction — use the full message as fallback
        elif ptype == "string" and ("text" in param_name or "content" in param_name or "query" in param_name):
            # Remove common command words, keep the content
            content = re.sub(r'^(?:請|幫我|help me|please)\s*', '', msg, flags=re.IGNORECASE)
            content = re.sub(r'^(?:把|將|convert|resize|download|upload)\s*', '', content, flags=re.IGNORECASE)
            if content and content != msg:
                params[param_name] = content.strip()

    return params


def plan_execution(
    intent: Dict[str, Any],
    has_browser: bool = False,
) -> List[Dict[str, Any]]:
    """Plan a deterministic execution sequence from an intent.

    Returns a list of steps: [{"module": "...", "params": {...}}, ...]
    """
    intent_type = intent.get("intent", "")
    steps = []

    if intent_type == "open_website":
        if not has_browser:
            steps.append({"module": "browser.launch", "params": {"headless": False, "channel": "chrome"}})
        steps.append({"module": "browser.goto", "params": {"url": intent["url"]}})
        steps.append({"module": "browser.snapshot", "params": {}})

    elif intent_type == "open_and_search":
        if not has_browser:
            steps.append({"module": "browser.launch", "params": {"headless": False, "channel": "chrome"}})
        steps.append({"module": "browser.goto", "params": {"url": intent["url"]}})
        steps.append({"module": "browser.snapshot", "params": {}})
        steps.append({
            "module": "browser.type",
            "params": {"text": intent["search_text"], "press_enter": True},
            "needs_selector": True,
            "selector_hint": "search input",
        })
        steps.append({"module": "browser.snapshot", "params": {}})

    elif intent_type == "search_on_website":
        steps.append({"module": "browser.snapshot", "params": {}})
        steps.append({
            "module": "browser.type",
            "params": {"text": intent["search_text"], "press_enter": True},
            "needs_selector": True,
            "selector_hint": "search input",
        })
        # press_enter submits the search — no need for a separate click
        steps.append({"module": "browser.snapshot", "params": {}})

    elif intent_type == "click_element":
        steps.append({"module": "browser.snapshot", "params": {}})
        steps.append({
            "module": "browser.click",
            "params": {},
            "needs_selector": True,
            "selector_hint": intent.get("target", ""),
        })
        steps.append({"module": "browser.snapshot", "params": {}})

    elif intent_type == "single_module":
        steps.append({
            "module": intent["module_id"],
            "params": intent.get("params", {}),
        })

    elif intent_type == "blueprint":
        for bp_step in intent.get("steps", []):
            steps.append({
                "module": bp_step.get("module", bp_step.get("module_id", "")),
                "params": bp_step.get("params", {}),
            })

    return steps


async def execute_plan(
    steps: List[Dict[str, Any]],
    dispatch: Callable,
) -> Tuple[List[Dict[str, Any]], str]:
    """Execute a deterministic plan step by step.

    For steps that need_selector, extracts the selector from the
    previous snapshot result using hint matching.

    Returns (execution_results, summary_text).
    """
    results = []
    last_snapshot = {}  # Full structured snapshot result

    for step_idx, step in enumerate(steps):
        module_id = step["module"]
        params = dict(step.get("params", {}))
        logger.debug("execute_plan step %d: %s", step_idx, module_id)

        # Resolve selector from previous snapshot
        if step.get("needs_selector") and last_snapshot:
            hint = step.get("selector_hint", "")
            element_type = "input" if module_id == "browser.type" else "button" if module_id == "browser.click" else ""
            selector = _find_selector_from_structured(last_snapshot, hint, element_type)
            if selector:
                params["selector"] = selector
            else:
                # Can't find selector — skip this step
                results.append({
                    "module_id": module_id,
                    "ok": False,
                    "error": "Could not find selector for: {}".format(hint),
                })
                continue

        # Execute
        try:
            result = await dispatch("execute_module", {
                "module_id": module_id,
                "params": params,
            })
        except Exception as e:
            logger.error("execute_plan dispatch error step %d (%s): %s", step_idx, module_id, e)
            results.append({"module_id": module_id, "ok": False, "error": str(e)})
            break

        # Determine success: check ok, status, or absence of error
        is_ok = result.get("ok")
        if is_ok is None:
            is_ok = result.get("status") == "success" or (
                "error" not in result and "Error" not in str(result.get("message", ""))
            )

        exec_result = {
            "module_id": module_id,
            "ok": bool(is_ok),
            "error": result.get("error", ""),
        }
        results.append(exec_result)
        logger.debug("step %d %s ok=%s", step_idx, module_id, is_ok)

        # Track snapshot for selector resolution
        if module_id in ("browser.snapshot", "browser.goto") and is_ok:
            last_snapshot = result

        # Stop on failure (except snapshot failures)
        if not is_ok and module_id != "browser.snapshot":
            break

    # Build summary
    ok_count = sum(1 for r in results if r["ok"])
    fail_count = sum(1 for r in results if not r["ok"])
    summary = "Executed {} steps: {} ok, {} failed.".format(
        len(results), ok_count, fail_count,
    )

    return results, summary


def _resolve_url(text: str) -> Optional[str]:
    """Resolve a site name or URL to a full URL. No hardcoded site list."""
    text = text.strip().rstrip("。.，,")

    if text.startswith("http://") or text.startswith("https://"):
        return text

    if "." in text:
        return "https://" + text

    # Treat as site name → add .com (LLM already resolved the name)
    if text and len(text) > 1:
        return "https://www.{}.com".format(text)

    return None


def _find_selector_from_structured(snapshot_result: dict, hint: str, element_type: str = "") -> Optional[str]:
    """Find a CSS selector from structured snapshot data (inputs, buttons, links).

    The snapshot returns structured arrays:
      inputs: [{"selector": "input[name=search_query]", "label": "Search", ...}]
      buttons: [{"selector": "[data-flyto-hint=7]", "text": "Search", ...}]
      links: [{"selector": "a#video-title", "text": "Video Title", ...}]

    This is 100x more reliable than parsing text.
    """
    if not hint or not snapshot_result:
        return None

    hint_lower = hint.lower()

    # Determine which element arrays to search
    search_in = []
    if element_type == "input" or "input" in hint_lower or "search" in hint_lower:
        search_in.append(("inputs", snapshot_result.get("inputs", [])))
    if element_type == "button" or "button" in hint_lower or "search" in hint_lower:
        search_in.append(("buttons", snapshot_result.get("buttons", [])))
    if element_type == "link" or "click" in hint_lower or "play" in hint_lower:
        search_in.append(("links", snapshot_result.get("links", [])))

    # If no specific type, search all
    if not search_in:
        search_in = [
            ("inputs", snapshot_result.get("inputs", [])),
            ("buttons", snapshot_result.get("buttons", [])),
            ("links", snapshot_result.get("links", [])),
        ]

    # Handle ordinal hints (第一個, first, 1 etc.) — pick by index
    ordinal_map = {"第一": 0, "第二": 1, "第三": 2, "第四": 3, "第五": 4,
                   "first": 0, "second": 1, "third": 2, "1": 0, "2": 1, "3": 2}
    for word, idx in ordinal_map.items():
        if word in hint_lower:
            # Pick the Nth visible link/button
            for _, elements in search_in:
                visible = [el for el in elements
                           if el.get("selector") and el.get("rect", {}).get("top", -999) > 0]
                if idx < len(visible):
                    return visible[idx].get("selector")
            break

    best_selector = None
    best_score = 0

    for source_type, elements in search_in:
        for el in elements:
            selector = el.get("selector", "")
            if not selector:
                continue

            # Skip offscreen elements
            rect = el.get("rect", {})
            if rect.get("top", 0) < -500:
                continue

            # Score by matching hint against element attributes
            score = 0
            text = (el.get("text", "") or "").lower()
            label = (el.get("label", "") or "").lower()
            name = (el.get("name", "") or "").lower()
            placeholder = (el.get("placeholder", "") or "").lower()
            el_id = (el.get("id", "") or "").lower()

            searchable = " ".join([text, label, name, placeholder, el_id])

            for word in hint_lower.split():
                if word in searchable:
                    score += 10
                if word in selector.lower():
                    score += 5

            # Bonus for exact matches
            if hint_lower == text or hint_lower == label:
                score += 50
            if hint_lower in text or hint_lower in label:
                score += 20

            # Type-specific bonuses
            if source_type == "inputs" and ("input" in hint_lower or "search" in hint_lower or "type" in hint_lower):
                score += 5
            if source_type == "buttons" and ("button" in hint_lower or "submit" in hint_lower):
                score += 5

            if score > best_score:
                best_score = score
                best_selector = selector

    return best_selector
