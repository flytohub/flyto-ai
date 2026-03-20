# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Blueprint-first router — system-level enforcement, not prompt reliance.

Two layers:
1. prepare()  — pre-resolve blueprint, inject into system prompt
2. guard()    — intercept first tool call, redirect to use_blueprint
"""
import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def pre_resolve(message: str) -> str:
    """Search blueprints BEFORE the LLM call.

    Returns a prompt snippet to inject into admin_addition.
    If a matching blueprint is found, the snippet tells the LLM
    to call use_blueprint as its first action.
    """
    try:
        from flyto_blueprint import get_engine
        engine = get_engine()
        results = engine.search(message)
        if not results:
            return ""

        top = results[0]
        # Only enforce blueprint if score is very high AND has been used successfully
        score = top.get("score", 0)
        use_count = top.get("use_count", 0)
        if score < 80 or use_count < 2:
            return ""
        bp_id = top.get("id", "")
        bp_name = top.get("name", "")
        args_info = top.get("args", {})
        args_lines = []
        for name, meta in args_info.items():
            req = "REQUIRED" if meta.get("required") else "optional"
            args_lines.append("  - {} ({}): {}".format(name, req, meta.get("description", "")))
        args_str = "\n".join(args_lines) if args_lines else "  (no args)"

        alternatives = ""
        if len(results) > 1:
            alt_ids = [r.get("id") for r in results[1:3]]
            alternatives = "\nAlternatives: {}".format(", ".join(alt_ids))

        return (
            "\n\n## PRE-RESOLVED BLUEPRINT (use this)\n"
            "A matching blueprint was found for the user's request.\n"
            "⛔ You MUST call use_blueprint(blueprint_id=\"{}\", args={{...}}) as your FIRST action.\n"
            "Fill in the args from the user's message:\n"
            "{}\n"
            "Blueprint: {} — {}{}"
        ).format(bp_id, args_str, bp_id, bp_name, alternatives)

    except Exception:
        return ""


async def guard(
    func_name: str,
    func_args: dict,
    user_message: str,
    dispatch: Callable,
) -> Optional[Dict[str, Any]]:
    """Blueprint guard — intercept the first non-blueprint tool call.

    If the LLM skips list_blueprints and jumps directly to
    search_modules / execute_module / get_module_info, this guard
    searches blueprints using the original user message and returns
    a redirect hint.

    Returns:
        dict with _blueprint_redirect=True if a match is found.
        None to let the call through normally.
    """
    if func_name not in ("search_modules", "execute_module", "get_module_info"):
        return None

    if not user_message:
        return None

    try:
        bp_result = await dispatch("list_blueprints", {"query": user_message})
        blueprints = bp_result.get("blueprints", [])
        if not isinstance(blueprints, list) or not blueprints:
            return None

        top = blueprints[0]
        # Only redirect if blueprint is proven (high score + used successfully)
        if top.get("score", 0) < 80 or top.get("use_count", 0) < 2:
            return None
        return {
            "ok": True,
            "_blueprint_redirect": True,
            "message": (
                "STOP. A matching blueprint was found. "
                "You MUST call use_blueprint(blueprint_id=\"{}\", args={{...}}) now. "
                "Blueprint args: {}. "
                "Fill args from the user message and call use_blueprint."
            ).format(
                top.get("id", ""),
                json.dumps(list(top.get("args", {}).keys())),
            ),
            "blueprints": blueprints[:3],
        }
    except Exception as e:
        logger.debug("Blueprint guard failed: %s", e)
        return None


def init_storage() -> None:
    """Initialize flyto-blueprint with SQLite storage for local persistence."""
    try:
        from flyto_blueprint import get_engine
        from flyto_blueprint.storage.sqlite import SQLiteBackend
        get_engine(storage=SQLiteBackend())
    except ImportError:
        pass


def feedback(
    tool_calls: List[Dict[str, Any]],
    execution_results: List[Dict[str, Any]],
    user_message: str,
) -> None:
    """Closed-loop blueprint learning. Pure code — zero LLM involvement.

    1. If a blueprint was used → report outcome (score +5 / -10)
    2. If execution succeeded with 3+ steps (no blueprint) → learn new blueprint
    """
    try:
        from flyto_blueprint import get_engine
    except ImportError:
        return

    engine = get_engine()
    all_ok = all(r.get("ok", False) for r in execution_results)

    # Report outcome if a blueprint was used
    used_blueprint_id = None
    for tc in tool_calls:
        if tc.get("function") == "use_blueprint":
            used_blueprint_id = tc.get("arguments", {}).get("blueprint_id", "")
            break

    if used_blueprint_id:
        try:
            engine.report_outcome(used_blueprint_id, success=all_ok)
            logger.info("Blueprint outcome: %s %s", used_blueprint_id, "OK" if all_ok else "FAIL")
        except Exception as e:
            logger.debug("Blueprint report_outcome failed: %s", e)

    # Learn new blueprint from successful execution
    if not all_ok or len(execution_results) < 3:
        return

    steps = []
    for i, r in enumerate(execution_results):
        mid = r.get("module_id", "")
        if not mid:
            continue
        params = r.get("arguments", {}).get("params", {})
        steps.append({"id": "step_{}".format(i + 1), "module": mid, "params": params})

    if len(steps) < 3:
        return

    workflow = {"name": user_message[:80], "steps": steps}
    categories = list({s["module"].split(".")[0] for s in steps if "." in s["module"]})

    try:
        engine.learn_from_execution(workflow=workflow, name=user_message[:80], tags=categories)
        logger.info("Blueprint learned: %s (%d steps)", user_message[:40], len(steps))
    except Exception as e:
        logger.debug("Blueprint learn failed: %s", e)
