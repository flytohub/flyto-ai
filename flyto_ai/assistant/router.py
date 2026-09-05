# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Blueprint-first router — system-level enforcement, not prompt reliance.

Two layers:
1. prepare()  — pre-resolve blueprint, inject into system prompt
2. guard()    — intercept first tool call, redirect to use_blueprint
"""
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from flyto_ai.closed_loop_v3 import evaluate_distillation
from flyto_ai.intelligence.planner import blueprint_is_trusted

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
        # Automatic reuse requires a verified trust tier and runtime evidence.
        if not blueprint_is_trusted(top, min_score=80, min_samples=2):
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
        if not blueprint_is_trusted(top, min_score=80, min_samples=2):
            return None
        return {
            "ok": False,
            "status": "guidance_required",
            "action_executed": False,
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
    *,
    min_steps: int = 3,
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

    import hashlib
    import re
    from flyto_ai.tools.blueprint_tools import _CLOSED_LOOP_EVIDENCE_CAPABILITY

    safe_blueprint_id = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,191}\Z")
    safe_execution_id = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
    receipt_fields = {
        "receipt_version", "success", "status", "evidence_id",
        "evidence_sha256", "evidence",
    }

    def safe_identity(value: Any, pattern: Any) -> bool:
        return (
            type(value) is str and pattern.fullmatch(value) is not None
            and ".." not in value and "//" not in value
        )

    verification_receipt = None
    for item in [*tool_calls, *execution_results]:
        reserved = item.get("_blueprint_feedback_verification")
        if (
            type(reserved) is dict
            and set(reserved) == {"capability", "receipt"}
            and reserved.get("capability") is _CLOSED_LOOP_EVIDENCE_CAPABILITY
            and type(reserved.get("receipt")) is dict
            and set(reserved["receipt"]) == receipt_fields
        ):
            verification_receipt = reserved["receipt"]
            break

    def accepted_identity(
        result: Any,
        blueprint_id: str,
        evidence_tier: Optional[str] = None,
    ) -> bool:
        if type(result) is not dict or result.get("ok") is not True:
            return False
        data = result.get("data")
        identity = data if type(data) is dict else result
        returned_id = identity.get("id", identity.get("blueprint_id"))
        if returned_id != blueprint_id or not safe_identity(returned_id, safe_blueprint_id):
            return False
        return not (
            evidence_tier is not None
            and identity.get("evidence_tier") != evidence_tier
        )

    # Report outcome if a blueprint was used
    used_blueprint_id = None
    blueprint_tool_call = None
    for tc in tool_calls:
        if tc.get("function") == "use_blueprint":
            used_blueprint_id = tc.get("arguments", {}).get("blueprint_id", "")
            blueprint_tool_call = tc
            break

    if used_blueprint_id:
        if not safe_identity(used_blueprint_id, safe_blueprint_id):
            logger.debug("Blueprint outcome skipped without safe identity")
            return
        if not execution_results:
            logger.debug(
                "Blueprint outcome skipped without execution evidence: %s",
                used_blueprint_id,
            )
            return
        supplied_execution_id = (blueprint_tool_call or {}).get("execution_id")
        if safe_identity(supplied_execution_id, safe_execution_id):
            execution_id = supplied_execution_id
        else:
            digest = hashlib.sha256(used_blueprint_id.encode("utf-8")).hexdigest()[:24]
            execution_id = "assistant_community_{}".format(digest)
        trusted = verification_receipt is not None
        evidence_tier = "local_verified" if trusted else "community"
        try:
            result = engine.report_outcome(
                used_blueprint_id,
                success=all_ok,
                execution_id=execution_id,
                evidence_tier=evidence_tier,
                verification=verification_receipt if trusted else None,
            )
            if accepted_identity(
                result, used_blueprint_id, evidence_tier=evidence_tier,
            ):
                logger.info(
                    "Blueprint outcome: %s %s (%s)", used_blueprint_id,
                    "OK" if all_ok else "FAIL", evidence_tier,
                )
            else:
                logger.debug("Blueprint outcome rejected or missing identity: %s", used_blueprint_id)
        except Exception as e:
            logger.debug("Blueprint report_outcome failed: %s", e)
        # Reuse should update the existing blueprint exactly once. Learning the
        # same expanded steps again would deduplicate and boost it a second time.
        return

    # Distill only verified successes into a new reusable Blueprint.
    decision = evaluate_distillation(
        tool_calls,
        execution_results,
        user_message,
        min_steps=min_steps,
    )
    if not decision.eligible or decision.workflow is None:
        logger.debug("Blueprint distillation skipped: %s", decision.reason)
        return

    steps = []
    for step in decision.workflow["steps"]:
        mid = step.get("module", "")
        if not mid:
            continue
        steps.append(step)
    if len(steps) < 3:
        return

    if verification_receipt is None:
        logger.debug("Blueprint distillation skipped without exact verification receipt")
        return

    categories = list({s["module"].split(".")[0] for s in steps if "." in s["module"]})

    try:
        result = engine.learn_from_execution(
            workflow=decision.workflow,
            name=user_message[:80],
            tags=categories,
            verification=verification_receipt,
        )
        data = result.get("data") if type(result) is dict else None
        learned_id = data.get("id") if type(data) is dict else None
        if accepted_identity(result, learned_id or ""):
            logger.info(
                "Blueprint distilled: %s as %s (%d steps, %d evidence)",
                user_message[:40], learned_id, len(steps), decision.evidence_count,
            )
        else:
            logger.debug("Blueprint learning rejected or missing identity")
    except Exception as e:
        logger.debug("Blueprint learn failed: %s", e)
