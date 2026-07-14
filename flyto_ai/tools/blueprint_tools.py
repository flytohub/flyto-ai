# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Blueprint tool dispatch — bridges to flyto-blueprint."""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def get_blueprint_tool_defs() -> List[Dict]:
    """Return blueprint MCP tool definitions (empty list if not installed)."""
    try:
        from flyto_blueprint.tools import get_blueprint_tools
        return get_blueprint_tools()
    except ImportError:
        return []


async def dispatch_blueprint_tool(
    name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Dispatch a blueprint tool call to flyto-blueprint engine."""
    try:
        from flyto_blueprint import get_engine
    except ImportError:
        return {"ok": False, "error": "flyto-blueprint not installed. Run: pip install flyto-blueprint"}

    engine = get_engine()

    if name == "list_blueprints":
        query = arguments.get("query", "")
        if query:
            return {"ok": True, "blueprints": engine.search(query)}
        return {"ok": True, "blueprints": engine.list_blueprints()}

    elif name == "use_blueprint":
        raw = engine.expand(
            blueprint_id=arguments.get("blueprint_id", ""),
            args=arguments.get("args", {}),
        )
        if not raw.get("ok") or not raw.get("data", {}).get("steps"):
            return raw

        steps = raw["data"]["steps"]
        # Return a compact result with the execution instruction AT THE TOP
        # so it doesn't get truncated by the 8000-char limit
        return {
            "ok": True,
            "action_required": (
                "EXECUTE each step NOW with execute_module(module_id, params). "
                "Do NOT stop. Do NOT just return the YAML."
            ),
            "steps": [
                {"module": s["module"], "params": s.get("params", {})}
                for s in steps
            ],
        }

    elif name == "save_as_blueprint":
        return engine.learn_from_workflow(
            workflow=arguments.get("workflow", {}),
            name=arguments.get("name"),
            tags=arguments.get("tags"),
        )

    elif name == "report_blueprint_outcome":
        return engine.report_outcome(
            blueprint_id=arguments.get("blueprint_id", ""),
            success=arguments.get("success", False),
        )

    return {"ok": False, "error": "Unknown blueprint tool: {}".format(name)}
