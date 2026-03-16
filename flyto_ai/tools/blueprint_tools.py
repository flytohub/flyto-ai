# Copyright 2024 Flyto
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
        result = engine.expand(
            blueprint_id=arguments.get("blueprint_id", ""),
            args=arguments.get("args", {}),
        )
        # Tell LLM to execute each step — don't just return the YAML
        if result.get("ok") and result.get("data", {}).get("steps"):
            steps = result["data"]["steps"]
            result["_next_action"] = (
                "IMPORTANT: Now you MUST execute each step below with execute_module(module_id, params). "
                "Do NOT stop here. Steps: {}".format(
                    ", ".join("{} → {}".format(s["id"], s["module"]) for s in steps)
                )
            )
        return result

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
