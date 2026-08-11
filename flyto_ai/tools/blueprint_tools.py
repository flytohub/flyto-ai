# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Blueprint tool dispatch — bridges to flyto-blueprint."""
import inspect
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

from .core_tools import get_core_installed_module_ids

logger = logging.getLogger(__name__)


class _EvidenceCapability(str):
    """In-process capability that cannot survive a model JSON round trip."""


_CLOSED_LOOP_EVIDENCE_CAPABILITY = _EvidenceCapability(
    "flyto-ai.closed-loop-verified",
)

# The single keyword the host uses to hand installed Core module ids to the
# Blueprint engine. It is host-derived on every call and is never read from,
# nor exposed to, model arguments. Only module ids travel under it: a Blueprint
# step names the module that runs it, so a capability id or a plugin id here
# would let the engine offer a step nothing installed can execute.
MODULE_IDS_KWARG = "available_module_ids"

# Model-facing aliases that must never become an availability claim. The
# host-derived keyword is included: a model that supplies it is overwritten, not
# merged with, and it is stripped from every published schema.
_MODEL_AVAILABILITY_KEYS = (
    "available_module_ids",
    "available_capabilities",
    "installed_capabilities",
    "installed_module_ids",
    "module_ids",
    "capabilities",
)


def _without_availability_argument(tool_def: Any) -> Any:
    """Keep host-owned availability filtering out of the model-facing schema."""
    if not isinstance(tool_def, dict):
        return tool_def
    schema = tool_def.get("inputSchema")
    if not isinstance(schema, dict):
        return tool_def
    properties = schema.get("properties")
    required = schema.get("required")
    in_properties = isinstance(properties, dict) and any(
        key in properties for key in _MODEL_AVAILABILITY_KEYS
    )
    in_required = isinstance(required, list) and any(
        key in required for key in _MODEL_AVAILABILITY_KEYS
    )
    if not in_properties and not in_required:
        return tool_def

    stripped = deepcopy(tool_def)
    stripped_schema = stripped["inputSchema"]
    if isinstance(stripped_schema.get("properties"), dict):
        for key in _MODEL_AVAILABILITY_KEYS:
            stripped_schema["properties"].pop(key, None)
    if isinstance(stripped_schema.get("required"), list):
        stripped_schema["required"] = [
            key
            for key in stripped_schema["required"]
            if key not in _MODEL_AVAILABILITY_KEYS
        ]
    logger.info(
        "stripped model-facing availability argument from blueprint tool %s",
        tool_def.get("name", ""),
    )
    return stripped


def _host_installed_module_ids() -> Optional[frozenset]:
    """Resolve installed Core module ids on the host, never from arguments.

    None means the installed Core cannot report a manifest, so Blueprint is
    left on its legacy unfiltered behaviour. Any other outcome is a frozenset,
    so a broken Core narrows rather than widens what Blueprint offers.
    """
    try:
        return get_core_installed_module_ids()
    except Exception as e:
        logger.warning("installed module discovery failed: %s", e)
        return frozenset()


def _accepts_module_ids(fn: Any) -> bool:
    """Check whether a Blueprint engine call takes the module-id keyword.

    A legacy engine predating the keyword is called exactly as before, so
    installing a newer host never breaks an older Blueprint.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    declared = parameters.get(MODULE_IDS_KWARG)
    if declared is not None:
        return declared.kind is not inspect.Parameter.POSITIONAL_ONLY
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _module_id_kwargs(fn: Any, module_ids: Optional[frozenset]) -> Dict[str, Any]:
    """Build the host-derived module-id keyword for one engine call."""
    if module_ids is None or not _accepts_module_ids(fn):
        return {}
    return {MODULE_IDS_KWARG: module_ids}


def _warn_on_model_availability_arguments(arguments: Dict[str, Any]) -> None:
    """Model-supplied availability claims are ignored; record that they were."""
    for key in _MODEL_AVAILABILITY_KEYS:
        if key in arguments:
            logger.warning(
                "ignoring model-supplied %s on a blueprint tool call; "
                "installed module ids are host-derived",
                key,
            )


def get_blueprint_tool_defs() -> List[Dict]:
    """Return blueprint MCP tool definitions (empty list if not installed)."""
    try:
        from flyto_blueprint.tools import get_blueprint_tools
        tools = get_blueprint_tools()
    except ImportError:
        return []
    if not isinstance(tools, list):
        return tools
    return [_without_availability_argument(tool) for tool in tools]


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
    _warn_on_model_availability_arguments(arguments)

    if name == "list_blueprints":
        query = arguments.get("query", "")
        module_ids = _host_installed_module_ids()
        if query:
            return {
                "ok": True,
                "blueprints": engine.search(
                    query,
                    **_module_id_kwargs(engine.search, module_ids),
                ),
            }
        return {
            "ok": True,
            "blueprints": engine.list_blueprints(
                **_module_id_kwargs(engine.list_blueprints, module_ids),
            ),
        }

    elif name == "use_blueprint":
        module_ids = _host_installed_module_ids()
        raw = engine.expand(
            blueprint_id=arguments.get("blueprint_id", ""),
            args=arguments.get("args", {}),
            **_module_id_kwargs(engine.expand, module_ids),
        )
        if not raw.get("ok") or not raw.get("data", {}).get("steps"):
            return raw

        steps = raw["data"]["steps"]
        # Return a compact result with the execution instruction AT THE TOP
        # so it doesn't get truncated by the 8000-char limit
        execution_steps = []
        for step in steps:
            execution_step = {
                "module": step["module"],
                "params": step.get("params", {}),
            }
            for field in ("id", "retry", "assert", "assertions"):
                if field in step:
                    execution_step[field] = step[field]
            execution_steps.append(execution_step)

        return {
            "ok": True,
            "blueprint_id": arguments.get("blueprint_id", ""),
            "action_required": (
                "EXECUTE each step NOW with execute_module(module_id, params). "
                "Do NOT stop. Do NOT just return the YAML."
            ),
            "steps": execution_steps,
        }

    elif name == "save_as_blueprint":
        return engine.learn_from_workflow(
            workflow=arguments.get("workflow", {}),
            name=arguments.get("name"),
            tags=arguments.get("tags"),
        )

    elif name == "report_blueprint_outcome":
        host_verified = (
            arguments.get("_evidence_capability")
            is _CLOSED_LOOP_EVIDENCE_CAPABILITY
        )
        report_args = {
            "blueprint_id": arguments.get("blueprint_id", ""),
            "success": arguments.get("success", False),
            "execution_id": arguments.get("execution_id", ""),
            "evidence_tier": "local_verified" if host_verified else "community",
        }
        execution_evidence = arguments.get("_execution_evidence")
        if host_verified and isinstance(execution_evidence, dict):
            report_args["evidence"] = execution_evidence
        return engine.report_outcome(
            **report_args,
        )

    elif name == "export_blueprint":
        return engine.export_blueprint(
            blueprint_id=arguments.get("blueprint_id", ""),
            publisher=arguments.get("publisher", ""),
        )

    elif name == "import_blueprint":
        return engine.import_blueprint(
            bundle=arguments.get("bundle", {}),
        )

    return {"ok": False, "error": "Unknown blueprint tool: {}".format(name)}
