# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""YAML extraction and workflow step validation.

Two validation tiers:
1. Basic (flyto-core) — module existence + param name check
2. Deep  (flyto-pro)  — ContractEngine binding resolution + type checking
"""
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_YAML_BLOCK_RE = re.compile(r'```(?:yaml|yml)\s*\n(.*?)```', re.DOTALL)


def extract_yaml_from_response(text: str) -> Optional[str]:
    """Extract YAML content from markdown code blocks in AI response."""
    match = _YAML_BLOCK_RE.search(text)
    return match.group(1).strip() if match else None


def validate_workflow_steps(yaml_str: str) -> List[str]:
    """Validate module existence and param names for each workflow step.

    Returns a list of human-readable error strings (empty = all valid).
    Only checks that modules exist and param names match the schema.
    Does NOT validate param values (they may contain ${} variable refs).

    Requires flyto-core to be importable; returns empty list if unavailable.
    """
    import yaml as yaml_lib

    try:
        workflow = yaml_lib.safe_load(yaml_str)
    except Exception as e:
        return ["YAML parse error: {}".format(e)]

    if not isinstance(workflow, dict):
        return ["Workflow must be a YAML mapping (dict), got {}".format(type(workflow).__name__)]

    if "steps" not in workflow:
        return ["Workflow missing required 'steps' key"]

    steps = workflow.get("steps", [])
    if not isinstance(steps, list):
        return ["'steps' must be a list, got {}".format(type(steps).__name__)]

    try:
        from core.mcp_handler import get_module_info
    except ImportError:
        return []

    errors = []

    for step in steps:
        if not isinstance(step, dict):
            continue
        step_id = step.get("id", "unknown")
        module_id = step.get("module", "")
        params = step.get("params") or {}

        if not module_id:
            errors.append("Step '{}': missing module".format(step_id))
            continue

        info = get_module_info(module_id=module_id)
        if not info or info.get("error"):
            errors.append("Step '{}': module '{}' not found".format(step_id, module_id))
            continue

        schema = info.get("params_schema") or {}
        valid_params = set(schema.keys())

        if isinstance(params, dict):
            for param_name in params:
                if param_name not in valid_params:
                    errors.append(
                        "Step '{}' ({}): unknown param '{}'. "
                        "Valid params: {}".format(
                            step_id, module_id, param_name,
                            ", ".join(sorted(valid_params)),
                        )
                    )

    return errors


async def validate_workflow_deep(
    yaml_str: str,
    pro_bridge=None,
) -> Dict[str, List[str]]:
    """Deep validation using flyto-pro ContractEngine + basic validation.

    Returns {"basic": [...], "contract": [...], "missing_modules": [...]}.
    Basic errors are always checked. Contract errors are added when
    flyto-pro is available and enabled.

    Args:
        yaml_str: Raw YAML string.
        pro_bridge: ProBridge instance (optional). If None, only basic
                    validation is performed.
    """
    result = {
        "basic": validate_workflow_steps(yaml_str),
        "contract": [],
        "missing_modules": [],
    }

    if pro_bridge is None:
        return result

    # Extract missing module IDs from basic validation errors
    for err in result["basic"]:
        if "not found" in err:
            # "Step 'x': module 'y' not found" → extract 'y'
            parts = err.split("'")
            if len(parts) >= 4:
                result["missing_modules"].append(parts[3])

    # Deep contract validation
    try:
        deep_report = await pro_bridge.validate_workflow_deep(yaml_str)
        if deep_report and not deep_report.get("valid", True):
            for issue in deep_report.get("issues", []):
                msg = issue.get("message", "")
                node = issue.get("node_id", "")
                severity = issue.get("severity", "error")
                prefix = "[{}]".format(severity) if severity != "error" else ""
                if node:
                    result["contract"].append(
                        "{}Node '{}': {}".format(prefix + " " if prefix else "", node, msg)
                    )
                else:
                    result["contract"].append("{}{}".format(prefix + " " if prefix else "", msg))
    except Exception as e:
        logger.debug("Deep validation failed: %s", e)

    return result
