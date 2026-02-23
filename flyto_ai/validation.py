# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""YAML extraction and workflow step validation."""
import re
from typing import List, Optional

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

    if not isinstance(workflow, dict) or "steps" not in workflow:
        return []

    steps = workflow.get("steps", [])
    if not isinstance(steps, list):
        return []

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
