"""Core-owned parameter validation projected by the existing tool bridge."""

from typing import Any, Dict, Optional


def validate_execute_module_args(
    handler: Dict[str, Any],
    module_id: str,
    params: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Validate module params before execution when flyto-core exposes validation."""
    validate = handler.get("validate_params")
    if not validate or not module_id:
        return None
    try:
        result = validate(module_id=module_id, params=params or {})
    except Exception as e:
        return {
            "ok": False,
            "error": "flyto-core validate_params failed before execute_module: {}".format(e),
            "module_id": module_id,
            "params_valid": False,
        }

    if isinstance(result, dict):
        valid = result.get("valid")
        ok = result.get("ok")
        errors = result.get("errors") or result.get("error")
        if valid is False or ok is False:
            schema = {}
            info = handler.get("get_module_info")
            if callable(info):
                try:
                    metadata = info(module_id=module_id)
                    if isinstance(metadata, dict) and isinstance(metadata.get("params_schema"), dict):
                        schema = metadata["params_schema"]
                except Exception:
                    pass
            return {
                "ok": False,
                "error": "Invalid params for {}: {}".format(module_id, errors or "schema validation failed"),
                "module_id": module_id,
                "params_valid": False,
                "validation": result,
                "params_schema": schema,
                "suggestion": (
                    "No action was executed. Correct the call using this module's canonical "
                    "params_schema, including method selectors, active fields and defaults. "
                    "Do not assume generic selector/text arguments fit every browser module."
                ),
            }
    return None
