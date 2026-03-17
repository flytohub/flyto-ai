# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Param auto-fixer — system-level correction for LLM mistakes.

Three layers of correction:
1. **Module name fixer**: `api.post` (doesn't exist) → `http.request` + method=POST
2. **Param fixer**: missing `text` → fill from last step's result
3. **Variable resolver**: `${steps.x.result}` literal → actual value from exec history
"""
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Layer 1: Module name fixer ──────────────────────────────────

async def fix_module_id(
    dispatch: Callable,
    module_id: str,
) -> Tuple[str, dict, bool]:
    """Fix non-existent module names by searching for alternatives.

    Returns:
        (fixed_module_id, extra_params, was_fixed)
        extra_params may contain e.g. {"method": "POST"} for http.request
    """
    # Quick check: does this module exist?
    try:
        info = await dispatch("get_module_info", {"module_id": module_id})
        if isinstance(info, dict) and info.get("module_id"):
            return module_id, {}, False  # exists, no fix needed
    except Exception:
        pass

    # Module doesn't exist — try to find the right one
    extra_params: Dict[str, Any] = {}
    fixed_id = module_id

    # Common wrong names → correct names + extra params
    # Pattern: extract intent from the wrong name
    parts = module_id.lower().replace(".", " ").replace("_", " ").split()

    # HTTP method variants: api.post, api.get, http.post → http.request
    if any(p in parts for p in ("post", "put", "patch", "delete")):
        method = next((p.upper() for p in parts if p in ("post", "put", "patch", "delete")), "POST")
        fixed_id = "http.request"
        extra_params["method"] = method
        logger.info("Module fix: %s → %s (method=%s)", module_id, fixed_id, method)
        return fixed_id, extra_params, True

    if any(p in parts for p in ("get", "fetch")):
        fixed_id = "http.get"
        logger.info("Module fix: %s → %s", module_id, fixed_id)
        return fixed_id, {}, True

    # Fallback: search for similar module
    try:
        query = " ".join(parts)
        result = await dispatch("search_modules", {"query": query, "limit": 3})
        modules = result.get("modules", result.get("results", []))
        if isinstance(modules, list) and modules:
            top = modules[0]
            candidate = top.get("module_id", top.get("id", ""))
            if candidate and candidate != module_id:
                fixed_id = candidate
                logger.info("Module fix: %s → %s (via search)", module_id, fixed_id)
                return fixed_id, {}, True
    except Exception:
        pass

    return module_id, {}, False


# ── Layer 2: Param fixer ────────────────────────────────────────

# Field name aliases: LLM uses wrong names → map to correct ones
_ALIASES = {
    "text": ["input", "content", "value", "data", "string", "message", "body", "source"],
    "path": ["file", "file_path", "filepath", "output", "filename", "output_path"],
    "search": ["find", "pattern", "query", "old", "from"],
    "replace": ["replacement", "new", "to", "with"],
    "url": ["endpoint", "link", "href", "address", "uri"],
    "selector": ["css", "element", "target", "query_selector"],
    "content": ["text", "body", "data", "value", "message"],
}


async def fix_params(
    dispatch: Callable,
    module_id: str,
    params: dict,
    last_results: List[Dict[str, Any]],
) -> Tuple[dict, bool]:
    """Auto-fix params for a module call."""
    # Resolve any ${steps.x.result} variables first
    params = _resolve_variables(params, last_results)

    # Get schema
    try:
        info = await dispatch("get_module_info", {"module_id": module_id})
    except Exception:
        return params, False

    if not isinstance(info, dict):
        return params, False

    schema = info.get("params_schema", info.get("params", {}))
    if not schema:
        return params, False

    required = {
        k: v.get("type", "string")
        for k, v in schema.items()
        if v.get("required", False)
    }

    # Find missing required params
    missing = {k: t for k, t in required.items() if k not in params}
    if not missing:
        return params, False

    fixed = dict(params)
    was_fixed = False

    was_fixed |= _fix_aliases(missing, params, fixed, module_id)
    was_fixed |= _fix_from_history(missing, fixed, last_results, module_id)
    was_fixed |= _fix_cross_mapping(missing, fixed)

    return fixed, was_fixed


def _fix_aliases(
    missing: Dict[str, str],
    params: dict,
    fixed: dict,
    module_id: str,
) -> bool:
    """Strategy 1: Map aliased param names to correct ones."""
    was_fixed = False
    for field in list(missing):
        if field in _ALIASES:
            for alias in _ALIASES[field]:
                if alias in params:
                    fixed[field] = params[alias]
                    was_fixed = True
                    del missing[field]
                    logger.info("Param alias: %s.%s ← %s", module_id, field, alias)
                    break
    return was_fixed


def _fix_from_history(
    missing: Dict[str, str],
    fixed: dict,
    last_results: List[Dict[str, Any]],
    module_id: str,
) -> bool:
    """Strategy 2+4: Fill missing 'text' or 'path' from execution history."""
    was_fixed = False

    # Fill 'text' from last result
    if "text" in missing:
        last_text = _extract_text_from_results(last_results)
        if last_text:
            fixed["text"] = last_text
            was_fixed = True
            del missing["text"]
            logger.info("Param flow: %s.text ← last result", module_id)

    # Fill 'path' from last file.write result
    if "path" in missing:
        for r in reversed(last_results):
            if r.get("ok") and "path" in str(r.get("result_preview", "")):
                try:
                    data = json.loads(r["result_preview"])
                    p = data.get("path", data.get("filepath", ""))
                    if p:
                        fixed["path"] = p
                        was_fixed = True
                        break
                except (ValueError, TypeError):
                    pass

    return was_fixed


def _fix_cross_mapping(missing: Dict[str, str], fixed: dict) -> bool:
    """Strategy 3: Fill 'content' ↔ 'text' cross-mapping."""
    if "content" in missing and "text" in fixed:
        fixed["content"] = fixed["text"]
        return True
    elif "text" in missing and "content" in fixed:
        fixed["text"] = fixed["content"]
        return True
    return False


# ── Layer 3: Variable resolver ──────────────────────────────────

_VAR_PATTERN = re.compile(r'\$\{(?:steps\.)?(\w+)\.(\w+)\}')


def _resolve_variables(params: dict, exec_history: List[Dict[str, Any]]) -> dict:
    """Resolve ${steps.x.result} variables to actual values from exec history.

    LLM sometimes writes YAML variable syntax as literal strings.
    This resolver replaces them with actual values.
    """
    if not exec_history:
        return params

    resolved = {}
    any_resolved = False

    for key, value in params.items():
        if isinstance(value, str) and "${" in value:
            new_value = value
            for match in _VAR_PATTERN.finditer(value):
                step_ref = match.group(1)  # step name
                field_ref = match.group(2)  # field name (usually 'result')
                actual = _find_step_value(step_ref, field_ref, exec_history)
                if actual is not None:
                    if value == match.group(0):
                        # Entire value is a variable — replace completely (preserve type)
                        new_value = actual
                    else:
                        # Variable is part of a larger string — string replace
                        new_value = new_value.replace(match.group(0), str(actual))
                    any_resolved = True
                    logger.info("Var resolved: %s → %s", match.group(0), str(actual)[:50])
            resolved[key] = new_value
        else:
            resolved[key] = value

    return resolved if any_resolved else params


def _find_step_value(step_ref: str, field_ref: str, exec_history: List[Dict[str, Any]]) -> Any:
    """Find a value from execution history by step reference."""
    for r in reversed(exec_history):
        if not r.get("ok"):
            continue
        # Match by module_id suffix or step index
        mid = r.get("module_id", "")
        if step_ref in mid or step_ref.replace("_", ".") in mid:
            preview = r.get("result_preview", "")
            try:
                data = json.loads(preview) if isinstance(preview, str) else preview
                if isinstance(data, dict):
                    if field_ref in data:
                        return data[field_ref]
                    # 'result' often means the main output
                    if field_ref == "result":
                        for k in ("result", "text", "output", "content", "data", "value"):
                            if k in data and data[k]:
                                return data[k]
            except (ValueError, TypeError):
                if field_ref == "result" and isinstance(preview, str):
                    return preview
    return None


# ── Helper ──────────────────────────────────────────────────────

def _extract_text_from_results(results: List[Dict[str, Any]]) -> Optional[str]:
    """Extract text from the most recent successful execution result."""
    for r in reversed(results):
        if not r.get("ok"):
            continue
        preview = r.get("result_preview", "")
        if not preview:
            continue
        try:
            data = json.loads(preview) if isinstance(preview, str) else preview
            if isinstance(data, dict):
                for key in ("result", "text", "output", "content", "data", "value", "hash", "encoded"):
                    val = data.get(key)
                    if isinstance(val, str) and val:
                        return val
                if data.get("status") == "success":
                    for v in data.values():
                        if isinstance(v, str) and len(v) > 1 and v != "success":
                            return v
            elif isinstance(data, str):
                return data
        except (ValueError, TypeError):
            if isinstance(preview, str) and len(preview) > 1:
                return preview
    return None
