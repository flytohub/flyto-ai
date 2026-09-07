"""Resolve ad-hoc module references only against explicitly supplied JSON data."""

import re

_REFERENCE = re.compile(r"\$\{\{([^{}]+)\}\}|\$\{([^{}]+)\}|\{\{([^{}]+)\}\}")
_DROP = object()


class UnresolvedParameterReference(ValueError):
    """A module must not receive a workflow placeholder as a literal value."""


def _references(value, depth=0):
    if depth > 24:
        raise UnresolvedParameterReference("Module parameters exceed the reference depth limit")
    if isinstance(value, str):
        return [next(part for part in match.groups() if part is not None).strip()
                for match in _REFERENCE.finditer(value)]
    if type(value) is dict:
        return [ref for item in value.values() for ref in _references(item, depth + 1)]
    if type(value) is list:
        return [ref for item in value for ref in _references(item, depth + 1)]
    return []


def _json_context(value, depth=0):
    if depth > 24:
        return _DROP
    if value is None or type(value) in (str, int, float, bool):
        return value
    if type(value) is dict:
        return {key: clean for key, item in value.items()
                if type(key) is str and (clean := _json_context(item, depth + 1)) is not _DROP}
    if type(value) is list:
        return [None if (clean := _json_context(item, depth + 1)) is _DROP else clean
                for item in value]
    # Browser handles and other runtime capabilities are not variable data.
    return _DROP


def _normalize(value):
    if isinstance(value, str):
        return _REFERENCE.sub(lambda match: "${" + next(
            part for part in match.groups() if part is not None).strip() + "}", value)
    if type(value) is dict:
        return {key: _normalize(item) for key, item in value.items()}
    if type(value) is list:
        return [_normalize(item) for item in value]
    return value


def resolve_module_params(params, context):
    """Use Core's resolver; never read ambient environment credentials."""
    refs = _references(params)
    if not refs:
        return params
    clean = _json_context(context) if type(context) is dict else {}
    # Core can resolve env.* from os.environ. Ad-hoc calls have no authority to
    # discover service secrets, even when a model supplies an env-shaped map.
    if any(ref.split(".", 1)[0] == "env" or ref.split(".", 1)[0] not in clean for ref in refs):
        raise UnresolvedParameterReference("Module parameters contain unbound workflow references")
    try:
        from core.engine.variable_resolver import VariableResolver
    except ImportError as error:
        raise UnresolvedParameterReference("Core parameter resolution is unavailable") from error
    resolved = VariableResolver(clean.get("params", {}), clean).resolve(_normalize(params))
    if _references(resolved):
        raise UnresolvedParameterReference("Module parameters still contain unresolved references")
    return resolved
