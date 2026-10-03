# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Per-package capability groups and tools for non-core module packs.

flyto-core's own modules stay behind the meta-tools (search, describe,
execute): the model finds them by searching. A package installed beside Core
(anything registered through the ``flyto.modules`` entry point, so its
modules carry a non-empty ``plugin``) is different: a host that offers a
resource backed by that package wants the WHOLE package, grouped, so the model
can say "resource X can: ..." without a lexical shortlist dropping half of it.

This module turns Core's capability manifest plus per-module catalog detail
into those groups, and the groups into tool definitions:

* :func:`get_pack_capability_groups` -- ``{plugin, description, modules: [...]}``
  per package, every module carrying its own ``contract`` and
  ``params_schema`` from ``get_module_info`` (never the manifest's
  per-capability ``contracts`` map, which keeps only the lowest module id when
  two packages provide the same capability).
* :func:`build_pack_tools` -- one tool per (package, capability), named
  ``<plugin>__<capability with dots as _>``, collision-checked, with an index
  that maps every name back to its exact binding.
* :func:`resolve_pack_tool_call` -- the reverse direction for a dispatcher:
  name + arguments to ``{plugin, module_id, capability, resource_id, ...}``.
* :func:`bind_resource_capabilities` -- join one resource's declared
  capability ids (optionally scoped to the package that reported them) with
  the tools, reporting ambiguity instead of guessing.

Nothing here executes anything. Every tool is graded by its contract
(:func:`flyto_ai.permissions.grade_module_contract`); a host turns a call into
its own proposal / approval / safe-stop path, except capabilities graded
``immediate`` (a stop), which keep the host's immediate path.

No flyto-core import happens at module import time, so a host that only holds
reported data can still use the pure helpers.
"""
from __future__ import annotations

import hashlib
import logging
import re
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from flyto_ai.contract_recovery import declared_recovery
from flyto_ai.permissions import PermissionLevel, grade_module_contract

logger = logging.getLogger(__name__)

PACK_TOOL_SEPARATOR = "__"
PACK_TOOL_NAME_MAX = 64
PACK_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PACK_TOOL_SOURCE = "module_pack"

_UNSAFE_SEGMENT_CHARS = re.compile(r"[^A-Za-z0-9-]+")
_JSON_TYPES = {
    "string": "string",
    "number": "number",
    "integer": "integer",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
}

ManifestReader = Callable[[], Optional[Mapping[str, Any]]]
ModuleInfoReader = Callable[[str], Optional[Mapping[str, Any]]]
PluginModulesReader = Callable[[str], Optional[List[str]]]


# ── Tool naming ───────────────────────────────────────────────────────


def _segment(text: str) -> str:
    """Reduce ``text`` to ``[A-Za-z0-9-]`` runs joined by single underscores.

    A single underscore never doubles, so the ``__`` separator stays unique in
    a generated name.
    """
    return _UNSAFE_SEGMENT_CHARS.sub("_", text).strip("_")


def pack_tool_name(plugin: str, capability: str) -> str:
    """Deterministic tool name for ``capability`` provided by ``plugin``.

    ``<plugin>__<capability with dots as _>`` within ``^[A-Za-z0-9_-]{1,64}$``.
    A name that would exceed 64 characters is truncated and suffixed with a
    digest of the full identity, so it stays unique. The mapping is not
    invertible by string manipulation alone (``a.b`` and ``a_b`` both become
    ``a_b``); :func:`build_pack_tools` therefore detects collisions and keeps
    an explicit name -> binding index.
    """
    plugin_part = _segment(str(plugin or "")) or "pack"
    capability_part = _segment(str(capability or "")) or "capability"
    name = "{}{}{}".format(plugin_part, PACK_TOOL_SEPARATOR, capability_part)
    if len(name) <= PACK_TOOL_NAME_MAX:
        return name
    digest = hashlib.sha256(
        "{}\0{}".format(plugin, capability).encode("utf-8"),
    ).hexdigest()[:10]
    keep = PACK_TOOL_NAME_MAX - len(digest) - 1
    return "{}-{}".format(name[:keep].rstrip("_-"), digest)


# ── Group discovery ───────────────────────────────────────────────────


def _default_manifest_reader() -> Optional[Mapping[str, Any]]:
    from flyto_ai.tools.core_tools import _get_core_capability_manifest_fn

    read = _get_core_capability_manifest_fn()
    return read() if read is not None else None


def _default_module_info(module_id: str) -> Optional[Mapping[str, Any]]:
    from flyto_ai.permissions import core_module_info

    return core_module_info(module_id)


def _default_plugin_modules(plugin: str) -> Optional[List[str]]:
    try:
        from core.modules.registry import ModuleRegistry
    except ImportError:
        return None
    getter = getattr(ModuleRegistry, "get_plugin_modules", None)
    if not callable(getter):
        return None
    return list(getter(plugin))


def _plugin_module_ids(
    entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    plugin_modules: Optional[PluginModulesReader],
) -> List[str]:
    """Module ids a package contributed, from the best source available.

    1. ``plugins[].module_ids`` when Core reports it;
    2. the registry's ``get_plugin_modules`` (ownership Core assigned);
    3. every capability provider (ownership is re-checked per module below).
    """
    declared = entry.get("module_ids")
    if isinstance(declared, list):
        return [item for item in declared if isinstance(item, str) and item]
    if plugin_modules is not None:
        try:
            owned = plugin_modules(str(entry.get("id") or ""))
        except Exception as exc:
            logger.warning("get_plugin_modules failed: %s", exc)
            owned = None
        if owned is not None:
            return [item for item in owned if isinstance(item, str) and item]
    providers: List[str] = []
    for capability in manifest.get("capabilities") or []:
        if isinstance(capability, Mapping):
            providers.extend(
                item for item in capability.get("providers") or []
                if isinstance(item, str) and item
            )
    return providers


def _module_entry(module_id: str, info: Mapping[str, Any]) -> Dict[str, Any]:
    grade = grade_module_contract(info)
    capability = info.get("provides_capability")
    params_schema = info.get("params_schema")
    contract = info.get("contract")
    recovery = declared_recovery(contract)
    return {
        "module_id": module_id,
        "provides_capability": capability if isinstance(capability, str) else "",
        "params_schema": deepcopy(params_schema) if isinstance(params_schema, Mapping) else {},
        "contract": deepcopy(contract) if isinstance(contract, Mapping) else None,
        "label": str(info.get("label") or ""),
        "description": str(info.get("description") or ""),
        # A pack module always grades (missing contract fails closed).
        "grade": grade.to_dict() if grade is not None else None,
        # Declared recovery (contract ``recovery`` key, core 2.36+) or None.
        "recovery": recovery.to_dict() if recovery is not None else None,
    }


def get_pack_capability_groups(
    *,
    manifest: Optional[Mapping[str, Any]] = None,
    manifest_reader: Optional[ManifestReader] = None,
    module_info: Optional[ModuleInfoReader] = None,
    plugin_modules: Optional[PluginModulesReader] = None,
) -> Dict[str, Any]:
    """Group every non-core package's modules, each with its own contract.

    Returns::

        {
          "ok": bool,
          "source": "flyto-core",
          "groups": [
            {
              "plugin": str,           # entry-point name Core assigned
              "version": str,
              "description": str,      # plugins[].description, or ""
              "modules": [
                {
                  "module_id": str,
                  "provides_capability": str,   # "" when undeclared
                  "params_schema": dict,
                  "contract": dict | None,
                  "label": str,
                  "description": str,
                  "grade": ContractGrade.to_dict(),
                  "recovery": {"substitutes": [...], "context": str} | None,
                },
              ],
            },
          ],
          "error": str,    # only when ok is False
        }

    Groups are sorted by plugin id, modules by module id. Core's own modules
    (empty ``plugin``) never appear. A module whose catalog detail names a
    different owner than the group is dropped -- ownership comes from Core's
    registry, never from the package. An old Core with no ``plugins`` list
    yields ``ok: True`` and no groups (the host keeps search only).
    """
    if manifest is None:
        try:
            manifest = (manifest_reader or _default_manifest_reader)()
        except Exception as exc:
            logger.warning("flyto-core capability manifest failed: %s", exc)
            return {"ok": False, "source": "flyto-core", "groups": [],
                    "error": "capability manifest unavailable"}
    if not isinstance(manifest, Mapping):
        return {"ok": False, "source": "flyto-core", "groups": [],
                "error": "capability manifest unavailable"}

    info_reader = module_info or _default_module_info
    owned_reader = plugin_modules if plugin_modules is not None else (
        _default_plugin_modules if module_info is None else None
    )

    groups: List[Dict[str, Any]] = []
    for entry in manifest.get("plugins") or []:
        if not isinstance(entry, Mapping):
            continue
        plugin = entry.get("id")
        if not isinstance(plugin, str) or not plugin.strip():
            continue
        modules: List[Dict[str, Any]] = []
        seen = set()
        for module_id in sorted(set(_plugin_module_ids(entry, manifest, owned_reader))):
            if module_id in seen:
                continue
            seen.add(module_id)
            try:
                info = info_reader(module_id)
            except Exception as exc:
                logger.warning("get_module_info(%s) failed: %s", module_id, exc)
                continue
            if not isinstance(info, Mapping) or info.get("error"):
                continue
            if info.get("plugin") != plugin:
                continue
            modules.append(_module_entry(module_id, info))
        groups.append({
            "plugin": plugin,
            "version": str(entry.get("version") or ""),
            "description": str(entry.get("description") or ""),
            "modules": modules,
        })
    groups.sort(key=lambda group: group["plugin"])
    return {"ok": True, "source": "flyto-core", "groups": groups}


# ── Tools ─────────────────────────────────────────────────────────────


def _json_property(spec: Any) -> Dict[str, Any]:
    if not isinstance(spec, Mapping):
        return {}
    prop: Dict[str, Any] = {}
    json_type = _JSON_TYPES.get(str(spec.get("type") or ""))
    if json_type:
        prop["type"] = json_type
    text = spec.get("description") or spec.get("label")
    if isinstance(text, str) and text:
        prop["description"] = text
    for source, target in (("min", "minimum"), ("max", "maximum"),
                           ("minimum", "minimum"), ("maximum", "maximum")):
        value = spec.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            prop[target] = value
    options = spec.get("enum", spec.get("options"))
    if isinstance(options, list) and options and all(
        isinstance(item, (str, int, float)) and not isinstance(item, bool)
        for item in options
    ):
        prop["enum"] = list(options)
    return prop


def params_schema_to_json_schema(params_schema: Any) -> Dict[str, Any]:
    """Project a Core ``params_schema`` onto a JSON Schema object.

    Bounds are shown to the model as guidance only. The host validates the
    real call against ``params_schema`` and refuses out-of-bounds values; it
    never clamps them.
    """
    properties: Dict[str, Any] = {}
    required: List[str] = []
    if isinstance(params_schema, Mapping):
        for name, spec in params_schema.items():
            if not isinstance(name, str) or not name:
                continue
            properties[name] = _json_property(spec)
            if isinstance(spec, Mapping) and spec.get("required") is True:
                required.append(name)
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = sorted(required)
    return schema


def _tool_description(group: Mapping[str, Any], module: Mapping[str, Any]) -> str:
    grade = module.get("grade") or {}
    text = module.get("description") or module.get("label") or module["provides_capability"]
    notes = ["capability {}".format(module["provides_capability"]),
             "package {}".format(group["plugin"])]
    if grade.get("immediate"):
        notes.append("stop: runs immediately")
    elif grade.get("actuating"):
        notes.append("acts on the real world: requires confirmation")
    recovery = module.get("recovery") or {}
    if recovery.get("substitutes"):
        notes.append("on failure may be replaced only by: {}".format(
            ", ".join(recovery["substitutes"])))
    return "{} ({}).".format(str(text).rstrip("."), "; ".join(notes))


def build_pack_tools(groups: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """One tool per (package, capability), collision-checked.

    ``groups`` is the ``groups`` list from :func:`get_pack_capability_groups`.
    Returns::

        {
          "tools": [ {name, description, inputSchema, _meta} ],
          "index": { tool_name: binding },
          "collisions": [ {name, bindings: [binding, ...]} ],
        }

    where ``binding`` is ``{plugin, module_id, capability, permission_level,
    risk_level, actuating, immediate, params_schema, contract}``. Every tool
    takes ``{"resource_id": str, "arguments": {...}}``. ``_meta`` carries
    ``flyto2/source`` (``{"type": "module_pack", plugin, module_id,
    capability}``), ``flyto2/contractGrade`` and ``flyto2/immediate``.

    A name produced by two different bindings -- two modules of one package
    providing the same capability, or two package ids that sanitize alike --
    is withheld entirely and reported under ``collisions``: guessing which one
    the model meant is how the wrong device moves. Modules that declare no
    capability get no tool.
    """
    candidates: Dict[str, List[Dict[str, Any]]] = {}
    group_by_binding: Dict[int, Mapping[str, Any]] = {}
    module_by_binding: Dict[int, Mapping[str, Any]] = {}
    for group in groups or []:
        plugin = group.get("plugin") if isinstance(group, Mapping) else None
        if not isinstance(plugin, str) or not plugin:
            continue
        for module in group.get("modules") or []:
            capability = module.get("provides_capability") if isinstance(module, Mapping) else ""
            if not isinstance(capability, str) or not capability:
                continue
            grade = module.get("grade") or grade_module_contract(
                {"plugin": plugin, "provides_capability": capability,
                 "contract": module.get("contract")},
            ).to_dict()
            binding = {
                "plugin": plugin,
                "module_id": module["module_id"],
                "capability": capability,
                "permission_level": grade["permission_level"],
                "risk_level": grade["risk_level"],
                "actuating": grade["actuating"],
                "immediate": grade["immediate"],
                "params_schema": deepcopy(module.get("params_schema") or {}),
                "contract": deepcopy(module.get("contract")),
            }
            name = pack_tool_name(plugin, capability)
            candidates.setdefault(name, []).append(binding)
            group_by_binding[id(binding)] = group
            module_by_binding[id(binding)] = module

    tools: List[Dict[str, Any]] = []
    index: Dict[str, Dict[str, Any]] = {}
    collisions: List[Dict[str, Any]] = []
    for name in sorted(candidates):
        bindings = candidates[name]
        if len(bindings) > 1 or not PACK_TOOL_NAME_PATTERN.fullmatch(name):
            collisions.append({"name": name, "bindings": bindings})
            continue
        binding = bindings[0]
        group = group_by_binding[id(binding)]
        module = module_by_binding[id(binding)]
        index[name] = binding
        tools.append({
            "name": name,
            "description": _tool_description(group, module),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "resource_id": {
                        "type": "string",
                        "description": "The resource that should perform this capability.",
                    },
                    "arguments": params_schema_to_json_schema(binding["params_schema"]),
                },
                "required": ["resource_id"],
                "additionalProperties": False,
            },
            "_meta": {
                "flyto2/source": {
                    "type": PACK_TOOL_SOURCE,
                    "plugin": binding["plugin"],
                    "module_id": binding["module_id"],
                    "capability": binding["capability"],
                },
                "flyto2/contractGrade": {
                    "permission_level": binding["permission_level"],
                    "risk_level": binding["risk_level"],
                    "actuating": binding["actuating"],
                    "immediate": binding["immediate"],
                },
                "flyto2/immediate": binding["immediate"],
            },
        })
    return {"tools": tools, "index": index, "collisions": collisions}


def pack_tool_permission_overrides(
    index: Mapping[str, Mapping[str, Any]],
) -> Dict[str, PermissionLevel]:
    """``{tool_name: PermissionLevel}`` for a ``ToolExecutor``.

    Hosts must use these levels for pack tools rather than grading the names
    themselves: a name such as ``pack__get_status`` reads like a lookup, while
    its contract may say it actuates.
    """
    return {
        name: PermissionLevel[str(binding["permission_level"])]
        for name, binding in index.items()
    }


def resolve_pack_tool_call(
    name: str,
    arguments: Optional[Mapping[str, Any]],
    index: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Turn a pack tool call back into the exact capability request.

    Returns ``{"ok": True, "plugin", "module_id", "capability", "resource_id",
    "arguments", "permission_level", "risk_level", "actuating", "immediate"}``
    or ``{"ok": False, "error": str}``. Arguments are passed through as given:
    the host validates them against the module's ``params_schema`` and refuses
    out-of-bounds values rather than clamping. Nothing is executed here.
    """
    binding = index.get(name) if isinstance(name, str) else None
    if binding is None:
        return {"ok": False, "error": "unknown pack tool: {}".format(name)}
    arguments = arguments if isinstance(arguments, Mapping) else {}
    resource_id = arguments.get("resource_id")
    if not isinstance(resource_id, str) or not resource_id.strip():
        return {"ok": False, "error": "resource_id is required"}
    call_arguments = arguments.get("arguments", {})
    if call_arguments is None:
        call_arguments = {}
    if not isinstance(call_arguments, Mapping):
        return {"ok": False, "error": "arguments must be an object"}
    extra = sorted(set(arguments) - {"resource_id", "arguments"})
    if extra:
        return {"ok": False, "error": "unexpected fields: {}".format(", ".join(extra))}
    return {
        "ok": True,
        "plugin": binding["plugin"],
        "module_id": binding["module_id"],
        "capability": binding["capability"],
        "resource_id": resource_id.strip(),
        "arguments": deepcopy(dict(call_arguments)),
        "permission_level": binding["permission_level"],
        "risk_level": binding["risk_level"],
        "actuating": binding["actuating"],
        "immediate": binding["immediate"],
    }


def bind_resource_capabilities(
    index: Mapping[str, Mapping[str, Any]],
    capability_ids: Iterable[str],
    *,
    pack: Optional[str] = None,
) -> Dict[str, Any]:
    """Which pack tools serve one resource's declared capabilities.

    ``pack`` is the package (entry-point name) that reported the resource;
    when given, only its tools bind. Without it, a capability offered by more
    than one package is reported as ambiguous and bound to none.

    Returns ``{"bound": [{capability, tool_name, plugin, module_id}],
    "ambiguous": [{capability, plugins: [..]}], "missing": [capability]}``,
    each list in capability order.
    """
    by_capability: Dict[str, List[tuple]] = {}
    for name, binding in index.items():
        if pack is not None and binding["plugin"] != pack:
            continue
        by_capability.setdefault(binding["capability"], []).append((name, binding))

    bound: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []
    missing: List[str] = []
    for capability in sorted({c for c in capability_ids or [] if isinstance(c, str) and c}):
        matches = by_capability.get(capability) or []
        if not matches:
            missing.append(capability)
        elif len(matches) > 1:
            ambiguous.append({
                "capability": capability,
                "plugins": sorted(binding["plugin"] for _, binding in matches),
            })
        else:
            name, binding = matches[0]
            bound.append({
                "capability": capability,
                "tool_name": name,
                "plugin": binding["plugin"],
                "module_id": binding["module_id"],
            })
    return {"bound": bound, "ambiguous": ambiguous, "missing": missing}


def get_pack_tool_catalog(**kwargs: Any) -> Dict[str, Any]:
    """Groups and tools in one call, for a host building a turn.

    Accepts the keyword arguments of :func:`get_pack_capability_groups` and
    returns its result extended with ``tools``, ``index``, ``collisions`` (from
    :func:`build_pack_tools`) and ``permission_overrides`` (from
    :func:`pack_tool_permission_overrides`). With no packages installed, or an
    unreadable manifest, every list is empty and the host keeps search only.
    """
    result = get_pack_capability_groups(**kwargs)
    built = build_pack_tools(result.get("groups") or [])
    result.update(built)
    result["permission_overrides"] = pack_tool_permission_overrides(built["index"])
    return result


__all__ = [
    "get_pack_tool_catalog",
    "PACK_TOOL_NAME_PATTERN",
    "PACK_TOOL_SEPARATOR",
    "PACK_TOOL_SOURCE",
    "bind_resource_capabilities",
    "build_pack_tools",
    "get_pack_capability_groups",
    "pack_tool_name",
    "pack_tool_permission_overrides",
    "params_schema_to_json_schema",
    "resolve_pack_tool_call",
]
