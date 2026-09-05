# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""flyto-core MCP tool bridge — lazily imports core handler."""
import asyncio
import dataclasses
import hashlib
import hmac
import importlib
import importlib.metadata
import logging
import os
import re
import threading
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CORE_MCP_CONTRACT_VERSION = "flyto-core-mcp.v1"

# Shared browser session store across tool calls within a chat session.
# Keys are browser session IDs from browser.launch results.
# Cleared via clear_browser_sessions() between independent chat sessions.
_browser_sessions: Dict[str, Any] = {}
_browser_sessions_lock = threading.Lock()

# Browser cascade breaker: when browser.launch fails, skip subsequent browser.* calls.
# Reset on each new browser.launch attempt.
_browser_launch_failed: bool = False
_browser_launch_error: str = ""

# Goto circuit breaker: after N consecutive goto failures, return a non-retryable error.
_goto_consecutive_fails: int = 0
_GOTO_MAX_FAILS: int = 3

_READ_ONLY_CORE_TOOLS = frozenset({
    "list_modules",
    "search_modules",
    "get_module_info",
    "get_module_examples",
    "validate_params",
    "list_recipes",
    "get_core_capability_manifest",
})

_EXECUTION_CORE_TOOLS = frozenset({"execute_module", "run_recipe"})

_DANGER_MODULE_CATEGORIES = frozenset({
    "shell", "process", "docker", "k8s", "ssh", "network", "port", "dns",
    "file", "path", "env", "git",
})

CORE_CAPABILITY_MANIFEST_TOOL = {
    "name": "get_core_capability_manifest",
    "description": (
        "Return the flyto-core MCP capability manifest for agents and cloud UIs: "
        "contract version, installed core version, tool fingerprint, recipes support, "
        "module categories, and risk/approval metadata for callable tools."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "include_tools": {
                "type": "boolean",
                "description": "Include per-tool metadata in the response.",
                "default": True,
            },
            "include_categories": {
                "type": "boolean",
                "description": "Include flyto-core module category counts when available.",
                "default": True,
            },
        },
    },
}


def _active_browser_sessions() -> Dict[str, Any]:
    from flyto_ai.tools.browser_scope import current_browser_scope
    scope = current_browser_scope()
    return scope.sessions if scope is not None else _browser_sessions


def _browser_retry_state():
    from flyto_ai.tools.browser_scope import current_browser_scope
    scope = current_browser_scope()
    if scope is not None:
        return scope.launch_failed, scope.launch_error, scope.goto_failures
    return _browser_launch_failed, _browser_launch_error, _goto_consecutive_fails


def _set_browser_retry_state(launch_failed, launch_error, goto_failures):
    from flyto_ai.tools.browser_scope import current_browser_scope
    global _browser_launch_failed, _browser_launch_error, _goto_consecutive_fails
    scope = current_browser_scope()
    if scope is not None:
        scope.launch_failed, scope.launch_error, scope.goto_failures = launch_failed, launch_error, goto_failures
    else:
        _browser_launch_failed, _browser_launch_error, _goto_consecutive_fails = launch_failed, launch_error, goto_failures


def clear_browser_sessions() -> None:
    """Clear the shared browser session store (call between independent chats)."""
    with _browser_sessions_lock:
        _active_browser_sessions().clear()
    _set_browser_retry_state(False, "", 0)


def get_browser_status() -> str:
    """Get a prompt hint about browser state for the LLM.

    Returns empty string if no browser running, or an instruction
    telling the LLM to reuse the existing browser.
    """
    with _browser_sessions_lock:
        if not _active_browser_sessions():
            return ""
        return (
            "BROWSER IS ALREADY RUNNING. Do NOT call browser.launch again. "
            "Continue using browser.goto / browser.snapshot / browser.click directly."
        )


def _is_ok(result: Dict[str, Any]) -> bool:
    """Check if a module result indicates success.

    flyto-core modules return {"status": "success"} without an "ok" field.
    Normalize: ok=true OR status=="success" → success.
    """
    if result.get("ok"):
        return True
    return result.get("status") == "success"


_cached_handler = None
_handler_checked = False


def _get_mcp_handler():
    """Lazily import mcp_handler to avoid circular imports.

    Returns None if flyto-core is not installed. Result is cached.
    """
    global _cached_handler, _handler_checked
    if _handler_checked:
        return _cached_handler
    _handler_checked = True
    try:
        from core.mcp_handler import (
            TOOLS,
            list_modules,
            search_modules,
            get_module_info,
            get_module_examples,
            execute_module,
            validate_params,
        )
        _cached_handler = {
            "TOOLS": TOOLS,
            "list_modules": list_modules,
            "search_modules": search_modules,
            "get_module_info": get_module_info,
            "get_module_examples": get_module_examples,
            "execute_module": execute_module,
            "validate_params": validate_params,
        }
        # Recipe support is optional (flyto-core >= 2.15.0)
        try:
            from core.mcp_handler import list_recipes, run_recipe
            _cached_handler["list_recipes"] = list_recipes
            _cached_handler["run_recipe"] = run_recipe
        except ImportError:
            pass
    except ImportError:
        _cached_handler = None
    return _cached_handler


def _core_package_version() -> str:
    """Return installed flyto-core package version when available."""
    try:
        return importlib.metadata.version("flyto-core")
    except importlib.metadata.PackageNotFoundError:
        return ""
    except Exception:
        return ""


def _risk_for_tool(name: str) -> str:
    if name in _READ_ONLY_CORE_TOOLS:
        return "read_only"
    if name in _EXECUTION_CORE_TOOLS:
        return "workspace_write"
    return "workspace_write"


def _approval_policy_for_tool(name: str) -> str:
    if name == "execute_module":
        return "module_category_runtime"
    if name == "run_recipe":
        return "recipe_runtime"
    return "none"


def _tool_annotations(name: str) -> Dict[str, bool]:
    read_only = name in _READ_ONLY_CORE_TOOLS
    return {
        "readOnlyHint": read_only,
        "destructiveHint": not read_only,
        "idempotentHint": read_only,
    }


def _enrich_core_tool_def(tool_def: Dict[str, Any]) -> Dict[str, Any]:
    """Attach agent-facing MCP metadata without changing the callable schema."""
    tool = deepcopy(tool_def)
    name = tool.get("name", "")
    tool.setdefault("description", "")
    tool.setdefault("inputSchema", {"type": "object", "properties": {}})
    tool["annotations"] = _tool_annotations(name)
    tool["metadata"] = {
        "source": "flyto-core",
        "contract_version": CORE_MCP_CONTRACT_VERSION,
        "risk_level": _risk_for_tool(name),
        "approval_policy": _approval_policy_for_tool(name),
        "evidence_fields": [
            "run_id",
            "tool_name",
            "module_id",
            "recipe_name",
            "params_valid",
            "ok",
            "duration_ms",
        ],
    }
    return tool


def _manifest_fingerprint(tools: List[Dict[str, Any]]) -> str:
    payload = [
        {
            "name": t.get("name", ""),
            "inputSchema": t.get("inputSchema", {}),
            "metadata": t.get("metadata", {}),
        }
        for t in tools
    ]
    raw = json_dumps(payload)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def json_dumps(value: Any) -> str:
    """Stable JSON helper kept local to avoid importing heavier utilities."""
    import json
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


# Installation is host authority, never model authority. The rule is structural
# rather than a list of known names: any Core tool whose name *is* an install,
# uninstall, or reinstall verb is withheld from every model-facing catalog and
# refused at dispatch, so a future flyto-core that starts publishing one cannot
# widen this host's LLM surface by upgrade alone. The word boundary is explicit
# so a read-only reporting tool (`list_installed_modules`) is unaffected.
_HOST_ONLY_CORE_TOOL_RE = re.compile(
    r"(?:^|_)(?:un|re)?install(?:_|$)", re.IGNORECASE,
)


def _is_host_only_core_tool(name: Any) -> bool:
    """True when a Core tool name may never reach an LLM tool catalog."""
    return isinstance(name, str) and bool(_HOST_ONLY_CORE_TOOL_RE.search(name))


def _core_tool_defs_from_handler(handler: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = [
        _enrich_core_tool_def(t)
        for t in handler.get("TOOLS", [])
        if not _is_host_only_core_tool(
            t.get("name") if isinstance(t, dict) else None,
        )
    ]
    tools.append(_enrich_core_tool_def(CORE_CAPABILITY_MANIFEST_TOOL))
    return tools


def get_core_tool_defs():
    """Return flyto-core MCP tool definitions (empty list if not installed)."""
    handler = _get_mcp_handler()
    return _core_tool_defs_from_handler(handler) if handler else []


# ---------------------------------------------------------------------------
# Installed-capability manifest — host-derived, fail closed
# ---------------------------------------------------------------------------

CORE_CAPABILITY_MANIFEST_CONTRACT = "flyto.core.capability-manifest.v1"

# The contract stamps its identity under exactly one key. A payload that does
# not declare `schema` is not this contract, so it is rejected rather than
# guessed at from a list of look-alike keys.
_CAPABILITY_MANIFEST_SCHEMA_KEY = "schema"

# Core stamps the digest under `hash` and computes it over the rest of the body.
# The host recomputes it with Core's own function and compares: a digest that
# merely *looks* like a hash proves nothing, and a host-local reimplementation
# of the canonical form would drift from Core and start rejecting every real
# manifest. Both key names come from `core.capability_manifest`, the module that
# owns this wire contract; they are not host choices, and a fixture that renames
# them is testing something Core never emits.
_CAPABILITY_MANIFEST_HASH_KEY = "hash"

# Only used when Core cannot recompute its own digest. Never a substitute for
# a real comparison — see `_manifest_digest_matches`.
_SHA256_HEX_RE = re.compile(r"[a-f0-9]{64}")

# Shared with flyto_ai.capability_router so one identity is safe everywhere.
_SAFE_CAPABILITY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,191}$")
_MAX_CAPABILITY_ENTRIES = 10_000

_cached_capability_manifest_contract = None
_capability_manifest_contract_checked = False


def _get_core_capability_manifest_contract():
    """Lazily bind Core's manifest reader and its digest function together.

    Both live in ``core.capability_manifest``, the module that owns this
    contract. They are bound as a pair on purpose: the digest has to be
    recomputed by the same code that produced it, so the host never needs to
    know — or guess at — Core's canonical serialization.

    Returns ``(get_capability_manifest, compute_manifest_hash)``, or None when
    flyto-core is absent or installed but older than this contract. Result is
    cached like the MCP handler.
    """
    global _cached_capability_manifest_contract
    global _capability_manifest_contract_checked
    if _capability_manifest_contract_checked:
        return _cached_capability_manifest_contract
    _capability_manifest_contract_checked = True
    try:
        from core.capability_manifest import (
            compute_manifest_hash,
            get_capability_manifest,
        )
    except ImportError:
        return None
    if callable(get_capability_manifest) and callable(compute_manifest_hash):
        _cached_capability_manifest_contract = (
            get_capability_manifest,
            compute_manifest_hash,
        )
    return _cached_capability_manifest_contract


def _get_core_capability_manifest_fn():
    """Return Core's manifest reader alone, or None when Core cannot answer.

    Split out from `_get_core_capability_manifest_contract` so the *reader* and
    the *digest recomputation* are independently resolvable. Capability
    discovery has to answer three different states — absent Core, unreadable
    Core, and readable-but-untrusted Core — and binding them to one tuple made
    "Core is too old to report" indistinguishable from "Core cannot verify its
    own digest". Callers and tests substitute this seam, not the tuple.
    """
    contract = _get_core_capability_manifest_contract()
    return None if contract is None else contract[0]


def _get_core_manifest_hash_fn(read: Any = None):
    """Return the digest function paired with the reader that will be used.

    A digest only proves anything when it is recomputed by the same
    implementation that stamped it. Core's `compute_manifest_hash` verifies
    Core's own `get_capability_manifest` and nothing else: run it against a
    manifest from a substituted reader and it compares manifest A's body to
    function B's canonical form, which fails for a manifest that is perfectly
    valid for its own source.

    So the pairing is explicit. When `read` is Core's own reader the real
    function is returned and the digest is fully recomputed. When the reader
    has been substituted -- by a host seam or a test -- there is no matching
    digest function, and `_manifest_digest_matches` falls back to checking the
    digest's form while every other validation stays exactly as strict.
    """
    contract = _get_core_capability_manifest_contract()
    if contract is None:
        return None
    if read is not None and read is not contract[0]:
        return None
    return contract[1]


def _safe_identity(value: Any) -> Optional[str]:
    """Return ``value`` when it is one safe manifest identity, else None."""
    if not isinstance(value, str) or not _SAFE_CAPABILITY_ID.fullmatch(value):
        return None
    return value


def _declared_identity(entry: Any, key: str) -> Optional[str]:
    """Return the safe identity ``entry[key]`` declares, or None if it is unsafe.

    A record that omits the key, or carries anything other than a string, is
    rejected outright rather than defaulted to something harmless-looking.
    """
    if not isinstance(entry, dict):
        return None
    return _safe_identity(entry.get(key))


def _manifest_module_ids(entries: Any) -> Optional[List[str]]:
    """Validate ``modules``: a bounded list of unique module id **strings**.

    Core emits module identities as bare strings, not records. Accepting a
    record shape here would mean the host validates a manifest Core never
    produces while silently rejecting every real one, so the string form is the
    only accepted form.
    """
    if not isinstance(entries, list) or len(entries) > _MAX_CAPABILITY_ENTRIES:
        return None
    module_ids: List[str] = []
    seen = set()
    for entry in entries:
        module_id = _safe_identity(entry)
        if module_id is None or module_id in seen:
            return None
        seen.add(module_id)
        module_ids.append(module_id)
    return module_ids


def _manifest_capability_ids(
    entries: Any, module_ids: frozenset,
) -> Optional[List[str]]:
    """Validate ``capabilities``: unique ``capability`` records with providers.

    Each record declares one capability identity under ``capability`` and the
    installed modules that provide it under ``providers``. Every provider is
    cross-checked against the manifest's own ``modules``, so a capability can
    never introduce a module identity Core did not also declare installed. A
    capability with no providers is not installed by anything, so it is a
    malformed record rather than an empty-but-fine one.
    """
    if not isinstance(entries, list) or len(entries) > _MAX_CAPABILITY_ENTRIES:
        return None
    capability_ids: List[str] = []
    seen = set()
    for entry in entries:
        capability_id = _declared_identity(entry, "capability")
        if capability_id is None or capability_id in seen:
            return None
        providers = entry.get("providers")
        if (
            not isinstance(providers, list)
            or not providers
            or len(providers) > _MAX_CAPABILITY_ENTRIES
        ):
            return None
        for provider in providers:
            if _safe_identity(provider) is None or provider not in module_ids:
                return None
        seen.add(capability_id)
        capability_ids.append(capability_id)
    return capability_ids


def _manifest_plugin_ids(entries: Any) -> Optional[List[str]]:
    """Validate ``plugins``: unique ``id`` records with a version and a count.

    A plugin record names who ships a set of modules: ``id`` is its identity,
    ``version`` is the installed release, and ``module_count`` is how many
    modules it contributes. All three are required, because a partially
    declared plugin is provenance the host cannot report honestly.
    """
    if not isinstance(entries, list) or len(entries) > _MAX_CAPABILITY_ENTRIES:
        return None
    plugin_ids: List[str] = []
    seen = set()
    for entry in entries:
        plugin_id = _declared_identity(entry, "id")
        if plugin_id is None or plugin_id in seen:
            return None
        version = entry.get("version")
        if not isinstance(version, str) or not version.strip():
            return None
        module_count = entry.get("module_count")
        if (
            isinstance(module_count, bool)
            or not isinstance(module_count, int)
            or module_count < 0
        ):
            return None
        seen.add(plugin_id)
        plugin_ids.append(plugin_id)
    return plugin_ids


def _manifest_digest_matches(
    manifest: Dict[str, Any], declared_hash: str, compute_manifest_hash: Any,
) -> bool:
    """Recompute Core's digest over the full body minus the digest itself.

    The body is handed back to Core's own ``compute_manifest_hash`` verbatim,
    minus the one key the digest cannot cover. Covering the *whole* body rather
    than a host-chosen subset is what makes the digest prove the manifest: any
    field Core adds later is protected without a change here, and a field the
    host does not read still cannot be edited after Core signed it.
    """
    if compute_manifest_hash is None:
        # Core can report a manifest but cannot recompute its digest, so there
        # is nothing to compare against. The digest is then only checked for
        # form: it must be a full SHA-256, which rejects an empty or truncated
        # field. This is deliberately weaker than a recomputed match and is the
        # *only* check that degrades here — schema, entry shapes, and all three
        # declared counts are still enforced, and any failure among them still
        # yields an empty frozenset rather than an unfiltered set.
        return bool(_SHA256_HEX_RE.fullmatch(declared_hash.strip().lower()))
    body = {
        key: value
        for key, value in manifest.items()
        if key != _CAPABILITY_MANIFEST_HASH_KEY
    }
    try:
        expected = compute_manifest_hash(body)
    except Exception as e:
        logger.warning("flyto-core compute_manifest_hash failed: %s", e)
        return False
    if not isinstance(expected, str) or not expected.strip():
        return False
    return hmac.compare_digest(
        declared_hash.strip().lower(), expected.strip().lower(),
    )


def _exact_count(manifest: Dict[str, Any], key: str, actual: int) -> bool:
    """A declared count must be present and an exact integer match."""
    declared = manifest.get(key)
    if isinstance(declared, bool) or not isinstance(declared, int):
        return False
    return declared == actual


def _validate_core_capability_manifest(
    manifest: Any, compute_manifest_hash: Any,
) -> Optional[Dict[str, Any]]:
    """Validate schema, real entry shapes, declared counts, and the digest.

    Returns the validated summary, or None when the manifest is malformed.
    Only installed **module** identities leave this function. A capability id
    describes what a module provides and a plugin id names who ships it;
    neither is executable, so unioning them into the module set would let a
    downstream engine offer work no installed module can actually run.
    """
    if not isinstance(manifest, dict) or manifest.get("ok") is False:
        return None
    if (
        manifest.get(_CAPABILITY_MANIFEST_SCHEMA_KEY)
        != CORE_CAPABILITY_MANIFEST_CONTRACT
    ):
        return None

    module_ids = _manifest_module_ids(manifest.get("modules"))
    if module_ids is None:
        return None

    capability_ids = _manifest_capability_ids(
        manifest.get("capabilities"), frozenset(module_ids),
    )
    plugin_ids = _manifest_plugin_ids(manifest.get("plugins"))
    if capability_ids is None or plugin_ids is None:
        return None

    if not (
        _exact_count(manifest, "module_count", len(module_ids))
        and _exact_count(manifest, "capability_count", len(capability_ids))
        and _exact_count(manifest, "plugin_count", len(plugin_ids))
    ):
        return None

    declared_hash = manifest.get(_CAPABILITY_MANIFEST_HASH_KEY)
    if not isinstance(declared_hash, str) or not declared_hash.strip():
        return None
    if not _manifest_digest_matches(
        manifest, declared_hash, compute_manifest_hash,
    ):
        return None

    return {
        "hash": declared_hash.strip().lower(),
        "module_ids": frozenset(module_ids),
        "capability_count": len(capability_ids),
        "plugin_count": len(plugin_ids),
    }


def _read_core_installed_module_ids() -> Tuple[
    Optional[frozenset], Dict[str, Any],
]:
    """Read and validate Core's installed-capability manifest once.

    Returns ``(module_ids, summary)``. ``module_ids`` is None only when the
    installed Core cannot answer at all; every other outcome, including a
    failing or malformed answer from a manifest-capable Core, is a frozenset so
    an unreadable runtime can never widen what downstream engines offer.

    The identity set and the summary are returned separately on purpose. The
    summary is published verbatim as ``installed_capabilities`` on the public
    manifest, so it carries provenance -- Core's digest and aggregate counts --
    and never an identity. Keeping the ids out of it entirely means a future
    edit cannot leak them by forgetting to strip a key.
    """
    summary: Dict[str, Any] = {
        "contract": CORE_CAPABILITY_MANIFEST_CONTRACT,
        "source": "flyto-core",
        "supported": False,
        "status": "unsupported_core",
        "manifest_hash": "",
        "module_count": 0,
        "capability_count": 0,
        "plugin_count": 0,
    }
    read = _get_core_capability_manifest_fn()
    if read is None:
        return None, summary
    compute_manifest_hash = _get_core_manifest_hash_fn(read)
    summary["supported"] = True
    try:
        raw = read()
    except Exception as e:
        logger.warning("flyto-core get_capability_manifest failed: %s", e)
        summary["status"] = "error"
        return frozenset(), summary

    validated = _validate_core_capability_manifest(raw, compute_manifest_hash)
    if validated is None:
        logger.warning(
            "flyto-core capability manifest rejected: schema, digest, count, "
            "module, capability or plugin validation failed",
        )
        summary["status"] = "invalid"
        return frozenset(), summary

    module_ids = validated["module_ids"]
    summary.update({
        "status": "ok",
        "manifest_hash": validated["hash"],
        "module_count": len(module_ids),
        # Exactly what Core declared, each as its own number. These are
        # provenance, not a filter input: only `module_count` describes the set
        # that is handed to a Blueprint engine.
        "capability_count": validated["capability_count"],
        "plugin_count": validated["plugin_count"],
    })
    return module_ids, summary


def get_core_installed_module_ids() -> Optional[frozenset]:
    """Return the host-derived set of installed Core module identifiers.

    None means the installed flyto-core is absent or too old to report an
    installed-capability manifest, so callers must leave their own filtering
    alone. A validated manifest always yields a frozenset — empty when Core
    reports nothing installed, and also empty when a manifest-capable Core
    errors or returns a malformed manifest.

    The result holds module IDs only. A capability id names what a module
    provides and a plugin id names who ships it; neither is executable, so
    unioning either into this set would let a downstream engine offer work no
    installed module can actually run. Both are validated as provenance and
    counted, and neither is ever joined in.
    """
    module_ids, _summary = _read_core_installed_module_ids()
    return module_ids


def get_core_capability_manifest(
    include_tools: bool = True,
    include_categories: bool = True,
) -> Dict[str, Any]:
    """Build the flyto-core MCP manifest used by agents and cloud diagnostics."""
    handler = _get_mcp_handler()
    if handler is None:
        return {
            "ok": False,
            "source": "flyto-core",
            "contract_version": CORE_MCP_CONTRACT_VERSION,
            "error": "flyto-core not installed. Run: pip install flyto-core",
        }

    tools = _core_tool_defs_from_handler(handler)
    manifest: Dict[str, Any] = {
        "ok": True,
        "source": "flyto-core",
        "contract_version": CORE_MCP_CONTRACT_VERSION,
        "core_version": _core_package_version(),
        "tool_count": len(tools),
        "tool_fingerprint": _manifest_fingerprint(tools),
        "recipes_available": bool(handler.get("list_recipes") and handler.get("run_recipe")),
        "approval_model": {
            "execute_module": "module category decides runtime approval",
            "run_recipe": "recipe content decides runtime approval",
            "sensitive_inputs": "runtime secrets only; never request credentials through MCP elicitation",
        },
    }

    if include_tools:
        manifest["tools"] = [
            {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "inputSchema": t.get("inputSchema", {}),
                "annotations": t.get("annotations", {}),
                "metadata": t.get("metadata", {}),
            }
            for t in tools
        ]

    if include_categories:
        try:
            categories = handler["list_modules"](category=None).get("categories", [])
            manifest["categories"] = categories
            manifest["module_count"] = sum(c.get("count", 0) for c in categories)
        except Exception as e:
            manifest["categories_error"] = str(e)

    # Installed-capability provenance for agents and cloud diagnostics. The
    # identifiers themselves stay host-side; only validated counts and the
    # Core-owned digest cross the model-facing boundary.
    _module_ids, capability_summary = _read_core_installed_module_ids()
    manifest["installed_capabilities"] = capability_summary

    return manifest


# ---------------------------------------------------------------------------
# Browser retry — transient error detection + smart retry at dispatch level
# ---------------------------------------------------------------------------

# Import shared error classification from flyto-core (canonical source)
try:
    from core.modules.atomic.llm._resilience import (
        is_transient_error as _is_transient_error,
        is_session_dead as _is_session_dead,
    )
except ImportError:
    # No second copy of Core's pattern lists. Both classifiers are only ever
    # asked about an error returned by a `browser.*` dispatch, and that dispatch
    # goes through Core — with no Core installed there is no error here to
    # classify. A duplicated list would be an unreachable branch that silently
    # falls behind whatever Core learns about browser failures, which is how the
    # snapshot/interact sets in assistant/resilience.py drifted.
    def _is_transient_error(error_msg: str) -> bool:
        return False

    def _is_session_dead(error_msg: str) -> bool:
        return False


async def _relaunch_browser() -> Dict[str, Any]:
    """Attempt to relaunch a fresh browser session."""
    handler = _get_mcp_handler()
    if handler is None:
        return {"ok": False, "error": "flyto-core not installed"}
    try:
        result = await _dispatch_core_tool_inner("execute_module", {
            "module_id": "browser.launch", "params": {},
        })
        return result
    except Exception as e:
        return {"ok": False, "error": "Relaunch failed: {}".format(e)}


async def dispatch_core_tool(
    name: str,
    arguments: Dict[str, Any],
    *,
    trusted_outbound_scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dispatch a tool call to the flyto-core MCP handler.

    For browser.* modules, transient errors trigger one automatic retry.
    If the session is dead, a fresh browser.launch is attempted before retrying.

    ``trusted_outbound_scope`` is an internal Runner-to-Core integration hook.
    It is deliberately absent from the public MCP tool schema and is applied
    only around the current async Core call.
    """
    result = await _dispatch_core_tool_inner(
        name,
        arguments,
        trusted_outbound_scope=trusted_outbound_scope,
    )

    # Smart retry — only for browser module execute_module calls
    if (
        name == "execute_module"
        and isinstance(result, dict)
        and not _is_ok(result)
    ):
        module_id = arguments.get("module_id", "")
        error_msg = str(result.get("error", ""))

        if module_id.startswith("browser.") and _is_transient_error(error_msg):
            logger.info("Browser transient error on %s, retrying: %s", module_id, error_msg[:100])

            # Session dead → relaunch first
            if _is_session_dead(error_msg):
                relaunch = await _relaunch_browser()
                if not _is_ok(relaunch):
                    relaunch_err = relaunch.get("error", "unknown")
                    logger.warning("Browser relaunch failed: %s", relaunch_err)
                    return {
                        "ok": False,
                        "error": "Browser session dead ({}). Relaunch also failed: {}".format(
                            error_msg[:100], relaunch_err,
                        ),
                    }

            # Retry once
            result = await _dispatch_core_tool_inner(
                name,
                arguments,
                trusted_outbound_scope=trusted_outbound_scope,
            )

    return result


# ---------------------------------------------------------------------------
# search_modules guardrail — detect web search misuse
# ---------------------------------------------------------------------------

# Module-like pattern: dot notation (browser.launch, string.upper)
_MODULE_QUERY_RE = re.compile(r"[a-z][a-z0-9_]*\.[a-z]", re.IGNORECASE)

# Automation-related keywords that signal a legitimate module search
_AUTOMATION_KEYWORDS = frozenset([
    "click", "type", "extract", "screenshot", "resize", "convert", "send",
    "email", "file", "image", "api", "http", "json", "csv", "pdf", "database",
    "sql", "scrape", "download", "upload", "parse", "format", "encode", "decode",
    "compress", "encrypt", "hash", "wait", "scroll", "select", "form", "login",
    "notify", "slack", "telegram", "webhook", "string", "array", "datetime",
    "evaluate", "snapshot", "launch", "goto", "navigate", "browser", "fill",
    "submit", "button", "input", "checkbox", "dropdown", "module", "workflow",
])


def _looks_like_module_query(query: str) -> bool:
    """Check if a query looks like it's searching for an automation module.

    Returns True if the query contains module-like patterns (dot notation)
    or automation-related keywords.
    """
    q = query.strip()
    if not q:
        return False

    # Dot notation like "browser.launch" → definitely module search
    if _MODULE_QUERY_RE.search(q):
        return True

    # Contains automation keywords → module search
    q_lower = q.lower()
    return any(kw in q_lower for kw in _AUTOMATION_KEYWORDS)


_sandbox_mgr = None


def set_sandbox_manager(mgr) -> None:
    """Set the sandbox manager for sandboxed module execution."""
    global _sandbox_mgr
    _sandbox_mgr = mgr


def _trusted_outbound_context(scope: Optional[Dict[str, Any]]):
    """Resolve Core's task-local scope lazily and fail closed when unavailable."""
    if scope is None:
        return nullcontext()
    from core.utils import trusted_outbound_network_scope

    return trusted_outbound_network_scope(
        allowed_hosts=scope.get("allowed_hosts", []),
        allowed_ports=scope.get("allowed_ports", []),
        allow_private_targets=(scope.get("allow_private_targets") is True),
    )


async def _dispatch_core_tool_inner(
    name: str,
    arguments: Dict[str, Any],
    *,
    trusted_outbound_scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Core dispatch logic (no retry)."""
    from flyto_ai.tools.browser_scope import current_browser_scope
    browser_scope = current_browser_scope()
    if browser_scope is not None and browser_scope.closed and name in _EXECUTION_CORE_TOOLS:
        return {"ok": False, "error": "The owned browser execution scope is closed"}
    if browser_scope is not None and browser_scope.closing and name in _EXECUTION_CORE_TOOLS:
        if name != "execute_module" or arguments.get("module_id") != "browser.close":
            return {"ok": False, "error": "The owned browser execution scope is closing"}
    browser_sessions = _active_browser_sessions()
    # Withheld from the catalog *and* refused here. A name that never appears in
    # `get_core_tool_defs` can still be typed into a tool call by a model or a
    # forwarding client, so the catalog filter alone is not the boundary.
    if _is_host_only_core_tool(name):
        return {
            "ok": False,
            "error": (
                "Extension installation is host-only and is not callable "
                "through the flyto-core MCP tool surface."
            ),
        }

    handler = _get_mcp_handler()
    if handler is None:
        return {"ok": False, "error": "flyto-core not installed. Run: pip install flyto-core"}

    if name == "list_modules":
        return handler["list_modules"](category=arguments.get("category"))

    elif name == "get_core_capability_manifest":
        return get_core_capability_manifest(
            include_tools=arguments.get("include_tools", True),
            include_categories=arguments.get("include_categories", True),
        )

    elif name == "search_modules":
        query = arguments.get("query", "")
        result = handler["search_modules"](
            query=query,
            category=arguments.get("category"),
            limit=arguments.get("limit", 20),
        )
        # Guardrail: if no results and query doesn't look like a module search, hint browser
        if isinstance(result, dict) and result.get("total", 0) == 0 and not _looks_like_module_query(query):
            result["web_search_hint"] = (
                "No modules match this query. This looks like a web search request. "
                "Use Browser Protocol instead: "
                "execute_module('browser.launch') → "
                "execute_module('browser.goto', {url: 'https://www.google.com/search?q=...'}) → "
                "execute_module('browser.snapshot') to read the results."
            )
            logger.info("search_modules guardrail: query looks like web search: %s", query[:50])
        return result

    elif name == "get_module_info":
        return handler["get_module_info"](module_id=arguments.get("module_id", ""))

    elif name == "get_module_examples":
        return handler["get_module_examples"](module_id=arguments.get("module_id", ""))

    elif name == "execute_module":
        launch_failed, launch_error, goto_failures = _browser_retry_state()
        module_id = arguments.get("module_id", "")
        from flyto_ai.tools.parameter_references import resolve_module_params, UnresolvedParameterReference
        try:
            arguments = {**arguments, "params": resolve_module_params(
                arguments.get("params", {}), arguments.get("context"),
            )}
        except UnresolvedParameterReference:
            return {
                "ok": False, "error_code": "unresolved_parameter_reference",
                "params_valid": False, "retryable": True, "module_id": module_id,
                "error": "Module parameters contain an unresolved workflow reference; no action was executed.",
                "suggestion": (
                    "Use the actual authorized input value or an explicitly supplied workflow binding. "
                    "Schema examples are templates, not live credentials. Keep the current browser "
                    "and correct this call without repeating successful mutations."
                ),
            }

        # Browser cascade breaker: if browser.launch already failed,
        # skip all subsequent browser.* calls immediately.
        if module_id.startswith("browser.") and module_id not in {"browser.launch", "browser.close"} and launch_failed:
            return {
                "ok": False,
                "error": "Skipped: browser.launch failed earlier ({}). Fix browser.launch first.".format(
                    launch_error[:100],
                ),
            }

        # Goto circuit breaker: stop retrying after N consecutive failures
        if module_id == "browser.goto" and goto_failures >= _GOTO_MAX_FAILS:
            return {
                "ok": False,
                "error": (
                    "STOP: browser.goto has failed {} times consecutively. "
                    "Do NOT call browser.goto again. "
                    "Try browser.goto with a Google search URL instead: "
                    "https://www.google.com/search?q=YOUR+QUERY"
                ).format(goto_failures),
            }

        # On new browser.launch: close existing sessions to avoid
        # "Multiple browser sessions active" errors.
        if module_id == "browser.launch":
            _set_browser_retry_state(False, "", 0)
            if browser_sessions:
                for sid in list(browser_sessions.keys()):
                    try:
                        closed_result = await handler["execute_module"](
                            module_id="browser.close",
                            params={},
                            context={"browser_session": sid},
                            browser_sessions=browser_sessions,
                        )
                        if browser_scope is not None:
                            confirmed = closed_result.get("ok") if isinstance(closed_result.get("ok"), bool) else closed_result.get("status") == "success"
                            if not confirmed:
                                return {"ok": False, "error": "An owned browser could not be closed before relaunch"}
                            browser_sessions.pop(sid, None)
                            browser_scope.closed_session_ids.append(sid)
                    except Exception:
                        if browser_scope is not None:
                            return {"ok": False, "error": "An owned browser could not be closed before relaunch"}
                with _browser_sessions_lock:
                    if browser_scope is None:
                        browser_sessions.clear()

        validation = _validate_execute_module_args(handler, module_id, arguments.get("params", {}))
        if validation is not None:
            return validation

        # Sandbox: route dangerous categories to Docker container
        if _sandbox_mgr and _sandbox_mgr.needs_sandbox(module_id):
            if trusted_outbound_scope is not None:
                return {
                    "ok": False,
                    "error": (
                        "Trusted outbound scope cannot be enforced inside "
                        "the configured sandbox"
                    ),
                }
            return await _sandbox_mgr.execute(
                module_id, arguments.get("params", {}), arguments.get("context"),
            )
        try:
            scope_context = _trusted_outbound_context(
                trusted_outbound_scope,
            )
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            return {
                "ok": False,
                "error": (
                    "Installed flyto-core cannot enforce trusted outbound "
                    "scope: {}"
                ).format(exc),
            }
        previous_sessions = set(browser_sessions)
        with scope_context:
            result = await handler["execute_module"](
                module_id=module_id,
                params=arguments.get("params", {}),
                context=arguments.get("context"),
                browser_sessions=browser_sessions,
            )
        if browser_scope is not None:
            browser_scope.owned_session_ids.update(previous_sessions)
            browser_scope.owned_session_ids.update(browser_sessions)
            if module_id == "browser.close" and isinstance(result, dict):
                confirmed = result.get("ok") if isinstance(result.get("ok"), bool) else result.get("status") == "success"
                if confirmed:
                    for sid in previous_sessions - set(browser_sessions):
                        if sid not in browser_scope.closed_session_ids:
                            browser_scope.closed_session_ids.append(sid)

        # Track browser.launch failure for cascade breaker
        if module_id == "browser.launch" and isinstance(result, dict) and not _is_ok(result):
            _set_browser_retry_state(True, str(result.get("error", "unknown error")), goto_failures)

        # Track goto failures for circuit breaker
        if module_id == "browser.goto":
            if isinstance(result, dict) and _is_ok(result):
                goto_failures = 0
            else:
                goto_failures += 1
            _set_browser_retry_state(launch_failed, launch_error, goto_failures)

        return result

    elif name == "validate_params":
        return handler["validate_params"](
            module_id=arguments.get("module_id", ""),
            params=arguments.get("params", {}),
        )

    elif name == "list_recipes":
        fn = handler.get("list_recipes")
        if not fn:
            return {"ok": False, "error": "Recipe support requires flyto-core >= 2.15.0"}
        return fn()

    elif name == "run_recipe":
        fn = handler.get("run_recipe")
        if not fn:
            return {"ok": False, "error": "Recipe support requires flyto-core >= 2.15.0"}
        return await fn(
            recipe_name=arguments.get("recipe_name", ""),
            args=arguments.get("args", {}),
            browser_sessions=browser_sessions,
        )

    return {"ok": False, "error": "Unknown core tool: {}".format(name)}


def _validate_execute_module_args(
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


# ---------------------------------------------------------------------------
# Core extension management — host-only, generic over Core's own kinds
# ---------------------------------------------------------------------------
#
# Core owns what an extension *is*. The surface lives in `core.plugin.loader`:
# `get_plugin_loader()` returns the loader that performs work, `EXTENSION_KINDS`
# declares the kinds, `normalize_extension_name` folds a requested name to the
# identity Core acts on, and `ExtensionResult` carries the outcome. This adapter
# adds no taxonomy and no name rules of its own — it is generic over whatever
# `EXTENSION_KINDS` declares — and it deliberately gives back less than Core does.
#
# Three properties hold for every function below.
#
# 1. Host-only. None of these are Core MCP tools, none are reachable through
#    `dispatch_core_tool`, and `get_core_tool_defs` filters install-shaped names
#    out of the catalog. Installing software is host authority; a model may ask
#    an operator for it, but it may not perform it.
# 2. Mutation is opt-in. `FLYTO_EXTENSIONS_INSTALL_ENABLED` gates install *and*
#    uninstall, is read per call, and is checked before anything else — a host
#    that has not opted in never validates a request, never imports Core's
#    installer, and answers `install_disabled` for every input.
# 3. The envelope is fixed and carries no installer output. Every result has
#    exactly the same keys, every free-form field is a bounded safe token, and
#    no pip/subprocess stdout, stderr, traceback, or exception text is ever
#    copied into it. Installer detail belongs in Core's logs, not in a payload
#    that may be rendered by a cloud UI or handed back to a model.

CORE_EXTENSION_CONTRACT = "flyto.core.extension-management.v1"

# Read per call, never cached: an operator toggles this between runs.
CORE_EXTENSION_INSTALL_ENV = "FLYTO_EXTENSIONS_INSTALL_ENABLED"

_CORE_EXTENSION_INSTALL_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Fixed envelope. Every outcome — success, refusal, absent Core, malformed Core
# answer — returns exactly these keys, so nothing can be inferred from the shape
# of a result and no later edit can leak an extra field by accident.
_EXTENSION_ENVELOPE_KEYS = (
    "ok",
    "contract",
    "source",
    "operation",
    "code",
    "name",
    "kind",
    "version",
    "previous_version",
    "install_enabled",
    "restart_required",
    "rolled_back",
    "refresh_failed",
    "extensions",
    "kinds",
)

# Core's loader method per operation. The host's operation names and Core's
# method names are not the same words, and pretending they are (calling
# `getattr(loader, operation)`) is how this bridge previously called an
# `install` that does not exist.
_EXTENSION_LOADER_METHODS = {
    "list": "list_extensions",
    "install": "install_extension",
    "uninstall": "uninstall_extension",
}

# The fields this host publishes out of Core's `ExtensionResult`, under Core's
# own names. There is one accepted name per field and no alias list: a
# look-alike fallback would let a Core rename pass silently as "field absent",
# which is the failure mode that made the capability-manifest bridge report an
# empty world. `tests/test_core_extension_management.py` binds the installed
# `ExtensionResult` and fails when a name here is not one of its fields, so a
# rename is a loud test failure and a one-line change here.
_EXTENSION_RESULT_OK_FIELD = "ok"

# What Core reports about the *consequences* of a mutation. `restart_required`
# says the host process must restart before the change takes effect,
# `rolled_back` says Core undid a failed install, and `refresh_failed` says the
# registry refresh after the change did not succeed — a partial success an
# operator has to know about, since the extension may be installed but not yet
# usable.
_EXTENSION_RESULT_FLAG_FIELDS = (
    "restart_required",
    "rolled_back",
    "refresh_failed",
)
_EXTENSION_RESULT_FIELDS = (
    _EXTENSION_RESULT_OK_FIELD,
    "code",
    "name",
    "kind",
    "version",
    # The version that was installed before this operation. It is what an
    # operator needs to undo an upgrade by hand, so it is published rather
    # than dropped.
    "previous_version",
    *_EXTENSION_RESULT_FLAG_FIELDS,
)

# One `EXTENSION_KINDS` record. `kind` is the identity; `prefix` is the package
# name prefix a kind's distributions carry and `entry_point_group` is the group
# Core discovers them under. Both are provenance an operator needs to know what
# a kind actually selects, and neither is host-authored.
_EXTENSION_KIND_FIELDS = ("kind", "prefix", "entry_point_group")

# Core-authored identifiers are passed through bounded, never reformatted.
_MAX_EXTENSION_LABEL = 200

# Host-owned outcome codes. Core's own code is preserved when it supplies one
# (see `_core_extension_code`); these are the fallbacks for the states the host
# decides by itself.
EXTENSION_CODE_OK = "ok"
EXTENSION_CODE_CORE_UNAVAILABLE = "core_unavailable"
EXTENSION_CODE_INSTALL_DISABLED = "install_disabled"
EXTENSION_CODE_INVALID_REQUEST = "invalid_request"
EXTENSION_CODE_CORE_ERROR = "core_error"
EXTENSION_CODE_INVALID_RESULT = "invalid_core_result"

# A requested name is validated before it reaches Core. The pattern cannot start
# with `-`, so a request can never arrive at an installer as an option, and it
# admits no whitespace, path separator, URL, or shell metacharacter.
_SAFE_EXTENSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_EXTENSION_KIND = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SAFE_EXTENSION_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")

# Codes are bounded short tokens. This is what keeps a Core failure that stuffed
# an installer log into its code field from becoming installer output on the
# wire: an oversized or free-form value simply is not a code.
_SAFE_EXTENSION_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

_MAX_EXTENSION_ENTRIES = 1_000

_cached_extension_manager = None
_extension_manager_checked = False


def _get_core_extension_manager():
    """Lazily bind Core's extension surface from ``core.plugin.loader``.

    The four names are bound as one unit because they only mean anything
    together: `get_plugin_loader` performs the work, `EXTENSION_KINDS` says what
    kinds exist, `normalize_extension_name` decides which identity a request
    refers to, and `ExtensionResult` is the shape an outcome arrives in. A Core
    missing any of them is a partial contract this host will not run half of.

    Returns None when flyto-core is absent or predates the surface. Callers
    report that as ``core_unavailable`` rather than as an empty installed set —
    "cannot answer" and "nothing installed" are different states, and collapsing
    them is how a host starts reporting a fully installed Core as empty.

    The loader itself is *not* resolved here. `get_plugin_loader()` may touch
    the filesystem, so it is called inside the worker thread with the operation
    it serves.

    This is the substitution seam for tests; nothing else here imports Core.
    """
    global _cached_extension_manager, _extension_manager_checked
    if _extension_manager_checked:
        return _cached_extension_manager
    _extension_manager_checked = True
    try:
        from core.plugin.loader import (
            EXTENSION_KINDS,
            ExtensionResult,
            get_plugin_loader,
            normalize_extension_name,
        )
    except ImportError:
        return None
    if not (callable(get_plugin_loader) and callable(normalize_extension_name)):
        return None
    _cached_extension_manager = {
        "loader": get_plugin_loader,
        "kinds": EXTENSION_KINDS,
        "normalize": normalize_extension_name,
        "result_type": ExtensionResult,
    }
    return _cached_extension_manager


def core_extension_install_enabled() -> bool:
    """True when the operator opted this host into extension mutation."""
    raw = os.getenv(CORE_EXTENSION_INSTALL_ENV, "")
    return raw.strip().lower() in _CORE_EXTENSION_INSTALL_TRUTHY


def _extension_envelope(
    operation: str,
    *,
    ok: bool = False,
    code: str,
    name: str = "",
    kind: str = "",
    version: str = "",
    previous_version: str = "",
    install_enabled: bool = False,
    restart_required: bool = False,
    rolled_back: bool = False,
    refresh_failed: bool = False,
    extensions: Optional[List[Dict[str, Any]]] = None,
    kinds: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the one envelope shape this adapter is allowed to return.

    The three flags are Core's own, under Core's own names, and they all
    default to False so a Core that does not report one is never read as "yes".
    ``install_enabled`` is the *host's* opt-in state, not a Core field; it is
    published so a caller can tell "refused because this host has not opted in"
    from "Core refused".
    """
    return {
        "ok": bool(ok),
        "contract": CORE_EXTENSION_CONTRACT,
        "source": "flyto-core",
        "operation": operation,
        "code": code,
        "name": name,
        "kind": kind,
        "version": version,
        "previous_version": previous_version,
        "install_enabled": bool(install_enabled),
        "restart_required": bool(restart_required),
        "rolled_back": bool(rolled_back),
        "refresh_failed": bool(refresh_failed),
        "extensions": list(extensions or []),
        "kinds": list(kinds or []),
    }


def _extension_result_mapping(result: Any) -> Optional[Dict[str, Any]]:
    """Read Core's ``ExtensionResult`` without assuming how it is built.

    Core owns whether the result is a dataclass, a NamedTuple, a plain object,
    or a mapping, and that choice is not part of the contract this host depends
    on — the *field names* are. Reading it structurally means a Core that swaps
    representation does not break the bridge, while a Core that renames a field
    fails the real-contract test instead of silently publishing empty values.
    """
    if isinstance(result, dict):
        return dict(result)
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return {
            field.name: getattr(result, field.name, None)
            for field in dataclasses.fields(result)
        }
    as_dict = getattr(result, "_asdict", None)
    if callable(as_dict):
        try:
            values = as_dict()
        except Exception:
            return None
        if isinstance(values, dict):
            return dict(values)
        return None
    values = getattr(result, "__dict__", None)
    return dict(values) if isinstance(values, dict) else None


def _safe_extension_prose(value: Any) -> str:
    """Bound one Core-authored identifier; never installer output."""
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return text[:_MAX_EXTENSION_LABEL]


def _safe_extension_token(
    value: Any, pattern: Any, *, lower: bool = False,
) -> Optional[str]:
    """Return the bounded token ``value`` carries, or None when it is unsafe."""
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower() if lower else value.strip()
    return candidate if pattern.fullmatch(candidate) else None


def _core_extension_code(result: Dict[str, Any], fallback: str) -> str:
    """Preserve Core's own result/error code, under Core's own key.

    Core reports the outcome of an extension operation under ``code`` — the same
    field for success and failure — so the host does not need, and does not
    keep, a translation table that would drift from Core's vocabulary. Only one
    key is read: accepting a list of look-alike keys would mean inventing a
    contract Core never agreed to. A missing or unsafe code degrades to the
    host's fallback rather than to raw text.
    """
    code = _safe_extension_token(result.get("code"), _SAFE_EXTENSION_CODE, lower=True)
    return code or fallback


def _normalize_extension_record(entry: Any) -> Optional[Dict[str, Any]]:
    """Normalize one Core extension record to fixed, bounded fields.

    ``name`` is Core's normalized identity and is passed through verbatim; the
    host validates its form but never rewrites it, because a host-side rename
    would make an operator's installed set unaddressable in Core's own terms.
    ``kind`` and ``version`` are optional in the wire record and become empty
    strings when Core does not report them. An entry Core lists counts as
    installed unless Core explicitly says ``installed: False``.
    """
    entry = _extension_result_mapping(entry)
    if entry is None:
        return None
    name = _safe_extension_token(entry.get("name"), _SAFE_EXTENSION_NAME)
    if name is None:
        return None

    raw_kind = entry.get("kind")
    if raw_kind is None or raw_kind == "":
        kind = ""
    else:
        kind = _safe_extension_token(raw_kind, _SAFE_EXTENSION_KIND, lower=True)
        if kind is None:
            return None

    raw_version = entry.get("version")
    if raw_version is None or raw_version == "":
        version = ""
    else:
        version = _safe_extension_token(raw_version, _SAFE_EXTENSION_VERSION)
        if version is None:
            return None

    installed = entry.get("installed")
    if installed is not None and not isinstance(installed, bool):
        return None

    return {
        "name": name,
        "kind": kind,
        "version": version,
        "installed": installed is not False,
    }


def _normalize_extension_records(entries: Any) -> Optional[List[Dict[str, Any]]]:
    """Validate ``extensions``: a bounded list of unique, well-formed records."""
    if not isinstance(entries, list) or len(entries) > _MAX_EXTENSION_ENTRIES:
        return None
    records: List[Dict[str, Any]] = []
    seen = set()
    for entry in entries:
        record = _normalize_extension_record(entry)
        if record is None:
            return None
        identity = (record["kind"], record["name"])
        if identity in seen:
            return None
        seen.add(identity)
        records.append(record)
    return records


def _normalize_extension_kinds(entries: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize ``EXTENSION_KINDS``: bounded, unique kind **records**.

    Core declares kinds as records, not bare strings, so the identity and the
    operator-facing copy travel together. The host has no kind taxonomy of its
    own: whatever Core declares here is exactly what `list` will filter on and
    what an operator can choose, which is what makes this adapter generic
    instead of a fixed switch over today's extension families.
    """
    if not isinstance(entries, (list, tuple)) or len(entries) > _MAX_EXTENSION_ENTRIES:
        return None
    kinds: List[Dict[str, Any]] = []
    seen = set()
    for entry in entries:
        record = _extension_result_mapping(entry)
        if record is None:
            return None
        kind = _safe_extension_token(
            record.get("kind"), _SAFE_EXTENSION_KIND, lower=True,
        )
        if kind is None or kind in seen:
            return None
        seen.add(kind)
        kinds.append({
            "kind": kind,
            "prefix": _safe_extension_prose(record.get("prefix")),
            "entry_point_group": _safe_extension_prose(
                record.get("entry_point_group"),
            ),
        })
    return kinds


async def _call_core_extension(
    operation: str, manager: Dict[str, Any], **kwargs: Any,
) -> Any:
    """Resolve Core's plugin loader and run one blocking call off the loop.

    The loader method is looked up from `_EXTENSION_LOADER_METHODS` rather than
    derived from the operation name, because Core's methods are
    `list_extensions` / `install_extension` / `uninstall_extension` and the
    host's operations are `list` / `install` / `uninstall`.

    `get_plugin_loader()` is called here rather than at bind time because it may
    touch the filesystem, and the whole thing runs in a worker thread: Core's
    extension operations are synchronous and an install shells out to a package
    installer, so calling one inline would stall every other task on the loop
    for the length of a network install.

    Only the exception *type* is logged. An installer failure raises with the
    whole build log attached, and that log must not reach this host's transcript
    any more than it reaches the returned envelope. Core logs the detail.
    """
    method = _EXTENSION_LOADER_METHODS[operation]

    def run() -> Any:
        loader = manager["loader"]()
        fn = getattr(loader, method, None)
        if not callable(fn):
            raise AttributeError(
                "flyto-core plugin loader has no {} method".format(method),
            )
        return fn(**kwargs)

    try:
        return await asyncio.to_thread(run)
    except Exception as e:
        logger.warning(
            "flyto-core extension %s failed: %s",
            operation,
            type(e).__name__,
        )
        return None


def _safe_requested_kind(kind: Any) -> Optional[str]:
    """Resolve an optional caller-supplied kind filter.

    Returns ``""`` for "no filter", the bounded token for a usable one, and
    None when the caller supplied something that is not a kind at all.
    """
    if kind is None or kind == "":
        return ""
    return _safe_extension_token(kind, _SAFE_EXTENSION_KIND, lower=True)


def _filter_extensions_by_kind(
    records: List[Dict[str, Any]], kind: str,
) -> List[Dict[str, Any]]:
    """Narrow a listing to one kind, host-side.

    Core's `list_extensions` takes no kind parameter — it answers with
    everything it knows — so the filter belongs here. It is applied after
    normalization, so it matches on the same bounded token this host publishes
    rather than on whatever raw value Core happened to carry.
    """
    if not kind:
        return records
    return [record for record in records if record["kind"] == kind]


async def list_core_extensions(kind: Optional[str] = None) -> Dict[str, Any]:
    """List the extensions Core reports, optionally narrowed to one kind.

    Core's `list_extensions` answers with a plain list of records, not an
    envelope, so there is no Core-level ok/code to preserve here: reaching a
    list at all is the success, and any failure surfaces as a raised exception.
    It takes no kind argument either, so ``kind`` is applied host-side.
    """
    operation = "list"
    enabled = core_extension_install_enabled()
    safe_kind = _safe_requested_kind(kind)
    if safe_kind is None:
        return _extension_envelope(
            operation,
            code=EXTENSION_CODE_INVALID_REQUEST,
            install_enabled=enabled,
        )

    manager = _get_core_extension_manager()
    if manager is None:
        return _extension_envelope(
            operation,
            code=EXTENSION_CODE_CORE_UNAVAILABLE,
            kind=safe_kind,
            install_enabled=enabled,
        )

    result = await _call_core_extension(operation, manager)
    if result is None:
        return _extension_envelope(
            operation,
            code=EXTENSION_CODE_CORE_ERROR,
            kind=safe_kind,
            install_enabled=enabled,
        )

    records = _normalize_extension_records(result)
    if records is None:
        logger.warning("flyto-core extension list rejected: malformed records")
        return _extension_envelope(
            operation,
            code=EXTENSION_CODE_INVALID_RESULT,
            kind=safe_kind,
            install_enabled=enabled,
        )
    return _extension_envelope(
        operation,
        ok=True,
        code=EXTENSION_CODE_OK,
        kind=safe_kind,
        install_enabled=enabled,
        extensions=_filter_extensions_by_kind(records, safe_kind),
    )


async def list_core_extension_kinds() -> Dict[str, Any]:
    """List the extension kinds Core itself declares.

    `EXTENSION_KINDS` is a module constant, so this reads no filesystem and
    needs no worker thread. It stays async so every operation on this adapter
    has one calling convention.
    """
    operation = "kinds"
    enabled = core_extension_install_enabled()
    manager = _get_core_extension_manager()
    if manager is None:
        return _extension_envelope(
            operation,
            code=EXTENSION_CODE_CORE_UNAVAILABLE,
            install_enabled=enabled,
        )

    kinds = _normalize_extension_kinds(manager["kinds"])
    if kinds is None:
        logger.warning("flyto-core extension kinds rejected: malformed entries")
        return _extension_envelope(
            operation,
            code=EXTENSION_CODE_INVALID_RESULT,
            install_enabled=enabled,
        )
    return _extension_envelope(
        operation,
        ok=True,
        code=EXTENSION_CODE_OK,
        install_enabled=enabled,
        kinds=kinds,
    )


async def _mutate_core_extension(
    operation: str,
    name: Any,
    *,
    version: Any = None,
    upgrade: Any = False,
) -> Dict[str, Any]:
    """Shared install/uninstall path: opt-in gate, normalize, call, publish."""
    # The gate comes first and takes no input into account. A host that has not
    # opted in gives the same answer to every request and never reaches Core.
    enabled = core_extension_install_enabled()
    if not enabled:
        return _extension_envelope(
            operation, code=EXTENSION_CODE_INSTALL_DISABLED,
        )

    requested = _safe_extension_token(name, _SAFE_EXTENSION_NAME)
    safe_version = (
        None if version is None
        else _safe_extension_token(version, _SAFE_EXTENSION_VERSION)
    )
    if (
        requested is None
        or (version is not None and safe_version is None)
        or not isinstance(upgrade, bool)
    ):
        return _extension_envelope(
            operation,
            code=EXTENSION_CODE_INVALID_REQUEST,
            install_enabled=True,
        )

    manager = _get_core_extension_manager()
    if manager is None:
        return _extension_envelope(
            operation,
            code=EXTENSION_CODE_CORE_UNAVAILABLE,
            name=requested,
            install_enabled=True,
        )

    # Core folds a requested name to the identity it will actually act on;
    # case, separator, and alias rules are Core's, not the host's. Normalizing
    # here means the envelope names the same extension Core operated on, and a
    # host-side rename can never make an installed set unaddressable in Core's
    # own terms.
    try:
        normalized = manager["normalize"](requested)
    except Exception as e:
        logger.warning(
            "flyto-core normalize_extension_name failed: %s", type(e).__name__,
        )
        return _extension_envelope(
            operation,
            code=EXTENSION_CODE_CORE_ERROR,
            name=requested,
            install_enabled=True,
        )
    normalized = _safe_extension_token(normalized, _SAFE_EXTENSION_NAME)
    if normalized is None:
        return _extension_envelope(
            operation,
            code=EXTENSION_CODE_INVALID_RESULT,
            name=requested,
            install_enabled=True,
        )

    kwargs: Dict[str, Any] = {"name": normalized}
    if operation == "install":
        kwargs["version"] = safe_version
        kwargs["upgrade"] = upgrade

    raw = await _call_core_extension(operation, manager, **kwargs)
    if raw is None:
        return _extension_envelope(
            operation,
            code=EXTENSION_CODE_CORE_ERROR,
            name=normalized,
            install_enabled=True,
        )

    data = _extension_result_mapping(raw)
    if data is None:
        logger.warning("flyto-core ExtensionResult rejected: unreadable shape")
        return _extension_envelope(
            operation,
            code=EXTENSION_CODE_INVALID_RESULT,
            name=normalized,
            install_enabled=True,
        )

    # ExtensionResult owns one exact success field.  The general Core module
    # helper also accepts ``status == "success"``, but carrying that alias into
    # this security-sensitive host adapter would let an extra mapping key
    # override Core's explicit ``ok=False``.
    ok = data.get(_EXTENSION_RESULT_OK_FIELD) is True
    reported = _safe_extension_token(data.get("name"), _SAFE_EXTENSION_NAME)
    reported_kind = _safe_extension_token(
        data.get("kind"), _SAFE_EXTENSION_KIND, lower=True,
    )
    reported_version = _safe_extension_token(
        data.get("version"), _SAFE_EXTENSION_VERSION,
    )
    previous_version = _safe_extension_token(
        data.get("previous_version"), _SAFE_EXTENSION_VERSION,
    )
    # The three flags are read under Core's own names and default to False, so
    # a Core that stops reporting one degrades to "no consequence claimed"
    # rather than to a truthy value. `install_enabled` is this host's opt-in
    # state; Core does not report it and nothing here pretends it does.
    return _extension_envelope(
        operation,
        ok=ok,
        code=_core_extension_code(
            data, EXTENSION_CODE_OK if ok else EXTENSION_CODE_CORE_ERROR,
        ),
        name=reported or normalized,
        kind=reported_kind or "",
        version=reported_version or "",
        previous_version=previous_version or "",
        install_enabled=True,
        restart_required=data.get("restart_required") is True,
        rolled_back=data.get("rolled_back") is True,
        refresh_failed=data.get("refresh_failed") is True,
    )


async def install_core_extension(
    name: str, version: Optional[str] = None, upgrade: bool = False,
) -> Dict[str, Any]:
    """Install one Core extension. Host-only and opt-in; never an LLM tool.

    ``version`` pins a release and ``upgrade`` asks Core to move an already
    installed extension forward. Both are passed to Core's loader; neither is
    interpreted here, and an unsafe value is refused before Core is reached
    rather than forwarded to a package installer.
    """
    return await _mutate_core_extension(
        "install", name, version=version, upgrade=upgrade,
    )


async def uninstall_core_extension(name: str) -> Dict[str, Any]:
    """Uninstall one Core extension. Host-only and opt-in; never an LLM tool.

    Uninstall shares the install opt-in on purpose: both change what the
    installed Core can execute, and a host that may not add capability may not
    silently remove it either.
    """
    return await _mutate_core_extension("uninstall", name)
