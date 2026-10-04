# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Recovery guidance read from declared capability contracts.

When a step fails, a planner may only suggest a way round that the provider
itself declared. This module reads that declaration from the capability
contract a module registered through ``@register_module(contract=...)`` and
turns it into bounded guidance a host can hand to a planner. It contains no
provider knowledge: which capabilities may stand in for a failed one, and which
observation carries the provider's recovery context, are both data in the
contract.

Contract shape (optional ``recovery`` key, flyto-core 2.36+)::

    "recovery": {
        "substitutes": ["<capability id>", ...],  # may replace a failed call
        "context": "<observation id>",            # provider recovery context
    }

``recovery_context`` is read as an alias of ``context``. Older cores reject an
unknown contract key at registration, so on them no stored contract carries
``recovery`` and every lookup here yields "nothing declared"; that is the
feature detection, and :func:`core_supports_recovery` reports it for display.

Nothing here executes, approves or widens anything:

* a substitute is offered only when a module in the same candidate set (the
  same pack, or the same resource's approved capabilities) provides it;
* the failed capability is never offered as its own substitute;
* every offered substitute keeps its own module id, contract and grade, so the
  host still runs its proposal / approval / safe-stop path for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from flyto_ai.permissions import grade_module_contract

RECOVERY_KEY = "recovery"
GUIDANCE_VERSION = "flyto.ai.contract-recovery-guidance.v1"
SOURCE_CONTRACT = "contract"
SOURCE_NONE = "none"

# Grades of a contract the grader could not read; its recovery is not trusted.
_UNTRUSTED_GRADES = frozenset({"missing_contract", "invalid_contract"})

# The registry's bounded identifier grammar (flyto.capability-contract.v1).
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_IDENTIFIER_MAX = 96
_SUBSTITUTES_MAX = 16

_core_support: Optional[bool] = None


@dataclass(frozen=True)
class RecoveryDeclaration:
    """The recovery part of one capability contract, normalized."""

    substitutes: Tuple[str, ...]
    context: str

    def to_dict(self) -> Dict[str, Any]:
        return {"substitutes": list(self.substitutes), "context": self.context}


def _identifier(value: Any) -> str:
    if (
        isinstance(value, str)
        and len(value) <= _IDENTIFIER_MAX
        and _IDENTIFIER.fullmatch(value)
    ):
        return value
    return ""


def declared_recovery(contract: Any) -> Optional[RecoveryDeclaration]:
    """Return the contract's recovery declaration, or ``None``.

    Fails closed: a malformed declaration yields ``None`` rather than a
    partially trusted one, because a substitute list is authority to suggest
    an alternative action.
    """
    if not isinstance(contract, Mapping):
        return None
    # A contract the grader rejects (and so treats as actuating) declares
    # nothing else either: its substitute list is not authority.
    grade = grade_module_contract({"plugin": "-", "contract": contract})
    if grade is None or grade.source in _UNTRUSTED_GRADES:
        return None
    raw = contract.get(RECOVERY_KEY)
    if not isinstance(raw, Mapping):
        return None

    substitutes_raw = raw.get("substitutes", [])
    if not isinstance(substitutes_raw, (list, tuple)):
        return None
    if len(substitutes_raw) > _SUBSTITUTES_MAX:
        return None
    substitutes: List[str] = []
    for item in substitutes_raw:
        identifier = _identifier(item)
        if not identifier:
            return None
        if identifier not in substitutes:
            substitutes.append(identifier)

    context_raw = raw.get("context", raw.get("recovery_context", ""))
    context = ""
    if context_raw not in ("", None):
        context = _identifier(context_raw)
        if not context:
            return None

    if not substitutes and not context:
        return None
    return RecoveryDeclaration(substitutes=tuple(substitutes), context=context)


def core_supports_recovery() -> bool:
    """Whether the installed flyto-core accepts a ``recovery`` contract key.

    Reported for display and diagnostics only; guidance never depends on it,
    because a contract that carries ``recovery`` was already accepted by the
    registry that stored it.
    """
    global _core_support
    if _core_support is not None:
        return _core_support
    supported = False
    try:
        from core import capability_contract  # type: ignore[import-not-found]
    except Exception:
        capability_contract = None
    if capability_contract is not None:
        keys = set()
        for name in ("CONTRACT_KEYS", "_REQUIRED_KEYS", "_OPTIONAL_KEYS"):
            value = getattr(capability_contract, name, None)
            if isinstance(value, (set, frozenset, tuple, list)):
                keys.update(str(item) for item in value)
        supported = RECOVERY_KEY in keys
    _core_support = supported
    return supported


def _module_capability(module: Mapping[str, Any]) -> str:
    capability = module.get("provides_capability")
    return capability if isinstance(capability, str) else ""


def _offer(module: Mapping[str, Any]) -> Dict[str, Any]:
    offer: Dict[str, Any] = {
        "capability": _module_capability(module),
        "module_id": str(module.get("module_id") or ""),
    }
    for key in ("plugin", "tool_name", "grade"):
        if module.get(key) is not None:
            offer[key] = module[key]
    return offer


def recovery_guidance(
    failed_capability: str,
    candidates: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build recovery guidance for one failed capability.

    ``candidates`` are the modules the host already offers for this resource or
    pack (for example the ``modules`` of one pack capability group). Each is a
    mapping with ``provides_capability``, ``module_id`` and ``contract``, plus
    optional ``plugin`` / ``tool_name`` / ``grade`` that are carried through.

    Returns ``{"version", "failed_capability", "source", "substitutes",
    "unavailable", "context"}``. ``source`` is ``"contract"`` when the failed
    capability's contract declared recovery, otherwise ``"none"`` and the
    lists are empty: a host then falls back to its generic continuation and the
    planner is told nothing was declared, never invited to improvise.
    """
    modules = [item for item in candidates if isinstance(item, Mapping)]
    failed = [item for item in modules if _module_capability(item) == failed_capability]

    declaration: Optional[RecoveryDeclaration] = None
    for module in failed:
        declaration = declared_recovery(module.get("contract"))
        if declaration is not None:
            break

    guidance: Dict[str, Any] = {
        "version": GUIDANCE_VERSION,
        "failed_capability": failed_capability,
        "source": SOURCE_NONE,
        "substitutes": [],
        "unavailable": [],
        "context": "",
    }
    if declaration is None:
        return guidance

    guidance["source"] = SOURCE_CONTRACT
    guidance["context"] = declaration.context
    for capability in declaration.substitutes:
        if capability == failed_capability:
            continue
        providers = [
            item for item in modules if _module_capability(item) == capability
        ]
        if not providers:
            guidance["unavailable"].append(capability)
            continue
        guidance["substitutes"].extend(_offer(item) for item in providers)
    return guidance


def render_recovery_guidance(guidance: Mapping[str, Any]) -> str:
    """Render guidance as domain-neutral planner text."""
    failed = str(guidance.get("failed_capability") or "")
    if guidance.get("source") != SOURCE_CONTRACT:
        return (
            "The contract of {} declares no recovery. Do not invent a "
            "substitute; report the failure or ask the operator.".format(failed)
        )
    lines = ["The contract of {} declares this recovery:".format(failed)]
    substitutes = guidance.get("substitutes") or []
    if substitutes:
        lines.append(
            "- It may be replaced only by: {}.".format(
                ", ".join(
                    "{} ({})".format(item.get("capability"), item.get("module_id"))
                    for item in substitutes
                )
            )
        )
        lines.append(
            "- Each replacement still needs its own approval and argument bounds."
        )
    else:
        lines.append("- No declared substitute is installed here.")
    context = guidance.get("context")
    if context:
        lines.append(
            "- The provider reports its recovery context in observation "
            "'{}'; use it as given.".format(context)
        )
    return "\n".join(lines)


__all__ = [
    "GUIDANCE_VERSION",
    "RECOVERY_KEY",
    "RecoveryDeclaration",
    "core_supports_recovery",
    "declared_recovery",
    "recovery_guidance",
    "render_recovery_guidance",
]
