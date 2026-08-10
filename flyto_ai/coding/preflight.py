# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Fast, read-only feasibility check run before a job exists at all.

The audited route used to discover that a repository had no verification
contract only once an implementer session was already open: the config was read
inside the implementer, so "this repository never declared how to verify
itself" arrived as ``route_implementation_not_successful`` - a statement about
the change, when nothing about the change was ever at fault.  By then a job
record existed, a workspace claim was held, and a session had burned turns.

This module answers the same question before any of that happens.  It is
deliberately:

*Read-only.*  It opens one bounded YAML file and returns.  It creates no job,
takes no claim, starts no session, and mutates nothing, so a refusal leaves the
service exactly as it was found.

*Fast.*  One stat and one bounded parse, with no subprocess, no network and no
capability handshake, so it is safe to run while the service state guard is
held.

*Typed.*  Every refusal carries a stable code, the phase it happened in, whether
retrying could ever help, and a bounded list of actions from a closed
allowlist.  A caller never has to parse prose to know what to do next.

The vocabulary here is about *repository verification feasibility* and nothing
else.  It names no domain, no product and no lane.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import AbstractSet, Optional, Tuple

from flyto_ai.coding.checks import read_project_contract
from flyto_ai.coding.contracts import CapabilitySpec, CheckSpec
from flyto_ai.coding.checks import (
    unlaunchable_required_checks as _unlaunchable,
)

__all__ = [
    "ACTION_ADD_REPOSITORY_VERIFICATION_CONTRACT",
    "ACTION_DECLARE_REQUIRED_VERIFICATION_CHECK",
    "ACTION_FIX_REPOSITORY_VERIFICATION_CONTRACT",
    "ACTION_INSTALL_REQUIRED_CAPABILITY",
    "ACTION_INSTALL_VERIFICATION_TOOL",
    "CODE_CAPABILITY_UNAVAILABLE",
    "CODE_VERIFICATION_CONTRACT_INVALID",
    "CODE_VERIFICATION_REQUIRED",
    "CODE_VERIFICATION_TOOL_MISSING",
    "FAILURE_PHASE_PREFLIGHT",
    "MAX_PREFLIGHT_BLOCKERS",
    "PREFLIGHT_ACTIONS",
    "PREFLIGHT_CODES",
    "PreflightOutcome",
    "preflight_repository",
    "unlaunchable_required_checks",
]

#: The phase every refusal in this module reports.  It is the earliest phase a
#: caller can observe, and the only one that guarantees no session and no claim.
FAILURE_PHASE_PREFLIGHT = "preflight"

#: No contract at all, or one that declares nothing that could ever verify a
#: change.  The repository has not said how it wants to be checked.
CODE_VERIFICATION_REQUIRED = "verification_required"
#: A contract exists but cannot be honoured as written.  Distinct from the
#: above on purpose: "you never wrote one" and "the one you wrote is wrong" need
#: different work from different people.
CODE_VERIFICATION_CONTRACT_INVALID = "verification_contract_invalid"
#: A contract declares a capability it marks required, and this host cannot
#: attach it.  The contract is fine; the environment is not.
CODE_CAPABILITY_UNAVAILABLE = "capability_unavailable"
#: A required check is declared correctly and cannot be launched, because the
#: program it names is not installed on this host.  Deliberately distinct from
#: both neighbours: the repository's contract is right (unlike
#: `verification_contract_invalid`) and nothing about a capability bridge is
#: involved (unlike `capability_unavailable`).  What is missing is a tool, and
#: the person who fixes it installs software rather than editing YAML.
CODE_VERIFICATION_TOOL_MISSING = "verification_tool_missing"

PREFLIGHT_CODES: Tuple[str, ...] = (
    CODE_CAPABILITY_UNAVAILABLE,
    CODE_VERIFICATION_CONTRACT_INVALID,
    CODE_VERIFICATION_REQUIRED,
    CODE_VERIFICATION_TOOL_MISSING,
)

#: How many check names a refusal will name.  A contract can declare more; the
#: list is evidence for an operator, not a full inventory.
MAX_PREFLIGHT_BLOCKERS = 8

ACTION_ADD_REPOSITORY_VERIFICATION_CONTRACT = "add_repository_verification_contract"
ACTION_DECLARE_REQUIRED_VERIFICATION_CHECK = "declare_required_verification_check"
ACTION_FIX_REPOSITORY_VERIFICATION_CONTRACT = "fix_repository_verification_contract"
ACTION_INSTALL_REQUIRED_CAPABILITY = "install_required_capability"
ACTION_INSTALL_VERIFICATION_TOOL = "install_verification_tool"

#: The closed allowlist.  A required action that is not in here is a bug, not a
#: new message: callers are expected to branch on these tokens.
PREFLIGHT_ACTIONS: Tuple[str, ...] = (
    ACTION_ADD_REPOSITORY_VERIFICATION_CONTRACT,
    ACTION_DECLARE_REQUIRED_VERIFICATION_CHECK,
    ACTION_FIX_REPOSITORY_VERIFICATION_CONTRACT,
    ACTION_INSTALL_REQUIRED_CAPABILITY,
    ACTION_INSTALL_VERIFICATION_TOOL,
)


@dataclass(frozen=True)
class PreflightOutcome:
    """Whether a repository can be verified, and what to do when it cannot."""

    ok: bool
    checks: Tuple[CheckSpec, ...] = ()
    capabilities: Tuple[CapabilitySpec, ...] = ()
    code: str = ""
    required_actions: Tuple[str, ...] = ()
    #: Which declared things caused this refusal, by name.  Check names are
    #: already validated safe identifiers from the contract, so they carry no
    #: path, no argv and no prose - an operator learns *which* check to install
    #: a tool for without the refusal becoming a text channel.
    blockers: Tuple[str, ...] = ()
    #: SHA-256 of the exact contract bytes this outcome was decided from. The
    #: caller pins it for the job's lifetime, so later rounds are authorized
    #: against the document preflight actually validated.
    config_sha256: str = ""

    def __post_init__(self) -> None:
        if self.ok:
            if self.code or self.required_actions or self.blockers:
                raise ValueError("a successful preflight carries no refusal")
            return
        if self.code not in PREFLIGHT_CODES:
            raise ValueError("preflight code must come from the closed set")
        if not self.required_actions:
            raise ValueError("a preflight refusal must name at least one action")
        for action in self.required_actions:
            if action not in PREFLIGHT_ACTIONS:
                raise ValueError("preflight action must come from the closed allowlist")
        if len(self.blockers) > MAX_PREFLIGHT_BLOCKERS:
            raise ValueError("preflight blockers must stay bounded")


def preflight_repository(
    workspace: str,
    relative_path: str = ".flyto/coding.yaml",
    attachable_capability_kinds: Optional[AbstractSet[str]] = None,
) -> PreflightOutcome:
    """Decide whether this repository has declared a usable way to verify itself.

    Missing and empty are both :data:`CODE_VERIFICATION_REQUIRED`: a contract
    that declares no required check cannot certify anything, so accepting the
    job would only defer the same refusal until after a session had run.
    Unparseable or out-of-bounds is :data:`CODE_VERIFICATION_CONTRACT_INVALID`.

    Fails closed: any unexpected error reading the contract is reported as an
    invalid contract rather than waved through, because a repository whose
    verification contract cannot be read is exactly the case where proceeding
    would produce an unverifiable change.
    """

    try:
        contract = read_project_contract(workspace, relative_path)
        checks, capabilities = contract.checks, contract.capabilities
    except (ValueError, OSError) as exc:  # bounded parse/IO refusal
        del exc  # the message may quote file content; the code is the contract
        return PreflightOutcome(
            ok=False,
            code=CODE_VERIFICATION_CONTRACT_INVALID,
            required_actions=(ACTION_FIX_REPOSITORY_VERIFICATION_CONTRACT,),
        )
    except Exception:  # pragma: no cover - defensive: never pass an unknown read
        return PreflightOutcome(
            ok=False,
            code=CODE_VERIFICATION_CONTRACT_INVALID,
            required_actions=(ACTION_FIX_REPOSITORY_VERIFICATION_CONTRACT,),
        )

    if not checks:
        # No file, or a file with no `checks:` at all.  Both mean the same
        # thing to a caller: this repository has not declared verification.
        return PreflightOutcome(
            ok=False,
            code=CODE_VERIFICATION_REQUIRED,
            required_actions=(ACTION_ADD_REPOSITORY_VERIFICATION_CONTRACT,),
        )
    if not any(check.required for check in checks):
        # Declared, but nothing that can fail a round.  Naming the narrower
        # action keeps "add a contract" honest for the repositories that really
        # have none.
        return PreflightOutcome(
            ok=False,
            code=CODE_VERIFICATION_REQUIRED,
            required_actions=(ACTION_DECLARE_REQUIRED_VERIFICATION_CHECK,),
        )
    if _infeasible_required_capability(capabilities, attachable_capability_kinds):
        return PreflightOutcome(
            ok=False,
            code=CODE_CAPABILITY_UNAVAILABLE,
            required_actions=(ACTION_INSTALL_REQUIRED_CAPABILITY,),
        )
    unlaunchable = unlaunchable_required_checks(checks)
    if unlaunchable:
        # Proving the check *exists* was never the same as proving it can run.
        # A required check whose program is not installed is a defect of the
        # environment that is fully decidable right here, and letting it through
        # spends a session, a claim and a provider call only to report it as a
        # failed verification - a statement about the change, when nothing about
        # the change was ever at fault.
        return PreflightOutcome(
            ok=False,
            code=CODE_VERIFICATION_TOOL_MISSING,
            required_actions=(ACTION_INSTALL_VERIFICATION_TOOL,),
            blockers=unlaunchable,
        )
    return PreflightOutcome(
        ok=True,
        checks=checks,
        capabilities=capabilities,
        config_sha256=contract.digest,
    )


def unlaunchable_required_checks(checks: Tuple[CheckSpec, ...]) -> Tuple[str, ...]:
    """Name the required checks whose program cannot be launched on this host.

    Uses :func:`flyto_ai.coding.workspace.resolve_executable` - the same
    function the check runner calls on the path that really starts the process,
    not a reimplementation of it. That is the whole mechanism preventing the two
    from disagreeing: if the runner's resolution ever changes, this changes with
    it, because there is only one of them.

    Optional checks are ignored on purpose. They cannot fail a round, so a host
    without their tooling is merely less thorough, not unable to verify. Nothing
    is executed: a required check that exits non-zero may be reporting the very
    defect the task exists to fix.
    """

    return _unlaunchable(checks)[:MAX_PREFLIGHT_BLOCKERS]


def _infeasible_required_capability(
    capabilities: Tuple[CapabilitySpec, ...],
    attachable_capability_kinds: Optional[AbstractSet[str]],
) -> bool:
    """Whether a required capability cannot be attached *by the selected backend*.

    Two independent reasons, and both must be checked, because passing
    preflight has to mean the round will really get the capability - not merely
    that something by that name is installed somewhere:

    *The backend cannot bridge that kind at all.*  Implementers differ: one
    drives a real capability manager, another has no bridge and refuses every
    required capability the moment it reads the contract.  A preflight that
    ignored this would hand back a green light for a job already guaranteed to
    die, which is precisely the late failure preflight exists to remove.
    ``attachable_capability_kinds`` is the caller's declaration of what its
    selected backend can bridge; ``None`` means *unproven*.

    *The executable cannot be resolved.*  Cheap, necessary, and not sufficient:
    it performs no handshake and cannot tell a working server from a broken
    one.  A capability that resolves here and still fails to attach fails
    closed in the implementer exactly as before.

    Unproven fails closed.  A host that has not declared what its backend can
    bridge is treated as able to bridge nothing, because the alternative -
    assuming it can - is what produced a green preflight in front of a
    guaranteed refusal.
    """

    attachable = frozenset(attachable_capability_kinds or ())
    for capability in capabilities:
        if not capability.required:
            continue
        if capability.kind not in attachable or not capability.argv:
            return True
        executable = capability.argv[0]
        if os.path.sep in executable or (os.path.altsep and os.path.altsep in executable):
            # An explicit path is checked as a path, never searched on PATH.
            if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
                return True
            continue
        if shutil.which(executable) is None:
            return True
    return False
