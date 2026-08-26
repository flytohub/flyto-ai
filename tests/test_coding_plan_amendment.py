# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""A multi-round rework is one root task, or it is not auditable at all.

Job `job_1be3e31602264f88b617b42a` took three implementation rounds. The third
passed every host check - compile, lint, generated reference, 3032 tests - and
was then refused by strict Indexer post-work with `unplanned_diff`, surfacing
only as `route_domain_failure`. Two independent defects produced that.

*Every round planned a new root task.* The pre-lane always called
``task(action="plan")`` from the current request. A rework message carries audit
feedback plus the original task, so round three asked the Indexer to authorize a
plan that knew nothing about rounds one and two - and then changed files those
earlier rounds had opened.

*Post-work validated a narrower set than the audit would bind.* The lane
received only ``result.files_changed`` for the last round, while
``CodingService._record_outcome`` later made the revision Codex signs cumulative.
The scope that was validated and the scope that was audited were therefore not
required to be the same set of bytes, which is the whole point of validating.

Underneath both sat a smaller bug with the same shape: the round-three message
contained the audit code ``check.generated_reference``, and explicit-target
extraction read it as a request to create a file called
``check.generated_reference`` in the repository root. A machine identifier is not
edit-path authority.
"""
import asyncio
import copy
import hashlib
import json
import sys

import pytest

from flyto_ai.coding.contracts import (
    CapabilitySpec,
    CodingJobState,
    CodingTaskRequest,
)
from flyto_ai.coding.amendment_contract import (
    amendment_contract_id,
    chain_entry,
    parent_contract_digest,
    root_contract_id,
)
from flyto_ai.coding.route import (
    CodingRouteOrchestrator,
    CodingRoutePolicy,
    RouteLimits,
)

# ──────────────────────────────────────────────────────────────────────
# explicit targets: a dotted identifier is not a path
# ──────────────────────────────────────────────────────────────────────


def _targets(message, workspace):
    return CodingRouteOrchestrator._explicit_request_targets(message, str(workspace))


def _amendment_targets(message, workspace):
    return CodingRouteOrchestrator._explicit_amendment_targets(
        message, str(workspace),
    )


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "docs" / "reference" / "python").mkdir(parents=True)
    (root / "app" / "(tabs)").mkdir(parents=True)
    (root / "app" / "[id]").mkdir(parents=True)
    (root / "pkg").mkdir()
    (root / "artifacts").mkdir()
    (root / "tools").mkdir()
    (root / "README.md").write_text("# root\n", encoding="utf-8")
    (root / "docs" / "reference" / "python" / "README.md").write_text("x\n", encoding="utf-8")
    (root / "docs" / "reference" / "python" / "coding.md").write_text("y\n", encoding="utf-8")
    (root / "artifacts" / "result.txt").write_text("result\n", encoding="utf-8")
    (root / "tools" / "builder.py").write_text("pass\n", encoding="utf-8")
    return root


@pytest.mark.parametrize(
    "identifier",
    [
        "check.generated_reference",
        "check.some_capability",
        "pkg/check.some_capability",
        "human.approval",
        "module.identifier",
        "finding.unplanned_diff",
        "fix_intent_ledger:task",
        "evidence.run_a91ed5aff05804aa5f7e",
    ],
)
def test_a_dotted_machine_identifier_is_never_an_edit_target(repo, identifier):
    """The production trigger, and its whole family.

    These all parse as "a name with a suffix" under the conservative path
    grammar, which is exactly why a suffix being non-empty was never enough.
    """

    message = "rework: the required check {} failed, please repair".format(identifier)
    assert _targets(message, repo) == []
    assert not (repo / identifier).exists(), "nothing was created by parsing"


def test_real_paths_and_typed_new_filenames_still_resolve(repo):
    """The rule may not cost a single legitimate target."""

    assert _targets("update docs/reference/python/coding.md", repo) == [
        "docs/reference/python/coding.md",
    ]
    # Exact file authority survives two missing parent levels without granting
    # either directory as a target.
    assert _targets("add tests/contracts/test_new_thing.py", repo) == [
        "tests/contracts/test_new_thing.py",
    ]
    assert _targets("add tests/test_new_thing.py", repo) == ["tests/test_new_thing.py"]
    assert _targets("create artifacts/archive.7z", repo) == [
        "artifacts/archive.7z",
    ]
    # Existing root file, bare.
    assert _targets("fix README.md please.", repo) == ["README.md"]
    # Expo route segments survive.
    (repo / "app" / "(tabs)" / "index.tsx").write_text("x", encoding="utf-8")
    assert _targets("edit app/(tabs)/index.tsx", repo) == ["app/(tabs)/index.tsx"]
    assert _targets("create app/[id]/page.tsx", repo) == ["app/[id]/page.tsx"]


def test_the_exact_code_rework_does_not_treat_milestone_m1_1_as_a_new_file(repo):
    """A milestone label is prose, even when a mutation verb precedes it.

    The live Code repair starts its original task with ``Implement M1.1``.
    ``.1`` used to satisfy the new-file suffix grammar, so the amendment sent
    a nonexistent root file named ``M1.1`` to the Indexer and was refused
    before the provider could run.
    """

    message = (
        "Audit verdict: rework. Resolve every finding below in this same thread.\n"
        "1. [blocker] focused_sidebar_suite_does_not_execute: repair the "
        "allowed Sidebar import and rerun both new tests.\n\n"
        "Original task:\n"
        "Implement M1.1 module-state coherence on exact base e233a0b3. "
        "Allowed product files: src-next/hooks/useEffectivePageAccess.ts."
    )
    assert "M1.1" in message
    assert _targets(message, repo) == []
    assert _amendment_targets(message, repo) == []


def test_the_target_bound_still_holds(repo):
    (repo / "many").mkdir()
    names = []
    for index in range(80):
        name = "many/f{:03d}.py".format(index)
        (repo / name).write_text("x", encoding="utf-8")
        names.append(name)
    found = _targets(" ".join(names), repo)
    assert len(found) == 64
    assert found == names[:64]


def test_rework_evidence_and_command_only_paths_grant_no_authority(repo):
    message = (
        "Run `python3 tools/builder.py`, then attach artifacts/result.txt as "
        "evidence."
    )
    assert _targets(message, repo) == [
        "tools/builder.py", "artifacts/result.txt",
    ]
    assert _amendment_targets(message, repo) == []


def test_rework_explicitly_modifying_a_command_program_grants_authority(repo):
    assert _amendment_targets(
        "Modify tools/builder.py, then run its canonical command.", repo,
    ) == ["tools/builder.py"]


@pytest.mark.parametrize(
    "syntax",
    [
        "using",
        "via",
        "use",
        "call",
        "calling",
        "run",
        "execute",
        "invoke",
        "with",
        "through",
        "by",
        "by running",
        "by executing",
        "by invoking",
    ],
)
def test_rework_output_through_program_authorizes_only_output(repo, syntax):
    message = "Regenerate artifacts/result.txt {} tools/builder.py".format(syntax)
    assert _amendment_targets(message, repo) == ["artifacts/result.txt"]


@pytest.mark.parametrize(
    "syntax",
    [
        "using", "via", "use", "call", "calling", "run", "execute",
        "invoke", "with", "through", "by", "by running", "by executing",
        "by invoking",
    ],
)
def test_rework_updated_output_through_program_authorizes_only_output(repo, syntax):
    message = "Update artifacts/result.txt {} tools/builder.py".format(syntax)
    assert _amendment_targets(message, repo) == ["artifacts/result.txt"]


def test_rework_explicitly_modifying_output_and_program_authorizes_both(repo):
    assert _amendment_targets(
        "Modify artifacts/result.txt and tools/builder.py.", repo,
    ) == ["artifacts/result.txt", "tools/builder.py"]


@pytest.mark.parametrize("mutation", ["modifying", "editing", "updating", "changing"])
def test_rework_by_explicit_program_mutation_authorizes_program(repo, mutation):
    message = "Regenerate artifacts/result.txt by {} tools/builder.py".format(mutation)
    assert _amendment_targets(message, repo) == [
        "artifacts/result.txt", "tools/builder.py",
    ]


def test_rework_regenerate_live_form_authorizes_the_tracked_output(repo):
    message = (
        "Regenerate and include that tracked generated target "
        "artifacts/result.txt using the repository canonical action"
    )
    assert _amendment_targets(message, repo) == ["artifacts/result.txt"]


def test_rework_fix_by_running_keeps_the_command_program_excluded(repo):
    assert _amendment_targets(
        "Fix tracked outputs by running tools/builder.py", repo,
    ) == []
    assert _amendment_targets(
        "tools/builder.py must be repaired.", repo,
    ) == ["tools/builder.py"]


def test_rework_polarity_and_normal_positive_targets_remain_intact(repo):
    message = (
        "Do not modify tools/builder.py; repair artifacts/result.txt."
    )
    assert _amendment_targets(message, repo) == ["artifacts/result.txt"]
    assert _amendment_targets("modify ../../outside.py", repo) == []


# ──────────────────────────────────────────────────────────────────────
# a faithful fake of the declared Indexer capability contract
# ──────────────────────────────────────────────────────────────────────


class FakeIndexer:
    """Implements the declared `task_contract` amendment contract.

    Faithful to what the sibling package declares: absent argument means a fresh
    root plan; a valid parent means a same-root cumulative amendment whose intent
    ledger is the union of old and new paths, chained and integrity checked.
    Nothing here is the real Indexer - the live proof is Codex's - but every
    assertion below is about what this host *sends* and *does*, which is exactly
    what the defect was about.
    """

    ROOT = "task_root_0001"

    def __init__(self):
        self.plans = []
        self.validations = []

    @staticmethod
    def _chain(paths):
        return hashlib.sha256(
            json.dumps(sorted(paths), sort_keys=True).encode("utf-8"),
        ).hexdigest()

    async def __call__(self, tool, arguments):
        arguments = dict(arguments or {})
        action = arguments.get("action")
        if tool == "structure":
            return {"ok": True, "files": []}
        if tool in ("search", "list_blueprints", "search_blueprints"):
            return {"ok": True, "results": []}
        if tool == "impact":
            return {"ok": True, "result": {}}
        if tool == "verify":
            return {"ok": True, "pass": True}
        if tool == "task" and action == "plan":
            self.plans.append(arguments)
            parent = arguments.get("task_contract")
            targets = [str(item) for item in (arguments.get("targets") or [])]
            if isinstance(parent, dict) and parent:
                original = list(parent["intent_ledger"]["allowed_paths"])
                ledger = list(dict.fromkeys(original + targets))
                root = parent["root_task_id"]
                objective = parent["objective"]
                generation = int(parent["generation"]) + 1
            else:
                original = []
                ledger = list(dict.fromkeys(targets))
                root = self.ROOT
                objective = str(arguments.get("description", ""))[:200]
                generation = 1
            fingerprint = self._chain(ledger)
            contract = {
                "ok": True,
                "root_task_id": root,
                "objective": objective,
                "generation": generation,
                "intent_ledger": {
                    "version": "intent-ledger.v1",
                    "fingerprint": fingerprint,
                    "description": objective,
                    "allowed_paths": ledger,
                },
                "instruction_context": {
                    "version": "task-context.v1",
                    "fingerprint": self._chain(["instructions"] + ledger),
                },
                "chain_sha256": fingerprint,
                "task_profile": {
                    "kind": "generic",
                    "version": "task-contract.v2",
                    "task_id": root,
                    "intent": "feature",
                    "project": arguments.get("project"),
                    "description": objective,
                    "intent_fingerprint": fingerprint,
                    "instruction_fingerprint": self._chain(["instructions"] + ledger),
                    "targets": ledger,
                    "resolved_targets": [
                        {"input": item, "path": item, "symbol_id": None}
                        for item in ledger
                    ],
                },
                "targets": ledger,
                "resolved_targets": [
                    {"input": item, "path": item, "symbol_id": None}
                    for item in ledger
                ],
                "execution_plan": [],
            }
            if isinstance(parent, dict) and parent:
                parent_state = parent.get("task_amendment") or {}
                amendment_index = int(parent_state.get("amendment_index") or 0) + 1
                parent_digest = (
                    parent_contract_digest(parent)
                )
                parent_contract_id = parent_state.get("contract_id") or (
                    root_contract_id(
                        root, objective, original,
                    )
                )
                contract_id = amendment_contract_id(
                    root,
                    amendment_index,
                    objective,
                    ledger,
                    parent_contract_id,
                    parent_digest,
                )
                parent_chain = list(parent_state.get("chain") or [])
                chain = parent_chain + [
                    chain_entry(
                        contract_id=parent_contract_id,
                        parent_contract_id=(
                            parent_chain[-1]["contract_id"] if parent_chain else None
                        ),
                        root_task_id=root,
                        project=str(arguments.get("project") or ""),
                        amendment_index=amendment_index - 1,
                        target_count=len(original),
                        contract_digest=parent_digest,
                    )
                ]
                contract["task_amendment"] = {
                    "version": "task-amendment.v1",
                    "status": "amended",
                    "root_task_id": root,
                    "objective": objective,
                    "amendment_index": amendment_index,
                    "contract_id": contract_id,
                    "parent_contract_id": parent_contract_id,
                    "parent_contract_digest": parent_digest,
                    "chain": chain,
                    "original_paths": original,
                    "added_paths": [item for item in ledger if item not in original],
                    "cumulative_paths": ledger,
                }
                contract["task_profile"].update({
                    "root_task_id": root,
                    "description": objective,
                    "amendment_index": amendment_index,
                    "amendment_contract_id": contract_id,
                })
            return contract
        if tool == "task" and action == "gate":
            return {"ok": True, "pass": True, "required_state": {}}
        if tool == "task" and action == "validate":
            self.validations.append(arguments)
            contract = arguments.get("task_contract") or {}
            changed = list(
                (arguments.get("current_state") or {}).get("changed_paths") or []
            )
            ledger = set(
                (contract.get("intent_ledger") or {}).get("allowed_paths") or ()
            )
            # Faithful to the cumulative amendment contract in both directions.
            # An undeclared change is drift; an omitted cumulative path means the
            # caller validated a narrower scope than the contract now covers,
            # which is exactly the closure this repair exists for.
            unplanned = sorted(set(changed) - ledger)
            omitted = sorted(ledger - set(changed))
            if unplanned or omitted:
                reasons = []
                if unplanned:
                    reasons.append("fix_intent_ledger:task:unplanned_diff")
                if omitted:
                    reasons.append("fix_intent_ledger:task:incomplete_scope")
                return {
                    "ok": True, "pass": False,
                    "reason_codes": reasons,
                    "required_actions": ["amend_intent_ledger"],
                }
            return {"ok": True, "pass": True}
        return {"ok": True, "result": {}}


def _spec(name, tools=("task", "verify", "structure", "search", "impact")):
    return CapabilitySpec(
        name=name, argv=("python3", "-c", "pass"), required=True,
        required_tools=tools, allowed_tools=tools,
        tool_permissions=tuple(
            (tool, "workspace_write" if tool == "task" else "read_only")
            for tool in tools
        ),
    )


_BLUEPRINT_TOOLS = ("list_blueprints", "search_blueprints")


def _orchestrator(dispatch):
    policy = CodingRoutePolicy(
        strict=False, indexer=_spec("flyto-indexer"), limits=RouteLimits(),
    )
    return CodingRouteOrchestrator(policy, capability_dispatch=dispatch)


def test_a_first_round_sends_no_task_contract_argument(repo):
    """Legacy compatibility is byte-for-byte, not merely 'similar'."""

    indexer = FakeIndexer()
    route = _orchestrator(indexer)
    asyncio.run(route._indexer_pre(
        CodingTaskRequest(message="add docs/reference/python/coding.md", working_dir=str(repo)),
    ))
    assert len(indexer.plans) == 1
    assert "task_contract" not in indexer.plans[0]
    assert set(indexer.plans[0]) == {
        "action", "description", "targets", "intent", "project",
    }


def test_a_rework_amends_the_exact_parent_and_keeps_the_root(repo):
    indexer = FakeIndexer()
    route = _orchestrator(indexer)
    request = CodingTaskRequest(
        message="edit docs/reference/python/coding.md", working_dir=str(repo),
    )
    _, first = asyncio.run(route._indexer_pre(request))
    parent = first["task_contract"]

    second_request = CodingTaskRequest(
        message="rework: also edit docs/reference/python/README.md",
        working_dir=str(repo),
    )
    _, second = asyncio.run(route._indexer_pre(second_request, parent))
    amended = second["task_contract"]

    assert indexer.plans[1]["task_contract"] == dict(parent)
    assert amended["root_task_id"] == parent["root_task_id"] == FakeIndexer.ROOT
    assert amended["objective"] == parent["objective"], "the root task moved"
    assert amended["generation"] == parent["generation"] + 1
    assert set(amended["intent_ledger"]["allowed_paths"]) == {
        "docs/reference/python/coding.md", "docs/reference/python/README.md",
    }
    assert amended["chain_sha256"] != parent["chain_sha256"]


def test_a_rework_plan_carries_host_proven_prior_scope(repo):
    """Audit prose cannot silently narrow paths the same job already owns."""

    indexer = FakeIndexer()
    route = _orchestrator(indexer)
    request = CodingTaskRequest(
        message="edit docs/reference/python/coding.md", working_dir=str(repo),
    )
    _, first = asyncio.run(route._indexer_pre(request))

    second_request = CodingTaskRequest(
        message="rework: also edit docs/reference/python/README.md",
        working_dir=str(repo),
    )
    _, second = asyncio.run(route._indexer_pre(
        second_request,
        first["task_contract"],
        ("README.md", "docs/reference/python/coding.md"),
    ))

    assert indexer.plans[-1]["targets"] == [
        "README.md", "docs/reference/python/README.md",
    ]
    assert set(second["task_contract"]["intent_ledger"]["allowed_paths"]) == {
        "README.md",
        "docs/reference/python/coding.md",
        "docs/reference/python/README.md",
    }


def test_parent_ledger_authority_survives_a_touched_scope_subset(repo):
    """Touched files cannot narrow the authenticated parent plan ledger."""

    indexer = FakeIndexer()
    route = _orchestrator(indexer)
    request = CodingTaskRequest(
        message=(
            "edit docs/reference/python/coding.md and "
            "docs/reference/python/README.md"
        ),
        working_dir=str(repo),
    )
    _, first = asyncio.run(route._indexer_pre(request))
    parent = first["task_contract"]
    assert parent["intent_ledger"]["allowed_paths"] == [
        "docs/reference/python/coding.md",
        "docs/reference/python/README.md",
    ]

    rework = CodingTaskRequest(
        message="rework: also edit artifacts/result.txt",
        working_dir=str(repo),
    )
    _, second = asyncio.run(route._indexer_pre(
        rework, parent,
        ("docs/reference/python/coding.md", "README.md"),
    ))

    assert indexer.plans[-1]["targets"] == [
        "README.md", "artifacts/result.txt",
    ]
    assert second["task_contract"]["intent_ledger"]["allowed_paths"] == [
        "docs/reference/python/coding.md",
        "docs/reference/python/README.md",
        "README.md",
        "artifacts/result.txt",
    ]


def test_audited_prior_scope_outside_parent_ledger_is_amendment_authority(repo):
    indexer = FakeIndexer()
    route = _orchestrator(indexer)
    request = CodingTaskRequest(
        message="edit docs/reference/python/coding.md", working_dir=str(repo),
    )
    _, first = asyncio.run(route._indexer_pre(request))
    rework = CodingTaskRequest(
        message="rework: also edit artifacts/result.txt", working_dir=str(repo),
    )

    _, second = asyncio.run(route._indexer_pre(
        rework, first["task_contract"], ("README.md",),
    ))
    assert second["task_contract"]["intent_ledger"]["allowed_paths"] == [
        "docs/reference/python/coding.md", "README.md", "artifacts/result.txt",
    ]


def test_large_parent_scope_is_not_redeclared_as_one_amendment(repo):
    """The per-amendment 32-target bound is not a cumulative-scope bound."""

    many = repo / "many"
    many.mkdir()
    paths = []
    for index in range(41):
        path = "many/f{:02d}.py".format(index)
        (repo / path).write_text("x\n", encoding="utf-8")
        paths.append(path)
    indexer = FakeIndexer()
    route = _orchestrator(indexer)
    _, first = asyncio.run(route._indexer_pre(CodingTaskRequest(
        message="edit exactly these files: " + " ".join(paths),
        working_dir=str(repo),
    )))
    _, second = asyncio.run(route._indexer_pre(
        CodingTaskRequest(
            message="rework: update many/f00.py to add the regression",
            working_dir=str(repo),
        ),
        first["task_contract"],
        tuple(paths),
    ))

    assert indexer.plans[-1]["targets"] == ["many/f00.py"]
    assert second["task_contract"]["intent_ledger"]["allowed_paths"] == paths


def test_same_scope_rework_redeclares_one_authenticated_parent_target(repo):
    indexer = FakeIndexer()
    route = _orchestrator(indexer)
    _, first = asyncio.run(route._indexer_pre(CodingTaskRequest(
        message="edit docs/reference/python/coding.md",
        working_dir=str(repo),
    )))
    _, second = asyncio.run(route._indexer_pre(
        CodingTaskRequest(message="rework: add coverage", working_dir=str(repo)),
        first["task_contract"],
        ("docs/reference/python/coding.md",),
    ))

    assert indexer.plans[-1]["targets"] == [
        "docs/reference/python/coding.md",
    ]
    assert second["task_contract"]["intent_ledger"]["allowed_paths"] == [
        "docs/reference/python/coding.md",
    ]


def _synthetic_plan_step(identifier, *, purpose, target="", gate=False, search=False):
    if gate:
        return {
            "id": identifier,
            "tool": "task",
            "args": {"action": "gate", "next_phase": purpose},
            "purpose": "gate_{}".format(purpose),
            "required": True,
            "depends_on": [],
        }
    return {
        "id": identifier,
        "tool": "search" if search else "impact",
        "args": (
            {"query": "tests covering {}".format(target)}
            if search else {"target": target, "change_type": "modify"}
        ),
        "purpose": purpose,
        "required": True,
        "depends_on": [],
    }


def _synthetic_plan_group(prefix, original_count, added_count):
    steps = [
        _synthetic_plan_step(
            "{}_o{:02d}".format(prefix, index),
            purpose="{}_original_{:02d}".format(prefix, index),
            target="repo:orig/a.py:file:a",
            search=index == 0,
        )
        for index in range(original_count)
    ] + [
        _synthetic_plan_step(
            "{}_a{:02d}".format(prefix, index),
            purpose="{}_added_{:02d}".format(prefix, index),
            target="add/a.py" if index == 0 else "repo:add/a.py:file:a",
            search=index == 0,
        )
        for index in range(added_count)
    ]
    steps.extend([
        _synthetic_plan_step("{}_g1".format(prefix), purpose="assess", gate=True),
        _synthetic_plan_step("{}_g2".format(prefix), purpose="implement", gate=True),
    ])
    return steps


def _bind_successor_amendment(parent, successor):
    """Attach the exact public Indexer content-addressed successor chain."""

    amendment = successor["task_amendment"]
    parent_state = parent.get("task_amendment") or {}
    parent_index = int(parent_state.get("amendment_index") or 0)
    objective = parent["intent_ledger"]["description"]
    original = list(amendment["original_paths"])
    cumulative = list(amendment["cumulative_paths"])
    root = amendment["root_task_id"]
    digest = parent_contract_digest(parent)
    parent_id = parent_state.get("contract_id") or root_contract_id(
        root, objective, original,
    )
    contract_id = amendment_contract_id(
        root, parent_index + 1, objective, cumulative, parent_id, digest,
    )
    parent_chain = list(parent_state.get("chain") or [])
    amendment.update({
        "objective": objective,
        "amendment_index": parent_index + 1,
        "contract_id": contract_id,
        "parent_contract_id": parent_id,
        "parent_contract_digest": digest,
        "chain": parent_chain + [
            chain_entry(
                contract_id=parent_id,
                parent_contract_id=(
                    parent_chain[-1]["contract_id"] if parent_chain else None
                ),
                root_task_id=root,
                project=parent["task_profile"]["project"],
                amendment_index=parent_index,
                target_count=len(original),
                contract_digest=digest,
            )
        ],
    })
    successor["task_profile"].update({
        "root_task_id": root,
        "description": objective,
        "amendment_index": parent_index + 1,
        "amendment_contract_id": contract_id,
    })


def _promote_parent_to_generation_one(parent):
    """Give a root fixture one valid first-generation public ancestry."""

    ancestor = copy.deepcopy(parent)
    ancestor.pop("task_amendment", None)
    root = parent["task_profile"]["task_id"]
    project = parent["task_profile"]["project"]
    objective = parent["intent_ledger"]["description"]
    paths = list(parent["intent_ledger"]["allowed_paths"])
    digest = parent_contract_digest(ancestor)
    predecessor = root_contract_id(root, objective, paths)
    contract_id = amendment_contract_id(
        root, 1, objective, paths, predecessor, digest,
    )
    parent["task_amendment"] = {
        "version": "task-amendment.v1", "status": "amended",
        "root_task_id": root, "objective": objective, "amendment_index": 1,
        "contract_id": contract_id, "parent_contract_id": predecessor,
        "parent_contract_digest": digest,
        "chain": [
            chain_entry(
                contract_id=predecessor, parent_contract_id=None,
                root_task_id=root, project=project, amendment_index=0,
                target_count=len(paths), contract_digest=digest,
            )
        ],
        "original_paths": paths, "added_paths": [], "cumulative_paths": paths,
    }
    parent["task_profile"].update({
        "root_task_id": root,
        "description": objective,
        "amendment_index": 1,
        "amendment_contract_id": contract_id,
    })


def _contract_section(*, task_id, project, paths, fingerprint, steps):
    """Build one pinned-v2 root contract section for amendment tests."""

    objective = "Implement the exact amendment test contract"
    return {
        "task_profile": {
            "version": "task-contract.v2", "task_id": task_id,
            "intent": "feature", "project": project,
            "description": objective,
            "intent_fingerprint": fingerprint,
            "instruction_fingerprint": "9" * 64,
        },
        "intent_ledger": {
            "version": "intent-ledger.v1", "fingerprint": fingerprint,
            "description": objective, "allowed_paths": list(paths),
        },
        "instruction_context": {
            "version": "task-context.v1", "fingerprint": "9" * 64,
        },
        "execution_plan": list(steps),
    }


def test_amendment_executes_only_parent_proven_delta_without_relaxing_bound(repo):
    """The live Engine 18 + 16 successor has a bounded 22-step delta.

    The Indexer-issued digest-bound amendment boundary, rather than fuzzy step similarity,
    says which paths are original and which were added. Analysis tied exactly
    to the six original paths was completed by the successful parent pre-lane.
    Gates are never reused: the successor contract must pass all four of its
    own scoped gates.
    """

    from flyto_ai.coding.route import CodingRouteError, RouteLane

    parent = _contract_section(
        task_id="task_feature_engine",
        project=repo.name,
        paths=["orig/a.py"],
        fingerprint="1" * 64,
        steps=[],
    )
    parent.update({
        "sub_tasks": [
            {"execution_plan": _synthetic_plan_group("parent_cleanup", 4, 0)},
            {"execution_plan": _synthetic_plan_group("parent_feature", 10, 0)},
        ],
    })
    successor = {
        "task_profile": {
            "version": "task-contract.v2", "task_id": "task_feature_engine",
            "intent": "feature", "project": repo.name,
            "intent_fingerprint": "2" * 64,
            "instruction_fingerprint": "8" * 64,
        },
        "intent_ledger": {
            "version": "intent-ledger.v1", "fingerprint": "2" * 64,
            "description": parent["intent_ledger"]["description"],
            "allowed_paths": ["orig/a.py", "add/a.py"],
        },
        "instruction_context": {
            "version": "task-context.v1", "fingerprint": "8" * 64,
        },
        "task_amendment": {
            "version": "task-amendment.v1",
            "status": "amended",
            "root_task_id": "task_feature_engine",
            "amendment_index": 1,
            "parent_contract_digest": "",
            "original_paths": ["orig/a.py"],
            "added_paths": ["add/a.py"],
            "cumulative_paths": ["orig/a.py", "add/a.py"],
        },
        "sub_tasks": [
            {
                "targets": ["orig/a.py", "add/a.py"],
                "resolved_targets": [
                    {"input": "orig/a.py", "path": "orig/a.py",
                     "symbol_id": "repo:orig/a.py:file:a"},
                    {"input": "add/a.py", "path": "add/a.py",
                     "symbol_id": "repo:add/a.py:file:a"},
                ],
                "execution_plan": _synthetic_plan_group("parent_cleanup", 3, 13),
            },
            {
                "targets": ["orig/a.py", "add/a.py"],
                "resolved_targets": [
                    {"input": "orig/a.py", "path": "orig/a.py",
                     "symbol_id": "repo:orig/a.py:file:a"},
                    {"input": "add/a.py", "path": "add/a.py",
                     "symbol_id": "repo:add/a.py:file:a"},
                ],
                "execution_plan": _synthetic_plan_group("parent_feature", 9, 5),
            },
        ],
    }
    route = _orchestrator(FakeIndexer())
    _bind_successor_amendment(parent, successor)
    with pytest.raises(CodingRouteError, match="plan_bound_exceeded"):
        route._plan_groups(successor, RouteLane.INDEXER_PRE)

    groups = route._plan_groups(
        successor, RouteLane.INDEXER_PRE, parent_contract=parent,
        host_project=repo.name,
        host_requested_paths=successor["task_amendment"]["added_paths"],
    )
    delta = [step for _scope, steps in groups for step in steps]
    assert sum(
        len(item["execution_plan"]) for item in successor["sub_tasks"]
    ) == 34
    assert len(delta) == 22 <= RouteLimits().max_plan_steps
    assert sum(step["tool"] == "task" for step in delta) == 4
    assert not any("_original_" in step["purpose"] for step in delta)
    assert sum("_added_" in step["purpose"] for step in delta) == 18


def test_amendment_executes_a_novel_original_path_step_not_proven_by_parent(repo):
    """Path ownership alone cannot erase new successor analysis."""

    from flyto_ai.coding.route import RouteLane

    route, parent, successor = _digest_bound_amendment_fixture(repo)
    proven = copy.deepcopy(parent["execution_plan"][0])
    novel = {
        "id": "successor_novel_original_callers",
        "tool": "call_hierarchy",
        "args": {"symbol_id": "repo:orig/a.py:file:a", "direction": "callers"},
        "purpose": "inspect_new_original_callers",
        "required": True,
        "depends_on": [],
    }
    added = _synthetic_plan_step(
        "successor_added", purpose="inspect_added", target="add/a.py", search=True,
    )
    gates = [
        _synthetic_plan_step("successor_g1", purpose="assess", gate=True),
        _synthetic_plan_step("successor_g2", purpose="implement", gate=True),
    ]
    successor["execution_plan"] = [proven, novel, added] + gates

    groups = route._plan_groups(
        successor, RouteLane.INDEXER_PRE, parent_contract=parent,
        host_project=repo.name,
        host_requested_paths=successor["task_amendment"]["added_paths"],
    )
    delta = [step for _scope, steps in groups for step in steps]
    assert proven not in delta
    assert novel in delta
    assert added in delta
    assert sum(step["tool"] == "task" for step in delta) == 2


def test_redigested_foreign_parent_and_successor_cannot_replace_host_project(repo):
    """Matching producer claims never outrank the normalized host project."""

    from flyto_ai.coding.route import CodingRouteError, RouteLane

    route, parent, successor = _digest_bound_amendment_fixture(repo)
    parent["task_profile"]["project"] = "foreign_project"
    successor["task_profile"]["project"] = "foreign_project"
    _bind_successor_amendment(parent, successor)

    with pytest.raises(CodingRouteError, match="amendment_parent_proof_mismatch"):
        route._plan_groups(
            successor, RouteLane.INDEXER_PRE, parent_contract=parent,
            host_project=repo.name,
            host_requested_paths=successor["task_amendment"]["added_paths"],
        )


def test_self_consistent_unrequested_path_cannot_replace_host_target_authority(repo):
    """Producer digests cannot add a path absent from the host request."""

    from flyto_ai.coding.route import CodingRouteError, RouteLane

    route, parent, successor = _digest_bound_amendment_fixture(repo)

    with pytest.raises(CodingRouteError, match="amendment_path_boundary_invalid"):
        route._plan_groups(
            successor, RouteLane.INDEXER_PRE, parent_contract=parent,
            host_project=repo.name,
            host_requested_paths=[],
        )


def _digest_bound_amendment_fixture(repo):
    parent = _contract_section(
        task_id="task_feature_boundary", project=repo.name,
        paths=["orig/a.py"], fingerprint="1" * 64,
        steps=_synthetic_plan_group("parent", 1, 0),
    )
    successor = {
        "task_profile": {
            "version": "task-contract.v2",
            "task_id": "task_feature_boundary",
            "intent": "feature",
            "project": repo.name,
            "intent_fingerprint": "3" * 64,
            "instruction_fingerprint": "8" * 64,
        },
        "intent_ledger": {
            "version": "intent-ledger.v1",
            "fingerprint": "3" * 64,
            "description": parent["intent_ledger"]["description"],
            "allowed_paths": ["orig/a.py", "add/a.py"],
        },
        "instruction_context": {
            "version": "task-context.v1", "fingerprint": "8" * 64,
        },
        "task_amendment": {
            "version": "task-amendment.v1",
            "status": "amended",
            "root_task_id": "task_feature_boundary",
            "amendment_index": 1,
            "parent_contract_digest": "",
            "original_paths": ["orig/a.py"],
            "added_paths": ["add/a.py"],
            "cumulative_paths": ["orig/a.py", "add/a.py"],
        },
        "execution_plan": _synthetic_plan_group("successor", 1, 1),
        "resolved_targets": [
            {"input": "orig/a.py", "path": "orig/a.py",
             "symbol_id": "repo:orig/a.py:file:a"},
            {"input": "add/a.py", "path": "add/a.py",
             "symbol_id": "repo:add/a.py:file:a"},
        ],
        "targets": ["orig/a.py", "add/a.py"],
    }
    route = _orchestrator(FakeIndexer())
    _bind_successor_amendment(parent, successor)
    return route, parent, successor


@pytest.mark.parametrize("drift", ["profile", "ledger", "instruction", "digest"])
def test_amendment_delta_refuses_any_parent_digest_drift(repo, drift):
    from flyto_ai.coding.route import CodingRouteError, RouteLane

    route, parent, successor = _digest_bound_amendment_fixture(repo)
    parent = copy.deepcopy(parent)
    successor = copy.deepcopy(successor)
    if drift == "profile":
        parent["task_profile"]["version"] = "task-contract.v999"
    elif drift == "ledger":
        parent["intent_ledger"]["fingerprint"] = "4" * 64
        parent["task_profile"]["intent_fingerprint"] = "4" * 64
    elif drift == "instruction":
        parent["instruction_context"]["fingerprint"] = "5" * 64
        parent["task_profile"]["instruction_fingerprint"] = "5" * 64
    else:
        successor["task_amendment"]["parent_contract_digest"] = "f" * 64
    with pytest.raises(CodingRouteError, match="amendment_parent_proof_mismatch"):
        route._plan_groups(
            successor, RouteLane.INDEXER_PRE, parent_contract=parent,
            host_project=repo.name,
            host_requested_paths=successor["task_amendment"]["added_paths"],
        )


@pytest.mark.parametrize(
    "drift",
    [
        "not_mapping", "version", "status", "negative_index", "zero_index",
        "oversized_index", "boolean_index", "root", "missing_contract_id",
    ],
)
def test_amendment_delta_requires_a_valid_parent_generation_shape(repo, drift):
    from flyto_ai.coding.route import CodingRouteError, RouteLane

    route, parent, successor = _digest_bound_amendment_fixture(repo)
    parent["task_amendment"] = {
        "version": "task-amendment.v1",
        "status": "amended",
        "root_task_id": "task_feature_boundary",
        "amendment_index": 1,
        "contract_id": "amd_generation_one",
    }
    successor["task_amendment"]["amendment_index"] = 2
    if drift == "not_mapping":
        parent["task_amendment"] = "not-a-contract"
    elif drift == "version":
        parent["task_amendment"]["version"] = "task-amendment.v999"
    elif drift == "status":
        parent["task_amendment"]["status"] = "blocked"
    elif drift == "negative_index":
        parent["task_amendment"]["amendment_index"] = -1
    elif drift == "zero_index":
        parent["task_amendment"]["amendment_index"] = 0
    elif drift == "oversized_index":
        parent["task_amendment"]["amendment_index"] = 9
    elif drift == "boolean_index":
        parent["task_amendment"]["amendment_index"] = True
    elif drift == "root":
        parent["task_amendment"]["root_task_id"] = "task_other"
    else:
        parent["task_amendment"].pop("contract_id")
    # Rebind the digest after tampering: structural validation must reject the
    # malformed generation even when its bounded bytes match the claimed hash.
    successor["task_amendment"]["parent_contract_digest"] = (
        parent_contract_digest(parent)
    )
    with pytest.raises(CodingRouteError, match="amendment_parent_proof_mismatch"):
        route._plan_groups(
            successor, RouteLane.INDEXER_PRE, parent_contract=parent,
            host_project=repo.name,
            host_requested_paths=successor["task_amendment"]["added_paths"],
        )


def test_amendment_delta_requires_every_added_path_in_resolved_union(repo):
    from flyto_ai.coding.route import CodingRouteError, RouteLane

    route, parent, successor = _digest_bound_amendment_fixture(repo)
    successor["task_amendment"]["added_paths"].append("add/omitted.py")
    successor["task_amendment"]["cumulative_paths"].append("add/omitted.py")
    successor["intent_ledger"]["allowed_paths"].append("add/omitted.py")
    _bind_successor_amendment(parent, successor)
    with pytest.raises(CodingRouteError, match="amendment_plan_boundary_incomplete"):
        route._plan_groups(
            successor, RouteLane.INDEXER_PRE, parent_contract=parent,
            host_project=repo.name,
            host_requested_paths=successor["task_amendment"]["added_paths"],
        )


def test_generation_two_cumulative_plan_grows_but_delta_stays_at_32(repo):
    from flyto_ai.coding.route import RouteLane

    route, parent, successor = _digest_bound_amendment_fixture(repo)
    parent["execution_plan"] = _synthetic_plan_group("generation_two", 44, 0)
    _promote_parent_to_generation_one(parent)
    # 65 cumulative steps would be refused by a fixed x2 ceiling. Only the 19
    # added-path analyses and two successor gates execute in this generation.
    successor["execution_plan"] = _synthetic_plan_group("generation_two", 44, 19)
    _bind_successor_amendment(parent, successor)
    groups = route._plan_groups(
        successor, RouteLane.INDEXER_PRE, parent_contract=parent,
        host_project=repo.name,
        host_requested_paths=successor["task_amendment"]["added_paths"],
    )
    assert sum(len(steps) for _scope, steps in groups) == 21


def test_generation_two_cumulative_plan_still_has_a_finite_growth_bound(repo):
    from flyto_ai.coding.route import CodingRouteError, RouteLane

    route, parent, successor = _digest_bound_amendment_fixture(repo)
    parent["execution_plan"] = _synthetic_plan_group("oversized", 44, 0)
    _promote_parent_to_generation_one(parent)
    successor["execution_plan"] = _synthetic_plan_group("oversized", 75, 20)
    _bind_successor_amendment(parent, successor)
    with pytest.raises(CodingRouteError, match="plan_bound_exceeded"):
        route._plan_groups(
            successor, RouteLane.INDEXER_PRE, parent_contract=parent,
            host_project=repo.name,
            host_requested_paths=successor["task_amendment"]["added_paths"],
        )


def test_amendment_chain_depth_beyond_public_indexer_bound_is_refused(repo):
    from flyto_ai.coding.route import CodingRouteError, RouteLane

    route, parent, successor = _digest_bound_amendment_fixture(repo)
    parent["task_amendment"] = {
        "version": "task-amendment.v1",
        "status": "amended",
        "root_task_id": "task_feature_boundary",
        "amendment_index": 8,
        "contract_id": "amd_generation_eight",
    }
    successor["task_amendment"]["amendment_index"] = 9
    successor["task_amendment"]["parent_contract_digest"] = (
        parent_contract_digest(parent)
    )
    with pytest.raises(CodingRouteError, match="amendment_parent_proof_mismatch"):
        route._plan_groups(
            successor, RouteLane.INDEXER_PRE, parent_contract=parent,
            host_project=repo.name,
            host_requested_paths=successor["task_amendment"]["added_paths"],
        )


def test_rework_without_a_mutation_target_retains_prior_scope_exactly(repo):
    indexer = FakeIndexer()
    route = _orchestrator(indexer)
    request = CodingTaskRequest(
        message="edit docs/reference/python/coding.md", working_dir=str(repo),
    )
    _, first = asyncio.run(route._indexer_pre(request))

    rework = CodingTaskRequest(
        message="Run python3 tools/builder.py and attach evidence.",
        working_dir=str(repo),
    )
    prior = ("README.md", "docs/reference/python/coding.md")
    asyncio.run(route._indexer_pre(rework, first["task_contract"], prior))

    assert indexer.plans[-1]["targets"] == ["README.md"]


def test_post_validation_receives_exactly_the_supplied_cumulative_set(repo):
    """The defect in one assertion: validate what the audit will bind."""

    from flyto_ai.coding.contracts import CheckResult, CodingTaskResult

    indexer = FakeIndexer()
    route = _orchestrator(indexer)
    request = CodingTaskRequest(
        message="edit docs/reference/python/coding.md", working_dir=str(repo),
    )
    _, context = asyncio.run(route._indexer_pre(request))
    # Widen the ledger the way a real amendment would, so the union is planned.
    context["task_contract"] = dict(context["task_contract"])
    context["task_contract"]["intent_ledger"]["allowed_paths"] = [
        "docs/reference/python/README.md", "docs/reference/python/coding.md",
    ]
    result = CodingTaskResult(
        ok=True, message="done", thread_id="sdk-1", attempts=1, status="completed",
        files_changed=["docs/reference/python/README.md"],
        checks=[CheckResult(
            name="unit", passed=True, required=True, exit_code=0,
            duration_ms=1, output_sha256="0" * 64,
        )],
    )
    cumulative = [
        "docs/reference/python/README.md", "docs/reference/python/coding.md",
    ]
    asyncio.run(route._indexer_post(request, context, result, cumulative))
    sent = indexer.validations[-1]["current_state"]["changed_paths"]
    assert sent == cumulative
    # ...and without the cumulative set, the round-three failure reproduces.
    indexer.validations.clear()
    from flyto_ai.coding.route import CodingRouteError

    with pytest.raises(CodingRouteError) as excinfo:
        asyncio.run(route._indexer_post(request, context, result))
    assert excinfo.value.code == "validation_failed"
    assert indexer.validations[-1]["current_state"]["changed_paths"] == [
        "docs/reference/python/README.md",
    ]
    # The exact production symptom: a correct round refused because a
    # cumulative path from an earlier round was never presented.
    assert "validation_fix_intent_ledger_task_incomplete_scope" in (
        excinfo.value.blockers
    )


def test_a_domain_validation_refusal_keeps_bounded_typed_evidence(repo):
    """`pass=false` must stay actionable without carrying provider prose."""

    from flyto_ai.coding.route import CodingRouteError

    indexer = FakeIndexer()
    route = _orchestrator(indexer)
    request = CodingTaskRequest(
        message="edit docs/reference/python/coding.md", working_dir=str(repo),
    )
    _, context = asyncio.run(route._indexer_pre(request))
    detail = route._validation_failure_detail({
        "ok": True, "pass": False,
        "reason_codes": ["fix_intent_ledger:task:unplanned_diff"],
        "required_actions": ["amend_intent_ledger"],
    })
    assert isinstance(detail, str) and detail
    assert len(detail) <= 200
    for forbidden in ("/Users/", "\n", "secret"):
        assert forbidden not in detail
    assert str(CodingRouteError("validation_failed", None).code) == "validation_failed"


def test_unbounded_or_malformed_domain_evidence_is_bounded(repo):
    """A capability that answers with a novel is not a channel."""

    indexer = FakeIndexer()
    route = _orchestrator(indexer)
    for payload in (
        {"ok": True, "pass": False, "reason_codes": ["x" * 5000]},
        {"ok": True, "pass": False, "reason_codes": "not-a-list"},
        {"ok": True, "pass": False, "reason_codes": [{"nested": "object"}]},
        {"ok": True, "pass": False},
    ):
        detail = route._validation_failure_detail(payload)
        assert isinstance(detail, str)
        assert len(detail) <= 200


def test_task_plan_inner_failure_keeps_only_bounded_indexer_reason(repo):
    """The Code refusal stays actionable without persisting its error prose."""

    from flyto_ai.coding.route import CodingRouteError, RouteLane

    route = _orchestrator(FakeIndexer())
    route._begin_lane(RouteLane.INDEXER_PRE)
    raw = {
        "ok": True,
        "result": {
            "isError": True,
            "structuredContent": {
                "ok": False,
                "pass": False,
                "error": "Amendment refused at /Users/alice/private/project",
                "reason_codes": ["AMENDMENT_TARGET_UNRESOLVED"],
                "required_actions": ["declare_resolvable_amendment_targets"],
            },
        },
    }
    with pytest.raises(CodingRouteError) as excinfo:
        route._domain_payload(raw, RouteLane.INDEXER_PRE, "task.plan")
    assert excinfo.value.code == "domain_amendment_target_unresolved"
    assert excinfo.value.blockers == (
        "action_declare_resolvable_amendment_targets",
        "validation_amendment_target_unresolved",
    )
    rendered = json.dumps([item.to_mapping() for item in route._trace.calls])
    assert "domain_amendment_target_unresolved" in rendered
    assert "alice" not in rendered and "/Users/" not in rendered


def test_task_plan_hostile_reason_prose_and_paths_stay_generic(repo):
    from flyto_ai.coding.route import CodingRouteError, RouteLane

    route = _orchestrator(FakeIndexer())
    route._begin_lane(RouteLane.INDEXER_PRE)
    raw = {
        "ok": True,
        "result": {
            "structuredContent": {
                "ok": False,
                "error": "open /etc/passwd",
                "reason_codes": ["PLEASE OPEN /Users/alice/private token"],
                "required_actions": ["https://evil.example/action"],
            },
        },
    }
    with pytest.raises(CodingRouteError) as excinfo:
        route._domain_payload(raw, RouteLane.INDEXER_PRE, "task.plan")
    assert excinfo.value.code == "domain_failure"
    assert excinfo.value.blockers == ()
    rendered = json.dumps([item.to_mapping() for item in route._trace.calls])
    assert "alice" not in rendered and "passwd" not in rendered


# ──────────────────────────────────────────────────────────────────────
# the real state machine: three rounds, one root task, one session
# ──────────────────────────────────────────────────────────────────────


_SESSION = "sdk-plan-amendment-1"
_SETTLED = {
    CodingJobState.COMPLETED,
    CodingJobState.FAILED,
    CodingJobState.CODEX_ACCEPTED,
    CodingJobState.AWAITING_CODEX_AUDIT,
}


class RoundBackend:
    """Writes one more file per round, like a real implementer being reworked."""

    def __init__(self, workspace, rounds):
        self.workspace = workspace
        self.rounds = list(rounds)
        self.calls = []

    async def run(self, request):
        from flyto_ai.agents.models import CodeTaskResponse

        self.calls.append(request)
        for relative, body in self.rounds[len(self.calls) - 1]:
            target = self.workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        return CodeTaskResponse(
            ok=True, message="round {}".format(len(self.calls)),
            session_id="local-{}".format(len(self.calls)),
            attempts=1, claude_session_id=_SESSION, claude_num_turns=3,
        )


def _story_workspace(tmp_path):
    workspace = tmp_path / "story"
    workspace.mkdir(parents=True, exist_ok=True)
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir(exist_ok=True)
    config.write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: declared\n"
        "    argv: {}\n"
        "    required: true\n".format(json.dumps([sys.executable, "--version"])),
        encoding="utf-8",
    )
    return workspace


class _FakeCapabilityStatus:
    def __init__(self, name):
        self.name = name
        self.required = True
        self.available = True


class _FakeCapabilityManager:
    """Stands in for the capability transport, and for nothing else.

    The brief permits deterministic fakes at the host/capability boundary. Every
    state transition below it - the job record, the workspace claim, the resume
    envelope, rework rounds, the revision digest, the audit - is the real
    service doing real work.
    """

    def __init__(self, working_dir, permission):
        self.working_dir = working_dir
        self.permission = permission
        self.required_available = True

    async def start(self, specs):
        return [_FakeCapabilityStatus(getattr(s, "name", "cap")) for s in specs]

    async def close(self):
        return None


def _story_service(
    tmp_path, workspace, backend, indexer, monkeypatch, state_dir="story-state",
):
    from flyto_ai.agents.claude_code import ClaudeCodingAgent
    from flyto_ai.coding import capabilities as capabilities_module
    from flyto_ai.coding.contracts import ApprovalPolicy, SandboxMode
    from flyto_ai.coding.service import CodingService

    # Undone by pytest at teardown. A bare module assignment here leaks the
    # fake into every later suite in the same session.
    monkeypatch.setattr(
        capabilities_module, "CapabilityManager", _FakeCapabilityManager,
    )
    policy = CodingRoutePolicy(
        strict=True,
        indexer=_spec("flyto-indexer"),
        blueprint=_spec("flyto-blueprint", _BLUEPRINT_TOOLS),
        core_enabled=True,
        limits=RouteLimits(),
    )
    service = CodingService(
        lambda store: ClaudeCodingAgent(store, agent=backend),
        state_root=str(tmp_path / state_dir),
        workspace_roots=(str(workspace),),
        max_workers=1, max_queued=8, require_codex_audit=True,
        implementation_backend="claude",
        route_policy=policy,
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        approval_policy=ApprovalPolicy.NEVER,
        max_rework_rounds=5,
    )
    # Both host lanes reach the same deterministic capability. Core is left
    # unconfigured, so that lane resolves `not_applicable` exactly as it does
    # for a deployment without a Core adapter.
    service._lane_dispatcher = lambda manager, specs: indexer
    service._core_dispatcher = lambda: None
    return service


def _wait(service, tenant, job_id, timeout=60):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        receipt = service.get(tenant, job_id)
        if receipt.state in _SETTLED:
            return receipt
        time.sleep(0.02)
    raise AssertionError("coding job did not settle")


def _rework(service, tenant, job_id, revision, message):
    from flyto_ai.coding.contracts import (
        CodingAuditFinding, CodingAuditSeverity, CodingAuditVerdict,
    )

    return service.audit(
        tenant, job_id, revision, CodingAuditVerdict.REWORK,
        (CodingAuditFinding(
            code="scope", severity=CodingAuditSeverity.MAJOR, message=message,
        ),),
    )


@pytest.fixture()
def story(tmp_path, monkeypatch):
    from flyto_ai.coding.contracts import CodingTaskRequest as _Request

    workspace = _story_workspace(tmp_path)
    indexer = FakeIndexer()
    backend = RoundBackend(workspace, [
        [("a.py", "A1\n")],
        [("b.py", "B1\n")],
        [("c.py", "C1\n")],
    ])
    service = _story_service(tmp_path, workspace, backend, indexer, monkeypatch)
    # Round one asks for exactly what round one does. The later files enter
    # the ledger by amendment, from the auditor's findings - which is the whole
    # behaviour under test.
    request = _Request(message="create a.py", working_dir=str(workspace))
    return service, indexer, backend, workspace, request


def test_three_rounds_stay_one_root_task_with_a_growing_cumulative_scope(story):
    """The whole production story, end to end, through real service state."""

    service, indexer, backend, workspace, request = story
    try:
        first = service.submit("t", "round-1", request)
        awaiting = _wait(service, "t", first.job_id)
        assert awaiting.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert indexer.validations[-1]["current_state"]["changed_paths"] == ["a.py"]

        _rework(service, "t", first.job_id, awaiting.implementation_revision_sha256,
                "also add b.py")
        second = _wait(service, "t", first.job_id)
        assert second.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert indexer.validations[-1]["current_state"]["changed_paths"] == ["a.py", "b.py"]

        _rework(service, "t", first.job_id, second.implementation_revision_sha256,
                "also add c.py")
        third = _wait(service, "t", first.job_id)
        assert third.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert indexer.validations[-1]["current_state"]["changed_paths"] == [
            "a.py", "b.py", "c.py",
        ]

        # One root task across all three plans, growing by amendment.
        assert len(indexer.plans) == 3
        assert "task_contract" not in indexer.plans[0]
        assert indexer.plans[1]["targets"] == ["b.py"]
        assert indexer.plans[2]["targets"] == ["c.py"]
        for index in (1, 2):
            parent = indexer.plans[index]["task_contract"]
            assert parent["root_task_id"] == FakeIndexer.ROOT
            assert parent["generation"] == index
            assert parent["objective"] == indexer.plans[1]["task_contract"]["objective"]

        # The same implementation session throughout, and the final revision
        # binds exactly the validated cumulative set.
        assert backend.calls[1].sdk_session_id == _SESSION
        assert backend.calls[2].sdk_session_id == _SESSION
        record = service._read_json(
            service._tenant_dir(service._tenant_ref("t")) / "jobs"
            / (first.job_id + ".json"),
        )
        assert record["implementation_files"] == ["a.py", "b.py", "c.py"]
        assert third.implementation_revision_sha256 == service._revision_digest(
            str(workspace), ["a.py", "b.py", "c.py"],
        )
    finally:
        service.close(wait=True)


def test_a_rework_that_only_modifies_an_existing_file_still_closes(tmp_path, monkeypatch):
    """The ordinary case: round two edits A rather than adding B."""

    from flyto_ai.coding.contracts import CodingTaskRequest as _Request

    workspace = _story_workspace(tmp_path)
    indexer = FakeIndexer()
    backend = RoundBackend(workspace, [
        [("a.py", "A1\n")], [("a.py", "A2 repaired\n")],
    ])
    service = _story_service(tmp_path, workspace, backend, indexer, monkeypatch)
    try:
        first = service.submit("t", "modify-1", _Request(
            message="create a.py", working_dir=str(workspace),
        ))
        awaiting = _wait(service, "t", first.job_id)
        original = awaiting.implementation_revision_sha256

        _rework(service, "t", first.job_id, original, "repair a.py")
        second = _wait(service, "t", first.job_id)
        assert second.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert indexer.validations[-1]["current_state"]["changed_paths"] == ["a.py"]
        # The prior digest was proven before the round, and the new bytes are
        # bound afterwards: same file, different revision.
        assert second.implementation_revision_sha256 != original
        assert second.implementation_revision_sha256 == service._revision_digest(
            str(workspace), ["a.py"],
        )
        assert (workspace / "a.py").read_text() == "A2 repaired\n"
    finally:
        service.close(wait=True)


def test_a_new_service_object_resumes_and_amends_the_exact_prior_plan(tmp_path, monkeypatch):
    """Durability, not memory: a restart must be able to continue the root task."""

    from flyto_ai.coding.contracts import CodingTaskRequest as _Request

    workspace = _story_workspace(tmp_path)
    indexer = FakeIndexer()
    backend = RoundBackend(workspace, [[("a.py", "A1\n")], [("b.py", "B1\n")]])
    first_service = _story_service(tmp_path, workspace, backend, indexer, monkeypatch)
    try:
        job = first_service.submit("t", "restart-1", _Request(
            message="create a.py", working_dir=str(workspace),
        ))
        awaiting = _wait(first_service, "t", job.job_id)
        revision = awaiting.implementation_revision_sha256
    finally:
        first_service.close(wait=True)

    # A different object over the same durable state root.
    resumed = _story_service(tmp_path, workspace, backend, indexer, monkeypatch)
    try:
        _rework(resumed, "t", job.job_id, revision, "also add b.py")
        second = _wait(resumed, "t", job.job_id)
        assert second.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert indexer.plans[-1]["task_contract"]["root_task_id"] == FakeIndexer.ROOT
        assert indexer.plans[-1]["task_contract"]["generation"] == 1
        assert indexer.validations[-1]["current_state"]["changed_paths"] == ["a.py", "b.py"]
    finally:
        resumed.close(wait=True)


@pytest.mark.parametrize(
    "tamper,expected",
    [
        (lambda rec: rec.pop("indexer_plan_authority"), "plan_authority_unavailable"),
        (lambda rec: rec["indexer_plan_authority"].__setitem__("job_id", "job_" + "9" * 24),
         "plan_authority_unavailable"),
        (lambda rec: rec["indexer_plan_authority"].__setitem__("request_sha256", "9" * 64),
         "plan_authority_unavailable"),
        (lambda rec: rec["indexer_plan_authority"].__setitem__("workspace_sha256", "9" * 64),
         "plan_authority_unavailable"),
        (lambda rec: rec["indexer_plan_authority"]["contract"].__setitem__(
            "intent_ledger", ["everything"]), "plan_authority_unavailable"),
        (lambda rec: rec["indexer_plan_authority"].__setitem__("version", "old.v0"),
         "plan_authority_unavailable"),
    ],
    ids=["absent", "wrong-job", "wrong-request", "wrong-workspace",
         "tampered-contract", "stale-version"],
)
def test_unprovable_plan_authority_refuses_before_the_implementer(
    tmp_path, monkeypatch, tamper, expected,
):
    """Every refusal here happens with zero new provider calls."""

    from flyto_ai.coding.contracts import CodingTaskRequest as _Request

    workspace = _story_workspace(tmp_path)
    indexer = FakeIndexer()
    backend = RoundBackend(workspace, [[("a.py", "A1\n")], [("b.py", "B1\n")]])
    service = _story_service(tmp_path, workspace, backend, indexer, monkeypatch)
    try:
        job = service.submit("t", "tamper-1", _Request(
            message="create a.py", working_dir=str(workspace),
        ))
        awaiting = _wait(service, "t", job.job_id)
        assert len(backend.calls) == 1

        path = (
            service._tenant_dir(service._tenant_ref("t")) / "jobs"
            / (job.job_id + ".json")
        )
        record = service._read_json(path)
        tamper(record)
        service._write_json(path, record)

        _rework(service, "t", job.job_id, awaiting.implementation_revision_sha256,
                "also add b.py")
        failed = _wait(service, "t", job.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == expected
        assert failed.landable is False
        assert len(backend.calls) == 1, "the implementer was invoked on a refusal"
    finally:
        service.close(wait=True)


def test_altered_prior_revision_bytes_refuse_before_the_implementer(tmp_path, monkeypatch):
    """The stored revision must still describe the tree the round starts in."""

    from flyto_ai.coding.contracts import CodingTaskRequest as _Request

    workspace = _story_workspace(tmp_path)
    indexer = FakeIndexer()
    backend = RoundBackend(workspace, [[("a.py", "A1\n")], [("b.py", "B1\n")]])
    service = _story_service(tmp_path, workspace, backend, indexer, monkeypatch)
    try:
        job = service.submit("t", "bytes-1", _Request(
            message="create a.py", working_dir=str(workspace),
        ))
        awaiting = _wait(service, "t", job.job_id)
        _rework(service, "t", job.job_id, awaiting.implementation_revision_sha256,
                "also add b.py")
        # Somebody edits an already-attributable file behind the job's back.
        (workspace / "a.py").write_text("tampered\n", encoding="utf-8")
        failed = _wait(service, "t", job.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == "cumulative_revision_mismatch"
        assert len(backend.calls) == 1
    finally:
        service.close(wait=True)


def test_a_failed_pre_lane_never_leaves_amendable_authority(tmp_path, monkeypatch):
    """Authority is only ever sealed after a lane genuinely closed."""

    from flyto_ai.coding.contracts import CodingTaskRequest as _Request

    workspace = _story_workspace(tmp_path)

    class RefusingIndexer(FakeIndexer):
        async def __call__(self, tool, arguments):
            if tool == "task" and dict(arguments or {}).get("action") == "plan":
                return {"ok": False}
            return await super().__call__(tool, arguments)

    indexer = RefusingIndexer()
    backend = RoundBackend(workspace, [[("a.py", "A1\n")]])
    service = _story_service(tmp_path, workspace, backend, indexer, monkeypatch)
    try:
        job = service.submit("t", "refused-1", _Request(
            message="create a.py", working_dir=str(workspace),
        ))
        failed = _wait(service, "t", job.job_id)
        assert failed.state is CodingJobState.FAILED
        assert backend.calls == [], "the implementer ran after a refused pre-lane"
        record = service._read_json(
            service._tenant_dir(service._tenant_ref("t")) / "jobs"
            / (job.job_id + ".json"),
        )
        assert "indexer_plan_authority" not in record
    finally:
        service.close(wait=True)


def test_the_private_plan_envelope_never_reaches_a_public_projection(tmp_path, monkeypatch):
    """The contract is host state, not caller-visible evidence."""

    from flyto_ai.coding.contracts import CodingTaskRequest as _Request
    from flyto_ai.coding.service import receipt_to_mapping

    workspace = _story_workspace(tmp_path)
    indexer = FakeIndexer()
    backend = RoundBackend(workspace, [[("a.py", "A1\n")]])
    service = _story_service(tmp_path, workspace, backend, indexer, monkeypatch)
    try:
        job = service.submit("t", "private-1", _Request(
            message="create a.py", working_dir=str(workspace),
        ))
        awaiting = _wait(service, "t", job.job_id)
        record = service._read_json(
            service._tenant_dir(service._tenant_ref("t")) / "jobs"
            / (job.job_id + ".json"),
        )
        # Sealed privately...
        assert record["indexer_plan_authority"]["contract"]["root_task_id"] == (
            FakeIndexer.ROOT
        )
        # ...and absent from every public projection.
        rendered = json.dumps(receipt_to_mapping(awaiting))
        for forbidden in (
            "indexer_plan_authority", "intent_ledger", "root_task_id",
            "chain_sha256", FakeIndexer.ROOT, str(workspace),
        ):
            assert forbidden not in rendered, forbidden
    finally:
        service.close(wait=True)


def test_a_domain_validation_refusal_survives_into_host_failure_evidence(tmp_path, monkeypatch):
    """`pass=false` must stay repairable through the public contract."""

    from flyto_ai.coding.contracts import CodingTaskRequest as _Request

    workspace = _story_workspace(tmp_path)

    class NarrowIndexer(FakeIndexer):
        """Plans an empty ledger, so the round's own change is undeclared."""

        async def __call__(self, tool, arguments):
            arguments = dict(arguments or {})
            if tool == "task" and arguments.get("action") == "plan":
                arguments["targets"] = []
            return await super().__call__(tool, arguments)

    indexer = NarrowIndexer()
    backend = RoundBackend(workspace, [[("a.py", "A1\n")]])
    service = _story_service(tmp_path, workspace, backend, indexer, monkeypatch)
    try:
        job = service.submit("t", "domain-1", _Request(
            message="create a.py", working_dir=str(workspace),
        ))
        failed = _wait(service, "t", job.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == "route_validation_failed"
        # The domain's own reason and action survive, bounded and typed.
        assert "validation_fix_intent_ledger_task_unplanned_diff" in (
            failed.verification_blockers
        )
        assert "action_amend_intent_ledger" in failed.verification_blockers
        for token in failed.verification_blockers:
            assert len(token) <= 64 and token.replace("_", "").isalnum()
    finally:
        service.close(wait=True)


# ──────────────────────────────────────────────────────────────────────
# a domain code is something the capability already wrote, not prose
# ──────────────────────────────────────────────────────────────────────


_HOSTILE_DOMAIN_VALUES = [
    "please open /Users/alice/private token",
    "run arbitrary shell command",
    "rm -rf workspace",
    "https://evil.example/path",
    "C:\\Users\\alice\\secret.txt",
    "/etc/passwd",
    "../../escape",
    "Mixed_Case",
    "UPPER",
    "has space",
    "tab\there",
    "newline\nhere",
    "null\x00byte",
    "a" * 200,
    "",
    "1",
]


@pytest.mark.parametrize("value", _HOSTILE_DOMAIN_VALUES)
def test_untrusted_domain_prose_never_becomes_a_host_token(value):
    """Normalizing first is what made a sentence look like a control token.

    `please open /Users/alice/private token` normalized to
    `validation_please_open_users_alice_private_token`, which reads exactly like
    a code this host owns. The value has to already *be* a machine identifier
    before any separator mapping happens.
    """

    payload = {"reason_codes": [value], "required_actions": [value]}
    assert CodingRouteOrchestrator._domain_evidence(payload) == ()
    assert CodingRouteOrchestrator._validation_failure_detail(payload) == (
        "validation_failed"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"reason_codes": [{"nested": "object"}], "required_actions": [["a"]]},
        {"reason_codes": [None, 1, True], "required_actions": [3.5]},
        {"reason_codes": "fix_intent_ledger:task:unplanned_diff"},
        {"reason_codes": {"a": "b"}},
        {"reason_codes": [], "required_actions": []},
        {},
        None,
        "not-a-mapping",
    ],
)
def test_malformed_domain_evidence_is_dropped_whole(payload):
    assert CodingRouteOrchestrator._domain_evidence(payload) == ()


def test_excessive_domain_cardinality_is_bounded():
    payload = {
        "reason_codes": ["code_{}".format(index) for index in range(50)],
        "required_actions": ["act_{}".format(index) for index in range(50)],
    }
    tokens = CodingRouteOrchestrator._domain_evidence(payload)
    assert len(tokens) <= 8
    assert all(token.startswith(("validation_", "action_")) for token in tokens)


def test_the_real_indexer_codes_still_cross():
    """The rule may not cost the codes this repair exists to carry."""

    payload = {
        "reason_codes": [
            "fix_intent_ledger:task:unplanned_diff",
            "fix_intent_ledger:task:incomplete_scope",
        ],
        "required_actions": ["amend_intent_ledger"],
    }
    assert CodingRouteOrchestrator._domain_evidence(payload) == (
        "validation_fix_intent_ledger_task_unplanned_diff",
        "validation_fix_intent_ledger_task_incomplete_scope",
        "action_amend_intent_ledger",
    )
    assert CodingRouteOrchestrator._validation_failure_detail(payload) == (
        "validation_fix_intent_ledger_task_unplanned_diff"
    )


def test_secret_shaped_machine_codes_are_unknown_to_the_closed_registry():
    payload = {
        "reason_codes": ["API_KEY_SK_LIVE_SECRET"],
        "required_actions": ["token_abcdef0123456789"],
    }

    assert CodingRouteOrchestrator._domain_evidence(payload) == ()
    assert CodingRouteOrchestrator._validation_failure_detail(payload) == (
        "validation_failed"
    )


def test_hostile_domain_evidence_never_reaches_any_public_projection(
    tmp_path, monkeypatch,
):
    """The same rule, proven through the real service and the real facade."""

    from flyto_ai.coding.contracts import CodingTaskRequest as _Request
    from flyto_ai.coding.mcp_server import CodingMCPServer
    from flyto_ai.coding.service import receipt_to_mapping

    secret = "/Users/alice/private token"
    workspace = _story_workspace(tmp_path)

    class HostileIndexer(FakeIndexer):
        async def __call__(self, tool, arguments):
            arguments = dict(arguments or {})
            if tool == "task" and arguments.get("action") == "validate":
                self.validations.append(arguments)
                return {
                    "ok": True, "pass": False,
                    "reason_codes": ["please open {}".format(secret),
                                     "https://evil.example/path",
                                     "API_KEY_SK_LIVE_SECRET"],
                    "required_actions": ["rm -rf workspace",
                                         "token_abcdef0123456789"],
                }
            return await super().__call__(tool, arguments)

    indexer = HostileIndexer()
    backend = RoundBackend(workspace, [[("a.py", "A1\n")]])
    service = _story_service(tmp_path, workspace, backend, indexer, monkeypatch)
    try:
        job = service.submit("t", "hostile-1", _Request(
            message="create a.py", working_dir=str(workspace),
        ))
        failed = _wait(service, "t", job.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.landable is False
        assert failed.verification_blockers == ()
        assert failed.failure_code == "route_validation_failed"

        server = CodingMCPServer(service, tenant_id="t")
        response = server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "flyto_coding_get",
                       "arguments": {"job_id": job.job_id}},
        })
        rendered = (
            json.dumps(response) + json.dumps(receipt_to_mapping(failed))
        ).lower()
        for forbidden in (
            "alice", "private", "evil.example", "rm_rf", "rm -rf", "https",
            "please_open", "workspace_token", "api_key_sk_live_secret",
            "token_abcdef", str(workspace).lower(),
        ):
            assert forbidden not in rendered, forbidden
    finally:
        service.close(wait=True)


# ──────────────────────────────────────────────────────────────────────
# the failure phase and the recovery action must be true
# ──────────────────────────────────────────────────────────────────────


def test_unprovable_plan_authority_reports_a_truthful_phase_and_action(
    tmp_path, monkeypatch,
):
    """`preflight` promises no job and no claim. Both existed here.

    This refusal happens inside an admitted job that already holds a durable
    worktree claim, after capability startup, before the implementer call. The
    truthful phase is `verification`: the host could not verify the authority
    the round would have run under. The recovery is a fresh job against the
    authority that exists now, because identical rework cannot repair a durable
    fact that is missing or contradicted.
    """

    from flyto_ai.coding.contracts import (
        ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,
        FAILURE_PHASE_PREFLIGHT,
        FAILURE_PHASE_VERIFICATION,
        CodingTaskRequest as _Request,
    )
    from flyto_ai.coding.mcp_server import CodingMCPServer
    from flyto_ai.coding.service import (
        PLAN_AUTHORITY_CODES, PlanAuthorityUnprovable, receipt_to_mapping,
    )

    error = PlanAuthorityUnprovable("plan_authority_unavailable")
    assert error.failure_phase == FAILURE_PHASE_VERIFICATION
    assert error.failure_phase != FAILURE_PHASE_PREFLIGHT
    assert error.retryable is False
    assert error.required_actions == (ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,)
    # The code stays inside its closed set whatever it is handed.
    assert PlanAuthorityUnprovable("not-a-known-code").code == (
        "plan_authority_unavailable"
    )
    assert set(PLAN_AUTHORITY_CODES) >= {
        "plan_authority_unavailable", "plan_authority_unsealable",
        "cumulative_revision_mismatch",
    }

    workspace = _story_workspace(tmp_path)
    indexer = FakeIndexer()
    backend = RoundBackend(workspace, [[("a.py", "A1\n")], [("b.py", "B1\n")]])
    service = _story_service(tmp_path, workspace, backend, indexer, monkeypatch)
    try:
        job = service.submit("t", "phase-1", _Request(
            message="create a.py", working_dir=str(workspace),
        ))
        awaiting = _wait(service, "t", job.job_id)
        assert len(backend.calls) == 1

        path = (
            service._tenant_dir(service._tenant_ref("t")) / "jobs"
            / (job.job_id + ".json")
        )
        record = service._read_json(path)
        # A claim really did exist at the moment of refusal.
        assert service._workspace_claim_path(str(workspace)).is_file()
        record.pop("indexer_plan_authority")
        service._write_json(path, record)

        _rework(service, "t", job.job_id, awaiting.implementation_revision_sha256,
                "also add b.py")
        failed = _wait(service, "t", job.job_id)

        assert failed.state is CodingJobState.FAILED
        assert failed.landable is False
        assert failed.failure_code == "plan_authority_unavailable"
        assert len(backend.calls) == 1, "the implementer ran on a refusal"
        # Terminal, so the claim is released.
        assert not service._workspace_claim_path(str(workspace)).is_file()

        payload = receipt_to_mapping(failed)
        assert payload["failure_phase"] == FAILURE_PHASE_VERIFICATION
        assert payload["retryable"] is False
        assert payload["required_actions"] == [
            ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,
        ]
        assert payload["job_terminal"] is True

        server = CodingMCPServer(service, tenant_id="t")
        response = server.handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "flyto_coding_get",
                       "arguments": {"job_id": job.job_id}},
        })
        job_payload = response["result"]["structuredContent"]["job"]
        assert job_payload["failure_phase"] == FAILURE_PHASE_VERIFICATION
        assert job_payload["required_actions"] == [
            ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,
        ]
        assert "indexer_plan_authority" not in json.dumps(job_payload)
    finally:
        service.close(wait=True)


# ──────────────────────────────────────────────────────────────────────
# a lane that refuses after the cumulative proof does not narrow the round
# ──────────────────────────────────────────────────────────────────────


def _record_of(service, tenant, job_id):
    return service._read_json(
        service._tenant_dir(service._tenant_ref(tenant)) / "jobs" / (job_id + ".json"),
    )


class LateRefusingIndexer(FakeIndexer):
    """Passes post-work once, then refuses every later round.

    The refusal lands in `indexer_post`, which runs *after* the route seam has
    already proven this round's cumulative scope. That ordering is the whole
    point: the scope was proven, and then something else said no.
    """

    async def __call__(self, tool, arguments):
        arguments = dict(arguments or {})
        if tool == "task" and arguments.get("action") == "validate":
            self.validations.append(arguments)
            if len(self.validations) == 1:
                return {"ok": True, "pass": True}
            return {
                "ok": True, "pass": False,
                "reason_codes": ["fix_intent_ledger:task:unplanned_diff"],
                "required_actions": ["amend_intent_ledger"],
            }
        return await super().__call__(tool, arguments)


def test_a_post_scope_refusal_still_binds_the_whole_proven_scope(tmp_path, monkeypatch):
    """The production shape: source+test, then a doc-only rework, then a refusal.

    Round one owns two files. Round two resumes the same session and touches
    only a third. A later lane refuses that round, so it stays failed and
    non-landable - but the evidence it leaves behind must describe what the job
    actually owns. Binding `files_changed` alone would publish a revision over
    the doc only, silently dropping the source and test earlier rounds opened.
    """

    from flyto_ai.coding.contracts import CodingTaskRequest as _Request

    workspace = _story_workspace(tmp_path)
    indexer = LateRefusingIndexer()
    backend = RoundBackend(workspace, [
        [("a.py", "A1\n"), ("test_a.py", "T1\n")],
        [("docs/guide.md", "D1\n")],
    ])
    service = _story_service(tmp_path, workspace, backend, indexer, monkeypatch)
    whole = ["a.py", "docs/guide.md", "test_a.py"]
    try:
        job = service.submit("t", "late-refusal-1", _Request(
            # One verb per path: the route deliberately refuses to let a single
            # mutation verb widen across a second filename.
            message="create a.py and create test_a.py", working_dir=str(workspace),
        ))
        awaiting = _wait(service, "t", job.job_id)
        assert awaiting.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert _record_of(service, "t", job.job_id)["implementation_files"] == [
            "a.py", "test_a.py",
        ]

        _rework(service, "t", job.job_id, awaiting.implementation_revision_sha256,
                "also document it")
        failed = _wait(service, "t", job.job_id)

        # The refusal is real: the round did not become auditable or landable.
        assert failed.state is CodingJobState.FAILED
        assert failed.landable is False
        assert failed.failure_code == "route_validation_failed"
        # The lane saw the proven union, and so does the terminal evidence.
        assert indexer.validations[-1]["current_state"]["changed_paths"] == whole
        record = _record_of(service, "t", job.job_id)
        assert record["implementation_files"] == whole
        assert record["implementation_revision_sha256"] == service._revision_digest(
            str(workspace), whole,
        )
        # Same session throughout, and the doc really is only one of three.
        assert record["implementation_session_id"] == _SESSION
        assert record["result"]["files_changed"] == ["docs/guide.md"]
    finally:
        service.close(wait=True)


def test_a_rework_that_left_its_session_adds_nothing_to_the_proven_scope(
    tmp_path, monkeypatch,
):
    """A foreign session cannot widen the scope, and cannot narrow it either.

    A round that answers in a session the record never bound is refused at the
    implementer/session boundary, before the route ever calls the cumulative
    callback. `progress.route_scope` is therefore still empty when the terminal
    evidence is built - and so is the refused round's own change set, because a
    result the host cannot attribute to the bound session reports no files.

    The safe outcome is that nothing moves. The last scope this job actually
    proved stays exactly as it was, and the edit the foreign session made is
    neither attributed to the job nor folded into a new cumulative revision.
    """

    from flyto_ai.coding.contracts import CodingTaskRequest as _Request

    class DriftingBackend(RoundBackend):
        """Round two answers in a session the record never bound."""

        async def run(self, request):
            response = await super().run(request)
            if len(self.calls) > 1:
                response.claude_session_id = "sdk-somewhere-else-1"
            return response

    workspace = _story_workspace(tmp_path)
    indexer = FakeIndexer()
    backend = DriftingBackend(workspace, [
        [("a.py", "A1\n"), ("test_a.py", "T1\n")],
        [("docs/guide.md", "D1\n")],
    ])
    service = _story_service(tmp_path, workspace, backend, indexer, monkeypatch)
    try:
        job = service.submit("t", "drift-1", _Request(
            # One verb per path: the route deliberately refuses to let a single
            # mutation verb widen across a second filename.
            message="create a.py and create test_a.py", working_dir=str(workspace),
        ))
        awaiting = _wait(service, "t", job.job_id)
        assert awaiting.state is CodingJobState.AWAITING_CODEX_AUDIT
        # The last scope this job actually proved, captured before the drifting
        # round runs. Nothing the refused round does may move either value.
        proven_revision = awaiting.implementation_revision_sha256
        assert _record_of(service, "t", job.job_id)["implementation_files"] == [
            "a.py", "test_a.py",
        ]

        _rework(service, "t", job.job_id, proven_revision, "also document it")
        failed = _wait(service, "t", job.job_id)

        assert failed.state is CodingJobState.FAILED
        assert failed.landable is False
        assert failed.failure_code == "route_implementation_not_successful"
        record = _record_of(service, "t", job.job_id)
        # The cumulative callback was never reached, so no union was proven -
        # and the unattributable round reported no files of its own either, so
        # the terminal record keeps exactly what was already proved.
        assert record["implementation_files"] == ["a.py", "test_a.py"]
        assert record["implementation_revision_sha256"] == proven_revision
        # The foreign session's edit exists on disk but is not this job's work,
        # so it is neither attributed nor hashed into a new revision.
        assert (workspace / "docs" / "guide.md").is_file()
        assert "docs/guide.md" not in record["implementation_files"]
        assert record["implementation_revision_sha256"] != service._revision_digest(
            str(workspace), ["a.py", "docs/guide.md", "test_a.py"],
        )
    finally:
        service.close(wait=True)


def test_terminal_evidence_without_a_proven_scope_stays_fail_closed(
    tmp_path, monkeypatch,
):
    """The seam itself: an unproven scope adds nothing, and no scope adds less.

    A round that failed before the cumulative proof succeeded hands the proof
    builder an empty scope. It must fall back to this round's own snapshot and
    never reach into the stored record for files an earlier round opened.
    """

    from flyto_ai.coding.contracts import CodingTaskRequest as _Request
    from flyto_ai.coding.contracts import CodingTaskResult

    workspace = _story_workspace(tmp_path)
    (workspace / "docs").mkdir(exist_ok=True)
    (workspace / "docs" / "guide.md").write_text("D1\n", encoding="utf-8")
    service = _story_service(
        tmp_path, workspace, RoundBackend(workspace, []), FakeIndexer(), monkeypatch,
    )
    try:
        request = _Request(message="document it", working_dir=str(workspace))
        result = CodingTaskResult(
            ok=False, message="refused", thread_id=_SESSION, attempts=1,
            status="failed", files_changed=["docs/guide.md"],
            failure_code="route_cumulative_scope_unsafe",
        )

        unproven = service._failed_round_proof(request, result, True, {}, ())
        assert unproven["implementation_files"] == ["docs/guide.md"]
        assert unproven["implementation_revision_sha256"] == service._revision_digest(
            str(workspace), ["docs/guide.md"],
        )
        assert unproven["implementation_session_id"] == _SESSION

        # A round that never invoked an implementer proves nothing at all, with
        # or without a scope argument.
        never_started = service._failed_round_proof(
            request, result, False, {}, ("a.py", "docs/guide.md"),
        )
        assert "implementation_files" not in never_started
        assert "implementation_revision_sha256" not in never_started
        assert "implementation_session_id" not in never_started
    finally:
        service.close(wait=True)
