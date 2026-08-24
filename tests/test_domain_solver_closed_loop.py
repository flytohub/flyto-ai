"""Locked Core source contract plus installed Core -> AI -> Blueprint closure.

This validates software arithmetic only. It is not sensor, hardware,
physical-frame, reaction, laboratory, medical, handling, or safety validation.

The governed stack check requires and runs the locked sibling Core source
contract. Its narrow skip supports an isolated flyto-ai checkout only. The AI
closure always exercises the installed/imported Core capabilities through all
three known-answer and rejection cases.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from flyto_ai.capability_router import GOAL_FRAME_VERSION, route_with_flyto
from flyto_ai.execution_verification import build_execution_verification_receipt
from flyto_ai.tools.core_tools import dispatch_core_tool

_CASES = (
    (
        "math.rigid_transform_3d",
        "domain.solve.rigid-transform-3d",
        "solve.rigid-transform-3d",
        "transform.point-3d",
        {
            "point": [1, 0, 0],
            "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            "translation": [10, 20, 30],
            "source_frame": "sensor",
            "target_frame": "world",
            "length_unit": "m",
        },
        {"point": [10.0, 21.0, 30.0]},
    ),
    (
        "physics.kinematics_constant_acceleration",
        "domain.solve.constant-acceleration-kinematics",
        "solve.constant-acceleration-kinematics",
        "compute.position-velocity",
        {
            "x0": 1,
            "v0": 2,
            "acceleration": 3,
            "time": 4,
            "solve_mode": "position_and_velocity",
            "position_unit": "m",
            "velocity_unit": "m/s",
            "acceleration_unit": "m/s^2",
            "time_unit": "s",
        },
        {"position": 33.0, "velocity": 14.0},
    ),
    (
        "chemistry.ideal_dilution",
        "domain.solve.ideal-dilution",
        "solve.ideal-dilution",
        "compute.stock-diluent-volume",
        {
            "stock_concentration": 2,
            "target_concentration": 0.5,
            "final_volume": 1,
            "concentration_unit": "mol/L",
            "volume_unit": "L",
            "solve_mode": "stock_and_diluent_volume",
        },
        {"stock_volume": 0.25, "diluent_volume": 0.75},
    ),
)


def test_real_core_domain_solver_semantic_contract() -> None:
    """Run the locked sibling's source contract in a governed stack workspace.

    An isolated flyto-ai checkout lacks the stack_lock-required sibling, so only
    that development shape skips this source-contract portion.
    """
    core = Path(__file__).resolve().parents[2] / "flyto-core"
    contract = core / "tests" / "core" / "test_domain_solvers.py"
    if not contract.is_file():
        pytest.skip("the locked flyto-core sibling checkout is unavailable")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            str(contract.relative_to(core)),
        ],
        cwd=core,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _goal_frame(intent: str, affordance: str) -> dict[str, object]:
    return {
        "contract_version": GOAL_FRAME_VERSION,
        "intent_ids": [intent],
        "required_affordances": [affordance],
        "desired_effects": ["data.compute-only"],
        "trigger_events": ["domain.solve.requested"],
        "constraints": [],
    }


@pytest.mark.asyncio
async def test_real_domain_solver_route_execution_learning_and_falsification():
    from flyto_blueprint import BlueprintEngine
    from flyto_blueprint.storage.memory import MemoryBackend

    executions = []
    for index, (
        module_id,
        capability_id,
        intent,
        affordance,
        params,
        answer,
    ) in enumerate(_CASES, start=1):
        decision = await route_with_flyto(
            "semantic identifiers are the only routing authority",
            [],
            goal_frame=_goal_frame(intent, affordance),
            blueprint_search=lambda _query: [],
        )
        selected = decision["route"]["candidates"][0]
        assert (selected["runtime_name"], selected["canonical_id"]) == (
            module_id,
            capability_id,
        )
        receipt = await dispatch_core_tool(
            "execute_module", {"module_id": module_id, "params": params}
        )
        assert set(receipt) == {
            "receipt_version",
            "success",
            "status",
            "evidence_id",
            "evidence_sha256",
            "evidence",
        }
        assert receipt["success"] is True
        assert all(
            receipt["evidence"]["result"].get(key) == value
            for key, value in answer.items()
        )
        executions.append((index, module_id, receipt))

    digest_material = [receipt["evidence_sha256"] for _, _, receipt in executions]
    structural_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                digest_material,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    ai_receipt = build_execution_verification_receipt(
        "domain-solvers:known-answers",
        {
            "modules": [module_id for _, module_id, _ in executions],
            "steps": [
                {
                    "step_id": f"solve_{index}",
                    "module_id": module_id,
                    "validation_ok": True,
                    "assertions_ok": True,
                    "executed": True,
                    "assertion_count": 1,
                    "attempt_count": 1,
                }
                for index, module_id, _ in executions
            ],
            "checks": {"core_receipts_valid": True, "known_answers": True},
            "counts": {"executed_steps": 3, "evidence_items": 3},
            "structural_digest": structural_digest,
        },
    )
    assert set(ai_receipt) == {
        "receipt_version",
        "success",
        "status",
        "evidence_id",
        "evidence_sha256",
        "evidence",
    }
    assert "params" not in repr(ai_receipt).lower()
    assert "result" not in repr(ai_receipt).lower()

    workflow = {
        "name": "Deterministic domain solver known answers",
        "description": "Three independent compute-only software validations.",
        "steps": [
            {"id": f"solve_{index}", "module": module_id, "params": {}}
            for index, module_id, _ in executions
        ],
    }
    engine = BlueprintEngine(storage=MemoryBackend())
    learned = engine.learn_from_execution(
        workflow,
        blueprint_id="verified_domain_solvers",
        verification=ai_receipt,
    )
    assert learned["ok"] is True

    trusted_state = copy.deepcopy(engine._blueprints)
    tampered = copy.deepcopy(ai_receipt)
    tampered["evidence"]["checks"]["known_answers"] = False
    rejected = engine.learn_from_execution(
        workflow,
        blueprint_id="tampered_domain_solvers",
        verification=tampered,
    )
    assert rejected["ok"] is False
    assert engine._blueprints == trusted_state

    invalid = (
        (
            _CASES[0][0],
            {**_CASES[0][4], "rotation": [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        ),
        (_CASES[1][0], {**_CASES[1][4], "time": -1}),
        (_CASES[2][0], {**_CASES[2][4], "target_concentration": 3}),
    )
    for module_id, params in invalid:
        refused = await dispatch_core_tool(
            "execute_module", {"module_id": module_id, "params": params}
        )
        assert refused["ok"] is False
    assert engine._blueprints == trusted_state
