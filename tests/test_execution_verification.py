import copy

import pytest

from flyto_ai.execution_verification import (
    EXECUTION_VERIFICATION_RECEIPT_VERSION,
    build_closed_loop_verification_receipt,
    build_execution_verification_receipt,
)


def _safe_evidence():
    return {
        "modules": ["string.uppercase", "string.lowercase"],
        "steps": [
            {"step_id": "upper", "module_id": "string.uppercase"},
            {"step_id": "lower", "module_id": "string.lowercase"},
        ],
        "checks": {"validation": True, "assertions": True},
        "counts": {"steps": 2, "assertions": 2},
        "structural_digest": "sha256:" + "a" * 64,
    }


def test_builder_returns_exact_canonical_six_field_receipt():
    evidence = _safe_evidence()
    reordered = dict(reversed(list(evidence.items())))

    receipt = build_execution_verification_receipt("execution:one", evidence)
    other = build_execution_verification_receipt("execution:one", reordered)

    assert set(receipt) == {
        "receipt_version", "success", "status", "evidence_id",
        "evidence_sha256", "evidence",
    }
    assert receipt["receipt_version"] == EXECUTION_VERIFICATION_RECEIPT_VERSION
    assert receipt["success"] is True
    assert receipt["status"] == "verified"
    assert receipt["evidence_sha256"] == other["evidence_sha256"]
    evidence["checks"]["validation"] = False
    assert receipt["evidence"]["checks"]["validation"] is True


@pytest.mark.parametrize("evidence", [
    {"payload": "raw user input"},
    {"foo": 3.14},
    {"checks": {"not_boolean": "raw text"}},
    {"Payload": "raw user input"},
    {"RAW_RESULT": "hidden"},
    {"User_Text": "hidden"},
    {"checks": {"Raw_Result": True}},
    {"counts": {"PAYLOAD": 1}},
])
def test_builder_rejects_hostile_or_raw_shaped_evidence(evidence):
    evidence["structural_digest"] = "sha256:" + "a" * 64
    with pytest.raises(ValueError):
        build_execution_verification_receipt("execution:safe", evidence)


def test_builder_rejects_unknown_step_fields_and_duplicate_identifiers():
    unknown = _safe_evidence()
    unknown["steps"][0]["payload"] = "raw"
    duplicate_steps = _safe_evidence()
    duplicate_steps["steps"][1]["step_id"] = "upper"
    duplicate_modules = _safe_evidence()
    duplicate_modules["modules"][1] = "string.uppercase"
    for evidence in (unknown, duplicate_steps, duplicate_modules):
        with pytest.raises(ValueError):
            build_execution_verification_receipt("execution:invalid", evidence)


@pytest.mark.parametrize("field,value", [
    ("checks", {"valid": 1}),
    ("counts", {"steps": True}),
    ("counts", {"steps": -1}),
    ("steps", [{"step_id": "one", "module_id": "safe", "executed": 1}]),
    ("steps", [{"step_id": "one", "module_id": "safe", "depends_on": ["bad value"]}]),
])
def test_builder_rejects_non_exact_nested_values(field, value):
    evidence = _safe_evidence()
    evidence[field] = value
    with pytest.raises(ValueError):
        build_execution_verification_receipt("execution:invalid", evidence)


def test_builder_rejects_oversized_collections():
    evidence = _safe_evidence()
    evidence["modules"] = ["module.{}".format(index) for index in range(257)]
    with pytest.raises(ValueError, match="bounded"):
        build_execution_verification_receipt("execution:oversized", evidence)


@pytest.mark.parametrize("relationships,match", [
    ({"upper": ["missing"]}, "missing a declared step"),
    ({"upper": ["upper"]}, "depend on itself"),
    ({"upper": ["lower"], "lower": ["upper"]}, "contain a cycle"),
])
def test_builder_rejects_impossible_step_relationships(relationships, match):
    evidence = _safe_evidence()
    for step in evidence["steps"]:
        step["depends_on"] = relationships.get(step["step_id"], [])
    with pytest.raises(ValueError, match=match):
        build_execution_verification_receipt("execution:relationships", evidence)


@pytest.mark.parametrize("execution_id", [
    "", "closed-loop:", "../execution", "execution/child", "execution:child",
    "execution//child", ".", "..", 7, None,
])
def test_closed_loop_receipt_requires_exact_safe_observed_execution_id(execution_id):
    with pytest.raises(ValueError, match="execution_id is malformed"):
        build_closed_loop_verification_receipt(
            execution_id,
            [],
            {"validation": True},
            0,
            "sha256:" + "a" * 64,
        )


def test_builder_binds_exact_observed_outcome_inside_hashed_evidence():
    receipt = build_execution_verification_receipt(
        "outcome:one", _safe_evidence(), outcome_success=False,
    )
    assert receipt["evidence"]["outcome_success"] is False
    with pytest.raises(TypeError, match="exact boolean"):
        build_execution_verification_receipt(
            "outcome:bad", _safe_evidence(), outcome_success=1,
        )
    mismatched = _safe_evidence()
    mismatched["outcome_success"] = True
    with pytest.raises(ValueError, match="does not match"):
        build_execution_verification_receipt(
            "outcome:mismatch", mismatched, outcome_success=False,
        )


def test_closed_loop_validator_binds_execution_workflow_and_outcome():
    workflow_hash = "sha256:" + "b" * 64
    structural = build_closed_loop_verification_receipt(
        "bp_exact",
        [{
            "step_id": "one",
            "module_id": "string.uppercase",
            "executed": True,
            "validation": {"valid": True},
            "assertions": [{"ok": True}],
        }],
        {"validation_passed": True},
        1,
        workflow_hash,
    )
    receipt = build_execution_verification_receipt(
        structural["evidence_id"], structural["evidence"], outcome_success=True,
    )
    assert receipt["evidence_id"] == "closed-loop:bp_exact"
    assert receipt["evidence"]["structural_digest"] == workflow_hash
    assert receipt["evidence"]["outcome_success"] is True
    tampered = copy.deepcopy(receipt)
    tampered["evidence"]["steps"][0]["executed"] = False
    rebuilt = build_execution_verification_receipt(
        tampered["evidence_id"], tampered["evidence"], outcome_success=True,
    )
    assert rebuilt != tampered


def test_blueprint_rejects_missing_malformed_tampered_and_mismatched_without_mutation():
    from flyto_blueprint import BlueprintEngine, MemoryBackend

    storage = MemoryBackend()
    engine = BlueprintEngine(storage=storage)
    workflow = {
        "name": "receipt boundary",
        "steps": [
            {"id": "one", "module": "string.uppercase", "params": {}},
            {"id": "two", "module": "string.lowercase", "params": {}},
            {"id": "three", "module": "string.reverse", "params": {}},
        ],
    }
    valid = build_execution_verification_receipt("learning:valid", _safe_evidence())
    malformed = {"receipt_version": EXECUTION_VERIFICATION_RECEIPT_VERSION}
    tampered = copy.deepcopy(valid)
    tampered["evidence"]["checks"]["validation"] = False

    for receipt in (None, malformed, tampered):
        result = engine.learn_from_execution(workflow, verification=receipt)
        assert result["ok"] is False
        assert storage.load_all() == []

    learned = engine.learn_from_execution(workflow, verification=valid)
    assert learned["ok"] is True
    blueprint_id = learned["data"]["id"]
    score = engine._blueprints[blueprint_id]["score"]
    outcome_receipt = build_execution_verification_receipt(
        "outcome:mismatch", _safe_evidence(), outcome_success=False,
    )
    rejected = engine.report_outcome(
        blueprint_id, True, execution_id="outcome:mismatch",
        verification=outcome_receipt,
    )
    assert rejected["ok"] is False
    assert engine._blueprints[blueprint_id]["score"] == score
