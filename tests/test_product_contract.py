# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Pin the product role this package declares.

`flyto-blueprint` and `flyto-core` each ship `flyto-product.toml` and assert it
exactly. flyto-ai did not, while both of those files disclaim "intent and
provider governance" and hand it here — so the layer that owns it was the one
layer no file claimed. This test closes that: layer 1 now states its own role,
and the statement is checked rather than described.
"""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = {
    "schema": "flyto.product-contract.v1",
    "product": "Flyto2",
    "promise": "Turn AI work into verified, replayable procedures.",
    "proof_line": "AI said it finished. Flyto2 shows the proof.",
}
AI_PACKAGE = {
    "name": "flyto-ai",
    "layer": "intent_governance",
    "layer_order": 1,
    "owns": [
        "intent and provider governance",
        "guarded dispatch into the execution layer",
        "the audited coding route and its lane evidence",
    ],
    "does_not_own": [
        "reusable procedure learning and scoring",
        "deterministic execution, replay, and evidence",
        "hosted product and account logic",
    ],
}


def _load_contract() -> dict:
    with (ROOT / "flyto-product.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_ai_product_contract_is_exact() -> None:
    assert _load_contract() == {**COMMON, "package": AI_PACKAGE}


def test_ai_claims_exactly_what_the_other_layers_disclaim() -> None:
    """The three packages must not leave a responsibility unowned or doubled.

    Blueprint and Core both list "intent and provider governance" under
    `does_not_own`. If this package ever stops owning it, or starts also owning
    what one of them owns, the topology has a hole or an overlap and this fails
    before the prose does.
    """
    package = _load_contract()["package"]
    assert "intent and provider governance" in package["owns"]
    for disclaimed in package["does_not_own"]:
        assert disclaimed not in package["owns"]
