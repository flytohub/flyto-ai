"""Security generation and fail-closed adaptive campaign contracts."""
from .campaign import (
    SECURITY_CAMPAIGN_VERSION,
    classify_security_action,
    compile_security_campaign,
    evaluate_campaign_action,
    project_evidence_for_planner,
    record_campaign_result,
    run_security_campaign,
    verify_security_campaign,
)
from .generator import generate_test_from_finding
from .schema import SecurityFinding

__all__ = [
    "SECURITY_CAMPAIGN_VERSION",
    "SecurityFinding",
    "classify_security_action",
    "compile_security_campaign",
    "evaluate_campaign_action",
    "generate_test_from_finding",
    "project_evidence_for_planner",
    "record_campaign_result",
    "run_security_campaign",
    "verify_security_campaign",
]
