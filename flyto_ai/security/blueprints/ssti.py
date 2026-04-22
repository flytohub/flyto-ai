"""SSTI (Server-Side Template Injection) test blueprint — flyto-core YAML.

Classic oracle: `{{7*7}}` / `${7*7}` / `<%= 7*7 %>` — if the server evaluates
and returns `49`, the template engine is rendering user input.

URL-encodes payloads so flyto-core's variable resolver (which normalizes
`{{...}}` → `${...}`) doesn't eat the mustache syntax before the URL
ever hits the network.
"""
from __future__ import annotations

import urllib.parse

import yaml

from ..schema import SecurityFinding


def generate(finding: SecurityFinding, target_url: str, auth_token: str | None) -> str:
    method = (finding.http_method or "GET").upper()
    param = finding.param_name or "name"
    path = finding.endpoint_path or "/hello"
    base = f"{target_url}{path}"

    # Each engine's "7 * 7 = 49" oracle. We detect the literal string "49".
    payloads = [
        ("jinja2_twig", "{{7*7}}"),
        ("el_ognl", "${7*7}"),
        ("erb_ejs", "<%= 7*7 %>"),
        ("handlebars_angular", "{{ 7 * 7 }}"),
        ("freemarker", "${'49'}"),  # double-check — keeps result stable
    ]

    probe_requests = [_req(method, base, param, "alice", auth_token)]
    for _label, p in payloads:
        probe_requests.append(_req(method, base, param, p, auth_token))

    workflow = {
        "id": f"verify_ssti_{param}",
        "name": f"SSTI Verify: {param} at {path}",
        "version": "1.0.0",
        "description": f"Auto-generated SSTI verification at {finding.source_file}:{finding.source_line}",
        "timeout": 120,
        "steps": [
            {"id": "start", "module": "flow.start", "label": "Start", "params": {}, "orderIndex": 0},
            {
                "id": "ssti_probes",
                "module": "http.batch",
                "label": "SSTI Probes",
                "params": {
                    "description": "Baseline + 5 template-expression payloads (Jinja2/EL/ERB/Handlebars/Freemarker)",
                    "requests": probe_requests,
                    "timeout": 10,
                    "ssrf_protection": False,
                    "detect_patterns": ["49"],
                },
                "orderIndex": 1,
            },
            {
                "id": "assert_ssti",
                "module": "test.assert_contains",
                "label": "Assert SSTI Exploitable",
                "params": {
                    "source": "${ssti_probes.items}",
                    "patterns": [">49<", " 49 ", '"49"'],  # expression evaluated in HTML/JSON context
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
                        "result": "${assert_ssti.data}",
                        "finding_source": f"{finding.source_file}:{finding.source_line}",
                        "finding_sink": f"{finding.sink_file}:{finding.sink_line}",
                    },
                },
                "orderIndex": 3,
            },
        ],
        "edges": [
            {"source": "start", "target": "ssti_probes"},
            {"source": "ssti_probes", "target": "assert_ssti"},
            {"source": "assert_ssti", "target": "report"},
        ],
    }
    return yaml.dump(workflow, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _req(method, base, param, value, auth_token):
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    if method == "GET":
        sep = "&" if "?" in base else "?"
        encoded_value = urllib.parse.quote(value, safe="")
        req: dict = {"method": "GET", "url": f"{base}{sep}{param}={encoded_value}"}
    else:
        headers["Content-Type"] = "application/json"
        req = {"method": method, "url": base, "body": {param: value}}
    if headers:
        req["headers"] = headers
    return req
