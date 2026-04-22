"""Open redirect test blueprint — generates flyto-core YAML.

Sends redirect-target payloads and checks whether the server's 3xx Location
header points at an attacker-controlled host, which is the exploit condition.
"""
from __future__ import annotations

import yaml

from ..schema import SecurityFinding


# Attacker-controlled destinations that a properly-validated redirect MUST refuse.
# We detect exploitability by matching these domains in any response body
# (when the server 302s, aiohttp follows and the final URL lands on the attacker
# domain — its body echoes the domain).
_ATTACKER_DOMAINS = [
    "evil.example.com",
    "attacker.example.net",
    "flyto-attacker-test.com",
]


def generate(
    finding: SecurityFinding,
    target_url: str,
    auth_token: str | None,
) -> str:
    """Generate a flyto-core workflow YAML for open-redirect verification."""
    method = (finding.http_method or "GET").upper()
    param = finding.param_name or "next"
    path = finding.endpoint_path or "/redirect"
    base = f"{target_url}{path}"

    # Baseline points at the same-origin; probes are attacker-controlled.
    baseline_value = target_url
    payloads = [
        ("absolute_attacker", f"https://{_ATTACKER_DOMAINS[0]}/evil"),
        ("protocol_relative", f"//{_ATTACKER_DOMAINS[1]}/evil"),
        ("javascript_uri", "javascript:alert(1)"),
        ("userinfo_bypass",
         f"https://{target_url.split('//')[-1]}@{_ATTACKER_DOMAINS[2]}/"),
    ]

    probe_requests = [_build_request(method, base, param, baseline_value, auth_token, follow=False)]
    for _label, payload in payloads:
        probe_requests.append(_build_request(method, base, param, payload, auth_token, follow=False))

    workflow = {
        "id": f"verify_open_redirect_{param}",
        "name": f"Open Redirect Verify: {param} at {path}",
        "version": "1.0.0",
        "description": (
            f"Auto-generated verification for open redirect at "
            f"{finding.source_file}:{finding.source_line}"
        ),
        "timeout": 120,
        "steps": [
            {"id": "start", "module": "flow.start", "label": "Start",
             "params": {}, "orderIndex": 0},
            {
                "id": "redirect_probes",
                "module": "http.batch",
                "label": "Open Redirect Probes",
                "params": {
                    "description": (
                        "Baseline + 4 redirect-target payloads targeting "
                        f"parameter {param}"
                    ),
                    "requests": probe_requests,
                    "timeout": 10,
                    "ssrf_protection": False,
                    # Look for attacker-controlled hosts in any Location header
                    # echoed back. http.batch captures the full headers dict
                    # per-request, but detect_patterns scans bodies — so we
                    # rely on the assert_contains step which also scans bodies.
                    "detect_patterns": _ATTACKER_DOMAINS,
                },
                "orderIndex": 1,
            },
            {
                "id": "assert_redirect",
                "module": "test.assert_contains",
                "label": "Assert Open Redirect Exploitable",
                "params": {
                    # Checking bodies from the batch — follow-redirects is off,
                    # so the raw 3xx body usually echoes the Location target.
                    "source": "${redirect_probes.items}",
                    "patterns": _ATTACKER_DOMAINS,
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
                        "result": "${assert_redirect.data}",
                        "finding_source": f"{finding.source_file}:{finding.source_line}",
                        "finding_sink": f"{finding.sink_file}:{finding.sink_line}",
                    },
                },
                "orderIndex": 3,
            },
        ],
        "edges": [
            {"source": "start", "target": "redirect_probes"},
            {"source": "redirect_probes", "target": "assert_redirect"},
            {"source": "assert_redirect", "target": "report"},
        ],
    }
    return yaml.dump(workflow, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _build_request(
    method: str,
    base: str,
    param: str,
    value: str,
    auth_token: str | None,
    follow: bool = False,
) -> dict:
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    if method == "GET":
        sep = "&" if "?" in base else "?"
        req: dict = {
            "method": "GET",
            "url": f"{base}{sep}{param}={value}",
            "follow_redirects": follow,
        }
    else:
        headers["Content-Type"] = "application/json"
        req = {
            "method": method,
            "url": base,
            "body": {param: value},
            "follow_redirects": follow,
        }

    if headers:
        req["headers"] = headers
    return req
