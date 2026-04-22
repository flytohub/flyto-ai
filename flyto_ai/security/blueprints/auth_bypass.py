"""Auth bypass test blueprint — generates flyto-core YAML."""
from __future__ import annotations

import yaml

from ..schema import SecurityFinding


def generate(
    finding: SecurityFinding,
    target_url: str,
    auth_token: str | None,
) -> str:
    """Generate a flyto-core workflow YAML for auth bypass verification.

    Pure HTTP (no browser needed). Tests: no auth header, empty auth header,
    forged JWT with alg:none, session token swap.
    """
    method = (finding.http_method or "GET").upper()
    path = finding.endpoint_path or "/api/protected"
    url = f"{target_url}{path}"

    # Forged JWT with alg:none (base64 of {"alg":"none","typ":"JWT"}.{"sub":"attacker","admin":true}.)
    forged_jwt = (
        "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0"
        ".eyJzdWIiOiJhdHRhY2tlciIsImFkbWluIjp0cnVlfQ."
    )

    probe_requests = [
        # 1. Baseline: valid auth (should return 200)
        {
            "method": method,
            "url": url,
            "headers": _auth_headers(auth_token),
            "label": "baseline_valid_auth",
        },
        # 2. No Authorization header at all
        {
            "method": method,
            "url": url,
            "headers": {},
            "label": "no_auth_header",
        },
        # 3. Empty Authorization header
        {
            "method": method,
            "url": url,
            "headers": {"Authorization": ""},
            "label": "empty_auth_header",
        },
        # 4. Forged JWT with alg:none
        {
            "method": method,
            "url": url,
            "headers": {"Authorization": f"Bearer {forged_jwt}"},
            "label": "forged_jwt_alg_none",
        },
        # 5. Swapped session token (random UUID)
        {
            "method": method,
            "url": url,
            "headers": {"Authorization": "Bearer aaaa-bbbb-cccc-dddd-eeee"},
            "label": "swapped_token",
        },
    ]

    workflow = {
        "id": f"verify_auth_bypass_{path.replace('/', '_').strip('_')}",
        "name": f"Auth Bypass Verify: {path}",
        "version": "1.0.0",
        "description": (
            f"Auto-generated verification for auth bypass "
            f"at {finding.source_file}:{finding.source_line}"
        ),
        "timeout": 300,
        "steps": [
            {
                "id": "start",
                "module": "flow.start",
                "label": "Start",
                "params": {},
                "orderIndex": 0,
            },
            {
                "id": "auth_probes",
                "module": "http.batch",
                "label": "Auth Bypass Probes",
                "params": {
                    "description": (
                        "Baseline + 4 auth bypass attempts "
                        "(no auth, empty, forged JWT, swapped token)"
                    ),
                    "requests": probe_requests,
                    "timeout": 60,
                },
                "orderIndex": 1,
            },
            {
                "id": "assert_bypass",
                "module": "test.assert_status",
                "label": "Assert Auth Bypass Detected",
                "params": {
                    "source": "${auth_probes.items}",
                    "baseline_index": 0,
                    "probe_indices": [1, 2, 3, 4],
                    "expected_blocked": [401, 403],
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
                        "result": "${assert_bypass.data}",
                        "finding_source": f"{finding.source_file}:{finding.source_line}",
                        "finding_sink": f"{finding.sink_file}:{finding.sink_line}",
                    },
                },
                "orderIndex": 3,
            },
        ],
        "edges": [
            {"source": "start", "target": "auth_probes"},
            {"source": "auth_probes", "target": "assert_bypass"},
            {"source": "assert_bypass", "target": "report"},
        ],
    }

    return yaml.dump(workflow, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _auth_headers(auth_token: str | None) -> dict[str, str]:
    """Build auth headers from token, or empty dict."""
    if auth_token:
        return {"Authorization": f"Bearer {auth_token}"}
    return {}
