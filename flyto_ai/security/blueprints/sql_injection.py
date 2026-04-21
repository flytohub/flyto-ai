"""SQL injection test blueprint — generates flyto-core YAML."""
from __future__ import annotations

import yaml

from ..schema import SecurityFinding


def generate(
    finding: SecurityFinding,
    target_url: str,
    auth_token: str | None,
) -> str:
    """Generate a flyto-core workflow YAML for SQL injection verification.

    Uses http.batch with classic, time-based, and error-based payloads,
    then test.assert_contains to detect SQL error patterns.
    """
    method = (finding.http_method or "GET").upper()
    param = finding.param_name or "q"
    path = finding.endpoint_path or "/"
    base_url = f"{target_url}{path}"

    # Payloads: classic, time-based, error-based
    payloads = [
        ("classic", "' OR '1'='1"),
        ("time_based", "'; WAITFOR DELAY '00:00:05'--"),
        ("error_based", "' AND 1=CONVERT(int, @@version)--"),
    ]

    # Build probe requests
    probe_requests = []
    # Baseline request first (safe value)
    probe_requests.append(_build_request(method, base_url, param, "1", auth_token))
    for _label, payload in payloads:
        probe_requests.append(
            _build_request(method, base_url, param, payload, auth_token)
        )

    workflow = {
        "id": f"verify_sqli_{param}",
        "name": f"SQL Injection Verify: {param} at {path}",
        "version": "1.0.0",
        "description": (
            f"Auto-generated verification for SQL injection "
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
                "id": "sql_probes",
                "module": "http.batch",
                "label": "SQL Injection Probes",
                "params": {
                    "description": (
                        f"Baseline + 3 SQL injection payloads "
                        f"targeting {param}"
                    ),
                    "requests": probe_requests,
                    "measure_time": True,
                    "timeout": 60,
                    "detect_patterns": [
                        "sql syntax",
                        "mysql_fetch",
                        "pg_query",
                        "ORA-",
                        "sqlite3",
                        "unclosed quotation",
                        "SQLSTATE",
                        "Microsoft SQL",
                        "ODBC Driver",
                    ],
                },
                "orderIndex": 1,
            },
            {
                "id": "assert_sqli",
                "module": "test.assert_contains",
                "label": "Assert SQL Injection Detected",
                "params": {
                    "source": "${sql_probes.data}",
                    "patterns": [
                        "sql syntax",
                        "mysql_fetch",
                        "pg_query",
                        "ORA-",
                        "sqlite3",
                        "unclosed quotation",
                        "SQLSTATE",
                    ],
                    "match_mode": "any",
                    "on_match": "exploitable",
                    "on_no_match": "sanitized",
                },
                "orderIndex": 2,
            },
            {
                "id": "assert_timing",
                "module": "test.assert_timing",
                "label": "Assert Time-Based Blind SQLi",
                "params": {
                    "source": "${sql_probes.data}",
                    "baseline_index": 0,
                    "probe_index": 2,
                    "threshold_ms": 3000,
                    "on_slow": "exploitable",
                    "on_normal": "inconclusive",
                },
                "orderIndex": 3,
            },
            {
                "id": "report",
                "module": "output.display",
                "label": "Report",
                "params": {
                    "format": "json",
                    "data": {
                        "pattern_result": "${assert_sqli.data}",
                        "timing_result": "${assert_timing.data}",
                        "finding_source": f"{finding.source_file}:{finding.source_line}",
                        "finding_sink": f"{finding.sink_file}:{finding.sink_line}",
                    },
                },
                "orderIndex": 4,
            },
        ],
        "edges": [
            {"source": "start", "target": "sql_probes"},
            {"source": "sql_probes", "target": "assert_sqli"},
            {"source": "assert_sqli", "target": "assert_timing"},
            {"source": "assert_timing", "target": "report"},
        ],
    }

    return yaml.dump(workflow, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _build_request(
    method: str,
    base_url: str,
    param: str,
    value: str,
    auth_token: str | None,
) -> dict:
    """Build a single HTTP request dict for http.batch."""
    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    if method == "GET":
        sep = "&" if "?" in base_url else "?"
        req: dict = {
            "method": "GET",
            "url": f"{base_url}{sep}{param}={value}",
        }
    else:
        headers["Content-Type"] = "application/json"
        req = {
            "method": method,
            "url": base_url,
            "body": f'{{{json_kv(param, value)}}}',
        }

    if headers:
        req["headers"] = headers
    return req


def json_kv(key: str, value: str) -> str:
    """Format a JSON key-value pair with proper escaping."""
    escaped_value = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{key}": "{escaped_value}"'
