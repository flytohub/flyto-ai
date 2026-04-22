"""Command-injection test blueprint — generates flyto-core YAML.

Two oracles:
  1. Pattern: `id` / `uname` output echoed in the body.
  2. Timing: `sleep 5` or `; sleep 5` causes measurable latency delta.
"""
from __future__ import annotations

import yaml

from ..schema import SecurityFinding


_COMMAND_OUTPUT_PATTERNS = [
    "uid=",          # output of `id`
    "gid=",
    "Linux",         # output of `uname -a`
    "root:x:0:0",    # output of `cat /etc/passwd`
]


def generate(
    finding: SecurityFinding,
    target_url: str,
    auth_token: str | None,
) -> str:
    """Generate a flyto-core workflow YAML for command-injection verification."""
    method = (finding.http_method or "POST").upper()
    param = finding.param_name or "cmd"
    path = finding.endpoint_path or "/exec"
    base = f"{target_url}{path}"

    # Baseline — benign payload. Probes are injection variants.
    baseline_value = "hello"
    payloads = [
        ("semicolon_id", "; id"),
        ("pipe_id", "| id"),
        ("backtick_id", "`id`"),
        ("newline_uname", "\nuname -a"),
        ("timing_sleep", "; sleep 4"),
    ]

    probe_requests = [_build_request(method, base, param, baseline_value, auth_token)]
    for _label, payload in payloads:
        probe_requests.append(_build_request(method, base, param, payload, auth_token))

    # Timing probe sits at index 5 (last payload).
    timing_probe_index = len(payloads)

    workflow = {
        "id": f"verify_cmd_injection_{param}",
        "name": f"Command Injection Verify: {param} at {path}",
        "version": "1.0.0",
        "description": (
            f"Auto-generated verification for command injection at "
            f"{finding.source_file}:{finding.source_line}"
        ),
        "timeout": 300,
        "steps": [
            {"id": "start", "module": "flow.start", "label": "Start",
             "params": {}, "orderIndex": 0},
            {
                "id": "cmd_probes",
                "module": "http.batch",
                "label": "Command Injection Probes",
                "params": {
                    "description": (
                        "Baseline + 5 command-injection payloads (meta chars, "
                        "pipe, backtick, newline, sleep)"
                    ),
                    "requests": probe_requests,
                    # Sequential so the sleep-based timing probe is reliable.
                    "measure_time": True,
                    "timeout": 20,
                    "ssrf_protection": False,
                    "detect_patterns": _COMMAND_OUTPUT_PATTERNS,
                },
                "orderIndex": 1,
            },
            {
                "id": "assert_cmd_pattern",
                "module": "test.assert_contains",
                "label": "Assert Shell Output Reflected",
                "params": {
                    "source": "${cmd_probes.items}",
                    "patterns": _COMMAND_OUTPUT_PATTERNS,
                    "match_mode": "any",
                    "on_match": "exploitable",
                    "on_no_match": "inconclusive",
                },
                "orderIndex": 2,
            },
            {
                "id": "assert_cmd_timing",
                "module": "test.assert_timing",
                "label": "Assert Time-Based Command Injection",
                "params": {
                    "source": "${cmd_probes.items}",
                    "baseline_index": 0,
                    "probe_index": timing_probe_index,
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
                        "pattern": "${assert_cmd_pattern.data}",
                        "timing": "${assert_cmd_timing.data}",
                        "finding_source": f"{finding.source_file}:{finding.source_line}",
                        "finding_sink": f"{finding.sink_file}:{finding.sink_line}",
                    },
                },
                "orderIndex": 4,
            },
        ],
        "edges": [
            {"source": "start", "target": "cmd_probes"},
            {"source": "cmd_probes", "target": "assert_cmd_pattern"},
            {"source": "assert_cmd_pattern", "target": "assert_cmd_timing"},
            {"source": "assert_cmd_timing", "target": "report"},
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
