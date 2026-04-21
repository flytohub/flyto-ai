"""XSS reflected test blueprint — generates flyto-core YAML."""
from __future__ import annotations

import yaml

from ..schema import SecurityFinding


def generate(
    finding: SecurityFinding,
    target_url: str,
    auth_token: str | None,
) -> str:
    """Generate a flyto-core workflow YAML for reflected XSS verification.

    Uses browser.launch + http.batch for payload delivery, then
    browser.evaluate to check DOM for unescaped script execution.
    """
    param = finding.param_name or "q"
    path = finding.endpoint_path or "/search"
    base_url = f"{target_url}{path}"

    payloads = [
        "<script>alert(1)</script>",
        '<img src=x onerror=alert(1)>',
        "<svg/onload=alert(1)>",
    ]

    # Build probe requests
    probe_requests = []
    # Baseline
    probe_requests.append(_build_get_request(base_url, param, "safe_value", auth_token))
    for payload in payloads:
        probe_requests.append(
            _build_get_request(base_url, param, payload, auth_token)
        )

    workflow = {
        "id": f"verify_xss_{param}",
        "name": f"XSS Reflected Verify: {param} at {path}",
        "version": "1.0.0",
        "description": (
            f"Auto-generated verification for reflected XSS "
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
                "id": "launch",
                "module": "browser.launch",
                "label": "Launch Browser",
                "params": {
                    "stealth": True,
                    "headless": True,
                },
                "orderIndex": 1,
            },
            {
                "id": "xss_probes",
                "module": "http.batch",
                "label": "Reflected XSS Probes",
                "params": {
                    "description": (
                        f"Baseline + 3 XSS payloads targeting {param}"
                    ),
                    "requests": probe_requests,
                    "timeout": 60,
                    "detect_patterns": [
                        "<script>alert",
                        "onerror=alert",
                        "onload=alert",
                    ],
                },
                "orderIndex": 2,
            },
            {
                "id": "dom_check",
                "module": "browser.evaluate",
                "label": "DOM XSS Verification",
                "params": {
                    "url": f"{base_url}?{param}=<script>alert(1)</script>",
                    "script": _dom_check_script(),
                },
                "orderIndex": 3,
            },
            {
                "id": "screenshot",
                "module": "browser.screenshot",
                "label": "Capture Evidence",
                "params": {
                    "full_page": True,
                },
                "orderIndex": 4,
            },
            {
                "id": "assert_xss",
                "module": "test.assert_contains",
                "label": "Assert XSS Detected",
                "params": {
                    "source": "${xss_probes.data}",
                    "patterns": [
                        "<script>alert",
                        "onerror=alert",
                        "onload=alert",
                    ],
                    "match_mode": "any",
                    "on_match": "exploitable",
                    "on_no_match": "sanitized",
                },
                "orderIndex": 5,
            },
            {
                "id": "report",
                "module": "output.display",
                "label": "Report",
                "params": {
                    "format": "json",
                    "data": {
                        "http_result": "${assert_xss.data}",
                        "dom_result": "${dom_check.data}",
                        "finding_source": f"{finding.source_file}:{finding.source_line}",
                        "finding_sink": f"{finding.sink_file}:{finding.sink_line}",
                    },
                },
                "orderIndex": 6,
            },
        ],
        "edges": [
            {"source": "start", "target": "launch"},
            {"source": "launch", "target": "xss_probes"},
            {"source": "xss_probes", "target": "dom_check"},
            {"source": "dom_check", "target": "screenshot"},
            {"source": "screenshot", "target": "assert_xss"},
            {"source": "assert_xss", "target": "report"},
        ],
    }

    return yaml.dump(workflow, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _build_get_request(
    base_url: str,
    param: str,
    value: str,
    auth_token: str | None,
) -> dict:
    """Build a GET request for http.batch."""
    sep = "&" if "?" in base_url else "?"
    req: dict = {
        "method": "GET",
        "url": f"{base_url}{sep}{param}={value}",
    }
    if auth_token:
        req["headers"] = {"Authorization": f"Bearer {auth_token}"}
    return req


def _dom_check_script() -> str:
    """JavaScript to evaluate in the browser to detect XSS execution."""
    return """\
(function() {
  var results = { script_tags_injected: false, event_handlers_injected: false, alert_detected: false };

  // Check if injected <script> tags exist in body
  var scripts = document.querySelectorAll('script');
  for (var i = 0; i < scripts.length; i++) {
    if (scripts[i].textContent && scripts[i].textContent.includes('alert(1)')) {
      results.script_tags_injected = true;
    }
  }

  // Check for event handler injection
  var allElements = document.querySelectorAll('[onerror], [onload]');
  if (allElements.length > 0) {
    results.event_handlers_injected = true;
  }

  // Override alert to detect if it fires
  var originalAlert = window.alert;
  var alertFired = false;
  window.alert = function() { alertFired = true; };
  try {
    // Re-evaluate inline scripts
    var inlineScripts = document.querySelectorAll('script:not([src])');
    for (var j = 0; j < inlineScripts.length; j++) {
      try { eval(inlineScripts[j].textContent); } catch(e) {}
    }
  } catch(e) {}
  window.alert = originalAlert;
  results.alert_detected = alertFired;

  return JSON.stringify(results);
})()"""
