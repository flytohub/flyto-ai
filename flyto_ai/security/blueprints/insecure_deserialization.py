"""Insecure-deserialization test blueprint — flyto-core YAML.

Hard to safely verify without RCE. We fire crafted payloads that, for a
vulnerable parser, raise distinctive error messages (PHP `__wakeup`, Python
pickle exceptions, Java `ObjectInputStream`). Pattern-matching the echoed
server error distinguishes "accepted the payload" from "rejected at the
boundary".
"""
from __future__ import annotations

import base64
import yaml

from ..schema import SecurityFinding


_DESER_ERROR_SIGNATURES = [
    "__wakeup",            # PHP
    "__destruct",          # PHP
    "pickle.UnpicklingError",  # Python (if body leaks)
    "ObjectInputStream",   # Java
    "readObject",          # Java
    "InvalidClassException",
    "marshal",             # Ruby marshal errors sometimes echoed
    "YAMLError",           # Ruby/Python YAML unsafe loader echoes
    "cPickle",
]


def generate(finding: SecurityFinding, target_url: str, auth_token: str | None) -> str:
    method = (finding.http_method or "POST").upper()
    param = finding.param_name or "data"
    path = finding.endpoint_path or "/api/import"
    url = f"{target_url}{path}"

    # A cross-format grab bag — server reveals which parser is in use by
    # the error message it echoes.
    # Python pickle: gzip'd trigger for reduce exploitation oracle
    python_pickle_b64 = base64.b64encode(b"\x80\x04\x95\x0c\x00\x00\x00\x00\x00\x00\x00\x8c\x07nothing\x94.").decode()
    # PHP serialized object with __wakeup
    php_serialized = 'O:8:"Attacker":0:{}'
    # Java serialized header bytes (hex: aced0005)
    java_b64 = base64.b64encode(bytes.fromhex("aced0005")).decode()
    # Ruby marshal trigger
    ruby_yaml = "!ruby/object:Gem::Installer\ni: x"

    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    probe_requests = [
        {"method": method, "url": url, "headers": headers,
         "body": {param: "plain-text-string"}, "label": "baseline"},
        {"method": method, "url": url, "headers": headers,
         "body": {param: python_pickle_b64, "encoding": "base64"}, "label": "python_pickle"},
        {"method": method, "url": url, "headers": headers,
         "body": {param: php_serialized}, "label": "php_serialized"},
        {"method": method, "url": url, "headers": headers,
         "body": {param: java_b64, "encoding": "base64"}, "label": "java_stream"},
        {"method": method, "url": url, "headers": headers,
         "body": {param: ruby_yaml, "format": "yaml"}, "label": "ruby_yaml"},
    ]

    workflow = {
        "id": "verify_insecure_deser",
        "name": f"Insecure Deserialization Verify at {path}",
        "version": "1.0.0",
        "description": f"Auto-generated insecure-deser verification at {finding.source_file}:{finding.source_line}",
        "timeout": 120,
        "steps": [
            {"id": "start", "module": "flow.start", "label": "Start", "params": {}, "orderIndex": 0},
            {
                "id": "deser_probes",
                "module": "http.batch",
                "label": "Insecure Deserialization Probes",
                "params": {
                    "description": "Baseline string + 4 format-specific serialized payloads",
                    "requests": probe_requests,
                    "timeout": 10,
                    "ssrf_protection": False,
                    "detect_patterns": _DESER_ERROR_SIGNATURES,
                },
                "orderIndex": 1,
            },
            {
                "id": "assert_deser",
                "module": "test.assert_contains",
                "label": "Assert Deserialization Pipeline Reached",
                "params": {
                    "source": "${deser_probes.items}",
                    "patterns": _DESER_ERROR_SIGNATURES,
                    "match_mode": "any",
                    # Matching any signature means the server engaged a deserializer.
                    # That's "exploitable surface" even if the gadget itself failed.
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
                        "result": "${assert_deser.data}",
                        "finding_source": f"{finding.source_file}:{finding.source_line}",
                        "finding_sink": f"{finding.sink_file}:{finding.sink_line}",
                    },
                },
                "orderIndex": 3,
            },
        ],
        "edges": [
            {"source": "start", "target": "deser_probes"},
            {"source": "deser_probes", "target": "assert_deser"},
            {"source": "assert_deser", "target": "report"},
        ],
    }
    return yaml.dump(workflow, default_flow_style=False, allow_unicode=True, sort_keys=False)
