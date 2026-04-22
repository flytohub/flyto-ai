"""Security test generation from structured findings."""
from __future__ import annotations

import os
from urllib.parse import urlparse

from .blueprints import (
    auth_bypass,
    command_injection,
    crlf_injection,
    insecure_deserialization,
    mass_assignment,
    nosql_injection,
    open_redirect,
    path_traversal,
    sql_injection,
    ssrf,
    ssti,
    xss_reflected,
    xxe,
)
from .schema import SecurityFinding

_BLUEPRINT_MAP = {
    "sql_injection": sql_injection.generate,
    "xss_reflected": xss_reflected.generate,
    "auth_bypass": auth_bypass.generate,
    "ssrf": ssrf.generate,
    "open_redirect": open_redirect.generate,
    "command_injection": command_injection.generate,
    "path_traversal": path_traversal.generate,
    "xxe": xxe.generate,
    "ssti": ssti.generate,
    "nosql_injection": nosql_injection.generate,
    "insecure_deserialization": insecure_deserialization.generate,
    "mass_assignment": mass_assignment.generate,
    "crlf_injection": crlf_injection.generate,
}


def generate_test_from_finding(
    finding: SecurityFinding,
    target_url: str,
    auth_token: str | None = None,
) -> str:
    """Turn a structured finding into an executable flyto-core YAML string.

    Args:
        finding: Structured vulnerability finding from flyto-indexer or flyto-engine.
        target_url: Base URL to test against (staging, never production).
        auth_token: Optional bearer token for authenticated tests.

    Returns:
        YAML string ready to POST to flyto-core's /v1/workflow/run.

    Raises:
        ValueError: if finding.category has no blueprint, or target_url
                    fails safety validation.
    """
    _validate_safety(target_url)

    generator = _BLUEPRINT_MAP.get(finding.category)
    if not generator:
        raise ValueError(
            f"No blueprint for category '{finding.category}'. "
            f"Available: {list(_BLUEPRINT_MAP.keys())}"
        )
    return generator(finding, target_url, auth_token)


def _validate_safety(target_url: str) -> None:
    """Block against accidental production targets and SSRF vectors."""
    parsed = urlparse(target_url)

    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid URL: {target_url}")

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported scheme: {parsed.scheme}")

    # Block loopback, link-local, metadata endpoints
    bad_hosts = {
        "169.254.169.254",          # AWS/GCP metadata
        "metadata.google.internal",  # GCP metadata
    }
    if parsed.hostname in bad_hosts:
        raise ValueError(f"Refusing to target {parsed.hostname} — SSRF risk")

    # Block private/internal IP ranges
    _check_private_ip(parsed.hostname)

    # Require explicit staging subdomain unless overridden
    if not os.environ.get("FLYTO_AI_ALLOW_PROD_TARGETS"):
        hostname = parsed.hostname
        safe_patterns = ("staging", "localhost", "127.0.0.1", "0.0.0.0")
        if not any(pat in hostname for pat in safe_patterns):
            raise ValueError(
                f"Target {hostname} not marked staging. "
                f"Set FLYTO_AI_ALLOW_PROD_TARGETS=1 to override."
            )


def _check_private_ip(hostname: str) -> None:
    """Reject RFC 1918 and link-local addresses."""
    # Quick check for obvious private IP patterns
    private_prefixes = (
        "10.",
        "192.168.",
        "172.16.", "172.17.", "172.18.", "172.19.",
        "172.20.", "172.21.", "172.22.", "172.23.",
        "172.24.", "172.25.", "172.26.", "172.27.",
        "172.28.", "172.29.", "172.30.", "172.31.",
        "169.254.",
    )
    if any(hostname.startswith(prefix) for prefix in private_prefixes):
        raise ValueError(
            f"Refusing to target private IP {hostname} — SSRF risk"
        )
