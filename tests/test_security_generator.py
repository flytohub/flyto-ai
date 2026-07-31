"""Tests for flyto_ai.security — blueprint generation + safety checks."""

import pytest
import yaml

from flyto_ai.security import SecurityFinding, generate_test_from_finding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_finding(**overrides) -> SecurityFinding:
    """Create a SecurityFinding with sensible defaults, overridable."""
    defaults = dict(
        category="sql_injection",
        source="request.args.get('user_id')",
        source_file="handler.py",
        source_line=42,
        sink="cursor.execute(query)",
        sink_file="handler.py",
        sink_line=55,
        severity="critical",
        param_name="user_id",
        endpoint_path="/api/users",
        http_method="GET",
    )
    defaults.update(overrides)
    return SecurityFinding(**defaults)


def _parse_yaml(yaml_str: str) -> dict:
    """Parse YAML string and return dict; fail if invalid."""
    result = yaml.safe_load(yaml_str)
    assert isinstance(result, dict), "YAML did not produce a dict"
    return result


# ---------------------------------------------------------------------------
# SQL Injection blueprint
# ---------------------------------------------------------------------------

class TestSQLInjection:
    def test_generates_valid_yaml(self):
        finding = _make_finding(category="sql_injection")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        workflow = _parse_yaml(yaml_str)
        assert "steps" in workflow
        assert "edges" in workflow

    def test_has_http_batch_step(self):
        finding = _make_finding(category="sql_injection")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        workflow = _parse_yaml(yaml_str)
        modules = [s["module"] for s in workflow["steps"]]
        assert "http.batch" in modules

    def test_has_assertion_step(self):
        finding = _make_finding(category="sql_injection")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        workflow = _parse_yaml(yaml_str)
        modules = [s["module"] for s in workflow["steps"]]
        assert "test.assert_contains" in modules

    def test_includes_payloads_in_requests(self):
        finding = _make_finding(category="sql_injection", param_name="user_id")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        workflow = _parse_yaml(yaml_str)
        batch_step = next(s for s in workflow["steps"] if s["module"] == "http.batch")
        requests = batch_step["params"]["requests"]
        # Baseline + 3 payloads = 4 requests
        assert len(requests) >= 4
        # At least one request should contain a SQL injection payload
        urls = " ".join(r.get("url", "") + r.get("body", "") for r in requests)
        assert "OR" in urls or "SLEEP" in urls or "CONVERT" in urls or "WAITFOR" in urls

    def test_has_timeout(self):
        finding = _make_finding(category="sql_injection")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        workflow = _parse_yaml(yaml_str)
        assert workflow.get("timeout", 0) > 0

    def test_uses_param_name_in_url(self):
        finding = _make_finding(category="sql_injection", param_name="search_q")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        assert "search_q" in yaml_str

    def test_post_method_uses_body(self):
        finding = _make_finding(category="sql_injection", http_method="POST")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        workflow = _parse_yaml(yaml_str)
        batch_step = next(s for s in workflow["steps"] if s["module"] == "http.batch")
        post_requests = [r for r in batch_step["params"]["requests"] if r["method"] == "POST"]
        assert len(post_requests) >= 1
        assert any("body" in r for r in post_requests)


# ---------------------------------------------------------------------------
# XSS Reflected blueprint
# ---------------------------------------------------------------------------

class TestXSSReflected:
    def test_generates_valid_yaml(self):
        finding = _make_finding(category="xss_reflected")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        workflow = _parse_yaml(yaml_str)
        assert "steps" in workflow
        assert "edges" in workflow

    def test_has_browser_launch(self):
        finding = _make_finding(category="xss_reflected")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        workflow = _parse_yaml(yaml_str)
        modules = [s["module"] for s in workflow["steps"]]
        assert "browser.launch" in modules

    def test_has_browser_evaluate(self):
        finding = _make_finding(category="xss_reflected")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        workflow = _parse_yaml(yaml_str)
        modules = [s["module"] for s in workflow["steps"]]
        assert "browser.evaluate" in modules

    def test_has_screenshot_step(self):
        finding = _make_finding(category="xss_reflected")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        workflow = _parse_yaml(yaml_str)
        modules = [s["module"] for s in workflow["steps"]]
        assert "browser.screenshot" in modules

    def test_includes_xss_payloads(self):
        finding = _make_finding(category="xss_reflected")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        assert "<script>" in yaml_str or "onerror=" in yaml_str or "onload=" in yaml_str


# ---------------------------------------------------------------------------
# Auth Bypass blueprint
# ---------------------------------------------------------------------------

class TestAuthBypass:
    def test_generates_valid_yaml(self):
        finding = _make_finding(category="auth_bypass")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        workflow = _parse_yaml(yaml_str)
        assert "steps" in workflow
        assert "edges" in workflow

    def test_no_browser_needed(self):
        finding = _make_finding(category="auth_bypass")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        workflow = _parse_yaml(yaml_str)
        modules = [s["module"] for s in workflow["steps"]]
        assert "browser.launch" not in modules

    def test_includes_multiple_auth_probes(self):
        finding = _make_finding(category="auth_bypass")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        workflow = _parse_yaml(yaml_str)
        batch_step = next(s for s in workflow["steps"] if s["module"] == "http.batch")
        requests = batch_step["params"]["requests"]
        # Baseline + 4 bypass attempts = 5 requests
        assert len(requests) >= 5

    def test_includes_forged_jwt(self):
        finding = _make_finding(category="auth_bypass")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        # alg:none JWT should be present
        assert "eyJhbGciOiJub25lIi" in yaml_str

    def test_has_assert_status_step(self):
        finding = _make_finding(category="auth_bypass")
        yaml_str = generate_test_from_finding(finding, "https://staging.example.com")
        workflow = _parse_yaml(yaml_str)
        modules = [s["module"] for s in workflow["steps"]]
        assert "test.assert_status" in modules


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------

class TestSafetyChecks:
    def test_refuses_production_target(self):
        finding = _make_finding()
        with pytest.raises(ValueError, match="staging"):
            generate_test_from_finding(finding, "https://production.example.com")

    def test_refuses_metadata_endpoint_gcp(self):
        finding = _make_finding()
        with pytest.raises(ValueError, match="SSRF"):
            generate_test_from_finding(finding, "http://169.254.169.254/latest/meta-data")

    def test_refuses_metadata_endpoint_gcp_hostname(self):
        finding = _make_finding()
        with pytest.raises(ValueError, match="SSRF"):
            generate_test_from_finding(finding, "http://metadata.google.internal/computeMetadata/v1/")

    def test_refuses_private_ip_10(self):
        finding = _make_finding()
        with pytest.raises(ValueError, match="SSRF"):
            generate_test_from_finding(finding, "http://10.0.0.1/api")

    def test_refuses_private_ip_192(self):
        finding = _make_finding()
        with pytest.raises(ValueError, match="SSRF"):
            generate_test_from_finding(finding, "http://192.168.1.1/api")

    def test_allows_staging_url(self):
        finding = _make_finding()
        # Should not raise
        result = generate_test_from_finding(finding, "https://staging.example.com")
        assert result  # non-empty YAML

    def test_allows_localhost(self):
        finding = _make_finding()
        result = generate_test_from_finding(finding, "http://localhost:3000")
        assert result

    def test_allows_prod_with_env_override(self, monkeypatch):
        monkeypatch.setenv("FLYTO_AI_ALLOW_PROD_TARGETS", "1")
        finding = _make_finding()
        result = generate_test_from_finding(finding, "https://production.example.com")
        assert result

    def test_allows_non_staging_target_with_verified_authorization(self):
        finding = _make_finding()
        result = generate_test_from_finding(
            finding,
            "https://production.example.com",
            authorization_verified=True,
        )
        assert result

    @pytest.mark.parametrize(
        "target",
        [
            "http://169.254.169.254/latest/meta-data",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://10.0.0.1/api",
        ],
    )
    def test_verified_authorization_never_bypasses_ssrf_guards(self, target):
        finding = _make_finding()
        with pytest.raises(ValueError, match="SSRF"):
            generate_test_from_finding(
                finding,
                target,
                authorization_verified=True,
            )

    def test_refuses_invalid_url(self):
        finding = _make_finding()
        with pytest.raises(ValueError, match="Invalid URL"):
            generate_test_from_finding(finding, "not-a-url")

    def test_refuses_ftp_scheme(self):
        finding = _make_finding()
        with pytest.raises(ValueError, match="Unsupported scheme"):
            generate_test_from_finding(finding, "ftp://staging.example.com/file")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unknown_category_raises(self):
        finding = _make_finding(category="unknown_vuln")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="No blueprint"):
            generate_test_from_finding(finding, "https://staging.example.com")

    def test_unknown_category_lists_available(self):
        finding = _make_finding(category="ldap_injection")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="sql_injection"):
            generate_test_from_finding(finding, "https://staging.example.com")


# ---------------------------------------------------------------------------
# Cross-cutting: auth token passthrough
# ---------------------------------------------------------------------------

class TestAuthTokenPassthrough:
    def test_sql_injection_with_auth_token(self):
        finding = _make_finding(category="sql_injection")
        yaml_str = generate_test_from_finding(
            finding, "https://staging.example.com", auth_token="my-secret-token"
        )
        assert "my-secret-token" in yaml_str

    def test_xss_with_auth_token(self):
        finding = _make_finding(category="xss_reflected")
        yaml_str = generate_test_from_finding(
            finding, "https://staging.example.com", auth_token="xss-token"
        )
        assert "xss-token" in yaml_str

    def test_auth_bypass_with_auth_token(self):
        finding = _make_finding(category="auth_bypass")
        yaml_str = generate_test_from_finding(
            finding, "https://staging.example.com", auth_token="valid-token"
        )
        assert "valid-token" in yaml_str
