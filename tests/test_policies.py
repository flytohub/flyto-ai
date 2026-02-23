# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for policy allowlist checks."""
from flyto_ai.prompt.policies import (
    is_module_allowed,
    is_tool_allowed,
    validate_base_url,
    get_default_policies,
)


class TestValidateBaseUrl:

    def test_valid_openai(self):
        assert validate_base_url("https://api.openai.com/v1") is True

    def test_valid_groq(self):
        assert validate_base_url("https://api.groq.com") is True

    def test_invalid_http(self):
        assert validate_base_url("http://api.openai.com") is False

    def test_invalid_domain(self):
        assert validate_base_url("https://evil.com") is False

    def test_azure_wildcard(self):
        assert validate_base_url("https://myorg.openai.azure.com") is True

    def test_empty(self):
        assert validate_base_url("") is False

    def test_with_custom_policies(self):
        policies = {"allowed_domains": ["custom.api.com"]}
        assert validate_base_url("https://custom.api.com", policies) is True
        assert validate_base_url("https://api.openai.com", policies) is False


class TestIsToolAllowed:

    def test_allowed(self):
        assert is_tool_allowed("list_modules") is True
        assert is_tool_allowed("inspect_page") is True

    def test_not_allowed(self):
        assert is_tool_allowed("evil_tool") is False

    def test_custom_policies(self):
        policies = {"allowed_tools": ["custom_tool"]}
        assert is_tool_allowed("custom_tool", policies) is True
        assert is_tool_allowed("list_modules", policies) is False


class TestIsModuleAllowed:

    def test_browser_allowed(self):
        assert is_module_allowed("browser.click") is True

    def test_file_not_allowed(self):
        assert is_module_allowed("file.read") is False

    def test_api_allowed(self):
        assert is_module_allowed("api.get") is True

    def test_custom_policies(self):
        policies = {"allowed_categories": ["custom"]}
        assert is_module_allowed("custom.tool", policies) is True
        assert is_module_allowed("browser.click", policies) is False


class TestGetDefaultPolicies:

    def test_returns_all_keys(self):
        policies = get_default_policies()
        assert "allowed_domains" in policies
        assert "allowed_tools" in policies
        assert "allowed_categories" in policies

    def test_sorted(self):
        policies = get_default_policies()
        assert policies["allowed_domains"] == sorted(policies["allowed_domains"])
