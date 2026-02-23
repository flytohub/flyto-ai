# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for sensitive key detection and redaction."""
from flyto_ai.redaction import is_sensitive_key, redact_args


class TestIsSensitiveKey:

    def test_password(self):
        assert is_sensitive_key("password") is True

    def test_api_key(self):
        assert is_sensitive_key("api_key") is True

    def test_partial_match(self):
        assert is_sensitive_key("user_password_hash") is True

    def test_case_insensitive(self):
        assert is_sensitive_key("API_KEY") is True
        assert is_sensitive_key("Authorization") is True

    def test_safe_key(self):
        assert is_sensitive_key("url") is False
        assert is_sensitive_key("name") is False
        assert is_sensitive_key("selector") is False


class TestRedactArgs:

    def test_redact_dict(self):
        result = redact_args({"url": "https://x.com", "password": "secret"})
        assert result["url"] == "https://x.com"
        assert result["password"] == "***"

    def test_nested_dict(self):
        result = redact_args({"data": {"api_key": "sk-123", "name": "test"}})
        assert result["data"]["api_key"] == "***"
        assert result["data"]["name"] == "test"

    def test_list(self):
        result = redact_args([{"token": "abc"}, {"url": "x"}])
        assert result[0]["token"] == "***"
        assert result[1]["url"] == "x"

    def test_non_dict(self):
        assert redact_args("hello") == "hello"
        assert redact_args(42) == 42

    def test_empty(self):
        assert redact_args({}) == {}
        assert redact_args([]) == []
