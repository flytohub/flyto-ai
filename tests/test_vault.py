# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for encrypted credential vault."""
import os
import tempfile
import shutil

import pytest

from flyto_ai.vault import Vault, redact_vault_values, _derive_key, _get_machine_id


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_machine_id():
    mid = _get_machine_id()
    assert isinstance(mid, str)
    assert len(mid) > 5


def test_derive_key():
    key = _derive_key("passphrase", b"0123456789abcdef")
    assert len(key) == 44  # base64-encoded 32 bytes
    # Same inputs → same key (deterministic)
    key2 = _derive_key("passphrase", b"0123456789abcdef")
    assert key == key2
    # Different passphrase → different key
    key3 = _derive_key("other", b"0123456789abcdef")
    assert key3 != key


def test_vault_set_get():
    vault = Vault.__new__(Vault)
    vault._credentials = {}
    vault.set("KEY1", "value1")
    assert vault.get("KEY1") == "value1"
    assert vault.get("MISSING") is None
    assert vault.get("MISSING", "default") == "default"


def test_vault_delete():
    vault = Vault.__new__(Vault)
    vault._credentials = {"KEY1": "v1"}
    assert vault.delete("KEY1") is True
    assert vault.delete("KEY1") is False
    assert vault.get("KEY1") is None


def test_vault_list_keys():
    vault = Vault.__new__(Vault)
    vault._credentials = {"B": "2", "A": "1", "C": "3"}
    assert vault.list_keys() == ["A", "B", "C"]


def test_vault_has():
    vault = Vault.__new__(Vault)
    vault._credentials = {"KEY1": "v1"}
    assert vault.has("KEY1") is True
    assert vault.has("KEY2") is False


def test_vault_count():
    vault = Vault.__new__(Vault)
    vault._credentials = {"A": "1", "B": "2"}
    assert vault.credential_count == 2


# --- Save/Load with encryption ---

@pytest.fixture
def vault_with_data(tmpdir):
    vault = Vault(
        vault_path=os.path.join(tmpdir, "test.enc"),
        passphrase="test-pass",
    )
    vault.set("API_KEY", "sk-test-12345678")
    vault.set("DB_PASS", "super-secret")
    return vault


def test_vault_save_load(vault_with_data, tmpdir):
    vault_with_data.save()

    vault2 = Vault(
        vault_path=os.path.join(tmpdir, "test.enc"),
        passphrase="test-pass",
    )
    assert vault2.load() is True
    assert vault2.get("API_KEY") == "sk-test-12345678"
    assert vault2.get("DB_PASS") == "super-secret"


def test_vault_wrong_passphrase(vault_with_data, tmpdir):
    vault_with_data.save()

    vault2 = Vault(
        vault_path=os.path.join(tmpdir, "test.enc"),
        passphrase="wrong-pass",
    )
    assert vault2.load() is False


def test_vault_file_not_found(tmpdir):
    vault = Vault(
        vault_path=os.path.join(tmpdir, "nonexistent.enc"),
        passphrase="test",
    )
    assert vault.load() is False


def test_vault_file_permissions(vault_with_data):
    vault_with_data.save()
    path = str(vault_with_data._path)
    mode = os.stat(path).st_mode & 0o777
    assert mode == 0o600


# --- Env injection ---

def test_vault_inject_to_env(vault_with_data):
    # Clean up env first
    for k in ["API_KEY", "DB_PASS"]:
        os.environ.pop(k, None)

    count = vault_with_data.inject_to_env()
    assert count == 2
    assert os.environ.get("API_KEY") == "sk-test-12345678"
    assert os.environ.get("DB_PASS") == "super-secret"

    # Cleanup
    vault_with_data.clear_from_env()
    assert "API_KEY" not in os.environ


def test_vault_inject_selective(vault_with_data):
    os.environ.pop("API_KEY", None)
    count = vault_with_data.inject_to_env(["API_KEY"])
    assert count == 1
    assert "API_KEY" in os.environ
    os.environ.pop("API_KEY", None)


# --- Redaction ---

def test_redact_vault_values():
    vault = Vault.__new__(Vault)
    vault._credentials = {"API_KEY": "sk-test-12345678", "SHORT": "ab"}

    text = "The key is sk-test-12345678 and short is ab"
    redacted = redact_vault_values(text, vault)
    assert "sk-test-12345678" not in redacted
    assert "[REDACTED:API_KEY]" in redacted
    # Short values (<8 chars) are NOT redacted (too many false positives)
    assert "ab" in redacted


def test_redact_no_match():
    vault = Vault.__new__(Vault)
    vault._credentials = {"KEY": "not-in-text-12345678"}
    text = "Some normal text without secrets"
    assert redact_vault_values(text, vault) == text
