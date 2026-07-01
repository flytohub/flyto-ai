# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Encrypted credential vault — secure storage for API keys and secrets.

Uses Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256).
Master key derived from a user-provided passphrase via PBKDF2.
Falls back to a machine-specific key if no passphrase is set.
"""
import base64
import hashlib
import json
import logging
import os
import platform
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_VAULT_PATH = "~/.flyto/vault.enc"
_SALT_FILE = "~/.flyto/.vault_salt"


def _get_machine_id() -> str:
    """Get a machine-specific identifier for default key derivation."""
    parts = [
        platform.node(),
        os.getenv("USER", os.getenv("USERNAME", "flyto")),
        platform.machine(),
    ]
    return ":".join(parts)


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a Fernet key from a passphrase using PBKDF2."""
    dk = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 100_000, dklen=32)
    return base64.urlsafe_b64encode(dk)


def _get_or_create_salt(salt_path: str) -> bytes:
    """Load or create the salt file."""
    path = Path(os.path.expanduser(salt_path))
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    salt = os.urandom(16)
    path.write_bytes(salt)
    # Restrict permissions (owner-only)
    try:
        os.chmod(str(path), 0o600)
    except OSError:
        pass
    return salt


class Vault:
    """Encrypted credential store.

    Credentials are stored as a JSON dict encrypted with Fernet.
    The vault file is a single encrypted blob (not per-key encryption)
    to keep the implementation simple and the attack surface small.

    Usage::

        vault = Vault()
        vault.set("OPENAI_API_KEY", "sk-...")
        vault.save()

        # Later:
        vault = Vault()
        vault.load()
        key = vault.get("OPENAI_API_KEY")
    """

    def __init__(
        self,
        vault_path: Optional[str] = None,
        passphrase: Optional[str] = None,
    ) -> None:
        self._path = Path(os.path.expanduser(vault_path or _DEFAULT_VAULT_PATH))
        self._credentials: Dict[str, str] = {}

        # Derive encryption key
        salt = _get_or_create_salt(_SALT_FILE)
        phrase = passphrase or os.getenv("FLYTO_VAULT_PASSPHRASE", "") or _get_machine_id()
        self._key = _derive_key(phrase, salt)
        self._fernet = None

    def _get_fernet(self):
        """Lazy-init Fernet cipher."""
        if self._fernet is None:
            try:
                from cryptography.fernet import Fernet
                self._fernet = Fernet(self._key)
            except ImportError:
                raise ImportError(
                    "cryptography package required for vault. "
                    "Install with: pip install cryptography"
                )
        return self._fernet

    def load(self) -> bool:
        """Load and decrypt the vault file. Returns True if loaded."""
        if not self._path.exists():
            logger.debug("Vault file not found: %s", self._path)
            return False

        try:
            encrypted = self._path.read_bytes()
            fernet = self._get_fernet()
            decrypted = fernet.decrypt(encrypted)
            self._credentials = json.loads(decrypted)
            logger.info("Vault loaded: %d credentials", len(self._credentials))
            return True
        except Exception as e:
            logger.warning("Vault load failed (wrong passphrase?): %s", e)
            return False

    def save(self) -> None:
        """Encrypt and save the vault file."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fernet = self._get_fernet()
            plaintext = json.dumps(self._credentials, ensure_ascii=False).encode()
            encrypted = fernet.encrypt(plaintext)
            self._path.write_bytes(encrypted)
            # Restrict permissions (owner-only)
            try:
                os.chmod(str(self._path), 0o600)
            except OSError:
                pass
            logger.info("Vault saved: %d credentials", len(self._credentials))
        except Exception as e:
            logger.warning("Vault save failed: %s", e)
            raise

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a credential by key."""
        return self._credentials.get(key, default)

    def set(self, key: str, value: str) -> None:
        """Set a credential."""
        self._credentials[key] = value

    def delete(self, key: str) -> bool:
        """Delete a credential. Returns True if it existed."""
        if key in self._credentials:
            del self._credentials[key]
            return True
        return False

    def list_keys(self) -> List[str]:
        """List all credential keys (values are never exposed)."""
        return sorted(self._credentials.keys())

    def has(self, key: str) -> bool:
        """Check if a credential exists."""
        return key in self._credentials

    def inject_to_env(self, keys: Optional[List[str]] = None) -> int:
        """Inject credentials as environment variables.

        Parameters
        ----------
        keys : list, optional
            Specific keys to inject. If None, injects all.

        Returns
        -------
        int
            Number of environment variables set.
        """
        target_keys = keys or list(self._credentials.keys())
        count = 0
        for key in target_keys:
            if key in self._credentials:
                os.environ[key] = self._credentials[key]
                count += 1
        if count:
            logger.info("Vault: injected %d credentials to env", count)
        return count

    def clear_from_env(self, keys: Optional[List[str]] = None) -> int:
        """Remove vault credentials from environment variables.

        Parameters
        ----------
        keys : list, optional
            Specific keys to clear. If None, clears all vault keys.

        Returns
        -------
        int
            Number of environment variables cleared.
        """
        target_keys = keys or list(self._credentials.keys())
        count = 0
        for key in target_keys:
            if key in os.environ and key in self._credentials:
                del os.environ[key]
                count += 1
        return count

    @property
    def credential_count(self) -> int:
        """Number of stored credentials."""
        return len(self._credentials)


def redact_vault_values(text: str, vault: Vault) -> str:
    """Redact any vault credential values found in text.

    Scans text for exact matches of stored credential values
    and replaces them with [REDACTED:<key>].
    """
    result = text
    for key in vault.list_keys():
        value = vault.get(key)
        if value and len(value) >= 8 and value in result:
            result = result.replace(value, "[REDACTED:{}]".format(key))
    return result
