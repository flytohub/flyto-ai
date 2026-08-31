# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Cheap, fail-closed cache for durable workspace demand."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from flyto_ai.coding.workspace_authority import workspace_digest


_CACHE_NAME = ".workspace-demand.json"
_CACHE_VERSION = "flyto.coding-workspace-demand.v1"
_JOB_ID = re.compile(r"^job_[a-f0-9]{24}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TENANT_REF = re.compile(r"^[a-f0-9]{64}$")


class WorkspaceDemandCache:
    """Cache the exact workspace digests required by one state inventory.

    The caller holds the coding service's cross-process state guard while using
    this object. Cache ambiguity is always a miss; it can never authorize a
    workspace release.
    """

    def __init__(self, state_root: Path, *, audit_required: bool) -> None:
        self._state_root = state_root
        self._audit_required = bool(audit_required)

    @property
    def path(self) -> Path:
        return self._state_root / _CACHE_NAME

    def inventory_sha256(self) -> Optional[str]:
        """Fingerprint names that change when durable workspace demand changes."""

        digest = hashlib.sha256()
        digest.update(b"flyto.coding-workspace-demand-inventory.v1\n")
        digest.update(b"audit\n" if self._audit_required else b"legacy\n")
        try:
            if not self._add_jobs_inventory(digest):
                return None
            if not self._add_claims_inventory(digest):
                return None
        except OSError:
            return None
        return digest.hexdigest()

    def load(self, inventory_sha256: str) -> Optional[set[str]]:
        """Read one exact cache projection, or require a full durable scan."""

        try:
            if self.path.is_symlink():
                return None
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        expected = {
            "cache_version", "audit_required", "inventory_sha256",
            "required_workspace_sha256s",
        }
        if not isinstance(value, dict) or set(value) != expected:
            return None
        required = value.get("required_workspace_sha256s")
        if (
            value.get("cache_version") != _CACHE_VERSION
            or value.get("audit_required") is not self._audit_required
            or value.get("inventory_sha256") != inventory_sha256
            or not isinstance(required, list)
            or any(not self._valid_digest(item) for item in required)
            or required != sorted(set(required))
        ):
            return None
        return set(required)

    def store(self, inventory_sha256: str, required: Sequence[Path]) -> set[str]:
        """Publish the exact digest set produced by one authoritative scan."""

        desired = sorted({workspace_digest(repository) for repository in required})
        self._write({
            "cache_version": _CACHE_VERSION,
            "audit_required": self._audit_required,
            "inventory_sha256": inventory_sha256,
            "required_workspace_sha256s": desired,
        })
        return set(desired)

    def _add_jobs_inventory(self, digest) -> bool:
        tenants = self._state_root / "tenants"
        if tenants.is_symlink():
            return False
        if not tenants.is_dir():
            return True
        with os.scandir(tenants) as entries:
            tenant_entries = sorted(entries, key=lambda item: item.name)
        for tenant in tenant_entries:
            if not tenant.is_dir(follow_symlinks=False):
                return False
            if not _TENANT_REF.fullmatch(tenant.name):
                return False
            if not self._add_tenant_jobs(digest, tenant.name, Path(tenant.path)):
                return False
        return True

    def _add_tenant_jobs(self, digest, tenant_name: str, tenant: Path) -> bool:
        jobs = tenant / "jobs"
        if jobs.is_symlink():
            return False
        if not jobs.is_dir():
            return True
        if not self._audit_required:
            info = jobs.stat()
            self._add_stat(digest, "jobs-stat/" + tenant_name, info)
        with os.scandir(jobs) as entries:
            job_entries = sorted(
                (entry for entry in entries if entry.name.endswith(".json")),
                key=lambda item: item.name,
            )
        for entry in job_entries:
            job_id = entry.name[: -len(".json")]
            if not entry.is_file(follow_symlinks=False):
                return False
            if not _JOB_ID.fullmatch(job_id):
                return False
            digest.update(f"job/{tenant_name}/{entry.name}\n".encode("ascii"))
        return True

    def _add_claims_inventory(self, digest) -> bool:
        claims = self._state_root / "locks" / "workspaces"
        if claims.is_symlink():
            return False
        if not claims.is_dir():
            return True
        if not self._audit_required:
            self._add_stat(digest, "claims-stat", claims.stat())
        with os.scandir(claims) as entries:
            claim_entries = sorted(
                (entry for entry in entries if entry.name.endswith(".owner.json")),
                key=lambda item: item.name,
            )
        for claim in claim_entries:
            name = claim.name[: -len(".owner.json")]
            if not claim.is_file(follow_symlinks=False):
                return False
            if not _SHA256_RE.fullmatch(name):
                return False
            digest.update(("claim/" + claim.name + "\n").encode("ascii"))
        return True

    @staticmethod
    def _add_stat(digest, prefix: str, info: os.stat_result) -> None:
        digest.update(
            f"{prefix}/{info.st_ino}/{info.st_mtime_ns}/{info.st_ctime_ns}\n".encode(
                "ascii"
            )
        )

    @staticmethod
    def _valid_digest(value: object) -> bool:
        return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))

    def _write(self, value: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        fd, temporary = tempfile.mkstemp(
            prefix=".workspace-demand-", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
