# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Cardinality-independent durable reads for ``flyto_coding_get``.

The coding worker has to reconcile the complete durable control plane before
it may mutate anything.  A status read does not.  This module gives the stable
MCP supervisor a narrower boundary: open exactly one authenticated tenant/job
record, validate the same public receipt and execution authority as the worker,
and return a secret-redacted projection without constructing a service.

There are two deliberate fallbacks.  A terminal mission whose projection is
still ready or dispatched may require the worker's existing reconciliation
mutation, and a platform without descriptor-relative no-follow opens cannot
offer this path safely.  Both go through the canonical worker; neither is
silently approximated here.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from flyto_ai.coding.continuation import (
    MAX_CONTINUATION_GENERATION,
    ContinuationStore,
    is_continuable_session,
)
from flyto_ai.coding.contracts import (
    MISSION_STATUS_DISPATCHED,
    MISSION_STATUS_READY,
    TERMINAL_CODING_JOB_STATES,
)
from flyto_ai.coding.mcp_contract import (
    tool_error,
    tool_job_result,
    validated_response,
)
from flyto_ai.coding.service import (
    _ROUTE_EVIDENCE_STATES,
    CodingJobNotFound,
    CodingService,
    CodingServiceError,
    error_details,
    receipt_to_mapping,
    require_execution_authority_record,
)

MAX_DURABLE_JOB_RECORD_BYTES = 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024
_JOB_ID_RE = re.compile(r"^job_[a-f0-9]{24}$")
_SAFE_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_BACKEND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_CODING_BACKENDS = frozenset({"native", "claude", "codex"})
_CODING_BACKEND_ENV = "FLYTO_AI_CODING_BACKEND"
_TERMINAL_STATES = frozenset(state.value for state in TERMINAL_CODING_JOB_STATES)
_MISSION_RECONCILIATION_STATES = frozenset({
    MISSION_STATUS_READY,
    MISSION_STATUS_DISPATCHED,
})
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


class DurableGetFallback(RuntimeError):
    """The exact read is safe, but canonical worker behavior may mutate state."""


class DurableGetUnavailable(RuntimeError):
    """The exact record cannot be read through the no-follow bounded contract."""


class DurableJobReceiptReader:
    """Read one tenant-local job receipt without enumerating durable history."""

    def __init__(
        self,
        state_dir: str,
        tenant_id: str,
        *,
        implementation_backend: str,
        emergency_enabled: bool = False,
    ) -> None:
        if not state_dir:
            raise ValueError("state_dir is required")
        if not _SAFE_TENANT_RE.fullmatch(tenant_id or ""):
            raise ValueError("tenant_id must be a safe identifier")
        if not _SAFE_BACKEND_RE.fullmatch(implementation_backend or ""):
            raise ValueError("implementation_backend must be a safe identifier")
        if not isinstance(emergency_enabled, bool):
            raise TypeError("emergency_enabled must be a boolean")
        expanded = os.path.expanduser(state_dir)
        self.state_root = Path(os.path.abspath(expanded))
        self.tenant_ref = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
        self.implementation_backend = implementation_backend
        self.emergency_enabled = emergency_enabled
        self._continuation = ContinuationStore(self.state_root)

    def read(self, job_id: str) -> dict[str, Any]:
        """Return the canonical public mapping for one exact job id."""

        if not _JOB_ID_RE.fullmatch(job_id or ""):
            raise CodingJobNotFound("coding job does not exist")
        record = self._read_record(job_id)
        if record.get("job_id") != job_id:
            raise ValueError("coding service record does not match its job id")
        state = str(record.get("state") or "")
        mission = record.get("mission")
        if (
            state in _TERMINAL_STATES
            and isinstance(mission, Mapping)
            and mission.get("status") in _MISSION_RECONCILIATION_STATES
        ):
            # The ordinary get may reclaim the exact mission work item, persist
            # its ready projection, and dispatch deferred closure accounting.
            # That mutation stays exclusively in CodingService.
            raise DurableGetFallback("terminal mission reconciliation is required")
        if state in _ROUTE_EVIDENCE_STATES or record.get("landable") is True:
            require_execution_authority_record(
                record,
                strict_route=True,
                emergency_enabled=self.emergency_enabled,
                implementation_backend=self.implementation_backend,
            )
        receipt = CodingService._receipt(record)
        session = str(record.get("continuation_session_id") or "")
        if is_continuable_session(session):
            try:
                authority = self._continuation.open_authority(self.tenant_ref, session)
            except (OSError, RuntimeError, ValueError):
                authority = None
            if authority is not None:
                receipt = dataclasses.replace(
                    receipt,
                    continuation_available=(
                        authority.generation < MAX_CONTINUATION_GENERATION
                    ),
                    continuation_generation=authority.generation,
                )
        return receipt_to_mapping(receipt)

    def _read_record(self, job_id: str) -> dict[str, Any]:
        directory = self._open_jobs_directory()
        try:
            try:
                handle = os.open(
                    job_id + ".json",
                    os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC,
                    dir_fd=directory,
                )
            except FileNotFoundError as exc:
                raise CodingJobNotFound("coding job does not exist") from exc
            except OSError as exc:
                raise DurableGetUnavailable("coding job record cannot be opened safely") from exc
        finally:
            os.close(directory)
        try:
            opened = os.fstat(handle)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_uid != os.geteuid()
                or opened.st_mode & 0o077
            ):
                raise DurableGetUnavailable("coding job record is not a private file")
            if opened.st_size > MAX_DURABLE_JOB_RECORD_BYTES:
                raise DurableGetUnavailable("coding job record exceeds its bound")
            chunks = []
            total = 0
            while True:
                chunk = os.read(handle, _READ_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DURABLE_JOB_RECORD_BYTES:
                    raise DurableGetUnavailable("coding job record exceeds its bound")
                chunks.append(chunk)
            after = os.fstat(handle)
            identity = (
                opened.st_dev,
                opened.st_ino,
                opened.st_mode,
                opened.st_nlink,
                opened.st_uid,
                opened.st_gid,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            if total != after.st_size or identity != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_uid,
                after.st_gid,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise DurableGetUnavailable("coding job record changed while it was read")
            raw = b"".join(chunks)
        finally:
            os.close(handle)
        try:
            value = json.loads(
                raw.decode("utf-8"),
                parse_constant=_reject_nonfinite,
                parse_float=_finite_float,
                object_pairs_hook=_unique_object,
            )
        except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise ValueError("coding service record is invalid") from exc
        if not isinstance(value, dict):
            raise ValueError("coding service record must be an object")  # noqa: TRY004
        return value

    def _open_jobs_directory(self) -> int:
        if not (_O_NOFOLLOW and _O_DIRECTORY):
            raise DurableGetUnavailable("this platform cannot refuse state symlinks")
        parts = list(PurePosixPath(str(self.state_root)).parts[1:]) + [
            "tenants",
            self.tenant_ref,
            "jobs",
        ]
        handle = os.open(
            "/",
            os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
        )
        try:
            for part in parts:
                if part in {"", ".", ".."}:
                    raise DurableGetUnavailable("coding state path is not canonical")
                try:
                    nested = os.open(
                        part,
                        os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC,
                        dir_fd=handle,
                    )
                except FileNotFoundError as exc:
                    raise CodingJobNotFound("coding job does not exist") from exc
                except OSError as exc:
                    raise DurableGetUnavailable(
                        "coding state directory cannot be opened safely",
                    ) from exc
                os.close(handle)
                handle = nested
            info = os.fstat(handle)
            if info.st_uid != os.geteuid() or info.st_mode & 0o077:
                raise DurableGetUnavailable("coding jobs directory is not private")
            return handle
        except BaseException:
            os.close(handle)
            raise


class DurableJobReceiptResponder:
    """Encode one exact reader result while its imported build stays current."""

    def __init__(
        self,
        reader: DurableJobReceiptReader,
        build_id_provider: Callable[[], str],
        max_response_bytes: int,
    ) -> None:
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.reader = reader
        self._build_id_provider = build_id_provider
        self._loaded_build_id = build_id_provider()
        self._max_response_bytes = max_response_bytes
        self._stale = False

    def respond(self, request: Mapping[str, Any]) -> bytes | None:
        """Return one validated MCP response, or defer to the canonical worker."""

        if (
            request.get("jsonrpc") != "2.0"
            or request.get("id") is None
            or _tool_name(request) != "flyto_coding_get"
        ):
            return None
        params = request.get("params")
        arguments = params.get("arguments") if isinstance(params, Mapping) else None
        if not isinstance(arguments, Mapping) or set(arguments) != {"job_id"}:
            return None
        if self._stale or self._build_id_provider() != self._loaded_build_id:
            self._stale = True
            return None
        job_id = str(arguments.get("job_id", ""))
        try:
            job = self.reader.read(job_id)
        except DurableGetFallback:
            return None
        except CodingJobNotFound:
            return tool_error(request.get("id"), "job_not_found")
        except CodingServiceError as exc:
            return tool_error(request.get("id"), exc.code, error_details(exc))
        except (DurableGetUnavailable, KeyError, OverflowError, TypeError, ValueError):
            return None
        response = tool_job_result(request.get("id"), job)
        if len(response) > self._max_response_bytes:
            return None
        validated_response(response, request, "durable get response")
        return response


def durable_job_receipt_reader(
    state_dir: str,
    tenant_id: str,
    *,
    implementation_backend: str,
    emergency_enabled: bool = False,
) -> DurableJobReceiptReader | None:
    """Build the exact reader, or disable it when safe opens are unavailable."""

    if not (_O_NOFOLLOW and _O_DIRECTORY):
        return None
    try:
        return DurableJobReceiptReader(
            state_dir,
            tenant_id,
            implementation_backend=implementation_backend,
            emergency_enabled=emergency_enabled,
        )
    except (RuntimeError, TypeError, ValueError):
        return None


def durable_job_receipt_reader_from_argv(
    argv: Sequence[str],
    *,
    default_state_dir: str,
    environ: Mapping[str, str] | None = None,
) -> DurableJobReceiptReader | None:
    """Bind a reader to the same scalar startup choices argparse will use."""

    environment = os.environ if environ is None else environ
    state_dir = _last_option_value(argv, "--state-dir") or default_state_dir
    tenant_id = _last_option_value(argv, "--tenant")
    backend = (
        _last_option_value(argv, "--implementation-backend")
        or environment.get(_CODING_BACKEND_ENV, "").strip()
        or "native"
    )
    emergency_backend = _last_option_value(argv, "--emergency-overflow-backend")
    if backend not in _CODING_BACKENDS or (
        emergency_backend and emergency_backend != backend
    ):
        return None
    return durable_job_receipt_reader(
        state_dir,
        tenant_id,
        implementation_backend=backend,
        emergency_enabled=bool(emergency_backend),
    )


def _last_option_value(argv: Sequence[str], name: str) -> str:
    """Return the last scalar flag spelling, matching argparse replacement."""

    items = tuple(argv)
    value = ""
    for index, item in enumerate(items):
        if item == name and index + 1 < len(items):
            value = items[index + 1]
        elif item.startswith(name + "="):
            value = item[len(name) + 1 :]
    return value


def _tool_name(request: Mapping[str, Any]) -> str:
    if request.get("method") != "tools/call":
        return ""
    params = request.get("params")
    name = params.get("name") if isinstance(params, Mapping) else None
    return name if isinstance(name, str) else ""


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number: {value}")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value
