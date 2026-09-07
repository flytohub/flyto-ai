"""Explicit local-model authority, bounded data and safe error classifications."""

import ipaddress
import math
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from flyto_ai.cli_runtime.contracts import CliRuntimeError, valid_model_id


class LocalModelError(CliRuntimeError):
    """A safe code; never provider response text or host credentials."""


def local_endpoint(value, provider):
    """Canonicalize literal loopback only, without DNS, credentials or redirects."""
    try:
        if not isinstance(value, str) or len(value) > 512 or any(ord(c) <= 32 for c in value):
            raise ValueError
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or url.username is not None or url.password is not None or url.query or url.fragment:
            raise ValueError
        host = url.hostname
        if host == "localhost":
            host = "127.0.0.1"
        address = ipaddress.ip_address(host)
        if not address.is_loopback or "%" in host or getattr(address, "ipv4_mapped", None):
            raise ValueError
        port = url.port
        if port is not None and port < 1:
            raise ValueError
        path = url.path.rstrip("/")
        if path not in {"", "/v1"}:
            raise ValueError
        authority = f"[{address}]" if address.version == 6 else str(address)
        if port:
            authority += f":{port}"
        path = "/v1" if provider == "openai_compatible" else ""
        return urlunsplit((url.scheme, authority, path, "", ""))
    except (TypeError, ValueError) as exc:
        raise LocalModelError("local_model_invalid_endpoint") from exc


@dataclass(frozen=True)
class LocalModelConfig:
    provider: Literal["ollama", "openai_compatible"]
    endpoint: str
    model: str
    timeout_seconds: float = 100.0

    def __post_init__(self):
        if self.provider not in {"ollama", "openai_compatible"}:
            raise LocalModelError("local_model_unsupported_provider")
        if not valid_model_id(self.model, allow_empty=False):
            raise LocalModelError("local_model_invalid_model")
        value = self.timeout_seconds
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.1 <= value <= 300:
            raise LocalModelError("local_model_invalid_timeout")
        object.__setattr__(self, "endpoint", local_endpoint(self.endpoint, self.provider))

    @property
    def source(self):
        return "local_ai"
