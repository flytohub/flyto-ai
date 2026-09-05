"""Caller-owned browser resources, isolated across concurrent async tasks."""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BrowserSessionScope:
    """One caller's runtime registry and metadata-only cleanup receipt."""

    owner_id: str
    sessions: dict[str, Any] = field(default_factory=dict, repr=False)
    launch_failed: bool = False
    launch_error: str = ""
    goto_failures: int = 0
    owned_session_ids: set = field(default_factory=set)
    closed_session_ids: list[str] = field(default_factory=list)
    cleanup_errors: list[str] = field(default_factory=list)
    closing: bool = False
    closed: bool = False

    def receipt(self) -> dict[str, Any]:
        return {
            "status": "failed" if self.cleanup_errors else ("closed" if self.closed else "active"),
            "owned_session_ids": sorted(self.owned_session_ids),
            "closed_session_ids": list(self.closed_session_ids),
            "error_types": list(self.cleanup_errors),
        }


class BrowserCleanupError(RuntimeError):
    """Owned browsers could not all be confirmed closed."""


_SCOPE: ContextVar[BrowserSessionScope | None] = ContextVar("flyto_browser_scope", default=None)


def current_browser_scope() -> BrowserSessionScope | None:
    return _SCOPE.get()


async def _close_sessions(scope: BrowserSessionScope) -> None:
    from flyto_ai.tools.core_tools import _dispatch_core_tool_inner

    scope.owned_session_ids.update(scope.sessions)
    for session_id in list(scope.sessions):
        try:
            # No smart retry: cleanup must never relaunch a dead browser.
            result = await _dispatch_core_tool_inner("execute_module", {
                "module_id": "browser.close", "params": {},
                "context": {"browser_session": session_id},
            })
            ok = result.get("ok") if isinstance(result.get("ok"), bool) else result.get("status") == "success"
            if not ok:
                scope.cleanup_errors.append("CloseNotAcknowledged")
                continue
            scope.sessions.pop(session_id, None)
            if session_id not in scope.closed_session_ids:
                scope.closed_session_ids.append(session_id)
        except Exception as error:  # noqa: BLE001 - attempt every owned close after any runtime failure
            scope.cleanup_errors.append(type(error).__name__)


async def _finish_scope(scope: BrowserSessionScope, timeout: float) -> None:
    try:
        await asyncio.wait_for(_close_sessions(scope), timeout=timeout)
    except TimeoutError:
        scope.cleanup_errors.append("TimeoutError")
    finally:
        scope.closed = True


@asynccontextmanager
async def browser_session_scope(owner_id: str, *, cleanup_timeout: float = 10.0):
    """Close only browsers created in this context, including on cancellation.

    The scope is internal caller authority, never a model tool parameter.
    Existing callers outside a scope keep the legacy adapter registry.
    """
    if not owner_id or not 0 < cleanup_timeout <= 60:
        raise ValueError("A browser scope needs an owner and a bounded cleanup timeout")
    scope = BrowserSessionScope(owner_id=owner_id)
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        scope.closing = True
        closing = asyncio.create_task(_finish_scope(scope, cleanup_timeout))
        try:
            interrupted = False
            while not closing.done():
                try:
                    await asyncio.shield(closing)
                except asyncio.CancelledError:
                    interrupted = True
            await closing
            if interrupted:
                raise asyncio.CancelledError
        finally:
            _SCOPE.reset(token)
        if scope.cleanup_errors:
            raise BrowserCleanupError("Owned browser sessions could not all be confirmed closed")
