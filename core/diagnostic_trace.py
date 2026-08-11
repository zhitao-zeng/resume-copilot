"""Opt-in, case-correlated diagnostic logging for platform evaluations.

The diagnostic image intentionally records the complete fictional business
payload used by the benchmark. Authentication material is never passed to this
module and credential-like mapping keys are suppressed defensively.
"""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar, Token
from dataclasses import asdict, is_dataclass
from typing import Any


_logger = logging.getLogger("resume_diagnostic")
_trace_id: ContextVar[str] = ContextVar("resume_diagnostic_trace_id", default="unbound")
_SECRET_KEY_PARTS = (
    "authorization",
    "cookie",
    "password",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "bearer",
)


def diagnostic_trace_enabled() -> bool:
    return os.getenv("RESUME_DIAGNOSTIC_TRACE", "0").strip().lower() in {
        "1", "true", "yes", "on", "full",
    }


def set_trace_id(value: str) -> Token:
    return _trace_id.set(str(value or "unbound"))


def reset_trace_id(token: Token) -> None:
    _trace_id.reset(token)


def current_trace_id() -> str:
    return _trace_id.get()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            normalized = key.casefold().replace("-", "_")
            clean[key] = (
                "<credential omitted>"
                if any(part in normalized for part in _SECRET_KEY_PARTS)
                else _jsonable(item)
            )
        return clean
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def trace_event(event: str, **payload: Any) -> None:
    """Emit one parseable JSON event when the diagnostic switch is enabled."""

    if not diagnostic_trace_enabled():
        return
    record = {
        "trace_id": current_trace_id(),
        "event": str(event),
        **{key: _jsonable(value) for key, value in payload.items()},
    }
    _logger.info(
        "RESUME_DIAG %s",
        json.dumps(record, ensure_ascii=False, separators=(",", ":")),
    )
