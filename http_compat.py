from __future__ import annotations

from typing import Any

try:
    from fastapi import HTTPException, UploadFile
except ImportError:
    class HTTPException(Exception):
        """Small fallback so business modules stay importable without FastAPI."""

        def __init__(self, status_code: int, detail: Any = None, headers: dict[str, str] | None = None) -> None:
            self.status_code = status_code
            self.detail = detail
            self.headers = headers
            super().__init__(str(detail))

    class UploadFile:  # pragma: no cover - runtime uses FastAPI's UploadFile when installed.
        filename: str | None = None
        content_type: str | None = None

        async def read(self) -> bytes:
            raise RuntimeError("FastAPI UploadFile is unavailable in this environment")
