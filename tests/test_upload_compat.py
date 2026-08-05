"""Regression tests for the upload compatibility layer.

Covers the platform failure where EVERY submission reported "file too large":
`_MockUpload.read()` ignored the `size` argument and re-opened the file from
byte 0 each call. `read_upload_limited` then fell back to a no-arg `read()`
per iteration, re-reading the whole file until the cumulative total tripped
MAX_FILE_SIZE — a real 125,735-byte CV was reported as >20MB and all 112
evaluation cases failed with 0 completions.

Fixed by giving `_MockUpload` cursor-based `read(size)` semantics and letting
the compat endpoint reuse already-extracted text (single parse, no re-OCR).
"""
from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path

from resume_generate_api import _MockUpload, app
from security_utils import read_upload_limited

MAX_FILE_SIZE = 20 * 1024 * 1024
REAL_CV_BYTES = 125_735  # actual size of case-1 CV on the platform
REAL_TEMPLATE_BYTES = 654_544


def _run(coro):
    return asyncio.run(coro)


# ── read(size) cursor semantics ──────────────────────────────────────

def test_read_size_advances_cursor_content_mode():
    async def run():
        upload = _MockUpload(None, "cv.txt", content="a" * 1000)
        chunks = []
        while True:
            chunk = await upload.read(256)
            if not chunk:
                break
            chunks.append(chunk)
        assert b"".join(chunks) == b"a" * 1000
        assert upload._offset == 1000  # fully consumed
    _run(run())


def test_read_no_size_returns_remaining_once():
    async def run():
        upload = _MockUpload(None, "cv.txt", content="abcdef")
        first = await upload.read(2)
        rest = await upload.read()
        empty = await upload.read()
        assert first == b"ab"
        assert rest == b"cdef"
        assert empty == b""  # EOF — no repeated full reads
    _run(run())


def test_path_mode_cursor():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"x" * 1000)
        path = f.name
    try:
        async def run():
            upload = _MockUpload(path, "cv")
            total = 0
            while True:
                chunk = await upload.read(300)
                if not chunk:
                    break
                total += len(chunk)
            assert total == 1000
            assert upload._offset == 1000
        _run(run())
    finally:
        Path(path).unlink(missing_ok=True)


# ── read_upload_limited integration (the platform regression) ────────

def test_small_file_not_rejected():
    """A real-size CV must NOT trip the 20MB limit (was the platform bug)."""
    async def run():
        upload = _MockUpload(None, "cv.txt", content="r" * REAL_CV_BYTES)
        data = await read_upload_limited(upload, MAX_FILE_SIZE)
        assert len(data) == REAL_CV_BYTES
    _run(run())


def test_template_size_not_rejected():
    async def run():
        upload = _MockUpload(None, "template.docx", content="t" * REAL_TEMPLATE_BYTES)
        data = await read_upload_limited(upload, MAX_FILE_SIZE)
        assert len(data) == REAL_TEMPLATE_BYTES
    _run(run())


def test_oversize_still_rejected():
    async def run():
        upload = _MockUpload(None, "cv.txt", content="r" * (MAX_FILE_SIZE + 100))
        try:
            await read_upload_limited(upload, MAX_FILE_SIZE)
            return False
        except ValueError:
            return True
    assert _run(run()) is True


def test_chunked_read_single_pass_no_duplicate():
    """The mock must be consumed in ONE pass — sized reads only, no
    no-arg fallback that would re-read the whole file every iteration."""
    sizes: list[int | None] = []

    class TrackingMock:
        def __init__(self, content: str):
            self._data = content.encode("utf-8")
            self.offset = 0

        async def read(self, size: int | None = None):
            sizes.append(size)
            result = self._data[self.offset:self.offset + size] if size is not None else self._data[self.offset:]
            self.offset += len(result)
            return result

    consumed = []

    class TrackingMock:
        def __init__(self, content: str):
            self._data = content.encode("utf-8")
            self.offset = 0

        async def read(self, size: int | None = None):
            sizes.append(size)
            result = self._data[self.offset:self.offset + size] if size is not None else self._data[self.offset:]
            self.offset += len(result)
            consumed.append(len(result))
            return result

    async def run():
        upload = TrackingMock("z" * 5 * 1024 * 1024)
        data = await read_upload_limited(upload, MAX_FILE_SIZE)
        assert len(data) == 5 * 1024 * 1024
        return upload
    upload = _run(run())
    # Every read was size-bounded (no TypeError → no no-arg fallback)
    assert all(s is not None for s in sizes)
    # Payload consumed exactly once — bytes read equal the file, no duplicates
    assert sum(consumed) == 5 * 1024 * 1024
    assert upload.offset == 5 * 1024 * 1024


# ── multipart endpoint acceptance ─────────────────────────────────────

class _FakeFuture:
    def done(self):
        return True

    def cancel(self):
        return True

    def add_done_callback(self, callback):
        callback(self)


def test_resume_optimize_multipart_accepted():
    """POST /resume_optimize with a real-size multipart file must be accepted
    (200 + task_id), not rejected at request time. The background worker is
    patched out so no real vLLM wait starts (it would block test exit)."""
    from unittest.mock import patch

    from resume_generate_api import async_tasks, task_lock

    client = app.test_client()
    tid = "upload_compat_it"
    data = {"id": tid, "query": "测试上传兼容"}
    files = {"cv": (io.BytesIO(b"r" * REAL_CV_BYTES), "cv.txt")}
    with patch("resume_generate_api._task_executor.submit", return_value=_FakeFuture()):
        resp = client.post(
            "/resume_optimize",
            data={**data, **files},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200, resp.get_data(as_text=True)[:300]
    payload = resp.get_json()
    assert payload.get("success") is True
    assert payload.get("task_id") == tid
    # cleanup in-memory task state
    with task_lock:
        async_tasks.pop(tid, None)
