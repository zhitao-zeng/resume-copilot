#!/usr/bin/env python3
"""R26 OCR numeric witness contract tests."""

from __future__ import annotations

import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "core")):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.v3 import ocr_numeric_witness as witness  # noqa: E402


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(buf, format="PNG")
    return buf.getvalue()


def _blocks():
    return [
        {"block_content": "办公室协调员，[公司]2004-2011\n处理超过50个预约。", "block_bbox": [10, 10, 500, 80]},
        {"block_content": "负责团队管理", "block_bbox": [10, 100, 500, 140]},
    ]


def test_disagreement_marks_confidence_without_editing_text(monkeypatch):
    monkeypatch.setattr(witness, "_ocr_read", lambda _bytes: "办公室协调员，[公司]204-2011\n处理超过5个预约。")
    out = witness.witness_ppstructure_blocks(_blocks(), _png())
    first = out[0]
    assert first["confidence"] == witness.WITNESS_CONFIDENCE
    assert first["numeric_witness"]["disagreement"] is True
    # 文本不被改动
    assert first["block_content"] == _blocks()[0]["block_content"]
    # 无数字块原样通过
    assert "confidence" not in out[1]


def test_agreement_passes_through(monkeypatch):
    same = "办公室协调员，[公司]2004-2011\n处理超过50个预约。"
    monkeypatch.setattr(witness, "_ocr_read", lambda _b: same)
    out = witness.witness_ppstructure_blocks(_blocks(), _png())
    assert out[0].get("numeric_witness") is None
    assert out[0].get("confidence") is None


def test_witness_read_failure_is_fail_open(monkeypatch):
    monkeypatch.setattr(witness, "_ocr_read", lambda _b: "")
    out = witness.witness_ppstructure_blocks(_blocks(), _png())
    assert out[0].get("numeric_witness") is None


def test_witness_disabled_by_env(monkeypatch):
    monkeypatch.setenv("V3_OCR_NUMERIC_WITNESS", "0")
    monkeypatch.setattr(witness, "_ocr_read", lambda _b: (_ for _ in ()).throw(AssertionError("must not run")))
    out = witness.witness_ppstructure_blocks(_blocks(), _png())
    assert out[0].get("numeric_witness") is None
