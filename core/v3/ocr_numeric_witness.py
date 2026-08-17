"""Second-read numeric witness for raster OCR blocks (R26).

Measured truth: the same scanned resume OCRs differently across runs
(``2004`` vs ``204``, ``56`` vs ``5``), and a verbatim-preserved wrong digit
is a critical fabrication.  Text-side rules cannot detect same-shape digit
loss, so digit-bearing blocks are re-read by an independent recognition
stack (RapidOCR) and cross-checked.  Disagreement never picks a winner:
the block's confidence is dropped below the numeric-guard quarantine
threshold, the fact leaves the resume, and the reply surfaces it under
待确认数字.
"""
from __future__ import annotations

from collections import Counter
import io
import logging
import os
import re
from typing import Any

import numpy as np

logger = logging.getLogger("v3.ocr_numeric_witness")

_DIGIT_RUN = re.compile(r"\d+(?:[.,，]\d+)*")
WITNESS_CONFIDENCE = 0.5  # below numeric_guard.LOW_CONFIDENCE_THRESHOLD (0.8)


def _numeric_multiset(text: str) -> Counter:
    return Counter(
        token.replace(",", "").replace("，", "")
        for token in _DIGIT_RUN.findall(str(text or ""))
    )


def _witness_enabled() -> bool:
    return os.getenv("V3_OCR_NUMERIC_WITNESS", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _ocr_read(image_bytes: bytes) -> str:
    """Run the shared RapidOCR stack on an image; '' on any failure."""

    try:
        from resume_io import _init_rapid_ocr, _run_rapid_ocr

        _init_rapid_ocr()
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = _run_rapid_ocr(np.asarray(img))
        txts = getattr(result, "txts", None)
        if txts is None and isinstance(result, (tuple, list)) and len(result) > 1:
            txts = result[1]
        if txts is None:
            return ""
        return "\n".join(str(item) for item in txts)
    except Exception as exc:
        logger.warning("numeric witness read failed: %s", exc)
        return ""


def witness_ppstructure_blocks(
    blocks: list[dict[str, Any]],
    image_bytes: bytes,
) -> list[dict[str, Any]]:
    """Cross-check digit-bearing blocks with an independent OCR read.

    Blocks without digits pass through untouched.  A disagreement marks the
    block (confidence sentinel + metadata), never edits its text — the
    numeric guard quarantines the affected facts downstream.
    """

    if not _witness_enabled() or not blocks or not image_bytes:
        return blocks
    try:
        from PIL import Image
    except ImportError:
        return blocks
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return blocks
    width, height = image.size
    output: list[dict[str, Any]] = []
    disagreements = 0
    for block in blocks:
        text = str(block.get("block_content") or block.get("text") or "")
        raw_bbox = block.get("block_bbox", block.get("bbox"))
        if not _DIGIT_RUN.search(text) or not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            output.append(block)
            continue
        x1, y1, x2, y2 = (float(v) for v in raw_bbox)
        pad = 4.0
        crop_box = (
            max(0, int(x1 - pad)),
            max(0, int(y1 - pad)),
            min(width, int(x2 + pad) + 1),
            min(height, int(y2 + pad) + 1),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            output.append(block)
            continue
        buffer = io.BytesIO()
        image.crop(crop_box).save(buffer, format="PNG")
        witness_text = _ocr_read(buffer.getvalue())
        if not witness_text:
            output.append(block)
            continue
        if _numeric_multiset(text) == _numeric_multiset(witness_text):
            output.append(block)
            continue
        disagreements += 1
        marked = dict(block)
        marked["confidence"] = WITNESS_CONFIDENCE
        marked["numeric_witness"] = {
            "disagreement": True,
            "block_numbers": sorted(_numeric_multiset(text)),
            "witness_numbers": sorted(_numeric_multiset(witness_text)),
        }
        output.append(marked)
    if disagreements:
        logger.info("numeric witness disagreements | blocks=%d", disagreements)
    return output


__all__ = ["WITNESS_CONFIDENCE", "witness_ppstructure_blocks"]
