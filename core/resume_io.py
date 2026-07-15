import io
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

from docx import Document as DocxDocument
from docx.oxml.ns import qn
from http_compat import HTTPException

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

# RapidOCR (GPU-accelerated, multi-language fallback)
_RAPID_OCR = None
_RAPID_OCR_PATH = None
_RAPID_OCR_INITED = False
_RAPID_OCR_VERSION = "v4"

def _init_rapid_ocr():
    global _RAPID_OCR, _RAPID_OCR_PATH, _RAPID_OCR_INITED, _RAPID_OCR_VERSION
    if _RAPID_OCR_INITED:
        return
    try:
        import numpy as np  # noqa: F811
        candidates = [
            Path(__file__).parent / "models" / "rapidocr_multilang",
            Path("/mounted_model/rapidocr_multilang"),
            Path("/mnt/disk1/zengzhitao/menu-translate/models/rapidocr_multilang"),
            Path("/root/app/models/rapidocr_multilang"),  # Docker container path
        ]
        candidates = [c for c in candidates if c is not None]
        model_dir = None
        for c in candidates:
            # Prefer PP-OCRv5 Chinese server models
            det = c / "ch_PP-OCRv5_det_server.onnx"
            rec = c / "ch_PP-OCRv5_rec_server.onnx"
            cls = c / "ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx"
            if det.exists() and rec.exists() and cls.exists():
                model_dir = str(c)
                version = "v5"
                break

        if model_dir is None:
            for c in candidates:
                # Fallback to PP-OCRv4 Chinese server models
                det = c / "ch_PP-OCRv4_det_server.onnx"
                rec = c / "ch_PP-OCRv4_rec_server.onnx"
                cls = c / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
                if det.exists() and rec.exists() and cls.exists():
                    model_dir = str(c)
                    version = "v4"
                    break

        if model_dir is None:
            for c in candidates:
                # Fallback to PP-OCRv5 Chinese mobile models
                det = c / "ch_PP-OCRv5_det_mobile.onnx"
                rec = c / "ch_PP-OCRv5_rec_mobile.onnx"
                cls = c / "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx"
                if det.exists() and rec.exists() and cls.exists():
                    model_dir = str(c)
                    version = "v5"
                    break

        if model_dir is None:
            for c in candidates:
                # Final fallback: PP-OCRv4 Chinese mobile models
                det = c / "ch_PP-OCRv4_det_mobile.onnx"
                rec = c / "ch_PP-OCRv4_rec_mobile.onnx"
                cls = c / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
                if det.exists() and rec.exists() and cls.exists():
                    model_dir = str(c)
                    version = "v4"
                    break

        if model_dir is None:
            logger.warning("RapidOCR model files not found, falling back to pytesseract")
            _RAPID_OCR_INITED = True
            return

        det_path = os.path.join(model_dir, f"ch_PP-OCR{version}_det_server.onnx") if version == "v5" else os.path.join(model_dir, f"ch_PP-OCR{version}_det_server.onnx")
        rec_path = os.path.join(model_dir, f"ch_PP-OCR{version}_rec_server.onnx") if version == "v5" else os.path.join(model_dir, f"ch_PP-OCR{version}_rec_server.onnx")
        # RapidOCR 默认 Cls.ocr_version=PP-OCRv4 => cls_image_shape=[3,48,192]，
        # 但 v5 CLS server 模型实际需要 [3,80,160]。通过显式传入 Cls.ocr_version
        # 让 TextClassifier 使用正确的输入形状，避免 dimension mismatch。
        if version == "v5":
            cls_path = os.path.join(model_dir, "ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx")
        else:
            cls_path = os.path.join(model_dir, "ch_ppocr_mobile_v2.0_cls_mobile.onnx")

        from rapidocr import RapidOCR
        from rapidocr.utils.typings import OCRVersion as _OCRVersion

        cls_params = {"Cls.ocr_version": _OCRVersion.PPOCRV5} if version == "v5" else {}
        ocr_instance = RapidOCR(params={
            "Det.model_path": det_path,
            "Cls.model_path": cls_path,
            "Rec.model_path": rec_path,
            **cls_params,
        })
        _RAPID_OCR = ocr_instance
        _RAPID_OCR_PATH = model_dir
        _RAPID_OCR_INITED = True
        _RAPID_OCR_VERSION = version
        logger.info(f"RapidOCR {version} initialized from {model_dir}")
    except Exception as exc:
        logger.warning(f"RapidOCR init failed: {exc}, falling back to pytesseract")
        _RAPID_OCR_INITED = True  # prevent repeated attempts


def _ocr_image_with_rapid(content: bytes) -> str:
    """Extract text from image using global RapidOCR (single engine call).

    For two-column resume templates with a colored left banner (red/blue)
    and white right body, extracts the green channel which maximizes
    contrast for white-on-red text while preserving dark-on-white body text.
    Calls the global engine ONCE.
    """
    import numpy as np  # noqa: F811
    from PIL import ImageOps

    pil_img = Image.open(io.BytesIO(content))
    if pil_img.mode not in {"RGB", "L"}:
        pil_img = pil_img.convert("RGB")

    arr = np.array(pil_img)
    # Extract green channel — ideal for white-on-red (red bg is dark in G,
    # white text is bright) and adequate for dark-on-white body text.
    green = arr[:, :, 1]  # shape (H, W), values 0-255
    green_img = Image.fromarray(green).convert("L")
    # Stretch contrast to full range
    enhanced = ImageOps.autocontrast(green_img, cutoff=3)
    np_img = np.array(enhanced.convert("RGB"))

    try:
        result = _RAPID_OCR(np_img)
    except Exception as exc:
        logger.warning("RapidOCR inference failed, falling back: %s", exc)
        return ""
    boxes = result.boxes if hasattr(result, "boxes") else (result[0] if isinstance(result, (tuple, list)) else None)
    txts = result.txts if hasattr(result, "txts") else (result[1] if isinstance(result, (tuple, list)) else None)
    if boxes is None or len(boxes) == 0:
        return ""

    texts = []
    # Reconstruct reading order using layout information
    ordered_texts = _reconstruct_ocr_reading_order(
        boxes, txts, img_width=np_img.shape[1], img_height=np_img.shape[0],
    )
    return "\n".join(ordered_texts) if ordered_texts else ""



# ── OCR Layout Reconstruction ─────────────────────────────────────────────────


def _reconstruct_ocr_reading_order(
    boxes,
    texts,
    *,
    img_width: int,
    img_height: int,
) -> list[str]:
    """Reconstruct reading order from OCR blocks using layout analysis.

    The raw OCR output orders blocks by detector confidence, not by reading
    order.  This function:
    1.  Extracts each block's bbox into structured fields
    2.  Detects narrow side-column blocks (e.g. red banner with contact info)
    3.  Identifies full-width blocks (e.g. paragraph text spanning the page)
    4.  Clusters blocks into visual rows by y-overlap
    5.  Orders: side column → full-width → main column, each sorted by y
    """
    import numpy as np  # noqa: F811

    if boxes is None or len(boxes) == 0:
        return []

    # ── 1. Build block list with structured bbox fields ──
    raw_blocks: list[dict] = []
    for i in range(len(boxes)):
        box = boxes[i]
        if texts and i < len(texts) and texts[i]:
            raw_val = str(texts[i][0]) if isinstance(texts[i], (tuple, list)) else str(texts[i])
        else:
            raw_val = str(texts[i]) if texts and i < len(texts) and texts[i] is not None else ""
        text = raw_val.strip()
        if not text:
            continue

        pts = np.array(box)
        x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
        y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())

        raw_blocks.append({
            "text": text,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "x_center": (x_min + x_max) / 2.0,
            "y_center": (y_min + y_max) / 2.0,
            "width": x_max - x_min,
            "height": y_max - y_min,
        })

    if not raw_blocks:
        return []

    # ── 2. Detect side column (banner) blocks ──
    # Chinese resume templates often have a narrow colored banner on the left
    # with contact info (name, phone, email, job target).  These blocks have
    # small x_max values compared to the main content blocks.
    #
    # Approach: cluster all blocks by x_max.  If there's a distinct cluster
    # with small x_max and the gap to the next cluster is large enough,
    # treat that cluster as a side column.
    sorted_by_xmax = sorted(raw_blocks, key=lambda b: b["x_max"])
    n = len(sorted_by_xmax)

    # Find the first large gap in x_max values that splits blocks into
    # two meaningful groups (both sides must have >= 2 blocks).
    gap_idx = -1
    gap_threshold = img_width * 0.08  # 8% page width
    for i in range(n - 1):
        gap = sorted_by_xmax[i + 1]["x_max"] - sorted_by_xmax[i]["x_max"]
        if gap >= gap_threshold:
            left_count = i + 1
            right_count = n - left_count
            if left_count >= 2 and right_count >= 2:
                gap_idx = i
                break

    side_col_x_max = 0
    if gap_idx >= 0:
        side_col_x_max = sorted_by_xmax[gap_idx]["x_max"]

    # ── 3. Classify blocks into regions ──
    SIDE = "side"
    FULL = "full"
    MAIN = "main"

    def _classify(b: dict) -> str:
        if side_col_x_max > 0 and b["x_max"] <= side_col_x_max and b["width"] < img_width * 0.4:
            return SIDE
        if b["width"] >= img_width * 0.6:
            return FULL
        return MAIN

    # ── 4. Row clustering within each region ──
    def _cluster_rows(region_blocks: list[dict]) -> list[list[dict]]:
        if not region_blocks:
            return []
        region_blocks.sort(key=lambda b: b["y_center"])
        rows: list[list[dict]] = []
        current = [region_blocks[0]]
        cur_y_min = region_blocks[0]["y_min"]
        cur_y_max = region_blocks[0]["y_max"]

        for b in region_blocks[1:]:
            overlap = min(b["y_max"], cur_y_max) - max(b["y_min"], cur_y_min)
            if overlap >= 0 or (b["y_min"] - cur_y_max) < 3:
                current.append(b)
                cur_y_min = min(cur_y_min, b["y_min"])
                cur_y_max = max(cur_y_max, b["y_max"])
            else:
                rows.append(current)
                current = [b]
                cur_y_min = b["y_min"]
                cur_y_max = b["y_max"]
        if current:
            rows.append(current)
        return rows

    # ── 5. Sort and flatten ──
    side_blocks = [b for b in raw_blocks if _classify(b) == SIDE]
    full_blocks = [b for b in raw_blocks if _classify(b) == FULL]
    main_blocks = [b for b in raw_blocks if _classify(b) == MAIN]

    # Within each row, sort blocks left-to-right
    def _flatten_rows(rows: list[list[dict]]) -> list[str]:
        result: list[str] = []
        for row in rows:
            row.sort(key=lambda b: b["x_center"])
            for b in row:
                result.append(b["text"])
        return result

    side_lines = _flatten_rows(_cluster_rows(side_blocks)) if side_blocks else []
    main_lines = _flatten_rows(_cluster_rows(main_blocks)) if main_blocks else []
    full_lines = _flatten_rows(_cluster_rows(full_blocks)) if full_blocks else []

    # Order: region groups, sorted internally by y.
    #
    # Full-width blocks (summary text) emit first — they are at the top
    # of the page and span both columns.
    #
    # Side-column blocks (name/contact/target in the left colored banner)
    # emit second — they form a visually distinct region that should be
    # read before entering the main content area.
    #
    # Main-content blocks (work/skills/projects) emit last.
    #
    # Within each region, blocks are clustered into rows and sorted by y.
    # Order: assign each block a sort key based on its region and position.
    # Only rank blocks within the same region by y to avoid merging
    # unrelated blocks from side column and main content.
    SIDE_PRIORITY = 0
    FULL_PRIORITY = 1
    MAIN_PRIORITY = 2

    # Sort each region's blocks by y, then interleave by vertical position.
    #
    # For side-column vs main content: side column blocks are emitted first
    # at each y-band, since the banner is logically read before the main
    # content area at the same vertical level.
    #
    # For full-width blocks: they are placed before side column content if
    # they sit above the side column (page header), or after main content
    # if they sit below it (page footer).
    side_ordered = sorted([b for b in raw_blocks if _classify(b) == SIDE], key=lambda b: b["y_center"])
    full_ordered = sorted([b for b in raw_blocks if _classify(b) == FULL], key=lambda b: b["y_center"])
    main_ordered = sorted([b for b in raw_blocks if _classify(b) == MAIN], key=lambda b: b["y_center"])

    # Find the y range of side column content to position full-width blocks
    if side_ordered:
        side_min_y = min(b["y_min"] for b in side_ordered)
        side_max_y = max(b["y_max"] for b in side_ordered)
    else:
        side_min_y, side_max_y = 0, 0

    full_headers = [b for b in full_ordered if b["y_max"] < side_min_y or not side_ordered]
    full_footers = [b for b in full_ordered if b["y_min"] >= side_max_y]
    full_mid = [b for b in full_ordered if b not in full_headers and b not in full_footers]

    result = [b["text"] for b in full_headers] + [b["text"] for b in side_ordered] + [b["text"] for b in full_mid] + [b["text"] for b in main_ordered] + [b["text"] for b in full_footers]
    return result


def _ocr_image_multicandidate(content: bytes) -> str:
    """Try multiple preprocessing strategies with fresh OCR engines.

    Each call creates a new engine, so this is safe even when the
    global engine state is corrupted.
    """
    import numpy as np  # noqa: F811
    from PIL import ImageOps, ImageEnhance

    pil_img = Image.open(io.BytesIO(content))
    if pil_img.mode not in {"RGB", "L"}:
        pil_img = pil_img.convert("RGB")

    if not _RAPID_OCR_PATH:
        logger.warning("_RAPID_OCR_PATH is None, skipping multicandidate OCR fallback")
        return ""

    def _eng():
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import OCRVersion as _OCRVersion
        det_model = f"ch_PP-OCR{_RAPID_OCR_VERSION}_det_server.onnx"
        rec_model = f"ch_PP-OCR{_RAPID_OCR_VERSION}_rec_server.onnx"
        if _RAPID_OCR_VERSION == "v5":
            cls_model = "ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx"
            cp = {"Cls.ocr_version": _OCRVersion.PPOCRV5}
        else:
            cls_model = "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
            cp = {}
        return RapidOCR(params={
            "Det.model_path": _RAPID_OCR_PATH + "/" + det_model,
            "Cls.model_path": _RAPID_OCR_PATH + "/" + cls_model,
            "Rec.model_path": _RAPID_OCR_PATH + "/" + rec_model,
            **cp,
        })

    def _run(pil: Image.Image) -> str:
        try:
            engine = _eng()
            result = engine(np.array(pil))
            boxes = result.boxes if hasattr(result, "boxes") else (result[0] if isinstance(result, (tuple, list)) else None)
            txts = result.txts if hasattr(result, "txts") else (result[1] if isinstance(result, (tuple, list)) else None)
            if boxes is None or len(boxes) == 0:
                return ""
            texts = []
            for i in range(len(boxes)):
                if texts and i < len(texts) and texts[i]:
                    txt = str(txts[i][0]) if isinstance(txts[i], (tuple, list)) else str(txts[i])
                else:
                    txt = ""
                if txt.strip():
                    texts.append(txt.strip())
            return "\n".join(texts)
        except Exception:
            return ""

    best_text = ""
    best_len = 0

    # Strategy 1: grayscale 2x scale (original multicandidate)
    gray_2x = pil_img.convert("L").resize((pil_img.width * 2, pil_img.height * 2), Image.BICUBIC)
    t = _run(gray_2x.convert("RGB"))
    if len(t.splitlines()) > best_len:
        best_text, best_len = t, len(t.splitlines())

    # Strategy 2: autocontrast (handles both light and dark areas)
    auto = ImageOps.autocontrast(pil_img.convert("L"), cutoff=3)
    t = _run(auto.convert("RGB"))
    if len(t.splitlines()) > best_len:
        best_text, best_len = t, len(t.splitlines())

    # Strategy 3: invert left column + keep right original (for two-column templates)
    w, h = pil_img.size
    if w > 200:
        arr = np.array(pil_img)
        left_inv = 255 - arr[:, :int(w * 0.35), :]
        right_orig = arr[:, int(w * 0.35):, :]
        merged = np.concatenate([left_inv, right_orig], axis=1)
        t = _run(Image.fromarray(merged).convert("RGB"))
        if len(t.splitlines()) > best_len:
            best_text, best_len = t, len(t.splitlines())

    return best_text

from server_runtime import (
    ALLOWED_UPLOAD_EXTENSIONS,
    AVATAR_DIR,
    ENABLE_AVATAR_EXTRACTION,
    ENABLE_AVATAR_FACE_SCORE,
    ENABLE_TEXT_LAYOUT_NORMALIZATION,
    PROJECT_SECTION_HEADERS,
    SECTION_HEADING_KEYWORDS,
    SUPPORTED_FILE_PATH_EXTENSIONS,
    _FACE_CASCADE,
    cv2,
    fitz,
    logger,
    np,
    pikepdf,
)
def _normalize_heading_line(line: str) -> str:
    text = str(line or "").strip()
    text = re.sub(r"\s+", "", text)
    return text.strip("：:;；|·•-—–")


def _looks_like_section_header(line: str, keywords: tuple[str, ...], max_len: int = 32) -> bool:
    normalized = _normalize_heading_line(line)
    if not normalized or len(normalized) > max_len:
        return False
    lowered = normalized.lower()
    for keyword in keywords:
        key = _normalize_heading_line(keyword)
        if key and (key in normalized or key.lower() in lowered):
            return True
    return False


def _normalize_extracted_resume_text(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    if not ENABLE_TEXT_LAYOUT_NORMALIZATION:
        return raw.strip()

    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u3000", " ").replace("\xa0", " ")
    normalized = normalized.replace("\ufeff", "").replace("\u200b", "")
    # Remove spaces/tabs between Chinese chars (NOT newlines \u2014 those separate
    # OCR blocks and must be preserved for correct reading order).
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])", "", normalized)

    heading_alt = "|".join(re.escape(item) for item in sorted(SECTION_HEADING_KEYWORDS, key=len, reverse=True))
    if heading_alt:
        normalized = re.sub(rf"\s*({heading_alt})\s*", r"\n\1\n", normalized, flags=re.IGNORECASE)

    normalized = re.sub(r"[•▪◦●○■□▶►▸▹◆◇\uF000-\uF8FF]+", "\n- ", normalized)
    normalized = re.sub(
        r"(?<!\n)\s*((?:19|20)\d{2}[./年]\d{1,2}(?:月)?\s*(?:[-–—~至到]+\s*(?:(?:19|20)\d{2}[./年]\d{1,2}(?:月)?|至今|Present|present)))",
        r"\n\1",
        normalized,
        flags=re.IGNORECASE,
    )

    lines: list[str] = []
    for line in normalized.splitlines():
        cleaned = re.sub(r"\s{2,}", " ", line).strip()
        if not cleaned:
            continue
        lines.append(cleaned)

    return "\n".join(lines).strip()


def _detect_extension(file_name: str) -> str:
    return Path(file_name or "").suffix.lower()


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _collect_docx_table_text(table: Any) -> list[str]:
    lines: list[str] = []
    rows = getattr(table, "rows", None)
    if not rows:
        return lines

    for row in rows:
        for cell in getattr(row, "cells", []) or []:
            for paragraph in getattr(cell, "paragraphs", []) or []:
                text = str(getattr(paragraph, "text", "") or "").strip()
                if text:
                    lines.append(text)
            for nested in getattr(cell, "tables", []) or []:
                lines.extend(_collect_docx_table_text(nested))
    return lines


def _extract_text_from_docx_document(doc: Any) -> str:
    """Extract text from DOCX preserving paragraph/table XML body order.

    Traverses ``<w:body>`` child elements (``<w:p>`` and ``<w:tbl>``) in their
    original document order, rather than collecting all paragraphs then all
    tables separately.  This preserves the structure that python-docx's
    ``.paragraphs`` / ``.tables`` iterators discard.
    """
    ns_w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    lines: list[str] = []
    seen: set[str] = set()

    body = doc.element.body
    for child in body:
        tag = child.tag
        if tag == qn("w:p"):
            parts: list[str] = []
            for node in child.iter(f"{{{ns_w}}}t"):
                if node.text:
                    parts.append(node.text)
            text = re.sub(r"\s+", " ", "".join(parts)).strip()
            if text and text not in seen:
                seen.add(text)
                lines.append(text)
        elif tag == qn("w:tbl"):
            _collect_docx_table_text_inline(child, ns_w, lines, seen)

    return "\n".join(lines).strip()


def _collect_docx_table_text_inline(
    tbl_elem: Any, ns_w: str, lines: list[str], seen: set[str]
) -> None:
    """Extract cell text from a single ``<w:tbl>`` element, in row order."""
    for row in tbl_elem.iter(f"{{{ns_w}}}tr"):
        for cell in row.iter(f"{{{ns_w}}}tc"):
            cell_texts: list[str] = []
            for p in cell.iter(f"{{{ns_w}}}p"):
                parts: list[str] = []
                for t_node in p.iter(f"{{{ns_w}}}t"):
                    if t_node.text:
                        parts.append(t_node.text)
                ct = re.sub(r"\s+", " ", "".join(parts)).strip()
                if ct:
                    cell_texts.append(ct)
            combined = " ".join(cell_texts)
            if combined and combined not in seen:
                seen.add(combined)
                lines.append(combined)


def _extract_text_from_docx_xml_bytes(content: bytes) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    ns_w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ns_a = "http://schemas.openxmlformats.org/drawingml/2006/main"
    candidate_pattern = re.compile(r"^word/(document|header\d+|footer\d+|footnotes|endnotes)\.xml$")

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = [name for name in zf.namelist() if candidate_pattern.match(name)]
            for name in sorted(names):
                try:
                    root = ET.fromstring(zf.read(name))
                except Exception:
                    continue
                for paragraph in root.iter(f"{{{ns_w}}}p"):
                    parts: list[str] = []
                    for node in paragraph.iter():
                        if node.tag in {f"{{{ns_w}}}t", f"{{{ns_a}}}t"}:
                            value = str(node.text or "")
                            if value:
                                parts.append(value)
                    text = re.sub(r"\s+", " ", "".join(parts)).strip()
                    if text and text not in seen:
                        seen.add(text)
                        lines.append(text)
    except Exception:
        return ""

    return "\n".join(lines).strip()


def _sanitize_pdf_bytes(content: bytes) -> bytes:
    """Normalize PDF bytes with pikepdf/qpdf when available."""
    if pikepdf is None:
        return content
    try:
        with pikepdf.open(io.BytesIO(content)) as pdf:
            output = io.BytesIO()
            pdf.save(output)
            sanitized = output.getvalue()
            if sanitized:
                return sanitized
    except Exception as exc:
        logger.warning("pikepdf sanitize failed; fallback to raw bytes | error=%s", exc)
    return content


def _normalize_image_ext(ext: str) -> str:
    value = str(ext or "").strip().lower().lstrip(".")
    if value == "jpeg":
        value = "jpg"
    if value not in {"jpg", "png", "webp", "bmp", "gif", "tif", "tiff"}:
        value = "png"
    return value


def _coerce_avatar_bytes_to_rgb_png(image_bytes: bytes, ext: str) -> tuple[bytes, str]:
    payload = bytes(image_bytes or b"")
    image_ext = _normalize_image_ext(ext)
    if not payload or fitz is None:
        return payload, image_ext

    try:
        image_doc = fitz.open(stream=payload, filetype=image_ext)
        if image_doc.page_count <= 0:
            image_doc.close()
            return payload, image_ext
        page = image_doc.load_page(0)
        pix = page.get_pixmap(alpha=False)
        png_bytes = pix.tobytes("png")
        image_doc.close()
        if png_bytes:
            return bytes(png_bytes), "png"
    except Exception:
        return payload, image_ext

    return payload, image_ext


def _persist_avatar_bytes(image_bytes: bytes, ext: str, source_name: str) -> Optional[str]:
    if not ENABLE_AVATAR_EXTRACTION:
        return None
    payload, image_ext = _coerce_avatar_bytes_to_rgb_png(image_bytes, ext)
    if len(payload) < 1024:
        return None
    stem = re.sub(r"[^\w\u4e00-\u9fff\-]+", "_", Path(source_name or "resume").stem).strip("_") or "resume"
    filename = f"{stem}_avatar_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.{image_ext}"
    output_path = AVATAR_DIR / filename
    output_path.write_bytes(payload)
    return str(output_path)


def _extract_pdf_image_png_bytes(doc: Any, xref: int, smask: int, source_name: str) -> bytes:
    pix = None
    try:
        base_pix = fitz.Pixmap(doc, xref)
        pix = base_pix
        if smask > 0:
            try:
                mask_pix = fitz.Pixmap(doc, smask)
                pix = fitz.Pixmap(base_pix, mask_pix)
            except Exception:
                pix = base_pix
        if pix.alpha:
            pix = fitz.Pixmap(pix, 0)
        if pix.colorspace is not None and pix.colorspace.n > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        png_bytes = pix.tobytes("png")
        if png_bytes:
            return bytes(png_bytes)
    except Exception as exc:
        logger.warning("Avatar pixmap normalize failed | source=%s xref=%s error=%s", source_name, xref, exc)

    try:
        base = doc.extract_image(xref) or {}
        return bytes(base.get("image") or b"")
    except Exception:
        return b""


def _avatar_face_stats(image_bytes: bytes) -> tuple[int, float]:
    if (
        not ENABLE_AVATAR_EXTRACTION
        or not ENABLE_AVATAR_FACE_SCORE
        or cv2 is None
        or np is None
        or _FACE_CASCADE is None
    ):
        return 0, 0.0

    try:
        arr = np.frombuffer(image_bytes or b"", dtype=np.uint8)
        if arr.size == 0:
            return 0, 0.0
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return 0, 0.0
        h, w = img.shape[:2]
        if w < 40 or h < 40:
            return 0, 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return 0, 0.0
        image_area = max(float(w * h), 1.0)
        max_ratio = 0.0
        for face in faces:
            try:
                fw = float(face[2] or 0.0)
                fh = float(face[3] or 0.0)
                max_ratio = max(max_ratio, (fw * fh) / image_area)
            except Exception:
                continue
        return int(len(faces)), float(max_ratio)
    except Exception:
        return 0, 0.0


def _apply_avatar_face_bonus(base_score: float, image_bytes: bytes) -> float:
    face_count, max_ratio = _avatar_face_stats(image_bytes)
    if face_count <= 0:
        return base_score

    # Use face signal as a bonus only; keep layout/size heuristics as primary score.
    multiplier = 1.0 + min(2.8, face_count * 1.05 + max_ratio * 4.0)
    additive = 25000.0 * float(min(face_count, 3))
    return base_score * multiplier + additive


def _extract_avatar_from_pdf_bytes(content: bytes, source_name: str) -> Optional[str]:
    if fitz is None or not ENABLE_AVATAR_EXTRACTION:
        return None

    sanitized = _sanitize_pdf_bytes(content)
    candidates: list[bytes] = [sanitized] if sanitized != content else []
    candidates.append(content)

    for pdf_bytes in candidates:
        doc = None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if doc.page_count <= 0:
                continue
            page = doc.load_page(0)
            rect = page.rect
            page_w = float(rect.width or 0.0)
            page_h = float(rect.height or 0.0)
            page_area = max(1.0, page_w * page_h)

            scored: list[tuple[float, int, int, bytes]] = []
            for img in page.get_images(full=True) or []:
                if not isinstance(img, (list, tuple)) or not img:
                    continue
                xref = int(img[0] or 0)
                if xref <= 0:
                    continue
                smask = int(img[1] or 0) if len(img) > 1 else 0
                rects: list[Any] = []
                try:
                    rects = page.get_image_rects(xref) or []
                except Exception:
                    rects = []

                best_rect = None
                best_area = 0.0
                for r in rects:
                    try:
                        rw = float(r.width or 0.0)
                        rh = float(r.height or 0.0)
                        area = rw * rh
                    except Exception:
                        continue
                    if area > best_area:
                        best_area = area
                        best_rect = r

                if best_rect is None:
                    # No placement rect (rare) - still allow as low-confidence fallback.
                    png_bytes = _extract_pdf_image_png_bytes(doc, xref, smask, source_name)
                    if not png_bytes:
                        continue
                    score = _apply_avatar_face_bonus(1.0, png_bytes)
                    scored.append((score, xref, smask, png_bytes))
                    continue

                width = max(0.0, float(best_rect.width or 0.0))
                height = max(0.0, float(best_rect.height or 0.0))
                area = width * height
                if width < 36 or height < 36:
                    continue
                ratio = area / page_area
                if ratio < 0.0015 or ratio > 0.28:
                    continue

                x0 = float(best_rect.x0 or 0.0)
                y0 = float(best_rect.y0 or 0.0)
                aspect = width / max(height, 1.0)
                score = float(area)
                if 0.55 <= aspect <= 1.45:
                    score *= 1.2
                if y0 <= page_h * 0.6:
                    score *= 1.2
                if x0 >= page_w * 0.38:
                    score *= 1.1
                png_bytes = _extract_pdf_image_png_bytes(doc, xref, smask, source_name)
                if not png_bytes:
                    continue
                score = _apply_avatar_face_bonus(score, png_bytes)
                scored.append((score, xref, smask, png_bytes))

            if not scored:
                continue

            scored.sort(key=lambda item: item[0], reverse=True)
            best_score, best_xref, best_smask, best_png_bytes = scored[0]
            if best_score <= 0:
                continue

            saved = _persist_avatar_bytes(best_png_bytes, "png", source_name=source_name)
            if saved:
                logger.info(
                    "Extracted avatar from PDF | source=%s path=%s xref=%s smask=%s",
                    source_name,
                    saved,
                    best_xref,
                    best_smask,
                )
                return saved
        except Exception as exc:
            logger.warning("Avatar extraction from PDF failed | source=%s error=%s", source_name, exc)
        finally:
            if doc is not None:
                doc.close()

    return None


def _extract_avatar_from_docx_bytes(content: bytes, source_name: str) -> Optional[str]:
    if not ENABLE_AVATAR_EXTRACTION:
        return None

    # Prefer inline-shape order as it better reflects the original document flow.
    try:
        doc = DocxDocument(io.BytesIO(content))
        scored: list[tuple[float, bytes, str]] = []
        for idx, shape in enumerate(getattr(doc, "inline_shapes", [])):
            try:
                inline = shape._inline  # type: ignore[attr-defined]
                rid = inline.graphic.graphicData.pic.blipFill.blip.embed
                image_part = doc.part.related_parts.get(rid)
                if image_part is None:
                    continue
                payload = bytes(image_part.blob or b"")
                if len(payload) < 1024:
                    continue
                ext = _normalize_image_ext(Path(getattr(image_part, "filename", "avatar.png")).suffix)
                width = float(getattr(shape, "width", 0) or 0)
                height = float(getattr(shape, "height", 0) or 0)
                area = max(1.0, width * height)
                score = area * (1.25 if idx == 0 else 1.0 / (idx + 1))
                score = _apply_avatar_face_bonus(score, payload)
                scored.append((score, payload, ext))
            except Exception:
                continue
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            saved = _persist_avatar_bytes(scored[0][1], scored[0][2], source_name=source_name)
            if saved:
                logger.info("Extracted avatar from DOCX inline shape | source=%s path=%s", source_name, saved)
                return saved
    except Exception as exc:
        logger.warning("DOCX inline-shape avatar extraction failed | source=%s error=%s", source_name, exc)

    # Fallback: scan embedded media files.
    try:
        scored_zip: list[tuple[float, bytes, str]] = []
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for info in zf.infolist():
                name = info.filename.lower()
                if not name.startswith("word/media/"):
                    continue
                ext = _normalize_image_ext(Path(name).suffix)
                payload = zf.read(info.filename)
                if len(payload) < 1024:
                    continue
                # Prefer earlier media entries and larger files.
                score = float(info.file_size) + max(0, 200000 - info.header_offset) * 0.1
                score = _apply_avatar_face_bonus(score, payload)
                scored_zip.append((score, payload, ext))
        if scored_zip:
            scored_zip.sort(key=lambda item: item[0], reverse=True)
            saved = _persist_avatar_bytes(scored_zip[0][1], scored_zip[0][2], source_name=source_name)
            if saved:
                logger.info("Extracted avatar from DOCX media | source=%s path=%s", source_name, saved)
                return saved
    except Exception as exc:
        logger.warning("DOCX media avatar extraction failed | source=%s error=%s", source_name, exc)

    return None


def _extract_avatar_from_upload_bytes(content: bytes, filename: str) -> Optional[str]:
    if not ENABLE_AVATAR_EXTRACTION:
        return None
    ext = _detect_extension(filename)
    if ext == ".pdf":
        return _extract_avatar_from_pdf_bytes(content, source_name=filename or "resume.pdf")
    if ext == ".docx":
        return _extract_avatar_from_docx_bytes(content, source_name=filename or "resume.docx")
    return None


def _extract_avatar_from_file_path(file_path: str) -> Optional[str]:
    if not ENABLE_AVATAR_EXTRACTION or not file_path:
        return None
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = path.read_bytes()
    except Exception:
        return None
    return _extract_avatar_from_upload_bytes(payload, path.name)


def _extract_text_from_pdf_bytes(content: bytes) -> str:
    if fitz is None:
        raise HTTPException(status_code=500, detail="pymupdf is required for PDF parsing")

    sanitized = _sanitize_pdf_bytes(content)
    candidates: list[tuple[str, bytes]] = []
    if sanitized != content:
        candidates.append(("sanitized", sanitized))
    candidates.append(("raw", content))

    last_exc: Optional[Exception] = None
    for source, pdf_bytes in candidates:
        doc = None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            return "\n".join(page.get_text() for page in doc).strip()
        except Exception as exc:
            last_exc = exc
            logger.warning("PDF parse failed via %s bytes | error=%s", source, exc)
        finally:
            if doc is not None:
                doc.close()

    raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {last_exc}") from last_exc


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff"}


def extract_text_from_image_bytes(content: bytes, filename: str) -> str:
    # Prefer RapidOCR (GPU, accurate), fall back to pytesseract (CPU)
    if not _RAPID_OCR_INITED:
        _init_rapid_ocr()

    text = ""
    if _RAPID_OCR is not None:
        text = _ocr_image_with_rapid(content)

    if not text.strip():
        # Try grayscale/scaled preprocessing as fallback
        try:
            text = _ocr_image_multicandidate(content)
        except Exception as exc:
            logger.warning("RapidOCR preprocessing fallback failed: %s", exc)

    if not text.strip() and pytesseract is not None:
        # Last resort: pytesseract (CPU, lower quality)
        try:
            image = Image.open(io.BytesIO(content))
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            text = pytesseract.image_to_string(image, lang="chi_sim+eng")
        except Exception as exc:
            logger.warning("pytesseract OCR also failed: %s", exc)

    if not text.strip():
        logger.warning("All OCR engines failed for %s", filename)
        return ""

    normalized = _normalize_extracted_resume_text(text)
    return normalized if normalized else ""

def extract_text_from_bytes(content: bytes, filename: str) -> str:
    ext = _detect_extension(filename)
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")

    if ext in IMAGE_EXTENSIONS:
        return extract_text_from_image_bytes(content, filename)

    if ext in {".txt", ".md"}:
        text = content.decode("utf-8", errors="replace").strip()
    elif ext == ".pdf":
        text = _extract_text_from_pdf_bytes(content)
    else:
        try:
            doc = DocxDocument(io.BytesIO(content))
            text = _extract_text_from_docx_document(doc)
            if len(text) < 20:
                xml_text = _extract_text_from_docx_xml_bytes(content)
                if len(xml_text) > len(text):
                    logger.info(
                        "DOCX XML fallback improved text extraction | filename=%s before_chars=%s after_chars=%s",
                        filename,
                        len(text),
                        len(xml_text),
                    )
                    text = xml_text
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse DOCX: {exc}") from exc

    text = _normalize_extracted_resume_text(text)
    if not text:
        raise HTTPException(status_code=400, detail="No valid text extracted from file")
    return text


def extract_text_from_path(file_path: str) -> str:
    if not file_path:
        raise HTTPException(status_code=400, detail="file_path is required")

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=400, detail=f"file_path not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_FILE_PATH_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {ext}")

    if ext in {".txt", ".md"}:
        text = _read_text_file(path)
    elif ext == ".pdf":
        try:
            text = _extract_text_from_pdf_bytes(path.read_bytes())
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc
    else:
        try:
            doc = DocxDocument(str(path))
            text = _extract_text_from_docx_document(doc)
            if len(text) < 20:
                xml_text = _extract_text_from_docx_xml_bytes(path.read_bytes())
                if len(xml_text) > len(text):
                    logger.info(
                        "DOCX XML fallback improved text extraction | path=%s before_chars=%s after_chars=%s",
                        file_path,
                        len(text),
                        len(xml_text),
                    )
                    text = xml_text
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse DOCX: {exc}") from exc

    text = _normalize_extracted_resume_text(text)
    if not text:
        raise HTTPException(status_code=400, detail="No valid text extracted from file")
    return text


def resolve_resume_text(resume_content: Optional[str], file_path: Optional[str]) -> str:
    if resume_content and str(resume_content).strip():
        return _normalize_extracted_resume_text(str(resume_content))
    if file_path and str(file_path).strip():
        return extract_text_from_path(str(file_path))
    raise HTTPException(status_code=400, detail="Either resume_content or file_path is required")
