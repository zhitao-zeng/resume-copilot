#!/usr/bin/env python3
"""Compare the production PP-OCR path and an optional VL endpoint.

The two backends receive byte-identical pages.  Synthetic pages have exact
line-level ground truth; acceptance pages use the matching PDF text layer as a
grouped regression reference.  Metrics intentionally penalize both omissions
and unsupported characters because a resume transcription must not invent.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import statistics
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))

import resume_io  # noqa: E402


ARTIFACT_ROOT = ROOT / ".codex" / "research-loop" / "artifacts" / "multicolumn-ocr-vl-ab"
DEFAULT_FONT = Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")


def _normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    return "".join(char for char in text if not char.isspace())


def _lcs_length(left: str, right: str) -> int:
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _metrics(expected: str, actual: str) -> dict[str, Any]:
    expected_norm = _normalized(expected)
    actual_norm = _normalized(actual)
    lcs = _lcs_length(expected_norm, actual_norm)
    expected_counts = Counter(expected_norm)
    actual_counts = Counter(actual_norm)
    unsupported = sum((actual_counts - expected_counts).values())
    missing = sum((expected_counts - actual_counts).values())
    return {
        "expected_chars": len(expected_norm),
        "actual_chars": len(actual_norm),
        "exact": expected_norm == actual_norm,
        "ordered_char_recall": round(lcs / max(1, len(expected_norm)), 4),
        "ordered_char_precision": round(lcs / max(1, len(actual_norm)), 4),
        "unsupported_chars": unsupported,
        "unsupported_rate": round(unsupported / max(1, len(actual_norm)), 4),
        "missing_chars": missing,
        "missing_rate": round(missing / max(1, len(expected_norm)), 4),
    }


def _fit_font(draw: ImageDraw.ImageDraw, text: str, width: int, height: int) -> ImageFont.FreeTypeFont:
    size = max(12, int(height * 0.78))
    while size > 10:
        font = ImageFont.truetype(str(DEFAULT_FONT), size=size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max(1, width) and bbox[3] - bbox[1] <= max(1, height):
            return font
        size -= 1
    return ImageFont.truetype(str(DEFAULT_FONT), size=10)


def _render_synthetic_cases(fixtures: Path, output_dir: Path) -> list[dict[str, Any]]:
    fixture = json.loads(fixtures.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    for case in fixture["cases"]:
        image = Image.new("RGB", (int(case["width"]), int(case["height"])), "white")
        draw = ImageDraw.Draw(image)
        for block in case["blocks"]:
            x_min, y_min, x_max, y_max = map(int, block["bbox"])
            font = _fit_font(draw, block["text"], x_max - x_min, y_max - y_min)
            draw.text((x_min, y_min), block["text"], font=font, fill="black")
        path = output_dir / f"synthetic-{case['id']}.png"
        image.save(path, format="PNG")
        cases.append({
            "id": f"synthetic/{case['id']}",
            "group": f"synthetic-{case['group']}",
            "path": path,
            "expected": "\n".join(case["expected"]),
        })
    return cases


def _acceptance_cases() -> list[dict[str, Any]]:
    case_dir = ROOT / "acceptance_testset" / "files" / "cv"
    cases: list[dict[str, Any]] = []
    for image_path in sorted(case_dir.glob("*.png")):
        docx_path = image_path.with_suffix(".docx")
        if not docx_path.exists():
            continue
        document = resume_io.DocxDocument(str(docx_path))
        expected = resume_io._extract_text_from_docx_document(document)
        if expected:
            cases.append({
                "id": f"acceptance/{image_path.stem}",
                "group": "acceptance",
                "path": image_path,
                "expected": expected,
            })
    return cases


def _run_ppocr(path: Path) -> tuple[str, float]:
    payload = path.read_bytes()
    started = time.perf_counter()
    text = resume_io.extract_text_from_image_bytes(payload, path.name)
    return text, time.perf_counter() - started


def _load_json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```")
        stripped = stripped.rsplit("```", 1)[0].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def _vl_model(endpoint: str) -> str:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{endpoint.rstrip('/')}/v1/models", timeout=30) as response:
        payload = json.loads(response.read())
    return payload["data"][0]["id"]


def _run_vl(path: Path, endpoint: str, model: str) -> tuple[str, float]:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    prompt = (
        "你是简历页面逐字转录器。按人类阅读顺序抄录图片中的全部可见文字。"
        "不得解释、改写、补充、推断、合并或纠错；保留数字、标点和每个文本块。"
        "只输出严格 JSON：{\"lines\":[\"第一块文字\",\"第二块文字\"]}。"
    )
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
            ],
        }],
    }
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.perf_counter()
    with opener.open(request, timeout=480) as response:
        payload = json.loads(response.read())
    elapsed = time.perf_counter() - started
    content = payload["choices"][0]["message"].get("content", "")
    parsed = _load_json_object(content)
    lines = parsed.get("lines", []) if isinstance(parsed, dict) else []
    if not isinstance(lines, list):
        raise ValueError("VL response did not contain a lines array")
    return "\n".join(str(line).strip() for line in lines if str(line).strip()), elapsed


def _summary(results: list[dict[str, Any]], backend: str) -> dict[str, Any]:
    rows = [row[backend] for row in results if backend in row and "error" not in row[backend]]
    if not rows:
        return {"cases": 0}
    return {
        "cases": len(rows),
        "exact_cases": sum(row["metrics"]["exact"] for row in rows),
        "mean_ordered_char_recall": round(statistics.mean(row["metrics"]["ordered_char_recall"] for row in rows), 4),
        "mean_ordered_char_precision": round(statistics.mean(row["metrics"]["ordered_char_precision"] for row in rows), 4),
        "total_unsupported_chars": sum(row["metrics"]["unsupported_chars"] for row in rows),
        "total_missing_chars": sum(row["metrics"]["missing_chars"] for row in rows),
        "median_latency_s": round(statistics.median(row["latency_s"] for row in rows), 3),
        "max_latency_s": round(max(row["latency_s"] for row in rows), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=ROOT / "tests" / "fixtures" / "multicolumn_layout_cases.json")
    parser.add_argument("--render-dir", type=Path, default=ARTIFACT_ROOT / "pages")
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "ocr-vl-ab.json")
    parser.add_argument("--vl-endpoint")
    parser.add_argument("--vl-model")
    parser.add_argument("--reuse-vl-from", type=Path)
    parser.add_argument("--skip-acceptance", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not DEFAULT_FONT.is_file():
        raise SystemExit(f"CJK font is missing: {DEFAULT_FONT}")
    os.environ.setdefault("PPOCRV6_MODEL_DIR", str(ROOT / "models_slim" / "ppocrv6-small-ort"))
    os.environ.setdefault("RAPID_OCR_DEVICE", "cpu")
    os.environ.setdefault("RAPID_OCR_CPU_THREADS", "2")
    resume_io._init_rapid_ocr()
    if resume_io._RAPID_OCR is None:
        raise SystemExit("PP-OCR failed to initialize")

    cases = _render_synthetic_cases(args.fixtures, args.render_dir)
    if not args.skip_acceptance:
        cases.extend(_acceptance_cases())
    if args.limit:
        cases = cases[: args.limit]

    reused_vl: dict[str, dict[str, Any]] = {}
    reused_report: dict[str, Any] = {}
    if args.reuse_vl_from:
        reused_report = json.loads(args.reuse_vl_from.read_text(encoding="utf-8"))
        reused_vl = {
            row["id"]: row
            for row in reused_report.get("results", [])
            if "vl" in row and "error" not in row["vl"]
        }

    model = args.vl_model or reused_report.get("vl_model")
    if args.vl_endpoint and not model:
        model = _vl_model(args.vl_endpoint)

    results: list[dict[str, Any]] = []
    for case in cases:
        row: dict[str, Any] = {
            "id": case["id"],
            "group": case["group"],
            "image": str(case["path"]),
            "expected": case["expected"],
        }
        try:
            text, elapsed = _run_ppocr(case["path"])
            row["ppocr"] = {
                "text": text,
                "latency_s": round(elapsed, 3),
                "metrics": _metrics(case["expected"], text),
            }
        except Exception as exc:  # keep the paired run inspectable
            row["ppocr"] = {"error": f"{type(exc).__name__}: {exc}"}
        if args.vl_endpoint:
            try:
                text, elapsed = _run_vl(case["path"], args.vl_endpoint, str(model))
                row["vl"] = {
                    "text": text,
                    "latency_s": round(elapsed, 3),
                    "metrics": _metrics(case["expected"], text),
                }
            except Exception as exc:
                row["vl"] = {"error": f"{type(exc).__name__}: {exc}"}
        elif case["id"] in reused_vl:
            previous = reused_vl[case["id"]]
            if previous.get("expected") != case["expected"]:
                row["vl"] = {"error": "reused VL reference has different ground truth"}
            else:
                row["vl"] = previous["vl"]
        results.append(row)

    report = {
        "dataset_revision": json.loads(args.fixtures.read_text(encoding="utf-8"))["revision"],
        "ppocr_model": {
            "version": resume_io._RAPID_OCR_VERSION,
            "type": resume_io._RAPID_OCR_MODEL_TYPE,
            "provider": resume_io._RAPID_OCR_PROVIDER,
        },
        "vl_model": model,
        "vl_reused_from": str(args.reuse_vl_from) if args.reuse_vl_from else None,
        "summary": {
            "ppocr": _summary(results, "ppocr"),
            "vl": _summary(results, "vl"),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
