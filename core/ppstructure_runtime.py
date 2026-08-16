"""Bounded PP-StructureV3 layout + OCR runtime.

Only raster pages reach this module.  Native PDF text remains on the cheap,
coordinate-aware path in :mod:`resume_io`; callers also retain the existing
OCR+BBOX engine as a deterministic fallback when this optional runtime cannot
initialize or infer.

The model is deliberately restricted to document layout and general OCR.
Table, formula, chart, seal, document-orientation, unwarping, and region
recognition modules are disabled so their weights are never needed in the
production image.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from typing import Any, Iterable


_PIPELINE: Any = None
_PIPELINE_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()

_MODEL_NAMES = {
    "layout_detection_model_dir": "PP-DocLayout_plus-L",
    "text_detection_model_dir": "PP-OCRv5_server_det",
    "textline_orientation_model_dir": "PP-LCNet_x1_0_textline_ori",
    "text_recognition_model_dir": "PP-OCRv5_server_rec",
}
_REQUIRED_MODEL_FILES = (
    "config.json",
    "inference.json",
    "inference.pdiparams",
    "inference.yml",
)


def ppstructure_enabled() -> bool:
    """Return whether the live StructureV3 parser is the requested engine."""

    return os.getenv("LAYOUT_ORDER_ENGINE", "bbox").strip().lower() == "ppstructure"


def _candidate_model_roots() -> list[Path]:
    configured = os.getenv("PPSTRUCTURE_MODEL_DIR", "").strip()
    candidates = [
        Path("/root/app/models/ppstructure-v3/official_models"),
        Path(__file__).resolve().parent.parent
        / "models_slim"
        / "ppstructure-v3"
        / "official_models",
    ]
    if configured:
        candidates.insert(0, Path(configured))
    return list(dict.fromkeys(candidates))


def _model_directories() -> dict[str, str]:
    for root in _candidate_model_roots():
        directories = {key: root / name for key, name in _MODEL_NAMES.items()}
        if all(
            all((directory / filename).is_file() for filename in _REQUIRED_MODEL_FILES)
            for directory in directories.values()
        ):
            return {key: str(directory) for key, directory in directories.items()}
    searched = ", ".join(str(path) for path in _candidate_model_roots())
    raise FileNotFoundError(
        "PP-StructureV3 model bundle is incomplete; searched: " + searched
    )


def _device() -> str:
    value = os.getenv("PPSTRUCTURE_DEVICE", "gpu:0").strip().lower()
    return value or "gpu:0"


def _ensure_device_available(device: str) -> None:
    if not device.startswith("gpu"):
        return
    import paddle

    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise RuntimeError("PP-StructureV3 GPU runtime is unavailable")


def _pipeline_options() -> dict[str, Any]:
    return {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": True,
        "use_seal_recognition": False,
        "use_table_recognition": False,
        "use_formula_recognition": False,
        "use_chart_recognition": False,
        "use_region_detection": False,
    }


def _create_pipeline() -> Any:
    # The four explicit model directories make initialization offline-only.
    # This flag also suppresses PaddleX's source connectivity probe.
    os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PPStructureV3

    device = _device()
    _ensure_device_available(device)
    return PPStructureV3(
        device=device,
        **_model_directories(),
        **_pipeline_options(),
    )


def _get_pipeline() -> Any:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = _create_pipeline()
    return _PIPELINE


def _configured_external_python() -> Path | None:
    configured = os.getenv("PPSTRUCTURE_PYTHON", "").strip()
    if not configured:
        return None
    interpreter = Path(configured)
    if not interpreter.is_file():
        raise FileNotFoundError(f"PP-StructureV3 interpreter not found: {interpreter}")
    # A virtual environment normally symlinks its executable to the base
    # interpreter.  Resolving both paths would therefore erase the venv's
    # distinct sys.prefix/site-packages and accidentally select the main
    # PyTorch environment.  Compare the configured executable paths without
    # following symlinks instead.
    if os.path.abspath(interpreter) == os.path.abspath(sys.executable):
        return None
    return interpreter


def _plain(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _prediction_payload(prediction: Any) -> dict[str, Any]:
    payload = getattr(prediction, "json", prediction)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    payload = _plain(payload)
    if not isinstance(payload, dict):
        raise ValueError("PP-StructureV3 returned a non-object prediction")
    root = payload.get("res", payload)
    if not isinstance(root, dict):
        raise ValueError("PP-StructureV3 prediction has no result object")
    return root


def _safe_order(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def ordered_blocks(prediction: Any, *, page: int | None = None) -> list[dict[str, Any]]:
    """Return non-empty parsing blocks without dropping StructureV3 fields."""

    root = _prediction_payload(prediction)
    rows = root.get("parsing_res_list") or []
    if not isinstance(rows, list):
        return []
    indexed: list[tuple[int, dict[str, Any]]] = [
        (index, row) for index, row in enumerate(rows) if isinstance(row, dict)
    ]
    indexed.sort(
        key=lambda entry: (
            _safe_order(entry[1].get("block_order"), 10**9 + entry[0]),
            _safe_order(entry[1].get("block_id"), 10**9 + entry[0]),
            entry[0],
        )
    )
    blocks: list[dict[str, Any]] = []
    for index, row in indexed:
        content = str(row.get("block_content") or "").strip()
        if not content:
            continue
        block = _plain(row)
        if not isinstance(block, dict):
            block = {}
        # Normalized keys are guaranteed while every additional PaddleX field
        # remains available to the V3 graph adapter and offline diagnostics.
        block.update({
            "block_order": _safe_order(row.get("block_order"), index + 1),
            "block_id": _safe_order(row.get("block_id"), index),
            "block_label": str(row.get("block_label") or ""),
            "block_bbox": _plain(row.get("block_bbox")),
            "block_content": content,
        })
        if page is not None:
            block["page"] = page
        blocks.append(block)
    return blocks


def blocks_from_predictions(predictions: Iterable[Any]) -> list[dict[str, Any]]:
    """Flatten page predictions into lossless, page-qualified block rows."""

    blocks: list[dict[str, Any]] = []
    for page_index, prediction in enumerate(predictions, start=1):
        blocks.extend(ordered_blocks(prediction, page=page_index))
    return blocks


def text_from_predictions(predictions: Iterable[Any]) -> str:
    pages: list[str] = []
    for prediction in predictions:
        blocks = ordered_blocks(prediction)
        text = "\n".join(block["block_content"] for block in blocks).strip()
        if text:
            pages.append(text)
    return "\n\n".join(pages).strip()


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff"}:
        suffix = ".png"
    return suffix


def _extract_path_in_process(path: Path) -> str:
    pipeline = _get_pipeline()
    with _RUN_LOCK:
        predictions = pipeline.predict(str(path), **_pipeline_options())
        return text_from_predictions(predictions)


def _extract_blocks_path_in_process(path: Path) -> list[dict[str, Any]]:
    pipeline = _get_pipeline()
    with _RUN_LOCK:
        predictions = pipeline.predict(str(path), **_pipeline_options())
        return blocks_from_predictions(predictions)


def _worker_timeout_seconds() -> float:
    try:
        value = float(os.getenv("PPSTRUCTURE_WORKER_TIMEOUT_SECONDS", "45"))
    except (TypeError, ValueError):
        value = 45.0
    return max(1.0, value)


def _extract_with_external_python(
    content: bytes,
    *,
    filename: str,
    interpreter: Path,
    output_format: str = "text",
) -> str | list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="resume-structure-") as temp_dir:
        root = Path(temp_dir)
        input_path = root / ("input" + _safe_suffix(filename))
        output_path = root / ("output.json" if output_format == "blocks" else "output.txt")
        input_path.write_bytes(content)
        completed = subprocess.run(
            [
                str(interpreter),
                str(Path(__file__).resolve()),
                "--worker-input",
                str(input_path),
                "--worker-output",
                str(output_path),
                "--worker-format",
                output_format,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=_worker_timeout_seconds(),
            check=False,
        )
        if completed.returncode != 0:
            diagnostics = (completed.stdout or "").strip()[-2000:]
            raise RuntimeError(
                f"PP-StructureV3 worker exited {completed.returncode}: {diagnostics}"
            )
        if not output_path.is_file():
            raise RuntimeError("PP-StructureV3 worker produced no output file")
        payload = output_path.read_text(encoding="utf-8").strip()
        if output_format == "blocks":
            decoded = json.loads(payload or "[]")
            if not isinstance(decoded, list) or not all(isinstance(item, dict) for item in decoded):
                raise RuntimeError("PP-StructureV3 worker produced invalid block JSON")
            return decoded
        return payload


def extract_ppstructure_text(content: bytes, *, filename: str) -> str:
    """Extract StructureV3 block-ordered text from one raster upload.

    Production uses an isolated Python environment because Paddle and vLLM's
    PyTorch build require different CUDA Python-package versions.  A local
    development environment can omit ``PPSTRUCTURE_PYTHON`` and run in-process.
    """

    if not content:
        return ""
    external_python = _configured_external_python()
    if external_python is not None:
        result = _extract_with_external_python(
            content,
            filename=filename,
            interpreter=external_python,
        )
        return str(result)
    suffix = _safe_suffix(filename)
    # A path follows the exact, validated PaddleX image decode path and avoids
    # RGB/BGR ambiguity in its ndarray input contract.
    with tempfile.NamedTemporaryFile(prefix="resume-structure-", suffix=suffix) as handle:
        handle.write(content)
        handle.flush()
        return _extract_path_in_process(Path(handle.name))


def extract_ppstructure_blocks(content: bytes, *, filename: str) -> list[dict[str, Any]]:
    """Extract raw StructureV3 blocks for the provenance-preserving V3 path."""

    if not content:
        return []
    external_python = _configured_external_python()
    if external_python is not None:
        result = _extract_with_external_python(
            content,
            filename=filename,
            interpreter=external_python,
            output_format="blocks",
        )
        return list(result) if isinstance(result, list) else []
    suffix = _safe_suffix(filename)
    with tempfile.NamedTemporaryFile(prefix="resume-structure-", suffix=suffix) as handle:
        handle.write(content)
        handle.flush()
        return _extract_blocks_path_in_process(Path(handle.name))


def release_ppstructure_runtime() -> None:
    """Release cached model references before falling back after a GPU error."""

    global _PIPELINE
    with _PIPELINE_LOCK:
        _PIPELINE = None
    gc.collect()
    try:
        import paddle

        if paddle.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception:
        pass


__all__ = [
    "blocks_from_predictions",
    "extract_ppstructure_blocks",
    "extract_ppstructure_text",
    "ordered_blocks",
    "ppstructure_enabled",
    "release_ppstructure_runtime",
    "text_from_predictions",
]


def _worker_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-input", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, required=True)
    parser.add_argument("--worker-format", choices=("text", "blocks"), default="text")
    args = parser.parse_args()
    if args.worker_format == "blocks":
        blocks = _extract_blocks_path_in_process(args.worker_input)
        args.worker_output.write_text(
            json.dumps(blocks, ensure_ascii=False), encoding="utf-8"
        )
    else:
        text = _extract_path_in_process(args.worker_input)
        args.worker_output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    _worker_main()
