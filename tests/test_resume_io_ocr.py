import io
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from PIL import Image

import resume_io


def _touch_bundle(root: Path, model_type: str) -> None:
    names = {
        "server": (
            "ch_PP-OCRv5_det_server.onnx",
            "ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx",
            "ch_PP-OCRv5_rec_server.onnx",
        ),
        "mobile": (
            "ch_PP-OCRv5_det_mobile.onnx",
            "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx",
            "ch_PP-OCRv5_rec_mobile.onnx",
        ),
    }[model_type]
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).touch()


def _sleep_forever(seconds: float) -> None:
    time.sleep(seconds)


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_cpu_auto_prefers_mobile_bundle(tmp_path):
    _touch_bundle(tmp_path, "server")
    _touch_bundle(tmp_path, "mobile")

    selected = resume_io._select_rapid_ocr_bundle(
        [tmp_path], prefer_server=False, forced_model_type="auto"
    )

    assert selected is not None
    assert selected["model_type"] == "mobile"
    assert selected["det"].name == "ch_PP-OCRv5_det_mobile.onnx"
    assert selected["rec"].name == "ch_PP-OCRv5_rec_mobile.onnx"


def test_cuda_auto_prefers_server_bundle(tmp_path):
    _touch_bundle(tmp_path, "server")
    _touch_bundle(tmp_path, "mobile")

    selected = resume_io._select_rapid_ocr_bundle(
        [tmp_path], prefer_server=True, forced_model_type="auto"
    )

    assert selected is not None
    assert selected["model_type"] == "server"


def test_auto_prefers_complete_ppocrv6_small_bundle(tmp_path):
    for name in ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt"):
        (tmp_path / name).touch()
    _touch_bundle(tmp_path, "mobile")

    selected = resume_io._select_rapid_ocr_bundle(
        [tmp_path], prefer_server=False, forced_model_type="auto"
    )

    assert selected is not None
    assert selected["version"] == "v6"
    assert selected["model_type"] == "small"
    assert selected["keys"].name == "keys.txt"
    assert selected["cls_version"] == "v4"


def test_incomplete_ppocrv6_bundle_falls_back_to_v5_mobile(tmp_path):
    for name in ("det.onnx", "rec.onnx", "cls.onnx"):
        (tmp_path / name).touch()
    _touch_bundle(tmp_path, "mobile")

    selected = resume_io._select_rapid_ocr_bundle(
        [tmp_path], prefer_server=False, forced_model_type="auto"
    )

    assert selected is not None
    assert selected["version"] == "v5"
    assert selected["model_type"] == "mobile"


def test_unreadable_candidate_is_skipped_instead_of_aborting(tmp_path, monkeypatch):
    _touch_bundle(tmp_path, "mobile")
    protected = Path("/protected-ocr-models")
    original_is_file = Path.is_file

    def guarded_is_file(path):
        if str(path).startswith(str(protected)):
            raise PermissionError("protected")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", guarded_is_file)

    selected = resume_io._select_rapid_ocr_bundle(
        [protected, tmp_path], prefer_server=False, forced_model_type="auto"
    )

    assert selected is not None
    assert selected["model_dir"] == tmp_path
    assert selected["version"] == "v5"


def test_mobile_only_bundle_does_not_reconstruct_server_paths(tmp_path):
    _touch_bundle(tmp_path, "mobile")

    selected = resume_io._select_rapid_ocr_bundle(
        [tmp_path], prefer_server=False, forced_model_type="mobile"
    )

    assert selected is not None
    assert all("mobile" in selected[key].name for key in ("det", "cls", "rec"))


def test_available_cpu_count_respects_cgroup_quota(monkeypatch):
    monkeypatch.setattr(resume_io.os, "sched_getaffinity", lambda _pid: set(range(128)))
    monkeypatch.setattr(resume_io, "_cgroup_cpu_limit", lambda: 2)

    assert resume_io._available_cpu_count() == 2


def test_prepare_ocr_image_normalizes_mode_and_bounds_pixels(monkeypatch):
    monkeypatch.setenv("RAPID_OCR_MAX_PIXELS", "6000000")
    monkeypatch.setenv("RAPID_OCR_MAX_LONG_EDGE", "3000")
    source = Image.new("RGBA", (4000, 3000), (255, 255, 255, 127))

    prepared, original_size = resume_io._prepare_ocr_image(_png_bytes(source))

    assert original_size == (4000, 3000)
    assert prepared.mode == "RGB"
    assert prepared.width * prepared.height <= 6_000_000
    assert max(prepared.size) <= 3000


def test_prepare_ocr_image_keeps_normal_resume_resolution():
    source = Image.new("L", (1300, 1800), 255)

    prepared, original_size = resume_io._prepare_ocr_image(_png_bytes(source))

    assert original_size == (1300, 1800)
    assert prepared.size == original_size
    assert prepared.mode == "RGB"


def test_multicandidate_reuses_global_engine_and_returns_first_success(monkeypatch):
    calls = []

    class Result:
        boxes = np.array([[[0, 0], [100, 0], [100, 20], [0, 20]]])
        txts = ("识别成功",)

    class Engine:
        def __call__(self, image):
            calls.append(image.shape)
            return Result()

    monkeypatch.setattr(resume_io, "_RAPID_OCR", Engine())
    monkeypatch.setenv("RAPID_OCR_FALLBACK_ATTEMPTS", "2")
    prepared = Image.new("RGB", (300, 200), "white")

    text = resume_io._ocr_image_multicandidate(b"unused", prepared_image=prepared)

    assert text == "识别成功"
    assert calls == [(200, 300, 3)]


def test_primary_handles_grayscale_upload(monkeypatch):
    class Result:
        boxes = np.array([[[0, 0], [100, 0], [100, 20], [0, 20]]])
        txts = ("hello",)

    monkeypatch.setattr(resume_io, "_RAPID_OCR", lambda image: Result())
    content = _png_bytes(Image.new("L", (200, 100), 255))

    text = resume_io._ocr_image_with_rapid(content)

    assert text == "hello"


def test_isolated_process_termination_is_real():
    context = mp.get_context("spawn")
    process = context.Process(target=_sleep_forever, args=(30.0,))
    process.start()

    resume_io._terminate_isolated_process(process, process_group_ready=False)

    assert not process.is_alive()
    assert process.exitcode is not None


def test_path_image_uses_the_same_isolated_byte_entrypoint(tmp_path, monkeypatch):
    source = tmp_path / "resume.png"
    source.write_bytes(_png_bytes(Image.new("RGB", (100, 100), "white")))
    captured = {}

    def fake_extract(content, filename):
        captured["content"] = content
        captured["filename"] = filename
        return "图片简历文本"

    monkeypatch.setattr(resume_io, "extract_text_from_bytes", fake_extract)

    assert resume_io.extract_text_from_path(str(source)) == "图片简历文本"
    assert captured["filename"] == "resume.png"
    assert captured["content"] == source.read_bytes()


def test_v5_bundle_carries_classifier_shape_for_rapidocr_391(tmp_path):
    _touch_bundle(tmp_path, "mobile")

    selected = resume_io._select_rapid_ocr_bundle(
        [tmp_path], prefer_server=False, forced_model_type="mobile"
    )

    assert selected is not None
    assert selected["cls_image_shape"] == [3, 80, 160]
