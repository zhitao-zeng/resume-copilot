import io
import json
import multiprocessing as mp
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

import resume_io
from tensorrt_ocr import (
    TensorRTConsensusOCR,
    TensorRTMediumRecognitionConsensus,
    TensorRTRecognitionConsensus,
)


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


def _touch_tensorrt_bundle(root: Path, *, missing: str = "") -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        "det.engine",
        "cls.engine",
        "primary-rec.engine",
        "secondary-rec.engine",
        "medium-rec.engine",
        "keys.txt",
    ):
        if name != missing:
            (root / name).touch()


def _sleep_forever(seconds: float) -> None:
    time.sleep(seconds)


def test_ppstructure_is_primary_raster_parser(tmp_path, monkeypatch):
    image_path = tmp_path / "resume.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    monkeypatch.setattr(resume_io, "ppstructure_enabled", lambda: True)
    monkeypatch.setattr(
        resume_io,
        "extract_ppstructure_text",
        lambda _content, *, filename: "左栏经历\n右栏技能",
    )
    monkeypatch.setattr(
        resume_io,
        "_init_rapid_ocr",
        lambda: (_ for _ in ()).throw(AssertionError("RapidOCR must not initialize")),
    )

    text = resume_io.extract_text_from_image_bytes(
        image_path.read_bytes(), image_path.name
    )

    assert text == "左栏经历\n右栏技能"


def test_ppstructure_failure_falls_back_to_existing_ocr_bbox(tmp_path, monkeypatch):
    image_path = tmp_path / "resume.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    released = []
    monkeypatch.setattr(resume_io, "ppstructure_enabled", lambda: True)
    monkeypatch.setattr(
        resume_io,
        "extract_ppstructure_text",
        lambda _content, *, filename: (_ for _ in ()).throw(RuntimeError("GPU OOM")),
    )
    monkeypatch.setattr(
        resume_io, "release_ppstructure_runtime", lambda: released.append(True)
    )
    monkeypatch.setattr(resume_io, "_RAPID_OCR_INITED", True)
    monkeypatch.setattr(resume_io, "_RAPID_OCR", object())
    monkeypatch.setattr(
        resume_io,
        "_ocr_image_with_rapid",
        lambda _content, *, prepared_image: "OCR+BBOX 回退内容",
    )

    text = resume_io.extract_text_from_image_bytes(
        image_path.read_bytes(), image_path.name
    )

    assert text == "OCR+BBOX 回退内容"
    assert released


def test_ppstructure_hybrid_uses_only_geometry_with_rapidocr_text(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "resume.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    regions = [
        {"bbox": [0, 0, 80, 100], "confidence": 1.0, "order": 1},
        {"bbox": [120, 0, 200, 100], "confidence": 1.0, "order": 2},
    ]
    captured = {}
    monkeypatch.setattr(resume_io, "ppstructure_enabled", lambda: False)
    monkeypatch.setattr(resume_io, "_ppstructure_hybrid_enabled", lambda: True)
    monkeypatch.setattr(
        resume_io,
        "_ppstructure_regions_for_prepared_image",
        lambda _image, *, filename: regions,
    )
    monkeypatch.setattr(resume_io, "_RAPID_OCR_INITED", True)
    monkeypatch.setattr(resume_io, "_RAPID_OCR", object())

    def rapid_text(_content, *, prepared_image, layout_regions=None):
        captured["regions"] = layout_regions
        return "PP-OCRv6 保留文本"

    monkeypatch.setattr(resume_io, "_ocr_image_with_rapid", rapid_text)

    text = resume_io.extract_text_from_image_bytes(
        image_path.read_bytes(), image_path.name
    )

    assert text == "PP-OCRv6 保留文本"
    assert captured["regions"] == regions


def test_layout_numeric_witness_only_suppresses_explicit_disagreement():
    boxes = np.asarray([
        [[0, 0], [100, 0], [100, 20], [0, 20]],
        [[0, 30], [100, 30], [100, 50], [0, 50]],
    ], dtype=float)
    result = SimpleNamespace(
        boxes=boxes,
        txts=("日均处理超过5个预约", "管理1.1亿美元业务"),
    )
    regions = [
        {
            "bbox": [0, 0, 100, 20],
            "text": "日均处理超过50个预约",
            "confidence": 1.0,
        },
        {
            "bbox": [0, 30, 100, 50],
            "text": "管理1.1亿美元业务",
            "confidence": 1.0,
        },
    ]

    reconciled = resume_io._apply_layout_numeric_witness(result, regions)

    assert reconciled.txts[0] == "日均处理超过个预约"
    assert reconciled.txts[1] == "管理1.1亿美元业务"


def test_layout_numeric_witness_abstains_when_structure_has_no_number():
    result = SimpleNamespace(
        boxes=np.asarray([[[0, 0], [100, 0], [100, 20], [0, 20]]], dtype=float),
        txts=("完成10次访谈",),
    )

    reconciled = resume_io._apply_layout_numeric_witness(result, [{
        "bbox": [0, 0, 100, 20],
        "text": "完成用户访谈",
        "confidence": 1.0,
    }])

    assert reconciled.txts == ("完成10次访谈",)


def test_layout_lexical_witness_uses_secondary_only_on_independent_two_of_three_vote():
    result = SimpleNamespace(
        boxes=np.asarray([
            [[0, 0], [100, 0], [100, 20], [0, 20]],
            [[0, 30], [100, 30], [100, 50], [0, 50]],
            [[0, 60], [100, 60], [100, 80], [0, 80]],
        ], dtype=float),
        txts=(
            "檀长招聘、培养团队",
            "分析国内市场",
            "完成10次访谈",
        ),
        ocr_secondary_txts=(
            "擅长招聘、培养团队",
            "分折国内市场",
            "完成11次访谈",
        ),
    )
    regions = [
        {"bbox": [0, 0, 100, 20], "text": "擅长招聘、培养团队"},
        # Structure agrees with the primary, so the secondary is rejected.
        {"bbox": [0, 30, 100, 50], "text": "分析国内市场"},
        # Numeric lines remain under the stricter numeric consensus.
        {"bbox": [0, 60, 100, 80], "text": "完成11次访谈"},
    ]

    reconciled = resume_io._apply_layout_lexical_witness(result, regions)

    assert reconciled.txts == (
        "擅长招聘、培养团队",
        "分析国内市场",
        "完成10次访谈",
    )


def test_layout_lexical_witness_abstains_on_large_or_unaligned_disagreement():
    result = SimpleNamespace(
        boxes=np.asarray([[[0, 0], [100, 0], [100, 20], [0, 20]]], dtype=float),
        txts=("负责客户运营管理",),
        ocr_secondary_txts=("主导平台架构设计",),
    )

    reconciled = resume_io._apply_layout_lexical_witness(result, [{
        "bbox": [0, 0, 100, 20],
        "text": "主导平台架构设计",
    }])

    assert reconciled.txts == ("负责客户运营管理",)


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


def test_tensorrt_bundle_requires_all_three_recognition_heads(tmp_path):
    complete = tmp_path / "complete"
    incomplete = tmp_path / "incomplete"
    _touch_tensorrt_bundle(complete)
    _touch_tensorrt_bundle(incomplete, missing="medium-rec.engine")

    assert resume_io._select_tensorrt_ocr_bundle([incomplete, complete]) == complete
    assert resume_io._select_tensorrt_ocr_bundle([incomplete]) is None


def test_tensorrt_hybrid_bundle_only_requires_two_witnesses(tmp_path):
    for name in ("secondary-rec.engine", "medium-rec.engine", "keys.txt"):
        (tmp_path / name).touch()

    assert resume_io._select_tensorrt_ocr_bundle(
        [tmp_path], recognition_only=True
    ) == tmp_path
    assert resume_io._select_tensorrt_ocr_bundle([tmp_path]) is None


def test_tensorrt_medium_primary_bundle_requires_both_small_witnesses(tmp_path):
    for name in (
        "primary-rec.engine",
        "secondary-rec.engine",
        "medium-rec.engine",
        "keys.txt",
    ):
        (tmp_path / name).touch()

    assert resume_io._select_tensorrt_ocr_bundle(
        [tmp_path], recognition_only=True, medium_primary=True
    ) == tmp_path
    (tmp_path / "primary-rec.engine").unlink()
    assert resume_io._select_tensorrt_ocr_bundle(
        [tmp_path], recognition_only=True, medium_primary=True
    ) is None


def test_tensorrt_recognition_padding_matches_rapidocr_normalized_zero():
    crop = np.full((24, 48, 3), 255, dtype=np.uint8)

    padded, _ratio = TensorRTConsensusOCR._prepare_recognition_crop(crop)
    classifier = TensorRTConsensusOCR._prepare_classifier_crop(crop)

    assert padded.dtype == np.float32
    assert classifier.dtype == np.float32
    assert np.all(padded[:, -1] == 0.0)
    assert np.all(classifier[:, -1] == 0.0)
    assert np.allclose(padded[:, :96], 1.0)


def test_tensorrt_native_height_padding_preserves_source_raster():
    crop = np.zeros((24, 48, 3), dtype=np.uint8)
    crop[:, :24] = 255

    padded, ratio = TensorRTConsensusOCR._prepare_recognition_crop(
        crop,
        preprocess_mode="native-height-pad",
    )

    assert padded.shape == (48, 320, 3)
    assert ratio == 2.0
    assert np.allclose(padded[:24, :24], 1.0)
    assert np.allclose(padded[:24, 24:48], -1.0)
    assert np.allclose(padded[24:, :], 0.0)


def test_tensorrt_native_height_padding_falls_back_for_tall_crop():
    crop = np.full((96, 48, 3), 255, dtype=np.uint8)

    native, native_ratio = TensorRTConsensusOCR._prepare_recognition_crop(
        crop,
        preprocess_mode="native-height-pad",
    )
    standard, standard_ratio = TensorRTConsensusOCR._prepare_recognition_crop(
        crop,
        preprocess_mode="standard-resize",
    )

    assert native_ratio == standard_ratio
    assert np.array_equal(native, standard)


def test_tensorrt_native_height_preparation_keeps_common_width_batchable():
    engine = object.__new__(TensorRTConsensusOCR)
    engine._np = np
    engine._recognition_preprocess = "native-height-pad"
    crops = [
        np.zeros((24, 48, 3), dtype=np.uint8),
        np.zeros((32, 80, 3), dtype=np.uint8),
    ]

    prepared = engine._prepare_recognition_crops(crops)

    assert [entry[1].shape for entry in prepared] == [
        (48, 320, 3),
        (48, 320, 3),
    ]

    standard = engine._prepare_recognition_crops(
        crops,
        preprocess_mode="standard-resize",
    )
    assert np.allclose(standard[0][1][:, :96], -1.0)
    assert not np.array_equal(prepared[0][1], standard[0][1])


def test_explicit_tensorrt_backend_is_selected_without_onnx_cuda(tmp_path, monkeypatch):
    _touch_tensorrt_bundle(tmp_path)
    fake_engine = object()
    monkeypatch.setenv("RAPID_OCR_BACKEND", "tensorrt")
    monkeypatch.setenv("PPOCRV6_TRT_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(resume_io, "_RAPID_OCR", None)
    monkeypatch.setattr(resume_io, "_RAPID_OCR_INITED", False)
    monkeypatch.setattr(
        resume_io,
        "_build_tensorrt_recognition_consensus",
        lambda _root, **_kwargs: fake_engine,
    )

    resume_io._init_rapid_ocr()

    assert resume_io._RAPID_OCR is fake_engine
    assert resume_io._RAPID_OCR_PROVIDER == "TensorRTMediumRecognition+CPUDetection"
    assert (
        resume_io._RAPID_OCR_MODEL_TYPE
        == "medium+two-small-trt-fp32-consensus"
    )


def test_unsupported_ppocrv6_runtime_falls_back_to_v5_mobile(tmp_path, monkeypatch):
    for name in ("det.onnx", "rec.onnx", "cls.onnx", "keys.txt"):
        (tmp_path / name).touch()
    _touch_bundle(tmp_path, "mobile")
    selected = resume_io._select_rapid_ocr_bundle(
        [tmp_path], prefer_server=False, forced_model_type="auto"
    )
    attempted = []

    def fake_build(bundle, *, use_cuda, cpu_threads):
        attempted.append((bundle["version"], bundle["model_type"]))
        if bundle["version"] == "v6":
            raise RuntimeError("runtime has no PPOCRV6 enum")
        return object()

    monkeypatch.setattr(resume_io, "_build_rapid_ocr", fake_build)

    engine, actual = resume_io._build_rapid_ocr_with_fallback(
        selected,
        candidates=[tmp_path],
        prefer_server=False,
        forced_model_type="auto",
        use_cuda=False,
        cpu_threads=2,
    )

    assert engine is not None
    assert actual["version"] == "v5"
    assert actual["model_type"] == "mobile"
    assert attempted == [("v6", "small"), ("v5", "mobile")]


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


def test_ocr_normalization_repairs_only_internal_numeric_parentheses():
    actual = resume_io._normalize_extracted_resume_text(
        "2004年\n2((4年\n35(0万元\n(2019年)\n负责2(19+1)个测试",
        repair_ocr_artifacts=True,
    )

    assert actual.splitlines() == [
        "2004年",
        "2004年",
        "3500万元",
        "(2019年)",
        "负责2019+1)个测试",
    ]


def test_ocr_normalization_canonicalizes_noisy_fixture_placeholders():
    actual = resume_io._normalize_extracted_resume_text(
        "【公司1\n［学校］\n[城市2]\n【项目】 数据平台\n【公司12012-2022",
        repair_ocr_artifacts=True,
    )

    assert actual.splitlines() == [
        "[公司]", "[学校]", "[城市]", "[项目] 数据平台", "[公司]2012-2022",
    ]


def test_numeric_consensus_suppression_keeps_clause_but_removes_anchor():
    assert resume_io._ocr_numeric_signature("收入35(0万元") == ("3500",)
    assert resume_io._suppress_disputed_numeric_anchors(
        "监督价值250万美元的业务运营"
    ) == "监督价值万美元的业务运营"
    assert resume_io._suppress_disputed_numeric_anchors(
        "2012年9月-至今"
    ) == "年月-至今"


def test_numeric_consensus_signature_keeps_ambiguous_glyph_evidence():
    assert resume_io._ocr_numeric_consensus_signature(
        "处理超过5(个预约。"
    ) == ("5", "<ambiguous-numeric-glyph>")
    assert resume_io._ocr_numeric_consensus_signature(
        "处理超过5个预约。"
    ) == ("5",)
    # A bracket between two digits is the established deterministic OCR-zero
    # repair, not an unmatched glyph after a complete number.
    assert resume_io._ocr_numeric_consensus_signature(
        "收入35(0万元"
    ) == ("3500",)
    assert resume_io._ocr_numeric_consensus_signature(
        "覆盖(2019年)项目"
    ) == ("2019",)


def test_numeric_consensus_rejects_same_digits_with_unmatched_glyph_witness(
    monkeypatch,
):
    class PrimaryEngine:
        @staticmethod
        def crop_text_regions(_image, boxes):
            return [np.zeros((8, 8, 3), dtype=np.uint8) for _ in boxes]

        @staticmethod
        def cls_and_rotate(crops):
            return crops, None

    class RecognitionResult:
        def __init__(self, txts):
            self.txts = tuple(txts)

    class SecondaryRecognizer:
        def __call__(self, _input):
            return RecognitionResult(("处理超过5(个预约。",))

    class MediumRecognizer:
        def __init__(self):
            self.called = False

        def __call__(self, _input):
            self.called = True
            return RecognitionResult(("处理超过5(个预约。",))

    class PrimaryResult:
        boxes = np.zeros((1, 4, 2), dtype=np.float32)
        txts = ("处理超过5个预约。",)

    medium = MediumRecognizer()
    monkeypatch.setattr(resume_io, "_RAPID_OCR", PrimaryEngine())
    monkeypatch.setattr(resume_io, "_RAPID_OCR_SECONDARY_REC", SecondaryRecognizer())
    monkeypatch.setattr(resume_io, "_RAPID_OCR_NUMERIC_REC", medium)
    monkeypatch.setattr(resume_io, "_init_numeric_ocr_consensus", lambda: True)

    result = resume_io._apply_numeric_ocr_consensus(
        np.zeros((40, 40, 3), dtype=np.uint8),
        PrimaryResult(),
    )

    # The independent small witness already proves the apparent agreement is
    # unsafe, so the medium head is skipped and the number is quarantined.
    assert medium.called is False
    assert result.txts == ("处理超过个预约。",)


def test_numeric_consensus_skips_medium_when_small_heads_already_disagree(monkeypatch):
    class PrimaryEngine:
        @staticmethod
        def crop_text_regions(_image, boxes):
            return [np.full((8, 8, 3), index, dtype=np.uint8) for index, _ in enumerate(boxes)]

        @staticmethod
        def cls_and_rotate(crops):
            return crops, None

    class RecognitionResult:
        def __init__(self, txts):
            self.txts = tuple(txts)

    class SecondaryRecognizer:
        def __call__(self, _input):
            return RecognitionResult(("负责11个项目", "覆盖20名用户", "无数字"))

    class MediumRecognizer:
        def __init__(self):
            self.crop_markers = []

        def __call__(self, rec_input):
            self.crop_markers = [int(crop[0, 0, 0]) for crop in rec_input.img]
            return RecognitionResult(("覆盖20名用户",))

    class PrimaryResult:
        boxes = np.zeros((3, 4, 2), dtype=np.float32)
        txts = ("负责10个项目", "覆盖20名用户", "无数字")

    medium = MediumRecognizer()
    monkeypatch.setattr(resume_io, "_RAPID_OCR", PrimaryEngine())
    monkeypatch.setattr(resume_io, "_RAPID_OCR_SECONDARY_REC", SecondaryRecognizer())
    monkeypatch.setattr(resume_io, "_RAPID_OCR_NUMERIC_REC", medium)
    monkeypatch.setattr(resume_io, "_init_numeric_ocr_consensus", lambda: True)

    result = resume_io._apply_numeric_ocr_consensus(
        np.zeros((40, 40, 3), dtype=np.uint8),
        PrimaryResult(),
    )

    assert medium.crop_markers == [1]
    assert result.txts == ("负责个项目", "覆盖20名用户", "无数字")
    assert result.ocr_secondary_txts == (
        "负责11个项目", "覆盖20名用户", "无数字",
    )


def test_tensorrt_result_does_not_run_cpu_consensus_twice(monkeypatch):
    result = SimpleNamespace(
        boxes=np.zeros((1, 4, 2), dtype=np.float32),
        txts=("覆盖20名用户",),
        numeric_consensus_applied=True,
    )
    monkeypatch.setattr(
        resume_io,
        "_init_numeric_ocr_consensus",
        lambda: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    assert resume_io._apply_numeric_ocr_consensus(
        np.zeros((40, 40, 3), dtype=np.uint8), result
    ) is result


def test_tensorrt_hybrid_keeps_cpu_primary_and_accelerates_only_witnesses():
    class Primary:
        def __call__(self, _image):
            return SimpleNamespace(
                boxes=np.zeros((3, 4, 2), dtype=np.float32),
                txts=("负责10个项目", "覆盖20名用户", "无数字"),
                scores=(0.99, 0.99, 0.99),
            )

        @staticmethod
        def crop_text_regions(_image, boxes):
            return [np.zeros((8, 8, 3), dtype=np.uint8) for _ in boxes]

        @staticmethod
        def cls_and_rotate(crops):
            return crops, None

    engine = object.__new__(TensorRTRecognitionConsensus)
    engine._np = np
    engine._primary_ocr = Primary()
    engine._secondary_rec = "secondary"
    engine._medium_rec = "medium"
    engine._numeric_signature = resume_io._ocr_numeric_signature
    engine._suppress_numeric = resume_io._suppress_disputed_numeric_anchors
    engine._repair_text = resume_io._repair_ocr_text_artifacts
    captured_medium_indexes = []

    def prepare(_crops, *, result_indexes=None, batch_size=6):
        del batch_size
        indexes = list(result_indexes) if result_indexes is not None else [0, 1, 2]
        if result_indexes is not None:
            captured_medium_indexes.extend(indexes)
        return [(index, np.zeros((48, 320, 3), dtype=np.float32), 1.0) for index in indexes]

    def recognize(session, _prepared, result_size):
        if session == "secondary":
            return [("负责11个项目", 0.99), ("覆盖20名用户", 0.99), ("无数字", 0.99)]
        values = [("", 0.0)] * result_size
        values[1] = ("覆盖20名用户", 0.99)
        return values

    engine._prepare_recognition_crops = prepare
    engine._recognize = recognize

    result = engine(np.zeros((40, 40, 3), dtype=np.uint8))

    assert captured_medium_indexes == [1]
    assert result.txts == ("负责个项目", "覆盖20名用户", "无数字")
    assert result.numeric_consensus_applied is True


def test_tensorrt_medium_primary_uses_small_heads_only_for_numeric_lines():
    class Primary:
        text_score = 0.5

        def __init__(self):
            self.calls = []

        def __call__(self, _image, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                boxes=np.arange(24, dtype=np.float32).reshape(3, 4, 2),
            )

        @staticmethod
        def crop_text_regions(_image, boxes):
            return [np.zeros((8, 8, 3), dtype=np.uint8) for _ in boxes]

        @staticmethod
        def cls_and_rotate(crops):
            return crops, None

    engine = object.__new__(TensorRTMediumRecognitionConsensus)
    engine._np = np
    engine._primary_ocr = Primary()
    engine._primary_rec = "small-primary"
    engine._secondary_rec = "small-secondary"
    engine._medium_rec = "medium"
    engine._numeric_signature = resume_io._ocr_numeric_signature
    engine._suppress_numeric = resume_io._suppress_disputed_numeric_anchors
    engine._repair_text = resume_io._repair_ocr_text_artifacts
    witness_indexes = []
    witness_preprocess = []

    def prepare(
        _crops,
        *,
        result_indexes=None,
        batch_size=6,
        preprocess_mode=None,
    ):
        del batch_size
        indexes = list(result_indexes) if result_indexes is not None else [0, 1, 2]
        if result_indexes is not None:
            witness_indexes.extend(indexes)
            witness_preprocess.append(preprocess_mode)
        return [
            (index, np.zeros((48, 320, 3), dtype=np.float32), 1.0)
            for index in indexes
        ]

    def recognize(session, _prepared, result_size):
        if session == "medium":
            return [
                ("负责10个项目", 0.99),
                ("覆盖20名用户", 0.99),
                ("改善流程", 0.99),
            ]
        values = [("", 0.0)] * result_size
        values[0] = ("负责11个项目", 0.99)
        values[1] = ("覆盖20名用户", 0.99)
        return values

    engine._prepare_recognition_crops = prepare
    engine._recognize = recognize

    result = engine(np.zeros((40, 40, 3), dtype=np.uint8))

    assert engine._primary_ocr.calls == [
        {"use_det": True, "use_cls": False, "use_rec": False}
    ]
    assert witness_indexes == [0, 1]
    assert witness_preprocess == ["standard-resize"]
    assert result.txts == ("负责个项目", "覆盖20名用户", "改善流程")
    assert result.numeric_consensus_applied is True


def test_tensorrt_runtime_error_falls_back_to_cpu_engine(monkeypatch):
    class Result:
        boxes = np.array([[[0, 0], [100, 0], [100, 20], [0, 20]]])
        txts = ("识别成功",)
        numeric_consensus_applied = True

    def failing_engine(_image):
        raise RuntimeError("TensorRT execution failed")

    def switch_to_cpu(_reason):
        monkeypatch.setattr(resume_io, "_RAPID_OCR", lambda _image: Result())
        monkeypatch.setattr(resume_io, "_RAPID_OCR_PROVIDER", "CPUExecutionProvider")
        return True

    monkeypatch.setattr(resume_io, "_RAPID_OCR", failing_engine)
    monkeypatch.setattr(resume_io, "_RAPID_OCR_PROVIDER", "TensorRTExecutionProvider")
    monkeypatch.setattr(resume_io, "_RAPID_OCR_VERSION", "v6")
    monkeypatch.setattr(resume_io, "_switch_rapid_ocr_to_cpu_mobile", switch_to_cpu)

    text = resume_io._ocr_image_with_rapid(
        _png_bytes(Image.new("RGB", (200, 100), "white"))
    )

    assert text == "识别成功"


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


def test_ppocrv6_primary_preserves_rgb_channels(monkeypatch):
    captured = {}

    class Result:
        boxes = np.array([[[0, 0], [100, 0], [100, 20], [0, 20]]])
        txts = ("彩色标题",)

    def engine(image):
        captured["image"] = image.copy()
        return Result()

    source = Image.new("RGB", (200, 100), (25, 120, 230))
    monkeypatch.setattr(resume_io, "_RAPID_OCR", engine)
    monkeypatch.setattr(resume_io, "_RAPID_OCR_VERSION", "v6")

    text = resume_io._ocr_image_with_rapid(_png_bytes(source))

    assert text == "彩色标题"
    assert tuple(captured["image"][0, 0]) == (25, 120, 230)


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


def test_multicolumn_layout_fixture_preserves_text_and_reading_order():
    fixture_path = Path(__file__).parent / "fixtures" / "multicolumn_layout_cases.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    for case in fixture["cases"]:
        boxes = []
        for block in case["blocks"]:
            x_min, y_min, x_max, y_max = block["bbox"]
            boxes.append([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ])
        actual = resume_io._reconstruct_ocr_reading_order(
            boxes,
            [block["text"] for block in case["blocks"]],
            img_width=case["width"],
            img_height=case["height"],
        )
        assert actual == case["expected"], case["id"]
        assert sorted(actual) == sorted(block["text"] for block in case["blocks"]), case["id"]


def test_native_pdf_text_layer_uses_coordinate_aware_column_order():
    def line(text, bbox):
        return {"bbox": bbox, "spans": [{"text": text}]}

    # Deliberately interleave the PDF object order: right, left, right, left.
    # Visual reading order should still consume the left sidebar before body.
    page_dict = {
        "blocks": [
            {"type": 0, "lines": [line("工作经历", [300, 120, 410, 145])]},
            {"type": 0, "lines": [line("个人技能", [40, 120, 150, 145])]},
            {"type": 0, "lines": [line("星河科技｜产品经理", [300, 165, 540, 190])]},
            {"type": 0, "lines": [line("SQL与数据分析", [40, 165, 180, 190])]},
            {"type": 0, "lines": [line("张晨简历", [40, 35, 300, 70])]},
        ]
    }

    class FakePage:
        rect = SimpleNamespace(width=600, height=840)

        def get_text(self, mode=None, **kwargs):
            if mode == "dict":
                return page_dict
            return "对象流中的错误顺序"

    assert resume_io._extract_native_pdf_page_text(FakePage()).splitlines() == [
        "张晨简历",
        "个人技能",
        "SQL与数据分析",
        "工作经历",
        "星河科技｜产品经理",
    ]
