import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

import ppstructure_runtime


def _complete_model_root(root: Path) -> Path:
    for model_name in ppstructure_runtime._MODEL_NAMES.values():
        directory = root / model_name
        directory.mkdir(parents=True)
        for filename in ppstructure_runtime._REQUIRED_MODEL_FILES:
            (directory / filename).touch()
    return root


def test_model_directories_require_complete_offline_bundle(tmp_path, monkeypatch):
    complete = _complete_model_root(tmp_path / "complete")
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    monkeypatch.setenv("PPSTRUCTURE_MODEL_DIR", str(incomplete))
    monkeypatch.setattr(
        ppstructure_runtime,
        "_candidate_model_roots",
        lambda: [incomplete, complete],
    )

    directories = ppstructure_runtime._model_directories()

    assert set(directories) == set(ppstructure_runtime._MODEL_NAMES)
    assert all(Path(value).parent == complete for value in directories.values())


def test_ordered_blocks_uses_block_order_and_ignores_empty_content():
    prediction = SimpleNamespace(
        json={
            "res": {
                "parsing_res_list": [
                    {
                        "block_order": 3,
                        "block_id": 9,
                        "block_label": "text",
                        "block_content": "第三段",
                    },
                    {
                        "block_order": 1,
                        "block_id": 4,
                        "block_label": "title",
                        "block_content": "个人简历",
                    },
                    {
                        "block_order": 2,
                        "block_id": 7,
                        "block_label": "text",
                        "block_content": "  ",
                    },
                ]
            }
        }
    )

    blocks = ppstructure_runtime.ordered_blocks(prediction)

    assert [block["block_content"] for block in blocks] == ["个人简历", "第三段"]
    assert ppstructure_runtime.text_from_predictions([prediction]) == "个人简历\n第三段"


def test_prediction_payload_accepts_json_string():
    prediction = SimpleNamespace(
        json=json.dumps(
            {
                "res": {
                    "parsing_res_list": [
                        {"block_order": "1", "block_content": "结构化内容"}
                    ]
                }
            },
            ensure_ascii=False,
        )
    )

    assert ppstructure_runtime.text_from_predictions([prediction]) == "结构化内容"


def test_extract_uses_file_path_to_avoid_rgb_bgr_ambiguity(monkeypatch):
    seen = {}

    class Pipeline:
        def predict(self, path, **options):
            source = Path(path)
            seen["suffix"] = source.suffix
            seen["content"] = source.read_bytes()
            seen["options"] = options
            return [
                {
                    "res": {
                        "parsing_res_list": [
                            {"block_order": 1, "block_content": "识别结果"}
                        ]
                    }
                }
            ]

    monkeypatch.setattr(ppstructure_runtime, "_get_pipeline", lambda: Pipeline())
    monkeypatch.delenv("PPSTRUCTURE_PYTHON", raising=False)

    text = ppstructure_runtime.extract_ppstructure_text(b"image-bytes", filename="cv.png")

    assert text == "识别结果"
    assert seen["suffix"] == ".png"
    assert seen["content"] == b"image-bytes"
    assert seen["options"] == ppstructure_runtime._pipeline_options()


def test_external_paddle_environment_is_used_when_configured(tmp_path, monkeypatch):
    interpreter = tmp_path / "python"
    interpreter.touch()
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output_path = Path(command[command.index("--worker-output") + 1])
        output_path.write_text("隔离环境识别结果", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="worker log")

    monkeypatch.setenv("PPSTRUCTURE_PYTHON", str(interpreter))
    monkeypatch.setattr(ppstructure_runtime.subprocess, "run", fake_run)

    text = ppstructure_runtime.extract_ppstructure_text(
        b"image-bytes", filename="resume.png"
    )

    assert text == "隔离环境识别结果"
    assert captured["command"][0] == str(interpreter)
    assert "--worker-input" in captured["command"]
    assert captured["kwargs"]["timeout"] == 45.0


def test_external_venv_symlink_is_not_mistaken_for_main_python(
    tmp_path, monkeypatch
):
    base_python = tmp_path / "base-python"
    base_python.touch()
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(base_python)
    monkeypatch.setattr(ppstructure_runtime.sys, "executable", str(base_python))
    monkeypatch.setenv("PPSTRUCTURE_PYTHON", str(venv_python))

    assert ppstructure_runtime._configured_external_python() == venv_python


def test_create_pipeline_uses_only_four_explicit_models(tmp_path, monkeypatch):
    root = _complete_model_root(tmp_path)
    captured = {}

    class Pipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = SimpleNamespace(PPStructureV3=Pipeline)
    monkeypatch.setitem(__import__("sys").modules, "paddleocr", fake_module)
    monkeypatch.setattr(ppstructure_runtime, "_ensure_device_available", lambda _device: None)
    monkeypatch.setattr(
        ppstructure_runtime,
        "_candidate_model_roots",
        lambda: [root],
    )
    monkeypatch.setenv("PPSTRUCTURE_DEVICE", "gpu:0")

    ppstructure_runtime._create_pipeline()

    assert captured["device"] == "gpu:0"
    assert captured["use_table_recognition"] is False
    assert captured["use_formula_recognition"] is False
    assert captured["use_chart_recognition"] is False
    assert captured["use_seal_recognition"] is False
    assert {
        key: Path(captured[key]).name for key in ppstructure_runtime._MODEL_NAMES
    } == ppstructure_runtime._MODEL_NAMES
