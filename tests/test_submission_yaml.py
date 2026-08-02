"""Regression checks for the platform submission manifest."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict)
    return value


def test_latest_submission_yaml_is_minimal_and_mirrored():
    root_manifest = _load(ROOT / "leaderboard.yml")
    config_manifest = _load(ROOT / "config" / "leaderboard.yml")

    assert root_manifest == config_manifest
    assert set(root_manifest) == {"inferenceVolumeMounts", "inferenceImage"}
    assert root_manifest["inferenceVolumeMounts"] == [{
        "name": "model",
        "mountPoint": "/model",
        "source": "ceph_customer",
        "srcRelativePath": "zengzhitao/resume-copilot/models/Qwen3.5-27B-AWQ",
    }]
    assert root_manifest["inferenceImage"] == {
        "repository": "harbor-contest.4pd.io/zengzhitao/resume-copilot",
        "tag": "latest",
    }
