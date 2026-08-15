import json

from tools.run_pipeline_ablation import _load_cases, _merge_shards, _profile_env


def test_profile_env_never_allocates_gpu_zero_to_two(monkeypatch):
    monkeypatch.delenv("LOCAL_EVAL_GPU_IDS", raising=False)
    env = _profile_env("local_repair")
    assert env["LOCAL_EVAL_GPU_IDS"] == "3,4,5,6"
    assert env["LOCAL_EVAL_PIPELINE_PROFILE"] == "local_repair"


def test_merge_shards_preserves_rows_and_latency(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps({"rows": [{"id": "a", "ok": True, "elapsed_s": 2}]}))
    second.write_text(json.dumps({"rows": [{"id": "b", "ok": False, "elapsed_s": 4}]}))
    output = tmp_path / "merged.json"

    _merge_shards([first, second], output)

    payload = json.loads(output.read_text())
    assert payload["summary"] == {
        "case_count": 2,
        "success_count": 1,
        "failure_count": 1,
        "average_elapsed_s": 3.0,
        "max_elapsed_s": 4.0,
    }
    assert [row["id"] for row in payload["rows"]] == ["a", "b"]


def test_frozen_narrative_development_set_has_twelve_unique_cases():
    cases = _load_cases(
        __import__("pathlib").Path("validation_sets/narrative_development/cases.jsonl")
    )
    assert len(cases) == 12
    assert len({case["id"] for case in cases}) == 12
