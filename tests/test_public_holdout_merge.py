from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "validation_sets/public_resume_holdout/merge_results.py"
SPEC = importlib.util.spec_from_file_location("public_holdout_merge", PATH)
assert SPEC and SPEC.loader
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


def _payload(base_url: str, rows: list[dict]) -> dict:
    return {
        "metadata": {
            "evaluator_version": "1",
            "version": "candidate",
            "image_digest": "digest",
            "base_url": base_url,
            "cases_sha256": "cases",
            "annotations_sha256": "annotations",
            "evaluator_hashes": {"a": "b"},
            "candidate_self_score_ignored": True,
            "shadow_v3_sealed": True,
            "selected_case_ids": [row["id"] for row in rows],
        },
        "rows": rows,
    }


def test_merge_reorders_disjoint_shards(monkeypatch) -> None:
    rows = {
        case_id: {
            "id": case_id,
            "request_ok": False,
            "audit_ok": False,
            "elapsed_s": 0,
            "scenario": "scenario1",
            "industry": "operations",
            "input_profile": "plain_text",
        }
        for case_id in ("a", "b", "c")
    }
    monkeypatch.setattr(MERGE.EVALUATOR, "summarize_rows", lambda values: {"ids": [row["id"] for row in values]})
    result = MERGE.merge(
        [_payload("one", [rows["c"]]), _payload("two", [rows["a"], rows["b"]])],
        ["a", "b", "c"],
    )
    assert [row["id"] for row in result["rows"]] == ["a", "b", "c"]
    assert result["summary"] == {"ids": ["a", "b", "c"]}
    assert result["metadata"]["base_url"] == ["one", "two"]


def test_merge_rejects_missing_case() -> None:
    try:
        MERGE.merge([_payload("one", [{"id": "a"}])], ["a", "b"])
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("incomplete merge should fail")


def test_partial_merge_keeps_master_order_and_records_selected_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        MERGE.EVALUATOR,
        "summarize_rows",
        lambda values: {"ids": [row["id"] for row in values]},
    )
    result = MERGE.merge(
        [_payload("one", [{"id": "c"}, {"id": "a"}])],
        ["a", "b", "c"],
        allow_partial=True,
    )

    assert [row["id"] for row in result["rows"]] == ["a", "c"]
    assert result["metadata"]["selected_case_ids"] == ["a", "c"]


def test_merge_rejects_different_evaluator() -> None:
    left = _payload("one", [{"id": "a"}])
    right = _payload("two", [{"id": "b"}])
    right["metadata"]["evaluator_hashes"] = {"a": "different"}
    try:
        MERGE.merge([left, right], ["a", "b"])
    except ValueError as exc:
        assert "evaluator_hashes" in str(exc)
    else:
        raise AssertionError("non-comparable merge should fail")
