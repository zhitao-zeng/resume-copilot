from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "validation_sets/public_resume_holdout/compare.py"
SPEC = importlib.util.spec_from_file_location("public_holdout_compare", PATH)
assert SPEC and SPEC.loader
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def _row(case_id: str, precision: float, recall: float) -> dict:
    return {
        "id": case_id,
        "request_ok": True,
        "audit_ok": True,
        "response_contract": {"reply_components": {"生成方向": True}},
        "external_audit": {
            "atomic_factuality": {
                "precision": precision,
                "recall": recall,
                "source_fact_count": 2,
            },
            "structural_invariants": {
                category: {"added_count": 0}
                for category in COMPARE.CRITICAL_CATEGORIES
            },
        },
    }


def _payload(version: str, precisions: list[float], recalls: list[float]) -> dict:
    rows = [_row(str(index), precision, recall) for index, (precision, recall) in enumerate(zip(precisions, recalls))]
    summary = {
        "case_count": len(rows),
        "request_success_count": len(rows),
        "request_failure_count": 0,
        "audit_failure_count": 0,
        "atomic_factuality": {
            "micro_precision": sum(precisions) / len(precisions),
            "micro_recall": sum(recalls) / len(recalls),
        },
        "critical_additions": 0,
        "latency_seconds": {"max": 100},
        "groups": {},
    }
    return {
        "metadata": {
            "version": version,
            "image_digest": version,
            "selected_case_ids": [row["id"] for row in rows],
            "evaluator_hashes": {"evaluator": "same"},
            "cases_sha256": "cases",
            "annotations_sha256": "annotations",
        },
        "summary": summary,
        "rows": rows,
    }


def test_compare_reports_paired_improvement_and_selects_passing_candidate() -> None:
    baseline = _payload("baseline", [0.98, 0.98], [0.5, 0.6])
    candidate = _payload("candidate", [0.99, 0.99], [0.7, 0.8])
    report = COMPARE.compare([baseline, candidate])
    paired = report["paired_against_baseline"]["candidate"]
    assert paired["source_fact_recall"]["mean_delta"] == 0.2
    assert paired["source_fact_recall"]["improved_count"] == 2
    assert report["selected_version"] == "candidate"


def test_compare_rejects_different_evaluator_hashes() -> None:
    baseline = _payload("baseline", [0.98], [0.5])
    candidate = _payload("candidate", [0.99], [0.6])
    candidate["metadata"]["evaluator_hashes"] = {"evaluator": "different"}
    try:
        COMPARE.compare([baseline, candidate])
    except ValueError as exc:
        assert "different evaluator" in str(exc)
    else:
        raise AssertionError("comparison should reject non-comparable runs")


def test_lower_critical_addition_delta_is_counted_as_improvement() -> None:
    baseline = _payload("baseline", [0.98], [0.5])
    candidate = _payload("candidate", [0.99], [0.6])
    baseline["rows"][0]["external_audit"]["structural_invariants"]["role"]["added_count"] = 2
    candidate["rows"][0]["external_audit"]["structural_invariants"]["role"]["added_count"] = 1
    report = COMPARE.compare([baseline, candidate])
    metric = report["paired_against_baseline"]["candidate"]["critical_additions"]
    assert metric["mean_delta"] == -1.0
    assert metric["improved_count"] == 1
    assert metric["regressed_count"] == 0
