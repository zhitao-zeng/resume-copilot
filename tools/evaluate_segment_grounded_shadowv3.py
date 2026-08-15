#!/usr/bin/env python3
"""Blind span-eligibility evaluation for the frozen A3 shadow extractor.

`shadow_v3` does not have human-reviewed semantic types or ownership labels.
This evaluator intentionally limits its claims to exact source localization,
candidate-span eligibility, eligible-unit coverage, critical additions, JD-only
negatives, completion, and latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from segment_grounded_extractor import GroundingValidationResult  # noqa: E402
from tools.evaluate_segment_grounded_shadow import (  # noqa: E402
    CRITICAL_TYPES,
    _atomic_write_json,
    _flatten_baseline,
    _flatten_candidate,
    _percentile,
    _ratio,
    _read_jsonl,
    run_shadow,
)


DATA_ROOT = ROOT / "validation_sets" / "public_resume_holdout"
CASES_PATH = DATA_ROOT / "shadow_v3" / "cases.jsonl"
ANNOTATIONS_PATH = DATA_ROOT / "shadow_v3" / "annotations.jsonl"
SOURCE_ID = {"cv": "resume", "query": "query", "jd": "jd"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_blind_cases() -> list[dict[str, Any]]:
    cases = {str(case["id"]): case for case in _read_jsonl(CASES_PATH)}
    annotations = {
        str(annotation["case_id"]): annotation
        for annotation in _read_jsonl(ANNOTATIONS_PATH)
    }
    if cases.keys() != annotations.keys():
        raise ValueError("shadow_v3 case and annotation IDs differ")

    loaded: list[dict[str, Any]] = []
    for case_id in sorted(cases):
        case = cases[case_id]
        annotation = annotations[case_id]
        if annotation.get("status") != "exact_spans_ready_semantic_human_review_pending":
            raise ValueError(f"{case_id}: unexpected annotation status")
        texts = {"resume": "", "query": "", "jd": ""}
        eligible_units: list[dict[str, Any]] = []
        for source in annotation.get("sources") or []:
            kind = str(source["kind"])
            source_id = SOURCE_ID[kind]
            path = (DATA_ROOT / str(source["canonical_text_path"])).resolve()
            if not path.is_file() or DATA_ROOT.resolve() not in path.parents:
                raise ValueError(f"{case_id}: invalid canonical source path {path}")
            expected_hash = str(source.get("sha256") or "")
            if expected_hash and _sha256(path) != expected_hash:
                raise ValueError(f"{case_id}: canonical source hash mismatch")
            text = path.read_text(encoding="utf-8")
            texts[source_id] = text
            for unit in source.get("units") or []:
                if not unit.get("candidate_for_resume"):
                    continue
                start, end = unit["source_span"]
                start, end = int(start), int(end)
                if text[start:end] != str(unit["text"]):
                    raise ValueError(f"{case_id}: annotation span mismatch")
                eligible_units.append({
                    "unit_id": str(unit["fact_id"]),
                    "source_id": source_id,
                    "start": start,
                    "end": end,
                    "text": str(unit["text"]),
                    "section": str(unit.get("section") or "unknown"),
                })

        # Scenario instructions are useful context but are deliberately absent
        # from candidate-eligible annotations. Query-only cases already have a
        # canonical annotated query and must not be replaced by case metadata.
        if not texts["query"]:
            texts["query"] = str(case.get("query") or "")
        loaded.append({
            "id": case_id,
            "scenario": str(case["scenario"]),
            "industry": str(case["industry"]),
            "cv_text": texts["resume"],
            "query_text": texts["query"],
            "jd_text": texts["jd"],
            "expected_fields": [],
            "eligible_units": eligible_units,
        })
    return loaded


def _prediction_is_eligible(
    prediction: dict[str, Any], eligible_units: list[dict[str, Any]],
) -> bool:
    if prediction["field_type"] == "target_role":
        return True
    return bool(prediction["parts"]) and all(
        any(
            part["source_id"] == unit["source_id"]
            and int(unit["start"]) <= int(part["start"])
            and int(part["end"]) <= int(unit["end"])
            for unit in eligible_units
        )
        for part in prediction["parts"]
    )


def score_eligibility(
    predictions: list[dict[str, Any]], eligible_units: list[dict[str, Any]],
) -> dict[str, Any]:
    factual_predictions = [
        prediction
        for prediction in predictions
        if prediction["field_type"] != "target_role"
    ]
    eligible_predictions = [
        prediction
        for prediction in factual_predictions
        if _prediction_is_eligible(prediction, eligible_units)
    ]
    covered_units: set[str] = set()
    for unit in eligible_units:
        for prediction in factual_predictions:
            if any(
                part["source_id"] == unit["source_id"]
                and int(part["start"]) < int(unit["end"])
                and int(unit["start"]) < int(part["end"])
                for part in prediction["parts"]
            ):
                covered_units.add(str(unit["unit_id"]))
                break
    ineligible = [
        prediction
        for prediction in factual_predictions
        if not _prediction_is_eligible(prediction, eligible_units)
    ]
    critical_ineligible = [
        prediction
        for prediction in ineligible
        if prediction["field_type"] in CRITICAL_TYPES
    ]
    return {
        "counts": {
            "factual_predictions": len(factual_predictions),
            "eligible_predictions": len(eligible_predictions),
            "ineligible_predictions": len(ineligible),
            "critical_ineligible_predictions": len(critical_ineligible),
            "eligible_units": len(eligible_units),
            "covered_eligible_units": len(covered_units),
        },
        "candidate_span_precision": _ratio(
            len(eligible_predictions), len(factual_predictions),
        ),
        "eligible_unit_coverage": _ratio(
            len(covered_units), len(eligible_units),
        ),
        "critical_ineligible": critical_ineligible,
        "ineligible": ineligible,
    }


def _aggregate(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
    names = (
        "factual_predictions",
        "eligible_predictions",
        "ineligible_predictions",
        "critical_ineligible_predictions",
        "eligible_units",
        "covered_eligible_units",
    )
    counts = {
        name: sum(int(item[key]["counts"][name]) for item in items)
        for name in names
    }
    return {
        "counts": counts,
        "candidate_span_precision": _ratio(
            counts["eligible_predictions"], counts["factual_predictions"],
        ),
        "eligible_unit_coverage": _ratio(
            counts["covered_eligible_units"], counts["eligible_units"],
        ),
    }


def evaluate_blind(
    cases: list[dict[str, Any]], raw_results: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_by_id = {str(item["case_id"]): item for item in raw_results}
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        baseline = score_eligibility(
            _flatten_baseline(case), case["eligible_units"],
        )
        item: dict[str, Any] = {
            "case_id": case["id"],
            "scenario": case["scenario"],
            "industry": case["industry"],
            "baseline": baseline,
        }
        raw = raw_by_id.get(str(case["id"]))
        if raw is None or raw.get("error"):
            failure = raw or {"case_id": case["id"], "error": "missing result"}
            failures.append(failure)
            item["error"] = failure["error"]
            results.append(item)
            continue
        validation = GroundingValidationResult.model_validate(raw["validation"])
        predictions = _flatten_candidate(validation)
        shadow = score_eligibility(predictions, case["eligible_units"])
        item.update({
            "shadow": shadow,
            "localization": {
                "valid": validation.valid,
                "returned_references": validation.returned_reference_count,
                "valid_references": validation.valid_reference_count,
                "validity_rate": _ratio(
                    validation.valid_reference_count,
                    validation.returned_reference_count,
                ),
                "issues": [issue.model_dump() for issue in validation.issues],
            },
            "runtime": {
                "endpoint": raw.get("endpoint"),
                "elapsed_seconds": raw.get("elapsed_seconds"),
                "finish_reason": raw.get("finish_reason"),
            },
            "raw_extraction": raw.get("raw_extraction"),
        })
        results.append(item)

    completed = [item for item in results if "shadow" in item]
    report: dict[str, Any] = {
        "schema_version": 1,
        "semantic_field_labels_available": False,
        "ownership_labels_available": False,
        "layout_formats_available": False,
        "case_count": len(cases),
        "completed_count": len(completed),
        "failures": failures,
        "baseline": _aggregate(results, "baseline"),
        "cases": results,
    }
    if completed:
        report["shadow"] = _aggregate(completed, "shadow")
        returned = sum(item["localization"]["returned_references"] for item in completed)
        valid = sum(item["localization"]["valid_references"] for item in completed)
        latencies = [
            float(item["runtime"]["elapsed_seconds"])
            for item in completed
            if item["runtime"].get("elapsed_seconds") is not None
        ]
        report["shadow"].update({
            "localization": {
                "returned_references": returned,
                "valid_references": valid,
                "validity_rate": _ratio(valid, returned),
                "invalid_case_count": sum(
                    not item["localization"]["valid"] for item in completed
                ),
            },
            "latency": {
                "mean_seconds": statistics.fmean(latencies) if latencies else 0.0,
                "p95_seconds": _percentile(latencies, 0.95),
                "max_seconds": max(latencies, default=0.0),
            },
            "scenario4_candidate_fact_count": sum(
                item["shadow"]["counts"]["factual_predictions"]
                for item in completed
                if item["scenario"] == "scenario4"
            ),
        })

        grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
            "scenario": defaultdict(list),
            "industry": defaultdict(list),
        }
        for item in completed:
            grouped["scenario"][item["scenario"]].append(item)
            grouped["industry"][item["industry"]].append(item)
        report["shadow"]["groups"] = {
            dimension: {
                name: _aggregate(members, "shadow")
                for name, members in sorted(values.items())
            }
            for dimension, values in grouped.items()
        }
    return report


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--model", default="Qwen3.5-27B-AWQ")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    cases = load_blind_cases()
    endpoints = args.endpoint or ["http://127.0.0.1:8007"]
    raw = run_shadow(
        cases,
        endpoints=endpoints,
        model=args.model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
    )
    _atomic_write_json(args.raw_output, {
        "schema_version": 1,
        "model": args.model,
        "endpoints": endpoints,
        "results": raw,
    })
    report = evaluate_blind(cases, raw)
    report["dataset_sha256"] = hashlib.sha256(
        CASES_PATH.read_bytes() + ANNOTATIONS_PATH.read_bytes()
    ).hexdigest()
    report["model"] = args.model
    _atomic_write_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "case_count": report["case_count"],
        "completed_count": report["completed_count"],
        "baseline": report["baseline"],
        "shadow": report.get("shadow"),
        "failures": report["failures"],
    }, ensure_ascii=False, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
