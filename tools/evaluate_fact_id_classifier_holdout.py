#!/usr/bin/env python3
"""Run candidate-first classification on the frozen holdout-v2 source set.

Only source eligibility and runtime are scored because holdout-v2 annotations
do not provide semantic field or ownership labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from fact_id_classifier_shadow import (  # noqa: E402
    CandidateClassification,
    CandidateValidation,
    build_candidate_inventory,
    choices_to_predictions,
    validate_candidate_classification,
)
from tools.evaluate_fact_id_classifier_shadow import (  # noqa: E402
    DEFAULT_ENDPOINTS,
    _atomic_write_json,
    run_shadow,
)
from tools.evaluate_segment_grounded_shadow import _percentile, _read_jsonl  # noqa: E402
from tools.evaluate_segment_grounded_shadowv3 import (  # noqa: E402
    CRITICAL_TYPES,
    _aggregate,
    _ratio,
    score_eligibility,
)


DATA_ROOT = ROOT / "validation_sets" / "public_resume_holdout"
HOLDOUT_ROOT = DATA_ROOT / "holdout_v2"
CASES_PATH = HOLDOUT_ROOT / "cases.jsonl"
ANNOTATIONS_PATH = HOLDOUT_ROOT / "annotations.jsonl"
SOURCE_ID = {"cv": "resume", "query": "query", "jd": "jd"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_holdout_cases() -> list[dict[str, Any]]:
    cases = {str(item["id"]): item for item in _read_jsonl(CASES_PATH)}
    annotations = {
        str(item["case_id"]): item for item in _read_jsonl(ANNOTATIONS_PATH)
    }
    if cases.keys() != annotations.keys():
        raise ValueError("holdout-v2 case and annotation IDs differ")
    loaded: list[dict[str, Any]] = []
    for case_id in sorted(cases):
        case = cases[case_id]
        annotation = annotations[case_id]
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
                start, end = map(int, unit["source_span"])
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
        if not texts["query"]:
            texts["query"] = str(case.get("query") or "")
        loaded.append({
            "id": case_id,
            "scenario": str(case.get("scenario") or ""),
            "industry": str(case.get("industry") or ""),
            "input_profile": str(case.get("input_profile") or ""),
            "cv_text": texts["resume"],
            "query_text": texts["query"],
            "jd_text": texts["jd"],
            "expected_fields": [],
            "eligible_units": eligible_units,
        })
    return loaded


def evaluate_holdout(
    cases: list[dict[str, Any]],
    raw_results: list[dict[str, Any]],
) -> dict[str, Any]:
    by_id = {str(item["case_id"]): item for item in raw_results}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        raw = by_id.get(str(case["id"]))
        row: dict[str, Any] = {
            "case_id": case["id"],
            "scenario": case["scenario"],
            "industry": case["industry"],
            "input_profile": case["input_profile"],
            "eligible_units": len(case["eligible_units"]),
        }
        if raw is None or raw.get("error"):
            failure = raw or {"case_id": case["id"], "error": "missing result"}
            failures.append(failure)
            row["error"] = failure["error"]
            rows.append(row)
            continue
        _, candidates = build_candidate_inventory(
            case["cv_text"], case["query_text"], case["jd_text"]
        )
        extraction = CandidateClassification.model_validate(
            raw.get("raw_extraction") or {"choices": []}
        )
        choice_ids = [choice.candidate_id for choice in extraction.choices]
        row["duplicate_choice_count"] = len(choice_ids) - len(set(choice_ids))
        validation = validate_candidate_classification(
            extraction,
            candidates,
            allow_unassigned_records=True,
            allow_duplicate_choices=True,
            allow_partial_candidates=True,
        )
        row["dropped_choice_count"] = max(
            0, len(extraction.choices) - len(validation.choices)
        )
        row["candidate_count"] = len(candidates)
        row["runtime"] = {
            key: raw.get(key)
            for key in ("endpoint", "elapsed_seconds", "wall_seconds", "finish_reason")
            if raw.get(key) is not None
        }
        row["validation"] = validation.model_dump()
        if validation.valid:
            predictions = choices_to_predictions(validation)
        else:
            predictions = []
            row["invalid_validation"] = True
        row["eligibility"] = score_eligibility(
            predictions, case["eligible_units"],
        )
        row["predictions"] = predictions
        rows.append(row)

    completed = [row for row in rows if "eligibility" in row]
    aggregate = _aggregate(completed, "eligibility") if completed else None
    latencies = [
        float(row["runtime"]["elapsed_seconds"])
        for row in completed
        if row.get("runtime", {}).get("elapsed_seconds") is not None
    ]
    counts = aggregate["counts"] if aggregate else {
        "factual_predictions": 0,
        "eligible_predictions": 0,
        "ineligible_predictions": 0,
        "critical_ineligible_predictions": 0,
        "eligible_units": 0,
        "covered_eligible_units": 0,
    }
    return {
        "schema_version": 1,
        "dataset": "holdout_v2",
        "dataset_sha256": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
        "annotation_sha256": hashlib.sha256(ANNOTATIONS_PATH.read_bytes()).hexdigest(),
        "annotation_semantic_labels_available": False,
        "annotation_ownership_labels_available": False,
        "case_count": len(cases),
        "completed_case_count": len(completed),
        "completion_rate": _ratio(len(completed), len(cases)),
        "eligibility": aggregate,
        "counts": counts,
        "critical_ineligible": [
            prediction
            for row in completed
            for prediction in row["eligibility"].get("critical_ineligible", [])
        ],
        "invalid_validation_case_count": sum(
            bool(row.get("invalid_validation")) for row in rows
        ),
        "duplicate_choice_count": sum(
            int(row.get("duplicate_choice_count") or 0) for row in rows
        ),
        "dropped_choice_count": sum(
            int(row.get("dropped_choice_count") or 0) for row in rows
        ),
        "latency": {
            "mean_seconds": statistics.fmean(latencies) if latencies else 0.0,
            "p95_seconds": _percentile(latencies, 0.95),
            "max_seconds": max(latencies, default=0.0),
        },
        "failures": failures,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--model", default="Qwen3.5-27B-AWQ")
    parser.add_argument("--hard-timeout", type=float, default=150.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate-raw", type=Path, action="append")
    args = parser.parse_args()
    cases = load_holdout_cases()
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["id"] in selected]
    if args.evaluate_raw:
        results: list[dict[str, Any]] = []
        endpoints: list[str] = []
        for raw_path in args.evaluate_raw:
            loaded = json.loads(raw_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                results.extend(list(loaded.get("results") or []))
                endpoints.extend(list(loaded.get("endpoints") or []))
            else:
                results.extend(list(loaded))
        endpoints = list(dict.fromkeys(endpoints or args.endpoint or []))
    else:
        endpoints = args.endpoint or list(DEFAULT_ENDPOINTS)
        results = run_shadow(
            cases,
            endpoints=endpoints,
            model=args.model,
            hard_timeout=args.hard_timeout,
            max_tokens=args.max_tokens,
        )
        _atomic_write_json(args.raw_output, {
            "schema_version": 1,
            "model": args.model,
            "endpoints": endpoints,
            "results": results,
        })
    report = evaluate_holdout(cases, results)
    _atomic_write_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "completion_rate": report["completion_rate"],
        "eligibility": report["eligibility"],
        "latency": report["latency"],
        "invalid_validation_case_count": report["invalid_validation_case_count"],
        "failure_count": len(report["failures"]),
    }, ensure_ascii=False, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
