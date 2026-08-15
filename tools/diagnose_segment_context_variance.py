#!/usr/bin/env python3
"""Paired repeated diagnostic for A3 cross-source context and decode variance."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from record_window_extractor import SegmentWindow  # noqa: E402
from segment_grounded_extractor import (  # noqa: E402
    GroundingValidationResult,
    build_document_segments,
)
from tools.evaluate_record_window_shadow import (  # noqa: E402
    DEFAULT_ENDPOINTS,
    execute_window_hard_timeout,
)
from tools.evaluate_segment_grounded_shadow import (  # noqa: E402
    DEFAULT_CASES,
    _atomic_write_json,
    _documents,
    _flatten_candidate,
    _read_jsonl,
    _semantic_signature,
    resolve_gold,
    score_predictions,
)


CASE_IDS = ("SGD-FINANCE-06", "SGD-MULTI-PRODUCT-01")
CONDITIONS = ("full", "resume_only", "query_only")
REPEATS = 3
MAX_TOKENS = 3_072
HARD_TIMEOUT_SECONDS = 120.0


def condition_segments(case: dict[str, Any], condition: str):
    segments = [
        segment
        for segment in build_document_segments(_documents(case))
        if segment.source_type != "jd"
    ]
    if condition == "full":
        return segments
    if condition == "resume_only":
        return [segment for segment in segments if segment.source_type == "resume"]
    if condition == "query_only":
        return [segment for segment in segments if segment.source_type == "query"]
    raise ValueError(f"unknown condition {condition!r}")


def _prompt_container(
    case: dict[str, Any], condition: str, repeat: int,
) -> SegmentWindow:
    segments = condition_segments(case, condition)
    if not segments:
        raise ValueError(f"{case['id']}/{condition} has no candidate segments")
    # SegmentWindow is used only as the already-tested subprocess transport.
    # Every nested DocumentSegment retains its true source ID/type and offsets;
    # the synthetic outer ID has no prompt or validation semantics.
    return SegmentWindow(
        window_id=f"{case['id']}:{condition}:R{repeat}",
        source_id="diagnostic",
        source_type=segments[0].source_type,
        segments=tuple(segments),
        character_count=sum(len(segment.text) for segment in segments),
    )


def endpoint_for(
    *,
    item_index: int,
    repeat: int,
    endpoints: Sequence[str],
) -> str:
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    return endpoints[(item_index + repeat - 1) % len(endpoints)]


def _run_item(
    case: dict[str, Any],
    condition: str,
    repeat: int,
    endpoint: str,
    model: str,
) -> dict[str, Any]:
    container = _prompt_container(case, condition, repeat)
    result = execute_window_hard_timeout(
        container,
        endpoint=endpoint,
        model=model,
        hard_timeout=HARD_TIMEOUT_SECONDS,
        max_tokens=MAX_TOKENS,
    )
    return {
        "case_id": case["id"],
        "condition": condition,
        "repeat": repeat,
        "endpoint": endpoint,
        **result,
    }


def run_diagnostic(
    cases: Sequence[dict[str, Any]],
    *,
    endpoints: Sequence[str],
    model: str,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    case_by_id = {str(case["id"]): case for case in cases}
    missing = [case_id for case_id in CASE_IDS if case_id not in case_by_id]
    if missing:
        raise ValueError(f"missing diagnostic cases: {missing}")

    long_items = [
        (case_by_id[case_id], condition)
        for case_id in CASE_IDS
        for condition in ("full", "resume_only")
    ]
    query_items = [
        (case_by_id[case_id], "query_only") for case_id in CASE_IDS
    ]
    results: list[dict[str, Any]] = []
    for repeat in range(1, REPEATS + 1):
        for wave_name, wave_items in (("long", long_items), ("query", query_items)):
            with ThreadPoolExecutor(max_workers=len(wave_items)) as executor:
                futures = []
                for item_index, (case, condition) in enumerate(wave_items):
                    # Offset the short wave so it also rotates independently.
                    rotation_index = item_index + (0 if wave_name == "long" else 2)
                    endpoint = endpoint_for(
                        item_index=rotation_index,
                        repeat=repeat,
                        endpoints=endpoints,
                    )
                    futures.append(executor.submit(
                        _run_item,
                        case,
                        condition,
                        repeat,
                        endpoint,
                        model,
                    ))
                for future in as_completed(futures):
                    results.append(future.result())
            results.sort(key=lambda item: (
                int(item["repeat"]), str(item["case_id"]), str(item["condition"]),
            ))
            _atomic_write_json(checkpoint_path, {
                "schema_version": 1,
                "status": "running" if repeat < REPEATS or wave_name == "long" else "complete",
                "model": model,
                "endpoints": list(endpoints),
                "completed_requests": len(results),
                "expected_requests": len(CASE_IDS) * len(CONDITIONS) * REPEATS,
                "results": results,
            })
    return results


def _filtered_gold(case: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    gold = resolve_gold(case)
    if condition == "full":
        return gold
    source_id = "resume" if condition == "resume_only" else "query"
    return [item for item in gold if item["source_id"] == source_id]


def _labels_for_gold(
    predictions: list[dict[str, Any]], gold: list[dict[str, Any]],
) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for item in gold:
        key = f"{item['source_id']}:{item['start']}:{item['end']}:{item['quote']}"
        labels[key] = sorted({
            str(prediction["field_type"])
            for prediction in predictions
            if any(
                part["source_id"] == item["source_id"]
                and int(part["start"]) == int(item["start"])
                and int(part["end"]) == int(item["end"])
                for part in prediction["parts"]
            )
        })
    return labels


def classify_cause(
    full_labels: Sequence[Sequence[str]],
    isolated_labels: Sequence[Sequence[str]],
    *,
    expected_type: str,
) -> str:
    full_correct = sum(expected_type in labels for labels in full_labels)
    isolated_correct = sum(expected_type in labels for labels in isolated_labels)
    full_varies = len({tuple(labels) for labels in full_labels}) > 1
    isolated_varies = len({tuple(labels) for labels in isolated_labels}) > 1
    if full_correct == REPEATS and isolated_correct <= 1:
        return "context_effect"
    if full_varies or isolated_varies:
        return "mixed" if full_correct != isolated_correct else "decoding_variance"
    return "no_supported_cause"


def evaluate_diagnostic(
    cases: Sequence[dict[str, Any]], results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    case_by_id = {str(case["id"]): case for case in cases}
    runs: list[dict[str, Any]] = []
    for result in results:
        item = {
            key: result.get(key)
            for key in (
                "case_id", "condition", "repeat", "endpoint", "request_id",
                "elapsed_seconds", "wall_seconds", "finish_reason", "error",
            )
            if result.get(key) is not None
        }
        if not result.get("error"):
            validation = GroundingValidationResult.model_validate(result["validation"])
            predictions = _flatten_candidate(validation)
            gold = _filtered_gold(case_by_id[str(result["case_id"])], str(result["condition"]))
            item.update({
                "validation_valid": validation.valid,
                "issues": [issue.model_dump() for issue in validation.issues],
                "score": score_predictions(predictions, gold),
                "signature": _semantic_signature(predictions),
                "labels_by_gold_span": _labels_for_gold(predictions, gold),
                "predictions": predictions,
            })
        runs.append(item)
    runs.sort(key=lambda item: (
        str(item["case_id"]), str(item["condition"]), int(item["repeat"]),
    ))

    groups: dict[str, Any] = {}
    for case_id in CASE_IDS:
        groups[case_id] = {}
        for condition in CONDITIONS:
            members = [
                item for item in runs
                if item["case_id"] == case_id and item["condition"] == condition
            ]
            completed = [item for item in members if "score" in item]
            groups[case_id][condition] = {
                "completed": len(completed),
                "invalid": sum(not item["validation_valid"] for item in completed),
                "mean_exact_precision": statistics.fmean(
                    item["score"]["exact"]["precision"] for item in completed
                ) if completed else 0.0,
                "mean_exact_recall": statistics.fmean(
                    item["score"]["exact"]["recall"] for item in completed
                ) if completed else 0.0,
                "critical_unsupported_additions": sum(
                    item["score"]["counts"]["critical_unsupported_additions"]
                    for item in completed
                ),
                "unique_signature_count": len({
                    json.dumps(item["signature"], sort_keys=True) for item in completed
                }),
                "latencies": [item.get("wall_seconds") for item in completed],
            }

    observations = []
    probes = (
        (
            "product_target_role",
            "SGD-MULTI-PRODUCT-01",
            "query:4:8:产品经理",
            "target_role",
            "query_only",
        ),
        (
            "finance_project_1_type",
            "SGD-FINANCE-06",
            "resume:84:92:异常交易筛查项目",
            "deliverable",
            "resume_only",
        ),
        (
            "finance_project_2_type",
            "SGD-FINANCE-06",
            "resume:124:132:授信流程优化项目",
            "deliverable",
            "resume_only",
        ),
        (
            "finance_result_type",
            "SGD-FINANCE-06",
            "resume:109:123:识别37笔异常交易并提交复核",
            "result",
            "resume_only",
        ),
    )
    for name, case_id, span_key, expected_type, isolated_condition in probes:
        full = [
            item["labels_by_gold_span"].get(span_key, [])
            for item in runs
            if item["case_id"] == case_id
            and item["condition"] == "full"
            and "labels_by_gold_span" in item
        ]
        isolated = [
            item["labels_by_gold_span"].get(span_key, [])
            for item in runs
            if item["case_id"] == case_id
            and item["condition"] == isolated_condition
            and "labels_by_gold_span" in item
        ]
        observations.append({
            "name": name,
            "expected_type": expected_type,
            "full_labels": full,
            "isolated_labels": isolated,
            "cause": (
                classify_cause(full, isolated, expected_type=expected_type)
                if len(full) == REPEATS and len(isolated) == REPEATS
                else "no_supported_cause"
            ),
        })

    return {
        "schema_version": 1,
        "evidence_level": "same-set-diagnostic",
        "request_count": len(results),
        "completed_count": sum("score" in item for item in runs),
        "failure_count": sum("score" not in item for item in runs),
        "groups": groups,
        "observations": observations,
        "runs": runs,
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--model", default="Qwen3.5-27B-AWQ")
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    all_cases = _read_jsonl(args.cases)
    cases = [case for case in all_cases if str(case["id"]) in CASE_IDS]
    if {str(case["id"]) for case in cases} != set(CASE_IDS):
        raise SystemExit("predeclared diagnostic cases are missing")
    endpoints = args.endpoint or DEFAULT_ENDPOINTS
    if len(endpoints) != 4:
        raise SystemExit("frozen diagnostic requires exactly four endpoints")
    results = run_diagnostic(
        cases,
        endpoints=endpoints,
        model=args.model,
        checkpoint_path=args.raw_output,
    )
    report = evaluate_diagnostic(cases, results)
    report["model"] = args.model
    report["endpoints"] = endpoints
    _atomic_write_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "request_count": report["request_count"],
        "completed_count": report["completed_count"],
        "failure_count": report["failure_count"],
        "observations": report["observations"],
    }, ensure_ascii=False, indent=2))
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
