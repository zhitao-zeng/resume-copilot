#!/usr/bin/env python3
"""Evaluate the output-neutral segment-grounded parser shadow.

The tool scores model references directly against hand-authored source spans.
It never calls the resume API and never feeds shadow output into production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from llm_gateway import strip_thinking  # noqa: E402
from segment_grounded_extractor import (  # noqa: E402
    GroundedExtraction,
    GroundingValidationResult,
    build_document_segments,
    build_shadow_prompt,
    validate_grounded_extraction,
)
from source_adapter import build_source_bundle  # noqa: E402
from v2_schemas import SourceDocument  # noqa: E402


DEFAULT_CASES = (
    ROOT / "validation_sets" / "segment_grounded_development" / "cases.jsonl"
)
CRITICAL_TYPES = {"organization", "role", "period", "education", "credential", "metric"}
PROFILE_TYPES = {"identity", "contact", "target_role"}
COMPARABLE_TYPES = {
    "identity",
    "contact",
    "organization",
    "role",
    "period",
    "education",
    "action",
    "method",
    "deliverable",
    "result",
    "skill",
    "credential",
    "metric",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _documents(case: dict[str, Any]) -> list[SourceDocument]:
    values = (
        ("resume", "resume", str(case.get("cv_text") or "")),
        ("query", "query", str(case.get("query_text") or "")),
        ("jd", "jd", str(case.get("jd_text") or "")),
    )
    return [
        SourceDocument(source_id=source_id, source_type=source_type, text=text)  # type: ignore[arg-type]
        for source_id, source_type, text in values
        if text
    ]


def _nth_span(text: str, quote: str, occurrence: int) -> tuple[int, int]:
    if not quote:
        raise ValueError("gold quote must be non-empty")
    cursor = 0
    start = -1
    for _ in range(occurrence):
        start = text.find(quote, cursor)
        if start < 0:
            raise ValueError(
                f"quote {quote!r} occurrence {occurrence} does not exist in source"
            )
        cursor = start + len(quote)
    return start, start + len(quote)


def resolve_gold(case: dict[str, Any]) -> list[dict[str, Any]]:
    texts = {document.source_id: document.text for document in _documents(case)}
    resolved: list[dict[str, Any]] = []
    for index, item in enumerate(case.get("expected_fields") or []):
        source_id = str(item["source_id"])
        if source_id == "jd":
            raise ValueError(f"{case['id']}: JD cannot be annotated as candidate fact")
        if source_id not in texts:
            raise ValueError(f"{case['id']}: missing source {source_id!r}")
        occurrence = int(item.get("occurrence") or 1)
        start, end = _nth_span(texts[source_id], str(item["quote"]), occurrence)
        resolved.append({
            "gold_id": f"gold-{index:03d}",
            "scope": str(item["scope"]),
            "record_id": item.get("record_id"),
            "record_type": item.get("record_type"),
            "field_type": str(item["field_type"]),
            "source_id": source_id,
            "start": start,
            "end": end,
            "quote": str(item["quote"]),
        })
    return resolved


def _flatten_candidate(result: GroundingValidationResult) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for index, field in enumerate(result.profile_fields):
        flattened.append({
            "prediction_id": f"profile-{index:03d}",
            "scope": "profile",
            "record_id": None,
            "record_type": None,
            "field_type": field.field_type,
            "parts": [
                {
                    "source_id": span.source_id,
                    "start": span.absolute_start,
                    "end": span.absolute_end,
                    "text": span.text,
                }
                for span in field.spans
            ],
            "value": field.value,
        })
    for record_index, record in enumerate(result.records):
        for field_index, field in enumerate(record.fields):
            flattened.append({
                "prediction_id": f"record-{record_index:03d}-field-{field_index:03d}",
                "scope": "record",
                "record_id": f"pred-record-{record_index:03d}",
                "record_type": record.record_type,
                "field_type": field.field_type,
                "parts": [
                    {
                        "source_id": span.source_id,
                        "start": span.absolute_start,
                        "end": span.absolute_end,
                        "text": span.text,
                    }
                    for span in field.spans
                ],
                "value": field.value,
            })
    return flattened


def _flatten_baseline(case: dict[str, Any]) -> list[dict[str, Any]]:
    bundle = build_source_bundle(
        str(case.get("cv_text") or ""),
        str(case.get("query_text") or ""),
        str(case.get("jd_text") or ""),
    )
    flattened: list[dict[str, Any]] = []
    for fact_index, fact in enumerate(bundle.fact_units):
        if not fact.fact_eligible or fact.source_type == "jd":
            continue
        for field_type in fact.dimensions:
            if field_type not in COMPARABLE_TYPES:
                continue
            scope = "profile" if field_type in {"identity", "contact"} else "record"
            if scope == "profile":
                record_id = None
                record_type = None
            else:
                record_id = fact.record_id or f"unassigned-{fact.fact_id}"
                section = str(fact.section_hint or "")
                record_type = {
                    "projects": "project",
                    "education": "education",
                    "activities": "campus",
                    "research": "project",
                    "experience": "work",
                }.get(section, "other")
            parts = [
                {
                    "source_id": span.source_id,
                    "start": span.char_start,
                    "end": span.char_end,
                    "text": next(
                        document.text[span.char_start:span.char_end]
                        for document in bundle.documents
                        if document.source_id == span.source_id
                    ),
                }
                for span in fact.source_spans
            ]
            flattened.append({
                "prediction_id": f"baseline-{fact_index:03d}-{field_type}",
                "scope": scope,
                "record_id": record_id,
                "record_type": record_type,
                "field_type": field_type,
                "parts": parts,
                "value": fact.verbatim_text,
            })
    return flattened


def _exact_match(prediction: dict[str, Any], gold: dict[str, Any]) -> bool:
    parts = prediction["parts"]
    return bool(
        prediction["field_type"] == gold["field_type"]
        and prediction["scope"] == gold["scope"]
        and len(parts) == 1
        and parts[0]["source_id"] == gold["source_id"]
        and parts[0]["start"] == gold["start"]
        and parts[0]["end"] == gold["end"]
    )


def _overlap_score(prediction: dict[str, Any], gold: dict[str, Any]) -> float:
    if (
        prediction["field_type"] != gold["field_type"]
        or prediction["scope"] != gold["scope"]
    ):
        return 0.0
    intersection = 0
    predicted_length = 0
    for part in prediction["parts"]:
        predicted_length += max(0, int(part["end"]) - int(part["start"]))
        if part["source_id"] != gold["source_id"]:
            continue
        intersection += max(
            0,
            min(int(part["end"]), int(gold["end"]))
            - max(int(part["start"]), int(gold["start"])),
        )
    denominator = max(predicted_length, int(gold["end"]) - int(gold["start"]), 1)
    return intersection / denominator


def _match_fields(
    predictions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    *,
    exact: bool,
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        for gold_index, gold_item in enumerate(gold):
            score = 1.0 if _exact_match(prediction, gold_item) else _overlap_score(
                prediction, gold_item,
            )
            threshold = 1.0 if exact else 0.5
            if score >= threshold:
                candidates.append((score, prediction_index, gold_index))
    candidates.sort(reverse=True)
    used_predictions: set[int] = set()
    used_gold: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, prediction_index, gold_index in candidates:
        if prediction_index in used_predictions or gold_index in used_gold:
            continue
        used_predictions.add(prediction_index)
        used_gold.add(gold_index)
        matches.append((prediction_index, gold_index, score))
    return matches


def _ratio(numerator: int, denominator: int, *, empty: float = 1.0) -> float:
    return numerator / denominator if denominator else empty


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score_predictions(
    predictions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> dict[str, Any]:
    exact_matches = _match_fields(predictions, gold, exact=True)
    overlap_matches = _match_fields(predictions, gold, exact=False)
    exact_precision = _ratio(len(exact_matches), len(predictions))
    exact_recall = _ratio(len(exact_matches), len(gold))
    overlap_precision = _ratio(len(overlap_matches), len(predictions))
    overlap_recall = _ratio(len(overlap_matches), len(gold))
    matched_predictions = {prediction_index for prediction_index, _, _ in overlap_matches}
    critical_unsupported = [
        prediction
        for index, prediction in enumerate(predictions)
        if index not in matched_predictions and prediction["field_type"] in CRITICAL_TYPES
    ]

    association_pairs = 0
    association_gold_positive = 0
    association_predicted_positive = 0
    association_true_positive = 0
    record_matches = [
        (predictions[prediction_index], gold[gold_index])
        for prediction_index, gold_index, _ in overlap_matches
        if predictions[prediction_index]["scope"] == "record"
    ]
    for left in range(len(record_matches)):
        for right in range(left + 1, len(record_matches)):
            left_prediction, left_gold = record_matches[left]
            right_prediction, right_gold = record_matches[right]
            gold_same = left_gold["record_id"] == right_gold["record_id"]
            predicted_same = (
                left_prediction["record_id"] == right_prediction["record_id"]
            )
            association_pairs += 1
            association_gold_positive += int(gold_same)
            association_predicted_positive += int(predicted_same)
            association_true_positive += int(gold_same and predicted_same)
    ownership_precision = _ratio(
        association_true_positive,
        association_predicted_positive,
        empty=1.0 if association_gold_positive == 0 else 0.0,
    )
    ownership_recall = _ratio(
        association_true_positive,
        association_gold_positive,
        empty=1.0,
    )

    return {
        "counts": {
            "predictions": len(predictions),
            "gold": len(gold),
            "exact_matches": len(exact_matches),
            "overlap_matches": len(overlap_matches),
            "critical_unsupported_additions": len(critical_unsupported),
            "ownership_pairs": association_pairs,
            "ownership_gold_positive": association_gold_positive,
            "ownership_predicted_positive": association_predicted_positive,
            "ownership_true_positive": association_true_positive,
        },
        "exact": {
            "precision": exact_precision,
            "recall": exact_recall,
            "f1": _f1(exact_precision, exact_recall),
        },
        "overlap": {
            "precision": overlap_precision,
            "recall": overlap_recall,
            "f1": _f1(overlap_precision, overlap_recall),
        },
        "ownership": {
            "precision": ownership_precision,
            "recall": ownership_recall,
            "f1": _f1(ownership_precision, ownership_recall),
            "matched_record_field_coverage": _ratio(
                len(record_matches),
                sum(1 for item in gold if item["scope"] == "record"),
            ),
        },
        "critical_unsupported": critical_unsupported,
        "overlap_matches": [
            {
                "prediction_id": predictions[prediction_index]["prediction_id"],
                "gold_id": gold[gold_index]["gold_id"],
                "score": score,
            }
            for prediction_index, gold_index, score in overlap_matches
        ],
    }


def _semantic_signature(items: list[dict[str, Any]]) -> dict[str, Any]:
    profile = Counter(
        item["field_type"] for item in items if item["scope"] == "profile"
    )
    records: dict[str, dict[str, Any]] = {}
    for item in items:
        if item["scope"] != "record":
            continue
        record_id = str(item["record_id"])
        record = records.setdefault(record_id, {
            "record_type": item.get("record_type") or "other",
            "fields": Counter(),
        })
        record["fields"][item["field_type"]] += 1
    return {
        "profile": sorted(profile.items()),
        "records": sorted(
            (
                value["record_type"],
                tuple(sorted(value["fields"].items())),
            )
            for value in records.values()
        ),
    }


def _aggregate_case_scores(case_results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    count_keys = (
        "predictions",
        "gold",
        "exact_matches",
        "overlap_matches",
        "critical_unsupported_additions",
        "ownership_pairs",
        "ownership_gold_positive",
        "ownership_predicted_positive",
        "ownership_true_positive",
    )
    counts = {
        name: sum(int(case[key]["counts"][name]) for case in case_results)
        for name in count_keys
    }
    exact_precision = _ratio(counts["exact_matches"], counts["predictions"])
    exact_recall = _ratio(counts["exact_matches"], counts["gold"])
    overlap_precision = _ratio(counts["overlap_matches"], counts["predictions"])
    overlap_recall = _ratio(counts["overlap_matches"], counts["gold"])
    ownership_precision = _ratio(
        counts["ownership_true_positive"],
        counts["ownership_predicted_positive"],
        empty=1.0 if counts["ownership_gold_positive"] == 0 else 0.0,
    )
    ownership_recall = _ratio(
        counts["ownership_true_positive"],
        counts["ownership_gold_positive"],
    )
    return {
        "counts": counts,
        "exact": {
            "precision": exact_precision,
            "recall": exact_recall,
            "f1": _f1(exact_precision, exact_recall),
        },
        "overlap": {
            "precision": overlap_precision,
            "recall": overlap_recall,
            "f1": _f1(overlap_precision, overlap_recall),
        },
        "ownership": {
            "precision": ownership_precision,
            "recall": ownership_recall,
            "f1": _f1(ownership_precision, ownership_recall),
            "macro_f1": statistics.fmean(
                case[key]["ownership"]["f1"] for case in case_results
            ) if case_results else 0.0,
        },
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[rank]


def _call_model(
    *,
    endpoint: str,
    model: str,
    case: dict[str, Any],
    timeout: float,
    max_tokens: int,
) -> dict[str, Any]:
    from openai import OpenAI

    documents = _documents(case)
    segments = build_document_segments(documents)
    system_prompt, user_prompt = build_shadow_prompt(segments)
    client = OpenAI(
        base_url=endpoint.rstrip("/") + ("" if endpoint.rstrip("/").endswith("/v1") else "/v1"),
        api_key="EMPTY",
        timeout=timeout,
        # A benchmark timeout must describe one attempt. SDK retries can make
        # the observed wall time a multiple of the configured limit and leave
        # an accelerator request running after the evaluator has given up.
        max_retries=0,
    )
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "segment_grounded_extraction",
                "strict": True,
                "schema": GroundedExtraction.model_json_schema(),
            },
        },
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "top_p": 1.0,
        },
    )
    elapsed = time.perf_counter() - started
    content = strip_thinking(response.choices[0].message.content or "")
    raw = json.loads(content)
    extraction = GroundedExtraction.model_validate(raw)
    validation = validate_grounded_extraction(extraction, segments)
    return {
        "case_id": case["id"],
        "endpoint": endpoint,
        "elapsed_seconds": elapsed,
        "finish_reason": response.choices[0].finish_reason,
        "raw_extraction": extraction.model_dump(),
        "validation": validation.model_dump(),
    }


def _run_endpoint_cases(
    endpoint: str,
    cases: list[dict[str, Any]],
    *,
    model: str,
    timeout: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            results.append(_call_model(
                endpoint=endpoint,
                model=model,
                case=case,
                timeout=timeout,
                max_tokens=max_tokens,
            ))
        except Exception as exc:
            results.append({
                "case_id": case["id"],
                "endpoint": endpoint,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return results


def run_shadow(
    cases: list[dict[str, Any]],
    *,
    endpoints: list[str],
    model: str,
    timeout: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    assignments = [[] for _ in endpoints]
    for index, case in enumerate(cases):
        assignments[index % len(endpoints)].append(case)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
        futures = [
            executor.submit(
                _run_endpoint_cases,
                endpoint,
                assigned,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            for endpoint, assigned in zip(endpoints, assignments)
            if assigned
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    return sorted(results, key=lambda item: str(item["case_id"]))


def evaluate(
    cases: list[dict[str, Any]],
    shadow_results: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    shadow_by_id = {
        str(item["case_id"]): item for item in (shadow_results or [])
    }
    case_results: list[dict[str, Any]] = []
    for case in cases:
        gold = resolve_gold(case)
        baseline_predictions = _flatten_baseline(case)
        item: dict[str, Any] = {
            "case_id": case["id"],
            "group": case.get("group"),
            "metamorphic_family": case.get("metamorphic_family"),
            "variant": case.get("variant"),
            "gold_signature": _semantic_signature(gold),
            "baseline": score_predictions(baseline_predictions, gold),
            "baseline_signature": _semantic_signature(baseline_predictions),
        }
        shadow = shadow_by_id.get(str(case["id"]))
        if shadow is not None:
            item["runtime"] = {
                key: shadow.get(key)
                for key in ("endpoint", "elapsed_seconds", "finish_reason", "error")
                if shadow.get(key) is not None
            }
            if not shadow.get("error"):
                validation = GroundingValidationResult.model_validate(shadow["validation"])
                predictions = _flatten_candidate(validation)
                item["shadow"] = score_predictions(predictions, gold)
                item["shadow_signature"] = _semantic_signature(predictions)
                item["localization"] = {
                    "valid": validation.valid,
                    "returned_references": validation.returned_reference_count,
                    "valid_references": validation.valid_reference_count,
                    "validity_rate": _ratio(
                        validation.valid_reference_count,
                        validation.returned_reference_count,
                    ),
                    "issues": [issue.model_dump() for issue in validation.issues],
                }
                item["raw_extraction"] = shadow["raw_extraction"]
        case_results.append(item)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset_sha256": hashlib.sha256(
            DEFAULT_CASES.read_bytes() if DEFAULT_CASES.is_file() else b""
        ).hexdigest(),
        "case_count": len(cases),
        "baseline": _aggregate_case_scores(case_results, "baseline"),
        "cases": case_results,
    }
    shadow_cases = [item for item in case_results if "shadow" in item]
    if shadow_cases:
        payload["shadow"] = _aggregate_case_scores(shadow_cases, "shadow")
        localization_returned = sum(
            item["localization"]["returned_references"] for item in shadow_cases
        )
        localization_valid = sum(
            item["localization"]["valid_references"] for item in shadow_cases
        )
        latencies = [
            float(item["runtime"]["elapsed_seconds"])
            for item in shadow_cases
            if item.get("runtime", {}).get("elapsed_seconds") is not None
        ]
        payload["shadow"]["localization"] = {
            "returned_references": localization_returned,
            "valid_references": localization_valid,
            "validity_rate": _ratio(localization_valid, localization_returned),
            "invalid_case_count": sum(
                not item["localization"]["valid"] for item in shadow_cases
            ),
        }
        payload["shadow"]["latency"] = {
            "mean_seconds": statistics.fmean(latencies) if latencies else 0.0,
            "p95_seconds": _percentile(latencies, 0.95),
            "max_seconds": max(latencies, default=0.0),
        }

        families: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in shadow_cases:
            if item.get("metamorphic_family"):
                families[str(item["metamorphic_family"])].append(item)
        payload["shadow"]["metamorphic"] = {}
        for family, members in sorted(families.items()):
            gold_signatures = {
                json.dumps(member["gold_signature"], sort_keys=True)
                for member in members
            }
            shadow_signatures = {
                json.dumps(member["shadow_signature"], sort_keys=True)
                for member in members
            }
            payload["shadow"]["metamorphic"][family] = {
                "case_ids": [member["case_id"] for member in members],
                "gold_is_invariant": len(gold_signatures) == 1,
                "shadow_is_invariant": len(shadow_signatures) == 1,
                "all_match_gold_signature": all(
                    member["shadow_signature"] == member["gold_signature"]
                    for member in members
                ),
            }
    payload["failures"] = [
        item
        for item in (shadow_results or [])
        if item.get("error")
    ]
    return payload


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--model", default="Qwen3.5-27B-AWQ")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--evaluate-raw", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    cases = _read_jsonl(args.cases)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if str(case["id"]) in selected]
    if not cases:
        raise SystemExit("no cases selected")

    # Resolve every annotation before spending model time.
    for case in cases:
        gold = resolve_gold(case)
        segments = build_document_segments(_documents(case))
        for item in gold:
            if not any(
                segment.source_id == item["source_id"]
                and segment.char_start <= item["start"]
                and segment.char_end >= item["end"]
                for segment in segments
            ):
                raise ValueError(
                    f"{case['id']}: gold span {item['quote']!r} crosses segment boundary"
                )

    raw_results: list[dict[str, Any]] | None = None
    if args.evaluate_raw:
        loaded = json.loads(args.evaluate_raw.read_text(encoding="utf-8"))
        raw_results = list(loaded.get("results") or loaded)
    elif not args.baseline_only:
        endpoints = args.endpoint or ["http://127.0.0.1:8007"]
        raw_results = run_shadow(
            cases,
            endpoints=endpoints,
            model=args.model,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
        if args.raw_output:
            _atomic_write_json(args.raw_output, {
                "schema_version": 1,
                "model": args.model,
                "endpoints": endpoints,
                "results": raw_results,
            })

    report = evaluate(cases, raw_results)
    report["dataset_path"] = str(args.cases.resolve())
    report["model"] = None if args.baseline_only else args.model
    _atomic_write_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "case_count": report["case_count"],
        "baseline": report["baseline"],
        "shadow": report.get("shadow"),
        "failures": report.get("failures"),
    }, ensure_ascii=False, indent=2))
    return 1 if report.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
