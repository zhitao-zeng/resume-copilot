#!/usr/bin/env python3
"""Evaluate frozen-A3 cross-source context windows without production changes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from cross_source_context_extractor import (  # noqa: E402
    CrossSourceWindow,
    build_cross_source_windows,
)
from record_window_extractor import (  # noqa: E402
    SegmentWindow,
    WindowValidation,
    aggregate_window_validations,
)
from segment_grounded_extractor import (  # noqa: E402
    GroundingValidationResult,
    build_document_segments,
)
from tools.evaluate_record_window_shadow import (  # noqa: E402
    CASE_TIMEOUT_SECONDS,
    DEFAULT_ENDPOINTS,
    HARD_TIMEOUT_SECONDS,
    _wait_until_idle,
    execute_window_hard_timeout,
    predeclared_shadow_probe_cases,
)
from tools.evaluate_segment_grounded_shadow import (  # noqa: E402
    DEFAULT_CASES,
    _atomic_write_json,
    _documents,
    _read_jsonl,
    evaluate,
    resolve_gold,
)
from tools.evaluate_segment_grounded_shadowv3 import (  # noqa: E402
    evaluate_blind,
    load_blind_cases,
)


MAX_SEGMENTS = 20
MAX_CHARACTERS = 1_600
OVERLAP_SEGMENTS = 5
MAX_TOKENS = 3_072


def _transport_window(window: CrossSourceWindow) -> SegmentWindow:
    return SegmentWindow(
        window_id=window.window_id,
        source_id="candidate",
        source_type=window.segments[0].source_type,
        segments=window.segments,
        character_count=window.character_count,
    )


def _context_case(
    case: dict[str, Any],
    *,
    endpoint: str,
    model: str,
    hard_timeout: float,
    case_timeout: float,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    segments = build_document_segments(_documents(case))
    context_windows = build_cross_source_windows(
        segments,
        max_segments=MAX_SEGMENTS,
        max_characters=MAX_CHARACTERS,
        overlap_segments=OVERLAP_SEGMENTS,
    )
    transports = [_transport_window(window) for window in context_windows]
    results: list[dict[str, Any]] = []
    validations: list[WindowValidation] = []
    for context_window, transport in zip(context_windows, transports):
        remaining = case_timeout - (time.perf_counter() - started)
        if remaining <= 0:
            return {
                "case_id": case["id"],
                "endpoint": endpoint,
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"CaseTimeout: exceeded {case_timeout:.3f}s",
                "window_count": len(context_windows),
                "windows": results,
            }
        item = execute_window_hard_timeout(
            transport,
            endpoint=endpoint,
            model=model,
            hard_timeout=min(hard_timeout, remaining),
            max_tokens=max_tokens,
        )
        item["mode"] = context_window.mode
        results.append(item)
        if item.get("error"):
            return {
                "case_id": case["id"],
                "endpoint": endpoint,
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{context_window.window_id}: {item['error']}",
                "window_count": len(context_windows),
                "windows": results,
            }
        validations.append(WindowValidation(
            window_id=context_window.window_id,
            validation=GroundingValidationResult.model_validate(item["validation"]),
        ))

    aggregate = aggregate_window_validations(transports, validations, segments)
    elapsed = time.perf_counter() - started
    mode_counts = Counter(window.mode for window in context_windows)
    return {
        "case_id": case["id"],
        "endpoint": endpoint,
        "elapsed_seconds": elapsed,
        "finish_reason": (
            "stop" if all(item.get("finish_reason") == "stop" for item in results)
            else "mixed"
        ),
        "window_count": len(context_windows),
        "mode_counts": dict(mode_counts),
        "cross_source_context_applied": bool(mode_counts.get("resume_with_query")),
        "independent_fallback": bool(mode_counts.get("independent_fallback")),
        "raw_extraction": {
            "windows": [
                {
                    "window_id": item["window_id"],
                    "request_id": item["request_id"],
                    "mode": item["mode"],
                    "finish_reason": item.get("finish_reason"),
                    "elapsed_seconds": item.get("elapsed_seconds"),
                    "wall_seconds": item.get("wall_seconds"),
                    "raw_extraction": item.get("raw_extraction"),
                }
                for item in results
            ],
            "aggregation_issues": [
                issue.model_dump() for issue in aggregate.aggregation_issues
            ],
        },
        "aggregation_issue_count": len(aggregate.aggregation_issues),
        "validation": aggregate.validation.model_dump(),
    }


def _run_endpoint_cases(
    endpoint: str,
    cases: Sequence[dict[str, Any]],
    *,
    model: str,
    hard_timeout: float,
    case_timeout: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    idle, metrics = _wait_until_idle(endpoint, timeout=3.0)
    if not idle:
        return [{
            "case_id": case["id"],
            "endpoint": endpoint,
            "error": f"EndpointBusy: {metrics}",
        } for case in cases]
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        item = _context_case(
            case,
            endpoint=endpoint,
            model=model,
            hard_timeout=hard_timeout,
            case_timeout=case_timeout,
            max_tokens=max_tokens,
        )
        results.append(item)
        if item.get("error") and any(
            window.get("timed_out") and not window.get("server_idle_after_disconnect")
            for window in item.get("windows") or []
        ):
            results.extend({
                "case_id": remaining["id"],
                "endpoint": endpoint,
                "error": "EndpointUnsafe: previous timeout did not cancel server request",
            } for remaining in cases[index + 1:])
            break
    return results


def run_context_shadow(
    cases: Sequence[dict[str, Any]],
    *,
    endpoints: Sequence[str],
    model: str,
    hard_timeout: float,
    case_timeout: float,
    max_tokens: int,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    assignments: list[list[dict[str, Any]]] = [[] for _ in endpoints]
    for index, case in enumerate(cases):
        assignments[index % len(endpoints)].append(case)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
        futures = [
            executor.submit(
                _run_endpoint_cases,
                endpoint,
                assignment,
                model=model,
                hard_timeout=hard_timeout,
                case_timeout=case_timeout,
                max_tokens=max_tokens,
            )
            for endpoint, assignment in zip(endpoints, assignments)
            if assignment
        ]
        for future in as_completed(futures):
            results.extend(future.result())
            results.sort(key=lambda item: str(item["case_id"]))
            if checkpoint_path:
                _atomic_write_json(checkpoint_path, {
                    "schema_version": 1,
                    "runner": "cross-source-context-c1",
                    "status": "running" if len(results) < len(cases) else "complete",
                    "model": model,
                    "endpoints": list(endpoints),
                    "completed_cases": len(results),
                    "expected_cases": len(cases),
                    "results": results,
                })
    return results


def _validate_dev(cases: Sequence[dict[str, Any]]) -> None:
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


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("development", "shadow-v3"), default="development")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--model", default="Qwen3.5-27B-AWQ")
    parser.add_argument("--hard-timeout", type=float, default=HARD_TIMEOUT_SECONDS)
    parser.add_argument("--case-timeout", type=float, default=CASE_TIMEOUT_SECONDS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--predeclared-shadow-probe", action="store_true")
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.max_tokens != MAX_TOKENS:
        raise SystemExit(f"frozen C1 requires --max-tokens={MAX_TOKENS}")
    endpoints = args.endpoint or DEFAULT_ENDPOINTS
    if args.dataset == "development":
        cases = _read_jsonl(args.cases)
        _validate_dev(cases)
    else:
        cases = load_blind_cases()
        if args.predeclared_shadow_probe:
            cases = predeclared_shadow_probe_cases(cases)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if str(case["id"]) in selected]
    if not cases:
        raise SystemExit("no cases selected")

    raw = run_context_shadow(
        cases,
        endpoints=endpoints,
        model=args.model,
        hard_timeout=args.hard_timeout,
        case_timeout=args.case_timeout,
        max_tokens=args.max_tokens,
        checkpoint_path=args.raw_output,
    )
    report = evaluate(cases, raw) if args.dataset == "development" else evaluate_blind(cases, raw)
    report.update({
        "runner": "cross-source-context-c1",
        "model": args.model,
        "context_config": {
            "max_segments": MAX_SEGMENTS,
            "max_characters": MAX_CHARACTERS,
            "overlap_segments": OVERLAP_SEGMENTS,
            "max_tokens": args.max_tokens,
            "hard_timeout": args.hard_timeout,
            "case_timeout": args.case_timeout,
        },
        "context_topology": {
            "cross_source_cases": sum(
                bool(item.get("cross_source_context_applied"))
                for item in raw if not item.get("error")
            ),
            "independent_fallback_cases": sum(
                bool(item.get("independent_fallback"))
                for item in raw if not item.get("error")
            ),
            "aggregation_issue_count": sum(
                int(item.get("aggregation_issue_count") or 0)
                for item in raw if not item.get("error")
            ),
        },
    })
    _atomic_write_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "case_count": report["case_count"],
        "completed_count": report.get("completed_count"),
        "context_topology": report["context_topology"],
        "shadow": report.get("shadow"),
        "failures": report.get("failures"),
    }, ensure_ascii=False, indent=2))
    return 1 if report.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
