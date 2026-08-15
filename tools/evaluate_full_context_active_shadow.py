#!/usr/bin/env python3
"""Evaluate A4 full-context, active-output segment extraction offline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from full_context_active_extractor import (  # noqa: E402
    ActiveSegmentWindow,
    build_active_segment_windows,
    build_active_window_prompt,
    validate_active_extraction,
)
from llm_gateway import strip_thinking  # noqa: E402
from record_window_extractor import (  # noqa: E402
    SegmentWindow,
    WindowValidation,
    aggregate_window_validations,
)
from segment_grounded_extractor import (  # noqa: E402
    DocumentSegment,
    GroundedExtraction,
    GroundingValidationResult,
    build_document_segments,
)
from tools.evaluate_record_window_shadow import (  # noqa: E402
    CASE_TIMEOUT_SECONDS,
    DEFAULT_ENDPOINTS,
    HARD_TIMEOUT_SECONDS,
    _base_endpoint,
    _terminate_worker,
    _wait_until_idle,
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


MAX_ACTIVE_SEGMENTS = 20
MAX_TOKENS = 2_048


def _worker_payload(
    segments: Sequence[DocumentSegment],
    active_window: ActiveSegmentWindow,
    *,
    endpoint: str,
    model: str,
    max_tokens: int,
    socket_timeout: float,
) -> dict[str, Any]:
    return {
        "segments": [segment.model_dump() for segment in segments],
        "active_window": active_window.model_dump(),
        "endpoint": endpoint,
        "model": model,
        "max_tokens": max_tokens,
        "socket_timeout": socket_timeout,
        "request_id": f"fca-{uuid.uuid4().hex}",
    }


def _worker_call(payload: dict[str, Any]) -> dict[str, Any]:
    import httpx
    from openai import OpenAI

    segments = tuple(
        DocumentSegment.model_validate(item) for item in payload["segments"]
    )
    active_window = ActiveSegmentWindow.model_validate(payload["active_window"])
    system_prompt, user_prompt = build_active_window_prompt(
        segments, active_window.active_segment_ids,
    )
    http_client = httpx.Client(trust_env=False, timeout=float(payload["socket_timeout"]))
    client = OpenAI(
        base_url=_base_endpoint(str(payload["endpoint"])) + "/v1",
        api_key="EMPTY",
        timeout=float(payload["socket_timeout"]),
        max_retries=0,
        http_client=http_client,
    )
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=str(payload["model"]),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=int(payload["max_tokens"]),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "segment_grounded_extraction",
                    "strict": True,
                    "schema": GroundedExtraction.model_json_schema(),
                },
            },
            extra_headers={"x-request-id": str(payload["request_id"])},
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "top_p": 1.0,
            },
        )
        elapsed = time.perf_counter() - started
        content = strip_thinking(response.choices[0].message.content or "")
        extraction = GroundedExtraction.model_validate(json.loads(content))
        active_validation = validate_active_extraction(
            extraction, segments, active_window.active_segment_ids,
        )
        return {
            "request_id": payload["request_id"],
            "window_id": active_window.window_id,
            "active_segment_ids": list(active_window.active_segment_ids),
            "elapsed_seconds": elapsed,
            "finish_reason": response.choices[0].finish_reason,
            "raw_extraction": extraction.model_dump(),
            "validation": active_validation.validation.model_dump(),
            "active_issues": [
                issue.model_dump() for issue in active_validation.active_issues
            ],
        }
    finally:
        client.close()


def _worker_main() -> int:
    try:
        result = _worker_call(json.loads(sys.stdin.read()))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


def _spawn_worker(payload: dict[str, Any]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def execute_active_window(
    segments: Sequence[DocumentSegment],
    active_window: ActiveSegmentWindow,
    *,
    endpoint: str,
    model: str,
    hard_timeout: float,
    max_tokens: int,
) -> dict[str, Any]:
    payload = _worker_payload(
        segments,
        active_window,
        endpoint=endpoint,
        model=model,
        max_tokens=max_tokens,
        socket_timeout=hard_timeout + 30.0,
    )
    process = _spawn_worker(payload)
    started = time.perf_counter()
    try:
        stdout, stderr = process.communicate(
            json.dumps(payload, ensure_ascii=False), timeout=hard_timeout,
        )
    except subprocess.TimeoutExpired:
        signal_sent, stdout, stderr = _terminate_worker(process)
        idle, post_metrics = _wait_until_idle(endpoint)
        return {
            "window_id": active_window.window_id,
            "request_id": payload["request_id"],
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"HardTimeout: exceeded {hard_timeout:.3f}s",
            "timed_out": True,
            "termination_signal": signal_sent,
            "server_idle_after_disconnect": idle,
            "post_metrics": post_metrics,
            "worker_stdout": stdout[-2_000:],
            "worker_stderr": stderr[-2_000:],
        }

    wall_seconds = time.perf_counter() - started
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return {
            "window_id": active_window.window_id,
            "request_id": payload["request_id"],
            "elapsed_seconds": wall_seconds,
            "error": f"WorkerProtocolError: {exc}",
            "worker_stdout": stdout[-2_000:],
            "worker_stderr": stderr[-2_000:],
        }
    if process.returncode != 0 or result.get("error"):
        return {
            "window_id": active_window.window_id,
            "request_id": payload["request_id"],
            "elapsed_seconds": wall_seconds,
            "error": str(result.get("error") or f"worker exited {process.returncode}"),
            "worker_stderr": stderr[-2_000:],
        }
    result["wall_seconds"] = wall_seconds
    return result


def _aggregation_window(
    active_window: ActiveSegmentWindow,
    segment_index: dict[str, DocumentSegment],
) -> SegmentWindow:
    active_segments = tuple(
        segment_index[segment_id] for segment_id in active_window.active_segment_ids
    )
    return SegmentWindow(
        window_id=active_window.window_id,
        source_id="active",
        source_type=active_segments[0].source_type,
        segments=active_segments,
        character_count=sum(len(segment.text) for segment in active_segments),
    )


def _active_case(
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
    active_windows = build_active_segment_windows(
        segments, max_segments=MAX_ACTIVE_SEGMENTS,
    )
    segment_index = {segment.segment_id: segment for segment in segments}
    window_results: list[dict[str, Any]] = []
    validations: list[WindowValidation] = []
    aggregation_windows: list[SegmentWindow] = []
    for active_window in active_windows:
        remaining = case_timeout - (time.perf_counter() - started)
        if remaining <= 0:
            return {
                "case_id": case["id"],
                "endpoint": endpoint,
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"CaseTimeout: exceeded {case_timeout:.3f}s",
                "window_count": len(active_windows),
                "windows": window_results,
            }
        result = execute_active_window(
            segments,
            active_window,
            endpoint=endpoint,
            model=model,
            hard_timeout=min(hard_timeout, remaining),
            max_tokens=max_tokens,
        )
        window_results.append(result)
        if result.get("error"):
            return {
                "case_id": case["id"],
                "endpoint": endpoint,
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{active_window.window_id}: {result['error']}",
                "window_count": len(active_windows),
                "windows": window_results,
            }
        validations.append(WindowValidation(
            window_id=active_window.window_id,
            validation=GroundingValidationResult.model_validate(result["validation"]),
        ))
        aggregation_windows.append(_aggregation_window(active_window, segment_index))

    aggregate = aggregate_window_validations(
        aggregation_windows, validations, segments,
    )
    candidate_ids = [
        segment.segment_id for segment in segments if segment.source_type != "jd"
    ]
    active_ids = [
        segment_id
        for window in active_windows
        for segment_id in window.active_segment_ids
    ]
    elapsed = time.perf_counter() - started
    return {
        "case_id": case["id"],
        "endpoint": endpoint,
        "elapsed_seconds": elapsed,
        "finish_reason": (
            "stop" if all(item.get("finish_reason") == "stop" for item in window_results)
            else "mixed"
        ),
        "window_count": len(active_windows),
        "active_partition_valid": (
            active_ids == candidate_ids and len(active_ids) == len(set(active_ids))
        ),
        "active_issue_count": sum(
            len(item.get("active_issues") or []) for item in window_results
        ),
        "raw_extraction": {
            "windows": [
                {
                    "window_id": item["window_id"],
                    "request_id": item["request_id"],
                    "active_segment_ids": item["active_segment_ids"],
                    "finish_reason": item.get("finish_reason"),
                    "elapsed_seconds": item.get("elapsed_seconds"),
                    "wall_seconds": item.get("wall_seconds"),
                    "active_issues": item.get("active_issues"),
                    "raw_extraction": item.get("raw_extraction"),
                }
                for item in window_results
            ],
            "aggregation_issues": [
                issue.model_dump() for issue in aggregate.aggregation_issues
            ],
        },
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
        item = _active_case(
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


def run_active_shadow(
    cases: Sequence[dict[str, Any]],
    *,
    endpoints: Sequence[str],
    model: str,
    hard_timeout: float,
    case_timeout: float,
    max_tokens: int,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    if not endpoints:
        raise ValueError("at least one endpoint is required")
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
                    "runner": "full-context-active-a4",
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
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return _worker_main()

    args = _args()
    if args.max_tokens != MAX_TOKENS:
        raise SystemExit(f"frozen A4 requires --max-tokens={MAX_TOKENS}")
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

    raw = run_active_shadow(
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
        "runner": "full-context-active-a4",
        "model": args.model,
        "active_config": {
            "max_active_segments": MAX_ACTIVE_SEGMENTS,
            "max_tokens": args.max_tokens,
            "hard_timeout": args.hard_timeout,
            "case_timeout": args.case_timeout,
        },
        "active_scope": {
            "partition_invalid_cases": sum(
                not item.get("active_partition_valid", False)
                for item in raw if not item.get("error")
            ),
            "issue_count": sum(
                int(item.get("active_issue_count") or 0)
                for item in raw if not item.get("error")
            ),
        },
    })
    _atomic_write_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "case_count": report["case_count"],
        "completed_count": report.get("completed_count"),
        "active_scope": report["active_scope"],
        "shadow": report.get("shadow"),
        "failures": report.get("failures"),
    }, ensure_ascii=False, indent=2))
    return 1 if report.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
