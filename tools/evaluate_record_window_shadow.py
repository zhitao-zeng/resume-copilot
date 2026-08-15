#!/usr/bin/env python3
"""Evaluate frozen A3 through bounded windows with hard-cancellable requests.

The runner is output-neutral: it never calls or feeds results into the resume
API.  Each model request lives in a child process; the parent owns the hard
wall-clock deadline and closes the HTTP connection by terminating that process.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from llm_gateway import strip_thinking  # noqa: E402
from record_window_extractor import (  # noqa: E402
    SegmentWindow,
    WindowValidation,
    aggregate_window_validations,
    build_segment_windows,
)
from segment_grounded_extractor import (  # noqa: E402
    DocumentSegment,
    GroundedExtraction,
    GroundingValidationResult,
    build_document_segments,
    build_shadow_prompt,
    validate_grounded_extraction,
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


DEFAULT_ENDPOINTS = [f"http://127.0.0.1:{port}" for port in range(8007, 8011)]
MAX_SEGMENTS = 20
MAX_CHARACTERS = 1_600
OVERLAP_SEGMENTS = 5
MAX_TOKENS = 2_048
HARD_TIMEOUT_SECONDS = 120.0
CASE_TIMEOUT_SECONDS = 480.0


def _base_endpoint(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


def _endpoint_metrics(endpoint: str, *, timeout: float = 3.0) -> dict[str, float]:
    """Read the three lifecycle metrics used by the cancellation gate."""

    opener = build_opener(ProxyHandler({}))
    with opener.open(f"{_base_endpoint(endpoint)}/metrics", timeout=timeout) as response:
        text = response.read().decode("utf-8", errors="replace")
    result = {
        "running": 0.0,
        "waiting": 0.0,
        "abort_total": 0.0,
        "completed_total": 0.0,
    }
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        if line.startswith("vllm:num_requests_running{"):
            result["running"] += value
        elif line.startswith("vllm:num_requests_waiting{"):
            result["waiting"] += value
        elif (
            line.startswith("vllm:request_success_total{")
            and 'finished_reason="abort"' in line
        ):
            result["abort_total"] += value
        elif line.startswith("vllm:request_success_total{"):
            result["completed_total"] += value
    return result


def _wait_until_idle(
    endpoint: str,
    *,
    timeout: float = 20.0,
    poll_interval: float = 0.2,
) -> tuple[bool, dict[str, float]]:
    deadline = time.monotonic() + timeout
    latest = _endpoint_metrics(endpoint)
    while latest["running"] or latest["waiting"]:
        if time.monotonic() >= deadline:
            return False, latest
        time.sleep(poll_interval)
        latest = _endpoint_metrics(endpoint)
    return True, latest


def _worker_payload(
    window: SegmentWindow,
    *,
    endpoint: str,
    model: str,
    max_tokens: int,
    socket_timeout: float,
) -> dict[str, Any]:
    return {
        "endpoint": endpoint,
        "model": model,
        "max_tokens": max_tokens,
        "socket_timeout": socket_timeout,
        "request_id": f"rws-{uuid.uuid4().hex}",
        "window": window.model_dump(),
    }


def _worker_call(payload: dict[str, Any]) -> dict[str, Any]:
    """Perform one request inside a disposable child process."""

    import httpx
    from openai import OpenAI

    window = SegmentWindow.model_validate(payload["window"])
    system_prompt, user_prompt = build_shadow_prompt(window.segments)
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
        validation = validate_grounded_extraction(extraction, window.segments)
        return {
            "request_id": payload["request_id"],
            "window_id": window.window_id,
            "elapsed_seconds": elapsed,
            "finish_reason": response.choices[0].finish_reason,
            "raw_extraction": extraction.model_dump(),
            "validation": validation.model_dump(),
        }
    finally:
        client.close()


def _worker_main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        result = _worker_call(payload)
    except Exception as exc:  # Child must return a bounded, machine-readable error.
        print(json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False))
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


def _terminate_worker(process: subprocess.Popen[str]) -> tuple[str, str, str]:
    signal_sent = "none"
    if process.poll() is None:
        signal_sent = "SIGTERM"
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            signal_sent = "SIGKILL"
            os.killpg(process.pid, signal.SIGKILL)
    stdout, stderr = process.communicate()
    return signal_sent, stdout, stderr


def execute_window_hard_timeout(
    window: SegmentWindow,
    *,
    endpoint: str,
    model: str,
    hard_timeout: float = HARD_TIMEOUT_SECONDS,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    """Execute one window and enforce a parent-owned wall-clock deadline."""

    if hard_timeout <= 0:
        raise ValueError("hard_timeout must be positive")
    payload = _worker_payload(
        window,
        endpoint=endpoint,
        model=model,
        max_tokens=max_tokens,
        socket_timeout=hard_timeout + 30.0,
    )
    process = _spawn_worker(payload)
    started = time.perf_counter()
    try:
        stdout, stderr = process.communicate(
            json.dumps(payload, ensure_ascii=False),
            timeout=hard_timeout,
        )
    except subprocess.TimeoutExpired:
        signal_sent, stdout, stderr = _terminate_worker(process)
        idle, post_metrics = _wait_until_idle(endpoint)
        return {
            "window_id": window.window_id,
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

    elapsed = time.perf_counter() - started
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return {
            "window_id": window.window_id,
            "request_id": payload["request_id"],
            "elapsed_seconds": elapsed,
            "error": f"WorkerProtocolError: {exc}",
            "worker_stdout": stdout[-2_000:],
            "worker_stderr": stderr[-2_000:],
        }
    if process.returncode != 0 or result.get("error"):
        return {
            "window_id": window.window_id,
            "request_id": payload["request_id"],
            "elapsed_seconds": elapsed,
            "error": str(result.get("error") or f"worker exited {process.returncode}"),
            "worker_stderr": stderr[-2_000:],
        }
    result["wall_seconds"] = elapsed
    return result


def probe_server_cancellation(
    window: SegmentWindow,
    *,
    endpoint: str,
    model: str,
    max_tokens: int = MAX_TOKENS,
    observe_timeout: float = 15.0,
) -> dict[str, Any]:
    """Force a disconnect only after vLLM has visibly accepted the request."""

    idle, before = _wait_until_idle(endpoint, timeout=3.0)
    if not idle:
        return {"passed": False, "error": "endpoint was not idle", "before": before}
    payload = _worker_payload(
        window,
        endpoint=endpoint,
        model=model,
        max_tokens=max_tokens,
        socket_timeout=observe_timeout + 30.0,
    )
    process = _spawn_worker(payload)
    assert process.stdin is not None
    process.stdin.write(json.dumps(payload, ensure_ascii=False))
    process.stdin.close()
    process.stdin = None

    observed: dict[str, float] | None = None
    deadline = time.monotonic() + observe_timeout
    while time.monotonic() < deadline and process.poll() is None:
        current = _endpoint_metrics(endpoint)
        # A queued request can disappear without incrementing vLLM's abort
        # counter.  The cancellation gate specifically needs proof that an
        # executing generation is aborted, so waiting alone is insufficient.
        if current["running"] > before["running"]:
            observed = current
            break
        time.sleep(0.1)
    if observed is None:
        signal_sent, stdout, stderr = _terminate_worker(process)
        idle_after, after = _wait_until_idle(endpoint)
        return {
            "passed": False,
            "error": "request was never observed running before worker exit/deadline",
            "request_id": payload["request_id"],
            "before": before,
            "after": after,
            "server_idle_after_disconnect": idle_after,
            "termination_signal": signal_sent,
            "worker_stdout": stdout[-2_000:],
            "worker_stderr": stderr[-2_000:],
        }

    signal_sent, stdout, stderr = _terminate_worker(process)
    idle_after, after = _wait_until_idle(endpoint)
    abort_delta = after["abort_total"] - before["abort_total"]
    completed_delta = after["completed_total"] - before["completed_total"]
    return {
        # vLLM 0.19.1 exposes an abort-labelled counter but does not increment
        # it for this client-disconnect path.  The observable cancellation proof
        # is: executing request seen, worker killed before a response, endpoint
        # returns idle, and no normal completion counter advances.
        "passed": bool(idle_after and completed_delta == 0.0 and not stdout),
        "request_id": payload["request_id"],
        "window_id": window.window_id,
        "before": before,
        "observed": observed,
        "after": after,
        "abort_delta": abort_delta,
        "completed_delta": completed_delta,
        "abort_counter_incremented": abort_delta >= 1.0,
        "server_idle_after_disconnect": idle_after,
        "termination_signal": signal_sent,
        "worker_stdout": stdout[-2_000:],
        "worker_stderr": stderr[-2_000:],
    }


def _windowed_case(
    case: dict[str, Any],
    *,
    endpoint: str,
    model: str,
    hard_timeout: float,
    case_timeout: float,
    max_tokens: int,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    segments = build_document_segments(_documents(case))
    windows = build_segment_windows(
        segments,
        max_segments=MAX_SEGMENTS,
        max_characters=MAX_CHARACTERS,
        overlap_segments=OVERLAP_SEGMENTS,
    )
    window_results: list[dict[str, Any]] = []
    validations: list[WindowValidation] = []
    for window in windows:
        remaining = case_timeout - (time.perf_counter() - case_started)
        if remaining <= 0:
            return {
                "case_id": case["id"],
                "endpoint": endpoint,
                "elapsed_seconds": time.perf_counter() - case_started,
                "error": f"CaseTimeout: exceeded {case_timeout:.3f}s",
                "window_count": len(windows),
                "windows": window_results,
            }
        result = execute_window_hard_timeout(
            window,
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
                "elapsed_seconds": time.perf_counter() - case_started,
                "error": f"{window.window_id}: {result['error']}",
                "window_count": len(windows),
                "windows": window_results,
            }
        validations.append(WindowValidation(
            window_id=window.window_id,
            validation=GroundingValidationResult.model_validate(result["validation"]),
        ))

    aggregate = aggregate_window_validations(windows, validations, segments)
    elapsed = time.perf_counter() - case_started
    return {
        "case_id": case["id"],
        "endpoint": endpoint,
        "elapsed_seconds": elapsed,
        "finish_reason": (
            "stop" if all(item.get("finish_reason") == "stop" for item in window_results)
            else "mixed"
        ),
        "window_count": len(windows),
        "raw_extraction": {
            "windows": [
                {
                    "window_id": item["window_id"],
                    "request_id": item["request_id"],
                    "finish_reason": item.get("finish_reason"),
                    "elapsed_seconds": item.get("elapsed_seconds"),
                    "wall_seconds": item.get("wall_seconds"),
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
        return [
            {
                "case_id": case["id"],
                "endpoint": endpoint,
                "error": f"EndpointBusy: {metrics}",
            }
            for case in cases
        ]
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        item = _windowed_case(
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


def run_windowed_shadow(
    cases: Sequence[dict[str, Any]],
    *,
    endpoints: Sequence[str],
    model: str,
    hard_timeout: float = HARD_TIMEOUT_SECONDS,
    case_timeout: float = CASE_TIMEOUT_SECONDS,
    max_tokens: int = MAX_TOKENS,
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
    return sorted(results, key=lambda item: str(item["case_id"]))


def predeclared_shadow_probe_cases(
    cases: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for scenario in sorted({str(case["scenario"]) for case in cases}):
        selected.append(min(
            (case for case in cases if str(case["scenario"]) == scenario),
            key=lambda case: str(case["id"]),
        ))
    return selected


def _validate_development_annotations(cases: Sequence[dict[str, Any]]) -> None:
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
    parser.add_argument("--cancellation-only", action="store_true")
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return _worker_main()

    args = _args()
    if args.max_tokens != MAX_TOKENS:
        raise SystemExit(f"frozen W1 requires --max-tokens={MAX_TOKENS}")
    endpoints = args.endpoint or DEFAULT_ENDPOINTS
    if args.dataset == "development":
        cases = _read_jsonl(args.cases)
        _validate_development_annotations(cases)
    else:
        cases = load_blind_cases()
        if args.predeclared_shadow_probe:
            cases = predeclared_shadow_probe_cases(cases)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if str(case["id"]) in selected]
    if not cases:
        raise SystemExit("no cases selected")

    if args.cancellation_only:
        windows = [
            window
            for case in cases
            for window in build_segment_windows(
                build_document_segments(_documents(case)),
                max_segments=MAX_SEGMENTS,
                max_characters=MAX_CHARACTERS,
                overlap_segments=OVERLAP_SEGMENTS,
            )
        ]
        selected_window = max(
            windows,
            key=lambda window: (len(window.segments), window.character_count, window.window_id),
        )
        result = probe_server_cancellation(
            selected_window,
            endpoint=endpoints[0],
            model=args.model,
            max_tokens=args.max_tokens,
        )
        payload = {
            "schema_version": 1,
            "runner": "record-window-w1",
            "endpoint": endpoints[0],
            "selected_window": selected_window.model_dump(),
            "result": result,
        }
        _atomic_write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if result.get("passed") else 1

    raw = run_windowed_shadow(
        cases,
        endpoints=endpoints,
        model=args.model,
        hard_timeout=args.hard_timeout,
        case_timeout=args.case_timeout,
        max_tokens=args.max_tokens,
    )
    if args.raw_output:
        _atomic_write_json(args.raw_output, {
            "schema_version": 1,
            "runner": "record-window-w1",
            "model": args.model,
            "endpoints": endpoints,
            "config": {
                "max_segments": MAX_SEGMENTS,
                "max_characters": MAX_CHARACTERS,
                "overlap_segments": OVERLAP_SEGMENTS,
                "max_tokens": args.max_tokens,
                "hard_timeout": args.hard_timeout,
                "case_timeout": args.case_timeout,
            },
            "results": raw,
        })
    report = evaluate(cases, raw) if args.dataset == "development" else evaluate_blind(cases, raw)
    report["runner"] = "record-window-w1"
    report["model"] = args.model
    report["window_config"] = {
        "max_segments": MAX_SEGMENTS,
        "max_characters": MAX_CHARACTERS,
        "overlap_segments": OVERLAP_SEGMENTS,
        "max_tokens": args.max_tokens,
        "hard_timeout": args.hard_timeout,
        "case_timeout": args.case_timeout,
    }
    _atomic_write_json(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "case_count": report["case_count"],
        "completed_count": report.get("completed_count"),
        "shadow": report.get("shadow"),
        "failures": report.get("failures"),
    }, ensure_ascii=False, indent=2))
    return 1 if report.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
