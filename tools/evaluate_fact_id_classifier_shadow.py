#!/usr/bin/env python3
"""Evaluate the offline candidate-first classifier on frozen cases.

Every model request runs in a disposable child process.  The parent owns the
hard timeout and reconstructs predictions from the immutable candidate
inventory, so a partial/late response cannot leak into a later case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from llm_gateway import strip_thinking  # noqa: E402
from fact_id_classifier_shadow import (  # noqa: E402
    CandidateClassification,
    CandidateSpan,
    CandidateValidation,
    build_candidate_inventory,
    build_classifier_prompt,
    choices_to_predictions,
    validate_candidate_classification,
)
from tools.evaluate_record_window_shadow import (  # noqa: E402
    _wait_until_idle,
)
from tools.evaluate_segment_grounded_shadow import (  # noqa: E402
    DEFAULT_CASES,
    _aggregate_case_scores,
    _atomic_write_json,
    _documents,
    _flatten_baseline,
    _read_jsonl,
    resolve_gold,
    score_predictions,
)


DEFAULT_ENDPOINTS = [f"http://127.0.0.1:{port}" for port in range(8007, 8011)]
HARD_TIMEOUT_SECONDS = 150.0
MAX_TOKENS = 4_096


def _base_endpoint(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


def _worker_call(payload: dict[str, Any]) -> dict[str, Any]:
    import httpx
    from openai import OpenAI

    case = dict(payload["case"])
    bundle, candidates = build_candidate_inventory(
        str(case.get("cv_text") or ""),
        str(case.get("query_text") or ""),
        str(case.get("jd_text") or ""),
    )
    system_prompt, user_prompt = build_classifier_prompt(candidates)
    http_client = httpx.Client(
        trust_env=False,
        timeout=float(payload["socket_timeout"]),
    )
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
                    "name": "candidate_first_classification",
                    "strict": True,
                    "schema": CandidateClassification.model_json_schema(),
                },
            },
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "top_p": 1.0,
            },
        )
        elapsed = time.perf_counter() - started
        content = strip_thinking(response.choices[0].message.content or "")
        extraction = CandidateClassification.model_validate(json.loads(content))
        validation = validate_candidate_classification(extraction, candidates)
        return {
            "case_id": case["id"],
            "endpoint": payload["endpoint"],
            "elapsed_seconds": elapsed,
            "finish_reason": response.choices[0].finish_reason,
            "candidate_count": len(candidates),
            "raw_extraction": extraction.model_dump(),
            "validation": validation.model_dump(),
            "candidate_inventory": [item.model_dump() for item in candidates],
        }
    finally:
        client.close()


def _worker_main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        result = _worker_call(payload)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


def _spawn_worker() -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _kill_worker(process: subprocess.Popen[str]) -> tuple[str, str, str]:
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


def execute_case_hard_timeout(
    case: dict[str, Any],
    *,
    endpoint: str,
    model: str,
    hard_timeout: float = HARD_TIMEOUT_SECONDS,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    payload = {
        "case": case,
        "endpoint": endpoint,
        "model": model,
        "max_tokens": max_tokens,
        "socket_timeout": hard_timeout + 30.0,
    }
    process = _spawn_worker()
    started = time.perf_counter()
    try:
        stdout, stderr = process.communicate(
            json.dumps(payload, ensure_ascii=False),
            timeout=hard_timeout,
        )
    except subprocess.TimeoutExpired:
        signal_sent, stdout, stderr = _kill_worker(process)
        idle, metrics = _wait_until_idle(endpoint, timeout=20.0)
        return {
            "case_id": case["id"],
            "endpoint": endpoint,
            "elapsed_seconds": time.perf_counter() - started,
            "error": f"HardTimeout: exceeded {hard_timeout:.3f}s",
            "timed_out": True,
            "termination_signal": signal_sent,
            "server_idle_after_disconnect": idle,
            "post_metrics": metrics,
            "worker_stdout": stdout[-2_000:],
            "worker_stderr": stderr[-2_000:],
        }
    elapsed = time.perf_counter() - started
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return {
            "case_id": case["id"],
            "endpoint": endpoint,
            "elapsed_seconds": elapsed,
            "error": f"WorkerProtocolError: {exc}",
            "worker_stdout": stdout[-2_000:],
            "worker_stderr": stderr[-2_000:],
        }
    if process.returncode != 0 or result.get("error"):
        return {
            "case_id": case["id"],
            "endpoint": endpoint,
            "elapsed_seconds": elapsed,
            "error": str(result.get("error") or f"worker exited {process.returncode}"),
            "worker_stdout": stdout[-2_000:],
            "worker_stderr": stderr[-2_000:],
        }
    result["wall_seconds"] = elapsed
    return result


def _run_endpoint_cases(
    endpoint: str,
    cases: Sequence[dict[str, Any]],
    *,
    model: str,
    hard_timeout: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    idle, metrics = _wait_until_idle(endpoint, timeout=5.0)
    if not idle:
        return [
            {
                "case_id": case["id"],
                "endpoint": endpoint,
                "error": f"EndpointBusy: {metrics}",
            }
            for case in cases
        ]
    return [
        execute_case_hard_timeout(
            case,
            endpoint=endpoint,
            model=model,
            hard_timeout=hard_timeout,
            max_tokens=max_tokens,
        )
        for case in cases
    ]


def run_shadow(
    cases: Sequence[dict[str, Any]],
    *,
    endpoints: Sequence[str],
    model: str,
    hard_timeout: float,
    max_tokens: int,
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
                assigned,
                model=model,
                hard_timeout=hard_timeout,
                max_tokens=max_tokens,
            )
            for endpoint, assigned in zip(endpoints, assignments)
            if assigned
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    return sorted(results, key=lambda item: str(item["case_id"]))


def _candidate_pool_recall(
    case: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
) -> float:
    gold = resolve_gold(case)
    covered = 0
    for item in gold:
        if any(
            candidate["source_id"] == item["source_id"]
            and int(candidate["start"]) <= int(item["start"])
            and int(candidate["end"]) >= int(item["end"])
            for candidate in candidates
        ):
            covered += 1
    return covered / len(gold) if gold else 1.0


def evaluate_report(
    cases: Sequence[dict[str, Any]],
    results: Sequence[dict[str, Any]],
    *,
    dataset_path: Path,
    model: str,
) -> dict[str, Any]:
    by_id = {str(item["case_id"]): item for item in results}
    case_results: list[dict[str, Any]] = []
    for case in cases:
        gold = resolve_gold(case)
        baseline = _flatten_baseline(case)
        item: dict[str, Any] = {
            "case_id": case["id"],
            "group": case.get("group"),
            "gold_count": len(gold),
            "candidate_pool_recall": None,
            "baseline": score_predictions(baseline, gold),
        }
        result = by_id.get(str(case["id"]))
        if result is None:
            item["runtime"] = {"error": "missing result"}
            case_results.append(item)
            continue
        item["runtime"] = {
            key: result.get(key)
            for key in ("endpoint", "elapsed_seconds", "wall_seconds", "finish_reason", "error")
            if result.get(key) is not None
        }
        item["candidate_count"] = result.get("candidate_count")
        _, current_candidates = build_candidate_inventory(
            str(case.get("cv_text") or ""),
            str(case.get("query_text") or ""),
            str(case.get("jd_text") or ""),
        )
        item["candidate_pool_recall"] = _candidate_pool_recall(
            case, [candidate.model_dump() for candidate in current_candidates],
        )
        if not result.get("error"):
            # Re-run deterministic validation from the raw model choices and
            # the frozen inventory.  This makes --evaluate-raw useful for
            # replaying a postprocessor change without spending another 27B
            # request, and prevents stale child validation from masking it.
            inventory = tuple(current_candidates)
            extraction = CandidateClassification.model_validate(
                result.get("raw_extraction") or {"choices": []}
            )
            validation_model = validate_candidate_classification(extraction, inventory)
            validation = validation_model.model_dump()
            item["validation"] = validation
            if validation.get("valid"):
                predictions = choices_to_predictions(
                    CandidateValidation.model_validate(validation),
                )
                item["candidate"] = score_predictions(predictions, gold)
                item["predictions"] = predictions
            else:
                item["candidate"] = score_predictions([], gold)
                item["invalid_validation"] = True
            item["raw_extraction"] = result.get("raw_extraction")
        case_results.append(item)

    candidate_cases = [item for item in case_results if "candidate" in item]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "dataset_path": str(dataset_path.resolve()),
        "model": model,
        "case_count": len(cases),
        "baseline": _aggregate_case_scores(case_results, "baseline"),
        "candidate": _aggregate_case_scores(candidate_cases, "candidate") if candidate_cases else None,
        "cases": case_results,
        "failures": [item for item in results if item.get("error")],
        "invalid_validation_case_count": sum(
            bool(item.get("invalid_validation")) for item in case_results
        ),
        "candidate_pool_recall": {
            "mean": sum(float(item["candidate_pool_recall"] or 0.0) for item in case_results) / len(case_results)
            if case_results else 0.0,
            "min": min((float(item["candidate_pool_recall"] or 0.0) for item in case_results), default=0.0),
        },
    }
    latencies = [
        float(item["runtime"]["elapsed_seconds"])
        for item in candidate_cases
        if item.get("runtime", {}).get("elapsed_seconds") is not None
    ]
    payload["latency"] = {
        "mean_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "max_seconds": max(latencies, default=0.0),
    }
    return payload


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--model", default="Qwen3.5-27B-AWQ")
    parser.add_argument("--hard-timeout", type=float, default=HARD_TIMEOUT_SECONDS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--evaluate-raw", type=Path)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.worker:
        return _worker_main()
    cases = _read_jsonl(args.cases)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if str(case["id"]) in selected]
    if not cases:
        raise SystemExit("no cases selected")
    if args.inventory_only:
        for case in cases:
            bundle, candidates = build_candidate_inventory(
                str(case.get("cv_text") or ""),
                str(case.get("query_text") or ""),
                str(case.get("jd_text") or ""),
            )
            print(json.dumps({
                "case_id": case["id"],
                "candidate_count": len(candidates),
                "candidates": [item.model_dump() for item in candidates],
            }, ensure_ascii=False))
        return 0
    if args.evaluate_raw:
        loaded = json.loads(args.evaluate_raw.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            results = list(loaded.get("results") or [])
            endpoints = list(loaded.get("endpoints") or args.endpoint or [])
        else:
            results = list(loaded)
            endpoints = list(args.endpoint or [])
    else:
        endpoints = args.endpoint or [DEFAULT_ENDPOINTS[0]]
        results = run_shadow(
            cases,
            endpoints=endpoints,
            model=args.model,
            hard_timeout=args.hard_timeout,
            max_tokens=args.max_tokens,
        )
        if args.raw_output:
            _atomic_write_json(args.raw_output, {
                "schema_version": 1,
                "model": args.model,
                "endpoints": endpoints,
                "results": results,
            })
    output = args.output or Path("/tmp/fact-id-classifier-report.json")
    report = evaluate_report(
        cases,
        results,
        dataset_path=args.cases,
        model=args.model,
    )
    _atomic_write_json(output, report)
    print(json.dumps({
        "output": str(output),
        "candidate": report.get("candidate"),
        "candidate_pool_recall": report.get("candidate_pool_recall"),
        "latency": report.get("latency"),
        "failures": report.get("failures"),
        "invalid_validation_case_count": report.get("invalid_validation_case_count"),
    }, ensure_ascii=False, indent=2))
    return 1 if report.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
