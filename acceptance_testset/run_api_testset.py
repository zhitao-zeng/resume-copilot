"""Run the acceptance testset against POST /resume-copilot.

Usage:
  python acceptance_testset/run_api_testset.py --base-url http://127.0.0.1:8001
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _filter_cases(
    cases: list[dict[str, Any]],
    requested_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if not requested_ids:
        return cases
    selected_ids = [
        item.strip()
        for value in requested_ids
        for item in str(value).split(",")
        if item.strip()
    ]
    by_id = {str(case.get("id")): case for case in cases}
    missing = [case_id for case_id in selected_ids if case_id not in by_id]
    if missing:
        raise SystemExit(f"Unknown --case-id value(s): {', '.join(missing)}")
    return [by_id[case_id] for case_id in selected_ids]


def _part_text(boundary: str, name: str, value: str) -> bytes:
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
        f"{value}\r\n"
    ).encode("utf-8")


def _part_file(boundary: str, name: str, path: Path) -> bytes:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("utf-8")
    return header + path.read_bytes() + b"\r\n"


def _part_inline_file(boundary: str, name: str, filename: str, value: str) -> bytes:
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        "Content-Type: text/plain; charset=utf-8\r\n\r\n"
    ).encode("utf-8")
    return header + str(value).encode("utf-8") + b"\r\n"


def _multipart(case: dict[str, Any]) -> tuple[bytes, str]:
    boundary = f"resume-copilot-{int(time.time() * 1000)}"
    parts: list[bytes] = []
    if case.get("query"):
        parts.append(_part_text(boundary, "query", str(case["query"])))
    target_jd = case.get("target_jd") or case.get("jd_text")
    if target_jd:
        parts.append(_part_text(boundary, "target_jd", str(target_jd)))
    if case.get("cv_text"):
        parts.append(_part_inline_file(
            boundary,
            "cv",
            f"{case.get('id', 'case')}.txt",
            str(case["cv_text"]),
        ))
    for field, api_name in (
        ("cv_path", "cv"),
        ("target_jd_file_path", "target_jd_file"),
        ("cv_template_path", "cv_template"),
    ):
        value = case.get(field)
        if not value:
            continue
        path = ROOT / value
        parts.append(_part_file(boundary, api_name, path))
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), boundary


def _call_case(base_url: str, case: dict[str, Any], timeout: int) -> dict[str, Any]:
    body, boundary = _multipart(case)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/resume-copilot",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            data = json.loads(payload)
            validation_failures = _validate_response(case, data)
            elapsed = time.perf_counter() - started
            score_payload = data.get("score") or {}
            score_total = score_payload.get("total") if isinstance(score_payload, dict) else score_payload
            return {
                "id": case["id"],
                "ok": True,
                "status": resp.status,
                "elapsed_s": round(elapsed, 3),
                "score": score_total,
                "score_breakdown": score_payload if isinstance(score_payload, dict) else {},
                "scenario": data.get("scenario"),
                "industry": data.get("industry"),
                "user_stage": data.get("user_stage"),
                "missing_count": len(data.get("missing_fields") or []),
                "conflict_count": len(data.get("conflicts") or []),
                "docx": (data.get("files") or {}).get("docx"),
                "reply_text": data.get("reply_text", ""),
                "validation_failures": validation_failures,
                "raw": data,
            }
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started
        return {"id": case["id"], "ok": False, "status": exc.code, "elapsed_s": round(elapsed, 3), "error": exc.read().decode("utf-8", errors="replace")}
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {"id": case["id"], "ok": False, "status": None, "elapsed_s": round(elapsed, 3), "error": str(exc)}


def _validate_response(case: dict[str, Any], data: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if data.get("scenario") != case.get("scenario"):
        failures.append(f"scenario_mismatch:{data.get('scenario')}!={case.get('scenario')}")
    if data.get("industry") != case.get("industry"):
        failures.append(f"industry_mismatch:{data.get('industry')}!={case.get('industry')}")

    missing_fields = data.get("missing_fields") or []
    missing_blob = json.dumps(missing_fields, ensure_ascii=False)
    for expected in case.get("expected_missing_fields") or []:
        if str(expected) not in missing_blob:
            failures.append(f"missing_field_not_reported:{expected}")

    conflicts = data.get("conflicts") or []
    conflict_blob = json.dumps(conflicts, ensure_ascii=False)
    for expected in case.get("expected_conflicts") or []:
        expected_text = str(expected)
        mapped = False
        if expected_text == "experience_time_overlap":
            mapped = any("重叠" in str(item.get("description", "")) or item.get("field") == "experience" for item in conflicts if isinstance(item, dict))
        if not mapped and expected_text not in conflict_blob and expected_text.replace("_", "") not in conflict_blob.replace("_", ""):
            failures.append(f"conflict_not_reported:{expected}")

    resume_blob = json.dumps(data.get("resume_data") or {}, ensure_ascii=False)
    reply_text = str(data.get("reply_text") or "")
    for forbidden in case.get("forbidden_fabrication") or []:
        if str(forbidden) and (str(forbidden) in resume_blob or str(forbidden) in reply_text):
            failures.append(f"forbidden_fabrication_present:{forbidden}")

    files = data.get("files") or {}
    docx = files.get("docx")
    if not docx:
        failures.append("docx_missing")
    if not reply_text.strip():
        failures.append("reply_text_missing")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--cases", default=str(ROOT / "cases.jsonl"))
    parser.add_argument("--out", default=None, help="Output JSON path. If omitted, auto-creates results_<timestamp>/results.json")
    parser.add_argument("--timeout", type=int, default=520)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run selected case ID(s); repeat the option or pass comma-separated IDs",
    )
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    cases = _filter_cases(cases, args.case_id)
    if args.limit:
        cases = cases[: args.limit]

    # Auto-create timestamped output directory
    if args.out is None:
        ts = time.strftime("%m%d_%H%M%S")
        out_dir = ROOT / f"results_{ts}"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "results.json"
        args.out = str(out_path)
        print(f"Results will be saved to: {out_path}", flush=True)

    rows = []
    for i, case in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] Running {case.get('id', '?')}...", flush=True)
        rows.append(_call_case(args.base_url, case, args.timeout))
        print(f"[{i+1}/{len(cases)}] {case.get('id', '?')}: score={rows[-1].get('score','-')} ok={rows[-1].get('ok')} elapsed={rows[-1].get('elapsed_s','-')}s", flush=True)
    ok_rows = [row for row in rows if row["ok"]]
    usable_rows = [
        row
        for row in ok_rows
        if isinstance(row.get("score"), (int, float)) and row["score"] >= 90 and not row.get("validation_failures")
    ]
    validation_failed = [row for row in ok_rows if row.get("validation_failures")]
    def _avg_breakdown(key: str) -> float:
        return round(
            sum(float((row.get("score_breakdown") or {}).get(key) or 0) for row in ok_rows) / max(len(ok_rows), 1),
            2,
        )

    summary = {
        "case_count": len(cases),
        "success_count": len(ok_rows),
        "failure_count": len(rows) - len(ok_rows),
        "failure_rate": round((len(rows) - len(ok_rows)) / max(len(rows), 1), 3),
        "usable_count": len(usable_rows),
        "usable_rate": round(len(usable_rows) / max(len(ok_rows), 1), 3),
        "validation_failure_count": len(validation_failed),
        "average_score": round(sum(float(row.get("score") or 0) for row in ok_rows) / max(len(ok_rows), 1), 2),
        "average_readability": _avg_breakdown("readability"),
        "average_completeness": _avg_breakdown("completeness"),
        "average_expression": _avg_breakdown("expression"),
        "average_response": _avg_breakdown("response"),
        "average_elapsed_s": round(sum(float(row["elapsed_s"]) for row in rows) / max(len(rows), 1), 3),
    }
    output = {"summary": summary, "rows": rows}
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
