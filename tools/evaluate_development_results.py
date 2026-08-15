#!/usr/bin/env python3
"""Independently audit development-set API responses against case inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "core"
for value in (REPO_ROOT, CORE_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from atomic_fact_audit import audit_atomic_facts  # noqa: E402
from evidence_binding import bind_resume_evidence  # noqa: E402
from resume_copilot_service import _canonical_resume_from_render_data  # noqa: E402
from source_adapter import build_source_bundle  # noqa: E402
from validation_sets.public_resume_holdout.evaluate import (  # noqa: E402
    summarize_rows,
    validate_public_response,
)


EVALUATOR_VERSION = "development-factuality-1.0"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _result_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("rows", []) if isinstance(payload, dict) else []
        if not isinstance(values, list):
            raise ValueError(f"{path}: rows must be a list")
        rows.extend(item for item in values if isinstance(item, dict))
    return rows


def _response(row: dict[str, Any]) -> dict[str, Any] | None:
    raw = row.get("raw")
    if isinstance(raw, dict) and isinstance(raw.get("resume_data"), dict):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("raw"), dict):
        nested = raw["raw"]
        if isinstance(nested.get("resume_data"), dict):
            return nested
    return None


def audit_development_rows(
    cases: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_case = {str(case["id"]): case for case in cases}
    seen: set[str] = set()
    audited_rows: list[dict[str, Any]] = []
    for result_row in result_rows:
        case_id = str(result_row.get("id") or "")
        if case_id not in by_case:
            raise ValueError(f"result contains unknown case: {case_id!r}")
        if case_id in seen:
            raise ValueError(f"duplicate result case: {case_id}")
        seen.add(case_id)
        case = by_case[case_id]
        response = _response(result_row)
        row: dict[str, Any] = {
            "id": case_id,
            "scenario": case.get("scenario"),
            "industry": case.get("industry"),
            "input_profile": "inline-development",
            "request_ok": response is not None and bool(result_row.get("ok", True)),
            "status": result_row.get("status"),
            "elapsed_s": float(result_row.get("elapsed_s") or 0.0),
            "audit_ok": False,
            "audit_error": "",
        }
        if response is None:
            audited_rows.append(row)
            continue
        row["response_contract"] = validate_public_response(case, response)
        try:
            source = build_source_bundle(
                str(case.get("cv_text") or ""),
                str(case.get("query") or ""),
                str(case.get("jd_text") or case.get("target_jd") or ""),
            )
            resume = _canonical_resume_from_render_data(response["resume_data"])
            bindings = bind_resume_evidence(resume, source)
            row["external_audit"] = audit_atomic_facts(
                source=source,
                resume=resume,
                evidence_bindings=bindings,
            )
            row["external_audit"]["binding_count"] = len(bindings)
            row["audit_ok"] = True
        except Exception as exc:
            row["audit_error"] = f"{type(exc).__name__}:{exc}"
        audited_rows.append(row)

    missing = sorted(set(by_case) - seen)
    if missing:
        raise ValueError(f"missing result case(s): {', '.join(missing)}")
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "summary": summarize_rows(audited_rows),
        "rows": audited_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_development_rows(
        _load_jsonl(Path(args.cases)),
        _result_rows(Path(value) for value in args.input),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
