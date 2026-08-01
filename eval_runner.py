"""Offline evaluator for resume-copilot heuristic quality checks.

This runner avoids network/LLM calls. It validates the product scoring and
field-completeness path on the fixed JSONL case set.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

CORE_DIR = Path(__file__).resolve().parent / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from resume_product_logic import (
    heuristic_resume_from_text,
    infer_industry,
    normalize_resume_data_for_product,
)
from resume_scoring import score_resume
from resume_validator import check_required_fields, check_sort_order, check_time_conflicts


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def run_eval(path: Path) -> dict[str, Any]:
    cases = _load_cases(path)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        try:
            query_text = str(case.get("query", "") or "")
            cv_text = str(case.get("cv_text", "") or "")
            source_text = "\n".join(part for part in (query_text, cv_text) if part)
            jd_text = str(case.get("jd_text", ""))
            industry = infer_industry(query_text, cv_text, jd_text)
            resume = heuristic_resume_from_text(source_text, industry, "")
            resume = normalize_resume_data_for_product(resume, raw_text=source_text, industry=industry, target_role="")
            missing = check_required_fields(resume)
            conflicts = check_time_conflicts(resume) + check_sort_order(resume)
            user_report = {
                "missing_field_suggestions": [m.model_dump() for m in missing],
                "conflict_confirmations": [c.model_dump() for c in conflicts],
                "generation_direction": f"评测方向：{industry}",
                "ocr_warnings": [],
            }
            score = score_resume(
                resume,
                original_text=source_text,
                user_report=user_report,
                job_family=industry,
                missing_fields=missing,
                conflicts=conflicts,
            )
            failure_reasons: list[str] = []
            if len(missing) > 0:
                failure_reasons.append("missing_fields")
            if len(conflicts) > 0:
                failure_reasons.append("conflicts")
            if score.readability < 8:
                failure_reasons.append("readability")
            if score.expression < 40:
                failure_reasons.append("expression")
            if industry != case.get("industry"):
                failure_reasons.append("industry_mismatch")
            rows.append(
                {
                    "id": case.get("id"),
                    "expected_industry": case.get("industry"),
                    "industry": industry,
                    "score": score.model_dump(),
                    "usable": score.total >= 90,
                    "missing_count": len(missing),
                    "conflict_count": len(conflicts),
                    "failure_reasons": failure_reasons if score.total < 90 else [],
                }
            )
        except Exception as exc:
            errors.append({"id": case.get("id"), "error": str(exc)})
    elapsed = time.perf_counter() - started
    generated = len(rows)
    usable = sum(1 for row in rows if row["usable"])
    reason_counts: dict[str, int] = {}
    for row in rows:
        for reason in row.get("failure_reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    avg = lambda key: round(sum(row["score"][key] for row in rows) / max(generated, 1), 2)
    return {
        "mode": "heuristic_baseline_no_llm",
        "notes": "This offline run validates parsers, validators, scoring, and reporting only. Final acceptance should use /resume-copilot with LLM enabled.",
        "case_count": len(cases),
        "generated_count": generated,
        "usable_count": usable,
        "usable_rate": round(usable / max(generated, 1), 3),
        "failure_count": len(errors),
        "failure_rate": round(len(errors) / max(len(cases), 1), 3),
        "average_total": avg("total"),
        "average_readability": avg("readability"),
        "average_completeness": avg("completeness"),
        "average_expression": avg("expression"),
        "average_response": avg("response"),
        "average_generation_time_s": round(elapsed / max(generated, 1), 3),
        "low_score_reason_counts": reason_counts,
        "errors": errors,
        "rows": rows,
    }


if __name__ == "__main__":
    report = run_eval(Path(__file__).with_name("eval_cases.jsonl"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
