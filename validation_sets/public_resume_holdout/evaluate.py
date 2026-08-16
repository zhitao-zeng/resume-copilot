#!/usr/bin/env python3
"""Run the frozen public holdout with one version-independent evaluator.

Candidate-provided scores and quality reports are deliberately ignored.  The
same local evaluator rebuilds a source ledger from canonical, span-annotated
texts and audits each candidate's public ``resume_data`` response.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import mimetypes
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EVALUATOR_VERSION = "public-holdout-evaluator-1.2"
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
ACCEPTANCE_ROOT = REPO_ROOT / "acceptance_testset"
CORE_ROOT = REPO_ROOT / "core"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from atomic_fact_audit import audit_atomic_facts  # noqa: E402
from evidence_binding import bind_resume_evidence  # noqa: E402
from quality_report import (  # noqa: E402
    _ACTION_SIGNAL,
    _METHOD_SIGNAL,
    _OUTPUT_SIGNAL,
    _QUANTIFIED_SIGNAL,
    build_quality_report,
)
from resume_copilot_service import _canonical_resume_from_render_data  # noqa: E402
from source_adapter import build_source_bundle  # noqa: E402
from v2_schemas import SourceBundle, SourceSpan  # noqa: E402


SOURCE_KIND_TO_ID = {"cv": "resume", "query": "query", "jd": "jd"}
CRITICAL_STRUCTURAL_CATEGORIES = (
    "organization",
    "role",
    "period",
    "education",
    "credential",
    "metric",
)
ALL_STRUCTURAL_CATEGORIES = CRITICAL_STRUCTURAL_CATEGORIES + ("skill_tool",)
REPLY_COMPONENTS = ("生成方向", "缺失信息", "岗位建议", "冲突检查")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _resolve_case_file(value: str) -> Path:
    path = (ACCEPTANCE_ROOT / value).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"case file escapes repository: {value}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _resolve_canonical_file(value: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"canonical source escapes dataset root: {value}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _overlaps(span: SourceSpan, intervals: list[tuple[int, int]]) -> bool:
    return any(
        span.char_start < end and start < span.char_end
        for start, end in intervals
    )


def build_annotated_source(annotation: dict[str, Any]) -> SourceBundle:
    """Rebuild a source ledger, then enforce the dataset's eligible spans."""

    texts = {"resume": "", "query": "", "jd": ""}
    eligible: dict[str, list[tuple[int, int]]] = defaultdict(list)
    expected_hashes: dict[str, str] = {}

    for source in annotation.get("sources", []):
        kind = str(source.get("kind") or "")
        source_id = SOURCE_KIND_TO_ID.get(kind)
        if not source_id:
            raise ValueError(f"unknown annotation source kind: {kind!r}")
        path = _resolve_canonical_file(str(source["canonical_text_path"]))
        actual_hash = _sha256(path)
        expected_hash = str(source.get("sha256") or "")
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(
                f"canonical source hash mismatch for {path}: "
                f"{actual_hash} != {expected_hash}"
            )
        expected_hashes[source_id] = actual_hash
        texts[source_id] = path.read_text(encoding="utf-8")
        for unit in source.get("units", []):
            if not unit.get("candidate_for_resume"):
                continue
            start, end = unit["source_span"]
            eligible[source_id].append((int(start), int(end)))

    bundle = build_source_bundle(texts["resume"], texts["query"], texts["jd"])
    for fact in bundle.fact_units:
        fact.fact_eligible = bool(
            fact.source_type != "jd"
            and any(
                _overlaps(span, eligible.get(span.source_id, []))
                for span in fact.source_spans
            )
        )

    eligible_fact_ids = {
        fact.fact_id for fact in bundle.fact_units if fact.fact_eligible
    }
    for block in bundle.blocks:
        block.fact_eligible = bool(
            block.source_type != "jd"
            and any(
                _overlaps(span, eligible.get(span.source_id, []))
                for span in block.source_spans
            )
        )
        block.fact_ids = [
            fact_id for fact_id in block.fact_ids if fact_id in eligible_fact_ids
        ]
    # Evidence binding historically treats every resume block as eligible.
    # Remove annotation-ineligible candidate blocks entirely so a direct
    # binding cannot smuggle a placeholder back into the audit through
    # ``binding.source_claim``. JD blocks may remain as context because the
    # binder already excludes them from candidate evidence.
    bundle.blocks = [
        block for block in bundle.blocks
        if block.source_type == "jd" or block.fact_eligible
    ]
    return bundle


def audit_response(
    response: dict[str, Any],
    annotation: dict[str, Any],
) -> dict[str, Any]:
    """Audit one public API response without consulting its own score fields."""

    resume_data = response.get("resume_data")
    if not isinstance(resume_data, dict):
        raise ValueError("response.resume_data must be an object")
    source = build_annotated_source(annotation)
    resume = _canonical_resume_from_render_data(resume_data)
    bindings = bind_resume_evidence(resume, source)
    audit = audit_atomic_facts(
        source=source,
        resume=resume,
        evidence_bindings=bindings,
    )
    audit["binding_count"] = len(bindings)
    return audit


def _case_jd_text(case: dict[str, Any]) -> str:
    """Load only explicit JD text for the deterministic quality probe."""

    value = str(case.get("target_jd") or "")
    if value.strip():
        return value
    path_value = case.get("target_jd_file_path")
    if not path_value:
        return ""
    path = _resolve_case_file(str(path_value))
    if path.suffix.casefold() not in {".txt", ".md", ".text"}:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _resume_bullets(resume_data: dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    for section in ("experience", "projects", "research", "activities", "teaching"):
        records = resume_data.get(section)
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            values = record.get("bullets")
            if not isinstance(values, list):
                continue
            bullets.extend(
                str(value).strip()
                for value in values
                if str(value or "").strip()
            )
    return bullets


def assess_generation_quality(
    case: dict[str, Any],
    annotation: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    """Return a deterministic generation-quality vector, not a single score.

    Factuality remains in ``external_audit``.  This vector measures expression
    and usefulness separately so a candidate cannot trade unsupported content
    for a better-looking quality number.  The STAR signals are deliberately
    generic linguistic cues; they are a reproducible proxy and not a
    replacement for blinded human review.
    """

    resume_data = response.get("resume_data")
    resume_data = resume_data if isinstance(resume_data, dict) else {}
    source = build_annotated_source(annotation)
    resume = _canonical_resume_from_render_data(resume_data)
    bindings = bind_resume_evidence(resume, source)
    quality_report = build_quality_report(
        source=source,
        resume=resume,
        evidence_bindings=bindings,
        missing_fields=response.get("missing_fields") or [],
        jd_text=_case_jd_text(case),
        jd_supplied=bool(case.get("target_jd") or case.get("target_jd_file_path")),
        target_role=str(case.get("target_role") or ""),
        framework_mode=bool(resume_data.get("framework")),
    )

    bullets = _resume_bullets(resume_data)
    bullet_details: list[dict[str, Any]] = []
    for bullet in bullets:
        action = bool(_ACTION_SIGNAL.search(bullet))
        method = bool(_METHOD_SIGNAL.search(bullet))
        result = bool(_OUTPUT_SIGNAL.search(bullet) or _QUANTIFIED_SIGNAL.search(bullet))
        dimensions = sum((action, method, result))
        # A very short bullet is a fragmentation signal only; it is never
        # considered factually wrong by this layer.
        compact = len(re.sub(r"\s+", "", bullet)) <= 12
        bullet_details.append({
            "chars": len(bullet),
            "action": action,
            "method": method,
            "result": result,
            "dimension_count": dimensions,
            "compact": compact,
        })

    reply = str(response.get("reply_text") or "")
    alignment = quality_report.get("job_alignment") or {}
    requirements = alignment.get("requirements") or []
    missing_fields = response.get("missing_fields")
    missing_fields = missing_fields if isinstance(missing_fields, list) else []
    report = response.get("user_report")
    report = report if isinstance(report, dict) else {}
    recommendations = report.get("targeted_suggestions")
    recommendations = recommendations if isinstance(recommendations, list) else []
    if not recommendations:
        generated_alignment = quality_report.get("job_alignment")
        generated_alignment = (
            generated_alignment if isinstance(generated_alignment, dict) else {}
        )
        recommendations = generated_alignment.get("recommendations")
        recommendations = recommendations if isinstance(recommendations, list) else []

    return {
        "proxy_version": "generation-quality-proxy-1.0",
        "blinded_human_review_required": True,
        "bullets": {
            "count": len(bullet_details),
            "avg_chars": round(
                sum(item["chars"] for item in bullet_details) / len(bullet_details), 2
            ) if bullet_details else 0.0,
            "star_complete_count": sum(
                item["action"] and item["method"] and item["result"]
                for item in bullet_details
            ),
            "star_complete_rate": round(
                sum(
                    item["action"] and item["method"] and item["result"]
                    for item in bullet_details
                ) / len(bullet_details), 4
            ) if bullet_details else 1.0,
            "two_or_more_dimension_rate": round(
                sum(item["dimension_count"] >= 2 for item in bullet_details)
                / len(bullet_details), 4
            ) if bullet_details else 1.0,
            "compact_bullet_rate": round(
                sum(item["compact"] for item in bullet_details) / len(bullet_details), 4
            ) if bullet_details else 0.0,
            "details": bullet_details[:40],
        },
        "job_alignment": {
            "available": bool(alignment.get("job_description_available")),
            "requirement_count": len(requirements),
            "supported_count": int(alignment.get("supported_requirement_count") or 0),
            "partial_count": int(alignment.get("partial_requirement_count") or 0),
            "missing_count": int(alignment.get("missing_requirement_count") or 0),
            "support_rate": round(
                (
                    int(alignment.get("supported_requirement_count") or 0)
                    + 0.5 * int(alignment.get("partial_requirement_count") or 0)
                ) / len(requirements), 4
            ) if requirements else None,
        },
        "reply_detail": {
            "chars": len(reply.strip()),
            "missing_field_count": len(missing_fields),
            "recommendation_count": len(recommendations),
            "component_coverage": sum(
                bool(value) for value in _reply_components(response, case).values()
            ),
        },
        "quality_report_opportunities": {
            "claim_improvement_count": len(
                quality_report.get("claim_improvement_opportunities") or []
            ),
            "unrepresented_source_count": int(
                (quality_report.get("source_preservation") or {}).get(
                    "unrepresented_item_count", 0
                )
            ),
        },
    }


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


def _multipart(case: dict[str, Any]) -> tuple[bytes, str]:
    boundary = f"resume-holdout-{time.time_ns()}"
    parts: list[bytes] = []
    if case.get("query"):
        parts.append(_part_text(boundary, "query", str(case["query"])))
    if case.get("target_jd"):
        parts.append(_part_text(boundary, "target_jd", str(case["target_jd"])))
    for key, api_name in (
        ("cv_path", "cv"),
        ("target_jd_file_path", "target_jd_file"),
        ("cv_template_path", "cv_template"),
    ):
        value = case.get(key)
        if value:
            parts.append(_part_file(boundary, api_name, _resolve_case_file(str(value))))
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), boundary


def _request_once(
    base_url: str,
    case: dict[str, Any],
    timeout: int,
) -> tuple[int | None, dict[str, Any] | None, str, float]:
    body, boundary = _multipart(case)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/resume-copilot",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
            elapsed = time.perf_counter() - started
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                return response.status, None, f"invalid_json:{exc}", elapsed
            return response.status, data, "", elapsed
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - started
        detail = exc.read().decode("utf-8", errors="replace")
        return exc.code, None, f"http_error:{detail[:1000]}", elapsed
    except Exception as exc:  # network timeout/errors must remain visible
        return None, None, f"request_error:{type(exc).__name__}:{exc}", (
            time.perf_counter() - started
        )


def _reply_components(response: dict[str, Any], case: dict[str, Any]) -> dict[str, bool]:
    reply = str(response.get("reply_text") or "")
    report = response.get("user_report")
    report = report if isinstance(report, dict) else {}
    quality = report.get("quality_report")
    quality = quality if isinstance(quality, dict) else {}

    generation = bool(
        report.get("generation_direction")
        or re.search(r"生成方向|优化方向|改写方向|求职方向", reply)
    )
    missing = bool(
        "missing_fields" in response
        or "missing_field_suggestions" in report
        or re.search(r"缺失信息|信息缺失|待补充|无需补充", reply)
    )
    job = bool(
        report.get("targeted_suggestions")
        or quality.get("job_alignment")
        or re.search(r"岗位建议|匹配建议|岗位匹配|目标岗位", reply)
    )
    conflict = bool(
        "conflicts" in response
        or "conflict_confirmations" in report
        or re.search(r"冲突检查|时间冲突|信息冲突|未发现.{0,8}(?:冲突|矛盾)", reply)
    )
    present = {
        "生成方向": generation,
        "缺失信息": missing,
        "岗位建议": job,
        "冲突检查": conflict,
    }
    required = set((case.get("expected_output") or {}).get("reply_text_must_cover") or [])
    return {name: bool(present[name]) for name in REPLY_COMPONENTS if name in required}


def _missing_field_present(expected: str, response: dict[str, Any]) -> bool:
    missing_fields = response.get("missing_fields")
    blob = json.dumps(missing_fields or [], ensure_ascii=False)
    expected = str(expected)
    if expected in blob:
        return True
    aliases = {
        "个人信息": ("姓名", "电话", "邮箱", "联系方式", "基本信息"),
        "教育背景": ("教育", "学校", "学历", "专业"),
        "经历": ("工作经历", "实习经历", "项目经历", "校园经历", "经历"),
        "技能": ("技能", "工具", "语言能力"),
    }
    return any(alias in blob for alias in aliases.get(expected, (expected,)))


def _reported_but_written(response: dict[str, Any]) -> list[str]:
    """Find only high-confidence contradictions in the missing-info reply."""

    data = response.get("resume_data")
    data = data if isinstance(data, dict) else {}
    meta = data.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    skills = data.get("skills")
    has_skills = bool(skills) and any(
        value for value in skills.values()
    ) if isinstance(skills, dict) else bool(skills)
    has_experience = any(
        isinstance(data.get(key), list) and data.get(key)
        for key in ("experience", "projects", "campus_experience", "activities", "research")
    )
    has_education = bool(data.get("education"))
    values = {
        "姓名": bool(meta.get("name")),
        "电话": bool(meta.get("phone")),
        "邮箱": bool(meta.get("email")),
        "教育": has_education,
        "经历": has_experience,
        "技能": has_skills,
    }
    entries = response.get("missing_fields")
    entries = entries if isinstance(entries, list) else []
    contradictions: list[str] = []
    for entry in entries:
        text = json.dumps(entry, ensure_ascii=False) if isinstance(entry, dict) else str(entry)
        if not re.search(r"未提供|缺失|为空|没有提供|待补充", text):
            continue
        if isinstance(entry, dict):
            field = str(entry.get("field") or "").casefold()
            structured_labels = {
                "name": {"姓名"},
                "contact": {"电话", "邮箱"},
                "phone": {"电话"},
                "email": {"邮箱"},
                "education": {"教育"},
                "experience": {"经历"},
                "work_experience": {"经历"},
                "projects": {"经历"},
                "skills": {"技能"},
            }.get(field)
            if structured_labels is not None:
                for label in structured_labels:
                    if values.get(label, False):
                        contradictions.append(label)
                continue
        for label, written in values.items():
            # ``教育经历缺失`` cannot contradict a written work/project
            # experience merely because both contain the substring ``经历``.
            if label == "经历" and re.search(r"教育\s*经历", text):
                continue
            if written and label in text:
                contradictions.append(label)
    return list(dict.fromkeys(contradictions))


def validate_public_response(
    case: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    expected_missing = [str(item) for item in case.get("expected_missing_fields") or []]
    missing_hits = {
        item: _missing_field_present(item, response) for item in expected_missing
    }
    reply_components = _reply_components(response, case)
    files = response.get("files")
    files = files if isinstance(files, dict) else {}
    return {
        "scenario_match": response.get("scenario") == case.get("scenario"),
        "industry_match": response.get("industry") == case.get("industry"),
        "docx_present": bool(files.get("docx")),
        "reply_present": bool(str(response.get("reply_text") or "").strip()),
        "expected_missing": missing_hits,
        "reply_components": reply_components,
        "reported_but_written": _reported_but_written(response),
    }


def run_case(
    *,
    base_url: str,
    case: dict[str, Any],
    annotation: dict[str, Any],
    timeout: int,
    max_attempts: int,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    response: dict[str, Any] | None = None
    final_status: int | None = None
    for attempt in range(1, max_attempts + 1):
        status, data, error, elapsed = _request_once(base_url, case, timeout)
        attempts.append({
            "attempt": attempt,
            "status": status,
            "elapsed_s": round(elapsed, 3),
            "error": error,
        })
        final_status = status
        if data is not None and status is not None and 200 <= status < 300:
            response = data
            break
        if attempt < max_attempts:
            time.sleep(5)

    row: dict[str, Any] = {
        "id": case["id"],
        "scenario": case.get("scenario"),
        "industry": case.get("industry"),
        "input_profile": case.get("input_profile"),
        "request_ok": response is not None,
        "status": final_status,
        "elapsed_s": round(sum(item["elapsed_s"] for item in attempts), 3),
        "attempts": attempts,
        "audit_ok": False,
        "audit_error": "",
    }
    if response is None:
        return row

    # Preserve the raw response for reproducibility. Candidate self-scores are
    # retained only as uninterpreted raw evidence and never read by summaries.
    row["raw"] = response
    row["response_contract"] = validate_public_response(case, response)
    try:
        row["external_audit"] = audit_response(response, annotation)
        row["audit_ok"] = True
    except Exception as exc:
        row["audit_error"] = f"{type(exc).__name__}:{exc}"
    try:
        row["generation_quality"] = assess_generation_quality(
            case, annotation, response,
        )
        row["generation_quality_error"] = ""
    except Exception as exc:
        row["generation_quality"] = {}
        row["generation_quality_error"] = f"{type(exc).__name__}:{exc}"
    return row


def _ratio(numerator: int | float, denominator: int | float, empty: float = 1.0) -> float:
    return round(float(numerator) / float(denominator), 4) if denominator else empty


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 4) if items else 0.0


def _percentile(values: Iterable[float], percentile: float) -> float:
    items = sorted(float(value) for value in values)
    if not items:
        return 0.0
    if len(items) == 1:
        return round(items[0], 3)
    index = (len(items) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(items[lower], 3)
    weight = index - lower
    return round(items[lower] * (1 - weight) + items[upper] * weight, 3)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    request_ok = [row for row in rows if row.get("request_ok")]
    audited = [row for row in request_ok if row.get("audit_ok")]
    atomic = [row["external_audit"]["atomic_factuality"] for row in audited]
    ownership = [row["external_audit"]["ownership_integrity"] for row in audited]

    generated = sum(int(item["generated_atom_count"]) for item in atomic)
    supported = sum(int(item["supported_atom_count"]) for item in atomic)
    source_facts = sum(int(item["source_fact_count"]) for item in atomic)
    represented = sum(int(item["represented_source_fact_count"]) for item in atomic)
    source_cases = [item for item in atomic if int(item["source_fact_count"]) > 0]

    correct = sum(int(item["correct_assignment_count"]) for item in ownership)
    incorrect = sum(int(item["incorrect_assignment_count"]) for item in ownership)
    undetermined = sum(int(item["undetermined_assignment_count"]) for item in ownership)

    structural = {
        category: {
            "added_count": sum(
                int(row["external_audit"]["structural_invariants"][category]["added_count"])
                for row in audited
            ),
            "missing_count": sum(
                int(row["external_audit"]["structural_invariants"][category]["missing_count"])
                for row in audited
            ),
        }
        for category in ALL_STRUCTURAL_CATEGORIES
    }
    critical_additions = sum(
        structural[category]["added_count"]
        for category in CRITICAL_STRUCTURAL_CATEGORIES
    )

    contracts = [row.get("response_contract") or {} for row in request_ok]
    missing_total = sum(len(item.get("expected_missing") or {}) for item in contracts)
    missing_hits = sum(
        sum(bool(value) for value in (item.get("expected_missing") or {}).values())
        for item in contracts
    )
    reply_total = sum(len(item.get("reply_components") or {}) for item in contracts)
    reply_hits = sum(
        sum(bool(value) for value in (item.get("reply_components") or {}).values())
        for item in contracts
    )
    reply_by_component = {
        component: {
            "required": sum(component in (item.get("reply_components") or {}) for item in contracts),
            "present": sum(bool((item.get("reply_components") or {}).get(component)) for item in contracts),
        }
        for component in REPLY_COMPONENTS
    }

    quality_rows = [
        row.get("generation_quality") or {}
        for row in rows
        if row.get("generation_quality")
    ]
    bullet_rows = [item.get("bullets") or {} for item in quality_rows]
    alignment_rows = [item.get("job_alignment") or {} for item in quality_rows]
    reply_detail_rows = [item.get("reply_detail") or {} for item in quality_rows]
    quality_bullet_count = sum(int(item.get("count") or 0) for item in bullet_rows)
    quality_star_count = sum(int(item.get("star_complete_count") or 0) for item in bullet_rows)
    quality_two_dim_count = sum(
        round(float(item.get("two_or_more_dimension_rate") or 0.0) * int(item.get("count") or 0))
        for item in bullet_rows
    )
    quality_compact_count = sum(
        round(float(item.get("compact_bullet_rate") or 0.0) * int(item.get("count") or 0))
        for item in bullet_rows
    )
    requirement_count = sum(int(item.get("requirement_count") or 0) for item in alignment_rows)
    supported_count = sum(int(item.get("supported_count") or 0) for item in alignment_rows)
    partial_count = sum(int(item.get("partial_count") or 0) for item in alignment_rows)
    alignment_available_count = sum(bool(item.get("available")) for item in alignment_rows)

    return {
        "case_count": len(rows),
        "request_success_count": len(request_ok),
        "request_failure_count": len(rows) - len(request_ok),
        "audit_success_count": len(audited),
        "audit_failure_count": len(request_ok) - len(audited),
        "request_failure_rate": _ratio(len(rows) - len(request_ok), len(rows), 0.0),
        "latency_seconds": {
            "mean": _mean(float(row.get("elapsed_s") or 0) for row in rows),
            "p95": _percentile((float(row.get("elapsed_s") or 0) for row in rows), 0.95),
            "max": round(max((float(row.get("elapsed_s") or 0) for row in rows), default=0.0), 3),
        },
        "atomic_factuality": {
            "generated_atom_count": generated,
            "supported_atom_count": supported,
            "unsupported_atom_count": generated - supported,
            "micro_precision": _ratio(supported, generated),
            "macro_precision": _mean(float(item["precision"]) for item in atomic),
            "source_fact_count": source_facts,
            "represented_source_fact_count": represented,
            "unrepresented_source_fact_count": source_facts - represented,
            "micro_recall": _ratio(represented, source_facts),
            "macro_recall_source_cases": _mean(float(item["recall"]) for item in source_cases),
        },
        "ownership_integrity": {
            "correct_assignment_count": correct,
            "incorrect_assignment_count": incorrect,
            "undetermined_assignment_count": undetermined,
            "integrity_rate": _ratio(correct, correct + incorrect),
        },
        "structural_invariants": structural,
        "critical_additions": critical_additions,
        "response_contract": {
            "scenario_mismatch_count": sum(not item.get("scenario_match", False) for item in contracts),
            "industry_mismatch_count": sum(not item.get("industry_match", False) for item in contracts),
            "docx_missing_count": sum(not item.get("docx_present", False) for item in contracts),
            "reply_missing_count": sum(not item.get("reply_present", False) for item in contracts),
            "expected_missing_coverage": _ratio(missing_hits, missing_total),
            "expected_missing_hit_count": missing_hits,
            "expected_missing_required_count": missing_total,
            "reply_component_coverage": _ratio(reply_hits, reply_total),
            "reply_component_hit_count": reply_hits,
            "reply_component_required_count": reply_total,
            "reply_by_component": reply_by_component,
            "reported_but_written_count": sum(
                len(item.get("reported_but_written") or []) for item in contracts
            ),
        },
        "generation_quality": {
            "proxy_version": "generation-quality-proxy-1.0",
            "human_review_required": True,
            "quality_case_count": len(quality_rows),
            "bullets": {
                "count": quality_bullet_count,
                "avg_chars": _mean(
                    float(item.get("avg_chars") or 0.0) for item in bullet_rows
                ),
                "star_complete_rate": _ratio(quality_star_count, quality_bullet_count),
                "two_or_more_dimension_rate": _ratio(
                    quality_two_dim_count, quality_bullet_count,
                ),
                "compact_bullet_rate": _ratio(
                    quality_compact_count, quality_bullet_count, 0.0,
                ),
            },
            "job_alignment": {
                "cases_with_jd": alignment_available_count,
                "requirement_count": requirement_count,
                "supported_count": supported_count,
                "partial_count": partial_count,
                "support_rate": _ratio(
                    supported_count + 0.5 * partial_count,
                    requirement_count,
                    0.0,
                ) if requirement_count else None,
            },
            "reply_detail": {
                "avg_chars": _mean(
                    float(item.get("chars") or 0.0) for item in reply_detail_rows
                ),
                "avg_missing_field_count": _mean(
                    float(item.get("missing_field_count") or 0.0)
                    for item in reply_detail_rows
                ),
                "avg_recommendation_count": _mean(
                    float(item.get("recommendation_count") or 0.0)
                    for item in reply_detail_rows
                ),
            },
        },
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _aggregate(rows)
    groups: dict[str, dict[str, Any]] = {}
    for key in ("scenario", "industry", "input_profile"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(key) or "unknown")].append(row)
        groups[key] = {
            name: _aggregate(group_rows) for name, group_rows in sorted(grouped.items())
        }
    summary["groups"] = groups
    return summary


def _evaluator_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        CORE_ROOT / "atomic_fact_audit.py",
        CORE_ROOT / "evidence_binding.py",
        CORE_ROOT / "source_adapter.py",
        CORE_ROOT / "resume_copilot_service.py",
    )
    return {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in paths}


def _select_cases(
    cases: list[dict[str, Any]],
    requested: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if requested:
        wanted = [item for value in requested for item in value.split(",") if item]
        by_id = {str(case["id"]): case for case in cases}
        missing = [item for item in wanted if item not in by_id]
        if missing:
            raise ValueError(f"unknown case IDs: {', '.join(missing)}")
        cases = [by_id[item] for item in wanted]
    return cases[:limit] if limit else cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default=str(ROOT / "holdout_v2/cases.jsonl"))
    parser.add_argument("--annotations", default=str(ROOT / "holdout_v2/annotations.jsonl"))
    parser.add_argument("--timeout", type=int, default=500)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()

    cases_path = Path(args.cases).resolve()
    annotations_path = Path(args.annotations).resolve()
    # The sealed shadow split must never be selected by this routine.
    if "shadow_v3" in cases_path.parts or "shadow_v3" in annotations_path.parts:
        raise SystemExit("shadow_v3 is sealed and cannot be evaluated by this command")

    all_cases = _load_jsonl(cases_path)
    cases = _select_cases(all_cases, args.case_id, args.limit)
    annotations = {item["case_id"]: item for item in _load_jsonl(annotations_path)}
    if any(case["id"] not in annotations for case in cases):
        raise SystemExit("one or more selected cases have no annotation")

    out_path = Path(args.out).resolve()
    signature = {
        "evaluator_version": EVALUATOR_VERSION,
        "version": args.version,
        "image_digest": args.image_digest,
        "base_url": args.base_url,
        "cases_sha256": _sha256(cases_path),
        "annotations_sha256": _sha256(annotations_path),
        "selected_case_ids": [case["id"] for case in cases],
        "evaluator_hashes": _evaluator_hashes(),
        "candidate_self_score_ignored": True,
        "shadow_v3_sealed": True,
    }
    payload: dict[str, Any]
    if out_path.is_file():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if payload.get("metadata") != signature:
            raise SystemExit("existing output metadata differs; choose a new output path")
    else:
        payload = {"metadata": signature, "summary": {}, "rows": []}

    existing = {str(row["id"]): row for row in payload.get("rows", [])}
    for index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        if case_id in existing:
            print(f"[{index}/{len(cases)}] {case_id}: resumed", flush=True)
            continue
        print(f"[{index}/{len(cases)}] {case_id}: running", flush=True)
        row = run_case(
            base_url=args.base_url,
            case=case,
            annotation=annotations[case_id],
            timeout=args.timeout,
            max_attempts=max(1, args.max_attempts),
        )
        existing[case_id] = row
        payload["rows"] = [
            existing[item["id"]] for item in cases if item["id"] in existing
        ]
        payload["summary"] = summarize_rows(payload["rows"])
        _atomic_write_json(out_path, payload)
        atom = (row.get("external_audit") or {}).get("atomic_factuality") or {}
        print(
            f"[{index}/{len(cases)}] {case_id}: "
            f"request={row['request_ok']} audit={row['audit_ok']} "
            f"precision={atom.get('precision', '-')} recall={atom.get('recall', '-')} "
            f"elapsed={row['elapsed_s']}s",
            flush=True,
        )

    payload["rows"] = [existing[case["id"]] for case in cases]
    payload["summary"] = summarize_rows(payload["rows"])
    _atomic_write_json(out_path, payload)
    aggregate = copy.deepcopy(payload["summary"])
    aggregate.pop("groups", None)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
