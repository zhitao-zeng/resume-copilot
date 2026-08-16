#!/usr/bin/env python3
"""Darvin-aligned component evaluator (r3).

Exposes the 15 scored Darvin subdimensions as per-case component signals with
explicit applicability and non-applicable weight redistribution.  Safety and
reliability live in a separate pass/fail gate and never enter candidate
ranking once passed.  This module never emits a synthetic Darvin total:
component signals stay component signals until a calibrated judge exists.

Evidence tiers:
- ``json``: measurable from the public response + source ledger (this module).
- ``rendered_docx``: requires the rendered document gate; reported as
  unmeasured here rather than scored from JSON proxies (the platform rubric
  reserves visual layout and template fidelity for rendered evidence).

All scorers are deterministic proxies; ``blinded_human_review_required`` stays
true.  Fact safety is only read from the frozen external audit, never
re-derived here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

EVALUATOR_VERSION = "darvin-component-evaluator-r3"
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
CORE_ROOT = REPO_ROOT / "core"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate  # noqa: E402
from evidence_binding import bind_resume_evidence  # noqa: E402
from quality_report import measure_source_coverage, source_fact_units  # noqa: E402
from resume_copilot_service import _canonical_resume_from_render_data  # noqa: E402

try:  # rendered-document gate is optional evidence (DOCX may be absent)
    from rendered_audit import (
        audit_rendered_docx,
        template_fidelity_score01,
        visual_layout_score01,
    )
except ImportError:  # pragma: no cover - python-docx missing in minimal envs
    audit_rendered_docx = None


# ---------------------------------------------------------------------------
# Rubric definition (frozen in metric-contract.md, 2026-08-16)
# ---------------------------------------------------------------------------

# (component, subdimension, platform weight, evidence tier)
SUBDIMENSIONS: tuple[tuple[str, str, int, str], ...] = (
    ("readability", "structure_clarity", 4, "json"),
    ("readability", "visual_layout", 4, "rendered_docx"),
    ("readability", "template_fidelity", 2, "rendered_docx"),
    ("completeness", "profile", 5, "json"),
    ("completeness", "experience", 10, "json"),
    ("completeness", "education", 5, "json"),
    ("completeness", "summary", 5, "json"),
    ("completeness", "skills", 5, "json"),
    ("expression", "professional_writing", 10, "json"),
    ("expression", "star_richness", 20, "json"),
    ("expression", "ability_emphasis", 10, "json"),
    ("reply", "generation_direction", 2, "json"),
    ("reply", "jd_analysis", 8, "json"),
    ("reply", "missing_info_advice", 5, "json"),
    ("reply", "conflict_advice", 5, "json"),
)
COMPONENT_ORDER = ("readability", "completeness", "expression", "reply")
COMPONENT_WEIGHTS = {"readability": 10, "completeness": 30, "expression": 40, "reply": 20}

# Experience-family sections from source_fact_units(section_hint=...).
_EXPERIENCE_SECTIONS = {"experience", "projects", "volunteer"}
_EDUCATION_SECTIONS = {"education"}
_SKILL_SECTIONS = {"skills", "certifications", "language", "languages"}
_PROFILE_SECTIONS = {"profile", "contact", "personal", "header"}

_PURE_LABEL = re.compile(r"^[^：:]{1,15}[：:]\s*$")
_SEPARATOR_ARTIFACT = re.compile(r"__+|——{2,}")
_YEAR_RANGE = re.compile(r"\d{4}\s*[年./-]")
_REPLY_CONCISE_MAX = 600
_REPLY_OVERLONG = 900


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(handle.name, path)


def _ratio(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return round(num / den, 4)


def _case_has_jd(case: dict[str, Any]) -> bool:
    return bool(case.get("target_jd") or case.get("target_jd_file_path"))


def _case_has_cv(case: dict[str, Any]) -> bool:
    return bool(case.get("cv_path"))


# ---------------------------------------------------------------------------
# Source coverage per section family
# ---------------------------------------------------------------------------

def _section_coverage(
    case: dict[str, Any],
    annotation: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    """Recompute deterministic per-section source coverage (no LLM)."""

    source = evaluate.build_annotated_source(annotation)
    resume_data = raw.get("resume_data")
    resume_data = resume_data if isinstance(resume_data, dict) else {}
    resume = _canonical_resume_from_render_data(resume_data)
    bindings = bind_resume_evidence(resume, source)
    units = source_fact_units(source)
    _, missing_ids = measure_source_coverage(source, bindings, allow_distributed=True)
    missing = set(missing_ids)

    families: dict[str, dict[str, int]] = {
        "experience": {"total": 0, "represented": 0},
        "education": {"total": 0, "represented": 0},
        "skills": {"total": 0, "represented": 0},
        "profile": {"total": 0, "represented": 0},
    }
    section_map = (
        ("experience", _EXPERIENCE_SECTIONS),
        ("education", _EDUCATION_SECTIONS),
        ("skills", _SKILL_SECTIONS),
        ("profile", _PROFILE_SECTIONS),
    )
    for unit in units:
        hint = (unit.get("section_hint") or "").casefold()
        for family, hints in section_map:
            if hint in hints:
                families[family]["total"] += 1
                if unit["unit_id"] not in missing:
                    families[family]["represented"] += 1
                break
    return {
        family: {
            **counts,
            "recall": _ratio(counts["represented"], counts["total"]),
        }
        for family, counts in families.items()
    }


# ---------------------------------------------------------------------------
# Per-case subdimension scorers (deterministic proxies)
# ---------------------------------------------------------------------------

def _sub(component: str, name: str, weight: int, tier: str) -> dict[str, Any]:
    return {
        "component": component,
        "subdimension": name,
        "weight": weight,
        "evidence_tier": tier,
        "applicable": True,
        "applicability_reason": "",
        "measurable": tier == "json",
        "score01": None,
        "evidence": {},
    }


def _na(entry: dict[str, Any], reason: str) -> dict[str, Any]:
    entry["applicable"] = False
    entry["applicability_reason"] = reason
    entry["measurable"] = False
    return entry


def assess_case_components(
    case: dict[str, Any],
    annotation: dict[str, Any],
    raw: dict[str, Any],
    audit: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Score all 15 subdimensions for one case; never raises on odd input."""

    subs: dict[str, dict[str, Any]] = {
        name: _sub(component, name, weight, tier)
        for component, name, weight, tier in SUBDIMENSIONS
    }
    resume_data = raw.get("resume_data")
    resume_data = resume_data if isinstance(resume_data, dict) else {}
    framework_mode = bool(resume_data.get("framework"))
    has_jd = _case_has_jd(case)
    has_cv = _case_has_cv(case)
    bullets_info = quality.get("bullets") or {}
    bullet_details = bullets_info.get("details") or []
    reply_text = str(raw.get("reply_text") or "")
    reply_components = evaluate._reply_components(raw, case)
    expected_missing = [str(x) for x in (case.get("expected_missing_fields") or [])]
    reported_missing = raw.get("missing_fields")
    reported_missing = [str(x) for x in reported_missing] if isinstance(reported_missing, list) else []
    reported_but_written = evaluate._reported_but_written(raw)
    expected_conflicts = case.get("expected_conflicts") or []

    coverage: dict[str, Any] = {}
    coverage_error = ""
    try:
        coverage = _section_coverage(case, annotation, raw)
    except Exception as exc:  # keep the row comparable; gate fails separately
        coverage_error = f"{type(exc).__name__}:{exc}"

    # ---- readability -----------------------------------------------------
    # evaluator 1.2 bullet details carry no raw text; scan the resume payload
    # directly for label remnants and separator artifacts.
    resume_text = json.dumps(resume_data, ensure_ascii=False)
    separator_artifacts = len(_SEPARATOR_ARTIFACT.findall(resume_text))
    bullet_texts = evaluate._resume_bullets(resume_data)
    label_remnants = sum(1 for text in bullet_texts if _PURE_LABEL.match(text.strip()))
    artifact_issues = label_remnants + separator_artifacts
    entry = subs["structure_clarity"]
    issue_rate = artifact_issues / max(1, len(bullet_texts))
    entry["score01"] = round(max(0.0, 1.0 - 2.0 * issue_rate), 4)
    entry["evidence"] = {
        "label_remnants": label_remnants,
        "separator_artifacts": separator_artifacts,
        "bullet_count": len(bullet_texts),
    }

    for name in ("visual_layout", "template_fidelity"):
        entry = subs[name]
        entry["measurable"] = False
        entry["applicability_reason"] = (
            "rendered-docx evidence required; JSON cannot establish this subdimension"
        )
        entry["evidence"] = {"docx_present": bool(raw.get("docx_base64") or raw.get("docx_url"))}
    # Rendered-document gate: when the immutable row carries a DOCX path and
    # the artifact still exists, visual_layout and template_fidelity become
    # measurable rendered-tier signals instead of unmeasured placeholders.
    files = raw.get("files") if isinstance(raw.get("files"), dict) else {}
    docx_path = str(files.get("docx") or "")
    if docx_path and audit_rendered_docx is not None and Path(docx_path).is_file():
        rendered = audit_rendered_docx(docx_path, resume_data=resume_data)
        visual = visual_layout_score01(rendered)
        fidelity = template_fidelity_score01(rendered)
        if visual is not None:
            entry = subs["visual_layout"]
            entry["measurable"] = True
            entry["score01"] = visual
            entry["applicability_reason"] = ""
            entry["evidence"] = {
                "rendered_gate": rendered.get("gate_version"),
                "label_remnants": len(rendered.get("label_remnants") or []),
                "separator_artifacts": len(rendered.get("separator_artifacts") or []),
                "sparse_trailing": rendered.get("sparse_trailing"),
                "fact_retention": rendered.get("fact_retention"),
            }
        if fidelity is not None:
            entry = subs["template_fidelity"]
            entry["measurable"] = True
            entry["score01"] = fidelity
            entry["applicability_reason"] = ""
            entry["evidence"] = {
                "rendered_gate": rendered.get("gate_version"),
                "template_mode": rendered.get("template_mode"),
                "leftover_tags": len(rendered.get("leftover_tags") or []),
            }

    # ---- completeness ----------------------------------------------------
    if coverage_error:
        for name in ("profile", "experience", "education", "skills"):
            _na(subs[name], f"source coverage unavailable: {coverage_error}")
    else:
        entry = subs["profile"]
        profile_cov = coverage["profile"]
        if has_cv and profile_cov["total"] > 0:
            entry["score01"] = profile_cov["recall"]
            entry["evidence"] = dict(profile_cov)
        elif has_cv:
            # cv supplies no explicit profile units; check meta fields instead
            meta = resume_data.get("meta") if isinstance(resume_data.get("meta"), dict) else {}
            filled = sum(1 for value in meta.values() if str(value or "").strip())
            entry["score01"] = 1.0 if filled > 0 else 0.0
            entry["evidence"] = {"meta_filled_fields": filled, "proxy": "meta_fields"}
        else:
            _na(entry, "no cv supplied; profile absence is scored under missing_info_advice")

        entry = subs["experience"]
        exp_cov = coverage["experience"]
        if exp_cov["total"] > 0:
            structural = (audit.get("structural_invariants") or {})
            org_miss = int((structural.get("organization") or {}).get("missing_count") or 0)
            role_miss = int((structural.get("role") or {}).get("missing_count") or 0)
            period_miss = int((structural.get("period") or {}).get("missing_count") or 0)
            ownership = audit.get("ownership_integrity") or {}
            incorrect = int(ownership.get("incorrect_assignment_count") or 0)
            header_issues = org_miss + role_miss + period_miss
            header_integrity = max(0.0, 1.0 - header_issues / max(1, exp_cov["total"]))
            score = 0.6 * float(exp_cov["recall"] or 0.0) + 0.4 * header_integrity
            score = max(0.0, score - 0.05 * incorrect)
            entry["score01"] = round(score, 4)
            entry["evidence"] = {
                **exp_cov,
                "org_missing": org_miss,
                "role_missing": role_miss,
                "period_missing": period_miss,
                "incorrect_ownership": incorrect,
            }
        elif not has_cv:
            _na(entry, "no cv supplied; experience absence is scored under missing_info_advice")
        else:
            _na(entry, "source contains no experience units")

        entry = subs["education"]
        edu_cov = coverage["education"]
        if edu_cov["total"] > 0:
            structural = (audit.get("structural_invariants") or {})
            edu_miss = int((structural.get("education") or {}).get("missing_count") or 0)
            integrity = max(0.0, 1.0 - edu_miss / max(1, edu_cov["total"]))
            entry["score01"] = round(0.7 * float(edu_cov["recall"] or 0.0) + 0.3 * integrity, 4)
            entry["evidence"] = {**edu_cov, "education_missing": edu_miss}
        elif not has_cv:
            _na(entry, "no cv supplied; education absence is scored under missing_info_advice")
        else:
            _na(entry, "source contains no education units")

        entry = subs["skills"]
        skill_cov = coverage["skills"]
        if skill_cov["total"] > 0:
            skills_block = resume_data.get("skills")
            categorized = bool(
                isinstance(skills_block, dict) and any(skills_block.values())
            ) or bool(isinstance(skills_block, list) and skills_block)
            entry["score01"] = round(
                0.7 * float(skill_cov["recall"] or 0.0) + 0.3 * (1.0 if categorized else 0.0), 4
            )
            entry["evidence"] = {**skill_cov, "categorized": categorized}
        elif not has_cv:
            _na(entry, "no cv supplied; skills absence is scored under missing_info_advice")
        else:
            _na(entry, "source contains no skill units")

    entry = subs["summary"]
    if framework_mode:
        _na(entry, "framework mode; no summary expected")
    else:
        summary_text = str(resume_data.get("summary") or (resume_data.get("meta") or {}).get("summary") or "")
        summary_text = summary_text.strip()
        present = bool(summary_text)
        compact_chars = len(re.sub(r"\s+", "", summary_text))
        within_limit = present and compact_chars <= 100
        segments = [seg for seg in re.split(r"[，；;]", summary_text) if seg.strip()]
        timeline_like = sum(1 for seg in segments if _YEAR_RANGE.search(seg))
        concatenated = timeline_like >= 2 or (
            len(segments) >= 4 and compact_chars > 100
        )
        checks = [present, within_limit, present and not concatenated]
        entry["score01"] = round(sum(checks) / len(checks), 4)
        entry["evidence"] = {
            "present": present,
            "chars": compact_chars,
            "within_100_chars": within_limit,
            "timeline_concatenation": concatenated,
        }

    # ---- expression ------------------------------------------------------
    compact_rate = bullets_info.get("compact_bullet_rate")
    star_rate = bullets_info.get("star_complete_rate")
    two_dim_rate = bullets_info.get("two_or_more_dimension_rate")
    if not bullet_details:
        for name in ("professional_writing", "star_richness", "ability_emphasis"):
            _na(subs[name], "no experience bullets in response")
    else:
        entry = subs["professional_writing"]
        entry["score01"] = round(max(0.0, 1.0 - float(compact_rate or 0.0)), 4)
        entry["evidence"] = {
            "compact_bullet_rate": compact_rate,
            "avg_chars": bullets_info.get("avg_chars"),
            "bullet_count": bullets_info.get("count"),
        }

        entry = subs["star_richness"]
        entry["score01"] = round(
            0.5 * float(star_rate or 0.0) + 0.5 * float(two_dim_rate or 0.0), 4
        )
        entry["evidence"] = {
            "star_complete_rate": star_rate,
            "two_or_more_dimension_rate": two_dim_rate,
        }

        entry = subs["ability_emphasis"]
        alignment = quality.get("job_alignment") or {}
        if has_jd and alignment.get("available") and alignment.get("support_rate") is not None:
            entry["score01"] = float(alignment["support_rate"])
            entry["evidence"] = {"proxy": "jd_support_rate", **{
                k: alignment.get(k) for k in ("requirement_count", "supported_count", "partial_count", "missing_count")
            }}
        else:
            result_or_quantified = sum(
                1 for item in bullet_details if item.get("result")
            )
            rate = result_or_quantified / max(1, len(bullet_details))
            entry["score01"] = round(rate, 4)
            entry["evidence"] = {"proxy": "result_or_quantified_bullet_rate", "rate": round(rate, 4)}

    # ---- reply ------------------------------------------------------------
    reply_chars = len(reply_text.strip())
    entry = subs["generation_direction"]
    direction_present = bool(reply_components.get("生成方向"))
    if reply_chars <= _REPLY_CONCISE_MAX:
        conciseness = 1.0
    elif reply_chars <= _REPLY_OVERLONG:
        conciseness = 0.5
    else:
        conciseness = 0.25
    entry["score01"] = round((1.0 if direction_present else 0.0) * conciseness, 4)
    entry["evidence"] = {
        "direction_present": direction_present,
        "reply_chars": reply_chars,
        "conciseness": conciseness,
    }

    entry = subs["jd_analysis"]
    if not has_jd:
        _na(entry, "no target JD supplied for this case")
    else:
        alignment = quality.get("job_alignment") or {}
        reply_detail = quality.get("reply_detail") or {}
        has_requirements = int(alignment.get("requirement_count") or 0) > 0
        has_recommendations = int(reply_detail.get("recommendation_count") or 0) > 0
        advice_present = bool(reply_components.get("岗位建议"))
        support_rate = alignment.get("support_rate")
        entry["score01"] = round(
            0.3 * (1.0 if has_requirements else 0.0)
            + 0.2 * (1.0 if has_recommendations else 0.0)
            + 0.2 * (1.0 if advice_present else 0.0)
            + 0.3 * float(support_rate if support_rate is not None else 0.0),
            4,
        )
        entry["evidence"] = {
            "requirement_count": alignment.get("requirement_count"),
            "recommendation_count": reply_detail.get("recommendation_count"),
            "job_advice_component": advice_present,
            "support_rate": support_rate,
        }

    entry = subs["missing_info_advice"]
    if reported_but_written:
        entry["score01"] = 0.0
        entry["evidence"] = {"reported_but_written": reported_but_written}
    elif not expected_missing:
        entry["score01"] = 1.0 if reply_components.get("缺失信息") else 0.5
        entry["evidence"] = {
            "expected_missing_count": 0,
            "reported_missing_count": len(reported_missing),
            "component_present": bool(reply_components.get("缺失信息")),
        }
    else:
        expected_set = set(expected_missing)
        reported_set = set(reported_missing)
        hit = len(expected_set & reported_set)
        precision = hit / max(1, len(reported_set))
        recall = hit / max(1, len(expected_set))
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        entry["score01"] = round(f1, 4)
        entry["evidence"] = {
            "expected_missing": expected_missing,
            "reported_missing": reported_missing,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

    entry = subs["conflict_advice"]
    conflict_component = bool(reply_components.get("冲突检查"))
    if not expected_conflicts:
        entry["score01"] = 1.0 if conflict_component else 0.0
        entry["evidence"] = {
            "expected_conflicts": 0,
            "explicit_no_conflict_statement": conflict_component,
        }
    else:
        reply_blob = reply_text + json.dumps(raw.get("user_report") or {}, ensure_ascii=False)
        hits = sum(
            1 for item in expected_conflicts
            if str(item) and str(item)[:12] in reply_blob
        )
        entry["score01"] = round(hits / max(1, len(expected_conflicts)), 4)
        entry["evidence"] = {
            "expected_conflicts": len(expected_conflicts),
            "matched": hits,
        }

    return subs


# ---------------------------------------------------------------------------
# Layer-1 gate (pass/fail only; never ranked after passing)
# ---------------------------------------------------------------------------

def evaluate_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    ok_rows = [row for row in rows if row.get("request_ok")]
    audit_ok = [row for row in rows if row.get("audit_ok")]
    audits = [row.get("external_audit") or {} for row in audit_ok]
    generated = sum(int((a.get("atomic_factuality") or {}).get("generated_atom_count") or 0) for a in audits)
    supported = sum(int((a.get("atomic_factuality") or {}).get("supported_atom_count") or 0) for a in audits)
    micro_precision = supported / generated if generated else 1.0
    correct = sum(int((a.get("ownership_integrity") or {}).get("correct_assignment_count") or 0) for a in audits)
    incorrect = sum(int((a.get("ownership_integrity") or {}).get("incorrect_assignment_count") or 0) for a in audits)
    undetermined = sum(int((a.get("ownership_integrity") or {}).get("undetermined_assignment_count") or 0) for a in audits)
    # Frozen evaluator 1.2 semantics: undetermined assignments are reported
    # separately and never enter the integrity denominator.
    integrity = correct / (correct + incorrect) if (correct + incorrect) else 1.0
    critical_additions = 0
    for a in audits:
        structural = a.get("structural_invariants") or {}
        critical_additions += sum(
            int((structural.get(cat) or {}).get("added_count") or 0)
            for cat in evaluate.CRITICAL_STRUCTURAL_CATEGORIES
        )
    latencies = [float(row.get("elapsed_s") or 0.0) for row in rows]
    max_latency = max(latencies) if latencies else 0.0
    docx_missing = sum(
        1 for row in ok_rows
        if not bool((row.get("response_contract") or {}).get("docx_present"))
    )

    checks = {
        "request_success": {"pass": len(ok_rows) == total, "value": f"{len(ok_rows)}/{total}"},
        "audit_success": {"pass": len(audit_ok) == total, "value": f"{len(audit_ok)}/{total}"},
        "zero_critical_additions": {"pass": critical_additions == 0, "value": critical_additions},
        "atomic_precision_gte_0.99": {"pass": micro_precision >= 0.99, "value": round(micro_precision, 6)},
        "ownership_integrity_gte_0.98": {"pass": integrity >= 0.98, "value": round(integrity, 6)},
        "max_latency_lt_480s": {"pass": max_latency < 480.0, "value": round(max_latency, 3)},
        "docx_present": {"pass": docx_missing == 0, "value": f"{total - docx_missing}/{total}"},
    }
    return {
        "pass": all(check["pass"] for check in checks.values()),
        "checks": checks,
        "informational": {
            "undetermined_ownership_count": undetermined,
            "note": "undetermined assignments are tracked outside the integrity rate (evaluator 1.2 semantics)",
        },
        "note": "pass/fail only; guardrail decimals are never used for candidate ranking",
    }


# ---------------------------------------------------------------------------
# Aggregation with non-applicable redistribution (component signals only)
# ---------------------------------------------------------------------------

def _aggregate_subdimension(entries: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [e for e in entries if e["applicable"]]
    measured = [e for e in applicable if e["measurable"] and e["score01"] is not None]
    scores = [float(e["score01"]) for e in measured]
    return {
        "cases": len(entries),
        "applicable": len(applicable),
        "measured": len(measured),
        "mean_score01": round(sum(scores) / len(scores), 4) if scores else None,
        "min_score01": round(min(scores), 4) if scores else None,
        "not_applicable_reasons": sorted({
            e["applicability_reason"] for e in entries if not e["applicable"]
        }),
    }


def aggregate_components(
    per_case: list[dict[str, Any]],
    scenario_of: dict[str, str],
) -> dict[str, Any]:
    """Aggregate per-case subdimensions into component signals.

    Non-applicable or unmeasurable subdimension weight is redistributed
    proportionally within its component.  No cross-component total is ever
    produced.
    """

    def aggregate_subset(case_entries: list[dict[str, Any]]) -> dict[str, Any]:
        components: dict[str, Any] = {}
        for component in COMPONENT_ORDER:
            sub_specs = [
                (name, weight, tier)
                for component_name, name, weight, tier in SUBDIMENSIONS
                if component_name == component
            ]
            sub_signals: dict[str, Any] = {}
            for name, weight, tier in sub_specs:
                signal = _aggregate_subdimension(
                    [case_["subdimensions"][name] for case_ in case_entries]
                )
                signal["weight"] = weight
                signal["evidence_tier"] = tier
                sub_signals[name] = signal
            # Per-case component score with within-case redistribution, then
            # average across cases — the rubric skips non-applicable items per
            # case and redistributes their weight inside the component.
            per_case_scores: list[float] = []
            per_case_measured_weight: list[int] = []
            for case_ in case_entries:
                measured_weight = 0
                weighted_sum = 0.0
                for name, weight, _tier in sub_specs:
                    entry = case_["subdimensions"][name]
                    if entry["applicable"] and entry["measurable"] and entry["score01"] is not None:
                        measured_weight += weight
                        weighted_sum += weight * float(entry["score01"])
                if measured_weight > 0:
                    per_case_scores.append(weighted_sum / measured_weight)
                    per_case_measured_weight.append(measured_weight)
            shares: dict[str, float] = {}
            for name, weight, _tier in sub_specs:
                numer = sum(
                    weight
                    for case_ in case_entries
                    if case_["subdimensions"][name]["applicable"]
                    and case_["subdimensions"][name]["measurable"]
                    and case_["subdimensions"][name]["score01"] is not None
                )
                if numer:
                    shares[name] = float(numer)
            share_total = sum(shares.values())
            components[component] = {
                "component_weight": COMPONENT_WEIGHTS[component],
                "score01": round(sum(per_case_scores) / len(per_case_scores), 4)
                if per_case_scores else None,
                "scored_cases": len(per_case_scores),
                "mean_measured_weight": round(
                    sum(per_case_measured_weight) / len(per_case_measured_weight), 2
                ) if per_case_measured_weight else 0,
                "redistributed_subdimension_shares": {
                    name: round(value / share_total, 4) for name, value in shares.items()
                } if share_total else {},
                "subdimensions": sub_signals,
            }
        return components

    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for case_entry in per_case:
        by_scenario.setdefault(scenario_of.get(case_entry["case_id"], "unknown"), []).append(case_entry)

    return {
        "overall": aggregate_subset(per_case),
        "by_scenario": {
            scenario: aggregate_subset(entries)
            for scenario, entries in sorted(by_scenario.items())
        },
        "synthetic_total_emitted": False,
        "note": (
            "component signals only; a locally invented composite must not be "
            "presented as a Darvin score (metric-contract.md)"
        ),
    }


# ---------------------------------------------------------------------------
# Report builder + CLI over immutable re-audit artifacts
# ---------------------------------------------------------------------------

def build_r3_report(
    rows: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    scenario_of: dict[str, str] = {}
    for row in rows:
        case_id = str(row.get("id") or "")
        case = cases.get(case_id) or {}
        scenario_of[case_id] = str(case.get("scenario") or row.get("scenario") or "unknown")
        if not row.get("request_ok") or not isinstance(row.get("raw"), dict):
            per_case.append({
                "case_id": case_id,
                "evaluable": False,
                "reason": "request failed; subdimensions not measurable",
                "subdimensions": {
                    name: _na(_sub(component, name, weight, tier), "request failed")
                    for component, name, weight, tier in SUBDIMENSIONS
                },
            })
            continue
        raw = row["raw"]
        audit = row.get("external_audit") or {}
        quality = row.get("generation_quality") or {}
        if not row.get("audit_ok"):
            per_case.append({
                "case_id": case_id,
                "evaluable": False,
                "reason": f"external audit unavailable: {row.get('audit_error') or 'unknown'}",
                "subdimensions": {
                    name: _na(_sub(component, name, weight, tier), "audit unavailable")
                    for component, name, weight, tier in SUBDIMENSIONS
                },
            })
            continue
        try:
            subs = assess_case_components(case, annotations.get(case_id) or {}, raw, audit, quality)
        except Exception as exc:
            per_case.append({
                "case_id": case_id,
                "evaluable": False,
                "reason": f"component assessment error: {type(exc).__name__}:{exc}",
                "subdimensions": {
                    name: _na(_sub(component, name, weight, tier), "assessment error")
                    for component, name, weight, tier in SUBDIMENSIONS
                },
            })
            continue
        per_case.append({"case_id": case_id, "evaluable": True, "subdimensions": subs})

    return {
        "evaluator_version": EVALUATOR_VERSION,
        "blinded_human_review_required": True,
        "gate": evaluate_gate(rows),
        "components": aggregate_components(per_case, scenario_of),
        "per_case": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="immutable (re)audit artifact with rows[]")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", default=str(evaluate.ROOT / "holdout_v2/cases.jsonl"))
    parser.add_argument("--annotations", default=str(evaluate.ROOT / "holdout_v2/annotations.jsonl"))
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    cases = {item["id"]: item for item in evaluate._load_jsonl(Path(args.cases))}
    annotations = {
        item["case_id"]: item for item in evaluate._load_jsonl(Path(args.annotations))
    }

    report = build_r3_report(payload.get("rows") or [], cases, annotations)
    report["input"] = {
        "path": str(input_path),
        "sha256": _sha256(input_path),
        "source_evaluator_version": (payload.get("metadata") or {}).get("evaluator_version"),
        "inference_reused": True,
    }
    _write_json_atomic(output_path, report)
    gate = report["gate"]
    print(f"gate: {'PASS' if gate['pass'] else 'FAIL'}")
    for name, check in gate["checks"].items():
        print(f"  {name}: {'pass' if check['pass'] else 'FAIL'} ({check['value']})")
    for component, signal in report["components"]["overall"].items():
        print(
            f"{component}: score01={signal['score01']} "
            f"(scored_cases={signal['scored_cases']}, "
            f"mean_measured_weight={signal['mean_measured_weight']}/{signal['component_weight']})"
        )


if __name__ == "__main__":
    main()
