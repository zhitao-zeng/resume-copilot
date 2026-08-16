#!/usr/bin/env python3
"""Tests for the Darvin-aligned component evaluator (r3).

Covers the four holdout scenarios and the Phase 0 failure classes
(truncation-class audit rows, request/audit failures, framework-mode cases,
placeholder-heavy fragmented responses).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_ROOT = REPO_ROOT / "validation_sets" / "public_resume_holdout"
for path in (str(REPO_ROOT), str(REPO_ROOT / "core"), str(HOLDOUT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import darvin_components as dc  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _case(scenario: str, *, cv: bool, jd: bool, expected_missing=None, expected_conflicts=None):
    return {
        "id": f"T-{scenario}",
        "scenario": scenario,
        "cv_path": "x.txt" if cv else "",
        "target_jd": "岗位需要 Python 与数据分析能力。" if jd else "",
        "query": "优化简历",
        "expected_missing_fields": expected_missing or [],
        "expected_conflicts": expected_conflicts or [],
    }


def _audit(*, precision_atoms=(100, 100), ownership=(50, 0, 0), added=0, missing=None):
    generated, supported = precision_atoms
    correct, incorrect, undetermined = ownership
    structural = {
        cat: {"added_count": added if cat == "organization" else 0,
              "missing_count": (missing or {}).get(cat, 0)}
        for cat in ("organization", "role", "period", "education", "credential", "metric", "skill_tool")
    }
    return {
        "atomic_factuality": {
            "generated_atom_count": generated,
            "supported_atom_count": supported,
            "precision": supported / generated if generated else 1.0,
        },
        "ownership_integrity": {
            "correct_assignment_count": correct,
            "incorrect_assignment_count": incorrect,
            "undetermined_assignment_count": undetermined,
            "integrity_rate": correct / max(1, correct + incorrect + undetermined),
        },
        "structural_invariants": structural,
    }


def _quality(*, bullets=10, star=0.5, two_dim=0.6, compact=0.1, jd_available=False,
             support_rate=None, recommendations=2):
    return {
        "bullets": {
            "count": bullets,
            "avg_chars": 25.0,
            "star_complete_rate": star,
            "two_or_more_dimension_rate": two_dim,
            "compact_bullet_rate": compact,
            "details": [{"chars": 25, "action": True, "method": True, "result": True,
                         "dimension_count": 3, "compact": False}] * bullets,
        },
        "job_alignment": {
            "available": jd_available,
            "requirement_count": 3 if jd_available else 0,
            "supported_requirement_count": 2 if jd_available else 0,
            "partial_requirement_count": 0,
            "missing_requirement_count": 1 if jd_available else 0,
            "support_rate": support_rate if jd_available else None,
        },
        "reply_detail": {
            "chars": 300,
            "missing_field_count": 0,
            "recommendation_count": recommendations,
            "component_coverage": 4,
        },
    }


def _raw(*, framework=False, with_summary=True, reply=None):
    resume_data = {
        "meta": {"name": "张三", "phone": "138"},
        "experience": [{"bullets": ["负责数据分析平台搭建，使用 Python 实现 ETL，产出周报系统，覆盖 20 个业务方"]}],
        "projects": [],
        "education": [{"degree": "软件工程学士", "period": "2016-2020"}],
        "skills": {"技术": ["Python", "SQL"]},
        "additional_sections": {},
    }
    if with_summary:
        resume_data["summary"] = "三年数据分析经验，熟悉 Python 与 SQL，主导过报表平台建设。"
    if framework:
        resume_data = {"framework": True, "meta": {}, "experience": [], "projects": [],
                       "education": [], "skills": {}, "additional_sections": {}}
    return {
        "resume_data": resume_data,
        "reply_text": reply if reply is not None else (
            "生成方向：已按目标岗位优化。缺失信息：无。岗位建议：建议补充量化成果。"
            "冲突检查：未发现时间冲突或信息冲突。"
        ),
        "missing_fields": [],
        "user_report": {},
        "docx_base64": "AAAA",
    }


def _row(case_id, *, request_ok=True, audit_ok=True, elapsed=100.0, audit=None, quality=None, raw=None):
    row = {
        "id": case_id,
        "request_ok": request_ok,
        "elapsed_s": elapsed,
        "audit_ok": audit_ok,
        "audit_error": "" if audit_ok else "ContextBudgetError:truncated at 4096 tokens",
        "response_contract": {"docx_present": True},
        "external_audit": audit if audit is not None else _audit(),
        "generation_quality": quality if quality is not None else _quality(),
        "raw": raw if raw is not None else _raw(),
    }
    if not request_ok:
        row.pop("raw", None)
    return row


# ---------------------------------------------------------------------------
# Rubric shape
# ---------------------------------------------------------------------------

def test_rubric_exposes_exactly_15_subdimensions():
    assert len(dc.SUBDIMENSIONS) == 15
    by_component = {}
    for component, name, weight, tier in dc.SUBDIMENSIONS:
        by_component.setdefault(component, []).append((name, weight))
    assert set(by_component) == {"readability", "completeness", "expression", "reply"}
    assert sum(w for _, w in by_component["readability"]) == 10
    assert sum(w for _, w in by_component["completeness"]) == 30
    assert sum(w for _, w in by_component["expression"]) == 40
    assert sum(w for _, w in by_component["reply"]) == 20


def test_rendered_docx_tier_is_exposed_but_not_json_scored():
    raw = _raw()
    subs = dc.assess_case_components(
        _case("scenario1", cv=True, jd=True), {"sources": []}, raw, _audit(), _quality(),
    )
    for name in ("visual_layout", "template_fidelity"):
        entry = subs[name]
        assert entry["evidence_tier"] == "rendered_docx"
        assert entry["measurable"] is False
        assert entry["score01"] is None
        assert "rendered" in entry["applicability_reason"]


# ---------------------------------------------------------------------------
# Applicability across the four scenarios
# ---------------------------------------------------------------------------

def _subs_for(scenario, *, cv, jd, raw=None, quality=None):
    return dc.assess_case_components(
        _case(scenario, cv=cv, jd=jd), {"sources": []},
        raw if raw is not None else _raw(),
        _audit(), quality if quality is not None else _quality(jd_available=jd, support_rate=0.67 if jd else None),
    )


def test_scenario1_cv_plus_jd_all_reply_dimensions_applicable():
    subs = _subs_for("scenario1", cv=True, jd=True)
    assert subs["jd_analysis"]["applicable"] is True
    assert subs["jd_analysis"]["score01"] is not None and subs["jd_analysis"]["score01"] > 0


def test_scenario2_query_only_marks_source_sections_not_applicable():
    subs = _subs_for("scenario2", cv=False, jd=False)
    for name in ("profile", "experience", "education", "skills"):
        assert subs[name]["applicable"] is False
        assert "no cv" in subs[name]["applicability_reason"]
    assert subs["jd_analysis"]["applicable"] is False


def test_scenario3_cv_without_jd_keeps_completeness_applicable():
    subs = _subs_for("scenario3", cv=True, jd=False)
    assert subs["jd_analysis"]["applicable"] is False
    # cv 存在但 annotation 无 source units 时按 no-units 原因标记 N/A，不得崩溃
    assert subs["experience"]["applicable"] is False
    assert "no experience units" in subs["experience"]["applicability_reason"]


def test_scenario4_framework_mode_marks_summary_not_applicable():
    subs = _subs_for("scenario4", cv=False, jd=True, raw=_raw(framework=True))
    assert subs["summary"]["applicable"] is False
    assert "framework" in subs["summary"]["applicability_reason"]
    assert subs["jd_analysis"]["applicable"] is True


# ---------------------------------------------------------------------------
# Gate (pass/fail only)
# ---------------------------------------------------------------------------

def test_gate_passes_clean_rows():
    rows = [_row(f"C{i}") for i in range(3)]
    gate = dc.evaluate_gate(rows)
    assert gate["pass"] is True
    assert all(check["pass"] for check in gate["checks"].values())


def test_gate_fails_on_critical_additions():
    rows = [_row("C0", audit=_audit(added=1))]
    gate = dc.evaluate_gate(rows)
    assert gate["pass"] is False
    assert gate["checks"]["zero_critical_additions"]["pass"] is False


def test_gate_fails_on_precision_below_threshold():
    rows = [_row("C0", audit=_audit(precision_atoms=(1000, 985)))]
    gate = dc.evaluate_gate(rows)
    assert gate["checks"]["atomic_precision_gte_0.99"]["pass"] is False


def test_gate_fails_on_ownership_below_threshold():
    rows = [_row("C0", audit=_audit(ownership=(90, 5, 10)))]
    gate = dc.evaluate_gate(rows)
    assert gate["checks"]["ownership_integrity_gte_0.98"]["pass"] is False


def test_gate_fails_on_latency_and_request_failure():
    rows = [_row("C0", elapsed=481.0), _row("C1", request_ok=False)]
    gate = dc.evaluate_gate(rows)
    assert gate["checks"]["max_latency_lt_480s"]["pass"] is False
    assert gate["checks"]["request_success"]["pass"] is False


def test_gate_fails_on_audit_error_rows():
    # Phase 0 truncation class: audit could not complete on the row.
    rows = [_row("C0", audit_ok=False)]
    gate = dc.evaluate_gate(rows)
    assert gate["checks"]["audit_success"]["pass"] is False


# ---------------------------------------------------------------------------
# Aggregation: redistribution and the no-total invariant
# ---------------------------------------------------------------------------

def _per_case_entries():
    entries = []
    specs = [
        ("S1", "scenario1", dict(cv=True, jd=True)),
        ("S2", "scenario2", dict(cv=False, jd=False)),
        ("S3", "scenario3", dict(cv=True, jd=False)),
        ("S4", "scenario4", dict(cv=False, jd=True)),
    ]
    for case_id, scenario, flags in specs:
        case = _case(scenario, **flags)
        subs = dc.assess_case_components(case, {"sources": []}, _raw(), _audit(), _quality())
        entries.append({"case_id": case_id, "evaluable": True, "subdimensions": subs})
    return entries, {cid: sc for cid, sc, _ in specs}


def test_aggregation_redistributes_weight_within_component_only():
    entries, scenario_of = _per_case_entries()
    agg = dc.aggregate_components(entries, scenario_of)
    reply = agg["overall"]["reply"]
    # jd_analysis 在 scenario2/3 不适用 → 平均已测权重 < 组件权重
    assert reply["mean_measured_weight"] < reply["component_weight"]
    shares = reply["redistributed_subdimension_shares"]
    assert abs(sum(shares.values()) - 1.0) < 1e-2
    # readability 的 rendered-docx 子维度不可测 → 权重重分给 structure_clarity
    readability = agg["overall"]["readability"]
    assert readability["mean_measured_weight"] == 4
    assert readability["redistributed_subdimension_shares"] == {"structure_clarity": 1.0}


def test_aggregation_never_emits_synthetic_total():
    entries, scenario_of = _per_case_entries()
    agg = dc.aggregate_components(entries, scenario_of)
    assert agg["synthetic_total_emitted"] is False
    blob = json.dumps(agg)
    for banned in ("darvin_total", "total_score", "grand_total", "overall_score"):
        assert banned not in blob
    assert set(agg["by_scenario"]) == {"scenario1", "scenario2", "scenario3", "scenario4"}


# ---------------------------------------------------------------------------
# Report builder over failure classes
# ---------------------------------------------------------------------------

def test_report_handles_phase0_failure_classes():
    cases = {
        "OK": _case("scenario1", cv=True, jd=True),
        "REQ-FAIL": _case("scenario1", cv=True, jd=True),
        "TRUNC": _case("scenario3", cv=True, jd=False),
        "FRAG": _case("scenario3", cv=True, jd=False),
    }
    annotations = {cid: {"sources": []} for cid in cases}
    rows = [
        _row("OK"),
        _row("REQ-FAIL", request_ok=False),
        _row("TRUNC", audit_ok=False),  # truncation/deadline class
        _row("FRAG", quality=_quality(compact=0.9, star=0.0, two_dim=0.05)),  # fragmented class
    ]
    report = dc.build_r3_report(rows, cases, annotations)
    assert report["evaluator_version"] == "darvin-component-evaluator-r3"
    assert report["gate"]["pass"] is False
    per_case = {entry["case_id"]: entry for entry in report["per_case"]}
    assert per_case["REQ-FAIL"]["evaluable"] is False
    assert per_case["TRUNC"]["evaluable"] is False
    # 失败案例的 15 个子维度全部保留且标记不适用，不丢失维度
    assert len(per_case["TRUNC"]["subdimensions"]) == 15
    assert all(not s["applicable"] for s in per_case["TRUNC"]["subdimensions"].values())
    # 碎片化案例的表达分显著低于干净案例
    ok_star = per_case["OK"]["subdimensions"]["star_richness"]["score01"]
    frag_star = per_case["FRAG"]["subdimensions"]["star_richness"]["score01"]
    assert frag_star < ok_star
    assert per_case["FRAG"]["subdimensions"]["professional_writing"]["score01"] < 0.2


def test_report_is_json_serializable(tmp_path):
    entries, scenario_of = _per_case_entries()
    agg = dc.aggregate_components(entries, scenario_of)
    json.dumps(agg, ensure_ascii=False)
