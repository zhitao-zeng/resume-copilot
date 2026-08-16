#!/usr/bin/env python3
"""Record-local realization (R24 Phase 3) contract tests.

Acceptance covered here:
- a semantic fallback fact degrades only its own record unit; clean records
  keep LLM realization; failed units restore exact record-local source text
- per-unit verifier violations and physical-call failures fall back per unit
- cross-record fact usage is rejected unit-locally, never globally
- optional summary only when every unit is clean and single-pack
- budget admission, packing, and full fact-coverage invariants
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "core")):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.v3.pipeline import run_v3_pipeline  # noqa: E402
from core.v3.training_schema import SCHEMA_VERSION  # noqa: E402


CV_TWO_RECORDS = (
    "甲公司 后端工程师 2019年1月 - 2021年6月\n"
    "负责订单系统重构，使用 Python 重写核心服务，接口耗时降低 30%\n"
    "乙公司 数据工程师 2021年7月 - 2023年12月\n"
    "搭建用户行为数仓，使用 SQL 完成 20 张主题表建模，报表延迟从 4 小时降到 10 分钟\n"
)


def _echo_semantic(_model, _system, user_prompt, **_kwargs):
    """Compile every candidate cleanly, keeping locked record ownership."""

    decisions = []
    for candidate in json.loads(user_prompt)["candidates"]:
        decisions.append({
            "candidate_fact_id": candidate["candidate_fact_id"],
            "classification": "fact",
            "record_id": candidate["locked_record_id"],
            "atoms": [{
                "quote": candidate["candidate_text"],
                "fact_type": "action",
                "destination_section": "experience",
                "destination_field": "bullet",
            }],
            "context_spans": [],
        })
    return {"schema_version": SCHEMA_VERSION, "decisions": decisions}


def _degrading_semantic(degrade_text: str):
    """Compile cleanly except candidates containing ``degrade_text``."""

    def semantic(_model, _system, user_prompt, **_kwargs):
        decisions = []
        for candidate in json.loads(user_prompt)["candidates"]:
            quote = candidate["candidate_text"]
            if degrade_text in quote:
                quote = "这不是逐字原文"  # force a semantic fallback for this fact
            decisions.append({
                "candidate_fact_id": candidate["candidate_fact_id"],
                "classification": "fact",
                "record_id": candidate["locked_record_id"],
                "atoms": [{
                    "quote": quote,
                    "fact_type": "action",
                    "destination_section": "experience",
                    "destination_field": "bullet",
                }],
                "context_spans": [],
            })
        return {"schema_version": SCHEMA_VERSION, "decisions": decisions}

    return semantic


def _echo_realizer(_model, _system, user_prompt, **_kwargs):
    request = json.loads(user_prompt)
    return {
        "schema_version": SCHEMA_VERSION,
        "request_fact_ids": request["request_fact_ids"],
        "units": [
            {
                "unit_id": unit["unit_id"],
                "claims": [
                    {
                        "claim_id": f"claim:{index}",
                        "section": fact["destination_section"],
                        "field": fact["destination_field"],
                        "text": fact["source_text"],
                        "fact_ids": [fact["fact_id"]],
                        "record_id": fact["record_id"],
                        "group_id": group["group_id"],
                    }
                    for group in unit["groups"]
                    for index, fact in enumerate(group["facts"])
                ],
            }
            for unit in request["units"]
        ],
    }


def _unit_reports(result):
    return {report["unit_id"]: report for report in result.realization_report.unit_reports}


def test_fallback_fact_degrades_only_its_own_record():
    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_degrading_semantic("订单系统重构"),
        realizer_llm_call=_echo_realizer,
    )

    assert result.semantic_report.status == "partial"
    assert result.semantic_report.fallback_fact_ids
    assert result.realization_report.status == "partial"
    reports = _unit_reports(result)
    statuses = sorted(report["status"] for report in reports.values())
    assert "deterministic_degraded" in statuses
    assert "llm" in statuses
    # The degraded record restores exact record-local source text; the clean
    # record is realized by the model.  No fact is lost either way.
    texts = [claim.text for claim in result.output.frozen.claims]
    assert any("订单系统重构" in text for text in texts)
    assert any("用户行为数仓" in text for text in texts)
    assert result.quality_report["atomic_factuality"]["recall"] == 1.0
    assert result.quality_report["atomic_factuality"]["precision"] == 1.0
    assert result.output.audit.ownership_errors == []


def test_clean_resume_all_units_llm_success():
    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=_echo_realizer,
    )
    assert result.semantic_report.status == "success"
    assert result.realization_report.status == "success"
    assert all(
        report["status"] == "llm"
        for report in result.realization_report.unit_reports
    )
    assert result.quality_report["atomic_factuality"]["recall"] == 1.0


def test_unit_verifier_violation_falls_back_per_unit():
    def violating_realizer(_model, _system, user_prompt, **_kwargs):
        request = json.loads(user_prompt)
        units = []
        for unit in request["units"]:
            claims = []
            for group in unit["groups"]:
                for index, fact in enumerate(group["facts"]):
                    text = fact["source_text"]
                    if "订单系统重构" in text:
                        text += "，新增指标 999%"  # novel numeric anchor
                    claims.append({
                        "claim_id": f"claim:{index}",
                        "section": fact["destination_section"],
                        "field": fact["destination_field"],
                        "text": text,
                        "fact_ids": [fact["fact_id"]],
                        "record_id": fact["record_id"],
                        "group_id": group["group_id"],
                    })
            units.append({"unit_id": unit["unit_id"], "claims": claims})
        return {
            "schema_version": SCHEMA_VERSION,
            "request_fact_ids": request["request_fact_ids"],
            "units": units,
        }

    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=violating_realizer,
    )

    assert result.realization_report.status == "partial"
    reports = _unit_reports(result)
    violating = [
        report for report in reports.values()
        if report["status"] == "deterministic_fallback"
    ]
    assert len(violating) == 1
    assert any("novel_numeric_anchor" in v for v in violating[0]["violations"])
    # The clean record is still LLM-realized and every fact survives.
    assert any(report["status"] == "llm" for report in reports.values())
    assert result.quality_report["atomic_factuality"]["recall"] == 1.0
    assert "999%" not in json.dumps(result.resume_data, ensure_ascii=False)


def test_cross_record_fact_use_rejected_unit_locally():
    def cross_record_realizer(_model, _system, user_prompt, **_kwargs):
        request = json.loads(user_prompt)
        # Steal one fact from the second unit into the first unit's claim.
        first, second = request["units"][0], request["units"][1]
        stolen_group = second["groups"][0]
        stolen_fact = stolen_group["facts"][0]
        own_group = first["groups"][0]
        own_fact = own_group["facts"][0]
        return {
            "schema_version": SCHEMA_VERSION,
            "request_fact_ids": request["request_fact_ids"],
            "units": [
                {
                    "unit_id": first["unit_id"],
                    "claims": [
                        {
                            "claim_id": "merged",
                            "section": "experience",
                            "field": "bullet",
                            "text": own_fact["source_text"] + "；" + stolen_fact["source_text"],
                            "fact_ids": [own_fact["fact_id"], stolen_fact["fact_id"]],
                            "record_id": own_fact["record_id"],
                            "group_id": own_group["group_id"],
                        },
                    ],
                },
                {
                    "unit_id": second["unit_id"],
                    "claims": [
                        {
                            "claim_id": f"echo:{index}",
                            "section": fact["destination_section"],
                            "field": fact["destination_field"],
                            "text": fact["source_text"],
                            "fact_ids": [fact["fact_id"]],
                            "record_id": fact["record_id"],
                            "group_id": stolen_group["group_id"],
                        }
                        for group in second["groups"]
                        for index, fact in enumerate(group["facts"])
                    ],
                },
            ],
        }

    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=cross_record_realizer,
    )

    reports = _unit_reports(result)
    first_report = reports[f"record:{result.output.graph.records[0].record_id}"]
    assert first_report["status"] == "deterministic_fallback"
    assert any("record_mismatch" in v or "fact_not_requested" in v for v in first_report["violations"])
    # The honest record is untouched by the sibling unit's failure.
    statuses = {report["unit_id"]: report["status"] for report in reports.values()}
    assert "llm" in statuses.values()
    assert result.quality_report["atomic_factuality"]["recall"] == 1.0
    assert result.output.audit.ownership_errors == []


def test_physical_pack_failure_falls_back_only_its_units(monkeypatch):
    monkeypatch.setenv("V3_REALIZER_PACK_CHARS", "120")  # force two packs
    monkeypatch.setenv("V3_REALIZER_CONCURRENCY", "1")
    calls = {"n": 0}

    def flaky_realizer(_model, _system, user_prompt, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated pack failure")
        return _echo_realizer(_model, _system, user_prompt, **_kwargs)

    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=flaky_realizer,
    )

    assert calls["n"] >= 2
    assert result.realization_report.status == "partial"
    reports = _unit_reports(result)
    failed = [r for r in reports.values() if r["status"] == "deterministic_fallback"]
    assert failed and "RuntimeError" in failed[0]["error"]
    assert any(r["status"] == "llm" for r in reports.values())
    assert result.quality_report["atomic_factuality"]["recall"] == 1.0


def test_packing_respects_character_budget(monkeypatch):
    monkeypatch.setenv("V3_REALIZER_PACK_CHARS", "120")
    monkeypatch.setenv("V3_REALIZER_CONCURRENCY", "1")
    calls = []

    def counting_realizer(_model, _system, user_prompt, **_kwargs):
        calls.append(json.loads(user_prompt))
        return _echo_realizer(_model, _system, user_prompt, **_kwargs)

    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=counting_realizer,
    )

    assert len(calls) >= 2
    # No physical request mixes two record units when the budget is tiny.
    for payload in calls:
        record_units = [u for u in payload["units"] if u["record_id"]]
        assert len(record_units) <= 1
    assert result.quality_report["atomic_factuality"]["recall"] == 1.0


def test_summary_only_when_all_units_clean_and_single_pack():
    def summarizing_realizer(_model, _system, user_prompt, **_kwargs):
        response = _echo_realizer(_model, _system, user_prompt, **_kwargs)
        request = json.loads(user_prompt)
        if "optional_summary" in request:
            fact_id = request["request_fact_ids"][0]
            source = next(
                fact["source_text"]
                for unit in request["units"]
                for group in unit["groups"]
                for fact in group["facts"]
                if fact["fact_id"] == fact_id
            )
            response["summary_claims"] = [{
                "claim_id": "summary",
                "section": "summary",
                "field": "summary",
                "text": f"具备{source}相关经验。",
                "fact_ids": [fact_id],
                "record_id": None,
                "group_id": "summary:profile",
            }]
        return response

    clean = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=summarizing_realizer,
    )
    assert clean.realization_report.status == "success"
    assert clean.resume_data.get("summary")

    degraded = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_degrading_semantic("订单系统重构"),
        realizer_llm_call=summarizing_realizer,
    )
    assert degraded.realization_report.status == "partial"
    # Degraded units disable the optional summary request entirely.
    assert not degraded.resume_data.get("summary")


def test_invalid_summary_claim_is_rejected_without_touching_body():
    def bad_summary_realizer(_model, _system, user_prompt, **_kwargs):
        response = _echo_realizer(_model, _system, user_prompt, **_kwargs)
        request = json.loads(user_prompt)
        if "optional_summary" in request:
            response["summary_claims"] = [{
                "claim_id": "summary",
                "section": "summary",
                "field": "summary",
                "text": "十年经验的管理者。",  # unsupported verbatim + novel number
                "fact_ids": [request["request_fact_ids"][0]],
                "record_id": None,
                "group_id": "summary:profile",
            }]
        return response

    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=bad_summary_realizer,
    )
    # Body claims survive; only the summary surface is dropped.
    assert result.realization_report.status == "success"
    assert not result.resume_data.get("summary")
    assert any(
        report["status"] == "summary_rejected"
        for report in result.realization_report.unit_reports
    )
    assert result.quality_report["atomic_factuality"]["recall"] == 1.0


def test_budget_admission_keeps_everything_deterministic(monkeypatch):
    import server_runtime

    calls = {"realizer": 0}

    def realizer(*_args, **_kwargs):
        calls["realizer"] += 1
        raise AssertionError("realizer must not start below its declared budget")

    monkeypatch.setenv("V3_REALIZER_MIN_REMAINING_SECONDS", "240")
    monkeypatch.setattr(server_runtime, "remaining_request_seconds", lambda: 120.0)
    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=realizer,
    )

    assert calls["realizer"] == 0
    assert result.realization_report.status == "budget_fallback"
    assert result.quality_report["atomic_factuality"]["recall"] == 1.0
    assert result.quality_report["atomic_factuality"]["precision"] == 1.0


def test_multi_pack_skips_summary_request(monkeypatch):
    monkeypatch.setenv("V3_REALIZER_PACK_CHARS", "120")
    monkeypatch.setenv("V3_REALIZER_CONCURRENCY", "1")
    seen = []

    def realizer(_model, _system, user_prompt, **_kwargs):
        seen.append(json.loads(user_prompt))
        return _echo_realizer(_model, _system, user_prompt, **_kwargs)

    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=realizer,
    )
    assert len(seen) >= 2
    # v1 contract: the cross-record summary is single-request only; multi-pack
    # resumes report the skip instead of synthesizing from a partial view.
    assert all("optional_summary" not in payload for payload in seen)
    assert result.realization_report.status == "success"


def _label_splitting_semantic(_model, _system, user_prompt, **_kwargs):
    """Mirror the production label/value split: value atom + label context."""

    decisions = []
    for candidate in json.loads(user_prompt)["candidates"]:
        text = candidate["candidate_text"]
        if "：" in text or ":" in text:
            sep = "：" if "：" in text else ":"
            label, _, value = text.partition(sep)
            quote = value.strip()
            context = [{
                "quote": label + sep + " ",
                "reason": "label",
                "char_start": 0,
                "char_end": len(label) + len(sep),
            }]
        else:
            quote = text
            context = []
        decisions.append({
            "candidate_fact_id": candidate["candidate_fact_id"],
            "classification": "fact",
            "record_id": candidate["locked_record_id"],
            "atoms": [{
                "quote": quote,
                "fact_type": "metric" if any(ch.isdigit() for ch in quote) else "action",
                "destination_section": "education",
                "destination_field": "bullet",
            }],
            "context_spans": context,
        })
    return {"schema_version": SCHEMA_VERSION, "decisions": decisions}


def test_unmoored_value_claim_falls_back_to_labeled_source_text():
    """A bare '3.84' without its label must not survive per-unit validation."""

    def bare_value_realizer(_model, _system, user_prompt, **_kwargs):
        request = json.loads(user_prompt)
        # 请求应携带 label_prefix 提示；这里故意忽略它，只写剥离标签的数值。
        labeled = [
            fact
            for unit in request["units"]
            for group in unit["groups"]
            for fact in group["facts"]
            if fact.get("label_prefix")
        ]
        assert labeled, "label-split facts must advertise their label_prefix"
        return {
            "schema_version": SCHEMA_VERSION,
            "request_fact_ids": request["request_fact_ids"],
            "units": [
                {
                    "unit_id": unit["unit_id"],
                    "claims": [
                        {
                            "claim_id": f"claim:{index}",
                            "section": fact["destination_section"],
                            "field": fact["destination_field"],
                            "text": fact["source_text"],  # bare value, label dropped
                            "fact_ids": [fact["fact_id"]],
                            "record_id": fact["record_id"],
                            "group_id": group["group_id"],
                        }
                        for group in unit["groups"]
                        for index, fact in enumerate(group["facts"])
                    ],
                }
                for unit in request["units"]
            ],
        }

    result = run_v3_pipeline(
        cv_text="乙大学 软件工程 2016年9月 - 2020年6月\n成绩/平均绩点: 3.84",
        semantic_llm_call=_label_splitting_semantic,
        realizer_llm_call=bare_value_realizer,
    )

    reports = _unit_reports(result)
    labeled_reports = [
        report for report in reports.values()
        if any("label_not_preserved" in violation for violation in report["violations"])
    ]
    assert labeled_reports, "unmoored value claim must be rejected unit-locally"
    assert labeled_reports[0]["status"] == "deterministic_fallback"
    # 确定性回退恢复带标签的完整原文，事实不丢失。
    rendered = json.dumps(result.resume_data, ensure_ascii=False)
    assert "成绩/平均绩点: 3.84" in rendered
    assert result.quality_report["atomic_factuality"]["recall"] == 1.0
    assert result.quality_report["atomic_factuality"]["precision"] == 1.0


def test_labeled_value_claim_is_accepted():
    def labeled_realizer(_model, _system, user_prompt, **_kwargs):
        request = json.loads(user_prompt)
        return {
            "schema_version": SCHEMA_VERSION,
            "request_fact_ids": request["request_fact_ids"],
            "units": [
                {
                    "unit_id": unit["unit_id"],
                    "claims": [
                        {
                            "claim_id": f"claim:{index}",
                            "section": fact["destination_section"],
                            "field": fact["destination_field"],
                            "text": (fact.get("label_prefix") or "") + fact["source_text"],
                            "fact_ids": [fact["fact_id"]],
                            "record_id": fact["record_id"],
                            "group_id": group["group_id"],
                        }
                        for group in unit["groups"]
                        for index, fact in enumerate(group["facts"])
                    ],
                }
                for unit in request["units"]
            ],
        }

    result = run_v3_pipeline(
        cv_text="乙大学 软件工程 2016年9月 - 2020年6月\n成绩/平均绩点: 3.84",
        semantic_llm_call=_label_splitting_semantic,
        realizer_llm_call=labeled_realizer,
    )

    assert result.realization_report.status == "success"
    assert not [
        violation
        for report in result.realization_report.unit_reports
        for violation in report["violations"]
    ]
    assert "成绩/平均绩点: 3.84" in json.dumps(result.resume_data, ensure_ascii=False)
