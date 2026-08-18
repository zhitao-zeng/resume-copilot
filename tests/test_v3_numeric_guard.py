#!/usr/bin/env python3
"""R25 OCR numeric guard contract tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "core")):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.v3.numeric_guard import (  # noqa: E402
    find_suspect_numeric_facts,
    quarantine_suspect_numeric_facts,
)
from core.v3.pipeline import run_v3_pipeline  # noqa: E402
from core.v3.training_schema import SCHEMA_VERSION  # noqa: E402

from tests.test_v3_record_local import _echo_realizer  # noqa: E402


def _semantic_passthrough(_model, _system, user_prompt, **_kwargs):
    decisions = []
    for candidate in json.loads(user_prompt)["candidates"]:
        decisions.append({
            "candidate_fact_id": candidate["candidate_fact_id"],
            "classification": "fact",
            "record_id": candidate["locked_record_id"],
            "atoms": [{
                "quote": candidate["candidate_text"],
                "fact_type": "period" if any(ch.isdigit() for ch in candidate["candidate_text"]) else "action",
                "destination_section": "experience",
                "destination_field": "bullet",
            }],
            "context_spans": [],
        })
    return {"schema_version": SCHEMA_VERSION, "decisions": decisions}


def test_truncated_year_is_quarantined_and_surfaced():
    result = run_v3_pipeline(
        cv_text="办公室协调员，某公司 204-2011\n负责行政支持和日程安排",
        semantic_llm_call=_semantic_passthrough,
        realizer_llm_call=_echo_realizer,
    )
    quarantine = result.quality_report["model_contract"]["numeric_quarantine"]
    assert quarantine, "truncated year 204 must be quarantined"
    assert any("204" in item["text"] for item in quarantine)
    rendered = json.dumps(result.resume_data, ensure_ascii=False)
    assert "204-2011" not in rendered
    assert "204" not in rendered
    # 回复中以待确认数字呈现，无内部 ID
    assert "待确认数字" in result.reply_text
    assert "请核对原件" in result.reply_text
    assert "cv:fact:" not in result.reply_text


def test_valid_period_is_not_quarantined():
    result = run_v3_pipeline(
        cv_text="甲公司 后端工程师 2019年1月 - 2021年6月\n负责订单系统重构",
        semantic_llm_call=_semantic_passthrough,
        realizer_llm_call=_echo_realizer,
    )
    quarantine = result.quality_report["model_contract"]["numeric_quarantine"]
    assert quarantine == []
    assert "2019" in json.dumps(result.resume_data, ensure_ascii=False)


def test_reversed_and_overlong_periods_are_suspect():
    result = run_v3_pipeline(
        cv_text="乙公司 顾问 2023年 - 2019年\n负责咨询交付",
        semantic_llm_call=_semantic_passthrough,
        realizer_llm_call=_echo_realizer,
    )
    quarantine = result.quality_report["model_contract"]["numeric_quarantine"]
    assert any("reversed_period" in str(item["reasons"]) for item in quarantine)


def test_low_confidence_ocr_numeric_quarantined_only_when_untrusted():
    from core.v3.contracts import FactGraph, FactUnit, SourceSpan

    doc = "处理超过5个预约\n处理超过50个预约"

    def fact(fid, text, confidence, start):
        return FactUnit(
            fact_id=fid, source_id="cv", source_type="cv", text=text,
            spans=[SourceSpan(source_id="cv", char_start=start, char_end=start + len(text))],
            confidence=confidence,
        )

    graph = FactGraph(
        documents={"cv": doc},
        facts=[
            fact("cv:fact:1", "处理超过5个预约", 0.55, 0),
            fact("cv:fact:2", "处理超过50个预约", 0.97, 9),
        ],
    )
    suspects = find_suspect_numeric_facts(graph)
    assert [item["fact_id"] for item in suspects] == ["cv:fact:1"]
    quarantine_suspect_numeric_facts(graph)
    assert graph.fact_map()["cv:fact:1"].eligible is False
    assert graph.fact_map()["cv:fact:2"].eligible is True


def test_quarantine_cascades_to_semantic_atoms():
    result = run_v3_pipeline(
        cv_text="办公室协调员，某公司 204-2011\n负责行政支持",
        semantic_llm_call=_semantic_passthrough,
        realizer_llm_call=_echo_realizer,
    )
    graph = result.output.graph
    atom = next(
        (fact for fact in graph.facts if fact.base_fact_id),
        None,
    )
    if atom is not None and "204" in atom.text:
        assert atom.eligible is False
    assert result.quality_report["atomic_factuality"]["precision"] == 1.0


def test_bare_separator_is_not_a_date_shell():
    from core.v3.contracts import FactGraph, FactUnit, SourceSpan

    doc = "2019年 - 2021年\n-\n年月\n"
    facts = [
        FactUnit(fact_id="cv:fact:0", source_id="cv", source_type="cv", text="2019年 - 2021年",
                 spans=[SourceSpan(source_id="cv", char_start=0, char_end=13)]),
        FactUnit(fact_id="cv:fact:1", source_id="cv", source_type="cv", text="-",
                 spans=[SourceSpan(source_id="cv", char_start=14, char_end=15)]),
        FactUnit(fact_id="cv:fact:2", source_id="cv", source_type="cv", text="年月",
                 spans=[SourceSpan(source_id="cv", char_start=16, char_end=18)]),
    ]
    graph = FactGraph(documents={"cv": doc}, facts=facts)
    suspects = find_suspect_numeric_facts(graph)
    assert [item["fact_id"] for item in suspects] == ["cv:fact:2"]


# --- R28 task 1: phone shapes are contact data, never suspect year ranges ---

from core.v3.contracts import FactUnit, SourceSpan  # noqa: E402
from core.v3.numeric_guard import _suspicion_reasons  # noqa: E402


def _bare_fact(text: str) -> FactUnit:
    return FactUnit(
        fact_id="f", source_id="cv", source_type="cv", fact_type="other",
        text=text,
        spans=[SourceSpan(source_id="cv", char_start=0, char_end=len(text))],
        confidence=1.0,
    )


@pytest.mark.parametrize("text", [
    "电话：189-2758-0370",
    "手机 13812345678",
    "+86 138 1234 5678",
    "座机 0755-8888 7777",
])
def test_phone_shapes_are_not_suspect(text):
    assert _suspicion_reasons(_bare_fact(text)) == []


@pytest.mark.parametrize("text", [
    "2004年-204年",
    "204-2011",
    "1980-2035年",
])
def test_broken_year_ranges_still_quarantined(text):
    assert _suspicion_reasons(_bare_fact(text))


def test_phone_next_to_real_period_keeps_period_scrutiny():
    # A phone must not shield an unrelated broken range in the same fact.
    reasons = _suspicion_reasons(_bare_fact("13812345678，任职 204-2011"))
    assert any("204" in r for r in reasons)
