#!/usr/bin/env python3
"""Phase 4 summary compiler + concise reply compiler contract tests."""

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
from core.v3.summary_compiler import (  # noqa: E402
    MAX_COMPACT_CHARS,
    compile_summary,
    validate_summary_sentences,
)
from core.v3.training_schema import SCHEMA_VERSION  # noqa: E402

from tests.test_v3_record_local import (  # noqa: E402
    CV_TWO_RECORDS,
    _echo_realizer,
    _echo_semantic,
)


# ---------------------------------------------------------------------------
# Sentence-level verification rules
# ---------------------------------------------------------------------------

def _graph_with_sentences():
    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=_echo_realizer,
    )
    return result


def _fact_ids(result):
    return [fact.fact_id for fact in result.output.graph.eligible_facts()]


def test_valid_bound_sentence_passes():
    result = _graph_with_sentences()
    fid = _fact_ids(result)[0]
    text = result.output.graph.fact_map()[fid].text
    # 完整成句（终止标点）的绑定句通过；裸记录头不是合法总结句。
    verified, violations = validate_summary_sentences(
        [{"text": f"曾{text}。", "fact_ids": [fid]}], result.output.graph,
    )
    assert violations == [], violations
    assert verified == [{"text": f"曾{text}。", "fact_ids": [fid]}]
    verified, violations = validate_summary_sentences(
        [{"text": text, "fact_ids": [fid]}], result.output.graph,
    )
    assert verified == []
    assert any("incomplete_ending" in v for v in violations)


def test_computed_tenure_rejected_unless_stated():
    result = _graph_with_sentences()
    graph = result.output.graph
    fid = _fact_ids(result)[0]
    verified, violations = validate_summary_sentences(
        [{"text": "拥有5年工作经验。", "fact_ids": [fid]}], graph,
    )
    assert verified == []
    assert any("computed_tenure" in violation for violation in violations)


def test_unsupported_adjective_and_comparative_rejected():
    result = _graph_with_sentences()
    graph = result.output.graph
    fid = _fact_ids(result)[0]
    for text, marker in (
        ("资深后端工程师，精通 Python。", "unsupported_adjective"),
        ("最优秀的数据工程师。", "unsupported_comparative"),
    ):
        verified, violations = validate_summary_sentences(
            [{"text": text, "fact_ids": [fid]}], graph,
        )
        assert verified == []
        assert any(marker in violation for violation in violations)


def test_adjective_allowed_when_source_states_it():
    result = _graph_with_sentences()
    graph = result.output.graph
    fid = _fact_ids(result)[0]
    fact_text = graph.fact_map()[fid].text
    verified, violations = validate_summary_sentences(
        [{"text": f"擅长{fact_text}", "fact_ids": [fid]}],
        graph,
        allowed_fact_ids=[fid],
    )
    # "擅长"不在事实原文中 → 仍应拒绝；把形容词放进引用事实再测一次。
    assert verified == []
    assert any("unsupported_adjective" in v for v in violations)


def test_novel_number_and_unbound_claim_rejected():
    result = _graph_with_sentences()
    graph = result.output.graph
    fid = _fact_ids(result)[0]
    verified, violations = validate_summary_sentences(
        [{"text": "管理 12 人团队。", "fact_ids": [fid]}], graph,
    )
    assert verified == []
    assert any("novel_numeric_anchor" in v for v in violations)
    verified, violations = validate_summary_sentences(
        [{"text": "后端工程师。", "fact_ids": []}], graph,
    )
    assert verified == []
    assert any("unbound_claim" in v for v in violations)


def test_timeline_concatenation_rejected():
    result = _graph_with_sentences()
    graph = result.output.graph
    fids = _fact_ids(result)
    verified, violations = validate_summary_sentences(
        [{"text": "2019年1月 - 2021年6月在甲公司，2021年7月 - 2023年12月在乙公司。", "fact_ids": fids[:2]}],
        graph,
    )
    assert verified == []
    assert any("timeline_concatenation" in v for v in violations)


def test_foreign_anchor_rejected_when_not_bound():
    def typed_semantic(_model, _system, user_prompt, **_kwargs):
        decisions = []
        for candidate in json.loads(user_prompt)["candidates"]:
            text = candidate["candidate_text"]
            fact_type = "organization" if "公司" in text else "action"
            decisions.append({
                "candidate_fact_id": candidate["candidate_fact_id"],
                "classification": "fact",
                "record_id": candidate["locked_record_id"],
                "atoms": [{
                    "quote": text,
                    "fact_type": fact_type,
                    "destination_section": "experience",
                    "destination_field": "bullet",
                }],
                "context_spans": [],
            })
        return {"schema_version": SCHEMA_VERSION, "decisions": decisions}

    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=typed_semantic,
        realizer_llm_call=_echo_realizer,
    )
    graph = result.output.graph
    b_facts = [
        fact.fact_id for fact in graph.eligible_facts()
        if fact.record_id and "record:2" in fact.record_id
    ]
    all_ids = [fact.fact_id for fact in graph.eligible_facts()]
    # 只绑定乙公司（record:2）的事实，却在总结里提到甲公司 → foreign_entity
    verified, violations = validate_summary_sentences(
        [{"text": "曾在甲公司负责订单系统。", "fact_ids": b_facts or all_ids[-2:]}],
        graph,
        allowed_fact_ids=all_ids,
    )
    assert verified == []
    assert any("foreign_entity" in v for v in violations)


def test_reply_conflicts_hide_internal_ids():
    from core.v3.reply_builder import friendly_conflicts

    friendly = friendly_conflicts(["cv:record:4:period", "cv:record:4:period", "cv:record:7:organization"])
    assert friendly == ["存在多处不一致的时间表述，请核对确认。", "存在多处不一致的组织表述，请核对确认。"]
    assert all("record:" not in item and "cv:" not in item for item in friendly)


def test_excerpt_preserves_latin_word_boundaries():
    """R27 task 9 (D7): reply excerpts fold, never delete, whitespace."""

    from core.v3.reply_builder import _excerpt

    assert _excerpt("Delivered results using structured workflows", 48) == "Delivered results using structured workflows"
    assert " " in _excerpt("Delivered results using structured workflows and clear communication", 30)
    # 中文无内空格输入逐字节不变
    assert _excerpt("负责订单系统重构") == "负责订单系统重构"
    # 超长仍按 limit 截断
    assert _excerpt("a" * 40, 10) == "aaaaaaaaa…"


def test_english_source_uses_english_summary_prompt():
    """D10: Latin-dominant evidence gets the English summary surface.

    The woven sentence may only add true glue around cited source text:
    evaluative words (Experienced/proficient/skilled) are content, not glue,
    and require verbatim source support like any other content word.
    """

    seen = {}

    def summary_llm(_model, system_prompt, user_prompt, **_kwargs):
        seen["prompt"] = system_prompt
        request = json.loads(user_prompt)
        fact = request["evidence_facts"][0]
        return {
            "schema_version": SCHEMA_VERSION,
            "sentences": [{"text": f"{fact['source_text']}.", "fact_ids": [fact["fact_id"]]}],
        }

    def english_semantic(_model, _system, user_prompt, **_kwargs):
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

    result = run_v3_pipeline(
        cv_text="Data Analyst 2019 - 2023\nDelivered results using structured workflows",
        semantic_llm_call=english_semantic,
        realizer_llm_call=_echo_realizer,
        summary_llm_call=summary_llm,
    )
    assert seen.get("prompt") and "English" in seen["prompt"]
    assert result.resume_data.get("summary")


def test_english_summary_rejects_unsupported_evaluative_glue():
    """An English summary may not assert Experienced/proficiency the source
    never states: evaluative adjectives are content words, not connectors."""

    def summary_llm(_model, _system_prompt, user_prompt, **_kwargs):
        request = json.loads(user_prompt)
        fact = request["evidence_facts"][0]
        return {
            "schema_version": SCHEMA_VERSION,
            "sentences": [{"text": f"Experienced in {fact['source_text']}.", "fact_ids": [fact["fact_id"]]}],
        }

    def english_semantic(_model, _system, user_prompt, **_kwargs):
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

    result = run_v3_pipeline(
        cv_text="Data Analyst 2019 - 2023\nDelivered results using structured workflows",
        semantic_llm_call=english_semantic,
        realizer_llm_call=_echo_realizer,
        summary_llm_call=summary_llm,
    )
    assert not result.resume_data.get("summary")
    contract = result.quality_report["model_contract"]
    assert contract["summary_status"] == "dropped_unverifiable"
    assert any("escape_latin:Experienced" in item for item in contract["summary_violations"])


def test_missing_fields_summary_three_state():
    """R27 task 7: unrelated additional content must not imply a summary."""

    from core.v3.pipeline import _missing_fields

    base = {
        "meta": {"name": "张三", "phone": "138"},
        "experience": [{"bullets": ["负责后端开发"]}],
        "education": [{"school": "甲大学"}],
        "skills": {"others": ["Python"]},
    }
    # 补充信息只有兴趣爱好、summary 未降级 → not_provided
    with_hobby = {
        **base,
        "additional_sections": {"补充信息": ["冲浪、创意设计、烹饪艺术"]},
    }
    missing = _missing_fields(with_hobby)
    summary_item = next(item for item in missing if item["field"] == "summary")
    assert summary_item["source"] == "not_provided"
    # summary 真被降级 → not_rendered
    missing = _missing_fields(with_hobby, degraded={"summary": "dropped_unverifiable"})
    summary_item = next(item for item in missing if item["field"] == "summary")
    assert summary_item["source"] == "not_rendered"
    assert "未能通过校验" in summary_item["reason"]


def test_length_budget_drops_trailing_sentences():
    result = _graph_with_sentences()
    graph = result.output.graph
    fid = _fact_ids(result)[0]
    fact_text = graph.fact_map()[fid].text
    sentences = [{"text": f"曾{fact_text}。", "fact_ids": [fid]}] * 6  # 远超 100 字
    verified, violations = validate_summary_sentences(sentences, graph)
    total = sum(len(s["text"].replace(" ", "")) for s in verified)
    assert total <= MAX_COMPACT_CHARS
    assert any("exceeds" in v for v in violations)


def test_rejected_sentences_do_not_consume_length_budget():
    """Task D4: rejected sentences must not empty the valid ones."""

    result = _graph_with_sentences()
    graph = result.output.graph
    fid = _fact_ids(result)[0]
    fact_text = graph.fact_map()[fid].text
    long_bad = "拥有多年丰富经验并且成绩为最" + fact_text * 8  # unsupported adjective + comparative
    sentences = [
        {"text": long_bad, "fact_ids": [fid]},
        {"text": long_bad + "更多内容", "fact_ids": [fid]},
        {"text": f"曾{fact_text}。", "fact_ids": [fid]},
    ]
    verified, violations = validate_summary_sentences(sentences, graph)
    assert verified == [{"text": f"曾{fact_text}。", "fact_ids": [fid]}]
    assert "summary_empty_after_length_repair" not in violations


# ---------------------------------------------------------------------------
# compile_summary through the pipeline
# ---------------------------------------------------------------------------

def _summary_llm(sentences):
    def call(_model, _system, user_prompt, **_kwargs):
        request = json.loads(user_prompt)
        assert request["task"] == "profile_summary"
        return {
            "schema_version": SCHEMA_VERSION,
            "sentences": sentences(request),
        }

    return call


def test_pipeline_generates_verified_summary():
    def sentences(request):
        fact = request["evidence_facts"][0]
        return [{"text": f"曾{fact['source_text']}。", "fact_ids": [fact["fact_id"]]}]

    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=_echo_realizer,
        summary_llm_call=_summary_llm(sentences),
    )
    assert result.resume_data.get("summary")
    assert result.quality_report["model_contract"]["summary_status"] in {
        "generated", "revalidated",
    }
    assert result.output.audit.clean


def test_pipeline_drops_unverifiable_summary_fail_closed():
    def sentences(_request):
        fact_id = _fact_ids(_graph_with_sentences())[0]
        return [{"text": "十年经验的资深专家，最擅长一切。", "fact_ids": [fact_id]}]

    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=_echo_realizer,
        summary_llm_call=_summary_llm(sentences),
    )
    assert not result.resume_data.get("summary")
    assert result.quality_report["model_contract"]["summary_status"] == "dropped_unverifiable"
    assert result.output.audit.clean
    # 正文事实不受影响
    assert result.quality_report["atomic_factuality"]["recall"] == 1.0


def test_skeleton_skips_summary():
    result = run_v3_pipeline(
        jd_text="招聘产品经理",
        use_llm=False,
    )
    assert result.quality_report["model_contract"]["summary_status"] == "skipped_skeleton"
    assert not result.resume_data.get("summary")


# ---------------------------------------------------------------------------
# Concise reply compiler
# ---------------------------------------------------------------------------

def test_reply_is_concise_with_all_components():
    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=_echo_realizer,
    )
    reply = result.reply_text
    assert "生成方向：" in reply
    assert "岗位建议：" in reply or "岗位匹配：" in reply
    assert "缺失信息：" in reply
    assert "冲突检查：" in reply
    # 不再逐条回声全部写入事实
    assert "已写入信息" not in reply
    assert len(reply) <= 800


def test_reply_jd_analysis_bounded_and_evidence_backed():
    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        jd_text="招聘后端工程师\n要求：Python 开发经验\n要求：团队管理\n要求：云计算架构\n要求： SQL 数据分析",
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=_echo_realizer,
    )
    reply = result.reply_text
    assert "岗位匹配：" in reply
    assert "来自JD" in reply
    match_lines = [line for line in reply.splitlines() if line.startswith("- 匹配：")]
    gap_lines = [line for line in reply.splitlines() if line.startswith("- 差距：")]
    assert len(match_lines) <= 3
    assert len(gap_lines) <= 3
    assert any("依据：" in line for line in match_lines)


def test_reply_never_shows_jd_structure_artifacts_as_gaps():
    jd = (
        "前端开发工程师\n"
        "职位概述：\n"
        "作为前端开发工程师，负责使用现代框架开发高可用界面\n"
        "主要职责：\n"
        "要求：5 年以上 React 开发经验\n"
        "要求：熟悉 TypeScript 与构建工具链"
    )
    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        jd_text=jd,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=_echo_realizer,
    )
    reply = result.reply_text
    assert "职位概述" not in reply
    assert "主要职责" not in reply
    gap_lines = [line for line in reply.splitlines() if line.startswith("- 差距：")]
    assert all("前端开发工程师；" not in line for line in gap_lines)
    # 真正的要求内容仍然在
    assert "React" in reply or "TypeScript" in reply or "现代框架" in reply


def test_reply_groups_undetermined_ownership_without_internal_ids():
    query = (
        "负责海外仓系统搭建，提升发货效率\n"
        "参与门店数字化改造项目\n"
        "熟练使用 SQL 与 Python 做经营分析"
    )
    result = run_v3_pipeline(
        query_text=query,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=_echo_realizer,
    )
    reply = result.reply_text
    if "待确认归属：" in reply:
        section = reply.split("待确认归属：", 1)[1].split("\n冲突检查：", 1)[0]
        assert "cv:fact:" not in section and "query:fact:" not in section
        assert "record:" not in section
        assert "请确认" in section
    assert "冲突检查：" in reply


def test_reply_framework_mode_is_short_and_exact():
    result = run_v3_pipeline(jd_text="招聘产品经理", use_llm=False)
    reply = result.reply_text
    assert "结构化待填写框架" in reply
    assert "缺失信息：" in reply
    assert len(reply) <= 800


def test_reply_conflicts_real_or_explicit_none():
    result = run_v3_pipeline(
        cv_text=CV_TWO_RECORDS,
        semantic_llm_call=_echo_semantic,
        realizer_llm_call=_echo_realizer,
    )
    reply = result.reply_text
    assert "未发现时间或信息冲突" in reply
