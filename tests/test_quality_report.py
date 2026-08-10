"""Regression tests for deterministic, score-free output quality reporting."""

from __future__ import annotations

import asyncio

import pytest

from evidence_binding import bind_resume_evidence
from quality_report import (
    assess_jd_requirements,
    build_quality_report,
    extract_jd_requirements,
)
from resume_copilot_pipeline import PipelineContext, build_reply_text, stage_prepare_report
from source_adapter import build_source_bundle
from v2_schemas import CanonicalResume, Change


def _report_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).casefold()
            yield from _report_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _report_keys(item)


def _statuses(report: dict) -> list[str]:
    return [
        item["status"]
        for item in report["job_alignment"]["requirements"]
    ]


def test_quality_report_separates_source_omissions_from_input_noise():
    source = build_source_bundle(
        "姓名：李明\n项目经历\n涂布优化项目\n"
        "记录12批次生产参数。\n分析异常批次原因。\n输出工艺改进建议。",
        "请帮我突出管理能力",
        "岗位要求\n熟悉六西格玛",
    )
    resume = CanonicalResume.model_validate({
        "meta": {"name": "李明"},
        "projects": [{
            "name": "涂布优化项目",
            "bullets": ["记录12批次生产参数。"],
        }],
    })
    bindings = bind_resume_evidence(resume, source)

    report = build_quality_report(
        source=source,
        resume=resume,
        evidence_bindings=bindings,
        jd_text="岗位要求\n熟悉六西格玛",
    )

    preservation = report["source_preservation"]
    assert preservation["status"] == "unrepresented_items_detected"
    excerpts = [item["excerpt"] for item in preservation["unrepresented_items"]]
    assert "分析异常批次原因。" in excerpts
    assert "输出工艺改进建议。" in excerpts
    joined = "\n".join(excerpts)
    assert "项目经历" not in joined
    assert "管理能力" not in joined
    assert "六西格玛" not in joined


def test_quality_report_does_not_flag_headings_or_distributed_record_fields():
    source = build_source_bundle(
        "个人信息\n张晨\n教育经历\n复旦大学 本科 计算机科学 09-2016 - 06-2020\n"
        "工作经历\n07-2022 - 05-2026 第四范式 产品经理\n"
        "负责企业数据平台需求调研、版本规划与跨团队推进，推动报表配置效率提升30%。",
        "",
        "",
    )
    resume = CanonicalResume.model_validate({
        "meta": {"name": "张晨"},
        "education": [{
            "school": "复旦大学", "degree": "本科", "major": "计算机科学",
            "period": "09-2016 - 06-2020",
        }],
        "experience": [{
            "organization": "第四范式", "role": "产品经理",
            "period": "07-2022 - 05-2026",
            "bullets": [
                "负责企业数据平台需求调研、版本规划与跨团队推进",
                "推动报表配置效率提升30%",
            ],
        }],
    })

    report = build_quality_report(
        source=source,
        resume=resume,
        evidence_bindings=bind_resume_evidence(resume, source),
    )

    assert report["source_preservation"]["status"] == "no_unrepresented_items_detected"
    assert report["source_preservation"]["unrepresented_items"] == []


def test_quality_report_lists_generated_items_removed_without_evidence():
    source = build_source_bundle("姓名：李明", "", "")
    resume = CanonicalResume.model_validate({"meta": {"name": "李明"}})
    bindings = bind_resume_evidence(resume, source)

    report = build_quality_report(
        source=source,
        resume=resume,
        evidence_bindings=bindings,
        changes=[
            Change(
                path="projects[0].bullets[1]",
                action="remove",
                reason="No candidate evidence binding",
            ),
            Change(
                path="experience[0].bullets[0]",
                action="replace",
                reason="Evidence-preserving wording optimization",
            ),
        ],
    )

    grounding = report["fact_grounding"]
    assert grounding["unsupported_item_count"] == 1
    assert grounding["unsupported_items_removed"] == [{
        "canonical_field_path": "projects[0].bullets[1]",
        "field_label": "项目经历第1项第2条",
        "message": "缺少候选人事实依据，未写入最终简历。",
    }]


def test_dynamic_jd_requirements_distinguish_supported_partial_and_missing():
    cv_text = (
        "工作经历\n某科技公司｜产品经理｜2022.01-2024.01\n"
        "负责用户调研并输出PRD。"
    )
    jd_text = (
        "岗位职责\n"
        "1. 负责用户调研并输出PRD\n"
        "2. 负责用户调研、竞品分析和需求优先级管理\n"
        "3. 熟悉SQL并搭建经营数据看板"
    )
    source = build_source_bundle(cv_text, "", jd_text)
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "某科技公司",
            "role": "产品经理",
            "period": "2022.01-2024.01",
            "bullets": ["负责用户调研并输出PRD。"],
        }],
    })
    bindings = bind_resume_evidence(resume, source)

    report = build_quality_report(
        source=source,
        resume=resume,
        evidence_bindings=bindings,
        jd_text=jd_text,
        target_role="产品经理",
    )

    assert _statuses(report) == ["supported", "partial", "missing"]
    partial = report["job_alignment"]["requirements"][1]
    assert partial["evidence"][0]["canonical_field_path"] == "experience[0].bullets[0]"
    assert "竞品分析和需求优先级管理" in partial["missing_aspects"]


def test_jd_only_request_returns_framework_report_without_fake_support():
    jd_text = (
        "任职要求\n"
        "1. 熟悉Python和SQL\n"
        "2. 负责经营数据看板搭建"
    )
    source = build_source_bundle(
        "",
        "请根据这份JD生成待填写简历框架，不要编造个人经历",
        jd_text,
    )
    resume = CanonicalResume.model_validate({
        "meta": {"target_role": "数据分析师"},
    })

    report = build_quality_report(
        source=source,
        resume=resume,
        evidence_bindings=[],
        jd_text=jd_text,
        target_role="数据分析师",
        framework_mode=True,
    )

    assert report["document_mode"] == "framework"
    assert report["source_preservation"]["status"] == "not_applicable"
    assert report["source_preservation"]["source_item_count"] == 0
    assert _statuses(report) == ["missing", "missing"]
    assert report["follow_up_questions"]
    forbidden = ("score", "ratio", "similarity")
    assert not any(any(token in key for token in forbidden) for key in _report_keys(report))


@pytest.mark.parametrize(
    ("target_role", "fact", "requirement"),
    [
        (
            "BIM工程师",
            "使用Revit与Navisworks完成BIM碰撞检查。",
            "使用Revit与Navisworks完成BIM碰撞检查",
        ),
        (
            "病理技师",
            "负责HE染色切片判读并记录FISH检测结果。",
            "负责HE染色切片判读并记录FISH检测结果",
        ),
        (
            "工艺质量工程师",
            "使用Minitab开展SPC分析并输出PFMEA改进项。",
            "使用Minitab开展SPC分析并输出PFMEA改进项",
        ),
    ],
)
def test_dynamic_requirement_matching_handles_unlisted_industries(
    target_role: str,
    fact: str,
    requirement: str,
):
    source = build_source_bundle(f"项目经历\n专项项目\n{fact}", "", requirement)
    resume = CanonicalResume.model_validate({
        "projects": [{"name": "专项项目", "bullets": [fact]}],
    })
    bindings = bind_resume_evidence(resume, source)

    alignment = assess_jd_requirements(
        requirement,
        target_role,
        resume,
        bindings,
        source,
    )

    assert [item["status"] for item in alignment["requirements"]] == ["supported"]


def test_ownership_and_skill_only_evidence_are_not_overstated():
    cv_text = (
        "工作经历\n甲公司｜产品专员\n参与项目交付。\n"
        "专业技能：SQL"
    )
    jd_text = "岗位职责\n独立主导项目交付\n使用SQL搭建经营分析模型并交付报告"
    source = build_source_bundle(cv_text, "", jd_text)
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "产品专员",
            "bullets": ["参与项目交付。"],
        }],
        "skills": {"items": [{"name": "SQL", "category": "tool"}]},
    })
    bindings = bind_resume_evidence(resume, source)

    alignment = assess_jd_requirements(
        jd_text, "高级产品经理", resume, bindings, source,
    )

    assert [item["status"] for item in alignment["requirements"]] == ["partial", "partial"]


@pytest.mark.parametrize(
    "fact",
    [
        "负责无人机控制系统设计。",
        "完成无菌操作并记录实验结果。",
    ],
)
def test_words_containing_wu_are_not_treated_as_negative_evidence(fact: str):
    source = build_source_bundle(f"项目经历\n专项\n{fact}", "", fact)
    resume = CanonicalResume.model_validate({
        "projects": [{"name": "专项", "bullets": [fact]}],
    })
    bindings = bind_resume_evidence(resume, source)

    alignment = assess_jd_requirements(fact, "", resume, bindings, source)

    assert [item["status"] for item in alignment["requirements"]] == ["supported"]


def test_job_title_with_ying_is_not_extracted_as_a_requirement():
    assert extract_jd_requirements("职位名称\n应用工程师") == []
    assert extract_jd_requirements("职位名称\n开发工程师\n数据分析师") == []


def test_degree_does_not_hide_missing_major_in_composite_requirement():
    source = build_source_bundle(
        "教育经历\n某大学 汉语言文学 本科",
        "",
        "本科及以上学历，计算机相关专业",
    )
    resume = CanonicalResume.model_validate({
        "education": [{"school": "某大学", "major": "汉语言文学", "degree": "本科"}],
    })
    bindings = bind_resume_evidence(resume, source)

    alignment = assess_jd_requirements(
        "本科及以上学历，计算机相关专业",
        "",
        resume,
        bindings,
        source,
    )

    assert [item["status"] for item in alignment["requirements"]] == ["partial"]
    assert any("计算机" in item for item in alignment["requirements"][0]["missing_aspects"])


def test_responsibility_verb_requires_a_real_ownership_boundary():
    source = build_source_bundle("工作经历\n甲公司\n客户沟通", "", "负责客户沟通")
    resume = CanonicalResume.model_validate({
        "experience": [{"organization": "甲公司", "bullets": ["客户沟通"]}],
    })
    bindings = bind_resume_evidence(resume, source)

    alignment = assess_jd_requirements("负责客户沟通", "", resume, bindings, source)

    item = alignment["requirements"][0]
    assert item["status"] == "partial"
    assert "责任边界" in item["missing_aspects"]


def test_explicit_absence_cannot_support_a_tenure_requirement():
    source = build_source_bundle("", "我没有相关经验", "3年以上数据分析经验")
    resume = CanonicalResume.model_validate({
        "additional_sections": {"补充说明": ["我没有相关经验"]},
    })
    bindings = bind_resume_evidence(resume, source)

    alignment = assess_jd_requirements(
        "3年以上数据分析经验", "", resume, bindings, source,
    )

    assert [item["status"] for item in alignment["requirements"]] == ["missing"]


def test_jd_extraction_stops_at_company_profile_and_reserves_hard_requirements():
    jd_text = (
        "任职要求\n熟悉Python\n公司介绍\n年轻开放的创业团队"
    )
    assert extract_jd_requirements(jd_text) == ["熟悉Python"]

    long_jd = "岗位职责\n" + "\n".join(
        f"{index}. 负责业务事项{index}" for index in range(1, 13)
    ) + "\n任职要求\n13. 必须持有注册执业资格证书"
    requirements = extract_jd_requirements(long_jd)
    assert len(requirements) == 10
    assert "必须持有注册执业资格证书" in requirements


def test_long_vague_claim_still_requests_method_and_result():
    fact = (
        "负责客户沟通、需求协调、会议组织、内部协作、资料整理、流程跟进、"
        "跨部门对接、业务情况说明及日常事项处理"
    )
    source = build_source_bundle(f"项目经历\n专项\n{fact}", "", "")
    resume = CanonicalResume.model_validate({
        "projects": [{"name": "专项", "bullets": [fact]}],
    })
    bindings = bind_resume_evidence(resume, source)

    report = build_quality_report(
        source=source, resume=resume, evidence_bindings=bindings,
    )

    gap = report["claim_improvement_opportunities"][0]
    assert gap["missing_dimensions"] == ["方法或过程", "交付物或结果"]


def test_custom_section_paths_and_duplicate_bindings_remain_stable():
    fact = "执行标准流程并记录处理结果"
    source = build_source_bundle(f"标准.A[版]\n{fact}", "", fact)
    resume = CanonicalResume.model_validate({
        "additional_sections": {"标准.A[版]": [fact]},
    })
    bindings = bind_resume_evidence(resume, source)
    assert bindings

    report = build_quality_report(
        source=source,
        resume=resume,
        evidence_bindings=bindings + bindings,
        jd_text=fact,
    )

    assert report["fact_grounding"]["evidence_bound_item_count"] == 1
    assert _statuses(report) == ["supported"]


def test_quality_report_is_exposed_in_user_report_and_reply_details():
    quality_report = {
        "job_alignment": {
            "recommendations": ["当前材料未找到SQL数据看板的直接证据。"],
        },
        "source_preservation": {
            "unrepresented_item_count": 1,
            "unrepresented_items": [{"excerpt": "曾输出异常批次分析报告。"}],
        },
        "fact_grounding": {"unsupported_item_count": 0},
        "follow_up_questions": ["请补充你使用SQL的真实场景和交付结果。"],
    }
    ctx = PipelineContext(
        resume_data={"meta": {"target_role": "数据分析师"}},
        target_role="数据分析师",
        industry="internet",
        user_stage="professional",
        _has_audit=True,
        audit_report={"overall_score": 0, "issues": [], "summary": ""},
        quality_report=quality_report,
    )

    asyncio.run(stage_prepare_report(ctx))

    assert ctx.user_report["quality_report"] == quality_report
    assert ctx.user_report["targeted_suggestions"] == quality_report["job_alignment"]["recommendations"]
    reply = build_reply_text(
        scenario="scenario3",
        industry=ctx.industry,
        user_stage=ctx.user_stage,
        missing_fields=[],
        conflicts=[],
        ocr_warnings=[],
        direction=ctx.user_report["generation_direction"],
        score_total=0,
        targeted_suggestions=ctx.user_report["targeted_suggestions"],
        quality_report=quality_report,
    )
    assert "原始材料中未充分写入成稿的信息（1项）" in reply
    assert "建议补充回答" in reply
    assert "使用SQL的真实场景" in reply
