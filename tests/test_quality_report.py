"""Regression tests for deterministic, score-free output quality reporting."""

from __future__ import annotations

import asyncio

import pytest

from evidence_binding import bind_resume_evidence
from quality_report import (
    _requirement_aspects,
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
                path="projects[0].bullets[1]",
                action="remove",
                reason="No candidate evidence binding",
            ),
            Change(
                path="summary[0]",
                action="remove",
                reason="No candidate evidence binding",
            ),
            Change(
                path="meta.name",
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
    assert partial["missing_aspects"] == ["竞品分析", "需求优先级管理"]
    assert any(
        "当前已有部分证据" in question
        and "竞品分析" in question
        and "需求优先级管理" in question
        for question in report["follow_up_questions"]
    )


def test_short_jd_document_title_is_not_reported_as_a_requirement():
    jd_text = (
        "design JD\n"
        "岗位职责\n"
        "1. 负责SaaS后台界面设计与组件规范维护\n"
        "任职要求\n"
        "2. 熟悉Figma"
    )

    assert extract_jd_requirements(jd_text) == [
        "负责SaaS后台界面设计与组件规范维护",
        "熟悉Figma",
    ]
    assert extract_jd_requirements(
        "Senior Product Manager JD\n任职要求\n熟悉SQL"
    ) == ["熟悉SQL"]
    assert extract_jd_requirements(
        "design\n岗位职责\n负责组件规范维护"
    ) == ["负责组件规范维护"]


def test_composite_jd_facets_can_be_supported_across_grounded_claims():
    cv_text = (
        "工作经历\n甲公司｜销售顾问\n"
        "负责客户拓展、商机跟进和回款管理。\n"
        "乙公司｜售前顾问\n负责方案演示与投标支持。"
    )
    requirement = "负责客户拓展、商机跟进以及回款管理，负责方案演示与投标支持"
    source = build_source_bundle(cv_text, "", requirement)
    resume = CanonicalResume.model_validate({
        "experience": [
            {
                "organization": "甲公司",
                "role": "销售顾问",
                "bullets": ["负责客户拓展、商机跟进和回款管理。"],
            },
            {
                "organization": "乙公司",
                "role": "售前顾问",
                "bullets": ["负责方案演示与投标支持。"],
            },
        ],
    })

    alignment = assess_jd_requirements(
        requirement, "销售顾问", resume, bind_resume_evidence(resume, source), source,
    )

    assert [item["status"] for item in alignment["requirements"]] == ["supported"]
    assert alignment["requirements"][0]["missing_aspects"] == []


@pytest.mark.parametrize("fact", ["参与项目交付", "涉及数据分析", "负责组织与会人员签到"])
def test_chinese_conjunction_splitting_preserves_lexical_words(fact: str):
    source = build_source_bundle(f"工作经历\n甲公司｜项目专员\n{fact}", "", fact)
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "项目专员",
            "bullets": [fact],
        }],
    })

    alignment = assess_jd_requirements(
        fact, "项目专员", resume, bind_resume_evidence(resume, source), source,
    )

    assert [item["status"] for item in alignment["requirements"]] == ["supported"]


@pytest.mark.parametrize(
    "requirement",
    [
        "负责数据分析、维护和平谈判",
        "负责工艺记录、开展中和反应",
        "负责材料审核、组织与会人员签到",
    ],
)
def test_enumerated_conjunction_does_not_split_lexical_compounds(requirement: str):
    aspects = _requirement_aspects(requirement)

    assert len(aspects) == 2
    assert "；" not in aspects[-1][0]


def test_spaced_numeric_unit_is_not_reported_as_missing_source_fact():
    source = build_source_bundle(
        "张晨 | 4年经验 | 13800000000\n工作经历\n甲公司｜产品经理｜2022-至今",
        "",
        "",
    )
    resume = CanonicalResume.model_validate({
        "meta": {
            "name": "张晨",
            "work_experience": "4 年经验",
            "phone": "13800000000",
        },
        "experience": [{
            "organization": "甲公司",
            "role": "产品经理",
            "period": "2022-至今",
        }],
    })

    report = build_quality_report(
        source=source,
        resume=resume,
        evidence_bindings=bind_resume_evidence(resume, source),
    )

    excerpts = [
        item["excerpt"]
        for item in report["source_preservation"]["unrepresented_items"]
    ]
    assert "4年经验" not in excerpts


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


def test_claim_improvement_names_exact_record_and_dimension_questions():
    fact = "负责客户反馈整理、竞品分析和需求文档维护"
    source = build_source_bundle(
        f"工作经历\n星河科技｜产品助理｜2020.07-2022.06\n{fact}",
        "",
        "",
    )
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "星河科技",
            "role": "产品助理",
            "period": "2020.07-2022.06",
            "bullets": [fact],
        }],
    })

    report = build_quality_report(
        source=source,
        resume=resume,
        evidence_bindings=bind_resume_evidence(resume, source),
    )

    gap = report["claim_improvement_opportunities"][0]
    assert gap["record_label"] == "星河科技｜产品助理"
    assert "具体通过哪些步骤、工具或协作方式完成" in gap["question"]
    assert "最终产出了什么、如何验收" in gap["question"]
    assert "只填写真实发生且能够核验的信息" in gap["question"]


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


def test_reply_expands_jd_gaps_star_dimensions_and_all_reported_source_omissions():
    quality_report = {
        "job_alignment": {
            "has_job_description": True,
            "supported_requirement_count": 1,
            "partial_requirement_count": 1,
            "missing_requirement_count": 1,
            "requirements": [
                {
                    "requirement": "负责经营分析并输出看板",
                    "status": "partial",
                    "missing_aspects": ["交付物或结果"],
                },
                {
                    "requirement": "熟悉SQL",
                    "status": "missing",
                    "missing_aspects": ["SQL使用场景"],
                },
            ],
            "recommendations": ["若确有SQL实践，请补充真实使用场景和交付结果。"],
        },
        "source_preservation": {
            "unrepresented_item_count": 13,
            "unrepresented_items": [
                {"excerpt": f"原始事实{index}"} for index in range(1, 14)
            ],
        },
        "claim_improvement_opportunities": [{
            "record_label": "甲公司｜数据分析师",
            "excerpt": "负责数据整理",
            "missing_dimensions": ["方法或过程", "交付物或结果"],
        }],
        "fact_grounding": {"unsupported_item_count": 0},
        "follow_up_questions": [],
    }

    reply = build_reply_text(
        scenario="scenario3",
        industry="互联网",
        user_stage="职场人士",
        missing_fields=[],
        conflicts=[],
        ocr_warnings=[],
        direction="突出数据分析相关经历",
        score_total=0,
        targeted_suggestions=quality_report["job_alignment"]["recommendations"],
        quality_report=quality_report,
        resume_data={
            "experience": [{"bullets": ["负责数据整理"]}],
            "skills": {"languages": ["Python"]},
        },
    )

    assert "1项有直接证据，1项仅部分匹配，1项尚无直接证据" in reply
    assert "部分匹配：负责经营分析并输出看板；需补：交付物或结果" in reply
    assert "未匹配：熟悉SQL；需补：SQL使用场景" in reply
    assert "原始事实13" in reply
    assert "经历表达仍可补充" in reply
    assert "甲公司｜数据分析师" in reply
    assert "方法或过程、交付物或结果" in reply
