"""Regression coverage for the deterministic V2 Verifier fast path."""

from unittest.mock import patch

from evidence_binding import bind_resume_evidence, measure_source_coverage
from source_adapter import build_source_bundle
from v2_pipeline import _deterministic_verify_draft, run_v2_pipeline
from v2_schemas import CanonicalResume, DraftResume, VerifiedResult


PRODUCT_CV = """张晨简历
个人信息
张晨 | 13810001001 | zhangchen@example.com | 4年经验 | 男
个人总结
候选人具备清晰的问题拆解和执行闭环能力，过往经历以真实岗位职责和结果为准。
教育背景
复旦大学本科计算机科学 09-2016 - 06-2020
经历
07-2022 - 05-2026 第四范式产品经理，负责企业数据平台需求调研、版本规划与跨团队推进，推动报表配置效率提升30%。
07-2020 - 06-2022 星河科技产品助理，负责客户反馈整理、竞品分析和需求文档维护。
技能
SQL、Axure、Figma、数据分析、项目管理"""


def _product_draft() -> DraftResume:
    return DraftResume.model_validate({
        "meta": {
            "name": "张晨",
            "phone": "13810001001",
            "email": "zhangchen@example.com",
            "work_experience": "4年经验",
            "target_role": "产品经理",
        },
        "education": [{
            "school": "复旦大学",
            "degree": "本科",
            "major": "计算机科学",
            "period": "09-2016 - 06-2020",
        }],
        "experience": [{
            "organization": "第四范式",
            "role": "产品经理",
            "period": "07-2022 - 05-2026",
            "bullets": [
                "负责企业数据平台需求调研、版本规划与跨团队推进，推动报表配置效率提升30%。",
            ],
        }, {
            "organization": "星河科技",
            "role": "产品助理",
            "period": "07-2020 - 06-2022",
            "bullets": ["负责客户反馈整理、竞品分析和需求文档维护。"],
        }],
        "skills": {"items": [
            {"name": name, "category": "other"}
            for name in ("SQL", "Axure", "Figma", "数据分析", "项目管理")
        ]},
    })


def test_complete_split_fields_skip_llm_verifier_despite_low_raw_coverage():
    source = build_source_bundle(
        PRODUCT_CV,
        "请根据目标JD优化我的简历，突出B端产品和数据分析能力。",
        "",
    )
    draft = _product_draft()

    # This documents the original false negative: names can bind to their
    # first duplicate occurrence, while education/job header lines are split
    # into multiple canonical fields. The strict per-claim metric remains
    # below the fast-path threshold even though every required field exists.
    raw_resume = CanonicalResume.model_validate(draft.model_dump())
    raw_coverage, _ = measure_source_coverage(
        source,
        bind_resume_evidence(raw_resume, source),
    )
    assert raw_coverage < 0.80

    result = _deterministic_verify_draft(source, draft)
    assert result is not None
    assert len(result.resume.experience) == 2
    assert result.resume.experience[0].organization == "第四范式"
    assert result.resume.experience[0].period == "07-2022 - 05-2026"
    assert any("30%" in bullet for bullet in result.resume.experience[0].bullets)


def test_final_coverage_keeps_complete_structured_result_instead_of_raw_fallback():
    draft = _product_draft()
    with patch("v2_pipeline.compose_resume", return_value=draft), patch(
        "v2_pipeline._needs_optimizer", return_value=False
    ), patch("v2_pipeline._deterministic_fallback") as fallback:
        result = run_v2_pipeline(
            PRODUCT_CV,
            "请根据目标JD优化我的简历，突出B端产品和数据分析能力。",
            "",
        )

    fallback.assert_not_called()
    assert len(result.resume.experience) == 2
    assert result.resume.education[0].school == "复旦大学"
    assert any("30%" in bullet for bullet in result.resume.experience[0].bullets)


def test_high_coverage_source_fallback_skips_verifier_that_would_be_discarded():
    draft = _product_draft()
    fallback_result = VerifiedResult(
        resume=CanonicalResume.model_validate(draft.model_dump()),
    )
    with patch("v2_pipeline.compose_resume", return_value=draft), patch(
        "v2_pipeline._deterministic_verify_draft", return_value=None
    ), patch("v2_pipeline._needs_optimizer", return_value=False), patch(
        "v2_pipeline._grounded_source_fallback",
        return_value=(fallback_result, 0.95, []),
    ), patch(
        "v2_pipeline.verify_resume"
    ) as llm_verifier:
        result = run_v2_pipeline(
            PRODUCT_CV,
            "请根据目标JD优化我的简历，突出B端产品和数据分析能力。",
            "",
        )

    llm_verifier.assert_not_called()
    assert len(result.resume.experience) == 2
    assert result.evidence_bindings
    assert result.changes[0].reason == (
        "Used high-coverage deterministic parser before LLM verification"
    )


def test_high_coverage_raw_extras_do_not_replace_a_structured_verifier_result():
    draft = _product_draft()
    unstructured = VerifiedResult(resume=CanonicalResume.model_validate({
        "additional_sections": {"待整理的原始信息": [PRODUCT_CV]},
    }))
    verified = VerifiedResult(
        resume=CanonicalResume.model_validate(draft.model_dump()),
    )
    with patch("v2_pipeline.compose_resume", return_value=draft), patch(
        "v2_pipeline._deterministic_verify_draft", return_value=None
    ), patch("v2_pipeline._needs_optimizer", return_value=False), patch(
        "v2_pipeline._grounded_source_fallback",
        return_value=(unstructured, 1.0, []),
    ), patch(
        "v2_pipeline.verify_resume",
        return_value=verified,
    ) as llm_verifier:
        result = run_v2_pipeline(PRODUCT_CV, "请优化", "")

    llm_verifier.assert_called_once()
    assert len(result.resume.experience) == 2


def test_ocr_compound_lines_and_edit_direction_do_not_force_llm_verifier():
    source = build_source_bundle(
        "王宁简历\n"
        "华东师范大学硕士教育学 09-2017-06-2020\n"
        "09-2020-06-2025上海市第一实验学校语文教师，"
        "负责初中语文授课、班级管理和校本教研。\n"
        "王宁|13710003003|wangning@example.com|3年经验|女\n"
        "候选人具备清晰的问题拆解和执行闭环能力，"
        "过往经历以真实岗位职责和结果为准。",
        "请优化为初中语文老师岗位，保留原学校。",
        "",
    )
    draft = DraftResume.model_validate({
        "meta": {
            "name": "王宁",
            "phone": "13710003003",
            "email": "wangning@example.com",
            "work_experience": "3年经验",
            "target_role": "初中语文老师",
        },
        "education": [{
            "school": "华东师范大学",
            "degree": "硕士",
            "major": "教育学",
            "period": "09-2017-06-2020",
        }],
        "experience": [{
            "organization": "上海市第一实验学校",
            "role": "语文教师",
            "period": "09-2020-06-2025",
            "bullets": ["负责初中语文授课、班级管理和校本教研。"],
        }],
    })

    result = _deterministic_verify_draft(source, draft)
    assert result is not None
    assert result.resume.education[0].school == "华东师范大学"
    assert result.resume.experience[0].organization == "上海市第一实验学校"


def test_fastpath_still_rejects_material_source_omissions():
    source = build_source_bundle(
        "姓名：李明\n教育经历\n华南理工大学 材料科学与工程 本科\n"
        "项目经历\n涂布优化项目\n记录12批次生产参数。\n"
        "分析异常批次原因。\n输出工艺改进建议。",
        "",
        "",
    )
    draft = DraftResume.model_validate({
        "meta": {"name": "李明"},
        "education": [{
            "school": "华南理工大学",
            "major": "材料科学与工程",
            "degree": "本科",
        }],
        "projects": [{
            "name": "涂布优化项目",
            "bullets": ["记录12批次生产参数。"],
        }],
    })

    assert _deterministic_verify_draft(source, draft) is None


def test_compound_header_cannot_hide_an_omitted_organization_or_date():
    source = build_source_bundle(
        "姓名：李明\n工作经历\n2020.01-2022.01 甲公司 产品经理\n"
        "负责需求分析并输出PRD。",
        "",
        "",
    )
    common = {
        "meta": {"name": "李明"},
        "experience": [{
            "organization": "甲公司",
            "role": "产品经理",
            "period": "2020.01-2022.01",
            "bullets": ["负责需求分析并输出PRD。"],
        }],
    }
    for omitted_field in ("organization", "period"):
        data = DraftResume.model_validate(common).model_dump()
        data["experience"][0][omitted_field] = ""
        assert _deterministic_verify_draft(
            source,
            DraftResume.model_validate(data),
        ) is None


def test_fastpath_never_retains_fabricated_number_organization_or_date():
    source = build_source_bundle(PRODUCT_CV, "", "")
    draft_data = _product_draft().model_dump()
    draft_data["experience"].append({
        "organization": "新增算法公司",
        "role": "算法工程师",
        "period": "01-2099 - 12-2099",
        "bullets": ["主导算法平台建设并提升效率99%。"],
    })

    result = _deterministic_verify_draft(source, DraftResume.model_validate(draft_data))
    assert result is not None
    serialized = result.resume.model_dump_json()
    assert "新增算法公司" not in serialized
    assert "算法工程师" not in serialized
    assert "2099" not in serialized
    assert "99%" not in serialized
