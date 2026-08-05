"""P0 regressions for factuality, source completeness and user-facing detail."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from docx import Document

from evidence_binding import bind_resume_evidence, enforce_resume_evidence, measure_source_coverage
from resume_composer import compose_from_query, compose_resume
from resume_copilot_pipeline import PipelineContext, _build_llm_reply, stage_classify
from resume_copilot_service import _collect_content_conflicts
from resume_renderer import render_docx
from resume_validator import check_required_fields
from source_adapter import build_source_bundle, candidate_blocks
from v2_pipeline import (
    _compact_canonical,
    _deterministic_fallback,
    _ground_bullets,
    _ground_optimizer_output,
    run_v2_pipeline,
)
from v2_schemas import CanonicalResume, DraftResume, SourceBlock, SourceBundle


def _doc_text(path) -> str:
    document = Document(path)
    values = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            values.extend(cell.text for cell in row.cells)
    return "\n".join(values)


def test_jd_only_request_reaches_framework_without_query_or_cv():
    ctx = PipelineContext(query_text="", cv_text="", jd_text="岗位：病理科技师", has_jd=True)
    classification = SimpleNamespace(
        target_role="病理科技师", industry="医疗", user_stage="job_seeker",
    )
    with patch("resume_copilot_pipeline.classify_resume_request", return_value=classification):
        result = asyncio.run(stage_classify(ctx))
    assert result.scenario == "scenario4"
    assert result.generation_text == ""

    with patch("v2_pipeline.compose_from_query", return_value=CanonicalResume.model_validate({
        "meta": {"target_role": "病理科技师"},
    })):
        generated = run_v2_pipeline("", "", "岗位：病理科技师")
    framework = generated.resume_dict["framework"]
    assert framework["mode"] == "empty_profile"
    assert [item["title"] for item in framework["sections"]] == [
        "基本信息", "个人总结", "教育经历", "工作/实习经历", "项目经历", "专业技能",
    ]


def test_query_embedded_jd_and_bare_role_are_not_candidate_facts():
    source = build_source_bundle(
        "",
        "请根据JD生成简历：负责用户调研与竞品分析\n数据分析师",
        "",
    )
    assert candidate_blocks(source) == []
    assert candidate_blocks(build_source_bundle("", "保留原学校", "")) == []


def test_structured_query_sections_remain_candidate_facts():
    source = build_source_bundle(
        "",
        "工作经历\n甲公司｜产品经理｜2022.01-2024.01\n负责用户调研并输出PRD",
        "",
    )
    facts = [block.text for block in candidate_blocks(source)]
    assert "甲公司｜产品经理｜2022.01-2024.01" in facts
    assert "负责用户调研并输出PRD" in facts
    assert "工作经历" not in facts


def test_no_cv_long_candidate_query_uses_chunked_path_without_2000_char_cut():
    marker = "唯一证书：注册安全工程师"
    query = "工作经历\n甲公司｜工程师｜2020.01-2024.01\n" + ("负责现场巡检与问题闭环。" * 220) + "\n" + marker
    captured = []

    def fake_compose(source):
        captured.extend(block.text for block in candidate_blocks(source))
        return DraftResume(additional_sections={"证书": [marker]})

    with patch("resume_composer.llm_enabled", return_value=True), patch(
        "resume_composer.compose_resume", side_effect=fake_compose,
    ):
        result = compose_from_query(query, "目标岗位：安全工程师")
    assert marker in "\n".join(captured)
    assert result.additional_sections["证书"] == [marker]


def test_sparse_no_cv_profile_keeps_explicit_domain_and_target_role():
    query = "我是做智能硬件产品的，这是一个IoT智能硬件产品经理的岗位JD，帮我优化。"
    result = _deterministic_fallback("", query, "负责智能硬件产品规划")
    assert result.meta.target_role == "IoT智能硬件产品经理"
    assert any(item.name == "智能硬件产品" for item in result.skills.items)


def test_composer_discards_partial_result_when_any_fact_chunk_is_empty():
    chunks = [
        SourceBundle(blocks=[SourceBlock(block_id="resume_0", source_type="resume", text="甲公司｜工程师")]),
        SourceBundle(blocks=[SourceBlock(block_id="resume_1", source_type="resume", text="乙公司｜工程师")]),
    ]
    responses = [
        {"experience": [{"organization": "甲公司", "role": "工程师"}]},
        {},
    ]
    with patch("resume_composer.llm_enabled", return_value=True), patch(
        "resume_composer._split_source_bundle", return_value=chunks,
    ), patch("resume_composer.call_llm_typed", side_effect=responses):
        result = compose_resume(SourceBundle(blocks=chunks[0].blocks + chunks[1].blocks))
    assert result == DraftResume()


def test_fabricated_bullet_with_short_shared_prefix_cannot_survive():
    source = "负责客户沟通、资料整理和流程跟进"
    resume = CanonicalResume.model_validate({
        "experience": [{
            "bullets": ["负责客户沟通，制定年度品牌战略并推动全国渠道增长"],
        }],
    })
    grounded = _ground_bullets(resume, source)
    rendered = "\n".join(grounded.experience[0].bullets)
    assert "年度品牌战略" not in rendered
    assert "全国渠道增长" not in rendered


def test_cross_company_identity_and_bullet_splice_is_removed_with_or_without_dates():
    for source_text in (
        "工作经历\n甲公司｜产品经理｜2020.01-2022.01\n负责需求分析\n"
        "乙公司｜运营经理｜2022.02-2024.01\n负责活动运营",
        "工作经历\n甲公司\n产品经理\n负责需求分析\n乙公司\n运营经理\n负责活动运营",
    ):
        source = build_source_bundle(source_text, "", "")
        resume = CanonicalResume.model_validate({
            "experience": [{
                "organization": "甲公司",
                "role": "运营经理",
                "bullets": ["负责活动运营"],
            }],
        })
        gated, _bindings, removed = enforce_resume_evidence(resume, source)
        assert gated.experience == []
        assert "experience[0]" in removed


def test_long_ocr_line_coverage_detects_each_omitted_fact():
    source = build_source_bundle(
        "工作经历\n甲公司｜产品经理｜2020.01-2024.01\n"
        "负责客户沟通、用户调研、竞品分析、输出PRD、推动研发上线、分析数据复盘",
        "",
        "",
    )
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司", "role": "产品经理", "period": "2020.01-2024.01",
            "bullets": ["负责客户沟通"],
        }],
    })
    bindings = bind_resume_evidence(resume, source)
    coverage, missing = measure_source_coverage(source, bindings)
    assert coverage < 0.60
    assert len([item for item in missing if item.startswith("resume_2#u")]) >= 5


def test_lossless_fallback_preserves_all_records_and_section_types():
    education = "\n".join(
        f"第{i}学校｜专业{i}｜本科｜20{i:02d}.09-20{i + 4:02d}.06" for i in range(1, 6)
    )
    experience = "\n".join(
        f"第{i}公司｜工程师｜20{i:02d}.01-20{i:02d}.12\n负责事项{i}并完成交付{i}"
        for i in range(1, 8)
    )
    source = (
        f"教育经历\n{education}\n工作经历\n{experience}\n"
        "科研经历\n某大学实验室｜医学影像课题｜2023.01-2024.01\n完成实验记录\n"
        "校园经历\n学生会｜宣传部长｜2022.01-2022.12\n完成活动宣传\n"
        "项目经历\n病历质量项目｜2024.01-2024.06\n抽查病历并完成问题清单"
    )
    result = _deterministic_fallback(source, "", "")
    assert len(result.education) == 5
    assert len(result.experience) == 7
    assert len(result.research) == 1
    assert len(result.activities) == 1
    assert len(result.projects) == 1
    assert result.experience[-1].role == "工程师"
    assert "负责事项7" in result.experience[-1].bullets[0]


def test_fallback_structures_ocr_compact_teacher_history_without_leaking_directions():
    source = (
        "王宁简历\n"
        "华东师范大学硕士教育学 09-2017-06-2020\n"
        "09-2020- 06-2025上海市第一实验学校语文教师，"
        "负责初中语文授课、班级管理和校本教研。\n"
        "王宁|13710003003|wangning@example.com|3年经验|女\n"
        "候选人具备清晰的问题拆解和执行闭环能力，"
        "过往经历以真实岗位职责和结果为准。\n"
        "03-2019- 06-2019上海市育才中学教育实习，"
        "参与课堂观察和作业批改。"
    )

    result = _deterministic_fallback(
        source,
        "请优化为初中语文老师岗位，保留原学校。",
        "",
    )

    assert result.meta.name == "王宁"
    assert result.meta.target_role == "初中语文老师"
    assert result.education[0].school == "华东师范大学"
    assert result.education[0].major == "教育学"
    assert [item.organization for item in result.experience] == [
        "上海市第一实验学校",
        "上海市育才中学",
    ]
    bullets = "\n".join(
        bullet for item in result.experience for bullet in item.bullets
    )
    assert "班级管理和校本教研" in bullets
    assert "课堂观察和作业批改" in bullets
    assert "保留原学校" not in bullets
    assert "13710003003" not in bullets


def test_rejected_optimizer_wording_restores_original_bullet_instead_of_dropping_it():
    evidence = (
        "03-2019- 06-2019上海市育才中学教育实习，"
        "参与课堂观察和作业批改。"
    )
    original = CanonicalResume.model_validate({
        "experience": [{
            "organization": "上海市育才中学",
            "role": "教育实习",
            "period": "03-2019- 06-2019",
            "bullets": ["参与课堂观察和作业批改。"],
        }],
    })
    optimized = original.model_copy(deep=True)
    optimized.experience[0].bullets = [
        "协助开展课堂观察记录与作业批改工作，支持日常教学运行。"
    ]

    grounded = _ground_optimizer_output(original, optimized, evidence)

    assert grounded.experience[0].bullets == ["参与课堂观察和作业批改。"]


def test_grounded_original_summary_keeps_unique_fact_and_stays_complete():
    text = (
        "个人总结\n拥有8年三甲医院临床经验，完成急重症诊疗与病例复核。\n"
        "工作经历\n甲医院｜主治医师｜2016.01-2024.01\n完成急重症诊疗与病例复核"
    )
    source = build_source_bundle(text, "", "")
    resume = CanonicalResume.model_validate({
        "summary": "拥有8年三甲医院临床经验，完成急重症诊疗与病例复核。",
        "experience": [{
            "organization": "甲医院", "role": "主治医师", "period": "2016.01-2024.01",
            "bullets": ["完成急重症诊疗与病例复核"],
        }],
    })
    gated, _bindings, _removed = enforce_resume_evidence(resume, source)
    compacted = _compact_canonical(gated)
    assert "8年三甲医院临床经验" in compacted.summary
    assert len(compacted.summary) <= 100
    assert compacted.summary.endswith("。")


def test_summary_formats_machine_role_slug_and_bare_seniority_for_people():
    resume = CanonicalResume.model_validate({
        "meta": {"target_role": "product_pm", "work_experience": "4年"},
        "experience": [{
            "organization": "甲公司",
            "role": "产品经理",
            "bullets": ["负责需求分析和版本规划"],
        }],
    })

    compacted = _compact_canonical(resume)

    assert "Product PM" in compacted.summary
    assert "4年经验" in compacted.summary
    assert "product_pm" not in compacted.summary


def test_new_required_rules_do_not_require_seniority_and_count_campus():
    project_profile = {
        "meta": {"name": "李明", "phone": "13812345678", "email": "li@example.com"},
        "summary": "求职方向为产品经理。",
        "education": [{"school": "某大学", "degree": "本科", "major": "管理", "period": "2020-2024"}],
        "projects": [{"name": "调研项目", "period": "2023-2024", "bullets": ["完成用户调研报告"]}],
        "skills": {"others": ["用户调研"]},
    }
    fields = {item.field for item in check_required_fields(project_profile, user_stage="experienced")}
    assert "meta.work_experience" not in fields

    campus_profile = {
        **project_profile,
        "projects": [],
        "campus_experience": [{
            "company": "学生会", "role": "宣传部长", "period": "2022-2023",
            "bullets": ["完成活动宣传"],
        }],
    }
    fields = {item.field for item in check_required_fields(campus_profile, user_stage="student")}
    assert "experience/projects/campus" not in fields


def test_production_conflicts_and_llm_reply_always_include_confirmation_and_framework_modules():
    resume_data = {
        "meta": {"target_role": "产品经理"},
        "summary": "求职方向为产品经理。",
        "experience": [
            {"company": "甲公司", "role": "产品经理", "period": "2020.01-2022.12", "bullets": ["完成需求文档"]},
            {"company": "乙公司", "role": "产品经理", "period": "2022.01-2023.12", "bullets": ["完成产品上线"]},
        ],
    }
    conflicts = [item.model_dump() for item in _collect_content_conflicts(resume_data, "产品经理")]
    assert any("甲公司" in item["description"] and "乙公司" in item["description"] for item in conflicts)

    framework = {
        "framework": {
            "mode": "empty_profile",
            "sections": [
                {"title": "基本信息"}, {"title": "个人总结"}, {"title": "教育经历"},
                {"title": "工作/实习经历"}, {"title": "项目经历"}, {"title": "专业技能"},
            ],
        },
    }
    with patch("resume_copilot_pipeline.ENABLE_LLM_REPLY", True), patch(
        "resume_copilot_pipeline.llm_enabled", return_value=True,
    ), patch("resume_copilot_pipeline.call_llm_text", return_value="我先给你整理了一版。"):
        reply = _build_llm_reply(
            audit_report={}, score=0, missing_fields=[], changes=[],
            resume_data=framework, conflicts=conflicts,
            direction="根据目标JD搭建待填写结构", framework_mode=True,
        )
    assert "待填写框架" in reply
    assert "已生成模块：基本信息、个人总结、教育经历、工作/实习经历、项目经历、专业技能" in reply
    assert "需要确认的时间或内容冲突" in reply
    assert "甲公司" in reply and "乙公司" in reply


def test_minimal_and_custom_docx_keep_cross_industry_sections(tmp_path):
    payload = {
        "meta": {"name": "李明"},
        "summary": "求职方向为临床研究。",
        "education": [{"school": "某医科大学", "degree": "硕士", "major": "临床医学", "period": "2020-2023"}],
        "research": [{"company": "某实验室", "role": "影像研究", "period": "2022-2023", "bullets": ["完成影像标注"]}],
        "campus_experience": [{"company": "志愿协会", "role": "志愿者", "period": "2021", "bullets": ["完成义诊支持"]}],
        "skills": {"domains": ["临床研究"]},
        "publications": ["医学影像研究，第一作者，2023"],
    }
    minimal = tmp_path / "minimal.docx"
    with patch("resume_renderer.inspect_docx_layout", return_value={"available": False, "issues": []}):
        render_docx(payload, minimal, template="minimal")
    minimal_text = _doc_text(minimal)
    for expected in ("某医科大学", "某实验室", "志愿协会", "临床研究", "医学影像研究"):
        assert expected in minimal_text

    template = tmp_path / "custom-template.docx"
    template_doc = Document()
    template_doc.add_paragraph("{{name}}")
    template_doc.save(template)
    custom = tmp_path / "custom.docx"
    with patch("resume_renderer.inspect_docx_layout", return_value={"available": False, "issues": []}):
        render_docx(payload, custom, template=str(template))
    custom_text = _doc_text(custom)
    for expected in ("某医科大学", "某实验室", "志愿协会", "临床研究", "医学影像研究"):
        assert expected in custom_text
