"""Cross-industry regressions for source-record and output integrity."""

from __future__ import annotations

from pathlib import Path

import pytest

from evidence_binding import (
    bind_resume_evidence,
    enforce_resume_evidence,
    measure_source_coverage,
    source_fact_units,
)
from resume_io import _extract_text_from_pdf_bytes, extract_text_from_bytes
from resume_validator import check_required_fields, check_time_conflicts
from source_adapter import build_source_bundle
from v2_pipeline import (
    _compact_canonical,
    _deterministic_fallback,
    _recover_grounded_source_structure,
)
from v2_schemas import CanonicalResume, SourceBlock, SourceBundle


def _gate(cv_text: str, resume_data: dict) -> tuple[CanonicalResume, list[str]]:
    source = build_source_bundle(cv_text, "", "")
    resume = CanonicalResume.model_validate(resume_data)
    gated, _, removed = enforce_resume_evidence(resume, source)
    return gated, removed


def test_generic_experience_heading_preserves_cross_industry_roles_and_major():
    cv_text = (
        "王芳\n"
        "教育背景\n"
        "复旦大学 本科 市场营销 2018-2022\n"
        "经历\n"
        "2022.07-至今 某科技公司 用户运营，负责用户访谈和活动运营\n"
        "2021.06-2021.09 某互联网公司 内容运营，参与公众号选题与数据复盘\n"
        "技能\n"
        "Excel、用户研究"
    )
    source = build_source_bundle(cv_text, "", "")
    by_text = {block.text: block for block in source.blocks}

    assert by_text["经历"].section_hint == "experience"
    assert by_text["技能"].section_hint == "skills"
    assert by_text["Excel、用户研究"].section_hint == "skills"
    assert by_text["2022.07-至今 某科技公司 用户运营，负责用户访谈和活动运营"].record_id != (
        by_text["2021.06-2021.09 某互联网公司 内容运营，参与公众号选题与数据复盘"].record_id
    )

    gated, removed = _gate(cv_text, {
        "education": [{
            "school": "复旦大学", "degree": "本科", "major": "市场营销",
            "period": "2018-2022",
        }],
        "experience": [
            {
                "organization": "某科技公司", "role": "用户运营",
                "period": "2022.07-至今",
                "bullets": ["负责用户访谈和活动运营"],
            },
            {
                "organization": "某互联网公司", "role": "内容运营",
                "period": "2021.06-2021.09",
                "bullets": ["参与公众号选题与数据复盘"],
            },
        ],
    })

    assert gated.education[0].major == "市场营销"
    assert [item.role for item in gated.experience] == ["用户运营", "内容运营"]
    assert not any(path.endswith(".role") or path.endswith(".major") for path in removed)


def test_generic_experience_heading_supports_school_employers():
    cv_text = (
        "教育背景\n北京师范大学 本科 汉语言文学 2018-2022\n"
        "经历\n"
        "2022.09-至今 第一中学 语文教师，负责初中语文授课与班级管理\n"
        "2021.09-2021.12 第二中学 教育实习，参与备课和课堂教学\n"
        "技能\n普通话一级乙等"
    )
    source = build_source_bundle(cv_text, "", "")
    experience = [block for block in source.blocks if block.section_hint == "experience"]

    assert [block.text for block in experience] == [
        "经历",
        "2022.09-至今 第一中学 语文教师，负责初中语文授课与班级管理",
        "2021.09-2021.12 第二中学 教育实习，参与备课和课堂教学",
    ]
    assert experience[1].record_id != experience[2].record_id


def test_structural_role_gate_accepts_header_role_but_rejects_duty_noun():
    source = SourceBundle(blocks=[SourceBlock(
        block_id="resume_0",
        source_type="resume",
        # Deliberately wrong section reproduces a layout-parser leak.
        section_hint="education",
        record_id="resume:education:0",
        text="2019.07-至今 北京协和医院 内科医师，负责患者诊疗和单元测试",
    )])
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "北京协和医院",
            "role": "内科医师",
            "period": "2019.07-至今",
            "bullets": ["负责患者诊疗和单元测试"],
        }],
    })
    gated, _, removed = enforce_resume_evidence(resume, source)

    assert gated.experience[0].role == "内科医师"
    assert "experience[0].role" not in removed

    false_role = resume.model_copy(deep=True)
    false_role.experience[0].role = "单元测试"
    false_gated, _, false_removed = enforce_resume_evidence(false_role, source)
    assert false_gated.experience[0].role == ""
    assert "experience[0].role" in false_removed


def test_incoherent_project_salvages_anchored_record_and_drops_only_foreign_bullet():
    cv_text = (
        "项目经历\n"
        "- 项目A\n个人项目 | 产品设计 |\n2024.01-2024.06\n"
        "负责需求分析并输出PRD。\n"
        "- 项目B\n个人项目 | 工程开发 |\n2025.01-2025.06\n"
        "完成系统上线并交付。"
    )
    resume = CanonicalResume.model_validate({
        "projects": [{
            "name": "项目A",
            "period": "2024.01-2024.06",
            "bullets": ["负责需求分析并输出PRD。", "完成系统上线并交付。"],
        }],
    })

    gated, _, removed = enforce_resume_evidence(
        resume,
        build_source_bundle(cv_text, "", ""),
    )

    assert len(gated.projects) == 1
    assert gated.projects[0].name == "项目A"
    assert gated.projects[0].bullets == ["负责需求分析并输出PRD。"]
    assert removed == ["projects[0].bullets[1]"]


def test_speech_docx_builds_three_complete_source_project_records():
    path = Path(__file__).parents[1] / (
        "acceptance_testset/0611case_testset/files/cv/cv_speech_engineer.docx"
    )
    cv_text = extract_text_from_bytes(path.read_bytes(), path.name)
    source = build_source_bundle(cv_text, "", "")
    record_ids = {
        block.record_id
        for block in source.blocks
        if block.section_hint == "projects" and block.record_id
    }
    fallback = _deterministic_fallback(cv_text, "", "")

    assert len(record_ids) == 3
    assert [record.name for record in fallback.projects] == [
        "基于软硬数据融合的多目标跟踪算法",
        "基于CS336 课程：从零实现一个LLM",
        "英雄联盟背景信息RAG 智能助手",
    ]
    assert [record.period for record in fallback.projects] == [
        "2024.6 – 2024.9", "2025.5 – 至今", "2025.7 – 至今",
    ]


def test_native_pdf_and_docx_preserve_the_same_financial_resume_record_anchors():
    cv_root = Path(__file__).parents[1] / "acceptance_testset/0611case_testset/files/cv"
    docx_path = cv_root / "cv_data_engineer.docx"
    pdf_path = cv_root / "cv_financial.pdf"
    docx_text = extract_text_from_bytes(docx_path.read_bytes(), docx_path.name)
    pdf_text = _extract_text_from_pdf_bytes(pdf_path.read_bytes())

    docx_resume = _compact_canonical(_deterministic_fallback(docx_text, "", ""))
    pdf_resume = _compact_canonical(_deterministic_fallback(pdf_text, "", ""))

    expected_experience = [
        ("布里斯托大学", "科研实习", "05/2025 – 至今"),
        ("布里斯托大学", "研究助理", "11/2024 – 03/2025"),
    ]
    expected_projects = [
        ("城市交通流量预测系统", "07/2024 – 10/2024"),
        ("Samsung IT School Java 开发项目", "03/2021 – 05/2021"),
    ]

    for resume in (docx_resume, pdf_resume):
        assert [
            (record.organization, record.role, record.period)
            for record in resume.experience
        ] == expected_experience
        assert [(record.name, record.period) for record in resume.projects] == expected_projects
        assert all("指导老师" not in record.role for record in resume.projects)

    traffic = docx_resume.projects[0]
    assert not any(
        bullet.endswith("城市交通") or bullet.endswith("确保输入数据质量，以提")
        for bullet in traffic.bullets
    )


def test_record_guard_restores_a_missing_uniquely_grounded_project():
    cv_text = (
        "项目经历\n"
        "- 项目A\n个人项目 | 产品设计 |\n2024.01-2024.06\n"
        "负责需求分析并输出PRD。\n"
        "- 项目B\n个人项目 | 工程开发 |\n2025.01-2025.06\n"
        "完成系统上线并交付。"
    )
    source = build_source_bundle(cv_text, "", "")
    baseline = _deterministic_fallback(cv_text, "", "")
    current = baseline.model_copy(deep=True)
    current.projects = current.projects[:1]

    restored, stats = _recover_grounded_source_structure(current, baseline, source)

    assert [record.name for record in restored.projects] == ["项目A", "项目B"]
    assert stats.appended_records == 1


def test_section_headings_are_not_reported_as_missing_source_facts():
    source = build_source_bundle(
        "工作/实习经历\n甲公司｜产品经理｜2022-2024\n负责需求分析\n"
        "研究项目\n- 用户研究项目\n2023.01-2023.06\n完成访谈纪要",
        "",
        "",
    )

    unit_text = [unit["text"] for unit in source_fact_units(source)]

    assert "工作/实习经历" not in unit_text
    assert "研究项目" not in unit_text
    assert any("甲公司" in value for value in unit_text)
    assert any("负责需求分析" in value for value in unit_text)


def test_unheaded_awards_do_not_leak_into_the_preceding_skill_section():
    cv_text = (
        "技能：Axure RP（熟练），Sketch（熟练），用户研究\n"
        "校级一等奖学金\n"
        "优秀学生干部\n"
        "全国大学生创新创业大赛三等奖"
    )
    source = build_source_bundle(cv_text, "", "")
    fallback = _compact_canonical(_deterministic_fallback(cv_text, "", ""))

    assert [block.section_hint for block in source.blocks] == [
        "skills", "awards", "awards", "awards",
    ]
    assert [item.name for item in fallback.skills.items] == [
        "Axure RP", "Sketch", "用户研究",
    ]
    assert fallback.awards == [
        "校级一等奖学金", "优秀学生干部", "全国大学生创新创业大赛三等奖",
    ]


def test_complete_long_summary_is_not_reported_as_missing():
    summary = (
        "具备跨团队产品交付经验，曾围绕真实业务问题完成用户访谈、需求分析与方案设计。"
        "能够协调研发和测试推进版本上线，并根据用户反馈持续跟踪问题闭环。"
        "相关描述均来自已提供经历，未补写未经确认的成果数据。"
        "求职时可根据目标岗位调整相关经历的排序与表达重点。"
    )
    assert len(summary) > 100
    missing = check_required_fields({
        "meta": {
            "name": "李明", "phone": "13812345678", "email": "li@example.com",
            "education_level": "本科",
        },
        "summary": summary,
        "education": [{
            "school": "某大学", "degree": "本科", "major": "管理学",
            "period": "2018-2022",
        }],
        "experience": [{
            "organization": "甲公司", "role": "产品经理", "period": "2022-2024",
            "bullets": ["负责用户访谈并输出需求文档，推动版本上线。"],
        }],
        "skills": {"others": ["需求分析"]},
    })

    assert not any(item.field == "summary" for item in missing)


def test_partial_records_report_exact_fields_without_claiming_section_missing():
    missing = check_required_fields({
        "meta": {
            "name": "某候选人",
            "phone": "13812345678",
            "email": "candidate@example.com",
            "education_level": "硕士",
        },
        "summary": "拥有财务管理与内部审计方面的真实工作背景。",
        "education": [{
            "school": "", "degree": "硕士", "major": "工商管理", "period": "1999年",
        }],
        "experience": [{
            "company": "独立财务咨询公司",
            "role": "",
            "period": "2015年12月至今",
            "bullets": ["制定财务战略和内部审计程序"],
        }],
        "projects": [{
            "name": "",
            "period": "",
            "bullets": ["开发自动化财务预测模型，使处理时间减少40%"],
        }],
        "skills": {"others": ["SAP"]},
    })

    fields = {item.field for item in missing}
    assert "education[0].school" in fields
    assert "experience[0].role" in fields
    assert "projects[0].name" in fields
    assert "projects[0].period" in fields
    assert "experience[0]" not in fields
    assert "projects[0]" not in fields
    assert all("经历" not in item.reason for item in missing)


def test_chinese_december_and_year_only_periods_sort_without_false_conflict():
    resume = {
        "experience": [
            {
                "organization": "独立财务咨询公司",
                "role": "顾问",
                "period": "2015年12月至今",
            },
            {
                "organization": "BUILDTECH SUPPLIES INC.",
                "role": "财务与资源经理",
                "period": "2015年6月至2015年11月",
            },
            {
                "organization": "工业阀门有限公司",
                "role": "财务与系统主管",
                "period": "2014年至2015年",
            },
        ],
    }

    assert check_time_conflicts(resume) == []


def test_student_internship_and_research_overlap_is_not_a_time_conflict():
    resume = {
        "education": [{
            "school": "布里斯托大学", "period": "2023.09-2026.06",
        }],
        "experience": [
            {
                "organization": "布里斯托大学", "role": "研究助理",
                "period": "2024.11-2025.03",
            },
            {
                "organization": "布里斯托大学", "role": "科研实习",
                "period": "2025.02-2025.06",
            },
        ],
    }

    assert check_time_conflicts(resume) == []


def test_medical_residency_overlap_with_degree_is_not_a_time_conflict():
    resume = {
        "meta": {"work_experience": "6年经验"},
        "education": [{
            "school": "北京大学医学部",
            "period": "2016.09-2019.06",
        }],
        "experience": [
            {
                "organization": "北京协和医院",
                "role": "内科医师",
                "period": "2019.07-2026.05",
            },
            {
                "organization": "北京大学第三医院",
                "role": "规培医师",
                "period": "2017.09-2019.06",
            },
        ],
    }

    assert check_time_conflicts(resume) == []


def test_material_tenure_mismatch_is_reported_from_full_time_date_union():
    conflicts = check_time_conflicts({
        "meta": {"work_experience": "4年经验"},
        "experience": [
            {
                "organization": "第四范式",
                "role": "产品经理",
                "period": "2022.07-2026.05",
            },
            {
                "organization": "星河科技",
                "role": "产品助理",
                "period": "2020.07-2022.06",
            },
        ],
    })

    mismatch = [item for item in conflicts if item.field == "meta.work_experience"]
    assert len(mismatch) == 1
    assert "基本信息填写“4年经验”" in mismatch[0].description
    assert "约5.9年" in mismatch[0].description
    assert "请确认是否按相关经验计算" in mismatch[0].description


def test_internship_does_not_create_a_false_tenure_mismatch():
    conflicts = check_time_conflicts({
        "meta": {"work_experience": "5年经验"},
        "experience": [
            {
                "organization": "中国银行",
                "role": "贷款管理岗",
                "period": "2020.03-2025.06",
            },
            {
                "organization": "中信证券",
                "role": "实习生",
                "period": "2019.06-2020.02",
            },
        ],
    })

    assert not any(item.field == "meta.work_experience" for item in conflicts)


def test_lower_bound_tenure_label_is_not_treated_as_an_exact_contradiction():
    conflicts = check_time_conflicts({
        "meta": {"work_experience": "3年以上经验"},
        "experience": [{
            "organization": "甲公司",
            "role": "产品经理",
            "period": "2019.01-2025.12",
        }],
    })

    assert not any(item.field == "meta.work_experience" for item in conflicts)


def test_two_overlapping_full_time_roles_still_require_confirmation():
    conflicts = check_time_conflicts({
        "experience": [
            {"organization": "甲公司", "role": "产品经理", "period": "2022.01-2023.06"},
            {"organization": "乙公司", "role": "运营经理", "period": "2023.01-2024.01"},
        ],
    })

    assert len(conflicts) == 1
    assert conflicts[0].field == "experience"


@pytest.mark.parametrize(
    ("stem", "education", "experience", "required_skill"),
    [
        (
            "product_pm", ("复旦大学", "本科", "计算机科学"),
            [("第四范式", "产品经理"), ("星河科技", "产品助理")], "项目管理",
        ),
        (
            "operations", ("武汉大学", "本科", "市场营销"),
            [("云舟科技", "用户运营"), ("南方电商", "内容运营")], "活动运营",
        ),
        (
            "sales", ("南京大学", "本科", "工商管理"),
            [("明源云", "售前顾问"), ("东华软件", "销售经理")], "CRM",
        ),
        (
            "design", ("中国美术学院", "本科", "视觉传达"),
            [("蓝湖科技", "UI设计师"), ("风起互动", "视觉设计师")], "设计系统",
        ),
        (
            "doctor", ("北京大学医学部", "硕士", "临床医学"),
            [("北京协和医院", "内科医师"), ("北京大学第三医院", "规培医师")], "临床诊疗",
        ),
        (
            "teacher", ("华东师范大学", "硕士", "教育学"),
            [("上海市第一实验学校", "语文教师"), ("上海市育才中学", "教育实习")], "课程设计",
        ),
        (
            "finance_bank", ("中央财经大学", "学士", "金融学"),
            [("中国银行", "贷款管理岗"), ("中信证券", "实习生")], "信贷审核",
        ),
    ],
)
def test_compact_cross_industry_docx_keeps_every_structured_record(
    stem: str,
    education: tuple[str, str, str],
    experience: list[tuple[str, str]],
    required_skill: str,
):
    path = Path(__file__).parents[1] / f"acceptance_testset/files/cv/cv_{stem}.docx"
    cv_text = extract_text_from_bytes(path.read_bytes(), path.name)
    source = build_source_bundle(cv_text, "", "")
    fallback = _compact_canonical(_deterministic_fallback(cv_text, "", ""))
    grounded, _, _ = enforce_resume_evidence(fallback, source)
    grounded = _compact_canonical(grounded)
    bindings = bind_resume_evidence(grounded, source)
    coverage, missing = measure_source_coverage(
        source,
        bindings,
        allow_distributed=True,
    )

    assert (
        grounded.education[0].school,
        grounded.education[0].degree,
        grounded.education[0].major,
    ) == education
    assert [
        (record.organization, record.role) for record in grounded.experience
    ] == experience
    assert required_skill in [item.name for item in grounded.skills.items]
    assert not grounded.additional_sections
    assert coverage == 1.0
    assert missing == []
