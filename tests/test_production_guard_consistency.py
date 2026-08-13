from __future__ import annotations

from evidence_binding import bind_resume_evidence
from quality_report import build_quality_report, extract_jd_requirements
from resume_copilot_pipeline import _reply_detail_block
from resume_copilot_service import (
    _canonical_resume_from_render_data,
    _final_evidence_bindings,
    _protected_render_paths,
    final_fact_guard,
)
from resume_validator import check_fabrication_heuristic
from source_adapter import build_source_bundle
from v2_schemas import CanonicalResume, EvidenceBinding
from v2_pipeline import _deterministic_fallback


def _resume_with_period(period: str) -> dict:
    return {
        "meta": {"name": "", "target_role": ""},
        "education": [],
        "experience": [{
            "company": "XX实验小学",
            "role": "语文老师",
            "period": period,
            "bullets": ["负责语文教学。"],
        }],
        "projects": [],
        "skills": {},
    }


def test_yyyy_mm_source_period_is_not_reported_as_fabricated() -> None:
    source = (
        "工作经历\nXX实验小学｜语文老师\n"
        "2022-09 至 2023-01\n负责语文教学。"
    )
    resume = _resume_with_period("2022-09 至 2023-01")

    report = check_fabrication_heuristic(source, resume)

    assert not [item for item in report.details if item.type == "date"]


def test_production_guard_preserves_bound_field_but_removes_unbound_date() -> None:
    source = "XX实验小学语文老师，原始文件中的时间排版无法可靠解析。负责语文教学。"
    supported = _resume_with_period("2022-09 至 2023-01")
    protected = {"experience[0].period"}

    kept, kept_report = final_fact_guard(
        source,
        supported,
        protected_paths=protected,
    )
    removed, removed_report = final_fact_guard(source, supported)

    assert kept["experience"][0]["period"] == "2022-09 至 2023-01"
    assert not kept_report.fabrication_found
    assert removed["experience"][0]["period"] == ""
    assert not removed_report.fabrication_found


def test_canonical_report_is_rebuilt_from_post_guard_render_data() -> None:
    source = build_source_bundle(
        "工作经历\nXX实验小学｜语文老师\n2022-09 至 2023-01\n负责语文教学。",
        "",
        "",
    )
    before = CanonicalResume.model_validate({
        "experience": [{
            "organization": "XX实验小学",
            "role": "语文老师",
            "period": "2022-09 至 2023-01",
            "bullets": ["负责语文教学。"],
        }],
    })
    old_bindings = bind_resume_evidence(before, source)
    render_data = {
        "meta": {},
        "education": [],
        "experience": [{
            "company": "XX实验小学",
            "role": "语文老师",
            "period": "",
            "bullets": ["负责语文教学。"],
        }],
        "projects": [],
        "campus_experience": [],
        "research": [],
        "skills": {},
    }

    final_resume = _canonical_resume_from_render_data(render_data)
    final_bindings = _final_evidence_bindings(final_resume, source, old_bindings)
    report = build_quality_report(
        source=source,
        resume=final_resume,
        evidence_bindings=final_bindings,
    )

    assert final_resume.experience[0].period == ""
    assert "experience[0].period" not in {item.path for item in final_bindings}
    assert report["source_preservation"]["unrepresented_item_count"] > 0


def test_render_skill_buckets_survive_final_canonical_roundtrip() -> None:
    final_resume = _canonical_resume_from_render_data({
        "meta": {},
        "education": [],
        "experience": [],
        "projects": [],
        "research": [],
        "campus_experience": [],
        "skills": {
            "languages": ["Python"],
            "tools": ["Power BI"],
            "domains": ["经营分析"],
        },
    })

    assert [(item.name, item.category) for item in final_resume.skills.items] == [
        ("Python", "language"),
        ("Power BI", "tool"),
        ("经营分析", "domain"),
    ]


def test_v2_evidence_paths_are_mapped_to_render_schema() -> None:
    bindings = [
        EvidenceBinding(
            path="experience[0].organization",
            block_id="resume:experience:0",
            quote="XX实验小学",
            claim="XX实验小学",
            mode="direct",
        ),
        EvidenceBinding(
            path="activities[1].period",
            block_id="resume:activities:1",
            quote="2021-03 至 2021-08",
            claim="2021-03 至 2021-08",
            mode="direct",
        ),
    ]

    assert _protected_render_paths(bindings) == {
        "experience[0].company",
        "campus_experience[1].period",
    }


def test_compact_training_institution_and_assistant_role_are_preserved() -> None:
    source = (
        "工作经历\n"
        "XX教育培训机构助教\n"
        "2020-08 至 2021-02\n"
        "1. 跟踪课程情况，维护课堂纪律。\n"
        "2. 根据教学计划向学生同步课程内容。"
    )

    resume = _deterministic_fallback(source, "", "")
    dated = [item for item in resume.experience if item.period]

    assert len(dated) == 1
    assert dated[0].organization == "XX教育培训机构"
    assert dated[0].role == "助教"


def test_jd_locations_are_not_requirements() -> None:
    jd = (
        "任职要求：\n"
        "1. 具备扎实的教学能力和学科知识\n"
        "2. 具有良好的沟通技巧\n"
        "校区一：杨浦五角场\n"
        "工作地址：浦东南汇"
    )

    requirements = extract_jd_requirements(jd)

    assert any("教学能力" in item for item in requirements)
    assert all("校区" not in item and "工作地址" not in item for item in requirements)


def test_reply_lists_bounded_actionable_gaps() -> None:
    unrepresented = [
        {"excerpt": f"负责第{index}项真实工作并形成交付物", "section_hint": "experience"}
        for index in range(12)
    ]
    unrepresented.extend([
        {"excerpt": "QQ 2778164751", "section_hint": None},
        {"excerpt": "微信 knowpage", "section_hint": None},
        {"excerpt": "求职意向：语文教师", "section_hint": None},
    ])
    quality_report = {
        "source_preservation": {
            "unrepresented_item_count": 15,
            "unrepresented_items": unrepresented,
        },
        "fact_grounding": {"unsupported_item_count": 0},
        "claim_improvement_opportunities": [
            {
                "record_label": "XX公司｜产品经理",
                "excerpt": f"第{index}条经历",
                "missing_dimensions": ["方法或过程"],
            }
            for index in range(6)
        ],
        "follow_up_questions": [f"追问{index}" for index in range(6)],
        "job_alignment": {"has_job_description": False},
    }

    reply = _reply_detail_block([], [], quality_report)

    assert reply.count("负责第") <= 5
    assert reply.count("缺少方法或过程") <= 3
    assert reply.count("- 追问") <= 3
    assert "另有 10 项" in reply
    assert "2778164751" not in reply
    assert "knowpage" not in reply
    assert "求职意向：语文教师" not in reply
