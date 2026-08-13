"""Calibration checks for source-grounded atomic resume auditing."""
from __future__ import annotations

import pytest

from atomic_fact_audit import audit_atomic_facts
from evidence_binding import bind_resume_evidence
from quality_report import build_quality_report
from source_adapter import build_source_bundle
from v2_schemas import CanonicalResume


def _audit(cv_text: str, resume_data: dict, *, query: str = "", jd: str = "") -> dict:
    source = build_source_bundle(cv_text, query, jd)
    resume = CanonicalResume.model_validate(resume_data)
    return audit_atomic_facts(
        source=source,
        resume=resume,
        evidence_bindings=bind_resume_evidence(resume, source),
    )


def test_quality_report_1_1_adds_audits_without_removing_legacy_sections():
    source = build_source_bundle("姓名：李明", "", "")
    resume = CanonicalResume.model_validate({"meta": {"name": "李明"}})
    report = build_quality_report(
        source=source,
        resume=resume,
        evidence_bindings=bind_resume_evidence(resume, source),
    )

    assert report["schema_version"] == "1.1"
    assert "source_preservation" in report
    assert "fact_grounding" in report
    assert "atomic_factuality" in report
    assert "ownership_integrity" in report
    assert "structural_invariants" in report
    assert not any("score" in key for key in report)


def test_atomic_audit_isolates_unsupported_clause_instead_of_condemning_bullet():
    audit = _audit(
        "工作经历\n甲公司｜产品经理｜2022.01-2024.01\n负责用户访谈。",
        {
            "experience": [{
                "organization": "甲公司",
                "role": "产品经理",
                "period": "2022.01-2024.01",
                "bullets": ["负责用户访谈，并提升转化率30%。"],
            }],
        },
        jd="岗位要求：提升转化率30%",
    )

    factuality = audit["atomic_factuality"]
    assert factuality["unsupported_atom_count"] == 1
    assert factuality["unsupported_output"][0]["excerpt"] == "提升转化率30%"
    assert factuality["unsupported_output"][0]["unmatched_anchors"] == ["30"]
    assert audit["structural_invariants"]["metric"]["added"] == [{
        "canonical_field_path": "experience[0].bullets[0]",
        "value": "30%",
    }]


def test_jd_and_direction_query_never_support_candidate_atoms():
    audit = _audit(
        "姓名：李明",
        {
            "meta": {"name": "李明"},
            "skills": {"items": [{"name": "Kubernetes", "category": "tool"}]},
        },
        query="请帮我申请云平台岗位并突出 Kubernetes",
        jd="任职要求：熟悉 Kubernetes",
    )

    unsupported = audit["atomic_factuality"]["unsupported_output"]
    assert [item["excerpt"] for item in unsupported] == ["Kubernetes"]
    assert audit["atomic_factuality"]["source_fact_count"] == 1
    assert audit["structural_invariants"]["skill_tool"]["added_count"] == 1


def test_summary_presentation_shells_do_not_create_false_unsupported_atoms():
    audit = _audit(
        "姓名：张晨｜4年经验\n复旦大学本科计算机科学\n"
        "第四范式产品经理\n技能：SQL",
        {
            "meta": {"name": "张晨", "work_experience": "4年经验"},
            "summary": (
                "拥有4年经验。代表成果：在第四范式担任产品经理期间。"
                "核心技能包括SQL。教育背景为复旦大学。"
                "求职方向为Product PM。"
            ),
            "education": [{"school": "复旦大学", "degree": "本科", "major": "计算机科学"}],
            "experience": [{"organization": "第四范式", "role": "产品经理", "bullets": []}],
            "skills": {"items": [{"name": "SQL", "category": "tool"}]},
        },
        jd="Product PM",
    )

    factuality = audit["atomic_factuality"]
    assert factuality["unsupported_atom_count"] == 0
    assert factuality["precision"] == 1.0


def test_summary_accepts_equivalent_month_padding_and_nested_record_shell():
    audit = _audit(
        "复旦大学市场营销本科在读，09-2022 到 06-2026。\n"
        "在校期间做过社群活动项目。",
        {
            "summary": (
                "复旦大学市场营销专业本科在读（2022年9月至2026年6月）。"
                "代表经历：项目经历（社群活动项目）：在校期间做过社群活动项目。"
                "项目经历包括社群活动项目。"
            ),
            "education": [{
                "school": "复旦大学", "degree": "本科", "major": "市场营销",
                "period": "2022年9月至2026年6月",
            }],
            "projects": [{
                "name": "社群活动项目",
                "bullets": ["在校期间做过社群活动项目。"],
            }],
        },
    )

    assert audit["atomic_factuality"]["unsupported_atom_count"] == 0
    assert audit["atomic_factuality"]["precision"] == 1.0


def test_summary_accepts_role_only_work_and_context_wrappers():
    audit = _audit(
        "2017年7月至2025年5月做企业软件销售，负责客户开发和方案演示。",
        {
            "summary": (
                "工作或实习经历包括企业软件销售。"
                "代表经历：担任企业软件销售期间，负责客户开发和方案演示。"
            ),
            "experience": [{
                "organization": "",
                "role": "企业软件销售",
                "period": "2017年7月至2025年5月",
                "bullets": ["负责客户开发和方案演示"],
            }],
        },
    )

    assert audit["atomic_factuality"]["unsupported_atom_count"] == 0
    assert audit["atomic_factuality"]["precision"] == 1.0


def test_ownership_audit_detects_cross_record_bullet_swap():
    audit = _audit(
        "工作经历\n甲公司｜产品经理｜2022.01-2024.01\n负责用户访谈。\n"
        "乙公司｜运营专员｜2020.01-2021.12\n策划线下活动。",
        {
            "experience": [
                {
                    "organization": "甲公司", "role": "产品经理",
                    "period": "2022.01-2024.01", "bullets": ["策划线下活动。"],
                },
                {
                    "organization": "乙公司", "role": "运营专员",
                    "period": "2020.01-2021.12", "bullets": ["负责用户访谈。"],
                },
            ],
        },
    )

    ownership = audit["ownership_integrity"]
    assert ownership["incorrect_assignment_count"] == 2
    assert ownership["integrity_rate"] == 0.75
    assert {
        (item["expected_record_id"], item["actual_record_id"])
        for item in ownership["issues"]
    } == {
        ("resume:experience:0", "resume:experience:1"),
        ("resume:experience:1", "resume:experience:0"),
    }


def test_distributed_header_fields_represent_one_source_fact():
    audit = _audit(
        "工作经历\n2022.01-2024.01 星河科技公司 产品经理\n负责需求调研。",
        {
            "experience": [{
                "organization": "星河科技公司", "role": "产品经理",
                "period": "2022.01-2024.01", "bullets": ["负责需求调研。"],
            }],
        },
    )

    factuality = audit["atomic_factuality"]
    assert factuality["precision"] == 1.0
    assert factuality["recall"] == 1.0
    assert factuality["unrepresented_source_facts"] == []


# Fixed, manually labelled atomic probes span nine domains.  Twelve are exact
# supported facts and twelve contain one deliberately unsupported addition.
# This fixture calibrates the detector independently from generator quality.
_CALIBRATION_CASES = [
    ("产研", "负责用户访谈", "负责用户访谈", True),
    ("教师", "承担高一数学教学", "承担高一数学教学", True),
    ("医疗", "完成门诊患者诊疗", "完成门诊患者诊疗", True),
    ("运营", "策划会员活动", "策划会员活动", True),
    ("金融", "复核授信材料", "复核授信材料", True),
    ("销售", "跟进重点客户", "跟进重点客户", True),
    ("设计", "输出交互原型", "输出交互原型", True),
    ("科研", "开展样本分析", "开展样本分析", True),
    ("制造", "记录生产参数", "记录生产参数", True),
    ("产研", "使用SQL分析数据", "使用 SQL 分析数据", True),
    ("教师", "组织教研活动", "组织教研活动", True),
    ("医疗", "参与病例讨论", "参与病例讨论", True),
    ("运营", "策划会员活动", "策划会员活动并提升留存率40%", False),
    ("金融", "复核授信材料", "复核授信材料并避免损失200万元", False),
    ("销售", "跟进重点客户", "跟进重点客户并成交50单", False),
    ("设计", "输出交互原型", "输出交互原型并获得设计大奖", False),
    ("科研", "开展样本分析", "使用Python开展样本分析", False),
    ("制造", "记录生产参数", "记录100批次生产参数", False),
    ("产研", "负责需求整理", "主导需求整理并管理10人团队", False),
    ("教师", "承担高一数学教学", "承担高一数学教学并提升平均分20分", False),
    ("医疗", "完成门诊患者诊疗", "独立完成300例门诊患者诊疗", False),
    ("运营", "整理用户反馈", "整理用户反馈并搭建Tableau看板", False),
    ("金融", "核对交易记录", "核对交易记录并实现零差错", False),
    ("销售", "准备方案演示", "准备方案演示并中标千万项目", False),
]


@pytest.mark.parametrize("domain,source_fact,output_fact,expected", _CALIBRATION_CASES)
def test_fixed_atomic_precision_calibration(
    domain: str,
    source_fact: str,
    output_fact: str,
    expected: bool,
):
    del domain
    audit = _audit(
        f"项目经历\n专项项目\n{source_fact}",
        {"projects": [{"name": "专项项目", "bullets": [output_fact]}]},
    )
    unsupported = {
        item["canonical_field_path"]
        for item in audit["atomic_factuality"]["unsupported_output"]
    }
    assert ("projects[0].bullets[0]" not in unsupported) is expected


def test_fixed_ownership_calibration_macro_f1_exceeds_gate():
    labels: list[str] = []
    predictions: list[str] = []
    domains = ("产研", "教师", "医疗", "运营", "金融", "销售")
    for index, domain in enumerate(domains):
        first_fact = f"负责{domain}事项甲{index}"
        second_fact = f"负责{domain}事项乙{index}"
        cv_text = (
            f"工作经历\n甲{index}公司｜岗位甲｜2022.01-2023.01\n{first_fact}。\n"
            f"乙{index}公司｜岗位乙｜2020.01-2021.01\n{second_fact}。"
        )
        for swapped in (False, True):
            first_output, second_output = (
                (second_fact, first_fact) if swapped else (first_fact, second_fact)
            )
            audit = _audit(cv_text, {
                "experience": [
                    {
                        "organization": f"甲{index}公司", "role": "岗位甲",
                        "period": "2022.01-2023.01", "bullets": [first_output],
                    },
                    {
                        "organization": f"乙{index}公司", "role": "岗位乙",
                        "period": "2020.01-2021.01", "bullets": [second_output],
                    },
                ],
            })
            issue_paths = {
                item["canonical_field_path"]
                for item in audit["ownership_integrity"]["issues"]
            }
            for record_index in range(2):
                path = f"experience[{record_index}].bullets[0]"
                labels.append("incorrect" if swapped else "correct")
                predictions.append("incorrect" if path in issue_paths else "correct")

    def f1(label: str) -> float:
        tp = sum(a == label and b == label for a, b in zip(labels, predictions))
        fp = sum(a != label and b == label for a, b in zip(labels, predictions))
        fn = sum(a == label and b != label for a, b in zip(labels, predictions))
        return 2 * tp / max(1, 2 * tp + fp + fn)

    macro_f1 = (f1("correct") + f1("incorrect")) / 2
    assert len(labels) == 24
    assert macro_f1 >= 0.95
