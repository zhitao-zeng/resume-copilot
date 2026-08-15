import json
from unittest.mock import patch

from atomic_fact_audit import audit_atomic_facts
from resume_optimizer import (
    _safe_rewrite,
    optimize_resume_with_provenance,
    select_narrative_record_keys,
)
from resume_copilot_service import _enforce_final_atomic_gate
from evidence_binding import bind_resume_evidence
from source_adapter import build_source_bundle
from v2_schemas import CanonicalResume


def test_optimizer_accepts_connective_only_surface_changes():
    assert _safe_rewrite(
        "搜集研究数据及信息，构建财务模型",
        "搜集研究数据及信息并构建财务模型",
    )
    assert _safe_rewrite(
        "采用图论方法，利用欧几里得距离、余弦相似度识别冗余数据",
        "采用图论方法，利用欧几里得距离和余弦相似度识别冗余数据",
    )
    assert _safe_rewrite(
        "调研前沿算法，对开源语音算法进行测试与评估",
        "调研前沿算法，并对开源语音算法进行测试与评估",
    )


def test_optimizer_still_rejects_a_new_named_method_after_connector_normalization():
    assert not _safe_rewrite(
        "负责用户调研并输出产品文档",
        "负责用户调研并使用六西格玛方法输出产品文档",
    )


def test_narrative_selector_includes_complementary_action_led_fragments():
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "产品经理",
            "bullets": [
                "负责梳理客户需求",
                "负责通过10次访谈收集反馈",
                "负责输出需求优先级清单",
            ],
        }],
    })

    assert select_narrative_record_keys(resume) == {("experience", 0)}


def test_record_level_star_composition_uses_only_declared_source_indices():
    source_bullets = [
        "负责梳理客户需求",
        "通过10次用户访谈收集反馈",
        "输出需求优先级清单",
    ]
    resume = CanonicalResume.model_validate({
        "experience": [{"bullets": source_bullets}],
    })
    combined = "负责梳理客户需求，通过10次用户访谈收集反馈并输出需求优先级清单"
    response = json.dumps({
        "experience": [{
            "index": 0,
            "bullets": [{"text": combined, "source_indices": [0, 1, 2]}],
        }],
    }, ensure_ascii=False)

    with patch("resume_optimizer.llm_enabled", return_value=True), patch(
        "resume_optimizer.call_llm_text", return_value=response,
    ), patch(
        "resume_optimizer.review_entailment_batch", return_value=[True],
    ):
        outcome = optimize_resume_with_provenance(resume)

    assert outcome.resume.experience[0].bullets == [combined]
    assert outcome.trusted_rewrites == {
        "experience[0].bullets[0]": "\n".join(source_bullets),
    }


def test_final_atomic_gate_removes_only_jd_only_skill():
    source = build_source_bundle(
        "姓名：李明\n技能：SQL",
        "请申请云平台岗位",
        "任职要求：熟悉Kubernetes",
    )
    rendered, _canonical, _bindings, removed, audit = _enforce_final_atomic_gate(
        {
            "meta": {"name": "李明"},
            "skills": {"tools": ["SQL", "Kubernetes"]},
        },
        source,
        [],
    )

    assert rendered["skills"]["tools"] == ["SQL"]
    assert "Kubernetes" not in json.dumps(rendered, ensure_ascii=False)
    assert removed == ["skills.items[1].name"]
    assert audit["atomic_factuality"]["precision"] == 1.0


def test_final_atomic_gate_repairs_with_education_record_present():
    source = build_source_bundle(
        "姓名：李明\n北京大学｜计算机科学｜本科\n技能：SQL",
        "请申请云平台岗位",
        "任职要求：熟悉Kubernetes",
    )
    rendered, _canonical, _bindings, removed, audit = _enforce_final_atomic_gate(
        {
            "meta": {"name": "李明"},
            "education": [{
                "school": "北京大学",
                "major": "计算机科学",
                "degree": "本科",
            }],
            "skills": {"tools": ["SQL", "Kubernetes"]},
        },
        source,
        [],
    )

    assert rendered["education"][0]["school"] == "北京大学"
    assert rendered["skills"]["tools"] == ["SQL"]
    assert removed == ["skills.items[1].name"]
    assert audit["atomic_factuality"]["precision"] == 1.0


def test_atomic_audit_ignores_grounded_campus_summary_wrapper():
    source = build_source_bundle(
        "项目经历\n远山公益｜志愿者｜2022.07\n整理30份学员反馈并完成复盘。",
        "",
        "",
    )
    resume = CanonicalResume.model_validate({
        "summary": (
            "相关经历：校园或社会经历（远山公益｜志愿者）："
            "整理30份学员反馈并完成复盘。"
        ),
        "activities": [{
            "organization": "远山公益",
            "role": "志愿者",
            "period": "2022.07",
            "bullets": ["整理30份学员反馈并完成复盘"],
        }],
    })
    bindings = bind_resume_evidence(resume, source)

    audit = audit_atomic_facts(
        source=source,
        resume=resume,
        evidence_bindings=bindings,
    )

    assert audit["atomic_factuality"]["unsupported_atom_count"] == 0
    assert audit["atomic_factuality"]["precision"] == 1.0


def test_final_atomic_gate_repairs_unsupported_result_clause_locally():
    source = build_source_bundle(
        "工作经历\n甲公司｜产品经理｜2022.01-2024.01\n负责用户访谈。",
        "",
        "岗位要求：提升转化率30%",
    )
    rendered, _canonical, _bindings, removed, audit = _enforce_final_atomic_gate(
        {
            "experience": [{
                "company": "甲公司",
                "role": "产品经理",
                "period": "2022.01-2024.01",
                "bullets": ["负责用户访谈，并提升转化率30%。"],
            }],
        },
        source,
        [],
    )

    assert rendered["experience"][0]["bullets"] == ["负责用户访谈。"]
    assert "experience[0].bullets[0]" in removed
    assert audit["atomic_factuality"]["precision"] == 1.0


def test_final_atomic_gate_preserves_reviewed_grounded_rewrite():
    source_text = "工作经历\n甲公司｜产品经理\n负责收集用户反馈并维护需求文档"
    source = build_source_bundle(source_text, "", "")
    rewritten = "负责围绕用户反馈开展整理归纳，持续维护需求文档"
    canonical = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "产品经理",
            "bullets": [rewritten],
        }],
    })
    prior = bind_resume_evidence(
        canonical,
        source,
        trusted_rewrites={
            "experience[0].bullets[0]": "负责收集用户反馈并维护需求文档",
        },
    )

    rendered, _canonical, _bindings, removed, audit = _enforce_final_atomic_gate(
        {
            "experience": [{
                "company": "甲公司",
                "role": "产品经理",
                "bullets": [rewritten],
            }],
        },
        source,
        prior,
    )

    assert rendered["experience"][0]["bullets"] == [rewritten]
    assert removed == []
    assert audit["atomic_factuality"]["precision"] == 1.0
