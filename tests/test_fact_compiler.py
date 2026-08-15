from __future__ import annotations

import v2_pipeline

from evidence_binding import bind_resume_evidence
from fact_compiler import (
    _owner_map,
    compile_fact_coverage,
    route_source_facts,
    sanitize_resume_placeholders,
)
from source_adapter import build_source_bundle, candidate_blocks
from v2_schemas import CanonicalResume, EvidenceBinding
from v2_pipeline import (
    _audit_existing_scaffold,
    _apply_fact_compiler_candidate,
    _deterministic_fallback,
    _fallback_is_structurally_safe,
    _grounded_source_fallback,
    run_v2_pipeline,
)


def test_numbered_markdown_and_english_sections_enter_fact_ledger():
    source = build_source_bundle(
        "",
        """以下是我的全部个人信息，请勿编造。

1. **工作经历**
产品经理 2022年1月 - 至今
- 访谈用户并输出需求优先级清单

2. **技能**
Python, SQL

Experience Highlights
- Collaborated with stakeholders to define needs
""",
        "",
    )

    eligible = [fact for fact in source.fact_units if fact.fact_eligible]
    assert any(fact.section_hint == "experience" for fact in eligible)
    assert any(fact.section_hint == "skills" for fact in eligible)
    assert any(fact.section_hint == "highlights" for fact in eligible)


def test_numbered_profile_sections_do_not_inherit_previous_record_scope():
    source = build_source_bundle(
        "",
        """5. 教育背景
- [学校]，1999年，工商管理硕士
6. 认证和执照
- 注册内部审计师 (CIA)
8. 奖项和荣誉
- 2015年财务卓越奖
9. 志愿服务经历
- 财务委员会成员，2016年至今
10. 语言能力
- 法语：日常会话
11. 兴趣爱好
- 可持续经济学
12. 推荐人
推荐信可按需提供。
""",
        "",
    )
    by_text = {block.text: block for block in source.blocks}

    assert by_text["- 注册内部审计师 (CIA)"].section_hint == "certifications"
    assert by_text["- 2015年财务卓越奖"].section_hint == "awards"
    assert by_text["- 财务委员会成员，2016年至今"].section_hint == "activities"
    assert by_text["- 法语：日常会话"].section_hint == "skills"
    assert by_text["- 可持续经济学"].section_hint == "hobbies"
    assert by_text["推荐信可按需提供。"].fact_eligible is False


def test_generic_professional_project_and_credential_headings_keep_structure():
    source = build_source_bundle(
        """专业经历
科技初创公司 | 应用开发实习生 | 2020年3月 - 2020年5月
- 开发短信应用程序
学术项目与研讨会
客户反馈情感评估 2020年6月
认证资质
- 云服务基础认证
""",
        "",
        "",
    )
    by_text = {block.text: block for block in source.blocks}

    role_block = by_text["科技初创公司 | 应用开发实习生 | 2020年3月 - 2020年5月"]
    assert role_block.section_hint == "experience"
    assert role_block.record_id
    assert by_text["客户反馈情感评估 2020年6月"].section_hint == "projects"
    assert by_text["- 云服务基础认证"].section_hint == "certifications"


def test_inline_project_technology_does_not_steal_following_project_records():
    text = """项目经历
在线零售管理面板 2023年3-4月 • 1个月
技术栈：VueJS • Node.js • Git
AI助手机器人 2023年3月 - 4月 • 1个月
技术：JavaScript • Express.js
集成视频社交网络平台 2023年4月 - 2023年5月 • 1个月
技术栈：VueJS • TypeScript • Oauth2
技能
Python • SQL
"""
    source = build_source_bundle(text, "", "")
    by_text = {block.text: block for block in source.blocks}

    assert by_text["技术栈：VueJS • Node.js • Git"].section_hint == "projects"
    assert by_text["AI助手机器人 2023年3月 - 4月 • 1个月"].section_hint == "projects"
    assert by_text["集成视频社交网络平台 2023年4月 - 2023年5月 • 1个月"].section_hint == "projects"
    project_ids = {
        by_text["在线零售管理面板 2023年3-4月 • 1个月"].record_id,
        by_text["AI助手机器人 2023年3月 - 4月 • 1个月"].record_id,
        by_text["集成视频社交网络平台 2023年4月 - 2023年5月 • 1个月"].record_id,
    }
    assert None not in project_ids
    assert len(project_ids) == 3
    assert by_text["Python • SQL"].section_hint == "skills"

    resume = _deterministic_fallback(text, "", "")

    assert [project.name for project in resume.projects] == [
        "在线零售管理面板",
        "AI助手机器人",
        "集成视频社交网络平台",
    ]
    assert [project.period for project in resume.projects] == [
        "2023年3-4月 • 1个月",
        "2023年3月 - 4月 • 1个月",
        "2023年4月 - 2023年5月 • 1个月",
    ]
    assert resume.projects[0].bullets == ["技术栈：VueJS • Node.js • Git"]
    assert resume.projects[1].bullets == ["技术：JavaScript • Express.js"]
    assert resume.projects[2].bullets == ["技术栈：VueJS • TypeScript • Oauth2"]


def test_split_docx_project_header_duration_and_technology_keep_one_owner():
    text = """项目
在线零售管理面板 2023年3-4月
- 1个月
技术栈：VueJS
- Node.js
- Git
AI助手机器人 2023年3月 - 4月
- 1个月
技术：JavaScript
- Express.js
- VueJS
集成视频社交网络平台
2023年4月 - 2023年5月
- 1个月
技术：TypeScript
- Oauth2
"""
    source = build_source_bundle(text, "", "")
    by_text = {block.text: block for block in source.blocks}

    assert "VueJS集成视频社交网络平台" not in by_text
    assert by_text["集成视频社交网络平台"].record_id
    assert by_text["集成视频社交网络平台"].record_id == by_text[
        "2023年4月 - 2023年5月"
    ].record_id
    assert by_text["集成视频社交网络平台"].record_id != by_text[
        "AI助手机器人 2023年3月 - 4月"
    ].record_id

    resume = _deterministic_fallback(text, "", "")

    assert [project.name for project in resume.projects] == [
        "在线零售管理面板",
        "AI助手机器人",
        "集成视频社交网络平台",
    ]
    assert [project.period for project in resume.projects] == [
        "2023年3-4月 • 1个月",
        "2023年3月 - 4月 • 1个月",
        "2023年4月 - 2023年5月 • 1个月",
    ]
    assert resume.projects[0].bullets == ["技术栈：VueJS • Node.js • Git"]
    assert resume.projects[1].bullets == ["技术：JavaScript • Express.js • VueJS"]
    assert resume.projects[2].bullets == ["技术：TypeScript • Oauth2"]

    bindings = bind_resume_evidence(resume, source)
    project_technology_bindings = {
        binding.path: binding
        for binding in bindings
        if binding.path.startswith("projects[") and ".bullets[" in binding.path
    }
    assert set(project_technology_bindings) == {
        "projects[0].bullets[0]",
        "projects[1].bullets[0]",
        "projects[2].bullets[0]",
    }
    assert all(
        len(binding.block_ids) >= 2
        for binding in project_technology_bindings.values()
    )
    assert all(
        len({
            block.record_id
            for block in source.blocks
            if block.block_id in binding.block_ids
        }) == 1
        for binding in project_technology_bindings.values()
    )

    evidence_text = "\n".join(
        block.text for block in candidate_blocks(source)
    )
    grounded_result, _coverage, _missing = _grounded_source_fallback(
        text,
        "",
        "",
        source,
        evidence_text,
    )
    assert [
        project.bullets for project in grounded_result.resume.projects
    ] == [
        ["技术栈：VueJS • Node.js • Git"],
        ["技术：JavaScript • Express.js • VueJS"],
        ["技术：TypeScript • Oauth2"],
    ]
    assert _fallback_is_structurally_safe(
        grounded_result.resume,
        allow_partial_record_identity=True,
    ) is True


def test_fact_compiler_does_not_treat_credential_fragment_as_complete():
    text = "认证和执照\n- 监管合规证书 - 2011"
    source = build_source_bundle(text, "", "")
    scaffold = _deterministic_fallback(text, "", "")
    fragmented = CanonicalResume.model_validate({
        "certifications": ["- 监管合规", "- 2011"],
    })

    compiled, _report, _routes = compile_fact_coverage(
        fragmented,
        source,
        scaffold=scaffold,
    )

    assert "监管合规证书 - 2011" in compiled.certifications


def test_record_subsection_labels_are_layout_not_candidate_facts():
    source = build_source_bundle(
        "",
        """专业经验
甲公司｜产品经理｜2022年至2024年
主要成就：
- 完成10次用户访谈
成就：
- 输出需求优先级清单
""",
        "",
    )
    eligible_text = {
        fact.verbatim_text.strip(" ：:")
        for fact in source.fact_units
        if fact.fact_eligible
    }

    assert "主要成就" not in eligible_text
    assert "成就" not in eligible_text
    assert "完成10次用户访谈" in eligible_text
    assert "输出需求优先级清单" in eligible_text


def test_year_only_ranges_and_brand_organizations_survive_evidence_gate():
    query = """专业经验
BUILDTECH SUPPLIES INC. (全球制造集团) 2015年6月至2015年11月
财务与资源经理
- 建立库存管理制度
RETAIL HUB INTERNATIONAL 2011年至2012年
审计与合规经理
- 实施财务管理系统
"""
    source = build_source_bundle("", query, "")
    resume = _deterministic_fallback("", query, "")
    gated, _, _ = v2_pipeline.enforce_resume_evidence(resume, source)

    assert gated.experience[0].organization == "BUILDTECH SUPPLIES INC. (全球制造集团)"
    assert gated.experience[1].organization == "RETAIL HUB INTERNATIONAL"
    assert gated.experience[1].period == "2011年至2012年"


def test_query_fallback_preserves_cross_industry_role_and_anonymous_projects():
    query = """专业经验
未来味道有限公司（全球食品集团） 2012年至2014年
财务运营协调员
- 创新电子发票流程，提高应收账款周转率
项目经历
- 开发自动化财务预测模型，使处理时间减少40%。
- 在跨职能团队中发挥关键作用，实施集成销售和财务报告系统。
"""

    resume = _deterministic_fallback("", query, "")

    assert resume.experience[0].role == "财务运营协调员"
    assert len(resume.projects) == 1
    assert resume.projects[0].name == ""
    assert resume.projects[0].bullets == [
        "开发自动化财务预测模型，使处理时间减少40%。",
        "在跨职能团队中发挥关键作用，实施集成销售和财务报告系统。",
    ]
    assert _fallback_is_structurally_safe(resume) is True


def test_query_skill_phrases_keep_chinese_semantics_and_language_proficiency():
    resume = _deterministic_fallback(
        "",
        """技能
- 财务分析和建模方面的深厚专业知识
- 战略规划和实施
- 出色的团队领导和管理技能
- 精通ERP和CRM系统，包括SAP和JD Edwards
语言能力
- 西班牙语：流利，书面和口语
- 法语：日常会话
""",
        "",
    )

    items = {(item.name, item.category) for item in resume.skills.items}
    assert ("财务分析和建模方面的深厚专业知识", "other") in items
    assert ("战略规划和实施", "other") in items
    assert ("出色的团队领导和管理技能", "other") in items
    assert {"ERP", "CRM系统", "SAP", "JD Edwards"}.issubset(
        {item.name for item in resume.skills.items}
    )
    assert ("西班牙语：流利，书面和口语", "natural_language") in items
    assert ("法语：日常会话", "natural_language") in items


def test_roman_numbered_work_sections_share_generic_record_grammar():
    source = build_source_bundle(
        """专业概述
拥有12年ERP销售经验。

工作经历 - V
职位：销售主管
公司：甲公司
持续时间：从2013年11月开始
• 负责ERP销售策略

工作经历 - IV
区域销售主管
组织：乙集团
时期：2012年9月至2013年11月
• 开展产品培训和演示
""",
        "",
        "",
    )

    summary = [fact for fact in source.fact_units if fact.section_hint == "summary"]
    experience = [fact for fact in source.fact_units if fact.section_hint == "experience"]
    owners = {fact.record_id for fact in experience if fact.record_id}

    assert summary
    assert experience
    assert len(owners) == 2


def test_split_roman_layout_marker_does_not_create_candidate_or_record():
    source = build_source_bundle(
        """工作经历
- IV
区域销售主管
组织：科技创新者集团
时期：2012年9月至2013年11月
职责：
- 开展产品培训和演示
""",
        "",
        "",
    )

    facts = [
        fact for fact in source.fact_units
        if fact.fact_eligible and fact.section_hint == "experience"
    ]
    owners = {fact.record_id for fact in facts if fact.record_id}

    assert all(fact.verbatim_text != "IV" for fact in facts)
    assert len(owners) == 1


def test_placeholder_cleanup_drops_split_roman_layout_record_duplicate():
    resume = CanonicalResume.model_validate({
        "experience": [
            {"bullets": ["IV区域销售主管"]},
            {
                "organization": "科技创新者集团",
                "role": "区域销售主管",
                "period": "2012年9月至2013年11月",
                "bullets": ["开展产品培训和演示"],
            },
        ],
    })

    cleaned, _ = sanitize_resume_placeholders(resume)

    assert len(cleaned.experience) == 1
    assert cleaned.experience[0].role == "区域销售主管"


def test_placeholder_cleanup_drops_standalone_ordinal_and_identity_bullet():
    resume = CanonicalResume.model_validate({
        "experience": [
            {"bullets": ["IV"]},
            {
                "organization": "机械解决方案有限公司",
                "role": "销售协调员",
                "period": "2004年5月至2007年2月",
                "bullets": ["销售协调员", "协调销售工作"],
            },
        ],
    })

    cleaned, _ = sanitize_resume_placeholders(resume)

    assert len(cleaned.experience) == 1
    assert cleaned.experience[0].bullets == ["协调销售工作"]


def test_placeholder_cleanup_moves_split_roman_role_to_adjacent_record():
    resume = CanonicalResume.model_validate({
        "experience": [
            {"bullets": ["III 销售和市场营销协调员"]},
            {
                "organization": "创新解决方案有限公司",
                "period": "2011年5月至2012年7月",
                "bullets": ["评估客户需求"],
            },
        ],
    })

    cleaned, _ = sanitize_resume_placeholders(resume)

    assert len(cleaned.experience) == 1
    assert cleaned.experience[0].role == "销售和市场营销协调员"
    assert cleaned.experience[0].organization == "创新解决方案有限公司"


def test_placeholder_cleanup_clears_empty_labeled_identity_remainder():
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "[公司]",
            "role": "公司：[公司]，[城市]和[国家]",
            "period": "2007年2月至2011年4月",
            "bullets": ["实施ERP解决方案"],
        }],
    })

    cleaned, _ = sanitize_resume_placeholders(resume)

    assert cleaned.experience[0].organization == ""
    assert cleaned.experience[0].role == ""


def test_structured_query_preserves_rows_sections_and_record_ownership():
    source = build_source_bundle(
        "",
        """请完整保留事实，不要补造。

摘要或目标
拥有生物科学学位，专注于生物医学研究。

工作经历
研究助理 I 2022年7月 - 至今
[公司], [城市], [州]
- 执行样品处理，包括研磨、取样和提取

药房技术员 2009年7月 - 2019年11月
[公司], [城市], [州]
- 处理和配发处方订单，维护库存水平

奖项和荣誉
- 获得2020年[奖项]实验室管理卓越奖
""",
        "",
    )

    eligible = [fact for fact in source.fact_units if fact.fact_eligible]
    experience = [fact for fact in eligible if fact.section_hint == "experience"]
    owners = {fact.record_id for fact in experience if fact.record_id}

    assert len(owners) == 2
    assert any(
        block.text == "- 执行样品处理，包括研磨、取样和提取"
        for block in source.blocks
    )
    assert any(fact.section_hint == "summary" for fact in eligible)
    assert any(fact.section_hint == "awards" for fact in eligible)
    assert all(
        not fact.fact_eligible
        for fact in source.fact_units
        if fact.verbatim_text in {"[公司]", "[城市]", "[州]"}
    )


def test_numbered_education_keeps_identity_rows_with_their_numbered_item():
    source = build_source_bundle(
        """教育
1. 学士学位
   [学校], [城市]
   2004
   68%
2. 高中毕业证书
   [学校], [城市]
   2001
   66%
3. 中学毕业证书
   [学校], [城市]
   1999
   71%
""",
        "",
        "",
    )

    facts = [
        fact for fact in source.fact_units
        if fact.fact_eligible and fact.section_hint == "education"
    ]
    owners = {fact.record_id for fact in facts if fact.record_id}

    assert owners == {
        "resume:education:0",
        "resume:education:1",
        "resume:education:2",
    }
    degree_owner = {
        fact.verbatim_text: fact.record_id
        for fact in facts
        if "毕业" in fact.verbatim_text or "学士" in fact.verbatim_text
    }
    assert degree_owner["学士学位"] == "resume:education:0"
    assert degree_owner["高中毕业证书"] == "resume:education:1"
    assert degree_owner["中学毕业证书"] == "resume:education:2"


def test_anonymized_numbered_education_keeps_degree_without_fake_school():
    resume = _deterministic_fallback(
        """教育
1. 学士学位
   [学校], [城市]
   2004
   68%
2. 高中毕业证书
   [学校], [城市]
   2001
   66%
3. 中学毕业证书
   [学校], [城市]
   1999
   71%
""",
        "",
        "",
    )

    assert [item.school for item in resume.education] == ["", "", ""]
    assert [item.degree for item in resume.education] == [
        "学士学位",
        "高中毕业证书",
        "中学毕业证书",
    ]
    assert [item.major for item in resume.education] == ["", "", ""]
    assert resume.additional_sections["教育成绩"] == ["68%", "66%", "71%"]
    assert _fallback_is_structurally_safe(resume) is False
    assert _fallback_is_structurally_safe(
        resume,
        allow_missing_education_school=True,
    ) is True


def test_anonymized_education_rows_still_separate_multiple_degrees():
    source = build_source_bundle(
        "",
        """教育背景
[学校] [城市], [州]
理学学士学位，生物科学专业 2022年5月
专业方向：生物医学研究 辅修：化学

[学校] [城市], [州]
通识研究文科副学士学位 2015年5月
""",
        "",
    )

    facts = [
        fact for fact in source.fact_units
        if fact.fact_eligible and fact.section_hint == "education"
    ]
    owners = {fact.record_id for fact in facts if fact.record_id}

    assert owners == {"query:education:0", "query:education:1"}


def test_fact_routes_reject_jd_placeholders_and_user_instructions():
    source = build_source_bundle(
        "",
        """请生成简历，不得编造。
联系方式
[姓名]
技能
Python
""",
        "岗位要求：具备Kubernetes经验",
    )

    routes = route_source_facts(source)
    route_by_id = {route.fact_id: route for route in routes}
    facts = {fact.fact_id: fact for fact in source.fact_units}

    assert any(
        facts[fact_id].verbatim_text == "Python" and route.status == "resume"
        for fact_id, route in route_by_id.items()
    )
    assert all(
        route.status == "rejected"
        for fact_id, route in route_by_id.items()
        if facts[fact_id].source_type == "jd"
    )
    assert all(
        route.status != "resume"
        for fact_id, route in route_by_id.items()
        if facts[fact_id].verbatim_text == "[姓名]"
    )


def test_compiler_restores_missing_fact_to_its_owned_record():
    source = build_source_bundle(
        """工作经历
甲公司 产品经理 2022.01-2023.01
开展10次用户访谈并完成竞品分析。
输出需求优先级清单，协调研发推动功能上线。

乙公司 运营专员 2020.01-2021.12
策划用户活动并维护活动台账。
""",
        "",
        "",
    )
    resume = CanonicalResume.model_validate({
        "experience": [
            {
                "organization": "甲公司",
                "role": "产品经理",
                "period": "2022.01-2023.01",
                "bullets": ["开展10次用户访谈并完成竞品分析。"],
            },
            {
                "organization": "乙公司",
                "role": "运营专员",
                "period": "2020.01-2021.12",
                "bullets": ["策划用户活动并维护活动台账。"],
            },
        ],
    })

    compiled, report, _ = compile_fact_coverage(resume, source)

    assert any("需求优先级清单" in bullet for bullet in compiled.experience[0].bullets)
    assert not any("需求优先级清单" in bullet for bullet in compiled.experience[1].bullets)
    assert report.after_coverage > report.before_coverage
    assert bind_resume_evidence(compiled, source)


def test_record_recovery_mode_cannot_merge_scaffold_or_loose_sections():
    source = build_source_bundle(
        """个人总结
资深产品从业者
工作经历
甲公司 产品经理 2022.01-2023.01
开展10次用户访谈并完成竞品分析。
输出需求优先级清单，协调研发推动功能上线。
乙公司 运营专员 2020.01-2021.12
策划用户活动并维护活动台账。
技能
Python
""",
        "",
        "",
    )
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "产品经理",
            "period": "2022.01-2023.01",
            "bullets": ["开展10次用户访谈并完成竞品分析。"],
        }],
    })
    scaffold = CanonicalResume.model_validate({
        "summary": "资深产品从业者",
        "experience": [
            {
                "organization": "甲公司",
                "role": "产品经理",
                "period": "2022.01-2023.01",
                "bullets": ["输出需求优先级清单，协调研发推动功能上线。"],
            },
            {
                "organization": "乙公司",
                "role": "运营专员",
                "period": "2020.01-2021.12",
                "bullets": ["策划用户活动并维护活动台账。"],
            },
        ],
        "skills": {"items": [{"name": "Python", "category": "language"}]},
    })

    compiled, report, _ = compile_fact_coverage(
        resume,
        source,
        scaffold=scaffold,
        merge_scaffold=False,
        allowed_destinations=frozenset({
            "experience", "research", "activities", "projects",
        }),
    )

    assert len(compiled.experience) == 1
    assert compiled.experience[0].organization == "甲公司"
    assert any("需求优先级清单" in item for item in compiled.experience[0].bullets)
    assert compiled.summary == ""
    assert compiled.skills.items == []
    assert report.merged_records == 0


def test_compiler_routes_unowned_action_to_public_semantic_section():
    source = build_source_bundle(
        "",
        "我还负责整理客户反馈并输出问题清单。",
        "",
    )
    compiled, report, _ = compile_fact_coverage(CanonicalResume(), source)

    assert compiled.additional_sections == {
        "经历亮点": ["我还负责整理客户反馈并输出问题清单"],
    }
    assert all("待整理" not in title for title in compiled.additional_sections)
    assert report.appended_values == 1


def test_compiler_publishes_field_value_without_non_fact_label():
    source = build_source_bundle(
        "",
        """Professional Direction: Product Manager
Career Level: Mid
Industry: Retail
""",
        "",
    )

    compiled, _, _ = compile_fact_coverage(CanonicalResume(), source)

    assert compiled.additional_sections["求职目标"] == ["Product Manager"]
    assert compiled.additional_sections["个人概况"] == ["Mid", "Retail"]


def test_compiler_merges_grounded_scaffold_without_replacing_richer_text():
    source = build_source_bundle(
        """工作经历
甲公司 产品经理 2022.01-2023.01
开展用户访谈并输出需求清单。
乙公司 运营专员 2020.01-2021.12
策划活动并维护数据台账。
""",
        "",
        "",
    )
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "产品经理",
            "period": "2022.01-2023.01",
            "bullets": ["开展用户访谈并输出需求清单。"],
        }],
    })
    scaffold = CanonicalResume.model_validate({
        "experience": [
            {
                "organization": "甲公司",
                "role": "产品经理",
                "period": "2022.01-2023.01",
                "bullets": ["开展用户访谈并输出需求清单。"],
            },
            {
                "organization": "乙公司",
                "role": "运营专员",
                "period": "2020.01-2021.12",
                "bullets": ["策划活动并维护数据台账。"],
            },
        ],
    })

    compiled, report, _ = compile_fact_coverage(resume, source, scaffold=scaffold)

    assert len(compiled.experience) == 2
    assert compiled.experience[0].bullets == ["开展用户访谈并输出需求清单。"]
    assert compiled.experience[1].organization == "乙公司"
    assert report.merged_records == 1


def test_compiler_does_not_import_scaffold_record_without_source_section():
    source = build_source_bundle(
        """工作经历
甲公司 产品经理 2022.01-2023.01
开展用户访谈并输出需求清单。
""",
        "",
        "",
    )
    scaffold = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "产品经理",
            "period": "2022.01-2023.01",
            "bullets": ["开展用户访谈并输出需求清单。"],
        }],
        "projects": [{
            "name": "需求分析项目",
            "bullets": ["开展用户访谈并输出需求清单。"],
        }],
    })

    compiled, _, _ = compile_fact_coverage(
        CanonicalResume(), source, scaffold=scaffold,
    )

    assert compiled.experience
    assert compiled.projects == []


def test_placeholder_cleanup_removes_tokens_but_keeps_literal_source_remainder():
    resume = CanonicalResume.model_validate({
        "education": [{
            "school": "[学校",
            "degree": "理学学士学位",
            "major": "生物科学",
            "period": "2022年5月",
        }],
        "experience": [{
            "organization": "",
            "role": "[公司], [城市], [州]",
            "period": "2009年7月 - 2019年11月",
            "bullets": ["在[公司]负责处理处方订单"],
        }],
        "summary": (
            "工作或实习经历包括[公司], [城市], [州]。"
            "核心技能包括ELISA和定量PCR。"
        ),
        "awards": [
            "获得2020年[奖项]实验室管理卓越奖",
            "质量保证专员，【公司",
        ],
        "additional_sections": {
            "补充信息": ["Professional Direction: Product Manager"],
        },
    })

    cleaned, paths = sanitize_resume_placeholders(resume)

    assert cleaned.education[0].school == ""
    assert cleaned.experience[0].role == ""
    assert cleaned.experience[0].bullets == ["负责处理处方订单"]
    assert cleaned.summary == "核心技能包括ELISA和定量PCR。"
    assert cleaned.awards == [
        "获得2020年实验室管理卓越奖",
        "质量保证专员",
    ]
    assert "补充信息" not in cleaned.additional_sections
    assert "summary" in paths


def test_query_fallback_keeps_arbitrary_dated_roles_and_full_year_range():
    resume = _deterministic_fallback(
        "",
        """工作经历
研究助理 I 2022年7月 - 至今
- 执行样品处理，包括研磨、取样和提取
药房技术员 2009年7月 - 2019年11月
- 处理和配发处方订单，维护库存水平

志愿者经历
- 志愿实验室助理，[组织]，2018-2020

项目
- 参与开发种子遗传学样本分析的标准化协议
- 参与制药服务中化合物制备方法学的研究
""",
        "",
    )

    assert [item.role for item in resume.experience] == ["研究助理 I", "药房技术员"]
    assert resume.activities[0].period == "2018-2020"
    assert len(resume.projects) == 1


def test_education_year_binding_does_not_use_same_year_employment_record():
    source = build_source_bundle(
        "",
        """工作经历
2000年10月 - 2003年3月
经济与数据分析师
- 负责经济报告
教育背景
硕士学位
2000 [学校]
经济学，硕士项目
""",
        "",
    )
    resume = CanonicalResume.model_validate({
        "education": [{
            "degree": "硕士学位",
            "major": "经济学",
            "period": "2000",
        }],
    })

    bindings = bind_resume_evidence(resume, source)
    period_binding = next(
        item for item in bindings if item.path == "education[0].period"
    )
    block = next(
        item for item in source.blocks if item.block_id == period_binding.block_id
    )

    assert block.section_hint == "education"
    assert block.record_id and ":education:" in block.record_id


def test_long_tail_product_section_never_becomes_anonymous_experience():
    resume = _deterministic_fallback(
        """产品与解决方案
1. 供应管理套件
2. 零售销售点系统
- SCM: 供应链管理
""",
        "",
        "",
    )

    assert resume.experience == []
    assert resume.additional_sections["产品与解决方案"] == [
        "1. 供应管理套件",
        "2. 零售销售点系统",
        "SCM: 供应链管理",
    ]


def test_unowned_action_is_a_highlight_not_a_manufactured_job():
    resume = _deterministic_fallback(
        "• 为超过40家企业提供集成解决方案\n",
        "",
        "",
    )

    assert resume.experience == []
    assert resume.additional_sections["经历亮点"] == [
        "• 为超过40家企业提供集成解决方案",
    ]


def test_pipeline_fact_compiler_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FACT_COMPILER_MODE", raising=False)
    source = build_source_bundle(
        """工作经历
甲公司 产品经理 2022.01-2023.01
开展用户访谈。
输出需求优先级清单。
""",
        "",
        "",
    )
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "产品经理",
            "period": "2022.01-2023.01",
            "bullets": ["开展用户访谈。"],
        }],
    })

    result, _, removed, diagnostics = _apply_fact_compiler_candidate(
        resume,
        source,
        scaffold=None,
    )

    assert result is resume
    assert removed == []
    assert diagnostics == {
        "mode": "legacy",
        "accepted": False,
        "reason": "disabled",
    }


def test_pipeline_fact_compiler_applies_only_audited_recall_gain(monkeypatch):
    monkeypatch.setenv("FACT_COMPILER_MODE", "on")
    source = build_source_bundle(
        """工作经历
甲公司 产品经理 2022.01-2023.01
开展10次用户访谈并完成竞品分析。
输出需求优先级清单，协调研发推动功能上线。
""",
        "",
        "",
    )
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "产品经理",
            "period": "2022.01-2023.01",
            "bullets": ["开展10次用户访谈并完成竞品分析。"],
        }],
    })

    result, _, _, diagnostics = _apply_fact_compiler_candidate(
        resume,
        source,
        scaffold=None,
    )

    assert diagnostics["safe"] is True
    assert diagnostics["accepted"] is True
    assert diagnostics["recall_gain"] > 0
    assert diagnostics["after_unsupported"] <= diagnostics["before_unsupported"]
    assert diagnostics["after_ownership_errors"] <= diagnostics["before_ownership_errors"]
    assert any(
        "需求优先级清单" in bullet
        for bullet in result.experience[0].bullets
    )


def test_strict_compiler_owner_ignores_body_similarity_until_identity_is_bound():
    source = build_source_bundle(
        """工作经历
甲公司｜产品经理｜2022.01-2023.01
开展用户访谈。
乙公司｜运营经理｜2023.02-2024.01
策划用户活动。
""",
        "",
        "",
    )
    first_header = next(
        block for block in source.blocks if "甲公司" in block.text
    )
    first_body = next(
        block for block in source.blocks if "开展用户访谈" in block.text
    )
    body_only = [EvidenceBinding(
        path="experience[0].bullets[0]",
        block_id=first_body.block_id,
        block_ids=[first_body.block_id],
        quote=first_body.text,
        claim="开展用户访谈",
        mode="direct",
    )]

    assert _owner_map(source, body_only)
    assert _owner_map(source, body_only, require_identity=True) == {}

    identity_bound = body_only + [
        EvidenceBinding(
            path=f"experience[0].{field}",
            block_id=first_header.block_id,
            block_ids=[first_header.block_id],
            quote=value,
            claim=value,
            mode="direct",
        )
        for field, value in (
            ("organization", "甲公司"),
            ("period", "2022.01-2023.01"),
        )
    ]
    assert _owner_map(source, identity_bound, require_identity=True) == {
        ("experience", first_header.record_id): 0,
    }


def test_pipeline_fact_compiler_does_not_reintroduce_ocr_layout_seams(monkeypatch):
    monkeypatch.setenv("FACT_COMPILER_MODE", "on")
    source = build_source_bundle(
        """工作经历
甲公司 产品经理 2022.01-2023.01
开展10次用户访谈。
输出需求优先级清单。
经历亮点
此前担任：副总裁，投资策略（收入1.1亿美元>副总裁，全球客户管理
""",
        "",
        "",
    )
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "产品经理",
            "period": "2022.01-2023.01",
            "bullets": ["开展10次用户访谈。"],
        }],
    })
    scaffold = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "产品经理",
            "period": "2022.01-2023.01",
            "bullets": ["开展10次用户访谈。", "输出需求优先级清单。"],
        }],
        "additional_sections": {
            "经历亮点": [
                "此前担任：副总裁，投资策略（收入1.1亿美元>副总裁，全球客户管理",
            ],
        },
    })

    result, _, _, diagnostics = _apply_fact_compiler_candidate(
        resume,
        source,
        scaffold=scaffold,
    )

    assert diagnostics["accepted"] is True
    assert result.experience[0].bullets[-1] == "输出需求优先级清单"
    assert "经历亮点" not in result.additional_sections


def test_pipeline_fact_compiler_prefers_complete_credential_over_fragments(monkeypatch):
    monkeypatch.setenv("FACT_COMPILER_MODE", "on")
    source_text = """认证和执照
- 项目管理专业人士 (PMP) - 2010
- 监管合规证书 - 2011
"""
    source = build_source_bundle(source_text, "", "")
    resume = CanonicalResume.model_validate({
        "certifications": [
            "项目管理专业人士 (PMP) - 2010",
            "- 监管合规",
            "2011",
        ],
    })
    scaffold = _deterministic_fallback(source_text, "", "")

    result, _, _, diagnostics = _apply_fact_compiler_candidate(
        resume,
        source,
        scaffold=scaffold,
    )

    assert diagnostics["accepted"] is True
    assert result.certifications == [
        "项目管理专业人士 (PMP) - 2010",
        "监管合规证书 - 2011",
    ]


def test_pipeline_fact_compiler_drops_inferred_credential_year_duplicate(monkeypatch):
    monkeypatch.setenv("FACT_COMPILER_MODE", "on")
    source_text = """认证和执照
- 医疗保健支持专员认证
2009
- 认证患者护理技师
2000
"""
    source = build_source_bundle(source_text, "", "")
    resume = CanonicalResume.model_validate({
        "certifications": [
            "医疗保健支持专员认证 (2009)",
            "认证患者护理技师 (2000)",
            "- 医疗保健支持专员认证",
            "- 认证患者护理技师",
        ],
    })

    result, _, _, diagnostics = _apply_fact_compiler_candidate(
        resume,
        source,
        scaffold=_deterministic_fallback(source_text, "", ""),
    )

    assert diagnostics["accepted"] is True
    assert result.certifications == [
        "- 医疗保健支持专员认证",
        "- 认证患者护理技师",
    ]


def test_pipeline_fact_compiler_rejects_unowned_resume_metric_highlight(monkeypatch):
    monkeypatch.setenv("FACT_COMPILER_MODE", "on")
    source = build_source_bundle("处理超过5个预约。", "", "")
    resume = CanonicalResume.model_validate({
        "additional_sections": {"经历亮点": ["处理超过5个预约。"]},
    })

    result, _, _, diagnostics = _apply_fact_compiler_candidate(
        resume,
        source,
        scaffold=None,
    )

    assert diagnostics["accepted"] is True
    assert "经历亮点" not in result.additional_sections


def test_existing_scaffold_audit_reuses_bindings_without_compiler(monkeypatch):
    monkeypatch.setenv("FACT_COMPILER_MODE", "on")
    source = build_source_bundle(
        "",
        """工作经历
甲公司｜产品经理｜2022.01-2024.01
负责用户访谈并输出需求优先级清单
""",
        "",
    )
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "产品经理",
            "period": "2022.01-2024.01",
            "bullets": ["负责用户访谈并输出需求优先级清单"],
        }],
    })
    bindings = bind_resume_evidence(resume, source)

    diagnostics = _audit_existing_scaffold(resume, source, bindings)

    assert diagnostics["accepted"] is True
    assert diagnostics["safe"] is True
    assert diagnostics["after_unsupported"] == 0
    assert diagnostics["after_ownership_errors"] == 0
    assert diagnostics["audit_reused_existing_bindings"] is True


def test_dense_query_audited_scaffold_returns_before_composer_and_compiler(monkeypatch):
    monkeypatch.setenv("FACT_COMPILER_MODE", "on")
    monkeypatch.setattr(v2_pipeline, "_QUERY_FACT_COMPILER_FASTPATH_MIN_BLOCKS", 1)
    monkeypatch.setattr(
        v2_pipeline,
        "_QUERY_FACT_COMPILER_FASTPATH_MIN_ATOMIC_RECALL",
        0.0,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an accepted query scaffold must not invoke an LLM/compiler tail")

    monkeypatch.setattr(v2_pipeline, "compose_from_query", forbidden)
    monkeypatch.setattr(v2_pipeline, "_apply_fact_compiler_candidate", forbidden)

    result = run_v2_pipeline(
        "",
        """以下是我的全部个人信息，请勿编造。
工作经历
甲公司｜产品经理｜2022.01-2024.01
负责用户访谈并输出需求优先级清单
""",
        "",
    )

    assert result.resume.experience[0].organization == "甲公司"
    assert result.resume.experience[0].role == "产品经理"
    assert any(
        "需求优先级清单" in bullet
        for bullet in result.resume.experience[0].bullets
    )
    assert result.changes[0].reason.startswith("Generated from an evidence-gated")
