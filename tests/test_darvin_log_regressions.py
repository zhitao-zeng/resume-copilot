"""Regression cases transcribed from the ee7a640 Darvin diagnostic log."""

from resume_optimizer import OptimizationOutcome
from source_adapter import build_source_bundle, candidate_blocks
from v2_pipeline import (
    _canonical_to_v1_format,
    _compact_canonical,
    _deterministic_fallback,
    _grounded_source_fallback,
    run_v2_pipeline,
)
from v2_schemas import CanonicalResume, Education


CASE_34_QUERY = (
    "我是会计学专业毕业的，后来因为对编程比较感兴趣，自学转行做前端开发。"
    "学过HTML、CSS、JavaScript和React，也跟着课程做过一些项目，比如电商商城页面和后台管理系统。"
    "不过目前还没有正式的开发工作经历，也没有整理过简历。想找一份初级前端开发工程师的工作，"
    "希望你帮我把现有经历整理成一份简历，如果信息不够的话也告诉我需要补充哪些内容。"
)

CASE_41_QUERY = (
    "想转运营。 之前一直做客户成功和售后服务，主要负责客户培训、问题处理和客户关系维护。"
    "工作中经常需要统计数据、分析客户使用情况，也会策划一些用户活动提高留存。 "
    "希望帮我整理一份偏用户运营方向的简历。"
)

CASE_44_QUERY = (
    "想找法务相关实习。 目前是中国政法大学研二学生，方向偏商法。"
    "做过两段律所实习，一段主要负责合同审查，另一段负责法律检索和案例分析。 "
    "会用北大法宝和Alpha系统。 帮我整理一份法务方向简历。"
)

CASE_16_CV = """工作经历
深圳市雅诺达电子商务有限公司
购物直播
2022-06 至 2022-12
1. 根据市场需求进行选品与定价，完成全部服装搭配及短视
频的拍摄，平均日上新50件。
4. 针对直播情况进行复盘，双十一期间成交额3w+，当
月销售率提升150%+。
项目经历
抖音账号运营
2022-01 至今
1. 观看100+个热门账号的爆款视频，分析用户喜好，锁定增长
的领域，确定个人抖音账号的主体风格和定位。
在校经历
社团经历 ◢
校艺术团主持队部员
2021-01 至 2022-01
1. 从50+人选拔面试入选校艺术团校主持人队。
"""

CASE_20_CV = """教育背景
波士顿大学
公共关系|本科
工作经历
Wonderlab
市场营销实习生
项目经历
抖音个人IP账号运营
在校经历
XX活动策划负责人
职业技能
Premiere Pro 熟练
兴趣爱好
滑雪
1. 根据品牌定位和用户分析，制定新媒体渠道营销投放策略。
2.挖掘推广渠道，负责合作洽谈及关系维护。
1. 深度体验抖音、快手、小红书等平台，确定账号定位。
2.展开对标账号信息收集，确定账号运营计划。
1. 关注学生活动需求，组织策划活动。
2. 调研选取活动举办地址，推进活动如期进行。
2022-05 至2022-10
3. 对市场营销的投放效果负责，监控推广数据。
4.把握市场动态，对接有投放需求商家。
"""

CASE_22_CV = """教育背景
中国政法大学法学本科
工作经历
XX事务所
民事诉讼律师
在校经历
校园法律服务咨询中心
会长
职业技能
案件流程熟练
兴趣爱好
影评（曾获10w+点赞）
阅读（政法领域）
- 精通民事案件办理流程，熟悉案件调解与开庭工作。
1.负责法律咨询，对接当事人进行沟通。
2.帮助当事人梳理案件思路，完成签约。
1.关注热点事件及学生需求，参与策划创意活动。
2.明确活动细节及流程，带领社团统筹策划法律表演。
"""

CASE_27_CV = """工作经历
广西建筑工程有限公司
施工员
工程实训基地
施工员
在校经历
创业协会-宣传部
副部长
论文期刊
建筑工程装饰装修施工的关键技术的研究
职业技能
扎实，熟练操作CAD等制图软件，能够独立制作施工图纸、施工方案。
1. 协助项目经理做好工程开工的准备工作。
2.编制工程总进度计划表和月进度计划表。
1.负责协助项目经理，并协调与统筹小组成员工作。
2.编制小组工程进度计划表，跟进组内成员工作进度。
1. 管理社团内部人员，统筹部门日常工作。
2.有效宣传创协各项活动，累计策划4场活动。
求职意向：施工员
Photoshop 熟练 / AI 掌握 / office 熟练 / CAD制图精通
"""

CASE_26_CV = """知页简历女
求职意向：内科医生
-13800000000
- knowpage
- job@weiapp.com
基本信息
籍贯：辽宁沈阳
居住地：辽宁沈阳
政治面貌：中共党员
教育背景
2020-09至
中国医科大学
内科|硕士
2023-06
GPA: 4.5
2016-09至
中国医科大学
内科|本科
2020-06
GPA: 4.4
工作经历
2021-06至
大连市第三人民医院
2022-02
实习生
2020-06至
辽宁省人民医院
2020-12
内科实习生
告上级医师。
职业技能
利用医疗设备、器械提供物理疗法·精通
分析心电图、胸片·精通
呼吸机、心电图机等常见仪器·精通
化学疗法、输氧、补充营养物质·精通
资格证书
执业医师资格证 2021-03 获得
雅思（7.5分）2020-03 获得
国家计算机二级 2018-09 获得
大学英语六级(580)
内科主治医师（在考）
普通话一乙 2017-09获得
2017-06获得
在校经历
社会实践
2019-06至
和平区中心医院-志愿者
2019-07
病人使用一卡通和预约挂号等服务。
奖学金
国家奖学金（全校仅2%)
求职意向：内科医生；一年内科医生实习工作经验，熟悉药物治疗，并能熟练利用医疗设备、器械提供物理疗法，
输氧、营养支持、输血、止血、替代治疗等；曾参加第八届全国高等医学院校大学生临床技能竞赛，获得一等奖。
2.配合主治医师完成接诊、查房、医疗文件的书写和治疗工作，对术后及危重的200名病人勤巡视，并将病人情况及时报
为300名门急诊患者提供温馨服务，包括导医、导诊、咨询、费用查询、护送、控烟、秩序维护等服务，积极引导、协助
1. 负责配合主治医生管理100位床位患者的住院期间所有诊疗过程；
2.负责值班，轮转6个内科门诊，并负责6个内科及急危重病诊疗规范及临床用药；
3. 负责检索、查阅中英文文献，并归纳总结，用于临床和科研工作；
4.学会利用医疗设备、器械、药物、等数10种手段治疗内科疾病。
1. 在科主任领导和主治医师指导下，分管300个病床，担任值班工作；
"""


def _disable_llm_stages(monkeypatch, composed: CanonicalResume | None = None) -> None:
    monkeypatch.setattr(
        "v2_pipeline.compose_from_query",
        lambda _query, _jd: (composed or CanonicalResume()).model_copy(deep=True),
    )
    monkeypatch.setattr(
        "v2_pipeline.optimize_resume_with_provenance",
        lambda resume, _jd="": OptimizationOutcome(resume=resume),
    )


def test_log_query_facts_survive_clause_segmentation() -> None:
    facts_34 = [block.text for block in candidate_blocks(build_source_bundle("", CASE_34_QUERY, ""))]
    assert "学过HTML、CSS、JavaScript和React" in facts_34
    assert "比如电商商城页面和后台管理系统" in facts_34
    assert not any("希望你帮我" in value for value in facts_34)

    facts_41 = [block.text for block in candidate_blocks(build_source_bundle("", CASE_41_QUERY, ""))]
    assert facts_41 == [
        "之前一直做客户成功和售后服务",
        "主要负责客户培训、问题处理和客户关系维护",
        "工作中经常需要统计数据、分析客户使用情况",
        "也会策划一些用户活动提高留存",
    ]

    facts_44 = [block.text for block in candidate_blocks(build_source_bundle("", CASE_44_QUERY, ""))]
    assert "一段主要负责合同审查" in facts_44
    assert "另一段负责法律检索和案例分析" in facts_44
    assert "会用北大法宝和Alpha系统" in facts_44
    assert not any("帮我整理" in value for value in facts_44)


def test_log_case_34_generates_grounded_transition_resume(monkeypatch) -> None:
    _disable_llm_stages(monkeypatch)

    result = run_v2_pipeline("", CASE_34_QUERY, "")
    resume = result.resume

    assert resume.meta.target_role == "初级前端开发工程师"
    assert [(item.school, item.major) for item in resume.education] == [("", "会计学")]
    assert [item.name for item in resume.projects] == ["电商商城页面", "后台管理系统"]
    assert {item.name for item in resume.skills.items} >= {
        "前端开发", "HTML", "CSS", "JavaScript", "React",
    }
    assert "framework" not in result.resume_dict
    assert resume.additional_sections == {}


def test_log_case_41_keeps_all_customer_success_duties(monkeypatch) -> None:
    _disable_llm_stages(monkeypatch)

    resume = run_v2_pipeline("", CASE_41_QUERY, "").resume

    assert resume.meta.target_role == "用户运营"
    assert len(resume.experience) == 1
    assert resume.experience[0].role == "客户成功和售后服务"
    assert set(resume.experience[0].bullets) == {
        "主要负责客户培训、问题处理和客户关系维护",
        "工作中经常需要统计数据、分析客户使用情况",
        "也会策划一些用户活动提高留存",
    }
    assert "工作或实习经历包括客户成功和售后服务" in resume.summary


def test_log_case_44_recovers_sections_missing_from_partial_composer(monkeypatch) -> None:
    _disable_llm_stages(
        monkeypatch,
        CanonicalResume(education=[Education(school="中国政法大学")]),
    )

    resume = run_v2_pipeline("", CASE_44_QUERY, "").resume

    assert resume.meta.target_role == "法务"
    assert [(item.school, item.major) for item in resume.education] == [
        ("中国政法大学", "商法"),
    ]
    assert len(resume.experience) == 1
    assert resume.experience[0].organization == ""
    assert resume.experience[0].role == "律所实习"
    assert set(resume.experience[0].bullets) == {
        "一段主要负责合同审查",
        "另一段负责法律检索和案例分析",
    }
    assert {item.name for item in resume.skills.items} == {"北大法宝", "Alpha系统"}
    assert resume.additional_sections == {}


def test_log_case_16_joins_ocr_tails_without_fake_roles() -> None:
    resume = _compact_canonical(_deterministic_fallback(CASE_16_CV, "", ""))

    assert [(item.organization, item.role) for item in resume.experience] == [
        ("深圳市雅诺达电子商务有限公司", "购物直播"),
    ]
    assert len(resume.experience[0].bullets) == 2
    assert "短视频的拍摄" in resume.experience[0].bullets[0]
    assert "当月销售率提升150%+" in resume.experience[0].bullets[1]
    assert all(item.role != "月销售" for item in resume.experience)
    assert resume.projects[0].name == "抖音账号运营"
    assert "增长的领域" in resume.projects[0].bullets[0]
    assert [(item.organization, item.role) for item in resume.activities] == [
        ("校艺术团主持队", "部员"),
    ]


def test_log_case_20_routes_decolumnized_numbered_lists() -> None:
    resume = _compact_canonical(_deterministic_fallback(
        CASE_20_CV,
        "帮我把简历修改成更适合互联网市场营销岗位的版本。",
        "",
    ))

    assert resume.meta.target_role == "互联网市场营销"
    assert [(item.school, item.major) for item in resume.education] == [
        ("波士顿大学", "公共关系"),
    ]
    assert resume.experience[0].organization == "Wonderlab"
    assert resume.experience[0].role == "市场营销实习生"
    assert len(resume.experience[0].bullets) == 4
    assert resume.projects[0].name == "抖音个人IP账号运营"
    assert len(resume.projects[0].bullets) == 2
    assert len(resume.activities[0].bullets) == 2


def test_log_case_22_does_not_turn_duty_into_activity_identity() -> None:
    resume = _compact_canonical(_deterministic_fallback(CASE_22_CV, "", ""))

    assert [(item.organization, item.role) for item in resume.experience] == [
        ("XX事务所", "民事诉讼律师"),
    ]
    assert len(resume.experience[0].bullets) == 2
    assert [(item.organization, item.role) for item in resume.activities] == [
        ("校园法律服务咨询中心", "会长"),
    ]
    assert "带领社团" not in resume.activities[0].organization
    assert resume.additional_sections["兴趣爱好"] == [
        "影评（曾获10w+点赞）", "阅读（政法领域）",
    ]


def test_log_case_27_keeps_duties_out_of_skills_and_internal_sections() -> None:
    resume = _compact_canonical(_deterministic_fallback(CASE_27_CV, "", ""))
    rendered = _canonical_to_v1_format(resume)

    assert [(item.organization, item.role, len(item.bullets)) for item in resume.experience] == [
        ("广西建筑工程有限公司", "施工员", 2),
        ("工程实训基地", "施工员", 2),
    ]
    assert [(item.organization, item.role, len(item.bullets)) for item in resume.activities] == [
        ("创业协会", "副部长", 2),
    ]
    assert resume.publications == ["建筑工程装饰装修施工的关键技术的研究"]
    assert not any("负责" in item.name or "管理社团" in item.name for item in resume.skills.items)
    assert not any("补充" in title or "待整理" in title for title in rendered["additional_sections"])


def test_log_case_26_rebinds_medical_columns_without_fake_records() -> None:
    resume = _compact_canonical(_deterministic_fallback(CASE_26_CV, "", ""))

    assert [(item.school, item.degree, item.major, item.period) for item in resume.education] == [
        ("中国医科大学", "硕士", "内科", "2020-09 至 2023-06"),
        ("中国医科大学", "本科", "内科", "2016-09 至 2020-06"),
    ]
    assert [(item.organization, item.role, item.period, len(item.bullets)) for item in resume.experience] == [
        ("大连市第三人民医院", "实习生", "2021-06 至 2022-02", 2),
        ("辽宁省人民医院", "内科实习生", "2020-06 至 2020-12", 4),
    ]
    assert resume.experience[0].bullets[1].endswith("及时报告上级医师。")
    assert [(item.organization, item.role, item.period) for item in resume.activities] == [
        ("和平区中心医院", "志愿者", "2019-06 至 2019-07"),
    ]
    assert resume.activities[0].bullets == [
        "为300名门急诊患者提供温馨服务，包括导医、导诊、咨询、费用查询、护送、控烟、秩序维护等服务，"
        "积极引导、协助病人使用一卡通和预约挂号等服务。",
    ]
    assert {item.name for item in resume.skills.items} == {
        "利用医疗设备、器械提供物理疗法",
        "分析心电图、胸片",
        "呼吸机、心电图机等常见仪器",
        "化学疗法、输氧、补充营养物质",
    }
    assert "2017-06获得" not in resume.certifications
    assert any("临床技能竞赛" in value and "一等奖" in value for value in resume.awards)
    assert all(item.organization != "大学" and item.role != "执业医师" for item in resume.experience)


def test_log_case_26_records_survive_the_evidence_gate() -> None:
    source = build_source_bundle(CASE_26_CV, "", "")
    evidence = "\n".join(block.text for block in candidate_blocks(source))

    result, coverage, _missing = _grounded_source_fallback(
        CASE_26_CV,
        "",
        "",
        source,
        evidence,
    )

    assert coverage >= 0.80
    assert [(item.organization, len(item.bullets)) for item in result.resume.experience] == [
        ("大连市第三人民医院", 2),
        ("辽宁省人民医院", 4),
    ]
    assert [(item.organization, item.role) for item in result.resume.activities] == [
        ("和平区中心医院", "志愿者"),
    ]


def test_reordered_object_fragment_join_is_generic_and_has_no_fixed_window() -> None:
    filler = "\n".join(f"荣誉项目{index:02d}" for index in range(25))
    source = build_source_bundle(
        "校园经历\n"
        "社区服务中心-成员\n"
        "2023-01 至 2023-06\n"
        "居民完成线上预约登记。\n"
        "荣誉奖项\n"
        f"{filler}\n"
        "为200名社区居民提供咨询、秩序维护等服务，并协助",
        "",
        "",
    )

    joined = [
        block for block in candidate_blocks(source)
        if "为200名社区居民" in block.text
    ]
    assert len(joined) == 1
    assert joined[0].text.endswith("协助居民完成线上预约登记。")
    assert joined[0].section_hint == "activities"
    assert joined[0].record_id
    assert not any(
        block.text == "居民完成线上预约登记。"
        for block in candidate_blocks(source)
    )


def test_reordered_object_fragment_join_rejects_ambiguous_destinations() -> None:
    source = build_source_bundle(
        "校园经历\n"
        "甲协会-成员\n"
        "居民完成登记。\n"
        "乙协会-成员\n"
        "用户完成注册。\n"
        "荣誉奖项\n"
        "荣誉A\n"
        "为社区用户提供现场服务并协助",
        "",
        "",
    )
    values = [block.text for block in candidate_blocks(source)]

    assert "为社区用户提供现场服务并协助" in values
    assert "居民完成登记。" in values
    assert "用户完成注册。" in values
