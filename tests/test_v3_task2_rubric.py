"""Task 2 rubric quick wins: ordering, skills taxonomy, prompts, overlaps."""
from types import SimpleNamespace

from core.v3.pipeline import _missing_fields, _record_detail_prompts
from core.v3.planner import _record_start_key
from core.v3.resume_adapter import _skill_bucket
from core.v3.time_conflicts import experience_overlaps, parse_period


# --- 2.1 planner time key -------------------------------------------------

def _fact(fact_type, text):
    return SimpleNamespace(fact_type=fact_type, text=text)


def test_record_start_key_uses_latest_period_start():
    fact_map = {
        "f1": _fact("period", "2020年3月 - 2022年6月"),
        "f2": _fact("action", "负责贷款审核 2019"),
    }
    assert _record_start_key(["f1", "f2"], fact_map) == 202003


def test_record_start_key_undated_record():
    fact_map = {"f1": _fact("action", "协助组织社区活动")}
    assert _record_start_key(["f1"], fact_map) == -1


def test_record_start_key_year_only():
    fact_map = {"f1": _fact("period", "2019 - 2021")}
    assert _record_start_key(["f1"], fact_map) == 201901


# --- 2.2 skill routing ----------------------------------------------------

def test_skill_bucket_credential_fact_type_wins():
    assert _skill_bucket("注册会计师", {"credential"}) == "certifications"


def test_skill_bucket_language_morphology():
    assert _skill_bucket("英语（流利）", set()) == "natural_languages"
    assert _skill_bucket("普通话", set()) == "natural_languages"
    assert _skill_bucket("Spanish - fluent", set()) == "natural_languages"


def test_skill_bucket_latin_token_is_tool():
    assert _skill_bucket("Python", set()) == "tools"
    assert _skill_bucket("Excel", set()) == "tools"


def test_skill_bucket_c_language_and_sentences_stay_others():
    # "C语言" must not be treated as a natural language, and long duty
    # sentences (D17's failure shape) never masquerade as a category.
    assert _skill_bucket("C语言", set()) == "others"
    assert _skill_bucket("使用人力资源系统处理薪资，并为团队领导提供系统功能", set()) == "others"


# --- 2.4 missing prompts --------------------------------------------------

def test_missing_fields_phone_and_email_are_individually_required():
    data = {"meta": {"phone": "13800000000"}, "summary": "x", "education": [1],
            "experience": [1], "skills": ["a"]}
    data["meta"]["name"] = "张三"
    missing = _missing_fields(data)
    fields = {item["field"] for item in missing}
    assert "email" in fields
    assert "phone" not in fields


def test_record_detail_prompts_aggregate_per_section():
    data = {
        "experience": [
            {"company": "A公司", "role": "工程师", "period": "2020年1月 - 2021年2月", "bullets": ["x"]},
            {"company": "B公司", "bullets": ["y"]},
        ],
    }
    prompts = _record_detail_prompts(data)
    assert len(prompts) == 1
    assert prompts[0].startswith("工作/实习经历：")
    assert "1 段缺少起止时间" in prompts[0]
    assert "1 段缺少岗位" in prompts[0]


def test_record_detail_prompts_silent_when_complete():
    data = {
        "experience": [
            {"company": "A公司", "role": "工程师", "period": "2020年1月 - 2021年2月"},
        ],
    }
    assert _record_detail_prompts(data) == []


# --- 2.5 time overlap -----------------------------------------------------

def test_parse_period_chinese_range():
    assert parse_period("2020年3月 - 2025年6月") == (202003, 202506)


def test_parse_period_ongoing():
    assert parse_period("2023年4月至今") == (202304, 999912)


def test_parse_period_reversed_is_rejected():
    assert parse_period("2025年6月 - 2020年3月") is None


def test_parse_period_english_months():
    assert parse_period("Apr 2023 - May 2024") == (202304, 202405)


def test_overlap_flagged_and_adjacency_tolerated():
    records = [
        {"company": "甲公司", "period": "2020年1月 - 2021年6月"},
        {"company": "乙公司", "period": "2021年1月 - 2022年3月"},
        {"company": "丙公司", "period": "2022年3月 - 2023年1月"},
    ]
    overlaps = experience_overlaps(records)
    assert len(overlaps) == 1
    assert "甲公司" in overlaps[0]["description"]
    assert "乙公司" in overlaps[0]["description"]
    assert "请确认" in overlaps[0]["description"]


def test_overlap_unparseable_periods_never_flag():
    records = [
        {"company": "甲公司", "period": "早年"},
        {"company": "乙公司", "period": "2021年1月 - 2022年3月"},
    ]
    assert experience_overlaps(records) == []
