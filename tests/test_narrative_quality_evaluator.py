from tools.evaluate_narrative_quality import evaluate_rows


def _row(*, scenario="scenario1", bullets=None, summary="", reply="", framework=False):
    resume_data = {
        "experience": [{"bullets": bullets or []}],
        "summary": summary,
        "meta": {"target_role": "产品经理"},
    }
    if framework:
        resume_data["framework"] = {"sections": []}
    return {
        "id": "case",
        "raw": {
            "scenario": scenario,
            "resume_data": resume_data,
            "reply_text": reply,
        },
    }


def test_evaluator_accepts_direct_and_acceptance_nested_rows():
    direct = _row(
        bullets=["负责需求分析，通过用户访谈输出需求清单"],
        summary="产品经理。",
        reply="生成方向总结\n缺失信息\n岗位匹配与建议\n时间或内容冲突",
    )
    nested = {"id": "nested", "raw": {"raw": direct["raw"]}}

    report = evaluate_rows([direct, nested])["summary"]

    assert report["case_count"] == 2
    assert report["bullet_count"] == 2
    assert report["bullet_action_rate"] == 1.0
    assert report["bullet_accomplishment_rate"] == 1.0
    assert report["reply_contract_rate"] == 1.0


def test_short_noun_fragment_is_distinguished_from_context_label():
    row = _row(bullets=["客户沟通", "行业：医疗器械", "负责客户需求整理"])

    report = evaluate_rows([row])["summary"]

    assert report["bullet_count"] == 3
    assert report["bullet_fragment_count"] == 1
    assert report["bullet_context_only_count"] == 1


def test_framework_does_not_count_as_missing_summary():
    row = _row(
        scenario="scenario4",
        bullets=[],
        summary="",
        framework=True,
        reply="生成方向总结\n缺失信息\n岗位匹配与建议\n时间或内容冲突",
    )

    report = evaluate_rows([row])["summary"]

    assert report["framework_case_count"] == 1
    assert report.get("nonframework_empty_summary_count", 0) == 0
