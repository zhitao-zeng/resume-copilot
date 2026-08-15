from tools.evaluate_development_results import audit_development_rows


def test_development_evaluator_audits_inline_case_without_candidate_score():
    cases = [{
        "id": "dev-1",
        "scenario": "scenario3",
        "industry": "operations",
        "query": "优化简历",
        "cv_text": "甲公司｜用户运营｜2022-2024\n负责社群维护",
    }]
    result_rows = [{
        "id": "dev-1",
        "ok": True,
        "status": 200,
        "elapsed_s": 1.0,
        "score": 100,
        "raw": {
            "scenario": "scenario3",
            "industry": "operations",
            "files": {"docx": "/tmp/dev.docx"},
            "reply_text": "生成方向总结\n缺失信息\n岗位匹配与建议\n时间或内容冲突",
            "missing_fields": [],
            "resume_data": {
                "experience": [{
                    "company": "甲公司",
                    "role": "用户运营",
                    "period": "2022-2024",
                    "bullets": ["负责社群维护"],
                }],
            },
        },
    }]

    report = audit_development_rows(cases, result_rows)
    summary = report["summary"]

    assert summary["request_success_count"] == 1
    assert summary["audit_success_count"] == 1
    assert summary["atomic_factuality"]["unsupported_atom_count"] == 0
    assert "score" not in summary
