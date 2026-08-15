from acceptance_testset.run_api_testset import _filter_cases, _multipart


def test_acceptance_case_filter_preserves_requested_order_and_supports_commas():
    cases = [{"id": "A"}, {"id": "B"}, {"id": "C"}]

    assert _filter_cases(cases, ["C,A", "B"]) == [
        {"id": "C"}, {"id": "A"}, {"id": "B"},
    ]


def test_multipart_supports_inline_development_cv_and_jd_text():
    body, boundary = _multipart({
        "id": "dev-1",
        "query": "优化简历",
        "cv_text": "甲公司 产品经理 负责需求分析",
        "jd_text": "招聘产品经理",
    })

    assert boundary.encode() in body
    assert b'name="cv"; filename="dev-1.txt"' in body
    assert "甲公司 产品经理 负责需求分析".encode() in body
    assert b'name="target_jd"' in body
    assert "招聘产品经理".encode() in body
