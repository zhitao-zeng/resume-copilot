from acceptance_testset.run_api_testset import _filter_cases


def test_acceptance_case_filter_preserves_requested_order_and_supports_commas():
    cases = [{"id": "A"}, {"id": "B"}, {"id": "C"}]

    assert _filter_cases(cases, ["C,A", "B"]) == [
        {"id": "C"}, {"id": "A"}, {"id": "B"},
    ]
