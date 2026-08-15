from __future__ import annotations

from collections import Counter

from tools.evaluate_fact_id_classifier_holdout import load_holdout_cases


def test_holdout_v2_loader_is_frozen_and_hash_checked() -> None:
    cases = load_holdout_cases()
    assert len(cases) == 60
    assert Counter(case["scenario"] for case in cases) == {
        "scenario1": 15,
        "scenario2": 15,
        "scenario3": 15,
        "scenario4": 15,
    }
    assert all("eligible_units" in case for case in cases)
    assert sum(not case["cv_text"].strip() for case in cases) >= 15
