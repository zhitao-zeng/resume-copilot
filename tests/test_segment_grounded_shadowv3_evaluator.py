from __future__ import annotations

from tools.evaluate_segment_grounded_shadowv3 import score_eligibility


def _prediction(field_type: str, start: int, end: int):
    return {
        "prediction_id": f"{field_type}-{start}",
        "scope": "record",
        "record_id": "record-1",
        "record_type": "work",
        "field_type": field_type,
        "parts": [{
            "source_id": "resume",
            "start": start,
            "end": end,
            "text": "x" * (end - start),
        }],
        "value": "x" * (end - start),
    }


def test_span_eligibility_is_independent_of_semantic_type() -> None:
    units = [{
        "unit_id": "u1",
        "source_id": "resume",
        "start": 10,
        "end": 20,
        "text": "abcdefghij",
        "section": "工作经历",
    }]
    score = score_eligibility(
        [_prediction("role", 12, 16), _prediction("action", 30, 35)],
        units,
    )

    assert score["candidate_span_precision"] == 0.5
    assert score["eligible_unit_coverage"] == 1.0
    assert score["counts"]["critical_ineligible_predictions"] == 0


def test_ineligible_critical_field_is_counted_but_target_role_is_excluded() -> None:
    score = score_eligibility(
        [_prediction("organization", 0, 4), _prediction("target_role", 5, 9)],
        [],
    )

    assert score["counts"]["factual_predictions"] == 1
    assert score["counts"]["critical_ineligible_predictions"] == 1
    assert score["eligible_unit_coverage"] == 1.0
