from __future__ import annotations

from tools.evaluate_segment_grounded_shadow import (
    _semantic_signature,
    resolve_gold,
    score_predictions,
)


def _case():
    return {
        "id": "unit",
        "cv_text": "星河科技｜产品经理｜2023.01-2025.06",
        "query_text": "",
        "jd_text": "",
        "expected_fields": [
            {
                "scope": "record",
                "record_id": "work-1",
                "record_type": "work",
                "field_type": "organization",
                "source_id": "resume",
                "quote": "星河科技",
            },
            {
                "scope": "record",
                "record_id": "work-1",
                "record_type": "work",
                "field_type": "role",
                "source_id": "resume",
                "quote": "产品经理",
            },
        ],
    }


def test_gold_quotes_resolve_to_exact_source_offsets() -> None:
    gold = resolve_gold(_case())
    assert [(item["start"], item["end"], item["quote"]) for item in gold] == [
        (0, 4, "星河科技"),
        (5, 9, "产品经理"),
    ]


def test_exact_field_and_ownership_metrics() -> None:
    gold = resolve_gold(_case())
    predictions = [
        {
            "prediction_id": "p1",
            "scope": "record",
            "record_id": "pred-1",
            "record_type": "work",
            "field_type": "organization",
            "parts": [{"source_id": "resume", "start": 0, "end": 4, "text": "星河科技"}],
            "value": "星河科技",
        },
        {
            "prediction_id": "p2",
            "scope": "record",
            "record_id": "pred-1",
            "record_type": "work",
            "field_type": "role",
            "parts": [{"source_id": "resume", "start": 5, "end": 9, "text": "产品经理"}],
            "value": "产品经理",
        },
    ]
    score = score_predictions(predictions, gold)
    assert score["exact"]["f1"] == 1.0
    assert score["overlap"]["f1"] == 1.0
    assert score["ownership"]["f1"] == 1.0
    assert score["counts"]["critical_unsupported_additions"] == 0
    assert _semantic_signature(predictions) == _semantic_signature(gold)


def test_wrong_role_type_is_a_critical_unsupported_addition() -> None:
    gold = resolve_gold(_case())
    predictions = [{
        "prediction_id": "p1",
        "scope": "record",
        "record_id": "pred-1",
        "record_type": "work",
        "field_type": "role",
        "parts": [{"source_id": "resume", "start": 0, "end": 4, "text": "星河科技"}],
        "value": "星河科技",
    }]
    score = score_predictions(predictions, gold)
    assert score["overlap"]["recall"] == 0.0
    assert score["counts"]["critical_unsupported_additions"] == 1
