from __future__ import annotations

from tools.diagnose_segment_context_variance import (
    REPEATS,
    classify_cause,
    condition_segments,
    endpoint_for,
)


def _case():
    return {
        "id": "diagnostic",
        "cv_text": "甲公司|工程师",
        "query_text": "目标：技术经理",
        "jd_text": "要求五年经验",
    }


def test_conditions_keep_true_source_metadata_and_exclude_jd() -> None:
    assert {segment.source_type for segment in condition_segments(_case(), "full")} == {
        "resume",
        "query",
    }
    assert {segment.source_type for segment in condition_segments(_case(), "resume_only")} == {
        "resume",
    }
    assert {segment.source_type for segment in condition_segments(_case(), "query_only")} == {
        "query",
    }


def test_endpoint_assignment_rotates_by_repeat() -> None:
    endpoints = ["e0", "e1", "e2", "e3"]
    assert [
        endpoint_for(item_index=0, repeat=repeat, endpoints=endpoints)
        for repeat in range(1, REPEATS + 1)
    ] == ["e0", "e1", "e2"]


def test_frozen_causal_rules_distinguish_context_and_variance() -> None:
    assert classify_cause(
        [["deliverable"]] * 3,
        [["organization"]] * 3,
        expected_type="deliverable",
    ) == "context_effect"
    assert classify_cause(
        [["deliverable"], ["organization"], ["deliverable"]],
        [["deliverable"], ["organization"], ["deliverable"]],
        expected_type="deliverable",
    ) == "decoding_variance"
    assert classify_cause(
        [["deliverable"], ["organization"], ["deliverable"]],
        [["organization"]] * 3,
        expected_type="deliverable",
    ) == "mixed"
