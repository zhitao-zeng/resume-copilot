from __future__ import annotations

import json
from pathlib import Path

from core.fact_id_classifier_shadow import (
    CandidateChoice,
    CandidateClassification,
    CandidateSpan,
    build_candidate_inventory,
    choices_to_predictions,
    validate_candidate_classification,
)


CASES = Path("validation_sets/segment_grounded_development/cases.jsonl")


def _cases():
    return [json.loads(line) for line in CASES.read_text().splitlines()]


def test_candidate_pool_covers_every_dev_gold_span() -> None:
    for case in _cases():
        bundle, candidates = build_candidate_inventory(
            case.get("cv_text", ""), case.get("query_text", ""), case.get("jd_text", "")
        )
        documents = {document.source_id: document.text for document in bundle.documents}
        for gold in case.get("expected_fields", []):
            source = documents[gold["source_id"]]
            cursor = 0
            start = -1
            for _ in range(int(gold.get("occurrence") or 1)):
                start = source.find(gold["quote"], cursor)
                cursor = start + len(gold["quote"])
            assert start >= 0, (case["id"], gold)
            assert any(
                item.source_id == gold["source_id"]
                and item.start <= start
                and item.end >= start + len(gold["quote"])
                for item in candidates
            ), (case["id"], gold)


def test_empty_profile_has_no_candidate_facts() -> None:
    case = next(case for case in _cases() if case["id"] == "SGD-EMPTY-08")
    _, candidates = build_candidate_inventory(
        case["cv_text"], case["query_text"], case["jd_text"]
    )
    assert candidates == ()


def test_classifier_accepts_known_group_and_reconstructs_source() -> None:
    case = next(case for case in _cases() if case["id"] == "SGD-TEACHER-04")
    _, candidates = build_candidate_inventory(
        case["cv_text"], case["query_text"], case["jd_text"]
    )
    role = next(item for item in candidates if item.text == "数学教师")
    result = validate_candidate_classification(
        CandidateClassification.model_validate({
            "choices": [{
                "candidate_id": role.candidate_id,
                "field_type": "role",
                "group_id": role.group_id,
            }],
        }),
        candidates,
    )
    assert result.valid
    prediction = choices_to_predictions(result)[0]
    assert prediction["value"] == "数学教师"
    assert prediction["parts"][0]["text"] == "数学教师"
    assert prediction["record_id"] == role.group_id


def test_unknown_duplicate_and_wrong_group_are_rejected() -> None:
    case = next(case for case in _cases() if case["id"] == "SGD-MULTI-PRODUCT-01")
    _, candidates = build_candidate_inventory(
        case["cv_text"], case["query_text"], case["jd_text"]
    )
    role = next(item for item in candidates if item.text == "产品助理")
    result = validate_candidate_classification({
        "choices": [
            {"candidate_id": "unknown", "field_type": "role", "group_id": "x"},
            {"candidate_id": role.candidate_id, "field_type": "role", "group_id": "wrong"},
            {"candidate_id": role.candidate_id, "field_type": "role", "group_id": role.group_id},
        ],
    }, candidates)
    assert not result.valid
    assert result.unknown_candidate_count == 1
    assert any(issue.code == "invalid_group" for issue in result.issues)
    assert not choices_to_predictions(result)


def test_unassigned_mode_is_opt_in_and_keeps_strict_default() -> None:
    candidate = CandidateSpan(
        candidate_id="c0000",
        source_id="resume",
        source_type="resume",
        start=0,
        end=4,
        text="事实",
        group_id=None,
    )
    extraction = CandidateClassification(
        choices=[CandidateChoice(candidate_id="c0000", field_type="action")]
    )
    strict = validate_candidate_classification(extraction, [candidate])
    relaxed = validate_candidate_classification(
        extraction, [candidate], allow_unassigned_records=True
    )
    assert not strict.valid
    assert relaxed.valid


def test_duplicate_choice_relaxation_is_opt_in() -> None:
    candidate = CandidateSpan(
        candidate_id="c0000",
        source_id="resume",
        source_type="resume",
        start=0,
        end=4,
        text="事实",
        group_id=None,
    )
    extraction = CandidateClassification(
        choices=[
            CandidateChoice(candidate_id="c0000", field_type="action"),
            CandidateChoice(candidate_id="c0000", field_type="action"),
        ]
    )
    strict = validate_candidate_classification(extraction, [candidate])
    relaxed = validate_candidate_classification(
        extraction,
        [candidate],
        allow_unassigned_records=True,
        allow_duplicate_choices=True,
    )
    assert not strict.valid
    assert relaxed.valid
    assert len(relaxed.choices) == 1


def test_partial_mode_abstains_on_placeholder_without_relaxing_strict_default() -> None:
    candidate = CandidateSpan(
        candidate_id="c0000",
        source_id="resume",
        source_type="resume",
        start=0,
        end=8,
        text="[公司] Technologies",
        group_id="resume:record:1",
    )
    extraction = CandidateClassification(
        choices=[CandidateChoice(
            candidate_id="c0000",
            field_type="organization",
            group_id="resume:record:1",
        )]
    )
    strict = validate_candidate_classification(extraction, [candidate])
    partial = validate_candidate_classification(
        extraction,
        [candidate],
        allow_unassigned_records=True,
        allow_partial_candidates=True,
    )
    assert not strict.valid
    assert any(issue.code == "placeholder_candidate" for issue in strict.issues)
    assert partial.valid
    assert partial.choices == ()
