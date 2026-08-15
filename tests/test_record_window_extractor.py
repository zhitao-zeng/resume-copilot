from __future__ import annotations

from record_window_extractor import (
    SegmentWindow,
    WindowValidation,
    aggregate_window_validations,
    build_segment_windows,
)
from segment_grounded_extractor import (
    build_document_segments,
    validate_grounded_extraction,
)
from v2_schemas import SourceDocument


def _segments():
    return build_document_segments([
        SourceDocument(
            source_id="resume",
            source_type="resume",
            text="甲公司|工程师|2020-2022|开发平台",
        ),
        SourceDocument(
            source_id="query",
            source_type="query",
            text="目标岗位|技术经理",
        ),
        SourceDocument(
            source_id="jd",
            source_type="jd",
            text="要求5年经验",
        ),
    ])


def _window(window_id, segments):
    return SegmentWindow(
        window_id=window_id,
        source_id=segments[0].source_id,
        source_type=segments[0].source_type,
        segments=tuple(segments),
        character_count=sum(len(segment.text) for segment in segments),
    )


def _validation(window, *, record_type="work", fields=()):
    return validate_grounded_extraction({
        "profile_fields": [],
        "records": [{
            "record_type": record_type,
            "segment_ids": list(window.segment_ids),
            "fields": [
                {
                    "field_type": field_type,
                    "refs": [{
                        "segment_id": segment.segment_id,
                        "exact_quote": segment.text,
                    }],
                }
                for field_type, segment in fields
            ],
        }],
        "unassigned_segment_ids": [],
    }, window.segments)


def test_fixed_windows_preserve_sources_offsets_and_complete_coverage() -> None:
    segments = _segments()
    windows = build_segment_windows(
        segments,
        max_segments=3,
        max_characters=100,
        overlap_segments=1,
    )

    assert [window.source_id for window in windows] == [
        "resume",
        "resume",
        "query",
    ]
    assert all(window.source_type != "jd" for window in windows)
    covered = {segment_id for window in windows for segment_id in window.segment_ids}
    expected = {
        segment.segment_id for segment in segments if segment.source_type != "jd"
    }
    assert covered == expected
    assert windows[0].segment_ids[-1] == windows[1].segment_ids[0]


def test_single_overlong_segment_is_not_split() -> None:
    document = SourceDocument(
        source_id="resume",
        source_type="resume",
        text="一" * 50,
    )
    segments = build_document_segments([document])
    windows = build_segment_windows(
        segments,
        max_segments=2,
        max_characters=10,
        overlap_segments=1,
    )

    assert len(windows) == 1
    assert windows[0].segments == tuple(segments)
    assert windows[0].character_count == 50


def test_overlapping_record_windows_merge_and_deduplicate_exact_fields() -> None:
    segments = [segment for segment in _segments() if segment.source_id == "resume"]
    first = _window("resume:W0001", segments[:3])
    second = _window("resume:W0002", segments[1:])
    first_validation = _validation(first, fields=(
        ("organization", segments[0]),
        ("role", segments[1]),
        ("period", segments[2]),
    ))
    second_validation = _validation(second, fields=(
        ("role", segments[1]),
        ("period", segments[2]),
        ("action", segments[3]),
    ))

    result = aggregate_window_validations(
        [first, second],
        [
            WindowValidation(window_id=first.window_id, validation=first_validation),
            WindowValidation(window_id=second.window_id, validation=second_validation),
        ],
        segments,
    )

    assert result.validation.valid is True
    assert result.aggregation_issues == ()
    assert len(result.validation.records) == 1
    assert [field.field_type for field in result.validation.records[0].fields] == [
        "organization",
        "role",
        "period",
        "action",
    ]
    assert result.validation.returned_reference_count == 6
    assert result.validation.valid_reference_count == 6


def test_conflicting_record_type_is_rejected_instead_of_guessed() -> None:
    segments = [segment for segment in _segments() if segment.source_id == "resume"]
    first = _window("resume:W0001", segments[:3])
    second = _window("resume:W0002", segments[1:])
    first_validation = _validation(
        first,
        record_type="work",
        fields=(("role", segments[1]),),
    )
    second_validation = _validation(
        second,
        record_type="project",
        fields=(("role", segments[1]),),
    )

    result = aggregate_window_validations(
        [first, second],
        [
            WindowValidation(window_id=first.window_id, validation=first_validation),
            WindowValidation(window_id=second.window_id, validation=second_validation),
        ],
        segments,
    )

    assert result.validation.valid is False
    assert result.validation.records == ()
    assert [issue.code for issue in result.aggregation_issues] == [
        "record_type_conflict",
    ]


def test_overlapping_field_types_are_rejected_instead_of_selected() -> None:
    segments = [segment for segment in _segments() if segment.source_id == "resume"]
    window = _window("resume:W0001", segments)
    validation = validate_grounded_extraction({
        "profile_fields": [],
        "records": [{
            "record_type": "work",
            "segment_ids": list(window.segment_ids),
            "fields": [
                {
                    "field_type": "role",
                    "refs": [{
                        "segment_id": segments[1].segment_id,
                        "exact_quote": segments[1].text,
                    }],
                },
                {
                    "field_type": "skill",
                    "refs": [{
                        "segment_id": segments[1].segment_id,
                        "exact_quote": segments[1].text,
                    }],
                },
            ],
        }],
        "unassigned_segment_ids": [],
    }, window.segments)
    # The frozen per-window decoder flags the duplicate exact reference before
    # aggregation, and aggregation must keep the aggregate invalid.
    result = aggregate_window_validations(
        [window],
        [WindowValidation(window_id=window.window_id, validation=validation)],
        segments,
    )

    assert result.validation.valid is False
    assert any(issue.code == "duplicate_reference" for issue in result.validation.issues)
