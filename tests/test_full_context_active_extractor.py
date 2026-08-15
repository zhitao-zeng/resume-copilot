from __future__ import annotations

import json

from full_context_active_extractor import (
    build_active_segment_windows,
    build_active_window_prompt,
    validate_active_extraction,
)
from segment_grounded_extractor import build_document_segments
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


def test_active_windows_are_non_overlapping_complete_and_domain_neutral() -> None:
    segments = _segments()
    windows = build_active_segment_windows(segments, max_segments=3)
    active_ids = [
        segment_id for window in windows for segment_id in window.active_segment_ids
    ]
    expected = [
        segment.segment_id for segment in segments if segment.source_type != "jd"
    ]

    assert active_ids == expected
    assert len(active_ids) == len(set(active_ids))
    assert [len(window.active_segment_ids) for window in windows] == [3, 3]


def test_prompt_keeps_full_candidate_context_and_marks_only_active_ids() -> None:
    segments = _segments()
    active_ids = build_active_segment_windows(segments, max_segments=3)[0].active_segment_ids
    system_prompt, user_prompt = build_active_window_prompt(segments, active_ids)
    payload = json.loads(user_prompt)

    assert "甲公司" in user_prompt
    assert "技术经理" in user_prompt
    assert "要求5年经验" not in user_prompt
    assert payload["active_segment_ids"] == list(active_ids)
    assert "全部 segments 都可用于理解语义" in system_prompt
    assert "只输出“最左侧/最先出现的引用 segment_id”" in system_prompt


def test_active_validator_keeps_active_owned_field_and_full_record_partition() -> None:
    segments = _segments()
    by_text = {segment.text: segment for segment in segments}
    result = validate_active_extraction({
        "profile_fields": [],
        "records": [{
            "record_type": "work",
            "segment_ids": [
                by_text["甲公司"].segment_id,
                by_text["工程师"].segment_id,
                by_text["开发平台"].segment_id,
            ],
            "fields": [{
                "field_type": "organization",
                "refs": [{
                    "segment_id": by_text["甲公司"].segment_id,
                    "exact_quote": "甲公司",
                }],
            }],
        }],
        "unassigned_segment_ids": [],
    }, segments, [by_text["甲公司"].segment_id])

    assert result.validation.valid is True
    assert result.active_issues == ()
    assert result.validation.records[0].segment_ids == (
        by_text["甲公司"].segment_id,
        by_text["工程师"].segment_id,
        by_text["开发平台"].segment_id,
    )
    assert result.validation.records[0].fields[0].value == "甲公司"


def test_active_validator_drops_and_flags_inactive_owned_fields() -> None:
    segments = _segments()
    by_text = {segment.text: segment for segment in segments}
    result = validate_active_extraction({
        "profile_fields": [],
        "records": [{
            "record_type": "work",
            "segment_ids": [
                by_text["甲公司"].segment_id,
                by_text["工程师"].segment_id,
            ],
            "fields": [{
                "field_type": "role",
                "refs": [{
                    "segment_id": by_text["工程师"].segment_id,
                    "exact_quote": "工程师",
                }],
            }],
        }],
        "unassigned_segment_ids": [by_text["工程师"].segment_id],
    }, segments, [by_text["甲公司"].segment_id])

    assert result.validation.valid is False
    assert result.validation.records == ()
    assert {issue.code for issue in result.active_issues} == {
        "inactive_field",
        "inactive_unassigned_segment",
    }


def test_multispan_field_is_owned_by_earliest_referenced_segment() -> None:
    segments = _segments()
    by_text = {segment.text: segment for segment in segments}
    result = validate_active_extraction({
        "profile_fields": [],
        "records": [{
            "record_type": "work",
            "segment_ids": [
                by_text["工程师"].segment_id,
                by_text["开发平台"].segment_id,
            ],
            "fields": [{
                "field_type": "action",
                "refs": [
                    {
                        "segment_id": by_text["工程师"].segment_id,
                        "exact_quote": "工程师",
                    },
                    {
                        "segment_id": by_text["开发平台"].segment_id,
                        "exact_quote": "开发平台",
                    },
                ],
            }],
        }],
        "unassigned_segment_ids": [],
    }, segments, [by_text["工程师"].segment_id])

    assert result.validation.valid is True
    assert result.validation.records[0].fields[0].value == "工程师 开发平台"
