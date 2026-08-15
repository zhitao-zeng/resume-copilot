from __future__ import annotations

from segment_grounded_extractor import (
    GroundedExtraction,
    build_document_segments,
    build_shadow_prompt,
    validate_grounded_extraction,
)
from v2_schemas import SourceDocument


def _documents() -> list[SourceDocument]:
    return [
        SourceDocument(
            source_id="resume",
            source_type="resume",
            text="工作经历\n星河科技｜产品经理｜2021.07-2025.06\n负责需求分析。",
        ),
        SourceDocument(
            source_id="query",
            source_type="query",
            text="请优化成产品经理方向",
        ),
        SourceDocument(
            source_id="jd",
            source_type="jd",
            text="要求3年经验",
        ),
    ]


def test_segments_are_exact_source_slices_and_domain_neutral() -> None:
    documents = _documents()
    segments = build_document_segments(documents)

    assert [segment.text for segment in segments] == [
        "工作经历",
        "星河科技",
        "产品经理",
        "2021.07-2025.06",
        "负责需求分析。",
        "请优化成产品经理方向",
        "要求3年经验",
    ]
    source_by_id = {document.source_id: document.text for document in documents}
    assert all(
        source_by_id[segment.source_id][segment.char_start:segment.char_end]
        == segment.text
        for segment in segments
    )
    assert all(segment.page is None and segment.bbox is None for segment in segments)


def test_decoder_reconstructs_values_without_model_supplied_text() -> None:
    segments = build_document_segments(_documents())
    by_text = {segment.text: segment for segment in segments}
    extraction = GroundedExtraction.model_validate({
        "profile_fields": [{
            "field_type": "target_role",
            "refs": [{
                "segment_id": by_text["请优化成产品经理方向"].segment_id,
                "exact_quote": "产品经理",
            }],
        }],
        "records": [{
            "record_type": "work",
            "segment_ids": [
                by_text["星河科技"].segment_id,
                by_text["产品经理"].segment_id,
            ],
            "fields": [
                {
                    "field_type": "organization",
                    "refs": [{
                        "segment_id": by_text["星河科技"].segment_id,
                        "exact_quote": "星河科技",
                    }],
                },
                {
                    "field_type": "role",
                    "refs": [{
                        "segment_id": by_text["产品经理"].segment_id,
                        "exact_quote": "产品经理",
                    }],
                },
            ],
        }],
        "unassigned_segment_ids": [],
    })

    result = validate_grounded_extraction(extraction, segments)

    assert result.valid is True
    assert result.returned_reference_count == result.valid_reference_count == 3
    assert result.profile_fields[0].value == "产品经理"
    assert [field.value for field in result.records[0].fields] == [
        "星河科技",
        "产品经理",
    ]


def test_decoder_rejects_unknown_missing_quote_jd_and_duplicates() -> None:
    segments = build_document_segments(_documents())
    by_text = {segment.text: segment for segment in segments}
    result = validate_grounded_extraction({
        "profile_fields": [],
        "records": [{
            "record_type": "work",
            "segment_ids": [
                by_text["星河科技"].segment_id,
                by_text["产品经理"].segment_id,
            ],
            "fields": [
                {
                    "field_type": "role",
                    "refs": [{"segment_id": "missing:S9999", "exact_quote": "x"}],
                },
                {
                    "field_type": "period",
                    "refs": [{
                        "segment_id": by_text["要求3年经验"].segment_id,
                        "exact_quote": "3年",
                    }],
                },
                {
                    "field_type": "organization",
                    "refs": [{
                        "segment_id": by_text["星河科技"].segment_id,
                        "exact_quote": "不存在的公司",
                    }],
                },
                {
                    "field_type": "role",
                    "refs": [{
                        "segment_id": by_text["产品经理"].segment_id,
                        "exact_quote": "产品经理",
                    }],
                },
                {
                    "field_type": "role",
                    "refs": [{
                        "segment_id": by_text["产品经理"].segment_id,
                        "exact_quote": "产品经理",
                    }],
                },
            ],
        }],
        "unassigned_segment_ids": [by_text["要求3年经验"].segment_id],
    }, segments)

    assert result.valid is False
    assert {issue.code for issue in result.issues} == {
        "unknown_segment",
        "forbidden_source",
        "quote_not_found",
        "duplicate_reference",
        "invalid_unassigned_segment",
    }
    assert result.returned_reference_count == 5
    assert result.valid_reference_count == 2


def test_decoder_rejects_an_ambiguous_exact_quote() -> None:
    document = SourceDocument(
        source_id="resume",
        source_type="resume",
        text="测试测试",
    )
    segments = build_document_segments([document])
    result = validate_grounded_extraction({
        "profile_fields": [],
        "records": [{
            "record_type": "project",
            "segment_ids": [segments[0].segment_id],
            "fields": [{
                "field_type": "action",
                "refs": [{
                    "segment_id": segments[0].segment_id,
                    "exact_quote": "测试",
                }],
            }],
        }],
        "unassigned_segment_ids": [],
    }, segments)

    assert result.valid is False
    assert [issue.code for issue in result.issues] == ["ambiguous_quote"]


def test_record_partitions_must_be_disjoint_and_fields_must_stay_inside() -> None:
    segments = build_document_segments(_documents())
    by_text = {segment.text: segment for segment in segments}
    result = validate_grounded_extraction({
        "profile_fields": [],
        "records": [
            {
                "record_type": "work",
                "segment_ids": [by_text["星河科技"].segment_id],
                "fields": [{
                    "field_type": "organization",
                    "refs": [{
                        "segment_id": by_text["星河科技"].segment_id,
                        "exact_quote": "星河科技",
                    }],
                }],
            },
            {
                "record_type": "project",
                "segment_ids": [by_text["星河科技"].segment_id],
                "fields": [{
                    "field_type": "role",
                    "refs": [{
                        "segment_id": by_text["产品经理"].segment_id,
                        "exact_quote": "产品经理",
                    }],
                }],
            },
        ],
        "unassigned_segment_ids": [],
    }, segments)

    assert result.valid is False
    assert {issue.code for issue in result.issues} == {
        "duplicate_record_segment",
        "field_outside_record",
    }
    assert result.valid_reference_count == 2


def test_prompt_excludes_jd_segments_and_never_requests_fact_text() -> None:
    segments = build_document_segments(_documents())
    system_prompt, user_prompt = build_shadow_prompt(segments)

    assert "要求3年经验" not in user_prompt
    assert "请优化成产品经理方向" in user_prompt
    assert "exact_quote 必须逐字复制自该 segment" in system_prompt
    assert GroundedExtraction.model_json_schema()["additionalProperties"] is False
