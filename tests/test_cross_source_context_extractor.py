from __future__ import annotations

from cross_source_context_extractor import build_cross_source_windows
from segment_grounded_extractor import build_document_segments
from v2_schemas import SourceDocument


def _documents(*, resume_lines=4, query_lines=2):
    documents = []
    if resume_lines:
        documents.append(SourceDocument(
            source_id="resume",
            source_type="resume",
            text="|".join(f"经历{i}" for i in range(resume_lines)),
        ))
    if query_lines:
        documents.append(SourceDocument(
            source_id="query",
            source_type="query",
            text="|".join(f"需求{i}" for i in range(query_lines)),
        ))
    documents.append(SourceDocument(
        source_id="jd",
        source_type="jd",
        text="岗位要求",
    ))
    return documents


def test_bounded_query_is_attached_to_every_resume_window() -> None:
    segments = build_document_segments(_documents(resume_lines=6, query_lines=2))
    windows = build_cross_source_windows(
        segments,
        max_segments=4,
        max_characters=100,
        overlap_segments=1,
    )

    assert len(windows) == 2
    assert {window.mode for window in windows} == {"resume_with_query"}
    query_ids = {
        segment.segment_id for segment in segments if segment.source_type == "query"
    }
    assert all(query_ids.issubset(window.segment_ids) for window in windows)
    assert all(
        all(segment.source_type != "jd" for segment in window.segments)
        for window in windows
    )


def test_query_only_input_uses_normal_overlapping_windows() -> None:
    segments = build_document_segments(_documents(resume_lines=0, query_lines=6))
    windows = build_cross_source_windows(
        segments,
        max_segments=4,
        max_characters=100,
        overlap_segments=1,
    )

    assert len(windows) == 2
    assert {window.mode for window in windows} == {"query_only"}
    assert windows[0].segment_ids[-1] == windows[1].segment_ids[0]


def test_dual_long_source_falls_back_without_truncating_query() -> None:
    segments = build_document_segments(_documents(resume_lines=6, query_lines=5))
    windows = build_cross_source_windows(
        segments,
        max_segments=4,
        max_characters=100,
        overlap_segments=1,
    )
    query_ids = {
        segment.segment_id for segment in segments if segment.source_type == "query"
    }
    returned_query_ids = {
        segment.segment_id
        for window in windows
        for segment in window.segments
        if segment.source_type == "query"
    }

    assert {window.mode for window in windows} == {"independent_fallback"}
    assert returned_query_ids == query_ids


def test_small_resume_and_query_match_full_non_jd_a3_context_exactly() -> None:
    segments = build_document_segments(_documents(resume_lines=4, query_lines=2))
    windows = build_cross_source_windows(segments)
    expected = tuple(segment for segment in segments if segment.source_type != "jd")

    assert len(windows) == 1
    assert windows[0].segments == expected
