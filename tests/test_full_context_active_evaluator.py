from __future__ import annotations

from full_context_active_extractor import build_active_segment_windows
from segment_grounded_extractor import build_document_segments
from tools.evaluate_full_context_active_shadow import (
    MAX_ACTIVE_SEGMENTS,
    MAX_TOKENS,
)
from v2_schemas import SourceDocument


def test_a4_parameters_match_frozen_contract() -> None:
    assert MAX_ACTIVE_SEGMENTS == 20
    assert MAX_TOKENS == 2_048


def test_125_candidate_segments_require_seven_active_calls() -> None:
    document = SourceDocument(
        source_id="resume",
        source_type="resume",
        text="|".join(f"片段{i:03d}" for i in range(125)),
    )
    segments = build_document_segments([document])
    windows = build_active_segment_windows(segments, max_segments=MAX_ACTIVE_SEGMENTS)

    assert len(segments) == 125
    assert [len(window.active_segment_ids) for window in windows] == [
        20,
        20,
        20,
        20,
        20,
        20,
        5,
    ]
