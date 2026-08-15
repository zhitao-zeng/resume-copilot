"""Domain-neutral cross-source context windows for the offline C1 experiment."""

from __future__ import annotations

from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from record_window_extractor import build_segment_windows
from segment_grounded_extractor import DocumentSegment


class CrossSourceWindow(BaseModel):
    """One unchanged-A3 prompt payload with an explicit topology audit mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    window_id: str
    mode: Literal[
        "resume_with_query",
        "resume_only",
        "query_only",
        "independent_fallback",
    ]
    segments: tuple[DocumentSegment, ...] = Field(min_length=1)
    character_count: int = Field(ge=1)

    @property
    def segment_ids(self) -> tuple[str, ...]:
        return tuple(segment.segment_id for segment in self.segments)


def _bounded_context(
    segments: Sequence[DocumentSegment],
    *,
    max_segments: int,
    max_characters: int,
) -> bool:
    return (
        len(segments) <= max_segments
        and sum(len(segment.text) for segment in segments) <= max_characters
    )


def build_cross_source_windows(
    segments: Sequence[DocumentSegment],
    *,
    max_segments: int = 20,
    max_characters: int = 1_600,
    overlap_segments: int = 5,
) -> list[CrossSourceWindow]:
    """Attach complete bounded Query context to each local Resume window.

    The function consults only source type, source order, segment count, and
    character count.  It never reads semantic labels or text patterns.  When
    both sources exist but Query exceeds the frozen context bound, it falls
    back explicitly to independent windows instead of silently truncating it.
    """

    resume = [segment for segment in segments if segment.source_type == "resume"]
    query = [segment for segment in segments if segment.source_type == "query"]
    if not resume and not query:
        return []

    def local_windows(source_segments: Sequence[DocumentSegment]):
        return build_segment_windows(
            source_segments,
            max_segments=max_segments,
            max_characters=max_characters,
            overlap_segments=overlap_segments,
        )

    windows: list[CrossSourceWindow] = []
    if resume and query and _bounded_context(
        query,
        max_segments=max_segments,
        max_characters=max_characters,
    ):
        for index, resume_window in enumerate(local_windows(resume), start=1):
            combined = tuple(resume_window.segments) + tuple(query)
            windows.append(CrossSourceWindow(
                window_id=f"cross:W{index:04d}",
                mode="resume_with_query",
                segments=combined,
                character_count=sum(len(segment.text) for segment in combined),
            ))
        return windows

    if resume and query:
        independent = list(local_windows(resume)) + list(local_windows(query))
        return [
            CrossSourceWindow(
                window_id=f"fallback:W{index:04d}",
                mode="independent_fallback",
                segments=tuple(window.segments),
                character_count=window.character_count,
            )
            for index, window in enumerate(independent, start=1)
        ]

    source_segments = resume or query
    mode = "resume_only" if resume else "query_only"
    return [
        CrossSourceWindow(
            window_id=f"{mode}:W{index:04d}",
            mode=mode,
            segments=tuple(window.segments),
            character_count=window.character_count,
        )
        for index, window in enumerate(local_windows(source_segments), start=1)
    ]


__all__ = ["CrossSourceWindow", "build_cross_source_windows"]
