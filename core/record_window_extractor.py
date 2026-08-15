"""Deterministic windowing and aggregation for the offline A3 shadow parser.

This module deliberately has no production import edge.  It changes only the
call topology around :mod:`segment_grounded_extractor`: the prompt, schema,
field taxonomy, exact-quote decoder, and source segmentation remain frozen.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from segment_grounded_extractor import (
    DecodedField,
    DecodedRecord,
    DocumentSegment,
    GroundingIssue,
    GroundingValidationResult,
)


class SegmentWindow(BaseModel):
    """A bounded, contiguous run of segments from one source document."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    window_id: str
    source_id: str
    source_type: Literal["resume", "query"]
    segments: tuple[DocumentSegment, ...] = Field(min_length=1)
    character_count: int = Field(ge=1)

    @property
    def segment_ids(self) -> tuple[str, ...]:
        return tuple(segment.segment_id for segment in self.segments)


class WindowValidation(BaseModel):
    """One independently decoded A3 result associated with its source window."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    window_id: str
    validation: GroundingValidationResult


class WindowAggregationIssue(BaseModel):
    """A cross-window ambiguity that is rejected instead of guessed."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    code: Literal[
        "record_type_conflict",
        "record_partition_conflict",
        "field_assignment_conflict",
    ]
    path: str
    message: str


class WindowAggregationResult(BaseModel):
    """Strict aggregate plus explicit cross-window rejection evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    validation: GroundingValidationResult
    aggregation_issues: tuple[WindowAggregationIssue, ...] = ()
    window_count: int = Field(ge=0)


def build_segment_windows(
    segments: Sequence[DocumentSegment],
    *,
    max_segments: int = 20,
    max_characters: int = 1_600,
    overlap_segments: int = 5,
) -> list[SegmentWindow]:
    """Partition candidate sources into fixed, overlapping source-order windows.

    Sizing is deliberately independent of semantic labels, industry terms, and
    model output.  JD segments are excluded exactly as they are from the frozen
    A3 prompt.  A single over-long segment is retained whole so source offsets
    are never changed.
    """

    if max_segments < 1:
        raise ValueError("max_segments must be positive")
    if max_characters < 1:
        raise ValueError("max_characters must be positive")
    if not 0 <= overlap_segments < max_segments:
        raise ValueError("overlap_segments must be in [0, max_segments)")

    by_source: dict[str, list[DocumentSegment]] = {}
    source_order: list[str] = []
    for segment in segments:
        if segment.source_type == "jd":
            continue
        if segment.source_id not in by_source:
            by_source[segment.source_id] = []
            source_order.append(segment.source_id)
        by_source[segment.source_id].append(segment)

    windows: list[SegmentWindow] = []
    for source_id in source_order:
        source_segments = by_source[source_id]
        source_type = source_segments[0].source_type
        if source_type not in {"resume", "query"}:
            continue
        start = 0
        window_number = 0
        while start < len(source_segments):
            end = start
            character_count = 0
            while end < len(source_segments) and end - start < max_segments:
                next_count = len(source_segments[end].text)
                if end > start and character_count + next_count > max_characters:
                    break
                character_count += next_count
                end += 1
            if end == start:
                character_count = len(source_segments[end].text)
                end += 1

            window_number += 1
            selected = tuple(source_segments[start:end])
            windows.append(SegmentWindow(
                window_id=f"{source_id}:W{window_number:04d}",
                source_id=source_id,
                source_type=source_type,
                segments=selected,
                character_count=character_count,
            ))
            if end == len(source_segments):
                break
            start = max(start + 1, end - overlap_segments)

    return windows


def _span_key(field: DecodedField) -> tuple[tuple[str, int, int], ...]:
    return tuple(sorted(
        (
            span.source_id,
            span.absolute_start,
            span.absolute_end,
        )
        for span in field.spans
    ))


def _field_signature(field: DecodedField) -> tuple[object, ...]:
    return field.field_type, _span_key(field)


def _fields_overlap(left: DecodedField, right: DecodedField) -> bool:
    return any(
        left_span.source_id == right_span.source_id
        and left_span.absolute_start < right_span.absolute_end
        and right_span.absolute_start < left_span.absolute_end
        for left_span in left.spans
        for right_span in right.spans
    )


def aggregate_window_validations(
    windows: Sequence[SegmentWindow],
    results: Sequence[WindowValidation],
    all_segments: Sequence[DocumentSegment],
) -> WindowAggregationResult:
    """Merge only unambiguous exact-span outputs from overlapping windows.

    Record nodes are joined only through a shared declared source segment.  If
    that graph disagrees about record type, collapses two records that one
    window kept separate, or assigns overlapping source text to incompatible
    fields, the ambiguous component/fields are dropped and the aggregate is
    marked invalid.  This preserves precision and makes abstention observable.
    """

    window_by_id = {window.window_id: window for window in windows}
    if len(window_by_id) != len(windows):
        raise ValueError("window IDs must be unique")
    result_by_id = {result.window_id: result for result in results}
    if len(result_by_id) != len(results):
        raise ValueError("window result IDs must be unique")
    if set(result_by_id) != set(window_by_id):
        missing = sorted(set(window_by_id) - set(result_by_id))
        extra = sorted(set(result_by_id) - set(window_by_id))
        raise ValueError(f"window/result mismatch: missing={missing}, extra={extra}")

    segment_order = {
        segment.segment_id: index for index, segment in enumerate(all_segments)
    }
    grounding_issues: list[GroundingIssue] = []
    aggregation_issues: list[WindowAggregationIssue] = []
    returned_reference_count = 0
    valid_reference_count = 0

    profile_entries: list[tuple[str, DecodedField]] = []
    record_nodes: list[tuple[str, int, DecodedRecord]] = []
    for window in windows:
        validation = result_by_id[window.window_id].validation
        returned_reference_count += validation.returned_reference_count
        valid_reference_count += validation.valid_reference_count
        grounding_issues.extend(
            issue.model_copy(update={"path": f"{window.window_id}.{issue.path}"})
            for issue in validation.issues
        )
        profile_entries.extend(
            (window.window_id, field) for field in validation.profile_fields
        )
        record_nodes.extend(
            (window.window_id, index, record)
            for index, record in enumerate(validation.records)
        )

    # Build connected components from declared record-segment overlap.
    parent = list(range(len(record_nodes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    segment_sets = [set(node[2].segment_ids) for node in record_nodes]
    for left in range(len(record_nodes)):
        for right in range(left + 1, len(record_nodes)):
            if segment_sets[left].intersection(segment_sets[right]):
                union(left, right)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(record_nodes)):
        components[find(index)].append(index)

    rejected_components: set[int] = set()
    for root, members in components.items():
        record_types = {record_nodes[index][2].record_type for index in members}
        if len(record_types) > 1:
            rejected_components.add(root)
            issue = WindowAggregationIssue(
                code="record_type_conflict",
                path=f"components[{root}]",
                message=f"overlapping record partition has types {sorted(record_types)}",
            )
            aggregation_issues.append(issue)
            grounding_issues.append(GroundingIssue(
                code="duplicate_record_segment",
                path=issue.path,
                message=issue.message,
            ))
            continue
        windows_in_component = [record_nodes[index][0] for index in members]
        if len(windows_in_component) != len(set(windows_in_component)):
            rejected_components.add(root)
            issue = WindowAggregationIssue(
                code="record_partition_conflict",
                path=f"components[{root}]",
                message="cross-window merge would collapse records separated in one window",
            )
            aggregation_issues.append(issue)
            grounding_issues.append(GroundingIssue(
                code="duplicate_record_segment",
                path=issue.path,
                message=issue.message,
            ))

    component_by_node = {index: find(index) for index in range(len(record_nodes))}

    # Gather fields under their aggregate owner. Components already rejected
    # above do not contribute candidate facts.
    field_entries: list[tuple[str, int | None, DecodedField]] = [
        ("profile", None, field) for _, field in profile_entries
    ]
    for node_index, (_, _, record) in enumerate(record_nodes):
        root = component_by_node[node_index]
        if root in rejected_components:
            continue
        field_entries.extend(("record", root, field) for field in record.fields)

    # Exact duplicates from overlap are benign. Any other overlapping
    # assignment is ambiguous and all implicated fields are dropped.
    rejected_fields: set[int] = set()
    duplicate_fields: set[int] = set()
    for left in range(len(field_entries)):
        if left in duplicate_fields:
            continue
        left_scope, left_owner, left_field = field_entries[left]
        for right in range(left + 1, len(field_entries)):
            if right in duplicate_fields:
                continue
            right_scope, right_owner, right_field = field_entries[right]
            same_owner = (left_scope, left_owner) == (right_scope, right_owner)
            if same_owner and _field_signature(left_field) == _field_signature(right_field):
                duplicate_fields.add(right)
                continue
            if not _fields_overlap(left_field, right_field):
                continue
            rejected_fields.update({left, right})
            issue = WindowAggregationIssue(
                code="field_assignment_conflict",
                path=f"fields[{left},{right}]",
                message=(
                    "overlapping source text has incompatible field type or ownership: "
                    f"{left_field.field_type}/{left_scope}:{left_owner} vs "
                    f"{right_field.field_type}/{right_scope}:{right_owner}"
                ),
            )
            aggregation_issues.append(issue)
            grounding_issues.append(GroundingIssue(
                code=(
                    "profile_segment_in_record"
                    if {left_scope, right_scope} == {"profile", "record"}
                    else "duplicate_reference"
                ),
                path=issue.path,
                message=issue.message,
            ))

    accepted_profile: list[DecodedField] = []
    accepted_by_component: dict[int, list[DecodedField]] = defaultdict(list)
    for index, (scope, owner, field) in enumerate(field_entries):
        if index in rejected_fields or index in duplicate_fields:
            continue
        if scope == "profile":
            accepted_profile.append(field)
        else:
            assert owner is not None
            accepted_by_component[owner].append(field)

    def field_order(field: DecodedField) -> tuple[int, int, str]:
        first = min(
            (
                segment_order.get(span.segment_id, len(segment_order)),
                span.absolute_start,
            )
            for span in field.spans
        )
        return first[0], first[1], field.field_type

    accepted_profile.sort(key=field_order)
    decoded_records: list[DecodedRecord] = []
    for root, members in components.items():
        if root in rejected_components:
            continue
        fields = accepted_by_component.get(root, [])
        if not fields:
            continue
        fields.sort(key=field_order)
        segment_ids = sorted(
            {
                segment_id
                for node_index in members
                for segment_id in record_nodes[node_index][2].segment_ids
            },
            key=lambda item: segment_order.get(item, len(segment_order)),
        )
        decoded_records.append(DecodedRecord(
            record_type=record_nodes[members[0]][2].record_type,
            segment_ids=tuple(segment_ids),
            fields=tuple(fields),
        ))
    decoded_records.sort(
        key=lambda record: min(
            (segment_order.get(item, len(segment_order)) for item in record.segment_ids),
            default=len(segment_order),
        )
    )

    validation = GroundingValidationResult(
        valid=not grounding_issues,
        profile_fields=tuple(accepted_profile),
        records=tuple(decoded_records),
        issues=tuple(grounding_issues),
        returned_reference_count=returned_reference_count,
        valid_reference_count=valid_reference_count,
    )
    return WindowAggregationResult(
        validation=validation,
        aggregation_issues=tuple(aggregation_issues),
        window_count=len(windows),
    )


__all__ = [
    "SegmentWindow",
    "WindowAggregationIssue",
    "WindowAggregationResult",
    "WindowValidation",
    "aggregate_window_validations",
    "build_segment_windows",
]
