"""Full-context, active-output primitives for the offline A4 shadow parser."""

from __future__ import annotations

import json
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from segment_grounded_extractor import (
    DecodedField,
    DecodedRecord,
    DocumentSegment,
    GroundedExtraction,
    GroundingIssue,
    GroundingValidationResult,
    build_shadow_prompt,
    validate_grounded_extraction,
)


class ActiveSegmentWindow(BaseModel):
    """A deterministic, non-overlapping ownership group over full context."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    window_id: str
    active_segment_ids: tuple[str, ...] = Field(min_length=1, max_length=20)


class ActiveScopeIssue(BaseModel):
    """A model output whose deterministic owner is not the current group."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    code: Literal["inactive_field", "inactive_unassigned_segment"]
    path: str
    message: str


class ActiveScopeValidationResult(BaseModel):
    """A3 validation filtered by the active first-span ownership rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    validation: GroundingValidationResult
    active_issues: tuple[ActiveScopeIssue, ...] = ()


def build_active_segment_windows(
    segments: Sequence[DocumentSegment],
    *,
    max_segments: int = 20,
) -> list[ActiveSegmentWindow]:
    """Assign every non-JD segment exactly once without semantic routing."""

    if max_segments < 1 or max_segments > 20:
        raise ValueError("max_segments must be in [1, 20]")
    candidate_ids = [
        segment.segment_id for segment in segments if segment.source_type != "jd"
    ]
    return [
        ActiveSegmentWindow(
            window_id=f"active:W{index // max_segments + 1:04d}",
            active_segment_ids=tuple(candidate_ids[index:index + max_segments]),
        )
        for index in range(0, len(candidate_ids), max_segments)
    ]


def build_active_window_prompt(
    segments: Sequence[DocumentSegment],
    active_segment_ids: Sequence[str],
) -> tuple[str, str]:
    """Append only the frozen A4 active-ownership constraint to A3."""

    segment_index = {segment.segment_id: segment for segment in segments}
    active_ids = tuple(active_segment_ids)
    if len(active_ids) != len(set(active_ids)):
        raise ValueError("active segment IDs must be unique")
    if not active_ids:
        raise ValueError("at least one active segment is required")
    for segment_id in active_ids:
        segment = segment_index.get(segment_id)
        if segment is None or segment.source_type == "jd":
            raise ValueError(f"invalid active candidate segment {segment_id!r}")

    system_prompt, user_prompt = build_shadow_prompt(segments)
    system_prompt += """

当前请求采用 active segment 输出所有权规则：
15. 输入中的全部 segments 都可用于理解语义、章节和记录归属；不得因为某片段不是 active 就忽略其上下文。
16. 只输出“最左侧/最先出现的引用 segment_id”位于 active_segment_ids 的字段。若一个字段引用多个片段，以原文顺序最早的引用决定它属于哪次请求。
17. records[i].segment_ids 应保留该记录在完整上下文中的全部明确事实片段，包括非 active 片段；但一条 record 只有在本次至少输出一个 active 所有的字段时才可返回。
18. profile_fields 同样遵守 active 所有权。unassigned_segment_ids 只能列 active_segment_ids 中的片段。
19. 非 active 片段只是本次上下文，不得重复输出其字段。不要改变前述字段类型、事实边界和归属规则。
"""
    payload = json.loads(user_prompt)
    payload["active_segment_ids"] = list(active_ids)
    return system_prompt, json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"),
    )


def _first_span_segment_id(
    field: DecodedField,
    segment_order: dict[str, int],
) -> str:
    return min(
        field.spans,
        key=lambda span: (
            segment_order.get(span.segment_id, len(segment_order)),
            span.absolute_start,
            span.absolute_end,
        ),
    ).segment_id


def validate_active_extraction(
    extraction: GroundedExtraction | dict,
    segments: Sequence[DocumentSegment],
    active_segment_ids: Sequence[str],
) -> ActiveScopeValidationResult:
    """Decode with A3, then reject every field owned by another active group."""

    raw = (
        extraction
        if isinstance(extraction, GroundedExtraction)
        else GroundedExtraction.model_validate(extraction)
    )
    segment_index = {segment.segment_id: segment for segment in segments}
    segment_order = {
        segment.segment_id: index for index, segment in enumerate(segments)
    }
    active = set(active_segment_ids)
    if len(active) != len(tuple(active_segment_ids)):
        raise ValueError("active segment IDs must be unique")
    for segment_id in active:
        segment = segment_index.get(segment_id)
        if segment is None or segment.source_type == "jd":
            raise ValueError(f"invalid active candidate segment {segment_id!r}")

    base = validate_grounded_extraction(raw, segments)
    active_issues: list[ActiveScopeIssue] = []
    grounding_issues = list(base.issues)

    profile_fields: list[DecodedField] = []
    for index, field in enumerate(base.profile_fields):
        owner = _first_span_segment_id(field, segment_order)
        if owner in active:
            profile_fields.append(field)
            continue
        issue = ActiveScopeIssue(
            code="inactive_field",
            path=f"profile_fields[{index}]",
            message=f"field is owned by inactive first segment {owner!r}",
        )
        active_issues.append(issue)
        grounding_issues.append(GroundingIssue(
            code="field_outside_record",
            path=issue.path,
            message=issue.message,
        ))

    records: list[DecodedRecord] = []
    for record_index, record in enumerate(base.records):
        fields: list[DecodedField] = []
        for field_index, field in enumerate(record.fields):
            owner = _first_span_segment_id(field, segment_order)
            if owner in active:
                fields.append(field)
                continue
            issue = ActiveScopeIssue(
                code="inactive_field",
                path=f"records[{record_index}].fields[{field_index}]",
                message=f"field is owned by inactive first segment {owner!r}",
            )
            active_issues.append(issue)
            grounding_issues.append(GroundingIssue(
                code="field_outside_record",
                path=issue.path,
                message=issue.message,
            ))
        if fields:
            records.append(DecodedRecord(
                record_type=record.record_type,
                segment_ids=record.segment_ids,
                fields=tuple(fields),
            ))

    for index, segment_id in enumerate(raw.unassigned_segment_ids):
        if segment_id in active:
            continue
        issue = ActiveScopeIssue(
            code="inactive_unassigned_segment",
            path=f"unassigned_segment_ids[{index}]",
            message=f"unassigned segment {segment_id!r} is not active",
        )
        active_issues.append(issue)
        grounding_issues.append(GroundingIssue(
            code="invalid_unassigned_segment",
            path=issue.path,
            message=issue.message,
        ))

    validation = GroundingValidationResult(
        valid=not grounding_issues,
        profile_fields=tuple(profile_fields),
        records=tuple(records),
        issues=tuple(grounding_issues),
        returned_reference_count=base.returned_reference_count,
        valid_reference_count=base.valid_reference_count,
    )
    return ActiveScopeValidationResult(
        validation=validation,
        active_issues=tuple(active_issues),
    )


__all__ = [
    "ActiveScopeIssue",
    "ActiveScopeValidationResult",
    "ActiveSegmentWindow",
    "build_active_segment_windows",
    "build_active_window_prompt",
    "validate_active_extraction",
]
