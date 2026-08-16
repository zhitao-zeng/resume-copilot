"""Frozen, source-grounded hooks for optional model-candidate evaluation.

The production defaults never read these caches.  They exist so an external
model can be run once on a frozen evaluation set and its exact decisions can
then pass through the normal API, grounding, rendering, and independent audit
without installing multi-gigabyte experimental runtimes in the API image.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable


# Internal transport marker used only between OCR extraction and the V3 graph
# adapter.  It is removed before prompts, replies, artifacts or rendered output
# are built.
OCR_LAYOUT_REGION_SEPARATOR = "\ue000V3_LAYOUT_REGION\ue001"


_DIRECTION = re.compile(
    r"(?:请|帮我|麻烦|希望|想要|生成|优化|润色|修改|调整|删除|去掉|"
    r"应聘|求职|岗位|简历|\b(?:resume|cv|job|apply|application)\b)",
    re.IGNORECASE,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


@lru_cache(maxsize=8)
def _load_cache(path_value: str, expected_kind: str) -> dict[str, Any]:
    path = Path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("kind") != expected_kind:
        raise ValueError(f"invalid {expected_kind} candidate cache: {path}")
    items = payload.get("items")
    if not isinstance(items, dict):
        raise ValueError(f"candidate cache has no item map: {path}")
    return payload


def cached_query_spans(query: str) -> list[dict[str, Any]]:
    """Return PP-UIE spans only when the explicit candidate mode is enabled."""

    if os.getenv("QUERY_EXTRACTOR", "heuristic").strip().lower() != "ppuie":
        return []
    path = os.getenv("QUERY_EXTRACTOR_CACHE_FILE", "").strip()
    if not path or not query:
        return []
    payload = _load_cache(path, "ppuie-query-spans")
    digest = _sha256_text(query)
    item = payload["items"].get(digest)
    if not isinstance(item, dict) or item.get("query_sha256") != digest:
        return []

    accepted: list[dict[str, Any]] = []
    for candidate in item.get("spans") or []:
        try:
            start = int(candidate["start"])
            end = int(candidate["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= start < end <= len(query)):
            continue
        text = query[start:end]
        if text != str(candidate.get("text") or ""):
            continue
        compact = re.sub(r"\W+", "", text, flags=re.UNICODE)
        if len(compact) < 2 or _DIRECTION.search(text):
            continue
        accepted.append({
            "start": start,
            "end": end,
            "text": text,
            "groups": list(candidate.get("groups") or []),
        })
    return accepted


def cached_layout_regions(
    content: bytes,
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Return frozen PP-Structure regions for the exact uploaded image."""

    if os.getenv("LAYOUT_ORDER_ENGINE", "bbox").strip().lower() != "ppstructure":
        return []
    path = os.getenv("LAYOUT_ORDER_CACHE_FILE", "").strip()
    if not path or not content:
        return []
    payload = _load_cache(path, "ppstructure-regions")
    digest = _sha256_bytes(content)
    item = payload["items"].get(digest)
    if not isinstance(item, dict) or item.get("content_sha256") != digest:
        return []
    if int(item.get("prepared_width") or 0) != width:
        return []
    if int(item.get("prepared_height") or 0) != height:
        return []

    regions: list[dict[str, Any]] = []
    for candidate in item.get("regions") or []:
        bbox = candidate.get("bbox")
        try:
            values = [float(value) for value in bbox]
            confidence = float(candidate.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if len(values) != 4:
            continue
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            continue
        region = {"bbox": values, "confidence": confidence}
        try:
            region["order"] = int(candidate["order"])
        except (KeyError, TypeError, ValueError):
            pass
        regions.append(region)
    return regions


def region_aware_ocr_order(
    boxes: Any,
    texts: Any,
    *,
    width: int,
    height: int,
    regions: list[dict[str, Any]],
    baseline_order: Callable[..., list[str]],
    group_separator: str = "",
) -> list[str] | None:
    """Apply the frozen A3 region policy while conserving every OCR line."""

    import numpy as np

    if boxes is None or len(boxes) == 0 or len(regions) < 2:
        return None

    blocks: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        points = np.asarray(box, dtype=float)
        if points.ndim != 2 or points.shape[1] < 2:
            continue
        value = texts[index] if texts is not None and index < len(texts) else ""
        text = str(value[0] if isinstance(value, (tuple, list)) else value).strip()
        if not text:
            continue
        blocks.append({
            "source_index": index,
            "text": text,
            "bbox": [
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            ],
        })
    if len(blocks) < 2:
        return None

    def independent_columns(left: dict[str, Any], right: dict[str, Any]) -> bool:
        lx1, ly1, lx2, ly2 = left["bbox"]
        rx1, ry1, rx2, ry2 = right["bbox"]
        horizontal = max(0.0, min(lx2, rx2) - max(lx1, rx1))
        vertical = max(0.0, min(ly2, ry2) - max(ly1, ry1))
        width_overlap = horizontal / max(1.0, min(lx2 - lx1, rx2 - rx1))
        height_overlap = vertical / max(1.0, min(ly2 - ly1, ry2 - ry1))
        return width_overlap <= 0.20 and height_overlap >= 0.35

    trusted = {
        index
        for index, region in enumerate(regions)
        if float(region.get("confidence", 0.0)) >= 0.30
        and (region["bbox"][2] - region["bbox"][0]) <= width * 0.82
        and any(
            other_index != index and independent_columns(region, other)
            for other_index, other in enumerate(regions)
        )
    }
    if len(trusted) < 2:
        return None

    def intersection_ratio(block: dict[str, Any], region: list[float]) -> float:
        x1, y1, x2, y2 = block["bbox"]
        rx1, ry1, rx2, ry2 = region
        area = max(0.0, min(x2, rx2) - max(x1, rx1)) * max(
            0.0, min(y2, ry2) - max(y1, ry1)
        )
        return area / max(1.0, (x2 - x1) * (y2 - y1))

    assignments: dict[int, list[int]] = {}
    unassigned: list[int] = []
    for local_index, block in enumerate(blocks):
        scores = sorted(
            (
                (intersection_ratio(block, regions[region_index]["bbox"]), region_index)
                for region_index in trusted
            ),
            reverse=True,
        )
        best_score, best_region = scores[0] if scores else (0.0, -1)
        second_score = scores[1][0] if len(scores) > 1 else 0.0
        spans_page = (block["bbox"][2] - block["bbox"][0]) >= width * 0.60
        if best_score >= 0.55 and second_score < 0.35 and not spans_page:
            assignments.setdefault(best_region, []).append(local_index)
        else:
            unassigned.append(local_index)

    groups = [
        {
            "bbox": regions[index]["bbox"],
            "members": members,
            "structure_order": regions[index].get("order"),
        }
        for index, members in assignments.items()
        if members
    ]
    if sum(len(group["members"]) >= 2 for group in groups) < 2:
        return None
    groups.extend({
        "bbox": blocks[index]["bbox"],
        "members": [index],
        "structure_order": None,
    } for index in unassigned)

    def polygon(bbox: list[float]) -> list[list[float]]:
        x1, y1, x2, y2 = bbox
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def baseline_indexes(items: list[dict[str, Any]]) -> list[int]:
        tagged = [(f"IDX{index:06d}", 1.0) for index in range(len(items))]
        ordered = baseline_order(
            [polygon(item["bbox"]) for item in items],
            tagged,
            img_width=width,
            img_height=height,
        )
        return [int(value.removeprefix("IDX")) for value in ordered]

    group_order = baseline_indexes(groups)
    # Preserve the deterministic BBOX positions of unassigned OCR lines while
    # replacing only the relative order of confidently mapped structure
    # regions.  This makes PP-Structure a layout oracle, never a text source,
    # and conserves every PP-OCRv6 line even when region coverage is partial.
    structured_positions = [
        position for position, group_index in enumerate(group_order)
        if groups[group_index].get("structure_order") is not None
    ]
    structured_groups = sorted(
        (group_index for group_index in group_order
         if groups[group_index].get("structure_order") is not None),
        key=lambda group_index: (
            int(groups[group_index]["structure_order"]),
            group_index,
        ),
    )
    for position, group_index in zip(structured_positions, structured_groups):
        group_order[position] = group_index

    output: list[int] = []
    ordered_group_members: list[tuple[bool, list[int]]] = []
    for group_index in group_order:
        members = groups[group_index]["members"]
        if len(members) == 1:
            output.extend(members)
            ordered_group_members.append((
                groups[group_index].get("structure_order") is not None,
                list(members),
            ))
            continue
        local = baseline_indexes([blocks[index] for index in members])
        ordered_members = [members[index] for index in local]
        output.extend(ordered_members)
        ordered_group_members.append((
            groups[group_index].get("structure_order") is not None,
            ordered_members,
        ))
    if Counter(output) != Counter(range(len(blocks))):
        return None
    if not group_separator:
        return [blocks[index]["text"] for index in output]
    ordered_text: list[str] = []
    for structured, members in ordered_group_members:
        if structured and ordered_text:
            ordered_text.append(group_separator)
        ordered_text.extend(blocks[index]["text"] for index in members)
    return ordered_text


__all__ = [
    "OCR_LAYOUT_REGION_SEPARATOR",
    "cached_layout_regions",
    "cached_query_spans",
    "region_aware_ocr_order",
]
