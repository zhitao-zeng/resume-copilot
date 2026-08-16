"""Layout graph adapters used by V3.

The adapter boundary intentionally accepts plain dictionaries so the live
PaddleX/PP-Structure runtime is optional.  A real PP-Structure prediction is
never flattened to a string here: label, page, column/region, bbox, order and
confidence remain on every node.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .contracts import DocumentGraph, LayoutNode, SourceAsset, SourceSpan
from .section_ontology import is_document_title, is_section_heading


@dataclass(frozen=True)
class LayoutDecision:
    engine: str
    reason: str
    conditional: bool = False


def route_asset(asset: SourceAsset, *, quality: float | None = None, multi_column: bool = False) -> LayoutDecision:
    """Choose the least lossy parser.

    Native DOCX/PDF paths win whenever their text/layout is available.  Full
    PP-Structure is reserved for scans/images or pages whose native extraction
    is absent/low quality.  This is the key reason V3 does not route every
    upload through PP-Structure.
    """

    filename = asset.filename.lower()
    media = asset.media_type.lower()
    if asset.native and (filename.endswith(".docx") or "wordprocessing" in media):
        return LayoutDecision("native_docx", "editable paragraph/table/style structure is lossless")
    if asset.native and (filename.endswith(".pdf") or media == "application/pdf") and (quality is None or quality >= 0.85):
        return LayoutDecision("native_pdf", "native PDF character spans and bboxes avoid OCR substitutions")
    if asset.native and not multi_column and quality is not None and quality >= 0.95:
        return LayoutDecision("text", "already reliable text; no layout recovery needed")
    if media.startswith("text/") or filename.endswith((".txt", ".md", ".markdown")):
        return LayoutDecision("text", "plain text has no image layout to recover")
    return LayoutDecision("ppstructure", "scan/image/low-quality or ambiguous layout requires full structure parsing", conditional=True)


def _line_kind(line: str) -> str:
    value = line.strip()
    if not value:
        return "paragraph"
    # Only a known, domain-neutral resume section (or explicit Markdown title)
    # is promoted to a top-level heading.  A record-local label such as
    # ``主要成就：`` must not reset section or record ownership.
    if len(value) <= 80 and (
        is_section_heading(value) or is_document_title(value) or value.startswith("#")
    ):
        return "heading"
    return "paragraph"


def from_native_text(
    asset: SourceAsset,
    text: str | None = None,
    *,
    engine: str = "text",
    page_numbers: Iterable[int] | None = None,
) -> DocumentGraph:
    """Create exact source spans for a native text representation."""

    source_text = asset.text if text is None else text
    nodes: list[LayoutNode] = []
    cursor = 0
    order = 0
    explicit_pages = list(page_numbers) if page_numbers is not None else None
    lines = source_text.splitlines(keepends=True)
    if explicit_pages is not None and len(explicit_pages) != len(lines):
        raise ValueError("page_numbers must contain one page number per native line")
    for line_index, line in enumerate(lines):
        page = explicit_pages[line_index] if explicit_pages is not None else 1
        if page < 1:
            raise ValueError("native page numbers must be positive")
        raw = line.rstrip("\r\n")
        start, end = cursor, cursor + len(raw)
        cursor += len(line)
        if not raw.strip():
            continue
        node_id = f"{asset.source_id}:line:{order}"
        nodes.append(LayoutNode(
            node_id=node_id,
            source_id=asset.source_id,
            kind=_line_kind(raw),
            text=raw,
            page=page,
            order=order,
            confidence=1.0,
            source_spans=[SourceSpan(source_id=asset.source_id, char_start=start, char_end=end, page=page, node_id=node_id)],
        ))
        order += 1
    return DocumentGraph(
        source_id=asset.source_id,
        source_type=asset.source_type,
        extraction_engine=engine,
        source_text=source_text,
        nodes=nodes,
        metadata={"native": asset.native},
    )


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def from_ppstructure_blocks(
    asset: SourceAsset,
    blocks: Iterable[dict[str, Any]],
    *,
    source_text: str | None = None,
) -> DocumentGraph:
    """Adapt ordered PP-StructureV3 parsing blocks without discarding metadata."""

    indexed_rows = [(index, row) for index, row in enumerate(blocks) if isinstance(row, dict)]
    indexed_rows.sort(
        key=lambda item: (
            _safe_int(item[1].get("page", item[1].get("page_id", 1)), 1, minimum=1),
            _safe_int(
                item[1].get("block_order", item[1].get("order", item[0])),
                10**9 + item[0],
            ),
            item[0],
        )
    )
    rows = [row for _, row in indexed_rows]
    normalized_rows: list[tuple[dict[str, Any], str]] = []
    for row in rows:
        text = str(row.get("block_content") or row.get("text") or "").strip()
        if text:
            normalized_rows.append((row, text))
    normalized_document = "\n".join(text for _, text in normalized_rows)
    # A prediction often has no original character offsets.  Use the caller's
    # native source only when every block can be located, in order, exactly;
    # otherwise use a deterministic parser document whose spans are exact.
    document = normalized_document
    locations: list[tuple[int, int]] = []
    if source_text is not None:
        search_cursor = 0
        candidate: list[tuple[int, int]] = []
        for _, text in normalized_rows:
            start = source_text.find(text, search_cursor)
            if start < 0:
                candidate = []
                break
            candidate.append((start, start + len(text)))
            search_cursor = start + len(text)
        if len(candidate) == len(normalized_rows):
            document = source_text
            locations = candidate
    if not locations:
        cursor = 0
        for index, (_, text) in enumerate(normalized_rows):
            start = cursor
            end = start + len(text)
            locations.append((start, end))
            cursor = end + (1 if index < len(normalized_rows) - 1 else 0)
    nodes: list[LayoutNode] = []
    used_ids: dict[str, int] = {}
    for index, (row, text) in enumerate(normalized_rows):
        start, end = locations[index]
        raw_id = f"{asset.source_id}:pp:{row.get('block_id', index)}"
        occurrence = used_ids.get(raw_id, 0)
        used_ids[raw_id] = occurrence + 1
        node_id = raw_id if occurrence == 0 else f"{raw_id}#{occurrence}"
        page = _safe_int(row.get("page", row.get("page_id", 1)), 1, minimum=1)
        order = _safe_int(row.get("block_order", row.get("order", index)), index)
        confidence = row.get("confidence", row.get("score", 1.0))
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 1.0
        metadata = dict(row)
        # Keep the original PP-Structure fields in metadata as well as the
        # normalized fields, which makes later A/B comparison possible.
        nodes.append(LayoutNode(
            node_id=node_id,
            source_id=asset.source_id,
            kind="heading" if str(row.get("block_label", "")).lower() in {"title", "heading"} or _line_kind(text) == "heading" else "region",
            text=text,
            page=page,
            column_id=str(row["column_id"]) if row.get("column_id") is not None else None,
            region_id=str(row["region_id"]) if row.get("region_id") is not None else None,
            parent_id=str(row["parent_id"]) if row.get("parent_id") is not None else None,
            bbox=_bbox(row.get("block_bbox", row.get("bbox"))),
            order=order,
            confidence=confidence,
            label=str(row.get("block_label", row.get("label", "")) or ""),
            source_spans=[SourceSpan(source_id=asset.source_id, char_start=start, char_end=end, page=page, node_id=node_id)],
            metadata=metadata,
        ))
    return DocumentGraph(
        source_id=asset.source_id,
        source_type=asset.source_type,
        extraction_engine="ppstructure",
        source_text=document,
        nodes=nodes,
        metadata={"layout_source": "PP-StructureV3", "preserved_fields": ["page", "column_id", "region_id", "parent_id", "bbox", "order", "confidence", "label", "metadata"]},
    )


def build_document_graph(
    asset: SourceAsset,
    *,
    text: str | None = None,
    ppstructure_blocks: Iterable[dict[str, Any]] | None = None,
    quality: float | None = None,
    multi_column: bool = False,
    force_ppstructure: bool = False,
    shadow_ppstructure: bool = False,
) -> DocumentGraph:
    decision = route_asset(asset, quality=quality, multi_column=multi_column)
    # Native-first is authoritative.  A caller must explicitly request a
    # PP-Structure A/B takeover; merely passing shadow blocks cannot replace a
    # reliable DOCX/native PDF graph.
    use_ppstructure = force_ppstructure or decision.engine == "ppstructure"
    if use_ppstructure:
        if ppstructure_blocks is None:
            # A live runtime is deliberately not imported by this shadow
            # compiler.  The caller must provide adapter output or explicitly
            # use from_native_text after a runtime failure.
            raise ValueError("PP-Structure route requires ppstructure_blocks in the shadow compiler")
        graph = from_ppstructure_blocks(asset, ppstructure_blocks, source_text=text)
        if force_ppstructure:
            graph.metadata["forced_engine"] = True
        return graph
    graph = from_native_text(asset, text, engine=decision.engine)
    if shadow_ppstructure and ppstructure_blocks is not None:
        graph.metadata["shadow_ppstructure_available"] = True
        graph.metadata["shadow_ppstructure_block_count"] = sum(1 for item in ppstructure_blocks if isinstance(item, dict))
    return graph


__all__ = ["LayoutDecision", "build_document_graph", "from_native_text", "from_ppstructure_blocks", "route_asset"]
