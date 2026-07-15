"""SourceAdapter: build SourceBundle from raw extracted text.

V2 Layer 1 — fully deterministic, no LLM involvement.
"""
from __future__ import annotations

from v2_schemas import SourceBlock, SourceBundle


def _split_into_blocks(text: str, source_type: str) -> list[SourceBlock]:
    """Split text into SourceBlocks by newline."""
    blocks: list[SourceBlock] = []
    for i, line in enumerate(text.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        blocks.append(SourceBlock(
            block_id=f"{source_type}_{i}",
            source_type=source_type,  # type: ignore
            text=line,
        ))
    return blocks


def build_source_bundle(
    cv_text: str,
    query_text: str,
    jd_text: str,
) -> SourceBundle:
    """Build a SourceBundle from extracted text inputs.

    Each line becomes one SourceBlock with a unique block_id.
    query_text and jd_text are each single blocks.
    """
    blocks: list[SourceBlock] = []

    # Resume / CV text blocks
    if cv_text.strip():
        blocks.extend(_split_into_blocks(cv_text, "resume"))

    # Query as a single block
    if query_text.strip():
        blocks.append(SourceBlock(
            block_id="query_0",
            source_type="query",
            text=query_text.strip(),
        ))

    # JD as a single block
    if jd_text.strip():
        blocks.append(SourceBlock(
            block_id="jd_0",
            source_type="jd",
            text=jd_text.strip(),
        ))

    return SourceBundle(blocks=blocks)
