"""ResumeComposer: LLM Call 1 — produce DraftResume from SourceBundle.

V2 Layer 2.
"""
from __future__ import annotations

import logging
from v2_schemas import SourceBlock, EvidenceRef

logger = logging.getLogger(__name__)


def evidence_exists(ref: EvidenceRef, blocks: list[SourceBlock]) -> bool:
    """Check that the evidence quote actually exists in the referenced block.
    This is a deterministic check — NOT another LLM call."""
    block = next((b for b in blocks if b.block_id == ref.block_id), None)
    if block is None:
        return False
    return ref.quote in block.text
