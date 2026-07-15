"""ResumeComposer: LLM Call 1 — produce DraftResume from SourceBundle.

V2 Layer 2.
"""
from __future__ import annotations

import logging

from prompts import RESUME_COMPOSER_SYSTEM_PROMPT
from server_runtime import call_llm_typed, llm_enabled
from v2_schemas import (
    SourceBlock, SourceBundle, EvidenceRef, DraftResume, DraftField,
)

logger = logging.getLogger(__name__)


def evidence_exists(ref: EvidenceRef, blocks: list[SourceBlock]) -> bool:
    """Check that the evidence quote actually exists in the referenced block.
    This is a deterministic check — NOT another LLM call."""
    block = next((b for b in blocks if b.block_id == ref.block_id), None)
    if block is None:
        return False
    return ref.quote in block.text


def _strip_invalid_evidence(draft: DraftResume, blocks: list[SourceBlock]) -> None:
    """Remove evidence refs that don't resolve to real text."""
    def _clean(field: DraftField) -> None:
        field.evidence = [e for e in field.evidence if evidence_exists(e, blocks)]

    for edu in draft.education:
        for f in (edu.school, edu.degree, edu.major, edu.period):
            _clean(f)
    for exp in draft.experience:
        for f in (exp.organization, exp.role, exp.period):
            _clean(f)
        for b in exp.bullets:
            _clean(b)
    for proj in draft.projects:
        for f in (proj.name, proj.organization, proj.role, proj.period):
            _clean(f)
    _clean(draft.summary)
    for f in (draft.meta.name, draft.meta.phone, draft.meta.email, draft.meta.target_role):
        _clean(f)


def compose_resume(source: SourceBundle) -> DraftResume:
    """Call LLM to produce DraftResume from source material."""
    if not llm_enabled():
        return DraftResume()

    # Build source text for LLM input
    parts = []
    for block in source.blocks:
        parts.append(f"[{block.block_id}] {block.text}")
    source_text = "\n".join(parts)

    prompt = (
        "请将以下简历文本解析为结构化 JSON。每个字段必须附带证据引用。\n\n"
        "【原始材料】\n"
        f"{source_text}"
    )

    try:
        parsed = call_llm_typed(
            DraftResume,
            RESUME_COMPOSER_SYSTEM_PROMPT,
            prompt,
            temperature=0.0,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.warning("ResumeComposer LLM call failed: %s", exc)
        return DraftResume()

    if not isinstance(parsed, dict) or not parsed:
        return DraftResume()

    draft = DraftResume(**parsed)

    # Strip invalid evidence
    _strip_invalid_evidence(draft, source.blocks)

    return draft
