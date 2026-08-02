"""ResumeComposer: LLM Call 1 — extract structured JSON from source material.

V2 Layer 2.  Outputs plain JSON (no evidence wrapping).
Verifier judges factuality.

Source separation: JD (Target Context) is NEVER a candidate fact source.
Only Candidate Evidence (resume + query) can produce experience/education/projects/skills.
"""
from __future__ import annotations

import logging
from typing import Any

from prompts import RESUME_COMPOSER_SYSTEM_PROMPT, GEN_COMPOSER_SYSTEM_PROMPT
from server_runtime import call_llm_typed, llm_enabled
from source_adapter import candidate_blocks
from v2_schemas import SourceBundle, DraftResume, CanonicalResume

logger = logging.getLogger(__name__)


def _build_source_text(source: SourceBundle) -> str:
    """Build source text with strict semantic separation.

    Two sections:
      1. CANDIDATE EVIDENCE — resume text + user query.  This is the
         ONLY source for experience, education, projects, skills.
      2. TARGET CONTEXT — job description.  ONLY for target_role and
         writing style hints.  NOT a source of candidate facts.
    """
    parts = []

    # Section 1: Candidate Evidence (resume + query)
    evidence_parts = []
    resume_texts = [
        f"[{b.block_id}{'|section=' + b.section_hint if b.section_hint else ''}] {b.text}"
        for b in source.blocks if b.source_type == "resume"
    ]
    if resume_texts:
        evidence_parts.append("-- CANDIDATE RESUME --\n" + "\n".join(resume_texts))

    query_texts = [
        f"[{b.block_id}{'|section=' + b.section_hint if b.section_hint else ''}] {b.text}"
        for b in candidate_blocks(source) if b.source_type == "query"
    ]
    if query_texts:
        evidence_parts.append("-- EXPLICIT CANDIDATE FACT ADDITIONS --\n" + "\n".join(query_texts))

    if evidence_parts:
        parts.append("## CANDIDATE EVIDENCE (facts)\n" + "\n\n".join(evidence_parts))

    direction_texts = [
        f"[{b.block_id}] {b.text}"
        for b in source.blocks if b.source_type == "query" and not b.fact_eligible
    ]
    if direction_texts:
        parts.append("## USER DIRECTIONS (instructions only — NOT candidate facts)\n"
                     + "\n".join(direction_texts))

    # Section 2: Target Context (JD only)
    jd_texts = [
        f"[{b.block_id}{'|section=' + b.section_hint if b.section_hint else ''}] {b.text}"
        for b in source.blocks if b.source_type == "jd"
    ]
    if jd_texts:
        parts.append("## TARGET CONTEXT (reference only — NOT candidate facts)\n"
                     + "\n".join(jd_texts))

    return "\n\n".join(parts)


def _split_source_bundle(source: SourceBundle, max_fact_chars: int = 6000) -> list[SourceBundle]:
    """Chunk long candidate evidence without splitting individual source lines."""

    factual = [block for block in source.blocks if block in candidate_blocks(source)]
    context = [block for block in source.blocks if block not in factual]
    if sum(len(block.text) for block in factual) <= max_fact_chars:
        return [source]
    chunks: list[list] = []
    current: list = []
    current_chars = 0
    for block in factual:
        size = len(block.text) + 1
        if current and current_chars + size > max_fact_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(block)
        current_chars += size
    if current:
        chunks.append(current)
    # Target/instruction context is capped and repeated so every extraction
    # chunk applies the same source-isolation rules and target direction.
    shared_context = []
    context_chars = 0
    for block in context:
        if context_chars + len(block.text) > 1800:
            break
        shared_context.append(block)
        context_chars += len(block.text) + 1
    return [SourceBundle(blocks=list(chunk) + shared_context) for chunk in chunks]


def _merge_drafts(drafts: list[DraftResume]) -> DraftResume:
    """Merge independently extracted chunks without inventing cross-chunk facts."""

    if not drafts:
        return DraftResume()
    merged = drafts[0].model_copy(deep=True)
    for draft in drafts[1:]:
        for field in ("name", "phone", "email", "target_role", "work_experience"):
            if not getattr(merged.meta, field) and getattr(draft.meta, field):
                setattr(merged.meta, field, getattr(draft.meta, field))
        for section in ("education", "experience", "research", "activities", "projects"):
            getattr(merged, section).extend(getattr(draft, section))
        merged.skills.items.extend(draft.skills.items)
        for section in ("awards", "publications", "patents", "certifications", "training", "teaching"):
            getattr(merged, section).extend(getattr(draft, section))
        for title, items in draft.additional_sections.items():
            merged.additional_sections.setdefault(title, []).extend(items)
        if not merged.summary and draft.summary:
            merged.summary = draft.summary
    return merged


def compose_resume(source: SourceBundle) -> DraftResume:
    """Call LLM to extract structured resume from source material."""
    if not llm_enabled():
        return DraftResume()

    drafts: list[DraftResume] = []
    chunks = _split_source_bundle(source)
    for chunk_index, chunk in enumerate(chunks):
        source_text = _build_source_text(chunk)
        prompt = (
            "请从以下材料中完整提取简历信息。注意材料分为两部分：\n"
            "1) CANDIDATE EVIDENCE：候选人的简历原文和用户明确补充，这是事实来源。\n"
            "2) USER DIRECTIONS/TARGET CONTEXT：只做编辑和目标参考，不是候选人事实。\n\n"
            "【材料】\n"
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
            logger.warning("ResumeComposer chunk %d/%d failed: %s", chunk_index + 1, len(chunks), exc)
            continue
        if not isinstance(parsed, dict):
            continue
        try:
            drafts.append(DraftResume(**parsed))
        except Exception as exc:
            logger.warning(
                "ResumeComposer chunk %d/%d validation failed: %s",
                chunk_index + 1,
                len(chunks),
                exc,
            )
    if not drafts:
        return DraftResume()
    logger.info("ResumeComposer extracted %d chunk(s)", len(drafts))
    return _merge_drafts(drafts)


def compose_from_query(query_text: str, jd_text: str) -> CanonicalResume:
    """Generate structured resume framework from query + JD (no CV).

    Used when the user has no resume file — constructs a framework from their
    written description and optional job description.
    """
    if not llm_enabled():
        return CanonicalResume()

    prompt = ""
    if query_text.strip():
        prompt += f"【用户描述】\n{query_text.strip()[:2000]}\n\n"
    if jd_text.strip():
        prompt += f"【目标岗位 JD】\n{jd_text.strip()[:1500]}\n\n"
    prompt += "请根据以上信息生成简历结构化框架。"

    try:
        parsed = call_llm_typed(
            CanonicalResume,
            GEN_COMPOSER_SYSTEM_PROMPT,
            prompt,
            temperature=0.2,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.warning("GenerateComposer LLM call failed: %s", exc)
        return CanonicalResume()

    if not isinstance(parsed, dict):
        return CanonicalResume()

    try:
        return CanonicalResume(**parsed)
    except Exception as exc:
        logger.warning("GenerateComposer output validation failed: %s", exc)
        return CanonicalResume()
