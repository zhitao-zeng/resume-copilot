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
    resume_texts = [b.text for b in source.blocks if b.source_type == "resume"]
    if resume_texts:
        evidence_parts.append("-- CANDIDATE RESUME --\n" + "\n".join(resume_texts))

    query_texts = [b.text for b in source.blocks if b.source_type == "query"]
    if query_texts:
        evidence_parts.append("-- USER QUERY --\n" + "\n".join(query_texts))

    if evidence_parts:
        parts.append("## CANDIDATE EVIDENCE (facts)\n" + "\n\n".join(evidence_parts))

    # Section 2: Target Context (JD only)
    jd_texts = [b.text for b in source.blocks if b.source_type == "jd"]
    if jd_texts:
        parts.append("## TARGET CONTEXT (reference only — NOT candidate facts)\n"
                     + "\n".join(jd_texts))

    return "\n\n".join(parts)


def compose_resume(source: SourceBundle) -> DraftResume:
    """Call LLM to extract structured resume from source material."""
    if not llm_enabled():
        return DraftResume()

    source_text = _build_source_text(source)

    prompt = (
        "请从以下材料中提取简历信息。注意材料分为两部分：\n"
        "1) CANDIDATE EVIDENCE：候选人的简历原文和用户补充，这是事实来源。\n"
        "2) TARGET CONTEXT：目标岗位描述，只做参考。\n\n"
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
        logger.warning("ResumeComposer LLM call failed: %s", exc)
        return DraftResume()

    if not isinstance(parsed, dict):
        return DraftResume()

    try:
        return DraftResume(**parsed)
    except Exception as exc:
        logger.warning("ResumeComposer output validation failed: %s", exc)
        return DraftResume()


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
