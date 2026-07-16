"""ResumeVerifier: LLM Call 2 — verify DraftResume, output CanonicalResume.

V2 Layer 3.  Directly returns corrected resume — no intermediate report.
"""
from __future__ import annotations

import json
import logging

from prompts import RESUME_VERIFIER_SYSTEM_PROMPT
from server_runtime import call_llm_typed, llm_enabled
from v2_schemas import (
    SourceBundle, DraftResume, VerifiedResult, CanonicalResume,
    Change, Meta, Education, Experience, Project,
)

logger = logging.getLogger(__name__)


def conservative_fallback() -> VerifiedResult:
    """When Verifier fails, return empty safe result — never return unverified Draft."""
    return VerifiedResult(
        resume=CanonicalResume(
            meta=Meta(),
            education=[],
            experiences=[],
            projects=[],
            summary="",
        ),
        changes=[Change(path="*", action="remove",
                        reason="Verifier failed, emitted empty fallback")],
    )


def verify_resume(source: SourceBundle, draft: DraftResume) -> VerifiedResult:
    """Call LLM to verify and produce CanonicalResume."""
    if not llm_enabled():
        return conservative_fallback()

    source_parts = [f"[{b.block_id}] {b.text}" for b in source.blocks]
    draft_json = draft.model_dump_json(exclude_none=True)

    prompt = (
        "请审核以下 DraftResume，输出修正后的最终简历。\n\n"
        "【原始材料】\n"
        f"{chr(10).join(source_parts)}\n\n"
        "【DraftResume】\n"
        f"{draft_json}"
    )

    try:
        parsed = call_llm_typed(
            CanonicalResume,
            RESUME_VERIFIER_SYSTEM_PROMPT,
            prompt,
            temperature=0.0,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.warning("ResumeVerifier LLM call failed: %s", exc)
        return conservative_fallback()

    if not isinstance(parsed, dict) or not parsed:
        return conservative_fallback()

    try:
        resume = CanonicalResume(**parsed)
        result = VerifiedResult(resume=resume)
        return result
    except Exception as exc:
        logger.warning("ResumeVerifier output validation failed: %s", exc)
        return conservative_fallback()
