"""ResumeComposer: LLM Call 1 — extract structured JSON from source material.

V2 Layer 2.  Outputs plain JSON (no evidence wrapping).
Verifier judges factuality.
"""
from __future__ import annotations

import logging
from typing import Any

from prompts import RESUME_COMPOSER_SYSTEM_PROMPT
from server_runtime import call_llm_typed, llm_enabled
from v2_schemas import SourceBundle, DraftResume

logger = logging.getLogger(__name__)


def compose_resume(source: SourceBundle) -> DraftResume:
    """Call LLM to extract structured resume from source material."""
    if not llm_enabled():
        return DraftResume()

    parts = []
    for block in source.blocks:
        parts.append(f"[{block.block_id}] {block.text}")
    source_text = "\n".join(parts)

    prompt = (
        "请从以下原始材料中提取简历信息，输出结构化 JSON。\n\n"
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

    if not isinstance(parsed, dict):
        return DraftResume()

    try:
        return DraftResume(**parsed)
    except Exception as exc:
        logger.warning("ResumeComposer output validation failed: %s", exc)
        return DraftResume()
