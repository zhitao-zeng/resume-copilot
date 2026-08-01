"""SourceAdapter: build SourceBundle from raw extracted text.

V2 Layer 1 — fully deterministic, no LLM involvement.
"""
from __future__ import annotations

import re

from v2_schemas import SourceBlock, SourceBundle


_SECTION_ALIASES = {
    "summary": ("个人总结", "个人简介", "职业概述", "自我评价"),
    "education": ("教育经历", "教育背景", "学历信息"),
    "experience": ("工作经历", "实习经历", "任职经历", "职业经历"),
    "research": ("科研经历", "研究经历", "实验室经历", "论文成果"),
    "projects": ("项目经历", "项目经验", "课程项目", "个人项目", "开源项目"),
    "activities": ("校园经历", "社团经历", "志愿经历", "社会实践", "学生工作"),
    "skills": ("专业技能", "技能清单", "技术栈", "工具", "语言能力", "证书", "资质"),
    "awards": ("荣誉奖项", "荣誉与奖项", "获奖经历", "奖项"),
}


def _section_hint(line: str) -> str:
    normalized = re.sub(r"[\s:：|｜/\\【】\[\]()（）]+", "", line).casefold()
    for section, aliases in _SECTION_ALIASES.items():
        if any(normalized == re.sub(r"\s+", "", alias).casefold() for alias in aliases):
            return section
    # Inline labels such as "专业技能：Python、SQL" should also carry a
    # structural hint while preserving the complete source line.
    prefix = re.split(r"[:：]", line, maxsplit=1)[0]
    normalized_prefix = re.sub(r"\s+", "", prefix).casefold()
    for section, aliases in _SECTION_ALIASES.items():
        if any(normalized_prefix == re.sub(r"\s+", "", alias).casefold() for alias in aliases):
            return section
    return ""


def _split_into_blocks(text: str, source_type: str) -> list[SourceBlock]:
    """Split text into ordered blocks and retain deterministic section hints."""
    blocks: list[SourceBlock] = []
    current_section = ""
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        detected = _section_hint(line)
        if detected:
            current_section = detected
        blocks.append(SourceBlock(
            block_id=f"{source_type}_{len(blocks)}",
            source_type=source_type,  # type: ignore
            text=line,
            section_hint=detected or current_section or None,
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

    # A multiline query often contains the only candidate profile.  Segment it
    # with the same logic so headings survive in no-CV scenarios.
    if query_text.strip():
        blocks.extend(_split_into_blocks(query_text, "query"))

    # JD is still target context only, but section hints help the model avoid
    # treating headings such as 任职要求 as a role title.
    if jd_text.strip():
        blocks.extend(_split_into_blocks(jd_text, "jd"))

    return SourceBundle(blocks=blocks)
