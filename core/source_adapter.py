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
    "research": ("科研经历", "研究经历", "实验室经历"),
    "projects": ("项目经历", "项目经验", "课程项目", "个人项目", "开源项目"),
    "activities": ("校园经历", "社团经历", "志愿经历", "社会实践", "学生工作"),
    "skills": ("专业技能", "技能清单", "技术栈", "工具", "语言能力"),
    "awards": ("荣誉奖项", "荣誉与奖项", "获奖经历", "奖项"),
    "publications": ("论文", "论文发表", "论文成果", "学术成果", "出版物"),
    "patents": ("专利", "专利成果"),
    "certifications": ("证书", "证书与资质", "职业资格", "执业资格", "执照"),
    "training": ("培训经历", "进修经历", "住院医师规范化培训"),
    "teaching": ("教学经历", "授课经历", "培养经历"),
}


_QUERY_DIRECTION_ONLY = re.compile(
    r"(?:请|帮我|麻烦|需要|希望|想要|优化|润色|修改|调整|改成|删除|去掉|不要|"
    r"禁止|避免|突出|侧重|针对|适配|申请|应聘|求职|目标岗位|岗位要求|JD)",
    re.IGNORECASE,
)
_QUERY_FACT_SIGNAL = re.compile(
    r"(?:我(?:会|有|曾|在|负责|参与|主导|获得|毕业|就读|熟悉|擅长)|本人|我的|"
    r"补充(?:信息|经历|技能)?|新增(?:信息|经历|技能)?|曾任|任职|就职|毕业于|就读于|"
    r"负责|参与|主导|获得|持有|熟悉|擅长|\d+(?:\.\d+)?\s*(?:年|个月)\s*(?:经验|经历))",
    re.IGNORECASE,
)
_QUERY_CONTACT_FACT = re.compile(
    r"(?:1[3-9]\d(?:[\s-]?\d){8}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
)
_QUERY_NEGATIVE_INSTRUCTION = re.compile(r"(?:不要|禁止|删除|去掉|避免|不得|不能写|不要写)")


def _query_line_is_fact(line: str, *, has_cv: bool) -> bool:
    """Separate factual additions from editing/target instructions.

    With an uploaded CV we require an explicit first-person/factual signal;
    otherwise a request such as "不要写管理经验" could become evidence that
    the candidate actually has management experience. In query-only mode,
    neutral profile lines remain eligible so terse form submissions still work.
    """

    value = str(line or "").strip()
    if not value:
        return False
    if _QUERY_NEGATIVE_INSTRUCTION.search(value):
        return False
    if _QUERY_FACT_SIGNAL.search(value) or _QUERY_CONTACT_FACT.search(value):
        return True
    if _QUERY_DIRECTION_ONLY.search(value):
        return False
    return not has_cv


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
        query_blocks = _split_into_blocks(query_text, "query")
        has_cv = bool(cv_text.strip())
        for block in query_blocks:
            block.fact_eligible = _query_line_is_fact(block.text, has_cv=has_cv)
        blocks.extend(query_blocks)

    # JD is still target context only, but section hints help the model avoid
    # treating headings such as 任职要求 as a role title.
    if jd_text.strip():
        blocks.extend(_split_into_blocks(jd_text, "jd"))

    return SourceBundle(blocks=blocks)


def candidate_blocks(source: SourceBundle) -> list[SourceBlock]:
    """Return blocks allowed to support candidate resume facts."""

    return [
        block for block in source.blocks
        if block.source_type == "resume"
        or (block.source_type == "query" and block.fact_eligible)
    ]
