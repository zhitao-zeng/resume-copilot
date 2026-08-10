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
    r"禁止|避免|保留|突出|侧重|针对|适配|申请|应聘|求职|目标岗位|岗位要求|JD)",
    re.IGNORECASE,
)
_QUERY_FACT_SIGNAL = re.compile(
    r"(?:我(?:叫|是|会|有|曾|在|负责|参与|主导|获得|毕业|就读|熟悉|擅长)|"
    r"本人(?:拥有|具备|曾|在|负责|参与|主导|获得|毕业|就读|熟悉|擅长)|"
    r"姓名是|曾任|任职于|就职于|毕业于|就读于|"
    r"\d+(?:\.\d+)?\s*(?:年|个月)\s*(?:工作|从业|实习)?(?:经验|经历))",
    re.IGNORECASE,
)
_QUERY_CONTACT_FACT = re.compile(
    r"(?:1[3-9]\d(?:[\s-]?\d){8}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
)
_QUERY_NEGATIVE_INSTRUCTION = re.compile(r"(?:不要|禁止|删除|去掉|避免|不得|不能写|不要写)")
_QUERY_STRUCTURED_FACT = re.compile(
    r"(?:姓名|电话|手机|邮箱|学校|院校|学历|学位|专业|公司|单位|岗位|职位|"
    r"项目名称|项目角色|技能|证书|奖项|任职时间|起止时间)\s*[:：]|"
    r"姓名\s*[\u4e00-\u9fff·]{2,16}(?:\s|$)|"
    r"(?:做过|参与|负责|主导|开发|设计|搭建|完成|开展)[^。；;]{0,80}"
    r"(?:项目|系统|平台|课题|作品)|"
    r"(?:19|20)\d{2}(?:[./年-]\d{1,2}月?)?\s*(?:[-—~至到]|年\s*[-—~至到])\s*"
    r"(?:(?:19|20)\d{2}(?:[./年-]\d{1,2}月?)?|至今|现在)|"
    r"(?:本科|硕士|博士|大专|专科|高中)(?:在读|毕业)?|"
    r"[\u4e00-\u9fffA-Za-z0-9·.&（）()_-]{2,40}(?:大学|学院|学校|医院|公司|集团|研究院|实验室)",
    re.IGNORECASE,
)

_INLINE_PROJECT_FACT = re.compile(
    r"(?:做过|参与|负责|主导|开发|设计|搭建|完成|开展)[^。；;]{0,80}"
    r"(?:项目|系统|平台|课题|作品)",
    re.IGNORECASE,
)
_INLINE_EDUCATION_FACT = re.compile(
    r"(?:大学|学院|学校|本科|硕士|博士|大专|专科|高中)",
    re.IGNORECASE,
)
_INLINE_EXPERIENCE_FACT = re.compile(
    r"(?:(?:19|20)\d{2}[^。；;]{0,80}(?:公司|医院|银行|学校|机构|中心|集团|"
    r"事务所|研究院|实验室|部门)|(?:在|于)[^。；;]{1,40}(?:任职|工作|担任|负责))",
    re.IGNORECASE,
)
_INLINE_SKILL_FACT = re.compile(r"^(?:技能|专业技能|工具|技术栈|语言能力)\s*[:：]?", re.IGNORECASE)


def _query_inline_section_hint(value: str) -> str:
    """Classify compact profile clauses by structure, not profession."""

    text = str(value or "").strip()
    if not text:
        return ""
    if _INLINE_PROJECT_FACT.search(text):
        return "projects"
    if _INLINE_EXPERIENCE_FACT.search(text):
        return "experience"
    if _INLINE_EDUCATION_FACT.search(text):
        return "education"
    if _INLINE_SKILL_FACT.search(text):
        return "skills"
    return ""


def _is_section_heading(value: str) -> bool:
    normalized = re.sub(r"[\s:：|｜/\\【】\[\]()（）]+", "", value).casefold()
    return any(
        normalized == re.sub(r"\s+", "", alias).casefold()
        for aliases in _SECTION_ALIASES.values()
        for alias in aliases
    )


def _query_line_is_fact(
    line: str,
    *,
    has_cv: bool,
    section_hint: str | None = None,
) -> bool:
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
    if _QUERY_CONTACT_FACT.search(value) or _QUERY_FACT_SIGNAL.search(value):
        return True
    if _is_section_heading(value):
        return False
    if _QUERY_DIRECTION_ONLY.search(value):
        return False
    # Lines placed under an explicit resume section are structured candidate
    # evidence even when they omit first-person wording.  This supports pasted
    # form data while keeping a bare role title or an embedded JD out of the
    # evidence pool.
    if section_hint in _SECTION_ALIASES:
        return True
    if _QUERY_STRUCTURED_FACT.search(value):
        return True
    # Ambiguous query-only prose is an instruction/target by default.  It is
    # safer to ask the user to confirm a fact than to turn a JD sentence into a
    # fabricated candidate experience.
    return False


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


_RECORD_SECTIONS = {"education", "experience", "research", "activities", "projects"}
_RECORD_BODY_SIGNAL = re.compile(
    r"^(?:[-*•·▪◦]\s*)?(?:负责|参与|主导|协助|支持|配合|完成|推动|推进|组织|"
    r"设计|开发|构建|实现|制定|管理|运营|分析|研究|撰写|输出|交付|维护|优化|"
    r"搭建|建立|开展|承担|提供|跟进|协调|带领|执行)|"
    r"(?:提升|降低|增长|减少|缩短|节省|达到|达成|上线|获奖|录用|复核|验证)",
    re.IGNORECASE,
)
_RECORD_RESULT_SIGNAL = re.compile(
    r"(?:提升|降低|增长|减少|缩短|节省|达到|达成|上线|交付|完成|获奖|录用|复核|验证)"
)
_RECORD_DATE = re.compile(
    r"(?:19|20)\d{2}(?:[./年-]\d{1,2}月?)?(?:\s*[-—~至到]\s*"
    r"(?:(?:19|20)\d{2}(?:[./年-]\d{1,2}月?)?|至今|现在))?"
)
_RECORD_ENTITY = re.compile(
    r"(?:大学|学院|学校|医院|公司|集团|研究院|实验室|中心|部门|协会|学会|"
    r"学生会|社团|委员会|事务所|银行|政府|基金会|工作室|团队|项目)$"
)
_RECORD_ROLE = re.compile(
    r"(?:工程师|设计师|教师|老师|医生|医师|护士|经理|主管|总监|主任|顾问|"
    r"研究员|专员|助理|负责人|组长|队长|主席|部长|实习生|分析师|架构师|"
    r"运营|产品|开发|测试|销售|讲师)$"
)


def _looks_like_record_body(value: str) -> bool:
    text = value.strip()
    return bool(
        re.match(r"^[-*•·▪◦]\s*", text)
        or _RECORD_BODY_SIGNAL.search(text)
        or _RECORD_RESULT_SIGNAL.search(text)
    )


def _looks_like_record_header(value: str, section: str) -> bool:
    text = value.strip(" \t-•")
    if not text or _looks_like_record_body(text):
        return False
    if len(re.split(r"[|｜\t]", text)) >= 2:
        return True
    if _RECORD_DATE.search(text):
        return True
    if _RECORD_ENTITY.search(text):
        return True
    if section == "projects" and re.search(r"(?:项目|系统|平台|课题|作品|方案)$", text):
        return True
    return False


def _assign_record_ids(blocks: list[SourceBlock]) -> None:
    """Attach conservative record boundaries without using an industry lexicon."""

    current_section = ""
    current_id: str | None = None
    record_index = -1
    saw_body = False
    saw_entity_header = False
    for index, block in enumerate(blocks):
        section = block.section_hint or ""
        if section not in _RECORD_SECTIONS:
            current_section = ""
            current_id = None
            continue
        if _is_section_heading(block.text):
            current_section = section
            current_id = None
            saw_body = False
            saw_entity_header = False
            continue
        if section != current_section:
            current_section = section
            current_id = None
            saw_body = False
            saw_entity_header = False

        value = block.text.strip()
        is_body = _looks_like_record_body(value)
        is_header = _looks_like_record_header(value, section)
        is_entity = bool(_RECORD_ENTITY.search(value.strip(" \t-•")))
        next_value = blocks[index + 1].text.strip() if index + 1 < len(blocks) else ""
        next_same_section = (
            index + 1 < len(blocks)
            and (blocks[index + 1].section_hint or "") == section
        )
        short_header_before_role = bool(
            next_same_section
            and len(value) <= 32
            and not is_body
            and _RECORD_ROLE.search(next_value.strip(" \t-•"))
        )

        starts_record = current_id is None
        if current_id is not None:
            if saw_body and (is_header or short_header_before_role):
                starts_record = True
            elif is_entity and saw_entity_header:
                starts_record = True
            elif is_header and saw_entity_header and len(re.split(r"[|｜\t]", value)) >= 2:
                starts_record = True
        if starts_record:
            record_index += 1
            current_id = f"{block.source_type}:{section}:{record_index}"
            saw_body = False
            saw_entity_header = False
        block.record_id = current_id
        saw_body = saw_body or is_body
        saw_entity_header = saw_entity_header or is_entity or (
            is_header and len(re.split(r"[|｜\t]", value)) >= 2
        )


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
    _assign_record_ids(blocks)
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
        # Split mixed “fact + instruction” prose into clauses. This lets
        # “我是做智能硬件产品的，帮我优化简历” retain only the first clause as
        # evidence instead of legitimizing the instruction as a candidate fact.
        segmented_query = re.sub(r"[，；;。]+", "\n", query_text)
        query_blocks = _split_into_blocks(segmented_query, "query")
        has_cv = bool(cv_text.strip())
        active_record_section = ""
        for block in query_blocks:
            inferred_section = _query_inline_section_hint(block.text)
            if inferred_section:
                block.section_hint = inferred_section
                active_record_section = (
                    inferred_section if inferred_section in _RECORD_SECTIONS else ""
                )
            elif active_record_section and _looks_like_record_body(block.text):
                block.section_hint = active_record_section
            elif block.section_hint not in _RECORD_SECTIONS:
                active_record_section = ""
            block.fact_eligible = _query_line_is_fact(
                block.text,
                has_cv=has_cv,
                section_hint=block.section_hint,
            )
        # Inline hints are assigned after the initial line split, so rebuild
        # record boundaries once the compact clauses have been classified.
        _assign_record_ids(query_blocks)
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
