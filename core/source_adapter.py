"""SourceAdapter: build SourceBundle from raw extracted text.

V2 Layer 1 — fully deterministic, no LLM involvement.
"""
from __future__ import annotations

import re

from v2_schemas import (
    FactType,
    FactUnit,
    SourceBlock,
    SourceBundle,
    SourceDocument,
    SourceSpan,
)


_SECTION_ALIASES = {
    "meta": ("个人信息", "基本信息", "联系方式"),
    "summary": ("个人总结", "个人简介", "职业概述", "自我评价"),
    "education": ("教育经历", "教育背景", "学历信息"),
    "experience": (
        "工作经历", "实习经历", "工作/实习经历", "工作与实习经历",
        "任职经历", "职业经历",
    ),
    "research": ("科研经历", "研究经历", "实验室经历"),
    "projects": (
        "项目经历", "项目经验", "研究项目", "课程项目", "个人项目", "开源项目",
    ),
    "activities": (
        "校园经历", "在校经历", "社团经历", "志愿经历", "社会实践", "学生工作",
        "组织经历", "社团和组织经历", "社团和",
    ),
    "skills": (
        "技能", "专业技能", "职业技能", "技能清单", "技术栈", "工具", "语言能力",
        "编程语言", "开发工具", "机器学习", "深度学习",
    ),
    "awards": ("荣誉奖项", "荣誉与奖项", "获奖经历", "奖项", "奖学金"),
    "publications": ("论文", "论文发表", "论文成果", "论文期刊", "学术成果", "出版物"),
    "patents": ("专利", "专利成果"),
    "certifications": (
        "证书", "资格证书", "证书与资质", "职业资格", "执业资格", "执照",
    ),
    "training": ("培训经历", "进修经历", "住院医师规范化培训"),
    "teaching": ("教学经历", "授课经历", "培养经历"),
    "hobbies": ("兴趣爱好", "个人爱好"),
    "coursework": ("相关课程", "主修课程"),
}

_LAYOUT_RESET_HEADINGS = {"其他", "其它", "其他信息", "其它信息"}
_GENERIC_SECTION_HEADINGS = {"经历"}
_HEADING_SUFFIX = re.compile(
    r"(?:经历|经验|背景|技能|能力|证书|资质|奖项|荣誉|成果|信息|简介|评价|"
    r"课程|兴趣|爱好|专利|论文|实践|培训|教育)$",
    re.IGNORECASE,
)
_COMPACT_RECORD_DUTY = re.compile(
    r"(?:^|[，,。；;|｜])\s*(?:负责|参与|主导|协助|支持|配合|完成|推动|推进|"
    r"组织|设计|开发|构建|实现|制定|管理|运营|分析|研究|撰写|输出|交付|"
    r"维护|优化|搭建|建立|开展|承担|提供|跟进|协调|带领|执行)"
)
_SOURCE_RELATION_LABEL = re.compile(
    r"^(?:指导老师|指导教师|导师|项目导师|论文导师|推荐人|联系人)\s*[:：]",
    re.IGNORECASE,
)
_PROJECT_TITLE_SIGNAL = re.compile(
    r"(?:项目|系统|平台|课题|作品|游戏|模型|算法)(?:\s*[-–—:]\s*[^。；;!?！？]{2,60})?$",
    re.IGNORECASE,
)
_PROJECT_ACTION_START = re.compile(
    r"^(?:负责|参与|主导|协助|支持|完成|推动|设计|开发|构建|实现|优化|"
    r"研究|分析|测试|搭建|维护|撰写|输出|交付|采用|使用|通过|基于)",
    re.IGNORECASE,
)
_INLINE_AWARD_FACT = re.compile(
    r"(?:奖学金|一等奖|二等奖|三等奖|特等奖|金奖|银奖|铜奖|优秀学生干部|"
    r"优秀志愿者|优秀毕业生|荣誉称号|大赛获奖|竞赛获奖)$"
)


def _normalize_section_heading(value: str) -> str:
    return re.sub(
        r"[\s:：|｜/\\【】\[\]()（）]+", "", str(value or "").strip()
    ).casefold()


_QUERY_DIRECTION_ONLY = re.compile(
    r"(?:请|帮我|麻烦|希望|想要|优化|润色|修改|调整|改成|删除|去掉|不要|"
    r"想(?:找|做|转|投)|转(?:到|向)|禁止|避免|保留|突出|侧重|针对|适配|"
    r"申请|应聘|求职|目标岗位|岗位要求|JD)",
    re.IGNORECASE,
)
_QUERY_FACT_SIGNAL = re.compile(
    r"(?:我(?:叫|是|会|有|曾|在|负责|参与|主导|获得|毕业|就读|熟悉|擅长)|"
    r"本人(?:拥有|具备|曾|在|负责|参与|主导|获得|毕业|就读|熟悉|擅长)|"
    r"姓名是|曾任|任职于|就职于|毕业于|就读于|"
    r"(?:目前|现在|之前|过去|毕业后)?(?:一直)?(?:做|从事|担任|任职|负责)过?|"
    r"(?:主要|日常|平时|工作中)(?:负责|参与|会|需要|经常)|"
    r"(?:学过|自学过?|会用|会使用|使用过|掌握|熟悉|做过|参加过)|"
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
_INLINE_PORTFOLIO_FACT = re.compile(
    r"^(?:我|本人)?(?:在校期间)?(?:做过|完成过|参加过|参与过)"
    r"[^。；;]{2,100}$|"
    r"^(?:我|本人)?有(?:过)?[^。；;]{2,100}经历$",
    re.IGNORECASE,
)
_INLINE_EMPLOYMENT_SIGNAL = re.compile(
    r"(?:工作|实习|任职|就职|全职|兼职|雇员|员工)",
    re.IGNORECASE,
)
_INLINE_EDUCATION_FACT = re.compile(
    # ``大学生`` is an identity word in awards such as ``全国大学生创新创业``;
    # treating its ``大学`` substring as a school silently moves the award into
    # education.  Institution suffixes and explicit qualifications are strong
    # enough structural signals without that false positive.
    r"(?:[一-鿿A-Za-z0-9·.&（）()_-]{1,40}(?:大学(?!生)|学院|学校|研究院)|"
    r"本科|硕士|博士|大专|专科|高中)(?:在读|毕业)?|"
    r"[^，。；;]{1,40}专业(?:毕业|在读)|学历(?:是|为)?(?:本科|硕士|博士|大专|专科)",
    re.IGNORECASE,
)
_INLINE_EXPERIENCE_FACT = re.compile(
    r"(?:(?:19|20)\d{2}[^。；;]{0,80}(?:公司|医院|银行|学校|机构|中心|集团|"
    r"事务所|律所|研究院|实验室|部门)|(?:在|于)[^。；;]{1,40}(?:任职|工作|担任|负责)|"
    r"(?:目前|现在|之前|过去|毕业后)?(?:一直)?(?:做(?!过)|从事|担任|任职于)[^。；;]{2,60}|"
    r"做过\s*(?:\d+|[一二两三四五六七八九十]+)?\s*段?[^。；;]{0,30}(?:实习|工作)|"
    r"(?:工作中|日常工作|平时工作|主要工作)[^。；;]{0,80}(?:负责|参与|统计|分析|策划))",
    re.IGNORECASE,
)
_INLINE_SKILL_FACT = re.compile(
    r"^(?:(?:技能|专业技能|工具|技术栈|语言能力)\s*[:：]?|"
    r"(?:我)?(?:学过|自学过?|会用|会使用|使用过|掌握|熟悉|了解)\s*[^，。；;]{2,100})",
    re.IGNORECASE,
)
_QUERY_FACT_CONTINUATION = re.compile(
    r"^(?:主要|日常|平时|工作中|期间|其中|包括|比如|例如|一段|另一段|"
    r"也|并|同时|此外|另外|曾|过去|目前|现在|方向偏|研究方向|"
    r"负责|参与|主导|协助|支持|完成|开发|设计|搭建|建设|维护|优化|"
    r"统计|分析|策划|撰写|组织|推动|跟进|处理|培训)",
    re.IGNORECASE,
)

_FACT_FIELD_LABEL = re.compile(
    r"^(?:姓名|电话|手机|邮箱|学校|院校|学历|学位|专业|公司|单位|岗位|职位|"
    r"项目名称|项目角色|技能|证书|奖项|任职时间|起止时间|个人总结|自我评价)\s*[:：]\s*",
    re.IGNORECASE,
)
_FACT_SEGMENT_SPLIT = re.compile(
    r"(?:[。；;]+|[|｜]+|(?<=[^\s])[,，、](?=[^\s]))"
)
_FACT_NON_CONTENT = re.compile(
    r"^(?:个人简历|简历|resume|curriculum\s+vitae|cv|"
    r"(?!.*(?:制作|设计|优化|撰写|生成|开发))[\u4e00-\u9fffA-Za-z·]{1,20}(?:个人)?简历)$",
    re.IGNORECASE,
)
_FACT_DISCLAIMER = re.compile(
    r"(?:以(?:真实|实际|最终)[^。；;]{0,40}(?:为准|为依据)|"
    r"(?:不得|不要|避免|严禁)[^。；;]{0,30}(?:编造|虚构|杜撰)|"
    r"(?:信息|内容)[^。；;]{0,20}(?:仅供参考|以.+为准))",
    re.IGNORECASE,
)
_FACT_CONTACT = re.compile(
    r"(?:1[3-9]\d(?:[\s-]?\d){8}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
)
_FACT_DATE = re.compile(
    r"(?:19|20)\d{2}(?:[./年-]\d{1,2}月?)?|至今|现在|应届",
    re.IGNORECASE,
)
_FACT_ORGANIZATION = re.compile(
    r"(?:大学|学院|学校|医院|公司|企业|集团|研究院|实验室|中心|银行|"
    r"事务所|律所|协会|基金会|工作室|团队|部门)$"
)
_FACT_ROLE = re.compile(
    r"(?:(?:岗位|职位|角色|职务)\s*[:：]|担任|任职(?:于|为)?|作为)",
    re.IGNORECASE,
)
_FACT_ACTION = re.compile(
    r"(?:负责|参与|主导|协助|支持|配合|推动|推进|组织|协调|带领|执行|"
    r"设计|开发|构建|实现|制定|管理|运营|分析|统计|策划|培训|处理|研究|"
    r"撰写|输出|交付|维护|优化|搭建|建立|开展|承担|提供|跟进|编制|制作|"
    r"诊断|治疗|授课|教学|复核|检索|调研)"
)
_FACT_METHOD = re.compile(
    r"(?:通过|使用|采用|基于|借助|运用|利用|结合|按照|依托|围绕|经由)"
)
_FACT_DELIVERABLE = re.compile(
    r"(?:输出|交付|完成|形成|上线|发布|落地|搭建|建立|制定|编制|制作|"
    r"撰写|产出|提交|复核|验证|诊断|治疗|授课|培养)"
)
_FACT_RESULT = re.compile(
    r"(?:提升|提高|降低|减少|增长|缩短|节省|达到|达成|获得|获奖|录用|"
    r"成交|销售率|准确率|转化率|留存率|满意度)"
)
_FACT_METRIC = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:%|万|亿|w|k|人|次|个|条|元|年|月|日)?",
    re.IGNORECASE,
)
_FACT_CREDENTIAL = re.compile(
    r"(?:证书|资格|资质|执照|认证|奖学金|一等奖|二等奖|三等奖|金奖|银奖|铜奖)"
)
_FACT_EDUCATION = re.compile(r"(?:本科|硕士|博士|大专|专科|学士|学校|大学|学院|专业)")
_FACT_SKILL = re.compile(r"(?:熟悉|熟练|精通|掌握|会用|使用过|技能|工具|语言)")


def _source_line_entries(text: str) -> list[tuple[str, int, int]]:
    """Return stripped physical lines with offsets into the unchanged text."""

    entries: list[tuple[str, int, int]] = []
    for match in re.finditer(r"[^\r\n]+", text):
        raw = match.group(0)
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right <= left:
            continue
        entries.append((raw[left:right], match.start() + left, match.start() + right))
    return entries


def _query_clause_entries(text: str) -> list[tuple[str, int, int]]:
    """Segment mixed query prose while retaining offsets in the original query."""

    separator = re.compile(
        r"(?:[\r\n，；;。]+|(?<!不)(?:但是|但|不过|然而))",
        re.IGNORECASE,
    )
    entries: list[tuple[str, int, int]] = []
    cursor = 0
    for match in separator.finditer(text):
        raw = text[cursor:match.start()]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            entries.append((raw[left:right], cursor + left, cursor + right))
        cursor = match.end()
    raw = text[cursor:]
    left = len(raw) - len(raw.lstrip())
    right = len(raw.rstrip())
    if right > left:
        entries.append((raw[left:right], cursor + left, cursor + right))
    return entries


def _trim_fact_range(value: str, start: int, end: int) -> tuple[int, int]:
    while start < end and value[start].isspace():
        start += 1
    while start < end and value[end - 1].isspace():
        end -= 1
    bullet = re.match(r"(?:[-*•·▪◦]|\d{1,3}(?:[、)]|\.(?!\d)))\s*", value[start:end])
    if bullet:
        start += bullet.end()
    return start, end


def _fact_dimensions(value: str, section: str | None) -> list[FactType]:
    dimensions: list[FactType] = []
    if _FACT_CONTACT.search(value):
        dimensions.append("contact")
    if _FACT_DATE.search(value):
        dimensions.append("period")
    if _FACT_ORGANIZATION.search(value):
        dimensions.append("organization")
    if _FACT_ROLE.search(value):
        dimensions.append("role")
    if section == "education" or _FACT_EDUCATION.search(value):
        dimensions.append("education")
    if _FACT_ACTION.search(value):
        dimensions.append("action")
    if _FACT_METHOD.search(value):
        dimensions.append("method")
    if _FACT_DELIVERABLE.search(value):
        dimensions.append("deliverable")
    if _FACT_RESULT.search(value):
        dimensions.append("result")
    if section == "skills" or _FACT_SKILL.search(value):
        dimensions.append("skill")
    if section in {"certifications", "awards"} or _FACT_CREDENTIAL.search(value):
        dimensions.append("credential")
    if _FACT_METRIC.search(value):
        dimensions.append("metric")
    if section == "meta" and not dimensions:
        dimensions.append("identity")
    return list(dict.fromkeys(dimensions))


def _primary_fact_type(dimensions: list[FactType]) -> FactType:
    for value in (
        "contact", "organization", "role", "period", "education", "credential",
        "result", "deliverable", "method", "action", "skill", "metric", "identity",
    ):
        if value in dimensions:
            return value  # type: ignore[return-value]
    return "other"


def _query_clause_continues_section(value: str, section: str) -> bool:
    """Carry omitted subjects only inside a compatible factual section."""

    text = str(value or "").strip()
    if not text or not _QUERY_FACT_CONTINUATION.search(text):
        return False
    if section == "education":
        return bool(re.match(
            r"^(?:方向偏|研究方向|专业(?:是|为)|学历(?:是|为)|预计|将于|期间)",
            text,
        ))
    if section in {"experience", "activities", "research"}:
        return not re.match(r"^(?:比如|例如)(?:.+?)(?:项目|系统|平台|课题|作品)", text)
    if section == "projects":
        return True
    return False


def _query_inline_section_hint(value: str) -> str:
    """Classify compact profile clauses by structure, not profession."""

    text = str(value or "").strip()
    if not text:
        return ""
    if _INLINE_PROJECT_FACT.search(text):
        return "projects"
    if (
        _INLINE_PORTFOLIO_FACT.search(text)
        and not _INLINE_EMPLOYMENT_SIGNAL.search(text)
    ):
        # A self-contained activity, trial, coursework item or other portfolio
        # fact is evidence, but ``做过`` alone does not prove employment.  Keep
        # it in the neutral project bucket unless the user explicitly says it
        # was work/internship; this avoids manufacturing an employer or title
        # from phrases such as “做过小学数学试讲”.
        return "projects"
    if _INLINE_SKILL_FACT.search(text):
        return "skills"
    if (
        re.fullmatch(r"(?:我|本人)?(?:是)?做[^，。；;]{2,40}的", text)
        and not _RECORD_DATE.search(text)
        and not _RECORD_ENTITY_TOKEN.search(text)
    ):
        # “我是做智能硬件产品的” is a domain/profile fact, not evidence of a
        # named employment record. The deterministic fallback keeps it as a
        # domain skill and summary fact.
        return ""
    if _INLINE_EXPERIENCE_FACT.search(text):
        return "experience"
    if _INLINE_EDUCATION_FACT.search(text):
        return "education"
    return ""


def _is_section_heading(value: str) -> bool:
    value = re.sub(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff]+$", "", value.strip())
    normalized = _normalize_section_heading(value)
    return normalized in _GENERIC_SECTION_HEADINGS or any(
        normalized == _normalize_section_heading(alias)
        for aliases in _SECTION_ALIASES.values()
        for alias in aliases
    )


def _looks_like_layout_heading(value: str) -> bool:
    """Detect a short visual heading so the previous section cannot leak.

    Unknown headings are deliberately left untyped.  Treating them as a layout
    boundary is safer than assigning every following line to the last known
    section, which can turn an entire employment history into education.
    """

    text = str(value or "").strip().strip(":：|｜/\\【】[]()（）")
    normalized = re.sub(r"\s+", "", text).casefold()
    if not normalized or len(normalized) > 18:
        return False
    if _RECORD_DATE.search(text) or _QUERY_CONTACT_FACT.search(text):
        return False
    if re.search(r"[，,。；;!?！？]", text):
        return False
    return normalized in _GENERIC_SECTION_HEADINGS or bool(_HEADING_SUFFIX.search(normalized))


def _resolve_generic_section_heading(lines: list[str], index: int) -> str:
    """Resolve an ambiguous ``经历`` heading from nearby field grammar.

    The resolver uses dates, entities and action clauses rather than job-title
    vocabulary, so it applies equally to medicine, teaching and operations.
    An ambiguous heading with no structural evidence remains unscoped.
    """

    value = re.sub(r"[\s:：|｜/\\【】\[\]()（）]+", "", lines[index]).casefold()
    if value not in _GENERIC_SECTION_HEADINGS:
        return ""
    following: list[str] = []
    for candidate in lines[index + 1:index + 5]:
        if _section_hint(candidate) or _looks_like_layout_heading(candidate):
            break
        following.append(candidate.strip())
    if not following:
        return ""

    joined = "\n".join(following)
    dated_duty_rows = [
        item for item in following
        if _RECORD_DATE.match(item) and _COMPACT_RECORD_DUTY.search(item)
    ]
    dated_role_rows = [
        item for item in dated_duty_rows
        if _RECORD_ROLE.search(
            re.split(r"[，,。；;]", item, maxsplit=1)[0].strip()
        )
    ]
    # A generic “经历” followed by one explicitly titled dated row, or by
    # multiple dated responsibility rows, is employment/internship history.
    # This structural signal must win before words such as “平台/系统” inside a
    # product-manager duty are mistaken for a project heading.
    if dated_role_rows or len(dated_duty_rows) >= 2:
        return "experience"
    if re.search(r"(?:项目|系统|平台|课题|作品)(?:名称)?\s*[:：]", joined):
        return "projects"
    if any(_INLINE_PROJECT_FACT.search(item) for item in following):
        return "projects"
    if any(
        _RECORD_DATE.search(item)
        and (_RECORD_ENTITY_TOKEN.search(item) or _RECORD_ENTITY.search(item))
        and (_COMPACT_RECORD_DUTY.search(item) or len(re.split(r"[|｜\t]", item)) >= 2)
        for item in following
    ):
        return "experience"
    if any(_INLINE_EXPERIENCE_FACT.search(item) for item in following):
        return "experience"
    if any(_INLINE_EDUCATION_FACT.search(item) for item in following):
        return "education"
    return ""


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
    if _QUERY_DIRECTION_ONLY.search(value):
        return False
    if _QUERY_CONTACT_FACT.search(value) or _QUERY_FACT_SIGNAL.search(value):
        return True
    if _is_section_heading(value):
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
    line = re.sub(r"^[^\w\u4e00-\u9fff]+|[^\w\u4e00-\u9fff]+$", "", line.strip())
    normalized = _normalize_section_heading(line)
    for section, aliases in _SECTION_ALIASES.items():
        if any(normalized == _normalize_section_heading(alias) for alias in aliases):
            return section
    # Inline labels such as "专业技能：Python、SQL" should also carry a
    # structural hint while preserving the complete source line.
    prefix = re.split(r"[:：]", line, maxsplit=1)[0]
    normalized_prefix = _normalize_section_heading(prefix)
    for section, aliases in _SECTION_ALIASES.items():
        if any(normalized_prefix == _normalize_section_heading(alias) for alias in aliases):
            return section
    return ""


_RECORD_SECTIONS = {"education", "experience", "research", "activities", "projects"}
_RECORD_BODY_SIGNAL = re.compile(
    r"^(?:[-*•·▪◦]\s*)?(?:\d{1,3}[.、)]\s*)?"
    r"(?:(?:一段|另一段|工作中|日常工作中?|平时工作中?|期间|其中|也会)\s*)?"
    r"(?:主要\s*)?(?:(?:经常)?(?:需要|会)\s*)?"
    r"(?:负责|参与|主导|协助|支持|配合|完成|推动|推进|组织|"
    r"设计|开发|构建|实现|制定|管理|运营|分析|统计|策划|培训|处理|研究|撰写|输出|交付|维护|优化|"
    r"搭建|建立|开展|承担|提供|跟进|协调|带领|执行|学会)",
    re.IGNORECASE,
)
_RECORD_SERVICE_ACTION = re.compile(
    r"^(?:[-*•·▪◦]\s*)?(?:\d{1,3}[.、)]\s*)?为[^，。；;]{0,32}提供"
)
_RECORD_CONTEXT_ACTION = re.compile(
    r"^(?:[-*•·▪◦]\s*)?(?:\d{1,3}[.、)]\s*)?在[^，。；;]{0,40}"
    r"(?:领导|指导)下[，,]?\s*[^，。；;]{0,48}(?:负责|分管|担任|参与|协助)"
)
_RECORD_RESULT_SIGNAL = re.compile(
    r"(?:提升|降低|增长|减少|缩短|节省|达到|达成|上线|交付|完成|获奖|录用|复核|验证)"
)
_RECORD_DATE_ATOM = (
    r"(?:(?:19|20)\d{2}(?:[./年-]\d{1,2}月?)?"
    r"|(?:0?[1-9]|1[0-2])[./-](?:19|20)\d{2})"
)
_RECORD_DATE = re.compile(
    rf"{_RECORD_DATE_ATOM}(?:\s*[-–—~至到]\s*"
    rf"(?:{_RECORD_DATE_ATOM}|今|至今|现在))?"
)
_RECORD_OPEN_START = re.compile(
    r"^(?:19|20)\d{2}(?:[./年-]\d{1,2}月?)?\s*至\s*$"
)
_RECORD_ENTITY = re.compile(
    r"(?:大学|学院|学校|中学|小学|幼儿园|医院|公司|企业|集团|研究院|实验室|中心|部门|协会|学会|"
    r"学生会|社团|委员会|事务所|律所|银行|政府|基金会|工作室|团队|基地|项目)$"
)
_RECORD_ENTITY_TOKEN = re.compile(
    r"(?:大学|学院|学校|中学|小学|幼儿园|医院|公司|企业|集团|研究院|实验室|中心|部门|协会|学会|"
    r"学生会|社团|委员会|事务所|律所|银行|政府|基金会|工作室|团队|基地)"
)
_RECORD_ROLE = re.compile(
    r"(?:工程师|设计师|教师|老师|医生|医师|护士|经理|主管|总监|主任|顾问|"
    r"研究员|专员|助理|负责人|组长|队长|主席|部长|实习生|实习|分析师|架构师|"
    r"运营|产品|开发|测试|销售|讲师|管理岗|岗位|岗)$"
)


def _looks_like_record_body(value: str) -> bool:
    text = value.strip()
    bullet = bool(re.match(r"^[-*•·▪◦]\s*", text))
    numbered = bool(re.match(r"^\d{1,3}(?:[、)]|\.(?!\d))\s*\S{3,}", text))
    result_at_start = bool(re.match(
        r"^(?:提升|降低|增长|减少|缩短|节省|达到|达成|上线|交付|完成|"
        r"获奖|录用|复核|验证)",
        text,
        re.IGNORECASE,
    ))
    return bool(
        bullet
        or numbered
        or _RECORD_BODY_SIGNAL.search(text)
        or _RECORD_SERVICE_ACTION.search(text)
        or _RECORD_CONTEXT_ACTION.search(text)
        or result_at_start
    )


def _looks_like_record_header(value: str, section: str) -> bool:
    raw = value.strip()
    if re.match(r"^[-*•·▪◦]\s*", raw):
        return False
    text = raw.strip(" \t-•")
    if not text:
        return False
    if re.search(r"[。；;!?！？]$", text):
        return False
    # Compact OCR commonly joins organization, role and department without a
    # delimiter (for example ``某公司 用户增长运营专员运营部``).  Treat the
    # organization token as structural before scanning words such as “增长”,
    # which are valid title text but were previously mistaken for result prose.
    if (
        section in {"experience", "research", "activities"}
        and len(text) <= 80
        and _RECORD_ENTITY_TOKEN.search(text)
        and not _looks_like_record_body(text)
    ):
        return True
    if _looks_like_record_body(text):
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
    saw_dated_header = False
    previous_was_project_title = False
    last_number: int | None = None
    for index, block in enumerate(blocks):
        section = block.section_hint or ""
        if section not in _RECORD_SECTIONS:
            current_section = ""
            current_id = None
            last_number = None
            continue
        if _is_section_heading(block.text):
            current_section = section
            current_id = None
            saw_body = False
            saw_entity_header = False
            saw_dated_header = False
            previous_was_project_title = False
            last_number = None
            continue
        if section != current_section:
            current_section = section
            current_id = None
            saw_body = False
            saw_entity_header = False
            saw_dated_header = False
            previous_was_project_title = False
            last_number = None

        value = block.text.strip()
        number_match = re.match(r"^(\d{1,3})(?:[、)]|\.(?!\d))\s*", value)
        item_number = int(number_match.group(1)) if number_match else None
        is_body = _looks_like_record_body(value)
        is_header = _looks_like_record_header(value, section)
        is_entity = bool(_RECORD_ENTITY.search(value.strip(" \t-•")))
        is_relation_label = bool(_SOURCE_RELATION_LABEL.match(value))
        project_named_header = bool(
            section == "projects"
            and len(value) <= 100
            and not re.search(r"[。；;!?！？]$", value)
            and re.search(r"(?:项目|系统|平台|课题|作品|游戏|模型|算法)$", value)
        )
        if project_named_header:
            is_body = False
            is_header = True
        if is_relation_label:
            is_header = False
            is_entity = False
        has_date = bool(_RECORD_DATE.search(value))
        has_entity_token = bool(_RECORD_ENTITY_TOKEN.search(value))
        previous_value = blocks[index - 1].text.strip() if index > 0 else ""
        previous_same_section = (
            index > 0 and (blocks[index - 1].section_hint or "") == section
        )
        continuation_from_previous = bool(
            previous_same_section
            and previous_value
            and not re.search(r"[。；;!?！？]$", previous_value)
            and (
                _looks_like_record_body(previous_value)
                or (
                    saw_body
                    and not _looks_like_record_header(previous_value, section)
                    and not _RECORD_DATE.fullmatch(previous_value)
                )
            )
            and item_number is None
            and not _RECORD_DATE.fullmatch(value)
            and not (
                _RECORD_ENTITY_TOKEN.search(value)
                and not re.search(r"[。；;!?！？]$", value)
            )
            and len(re.split(r"[|｜\t]", value)) < 2
            and len(value) <= 160
        )
        if continuation_from_previous:
            # PDF/OCR line wrapping can put the tail of a duty on the next
            # physical line.  It belongs to the current record even when the
            # tail itself looks like a short header (for example “月销售率…”).
            is_body = True
            is_header = False
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
        project_title_text = re.sub(r"^[-*•·▪◦]\s*", "", value).strip()
        is_project_title = bool(
            section == "projects"
            and re.match(r"^[-*•·▪◦]\s*\S", value)
            and next_same_section
            and (
                len(re.split(r"[|｜\t]", next_value)) >= 2
                or bool(_RECORD_ENTITY_TOKEN.search(next_value))
                or bool(_RECORD_DATE.fullmatch(next_value))
                or (
                    len(project_title_text) <= 140
                    and not _PROJECT_ACTION_START.match(project_title_text)
                    and bool(_PROJECT_TITLE_SIGNAL.search(project_title_text))
                )
            )
        )

        starts_record = current_id is None
        if current_id is not None:
            if is_project_title:
                starts_record = True
            elif (
                item_number is not None
                and last_number is not None
                and item_number <= last_number
                and saw_body
            ):
                # Multi-column OCR frequently removes later project/job
                # headers while retaining numbered duty lists. A numbering
                # restart is a domain-neutral, explicit record boundary.
                starts_record = True
            elif _RECORD_OPEN_START.fullmatch(value) and saw_entity_header:
                starts_record = True
            elif (
                saw_dated_header
                and has_date
                and (
                    has_entity_token
                    or (
                        _RECORD_DATE.match(value)
                        and _COMPACT_RECORD_DUTY.search(value)
                    )
                )
            ):
                # Compact rows commonly contain date, organization, title and
                # duty in one physical line.  A later date-leading duty row is
                # an explicit new record even when the organization is a brand
                # name without a legal suffix such as “公司/医院”.
                starts_record = True
            elif (
                saw_body
                and not previous_was_project_title
                and not is_relation_label
                and (is_header or short_header_before_role)
            ):
                starts_record = True
            elif (
                saw_body
                and not is_body
                and not is_relation_label
                and len(value) <= 64
                and not re.search(r"[。；;!?！？]$", value)
                and (
                    section != "projects"
                    or (
                        next_same_section
                        and _looks_like_record_body(next_value)
                        and not value.endswith((":", "："))
                    )
                )
            ):
                # In undated project/campus sections, a short plain line after
                # bullets is normally the next record header. Requiring a
                # profession dictionary here caused later projects and clubs
                # to collapse into the first record.
                starts_record = True
            elif is_entity and not is_relation_label and saw_entity_header:
                starts_record = True
            elif (
                is_header
                and not is_relation_label
                and saw_entity_header
                and len(re.split(r"[|｜\t]", value)) >= 2
                and (
                    section != "education"
                    or bool(_RECORD_DATE.search(value))
                    or bool(_RECORD_ENTITY_TOKEN.search(value))
                )
            ):
                starts_record = True
        if starts_record:
            record_index += 1
            current_id = f"{block.source_type}:{section}:{record_index}"
            saw_body = False
            saw_entity_header = False
            saw_dated_header = False
            last_number = None
        block.record_id = current_id
        # A bullet marker is also commonly used for a project title.  It starts
        # a record but must not make the immediately following organization/date
        # header look like a new record body transition.
        saw_body = saw_body or (is_body and not is_project_title)
        if item_number is not None:
            last_number = item_number
        saw_entity_header = saw_entity_header or is_entity or (
            is_header and len(re.split(r"[|｜\t]", value)) >= 2
        )
        saw_dated_header = saw_dated_header or bool(
            has_date
            and (
                has_entity_token
                or (
                    _RECORD_DATE.match(value)
                    and _COMPACT_RECORD_DUTY.search(value)
                )
            )
        )
        previous_was_project_title = is_project_title


def _coalesce_wrapped_blocks(blocks: list[SourceBlock]) -> list[SourceBlock]:
    """Join physical OCR wraps before they enter Composer/evidence checks."""

    merged: list[SourceBlock] = []
    for block in blocks:
        value = block.text.strip()
        previous = merged[-1] if merged else None
        previous_value = previous.text.strip() if previous else ""
        same_context = bool(
            previous
            and previous.source_type == block.source_type
            and previous.section_hint == block.section_hint
            and (
                previous.record_id == block.record_id
                or (previous.record_id is None and block.record_id is None)
            )
        )
        continuation = bool(
            same_context
            and previous_value
            and not re.search(r"[。；;!?！？]$", previous_value)
            and _looks_like_record_body(previous_value)
            and not _looks_like_record_body(value)
            and not re.match(r"^(?:[-*•·▪◦]|\d{1,3}(?:[、)]|\.(?!\d)))\s*", value)
            and not _RECORD_DATE.fullmatch(value)
            and not _RECORD_ENTITY_TOKEN.search(value)
            and len(re.split(r"[|｜\t]", value)) < 2
            and not _is_section_heading(value)
            and not _QUERY_CONTACT_FACT.search(value)
            and not re.search(
                r"[·•]\s*(?:精通|熟练(?:使用|掌握)?|熟悉|了解|掌握)\s*$",
                previous_value,
            )
            and len(value) <= 180
        )
        if continuation and previous is not None:
            previous.text = previous_value + value
            previous.source_spans.extend(block.source_spans)
            previous.origin_block_ids.extend(
                block.origin_block_ids or [block.block_id]
            )
        else:
            merged.append(block)
    # A multi-column extractor can emit the tail of a sentence before the
    # sentence head.  Rank every structurally eligible earlier fragment and
    # join only a unique high-confidence match.  This avoids occupation/role
    # special cases and fixed look-back windows while preserving literal text.
    split_word_boundaries = {
        "报告", "提供", "进行", "完成", "负责", "支持", "协助", "管理",
        "维护", "分析", "统计", "策划", "培训", "处理", "研究", "撰写",
        "输出", "交付", "推动", "推进", "组织", "设计", "开发", "构建",
        "实现", "制定", "运营", "建立", "开展", "承担", "跟进", "协调",
        "带领", "执行", "治疗", "服务",
    }
    trailing_object_verb = re.compile(
        r"(?:协助|引导|支持|帮助|提供|完成|处理|维护|管理|分析|统计|"
        r"策划|培训|服务)$"
    )
    consumed_tail_ids: set[int] = set()
    for head_index, head in enumerate(merged):
        head_value = head.text.strip()
        if (
            id(head) in consumed_tail_ids
            or not _looks_like_record_body(head_value)
            or re.search(r"[。；;!?！？]$", head_value)
        ):
            continue
        candidates: list[tuple[float, str, int, SourceBlock]] = []
        for tail_index, tail in enumerate(merged[:head_index]):
            tail_value = tail.text.strip()
            if (
                id(tail) in consumed_tail_ids
                or tail.source_type != head.source_type
                or len(tail_value) < 2
                or len(tail_value) > 120
                or not re.search(r"[。；;!?！？]$", tail_value)
                or _looks_like_record_body(tail_value)
                or _RECORD_DATE.fullmatch(tail_value)
                or _RECORD_ENTITY_TOKEN.search(tail_value)
                or _is_section_heading(tail_value)
                or _QUERY_CONTACT_FACT.search(tail_value)
            ):
                continue
            distance = head_index - tail_index
            split_word = head_value[-1:] + tail_value[:1]
            if split_word in split_word_boundaries and len(tail_value) <= 24:
                candidates.append((4.0 + 1.0 / distance, "split_word", tail_index, tail))
                continue
            if (
                trailing_object_verb.search(head_value.strip("。；; "))
                and tail.record_id
                and tail.section_hint in _RECORD_SECTIONS
            ):
                candidates.append((3.0 + 1.0 / distance, "object", tail_index, tail))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, mode, _tail_index, tail = candidates[0]
        if len(candidates) > 1 and best_score - candidates[1][0] < 0.20:
            continue
        if mode == "object":
            object_candidates = [item for item in candidates if item[1] == "object"]
            latest_record_id = next((
                candidate.record_id
                for candidate in reversed(merged[:head_index])
                if candidate.record_id and id(candidate) not in consumed_tail_ids
            ), None)
            if len(object_candidates) != 1 or tail.record_id != latest_record_id:
                continue
        head.text = head_value.rstrip("。；; ") + tail.text.strip()
        head.source_spans.extend(tail.source_spans)
        head.origin_block_ids.extend(tail.origin_block_ids or [tail.block_id])
        # A non-numbered object continuation inherits the record provenance of
        # its consumed fragment. Numbered duties keep their own unscoped order
        # so the separate numbering matcher can associate the complete list.
        if mode == "object" and not re.match(
            r"^(?:[-*•·▪◦]|\d{1,3}(?:[、)]|\.(?!\d)))\s*",
            head_value,
        ):
            head.section_hint = tail.section_hint
            head.record_id = tail.record_id
        consumed_tail_ids.add(id(tail))
    if consumed_tail_ids:
        merged = [block for block in merged if id(block) not in consumed_tail_ids]
    return merged


def _split_into_blocks(
    text: str,
    source_type: str,
    *,
    source_id: str | None = None,
    entries: list[tuple[str, int, int]] | None = None,
) -> list[SourceBlock]:
    """Split text into ordered blocks and retain deterministic section hints."""
    blocks: list[SourceBlock] = []
    current_section = ""
    resolved_source_id = source_id or source_type
    line_entries = entries if entries is not None else _source_line_entries(text)
    lines = [line for line, _start, _end in line_entries]
    for line_index, (line, char_start, char_end) in enumerate(line_entries):
        normalized_line = _normalize_section_heading(line)
        if re.fullmatch(r"[-*•·▪◦—–_]+", line):
            continue
        if normalized_line in _LAYOUT_RESET_HEADINGS:
            current_section = ""
            block_id = f"{source_type}_{len(blocks)}"
            blocks.append(SourceBlock(
                block_id=block_id,
                source_type=source_type,  # type: ignore
                text=line,
                source_id=resolved_source_id,
                source_spans=[SourceSpan(
                    source_id=resolved_source_id,
                    char_start=char_start,
                    char_end=char_end,
                )],
                origin_block_ids=[block_id],
                section_hint=None,
                fact_eligible=False,
            ))
            continue
        detected = _section_hint(line)
        if not detected and normalized_line in _GENERIC_SECTION_HEADINGS:
            detected = _resolve_generic_section_heading(lines, line_index)
        heading_boundary = _looks_like_layout_heading(line)
        layout_boundary = bool(
            re.search(r"(?:^|[。；;，,])\s*(?:求职意向|目标岗位|应聘岗位)\s*[:：]", line)
            or _QUERY_CONTACT_FACT.search(line)
        )
        if (layout_boundary or heading_boundary) and not detected:
            current_section = ""
        if detected:
            current_section = detected
        assigned_section = None if layout_boundary else (detected or current_section or None)
        if not detected and _INLINE_AWARD_FACT.search(line.strip(" 	-•·，,。；;")):
            # Strong award morphology overrides a stale visual skills column.
            # This is a per-line correction so following unrelated content
            # does not silently inherit an inferred section.
            assigned_section = "awards"
        skill_dense = len(re.findall(r"(?:熟练|精通|掌握|熟悉|了解)", line)) >= 2
        standalone_skill = bool(re.fullmatch(
            r"[-•·]?\s*[^，,。；;]{2,48}?(?:熟练|精通|掌握|熟悉|了解)",
            line,
        ))
        if (
            not detected
            and assigned_section not in _RECORD_SECTIONS
            and assigned_section != "skills"
            and (skill_dense or standalone_skill)
        ):
            assigned_section = "skills"
        if not detected and assigned_section in {"skills", "hobbies", "coursework"}:
            compact = line.strip(" \t-•·")
            detailed_prose = bool(
                len(compact) >= 28
                and re.search(r"[，,。；;]", compact)
                and re.search(
                    r"(?:负责|参与|主导|协助|完成|推动|运营|策划|项目|实习|"
                    r"工作|用户|客户|活动|数据|成果|提升|增长|降低|达到|输出|"
                    r"专业|经验|大学|本科|硕士|营销|担任)",
                    compact,
                )
            )
            if (
                (_looks_like_record_body(compact) and not standalone_skill)
                or detailed_prose
                or bool(_RECORD_DATE.fullmatch(compact))
                or bool(_QUERY_CONTACT_FACT.search(compact))
            ):
                # Multi-column PDF/OCR often emits all prose first and the
                # aligned dates last, after a visually separate skills column.
                # Keeping these lines under skills converts entire jobs into
                # dozens of bogus skill names. Leave them unscoped so Composer
                # or the conservative anonymous-history fallback can retain
                # them without inventing an association.
                assigned_section = None
        # Multi-column OCR can leave a later duty list below an awards column.
        # A numbered/action-led sentence is not an award merely because the
        # most recent visual heading was “奖学金”. Leave it unscoped so the
        # Composer may preserve it without inventing an employer association.
        if (
            not detected
            and assigned_section == "awards"
            and (
                _RECORD_BODY_SIGNAL.search(line)
                or _RECORD_SERVICE_ACTION.search(line)
                or _RECORD_CONTEXT_ACTION.search(line)
            )
        ):
            assigned_section = None
        block_id = f"{source_type}_{len(blocks)}"
        blocks.append(SourceBlock(
            block_id=block_id,
            source_type=source_type,  # type: ignore
            text=line,
            source_id=resolved_source_id,
            source_spans=[SourceSpan(
                source_id=resolved_source_id,
                char_start=char_start,
                char_end=char_end,
            )],
            origin_block_ids=[block_id],
            section_hint=assigned_section,
        ))
    _assign_record_ids(blocks)
    return _coalesce_wrapped_blocks(blocks) if source_type == "resume" else blocks


def _build_fact_units(
    blocks: list[SourceBlock],
    documents: list[SourceDocument],
) -> list[FactUnit]:
    """Build exact physical-source facts while retaining logical block owners."""

    document_text = {document.source_id: document.text for document in documents}
    facts: list[FactUnit] = []
    for block in blocks:
        origin_ids = block.origin_block_ids or [block.block_id]
        spans = block.source_spans
        if not spans:
            continue
        block_fact_ids: list[str] = []
        for physical_index, span in enumerate(spans):
            source_text = document_text.get(span.source_id, "")
            if not (0 <= span.char_start <= span.char_end <= len(source_text)):
                continue
            physical_text = source_text[span.char_start:span.char_end]
            if not physical_text.strip() or _is_section_heading(physical_text):
                continue
            if _FACT_NON_CONTENT.fullmatch(physical_text.strip()):
                continue

            content_start = 0
            label = _FACT_FIELD_LABEL.match(physical_text)
            if label:
                content_start = label.end()
            ranges: list[tuple[int, int]] = []
            cursor = content_start
            for delimiter in _FACT_SEGMENT_SPLIT.finditer(
                physical_text, content_start,
            ):
                start, end = _trim_fact_range(
                    physical_text, cursor, delimiter.start(),
                )
                if end > start:
                    ranges.append((start, end))
                cursor = delimiter.end()
            start, end = _trim_fact_range(
                physical_text, cursor, len(physical_text),
            )
            if end > start:
                ranges.append((start, end))
            ranges = [
                (start, end) for start, end in ranges
                if len(re.sub(r"\s+", "", physical_text[start:end])) >= 2
            ]
            if not ranges:
                continue

            origin_block_id = (
                origin_ids[physical_index]
                if physical_index < len(origin_ids)
                else block.block_id
            )
            for fact_index, (start, end) in enumerate(ranges):
                value = physical_text[start:end]
                normalized = re.sub(
                    r"[\s:：|｜/\\【】\[\]()（）]+", "", value,
                ).casefold()
                if not normalized:
                    continue
                dimensions = _fact_dimensions(value, block.section_hint)
                fact_id = (
                    origin_block_id
                    if len(ranges) == 1
                    else f"{origin_block_id}#u{fact_index}"
                )
                fact = FactUnit(
                    fact_id=fact_id,
                    block_id=block.block_id,
                    origin_block_id=origin_block_id,
                    source_type=block.source_type,
                    source_spans=[SourceSpan(
                        source_id=span.source_id,
                        char_start=span.char_start + start,
                        char_end=span.char_start + end,
                    )],
                    section_hint=block.section_hint,
                    record_id=block.record_id,
                    fact_type=_primary_fact_type(dimensions),
                    dimensions=dimensions,
                    verbatim_text=value,
                    normalized_text=normalized,
                    fact_eligible=(
                        (
                            block.source_type == "resume"
                            and not _FACT_DISCLAIMER.search(value)
                        ) or (
                            block.source_type == "query"
                            and block.fact_eligible
                            and not _FACT_DISCLAIMER.search(value)
                        )
                    ),
                    confidence=1.0,
                )
                facts.append(fact)
                block_fact_ids.append(fact_id)
        block.fact_ids = list(dict.fromkeys(block_fact_ids))
    return facts


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
    documents: list[SourceDocument] = []

    # Resume / CV text blocks
    if cv_text.strip():
        documents.append(SourceDocument(
            source_id="resume", source_type="resume", text=cv_text,
        ))
        blocks.extend(_split_into_blocks(
            cv_text, "resume", source_id="resume",
        ))

    # A multiline query often contains the only candidate profile.  Segment it
    # with the same logic so headings survive in no-CV scenarios.
    if query_text.strip():
        documents.append(SourceDocument(
            source_id="query", source_type="query", text=query_text,
        ))
        # Split mixed “fact + instruction” prose into clauses while offsets
        # continue to point into the original query string.
        query_blocks = _split_into_blocks(
            query_text,
            "query",
            source_id="query",
            entries=_query_clause_entries(query_text),
        )
        has_cv = bool(cv_text.strip())
        active_record_section = ""
        for block in query_blocks:
            inferred_section = _query_inline_section_hint(block.text)
            if inferred_section:
                block.section_hint = inferred_section
                active_record_section = (
                    inferred_section if inferred_section in _RECORD_SECTIONS else ""
                )
            elif active_record_section and (
                _looks_like_record_body(block.text)
                or bool(_RECORD_DATE.fullmatch(block.text.strip()))
                or _query_clause_continues_section(block.text, active_record_section)
            ):
                # Compact descriptions are often comma-delimited as
                # ``company/role, period, action/result``.  Keep a standalone
                # date with the active record so the following action does not
                # become an orphaned query fact.
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
        documents.append(SourceDocument(
            source_id="jd", source_type="jd", text=jd_text,
        ))
        blocks.extend(_split_into_blocks(jd_text, "jd", source_id="jd"))

    facts = _build_fact_units(blocks, documents)
    return SourceBundle(
        blocks=blocks,
        documents=documents,
        fact_units=facts,
    )


def candidate_blocks(source: SourceBundle) -> list[SourceBlock]:
    """Return blocks allowed to support candidate resume facts."""

    return [
        block for block in source.blocks
        if block.source_type == "resume"
        or (block.source_type == "query" and block.fact_eligible)
    ]
