"""Domain-neutral resume section names shared by every V3 input adapter."""
from __future__ import annotations

import re


SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "contact": (
        "个人信息", "基本信息", "联系方式", "联系信息", "contact",
        "contact information", "contact details", "personal information",
    ),
    "summary": (
        "个人总结", "个人简介", "职业概述", "专业概述", "自我评价", "职业目标",
        "求职目标", "摘要或目标", "总结或目标", "summary", "profile", "objective",
        "professional summary", "summary or objective", "career objective",
    ),
    "experience": (
        "工作经历", "工作经验", "工作实习经历", "工作/实习经历", "工作与实习经历",
        "实习经历", "职业经历", "专业经历", "专业经验", "任职经历", "experience",
        "employment", "work experience", "professional experience", "career history",
        "experience highlights",
    ),
    "projects": (
        "项目经历", "项目经验", "项目", "课程项目", "个人项目", "开源项目",
        "学术项目与研讨会", "projects", "project experience", "academic projects",
    ),
    "research": (
        "科研经历", "研究经历", "实验室经历", "科研项目", "研究项目", "research",
        "research experience",
    ),
    "activities": (
        "校园经历", "校园活动", "在校经历", "社团经历", "社会实践", "学生工作",
        "组织经历", "社团和组织经历", "志愿经历", "志愿者经历", "志愿服务",
        "志愿服务经历", "activities", "campus experience", "campus activities",
        "volunteer experience", "volunteering",
    ),
    "education": (
        "教育经历", "教育背景", "学历信息", "教育", "education", "education background",
        "academic background",
    ),
    "skills": (
        "专业技能", "职业技能", "核心能力", "技能", "技能清单", "技术能力", "技术栈",
        "工具", "语言能力", "语言水平", "编程语言", "开发工具", "语言", "skills",
        "technical skills", "languages", "language skills",
    ),
    "credentials": (
        "证书", "资格证书", "资质", "认证", "认证资质", "资质证书和执照",
        "认证和执照", "证书与资质", "职业资格", "执业资格", "执照",
        "certifications", "licenses", "certifications and licenses",
    ),
    "awards": (
        "荣誉奖项", "荣誉与奖项", "奖项和荣誉", "获奖经历", "获奖情况", "奖项",
        "奖学金", "awards", "honors", "awards and honors", "honors and awards",
    ),
    "publications": (
        "论文", "论文发表", "论文成果", "论文期刊", "发表论文", "学术成果", "出版物",
        "著作", "专利", "专利成果", "publications", "papers", "patents",
    ),
    "training": (
        "培训经历", "培训与进修", "进修经历", "职业发展", "住院医师规范化培训",
        "training", "professional development",
    ),
    "teaching": (
        "教学经历", "授课经历", "培养经历", "teaching", "teaching experience",
    ),
    "additional": (
        "兴趣爱好", "个人爱好", "推荐信", "推荐人", "参考资料", "其他信息", "补充信息",
        "interests", "hobbies", "references", "additional information",
    ),
}

_DOCUMENT_TITLES = {
    "个人简历", "简历", "resume", "curriculumvitae", "cv",
}

_ENUMERATION_PREFIX_RE = re.compile(
    r"^\s*(?:(?:第\s*)?(?:[1-9]|[1-9]\d|[一二三四五六七八九十百]+)\s*[.．、:：)）]|"
    r"[（(]\s*(?:[1-9]|[1-9]\d|[一二三四五六七八九十百]+)\s*[)）])\s*"
)
_MARKDOWN_PREFIX_RE = re.compile(r"^\s*#{1,6}\s*")
_TRAILING_LAYOUT_ORDINAL_RE = re.compile(
    r"(?:\s*[-_–—]\s*|\s+)(?:[ivxlcdm]{1,8}|\d{1,3})\s*$",
    re.IGNORECASE,
)
_NORMALIZE_RE = re.compile(r"[\s:：|｜/\\【】\[\]()（）*#._\-—–]+")


def normalize_section_heading(value: str) -> str:
    """Normalize formatting around a heading without inspecting its industry."""

    text = _MARKDOWN_PREFIX_RE.sub("", str(value or "").strip())
    text = _ENUMERATION_PREFIX_RE.sub("", text)
    # Repeated generic sections are often rendered as ``Work Experience - II``
    # or ``工作经历 2``.  The trailing ordinal is layout, not semantic
    # content.  Requiring a visible separator/space avoids changing ordinary
    # words that merely end in a Roman-numeral letter.
    text = _TRAILING_LAYOUT_ORDINAL_RE.sub("", text)
    return _NORMALIZE_RE.sub("", text).casefold()


_NORMALIZED_TO_SECTION = {
    normalize_section_heading(alias): section
    for section, aliases in SECTION_ALIASES.items()
    for alias in aliases
}


def section_type(value: str) -> str:
    return _NORMALIZED_TO_SECTION.get(normalize_section_heading(value), "other")


def is_section_heading(value: str) -> bool:
    return section_type(value) != "other"


def is_document_title(value: str) -> bool:
    """Return whether a line is only a generic document title.

    A title carries no candidate biography fact.  Keeping this structural
    invariant outside the model prevents a schema fallback from rendering
    ``个人简历`` or ``CV`` as a long-tail resume item.
    """

    return normalize_section_heading(value) in _DOCUMENT_TITLES


__all__ = [
    "SECTION_ALIASES", "is_document_title", "is_section_heading",
    "normalize_section_heading", "section_type",
]
