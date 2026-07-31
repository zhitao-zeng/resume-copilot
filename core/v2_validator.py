"""Basic deterministic validator for V2 pipeline.

Structural cleanup + deterministic backfill (school name). No LLM calls.
"""
from __future__ import annotations

import re

from v2_schemas import CanonicalResume, Education, Project

# Matches Chinese institution names like 北京邮电大学 / 上海交通大学 / 某某学院.
# "学校/中学" excluded as suffixes — too generic (匹配到"没有学校"这类普通词).
# Common prefixes (毕业于/就读于/在/来自) consumed but excluded from capture;
# capture chars cannot be 在/于/从 (prevents running-text verbs from bleeding in).
_SCHOOL_RE = re.compile(r"(?:毕业于|就读于|就读|毕业|在|来自)?((?:(?!在|于|从)[一-龥]){2,8}(?:大学|学院))")


def _is_empty_edu(edu: Education) -> bool:
    return not any([edu.school, edu.degree, edu.major])


def _backfill_school_names(resume: CanonicalResume, source_text: str) -> int:
    """Fill empty education.school from summary/source when degree/major exist.

    Deterministic repair for Composer misses on prose-style CVs (school name
    mentioned in running text but not mapped to the education record).
    """
    if not resume.education:
        return 0
    haystack = " ".join([resume.summary or "", source_text or ""])
    schools = _SCHOOL_RE.findall(haystack)
    if not schools:
        return 0
    # Prefer the school that also appears in the resume's own summary (LLM
    # usually writes it there even when it forgets the education field).
    summary_schools = _SCHOOL_RE.findall(resume.summary or "")
    ordered = summary_schools + [s for s in schools if s not in summary_schools]
    filled = 0
    for edu in resume.education:
        if not edu.school and (edu.degree or edu.major) and ordered:
            edu.school = ordered[0]
            filled += 1
    return filled


def validate_resume(resume: CanonicalResume, source_text: str = "") -> CanonicalResume:
    # Remove empty education records
    resume.education = [e for e in resume.education if not _is_empty_edu(e)]

    # Deterministic backfill: Composer may leave school empty on prose CVs
    _backfill_school_names(resume, source_text)

    # Deduplicate projects by name
    seen: set[str] = set()
    deduped: list[Project] = []
    for p in resume.projects:
        key = p.name.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(p)
    resume.projects = deduped

    # Phone format (basic)
    if resume.meta.phone:
        phones = re.findall(r"1[3-9]\d[\s-]?\d{4}[\s-]?\d{4}", resume.meta.phone)
        if phones:
            resume.meta.phone = phones[0].replace(" ", "").replace("-", "")

    return resume
