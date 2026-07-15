"""Basic deterministic validator for V2 pipeline.

Structural cleanup only — no semantic repair, no school/lab/position rules.
"""
from __future__ import annotations

import re

from v2_schemas import CanonicalResume, Education, Project


def _is_empty_edu(edu: Education) -> bool:
    return not any([edu.school, edu.degree, edu.major])


def validate_resume(resume: CanonicalResume) -> CanonicalResume:
    # Remove empty education records
    resume.education = [e for e in resume.education if not _is_empty_edu(e)]

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
