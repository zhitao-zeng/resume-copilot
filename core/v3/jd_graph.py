"""Requirement graph extraction.  JD text is never a candidate fact source."""
from __future__ import annotations

import re

from .contracts import JobRequirement, RequirementGraph, SourceSpan


def build_requirement_graph(text: str, *, source_id: str = "jd") -> RequirementGraph:
    rows = []
    cursor = 0
    for index, raw in enumerate((text or "").splitlines()):
        value = raw.strip()
        start = text.find(raw, cursor)
        start = max(0, start)
        cursor = start + len(raw) + 1
        if not value:
            continue
        # Pure section headers ("职位概述：", "主要职责：") are document
        # structure, never requirements — they must not surface as match or
        # gap items anywhere downstream.
        if re.fullmatch(r"[^：:\n]{1,15}[：:]\s*", value):
            continue
        lowered = value.casefold()
        if any(token in lowered for token in ("要求", "职责", "负责", "requirement", "responsibilit", "qualification")):
            req_type = "responsibility" if any(token in lowered for token in ("负责", "职责", "responsibilit")) else "qualification"
        else:
            req_type = "skill" if re.search(r"[A-Za-z][A-Za-z0-9+#.-]{1,}|技能|能力|经验", value) else "other"
        leading = len(raw) - len(raw.lstrip())
        exact_start = start + leading
        rows.append(JobRequirement(
            requirement_id=f"{source_id}:requirement:{index}",
            text=value,
            requirement_type=req_type,  # type: ignore[arg-type]
            priority=2 if req_type in {"skill", "qualification"} else 1,
            source_span=SourceSpan(source_id=source_id, char_start=exact_start, char_end=exact_start + len(value)),
        ))
    keywords: list[str] = []
    for req in rows:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{1,}|[\u4e00-\u9fff]{2,8}", req.text):
            if token not in keywords:
                keywords.append(token)
    return RequirementGraph(source_id=source_id if text else None, requirements=rows, keywords=keywords)


__all__ = ["build_requirement_graph"]
