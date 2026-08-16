"""Adapt a frozen V3 claim graph to the established editable renderer schema."""
from __future__ import annotations

from collections import OrderedDict
import re
from typing import Any

from .contracts import FactGraph, FrozenResume, RealizedClaim


_FRAMEWORK_SECTIONS = [
    {"key": "contact", "title": "基本信息", "fields": ["姓名", "电话", "邮箱", "所在地"]},
    {"key": "summary", "title": "个人总结", "fields": ["真实背景", "核心优势", "求职方向"]},
    {"key": "experience", "title": "工作/实习经历", "fields": ["组织", "岗位", "时间", "职责", "方法", "交付物", "结果"]},
    {"key": "projects", "title": "项目经历", "fields": ["项目名称", "角色", "时间", "行动", "成果"]},
    {"key": "education", "title": "教育经历", "fields": ["学校", "学历", "专业", "时间"]},
    {"key": "skills", "title": "技能与资质", "fields": ["技能", "工具", "证书", "语言能力"]},
]

_RECORD_FIELD_MAPS: dict[str, dict[str, str]] = {
    "experience": {
        "organization": "company", "role": "role", "period": "period",
    },
    "projects": {
        "organization": "company", "role": "role", "period": "period",
        "title": "name",
    },
    # The editable renderer uses company/role for research and the canonical
    # bridge later converts them to institution/topic.
    "research": {
        "organization": "company", "role": "role", "period": "period",
        "title": "role",
    },
    "activities": {
        "organization": "company", "role": "role", "period": "period",
        "title": "role",
    },
    "education": {
        "organization": "school", "school": "school", "role": "degree",
        "degree": "degree", "major": "major", "period": "period",
        "title": "school",
    },
}

_RECORD_PUBLIC_FIELDS: dict[str, frozenset[str]] = {
    "experience": frozenset({"company", "role", "period", "bullets"}),
    "projects": frozenset({"name", "company", "role", "period", "bullets"}),
    "research": frozenset({"company", "role", "period", "bullets"}),
    "activities": frozenset({"company", "role", "period", "bullets"}),
    "education": frozenset({"school", "degree", "major", "period"}),
}


def _framework(target_role: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if target_role:
        meta["target_role"] = target_role
    return {
        "meta": meta,
        "framework": {
            "mode": "empty_profile",
            "target_role": target_role,
            "notice": "以下内容均为待填写结构，不代表候选人已有事实。",
            "sections": _FRAMEWORK_SECTIONS,
        },
    }


def _claim_value(claim: RealizedClaim) -> str:
    value = str(claim.text or "").strip()
    # Markdown/list markers are source presentation, not candidate facts.
    # Removing only a balanced whole-value wrapper keeps every lexical token
    # source-backed while avoiding public roles such as ``**机械工长**`` that
    # strict consumers interpret as a different title.
    value = re.sub(r"^(?:[•·]\s*|-\s+|\*\s+)", "", value).strip()
    wrappers = (("**", "**"), ("__", "__"), ("`", "`"))
    changed = True
    while changed and value:
        changed = False
        for left, right in wrappers:
            if value.startswith(left) and value.endswith(right) and len(value) > len(left) + len(right):
                value = value[len(left):-len(right)].strip()
                changed = True
                break
    return value


def _record_bucket(
    buckets: OrderedDict[str, dict[str, Any]],
    claim: RealizedClaim,
    index: int,
) -> dict[str, Any]:
    key = claim.record_id or claim.group_id or f"unscoped:{claim.section}:{index}"
    return buckets.setdefault(key, {"_record_id": claim.record_id or "", "_group_id": claim.group_id or ""})


def _append_unique(container: list[str], value: str) -> None:
    if value and value not in container:
        container.append(value)


def _summary_value_is_represented(value: str, candidate: str) -> bool:
    """Return whether a longer summary claim already contains ``value``.

    Semantic compilation often receives both profile attributes and an
    author-written summary carrying the same role/tenure/domain.  Keeping all
    of them separated by spaces creates an accidental new sentence (and even
    cross-field numeric tokens).  This exact lexical containment removes no
    unique fact and remains independent of profession.
    """

    needle = value.strip().casefold()
    haystack = candidate.strip().casefold()
    if not needle or needle == haystack or len(needle) >= len(haystack):
        return False
    if re.fullmatch(r"[A-Za-z0-9 .+/#-]+", needle):
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])", haystack) is not None
    return len(needle) >= 2 and needle in haystack


def _compact_summary_values(values: list[str]) -> list[str]:
    unique = list(dict.fromkeys(value for value in values if value))
    return [
        value
        for index, value in enumerate(unique)
        if not any(
            _summary_value_is_represented(value, other)
            for other_index, other in enumerate(unique)
            if other_index != index
        )
    ]


def _record_section(
    claims: list[RealizedClaim],
    *,
    section: str,
    overflow: list[str] | None = None,
) -> list[dict[str, Any]]:
    buckets: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for index, claim in enumerate(claims):
        record = _record_bucket(buckets, claim, index)
        value = _claim_value(claim)
        if not value:
            continue
        field = claim.field
        key = _RECORD_FIELD_MAPS.get(section, _RECORD_FIELD_MAPS["experience"]).get(field)
        if key:
            if not record.get(key):
                record[key] = value
            elif record[key] != value:
                # The public renderer contract has no free-form education
                # bullet field.  Preserve additional exact education facts in
                # a canonical long-tail section instead of emitting the old
                # ``highlights`` extension that strict evaluators reject.
                if section == "education" and overflow is not None:
                    _append_unique(overflow, value)
                else:
                    record.setdefault("bullets", [])
                    _append_unique(record["bullets"], value)
            continue
        if section == "education" and overflow is not None:
            _append_unique(overflow, value)
        else:
            record.setdefault("bullets", [])
            _append_unique(record["bullets"], value)
    result: list[dict[str, Any]] = []
    allowed_fields = _RECORD_PUBLIC_FIELDS.get(
        section, _RECORD_PUBLIC_FIELDS["experience"],
    )
    for record in buckets.values():
        clean = {
            key: value
            for key, value in record.items()
            if key in allowed_fields and value not in (None, "", [])
        }
        if clean:
            result.append(clean)
    return result


def frozen_to_resume_data(
    frozen: FrozenResume,
    graph: FactGraph,
    *,
    target_role: str = "",
) -> dict[str, Any]:
    """Convert frozen claims without adding or rewriting candidate facts."""

    if frozen.skeleton:
        return _framework(target_role)
    data: dict[str, Any] = {"meta": {}}
    if target_role:
        # Target role is user intent and is isolated from factual claims.
        data["meta"]["target_role"] = target_role

    contact_claims = frozen.sections.get("contact", [])
    contact_extras: list[str] = []
    for claim in contact_claims:
        value = _claim_value(claim)
        if not value:
            continue
        if claim.field in {"name", "phone", "email", "location"}:
            target_key = "expected_city" if claim.field == "location" else claim.field
            if not data["meta"].get(target_key):
                data["meta"][target_key] = value
        else:
            _append_unique(contact_extras, value)
    if contact_extras:
        # The established CanonicalResume meta contract is closed.  Unknown
        # but source-backed values must stay visible without inventing a meta
        # field that downstream evaluators/renderers cannot validate.
        data.setdefault("additional_sections", {})["补充信息"] = contact_extras

    summaries = _compact_summary_values([
        _claim_value(claim)
        for claim in frozen.sections.get("summary", [])
        if _claim_value(claim)
    ])
    if summaries:
        data["summary"] = "；".join(summaries)

    section_targets = {
        "experience": "experience",
        "projects": "projects",
        "research": "research",
        "activities": "campus_experience",
        "education": "education",
    }
    education_overflow: list[str] = []
    for section, target in section_targets.items():
        claims = frozen.sections.get(section, [])
        records = _record_section(
            claims,
            section=section,
            overflow=education_overflow if section == "education" else None,
        )
        if records:
            data[target] = records

    skills = [_claim_value(claim) for claim in frozen.sections.get("skills", []) if _claim_value(claim)]
    if skills:
        data["skills"] = {"others": list(dict.fromkeys(skills))}

    flat_targets = {
        "credentials": "certifications",
        "awards": "awards",
        "publications": "publications",
        "training": "training",
        "teaching": "teaching",
    }
    for section, target in flat_targets.items():
        values = [_claim_value(claim) for claim in frozen.sections.get(section, []) if _claim_value(claim)]
        if values:
            data[target] = list(dict.fromkeys(values))

    additional: dict[str, list[str]] = {}
    for section in ("additional", "other"):
        claims = frozen.sections.get(section, [])
        values = [_claim_value(claim) for claim in claims if _claim_value(claim)]
        if not values:
            continue
        title = "补充经历" if section == "additional" else "其他经历"
        additional[title] = list(dict.fromkeys(values))
    if additional:
        data.setdefault("additional_sections", {}).update(additional)
    if education_overflow:
        data.setdefault("additional_sections", {})["教育补充信息"] = education_overflow
    return data


__all__ = ["frozen_to_resume_data"]
