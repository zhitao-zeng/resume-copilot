"""Coverage-constrained assembly from the source fact ledger.

The legacy pipeline lets the free-form composer select content and attempts to
recover omissions afterwards.  This module makes the opposite contract
explicit: every eligible fact receives a route first, source-backed structure
is merged without deleting existing claims, and only then may wording be
optimized.  All additions remain verbatim or come from an already grounded
deterministic scaffold.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from evidence_binding import bind_resume_evidence, measure_source_coverage
from source_adapter import _is_section_heading, _looks_like_record_body
from v2_schemas import (
    CanonicalResume,
    EvidenceBinding,
    FactRoute,
    FactUnit,
    SkillItem,
    SourceBlock,
    SourceBundle,
)


_RECORD_SECTIONS = {"experience", "research", "activities", "projects"}
_SCALAR_SECTIONS = {
    "awards", "publications", "patents", "certifications", "training", "teaching",
}
_ADDITIONAL_TITLES = {
    "hobbies": "兴趣爱好",
    "coursework": "相关课程",
    "products": "产品与解决方案",
    "highlights": "经历亮点",
    "references": "推荐信息",
    "target": "求职目标",
    "profile": "个人概况",
}
_INTERNAL_SECTION = re.compile(r"^(?:待整理(?:的)?原始(?:信息|经历)|教育经历补充|补充信息)$")
_BULLET_PREFIX = re.compile(r"^(?:[-*•·▪◦]\s*|\d{1,3}(?:[.、)])\s*)")
_PLACEHOLDER_TOKEN = re.compile(r"\[[^\]\n]{1,40}\]|<[^>\n]{1,40}>|\{\{[^}\n]{1,60}\}\}")
_PLACEHOLDER_FRAGMENT = re.compile(
    r"\[[^\]\n]{1,48}\]|<[^>\n]{1,48}>|\{\{[^}\n]{1,80}\}\}|"
    r"\[(?:姓名|姓氏|电话|手机|邮箱|地址|城市|州|国家|公司|学校|大学|组织|奖项|"
    r"linkedin[_\s-]*档案|github[_\s-]*链接)(?=$|[\s,，、:：|｜/\\.()（）•·_-])|"
    r"【(?:姓名|姓氏|电话|手机|邮箱|地址|城市|州|国家|公司|学校|大学|组织|奖项)"
    r"(?=$|[\s,，、:：|｜/\\.()（）•·_-])",
    re.IGNORECASE,
)
_PLACEHOLDER_WORD = re.compile(
    r"^(?:姓名|姓氏|电话|手机|邮箱|地址|城市|州|国家|邮政编码|公司|学校|大学|"
    r"组织|奖项|许可证|linkedin档案|github链接|apt#?|公寓|n/?a|x{2,})$",
    re.IGNORECASE,
)
_EMPTY_FIELD_LABEL_REMAINDER = re.compile(
    r"^(?:公司|组织|机构|学校|大学|职位|岗位|城市|州|国家)\s*[:：]?\s*"
    r"(?:和|与|及|and)?$",
    re.IGNORECASE,
)
_REPLY_ONLY = re.compile(
    r"^(?:请|帮我|麻烦|不要|不得|禁止|避免|完整保留|只能使用|根据.+优化|"
    r"推荐信可按需提供|参考资料如需.+提供)$",
    re.IGNORECASE,
)
_TARGET_FACT = re.compile(
    r"(?:职业目标|求职目标|目标岗位|期望职位|professional\s+direction|"
    r"正在寻求|希望获得|寻求.+职位)",
    re.IGNORECASE,
)
_WORK_DURATION = re.compile(
    r"\d+(?:\.\d+)?\s*(?:年|个月|月)(?:工作|从业|实习)?(?:经验|经历)|"
    r"years?\s+of\s+experience",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"1[3-9]\d(?:[\s-]?\d){8}")
_RECORD_PATH = re.compile(r"^(experience|research|activities|projects)\[(\d+)\]")
_RECORD_IDENTITY_PATH = re.compile(
    r"^(experience|research|activities|projects)\[(\d+)\]\."
    r"(organization|role|period|institution|topic|name)$"
)
_RECORD_CONTEXT_LABEL = re.compile(
    r"^(?:行业|业务领域|业务范围)\s*[:：]",
    re.IGNORECASE,
)
_LAYOUT_ORDINAL_PREFIX = re.compile(
    r"^(?:[ivxlcdm]{1,7}|\d{1,3})[\s:：.、\-—_]*"
    r"(?=[\u4e00-\u9fff])",
    re.IGNORECASE,
)
_LAYOUT_ORDINAL_ONLY = re.compile(
    r"^(?:[ivxlcdm]{1,8})[\s:：.、\-—_]*$",
    re.IGNORECASE,
)
_MIN_COMPILE_CONFIDENCE = 0.70


@dataclass
class FactCompilationReport:
    before_coverage: float = 1.0
    after_coverage: float = 1.0
    routed_facts: int = 0
    resume_facts: int = 0
    reply_only_facts: int = 0
    rejected_facts: int = 0
    merged_records: int = 0
    filled_fields: int = 0
    appended_values: int = 0
    appended_bullets: int = 0
    unresolved_fact_ids: list[str] = field(default_factory=list)
    added_paths: list[str] = field(default_factory=list)


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def _identity_normalize(value: object) -> str:
    """Normalize presentation-only identity variants, never infer an alias."""

    text = _normalize(value)
    return re.sub(r"(?:与|和|及|and)", "", text, flags=re.IGNORECASE)


def _period_signature(value: object) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    dates = [
        "-".join(part for part in match if part)
        for match in re.findall(
            r"((?:19|20)\d{2})(?:[./年-](\d{1,2}))?(?:月)?",
            text,
        )
    ]
    if re.search(r"(?:至今|现在|开始|在职|present|current)", text):
        dates.append("open")
    return tuple(dates)


def _clean_source_value(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = _BULLET_PREFIX.sub("", text)
    return text.strip(" \t。；;")


def _fact_value_for_block(
    block: SourceBlock,
    facts: Iterable[FactUnit],
) -> str:
    """Return readable source wording without re-introducing field labels.

    ``FactUnit.verbatim_text`` deliberately excludes labels such as
    ``Career Level:`` while preserving exact source spans.  Reusing the whole
    block here made the compiler's own direct additions fail the atomic audit
    because the label was not a candidate fact.  A normal prose block remains
    untouched; labeled or segmented blocks are reconstructed only from their
    eligible fact values.
    """

    values = list(dict.fromkeys(
        _clean_source_value(fact.verbatim_text)
        for fact in facts
        if fact.fact_eligible and _clean_source_value(fact.verbatim_text)
    ))
    block_value = _clean_source_value(block.text)
    if not values:
        return block_value
    if len(values) == 1:
        fact_value = values[0]
        # Exact prose and bullet markers retain the original sentence.  A
        # larger block around one fact is normally a field label or layout
        # prefix and should not be published as candidate content.
        return block_value if _normalize(block_value) == _normalize(fact_value) else fact_value
    # Multiple units came from one physical sentence.  The original sentence
    # is the most readable lossless representation unless it contains a label
    # outside every fact span.
    all_fact_text = "".join(_normalize(value) for value in values)
    block_text = _normalize(block_value)
    if all_fact_text and all(
        _normalize(value) in block_text for value in values
    ) and len(block_text) <= len(all_fact_text) + 6:
        return block_value
    return "；".join(values)


def _placeholder_only(value: str) -> bool:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not text:
        return True
    without_tokens = _PLACEHOLDER_TOKEN.sub(" ", text)
    residuals = [
        item for item in re.split(r"[\s,，、:：|｜/\\.()（）#•·_-]+", without_tokens)
        if item and not _PLACEHOLDER_WORD.fullmatch(item)
    ]
    return bool(_PLACEHOLDER_TOKEN.search(text) and not residuals)


def _contains_placeholder(value: object) -> bool:
    return bool(_PLACEHOLDER_FRAGMENT.search(str(value or "")))


def _strip_placeholder_fragment(value: object) -> str:
    """Remove template tokens without inventing replacements.

    Mixed source rows may still contain useful literal facts, for example
    ``获得2020年[奖项]实验室管理卓越奖``.  We retain the literal remainder;
    placeholder-only identities become empty fields.  This is deliberately a
    presentation cleanup, never a value imputation step.
    """

    text = str(value or "").strip()
    if not text or not _contains_placeholder(text):
        return text
    cleaned = _PLACEHOLDER_FRAGMENT.sub(" ", text)
    cleaned = re.sub(r"\s+([,，、:：;；。])", r"\1", cleaned)
    cleaned = re.sub(r"([,，、:：;；])(?:\s*[,，、:：;；])+", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", cleaned)
    cleaned = cleaned.strip(" \t,，、:：;；|｜/\\-—_•·")
    cleaned = re.sub(
        r"^(?:在|于)\s*(?=(?:担任|任职|负责|参与|开展|执行|提供|协助|主导))",
        "",
        cleaned,
    )
    if _EMPTY_FIELD_LABEL_REMAINDER.fullmatch(cleaned):
        return ""
    return cleaned.strip()


def sanitize_resume_placeholders(
    resume: CanonicalResume,
) -> tuple[CanonicalResume, list[str]]:
    """Clear non-factual template values from a canonical resume copy.

    Summary sentences containing a token are removed as a whole because their
    generated connective text can become ungrammatical after token deletion.
    Structured fields and list items are repaired locally so any literal
    source fact surrounding a token is retained.
    """

    cleaned = resume.model_copy(deep=True)
    changed_paths: list[str] = []

    for field_name in (
        "name", "phone", "email", "target_role", "work_experience",
    ):
        value = str(getattr(cleaned.meta, field_name, "") or "")
        repaired = _strip_placeholder_fragment(value)
        if repaired != value:
            setattr(cleaned.meta, field_name, repaired)
            changed_paths.append(f"meta.{field_name}")

    summary_parts = [
        part.strip() for part in re.split(r"[。；;！？!?\r\n]+", cleaned.summary)
        if part.strip()
    ]
    kept_summary = [part for part in summary_parts if not _contains_placeholder(part)]
    if len(kept_summary) != len(summary_parts):
        changed_paths.append("summary")
        cleaned.summary = "。".join(kept_summary) + ("。" if kept_summary else "")

    section_fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, field_names in section_fields.items():
        records = getattr(cleaned, section)
        for record_index, record in enumerate(records):
            for field_name in field_names:
                value = str(getattr(record, field_name, "") or "")
                repaired = _strip_placeholder_fragment(value)
                if repaired != value:
                    setattr(record, field_name, repaired)
                    changed_paths.append(f"{section}[{record_index}].{field_name}")
            if hasattr(record, "bullets"):
                repaired_bullets: list[str] = []
                for bullet_index, bullet in enumerate(record.bullets):
                    repaired = _strip_placeholder_fragment(bullet)
                    if repaired != bullet:
                        changed_paths.append(
                            f"{section}[{record_index}].bullets[{bullet_index}]"
                        )
                    if repaired and repaired not in repaired_bullets:
                        repaired_bullets.append(repaired)
                record.bullets = repaired_bullets

    repaired_skills: list[SkillItem] = []
    for item_index, item in enumerate(cleaned.skills.items):
        repaired = _strip_placeholder_fragment(item.name)
        if repaired != item.name:
            changed_paths.append(f"skills.items[{item_index}].name")
        if repaired and all(_normalize(existing.name) != _normalize(repaired) for existing in repaired_skills):
            repaired_skills.append(item.model_copy(update={"name": repaired}))
    cleaned.skills.items = repaired_skills

    for section in _SCALAR_SECTIONS:
        repaired_values: list[str] = []
        for item_index, value in enumerate(getattr(cleaned, section)):
            repaired = _strip_placeholder_fragment(value)
            if repaired != value:
                changed_paths.append(f"{section}[{item_index}]")
            if repaired and repaired not in repaired_values:
                repaired_values.append(repaired)
        setattr(cleaned, section, repaired_values)

    repaired_additional: dict[str, list[str]] = {}
    for title, values in cleaned.additional_sections.items():
        if _INTERNAL_SECTION.fullmatch(str(title).strip()):
            changed_paths.extend(
                f"additional_sections.{title}[{item_index}]"
                for item_index, _value in enumerate(values)
            )
            continue
        repaired_values = []
        for item_index, value in enumerate(values):
            repaired = _strip_placeholder_fragment(value)
            if repaired != value:
                changed_paths.append(f"additional_sections.{title}[{item_index}]")
            if repaired and repaired not in repaired_values:
                repaired_values.append(repaired)
        if repaired_values:
            repaired_additional[title] = repaired_values
    cleaned.additional_sections = repaired_additional
    _drop_empty_records_after_placeholder_cleanup(cleaned)
    return cleaned, list(dict.fromkeys(changed_paths))


def _drop_empty_records_after_placeholder_cleanup(resume: CanonicalResume) -> None:
    fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, field_names in fields.items():
        section_records = list(getattr(resume, section))
        identity_fields = tuple(
            field_name for field_name in field_names if field_name != "period"
        )
        primary_identity_field = {
            "experience": "role",
            "research": "topic",
            "activities": "role",
            "projects": "name",
        }.get(section, "")
        consumed_layout_records: set[int] = set()
        if primary_identity_field:
            for record_index, record in enumerate(section_records[:-1]):
                bullets = [
                    str(value or "").strip()
                    for value in getattr(record, "bullets", [])
                    if str(value or "").strip()
                ]
                has_identity = any(
                    str(getattr(record, field_name, "") or "").strip()
                    for field_name in identity_fields
                )
                has_period = bool(
                    str(getattr(record, "period", "") or "").strip()
                )
                if has_identity or has_period or len(bullets) != 1:
                    continue
                without_ordinal = _LAYOUT_ORDINAL_PREFIX.sub(
                    "", bullets[0], count=1,
                ).strip()
                if (
                    not without_ordinal
                    or without_ordinal == bullets[0]
                    or len(without_ordinal) > 60
                    or _looks_like_record_body(without_ordinal)
                ):
                    continue
                next_record = section_records[record_index + 1]
                if str(
                    getattr(next_record, primary_identity_field, "") or ""
                ).strip():
                    continue
                next_has_structure = bool(
                    str(getattr(next_record, "period", "") or "").strip()
                    or any(
                        str(getattr(next_record, field_name, "") or "").strip()
                        for field_name in identity_fields
                    )
                    or getattr(next_record, "bullets", [])
                )
                if not next_has_structure:
                    continue
                # The ordinal fragment and the following structured row are
                # adjacent pieces of one normalized source record. Move the
                # exact suffix into the missing identity field; no text is
                # generated and ownership stays with the adjacent record.
                setattr(next_record, primary_identity_field, without_ordinal)
                consumed_layout_records.add(record_index)

        known_identities = {
            _normalize(getattr(record, field_name, ""))
            for record in section_records
            for field_name in identity_fields
            if _normalize(getattr(record, field_name, ""))
        }
        records = []
        for record_index, record in enumerate(section_records):
            if record_index in consumed_layout_records:
                continue
            identity_values = {
                _normalize(getattr(record, field_name, ""))
                for field_name in identity_fields
                if _normalize(getattr(record, field_name, ""))
            }
            if hasattr(record, "bullets"):
                record.bullets = [
                    str(value or "").strip()
                    for value in record.bullets
                    if str(value or "").strip()
                    and not _LAYOUT_ORDINAL_ONLY.fullmatch(
                        str(value or "").strip()
                    )
                    and _normalize(value) not in identity_values
                ]
            values = [str(getattr(record, field_name, "") or "") for field_name in field_names]
            values.extend(str(value or "") for value in getattr(record, "bullets", []))
            if any(value.strip() for value in values):
                bullets = [
                    str(value or "").strip()
                    for value in getattr(record, "bullets", [])
                    if str(value or "").strip()
                ]
                has_identity = any(
                    str(getattr(record, field_name, "") or "").strip()
                    for field_name in identity_fields
                )
                has_period = bool(str(getattr(record, "period", "") or "").strip())
                # Plain-text normalization can split ``工作经历 - IV`` into a
                # heading plus ``- IV``.  The deterministic parser may then
                # fuse that ordinal with the following role and create an
                # anonymous one-bullet record (``IV区域销售主管``) alongside
                # the real record. Drop it only when the suffix exactly
                # duplicates another structured identity in the same section.
                if (
                    section in _RECORD_SECTIONS
                    and not has_identity
                    and not has_period
                    and len(bullets) == 1
                ):
                    without_ordinal = _LAYOUT_ORDINAL_PREFIX.sub(
                        "", bullets[0], count=1,
                    ).strip()
                    if (
                        without_ordinal != bullets[0]
                        and _normalize(without_ordinal) in known_identities
                    ):
                        continue
                records.append(record)
        setattr(resume, section, records)


def _record_section_from_id(record_id: str | None) -> str:
    if not record_id:
        return ""
    parts = str(record_id).split(":")
    return parts[-2] if len(parts) >= 3 and parts[-2] in _RECORD_SECTIONS | {"education"} else ""


def route_source_facts(source: SourceBundle) -> list[FactRoute]:
    """Assign every ledger fact to resume, reply-only, or rejected output."""

    routes: list[FactRoute] = []
    for fact in source.fact_units:
        text = str(fact.verbatim_text or "").strip()
        if fact.source_type == "jd" or not fact.fact_eligible:
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="rejected",
                reason="non_candidate_source",
            ))
            continue
        if not text or _is_section_heading(text) or _placeholder_only(text):
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="rejected",
                reason="heading_or_placeholder",
            ))
            continue
        if _REPLY_ONLY.search(text):
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="reply_only",
                reason="user_instruction_or_reference_offer",
            ))
            continue

        section = str(fact.section_hint or "")
        record_section = section if section in _RECORD_SECTIONS | {"education"} else _record_section_from_id(fact.record_id)
        if record_section:
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="resume",
                destination_section=record_section,
                destination_record_id=fact.record_id,
                target_field="record",
                confidence=1.0 if fact.record_id else 0.82,
                reason="source_record_scope" if fact.record_id else "section_scope",
            ))
            continue
        if section in _SCALAR_SECTIONS:
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="resume",
                destination_section=section,
                target_field="item",
                reason="typed_scalar_section",
            ))
            continue
        if section == "skills":
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="resume",
                destination_section="skills",
                target_field="items",
                reason="typed_skill_section",
            ))
            continue
        if section == "summary":
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="resume",
                destination_section="summary",
                target_field="summary",
                reason="typed_summary_section",
            ))
            continue
        if section in _ADDITIONAL_TITLES:
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="resume",
                destination_section=f"additional_sections.{_ADDITIONAL_TITLES[section]}",
                target_field="item",
                reason="typed_long_tail_section",
            ))
            continue

        dimensions = set(fact.dimensions or [fact.fact_type])
        if "contact" in dimensions:
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="resume",
                destination_section="meta",
                target_field="contact",
                confidence=0.95,
                reason="contact_morphology",
            ))
        elif _TARGET_FACT.search(text):
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="resume",
                destination_section="additional_sections.求职目标",
                target_field="item",
                confidence=0.90,
                reason="explicit_candidate_target",
            ))
        elif _WORK_DURATION.search(text):
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="resume",
                destination_section="meta",
                target_field="work_experience",
                confidence=0.90,
                reason="explicit_work_duration",
            ))
        elif "credential" in dimensions:
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="resume",
                destination_section="certifications",
                target_field="item",
                confidence=0.86,
                reason="credential_morphology",
            ))
        elif "skill" in dimensions:
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="resume",
                destination_section="skills",
                target_field="items",
                confidence=0.84,
                reason="skill_morphology",
            ))
        elif dimensions & {"action", "method", "deliverable", "result"}:
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="resume",
                destination_section="additional_sections.经历亮点",
                target_field="item",
                confidence=0.72,
                reason="unowned_professional_statement",
            ))
        elif "education" in dimensions:
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="resume",
                destination_section="additional_sections.教育背景",
                target_field="item",
                confidence=0.72,
                reason="unowned_education_statement",
            ))
        else:
            routes.append(FactRoute(
                fact_id=fact.fact_id,
                status="resume",
                destination_section="additional_sections.其他信息",
                target_field="item",
                confidence=0.62,
                reason="eligible_untyped_fact",
            ))
    return routes


def _claims(values: Iterable[str]) -> list[str]:
    return [str(value).strip() for value in values if str(value or "").strip()]


def _is_represented(value: str, existing: Iterable[str]) -> bool:
    target = _normalize(value)
    if not target:
        return True
    target_bigrams = {
        target[index:index + 2]
        for index in range(max(0, len(target) - 1))
    }
    for candidate in existing:
        normalized = _normalize(candidate)
        if target in normalized:
            return True
        candidate_bigrams = {
            normalized[index:index + 2]
            for index in range(max(0, len(normalized) - 1))
        }
        # Coverage is directional: a short output fragment must not claim a
        # longer source fact merely because the fragment occurs inside it.
        # A fluent rewrite may still represent the source when it preserves
        # nearly all of the source's lexical content.
        if target_bigrams and (
            len(target_bigrams & candidate_bigrams) / len(target_bigrams)
        ) >= 0.78:
            return True
    return False


def _record_values(record: object, section: str) -> list[str]:
    fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }[section]
    values = [str(getattr(record, field_name, "") or "") for field_name in fields]
    values.extend(str(value or "") for value in getattr(record, "bullets", []))
    return _claims(values)


def _record_conflicts(left: object, right: object, section: str) -> bool:
    identity_fields = {
        "education": ("school", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "period"),
    }[section]
    if section == "experience":
        left_period = _period_signature(getattr(left, "period", ""))
        right_period = _period_signature(getattr(right, "period", ""))
        left_role = _identity_normalize(getattr(left, "role", ""))
        right_role = _identity_normalize(getattr(right, "role", ""))
        if (
            left_period
            and left_period == right_period
            and left_role
            and left_role == right_role
        ):
            # The same explicitly dated role is a stronger identity than an
            # organization spelling/translation variant.  This joins a compact
            # career-history row with its detailed record without claiming
            # that the two organization strings are aliases in general.
            return False
    for field_name in identity_fields:
        left_raw = getattr(left, field_name, "")
        right_raw = getattr(right, field_name, "")
        if field_name == "period":
            left_period = _period_signature(left_raw)
            right_period = _period_signature(right_raw)
            if left_period and right_period and left_period != right_period:
                return True
            continue
        left_value = _identity_normalize(left_raw)
        right_value = _identity_normalize(right_raw)
        if left_value and right_value and left_value != right_value:
            return True
    return False


def _record_match_score(left: object, right: object, section: str) -> int:
    identity_fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }[section]
    score = 0
    for field_name in identity_fields:
        if field_name == "period":
            left_period = _period_signature(getattr(left, field_name, ""))
            right_period = _period_signature(getattr(right, field_name, ""))
            if left_period and right_period and left_period == right_period:
                score += 5
            continue
        left_value = _identity_normalize(getattr(left, field_name, ""))
        right_value = _identity_normalize(getattr(right, field_name, ""))
        if not left_value or not right_value:
            continue
        if left_value == right_value:
            score += 5 if field_name in {"period", "organization", "school", "name", "institution"} else 3
        elif left_value in right_value or right_value in left_value:
            score += 2
    left_claims = _record_values(left, section)
    for value in _record_values(right, section):
        if _is_represented(value, left_claims):
            score += 1
    return score


def _safe_scaffold_record(record: object, section: str) -> bool:
    values = _record_values(record, section)
    if not values or any(_placeholder_only(value) for value in values if _PLACEHOLDER_TOKEN.search(value)):
        # A placeholder field does not invalidate real sibling fields; it is
        # removed by grounding.  Reject only records whose every value is a
        # placeholder or empty.
        if not any(value and not _placeholder_only(value) for value in values):
            return False
    bullets = _claims(getattr(record, "bullets", []))
    if section == "education":
        return any(_normalize(getattr(record, field_name, "")) for field_name in ("school", "degree", "major"))
    if section == "projects":
        return bool(_normalize(getattr(record, "name", "")) or bullets)
    identities = {
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
    }[section]
    return bool(bullets and any(_normalize(getattr(record, field_name, "")) for field_name in identities))


def _merge_grounded_scaffold(
    resume: CanonicalResume,
    scaffold: CanonicalResume | None,
    report: FactCompilationReport,
    allowed_sections: set[str],
) -> CanonicalResume:
    if scaffold is None:
        return resume.model_copy(deep=True)
    scaffold, _ = sanitize_resume_placeholders(scaffold)
    merged = resume.model_copy(deep=True)
    for field_name in ("name", "phone", "email", "target_role", "work_experience"):
        value = str(getattr(scaffold.meta, field_name, "") or "").strip()
        if value and not getattr(merged.meta, field_name) and not _placeholder_only(value):
            setattr(merged.meta, field_name, value)
            report.filled_fields += 1
            report.added_paths.append(f"meta.{field_name}")

    for section in ("education", "experience", "research", "activities", "projects"):
        if section not in allowed_sections:
            continue
        destination = getattr(merged, section)
        for source_record in getattr(scaffold, section):
            if not _safe_scaffold_record(source_record, section):
                continue
            ranked = sorted(
                (
                    (_record_match_score(candidate, source_record, section), index)
                    for index, candidate in enumerate(destination)
                    if not _record_conflicts(candidate, source_record, section)
                ),
                reverse=True,
            )
            threshold = 3
            if section == "projects":
                source_claims = _record_values(source_record, section)
                if any(
                    _is_represented(value, source_claims)
                    for candidate in destination
                    for value in _record_values(candidate, section)
                ):
                    # Unnamed project groups are legal when the source only
                    # supplies project duties.  Exact claim overlap identifies
                    # the same group more reliably than an invented name.
                    threshold = 1
            match_index = (
                ranked[0][1] if ranked and ranked[0][0] >= threshold else None
            )
            if match_index is None:
                destination.append(source_record.model_copy(deep=True))
                report.merged_records += 1
                report.added_paths.append(f"{section}[{len(destination) - 1}]")
                continue
            target = destination[match_index]
            field_names = {
                "education": ("school", "degree", "major", "period"),
                "experience": ("organization", "role", "period"),
                "research": ("institution", "topic", "period"),
                "activities": ("organization", "role", "period"),
                "projects": ("name", "organization", "role", "period"),
            }[section]
            for field_name in field_names:
                value = str(getattr(source_record, field_name, "") or "").strip()
                if value and not getattr(target, field_name) and not _placeholder_only(value):
                    setattr(target, field_name, value)
                    report.filled_fields += 1
                    report.added_paths.append(f"{section}[{match_index}].{field_name}")
            if hasattr(target, "bullets"):
                existing = list(target.bullets)
                for bullet in source_record.bullets:
                    value = _clean_source_value(bullet)
                    if value and not _is_represented(value, existing):
                        target.bullets.append(value)
                        existing.append(value)
                        report.appended_bullets += 1
                        report.added_paths.append(
                            f"{section}[{match_index}].bullets[{len(target.bullets) - 1}]"
                        )

    if "skills" in allowed_sections:
        existing_skills = [item.name for item in merged.skills.items]
        for item in scaffold.skills.items:
            value = _clean_source_value(item.name)
            if value and not _placeholder_only(value) and not _is_represented(value, existing_skills):
                merged.skills.items.append(item.model_copy(deep=True))
                existing_skills.append(value)
                report.appended_values += 1
                report.added_paths.append(f"skills.items[{len(merged.skills.items) - 1}].name")

    for section in _SCALAR_SECTIONS:
        if section not in allowed_sections:
            continue
        destination = getattr(merged, section)
        for raw in getattr(scaffold, section):
            value = _clean_source_value(raw)
            if value and not _placeholder_only(value) and not _is_represented(value, destination):
                destination.append(value)
                report.appended_values += 1
                report.added_paths.append(f"{section}[{len(destination) - 1}]")

    for title, values in scaffold.additional_sections.items():
        if _INTERNAL_SECTION.fullmatch(str(title).strip()):
            continue
        destination = merged.additional_sections.setdefault(str(title).strip(), [])
        for raw in values:
            value = _clean_source_value(raw)
            if value and not _placeholder_only(value) and not _is_represented(value, destination):
                destination.append(value)
                report.appended_values += 1
                report.added_paths.append(
                    f"additional_sections.{title}[{len(destination) - 1}]"
                )
    return merged


def _owner_map(
    source: SourceBundle,
    bindings: list[EvidenceBinding],
    *,
    require_identity: bool = False,
) -> dict[tuple[str, str], int]:
    facts = {fact.fact_id: fact for fact in source.fact_units}
    blocks = {block.block_id: block for block in source.blocks}
    scores: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    identity_fields: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for binding in bindings:
        match = _RECORD_PATH.match(binding.path)
        if not match:
            continue
        section, index_text = match.groups()
        identity_match = _RECORD_IDENTITY_PATH.fullmatch(binding.path)
        if require_identity and identity_match is None:
            # A duty bullet may resemble another record's source text.  It is
            # useful evidence for factuality, but must never establish record
            # ownership for coverage recovery.
            continue
        record_ids = {
            facts[fact_id].record_id
            for fact_id in binding.fact_ids
            if fact_id in facts and facts[fact_id].record_id
        }
        for block_id in binding.block_ids or [binding.block_id]:
            block = blocks.get(block_id)
            if block is not None and block.record_id:
                record_ids.add(block.record_id)
        for record_id in record_ids:
            if _record_section_from_id(record_id) == section:
                scores[(section, int(index_text))][record_id] += 1
                if identity_match is not None:
                    identity_fields[(section, int(index_text), record_id)].add(
                        identity_match.group(3)
                    )

    if require_identity:
        # Require two independent structured identity fields.  In addition,
        # make ownership one-to-one in both directions: one output row cannot
        # absorb two OCR/source rows merely because they share a company or
        # date, and one source row cannot feed two output rows.
        qualified = {
            (section, index, record_id): len(fields)
            for (section, index, record_id), fields in identity_fields.items()
            if len(fields) >= 2
        }
        source_candidates: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        output_candidates: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
        for (section, index, record_id), score in qualified.items():
            source_candidates[(section, record_id)].append((score, index))
            output_candidates[(section, index)].append((score, record_id))

        owners: dict[tuple[str, str], int] = {}
        for (section, record_id), ranked_outputs in source_candidates.items():
            ranked_outputs.sort(reverse=True)
            if len(ranked_outputs) > 1 and ranked_outputs[0][0] == ranked_outputs[1][0]:
                continue
            score, index = ranked_outputs[0]
            ranked_sources = sorted(output_candidates[(section, index)], reverse=True)
            if len(ranked_sources) > 1 and ranked_sources[0][0] == ranked_sources[1][0]:
                continue
            if ranked_sources[0] != (score, record_id):
                continue
            owners[(section, record_id)] = index
        return owners

    candidates: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for (section, index), counter in scores.items():
        for record_id, score in counter.items():
            candidates[(section, record_id)].append((score, index))
    owners: dict[tuple[str, str], int] = {}
    for key, ranked in candidates.items():
        ranked.sort(reverse=True)
        if len(ranked) == 1 or ranked[0][0] > ranked[1][0]:
            owners[key] = ranked[0][1]
    return owners


def _split_skill_values(value: str) -> list[str]:
    cleaned = _clean_source_value(value)
    cleaned = re.sub(
        r"^(?:技能|专业技能|职业技能|技术栈|工具|语言能力|语言)\s*[:：]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    parts = [
        item.strip(" \t-•·")
        for item in re.split(r"[、，,；;|｜•]+", cleaned)
        if item.strip(" \t-•·")
    ]
    return parts if len(parts) > 1 and all(len(item) <= 80 for item in parts) else [cleaned]


def _append_additional(
    resume: CanonicalResume,
    title: str,
    value: str,
    report: FactCompilationReport,
) -> None:
    destination = resume.additional_sections.setdefault(title, [])
    if value and not _is_represented(value, destination):
        destination.append(value)
        report.appended_values += 1
        report.added_paths.append(f"additional_sections.{title}[{len(destination) - 1}]")


def compile_fact_coverage(
    resume: CanonicalResume,
    source: SourceBundle,
    *,
    scaffold: CanonicalResume | None = None,
    merge_scaffold: bool = True,
    allowed_destinations: frozenset[str] | None = None,
    require_identity_owner: bool = False,
) -> tuple[CanonicalResume, FactCompilationReport, list[FactRoute]]:
    """Merge grounded structure and close every safely routable fact gap."""

    routes = route_source_facts(source)
    report = FactCompilationReport(
        routed_facts=len(routes),
        resume_facts=sum(route.status == "resume" for route in routes),
        reply_only_facts=sum(route.status == "reply_only" for route in routes),
        rejected_facts=sum(route.status == "rejected" for route in routes),
    )
    initial_bindings = bind_resume_evidence(resume, source)
    report.before_coverage, _ = measure_source_coverage(
        source, initial_bindings, allow_distributed=True,
    )
    allowed_sections = {
        route.destination_section
        for route in routes
        if (
            route.status == "resume"
            and route.destination_section
            and (
                allowed_destinations is None
                or route.destination_section in allowed_destinations
            )
        )
    }
    compiled = (
        _merge_grounded_scaffold(resume, scaffold, report, allowed_sections)
        if merge_scaffold
        else resume.model_copy(deep=True)
    )
    bindings = bind_resume_evidence(compiled, source)
    _, missing_ids = measure_source_coverage(source, bindings, allow_distributed=True)
    missing_blocks = {unit_id.split("#u", 1)[0] for unit_id in missing_ids}
    if not missing_blocks:
        report.after_coverage = 1.0
        return compiled, report, routes

    route_by_fact = {route.fact_id: route for route in routes}
    block_by_id = {block.block_id: block for block in source.blocks}
    facts_by_block: dict[str, list[FactUnit]] = defaultdict(list)
    for fact in source.fact_units:
        facts_by_block[fact.block_id].append(fact)
    owners = _owner_map(
        source,
        bindings,
        require_identity=require_identity_owner,
    )

    for block_id in sorted(
        missing_blocks,
        key=lambda value: next(
            (
                span.char_start
                for block in source.blocks if block.block_id == value
                for span in block.source_spans
            ),
            10**12,
        ),
    ):
        block = block_by_id.get(block_id)
        if block is None or block.source_type == "jd" or _is_section_heading(block.text):
            continue
        facts = [fact for fact in facts_by_block.get(block_id, []) if fact.fact_eligible]
        fact_routes = [route_by_fact[fact.fact_id] for fact in facts if fact.fact_id in route_by_fact]
        resume_routes = [
            route for route in fact_routes
            if (
                route.status == "resume"
                and (
                    allowed_destinations is None
                    or route.destination_section in allowed_destinations
                )
            )
        ]
        if not resume_routes:
            continue
        route = max(resume_routes, key=lambda item: item.confidence)
        if route.confidence < _MIN_COMPILE_CONFIDENCE:
            # Keep low-confidence eligible facts visible in the route audit,
            # but do not turn an arbitrary OCR fragment into a public resume
            # section merely to increase a metric.
            continue
        value = _fact_value_for_block(block, facts)
        if not value or _placeholder_only(value) or _REPLY_ONLY.search(value):
            continue

        destination = route.destination_section
        if destination in _RECORD_SECTIONS:
            record_id = route.destination_record_id or block.record_id
            index = owners.get((destination, str(record_id or "")))
            records = getattr(compiled, destination)
            if index is None and len(records) == 1 and record_id:
                source_record_ids = {
                    fact.record_id
                    for fact in source.fact_units
                    if fact.fact_eligible and _record_section_from_id(fact.record_id) == destination and fact.record_id
                }
                if source_record_ids == {record_id}:
                    index = 0
            if index is None or index >= len(records):
                continue
            dimensions = set().union(*(set(fact.dimensions) for fact in facts))
            if not (
                _looks_like_record_body(value)
                or dimensions & {"action", "method", "deliverable", "result"}
                or _RECORD_CONTEXT_LABEL.match(block.text.strip())
            ):
                continue
            record = records[index]
            if not _is_represented(value, record.bullets):
                record.bullets.append(value)
                report.appended_bullets += 1
                report.added_paths.append(
                    f"{destination}[{index}].bullets[{len(record.bullets) - 1}]"
                )
            continue
        if destination == "education":
            # Education identity belongs in structured fields.  The grounded
            # scaffold handles it; do not create a prose-only fake school row.
            continue
        if destination == "skills":
            existing = [item.name for item in compiled.skills.items]
            for skill in _split_skill_values(value):
                if skill and not _placeholder_only(skill) and not _is_represented(skill, existing):
                    compiled.skills.items.append(SkillItem(name=skill, category="other"))
                    existing.append(skill)
                    report.appended_values += 1
                    report.added_paths.append(
                        f"skills.items[{len(compiled.skills.items) - 1}].name"
                    )
            continue
        if destination in _SCALAR_SECTIONS:
            values = getattr(compiled, destination)
            if not _is_represented(value, values):
                values.append(value)
                report.appended_values += 1
                report.added_paths.append(f"{destination}[{len(values) - 1}]")
            continue
        if destination == "summary":
            if not _is_represented(value, [compiled.summary]):
                candidate = "。".join(item for item in (compiled.summary.strip("。"), value) if item)
                if len(candidate) <= 360:
                    compiled.summary = candidate + ("。" if candidate else "")
                    report.appended_values += 1
                    report.added_paths.append("summary")
                else:
                    _append_additional(compiled, "个人优势", value, report)
            continue
        if destination == "meta":
            if route.target_field == "work_experience" and not compiled.meta.work_experience:
                match = _WORK_DURATION.search(value)
                if match:
                    compiled.meta.work_experience = match.group(0)
                    report.filled_fields += 1
                    report.added_paths.append("meta.work_experience")
            elif route.target_field == "contact":
                if not compiled.meta.email and (match := _EMAIL.search(value)):
                    compiled.meta.email = match.group(0)
                    report.filled_fields += 1
                    report.added_paths.append("meta.email")
                if not compiled.meta.phone and (match := _PHONE.search(value)):
                    compiled.meta.phone = match.group(0)
                    report.filled_fields += 1
                    report.added_paths.append("meta.phone")
            continue
        if destination.startswith("additional_sections."):
            _append_additional(compiled, destination.split(".", 1)[1], value, report)

    final_bindings = bind_resume_evidence(compiled, source)
    report.after_coverage, unresolved = measure_source_coverage(
        source, final_bindings, allow_distributed=True,
    )
    report.unresolved_fact_ids = list(unresolved)
    return compiled, report, routes
