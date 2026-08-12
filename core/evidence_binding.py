"""Deterministic evidence tracing for final V2 resume content.

Bindings are internal audit metadata.  They never appear in the rendered
resume, and JD blocks are intentionally excluded from candidate facts.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from source_adapter import _looks_like_record_body, candidate_blocks
from v2_schemas import CanonicalResume, EvidenceBinding, SourceBlock, SourceBundle


_SOURCE_HEADINGS = {
    "个人信息", "基本信息", "联系方式", "个人总结", "个人简介", "职业概述", "自我评价",
    "教育经历", "教育背景", "学历信息",
    "工作经历", "实习经历", "任职经历", "职业经历", "科研经历", "研究经历", "实验室经历",
    "项目经历", "项目经验", "课程项目", "个人项目", "开源项目", "校园经历", "社团经历", "组织经历",
    "志愿经历", "社会实践", "学生工作", "专业技能", "技能清单", "技术栈", "工具", "语言能力",
    "荣誉奖项", "荣誉与奖项", "获奖经历", "奖项", "论文", "论文发表", "论文成果", "学术成果",
    "出版物", "专利", "专利成果", "证书", "证书与资质", "职业资格", "执业资格", "执照",
    "培训经历", "进修经历", "教学经历", "授课经历", "培养经历",
    "学术会议", "会议经历", "专业会员", "专业组织", "专著", "作品集", "作品经历",
}

_NON_RECORD_SECTION_HINTS = {
    "summary", "skills", "awards", "publications", "patents",
    "certifications", "training", "teaching",
}
_SCHOOL_IDENTITY = re.compile(r"(?:大学|学院|学校|研究院)")


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)


def _bigrams(value: str) -> set[str]:
    normalized = _normalize(value)
    return {normalized[index:index + 2] for index in range(max(0, len(normalized) - 1))}


def _date_signature(value: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", str(value or ""))
    pattern = re.compile(
        r"(?<!\d)(?:(?P<year1>(?:19|20)\d{2})\s*[-./年]\s*(?P<month1>0?[1-9]|1[0-2])\s*月?"
        r"|(?P<month2>0?[1-9]|1[0-2])\s*[-./]\s*(?P<year2>(?:19|20)\d{2}))(?!\d)"
    )
    signature: list[str] = []
    for match in pattern.finditer(text):
        year = match.group("year1") or match.group("year2")
        month = match.group("month1") or match.group("month2")
        signature.append(f"{year}{int(month):02d}")
    return tuple(signature)


def _grouped_date_matches(
    value: str,
    blocks: list[SourceBlock],
) -> list[SourceBlock]:
    """Match a period whose endpoints were split across one OCR record."""

    target = _date_signature(value)
    if not target:
        return []
    grouped: dict[tuple[str, str, str], list[SourceBlock]] = {}
    for block in blocks:
        if not block.record_id:
            continue
        grouped.setdefault(
            (block.source_type, str(block.section_hint or ""), block.record_id),
            [],
        ).append(block)
    matches: list[SourceBlock] = []
    for group in grouped.values():
        source_signature = _date_signature("\n".join(item.text for item in group))
        if len(source_signature) < len(target):
            continue
        if not any(
            source_signature[index:index + len(target)] == target
            for index in range(len(source_signature) - len(target) + 1)
        ):
            continue
        matches.extend(item for item in group if _date_signature(item.text))
    return matches


def _best_block(value: str, blocks: list[SourceBlock]) -> tuple[SourceBlock | None, float, str]:
    normalized_value = _normalize(value)
    if not normalized_value:
        return None, 0.0, "rewritten"
    for block in blocks:
        if normalized_value in _normalize(block.text):
            mode = "direct" if str(value).casefold() in block.text.casefold() else "normalized"
            return block, 1.0, mode

    date_signature = _date_signature(value)
    if date_signature:
        for block in blocks:
            block_signature = _date_signature(block.text)
            if len(block_signature) >= len(date_signature) and any(
                block_signature[index:index + len(date_signature)] == date_signature
                for index in range(len(block_signature) - len(date_signature) + 1)
            ):
                return block, 1.0, "normalized"
        grouped_matches = _grouped_date_matches(value, blocks)
        if grouped_matches:
            return grouped_matches[0], 1.0, "normalized"

    target = _bigrams(value)
    best_block = None
    best_score = 0.0
    for block in blocks:
        candidate = _bigrams(block.text)
        if not candidate:
            continue
        # Truth checking is asymmetric: every part of the generated value must
        # be supported by the source.  Dividing by the shorter side allowed a
        # short shared prefix to validate a much longer fabricated claim.
        score = len(target & candidate) / max(1, len(target))
        if score > best_score:
            best_block, best_score = block, score
    return best_block, best_score, "rewritten"


def _bind(path: str, value: str, blocks: list[SourceBlock], *, minimum: float = 0.22) -> EvidenceBinding | None:
    block, similarity, mode = _best_block(value, blocks)
    if block is None or similarity < minimum:
        return None
    quote = block.text.strip()
    if len(quote) > 240:
        normalized_value = _normalize(value)
        position = _normalize(quote).find(normalized_value)
        if position >= 0:
            # Normalized offsets are approximate; retaining the beginning is
            # safer than claiming an exact character span after punctuation
            # removal.
            quote = quote[:240]
        else:
            quote = quote[:240]
    return EvidenceBinding(
        path=path,
        block_id=block.block_id,
        block_ids=[block.block_id],
        quote=quote,
        claim=str(value).strip(),
        mode=mode,
        similarity=round(similarity, 4),
    )


def _bind_with_provenance(
    path: str,
    value: str,
    blocks: list[SourceBlock],
    *,
    minimum: float,
    trusted_rewrites: Mapping[str, str] | None = None,
) -> EvidenceBinding | None:
    """Bind a final rewrite through its already-verified source wording.

    A rewrite is trusted only when the optimizer supplied a stable path and
    the original wording still binds to candidate evidence.  JD-only text can
    therefore never become evidence through this escape hatch.
    """

    source_value = (
        str(trusted_rewrites.get(path, "") or "").strip()
        if trusted_rewrites else ""
    )
    if source_value:
        # Record-level optimization can combine several already-grounded
        # bullets. Newlines are an internal provenance separator rather than
        # user-facing content. Every component must independently bind; a
        # partial match must never make the whole rewrite trusted.
        source_parts = [
            part.strip()
            for part in re.split(r"[\r\n]+", source_value)
            if part.strip()
        ]
        source_bindings = [
            _bind(path, part, blocks, minimum=minimum)
            for part in source_parts
        ]
        if source_parts and all(item is not None for item in source_bindings):
            resolved = [item for item in source_bindings if item is not None]
            block_ids = list(dict.fromkeys(
                block_id
                for item in resolved
                for block_id in (item.block_ids or [item.block_id])
            ))
            primary = resolved[0]
            return primary.model_copy(update={
                "claim": str(value or "").strip(),
                "source_claim": source_value,
                "mode": "rewritten",
                "block_ids": block_ids,
                "similarity": round(min(item.similarity for item in resolved), 4),
            })
        return None
    return _bind(path, value, blocks, minimum=minimum)


def _strong_matching_blocks(
    value: str,
    blocks: list[SourceBlock],
    *,
    minimum: float,
) -> list[SourceBlock]:
    """Return every strong direct/date match, or the single best fuzzy match."""

    normalized_value = _normalize(value)
    if not normalized_value:
        return []
    direct = [
        block for block in blocks
        if normalized_value in _normalize(block.text)
    ]
    if direct:
        return direct

    date_signature = _date_signature(value)
    if date_signature:
        date_matches = []
        for block in blocks:
            block_signature = _date_signature(block.text)
            if len(block_signature) >= len(date_signature) and any(
                block_signature[index:index + len(date_signature)] == date_signature
                for index in range(len(block_signature) - len(date_signature) + 1)
            ):
                date_matches.append(block)
        if date_matches:
            return date_matches
        grouped_matches = _grouped_date_matches(value, blocks)
        if grouped_matches:
            return grouped_matches

    block, similarity, _ = _best_block(value, blocks)
    return [block] if block is not None and similarity >= minimum else []


_ROLE_FIELD_LABEL = re.compile(r"(?:岗位|职位|角色|职务)\s*[:：]")
_ROLE_IDENTITY_VERB = re.compile(r"(?:担任|任职为|任职|作为|职位为|岗位为|曾任)\s*")
_RECORD_DUTY_START = re.compile(
    r"(?:^|[，,。；;|｜])\s*(?:[-*•·▪◦]\s*)?(?:负责|参与|主导|协助|支持|配合|完成|推动|推进|"
    r"组织|设计|开发|构建|实现|制定|管理|运营|分析|研究|撰写|输出|交付|"
    r"维护|优化|搭建|建立|开展|承担|提供|跟进|协调|带领|执行)"
)
_NON_ROLE_CONTEXT = re.compile(r"(?:熟悉|擅长|掌握|具备|负责|参与|协助|完成|开展)\s*$")
_ROLE_HEADER_CONTEXT = re.compile(
    r"(?:19|20)\d{2}|(?:大学|学院|学校|医院|公司|企业|集团|研究院|实验室|中心|"
    r"部门|协会|学会|学生会|社团|委员会|事务所|律所|银行|政府|基金会|工作室|团队|基地)"
)


def _flexible_literal(value: str) -> str:
    return r"\s*".join(re.escape(char) for char in str(value or "").strip())


def _role_block_is_valid(block: SourceBlock, value: str) -> bool:
    """Require role evidence to occur in identity context, not a duty clause.

    This is a field-type gate rather than an occupation dictionary.  It works
    the same way for engineering, teaching, medicine and operations: a phrase
    under ``负责/参与/...`` may support a bullet, but not a job title.
    """

    text = str(block.text or "").strip()
    role = str(value or "").strip()
    if not text or not role:
        return False
    literal = _flexible_literal(role)
    occurrences = list(re.finditer(literal, text, re.IGNORECASE))
    if not occurrences:
        # Normalized OCR spacing may prevent positional analysis.  A clear body
        # line is never sufficient role evidence; a non-body record header can
        # still support the normalized title.
        return not _looks_like_record_body(text) and bool(
            block.section_hint in {"experience", "activities", "projects"}
            or _normalize(text) == _normalize(role)
        )

    if re.search(rf"{_ROLE_FIELD_LABEL.pattern}\s*{literal}", text, re.IGNORECASE):
        return True
    if re.search(rf"{_ROLE_IDENTITY_VERB.pattern}{literal}", text, re.IGNORECASE):
        return True

    duty = _RECORD_DUTY_START.search(text)
    body_like = _looks_like_record_body(text)
    for occurrence in occurrences:
        prefix = text[max(0, occurrence.start() - 12):occurrence.start()]
        if _NON_ROLE_CONTEXT.search(prefix):
            continue
        left = text[:occurrence.start()].rstrip()
        right = text[occurrence.end():].lstrip()
        if (
            block.section_hint in {"experience", "activities", "projects"}
            and (left.endswith(("|", "｜")) or right.startswith(("|", "｜")))
        ):
            # Delimited record headers such as ``公司｜运营专员｜日期``
            # legitimately contain titles that are also verbs in other prose.
            return True
        if body_like or duty is not None:
            # A leading duty noun followed by ``负责/参与/...`` is still body
            # prose (for example ``需求分析，负责梳理...``), not a title.  Accept
            # a title embedded in a compact body line only when the text before
            # it also contains structural identity evidence such as an employer
            # or a date.  This deliberately uses field grammar, not an industry
            # title dictionary.
            header_prefix = text[:occurrence.start()]
            if not _ROLE_HEADER_CONTEXT.search(header_prefix):
                continue
        if duty is not None and occurrence.start() >= duty.start():
            continue
        if block.section_hint in {"experience", "activities", "projects"}:
            return True
        if _normalize(text) == _normalize(role):
            return True
        # Compact unsectioned headers often combine organization, title and
        # date on one line.  Accept only when no duty clause is present.
        if duty is None and len(text) <= 100:
            return True
    return False


def _bind_role(
    path: str,
    value: str,
    blocks: list[SourceBlock],
    *,
    minimum: float = 0.65,
) -> EvidenceBinding | None:
    for block in _strong_matching_blocks(value, blocks, minimum=minimum):
        if not _role_block_is_valid(block, value):
            continue
        binding = _bind(path, value, [block], minimum=minimum)
        if binding is not None:
            return binding
    return None


def _record_scope_key(
    block: SourceBlock,
    source: SourceBundle,
    section: str,
) -> str | None:
    """Resolve a source block to the surrounding dated record.

    Resume extraction keeps one source line per block. A record header usually
    contains its period, while following lines contain bullets. Assigning every
    block to the nearest preceding dated header lets us detect impossible
    combinations such as an organization from one job and a role from the next
    job. Lines before the first dated header belong to that first record so
    layouts with organization/title above the date remain supported.
    """

    if block.record_id and block.section_hint == section:
        return block.record_id
    if block.section_hint and block.section_hint != section:
        return None

    eligible = candidate_blocks(source)
    section_blocks = [
        item for item in eligible
        if item.source_type == block.source_type and item.section_hint == section
    ]
    if not section_blocks:
        # Without structural hints, record boundaries are too ambiguous to
        # enforce safely. Field-level grounding still applies.
        return None
    anchors = [item for item in section_blocks if _date_signature(item.text)]
    if not anchors:
        return None
    order = {item.block_id: index for index, item in enumerate(source.blocks)}
    block_index = order.get(block.block_id, -1)
    preceding = [item for item in anchors if order.get(item.block_id, -1) <= block_index]
    anchor = preceding[-1] if preceding else anchors[0]
    return f"{block.source_type}:{section}:{anchor.block_id}"


def _find_incoherent_records(
    resume: CanonicalResume,
    source: SourceBundle,
    blocks: list[SourceBlock],
    trusted_rewrites: Mapping[str, str] | None = None,
    *,
    allow_reordered_record_bullets: bool = False,
) -> dict[str, set[int]]:
    """Find records whose identity fields resolve to different source records."""

    section_fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    incoherent: dict[str, set[int]] = {section: set() for section in section_fields}
    for section, fields in section_fields.items():
        for index, record in enumerate(getattr(resume, section)):
            scope_options: list[set[str]] = []
            identity_scope_options: list[set[str]] = []
            bullet_scope_options: list[set[str]] = []
            bullet_evidence_is_unsectioned = True
            explicit_hints: set[str] = set()
            for field in fields:
                value = str(getattr(record, field, "") or "").strip()
                if not value:
                    continue
                matching_blocks = _strong_matching_blocks(value, blocks, minimum=0.65)
                if field == "role":
                    matching_blocks = [
                        block for block in matching_blocks
                        if _role_block_is_valid(block, value)
                    ]
                field_scopes: set[str] = set()
                for source_block in matching_blocks:
                    if source_block.section_hint:
                        explicit_hints.add(source_block.section_hint)
                    scope = _record_scope_key(source_block, source, section)
                    if scope:
                        field_scopes.add(scope)
                if field_scopes:
                    scope_options.append(field_scopes)
                    identity_scope_options.append(field_scopes)
            if hasattr(record, "bullets"):
                for bullet_index, value in enumerate(record.bullets):
                    path = f"{section}[{index}].bullets[{bullet_index}]"
                    provenance_value = str(
                        (trusted_rewrites or {}).get(path, "") or ""
                    ).strip()
                    provenance_parts = [
                        part.strip()
                        for part in re.split(r"[\r\n]+", provenance_value)
                        if part.strip()
                    ] or [str(value)]
                    matching_blocks = []
                    for provenance_part in provenance_parts:
                        matching_blocks.extend(_strong_matching_blocks(
                            provenance_part,
                            blocks,
                            minimum=0.30,
                        ))
                    bullet_scopes: set[str] = set()
                    for source_block in matching_blocks:
                        if source_block.section_hint:
                            explicit_hints.add(source_block.section_hint)
                        scope = _record_scope_key(source_block, source, section)
                        if scope:
                            bullet_scopes.add(scope)
                    if bullet_scopes:
                        # Repeated phrases such as “进行用户测试并收集反馈”
                        # legitimately occur in multiple projects. Preserve all
                        # possible scopes here instead of binding to the first
                        # global occurrence and falsely deleting later records.
                        scope_options.append(bullet_scopes)
                        bullet_scope_options.append(bullet_scopes)
                        if any(item.section_hint for item in matching_blocks):
                            bullet_evidence_is_unsectioned = False
            # A repeated value such as "本科" legitimately matches several
            # education records. The record is coherent when at least one
            # source scope can satisfy every scoped identity field; binding a
            # repeated short value to the first global match must not delete a
            # later valid record.
            if len(scope_options) > 1 and not set.intersection(*scope_options):
                # Multi-column PDF extraction can place the sole education
                # period after unrelated columns while preserving every field
                # verbatim. When there is only one output education record and
                # one school identity in the source, field grounding is more
                # reliable than artificial line-order scopes.
                source_school_scopes = {
                    _record_scope_key(item, source, "education")
                    for item in blocks
                    if item.section_hint == "education"
                    and _SCHOOL_IDENTITY.search(item.text)
                }
                allow_single_education_reorder = (
                    section == "education"
                    and len(resume.education) == 1
                    and len({value for value in source_school_scopes if value}) <= 1
                )
                # Deterministic layout recovery may pair explicit identity rows
                # with numbered duties emitted later as an unsectioned OCR
                # column.  The identity fields must still agree on one source
                # record and every bullet must bind independently; only the
                # artificial line-order scope is relaxed.
                allow_deterministic_column_reorder = bool(
                    allow_reordered_record_bullets
                    and section in {"experience", "research", "activities", "projects"}
                    and identity_scope_options
                    and set.intersection(*identity_scope_options)
                    and bullet_scope_options
                    and bullet_evidence_is_unsectioned
                )
                if not allow_single_education_reorder and not allow_deterministic_column_reorder:
                    incoherent[section].add(index)
            elif explicit_hints and explicit_hints.issubset(_NON_RECORD_SECTION_HINTS):
                # A publication/certificate/training line may not be duplicated
                # as a fabricated work, project, or research record.
                incoherent[section].add(index)
    return incoherent


def _possible_record_scopes(
    record,
    *,
    section: str,
    source: SourceBundle,
    blocks: list[SourceBlock],
) -> set[str]:
    """Return source records that can support every populated record claim."""

    fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }[section]
    options: list[set[str]] = []
    for field in fields:
        value = str(getattr(record, field, "") or "").strip()
        if not value:
            continue
        matching = _strong_matching_blocks(value, blocks, minimum=0.65)
        if field == "role":
            matching = [item for item in matching if _role_block_is_valid(item, value)]
        scopes = {
            scope
            for item in matching
            if (scope := _record_scope_key(item, source, section))
        }
        if scopes:
            options.append(scopes)
    if hasattr(record, "bullets"):
        for bullet in record.bullets:
            scopes = {
                scope
                for item in _strong_matching_blocks(str(bullet), blocks, minimum=0.30)
                if (scope := _record_scope_key(item, source, section))
            }
            if scopes:
                options.append(scopes)
    if not options:
        return set()
    return set.intersection(*options)


def _merge_identityless_continuations(
    resume: CanonicalResume,
    source: SourceBundle,
    blocks: list[SourceBlock],
) -> None:
    """Join chunk continuations only when both rows resolve to one source record."""

    identity_fields = {
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, fields in identity_fields.items():
        merged = []
        for record in getattr(resume, section):
            if section == "activities" and merged and record.bullets and not merged[-1].bullets:
                current_scopes = _possible_record_scopes(
                    record, section=section, source=source, blocks=blocks,
                )
                previous_scopes = _possible_record_scopes(
                    merged[-1], section=section, source=source, blocks=blocks,
                )
                current_org = str(record.organization or "").strip()
                current_role = str(record.role or "").strip()
                weak_department_org = bool(
                    current_org
                    and current_role
                    and _normalize(current_org) in _normalize(current_role)
                )
                organizations_compatible = bool(
                    not merged[-1].organization
                    or not current_org
                    or _normalize(merged[-1].organization) == _normalize(current_org)
                    or weak_department_org
                )
                periods_compatible = bool(
                    not merged[-1].period
                    or not record.period
                    or _date_signature(merged[-1].period) == _date_signature(record.period)
                )
                if (
                    current_scopes & previous_scopes
                    and organizations_compatible
                    and periods_compatible
                ):
                    if not merged[-1].organization and current_org:
                        merged[-1].organization = current_org
                    if not merged[-1].role and current_role:
                        merged[-1].role = current_role
                    if not merged[-1].period and record.period:
                        merged[-1].period = record.period
                    merged[-1].bullets = list(dict.fromkeys(record.bullets))
                    continue
            has_identity = any(
                str(getattr(record, field, "") or "").strip()
                for field in fields
            )
            if not has_identity and record.bullets and merged:
                current_scopes = _possible_record_scopes(
                    record, section=section, source=source, blocks=blocks,
                )
                previous_scopes = _possible_record_scopes(
                    merged[-1], section=section, source=source, blocks=blocks,
                )
                if current_scopes & previous_scopes:
                    merged[-1].bullets = list(dict.fromkeys(
                        list(merged[-1].bullets) + list(record.bullets)
                    ))
                    continue
            merged.append(record)
        setattr(resume, section, merged)


def bind_resume_evidence(
    resume: CanonicalResume,
    source: SourceBundle,
    *,
    trusted_rewrites: Mapping[str, str] | None = None,
) -> list[EvidenceBinding]:
    """Bind final fields and bullets to Resume/Query blocks, never JD blocks."""

    blocks = candidate_blocks(source)
    bindings: list[EvidenceBinding] = []

    def add(path: str, value: str, minimum: float = 0.22) -> None:
        binding = (
            _bind_role(path, value, blocks, minimum=minimum)
            if path.endswith(".role")
            else _bind_with_provenance(
                path,
                value,
                blocks,
                minimum=minimum,
                trusted_rewrites=trusted_rewrites,
            )
        )
        if binding is not None:
            bindings.append(binding)

    for key in ("name", "phone", "email", "work_experience"):
        add(f"meta.{key}", getattr(resume.meta, key), 0.8)

    for index, sentence in enumerate(
        item.strip() for item in re.split(r"[。！？!?；;]+", resume.summary) if item.strip()
    ):
        add(f"summary[{index}]", sentence, 0.68)

    section_fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, fields in section_fields.items():
        for index, record in enumerate(getattr(resume, section)):
            for field in fields:
                add(f"{section}[{index}].{field}", getattr(record, field), 0.65)
            if hasattr(record, "bullets"):
                for bullet_index, bullet in enumerate(record.bullets):
                    add(f"{section}[{index}].bullets[{bullet_index}]", bullet)

    for index, skill in enumerate(resume.skills.items):
        add(f"skills.items[{index}].name", skill.name, 0.8)
    for index, award in enumerate(resume.awards):
        add(f"awards[{index}]", award, 0.8)
    for section in ("publications", "patents", "certifications", "training", "teaching"):
        for index, value in enumerate(getattr(resume, section)):
            add(f"{section}[{index}]", value, 0.65)
    for title, items in resume.additional_sections.items():
        for index, value in enumerate(items):
            add(f"additional_sections.{title}[{index}]", value, 0.65)
    return bindings


def enforce_resume_evidence(
    resume: CanonicalResume,
    source: SourceBundle,
    *,
    trusted_rewrites: Mapping[str, str] | None = None,
    allow_reordered_record_bullets: bool = False,
) -> tuple[CanonicalResume, list[EvidenceBinding], list[str]]:
    """Remove final candidate claims that cannot bind to Resume/Query evidence.

    ``meta.target_role`` is intentionally exempt because it is a target
    direction and may come from JD.  Summary is rebuilt deterministically after
    this gate, so it is not treated as an independent candidate claim.
    """

    gated = resume.model_copy(deep=True)
    initial = bind_resume_evidence(
        gated,
        source,
        trusted_rewrites=trusted_rewrites,
    )
    bound_paths = {binding.path for binding in initial}
    eligible_blocks = candidate_blocks(source)
    incoherent_records = _find_incoherent_records(
        gated,
        source,
        eligible_blocks,
        trusted_rewrites=trusted_rewrites,
        allow_reordered_record_bullets=allow_reordered_record_bullets,
    )
    removed: list[str] = []

    for key in ("name", "phone", "email", "work_experience"):
        path = f"meta.{key}"
        if getattr(gated.meta, key) and path not in bound_paths:
            setattr(gated.meta, key, "")
            removed.append(path)

    grounded_summary: list[str] = []
    for index, sentence in enumerate(
        item.strip() for item in re.split(r"[。！？!?；;]+", gated.summary) if item.strip()
    ):
        binding = _bind(f"summary[{index}]", sentence, eligible_blocks, minimum=0.68)
        if binding is not None and (binding.mode in {"direct", "normalized"} or binding.similarity >= 0.78):
            grounded_summary.append(sentence)
        else:
            removed.append(f"summary[{index}]")
    gated.summary = "。".join(grounded_summary) + ("。" if grounded_summary else "")

    section_fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, fields in section_fields.items():
        for index, record in enumerate(getattr(gated, section)):
            if index in incoherent_records[section]:
                removed.append(f"{section}[{index}]")
                continue
            for field in fields:
                path = f"{section}[{index}].{field}"
                if getattr(record, field) and path not in bound_paths:
                    setattr(record, field, "")
                    removed.append(path)
            if not hasattr(record, "bullets"):
                continue
            kept_bullets: list[str] = []
            for bullet_index, bullet in enumerate(record.bullets):
                path = f"{section}[{index}].bullets[{bullet_index}]"
                binding = _bind_with_provenance(
                    path,
                    bullet,
                    eligible_blocks,
                    minimum=0.30,
                    trusted_rewrites=trusted_rewrites,
                )
                if binding is None:
                    removed.append(path)
                else:
                    kept_bullets.append(bullet)
            record.bullets = kept_bullets

        if incoherent_records[section]:
            setattr(gated, section, [
                record for index, record in enumerate(getattr(gated, section))
                if index not in incoherent_records[section]
            ])

    # Before optimization there is no positional rewrite provenance to remap.
    # This is the safe point to join records split only by Composer chunking.
    if not trusted_rewrites:
        _merge_identityless_continuations(gated, source, eligible_blocks)

    kept_skills = []
    for index, skill in enumerate(gated.skills.items):
        path = f"skills.items[{index}].name"
        if path in bound_paths:
            kept_skills.append(skill)
        else:
            removed.append(path)
    gated.skills.items = kept_skills

    kept_awards: list[str] = []
    for index, award in enumerate(gated.awards):
        path = f"awards[{index}]"
        if path in bound_paths:
            kept_awards.append(award)
        else:
            removed.append(path)
    gated.awards = kept_awards

    for section in ("publications", "patents", "certifications", "training", "teaching"):
        kept_values: list[str] = []
        for index, value in enumerate(getattr(gated, section)):
            path = f"{section}[{index}]"
            if path in bound_paths:
                kept_values.append(value)
            else:
                removed.append(path)
        setattr(gated, section, kept_values)

    kept_additional: dict[str, list[str]] = {}
    for title, items in gated.additional_sections.items():
        kept_items: list[str] = []
        for index, value in enumerate(items):
            path = f"additional_sections.{title}[{index}]"
            if path in bound_paths:
                kept_items.append(value)
            else:
                removed.append(path)
        if kept_items:
            kept_additional[title] = kept_items
    gated.additional_sections = kept_additional

    bindings = bind_resume_evidence(
        gated,
        source,
        trusted_rewrites=trusted_rewrites,
    )
    return gated, bindings, removed


_SOURCE_FIELD_LABEL = re.compile(
    r"^(?:姓名|电话|手机|邮箱|学校|院校|学历|学位|专业|公司|单位|岗位|职位|"
    r"项目名称|项目角色|技能|证书|奖项|任职时间|起止时间|个人总结|自我评价)\s*[:：]\s*"
)
_COMPACT_SOURCE_FIELD_LABEL = re.compile(
    r"^(?:姓名|电话|手机|邮箱)\s*[:：]?\s*",
    re.IGNORECASE,
)
_FACT_SPLIT = re.compile(r"(?:[。；;]+|[|｜]+|(?<=[^\s])[,，、](?=[^\s]))")
_STRONG_FACT_ANCHOR = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:%|万|亿|w|k|人|次|个|条|元|年|月|日)?|"
    r"[A-Za-z][A-Za-z0-9+.#/_-]{1,}",
    re.IGNORECASE,
)
_FACT_ACTION_SIGNAL = re.compile(
    r"(?:负责|参与|主导|协助|支持|配合|推动|推进|组织|协调|带领|执行|"
    r"设计|开发|构建|实现|制定|管理|运营|分析|统计|策划|培训|处理|研究|"
    r"撰写|输出|交付|维护|优化|搭建|建立|开展|承担|提供|跟进|编制|制作|"
    r"诊断|治疗|授课|教学|复核|检索|调研)"
)
_FACT_METHOD_SIGNAL = re.compile(
    r"(?:通过|使用|采用|基于|借助|运用|利用|结合|按照|依托|围绕|"
    r"经由|以[^，。；;]{1,20}(?:方式|方法|流程|标准|规范))"
)
_FACT_DELIVERABLE_SIGNAL = re.compile(
    r"(?:输出|交付|完成|形成|上线|发布|落地|搭建|建立|制定|编制|制作|"
    r"撰写|产出|提交|复核|验证|诊断|治疗|授课|培养)"
)
_FACT_RESULT_SIGNAL = re.compile(
    r"(?:提升|提高|降低|减少|增长|缩短|节省|达到|达成|获得|获奖|录用|"
    r"成交|销售率|准确率|转化率|留存率|满意度)"
)


def _fact_dimensions(value: str) -> str:
    """Return generic evidence dimensions without an industry dictionary."""

    dimensions: list[str] = []
    for name, pattern in (
        ("action", _FACT_ACTION_SIGNAL),
        ("method", _FACT_METHOD_SIGNAL),
        ("deliverable", _FACT_DELIVERABLE_SIGNAL),
        ("result", _FACT_RESULT_SIGNAL),
        ("anchor", _STRONG_FACT_ANCHOR),
    ):
        if pattern.search(value):
            dimensions.append(name)
    return ",".join(dimensions)


def source_fact_units(source: SourceBundle) -> list[dict[str, str]]:
    """Split candidate source into auditable facts, including OCR-long lines."""

    result: list[dict[str, str]] = []
    for block in candidate_blocks(source):
        compact = re.sub(r"[\s:：|｜/\\【】\[\]()（）]+", "", block.text)
        if not compact or compact in _SOURCE_HEADINGS or len(compact) < 2:
            continue
        stripped = _SOURCE_FIELD_LABEL.sub("", block.text.strip())
        stripped = _COMPACT_SOURCE_FIELD_LABEL.sub("", stripped)
        raw_parts = [part.strip(" \t-•·▪◦") for part in _FACT_SPLIT.split(stripped)]
        parts = [part for part in raw_parts if len(_normalize(part)) >= 2]
        if not parts:
            continue
        # Preserve the original punctuation in reports for ordinary one-fact
        # lines. Multi-fact/OCR-compressed lines expose each missing clause.
        displays = [block.text.strip()] if len(parts) == 1 else parts
        for index, (part, display) in enumerate(zip(parts, displays)):
            unit_id = block.block_id if len(parts) == 1 else f"{block.block_id}#u{index}"
            result.append({
                "unit_id": unit_id,
                "block_id": block.block_id,
                "source_type": block.source_type,
                "section_hint": block.section_hint or "",
                "record_id": block.record_id or "",
                "record_body": "true" if _looks_like_record_body(block.text) else "",
                "dimensions": _fact_dimensions(part),
                "text": display,
                "match_text": part,
            })
    return result


def _unit_is_represented(
    unit: dict[str, str],
    claims: list[str],
    *,
    allow_distributed: bool = False,
) -> bool:
    source_value = _normalize(unit.get("match_text", ""))
    if not source_value:
        return True
    anchors = {item.casefold() for item in _STRONG_FACT_ANCHOR.findall(unit.get("match_text", ""))}
    source_bigrams = _bigrams(source_value)
    for claim in claims:
        claim_value = _normalize(claim)
        if not claim_value:
            continue
        claim_anchors = {item.casefold() for item in _STRONG_FACT_ANCHOR.findall(claim)}
        if anchors and not anchors.issubset(claim_anchors):
            continue
        if source_value in claim_value:
            return True
        if len(source_value) <= 4 and claim_value in source_value:
            return True
        claim_bigrams = _bigrams(claim_value)
        recall = len(source_bigrams & claim_bigrams) / max(1, len(source_bigrams))
        if recall >= 0.58:
            return True
    # Structural source lines commonly distribute one fact across multiple
    # final fields (period + organization + role, or school + degree + major).
    # Evaluate their aggregate only after every individual claim check, so a
    # heading or unrelated short value cannot mark the whole unit represented.
    if allow_distributed and len(claims) >= 2:
        aggregate = " ".join(claims)
        aggregate_value = _normalize(aggregate)
        aggregate_anchors = {
            item.casefold() for item in _STRONG_FACT_ANCHOR.findall(aggregate)
        }
        if not anchors or anchors.issubset(aggregate_anchors):
            aggregate_bigrams = _bigrams(aggregate_value)
            recall = len(source_bigrams & aggregate_bigrams) / max(1, len(source_bigrams))
            if source_value in aggregate_value or recall >= 0.58:
                return True
    return False


def measure_source_coverage(
    source: SourceBundle,
    bindings: list[EvidenceBinding],
    *,
    allow_distributed: bool = False,
) -> tuple[float, list[str]]:
    """Measure source fact-unit coverage in the generated resume.

    This is the reverse of hallucination checking. A long OCR line may contain
    several duties, methods and results; one short binding no longer marks all
    of them as represented.
    """

    units = source_fact_units(source)
    if not units:
        return 1.0, []
    claims_by_block: dict[str, list[str]] = {}
    legacy_covered: set[str] = set()
    for binding in bindings:
        linked_block_ids = binding.block_ids or [binding.block_id]
        legacy_covered.update(linked_block_ids)
        coverage_claim = str(binding.source_claim or binding.claim).strip()
        if coverage_claim:
            for block_id in linked_block_ids:
                claims_by_block.setdefault(block_id, []).append(coverage_claim)
    missing: list[str] = []
    for unit in units:
        block_id = unit["block_id"]
        claims = claims_by_block.get(block_id, [])
        # Bindings serialized before claim tracing was introduced remain
        # backward-compatible at block granularity.
        represented = (
            _unit_is_represented(
                unit,
                claims,
                allow_distributed=allow_distributed,
            )
            if claims else block_id in legacy_covered
        )
        if not represented:
            missing.append(unit["unit_id"])
    ratio = (len(units) - len(missing)) / len(units)
    return round(ratio, 4), missing
