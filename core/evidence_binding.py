"""Deterministic evidence tracing for final V2 resume content.

Bindings are internal audit metadata.  They never appear in the rendered
resume, and JD blocks are intentionally excluded from candidate facts.
"""
from __future__ import annotations

import re
import unicodedata

from source_adapter import candidate_blocks
from v2_schemas import CanonicalResume, EvidenceBinding, SourceBlock, SourceBundle


_SOURCE_HEADINGS = {
    "个人总结", "个人简介", "职业概述", "自我评价", "教育经历", "教育背景", "学历信息",
    "工作经历", "实习经历", "任职经历", "职业经历", "科研经历", "研究经历", "实验室经历",
    "项目经历", "项目经验", "课程项目", "个人项目", "开源项目", "校园经历", "社团经历",
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
        quote=quote,
        claim=str(value).strip(),
        mode=mode,
        similarity=round(similarity, 4),
    )


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

    block, similarity, _ = _best_block(value, blocks)
    return [block] if block is not None and similarity >= minimum else []


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
) -> dict[str, set[int]]:
    """Find records whose identity fields resolve to different source records."""

    block_by_id = {block.block_id: block for block in blocks}
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
            explicit_hints: set[str] = set()
            for field in fields:
                value = str(getattr(record, field, "") or "").strip()
                if not value:
                    continue
                matching_blocks = _strong_matching_blocks(value, blocks, minimum=0.65)
                field_scopes: set[str] = set()
                for source_block in matching_blocks:
                    if source_block.section_hint:
                        explicit_hints.add(source_block.section_hint)
                    scope = _record_scope_key(source_block, source, section)
                    if scope:
                        field_scopes.add(scope)
                if field_scopes:
                    scope_options.append(field_scopes)
            if hasattr(record, "bullets"):
                for bullet_index, value in enumerate(record.bullets):
                    binding = _bind(
                        f"{section}[{index}].bullets[{bullet_index}]",
                        str(value),
                        blocks,
                        minimum=0.30,
                    )
                    source_block = block_by_id.get(binding.block_id) if binding else None
                    if source_block and source_block.section_hint:
                        explicit_hints.add(source_block.section_hint)
                    if source_block is not None:
                        scope = _record_scope_key(source_block, source, section)
                        if scope:
                            scope_options.append({scope})
            # A repeated value such as "本科" legitimately matches several
            # education records. The record is coherent when at least one
            # source scope can satisfy every scoped identity field; binding a
            # repeated short value to the first global match must not delete a
            # later valid record.
            if len(scope_options) > 1 and not set.intersection(*scope_options):
                incoherent[section].add(index)
            elif explicit_hints and explicit_hints.issubset(_NON_RECORD_SECTION_HINTS):
                # A publication/certificate/training line may not be duplicated
                # as a fabricated work, project, or research record.
                incoherent[section].add(index)
    return incoherent


def bind_resume_evidence(resume: CanonicalResume, source: SourceBundle) -> list[EvidenceBinding]:
    """Bind final fields and bullets to Resume/Query blocks, never JD blocks."""

    blocks = candidate_blocks(source)
    bindings: list[EvidenceBinding] = []

    def add(path: str, value: str, minimum: float = 0.22) -> None:
        binding = _bind(path, value, blocks, minimum=minimum)
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
) -> tuple[CanonicalResume, list[EvidenceBinding], list[str]]:
    """Remove final candidate claims that cannot bind to Resume/Query evidence.

    ``meta.target_role`` is intentionally exempt because it is a target
    direction and may come from JD.  Summary is rebuilt deterministically after
    this gate, so it is not treated as an independent candidate claim.
    """

    gated = resume.model_copy(deep=True)
    initial = bind_resume_evidence(gated, source)
    bound_paths = {binding.path for binding in initial}
    eligible_blocks = candidate_blocks(source)
    incoherent_records = _find_incoherent_records(gated, source, eligible_blocks)
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
                binding = _bind(path, bullet, eligible_blocks, minimum=0.30)
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

    bindings = bind_resume_evidence(gated, source)
    return gated, bindings, removed


_SOURCE_FIELD_LABEL = re.compile(
    r"^(?:姓名|电话|手机|邮箱|学校|院校|学历|学位|专业|公司|单位|岗位|职位|"
    r"项目名称|项目角色|技能|证书|奖项|任职时间|起止时间|个人总结|自我评价)\s*[:：]\s*"
)
_FACT_SPLIT = re.compile(r"(?:[。；;]+|[|｜]+|(?<=[^\s])[,，、](?=[^\s]))")
_STRONG_FACT_ANCHOR = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?:%|万|亿|w|k|人|次|个|条|元|年|月|日)?|"
    r"[A-Za-z][A-Za-z0-9+.#/_-]{1,}",
    re.IGNORECASE,
)


def source_fact_units(source: SourceBundle) -> list[dict[str, str]]:
    """Split candidate source into auditable facts, including OCR-long lines."""

    result: list[dict[str, str]] = []
    for block in candidate_blocks(source):
        compact = re.sub(r"[\s:：|｜/\\【】\[\]()（）]+", "", block.text)
        if not compact or compact in _SOURCE_HEADINGS or len(compact) < 2:
            continue
        stripped = _SOURCE_FIELD_LABEL.sub("", block.text.strip())
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
                "text": display,
                "match_text": part,
            })
    return result


def _unit_is_represented(unit: dict[str, str], claims: list[str]) -> bool:
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
    return False


def measure_source_coverage(
    source: SourceBundle,
    bindings: list[EvidenceBinding],
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
        legacy_covered.add(binding.block_id)
        if binding.claim.strip():
            claims_by_block.setdefault(binding.block_id, []).append(binding.claim)
    missing: list[str] = []
    for unit in units:
        block_id = unit["block_id"]
        claims = claims_by_block.get(block_id, [])
        # Bindings serialized before claim tracing was introduced remain
        # backward-compatible at block granularity.
        represented = _unit_is_represented(unit, claims) if claims else block_id in legacy_covered
        if not represented:
            missing.append(unit["unit_id"])
    ratio = (len(units) - len(missing)) / len(units)
    return round(ratio, 4), missing
