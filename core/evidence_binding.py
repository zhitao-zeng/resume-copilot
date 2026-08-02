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
        score = len(target & candidate) / max(1, min(len(target), len(candidate)))
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
        mode=mode,
        similarity=round(similarity, 4),
    )


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
            scopes: set[str] = set()
            explicit_hints: set[str] = set()
            for field in fields:
                value = str(getattr(record, field, "") or "").strip()
                if not value:
                    continue
                binding = _bind(f"{section}[{index}].{field}", value, blocks, minimum=0.65)
                if binding is None:
                    continue
                source_block = block_by_id.get(binding.block_id)
                if source_block is None:
                    continue
                if source_block.section_hint:
                    explicit_hints.add(source_block.section_hint)
                scope = _record_scope_key(source_block, source, section)
                if scope:
                    scopes.add(scope)
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
            if len(scopes) > 1:
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


def measure_source_coverage(
    source: SourceBundle,
    bindings: list[EvidenceBinding],
) -> tuple[float, list[str]]:
    """Measure source-line coverage in the generated resume.

    This is the reverse of hallucination checking: it detects factual source
    lines that disappeared entirely. Headings are excluded because they are
    structure rather than candidate facts.
    """

    content_blocks: list[SourceBlock] = []
    for block in candidate_blocks(source):
        compact = re.sub(r"[\s:：|｜/\\【】\[\]()（）]+", "", block.text)
        if not compact or compact in _SOURCE_HEADINGS:
            continue
        if len(compact) < 3:
            continue
        content_blocks.append(block)
    if not content_blocks:
        return 1.0, []
    covered = {binding.block_id for binding in bindings}
    missing = [block.block_id for block in content_blocks if block.block_id not in covered]
    ratio = (len(content_blocks) - len(missing)) / len(content_blocks)
    return round(ratio, 4), missing
