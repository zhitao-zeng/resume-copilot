"""Deterministic evidence tracing for final V2 resume content.

Bindings are internal audit metadata.  They never appear in the rendered
resume, and JD blocks are intentionally excluded from candidate facts.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from atomic_fact_audit import atomize_claim_text, match_atomic_claim
from diagnostic_trace import trace_event
from source_adapter import _is_section_heading, _looks_like_record_body, candidate_blocks
from v2_schemas import CanonicalResume, EvidenceBinding, FactUnit, SourceBlock, SourceBundle


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
_SOURCE_DOCUMENT_TITLE = re.compile(
    r"^(?:[\u4e00-\u9fff·]{2,16}|[A-Za-z][A-Za-z .'-]{1,40})?(?:个人)?(?:简历|履历)$",
    re.IGNORECASE,
)
_SOURCE_NON_FACTUAL_COPY = re.compile(
    r"^(?:候选人)?具备清晰的问题拆解和执行闭环能力$|"
    r"^过往经历以真实岗位职责和结果为准$"
)

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
    # Prefer the exact field occurrence over an earlier document title such
    # as “张晨简历”.  Otherwise the name binds to a layout-only block and the
    # actual contact-row name is incorrectly reported as unrepresented.
    for block in blocks:
        if normalized_value == _normalize(block.text):
            mode = "direct" if str(value).casefold() == block.text.casefold() else "normalized"
            return block, 1.0, mode
    for block in blocks:
        if _SOURCE_DOCUMENT_TITLE.fullmatch(block.text.strip()):
            continue
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
        fact_ids=list(block.fact_ids),
        source_spans=list(block.source_spans),
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
            return _merge_bindings(
                path,
                value,
                resolved,
                source_claim=source_value,
                mode="rewritten",
            )
        return None
    compound = _bind_exact_compound(path, value, blocks, minimum=minimum)
    return compound or _bind(path, value, blocks, minimum=minimum)


def _merge_bindings(
    path: str,
    value: str,
    resolved: list[EvidenceBinding],
    *,
    source_claim: str,
    mode: str,
) -> EvidenceBinding:
    block_ids = list(dict.fromkeys(
        block_id
        for item in resolved
        for block_id in (item.block_ids or [item.block_id])
    ))
    fact_ids = list(dict.fromkeys(
        fact_id
        for item in resolved
        for fact_id in item.fact_ids
    ))
    source_spans = []
    seen_spans: set[tuple[str, int, int]] = set()
    for item in resolved:
        for span in item.source_spans:
            key = (span.source_id, span.char_start, span.char_end)
            if key in seen_spans:
                continue
            seen_spans.add(key)
            source_spans.append(span)
    primary = resolved[0]
    return primary.model_copy(update={
        "path": path,
        "claim": str(value or "").strip(),
        "source_claim": source_claim,
        "mode": mode,
        "block_ids": block_ids,
        "fact_ids": fact_ids,
        "source_spans": source_spans,
        "similarity": round(min(item.similarity for item in resolved), 4),
    })


def _bind_exact_compound(
    path: str,
    value: str,
    blocks: list[SourceBlock],
    *,
    minimum: float,
) -> EvidenceBinding | None:
    """Bind a claim assembled from exact clauses in one source record.

    This is the deterministic counterpart to optimizer provenance.  It is
    intentionally unavailable for fuzzy components or cross-record clauses.
    """

    parts = atomize_claim_text(value)
    if len(parts) < 2:
        return None
    resolved = [
        _bind(path, part, blocks, minimum=max(0.58, minimum))
        for part in parts
    ]
    if not all(item is not None for item in resolved):
        return None
    bindings = [item for item in resolved if item is not None]
    if any(
        item.mode not in {"direct", "normalized"} and item.similarity < 0.78
        for item in bindings
    ):
        return None
    block_by_id = {block.block_id: block for block in blocks}
    records = {
        block_by_id[block_id].record_id
        for item in bindings
        for block_id in (item.block_ids or [item.block_id])
        if block_id in block_by_id and block_by_id[block_id].record_id
    }
    if len(records) > 1:
        return None
    source_claim = "\n".join(dict.fromkeys(item.quote for item in bindings if item.quote))
    generated_anchors = {
        _normalize(item) for item in _STRONG_FACT_ANCHOR.findall(value) if _normalize(item)
    }
    source_anchors = {
        _normalize(item)
        for item in _STRONG_FACT_ANCHOR.findall(source_claim)
        if _normalize(item)
    }
    if generated_anchors and not generated_anchors.issubset(source_anchors):
        return None
    return _merge_bindings(
        path,
        value,
        bindings,
        source_claim=source_claim,
        mode="rewritten",
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
    r"(?:19|20)\d{2}|(?:大学|学院|学校|中学|小学|幼儿园|医院|公司|企业|集团|研究院|实验室|中心|"
    r"部门|协会|学会|学生会|社团|委员会|事务所|律所|银行|政府|基金会|工作室|团队|基地)"
)
_RELATION_ONLY_ROLE = re.compile(
    r"^(?:指导老师|指导教师|导师|项目导师|论文导师|推荐人|联系人)$",
    re.IGNORECASE,
)
_ORGANIZATION_FIELD_LABEL = re.compile(
    r"(?:公司|单位|组织|机构|学校|院校|所属公司)\s*[:：]"
)
_ORGANIZATION_RELATION = re.compile(
    r"(?:就职于|任职于|供职于|受雇于|在|于)\s*"
)
_NON_ORGANIZATION_VALUE = re.compile(
    r"^(?:做过?|从事|担任|参与|负责|有|具备|完成|开展)"
)
_NON_ORGANIZATION_PREFIX = re.compile(
    r"(?:做过?|从事|担任|参与|负责|有|具备|完成|开展)\s*$"
)
_ACTIVITY_ORGANIZATION_END = re.compile(
    r"(?:协会|学会|学生会|社团|委员会|志愿队|服务队|部门|部|团队|中心|"
    r"医院|学校|学院|大学)$"
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
    if _RELATION_ONLY_ROLE.fullmatch(role):
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
        if duty is not None and _ROLE_HEADER_CONTEXT.search(text[:occurrence.start()]):
            # Compact source rows may have inherited a wrong or empty section
            # hint from OCR/layout parsing.  A value before the first duty clause
            # with a date/employer in its left context is still strong structural
            # identity evidence.  Duty nouns such as ``单元测试`` occur after the
            # clause marker and remain rejected by the check above.
            return True
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


def _organization_block_is_valid(block: SourceBlock, value: str) -> bool:
    """Require organization syntax, not just a shared source substring.

    Words such as ``小学`` and ``企业软件`` can be literal parts of a trial
    lesson or domain-qualified sales title.  They become an organization only
    when the source labels/relates them as one, places them in a structured
    record header, or supplies them as a standalone organization row.
    """

    text = str(block.text or "").strip()
    organization = str(value or "").strip()
    if not text or not organization or _NON_ORGANIZATION_VALUE.search(organization):
        return False
    literal = _flexible_literal(organization)
    occurrences = list(re.finditer(literal, text, re.IGNORECASE))
    if not occurrences:
        return False
    if re.search(
        rf"{_ORGANIZATION_FIELD_LABEL.pattern}\s*{literal}",
        text,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        rf"{_ORGANIZATION_RELATION.pattern}{literal}",
        text,
        re.IGNORECASE,
    ):
        return True
    if _normalize(text) == _normalize(organization):
        return True

    duty = _RECORD_DUTY_START.search(text)
    for occurrence in occurrences:
        prefix = text[:occurrence.start()].rstrip()
        suffix = text[occurrence.end():].lstrip()
        if _NON_ORGANIZATION_PREFIX.search(prefix):
            continue
        if prefix.endswith(("|", "｜")) or suffix.startswith(("|", "｜")):
            return block.section_hint in {
                "experience", "activities", "projects", "research",
            }
        # A date-leading compact row is a structural identity record.  The
        # organization must precede the first duty clause; otherwise a company
        # mentioned in a result/bullet cannot become the current employer.
        if _date_signature(prefix) and (duty is None or occurrence.start() < duty.start()):
            return True
        if (
            block.section_hint == "activities"
            and block.record_id
            and occurrence.start() == 0
            and not _looks_like_record_body(text)
            and (
                bool(text[occurrence.end():occurrence.end() + 1].isspace())
                or bool(_ACTIVITY_ORGANIZATION_END.search(organization))
            )
        ):
            # Campus/volunteer headers are often ``组织 角色`` or a compact
            # department-title row without dates.  Restrict this relaxation to
            # the activities section so ``小学语文试讲`` in a project cannot be
            # reinterpreted as an employer.
            return True
    return False


def _bind_organization(
    path: str,
    value: str,
    blocks: list[SourceBlock],
    *,
    minimum: float = 0.65,
) -> EvidenceBinding | None:
    for block in _strong_matching_blocks(value, blocks, minimum=minimum):
        if not _organization_block_is_valid(block, value):
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
                elif field == "organization":
                    matching_blocks = [
                        block for block in matching_blocks
                        if _organization_block_is_valid(block, value)
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
        elif field == "organization":
            matching = [
                item for item in matching
                if _organization_block_is_valid(item, value)
            ]
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


def _value_record_scopes(
    value: str,
    *,
    section: str,
    source: SourceBundle,
    blocks: list[SourceBlock],
    role: bool = False,
    organization: bool = False,
    minimum: float = 0.65,
) -> set[str]:
    matching = _strong_matching_blocks(value, blocks, minimum=minimum)
    if role:
        matching = [item for item in matching if _role_block_is_valid(item, value)]
    elif organization:
        matching = [
            item for item in matching
            if _organization_block_is_valid(item, value)
        ]
    return {
        scope
        for item in matching
        if (scope := _record_scope_key(item, source, section))
    }


def _anchored_record_scope(
    record,
    *,
    section: str,
    source: SourceBundle,
    blocks: list[SourceBlock],
) -> str | None:
    """Choose one source record from identity/period evidence only.

    Bullets are intentionally excluded: one optimizer or Composer sentence can
    be attached to the wrong row, but that must not outweigh a matching project
    name, employer or period and cause deletion of the complete grounded record.
    """

    primary_fields = {
        "education": ("school",),
        "experience": ("organization",),
        "research": ("institution", "topic"),
        "activities": ("organization",),
        "projects": ("name",),
    }[section]
    secondary_fields = {
        "education": ("degree", "major"),
        "experience": ("role",),
        "research": (),
        "activities": ("role",),
        "projects": ("organization", "role"),
    }[section]
    scores: dict[str, int] = {}
    strong_scopes: set[str] = set()

    for field in primary_fields:
        value = str(getattr(record, field, "") or "").strip()
        if not value:
            continue
        scopes = _value_record_scopes(
            value,
            section=section,
            source=source,
            blocks=blocks,
            role=False,
            organization=(field == "organization"),
        )
        for scope in scopes:
            scores[scope] = scores.get(scope, 0) + 8
        if len(scopes) == 1:
            strong_scopes.update(scopes)

    period = str(getattr(record, "period", "") or "").strip()
    if period:
        scopes = _value_record_scopes(
            period,
            section=section,
            source=source,
            blocks=blocks,
            minimum=0.65,
        )
        for scope in scopes:
            scores[scope] = scores.get(scope, 0) + 7
        if len(scopes) == 1:
            strong_scopes.update(scopes)

    for field in secondary_fields:
        value = str(getattr(record, field, "") or "").strip()
        if not value:
            continue
        scopes = _value_record_scopes(
            value,
            section=section,
            source=source,
            blocks=blocks,
            role=(field == "role"),
            organization=(field == "organization"),
        )
        for scope in scopes:
            scores[scope] = scores.get(scope, 0) + 2

    if not scores or not strong_scopes:
        return None
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_scope, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0
    if best_scope not in strong_scopes or best_score < 7 or best_score == second_score:
        return None
    return best_scope


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
            else _bind_organization(path, value, blocks, minimum=minimum)
            if path.endswith(".organization")
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


def _facts_for_record_scope(
    source: SourceBundle,
    blocks: list[SourceBlock],
    *,
    section: str,
    scope: str,
) -> list[FactUnit]:
    block_by_id = {block.block_id: block for block in blocks}
    facts: list[FactUnit] = []
    for fact in source.fact_units:
        if not fact.fact_eligible or fact.source_type == "jd":
            continue
        block = block_by_id.get(fact.block_id)
        if block is None or block.section_hint != section:
            continue
        if _record_scope_key(block, source, section) == scope:
            facts.append(fact)
    return facts


def _facts_for_binding(
    source: SourceBundle,
    binding: EvidenceBinding | None,
) -> list[FactUnit]:
    if binding is None:
        return []
    fact_ids = set(binding.fact_ids)
    block_ids = set(binding.block_ids or [binding.block_id])
    return [
        fact for fact in source.fact_units
        if fact.fact_eligible
        and fact.source_type != "jd"
        and (fact.fact_id in fact_ids or fact.block_id in block_ids)
    ]


def _source_body_sentences(
    source: SourceBundle,
    blocks: list[SourceBlock],
    *,
    section: str,
    scope: str,
) -> list[str]:
    result: list[str] = []
    for block in blocks:
        if block.section_hint != section:
            continue
        if _record_scope_key(block, source, section) != scope:
            continue
        text = str(block.text or "").strip()
        duty = _RECORD_DUTY_START.search(text)
        if duty is not None:
            text = text[duty.start():].lstrip("，,。；;|｜ \t-•·▪◦")
        elif not _looks_like_record_body(text):
            continue
        for sentence in re.split(r"[。；;\r\n]+", text):
            value = sentence.strip("，,。；; \t-•·▪◦")
            if len(_normalize(value)) >= 4:
                result.append(value)
    return list(dict.fromkeys(result))


def _unique_source_fallback(value: str, candidates: list[str]) -> str:
    target = _bigrams(value)
    if not target:
        return ""
    def shared_literal_length(left: str, right: str) -> int:
        first = _normalize(left)
        second = _normalize(right)
        for width in range(min(len(first), len(second)), 3, -1):
            if any(first[index:index + width] in second for index in range(len(first) - width + 1)):
                return width
        return 0

    ranked: list[tuple[float, float, int, str]] = []
    for candidate in candidates:
        source_bigrams = _bigrams(candidate)
        if not source_bigrams:
            continue
        shared = len(target & source_bigrams)
        ranked.append((
            shared / max(1, len(target)),
            shared / max(1, len(source_bigrams)),
            shared_literal_length(value, candidate),
            candidate,
        ))
    if not ranked:
        return ""
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    generated_coverage, source_recall, literal_length, best = ranked[0]
    if not (
        generated_coverage >= 0.42
        or (generated_coverage >= 0.20 and source_recall >= 0.55)
        or (generated_coverage >= 0.15 and literal_length >= 4)
    ):
        return ""
    if len(ranked) > 1:
        second = ranked[1]
        if (
            abs(generated_coverage - second[0]) < 0.05
            and abs(source_recall - second[1]) < 0.08
            and abs(literal_length - second[2]) <= 1
        ):
            return ""
    return best


def _repair_record_bullet(
    value: str,
    *,
    path: str,
    facts: list[FactUnit],
    binding: EvidenceBinding | None,
    source_fallbacks: list[str],
) -> tuple[str, str] | None:
    """Minimum-edit one bullet using only an explicit record fact allow-list."""

    if not facts:
        return None
    fact_ids = {fact.fact_id for fact in facts}
    scoped_binding = binding
    if binding is not None and binding.fact_ids and not set(binding.fact_ids).issubset(fact_ids):
        scoped_binding = None

    if (
        scoped_binding is not None
        and scoped_binding.mode == "rewritten"
        and scoped_binding.source_claim
        and scoped_binding.similarity >= 0.75
    ):
        whole_match = match_atomic_claim(
            value,
            facts,
            path=path,
            binding=scoped_binding,
        )
        # This provenance exists only after the optimizer's deterministic hard
        # gate and, for low-overlap wording, its semantic entailment review.
        # Recheck immutable anchors and record ownership here, but do not undo
        # an approved paraphrase merely because a single clause has low lexical
        # overlap with the exact source sentence.
        if whole_match.reason != "new_hard_anchor":
            return str(value or "").strip(), "accepted_reviewed_rewrite"

    clauses = atomize_claim_text(value)
    if not clauses:
        return "", "removed"
    safe: list[str] = []
    for clause in clauses:
        match = match_atomic_claim(
            clause,
            facts,
            path=path,
            binding=scoped_binding,
        )
        if match.status == "supported" and clause not in safe:
            safe.append(clause)
    if len(safe) == len(clauses):
        return str(value or "").strip(), "accepted"
    if safe:
        repaired = "，".join(safe)
        repaired = re.sub(r"^(?:并且|并|同时|此外|另外)[，,\s]*", "", repaired)
        repaired = repaired.rstrip("，,。；; ") + "。"
        return repaired, "trimmed"

    fallback = _unique_source_fallback(value, source_fallbacks)
    if fallback:
        return fallback.rstrip("，,。；; ") + "。", "restored"
    return "", "removed"


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
        records_to_remove: set[int] = set()
        for index, record in enumerate(getattr(gated, section)):
            strict_record_scope = index in incoherent_records[section]
            anchored_scope = _anchored_record_scope(
                record,
                section=section,
                source=source,
                blocks=eligible_blocks,
            )
            if strict_record_scope:
                if anchored_scope is None:
                    removed.append(f"{section}[{index}]")
                    records_to_remove.add(index)
                    continue
            for field in fields:
                path = f"{section}[{index}].{field}"
                value = str(getattr(record, field, "") or "").strip()
                wrong_record = bool(
                    value
                    and strict_record_scope
                    and anchored_scope
                    and anchored_scope not in _value_record_scopes(
                        value,
                        section=section,
                        source=source,
                        blocks=eligible_blocks,
                        role=(field == "role"),
                        organization=(field == "organization"),
                    )
                )
                if value and (path not in bound_paths or wrong_record):
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
                primary_block = next((
                    block for block in eligible_blocks
                    if binding is not None and block.block_id == binding.block_id
                ), None)
                binding_is_unsectioned = bool(
                    primary_block is not None and not primary_block.section_hint
                )
                record_facts = (
                    _facts_for_binding(source, binding)
                    if binding_is_unsectioned else (
                        _facts_for_record_scope(
                        source,
                        eligible_blocks,
                        section=section,
                        scope=anchored_scope,
                    )
                        if anchored_scope else _facts_for_binding(source, binding)
                    )
                )
                repair = _repair_record_bullet(
                    str(bullet or ""),
                    path=path,
                    facts=record_facts,
                    binding=binding,
                    source_fallbacks=(
                        _source_body_sentences(
                            source,
                            eligible_blocks,
                            section=section,
                            scope=anchored_scope,
                        )
                        if anchored_scope else []
                    ),
                )
                repaired_bullet = str(bullet or "").strip()
                repair_status = "not_applicable"
                if repair is not None:
                    repaired_bullet, repair_status = repair
                    if repaired_bullet != str(bullet or "").strip():
                        trace_event(
                            "atomic_evidence_repair",
                            path=path,
                            before=str(bullet or "").strip(),
                            after=repaired_bullet,
                            status=repair_status,
                            fact_ids=[fact.fact_id for fact in record_facts],
                            record_scope=anchored_scope,
                        )
                    binding = (
                        _bind_with_provenance(
                            path,
                            repaired_bullet,
                            eligible_blocks,
                            minimum=0.30,
                            trusted_rewrites=trusted_rewrites,
                        )
                        if repaired_bullet else None
                    )
                provenance_value = str((trusted_rewrites or {}).get(path, "") or "").strip()
                provenance_parts = [
                    part.strip()
                    for part in re.split(r"[\r\n]+", provenance_value)
                    if part.strip()
                ] or [repaired_bullet]
                bullet_scopes: set[str] = set()
                for provenance_part in provenance_parts:
                    bullet_scopes.update(_value_record_scopes(
                        provenance_part,
                        section=section,
                        source=source,
                        blocks=eligible_blocks,
                        minimum=0.30,
                    ))
                if binding is None or (
                    strict_record_scope
                    and anchored_scope
                    and anchored_scope not in bullet_scopes
                ):
                    removed.append(path)
                else:
                    if repaired_bullet and repaired_bullet not in kept_bullets:
                        kept_bullets.append(repaired_bullet)
            record.bullets = kept_bullets

        if records_to_remove:
            setattr(gated, section, [
                record for index, record in enumerate(getattr(gated, section))
                if index not in records_to_remove
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
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:%|万|亿|w|k|人|次|个|条|元|年|月|日)?|"
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
        if (
            not compact
            or compact in _SOURCE_HEADINGS
            or _is_section_heading(block.text)
            or len(compact) < 2
        ):
            continue
        if _SOURCE_DOCUMENT_TITLE.fullmatch(block.text.strip()):
            continue
        stripped = _SOURCE_FIELD_LABEL.sub("", block.text.strip())
        stripped = _COMPACT_SOURCE_FIELD_LABEL.sub("", stripped)
        raw_parts = [part.strip(" \t-•·▪◦") for part in _FACT_SPLIT.split(stripped)]
        factual_raw_parts = [part for part in raw_parts if len(_normalize(part)) >= 2]
        parts = [
            part for part in factual_raw_parts
            if not _SOURCE_NON_FACTUAL_COPY.fullmatch(part)
        ]
        if not parts:
            continue
        # Preserve the original punctuation in reports for ordinary one-fact
        # lines. Multi-fact/OCR-compressed lines expose each missing clause.
        displays = [block.text.strip()] if len(factual_raw_parts) == 1 else parts
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
    anchors = {
        _normalize(item)
        for item in _STRONG_FACT_ANCHOR.findall(unit.get("match_text", ""))
        if _normalize(item)
    }
    source_bigrams = _bigrams(source_value)
    for claim in claims:
        claim_value = _normalize(claim)
        if not claim_value:
            continue
        claim_anchors = {
            _normalize(item)
            for item in _STRONG_FACT_ANCHOR.findall(claim)
            if _normalize(item)
        }
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
            _normalize(item)
            for item in _STRONG_FACT_ANCHOR.findall(aggregate)
            if _normalize(item)
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
