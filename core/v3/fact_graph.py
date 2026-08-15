"""Deterministic evidence and ownership graph construction.

This module uses only structural signals and generic semantic anchors.  It does
not maintain a profession vocabulary.  When the source does not establish a
record boundary, the compiler keeps the fact unassigned instead of guessing
across neighboring experiences.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from .contracts import Anchor, DocumentGraph, FactGraph, FactUnit, RecordNode, SectionNode, SourcePolicy, SourceSpan


_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "contact": ("个人信息", "基本信息", "联系方式", "contact", "personal information"),
    "summary": ("个人总结", "个人简介", "职业概述", "summary", "objective"),
    "education": ("教育经历", "教育背景", "教育", "education"),
    "experience": ("工作经历", "工作经验", "实习经历", "职业经历", "experience", "employment"),
    "projects": ("项目经历", "项目经验", "项目", "projects", "project experience"),
    "research": ("科研经历", "研究经历", "research"),
    "activities": ("校园经历", "社会实践", "志愿经历", "activities", "campus"),
    "skills": ("专业技能", "技能", "技术能力", "skills", "technical skills"),
    "credentials": ("证书", "资质", "认证", "certifications", "licenses"),
}
_DATE_RE = re.compile(r"(?:19|20)\d{2}\s*[./年-]\s*\d{1,2}|(?:19|20)\d{2}|至今|present", re.I)
_CONTACT_RE = re.compile(r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|1\d{10}|(?:电话|手机|邮箱|email|phone)\s*[:：])", re.I)
_METRIC_RE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:%|％|万元|万|千|百|人次|个|件|天|月|年|秒|毫秒|ms)"
    r"|(?:同比|提升|降低|增长|减少)\s*\d+(?:\.\d+)?(?:%|％)?)",
    re.I,
)
# Skills are classified by an explicit source section, not by a technology
# vocabulary.  This keeps the ontology applicable to doctors, teachers,
# operators, researchers and other professions without an industry dictionary.
_ACTION_RE = re.compile(r"^(?:负责|参与|主导|协助|支持|完成|推动|推进|组织|设计|开发|构建|实现|制定|管理|运营|分析|研究|撰写|输出|交付|维护|优化|搭建|建立|开展|承担|提供|跟进|协调|带领|执行)")
_ORG_ROLE_SEP_RE = re.compile(r"(?:\||｜|·|,|，|\s{2,}|\s[-—–]\s)")
_QUERY_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|\n|(?=请帮我|帮我|我的简历|我在|本人|曾在|目前)")


def _norm(value: str) -> str:
    return re.sub(r"[\s:：|｜/\\【】\[\]()（）*#\d.]+", "", value or "").casefold()


def section_type(title: str) -> str:
    value = _norm(title)
    for name, aliases in _SECTION_ALIASES.items():
        if any(_norm(alias) == value or _norm(alias) in value for alias in aliases):
            return name
    return "other"


def _pattern_anchors(
    *,
    source_id: str,
    text: str,
    span_start: int,
    page: int | None,
    node_id: str | None,
) -> list[Anchor]:
    """Extract every generic hard anchor from one exact source fragment."""

    anchors: list[Anchor] = []
    seen: set[tuple[str, int, int]] = set()
    for entity_type, pattern in (
        ("contact", _CONTACT_RE),
        ("metric", _METRIC_RE),
        ("period", _DATE_RE),
    ):
        for match in pattern.finditer(text):
            key = (entity_type, match.start(), match.end())
            if key in seen:
                continue
            seen.add(key)
            anchors.append(Anchor(
                entity_type=entity_type,  # type: ignore[arg-type]
                text=match.group(0),
                span=SourceSpan(
                    source_id=source_id,
                    char_start=span_start + match.start(),
                    char_end=span_start + match.end(),
                    page=page,
                    node_id=node_id,
                ),
            ))
    return anchors


def _fact_type(text: str, section: str) -> str:
    if _CONTACT_RE.search(text) or section == "contact":
        return "contact"
    if _METRIC_RE.search(text):
        return "metric"
    if section == "education":
        return "education"
    if section == "credentials":
        return "credential"
    if section == "skills":
        return "skill"
    if _DATE_RE.search(text):
        return "period"
    if _ACTION_RE.search(text):
        return "action"
    if section == "projects":
        return "project"
    return "other"


def _looks_like_record_header(text: str, section: str, previous: str = "") -> bool:
    if section not in {"experience", "projects", "research", "education", "activities"}:
        return False
    if _ACTION_RE.search(text.strip()):
        return False
    if _DATE_RE.search(text):
        return True
    if _ORG_ROLE_SEP_RE.search(text) and len(text) <= 120:
        return True
    if previous and previous.strip().endswith((":", "：")):
        return True
    return False


def _explicit_layout_key(node: object) -> tuple[str, str] | None:
    """Return an explicit record/container/table-row key when supplied."""
    metadata = getattr(node, "metadata", {}) or {}
    for name in ("record_id", "record", "container_id", "table_row_id", "row_id"):
        value = metadata.get(name)
        if value not in (None, ""):
            return name, str(value)
    parent_id = getattr(node, "parent_id", None)
    if parent_id:
        return "parent_id", str(parent_id)
    return None


def _region_key(node: object) -> tuple[str, str, str, str] | None:
    """Structural locality key; absent layout means no ownership inference."""
    page = str(getattr(node, "page", ""))
    column = str(getattr(node, "column_id", "") or "")
    region = str(getattr(node, "region_id", "") or "")
    parent = str(getattr(node, "parent_id", "") or "")
    if not any((column, region, parent)):
        return None
    return page, column, region, parent


def build_document_structure(graph: DocumentGraph) -> tuple[list[SectionNode], list[RecordNode], dict[str, str]]:
    """Create section/record boundaries without semantic guessing."""

    sections: list[SectionNode] = []
    records: list[RecordNode] = []
    node_to_section: dict[str, str] = {}
    current_section: SectionNode | None = None
    record_counter: dict[str, int] = defaultdict(int)
    current_record: RecordNode | None = None
    explicit_records: dict[tuple[str, str, str], RecordNode] = {}
    region_records: dict[tuple[str, tuple[str, str, str, str]], RecordNode] = {}
    for node in graph.ordered_nodes():
        if node.kind == "heading":
            candidate = section_type(node.text)
            # Generic or unknown headings are still represented as sections;
            # their content is never silently thrown away.
            if current_section is None or candidate != "other" or _norm(node.text) != _norm(current_section.title):
                section_id = f"{graph.source_id}:section:{len(sections)}"
                current_section = SectionNode(section_id=section_id, source_id=graph.source_id, title=node.text.strip(), section_type=candidate, node_ids=[node.node_id], order=len(sections))
                sections.append(current_section)
                current_record = None
            else:
                current_section.node_ids.append(node.node_id)
            node_to_section[node.node_id] = current_section.section_id
            continue
        if current_section is None:
            section_id = f"{graph.source_id}:section:0"
            current_section = SectionNode(section_id=section_id, source_id=graph.source_id, title="", section_type="other", order=0)
            sections.append(current_section)
        current_section.node_ids.append(node.node_id)
        node_to_section[node.node_id] = current_section.section_id
        explicit_key = _explicit_layout_key(node)
        region_key = _region_key(node)
        is_header = _looks_like_record_header(node.text, current_section.section_type, current_record.title if current_record else "")
        record: RecordNode | None = None
        if explicit_key is not None:
            record = explicit_records.get((current_section.section_id, *explicit_key))
        if record is None and region_key is not None and not is_header:
            record = region_records.get((current_section.section_id, region_key))
        if record is None and (is_header or explicit_key is not None):
            record_counter[current_section.section_id] += 1
            record = RecordNode(
                record_id=f"{graph.source_id}:record:{record_counter[current_section.section_id]}",
                source_id=graph.source_id,
                section_id=current_section.section_id,
                title=node.text.strip(),
                node_ids=[node.node_id],
                period="".join(_DATE_RE.findall(node.text))[:80],
            )
            records.append(record)
            if explicit_key is not None:
                explicit_records[(current_section.section_id, *explicit_key)] = record
            if region_key is not None:
                region_records[(current_section.section_id, region_key)] = record
        elif record is None and current_record is not None and current_record.section_id == current_section.section_id:
            # Adjacency is the weakest boundary.  It is safe only when the
            # node has no conflicting region/column; a new region abstains.
            current_layout = next((key for key, item in region_records.items() if item.record_id == current_record.record_id), None)
            if region_key is None or (current_layout is not None and current_layout[1] == region_key):
                record = current_record
        if record is not None:
            if node.node_id not in record.node_ids:
                record.node_ids.append(node.node_id)
            current_record = record
        elif region_key is not None:
            # A paragraph in a different unregistered region cannot inherit
            # the prior experience: ownership remains deliberately unknown.
            current_record = None
        # A standalone line in an experience section does not establish a
        # cross-record relation; it remains section-scoped until an explicit
        # new record header appears.
    return sections, records, node_to_section


def facts_from_graph(graph: DocumentGraph, policy: SourcePolicy | None = None) -> tuple[list[SectionNode], list[RecordNode], list[FactUnit]]:
    policy = policy or SourcePolicy()
    sections, records, node_to_section = build_document_structure(graph)
    record_by_node: dict[str, str] = {}
    for record in records:
        for node_id in record.node_ids:
            record_by_node.setdefault(node_id, record.record_id)
    facts: list[FactUnit] = []
    if graph.source_type in {"jd", "template"}:
        # Requirement/template content is intentionally not compiled into the
        # candidate ledger.  The graph still records an ineligible audit trail.
        return sections, records, facts
    for node in graph.ordered_nodes():
        text = node.text.strip()
        if not text or node.kind == "heading" or not node.source_spans:
            continue
        parent_span = node.source_spans[0]
        leading = len(node.text) - len(node.text.lstrip())
        source_span = SourceSpan(
            source_id=graph.source_id,
            char_start=parent_span.char_start + leading,
            char_end=parent_span.char_end - (len(node.text) - len(node.text.rstrip())),
            page=parent_span.page,
            node_id=parent_span.node_id,
        )
        classification = "fact"
        eligible = policy.source_can_support_facts(graph.source_type)
        if graph.source_type == "query" and policy.query_clause_kind(text) == "intent":
            classification, eligible = "intent", False
        ftype = _fact_type(text, section_type(next((section.title for section in sections if section.section_id == node_to_section.get(node.node_id)), "")))
        record_id = record_by_node.get(node.node_id)
        anchors = _pattern_anchors(
            source_id=graph.source_id,
            text=text,
            span_start=source_span.char_start,
            page=source_span.page,
            node_id=source_span.node_id,
        )
        fact = FactUnit(
            fact_id=f"{graph.source_id}:fact:{len(facts)}",
            source_id=graph.source_id,
            source_type=graph.source_type,
            fact_type=ftype,  # type: ignore[arg-type]
            text=text,
            spans=[source_span],
            section_id=node_to_section.get(node.node_id),
            record_id=record_id,
            anchors=anchors,
            eligible=eligible,
            confidence=node.confidence,
            classification=classification,  # type: ignore[arg-type]
        )
        facts.append(fact)
        if record_id:
            for record in records:
                if record.record_id == record_id:
                    record.fact_ids.append(fact.fact_id)
                    break
    return sections, records, facts


def split_query_clauses(text: str) -> list[tuple[str, int, int]]:
    """Split query into clauses while retaining exact character offsets."""
    clauses: list[tuple[str, int, int]] = []
    cursor = 0
    for part in _QUERY_SPLIT_RE.split(text or ""):
        if not part:
            continue
        start = text.find(part, cursor)
        if start < 0:
            start = cursor
        end = start + len(part)
        cursor = end
        value = part.strip()
        if not value:
            continue
        left = start + len(part) - len(part.lstrip())
        right = left + len(value)
        clauses.append((value, left, right))
    return clauses


def build_fact_graph(graphs: Iterable[DocumentGraph], policy: SourcePolicy | None = None) -> FactGraph:
    policy = policy or SourcePolicy()
    all_sections: list[SectionNode] = []
    all_records: list[RecordNode] = []
    all_facts: list[FactUnit] = []
    documents: dict[str, str] = {}
    for graph in graphs:
        documents[graph.source_id] = graph.source_text
        sections, records, facts = facts_from_graph(graph, policy)
        all_sections.extend(sections)
        all_records.extend(records)
        if graph.source_type != "query":
            all_facts.extend(facts)
        else:
            # Query text commonly mixes personal facts and instructions in one
            # paragraph.  Rebuild it at clause granularity so an instruction
            # cannot poison the factual clause's eligibility.
            base_fact_by_node = {
                fact.spans[0].node_id: fact
                for fact in facts
                if fact.spans and fact.spans[0].node_id is not None
            }
            records_by_id = {record.record_id: record for record in records}
            for record in records:
                record.fact_ids.clear()
            query_fact_index = 0
            for node in graph.ordered_nodes():
                if not node.text.strip() or not node.source_spans:
                    continue
                parent = node.source_spans[0]
                base_fact = base_fact_by_node.get(parent.node_id)
                for clause, relative_start, relative_end in split_query_clauses(node.text):
                    start = parent.char_start + relative_start
                    end = parent.char_start + relative_end
                    is_fact = policy.query_clause_kind(clause) == "fact"
                    clause_span = SourceSpan(
                        source_id=graph.source_id,
                        char_start=start,
                        char_end=end,
                        page=parent.page,
                        node_id=parent.node_id,
                    )
                    fact_id = f"{graph.source_id}:query-fact:{query_fact_index}"
                    query_fact_index += 1
                    query_fact = FactUnit(
                        fact_id=fact_id,
                        source_id=graph.source_id,
                        source_type="query",
                        fact_type=_fact_type(clause, "other"),  # type: ignore[arg-type]
                        text=clause,
                        spans=[clause_span],
                        section_id=base_fact.section_id if base_fact else None,
                        record_id=base_fact.record_id if base_fact else None,
                        anchors=_pattern_anchors(
                            source_id=graph.source_id,
                            text=clause,
                            span_start=start,
                            page=parent.page,
                            node_id=parent.node_id,
                        ),
                        eligible=is_fact,
                        confidence=base_fact.confidence if base_fact else node.confidence,
                        classification="fact" if is_fact else "intent",
                    )
                    all_facts.append(query_fact)
                    if query_fact.record_id and query_fact.eligible:
                        records_by_id[query_fact.record_id].fact_ids.append(query_fact.fact_id)
    return FactGraph(documents=documents, sections=all_sections, records=all_records, facts=all_facts)


__all__ = ["build_document_structure", "build_fact_graph", "facts_from_graph", "section_type", "split_query_clauses"]
