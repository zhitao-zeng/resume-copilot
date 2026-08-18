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
from .section_ontology import section_type

_YEAR_MONTH_TOKEN = (
    r"(?:(?:19|20)\d{2}(?:\s*[./年]\s*(?:0?[1-9]|1[0-2])(?:\s*月)?)?"
    # A two-digit year is accepted only with ``/``.  Treating ``2.85`` as a
    # month/year date incorrectly split GPA and decimal metrics into records.
    r"|(?:0?[1-9]|1[0-2])\s*/\s*(?:\d{2}|(?:19|20)\d{2}))"
)
_PRESENT_TOKEN = r"(?:至今|现在|目前|present|current)"
_DATE_RE = re.compile(rf"{_YEAR_MONTH_TOKEN}|{_PRESENT_TOKEN}", re.I)
_DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_YEAR_MONTH_TOKEN})\s*(?:-|–|—|~|至|到)\s*"
    rf"(?P<end>{_YEAR_MONTH_TOKEN}|{_PRESENT_TOKEN})",
    re.I,
)
_DATE_PLACEHOLDER_RANGE_RE = re.compile(
    rf"(?:年\s*月|\[\s*日期\s*\]|【\s*日期\s*】)\s*"
    rf"(?:-|–|—|~|至|到)+\s*"
    rf"(?:年\s*月|\[\s*日期\s*\]|【\s*日期\s*】|{_PRESENT_TOKEN})",
    re.I,
)
_CONTACT_RE = re.compile(r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|1\d{10})", re.I)
_METRIC_RE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:%|％|万元|万|千|百|人次|个|件|天|月|年|秒|毫秒|ms)"
    r"|(?:同比|提升|降低|增长|减少)\s*\d+(?:\.\d+)?(?:%|％)?)",
    re.I,
)
# Skills are classified by an explicit source section, not by a technology
# vocabulary.  This keeps the ontology applicable to doctors, teachers,
# operators, researchers and other professions without an industry dictionary.
_ACTION_RE = re.compile(r"^(?:负责|参与|主导|协助|支持|完成|推动|推进|组织|设计|开发|构建|实现|制定|管理|运营|分析|研究|撰写|输出|交付|维护|优化|搭建|建立|开展|承担|提供|跟进|协调|带领|执行)")
# A comma is ordinary narrative punctuation and cannot establish a record
# boundary.  Keep only separators that are structurally strong in compact
# organization/role headers.
_ORG_ROLE_SEP_RE = re.compile(
    # A leading middle dot is a list bullet.  Only an internal middle dot
    # between non-space tokens can mean a compact organization/role header.
    r"(?:\||｜|(?<=\S)·(?=\S)|\s{2,}|\s[-—–]\s)"
)
_QUERY_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|\n|(?=请帮我|帮我|我的简历|我在|本人|曾在|目前)")
_PLACEHOLDER_ONLY_RE = re.compile(
    r"(?:\[[^\]\n]{1,80}\]|【[^】\n]{1,80}】)"
    r"(?:\s*[,，|｜/]\s*(?:\[[^\]\n]{1,80}\]|【[^】\n]{1,80}】))*"
)
_LIST_PREFIX_RE = re.compile(r"^(?:[•·]\s*|-\s+|\*\s+)")
_STRUCTURAL_PLACEHOLDER_RE = re.compile(
    r"(?:\[[^\]\n]{1,80}(?:\]|$)|【[^】\n]{1,80}(?:】|$))"
)
_PURE_FIELD_LABEL_RE = re.compile(r"^[^\n:：]{1,48}\s*[:：]\s*$")


def _is_pure_field_label(value: str) -> bool:
    """Identify syntax-only labels without using a field-name vocabulary."""

    text = str(value or "").strip()
    return bool(
        _PURE_FIELD_LABEL_RE.fullmatch(text)
        and not _DATE_RE.search(text)
        and not _CONTACT_RE.search(text)
    )


def is_date_placeholder_range(value: str) -> bool:
    """Return whether ``value`` is a date-shaped template placeholder.

    These strings are useful layout boundaries but are not candidate facts.
    Keeping that distinction in the graph prevents a semantic model from
    publishing ``年月-年月`` as an employment period.
    """

    return bool(_DATE_PLACEHOLDER_RANGE_RE.fullmatch(str(value or "").strip()))


def _has_compact_placeholder_header(value: str) -> bool:
    """Return whether a compact record line combines text and a placeholder.

    A role plus an anonymised organization (for example ``岗位，[公司]``) is
    structural evidence even when OCR removed its dates.  Placeholder-only
    location/contact rows stay false, and no occupation vocabulary is used.
    """

    matches = list(_STRUCTURAL_PLACEHOLDER_RE.finditer(value))
    if not matches or len(value) > 120:
        return False
    outside = _STRUCTURAL_PLACEHOLDER_RE.sub("", value)
    substantive = re.sub(r"[\s,，|｜/;；:：()（）\-—–]+", "", outside)
    return len(substantive) >= 2


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
    if section == "awards":
        return "award"
    if section == "publications":
        return "publication"
    if section == "training":
        return "training"
    if section == "teaching":
        return "teaching"
    if section == "skills":
        return "skill"
    if _DATE_RE.search(text):
        return "period"
    if _ACTION_RE.search(text):
        return "action"
    if section == "projects":
        return "project"
    return "other"


# A record header is a compact label, not a sentence.  Narrative bullets can
# still contain date-shaped substrings — a ratio such as ``53/54`` parses as
# month/year — so the weaker single-date and separator signals additionally
# require header shape.  Clause punctuation is the discriminator: a header
# names an employer, role and period; it does not carry clauses.
_SENTENCE_PUNCT_RE = re.compile(r"[。！？；;]")
_MAX_HEADER_CHARS = 80


def _is_compact_header_shape(value: str) -> bool:
    if len(value) > _MAX_HEADER_CHARS:
        return False
    if _SENTENCE_PUNCT_RE.search(value):
        return False
    # A contact line is header-shaped but names no experience.  Its phone
    # digits also parse as a year (19975260767 contains 1997), which on real
    # PDFs opened a fresh record per contact line and shattered grouping.
    return not _CONTACT_RE.search(value)


def _looks_like_record_header(text: str, section: str, previous: str = "") -> bool:
    record_section = section in {"experience", "projects", "research", "education", "activities"}
    # Indentation is layout, not an organization/role separator.  Searching
    # the raw line made every indented responsibility look like a new record
    # because ``\s{2,}`` matched its leading spaces.
    value = text.strip()
    # A date *range* at either edge is a domain-neutral record boundary even
    # when the source omitted its "工作经历" heading.  This is common in
    # Query-only profiles and translated plain-text resumes.  Requiring an
    # edge range avoids turning narrative mentions such as "在2021年至2022年
    # 期间参与..." into a new job merely because they contain dates.
    date_range = _DATE_RANGE_RE.search(value) or _DATE_PLACEHOLDER_RANGE_RE.search(value)
    if date_range and (
        date_range.start() <= 8 or len(value) - date_range.end() <= 4
    ):
        return True
    if record_section and _has_compact_placeholder_header(value):
        return True
    # A company or project name may itself begin with a verb-like token (for
    # example "运营IT支持").  Test the stronger date-range boundary first;
    # only verb-led lines without such a boundary are treated as narrative.
    if _ACTION_RE.search(value):
        return False
    if not record_section:
        return False
    # Education and other explicitly-labelled record sections also use a
    # single year (for example "2000 甲大学") as a compact record header.
    if not _is_compact_header_shape(value):
        return False
    if _DATE_RE.search(value):
        return True
    if _ORG_ROLE_SEP_RE.search(value):
        return True
    if previous and previous.strip().endswith((":", "：")):
        return True
    return False


def _record_prefix_candidates(recent_nodes: list[object], date_node: object) -> list[object]:
    """Return compact header fragments immediately preceding a date range.

    Many resumes order a record as ``role -> organization -> period``.  A
    forward-only parser otherwise leaves the first role unowned and attaches
    every later role to the previous job.  This look-behind is intentionally
    structural: at most three adjacent short lines, never an action sentence,
    heading, earlier date, or content across a preserved blank line.
    """

    selected: list[object] = []
    right = date_node
    # Two lines cover the dominant lossless layouts (role + organization, or
    # degree + school).  Looking back farther can steal a preceding compact
    # achievement when OCR has dropped its bullet marker.
    for candidate in reversed(recent_nodes[-2:]):
        if getattr(candidate, "kind", "") in {"heading", "header", "footer"}:
            break
        text = str(getattr(candidate, "text", "") or "").strip()
        if not text or len(text) > 120 or _DATE_RE.search(text):
            break
        has_bullet = bool(_LIST_PREFIX_RE.match(text))
        explicit_header_bullet = bool(
            re.match(r"^(?:[•·]\s*|-\s+|\*\s+)(?:\*\*|__|`).+(?:\*\*|__|`)$", text)
            or _PLACEHOLDER_ONLY_RE.fullmatch(
                _LIST_PREFIX_RE.sub("", text).strip()
            )
            # A short bullet immediately followed by a placeholder row is a
            # compact role/degree header, not a responsibility belonging to
            # the previous record.  The placeholder supplies the structural
            # evidence; no occupation word list is involved.
            or (
                bool(selected)
                and _PLACEHOLDER_ONLY_RE.fullmatch(
                    _LIST_PREFIX_RE.sub(
                        "", str(getattr(right, "text", "") or "").strip()
                    ).strip()
                )
                is not None
                and len(_LIST_PREFIX_RE.sub("", text).strip()) <= 40
            )
        )
        if has_bullet and not explicit_header_bullet:
            break
        cleaned = _LIST_PREFIX_RE.sub("", text).strip()
        cleaned = re.sub(r"^(?:\*\*|__|`)+|(?:\*\*|__|`)+$", "", cleaned).strip()
        if not cleaned or _ACTION_RE.search(cleaned):
            break
        if cleaned.endswith(("。", "！", "？", "!", "?", "；", ";")):
            break
        left_spans = getattr(candidate, "source_spans", []) or []
        right_spans = getattr(right, "source_spans", []) or []
        if left_spans and right_spans:
            left_span, right_span = left_spans[-1], right_spans[0]
            if (
                left_span.source_id == right_span.source_id
                and right_span.char_start - left_span.char_end > 1
            ):
                break
        selected.append(candidate)
        right = candidate
    return list(reversed(selected))


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

    authoritative_layout_order = bool(graph.metadata.get("hybrid_layout"))
    sections: list[SectionNode] = []
    records: list[RecordNode] = []
    node_to_section: dict[str, str] = {}
    current_section: SectionNode | None = None
    record_counter: dict[str, int] = defaultdict(int)
    current_record: RecordNode | None = None
    explicit_records: dict[tuple[str, str, str], RecordNode] = {}
    region_records: dict[tuple[str, tuple[str, str, str, str]], RecordNode] = {}
    recent_nodes: list[object] = []
    next_record_number = 0
    for node in graph.ordered_nodes():
        if node.kind in {"header", "footer"}:
            # Story parts are structurally independent of the last body
            # section.  Inheriting (for example) a footer into "skills" would
            # lock the semantic compiler to a false section.
            section_id = f"{graph.source_id}:section:{len(sections)}"
            current_section = SectionNode(
                section_id=section_id,
                source_id=graph.source_id,
                title="",
                section_type="other",
                node_ids=[node.node_id],
                order=len(sections),
            )
            sections.append(current_section)
            node_to_section[node.node_id] = current_section.section_id
            current_record = None
            recent_nodes = []
            continue
        if node.kind == "heading":
            candidate = section_type(node.text)
            # Generic or unknown headings are still represented as sections;
            # their content is never silently thrown away.
            if current_section is None or candidate != "other" or node.text.strip().casefold() != current_section.title.strip().casefold():
                section_id = f"{graph.source_id}:section:{len(sections)}"
                current_section = SectionNode(section_id=section_id, source_id=graph.source_id, title=node.text.strip(), section_type=candidate, node_ids=[node.node_id], order=len(sections))
                sections.append(current_section)
                current_record = None
                recent_nodes = []
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
        strong_date_header = bool(
            (match := (
                _DATE_RANGE_RE.search(node.text.strip())
                or _DATE_PLACEHOLDER_RANGE_RE.search(node.text.strip())
            ))
            and (match.start() <= 8 or len(node.text.strip()) - match.end() <= 4)
        )
        date_tokens = list(_DATE_RE.finditer(node.text.strip()))
        prefix_date_header = bool(
            strong_date_header
            or (
                current_section.section_type in {
                    "experience", "projects", "research", "education", "activities",
                }
                and (
                    len(date_tokens) >= 2
                    or (current_section.section_type == "education" and date_tokens)
                )
            )
        )
        record: RecordNode | None = None
        if explicit_key is not None:
            record = explicit_records.get((current_section.section_id, *explicit_key))
        if record is None and region_key is not None and not is_header:
            record = region_records.get((current_section.section_id, region_key))
        if record is None and (is_header or explicit_key is not None):
            record_counter[current_section.section_id] += 1
            next_record_number += 1
            prefix_nodes = (
                _record_prefix_candidates(recent_nodes, node)
                if prefix_date_header and explicit_key is None
                else []
            )
            prefix_ids = [item.node_id for item in prefix_nodes]
            if prefix_ids:
                for prior_record in records:
                    prior_record.node_ids = [
                        node_id for node_id in prior_record.node_ids
                        if node_id not in prefix_ids
                    ]
            record = RecordNode(
                # Record IDs are document-global.  A per-section counter alone
                # produces duplicate IDs as soon as both work and education
                # sections contain a record.
                record_id=f"{graph.source_id}:record:{next_record_number}",
                source_id=graph.source_id,
                section_id=current_section.section_id,
                title=node.text.strip(),
                node_ids=[*prefix_ids, node.node_id],
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
            if (
                region_key is None
                or (current_layout is not None and current_layout[1] == region_key)
                # In the hybrid route PP-Structure supplies only a validated
                # region order while PP-OCRv6 supplies every line.  Adjacent
                # non-header regions may be continuation blocks of the same
                # record (notably when a two-column job header and its bullets
                # occupy separate regions).  A later date/placeholder header
                # still opens a new record and moves its bounded look-behind.
                or authoritative_layout_order
            ):
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
        recent_nodes.append(node)
    # A compact header can initially form a record through a separator and be
    # moved to the following date-anchored record.  Do not expose the emptied
    # intermediate node as an allowed ownership target.
    records = [record for record in records if record.node_ids]
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
        if is_date_placeholder_range(text):
            classification, eligible = "ineligible", False
        source_section = next(
            (section.section_type for section in sections if section.section_id == node_to_section.get(node.node_id)),
            "other",
        )
        if _is_pure_field_label(text):
            classification, eligible = "ineligible", False
        ftype = _fact_type(text, source_section)
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
                if not node.text.strip() or node.kind == "heading" or not node.source_spans:
                    continue
                parent = node.source_spans[0]
                base_fact = base_fact_by_node.get(parent.node_id)
                for clause, relative_start, relative_end in split_query_clauses(node.text):
                    start = parent.char_start + relative_start
                    end = parent.char_start + relative_end
                    is_fact = policy.query_clause_kind(clause) == "fact"
                    if _is_pure_field_label(clause):
                        is_fact = False
                    clause_span = SourceSpan(
                        source_id=graph.source_id,
                        char_start=start,
                        char_end=end,
                        page=parent.page,
                        node_id=parent.node_id,
                    )
                    fact_id = f"{graph.source_id}:query-fact:{query_fact_index}"
                    query_fact_index += 1
                    base_section = next(
                        (
                            section.section_type
                            for section in sections
                            if base_fact is not None and section.section_id == base_fact.section_id
                        ),
                        "other",
                    )
                    query_fact = FactUnit(
                        fact_id=fact_id,
                        source_id=graph.source_id,
                        source_type="query",
                        fact_type=_fact_type(clause, base_section),  # type: ignore[arg-type]
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
                        classification=(
                            "fact" if is_fact
                            else "ineligible" if _is_pure_field_label(clause)
                            else "intent"
                        ),
                    )
                    all_facts.append(query_fact)
                    if query_fact.record_id and query_fact.eligible:
                        records_by_id[query_fact.record_id].fact_ids.append(query_fact.fact_id)
    return FactGraph(documents=documents, sections=all_sections, records=all_records, facts=all_facts)


__all__ = ["build_document_structure", "build_fact_graph", "facts_from_graph", "section_type", "split_query_clauses"]
