"""Deterministic source-locked resume planning."""
from __future__ import annotations

from collections import defaultdict
import re

from .contracts import CoverageLedger, FactGraph, NarrativeGroup, RequirementGraph, ResumePlan, TemplateAST


_SECTION_ORDER = {
    "contact": 0,
    "summary": 1,
    "experience": 2,
    "projects": 3,
    "research": 4,
    "education": 5,
    "activities": 6,
    "skills": 7,
    "credentials": 8,
    "awards": 9,
    "publications": 10,
    "training": 11,
    "teaching": 12,
    "additional": 13,
    "other": 14,
}


def _overlap_score(text: str, keywords: list[str]) -> int:
    value = text.casefold()
    return sum(1 for keyword in keywords if keyword.casefold() in value)


# The month group must not swallow the leading digits of a second year
# ("2019 - 2021" is year-to-year, not year+month 20).
_PERIOD_START = re.compile(r"(\d{4})\s*[年./\-]\s*(\d{1,2}(?!\d))?")
# Sections whose entries the acceptance rubric orders most-recent first.
_TIME_ORDERED_SECTIONS = {"experience", "projects", "research", "activities", "teaching", "education"}


def _record_start_key(fact_ids: list[str], fact_map: dict) -> int:
    """Latest period start in the record as yyyymm; -1 when undated."""

    best = -1
    for fact_id in fact_ids:
        fact = fact_map.get(fact_id)
        if fact is None or fact.fact_type != "period":
            continue
        match = _PERIOD_START.search(fact.text)
        if not match:
            continue
        year = int(match.group(1))
        month = int(match.group(2) or 1)
        if 1900 <= year <= 2100 and 1 <= month <= 12:
            best = max(best, year * 100 + month)
    return best


def plan_resume(
    fact_graph: FactGraph,
    requirements: RequirementGraph | None = None,
    template: TemplateAST | None = None,
) -> ResumePlan:
    eligible = fact_graph.eligible_facts()
    requirements = requirements or RequirementGraph()
    groups: list[NarrativeGroup] = []
    facts_by_record: dict[str | None, list[str]] = defaultdict(list)
    for fact in eligible:
        facts_by_record[fact.record_id].append(fact.fact_id)
    # Record groups are immutable: a fact from one experience can never be
    # planned together with a different experience merely due to lexical fit.
    for record_id, fact_ids in facts_by_record.items():
        if record_id is not None:
            record = next((item for item in fact_graph.records if item.record_id == record_id), None)
            destinations = {
                fact_graph.fact_map()[fact_id].destination_section
                for fact_id in fact_ids
                if fact_graph.fact_map()[fact_id].destination_section
            }
            section = (
                next(iter(destinations))
                if len(destinations) == 1
                else next((item.section_type for item in fact_graph.sections if record and item.section_id == record.section_id), "experience")
            )
            groups.append(NarrativeGroup(
                group_id=f"record:{record_id}", section=section,
                source_id=record.source_id if record else None,
                section_id=record.section_id if record else None,
                record_id=record_id,
                fact_ids=fact_ids, purpose="record", priority=sum(_overlap_score(fact_graph.fact_map()[fid].text, requirements.keywords) for fid in fact_ids),
            ))
        else:
            by_scope: dict[tuple[str, str | None, str], list[str]] = defaultdict(list)
            for fact_id in fact_ids:
                fact = fact_graph.fact_map()[fact_id]
                section = fact.destination_section or next((item.section_type for item in fact_graph.sections if item.section_id == fact.section_id), "other")
                by_scope[(fact.source_id, fact.section_id, section)].append(fact_id)
            for (source_id, section_id, section), scoped_ids in by_scope.items():
                purpose = "skills" if section == "skills" else "education" if section == "education" else "contact" if section == "contact" else "other"
                groups.append(NarrativeGroup(
                    group_id=f"section:{source_id}:{section_id or 'unscoped'}:{len(groups)}",
                    section=section,
                    source_id=source_id,
                    section_id=section_id,
                    fact_ids=scoped_ids,
                    purpose=purpose,
                    priority=sum(_overlap_score(fact_graph.fact_map()[fid].text, requirements.keywords) for fid in scoped_ids),
                ))
    if not groups:
        # JD-only requests receive a visible structure, never invented
        # experience.  Placeholder wording is renderer responsibility.
        for section in ("contact", "summary", "experience", "projects", "education", "skills"):
            purpose = "framework"
            groups.append(NarrativeGroup(group_id=f"framework:{section}", section=section, fact_ids=[], purpose=purpose, priority=0))
    # Rubric: experience/education entries are ordered most-recent first by
    # start date.  JD relevance must not reorder dated records — it only
    # breaks ties among undated ones — and it must never move
    # identity/contact or summary behind an experience merely because a job
    # keyword happened to match the latter.
    fact_map = fact_graph.fact_map()
    start_keys = {
        group.group_id: _record_start_key(group.fact_ids, fact_map)
        for group in groups
        if group.record_id is not None and group.section in _TIME_ORDERED_SECTIONS
    }
    groups.sort(key=lambda group: (
        _SECTION_ORDER.get(group.section, 99),
        -start_keys.get(group.group_id, -1),
        -group.priority,
        group.group_id,
    ))
    planned = [fact_id for group in groups for fact_id in group.fact_ids]
    ledger = CoverageLedger(eligible_fact_ids=[fact.fact_id for fact in eligible], planned_fact_ids=planned, omitted_fact_ids=[])
    mode = template.mode if template else "none"
    notes = []
    if requirements.requirements:
        notes.append("JD requirements prioritize source-backed groups only; they do not add candidate facts")
    if not eligible:
        notes.append("No candidate facts supplied; generated sections are structured placeholders")
    return ResumePlan(groups=groups, ledger=ledger, skeleton=not bool(eligible), template_mode=mode, notes=notes)


__all__ = ["plan_resume"]
