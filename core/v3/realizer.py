"""Constrained narrative realization.

The first V3 implementation deliberately uses a deterministic connector
fallback.  A future LLM call can implement the same protocol, but it may only
rewrite a fixed group of fact IDs and must pass the same verifier afterwards.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re

from .contracts import FactGraph, FrozenResume, NarrativeGroup, RealizedClaim, RealizerResponse, ResumePlan


@dataclass(frozen=True)
class RealizerRequest:
    group: NarrativeGroup
    source_facts: tuple[str, ...]
    instruction: str = ""


def _placeholder(section: str) -> str:
    labels = {
        "contact": "姓名、联系方式",
        "summary": "个人概述",
        "experience": "工作/实习经历（公司、岗位、时间、职责与成果）",
        "projects": "项目经历（项目名称、角色、时间与成果）",
        "education": "教育经历（学校、专业、学历与时间）",
        "skills": "技能与工具",
    }
    return f"[待补充：{labels.get(section, section)}]"


def _join_facts(texts: list[str]) -> str:
    if len(texts) == 1:
        return texts[0]
    # This adds only a grammatical connector; every substantive phrase stays
    # verbatim and remains traceable to its own source span.
    return "；".join(text.rstrip("。；;") for text in texts) + "。"


def realize_plan(plan: ResumePlan, fact_graph: FactGraph) -> FrozenResume:
    fact_map = fact_graph.fact_map()
    claims: list[RealizedClaim] = []
    sections: dict[str, list[RealizedClaim]] = {}
    for index, group in enumerate(plan.groups):
        facts = [fact_map[fact_id] for fact_id in group.fact_ids if fact_id in fact_map and fact_map[fact_id].eligible]
        if not facts:
            if not plan.skeleton:
                continue
            claim = RealizedClaim(
                claim_id=f"claim:{index}", section=group.section,
                text=_placeholder(group.section), fact_ids=[], record_id=group.record_id,
                generated=False,
            )
        else:
            claim = RealizedClaim(
                claim_id=f"claim:{index}",
                section=group.section,
                text=_join_facts([fact.text for fact in facts]),
                fact_ids=[fact.fact_id for fact in facts],
                record_id=group.record_id,
                anchors=[anchor for fact in facts for anchor in fact.anchors],
                generated=len(facts) > 1,
            )
        claims.append(claim)
        sections.setdefault(group.section, []).append(claim)
    return FrozenResume(sections=sections, claims=claims, skeleton=plan.skeleton, template_mode=plan.template_mode)


def validate_realized_claims(frozen: FrozenResume, fact_graph: FactGraph) -> list[str]:
    """Return protocol violations without mutating generated content."""
    fact_map = fact_graph.fact_map()
    section_by_id = {section.section_id: section.section_type for section in fact_graph.sections}
    violations: list[str] = []
    for claim in frozen.claims:
        if not claim.fact_ids and not claim.text.startswith("[待补充："):
            violations.append(f"{claim.claim_id}:factless_non_placeholder")
        for fact_id in claim.fact_ids:
            fact = fact_map.get(fact_id)
            if fact is None:
                violations.append(f"{claim.claim_id}:unknown_fact:{fact_id}")
            elif not fact.eligible:
                violations.append(f"{claim.claim_id}:ineligible_fact:{fact_id}")
            elif fact.text not in claim.text:
                violations.append(f"{claim.claim_id}:source_text_not_preserved:{fact_id}")
            if fact is not None and fact.record_id != claim.record_id:
                violations.append(f"{claim.claim_id}:record_mismatch:{fact_id}")
            if fact is not None and fact.section_id is not None:
                expected_section = section_by_id.get(fact.section_id, "other")
                if expected_section != "other" and claim.section != expected_section:
                    violations.append(f"{claim.claim_id}:section_mismatch:{fact_id}")
    return violations


_NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?%?|\d{4}[./年-]\d{1,2})(?![A-Za-z])")


def validate_realizer_response(
    response: RealizerResponse,
    fact_graph: FactGraph,
    *,
    allowed_fact_ids: Iterable[str] | None = None,
) -> list[str]:
    """Validate the boundary of an optional constrained LLM response.

    This is deliberately conservative: it rejects unknown/out-of-request
    facts, record mixing, missing hard anchors and novel numeric tokens, while
    allowing ordinary connective words around source-backed text.
    """
    fact_map = fact_graph.fact_map()
    declared = set(response.request_fact_ids)
    requested = set(allowed_fact_ids) if allowed_fact_ids is not None else declared
    violations: list[str] = []
    if len(response.request_fact_ids) != len(declared):
        violations.append("request_fact_ids_not_unique")
    if allowed_fact_ids is not None and declared != requested:
        violations.append("declared_request_fact_ids_mismatch")
    for fact_id in sorted(requested):
        fact = fact_map.get(fact_id)
        if fact is None or not fact.eligible:
            violations.append(f"request_fact_not_eligible:{fact_id}")
    section_by_id = {section.section_id: section.section_type for section in fact_graph.sections}
    for claim in response.claims:
        if not claim.fact_ids:
            if not claim.text.startswith("[待补充："):
                violations.append(f"{claim.claim_id}:factless_non_placeholder")
            continue
        if len(claim.fact_ids) != len(set(claim.fact_ids)):
            violations.append(f"{claim.claim_id}:duplicate_fact_id")
        claim_facts = [fact_map[fact_id] for fact_id in claim.fact_ids if fact_id in fact_map and fact_id in requested and fact_map[fact_id].eligible]
        allowed_numbers = {
            number
            for fact in claim_facts
            for number in _NUMBER_RE.findall(fact.text)
        }
        output_numbers = set(_NUMBER_RE.findall(claim.text))
        if not output_numbers <= allowed_numbers:
            violations.append(f"{claim.claim_id}:novel_numeric_anchor")
        for fact_id in claim.fact_ids:
            fact = fact_map.get(fact_id)
            if fact_id not in requested:
                violations.append(f"{claim.claim_id}:fact_not_requested:{fact_id}")
                continue
            if fact is None or not fact.eligible:
                violations.append(f"{claim.claim_id}:fact_not_eligible:{fact_id}")
                continue
            if fact.record_id != claim.record_id:
                violations.append(f"{claim.claim_id}:record_mismatch:{fact_id}")
            if fact.section_id is not None:
                expected_section = section_by_id.get(fact.section_id, "other")
                if expected_section != "other" and claim.section != expected_section:
                    violations.append(f"{claim.claim_id}:section_mismatch:{fact_id}")
            for anchor in fact.anchors:
                if anchor.text not in claim.text:
                    violations.append(f"{claim.claim_id}:missing_anchor:{anchor.text}")
    return violations


__all__ = ["RealizerRequest", "realize_plan", "validate_realized_claims", "validate_realizer_response"]
