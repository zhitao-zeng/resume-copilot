"""Atomic, ownership-aware verification for frozen V3 output."""
from __future__ import annotations

from .contracts import Audit, FactGraph, FrozenResume, RequirementGraph
from .realizer import validate_realized_claims


def audit_frozen_resume(
    frozen: FrozenResume,
    fact_graph: FactGraph,
    requirements: RequirementGraph | None = None,
) -> Audit:
    fact_map = fact_graph.fact_map()
    violations = validate_realized_claims(frozen, fact_graph)
    supported: list[str] = []
    unsupported: list[str] = []
    written: list[str] = []
    ownership_errors: list[str] = []
    reasons: dict[str, str] = {}
    for claim in frozen.claims:
        claim_bad = any(item.startswith(f"{claim.claim_id}:") for item in violations)
        if claim_bad:
            unsupported.append(claim.claim_id)
        else:
            supported.append(claim.claim_id)
        for fact_id in claim.fact_ids:
            fact = fact_map.get(fact_id)
            if fact is None:
                reasons[fact_id] = "unknown fact id"
                continue
            if fact_id not in written and not claim_bad:
                written.append(fact_id)
            if fact.record_id != claim.record_id:
                ownership_errors.append(f"{claim.claim_id}:{fact_id}")
                reasons[fact_id] = "claim crosses source record boundary"
    eligible = {fact.fact_id for fact in fact_graph.eligible_facts()}
    missing = sorted(eligible - set(written))
    for fact_id in missing:
        reasons.setdefault(fact_id, "not selected by deterministic planner")
    conflicts: list[str] = []
    # Conflicts are reported only when two facts from the same record have the
    # same critical type but different exact text; no external truth is
    # inferred, so this remains an audit hint rather than destructive repair.
    by_record_type: dict[tuple[str | None, str], list[str]] = {}
    for fact in fact_graph.eligible_facts():
        if fact.fact_type in {"organization", "role", "period", "metric", "credential"}:
            by_record_type.setdefault((fact.record_id, fact.fact_type), []).append(fact.text)
    for (record_id, fact_type), texts in by_record_type.items():
        if len(set(texts)) > 1:
            conflicts.append(f"{record_id or 'unassigned'}:{fact_type}")
    recommendations: list[str] = []
    if missing:
        recommendations.append("补充未进入简历的源信息，或在回复中说明未采用原因")
    if not fact_graph.eligible_facts():
        recommendations.append("提供姓名、联系方式、教育、项目或工作/校园经历后再生成定向内容")
    if requirements and requirements.requirements and not missing:
        recommendations.append("岗位要求均应通过已有事实表达，不能从JD新增经历")
    return Audit(
        supported_claim_ids=supported,
        unsupported_claim_ids=unsupported,
        written_fact_ids=sorted(set(written)),
        missing_fact_ids=missing,
        ownership_errors=sorted(set(ownership_errors)),
        conflicts=sorted(set(conflicts)),
        recommendations=recommendations,
        violations=violations,
        fact_reasons=reasons,
    )


def verify_frozen_resume(frozen: FrozenResume, fact_graph: FactGraph, requirements: RequirementGraph | None = None) -> Audit:
    return audit_frozen_resume(frozen, fact_graph, requirements)


__all__ = ["audit_frozen_resume", "verify_frozen_resume"]
