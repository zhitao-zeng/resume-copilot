"""Minimal, evidence-preserving repair for V3 claims."""
from __future__ import annotations

from .contracts import Audit, FactGraph, FrozenResume, RealizedClaim
from .realizer import _join_facts


def minimal_repair(frozen: FrozenResume, audit: Audit, fact_graph: FactGraph) -> FrozenResume:
    """Repair only unsupported atomic references; retain source fallback.

    The implementation is intentionally conservative.  A future atomic parser
    may split a model sentence into multiple atoms; until then generated V3
    claims are fact joins, so unsupported IDs are removed and the remaining
    same-record source facts are rendered verbatim.
    """
    fact_map = fact_graph.fact_map()
    bad_claims = set(audit.unsupported_claim_ids)
    repaired: list[RealizedClaim] = []
    for claim in frozen.claims:
        if claim.claim_id not in bad_claims:
            repaired.append(claim)
            continue
        supported_ids = [fact_id for fact_id in claim.fact_ids if fact_id in fact_map and fact_map[fact_id].eligible and fact_map[fact_id].record_id == claim.record_id]
        if supported_ids:
            facts = [fact_map[fact_id] for fact_id in supported_ids]
            repaired.append(claim.model_copy(update={
                "text": _join_facts([fact.text for fact in facts]),
                "fact_ids": supported_ids,
                "anchors": [anchor for fact in facts for anchor in fact.anchors],
                "generated": len(facts) > 1,
            }))
            continue
        # Source sentence fallback is only allowed within the original record.
        fallback = None
        if claim.record_id is not None:
            candidates = [
                fact for fact in fact_graph.eligible_facts()
                if fact.record_id == claim.record_id
            ]
            if candidates:
                fallback = candidates[0]
        if fallback:
            repaired.append(claim.model_copy(update={"text": fallback.text, "fact_ids": [fallback.fact_id], "anchors": fallback.anchors, "generated": False}))
        # If no unique record fact exists, delete the claim; no cross-record
        # guess is permitted.
    sections: dict[str, list[RealizedClaim]] = {}
    for claim in repaired:
        sections.setdefault(claim.section, []).append(claim)
    return FrozenResume(sections=sections, claims=repaired, skeleton=frozen.skeleton, template_mode=frozen.template_mode)


__all__ = ["minimal_repair"]
