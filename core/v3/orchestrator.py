"""Shadow orchestrator for the isolated V3 compiler."""
from __future__ import annotations

from typing import Any, Iterable

from .atomic_verifier import audit_frozen_resume
from .contracts import CoverageLedger, DocumentGraph, SourceAsset, SourcePolicy, TemplateAST, V3Output
from .document_graph import build_document_graph
from .fact_graph import build_fact_graph
from .jd_graph import build_requirement_graph
from .planner import plan_resume
from .realizer import realize_plan
from .repair import minimal_repair
from .reply_builder import build_reply


V3Result = V3Output


def run_v3(
    *,
    cv_text: str = "",
    query_text: str = "",
    jd_text: str = "",
    template: TemplateAST | dict[str, Any] | None = None,
    policy: SourcePolicy | None = None,
    ppstructure_blocks: dict[str, Iterable[dict[str, Any]]] | None = None,
    document_graphs: Iterable[DocumentGraph] | None = None,
) -> V3Result:
    """Run V3 in shadow mode; no V2 production code is imported."""
    policy = policy or SourcePolicy()
    template_ast = TemplateAST.model_validate(template) if isinstance(template, dict) else template
    graphs = list(document_graphs or [])
    source_ids = [graph.source_id for graph in graphs]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("document_graphs must use unique source_id values")
    sources = (("cv", cv_text, "resume.txt"), ("query", query_text, "query.txt"))
    for source_id, text, filename in sources:
        if not text:
            continue
        if source_id in source_ids:
            raise ValueError(f"source supplied as both text and DocumentGraph: {source_id}")
        asset = SourceAsset(source_id=source_id, source_type=source_id, filename=filename, media_type="text/plain", text=text, native=True)
        blocks = ppstructure_blocks.get(source_id) if ppstructure_blocks else None
        graphs.append(build_document_graph(
            asset,
            text=text,
            ppstructure_blocks=blocks,
            quality=1.0,
            shadow_ppstructure=blocks is not None,
        ))
        source_ids.append(source_id)
    # An empty run still has a graph, allowing the JD-only framework path to
    # remain observable and testable.
    if not graphs:
        empty_asset = SourceAsset(source_id="cv", source_type="cv", filename="resume.txt", media_type="text/plain", text="", native=True)
        graphs.append(build_document_graph(empty_asset, text=""))
    fact_graph = build_fact_graph(graphs, policy)
    requirements = build_requirement_graph(jd_text) if jd_text else build_requirement_graph("")
    plan = plan_resume(fact_graph, requirements, template_ast)
    frozen = realize_plan(plan, fact_graph)
    audit = audit_frozen_resume(frozen, fact_graph, requirements)
    if not audit.clean:
        frozen = minimal_repair(frozen, audit, fact_graph)
        audit = audit_frozen_resume(frozen, fact_graph, requirements)
    ledger = CoverageLedger(
        eligible_fact_ids=plan.ledger.eligible_fact_ids,
        planned_fact_ids=plan.ledger.planned_fact_ids,
        written_fact_ids=audit.written_fact_ids,
        omitted_fact_ids=audit.missing_fact_ids,
        omission_reasons={
            fact_id: audit.fact_reasons.get(fact_id, "not present in frozen output")
            for fact_id in audit.missing_fact_ids
        },
    )
    plan = plan.model_copy(update={"ledger": ledger})
    reply = build_reply(audit, fact_graph, requirements)
    return V3Output(graph=fact_graph, plan=plan, frozen=frozen, audit=audit, reply=reply)


def shadow_orchestrate(**kwargs: Any) -> V3Result:
    return run_v3(**kwargs)


__all__ = ["V3Result", "run_v3", "shadow_orchestrate"]
