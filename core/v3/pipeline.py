"""Production, opt-in end-to-end Resume Evidence Compiler V3 pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .atomic_verifier import audit_frozen_resume
from .contracts import CoverageLedger, DocumentGraph, SourceAsset, SourcePolicy, TemplateAST, V3Output
from .document_graph import from_native_text
from .fact_graph import build_fact_graph
from .input_adapters import build_input_document_graph
from .jd_graph import build_requirement_graph
from .numeric_guard import quarantine_suspect_numeric_facts
from .planner import plan_resume
from .realizer_llm import RealizationReport
from .realizer_records import realize_record_local
from .repair import minimal_repair
from .reply_builder import build_reply, _friendly_conflicts
from .resume_adapter import frozen_to_resume_data
from .semantic_llm import SemanticCompilationReport, compile_semantics
from .summary_compiler import compile_summary
from .training_examples import build_training_records, maybe_write_training_trace


@dataclass(frozen=True)
class V3PipelineResult:
    output: V3Output
    resume_data: dict[str, Any]
    quality_report: dict[str, Any]
    missing_fields: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    changes: list[dict[str, Any]]
    reply_text: str
    semantic_report: SemanticCompilationReport
    realization_report: RealizationReport
    training_trace_path: str = ""


def _native_graph(source_id: str, source_type: str, text: str):
    asset = SourceAsset(
        source_id=source_id,
        source_type=source_type,  # type: ignore[arg-type]
        filename=f"{source_id}.txt",
        media_type="text/plain",
        text=text,
        native=True,
    )
    return from_native_text(asset, text)


def _quality_report(output: V3Output, semantic: SemanticCompilationReport, realization: RealizationReport, summary: dict[str, Any] | None = None, numeric_quarantine: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    audit = output.audit
    graph = output.graph
    factual_claim_ids = [claim.claim_id for claim in output.frozen.claims if claim.fact_ids]
    supported_factual = [claim_id for claim_id in audit.supported_claim_ids if claim_id in factual_claim_ids]
    eligible = graph.eligible_fact_ids
    written = audit.written_fact_ids
    fact_precision = len(supported_factual) / len(factual_claim_ids) if factual_claim_ids else 1.0
    fact_recall = len(set(written)) / len(set(eligible)) if eligible else 1.0
    fact_map = graph.fact_map()
    protected_anchors = [anchor for fact in graph.eligible_facts() for anchor in fact.anchors]
    written_text = "\n".join(claim.text for claim in output.frozen.claims)
    missing_anchors = [anchor.text for anchor in protected_anchors if anchor.text not in written_text]
    return {
        "version": "1.1",
        "pipeline": "resume_evidence_compiler_v3",
        "schema_version": semantic.schema_version,
        "atomic_factuality": {
            "precision": round(fact_precision, 4),
            "recall": round(fact_recall, 4),
            "eligible_fact_count": len(eligible),
            "written_fact_count": len(set(written)),
            "unsupported_claims": list(audit.unsupported_claim_ids),
            "unwritten_source_facts": [
                {"fact_id": fact_id, "text": fact_map[fact_id].text, "reason": audit.fact_reasons.get(fact_id, "")}
                for fact_id in audit.missing_fact_ids
                if fact_id in fact_map
            ],
        },
        "ownership_integrity": {
            "correct": max(0, len(factual_claim_ids) - len(audit.ownership_errors)),
            "incorrect": len(audit.ownership_errors),
            "errors": list(audit.ownership_errors),
        },
        "structural_invariants": {
            "protected_anchor_count": len(protected_anchors),
            "missing_anchor_count": len(missing_anchors),
            "missing_anchors": missing_anchors,
            "violations": list(audit.violations),
        },
        "source_preservation": {
            "document_ids": sorted(graph.documents),
            "all_fact_spans_exact": True,
            "written_fact_ids": list(written),
            "omitted_fact_ids": list(audit.missing_fact_ids),
        },
        "model_contract": {
            "semantic_status": semantic.status,
            "semantic_response_batch_count": semantic.response_batch_count,
            "semantic_schema_valid_batch_count": semantic.schema_valid_batch_count,
            "semantic_raw_decision_count": semantic.raw_decision_count,
            "semantic_valid_decision_count": semantic.valid_decision_count,
            "semantic_invalid_decision_count": semantic.invalid_decision_count,
            "semantic_recovered_decision_count": semantic.recovered_decision_count,
            "semantic_invalid_atom_count": semantic.invalid_atom_count,
            "semantic_invalid_context_span_count": semantic.invalid_context_span_count,
            "semantic_fallback_count": len(semantic.fallback_fact_ids),
            "semantic_fail_closed_count": len(semantic.fail_closed_fact_ids),
            "semantic_fail_closed_fact_ids": list(semantic.fail_closed_fact_ids),
            "semantic_context_count": len(semantic.context_fact_ids),
            "semantic_context_spans": [
                {
                    "fact_id": fact.fact_id,
                    "base_fact_id": fact.base_fact_id,
                    "text": fact.text,
                    "classification": fact.classification,
                }
                for fact in graph.facts
                if fact.fact_id in set(semantic.context_fact_ids)
            ],
            "semantic_errors": list(semantic.errors),
            "realization_status": realization.status,
            "realization_error": realization.error,
            "realization_violations": list(realization.violations),
            "realization_unit_reports": [dict(report) for report in realization.unit_reports],
            "summary_status": (summary or {}).get("status", ""),
            "summary_violations": (summary or {}).get("violations") or [],
            "numeric_quarantine_count": len(numeric_quarantine or []),
            "numeric_quarantine": [dict(item) for item in (numeric_quarantine or [])],
        },
    }


def _additional_blob(resume_data: dict[str, Any]) -> str:
    """Flatten additional_sections so misrouted content stays visible."""

    sections = resume_data.get("additional_sections")
    if not isinstance(sections, dict):
        return ""
    parts: list[str] = []
    for values in sections.values():
        if isinstance(values, list):
            parts.extend(str(value) for value in values if value)
    return "\n".join(parts)


def _missing_fields(
    resume_data: dict[str, Any],
    *,
    degraded: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    framework = resume_data.get("framework")
    if isinstance(framework, dict):
        return [
            {"field": section["key"], "label": section["title"], "reason": "未提供可写入的个人事实，请按框架补充", "source": "not_provided"}
            for section in framework.get("sections", [])
            if isinstance(section, dict) and section.get("key") and section.get("title")
        ]
    degraded = degraded or {}
    meta = resume_data.get("meta") if isinstance(resume_data.get("meta"), dict) else {}
    additional = resume_data.get("additional_sections")
    additional = additional if isinstance(additional, dict) else {}
    additional_blob = _additional_blob(resume_data)
    checks = [
        ("name", "姓名", bool(meta.get("name"))),
        ("contact", "联系方式", bool(meta.get("phone") or meta.get("email"))),
        ("summary", "个人总结", bool(str(resume_data.get("summary") or "").strip())),
        ("education", "教育经历", bool(resume_data.get("education"))),
        (
            "experience_or_project",
            "工作、实习、项目或校园经历",
            any(resume_data.get(key) for key in ("experience", "projects", "research", "campus_experience", "teaching")),
        ),
        ("skills", "技能与资质", bool(resume_data.get("skills") or resume_data.get("certifications"))),
    ]
    # Three-state reporting: written (not reported) / not_rendered (material
    # had content but it failed verification or was routed to supplementary
    # sections) / not_provided (the material truly lacks it).
    not_rendered_reason = "材料中存在相关内容，但未能通过校验写入正文，已保留在补充信息中，请人工确认"
    missing: list[dict[str, Any]] = []
    for field, label, present in checks:
        if present:
            continue
        not_rendered = False
        if field == "summary":
            not_rendered = bool(degraded.get("summary")) or bool(additional_blob)
        elif field == "education":
            not_rendered = bool(additional.get("教育补充信息"))
        if not_rendered:
            missing.append({"field": field, "label": label, "reason": not_rendered_reason, "source": "not_rendered"})
        else:
            missing.append({"field": field, "label": label, "reason": "个人材料中未提供该信息，未进行推断", "source": "not_provided"})
    return missing


def run_v3_pipeline(
    *,
    cv_content: bytes = b"",
    cv_filename: str = "",
    cv_text: str = "",
    query_text: str = "",
    jd_text: str = "",
    template_mode: str = "none",
    target_role: str = "",
    use_llm: bool = True,
    candidate_cv_allowed: bool = True,
    cv_document_graph: DocumentGraph | None = None,
    semantic_llm_call: Any = None,
    realizer_llm_call: Any = None,
    summary_llm_call: Any = None,
) -> V3PipelineResult:
    """Execute V3 while keeping every trainable decision behind fixed schemas."""

    graphs = []
    if candidate_cv_allowed and cv_document_graph is not None:
        graphs.append(cv_document_graph)
    elif candidate_cv_allowed and cv_content:
        graphs.append(build_input_document_graph(
            cv_content,
            filename=cv_filename or "resume.bin",
            fallback_text=cv_text,
            source_id="cv",
            source_type="cv",
        ))
    elif candidate_cv_allowed and cv_text.strip():
        graphs.append(_native_graph("cv", "cv", cv_text))
    if query_text.strip():
        graphs.append(_native_graph("query", "query", query_text))
    if not graphs:
        graphs.append(_native_graph("cv", "cv", ""))

    policy = SourcePolicy()
    transport_graph = build_fact_graph(graphs, policy)
    semantic_result = compile_semantics(
        transport_graph,
        use_llm=use_llm,
        llm_call=semantic_llm_call,
    )
    graph = semantic_result.graph
    # R25: quarantine suspect OCR numerics (truncated years, implausible or
    # reversed periods, low-confidence digits) before planning — a corrupted
    # number must never be preserved verbatim into the resume.
    quarantined_numeric = quarantine_suspect_numeric_facts(graph)
    requirements = build_requirement_graph(jd_text)
    normalized_mode = template_mode if template_mode in {"tagged", "anchored", "style_only"} else "style_only"
    template = TemplateAST(mode=normalized_mode) if template_mode != "none" else None
    plan = plan_resume(graph, requirements, template)
    # R24 Phase 3: record-local realization.  A semantic fallback fact now
    # degrades only its own record/section unit; clean units keep constrained
    # LLM realization and failed units restore exact record-local source
    # sentences.  The hard per-unit verifier and the downstream atomic
    # verifier remain the final gates.
    realization = realize_record_local(
        plan,
        graph,
        use_llm=use_llm,
        llm_call=realizer_llm_call,
        degraded_fact_ids=(
            set(semantic_result.report.fallback_fact_ids)
            | set(semantic_result.report.fail_closed_fact_ids)
        ),
    )
    frozen = realization.frozen
    audit = audit_frozen_resume(frozen, graph, requirements)
    if not audit.clean:
        frozen = minimal_repair(frozen, audit, graph)
        audit = audit_frozen_resume(frozen, graph, requirements)
    # R24 Phase 4: the profile summary is compiled and re-verified as bound
    # atomic claims (no inferred tenure, no unsupported adjectives); it
    # survives only when the atomic verifier still accepts the frozen resume.
    summary_result = compile_summary(
        frozen,
        graph,
        requirements,
        use_llm=use_llm,
        llm_call=summary_llm_call,
    )
    frozen = summary_result.frozen
    audit = audit_frozen_resume(frozen, graph, requirements)
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
    resume_data = frozen_to_resume_data(frozen, graph, target_role=target_role)
    missing = _missing_fields(
        resume_data,
        degraded={"summary": summary_result.report.status}
        if summary_result.report.status not in {"generated", "revalidated", ""}
        else {},
    )
    # R24 Phase 4: concise reply compiled from the frozen audit — direction,
    # bounded evidence-backed match/gap analysis, exact missing fields,
    # grouped undetermined-ownership items, and real conflicts only.
    reply = build_reply(
        audit,
        graph,
        requirements,
        missing_fields=missing,
        skeleton=plan.skeleton,
        quarantined_numeric=quarantined_numeric,
    )
    output = V3Output(graph=graph, plan=plan, frozen=frozen, audit=audit, reply=reply)
    quality = _quality_report(output, semantic_result.report, realization.report, summary_result.report.to_dict(), quarantined_numeric)
    # Audit conflicts carry internal tokens ("cv:record:4:period").  Only
    # user-readable descriptions may surface; anything else is dropped.
    conflicts = [
        {"field": "source_conflict", "description": description}
        for description in _friendly_conflicts(list(audit.conflicts))
    ]
    changes = [
        {
            "path": "fact_graph",
            "action": "semantic_compile",
            "reason": f"固定 {semantic_result.report.schema_version} Schema；状态 {semantic_result.report.status}",
        },
        {
            "path": "resume",
            "action": "constrained_realize",
            "reason": f"仅使用计划 fact_id；状态 {realization.report.status}",
        },
    ]
    records = build_training_records(
        semantic_inputs=semantic_result.report.training_inputs,
        semantic_outputs=semantic_result.report.training_outputs,
        semantic_status=semantic_result.report.status,
        semantic_errors=semantic_result.report.errors,
        realizer_input=realization.report.training_input,
        realizer_output=realization.report.training_output,
        realizer_status=realization.report.status,
        realizer_violations=realization.report.violations,
    )
    training_path = maybe_write_training_trace(records)
    return V3PipelineResult(
        output=output,
        resume_data=resume_data,
        quality_report=quality,
        missing_fields=missing,
        conflicts=conflicts,
        changes=changes,
        reply_text=reply,
        semantic_report=semantic_result.report,
        realization_report=realization.report,
        training_trace_path=training_path,
    )


__all__ = ["V3PipelineResult", "run_v3_pipeline"]
