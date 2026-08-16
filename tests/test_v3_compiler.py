from core.v3.atomic_verifier import audit_frozen_resume
from core.v3.contracts import (
    CoverageLedger,
    FrozenResume,
    RealizerResponse,
    RealizedClaim,
    SourceAsset,
    TemplateAST,
)
from core.v3.document_graph import build_document_graph, from_native_text, from_ppstructure_blocks, route_asset
from core.v3.fact_graph import build_fact_graph
from core.v3.jd_graph import build_requirement_graph
from core.v3.orchestrator import run_v3
from core.v3.realizer import validate_realizer_response
from core.v3.repair import minimal_repair
import pytest
from pydantic import ValidationError


def test_source_span_is_exact_and_non_empty():
    asset = SourceAsset(source_id="cv", source_type="cv", text="甲公司", native=True)
    graph = from_native_text(asset)
    span = graph.nodes[0].source_spans[0]
    assert span.quote(graph.documents()) == "甲公司"


def test_native_first_router_does_not_force_ppstructure():
    docx = SourceAsset(source_id="cv", source_type="cv", filename="cv.docx", media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", native=True)
    pdf = SourceAsset(source_id="pdf", source_type="cv", filename="cv.pdf", media_type="application/pdf", native=True)
    image = SourceAsset(source_id="img", source_type="cv", filename="scan.png", media_type="image/png", native=False)
    assert route_asset(docx).engine == "native_docx"
    assert route_asset(pdf).engine == "native_pdf"
    assert route_asset(image).engine == "ppstructure"


def test_ppstructure_adapter_preserves_layout_metadata():
    asset = SourceAsset(source_id="scan", source_type="cv", filename="scan.png")
    graph = from_ppstructure_blocks(asset, [{
        "block_id": 4, "block_order": 2, "block_label": "text",
        "block_content": "负责用户访谈", "block_bbox": [1, 2, 30, 40],
        "page": 2, "column_id": "right", "region_id": "r3", "confidence": 0.87,
    }])
    node = graph.nodes[0]
    assert graph.extraction_engine == "ppstructure"
    assert (node.page, node.column_id, node.region_id, node.bbox, node.order, node.label) == (2, "right", "r3", (1.0, 2.0, 30.0, 40.0), 2, "text")
    assert node.confidence == 0.87


def test_ppstructure_multiblock_spans_are_exact_with_repeated_ids_and_whitespace():
    asset = SourceAsset(source_id="scan", source_type="cv", filename="scan.png")
    graph = from_ppstructure_blocks(asset, [
        {"block_id": 7, "block_content": "  负责运营  "},
        {"block_id": 7, "block_content": "输出报告"},
        {"block_id": 8, "block_content": "输出报告"},
    ], source_text="标题\n负责运营\n输出报告\n输出报告")
    assert [node.node_id for node in graph.nodes] == ["scan:pp:7", "scan:pp:7#1", "scan:pp:8"]
    assert all(node.source_spans[0].quote(graph.documents()) == node.text for node in graph.nodes)
    assert graph.source_text == "标题\n负责运营\n输出报告\n输出报告"


def test_ppstructure_falls_back_to_normalized_document_when_source_cannot_align():
    asset = SourceAsset(source_id="scan", source_type="cv", filename="scan.png")
    graph = from_ppstructure_blocks(asset, [
        {"block_id": 1, "block_content": "甲"},
        {"block_id": 2, "block_content": "乙"},
    ], source_text="甲\n丙")
    assert graph.source_text == "甲\n乙"
    assert all(node.source_spans[0].quote(graph.documents()) == node.text for node in graph.nodes)


def test_ppstructure_orders_blocks_and_tolerates_invalid_numeric_metadata():
    asset = SourceAsset(source_id="scan", source_type="cv", filename="scan.png")
    graph = from_ppstructure_blocks(asset, [
        {"block_id": 2, "block_order": "?", "page": "?", "block_content": "后"},
        {"block_id": 1, "block_order": 0, "page": 1, "block_content": "前"},
    ])
    assert [node.text for node in graph.nodes] == ["前", "后"]
    assert all(node.source_spans[0].quote(graph.documents()) == node.text for node in graph.nodes)


def test_native_text_defaults_to_one_page_and_accepts_explicit_pages():
    asset = SourceAsset(source_id="cv", source_type="cv", filename="cv.txt", text="a\nb\nc", native=True)
    assert [node.page for node in from_native_text(asset).nodes] == [1, 1, 1]
    assert [node.page for node in from_native_text(asset, page_numbers=[1, 2, 2]).nodes] == [1, 2, 2]


def test_reliable_native_graph_wins_even_when_shadow_blocks_are_supplied():
    asset = SourceAsset(source_id="cv", source_type="cv", filename="cv.docx", media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", text="原生段落", native=True)
    graph = build_document_graph(asset, text=asset.text, ppstructure_blocks=[{"block_id": 1, "block_content": "OCR替代"}], shadow_ppstructure=True)
    assert graph.extraction_engine == "native_docx"
    assert graph.nodes[0].text == "原生段落"
    assert graph.metadata["shadow_ppstructure_available"] is True
    forced = build_document_graph(asset, text=asset.text, ppstructure_blocks=[{"block_id": 1, "block_content": "OCR替代"}], force_ppstructure=True)
    assert forced.extraction_engine == "ppstructure"


def test_layout_ownership_prefers_region_and_abstains_on_ambiguous_region():
    asset = SourceAsset(source_id="scan", source_type="cv", filename="scan.png")
    blocks = [
        {"block_id": 1, "block_label": "heading", "block_content": "工作经历"},
        {"block_id": 2, "block_content": "甲公司 产品经理 2020-2021", "column_id": "left", "region_id": "a"},
        {"block_id": 3, "block_content": "乙公司 运营专员 2022-2023", "column_id": "right", "region_id": "b"},
        {"block_id": 4, "block_content": "负责访谈", "column_id": "left", "region_id": "a"},
        {"block_id": 5, "block_content": "负责运营", "column_id": "right", "region_id": "b"},
    ]
    graph = from_ppstructure_blocks(asset, blocks)
    fact_graph = build_fact_graph([graph])
    by_text = {fact.text: fact.record_id for fact in fact_graph.facts}
    assert by_text["负责访谈"] != by_text["负责运营"]
    ambiguous = from_ppstructure_blocks(asset, [
        {"block_id": 1, "block_label": "heading", "block_content": "工作经历"},
        {"block_id": 2, "block_content": "负责未知", "region_id": "left"},
        {"block_id": 3, "block_content": "负责另一个未知", "region_id": "right"},
    ])
    ambiguous_facts = build_fact_graph([ambiguous]).facts
    assert all(fact.record_id is None for fact in ambiguous_facts)


def test_requirement_spans_quote_trimmed_text_exactly():
    text = "岗位要求\n  Python、项目管理\n\t负责用户运营"
    graph = build_requirement_graph(text)
    assert all(req.source_span and req.source_span.quote({"jd": text}) == req.text for req in graph.requirements)


def test_contracts_reject_invalid_coverage_and_freezing():
    with pytest.raises(ValidationError):
        CoverageLedger(eligible_fact_ids=["f1"], planned_fact_ids=["f1"], omitted_fact_ids=["f1"], omission_reasons={})
    claim = RealizedClaim(claim_id="c", section="experience", text="x", fact_ids=[])
    with pytest.raises(ValidationError):
        FrozenResume(sections={"experience": [claim]}, claims=[claim, claim])
    changed = claim.model_copy(update={"text": "different"})
    with pytest.raises(ValidationError):
        FrozenResume(sections={"experience": [changed]}, claims=[claim])


def test_realizer_protocol_rejects_unknown_fact_and_factless_non_placeholder():
    result = run_v3(cv_text="甲公司 产品经理 2020-2021")
    response = RealizerResponse(request_fact_ids=[], claims=[RealizedClaim(claim_id="x", section="experience", text="编造经历")])
    violations = validate_realizer_response(response, result.graph)
    assert "x:factless_non_placeholder" in violations
    assert validate_realizer_response(RealizerResponse(request_fact_ids=["unknown"], claims=[RealizedClaim(claim_id="y", section="experience", text="甲公司", fact_ids=["unknown"])]), result.graph)
    fact_id = result.graph.eligible_fact_ids[0]
    escalated = validate_realizer_response(
        RealizerResponse(request_fact_ids=[fact_id], claims=[]),
        result.graph,
        allowed_fact_ids=[],
    )
    assert "declared_request_fact_ids_mismatch" in escalated


def test_jd_is_not_candidate_fact_and_jd_only_returns_framework():
    result = run_v3(jd_text="招聘岗位：产品经理\n要求：Python、项目管理")
    assert result.plan.skeleton is True
    assert result.frozen.claims
    assert all(not claim.fact_ids for claim in result.frozen.claims)
    assert "产品经理" not in "\n".join(claim.text for claim in result.frozen.claims)
    assert "来自JD" in result.reply
    assert not result.graph.eligible_facts()


def test_four_input_scenarios_are_executable_without_cross_source_pollution():
    cv_jd = run_v3(cv_text="工作经历\n甲公司 产品经理 2020-2021", jd_text="要求：项目管理")
    cv_only = run_v3(cv_text="教育经历\n甲大学 计算机科学")
    query_jd = run_v3(query_text="我在乙公司担任老师", jd_text="岗位要求：教学经验")
    jd_only = run_v3(jd_text="岗位要求：医生资格")
    assert cv_jd.graph.eligible_facts() and cv_jd.plan.skeleton is False
    assert cv_only.graph.eligible_facts()
    assert any("乙公司" in fact.text for fact in query_jd.graph.eligible_facts())
    assert jd_only.plan.skeleton is True
    for result in (cv_jd, cv_only, query_jd, jd_only):
        assert set(result.audit.written_fact_ids).isdisjoint(result.audit.missing_fact_ids)


def test_query_fact_and_intent_are_separated():
    result = run_v3(query_text="我在甲公司担任产品经理，负责用户访谈。请帮我优化简历")
    texts = {fact.text: fact for fact in result.graph.facts}
    eligible = [fact for fact in result.graph.eligible_facts() if fact.source_type == "query"]
    assert any("甲公司" in fact.text for fact in eligible)
    assert any("优化简历" in fact.text and not fact.eligible for fact in result.graph.facts)
    assert all("优化简历" not in fact.text for fact in eligible)


def test_structured_query_keeps_record_ownership_when_cv_facts_precede_it():
    result = run_v3(
        cv_text="教育经历\n甲大学",
        query_text="工作经历\n乙公司 教师 2021-2024\n负责课程设计\n请帮我优化表达",
    )
    query_facts = [fact for fact in result.graph.facts if fact.source_type == "query"]
    owned = [fact for fact in query_facts if fact.eligible and fact.record_id]
    assert owned
    records = {record.record_id: record for record in result.graph.records}
    assert all(fact.fact_id in records[fact.record_id].fact_ids for fact in owned)
    assert all("优化表达" not in fact.text for fact in owned)


def test_records_do_not_cross_contaminate():
    text = "工作经历\n甲公司 产品经理 2020-2021\n负责用户访谈\n乙公司 运营专员 2022-2023\n负责活动运营"
    result = run_v3(cv_text=text)
    records = {record.record_id: record for record in result.graph.records}
    assert len(records) == 2
    record_claims = [claim for claim in result.frozen.claims if claim.record_id]
    # Deterministic realization is now atomic at the source-unit boundary:
    # each header/action remains independently bindable instead of becoming
    # one giant record paragraph.
    assert len(record_claims) == 4
    assert {claim.record_id for claim in record_claims} == set(records)
    for claim in record_claims:
        assert all(result.graph.fact_map()[fact_id].record_id == claim.record_id for fact_id in claim.fact_ids)


def test_unassigned_facts_keep_source_scope_and_sections_keep_resume_order():
    result = run_v3(
        cv_text="联系方式\n13800138000\n工作经历\n负责客户沟通",
        query_text="本人完成用户调研",
        jd_text="要求：用户调研",
    )
    unassigned_groups = [group for group in result.plan.groups if group.record_id is None and group.fact_ids]
    assert all(len({result.graph.fact_map()[fact_id].source_id for fact_id in group.fact_ids}) == 1 for group in unassigned_groups)
    sections = [group.section for group in result.plan.groups]
    assert sections.index("contact") < sections.index("experience")
    assert result.plan.ledger.written_fact_ids == result.audit.written_fact_ids
    assert result.plan.ledger.omitted_fact_ids == result.audit.missing_fact_ids


def test_repair_never_uses_an_unrelated_unassigned_fact_as_fallback():
    result = run_v3(cv_text="负责甲事项\n负责乙事项")
    bad = RealizedClaim(claim_id="bad", section="other", text="编造", fact_ids=["missing"], record_id=None)
    frozen = FrozenResume(sections={"other": [bad]}, claims=[bad])
    audit = audit_frozen_resume(frozen, result.graph)
    repaired = minimal_repair(frozen, audit, result.graph)
    assert repaired.claims == []


def test_template_text_cannot_leak_into_resume():
    result = run_v3(
        cv_text="李四\n教育经历\n甲大学 计算机科学",
        template=TemplateAST(mode="anchored", sample_text=["模板示例姓名 John Old", "Old Company"]),
    )
    output = "\n".join(claim.text for claim in result.frozen.claims)
    assert "John Old" not in output
    assert "Old Company" not in output
    assert "甲大学" in output


def test_audit_does_not_report_written_fact_as_missing():
    result = run_v3(cv_text="甲公司 产品经理 2020-2021")
    assert set(result.audit.written_fact_ids).isdisjoint(result.audit.missing_fact_ids)
    for fact_id in result.audit.written_fact_ids:
        assert fact_id not in result.audit.missing_fact_ids


def test_critical_anchor_missing_is_a_violation():
    result = run_v3(cv_text="甲公司 产品经理 2020-2021")
    fact = result.graph.eligible_facts()[0]
    claim = RealizedClaim(claim_id="bad", section="experience", text="不相关内容", fact_ids=[fact.fact_id], record_id=fact.record_id)
    frozen = FrozenResume(sections={"experience": [claim]}, claims=[claim])
    audit = audit_frozen_resume(frozen, result.graph)
    assert "bad" in audit.unsupported_claim_ids
