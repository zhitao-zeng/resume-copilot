"""Tests for V2 schemas."""
import pytest
from pydantic import ValidationError


def test_source_block_requires_block_id():
    from v2_schemas import SourceBlock
    SourceBlock(block_id="b1", source_type="resume", text="陈媛媛 Abbey")
    # must not raise


def test_source_block_rejects_extra_field():
    from v2_schemas import SourceBlock
    with pytest.raises(ValidationError):
        SourceBlock(block_id="b1", source_type="resume", text="test", unknown_field="x")


def test_source_block_rejects_invalid_source_type():
    from v2_schemas import SourceBlock
    with pytest.raises(ValidationError):
        SourceBlock(block_id="b1", source_type="weibo", text="test")


def test_source_bundle_accepts_blocks():
    from v2_schemas import SourceBundle, SourceBlock
    b = SourceBundle(blocks=[
        SourceBlock(block_id="b1", source_type="resume", text="hello"),
    ])
    assert len(b.blocks) == 1


def test_draft_field_defaults():
    from v2_schemas import DraftField
    f = DraftField()
    assert f.value is None
    assert f.mode == "none"
    assert f.evidence == []


def test_draft_field_rejects_bad_mode():
    from v2_schemas import DraftField
    with pytest.raises(ValidationError):
        DraftField(mode="invalid")


def test_evidence_ref_rejects_extra():
    from v2_schemas import EvidenceRef
    with pytest.raises(ValidationError):
        EvidenceRef(block_id="b1", quote="hello", extra=True)


def test_verified_result_has_resume_and_changes():
    from v2_schemas import VerifiedResult, CanonicalResume, Change
    r = VerifiedResult(
        resume=CanonicalResume(meta={"name": ""}, education=[], experiences=[], projects=[],
                               skills={"languages": [], "frameworks": [], "tools": [], "domains": []},
                               summary=""),
        changes=[Change(path="edu[0]", action="remove", reason="不真实")],
    )
    assert len(r.changes) == 1


# ── SourceAdapter tests ──

def test_build_source_bundle_from_cv_text():
    from source_adapter import build_source_bundle
    bundle = build_source_bundle(
        cv_text="陈媛媛Abbey 188-8888-8888\n工作经历\n超级公司",
        query_text="帮我的简历优化",
        jd_text="",
    )
    assert len(bundle.blocks) >= 2
    resume_blocks = [b for b in bundle.blocks if b.source_type == "resume"]
    assert any("陈媛媛" in b.text for b in resume_blocks)
    query_blocks = [b for b in bundle.blocks if b.source_type == "query"]
    assert len(query_blocks) == 1


def test_source_block_id_unique():
    from source_adapter import build_source_bundle
    bundle = build_source_bundle(cv_text="line1\nline2", query_text="q", jd_text="")
    ids = [b.block_id for b in bundle.blocks]
    assert len(ids) == len(set(ids))


def test_build_source_bundle_empty_cv():
    from source_adapter import build_source_bundle
    bundle = build_source_bundle(cv_text="", query_text="query only", jd_text="")
    assert len(bundle.blocks) >= 1


def test_source_bundle_preserves_exact_document_offsets_and_query_clauses():
    from source_adapter import build_source_bundle

    cv_text = "  工作经历  \n甲公司｜产品助理\n负责访谈，输出需求清单"
    query = "我在乙公司担任运营专员，但是请帮我优化简历。不要写虚构结果"
    jd = "岗位要求\n负责用户运营"
    bundle = build_source_bundle(cv_text, query, jd)
    documents = {item.source_id: item.text for item in bundle.documents}

    assert documents == {"resume": cv_text, "query": query, "jd": jd}
    for block in bundle.blocks:
        assert block.source_spans
        for span in block.source_spans:
            assert documents[span.source_id][span.char_start:span.char_end]

    query_blocks = [block for block in bundle.blocks if block.source_type == "query"]
    assert [block.text for block in query_blocks] == [
        "我在乙公司担任运营专员",
        "请帮我优化简历",
        "不要写虚构结果",
    ]
    assert all(
        documents[span.source_id][span.char_start:span.char_end] == block.text
        for block in query_blocks
        for span in block.source_spans
    )


def test_fact_ledger_quotes_are_exact_and_candidate_eligibility_is_bounded():
    from source_adapter import build_source_bundle

    query = "我在仁和医院担任住院医师，负责患者诊疗；请帮我优化简历"
    jd = "岗位要求：三年以上临床经验"
    bundle = build_source_bundle("", query, jd)
    documents = {item.source_id: item.text for item in bundle.documents}

    assert bundle.fact_units
    for fact in bundle.fact_units:
        reconstructed = "".join(
            documents[span.source_id][span.char_start:span.char_end]
            for span in fact.source_spans
        )
        assert reconstructed == fact.verbatim_text

    eligible = [fact.verbatim_text for fact in bundle.fact_units if fact.fact_eligible]
    ineligible = [fact.verbatim_text for fact in bundle.fact_units if not fact.fact_eligible]
    assert any("仁和医院" in value for value in eligible)
    assert any("患者诊疗" in value for value in eligible)
    assert not any("优化简历" in value for value in eligible)
    assert any("临床经验" in value for value in ineligible)


def test_resume_title_and_fact_disclaimer_are_not_candidate_fact_units():
    from source_adapter import build_source_bundle

    bundle = build_source_bundle(
        "张晨简历\n个人总结\n过往经历以真实岗位职责和结果为准。\n"
        "工作经历\n甲公司｜产品经理\n负责用户访谈。",
        "",
        "",
    )

    all_values = [fact.verbatim_text for fact in bundle.fact_units]
    eligible = [fact.verbatim_text for fact in bundle.fact_units if fact.fact_eligible]
    assert "张晨简历" not in all_values
    assert any("真实岗位职责" in value for value in all_values)
    assert not any("真实岗位职责" in value for value in eligible)
    assert any("用户访谈" in value for value in eligible)


def test_coalesced_resume_block_retains_every_physical_source_span():
    from source_adapter import build_source_bundle

    cv_text = "工作经历\n甲公司｜产品经理\n负责用户研究与需求分析\n并输出版本规划。"
    bundle = build_source_bundle(cv_text, "", "")
    documents = {item.source_id: item.text for item in bundle.documents}
    coalesced = next(block for block in bundle.blocks if "版本规划" in block.text)

    physical_parts = [
        documents[span.source_id][span.char_start:span.char_end]
        for span in coalesced.source_spans
    ]
    assert "负责用户研究与需求分析" in physical_parts
    assert "并输出版本规划。" in physical_parts
    assert len(coalesced.origin_block_ids) == len(coalesced.source_spans)


def test_evidence_binding_exposes_additive_fact_and_span_provenance():
    from evidence_binding import bind_resume_evidence
    from source_adapter import build_source_bundle
    from v2_schemas import CanonicalResume

    source = build_source_bundle(
        "工作经历\n甲公司｜产品助理\n负责用户访谈，输出需求清单", "", "",
    )
    resume = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "产品助理",
            "bullets": ["负责用户访谈，输出需求清单"],
        }],
    })
    bindings = bind_resume_evidence(resume, source)
    bullet = next(item for item in bindings if ".bullets[" in item.path)

    assert bullet.fact_ids
    assert bullet.source_spans
    fact_ids = {fact.fact_id for fact in source.fact_units}
    assert set(bullet.fact_ids) <= fact_ids


# ── Evidence integrity tests (removed - function no longer used) ──

def NOT_test_evidence_exists_valid():
    from v2_schemas import SourceBlock, EvidenceRef
    from resume_composer import evidence_exists
    blocks = [SourceBlock(block_id="b1", source_type="resume", text="陈媛媛 Abbey")]
    ref = EvidenceRef(block_id="b1", quote="陈媛媛")
    assert evidence_exists(ref, blocks)


def NOT_test_evidence_exists_invalid_block_id():
    from v2_schemas import EvidenceRef
    from resume_composer import evidence_exists
    ref = EvidenceRef(block_id="b_nonexist", quote="text")
    assert not evidence_exists(ref, [])


def NOT_test_evidence_exists_quote_missing():
    from v2_schemas import SourceBlock, EvidenceRef
    from resume_composer import evidence_exists
    blocks = [SourceBlock(block_id="b1", source_type="resume", text="陈媛媛 Abbey")]
    ref = EvidenceRef(block_id="b1", quote="北京大学")
    assert not evidence_exists(ref, blocks)
