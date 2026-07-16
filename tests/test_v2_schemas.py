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
