"""Tests for V2 pipeline components (Composer, Verifier, Validator)."""
import pytest


def test_composer_returns_empty_draft_on_no_llm():
    from resume_composer import compose_resume
    from v2_schemas import SourceBundle, SourceBlock
    source = SourceBundle(blocks=[
        SourceBlock(block_id="cv_0", source_type="resume", text="陈媛媛 产品经理"),
    ])
    draft = compose_resume(source)
    assert draft.meta.name.value is None


def test_evidence_exists():
    from resume_composer import evidence_exists
    from v2_schemas import SourceBlock, EvidenceRef
    blocks = [SourceBlock(block_id="b1", source_type="resume", text="北京大学硕士")]
    assert evidence_exists(EvidenceRef(block_id="b1", quote="北京大学"), blocks)
    assert not evidence_exists(EvidenceRef(block_id="b1", quote="清华大学"), blocks)
    assert not evidence_exists(EvidenceRef(block_id="b_none", quote="test"), blocks)
