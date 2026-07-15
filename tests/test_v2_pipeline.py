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


# ── Verifier tests ──

def test_conservative_fallback_empty():
    from resume_verifier import conservative_fallback
    result = conservative_fallback()
    assert result.resume.meta.name == ""
    assert result.resume.education == []
    assert len(result.changes) == 1


def test_verifier_returns_fallback_on_no_llm():
    from resume_verifier import verify_resume
    from v2_schemas import SourceBundle, SourceBlock, DraftResume
    source = SourceBundle(blocks=[])
    draft = DraftResume()
    result = verify_resume(source, draft)
    assert result.resume.meta.name == ""


# ── Validator tests ──

def test_validator_removes_empty_education():
    from v2_validator import validate_resume
    from v2_schemas import CanonicalResume, Meta, Education
    resume = CanonicalResume(
        meta=Meta(name="张三"),
        education=[
            Education(school="", degree="", major="", period=""),
            Education(school="北京大学", degree="", major="", period=""),
        ],
    )
    result = validate_resume(resume)
    assert len(result.education) == 1
    assert result.education[0].school == "北京大学"


def test_validator_deduplicates_projects():
    from v2_validator import validate_resume
    from v2_schemas import CanonicalResume, Project
    resume = CanonicalResume(
        projects=[
            Project(name="智能家居", organization="", role="", period=""),
            Project(name="智能家居", organization="", role="", period=""),
            Project(name="OCR识别", organization="", role="", period=""),
        ],
    )
    result = validate_resume(resume)
    assert len(result.projects) == 2
