"""Tests for V2 pipeline components (Composer, Verifier, Validator)."""
import pytest


def test_composer_returns_empty_draft_on_no_llm():
    from resume_composer import compose_resume
    from v2_schemas import SourceBundle, SourceBlock
    source = SourceBundle(blocks=[
        SourceBlock(block_id="cv_0", source_type="resume", text="陈媛媛 产品经理"),
    ])
    draft = compose_resume(source)
    assert draft.meta.name == ""
    assert draft.meta.target_role == ""


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


def test_validator_preserves_substantive_unnamed_projects():
    from v2_validator import validate_resume
    from v2_schemas import CanonicalResume, Project

    resume = CanonicalResume(projects=[
        Project(
            organization="星河科技",
            role="产品负责人",
            period="2023.03-2023.08",
            bullets=["完成10次用户访谈并输出需求优先级清单。"],
        ),
        Project(
            organization="远山公益",
            role="志愿者",
            period="2022.07",
            bullets=["组织社区数字技能培训。"],
        ),
    ])

    result = validate_resume(resume)

    assert len(result.projects) == 2
    assert result.projects[0].organization == "星河科技"
    assert result.projects[1].organization == "远山公益"


def test_validator_deduplicates_identical_unnamed_projects_and_drops_empty():
    from v2_validator import validate_resume
    from v2_schemas import CanonicalResume, Project

    project = Project(
        organization="星河科技",
        role="产品负责人",
        period="2023.03-2023.08",
        bullets=["完成10次用户访谈并输出需求优先级清单。"],
    )
    resume = CanonicalResume(projects=[
        project,
        project.model_copy(deep=True),
        Project(),
        Project(bullets=["   "]),
    ])

    result = validate_resume(resume)

    assert result.projects == [project]
