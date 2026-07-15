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
