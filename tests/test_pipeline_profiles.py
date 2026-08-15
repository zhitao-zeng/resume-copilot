from pipeline_profiles import resolve_pipeline_profile
from v2_pipeline import (
    _fact_compiler_mode,
    _recover_grounded_source_structure,
    _recover_missing_record_facts,
    _restore_attested_source_summary,
)
from v2_schemas import CanonicalResume, Meta


def test_current_control_preserves_existing_environment(monkeypatch):
    monkeypatch.delenv("PIPELINE_PROFILE", raising=False)
    monkeypatch.setenv("FACT_COMPILER_MODE", "shadow")
    monkeypatch.setenv("LLM_NARRATIVE_QUERY_FASTPATH", "1")
    monkeypatch.delenv("LLM_NARRATIVE_CV_FASTPATH", raising=False)

    profile = resolve_pipeline_profile()

    assert profile.name == "current_control"
    assert profile.fact_compiler_mode == "shadow"
    assert profile.source_structure_recovery is True
    assert profile.record_fact_recovery is True
    assert profile.attested_summary_recovery is True
    assert profile.query_narrative is True
    assert profile.cv_narrative is False
    assert profile.record_compiler_recovery is False


def test_f507_compatible_disables_only_post_reference_content_mutations(monkeypatch):
    monkeypatch.setenv("PIPELINE_PROFILE", "f507_compatible")
    monkeypatch.setenv("FACT_COMPILER_MODE", "on")
    monkeypatch.setenv("LLM_NARRATIVE_QUERY_FASTPATH", "1")
    monkeypatch.setenv("LLM_NARRATIVE_CV_FASTPATH", "1")

    profile = resolve_pipeline_profile()

    assert profile.name == "f507_compatible"
    assert profile.fact_compiler_mode == "legacy"
    assert profile.source_structure_recovery is False
    assert profile.record_fact_recovery is False
    assert profile.attested_summary_recovery is False
    assert profile.query_narrative is False
    assert profile.cv_narrative is False
    assert profile.record_compiler_recovery is False


def test_candidate_is_an_explicit_grounded_bundle(monkeypatch):
    monkeypatch.setenv("PIPELINE_PROFILE", "candidate")
    monkeypatch.setenv("FACT_COMPILER_MODE", "legacy")

    profile = resolve_pipeline_profile()

    assert profile.name == "candidate"
    assert profile.fact_compiler_mode == "on"
    assert profile.source_structure_recovery is True
    assert profile.record_fact_recovery is True
    assert profile.attested_summary_recovery is True
    assert profile.query_narrative is True
    assert profile.cv_narrative is True
    assert profile.record_compiler_recovery is False


def test_quality_v2_preserves_composer_and_enables_only_record_recovery(monkeypatch):
    monkeypatch.setenv("PIPELINE_PROFILE", "quality_v2")
    monkeypatch.setenv("FACT_COMPILER_MODE", "on")

    profile = resolve_pipeline_profile()

    assert profile.name == "quality_v2"
    assert profile.fact_compiler_mode == "legacy"
    assert profile.record_compiler_recovery is True
    assert profile.structure_fields_only is True
    assert profile.source_structure_recovery is True
    assert profile.record_fact_recovery is False
    assert profile.attested_summary_recovery is False
    assert profile.query_narrative is False
    assert profile.cv_narrative is False


def test_ablation_profiles_change_one_named_content_layer_at_a_time(monkeypatch):
    expected = {
        "ledger_shadow": ("shadow", False, False),
        "local_repair": ("legacy", True, False),
        "fact_compiler": ("on", True, False),
        "candidate": ("on", True, True),
    }

    for name, (compiler, recovery, narrative) in expected.items():
        monkeypatch.setenv("PIPELINE_PROFILE", name)
        profile = resolve_pipeline_profile()
        assert profile.fact_compiler_mode == compiler
        assert profile.source_structure_recovery is recovery
        assert profile.record_fact_recovery is recovery
        assert profile.cv_narrative is narrative


def test_unknown_profile_falls_back_to_current_control(monkeypatch):
    monkeypatch.setenv("PIPELINE_PROFILE", "not-a-profile")
    monkeypatch.setenv("FACT_COMPILER_MODE", "on")

    profile = resolve_pipeline_profile()

    assert profile.name == "current_control"
    assert profile.fact_compiler_mode == "on"


def test_f507_compatible_makes_post_reference_recovery_functions_noops(monkeypatch):
    monkeypatch.setenv("PIPELINE_PROFILE", "f507_compatible")
    original = CanonicalResume(meta=Meta(name="当前结果"))
    fallback = CanonicalResume(meta=Meta(name="回填结果"))

    structured, structured_stats = _recover_grounded_source_structure(
        original, fallback, object(),
    )
    record, record_stats, changed = _recover_missing_record_facts(
        original, object(), [],
    )
    summary, restored = _restore_attested_source_summary(original, object())

    assert structured is original
    assert structured_stats.total == 0
    assert record is original
    assert record_stats.total == 0
    assert changed == set()
    assert summary is original
    assert restored == []
    assert _fact_compiler_mode() == "legacy"
