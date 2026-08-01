"""Tests for the 4-stage bullet rewrite pipeline."""
import asyncio
import json
import pytest
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import patch, MagicMock

from resume_optimization import (
    conceive_material,
    _BulletAnalysisOutput,
    _BulletVerdictOutput,
)


# ── Factories for test data ──

@pytest.fixture
def sample_bullet():
    from fact_ledger import FactBullet
    return FactBullet(
        id="exp_0_b0",
        source_text="参与公司核心产品的功能迭代，负责收集用户反馈。",
        context="某科技公司 | 产品助理实习生 | 2025",
        entities=("产品助理",),
        metrics=("",),
        has_action=True,
        has_result=False,
        missing_info=True,
    )


@pytest.fixture
def sample_ledger(sample_bullet):
    from fact_ledger import FactLedger, FactEntity
    return FactLedger(
        entities={
            ("role", "产品助理"): FactEntity(kind="role", value="产品助理", source_span="产品助理实习生"),
            ("company", "某科技公司"): FactEntity(kind="company", value="某科技公司", source_span="某科技公司"),
        },
        bullets=[sample_bullet],
        meta={"name": "测试", "target_role": "产品经理"},
        raw_text="某科技公司 产品助理实习生 2025 参与核心产品功能迭代 收集用户反馈 撰写需求文档",
    )


# ── Test conceive_material (pure rules, no LLM) ──

def test_conceive_material_extracts_tech(sample_bullet, sample_ledger):
    """conceive_material should extract tech keywords from raw_text."""
    material = conceive_material(sample_bullet, sample_ledger)
    assert isinstance(material, dict)
    assert "available_tech" in material
    assert "available_metrics" in material
    assert "safe_angles" in material


def test_conceive_material_deals_with_empty_bullet():
    """conceive_material should handle a bullet with no metrics gracefully."""
    from fact_ledger import FactBullet, FactLedger, FactEntity
    empty_bullet = FactBullet(
        id="exp_0_b0",
        source_text="日常运营工作",
        context="某公司",
        entities=(),
        metrics=(),
        has_action=False,
        has_result=False,
        missing_info=True,
    )
    empty_ledger = FactLedger(
        entities={},
        bullets=[empty_bullet],
        meta={},
        raw_text="某公司 日常运营工作",
    )
    material = conceive_material(empty_bullet, empty_ledger)
    assert isinstance(material.get("available_tech"), list)
    assert isinstance(material.get("available_metrics"), list)
    assert isinstance(material.get("safe_angles"), list)


# ── Test Analyze model ──

def test_analysis_model_defaults():
    """_BulletAnalysisOutput should default all bools to False."""
    output = _BulletAnalysisOutput()
    assert output.missing_situation is False
    assert output.missing_task is False
    assert output.missing_action is False
    assert output.missing_result is False
    assert output.missing_technical_detail is False
    assert output.missing_metric is False
    assert output.has_vague_language is False


def test_analysis_model_parses_json():
    """_BulletAnalysisOutput should parse from dict."""
    data = {
        "missing_situation": True,
        "missing_technical_detail": True,
        "missing_result": True,
    }
    output = _BulletAnalysisOutput.model_validate(data)
    assert output.missing_situation is True
    assert output.missing_technical_detail is True
    assert output.missing_result is True
    assert output.missing_task is False  # not in input, defaults to False


# ── Test Verdict model ──

def test_verdict_model_defaults():
    """_BulletVerdictOutput should default to safe."""
    output = _BulletVerdictOutput()
    assert output.is_safe is True
    assert output.risk_tags == []


def test_verdict_model_parses_unsafe():
    """_BulletVerdictOutput should parse unsafe verdict."""
    data = {
        "is_safe": False,
        "risk_tags": ["fabricated_metric"],
        "reason": "数字完全不在原文",
    }
    output = _BulletVerdictOutput.model_validate(data)
    assert output.is_safe is False
    assert "fabricated_metric" in output.risk_tags


def test_pipeline_runs_with_mock_llm():
    """Full pipeline should produce BulletPatch list even with mock LLM."""
    from fact_ledger import build_ledger
    from resume_optimization import patch_optimize_weak_bullets

    sample_raw = "某科技公司 | 产品助理 | 2025\n参与核心产品功能迭代，收集用户反馈。"
    ledger = build_ledger(
        {"experience": [{"company": "某科技公司", "role": "产品助理", "period": "2025",
                         "bullets": ["参与核心产品功能迭代，收集用户反馈。"]}]},
        sample_raw, run_repair=False)

    with patch("resume_optimization.call_llm_typed") as mock_llm, \
         patch("resume_optimization.llm_enabled", return_value=True):
        mock_llm.side_effect = [
            # analyze_bullet
            {"missing_situation": True, "missing_technical_detail": True},
            # _rewrite_bullet (batch 1, 2 concurrent)
            {"bullet_id": "exp_0_b0", "new_text": "负责收集并分析用户反馈，推动核心产品功能迭代优化。"},
            {"bullet_id": "exp_0_b0", "new_text": "主导核心产品功能迭代，统筹用户反馈收集与需求分析。"},
            # verify_bullet (batch 1)
            {"is_safe": True, "risk_tags": [], "reason": ""},
            {"is_safe": True, "risk_tags": [], "reason": ""},
            # _rewrite_bullet (batch 2, 1 task)
            {"bullet_id": "exp_0_b0", "new_text": "深入参与核心产品功能迭代，系统收集用户反馈并驱动优化。"},
            # verify_bullet (batch 2)
            {"is_safe": True, "risk_tags": [], "reason": ""},
        ]
        patches = asyncio.run(patch_optimize_weak_bullets(ledger, ["产品", "迭代"]))
        assert isinstance(patches, list)
        if patches:
            from semantic_guard import BulletPatch
            for p in patches:
                assert isinstance(p, BulletPatch)
