#!/usr/bin/env python3
"""R25 sentence-level joining of OCR soft-wrapped fragments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "core")):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.v3.pipeline import run_v3_pipeline  # noqa: E402
from core.v3.training_schema import SCHEMA_VERSION  # noqa: E402


def _fallback_semantic(*_args, **_kwargs):
    """Force the deterministic path for every fact."""

    return {}


def test_soft_wrapped_lines_join_into_one_sentence():
    # OCR 把一句话折成两行：左行无终止标点，右行是延续。
    cv = (
        "甲公司 后端工程师 2019年1月 - 2021年6月\n"
        "制定增长机会战略并确立在不断变化的商业环\n"
        "境中的独特定位\n"
        "输出需求报告。\n"
    )
    result = run_v3_pipeline(
        cv_text=cv,
        semantic_llm_call=_fallback_semantic,
        use_llm=False,
    )
    bullets = [claim.text for claim in result.output.frozen.claims]
    joined = [text for text in bullets if "商业环境中的独特定位" in text]
    assert joined, f"soft-wrapped fragments must rejoin, got: {bullets}"
    assert not any(text.rstrip().endswith("商业环") for text in bullets)
    # 完整句子终止于句号的事实保持独立
    assert any("输出需求报告" in text for text in bullets)
    assert result.quality_report["atomic_factuality"]["recall"] == 1.0


def test_list_items_are_not_joined():
    # 两条都以终止标点结尾的独立事项不得合并。
    cv = (
        "甲公司 后端工程师 2019年1月 - 2021年6月\n"
        "负责订单系统重构。\n"
        "搭建数据仓库。\n"
    )
    result = run_v3_pipeline(
        cv_text=cv,
        semantic_llm_call=_fallback_semantic,
        use_llm=False,
    )
    texts = [claim.text for claim in result.output.frozen.claims]
    assert any("负责订单系统重构" in text for text in texts)
    assert any("搭建数据仓库" in text for text in texts)
    assert not any("重构。搭建" in text for text in texts)


def test_label_prefixed_line_is_not_joined_upward():
    # 右行带标签前缀（技术：…）时不能并入上一行。
    cv = (
        "甲公司 后端工程师 2019年1月 - 2021年6月\n"
        "远程\n"
        "技术：ReactJS、TypeScript\n"
    )
    result = run_v3_pipeline(
        cv_text=cv,
        semantic_llm_call=_fallback_semantic,
        use_llm=False,
    )
    texts = [claim.text for claim in result.output.frozen.claims]
    assert not any("远程技术" in text.replace(" ", "") for text in texts)


def test_merge_preserves_verifier_invariants():
    cv = (
        "甲公司 后端工程师 2019年1月 - 2021年6月\n"
        "制定增长机会战略并确立在不断变化的商业环\n"
        "境中的独特定位\n"
    )
    result = run_v3_pipeline(
        cv_text=cv,
        semantic_llm_call=_fallback_semantic,
        use_llm=False,
    )
    assert result.output.audit.clean
    assert result.quality_report["atomic_factuality"]["precision"] == 1.0
    assert result.quality_report["atomic_factuality"]["recall"] == 1.0
