#!/usr/bin/env python3
"""R27 text-integrity guard fixtures (real defect strings from the audit)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "core")):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.v3.text_integrity import bullet_defects  # noqa: E402


def test_fragment_defects():
    assert bullet_defects("政支持。") == ["bare_fragment"]
    assert bullet_defects("，顾问及") == ["fragment_start", "bare_fragment"]
    assert "unbalanced_bracket" in bullet_defects("（公司】")


def test_source_bullet_markers_are_not_fragments():
    # "- "/"· " 是源文的项目符号，是排版不是碎片（否则正常 bullet 被误判）
    assert bullet_defects("- 负责订单系统重构") == []
    assert bullet_defects("· 搭建数据仓库") == []
    assert bullet_defects("- Delivered results using structured workflows") == []


def test_high_frequency_normal_sentences_never_killed():
    # 否决项：这三条是中文简历最高频的正常表达，误杀一条即不通过。
    assert bullet_defects("负责项目全流程管理与交付") == []
    assert bullet_defects("管理人员规模 12 人") == []
    assert bullet_defects("负责例会组织与会议纪要输出") == []
    assert bullet_defects("管理条例修订与合规审查") == []


def test_legitimate_sentences_and_short_values_pass():
    assert bullet_defects("协助组织社区活动并为团队提供行政支持") == []
    assert bullet_defects("协助组织社区活动并为团队提供行政支持。") == []
    assert bullet_defects("Excel") == []
    assert bullet_defects("2022年至今") == []
    assert bullet_defects("7个月") == []
    assert bullet_defects("负责订单系统重构，使用 Python 重写核心服务，接口耗时降低 30%") == []


def test_bracket_balance():
    assert bullet_defects("副总裁（收入1.1亿美元）") == []
    assert "unbalanced_bracket" in bullet_defects("副总裁（收入1.1亿美元")
    assert "unbalanced_bracket" in bullet_defects("副总裁，投资策略，亚洲（收入1.1亿美元>副总裁")
