#!/usr/bin/env python3
"""R27 adapter-level remediation tests (tasks 3/4/6)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (str(REPO_ROOT), str(REPO_ROOT / "core")):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.v3.contracts import FactGraph, FrozenResume, RealizedClaim, SourceSpan, FactUnit  # noqa: E402
from core.v3.reply_builder import friendly_conflicts  # noqa: E402
from core.v3.resume_adapter import frozen_to_resume_data  # noqa: E402


def _graph_with_facts(texts: list[str]) -> FactGraph:
    doc = "\n".join(texts)
    facts = []
    cursor = 0
    for index, text in enumerate(texts):
        start = doc.index(text, cursor)
        cursor = start + len(text)
        facts.append(FactUnit(
            fact_id=f"cv:fact:{index}", source_id="cv", source_type="cv", text=text,
            spans=[SourceSpan(source_id="cv", char_start=start, char_end=start + len(text))],
        ))
    return FactGraph(documents={"cv": doc}, facts=facts)


def _claim(cid, section, text, fact_ids, record_id=None, field="bullet", group_id="g1"):
    return RealizedClaim(
        claim_id=cid, section=section, field=field, text=text,
        fact_ids=fact_ids, record_id=record_id, group_id=group_id,
    )


def _frozen(claims):
    sections: dict[str, list[RealizedClaim]] = {}
    for claim in claims:
        sections.setdefault(claim.section, []).append(claim)
    return FrozenResume(sections=sections, claims=claims)


def test_boilerplate_exact_match_only():
    graph = _graph_with_facts(["可应要求提供推荐信。", "负责薪酬面议谈判与定岗"])
    frozen = _frozen([
        _claim("c1", "additional", "可应要求提供推荐信。", ["cv:fact:0"]),
        _claim("c2", "additional", "负责薪酬面议谈判与定岗", ["cv:fact:1"]),
    ])
    data = frozen_to_resume_data(frozen, graph)
    rendered = str(data)
    assert "推荐信" not in rendered
    # 含「面议」片段的真实事实必须保留（整行精确匹配，不做子串误杀）
    assert "负责薪酬面议谈判与定岗" in rendered
    assert "补充信息" in str(data.get("additional_sections") or {})


def test_credential_fact_routes_to_certifications():
    graph = _graph_with_facts(["国际投资认证"])
    fact = graph.fact_map()["cv:fact:0"]
    fact.fact_type = "credential"
    frozen = _frozen([_claim("c1", "additional", "国际投资认证", ["cv:fact:0"])])
    data = frozen_to_resume_data(frozen, graph)
    assert "国际投资认证" in (data.get("certifications") or [])
    assert "国际投资认证" not in str(data.get("additional_sections") or {})


def test_metadata_kv_row_never_enters_experience():
    graph = _graph_with_facts(["Career Level: Junior", "Delivered results using structured workflows"])
    frozen = _frozen([
        _claim("c1", "experience", "Career Level: Junior", ["cv:fact:0"], record_id=None),
        _claim("c2", "experience", "Delivered results using structured workflows", ["cv:fact:1"], record_id=None),
        _claim("c3", "experience", "负责用户访谈", ["cv:fact:1"], record_id="cv:record:1"),
    ])
    data = frozen_to_resume_data(frozen, graph)
    experience_text = str(data.get("experience") or "")
    assert "Career Level" not in experience_text
    # 无归属但非 KV 形状的真实 bullet 保留在 experience
    assert "Delivered results" in experience_text
    assert "负责用户访谈" in experience_text
    # 元数据行的值不丢失
    assert "Junior" in str(data.get("additional_sections") or {})


def test_conflicts_hide_internal_codes_and_drop_unmappable():
    friendly = friendly_conflicts(["cv:record:4:period", "unassigned:metric", "garbage"])
    assert friendly == ["存在多处不一致的时间表述，请核对确认。"]
    assert all(":" not in item.split("表述")[0] for item in friendly)


# --- R28 task 2/2.5: markdown normalization and compound-title inheritance ---

import pytest  # noqa: E402

from core.v3.input_adapters import normalize_markdown_source  # noqa: E402
from core.v3.section_ontology import section_type as _sec_type  # noqa: E402
from core.v3.contracts import SourceAsset as _SA, SourcePolicy as _SP  # noqa: E402
from core.v3.document_graph import from_native_text as _fnt  # noqa: E402
from core.v3.fact_graph import build_fact_graph as _bfg  # noqa: E402

_MD_SKELETON = """# 张示例

## 工作经历

### 示例科技｜算法工程师｜2020.03 - 至今

#### 模型训练方向

- **搭建训练管线：** 完成数据清洗与评测闭环。
- 上线两个业务模型。

## 专业技能

- Python、PyTorch

<!--
私人备注：投递前再核对一遍数字。
-->
"""


def _md_graph():
    text = normalize_markdown_source(_MD_SKELETON)
    asset = _SA(source_id="cv", source_type="cv", filename="cv.md",
                media_type="text/markdown", text=text, native=True)
    return _bfg([_fnt(asset, text)], _SP()), text


def test_md_normalizer_strips_presentation_syntax():
    text = normalize_markdown_source(_MD_SKELETON)
    assert "**" not in text
    assert "<!--" not in text and "私人备注" not in text
    assert not any(line.startswith("#") for line in text.splitlines())


def test_md_deep_subheading_does_not_split_experience():
    graph, _ = _md_graph()
    sec_of = {s.section_id: s for s in graph.sections}
    exp_facts = [f for f in graph.facts
                 if getattr(sec_of.get(f.section_id), "section_type", "") == "experience"]
    assert exp_facts, "experience section must survive #### sub-headings"
    joined = "\n".join(f.text for f in exp_facts)
    assert "搭建训练管线" in joined
    assert "上线两个业务模型" in joined


def test_md_record_line_forms_record_with_company_and_period():
    graph, _ = _md_graph()
    titles = [getattr(r, "title", "") for r in graph.records]
    assert any("示例科技" in t and "2020.03" in t for t in titles)


def test_commented_content_never_reaches_facts():
    graph, _ = _md_graph()
    assert all("私人备注" not in f.text for f in graph.facts)


@pytest.mark.parametrize("title,expected", [
    ("科研项目经历", "projects"),
    ("主要工作经历", "experience"),
    ("经历", "other"),
    ("这是一句提到工作经历的正文长句子所以不该命中", "other"),
])
def test_compound_section_title_inherits_known_suffix(title, expected):
    assert _sec_type(title) == expected


# --- R28 task 3: record headers are compact labels, not sentences ---

from core.v3.fact_graph import _looks_like_record_header  # noqa: E402


@pytest.mark.parametrize("line", [
    "第四范式（4Paradigm）｜AI 算法工程师 → 语音方向 Tech Lead｜2025.02 - 至今",
    "伊利诺伊大学香槟分校（UIUC）｜预测分析与风险管理 · 理学硕士｜2022.08-2024.05",
    "2020.03 - 2023.06  示例科技  算法工程师",
    "示例科技 - 算法工程师（2020-2023）",
])
def test_record_header_shapes_are_detected(line):
    assert _looks_like_record_header(line, "experience")


@pytest.mark.parametrize("line", [
    # A ratio parses as month/year; narrative must not become a record.
    "局部 refinement 将 F1 0.6647→0.7319：新策略对其余 53/54 图输出保持不变；加速约 3.1 倍。",
    "在2021年至2022年期间参与了多个项目的研发工作，负责核心模块。",
])
def test_narrative_lines_are_not_record_headers(line):
    assert not _looks_like_record_header(line, "experience")


def test_list_markers_are_stripped_from_markdown():
    text = normalize_markdown_source("- 第一条\n1. 第二条\n* 第三条\n")
    assert text.splitlines() == ["第一条", "第二条", "第三条"]


# --- R28 task 6: presentation cleanup (feedback item 3) ---

from core.v3.text_integrity import is_junk_token, strip_ordinal_prefix  # noqa: E402


@pytest.mark.parametrize("raw,expected", [
    ("4. 设计相关拟合算法", "设计相关拟合算法"),
    ("1、负责数据清洗", "负责数据清洗"),
    ("① 主导项目评审", "主导项目评审"),
    ("一、工作职责说明", "工作职责说明"),
    ("(2) 参与架构评审", "参与架构评审"),
])
def test_source_ordinals_are_stripped(raw, expected):
    assert strip_ordinal_prefix(raw) == expected


@pytest.mark.parametrize("raw", ["3.1 倍加速", "2.5 万用户覆盖"])
def test_decimals_are_not_mistaken_for_ordinals(raw):
    assert strip_ordinal_prefix(raw) == raw


@pytest.mark.parametrize("value,junk", [
    ("32380b8d618fe5591XR639u_ElpVwI-_UPqb", True),
    ("github.com/example/repo", False),
    ("https://example.com/a1b2c3d4e5f6g7h8", False),
    ("someone@example.com", False),
    ("PP-OCRv6", False),
    ("负责端侧推理优化与部署", False),
])
def test_junk_token_detection_exempts_links_and_content(value, junk):
    assert is_junk_token(value) is junk


def test_contact_line_is_not_a_record_header():
    # Phone digits contain a year; a contact line must not open a record.
    assert not _looks_like_record_header("手机：19975260767 |", "projects")
    assert not _looks_like_record_header("邮箱：a@example.com ｜ 浙江温州", "projects")


# --- R28 task 7: education record lines are split into public fields ---

from core.v3.resume_adapter import _split_education_line  # noqa: E402


@pytest.mark.parametrize("line,school,period", [
    ("示例大学（Example）｜预测分析 · 理学硕士｜2022.08-2024.05｜GPA 4.0/4.0",
     "示例大学（Example）", "2022.08-2024.05"),
    ("示例学院｜数学与应用数学 · 理学学士｜2018.08-2022.06", "示例学院", "2018.08-2022.06"),
    ("2019-2023｜示例理工大学｜计算机科学", "示例理工大学", "2019-2023"),
])
def test_education_line_splits_into_fields(line, school, period):
    fields = _split_education_line(line)
    assert fields.get("school") == school
    assert fields.get("period") == period


def test_education_split_keeps_gpa_value_intact():
    fields = _split_education_line("示例大学｜理学硕士｜2020-2022｜GPA 4.0/4.0")
    assert "4.0/4.0" in " ".join(fields.values())


@pytest.mark.parametrize("line", ["示例大学", "在示例大学完成了硕士学业并获得学位"])
def test_unsplittable_education_line_returns_nothing(line):
    # The caller falls back to supplementary overflow rather than guessing.
    assert _split_education_line(line) == {}
