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
from core.v3.reply_builder import _friendly_conflicts  # noqa: E402
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
    friendly = _friendly_conflicts(["cv:record:4:period", "unassigned:metric", "garbage"])
    assert friendly == ["存在多处不一致的时间表述，请核对确认。"]
    assert all(":" not in item.split("表述")[0] for item in friendly)
