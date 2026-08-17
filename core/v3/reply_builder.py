"""Concise user-facing reply compiled from the frozen audit only (R24 Phase 4).

Structure (roughly 250-600 Chinese characters by design, never a hard cut):

- 生成方向: one or two sentences, no resume echo.
- 岗位匹配 (JD supplied): up to three evidence-backed matches and up to
  three gaps with an action; without a JD, positioning and evidence
  completeness advice instead.
- 缺失信息: exact absent field labels only.
- 待确认归属: material facts whose ownership stayed undetermined, grouped in
  source wording with one clarification question — never guessed into an
  experience, never silently dropped, never exposing internal IDs.
- 冲突检查: actual conflicts only, otherwise one explicit no-conflict line.
"""
from __future__ import annotations

import re

from .contracts import Audit, FactGraph, RequirementGraph


_JD_STOPWORDS = {"岗位要求", "任职要求", "工作职责", "职责", "要求", "负责", "能力", "经验"}
_MAX_MATCHES = 3
_MAX_GAPS = 3
_MAX_UNDETERMINED = 3


_REQUIREMENT_CONTENT_SIGNAL = re.compile(
    r"负责|要求|经验|技能|能力|熟练|熟悉|精通|优先|以上|年|学历|学位|证书"
    r"|manage|develop|build|lead|design|experience|skill|degree",
    re.IGNORECASE,
)
_PURE_LABEL_LINE = re.compile(r"^[^：:\n]{1,15}[：:]\s*$")


def _is_actionable_requirement(text: str) -> bool:
    """Only real requirement content may surface as a match/gap item.

    JD section headers and bare title lines are parsing artifacts; showing
    them to the user as "差距" would expose our own mistakes in the reply.
    """

    compact = re.sub(r"\s+", "", str(text))
    if not compact or _PURE_LABEL_LINE.match(str(text).strip()):
        return False
    if len(compact) <= 8 and not _REQUIREMENT_CONTENT_SIGNAL.search(compact):
        return False
    return True


def _excerpt(text: str, limit: int = 24) -> str:
    # Fold, never delete: latin word boundaries are content ("Delivered
    # results" must not become "Deliveredresults").  Chinese text has no
    # significant inner whitespace, so this is byte-identical there.
    compact = re.sub(r"\s+", " ", str(text)).strip()
    compact = compact.lstrip("-•·*▪◦")
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _requirement_tokens(requirement: str) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{1,}|[一-鿿]{2,8}", requirement)
        if token.casefold() not in {item.casefold() for item in _JD_STOPWORDS}
    ]


_GENERIC_MATCH_TOKENS = {
    "分析", "管理", "负责", "协助", "支持", "沟通", "协调", "能力", "经验",
    "工作", "项目", "团队", "业务", "相关", "进行", "完成",
}


def _supporting_fact(requirement: str, fact_texts: list[str]) -> str:
    """Return a fact with a *distinctive* token match, or '' for weak ones.

    A two-character generic overlap (分析/管理) is not evidence; require a
    Latin/skill token or a CJK token of at least 4 chars that is not a
    generic resume word.
    """

    tokens = _requirement_tokens(requirement)
    strong = [
        token for token in tokens
        if re.search(r"[A-Za-z]", token) or (len(token) >= 4 and token not in _GENERIC_MATCH_TOKENS)
    ]
    if not strong:
        return ""
    for fact in fact_texts:
        folded = fact.casefold()
        if any(token in folded for token in strong):
            return fact
    return ""


# Record-family destinations: facts routed here are expected to belong to a
# record.  Skills/education/contact legitimately own no record and must never
# be surfaced as "待确认归属" noise.
_RECORD_FAMILY_SECTIONS = {"experience", "projects", "research", "activities", "teaching"}


def undetermined_ownership_facts(audit: Audit, fact_graph: FactGraph) -> list[str]:
    """Material written facts that look record-bound but stayed unassigned."""

    fact_map = fact_graph.fact_map()
    items: list[str] = []
    for fact_id in audit.written_fact_ids:
        fact = fact_map.get(fact_id)
        if fact is None or fact.record_id is not None:
            continue
        if fact.fact_type in {"identity", "contact", "placeholder"}:
            continue
        destination = fact.destination_section or ""
        if destination not in _RECORD_FAMILY_SECTIONS:
            continue
        items.append(fact.text)
    return items


_CONFLICT_TYPE_LABELS = {
    "period": "时间",
    "organization": "组织",
    "role": "岗位",
    "metric": "数字",
    "credential": "证书或资质",
}


def friendly_conflicts(conflicts: list[str]) -> list[str]:
    """Render audit conflict hints without internal record/fact identifiers.

    Entries that cannot be mapped to a user-readable description (including
    ``unassigned`` pseudo-records) are dropped, never passed through.
    """

    labels = []
    for item in conflicts:
        owner, _, fact_type = str(item).rpartition(":")
        if not owner or owner == "unassigned":
            continue
        label = _CONFLICT_TYPE_LABELS.get(fact_type)
        if label is None:
            continue
        if label not in labels:
            labels.append(label)
    return [
        f"存在多处不一致的{label}表述，请核对确认。"
        for label in labels
    ]


def build_reply(
    audit: Audit,
    fact_graph: FactGraph,
    requirements: RequirementGraph | None = None,
    *,
    missing_fields: list[dict] | None = None,
    skeleton: bool = False,
    quarantined_numeric: list[dict] | None = None,
) -> str:
    fact_map = fact_graph.fact_map()
    written = [fact_map[fid].text for fid in audit.written_fact_ids if fid in fact_map]
    lines: list[str] = []

    if skeleton:
        lines.append("生成方向：当前未提供可写入的个人事实，已生成结构化待填写框架，不虚构任何经历。")
    else:
        lines.append("生成方向：仅依据可回指的个人事实组织简历；JD 仅用于排序与差距分析，不会补写为个人经历。")

    requirement_items = [
        item for item in (requirements.requirements if requirements else [])
        if _is_actionable_requirement(item.text)
    ]
    if requirement_items:
        matches: list[str] = []
        gaps: list[str] = []
        # Score every actionable requirement so the reported counts add up to
        # the JD's actual total; only the displayed items are capped below.
        for item in requirement_items:
            support = _supporting_fact(item.text, written)
            if support:
                matches.append(f"{_excerpt(item.text, 20)}（依据：{_excerpt(support)}）")
            else:
                gaps.append(f"{_excerpt(item.text, 20)}")
        lines.append(f"岗位匹配：共 {len(requirement_items)} 项要求，{len(matches)} 项要求找到直接事实依据，{len(gaps)} 项暂无依据（来自JD的要求仅作参考，不会补写为个人经历）。")
        for match in matches[:_MAX_MATCHES]:
            lines.append(f"- 匹配：{match}")
        for gap in gaps[:_MAX_GAPS]:
            lines.append(f"- 差距：{gap}；建议补充真实对应经历，不会代为虚构。")
    else:
        if skeleton:
            lines.append("岗位建议：提供目标 JD 后可基于写入事实做匹配分析；当前先按框架补充真实信息。")
        else:
            advice = "材料已覆盖通用必备模块。" if not missing_fields else "建议优先补齐下方缺失项后再投递。"
            lines.append(f"岗位建议：未提供目标岗位，已按通用结构保留全部可验证事实；{advice}")

    labels = [
        str(item.get("label") or item.get("field") or "")
        for item in (missing_fields or [])
        if isinstance(item, dict)
    ]
    labels = [label for label in labels if label]
    if labels:
        lines.append("缺失信息：" + "、".join(labels) + "。")
    else:
        lines.append("缺失信息：当前通用必备模块未发现明确缺项。")

    undetermined = undetermined_ownership_facts(audit, fact_graph)
    if undetermined:
        excerpts = "、".join(f"「{_excerpt(text, 30)}」" for text in undetermined[:_MAX_UNDETERMINED])
        shown = min(len(undetermined), _MAX_UNDETERMINED)
        lines.append(
            f"待确认归属：共 {len(undetermined)} 条信息未能确认所属经历，已按原文保留（展示 {shown} 条）：{excerpts}。"
            "请确认这些信息分别属于哪段经历。"
        )

    if quarantined_numeric:
        excerpts = "、".join(f"「{_excerpt(item['text'], 30)}」" for item in quarantined_numeric[:3])
        lines.append(
            f"待确认数字：{len(quarantined_numeric)} 条信息的数字/时间疑似识别错误，未写入简历：{excerpts}。"
            "请核对原件后补充真实数值。"
        )

    # Filter before slicing: taking the first three raw conflicts could pick
    # only unmappable internal entries and render a bare "冲突检查：" header.
    conflict_lines = friendly_conflicts(list(audit.conflicts))[:3]
    if conflict_lines:
        lines.append("冲突检查：")
        lines.extend(f"- {item}" for item in conflict_lines)
    else:
        lines.append("冲突检查：未发现时间或信息冲突。")
    return "\n".join(lines)


__all__ = ["build_reply", "friendly_conflicts", "undetermined_ownership_facts"]
