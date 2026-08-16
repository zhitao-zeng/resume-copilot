"""Build user-facing explanation from the frozen audit only."""
from __future__ import annotations

import re

from .contracts import Audit, FactGraph, RequirementGraph


_JD_STOPWORDS = {"岗位要求", "任职要求", "工作职责", "职责", "要求", "负责", "能力", "经验"}


def _requirement_supported(requirement: str, fact_texts: list[str]) -> bool:
    tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{1,}|[\u4e00-\u9fff]{2,8}", requirement)
        if token.casefold() not in {item.casefold() for item in _JD_STOPWORDS}
    ]
    return any(token in fact.casefold() for token in tokens for fact in fact_texts)


def build_reply(audit: Audit, fact_graph: FactGraph, requirements: RequirementGraph | None = None) -> str:
    facts = fact_graph.fact_map()
    written = [facts[fid].text for fid in audit.written_fact_ids if fid in facts]
    missing = [facts[fid].text for fid in audit.missing_fact_ids if fid in facts]
    lines = ["处理总结：", "生成方向总结：仅使用可回指的个人事实组织简历；JD只用于排序和差距分析。"]
    if written:
        lines.append("已写入信息：")
        lines.extend(f"- {text}" for text in written)
    else:
        lines.append("已写入信息：暂无可验证的个人事实，已保留结构化待补充框架。")
    if missing:
        lines.append("建议补充或未纳入的信息：")
        lines.extend(f"- {text}" for text in missing)
    else:
        lines.append("建议补充或未纳入的信息：暂无已识别但未写入的个人事实。")
    if audit.conflicts:
        lines.append("待确认的潜在冲突：")
        lines.extend(f"- {item}" for item in audit.conflicts)
    else:
        lines.append("待确认的潜在冲突：未发现结构化冲突。")
    if requirements and requirements.requirements:
        lines.append("岗位建议：")
        considered = requirements.requirements[:8]
        covered = [item.text for item in considered if _requirement_supported(item.text, written)]
        unsupported = [item.text for item in considered if item.text not in covered]
        lines.append("- 已按岗位要求对已有事实排序；以下内容来自JD，不会被补写为个人经历。")
        if covered:
            lines.append("- 现有事实中找到直接文本依据的要求：" + "；".join(covered))
        if unsupported:
            lines.append("- 尚未在个人材料中找到直接依据的要求：" + "；".join(unsupported))
    else:
        lines.append("岗位建议：提供目标JD后可基于现有事实进行岗位匹配排序。")
    if audit.recommendations:
        lines.append("下一步建议：")
        lines.extend(f"- {item}" for item in audit.recommendations)
    return "\n".join(lines)


__all__ = ["build_reply"]
