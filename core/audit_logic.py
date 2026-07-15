import json
import re
from typing import Any, Optional

from prompts import AUDIT_SYSTEM_PROMPT
from resume_common import (
    _calibrate_overall_score,
    _collect_substantive_changes,
    _contains_any,
    _dedupe_and_sort_issues,
    _extract_tech_anchor_tokens,
    _has_metric,
    _looks_over_generic,
    _normalize_compare_text,
    _normalize_dimension,
    _normalize_severity,
    clamp_score,
)
from resume_parsing import (
    _collect_text_entries,
    _compact_preview,
    _count_education,
    _count_projects,
    _count_publications,
    _count_resume_bullets,
    resume_data_to_text,
    split_bullets,
)
from schemas import AuditLLMOutput
from server_runtime import (
    ACTION_WORDS,
    DETAIL_HINT_WORDS,
    ENABLE_HEURISTIC_AUDIT_FALLBACK,
    GENERIC_AUDIT_PATTERNS,
    MAX_AUDIT_ISSUES,
    RESPONSIBILITY_WORDS,
    TECH_KEYWORDS,
    call_llm_typed,
    llm_enabled,
    logger,
    sanitize_user_text,
)
def extract_jd_keywords(jd_text: str) -> list[str]:
    if not jd_text:
        return []

    stop_words = {
        "负责", "熟悉", "具备", "优先", "以上", "相关", "经验", "能力", "要求", "岗位", "工作", "进行", "以及",
        "and", "with", "for", "the", "you", "will", "have", "years", "experience", "ability", "plus",
    }

    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_+./-]{1,24}|[\u4e00-\u9fff]{2,8}", jd_text)
    seen = set()
    result: list[str] = []

    for token in candidates:
        key = token.lower()
        if key in stop_words:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(token)
        if len(result) >= 40:
            break
    return result


def compute_jd_alignment(resume_text: str, jd_text: Optional[str]) -> dict[str, Any]:
    if not jd_text or not jd_text.strip():
        return {
            "matched_keywords": [],
            "missing_keywords": [],
            "coverage_score": 0,
        }

    resume_lower = resume_text.lower()
    keywords = extract_jd_keywords(jd_text)
    if not keywords:
        return {
            "matched_keywords": [],
            "missing_keywords": [],
            "coverage_score": 0,
        }

    matched = [kw for kw in keywords if kw.lower() in resume_lower]
    missing = [kw for kw in keywords if kw.lower() not in resume_lower]
    coverage = int(round(len(matched) / len(keywords) * 100))

    return {
        "matched_keywords": matched[:15],
        "missing_keywords": missing[:15],
        "coverage_score": coverage,
    }


def _minimal_audit_result(resume_text: str, jd_text: Optional[str], reason: str = "") -> dict[str, Any]:
    alignment = compute_jd_alignment(resume_text, jd_text)
    dim_scores = {
        "technical_depth": 5.0,
        "quantification": 5.0,
        "responsibility_clarity": 5.0,
        "authenticity": 5.0,
    }
    summary_reason = _compact_preview(reason, limit=120) if reason else "no_heuristic_fallback"
    return {
        "overall_score": 5.0,
        "dimension_scores": dim_scores,
        "issues": [],
        "jd_alignment": alignment,
        "summary": f"启发式审计回退已禁用；本次返回最小审计结构（reason={summary_reason}）。",
    }


def _audit_fallback(resume_text: str, jd_text: Optional[str], reason: str) -> dict[str, Any]:
    if ENABLE_HEURISTIC_AUDIT_FALLBACK:
        logger.warning("%s; fallback to heuristic audit", reason)
        return audit_with_heuristics(resume_text, jd_text)
    logger.warning("%s; heuristic fallback disabled, return minimal audit result", reason)
    return _minimal_audit_result(resume_text, jd_text, reason=reason)




def _contains_any(text_lower: str, words: set[str]) -> bool:
    return any(w in text_lower for w in words)


def audit_with_heuristics(resume_text: str, jd_text: Optional[str] = None) -> dict[str, Any]:
    bullets = split_bullets(resume_text)
    if not bullets:
        _log_parse_text_debug(
            stage="heuristic_no_bullets",
            resume_text=resume_text,
            extra={"jd_provided": bool(jd_text)},
        )
        alignment = compute_jd_alignment(resume_text, jd_text)
        return {
            "overall_score": 3.8,
            "dimension_scores": {
                "technical_depth": 3.5,
                "quantification": 3.5,
                "responsibility_clarity": 4.0,
                "authenticity": 4.2,
            },
            "issues": [
                {
                    "project": "简历整体",
                    "bullet_index": 1,
                    "dimension": "responsibility_clarity",
                    "severity": "high",
                    "problem": "未识别到可审计的项目要点（bullet）。当前文本结构化程度不足，难以稳定给出可追问风险评估。",
                    "suggestion": "请将每段经历整理为“项目名 + 3-5 条要点”，每条包含职责、技术方案、量化结果或验证方式。",
                    "interviewer_question": "请你按项目逐条说明：你的职责、关键技术决策、量化结果与验证方法分别是什么？",
                }
            ],
            "jd_alignment": alignment,
            "summary": "本次未识别到可审计的要点结构，已降级返回保守评分。建议先规范项目要点格式后再进行优化与审计。",
        }

    dimension_totals = {
        "technical_depth": 0.0,
        "quantification": 0.0,
        "responsibility_clarity": 0.0,
        "authenticity": 0.0,
    }
    issues: list[dict[str, Any]] = []

    high_count = 0

    for bullet in bullets:
        text = bullet["text"]
        text_lower = text.lower()

        has_tech = _contains_any(text_lower, TECH_KEYWORDS)
        has_detail = _contains_any(text_lower, DETAIL_HINT_WORDS)
        has_responsibility = _contains_any(text_lower, RESPONSIBILITY_WORDS)
        has_action = _contains_any(text_lower, ACTION_WORDS)
        has_metric = _has_metric(text)

        technical_depth = 4.0 + (2.2 if has_tech else 0.4) + (2.4 if has_detail else 0.2) + (1.0 if has_action else 0.0)
        quantification = 4.0 + (4.0 if has_metric else 0.6)
        responsibility = 4.0 + (3.2 if has_responsibility else 0.6) + (0.8 if has_action else 0.0)
        authenticity = 4.0 + (2.2 if has_tech else 0.4) + (2.0 if has_detail else 0.2) + (1.0 if has_metric else 0.2)

        dimension_totals["technical_depth"] += clamp_score(technical_depth)
        dimension_totals["quantification"] += clamp_score(quantification)
        dimension_totals["responsibility_clarity"] += clamp_score(responsibility)
        dimension_totals["authenticity"] += clamp_score(authenticity)

        if not has_detail:
            issues.append(
                {
                    "project": bullet["project"],
                    "bullet_index": bullet["bullet_index"],
                    "dimension": "technical_depth",
                    "severity": "high" if has_tech else "medium",
                    "problem": "技术描述停留在工具/框架层，缺少方案级细节（策略、权衡、边界条件）。",
                    "suggestion": "补充关键技术决策：为何这样设计、核心实现机制、线上问题与权衡。",
                    "interviewer_question": "这个方案的关键 trade-off 是什么？如果流量翻倍，你会先改哪一层？",
                }
            )

        if not has_metric:
            issues.append(
                {
                    "project": bullet["project"],
                    "bullet_index": bullet["bullet_index"],
                    "dimension": "quantification",
                    "severity": "medium",
                    "problem": "缺少可验证量化结果或基线，难以证明优化收益。",
                    "suggestion": "补充基线/测量口径/观测周期；没有精确数字时可给出定性收益与验证方式。",
                    "interviewer_question": "你这个优化的 baseline 和测量方法是什么？",
                }
            )

        if not has_responsibility:
            issues.append(
                {
                    "project": bullet["project"],
                    "bullet_index": bullet["bullet_index"],
                    "dimension": "responsibility_clarity",
                    "severity": "medium",
                    "problem": "个人职责边界不清晰，容易被追问团队分工。",
                    "suggestion": "明确你主导/负责的模块、协作对象与交付结果。",
                    "interviewer_question": "这个项目里你具体负责哪一部分？其他同学负责什么？",
                }
            )

    for issue in issues:
        if issue["severity"] == "high":
            high_count += 1

    n = len(bullets)
    dim_scores = {k: clamp_score(v / n) for k, v in dimension_totals.items()}
    normalized_issues = _dedupe_and_sort_issues(issues, limit=MAX_AUDIT_ISSUES)
    overall_score = _calibrate_overall_score(dim_scores, normalized_issues)

    alignment = compute_jd_alignment(resume_text, jd_text)
    if alignment["coverage_score"]:
        aligned = clamp_score(overall_score * 0.9 + alignment["coverage_score"] / 100 * 1.0)
        overall_score = _calibrate_overall_score(dim_scores, normalized_issues, aligned)

    summary = (
        "简历整体经历较真实，但技术细节与可验证指标仍需增强，建议优先补齐高风险条目。"
        if high_count > 0
        else "简历整体可读性尚可，建议进一步补充量化验证与职责边界，提升面试可辩护性。"
    )

    return {
        "overall_score": overall_score,
        "dimension_scores": dim_scores,
        "issues": normalized_issues,
        "jd_alignment": alignment,
        "summary": summary,
    }


def _short_snippet(text: str, max_len: int = 42) -> str:
    value = str(text or "").strip()
    if len(value) <= max_len:
        return value
    return value[:max_len].rstrip() + "..."


def _build_resume_bullet_lookup(resume_text: str) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    for item in split_bullets(resume_text):
        project = str(item.get("project") or "项目经历").strip() or "项目经历"
        bullet_index = int(item.get("bullet_index") or 0)
        text = str(item.get("text") or "").strip()
        if bullet_index > 0 and text:
            lookup[(project, bullet_index)] = text
    return lookup


def _needs_issue_expansion(text: str) -> bool:
    value = str(text or "").strip()
    if len(value) < 24:
        return True
    return any(token in value for token in GENERIC_AUDIT_PATTERNS)


def _expand_issue_with_context(issue: dict[str, Any], bullet_text: str) -> dict[str, Any]:
    expanded = dict(issue)
    dim = _normalize_dimension(issue.get("dimension"))
    snippet = _short_snippet(bullet_text)
    if snippet:
        expanded["context_snippet"] = snippet

    if _needs_issue_expansion(expanded.get("problem")):
        if dim == "technical_depth":
            expanded["problem"] = f"当前表述“{snippet}”主要停留在结论层，缺少可追问的技术路径信息：核心模块如何设计、关键参数/策略如何选择、以及在约束下做了哪些权衡。"
        elif dim == "quantification":
            expanded["problem"] = f"当前表述“{snippet}”缺少可复核证据，未给出 baseline、指标口径、评测范围或观测周期，难以判断收益是否稳定且可复现。"
        elif dim == "responsibility_clarity":
            expanded["problem"] = f"当前表述“{snippet}”没有清晰区分个人贡献与团队协作边界，面试追问时难以证明你实际主导了哪些关键工作。"
        else:
            expanded["problem"] = f"当前表述“{snippet}”缺少真实工程痕迹（限制条件、失败尝试、取舍依据、验证闭环），容易被判断为模板化表达。"

    if _needs_issue_expansion(expanded.get("suggestion")):
        if dim == "technical_depth":
            expanded["suggestion"] = "补充“方案-实现-权衡-验证”四段信息：说明为什么这样设计、关键模块怎么落地、与替代方案的取舍依据、以及最终如何验证有效。"
        elif dim == "quantification":
            expanded["suggestion"] = "补充可核验指标链路：基线版本、核心指标定义、测试数据集/流量范围、观测窗口；若无绝对数值，至少给出对比关系与验证方法。"
        elif dim == "responsibility_clarity":
            expanded["suggestion"] = "明确你的 owner 边界：你负责的模块、你做的关键决策、与同事的分工接口、以及你个人可归因的最终产出。"
        else:
            expanded["suggestion"] = "增加真实工程证据：约束条件、关键故障或失败路径、为何选择当前方案而非备选方案、上线后如何持续监控与回归。"

    question = str(expanded.get("interviewer_question") or "").strip()
    if len(question) < 18:
        expanded["interviewer_question"] = f"围绕“{snippet}”，请你按背景-方案-权衡-验证顺序详细复盘一次，并说明你个人负责的关键决策。"

    return expanded


def _build_audit_summary(
    issues: list[dict[str, Any]],
    dim_scores: dict[str, float],
    alignment: dict[str, Any],
) -> str:
    high_count = sum(1 for i in issues if _normalize_severity(i.get("severity")) == "high")
    medium_count = sum(1 for i in issues if _normalize_severity(i.get("severity")) == "medium")
    low_count = sum(1 for i in issues if _normalize_severity(i.get("severity")) == "low")
    weakest = sorted(dim_scores.items(), key=lambda x: x[1])[:2]
    weakest_text = "、".join(f"{k}:{v:.1f}" for k, v in weakest)
    coverage = int(alignment.get("coverage_score") or 0) if isinstance(alignment, dict) else 0
    return (
        f"本次审计共识别 {len(issues)} 个风险点（high {high_count} / medium {medium_count} / low {low_count}）。"
        f"当前短板维度为 {weakest_text}，建议优先补齐高风险条目的技术细节与验证证据。"
        f"JD 关键词覆盖率约为 {coverage}%（仅作参考，最终仍以事实可辩护性为准）。"
    )


def _target_issue_count_for_resume(resume_text: str) -> int:
    bullet_count = len(split_bullets(resume_text))
    if bullet_count >= 8:
        return 4
    if bullet_count >= 4:
        return 3
    if bullet_count >= 2:
        return 2
    return 1 if bullet_count > 0 else 0


def _supplement_sparse_audit_issues(
    issues: list[dict[str, Any]],
    resume_text: str,
    jd_text: Optional[str],
) -> list[dict[str, Any]]:
    target = _target_issue_count_for_resume(resume_text)
    if len(issues) >= target or target <= 0:
        return issues
    heuristic = audit_with_heuristics(resume_text, jd_text)
    heuristic_issues = heuristic.get("issues", []) if isinstance(heuristic, dict) else []
    if not isinstance(heuristic_issues, list):
        heuristic_issues = []
    return _dedupe_and_sort_issues(list(issues) + heuristic_issues, limit=MAX_AUDIT_ISSUES)


def _dimension_label(dim: Any) -> str:
    value = _normalize_dimension(dim)
    return {
        "technical_depth": "技术深度",
        "quantification": "量化证明",
        "responsibility_clarity": "职责边界",
        "authenticity": "真实性",
    }.get(value, "技术深度")


def _severity_label(severity: Any) -> str:
    value = _normalize_severity(severity)
    return {
        "high": "高",
        "medium": "中",
        "low": "低",
    }.get(value, "中")


def _dedupe_text_list(items: list[str], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        key = _normalize_compare_text(text).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _build_user_report(
    resume_data: dict[str, Any],
    audit_report: dict[str, Any],
    changes: list[dict[str, Any]],
    jd_text: Optional[str] = None,
    missing_fields: Optional[list] = None,
    time_conflicts: Optional[list] = None,
    fab_report: Optional[Any] = None,
) -> dict[str, Any]:
    def _safe_score(value: Any, default: float = 5.0) -> float:
        try:
            return clamp_score(float(value))
        except Exception:
            return clamp_score(default)

    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return default

    dim_scores = audit_report.get("dimension_scores", {}) if isinstance(audit_report, dict) else {}
    if not isinstance(dim_scores, dict):
        dim_scores = {}
    normalized_dim = {
        "technical_depth": _safe_score(dim_scores.get("technical_depth", 5.0)),
        "quantification": _safe_score(dim_scores.get("quantification", 5.0)),
        "responsibility_clarity": _safe_score(dim_scores.get("responsibility_clarity", 5.0)),
        "authenticity": _safe_score(dim_scores.get("authenticity", 5.0)),
    }
    issues = audit_report.get("issues", []) if isinstance(audit_report, dict) else []
    if not isinstance(issues, list):
        issues = []
    alignment = audit_report.get("jd_alignment", {}) if isinstance(audit_report, dict) else {}
    if not isinstance(alignment, dict):
        alignment = {}

    score = _safe_score(audit_report.get("overall_score", 5.0)) if isinstance(audit_report, dict) else 5.0
    project_count = _count_projects(resume_data)
    bullet_count = _count_resume_bullets(resume_data)
    publication_count = _count_publications(resume_data)
    education_count = _count_education(resume_data)
    honor_count = len(_collect_text_entries(resume_data, ("honors", "awards", "certifications")))
    matched_keywords = alignment.get("matched_keywords", [])
    matched_count = len(matched_keywords) if isinstance(matched_keywords, list) else 0
    sorted_dims = sorted(normalized_dim.items(), key=lambda item: item[1], reverse=True)
    weakest_dims = sorted(normalized_dim.items(), key=lambda item: item[1])[:2]
    high_count = sum(1 for item in issues if _normalize_severity(item.get("severity")) == "high")
    medium_count = sum(1 for item in issues if _normalize_severity(item.get("severity")) == "medium")
    substantive_changes = _collect_substantive_changes(changes)
    has_substantive_rewrite = bool(substantive_changes)

    def _issue_prefix(item: dict[str, Any]) -> str:
        project = str(item.get("project") or "相关经历").strip() or "相关经历"
        bullet_index = _safe_int(item.get("bullet_index"), 0) if str(item.get("bullet_index") or "").strip() else 0
        snippet = str(item.get("context_snippet") or "").strip()
        prefix = project
        if bullet_index > 0:
            prefix += f" 第{bullet_index}条"
        if snippet:
            prefix += f"“{snippet}”"
        return prefix

    if score >= 8.5:
        headline = f"整体得分 {score:.1f}/10，属于较强可投递版本。"
    elif score >= 7.5:
        headline = f"整体得分 {score:.1f}/10，主体内容扎实，但仍有若干可被追问的细节缺口。"
    else:
        headline = f"整体得分 {score:.1f}/10，建议先补强关键经历的证据链，再进入高强度投递。"
    if high_count > 0:
        headline += f" 当前识别到 {high_count} 个高风险点，建议优先处理。"
    elif medium_count > 0:
        headline += f" 当前暂无明显高风险项，但有 {medium_count} 个中风险点适合进一步补强。"
    else:
        headline += " 当前未发现明显高风险硬伤，更适合做表达增强和面试预演。"
    if has_substantive_rewrite:
        headline += f" 本轮已完成 {len(substantive_changes)} 处可验证的句子级改写。"
    else:
        headline += " 本轮主要输出诊断与建议，未发生可验证的句子级实质改写。"

    strengths: list[str] = []
    if sorted_dims and sorted_dims[0][1] >= 8.0:
        strengths.append(f"{_dimension_label(sorted_dims[0][0])}表现较强（{sorted_dims[0][1]:.1f}/10），说明核心经历具备较好的可辩护性。")
    if publication_count >= 3:
        strengths.append(f"学术/成果输出较完整，已识别到 {publication_count} 条论文或成果记录，有助于支撑研究型与分析型岗位 credibility。")
    if project_count >= 2 or bullet_count >= 8:
        strengths.append(f"项目与经历覆盖较完整，当前共识别 {project_count} 个项目、{bullet_count} 条要点，素材基础足够支撑进一步润色。")
    if education_count >= 2:
        strengths.append(f"教育背景信息完整，已识别 {education_count} 段教育经历，基础履历连续性较好。")
    if honor_count >= 2:
        strengths.append(f"荣誉与奖项信息较丰富，已识别 {honor_count} 条荣誉记录，可用于增强稳定性与竞争力印象。")
    if matched_count > 0:
        strengths.append(f"与目标 JD 已命中 {matched_count} 个关键词，说明现有经历与岗位要求已有一定贴合度。")
    strengths = _dedupe_text_list(strengths, limit=4)

    risk_analysis: list[str] = []
    for item in issues[:4]:
        risk_analysis.append(
            f"{_severity_label(item.get('severity'))}风险：{_issue_prefix(item)}，主要问题是{_compact_preview(item.get('problem', ''), limit=96)}"
        )
    if not risk_analysis:
        for dim, value in weakest_dims:
            risk_analysis.append(
                f"当前未发现明显高风险项，但 {_dimension_label(dim)} 是相对短板（{value:.1f}/10），继续补充证据链会更稳。"
            )
    risk_analysis = _dedupe_text_list(risk_analysis, limit=3)

    full_audit_details: list[dict[str, Any]] = []
    for item in issues[:MAX_AUDIT_ISSUES]:
        if not isinstance(item, dict):
            continue
        full_audit_details.append(
            {
                "project": str(item.get("project") or "相关经历").strip() or "相关经历",
                "bullet_index": _safe_int(item.get("bullet_index"), 0) if str(item.get("bullet_index") or "").strip() else 0,
                "dimension": _dimension_label(item.get("dimension")),
                "severity": _severity_label(item.get("severity")),
                "context_snippet": str(item.get("context_snippet") or "").strip(),
                "problem": str(item.get("problem") or "").strip(),
                "suggestion": str(item.get("suggestion") or "").strip(),
                "interviewer_question": str(item.get("interviewer_question") or "").strip(),
            }
        )

    priority_fixes = full_audit_details[: min(4, len(full_audit_details))]

    improvement_actions: list[str] = []
    for item in issues[:4]:
        suggestion = str(item.get("suggestion") or "").strip()
        if suggestion:
            improvement_actions.append(f"{_issue_prefix(item)}：{suggestion}")
    if not improvement_actions:
        advice_map = {
            "technical_depth": "优先把关键项目补成“背景约束-方案设计-关键权衡-验证闭环”四段式，而不是只写结果。",
            "quantification": "为核心成果补上 baseline、指标定义、样本范围或观测周期；没有精确数字时，也要写清验证方式。",
            "responsibility_clarity": "把“我负责什么、团队负责什么、最终交付了什么”拆开写，减少面试时职责追问的风险。",
            "authenticity": "增加真实工程痕迹，如失败尝试、异常 case、取舍原因与上线后的监控反馈。",
        }
        for dim, _ in weakest_dims:
            improvement_actions.append(advice_map.get(dim, advice_map["technical_depth"]))
    if has_substantive_rewrite:
        improvement_actions.append(f"本轮已完成 {len(substantive_changes)} 处实质改写，建议按同样模式继续扩展到其余类似条目。")
    improvement_actions = _dedupe_text_list(improvement_actions, limit=5)

    interview_prep: list[str] = []
    for item in issues[:3]:
        question = str(item.get("interviewer_question") or "").strip()
        if question:
            interview_prep.append(question)
    if not interview_prep:
        for dim, _ in weakest_dims:
            interview_prep.append(
                f"请围绕你最核心的一段经历，按背景-方案-权衡-验证顺序复盘一次，并重点说明 {_dimension_label(dim)} 相关证据。"
            )
    interview_prep = _dedupe_text_list(interview_prep, limit=3)

    rewrite_samples: list[dict[str, Any]] = []
    for item in substantive_changes:
        if not isinstance(item, dict):
            continue
        before = str(item.get("before") or "").strip()
        after = str(item.get("after") or "").strip()
        if not before or not after:
            continue
        rewrite_samples.append(
            {
                "project": str(item.get("project") or "相关经历").strip() or "相关经历",
                "bullet_index": _safe_int(item.get("bullet_index"), 0) if str(item.get("bullet_index") or "").strip() else 0,
                "before": _compact_preview(before, limit=120),
                "after": _compact_preview(after, limit=180),
                "reason": _compact_preview(item.get("reason", ""), limit=120),
            }
        )
        if len(rewrite_samples) >= 2:
            break

    report_summary_parts: list[str] = []
    if strengths:
        report_summary_parts.append(f"亮点方面，{strengths[0]}")
    if risk_analysis:
        report_summary_parts.append(f"风险方面，{risk_analysis[0]}")
    if improvement_actions:
        report_summary_parts.append(f"建议优先执行：{improvement_actions[0]}")

    # Validation data
    validation_missing: list[dict[str, Any]] = []
    if missing_fields:
        validation_missing = [{"field": mf.field, "reason": mf.reason} for mf in missing_fields]
    validation_conflicts: list[dict[str, Any]] = []
    if time_conflicts:
        validation_conflicts = [{"field": c.field, "description": c.description} for c in time_conflicts]
    fabrication_found = fab_report.fabrication_found if fab_report else False
    fabrication_details = []
    if fab_report and hasattr(fab_report, "details"):
        fabrication_details = [{"type": d.type, "content": d.content, "reason": d.reason} for d in fab_report.details]

    return {
        "headline": headline,
        "summary": " ".join(report_summary_parts).strip(),
        "strengths": strengths,
        "risk_analysis": risk_analysis,
        "priority_fixes": priority_fixes,
        "full_audit_details": full_audit_details,
        "improvement_actions": improvement_actions,
        "interview_prep": interview_prep,
        "rewrite_samples": rewrite_samples,
        "validation": {
            "missing_fields": validation_missing,
            "conflicts": validation_conflicts,
            "fabrication_found": fabrication_found,
            "fabrication_details": fabrication_details,
        },
        "stats": {
            "projects": project_count,
            "bullets": bullet_count,
            "publications": publication_count,
            "education": education_count,
            "honors": honor_count,
            "matched_keywords": matched_count,
            "has_jd": bool(jd_text and str(jd_text).strip()),
            "has_substantive_rewrite": has_substantive_rewrite,
            "substantive_change_count": len(substantive_changes),
            "audit_issue_count": len(issues),
        },
    }


def _jd_changed(new_jd: Optional[str], old_jd: Optional[str]) -> bool:
    return _normalize_compare_text(new_jd or "") != _normalize_compare_text(old_jd or "")


def _build_mcp_tools_instruction(
    *,
    files: dict[str, Optional[str]],
    changes: list[dict[str, Any]],
    user_report: dict[str, Any],
    draft_id: str,
) -> dict[str, Any]:
    available_links = [str(value) for value in (files or {}).values() if isinstance(value, str) and value.strip()]
    rewrite_samples = user_report.get("rewrite_samples", []) if isinstance(user_report, dict) else []
    substantive_changes = _collect_substantive_changes(changes)
    has_substantive_rewrite = bool(substantive_changes)
    return {
        "priority": "high",
        "tool_name": "z0ive_resume_audit_and_optimize",
        "must_follow": [
            "仅可陈述本次接口真实返回的事实、分数、问题、改写与下载链接。",
            "若 has_substantive_rewrite=false，禁止声称“已注入/已新增/已改写/已生成”任何具体项目内容。",
            "禁止编造模型名、技术栈、论文、GitHub star/fork、比赛名次、会议论文、链接可访问性等未在返回结果中出现的事实。",
            "若仅基于 JD 做诊断，可说“建议补充/建议显性化”，不得说“已经替你写入简历”。",
            "仅可展示 available_links 中真实返回的文件链接；若无链接，必须明确说明本次未生成可下载文件。",
            "只要用户围绕当前简历继续追问、要求按新 JD 分析、要求继续润色/改写/生成新文件/模拟追问，都必须再次调用 MCP，不要只凭历史回复直接作答。",
            "二次追问时优先复用 draft_id；不要要求用户重复上传同一份简历，除非 draft_id 无效或用户明确更换了简历。",
        ],
        "available_links": available_links,
        "applied_change_count": len(substantive_changes),
        "has_substantive_rewrite": has_substantive_rewrite,
        "substantive_change_count": len(substantive_changes),
        "has_rewrite_samples": bool(isinstance(rewrite_samples, list) and rewrite_samples),
        "draft_id": draft_id,
        "follow_up_required": bool(str(draft_id).strip()),
        "follow_up_trigger_examples": [
            "根据这个 JD 再给我一些意见",
            "把项目一改得更像大模型岗位",
            "继续润色刚才那版简历",
            "生成新的 PDF/DOCX",
            "针对风险点继续追问并给回答稿",
        ],
        "suggested_follow_up_payload": {
            "draft_id": draft_id,
            "jd_text": "<用户本轮提供的新JD，没有则留空>",
            "revision_instructions": "<用户本轮的追问/修改要求，没有则留空>",
        },
    }


def normalize_audit_result(raw: dict[str, Any], resume_text: str, jd_text: Optional[str]) -> dict[str, Any]:
    if not raw:
        return _audit_fallback(resume_text, jd_text, "normalize_audit_result: raw audit is empty")

    try:
        raw_issues = raw.get("issues") or []
        if not isinstance(raw_issues, list):
            raw_issues = []

        dim_scores = raw.get("dimension_scores") or {}
        if not isinstance(dim_scores, dict):
            dim_scores = {}

        def _resolve_dim_score(key: str) -> float:
            value = dim_scores.get(key)
            if isinstance(value, (int, float)):
                return clamp_score(float(value))
            return 5.0

        normalized_dim = {
            "technical_depth": _resolve_dim_score("technical_depth"),
            "quantification": _resolve_dim_score("quantification"),
            "responsibility_clarity": _resolve_dim_score("responsibility_clarity"),
            "authenticity": _resolve_dim_score("authenticity"),
        }

        normalized_issues = _dedupe_and_sort_issues(raw_issues, limit=MAX_AUDIT_ISSUES)
        normalized_issues = _supplement_sparse_audit_issues(normalized_issues, resume_text, jd_text)

        alignment = raw.get("jd_alignment") or compute_jd_alignment(resume_text, jd_text)
        matched = alignment.get("matched_keywords") if isinstance(alignment, dict) else []
        missing = alignment.get("missing_keywords") if isinstance(alignment, dict) else []
        coverage = alignment.get("coverage_score") if isinstance(alignment, dict) else 0
        alignment = {
            "matched_keywords": matched if isinstance(matched, list) else [],
            "missing_keywords": missing if isinstance(missing, list) else [],
            "coverage_score": int(coverage) if isinstance(coverage, (int, float)) else 0,
        }

        overall_raw = raw.get("overall_score")
        overall = _calibrate_overall_score(
            normalized_dim,
            normalized_issues,
            float(overall_raw) if isinstance(overall_raw, (int, float)) else None,
        )

        bullet_lookup = _build_resume_bullet_lookup(resume_text)
        normalized_issues = [
            _expand_issue_with_context(
                issue,
                bullet_lookup.get((issue["project"], int(issue["bullet_index"])), ""),
            )
            for issue in normalized_issues
        ]

        raw_summary = str(raw.get("summary") or "").strip()
        summary = raw_summary if len(raw_summary) >= 36 else _build_audit_summary(normalized_issues, normalized_dim, alignment)

        return {
            "overall_score": overall,
            "dimension_scores": normalized_dim,
            "issues": normalized_issues,
            "jd_alignment": alignment,
            "summary": summary,
        }
    except Exception as exc:
        logger.warning("normalize_audit_result failed; return minimal non-heuristic audit: %s", exc, exc_info=True)
        return _minimal_audit_result(resume_text, jd_text, reason="normalize_audit_result: exception")

def audit_resume_core(resume_text: str, jd_text: Optional[str], resume_data: Optional[dict[str, Any]] = None, source_text: Optional[str] = None) -> dict[str, Any]:
    if not llm_enabled():
        return _audit_fallback(resume_text, jd_text, "audit_resume_core: llm not enabled")

    safe_jd = sanitize_user_text(jd_text or "")
    user_prompt = (
        "请对以下简历做首轮深入但聚焦的质量审计，用于发现可执行修改点。\n"
        "如果简历内容足够，请不要只给泛泛结论，优先输出能落到具体项目/具体条目的问题。\n\n"
        "【简历文本】\n"
        f"{resume_text}\n\n"
    )
    if source_text and len(source_text) > 100:
        # Extract key experience/project snippets from original text for issue_type judgment
        exp_snippets = []
        for exp in (resume_data or {}).get("experience", [])[:5]:
            if not isinstance(exp, dict): continue
            company = str(exp.get("company", "") or "").strip()
            if company and len(company) >= 3:
                idx = source_text.lower().find(company.lower())
                if idx >= 0:
                    snippet = source_text[max(0,idx-30):idx+len(company)+400].strip()
                    snippet = snippet[:500]
                    exp_snippets.append(f"[{company}原文] {snippet}")
        if exp_snippets:
            user_prompt += "【简历原文摘录（用于判断 issue_type——原文有的信息你可以参考，原文没有的不能假设）】\n"
            user_prompt += "\n\n".join(exp_snippets) + "\n\n"
    if isinstance(resume_data, dict):
        user_prompt += (
            "【简历结构化数据（辅助参考，文本为主）】\n"
            f"{json.dumps(resume_data, ensure_ascii=False)}\n\n"
        )
    user_prompt += (
        "【目标 JD（可选）】\n"
        f"{safe_jd}\n\n"
        "【输出要求】\n"
        "- 必须返回 overall_score, dimension_scores, issues, jd_alignment, summary\n"
        "- issues 中每项必须包含：project, bullet_index, dimension, severity, issue_type, problem, suggestion, interviewer_question\n"
        "- issue_type 必须根据 system prompt 中的判定标准显式填写，不得遗漏或使用默认值\n"
        "- severity 仅允许 high/medium/low\n"
        f"- issues 最多 {MAX_AUDIT_ISSUES} 条，按风险排序\n"
        "- 若简历可审计 bullet >= 4，issues 至少输出 3 条；若 bullet >= 8，优先输出 4-8 条\n"
        "- 若存在 high 风险，overall_score 不得接近满分\n"
        "- 避免重复模板句，问题描述必须能指向具体内容缺口\n"
        "- 每条 issue 需要写清楚证据缺口与追问风险，不要只给一句泛化结论\n"
        "- 每条 suggestion 必须是可执行动作，最好能明确写出要补在哪个项目/哪条经历上\n"
        "- summary 使用 2-4 句，说明优先修复顺序\n"
        "- 若 JD 为空，jd_alignment 仍需返回合法空结构"
    )
    try:
        raw = call_llm_typed(AuditLLMOutput, AUDIT_SYSTEM_PROMPT, user_prompt, temperature=0.3)
        if not raw:
            return _audit_fallback(resume_text, jd_text, "audit_resume_core: llm returned empty audit")
        return normalize_audit_result(raw, resume_text, jd_text)
    except Exception as exc:
        return _audit_fallback(resume_text, jd_text, f"audit_resume_core: llm failed: {exc}")

