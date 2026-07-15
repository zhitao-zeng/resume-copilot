import json
import re
from difflib import SequenceMatcher
from typing import Any, Optional

from http_compat import HTTPException

from schemas import RevisionTarget
from server_runtime import (
    AI_PHRASE_REPLACEMENTS,
    DETAIL_HINT_WORDS,
    GENERIC_BULLET_HINTS,
    GENERIC_MARKERS,
    MAX_AUDIT_ISSUES,
    RESPONSIBILITY_WORDS,
    SEVERITY_PRIORITY,
    TECH_ANCHOR_STOPWORDS,
    TECH_KEYWORDS,
    ZH_TECH_TERMS,
    VALID_AUDIT_DIMENSIONS,
    VALID_SEVERITIES,
)
def clamp_score(score: float) -> float:
    return round(max(1.0, min(10.0, score)), 1)


def _normalize_dimension(value: Any) -> str:
    dim = str(value or "").strip()
    return dim if dim in VALID_AUDIT_DIMENSIONS else "technical_depth"


def _normalize_severity(value: Any) -> str:
    severity = str(value or "").strip().lower()
    return severity if severity in VALID_SEVERITIES else "medium"


def _calibrate_overall_score(
    dim_scores: dict[str, float],
    issues: list[dict[str, Any]],
    raw_overall_score: Optional[float] = None,
) -> float:
    base_score = clamp_score(sum(dim_scores.values()) / 4)
    model_score = clamp_score(raw_overall_score) if isinstance(raw_overall_score, (int, float)) else base_score
    blended = clamp_score(base_score * 0.75 + model_score * 0.25)

    high_count = 0
    medium_count = 0
    low_count = 0
    for issue in issues:
        sev = _normalize_severity(issue.get("severity"))
        if sev == "high":
            high_count += 1
        elif sev == "medium":
            medium_count += 1
        else:
            low_count += 1

    cap = 10.0
    if high_count >= 2:
        cap = min(cap, 7.6)
    elif high_count == 1:
        cap = min(cap, 8.2)
    elif medium_count >= 5:
        cap = min(cap, 8.4)
    if len(issues) >= 6:
        cap = min(cap, 7.8)
    elif len(issues) >= 4:
        cap = min(cap, 8.6)

    penalty = high_count * 0.9 + medium_count * 0.35 + low_count * 0.12
    adjusted = clamp_score(blended - penalty * 0.2)
    return min(adjusted, cap)


def _dedupe_and_sort_issues(issues: list[dict[str, Any]], limit: int = MAX_AUDIT_ISSUES) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()

    for idx, item in enumerate(issues, start=1):
        project = str(item.get("project") or "项目经历").strip() or "项目经历"
        bullet_index_raw = item.get("bullet_index")
        try:
            bullet_index = int(bullet_index_raw)
        except Exception:
            bullet_index = idx
        bullet_index = max(1, bullet_index)

        dimension = _normalize_dimension(item.get("dimension"))
        severity = _normalize_severity(item.get("severity"))
        problem = str(item.get("problem") or "描述细节不足").strip() or "描述细节不足"
        suggestion = str(item.get("suggestion") or "补充技术决策与验证方式").strip() or "补充技术决策与验证方式"
        interviewer_question = (
            str(item.get("interviewer_question") or "这个点你具体怎么做的？").strip() or "这个点你具体怎么做的？"
        )

        issue_key = (project, bullet_index, dimension, problem)
        if issue_key in seen:
            continue
        seen.add(issue_key)

        deduped.append(
            {
                "project": project,
                "bullet_index": bullet_index,
                "dimension": dimension,
                "severity": severity,
                "problem": problem,
                "suggestion": suggestion,
                "interviewer_question": interviewer_question,
                "context_snippet": str(item.get("context_snippet") or "").strip(),
            }
        )

    deduped.sort(
        key=lambda x: (
            SEVERITY_PRIORITY.get(_normalize_severity(x.get("severity")), 9),
            str(x.get("project", "")),
            int(x.get("bullet_index", 0)),
        )
    )
    return deduped[:limit]


def _is_substantive_change(change: Any) -> bool:
    if not isinstance(change, dict):
        return False
    if str(change.get("location") or "").strip().lower() == "format":
        return False
    if str(change.get("project") or "").strip() == "全局":
        return False
    before = str(change.get("before") or "").strip()
    after = str(change.get("after") or "").strip()
    if not before or not after:
        return False
    if _normalize_compare_text(before) == _normalize_compare_text(after):
        return False
    if _is_trivial_wording_change(before, after):
        return False
    return True


def _collect_substantive_changes(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in changes if _is_substantive_change(item)]


def _extract_tech_anchor_tokens(text: str) -> set[str]:
    anchors: set[str] = set()

    # English tech tokens
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+./-]{1,30}", text)
    for token in tokens:
        lower = token.lower()
        if lower in TECH_ANCHOR_STOPWORDS:
            continue
        if any(ch.isupper() for ch in token) or any(ch.isdigit() for ch in token) or len(token) >= 6:
            anchors.add(lower)
        elif lower in TECH_KEYWORDS:
            anchors.add(lower)

    # Chinese tech terms
    for term in ZH_TECH_TERMS:
        if term in text:
            anchors.add(term)

    return anchors


def _looks_over_generic(text: str) -> bool:
    content = str(text or "").strip()
    if not content:
        return False
    lower = content.lower()
    if any(hint in content for hint in GENERIC_BULLET_HINTS):
        return True
    no_signal = not _has_metric(content) and not _extract_tech_anchor_tokens(content)
    return no_signal and sum(1 for m in GENERIC_MARKERS if m in content or m in lower) >= 2



def _has_metric(text: str) -> bool:
    # [需补充] placeholder counts as metric intent — the bullet is structured
    if "[需补充]" in text:
        return True
    metric_pattern = re.compile(
        r"\d+(?:\.\d+)?\s*(?:%|ms|s|分钟|小时|天|倍|w|万|k|qps|tps|fps|mb|gb|tb|条|次|人|个|毫秒|秒|亿|rows|users|requests|rps)",
        re.IGNORECASE,
    )
    return bool(metric_pattern.search(text))


def _metric_strength(text: str) -> str:
    """Return metric strength level: 'strong', 'weak', 'placeholder', or 'none'.

    - strong: comparative metric (from X% to Y%, increased by Z%)
    - weak: standalone number with unit (3个, 5人, 100ms) but no baseline comparison
    - placeholder: [需补充] marker
    - none: no quantitative content
    """
    if "[需补充]" in text:
        return "placeholder"

    # Comparative metrics: "从...到...", "从...升至/降至...", "提升/降低 X% → Y%"
    comparative_patterns = [
        r"从\s*\d+(?:\.\d+)?\s*(?:%|ms|s|倍)\s*(?:到|至|降至|升至|提高到|降低到)",
        r"(?:提升|降低|减少|增加|缩短|节省|提高|改善|降至|升至)\s*(?:了|至|到|为)?\s*\d+(?:\.\d+)?\s*(?:%|倍|ms|s|分钟|小时|天|w|万|k|qps|tps)",
        r"(?:increased|decreased|reduced|improved|boosted)\s+(?:by|to)\s+\d+(?:\.\d+)?\s*(?:%|x|ms|s)",
    ]
    for pattern in comparative_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "strong"

    if _has_metric(text):
        return "weak"

    return "none"


def _contains_any(text_lower: str, words: set[str]) -> bool:
    return any(w in text_lower for w in words)



def _clone_json(data: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(data, ensure_ascii=False))


def _normalize_compare_text(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_project_name(value: Any) -> str:
    text = _normalize_compare_text(value).lower()
    if not text:
        return ""
    text = re.split(r"(指导老师|advisor|mentor)", text, maxsplit=1)[0]
    text = re.sub(
        r"((19|20)\d{2}[./-]\d{1,2}\s*(?:[-–—至到~]\s*((19|20)\d{2}[./-]\d{1,2}|至今|present))?)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(研究项目|项目经历|项目经验|project)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[-–—|:：()\[\]{}【】]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _project_content_signature(project: dict[str, Any]) -> tuple[str, str, str]:
    name = _normalize_project_name(project.get("name", ""))
    period = _normalize_compare_text(project.get("period", "")).lower()
    bullets = project.get("bullets", [])
    if isinstance(bullets, list):
        parts = [str(item).strip() for item in bullets[:2] if str(item).strip()]
    else:
        parts = []
    bullet_text = _normalize_compare_text(" ".join(parts)).lower()
    bullet_text = re.sub(
        r"((19|20)\d{2}[./-]\d{1,2}\s*(?:[-–—至到~]\s*((19|20)\d{2}[./-]\d{1,2}|至今|present))?)",
        " ",
        bullet_text,
        flags=re.IGNORECASE,
    )
    bullet_text = re.sub(r"[-–—|:：()\[\]{}【】]+", " ", bullet_text)
    bullet_text = re.sub(r"\s+", " ", bullet_text).strip()
    return name, period, bullet_text


def _is_duplicate_project_record(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_name, left_period, left_sig = _project_content_signature(left)
    right_name, right_period, right_sig = _project_content_signature(right)

    if left_name and right_name and left_name == right_name:
        if not left_sig or not right_sig or left_sig == right_sig:
            return True
        if left_sig and right_sig and (left_sig in right_sig or right_sig in left_sig):
            return True
        if left_period and right_period and left_period == right_period:
            return True

    if left_sig and right_sig and left_sig == right_sig:
        if not left_name or not right_name:
            return True
        if left_name in right_name or right_name in left_name:
            return True

    return False


def _list_bullet_locations(resume_data: dict[str, Any]) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    experiences = resume_data.get("experience", [])
    if not isinstance(experiences, list):
        return locations

    for exp_idx, exp in enumerate(experiences, start=1):
        if not isinstance(exp, dict):
            continue
        company = str(exp.get("company", "")).strip()
        projects = exp.get("projects", [])
        if not isinstance(projects, list):
            continue
        for project_idx, proj in enumerate(projects, start=1):
            if not isinstance(proj, dict):
                continue
            project_name = str(proj.get("name", "项目")).strip() or "项目"
            bullets = proj.get("bullets", [])
            if not isinstance(bullets, list):
                continue
            for bullet_idx, bullet in enumerate(bullets, start=1):
                locations.append(
                    {
                        "exp_index": exp_idx,
                        "project_index": project_idx,
                        "company": company,
                        "project": project_name,
                        "bullet_index": bullet_idx,
                        "text": str(bullet),
                    }
                )
    return locations


def _resolve_revision_target(
    resume_data: dict[str, Any],
    target: RevisionTarget,
) -> dict[str, Any]:
    locations = _list_bullet_locations(resume_data)
    if not locations:
        raise HTTPException(status_code=400, detail="resume has no editable bullets")

    target_project = (target.project or "").strip()
    target_company = (target.company or "").strip()

    if target.exp_index is not None and target.project_index is not None:
        direct = [
            item
            for item in locations
            if item["exp_index"] == target.exp_index
            and item["project_index"] == target.project_index
            and item["bullet_index"] == target.bullet_index
        ]
        if not direct:
            raise HTTPException(
                status_code=404,
                detail=f"revision target not found: exp_index={target.exp_index}, project_index={target.project_index}, bullet_index={target.bullet_index}",
            )
        found = direct[0]
        if target_project and found["project"] != target_project:
            raise HTTPException(
                status_code=409,
                detail=f"revision target project mismatch: expected {target_project}, actual {found['project']}",
            )
        if target_company and found["company"] != target_company:
            raise HTTPException(
                status_code=409,
                detail=f"revision target company mismatch: expected {target_company}, actual {found['company']}",
            )
        return found

    candidates = [
        item
        for item in locations
        if item["bullet_index"] == target.bullet_index
        and (not target_project or item["project"] == target_project)
        and (not target_company or item["company"] == target_company)
    ]

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"revision target not found: project={target_project or '*'}, company={target_company or '*'}, bullet_index={target.bullet_index}",
        )
    if len(candidates) > 1:
        raise HTTPException(
            status_code=409,
            detail="revision target is ambiguous; provide exp_index and project_index",
        )
    return candidates[0]


def _get_bullet_by_path(resume_data: dict[str, Any], exp_index: int, project_index: int, bullet_index: int) -> Optional[str]:
    try:
        exp = resume_data["experience"][exp_index - 1]
        proj = exp["projects"][project_index - 1]
        bullet = proj["bullets"][bullet_index - 1]
    except Exception:
        return None
    return str(bullet) if bullet is not None else None


def _set_bullet_by_path(resume_data: dict[str, Any], exp_index: int, project_index: int, bullet_index: int, value: str) -> None:
    resume_data["experience"][exp_index - 1]["projects"][project_index - 1]["bullets"][bullet_index - 1] = value


def _normalize_for_semantic_compare(value: str) -> str:
    normalized = str(value or "").lower()
    normalized = re.sub(r"[，。；：、“”‘’（）()【】\[\],.;:!?！？\-_/\\\"']", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _is_trivial_wording_change(before: str, after: str) -> bool:
    before_text = str(before or "").strip()
    after_text = str(after or "").strip()
    if not before_text or not after_text:
        return False

    before_norm = _normalize_for_semantic_compare(before_text)
    after_norm = _normalize_for_semantic_compare(after_text)
    if before_norm == after_norm:
        return True

    ratio = SequenceMatcher(None, before_norm, after_norm).ratio()
    if ratio < 0.88:
        return False

    gained_metric = _has_metric(after_text) and not _has_metric(before_text)
    gained_detail = _contains_any(after_text.lower(), DETAIL_HINT_WORDS) and not _contains_any(before_text.lower(), DETAIL_HINT_WORDS)
    gained_responsibility = _contains_any(after_text.lower(), RESPONSIBILITY_WORDS) and not _contains_any(before_text.lower(), RESPONSIBILITY_WORDS)
    before_anchors = _extract_tech_anchor_tokens(before_text)
    after_anchors = _extract_tech_anchor_tokens(after_text)
    gained_anchors = len(after_anchors - before_anchors) > 0

    return not (gained_metric or gained_detail or gained_responsibility or gained_anchors)


def _build_audit_issue_index(audit_result: dict[str, Any]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    index: dict[tuple[str, int], list[dict[str, Any]]] = {}
    issues = audit_result.get("issues", []) if isinstance(audit_result, dict) else []
    if not isinstance(issues, list):
        return index

    for item in issues:
        if not isinstance(item, dict):
            continue
        project = str(item.get("project") or "项目经历").strip() or "项目经历"
        try:
            bullet_index = int(item.get("bullet_index") or 0)
        except Exception:
            bullet_index = 0
        if bullet_index <= 0:
            continue
        index.setdefault((project, bullet_index), []).append(item)
    return index


def _build_change_reason(before: str, after: str, related_issues: list[dict[str, Any]]) -> str:
    reason_parts: list[str] = []

    if related_issues:
        high_dims = [str(i.get("dimension")) for i in related_issues if _normalize_severity(i.get("severity")) == "high"]
        medium_dims = [str(i.get("dimension")) for i in related_issues if _normalize_severity(i.get("severity")) == "medium"]
        if high_dims:
            reason_parts.append(f"优先修复审计高风险项（{', '.join(sorted(set(high_dims)))})")
        elif medium_dims:
            reason_parts.append(f"针对审计中风险项补强（{', '.join(sorted(set(medium_dims)))})")

    if _has_metric(after) and not _has_metric(before):
        reason_parts.append("补充了可验证的量化或观测口径")
    elif not _has_metric(after) and not _has_metric(before):
        reason_parts.append("补充了可追问的验证方式而非虚构数字")

    before_lower = before.lower()
    after_lower = after.lower()
    if _contains_any(after_lower, DETAIL_HINT_WORDS) and not _contains_any(before_lower, DETAIL_HINT_WORDS):
        reason_parts.append("增加方案级细节（设计决策/约束/权衡）")
    if _contains_any(after_lower, RESPONSIBILITY_WORDS) and not _contains_any(before_lower, RESPONSIBILITY_WORDS):
        reason_parts.append("明确了个人职责边界与交付产出")

    before_anchors = _extract_tech_anchor_tokens(before)
    after_anchors = _extract_tech_anchor_tokens(after)
    added_anchors = sorted(after_anchors - before_anchors)
    if added_anchors:
        reason_parts.append(f"保留并强化关键技术锚点（如 {', '.join(added_anchors[:3])}）")

    if not reason_parts:
        reason_parts.append("重构句式以提升技术可追问性，同时保持事实保真")

    return "；".join(dict.fromkeys(reason_parts))


def _derive_non_bullet_changes(
    source_resume_data: dict[str, Any],
    optimized_resume_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Derive changes for non-bullet fields: summary, skills, project names, experience role/description."""
    changes: list[dict[str, Any]] = []

    # Summary
    src_summary = _normalize_compare_text(source_resume_data.get("summary", ""))
    opt_summary = _normalize_compare_text(optimized_resume_data.get("summary", ""))
    if src_summary and opt_summary and src_summary != opt_summary:
        if not _is_trivial_wording_change(src_summary, opt_summary):
            changes.append({
                "project": "摘要",
                "bullet_index": 0,
                "before": src_summary,
                "after": opt_summary,
                "reason": "优化职业摘要表达",
            })

    # Skills buckets
    src_skills = source_resume_data.get("skills", {})
    opt_skills = optimized_resume_data.get("skills", {})
    if isinstance(src_skills, dict) and isinstance(opt_skills, dict):
        for bucket in ("languages", "frameworks", "tools", "domains"):
            src_items = set(_normalize_text_list(src_skills.get(bucket)))
            opt_items = set(_normalize_text_list(opt_skills.get(bucket)))
            added = opt_items - src_items
            removed = src_items - opt_items
            if added or removed:
                parts = []
                if added:
                    parts.append(f"新增 {', '.join(sorted(added)[:5])}")
                if removed:
                    parts.append(f"移除 {', '.join(sorted(removed)[:5])}")
                changes.append({
                    "project": "技能",
                    "bullet_index": 0,
                    "before": ", ".join(sorted(src_items)),
                    "after": ", ".join(sorted(opt_items)),
                    "reason": f"调整{bucket}分组：{'；'.join(parts)}",
                })

    # Experience-level fields: role, description, project names
    src_exps = source_resume_data.get("experience", [])
    opt_exps = optimized_resume_data.get("experience", [])
    if isinstance(src_exps, list) and isinstance(opt_exps, list):
        for exp_idx, src_exp in enumerate(src_exps):
            if not isinstance(src_exp, dict):
                continue
            opt_exp = opt_exps[exp_idx] if exp_idx < len(opt_exps) else None
            if not isinstance(opt_exp, dict):
                continue

            # Role change
            src_role = _normalize_compare_text(src_exp.get("role", ""))
            opt_role = _normalize_compare_text(opt_exp.get("role", ""))
            if src_role and opt_role and src_role != opt_role:
                changes.append({
                    "project": str(src_exp.get("company", "经历")),
                    "bullet_index": 0,
                    "before": src_role,
                    "after": opt_role,
                    "reason": "优化职位表述",
                })

            # Project name and description changes
            src_projs = src_exp.get("projects", [])
            opt_projs = opt_exp.get("projects", [])
            if isinstance(src_projs, list) and isinstance(opt_projs, list):
                for proj_idx, src_proj in enumerate(src_projs):
                    if not isinstance(src_proj, dict):
                        continue
                    opt_proj = opt_projs[proj_idx] if proj_idx < len(opt_projs) else None
                    if not isinstance(opt_proj, dict):
                        continue

                    src_name = _normalize_compare_text(src_proj.get("name", ""))
                    opt_name = _normalize_compare_text(opt_proj.get("name", ""))
                    if src_name and opt_name and src_name != opt_name:
                        changes.append({
                            "project": src_name,
                            "bullet_index": 0,
                            "before": src_name,
                            "after": opt_name,
                            "reason": "优化项目标题表达",
                        })

                    src_desc = _normalize_compare_text(src_proj.get("description", ""))
                    opt_desc = _normalize_compare_text(opt_proj.get("description", ""))
                    if src_desc and opt_desc and src_desc != opt_desc:
                        if not _is_trivial_wording_change(src_desc, opt_desc):
                            changes.append({
                                "project": src_name or "项目",
                                "bullet_index": 0,
                                "before": src_desc,
                                "after": opt_desc,
                                "reason": "优化项目简介表达",
                            })

    # Top-level projects (not under experience)
    src_top_projs = source_resume_data.get("projects", [])
    opt_top_projs = optimized_resume_data.get("projects", [])
    if isinstance(src_top_projs, list) and isinstance(opt_top_projs, list):
        for proj_idx, src_proj in enumerate(src_top_projs):
            if not isinstance(src_proj, dict):
                continue
            opt_proj = opt_top_projs[proj_idx] if proj_idx < len(opt_top_projs) else None
            if not isinstance(opt_proj, dict):
                continue
            src_name = _normalize_compare_text(src_proj.get("name", ""))
            opt_name = _normalize_compare_text(opt_proj.get("name", ""))
            if src_name and opt_name and src_name != opt_name:
                changes.append({
                    "project": src_name,
                    "bullet_index": 0,
                    "before": src_name,
                    "after": opt_name,
                    "reason": "优化项目标题表达",
                })
            src_desc = _normalize_compare_text(src_proj.get("description", ""))
            opt_desc = _normalize_compare_text(opt_proj.get("description", ""))
            if src_desc and opt_desc and src_desc != opt_desc:
                if not _is_trivial_wording_change(src_desc, opt_desc):
                    changes.append({
                        "project": src_name or "项目",
                        "bullet_index": 0,
                        "before": src_desc,
                        "after": opt_desc,
                        "reason": "优化项目简介表达",
                    })

    # Education changes
    src_edu = source_resume_data.get("education", [])
    opt_edu = optimized_resume_data.get("education", [])
    if isinstance(src_edu, list) and isinstance(opt_edu, list):
        for edu_idx, src_e in enumerate(src_edu):
            if not isinstance(src_e, dict):
                continue
            opt_e = opt_edu[edu_idx] if edu_idx < len(opt_edu) else None
            if not isinstance(opt_e, dict):
                continue
            for field in ("major", "degree", "school"):
                src_val = _normalize_compare_text(src_e.get(field, ""))
                opt_val = _normalize_compare_text(opt_e.get(field, ""))
                if src_val and opt_val and src_val != opt_val:
                    changes.append({
                        "project": f"教育经历-{field}",
                        "bullet_index": 0,
                        "before": src_val,
                        "after": opt_val,
                        "reason": f"优化教育经历{field}表达",
                    })

    # Publications changes
    src_pubs = source_resume_data.get("publications", [])
    opt_pubs = optimized_resume_data.get("publications", [])
    if isinstance(src_pubs, list) and isinstance(opt_pubs, list):
        for pub_idx, src_pub in enumerate(src_pubs):
            if pub_idx >= len(opt_pubs):
                changes.append({
                    "project": "论文",
                    "bullet_index": pub_idx,
                    "before": str(src_pub),
                    "after": "",
                    "reason": "论文条目被删除",
                })
                continue
            src_pub_text = _normalize_compare_text(str(src_pub))
            opt_pub_text = _normalize_compare_text(str(opt_pubs[pub_idx]))
            if src_pub_text and opt_pub_text and src_pub_text != opt_pub_text:
                if not _is_trivial_wording_change(src_pub_text, opt_pub_text):
                    changes.append({
                        "project": "论文",
                        "bullet_index": pub_idx,
                        "before": str(src_pub),
                        "after": str(opt_pubs[pub_idx]),
                        "reason": "优化论文条目表达",
                    })

    # Honors / Awards / Certifications / Personal Skills
    for section_key, section_label in [
        ("honors", "荣誉"),
        ("awards", "奖项"),
        ("certifications", "证书"),
        ("personal_skills", "个人技能"),
    ]:
        src_items = source_resume_data.get(section_key, [])
        opt_items = optimized_resume_data.get(section_key, [])
        if isinstance(src_items, list) and isinstance(opt_items, list):
            src_set = set(_normalize_text_list(src_items))
            opt_set = set(_normalize_text_list(opt_items))
            added = opt_set - src_set
            removed = src_set - opt_set
            if added or removed:
                parts = []
                if added:
                    parts.append(f"新增 {', '.join(sorted(added)[:5])}")
                if removed:
                    parts.append(f"移除 {', '.join(sorted(removed)[:5])}")
                changes.append({
                    "project": section_label,
                    "bullet_index": 0,
                    "before": ", ".join(sorted(src_set)),
                    "after": ", ".join(sorted(opt_set)),
                    "reason": f"调整{section_label}：{'；'.join(parts)}",
                })

    # Additional sections
    src_addl = source_resume_data.get("additional_sections", {})
    opt_addl = optimized_resume_data.get("additional_sections", {})
    if isinstance(src_addl, dict) and isinstance(opt_addl, dict):
        for key in set(list(src_addl.keys()) + list(opt_addl.keys())):
            src_items = src_addl.get(key, [])
            opt_items = opt_addl.get(key, [])
            if isinstance(src_items, list) and isinstance(opt_items, list):
                src_set = set(_normalize_text_list(src_items))
                opt_set = set(_normalize_text_list(opt_items))
                added = opt_set - src_set
                removed = src_set - opt_set
                if added or removed:
                    changes.append({
                        "project": key,
                        "bullet_index": 0,
                        "before": ", ".join(sorted(src_set)),
                        "after": ", ".join(sorted(opt_set)),
                        "reason": f"调整{key}版块内容",
                    })

    return changes


def _derive_structured_changes(
    source_resume_data: dict[str, Any],
    optimized_resume_data: dict[str, Any],
    audit_result: dict[str, Any],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    issue_index = _build_audit_issue_index(audit_result)

    for item in _list_bullet_locations(source_resume_data):
        before = item["text"]
        after = _get_bullet_by_path(
            optimized_resume_data,
            item["exp_index"],
            item["project_index"],
            item["bullet_index"],
        )
        if after is None:
            continue
        if _normalize_compare_text(before) == _normalize_compare_text(after):
            continue
        if _is_trivial_wording_change(before, after):
            continue

        related = issue_index.get((item["project"], item["bullet_index"]), [])
        reason = _build_change_reason(before, after, related)
        changes.append(
            {
                "project": item["project"],
                "bullet_index": item["bullet_index"],
                "before": before,
                "after": after,
                "reason": reason,
            }
        )

    # Add non-bullet changes (summary, skills, project names, etc.)
    non_bullet_changes = _derive_non_bullet_changes(source_resume_data, optimized_resume_data)
    changes.extend(non_bullet_changes)

    return changes


def _revert_trivial_rewrites(
    resume_data: dict[str, Any],
    source_resume_data: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    result = _clone_json(resume_data)
    notes: list[str] = []

    for item in _list_bullet_locations(source_resume_data):
        before = item["text"]
        after = _get_bullet_by_path(
            result,
            item["exp_index"],
            item["project_index"],
            item["bullet_index"],
        )
        if after is None:
            continue
        if _normalize_compare_text(before) == _normalize_compare_text(after):
            continue
        if not _is_trivial_wording_change(before, after):
            continue
        _set_bullet_by_path(result, item["exp_index"], item["project_index"], item["bullet_index"], before)
        notes.append("已回退仅同义替换或微调词序的改写")

    return result, notes


def _normalize_text_list(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    result: list[str] = []
    for item in items:
        if isinstance(item, str):
            value = item.strip()
            if value:
                result.append(value)
    return result


# Pre-compiled regex patterns for AI phrase removal (avoid recompiling on every call)
_AI_PHRASE_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(re.escape(src), re.IGNORECASE), dst)
    for src, dst in AI_PHRASE_REPLACEMENTS.items()
]
_EM_DASH_PATTERN = re.compile(r'(?<=[A-Za-z0-9%,.\s])—(?=[A-Za-z0-9%,.\s])')


def _remove_ai_phrases(text: str) -> tuple[str, bool]:
    result = text
    changed = False
    for pattern, dst in _AI_PHRASE_COMPILED:
        new_result = pattern.sub(dst, result)
        if new_result != result:
            changed = True
            result = new_result
    # Replace em-dash only when used as AI-style connector (not Chinese punctuation)
    new_result = _EM_DASH_PATTERN.sub(", ", result)
    if new_result != result:
        changed = True
        result = new_result
    return result, changed


# ---------------------------------------------------------------------------
# Chain-of-Verification: detect unverified claims in optimized bullets
# ---------------------------------------------------------------------------

_FABRICATED_NUMBER_PATTERN = re.compile(
    r"(?:(?:提升|降低|减少|增加|缩短|节省|提高|改善|下降|增长|缩减|降至|升至)"
    r"\s*(?:了|至|到|为)?\s*"
    r"(\d+(?:\.\d+)?)\s*(?:%|倍|ms|s|分钟|小时|天|w|万|k|qps|tps))"
    r"|"
    r"(?:(?:increased|decreased|reduced|improved|boosted|lowered|cut|saved|accelerated|grew|shrunk)"
    r"\s+(?:by|to|from)?\s*"
    r"(\d+(?:\.\d+)?)\s*(?:%|x|ms|s|minutes?|hours?|days?|K|M|B|qps|tps))",
    re.IGNORECASE,
)
_INFLATION_WORDS = {
    "主导", "独立", "牵头", "首创", "独创", "从零",
    "全面", "彻底", "根本性", "颠覆", "革命性",
}
# Role inflation: demotion from humble → inflated role title
_ROLE_INFLATION_MAP = {
    "实习生": {"负责人", "主导", "牵头", "独立"},
    "助理": {"负责人", "主导", "牵头"},
    "初级": {"主导", "牵头", "独立", "从零"},
    "参与": {"主导", "独立", "牵头", "从零"},
}


def _detect_unverified_claims(before: str, after: str) -> list[str]:
    """Compare original and rewritten bullet; return list of potential fabrications."""
    flags: list[str] = []

    # 1. Numbers in 'after' that don't appear in 'before'
    before_nums = set(re.findall(r"\d+(?:\.\d+)?", before))
    after_nums = set(re.findall(r"\d+(?:\.\d+)?", after))
    new_nums = after_nums - before_nums
    if new_nums:
        # Check if these new numbers appear in fabricated metric patterns
        for m in _FABRICATED_NUMBER_PATTERN.finditer(after):
            num = m.group(1) or m.group(2)
            if num in new_nums and num not in before_nums:
                flags.append(f"疑似编造数字 {m.group(0)}")

    # 2. Tech anchors in 'after' not present in 'before' or source context
    before_anchors = _extract_tech_anchor_tokens(before)
    after_anchors = _extract_tech_anchor_tokens(after)
    new_anchors = after_anchors - before_anchors
    if new_anchors:
        # Allow up to 1 new anchor (might be JD-aligned), flag if 2+
        if len(new_anchors) >= 2:
            flags.append(f"新增技术锚点未在原文出现: {', '.join(sorted(new_anchors)[:4])}")

    # 3. Inflation words not in original
    before_infl = {w for w in _INFLATION_WORDS if w in before}
    after_infl = {w for w in _INFLATION_WORDS if w in after}
    new_infl = after_infl - before_infl
    if new_infl:
        flags.append(f"疑似角色膨胀: {', '.join(sorted(new_infl))}")

    # 4. Role inflation: humble role in before → inflated claim in after
    for humble_role, inflated_words in _ROLE_INFLATION_MAP.items():
        if humble_role in before:
            matched = {w for w in inflated_words if w in after and w not in before}
            if matched:
                flags.append(f"疑似角色膨胀: '{humble_role}'→{', '.join(sorted(matched))}")

    return flags


def chain_of_verification(
    source_resume_data: dict[str, Any],
    optimized_resume_data: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Post-processing guard: verify each rewritten bullet against original.

    Returns (possibly modified optimized_resume_data, list of notes).
    Bullets with high-confidence fabrication (new numbers in metric patterns) are
    reverted to original; lower-confidence flags are noted but kept.
    """
    result = _clone_json(optimized_resume_data)
    notes: list[str] = []

    # 1. Experience.projects.bullets
    src_exps = source_resume_data.get("experience", [])
    opt_exps = result.get("experience", [])
    if isinstance(src_exps, list) and isinstance(opt_exps, list):
        for exp_idx, opt_exp in enumerate(opt_exps):
            if not isinstance(opt_exp, dict):
                continue
            src_exp = src_exps[exp_idx] if exp_idx < len(src_exps) and isinstance(src_exps[exp_idx], dict) else {}
            src_projs = src_exp.get("projects", []) if isinstance(src_exp.get("projects"), list) else []
            opt_projs = opt_exp.get("projects", [])
            if not isinstance(opt_projs, list):
                continue

            for proj_idx, opt_proj in enumerate(opt_projs):
                if not isinstance(opt_proj, dict):
                    continue
                src_proj = src_projs[proj_idx] if proj_idx < len(src_projs) and isinstance(src_projs[proj_idx], dict) else {}
                src_bullets = src_proj.get("bullets", []) if isinstance(src_proj.get("bullets"), list) else []

                bullets = opt_proj.get("bullets", [])
                if not isinstance(bullets, list):
                    continue

                new_bullets: list[str] = []
                changed = False

                for bullet_idx, bullet in enumerate(bullets):
                    after_text = str(bullet)
                    before_text = str(src_bullets[bullet_idx]) if bullet_idx < len(src_bullets) else ""

                    if not before_text or _normalize_compare_text(before_text) == _normalize_compare_text(after_text):
                        new_bullets.append(after_text)
                        continue

                    flags = _detect_unverified_claims(before_text, after_text)
                    if not flags:
                        new_bullets.append(after_text)
                        continue

                    has_fabricated_number = any("编造数字" in f for f in flags)
                    if has_fabricated_number:
                        new_bullets.append(before_text)
                        changed = True
                        notes.append(f"已回退疑似编造数字的改写: {'; '.join(flags)}")
                        continue

                    notes.append(f"验证提示（{', '.join(flags[:3])}）")
                    new_bullets.append(after_text)

                if changed:
                    opt_proj["bullets"] = new_bullets

            # 2. Experience-level fields: bullets, responsibilities, achievements
            for field in ("bullets", "responsibilities", "achievements"):
                src_items = src_exp.get(field, [])
                opt_items = opt_exp.get(field, [])
                if not isinstance(src_items, list) or not isinstance(opt_items, list):
                    continue
                new_items: list[str] = []
                field_changed = False
                for i, opt_item in enumerate(opt_items):
                    after_text = str(opt_item)
                    before_text = str(src_items[i]) if i < len(src_items) else ""
                    if not before_text or _normalize_compare_text(before_text) == _normalize_compare_text(after_text):
                        new_items.append(after_text)
                        continue
                    flags = _detect_unverified_claims(before_text, after_text)
                    if not flags:
                        new_items.append(after_text)
                        continue
                    has_fabricated_number = any("编造数字" in f for f in flags)
                    if has_fabricated_number:
                        new_items.append(before_text)
                        field_changed = True
                        notes.append(f"已回退 experience.{field} 疑似编造数字: {'; '.join(flags)}")
                        continue
                    notes.append(f"验证提示-experience.{field}（{', '.join(flags[:3])}）")
                    new_items.append(after_text)
                if field_changed:
                    opt_exp[field] = new_items

    # 3. Top-level projects.bullets
    src_top_projs = source_resume_data.get("projects", [])
    opt_top_projs = result.get("projects", [])
    if isinstance(src_top_projs, list) and isinstance(opt_top_projs, list):
        for proj_idx, opt_proj in enumerate(opt_top_projs):
            if not isinstance(opt_proj, dict):
                continue
            src_proj = src_top_projs[proj_idx] if proj_idx < len(src_top_projs) and isinstance(src_top_projs[proj_idx], dict) else {}
            src_bullets = src_proj.get("bullets", []) if isinstance(src_proj.get("bullets"), list) else []
            opt_bullets = opt_proj.get("bullets", [])
            if not isinstance(opt_bullets, list):
                continue
            new_bullets: list[str] = []
            changed = False
            for bullet_idx, bullet in enumerate(opt_bullets):
                after_text = str(bullet)
                before_text = str(src_bullets[bullet_idx]) if bullet_idx < len(src_bullets) else ""
                if not before_text or _normalize_compare_text(before_text) == _normalize_compare_text(after_text):
                    new_bullets.append(after_text)
                    continue
                flags = _detect_unverified_claims(before_text, after_text)
                if not flags:
                    new_bullets.append(after_text)
                    continue
                has_fabricated_number = any("编造数字" in f for f in flags)
                if has_fabricated_number:
                    new_bullets.append(before_text)
                    changed = True
                    notes.append(f"已回退顶层项目疑似编造数字: {'; '.join(flags)}")
                    continue
                notes.append(f"验证提示-顶层项目（{', '.join(flags[:3])}）")
                new_bullets.append(after_text)
            if changed:
                opt_proj["bullets"] = new_bullets

    # 4. Summary
    src_summary = str(source_resume_data.get("summary", "")).strip()
    opt_summary = str(result.get("summary", "")).strip()
    if src_summary and opt_summary and _normalize_compare_text(src_summary) != _normalize_compare_text(opt_summary):
        flags = _detect_unverified_claims(src_summary, opt_summary)
        if flags:
            has_fabricated_number = any("编造数字" in f for f in flags)
            if has_fabricated_number:
                result["summary"] = src_summary
                notes.append(f"已回退摘要疑似编造数字: {'; '.join(flags)}")
            else:
                notes.append(f"验证提示-摘要（{', '.join(flags[:3])}）")

    # 5. Skills fabrication check
    src_skills = source_resume_data.get("skills", {})
    opt_skills = result.get("skills", {})
    if isinstance(src_skills, dict) and isinstance(opt_skills, dict):
        for skill_field in ("languages", "frameworks", "tools", "domains"):
            src_list = src_skills.get(skill_field, [])
            opt_list = opt_skills.get(skill_field, [])
            if not isinstance(src_list, list) or not isinstance(opt_list, list):
                continue
            src_set = {str(s).strip().lower() for s in src_list if isinstance(s, str) and s.strip()}
            opt_set = {str(s).strip().lower() for s in opt_list if isinstance(s, str) and s.strip()}
            added = opt_set - src_set
            if added:
                # Check if added skills contain tech anchors not present anywhere in the original resume
                src_all_text = json.dumps(source_resume_data, ensure_ascii=False).lower()
                suspicious = [s for s in added if s not in src_all_text]
                if suspicious:
                    # Revert suspicious additions
                    result.setdefault("skills", {})[skill_field] = [
                        s for s in opt_list
                        if str(s).strip().lower() not in suspicious
                    ]
                    notes.append(f"已回退 skills.{skill_field} 疑似新增技术栈: {', '.join(sorted(suspicious)[:5])}")

    return result, notes
