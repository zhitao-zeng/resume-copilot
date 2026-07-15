"""New resume scoring system per the requirements specification."""

import re
from typing import Any, Optional

from pydantic import BaseModel

from resume_validator import check_fabrication_heuristic, check_required_fields, check_time_conflicts
from schemas import FabricationDetail, FabricationReport, MissingField, ResumeScore
from server_runtime import call_llm_typed, llm_enabled, logger, sanitize_user_text


def _collect_bullets(resume_data: dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    for exp in resume_data.get("experience", []):
        if not isinstance(exp, dict):
            continue
        for key in ("function_description", "result_description", "responsibilities", "achievements", "bullets"):
            value = exp.get(key)
            if isinstance(value, str) and value.strip():
                bullets.append(value.strip())
            elif isinstance(value, list):
                bullets.extend(str(item).strip() for item in value if str(item).strip())
        for proj in exp.get("projects", []):
            if isinstance(proj, dict):
                bullets.extend(_collect_project_bullets(proj))
    for proj in resume_data.get("projects", []):
        if isinstance(proj, dict):
            bullets.extend(_collect_project_bullets(proj))
    # Dedup with prefix matching: two bullets sharing the same first 60 chars
    # are the same content (collected from different fields like function_description
    # and responsibilities). Without this, the expression score denominator is inflated
    # 3-4x by duplicate counts, drowning out real STAR/metric signals.
    deduped: list[str] = []
    seen_prefixes: set[str] = set()
    for item in bullets:
        prefix = item[:60].strip()
        if prefix and prefix not in seen_prefixes:
            seen_prefixes.add(prefix)
            deduped.append(item)
    return deduped


def _collect_project_bullets(project: dict[str, Any]) -> list[str]:
    bullets: list[str] = []
    for key in ("description", "function_description", "result_description", "bullets", "achievements"):
        value = project.get(key)
        if isinstance(value, str) and value.strip():
            bullets.append(value.strip())
        elif isinstance(value, list):
            bullets.extend(str(item).strip() for item in value if str(item).strip())
    # Dedup by prefix within project
    deduped: list[str] = []
    seen: set[str] = set()
    for item in bullets:
        prefix = item[:60].strip()
        if prefix and prefix not in seen:
            seen.add(prefix)
            deduped.append(item)
    return deduped


def _compute_fabrication_score(report: FabricationReport) -> int:
    """Graduated fabrication scoring instead of hard zero.

    - 0 items: 100
    - 1-2 items: 60 (significant penalty but not fatal)
    - 3-5 items: 30
    - 6+ items: 0 (severe fabrication)
    """
    if not report.fabrication_found:
        return 100
    count = len(report.details) if hasattr(report, "details") and report.details else 0
    if count >= 6:
        return 0
    if count >= 3:
        return 30
    return 60


def _compute_readability_score(resume_data: dict[str, Any], template: str = "new_standard") -> float:
    score = 0.0

    # Section structure (0-3)
    sections = 0
    for key in ("meta", "education", "experience", "projects", "skills", "summary"):
        val = resume_data.get(key)
        if isinstance(val, (str, list, dict)) and (
            isinstance(val, str) and len(val.strip()) > 5
            or isinstance(val, list) and len(val) > 0
            or isinstance(val, dict) and len(val) > 0
        ):
            sections += 1
    score += min(3.0, (sections / 6) * 3.0)

    # Visual hierarchy (0-3)
    bullets = _collect_bullets(resume_data)
    has_metrics = any(re.search(r"\d+(?:\.\d+)?\s*(?:%|ms|s|分钟|小时|天|倍|w|万|k|qps|tps|fps|mb|gb|条|次|人|个)", str(b), re.IGNORECASE) for b in bullets)
    has_tech = any(re.search(r"[A-Z][A-Za-z0-9_+./-]{2,}", str(b)) for b in bullets)
    if has_metrics:
        score += 1.0
    if has_tech:
        score += 1.0
    if has_metrics and has_tech:
        score += 1.0

    # Layout/typography (0-4)
    if template in ("new_standard", "modern", "classic"):
        score += 2.0
    if resume_data.get("experience") and len(resume_data["experience"]) > 0:
        score += 1.0
    if resume_data.get("skills") and isinstance(resume_data["skills"], dict) and any(len(v) > 0 for v in resume_data["skills"].values() if isinstance(v, list)):
        score += 1.0

    return min(10.0, round(score, 1))


def _compute_completeness_score(
    resume_data: dict[str, Any],
    missing_fields: list[MissingField],
    conflicts: Optional[list[Any]] = None,
) -> float:
    score = 30.0
    hard_missing = 0
    soft_missing = 0
    for item in missing_fields:
        field = getattr(item, "field", "")
        if str(field).startswith(("meta.name", "meta.phone", "meta.email", "education", "experience/projects")):
            hard_missing += 1
        else:
            soft_missing += 1
    score -= hard_missing * 4.0
    score -= soft_missing * 2.0
    if conflicts:
        score -= min(6.0, len(conflicts) * 2.0)
    return max(0.0, min(30.0, round(score, 1)))


def _compute_expression_score(resume_data: dict[str, Any], job_family: Optional[str] = None) -> float:
    # 0-50 scale
    # Check bullet quality
    bullets = _collect_bullets(resume_data)

    # STAR principle check (0-20)
    star_score = 0.0
    if bullets:
        has_action = sum(1 for b in bullets if re.search(r"(负责|实现|设计|开发|优化|构建|主导|参与|推进|分析|创建|部署|审核|授课|诊疗|管理|复盘|制定|研究|训练|搭建|复现|对比|整合|撰写|配置|调度|评估|测试|采集|清洗|标注|编码|调试|移植)", b))
        has_result = sum(1 for b in bullets if re.search(r"(完成|提升|降低|减少|增加|改善|支持|处理|交付|上线|通过|覆盖|解决|输出|沉淀|达成|审核)", b))
        has_context = sum(1 for b in bullets if re.search(r"(基于|通过|使用|采用|利用|针对|为了)", b))
        total = len(bullets)
        star_score = min(20.0, ((has_action + has_result + has_context) / (total * 3)) * 20.0)

    # Personal summary match (0-10)
    summary_score = 0.0
    summary = str(resume_data.get("summary", "")).strip()
    if summary:
        if 20 <= len(summary) <= 100:
            summary_score = 5.0
            if job_family and any(kw in summary for kw in ("技术", "系统", "开发", "产品", "运营", "金融", "临床", "教学", "销售", "设计", "医疗")):
                summary_score = 10.0
        else:
            summary_score = 3.0

    # Specificity and traceability (0-20)
    specificity_score = 0.0
    if bullets:
        has_metrics = sum(1 for b in bullets if re.search(r"\d+(?:\.\d+)?\s*(?:%|ms|s|分钟|小时|天|倍|w|万|k|qps|tps|fps|mb|gb|条|次|人|个|万元|客户|学生|病例)", b, re.IGNORECASE))
        has_specific = sum(1 for b in bullets if re.search(r"[A-Z][A-Za-z0-9_+./-]{3,}|[\u4e00-\u9fff]{2,}(?:审核|诊疗|授课|运营|复盘|风控|贷款|课程|客户|项目|系统)", b))
        total = len(bullets)
        specificity_score = min(20.0, ((has_metrics + has_specific) / (total * 2)) * 20.0)

    return min(50.0, round(star_score + summary_score + specificity_score, 1))


def _compute_response_score(user_report: dict[str, Any]) -> float:
    # 0-10 scale
    score = 3.0

    if user_report.get("summary") or user_report.get("headline"):
        score += 1.0
    if user_report.get("issues") or user_report.get("priority_fixes"):
        score += 1.0
    if user_report.get("missing_field_suggestions"):
        score += 2.0
    if user_report.get("conflict_confirmations"):
        score += 2.0
    if user_report.get("generation_direction"):
        score += 1.0
    if user_report.get("ocr_warnings"):
        score += 1.0

    return min(10.0, score)


def score_resume(
    resume_data: dict[str, Any],
    original_text: Optional[str] = None,
    user_report: Optional[dict[str, Any]] = None,
    job_family: Optional[str] = None,
    user_stage: Optional[str] = None,
    missing_fields: Optional[list[Any]] = None,
    conflicts: Optional[list[Any]] = None,
    fabrication_report: Optional[FabricationReport] = None,
) -> ResumeScore:
    """Compute the new-standard resume score.

    Scoring breakdown (100 pts):
      - fabrication:  pass=100, fail=0 (and total becomes 0)
      - readability:   0-10 (visual structure from rendering)
      - completeness:  0-30 (required fields filled)
      - expression:    0-50 (STAR principle, JD alignment, summary match)
      - response:      0-10 (reply quality: suggestions, confirmations, direction)
    """
    if user_report is None:
        user_report = {}

    # 1. Fabrication check — use provided report when available, heuristic as fallback
    original = original_text or ""
    if fabrication_report is not None:
        fab_report = fabrication_report
    elif original:
        fab_report = check_fabrication_heuristic(original, resume_data)
    else:
        fab_report = FabricationReport(fabrication_found=False, details=[])
    fabrication_score = _compute_fabrication_score(fab_report)

    if fabrication_score == 0:
        return ResumeScore(
            fabrication=0,
            readability=0.0,
            completeness=0.0,
            expression=0.0,
            response=0.0,
            total=0.0,
        )

    # 2. Check required fields
    if missing_fields is None:
        missing_fields = check_required_fields(resume_data, user_stage=user_stage)
    if conflicts is None:
        conflicts = check_time_conflicts(resume_data)

    # 3. Compute individual scores
    readability = _compute_readability_score(resume_data)
    typed_missing = [item for item in missing_fields if isinstance(item, MissingField)]
    if len(typed_missing) != len(missing_fields):
        typed_missing = [
            item if isinstance(item, MissingField) else MissingField(
                field=str(getattr(item, "field", "") or (item.get("field", "") if isinstance(item, dict) else "")),
                label=str(getattr(item, "label", "") or (item.get("label", "") if isinstance(item, dict) else "")),
                reason=str(getattr(item, "reason", "") or (item.get("reason", "") if isinstance(item, dict) else "")),
            )
            for item in missing_fields
        ]
    completeness = _compute_completeness_score(resume_data, typed_missing, conflicts)
    expression = _compute_expression_score(resume_data, job_family)
    response = _compute_response_score(user_report)

    total = round(readability + completeness + expression + response, 1)
    total = min(100.0, max(0.0, total))

    return ResumeScore(
        fabrication=fabrication_score,
        readability=readability,
        completeness=completeness,
        expression=expression,
        response=response,
        total=total,
    )


def _llm_expression_score(resume_data: dict[str, Any], job_family: Optional[str] = None) -> Optional[float]:
    """Try to get expression score from LLM as a fallback."""
    if not llm_enabled():
        return None

    try:
        prompt = (
            "请评估这份简历的内容表达分（0-50），基于以下维度：\n"
            "1) STAR原则是否符合（0-20）\n"
            "2) 个人总结是否凸显岗位匹配能力（0-10）\n"
            "3) 经历描述是否具体、可追问（0-20）\n\n"
            f"目标岗位类型：{job_family or '未指定'}\n\n"
            f"简历内容：{sanitize_user_text(str(resume_data)[:2000])}"
        )
        result = call_llm_typed(_ExpressionScoreModel, "表达式评分", prompt, temperature=0.2)
        return result.get("score")
    except Exception as exc:
        logger.warning("LLM expression score failed: %s", exc)
        return None


class _ExpressionScoreModel(BaseModel):
    score: Optional[float] = None
