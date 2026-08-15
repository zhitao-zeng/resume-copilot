"""Resume field validation, conflict detection, and fabrication checking.

Model classes are re-exported from schemas to avoid duplication.
"""

import re
import unicodedata
from typing import Any, Optional

from pydantic import BaseModel

from schemas import FieldConflict, FabricationDetail, FabricationReport, MissingField
from server_runtime import logger


# ── Shared placeholder detection ──

MISSING_PLACEHOLDER_VALUES: frozenset[str] = frozenset({
    "", "未提供", "未明确", "未知", "暂无",
})

_CONCURRENT_ENGAGEMENT_MARKERS = (
    "实习", "intern", "兼职", "part-time", "顾问", "志愿", "见习", "助理",
    "科研", "项目", "校园", "campus", "支教", "轮转", "实训",
    "规培", "规范化培训", "住院医师培训", "residency", "resident trainee",
)
_TENURE_EXCLUDED_MARKERS = (
    "实习", "intern", "兼职", "part-time", "志愿", "见习", "校园", "campus",
    "支教", "轮转", "实训", "规培", "规范化培训", "住院医师培训",
    "residency", "resident trainee", "研究助理", "科研实习",
)


def _engagement_text(record: dict[str, Any]) -> str:
    return " ".join(
        str(record.get(key) or "").strip().casefold()
        for key in ("company", "organization", "role")
    )


def _is_concurrent_engagement(record: dict[str, Any]) -> bool:
    value = _engagement_text(record)
    return any(marker.casefold() in value for marker in _CONCURRENT_ENGAGEMENT_MARKERS)


def _substantive_experience_months(experience: list[Any]) -> int:
    """Return unioned months for dated non-internship/training employment."""

    from datetime import datetime, timezone

    intervals: list[tuple[int, int]] = []
    now = datetime.now(timezone.utc)
    for record in experience:
        if not isinstance(record, dict):
            continue
        identity = _engagement_text(record)
        if (
            not str(record.get("role") or "").strip()
            or not str(record.get("company") or record.get("organization") or "").strip()
            or any(marker.casefold() in identity for marker in _TENURE_EXCLUDED_MARKERS)
        ):
            continue
        period = str(record.get("period") or "").strip()
        start, end = _parse_period(period)
        if start is None:
            continue
        if end is None:
            if not re.search(r"(?:至今|目前|现在|present|current|now)", period, re.IGNORECASE):
                continue
            end = (now.year, now.month)
        start_month = start[0] * 12 + start[1]
        end_month = end[0] * 12 + end[1]
        if start_month > end_month:
            start_month, end_month = end_month, start_month
        intervals.append((start_month, end_month))

    if not intervals:
        return 0
    intervals.sort()
    merged = [intervals[0]]
    for start_month, end_month in intervals[1:]:
        if start_month <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_month))
        else:
            merged.append((start_month, end_month))
    return sum(end_month - start_month + 1 for start_month, end_month in merged)


def _explicit_tenure_months(value: object) -> int | None:
    text = str(value or "").strip()
    if (
        not text
        or re.search(r"\d+(?:\.\d+)?\s*[-~—至到]\s*\d+(?:\.\d+)?\s*年", text)
        or re.search(r"(?:至少|不少于|超过|约|近)\s*\d|(?:以上|左右|余年|年余|\+)", text)
    ):
        return None
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(年|个月|月)", text)
    if len(matches) != 1:
        return None
    amount, unit = matches[0]
    months = float(amount) * (12 if unit == "年" else 1)
    return max(0, round(months))


def is_missing_placeholder(value: object) -> bool:
    """Check if a field value is effectively empty (placeholder/non-informative).

    Narrow semantic — only values that explicitly indicate 'no information
    provided' or 'unknown'.  Does NOT include '无', '不限', '其他' etc.,
    which may indicate explicit user intent (e.g. '无工作经历')."""
    normalized = str(value or "").strip()
    return normalized in MISSING_PLACEHOLDER_VALUES


def _parse_mm_yyyy(text: str) -> Optional[tuple[int, int]]:
    """Try to parse mm-yyyy or yyyy-mm format, return (year, month)."""
    text = str(text).strip().lower()
    if not text or text == "至今" or text == "present":
        return None

    # Try yyyy-mm, yyyy/mm, or Chinese yyyy年m月.  Resume text commonly
    # contains one-digit months, so accepting only 01..12 loses otherwise
    # obvious overlaps before they reach the user confirmation report.
    m = re.match(
        r"(19|20)(\d{2})(?:[./-]|年\s*)(1[0-2]|0?[1-9])(?!\d)(?:月)?",
        text,
    )
    if m:
        year = int(m.group(1) + m.group(2))
        month = int(m.group(3))
        return (year, month)

    # Try mm-yyyy or mm/yyyy
    m = re.match(r"(1[0-2]|0?[1-9])(?!\d)[./-](19|20)(\d{2})(?!\d)", text)
    if m:
        month = int(m.group(1))
        year = int(m.group(2) + m.group(3))
        return (year, month)

    # Year-only ranges are common in senior resumes. Month precision is not
    # available, but retaining January as a stable comparison point is more
    # accurate than dropping the date and then emitting a false sort warning.
    m = re.match(r"((?:19|20)\d{2})(?!\d)", text)
    if m:
        return (int(m.group(1)), 1)

    return None


def _parse_period(period: str) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
    """Parse a period string like '2021-03 - 2024-06' into (start, end) tuples."""
    if not period or not str(period).strip():
        return (None, None)

    from datetime import datetime

    text = str(period).strip()

    # If the text contains ' - ' (space-hyphen-space) or similar separators,
    # use that to split instead of individual hyphens within dates
    sep_patterns = [" - ", "～", "～", "~", "至", "到"]
    separator = None
    sep_index = -1
    for sep in sep_patterns:
        idx = text.find(sep)
        if idx >= 0:
            if sep_index < 0 or idx < sep_index:
                sep_index = idx
                separator = sep

    if separator and 0 < sep_index < len(text) - 1:
        start_text = text[:sep_index].strip()
        end_text = text[sep_index + len(separator):].strip()
    else:
        # Fallback: split on the first separator found by regex
        parts = re.split(r"[-–—~至到]+", text)
        if len(parts) == 0:
            return (None, None)
        start_text = parts[0].strip()
        end_text = parts[1].strip() if len(parts) >= 2 else ""

    start = _parse_mm_yyyy(start_text) if start_text else None
    end = None
    if end_text and end_text != "至今" and end_text != "present":
        end = _parse_mm_yyyy(end_text)

    return (start, end)


def _period_has_month_precision(period: str) -> bool:
    """Return whether a period states at least one calendar month."""

    text = unicodedata.normalize("NFKC", str(period or ""))
    return bool(re.search(
        r"(?:(?:19|20)\d{2}\s*(?:[./-]\s*(?:1[0-2]|0?[1-9])|"
        r"年\s*(?:1[0-2]|0?[1-9])\s*月)|"
        r"(?:1[0-2]|0?[1-9])\s*[./-]\s*(?:19|20)\d{2})",
        text,
    ))


def _overlap_is_year_boundary_only(
    period_i: str,
    period_j: str,
    start_i: tuple[int, int],
    end_i: tuple[int, int],
    start_j: tuple[int, int],
    end_j: tuple[int, int],
) -> bool:
    """Avoid asserting overlap when coarse year ranges only touch a year.

    ``2012年至2014年`` followed by ``2014年至2015年`` may be perfectly
    sequential; the source does not state the months.  That uncertainty should
    not become a user-visible time-conflict claim.
    """

    return bool(
        max(start_i, start_j) == min(end_i, end_j)
        and (
            not _period_has_month_precision(period_i)
            or not _period_has_month_precision(period_j)
        )
    )


def _date_signatures(text: str) -> set[str]:
    """Return normalized ``YYYYMM`` tokens from common resume date spellings.

    The final resume uses normalized periods such as ``2022-09`` while OCR
    source text may use ``2022.09``, ``2022年9月`` or ``09/2022``.  Comparing
    an endpoint after reversing it to ``09-2022`` with raw source substrings
    caused real dates in the ordinary ``YYYY-MM`` form to be marked as
    fabricated.  Normalize both directions before making that decision.
    """

    value = unicodedata.normalize("NFKC", str(text or ""))
    signatures: set[str] = set()
    pattern = re.compile(
        r"(?<!\d)(?:(?P<year1>(?:19|20)\d{2})\s*(?:[-./]|年)\s*"
        r"(?P<month1>0?[1-9]|1[0-2])\s*月?"
        r"|(?P<month2>0?[1-9]|1[0-2])\s*[-./]\s*"
        r"(?P<year2>(?:19|20)\d{2}))(?!\d)"
    )
    for match in pattern.finditer(value):
        year = match.group("year1") or match.group("year2")
        month = match.group("month1") or match.group("month2")
        signatures.add(f"{year}{int(month):02d}")
    return signatures


def _extract_named_entities(text: str) -> dict[str, set[str]]:
    """Extract potential company/school names from text using heuristics."""
    companies = set()
    schools = set()

    # Heuristic: common patterns for companies/schools
    name_patterns = [
        r"[A-Za-z\u4e00-\u9fff]{2,40}(?:有限公司|集团|公司|大学|学院|研究所|医院|中心|学校|教育|科技)(?![\u4e00-\u9fff])",
        r"[\u4e00-\u9fff]{3,30}(?:大学|学院|医院|学校|集团|公司|研究院|实验室)(?![\u4e00-\u9fff])",
    ]

    # Verbs/phrases that should not prefix a valid entity name
    _verb_prefixes = frozenset({"实习", "参与", "负责", "完成", "撰写", "协助", "推动", "推进", "主导", "参与", "开发", "构建", "建立", "管理", "运营", "分析", "分析", "设计", "优化", "提供", "组织", "筹备", "参与", "开展"})

    for pattern in name_patterns:
        for m in re.finditer(pattern, text):
            cleaned = m.group().strip()
            # Skip matches longer than 30 chars (likely captures surrounding context)
            if len(cleaned) > 30:
                continue
            # Skip if match starts with a verb that would make it a phrase not an entity
            if cleaned and any(cleaned.startswith(vp) or vp in cleaned[:6] for vp in _verb_prefixes):
                continue
            if len(cleaned) >= 2:
                if any(kw in cleaned for kw in ("大学", "学院", "学校", "研究院")):
                    schools.add(cleaned.lower())
                else:
                    companies.add(cleaned.lower())

    return {"companies": companies, "schools": schools}


def calculate_experience_years(experience: list[dict]) -> int:
    """Calculate total work experience years from experience list."""
    from datetime import datetime, timezone

    if not experience or not isinstance(experience, list):
        return 0

    # Collect all (start, end) date ranges
    date_ranges: list[tuple[Any, Any]] = []
    for exp in experience:
        if not isinstance(exp, dict):
            continue
        period = str(exp.get("period", "")).strip()
        if not period:
            continue
        start, end = _parse_period(period)
        date_ranges.append((start, end))

    if not date_ranges:
        return 0

    # Merge overlapping ranges and calculate total months
    def to_months(y, m):
        return y * 12 + m

    intervals = []
    now = datetime.now(timezone.utc)
    now_year, now_month = now.year, now.month
    for start, end in date_ranges:
        if start is None:
            continue
        if end is None:
            end = (now_year, now_month)  # Still working there
        s = to_months(*start)
        e = to_months(*end)
        if s > e:
            s, e = e, s
        intervals.append((s, e))

    if not intervals:
        return 0

    # Merge overlapping intervals
    intervals.sort()
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + 1:  # +1 for adjacent months
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    total_months = sum(e - s + 1 for s, e in merged)
    return max(1, round(total_months / 12))


def _compare_period(period: str) -> Optional[tuple[int, int]]:
    """For sorting: return the start month for comparison (by period string like 'mm-yyyy - mm-yyyy')."""
    if not period:
        return None
    start, _ = _parse_period(str(period).strip())
    return start


def check_sort_order(resume_data: dict[str, Any]) -> list[FieldConflict]:
    """Check that education and experience are sorted by start time descending (newest first)."""
    conflicts: list[FieldConflict] = []

    # Education sort check
    education = resume_data.get("education", [])
    if isinstance(education, list) and len(education) >= 2:
        for i in range(len(education) - 1):
            edu_i = education[i] if isinstance(education, list) else {}
            edu_j = education[i + 1] if isinstance(education, list) else {}
            if not isinstance(edu_i, dict) or not isinstance(edu_j, dict):
                continue
            period_i = str(edu_i.get("period", "")).strip()
            period_j = str(edu_j.get("period", "")).strip()
            start_i = _compare_period(period_i)
            start_j = _compare_period(period_j)
            if start_i and start_j and start_i < start_j:
                school_i = str(edu_i.get("school", "")).strip() or f"教育经历{i+1}"
                school_j = str(edu_j.get("school", "")).strip() or f"教育经历{i+2}"
                conflicts.append(FieldConflict(
                    field="education_order",
                    description=f"教育经历未按要求按开始时间倒序排列：{school_i} 在 {school_j} 之前，但 {school_i} 的开始时间晚于 {school_j}",
                ))

    # Experience sort check
    experience = resume_data.get("experience", [])
    if isinstance(experience, list) and len(experience) >= 2:
        for i in range(len(experience) - 1):
            exp_i = experience[i]
            exp_j = experience[i + 1]
            if not isinstance(exp_i, dict) or not isinstance(exp_j, dict):
                continue
            period_i = str(exp_i.get("period", "")).strip()
            period_j = str(exp_j.get("period", "")).strip()
            start_i = _compare_period(period_i)
            start_j = _compare_period(period_j)
            if start_i and start_j and start_i < start_j:
                company_i = str(exp_i.get("company", "")).strip() or f"工作经历{i+1}"
                company_j = str(exp_j.get("company", "")).strip() or f"工作经历{i+2}"
                conflicts.append(FieldConflict(
                    field="experience_order",
                    description=f"工作经历未按要求按开始时间倒序排列：{company_i} 在 {company_j} 之前，但 {company_i} 的开始时间晚于 {company_j}",
                ))

    return conflicts


def _has_required_experience_fields(exp: dict) -> bool:
    """Check if an experience entry has all required fields."""
    if not isinstance(exp, dict):
        return False
    return bool(
        str(exp.get("period", "")).strip()
        and str(exp.get("company") or exp.get("organization") or "").strip()
        and str(exp.get("role", "")).strip()
    )


def _has_required_project_fields(proj: dict) -> bool:
    """Check if a project entry has all required fields."""
    if not isinstance(proj, dict):
        return False
    required = ["period", "name"]
    return all(str(proj.get(f, "")).strip() for f in required)


def _has_any_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return False


def _has_function_description(payload: dict[str, Any]) -> bool:
    return (
        _has_any_text(payload.get("function_description"))
        or _has_any_text(payload.get("responsibilities"))
        or _has_any_text(payload.get("bullets"))
    )


def _has_result_description(payload: dict[str, Any]) -> bool:
    if _has_any_text(payload.get("result_description")) or _has_any_text(payload.get("achievements")):
        return True
    bullets = payload.get("bullets", [])
    if isinstance(bullets, str):
        bullets = [bullets]
    if not isinstance(bullets, list):
        return False
    result_signal = re.compile(
        r"(?:结果|成果|交付|产出|完成|上线|发布|发表|获奖|录用|通过|解决|达成|达到|"
        r"提升|提高|降低|增长|减少|缩短|节省|覆盖|复核|验证|入选|落地|闭环|"
        r"delivered|launched|published|achieved|improved|reduced|increased|resolved)",
        re.IGNORECASE,
    )
    return any(result_signal.search(str(item)) for item in bullets if str(item).strip())


def _field_exists_in_source(key: str, source_text: str) -> bool:
    """Check if a field value likely exists in source_text despite extraction failure."""
    if not source_text or len(source_text.strip()) < 20:
        return False
    source_lower = source_text.lower()
    if key == "phone":
        return bool(re.search(r"1[3-9]\d\s*-?\s*\d{4}\s*-?\s*\d{4}", source_text))
    if key == "email":
        return bool(re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", source_text))
    if key == "name":
        return bool(re.search(r"(?:姓名|name|昵称|称呼)[:：\s]*([一-鿿]{2,8})", source_text))
    return False


def check_required_fields(
    resume_data: dict[str, Any],
    user_stage: Optional[str] = None,
    source_text: str = "",
) -> list[MissingField]:
    """Check that all required fields are present in resume_data.

    When source_text is provided, fields missing in resume_data but present
    in source_text (via regex) are marked as extraction_lost rather than
    not_provided — the field exists in the original document but the system
    failed to extract it cleanly.

    Args:
        resume_data: The structured resume data.
        user_stage: One of 'student', 'experienced', 'job_seeker'.
        source_text: Original CV/query text for extraction-loss detection.
    """
    if not isinstance(resume_data, dict):
        return [MissingField(field="root", label="简历数据", reason="简历数据为空，无法校验")]

    missing: list[MissingField] = []
    meta = resume_data.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    stage_value = getattr(user_stage, "value", user_stage)

    # Required meta fields
    for key, label in [("name", "姓名"), ("phone", "联系电话"), ("email", "邮箱")]:
        value = str(meta.get(key, "")).strip()
        if is_missing_placeholder(value):
            field_source = "not_provided"
            reason = f"{label}为必填项，请在简历中补充"
            if source_text and _field_exists_in_source(key, source_text):
                field_source = "extraction_lost"
                reason = f"系统未能从原文中稳定识别{label}，请确认后补充"
            missing.append(MissingField(
                field=f"meta.{key}",
                label=label,
                reason=reason,
                source=field_source,
            ))

    # Education is required
    education = resume_data.get("education", [])
    if not isinstance(education, list) or len(education) == 0:
        missing.append(MissingField(
            field="education",
            label="教育背景",
            reason="教育背景为必填项，请补充至少一段教育经历",
        ))
    else:
        for idx, edu in enumerate(education):
            if not isinstance(edu, dict):
                continue
            for key, label in [("school", "学校名称"), ("degree", "学位"), ("major", "专业名称"), ("period", "就读时间")]:
                value = str(edu.get(key, "")).strip()
                if is_missing_placeholder(value):
                    missing.append(MissingField(
                        field=f"education[{idx}].{key}",
                        label=label,
                        reason=f"第{idx+1}段学习记录缺少{label}，请按真实信息补充",
                    ))
        if not str(meta.get("education_level", "")).strip():
            degrees = [
                str(edu.get("degree", "")).strip()
                for edu in education
                if isinstance(edu, dict) and str(edu.get("degree", "")).strip()
            ]
            if degrees:
                meta["education_level"] = degrees[0]
            else:
                missing.append(MissingField(
                    field="meta.education_level",
                    label="学历",
                    reason="请补充最高学历或学位信息",
                ))

    # Summary is required.  Allow several complete factual sentences; the
    # renderer handles pagination and must not force a mid-sentence cut or
    # misreport a complete summary as missing merely because it exceeds an
    # arbitrary one-page-era character limit.
    summary = str(resume_data.get("summary", "")).strip()
    if not summary:
        missing.append(MissingField(
            field="summary",
            label="个人总结",
            reason="个人总结为必填项，请补充真实职业背景、核心优势和求职方向",
        ))

    # At least one substantive experience type is required. Work seniority is
    # optional, and campus/research records are valid for students and other
    # profiles without conventional employment.
    experience = resume_data.get("experience", [])
    projects = resume_data.get("projects", [])
    campus = resume_data.get("campus_experience", resume_data.get("activities", []))
    research = resume_data.get("research", [])
    if not isinstance(experience, list):
        experience = []
    if not isinstance(projects, list):
        projects = []
    if not isinstance(campus, list):
        campus = []
    if not isinstance(research, list):
        research = []

    has_exp = len(experience) > 0
    has_proj = len(projects) > 0
    has_campus = len(campus) > 0
    has_research = len(research) > 0

    # Check nested projects in experience
    nested_proj_count = 0
    if has_exp:
        for exp in experience:
            if isinstance(exp, dict):
                exp_projects = exp.get("projects", [])
                if isinstance(exp_projects, list):
                    nested_proj_count += len(exp_projects)

    has_any_experience = any((
        has_exp, has_proj, has_campus, has_research, nested_proj_count > 0,
    ))
    if not has_any_experience:
        missing.append(MissingField(
            field="experience/projects/campus",
            label="经历",
            reason="工作经历/实习经历/项目经历/校园经历不可全部为空，请至少补充一项经历",
        ))

    # Report partial records field-by-field. A broad "work experience is
    # missing" message contradicts a resume that already contains the record
    # and gives the user no indication of which source fact to add.
    if has_exp:
        for idx, exp in enumerate(experience):
            if not isinstance(exp, dict):
                continue
            identity_values = {
                "company": exp.get("company") or exp.get("organization"),
                "role": exp.get("role"),
                "period": exp.get("period"),
            }
            for key, label in (
                ("company", "公司或组织名称"),
                ("role", "岗位名称"),
                ("period", "任职时间"),
            ):
                if is_missing_placeholder(identity_values[key]):
                    missing.append(MissingField(
                        field=f"experience[{idx}].{key}",
                        label=label,
                        reason=f"第{idx+1}段任职记录缺少{label}，请按原始事实补充",
                    ))
            if not _has_function_description(exp):
                missing.append(MissingField(
                    field=f"experience[{idx}].function_description",
                    label="工作职能描述",
                    reason=f"第{idx+1}段任职记录缺少具体负责内容，请补充",
                ))
            if not _has_result_description(exp):
                missing.append(MissingField(
                    field=f"experience[{idx}].result_description",
                    label="工作成果描述",
                    reason=f"第{idx+1}段任职记录缺少可核验成果，请补充交付结果或业务影响",
                ))

    # Check projects have all required fields if user provided any
    for idx, proj in enumerate(projects):
        if isinstance(proj, dict) and any(
            _has_any_text(proj.get(key))
            for key in ("name", "organization", "company", "role", "period", "bullets", "description")
        ):
            for key, label in (("name", "项目名称"), ("period", "项目时间")):
                if is_missing_placeholder(proj.get(key)):
                    missing.append(MissingField(
                        field=f"projects[{idx}].{key}",
                        label=label,
                        reason=f"第{idx+1}个项目缺少{label}，请按原始事实补充",
                    ))
            # Independent, course and open-source projects do not necessarily
            # have a company/organization.  Do not turn an optional affiliation
            # into a false missing-field warning.
            if not str(proj.get("description", "")).strip() and not _has_function_description(proj):
                missing.append(MissingField(
                    field=f"projects[{idx}].description",
                    label="项目描述",
                    reason=f"第{idx+1}个项目缺少背景和目标说明，请补充",
                ))
            if not _has_function_description(proj):
                missing.append(MissingField(
                    field=f"projects[{idx}].function_description",
                    label="项目工作职能",
                    reason=f"第{idx+1}个项目缺少本人职责或具体行动，请补充",
                ))
            if not _has_result_description(proj):
                missing.append(MissingField(
                    field=f"projects[{idx}].result_description",
                    label="项目工作成果",
                    reason=f"第{idx+1}个项目缺少可核验成果，请补充结果、影响或验证口径",
                ))

    for section_name, records, label, identity_labels in (
        (
            "campus_experience",
            campus,
            "校园/社会活动",
            {"company": "组织名称", "role": "担任角色", "period": "活动时间"},
        ),
        (
            "research",
            research,
            "科研记录",
            {"company": "研究机构", "role": "研究主题", "period": "研究时间"},
        ),
    ):
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            # V2 canonical names are accepted as well as renderer/V1 names.
            identity_values = {
                "company": record.get("company") or record.get("organization") or record.get("institution"),
                "role": record.get("role") or record.get("topic"),
                "period": record.get("period"),
            }
            for key, field_label in identity_labels.items():
                if not str(identity_values.get(key) or "").strip():
                    missing.append(MissingField(
                        field=f"{section_name}[{idx}].{key}",
                        label=field_label,
                        reason=f"第{idx+1}段{label}缺少{field_label}，请按原始事实补充",
                    ))
            if not _has_function_description(record):
                missing.append(MissingField(
                    field=f"{section_name}[{idx}].function_description",
                    label=f"{label}具体行动",
                    reason=f"{label}第{idx+1}段缺少个人职责或具体行动，请补充",
                ))
            if not _has_result_description(record):
                missing.append(MissingField(
                    field=f"{section_name}[{idx}].result_description",
                    label=f"{label}成果",
                    reason=f"{label}第{idx+1}段缺少交付结果、影响或验证口径，请补充真实信息",
                ))

    # Skills are optional but we suggest
    skills = resume_data.get("skills", {})
    if not isinstance(skills, dict) or not any(
        isinstance(value, list) and any(str(item).strip() for item in value)
        for value in skills.values()
    ):
        missing.append(MissingField(
            field="skills",
            label="技能",
            reason="建议补充技能信息以提升简历完整度",
        ))

    return missing


def check_time_conflicts(resume_data: dict[str, Any]) -> list[FieldConflict]:
    """Check for time conflicts in education and experience periods."""
    conflicts: list[FieldConflict] = []

    if not isinstance(resume_data, dict):
        return conflicts

    def _add_conflict(field: str, desc: str) -> None:
        conflicts.append(FieldConflict(field=field, description=desc))

    # Education time conflicts
    education = resume_data.get("education", [])
    if isinstance(education, list) and len(education) >= 2:
        for i in range(len(education)):
            for j in range(i + 1, len(education)):
                edu_i = education[i] if isinstance(education, list) else {}
                edu_j = education[j] if isinstance(education, list) else {}
                if not isinstance(edu_i, dict) or not isinstance(edu_j, dict):
                    continue

                period_i = str(edu_i.get("period", "")).strip()
                period_j = str(edu_j.get("period", "")).strip()

                if not period_i or not period_j:
                    continue

                start_i, end_i = _parse_period(period_i)
                start_j, end_j = _parse_period(period_j)

                if start_i and end_j and start_j and end_i:
                    if start_i <= end_j and start_j <= end_i:
                        if _overlap_is_year_boundary_only(
                            period_i, period_j, start_i, end_i, start_j, end_j,
                        ):
                            continue
                        # Allow overlap when both entries share the same school
                        # (dual degree, minor, concurrent master+phd, etc.)
                        school_i = str(edu_i.get("school", "")).strip()
                        school_j = str(edu_j.get("school", "")).strip()
                        if school_i and school_j and school_i == school_j:
                            continue
                        school_label_i = school_i or f"教育经历{i+1}"
                        school_label_j = school_j or f"教育经历{j+1}"
                        _add_conflict(
                            "education",
                            f"教育经历时间冲突：{school_label_i}({period_i}) 与 {school_label_j}({period_j}) 时间段有重叠",
                        )

    # Experience time conflicts
    experience = resume_data.get("experience", [])
    if isinstance(experience, list) and len(experience) >= 2:
        for i in range(len(experience)):
            for j in range(i + 1, len(experience)):
                exp_i = experience[i]
                exp_j = experience[j]
                if not isinstance(exp_i, dict) or not isinstance(exp_j, dict):
                    continue

                period_i = str(exp_i.get("period", "")).strip()
                period_j = str(exp_j.get("period", "")).strip()

                if not period_i or not period_j:
                    continue

                start_i, end_i = _parse_period(period_i)
                start_j, end_j = _parse_period(period_j)

                if start_i and end_j and start_j and end_i:
                    if start_i <= end_j and start_j <= end_i:
                        if _overlap_is_year_boundary_only(
                            period_i, period_j, start_i, end_i, start_j, end_j,
                        ):
                            continue
                        company_i = str(exp_i.get("company", "")).strip() or f"工作经历{i+1}"
                        company_j = str(exp_j.get("company", "")).strip() or f"工作经历{j+1}"
                        role_i = str(exp_i.get("role", "")).strip()
                        role_j = str(exp_j.get("role", "")).strip()
                        flexible_hint = (
                            _is_concurrent_engagement(exp_i)
                            or _is_concurrent_engagement(exp_j)
                        )
                        company_value_i = str(
                            exp_i.get("company") or exp_i.get("organization") or ""
                        ).strip()
                        company_value_j = str(
                            exp_j.get("company") or exp_j.get("organization") or ""
                        ).strip()
                        if not flexible_hint:
                            _add_conflict(
                                "experience",
                                f"工作经历时间可能冲突：{company_value_i or company_i}（{role_i}）({period_i}) 与 {company_value_j or company_j}（{role_j}）({period_j}) 时间段有重叠，请确认是否为并行任职或时间填写有误",
                            )

    # Cross-check: experience overlaps with education
    # Exempt internships/campus/part-time — overlap with study is normal for students
    if isinstance(experience, list) and isinstance(education, list):
        for exp in experience:
            if not isinstance(exp, dict):
                continue
            exp_period = str(exp.get("period", "")).strip()
            if not exp_period:
                continue
            exp_start, exp_end = _parse_period(exp_period)
            if not exp_start:
                continue

            # Skip cross-check for internships, campus jobs, part-time work and
            # explicit professional training (for example medical residency).
            # Their overlap with formal education is normally expected.
            role = str(exp.get("role", "") or "").lower()
            company = str(
                exp.get("company") or exp.get("organization") or ""
            ).lower()
            if _is_concurrent_engagement(exp):
                continue
            # Without an explicit organization and title there is not enough
            # evidence to classify the record as incompatible full-time work.
            if not role or not company:
                continue

            for edu in education:
                if not isinstance(edu, dict):
                    continue
                edu_period = str(edu.get("period", "")).strip()
                if not edu_period:
                    continue
                edu_start, edu_end = _parse_period(edu_period)
                if not edu_start or not edu_end or not exp_end:
                    continue

                # Check overlap
                if exp_start <= edu_end and edu_start <= exp_end:
                    company = str(exp.get("company", "")).strip() or "工作"
                    school = str(edu.get("school", "")).strip() or "学校"
                    _add_conflict(
                        "cross_check",
                        f"工作经历与教育经历时间可能重叠：{company}（{exp_period}）与 {school}（{edu_period}）",
                    )

    # An explicit seniority label and the dated employment ledger are two
    # independent user facts. Surface a material mismatch for confirmation
    # instead of silently choosing one. Internships and formal training do not
    # inflate the full-time ledger used for this comparison.
    meta = resume_data.get("meta", {})
    seniority = str(meta.get("work_experience") or "").strip() if isinstance(meta, dict) else ""
    stated_months = _explicit_tenure_months(seniority)
    dated_months = _substantive_experience_months(experience) if isinstance(experience, list) else 0
    if (
        stated_months is not None
        and dated_months
        and abs(stated_months - dated_months) >= 18
    ):
        dated_years = dated_months / 12
        dated_label = (
            str(int(dated_years))
            if dated_years.is_integer()
            else f"{dated_years:.1f}".rstrip("0").rstrip(".")
        )
        _add_conflict(
            "meta.work_experience",
            f"工作年限口径可能不一致：基本信息填写“{seniority}”，"
            f"非实习/培训类经历日期合计约{dated_label}年，请确认是否按相关经验计算，"
            "或更新工作年限。",
        )

    return conflicts


def check_fabrication_heuristic(original_text: str, resume_data: dict[str, Any]) -> FabricationReport:
    """Heuristic fabrication check: verify resume entities against original input text."""
    if not original_text or not original_text.strip():
        return FabricationReport(fabrication_found=False, details=[])

    original_lower = original_text.lower()

    # Extract entities from original text
    orig_entities = _extract_named_entities(original_text)
    orig_companies = orig_entities["companies"]
    orig_schools = orig_entities["schools"]

    fab_details: list[FabricationDetail] = []

    def _add_detail(kind: str, content: str, reason: str) -> None:
        content = str(content or "").strip()
        if not content:
            return
        if any(item.type == kind and item.content == content for item in fab_details):
            return
        fab_details.append(FabricationDetail(type=kind, content=content, reason=reason))

    def _normalize_for_match(text: str) -> str:
        """Normalize text for comparison: full-width→half-width, NFKC, whitespace collapse, strip quotes."""
        text = str(text or "")
        # Common full-width to half-width mappings (including quote variants)
        full_to_half = str.maketrans(
            "（）［］｛｝！＂＃＄％＆＇＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～"
            "０１２３４５６７８９"
            "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
            "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
            "“”‘’「」『』",  # Chinese/smart quotes
            "()[]{}!\"#$%&'*+,-./:;<=>?@[\\]^_`{|}~"
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "\"\"''\"\"\"\"",  # → half-width quotes
        )
        text = text.translate(full_to_half)
        text = unicodedata.normalize("NFKC", text)
        # Strip all quote/bracket characters for comparison
        text = re.sub(r"[\"\'\"''「」『』]", "", text)
        # Normalize paren spacing: "foo (bar)" ↔ "foo(bar)"
        text = re.sub(r"\s*\(\s*", "(", text)
        text = re.sub(r"\s*\)\s*", ")", text)
        # Collapse repeated whitespace, strip
        text = " ".join(text.split())
        return text.strip()

    def _supported_text(value: str, *, allow_unverifiable_short: bool = True) -> bool:
        value = str(value or "").strip()
        if not value or value in {"至今", "present", "Present"}:
            return True
        if value.lower() in original_lower:
            return True
        # Normalized comparison: handle full-width/half-width, NFKC differences
        norm_value = _normalize_for_match(value.lower())
        if norm_value in _normalize_for_match(original_lower):
            return True
        # Fuzzy company/school name: try removing common suffixes
        # e.g. "中国银行股份有限公司" should match "中国银行" in source
        if len(value) >= 4:
            _fuzzy_val = re.sub(
                r"(股份有限公司|有限责任公司|有限公司|集团|实验室|研究院|中心|（.*）|\(.*\))$",
                "", norm_value
            ).strip()
            if _fuzzy_val and len(_fuzzy_val) >= 3 and _fuzzy_val in _normalize_for_match(original_lower):
                return True
            # Reverse: source has full name, resume has short name
            # e.g. source has "中国银行股份有限公司", resume has "中国银行"
            _src_norm = _normalize_for_match(original_lower)
            _fuzzy_src = re.sub(
                r"(股份有限公司|有限责任公司|有限公司|集团|实验室|研究院|中心|（.*）|\(.*\))$",
                "", _src_norm
            ).strip()
            if _fuzzy_src and len(_fuzzy_src) >= 3 and norm_value in _fuzzy_src:
                return True
        # Normalized dates like 03-2020 should match source variants:
        # 2020年3月, 2020.03, 2020-3, 2020/03, 3/2020, 2020.3, March 2020
        date_match = re.match(r"^(0?[1-9]|1[0-2])[-/](19\d{2}|20\d{2})$", value)
        if date_match:
            month, year = int(date_match.group(1)), date_match.group(2)
            if f"{year}{month:02d}" in _date_signatures(original_text):
                return True
            year_short = year[-2:]
            norm_text = _normalize_for_match(original_text.lower())
            year_ok = (
                year in original_text
                or year in norm_text
                or f"{year_short}年" in original_text
                or f"{year_short} 年" in original_text
                or f".{year_short}" in original_text
                or f"/{year_short}" in original_text
            )
            month_ok = (
                f"{month}月" in original_text
                or f"{month:02d}月" in original_text
                or f"{month}-" in original_text
                or f"{month:02d}-" in original_text
                or f"/{month}" in original_text
                or f"/{month:02d}" in original_text
                or f"{month}/" in original_text
                or f"{month:02d}/" in original_text
                or f".{month}" in original_text
                or f".{month:02d}" in original_text
                or f"年{month}月" in original_text
            )
            return year_ok and month_ok
        # yy.mm format (20.03 → 03-2020)
        date_match2 = re.match(r"^(\d{2})\.(\d{2})$", value)
        if date_match2:
            yr, mo = int(date_match2.group(1)), int(date_match2.group(2))
            if 20 <= yr <= 99:
                full_year = f"20{yr:02d}"
                if f"{full_year}{mo:02d}" in _date_signatures(original_text):
                    return True
                month_ok = (
                    f"{mo}月" in original_text or f"{mo:02d}月" in original_text
                    or f"{mo}-" in original_text or f"{mo:02d}-" in original_text
                    or f"/{mo}" in original_text or f"/{mo:02d}" in original_text
                )
                year_ok = (
                    f"{yr:02d}" in original_text or f"{yr}" in original_text
                    or full_year in original_text
                )
                return year_ok and month_ok
        if len(value) <= 2:
            return allow_unverifiable_short
        return False

    def _check_period(period: str, label: str) -> None:
        period = str(period or "").strip()
        if not period:
            return
        start, end = _parse_period(period)
        values: list[str] = []
        if start:
            values.append(f"{start[1]:02d}-{start[0]}")
        if end:
            values.append(f"{end[1]:02d}-{end[0]}")
        if values and not all(_supported_text(value) for value in values):
            _add_detail("date", period, f"{label}时间未能在用户原始输入中找到对应依据")

    def _collect_resume_text_values(payload: Any) -> list[str]:
        values: list[str] = []
        if isinstance(payload, str):
            if payload.strip():
                values.append(payload.strip())
        elif isinstance(payload, list):
            for item in payload:
                values.extend(_collect_resume_text_values(item))
        elif isinstance(payload, dict):
            for key, value in payload.items():
                if key in {"meta", "summary", "phone", "email", "period", "start_date", "end_date", "work_experience", "education_level"}:
                    continue
                values.extend(_collect_resume_text_values(value))
        return values

    # Check company names in experience
    experience = resume_data.get("experience", [])
    if isinstance(experience, list):
        for exp in experience:
            if not isinstance(exp, dict):
                continue
            company = str(exp.get("company", "")).strip()
            if not company:
                continue

            company_lower = company.lower()
            if company_lower not in original_lower:
                found_partial = False
                for orig_comp in orig_companies:
                    if orig_comp in company_lower or company_lower in orig_comp:
                        found_partial = True
                        break

                if not found_partial:
                    _add_detail("company", company, "该公司/机构名称未出现在用户原始输入中")

            role = str(exp.get("role", "")).strip()
            if role and not _supported_text(role, allow_unverifiable_short=False):
                _add_detail("role", role, "该岗位名称未出现在用户原始输入中")
            _check_period(str(exp.get("period", "")).strip(), company or role or "工作经历")

    # Check school names in education
    education = resume_data.get("education", [])
    if isinstance(education, list):
        for edu in education:
            if not isinstance(edu, dict):
                continue
            school = str(edu.get("school", "")).strip()
            if not school:
                continue

            school_lower = school.lower()
            # Check against entity-extracted schools (authoritative).
            # A raw substring match (e.g. "全国大学" in "全国大学生创新创业大赛三等奖")
            # does NOT count as entity support — the entity extractor knows
            # which spans represent real institutions.
            _entity_match = any(
                orig_school == school_lower
                for orig_school in orig_schools
            ) or any(
                school_lower in orig_school
                for orig_school in orig_schools
            )
            if not _entity_match:
                # Institution names are often followed directly by Chinese
                # context ("北京邮电大学读人工智能").  The entity regex above
                # intentionally avoids "全国大学生..."; retain that protection
                # while accepting explicit contextual occurrences.
                start = original_lower.find(school_lower)
                while start >= 0:
                    next_char = original_lower[start + len(school_lower):start + len(school_lower) + 1]
                    if next_char not in {"生", "城", "排名"}:
                        _entity_match = True
                        break
                    start = original_lower.find(school_lower, start + 1)
            if not _entity_match:
                _add_detail("school", school, "该校名称未出现在用户原始输入中")

            for key, label in (("degree", "学位/学历"), ("major", "专业")):
                value = str(edu.get(key, "")).strip()
                if value and not _supported_text(value, allow_unverifiable_short=False):
                    _add_detail(key, value, f"该{label}未出现在用户原始输入中")
            _check_period(str(edu.get("period", "")).strip(), school or "教育经历")

    projects = resume_data.get("projects", [])
    if isinstance(projects, list):
        for proj in projects:
            if not isinstance(proj, dict):
                continue
            for key, label in (("name", "项目名称"), ("company", "项目归属"), ("role", "项目角色")):
                value = str(proj.get(key, "")).strip()
                if not value:
                    continue
                if key == "role" and value in {"项目负责人", "负责人", "学生", "志愿者", "研究项目", "成员", "组长", "开发者", "工程师", "算法工程师"}:
                    continue  # Generic role — not a fabrication risk
                # Strip common Chinese role suffixes ("者","员","人","师") before checking
                if key == "role" and any(value.endswith(s) for s in ("者", "员", "人", "师")):
                    stem = value[:-1]
                    if stem and len(stem) >= 2 and _supported_text(
                        stem,
                        allow_unverifiable_short=False,
                    ):
                        continue
                if key == "company" and re.search(r"(?:指导|导师|教授|大学|学院|学校)", value):
                    # Project company is a supervisor's institution — extract core name and check
                    core = re.sub(r"\s*[（(].*[）)]\s*", "", value).strip()
                    if core and (
                        _supported_text(core, allow_unverifiable_short=False)
                        or _supported_text(value, allow_unverifiable_short=False)
                    ):
                        continue
                if not _supported_text(value, allow_unverifiable_short=False):
                    _add_detail(key, value, f"该{label}未出现在用户原始输入中")
            _check_period(str(proj.get("period", "")).strip(), str(proj.get("name", "") or "项目经历"))

    skills = resume_data.get("skills", {})
    if isinstance(skills, dict):
        for bucket, values in skills.items():
            if not isinstance(values, list):
                continue
            for value in values:
                text = str(value or "").strip()
                if text and len(text) > 1 and not _supported_text(
                    text,
                    allow_unverifiable_short=False,
                ):
                    _add_detail("skill", text, f"技能/领域项未出现在用户原始输入中（{bucket}）")

    # Check meta.work_experience — common fabrication for student profiles.
    # _supported_text has len<=2 bypass, but "3年"/"5年" are critical to verify.
    meta = resume_data.get("meta", {})
    if isinstance(meta, dict):
        work_exp = str(meta.get("work_experience", "")).strip()
        if work_exp:
            work_exp_lower = work_exp.lower()
            if work_exp_lower not in original_lower:
                _add_detail("work_experience", work_exp, "工作年限未出现在用户原始输入中")
        edu_level = str(meta.get("education_level", "")).strip()
        if edu_level:
            edu_lower = edu_level.lower()
            if edu_lower not in original_lower:
                _add_detail("education_level", edu_level, "学历未出现在用户原始输入中")

    original_numbers = set(re.findall(r"\d+(?:\.\d+)?", original_text))
    metric_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|万元|万|人|个|次|条|倍|客户|学生|病例|日活|月活|转化|CTR|GMV|QPS|TPS)", re.IGNORECASE)
    for text in _collect_resume_text_values(resume_data):
        if "[需补充]" in text:
            continue
        for number in metric_pattern.findall(text):
            if number not in original_numbers:
                # Double-check with word boundary: avoid substring false positives (e.g. "15" matching "155")
                if not re.search(rf"\b{re.escape(number)}\b", original_text):
                    _add_detail("metric", number, "该数字结果未出现在用户原始输入中，疑似编造量化结果")

    return FabricationReport(
        fabrication_found=len(fab_details) > 0,
        details=fab_details,
    )


def check_summary_jd_alignment(
    summary: str,
    jd_text: Optional[str] = None,
) -> list[FieldConflict]:
    """Check that the personal summary is aligned with the JD (not conflicting).

    Returns FieldConflict if the summary contains capabilities that conflict
    with the JD focus, or is too generic to be meaningful.
    """
    conflicts: list[FieldConflict] = []

    if not summary or not summary.strip():
        return conflicts

    summary = summary.strip()

    # 1. Check if summary is too generic (no domain signal)
    if len(summary) < 10:
        conflicts.append(FieldConflict(
            field="summary",
            description="个人总结过短，缺乏岗位相关信息",
        ))
        return conflicts

    # 2. If JD is provided, check for alignment
    if not jd_text or not jd_text.strip():
        return conflicts

    # Dynamic character/token overlap works for doctors, teachers, operations,
    # R&D and long-tail professions without maintaining an industry dictionary.
    def signals(value: str) -> set[str]:
        normalized = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "", value).casefold()
        chinese_bigrams = {
            normalized[index:index + 2]
            for index in range(max(0, len(normalized) - 1))
        }
        latin = {
            token.casefold()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9+.#/_-]+", value)
        }
        return chinese_bigrams | latin

    jd_signals = signals(jd_text)
    summary_signals = signals(summary)
    if jd_signals and summary_signals and not (jd_signals & summary_signals):
        conflicts.append(FieldConflict(
            field="summary",
            description="个人总结与目标岗位JD缺乏明确的内容交集，建议核对求职方向并补充有事实依据的相关能力",
        ))

    return conflicts
