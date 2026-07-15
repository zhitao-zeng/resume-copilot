import json
import re
from typing import Any, Optional

from prompts import OCR_CLEANUP_SYSTEM_PROMPT, STRUCTURED_RESUME_SYSTEM_PROMPT
from schemas import StructuredResumeLLMOutput
from resume_common import _is_duplicate_project_record, _normalize_compare_text, _normalize_project_name
from server_runtime import (
    ENABLE_PARSE_DEBUG_LOG,
    ENABLE_RESUME_SHRINK_GUARD,
    PARSE_DEBUG_MAX_BULLETS,
    PARSE_DEBUG_PREVIEW_CHARS,
    PERSONAL_SECTION_STOP_HEADERS,
    PROJECT_DATE_RANGE_PATTERN,
    PROJECT_SECTION_HEADERS,
    PROJECT_SECTION_STOP_HEADERS,
    SHRINK_GUARD_MIN_SOURCE_CHARS,
    call_llm_text,
    call_llm_typed,
    llm_enabled,
    logger,
    sanitize_user_text,
)
from resume_io import _looks_like_section_header
def split_bullets(resume_text: str) -> list[dict[str, Any]]:
    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    bullets: list[dict[str, Any]] = []

    current_project = "项目经历"
    counters: dict[str, int] = {}

    bullet_pattern = re.compile(
        r"^(?:[-*•·▪◦‣∙●○■□▶►▸▹◆◇\uF000-\uF8FF]|"
        r"\d{1,2}[\.)]|[（(]?\d{1,2}[）)]|[一二三四五六七八九十]+[、.．])\s*(.+)$"
    )
    date_lead_pattern = re.compile(
        r"^((?:19|20)\d{2}[./年]\d{1,2}(?:月)?\s*(?:[-–—~至到]+\s*(?:(?:19|20)\d{2}[./年]\d{1,2}(?:月)?|至今|Present|present)))\s*[，,:：-]?\s*(.*)$",
        re.IGNORECASE,
    )
    heading_hint = re.compile(r"(项目|project|experience|实习|工作|系统|平台|优化|开发|科研)", re.IGNORECASE)
    pure_marker_pattern = re.compile(r"^[-*•·▪◦‣∙●○■□▶►▸▹◆◇\uF000-\uF8FF]+$")

    for ln in lines:
        if pure_marker_pattern.fullmatch(ln):
            continue

        match = bullet_pattern.match(ln)
        if match:
            text = match.group(1).strip()
            if len(text) < 3:
                continue
            idx = counters.get(current_project, 0) + 1
            counters[current_project] = idx
            bullets.append({"project": current_project, "bullet_index": idx, "text": text})
            continue

        date_lead = date_lead_pattern.match(ln)
        if date_lead:
            candidate = f"{date_lead.group(1)} {date_lead.group(2)}".strip()
            if len(candidate) >= 8:
                idx = counters.get(current_project, 0) + 1
                counters[current_project] = idx
                bullets.append({"project": current_project, "bullet_index": idx, "text": candidate})
                continue

        if _looks_like_section_header(ln, PROJECT_SECTION_HEADERS, max_len=36):
            current_project = ln.rstrip("：:").strip()
            counters.setdefault(current_project, 0)
            continue

        if (len(ln) <= 40 and heading_hint.search(ln)) or ln.endswith("项目"):
            current_project = ln.rstrip("：:")
            counters.setdefault(current_project, 0)

    if bullets:
        return bullets

    chunks = [c.strip() for c in re.split(r"[。；;\n]", resume_text) if len(c.strip()) >= 10]
    for i, chunk in enumerate(chunks[:12], 1):
        bullets.append({"project": "项目经历", "bullet_index": i, "text": chunk})
    return bullets


def _compact_preview(text: Any, limit: Optional[int] = None) -> str:
    max_len = limit or PARSE_DEBUG_PREVIEW_CHARS
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _log_parse_text_debug(stage: str, resume_text: str, extra: Optional[dict[str, Any]] = None) -> None:
    if not ENABLE_PARSE_DEBUG_LOG:
        return
    lines = [ln.strip() for ln in str(resume_text or "").splitlines() if ln.strip()]
    bullets = split_bullets(resume_text or "")
    bullet_samples = [
        _compact_preview(item.get("text", ""), limit=120)
        for item in bullets[:PARSE_DEBUG_MAX_BULLETS]
        if isinstance(item, dict)
    ]
    logger.info(
        "ParseDebug | stage=%s chars=%s lines=%s bullet_count=%s preview=%s bullet_samples=%s extra=%s",
        stage,
        len(resume_text or ""),
        len(lines),
        len(bullets),
        _compact_preview(resume_text),
        json.dumps(bullet_samples, ensure_ascii=False),
        json.dumps(extra or {}, ensure_ascii=False),
    )


def _log_resume_data_debug(stage: str, resume_data: dict[str, Any], extra: Optional[dict[str, Any]] = None) -> None:
    if not ENABLE_PARSE_DEBUG_LOG:
        return

    meta = resume_data.get("meta", {}) if isinstance(resume_data, dict) else {}
    name = str(meta.get("name", "")).strip() if isinstance(meta, dict) else ""

    experience_count = 0
    nested_project_count = 0
    bullet_count = 0
    sample_projects: list[str] = []

    experiences = resume_data.get("experience", []) if isinstance(resume_data, dict) else []
    if isinstance(experiences, list):
        experience_count = sum(1 for exp in experiences if isinstance(exp, dict))
        for exp in experiences:
            if not isinstance(exp, dict):
                continue
            for key in ("bullets", "responsibilities", "achievements"):
                items = exp.get(key, [])
                if isinstance(items, list):
                    bullet_count += sum(1 for item in items if str(item).strip())

            projects = exp.get("projects", [])
            if isinstance(projects, list):
                nested_project_count += sum(1 for proj in projects if isinstance(proj, dict))
                for proj in projects:
                    if not isinstance(proj, dict):
                        continue
                    proj_name = str(proj.get("name", "")).strip()
                    if proj_name and len(sample_projects) < PARSE_DEBUG_MAX_BULLETS:
                        sample_projects.append(_compact_preview(proj_name, limit=80))
                    bullets = proj.get("bullets", [])
                    if isinstance(bullets, list):
                        bullet_count += sum(1 for item in bullets if str(item).strip())

    top_projects = resume_data.get("projects", []) if isinstance(resume_data, dict) else []
    top_project_count = 0
    if isinstance(top_projects, list):
        top_project_count = sum(1 for proj in top_projects if isinstance(proj, dict))
        for proj in top_projects:
            if not isinstance(proj, dict):
                continue
            proj_name = str(proj.get("name", "")).strip()
            if proj_name and len(sample_projects) < PARSE_DEBUG_MAX_BULLETS:
                sample_projects.append(_compact_preview(proj_name, limit=80))
            bullets = proj.get("bullets", [])
            if isinstance(bullets, list):
                bullet_count += sum(1 for item in bullets if str(item).strip())

    logger.info(
        "ParseDebug | stage=%s name=%s experience_count=%s nested_project_count=%s top_project_count=%s bullet_count=%s sample_projects=%s extra=%s",
        stage,
        _compact_preview(name, limit=60),
        experience_count,
        nested_project_count,
        top_project_count,
        bullet_count,
        json.dumps(sample_projects, ensure_ascii=False),
        json.dumps(extra or {}, ensure_ascii=False),
    )


def _count_projects(resume_data: dict[str, Any]) -> int:
    total = 0
    top_projects = resume_data.get("projects", [])
    if isinstance(top_projects, list):
        total += sum(1 for item in top_projects if isinstance(item, dict))
    experiences = resume_data.get("experience", [])
    if isinstance(experiences, list):
        for exp in experiences:
            if not isinstance(exp, dict):
                continue
            projects = exp.get("projects", [])
            if isinstance(projects, list):
                total += sum(1 for item in projects if isinstance(item, dict))
    return total


def _count_resume_bullets(resume_data: dict[str, Any]) -> int:
    total = 0

    experiences = resume_data.get("experience", []) if isinstance(resume_data, dict) else []
    if isinstance(experiences, list):
        for exp in experiences:
            if not isinstance(exp, dict):
                continue
            for key in ("bullets", "responsibilities", "achievements"):
                items = exp.get(key, [])
                if isinstance(items, list):
                    total += sum(1 for item in items if str(item).strip())
            projects = exp.get("projects", [])
            if isinstance(projects, list):
                for proj in projects:
                    if not isinstance(proj, dict):
                        continue
                    bullets = proj.get("bullets", [])
                    if isinstance(bullets, list):
                        total += sum(1 for item in bullets if str(item).strip())

    top_projects = resume_data.get("projects", []) if isinstance(resume_data, dict) else []
    if isinstance(top_projects, list):
        for proj in top_projects:
            if not isinstance(proj, dict):
                continue
            bullets = proj.get("bullets", [])
            if isinstance(bullets, list):
                total += sum(1 for item in bullets if str(item).strip())

    return total


def _count_publications(resume_data: dict[str, Any]) -> int:
    publications = resume_data.get("publications", []) if isinstance(resume_data, dict) else []
    if not isinstance(publications, list):
        return 0
    count = 0
    for item in publications:
        if isinstance(item, dict):
            if any(str(item.get(k, "")).strip() for k in ("title", "venue", "authors", "year")):
                count += 1
        elif str(item).strip():
            count += 1
    return count


def _count_education(resume_data: dict[str, Any]) -> int:
    education = resume_data.get("education", []) if isinstance(resume_data, dict) else []
    if not isinstance(education, list):
        return 0
    count = 0
    for edu in education:
        if not isinstance(edu, dict):
            continue
        if any(str(edu.get(k, "")).strip() for k in ("school", "degree", "major", "period")):
            count += 1
    return count


def _collect_text_entries(resume_data: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    if not isinstance(resume_data, dict):
        return values
    for key in keys:
        items = resume_data.get(key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            text = str(item).strip()
            if text:
                values.append(text)
    return values


def _resume_core_stats(resume_data: dict[str, Any]) -> dict[str, int]:
    text_len = len(resume_data_to_text(resume_data))
    return {
        "text_len": text_len,
        "projects": _count_projects(resume_data),
        "bullets": _count_resume_bullets(resume_data),
        "publications": _count_publications(resume_data),
        "education": _count_education(resume_data),
    }


def _should_guard_resume_shrink(source_resume: dict[str, Any], candidate_resume: dict[str, Any]) -> tuple[bool, str]:
    if not ENABLE_RESUME_SHRINK_GUARD:
        return False, ""

    source = _resume_core_stats(source_resume)
    cand = _resume_core_stats(candidate_resume)

    if source["text_len"] < SHRINK_GUARD_MIN_SOURCE_CHARS:
        return False, ""

    if cand["text_len"] <= max(120, int(source["text_len"] * 0.35)):
        return True, f"text_len {cand['text_len']} << source {source['text_len']}"
    if source["projects"] >= 3 and cand["projects"] <= max(0, source["projects"] // 3):
        return True, f"projects {cand['projects']} << source {source['projects']}"
    if source["bullets"] >= 8 and cand["bullets"] <= max(1, source["bullets"] // 4):
        return True, f"bullets {cand['bullets']} << source {source['bullets']}"
    if source["publications"] >= 6 and cand["publications"] <= max(1, source["publications"] // 3):
        return True, f"publications {cand['publications']} << source {source['publications']}"
    if source["education"] >= 2 and cand["education"] == 0:
        return True, "education became empty"

    return False, ""


def _sanitize_project_name(name: str) -> str:
    cleaned = re.sub(r"[\uF000-\uF8FF•●◦▪◆▶►▸▹]+", " ", str(name or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—|:：,，;；")
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", cleaned):
        return "科研项目"
    return cleaned[:120]


def _extract_project_section_lines(resume_text: str) -> list[str]:
    lines = [ln.strip() for ln in str(resume_text or "").splitlines() if ln.strip()]
    if not lines:
        return []

    start_idx = -1
    for idx, ln in enumerate(lines):
        if _looks_like_section_header(ln, PROJECT_SECTION_HEADERS, max_len=42):
            start_idx = idx
            break
    if start_idx < 0:
        return []

    section: list[str] = []
    for ln in lines[start_idx + 1 :]:
        if _looks_like_section_header(ln, PROJECT_SECTION_STOP_HEADERS, max_len=42):
            break
        section.append(ln)
    return section


def _split_project_block_bullets(block_text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(block_text or "")).strip()
    cleaned = cleaned.strip("，,。；;：:- ")
    if not cleaned:
        return []

    candidates = [part.strip(" ，,。；;") for part in re.split(r"[。；;]\s*", cleaned) if part.strip(" ，,。；;")]
    if len(candidates) <= 1:
        candidates = [
            part.strip(" ，,。；;")
            for part in re.split(r"[，,]\s*(?=(?:负责|参与|承担|完成|主导|依托|并|主要|受|协助))", cleaned)
            if part.strip(" ，,。；;")
        ]

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        normalized = _normalize_compare_text(item)
        if len(normalized) < 8 or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item[:220])
    if deduped:
        return deduped[:8]
    return [cleaned[:220]]


def _infer_project_name_from_block(block_text: str, bullets: list[str]) -> str:
    text = str(block_text or "")
    quoted = re.search(r"[《“\"]([^》”\"]{4,80})[》”\"]", text)
    if quoted:
        return _sanitize_project_name(quoted.group(1))

    patterns = [
        r"(国家自然科学基金[^，。；;]{4,80})",
        r"(环保部公益性行业科研项目[^，。；;]{4,80})",
        r"(?:参与|承担)([^，。；;]{4,80}(?:项目|课题))",
        r"受([^，。；;]{2,40})委托",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        name = match.group(1).strip()
        if "委托" in pattern:
            name = f"{name}委托项目"
        return _sanitize_project_name(name)

    if bullets:
        head = bullets[0]
        if len(head) <= 32 and re.search(r"(项目|课题|研究)", head):
            return _sanitize_project_name(head)

    generic = re.search(r"([A-Za-z0-9\u4e00-\u9fff()（）《》\-]{4,80}(?:项目|课题))", text)
    if generic:
        return _sanitize_project_name(generic.group(1))
    return "科研项目"


def _extract_projects_from_date_blocks(section_lines: list[str]) -> list[dict[str, Any]]:
    if not section_lines:
        return []

    section_text = "\n".join(section_lines)
    section_text = re.sub(r"\n?[-*•·▪◦‣∙●○■□▶►▸▹◆◇\uF000-\uF8FF]+\s*", "\n", section_text)
    matches = list(PROJECT_DATE_RANGE_PATTERN.finditer(section_text))
    if not matches:
        return []

    projects: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        period = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(section_text)
        block = section_text[start:end]
        block = re.sub(r"^[\s,，:：;；\-–—]+", "", block)
        block = re.sub(r"\s+", " ", block).strip()
        if len(block) < 6:
            continue

        bullets = _split_project_block_bullets(block)
        if not bullets:
            continue

        projects.append(
            {
                "name": _infer_project_name_from_block(block, bullets),
                "period": period,
                "description": "",
                "bullets": bullets,
                "tech_stack": [],
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for proj in projects:
        key = (
            _normalize_project_name(proj.get("name", "")),
            _normalize_compare_text(proj.get("period", "")),
            _normalize_compare_text(" ".join(proj.get("bullets", [])[:2])),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(proj)
    return deduped[:10]


def _recover_projects_from_split_bullets(resume_text: str) -> list[dict[str, Any]]:
    section_projects = _extract_projects_from_date_blocks(_extract_project_section_lines(resume_text))
    if section_projects:
        return section_projects

    bullets = split_bullets(resume_text)
    if not bullets:
        return []

    grouped: dict[str, list[str]] = {}
    for item in bullets:
        if not isinstance(item, dict):
            continue
        project = str(item.get("project") or "").strip() or "项目经历"
        text = str(item.get("text") or "").strip()
        if len(text) < 10:
            continue
        if re.fullmatch(r"\d{1,2}[./-]\d{2,4}", text):
            continue
        # Skip publication-like lines to avoid mixing into project bullets.
        if re.search(r"(doi|quaternary|boreas|ecological|review|journal|论文)", text, re.IGNORECASE):
            continue
        if not re.search(r"(项目|科研|研究|负责|参与|承担|完成|设计|分析|采样|实验|报告)", f"{project} {text}"):
            continue
        grouped.setdefault(project, []).append(text)

    recovered: list[dict[str, Any]] = []
    for project_name, items in grouped.items():
        deduped: list[str] = []
        seen: set[str] = set()
        for text in items:
            key = _normalize_compare_text(text)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(text)
        if not deduped:
            continue
        recovered.append(
            {
                "name": _sanitize_project_name(_compact_preview(project_name or "项目经历", limit=80)),
                "period": "",
                "description": "",
                "bullets": deduped[:8],
                "tech_stack": [],
            }
        )

    return recovered[:6]


def _recover_education_from_text(resume_text: str) -> list[dict[str, Any]]:
    lines = [ln.strip() for ln in str(resume_text or "").splitlines() if ln.strip()]
    if not lines:
        return []

    school_degree_re = re.compile(
        r"(?P<school>[A-Za-z\u4e00-\u9fff]{2,40}(?:大学|学院))\s*(?P<degree>(?:理学|工学|文学|教育学|管理学)?(?:学士|硕士|博士))?"
    )
    period_re = re.compile(r"(?:19|20)\d{2}[./-]\d{1,2}\s*[-–—至到~]\s*(?:19|20)\d{2}[./-]\d{1,2}|(?:19|20)\d{2}[./-]\d{1,2}")

    pairs: list[tuple[str, str]] = []
    for ln in lines:
        m = school_degree_re.search(ln)
        if not m:
            continue
        school = m.group("school").strip()
        degree = (m.group("degree") or "").strip()
        if school:
            pairs.append((school, degree))

    if not pairs:
        return []

    periods = period_re.findall(resume_text)
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for idx, (school, degree) in enumerate(pairs):
        key = (school, degree)
        if key in seen:
            continue
        seen.add(key)
        period = periods[idx] if idx < len(periods) else ""
        entries.append(
            {
                "school": school,
                "degree": degree,
                "major": "",
                "period": period,
                "highlights": [],
            }
        )

    return entries[:6]


def _recover_publications_from_text(resume_text: str) -> list[dict[str, Any]]:
    lines = [ln.strip() for ln in str(resume_text or "").splitlines() if ln.strip()]
    if not lines:
        return []

    pubs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ln in lines:
        if len(ln) < 24:
            continue
        if not re.search(r"(doi|SCI|EI|CSCD|journal|review|boreas|quaternary|ecological|湖泊科学|古生物)", ln, re.IGNORECASE):
            continue
        key = _normalize_compare_text(ln)
        if key in seen:
            continue
        seen.add(key)
        pubs.append(
            {
                "title": _compact_preview(ln, limit=260),
                "venue": "",
                "year": "",
                "authors": "",
            }
        )
    return pubs[:20]


def _recover_awards_from_text(resume_text: str) -> list[str]:
    lines = [ln.strip() for ln in str(resume_text or "").splitlines() if ln.strip()]
    result: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        if len(ln) < 8:
            continue
        if not re.search(r"(获|荣获|奖学金|优秀|奖励|表彰|称号)", ln):
            continue
        key = _normalize_compare_text(ln)
        if key in seen:
            continue
        seen.add(key)
        result.append(_compact_preview(ln, limit=180))
    return result[:20]


def _recover_personal_skills_from_text(resume_text: str) -> list[str]:
    lines = [ln.strip() for ln in str(resume_text or "").splitlines() if ln.strip()]
    if not lines:
        return []

    collect = False
    result: list[str] = []
    for ln in lines:
        if re.search(r"(自我评价|专业素养培养|个人评价)", ln):
            collect = True
            continue
        if collect and _looks_like_section_header(ln, PERSONAL_SECTION_STOP_HEADERS, max_len=42):
            if result:
                break
            continue
        if collect:
            if len(ln) >= 10:
                result.append(_compact_preview(ln, limit=180))
            if len(result) >= 8:
                break
    return result


def _extract_standalone_projects_from_text(resume_text: str) -> list[dict[str, Any]]:
    section = _extract_project_section_lines(resume_text)
    if not section:
        return []

    # Prefer deterministic date-block extraction for academic resumes.
    projects_from_dates = _extract_projects_from_date_blocks(section)
    if projects_from_dates:
        return projects_from_dates

    bullet_re = re.compile(
        r"^(?:[-*•·▪◦‣∙●○■□▶►▸▹◆◇\uF000-\uF8FF]|"
        r"\d{1,2}[\.)]|[（(]?\d{1,2}[）)]|[一二三四五六七八九十]+[、.．])\s*(.+)$"
    )
    pure_marker_re = re.compile(r"^[-*•·▪◦‣∙●○■□▶►▸▹◆◇\uF000-\uF8FF]+$")
    title_hint_re = re.compile(r"(项目|系统|平台|助手|框架|模型|算法|工程|开发|研究|预测|优化)")

    projects: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None

    def _new_project(name: str) -> dict[str, Any]:
        return {
            "name": _sanitize_project_name(name or "科研项目"),
            "period": "",
            "description": "",
            "bullets": [],
            "tech_stack": [],
        }

    def _push_current() -> None:
        nonlocal current
        if not current:
            return
        current["name"] = _sanitize_project_name(str(current.get("name", "")).strip())
        has_content = bool(current.get("period")) or bool(current.get("bullets")) or bool(current.get("description"))
        if has_content:
            projects.append(current)
        current = None

    def _looks_like_project_title(text: str) -> bool:
        candidate = text.strip()
        if not candidate or len(candidate) > 72:
            return False
        if re.search(r"[。！？；;]$", candidate):
            return False
        if re.search(r"(负责|实现|采用|通过|进行|用于|包括|提升|优化|分析|构建|设计)", candidate):
            return False
        return bool(title_hint_re.search(candidate))

    for raw in section:
        ln = str(raw).strip().strip("|").strip()
        if not ln or pure_marker_re.fullmatch(ln):
            continue

        m_date = PROJECT_DATE_RANGE_PATTERN.search(ln)
        if m_date:
            period = m_date.group(1).strip()
            if current is not None and current.get("period") and (current.get("bullets") or current.get("description")):
                _push_current()
            if current is None:
                current = _new_project("科研项目")
            if not current.get("period"):
                current["period"] = period
            rest = ln.replace(period, "", 1).strip(" -–—至到~:：,，;；")
            if rest:
                current["bullets"].append(rest)
            continue

        m_bullet = bullet_re.match(ln)
        if m_bullet:
            if current is None:
                current = _new_project("科研项目")
            text = m_bullet.group(1).strip()
            if text:
                current["bullets"].append(text)
            continue

        if re.search(r"^(tech|技术栈|技术|Tech)", ln, re.IGNORECASE):
            if current is None:
                current = _new_project("科研项目")
            tech_items = [
                item.strip()
                for item in re.split(r"[，,、/|·\s]+", re.sub(r"^(tech|技术栈|技术)[:：]?", "", ln, flags=re.IGNORECASE))
                if item.strip()
            ]
            if tech_items:
                current["tech_stack"].extend(tech_items)
            continue

        if current is None:
            current = _new_project(ln if _looks_like_project_title(ln) else "科研项目")
            if current["name"] == "科研项目":
                current["bullets"].append(ln)
            continue

        if not current.get("period") and not current.get("bullets") and _looks_like_project_title(ln):
            current["name"] = _sanitize_project_name(ln)
            continue

        if current.get("bullets"):
            if _looks_like_project_title(ln):
                _push_current()
                current = _new_project(ln)
                continue
            last_bullet = str(current["bullets"][-1]).strip() if current["bullets"] else ""
            if last_bullet:
                current["bullets"][-1] = f"{last_bullet} {ln}".strip()
                continue

        current["bullets"].append(ln)

    _push_current()

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for proj in projects:
        name = _sanitize_project_name(str(proj.get("name", "")).strip())
        period = str(proj.get("period", "")).strip()
        bullets = proj.get("bullets", [])
        if not isinstance(bullets, list):
            bullets = []
        proj["name"] = name
        proj["bullets"] = [str(item).strip() for item in bullets if str(item).strip()][:8]
        if not proj["bullets"] and not period:
            continue
        key = (name, period, _normalize_compare_text(" ".join(proj["bullets"][:2])))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(proj)
    return deduped[:10]


def _backfill_experience_bullets_from_text(parsed: dict[str, Any], resume_text: str) -> None:
    experiences = parsed.get("experience", [])
    if not isinstance(experiences, list) or not experiences:
        return

    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    if not lines:
        return

    work_start = -1
    for idx, ln in enumerate(lines):
        if re.search(r"(实习经历|工作经历|工作/实习经历|work experience|internship)", ln, re.IGNORECASE):
            work_start = idx
            break
    if work_start < 0:
        return

    stop_headers = (
        "项目经历",
        "项目经验",
        "教育经历",
        "教育背景",
        "专业技能",
        "个人技能",
        "荣誉与奖项",
        "获奖情况",
        "论文发表情况",
        "论文成果",
        "学术成果",
        "证书",
    )
    section: list[str] = []
    for ln in lines[work_start + 1 :]:
        if _looks_like_section_header(ln, stop_headers, max_len=42):
            break
        section.append(ln)
    if not section:
        return

    bullet_re = re.compile(
        r"^(?:[-*•·▪◦‣∙●○■□▶►▸▹◆◇\uF000-\uF8FF]|"
        r"\d{1,2}[\.)]|[（(]?\d{1,2}[）)]|[一二三四五六七八九十]+[、.．])\s*(.+)$"
    )
    date_re = re.compile(r"(19|20)\d{2}[./-]\d{1,2}")
    all_company_names = [
        str(exp.get("company", "")).strip()
        for exp in experiences
        if isinstance(exp, dict) and str(exp.get("company", "")).strip()
    ]

    for exp in experiences:
        if not isinstance(exp, dict):
            continue
        existing = []
        for key in ("bullets", "responsibilities", "achievements", "highlights", "details"):
            val = exp.get(key, [])
            if isinstance(val, list):
                existing.extend(str(x).strip() for x in val if str(x).strip())
        if existing:
            continue

        company = str(exp.get("company", "")).strip()
        role = str(exp.get("role", "")).strip()
        if not company and not role:
            continue

        anchor_idx = -1
        for idx, ln in enumerate(section):
            if company and company in ln:
                anchor_idx = idx
                break
            if role and role in ln:
                anchor_idx = idx
                break
        if anchor_idx < 0:
            continue

        collected: list[str] = []
        for ln in section[anchor_idx + 1 :]:
            if company and company in ln:
                break
            if any(other and other != company and other in ln for other in all_company_names):
                break
            if len(ln) <= 2:
                continue
            m = bullet_re.match(ln)
            if m:
                text = m.group(1).strip()
                if text:
                    collected.append(text)
                continue
            if date_re.search(ln):
                continue
            if len(ln) >= 10 and not re.search(r"(指导老师|导师|地址|base|联系方式|邮箱|电话)", ln, re.IGNORECASE):
                collected.append(ln)
            if len(collected) >= 8:
                break

        if collected:
            deduped: list[str] = []
            seen: set[str] = set()
            for item in collected:
                key = _normalize_compare_text(item)
                if not key or key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            exp["bullets"] = deduped[:8]



def _coerce_structured_resume_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}

    if not isinstance(parsed.get("meta"), dict):
        parsed["meta"] = {"name": "候选人"}
    if not isinstance(parsed.get("experience"), list):
        parsed["experience"] = []
    if not isinstance(parsed.get("projects"), list):
        parsed["projects"] = []
    if not isinstance(parsed.get("education"), list):
        parsed["education"] = []
    if not isinstance(parsed.get("skills"), dict):
        parsed["skills"] = {"languages": [], "frameworks": [], "tools": [], "domains": []}
    if not isinstance(parsed.get("summary"), str):
        parsed["summary"] = ""
    if not isinstance(parsed.get("publications"), list):
        parsed["publications"] = []
    if not isinstance(parsed.get("honors"), list):
        parsed["honors"] = []
    if not isinstance(parsed.get("awards"), list):
        parsed["awards"] = []
    if not isinstance(parsed.get("certifications"), list):
        parsed["certifications"] = []
    if not isinstance(parsed.get("personal_skills"), list):
        parsed["personal_skills"] = []
    if not isinstance(parsed.get("additional_sections"), dict):
        parsed["additional_sections"] = {}

    for exp in parsed.get("experience", []):
        if not isinstance(exp, dict):
            continue
        for key in ("bullets", "responsibilities", "achievements"):
            if not isinstance(exp.get(key), list):
                exp[key] = []
        if not isinstance(exp.get("projects"), list):
            exp["projects"] = []
        for proj in exp.get("projects", []):
            if not isinstance(proj, dict):
                continue
            if not isinstance(proj.get("bullets"), list):
                proj["bullets"] = []
            if not isinstance(proj.get("tech_stack"), list):
                proj["tech_stack"] = []

    for proj in parsed.get("projects", []):
        if not isinstance(proj, dict):
            continue
        if not isinstance(proj.get("bullets"), list):
            proj["bullets"] = []
        if not isinstance(proj.get("tech_stack"), list):
            proj["tech_stack"] = []
        if not isinstance(proj.get("name"), str):
            proj["name"] = str(proj.get("name", "项目"))
        if not isinstance(proj.get("period"), str):
            proj["period"] = str(proj.get("period", ""))
        if not isinstance(proj.get("description"), str):
            proj["description"] = str(proj.get("description", ""))

    return parsed


def cleanup_ocr_text(raw_text: str) -> str:
    """Use LLM to clean up noisy OCR text before structured parsing.

    Handles: broken lines, garbled symbols, formatting issues.
    Falls back to raw_text if LLM is unavailable or fails.
    """
    if not llm_enabled() or not raw_text or not raw_text.strip():
        return raw_text
    # Skip cleanup for short or clean text
    if len(raw_text.strip()) < 100:
        return raw_text
    try:
        cleaned = call_llm_text(
            system_prompt=OCR_CLEANUP_SYSTEM_PROMPT,
            user_prompt=f"请修复以下 OCR 文本：\n\n{raw_text[:6000]}",
            temperature=0.0,
            max_tokens=4096,
        )
        if isinstance(cleaned, str) and len(cleaned.strip()) >= len(raw_text.strip()) * 0.5:
            return cleaned.strip()
        logger.warning("OCR cleanup returned unusable result, using raw text")
        return raw_text
    except Exception as exc:
        logger.warning("OCR cleanup LLM failed, using raw text: %s", exc)
        return raw_text


def structured_resume_from_text(resume_text: str) -> dict[str, Any]:
    if not llm_enabled():
        return {}

    prompt = (
        "请将以下简历文本精确解析为结构化 JSON。\n"
        "注意：准确识别候选人姓名（不要把机构名/页眉当人名）。\n\n"
        "【简历文本】\n"
        f"{sanitize_user_text(resume_text)}"
    )
    try:
        parsed = call_llm_typed(
            StructuredResumeLLMOutput,
            STRUCTURED_RESUME_SYSTEM_PROMPT,
            prompt,
            temperature=0.0,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.warning("LLM failed to parse resume: %s", exc)
        return {}

    if not isinstance(parsed, dict) or not parsed:
        logger.warning("LLM returned empty/invalid resume structure")
        return {}

    parsed = _coerce_structured_resume_payload(parsed)
    source_text_len = len(resume_text or "")
    parsed_text_len = len(resume_data_to_text(parsed))
    should_retry_sparse_parse = source_text_len >= SHRINK_GUARD_MIN_SOURCE_CHARS and parsed_text_len <= max(
        120, int(source_text_len * 0.12)
    )
    if should_retry_sparse_parse:
        retry_prompt = (
            "请重新做一次结构化解析。上一轮结果过于稀疏，疑似漏提取。\n"
            "务必完整保留原文中的项目、教育、论文、获奖、技能，不要只返回姓名。\n"
            "如果存在“科研项目参与情况/论文发表情况/学术会议参与情况”等板块，也必须结构化输出。\n\n"
            "【简历文本】\n"
            f"{sanitize_user_text(resume_text)}"
        )
        try:
            retry_parsed = call_llm_typed(
                StructuredResumeLLMOutput,
                STRUCTURED_RESUME_SYSTEM_PROMPT,
                retry_prompt,
                temperature=0.0,
                max_tokens=4096,
            )
            retry_parsed = _coerce_structured_resume_payload(retry_parsed) if isinstance(retry_parsed, dict) else {}
            retry_text_len = len(resume_data_to_text(retry_parsed)) if retry_parsed else 0
            if retry_parsed and retry_text_len > parsed_text_len:
                logger.info(
                    "Structured parse retry improved coverage | before_text_len=%s after_text_len=%s",
                    parsed_text_len,
                    retry_text_len,
                )
                parsed = retry_parsed
        except Exception as exc:
            logger.warning("Structured parse retry failed: %s", exc)

    _backfill_experience_bullets_from_text(parsed, resume_text)

    fallback_projects = _extract_standalone_projects_from_text(resume_text)
    if fallback_projects:
        existing_top = parsed.get("projects", [])
        if not isinstance(existing_top, list):
            existing_top = []
        existing_records: list[dict[str, Any]] = [proj for proj in existing_top if isinstance(proj, dict)]
        experience = parsed.get("experience", [])
        if isinstance(experience, list):
            for exp in experience:
                if not isinstance(exp, dict):
                    continue
                exp_projects = exp.get("projects", [])
                if not isinstance(exp_projects, list):
                    continue
                existing_records.extend(proj for proj in exp_projects if isinstance(proj, dict))

        added = 0
        for proj in fallback_projects:
            if not isinstance(proj, dict):
                continue
            if any(_is_duplicate_project_record(proj, existing) for existing in existing_records):
                continue
            existing_top.append(proj)
            existing_records.append(proj)
            added += 1
        parsed["projects"] = existing_top
        if added > 0:
            logger.info("Recovered %d standalone projects from raw resume text", added)

    # When LLM parsing is too sparse, recover core sections from raw text.
    existing_projects = _count_projects(parsed)
    existing_project_bullets = _count_resume_bullets(parsed)
    if existing_projects == 0 or (
        existing_projects <= 1
        and existing_project_bullets <= 2
        and len(resume_text or "") >= SHRINK_GUARD_MIN_SOURCE_CHARS
    ):
        recovered_projects = _recover_projects_from_split_bullets(resume_text)
        if recovered_projects and len(recovered_projects) >= existing_projects:
            parsed["projects"] = recovered_projects
            logger.info("Recovered %d projects from split bullets", len(recovered_projects))

    existing_education = _count_education(parsed)
    if existing_education == 0 or (
        existing_education <= 1 and len(resume_text or "") >= SHRINK_GUARD_MIN_SOURCE_CHARS
    ):
        recovered_education = _recover_education_from_text(resume_text)
        if recovered_education and len(recovered_education) >= existing_education:
            parsed["education"] = recovered_education
            logger.info("Recovered %d education entries from raw text", len(recovered_education))

    existing_publications = _count_publications(parsed)
    if existing_publications < 3 and len(resume_text or "") >= SHRINK_GUARD_MIN_SOURCE_CHARS:
        recovered_publications = _recover_publications_from_text(resume_text)
        if recovered_publications and len(recovered_publications) > existing_publications:
            parsed["publications"] = recovered_publications
            logger.info("Recovered %d publications from raw text", len(recovered_publications))

    honors = _collect_text_entries(parsed, ("honors", "awards", "certifications"))
    if not honors:
        recovered_awards = _recover_awards_from_text(resume_text)
        if recovered_awards:
            parsed["honors"] = recovered_awards
            logger.info("Recovered %d honor/award entries from raw text", len(recovered_awards))

    personal_skills = _collect_text_entries(parsed, ("personal_skills",))
    if not personal_skills:
        recovered_personal = _recover_personal_skills_from_text(resume_text)
        if recovered_personal:
            parsed["personal_skills"] = recovered_personal
            logger.info("Recovered %d personal skill entries from raw text", len(recovered_personal))

    # Last guard: if parsed text is unexpectedly tiny, keep conservative fallback text in summary.
    parsed_text_len = len(resume_data_to_text(parsed))
    source_text_len = len(resume_text or "")
    if source_text_len >= SHRINK_GUARD_MIN_SOURCE_CHARS and parsed_text_len <= max(80, int(source_text_len * 0.15)):
        if not str(parsed.get("summary") or "").strip():
            parsed["summary"] = _compact_preview(resume_text, limit=1000)
        logger.warning(
            "Structured parse remains sparse after recovery | parsed_text_len=%s source_text_len=%s",
            parsed_text_len,
            source_text_len,
        )

    return parsed


def resume_data_to_text(resume_data: dict[str, Any]) -> str:
    lines: list[str] = []
    meta = resume_data.get("meta", {})
    if isinstance(meta, dict):
        name = str(meta.get("name", "")).strip()
        if name:
            lines.append(name)
        for field in ("target_role", "job_intention", "expected_city"):
            value = str(meta.get(field, "")).strip()
            if value:
                lines.append(f"求职意向：{value}")

    summary = str(resume_data.get("summary", "")).strip()
    if summary:
        lines.append(f"摘要：{summary}")

    for exp in resume_data.get("experience", []) if isinstance(resume_data.get("experience"), list) else []:
        company = exp.get("company", "")
        role = exp.get("role", "")
        period = exp.get("period", "")
        header = " ".join([x for x in [company, role, period] if x])
        if header:
            lines.append(header)
        team = str(exp.get("team", "")).strip()
        if team:
            lines.append(f"团队：{team}")

        for key in ("bullets", "responsibilities", "achievements"):
            values = exp.get(key, [])
            if isinstance(values, list):
                for item in values:
                    text = str(item).strip()
                    if text:
                        lines.append(f"- {text}")

        for proj in exp.get("projects", []) if isinstance(exp.get("projects"), list) else []:
            name = proj.get("name", "项目")
            lines.append(str(name))
            for key in ("description", "period"):
                value = str(proj.get(key, "")).strip()
                if value:
                    lines.append(f"  {key}：{value}")
            tech_stack = proj.get("tech_stack", [])
            if isinstance(tech_stack, list) and tech_stack:
                lines.append(f"  技术栈：{', '.join(str(t) for t in tech_stack)}")
            for bullet in proj.get("bullets", []) if isinstance(proj.get("bullets"), list) else []:
                lines.append(f"- {bullet}")

    for proj in resume_data.get("projects", []) if isinstance(resume_data.get("projects"), list) else []:
        if not isinstance(proj, dict):
            continue
        name = str(proj.get("name", "项目")).strip() or "项目"
        period = str(proj.get("period", "")).strip()
        if period:
            lines.append(f"{name} {period}")
        else:
            lines.append(name)
        description = str(proj.get("description", "")).strip()
        if description:
            lines.append(f"- {description}")
        tech_stack = proj.get("tech_stack", [])
        if isinstance(tech_stack, list) and tech_stack:
            lines.append(f"技术栈：{', '.join(str(t) for t in tech_stack)}")
        for bullet in proj.get("bullets", []) if isinstance(proj.get("bullets"), list) else []:
            text = str(bullet).strip()
            if text:
                lines.append(f"- {text}")

    for edu in resume_data.get("education", []) if isinstance(resume_data.get("education"), list) else []:
        if not isinstance(edu, dict):
            continue
        parts = [str(edu.get(k, "")).strip() for k in ("school", "degree", "major", "period")]
        line = " ".join(p for p in parts if p)
        if line:
            lines.append(f"教育：{line}")

    skills = resume_data.get("skills", {})
    if isinstance(skills, dict):
        for bucket in ("languages", "frameworks", "tools", "domains"):
            items = skills.get(bucket, [])
            if isinstance(items, list) and items:
                lines.append(f"技能-{bucket}：{', '.join(str(t) for t in items)}")

    publications = resume_data.get("publications", [])
    if isinstance(publications, list):
        for pub in publications:
            text = str(pub).strip()
            if text:
                lines.append(f"论文：{text}")

    for key in ("honors", "awards", "certifications", "personal_skills"):
        items = resume_data.get(key, [])
        if isinstance(items, list) and items:
            label = {"honors": "荣誉", "awards": "奖项", "certifications": "证书", "personal_skills": "个人技能"}.get(key, key)
            for item in items:
                text = str(item).strip()
                if text:
                    lines.append(f"{label}：{text}")

    additional = resume_data.get("additional_sections", {})
    if isinstance(additional, dict):
        for section_name, items in additional.items():
            if isinstance(items, list):
                for item in items:
                    text = str(item).strip()
                    if text:
                        lines.append(f"{section_name}：{text}")

    return "\n".join(lines).strip()

