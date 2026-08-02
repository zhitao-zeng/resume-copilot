"""V2 Pipeline orchestration.

Layers: SourceAdapter → Composer → Verifier → Optimizer → Validator
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata

from v2_schemas import VerifiedResult, CanonicalResume, DraftResume, Meta, Change
from source_adapter import build_source_bundle, candidate_blocks
from resume_composer import compose_resume, compose_from_query
from resume_verifier import verify_resume
from resume_verifier import _ground_fixed_fields, _reclassify_non_work
from resume_optimizer import optimize_resume, _introduces_unsupported_fact
from v2_validator import validate_resume
from evidence_binding import bind_resume_evidence, enforce_resume_evidence, measure_source_coverage
import resume_product_logic as product_logic

logger = logging.getLogger(__name__)


def _is_empty_resume(resume: CanonicalResume) -> bool:
    return not any((
        resume.meta.name,
        resume.meta.phone,
        resume.meta.email,
        resume.education,
        resume.experience,
        resume.research,
        resume.activities,
        resume.projects,
        resume.skills.items,
        resume.summary,
        resume.awards,
        resume.publications,
        resume.patents,
        resume.certifications,
        resume.training,
        resume.teaching,
        resume.additional_sections,
    ))


def _has_candidate_profile(resume: CanonicalResume) -> bool:
    """Whether any candidate fact remains after grounding and evidence gates."""

    return any((
        resume.meta.name,
        resume.meta.phone,
        resume.meta.email,
        resume.meta.work_experience,
        resume.education,
        resume.experience,
        resume.research,
        resume.activities,
        resume.projects,
        resume.skills.items,
        resume.awards,
        resume.publications,
        resume.patents,
        resume.certifications,
        resume.training,
        resume.teaching,
        resume.additional_sections,
    ))


_STRONG_OWNERSHIP = ("主导", "统筹", "牵头", "独立负责", "全权负责", "从0到1", "从零到一")
_MEDIUM_OWNERSHIP = ("负责", "组织", "推动", "管理", "设计", "开发", "构建", "实现", "制定")
_WEAK_OWNERSHIP = ("参与", "协助", "支持", "配合", "接触", "了解", "学习")
_RESULT_CLAIMS = (
    "显著提升", "大幅提升", "提升了", "降低了", "减少了", "增长了", "增强了",
    "确保", "保障", "关键依据", "高质量交付", "打通", "性能达标", "降低成本",
    "提高准确率", "提升准确率", "提升效率", "提升用户体验",
)

_SKILL_CATEGORY_ALIASES = {
    "language": "language",
    "programming_language": "language",
    "programming language": "language",
    "编程语言": "language",
    "framework": "framework",
    "library": "framework",
    "框架": "framework",
    "库": "framework",
    "tool": "tool",
    "software": "tool",
    "platform": "tool",
    "工具": "tool",
    "软件": "tool",
    "平台": "tool",
    "domain": "domain",
    "professional": "domain",
    "专业领域": "domain",
    "业务领域": "domain",
    "method": "methodology",
    "methodology": "methodology",
    "process": "methodology",
    "方法": "methodology",
    "流程": "methodology",
    "certification": "certification",
    "certificate": "certification",
    "license": "certification",
    "证书": "certification",
    "资质": "certification",
    "natural_language": "natural_language",
    "natural language": "natural_language",
    "spoken_language": "natural_language",
    "自然语言": "natural_language",
    "语言能力": "natural_language",
    "other": "other",
    "其他": "other",
}

_NATURAL_LANGUAGE_SKILL = re.compile(
    r"(?:英语|英文|日语|日文|韩语|韩文|法语|德语|西班牙语|俄语|普通话|粤语|"
    r"CET[-\s]?[46]|TEM[-\s]?[48]|IELTS|TOEFL|雅思|托福|JLPT|TOPIK|HSK)",
    re.IGNORECASE,
)
_CERTIFICATION_SKILL = re.compile(
    r"(?:证书|资格证|执业资格|职业资格|认证|执照|license|certificate|certified)",
    re.IGNORECASE,
)
_QUANTIFIED_CHANGE = re.compile(
    r"(?:从\s*\d[^，。；]{0,24}(?:降至|提升至|提高到|增长至|减少到|缩短至|达到)\s*\d|"
    r"(?:降低|提升|提高|增长|减少|缩短|节省)[^，。；]{0,12}\d)",
    re.IGNORECASE,
)
_RESULT_SIGNAL = re.compile(
    r"(?:降至|提升至|提高到|增长至|减少到|缩短至|达到|完成|上线|交付|录用|获奖|复核|验证)",
    re.IGNORECASE,
)


def _action_level(text: str) -> int:
    if any(token in text for token in _STRONG_OWNERSHIP):
        return 3
    if any(token in text for token in _MEDIUM_OWNERSHIP):
        return 2
    if any(token in text for token in _WEAK_OWNERSHIP):
        return 1
    return 0


def _char_bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", str(text or ""))
    return {compact[i:i + 2] for i in range(max(0, len(compact) - 1))}


def _source_sentences(text: str) -> list[str]:
    values = [item.strip(" \t-•") for item in re.split(r"[\n。；]+", text) if item.strip()]
    return [item for item in values if len(item) >= 6]


def _closest_source_sentence(bullet: str, sentences: list[str]) -> tuple[str, float]:
    target = _char_bigrams(bullet)
    if not target:
        return "", 0.0
    best, score = "", 0.0
    for sentence in sentences:
        candidate = _char_bigrams(sentence)
        if not candidate:
            continue
        overlap = len(target & candidate) / max(1, min(len(target), len(candidate)))
        if overlap > score:
            best, score = sentence, overlap
    return best, score


def _ground_bullets(resume: CanonicalResume, evidence_text: str) -> CanonicalResume:
    """Fall back to the nearest source sentence when a bullet upgrades facts."""

    grounded = resume.model_copy(deep=True)
    sentences = _source_sentences(evidence_text)
    for section in (grounded.experience, grounded.research, grounded.activities, grounded.projects):
        for record in section:
            safe_bullets: list[str] = []
            for bullet in record.bullets:
                value = str(bullet or "").strip()
                if not value:
                    continue
                source, similarity = _closest_source_sentence(value, sentences)
                if not source or similarity < 0.22:
                    logger.info("Dropped ungrounded bullet: %s", value[:80])
                    continue
                upgraded = _action_level(value) > _action_level(source)
                unsupported_result = any(term in value and term not in source for term in _RESULT_CLAIMS)
                unsupported_fact = _introduces_unsupported_fact(source, value)
                if upgraded or unsupported_result or unsupported_fact:
                    logger.info("Restored source wording for over-claimed bullet: %s", value[:80])
                    value = source
                if value not in safe_bullets:
                    safe_bullets.append(value)
            record.bullets = safe_bullets
    return grounded


def _normalize_skill_category(name: str, category: str) -> str:
    """Normalize broad semantic buckets without maintaining an industry lexicon."""

    if _CERTIFICATION_SKILL.search(name):
        return "certification"
    if _NATURAL_LANGUAGE_SKILL.search(name):
        return "natural_language"
    normalized = re.sub(r"[-_/]+", "_", str(category or "").strip().lower())
    return _SKILL_CATEGORY_ALIASES.get(normalized, "other")


def _normalize_skill_name(name: str) -> str:
    value = unicodedata.normalize("NFKC", str(name or "")).strip()
    value = re.sub(r"\s+", " ", value)
    # Models vary between "DOE实验设计" and "DOE 实验设计".  This is
    # typographic normalization, not an industry dictionary.
    value = re.sub(r"(?<=[A-Za-z0-9])\s+(?=[\u4e00-\u9fff])", "", value)
    return value


def _bullet_priority(text: str, target_context: str = "") -> tuple[int, int, int, int]:
    """Rank an existing bullet by relevance and evidence density."""

    value = str(text or "")
    target_bigrams = _char_bigrams(target_context)
    overlap = len(_char_bigrams(value) & target_bigrams) if target_bigrams else 0
    quantified_change = int(bool(_QUANTIFIED_CHANGE.search(value)))
    result_signal = int(bool(_RESULT_SIGNAL.search(value)))
    numeric = int(bool(re.search(r"\d", value)))
    percentage = int(bool(re.search(r"\d+(?:\.\d+)?\s*%", value)))
    monetary = int(bool(re.search(r"\d+(?:\.\d+)?\s*(?:万|亿)?元", value)))
    multiple_metrics = int(len(re.findall(r"\d+(?:\.\d+)?", value)) >= 2)
    evidence_density = (
        quantified_change * 4 + result_signal * 2 + numeric
        + percentage * 3 + monetary * 3 + multiple_metrics * 2
    )
    # A verified before/after result should normally lead a project even when
    # a descriptive bullet repeats more JD keywords.  Very strong relevance
    # can still win through the combined score.
    return overlap + evidence_density * 3, evidence_density, overlap, min(len(value), 120)


def _rank_resume_content(resume: CanonicalResume, target_context: str = "") -> CanonicalResume:
    """Put relevant and verifiable bullets first without changing their text."""

    ranked = resume.model_copy(deep=True)
    for section in (ranked.experience, ranked.research, ranked.activities, ranked.projects):
        for record in section:
            record.bullets = sorted(
                record.bullets,
                key=lambda value: _bullet_priority(value, target_context),
                reverse=True,
            )
    return ranked


def _needs_optimizer(resume: CanonicalResume) -> bool:
    """Every factual bullet gets a dedicated evidence-preserving edit pass."""

    bullets = [
        str(bullet).strip()
        for section in (resume.experience, resume.research, resume.activities, resume.projects)
        for record in section
        for bullet in record.bullets
        if str(bullet).strip()
    ]
    return bool(bullets)


def _bullet_rewrite_changes(
    before: CanonicalResume,
    after: CanonicalResume,
) -> list[Change]:
    """Expose only bullet edits that the safety gate actually accepted."""

    changes: list[Change] = []
    for section_name in ("experience", "research", "activities", "projects"):
        before_records = getattr(before, section_name)
        after_records = getattr(after, section_name)
        for record_index, (before_record, after_record) in enumerate(
            zip(before_records, after_records)
        ):
            for bullet_index, (old_text, new_text) in enumerate(
                zip(before_record.bullets, after_record.bullets)
            ):
                if str(old_text).strip() == str(new_text).strip():
                    continue
                changes.append(Change(
                    path=f"{section_name}[{record_index}].bullets[{bullet_index}]",
                    action="replace",
                    reason="Evidence-preserving bullet rewrite",
                ))
    return changes


def _deterministic_verify_draft(
    source,
    draft: DraftResume,
) -> VerifiedResult | None:
    """Accept a clean Composer draft without paying for a second LLM call."""

    candidate_evidence = "\n".join(block.text for block in candidate_blocks(source))
    data = draft.model_dump()
    _ground_fixed_fields(data, candidate_evidence)
    _reclassify_non_work(data, candidate_evidence)
    try:
        resume = CanonicalResume.model_validate(data)
    except Exception:
        return None
    resume = _ground_bullets(resume, candidate_evidence)
    resume, bindings, removed = enforce_resume_evidence(resume, source)
    resume = _compact_canonical(resume)
    if _is_empty_resume(resume) or len(bindings) < 3:
        return None
    coverage, missing_blocks = measure_source_coverage(source, bindings)
    content_block_count = len({binding.block_id for binding in bindings}) + len(missing_blocks)
    if content_block_count >= 5 and coverage < 0.75:
        logger.info(
            "V2 | Deterministic verifier rejected draft: source coverage %.1f%%, missing=%s",
            coverage * 100,
            missing_blocks[:8],
        )
        return None
    if len(removed) > max(2, int(len(bindings) * 0.15)):
        logger.info("V2 | Deterministic verifier rejected draft: %d unbound claims", len(removed))
        return None
    changes = [
        Change(path=path, action="remove", reason="No candidate evidence binding")
        for path in removed
    ]
    logger.info(
        "V2 | Deterministic verifier accepted: %d bindings, %d removals",
        len(bindings), len(removed),
    )
    return VerifiedResult(resume=resume, changes=changes, evidence_bindings=bindings)


def _best_achievement(resume: CanonicalResume) -> str:
    candidates = [
        str(bullet).strip()
        for section in (resume.projects, resume.experience, resume.research, resume.activities)
        for record in section
        for bullet in record.bullets
        if str(bullet).strip()
    ]
    if not candidates:
        return ""
    best = max(candidates, key=lambda value: _bullet_priority(value))
    if not (_QUANTIFIED_CHANGE.search(best) or _RESULT_SIGNAL.search(best)):
        return ""
    return best.strip("。；; ")


def _build_evidence_summary(resume: CanonicalResume) -> str:
    """Build a stable summary only from fields that already passed grounding.

    This deliberately avoids industry-specific templates and subjective claims.
    Unknown professions therefore receive the same factual treatment as known
    ones, without requiring an ever-growing keyword dictionary.
    """

    if not any((
        resume.education,
        resume.experience,
        resume.research,
        resume.activities,
        resume.projects,
        resume.awards,
        resume.publications,
        resume.patents,
        resume.certifications,
        resume.training,
        resume.teaching,
        resume.additional_sections,
    )):
        return ""

    candidates: list[str] = []
    if resume.education:
        edu = resume.education[0]
        school = edu.school.strip()
        qualification = "、".join(part.strip() for part in (edu.major, edu.degree) if part.strip())
        education_text = "，".join(part for part in (school, qualification) if part)
        if education_text:
            candidates.append(education_text)

    experience_bits: list[str] = []
    for item in resume.experience[:2]:
        identity = "".join(part.strip() for part in (item.organization, item.role) if part.strip())
        if identity and identity not in experience_bits:
            experience_bits.append(identity)
    if experience_bits:
        candidates.append("曾任" + "、".join(experience_bits))

    research_bits: list[str] = []
    for item in resume.research[:2]:
        identity = "".join(part.strip() for part in (item.institution, item.topic) if part.strip())
        if identity and identity not in research_bits:
            research_bits.append(identity)
    if research_bits:
        candidates.append("科研经历包括" + "、".join(research_bits))

    project_names = list(dict.fromkeys(item.name.strip() for item in resume.projects if item.name.strip()))[:4]
    if project_names:
        candidates.append("项目经历包括" + "、".join(project_names))

    achievement = _best_achievement(resume)
    if achievement:
        candidates.append("代表成果：" + achievement)

    if resume.publications:
        candidates.append(f"论文成果{len(resume.publications)}项")
    if resume.patents:
        candidates.append(f"专利成果{len(resume.patents)}项")
    if resume.certifications:
        candidates.append("持有" + "、".join(resume.certifications[:2]))

    if not experience_bits and not research_bits:
        activity_bits = list(dict.fromkeys(
            " ".join(part.strip() for part in (item.organization, item.role) if part.strip())
            for item in resume.activities
            if item.organization.strip() or item.role.strip()
        ))[:2]
        if activity_bits:
            candidates.append("校园或社会活动包括" + "、".join(activity_bits))

    skill_names = list(dict.fromkeys(item.name.strip() for item in resume.skills.items if item.name.strip()))[:8]
    if skill_names:
        candidates.append("技能包括" + "、".join(skill_names))

    target_role = resume.meta.target_role.strip()
    if target_role:
        candidates.append("求职方向为" + target_role)

    compact: list[str] = []
    current_length = 0
    for sentence in candidates:
        sentence = sentence.strip("。；; ")
        if not sentence:
            continue
        added = len(sentence) + 1
        # Keep each selected sentence intact instead of slicing it mid-phrase.
        if current_length + added > 180:
            continue
        compact.append(sentence)
        current_length += added
        if len(compact) >= 6:
            break
    return "。".join(compact) + ("。" if compact else "")


def _compact_canonical(resume: CanonicalResume) -> CanonicalResume:
    """Remove blank records/items left by model repair or leakage cleanup."""

    data = resume.model_dump()
    for section in ("awards", "publications", "patents", "certifications", "training", "teaching"):
        data[section] = list(dict.fromkeys(
            str(v).strip() for v in data.get(section, []) if str(v).strip()
        ))
    additional = data.get("additional_sections") or {}
    if isinstance(additional, dict):
        data["additional_sections"] = {
            str(title).strip(): list(dict.fromkeys(
                str(v).strip() for v in values if str(v).strip()
            ))
            for title, values in additional.items()
            if str(title).strip() and isinstance(values, list)
        }
    skills = data.get("skills") or {}
    if isinstance(skills, dict):
        item_indexes: dict[str, int] = {}
        items: list[dict[str, str]] = []
        for item in skills.get("items", []):
            if not isinstance(item, dict):
                continue
            name = _normalize_skill_name(str(item.get("name", "")))
            category = _normalize_skill_category(name, str(item.get("category", "")))
            key = name.casefold()
            if not name:
                continue
            if key not in item_indexes:
                item_indexes[key] = len(items)
                items.append({"name": name, "category": category})
            elif items[item_indexes[key]]["category"] == "other" and category != "other":
                items[item_indexes[key]]["category"] = category
        skills["items"] = items
    for section, fixed_fields in {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }.items():
        cleaned: list[dict] = []
        record_indexes: dict[tuple[str, ...], int] = {}
        for item in data.get(section, []) or []:
            if not isinstance(item, dict):
                continue
            if section != "education":
                item["bullets"] = list(dict.fromkeys(
                    str(v).strip() for v in item.get("bullets", []) if str(v).strip()
                ))
            bullets = item.get("bullets", []) if section != "education" else []
            if any(str(item.get(field, "")).strip() for field in fixed_fields) or bullets:
                identity = tuple(
                    re.sub(r"\s+", "", str(item.get(field, "")).strip()).casefold()
                    for field in fixed_fields
                )
                if any(identity) and identity in record_indexes:
                    existing = cleaned[record_indexes[identity]]
                    if section != "education":
                        existing["bullets"] = list(dict.fromkeys(
                            list(existing.get("bullets", [])) + list(item.get("bullets", []))
                        ))
                    continue
                if any(identity):
                    record_indexes[identity] = len(cleaned)
                cleaned.append(item)
        data[section] = cleaned
    data["summary"] = str(data.get("summary", "") or "").strip()
    compacted = CanonicalResume.model_validate(data)
    compacted.summary = _build_evidence_summary(compacted)
    return compacted


def _deterministic_fallback(cv_text: str, query_text: str, jd_text: str) -> CanonicalResume:
    """Preserve source facts when the LLM pipeline is unavailable.

    This intentionally parses only CV text as candidate evidence. Query/JD are
    used for classification and direction, never as employment/education facts.
    """
    industry = product_logic.infer_industry(query_text, cv_text, jd_text)
    target_role = product_logic.extract_target_role(query_text, jd_text) if hasattr(product_logic, "extract_target_role") else ""
    raw = product_logic.heuristic_resume_from_text(cv_text, industry, target_role)
    raw = product_logic.normalize_resume_data_for_product(
        raw,
        raw_text=cv_text,
        industry=industry,
        target_role=target_role,
    )

    meta = dict(raw.get("meta") or {})
    experiences = []
    for item in raw.get("experience", []) or []:
        if not isinstance(item, dict):
            continue
        bullets = item.get("bullets") or item.get("responsibilities") or []
        if isinstance(bullets, str):
            bullets = [bullets]
        experiences.append({
            "organization": str(item.get("organization") or item.get("company") or ""),
            "role": str(item.get("role") or ""),
            "period": str(item.get("period") or ""),
            "bullets": [str(value) for value in bullets if str(value).strip()],
        })

    projects = []
    for item in raw.get("projects", []) or []:
        if not isinstance(item, dict):
            continue
        bullets = item.get("bullets") or item.get("description") or []
        if isinstance(bullets, str):
            bullets = [bullets]
        projects.append({
            "name": str(item.get("name") or ""),
            "organization": str(item.get("organization") or item.get("company") or ""),
            "role": str(item.get("role") or ""),
            "period": str(item.get("period") or ""),
            "bullets": [str(value) for value in bullets if str(value).strip()],
        })

    skill_items = []
    category_map = {
        "languages": "language",
        "frameworks": "framework",
        "tools": "tool",
        "domains": "domain",
        "methodologies": "methodology",
        "certifications": "certification",
        "natural_languages": "natural_language",
        "others": "other",
    }
    for category, values in (raw.get("skills") or {}).items():
        if isinstance(values, list):
            skill_items.extend(
                {"name": str(value), "category": category_map.get(category, category)}
                for value in values if str(value).strip()
            )

    return CanonicalResume.model_validate({
        "meta": {
            "name": meta.get("name", ""),
            "phone": meta.get("phone", ""),
            "email": meta.get("email", ""),
            "target_role": meta.get("target_role", target_role),
            "work_experience": meta.get("work_experience", ""),
        },
        "education": raw.get("education") or [],
        "experience": experiences,
        "projects": projects,
        "skills": {"items": skill_items},
        "summary": raw.get("summary", ""),
        "awards": raw.get("awards") or raw.get("honors") or [],
        "publications": raw.get("publications") or [],
        "patents": raw.get("patents") or [],
        "certifications": raw.get("certifications") or [],
        "training": raw.get("training") or [],
        "teaching": raw.get("teaching") or [],
        "additional_sections": raw.get("additional_sections") or {},
    })


def _canonical_to_v1_format(canonical: CanonicalResume) -> dict:
    """Bridge format for existing renderer compatibility."""
    data = canonical.model_dump()
    # Rename organization → company for V1 renderer
    for exp in data.get("experience", []):
        if isinstance(exp, dict) and "organization" in exp:
            exp["company"] = exp.pop("organization")
    for proj in data.get("projects", []):
        if isinstance(proj, dict) and "organization" in proj:
            proj["company"] = proj.pop("organization")
    research = data.get("research", [])
    for item in research:
        if isinstance(item, dict):
            item["company"] = item.pop("institution", "")
            item["role"] = item.pop("topic", "")
    data["campus_experience"] = data.pop("activities", [])
    for item in data["campus_experience"]:
        if isinstance(item, dict) and "organization" in item:
            item["company"] = item.pop("organization")
    # Reuse the renderer's research-output section while retaining the patent
    # type explicitly in display text.
    if data.get("patents"):
        data["publications"] = list(data.get("publications") or []) + [
            value if str(value).startswith("专利") else f"专利：{value}"
            for value in data.get("patents", []) if str(value).strip()
        ]
    # Convert flat skills.items to V1 categorized format
    skills = data.get("skills", {})
    if isinstance(skills, dict):
        items = skills.pop("items", []) if isinstance(skills.get("items"), list) else []
        categorized: dict[str, list[str]] = {
            "languages": [], "frameworks": [], "tools": [], "domains": [],
            "methodologies": [], "certifications": [],
            "natural_languages": [], "others": [],
        }
        if items:
            for item in items:
                if isinstance(item, dict):
                    name = str(item.get("name", "")).strip()
                    if not name:
                        continue
                    cat = item.get("category", "other")
                    normalized_cat = {
                        "language": "languages",
                        "framework": "frameworks",
                        "tool": "tools",
                        "domain": "domains",
                        "methodology": "methodologies",
                        "certification": "certifications",
                        "natural_language": "natural_languages",
                        "other": "others",
                    }.get(cat, cat)
                    if normalized_cat in categorized:
                        categorized[normalized_cat].append(name)
                    else:
                        categorized.setdefault("others", []).append(name)
            skills.update(categorized)
    return data


def _empty_profile_framework(target_role: str = "") -> dict:
    """Rendering-only skeleton for JD-only requests with no candidate facts."""

    return {
        "mode": "empty_profile",
        "notice": "以下内容均为待填写结构，不代表候选人已有事实。",
        "target_role": str(target_role or "").strip(),
        "sections": [
            {"key": "basic_info", "title": "基本信息", "fields": ["姓名", "联系电话", "邮箱", "所在城市"]},
            {"key": "summary", "title": "个人总结", "fields": ["职业背景", "核心优势", "求职方向"]},
            {"key": "education", "title": "教育经历", "fields": ["学校", "学历", "专业", "起止时间"]},
            {"key": "experience", "title": "工作/实习经历", "fields": ["公司", "岗位", "起止时间", "职责与成果"]},
            {"key": "projects", "title": "项目经历", "fields": ["项目名称", "项目角色", "项目时间", "行动与成果"]},
            {"key": "skills", "title": "专业技能", "fields": ["工具/技术", "专业领域", "方法与流程", "证书/语言"]},
        ],
    }


def run_v2_pipeline(
    cv_text: str,
    query_text: str,
    jd_text: str,
) -> VerifiedResult:
    """Run the V2 5-layer pipeline. Returns VerifiedResult or fallback."""
    t_start = time.perf_counter()

    # ── No CV: generate structured framework from query + JD ──
    if not cv_text or not cv_text.strip():
        logger.info("V2 | No CV — generating framework from query+JD")
        t_gen = time.perf_counter()
        resume = compose_from_query(query_text, jd_text)
        used_fallback = False
        if _is_empty_resume(resume) and query_text.strip():
            logger.warning("Generate composer produced an empty resume; using deterministic query fallback")
            resume = _deterministic_fallback(query_text, query_text, jd_text)
            used_fallback = not _is_empty_resume(resume)
        n_exp = len(resume.experience)
        n_proj = len(resume.projects)
        n_bullets = sum(len(e.bullets) for e in resume.experience) + \
                    sum(len(p.bullets) for p in resume.projects) + \
                    sum(len(r.bullets) for r in resume.research)
        logger.info("V2 | Generate done: %d exp, %d proj, %d bullets (%.1fs)",
                    n_exp, n_proj, n_bullets, time.perf_counter() - t_gen)

        grounded_data = resume.model_dump()
        _ground_fixed_fields(grounded_data, query_text)
        _reclassify_non_work(grounded_data, query_text)
        resume = CanonicalResume.model_validate(grounded_data)
        resume = _ground_bullets(resume, query_text)

        # With no structured candidate facts, a polished-sounding summary is
        # misleading.  Keep the target role and ask for missing information in
        # reply_text instead of manufacturing a resume profile.
        has_profile_records = _has_candidate_profile(resume)
        if not has_profile_records:
            resume.summary = ""

        # Generation also receives a dedicated bullet edit pass whenever the
        # user supplied enough factual material to produce bullets.
        # Rank first so optimizer change paths still point at the final bullet
        # positions exposed by the API.
        resume = _rank_resume_content(resume, jd_text or resume.meta.target_role)
        optimizer_changes: list[Change] = []
        if _needs_optimizer(resume):
            before_optimizer = resume.model_copy(deep=True)
            resume = optimize_resume(resume, jd_text)
            optimizer_changes = _bullet_rewrite_changes(before_optimizer, resume)
        else:
            logger.info("V2 | Optimizer skipped: no factual bullets")
        resume = _ground_bullets(resume, query_text)
        resume = validate_resume(resume, source_text=query_text)

        evidence_source = build_source_bundle("", query_text, jd_text)
        resume, evidence_bindings, evidence_removed = enforce_resume_evidence(resume, evidence_source)
        resume = _compact_canonical(resume)
        # Recompute only after unsupported JD-derived records have been
        # removed. Otherwise temporary model output suppresses the framework
        # and produces an almost blank document.
        has_profile_records = _has_candidate_profile(resume)
        if not has_profile_records:
            resume.summary = ""
        evidence_bindings = bind_resume_evidence(resume, evidence_source)
        logger.info("V2 | Evidence bindings: %d", len(evidence_bindings))
        resume_dict = _canonical_to_v1_format(resume)
        if not has_profile_records:
            resume_dict["framework"] = _empty_profile_framework(resume.meta.target_role)
        return VerifiedResult(
            resume=resume,
            changes=([Change(
                path="*",
                action="replace",
                reason="LLM unavailable or invalid; generated from explicit user facts with deterministic parser",
            )] if used_fallback else []) + optimizer_changes + [
                Change(path=path, action="remove", reason="No candidate evidence binding")
                for path in evidence_removed
            ],
            resume_dict=resume_dict,
            evidence_bindings=evidence_bindings,
        )

    # ── Has CV: full Composer → Verifier → Optimizer pipeline ──
    source = build_source_bundle(cv_text, query_text, jd_text)
    logger.info("V2 | SourceBundle: %d blocks (%.1fs)",
                len(source.blocks), time.perf_counter() - t_start)

    t_composer = time.perf_counter()
    draft = compose_resume(source)
    logger.info("V2 | Composer done: %d edu, %d exp, %d res, %d proj (%.1fs)",
                len(draft.education), len(draft.experience),
                len(draft.research), len(draft.projects),
                time.perf_counter() - t_composer)

    t_verifier = time.perf_counter()
    result = _deterministic_verify_draft(source, draft)
    if result is None:
        logger.info("V2 | Falling back to LLM Verifier")
        result = verify_resume(source, draft)
    else:
        logger.info("V2 | LLM Verifier skipped")
    candidate_evidence = "\n".join(block.text for block in candidate_blocks(source))
    result.resume = _ground_bullets(result.resume, candidate_evidence)
    if _is_empty_resume(result.resume):
        logger.warning("V2 verifier produced an empty resume; using deterministic source fallback")
        fallback = _deterministic_fallback(cv_text, query_text, jd_text)
        if not _is_empty_resume(fallback):
            result = VerifiedResult(
                resume=fallback,
                changes=[Change(
                    path="*",
                    action="replace",
                    reason="LLM unavailable or invalid; preserved source facts with deterministic parser",
                )],
            )
    logger.info("V2 | Verifier done: %d edu, %d exp, %d res, %d changes (%.1fs)",
                len(result.resume.education), len(result.resume.experience),
                len(result.resume.research), len(result.changes),
                time.perf_counter() - t_verifier)

    # Rank first so accepted rewrite paths remain stable in the final output.
    result.resume = _rank_resume_content(result.resume, jd_text or result.resume.meta.target_role)
    t_optimizer = time.perf_counter()
    if _needs_optimizer(result.resume):
        before_optimizer = result.resume.model_copy(deep=True)
        result.resume = optimize_resume(result.resume, jd_text)
        result.changes.extend(_bullet_rewrite_changes(before_optimizer, result.resume))
    else:
        logger.info("V2 | Optimizer skipped: no factual bullets")
    result.resume = _ground_bullets(result.resume, candidate_evidence)
    logger.info("V2 | Optimizer done (%.1fs)", time.perf_counter() - t_optimizer)

    # Deterministic repair uses candidate evidence only.  JD schools, employers
    # and dates must never backfill candidate fields.
    _source_for_validate = candidate_evidence
    result.resume = validate_resume(result.resume, source_text=_source_for_validate)
    result.resume, _, evidence_removed = enforce_resume_evidence(result.resume, source)
    result.changes.extend(
        Change(path=path, action="remove", reason="No candidate evidence binding")
        for path in evidence_removed
    )
    result.resume = _compact_canonical(result.resume)
    result.evidence_bindings = bind_resume_evidence(result.resume, source)
    coverage, missing_blocks = measure_source_coverage(source, result.evidence_bindings)
    content_block_count = len({binding.block_id for binding in result.evidence_bindings}) + len(missing_blocks)
    if content_block_count >= 5 and coverage < 0.62:
        # Prefer a less polished deterministic parse only when it demonstrably
        # preserves substantially more of the source. This prevents a valid
        # long resume from collapsing to a few attractive bullets.
        try:
            fallback = _deterministic_fallback(cv_text, query_text, jd_text)
            fallback = _ground_bullets(fallback, candidate_evidence)
            fallback, fallback_bindings, _ = enforce_resume_evidence(fallback, source)
            fallback = _compact_canonical(fallback)
            fallback_bindings = bind_resume_evidence(fallback, source)
            fallback_coverage, _ = measure_source_coverage(source, fallback_bindings)
            if fallback_coverage >= coverage + 0.10:
                logger.warning(
                    "V2 | Replaced low-coverage result with source-preserving fallback: %.1f%% -> %.1f%%",
                    coverage * 100,
                    fallback_coverage * 100,
                )
                result.resume = fallback
                result.evidence_bindings = fallback_bindings
                result.changes.append(Change(
                    path="*",
                    action="replace",
                    reason="Source coverage repair used the more complete deterministic parse",
                ))
                coverage = fallback_coverage
        except Exception as exc:
            logger.warning("V2 | Source coverage fallback failed: %s", exc)
    if missing_blocks:
        logger.info("V2 | Final source coverage %.1f%%, missing=%s", coverage * 100, missing_blocks[:8])
    result.resume_dict = _canonical_to_v1_format(result.resume)
    logger.info("V2 | Evidence bindings: %d", len(result.evidence_bindings))

    logger.info("V2 | Total: %.1fs (Composer+Verifier+Validate+Format)",
                time.perf_counter() - t_start)

    return result
