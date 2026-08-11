"""ResumeVerifier: LLM Call 2 — verify DraftResume, output CanonicalResume.

V2 Layer 3.  Directly returns corrected resume — no intermediate report.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata

from prompts import RESUME_VERIFIER_SYSTEM_PROMPT
from server_runtime import call_llm_text, llm_enabled
from llm_gateway import parse_json_content
from source_adapter import candidate_blocks
from v2_schemas import (
    SourceBundle, DraftResume, VerifiedResult, CanonicalResume,
    Change, Meta, Education, Experience, Project,
)
from diagnostic_trace import trace_event

logger = logging.getLogger(__name__)


def _normalize_evidence_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"[\s，。；、：:,.!?！？()（）\[\]【】'\"]+", "", value)


def _date_signature(value: str) -> tuple[str, ...]:
    """Normalize yyyy-mm/mm-yyyy/Chinese dates for factual comparison."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    pattern = re.compile(
        r"(?<!\d)(?:(?P<year1>(?:19|20)\d{2})\s*[-./年]\s*(?P<month1>0?[1-9]|1[0-2])\s*月?"
        r"|(?P<month2>0?[1-9]|1[0-2])\s*[-./]\s*(?P<year2>(?:19|20)\d{2}))(?!\d)"
    )
    signature: list[str] = []
    for match in pattern.finditer(text):
        year = match.group("year1") or match.group("year2")
        month = match.group("month1") or match.group("month2")
        signature.append(f"{year}{int(month):02d}")
    return tuple(signature)


def _has_equivalent_date_evidence(value: str, evidence_text: str) -> bool:
    signature = _date_signature(value)
    if not signature:
        return False
    for segment in re.split(r"[\n；;]", evidence_text):
        source_signature = _date_signature(segment)
        if len(source_signature) >= len(signature):
            for index in range(len(source_signature) - len(signature) + 1):
                if source_signature[index:index + len(signature)] == signature:
                    return True
    return False


def _has_positive_evidence(value: str, evidence_text: str) -> bool:
    """Return whether a fixed resume fact is explicitly supported.

    Negated instructions such as ``不要增加某公司`` do not count as facts.
    """

    value = str(value or "").strip()
    if not value:
        return True
    if _has_equivalent_date_evidence(value, evidence_text):
        return True
    normalized_value = _normalize_evidence_text(value)
    normalized_source = _normalize_evidence_text(evidence_text)
    if not normalized_value or normalized_value not in normalized_source:
        # A listed topic such as "OCR识别、目标检测等项目" can safely be
        # normalized to "OCR识别项目" without inventing a new project.
        core_value = re.sub(r"(?:项目|课题|系统|平台)$", "", normalized_value)
        if len(core_value) < 3 or core_value not in normalized_source:
            return False
    for match in re.finditer(re.escape(value), evidence_text, re.IGNORECASE):
        prefix = evidence_text[max(0, match.start() - 12):match.start()]
        if not re.search(r"(?:不要|禁止|不能|避免|不得|并非|没有)[^，。；\n]{0,8}$", prefix):
            return True
    # Normalized-only matches are accepted unless the source contains an
    # obvious negated occurrence of the same value.
    return not re.search(
        rf"(?:不要|禁止|不能|避免|不得|并非|没有)[^，。；\n]{{0,8}}{re.escape(value)}",
        evidence_text,
        re.IGNORECASE,
    )


def _ground_fixed_fields(parsed: dict, evidence_text: str) -> None:
    """Clear unsupported immutable fields and empty skill items in place."""

    meta = parsed.get("meta")
    if isinstance(meta, dict):
        for key in ("name", "phone", "email", "work_experience"):
            if meta.get(key) and not _has_positive_evidence(str(meta[key]), evidence_text):
                logger.info("ResumeVerifier cleared unsupported meta.%s: %s", key, meta[key])
                meta[key] = ""

    section_fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, fields in section_fields.items():
        items = parsed.get(section)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for field in fields:
                value = str(item.get(field, "") or "").strip()
                if value and not _has_positive_evidence(value, evidence_text):
                    logger.info("ResumeVerifier cleared unsupported %s.%s: %s", section, field, value)
                    item[field] = ""

    skills = parsed.get("skills")
    if isinstance(skills, dict) and isinstance(skills.get("items"), list):
        skills["items"] = [
            item for item in skills["items"]
            if isinstance(item, dict)
            and str(item.get("name", "")).strip()
            and _has_positive_evidence(str(item.get("name", "")), evidence_text)
        ]

    awards = parsed.get("awards")
    if isinstance(awards, list):
        parsed["awards"] = [
            str(item).strip() for item in awards
            if str(item).strip() and _has_positive_evidence(str(item), evidence_text)
        ]
    for section in ("publications", "patents", "certifications", "training", "teaching"):
        values = parsed.get(section)
        if isinstance(values, list):
            parsed[section] = [
                str(item).strip() for item in values
                if str(item).strip() and _has_positive_evidence(str(item), evidence_text)
            ]
    additional = parsed.get("additional_sections")
    if isinstance(additional, dict):
        parsed["additional_sections"] = {
            str(title).strip(): [
                str(item).strip() for item in values
                if str(item).strip() and _has_positive_evidence(str(item), evidence_text)
            ]
            for title, values in additional.items()
            if str(title).strip() and isinstance(values, list)
        }


def _reclassify_non_work(parsed: dict, evidence_text: str) -> None:
    experiences = parsed.get("experience")
    if not isinstance(experiences, list):
        return
    kept: list[dict] = []
    activities = list(parsed.get("activities") or [])
    research = list(parsed.get("research") or [])
    student_context = bool(re.search(r"(?:在读|学生|本科|硕士|博士|毕业)", evidence_text))
    for item in experiences:
        if not isinstance(item, dict):
            continue
        org = str(item.get("organization", "") or "")
        role = str(item.get("role", "") or "")
        combined = f"{org} {role}"
        if re.search(r"(?:学生会|志愿者|志愿服务|社团|协会|校园组织)", combined):
            activities.append(item)
            continue
        if student_context and (
            re.search(r"(?:实验室|课题组|研究院|科研)", combined)
            or (re.search(r"(?:大学|学院)", org) and re.search(r"(?:研究|算法|视觉|多模态)", role))
        ):
            research.append({
                "institution": org,
                "topic": role,
                "period": item.get("period", ""),
                "bullets": item.get("bullets", []),
            })
            continue
        kept.append(item)
    parsed["experience"] = kept
    parsed["activities"] = activities
    parsed["research"] = research


def conservative_fallback() -> VerifiedResult:
    """When Verifier fails, return empty safe result — never return unverified Draft."""
    return VerifiedResult(
        resume=CanonicalResume(
            meta=Meta(),
            education=[],
            experience=[],
            projects=[],
            summary="",
        ),
        changes=[Change(path="*", action="remove",
                        reason="Verifier failed, emitted empty fallback")],
    )


def verify_resume(source: SourceBundle, draft: DraftResume) -> VerifiedResult:
    """Call LLM to verify and produce CanonicalResume.

    Uses call_llm_text (no schema injection) because Qwen3.5-8B's
    json_object mode + full JSON Schema causes the model to echo
    the schema itself instead of producing actual data.
    """
    if not llm_enabled():
        return conservative_fallback()

    source_parts = [
        f"[{b.block_id}{'|section=' + b.section_hint if b.section_hint else ''}] {b.text}"
        for b in source.blocks
    ]
    draft_json = draft.model_dump_json(exclude_none=True)

    prompt = (
        "请审核以下 DraftResume，输出修正后的最终简历。\n\n"
        "【原始材料】\n"
        f"{chr(10).join(source_parts)}\n\n"
        "【DraftResume】\n"
        f"{draft_json}"
    )
    trace_event(
        "llm_verifier_request",
        source=source,
        draft=draft,
        system_prompt=RESUME_VERIFIER_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=4096,
    )

    try:
        content = call_llm_text(
            RESUME_VERIFIER_SYSTEM_PROMPT,
            prompt,
            temperature=0.0,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.warning("ResumeVerifier LLM call failed: %s", exc)
        return conservative_fallback()

    parsed = parse_json_content(content)
    trace_event("llm_verifier_raw_response", content=content, parsed=parsed)
    if not isinstance(parsed, dict) or not parsed:
        logger.warning("ResumeVerifier JSON parse failed, content len=%d", len(content))
        return conservative_fallback()

    # Strip wrapper key if LLM wrapped output (e.g. {"resume": {…}})
    WRAPPER_KEYS = {"resume", "data", "result"}
    if len(parsed) == 1:
        wrapped_key = next(iter(parsed))
        if wrapped_key in WRAPPER_KEYS:
            inner = parsed[wrapped_key]
            if isinstance(inner, dict):
                parsed = inner
                logger.info("ResumeVerifier unwrapped key %s", wrapped_key)

    # Detect if LLM nested all content under meta (e.g. {"meta": {"name":"", "education":[...]}})
    TOP_LEVEL_KEYS = {
        "education", "experience", "research", "activities", "projects", "skills", "summary",
        "awards", "publications", "patents", "certifications", "training", "teaching",
        "additional_sections",
    }
    meta = parsed.get("meta")
    if isinstance(meta, dict):
        for key in TOP_LEVEL_KEYS:
            if key in meta and not parsed.get(key):
                parsed[key] = meta.pop(key)
        if any(k in meta for k in TOP_LEVEL_KEYS):
            logger.info("ResumeVerifier unnested content from meta")

    # Detect flat output: if parsed has flat fields like "school" but no "education"/"experience"/"meta"
    FLAT_TO_NESTED = {
        "meta": {"name", "phone", "email", "target_role", "work_experience"},
        "education_single": {"school", "degree", "major", "period"},
        "experience_single": {"organization", "role", "period", "bullets"},
    }
    has_flat_fields = any(k in parsed for k in
                          {"name", "phone", "email", "school", "organization"})
    has_nested = ("meta" in parsed or "education" in parsed or "experience" in parsed)
    if has_flat_fields and not has_nested:
        logger.info("ResumeVerifier detected flat output, normalizing to nested")
        # Extract meta fields
        meta_fields = {k: parsed.pop(k, "") for k in FLAT_TO_NESTED["meta"] if k in parsed}
        parsed["meta"] = meta_fields

        # Extract education (if school field exists)
        edu_fields = {k: parsed.pop(k, "") for k in FLAT_TO_NESTED["education_single"] if k in parsed}
        if edu_fields.get("school"):
            # Handle bullet → bullets
            if "bullet" in parsed and "bullets" not in edu_fields:
                edu_fields["bullets"] = [parsed.pop("bullet")] if isinstance(parsed["bullet"], str) else parsed.pop("bullet", [])
            parsed["education"] = [edu_fields]

        # Extract experience (if organization field exists)
        exp_fields = {k: parsed.pop(k, "") for k in FLAT_TO_NESTED["experience_single"] if k in parsed}
        if exp_fields.get("organization"):
            if "bullet" in parsed and "bullets" not in exp_fields:
                exp_fields["bullets"] = [parsed.pop("bullet")] if isinstance(parsed["bullet"], str) else parsed.pop("bullet", [])
            parsed["experience"] = [exp_fields]

    # Strip unknown fields from sub-models (LLM may add extra fields like
    # "description" which CanonicalResume ignores but sub-models forbid).
    _STRIP_KEYS = {
        "education": {"school", "degree", "major", "period"},
        "experience": {"organization", "role", "period", "bullets"},
        "research": {"institution", "topic", "period", "bullets"},
        "activities": {"organization", "role", "period", "bullets"},
        "projects": {"name", "organization", "role", "period", "bullets"},
        "meta": {"name", "phone", "email", "target_role", "work_experience"},
        "skills": {"items"},  # Skills is flat items format, not categorized dict
    }
    for section, allowed in _STRIP_KEYS.items():
        items = parsed.get(section)
        if isinstance(items, list):
            parsed[section] = [
                {k: v for k, v in item.items() if k in allowed}
                if isinstance(item, dict) else item
                for item in items
            ]

    # Fix skills format: if skills is a flat list, convert to items format
    skills = parsed.get("skills")
    if isinstance(skills, list):
        # Flat list like ["Python", "PyTorch"] → items format with category="other"
        parsed["skills"] = {"items": [
            {"name": s, "category": "other"} for s in skills if isinstance(s, str)
        ]}

    # ── Post-processing: merge adjacent duplicate experiences ──
    exp_list = parsed.get("experience", [])
    if isinstance(exp_list, list):
        merged = []
        for entry in exp_list:
            if not isinstance(entry, dict):
                merged.append(entry)
                continue
            if merged and (
                merged[-1].get("organization", "") == entry.get("organization", "")
                and merged[-1].get("role", "") == entry.get("role", "")
            ):
                # Merge bullets
                existing_bullets = merged[-1].get("bullets", [])
                new_bullets = entry.get("bullets", [])
                # Dedup and append
                seen = set(existing_bullets)
                for b in new_bullets:
                    if b not in seen:
                        existing_bullets.append(b)
                        seen.add(b)
            else:
                merged.append(dict(entry))
        parsed["experience"] = merged

    # ── Post-processing: source-based guard ──
    source_text = "\n".join(b.text for b in source.blocks)

    # ── JD isolation: values only in JD text → clear ──
    jd_text = "\n".join(b.text for b in source.blocks if b.source_type == "jd")
    resume_query_text = "\n".join(b.text for b in candidate_blocks(source))
    if jd_text.strip():
        for entry in parsed.get("experience", []):
            if not isinstance(entry, dict):
                continue
            org = entry.get("organization", "")
            role = entry.get("role", "")
            if org and org not in resume_query_text and org in jd_text:
                entry["organization"] = ""
                logger.info("ResumeVerifier cleared JD-only org: %s", org)
            if role and role not in resume_query_text and role in jd_text:
                entry["role"] = ""
                logger.info("ResumeVerifier cleared JD-only role: %s", role)

    for entry in parsed.get("education", []):
        if not isinstance(entry, dict):
            continue
        school = entry.get("school", "")
        if school and school not in resume_query_text and school in jd_text:
            entry["school"] = ""
            logger.info("ResumeVerifier cleared JD-only school: %s", school)
        # (education school validation handled by Verifier LLM prompt)

    # Restore role from draft if Verifier changed it
    for entry in parsed.get("experience", []):
        if not isinstance(entry, dict):
            continue
        org = entry.get("organization", "")
        verifier_role = entry.get("role", "")
        for de in draft.experience:
            if de.organization == org and de.role and de.role != verifier_role:
                if de.role in source_text:
                    logger.info("ResumeVerifier restored role '%s' from draft (was '%s')", de.role, verifier_role)
                    entry["role"] = de.role
                    break

    # Clear fabricated identity fields (name/phone/email not in source)
    if isinstance(parsed.get("meta"), dict):
        PLACEHOLDER_PHONE_PATTERNS = {"138-xxxx", "13800138000", "13900139000",
                                       "13800000000", "13900000000", "10086",
                                       "12345678901", "1234567890"}
        PLACEHOLDER_EMAIL_DOMAINS = {"example.com", "test.com", "mail.com",
                                      "email.com", "domain.com"}
        PLACEHOLDER_EMAIL_FULL = {"xxxx@xxxx.com", "test@test.com",
                                   "zhangsan@example.com", "lisi@example.com"}

        for field in ("name", "phone", "email"):
            val = parsed["meta"].get(field, "")
            if not val:
                continue

            # Check placeholder patterns FIRST, before source_text check
            # (placeholder like "用户" may match source_text as a substring)
            DEFAULT_NAMES = {"张三", "李四", "王五", "用户", "test", "测试", "姓名", "用户姓名"}
            if field == "name" and val in DEFAULT_NAMES:
                parsed["meta"][field] = ""
                logger.info("ResumeVerifier cleared placeholder name '%s'", val)
                continue

            if val in source_text:
                continue  # Has direct evidence
            if field == "phone":
                clean = val.replace("-", "").replace(" ", "").replace("×", "x")
                if val in PLACEHOLDER_PHONE_PATTERNS or len(clean) < 8:
                    parsed["meta"][field] = ""
                    logger.info("ResumeVerifier cleared placeholder phone '%s'", val)
                elif clean.isdigit() and clean not in source_text:
                    parsed["meta"][field] = ""
                    logger.info("ResumeVerifier cleared phone '%s' (not in source)", val)
            elif field == "email":
                if val.lower() in PLACEHOLDER_EMAIL_FULL:
                    parsed["meta"][field] = ""
                    logger.info("ResumeVerifier cleared placeholder email '%s'", val)
                elif "@" in val:
                    domain = val.split("@")[1].lower()
                    if domain in PLACEHOLDER_EMAIL_DOMAINS:
                        parsed["meta"][field] = ""
                        logger.info("ResumeVerifier cleared placeholder email domain '%s'", val)
                    elif val not in source_text:
                        parsed["meta"][field] = ""
                        logger.info("ResumeVerifier cleared email '%s' (not in source)", val)
            elif field == "name":
                if len(val) <= 4 and val not in source_text:
                    parsed["meta"][field] = ""
                    logger.info("ResumeVerifier cleared name '%s' (not in source)", val)

    # Restore draft projects/skills if Verifier dropped them
    if not parsed.get("projects") and draft.projects:
        logger.info("ResumeVerifier restored %d projects from draft", len(draft.projects))
        parsed["projects"] = [p.model_dump() for p in draft.projects]
    if not parsed.get("skills") or (
        isinstance(parsed.get("skills"), dict)
        and not parsed["skills"].get("items")
    ):
        draft_skills = draft.skills.model_dump() if draft.skills else {}
        draft_items = draft_skills.get("items", []) if isinstance(draft_skills, dict) else []
        if draft_items:
            logger.info("ResumeVerifier restored skills from draft (%d items)", len(draft_items))
            parsed["skills"] = draft_skills
    for section in (
        "publications", "patents", "certifications", "training", "teaching", "additional_sections",
    ):
        draft_value = getattr(draft, section)
        if draft_value and not parsed.get(section):
            parsed[section] = draft_value
            logger.info("ResumeVerifier restored %s from draft", section)

    # The LLM verifier is advisory; immutable facts are grounded again in
    # code against candidate evidence (resume + factual user additions).
    _ground_fixed_fields(parsed, resume_query_text)
    _reclassify_non_work(parsed, resume_query_text)

    try:
        resume = CanonicalResume(**parsed)
        result = VerifiedResult(resume=resume)
        trace_event("llm_verifier_final", result=result)
        return result
    except Exception as exc:
        logger.warning("ResumeVerifier output validation failed: %s", exc)
        return conservative_fallback()
