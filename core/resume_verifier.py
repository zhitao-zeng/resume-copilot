"""ResumeVerifier: LLM Call 2 — verify DraftResume, output CanonicalResume.

V2 Layer 3.  Directly returns corrected resume — no intermediate report.
"""
from __future__ import annotations

import json
import logging

from prompts import RESUME_VERIFIER_SYSTEM_PROMPT
from server_runtime import call_llm_text, llm_enabled
from llm_gateway import parse_json_content
from v2_schemas import (
    SourceBundle, DraftResume, VerifiedResult, CanonicalResume,
    Change, Meta, Education, Experience, Project,
)

logger = logging.getLogger(__name__)


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

    source_parts = [f"[{b.block_id}] {b.text}" for b in source.blocks]
    draft_json = draft.model_dump_json(exclude_none=True)

    prompt = (
        "请审核以下 DraftResume，输出修正后的最终简历。\n\n"
        "【原始材料】\n"
        f"{chr(10).join(source_parts)}\n\n"
        "【DraftResume】\n"
        f"{draft_json}"
    )

    try:
        content = call_llm_text(
            RESUME_VERIFIER_SYSTEM_PROMPT,
            prompt,
            temperature=0.0,
            max_tokens=3072,
        )
    except Exception as exc:
        logger.warning("ResumeVerifier LLM call failed: %s", exc)
        return conservative_fallback()

    parsed = parse_json_content(content)
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
    TOP_LEVEL_KEYS = {"education", "experience", "projects", "skills", "summary"}
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
    resume_query_text = "\n".join(b.text for b in source.blocks if b.source_type in ("resume", "query"))
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

    try:
        resume = CanonicalResume(**parsed)
        result = VerifiedResult(resume=resume)
        return result
    except Exception as exc:
        logger.warning("ResumeVerifier output validation failed: %s", exc)
        return conservative_fallback()
