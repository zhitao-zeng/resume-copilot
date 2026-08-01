"""Resume generation from personal profile — scenarios 2 & 4."""

import json
import re
from typing import Any, Optional

from pydantic import BaseModel

from prompts import JD_PROFILE_SYSTEM_PROMPT
from resume_parsing import (
    _log_parse_text_debug,
    structured_resume_from_text,
)
from resume_renderer import export_resume_files, render_docx
from resume_scoring import score_resume
from resume_validator import (
    check_fabrication_heuristic,
    check_required_fields,
    check_time_conflicts,
    check_summary_jd_alignment,
    calculate_experience_years,
)
from schemas import GenerateResponse, JobFamily, ResumeScore, UserStage
from server_runtime import call_llm_typed, llm_enabled, logger, sanitize_user_text


def _classify_job_family(personal_profile: str, jd_text: Optional[str] = None) -> str:
    if not llm_enabled():
        logger.warning("LLM not enabled for job classification, defaulting to 'other'")
        return "other"

    try:
        prompt = (
            "请判断以下描述对应的目标行业类别，仅输出类别名称，不限于任何固定列表：\n\n"
            f"个人描述：{sanitize_user_text(personal_profile)[:1000]}\n\n"
            f"目标JD（可选）：{sanitize_user_text(jd_text or '')[:1000]}\n\n"
        )
        class_result = call_llm_typed(_JobClassifyModel, JD_PROFILE_SYSTEM_PROMPT, prompt, temperature=0.1)
        family = class_result.get("job_family", "other")
        return family or "other"
    except Exception as exc:
        logger.warning("Job classification failed: %s, defaulting to 'other'", exc)
        return "other"


class _JobClassifyModel(BaseModel):
    job_family: str = "other"


class _PersonalToResumeModel:
    """Mock model for personal-to-resume conversion."""
    pass


def generate_resume_from_profile(
    personal_profile: str,
    jd_text: Optional[str] = None,
    job_family: Optional[str] = None,
    user_stage: Optional[str] = None,
    target_description: Optional[str] = None,
    template: str = "new_standard",
    output_format: str = "both",
) -> dict[str, Any]:
    """Generate a structured resume from a personal profile (scenarios 2 & 4)."""

    if not job_family:
        job_family = _classify_job_family(personal_profile, jd_text)

    resume_text = personal_profile
    if jd_text:
        resume_text += f"\n\n目标JD：{jd_text}"

    # Parse to structured JSON
    try:
        resume_data = structured_resume_from_text(resume_text)
    except Exception as exc:
        logger.warning("Profile-to-resume generation failed: %s, falling back to heuristic", exc)
        resume_data = _heuristic_generate_from_profile(personal_profile, job_family)

    # Add experience years
    experience = resume_data.get("experience", [])
    if isinstance(experience, list):
        resume_data["experience_years"] = calculate_experience_years(experience)

    # Generate personal summary
    if not resume_data.get("summary") or len(resume_data.get("summary", "").strip()) < 20:
        summary = _build_summary(resume_data, job_family, target_description)
        if summary:
            resume_data["summary"] = summary

    # Validate
    missing_fields = check_required_fields(resume_data, user_stage)
    conflicts = check_time_conflicts(resume_data)
    summary_conflicts = check_summary_jd_alignment(resume_data.get("summary", ""), jd_text)
    all_conflicts = conflicts + summary_conflicts
    fab_report = check_fabrication_heuristic(personal_profile, resume_data)

    # Score
    score = score_resume(
        resume_data,
        personal_profile,
        user_report={},
        job_family=job_family,
        user_stage=user_stage,
        missing_fields=missing_fields,
        conflicts=all_conflicts,
        fabrication_report=fab_report,
    )

    # Build user report
    user_report = _build_generation_user_report(missing_fields, conflicts, fab_report)

    # Build generation direction
    generation_direction = _build_generation_direction(job_family, target_description)

    from server_runtime import OUTPUT_DIR

    # Render
    files = export_resume_files(resume_data, output_dir=OUTPUT_DIR, template=template, output_format=output_format)

    # Write to disk
    output_dirs = _get_output_dirs()
    for fmt, file_path in files.items():
        if file_path:
            logger.info("Generated resume saved to %s (format=%s)", file_path, fmt)

    return {
        "resume_data": resume_data,
        "score": score,
        "missing_fields": missing_fields,
        "conflicts": conflicts,
        "fabrication_report": fab_report,
        "files": files,
        "user_report": user_report,
        "generation_direction": generation_direction,
    }


def _heuristic_generate_from_profile(profile: str, job_family: Optional[str]) -> dict[str, Any]:
    """Heuristic fallback when LLM is not available."""
    resume_data = {
        "meta": {
            "name": "",
            "phone": "",
            "email": "",
        },
        "education": [],
        "experience": [],
        "projects": [],
        "skills": {"languages": [], "frameworks": [], "tools": [], "domains": []},
        "summary": f"具有{job_family or '综合'}背景的求职者",
    }

    # Try to extract email
    email_match = re.search(r"([\w.-]+@[\w.-]+\.\w+)", profile)
    if email_match:
        resume_data["meta"]["email"] = email_match.group(1)

    # Try to extract phone
    phone_match = re.search(r"(1[3-9]\d{9})|(\+?86[-]?\d{10,11})|(\d{11})", profile)
    if phone_match:
        resume_data["meta"]["phone"] = phone_match.group(1)

    # Try to extract name
    name_match = re.search(r"(?:姓名|名字|叫|昵称)[:：\s]*([\u4e00-\u9fff]{2,4})", profile)
    if name_match:
        resume_data["meta"]["name"] = name_match.group(1)

    return resume_data


def _build_summary(resume_data: dict[str, Any], job_family: Optional[str], target_description: Optional[str]) -> str:
    experience_years = resume_data.get("experience_years", 0)
    skills = resume_data.get("skills", {})
    skill_list = []
    if isinstance(skills, dict):
        for bucket in ("languages", "frameworks", "tools", "domains"):
            for s in skills.get(bucket, [])[:3]:
                skill_list.append(s)

    parts = []
    if job_family:
        family_names = {
            "product_research": "产研", "operations": "运营", "doctor": "医疗",
            "teacher": "教育", "sales_presale": "售前/销售", "finance": "金融",
            "design": "设计", "education": "教育", "legal": "法律",
        }
        family_name = family_names.get(job_family, "技术")
        parts.append(f"具备{experience_years}年经验的{family_name}领域从业者")
    else:
        parts.append(f"具备{experience_years}年经验的从业者")

    if skill_list:
        parts.append(f"熟练掌握{', '.join(skill_list[:3])}")

    if target_description:
        parts.append(f"期望从事{target_description[:20]}工作")

    return "；".join(parts) if parts else f"具有{experience_years}年经验的从业者"


def _build_generation_user_report(
    missing_fields: list[Any],
    conflicts: list[Any],
    fab_report: Any,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    if missing_fields:
        report["missing_field_suggestions"] = [
            {"field": mf.field, "reason": mf.reason} for mf in missing_fields
        ]
    if conflicts:
        report["conflict_confirmations"] = [
            {"field": c.field, "description": c.description} for c in conflicts
        ]
    if fab_report.fabrication_found:
        report["fabrication_details"] = [
            {"type": d.type, "content": d.content, "reason": d.reason}
            for d in fab_report.details
        ]
    return report


def _build_generation_direction(job_family: Optional[str], target_description: Optional[str]) -> str:
    family_names = {
        "product_research": "产研", "operations": "运营", "doctor": "医疗",
        "teacher": "教育", "sales_presale": "售前/销售", "finance": "金融",
        "design": "设计", "education": "教育", "legal": "法律",
    }
    family = family_names.get(job_family, "") if job_family else ""
    if target_description:
        return f"根据您提供的个人描述，建议投递{target_description[:20]}相关岗位"
    elif family:
        return f"您的背景适合{family}类岗位，建议针对性优化简历"
    return "建议明确目标岗位方向以优化简历"


def _build_generation_message(score: ResumeScore) -> str:
    if score.total == 0:
        return "简历生成失败：检测到捏造内容"
    if score.total >= 90:
        return "简历生成成功，评分优秀"
    elif score.total >= 70:
        return "简历生成成功，建议根据缺失字段提示补充信息"
    return "简历生成成功，但评分偏低，建议补充更多信息"


def _get_output_dirs() -> dict[str, str]:
    return {
        "docx": "",
        "pdf": "",
    }
