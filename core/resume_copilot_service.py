"""Unified resume copilot product flow.

This module implements the acceptance-oriented entrypoint described by the
product spec: query + optional resume/template/JD inputs -> editable DOCX +
natural-language reply + scoring metadata.
"""

from __future__ import annotations

import copy
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from http_compat import HTTPException, UploadFile

try:
    import aiohttp
except ImportError:
    aiohttp = None

import resume_product_logic as product_logic
from resume_classifier import classify_resume_request
from audit_logic import audit_resume_core
from drafts import create_new_draft
from resume_generator import _build_generation_direction
from resume_io import IMAGE_EXTENSIONS, extract_text_from_bytes
from resume_optimization import optimize_resume_core, run_single_optimize_with_audit_pass
from resume_parsing import cleanup_ocr_text, resume_data_to_text, structured_resume_from_text
from resume_renderer import export_resume_files
from resume_scoring import score_resume
from resume_validator import (
    calculate_experience_years,
    check_fabrication_heuristic,
    check_required_fields,
    check_summary_jd_alignment,
    check_sort_order,
    check_time_conflicts,
    FabricationReport,
)
from schemas import ResumeCopilotResponse, StructuredResumeLLMOutput
from server_runtime import AVATAR_DIR, DEFAULT_TEMPLATE, DRAFTS_DIR, MAX_FILE_SIZE, OUTPUT_DIR, REQUEST_TIMEOUT_SECONDS, call_llm_text, call_llm_typed, llm_enabled, logger, sanitize_user_text
from prompts import REPLY_GENERATION_SYSTEM_PROMPT
from resume_product_logic import INDUSTRY_LABELS


def _dicts(items: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            result.append(item.model_dump())
        elif isinstance(item, dict):
            result.append(item)
        else:
            result.append(dict(getattr(item, "__dict__", {})))
    return result


def _looks_like_url(value: str) -> bool:
    return bool(re.match(r"^https?://", str(value or "").strip(), re.IGNORECASE))


def _file_ext(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def _ensure_time_budget(started: float, stage: str) -> None:
    elapsed = time.perf_counter() - started
    if elapsed > REQUEST_TIMEOUT_SECONDS:
        raise HTTPException(
            status_code=504,
            detail=f"resume-copilot exceeded {REQUEST_TIMEOUT_SECONDS}s during {stage}; please retry with shorter files or clearer text",
        )


async def _extract_upload_text(upload: UploadFile, purpose: str, perf: dict[str, float], warnings: list[dict[str, Any]]) -> str:
    started = time.perf_counter()
    raw = await upload.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"{purpose} file too large (> {MAX_FILE_SIZE // (1024 * 1024)} MB)")
    filename = upload.filename or f"{purpose}.bin"
    ext = _file_ext(filename)
    if ext in IMAGE_EXTENSIONS:
        warnings.append(
            {
                "source": purpose,
                "filename": filename,
                "message": "已对图片执行本地 OCR；如识别不完整，请补充清晰图片或文本。",
            }
        )
    try:
        text = extract_text_from_bytes(raw, filename)
    except HTTPException as exc:
        if ext in IMAGE_EXTENSIONS:
            warnings.append(
                {
                    "source": purpose,
                    "filename": filename,
                    "message": "图片内容无法可靠识别，请补充文本或重新上传清晰图片。",
                }
            )
            perf[f"{purpose}_extract_s"] = round(time.perf_counter() - started, 3)
            return ""
        raise exc
    perf[f"{purpose}_extract_s"] = round(time.perf_counter() - started, 3)
    if ext in IMAGE_EXTENSIONS and len(text.strip()) < 30:
        warnings.append(
            {
                "source": purpose,
                "filename": filename,
                "message": "图片 OCR 文本较短，可能存在识别缺失，请确认关键信息。",
            }
        )
    return text


async def _fetch_jd_url(url: str, warnings: list[dict[str, Any]]) -> str:
    if aiohttp is None:
        warnings.append({"source": "target_jd", "message": "缺少 aiohttp 依赖，无法解析 JD 链接。"})
        return ""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    warnings.append({"source": "target_jd", "message": f"JD 链接请求失败：HTTP {resp.status}"})
                    return ""
                content_type = resp.headers.get("Content-Type", "")
                payload = await resp.read()
                if len(payload) > MAX_FILE_SIZE:
                    warnings.append({"source": "target_jd", "message": "JD 链接内容超过大小限制，已忽略。"})
                    return ""
                if "pdf" in content_type:
                    return extract_text_from_bytes(payload, "target_jd.pdf")
                text = payload.decode("utf-8", errors="ignore")
                text = re.sub(r"<[^>]+>", "\n", text)
                text = re.sub(r"&nbsp;", " ", text)
                return re.sub(r"\n\s*\n", "\n\n", text).strip()
    except Exception as exc:
        warnings.append({"source": "target_jd", "message": f"JD 链接解析失败：{exc}"})
        return ""


async def _resolve_jd_text(
    *,
    target_jd: Optional[str],
    jd_text: Optional[str],
    target_jd_url: Optional[str],
    jd_url: Optional[str],
    target_jd_file: Optional[UploadFile],
    perf: dict[str, float],
    warnings: list[dict[str, Any]],
) -> str:
    started = time.perf_counter()
    text_candidates = [target_jd, jd_text]
    for candidate in text_candidates:
        value = str(candidate or "").strip()
        if not value:
            continue
        if _looks_like_url(value):
            resolved = await _fetch_jd_url(value, warnings)
            perf["jd_resolve_s"] = round(time.perf_counter() - started, 3)
            return resolved
        perf["jd_resolve_s"] = round(time.perf_counter() - started, 3)
        return value

    url = str(target_jd_url or jd_url or "").strip()
    if url:
        resolved = await _fetch_jd_url(url, warnings)
        perf["jd_resolve_s"] = round(time.perf_counter() - started, 3)
        return resolved

    if target_jd_file is not None:
        text = await _extract_upload_text(target_jd_file, "target_jd", perf, warnings)
        perf["jd_resolve_s"] = round(time.perf_counter() - started, 3)
        return text

    perf["jd_resolve_s"] = round(time.perf_counter() - started, 3)
    return ""


async def _resolve_template_path(upload: Optional[UploadFile], warnings: list[dict[str, Any]]) -> str:
    if upload is None:
        return DEFAULT_TEMPLATE
    raw = await upload.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="cv_template file too large")
    filename = upload.filename or "template"
    ext = _file_ext(filename)
    if ext == ".docx":
        AVATAR_DIR.mkdir(parents=True, exist_ok=True)
        path = AVATAR_DIR / f"template_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.docx"
        path.write_bytes(raw)
        return str(path)
    if ext in {".pdf", *IMAGE_EXTENSIONS}:
        warnings.append(
            {
                "source": "cv_template",
                "filename": filename,
                "message": "PDF/图片模板当前仅参考版式偏好，已使用标准可编辑 DOCX 模板输出。",
            }
        )
        return DEFAULT_TEMPLATE
    warnings.append({"source": "cv_template", "filename": filename, "message": "不支持的模板格式，已使用标准模板。"})
    return DEFAULT_TEMPLATE

def _target_role_from_text(query: str, jd_text: str) -> str:
    """Legacy rule-based target role extraction. Kept for backward compatibility."""
    text = "\n".join([query or "", jd_text or ""])
    # Skip section headers like "岗位职责"
    non_role_headers = ("岗位职责", "职责", "任职要求", "任职资格", "岗位要求", "职位要求",
                        "工作职责", "工作内容", "要求", "资格", "职责描述", "岗位描述")
    match = re.search(r"(?:目标岗位|岗位|职位|想做|应聘|招聘)[:：\s]*([\u4e00-\u9fffA-Za-z0-9/ ]{2,30})", text)
    if match:
        role = match.group(1).strip("，。；; ")
        # Filter out section headers
        for header in non_role_headers:
            if role == header or role.endswith(header):
                role = role[:-len(header)].strip("：:，,。.、/\\")
            if role.startswith(header):
                role = role[len(header):].strip("：:，,。.、/\\")
        if len(role) >= 2:
            return role
    for role in ("产品经理", "运营", "医生", "教师", "老师", "销售", "售前", "金融风控", "设计师", "算法工程师", "软件工程师"):
        if role in text:
            return role
    return ""


def final_fact_guard(
    source_truth_text: str,
    resume_data: dict[str, Any],
    *,
    max_iterations: int = 1,
) -> tuple[dict[str, Any], FabricationReport]:
    """Check fabrication against source text; WARN only, do NOT delete fields.

    Previously this function deleted fabricated fields from resume_data and
    the hard-zero fabrication score wiped out the entire result. Now fabrication
    findings are reported as user_report entries but the data is preserved.
    Scoring uses graduated penalties instead of a binary pass/fail.
    """
    fab = check_fabrication_heuristic(source_truth_text, resume_data)
    return resume_data, fab


def _remove_fabricated_fields(
    resume_data: dict[str, Any],
    fab_report: FabricationReport,
    source_truth_text: str,
) -> dict[str, Any]:
    """Remove fields identified as fabricated from resume_data."""
    import copy
    data = copy.deepcopy(resume_data)

    for detail in fab_report.details:
        kind = detail.type
        content = detail.content

        if kind == "company":
            # Remove experience entries with fabricated company names
            experience = data.get("experience", [])
            if isinstance(experience, list):
                data["experience"] = [
                    exp for exp in experience
                    if not (isinstance(exp, dict) and str(exp.get("company", "")).strip() == content)
                ]

        elif kind == "school":
            # Remove education entries with fabricated school names
            education = data.get("education", [])
            if isinstance(education, list):
                data["education"] = [
                    edu for edu in education
                    if not (isinstance(edu, dict) and str(edu.get("school", "")).strip() == content)
                ]

        elif kind == "name":
            # Remove project entries with fabricated names
            projects = data.get("projects", [])
            if isinstance(projects, list):
                data["projects"] = [
                    proj for proj in projects
                    if not (isinstance(proj, dict) and str(proj.get("name", "")).strip() == content)
                ]

        elif kind == "role":
            # Clear fabricated roles but keep the experience entry
            experience = data.get("experience", [])
            if isinstance(experience, list):
                for exp in experience:
                    if isinstance(exp, dict) and str(exp.get("role", "")).strip() == content:
                        exp["role"] = ""

        elif kind == "skill":
            # Remove fabricated skills
            skills = data.get("skills", {})
            if isinstance(skills, dict):
                for bucket, values in list(skills.items()):
                    if isinstance(values, list):
                        skills[bucket] = [v for v in values if str(v).strip() != content]

        elif kind == "metric":
            # Actually remove the fabricated metric expression, don't leave [需补充]
            _remove_metric_from_data(data, content)

    # Clean projects without source support
    data = _clean_projects(data, source_truth_text)

    return data


def _remove_metric_from_data(data: dict[str, Any], metric_value: str) -> None:
    """Actually remove fabricated metric expressions from text fields in resume_data."""
    metric_pattern = re.compile(
        rf"{re.escape(metric_value)}\s*(?:%|万元|万|人|个|次|条|倍|客户|学生|病例|日活|月活|转化|CTR|GMV|QPS|TPS)",
        re.IGNORECASE,
    )
    for section in ("experience", "projects"):
        items = data.get(section, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("function_description", "result_description", "achievements", "bullets"):
                val = item.get(key)
                if isinstance(val, str):
                    item[key] = metric_pattern.sub("", val).strip("，, 。; ")
                elif isinstance(val, list):
                    item[key] = [metric_pattern.sub("", str(v)).strip("，, 。; ") if isinstance(v, str) else v for v in val]
                    # Remove empty bullets
                    item[key] = [b for b in item[key] if b and len(str(b).strip()) > 0]


def _clean_projects(resume_data: dict[str, Any], source_truth_text: str) -> dict[str, Any]:
    """Remove projects whose names can't be traced back to original input text.

    Allows project names derived from text phrases:
    - '课程选课系统项目' from '课程选课系统项目' → kept
    - '课程选课系统' from '做了课程选课系统' → kept
    - '负责需求分析和原型设计' → should NOT become '产品设计优化项目'

    Generic project names like 'XX优化项目' are always removed.
    """
    import copy
    data = copy.deepcopy(resume_data)
    projects = data.get("projects", [])
    if not isinstance(projects, list) or not projects:
        return data

    source_lower = source_truth_text.lower()

    # Generic project name patterns that indicate fabrication
    generic_project_patterns = [
        r".+优化项目$",
        r".+改进项目$",
        r".+升级项目$",
        r".+建设项目$",
        r"项目\d*$",
        r"未命名项目",
        r".+平台项目$",
    ]

    # Patterns that indicate a project name was derived from a responsibility statement
    # e.g. "负责需求分析和原型设计" -> "产品设计优化项目" (bad)
    responsibility_indicators = (
        "负责", "参与", "协助", "配合", "主要完成", "独立负责", "承担",
        "主要工作", "工作内容包括", "负责需求", "负责设计", "负责开发",
    )

    cleaned = []
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        name = str(proj.get("name", "")).strip()
        if not name:
            # Keep unnamed projects (they may be from experience split)
            cleaned.append(proj)
            continue

        # Skip generic project names
        is_generic = any(re.match(p, name) for p in generic_project_patterns)
        if is_generic:
            logger.info("Removing generic project name: %s", name)
            continue

        # Check if project name (or a significant substring) appears in source
        if name.lower() in source_lower:
            cleaned.append(proj)
            continue

        # Check if a core part of the name (without 项目/系统/平台 suffix) appears
        core_name = re.sub(r"(项目|系统|平台|工程|方案)$", "", name)
        if core_name and len(core_name) >= 2 and core_name.lower() in source_lower:
            cleaned.append(proj)
            continue

        # Check if the name is too long and likely fabricated from responsibility text
        # User descriptions are typically short; LLM hallucinations are verbose
        if len(name) > 10:
            logger.info("Removing long fabricated project name (10+ chars): %s", name)
            continue

        # Fabricated project name - remove
        logger.info("Removing fabricated project name not in source: %s", name)

    data["projects"] = cleaned
    return data


def _check_ocr_quality(text: str) -> Optional[dict[str, Any]]:
    """Check OCR text quality: Chinese char ratio, valid fields, noise ratio."""
    if not text or not text.strip():
        return {"acceptable": False, "reason": "空文本"}

    zh_pattern = re.compile(r"[\u4e00-\u9fff]")
    zh_count = len(zh_pattern.findall(text))
    total_chars = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
    zh_ratio = zh_count / max(total_chars, 1)

    # Check for valid resume fields
    valid_fields = sum(1 for kw in ("学校", "大学", "公司", "工作", "学历", "专业", "电话", "邮箱", "姓名", "经验") if kw in text)
    has_resume_structure = valid_fields >= 1

    # Check for noise ratio (repeated special chars, garbage)
    garbage_pattern = re.compile(r"[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\-./%]")
    garbage_chars = len(garbage_pattern.findall(text))
    noise_ratio = garbage_chars / max(total_chars, 1)

    # Heuristic score: 0-100
    score = 0

    # Chinese ratio (0-40)
    if zh_ratio >= 0.3:
        score += 40
    elif zh_ratio >= 0.15:
        score += 20
    if has_resume_structure:
        score += min(30, valid_fields * 8)
    # Noise ratio (0-30)
    if noise_ratio < 0.1:
        score += 30
    elif noise_ratio < 0.2:
        score += 15

    acceptable = score >= 50 and has_resume_structure
    return {"acceptable": acceptable, "score": score, "zh_ratio": round(zh_ratio, 3),
            "noise_ratio": round(noise_ratio, 3), "reason": "OCR质量过低" if not acceptable else ""}


def _build_user_report(
    *,
    missing_fields: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    fabrication: dict[str, Any],
    direction: str,
    ocr_warnings: list[dict[str, Any]],
    template_notes: list[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generation_direction": direction,
        "missing_field_suggestions": missing_fields,
        "conflict_confirmations": conflicts,
        "ocr_warnings": ocr_warnings,
        "template_notes": template_notes,
    }
    if fabrication.get("fabrication_found"):
        report["fabrication_details"] = fabrication.get("details", [])
    return report


def _build_llm_reply(
    *,
    audit_report: dict[str, Any],
    score: float,
    missing_fields: list[dict[str, Any]],
    changes: list[dict[str, Any]],
) -> str:
    """Generate reply_text via LLM. Returns empty string on failure."""
    if not llm_enabled():
        return ""
    try:
        summary_parts = []
        if isinstance(audit_report, dict):
            issues = audit_report.get("issues", [])
            if isinstance(issues, list) and issues:
                high = [i for i in issues if isinstance(i, dict) and i.get("severity") == "high"]
                medium = [i for i in issues if isinstance(i, dict) and i.get("severity") == "medium"]
                summary_parts.append(f"识别到 {len(high)} 项重点关注、{len(medium)} 项建议改进")
                for h in high[:3]:
                    p = h.get("problem", "")
                    if p:
                        summary_parts.append(f"重点关注：{p[:80]}")
        if missing_fields:
            reasons = "；".join(item.get("reason", "") for item in missing_fields[:3] if item.get("reason"))
            if reasons:
                summary_parts.append(f"需要补充的信息：{reasons}")
        if not missing_fields:
            summary_parts.append("提醒用户核对联系方式、教育时间等关键信息是否完整")
        substantive = [c for c in changes if isinstance(c, dict) and len(str(c.get("after", ""))) > len(str(c.get("before", ""))) * 1.2]
        if substantive:
            summary_parts.append(f"已完成 {len(substantive)} 处实质性改写")
        user_prompt = "请根据以下简历处理结果生成面向用户的自然语言回复（不要提及具体评分数值）：\n\n" + "\n".join(summary_parts)
        reply = call_llm_text(
            system_prompt=REPLY_GENERATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=512,
        )
        # Ensure missing field reminder is always present (concatenated if LLM missed it)
        if reply and missing_fields:
            _mf_reasons = "；".join(item.get("reason", "") for item in missing_fields[:3] if item.get("reason"))
            if _mf_reasons and _mf_reasons not in reply:
                reply += f"\n\n需要补充的信息：{_mf_reasons}"
        return reply if reply else ""
    except Exception as exc:
        logger.warning("LLM reply generation failed: %s", exc)
        return ""


def build_reply_text(
    *,
    scenario: str,
    industry: str,
    user_stage: str,
    missing_fields: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    ocr_warnings: list[dict[str, Any]],
    direction: str,
    score_total: float,
) -> str:
    scenario_label = {
        "scenario1": "原始简历与目标 JD 优化",
        "scenario2": "个人信息生成标准简历",
        "scenario3": "原始简历按目标岗位优化",
        "scenario4": "个人信息结合目标 JD 生成简历",
    }.get(scenario, "简历生成/优化")
    header = "已按"" + scenario_label + ""完成一版可编辑 DOCX，识别方向为" + INDUSTRY_LABELS.get(industry, "综合") + "，用户阶段为" + user_stage + "。"
    parts = [header, direction]
    # Missing fields: always remind user
    if missing_fields:
        reasons = "; ".join(item.get("reason", "") for item in missing_fields[:5] if item.get("reason"))
        if reasons:
            parts.append("需要补充: " + reasons)
    else:
        parts.append("请核对联系方式、教育时间、经历成果等关键信息是否完整。")
    if conflicts:
        reasons = "; ".join(item.get("description", "") for item in conflicts[:3] if item.get("description"))
        if reasons:
            parts.append("需要确认: " + reasons)
    if ocr_warnings:
        warnings = "; ".join(item.get("message", "") for item in ocr_warnings[:3] if item.get("message"))
        parts.append("OCR/文件提示: " + warnings)
    parts.append("建议优先补齐联系方式、教育时间、经历成果和可量化结果，再用于正式投递。")
    return "\n".join(part for part in parts if part)


def _repair_common_parse_errors(resume_data: dict[str, Any], raw_text: str) -> None:
    """Fix common LLM parse errors using LLM-based repair (replaces ~350 lines of regex rules).

    Merge strategy: only accept LLM improvements, never discard existing good data.
    """
    if not isinstance(resume_data, dict):
        return
    from resume_llm_repair import llm_repair_parse_errors
    try:
        repaired = llm_repair_parse_errors(resume_data, raw_text)
        if not repaired or not isinstance(repaired, dict):
            return
        # Merge education: only ADD items that fill gaps, don't discard existing
        for key in ("education", "projects", "publications"):
            existing = resume_data.get(key, [])
            llm_result = repaired.get(key)
            if not isinstance(llm_result, list) or not llm_result:
                continue
            # If LLM returned nothing useful, keep existing
            if not existing:
                resume_data[key] = llm_result
            elif len(llm_result) > 0 and all(isinstance(x, dict) and x.get("school") for x in llm_result):
                # LLM education is well-formed — use it
                resume_data[key] = llm_result
            # For projects/publications: prefer shorter list ONLY if LLM result has content.
            # Never replace a valid list with an empty one.
            if key in ("projects", "publications") and llm_result and len(llm_result) > 0 and len(llm_result) < len(existing):
                resume_data[key] = llm_result
        # Merge meta: only fill empty fields
        if "meta" in repaired and repaired["meta"]:
            for k, v in repaired["meta"].items():
                if v and not resume_data.get("meta", {}).get(k):
                    resume_data.setdefault("meta", {})[k] = v
    except Exception as exc:
        logger.warning("LLM repair failed (best-effort): %s", exc)

    # Post-repair cleanup: remove common field pollution patterns.
    # These are parse artifacts that LLM repair sometimes misses.
    if not isinstance(resume_data, dict):
        return

    # Publications: filter titles that are education rows or skill enumerations.
    # Accept both string pubs (LLM parse: "Title (venue)") and dict pubs ({"title":...}).
    pubs = resume_data.get("publications", [])
    if isinstance(pubs, list) and pubs:
        _pub_pollution = (
            r"(?:大学|University|学院).*(?:\d{4}|\d{2})",  # education row with year
            r"^(?:numpy|pandas|matplotlib|scikit).*",      # starts with framework name
            r"^(?:编程语言|框架工具|软件工具|语言水平)",         # section header
        )
        _kept = []
        for p in pubs:
            title = ""
            if isinstance(p, str):
                title = p
            elif isinstance(p, dict):
                title = str(p.get("title", "") or "")
            if not title:
                continue
            if not any(re.search(pat, title, re.IGNORECASE) for pat in _pub_pollution):
                # Normalize string pubs to dict format if needed
                if isinstance(p, str):
                    _kept.append({"title": p, "venue": "", "year": "", "authors": ""})
                else:
                    _kept.append(p)
        resume_data["publications"] = _kept

    # Honors: filter items that are experience bullets (>100 chars or start with verb)
    honors = resume_data.get("honors", [])
    if isinstance(honors, list) and honors:
        _honor_bullet_verbs = (
            "研究", "负责", "使用", "处理", "搭建", "建立", "实施",
            "采用", "复现", "测试", "设计", "构建", "集成", "执行",
            "implement", "design", "develop", "build", "test",
        )
        resume_data["honors"] = [
            h for h in honors
            if isinstance(h, str) and not (
                len(h) > 100 or
                any(h.strip().startswith(v) for v in _honor_bullet_verbs)
            )
        ]

    # Projects: remove empty-shell entries that duplicate experience content.
    # When parser creates projects from section headers but the actual content is
    # already in experience fields, the projects end up as empty titles — remove them.
    proj_list = resume_data.get("projects", [])
    if isinstance(proj_list, list) and proj_list:
        # Collect all text from experience bullets for overlap check
        exp_text = " ".join(
            str(b) for exp in resume_data.get("experience", []) or []
            if isinstance(exp, dict)
            for b in (exp.get("bullets", []) if isinstance(exp.get("bullets"), list) else [])
            if isinstance(b, str)
        ).lower()

        cleaned = []
        for p in proj_list:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name", "") or "").strip()
            blist = p.get("bullets", []) if isinstance(p.get("bullets"), list) else []
            has_content = blist and any(str(b).strip() for b in blist)
            has_desc = len(str(p.get("description", "") or "").strip()) > 20

            # Keep if it has real content
            if has_content or has_desc:
                cleaned.append(p)
                continue
            # Empty shell: check if this project's content is already in experience
            if not name:
                continue
            # Simple overlap: does any word from the project name appear in experience?
            name_words = re.findall(r"[一-鿿A-Za-z]{3,}", name)
            if name_words:
                matched = sum(1 for w in name_words if w.lower() in exp_text)
                # If >50% of project name words appear in experience, it's a duplicate
                if matched > len(name_words) * 0.5 and len(name_words) >= 2:
                    continue
            cleaned.append(p)
        resume_data["projects"] = cleaned

def generate_resume_with_llm_from_profile(
    *,
    query_text: str,
    jd_text: str,
    scenario: str,
    industry: str,
    target_role: str,
    user_stage: str,
) -> dict[str, Any]:
    """Generate resume data for scenarios without an original CV.

    The prompt intentionally separates user facts from JD requirements. JD text
    can shape wording and section priority, but must not become user facts.
    """
    if not llm_enabled():
        return {}
    system_prompt = (
        "你是一位多行业简历生成专家，负责把用户提供的个人事实整理为可投递简历 JSON。\n"
        "硬约束：\n"
        "1) 只能把【用户事实】写成候选人已有经历、学校、岗位、技能、证书、时间、数字结果。\n"
        "2) 【目标JD】只能用于确定表达方向、关键词侧重和个人总结，不得写成候选人已具备事实。\n"
        "3) 缺失的姓名、电话、邮箱、学校、学历、时间、岗位、成果、技能必须留空或保留原文弱表达，不得猜测。\n"
        "4) 原文没有数字结果时，不得编造百分比、金额、人数、病例数、学生数、客户数。\n"
        "5) 个人总结必须不超过100字，且要匹配目标行业。\n"
        "6) 输出必须是结构化简历 JSON，不要输出解释或 markdown。\n"
        "7) 输出语言必须与用户输入保持一致：用户用中文，则全部字段用中文输出。\n"
        "8) 不得使用知页、WonderCV 等模板网站的示例占位数据，必须使用描述中真实的用户信息。"
    )
    prompt = (
        f"【业务场景】{scenario}\n"
        f"【行业】{industry}\n"
        f"【用户阶段】{user_stage}\n"
        f"【目标岗位】{target_role or '未明确'}\n\n"
        "【用户事实】\n"
        f"{sanitize_user_text(query_text)}\n\n"
        "【目标JD，仅作方向，不得作为候选人事实】\n"
        f"{sanitize_user_text(jd_text or '')[:6000]}\n\n"
        "请生成字段：meta、summary、education、experience、projects、skills、personal_skills。"
        "经历按开始时间倒序；每段经历尽量拆出 responsibilities 与 achievements，证据不足时留空。"
    )
    try:
        parsed = call_llm_typed(
            StructuredResumeLLMOutput,
            system_prompt,
            prompt,
            temperature=0.2,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.warning("Scenario profile generation failed: %s", exc)
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def resume_copilot_service(
    *,
    query: Optional[str],
    cv: Optional[UploadFile],
    cv_template: Optional[UploadFile],
    target_jd: Optional[str],
    target_jd_file: Optional[UploadFile],
    target_jd_url: Optional[str],
    jd_text: Optional[str],
    jd_url: Optional[str],
    template: str = DEFAULT_TEMPLATE,
) -> ResumeCopilotResponse:
    """Two-strategy pipeline: rewrite-path (scenario 1/3) vs generate-path (scenario 2/4)."""
    from resume_copilot_pipeline import (
        stage_ingest, stage_classify, rewrite_path, generate_path,
        stage_score, stage_render,
    )

    ctx = await stage_ingest(
        query=query, cv=cv, cv_template=cv_template,
        target_jd=target_jd, jd_text=jd_text,
        target_jd_url=target_jd_url, jd_url=jd_url,
        target_jd_file=target_jd_file, template=template,
    )
    ctx = await stage_classify(ctx)

    # V2 pipeline flag
    _pipeline_version = os.environ.get("RESUME_PIPELINE_VERSION", "v1").strip()
    if _pipeline_version in ("v2", "shadow"):
        try:
            from v2_pipeline import run_v2_pipeline
            v2_result = run_v2_pipeline(
                cv_text=ctx.cv_text,
                query_text=ctx.query_text,
                jd_text=ctx.jd_text,
            )
            if _pipeline_version == "shadow":
                logger.info("SHADOW | V2 produced %d edu, %d exp",
                            len(v2_result.resume.education),
                            len(v2_result.resume.experience))
            else:
                ctx.resume_data = v2_result.resume_dict
                ctx.fabrication_report = None
                ctx.missing_fields = []
        except Exception as exc:
            logger.error("V2 pipeline failed: %s", exc)
            if _pipeline_version == "v2":
                if ctx.has_cv:
                    ctx = await rewrite_path(ctx)
                else:
                    ctx = await generate_path(ctx)

    if _pipeline_version == "v1" or (_pipeline_version == "v2" and not ctx.resume_data):
        if ctx.has_cv:
            ctx = await rewrite_path(ctx)
        else:
            ctx = await generate_path(ctx)

    ctx = await stage_score(ctx)
    ctx = await stage_render(ctx)

    return ResumeCopilotResponse(
        files=ctx.files,
        reply_text=ctx.reply_text,
        score=ctx.score,
        score_breakdown=ctx.score_breakdown,
        missing_fields=_dicts(ctx.missing_fields),
        conflicts=_dicts(ctx.conflicts),
        scenario=ctx.scenario,
        industry=ctx.industry,
        user_stage=ctx.user_stage,
        perf=ctx.perf,
        ocr_warnings=ctx.ocr_warnings,
        user_report=ctx.user_report,
        resume_data=ctx.resume_data,
        draft_id=ctx.draft_id,
        version=ctx.version,
    )

