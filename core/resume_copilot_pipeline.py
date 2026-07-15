"""Stage pipeline for resume-copilot: explicit phase functions with PipelineContext.

Breaks the monolithic resume_copilot_service() into 7 explicit stages.
Each stage reads from and writes to a PipelineContext dataclass.
No behavior change — pure code movement.
"""

from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from http_compat import HTTPException, UploadFile

try:
    import aiohttp
except ImportError:
    aiohttp = None

import resume_product_logic as product_logic
from audit_logic import audit_resume_core, extract_jd_keywords
from drafts import create_new_draft
from resume_classifier import classify_resume_request
from resume_generator import _build_generation_direction
from resume_io import IMAGE_EXTENSIONS, extract_text_from_bytes
from resume_optimization import (
    optimize_resume_core,
    patch_optimize_weak_bullets,
    apply_patches,
)
from resume_parsing import cleanup_ocr_text, _count_resume_bullets, resume_data_to_text, structured_resume_from_text
from resume_renderer import export_resume_files
from resume_scoring import score_resume
from resume_validator import (
    check_fabrication_heuristic,
    check_required_fields,
    check_sort_order,
    check_summary_jd_alignment,
    check_time_conflicts,
    FabricationReport,
)
from schemas import ResumeCopilotResponse
from server_runtime import (
    AVATAR_DIR,
    DEFAULT_TEMPLATE,
    DRAFTS_DIR,
    MAX_FILE_SIZE,
    OUTPUT_DIR,
    REQUEST_TIMEOUT_SECONDS,
    call_llm_text,
    call_llm_typed,
    llm_enabled,
    logger,
    sanitize_user_text,
)
from prompts import REPLY_GENERATION_SYSTEM_PROMPT
from resume_product_logic import INDUSTRY_LABELS

# ── Template watermark patterns ──
_TEMPLATE_WATERMARK_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(
        r"(abbey@wondercv\.com|job@weiapp\.com|job@wonder\.com)",
        re.IGNORECASE,
    ),
    "phone": re.compile(
        r"(13800000000|188-?8888-?8888|15900000000|"
        r"13600000000|010-?8888-?8888)",
    ),
    "company": re.compile(r"(超级公司|知页|WonderCV)"),
    "name": re.compile(r"知页\S+男|知页\S+女"),
}


def _clean_template_watermarks(resume_data: dict, query_text: str = "", has_cv: bool = False) -> list[str]:
    """Scan all fields in resume_data for known watermark/template placeholders
    and query leakage (copy-pasted query text in fields). Returns a list of warnings."""
    warnings: list[str] = []
    _query_lower = query_text.lower().strip() if query_text else ""


    def _is_query_leak(value: str, path: str) -> bool:
        if not has_cv:
            # generate_path: query IS the fact source — no field value
            # derived from query should be treated as leakage.
            return False
        if len(value) < 3:
            return False
        v = value.lower().strip()
        # Field value is a verbatim substring of query (no length limit)
        if v in _query_lower:
            return True
        return False

    def _scan_value(value: Any, path: str = "") -> Any:
        if isinstance(value, str):
            # Check template watermarks (only for generate_path — has_cv=False,
            # because has_cv=True means fields came from real CV, not LLM placeholder)
            if not has_cv:
                for name, pattern in _TEMPLATE_WATERMARK_PATTERNS.items():
                    if pattern.search(value):
                        warnings.append(f"已清除占位数据（{name}）位于 {path}")
                        return ""
            # Check query leakage (copy-pasted query text in fields)
            if _query_lower and _is_query_leak(value, path):
                warnings.append(f"已清除query泄漏位于 {path}")
                return ""
            return value
        if isinstance(value, dict):
            return {k: _scan_value(v, f"{path}.{k}") for k, v in value.items()}
        if isinstance(value, list):
            return [_scan_value(item, f"{path}[{i}]") for i, item in enumerate(value)]
        return value  # int/float/bool/None pass through unchanged

    cleaned = _scan_value(resume_data)
    resume_data.clear()
    resume_data.update(cleaned)
    return warnings



def _strip_placeholders(resume_data: dict) -> None:
    """Remove <某XX> placeholder patterns from enrichment output."""
    _pat = re.compile(r'某\S+(?:公司|学校|单位|企业|部门)')
    def _scan(v, path=""):
        if isinstance(v, str):
            if _pat.search(v):
                return ""
            return v
        if isinstance(v, dict):
            return {k: _scan(val, f"{path}.{k}") for k, val in v.items()}
        if isinstance(v, list):
            return [_scan(item, f"{path}[{i}]") for i, item in enumerate(v)]
        return v
    cleaned = _scan(resume_data)
    resume_data.clear()
    resume_data.update(cleaned)



@dataclass
class PipelineContext:
    """Data object that carries state through the 7 pipeline stages.

    Each field is written by exactly one stage. No stage mutates another's fields.
    """

    # Stage 0 (Ingest) output
    query_text: str = ""
    cv_text: str = ""
    jd_text: str = ""
    template_path: str = ""
    ocr_warnings: list = field(default_factory=list)
    template_notes: list = field(default_factory=list)
    perf: dict = field(default_factory=dict)
    started: float = 0.0

    # Stage 1 (Classify) output
    scenario: str = ""
    industry: str = ""
    user_stage: str = ""
    target_role: str = ""

    # Stage 2 (Parse) output
    generation_text: str = ""
    source_truth_text: str = ""
    ocr_quality: Optional[dict] = None
    _low_ocr_quality: bool = False
    has_cv: bool = False
    cv_uploaded: bool = False
    cv_extraction_failed: bool = False
    has_jd: bool = False
    resume_data: dict = field(default_factory=dict)

    # Stage 3 (Enrich) output
    audit_report: dict = field(default_factory=dict)
    _has_audit: bool = False
    patches: list = field(default_factory=list)
    changes: list = field(default_factory=list)

    # Stage 4 (Validate) output
    fabrication_report: Any = None
    missing_fields: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)

    # Stage 5 (Score) output
    user_report: dict = field(default_factory=dict)
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)

    # Stage 6 (Render) output
    files: dict = field(default_factory=dict)
    reply_text: str = ""
    draft_id: str = ""
    version: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers (moved from resume_copilot_service.py)
# ═══════════════════════════════════════════════════════════════════════════════

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


def _check_ocr_quality(text: str) -> Optional[dict[str, Any]]:
    """Check OCR text quality: Chinese char ratio, valid fields, noise ratio."""
    if not text or not text.strip():
        return {"acceptable": False, "reason": "空文本"}

    zh_pattern = re.compile(r"[一-鿿]")
    zh_count = len(zh_pattern.findall(text))
    total_chars = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
    zh_ratio = zh_count / max(total_chars, 1)

    valid_fields = sum(1 for kw in ("学校", "大学", "公司", "工作", "学历", "专业", "电话", "邮箱", "姓名", "经验") if kw in text)
    has_resume_structure = valid_fields >= 1

    garbage_pattern = re.compile(r"[^\w\s一-鿿　-〿＀-￯\-./%]")
    garbage_chars = len(garbage_pattern.findall(text))
    noise_ratio = garbage_chars / max(total_chars, 1)

    score = 0
    if zh_ratio >= 0.3:
        score += 40
    elif zh_ratio >= 0.15:
        score += 20
    if has_resume_structure:
        score += min(30, valid_fields * 8)
    if noise_ratio < 0.1:
        score += 30
    elif noise_ratio < 0.2:
        score += 15

    acceptable = score >= 50 and has_resume_structure
    return {"acceptable": acceptable, "score": score, "zh_ratio": round(zh_ratio, 3),
            "noise_ratio": round(noise_ratio, 3), "reason": "OCR质量过低" if not acceptable else ""}


async def _extract_upload_text(upload: UploadFile, purpose: str, perf: dict[str, float], warnings: list[dict[str, Any]]) -> str:
    started = time.perf_counter()
    raw = await upload.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"{purpose} file too large (> {MAX_FILE_SIZE // (1024 * 1024)} MB)")
    filename = upload.filename or f"{purpose}.bin"
    ext = _file_ext(filename)
    if ext in IMAGE_EXTENSIONS:
        warnings.append({
            "source": purpose, "filename": filename,
            "message": "已对图片执行本地 OCR；如识别不完整，请补充清晰图片或文本。",
        })
    try:
        text = extract_text_from_bytes(raw, filename)
    except HTTPException as exc:
        if ext in IMAGE_EXTENSIONS:
            warnings.append({
                "source": purpose, "filename": filename,
                "message": "图片内容无法可靠识别，请补充文本或重新上传清晰图片。",
            })
            perf[f"{purpose}_extract_s"] = round(time.perf_counter() - started, 3)
            return ""
        raise exc
    perf[f"{purpose}_extract_s"] = round(time.perf_counter() - started, 3)
    if ext in IMAGE_EXTENSIONS and len(text.strip()) < 30:
        warnings.append({
            "source": purpose, "filename": filename,
            "message": "图片 OCR 文本较短，可能存在识别缺失，请确认关键信息。",
        })
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
    *, target_jd, jd_text, target_jd_url, jd_url, target_jd_file, perf, warnings
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


async def _resolve_template_path(upload: Optional[UploadFile], warnings: list[dict[str, Any]], template: str = DEFAULT_TEMPLATE) -> str:
    if upload is None:
        return template
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
        warnings.append({
            "source": "cv_template", "filename": filename,
            "message": "PDF/图片模板当前仅参考版式偏好，已使用标准可编辑 DOCX 模板输出。",
        })
        return template
    warnings.append({"source": "cv_template", "filename": filename, "message": "不支持的模板格式，已使用标准模板。"})
    return template


def _build_user_report(
    *, missing_fields, conflicts, fabrication, direction, ocr_warnings, template_notes
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
    *, audit_report, score, missing_fields, changes
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
            temperature=0.3, max_tokens=512,
        )
        if reply and missing_fields:
            _mf_reasons = "；".join(item.get("reason", "") for item in missing_fields[:3] if item.get("reason"))
            if _mf_reasons and _mf_reasons not in reply:
                reply += f"\n\n需要补充的信息：{_mf_reasons}"
        if reply:
            logger.info("回复信息: %s", reply)
        return reply if reply else ""
    except Exception as exc:
        logger.warning("LLM reply generation failed: %s", exc)
        return ""


def build_reply_text(
    *, scenario, industry, user_stage, missing_fields, conflicts,
    ocr_warnings, direction, score_total
) -> str:
    scenario_label = {
        "scenario1": "原始简历与目标 JD 优化",
        "scenario2": "个人信息生成标准简历",
        "scenario3": "原始简历按目标岗位优化",
        "scenario4": "个人信息结合目标 JD 生成简历",
    }.get(scenario, "简历生成/优化")
    header = "已按\"" + scenario_label + "\"完成一版可编辑 DOCX，识别方向为" + INDUSTRY_LABELS.get(industry, "综合") + "，用户阶段为" + user_stage + "。"
    parts = [header, direction]
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


def final_fact_guard(
    source_truth_text: str,
    resume_data: dict[str, Any],
    has_cv: bool = True,
    *,
    max_iterations: int = 1,
    ledger: Any = None,
) -> tuple[dict[str, Any], FabricationReport]:
    """Fabrication check with entity-level field validation.

    Guard condition: skips only when source_truth_text is truly empty.
    ``has_cv`` is no longer the guard — user query is also a valid fact
    source (e.g. generate_path with query text).

    When a FactLedger is provided, ``company`` and ``role`` fields are
    validated against ledger entities — unsupported values are cleared.
    For ``company``, if the full value is unsupported but a known entity
    (school/company) from the ledger is a substring of it, that entity
    is used as a safe fallback.
    """
    if not source_truth_text.strip():
        return resume_data, FabricationReport(fabrication_found=False, details=[])
    fab = check_fabrication_heuristic(source_truth_text, resume_data)

    # Entity-level validation with ledger
    if ledger is not None:
        _validate_experience_entities(resume_data, ledger, fab)
        _validate_role_entities(resume_data, ledger, fab)

    return resume_data, fab


def _validate_experience_entities(
    resume_data: dict, ledger: Any, fab: FabricationReport,
) -> None:
    """Validate experience[].company against ledger entities.

    If the full company value has no ledger support, attempt to fallback
    to a known entity (school/organization) that IS in the ledger and is
    contained within the company value.  This handles cases like
    '北京邮电大学实验室' → fallback to '北京邮电大学' (known school).
    """
    from schemas import FabricationDetail

    ledger_org_values: set[str] = set()
    for (kind, val_lower), entity in ledger.entities.items():
        if kind in ("company", "school", "organization"):
            ledger_org_values.add(entity.value)

    for exp in resume_data.get("experience", []):
        if not isinstance(exp, dict):
            continue
        company = str(exp.get("company", "")).strip()
        if not company:
            continue
        company_lower = company.lower()

        # Check if company is directly supported
        if any(entity.value.lower() == company_lower for entity in ledger.entities.values()):
            continue  # fully supported, keep

        # Check if company value appears as contiguous substring in raw_text
        # (already done by check_fabrication_heuristic, but double-check)
        if company_lower in ledger.raw_text.lower():
            continue  # supported as-is

        # Not fully supported — try fallback to a known entity
        best_fallback = ""
        for org_val in sorted(ledger_org_values, key=len, reverse=True):
            if org_val.lower() in company_lower:
                best_fallback = org_val
                break

        if best_fallback:
            exp["company"] = best_fallback
            fab.details.append(FabricationDetail(
                type="company",
                content=company,
                reason=f"组织名'{company}'部分不受支持，已回退到已知实体'{best_fallback}'",
            ))
        else:
            exp["company"] = ""
            fab.details.append(FabricationDetail(
                type="company",
                content=company,
                reason="该公司名未在用户原始输入中出现",
            ))


def _validate_role_entities(
    resume_data: dict, ledger: Any, fab: FabricationReport,
) -> None:
    """Clear experience[].role if it has no support in the FactLedger.

    Role is a strict field — it must have direct or normalized evidence
    in the source text.  If not, it is cleared (not guessed).
    """
    from schemas import FabricationDetail

    ledger_role_values: set[str] = set()
    for (kind, val_lower), entity in ledger.entities.items():
        if kind == "role":
            ledger_role_values.add(entity.value.lower())

    for exp in resume_data.get("experience", []):
        if not isinstance(exp, dict):
            continue
        role = str(exp.get("role", "")).strip()
        if not role:
            continue
        role_lower = role.lower()

        # Direct substring match in raw_text
        if role_lower in ledger.raw_text.lower():
            continue

        # Also check ledger entities
        if role_lower in ledger_role_values:
            continue

        # Unsupported — clear
        exp["role"] = ""
        fab.details.append(FabricationDetail(
            type="role",
            content=role,
            reason="该岗位名称未出现在用户原始输入中",
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Stages
# ═══════════════════════════════════════════════════════════════════════════════

async def stage_ingest(
    query: Optional[str],
    cv: Optional[UploadFile],
    cv_template: Optional[UploadFile],
    target_jd: Optional[str],
    jd_text: Optional[str],
    target_jd_url: Optional[str],
    jd_url: Optional[str],
    target_jd_file: Optional[UploadFile],
    template: str = DEFAULT_TEMPLATE,
) -> PipelineContext:
    """Stage 0: Extract CV text, JD text, template path. Quality-gate OCR."""
    ctx = PipelineContext()
    ctx.started = time.perf_counter()
    ctx.query_text = str(query or "").strip()

    # CV extraction + upload tracking
    ctx.cv_text = ""
    ctx.cv_uploaded = cv is not None
    if cv is not None:
        ctx.cv_text = await _extract_upload_text(cv, "cv", ctx.perf, ctx.ocr_warnings)

    # OCR quality gate
    ctx._low_ocr_quality = False
    ctx.cv_extraction_failed = False
    ctx.ocr_quality = None
    if ctx.cv_text:
        ctx.ocr_quality = _check_ocr_quality(ctx.cv_text)
        if ctx.ocr_quality is not None and not ctx.ocr_quality.get("acceptable"):
            ctx._low_ocr_quality = True
            ctx.ocr_warnings.append({
                "source": "pdf_ocr_quality",
                "message": f"OCR质量过低（{ctx.ocr_quality.get('score', 0)}分），已跳过CV优化链路",
                "quality_score": ctx.ocr_quality.get("score"),
            })
    elif ctx.cv_uploaded:
        # User uploaded a CV file but extraction returned nothing
        ctx.cv_extraction_failed = True
    ctx.has_cv = bool(ctx.cv_text.strip())

    # JD resolution
    ctx.jd_text = await _resolve_jd_text(
        target_jd=target_jd, jd_text=jd_text,
        target_jd_url=target_jd_url, jd_url=jd_url,
        target_jd_file=target_jd_file,
        perf=ctx.perf, warnings=ctx.ocr_warnings,
    )
    _ensure_time_budget(ctx.started, "input_resolve")
    ctx.has_jd = bool(ctx.jd_text.strip())

    # Diagnostic log
    cv_text_len = len(ctx.cv_text.strip()) if ctx.cv_text else 0
    logger.info(
        "CV\xe2\x90\xa4\xe2\x90\xa4extraction summary | has_cv=%s | cv_text_chars=%d | ocr_quality=%s | _low_ocr_quality=%s | query_chars=%d | has_jd=%s",
        ctx.has_cv, cv_text_len,
        ctx.ocr_quality.get("score") if ctx.ocr_quality else "N/A",
        ctx._low_ocr_quality, len(ctx.query_text), ctx.has_jd,
    )
    if ctx.cv_text and ctx.cv_text.strip():
        logger.info("OCR结果: %s", ctx.cv_text.strip())

    # Template
    effective_template = await _resolve_template_path(cv_template, ctx.ocr_warnings, template)
    if effective_template == DEFAULT_TEMPLATE:
        effective_template = template or DEFAULT_TEMPLATE
    if cv_template is not None and effective_template == (template or DEFAULT_TEMPLATE):
        ctx.template_notes.append("用户模板未能完整复刻，已使用标准 DOCX 模板输出。")
    ctx.template_path = effective_template

    return ctx


async def stage_classify(ctx: PipelineContext) -> PipelineContext:
    """Stage 1: Classify scenario, industry, user_stage, target_role."""
    ctx.scenario = product_logic.detect_scenario(
        has_cv=ctx.has_cv, has_jd=ctx.has_jd, query=ctx.query_text,
    )

    # source_truth_text (CV-only for fact checking) vs generation_text (for LLM parsing).
    # Query must NOT be in source_truth_text — it's commentary, not verified facts.
    # Query must NOT be in generation_text when CV is present — LLM would parse it as resume.
    if ctx._low_ocr_quality and ctx.has_cv:
        ctx.ocr_warnings.append({
            "source": "pdf_ocr_quality",
            "message": "低质量OCR文本已排除在 source_truth_text 外，仅基于 query 生成弱简历",
            "quality_score": 0,
        })
        ctx.source_truth_text = ctx.cv_text
        ctx.generation_text = ctx.query_text if ctx.query_text.strip() else ctx.cv_text
    elif ctx.has_cv:
        ctx.source_truth_text = ctx.cv_text
        ctx.generation_text = ctx.cv_text
    else:
        ctx.source_truth_text = ctx.query_text
        ctx.generation_text = ctx.query_text

    if not ctx.generation_text.strip():
        raise HTTPException(status_code=400, detail="query or cv is required")

    t_classify = time.perf_counter()
    classification = classify_resume_request(
        query=ctx.query_text, cv_text=ctx.cv_text, jd_text=ctx.jd_text,
        has_cv=ctx.has_cv, has_jd=ctx.has_jd,
        resume_data=None,
    )
    ctx.target_role = classification.target_role
    ctx.industry = classification.industry
    ctx.user_stage = classification.user_stage
    ctx.perf["classify_s"] = round(time.perf_counter() - t_classify, 3)

    return ctx



async def rewrite_path(ctx: PipelineContext) -> PipelineContext:
    """scenario 1/3: has CV -> parse + FactLedger + audit + patch_optimize + validate.

    Requires: ctx.has_cv=True
    """
    # ── Parse ──
    # OCR cleanup
    _ocr_score = ctx.ocr_quality.get("score", 0) if ctx.ocr_quality else 0
    _ocr_acceptable = ctx.ocr_quality.get("acceptable", False) if ctx.ocr_quality else False
    _skip_ocr_cleanup = ctx.has_cv and _ocr_acceptable and _ocr_score >= 80
    if ctx.has_cv and llm_enabled() and not ctx._low_ocr_quality and not _skip_ocr_cleanup and len(ctx.generation_text.strip()) >= 100:
        t_ocr_cleanup = time.perf_counter()
        try:
            cleaned = cleanup_ocr_text(ctx.generation_text)
            _clean_ok = cleaned and len(cleaned.strip()) >= len(ctx.generation_text.strip()) * 0.5
            if _clean_ok and len(cleaned) > len(ctx.generation_text.strip()) * 1.2:
                _clean_ok = False
                logger.warning("OCR cleanup output %d chars > 1.2x input %d chars, discarding", len(cleaned), len(ctx.generation_text.strip()))
            if _clean_ok and cleaned:
                _think_markers = ("Thinking Process:", "**Analyze the Request**", "**Analyze the Input**")
                if any(m in cleaned for m in _think_markers):
                    _clean_ok = False
                    logger.warning("OCR cleanup output contains thinking artifacts, discarding")
            if _clean_ok and cleaned:
                ctx.generation_text = cleaned
        except Exception as exc:
            logger.warning("OCR cleanup failed, using raw text: %s", exc)
        ctx.perf["ocr_cleanup_s"] = round(time.perf_counter() - t_ocr_cleanup, 3)
    elif _skip_ocr_cleanup:
        logger.info("OCR quality score=%d, skipping LLM cleanup", _ocr_score)

    # Structured parse from CV text
    t_parse = time.perf_counter()
    resume_data: dict[str, Any] = {}
    try:
        resume_data = structured_resume_from_text(ctx.generation_text)
    except Exception as exc:
        logger.warning("Structured parse failed in resume-copilot: %s", exc)
        resume_data = {}
    if not resume_data:
        resume_data = product_logic.heuristic_resume_from_text(
            ctx.generation_text, ctx.industry, ctx.target_role,
        )
    ctx.perf["structured_resume_s"] = round(time.perf_counter() - t_parse, 3)
    _ensure_time_budget(ctx.started, "structured_resume")

    resume_data = product_logic.normalize_resume_data_for_product(
        resume_data, raw_text=ctx.generation_text,
        industry=ctx.industry, target_role=ctx.target_role,
    )

    # Repair + user_stage fix
    from resume_copilot_service import _repair_common_parse_errors
    _repair_common_parse_errors(resume_data, ctx.generation_text)

    experience = resume_data.get("experience", [])
    education = resume_data.get("education", [])
    if isinstance(experience, list) and experience:
        work_years = product_logic.calculate_experience_years(experience)
        _has_student_roles = all(
            isinstance(e, dict) and (str(e.get("role", "") or "").strip().lower() in {
                "科研实习", "研究助理", "实习分析师", "实习审计师", "实习",
                "intern", "research assistant", "research intern",
            } or "实习" in str(e.get("role", "") or "") or "intern" in str(e.get("role", "") or "").lower())
            for e in experience
        )
        _still_studying = any(
            isinstance(e, dict) and (str(e.get("end_date", "") or "").strip() in {"至今", ""})
            for e in (education if isinstance(education, list) else [])
        )
        if not _still_studying:
            _now = datetime.now()
            _now_ym = (_now.year, _now.month)
            for e in (education if isinstance(education, list) else []):
                if not isinstance(e, dict): continue
                ed = str(e.get("end_date", "") or "").strip()
                m = re.match(r"(\d{2})-(\d{4})", ed)
                if m:
                    ey, em = int(m.group(2)), int(m.group(1))
                    if ey > _now_ym[0] or (ey == _now_ym[0] and em > _now_ym[1]):
                        _still_studying = True
                        break
        _is_intern_only = product_logic._is_student_with_internals(experience) if product_logic else False

        if ctx.user_stage == "student":
            if work_years >= 3 or (work_years >= 2 and not _is_intern_only):
                ctx.user_stage = "experienced"
        elif ctx.user_stage == "experienced" and _still_studying and _has_student_roles:
            ctx.user_stage = "student"
    ctx.perf["_classify_fix_user_stage_s"] = round(time.perf_counter() - t_parse, 3)

    ctx.resume_data = resume_data

    # ── Enrich (audit + FactLedger + patch_optimize) ──
    ctx.changes = []
    ctx.audit_report = {}
    ctx._has_audit = False

    _skip_optimize = ctx._low_ocr_quality and ctx.has_cv
    _enter_optimize = ctx.has_cv and llm_enabled() and not _skip_optimize
    logger.info(
        "optimize gate | entering=%s | has_cv=%s | llm_enabled=%s | _skip_optimize=%s | _low_ocr_quality=%s | scenario=%s",
        _enter_optimize, ctx.has_cv, llm_enabled(), _skip_optimize, ctx._low_ocr_quality, ctx.scenario,
    )

    if _enter_optimize:
        t_opt = time.perf_counter()
        try:
            ctx.audit_report = audit_resume_core(
                resume_data_to_text(ctx.resume_data),
                ctx.jd_text or ctx.target_role or ctx.query_text,
                resume_data=ctx.resume_data,
                source_text=ctx.generation_text,
            )
            ctx._has_audit = True

            _actionable_issues, _user_input_issues, _unclassified = [], [], []
            for issue in ctx.audit_report.get("issues", []) or []:
                if not isinstance(issue, dict): continue
                itype = issue.get("issue_type", "")
                if itype == "needs_data": _user_input_issues.append(issue)
                elif itype == "actionable": _actionable_issues.append(issue)
                else: _unclassified.append(issue)
            if _unclassified and llm_enabled():
                from resume_llm_repair import classify_audit_issues
                _labels = classify_audit_issues(_unclassified, ctx.generation_text)
                for i, issue in enumerate(_unclassified):
                    label = _labels[i] if i < len(_labels) else "actionable"
                    if label == "needs_data": _user_input_issues.append(issue)
                    else: _actionable_issues.append(issue)
            logger.info("Audit issue split: %d actionable + %d needs_user_input",
                       len(_actionable_issues), len(_user_input_issues))

            from fact_ledger import build_ledger
            ledger = build_ledger(ctx.resume_data, ctx.generation_text, run_repair=True)
            _jd_keywords = extract_jd_keywords(ctx.jd_text or ctx.target_role or ctx.query_text or "")
            patches = await patch_optimize_weak_bullets(ledger, _jd_keywords, n=3, resume_data=ctx.resume_data)

            if patches:
                ctx.resume_data = apply_patches(ctx.resume_data, patches)
                _bullet_map = {b.id: b for b in ledger.bullets}
                ctx.changes = []
                for p in patches:
                    _b = _bullet_map.get(p.bullet_id)
                    ctx.changes.append({
                        "project": p.bullet_id, "bullet_index": 0,
                        "before": _b.source_text if _b else "",
                        "after": p.new_text[:120],
                        "reason": f"表达优化 (置信度={p.confidence:.2f})",
                    })
                # Rebuild ledger to sync bullet texts with patched resume_data
                # (ledger is otherwise stale — its bullets still have original text)
                ledger = build_ledger(ctx.resume_data, ctx.source_truth_text, run_repair=False)
                logger.info("Applied %d bullet patches to resume_data", len(patches))
                for p_i, p in enumerate(patches[:5]):
                    _b = _bullet_map.get(p.bullet_id)
                    _before = _b.source_text[:80] if _b else "?"
                    logger.info("  patch[%d] %s: '%s' -> '%s'", p_i, p.bullet_id, _before, p.new_text[:80])
            else:
                logger.info("No patches applied, keeping original resume_data")

            ctx.audit_report["_issue_split"] = {
                "actionable_count": len(_actionable_issues),
                "user_input_count": len(_user_input_issues),
                "user_input_summary": [i.get("problem", "")[:100] for i in _user_input_issues[:3]],
            }
        except Exception as exc:
            logger.warning("Optimization skipped in resume-copilot: %s", exc)
        ctx.perf["optimize_s"] = round(time.perf_counter() - t_opt, 3)
        _ensure_time_budget(ctx.started, "optimize")
    elif _skip_optimize:
        ctx.ocr_warnings.append({
            "source": "pdf_ocr_quality",
            "message": "低质量OCR已跳过优化，仅基于 query 生成弱简历",
        })

    # ── Validate ──
    t_validate = time.perf_counter()

    ctx.resume_data, ctx.fabrication_report = final_fact_guard(
        ctx.source_truth_text, ctx.resume_data, has_cv=True,
    )
    if ctx.fabrication_report.fabrication_found:
        logger.info("事实核查: 发现 %d 项编造内容", len(ctx.fabrication_report.details))
        for _fab_item in ctx.fabrication_report.details[:5]:
            logger.info("  编造 %s='%s' (原文无此内容)", getattr(_fab_item, 'type', '?'), getattr(_fab_item, 'content', '?')[:60])
    else:
        logger.info("事实核查: 通过")

    from resume_llm_repair import llm_check_fabrication
    _fab_llm = llm_check_fabrication(ctx.source_truth_text, ctx.resume_data)
    if _fab_llm.get("fabrication_found"):
        _llm_details = _fab_llm.get("details", [])
        if _llm_details:
            logger.info("LLM fabrication hints (%d items, not applied to scoring)", len(_llm_details))
            for _hint in _llm_details[:5]:
                logger.info("  hint: %s", str(_hint)[:100])

    ctx.missing_fields = check_required_fields(ctx.resume_data, user_stage=ctx.user_stage, source_text=ctx.source_truth_text)
    ctx.conflicts = check_time_conflicts(ctx.resume_data)
    ctx.conflicts.extend(check_sort_order(ctx.resume_data))
    ctx.conflicts.extend(check_summary_jd_alignment(
        str(ctx.resume_data.get("summary", "")), ctx.jd_text or ctx.target_role
    ))

    if ctx.conflicts and llm_enabled():
        from resume_llm_repair import llm_resolve_conflicts
        try:
            _resolved = llm_resolve_conflicts(_dicts(ctx.conflicts), ctx.resume_data)
            if _resolved is not None and isinstance(_resolved, list):
                _kept = {r.get("description", "") for r in _resolved if isinstance(r, dict)}
                if _kept:
                    _before = len(ctx.conflicts)
                    ctx.conflicts = [c for c in ctx.conflicts if c.description in _kept]
                    logger.info("LLM resolved %d false-positive conflicts", _before - len(ctx.conflicts))
        except Exception as exc:
            logger.warning("LLM conflict resolution failed (rewrite_path): %s", exc)

    if ctx.missing_fields and llm_enabled():
        from resume_llm_repair import llm_enhance_missing_fields
        try:
            _missing_dicts = _dicts(ctx.missing_fields)
            _still_missing, _found_from_text = llm_enhance_missing_fields(
                _missing_dicts, ctx.resume_data, ctx.source_truth_text,
            )
            if _found_from_text:
                logger.info("LLM recovered %d missing fields from source text", len(_found_from_text))
                for item in _found_from_text:
                    field = item.get("field", "")
                    value = item.get("extracted_value", "")
                    if field.startswith("meta.") and value:
                        key = field.split(".", 1)[1]
                        ctx.resume_data.setdefault("meta", {})[key] = value
            if _still_missing and isinstance(_still_missing, list) and len(_still_missing) < len(ctx.missing_fields):
                _still_fields = {m.get("field", "") for m in _still_missing if isinstance(m, dict)}
                ctx.missing_fields = [m for m in ctx.missing_fields if hasattr(m, "field") and m.field in _still_fields]
        except Exception as exc:
            logger.warning("LLM missing-field enhancement failed (rewrite_path): %s", exc)

    ctx.perf["validation_s"] = round(time.perf_counter() - t_validate, 3)
    _ensure_time_budget(ctx.started, "validation")
    return ctx


def _profile_output_too_short(resume_data: dict) -> bool:
    """Check if LLM profile output is too sparse to use.

    Returns True when the output has fewer than 3 total bullet points
    or the serialized JSON is shorter than 500 chars — both indicate
    the LLM was too conservative and didn't extract enough from the query.
    """
    try:
        total_bullets = _count_resume_bullets(resume_data)
        if total_bullets < 3:
            return True
        text_len = len(json.dumps(resume_data, ensure_ascii=False))
        return text_len < 500
    except Exception:
        return True  # 无法分析时保守处理：当作太短


async def generate_path(ctx: PipelineContext) -> PipelineContext:
    """scenario 2/4: no CV -> profile generation + lite validation.

    Requires: ctx.has_cv=False
    """
    # ── Generate from profile ──
    t_parse = time.perf_counter()
    resume_data: dict[str, Any] = {}
    if llm_enabled():
        from resume_copilot_service import generate_resume_with_llm_from_profile
        resume_data = generate_resume_with_llm_from_profile(
            query_text=ctx.query_text, jd_text=ctx.jd_text,
            scenario=ctx.scenario, industry=ctx.industry,
            target_role=ctx.target_role, user_stage=ctx.user_stage,
        )
    if resume_data:
        try:
            _dbg_chars = len(json.dumps(resume_data, ensure_ascii=False, default=str))
            _dbg_blt = _count_resume_bullets(resume_data)
            logger.info("generate_path profile: %d chars, %d bullets, keys=%s",
                        _dbg_chars, _dbg_blt, list(resume_data.keys()))
        except Exception as exc:
            logger.info("generate_path profile debug failed: %s", exc)
    try:
        _layer2_needed = not resume_data or _profile_output_too_short(resume_data)
    except Exception:
        _layer2_needed = True
    if _layer2_needed:
        # 第2层: LLM把query文本当原始简历内容解析
        _elapsed = time.perf_counter() - ctx.started
        _remaining = REQUEST_TIMEOUT_SECONDS - _elapsed
        if _remaining < 10:
            logger.info("Skipping structured_resume_from_text fallback (only %.1fs remaining)", _remaining)
            resume_data = {}
        else:
            try:
                resume_data = structured_resume_from_text(ctx.generation_text)
            except Exception as exc:
                logger.warning("structured_resume_from_text fallback failed: %s", exc)
                resume_data = {}
    try:
        _layer3_needed = not resume_data or _profile_output_too_short(resume_data)
    except Exception:
        _layer3_needed = True
    if _layer3_needed:
        # 第3层: 规则解析（兜底）
        resume_data = product_logic.heuristic_resume_from_text(
            ctx.generation_text, ctx.industry, ctx.target_role,
        )
    ctx.perf["structured_resume_s"] = round(time.perf_counter() - t_parse, 3)
    _ensure_time_budget(ctx.started, "structured_resume")

    resume_data = product_logic.normalize_resume_data_for_product(
        resume_data, raw_text=ctx.generation_text,
        industry=ctx.industry, target_role=ctx.target_role,
    )
    ctx.resume_data = resume_data
    ctx.audit_report = {}
    ctx.changes = []

    # ── Enrich sparse profile for generate_path (no CV) ──
    if not ctx.has_cv and llm_enabled() and ctx.query_text:
        _blt = _count_resume_bullets(resume_data)
        _txt = len(json.dumps(resume_data, ensure_ascii=False, default=str))
        if _blt < 3 or _txt < 600:
            logger.info("generate_path enrichment needed: %d bullets, %d chars", _blt, _txt)
            try:
                _sparse = json.dumps(resume_data, ensure_ascii=False, default=str)
                _sys = (
                    '你是一名简历内容充实专家。现有简历JSON的experience和projects字段为空或内容很少，'
                    '请根据用户描述充实这些字段。\n\n'
                    '关键要求：\n'
                    '- 必须把用户描述中的信息写成experience[].bullets数组（每条bullet是一句话）\n'
                    '- 每个experience条目至少写2-4条bullets\n'
                    '- 必须重写summary字段为自然的一两句话描述，不要使用目标投递等模板化表达\n'
                    '- 公司/学校名称如果不确定，请留空不要编造，也不要使用任何占位词\n'
                    '- 禁止编造任何数字结果，包括百分比、金额、人数、增速、准确率、时长等\n'
                    '- 禁止编造姓名、电话、邮箱\n'
                    '- 输出完整JSON，结构不变'
                )
                _usr = (
                    f'用户描述：\n{ctx.query_text}\n\n'
                    f'当前简历JSON：\n{_sparse}\n\n'
                    '请输出充实后的简历JSON（特别注意填充experience[].bullets）：'
                )
                _enriched = call_llm_text(system_prompt=_sys, user_prompt=_usr, temperature=0.3, max_tokens=4096)
                if _enriched:
                    _c = _enriched.strip()
                    if '```json' in _c: _c = _c.split('```json')[1].split('```')[0].strip()
                    elif '```' in _c: _c = _c.split('```')[1].split('```')[0].strip()
                    _p = json.loads(_c)
                    _strip_placeholders(_p)
                    if isinstance(_p, dict) and _p:
                        _nb = _count_resume_bullets(_p)
                        _nt = len(json.dumps(_p, ensure_ascii=False, default=str))
                        logger.info("generate_path enrichment: %d->%d bullets, %d->%d chars", _blt, _nb, _txt, _nt)
                        if _nb > _blt or _nt > _txt:
                            # Replace resume_data and re-normalize
                            resume_data.clear()
                            resume_data.update(_p)
                            resume_data = product_logic.normalize_resume_data_for_product(
                                resume_data, raw_text=ctx.generation_text,
                                industry=ctx.industry, target_role=ctx.target_role,
                            )
                            ctx.resume_data = resume_data
            except Exception as exc:
                logger.warning("generate_path enrichment failed: %s", exc)

    # ── Validate (with source-truth-based fact check) ──
    t_validate = time.perf_counter()

    # Build FactLedger from user-provided source_truth_text (query),
    # NOT from generated LLM output.  This gives us a trusted entity
    # set for field-level validation.
    _gen_ledger: Any = None
    if ctx.source_truth_text.strip():
        try:
            from fact_ledger import build_ledger
            _gen_ledger = build_ledger(
                ctx.resume_data, ctx.source_truth_text, run_repair=False,
            )
        except Exception as exc:
            logger.warning("generate_path FactLedger build failed, skip entity validation: %s", exc)

    ctx.resume_data, ctx.fabrication_report = final_fact_guard(
        ctx.source_truth_text, ctx.resume_data, has_cv=ctx.has_cv,
        ledger=_gen_ledger,
    )

    ctx.missing_fields = check_required_fields(ctx.resume_data, user_stage=ctx.user_stage, source_text=ctx.source_truth_text)

    ctx.conflicts = check_time_conflicts(ctx.resume_data)
    ctx.conflicts.extend(check_sort_order(ctx.resume_data))
    ctx.conflicts.extend(check_summary_jd_alignment(
        str(ctx.resume_data.get("summary", "")), ctx.jd_text or ctx.target_role
    ))

    if ctx.conflicts and llm_enabled():
        from resume_llm_repair import llm_resolve_conflicts
        try:
            _resolved = llm_resolve_conflicts(_dicts(ctx.conflicts), ctx.resume_data)
            if _resolved is not None and isinstance(_resolved, list):
                _kept = {r.get("description", "") for r in _resolved if isinstance(r, dict)}
                if _kept:
                    _before = len(ctx.conflicts)
                    ctx.conflicts = [c for c in ctx.conflicts if c.description in _kept]
                    logger.info("LLM resolved %d false-positive conflicts", _before - len(ctx.conflicts))
        except Exception as exc:
            logger.warning("LLM conflict resolution failed (generate_path): %s", exc)

    ctx.perf["validation_s"] = round(time.perf_counter() - t_validate, 3)
    _ensure_time_budget(ctx.started, "validation")
    return ctx

async def stage_score(ctx: PipelineContext) -> PipelineContext:
    """Stage 5: Score + build user_report."""
    if not ctx._has_audit:
        try:
            ctx.audit_report = audit_resume_core(
                resume_data_to_text(ctx.resume_data),
                ctx.jd_text or ctx.target_role,
                resume_data=ctx.resume_data,
            )
            ctx._has_audit = True
        except Exception as exc:
            logger.warning("Audit failed in stage_score, using fallback: %s", exc)
            ctx.audit_report = {"overall_score": 0, "issues": [], "summary": ""}

    direction = _build_generation_direction(ctx.industry, ctx.target_role) or "建议明确目标岗位方向以优化简历。"
    missing_dict = _dicts(ctx.missing_fields)
    conflict_dict = _dicts(ctx.conflicts)
    fab_dict = ctx.fabrication_report.model_dump() if hasattr(ctx.fabrication_report, "model_dump") else {}
    ctx.user_report = _build_user_report(
        missing_fields=missing_dict, conflicts=conflict_dict,
        fabrication=fab_dict, direction=direction,
        ocr_warnings=ctx.ocr_warnings, template_notes=ctx.template_notes,
    )
    score_obj = score_resume(
        ctx.resume_data, original_text=ctx.source_truth_text,
        user_report=ctx.user_report, job_family=ctx.industry,
        user_stage=ctx.user_stage,
        missing_fields=ctx.missing_fields, conflicts=ctx.conflicts,
        fabrication_report=ctx.fabrication_report,
    )
    ctx.score_breakdown = score_obj.model_dump() if hasattr(score_obj, "model_dump") else score_obj.__dict__
    ctx.score = float(ctx.score_breakdown.get("total", 0.0))
    logger.info("评分: 总分=%.1f | 可读性=%s | 完整度=%s | 表达=%s | 反馈=%s",
        ctx.score,
        ctx.score_breakdown.get("readability", "?"),
        ctx.score_breakdown.get("completeness", "?"),
        ctx.score_breakdown.get("expression", "?"),
        ctx.score_breakdown.get("response", "?"),
    )
    return ctx


async def stage_render(ctx: PipelineContext) -> PipelineContext:
    """Stage 6: Render DOCX + generate reply + create draft."""
    missing_dict = _dicts(ctx.missing_fields)
    conflict_dict = _dicts(ctx.conflicts)

    # ── Clean template watermarks + query leakage before rendering ──
    _watermark_warnings = _clean_template_watermarks(
        ctx.resume_data, ctx.query_text, ctx.has_cv,
    )
    if _watermark_warnings:
        for _w in _watermark_warnings:
            ctx.ocr_warnings.append({"source": "template_watermark", "message": _w})
        logger.info("已清除 %d 个水印/泄漏项:", len(_watermark_warnings))
        for _w in _watermark_warnings:
            logger.info("  清除: %s", _w)

    reply_text = _build_llm_reply(
        audit_report=ctx.audit_report, score=ctx.score,
        missing_fields=missing_dict, changes=ctx.changes,
    )
    if not reply_text:
        reply_text = build_reply_text(
            scenario=ctx.scenario, industry=ctx.industry,
            user_stage=ctx.user_stage, missing_fields=missing_dict,
            conflicts=conflict_dict, ocr_warnings=ctx.ocr_warnings,
            direction=ctx.user_report.get("generation_direction", ""),
            score_total=ctx.score,
        )
    ctx.reply_text = reply_text

    t_export = time.perf_counter()
    ctx.files = export_resume_files(
        resume_data=ctx.resume_data, output_dir=OUTPUT_DIR,
        output_format="docx", template=ctx.template_path,
    )
    ctx.perf["export_files_s"] = round(time.perf_counter() - t_export, 3)
    _ensure_time_budget(ctx.started, "export_files")

    t_draft = time.perf_counter()
    ctx.draft_id, ctx.version = create_new_draft(
        drafts_dir=DRAFTS_DIR, resume_data=ctx.resume_data,
        audit_report=ctx.audit_report, jd_text=ctx.jd_text,
        template=ctx.template_path, output_format="docx",
        changes=ctx.changes,
    )
    ctx.perf["draft_s"] = round(time.perf_counter() - t_draft, 3)
    ctx.perf["total_s"] = round(time.perf_counter() - ctx.started, 3)
    return ctx
