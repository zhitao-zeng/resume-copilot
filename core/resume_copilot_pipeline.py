"""Stage pipeline for resume-copilot: explicit phase functions with PipelineContext.

Breaks the monolithic resume_copilot_service() into 7 explicit stages.
Each stage reads from and writes to a PipelineContext dataclass.
No behavior change — pure code movement.
"""

from __future__ import annotations

import copy
import asyncio
import json
import re
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    DATA_RETENTION_SECONDS,
    MAX_FILE_SIZE,
    OUTPUT_DIR,
    REQUEST_TIMEOUT_SECONDS,
    ENABLE_LLM_REPLY,
    call_llm_text,
    call_llm_typed,
    llm_enabled,
    logger,
    sanitize_user_text,
)
from prompts import REPLY_GENERATION_SYSTEM_PROMPT
from input_normalization import (
    html_to_visible_text,
    is_pure_http_url,
    merge_fetched_jd,
    split_url_and_text,
)
from security_utils import (
    cleanup_old_files,
    is_forbidden_ip,
    private_file_mode,
    read_upload_limited,
    safe_filename,
    validate_public_http_url,
)


if aiohttp is not None:
    class _PublicOnlyResolver(aiohttp.abc.AbstractResolver):
        async def resolve(self, host: str, port: int = 0, family: int = 0):
            import socket
            infos = await asyncio.to_thread(
                socket.getaddrinfo,
                host,
                port,
                family,
                socket.SOCK_STREAM,
            )
            results = []
            for resolved_family, _, proto, _, address in infos:
                ip = address[0]
                if is_forbidden_ip(ip):
                    raise OSError("resolved address is not public")
                results.append({
                    "hostname": host,
                    "host": ip,
                    "port": port,
                    "family": resolved_family,
                    "proto": proto,
                    "flags": 0,
                })
            if not results:
                raise OSError("host has no public address")
            return results

        async def close(self) -> None:
            return None


_cleanup_lock = threading.Lock()
_last_cleanup_at = 0.0


def _maybe_cleanup_outputs() -> None:
    global _last_cleanup_at
    now = time.monotonic()
    with _cleanup_lock:
        if now - _last_cleanup_at < 3600:
            return
        _last_cleanup_at = now
    removed = cleanup_old_files(OUTPUT_DIR, DATA_RETENTION_SECONDS)
    if removed:
        logger.info("Expired %d retained output files", removed)

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


def _clean_template_watermarks(
    resume_data: dict, query_text: str = "", has_cv: bool = False, cv_text: str = "",
) -> list[str]:
    """Scan all fields in resume_data for known watermark/template placeholders
    and query leakage (copy-pasted query text in fields). Returns a list of warnings."""
    warnings: list[str] = []
    _query_lower = query_text.lower().strip() if query_text else ""
    _cv_lower = cv_text.lower().strip() if cv_text else ""
    # Core fact fields: values here come from the real CV; a value that also
    # appears in query (user re-stating their own CV) is NOT leakage.
    _FACT_FIELD_SUFFIXES = (
        ".school", ".organization", ".institution", ".company",
        ".degree", ".major", ".name", ".role", ".period",
    )

    def _is_query_leak(value: str, path: str) -> bool:
        if not has_cv:
            # generate_path: query IS the fact source — no field value
            # derived from query should be treated as leakage.
            return False
        # target_role and job_intention are user INTENT, not candidate facts.
        # They are expected to come from query/JD.  Never treat as leakage.
        if ".meta.target_role" in path or ".meta.job_intention" in path:
            return False
        if len(value) < 12:
            return False
        v = value.lower().strip()
        # Core fact fields are exempt from query-leak clearing: the CV is the
        # fact source, and users often re-state their own CV inside the query.
        if any(path.endswith(sfx) for sfx in _FACT_FIELD_SUFFIXES):
            return False
        # A value that already exists in the raw CV text is a real fact,
        # not a copy of the query.
        if _cv_lower and v in _cv_lower:
            return False
        # Query is also a legitimate fact source in V2.  Only remove a value
        # when it is clearly an instruction copied wholesale into a content
        # field; short facts such as Python/PyTorch/school names must survive.
        instruction_markers = (
            "帮我", "请帮", "请根据", "希望你", "按照", "输出", "生成简历",
            "优化简历", "整理简历", "不要编造", "不要增加", "想投", "应聘",
        )
        return v in _query_lower and any(marker in v for marker in instruction_markers)

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


def _prune_empty_resume_values(resume_data: dict) -> None:
    """Remove blank list items/records after all safety cleaners have run."""

    def _clean(value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            cleaned = []
            for item in value:
                normalized = _clean(item)
                if normalized in ("", None, [], {}):
                    continue
                if isinstance(normalized, dict) and not any(
                    child not in ("", None, [], {}) for child in normalized.values()
                ):
                    continue
                if normalized not in cleaned:
                    cleaned.append(normalized)
            return cleaned
        if isinstance(value, dict):
            return {key: _clean(item) for key, item in value.items()}
        return value

    cleaned = _clean(resume_data)
    resume_data.clear()
    resume_data.update(cleaned)



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
    quality_report: dict = field(default_factory=dict)
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)

    # Stage 6 (Render) output
    files: dict = field(default_factory=dict)
    reply_text: str = ""
    draft_id: str = ""
    version: str = ""

    # Debug output (only set when DEBUG_OUTPUT_DIR env is set)
    _debug_dir: str = ""
    _debug_prefix: str = ""

    def _write_debug(self, name: str, data: Any) -> None:
        if not self._debug_dir:
            return
        import json, os
        os.makedirs(self._debug_dir, exist_ok=True)
        path = os.path.join(self._debug_dir, f"{self._debug_prefix}_{name}")
        try:
            if isinstance(data, str):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(data)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("Debug output failed for %s: %s", name, exc)


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
    return is_pure_http_url(value)


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
    try:
        raw = await read_upload_limited(upload, MAX_FILE_SIZE)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{purpose} file too large (> {MAX_FILE_SIZE // (1024 * 1024)} MB)")
    filename = upload.filename or f"{purpose}.bin"
    ext = _file_ext(filename)
    if ext in IMAGE_EXTENSIONS:
        warnings.append({
            "source": purpose, "filename": filename,
            "message": "已对图片执行本地 OCR；如识别不完整，请补充清晰图片或文本。",
        })
    try:
        text = await asyncio.to_thread(extract_text_from_bytes, raw, filename)
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
        current_url = await validate_public_http_url(url)
        timeout = aiohttp.ClientTimeout(total=20, connect=5, sock_read=10)
        connector = aiohttp.TCPConnector(resolver=_PublicOnlyResolver(), use_dns_cache=False)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            for _redirect in range(4):
                async with session.get(
                    current_url,
                    allow_redirects=False,
                    headers={"User-Agent": "resume-copilot/3.0"},
                ) as resp:
                    if 300 <= resp.status < 400 and resp.headers.get("Location"):
                        from urllib.parse import urljoin
                        current_url = await validate_public_http_url(
                            urljoin(current_url, resp.headers["Location"])
                        )
                        continue
                    if resp.status != 200:
                        warnings.append({"source": "target_jd", "message": f"JD 链接请求失败：HTTP {resp.status}"})
                        return ""
                    content_type = resp.headers.get("Content-Type", "")
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        total += len(chunk)
                        if total > MAX_FILE_SIZE:
                            warnings.append({"source": "target_jd", "message": "JD 链接内容超过大小限制，已忽略。"})
                            return ""
                        chunks.append(chunk)
                    payload = b"".join(chunks)
                    if "pdf" in content_type:
                        return await asyncio.to_thread(extract_text_from_bytes, payload, "target_jd.pdf")
                    return html_to_visible_text(payload)
            warnings.append({"source": "target_jd", "message": "JD 链接重定向次数过多。"})
            return ""
    except Exception as exc:
        logger.warning("JD URL rejected or failed: %s", type(exc).__name__)
        warnings.append({"source": "target_jd", "message": "JD 链接不安全、不可访问或内容无效。"})
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
        url, supplied_text = split_url_and_text(value)
        if url:
            resolved = await _fetch_jd_url(url, warnings)
            perf["jd_resolve_s"] = round(time.perf_counter() - started, 3)
            return merge_fetched_jd(resolved, supplied_text)
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
    try:
        raw = await read_upload_limited(upload, MAX_FILE_SIZE)
    except ValueError:
        raise HTTPException(status_code=400, detail="cv_template file too large")
    filename = upload.filename or "template"
    ext = _file_ext(filename)
    if ext == ".docx" or ext == ".pdf" or ext in IMAGE_EXTENSIONS:
        AVATAR_DIR.mkdir(parents=True, exist_ok=True)
        original_name = safe_filename(filename, f"template{ext}")
        path = AVATAR_DIR / (
            f"template_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{original_name}"
        )
        path.write_bytes(raw)
        private_file_mode(path)
        return str(path)
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


def _build_targeted_suggestions(
    jd_text: str,
    resume_data: dict | None,
    target_role: str = "",
) -> list[str]:
    """Create evidence-aware advice without treating JD requirements as facts."""

    data = copy.deepcopy(resume_data or {})
    data.pop("framework", None)
    if isinstance(data.get("meta"), dict):
        # Target intent is not evidence that the candidate has performed the
        # corresponding JD responsibility.
        data["meta"].pop("target_role", None)
        data["meta"].pop("job_intention", None)
    resume_blob = json.dumps(data, ensure_ascii=False).casefold()
    normalized_resume = re.sub(r"[^\w\u4e00-\u9fff]+", "", resume_blob)

    def keyword_supported(keyword: str) -> bool:
        normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", keyword.casefold())
        if len(normalized) < 2:
            return False
        if normalized in normalized_resume:
            return True
        target_bigrams = {normalized[index:index + 2] for index in range(len(normalized) - 1)}
        if not target_bigrams:
            return False
        resume_bigrams = {
            normalized_resume[index:index + 2]
            for index in range(max(0, len(normalized_resume) - 1))
        }
        return len(target_bigrams & resume_bigrams) / len(target_bigrams) >= 0.72
    suggestions: list[str] = []
    focus_items: list[str] = []
    advice_jd = str(jd_text or "")
    # When a URL was accompanied by user-pasted JD text, advice must follow
    # the explicit user text rather than unrelated navigation/page content.
    if "【链接页面正文（仅作补充）】" in advice_jd:
        advice_jd = advice_jd.split("【链接页面正文（仅作补充）】", 1)[0]
    for raw_line in re.split(r"[\r\n]+", advice_jd):
        line = re.sub(r"^\s*(?:[-*•]|\d+[.、)）])\s*", "", raw_line).strip()
        if not line or line in {"岗位职责", "任职要求", "职位描述", "岗位要求"}:
            continue
        line = re.sub(r"^【[^】]+】\s*", "", line).strip()
        if len(line) < 8 or line.lower().startswith(("http://", "https://")):
            continue
        if any(marker in line for marker in (
            "职位详情", "招聘官网", "工作地点", "办公地点", "工作地址", "办公地址",
            "校区", "公司介绍", "立即申请", "职位类别",
        )):
            continue
        if not any(marker in line for marker in (
            "负责", "参与", "输出", "推动", "规划", "分析", "设计", "管理",
            "学历", "经验", "能力", "熟悉", "具备", "要求",
        )):
            continue
        if line not in focus_items:
            focus_items.append(line[:90])
        if len(focus_items) >= 2:
            break

    for focus in focus_items:
        keywords = [item for item in extract_jd_keywords(focus) if len(item.strip()) >= 2]
        matched = [item for item in keywords if keyword_supported(item)]
        unmatched = [item for item in keywords if item not in matched]
        if matched and unmatched:
            suggestions.append(
                f"JD重点“{focus}”已匹配{'、'.join(matched[:3])}；"
                f"仍缺{'、'.join(unmatched[:3])}的直接证据，请仅在确有经历时补充具体动作、交付物和结果。"
            )
        elif matched:
            suggestions.append(
                f"JD重点“{focus}”已有对应证据（{'、'.join(matched[:3])}）；"
                "建议把最相关的一条经历前置，并补清个人动作、交付物和可核验结果。"
            )
        else:
            suggestions.append(
                f"JD重点包含“{focus}”，当前简历缺少对应证据；若确有相关经历，请补充，未参与则不要写入。"
            )

    projects = data.get("projects", []) if isinstance(data, dict) else []
    project_names = [
        str(item.get("name", "")).strip()
        for item in projects if isinstance(item, dict) and str(item.get("name", "")).strip()
    ][:3]
    role = str(target_role or (data.get("meta", {}) or {}).get("target_role", "")).strip()
    if len(suggestions) < 3 and project_names:
        suggestions.append(
            f"针对{role or '目标岗位'}，建议为{'、'.join(project_names)}补充项目时间、个人职责、方法/工具及真实评估结果。"
        )
    elif len(suggestions) < 3 and role:
        suggestions.append(
            f"围绕{role}补充1–2段最相关的真实经历，写清背景、个人动作、交付物和可核验结果。"
        )
    research = data.get("research", []) if isinstance(data, dict) else []
    research_names = [
        str(item.get("role") or item.get("topic") or item.get("company") or "").strip()
        for item in research if isinstance(item, dict)
        and str(item.get("role") or item.get("topic") or item.get("company") or "").strip()
    ][:2]
    if len(suggestions) < 3 and research_names:
        suggestions.append(
            f"为{'、'.join(research_names)}科研经历补充数据来源、采用方法、个人贡献和真实评估指标，以支撑{role or '目标岗位'}匹配度。"
        )
    return suggestions[:3]


_REPLY_MAX_SOURCE_GAPS = 5
_REPLY_MAX_CLAIM_GAPS = 3
_REPLY_MAX_FOLLOW_UPS = 3
_REPLY_CONTACT_OR_URL = re.compile(
    r"(?:\b1[3-9]\d{9}\b|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"https?://\S+|www\.\S+|(?:电话|手机|邮箱|联系方式)\s*[:：]|"
    r"\bQQ\s*[:：]?\s*[1-9]\d{4,11}\b|"
    r"(?:微信(?:号|ID)?|WeChat)\s*[:：]?\s*[A-Za-z][\w-]{3,})",
    re.IGNORECASE,
)
_REPLY_NON_ACTIONABLE_PROFILE = re.compile(
    r"^(?:(?:年龄\s*[:：]?)?\d{1,2}\s*岁(?:\s*[男女])?|性别\s*[:：]?\s*[男女]|"
    r"(?:求职意向|期望岗位|应聘岗位)\s*[:：])",
    re.IGNORECASE,
)
_REPLY_STRUCTURAL_ONLY = re.compile(
    r"^(?:个人信息|基本信息|联系方式|教育经历|教育背景|工作经历|实习经历|"
    r"项目经历|科研经历|校园经历|专业技能|技能|证书|荣誉奖项|个人总结|自我评价)$",
    re.IGNORECASE,
)


def _clean_report_excerpt(value: Any) -> str:
    """Remove contact and layout debris before exposing a source gap."""

    text = str(value or "").strip()
    text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "", text)
    text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "", text)
    text = re.sub(r"\bQQ\s*[:：]?\s*[1-9]\d{4,11}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?:微信(?:号|ID)?|WeChat)\s*[:：]?\s*[A-Za-z][\w-]{3,}",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^[\s\-•·▪◦\d.、)）]+", "", text)
    text = re.sub(r"[✉☎☁▯□]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" ，,。；;|｜-/")


def _select_reply_source_gaps(values: Any) -> list[dict[str, Any]]:
    """Rank substantive omissions instead of dumping raw OCR fragments."""

    if not isinstance(values, list):
        return []
    section_rank = {
        "experience": 6,
        "projects": 6,
        "research": 5,
        "activities": 5,
        "education": 4,
        "skills": 3,
        "awards": 2,
        "certifications": 2,
        "meta": 0,
    }
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            continue
        raw_excerpt = str(item.get("excerpt", "")).strip()
        excerpt = _clean_report_excerpt(raw_excerpt)
        compact = re.sub(r"[^\w\u4e00-\u9fff]+", "", excerpt).casefold()
        if (
            len(compact) < 4
            or _REPLY_STRUCTURAL_ONLY.fullmatch(excerpt)
            or _REPLY_NON_ACTIONABLE_PROFILE.search(excerpt)
            or (_REPLY_CONTACT_OR_URL.search(raw_excerpt) and len(compact) < 12)
        ):
            continue
        section = str(item.get("section_hint", "") or "").strip()
        dimensions = str(item.get("dimensions", "") or "")
        score = section_rank.get(section, 1)
        score += 2 if "action" in dimensions else 0
        score += 1 if any(token in dimensions for token in ("deliverable", "result", "anchor")) else 0
        score += 1 if re.search(r"\d|%|万|亿|人|次|个|条|元", excerpt) else 0
        normalized_item = dict(item)
        normalized_item["excerpt"] = excerpt
        candidates.append((score, index, normalized_item))

    selected: list[dict[str, Any]] = []
    selected_keys: list[str] = []
    for _, _, item in sorted(candidates, key=lambda row: (-row[0], row[1])):
        key = re.sub(r"[^\w\u4e00-\u9fff]+", "", item["excerpt"]).casefold()
        if any(key == existing or (len(key) >= 8 and key in existing) for existing in selected_keys):
            continue
        selected.append(item)
        selected_keys.append(key)
        if len(selected) >= _REPLY_MAX_SOURCE_GAPS:
            break
    return selected


def _reply_detail_block(
    missing_fields: list[dict[str, Any]],
    targeted_suggestions: list[str],
    quality_report: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = ["缺失信息："]
    if missing_fields:
        unique_items: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in missing_fields:
            key = (str(item.get("field", "")), str(item.get("reason", "")))
            if key not in seen:
                seen.add(key)
                unique_items.append(item)
        lines.append(f"缺失或待补充信息（{len(unique_items)}项）：")
        for item in unique_items:
            label = str(item.get("label") or item.get("field") or "字段").strip()
            field = str(item.get("field", ""))
            indexed = re.match(
                r"(education|experience|research|activities|campus_experience|projects)\[(\d+)]",
                field,
            )
            if indexed:
                noun = {
                    "education": "段教育",
                    "experience": "段工作/实习",
                    "research": "段科研",
                    "activities": "段校园/社会",
                    "campus_experience": "段校园/社会",
                    "projects": "个项目",
                }[indexed.group(1)]
                label += f"（第{int(indexed.group(2)) + 1}{noun}）"
            reason = str(item.get("reason", "")).strip()
            lines.append(f"- {label}：{reason}" if reason else f"- {label}")
    else:
        lines.append("- 未检测到必填信息缺失；仍建议核对联系方式、时间和成果数据。")
    lines.append("岗位匹配与建议：")
    report = quality_report if isinstance(quality_report, dict) else {}
    alignment = report.get("job_alignment", {})
    if isinstance(alignment, dict) and alignment.get("has_job_description"):
        supported = int(alignment.get("supported_requirement_count", 0) or 0)
        partial = int(alignment.get("partial_requirement_count", 0) or 0)
        missing = int(alignment.get("missing_requirement_count", 0) or 0)
        lines.append(
            f"- JD逐项核对：{supported}项有直接证据，{partial}项仅部分匹配，{missing}项尚无直接证据。"
        )
        gap_items = [
            item for item in (alignment.get("requirements", []) or [])
            if isinstance(item, dict) and item.get("status") in {"partial", "missing"}
        ]
        for item in gap_items[:3]:
            requirement = str(item.get("requirement", "")).strip()
            missing_aspects = [
                str(value).strip() for value in (item.get("missing_aspects", []) or [])
                if str(value).strip()
            ]
            if requirement:
                suffix = (
                    "；需补：" + "、".join(missing_aspects[:3])
                    if missing_aspects else ""
                )
                lines.append(f"- {'部分匹配' if item.get('status') == 'partial' else '未匹配'}：{requirement}{suffix}。")
    if targeted_suggestions:
        lines.append("针对岗位的建议：")
        lines.extend(f"- {item}" for item in targeted_suggestions)
    else:
        lines.append("- 暂无足够岗位信息形成具体匹配结论；提供完整JD后可进一步优化关键词与经历排序。")
    preservation = report.get("source_preservation", {})
    unrepresented = (
        preservation.get("unrepresented_items", [])
        if isinstance(preservation, dict) else []
    )
    if unrepresented:
        total = int(preservation.get("unrepresented_item_count", len(unrepresented)) or 0)
        lines.append(f"原始材料中未充分写入成稿的信息（{total}项）：")
        selected_unrepresented = _select_reply_source_gaps(unrepresented)
        for item in selected_unrepresented:
            excerpt = str(item.get("excerpt", "")).strip()
            if excerpt:
                lines.append(f"- {excerpt}")
        hidden_count = max(0, total - len(selected_unrepresented))
        if hidden_count:
            lines.append(
                f"- 另有 {hidden_count} 项未逐项展开（包含重复或结构性片段），"
                "建议优先对照上述关键事实复核。"
            )
    grounding = report.get("fact_grounding", {})
    if isinstance(grounding, dict):
        unsupported_count = int(grounding.get("unsupported_item_count", 0) or 0)
        if unsupported_count:
            lines.append(
                f"为避免编造，已移除 {unsupported_count} 处缺少候选人事实依据的生成内容。"
            )
    improvement_items = report.get("claim_improvement_opportunities", [])
    if isinstance(improvement_items, list) and improvement_items:
        lines.append("经历表达仍可补充：")
        for item in improvement_items[:_REPLY_MAX_CLAIM_GAPS]:
            if not isinstance(item, dict):
                continue
            record_label = str(item.get("record_label", "")).strip()
            excerpt = str(item.get("excerpt", "")).strip()
            dimensions = [
                str(value).strip() for value in (item.get("missing_dimensions", []) or [])
                if str(value).strip()
            ]
            if excerpt and dimensions:
                location = f"{record_label}：" if record_label else ""
                lines.append(
                    f"- {location}“{excerpt}”缺少{'、'.join(dimensions)}；"
                    "仅在有真实信息时补充。"
                )
    follow_ups = report.get("follow_up_questions", [])
    if isinstance(follow_ups, list) and follow_ups:
        lines.append("建议补充回答：")
        lines.extend(
            f"- {str(item).strip()}"
            for item in follow_ups[:_REPLY_MAX_FOLLOW_UPS]
            if str(item).strip()
        )
    return "\n".join(lines)


def _resume_section_overview(resume_data: dict[str, Any] | None) -> str:
    data = resume_data if isinstance(resume_data, dict) else {}
    framework = data.get("framework")
    if isinstance(framework, dict):
        titles = [
            str(item.get("title", "")).strip()
            for item in framework.get("sections", [])
            if isinstance(item, dict) and str(item.get("title", "")).strip()
        ]
        return "、".join(titles)

    labels = {
        "summary": "个人总结",
        "education": "教育经历",
        "experience": "工作/实习经历",
        "research": "科研经历",
        "campus_experience": "校园经历",
        "projects": "项目经历",
        "awards": "荣誉奖项",
        "publications": "论文/专利",
        "certifications": "证书与资质",
        "training": "培训经历",
        "teaching": "教学经历",
        "additional_sections": "其他专业经历",
    }
    sections: list[str] = []
    for key, label in labels.items():
        value = data.get(key)
        if (isinstance(value, str) and value.strip()) or (
            isinstance(value, (list, dict)) and bool(value)
        ):
            sections.append(label)
    skills = data.get("skills")
    if isinstance(skills, dict) and any(
        isinstance(value, list) and any(str(item).strip() for item in value)
        for value in skills.values()
    ):
        sections.append("专业技能")
    return "、".join(dict.fromkeys(sections))


def _reply_result_block(
    *,
    direction: str,
    resume_data: dict[str, Any] | None,
    framework_mode: bool,
) -> str:
    lines = ["生成方向总结："]
    if framework_mode:
        lines.append("- 未收到可核验的个人信息；已生成待填写框架，框架内容不代表候选人已有经历。")
    elif direction.strip():
        lines.append(f"- {direction.strip()}")
    else:
        lines.append("- 基于已提供的个人事实完成结构化整理，并优先保留可核验的经历、方法和结果。")
    overview = _resume_section_overview(resume_data)
    if overview:
        lines.append(f"- 已生成模块：{overview}。")
    data = resume_data if isinstance(resume_data, dict) else {}
    if not framework_mode and data:
        education_count = len(data.get("education", []) or [])
        experience_count = len(data.get("experience", []) or [])
        project_count = len(data.get("projects", []) or [])
        campus_count = len(data.get("campus_experience", []) or [])
        bullet_count = sum(
            len(item.get("bullets", []) or [])
            for section in ("experience", "projects", "research", "campus_experience")
            for item in (data.get(section, []) or [])
            if isinstance(item, dict)
        )
        skills = data.get("skills", {})
        skill_count = (
            sum(len(value) for value in skills.values() if isinstance(value, list))
            if isinstance(skills, dict) else 0
        )
        structured_count = sum(
            len(data.get(section, []) or [])
            for section in (
                "awards", "certifications", "publications", "patents", "training", "teaching",
            )
        )
        counts = []
        for count, label in (
            (education_count, "段教育经历"),
            (experience_count, "段工作/实习经历"),
            (project_count, "个项目"),
            (campus_count, "段校园/社会经历"),
            (bullet_count, "条职责与成果描述"),
            (skill_count, "项专业技能"),
            (structured_count, "项证书/奖项/专业成果"),
        ):
            if count:
                counts.append(f"{count}{label}")
        if counts:
            lines.append("- 本次成稿整理了" + "、".join(counts) + "；未补写材料中不存在的经历或数据。")
    return "\n".join(lines)


def _reply_conflict_block(conflicts: list[dict[str, Any]]) -> str:
    if not conflicts:
        return "时间或内容冲突：\n- 未检测到明显时间或内容冲突；正式投递前仍请人工复核。"
    lines = ["时间或内容冲突：", f"需要确认的时间或内容冲突（{len(conflicts)}项）："]
    for item in conflicts[:8]:
        description = str(item.get("description", "")).strip()
        if description:
            lines.append(f"- {description}")
    if len(conflicts) > 8:
        lines.append(f"- 另有 {len(conflicts) - 8} 项，请在结构化报告中继续核对。")
    return "\n".join(lines) if len(lines) > 1 else ""


def _build_llm_reply(
    *, audit_report, score, missing_fields, changes,
    jd_text: str = "", resume_data: dict | None = None,
    quality_report: dict[str, Any] | None = None,
    conflicts: list[dict[str, Any]] | None = None,
    direction: str = "",
    framework_mode: bool = False,
) -> str:
    """Generate reply_text via LLM. Returns empty string on failure."""
    if not ENABLE_LLM_REPLY or not llm_enabled():
        return ""
    try:
        summary_parts = []

        # Resume content overview
        if resume_data and isinstance(resume_data, dict):
            sections = []
            edu = resume_data.get("education", [])
            exp = resume_data.get("experience", [])
            proj = resume_data.get("projects", [])
            skills = resume_data.get("skills", {})
            summary = resume_data.get("summary", "")
            if edu:
                sections.append(f"{len(edu)}段教育经历")
            if exp:
                sections.append(f"{len(exp)}段工作/实习经历")
            if proj:
                sections.append(f"{len(proj)}个项目")
            if isinstance(skills, dict) and skills.get("items"):
                sections.append(f"{len(skills['items'])}项技能")
            elif isinstance(skills, dict):
                total = sum(len(v) for v in skills.values() if isinstance(v, list))
                if total:
                    sections.append(f"{total}项技能")
            if summary:
                sections.append("个人总结")
            if sections:
                summary_parts.append("简历包含：" + "、".join(sections))
            else:
                summary_parts.append("简历内容为空或极少，需要用户补充个人信息")

        if framework_mode:
            summary_parts.append("本次没有可核验的个人事实，必须明确说明已生成待填写框架，不能声称已生成个人经历")
        if direction.strip():
            summary_parts.append("生成方向：" + direction.strip())
        if conflicts:
            summary_parts.append(
                "需要用户确认的冲突：" + "；".join(
                    str(item.get("description", "")).strip()
                    for item in conflicts[:8]
                    if str(item.get("description", "")).strip()
                )
            )

        # Audit issues (V1 path)
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

        # Missing fields — be specific
        if missing_fields:
            labels = [item.get("label", "") for item in missing_fields if item.get("label")]
            reasons = [item.get("reason", "") for item in missing_fields if item.get("reason")]
            if labels:
                summary_parts.append(f"缺失字段：{'、'.join(labels)}")
            if reasons:
                summary_parts.append(f"具体说明：{'；'.join(reasons)}")
        else:
            summary_parts.append("提醒用户核对联系方式、教育时间等关键信息是否完整")

        # V2 changes (verified/corrected items)
        if changes:
            actions = [c for c in changes if isinstance(c, dict)]
            if actions:
                rewrite_count = _rewrite_change_count(actions)
                if rewrite_count:
                    summary_parts.append(
                        f"已在不新增事实的前提下优化 {rewrite_count} 条经历/项目表述"
                    )
                correction_count = len(actions) - rewrite_count
                if correction_count:
                    summary_parts.append(f"校验阶段处理了 {correction_count} 处修正")
                for c in actions[:3]:
                    reason = c.get("reason", "")
                    if reason:
                        summary_parts.append(f"修正：{reason[:60]}")

        # JD context for targeted advice
        if jd_text and jd_text.strip():
            jd_snippet = jd_text.strip()[:200]
            summary_parts.append(f"目标岗位参考：{jd_snippet}")
            summary_parts.append("请根据目标岗位给出 1-2 条针对性建议")

        report = quality_report if isinstance(quality_report, dict) else {}
        job_alignment = report.get("job_alignment", {})
        targeted_suggestions = (
            list(job_alignment.get("recommendations", []))
            if isinstance(job_alignment, dict) else []
        )
        if not targeted_suggestions:
            targeted_suggestions = _build_targeted_suggestions(
                jd_text, resume_data, str((resume_data or {}).get("meta", {}).get("target_role", ""))
            )

        user_prompt = "请根据以下简历处理结果生成面向用户的自然语言回复（不要提及具体评分数值）：\n\n" + "\n".join(summary_parts)
        reply = call_llm_text(
            system_prompt=REPLY_GENERATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3, max_tokens=768,
        )
        detail_block = _reply_detail_block(
            missing_fields, targeted_suggestions, quality_report=report,
        )
        if reply and detail_block and detail_block not in reply:
            reply += "\n\n" + detail_block
        result_block = _reply_result_block(
            direction=direction,
            resume_data=resume_data,
            framework_mode=framework_mode,
        )
        if reply and result_block and result_block not in reply:
            reply += "\n\n" + result_block
        conflict_block = _reply_conflict_block(conflicts or [])
        if reply and conflict_block and conflict_block not in reply:
            reply += "\n\n" + conflict_block
        rewrite_count = _rewrite_change_count(changes)
        rewrite_message = f"已在不新增事实的前提下优化 {rewrite_count} 条经历/项目表述。"
        if reply and rewrite_count and f"优化 {rewrite_count} 条" not in reply:
            reply += "\n\n" + rewrite_message
        if reply:
            logger.info("Reply generated: chars=%d", len(reply))
        return reply if reply else ""
    except Exception as exc:
        logger.warning("LLM reply generation failed: %s", exc)
        return ""


def _rewrite_change_count(changes) -> int:
    return sum(
        1
        for item in (changes or [])
        if isinstance(item, dict)
        and item.get("action") == "replace"
        and "bullets[" in str(item.get("path", ""))
    )


def build_reply_text(
    *, scenario, industry, user_stage, missing_fields, conflicts,
    ocr_warnings, direction, score_total, changes=None,
    targeted_suggestions=None,
    quality_report: dict[str, Any] | None = None,
    resume_data: dict[str, Any] | None = None,
    framework_mode: bool = False,
    template_notes: list[str] | None = None,
) -> str:
    scenario_label = {
        "scenario1": "原始简历与目标 JD 优化",
        "scenario2": "个人信息生成标准简历",
        "scenario3": "原始简历按目标岗位优化",
        "scenario4": "个人信息结合目标 JD 生成简历",
    }.get(scenario, "简历生成/优化")
    if framework_mode:
        scenario_label = "目标 JD 待填写简历框架"
    header = "已按\"" + scenario_label + "\"完成一版可编辑 DOCX，识别方向为" + product_logic.display_industry(industry) + "，用户阶段为" + product_logic.display_user_stage(user_stage) + "。"
    parts = [header, _reply_result_block(
        direction=direction,
        resume_data=resume_data,
        framework_mode=framework_mode,
    )]
    if template_notes:
        parts.append("模板处理: " + "；".join(str(note) for note in template_notes[:2] if str(note).strip()))
    rewrite_count = _rewrite_change_count(changes)
    if rewrite_count:
        parts.append(f"已在不新增事实的前提下优化 {rewrite_count} 条经历/项目表述。")
    parts.append(_reply_detail_block(
        missing_fields,
        targeted_suggestions or [],
        quality_report=quality_report,
    ))
    if missing_fields:
        parts.append("建议补齐上述信息后再用于正式投递。")
    parts.append(_reply_conflict_block(conflicts))
    if ocr_warnings:
        warnings = "; ".join(item.get("message", "") for item in ocr_warnings[:3] if item.get("message"))
        parts.append("OCR/文件提示: " + warnings)
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
        _validate_period_entities(resume_data, source_truth_text, fab)

    return resume_data, fab


def _OLD_apply_fabrication_report(
    resume_data: dict, fab: FabricationReport,
) -> None:
    """Clear fabricated fields from resume_data based on report details.

    The fabrication report catches field-level fabrications (school, degree,
    major, period, company, role).  This function enforces those findings
    by clearing the corresponding values from resume_data.

    Does NOT clear date/number/skill fabrications — those are mixed with
    legitimate values in text fields and need finer handling.
    """
    _CLEARABLE_TYPES = frozenset({"school", "degree", "major", "company",
                                   "role", "work_experience", "education_level"})
    for detail in fab.details:
        if detail.type not in _CLEARABLE_TYPES:
            continue
        target = detail.content

        # Meta-level fields
        meta = resume_data.get("meta", {})
        if isinstance(meta, dict):
            if detail.type == "work_experience" and meta.get("work_experience") == target:
                meta["work_experience"] = ""
                continue
            if detail.type == "education_level" and meta.get("education_level") == target:
                meta["education_level"] = ""
                continue

        # Education fields
        for edu in resume_data.get("education", []):
            if not isinstance(edu, dict):
                continue
            if detail.type == "school" and edu.get("school") == target:
                edu["school"] = ""
            if detail.type == "degree" and edu.get("degree") == target:
                edu["degree"] = ""
            if detail.type == "major" and edu.get("major") == target:
                edu["major"] = ""
            if detail.type == "date" and edu.get("period") and target in edu.get("period", ""):
                edu["period"] = ""

        # Experience fields
        for exp in resume_data.get("experience", []):
            if not isinstance(exp, dict):
                continue
            if detail.type == "company" and exp.get("company") == target:
                exp["company"] = ""
            if detail.type == "role" and exp.get("role") == target:
                exp["role"] = ""


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


def _validate_period_entities(
    resume_data: dict, raw_text: str, fab: FabricationReport,
) -> None:
    """Clear period fields that completely lack year evidence in source text.

    NEGATIVE GUARD ONLY — clears periods where no single year from the
    period value appears anywhere in source_truth_text.  This catches
    e.g. an LLM fabricating "01-2020 - 12-2023" when the user's query
    contains no year dates at all.

    LIMITATIONS (do NOT expand semantic without provenance tracking):
    - Year existence ≠ period verified.  A period's year may match a
      year belonging to a *different* section (e.g. education year
      appearing in experience period).
    - Month-level alignment is NOT checked.
    - Section-level provenance is NOT tracked.

    This is a cheap negative filter, not a date audit system.
    """
    import re
    from schemas import FabricationDetail

    def _year_supported(period: str, source_lower: str) -> bool:
        years = re.findall(r"(?:19|20)\d{2}", period)
        if not years:
            return True
        for y in years:
            if y in source_lower:
                return True
        return False

    source_lower = str(raw_text or "").lower()

    for exp in resume_data.get("experience", []):
        if not isinstance(exp, dict):
            continue
        period = str(exp.get("period", "")).strip()
        if period and not _year_supported(period, source_lower):
            exp["period"] = ""
            fab.details.append(FabricationDetail(
                type="period",
                content=period,
                reason="该时间未出现在用户原始输入中",
            ))

    for edu in resume_data.get("education", []):
        if not isinstance(edu, dict):
            continue
        period = str(edu.get("period", "")).strip()
        if period and not _year_supported(period, source_lower):
            edu["period"] = ""
            fab.details.append(FabricationDetail(
                type="period",
                content=period,
                reason="该时间未出现在用户原始输入中",
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
    await asyncio.to_thread(_maybe_cleanup_outputs)
    ctx.query_text = str(query or "").strip()
    # Init debug output dir if configured (with unique request ID per run
    # to prevent cross-contamination between concurrent requests)
    import os as _os
    _debug_out = _os.environ.get("DEBUG_OUTPUT_DIR", "").strip()
    if _debug_out:
        _req_id = _os.environ.get("REQUEST_ID", "") or str(int(ctx.started * 1000000))[-10:]
        ctx._debug_dir = _os.path.join(_debug_out, f"req_{_req_id}")
        _os.makedirs(ctx._debug_dir, exist_ok=True)
        case_tag = (str(query or "")[:30] if query else "no-query").replace("\n", " ").replace("/", "_")
        ctx._debug_prefix = "00"
        ctx._write_debug("01_query.txt", query or "")
        ctx._write_debug("00_input_raw.txt", {
            "query": query, "has_cv": cv is not None,
            "has_jd": bool(target_jd or jd_text or target_jd_url or jd_url),
            "has_template": cv_template is not None,
        })
        # Set prefix for next stage
        ctx._debug_prefix = "01"

    # CV extraction + upload tracking
    ctx.cv_text = ""
    ctx.cv_uploaded = cv is not None
    if cv is not None:
        ctx.cv_text = await _extract_upload_text(cv, "cv", ctx.perf, ctx.ocr_warnings)

    # OCR quality gate
    ctx._write_debug("02_cv_text.txt", ctx.cv_text)
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
        "CV extraction summary | has_cv=%s | cv_text_chars=%d | ocr_quality=%s | low_ocr_quality=%s | query_chars=%d | has_jd=%s",
        ctx.has_cv, cv_text_len,
        ctx.ocr_quality.get("score") if ctx.ocr_quality else "N/A",
        ctx._low_ocr_quality, len(ctx.query_text), ctx.has_jd,
    )
    if ctx.cv_text and ctx.cv_text.strip():
        logger.info("CV extraction succeeded | chars=%d", len(ctx.cv_text))

    # Template
    effective_template = await _resolve_template_path(cv_template, ctx.ocr_warnings, template)
    if effective_template == DEFAULT_TEMPLATE:
        effective_template = template or DEFAULT_TEMPLATE
    if cv_template is not None and effective_template == (template or DEFAULT_TEMPLATE):
        ctx.template_notes.append("用户模板未能完整复刻，已使用标准 DOCX 模板输出。")
    elif cv_template is not None and Path(effective_template).is_file():
        suffix = Path(effective_template).suffix.lower()
        if suffix == ".docx":
            ctx.template_notes.append("已优先复用用户 DOCX 模板的结构、页边距与视觉样式。")
        else:
            ctx.template_notes.append("已从用户模板提取配色、对齐、间距与标题装饰，并生成可编辑 DOCX。")
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

    # A JD-only request is valid: V2 renders a structured, explicitly
    # unfilled resume framework without inventing candidate facts.
    if not ctx.generation_text.strip() and not ctx.has_jd:
        raise HTTPException(status_code=400, detail="query or cv is required")

    t_classify = time.perf_counter()
    classification = await asyncio.to_thread(
        classify_resume_request,
        query=ctx.query_text, cv_text=ctx.cv_text, jd_text=ctx.jd_text,
        has_cv=ctx.has_cv, has_jd=ctx.has_jd, resume_data=None,
    )
    ctx.target_role = classification.target_role
    ctx.industry = classification.industry
    ctx.user_stage = classification.user_stage
    ctx.perf["classify_s"] = round(time.perf_counter() - t_classify, 3)
    ctx._debug_prefix = "02"
    ctx._write_debug("03_source_truth.txt", ctx.source_truth_text)
    ctx._write_debug("03_generation_text.txt", ctx.generation_text)

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
                logger.info(
                    "Applied bullet patch telemetry: accepted=%d sampled_paths=%d",
                    len(patches),
                    min(5, len(patches)),
                )
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

    # Debug: parse result
    ctx._debug_prefix = "03"
    ctx._write_debug("05_parsed_resume.json", ctx.resume_data)

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
    ctx._debug_prefix = "04"
    ctx._write_debug("06_after_fact_guard.json", {
        "resume_data": ctx.resume_data,
        "fabrication_report": ctx.fabrication_report.model_dump() if hasattr(ctx.fabrication_report, "model_dump") else {},
    })
    ctx._write_debug("08_missing_fields.json", [{"field": m.field, "label": m.label, "reason": m.reason, "source": m.source} for m in ctx.missing_fields] if ctx.missing_fields else [])

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
    ctx._debug_prefix = "04"
    ctx._write_debug("06_after_fact_guard.json", {
        "resume_data": ctx.resume_data,
        "fabrication_report": ctx.fabrication_report.model_dump() if hasattr(ctx.fabrication_report, "model_dump") else {},
    })
    ctx._write_debug("08_missing_fields.json", [{"field": m.field, "label": m.label, "reason": m.reason, "source": m.source} for m in ctx.missing_fields] if ctx.missing_fields else [])

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
    """Legacy scoring stage retained for older API callers."""
    ctx = await stage_prepare_report(ctx)
    score_obj = score_resume(
        ctx.resume_data, original_text=ctx.source_truth_text,
        user_report=ctx.user_report, job_family=ctx.industry,
        user_stage=ctx.user_stage,
        missing_fields=ctx.missing_fields, conflicts=ctx.conflicts,
        fabrication_report=ctx.fabrication_report,
    )
    ctx.score_breakdown = score_obj.model_dump() if hasattr(score_obj, "model_dump") else score_obj.__dict__
    ctx.score = float(ctx.score_breakdown.get("total", 0.0))
    return ctx


async def stage_prepare_report(ctx: PipelineContext) -> PipelineContext:
    """Build actionable output metadata without assigning a synthetic score."""
    if not ctx._has_audit:
        try:
            ctx.audit_report = await asyncio.to_thread(
                audit_resume_core,
                resume_data_to_text(ctx.resume_data),
                ctx.jd_text or ctx.target_role,
                resume_data=ctx.resume_data,
            )
            ctx._has_audit = True
        except Exception as exc:
            logger.warning("Audit failed while preparing report, using fallback: %s", exc)
            ctx.audit_report = {"overall_score": 0, "issues": [], "summary": ""}

    if isinstance(ctx.resume_data.get("framework"), dict):
        direction = (
            f"未收到可写入简历的个人事实，已按{ctx.target_role or '目标岗位'}方向生成待填写框架。"
        )
    else:
        direction = _build_generation_direction(ctx.industry, ctx.target_role) or "建议明确目标岗位方向以优化简历。"
    missing_dict = _dicts(ctx.missing_fields)
    conflict_dict = _dicts(ctx.conflicts)
    fab_dict = ctx.fabrication_report.model_dump() if hasattr(ctx.fabrication_report, "model_dump") else {}
    ctx.user_report = _build_user_report(
        missing_fields=missing_dict, conflicts=conflict_dict,
        fabrication=fab_dict, direction=direction,
        ocr_warnings=ctx.ocr_warnings, template_notes=ctx.template_notes,
    )
    job_alignment = (
        ctx.quality_report.get("job_alignment", {})
        if isinstance(ctx.quality_report, dict) else {}
    )
    targeted_suggestions = (
        list(job_alignment.get("recommendations", []))
        if isinstance(job_alignment, dict) else []
    )
    if not targeted_suggestions:
        targeted_suggestions = _build_targeted_suggestions(
            ctx.jd_text, ctx.resume_data, ctx.target_role,
        )
    if targeted_suggestions:
        ctx.user_report["targeted_suggestions"] = targeted_suggestions
    if ctx.quality_report:
        ctx.user_report["quality_report"] = ctx.quality_report
    if isinstance(ctx.resume_data.get("framework"), dict):
        ctx.user_report["framework_mode"] = True
    if ctx.changes:
        ctx.user_report["changes"] = ctx.changes
    ctx.score = 0.0
    ctx.score_breakdown = {}
    return ctx


async def stage_render(ctx: PipelineContext) -> PipelineContext:
    """Stage 6: Render DOCX + generate reply + create draft."""
    missing_dict = _dicts(ctx.missing_fields)
    conflict_dict = _dicts(ctx.conflicts)

    # ── Clean template watermarks + query leakage before rendering ──
    _watermark_warnings = _clean_template_watermarks(
        ctx.resume_data, ctx.query_text, ctx.has_cv, ctx.cv_text,
    )
    if _watermark_warnings:
        for _w in _watermark_warnings:
            ctx.ocr_warnings.append({"source": "template_watermark", "message": _w})
        logger.info("已清除 %d 个水印/泄漏项:", len(_watermark_warnings))
        for _w in _watermark_warnings:
            logger.info("  清除: %s", _w)
    _prune_empty_resume_values(ctx.resume_data)

    # The free-form reply model could relabel a factual role (for example,
    # calling a product-assistant job an internship) even though the resume
    # itself remained grounded.  Build the public reply exclusively from the
    # validated structured report; this is also faster and keeps every role,
    # omission and conflict traceable to the final document.
    reply_text = build_reply_text(
        scenario=ctx.scenario, industry=ctx.industry,
        user_stage=ctx.user_stage, missing_fields=missing_dict,
        conflicts=conflict_dict, ocr_warnings=ctx.ocr_warnings,
        direction=ctx.user_report.get("generation_direction", ""),
        score_total=ctx.score,
        changes=ctx.changes,
        targeted_suggestions=ctx.user_report.get("targeted_suggestions", []),
        quality_report=ctx.quality_report,
        resume_data=ctx.resume_data,
        framework_mode=bool(ctx.user_report.get("framework_mode")),
        template_notes=ctx.template_notes,
    )
    ctx.reply_text = reply_text

    t_export = time.perf_counter()
    ctx.files = await asyncio.to_thread(
        export_resume_files,
        resume_data=ctx.resume_data, output_dir=OUTPUT_DIR,
        output_format="docx", template=ctx.template_path,
    )
    for generated_path in ctx.files.values():
        if generated_path:
            private_file_mode(Path(generated_path))
    ctx.perf["export_files_s"] = round(time.perf_counter() - t_export, 3)
    _ensure_time_budget(ctx.started, "export_files")

    t_draft = time.perf_counter()
    ctx.draft_id, ctx.version = await asyncio.to_thread(
        create_new_draft,
        drafts_dir=DRAFTS_DIR, resume_data=ctx.resume_data,
        audit_report=ctx.audit_report, jd_text=ctx.jd_text,
        template=ctx.template_path, output_format="docx", changes=ctx.changes,
    )
    ctx.perf["draft_s"] = round(time.perf_counter() - t_draft, 3)
    ctx.perf["total_s"] = round(time.perf_counter() - ctx.started, 3)
    ctx._debug_prefix = "09"
    ctx._write_debug("09_reply_context.json", {
        "reply_text": ctx.reply_text,
        "missing_fields": [{"field": m.field, "label": m.label, "reason": m.reason, "source": m.source} for m in ctx.missing_fields] if ctx.missing_fields else [],
        "fabrication_found": ctx.fabrication_report.fabrication_found if hasattr(ctx.fabrication_report, "fabrication_found") else False,
        "score": ctx.score,
        "user_stage": ctx.user_stage,
        "target_role": ctx.target_role,
        "scenario": ctx.scenario,
        "has_cv": ctx.has_cv,
    })
    return ctx
