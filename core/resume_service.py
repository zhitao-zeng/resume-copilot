import time
from pathlib import Path
from typing import Any, Literal, Optional

from http_compat import HTTPException, UploadFile

from audit_logic import (
    _audit_fallback,
    _build_mcp_tools_instruction,
    _build_user_report,
    _jd_changed,
    audit_resume_core,
)
from drafts import append_draft_version, create_new_draft, load_draft_state
from request_utils import _normalize_revision_type, _parse_revision_targets_form, infer_format_preferences
from resume_common import _clone_json, _collect_substantive_changes
from resume_io import (
    _extract_avatar_from_file_path,
    _extract_avatar_from_upload_bytes,
    extract_text_from_bytes,
    resolve_resume_text,
)
from resume_optimization import revise_resume_data, run_single_optimize_with_audit_pass
from resume_parsing import (
    _log_parse_text_debug,
    _log_resume_data_debug,
    resume_data_to_text,
    structured_resume_from_text,
)
from resume_renderer import export_resume_files
from resume_scoring import score_resume
from resume_validator import check_fabrication_heuristic, check_required_fields, check_sort_order, check_time_conflicts
from schemas import AuditAndOptimizeRequest, AuditAndOptimizeResponse
from server_runtime import DEFAULT_TEMPLATE, DRAFTS_DIR, FIXED_OUTPUT_FORMAT, MAX_FILE_SIZE, OUTPUT_DIR, logger


def _allowed_server_file(value: Optional[str], field: str) -> Optional[str]:
    if not value:
        return None
    path = Path(value).resolve()
    if not path.is_relative_to(OUTPUT_DIR.resolve()) or not path.is_file():
        raise HTTPException(status_code=403, detail=f"{field} must reference a generated output file")
    return str(path)


async def resume_audit_and_optimize_service(request: AuditAndOptimizeRequest) -> AuditAndOptimizeResponse:
    request_started = time.perf_counter()
    perf: dict[str, float] = {}

    def _add_stage(name: str, started_at: float) -> None:
        elapsed = max(0.0, time.perf_counter() - started_at)
        perf[name] = round(perf.get(name, 0.0) + elapsed, 3)

    style = request.style if request.style in {"conservative", "aggressive"} else "aggressive"
    revision_type = _normalize_revision_type(request.revision_type)
    requested_template = request.template or DEFAULT_TEMPLATE
    requested_output_format = FIXED_OUTPUT_FORMAT
    if request.revision_targets and not (request.revision_instructions and request.revision_instructions.strip()):
        raise HTTPException(status_code=400, detail="revision_instructions is required when revision_targets is provided")

    changes: list[dict[str, Any]] = []
    response_message: Optional[str] = None

    if request.resume_content or request.file_path:
        did_content_revision = False
        current_audit: Optional[dict[str, Any]] = None
        t_stage = time.perf_counter()
        safe_file_path = _allowed_server_file(request.file_path, "file_path")
        resume_text = resolve_resume_text(request.resume_content, safe_file_path)
        _add_stage("resolve_resume_text_s", t_stage)
        avatar_path = _allowed_server_file(str(request.avatar_path or "").strip() or None, "avatar_path")
        if not avatar_path and safe_file_path:
            t_avatar = time.perf_counter()
            avatar_path = _extract_avatar_from_file_path(safe_file_path)
            _add_stage("avatar_extract_s", t_avatar)
        _log_parse_text_debug(
            stage="pipeline_input_text",
            resume_text=resume_text,
            extra={
                "has_resume_content": bool(request.resume_content),
                "has_file_path": bool(request.file_path),
                "requested_output_format": requested_output_format,
                "requested_template": requested_template,
                "pipeline_mode": "single_pass",
                "has_avatar": bool(avatar_path),
            },
        )

        t_stage = time.perf_counter()
        resume_data = structured_resume_from_text(resume_text)
        _add_stage("structured_resume_s", t_stage)
        if isinstance(resume_data, dict):
            if avatar_path:
                meta = resume_data.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                    resume_data["meta"] = meta
                if not str(meta.get("avatar_path") or "").strip():
                    meta["avatar_path"] = avatar_path
            _log_resume_data_debug(
                stage="structured_resume_output",
                resume_data=resume_data,
                extra={
                    "is_empty": not bool(resume_data),
                    "has_avatar": bool(avatar_path),
                },
            )

        # Validate resume early
        t_stage = time.perf_counter()
        missing_fields = check_required_fields(resume_data, user_stage=request.user_stage)
        time_conflicts = check_time_conflicts(resume_data)
        sort_conflicts = check_sort_order(resume_data) if isinstance(resume_data, dict) else []
        all_conflicts = time_conflicts + sort_conflicts
        fab_report = check_fabrication_heuristic(resume_text, resume_data) if isinstance(resume_data, dict) else None
        _add_stage("validation_s", t_stage)
        if not resume_data:
            t_stage = time.perf_counter()
            current_audit = _audit_fallback(
                resume_text,
                request.jd_text,
                "resume_audit_and_optimize: structured resume parsing returned empty",
            )
            _add_stage("audit_heuristic_s", t_stage)
            score = round(float(current_audit.get("overall_score", 0.0)), 1)
            user_report = _build_user_report(
                resume_data=resume_data if isinstance(resume_data, dict) else {},
                audit_report=current_audit,
                changes=[],
                jd_text=request.jd_text,
                missing_fields=missing_fields,
                time_conflicts=all_conflicts,
                fab_report=fab_report,
            )
            mcp_tools_instruction = _build_mcp_tools_instruction(
                files={"docx": None, "pdf": None},
                changes=[],
                user_report=user_report,
                draft_id=request.draft_id or "",
            )
            substantive_changes: list[dict[str, Any]] = []
            perf["total_s"] = round(max(0.0, time.perf_counter() - request_started), 3)
            return AuditAndOptimizeResponse(
                files={"docx": None, "pdf": None},
                audit_report=current_audit,
                optimization_history=[
                    {
                        "round": 0,
                        "score": score,
                        "issues_fixed": 0,
                        "issues_remaining": len(current_audit.get("issues", [])),
                    }
                ],
                final_score=score,
                draft_id=request.draft_id or "",
                version=0,
                changes=[],
                has_substantive_rewrite=bool(substantive_changes),
                substantive_change_count=len(substantive_changes),
                user_report=user_report,
                mcp_tools_instruction=mcp_tools_instruction,
                perf=perf,
                message="简历结构化解析失败，已返回原始文本；本次无法生成 PDF/DOCX。",
                plain_text_output=resume_text,
            )

        requested_template, _ = infer_format_preferences(
            request.revision_instructions,
            requested_template,
            requested_output_format,
        )

        if revision_type in {"content", "both"} and request.revision_instructions and request.revision_instructions.strip():
            t_stage = time.perf_counter()
            resume_data, revision_changes = revise_resume_data(
                resume_data=resume_data,
                revision_instructions=request.revision_instructions.strip(),
                jd_text=request.jd_text,
                revision_targets=request.revision_targets,
            )
            _add_stage("revision_s", t_stage)
            changes.extend(revision_changes)
            did_content_revision = True
            t_stage = time.perf_counter()
            current_audit = audit_resume_core(resume_data_to_text(resume_data), request.jd_text, resume_data=resume_data)
            _add_stage("audit_after_revision_s", t_stage)
        elif revision_type == "format" and request.revision_instructions and request.revision_instructions.strip():
            changes.append(
                {
                    "location": "format",
                    "before": f"template={request.template or DEFAULT_TEMPLATE}, output_format={FIXED_OUTPUT_FORMAT}",
                    "after": f"template={requested_template}, output_format={requested_output_format}",
                    "reason": request.revision_instructions.strip()[:160],
                }
            )

        use_targeted_revision = bool(request.revision_targets)
        fallback_to_audit_only = False
        if revision_type == "format" or use_targeted_revision or did_content_revision:
            if not isinstance(current_audit, dict):
                t_stage = time.perf_counter()
                current_audit = audit_resume_core(resume_data_to_text(resume_data), request.jd_text, resume_data=resume_data)
                _add_stage("audit_initial_s", t_stage)
            history = [
                {
                    "round": 0,
                    "score": round(float(current_audit.get("overall_score", 0.0)), 1),
                    "issues_fixed": 0,
                    "issues_remaining": len(current_audit.get("issues", [])),
                }
            ]
        else:
            source_resume_data = _clone_json(resume_data)
            t_stage = time.perf_counter()
            try:
                resume_data, current_audit, history, pass_changes, opt_warnings = run_single_optimize_with_audit_pass(
                    resume_data=resume_data,
                    jd_text=request.jd_text,
                    style=style,
                    source_resume_data=source_resume_data,
                )
                if pass_changes:
                    changes.extend(pass_changes)
                if opt_warnings:
                    if response_message:
                        response_message += "；" + "；".join(opt_warnings)
                    else:
                        response_message = "；".join(opt_warnings)
            except HTTPException as exc:
                if exc.status_code >= 500 and "LLM optimization failed" in str(exc.detail):
                    logger.warning("LLM optimization failed, degrade to audit-only result: %s", exc.detail)
                    if not isinstance(current_audit, dict):
                        t_audit = time.perf_counter()
                        current_audit = audit_resume_core(resume_data_to_text(resume_data), request.jd_text)
                        _add_stage("audit_initial_s", t_audit)
                    history = [
                        {
                            "round": 0,
                            "score": round(float(current_audit.get("overall_score", 0.0)), 1),
                            "issues_fixed": 0,
                            "issues_remaining": len(current_audit.get("issues", [])),
                        }
                    ]
                    response_message = "优化阶段调用模型失败，已降级返回审计结果；请稍后重试优化。"
                    fallback_to_audit_only = True
                else:
                    raise
            _add_stage("optimize_single_pass_s", t_stage)

        substantive_changes = _collect_substantive_changes(changes)
        issue_count = len(current_audit.get("issues", [])) if isinstance(current_audit, dict) else 0
        if not fallback_to_audit_only and style == "aggressive" and issue_count >= 3 and len(substantive_changes) < 2:
            fallback_to_audit_only = True
            response_message = (
                "本轮已完成审计，但未产生足够的实质改写；已按诊断结果返回，未输出伪优化文件。"
            )

        if fallback_to_audit_only:
            file_map = {"docx": None, "pdf": None}
        else:
            try:
                t_stage = time.perf_counter()
                file_map = export_resume_files(
                    resume_data=resume_data,
                    output_dir=OUTPUT_DIR,
                    output_format=requested_output_format,
                    template=requested_template,
                )
                _add_stage("export_files_s", t_stage)
            except ValueError as exc:
                logger.exception(
                    "Export failed with ValueError | output_format=%s template=%s draft_id=%s",
                    requested_output_format,
                    requested_template,
                    request.draft_id,
                )
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        t_stage = time.perf_counter()
        if request.draft_id:
            draft_state = load_draft_state(DRAFTS_DIR, request.draft_id)
            draft_id, version = append_draft_version(
                drafts_dir=DRAFTS_DIR,
                state=draft_state,
                resume_data=resume_data,
                audit_report=current_audit,
                jd_text=request.jd_text,
                template=requested_template,
                output_format=requested_output_format,
                changes=changes,
            )
        else:
            draft_id, version = create_new_draft(
                drafts_dir=DRAFTS_DIR,
                resume_data=resume_data,
                audit_report=current_audit,
                jd_text=request.jd_text,
                template=requested_template,
                output_format=requested_output_format,
                changes=changes,
            )
        _add_stage("draft_persist_s", t_stage)

        perf["total_s"] = round(max(0.0, time.perf_counter() - request_started), 3)
        user_report = _build_user_report(
            resume_data=resume_data,
            audit_report=current_audit,
            changes=changes,
            jd_text=request.jd_text,
            missing_fields=missing_fields,
            time_conflicts=all_conflicts,
            fab_report=fab_report,
        )
        mcp_tools_instruction = _build_mcp_tools_instruction(
            files=file_map,
            changes=changes,
            user_report=user_report,
            draft_id=draft_id,
        )

        return AuditAndOptimizeResponse(
            files=file_map,
            audit_report=current_audit,
            optimization_history=history,
            final_score=round(float(current_audit.get("overall_score", 0.0)), 1),
            draft_id=draft_id,
            version=version,
            changes=changes,
            has_substantive_rewrite=bool(substantive_changes),
            substantive_change_count=len(substantive_changes),
            user_report=user_report,
            mcp_tools_instruction=mcp_tools_instruction,
            perf=perf,
            message=response_message,
        )

    if not request.draft_id:
        raise HTTPException(status_code=400, detail="Provide resume_content/file_path for new run, or draft_id for revision")

    t_stage = time.perf_counter()
    draft_state = load_draft_state(DRAFTS_DIR, request.draft_id)
    _add_stage("draft_load_s", t_stage)
    versions = draft_state.get("versions", [])
    if not versions:
        raise HTTPException(status_code=500, detail="draft has no versions")

    latest = versions[-1]
    resume_data = latest.get("resume_data") if isinstance(latest, dict) else None
    if not isinstance(resume_data, dict):
        raise HTTPException(status_code=500, detail="draft resume_data is invalid")

    latest_jd = latest.get("jd_text") if isinstance(latest, dict) else None
    effective_jd = request.jd_text if request.jd_text is not None else latest_jd

    # Load or recompute validation data from draft
    missing_fields: list = latest.get("validation", {}).get("missing_fields", []) if isinstance(latest, dict) else []
    time_conflicts: list = latest.get("validation", {}).get("conflicts", []) if isinstance(latest, dict) else []
    fab_report_data = latest.get("validation", {}).get("fabrication_report") if isinstance(latest, dict) else None

    current_audit = latest.get("audit_report") if isinstance(latest, dict) else None
    needs_audit_refresh = not isinstance(current_audit, dict) or _jd_changed(effective_jd, latest_jd)
    if needs_audit_refresh:
        t_stage = time.perf_counter()
        current_audit = audit_resume_core(resume_data_to_text(resume_data), effective_jd)
        _add_stage("audit_initial_s", t_stage)
    effective_template = request.template or latest.get("template") or DEFAULT_TEMPLATE
    effective_output_format = FIXED_OUTPUT_FORMAT

    effective_template, _ = infer_format_preferences(
        request.revision_instructions,
        effective_template,
        effective_output_format,
    )

    history: list[dict[str, Any]] = []
    did_content_revision = False

    if revision_type in {"content", "both"} and request.revision_instructions and request.revision_instructions.strip():
        t_stage = time.perf_counter()
        resume_data, revision_changes = revise_resume_data(
            resume_data=resume_data,
            revision_instructions=request.revision_instructions.strip(),
            jd_text=effective_jd,
            revision_targets=request.revision_targets,
        )
        _add_stage("revision_s", t_stage)
        changes.extend(revision_changes)
        did_content_revision = True
        if request.revision_targets:
            t_stage = time.perf_counter()
            current_audit = audit_resume_core(resume_data_to_text(resume_data), effective_jd, resume_data=resume_data)
            _add_stage("audit_after_revision_s", t_stage)
        else:
            t_stage = time.perf_counter()
            current_audit = audit_resume_core(resume_data_to_text(resume_data), effective_jd, resume_data=resume_data)
            _add_stage("audit_after_revision_s", t_stage)
        history = [
            {
                "round": 0,
                "score": round(float(current_audit.get("overall_score", 0.0)), 1),
                "issues_fixed": 0,
                "issues_remaining": len(current_audit.get("issues", [])),
            }
        ]
    else:
        history = [
            {
                "round": 0,
                "score": round(float(current_audit.get("overall_score", 0.0)), 1),
                "issues_fixed": 0,
                "issues_remaining": len(current_audit.get("issues", [])),
            }
        ]
        if request.revision_instructions and request.revision_instructions.strip():
            changes.append(
                {
                    "location": "format",
                    "before": f"template={latest.get('template', DEFAULT_TEMPLATE)}, output_format={FIXED_OUTPUT_FORMAT}",
                    "after": f"template={effective_template}, output_format={effective_output_format}",
                    "reason": request.revision_instructions.strip()[:160],
                }
            )

    try:
        t_stage = time.perf_counter()
        file_map = export_resume_files(
            resume_data=resume_data,
            output_dir=OUTPUT_DIR,
            output_format=effective_output_format,
            template=effective_template,
        )
        _add_stage("export_files_s", t_stage)
    except ValueError as exc:
        logger.exception(
            "Export failed with ValueError | output_format=%s template=%s draft_id=%s",
            effective_output_format,
            effective_template,
            request.draft_id,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    t_stage = time.perf_counter()
    draft_id, version = append_draft_version(
        drafts_dir=DRAFTS_DIR,
        state=draft_state,
        resume_data=resume_data,
        audit_report=current_audit,
        jd_text=effective_jd,
        template=effective_template,
        output_format=effective_output_format,
        changes=changes,
    )
    _add_stage("draft_persist_s", t_stage)
    perf["total_s"] = round(max(0.0, time.perf_counter() - request_started), 3)
    user_report = _build_user_report(
        resume_data=resume_data,
        audit_report=current_audit,
        changes=changes,
        jd_text=effective_jd,
        missing_fields=missing_fields,
        time_conflicts=time_conflicts,
        fab_report=fab_report,
    )
    mcp_tools_instruction = _build_mcp_tools_instruction(
        files=file_map,
        changes=changes,
        user_report=user_report,
        draft_id=draft_id,
    )
    substantive_changes = _collect_substantive_changes(changes)

    return AuditAndOptimizeResponse(
        files=file_map,
        audit_report=current_audit,
        optimization_history=history,
        final_score=round(float(current_audit.get("overall_score", 0.0)), 1),
        draft_id=draft_id,
        version=version,
        changes=changes,
        has_substantive_rewrite=bool(substantive_changes),
        substantive_change_count=len(substantive_changes),
        user_report=user_report,
        mcp_tools_instruction=mcp_tools_instruction,
        perf=perf,
    )


async def resume_audit_and_optimize_upload_service(
    resume_file: Optional[UploadFile] = None,
    file: Optional[UploadFile] = None,
    upload: Optional[UploadFile] = None,
    jd_text: Optional[str] = None,
    style: str = "aggressive",
    template: str = DEFAULT_TEMPLATE,
    draft_id: Optional[str] = None,
    revision_instructions: Optional[str] = None,
    revision_type: Literal["content", "format", "both"] = "both",
    revision_targets_json: Optional[str] = None,
) -> AuditAndOptimizeResponse:
    incoming_file = resume_file or file or upload
    if incoming_file is None:
        raise HTTPException(status_code=400, detail="Missing upload file: expected one of resume_file/file/upload")

    raw = await incoming_file.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large (> {MAX_FILE_SIZE // (1024 * 1024)} MB)")
    avatar_path = _extract_avatar_from_upload_bytes(raw, incoming_file.filename or "resume.pdf")
    resume_text = extract_text_from_bytes(raw, incoming_file.filename or "resume.pdf")
    _log_parse_text_debug(
        stage="upload_extracted_text",
        resume_text=resume_text,
        extra={
            "filename": incoming_file.filename or "",
            "size_bytes": len(raw),
            "output_format": FIXED_OUTPUT_FORMAT,
            "template": template,
            "pipeline_mode": "single_pass",
            "avatar_path": avatar_path or "",
        },
    )

    revision_targets = _parse_revision_targets_form(revision_targets_json)

    req = AuditAndOptimizeRequest(
        resume_content=resume_text,
        avatar_path=avatar_path,
        jd_text=jd_text,
        style=style,
        template=template,
        draft_id=draft_id,
        revision_instructions=revision_instructions,
        revision_type=revision_type,
        revision_targets=revision_targets,
    )
    try:
        return await resume_audit_and_optimize_service(req)
    except HTTPException as exc:
        logger.warning(
            "Upload pipeline failed | filename=%s size=%s output_format=%s template=%s revision_type=%s draft_id=%s detail=%s",
            incoming_file.filename or "",
            len(raw),
            FIXED_OUTPUT_FORMAT,
            template,
            revision_type,
            draft_id or "",
            exc.detail,
        )
        raise


async def download_output_file_service(path: str):
    file_path = Path(path).resolve()
    if not file_path.is_relative_to(OUTPUT_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Access denied: path outside output directory")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return file_path
