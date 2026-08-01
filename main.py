import os
import sys
import asyncio
import hmac
from datetime import datetime
from typing import Any, Optional

from pathlib import Path

# Keep the repository runnable without a caller-specific PYTHONPATH while the
# historical core modules are migrated into a conventional package.
CORE_DIR = Path(__file__).resolve().parent / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import aiohttp
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.wsgi import WSGIMiddleware

from resume_copilot_service import resume_copilot_service
from resume_service import (
    download_output_file_service,
    resume_audit_and_optimize_service,
    resume_audit_and_optimize_upload_service,
)
from resume_scoring import score_resume
from schemas import (
    AuditAndOptimizeRequest,
    AuditAndOptimizeResponse,
    GenerateResponse,
    HealthResponse,
    ResumeCopilotResponse,
    ScoreRequest,
    ScoreResponse,
)
from server_runtime import (
    API_BASE_URL,
    DEFAULT_TEMPLATE,
    MODEL_NAME,
    REQUEST_TIMEOUT_SECONDS,
    llm_enabled,
    logger,
)

app = FastAPI(title="Resume Audit & Optimize API", version="3.1.0")
cors_origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "*").split(",") if item.strip()]
MAX_REQUEST_SIZE_BYTES = int(os.getenv("MAX_REQUEST_SIZE_BYTES", str(64 * 1024 * 1024)))
REQUEST_CONCURRENCY = max(1, int(os.getenv("REQUEST_CONCURRENCY", "2")))
_request_slots = asyncio.Semaphore(REQUEST_CONCURRENCY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def authenticate_requests(request: Request, call_next):
    content_length = request.headers.get("Content-Length", "")
    if content_length.isdigit() and int(content_length) > MAX_REQUEST_SIZE_BYTES:
        return JSONResponse(status_code=413, content={"detail": "request body is too large"})
    expected = os.getenv("API_AUTH_TOKEN", "").strip()
    if expected and request.url.path not in {"/ready", "/health"}:
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {expected}"):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return await call_next(request)


async def _backend_ready() -> bool:
    if not llm_enabled():
        return False
    try:
        url = f"{API_BASE_URL.rstrip('/')}/models"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
            async with session.get(url) as response:
                return response.status == 200
    except Exception:
        return False


async def _run_copilot_with_timeout(**kwargs):
    try:
        await asyncio.wait_for(_request_slots.acquire(), timeout=0.1)
    except TimeoutError as exc:
        raise HTTPException(status_code=429, detail="server is at generation capacity; retry later") from exc
    try:
        async with asyncio.timeout(REQUEST_TIMEOUT_SECONDS):
            return await resume_copilot_service(**kwargs)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"resume-copilot exceeded the {REQUEST_TIMEOUT_SECONDS}s request budget",
        ) from exc
    finally:
        _request_slots.release()


@app.exception_handler(HTTPException)
async def http_exception_logger(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning(
        "HTTPException | method=%s path=%s status=%s detail=%s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": jsonable_encoder(exc.detail)},
        headers=exc.headers or None,
    )


@app.get("/ready")
async def readiness_probe() -> Response:
    """K8s readiness probe that verifies the configured LLM backend."""
    if not await _backend_ready():
        return Response("False", status_code=503, media_type="text/plain")
    return Response("True", media_type="text/plain")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="resume-audit-optimize",
        version="3.1.0",
        model=MODEL_NAME,
        llm_enabled=llm_enabled(),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/models")
async def list_models() -> dict[str, Any]:
    return {
        "models": [
            {
                "id": MODEL_NAME,
                "type": "text-generation",
                "enabled": llm_enabled(),
            }
        ],
    }


@app.post("/audit-and-optimize", response_model=AuditAndOptimizeResponse)
async def resume_audit_and_optimize(request: AuditAndOptimizeRequest) -> AuditAndOptimizeResponse:
    return await resume_audit_and_optimize_service(request)


@app.post("/audit-and-optimize/upload", response_model=AuditAndOptimizeResponse)
async def resume_audit_and_optimize_upload(
    resume_file: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    upload: Optional[UploadFile] = File(None),
    jd_text: Optional[str] = Form(None),
    style: str = Form("aggressive"),
    template: str = Form(DEFAULT_TEMPLATE),
    draft_id: Optional[str] = Form(None),
    revision_instructions: Optional[str] = Form(None),
    revision_type: str = Form("both"),
    revision_targets_json: Optional[str] = Form(None),
) -> AuditAndOptimizeResponse:
    return await resume_audit_and_optimize_upload_service(
        resume_file=resume_file,
        file=file,
        upload=upload,
        jd_text=jd_text,
        style=style,
        template=template,
        draft_id=draft_id,
        revision_instructions=revision_instructions,
        revision_type=revision_type,
        revision_targets_json=revision_targets_json,
    )


@app.post("/resume-copilot", response_model=ResumeCopilotResponse)
async def resume_copilot(
    query: Optional[str] = Form(None, description="用户请求的文本内容"),
    cv: Optional[UploadFile] = File(None, description="个人简历文件：PDF/Word/图片"),
    cv_template: Optional[UploadFile] = File(None, description="目标格式简历模板文件"),
    target_jd: Optional[str] = Form(None, description="目标岗位 JD 文本或链接"),
    target_jd_file: Optional[UploadFile] = File(None, description="目标岗位 JD 文件：PDF/Word/图片"),
    target_jd_url: Optional[str] = Form(None, description="目标岗位 JD 链接"),
    jd_text: Optional[str] = Form(None, description="兼容字段：目标岗位 JD 文本"),
    jd_url: Optional[str] = Form(None, description="兼容字段：目标岗位 JD 链接"),
    template: str = Form(DEFAULT_TEMPLATE),
) -> ResumeCopilotResponse:
    return await _run_copilot_with_timeout(
        query=query,
        cv=cv,
        cv_template=cv_template,
        target_jd=target_jd,
        target_jd_file=target_jd_file,
        target_jd_url=target_jd_url,
        jd_text=jd_text,
        jd_url=jd_url,
        template=template,
    )


@app.post("/generate")
async def generate_resume(
    personal_profile: str = Form(..., description="Personal profile text"),
    jd_text: Optional[str] = Form(None, description="Job description text"),
    jd_url: Optional[str] = Form(None, description="Job description URL"),
    job_family: Optional[str] = Form(None, description="Optional job family override"),
    target_description: Optional[str] = Form(None, description="Target job description"),
    user_stage: Optional[str] = Form(None, description="User stage: student/experienced/job_seeker"),
    cv_template: Optional[UploadFile] = File(None, description="Custom resume template"),
    template: str = Form(DEFAULT_TEMPLATE),
) -> GenerateResponse:
    query_parts = [personal_profile]
    if target_description:
        query_parts.append(f"目标说明：{target_description}")
    if user_stage:
        query_parts.append(f"用户阶段：{user_stage}")
    if job_family:
        query_parts.append(f"目标行业：{job_family}")
    response = await _run_copilot_with_timeout(
        query="\n".join(part for part in query_parts if part),
        cv=None,
        cv_template=cv_template,
        target_jd=jd_text,
        target_jd_file=None,
        target_jd_url=jd_url,
        jd_text=None,
        jd_url=None,
        template=template,
    )
    return GenerateResponse(
        resume_data=response.resume_data,
        files=response.files,
        score=response.score,
        score_breakdown=response.score_breakdown,
        draft_id=response.draft_id,
        version=response.version,
        missing_fields=response.missing_fields,
        conflicts=response.conflicts,
        fabrication_report={"fabrication_found": bool(response.user_report.get("fabrication_details")), "details": response.user_report.get("fabrication_details", [])},
        user_report=response.user_report,
        generation_direction=str(response.user_report.get("generation_direction", "")),
        reply_text=response.reply_text,
        perf=response.perf,
    )


@app.post("/score")
async def score_resume_endpoint(request: ScoreRequest) -> ScoreResponse:
    score = score_resume(
        resume_data=request.resume_data,
        original_text=request.original_text,
        user_report=request.user_report,
        job_family=request.job_family,
        user_stage=request.user_stage,
        missing_fields=request.missing_fields,
        conflicts=request.conflicts,
    )
    return ScoreResponse(
        fabrication=score.fabrication,
        readability=score.readability,
        completeness=score.completeness,
        expression=score.expression,
        response=score.response,
        total=score.total,
    )


@app.get("/files/download")
async def download_output_file(path: str = Query(..., description="Absolute path of the generated file")):
    file_path = await download_output_file_service(path)
    return FileResponse(file_path, filename=file_path.name)


# Mount evaluator-compatible Flask routes only as a legacy fallback. FastAPI
# routes above always win, and every port now runs through one ASGI server.
try:
    from resume_generate_api import app as legacy_flask_app
    app.mount("/", WSGIMiddleware(legacy_flask_app), name="legacy-evaluator-api")
except ImportError as exc:
    logger.warning("Legacy evaluator routes unavailable: %s", exc)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=port, log_level="info")
