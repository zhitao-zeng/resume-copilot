"""Async resume generation API compatible with starting-kit evaluator.

Provides three endpoints:
  1. POST /resume_generate - receive async resume generation request
  2. GET /generate_progress/<task_id> - poll task status
  3. GET /download_resume/<task_id> - download generated docx
  4. GET /ready - health check

Runs on port 80 (or MOCK_SERVER_PORT env).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import shutil
import time
import threading
from pathlib import Path
from typing import Any

from flask import Flask, request, Response, send_file

from resume_copilot_service import resume_copilot_service
from resume_io import extract_text_from_bytes, IMAGE_EXTENSIONS
from server_runtime import OUTPUT_DIR, DEFAULT_TEMPLATE, MAX_FILE_SIZE, logger
from security_utils import safe_child_path, safe_filename, safe_task_id

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_REQUEST_SIZE_BYTES", str(64 * 1024 * 1024)))

# ── 压制 werkzeug 噪声(polling/health) ────────────────────────────
import logging as _logging
_werkzeug_log = _logging.getLogger("werkzeug")
_SILENT_PATHS = ("/optimize_progress", "/generate_progress", "/ready")
_orig_werkzeug_log = _werkzeug_log.info

def _quiet_werkzeug(msg, *args, **kwargs):
    if args and any(p in str(args[0]) for p in _SILENT_PATHS):
        return
    _orig_werkzeug_log(msg, *args, **kwargs)

_werkzeug_log.info = _quiet_werkzeug

# ── In-memory task store ────────────────────────────────────────────
async_tasks: dict[str, dict] = {}
task_lock = threading.Lock()

# ── Task queue (single worker to avoid thread/event-loop accumulation) ──
_task_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("TASK_MAX_WORKERS", "1")),
)
_task_futures: dict[str, concurrent.futures.Future] = {}
_task_futures_lock = threading.Lock()
_TASK_QUEUE_LIMIT = max(1, int(os.getenv("TASK_QUEUE_LIMIT", "8")))
_TASK_STATE_TTL = max(60, int(os.getenv("TASK_STATE_TTL_SECONDS", "86400")))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = OUTPUT_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

_json_logger = logging.getLogger("resume_api")


def _discard_future(task_id: str) -> None:
    with _task_futures_lock:
        _task_futures.pop(task_id, None)


def _cleanup_task_state() -> None:
    cutoff = time.time() - _TASK_STATE_TTL
    with task_lock:
        expired = [
            task_id for task_id, state in async_tasks.items()
            if state.get("finished") and state.get("end_time", state.get("start_time", 0)) < cutoff
        ]
        for task_id in expired:
            async_tasks.pop(task_id, None)


def _cleanup_saved_inputs(form_data: Any) -> None:
    for value in getattr(form_data, "values", lambda: [])():
        if not isinstance(value, str) or not value:
            continue
        try:
            path = Path(value).resolve()
            if path.is_relative_to(UPLOADS_DIR.resolve()) and path.is_file():
                path.unlink(missing_ok=True)
        except OSError:
            continue


@app.before_request
def _check_api_token():
    expected = os.getenv("API_AUTH_TOKEN", "").strip()
    if not expected or request.path == "/ready":
        return None
    supplied = request.headers.get("Authorization", "")
    if supplied != f"Bearer {expected}":
        return _json_response({"success": False, "message": "unauthorized"}, 401)
    return None


def _json_response(data: dict, status_code: int = 200) -> Response:
    return Response(
        __import__("json").dumps(data, ensure_ascii=False, indent=2),
        status=status_code,
        mimetype="application/json; charset=utf-8",
    )


def _save_upload_file(file_storage: Any, task_id: str, field_name: str) -> str | None:
    """Save an uploaded file and return its path, or None if absent.

    Handles both Flask FileStorage objects (legacy) and pre-saved path strings
    (from resume_generate route which persists files before spawning threads).
    """
    if file_storage is None:
        return None
    # Case 1: Already saved as a path string
    if isinstance(file_storage, str) and file_storage:
        if Path(file_storage).exists():
            return file_storage
        _json_logger.warning("Pre-saved %s file not found: %s", field_name, file_storage)
        return None
    # Case 2: Flask FileStorage
    if not getattr(file_storage, "filename", None):
        return None
    filename = safe_filename(file_storage.filename)
    save_path = safe_child_path(UPLOADS_DIR, f"{safe_task_id(task_id)}_{field_name}_{filename}")
    try:
        file_storage.save(str(save_path))
        if save_path.stat().st_size > MAX_FILE_SIZE:
            save_path.unlink(missing_ok=True)
            _json_logger.warning("Rejected oversized %s upload", field_name)
            return None
    except Exception as exc:
        _json_logger.warning("Failed to save upload %s: %s", field_name, exc)
        return None
    return str(save_path)


def _extract_file_field(form_data: dict, field_name: str, id_str: str) -> tuple[str | None, str | None]:
    """Extract a file field from multipart request.

    Returns (file_path_or_none, file_text_or_none).
    - If the uploaded file is text/image/PDF, returns (path, extracted_text).
    - If the form contains text content, returns (None, text_content).

    Handles both Flask FileStorage objects and pre-saved file path strings
    (the latter from resume_generate route which saves files before spawning threads).
    """
    file_storage = form_data.get(field_name)

    # Case 1: Pre-saved file path string (from resume_generate route)
    # Guard: text fields (JD content, query text) are also strings — only treat
    # as file path if it looks like one: short, no newlines, has a file extension,
    # and exists on disk.
    if isinstance(file_storage, str) and file_storage:
        _looks_like_path = (
            len(file_storage) < 500
            and "\n" not in file_storage
            and Path(file_storage).suffix.lower() in {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".txt", ".md"}
        )
        if not _looks_like_path:
            # Fall through to Case 3 (text content)
            if file_storage and len(file_storage) > 10:
                return None, file_storage
            return None, None
        try:
            save_path = Path(file_storage).resolve()
            if not save_path.is_relative_to(UPLOADS_DIR.resolve()):
                logger.warning("Rejected local path input for %s", field_name)
                return None, None
        except (OSError, ValueError):
            return None, None
        if not save_path.exists():
            logger.warning("Pre-saved file for %s not found: %s", field_name, file_storage)
            return None, None
        if save_path.stat().st_size > MAX_FILE_SIZE:
            logger.warning("Rejected oversized pre-saved %s upload", field_name)
            return None, None
        ext = save_path.suffix.lower()
        try:
            with open(save_path, "rb") as f:
                content = f.read()
            text = extract_text_from_bytes(content, save_path.name)
            return str(save_path), text
        except Exception as exc:
            logger.warning("Failed to extract text from pre-saved %s: %s", field_name, exc)
            return str(save_path), None

    # Case 2: Flask FileStorage object (legacy, only works in synchronous context)
    if hasattr(file_storage, "filename"):
        if not file_storage.filename:
            return None, None
        save_path = safe_child_path(
            UPLOADS_DIR,
            f"{safe_task_id(id_str)}_{field_name}_{safe_filename(file_storage.filename)}",
        )
        file_storage.save(str(save_path))
        if save_path.stat().st_size > MAX_FILE_SIZE:
            save_path.unlink(missing_ok=True)
            return None, None
        ext = Path(save_path).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            with open(save_path, "rb") as f:
                content = f.read()
            text = extract_text_from_bytes(content, save_path.name)
            return str(save_path), text
        try:
            with open(save_path, "rb") as f:
                content = f.read()
            text = extract_text_from_bytes(content, save_path.name)
            return str(save_path), text
        except Exception as exc:
            logger.warning("Failed to extract text from %s: %s", field_name, exc)
            return str(save_path), None

    # Case 3: Text content
    text = str(file_storage or "")
    if text and len(text) > 10:
        return None, text

    return None, None


def _wait_vllm_ready(max_wait: int = 900) -> float:
    """Block until vLLM /health returns 200, or timeout. Returns elapsed seconds."""
    import urllib.request
    import urllib.error
    started = time.time()
    for i in range(max_wait):
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=2)
            elapsed = time.time() - started
            _json_logger.info("vLLM ready after %.0fs", elapsed)
            return elapsed
        except Exception:
            if i == 0:
                _json_logger.info("Waiting for vLLM to be ready...")
            time.sleep(1)
    _json_logger.warning("vLLM not ready after %ds, proceeding anyway", max_wait)
    return time.time() - started


def _process_resume(task_id: str, form_data: Any) -> None:
    """Background task: call resume_copilot_service and store results."""
    start_time = time.time()

    with task_lock:
        async_tasks[task_id] = {
            "status": "processing",
            "finished": False,
            "summary": "",
            "file_path": "",
            "error": None,
            "start_time": start_time,
        }

    try:
        # Wait for vLLM if not ready yet (before extracting fields to avoid wasted work)
        _wait_vllm_ready()

        # Extract text fields
        query = str(form_data.get("Query") or form_data.get("query") or "")
        target_jd_text = str(form_data.get("target_jd") or "")

        # 诊断：列出 form_data 中所有 key 及是否文件
        _keys_summary = []
        for k in form_data:
            v = form_data[k]
            is_file = hasattr(v, "filename")
            _keys_summary.append(f"{k}=<file>" if is_file else f"{k}=<str:{len(str(v))}c>")
        _json_logger.info("[%s] form keys: %d fields, %s", task_id, len(form_data), ", ".join(_keys_summary) if _keys_summary else "(none)")

        # Extract files
        cv_path, cv_text = _extract_file_field(form_data, "cv", task_id)
        cv_template_path = _save_upload_file(form_data.get("cv_template"), task_id, "cv_template")
        jd_path, jd_text = _extract_file_field(form_data, "target_jd", task_id)

        # Use extracted text from file first (jd_text), fall back to raw form field
        # (target_jd_text may be a file path after pre-save, not actual JD content)
        final_jd = jd_text or target_jd_text or ""
        final_cv = cv_text

        # Build mock UploadFile-like objects for resume_copilot_service
        from http_compat import HTTPException
        from resume_copilot_service import _ensure_time_budget

        class _MockUpload:
            def __init__(self, path: str | None, name: str = "", content: str | None = None):
                self.path = path
                self.content = content
                self.filename = Path(path).name if path else name
                self.content_type = None

            async def read(self) -> bytes:
                if self.content is not None:
                    return self.content.encode("utf-8")
                if not self.path:
                    return b""
                with open(self.path, "rb") as f:
                    return f.read()

        cv_upload = _MockUpload(cv_path, "cv") if cv_path else None

        # Fallback: if cv was sent as a text field (JSON mode or form text field)
        # rather than a file upload. Two sub-cases:
        #   1. cv_text is a real file path string  → read that file
        #   2. cv_text is the CV content itself    → use it directly as text CV
        # Guard the path check: CV text can be long; calling .exists() on it
        # raises FileNameTooLong (OSError 36), so only stat short path-like strings.
        if cv_upload is None and cv_text and len(cv_text) > 5:
            _cv_candidate = cv_text.strip()
            _looks_like_path = (
                len(_cv_candidate) < 500
                and "\n" not in _cv_candidate
                and Path(_cv_candidate).suffix.lower() in {".pdf", ".docx", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".txt", ".md"}
            )
            if _looks_like_path:
                _cv_text_path = Path(_cv_candidate)
                try:
                    _is_cv_file = _cv_text_path.exists() and _cv_text_path.is_file()
                except OSError:
                    _is_cv_file = False
                if _is_cv_file:
                    _json_logger.info("[%s] cv received as persisted upload", task_id)
                    cv_upload = _MockUpload(str(_cv_text_path), _cv_text_path.name)
                    cv_path = str(_cv_text_path)  # for logging
                else:
                    _json_logger.warning("[%s] cv_text=%s does not exist as file, cv_upload=None", task_id, cv_text[:100])
            else:
                _json_logger.info("[%s] cv received as plain text content (%d chars), using directly", task_id, len(cv_text))
                cv_upload = _MockUpload(None, "cv.txt", content=cv_text)

        cv_template_upload = _MockUpload(cv_template_path, "template") if cv_template_path else None
        jd_upload = _MockUpload(jd_path, "target_jd") if jd_path else None

        # JD as URL or text or file
        import re as _re
        _is_url = _re.match(r"^https?://", final_jd.strip(), _re.IGNORECASE) if final_jd else False
        target_jd_url = final_jd if _is_url else None
        jd_text_value = final_jd if final_jd and not _is_url else None

        # Run async service in event loop
        _json_logger.info(
            "[%s] calling resume_copilot_service | cv_upload=%s | cv_path=%s | has_jd=%s | query_chars=%d",
            task_id,
            "present" if cv_upload is not None else "NONE",
            "present" if cv_path else "N/A",
            bool(jd_upload is not None or jd_text_value or target_jd_url),
            len(query or ""),
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                resume_copilot_service(
                    query=query,
                    cv=cv_upload,
                    cv_template=cv_template_upload,
                    target_jd=None,
                    target_jd_file=jd_upload,
                    target_jd_url=target_jd_url,
                    jd_text=jd_text_value,
                    jd_url=None,
                    template=DEFAULT_TEMPLATE,
                )
            )
        finally:
            loop.close()

        elapsed = round(time.time() - start_time, 3)
        score_val = getattr(response, "score", "?")
        _json_logger.info("Task %s completed in %.1fs (score=%s)", task_id, elapsed, score_val)

        # Save generated docx
        files = getattr(response, "files", {}) or {}
        docx_path = files.get("docx", "")
        if docx_path and Path(docx_path).exists():
            dest = OUTPUT_DIR / f"{task_id}.docx"
            shutil.copy2(docx_path, dest)
            file_path = str(dest)
        else:
            file_path = ""

        summary = getattr(response, "reply_text", "") or "简历生成完成"

        with task_lock:
            async_tasks[task_id] = {
                "status": "done",
                "finished": True,
                "summary": summary,
                "file_path": file_path,
                "error": None,
                "start_time": start_time,
                "end_time": time.time(),
            }
    except Exception as exc:
        elapsed = round(time.time() - start_time, 3)
        _json_logger.error("Task %s failed after %.1fs: %s", task_id, elapsed, exc, exc_info=True)
        error_msg = str(exc)
        if len(error_msg) > 500:
            error_msg = error_msg[:500] + "..."

        with task_lock:
            async_tasks[task_id] = {
                "status": "error",
                "finished": True,
                "summary": f"生成失败: {error_msg}",
                "file_path": "",
                "error": error_msg,
                "start_time": start_time,
                "end_time": time.time(),
            }
    finally:
        _cleanup_saved_inputs(form_data)


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/ready", methods=["GET"])
def ready():
    """Health check probe endpoint."""
    return "True"


@app.route("/resume_generate", methods=["POST"])
@app.route("/resume_optimize", methods=["POST"])   # 兼容旧版评测端
def resume_generate():
    """Receive resume generation request, start background processing."""
    try:
        _cleanup_task_state()
        task_id = "unknown"
        data: Any = {}

        if request.content_type and "multipart/form-data" in request.content_type:
            task_id = safe_task_id(request.form.get("id"))
            # request.form only contains text fields; file uploads are in request.files
            data = dict(request.form)
            for key in request.files:
                data[key] = request.files[key]
        else:
            # JSON mode
            data = request.get_json(silent=True) or {}
            if not isinstance(data, dict):
                return _json_response({"success": False, "message": "请求数据必须是 dict 格式"}, 400)
            task_id = safe_task_id(data.get("id"))

        # Normalize task_id
        task_id = safe_task_id(task_id)

        with task_lock:
            existing = async_tasks.get(task_id)
            if existing and not existing.get("finished", False):
                return _json_response({"success": False, "message": "task id is already active"}, 409)

        # ── Save files to disk BEFORE spawning background thread ──
        # Flask/Werkzeug stores uploaded files as temp files; those are cleaned up
        # when the request returns. The background thread runs after the response
        # is sent, so we must persist files now and pass paths instead of FileStorage.
        for key, val in list(data.items()):
            if hasattr(val, "filename") and getattr(val, "filename", None):
                save_path = safe_child_path(
                    UPLOADS_DIR,
                    f"{task_id}_{safe_filename(key, 'file')}_{safe_filename(val.filename)}",
                )
                val.save(str(save_path))
                if save_path.stat().st_size > MAX_FILE_SIZE:
                    save_path.unlink(missing_ok=True)
                    _cleanup_saved_inputs(data)
                    return _json_response({"success": False, "message": "uploaded file is too large"}, 413)
                data[key] = str(save_path)  # Replace FileStorage with persisted path
                _json_logger.info("[%s] saved upload field=%s bytes=%d", task_id, key, save_path.stat().st_size)

        # Register task
        with task_lock:
            async_tasks[task_id] = {
                "status": "processing",
                "finished": False,
                "summary": "",
                "file_path": "",
                "error": None,
                "start_time": time.time(),
            }

        # Log received fields
        query = data.get("Query", data.get("query", ""))
        _json_logger.info("[%s] 收到简历生成请求: query_chars=%d", task_id, len(str(query)))

        # Submit to thread pool (max_workers=1 prevents thread accumulation)
        _task_id = task_id
        with _task_futures_lock:
            active = sum(1 for item in _task_futures.values() if not item.done())
            if active >= _TASK_QUEUE_LIMIT:
                with task_lock:
                    async_tasks.pop(_task_id, None)
                _cleanup_saved_inputs(data)
                return _json_response({"success": False, "message": "任务队列已满，请稍后重试"}, 429)
            future = _task_executor.submit(_process_resume, _task_id, data)
            _task_futures[_task_id] = future
        future.add_done_callback(lambda _f, tid=_task_id: _discard_future(tid))

        return _json_response({
            "success": True,
            "message": "简历生成请求已接收，正在异步处理中",
            "task_id": task_id,
        })

    except ValueError as exc:
        return _json_response({"success": False, "message": str(exc)}, 400)
    except Exception as exc:
        _json_logger.error("resume_generate error: %s", exc, exc_info=True)
        return _json_response({"success": False, "message": "internal server error"}, 500)


@app.route("/generate_progress/<task_id>", methods=["GET"])
@app.route("/optimize_progress/<task_id>", methods=["GET"])  # 兼容旧版评测端
def generate_progress(task_id: str):
    """Query resume generation progress and summary."""
    _cleanup_task_state()
    try:
        task_id = safe_task_id(task_id)
    except ValueError:
        return _json_response({"success": False, "message": "invalid task id"}, 400)
    with task_lock:
        status = async_tasks.get(task_id, {})

    return _json_response({
        "success": True,
        "task_id": task_id,
        "finished": status.get("finished", False),
        "summary": status.get("summary", ""),
    })


@app.route("/download_resume/<task_id>", methods=["GET"])
@app.route("/optimize_download/<task_id>", methods=["GET"])  # 兼容旧版评测端
def download_resume(task_id: str):
    """Download the generated resume file (docx)."""
    try:
        task_id = safe_task_id(task_id)
    except ValueError:
        return _json_response({"success": False, "message": "invalid task id"}, 400)
    with task_lock:
        status = async_tasks.get(task_id, {})
        file_path = status.get("file_path", "")

    # Fallback: check OUTPUT_DIR directly
    if not file_path:
        fallback = safe_child_path(OUTPUT_DIR, f"{task_id}.docx")
        if fallback.exists():
            file_path = str(fallback)

    if not file_path or not Path(file_path).exists():
        return _json_response(
            {"success": False, "message": f"简历文件不存在: {task_id}"},
            404,
        )

    try:
        return send_file(
            file_path,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            as_attachment=True,
            download_name=f"{task_id}_resume.docx",
        )
    except Exception as exc:
        _json_logger.error("download_resume %s error: %s", task_id, exc)
        return _json_response({"success": False, "message": str(exc)}, 500)


if __name__ == "__main__":
    port = int(os.getenv("MOCK_SERVER_PORT", os.getenv("PORT", "80")))
    print("=" * 60)
    print("简历助手异步服务端启动 (starting-kit 兼容)")
    print("=" * 60)
    print("接口:")
    print("  POST /resume_generate          - 接收简历生成请求")
    print("  GET  /generate_progress/<id>   - 查询生成进度")
    print("  GET  /download_resume/<id>     - 下载生成的 docx 简历")
    print("  GET  /ready                    - 健康检查")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
