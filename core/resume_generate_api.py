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
import multiprocessing as mp
import os
import re
import shutil
import signal
import secrets
import time
import threading
from pathlib import Path
from typing import Any

from flask import Flask, request, Response, send_file

from http_compat import HTTPException
from resume_copilot_service import resume_copilot_service
from resume_io import extract_text_from_bytes, IMAGE_EXTENSIONS
from server_runtime import (
    API_BASE_URL,
    OUTPUT_DIR,
    DEFAULT_TEMPLATE,
    MAX_FILE_SIZE,
    REQUEST_TIMEOUT_SECONDS,
    logger,
    reset_request_deadline,
    set_request_deadline,
)
from security_utils import safe_child_path, safe_filename, safe_task_id
from diagnostic_trace import reset_trace_id, set_trace_id, trace_event

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_REQUEST_SIZE_BYTES", str(64 * 1024 * 1024)))


class _MockUpload:
    """UploadFile-compatible wrapper with cursor-based reads.

    read(size) advances a cursor so read_upload_limited's chunked loop reads the
    underlying data exactly once. Previously this class ignored `size` and
    re-opened the file from byte 0 on every call; read_upload_limited then fell
    back to the no-arg read() each iteration, re-reading the whole file until
    the cumulative total tripped MAX_FILE_SIZE and every submission failed with
    "file too large" (a small 125KB CV, e.g., reported >20MB).
    """

    def __init__(self, path: str | None, name: str = "", content: str | None = None):
        self.path = path
        self.content = content
        self.filename = Path(path).name if path else name
        self.content_type = None
        self._offset = 0

    async def read(self, size: int | None = None) -> bytes:
        if self.content is not None:
            data = self.content.encode("utf-8")
            result = data[self._offset:] if size is None else data[self._offset:self._offset + size]
            self._offset += len(result)
            return result
        if not self.path:
            return b""
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            result = f.read() if size is None else f.read(size)
            self._offset += len(result)
            return result

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


def _positive_float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# ── Task queue (single worker to avoid thread/event-loop accumulation) ──
_task_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("TASK_MAX_WORKERS", "1")),
)
_task_futures: dict[str, tuple[str, concurrent.futures.Future]] = {}
_task_futures_lock = threading.Lock()
_task_timers: dict[str, tuple[str, threading.Timer]] = {}
_task_timers_lock = threading.Lock()
_TASK_QUEUE_LIMIT = max(1, int(os.getenv("TASK_QUEUE_LIMIT", "8")))
_TASK_STATE_TTL = max(60, int(os.getenv("TASK_STATE_TTL_SECONDS", "86400")))
_TASK_DEADLINE_SECONDS = max(
    1.0,
    min(
        _positive_float_env("TASK_DEADLINE_SECONDS", 475.0),
        max(1.0, float(REQUEST_TIMEOUT_SECONDS) - 5.0),
    ),
)
_TASK_FINALIZATION_RESERVE_SECONDS = min(
    max(0.0, _positive_float_env("TASK_FINALIZATION_RESERVE_SECONDS", 30.0)),
    max(0.0, _TASK_DEADLINE_SECONDS - 1.0),
)
_TASK_PROCESS_KILL_GRACE_SECONDS = max(
    0.2, _positive_float_env("TASK_PROCESS_KILL_GRACE_SECONDS", 1.5)
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = OUTPUT_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

_json_logger = logging.getLogger("resume_api")


def _llm_work_deadline(deadline_at: float) -> float:
    """Return the optional-work cutoff while preserving hard finalization time."""

    return float(deadline_at) - _TASK_FINALIZATION_RESERVE_SECONDS


def _discard_future(task_id: str, run_id: str) -> None:
    with _task_futures_lock:
        current = _task_futures.get(task_id)
        if current and current[0] == run_id:
            _task_futures.pop(task_id, None)
    with _task_timers_lock:
        current_timer = _task_timers.get(task_id)
        if current_timer and current_timer[0] == run_id:
            _task_timers.pop(task_id, None)
            current_timer[1].cancel()


def _task_is_active(task_id: str, run_id: str) -> bool:
    with task_lock:
        state = async_tasks.get(task_id)
        return bool(
            state
            and state.get("run_id") == run_id
            and not state.get("finished", False)
        )


def _publish_terminal_state(task_id: str, run_id: str, **updates: Any) -> bool:
    """Publish exactly one terminal result for the current incarnation."""

    with task_lock:
        state = async_tasks.get(task_id)
        if (
            not state
            or state.get("run_id") != run_id
            or state.get("finished", False)
        ):
            return False
        state.update(updates)
        state["finished"] = True
        state["end_time"] = time.time()
        return True


def _remove_task_state(task_id: str, run_id: str) -> None:
    """Remove only the exact task incarnation owned by the caller."""

    with task_lock:
        state = async_tasks.get(task_id)
        if state and state.get("run_id") == run_id:
            async_tasks.pop(task_id, None)


def _publish_success_file(
    task_id: str,
    run_id: str,
    *,
    temp_path: Path,
    final_path: Path,
    summary: str,
) -> bool:
    """Atomically publish both the DOCX and its successful task state."""

    with task_lock:
        state = async_tasks.get(task_id)
        if (
            not state
            or state.get("run_id") != run_id
            or state.get("finished", False)
        ):
            return False
        os.replace(temp_path, final_path)
        state.update(
            status="done",
            summary=summary,
            file_path=str(final_path),
            error=None,
            finished=True,
            end_time=time.time(),
        )
        return True


def _expire_task(task_id: str, run_id: str, form_data: Any) -> None:
    """Deadline callback for both queued and supervised running jobs."""

    published = _publish_terminal_state(
        task_id,
        run_id,
        status="error",
        summary=f"生成失败: 端到端处理超过 {_TASK_DEADLINE_SECONDS:.0f} 秒，任务已终止",
        file_path="",
        error="end-to-end task deadline exceeded",
    )
    if not published:
        return
    # Do not hold the mapping lock while cancelling: Future callbacks may
    # immediately re-enter _discard_future.
    with _task_futures_lock:
        current = _task_futures.get(task_id)
        future = current[1] if current and current[0] == run_id else None
    cancelled_before_start = bool(future and future.cancel())
    if cancelled_before_start:
        _cleanup_saved_inputs(form_data)


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
        except HTTPException:
            raise
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
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Failed to extract text from %s: %s", field_name, exc)
            return str(save_path), None

    # Case 3: Text content
    text = str(file_storage or "")
    if text and len(text) > 10:
        return None, text

    return None, None


def _wait_vllm_ready(max_wait: float = 15.0, *, deadline_at: float | None = None) -> float:
    """Wait briefly for vLLM, bounded by the end-to-end task deadline."""
    import urllib.request
    from urllib.parse import urlsplit, urlunsplit

    configured_health_url = os.getenv("MODELHUB_HEALTH_URL", "").strip()
    if configured_health_url:
        health_url = configured_health_url
    else:
        api_url = urlsplit(API_BASE_URL)
        if not api_url.scheme or not api_url.netloc:
            raise RuntimeError(f"Invalid MODELHUB_BASE_URL: {API_BASE_URL!r}")
        # vLLM exposes /health outside the OpenAI-compatible /v1 namespace.
        # Reverse proxies with a path prefix can provide MODELHUB_HEALTH_URL.
        health_url = urlunsplit((api_url.scheme, api_url.netloc, "/health", "", ""))

    started = time.monotonic()
    wait_deadline = started + max(0.1, float(max_wait))
    if deadline_at is not None:
        wait_deadline = min(wait_deadline, float(deadline_at))
    first_attempt = True
    while time.monotonic() < wait_deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2):
                pass
            elapsed = time.monotonic() - started
            _json_logger.info("vLLM ready after %.0fs", elapsed)
            return elapsed
        except Exception:
            if first_attempt:
                _json_logger.info("Waiting for vLLM to be ready...")
                first_attempt = False
            time.sleep(min(0.5, max(0.0, wait_deadline - time.monotonic())))
    elapsed = time.monotonic() - started
    raise RuntimeError(f"vLLM not ready within {elapsed:.1f}s")


def _execute_resume_job(task_id: str, form_data: Any, deadline_at: float) -> dict[str, Any]:
    """Execute one resume job inside an isolated child process."""
    trace_token = set_trace_id(task_id)
    try:
        _wait_vllm_ready(deadline_at=deadline_at)
        query = str(form_data.get("Query") or form_data.get("query") or "")
        target_jd_text = str(form_data.get("target_jd") or "")
        keys_summary = [f"{key}=<str:{len(str(form_data[key]))}c>" for key in form_data]
        _json_logger.info(
            "[%s] form keys: %d fields, %s",
            task_id,
            len(form_data),
            ", ".join(keys_summary) if keys_summary else "(none)",
        )

        cv_path, cv_text = _extract_file_field(form_data, "cv", task_id)
        cv_template_path = _save_upload_file(form_data.get("cv_template"), task_id, "cv_template")
        jd_path, jd_text = _extract_file_field(form_data, "target_jd", task_id)
        final_jd = jd_text or target_jd_text or ""
        trace_event(
            "request_input",
            query=query,
            extracted_cv=cv_text,
            extracted_jd=final_jd,
            has_cv_file=bool(cv_path),
            has_template=bool(cv_template_path),
            has_jd_file=bool(jd_path),
            form_fields=list(form_data.keys()),
        )

        cv_upload = None
        if cv_text and len(cv_text.strip()) > 5:
            _json_logger.info("[%s] cv uses pre-extracted text (%d chars)", task_id, len(cv_text))
            cv_upload = _MockUpload(None, "cv.txt", content=cv_text)
        elif cv_path:
            # Text extraction may legitimately be empty for an unsupported/blank
            # document. The hard OCR error is propagated instead of retrying here.
            cv_upload = _MockUpload(cv_path, Path(cv_path).name)

        cv_template_upload = _MockUpload(cv_template_path, "template") if cv_template_path else None
        jd_upload = _MockUpload(jd_path, Path(jd_path).name) if jd_path else None
        is_url = bool(re.match(r"^https?://", final_jd.strip(), re.IGNORECASE)) if final_jd else False
        target_jd_url = final_jd if is_url else None
        jd_text_value = final_jd if final_jd and not is_url else None

        _json_logger.info(
            "[%s] calling resume_copilot_service | cv_upload=%s | has_jd=%s | query_chars=%d",
            task_id,
            "present" if cv_upload is not None else "NONE",
            bool(jd_upload is not None or jd_text_value or target_jd_url),
            len(query),
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
                    hard_deadline_at=deadline_at,
                )
            )
        finally:
            loop.close()

        files = getattr(response, "files", {}) or {}
        result = {
            "status": "done",
            "summary": getattr(response, "reply_text", "") or "简历生成完成",
            "generated_docx_path": str(files.get("docx", "") or ""),
            "score": getattr(response, "score", "?"),
        }
        trace_event(
            "request_output",
            resume_data=getattr(response, "resume_data", {}) or {},
            reply_text=getattr(response, "reply_text", "") or "",
            missing_fields=getattr(response, "missing_fields", []) or [],
            conflicts=getattr(response, "conflicts", []) or [],
            scenario=getattr(response, "scenario", ""),
            result=result,
        )
        return result
    except Exception as exc:
        trace_event("request_error", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        reset_trace_id(trace_token)


def _resume_job_child(connection, task_id: str, run_id: str, form_data: Any, deadline_at: float) -> None:
    """Spawn child entry. The ready handshake makes later killpg safe."""

    group_ready = False
    deadline_token = None
    try:
        if os.name == "posix":
            os.setsid()
            group_ready = True
            os.environ["RESUME_TASK_PROCESS_GROUP"] = "1"
        connection.send(("ready", {"process_group": group_ready, "run_id": run_id}))
        # LLM work is optional once the finalization window begins.  Keep the
        # child alive until the hard deadline so deterministic reply fallback,
        # DOCX rendering and atomic publication can still complete.
        llm_deadline_at = _llm_work_deadline(deadline_at)
        deadline_token = set_request_deadline(deadline_at=llm_deadline_at)
        result = _execute_resume_job(task_id, form_data, deadline_at)
        connection.send(("result", result))
    except BaseException as exc:
        error_msg = str(getattr(exc, "detail", exc) or type(exc).__name__)
        connection.send(("error", {"message": error_msg[:500]}))
    finally:
        if deadline_token is not None:
            reset_request_deadline(deadline_token)
        connection.close()


def _linux_descendant_pids(root_pid: int) -> set[int]:
    """Return a best-effort recursive process tree snapshot from procfs."""

    if os.name != "posix" or not Path("/proc").is_dir():
        return set()
    descendants: set[int] = set()
    pending = [int(root_pid)]
    while pending:
        parent_pid = pending.pop()
        task_dir = Path(f"/proc/{parent_pid}/task")
        try:
            child_files = list(task_dir.glob("*/children"))
        except OSError:
            continue
        for child_file in child_files:
            try:
                child_ids = [int(value) for value in child_file.read_text().split()]
            except (OSError, ValueError):
                continue
            for child_pid in child_ids:
                if child_pid in descendants or child_pid == root_pid:
                    continue
                descendants.add(child_pid)
                pending.append(child_pid)
    return descendants


def _signal_descendant_tree(root_pid: int, descendants: set[int], sig: int) -> None:
    """Signal detached descendant groups plus individual descendants safely."""

    try:
        supervisor_group = os.getpgrp()
    except OSError:
        supervisor_group = -1
    try:
        root_group = os.getpgid(root_pid)
    except OSError:
        root_group = -1

    detached_groups: set[int] = set()
    for child_pid in descendants:
        try:
            child_group = os.getpgid(child_pid)
        except OSError:
            continue
        if child_group > 1 and child_group not in {supervisor_group, root_group}:
            detached_groups.add(child_group)

    for process_group in detached_groups:
        try:
            os.killpg(process_group, sig)
        except (OSError, ProcessLookupError):
            pass
    # Also address descendants which did not establish a separate process group.
    # Individual signalling cannot accidentally terminate the API's own group.
    for child_pid in descendants:
        try:
            os.kill(child_pid, sig)
        except (OSError, ProcessLookupError):
            pass


def _terminate_task_process(process: mp.Process, *, process_group_ready: bool) -> None:
    if process.pid is None:
        return
    if not process.is_alive():
        process.join(timeout=0.2)
        if os.name == "posix" and process_group_ready:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            time.sleep(0.05)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        return
    descendants = _linux_descendant_pids(process.pid)
    _signal_descendant_tree(process.pid, descendants, signal.SIGTERM)
    try:
        if os.name == "posix" and process_group_ready:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        pass
    process.join(timeout=_TASK_PROCESS_KILL_GRACE_SECONDS)
    if not process.is_alive():
        _signal_descendant_tree(process.pid, descendants, signal.SIGKILL)
        return
    descendants.update(_linux_descendant_pids(process.pid))
    _signal_descendant_tree(process.pid, descendants, signal.SIGKILL)
    try:
        if os.name == "posix" and process_group_ready:
            os.killpg(process.pid, signal.SIGKILL)
        elif hasattr(process, "kill"):
            process.kill()
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        pass
    process.join(timeout=1.0)


def _process_resume(
    task_id: str,
    run_id: str,
    form_data: Any,
    deadline_at: float,
) -> None:
    """Supervise a killable child so timed-out native work cannot survive."""

    if not _task_is_active(task_id, run_id) or time.monotonic() >= deadline_at:
        _expire_task(task_id, run_id, form_data)
        _cleanup_saved_inputs(form_data)
        return

    context = mp.get_context(os.getenv("TASK_PROCESS_START_METHOD", "spawn"))
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_resume_job_child,
        args=(child_connection, task_id, run_id, form_data, deadline_at),
        name=f"resume-job-{task_id}",
    )
    process_group_ready = False
    terminal_message: tuple[str, Any] | None = None
    started = time.monotonic()
    try:
        with task_lock:
            state = async_tasks.get(task_id)
            if state and state.get("run_id") == run_id and not state.get("finished"):
                state["status"] = "processing"
        process.start()
        child_connection.close()
        while time.monotonic() < deadline_at and _task_is_active(task_id, run_id):
            wait_seconds = min(0.1, max(0.0, deadline_at - time.monotonic()))
            if parent_connection.poll(wait_seconds):
                message = parent_connection.recv()
                if not isinstance(message, tuple) or len(message) != 2:
                    continue
                status, payload = message
                if status == "ready":
                    process_group_ready = bool(
                        isinstance(payload, dict) and payload.get("process_group")
                    )
                    continue
                terminal_message = (str(status), payload)
                break
            if not process.is_alive():
                break

        if terminal_message is None:
            _terminate_task_process(process, process_group_ready=process_group_ready)
            if _task_is_active(task_id, run_id):
                _publish_terminal_state(
                    task_id,
                    run_id,
                    status="error",
                    summary=f"生成失败: 端到端处理超过 {_TASK_DEADLINE_SECONDS:.0f} 秒，任务已终止",
                    file_path="",
                    error="end-to-end task deadline exceeded",
                )
            return

        status, payload = terminal_message
        process.join(timeout=1.0)
        if process.is_alive():
            _terminate_task_process(process, process_group_ready=process_group_ready)
        if status == "result" and isinstance(payload, dict):
            generated_path = Path(str(payload.get("generated_docx_path", "") or ""))
            if not generated_path.is_file():
                raise RuntimeError("resume generation completed without a DOCX output")
            if not _task_is_active(task_id, run_id):
                return
            temp_dest = OUTPUT_DIR / f".{task_id}.{run_id}.docx.tmp"
            final_dest = OUTPUT_DIR / f"{task_id}.{run_id}.docx"
            shutil.copy2(generated_path, temp_dest)
            published = _publish_success_file(
                task_id,
                run_id,
                temp_path=temp_dest,
                final_path=final_dest,
                summary=str(payload.get("summary", "") or "简历生成完成"),
            )
            if not published:
                temp_dest.unlink(missing_ok=True)
                return
            _json_logger.info(
                "Task %s completed in %.1fs (score=%s)",
                task_id,
                time.monotonic() - started,
                payload.get("score", "?"),
            )
        else:
            error_msg = str(payload.get("message", "child process failed")) if isinstance(payload, dict) else str(payload)
            _publish_terminal_state(
                task_id,
                run_id,
                status="error",
                summary=f"生成失败: {error_msg[:500]}",
                file_path="",
                error=error_msg[:500],
            )
    except Exception as exc:
        _terminate_task_process(process, process_group_ready=process_group_ready)
        _json_logger.error("Task %s supervisor failed: %s", task_id, exc, exc_info=True)
        _publish_terminal_state(
            task_id,
            run_id,
            status="error",
            summary=f"生成失败: {str(exc)[:500]}",
            file_path="",
            error=str(exc)[:500],
        )
    finally:
        try:
            child_connection.close()
        except OSError:
            pass
        parent_connection.close()
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
    received_wall = time.time()
    received_mono = time.monotonic()
    task_id = "unknown"
    run_id = ""
    data: Any = {}
    submitted = False
    try:
        _cleanup_task_state()

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
        run_id = secrets.token_hex(8)
        deadline_at = received_mono + _TASK_DEADLINE_SECONDS

        with _task_futures_lock:
            prior_future = _task_futures.get(task_id)
            prior_run_active = bool(prior_future and not prior_future[1].done())

        with task_lock:
            existing = async_tasks.get(task_id)
            if prior_run_active or (existing and not existing.get("finished", False)):
                return _json_response({"success": False, "message": "task id is already active"}, 409)
            # Reserve the ID before persisting uploads. This closes the race in
            # which two concurrent requests with the same evaluator ID both
            # passed the duplicate check and overwrote one another.
            async_tasks[task_id] = {
                "status": "receiving",
                "finished": False,
                "summary": "",
                "file_path": "",
                "error": None,
                "start_time": received_wall,
                "deadline_at": deadline_at,
                "run_id": run_id,
            }

        # ── Save files to disk BEFORE spawning background thread ──
        # Flask/Werkzeug stores uploaded files as temp files; those are cleaned up
        # when the request returns. The background thread runs after the response
        # is sent, so we must persist files now and pass paths instead of FileStorage.
        for key, val in list(data.items()):
            if hasattr(val, "filename") and getattr(val, "filename", None):
                save_path = safe_child_path(
                    UPLOADS_DIR,
                    f"{task_id}_{run_id}_{safe_filename(key, 'file')}_{safe_filename(val.filename)}",
                )
                val.save(str(save_path))
                if save_path.stat().st_size > MAX_FILE_SIZE:
                    save_path.unlink(missing_ok=True)
                    _cleanup_saved_inputs(data)
                    _remove_task_state(task_id, run_id)
                    return _json_response({"success": False, "message": "uploaded file is too large"}, 413)
                data[key] = str(save_path)  # Replace FileStorage with persisted path
                _json_logger.info("[%s] saved upload field=%s bytes=%d", task_id, key, save_path.stat().st_size)

        # Uploads are durable; expose the task as queued.
        with task_lock:
            state = async_tasks.get(task_id)
            if not state or state.get("run_id") != run_id:
                raise RuntimeError("task reservation was lost")
            state["status"] = "queued"

        # Log received fields
        query = data.get("Query", data.get("query", ""))
        _json_logger.info("[%s] 收到简历生成请求: query_chars=%d", task_id, len(str(query)))

        # Submit to thread pool (max_workers=1 prevents thread accumulation)
        _task_id = task_id
        with _task_futures_lock:
            active = sum(1 for _, item in _task_futures.values() if not item.done())
            if active >= _TASK_QUEUE_LIMIT:
                _remove_task_state(_task_id, run_id)
                _cleanup_saved_inputs(data)
                return _json_response({"success": False, "message": "任务队列已满，请稍后重试"}, 429)
            future = _task_executor.submit(
                _process_resume,
                _task_id,
                run_id,
                data,
                deadline_at,
            )
            _task_futures[_task_id] = (run_id, future)
            submitted = True

        timer = threading.Timer(
            max(0.01, deadline_at - time.monotonic()),
            _expire_task,
            args=(_task_id, run_id, data),
        )
        timer.daemon = True
        with _task_timers_lock:
            _task_timers[_task_id] = (run_id, timer)
        timer.start()
        future.add_done_callback(
            lambda _f, tid=_task_id, rid=run_id: _discard_future(tid, rid)
        )

        return _json_response({
            "success": True,
            "message": "简历生成请求已接收，正在异步处理中",
            "task_id": task_id,
            "deadline_seconds": int(_TASK_DEADLINE_SECONDS),
        })

    except ValueError as exc:
        if run_id and not submitted:
            _remove_task_state(task_id, run_id)
            _cleanup_saved_inputs(data)
        return _json_response({"success": False, "message": str(exc)}, 400)
    except Exception as exc:
        if run_id:
            if submitted:
                _publish_terminal_state(
                    task_id,
                    run_id,
                    status="error",
                    summary="生成失败: 请求入队时发生内部错误",
                    file_path="",
                    error=str(exc)[:500],
                )
                with _task_futures_lock:
                    current = _task_futures.get(task_id)
                    future_to_cancel = (
                        current[1] if current and current[0] == run_id else None
                    )
                if future_to_cancel is not None:
                    future_to_cancel.cancel()
            else:
                _remove_task_state(task_id, run_id)
            _cleanup_saved_inputs(data)
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
        "status": status.get("status", "unknown"),
        "summary": status.get("summary", ""),
        "error": status.get("error"),
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
        status = async_tasks.get(task_id)
        file_path = status.get("file_path", "") if status else ""

    # Historical fallback is allowed only when no live task state exists. A new
    # run with the same ID must never expose the previous run's DOCX while it is
    # processing or after it failed.
    if status is None and not file_path:
        fallback = safe_child_path(OUTPUT_DIR, f"{task_id}.docx")
        if fallback.exists():
            file_path = str(fallback)

    if status is not None and (
        not status.get("finished", False) or status.get("status") != "done"
    ):
        return _json_response(
            {"success": False, "message": f"简历尚未成功生成: {task_id}"},
            404,
        )

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
