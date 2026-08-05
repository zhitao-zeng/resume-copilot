import concurrent.futures
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

import resume_generate_api as api


class _ReadyResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_vllm_health_check_follows_configured_model_host(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "API_BASE_URL", "http://model-host:8123/v1")
    monkeypatch.delenv("MODELHUB_HEALTH_URL", raising=False)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url, timeout: calls.append((url, timeout)) or _ReadyResponse(),
    )

    api._wait_vllm_ready(max_wait=0.1)

    assert calls == [("http://model-host:8123/health", 2)]


def _detached_grandchild_worker(connection):
    os.setsid()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,time; os.setsid(); print(os.getpid(), flush=True); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    child_pid = int(child.stdout.readline().strip())
    connection.send(child_pid)
    connection.close()
    time.sleep(30)


def _exited_group_leader(connection):
    os.setsid()
    child = subprocess.Popen(
        [sys.executable, "-c", "import os,time; print(os.getpid(), flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE,
        text=True,
    )
    connection.send(int(child.stdout.readline().strip()))
    connection.close()
    os._exit(0)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_queued_deadline_cancels_future_and_cleans_upload(tmp_path, monkeypatch):
    task_id = "queued-timeout"
    run_id = "run-a"
    upload = tmp_path / "resume.txt"
    upload.write_text("resume")
    future = concurrent.futures.Future()
    monkeypatch.setattr(api, "UPLOADS_DIR", tmp_path)
    with api.task_lock:
        api.async_tasks[task_id] = {
            "run_id": run_id,
            "finished": False,
            "status": "queued",
            "start_time": time.time(),
        }
    with api._task_futures_lock:
        api._task_futures[task_id] = (run_id, future)

    try:
        api._expire_task(task_id, run_id, {"cv": str(upload)})

        assert future.cancelled()
        assert not upload.exists()
        with api.task_lock:
            state = api.async_tasks[task_id]
            assert state["finished"] is True
            assert state["status"] == "error"
    finally:
        with api.task_lock:
            api.async_tasks.pop(task_id, None)
        with api._task_futures_lock:
            api._task_futures.pop(task_id, None)


def test_old_deadline_cannot_overwrite_reused_task_id():
    task_id = "reused-id"
    with api.task_lock:
        api.async_tasks[task_id] = {
            "run_id": "new-run",
            "finished": False,
            "status": "queued",
        }
    try:
        api._expire_task(task_id, "old-run", {})
        with api.task_lock:
            assert api.async_tasks[task_id]["finished"] is False
    finally:
        with api.task_lock:
            api.async_tasks.pop(task_id, None)


def test_success_file_and_state_publish_are_atomic(tmp_path):
    task_id = "atomic-success"
    run_id = "run-success"
    temp_path = tmp_path / "resume.tmp"
    final_path = tmp_path / "resume.docx"
    temp_path.write_bytes(b"docx")
    with api.task_lock:
        api.async_tasks[task_id] = {
            "run_id": run_id,
            "finished": False,
            "status": "processing",
        }
    try:
        assert api._publish_success_file(
            task_id,
            run_id,
            temp_path=temp_path,
            final_path=final_path,
            summary="done",
        )
        api._expire_task(task_id, run_id, {})
        with api.task_lock:
            state = api.async_tasks[task_id]
            assert state["status"] == "done"
            assert state["file_path"] == str(final_path)
        assert final_path.read_bytes() == b"docx"
    finally:
        with api.task_lock:
            api.async_tasks.pop(task_id, None)


@pytest.mark.skipif(os.name != "posix" or not Path("/proc").is_dir(), reason="Linux procfs required")
def test_outer_supervisor_kills_detached_grandchild():
    context = api.mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_detached_grandchild_worker, args=(child,))
    grandchild_pid = None
    process.start()
    child.close()
    try:
        assert parent.poll(10)
        grandchild_pid = parent.recv()
        assert _pid_exists(grandchild_pid)

        api._terminate_task_process(process, process_group_ready=True)

        deadline = time.monotonic() + 3
        while _pid_exists(grandchild_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not process.is_alive()
        assert not _pid_exists(grandchild_pid)
    finally:
        parent.close()
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        if grandchild_pid and _pid_exists(grandchild_pid):
            try:
                os.kill(grandchild_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_dead_task_leader_still_kills_live_process_group():
    context = api.mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_exited_group_leader, args=(child,))
    child_pid = None
    process.start()
    child.close()
    try:
        assert parent.poll(10)
        child_pid = parent.recv()
        process.join(timeout=5)
        assert not process.is_alive()
        assert _pid_exists(child_pid)

        api._terminate_task_process(process, process_group_ready=True)

        deadline = time.monotonic() + 3
        while _pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_exists(child_pid)
    finally:
        parent.close()
        if child_pid and _pid_exists(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
