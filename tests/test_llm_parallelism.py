"""Regression tests for bounded, deterministic LLM fan-out.

Composer chunks and Optimizer batches are independent LLM requests, but their
results still belong to one ordered resume.  These tests keep the concurrency
contract observable without making a real backend call.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import resume_optimizer
import server_runtime
from llm_gateway import LLMDeadlineExceeded
from resume_composer import compose_resume
from resume_optimizer import optimize_resume
from server_runtime import (
    get_request_deadline,
    reset_request_deadline,
    set_request_deadline,
)
from v2_schemas import CanonicalResume, DraftResume, SourceBlock, SourceBundle


class _ConcurrencyProbe:
    """Measure overlapping calls while making completion order deterministic."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.thread_ids: list[int] = []
        self.deadlines: list[float | None] = []

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.thread_ids.append(threading.get_ident())
            self.deadlines.append(get_request_deadline())

    def leave(self) -> None:
        with self._lock:
            self.active -= 1


def _fresh_two_slot_limiter(monkeypatch) -> None:
    """Keep global-limiter tests isolated from environment and prior calls."""

    monkeypatch.setattr(server_runtime, "LLM_INFLIGHT_LIMIT", 2)
    monkeypatch.setattr(
        server_runtime,
        "_LLM_INFLIGHT_SLOTS",
        threading.BoundedSemaphore(2),
    )


def test_call_llm_text_global_limiter_caps_four_threads_at_two(monkeypatch):
    _fresh_two_slot_limiter(monkeypatch)
    probe = _ConcurrencyProbe()
    callers_ready = threading.Barrier(4)

    class FakeGateway:
        def call_text(self, **kwargs):
            probe.enter()
            try:
                # Sleeping releases the GIL and gives all four callers a fair
                # chance to expose an incorrectly unbounded implementation.
                time.sleep(0.06)
                return kwargs["user_prompt"]
            finally:
                probe.leave()

    monkeypatch.setattr(server_runtime, "get_llm_gateway", lambda: FakeGateway())

    def invoke(index: int) -> str:
        callers_ready.wait(timeout=1)
        return server_runtime.call_llm_text("system", f"request-{index}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(invoke, index) for index in range(4)]
        results = [future.result(timeout=2) for future in futures]

    assert probe.peak == 2
    assert sorted(results) == [
        "request-0", "request-1", "request-2", "request-3",
    ]


def test_call_llm_text_slot_wait_stops_at_request_deadline(monkeypatch):
    _fresh_two_slot_limiter(monkeypatch)
    monkeypatch.setattr(server_runtime, "LLM_DEADLINE_RESERVE_SECONDS", 0.02)
    holders_release = threading.Event()
    both_holders_entered = threading.Event()
    state_lock = threading.Lock()
    holder_count = 0
    waiting_call_reached_gateway = False

    class BlockingGateway:
        def call_text(self, **kwargs):
            nonlocal holder_count, waiting_call_reached_gateway
            if kwargs["user_prompt"].startswith("holder-"):
                with state_lock:
                    holder_count += 1
                    if holder_count == 2:
                        both_holders_entered.set()
                assert holders_release.wait(timeout=2)
                return kwargs["user_prompt"]
            waiting_call_reached_gateway = True
            return "should not acquire a slot"

    gateway = BlockingGateway()
    monkeypatch.setattr(server_runtime, "get_llm_gateway", lambda: gateway)

    with ThreadPoolExecutor(max_workers=2) as executor:
        holder_futures = [
            executor.submit(
                server_runtime.call_llm_text,
                "system",
                f"holder-{index}",
            )
            for index in range(2)
        ]
        try:
            assert both_holders_entered.wait(timeout=1)
            deadline_at = time.monotonic() + 0.18
            deadline_token = set_request_deadline(deadline_at=deadline_at)
            try:
                with pytest.raises(LLMDeadlineExceeded):
                    server_runtime.call_llm_text("system", "waiting-request")
                finished_at = time.monotonic()
            finally:
                reset_request_deadline(deadline_token)

            assert finished_at <= deadline_at + 0.05
            assert waiting_call_reached_gateway is False
        finally:
            holders_release.set()
            for future in holder_futures:
                future.result(timeout=1)


def _composer_chunks(count: int = 4) -> list[SourceBundle]:
    return [
        SourceBundle(blocks=[
            SourceBlock(
                block_id=f"resume_{index}",
                source_type="resume",
                text=f"chunk-{index} 候选人真实经历",
            ),
        ])
        for index in range(count)
    ]


def test_composer_initial_chunks_use_two_workers_and_merge_in_input_order(monkeypatch):
    chunks = _composer_chunks()
    probe = _ConcurrencyProbe()

    def fake_call(_model, _system, prompt, **_kwargs):
        index = int(re.search(r"chunk-(\d+)", prompt).group(1))
        probe.enter()
        try:
            # Later input chunks finish first, so an as-completed merge would
            # produce the wrong resume order.
            time.sleep((4 - index) * 0.025)
            return {
                "experience": [{
                    "organization": f"公司{index}",
                    "role": "工程师",
                    "bullets": [f"负责事项{index}"],
                }],
            }
        finally:
            probe.leave()

    monkeypatch.setattr("resume_composer.llm_enabled", lambda: True)
    monkeypatch.setattr("resume_composer._split_source_bundle", lambda _source: chunks)
    monkeypatch.setattr("resume_composer.call_llm_typed", fake_call)

    result = compose_resume(SourceBundle(
        blocks=[block for chunk in chunks for block in chunk.blocks],
    ))

    assert probe.peak == 2
    assert [item.organization for item in result.experience] == [
        "公司0", "公司1", "公司2", "公司3",
    ]


def test_composer_parallel_failure_retains_successful_chunks(monkeypatch):
    chunks = _composer_chunks(3)

    def fake_call(_model, _system, prompt, **_kwargs):
        index = int(re.search(r"chunk-(\d+)", prompt).group(1))
        if index == 1:
            raise RuntimeError("backend failed")
        time.sleep(0.03)
        return {
            "experience": [{
                "organization": f"公司{index}",
                "role": "工程师",
                "bullets": [f"负责事项{index}"],
            }],
        }

    monkeypatch.setattr("resume_composer.llm_enabled", lambda: True)
    monkeypatch.setattr("resume_composer._split_source_bundle", lambda _source: chunks)
    monkeypatch.setattr("resume_composer.call_llm_typed", fake_call)

    result = compose_resume(SourceBundle(
        blocks=[block for chunk in chunks for block in chunk.blocks],
    ))

    assert {item.organization for item in result.experience} == {"公司0", "公司2"}


def test_composer_workers_inherit_request_deadline_context(monkeypatch):
    chunks = _composer_chunks(2)
    caller_thread = threading.get_ident()
    observed: list[tuple[int, float | None]] = []
    observed_lock = threading.Lock()

    def fake_call(_model, _system, prompt, **_kwargs):
        index = int(re.search(r"chunk-(\d+)", prompt).group(1))
        with observed_lock:
            observed.append((threading.get_ident(), get_request_deadline()))
        time.sleep(0.02)
        return {
            "experience": [{
                "organization": f"公司{index}",
                "role": "工程师",
                "bullets": [f"负责事项{index}"],
            }],
        }

    monkeypatch.setattr("resume_composer.llm_enabled", lambda: True)
    monkeypatch.setattr("resume_composer._split_source_bundle", lambda _source: chunks)
    monkeypatch.setattr("resume_composer.call_llm_typed", fake_call)

    deadline_at = time.monotonic() + 60
    deadline_token = set_request_deadline(deadline_at=deadline_at)
    try:
        result = compose_resume(SourceBundle(
            blocks=[block for chunk in chunks for block in chunk.blocks],
        ))
    finally:
        reset_request_deadline(deadline_token)

    assert len(result.experience) == 2
    assert len(observed) == 2
    assert all(thread_id != caller_thread for thread_id, _deadline in observed)
    assert all(value == deadline_at for _thread_id, value in observed)


def test_optimizer_batches_use_two_workers_and_apply_stable_indexes_on_caller_thread(
    monkeypatch,
):
    labels = ("甲", "乙", "丙", "丁")
    resume = CanonicalResume.model_validate({
        "experience": [
            {
                "organization": f"{label}公司",
                "role": "产品经理",
                # Eight bullets force one record per initial optimizer batch.
                "bullets": [
                    f"负责{label}类用户反馈整理、{label}类需求文档维护并跟进{label}类交付"
                    for _ in range(8)
                ],
            }
            for label in labels
        ],
    })
    probe = _ConcurrencyProbe()
    caller_thread = threading.get_ident()
    collect_threads: list[int] = []
    original_collect = resume_optimizer._section_patch_proposals

    def fake_call(_system, prompt, **_kwargs):
        payload = json.loads(prompt.split("【只读事实与原始 bullets】\n", 1)[1])
        record = payload["experience"][0]
        index = record["index"]
        label = labels[index]
        probe.enter()
        try:
            time.sleep((4 - index) * 0.025)
            return json.dumps({
                "experience": [{
                    # This is a global resume index, not a batch-local index.
                    "index": index,
                    "bullets": [
                        f"负责{label}类需求文档维护、{label}类用户反馈整理并跟进{label}类交付"
                        for _ in record["bullets"]
                    ],
                }],
            }, ensure_ascii=False)
        finally:
            probe.leave()

    def recording_collect(optimized, section, patches):
        collect_threads.append(threading.get_ident())
        return original_collect(optimized, section, patches)

    monkeypatch.setattr(resume_optimizer, "llm_enabled", lambda: True)
    monkeypatch.setattr(resume_optimizer, "call_llm_text", fake_call)
    monkeypatch.setattr(resume_optimizer, "_section_patch_proposals", recording_collect)

    deadline_at = time.monotonic() + 60
    deadline_token = set_request_deadline(deadline_at=deadline_at)
    try:
        result = optimize_resume(resume, "产品经理")
    finally:
        reset_request_deadline(deadline_token)

    assert probe.peak == 2
    assert all(thread_id != caller_thread for thread_id in probe.thread_ids)
    assert all(value == deadline_at for value in probe.deadlines)
    assert collect_threads and set(collect_threads) == {caller_thread}
    assert [item.organization for item in result.experience] == [
        "甲公司", "乙公司", "丙公司", "丁公司",
    ]
    for index, label in enumerate(labels):
        assert result.experience[index].bullets == [
            f"负责{label}类需求文档维护、{label}类用户反馈整理并跟进{label}类交付"
            for _ in range(8)
        ]
