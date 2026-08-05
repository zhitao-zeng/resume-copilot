import logging
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import BaseModel

import server_runtime
from llm_gateway import LLMDeadlineExceeded, LLMGateway
from resume_composer import compose_resume
from server_runtime import (
    get_request_deadline,
    reset_request_deadline,
    set_request_deadline,
)
from v2_schemas import DraftResume, Meta, SourceBlock, SourceBundle


class _Payload(BaseModel):
    ok: bool


class APITimeoutError(Exception):
    pass


class _FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=outcome),
                finish_reason="stop",
            )]
        )


class _RemainingSequence:
    def __init__(self, *values):
        self.values = list(values)
        self.last = self.values[-1]

    def __call__(self):
        if self.values:
            self.last = self.values.pop(0)
        return self.last


def _gateway(completions, **kwargs):
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return LLMGateway(
        client_factory=lambda: client,
        model_name="deadline-test-model",
        logger=logging.getLogger("test.llm_deadline"),
        enable_json_repair=kwargs.pop("enable_json_repair", False),
        dump_failure_payload=lambda *_args: None,
        **kwargs,
    )


def test_openai_client_disables_sdk_automatic_retries(monkeypatch):
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(server_runtime, "OpenAI", fake_openai)
    monkeypatch.setattr(server_runtime, "API_KEY", "test-key")
    monkeypatch.setattr(server_runtime, "API_BASE_URL", "http://localhost/v1")
    monkeypatch.setattr(server_runtime, "_client", None)

    server_runtime.get_client()

    assert captured["max_retries"] == 0
    assert captured["timeout"] == server_runtime.LLM_TIMEOUT_SECONDS


def test_gateway_recalculates_timeout_for_each_backend_attempt():
    unsupported = RuntimeError("response_format json_object is unsupported")
    completions = _FakeCompletions([unsupported, '{"ok": true}'])
    gateway = _gateway(
        completions,
        call_timeout_seconds=20,
        request_time_remaining=_RemainingSequence(50, 7),
        deadline_reserve_seconds=2,
        retry_min_remaining_seconds=5,
    )

    content = gateway._call_raw([{"role": "user", "content": "test"}])

    assert content == '{"ok": true}'
    assert [call["timeout"] for call in completions.calls] == [20, 5]


def test_gateway_deadline_exhaustion_does_not_open_shared_circuit():
    completions = _FakeCompletions(['{"ok": true}'])
    gateway = _gateway(
        completions,
        call_timeout_seconds=20,
        request_time_remaining=lambda: 1,
        deadline_reserve_seconds=2,
    )

    with pytest.raises(LLMDeadlineExceeded):
        gateway._call_raw([{"role": "user", "content": "test"}])

    assert completions.calls == []
    assert gateway._consecutive_failures == 0


def test_deadline_limited_transport_timeout_does_not_open_shared_circuit():
    completions = _FakeCompletions([APITimeoutError("read timed out")])
    gateway = _gateway(
        completions,
        call_timeout_seconds=20,
        request_time_remaining=lambda: 7,
        deadline_reserve_seconds=2,
    )

    with pytest.raises(LLMDeadlineExceeded):
        gateway._call_raw([{"role": "user", "content": "test"}])

    assert completions.calls[0]["timeout"] == 5
    assert gateway._consecutive_failures == 0


def test_typed_json_repair_is_skipped_when_request_budget_is_low():
    completions = _FakeCompletions(["not-json"])
    gateway = _gateway(
        completions,
        enable_json_repair=True,
        call_timeout_seconds=20,
        request_time_remaining=_RemainingSequence(50, 5),
        deadline_reserve_seconds=2,
        retry_min_remaining_seconds=10,
    )

    assert gateway.call_typed(_Payload, "system", "user") == {}
    assert len(completions.calls) == 1


def test_nested_deadline_never_extends_outer_budget():
    outer_at = time.monotonic() + 30
    outer_token = set_request_deadline(deadline_at=outer_at)
    try:
        inner_token = set_request_deadline(timeout_seconds=300)
        try:
            assert get_request_deadline() == outer_at
        finally:
            reset_request_deadline(inner_token)
        assert get_request_deadline() == outer_at
    finally:
        reset_request_deadline(outer_token)
    assert get_request_deadline() is None


def test_composer_discards_partial_chunks_when_deadline_budget_runs_low():
    source = SourceBundle(blocks=[
        SourceBlock(block_id="resume_0", source_type="resume", text="候选人事实"),
    ])
    partial = DraftResume(meta=Meta(name="张三")).model_dump()

    with patch("resume_composer.llm_enabled", return_value=True), patch(
        "resume_composer._split_source_bundle", return_value=[source, source]
    ), patch(
        "resume_composer._composer_has_time_budget", side_effect=[True, False]
    ), patch(
        "resume_composer.call_llm_typed", return_value=partial
    ) as llm_call:
        result = compose_resume(source)

    assert result == DraftResume()
    assert llm_call.call_count == 1
