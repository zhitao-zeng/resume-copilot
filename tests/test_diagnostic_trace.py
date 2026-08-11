import json
import logging

from diagnostic_trace import reset_trace_id, set_trace_id, trace_event
from resume_optimizer import _safe_rewrite_diagnostics


def _record_payload(record) -> dict:
    marker = "RESUME_DIAG "
    message = record.getMessage()
    assert marker in message
    return json.loads(message.split(marker, 1)[1])


def test_trace_disabled_by_default(monkeypatch, caplog):
    monkeypatch.delenv("RESUME_DIAGNOSTIC_TRACE", raising=False)
    with caplog.at_level(logging.INFO, logger="resume_diagnostic"):
        trace_event("request_input", query="姓名：测试用户")
    assert not caplog.records


def test_trace_keeps_raw_fictional_business_fields_and_omits_credentials(
    monkeypatch,
    caplog,
):
    monkeypatch.setenv("RESUME_DIAGNOSTIC_TRACE", "1")
    token = set_trace_id("case-13")
    try:
        with caplog.at_level(logging.INFO, logger="resume_diagnostic"):
            trace_event(
                "request_input",
                query="姓名：测试用户；电话：13800138000；邮箱：demo@example.com",
                nested={
                    "role": "单元测试工程师",
                    "authorization": "Bearer must-not-leak",
                    "api_key": "must-not-leak",
                },
            )
    finally:
        reset_trace_id(token)

    payload = _record_payload(caplog.records[-1])
    assert payload["trace_id"] == "case-13"
    assert payload["query"] == "姓名：测试用户；电话：13800138000；邮箱：demo@example.com"
    assert payload["nested"]["role"] == "单元测试工程师"
    assert payload["nested"]["authorization"] == "<credential omitted>"
    assert payload["nested"]["api_key"] == "<credential omitted>"
    assert "must-not-leak" not in caplog.records[-1].getMessage()


def test_rewrite_diagnostics_explain_multiple_hard_gate_reasons():
    accepted, reasons = _safe_rewrite_diagnostics(
        "参与整理接口测试结果并提交报告",
        "主导使用 Selenium 平台完成接口测试，提升效率 30%",
    )

    assert accepted is False
    assert "ownership_level_changed" in reasons
    assert "new_numeric_fact" in reasons
    assert "new_latin_token" in reasons
    assert "new_named_method_or_tool" in reasons
    assert "new_result_claim:提升效率" in reasons
