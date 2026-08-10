import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evidence_binding import enforce_resume_evidence
from llm_gateway import ContextBudgetError, LLMGateway, estimate_chat_tokens
from prompts import GEN_COMPOSER_SYSTEM_PROMPT
from resume_composer import (
    _COMPOSER_MAX_TOKENS,
    _COMPOSER_SAFETY_TOKENS,
    _composer_prompt_token_estimate,
    _prepare_generate_request,
    _split_source_bundle,
    _typed_prompt_token_estimate,
    compose_resume,
)
from source_adapter import build_source_bundle, candidate_blocks
from v2_pipeline import _deterministic_fallback, _merge_source_recovery, run_v2_pipeline
from v2_schemas import (
    CanonicalResume,
    Change,
    DraftResume,
    Education,
    Meta,
    SourceBlock,
    SourceBundle,
    VerifiedResult,
)


class _FakeCompletions:
    def __init__(self, errors, finish_reasons=None):
        self.errors = list(errors)
        self.finish_reasons = list(finish_reasons or [])
        self.max_token_calls = []

    def create(self, **kwargs):
        self.max_token_calls.append(kwargs["max_tokens"])
        if self.errors:
            raise self.errors.pop(0)
        finish_reason = self.finish_reasons.pop(0) if self.finish_reasons else "stop"
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"ok": true}'),
                finish_reason=finish_reason,
            )]
        )


def _gateway(completions: _FakeCompletions) -> LLMGateway:
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return LLMGateway(
        client_factory=lambda: client,
        model_name="local-test-model",
        logger=logging.getLogger("test.llm_gateway"),
        enable_json_repair=False,
        dump_failure_payload=lambda *_args: None,
    )


def test_query_adapter_accepts_compact_name_and_chinese_date_records():
    query = (
        "姓名白宁，电话13210008025，邮箱baining@example.com，"
        "2020年3月到2025年6月在A银行信贷审核，"
        "2024年6月到2025年5月在B银行风控审核。"
    )

    blocks = build_source_bundle("", query, "").blocks
    eligible = [block.text for block in blocks if block.fact_eligible]

    assert "姓名白宁" in eligible
    assert "2020年3月到2025年6月在A银行信贷审核" in eligible
    assert "2024年6月到2025年5月在B银行风控审核" in eligible


def test_query_adapter_groups_compact_project_and_following_action():
    query = (
        "姓名程洛，上海交通大学软件工程本科09-2022到06-2026，"
        "做过课程选课系统项目，负责需求分析和原型设计，技能SQL、Axure。"
    )

    project_blocks = [
        block for block in candidate_blocks(build_source_bundle("", query, ""))
        if block.section_hint == "projects"
    ]

    assert [block.text for block in project_blocks] == [
        "做过课程选课系统项目",
        "负责需求分析和原型设计",
    ]
    assert project_blocks[0].record_id == project_blocks[1].record_id


def test_gateway_retries_context_overflow_with_backend_safe_budget():
    error = RuntimeError(
        "This model's maximum context length is 8192 tokens. However, you "
        "requested 4096 output tokens and your prompt contains at least 4097 input tokens."
    )
    completions = _FakeCompletions([error])
    gateway = _gateway(completions)

    content = gateway._call_raw(
        [{"role": "user", "content": "test"}],
        max_tokens=4096,
        prefill="",
    )

    assert content == '{"ok": true}'
    assert completions.max_token_calls == [4096, 3583]
    assert gateway._consecutive_failures == 0


def test_gateway_parses_standard_messages_completion_context_error():
    error = RuntimeError(
        "maximum context length is 8192 tokens; requested 8596 tokens "
        "(4500 in the messages, 4096 in the completion)"
    )

    assert LLMGateway._context_retry_budget(error, 4096) == 3564


def test_text_gateway_context_retry_also_avoids_circuit_breaker():
    error = RuntimeError(
        "maximum context length is 8192 tokens; "
        "your prompt contains at least 4097 input tokens"
    )
    completions = _FakeCompletions([error])
    gateway = _gateway(completions)

    content = gateway.call_text("system", "user", max_tokens=4096)

    assert content == '{"ok": true}'
    assert completions.max_token_calls == [4096, 3583]
    assert gateway._consecutive_failures == 0


def test_unrecoverable_context_error_does_not_open_circuit_breaker():
    error = RuntimeError(
        "maximum context length is 8192 tokens; your prompt contains 8100 input tokens"
    )
    completions = _FakeCompletions([error])
    gateway = _gateway(completions)

    with pytest.raises(ContextBudgetError):
        gateway._call_raw(
            [{"role": "user", "content": "oversized"}],
            max_tokens=4096,
            prefill="",
        )

    assert completions.max_token_calls == [4096]
    assert gateway._consecutive_failures == 0


def test_repeated_context_lower_bounds_exhaust_without_opening_circuit():
    errors = [
        RuntimeError(
            "maximum context length is 8192 tokens; "
            f"your prompt contains at least {4097 + index * 513} input tokens"
        )
        for index in range(5)
    ]
    completions = _FakeCompletions(errors)
    gateway = _gateway(completions)

    with pytest.raises(ContextBudgetError):
        gateway._call_raw(
            [{"role": "user", "content": "still oversized"}],
            max_tokens=4096,
            prefill="",
        )

    assert len(completions.max_token_calls) == 4
    assert gateway._consecutive_failures == 0


def test_truncated_structured_completion_requests_composer_resplit():
    completions = _FakeCompletions([], finish_reasons=["length"])
    gateway = _gateway(completions)

    with pytest.raises(ContextBudgetError, match="truncated"):
        gateway._call_raw(
            [{"role": "user", "content": "large structured output"}],
            max_tokens=2048,
            prefill="",
        )

    assert gateway._consecutive_failures == 0


def test_fallback_token_estimate_is_conservative_for_adversarial_text():
    with patch("llm_gateway._load_local_tokenizer", return_value=None):
        spaced_ascii = estimate_chat_tokens([
            {"role": "user", "content": "a " * 3000},
        ])
        punctuation = estimate_chat_tokens([
            {"role": "user", "content": "!@#$%^&*()_" * 500},
        ])
        emoji = estimate_chat_tokens([
            {"role": "user", "content": "😀" * 3000},
        ])

    assert spaced_ascii >= 3001
    assert punctuation >= 2500
    assert emoji >= 6600


def test_composer_chunks_complete_typed_requests_with_long_jd(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "8192")
    source = build_source_bundle(
        "张三\n产品经理\n负责用户调研和需求分析",
        "请不要编造数字",
        "目标岗位：高级产品经理。" + "负责复杂业务规划、用户研究和跨团队交付。" * 500,
    )

    chunks = _split_source_bundle(source)
    prompt_limit = 8192 - _COMPOSER_MAX_TOKENS - _COMPOSER_SAFETY_TOKENS

    assert chunks
    assert all(_composer_prompt_token_estimate(chunk) <= prompt_limit for chunk in chunks)
    assert all(any(block.source_type == "jd" for block in chunk.blocks) for chunk in chunks)


def test_composer_splits_single_oversized_line_without_losing_text(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "16384")
    long_line = "负责跨行业数据治理、流程设计与项目交付；" * 500
    source = SourceBundle(blocks=[
        SourceBlock(block_id="resume_0", source_type="resume", text=long_line),
    ])

    chunks = _split_source_bundle(source)
    reconstructed = "".join(
        block.text
        for chunk in chunks
        for block in chunk.blocks
        if block.source_type == "resume"
    )
    prompt_limit = 16384 - _COMPOSER_MAX_TOKENS - _COMPOSER_SAFETY_TOKENS

    assert len(chunks) >= 2
    assert reconstructed == long_line
    assert all(_composer_prompt_token_estimate(chunk) <= prompt_limit for chunk in chunks)


def test_composer_caps_short_fact_count_to_prevent_output_truncation(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "16384")
    source = SourceBundle(blocks=[
        SourceBlock(
            block_id=f"resume_{index}",
            source_type="resume",
            section_hint="experience",
            text=f"第{index}条真实职责",
        )
        for index in range(80)
    ])

    chunks = _split_source_bundle(source, max_fact_chars=60_000, max_fact_blocks=36)
    reconstructed = [
        block.text
        for chunk in chunks
        for block in candidate_blocks(chunk)
    ]

    assert len(chunks) == 3
    assert all(len(candidate_blocks(chunk)) <= 36 for chunk in chunks)
    assert reconstructed == [f"第{index}条真实职责" for index in range(80)]


def test_composer_bisects_backend_rejected_chunk_without_losing_facts(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "16384")
    source = build_source_bundle(
        "第一段候选人事实\n第二段候选人事实",
        "不要编造",
        "目标岗位：分析师",
    )
    prompts = []

    def fake_call(*_args, **kwargs):
        prompts.append(_args[2])
        if len(prompts) == 1:
            raise ContextBudgetError("backend reports a smaller context")
        fact = "第一段候选人事实" if "第一段候选人事实" in _args[2] else "第二段候选人事实"
        return {"additional_sections": {"候选人事实": [fact]}}

    with patch("resume_composer.llm_enabled", return_value=True), patch(
        "resume_composer.call_llm_typed", side_effect=fake_call
    ):
        compose_resume(source)

    assert len(prompts) == 3
    successful_prompts = "\n".join(prompts[1:])
    assert successful_prompts.count("第一段候选人事实") == 1
    assert successful_prompts.count("第二段候选人事实") == 1


def test_composer_skips_pathological_input_before_creating_thousands_of_calls():
    source = SourceBundle(blocks=[
        SourceBlock(
            block_id="resume_0",
            source_type="resume",
            text="事实" * 30_001,
        ),
    ])
    with patch("resume_composer.llm_enabled", return_value=True), patch(
        "resume_composer.call_llm_typed"
    ) as llm_call:
        draft = compose_resume(source)

    assert draft == DraftResume()
    llm_call.assert_not_called()


def test_no_cv_request_keeps_minimum_output_budget(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "8192")
    query = "我负责数据分析和项目交付。" * 10
    jd = "岗位要求风险识别、业务分析和跨团队协作。" * 200

    prompt, max_tokens = _prepare_generate_request(query, jd)
    prompt_tokens = _typed_prompt_token_estimate(
        CanonicalResume,
        GEN_COMPOSER_SYSTEM_PROMPT,
        prompt,
    )
    assert prompt
    assert max_tokens == 4096
    assert prompt_tokens + max_tokens + _COMPOSER_SAFETY_TOKENS <= 8192


def test_no_cv_query_facts_are_kept_when_full_output_budget_cannot_fit(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "8192")
    query = "我负责数据分析和项目交付。" * 200

    prompt, max_tokens = _prepare_generate_request(query, "")
    prompt_tokens = _typed_prompt_token_estimate(
        CanonicalResume,
        GEN_COMPOSER_SYSTEM_PROMPT,
        prompt,
    )

    assert "我负责数据分析和项目交付" in prompt
    assert max_tokens >= 2048
    assert prompt_tokens + max_tokens + _COMPOSER_SAFETY_TOKENS <= 8192


def test_deterministic_fallback_adapts_legacy_education_fields():
    raw = {
        "meta": {"name": "张三"},
        "education": [
            {
                "school": f"学校{index}",
                "degree": "硕士",
                "major": "计算机",
                "start_date": "2020.09",
                "end_date": "2023.06",
                "highlights": [],
            }
            for index in range(4)
        ],
        "experience": [],
        "projects": [],
        "skills": {},
    }
    with patch("v2_pipeline.product_logic.infer_industry", return_value="other"), patch(
        "v2_pipeline.product_logic.heuristic_resume_from_text", return_value=raw
    ), patch(
        "v2_pipeline.product_logic.normalize_resume_data_for_product", return_value=raw
    ):
        cv_text = "教育经历\n" + "\n".join(
            f"学校{index}｜计算机｜硕士｜2020.09 - 2023.06"
            for index in range(4)
        )
        resume = _deterministic_fallback(cv_text, "", "")

    assert len(resume.education) == 4
    assert resume.education[0].period == "2020.09 - 2023.06"
    assert set(resume.education[0].model_dump()) == {"school", "degree", "major", "period"}

    source = build_source_bundle(cv_text, "", "")
    gated, _bindings, removed = enforce_resume_evidence(resume, source)
    assert len(gated.education) == 4
    assert not any(path.startswith("education[") for path in removed)


def test_low_coverage_repair_keeps_repeated_degrees_and_rebuilds_changes():
    education = [
        Education(
            school=f"学校{index}",
            degree="硕士",
            major="计算机",
            period="2020.09 - 2023.06",
        )
        for index in range(4)
    ]
    cv_text = "张三\n教育经历\n" + "\n".join(
        f"学校{index}｜计算机｜硕士｜2020.09 - 2023.06"
        for index in range(4)
    )
    low_coverage = VerifiedResult(
        resume=CanonicalResume(meta=Meta(name="张三")),
        changes=[Change(
            path="projects[9].bullets[0]",
            action="replace",
            reason="stale optimizer change",
        )],
    )
    full_fallback = CanonicalResume(
        meta=Meta(name="张三"),
        education=education,
    )

    with patch("v2_pipeline.compose_resume", return_value=DraftResume()), patch(
        "v2_pipeline._deterministic_verify_draft", return_value=low_coverage
    ), patch(
        "v2_pipeline._needs_optimizer", return_value=False
    ), patch(
        "v2_pipeline._deterministic_fallback", return_value=full_fallback
    ):
        result = run_v2_pipeline(cv_text, "请优化", "目标岗位：分析师")

    assert len(result.resume.education) == 4
    assert all(change.path != "*" for change in result.changes)
    assert all(change.reason != "stale optimizer change" for change in result.changes)


def test_complete_fallback_is_merged_before_the_single_optimizer_pass():
    cv_text = (
        "工作经历\n甲公司｜工程师｜2020.01-2023.01\n"
        "负责需求分析\n负责项目交付"
    )
    sparse = VerifiedResult(
        resume=CanonicalResume(meta=Meta(name="张三")),
    )
    full_fallback = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "工程师",
            "period": "2020.01-2023.01",
            "bullets": ["负责需求分析", "负责项目交付"],
        }],
    })
    optimized_inputs = []

    def fake_optimize(resume, _jd):
        from resume_optimizer import OptimizationOutcome

        optimized_inputs.append(resume.model_copy(deep=True))
        return OptimizationOutcome(resume=resume)

    with patch("v2_pipeline.compose_resume", return_value=DraftResume()), patch(
        "v2_pipeline._deterministic_verify_draft", return_value=sparse,
    ), patch(
        "v2_pipeline._deterministic_fallback", return_value=full_fallback,
    ), patch(
        "v2_pipeline.optimize_resume_with_provenance", side_effect=fake_optimize,
    ):
        result = run_v2_pipeline(cv_text, "请优化", "目标岗位：工程师")

    assert optimized_inputs
    assert optimized_inputs[0].experience[0].organization == "甲公司"
    assert set(result.resume.experience[0].bullets) == {"负责需求分析", "负责项目交付"}
    assert all(change.path != "*" for change in result.changes)


def test_source_recovery_keeps_optimized_bullet_and_appends_only_missing_fact():
    optimized = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "工程师",
            "period": "2020.01-2023.01",
            "bullets": ["负责梳理业务需求并形成可执行方案"],
        }],
    })
    fallback = CanonicalResume.model_validate({
        "experience": [{
            "organization": "甲公司",
            "role": "工程师",
            "period": "2020.01-2023.01",
            "bullets": ["负责需求分析", "负责项目交付"],
        }],
    })

    merged, stats = _merge_source_recovery(
        optimized,
        fallback,
        trusted_rewrites={
            "experience[0].bullets[0]": "负责需求分析",
        },
    )

    assert merged.experience[0].bullets == [
        "负责梳理业务需求并形成可执行方案",
        "负责项目交付",
    ]
    assert stats.appended_bullets == 1
    assert stats.appended_records == 0


def test_production_startup_shares_configurable_context_window():
    startup = (Path(__file__).parents[1] / "config" / "start.sh").read_text()

    assert 'MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"' in startup
    assert 'MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"' in startup
    assert 'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"' in startup
    assert 'export LLM_CONTEXT_WINDOW="$MAX_MODEL_LEN"' in startup
    assert 'export LLM_INFLIGHT_LIMIT="${LLM_INFLIGHT_LIMIT:-$MAX_NUM_SEQS}"' in startup
    assert '--max-model-len "$MAX_MODEL_LEN"' in startup
    assert '--max-num-seqs "$MAX_NUM_SEQS"' in startup
    assert '--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"' in startup

    for compose_name in ("docker-compose.yml", "config/docker-compose.yml"):
        compose = (Path(__file__).parents[1] / compose_name).read_text()
        assert "LLM_CONTEXT_WINDOW=${MAX_MODEL_LEN:-32768}" in compose
