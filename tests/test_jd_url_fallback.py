import asyncio
from unittest.mock import patch

from quality_report import assess_jd_requirements
from resume_copilot_pipeline import _resolve_jd_text, stage_ingest
from source_adapter import build_source_bundle
from v2_schemas import CanonicalResume


def test_explicit_mixed_jd_url_uses_inline_text_when_fetch_fails():
    perf: dict[str, float] = {}
    warnings: list[dict[str, str]] = []

    with patch("resume_copilot_pipeline._fetch_jd_url", return_value=""):
        result = asyncio.run(_resolve_jd_text(
            target_jd=None,
            jd_text=None,
            target_jd_url="https://example.com/jobs/42、负责用户研究并输出需求文档",
            jd_url=None,
            target_jd_file=None,
            perf=perf,
            warnings=warnings,
        ))

    assert result == "负责用户研究并输出需求文档"


def test_url_only_fetch_failure_is_supplied_but_unavailable():
    with patch("resume_copilot_pipeline._fetch_jd_url", return_value=""):
        ctx = asyncio.run(stage_ingest(
            query="请优化我的简历",
            cv=None,
            cv_template=None,
            target_jd=None,
            jd_text=None,
            target_jd_url="https://example.com/jobs/42",
            jd_url=None,
            target_jd_file=None,
        ))

    assert ctx.has_jd is True
    assert ctx.jd_supplied is True
    assert ctx.jd_available is False
    assert ctx.jd_unavailable is True
    assert ctx.jd_text == ""


def test_unavailable_jd_report_has_no_invented_requirements():
    source = build_source_bundle("张三\n负责用户访谈", "", "")

    alignment = assess_jd_requirements(
        "",
        "产品经理",
        CanonicalResume(),
        [],
        source,
        jd_supplied=True,
        jd_unavailable=True,
    )

    assert alignment["has_job_description"] is True
    assert alignment["job_description_available"] is False
    assert alignment["source_status"] == "unavailable"
    assert alignment["requirements"] == []
    assert "无法读取" in alignment["recommendations"][0]


def test_no_jd_remains_not_provided():
    source = build_source_bundle("张三", "", "")

    alignment = assess_jd_requirements("", "", CanonicalResume(), [], source)

    assert alignment["has_job_description"] is False
    assert alignment["job_description_available"] is False
    assert alignment["source_status"] == "not_provided"
