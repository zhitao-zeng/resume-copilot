#!/usr/bin/env python3
"""R27 task-5 acceptance: replay the retired holdout fixtures end-to-end.

The six cases retired by the holdout-hygiene task are replayed from their
canonical source texts through the deterministic pipeline (use_llm=False).
The assertions are the plan's acceptance: no fragment_start /
unbalanced_bracket bullets reach the frozen resume, conflicts never leak
internal tokens, and missing-field reporting uses the three-state contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "holdout_retired"
for path in (str(REPO_ROOT), str(REPO_ROOT / "core")):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.v3.pipeline import run_v3_pipeline  # noqa: E402
from core.v3.text_integrity import bullet_defects  # noqa: E402

_DROP_DEFECTS = {"fragment_start", "unbalanced_bracket"}


def _fixture_cases() -> list[tuple[dict, dict]]:
    cases = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (FIXTURES / "cases.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    annotations = {
        row["case_id"]: row
        for row in (
            json.loads(line)
            for line in (FIXTURES / "annotations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    return [(cases[cid], annotations[cid]) for cid in sorted(cases)]


def _case_inputs(annotation: dict) -> dict[str, str]:
    inputs = {"cv_text": "", "query_text": "", "jd_text": ""}
    root = REPO_ROOT / "validation_sets" / "public_resume_holdout"
    for source in annotation.get("sources") or []:
        text_path = root / source["canonical_text_path"]
        text = text_path.read_text(encoding="utf-8")
        if source["kind"] == "cv":
            inputs["cv_text"] = text
        elif source["kind"] == "query":
            inputs["query_text"] = text
        elif source["kind"] == "jd":
            inputs["jd_text"] = text
    return inputs


@pytest.mark.parametrize("case,annotation", _fixture_cases(), ids=lambda item: item.get("id") or item.get("case_id"))
def test_retired_fixture_replay(case, annotation):
    inputs = _case_inputs(annotation)
    result = run_v3_pipeline(
        cv_text=inputs["cv_text"],
        query_text=inputs["query_text"],
        jd_text=inputs["jd_text"],
        use_llm=False,
    )
    # 1. No fragment/broken-bracket bullet reaches the frozen resume.
    for claim in result.output.frozen.claims:
        if claim.field != "bullet":
            continue
        defects = set(bullet_defects(claim.text))
        assert not (defects & _DROP_DEFECTS), (case["id"], claim.text, defects)
    # 2. User-visible conflicts never contain internal tokens.
    for conflict in result.conflicts:
        description = conflict.get("description", "")
        assert "record:" not in description
        assert "unassigned" not in description
    # 3. Missing-field sources stay within the three-state contract.
    for item in result.missing_fields:
        assert item.get("source") in {"not_provided", "not_rendered"}
    # 4. The pipeline never fails a fixture outright.
    assert result.resume_data is not None


# R27 task 8 step 4: bare_fragment is intentionally kept out of _DROP_DEFECTS
# (dropping short bullets would kill legitimate short facts like 熟练英语).
# Measured baseline on the 6 retired fixtures: 4, all bare job-title values
# (物业协调员/行政助理/餐饮服务员/物流专员 in HV2-S3-009).  This test measures
# the residual; it does not change the tradeoff.
BARE_FRAGMENT_BASELINE = 4


def test_bare_fragment_residual_within_baseline():
    residual = 0
    for case, annotation in _fixture_cases():
        inputs = _case_inputs(annotation)
        result = run_v3_pipeline(
            cv_text=inputs["cv_text"],
            query_text=inputs["query_text"],
            jd_text=inputs["jd_text"],
            use_llm=False,
        )
        for claim in result.output.frozen.claims:
            if claim.field == "bullet" and "bare_fragment" in bullet_defects(claim.text):
                residual += 1
    assert residual <= BARE_FRAGMENT_BASELINE


def test_additional_sections_never_reported_not_provided():
    """R27 task 8 step 5 (D1 regression): routed additional content is not missing."""

    for case, annotation in _fixture_cases():
        inputs = _case_inputs(annotation)
        result = run_v3_pipeline(
            cv_text=inputs["cv_text"],
            query_text=inputs["query_text"],
            jd_text=inputs["jd_text"],
            use_llm=False,
        )
        additional = result.resume_data.get("additional_sections") or {}
        if not additional.get("教育补充信息"):
            continue
        for item in result.missing_fields:
            if item["field"] == "education":
                assert item["source"] == "not_rendered", case["id"]

