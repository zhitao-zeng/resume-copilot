from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = ROOT / "validation_sets/public_resume_holdout/evaluate.py"
SPEC = importlib.util.spec_from_file_location("public_holdout_evaluator", EVALUATOR_PATH)
assert SPEC and SPEC.loader
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def _annotation(tmp_path: Path) -> dict:
    source = "姓名：张三\n甲公司 | 产品经理 | 2022-2024\n负责用户调研并输出PRD\n[学校]"
    source_path = tmp_path / "source.txt"
    source_path.write_text(source, encoding="utf-8")
    # Dataset paths are constrained beneath the evaluator root. For this unit
    # test, point its root at the isolated temporary directory.
    EVALUATOR.ROOT = tmp_path
    eligible_end = source.index("\n[学校]")
    return {
        "case_id": "fixture",
        "sources": [{
            "kind": "cv",
            "canonical_text_path": "source.txt",
            "sha256": EVALUATOR._sha256(source_path),
            "candidate_for_resume": True,
            "units": [
                {
                    "candidate_for_resume": True,
                    "source_span": [0, eligible_end],
                },
                {
                    "candidate_for_resume": False,
                    "source_span": [eligible_end + 1, len(source)],
                },
            ],
        }],
    }


def test_external_audit_uses_only_eligible_source_spans(tmp_path: Path) -> None:
    annotation = _annotation(tmp_path)
    response = {
        "resume_data": {
            "meta": {"name": "张三"},
            "experience": [{
                "company": "甲公司",
                "role": "产品经理",
                "period": "2022-2024",
                "bullets": ["负责用户调研并输出PRD"],
            }],
            "education": [{"school": "[学校]"}],
        },
    }

    audit = EVALUATOR.audit_response(response, annotation)
    atomic = audit["atomic_factuality"]
    structural = audit["structural_invariants"]
    assert atomic["supported_atom_count"] >= 4
    assert atomic["unsupported_atom_count"] == 1
    assert structural["education"]["added_count"] == 1


def test_summary_is_micro_aggregated_and_ignores_candidate_score() -> None:
    def row(case_id: str, supported: int, generated: int, represented: int, facts: int) -> dict:
        return {
            "id": case_id,
            "scenario": "scenario1",
            "industry": "operations",
            "input_profile": "plain_text",
            "request_ok": True,
            "audit_ok": True,
            "elapsed_s": 10,
            "raw": {"score": {"total": 100 if case_id == "a" else 0}},
            "response_contract": {
                "scenario_match": True,
                "industry_match": True,
                "docx_present": True,
                "reply_present": True,
                "expected_missing": {},
                "reply_components": {name: True for name in EVALUATOR.REPLY_COMPONENTS},
                "reported_but_written": [],
            },
            "external_audit": {
                "atomic_factuality": {
                    "generated_atom_count": generated,
                    "supported_atom_count": supported,
                    "source_fact_count": facts,
                    "represented_source_fact_count": represented,
                    "precision": supported / generated,
                    "recall": represented / facts,
                },
                "ownership_integrity": {
                    "correct_assignment_count": 1,
                    "incorrect_assignment_count": 0,
                    "undetermined_assignment_count": 0,
                },
                "structural_invariants": {
                    category: {"added_count": 0, "missing_count": 0}
                    for category in EVALUATOR.ALL_STRUCTURAL_CATEGORIES
                },
            },
        }

    summary = EVALUATOR.summarize_rows([
        row("a", 1, 1, 1, 1),
        row("b", 1, 3, 1, 3),
    ])
    assert summary["atomic_factuality"]["micro_precision"] == 0.5
    assert summary["atomic_factuality"]["micro_recall"] == 0.5
    assert "score" not in summary


def test_shadow_split_is_rejected_by_cli_contract() -> None:
    source = EVALUATOR_PATH.read_text(encoding="utf-8")
    assert '"shadow_v3" in cases_path.parts' in source
