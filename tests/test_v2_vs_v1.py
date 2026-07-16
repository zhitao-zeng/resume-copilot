"""V2 vs V1 batch comparison on eval_cases.jsonl.

Run: PYTHONPATH=core MODELHUB_BASE_URL=http://localhost:8000/v1 MODELHUB_API_KEY=not-needed MODELHUB_MODEL_NAME=/models/Qwen3.5-8B python -m pytest tests/test_v2_vs_v1.py -v -s
"""
import json, os, sys
import logging
logging.basicConfig(level=logging.WARNING)

os.environ.setdefault("MODELHUB_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("MODELHUB_API_KEY", "not-needed")
os.environ.setdefault("MODELHUB_MODEL_NAME", "/models/Qwen3.5-8B")

from v2_pipeline import run_v2_pipeline

CASES = []
with open("eval_cases.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            CASES.append(json.loads(line))

print(f"Loaded {len(CASES)} eval cases\n")


def _summarize_v2(result) -> dict:
    r = result.resume
    meta = r.meta
    return {
        "name": meta.name or "(empty)",
        "target_role": meta.target_role or "(empty)",
        "edu": [e.school for e in r.education],
        "exp": [f"{e.organization}/{e.role}" for e in r.experience[:3]],
        "exp_count": len(r.experience),
        "proj": [p.name for p in r.projects[:3]],
        "skills": {
            k: v for k, v in (r.skills.model_dump().items() if r.skills else {})
        } if r.skills else {},
    }


import pytest

@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_v2_runs(case):
    """V2 must not crash on any eval case."""
    result = run_v2_pipeline(
        cv_text=case.get("cv_text", ""),
        query_text=case.get("query", ""),
        jd_text=case.get("jd_text", ""),
    )
    s = _summarize_v2(result)

    # Print summary for manual inspection
    print(f"\n[{case['id']}] name={s['name']} | role={s['target_role']} | "
          f"edu={s['edu']} | exp={s['exp_count']} | proj={len(s['proj'])}")

    # Basic assertions
    assert result is not None
    assert result.resume is not None

    # If cv has a real person name, V2 should extract it or leave empty
    # (not fabricate default names)
    if s["name"] and s["name"] not in ("(empty)",):
        # Must not be a default placeholder
        assert s["name"] not in ("张三", "李四", "王五", "用户", "test", "测试", "姓名"), (
            f"V2 fabricated default name: {s['name']}"
        )

    # School should not be "全国大学"
    for school in s["edu"]:
        assert "全国大学" not in school, f"V2 has award text in school: {school}"
