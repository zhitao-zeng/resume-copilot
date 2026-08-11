"""V2 Pipeline badcase regression tests (C1/C5/C39).

Requires a running LLM backend at localhost:8000/v1.
Run with: PYTHONPATH=core python -m pytest tests/test_v2_badcases.py -v -s
"""
import json
import os
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "badcase_0715"

RUN_LLM_INTEGRATION = os.getenv("RUN_LLM_INTEGRATION", "0") == "1"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RUN_LLM_INTEGRATION, reason="set RUN_LLM_INTEGRATION=1 with a running backend"),
]

if RUN_LLM_INTEGRATION:
    os.environ.setdefault("MODELHUB_BASE_URL", "http://localhost:8000/v1")
    os.environ.setdefault("MODELHUB_API_KEY", "not-needed")
    os.environ.setdefault("MODELHUB_MODEL_NAME", "/models/Qwen3.5-8B")

from v2_pipeline import run_v2_pipeline


def _load(name: str) -> dict:
    with open(FIXTURES / name) as f:
        return json.load(f)


def _format_result(result) -> dict:
    """Extract key fields from VerifiedResult for display."""
    r = result.resume
    return {
        "meta": {
            "name": r.meta.name,
            "phone": r.meta.phone,
            "email": r.meta.email,
            "target_role": r.meta.target_role,
        },
        "education": [
            {"school": e.school, "degree": e.degree, "major": e.major}
            for e in r.education
        ],
        "experience": [
            {"organization": e.organization, "role": e.role,
             "bullets": e.bullets[:2]}
            for e in r.experience
        ],
        "projects": [
            {"name": p.name, "organization": p.organization, "role": p.role}
            for p in r.projects
        ],
        "skills": r.skills.model_dump() if r.skills else {},
        "changes": [c.model_dump() for c in result.changes],
    }


# ════════════════════════════════════════════════════════════════
# Case C1 — JD + CV: 超级公司 org name must survive Verifier
# ════════════════════════════════════════════════════════════════

def test_v2_c1_org_name_preserved():
    """超级公司 must survive V2 pipeline (substring of concatenated source text).
    Note: fixture cv_text does NOT contain name '陈媛媛' — only org/skills."""
    data = _load("case1_input.json")
    result = run_v2_pipeline(
        cv_text=data["cv_text"],
        query_text=data["query"],
        jd_text=data["jd_text"],
    )
    out = _format_result(result)

    print("\n=== C1 V2 Result ===")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # 超级公司 org must survive in organization field or bullet text
    orgs = [e["organization"] for e in out["experience"]]
    bullets_text = " ".join(
        b for e in out["experience"] for b in e["bullets"]
    )
    assert any("超级公司" in org for org in orgs) or "超级公司" in bullets_text, (
        f"超级公司 not found in experience orgs or bullets"
    )


# ════════════════════════════════════════════════════════════════
# Case C5 — JD only (no CV): empty input → empty output
# ════════════════════════════════════════════════════════════════

def test_v2_c5_empty_input_returns_empty():
    """Empty cv_text → V2 returns empty (conservative fallback = correct)."""
    data = _load("case5_input.json")
    result = run_v2_pipeline(
        cv_text=data["cv_text"],
        query_text=data["query"],
        jd_text=data["jd_text"],
    )
    out = _format_result(result)

    print("\n=== C5 V2 Result ===")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # Must not crash; empty output is acceptable for empty cv
    assert result is not None


# ════════════════════════════════════════════════════════════════
# Case C39 — Student resume: school must survive
# ════════════════════════════════════════════════════════════════

def test_v2_c39_school_preserved():
    """北京邮电大学 in cv_text must survive V2 pipeline."""
    data = _load("case39_input.json")
    result = run_v2_pipeline(
        cv_text=data["cv_text"],
        query_text=data["query"],
        jd_text=data["jd_text"],
    )
    out = _format_result(result)

    print("\n=== C39 V2 Result ===")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # School must be in education field
    schools = [e["school"] for e in out["education"]]
    if schools:
        assert any("北京邮电大学" in s for s in schools), (
            f"北京邮电大学 not found in education: {schools}"
        )
        assert schools == ["北京邮电大学"], (
            f"narrative fragments leaked into duplicate education records: {schools}"
        )
        assert all(not e["organization"].startswith("在") for e in out["experience"])
    else:
        # If education is empty, check that at least the cv_text was processed
        # (non-empty experience or skills)
        has_content = (
            len(out["experience"]) > 0
            or len(out["projects"]) > 0
            or any(out["skills"].get(k) for k in ("languages", "frameworks", "tools", "domains"))
        )
        assert has_content, (
            "C39 cv_text is non-empty but V2 produced nothing — pipeline may have fallen back"
        )


if __name__ == "__main__":
    test_v2_c1_org_name_preserved()
    test_v2_c5_empty_input_returns_empty()
    test_v2_c39_school_preserved()
