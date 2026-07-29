"""V2 Pipeline orchestration.

Layers: SourceAdapter → Composer → Verifier → Optimizer → Validator
"""
from __future__ import annotations

import logging
import time

from v2_schemas import VerifiedResult, CanonicalResume, Meta
from source_adapter import build_source_bundle
from resume_composer import compose_resume
from resume_verifier import verify_resume
from resume_optimizer import optimize_resume
from v2_validator import validate_resume

logger = logging.getLogger(__name__)


def _canonical_to_v1_format(canonical: CanonicalResume) -> dict:
    """Bridge format for existing renderer compatibility."""
    data = canonical.model_dump()
    # Rename organization → company for V1 renderer
    for exp in data.get("experience", []):
        if isinstance(exp, dict) and "organization" in exp:
            exp["company"] = exp.pop("organization")
    for proj in data.get("projects", []):
        if isinstance(proj, dict) and "organization" in proj:
            proj["company"] = proj.pop("organization")
    # Merge research into experience (V1 renderer doesn't know research)
    research = data.pop("research", [])
    if research:
        for r in research:
            if isinstance(r, dict):
                data["experience"].append({
                    "organization": r.get("institution", ""),
                    "role": r.get("topic", "科研经历"),
                    "period": r.get("period", ""),
                    "bullets": r.get("bullets", []),
                    "company": r.get("institution", ""),
                })
    # Convert flat skills.items to V1 categorized format
    skills = data.get("skills", {})
    if isinstance(skills, dict):
        items = skills.pop("items", []) if isinstance(skills.get("items"), list) else []
        categorized: dict[str, list[str]] = {
            "languages": [], "frameworks": [], "tools": [], "domains": []}
        if items:
            for item in items:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    cat = item.get("category", "other")
                    if cat in categorized:
                        categorized[cat].append(name)
                    else:
                        categorized.setdefault("tools", []).append(name)
            skills.update(categorized)
    return data


def run_v2_pipeline(
    cv_text: str,
    query_text: str,
    jd_text: str,
) -> VerifiedResult:
    """Run the V2 5-layer pipeline. Returns VerifiedResult or fallback."""
    t_start = time.perf_counter()

    source = build_source_bundle(cv_text, query_text, jd_text)
    logger.info("V2 | SourceBundle: %d blocks (%.1fs)",
                len(source.blocks), time.perf_counter() - t_start)

    t_composer = time.perf_counter()
    draft = compose_resume(source)
    logger.info("V2 | Composer done: %d edu, %d exp, %d res, %d proj (%.1fs)",
                len(draft.education), len(draft.experience),
                len(draft.research), len(draft.projects),
                time.perf_counter() - t_composer)

    t_verifier = time.perf_counter()
    result = verify_resume(source, draft)
    logger.info("V2 | Verifier done: %d edu, %d exp, %d res, %d changes (%.1fs)",
                len(result.resume.education), len(result.resume.experience),
                len(result.resume.research), len(result.changes),
                time.perf_counter() - t_verifier)

    t_optimizer = time.perf_counter()
    result.resume = optimize_resume(result.resume, jd_text)
    logger.info("V2 | Optimizer done (%.1fs)", time.perf_counter() - t_optimizer)

    result.resume = validate_resume(result.resume)
    result.resume_dict = _canonical_to_v1_format(result.resume)

    logger.info("V2 | Total: %.1fs (Composer+Verifier+Validate+Format)",
                time.perf_counter() - t_start)

    return result
