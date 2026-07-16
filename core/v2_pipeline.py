"""V2 Pipeline orchestration.

Layers: SourceAdapter → Composer → Verifier → Validator
"""
from __future__ import annotations

import logging

from v2_schemas import VerifiedResult, CanonicalResume, Meta
from source_adapter import build_source_bundle
from resume_composer import compose_resume
from resume_verifier import verify_resume
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
    source = build_source_bundle(cv_text, query_text, jd_text)
    logger.info("V2 | SourceBundle: %d blocks", len(source.blocks))

    draft = compose_resume(source)
    logger.info("V2 | DraftResume: %d edu, %d exp, %d proj",
                len(draft.education), len(draft.experience), len(draft.projects))

    result = verify_resume(source, draft)
    logger.info("V2 | VerifiedResult: %d education, %d experiences, %d changes",
                len(result.resume.education), len(result.resume.experience),
                len(result.changes))

    result.resume = validate_resume(result.resume)

    # Attach V1-format dict for renderer compatibility
    result.resume_dict = _canonical_to_v1_format(result.resume)

    return result
