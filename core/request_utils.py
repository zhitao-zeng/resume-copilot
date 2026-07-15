from typing import Optional

from http_compat import HTTPException
import json

from schemas import RevisionTarget, RevisionType
def _normalize_revision_type(value: Optional[str | RevisionType]) -> str:
    if isinstance(value, RevisionType):
        return value.value
    revision_type = (value or "both").strip().lower()
    if revision_type not in {"content", "format", "both"}:
        raise HTTPException(status_code=400, detail="revision_type must be content/format/both")
    return revision_type


def _parse_revision_targets_form(raw_value: Optional[str]) -> Optional[list[RevisionTarget]]:
    if raw_value is None or not raw_value.strip():
        return None

    try:
        parsed = json.loads(raw_value)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid revision_targets_json: {exc}") from exc

    if not isinstance(parsed, list):
        raise HTTPException(status_code=400, detail="revision_targets_json must be a JSON array")

    targets: list[RevisionTarget] = []
    for idx, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"revision_targets[{idx}] must be a JSON object")
        try:
            targets.append(RevisionTarget(**item))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid revision_targets[{idx}]: {exc}") from exc
    return targets


def infer_format_preferences(instructions: Optional[str], template: str, output_format: str) -> tuple[str, str]:
    if not instructions:
        return template, output_format

    text = instructions.lower()
    resolved_template = template
    resolved_output_format = output_format

    if any(x in text for x in ["极简", "minimal"]):
        resolved_template = "minimal"
    elif any(x in text for x in ["现代", "modern"]):
        resolved_template = "modern"
    elif any(x in text for x in ["经典", "classic"]):
        resolved_template = "classic"

    wants_pdf_only = any(x in text for x in ["只要pdf", "仅pdf", "pdf版", "导出pdf"])
    wants_docx_only = any(x in text for x in ["只要docx", "仅docx", "word版", "导出word", "导出docx"])
    wants_both = any(x in text for x in ["都要", "两种", "both", "pdf和docx", "docx和pdf"])

    if wants_both:
        resolved_output_format = "both"
    elif wants_pdf_only and not wants_docx_only:
        resolved_output_format = "pdf"
    elif wants_docx_only and not wants_pdf_only:
        resolved_output_format = "docx"

    return resolved_template, resolved_output_format

