"""Draft version persistence helpers."""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from http_compat import HTTPException


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _sanitize_draft_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", value or ""):
        raise HTTPException(status_code=400, detail="invalid draft_id")
    return value


def _draft_file_path(drafts_dir: Path, draft_id: str) -> Path:
    return drafts_dir / f"{_sanitize_draft_id(draft_id)}.json"


def load_draft_state(drafts_dir: Path, draft_id: str) -> dict[str, Any]:
    draft_path = _draft_file_path(drafts_dir, draft_id)
    if not draft_path.exists():
        raise HTTPException(status_code=404, detail=f"draft not found: {draft_id}")
    try:
        return json.loads(draft_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to load draft: {exc}") from exc


def save_draft_state(drafts_dir: Path, state: dict[str, Any]) -> None:
    draft_id = state.get("draft_id")
    if not isinstance(draft_id, str):
        raise HTTPException(status_code=500, detail="invalid draft state")
    draft_path = _draft_file_path(drafts_dir, draft_id)
    draft_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def create_new_draft(
    drafts_dir: Path,
    resume_data: dict[str, Any],
    audit_report: dict[str, Any],
    jd_text: Optional[str],
    template: str,
    output_format: str,
    changes: Optional[list[dict[str, Any]]] = None,
) -> tuple[str, int]:
    draft_id = uuid.uuid4().hex[:10]
    state = {
        "draft_id": draft_id,
        "latest_version": 1,
        "created_at": _now_iso(),
        "versions": [
            {
                "version": 1,
                "updated_at": _now_iso(),
                "resume_data": resume_data,
                "audit_report": audit_report,
                "jd_text": jd_text,
                "template": template,
                "output_format": output_format,
                "changes": changes or [],
            }
        ],
    }
    save_draft_state(drafts_dir, state)
    return draft_id, 1


def append_draft_version(
    drafts_dir: Path,
    state: dict[str, Any],
    resume_data: dict[str, Any],
    audit_report: dict[str, Any],
    jd_text: Optional[str],
    template: str,
    output_format: str,
    changes: Optional[list[dict[str, Any]]] = None,
) -> tuple[str, int]:
    latest = int(state.get("latest_version", 0))
    version = latest + 1
    state["latest_version"] = version
    state.setdefault("versions", []).append(
        {
            "version": version,
            "updated_at": _now_iso(),
            "resume_data": resume_data,
            "audit_report": audit_report,
            "jd_text": jd_text,
            "template": template,
            "output_format": output_format,
            "changes": changes or [],
        }
    )
    save_draft_state(drafts_dir, state)
    return str(state["draft_id"]), version
