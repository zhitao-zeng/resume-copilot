"""Deterministic evidence tracing for final V2 resume content.

Bindings are internal audit metadata.  They never appear in the rendered
resume, and JD blocks are intentionally excluded from candidate facts.
"""
from __future__ import annotations

import re
import unicodedata

from v2_schemas import CanonicalResume, EvidenceBinding, SourceBlock, SourceBundle


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)


def _bigrams(value: str) -> set[str]:
    normalized = _normalize(value)
    return {normalized[index:index + 2] for index in range(max(0, len(normalized) - 1))}


def _date_signature(value: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", str(value or ""))
    pattern = re.compile(
        r"(?<!\d)(?:(?P<year1>(?:19|20)\d{2})\s*[-./年]\s*(?P<month1>0?[1-9]|1[0-2])\s*月?"
        r"|(?P<month2>0?[1-9]|1[0-2])\s*[-./]\s*(?P<year2>(?:19|20)\d{2}))(?!\d)"
    )
    signature: list[str] = []
    for match in pattern.finditer(text):
        year = match.group("year1") or match.group("year2")
        month = match.group("month1") or match.group("month2")
        signature.append(f"{year}{int(month):02d}")
    return tuple(signature)


def _best_block(value: str, blocks: list[SourceBlock]) -> tuple[SourceBlock | None, float, str]:
    normalized_value = _normalize(value)
    if not normalized_value:
        return None, 0.0, "rewritten"
    for block in blocks:
        if normalized_value in _normalize(block.text):
            mode = "direct" if str(value).casefold() in block.text.casefold() else "normalized"
            return block, 1.0, mode

    date_signature = _date_signature(value)
    if date_signature:
        for block in blocks:
            block_signature = _date_signature(block.text)
            if len(block_signature) >= len(date_signature) and any(
                block_signature[index:index + len(date_signature)] == date_signature
                for index in range(len(block_signature) - len(date_signature) + 1)
            ):
                return block, 1.0, "normalized"

    target = _bigrams(value)
    best_block = None
    best_score = 0.0
    for block in blocks:
        candidate = _bigrams(block.text)
        if not candidate:
            continue
        score = len(target & candidate) / max(1, min(len(target), len(candidate)))
        if score > best_score:
            best_block, best_score = block, score
    return best_block, best_score, "rewritten"


def _bind(path: str, value: str, blocks: list[SourceBlock], *, minimum: float = 0.22) -> EvidenceBinding | None:
    block, similarity, mode = _best_block(value, blocks)
    if block is None or similarity < minimum:
        return None
    quote = block.text.strip()
    if len(quote) > 240:
        normalized_value = _normalize(value)
        position = _normalize(quote).find(normalized_value)
        if position >= 0:
            # Normalized offsets are approximate; retaining the beginning is
            # safer than claiming an exact character span after punctuation
            # removal.
            quote = quote[:240]
        else:
            quote = quote[:240]
    return EvidenceBinding(
        path=path,
        block_id=block.block_id,
        quote=quote,
        mode=mode,
        similarity=round(similarity, 4),
    )


def bind_resume_evidence(resume: CanonicalResume, source: SourceBundle) -> list[EvidenceBinding]:
    """Bind final fields and bullets to Resume/Query blocks, never JD blocks."""

    blocks = [block for block in source.blocks if block.source_type in {"resume", "query"}]
    bindings: list[EvidenceBinding] = []

    def add(path: str, value: str, minimum: float = 0.22) -> None:
        binding = _bind(path, value, blocks, minimum=minimum)
        if binding is not None:
            bindings.append(binding)

    for key in ("name", "phone", "email", "work_experience"):
        add(f"meta.{key}", getattr(resume.meta, key), 0.8)

    section_fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, fields in section_fields.items():
        for index, record in enumerate(getattr(resume, section)):
            for field in fields:
                add(f"{section}[{index}].{field}", getattr(record, field), 0.65)
            if hasattr(record, "bullets"):
                for bullet_index, bullet in enumerate(record.bullets):
                    add(f"{section}[{index}].bullets[{bullet_index}]", bullet)

    for index, skill in enumerate(resume.skills.items):
        add(f"skills.items[{index}].name", skill.name, 0.8)
    for index, award in enumerate(resume.awards):
        add(f"awards[{index}]", award, 0.8)
    return bindings


def enforce_resume_evidence(
    resume: CanonicalResume,
    source: SourceBundle,
) -> tuple[CanonicalResume, list[EvidenceBinding], list[str]]:
    """Remove final candidate claims that cannot bind to Resume/Query evidence.

    ``meta.target_role`` is intentionally exempt because it is a target
    direction and may come from JD.  Summary is rebuilt deterministically after
    this gate, so it is not treated as an independent candidate claim.
    """

    gated = resume.model_copy(deep=True)
    initial = bind_resume_evidence(gated, source)
    bound_paths = {binding.path for binding in initial}
    candidate_blocks = [
        block for block in source.blocks
        if block.source_type in {"resume", "query"}
    ]
    removed: list[str] = []

    for key in ("name", "phone", "email", "work_experience"):
        path = f"meta.{key}"
        if getattr(gated.meta, key) and path not in bound_paths:
            setattr(gated.meta, key, "")
            removed.append(path)

    section_fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, fields in section_fields.items():
        for index, record in enumerate(getattr(gated, section)):
            for field in fields:
                path = f"{section}[{index}].{field}"
                if getattr(record, field) and path not in bound_paths:
                    setattr(record, field, "")
                    removed.append(path)
            if not hasattr(record, "bullets"):
                continue
            kept_bullets: list[str] = []
            for bullet_index, bullet in enumerate(record.bullets):
                path = f"{section}[{index}].bullets[{bullet_index}]"
                binding = _bind(path, bullet, candidate_blocks, minimum=0.30)
                if binding is None:
                    removed.append(path)
                else:
                    kept_bullets.append(bullet)
            record.bullets = kept_bullets

    kept_skills = []
    for index, skill in enumerate(gated.skills.items):
        path = f"skills.items[{index}].name"
        if path in bound_paths:
            kept_skills.append(skill)
        else:
            removed.append(path)
    gated.skills.items = kept_skills

    kept_awards: list[str] = []
    for index, award in enumerate(gated.awards):
        path = f"awards[{index}]"
        if path in bound_paths:
            kept_awards.append(award)
        else:
            removed.append(path)
    gated.awards = kept_awards

    bindings = bind_resume_evidence(gated, source)
    return gated, bindings, removed
