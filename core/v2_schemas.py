"""V2 Pipeline Schema Models.

SourceBundle, DraftResume (with GroundedValue), CanonicalResume (clean fields).
"""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any, Literal, Optional


class SourceBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block_id: str
    source_type: Literal["resume", "query", "jd"]
    text: str
    section_hint: Optional[str] = None
    # Deterministic source-side record boundary.  Evidence binding uses this to
    # prevent an organization/role/bullet from different jobs or projects from
    # being spliced into one generated record.
    record_id: Optional[str] = None
    # Query text mixes user instructions with optional factual additions.
    # Only blocks explicitly classified as candidate facts may support resume
    # claims. Resume blocks are always eligible; JD blocks never are.
    fact_eligible: bool = True


class SourceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blocks: list[SourceBlock]


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block_id: str
    quote: str


class DraftField(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: Optional[str] = None
    mode: Literal["direct", "normalized", "derived", "rewritten", "none"] = "none"
    evidence: list[EvidenceRef] = Field(default_factory=list)


# ---- Canonical (clean, no DraftField) ----


class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    phone: str = ""
    email: str = ""
    target_role: str = ""
    work_experience: str = ""

    @field_validator("work_experience", mode="before")
    @classmethod
    def normalize_work_experience(cls, v: Any) -> str:
        if isinstance(v, list):
            return ""
        value = str(v or "").strip()
        if not value:
            return ""
        # A model sometimes copies an employment period into this duration
        # field (for example "2024.07-2024.12").  A date range is not work
        # experience and is already represented on the experience record.
        date_tokens = re.findall(r"(?:19|20)\d{2}(?:[./年-]\d{1,2})?", value)
        if len(date_tokens) >= 2 or re.search(
            r"(?:19|20)\d{2}[./-]\d{1,2}\s*[-—~至到]\s*(?:19|20)\d{2}[./-]\d{1,2}",
            value,
        ):
            return ""
        # Keep explicit duration statements only.  Ambiguous values such as a
        # lone year are safer left blank than rendered as seniority.
        if re.search(r"\d+(?:\.\d+)?\s*(?:年|个月|月)(?:工作|从业|实习)?(?:经验|经历)?", value):
            return value
        if value in {"应届", "应届生", "无工作经验"}:
            return value
        return ""


class Education(BaseModel):
    model_config = ConfigDict(extra="forbid")
    school: str = ""
    degree: str = ""
    major: str = ""
    period: str = ""


class Experience(BaseModel):
    model_config = ConfigDict(extra="forbid")
    organization: str = ""
    role: str = ""
    period: str = ""
    bullets: list[str] = Field(default_factory=list)


class Project(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    organization: str = ""
    role: str = ""
    period: str = ""
    bullets: list[str] = Field(default_factory=list)


class Research(BaseModel):
    """Research/lab experience — distinct from employment (Experience).

    Student identity (研究生, 研究助理) belongs here, not in Experience.
    """
    model_config = ConfigDict(extra="forbid")
    institution: str = ""
    topic: str = ""
    period: str = ""
    bullets: list[str] = Field(default_factory=list)


class Activity(BaseModel):
    """Campus, association and volunteer experience kept out of employment."""
    model_config = ConfigDict(extra="forbid")
    organization: str = ""
    role: str = ""
    period: str = ""
    bullets: list[str] = Field(default_factory=list)


class SkillItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    category: str = ""


class SkillsDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")
    items: list[SkillItem] = Field(default_factory=list)

    @field_validator("items", mode="before")
    @classmethod
    def normalize_items(cls, v: Any) -> list:
        """Accept both flat list of SkillItem and old dict format."""
        if isinstance(v, dict):
            # Old format: {"languages": [...], "tools": [...]}
            items = []
            cat_map = {
                "languages": "language",
                "frameworks": "framework",
                "tools": "tool",
                "domains": "domain",
                "methodologies": "methodology",
                "certifications": "certification",
                "natural_languages": "natural_language",
                "others": "other",
            }
            for cat, names in v.items():
                target_cat = cat_map.get(cat, cat)
                if isinstance(names, list):
                    for n in names:
                        if isinstance(n, str):
                            items.append({"name": n, "category": target_cat})
            return items
        if isinstance(v, list):
            return v
        return []


class CanonicalResume(BaseModel):
    model_config = ConfigDict(extra="ignore")
    meta: Meta = Field(default_factory=Meta)
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    research: list[Research] = Field(default_factory=list)
    activities: list[Activity] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: SkillsDraft = Field(default_factory=SkillsDraft)
    summary: str = ""
    awards: list[str] = Field(default_factory=list)
    publications: list[str] = Field(default_factory=list)
    patents: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    training: list[str] = Field(default_factory=list)
    teaching: list[str] = Field(default_factory=list)
    # Flexible fallback for legitimate long-tail industries and source
    # sections that do not fit a fixed taxonomy. Keys are original section
    # titles and values are verbatim factual entries.
    additional_sections: dict[str, list[str]] = Field(default_factory=dict)


# ---- DraftResume (plain strings, no evidence wrapper) ----

class DraftResume(BaseModel):
    model_config = ConfigDict(extra="ignore")
    meta: Meta = Field(default_factory=Meta)
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    research: list[Research] = Field(default_factory=list)
    activities: list[Activity] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: SkillsDraft = Field(default_factory=SkillsDraft)
    summary: str = ""
    awards: list[str] = Field(default_factory=list)
    publications: list[str] = Field(default_factory=list)
    patents: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    training: list[str] = Field(default_factory=list)
    teaching: list[str] = Field(default_factory=list)
    additional_sections: dict[str, list[str]] = Field(default_factory=dict)


class Change(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    action: Literal["clear", "remove", "replace"]
    reason: str


class EvidenceBinding(BaseModel):
    """Internal trace from one final resume path to candidate source text."""

    model_config = ConfigDict(extra="forbid")
    path: str
    block_id: str
    quote: str
    # Final claim text is kept internally so reverse coverage can verify which
    # source fact units actually survived, rather than treating one binding as
    # coverage for an entire OCR line.
    claim: str = ""
    # For an optimizer rewrite that passed the hard fact guard, retain the
    # exact pre-rewrite claim used to establish provenance.  Coverage checks
    # use this source claim while renderers and audits still see ``claim`` as
    # the final user-facing wording.
    source_claim: str = ""
    mode: Literal["direct", "normalized", "rewritten", "derived"]
    similarity: float = 1.0


class VerifiedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resume: CanonicalResume
    changes: list[Change] = Field(default_factory=list)
    resume_dict: dict = Field(default_factory=dict)
    evidence_bindings: list[EvidenceBinding] = Field(default_factory=list)
