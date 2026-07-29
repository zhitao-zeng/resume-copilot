"""V2 Pipeline Schema Models.

SourceBundle, DraftResume (with GroundedValue), CanonicalResume (clean fields).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Any, Literal, Optional


class SourceBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    block_id: str
    source_type: Literal["resume", "query", "jd"]
    text: str
    section_hint: Optional[str] = None


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
        return str(v) if v else ""


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
            cat_map = {"languages": "language", "frameworks": "framework",
                       "tools": "tool", "domains": "domain"}
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
    projects: list[Project] = Field(default_factory=list)
    skills: SkillsDraft = Field(default_factory=SkillsDraft)
    summary: str = ""
    awards: list[str] = Field(default_factory=list)


# ---- DraftResume (plain strings, no evidence wrapper) ----

class DraftResume(BaseModel):
    model_config = ConfigDict(extra="ignore")
    meta: Meta = Field(default_factory=Meta)
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    research: list[Research] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: SkillsDraft = Field(default_factory=SkillsDraft)
    summary: str = ""
    awards: list[str] = Field(default_factory=list)


class Change(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    action: Literal["clear", "remove", "replace"]
    reason: str


class VerifiedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resume: CanonicalResume
    changes: list[Change] = Field(default_factory=list)
    resume_dict: dict = Field(default_factory=dict)
