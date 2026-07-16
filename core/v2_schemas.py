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


class SkillsDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")
    languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)


class CanonicalResume(BaseModel):
    model_config = ConfigDict(extra="ignore")  # LLM output may have extra fields
    meta: Meta = Field(default_factory=Meta)
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: SkillsDraft = Field(default_factory=SkillsDraft)
    summary: str = ""


# ---- DraftResume (plain strings, no evidence wrapper) ----
# Verifier judges factuality; Composer just extracts structure.

class DraftResume(BaseModel):
    model_config = ConfigDict(extra="ignore")  # LLM may add extra fields
    meta: Meta = Field(default_factory=Meta)
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    skills: SkillsDraft = Field(default_factory=SkillsDraft)
    summary: str = ""


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
