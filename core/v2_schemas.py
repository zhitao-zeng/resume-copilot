"""V2 Pipeline Schema Models.

SourceBundle, DraftResume (with GroundedValue), CanonicalResume (clean fields).
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Optional


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


class MetaDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: DraftField = Field(default_factory=DraftField)
    phone: DraftField = Field(default_factory=DraftField)
    email: DraftField = Field(default_factory=DraftField)
    target_role: DraftField = Field(default_factory=DraftField)


class EducationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    school: DraftField = Field(default_factory=DraftField)
    degree: DraftField = Field(default_factory=DraftField)
    major: DraftField = Field(default_factory=DraftField)
    period: DraftField = Field(default_factory=DraftField)


class ExperienceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    organization: DraftField = Field(default_factory=DraftField)
    role: DraftField = Field(default_factory=DraftField)
    period: DraftField = Field(default_factory=DraftField)
    bullets: list[DraftField] = Field(default_factory=list)


class ProjectDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: DraftField = Field(default_factory=DraftField)
    organization: DraftField = Field(default_factory=DraftField)
    role: DraftField = Field(default_factory=DraftField)
    period: DraftField = Field(default_factory=DraftField)


class SkillsDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    languages: list[DraftField] = Field(default_factory=list)
    frameworks: list[DraftField] = Field(default_factory=list)
    tools: list[DraftField] = Field(default_factory=list)
    domains: list[DraftField] = Field(default_factory=list)


class DraftResume(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meta: MetaDraft = Field(default_factory=MetaDraft)
    education: list[EducationDraft] = Field(default_factory=list)
    experience: list[ExperienceDraft] = Field(default_factory=list)
    projects: list[ProjectDraft] = Field(default_factory=list)
    skills: SkillsDraft = Field(default_factory=SkillsDraft)
    summary: DraftField = Field(default_factory=DraftField)


# ---- Canonical (clean, no DraftField) ----


class Meta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    phone: str = ""
    email: str = ""
    target_role: str = ""
    work_experience: str = ""


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


class CanonicalResume(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meta: Meta = Field(default_factory=Meta)
    education: list[Education] = Field(default_factory=list)
    experiences: list[Experience] = Field(default_factory=list)
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
