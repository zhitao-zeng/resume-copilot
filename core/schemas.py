"""Pydantic request/response schemas."""

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model: str
    llm_enabled: bool
    timestamp: str


class RevisionTarget(BaseModel):
    project: Optional[str] = None
    bullet_index: int
    company: Optional[str] = None
    exp_index: Optional[int] = None
    project_index: Optional[int] = None
    expected_before: Optional[str] = None


class RevisionType(str, Enum):
    content = "content"
    format = "format"
    both = "both"


class GenerateResponse(BaseModel):
    resume_data: dict[str, Any]
    files: dict[str, Optional[str]]
    score: float = 0.0
    score_breakdown: dict[str, Any] = Field(default_factory=dict)
    draft_id: str
    version: int = 0
    missing_fields: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    fabrication_report: dict[str, Any] = Field(default_factory=dict)
    user_report: dict[str, Any] = Field(default_factory=dict)
    generation_direction: str = ""
    reply_text: str = ""
    perf: dict[str, float] = Field(default_factory=dict)


class UserStage(str, Enum):
    student = "student"
    experienced = "experienced"
    job_seeker = "job_seeker"


class JobFamily(str, Enum):
    product_research = "product_research"
    operations = "operations"
    doctor = "doctor"
    teacher = "teacher"
    sales_presale = "sales_presale"
    finance = "finance"
    design = "design"
    education = "education"
    legal = "legal"
    other = "other"


class MissingField(BaseModel):
    field: str
    label: str
    reason: str
    source: str = "not_provided"  # "not_provided" | "extraction_lost" | "removed_unsupported"


class FieldConflict(BaseModel):
    field: str
    description: str


class FabricationDetail(BaseModel):
    type: str
    content: str
    reason: str


class FabricationReport(BaseModel):
    fabrication_found: bool = False
    details: list[FabricationDetail] = Field(default_factory=list)


class ResumeScore(BaseModel):
    fabrication: int
    readability: float
    completeness: float
    expression: float
    response: float
    total: float


class ScoreRequest(BaseModel):
    resume_data: dict[str, Any]
    original_text: Optional[str] = None
    user_report: Optional[dict[str, Any]] = None
    job_family: Optional[str] = None
    user_stage: Optional[str] = None
    missing_fields: Optional[list[dict[str, Any]]] = None
    conflicts: Optional[list[dict[str, Any]]] = None


class ScoreResponse(BaseModel):
    fabrication: int
    readability: float
    completeness: float
    expression: float
    response: float
    total: float


class PolishLLMRequest(BaseModel):
    resume_data: dict[str, Any]
    jd_text: Optional[str] = None
    template: str = "new_standard"


class AuditAndOptimizeRequest(BaseModel):
    resume_content: Optional[str] = None
    file_path: Optional[str] = None
    avatar_path: Optional[str] = None
    jd_text: Optional[str] = None
    query_text: Optional[str] = None
    target_description: Optional[str] = None
    user_stage: Optional[UserStage] = None
    style: str = "aggressive"
    template: Optional[str] = None
    draft_id: Optional[str] = None
    revision_instructions: Optional[str] = None
    revision_type: RevisionType = RevisionType.both
    revision_targets: Optional[list[RevisionTarget]] = None


class AuditAndOptimizeResponse(BaseModel):
    files: dict[str, Optional[str]]
    audit_report: dict[str, Any]
    optimization_history: list[dict[str, Any]]
    final_score: float
    draft_id: str
    version: int
    changes: list[dict[str, Any]]
    has_substantive_rewrite: bool = False
    substantive_change_count: int = 0
    user_report: dict[str, Any] = Field(default_factory=dict)
    mcp_tools_instruction: dict[str, Any] = Field(default_factory=dict)
    perf: dict[str, float] = Field(default_factory=dict)
    message: Optional[str] = None
    reply_text: str = ""
    scenario: str = ""
    industry: str = ""
    user_stage: str = ""
    score_breakdown: dict[str, Any] = Field(default_factory=dict)
    ocr_warnings: list[dict[str, Any]] = Field(default_factory=list)
    plain_text_output: Optional[str] = None


class ResumeCopilotResponse(BaseModel):
    files: dict[str, Optional[str]]
    reply_text: str
    score: float = 0.0
    missing_fields: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    scenario: str
    industry: str
    user_stage: str
    perf: dict[str, float] = Field(default_factory=dict)
    score_breakdown: dict[str, Any] = Field(default_factory=dict)
    ocr_warnings: list[dict[str, Any]] = Field(default_factory=list)
    user_report: dict[str, Any] = Field(default_factory=dict)
    resume_data: dict[str, Any] = Field(default_factory=dict)
    draft_id: str = ""
    version: int = 0


class JDProfileLLMOutput(BaseModel):
    job_family: str = ""
    seniority: str = ""
    must_have_keywords: list[str] = Field(default_factory=list)
    nice_to_have_keywords: list[str] = Field(default_factory=list)
    core_responsibilities: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)


class AuditDimensionScoresLLM(BaseModel):
    technical_depth: float = 0
    quantification: float = 0
    responsibility_clarity: float = 0
    authenticity: float = 0


class AuditIssueLLM(BaseModel):
    project: str = "项目经历"
    bullet_index: int = 1
    dimension: Literal["technical_depth", "quantification", "responsibility_clarity", "authenticity"] = "technical_depth"
    severity: Literal["high", "medium", "low"] = "medium"
    issue_type: Literal["actionable", "needs_data"] = "actionable"
    problem: str = ""
    suggestion: str = ""
    interviewer_question: str = ""


class JdAlignmentLLM(BaseModel):
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    coverage_score: float = 0


class AuditLLMOutput(BaseModel):
    overall_score: float = 0
    dimension_scores: AuditDimensionScoresLLM = Field(default_factory=AuditDimensionScoresLLM)
    issues: list[AuditIssueLLM] = Field(default_factory=list)
    jd_alignment: JdAlignmentLLM = Field(default_factory=JdAlignmentLLM)
    summary: str = ""


class OptimizeChangeLLM(BaseModel):
    project: str = "项目经历"
    bullet_index: int = 1
    before: str = ""
    after: str = ""
    reason: str = ""


class OptimizeLLMOutput(BaseModel):
    optimized_resume: dict[str, Any] = Field(default_factory=dict)
    changes: list[OptimizeChangeLLM] = Field(default_factory=list)


class OptimizeWithAuditLLMOutput(BaseModel):
    optimized_resume: dict[str, Any] = Field(default_factory=dict)
    audit_report: dict[str, Any] = Field(default_factory=dict)
    changes: list[OptimizeChangeLLM] = Field(default_factory=list)


class RevisionChangeLLM(BaseModel):
    location: str = ""
    before: str = ""
    after: str = ""
    reason: str = ""


class RevisionLLMOutput(BaseModel):
    resume_data: dict[str, Any] = Field(default_factory=dict)
    changes: list[RevisionChangeLLM] = Field(default_factory=list)


class StructuredResumeMetaLLM(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = ""
    age: str = ""
    gender: str = ""
    email: str = ""
    phone: str = ""
    wechat: str = ""
    github: str = ""
    linkedin: str = ""
    website: str = ""
    education_level: str = ""
    work_experience: str = ""
    political_status: str = ""
    expected_city: str = ""
    target_role: str = ""
    job_intention: str = ""


class StructuredResumeProjectLLM(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = ""
    company: str = ""
    role: str = ""
    period: str = ""
    description: str = ""
    function_description: str = ""
    result_description: str = ""
    bullets: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)


class StructuredResumeExperienceLLM(BaseModel):
    model_config = ConfigDict(extra="allow")
    company: str = ""
    role: str = ""
    team: str = ""
    period: str = ""
    function_description: str = ""
    result_description: str = ""
    bullets: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    projects: list[StructuredResumeProjectLLM] = Field(default_factory=list)


class StructuredResumeEducationLLM(BaseModel):
    model_config = ConfigDict(extra="allow")
    school: str = ""
    degree: str = ""
    major: str = ""
    period: str = ""
    highlights: list[str] = Field(default_factory=list)


class StructuredResumeSkillsLLM(BaseModel):
    model_config = ConfigDict(extra="allow")
    languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)


class StructuredResumeLLMOutput(BaseModel):
    model_config = ConfigDict(extra="allow")
    meta: StructuredResumeMetaLLM = Field(default_factory=StructuredResumeMetaLLM)
    summary: str = ""
    experience: list[StructuredResumeExperienceLLM] = Field(default_factory=list)
    projects: list[StructuredResumeProjectLLM] = Field(default_factory=list)
    education: list[StructuredResumeEducationLLM] = Field(default_factory=list)
    skills: StructuredResumeSkillsLLM = Field(default_factory=StructuredResumeSkillsLLM)
    publications: list[Any] = Field(default_factory=list)
    honors: list[str] = Field(default_factory=list)
    awards: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    personal_skills: list[str] = Field(default_factory=list)
    additional_sections: dict[str, Any] = Field(default_factory=dict)


class PolishLLMOutput(BaseModel):
    optimized_resume: dict[str, Any] = Field(default_factory=dict)
    polish_notes: list[str] = Field(default_factory=list)
