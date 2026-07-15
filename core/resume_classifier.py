"""LLM-based request classifier for resume-copilot.

Determines industry, user_stage, and target_role using LLM with evidence.
Falls back to keyword rules when LLM is unavailable or confidence is low.
Scenario detection remains rule-based (depends on file existence, not text).

Evidence tracks source (query|cv|jd) and usage (fact|direction):
- Fact fields (industry, user_stage) only allow query/cv
- Target_role and industry can reference JD direction
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

import resume_product_logic as product_logic
from resume_product_logic import _is_student_with_internals
from server_runtime import call_llm_typed, llm_enabled

logger = logging.getLogger(__name__)

VALID_USER_STAGES = {"student", "experienced", "job_seeker"}

NON_ROLE_HEADERS = {
    "岗位职责", "职责", "任职要求", "任职资格", "岗位要求", "职位要求",
    "工作职责", "工作内容", "要求", "资格", "职责描述", "岗位描述",
    "job responsibilities", "requirements", "qualifications",
}

_SYSTEM_PROMPT = (
    "你是简历生成服务的请求分类器。只做分类，不生成简历内容。\n"
    "硬约束：\n"
    "1. 事实字段（industry、user_stage）的证据必须来自 query 或 cv_text，不得来自 JD。\n"
    "2. target_role 可以参考 JD 的方向，但必须与用户输入中的意向一致。\n"
    "3. 证据不足时 confidence 降低，使用 other/job_seeker/空字符串。\n"
    "4. 只输出 JSON，不要输出解释。\n"
    "5. target_role 不得是'职责'、'任职要求'等 JD 章节标题。\n\n"
    "industry 在充分判断后自由输出，不限于任何固定列表。\n"
    "允许的 user_stage：student, experienced, job_seeker\n\n"
    "evidence 格式：[{\"text\": \"关键词\", \"source\": \"query|cv|jd\", \"usage\": \"fact|direction\"}]\n"
    "- 事实字段 source 只能是 query 或 cv；target_role 的 industry 参考可来自 jd\n"
    "- usage: fact 表示用户已有事实；direction 表示方向参考\n\n"
    '输出 JSON：{"industry":"other","user_stage":"job_seeker","target_role":"","confidence":0.0,'
    '"evidence":{"industry":[],"user_stage":[],"target_role":[]},"warnings":[]}'
)


class _EvidenceItem(BaseModel):
    text: str = ""
    source: str = ""
    usage: str = ""


class _LLMEvidence(BaseModel):
    industry: list[_EvidenceItem] = Field(default_factory=list)
    user_stage: list[_EvidenceItem] = Field(default_factory=list)
    target_role: list[_EvidenceItem] = Field(default_factory=list)


class _ClassifierLLMOutput(BaseModel):
    industry: str = "other"
    user_stage: str = "job_seeker"
    target_role: str = ""
    confidence: float = 0.0
    evidence: _LLMEvidence = Field(default_factory=_LLMEvidence)
    warnings: list[str] = Field(default_factory=list)


class EvidenceSource(BaseModel):
    """Structured evidence item."""
    text: str
    source: str = ""  # query | cv | jd
    usage: str = ""   # fact | direction


class ClassificationEvidence(BaseModel):
    industry: list[EvidenceSource] = Field(default_factory=list)
    user_stage: list[EvidenceSource] = Field(default_factory=list)
    target_role: list[EvidenceSource] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.industry) + len(self.user_stage) + len(self.target_role)

    @property
    def has_fact_evidence(self) -> bool:
        for bucket in (self.industry, self.user_stage):
            for e in bucket:
                if e.usage == "fact" and e.text:
                    return True
        return False

    def to_legacy_dict(self) -> dict[str, list[str]]:
        return {
            "industry": [e.text for e in self.industry],
            "user_stage": [e.text for e in self.user_stage],
            "target_role": [e.text for e in self.target_role],
        }


class ClassificationResult(BaseModel):
    industry: str = "other"
    user_stage: str = "job_seeker"
    target_role: str = ""
    confidence: float = 0.0
    evidence: ClassificationEvidence = Field(default_factory=ClassificationEvidence)
    warnings: list[str] = Field(default_factory=list)
    used_llm: bool = False

    def to_legacy_dict(self) -> dict[str, list[str]]:
        return self.evidence.to_legacy_dict()


def _extract_cv_summary(cv_text: str, max_chars: int = 800) -> str:
    """Extract key info from cv_text for classification: experience titles, skills, education."""
    if not cv_text or len(cv_text) <= max_chars:
        return cv_text or ""
    lines = cv_text.splitlines()
    summary_lines: list[str] = []
    # Heuristic: keep lines that look like section headers, job titles, degrees, or skills
    header_pattern = re.compile(r"(教育|经历|工作|项目|技能|荣誉|实习|自我|求职|专业|公司|院校|学位|岗位)")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Keep section headers
        if header_pattern.search(stripped) and len(stripped) < 60:
            summary_lines.append(stripped)
            continue
        # Keep short lines that look like titles/companies/degrees (likely key info)
        if len(stripped) < 80 and any(kw in stripped for kw in
            ("工程师", "经理", "总监", "分析师", "设计师", "开发", "算法", "产品", "运营",
             "大学", "学院", "硕士", "本科", "博士", "专业", "有限公司", "集团")):
            summary_lines.append(stripped)
        if sum(len(l) for l in summary_lines) >= max_chars:
            break
    result = "\n".join(summary_lines)
    return result if result else cv_text[:max_chars]


def _build_classifier_prompt(*, query: str, cv_text: str, jd_text: str, has_cv: bool, has_jd: bool) -> str:
    cv_summary = _extract_cv_summary(cv_text) if cv_text else ""
    return (
        f"has_cv: {has_cv}\n"
        f"has_jd: {has_jd}\n\n"
        f"query:\n{(query or '')[:2000]}\n\n"
        f"cv_summary (key roles, education, skills):\n{cv_summary[:800]}\n\n"
        f"jd_text:\n{(jd_text or '')[:2000]}\n"
    )


def _classify_via_llm(*, query: str, cv_text: str, jd_text: str, has_cv: bool, has_jd: bool) -> Optional[_ClassifierLLMOutput]:
    """Call LLM for lightweight classification. Returns None on failure."""
    if not llm_enabled():
        return None
    try:
        result = call_llm_typed(
            output_model=_ClassifierLLMOutput,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=_build_classifier_prompt(query=query, cv_text=cv_text, jd_text=jd_text, has_cv=has_cv, has_jd=has_jd),
            temperature=0.1,
            max_tokens=1024,
        )
        if isinstance(result, dict):
            return _ClassifierLLMOutput(**result)
        return None
    except Exception as exc:
        logger.warning("LLM classification failed, falling back to rules: %s", exc)
        return None


def _sanitize_target_role(role: str) -> str:
    role = str(role or "").strip()
    if not role or len(role) < 2:
        return ""
    if role.lower() in {h.lower() for h in NON_ROLE_HEADERS}:
        return ""
    for header in NON_ROLE_HEADERS:
        if role == header or role.endswith(header):
            role = role[: -len(header)].strip("：:，,。.、/\\")
        if role.startswith(header):
            role = role[len(header):].strip("：:，,。.、/\\")
    if not role or len(role) < 2:
        return ""
    return role


def _extract_target_role_rules(query: str, jd_text: str) -> str:
    text = "\n".join([query or "", jd_text or ""])
    for pattern in [
        r"(?:目标岗位|求职意向|应聘岗位|想投|想做|应聘)[:：\s]*([\u4e00-\u9fffA-Za-z0-9/ ]{2,30})",
        r"(?:招聘岗位|招聘职位|岗位名称)[:：\s]*([\u4e00-\u9fffA-Za-z0-9/ ]{2,30})",
    ]:
        match = re.search(pattern, text)
        if match:
            role = _sanitize_target_role(match.group(1).strip("，。；; "))
            if role:
                return role
    for role in ("产品经理", "运营经理", "运营专员", "医生", "教师", "老师",
                 "销售经理", "售前工程师", "金融风控", "交互设计师", "UI设计师",
                 "算法工程师", "软件工程师", "数据分析师", "项目经理"):
        if role in text:
            return role
    return ""


def _rule_fallback(*, query: str, cv_text: str, jd_text: str, resume_data: Optional[dict[str, Any]] = None) -> ClassificationResult:
    combined_text = "\n".join(part for part in (query, cv_text) if part.strip())
    industry = product_logic.infer_industry(query, cv_text, jd_text)
    user_stage = product_logic.infer_user_stage(combined_text, resume_data)
    target_role = _extract_target_role_rules(query, jd_text)
    return ClassificationResult(
        industry=industry, user_stage=user_stage, target_role=target_role,
        confidence=0.4,
        evidence=ClassificationEvidence(
            industry=[EvidenceSource(text="rule-based", source="query", usage="fact")],
            user_stage=[EvidenceSource(text="rule-based", source="query", usage="fact")],
            target_role=[EvidenceSource(text="rule-based", source="query", usage="fact")],
        ),
        warnings=["Rule-based fallback for classification."],
        used_llm=False,
    )


# English category name → Chinese target_role mapping
_EN_TO_ZH_ROLE = {
    "finance_risk": "金融风控",
    "financial_analyst": "金融分析师",
    "accountant": "会计师",
    "auditor": "审计师",
    "investment_banker": "投行分析师",
    "doctor": "内科医师",
    "surgeon": "外科医师",
    "pediatrician": "儿科医师",
    "general_practitioner": "全科医生",
    "nurse": "护士",
    "education_ops": "教务运营",
    "curriculum_designer": "课程设计师",
    "product_manager": "产品经理",
    "operations_manager": "运营经理",
    "operations_staff": "运营专员",
    "teacher": "教师",
    "college_professor": "教授/讲师",
    "sales_engineer": "售前工程师",
    "sales_manager": "销售经理",
    "algorithm_engineer": "算法工程师",
    "software_engineer": "软件工程师",
    "data_analyst": "数据分析师",
    "project_manager": "项目经理",
    "legal_counsel": "法律顾问",
    "lawyer": "律师",
    "paralegal": "法务专员",
    "compliance_officer": "合规官",
    "ui_designer": "UI设计师",
    "ux_designer": "UX设计师",
    "visual_designer": "视觉设计师",
    "product_designer": "产品设计师",
    "architect": "建筑设计师",
}

def _validate_and_correct(
    result: ClassificationResult,
    *,
    resume_data: Optional[dict[str, Any]] = None,
    query: str = "",
    cv_text: str = "",
) -> ClassificationResult:
    warnings = list(result.warnings)

    # 1. Validate user_stage (industry is free-text, not validated against a fixed list)
    if result.user_stage not in VALID_USER_STAGES:
        warnings.append(f"Invalid user_stage '{result.user_stage}', correcting to 'job_seeker'")
        result.user_stage = "job_seeker"

    # 3. Sanitize target_role
    result.target_role = _sanitize_target_role(result.target_role)

    # 3.1. Map English category names → Chinese
    if result.target_role in _EN_TO_ZH_ROLE:
        zh_role = _EN_TO_ZH_ROLE[result.target_role]
        warnings.append(f"target_role '{result.target_role}' mapped to '{zh_role}'")
        result.target_role = zh_role

    # 4. Student + experience: nuanced check
    # Don't override student→experienced for internships, campus projects, part-time
    if result.user_stage == "student" and isinstance(resume_data, dict):
        experience = resume_data.get("experience", [])
        if isinstance(experience, list) and len(experience) >= 1:
            work_years = product_logic.calculate_experience_years(experience)
            # If years >= 3, likely graduated full-time
            if work_years >= 3:
                warnings.append(f"user_stage overridden to 'experienced': {work_years} years work experience")
                result.user_stage = "experienced"
            elif not _is_student_with_internals(experience):
                # No internship keywords and has experience — check if any company name looks post-graduation
                # If experience exists and work_years >= 2, override
                if work_years >= 2:
                    warnings.append(f"user_stage overridden to 'experienced': {work_years} years work experience")
                    result.user_stage = "experienced"

    # 5. Empty evidence warning
    if result.industry != "other" and not result.evidence.industry:
        warnings.append(f"Industry '{result.industry}' has no evidence; may be unreliable")

    # 6. JD-only target_role warning
    if result.target_role and result.evidence.target_role:
        role_evidence = result.evidence.target_role
        jd_only = all(e.source == "jd" for e in role_evidence)
        query_role = _extract_target_role_rules(query, "")
        if jd_only and not query_role:
            warnings.append(f"target_role '{result.target_role}' only supported by JD")

    # 7. Empty evidence fallback
    if result.evidence.total == 0:
        warnings.append("LLM returned empty evidence; falling back to rules")

    result.warnings = warnings
    return result


def classify_resume_request(
    *,
    query: str = "",
    cv_text: str = "",
    jd_text: str = "",
    has_cv: bool = False,
    has_jd: bool = False,
    resume_data: Optional[dict[str, Any]] = None,
) -> ClassificationResult:
    """Classify a resume request using LLM + rule fallback. Single LLM call."""
    llm_result = _classify_via_llm(query=query, cv_text=cv_text, jd_text=jd_text, has_cv=has_cv, has_jd=has_jd)

    if llm_result is not None:
        # Convert LLM evidence to structured format
        industry_list = [
            e.model_dump() if hasattr(e, "model_dump") else (e if isinstance(e, dict) else {"text": e})
            for e in llm_result.evidence.industry
        ]
        user_stage_list = [
            e.model_dump() if hasattr(e, "model_dump") else (e if isinstance(e, dict) else {"text": e})
            for e in llm_result.evidence.user_stage
        ]
        target_role_list = [
            e.model_dump() if hasattr(e, "model_dump") else (e if isinstance(e, dict) else {"text": e})
            for e in llm_result.evidence.target_role
        ]
        evidence = ClassificationEvidence(
            industry=[EvidenceSource(**e) for e in industry_list],
            user_stage=[EvidenceSource(**e) for e in user_stage_list],
            target_role=[EvidenceSource(**e) for e in target_role_list],
        )
        result = ClassificationResult(
            industry=llm_result.industry,
            user_stage=llm_result.user_stage,
            target_role=llm_result.target_role,
            confidence=llm_result.confidence,
            evidence=evidence,
            warnings=llm_result.warnings or [],
            used_llm=True,
        )

        # High confidence: use LLM result directly
        if result.confidence >= 0.6 and result.evidence.has_fact_evidence:
            return _validate_and_correct(result, resume_data=resume_data, query=query, cv_text=cv_text)

        # Low confidence OR empty evidence: merge with rules
        rule_result = _rule_fallback(query=query, cv_text=cv_text, jd_text=jd_text, resume_data=resume_data)

        if result.confidence < 0.6 or not result.evidence.has_fact_evidence:
            logger.info("LLM classification confidence %.2f < 0.6 or no fact evidence, merging with rules", result.confidence)
            if not result.evidence.has_fact_evidence:
                result.industry = rule_result.industry
                result.evidence = rule_result.evidence
            # Keep LLM target_role if it has evidence, otherwise use rules
            if not result.target_role or not result.evidence.target_role:
                result.target_role = rule_result.target_role

        return _validate_and_correct(result, resume_data=resume_data, query=query, cv_text=cv_text)

    # LLM failed entirely — pure rule fallback
    return _rule_fallback(query=query, cv_text=cv_text, jd_text=jd_text, resume_data=resume_data)
