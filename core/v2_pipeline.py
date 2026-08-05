"""V2 Pipeline orchestration.

Layers: SourceAdapter → Composer → Verifier → Optimizer → Validator
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata

from v2_schemas import VerifiedResult, CanonicalResume, DraftResume, Meta, Change
from source_adapter import (
    _is_section_heading,
    _looks_like_record_body,
    build_source_bundle,
    candidate_blocks,
)
from resume_composer import compose_resume, compose_from_query
from resume_verifier import verify_resume
from resume_verifier import _ground_fixed_fields, _reclassify_non_work
from resume_optimizer import optimize_resume, _introduces_unsupported_fact
from v2_validator import validate_resume
from evidence_binding import (
    bind_resume_evidence,
    enforce_resume_evidence,
    measure_source_coverage,
    source_fact_units,
)
import resume_product_logic as product_logic

logger = logging.getLogger(__name__)


def _is_empty_resume(resume: CanonicalResume) -> bool:
    return not any((
        resume.meta.name,
        resume.meta.phone,
        resume.meta.email,
        resume.education,
        resume.experience,
        resume.research,
        resume.activities,
        resume.projects,
        resume.skills.items,
        resume.summary,
        resume.awards,
        resume.publications,
        resume.patents,
        resume.certifications,
        resume.training,
        resume.teaching,
        resume.additional_sections,
    ))


def _has_candidate_profile(resume: CanonicalResume) -> bool:
    """Whether any candidate fact remains after grounding and evidence gates."""

    return any((
        resume.meta.name,
        resume.meta.phone,
        resume.meta.email,
        resume.meta.work_experience,
        resume.education,
        resume.experience,
        resume.research,
        resume.activities,
        resume.projects,
        resume.skills.items,
        resume.awards,
        resume.publications,
        resume.patents,
        resume.certifications,
        resume.training,
        resume.teaching,
        resume.additional_sections,
    ))


_STRONG_OWNERSHIP = ("主导", "统筹", "牵头", "独立负责", "全权负责", "从0到1", "从零到一")
_MEDIUM_OWNERSHIP = ("负责", "组织", "推动", "管理", "设计", "开发", "构建", "实现", "制定")
_WEAK_OWNERSHIP = ("参与", "协助", "支持", "配合", "接触", "了解", "学习")
_RESULT_CLAIMS = (
    "显著提升", "大幅提升", "提升了", "降低了", "减少了", "增长了", "增强了",
    "确保", "保障", "关键依据", "高质量交付", "打通", "性能达标", "降低成本",
    "提高准确率", "提升准确率", "提升效率", "提升用户体验",
)

_SKILL_CATEGORY_ALIASES = {
    "language": "language",
    "programming_language": "language",
    "programming language": "language",
    "编程语言": "language",
    "framework": "framework",
    "library": "framework",
    "框架": "framework",
    "库": "framework",
    "tool": "tool",
    "software": "tool",
    "platform": "tool",
    "工具": "tool",
    "软件": "tool",
    "平台": "tool",
    "domain": "domain",
    "professional": "domain",
    "专业领域": "domain",
    "业务领域": "domain",
    "method": "methodology",
    "methodology": "methodology",
    "process": "methodology",
    "方法": "methodology",
    "流程": "methodology",
    "certification": "certification",
    "certificate": "certification",
    "license": "certification",
    "证书": "certification",
    "资质": "certification",
    "natural_language": "natural_language",
    "natural language": "natural_language",
    "spoken_language": "natural_language",
    "自然语言": "natural_language",
    "语言能力": "natural_language",
    "other": "other",
    "其他": "other",
}

_NATURAL_LANGUAGE_SKILL = re.compile(
    r"(?:英语|英文|日语|日文|韩语|韩文|法语|德语|西班牙语|俄语|普通话|粤语|"
    r"CET[-\s]?[46]|TEM[-\s]?[48]|IELTS|TOEFL|雅思|托福|JLPT|TOPIK|HSK)",
    re.IGNORECASE,
)
_CERTIFICATION_SKILL = re.compile(
    r"(?:证书|资格证|执业资格|职业资格|认证|执照|license|certificate|certified)",
    re.IGNORECASE,
)
_QUANTIFIED_CHANGE = re.compile(
    r"(?:从\s*\d[^，。；]{0,24}(?:降至|提升至|提高到|增长至|减少到|缩短至|达到)\s*\d|"
    r"(?:降低|提升|提高|增长|减少|缩短|节省)[^，。；]{0,12}\d)",
    re.IGNORECASE,
)
_RESULT_SIGNAL = re.compile(
    r"(?:降至|提升至|提高到|增长至|减少到|缩短至|达到|完成|上线|交付|录用|获奖|复核|验证)",
    re.IGNORECASE,
)
_SUMMARY_SUBJECTIVE = re.compile(
    r"(?:扎实|敏锐|优秀|热爱|致力于|较强|出色|卓越|丰富的|良好(?:的)?(?:能力|素养)|"
    r"责任心|学习能力|抗压能力|团队精神|积极主动|自驱力|"
    r"清晰的问题拆解|执行闭环|以真实岗位职责和结果为准)"
)


def _action_level(text: str) -> int:
    if any(token in text for token in _STRONG_OWNERSHIP):
        return 3
    if any(token in text for token in _MEDIUM_OWNERSHIP):
        return 2
    if any(token in text for token in _WEAK_OWNERSHIP):
        return 1
    return 0


def _char_bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", str(text or ""))
    return {compact[i:i + 2] for i in range(max(0, len(compact) - 1))}


def _source_sentences(text: str) -> list[str]:
    values = [item.strip(" \t-•") for item in re.split(r"[\n。；]+", text) if item.strip()]
    return [item for item in values if len(item) >= 6]


def _closest_source_sentence(bullet: str, sentences: list[str]) -> tuple[str, float, float]:
    target = _char_bigrams(bullet)
    if not target:
        return "", 0.0, 0.0
    best, score, best_recall = "", 0.0, 0.0
    for sentence in sentences:
        candidate = _char_bigrams(sentence)
        if not candidate:
            continue
        shared = len(target & candidate)
        generated_coverage = shared / max(1, len(target))
        source_recall = shared / max(1, len(candidate))
        if (generated_coverage, source_recall) > (score, best_recall):
            best, score, best_recall = sentence, generated_coverage, source_recall
    return best, score, best_recall


def _ground_bullet_value(value: str, sentences: list[str]) -> tuple[str, str]:
    """Return ``(text, status)`` where status is accepted/restored/dropped."""

    source, generated_coverage, source_recall = _closest_source_sentence(value, sentences)
    if not source:
        logger.info("Dropped ungrounded bullet: %s", value[:80])
        return "", "dropped"
    # If the model retained a recognizable source clause but added unsupported
    # material, restore the complete source sentence. Very weak matches are
    # dropped instead of being legitimized by a common verb such as “负责”.
    if generated_coverage < 0.58:
        if generated_coverage >= 0.18 and source_recall >= 0.45:
            logger.info("Restored source wording for weakly grounded bullet: %s", value[:80])
            return source, "restored"
        logger.info("Dropped weakly grounded bullet: %s", value[:80])
        return "", "dropped"
    upgraded = _action_level(value) > _action_level(source)
    unsupported_result = any(term in value and term not in source for term in _RESULT_CLAIMS)
    unsupported_fact = _introduces_unsupported_fact(source, value)
    if upgraded or unsupported_result or unsupported_fact:
        logger.info("Restored source wording for over-claimed bullet: %s", value[:80])
        return source, "restored"
    return value, "accepted"


def _ground_bullets(resume: CanonicalResume, evidence_text: str) -> CanonicalResume:
    """Fall back to the nearest source sentence when a bullet upgrades facts."""

    grounded = resume.model_copy(deep=True)
    sentences = _source_sentences(evidence_text)
    for section in (grounded.experience, grounded.research, grounded.activities, grounded.projects):
        for record in section:
            safe_bullets: list[str] = []
            for bullet in record.bullets:
                value = str(bullet or "").strip()
                if not value:
                    continue
                safe_value, _status = _ground_bullet_value(value, sentences)
                if safe_value and safe_value not in safe_bullets:
                    safe_bullets.append(safe_value)
            record.bullets = safe_bullets
    return grounded


def _ground_optimizer_output(
    original: CanonicalResume,
    optimized: CanonicalResume,
    evidence_text: str,
) -> CanonicalResume:
    """Revert any optimizer patch that fails the final evidence grounder.

    Optimizer patches are optional wording changes. Losing a factual bullet is
    never an acceptable consequence of rejecting one, so the already-verified
    original is retained at the same stable index.
    """

    grounded = optimized.model_copy(deep=True)
    sentences = _source_sentences(evidence_text)
    for section_name in ("experience", "research", "activities", "projects"):
        before_records = getattr(original, section_name)
        after_records = getattr(grounded, section_name)
        for record_index, after_record in enumerate(after_records):
            before_bullets = (
                list(before_records[record_index].bullets)
                if record_index < len(before_records) else []
            )
            safe_bullets: list[str] = []
            for bullet_index, bullet in enumerate(after_record.bullets):
                value = str(bullet or "").strip()
                safe_value, status = _ground_bullet_value(value, sentences)
                if status != "accepted" and bullet_index < len(before_bullets):
                    safe_value = str(before_bullets[bullet_index] or "").strip()
                    logger.info(
                        "Reverted unsupported optimizer patch at %s[%d].bullets[%d]",
                        section_name,
                        record_index,
                        bullet_index,
                    )
                if safe_value and safe_value not in safe_bullets:
                    safe_bullets.append(safe_value)
            after_record.bullets = safe_bullets
    return grounded


def _normalize_skill_category(name: str, category: str) -> str:
    """Normalize broad semantic buckets without maintaining an industry lexicon."""

    if _CERTIFICATION_SKILL.search(name):
        return "certification"
    if _NATURAL_LANGUAGE_SKILL.search(name):
        return "natural_language"
    normalized = re.sub(r"[-_/]+", "_", str(category or "").strip().lower())
    return _SKILL_CATEGORY_ALIASES.get(normalized, "other")


def _normalize_skill_name(name: str) -> str:
    value = unicodedata.normalize("NFKC", str(name or "")).strip()
    value = re.sub(r"\s+", " ", value)
    # Models vary between "DOE实验设计" and "DOE 实验设计".  This is
    # typographic normalization, not an industry dictionary.
    value = re.sub(r"(?<=[A-Za-z0-9])\s+(?=[\u4e00-\u9fff])", "", value)
    return value


def _bullet_priority(text: str, target_context: str = "") -> tuple[int, int, int, int]:
    """Rank an existing bullet by relevance and evidence density."""

    value = str(text or "")
    target_bigrams = _char_bigrams(target_context)
    overlap = len(_char_bigrams(value) & target_bigrams) if target_bigrams else 0
    quantified_change = int(bool(_QUANTIFIED_CHANGE.search(value)))
    result_signal = int(bool(_RESULT_SIGNAL.search(value)))
    numeric = int(bool(re.search(r"\d", value)))
    percentage = int(bool(re.search(r"\d+(?:\.\d+)?\s*%", value)))
    monetary = int(bool(re.search(r"\d+(?:\.\d+)?\s*(?:万|亿)?元", value)))
    multiple_metrics = int(len(re.findall(r"\d+(?:\.\d+)?", value)) >= 2)
    evidence_density = (
        quantified_change * 4 + result_signal * 2 + numeric
        + percentage * 3 + monetary * 3 + multiple_metrics * 2
    )
    # A verified before/after result should normally lead a project even when
    # a descriptive bullet repeats more JD keywords.  Very strong relevance
    # can still win through the combined score.
    return overlap + evidence_density * 3, evidence_density, overlap, min(len(value), 120)


def _rank_resume_content(resume: CanonicalResume, target_context: str = "") -> CanonicalResume:
    """Put relevant and verifiable bullets first without changing their text."""

    ranked = resume.model_copy(deep=True)
    for section in (ranked.experience, ranked.research, ranked.activities, ranked.projects):
        for record in section:
            record.bullets = sorted(
                record.bullets,
                key=lambda value: _bullet_priority(value, target_context),
                reverse=True,
            )
    return ranked


def _needs_optimizer(resume: CanonicalResume) -> bool:
    """Every factual bullet gets a dedicated evidence-preserving edit pass."""

    bullets = [
        str(bullet).strip()
        for section in (resume.experience, resume.research, resume.activities, resume.projects)
        for record in section
        for bullet in record.bullets
        if str(bullet).strip()
    ]
    return bool(bullets)


def _bullet_rewrite_changes(
    before: CanonicalResume,
    after: CanonicalResume,
) -> list[Change]:
    """Expose only bullet edits that the safety gate actually accepted."""

    changes: list[Change] = []
    for section_name in ("experience", "research", "activities", "projects"):
        before_records = getattr(before, section_name)
        after_records = getattr(after, section_name)
        for record_index, (before_record, after_record) in enumerate(
            zip(before_records, after_records)
        ):
            for bullet_index, (old_text, new_text) in enumerate(
                zip(before_record.bullets, after_record.bullets)
            ):
                if str(old_text).strip() == str(new_text).strip():
                    continue
                changes.append(Change(
                    path=f"{section_name}[{record_index}].bullets[{bullet_index}]",
                    action="replace",
                    reason="Evidence-preserving bullet rewrite",
                ))
    return changes


_COVERAGE_LAYOUT_LABEL = re.compile(
    r"^(?:个人信息|基本信息|联系方式|经历|工作经验|实习经验|项目经验|"
    r"技能|核心技能|其他信息|补充信息)$"
)
_COVERAGE_BOILERPLATE = re.compile(
    r"^(?:候选人)?具备清晰的问题拆解和执行闭环能力$|"
    r"^过往经历以真实岗位职责和结果为准$"
)
_COVERAGE_QUERY_DIRECTION = re.compile(
    r"^(?:请|帮我|麻烦)?(?:保留|突出|侧重|优化|润色|修改|调整|删除|去掉|"
    r"不要|避免|针对|适配)"
)
_COVERAGE_QUERY_FACT_SIGNAL = re.compile(
    r"(?:我(?:叫|是|会|有|曾|在|负责|参与|主导|获得|毕业|就读|熟悉|擅长)|"
    r"本人|姓名是|曾任|任职于|就职于|毕业于|就读于|"
    r"\d+(?:\.\d+)?\s*(?:年|个月)\s*(?:工作|从业|实习)?(?:经验|经历))"
)
_COVERAGE_ANCHOR = re.compile(
    r"\d+(?:\.\d+)?%?|[A-Za-z][A-Za-z0-9+.#/_@-]*",
    re.IGNORECASE,
)


def _coverage_normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)


def _coverage_bigrams(value: str) -> set[str]:
    normalized = _coverage_normalize(value)
    return {
        normalized[index:index + 2]
        for index in range(max(0, len(normalized) - 1))
    }


def _coverage_unit_is_optional(unit: dict[str, str]) -> bool:
    """Exclude extraction scaffolding, never candidate facts, from recall.

    Source coverage is a completeness signal, not a second truth gate.  OCR and
    DOCX extraction can emit layout labels that have no destination in the
    canonical schema.  A small amount of stock summary copy is similarly not a
    candidate fact and is replaced by the grounded summary builder later.
    """

    text = str(unit.get("match_text", "") or "").strip()
    compact = re.sub(r"[\s:：|｜/\\【】\[\]()（）]+", "", text)
    if not compact or _is_section_heading(compact) or _COVERAGE_LAYOUT_LABEL.fullmatch(compact):
        return True
    if _COVERAGE_BOILERPLATE.fullmatch(compact):
        return True
    if (
        unit.get("source_type") == "query"
        and _COVERAGE_QUERY_DIRECTION.search(text)
        and not _COVERAGE_QUERY_FACT_SIGNAL.search(text)
    ):
        # SourceAdapter deliberately errs on the side of retaining ambiguous
        # query clauses.  Editing directions such as “保留原学校” must not make
        # a complete Composer draft look incomplete.
        return True
    return False


def _coverage_unit_is_represented(
    unit: dict[str, str],
    claims_by_block: dict[str, list[str]],
    all_claims: list[str],
) -> bool:
    """Recognize split structured fields without weakening factual gates.

    A source line such as ``学校 + 学历 + 专业 + 日期`` becomes four canonical
    fields.  The legacy reverse-coverage check compares each field separately
    with the whole source line, so all four can be truth-bound while the line is
    still reported missing.  Here exact cross-block duplicates and the union of
    claims bound to the *same* source block are accepted.  Every numeric/Latin
    anchor must survive, which keeps dates, metrics and named tools strict.
    """

    source_text = str(unit.get("match_text", "") or "").strip()
    source_value = _coverage_normalize(source_text)
    if not source_value:
        return True

    normalized_claims: list[str] = []
    for claim in all_claims:
        normalized_claim = _coverage_normalize(claim)
        if normalized_claim:
            normalized_claims.append(normalized_claim)
    if any(source_value == claim_value for claim_value in normalized_claims):
        # This also resolves duplicated name/contact lines whose binding points
        # at the first identical occurrence in the extracted document.
        return True

    block_claims = [
        str(claim).strip()
        for claim in claims_by_block.get(str(unit.get("block_id", "")), [])
        if str(claim).strip()
    ]
    if len(block_claims) < 2:
        return False

    source_anchors = {
        anchor.casefold() for anchor in _COVERAGE_ANCHOR.findall(source_text)
    }
    claim_anchors = {
        anchor.casefold()
        for claim in block_claims
        for anchor in _COVERAGE_ANCHOR.findall(claim)
    }
    if source_anchors and not source_anchors.issubset(claim_anchors):
        return False

    source_bigrams = _coverage_bigrams(source_text)
    claim_bigrams = set().union(*(_coverage_bigrams(claim) for claim in block_claims))
    recall = len(source_bigrams & claim_bigrams) / max(1, len(source_bigrams))
    # Only field-boundary bigrams should be absent. A lower threshold can hide
    # a genuinely omitted short field such as a degree or role.
    return recall >= 0.82


def _deterministic_source_coverage(
    source,
    bindings,
) -> tuple[float, list[str]]:
    """Measure representable source recall for the Verifier fast path."""

    units = [unit for unit in source_fact_units(source) if not _coverage_unit_is_optional(unit)]
    if not units:
        return 1.0, []

    _, legacy_missing_values = measure_source_coverage(source, bindings)
    legacy_missing = set(legacy_missing_values)
    claims_by_block: dict[str, list[str]] = {}
    all_claims: list[str] = []
    for binding in bindings:
        claim = str(binding.claim or "").strip()
        if claim:
            claims_by_block.setdefault(binding.block_id, []).append(claim)
            all_claims.append(claim)

    missing: list[str] = []
    for unit in units:
        # ``measure_source_coverage`` remains the authority for ordinary
        # one-claim fuzzy rewrites.  Only reconsider units it marked missing.
        if (
            unit["unit_id"] in legacy_missing
            and not _coverage_unit_is_represented(unit, claims_by_block, all_claims)
        ):
            missing.append(unit["unit_id"])

    ratio = (len(units) - len(missing)) / len(units)
    return round(ratio, 4), missing


def _deterministic_verify_draft(
    source,
    draft: DraftResume,
) -> VerifiedResult | None:
    """Accept a clean Composer draft without paying for a second LLM call."""

    candidate_evidence = "\n".join(block.text for block in candidate_blocks(source))
    data = draft.model_dump()
    _ground_fixed_fields(data, candidate_evidence)
    _reclassify_non_work(data, candidate_evidence)
    try:
        resume = CanonicalResume.model_validate(data)
    except Exception:
        return None
    resume = _ground_bullets(resume, candidate_evidence)
    resume, bindings, removed = enforce_resume_evidence(resume, source)
    resume = _compact_canonical(resume)
    if _is_empty_resume(resume) or len(bindings) < 3:
        return None
    raw_coverage, raw_missing = measure_source_coverage(source, bindings)
    coverage, missing_blocks = _deterministic_source_coverage(source, bindings)
    if missing_blocks and coverage < 0.80:
        logger.info(
            "V2 | Deterministic verifier rejected draft: representable coverage %.1f%% "
            "(raw %.1f%%), missing=%s",
            coverage * 100,
            raw_coverage * 100,
            missing_blocks[:8],
        )
        return None
    if raw_missing and coverage > raw_coverage:
        logger.info(
            "V2 | Deterministic verifier resolved structural coverage artifacts: "
            "%.1f%% -> %.1f%%",
            raw_coverage * 100,
            coverage * 100,
        )
    if len(removed) > max(2, int(len(bindings) * 0.15)):
        logger.info("V2 | Deterministic verifier rejected draft: %d unbound claims", len(removed))
        return None
    changes = [
        Change(path=path, action="remove", reason="No candidate evidence binding")
        for path in removed
    ]
    logger.info(
        "V2 | Deterministic verifier accepted: %d bindings, %d removals",
        len(bindings), len(removed),
    )
    return VerifiedResult(resume=resume, changes=changes, evidence_bindings=bindings)


def _best_achievement(resume: CanonicalResume) -> str:
    candidates = [
        str(bullet).strip()
        for section in (resume.projects, resume.experience, resume.research, resume.activities)
        for record in section
        for bullet in record.bullets
        if str(bullet).strip()
    ]
    if not candidates:
        return ""
    best = max(candidates, key=lambda value: _bullet_priority(value))
    if not (_QUANTIFIED_CHANGE.search(best) or _RESULT_SIGNAL.search(best)):
        return ""
    return best.strip("。；; ")


def _build_evidence_summary(resume: CanonicalResume) -> str:
    """Build a stable summary only from fields that already passed grounding.

    This deliberately avoids industry-specific templates and subjective claims.
    Unknown professions therefore receive the same factual treatment as known
    ones, without requiring an ever-growing keyword dictionary.
    """

    if not any((
        resume.summary,
        resume.meta.work_experience,
        resume.education,
        resume.experience,
        resume.research,
        resume.activities,
        resume.projects,
        resume.skills.items,
        resume.awards,
        resume.publications,
        resume.patents,
        resume.certifications,
        resume.training,
        resume.teaching,
        resume.additional_sections,
    )):
        return ""

    candidates: list[str] = []

    target_role = resume.meta.target_role.strip()
    if re.fullmatch(r"[a-z0-9]+(?:[_-][a-z0-9]+)+", target_role, re.IGNORECASE):
        target_role = " ".join(
            token.upper() if len(token) <= 3 else token.title()
            for token in re.split(r"[_-]+", target_role)
            if token
        )
    if target_role:
        candidates.append("求职方向为" + target_role)

    # A source-grounded summary can carry valuable facts (for example eight
    # years of clinical work) that are not duplicated in another field. Keep
    # complete factual sentences, while filtering subjective filler.
    for sentence in (
        item.strip() for item in re.split(r"[。！？!?；;]+", resume.summary) if item.strip()
    ):
        if not _SUMMARY_SUBJECTIVE.search(sentence):
            candidates.append(sentence)

    experience_bits: list[str] = []
    for item in resume.experience[:2]:
        identity = "".join(part.strip() for part in (item.organization, item.role) if part.strip())
        if identity and identity not in experience_bits:
            experience_bits.append(identity)
    if experience_bits:
        seniority = resume.meta.work_experience.strip()
        if seniority and re.fullmatch(r"\d+(?:\.\d+)?\s*(?:年|个月|月)", seniority):
            seniority += "经验"
        prefix = f"有{seniority}，" if seniority else ""
        candidates.append(prefix + "曾任" + "、".join(experience_bits))

    research_bits: list[str] = []
    for item in resume.research[:2]:
        identity = "".join(part.strip() for part in (item.institution, item.topic) if part.strip())
        if identity and identity not in research_bits:
            research_bits.append(identity)
    if research_bits:
        candidates.append("科研经历包括" + "、".join(research_bits))

    achievement = _best_achievement(resume)
    if achievement:
        candidates.append("代表成果：" + achievement)

    skill_names = list(dict.fromkeys(item.name.strip() for item in resume.skills.items if item.name.strip()))[:4]
    if skill_names:
        candidates.append("技能包括" + "、".join(skill_names))

    if resume.education:
        edu = resume.education[0]
        qualification = "、".join(part.strip() for part in (edu.major, edu.degree) if part.strip())
        education_text = "，".join(part for part in (edu.school.strip(), qualification) if part)
        if education_text:
            candidates.append(education_text)

    project_names = list(dict.fromkeys(item.name.strip() for item in resume.projects if item.name.strip()))[:2]
    if project_names:
        candidates.append("项目经历包括" + "、".join(project_names))

    if resume.publications:
        candidates.append(f"论文成果{len(resume.publications)}项")
    if resume.patents:
        candidates.append(f"专利成果{len(resume.patents)}项")
    if resume.certifications:
        candidates.append("持有" + "、".join(resume.certifications[:2]))

    if not experience_bits and not research_bits:
        activity_bits = list(dict.fromkeys(
            " ".join(part.strip() for part in (item.organization, item.role) if part.strip())
            for item in resume.activities
            if item.organization.strip() or item.role.strip()
        ))[:2]
        if activity_bits:
            candidates.append("校园或社会活动包括" + "、".join(activity_bits))

    compact: list[str] = []
    current_length = 0
    seen: set[str] = set()
    for sentence in candidates:
        sentence = sentence.strip("。；; ")
        if not sentence:
            continue
        normalized = re.sub(r"\W+", "", sentence).casefold()
        if not normalized or normalized in seen:
            continue
        added = len(sentence) + 1
        # Keep each selected sentence intact instead of slicing it mid-phrase.
        if current_length + added > 100:
            continue
        compact.append(sentence)
        seen.add(normalized)
        current_length += added
        if len(compact) >= 4:
            break
    return "。".join(compact) + ("。" if compact else "")


def _compact_canonical(resume: CanonicalResume) -> CanonicalResume:
    """Remove blank records/items left by model repair or leakage cleanup."""

    data = resume.model_dump()
    for section in ("awards", "publications", "patents", "certifications", "training", "teaching"):
        data[section] = list(dict.fromkeys(
            str(v).strip() for v in data.get(section, []) if str(v).strip()
        ))
    additional = data.get("additional_sections") or {}
    if isinstance(additional, dict):
        data["additional_sections"] = {
            str(title).strip(): list(dict.fromkeys(
                str(v).strip() for v in values if str(v).strip()
            ))
            for title, values in additional.items()
            if str(title).strip() and isinstance(values, list)
        }
    skills = data.get("skills") or {}
    if isinstance(skills, dict):
        item_indexes: dict[str, int] = {}
        items: list[dict[str, str]] = []
        for item in skills.get("items", []):
            if not isinstance(item, dict):
                continue
            name = _normalize_skill_name(str(item.get("name", "")))
            category = _normalize_skill_category(name, str(item.get("category", "")))
            key = name.casefold()
            if not name:
                continue
            if key not in item_indexes:
                item_indexes[key] = len(items)
                items.append({"name": name, "category": category})
            elif items[item_indexes[key]]["category"] == "other" and category != "other":
                items[item_indexes[key]]["category"] = category
        skills["items"] = items
    for section, fixed_fields in {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }.items():
        cleaned: list[dict] = []
        record_indexes: dict[tuple[str, ...], int] = {}
        for item in data.get(section, []) or []:
            if not isinstance(item, dict):
                continue
            if section != "education":
                item["bullets"] = list(dict.fromkeys(
                    str(v).strip() for v in item.get("bullets", []) if str(v).strip()
                ))
            bullets = item.get("bullets", []) if section != "education" else []
            if any(str(item.get(field, "")).strip() for field in fixed_fields) or bullets:
                identity = tuple(
                    re.sub(r"\s+", "", str(item.get(field, "")).strip()).casefold()
                    for field in fixed_fields
                )
                if any(identity) and identity in record_indexes:
                    existing = cleaned[record_indexes[identity]]
                    if section != "education":
                        existing["bullets"] = list(dict.fromkeys(
                            list(existing.get("bullets", [])) + list(item.get("bullets", []))
                        ))
                    continue
                if any(identity):
                    record_indexes[identity] = len(cleaned)
                cleaned.append(item)
        data[section] = cleaned
    data["summary"] = str(data.get("summary", "") or "").strip()
    compacted = CanonicalResume.model_validate(data)
    compacted.summary = _build_evidence_summary(compacted)
    return compacted


_FALLBACK_PERIOD = re.compile(
    r"(?:(?:19|20)\d{2}(?:(?:[./-]\d{1,2})|(?:年\d{1,2}月?))?|(?:0?[1-9]|1[0-2])[-/](?:19|20)\d{2})"
    r"\s*(?:[-—~至到]\s*(?:(?:19|20)\d{2}(?:(?:[./-]\d{1,2})|(?:年\d{1,2}月?))?|"
    r"(?:0?[1-9]|1[0-2])[-/](?:19|20)\d{2}|至今|现在))?"
)
_FALLBACK_ORGANIZATION = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9·.&（）()_-]{0,40}(?:大学|学院|学校|医院|公司|集团|"
    r"中学|小学|幼儿园|研究院|实验室|中心|部门|协会|学会|学生会|社团|委员会|事务所|银行|"
    r"基金会|工作室|团队)"
)
_FALLBACK_ROLE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9/+.#_-]{0,24}(?:工程师|设计师|教师|老师|医生|医师|"
    r"护士|经理|主管|总监|主任|顾问|研究员|专员|助理|负责人|组长|队长|主席|"
    r"部长|实习生|实习|见习|分析师|架构师|运营|产品|开发|测试|销售|讲师)"
)
_FALLBACK_DEGREE = re.compile(r"(?:博士研究生|硕士研究生|本科|硕士|博士|大专|专科|高中)(?:在读|毕业)?")


def _first_match(pattern: re.Pattern[str], value: str) -> str:
    match = pattern.search(value)
    return match.group(0).strip() if match else ""


def _labeled_value(value: str, labels: tuple[str, ...]) -> str:
    joined = "|".join(re.escape(label) for label in labels)
    match = re.search(rf"(?:{joined})\s*[:：]\s*([^|｜，,；;\n]+)", value)
    return match.group(1).strip() if match else ""


def _organization_from_text(value: str) -> str:
    cleaned = re.sub(
        r"^(?:负责|参与|协助|支持|组织|推动|运营|管理|加入|担任)\s*",
        "",
        value.strip(),
    )
    # OCR often joins a leading period directly to the organization.  Remove
    # only the recognized date span before matching so the date cannot become
    # part of the company/school name.
    cleaned = _FALLBACK_PERIOD.sub(" ", cleaned)
    cleaned = re.sub(r"^[\s\-—~至到]+", "", cleaned)
    contextual = re.search(
        r"(?:就职于|任职于|供职于|在)\s*"
        r"([\u4e00-\u9fffA-Za-z0-9·.&（）()_-]{1,40}?(?:大学|学院|学校|医院|公司|集团|"
        r"研究院|实验室|中心|部门|协会|学会|学生会|社团|委员会|事务所|银行|"
        r"基金会|工作室|团队))(?=工作|任职|担任|就职|[，,。；;\s])",
        cleaned,
    )
    if contextual:
        return contextual.group(1).strip()
    return _first_match(_FALLBACK_ORGANIZATION, cleaned)


def _role_from_text(value: str, organization: str = "") -> str:
    labeled = _labeled_value(value, ("岗位", "职位", "角色", "职务"))
    if labeled:
        return labeled
    # Identity normally appears before the first duty clause. Strip the period
    # and already-grounded organization so a greedy role regex cannot absorb
    # them from OCR-compressed headers.
    header = re.split(r"[，,。；;]", value, maxsplit=1)[0]
    period = _first_match(_FALLBACK_PERIOD, header)
    for identity in (period, organization):
        if identity:
            header = header.replace(identity, " ")
    header = re.sub(r"^[\s|｜:：\-—~至到]+|[\s|｜:：\-—~至到]+$", "", header)
    return _first_match(_FALLBACK_ROLE, header)


def _fallback_target_role(query_text: str, jd_text: str) -> str:
    extracted = (
        product_logic.extract_target_role(query_text, jd_text)
        if hasattr(product_logic, "extract_target_role") else ""
    )
    invalid = (
        not extracted
        or extracted.casefold() in {"jd", "的jd", "岗位", "目标岗位", "简历"}
        or any(token in extracted.casefold() for token in ("简历", "修改", "优化"))
    )
    if not invalid:
        return extracted
    embedded = re.search(
        r"(?:一个|目标(?:岗位)?\s*[:：]?)\s*"
        r"([A-Za-z0-9+.#/_\-\u4e00-\u9fff]{2,32}?)(?:的)?岗位(?:JD|描述)?",
        query_text,
        re.IGNORECASE,
    )
    if embedded:
        return embedded.group(1).strip()
    rewrite = re.search(
        r"(?:优化|调整|修改|改写|适配)(?:成|为)\s*"
        r"([A-Za-z0-9+.#/_\-\u4e00-\u9fff]{2,32}?)(?=岗位|方向|[，,。；;\n]|$)",
        query_text,
        re.IGNORECASE,
    )
    if rewrite:
        return rewrite.group(1).strip()
    matches = re.finditer(
        r"(?:投递|应聘|申请|求职|想投|继续投|目标岗位(?:是|为)?)\s*"
        r"([A-Za-z0-9+.#/_\-\u4e00-\u9fff]{2,32}?)(?=岗位|[，,。；;\n]|帮我|$)",
        query_text,
        re.IGNORECASE,
    )
    for match in matches:
        candidate = match.group(1).strip()
        if candidate.casefold() not in {"jd", "的jd", "岗位", "目标岗位", "简历"}:
            return candidate
    return ""


def _record_groups(source, section: str) -> list[list[str]]:
    groups: list[list[str]] = []
    group_ids: list[str] = []
    for block in candidate_blocks(source):
        if block.section_hint != section or _is_section_heading(block.text):
            continue
        group_id = block.record_id or f"{block.block_id}:single"
        if not group_ids or group_ids[-1] != group_id:
            group_ids.append(group_id)
            groups.append([])
        groups[-1].append(block.text.strip())
    return groups


def _fallback_record(section: str, lines: list[str]) -> tuple[dict, list[str]]:
    """Parse identities conservatively and keep every unconsumed line."""

    joined = " ｜ ".join(line for line in lines if line)
    period = _labeled_value(joined, ("时间", "任职时间", "起止时间", "项目时间")) or _first_match(
        _FALLBACK_PERIOD, joined,
    )
    organization = _labeled_value(joined, ("公司", "单位", "组织", "机构", "学校"))
    if not organization:
        organization = _organization_from_text(joined)
    role = _labeled_value(joined, ("岗位", "职位", "角色", "职务"))
    if not role:
        for line in lines:
            role = _role_from_text(line, organization)
            if role:
                break

    consumed = {value for value in (period, organization, role) if value}
    bullets: list[str] = []
    for line in lines:
        residual = line
        for value in sorted(consumed, key=len, reverse=True):
            residual = residual.replace(value, " ")
        residual = re.sub(
            r"(?:公司|单位|组织|机构|学校|岗位|职位|角色|职务|时间|任职时间|起止时间|项目时间)\s*[:：]",
            " ",
            residual,
        )
        residual = re.sub(r"^[\s|｜,，;；:：\-—~至到]+|[\s|｜,，;；:：\-—~至到]+$", "", residual)
        if len(re.sub(r"\W+", "", residual)) >= 2:
            bullets.append(residual)
    bullets = list(dict.fromkeys(bullets))

    if section == "experience":
        return {
            "organization": organization,
            "role": role,
            "period": period,
            "bullets": bullets,
        }, []
    if section == "research":
        topic = _labeled_value(joined, ("课题", "研究方向", "研究主题"))
        if not topic:
            topic = next((item for item in bullets if not _looks_like_record_body(item)), "")
            if topic in bullets:
                bullets.remove(topic)
        return {
            "institution": organization,
            "topic": topic or role,
            "period": period,
            "bullets": bullets,
        }, []
    if section == "activities":
        return {
            "organization": organization,
            "role": role,
            "period": period,
            "bullets": bullets,
        }, []

    name = _labeled_value(joined, ("项目名称", "项目"))
    if not name:
        header_candidates = [
            line for line in lines
            if not _looks_like_record_body(line) and line not in consumed
        ]
        if header_candidates:
            candidate = header_candidates[0]
            candidate = candidate.replace(period, " ") if period else candidate
            candidate = candidate.replace(organization, " ") if organization else candidate
            candidate = candidate.replace(role, " ") if role else candidate
            name = re.sub(r"^[\s|｜,，;；:：\-]+|[\s|｜,，;；:：\-]+$", "", candidate)
    if name in bullets:
        bullets.remove(name)
    return {
        "name": name,
        "organization": organization,
        "role": role,
        "period": period,
        "bullets": bullets,
    }, []


def _fallback_education(lines: list[str]) -> tuple[dict, list[str]]:
    joined = " ｜ ".join(lines)
    school = _labeled_value(joined, ("学校", "院校")) or _first_match(_FALLBACK_ORGANIZATION, joined)
    degree = _labeled_value(joined, ("学历", "学位")) or _first_match(_FALLBACK_DEGREE, joined)
    major = _labeled_value(joined, ("专业",))
    period = _labeled_value(joined, ("时间", "就读时间", "起止时间")) or _first_match(_FALLBACK_PERIOD, joined)
    parts = [
        part.strip()
        for part in re.split(r"[|｜\t，,;；]+|\s{2,}", joined)
        if part.strip()
    ]
    if len(parts) >= 2 and not _FALLBACK_PERIOD.fullmatch(parts[0]) and not _FALLBACK_DEGREE.fullmatch(parts[0]):
        # Delimited education headers conventionally put the institution first;
        # this also preserves anonymized names such as “学校0”.
        school = parts[0]
    if not school:
        school = next((
            value for value in parts
            if not _FALLBACK_DEGREE.fullmatch(value)
            and not _FALLBACK_PERIOD.fullmatch(value)
        ), "")
    if not major:
        for value in parts:
            if not value or any(item and item in value for item in (school, degree, period)):
                continue
            value = re.sub(r"(?:专业|学历|学位|时间|就读时间|起止时间)\s*[:：]", "", value).strip()
            if value:
                major = value
                break
    if not major:
        residual = joined
        for value in (school, degree, period):
            if value:
                residual = residual.replace(value, " ")
        residual = re.sub(
            r"(?:学校|院校|学历|学位|专业|时间|就读时间|起止时间)\s*[:：]",
            " ",
            residual,
        )
        residual = re.sub(r"[\s|｜,，;；:：\-—~至到]+", "", residual)
        if 2 <= len(residual) <= 40:
            major = residual
    leftovers = []
    for line in lines:
        residual = line
        for value in (school, degree, major, period):
            if value:
                residual = residual.replace(value, " ")
        residual = re.sub(r"[\s|｜,，;；:：\-—~至到]+", "", residual)
        if len(residual) >= 2:
            leftovers.append(line)
    return {"school": school, "degree": degree, "major": major, "period": period}, leftovers


def _deterministic_fallback(cv_text: str, query_text: str, jd_text: str) -> CanonicalResume:
    """Lossless, section-aware fallback for any profession and resume length."""

    source = build_source_bundle(cv_text, query_text, jd_text)
    factual_text = "\n".join(block.text for block in candidate_blocks(source))
    industry = product_logic.infer_industry(query_text, factual_text, jd_text)
    target_role = _fallback_target_role(query_text, jd_text)
    raw = product_logic.heuristic_resume_from_text(factual_text, industry, target_role)
    meta = dict(raw.get("meta") or {})

    phone_match = re.search(r"1[3-9]\d(?:[\s-]?\d){8}", factual_text)
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", factual_text)
    name_match = re.search(r"(?:姓名|name)\s*[:：]\s*([\u4e00-\u9fff·]{2,12})", factual_text, re.IGNORECASE)
    if not name_match:
        name_match = re.search(
            r"(?m)^\s*([\u4e00-\u9fff·]{2,12})(?:个人)?简历\s*$",
            factual_text,
        )
    seniority_match = re.search(
        r"\d+(?:\.\d+)?\s*(?:年|个月|月)(?:工作|从业|实习)?(?:经验|经历)",
        factual_text,
    )

    additional: dict[str, list[str]] = {}
    education: list[dict] = []
    for lines in _record_groups(source, "education"):
        record, leftovers = _fallback_education(lines)
        if any(record.values()):
            education.append(record)
        if leftovers:
            additional.setdefault("教育经历补充", []).extend(leftovers)

    records: dict[str, list[dict]] = {name: [] for name in ("experience", "research", "activities", "projects")}
    for section in records:
        for lines in _record_groups(source, section):
            record, leftovers = _fallback_record(section, lines)
            if any(value for key, value in record.items() if key != "bullets") or record.get("bullets"):
                records[section].append(record)
            if leftovers:
                additional.setdefault("待整理的原始经历", []).extend(leftovers)

    scalar_sections = {
        "awards": "awards",
        "publications": "publications",
        "patents": "patents",
        "certifications": "certifications",
        "training": "training",
        "teaching": "teaching",
    }
    scalars: dict[str, list[str]] = {target: [] for target in scalar_sections.values()}
    summary_lines: list[str] = []
    skill_items: list[dict[str, str]] = []
    unclassified: list[str] = []
    unstructured_current: tuple[str, int] | None = None
    for block in candidate_blocks(source):
        if _is_section_heading(block.text):
            continue
        section = block.section_hint or ""
        value = block.text.strip()
        if section == "summary":
            summary_lines.append(value)
        elif section == "skills":
            value = re.sub(r"^(?:专业技能|技能清单|技术栈|工具|语言能力)\s*[:：]\s*", "", value)
            for item in re.split(r"[、，,；;|｜/]+", value):
                item = item.strip(" \t-•")
                if len(item) >= 2:
                    skill_items.append({"name": item, "category": "other"})
        elif section in scalar_sections:
            cleaned = re.sub(r"^[^:：]{1,12}[:：]\s*", "", value).strip()
            if cleaned:
                scalars[scalar_sections[section]].append(cleaned)
        elif section not in {"education", "experience", "research", "activities", "projects"}:
            skill_match = re.match(r"^(?:技能|专业技能|工具)\s*[:：]\s*(.+)$", value)
            language_match = re.match(r"^(?:语言|语言能力)\s*[:：]\s*(.+)$", value)
            if skill_match or language_match:
                category = "natural_language" if language_match else "other"
                raw_values = (skill_match or language_match).group(1)
                for item in re.split(r"[、，,；;|｜/]+", raw_values):
                    item = item.strip(" \t-•")
                    if len(item) >= 2:
                        skill_items.append({"name": item, "category": category})
                unstructured_current = None
                continue
            if re.search(r"(?:奖学金|一等奖|二等奖|三等奖|优秀学生干部|荣誉称号|获奖)$", value):
                scalars["awards"].append(value)
                unstructured_current = None
                continue
            if re.search(r"(?:证书|资格证|执业证|职业资格|认证|执照)$", value):
                scalars["certifications"].append(value)
                unstructured_current = None
                continue
            profile_domain = re.search(
                r"我(?:是)?(?:做|从事|负责)\s*([^，,。；;]{2,30}?)(?:的|方向|工作|$)",
                value,
            )
            if profile_domain:
                domain = profile_domain.group(1).strip()
                if domain:
                    skill_items.append({"name": domain, "category": "domain"})
                    summary_lines.append(value)
                    continue

            organization = _organization_from_text(value)
            period = _first_match(_FALLBACK_PERIOD, value)
            degree = _first_match(_FALLBACK_DEGREE, value)
            compact_role = _role_from_text(value, organization)
            activity_hint = bool(re.search(
                r"(?:学生会|志愿者?(?:协会|团队)?|社团|校园组织|校团委|协会活动|社区义工)",
                value,
            ))
            explicit_work = bool(
                organization
                and re.search(r"(?:就职于|任职于|供职于|在.+?(?:工作|担任|任职))", value)
            )
            project_match = re.search(
                r"(?:参与|负责|主导|完成)\s*([^，。；;]{2,36}?(?:APP|系统|平台|项目|课题|产品))"
                r"(?:的|设计|开发|建设|研究|$)",
                value,
                re.IGNORECASE,
            )

            if organization and degree:
                record, leftovers = _fallback_education([value])
                education.append(record)
                if leftovers:
                    additional.setdefault("教育经历补充", []).extend(leftovers)
                unstructured_current = None
                continue
            if explicit_work or (organization and period and compact_role):
                record, _leftovers = _fallback_record("experience", [value])
                records["experience"].append(record)
                unstructured_current = ("experience", len(records["experience"]) - 1)
                continue
            if activity_hint:
                target_section = "activities"
                if unstructured_current and unstructured_current[0] == target_section:
                    current_record = records[target_section][unstructured_current[1]]
                    current_org = str(current_record.get("organization", ""))
                    same_activity = bool(
                        current_org and (
                            current_org in value
                            or ("志愿" in current_org and "志愿" in value)
                            or ("学生会" in current_org and "学生会" in value)
                        )
                    )
                    if same_activity:
                        current_record["bullets"].append(value)
                        continue
                if (
                    unstructured_current
                    and unstructured_current[0] == target_section
                    and organization
                    and records[target_section][unstructured_current[1]].get("organization") == organization
                ):
                    records[target_section][unstructured_current[1]]["bullets"].append(value)
                else:
                    records[target_section].append({
                        "organization": organization,
                        "role": _labeled_value(value, ("岗位", "职位", "角色", "职务")),
                        "period": _first_match(_FALLBACK_PERIOD, value),
                        "bullets": [value],
                    })
                    unstructured_current = (target_section, len(records[target_section]) - 1)
                continue
            if project_match:
                name = project_match.group(1).strip()
                if (
                    unstructured_current
                    and unstructured_current[0] == "projects"
                    and re.search(r"^(?:撰写|完成|优化|设计|开发|建设|维护|测试)", name)
                ):
                    records["projects"][unstructured_current[1]]["bullets"].append(value)
                    continue
                records["projects"].append({
                    "name": name,
                    "organization": "",
                    "role": "",
                    "period": _first_match(_FALLBACK_PERIOD, value),
                    "bullets": [value],
                })
                unstructured_current = ("projects", len(records["projects"]) - 1)
                continue
            if unstructured_current is not None and _looks_like_record_body(value):
                current_section, current_index = unstructured_current
                records[current_section][current_index]["bullets"].append(value)
                continue
            if unstructured_current is not None:
                # Contact rows, summaries and a following record header must
                # not leak into the preceding experience merely because OCR
                # removed the section headings.
                unstructured_current = None
            if (
                (phone_match and phone_match.group(0) in value)
                or (email_match and email_match.group(0) in value)
            ):
                # The values have already been extracted into meta. Keeping a
                # compact contact row as the personal summary duplicates PII
                # and crowds out grounded professional content.
                continue
            if not summary_lines and len(value) >= 20:
                summary_lines.append(value)
                continue
            if not any(token and token in value for token in (
                name_match.group(1) if name_match else "",
                phone_match.group(0) if phone_match else "",
                email_match.group(0) if email_match else "",
            )):
                unclassified.append(value)
    if unclassified:
        additional["待整理的原始信息"] = list(dict.fromkeys(unclassified))

    return CanonicalResume.model_validate({
        "meta": {
            "name": (name_match.group(1) if name_match else meta.get("name", "")),
            "phone": (phone_match.group(0) if phone_match else meta.get("phone", "")),
            "email": (email_match.group(0) if email_match else meta.get("email", "")),
            "target_role": target_role or meta.get("target_role", ""),
            "work_experience": seniority_match.group(0) if seniority_match else "",
        },
        "summary": "。".join(item.strip("。") for item in summary_lines if item.strip("。")),
        "education": education,
        "experience": records["experience"],
        "research": records["research"],
        "activities": records["activities"],
        "projects": records["projects"],
        "skills": {"items": list({item["name"]: item for item in skill_items}.values())},
        **{key: list(dict.fromkeys(values)) for key, values in scalars.items()},
        "additional_sections": {
            title: list(dict.fromkeys(value for value in values if value.strip()))
            for title, values in additional.items() if values
        },
    })


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
    research = data.get("research", [])
    for item in research:
        if isinstance(item, dict):
            item["company"] = item.pop("institution", "")
            item["role"] = item.pop("topic", "")
    data["campus_experience"] = data.pop("activities", [])
    for item in data["campus_experience"]:
        if isinstance(item, dict) and "organization" in item:
            item["company"] = item.pop("organization")
    # Reuse the renderer's research-output section while retaining the patent
    # type explicitly in display text.
    if data.get("patents"):
        data["publications"] = list(data.get("publications") or []) + [
            value if str(value).startswith("专利") else f"专利：{value}"
            for value in data.get("patents", []) if str(value).strip()
        ]
    # Convert flat skills.items to V1 categorized format
    skills = data.get("skills", {})
    if isinstance(skills, dict):
        items = skills.pop("items", []) if isinstance(skills.get("items"), list) else []
        categorized: dict[str, list[str]] = {
            "languages": [], "frameworks": [], "tools": [], "domains": [],
            "methodologies": [], "certifications": [],
            "natural_languages": [], "others": [],
        }
        if items:
            for item in items:
                if isinstance(item, dict):
                    name = str(item.get("name", "")).strip()
                    if not name:
                        continue
                    cat = item.get("category", "other")
                    normalized_cat = {
                        "language": "languages",
                        "framework": "frameworks",
                        "tool": "tools",
                        "domain": "domains",
                        "methodology": "methodologies",
                        "certification": "certifications",
                        "natural_language": "natural_languages",
                        "other": "others",
                    }.get(cat, cat)
                    if normalized_cat in categorized:
                        categorized[normalized_cat].append(name)
                    else:
                        categorized.setdefault("others", []).append(name)
            skills.update(categorized)
    return data


def _empty_profile_framework(target_role: str = "") -> dict:
    """Rendering-only skeleton for JD-only requests with no candidate facts."""

    return {
        "mode": "empty_profile",
        "notice": "以下内容均为待填写结构，不代表候选人已有事实。",
        "target_role": str(target_role or "").strip(),
        "sections": [
            {"key": "basic_info", "title": "基本信息", "fields": ["姓名", "联系电话", "邮箱", "所在城市"]},
            {"key": "summary", "title": "个人总结", "fields": ["职业背景", "核心优势", "求职方向"]},
            {"key": "education", "title": "教育经历", "fields": ["学校", "学历", "专业", "起止时间"]},
            {"key": "experience", "title": "工作/实习经历", "fields": ["公司", "岗位", "起止时间", "职责与成果"]},
            {"key": "projects", "title": "项目经历", "fields": ["项目名称", "项目角色", "项目时间", "行动与成果"]},
            {"key": "skills", "title": "专业技能", "fields": ["工具/技术", "专业领域", "方法与流程", "证书/语言"]},
        ],
    }


_SOURCE_FALLBACK_FASTPATH_MIN_COVERAGE = 0.90


def _has_structured_history(resume: CanonicalResume) -> bool:
    """Require real canonical records, not merely raw text parked in extras."""

    return any((
        resume.education,
        resume.experience,
        resume.research,
        resume.activities,
        resume.projects,
    ))


def _grounded_source_fallback(
    cv_text: str,
    query_text: str,
    jd_text: str,
    source,
    candidate_evidence: str,
) -> tuple[VerifiedResult, float, list[str]]:
    """Build the same evidence-gated fallback used by final coverage repair."""

    fallback = _deterministic_fallback(cv_text, query_text, jd_text)
    fallback = _ground_bullets(fallback, candidate_evidence)
    fallback, fallback_bindings, fallback_removed = enforce_resume_evidence(
        fallback,
        source,
    )
    fallback = _compact_canonical(fallback)
    fallback_bindings = bind_resume_evidence(fallback, source)
    fallback_coverage, fallback_missing = _deterministic_source_coverage(
        source,
        fallback_bindings,
    )
    return VerifiedResult(
        resume=fallback,
        changes=[
            Change(path=path, action="remove", reason="No candidate evidence binding")
            for path in fallback_removed
        ],
        evidence_bindings=fallback_bindings,
    ), fallback_coverage, fallback_missing


def run_v2_pipeline(
    cv_text: str,
    query_text: str,
    jd_text: str,
) -> VerifiedResult:
    """Run the V2 5-layer pipeline. Returns VerifiedResult or fallback."""
    t_start = time.perf_counter()

    # ── No CV: generate structured framework from query + JD ──
    if not cv_text or not cv_text.strip():
        logger.info("V2 | No CV — generating framework from query+JD")
        t_gen = time.perf_counter()
        resume = compose_from_query(query_text, jd_text)
        used_fallback = False
        if _is_empty_resume(resume) and query_text.strip():
            logger.warning("Generate composer produced an empty resume; using deterministic query fallback")
            resume = _deterministic_fallback("", query_text, jd_text)
            used_fallback = not _is_empty_resume(resume)
        n_exp = len(resume.experience)
        n_proj = len(resume.projects)
        n_bullets = sum(len(e.bullets) for e in resume.experience) + \
                    sum(len(p.bullets) for p in resume.projects) + \
                    sum(len(r.bullets) for r in resume.research)
        logger.info("V2 | Generate done: %d exp, %d proj, %d bullets (%.1fs)",
                    n_exp, n_proj, n_bullets, time.perf_counter() - t_gen)

        grounded_data = resume.model_dump()
        _ground_fixed_fields(grounded_data, query_text)
        _reclassify_non_work(grounded_data, query_text)
        resume = CanonicalResume.model_validate(grounded_data)
        resume = _ground_bullets(resume, query_text)

        # With no structured candidate facts, a polished-sounding summary is
        # misleading.  Keep the target role and ask for missing information in
        # reply_text instead of manufacturing a resume profile.
        has_profile_records = _has_candidate_profile(resume)
        if not has_profile_records:
            resume.summary = ""

        # Generation also receives a dedicated bullet edit pass whenever the
        # user supplied enough factual material to produce bullets.
        # Rank first so optimizer change paths still point at the final bullet
        # positions exposed by the API.
        resume = _rank_resume_content(resume, jd_text or resume.meta.target_role)
        optimizer_changes: list[Change] = []
        if _needs_optimizer(resume):
            before_optimizer = resume.model_copy(deep=True)
            resume = optimize_resume(resume, jd_text)
            resume = _ground_optimizer_output(before_optimizer, resume, query_text)
            optimizer_changes = _bullet_rewrite_changes(before_optimizer, resume)
        else:
            logger.info("V2 | Optimizer skipped: no factual bullets")
        resume = _ground_bullets(resume, query_text)
        resume = validate_resume(resume, source_text=query_text)

        evidence_source = build_source_bundle("", query_text, jd_text)
        resume, evidence_bindings, evidence_removed = enforce_resume_evidence(resume, evidence_source)
        resume = _compact_canonical(resume)
        # Recompute only after unsupported JD-derived records have been
        # removed. Otherwise temporary model output suppresses the framework
        # and produces an almost blank document.
        has_profile_records = _has_candidate_profile(resume)
        if not has_profile_records:
            resume.summary = ""
        evidence_bindings = bind_resume_evidence(resume, evidence_source)
        logger.info("V2 | Evidence bindings: %d", len(evidence_bindings))
        resume_dict = _canonical_to_v1_format(resume)
        if not has_profile_records:
            resume_dict["framework"] = _empty_profile_framework(resume.meta.target_role)
        return VerifiedResult(
            resume=resume,
            changes=([Change(
                path="*",
                action="replace",
                reason="LLM unavailable or invalid; generated from explicit user facts with deterministic parser",
            )] if used_fallback else []) + optimizer_changes + [
                Change(path=path, action="remove", reason="No candidate evidence binding")
                for path in evidence_removed
            ],
            resume_dict=resume_dict,
            evidence_bindings=evidence_bindings,
        )

    # ── Has CV: full Composer → Verifier → Optimizer pipeline ──
    source = build_source_bundle(cv_text, query_text, jd_text)
    logger.info("V2 | SourceBundle: %d blocks (%.1fs)",
                len(source.blocks), time.perf_counter() - t_start)

    t_composer = time.perf_counter()
    draft = compose_resume(source)
    logger.info("V2 | Composer done: %d edu, %d exp, %d res, %d proj (%.1fs)",
                len(draft.education), len(draft.experience),
                len(draft.research), len(draft.projects),
                time.perf_counter() - t_composer)

    t_verifier = time.perf_counter()
    candidate_evidence = "\n".join(block.text for block in candidate_blocks(source))
    result = _deterministic_verify_draft(source, draft)
    if result is None:
        fallback_candidate = None
        try:
            fallback_candidate = _grounded_source_fallback(
                cv_text,
                query_text,
                jd_text,
                source,
                candidate_evidence,
            )
        except Exception as exc:
            logger.warning("V2 | Pre-verifier source fallback failed: %s", exc)
        if (
            fallback_candidate is not None
            and fallback_candidate[1] >= _SOURCE_FALLBACK_FASTPATH_MIN_COVERAGE
            and _has_structured_history(fallback_candidate[0].resume)
        ):
            result, fallback_coverage, _fallback_missing = fallback_candidate
            result.changes.insert(0, Change(
                path="*",
                action="replace",
                reason="Used high-coverage deterministic parser before LLM verification",
            ))
            logger.info(
                "V2 | LLM Verifier skipped: evidence-gated fallback coverage %.1f%%",
                fallback_coverage * 100,
            )
        else:
            logger.info("V2 | Falling back to LLM Verifier")
            result = verify_resume(source, draft)
    else:
        logger.info("V2 | LLM Verifier skipped")
    result.resume = _ground_bullets(result.resume, candidate_evidence)
    if _is_empty_resume(result.resume):
        logger.warning("V2 verifier produced an empty resume; using deterministic source fallback")
        fallback = _deterministic_fallback(cv_text, query_text, jd_text)
        if not _is_empty_resume(fallback):
            result = VerifiedResult(
                resume=fallback,
                changes=[Change(
                    path="*",
                    action="replace",
                    reason="LLM unavailable or invalid; preserved source facts with deterministic parser",
                )],
            )
    logger.info("V2 | Verifier done: %d edu, %d exp, %d res, %d changes (%.1fs)",
                len(result.resume.education), len(result.resume.experience),
                len(result.resume.research), len(result.changes),
                time.perf_counter() - t_verifier)

    # Rank first so accepted rewrite paths remain stable in the final output.
    result.resume = _rank_resume_content(result.resume, jd_text or result.resume.meta.target_role)
    t_optimizer = time.perf_counter()
    if _needs_optimizer(result.resume):
        before_optimizer = result.resume.model_copy(deep=True)
        result.resume = optimize_resume(result.resume, jd_text)
        result.resume = _ground_optimizer_output(
            before_optimizer,
            result.resume,
            candidate_evidence,
        )
        result.changes.extend(_bullet_rewrite_changes(before_optimizer, result.resume))
    else:
        logger.info("V2 | Optimizer skipped: no factual bullets")
    result.resume = _ground_bullets(result.resume, candidate_evidence)
    logger.info("V2 | Optimizer done (%.1fs)", time.perf_counter() - t_optimizer)

    # Deterministic repair uses candidate evidence only.  JD schools, employers
    # and dates must never backfill candidate fields.
    _source_for_validate = candidate_evidence
    result.resume = validate_resume(result.resume, source_text=_source_for_validate)
    result.resume, _, evidence_removed = enforce_resume_evidence(result.resume, source)
    result.changes.extend(
        Change(path=path, action="remove", reason="No candidate evidence binding")
        for path in evidence_removed
    )
    result.resume = _compact_canonical(result.resume)
    result.evidence_bindings = bind_resume_evidence(result.resume, source)
    raw_coverage, raw_missing = measure_source_coverage(source, result.evidence_bindings)
    coverage, missing_blocks = _deterministic_source_coverage(
        source,
        result.evidence_bindings,
    )
    if raw_missing and coverage > raw_coverage:
        logger.info(
            "V2 | Final coverage resolved structural artifacts: %.1f%% -> %.1f%%",
            raw_coverage * 100,
            coverage * 100,
        )
    if missing_blocks and coverage < 0.80:
        # Prefer a less polished deterministic parse only when it demonstrably
        # preserves substantially more of the source. This prevents a valid
        # long resume from collapsing to a few attractive bullets.
        try:
            fallback_result, fallback_coverage, fallback_missing = _grounded_source_fallback(
                cv_text,
                query_text,
                jd_text,
                source,
                candidate_evidence,
            )
            if fallback_coverage >= coverage + 0.10:
                logger.warning(
                    "V2 | Replaced low-coverage result with source-preserving fallback: %.1f%% -> %.1f%%",
                    coverage * 100,
                    fallback_coverage * 100,
                )
                result.resume = fallback_result.resume
                result.evidence_bindings = fallback_result.evidence_bindings
                result.changes = [Change(
                    path="*",
                    action="replace",
                    reason="Source coverage repair used the more complete deterministic parse",
                )] + fallback_result.changes
                coverage = fallback_coverage
                missing_blocks = fallback_missing
        except Exception as exc:
            logger.warning("V2 | Source coverage fallback failed: %s", exc)
    if missing_blocks:
        logger.info("V2 | Final source coverage %.1f%%, missing=%s", coverage * 100, missing_blocks[:8])
    result.resume_dict = _canonical_to_v1_format(result.resume)
    logger.info("V2 | Evidence bindings: %d", len(result.evidence_bindings))

    logger.info("V2 | Total: %.1fs (Composer+Verifier+Validate+Format)",
                time.perf_counter() - t_start)

    return result
