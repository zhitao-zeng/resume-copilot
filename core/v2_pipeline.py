"""V2 Pipeline orchestration.

Layers: SourceAdapter → Composer → Verifier → Optimizer → Validator
"""
from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date

from v2_schemas import VerifiedResult, CanonicalResume, DraftResume, Meta, Change
from source_adapter import (
    _LAYOUT_RESET_HEADINGS,
    _is_section_heading,
    _looks_like_record_body,
    build_source_bundle,
    candidate_blocks,
)
from resume_composer import compose_resume, compose_from_query
from resume_verifier import verify_resume
from resume_verifier import _ground_fixed_fields, _reclassify_non_work
from resume_optimizer import (
    optimize_resume_with_provenance,
    select_narrative_record_keys,
    _introduces_unsupported_fact,
    _safe_rewrite_diagnostics,
)
from v2_validator import validate_resume
from evidence_binding import (
    bind_resume_evidence,
    enforce_resume_evidence,
    measure_source_coverage,
    source_fact_units,
)
from atomic_fact_audit import audit_atomic_facts
from fact_compiler import compile_fact_coverage, sanitize_resume_placeholders
import resume_product_logic as product_logic
from diagnostic_trace import trace_event
from pipeline_profiles import resolve_pipeline_profile

logger = logging.getLogger(__name__)


def _fact_compiler_mode() -> str:
    return resolve_pipeline_profile().fact_compiler_mode


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
_SUMMARY_MAX_CHARS = 360
_SUMMARY_MAX_SENTENCES = 6
_INTERNAL_ADDITIONAL_SECTION = re.compile(
    r"^(?:待整理(?:的)?原始(?:信息|经历)|教育经历补充|补充信息)$"
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
_SCALED_DELIVERABLE = re.compile(
    r"(?=.*\d+(?:\.\d+)?\s*(?:%|万|人|次|项|个|条|篇|例|台|套|元|万元))"
    r"(?=.*(?:输出|形成|产出|分析|处理|覆盖|支持|发布|交付|完成|上线))",
    re.IGNORECASE,
)
_SUMMARY_SUBJECTIVE = re.compile(
    r"(?:扎实|敏锐|优秀|热爱|致力于|较强|出色|卓越|丰富的|良好(?:的)?(?:能力|素养)|"
    r"责任心|学习能力|抗压能力|团队精神|积极主动|自驱力|"
    r"清晰的问题拆解|执行闭环|以真实岗位职责和结果为准)"
)
_INCOMPLETE_TEXT_TAIL = re.compile(
    r"(?:累计(?:粉丝|用户|曝光|销量|销售额|阅读量)|达|达到|提升|降低|增长|"
    r"减少|累计|实现|完成|负责|具备|熟悉|掌握|按计划|包括|例如|如)$"
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
    return [
        item for item in values
        if len(item) >= 6
        or re.search(
            r"\d+(?:\.\d+)?(?:%|万|亿|w|k|人|次|个|条|元|年|月|日)|"
            r"[A-Za-z][A-Za-z0-9+.#/_-]{1,}",
            item,
            re.IGNORECASE,
        )
    ]


def _fact_clauses(text: str) -> list[str]:
    """Split a compound statement only for evidence comparison.

    The returned clauses are reassembled before rendering.  This is therefore
    deliberately finer-grained than ``_split_grounded_fact_bullet``: commas
    often separate action, method and result evidence that may originate from
    different source lines, but they should still render as one coherent
    bullet.
    """

    return [
        item.strip(" \t-•·▪◦，,。；;")
        for item in re.split(r"[\n。；;,，•·▪◦]+", str(text or ""))
        if len(re.sub(r"\W+", "", item, flags=re.UNICODE)) >= 3
    ]


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


def _compound_fact_is_supported(value: str, sentences: list[str]) -> bool:
    """Recognize a coherent bullet assembled from several source lines.

    OCR often splits one original bullet between lines. Comparing a complete
    rewrite with only the nearest line makes real tools and metrics look new.
    This aggregate check is deliberately strict about every numeric/Latin/
    named fact and still requires each clause to resemble candidate evidence.
    """

    clauses = _fact_clauses(value)
    source_clauses = list(dict.fromkeys(
        clause
        for sentence in sentences
        for clause in (_fact_clauses(sentence) or [sentence])
    ))
    if len(clauses) < 2 or len(source_clauses) < 2:
        return False
    if any(
        not (
            generated_coverage >= 0.30
            or (generated_coverage >= 0.18 and source_recall >= 0.45)
            or (generated_coverage >= 0.12 and source_recall >= 0.80)
        )
        for clause in clauses
        for _source, generated_coverage, source_recall in [
            _closest_source_sentence(clause, source_clauses)
        ]
    ):
        return False
    aggregate = unicodedata.normalize("NFKC", "\n".join(sentences))
    candidate = unicodedata.normalize("NFKC", str(value or ""))
    unit_suffix = r"(?:%|万|亿|w|k|人|次|个|条|元|年|月|日)"
    aggregate = re.sub(rf"(?<=\d)\s+(?={unit_suffix})", "", aggregate, flags=re.IGNORECASE)
    candidate = re.sub(rf"(?<=\d)\s+(?={unit_suffix})", "", candidate, flags=re.IGNORECASE)
    _safe, reasons = _safe_rewrite_diagnostics(aggregate, candidate)
    blocking = {
        reason for reason in reasons
        if reason not in {"source_content_shrunk", "ownership_level_changed"}
    }
    return not blocking


def _ground_bullet_value(
    value: str,
    sentences: list[str],
    *,
    path: str = "",
    _allow_clause_split: bool = True,
) -> tuple[str, str]:
    """Ground a bullet as a whole or as independently checked clauses.

    Compound bullets are checked clause by clause.  A supported action and
    method can therefore survive when the model appends one unsupported
    result, instead of reverting or deleting the entire useful statement.
    """

    if _allow_clause_split:
        whole_value, whole_status = _ground_bullet_value(
            value,
            sentences,
            path=path,
            _allow_clause_split=False,
        )
        if whole_status == "accepted":
            return whole_value, whole_status
        if _compound_fact_is_supported(value, sentences):
            trace_event(
                "bullet_compound_grounding",
                path=path,
                candidate=value,
                output=value,
                status="accepted",
                whole_status=whole_status,
            )
            return value, "accepted"
        candidate_clauses = _fact_clauses(value)
        source_clauses = list(dict.fromkeys(
            clause
            for sentence in sentences
            for clause in (_fact_clauses(sentence) or [sentence])
        ))
        if len(candidate_clauses) >= 2 and len(source_clauses) >= 2:
            safe_clauses: list[str] = []
            changed = False
            clause_events: list[dict[str, object]] = []
            for clause in candidate_clauses:
                grounded_clause, status = _ground_bullet_value(
                    clause,
                    source_clauses,
                    path=path,
                    _allow_clause_split=False,
                )
                if grounded_clause and grounded_clause not in safe_clauses:
                    safe_clauses.append(grounded_clause)
                if status != "accepted" or grounded_clause != clause:
                    changed = True
                clause_events.append({
                    "candidate": clause,
                    "output": grounded_clause,
                    "status": status,
                })
            if safe_clauses:
                unchanged = not changed and len(safe_clauses) == len(candidate_clauses)
                output = (
                    str(value or "").strip()
                    if unchanged
                    else "，".join(safe_clauses).rstrip("，,。；; ") + "。"
                )
                status = "accepted" if unchanged else "trimmed"
                trace_event(
                    "bullet_clause_grounding",
                    path=path,
                    candidate=value,
                    output=output,
                    status=status,
                    clauses=clause_events,
                )
                return output, status
            trace_event(
                "bullet_clause_grounding",
                path=path,
                candidate=value,
                output=whole_value,
                status=whole_status,
                clauses=clause_events,
            )
            return whole_value, whole_status
        return whole_value, whole_status

    source, generated_coverage, source_recall = _closest_source_sentence(value, sentences)
    if not source:
        logger.info("Dropped ungrounded bullet")
        trace_event(
            "bullet_grounding",
            path=path,
            candidate=value,
            closest_source="",
            generated_coverage=generated_coverage,
            source_recall=source_recall,
            status="dropped",
            reasons=["no_source_match"],
        )
        return "", "dropped"
    # If the model retained a recognizable source clause but added unsupported
    # material, restore the complete source sentence. Very weak matches are
    # dropped instead of being legitimized by a common verb such as “负责”.
    if generated_coverage < 0.58:
        if generated_coverage >= 0.18 and source_recall >= 0.45:
            logger.info("Restored source wording for weakly grounded bullet")
            trace_event(
                "bullet_grounding",
                path=path,
                candidate=value,
                closest_source=source,
                generated_coverage=generated_coverage,
                source_recall=source_recall,
                output=source,
                status="restored",
                reasons=["weak_generated_coverage_with_source_recall"],
            )
            return source, "restored"
        logger.info("Dropped weakly grounded bullet")
        trace_event(
            "bullet_grounding",
            path=path,
            candidate=value,
            closest_source=source,
            generated_coverage=generated_coverage,
            source_recall=source_recall,
            status="dropped",
            reasons=["weak_generated_coverage"],
        )
        return "", "dropped"
    upgraded = _action_level(value) > _action_level(source)
    unsupported_result = any(term in value and term not in source for term in _RESULT_CLAIMS)
    unsupported_fact = _introduces_unsupported_fact(source, value)
    _hard_safe, hard_reasons = _safe_rewrite_diagnostics(source, value)
    # Grounding a concise source subset is valid, so the optimizer's
    # completeness-oriented shrink check does not apply here. All checks for
    # newly introduced facts and changed ownership still apply.
    hard_reasons = [
        reason for reason in hard_reasons
        if reason != "source_content_shrunk"
    ]
    hard_safe = not hard_reasons
    if upgraded or unsupported_result or unsupported_fact or not hard_safe:
        logger.info("Restored source wording for over-claimed bullet")
        reasons = []
        if upgraded:
            reasons.append("ownership_upgraded")
        if unsupported_result:
            reasons.append("unsupported_result_claim")
        if unsupported_fact:
            reasons.append("unsupported_named_fact")
        reasons.extend(reason for reason in hard_reasons if reason not in reasons)
        trace_event(
            "bullet_grounding",
            path=path,
            candidate=value,
            closest_source=source,
            generated_coverage=generated_coverage,
            source_recall=source_recall,
            output=source,
            status="restored",
            reasons=reasons,
        )
        return source, "restored"
    trace_event(
        "bullet_grounding",
        path=path,
        candidate=value,
        closest_source=source,
        generated_coverage=generated_coverage,
        source_recall=source_recall,
        output=value,
        status="accepted",
        reasons=[],
    )
    return value, "accepted"


def _ground_bullets(
    resume: CanonicalResume,
    evidence_text: str,
    *,
    trusted_rewrites: dict[str, str] | None = None,
) -> CanonicalResume:
    """Fall back to the nearest source sentence when a bullet upgrades facts."""

    grounded = resume.model_copy(deep=True)
    sentences = _source_sentences(evidence_text)
    for section_name in ("experience", "research", "activities", "projects"):
        section = getattr(grounded, section_name)
        for record_index, record in enumerate(section):
            safe_bullets: list[str] = []
            for bullet_index, bullet in enumerate(record.bullets):
                value = str(bullet or "").strip()
                if not value:
                    continue
                path = f"{section_name}[{record_index}].bullets[{bullet_index}]"
                # Provenance is useful for binding an accepted paraphrase back
                # to its source, but it must never bypass the final truth gate.
                # The external evaluator caught rewrites that were traceable to
                # an original bullet while still adding unsupported claims.
                safe_value, _status = _ground_bullet_value(value, sentences, path=path)
                if safe_value and safe_value not in safe_bullets:
                    safe_bullets.append(safe_value)
            record.bullets = safe_bullets
    return grounded


def _ground_optimizer_output(
    original: CanonicalResume,
    optimized: CanonicalResume,
    evidence_text: str,
    *,
    trusted_rewrites: dict[str, str] | None = None,
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
                path = f"{section_name}[{record_index}].bullets[{bullet_index}]"
                # A semantic-review provenance entry is not evidence that every
                # word in the rewrite is grounded. Always validate the emitted
                # wording; keep provenance only after this check succeeds.
                safe_value, status = _ground_bullet_value(value, sentences, path=path)
                if status not in {"accepted", "trimmed"} and bullet_index < len(before_bullets):
                    safe_value = str(before_bullets[bullet_index] or "").strip()
                    logger.info(
                        "Reverted unsupported optimizer patch at %s[%d].bullets[%d]",
                        section_name,
                        record_index,
                        bullet_index,
                    )
                    trace_event(
                        "optimizer_patch_reverted",
                        path=path,
                        proposed=value,
                        restored=safe_value,
                        grounding_status=status,
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
    value = re.sub(
        r"^\d+(?:\.\d+)?\s*(?:年|个月|月)\s*技术\s*[:：]\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    # Models vary between "DOE实验设计" and "DOE 实验设计".  This is
    # typographic normalization, not an industry dictionary.
    value = re.sub(r"(?<=[A-Za-z0-9])\s+(?=[\u4e00-\u9fff])", "", value)
    value = re.sub(
        r"\s*[（(](?:精通|熟练(?:使用|掌握)?|掌握|熟悉|了解|入门)[）)]$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    # A skill token concatenated with a dated project/header is an OCR layout
    # artifact, not one coherent skill claim. Keep the original token in its
    # evidence-bound record rather than exposing the malformed composite.
    if len(value) > 12 and re.search(
        r"(?:19|20)\d{2}年(?:1[0-2]|0?[1-9])?月?\s*"
        r"[-–—~至到]\s*(?:(?:19|20)\d{2}年)?(?:1[0-2]|0?[1-9])月?",
        value,
    ):
        return ""
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


def _split_grounded_fact_bullet(value: str, *, limit: int = 6) -> list[str]:
    """Split only genuinely independent sentences in a grounded bullet.

    Commas and conjunctions commonly connect Situation/Action/Result facts in
    Chinese resumes. Repeating the leading verb across those fragments turns a
    complete achievement into several thin task statements and can detach a
    quantified result from its context. Sentence/semicolon boundaries are the
    only safe deterministic split points; the optimizer can still polish each
    complete unit afterwards.
    """

    text = re.sub(r"\s+", " ", str(value or "")).strip(" \t。；;")
    if not text:
        return []
    atoms = [
        item.strip(" \t，,。；;")
        for item in re.split(r"[。；;]+", text)
        if item.strip(" \t，,。；;")
    ]

    deduped: list[str] = []
    seen: set[str] = set()
    for atom in atoms[:limit]:
        normalized = re.sub(r"\W+", "", atom).casefold()
        if len(normalized) < 4 or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(atom)
    return deduped or [text]


def _atomize_resume_bullets(
    resume: CanonicalResume,
) -> tuple[CanonicalResume, dict[str, str]]:
    """Expand verified compound bullets into independently traceable facts."""

    atomized = resume.model_copy(deep=True)
    provenance: dict[str, str] = {}
    for section_name in ("experience", "research", "activities", "projects"):
        for record_index, record in enumerate(getattr(atomized, section_name)):
            expanded: list[tuple[str, str]] = []
            for original in list(record.bullets):
                source_value = str(original or "").strip()
                if not source_value:
                    continue
                atoms = _split_grounded_fact_bullet(source_value)
                expanded.extend((atom, source_value) for atom in atoms)
            record.bullets = []
            seen: set[str] = set()
            # Multi-page resumes are valid. A former ``[:6]`` presentation
            # limit silently discarded every later source-grounded bullet and
            # was a primary cause of sparse output on dense OCR cases.
            for atom, source_value in expanded:
                normalized = re.sub(r"\W+", "", atom).casefold()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                bullet_index = len(record.bullets)
                record.bullets.append(atom)
                if atom != source_value:
                    provenance[
                        f"{section_name}[{record_index}].bullets[{bullet_index}]"
                    ] = source_value
    return atomized, provenance


_QUALITY_DURATION_ONLY = re.compile(
    r"^(?:[-*•·▪◦]\s*)?\d+(?:\.\d+)?\s*(?:年|个月|月)$"
)
_QUALITY_EMPTY_METRIC = re.compile(
    r"(?:实现|提升|提高|增长|降低|减少|缩短|达到|达成)?了?\s*(?<!\d)[%％]"
)


def _quality_v2_presentation_cleanup(resume: CanonicalResume) -> CanonicalResume:
    """Remove deterministic non-claims without using profession vocabulary.

    A bare elapsed duration belongs in the structured period.  A result clause
    containing an empty percentage placeholder is incomplete source data, so
    only that clause is removed and the rest of the grounded sentence stays.
    """

    cleaned = resume.model_copy(deep=True)
    for section_name in ("experience", "research", "activities", "projects"):
        for record in getattr(cleaned, section_name):
            bullets: list[str] = []
            for raw in record.bullets:
                value = re.sub(r"\s+", " ", str(raw or "")).strip()
                if not value or _QUALITY_DURATION_ONLY.fullmatch(value):
                    continue
                clauses = [
                    clause.strip(" \t，,。；;")
                    for clause in re.split(
                        r"[。；;]+|(?<=[^\d])[,，](?=\s*(?:实现|提升|提高|增长|降低|减少|缩短|达到|达成))",
                        value,
                    )
                    if clause.strip(" \t，,。；;")
                ]
                retained = [
                    clause for clause in clauses
                    if not _QUALITY_EMPTY_METRIC.search(clause)
                ]
                repaired = "，".join(retained).strip(" \t，,。；;")
                if repaired and repaired not in bullets:
                    bullets.append(repaired)
            record.bullets = bullets
    return cleaned


def _needs_optimizer(
    resume: CanonicalResume,
    *,
    audited_source_scaffold: bool = False,
    narrative_record_keys: set[tuple[str, int]] | None = None,
) -> bool:
    """Return whether another LLM wording pass can add useful information.

    A compiler scaffold must not be sent through an unbounded whole-document
    rewrite.  It may, however, receive a record-scoped pass when the planner
    selected a bounded set of multi-clause records. Ordinary Composer output
    continues to receive the existing evidence-preserving wording pass.
    """

    if audited_source_scaffold:
        return bool(narrative_record_keys)

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


def _change_path_exists(resume: CanonicalResume, path: str) -> bool:
    """Drop stale indexed changes after record-level recovery or validation."""

    match = re.fullmatch(
        r"(experience|research|activities|projects)\[(\d+)]\.bullets\[(\d+)]",
        str(path or ""),
    )
    if not match:
        return True
    section, record_index, bullet_index = match.groups()
    records = getattr(resume, section)
    record_position = int(record_index)
    bullet_position = int(bullet_index)
    return (
        record_position < len(records)
        and bullet_position < len(records[record_position].bullets)
    )


def _bullet_path_value(resume: CanonicalResume, path: str) -> str | None:
    match = re.fullmatch(
        r"(experience|research|activities|projects)\[(\d+)]\.bullets\[(\d+)]",
        str(path or ""),
    )
    if not match:
        return None
    section, record_index, bullet_index = match.groups()
    records = getattr(resume, section)
    record_position = int(record_index)
    bullet_position = int(bullet_index)
    if record_position >= len(records):
        return None
    bullets = records[record_position].bullets
    if bullet_position >= len(bullets):
        return None
    return str(bullets[bullet_position]).strip()


def _filter_trusted_rewrites(
    resume: CanonicalResume,
    trusted_rewrites: dict[str, str],
    expected_outputs: dict[str, str],
) -> dict[str, str]:
    """Discard positional provenance after a bullet moves or is replaced."""

    return {
        path: source_value
        for path, source_value in trusted_rewrites.items()
        if path in expected_outputs
        and _bullet_path_value(resume, path) == str(expected_outputs[path]).strip()
    }


def _expand_optimizer_provenance(
    path: str,
    source_value: str,
    before_optimizer: CanonicalResume,
    atom_provenance: dict[str, str],
) -> str:
    """Lift grouped atom provenance back to every original source sentence."""

    source_parts = [
        part.strip()
        for part in re.split(r"[\r\n]+", str(source_value or ""))
        if part.strip()
    ]
    if len(source_parts) <= 1:
        return atom_provenance.get(path, str(source_value or "").strip())

    match = re.fullmatch(
        r"(experience|research|activities|projects)\[(\d+)]\.bullets\[\d+]",
        str(path or ""),
    )
    if not match:
        return "\n".join(source_parts)
    section, record_index_text = match.groups()
    record_index = int(record_index_text)
    records = getattr(before_optimizer, section)
    if record_index >= len(records):
        return "\n".join(source_parts)

    original_bullets = [
        str(item or "").strip() for item in records[record_index].bullets
    ]
    used_positions: set[int] = set()
    expanded: list[str] = []
    for source_part in source_parts:
        matched_position = next((
            index
            for index, bullet in enumerate(original_bullets)
            if index not in used_positions and bullet == source_part
        ), None)
        if matched_position is None:
            expanded.append(source_part)
            continue
        used_positions.add(matched_position)
        original_path = (
            f"{section}[{record_index}].bullets[{matched_position}]"
        )
        expanded.append(atom_provenance.get(original_path, source_part))
    return "\n".join(dict.fromkeys(item for item in expanded if item))


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
        claim = str(binding.source_claim or binding.claim or "").strip()
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


def _missing_units_need_source_recovery(source, missing_ids: list[str]) -> bool:
    """Whether a missing ledger item has a safe canonical recovery route."""

    missing = set(missing_ids)
    for unit in source_fact_units(source):
        if unit.get("unit_id") not in missing or _coverage_unit_is_optional(unit):
            continue
        section = str(unit.get("section_hint") or "")
        text = str(unit.get("match_text") or "").strip()
        if unit.get("record_body") == "true":
            return True
        if section in {
            "experience", "research", "activities", "projects", "skills",
            "awards", "publications", "patents", "certifications", "training", "teaching",
        }:
            return True
        if section == "education" and re.search(
            r"(?:大学|学院|学校|研究院|本科|硕士|博士|大专|学士|专业)", text,
        ):
            return True
        if re.search(
            r"(?:姓名|name|电话|手机|邮箱|email)\s*[:：]|"
            r"1[3-9]\d(?:[\s-]?\d){8}|"
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            text,
            re.IGNORECASE,
        ):
            return True
    return False


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
        if str(bullet).strip() and not _INCOMPLETE_TEXT_TAIL.search(str(bullet).strip("。；; "))
    ]
    if not candidates:
        return ""
    best = max(candidates, key=lambda value: _bullet_priority(value))
    if not (
        _QUANTIFIED_CHANGE.search(best)
        or _RESULT_SIGNAL.search(best)
        or _SCALED_DELIVERABLE.search(best)
    ):
        return ""
    return best.strip("。；; ")


def _summary_fact_entries(
    resume: CanonicalResume,
    target_context: str = "",
) -> list[tuple[str, str, tuple[str, int]]]:
    """Return grounded facts with their existing record context attached.

    A sparse source often has only one complete sentence per role.  Inventing
    missing STAR dimensions would be unsafe, but repeating that sentence in a
    neutral organization/role context makes the summary materially more useful
    without introducing a new action, method, deliverable, or result.
    """

    entries: list[tuple[str, str, tuple[str, int]]] = []
    for section_name in ("experience", "projects", "research", "activities"):
        records = getattr(resume, section_name)
        for record_index, record in enumerate(records):
            for raw_bullet in record.bullets:
                bullet = str(raw_bullet or "").strip("。；; ")
                if not bullet or _INCOMPLETE_TEXT_TAIL.search(bullet):
                    continue
                if section_name == "experience":
                    organization = record.organization.strip()
                    role = record.role.strip()
                    # A source bullet may already contain its own temporal or
                    # role context.  Prefixing the record identity again made
                    # summaries read like “在银行担任经理期间，在银行做…”.
                    already_contextual = bool(re.match(
                        r"^(?:在|于|担任|任职)",
                        bullet,
                    ))
                    if already_contextual:
                        contextual = bullet
                    elif organization and role:
                        contextual = f"在{organization}担任{role}期间，{bullet}"
                    elif organization:
                        contextual = f"在{organization}期间，{bullet}"
                    elif role:
                        contextual = f"担任{role}期间，{bullet}"
                    else:
                        contextual = bullet
                elif section_name == "projects":
                    identity = record.name.strip()
                    contextual = f"项目经历（{identity}）：{bullet}" if identity else bullet
                elif section_name == "research":
                    identity = "｜".join(
                        part.strip() for part in (record.institution, record.topic) if part.strip()
                    )
                    contextual = f"科研经历（{identity}）：{bullet}" if identity else bullet
                else:
                    identity = "｜".join(
                        part.strip() for part in (record.organization, record.role) if part.strip()
                    )
                    contextual = f"校园或社会经历（{identity}）：{bullet}" if identity else bullet
                entries.append((contextual, bullet, (section_name, record_index)))

    return sorted(
        entries,
        key=lambda item: _bullet_priority(item[1], target_context),
        reverse=True,
    )


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

    target_role = _clean_target_role(resume.meta.target_role)
    if re.fullmatch(r"[a-z0-9]+(?:[_-][a-z0-9]+)+", target_role, re.IGNORECASE):
        target_role = " ".join(
            token.upper() if len(token) <= 3 else token.title()
            for token in re.split(r"[_-]+", target_role)
            if token
        )
    # A source-grounded summary can carry valuable facts (for example eight
    # years of clinical work) that are not duplicated in another field. Keep
    # complete factual sentences, while filtering subjective filler. Reserve
    # the early slots for quantitative/seniority facts; generic source summary
    # prose must not crowd out a concrete role, achievement, or skill profile.
    source_summary_priority: list[str] = []
    source_summary_other: list[str] = []
    for sentence in (
        item.strip() for item in re.split(r"[。！？!?；;]+", resume.summary) if item.strip()
    ):
        if (
            len(sentence) >= 12
            and not _SUMMARY_SUBJECTIVE.search(sentence)
            and not _INCOMPLETE_TEXT_TAIL.search(sentence)
        ):
            destination = (
                source_summary_priority
                if re.search(r"\d+(?:\.\d+)?\s*(?:年|个月|月|%|人|次|项|篇|例|台|套)", sentence)
                or _RESULT_SIGNAL.search(sentence)
                else source_summary_other
            )
            destination.append(sentence)
    experience_bits: list[str] = []
    role_only_bits: list[str] = []
    for item in resume.experience[:2]:
        organization = item.organization.strip()
        role = item.role.strip()
        if organization:
            identity = "".join(part for part in (organization, role) if part)
            if identity and identity not in experience_bits:
                experience_bits.append(identity)
        elif role and role not in role_only_bits:
            role_only_bits.append(role)
    seniority = re.sub(
        r"(?<=\d)\s+(?=(?:年|个月|月))",
        "",
        resume.meta.work_experience.strip(),
    )
    if seniority and re.fullmatch(r"\d+(?:\.\d+)?\s*(?:年|个月|月)", seniority):
        seniority += "经验"
    profile_bits: list[str] = []
    if seniority in {"应届", "应届生"}:
        profile_bits.append("应届生")
    elif seniority == "无工作经验":
        profile_bits.append("目前无工作经验")
    elif seniority:
        profile_bits.append("拥有" + seniority)
    if experience_bits:
        profile_bits.append("曾任" + "、".join(experience_bits))
    elif role_only_bits:
        profile_bits.append("工作或实习经历包括" + "、".join(role_only_bits))
    if target_role:
        profile_bits.append("求职方向为" + target_role)
    if profile_bits:
        candidates.append("，".join(profile_bits))

    # Retain at most one high-value original summary sentence here.  The
    # remaining slots are reserved for concrete role facts, skills and
    # education so generic prose cannot make an otherwise complete summary
    # look sparse.
    candidates.extend(source_summary_priority[:1])

    research_bits: list[str] = []
    for item in resume.research[:2]:
        identity = "".join(part.strip() for part in (item.institution, item.topic) if part.strip())
        if identity and identity not in research_bits:
            research_bits.append(identity)
    achievement = _best_achievement(resume)
    fact_entries = _summary_fact_entries(resume, target_role)
    selected_fact_keys: set[tuple[str, int]] = set()
    selected_fact_texts: set[str] = set()
    if fact_entries:
        primary_index = 0
        if achievement:
            achievement_norm = re.sub(r"\W+", "", achievement).casefold()
            for index, (_contextual, raw_fact, _record_key) in enumerate(fact_entries):
                if re.sub(r"\W+", "", raw_fact).casefold() == achievement_norm:
                    primary_index = index
                    break
        contextual, raw_fact, record_key = fact_entries[primary_index]
        candidates.append(("代表成果：" if achievement else "代表经历：") + contextual)
        selected_fact_keys.add(record_key)
        selected_fact_texts.add(re.sub(r"\W+", "", raw_fact).casefold())

        # Prefer a second record so a concise summary represents more than the
        # single strongest achievement.  Fall back to another independent fact
        # from the same record when only one record exists.
        remaining = [
            item for index, item in enumerate(fact_entries)
            if index != primary_index
            and re.sub(r"\W+", "", item[1]).casefold() not in selected_fact_texts
        ]
        secondary = next((item for item in remaining if item[2] not in selected_fact_keys), None)
        if secondary is None and remaining:
            secondary = remaining[0]
        if secondary is not None:
            candidates.append("相关经历：" + secondary[0])

    skill_names = list(dict.fromkeys(item.name.strip() for item in resume.skills.items if item.name.strip()))[:6]
    if skill_names:
        candidates.append("核心技能包括" + "、".join(skill_names))

    if resume.education:
        edu = resume.education[0]
        qualification = "、".join(part.strip() for part in (edu.major, edu.degree) if part.strip())
        education_text = "，".join(part for part in (edu.school.strip(), qualification) if part)
        if education_text:
            candidates.append("教育背景为" + education_text)

    if research_bits:
        candidates.append("科研经历包括" + "、".join(research_bits))

    project_names = list(dict.fromkeys(item.name.strip() for item in resume.projects if item.name.strip()))[:2]
    if project_names:
        candidates.append("项目经历包括" + "、".join(project_names))

    if resume.publications:
        candidates.append(f"论文成果{len(resume.publications)}项")
    if resume.patents:
        candidates.append(f"专利成果{len(resume.patents)}项")
    if resume.certifications:
        candidates.append("持有" + "、".join(resume.certifications[:2]))

    if not experience_bits and not role_only_bits and not research_bits:
        activity_bits = list(dict.fromkeys(
            " ".join(part.strip() for part in (item.organization, item.role) if part.strip())
            for item in resume.activities
            if item.organization.strip() or item.role.strip()
        ))[:2]
        if activity_bits:
            candidates.append("校园或社会活动包括" + "、".join(activity_bits))

    candidates.extend(source_summary_other)

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
        if current_length + added > _SUMMARY_MAX_CHARS:
            continue
        compact.append(sentence)
        seen.add(normalized)
        current_length += added
        if len(compact) >= _SUMMARY_MAX_SENTENCES:
            break
    return "。".join(compact) + ("。" if compact else "")


_STRUCTURED_ORG_SUFFIX = re.compile(
    r"(?:大学|学院|学校|医院|公司|企业|集团|研究院|实验室|中心|部门|协会|学会|"
    r"学生会|社团|委员会|事务所|律所|银行|基金会|工作室|团队|基地|学部)(?:\d+)?$"
)
_SCHOOL_SUFFIX = re.compile(r"(?:大学|学院|学校|研究院|学部)(?:\d+)?")
_NON_SCHOOL_SENTENCE = re.compile(
    r"(?:准备找|求职|岗位|工作|简历|马上要毕业|已经毕业|毕业了|开始准备|"
    r"相关的工作|专业硕士|专业博士)"
)
_NON_MAJOR_SENTENCE = re.compile(
    r"(?:准备找|求职|岗位|工作|简历|马上要毕业|已经毕业|毕业了|开始准备|"
    r"最近开始|具备|熟悉|掌握|负责|参与|经验|能力|环境|定位)"
)
_ORGANIZATION_NARRATIVE_VALUE = re.compile(
    r"^(?:从|由).{1,60}(?:晋升|调任|转任)|"
    r"[，,；;].{0,40}(?:晋升|取得|获得|实现|提升|增长|确保|负责|主导|管理|领导)"
)


def _clean_structured_organization(value: str) -> str:
    """Remove narrative wrappers from an otherwise explicit organization."""

    text = re.sub(r"\s+", " ", str(value or "")).strip(" ，,。；;")
    if _ORGANIZATION_NARRATIVE_VALUE.search(text):
        return ""
    candidate = re.sub(
        r"^(?:我)?(?:目前)?(?:就职于|任职于|供职于|就读于|就读|毕业于|来自|在)\s*",
        "",
        text,
    ).strip()
    if candidate != text and _STRUCTURED_ORG_SUFFIX.search(candidate):
        text = candidate
    if text in {
        "公司", "企业", "学校", "学院", "医院", "事务所", "律所", "部门", "协会",
        "社团", "组织", "团队", "基地", "学生职务",
    }:
        return ""
    return text


_ROLE_WRAPPER = re.compile(
    r"^(?:我|本人)?(?:目前|曾经|曾)?(?:担任|任职为|任职|作为|职位为|岗位为|是)\s*"
)
_DUTY_ONLY_ROLE = re.compile(
    r"(?:单元|集成|功能|性能|接口|回归|压力|兼容性|验收|冒烟|安全)测试(?:工作|任务)?",
    re.IGNORECASE,
)
_ROLE_PERIOD_SENTENCE = re.compile(
    r"(?:19|20)\d{2}(?:[./年-]\d{1,2}月?)?\s*"
    r"(?:[-–—~至到]\s*)(?:(?:19|20)\d{2}|今|至今|现在)",
    re.IGNORECASE,
)
_ROLE_DUTY_SENTENCE = re.compile(
    r"^(?:负责(?!人)|参与|主导|协助|配合|完成|推动|推进|确保|促进|取得|获得|实现)\S+|"
    r"[，,；;][^，,；;]{0,48}(?:负责(?!人)|参与|主导|协助|配合|完成|推动|推进|"
    r"确保|促进|取得|获得|实现)\S*",
    re.IGNORECASE,
)


def _clean_role_title(value: str, *, explicit: bool = False) -> str:
    """Normalize role grammar and reject a task accidentally bound as a title.

    Evidence presence alone is insufficient for typed fields: ``单元测试`` may
    appear in a source bullet, but that does not make it a job title.  The guard
    is deliberately grammatical and narrow rather than an industry dictionary.
    Explicitly labelled fallback values still have their wrappers normalized;
    canonical model output receives the stricter semantic check below.
    """

    text = re.sub(r"\s+", " ", str(value or "")).strip(" ，,。；;|｜:：-–—")
    text = re.sub(r"^(?:岗位|职位|角色|职务)\s*[:：]\s*", "", text)
    text = _ROLE_WRAPPER.sub("", text).strip(" ，,。；;|｜:：-–—")
    if _ROLE_PERIOD_SENTENCE.search(text) or _ROLE_DUTY_SENTENCE.search(text):
        return ""
    # A percentage/currency/count is an achievement or duty fragment, never a
    # standalone position title. Preserve it as a bullet through the caller's
    # existing fallback instead of allowing it into the role field.
    if re.search(
        r"\d+(?:\.\d+)?\s*(?:%|％|万|亿|元|万元|人|次|项|个|条|台|套)",
        text,
        re.IGNORECASE,
    ):
        return ""
    if not explicit and _DUTY_ONLY_ROLE.fullmatch(text):
        return ""
    return text


def _clean_education_fields(item: dict) -> None:
    """Keep grounded education facts while rejecting sentence fragments.

    Evidence substring checks alone cannot distinguish a school name from a
    nearby sentence such as “马上要毕业了”. This grammar-level cleanup is
    profession-independent and only normalizes explicit education syntax.
    """

    school = _clean_structured_organization(str(item.get("school", "") or ""))
    school = re.sub(r"^(?:教育经历|教育背景|学历信息)\s*[:：]\s*", "", school)
    school_matches = list(_SCHOOL_SUFFIX.finditer(school))
    if school_matches:
        school = school[:max(match.end() for match in school_matches)].strip()
    elif (
        _NON_SCHOOL_SENTENCE.search(school)
        or len(school) > 50
        or bool(_FALLBACK_DEGREE.fullmatch(
            re.sub(r"^\d{1,3}(?:[.、)]\s*)?", "", school).strip()
        ))
    ):
        school = ""
    item["school"] = school

    major = re.sub(r"\s+", " ", str(item.get("major", "") or "")).strip(" ，,。；;")
    explicit_major = re.search(
        r"(?:攻读|就读|读)\s*([^，,。；;]{2,40}?)(?:专业|方向)(?:硕士|博士|本科)?$",
        major,
    )
    if explicit_major:
        major = explicit_major.group(1).strip()
    else:
        major = re.sub(r"^(?:攻读|就读|读)\s*", "", major)
        major = re.sub(r"专业(?:硕士|博士|本科)?$", "", major).strip()
    if _NON_MAJOR_SENTENCE.search(major) or len(major) > 50:
        major = ""
    item["major"] = major


def _drop_subsumed_education(records: list[dict]) -> list[dict]:
    """Drop weak duplicate rows already represented by a richer row."""

    fields = ("school", "degree", "major", "period")
    signatures = [
        {
            field: re.sub(r"\s+", "", str(record.get(field, "") or "")).casefold()
            for field in fields
            if str(record.get(field, "") or "").strip()
        }
        for record in records
    ]
    retained: list[dict] = []
    for index, record in enumerate(records):
        signature = signatures[index]
        subsumed = bool(signature) and any(
            index != other_index
            and len(other_signature) > len(signature)
            and all(other_signature.get(field) == value for field, value in signature.items())
            for other_index, other_signature in enumerate(signatures)
        )
        if not subsumed:
            retained.append(record)
    return retained


def _coalesce_compatible_education(records: list[dict]) -> list[dict]:
    """Merge complementary rows for the same school and qualification."""

    fields = ("school", "degree", "major", "period")
    merged: list[dict] = []
    for record in records:
        school = re.sub(r"\s+", "", str(record.get("school", "") or "")).casefold()
        degree = re.sub(
            r"(?:在读|毕业)$", "",
            re.sub(r"\s+", "", str(record.get("degree", "") or "")).casefold(),
        )
        destination: dict | None = None
        if school and degree:
            for existing in merged:
                existing_school = re.sub(
                    r"\s+", "", str(existing.get("school", "") or "")
                ).casefold()
                existing_degree = re.sub(
                    r"(?:在读|毕业)$", "",
                    re.sub(r"\s+", "", str(existing.get("degree", "") or "")).casefold(),
                )
                if (school, degree) != (existing_school, existing_degree):
                    continue
                # Different renderings of the same graduation date (for
                # example “预计2026年毕业” and “2026年”) describe one record.
                # Distinct non-empty majors remain a real conflict.
                conflicts = [
                    field for field in ("major",)
                    if str(record.get(field, "") or "").strip()
                    and str(existing.get(field, "") or "").strip()
                    and re.sub(r"\s+", "", str(record.get(field))).casefold()
                    != re.sub(r"\s+", "", str(existing.get(field))).casefold()
                ]
                if not conflicts:
                    destination = existing
                    break
        if destination is None:
            merged.append(record)
            continue
        for field in fields:
            if not str(destination.get(field, "") or "").strip() and str(
                record.get(field, "") or ""
            ).strip():
                destination[field] = record[field]
        incoming_period = str(record.get("period", "") or "").strip()
        current_period = str(destination.get("period", "") or "").strip()
        if incoming_period and len(incoming_period) > len(current_period):
            destination["period"] = incoming_period
    return merged


def _coalesce_same_record_duplicates(
    records: list[dict],
    *,
    section: str,
) -> list[dict]:
    """Merge duplicate render rows only under a strong same-record anchor.

    Chunked LLM output and deterministic source recovery can emit the same
    source record twice with a shortened title or punctuation-only date
    variation.  An exact non-empty period plus compatible identity is enough;
    otherwise require literal bullet overlap.  Distinct concurrent roles with
    different identities and duties remain separate.
    """

    identity_fields = {
        "experience": ("organization", "role"),
        "research": ("institution", "topic"),
        "activities": ("organization", "role"),
        "projects": ("name", "organization", "role"),
    }.get(section, ())

    def normalized(value: object) -> str:
        return re.sub(r"\W+", "", str(value or "")).casefold()

    def period_key(record: dict) -> str:
        return normalized(record.get("period", ""))

    def compatible(left: str, right: str) -> bool:
        return bool(
            left == right
            or (min(len(left), len(right)) >= 3 and (left in right or right in left))
        )

    def bullet_overlap(left: dict, right: dict) -> bool:
        left_values = [normalized(value) for value in left.get("bullets", [])]
        right_values = [normalized(value) for value in right.get("bullets", [])]
        return any(
            min(len(first), len(second)) >= 8
            and (first in second or second in first)
            for first in left_values
            for second in right_values
            if first and second
        )

    def same_record(left: dict, right: dict) -> bool:
        left_period = period_key(left)
        if not left_period or left_period != period_key(right):
            return False
        if bullet_overlap(left, right):
            return True
        pairs = [
            (normalized(left.get(field)), normalized(right.get(field)))
            for field in identity_fields
            if normalized(left.get(field)) and normalized(right.get(field))
        ]
        return bool(pairs) and all(compatible(first, second) for first, second in pairs)

    def merge_bullets(target: dict, source: dict) -> None:
        for value in source.get("bullets", []):
            candidate = str(value or "").strip()
            candidate_norm = normalized(candidate)
            if not candidate_norm:
                continue
            replacement = None
            represented = False
            for index, current in enumerate(target.setdefault("bullets", [])):
                current_norm = normalized(current)
                if candidate_norm == current_norm or candidate_norm in current_norm:
                    represented = True
                    break
                if len(current_norm) >= 8 and current_norm in candidate_norm:
                    replacement = index
                    break
            if represented:
                continue
            if replacement is None:
                target["bullets"].append(candidate)
            else:
                target["bullets"][replacement] = candidate

    merged: list[dict] = []
    for record in records:
        destination = next(
            (candidate for candidate in merged if same_record(candidate, record)),
            None,
        )
        if destination is None:
            merged.append(record)
            continue
        for field in identity_fields:
            current = str(destination.get(field, "") or "").strip()
            value = str(record.get(field, "") or "").strip()
            if not current and value:
                destination[field] = value
            elif current and value and compatible(normalized(current), normalized(value)):
                if len(normalized(value)) > len(normalized(current)):
                    destination[field] = value
        merge_bullets(destination, record)
    return merged


def _clean_certification_fragments(values: list[str]) -> list[str]:
    """Prefer complete source credentials over model-created fragments."""

    def normalized(value: object) -> str:
        text = re.sub(r"^(?:[-*•·▪◦]\s*)", "", str(value or "").strip())
        return re.sub(r"\W+", "", text).casefold()

    values = [
        str(value or "").strip() for value in values
        if str(value or "").strip()
        if not re.fullmatch(
            r"[-*•·▪◦\s()（）]*(?:\d{2,5})[-*•·▪◦\s()（）]*",
            str(value or ""),
        )
    ]

    def without_trailing_year(value: str) -> str:
        return re.sub(
            r"(?:\s*[-–—]\s*(?:19|20)\d{2}|\s*[（(](?:19|20)\d{2}[)）])\s*$",
            "",
            re.sub(r"^(?:[-*•·▪◦]\s*)", "", value).strip(),
        ).strip()

    # If both ``证书名`` and ``证书名 (2009)`` exist, the year attachment is
    # an inferred cross-line relationship.  Keep the explicit undated item;
    # a dated credential remains untouched when it is the only source form.
    undated = {
        normalized(value)
        for value in values
        if normalized(without_trailing_year(value)) == normalized(value)
    }
    values = [
        value for value in values
        if not (
            normalized(without_trailing_year(value)) != normalized(value)
            and normalized(without_trailing_year(value)) in undated
        )
    ]
    complete = [
        normalized(value) for value in values
        if _CERTIFICATION_SKILL.search(value)
    ]
    result: list[str] = []
    for value in values:
        value_norm = normalized(value)
        if (
            value_norm
            and not _CERTIFICATION_SKILL.search(value)
            and any(
                value_norm != candidate
                and len(value_norm) >= 3
                and value_norm in candidate
                for candidate in complete
            )
        ):
            continue
        if value_norm and value_norm not in {normalized(item) for item in result}:
            result.append(value)
    return result


def _clean_record_period(value: object, *, section: str) -> str:
    """Reject impossible non-education dates introduced by OCR or generation.

    Education may legitimately contain a future expected-graduation date, so
    its period is left to the existing evidence gate.  Employment-like rows
    accept only four-digit calendar years and real months.  This is a syntax
    invariant, not an attempt to infer or correct a candidate's date.
    """

    period = re.sub(r"\s+", " ", str(value or "")).strip()
    if not period or section == "education":
        return period

    maximum_year = date.today().year + 1
    year_tokens = re.findall(r"(?<!\d)(\d{2,5})(?=\s*年)", period)
    year_month_dates = re.findall(
        r"(?<!\d)((?:19|20)\d{2})[./-](\d{1,2})(?!\d)",
        period,
    )
    month_year_dates = re.findall(
        r"(?<!\d)(\d{1,2})[./-]((?:19|20)\d{2})(?!\d)",
        period,
    )
    month_tokens = re.findall(r"(?:\d{4}\s*年\s*)(\d{1,2})(?=\s*月)", period)

    # Standalone years cover ranges such as ``2025 – 至今``.  Keeping this
    # separate from the formatted-date matches also lets us accept both the
    # Chinese/common YYYY/MM form and the international MM/YYYY form without
    # interpreting ``05/2025`` as year 05, month 2025.
    standalone_years = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", period)
    years = (
        list(year_tokens)
        + list(standalone_years)
        + [year for year, _month in year_month_dates]
        + [year for _month, year in month_year_dates]
    )
    months = (
        list(month_tokens)
        + [month for _year, month in year_month_dates]
        + [month for month, _year in month_year_dates]
    )
    if not years and re.search(r"(?:年|月|至今|现在|present)", period, re.IGNORECASE):
        return ""
    if any(
        len(year) != 4 or not (1900 <= int(year) <= maximum_year)
        for year in years
    ):
        return ""
    if any(not (1 <= int(month) <= 12) for month in months):
        return ""
    return period


def _prune_redundant_highlights(data: dict) -> None:
    """Keep genuine standalone highlights, not parser recovery duplicates.

    OCR recovery may temporarily route an unowned clause to ``经历亮点`` and
    later recover the same clause into its proper record.  Retaining both
    creates a duplicated, fragmented tail section.  Compare against the union
    of already-structured claims so a fact represented across two bullets is
    also recognized; a genuinely unique highlight remains untouched.
    """

    additional = data.get("additional_sections")
    if not isinstance(additional, dict) or not isinstance(additional.get("经历亮点"), list):
        return

    structured: list[str] = [str(data.get("summary", "") or "")]
    record_fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, fields in record_fields.items():
        for record in data.get(section, []) or []:
            if not isinstance(record, dict):
                continue
            structured.extend(str(record.get(field, "") or "") for field in fields)
            structured.extend(str(value or "") for value in record.get("bullets", []) or [])
    for section in (
        "awards", "publications", "patents", "certifications", "training", "teaching",
    ):
        structured.extend(str(value or "") for value in data.get(section, []) or [])
    skills = data.get("skills") or {}
    if isinstance(skills, dict):
        structured.extend(
            str(item.get("name", "") or "")
            for item in skills.get("items", []) or []
            if isinstance(item, dict)
        )

    claim_bigrams: set[str] = set()
    normalized_claims: list[str] = []
    for claim in structured:
        normalized = re.sub(r"\W+", "", claim).casefold()
        if not normalized:
            continue
        normalized_claims.append(normalized)
        claim_bigrams.update(_coverage_bigrams(claim))

    retained: list[str] = []
    for raw in additional.get("经历亮点", []):
        value = str(raw or "").strip()
        for heading in sorted(_LAYOUT_RESET_HEADINGS, key=len, reverse=True):
            if len(value) > len(heading) + 4 and value.endswith(heading):
                value = value[:-len(heading)].rstrip(" ：:；;，,。|｜-—")
                break
        normalized = re.sub(r"\W+", "", value).casefold()
        if not normalized:
            continue
        represented = any(
            normalized == claim
            or (len(normalized) >= 6 and normalized in claim)
            for claim in normalized_claims
        )
        if not represented:
            source_bigrams = _coverage_bigrams(value)
            represented = bool(
                source_bigrams
                and len(source_bigrams & claim_bigrams) / len(source_bigrams) >= 0.82
            )
        if not represented and value not in retained:
            retained.append(value)

    if retained:
        additional["经历亮点"] = retained
    else:
        additional.pop("经历亮点", None)


def _is_recovery_layout_fragment(value: object) -> bool:
    """Identify parser seams that are not publishable resume claims.

    OCR column recovery can concatenate the end of one record with the
    identity/date row of the next.  These strings may still overlap the OCR
    transcript strongly enough to pass an evidence check, but they are layout
    artifacts rather than candidate statements.  Keep the invariant narrow:
    reject only explicit empty date skeletons, dangling ``负责：`` headings,
    and ``此前担任`` record headers joined by a non-numeric ``>`` seam.
    """

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return False
    if re.search(r"年\s*月\s*[-–—~至到]\s*年\s*月", text):
        return True
    if re.search(r"(?:^|[。；;])\s*负责\s*[:：]\s*$", text):
        return True
    if re.search(r"(?:此前|曾经?)担任", text) and re.search(
        r"[>＞](?=[^\d\s])", text,
    ):
        return True
    return False


def _prune_recovery_layout_fragments(data: dict) -> None:
    """Remove explicit OCR record seams without rewriting valid prose."""

    for section in ("experience", "research", "activities", "projects"):
        for record in data.get(section, []) or []:
            if not isinstance(record, dict):
                continue
            record["bullets"] = [
                str(value).strip()
                for value in record.get("bullets", []) or []
                if str(value).strip() and not _is_recovery_layout_fragment(value)
            ]
    additional = data.get("additional_sections")
    if not isinstance(additional, dict):
        return
    highlights = additional.get("经历亮点")
    if isinstance(highlights, list):
        kept = [
            str(value).strip()
            for value in highlights
            if str(value).strip() and not _is_recovery_layout_fragment(value)
        ]
        if kept:
            additional["经历亮点"] = kept
        else:
            additional.pop("经历亮点", None)


_UNOWNED_HIGHLIGHT_HARD_ANCHOR = re.compile(
    r"(?<!\d)\d+(?:\.\d+)?\s*(?:%|％|万|亿|千|人|次|项|个|条|篇|例|台|套|"
    r"元|万元|美元|小时|天|个月|月|年)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _prune_unowned_compiler_highlight_metrics(
    data: dict,
    *,
    source,
    routes: list,
) -> None:
    """Require provenance for hard anchors routed to a loose highlight.

    A numeric claim attached to a structured record is protected by record
    ownership.  ``经历亮点`` has no such owner, so a newly recovered hard
    anchor is publishable only when the source explicitly labels it as a
    highlight, or when it came directly from the user's query.  This blocks a
    unanimous-but-wrong OCR number without discarding ordinary prose.
    """

    additional = data.get("additional_sections")
    if not isinstance(additional, dict) or not isinstance(additional.get("经历亮点"), list):
        return
    facts = {fact.fact_id: fact for fact in source.fact_units}
    trusted_source_values = [
        facts[route.fact_id].verbatim_text
        for route in routes
        if route.fact_id in facts
        and route.destination_section == "additional_sections.经历亮点"
        and (
            route.reason == "typed_long_tail_section"
            or facts[route.fact_id].source_type == "query"
        )
    ]

    def explicitly_supported(value: str) -> bool:
        value_norm = re.sub(r"\W+", "", value).casefold()
        value_bigrams = _coverage_bigrams(value)
        for source_value in trusted_source_values:
            source_norm = re.sub(r"\W+", "", str(source_value)).casefold()
            if value_norm and (value_norm in source_norm or source_norm in value_norm):
                return True
            source_bigrams = _coverage_bigrams(str(source_value))
            if value_bigrams and (
                len(value_bigrams & source_bigrams) / len(value_bigrams)
            ) >= 0.82:
                return True
        return False

    retained = []
    for raw in additional.get("经历亮点", []):
        value = str(raw or "").strip()
        if not value:
            continue
        if _UNOWNED_HIGHLIGHT_HARD_ANCHOR.search(value) and not explicitly_supported(value):
            continue
        retained.append(value)
    if retained:
        additional["经历亮点"] = retained
    else:
        additional.pop("经历亮点", None)


def _clean_compiler_presentation_data(data: dict, *, source, routes: list) -> None:
    """Apply compaction invariants without rebuilding summary prose."""

    for section in ("experience", "research", "activities", "projects"):
        records = [item for item in data.get(section, []) or [] if isinstance(item, dict)]
        data[section] = _coalesce_same_record_duplicates(records, section=section)
    data["certifications"] = _clean_certification_fragments([
        str(value or "").strip()
        for value in data.get("certifications", []) or []
        if str(value or "").strip()
    ])
    _prune_recovery_layout_fragments(data)
    _prune_redundant_highlights(data)
    _prune_unowned_compiler_highlight_metrics(
        data,
        source=source,
        routes=routes,
    )


def _compact_canonical(resume: CanonicalResume) -> CanonicalResume:
    """Remove blank records/items left by model repair or leakage cleanup."""

    data = resume.model_dump()
    meta = data.get("meta") or {}
    if isinstance(meta, dict):
        raw_name = re.sub(r"\s+", "", str(meta.get("name", "") or ""))
        if raw_name.casefold() in {
            "个人", "个人简历", "简历", "姓名", "候选人", "candidate", "resume", "cv",
        }:
            meta["name"] = ""
        phone = re.search(r"1[3-9]\d{9}", re.sub(r"[\s-]+", "", str(meta.get("phone", ""))))
        meta["phone"] = phone.group(0) if phone else ""
        raw_email = re.sub(r"\s+", "", str(meta.get("email", "")))
        at = raw_email.find("@")
        cleaned_email = ""
        if at > 0:
            local = re.search(r"[A-Za-z0-9._%+-]+$", raw_email[:at])
            domain = re.match(
                r"[A-Za-z0-9.-]+?\.(?:com|cn|net|org|edu|gov|io|ai|co)(?:\.cn)?",
                raw_email[at + 1:],
                re.IGNORECASE,
            )
            if local and domain:
                local_value = re.sub(r"^1[3-9]\d{9}", "", local.group(0))
                if local_value:
                    cleaned_email = f"{local_value}@{domain.group(0)}"
        meta["email"] = cleaned_email
        meta["target_role"] = _clean_target_role(
            str(meta.get("target_role", "") or "")
        )
    for section in ("awards", "publications", "patents", "certifications", "training", "teaching"):
        values = [str(v).strip() for v in data.get(section, []) if str(v).strip()]
        if section == "certifications":
            values = [
                value for value in values
                if value not in {"资格证书", "证书", "资质证书"}
                and not re.fullmatch(
                    r"(?:19|20)\d{2}(?:[./-]\d{1,2}|年\d{1,2}月?)?\s*(?:获得|获)",
                    value,
                )
            ]
            values = _clean_certification_fragments(values)
        data[section] = list(dict.fromkeys(values))
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
            item["period"] = _clean_record_period(item.get("period", ""), section=section)
            if section == "education":
                _clean_education_fields(item)
                major = str(item.get("major", "") or "").strip()
                if re.fullmatch(r"(?:\d+|[IVXLCDM]+)", major, re.IGNORECASE):
                    item["major"] = ""
            elif "organization" in item:
                item["organization"] = _clean_structured_organization(
                    str(item.get("organization", "") or "")
                )
            if section != "education":
                item["bullets"] = list(dict.fromkeys(
                    str(v).strip() for v in item.get("bullets", []) if str(v).strip()
                ))
                if "role" in item:
                    original_role = str(item.get("role", "") or "").strip()
                    cleaned_role = _clean_role_title(original_role)
                    item["role"] = cleaned_role
                    if original_role and not cleaned_role:
                        # Preserve the grounded duty as content instead of
                        # silently deleting it after removing the bad field.
                        normalized_role = re.sub(r"\W+", "", original_role).casefold()
                        represented = any(
                            normalized_role
                            and normalized_role in re.sub(r"\W+", "", bullet).casefold()
                            for bullet in item["bullets"]
                        )
                        if not represented:
                            replacement_index: int | None = None
                            original_bigrams = _char_bigrams(original_role)
                            if original_bigrams:
                                ranked = sorted(
                                    (
                                        len(_char_bigrams(bullet) & original_bigrams)
                                        / max(1, len(_char_bigrams(bullet))),
                                        index,
                                    )
                                    for index, bullet in enumerate(item["bullets"])
                                    if _char_bigrams(bullet)
                                )
                                if (
                                    ranked
                                    and ranked[-1][0] >= 0.60
                                    and len(original_role) > len(item["bullets"][ranked[-1][1]])
                                ):
                                    replacement_index = ranked[-1][1]
                            if replacement_index is None:
                                item["bullets"].insert(0, original_role)
                            else:
                                item["bullets"][replacement_index] = original_role
                if section == "projects":
                    project_name = str(item.get("name", "") or "").strip()
                    compact_name = re.sub(r"\W+", "", project_name)
                    invalid_name = bool(
                        re.fullmatch(r"\d+", compact_name)
                        or (2 <= len(compact_name) <= 3 and len(set(compact_name)) == 1)
                        or len(project_name) > 80
                        or _looks_like_record_body(project_name)
                    )
                    if invalid_name:
                        if _looks_like_record_body(project_name) and not any(
                            _identity_value(project_name) == _identity_value(bullet)
                            for bullet in item["bullets"]
                        ):
                            item["bullets"].insert(0, project_name)
                        item["name"] = ""
                    if not item.get("name"):
                        for bullet in list(item["bullets"]):
                            candidate = str(bullet or "").strip(" ，,。；;")
                            if (
                                2 <= len(candidate) <= 60
                                and not _looks_like_record_body(candidate)
                                and re.search(
                                    r"(?:项目|系统|平台|小程序|APP|课题|作品)$",
                                    candidate,
                                    re.IGNORECASE,
                                )
                            ):
                                item["name"] = candidate
                                item["bullets"].remove(bullet)
                                break
                identity_norms = {
                    re.sub(r"\W+", "", str(item.get(field, "") or "")).casefold()
                    for field in fixed_fields
                    if str(item.get(field, "") or "").strip()
                }
                item["bullets"] = [
                    bullet for bullet in item["bullets"]
                    if not (
                        not _looks_like_record_body(bullet)
                        and re.sub(r"\W+", "", bullet).casefold() in identity_norms
                    )
                ]
            bullets = item.get("bullets", []) if section != "education" else []
            if (
                section != "education"
                and not bullets
                and not any(
                    str(item.get(field, "") or "").strip()
                    for field in fixed_fields
                    if field != "period"
                )
            ):
                # A date without an employer/title/project is not a readable
                # record and often comes from a split compact source header.
                continue
            if any(str(item.get(field, "")).strip() for field in fixed_fields) or bullets:
                identity = tuple(
                    re.sub(r"\s+", "", str(item.get(field, "")).strip()).casefold()
                    for field in fixed_fields
                )
                if any(identity) and identity in record_indexes:
                    existing = cleaned[record_indexes[identity]]
                    if section != "education":
                        combined = list(dict.fromkeys(
                            list(existing.get("bullets", [])) + list(item.get("bullets", []))
                        ))
                        normalized = [
                            re.sub(r"\W+", "", str(value or "")).casefold()
                            for value in combined
                        ]
                        # Identical identity rows commonly come from a second
                        # OCR/DOCX column pass.  Prefer the complete sentence
                        # over its line-wrapped prefix, but only inside this
                        # already-proven duplicate record.
                        existing["bullets"] = [
                            value
                            for index, value in enumerate(combined)
                            if not (
                                len(normalized[index]) >= 16
                                and any(
                                    index != other_index
                                    and len(other) > len(normalized[index])
                                    and other.startswith(normalized[index])
                                    for other_index, other in enumerate(normalized)
                                )
                            )
                        ]
                    continue
                if any(identity):
                    record_indexes[identity] = len(cleaned)
                cleaned.append(item)
        if section == "education":
            cleaned = _coalesce_compatible_education(cleaned)
            cleaned = _drop_subsumed_education(cleaned)
        else:
            cleaned = _coalesce_same_record_duplicates(cleaned, section=section)
        data[section] = cleaned

    # Exact standalone skill tokens occasionally survive OCR routing as both a
    # skill and an experience bullet (for example a bullet containing only
    # ``HTML5``). The skill section already preserves the same grounded fact;
    # keeping the duplicate in a narrative record creates a visible fragment
    # without adding information.
    skill_norms = {
        re.sub(r"[^\w\u4e00-\u9fff+.#/_-]+", "", str(item.get("name", ""))).casefold()
        for item in (data.get("skills") or {}).get("items", [])
        if isinstance(item, dict) and str(item.get("name", "")).strip()
    }
    if skill_norms:
        for section in ("experience", "research", "activities", "projects"):
            for item in data.get(section, []) or []:
                if not isinstance(item, dict):
                    continue
                item["bullets"] = [
                    bullet
                    for bullet in item.get("bullets", []) or []
                    if re.sub(
                        r"[^\w\u4e00-\u9fff+.#/_-]+",
                        "",
                        str(bullet or ""),
                    ).casefold() not in skill_norms
                ]

    bullet_entries = [
        (section, item, bullet, re.sub(
            r"\W+", "",
            re.sub(r"^(?:[-*•·▪◦]\s*|\d{1,3}(?:[、)]|\.(?!\d))\s*)", "", bullet),
        ).casefold())
        for section in ("experience", "projects", "research", "activities")
        for item in data.get(section, [])
        for bullet in item.get("bullets", [])
        if str(bullet).strip()
    ]
    all_bullet_norms = [entry[3] for entry in bullet_entries if entry[3]]
    seen_bullets: set[str] = set()
    for section in ("experience", "projects", "research", "activities"):
        for item in data.get(section, []):
            unique: list[str] = []
            for bullet in item.get("bullets", []):
                normalized = re.sub(
                    r"\W+", "",
                    re.sub(
                        r"^(?:[-*•·▪◦]\s*|\d{1,3}(?:[、)]|\.(?!\d))\s*)",
                        "",
                        str(bullet),
                    ),
                ).casefold()
                if not normalized or normalized in seen_bullets:
                    continue
                if len(normalized) >= 8 and any(
                    normalized != other
                    and normalized in other
                    and len(normalized) <= int(len(other) * 0.72)
                    for other in all_bullet_norms
                ):
                    continue
                seen_bullets.add(normalized)
                unique.append(bullet)
            item["bullets"] = unique

    _prune_recovery_layout_fragments(data)
    _prune_redundant_highlights(data)
    data["summary"] = str(data.get("summary", "") or "").strip()
    compacted = CanonicalResume.model_validate(data)
    compacted.summary = _build_evidence_summary(compacted)
    return compacted


_FALLBACK_YEAR_MONTH = (
    r"(?:19|20)\d{2}(?:(?:[./-](?:1[0-2]|0?[1-9]))|"
    r"(?:年(?:1[0-2]|0?[1-9])月?)|年)?(?!\d)"
)
_FALLBACK_MONTH_YEAR = r"(?:1[0-2]|0?[1-9])[-/](?:19|20)\d{2}(?!\d)"
_FALLBACK_SAME_YEAR_MONTH_RANGE = (
    r"(?:19|20)\d{2}年(?:1[0-2]|0?[1-9])月?\s*"
    r"[-–—~至到]\s*(?:1[0-2]|0?[1-9])月"
)
_FALLBACK_PERIOD = re.compile(
    rf"(?:{_FALLBACK_SAME_YEAR_MONTH_RANGE}|"
    rf"(?:{_FALLBACK_YEAR_MONTH}|{_FALLBACK_MONTH_YEAR})"
    rf"\s*(?:[-–—~至到]\s*(?:{_FALLBACK_YEAR_MONTH}|{_FALLBACK_MONTH_YEAR}|今|至今|现在))?)"
)
_FALLBACK_DURATION_SUFFIX = re.compile(
    r"\s*[•·|｜]\s*\d+(?:\.\d+)?\s*(?:年|个月|月)(?![\u4e00-\u9fff])"
)
_FALLBACK_DURATION_LINE = re.compile(
    r"^(?:[-*•·▪◦]\s*)?(?P<duration>\d+(?:\.\d+)?\s*(?:年|个月|月))$"
)
_FALLBACK_ORGANIZATION = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9·.&（）()_-]{0,40}?(?:大学|学院|学校|医院|公司|企业|集团|"
    r"中学|小学|幼儿园|学部|研究院|实验室|机构|中心|部门|协会|学会|学生会|社团|委员会|事务所|律所|银行|"
    r"基金会|工作室|团队|基地|科技|软件|电商|证券|互动)(?:\d+)?"
)
_FALLBACK_ORGANIZATION_END = re.compile(
    r"(?:幼儿园|研究院|实验室|委员会|工作室|基金会|学生会|事务所|"
    r"大学|学院|学校|中学|小学|学部|医院|公司|企业|集团|机构|中心|部门|"
    r"协会|学会|社团|律所|银行|团队|基地|科技|软件|电商|证券|互动)"
)
_ROLE_NARRATIVE_PREFIX = re.compile(
    r"^(?:我|本人)?(?:目前|现在|之前|过去|毕业后)?(?:一直)?"
    r"(?:做(?!过)|从事|担任|任职为)\s*",
    re.IGNORECASE,
)
_EXPLICIT_ORGANIZATION_END = re.compile(
    r"(?:幼儿园|研究院|实验室|委员会|工作室|基金会|学生会|事务所|"
    r"大学|学院|学校|中学|小学|学部|医院|公司|集团|机构|中心|部门|"
    r"协会|学会|社团|律所|银行|团队|基地)$"
)
_FALLBACK_ROLE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9/+.#_-]{0,24}(?:工程师|设计师|教师|老师|医生|医师|"
    r"护士|经理|主管|总监|主任|顾问|研究员|专员|助理|助教|负责人|组长|队长|主席|"
    r"部长|干事|部员|成员|委员|志愿者|实习生|实习|见习|分析师|架构师|运营|产品|开发|测试|销售|讲师|管理岗|岗位|岗)"
)
_FALLBACK_DEGREE = re.compile(
    r"(?:(?:通识研究)?(?:文科|理科|哲学|经济学|法学|教育学|文学|历史学|"
    r"理学|工学|农学|医学|管理学|艺术学)?(?:副学士|学士|硕士|博士)(?:学位|研究生)?|"
    r"博士研究生|硕士研究生|本科|大专|专科)(?:在读|毕业|毕业证书)?|"
    r"(?:高中|中学|初中)(?:在读|毕业(?:证书)?)?(?![\u4e00-\u9fff])"
)
_FALLBACK_PLACEHOLDER_TOKEN = re.compile(
    r"\[(?:姓名|姓氏|电话|手机|邮箱|地址|城市|州|国家|公司|学校|大学|组织|奖项|"
    r"linkedin[_\s-]*档案|github[_\s-]*链接)(?:[^\]\n]{0,24})?\]?|"
    r"<[^>\n]{1,48}>|\{\{[^}\n]{1,80}\}\}",
    re.IGNORECASE,
)
_FALLBACK_DATE_TOKEN = re.compile(
    r"^(?:19|20)\d{2}(?:(?:[./-]\d{1,2})|(?:年\d{1,2}月?))?\s*(?:至)?\s*$"
)
_FALLBACK_LABELED_SKILL = re.compile(
    r"^(?:[-•·]\s*)?(?P<label>技能|专业技能|工具|语言|语言能力)"
    r"\s*[:：]\s*(?P<value>.+)$"
)
_FALLBACK_RELATION_LABEL = re.compile(
    r"^(?:指导老师|指导教师|导师|项目导师|论文导师|推荐人|联系人)\s*[:：]",
    re.IGNORECASE,
)


def _first_match(pattern: re.Pattern[str], value: str) -> str:
    match = pattern.search(value)
    return match.group(0).strip() if match else ""


def _labeled_value(value: str, labels: tuple[str, ...]) -> str:
    joined = "|".join(re.escape(label) for label in labels)
    match = re.search(rf"(?:{joined})\s*[:：]\s*([^|｜，,；;\n]+)", value)
    return match.group(1).strip() if match else ""


_FALLBACK_DUTY_BOUNDARY = re.compile(
    r"[，,。；;]\s*(?=(?:负责|参与|主导|协助|支持|配合|完成|推动|推进|"
    r"组织|设计|开发|构建|实现|制定|管理|运营|分析|研究|撰写|输出|交付|"
    r"维护|优化|搭建|建立|开展|承担|提供|跟进|协调|带领|执行)"
    r"(?!\s*(?:工程师|设计师|架构师|分析师|研究员|教师|老师|医生|医师|"
    r"经理|主管|总监|主任|顾问|专员|助理|负责人|运营|实习生|实习|见习|岗)))"
)


def _identity_prefix(value: str) -> str:
    """Return only the identity portion of a compact record row."""

    return _FALLBACK_DUTY_BOUNDARY.split(str(value or ""), maxsplit=1)[0].strip()


def _organization_from_text(value: str) -> str:
    cleaned = re.sub(
        r"^(?:负责|参与|协助|支持|组织|推动|运营|管理|加入|担任|带领)\s*",
        "",
        value.strip(),
    )
    cleaned = re.sub(
        r"^(?:我|本人)?(?:目前|现在)?(?:是|做过|曾在)?\s*"
        r"(?:(?:\d+|[一二两三四五六七八九十]+)\s*段)?\s*",
        "",
        cleaned,
    )
    # OCR often joins a leading period directly to the organization.  Remove
    # only the recognized date span before matching so the date cannot become
    # part of the company/school name.
    cleaned = _FALLBACK_PERIOD.sub(" ", cleaned)
    cleaned = re.sub(r"^[\s\-–—~至到]+", "", cleaned)
    contextual = re.search(
        r"(?:就职于|任职于|供职于|在)\s*"
        r"([\u4e00-\u9fffA-Za-z0-9·.&（）()_-]{1,40}?(?:大学|学院|学校|医院|公司|企业|集团|"
        r"研究院|实验室|中心|部门|协会|学会|学生会|社团|委员会|事务所|银行|"
        r"基金会|工作室|团队))(?=工作|任职|担任|就职|[，,。；;\s])",
        cleaned,
    )
    if contextual:
        return contextual.group(1).strip()
    identity = _identity_prefix(cleaned)
    # Prefer the longest explicit organization prefix.  A first-match regex
    # truncates names such as “北京大学第三医院” at “北京大学”, which then
    # turns the hospital suffix into the candidate's role.
    organization_matches = list(_FALLBACK_ORGANIZATION_END.finditer(identity))
    organization = ""
    if organization_matches:
        organization_end = max(match.end() for match in organization_matches)
        # Legal names often include a parenthetical qualifier whose closing
        # bracket immediately follows the organization suffix.  The suffix
        # matcher stops at ``集团/公司``; retain the literal closing bracket so
        # the public identity is not visibly truncated.
        if (
            organization_end < len(identity)
            and identity[organization_end] in ")）"
            and (
                (identity[organization_end] == ")" and "(" in identity[:organization_end])
                or (identity[organization_end] == "）" and "（" in identity[:organization_end])
            )
        ):
            organization_end += 1
        organization = identity[:organization_end].strip(" \t,，|｜:：-–—")
        organization = _FALLBACK_PERIOD.sub(" ", organization)
        organization = re.sub(r"^[\s\-–—~至到]+", "", organization).strip()
    # “做过两段律所实习” identifies an institution type, not a named
    # employer. Keep the role/duties and ask for the two firm names instead of
    # rendering a fake company called “律所”.
    if organization in {
        "律所", "事务所", "公司", "企业", "医院", "学校", "学院", "机构", "部门", "协会",
        "社团", "组织", "团队", "基地",
    }:
        return ""
    return organization


_COMPACT_ROLE_BASE_WIDTH: tuple[tuple[str, int], ...] = (
    ("实习生", 0), ("志愿者", 0), ("助教", 0), ("负责人", 2), ("工程师", 2),
    ("设计师", 2), ("架构师", 2), ("分析师", 2), ("研究员", 2),
    ("管理岗", 2), ("教师", 2), ("老师", 2), ("医生", 2), ("医师", 2),
    ("经理", 2), ("主管", 2), ("总监", 2), ("主任", 2), ("顾问", 2),
    ("专员", 2), ("助理", 2), ("讲师", 2), ("运营", 2), ("实习", 2),
    ("见习", 2), ("开发", 2), ("测试", 2), ("销售", 2), ("岗", 4),
)
_ROLE_SENIORITY_PREFIX = re.compile(r"(?:高级|资深|初级|首席|区域|大客户|副|总)$")


def _compact_identity_parts(value: str) -> tuple[str, str]:
    """Split one date-leading compact identity without an industry lexicon.

    Legal/institution suffixes are used first.  For brand-only names, the
    fallback consumes only a bounded modifier immediately before a generic
    title suffix (for example 产品+经理 or 售前+顾问), leaving the preceding
    literal text untouched as the organization.
    """

    identity = _identity_prefix(value)
    identity = _FALLBACK_PERIOD.sub(" ", identity)
    identity = re.sub(r"^[\s|｜:：\-–—~至到]+|[\s|｜:：\-–—~至到]+$", "", identity)
    role_narrative = bool(_ROLE_NARRATIVE_PREFIX.match(identity))
    identity = re.sub(
        r"^(?:我|本人)?(?:目前|现在|之前|过去)?(?:一直)?(?:做过|做|从事|担任|任职为)?\s*"
        r"(?:(?:\d+|[一二两三四五六七八九十]+)\s*段)?\s*",
        "",
        identity,
    ).strip()
    if not identity or _FALLBACK_RELATION_LABEL.match(identity):
        return "", ""

    organization = _organization_from_text(identity)
    if organization:
        role = _role_from_text(identity, organization)
        if (
            role_narrative
            and role
            and not _EXPLICIT_ORGANIZATION_END.search(organization)
        ):
            # “做企业软件销售” supplies a domain-qualified title, not a
            # company named “企业软件”.  Brand-like suffixes (科技/软件/电商)
            # are accepted for compact CV headers, but require an explicit
            # organization preposition/label in free-form role narratives.
            narrative_role = _clean_role_title(
                _first_match(_FALLBACK_ROLE, identity),
                explicit=True,
            )
            if narrative_role:
                return "", narrative_role
        return organization, role

    compact = re.sub(r"\s+", "", identity)
    for base, modifier_width in _COMPACT_ROLE_BASE_WIDTH:
        if not compact.endswith(base):
            continue
        prefix = compact[:-len(base)]
        if not prefix:
            return "", base
        modifier = prefix[-modifier_width:] if modifier_width else ""
        organization = prefix[:-len(modifier)] if modifier else prefix
        role = modifier + base
        qualifier = _ROLE_SENIORITY_PREFIX.search(organization)
        if qualifier:
            role = qualifier.group(0) + role
            organization = organization[:qualifier.start()]
        organization = organization.strip(" \t,，|｜:：-–—")
        if not organization:
            return "", _clean_role_title(role, explicit=True)
        if len(re.sub(r"\W+", "", organization)) >= 2:
            return organization, _clean_role_title(role, explicit=True)
    return "", ""


def _fallback_period_from_lines(
    lines: list[str],
    joined: str,
    labels: tuple[str, ...],
) -> str:
    """Recover a date span split into start/end OCR columns."""

    labeled = _labeled_value(joined, labels)
    open_period = re.compile(
        r"^((?:19|20)\d{2}(?:(?:[./-]\d{1,2})|(?:年\d{1,2}月?))?)\s*至\s*$"
    )
    standalone_date = re.compile(
        r"^((?:19|20)\d{2}(?:(?:[./-]\d{1,2})|(?:年\d{1,2}月?))?)$"
    )
    for index, line in enumerate(lines):
        start = open_period.fullmatch(str(line or "").strip())
        if not start:
            continue
        for later in lines[index + 1:]:
            end = standalone_date.fullmatch(str(later or "").strip())
            if end:
                return f"{start.group(1)} 至 {end.group(1)}"
    period = labeled or _first_match(_FALLBACK_PERIOD, joined)
    if not period:
        return ""
    # Portfolio headers often carry both a range and its literal elapsed
    # duration (``2023年3-4月 • 1个月``).  Keep both in the structured period so
    # the duration is neither lost nor left attached to the project name.
    for line_index, line in enumerate(lines):
        position = str(line or "").find(period)
        if position < 0:
            continue
        tail = str(line or "")[position + len(period):]
        duration = _FALLBACK_DURATION_SUFFIX.match(tail)
        if duration:
            return period + duration.group(0)
        if line_index + 1 < len(lines):
            duration_line = _FALLBACK_DURATION_LINE.fullmatch(
                str(lines[line_index + 1] or "").strip()
            )
            if duration_line:
                return f"{period} • {duration_line.group('duration')}"
    return period


def _role_from_text(value: str, organization: str = "") -> str:
    if _FALLBACK_RELATION_LABEL.match(str(value or "").strip()):
        return ""
    labeled = _labeled_value(value, ("岗位", "职位", "角色", "职务"))
    if labeled:
        return _clean_role_title(labeled, explicit=True)
    if organization and organization in value:
        tail = value.split(organization, 1)[1].lstrip(" \t,，|｜:：")
        identity_segment = re.split(
            r"\s+[-–—]\s+|[|｜,，;；]",
            tail,
            maxsplit=1,
        )[0].strip()
        structured_role = _clean_role_title(
            _first_match(_FALLBACK_ROLE, identity_segment),
            explicit=True,
        )
        if structured_role:
            return structured_role
    # Identity normally appears before the first duty clause. Strip the period
    # and already-grounded organization so a greedy role regex cannot absorb
    # them from OCR-compressed headers.
    header = re.split(
        r"[，,。；;]\s*(?=(?:负责|参与|主导|协助|支持|配合|完成|推动|推进|"
        r"组织|设计|开发|构建|实现|制定|管理|运营|分析|研究|撰写|输出|交付|"
        r"维护|优化|搭建|建立|开展|承担|提供|跟进|协调|带领|执行))",
        value,
        maxsplit=1,
    )[0]
    period = _first_match(_FALLBACK_PERIOD, header)
    for identity in (period, organization):
        if identity:
            header = header.replace(identity, " ")
    header = re.sub(r"^[\s|｜:：\-—~至到]+|[\s|｜:：\-—~至到]+$", "", header)
    header = re.sub(
        r"(?:技术研发部|产品技术部|研发部|技术部|产品部|运营部|市场部|销售部|"
        r"职能部门|事业部|校园创业团队)$",
        "",
        header,
    ).strip()
    narrative = re.search(
        r"(?:目前|现在|之前|过去|毕业后)?(?:一直)?"
        r"(?:做过|做|从事|担任|任职为)\s*"
        r"((?:\d+|[一二两三四五六七八九十]+)?\s*段?[^，,。；;]{2,48})$",
        header,
    )
    if narrative:
        role = re.sub(
            r"^(?:\d+|[一二两三四五六七八九十]+)\s*段\s*",
            "",
            narrative.group(1).strip(),
        )
        role = _clean_role_title(role, explicit=True)
        if role:
            return role
    return _clean_role_title(_first_match(_FALLBACK_ROLE, header))


def _clean_target_role(value: str) -> str:
    value = str(value or "").strip()
    instruction_like = bool(
        re.search(
            r"(?:请|帮我|麻烦|需要|想要)?.{0,6}"
            r"(?:优化|修改|改写|生成|调整|完善|整理).{0,8}"
            r"(?:简历|CV|履历)|"
            r"(?:下面|上述|以下)(?:是|的)?\s*(?:JD|岗位描述)|"
            r"(?:根据|按照|结合).{0,8}(?:JD|岗位描述).{0,8}(?:优化|修改|生成)",
            value,
            re.IGNORECASE,
        )
    )
    if instruction_like:
        return ""
    value = re.sub(r"^(?:(?:更)?适合|转向?|偏)\s*", "", value).strip()
    value = re.sub(r"(?:相关)?(?:岗位|方向|工作)$", "", value).strip()
    # Normalize internal taxonomy tokens before they can be copied into the
    # user-facing summary (the outer service normalizes meta later, which was
    # too late for strings such as ``Product PM`` or ``operations``).
    from resume_classifier import normalize_target_role

    normalized = normalize_target_role(value)
    if any(token in normalized.casefold() for token in ("下面jd", "上述jd", "这份简历")):
        return ""
    return normalized


def _fallback_target_role(query_text: str, jd_text: str) -> str:
    extracted = (
        product_logic.extract_target_role(query_text, jd_text)
        if hasattr(product_logic, "extract_target_role") else ""
    )
    extracted = _clean_target_role(extracted)
    invalid = (
        not extracted
        or extracted.casefold() in {"jd", "的jd", "岗位", "目标岗位", "简历"}
        or any(token in extracted.casefold() for token in ("简历", "修改", "优化"))
    )
    if not invalid:
        return extracted
    labeled = re.search(
        r"(?:目标岗位|应聘岗位|求职岗位|岗位)\s*(?:是|为|[:：])\s*"
        r"([^\n，,。；;]{2,40})",
        "\n".join(value for value in (query_text, jd_text) if value),
        re.IGNORECASE,
    )
    if labeled:
        return _clean_target_role(labeled.group(1))
    embedded = re.search(
        r"(?:一个|目标(?:岗位)?\s*[:：]?)\s*"
        r"([A-Za-z0-9+.#/_\-\u4e00-\u9fff]{2,32}?)(?:的)?岗位(?:JD|描述)?",
        query_text,
        re.IGNORECASE,
    )
    if embedded:
        return _clean_target_role(embedded.group(1))
    rewrite = re.search(
        r"(?:优化|调整|修改|改写|适配)(?:成|为)\s*"
        r"([A-Za-z0-9+.#/_\-\u4e00-\u9fff]{2,32}?)(?=岗位|方向|[，,。；;\n]|$)",
        query_text,
        re.IGNORECASE,
    )
    if rewrite:
        return _clean_target_role(rewrite.group(1))
    matches = re.finditer(
        r"(?:投递|应聘|申请|求职|想投|继续投|目标岗位(?:是|为)?)\s*"
        r"([A-Za-z0-9+.#/_\-\u4e00-\u9fff]{2,32}?)(?=岗位|[，,。；;\n]|帮我|$)",
        query_text,
        re.IGNORECASE,
    )
    for match in matches:
        candidate = _clean_target_role(match.group(1))
        if candidate.casefold() not in {"jd", "的jd", "岗位", "目标岗位", "简历"}:
            return candidate
    direction = re.search(
        r"(?:想找(?:一份)?|想做|(?:想)?转(?:到|向)?|偏)\s*"
        r"([^，,。；;]{2,32}?)(?=(?:相关)?(?:的)?(?:岗位|工作|实习|方向|赛道)|的简历|简历|$)",
        query_text,
        re.IGNORECASE,
    )
    if direction:
        candidate = _clean_target_role(direction.group(1))
        if candidate and not any(token in candidate for token in ("简历", "怎么", "如何")):
            return candidate
    domain_direction = re.search(
        r"(?:这是我在|继续(?:往|向)|希望继续(?:往|向)?)\s*"
        r"([^，,。；;]{2,24}?)(?=方向)",
        query_text,
        re.IGNORECASE,
    )
    if domain_direction:
        return _clean_target_role(domain_direction.group(1))
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


def _coalesce_ocr_record_lines(lines: list[str]) -> list[str]:
    """Join wrapped OCR tails before assigning typed record fields."""

    merged: list[str] = []
    for raw in lines:
        line = str(raw or "").strip()
        if not line:
            continue
        previous = merged[-1] if merged else ""
        continuation = bool(
            previous
            and not re.search(r"[。；;!?！？]$", previous)
            and _looks_like_record_body(previous)
            and not _FALLBACK_DURATION_LINE.fullmatch(previous)
            and not re.match(r"^(?:\d{1,3}(?:[、)]|\.(?!\d))|[-*•·▪◦])\s*", line)
            and not _looks_like_record_body(line)
            and not _FALLBACK_PERIOD.fullmatch(line)
            and not _FALLBACK_ORGANIZATION.search(line)
            and len(re.split(r"[|｜\t]", line)) < 2
            and len(line) <= 160
        )
        if continuation:
            merged[-1] = previous + line
        else:
            merged.append(line)
    return merged


def _coalesce_ocr_summary_lines(lines: list[str]) -> list[str]:
    """Rejoin visual line wraps in prose without altering wording."""

    merged: list[str] = []
    for raw in lines:
        line = str(raw or "").strip()
        if not line:
            continue
        if line == "\0":
            if merged and merged[-1] != "\0":
                merged.append(line)
            continue
        if (
            merged
            and merged[-1] != "\0"
            and not re.search(r"[。；;!?！？]$", merged[-1])
            and not re.match(r"^(?:[-*•·▪◦]|\d{1,3}(?:[、)]|\.(?!\d)))\s*", line)
            and not _FALLBACK_PERIOD.fullmatch(line)
        ):
            merged[-1] += line
        else:
            merged.append(line)
    return [value for value in merged if value != "\0"]


def _fallback_record(section: str, lines: list[str]) -> tuple[dict, list[str]]:
    """Parse identities conservatively and keep every unconsumed line."""

    lines = _coalesce_ocr_record_lines(lines)
    joined = " ｜ ".join(line for line in lines if line)
    header_lines = [
        line for line in lines
        if line and (
            not _looks_like_record_body(line)
            or bool(_FALLBACK_PERIOD.search(line))
        )
    ]
    identity_header_lines = list(dict.fromkeys(
        prefix
        for line in header_lines
        if not _FALLBACK_RELATION_LABEL.match(line.strip())
        if (prefix := _identity_prefix(line))
    ))
    header_text = " ｜ ".join(identity_header_lines)
    period = _fallback_period_from_lines(
        lines,
        joined,
        ("时间", "任职时间", "起止时间", "项目时间"),
    )
    organization = _labeled_value(header_text, ("公司", "单位", "组织", "机构", "学校"))
    role = _clean_role_title(
        _labeled_value(header_text, ("岗位", "职位", "角色", "职务")),
        explicit=True,
    )
    if section != "projects" and (not organization or not role):
        structural_identity_lines = [
            line.strip(" \t,，|｜:：-–—")
            for line in identity_header_lines
            if line.strip(" \t,，|｜:：-–—")
            and not _FALLBACK_DATE_TOKEN.fullmatch(line.strip())
            and not _FALLBACK_PERIOD.fullmatch(line.strip())
        ]
        for line_index, line in enumerate(structural_identity_lines):
            if line_index == 0:
                continue
            preceding = structural_identity_lines[line_index - 1]
            # If the dated header already contains a structurally parsed role,
            # let the compact-header parser below keep it.  A later method
            # sentence such as ``采用图论方法，...`` must not replace
            # ``研究助理`` merely because OCR put both on separate rows.
            _, preceding_role = _compact_identity_parts(preceding)
            if preceding_role:
                continue
            if re.match(
                r"^(?:采用|使用|通过|基于|利用|借助|按照|结合)\S+",
                line,
            ):
                continue
            structurally_positioned_role = bool(
                _FALLBACK_PERIOD.search(preceding)
                and 2 <= len(line) <= 48
                and not re.search(r"[:：]", line)
                and not _looks_like_record_body(line)
            )
            if not (_FALLBACK_ROLE.fullmatch(line) or structurally_positioned_role):
                continue
            if _FALLBACK_ROLE.fullmatch(preceding) or _looks_like_record_body(preceding):
                continue
            role = role or _clean_role_title(line, explicit=True)
            organization_source = _FALLBACK_PERIOD.sub(" ", preceding)
            organization_source = re.sub(
                r"[\s|｜,，;；:：\-–—~至到]+$", "", organization_source,
            ).strip()
            organization = (
                organization
                or _organization_from_text(organization_source)
                or organization_source
            )
            break
    if section != "projects" and (not organization or not role):
        for line in identity_header_lines:
            parsed_organization, parsed_role = _compact_identity_parts(line)
            if not organization and parsed_organization:
                organization = parsed_organization
            if not role and parsed_role:
                role = parsed_role
            if organization and role:
                break
    if not organization:
        organization = _organization_from_text(header_text)
    if section == "activities" and not organization and not role:
        for line in identity_header_lines:
            activity_identity = re.fullmatch(
                r"(.{2,40}?)(干事|部员|成员|委员)", line.strip(" ，,。；;|｜")
            )
            if activity_identity:
                organization = _clean_structured_organization(activity_identity.group(1))
                role = activity_identity.group(2)
                break
    if not role and section != "projects":
        # A dated identity row is explicit structure even when its profession
        # is absent from the generic title suffix grammar.  Remove only the
        # already parsed period/organization and keep the literal remainder as
        # the role; this handles titles across industries without a role
        # dictionary (for example pharmacy technician or laboratory analyst).
        for line in identity_header_lines:
            if not _FALLBACK_PERIOD.search(line):
                continue
            candidate = line
            for value in (period, organization):
                if value:
                    candidate = candidate.replace(value, " ")
            candidate = re.sub(
                r"(?:时间|任职时间|起止时间|期间|持续时间)\s*[:：]",
                " ",
                candidate,
            )
            candidate = re.sub(r"[\s|｜,，;；:：\-–—~至到]+", " ", candidate).strip()
            if (
                2 <= len(candidate) <= 48
                and not re.match(
                    r"^(?:负责|参与|主导|协助|支持|配合|完成|推动|推进|组织|协调|"
                    r"执行|开发|构建|实现|制定|分析|撰写|输出|交付|维护|优化|搭建|"
                    r"建立|开展|承担|提供|跟进|带领)\S{3,}",
                    candidate,
                )
                and not _FALLBACK_RELATION_LABEL.match(candidate)
            ):
                role = _clean_role_title(candidate, explicit=True)
                if role:
                    break
    if not role and section != "projects":
        for line in identity_header_lines:
            role = _role_from_text(line, organization)
            if role:
                break
    identity_lines = [
        line.strip(" ，,。；;|｜")
        for line in identity_header_lines
        if line.strip(" ，,。；;|｜")
        and line.strip(" ，,。；;|｜") not in {period, organization, role}
        and not (role and role in line)
        and not _FALLBACK_PERIOD.fullmatch(line.strip())
        and not re.fullmatch(r"[\d./年月\-—~至到今]+", line.strip())
        and not _looks_like_record_body(line)
        and len(line.strip()) <= 40
        and not re.search(r"[。；;!?！？]$", line.strip())
    ]
    if section != "projects" and not organization and role and identity_lines:
        # Brand/team names such as “Wonderlab” need not end in 公司/中心.
        # Their position before an independently recognized role is enough
        # structure to retain the literal name without guessing its type.
        role_line_index = next(
            (index for index, line in enumerate(identity_header_lines) if role in line),
            len(identity_header_lines),
        )
        preceding_identity = next(
            (
                candidate for candidate in identity_lines
                if identity_header_lines.index(candidate) < role_line_index
            ),
            "",
        )
        organization = preceding_identity
    if section != "projects" and organization and not role and identity_lines:
        # Conversely, a short line following a named organization is the
        # source-provided title even when the profession is absent from the
        # generic suffix grammar (律师、施工员、会长, etc.).
        role = _clean_role_title(identity_lines[0], explicit=True)

    consumed = {value for value in (period, organization, role) if value}
    if period:
        literal_period = _first_match(_FALLBACK_PERIOD, period)
        if literal_period:
            consumed.add(literal_period)
        duration_in_period = _FALLBACK_DURATION_SUFFIX.search(period)
        if duration_in_period:
            consumed.add(duration_in_period.group(0).strip(" •·|｜"))
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
        if section != "projects":
            residual = re.sub(
                r"^(?:我|本人)?(?:目前|曾经|曾)?(?:在|于|就职于|任职于|供职于|担任|任职为|作为)\s*",
                "",
                residual,
            )
            residual = re.sub(r"^(?:我|本人)?(?:在|于|担任|任职|作为)+\s*$", "", residual)
        residual = re.sub(r"^[\s|｜,，;；:：\-–—~至到]+|[\s|｜,，;；:：\-–—~至到]+$", "", residual)
        residual = re.sub(r"\s+([，,。；;])", r"\1", residual)
        residual = re.sub(r"([，,。；;])(?:\s*[，,。；;])+", r"\1", residual)
        # Keep a source-provided sentence terminator.  Removing ``。`` made
        # coalesced OCR sentences visibly abrupt and regressed expression
        # quality, while commas/semicolons at field boundaries remain debris.
        residual = residual.strip(" \t,，；;")
        if re.fullmatch(
            r"(?:之前|过去|目前|现在|毕业后)?(?:一直|曾经|曾)?(?:做过?|从事)?"
            r"(?:(?:\d+|[一二两三四五六七八九十]+)\s*段)?",
            residual,
        ):
            continue
        if (
            section != "projects"
            and line in header_lines
            and not _looks_like_record_body(residual)
            and len(residual) <= 24
            and not (
                section == "activities"
                and len(re.sub(r"\W+", "", residual)) >= 6
                and _FALLBACK_PERIOD.search(line)
                and re.search(r"[，,]", line)
            )
        ):
            # Remaining compact header tokens are departments/locations or
            # layout qualifiers, not accomplishment bullets.
            continue
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
    if not name and lines:
        first_line = str(lines[0] or "").strip()
        prefixed_title = re.match(
            r"^(?:研究项目|课程项目|个人项目|开源项目)\s*[-–—:：]\s*(\S.+)$",
            first_line,
            re.IGNORECASE,
        )
        if prefixed_title:
            candidate = prefixed_title.group(1).strip(" ，,。；;|｜")
            if 2 <= len(candidate) <= 100:
                name = candidate
        explicit_title = re.match(r"^[-*•·▪◦]\s*(\S.+)$", first_line)
        if not name and explicit_title:
            candidate = explicit_title.group(1).strip(" ，,。；;|｜")
            if (
                2 <= len(candidate) <= 100
                and not re.match(
                    r"^(?:负责|参与|主导|协助|支持|完成|推动|设计|开发|构建|实现|优化)",
                    candidate,
                )
                and not _FALLBACK_PERIOD.fullmatch(candidate)
            ):
                # Many editable resumes use a bullet glyph for the project name
                # itself.  The source record boundary is stronger evidence than
                # the visual marker, so retain that first line as identity.
                name = candidate
    if not name:
        for line in lines:
            action_name = re.search(
                r"(?:负责|参与|主导|开发|设计|搭建|完成|开展)\s*"
                r"([^，,。；;]{2,40}(?:项目|系统|平台|课题|作品))",
                line,
                re.IGNORECASE,
            )
            if action_name:
                name = action_name.group(1).strip()
                break
    if not name:
        header_candidates = [
            line for line in lines
            if not _looks_like_record_body(line) and line not in consumed
            and not _FALLBACK_PERIOD.fullmatch(line.strip())
            and not re.fullmatch(r"[\d./年月\-–—~至到今\s]+", line.strip())
            and not _FALLBACK_RELATION_LABEL.match(line.strip())
        ]
        if header_candidates:
            candidate = header_candidates[0]
            candidate = candidate.replace(period, " ") if period else candidate
            candidate = _FALLBACK_PERIOD.sub(" ", candidate)
            candidate = _FALLBACK_DURATION_SUFFIX.sub(" ", candidate)
            candidate = candidate.replace(organization, " ") if organization else candidate
            candidate = candidate.replace(role, " ") if role else candidate
            name = re.sub(r"^[\s|｜,，;；:：\-]+|[\s|｜,，;；:：\-]+$", "", candidate)
    raw_name = name
    if raw_name in bullets:
        bullets.remove(raw_name)
    # A relationship label is metadata about a project, not part of its
    # identity.  Multi-column DOCX extraction can join it to a bullet-style
    # project title; strip only the explicit labeled suffix so the same source
    # remains stable across DOCX and native-PDF extraction.
    relation_suffix = re.match(
        r"^(?P<name>.+?)\s*[-–—]\s*"
        r"(?P<relation>(?:指导老师|指导教师|导师|项目导师|论文导师|推荐人|联系人)\s*[:：].+)$",
        name,
        re.IGNORECASE,
    )
    if relation_suffix:
        name = relation_suffix.group("name").strip(" ，,。；;|｜")
        relation = relation_suffix.group("relation").strip()
        if relation and relation not in bullets:
            bullets.insert(0, relation)
    descriptor_suffix = re.match(
        r"^(?P<name>.{2,80}?(?:项目|系统|平台|课题|作品))\s+[-–—]\s+"
        r"(?P<descriptor>[^。；;!?！？]{2,80})$",
        name,
        re.IGNORECASE,
    )
    if descriptor_suffix:
        name = descriptor_suffix.group("name").strip()
        descriptor = descriptor_suffix.group("descriptor").strip()
        if descriptor and descriptor not in bullets:
            bullets.insert(0, descriptor)
    name = re.sub(
        r"^(?:我|本人)?(?:做过|参与|负责|主导|开发|设计|搭建|完成|开展)\s*",
        "",
        name,
    ).strip()
    bullets = [
        bullet
        for bullet in bullets
        if not re.fullmatch(r"(?:研究项目|课程项目|个人项目|开源项目)\s*[-–—:：]?", bullet)
    ]
    if not role:
        for line in identity_header_lines:
            if name and name in line:
                continue
            if section == "projects" and len(re.split(r"[|｜\t]", line)) < 2:
                continue
            candidate_role = _role_from_text(line, organization)
            if candidate_role and candidate_role != name:
                role = candidate_role
                break
    if role:
        bullets = [bullet for bullet in bullets if bullet.strip() != role]
    if name:
        cleaned_project_bullets: list[str] = []
        for bullet in bullets:
            residual = bullet
            if name in residual:
                residual = residual.replace(name, " ")
                residual = re.sub(
                    r"^(?:我|本人)?(?:参与|负责|主导|完成|开发|设计|搭建|开展)\s*",
                    "",
                    residual,
                )
                residual = re.sub(r"^(?:并|及|和|与)\s*", "", residual)
                residual = residual.strip(" ，,。；;|｜")
            if len(re.sub(r"\W+", "", residual)) >= 2:
                cleaned_project_bullets.append(residual)
        bullets = list(dict.fromkeys(cleaned_project_bullets))
    if section == "projects" and bullets:
        # DOCX table extraction can split one technology row into a labeled
        # head followed by several bullet cells.  Rejoin only this explicit
        # structural list; do not turn arbitrary short prose into skills.
        merged_project_bullets: list[str] = []
        technology_index: int | None = None
        for bullet in bullets:
            value = str(bullet or "").strip()
            if re.match(r"^(?:技术栈|技术)\s*[:：]\s*\S", value):
                merged_project_bullets.append(value)
                technology_index = len(merged_project_bullets) - 1
                continue
            technology_tail = bool(
                technology_index is not None
                and 1 <= len(value) <= 50
                and not _looks_like_record_body(value)
                and not _FALLBACK_PERIOD.search(value)
                and not re.search(r"[。；;!?！？]", value)
            )
            if technology_tail:
                merged_project_bullets[technology_index] += f" • {value}"
                continue
            merged_project_bullets.append(value)
            technology_index = None
        bullets = merged_project_bullets
    return {
        "name": name,
        "organization": organization,
        "role": role,
        "period": period,
        "bullets": bullets,
    }, []


def _fallback_education(lines: list[str]) -> tuple[dict, list[str]]:
    source_lines = list(lines)
    anonymized_school = any(
        re.search(r"\[(?:学校|大学|院校)\]", str(line or ""), re.IGNORECASE)
        for line in source_lines
    )
    cleaned_lines = []
    for line in lines:
        cleaned = _FALLBACK_PLACEHOLDER_TOKEN.sub(" ", str(line or ""))
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t,，、:：;；|｜/\\-—_•·")
        if cleaned:
            cleaned_lines.append(cleaned)
    lines = cleaned_lines
    joined = re.sub(
        r"^(?:教育经历|教育背景|学历信息)\s*[:：]\s*",
        "",
        " ｜ ".join(lines),
    )
    school = _labeled_value(joined, ("学校", "院校")) or _organization_from_text(joined)
    if anonymized_school:
        # A later credential issuer (for example ``MBA协会认证``) is not a
        # replacement for an explicitly anonymized school field.
        school = ""
    degree = _labeled_value(joined, ("学历", "学位")) or _first_match(_FALLBACK_DEGREE, joined)
    if school and degree:
        normalized_school = re.sub(
            r"^\d{1,3}(?:[.、)]\s*)?", "", school,
        ).strip(" \t,，|｜:：-–—")
        if normalized_school and normalized_school in degree:
            school = ""
    major = _labeled_value(joined, ("专业",))
    if not major:
        degree_major_match = re.search(
            r"(?:^|[|｜，,。；;\s])([^|｜，,。；;]{2,30}?)"
            r"(?=(?:副?学士|硕士|博士)(?:学位|研究生)?(?:毕业|在读)?(?:$|[，,。；;|｜\s]))",
            joined,
        )
        major_match = re.search(
            r"(?:我是|就读于?|毕业于?|攻读)?\s*([^|｜，,。；;]{2,40}?)\s*专业(?:毕业|在读|学生)?",
            joined,
        )
        direction_match = re.search(
            r"(?:方向偏|研究方向(?:是|为)?|专业方向(?:是|为)?)\s*([^|｜，,。；;]{2,40})",
            joined,
        )
        if degree_major_match:
            major = degree_major_match.group(1).strip(" \t,，、:：;；|｜/\\-—_•·.")
        elif major_match:
            major = major_match.group(1).strip()
        elif direction_match:
            major = direction_match.group(1).strip()
    period = _fallback_period_from_lines(
        lines,
        joined,
        ("时间", "就读时间", "起止时间"),
    )
    parts = [
        part.strip()
        for part in re.split(r"[|｜\t，,;；]+|\s{2,}", joined)
        if part.strip()
    ]
    if not major and school and degree:
        for line in lines:
            if school not in line or degree not in line:
                continue
            source_line = re.sub(
                r"^(?:教育经历|教育背景|学历信息)\s*[:：]\s*",
                "",
                line,
            )
            residual = source_line.replace(school, " ").replace(degree, " ")
            if period:
                residual = residual.replace(period, " ")
            residual = re.sub(r"(?:专业|学历|学位)\s*[:：]?", " ", residual)
            residual = re.sub(r"[\s|｜,，;；:：\-—~至到]+", "", residual)
            if 2 <= len(residual) <= 40:
                major = residual
                break
    if (
        not school
        and len(parts) >= 2
        and not _FALLBACK_PERIOD.fullmatch(parts[0])
        and not _FALLBACK_DEGREE.fullmatch(parts[0])
        and _STRUCTURED_ORG_SUFFIX.search(parts[0])
    ):
        # Delimited education headers conventionally put the institution first;
        # this also preserves anonymized names such as “学校0”.
        school = parts[0]
    if not major:
        for value in parts:
            if not value or any(item and item in value for item in (school, degree, period)):
                continue
            if _FALLBACK_DATE_TOKEN.fullmatch(value):
                continue
            if _STRUCTURED_ORG_SUFFIX.search(value):
                continue
            value = re.sub(r"(?:专业|学历|学位|时间|就读时间|起止时间)\s*[:：]", "", value).strip()
            if value and len(value) <= 40 and not _looks_like_record_body(value):
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
    if (
        re.fullmatch(r"\d+(?:\.\d+)?", major)
        or
        re.fullmatch(r"\d+(?:\.\d+)?\s*(?:%|/\s*\d+(?:\.\d+)?)", major)
        or re.fullmatch(r"[\s.,，。;；:：|｜/\\\-—_年月日]+", major)
    ):
        major = ""
    leftovers = []
    for line in lines:
        if line.strip(" \t-•·") in {"", "其他", "其它", "补充信息"}:
            continue
        residual = line
        for value in (school, degree, major, period):
            if value:
                residual = residual.replace(value, " ")
        residual = re.sub(r"[\s|｜,，;；:：\-—~至到]+", "", residual)
        represented_narrative = bool(
            (major and major in line and re.search(r"(?:专业(?:毕业|在读)|方向偏|研究方向)", line))
            or (school and school in line and re.search(r"(?:就读|毕业|学生|在读)", line))
        )
        if len(residual) >= 2 and not represented_narrative and not re.fullmatch(
            r"(?:我是|本人|目前是|现在是)?(?:研[一二三四]|大[一二三四]|学生|在读|毕业的?)+",
            residual,
        ):
            leftovers.append(line)
    return {"school": school, "degree": degree, "major": major, "period": period}, leftovers


def _fallback_project_examples(lines: list[str]) -> list[dict]:
    """Expand an explicit list of project names without inventing details."""

    if not any(re.search(
        r"(?:做过|完成过|参与过)[^，。；;]{0,24}(?:一些|多个|若干)[^，。；;]{0,8}项目",
        line,
    ) for line in lines):
        return []
    for line in lines:
        match = re.match(r"^(?:比如|例如)\s*(.+)$", line.strip())
        if not match:
            continue
        names = [
            value.strip(" ，,。；;、")
            for value in re.split(r"[、]|以及|和", match.group(1))
            if value.strip(" ，,。；;、")
        ]
        if len(names) < 2 or any(len(value) > 40 for value in names):
            continue
        if not all(re.search(r"(?:项目|系统|平台|页面|APP|小程序|课题|作品)$", value, re.IGNORECASE) for value in names):
            continue
        return [
            {"name": value, "organization": "", "role": "", "period": "", "bullets": []}
            for value in names
        ]
    return []


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
        education_extras = [
            value for value in leftovers
            if not _FALLBACK_LABELED_SKILL.fullmatch(value.strip())
        ]
        education_scores = [
            value for value in education_extras
            if re.fullmatch(
                r"\d+(?:\.\d+)?\s*(?:%|/\s*\d+(?:\.\d+)?)",
                value.strip(),
            )
        ]
        if education_scores:
            additional.setdefault("教育成绩", []).extend(education_scores)
        education_other = [
            value for value in education_extras if value not in education_scores
        ]
        if education_other:
            additional.setdefault("补充信息", []).extend(education_other)

    records: dict[str, list[dict]] = {name: [] for name in ("experience", "research", "activities", "projects")}
    for section in records:
        for lines in _record_groups(source, section):
            if section == "projects":
                example_records = _fallback_project_examples(lines)
                if example_records:
                    records[section].extend(example_records)
                    continue
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

    def add_skill_values(raw_values: str, category: str) -> None:
        """Parse labeled skill lists even when OCR assigns the wrong section."""

        raw_values = str(raw_values or "").strip()
        explicit_proficiency = re.fullmatch(
            r"(.{2,80}?)\s*[·•]\s*(?:精通|熟练(?:使用|掌握)?|熟悉|了解|掌握)\s*",
            raw_values,
        )
        if explicit_proficiency:
            name = explicit_proficiency.group(1).strip(" \t-•·")
            if name:
                skill_items.append({"name": name, "category": category})
            return
        raw_values = re.sub(
            r"^(?:我)?(?:学过|自学过?|会用|会使用|使用过|掌握|熟悉|了解)\s*",
            "",
            raw_values,
        )
        raw_values = re.sub(r"^转行(?:做|从事)?\s*", "", raw_values)
        raw_values = re.sub(r"^(?:包括|例如|如)\s*", "", raw_values).strip()

        # A language proficiency statement is one semantic item, not a
        # comma-separated tool list. Keep e.g. ``西班牙语：流利，书面和口语``
        # intact instead of producing three fragments.
        natural_language_statement = bool(
            re.match(
                r"^[^:：]{1,24}[:：]\s*(?:母语|精通|流利|熟练|中级|初级|"
                r"日常会话|工作交流|书面|口语|native|fluent|advanced|"
                r"intermediate|basic|conversational)",
                raw_values,
                re.IGNORECASE,
            )
        )
        values = (
            [raw_values]
            if natural_language_statement
            else re.split(r"[、，,；;|｜]+|\s+/\s+", raw_values)
        )
        expanded_values: list[str] = []
        for value in values:
            value = re.sub(r"^(?:包括|例如|如)\s*", "", value.strip())
            # Chinese conjunctions are ordinary grammar in phrases such as
            # ``战略规划和实施``. Split only around a nearby ASCII
            # product/tool token, as in ``SAP和JD Edwards`` or
            # ``北大法宝和Alpha系统``.
            conjunctions = list(re.finditer(r"\s*(?:和|及|与)\s*", value))
            split_cursor = 0
            for index, conjunction in enumerate(conjunctions):
                left_start = conjunctions[index - 1].end() if index else 0
                right_end = (
                    conjunctions[index + 1].start()
                    if index + 1 < len(conjunctions) else len(value)
                )
                left = value[left_start:conjunction.start()]
                right = value[conjunction.end():right_end]
                if re.search(r"[A-Za-z0-9.+#]", left + right):
                    expanded_values.append(value[split_cursor:conjunction.start()])
                    split_cursor = conjunction.end()
            expanded_values.append(value[split_cursor:])

        for item in expanded_values:
            item = item.strip(" \t-•·")
            if re.match(r"^(?:能|能够|可以|具备|善于)", item):
                continue
            item = re.sub(
                r"^(?:编程语言|机器学习|深度学习|开发工具|技术栈)\s*[:：]\s*",
                "",
                item,
            ).strip()
            if re.search(r"(?:算法|神经网络)\s*[:：]?\s*如\s*", item):
                item = re.split(r"(?:算法|神经网络)\s*[:：]?\s*如\s*", item, maxsplit=1)[-1]
            item = re.sub(
                r"^(?:精通|熟练(?:使用|掌握)?|熟悉|了解|掌握)\s*",
                "",
                item,
            ).strip()
            item = re.sub(r"^各种", "", item).strip()
            item = re.sub(
                r"等(?:相关)?(?:知识|技能|技术|工具|框架)?$",
                "",
                item,
            ).strip()
            item = re.sub(
                r"(?:精通|熟练(?:使用|掌握)?|熟悉|了解|掌握)$",
                "",
                item,
            ).strip()
            if (
                2 <= len(item) <= 50
                and not re.fullmatch(r"(?:扎实|熟练|精通|熟悉|掌握|了解|良好|优秀)", item)
                and not re.fullmatch(r"(?:主流)?(?:框架|平台|工具|技能|能力|知识|算法)", item)
                and not re.search(r"^主流.*平台$|能力$", item)
                and (
                    not _looks_like_record_body(item)
                    or len(item) <= 16
                )
                and not re.search(r"[。；;]", item)
            ):
                skill_items.append({
                    "name": item,
                    "category": (
                        "natural_language"
                        if natural_language_statement else category
                    ),
                })

    source_candidates = candidate_blocks(source)
    summary_open = False
    for source_block in source_candidates:
        value = source_block.text.strip()
        section = source_block.section_hint or ""
        if _is_section_heading(value):
            if summary_open:
                summary_lines.append("\0")
            summary_open = False
            continue
        if section == "summary":
            summary_lines.append(value)
            summary_open = True
            continue
        if section:
            if summary_open:
                summary_lines.append("\0")
            summary_open = False
            continue
        if (
            _looks_like_record_body(value)
            or _FALLBACK_PERIOD.fullmatch(value)
            or (phone_match and phone_match.group(0) in value)
            or (email_match and email_match.group(0) in value)
        ):
            if summary_open:
                summary_lines.append("\0")
            summary_open = False
            continue
        signal = bool(re.match(
            r"^(?:[-*•·▪◦]\s*)?(?:本人|我|具备|拥有|曾|有|熟悉|掌握|精通|"
            r"从事|就职|实习|近?\d+\s*(?:年|个月|月)|"
            r"[^，,。；;]{1,20}(?:大学|学院)(?:本科|硕士|博士)?(?:在读|毕业)?)",
            value,
        ))
        continuation = bool(
            summary_open
            and summary_lines
            and not re.search(r"[。；;!?！？]$", summary_lines[-1])
            and not re.match(r"^(?:[-*•·▪◦]|\d{1,3}(?:[、)]|\.(?!\d)))", value)
        )
        if len(value) >= 8 and (signal or continuation):
            summary_lines.append(value)
            summary_open = True
        else:
            if summary_open:
                summary_lines.append("\0")
            summary_open = False

    unclassified: list[str] = []
    unstructured_current: tuple[str, int] | None = None
    unstructured_last_number: int | None = None
    for block in source_candidates:
        if _is_section_heading(block.text):
            unstructured_current = None
            unstructured_last_number = None
            continue
        section = block.section_hint or ""
        value = block.text.strip()
        item_number_match = re.match(r"^(\d{1,3})(?:[、)]|\.(?!\d))\s*", value)
        item_number = int(item_number_match.group(1)) if item_number_match else None
        labeled_skill = _FALLBACK_LABELED_SKILL.fullmatch(value)
        if labeled_skill:
            add_skill_values(
                labeled_skill.group("value"),
                (
                    "natural_language"
                    if labeled_skill.group("label") in {"语言", "语言能力"}
                    else "other"
                ),
            )
            unstructured_current = None
            unstructured_last_number = None
            continue
        if section == "summary":
            unstructured_current = None
            unstructured_last_number = None
        elif section == "skills":
            value = value.lstrip(" \t-•·:：")
            value = re.sub(
                r"^(?:专业技能|技能清单|技能|技术栈|工具|语言能力)\s*[:：]?\s*",
                "",
                value,
            )
            # OCR may put the heading and its value in separate blocks, for
            # example ``语言能力`` followed by ``：CET-4``.
            value = value.lstrip(" \t-•·:：")
            add_skill_values(value, "other")
            unstructured_current = None
            unstructured_last_number = None
        elif section in {
            "hobbies", "coursework", "products", "highlights", "target", "profile",
        }:
            title = {
                "hobbies": "兴趣爱好",
                "coursework": "相关课程",
                "products": "产品与解决方案",
                "highlights": "经历亮点",
                "target": "求职目标",
                "profile": "个人概况",
            }[section]
            cleaned = value.strip()
            cleaned = re.sub(r"^[-*•·▪◦]\s*", "", cleaned)
            label_aliases = {
                "hobbies": ("兴趣爱好", "个人爱好"),
                "coursework": ("相关课程", "主修课程"),
                "products": ("产品", "产品与解决方案", "产品专业知识"),
                "highlights": ("经历亮点",),
                "target": (
                    "职业目标", "求职目标", "期望职位", "professional direction",
                    "career objective",
                ),
                "profile": (
                    "职业级别", "工作年限", "就业类型", "工作时间", "行业",
                    "career level", "years of experience", "employment type",
                    "work schedule", "industry",
                ),
            }[section]
            cleaned = re.sub(
                rf"^(?:{'|'.join(re.escape(item) for item in label_aliases)})\s*[:：]\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            ).strip()
            if cleaned and not re.fullmatch(r"[-•·]?\s*[A-Za-z0-9._-]{3,}", cleaned):
                additional.setdefault(title, []).append(cleaned)
            unstructured_current = None
            unstructured_last_number = None
        elif section == "references":
            # Availability of references belongs in the explanatory response,
            # not as a candidate accomplishment or an anonymous work record.
            unstructured_current = None
            unstructured_last_number = None
        elif section in scalar_sections:
            cleaned = re.sub(r"^[^:：]{1,12}[:：]\s*", "", value).strip()
            if cleaned:
                scalars[scalar_sections[section]].append(cleaned)
            unstructured_current = None
            unstructured_last_number = None
        elif section not in {"education", "experience", "research", "activities", "projects"}:
            for award_match in re.finditer(
                r"((?:曾)?参加[^；;。]{2,60}?(?:竞赛|比赛)[^；;。]{0,32}?"
                r"(?:获得|荣获)[^；;。]{0,16}?(?:一等奖|二等奖|三等奖|金奖|银奖|铜奖))",
                value,
            ):
                scalars["awards"].append(award_match.group(1).strip(" ，,。；;"))
            if re.search(r"(?:奖学金|一等奖|二等奖|三等奖|优秀学生干部|荣誉称号|获奖)$", value):
                scalars["awards"].append(value)
                unstructured_current = None
                unstructured_last_number = None
                continue
            if re.search(r"(?:证书|资格证|执业证|职业资格|认证|执照)$", value):
                scalars["certifications"].append(value)
                unstructured_current = None
                unstructured_last_number = None
                continue
            if (
                (phone_match and phone_match.group(0) in value)
                or (email_match and email_match.group(0) in value)
            ):
                # A bullet marker on a contact row is visual decoration, not a
                # professional action. Extract it into meta before body routing.
                unstructured_current = None
                unstructured_last_number = None
                continue
            if re.search(r"(?:求职意向|目标岗位|应聘岗位)\s*[:：]", value):
                unstructured_current = None
                unstructured_last_number = None
                continue
            if re.fullmatch(r"[-•·]?\s*[A-Za-z][A-Za-z0-9._-]{2,30}", value):
                # Bare social handles frequently sit beside phone/email in a
                # multi-column header. Without a skill heading or action they
                # must not become a one-line work experience.
                unstructured_current = None
                unstructured_last_number = None
                continue
            if re.match(
                r"^[-*•·▪◦]\s*(?:本人|我|具备|拥有|曾担任|有|熟悉|掌握|精通|"
                r"近?\d+\s*(?:年|个月|月))",
                value,
            ):
                # A résumé summary bullet remains summary evidence even when
                # de-columnization places it below skills or contact details.
                unstructured_current = None
                unstructured_last_number = None
                continue
            profile_domain = re.search(
                r"我(?:是)?(?:做|从事|负责)\s*([^，,。；;]{2,30}?)(?:的|方向|工作|$)",
                value,
            )
            if profile_domain:
                domain = profile_domain.group(1).strip()
                if domain:
                    skill_items.append({"name": domain, "category": "domain"})
                    unstructured_current = None
                    unstructured_last_number = None
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
                r"(?:参与|负责|主导|完成)\s*([^，。；;]{2,36}(?:APP|系统|平台|项目|课题|产品))"
                r"(?:的|设计|开发|建设|研究|$)",
                value,
                re.IGNORECASE,
            )

            if organization and degree:
                record, leftovers = _fallback_education([value])
                education.append(record)
                if leftovers:
                    additional.setdefault("补充信息", []).extend(leftovers)
                unstructured_current = None
                unstructured_last_number = None
                continue
            if explicit_work or (organization and period and compact_role):
                record, _leftovers = _fallback_record("experience", [value])
                records["experience"].append(record)
                unstructured_current = ("experience", len(records["experience"]) - 1)
                unstructured_last_number = item_number
                continue
            if activity_hint and not _looks_like_record_body(value):
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
                        unstructured_last_number = item_number or unstructured_last_number
                        continue
                if (
                    unstructured_current
                    and unstructured_current[0] == target_section
                    and organization
                    and records[target_section][unstructured_current[1]].get("organization") == organization
                ):
                    records[target_section][unstructured_current[1]]["bullets"].append(value)
                else:
                    body_only = _looks_like_record_body(value)
                    records[target_section].append({
                        "organization": "" if body_only else organization,
                        "role": "" if body_only else _labeled_value(value, ("岗位", "职位", "角色", "职务")),
                        "period": "" if body_only else _first_match(_FALLBACK_PERIOD, value),
                        "bullets": [value],
                    })
                    unstructured_current = (target_section, len(records[target_section]) - 1)
                unstructured_last_number = item_number
                continue
            if project_match:
                name = project_match.group(1).strip()
                if (
                    unstructured_current
                    and unstructured_current[0] == "projects"
                    and re.search(r"^(?:撰写|完成|优化|设计|开发|建设|维护|测试)", name)
                ):
                    records["projects"][unstructured_current[1]]["bullets"].append(value)
                    unstructured_last_number = item_number or unstructured_last_number
                    continue
                project_residual = value.replace(name, " ")
                project_residual = re.sub(
                    r"^(?:我|本人)?(?:参与|负责|主导|完成)\s*",
                    "",
                    project_residual,
                ).strip(" ，,。；;|｜")
                records["projects"].append({
                    "name": name,
                    "organization": "",
                    "role": "",
                    "period": _first_match(_FALLBACK_PERIOD, value),
                    "bullets": [project_residual] if len(project_residual) >= 2 else [],
                })
                unstructured_current = ("projects", len(records["projects"]) - 1)
                unstructured_last_number = item_number
                continue
            if (
                len(value) <= 60
                and not _looks_like_record_body(value)
                and re.search(
                    r"(?:项目|系统|平台|小程序|APP|课题|作品)$",
                    value,
                    re.IGNORECASE,
                )
            ):
                records["projects"].append({
                    "name": value,
                    "organization": "",
                    "role": "",
                    "period": _first_match(_FALLBACK_PERIOD, value),
                    "bullets": [],
                })
                unstructured_current = ("projects", len(records["projects"]) - 1)
                unstructured_last_number = item_number
                continue
            if _looks_like_record_body(value):
                if unstructured_current is not None:
                    current_section, current_index = unstructured_current
                    current_record = records[current_section][current_index]
                    has_identity = any(str(current_record.get(key, "") or "").strip() for key in (
                        "organization", "role", "name", "institution", "topic", "period",
                    ))
                    if (
                        item_number is not None
                        and unstructured_last_number is not None
                        and item_number <= unstructured_last_number
                        and current_record.get("bullets")
                    ):
                        # A numbering restart in a de-columnized OCR stream is
                        # the strongest available boundary between duty lists.
                        target_section = current_section if not has_identity else "experience"
                        records[target_section].append({
                            "organization": "",
                            "role": "",
                            "period": "",
                            "bullets": [value],
                        })
                        unstructured_current = (target_section, len(records[target_section]) - 1)
                    else:
                        current_record["bullets"].append(value)
                else:
                    if not activity_hint and item_number is None:
                        # An action with no source-side record owner is useful
                        # evidence, but calling it a separate job manufactures
                        # employment structure.  Keep it in a neutral public
                        # highlights section; numbered OCR lists retain the
                        # existing reattachment path below.
                        additional.setdefault("经历亮点", []).append(value)
                    else:
                        target_section = "activities" if activity_hint else "experience"
                        records[target_section].append({
                            "organization": "",
                            "role": "",
                            "period": "",
                            "bullets": [value],
                        })
                        unstructured_current = (
                            target_section,
                            len(records[target_section]) - 1,
                        )
                unstructured_last_number = item_number or unstructured_last_number
                continue
            if unstructured_current is not None:
                # Contact rows, summaries and a following record header must
                # not leak into the preceding experience merely because OCR
                # removed the section headings.
                unstructured_current = None
                unstructured_last_number = None
            if value.strip(" \t-•·") in _LAYOUT_RESET_HEADINGS:
                continue
            if not any(token and token in value for token in (
                name_match.group(1) if name_match else "",
                phone_match.group(0) if phone_match else "",
                email_match.group(0) if email_match else "",
            )):
                unclassified.append(value)
        else:
            # Typed records were already parsed through their record IDs.
            unstructured_current = None
            unstructured_last_number = None

    # Multi-column resume extractors often emit all identity rows first and
    # the aligned numbered duty lists later. Reattach those lists in source
    # order only when both sides are explicit: an identity-bearing empty/weak
    # record and an anonymous list beginning with an item number.
    anonymous_numbered = [
        record for record in records["experience"]
        if not any(str(record.get(key, "") or "").strip() for key in (
            "organization", "role", "period",
        ))
        and record.get("bullets")
        and any(
            re.match(r"^\d{1,3}(?:[、)]|\.(?!\d))\s*", str(value).strip())
            for value in record["bullets"]
        )
    ]
    identity_candidates: list[tuple[str, dict]] = []
    for candidate_section in ("experience", "projects", "activities"):
        identity_fields = (
            ("organization", "role", "period")
            if candidate_section != "projects"
            else ("name", "organization", "role", "period")
        )
        for record in records[candidate_section]:
            if not any(str(record.get(key, "") or "").strip() for key in identity_fields):
                continue
            existing = [str(value).strip() for value in record.get("bullets", []) if str(value).strip()]
            weak_existing = not existing
            if weak_existing:
                identity_candidates.append((candidate_section, record))

    routed_ids: set[int] = set()
    for anonymous, (_candidate_section, destination) in zip(
        anonymous_numbered, identity_candidates,
    ):
        existing = [
            str(value).strip() for value in destination.get("bullets", []) if str(value).strip()
        ]
        strong_existing = [
            value for value in existing
            if _looks_like_record_body(value) or len(value) > 32
        ]
        destination["bullets"] = list(dict.fromkeys(
            strong_existing
            + [str(value).strip() for value in anonymous.get("bullets", []) if str(value).strip()]
        ))
        routed_ids.add(id(anonymous))
    if routed_ids:
        records["experience"] = [
            record for record in records["experience"] if id(record) not in routed_ids
        ]

    # A late OCR column can contain a missing numbered continuation after the
    # dates/contacts column. Only groups starting above item 1 are eligible;
    # a fresh list beginning at 1 remains an independent anonymous record.
    late_continuations: list[dict] = []
    numbered_destinations = [
        record for record in records["experience"]
        if any(str(record.get(key, "") or "").strip() for key in ("organization", "role"))
        and any(
            re.match(r"^\d{1,3}(?:[、)]|\.(?!\d))\s*", str(value).strip())
            for value in record.get("bullets", [])
        )
    ]
    for record in records["experience"]:
        if any(str(record.get(key, "") or "").strip() for key in ("organization", "role", "period")):
            continue
        bullets = [str(value).strip() for value in record.get("bullets", []) if str(value).strip()]
        first_number = re.match(
            r"^(\d{1,3})(?:[、)]|\.(?!\d))\s*", bullets[0]
        ) if bullets else None
        if first_number and int(first_number.group(1)) > 1 and numbered_destinations:
            destination = numbered_destinations[0]
            destination["bullets"] = list(dict.fromkeys(
                list(destination.get("bullets", [])) + bullets
            ))
            late_continuations.append(record)
    if late_continuations:
        late_ids = {id(record) for record in late_continuations}
        records["experience"] = [
            record for record in records["experience"] if id(record) not in late_ids
        ]

    # A final OCR column may contain item 1 after item 2 was already routed to
    # its identity row. Attach only when exactly one destination has the
    # adjacent numbered suffix and no copy of the incoming numbers.
    leading_complements: list[dict] = []
    for record in records["experience"]:
        if any(str(record.get(key, "") or "").strip() for key in (
            "organization", "role", "period",
        )):
            continue
        bullets = [str(value).strip() for value in record.get("bullets", []) if str(value).strip()]
        incoming_numbers = {
            int(match.group(1))
            for value in bullets
            if (match := re.match(r"^(\d{1,3})(?:[、)]|\.(?!\d))\s*", value))
        }
        if not incoming_numbers or min(incoming_numbers) != 1:
            continue
        destinations: list[dict] = []
        for destination in numbered_destinations:
            destination_numbers = {
                int(match.group(1))
                for value in destination.get("bullets", [])
                if (match := re.match(r"^(\d{1,3})(?:[、)]|\.(?!\d))\s*", str(value).strip()))
            }
            if (
                destination_numbers
                and incoming_numbers.isdisjoint(destination_numbers)
                and max(incoming_numbers) + 1 == min(destination_numbers)
            ):
                destinations.append(destination)
        if len(destinations) == 1:
            destination = destinations[0]
            destination["bullets"] = list(dict.fromkeys(
                bullets + list(destination.get("bullets", []))
            ))
            leading_complements.append(record)
    if leading_complements:
        complement_ids = {id(record) for record in leading_complements}
        records["experience"] = [
            record for record in records["experience"] if id(record) not in complement_ids
        ]

    if unclassified:
        # These blocks have already passed query/resume fact eligibility.  A
        # neutral public section retains them without exposing an internal
        # parser label or pretending that their semantic type is known.
        additional["补充信息"] = list(dict.fromkeys(unclassified))

    # A credential can be embedded in an education row (for example an MBA
    # accreditation after the degree).  Education has no bullet field, so
    # retain the exact credential atom in the public certification section
    # instead of dropping it with parser scratch text.
    for fact in source.fact_units:
        value = str(fact.verbatim_text or "").strip(" \t-•·，,。；;")
        if (
            fact.fact_eligible
            and fact.source_type != "jd"
            and fact.section_hint == "education"
            and "credential" in fact.dimensions
            and value
            and value not in scalars["certifications"]
        ):
            scalars["certifications"].append(value)

    return CanonicalResume.model_validate({
        "meta": {
            "name": (name_match.group(1) if name_match else meta.get("name", "")),
            "phone": (phone_match.group(0) if phone_match else meta.get("phone", "")),
            "email": (email_match.group(0) if email_match else meta.get("email", "")),
            "target_role": target_role or meta.get("target_role", ""),
            "work_experience": seniority_match.group(0) if seniority_match else "",
        },
        "summary": "。".join(
            item.strip("。")
            for item in _coalesce_ocr_summary_lines(summary_lines)
            if len(item.strip("。")) >= 20
            and not _INCOMPLETE_TEXT_TAIL.search(item.strip("。；; "))
        ),
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
    additional = data.get("additional_sections")
    if isinstance(additional, dict):
        data["additional_sections"] = {
            str(title).strip(): values
            for title, values in additional.items()
            if str(title).strip()
            and not _INTERNAL_ADDITIONAL_SECTION.fullmatch(str(title).strip())
        }
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
_FACT_COMPILER_FASTPATH_MIN_BLOCKS = 72
_FACT_COMPILER_FASTPATH_MIN_ATOMIC_RECALL = 0.88
_QUERY_FACT_COMPILER_FASTPATH_MIN_BLOCKS = 24
_QUERY_FACT_COMPILER_FASTPATH_MIN_ATOMIC_RECALL = 0.88


def _fallback_is_structurally_safe(
    resume: CanonicalResume,
    *,
    allow_period_only_experience: bool = False,
    allow_missing_education_school: bool = False,
    allow_partial_record_identity: bool = False,
) -> bool:
    """Reject high-coverage fallbacks whose fields are only lexical matches."""

    required_fields = {
        "experience": ("organization", "role"),
        "research": ("institution", "topic"),
        "activities": ("organization", "role"),
        "projects": ("name",),
    }
    for education in resume.education:
        if (
            not str(education.school or "").strip()
            and not allow_missing_education_school
        ):
            return False
        if not any(str(value or "").strip() for value in (education.degree, education.major)):
            return False
    for section, fields in required_fields.items():
        for record in getattr(resume, section):
            identities = [
                str(getattr(record, field, "") or "").strip()
                for field in fields
            ]
            if not all(identities):
                # OCR sometimes keeps a distinct employment period and all of
                # its duties while dropping the employer/title line.  Keeping
                # that row with blank identity fields is safer than either
                # discarding the supplied history or inheriting the preceding
                # employer.  Require a grounded period and real bullets so an
                # arbitrary prose fragment still cannot pass the fast path.
                period_only_experience = bool(
                    section == "experience"
                    and allow_period_only_experience
                    and not any(identities)
                    and str(record.period or "").strip()
                    and record.bullets
                )
                source_partial_identity = bool(
                    allow_partial_record_identity
                    and any(identities)
                    and str(record.period or "").strip()
                    and record.bullets
                )
                anonymous_source_project = bool(
                    section == "projects"
                    and not any(identities)
                    and record.bullets
                )
                if not (
                    period_only_experience
                    or source_partial_identity
                    or anonymous_source_project
                ):
                    return False
            if any(_looks_like_record_body(value) for value in identities if value):
                return False
            if not record.bullets:
                return False
    return not any(
        "待整理" in str(title)
        for title in resume.additional_sections
    )


@dataclass(frozen=True)
class _RecoveryStats:
    filled_fields: int = 0
    appended_bullets: int = 0
    expanded_bullets: int = 0
    appended_records: int = 0
    appended_values: int = 0

    @property
    def total(self) -> int:
        return (
            self.filled_fields
            + self.appended_bullets
            + self.expanded_bullets
            + self.appended_records
            + self.appended_values
        )


def _identity_value(value: str) -> str:
    return re.sub(
        r"[^\w\u4e00-\u9fff]+",
        "",
        unicodedata.normalize("NFKC", str(value or "")).casefold(),
    )


def _resume_claim_fingerprints(resume: CanonicalResume) -> set[tuple[str, str]]:
    """Fingerprint candidate claims so completeness repair cannot delete any."""

    claims: set[tuple[str, str]] = set()

    def add(section: str, value) -> None:
        normalized = _identity_value(value)
        if normalized:
            claims.add((section, normalized))

    for field in ("name", "phone", "email", "work_experience"):
        add(f"meta.{field}", getattr(resume.meta, field))
    # Summary is a deterministic projection of the structured claims below;
    # it may legitimately choose a stronger recovered achievement.
    section_fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, fields in section_fields.items():
        for record in getattr(resume, section):
            for field in fields:
                add(f"{section}.{field}", getattr(record, field))
            if hasattr(record, "bullets"):
                for bullet in record.bullets:
                    add(f"{section}.bullets", bullet)
    for skill in resume.skills.items:
        add("skills", skill.name)
    for section in (
        "awards", "publications", "patents", "certifications", "training", "teaching",
    ):
        for value in getattr(resume, section):
            add(section, value)
    for title, values in resume.additional_sections.items():
        for value in values:
            add(f"additional_sections.{title}", value)
    return claims


def _resume_claims_are_preserved(
    before: CanonicalResume,
    after: CanonicalResume,
) -> bool:
    """Allow an old claim to gain exact source detail, never to disappear."""

    after_by_section: dict[str, list[str]] = {}
    for section, value in _resume_claim_fingerprints(after):
        after_by_section.setdefault(section, []).append(value)
    return all(
        any(value in candidate for candidate in after_by_section.get(section, []))
        for section, value in _resume_claim_fingerprints(before)
    )


def _critical_structural_additions(audit: dict) -> int:
    invariants = audit.get("structural_invariants") or {}
    return sum(
        int((invariants.get(section) or {}).get("added_count") or 0)
        for section in (
            "organization", "role", "period", "education", "credential", "metric",
        )
    )


def _audit_existing_scaffold(
    resume: CanonicalResume,
    source,
    evidence_bindings: list,
) -> dict:
    """Audit an already evidence-gated scaffold without recompiling it.

    ``enforce_resume_evidence`` has already bound and repaired every public
    claim in this candidate.  Dense query-only profiles used to feed that same
    scaffold through ``compile_fact_coverage`` and then bind/audit it several
    more times merely to decide whether Composer could be skipped.  On a long
    profile that CPU-only repetition could consume most of the request budget.

    This fast-path keeps the exact same promotion invariants: no unsupported
    atom, no ownership error, and no critical structural addition.  It only
    reuses the final binding set and performs the independent atomic audit
    once.  A scaffold below the recall threshold still follows the normal
    Composer/compiler path.
    """

    audit = audit_atomic_facts(
        source=source,
        resume=resume,
        evidence_bindings=evidence_bindings,
    )
    atomic = audit["atomic_factuality"]
    ownership = audit["ownership_integrity"]
    safe = bool(
        int(atomic["generated_atom_count"]) > 0
        and int(atomic["unsupported_atom_count"]) == 0
        and int(ownership["incorrect_assignment_count"]) == 0
        and _critical_structural_additions(audit) == 0
    )
    return {
        "mode": _fact_compiler_mode(),
        "accepted": _fact_compiler_mode() == "on" and safe,
        "safe": safe,
        "after_atomic_recall": float(atomic["recall"]),
        "after_atomic_precision": float(atomic["precision"]),
        "after_unsupported": int(atomic["unsupported_atom_count"]),
        "after_ownership_errors": int(ownership["incorrect_assignment_count"]),
        "critical_structural_additions": _critical_structural_additions(audit),
        "audit_reused_existing_bindings": True,
    }


_COMPILER_RECORD_PATH = re.compile(
    r"^(education|experience|research|activities|projects)\[(\d+)](?:\.(\w+)(?:\[(\d+)])?)?$"
)
_COMPILER_LIST_PATH = re.compile(
    r"^(awards|publications|patents|certifications|training|teaching)\[(\d+)]$"
)
_COMPILER_SKILL_PATH = re.compile(r"^skills\.items\[(\d+)]\.name$")
_COMPILER_ADDITIONAL_PATH = re.compile(r"^additional_sections\.(.+)\[(\d+)]$")


def _compiler_path_was_added(path: str, added_paths: list[str]) -> bool:
    """Whether an audited atom belongs to a compiler-created value/record."""

    return any(
        path == added
        or path.startswith(f"{added}.")
        or path.startswith(f"{added}[")
        for added in added_paths
    )


def _drop_compiler_path(resume: CanonicalResume, path: str) -> bool:
    """Remove one newly unsafe compiler value without deleting sibling facts."""

    if path.startswith("summary[") or path == "summary":
        resume.summary = ""
        return True
    if path.startswith("meta."):
        field_name = path.split(".", 1)[1]
        if hasattr(resume.meta, field_name):
            setattr(resume.meta, field_name, "")
            return True
        return False
    if match := _COMPILER_RECORD_PATH.fullmatch(path):
        section, index_text, field_name, child_index = match.groups()
        records = getattr(resume, section)
        index = int(index_text)
        if index >= len(records):
            return False
        if field_name is None:
            records.pop(index)
            return True
        record = records[index]
        if field_name == "bullets" and child_index is not None:
            bullet_index = int(child_index)
            if bullet_index < len(record.bullets):
                record.bullets.pop(bullet_index)
                return True
            return False
        if hasattr(record, field_name):
            setattr(record, field_name, "")
            return True
        return False
    if match := _COMPILER_SKILL_PATH.fullmatch(path):
        index = int(match.group(1))
        if index < len(resume.skills.items):
            resume.skills.items.pop(index)
            return True
        return False
    if match := _COMPILER_LIST_PATH.fullmatch(path):
        section, index_text = match.groups()
        values = getattr(resume, section)
        index = int(index_text)
        if index < len(values):
            values.pop(index)
            return True
        return False
    if match := _COMPILER_ADDITIONAL_PATH.fullmatch(path):
        title, index_text = match.groups()
        values = resume.additional_sections.get(title, [])
        index = int(index_text)
        if index < len(values):
            values.pop(index)
            if not values:
                resume.additional_sections.pop(title, None)
            return True
    return False


def _drop_empty_compiler_records(resume: CanonicalResume) -> None:
    fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, names in fields.items():
        kept = []
        for record in getattr(resume, section):
            values = [str(getattr(record, name, "") or "").strip() for name in names]
            values.extend(
                str(value or "").strip()
                for value in getattr(record, "bullets", [])
            )
            if any(values):
                kept.append(record)
        setattr(resume, section, kept)


def _prune_unsafe_compiler_additions(
    candidate: CanonicalResume,
    report,
    after_audit: dict,
) -> tuple[CanonicalResume, list[str]]:
    """Apply the plan's minimum-edit rule to a partially unsafe candidate.

    The safety audit can reject one malformed fallback field while dozens of
    exact source facts are valid.  Reverting the whole transaction recreates
    the old recall problem.  Remove only compiler-created atoms that the audit
    marks unsupported or incorrectly owned, then let the caller re-audit the
    remaining candidate.
    """

    unsafe_paths = {
        str(item.get("canonical_field_path") or "")
        for item in (after_audit.get("atomic_factuality") or {}).get(
            "unsupported_output", []
        )
    }
    unsafe_paths.update(
        str(item.get("canonical_field_path") or "")
        for item in (after_audit.get("ownership_integrity") or {}).get(
            "issues", []
        )
    )
    unsafe_paths = {
        path for path in unsafe_paths
        if path and _compiler_path_was_added(path, report.added_paths)
    }
    if not unsafe_paths:
        return candidate, []

    pruned = candidate.model_copy(deep=True)
    removed: list[str] = []
    # Remove list indexes from right to left so earlier paths stay stable.
    for path in sorted(unsafe_paths, reverse=True):
        if _drop_compiler_path(pruned, path):
            removed.append(path)
    _drop_empty_compiler_records(pruned)
    return pruned, sorted(removed)


def _apply_fact_compiler_candidate(
    resume: CanonicalResume,
    source,
    *,
    scaffold: CanonicalResume | None,
    trusted_rewrites: dict[str, str] | None = None,
    allow_reordered_record_bullets: bool = False,
    mode_override: str | None = None,
    merge_scaffold: bool = True,
    allowed_destinations: frozenset[str] | None = None,
    cleanup_presentation: bool = True,
    require_identity_owner: bool = False,
) -> tuple[CanonicalResume, list, list[str], dict]:
    """Apply the compiler transactionally under factual and ownership gates."""

    mode = mode_override or _fact_compiler_mode()
    before_bindings = bind_resume_evidence(
        resume,
        source,
        trusted_rewrites=trusted_rewrites,
    )
    if mode == "legacy":
        return resume, before_bindings, [], {
            "mode": mode,
            "accepted": False,
            "reason": "disabled",
        }

    if cleanup_presentation:
        working_resume, placeholder_paths = sanitize_resume_placeholders(resume)
    else:
        working_resume = resume
        placeholder_paths = []
    before_bindings = bind_resume_evidence(
        working_resume,
        source,
        trusted_rewrites=trusted_rewrites,
    )

    candidate, report, routes = compile_fact_coverage(
        working_resume,
        source,
        scaffold=scaffold,
        merge_scaffold=merge_scaffold,
        allowed_destinations=allowed_destinations,
        require_identity_owner=require_identity_owner,
    )
    if cleanup_presentation:
        candidate, compiler_placeholder_paths = sanitize_resume_placeholders(candidate)
        placeholder_paths = list(dict.fromkeys(
            [*placeholder_paths, *compiler_placeholder_paths]
        ))
    # ``resume`` has already passed the normal evidence gate, while the
    # compiler adds only verbatim facts or a separately grounded scaffold.
    # Re-running the destructive legacy gate over the entire merged document
    # can delete old claims when newly improved record IDs differ from those
    # used during the earlier pass.  Audit the append-only candidate directly
    # and prune only its unsafe additions below.
    removed: list[str] = []
    # The caller has already compacted and grounded the base resume.  Running
    # ``_compact_canonical`` here would rebuild the complete summary and turn a
    # local fact-recovery transaction into an unrelated prose rewrite.  Keep
    # the compiler append-only; normal pipeline finalization remains
    # responsible for presentation cleanup.
    candidate_bindings = bind_resume_evidence(
        candidate,
        source,
        trusted_rewrites=trusted_rewrites,
    )
    before_audit = audit_atomic_facts(
        source=source,
        resume=working_resume,
        evidence_bindings=before_bindings,
    )
    after_audit = audit_atomic_facts(
        source=source,
        resume=candidate,
        evidence_bindings=candidate_bindings,
    )
    candidate, pruned_paths = _prune_unsafe_compiler_additions(
        candidate,
        report,
        after_audit,
    )
    uncleaned_before_audit = before_audit
    # The compiler runs after normal compaction and may merge a grounded
    # scaffold containing duplicate records, credential fragments or OCR
    # recovery tails.  Apply the same local presentation invariants to both
    # sides of the transaction, then re-audit the exact candidate that may be
    # committed.  Summary prose is deliberately left untouched.
    base_presentation_pruned = False
    candidate_presentation_pruned = False
    if cleanup_presentation:
        working_data = working_resume.model_dump()
        _clean_compiler_presentation_data(
            working_data,
            source=source,
            routes=routes,
        )
        cleaned_working_resume = CanonicalResume.model_validate(working_data)
        base_presentation_pruned = cleaned_working_resume != working_resume
        if base_presentation_pruned:
            working_resume = cleaned_working_resume

        candidate_data = candidate.model_dump()
        _clean_compiler_presentation_data(
            candidate_data,
            source=source,
            routes=routes,
        )
        cleaned_candidate = CanonicalResume.model_validate(candidate_data)
        candidate_presentation_pruned = cleaned_candidate != candidate
        if candidate_presentation_pruned:
            candidate = cleaned_candidate
    presentation_pruned = base_presentation_pruned or candidate_presentation_pruned
    if pruned_paths or presentation_pruned:
        before_bindings = bind_resume_evidence(
            working_resume,
            source,
            trusted_rewrites=trusted_rewrites,
        )
        before_audit = audit_atomic_facts(
            source=source,
            resume=working_resume,
            evidence_bindings=before_bindings,
        )
        candidate_bindings = bind_resume_evidence(
            candidate,
            source,
            trusted_rewrites=trusted_rewrites,
        )
        after_audit = audit_atomic_facts(
            source=source,
            resume=candidate,
            evidence_bindings=candidate_bindings,
        )
    before_atomic = before_audit["atomic_factuality"]
    after_atomic = after_audit["atomic_factuality"]
    before_ownership = before_audit["ownership_integrity"]
    after_ownership = after_audit["ownership_integrity"]
    recall_gain = (
        int(after_atomic["represented_source_fact_count"])
        - int(before_atomic["represented_source_fact_count"])
    )
    candidate_safe = bool(
        recall_gain >= 0
        and bool(recall_gain > 0 or placeholder_paths or presentation_pruned)
        and int(after_atomic["unsupported_atom_count"])
        <= int(before_atomic["unsupported_atom_count"])
        and int(after_ownership["incorrect_assignment_count"])
        <= int(before_ownership["incorrect_assignment_count"])
        and _critical_structural_additions(after_audit)
        <= _critical_structural_additions(before_audit)
        and _resume_claims_are_preserved(working_resume, candidate)
    )
    uncleaned_atomic = uncleaned_before_audit["atomic_factuality"]
    uncleaned_ownership = uncleaned_before_audit["ownership_integrity"]
    cleanup_only_safe = bool(
        base_presentation_pruned
        and int(before_atomic["unsupported_atom_count"])
        <= int(uncleaned_atomic["unsupported_atom_count"])
        and int(before_ownership["incorrect_assignment_count"])
        <= int(uncleaned_ownership["incorrect_assignment_count"])
        and _critical_structural_additions(before_audit)
        <= _critical_structural_additions(uncleaned_before_audit)
    )
    safe = candidate_safe or cleanup_only_safe
    accepted = mode == "on" and safe
    selected = "candidate" if candidate_safe else "cleanup_only" if cleanup_only_safe else "original"
    diagnostics = {
        "mode": mode,
        "accepted": accepted,
        "safe": safe,
        "recall_gain": recall_gain,
        "before_atomic_recall": before_atomic["recall"],
        "after_atomic_recall": after_atomic["recall"],
        "before_unsupported": before_atomic["unsupported_atom_count"],
        "after_unsupported": after_atomic["unsupported_atom_count"],
        "before_ownership_errors": before_ownership["incorrect_assignment_count"],
        "after_ownership_errors": after_ownership["incorrect_assignment_count"],
        "placeholder_cleanup_paths": placeholder_paths,
        "presentation_cleanup": presentation_pruned,
        "selected": selected,
        "route_statuses": dict(Counter(route.status for route in routes)),
        "pruned_unsafe_paths": pruned_paths,
        "report": asdict(report),
    }
    trace_event("fact_compiler", **diagnostics)
    if accepted:
        if not candidate_safe:
            logger.warning(
                "V2 | Fact compiler accepted presentation-only cleanup; "
                "unsupported=%d ownership_errors=%d",
                int(before_atomic["unsupported_atom_count"]),
                int(before_ownership["incorrect_assignment_count"]),
            )
            return working_resume, before_bindings, removed, diagnostics
        logger.warning(
            "V2 | Fact compiler accepted: atomic recall %.1f%% -> %.1f%%; "
            "recovered=%d paths=%d",
            float(before_atomic["recall"]) * 100,
            float(after_atomic["recall"]) * 100,
            recall_gain,
            len(report.added_paths),
        )
        return candidate, candidate_bindings, removed, diagnostics
    logger.info(
        "V2 | Fact compiler %s: safe=%s atomic recall %.1f%% -> %.1f%%",
        "shadowed" if mode == "shadow" else "rejected",
        safe,
        float(before_atomic["recall"]) * 100,
        float(after_atomic["recall"]) * 100,
    )
    return resume, before_bindings, [], diagnostics


def _record_identity_keys(section: str, record) -> set[tuple[str, ...]]:
    """Build conservative record keys; false splits are safer than false joins."""

    def value(field: str) -> str:
        return _identity_value(getattr(record, field, ""))

    candidates: list[tuple[str, ...]] = []
    if section == "education":
        candidates = [
            (value("school"), value("period")),
            (value("school"), value("degree"), value("major")),
        ]
    elif section == "experience":
        candidates = [
            (value("organization"), value("period")),
            (value("organization"), value("role"), value("period")),
            (value("organization"), value("role")),
        ]
    elif section == "research":
        candidates = [
            (value("institution"), value("period")),
            (value("topic"), value("period")),
            (value("institution"), value("topic")),
        ]
    elif section == "activities":
        candidates = [
            (value("organization"), value("period")),
            (value("organization"), value("role")),
        ]
    elif section == "projects":
        candidates = [
            (value("name"), value("period")),
            (value("name"), value("organization")),
            (value("organization"), value("role"), value("period")),
        ]
    return {
        candidate for candidate in candidates
        if len(candidate) >= 2 and all(candidate)
    }


def _find_recovery_record(section: str, records: list, fallback_record) -> int | None:
    """Return one compatible record, never the first weak partial match.

    Repeated employers and roles are common.  The previous first-match lookup
    could attach a fallback row to the wrong employment period whenever the
    weaker ``organization + role`` key happened to match first.  Score all
    non-conflicting identities and require one unique best candidate instead.
    """

    fields_by_section = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    fields = fields_by_section.get(section, ())
    if not fields:
        return None

    fallback_values = {
        field: _identity_value(getattr(fallback_record, field, ""))
        for field in fields
    }
    if not any(fallback_values.values()):
        return None

    primary_fields = {
        "education": {"school"},
        "experience": {"organization"},
        "research": {"institution", "topic"},
        "activities": {"organization"},
        "projects": {"name"},
    }.get(section, set())
    candidates: list[tuple[int, int]] = []
    for index, record in enumerate(records):
        values = {
            field: _identity_value(getattr(record, field, ""))
            for field in fields
        }
        # Explicitly different periods or primary identities describe
        # different records even if a role/title happens to be repeated.
        conflicting = any(
            fallback_values[field]
            and values[field]
            and fallback_values[field] != values[field]
            for field in ({"period"} | primary_fields)
        )
        if conflicting:
            continue
        matched = [
            field for field in fields
            if fallback_values[field]
            and fallback_values[field] == values[field]
        ]
        if not matched:
            continue
        score = sum(
            5 if field == "period" else 4 if field in primary_fields else 2
            for field in matched
        )
        candidates.append((score, index))
    if not candidates:
        return None
    best_score = max(score for score, _ in candidates)
    best = [index for score, index in candidates if score == best_score]
    return best[0] if len(best) == 1 else None


def _bullet_is_represented(
    source_bullet: str,
    claims: list[str],
) -> bool:
    source_value = _identity_value(source_bullet)
    if not source_value:
        return True
    for claim in claims:
        claim_value = _identity_value(claim)
        if not claim_value:
            continue
        if source_value == claim_value:
            return True
        # Coverage is directional: an existing short clause does not represent
        # a longer source sentence that contains additional duties or results.
        # The reverse check used to suppress recovery for compact OCR lines such
        # as ``负责沟通、调研、分析、输出PRD`` as soon as ``负责沟通`` survived.
        if len(source_value) >= 6 and source_value in claim_value:
            return True
        source_anchors = {
            item.casefold() for item in _COVERAGE_ANCHOR.findall(source_bullet)
        }
        claim_anchors = {
            item.casefold() for item in _COVERAGE_ANCHOR.findall(claim)
        }
        if source_anchors and not source_anchors.issubset(claim_anchors):
            continue
        source_bigrams = _coverage_bigrams(source_bullet)
        recall = len(source_bigrams & _coverage_bigrams(claim)) / max(
            1, len(source_bigrams)
        )
        if recall >= 0.62:
            return True
    return False


_RECORD_FIELD_PATH = re.compile(
    r"^(experience|research|activities|projects)\[(\d+)\]\."
)


def _clean_recovered_source_bullet(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"^(?:[-*•·▪◦]\s*|\d{1,3}(?:[.、)])\s*)", "", text)
    return text.strip(" \t。；;")


def _coherent_recovery_bullets(
    source_text: str,
    missing_units: list[dict[str, str]],
) -> list[str]:
    """Keep audit units fine-grained but recover readable source clauses."""

    missing_values = [
        _identity_value(unit.get("match_text", ""))
        for unit in missing_units
        if _identity_value(unit.get("match_text", ""))
    ]
    coherent = [
        _clean_recovered_source_bullet(value)
        for value in re.split(r"[。；;]+", str(source_text or ""))
    ]
    return list(dict.fromkeys(
        value for value in coherent
        if len(_identity_value(value)) >= 6
        and any(missing in _identity_value(value) for missing in missing_values)
    ))


def _recover_missing_record_facts(
    resume: CanonicalResume,
    source,
    bindings,
) -> tuple[CanonicalResume, _RecoveryStats, set[str]]:
    """Restore omitted record-body facts from an auditable source ledger.

    Recovery is intentionally narrower than the deterministic parser fallback:
    it never creates a record and never guesses across records.  A missing
    source block is attached only when evidence already maps that source record
    to exactly one output record, or when both sides contain one unambiguous
    record.  The recovered wording is copied from candidate evidence, never JD.
    """

    if not resolve_pipeline_profile().record_fact_recovery:
        return resume, _RecoveryStats(), set()

    units = source_fact_units(source)
    if not units:
        return resume, _RecoveryStats(), set()
    _, missing_ids = measure_source_coverage(
        source,
        bindings,
        allow_distributed=True,
    )
    missing = set(missing_ids)
    if not missing:
        return resume, _RecoveryStats(), set()

    eligible = candidate_blocks(source)
    blocks_by_id = {block.block_id: block for block in eligible}
    block_targets: dict[tuple[str, str], set[int]] = {}
    record_targets: dict[tuple[str, str], set[int]] = {
        (section, record_id): {index}
        for (section, index), record_id in _record_source_owners(
            resume, source,
        ).items()
    }
    for binding in bindings:
        match = _RECORD_FIELD_PATH.match(str(binding.path or ""))
        if match is None:
            continue
        section, index_text = match.groups()
        index = int(index_text)
        # A record-level optimizer rewrite may bind through several original
        # source blocks.  Every linked block contributes record ownership; using
        # only the primary block left later facts with no safe recovery target.
        for block_id in binding.block_ids or [binding.block_id]:
            block = blocks_by_id.get(str(block_id or ""))
            if block is None:
                continue
            if block.section_hint and block.section_hint != section:
                continue
            block_targets.setdefault((section, block.block_id), set()).add(index)

    missing_by_block: dict[str, list[dict[str, str]]] = {}
    for unit in units:
        if unit.get("unit_id") not in missing:
            continue
        if unit.get("section_hint") not in {
            "experience", "research", "activities", "projects",
        }:
            continue
        if unit.get("record_body") != "true":
            continue
        missing_by_block.setdefault(unit["block_id"], []).append(unit)

    merged = resume.model_copy(deep=True)
    changed_paths: set[str] = set()
    recovered_blocks: list[str] = []
    appended_bullets = 0
    expanded_bullets = 0
    for block_id, block_units in missing_by_block.items():
        block = blocks_by_id.get(block_id)
        if block is None or block.section_hint not in {
            "experience", "research", "activities", "projects",
        }:
            continue
        section = str(block.section_hint)
        records = getattr(merged, section)
        if not records:
            continue

        targets = set(block_targets.get((section, block_id), set()))
        if block.record_id:
            targets.update(record_targets.get((section, block.record_id), set()))
        if not targets and len(records) == 1:
            source_record_ids = {
                item.record_id
                for item in eligible
                if item.section_hint == section
                and item.record_id
                and _looks_like_record_body(item.text)
            }
            if block.record_id and source_record_ids == {block.record_id}:
                targets.add(0)
        if len(targets) != 1:
            continue
        record_index = next(iter(targets))
        if record_index >= len(records):
            continue
        record = records[record_index]

        recovery_bullets = [
            value
            for value in _coherent_recovery_bullets(block.text, block_units)
            if not _bullet_is_represented(value, list(record.bullets))
        ]
        if not recovery_bullets:
            continue
        for source_bullet in recovery_bullets:
            source_value = _identity_value(source_bullet)
            prefix_index = next((
                bullet_index
                for bullet_index, existing in enumerate(record.bullets)
                if 6 <= len(_identity_value(existing)) < len(source_value)
                and _identity_value(existing) in source_value
            ), None)
            if prefix_index is not None:
                # Expand an already-retained short prefix in place. Appending
                # the complete source line created duplicate openings and left
                # quality dependent on whether the LLM happened to consolidate
                # them. Exact containment makes this deterministic and lossless.
                record.bullets[prefix_index] = source_bullet
                changed_paths.add(f"{section}[{record_index}].bullets[{prefix_index}]")
                expanded_bullets += 1
                continue
            record.bullets.append(source_bullet)
            changed_paths.add(
                f"{section}[{record_index}].bullets[{len(record.bullets) - 1}]"
            )
            appended_bullets += 1
        recovered_blocks.append(block_id)

    if recovered_blocks:
        trace_event(
            "source_fact_ledger_recovery",
            recovered_blocks=recovered_blocks,
            changed_paths=sorted(changed_paths),
        )
    return merged, _RecoveryStats(
        appended_bullets=appended_bullets,
        expanded_bullets=expanded_bullets,
    ), changed_paths


def _restore_attested_source_summary(
    resume: CanonicalResume,
    source,
) -> tuple[CanonicalResume, list[str]]:
    """Restore exact candidate-authored summary facts after summary rebuilding.

    ``_compact_canonical`` ranks concrete role and achievement facts ahead of
    generic prose.  That is useful for presentation, but it used to drop an
    explicitly supplied personal-summary sentence altogether.  This final,
    deterministic pass restores only eligible facts whose source section is
    ``summary``.  The text is copied verbatim, never inferred from a JD, and
    source-adapter disclaimer filtering has already removed instructions such
    as "不得编造" or "以真实信息为准".
    """

    if not resolve_pipeline_profile().attested_summary_recovery:
        return resume, []

    candidates = [
        fact.verbatim_text.strip(" \t。；;！!？?")
        for fact in source.fact_units
        if fact.fact_eligible
        and fact.source_type != "jd"
        and fact.section_hint == "summary"
        and fact.verbatim_text.strip()
    ]
    candidates = list(dict.fromkeys(value for value in candidates if value))
    if not candidates:
        return resume, []

    restored = resume.model_copy(deep=True)
    current = restored.summary.strip()
    existing_sentences = [
        item.strip()
        for item in re.split(r"[。！？!?；;]+", current)
        if item.strip()
    ]
    added: list[str] = []
    for value in candidates:
        if _bullet_is_represented(value, existing_sentences + added):
            continue
        # Candidate-authored evidence may exceed the generated-summary target
        # slightly, but remains bounded and is never truncated mid-sentence.
        projected = len(current.rstrip("。；; ")) + len(value) + 2
        if projected > _SUMMARY_MAX_CHARS + 80:
            continue
        added.append(value)
        current = "。".join(
            part for part in (current.rstrip("。；; "), value) if part
        ) + "。"
        if len(added) >= 2:
            break

    if not added:
        return resume, []
    restored.summary = current
    return restored, added


_ANY_RECORD_FIELD_PATH = re.compile(
    r"^(education|experience|research|activities|projects)\[(\d+)]\."
)


def _record_source_owners(
    resume: CanonicalResume,
    source,
) -> dict[tuple[str, int], str]:
    """Resolve canonical rows against complete source-side records.

    Field-level bindings are intentionally allowed to be non-unique: a company,
    title, degree or generic duty can legitimately occur several times.  They
    are therefore the wrong primitive for deciding ownership.  This resolver
    scores every canonical row against every deterministic source record using
    the *joint* identity, period and body evidence, rejects explicit period or
    primary-identity conflicts, and assigns source rows one-to-one.  A tie stays
    unowned instead of being attached to whichever matching block came first.
    """

    fields_by_section = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    primary_by_section = {
        "education": {"school"},
        "experience": {"organization"},
        "research": {"institution", "topic"},
        "activities": {"organization"},
        "projects": {"name"},
    }

    def date_signature(value: str) -> tuple[str, ...]:
        text = unicodedata.normalize("NFKC", str(value or ""))
        pattern = re.compile(
            r"(?<!\d)(?:(?P<year1>(?:19|20)\d{2})\s*[-./年]\s*(?P<month1>0?[1-9]|1[0-2])\s*月?"
            r"|(?P<month2>0?[1-9]|1[0-2])\s*[-./]\s*(?P<year2>(?:19|20)\d{2})"
            r"|(?P<year_only>(?:19|20)\d{2}))(?!\d)"
        )
        signature: list[str] = []
        for match in pattern.finditer(text):
            year = match.group("year1") or match.group("year2") or match.group("year_only")
            month = match.group("month1") or match.group("month2")
            signature.append(f"{year}{int(month):02d}" if month else str(year))
        return tuple(signature)

    def signature_is_contained(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
        return bool(needle) and len(haystack) >= len(needle) and any(
            haystack[index:index + len(needle)] == needle
            for index in range(len(haystack) - len(needle) + 1)
        )

    def body_similarity(value: str, group_values: list[str]) -> tuple[int, bool]:
        normalized = _identity_value(value)
        if not normalized:
            return 0, False
        for candidate in group_values:
            if normalized in candidate or (
                len(candidate) >= 8 and candidate in normalized
            ):
                return 8, True
        source_bigrams = {
            normalized[index:index + 2]
            for index in range(max(0, len(normalized) - 1))
        }
        best_recall = 0.0
        for candidate in group_values:
            candidate_bigrams = {
                candidate[index:index + 2]
                for index in range(max(0, len(candidate) - 1))
            }
            best_recall = max(
                best_recall,
                len(source_bigrams & candidate_bigrams) / max(1, len(source_bigrams)),
            )
        return (3, False) if best_recall >= 0.72 else (0, False)

    eligible = candidate_blocks(source)
    owners: dict[tuple[str, int], str] = {}
    for section, fields in fields_by_section.items():
        source_groups: dict[str, list] = {}
        for block in eligible:
            if block.section_hint == section and block.record_id:
                source_groups.setdefault(block.record_id, []).append(block)
        if not source_groups:
            continue

        candidates_by_output: dict[int, list[tuple[int, int, str]]] = {}
        for output_index, record in enumerate(getattr(resume, section)):
            record_period = date_signature(str(getattr(record, "period", "") or ""))
            for record_id, group in source_groups.items():
                group_text = "\n".join(str(block.text or "") for block in group)
                group_values = [_identity_value(block.text) for block in group]
                group_period = date_signature(group_text)
                score = 0
                strong_anchors = 0
                conflict = False

                if record_period:
                    if signature_is_contained(record_period, group_period):
                        score += 16
                        strong_anchors += 2
                    elif group_period:
                        conflict = True

                for field in fields:
                    if field == "period":
                        continue
                    value = _identity_value(getattr(record, field, ""))
                    if not value:
                        continue
                    direct = any(
                        value in candidate or (
                            len(candidate) >= 6 and candidate in value
                        )
                        for candidate in group_values
                    )
                    if direct:
                        if field in primary_by_section[section]:
                            score += 10
                            strong_anchors += 1
                        else:
                            score += 4
                    elif field in primary_by_section[section] and any(
                        # A different explicit primary identity in this source
                        # record is a hard conflict. Body-only lines are ignored.
                        not _looks_like_record_body(block.text)
                        and len(_identity_value(block.text)) <= 100
                        for block in group
                    ):
                        conflict = True

                for bullet in getattr(record, "bullets", []):
                    bullet_score, exact = body_similarity(str(bullet), group_values)
                    score += bullet_score
                    if exact:
                        strong_anchors += 1

                if not conflict and strong_anchors and score >= 8:
                    candidates_by_output.setdefault(output_index, []).append(
                        (score, strong_anchors, record_id)
                    )

        # Strongest and least ambiguous rows claim their source record first.
        # This prevents two duplicate generated rows from both owning one source
        # record while still allowing a weaker row to use its second-best unique
        # source record when that alternative has a clear margin.
        pending: list[
            tuple[int, int, int, int, list[tuple[int, int, str]]]
        ] = []
        for output_index, candidates in candidates_by_output.items():
            ordered = sorted(candidates, reverse=True)
            best_score, best_anchors, _best_record_id = ordered[0]
            second_score = ordered[1][0] if len(ordered) > 1 else 0
            margin = best_score - second_score
            if len(ordered) > 1 and margin < 4:
                continue
            pending.append((best_score, margin, best_anchors, output_index, [
                (score, anchors, record_id)
                for score, anchors, record_id in ordered
            ]))

        used_source_records: set[str] = set()
        for _best, _margin, _anchors, output_index, ordered in sorted(
            pending,
            key=lambda item: (item[0], item[1], item[2]),
            reverse=True,
        ):
            available = [item for item in ordered if item[2] not in used_source_records]
            if not available:
                continue
            selected_score, _selected_anchors, selected = available[0]
            if selected_score < _best - 2:
                # The preferred source row was already claimed by a stronger
                # duplicate output. Do not force this row onto a materially
                # weaker second choice merely to complete the assignment.
                continue
            runner_score = available[1][0] if len(available) > 1 else 0
            if len(available) > 1 and selected_score - runner_score < 4:
                continue
            owners[(section, output_index)] = selected
            used_source_records.add(selected)

    trace_event(
        "source_record_ownership_graph",
        owners={f"{section}[{index}]": record_id for (section, index), record_id in owners.items()},
    )
    return owners


def _record_is_complete_for_recovery(section: str, record) -> bool:
    """Require identity + period + content before creating a missing row."""

    period = str(getattr(record, "period", "") or "").strip()
    if not period or not _FALLBACK_PERIOD.search(period):
        return False
    identity_fields = {
        "education": ("school",),
        "experience": ("organization",),
        "research": ("institution", "topic"),
        "activities": ("organization",),
        "projects": ("name",),
    }.get(section, ())
    identities = [
        str(getattr(record, field, "") or "").strip()
        for field in identity_fields
    ]
    if not any(identities) or any(
        _looks_like_record_body(value) for value in identities if value
    ):
        return False
    if section == "education":
        return bool(str(record.degree or "").strip() or str(record.major or "").strip())
    bullets = [str(value or "").strip() for value in getattr(record, "bullets", [])]
    return any(_looks_like_record_body(value) for value in bullets if value)


def _recover_grounded_source_structure(
    resume: CanonicalResume,
    fallback: CanonicalResume,
    source,
) -> tuple[CanonicalResume, _RecoveryStats]:
    """Recover source-backed fields and complete rows independently of recall.

    The earlier fallback ran only when global source coverage was below 80%
    and at least ten points worse than the deterministic parser.  That allowed
    one whole job, certificate, or award to disappear from otherwise dense
    resumes.  Here each value is considered independently, while a new record
    additionally needs one unique source owner plus identity, period and body
    evidence.  Ambiguous or incomplete rows remain absent.
    """

    pipeline_profile = resolve_pipeline_profile()
    if not pipeline_profile.source_structure_recovery:
        return resume, _RecoveryStats()
    fields_only = pipeline_profile.structure_fields_only

    merged = resume.model_copy(deep=True)
    fallback_bindings = bind_resume_evidence(fallback, source)
    bound_paths = {str(binding.path or "") for binding in fallback_bindings}
    fallback_owners = _record_source_owners(fallback, source)
    output_owners = _record_source_owners(merged, source)
    filled_fields = appended_bullets = appended_records = appended_values = 0

    for field in ("name", "phone", "email", "work_experience"):
        path = f"meta.{field}"
        value = getattr(fallback.meta, field)
        if not getattr(merged.meta, field) and value and path in bound_paths:
            setattr(merged.meta, field, value)
            filled_fields += 1

    section_fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, fields in section_fields.items():
        records = getattr(merged, section)
        owned_targets: dict[str, list[int]] = {}
        for (owned_section, index), record_id in output_owners.items():
            if owned_section == section:
                owned_targets.setdefault(record_id, []).append(index)

        for fallback_index, fallback_record in enumerate(getattr(fallback, section)):
            owner = fallback_owners.get((section, fallback_index))
            owner_matches = list(owned_targets.get(owner, [])) if owner else []
            match_index = owner_matches[0] if len(owner_matches) == 1 else None
            if match_index is None and not owner_matches:
                match_index = _find_recovery_record(section, records, fallback_record)

            if match_index is None:
                if (
                    not fields_only
                    and owner
                    and not owner_matches
                    and _record_is_complete_for_recovery(section, fallback_record)
                ):
                    records.append(fallback_record.model_copy(deep=True))
                    new_index = len(records) - 1
                    output_owners[(section, new_index)] = owner
                    owned_targets.setdefault(owner, []).append(new_index)
                    appended_records += 1
                continue

            record = records[match_index]
            for field in fields:
                fallback_path = f"{section}[{fallback_index}].{field}"
                value = getattr(fallback_record, field)
                if (
                    not getattr(record, field)
                    and value
                    and fallback_path in bound_paths
                ):
                    setattr(record, field, value)
                    filled_fields += 1
            if not hasattr(record, "bullets"):
                continue
            if fields_only:
                continue
            claims = list(record.bullets)
            for bullet_index, bullet in enumerate(fallback_record.bullets):
                fallback_path = f"{section}[{fallback_index}].bullets[{bullet_index}]"
                if (
                    fallback_path in bound_paths
                    and not _bullet_is_represented(bullet, claims)
                ):
                    record.bullets.append(bullet)
                    claims.append(bullet)
                    appended_bullets += 1

    if fields_only:
        return merged, _RecoveryStats(filled_fields=filled_fields)

    for field in (
        "awards", "publications", "patents", "certifications", "training", "teaching",
    ):
        values = getattr(merged, field)
        seen = {_identity_value(value) for value in values}
        for index, value in enumerate(getattr(fallback, field)):
            normalized = _identity_value(value)
            if (
                normalized
                and normalized not in seen
                and f"{field}[{index}]" in bound_paths
            ):
                values.append(value)
                seen.add(normalized)
                appended_values += 1

    existing_skills = {_identity_value(item.name) for item in merged.skills.items}
    for index, item in enumerate(fallback.skills.items):
        normalized = _identity_value(item.name)
        if (
            normalized
            and normalized not in existing_skills
            and f"skills.items[{index}].name" in bound_paths
        ):
            merged.skills.items.append(item.model_copy(deep=True))
            existing_skills.add(normalized)
            appended_values += 1

    # Named long-tail sections are useful across industries, but parser
    # scratch buckets must never leak into the public resume.
    for title, values in fallback.additional_sections.items():
        if title == "补充信息" or _INTERNAL_ADDITIONAL_SECTION.fullmatch(title.strip()):
            continue
        destination = merged.additional_sections.setdefault(title, [])
        seen = {_identity_value(value) for value in destination}
        for index, value in enumerate(values):
            normalized = _identity_value(value)
            path = f"additional_sections.{title}[{index}]"
            if normalized and normalized not in seen and path in bound_paths:
                destination.append(value)
                seen.add(normalized)
                appended_values += 1

    return merged, _RecoveryStats(
        filled_fields=filled_fields,
        appended_bullets=appended_bullets,
        appended_records=appended_records,
        appended_values=appended_values,
    )


def _merge_source_recovery(
    resume: CanonicalResume,
    fallback: CanonicalResume,
    *,
    trusted_rewrites: dict[str, str] | None = None,
    allow_new_records: bool = True,
    restore_empty_sections: bool = False,
) -> tuple[CanonicalResume, _RecoveryStats]:
    """Fill missing source facts without replacing already-optimized content."""

    merged = resume.model_copy(deep=True)
    filled_fields = appended_bullets = appended_records = appended_values = 0

    for field in ("name", "phone", "email", "work_experience"):
        if not getattr(merged.meta, field) and getattr(fallback.meta, field):
            setattr(merged.meta, field, getattr(fallback.meta, field))
            filled_fields += 1

    section_fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, fields in section_fields.items():
        records = getattr(merged, section)
        section_was_empty = not records
        for fallback_record in getattr(fallback, section):
            match_index = _find_recovery_record(section, records, fallback_record)
            if match_index is None:
                if allow_new_records or (restore_empty_sections and section_was_empty):
                    records.append(fallback_record.model_copy(deep=True))
                    appended_records += 1
                continue
            record = records[match_index]
            for field in fields:
                if not getattr(record, field) and getattr(fallback_record, field):
                    setattr(record, field, getattr(fallback_record, field))
                    filled_fields += 1
            if not hasattr(record, "bullets"):
                continue
            claims = list(record.bullets)
            if trusted_rewrites:
                claims.extend(
                    source_value
                    for path, source_value in trusted_rewrites.items()
                    if path.startswith(f"{section}[{match_index}].bullets[")
                )
            for bullet in fallback_record.bullets:
                if not _bullet_is_represented(bullet, claims):
                    record.bullets.append(bullet)
                    claims.append(bullet)
                    appended_bullets += 1

    for field in (
        "awards", "publications", "patents", "certifications", "training", "teaching"
    ):
        values = getattr(merged, field)
        seen = {_identity_value(value) for value in values}
        for value in getattr(fallback, field):
            normalized = _identity_value(value)
            if normalized and normalized not in seen:
                values.append(value)
                seen.add(normalized)
                appended_values += 1

    skill_names = {_identity_value(item.name) for item in merged.skills.items}
    for item in fallback.skills.items:
        normalized = _identity_value(item.name)
        if normalized and normalized not in skill_names:
            merged.skills.items.append(item.model_copy(deep=True))
            skill_names.add(normalized)
            appended_values += 1

    for title, values in fallback.additional_sections.items():
        destination = merged.additional_sections.setdefault(title, [])
        seen = {_identity_value(value) for value in destination}
        for value in values:
            normalized = _identity_value(value)
            if normalized and normalized not in seen:
                destination.append(value)
                seen.add(normalized)
                appended_values += 1

    if not merged.summary and fallback.summary:
        merged.summary = fallback.summary
        filled_fields += 1

    return merged, _RecoveryStats(
        filled_fields=filled_fields,
        appended_bullets=appended_bullets,
        appended_records=appended_records,
        appended_values=appended_values,
    )


def _has_structured_history(resume: CanonicalResume) -> bool:
    """Require real canonical records, not merely raw text parked in extras."""

    return any((
        resume.education,
        resume.experience,
        resume.research,
        resume.activities,
        resume.projects,
    ))


def _merge_query_fallback_sections(
    resume: CanonicalResume,
    fallback: CanonicalResume,
) -> CanonicalResume:
    """Recover explicit query facts section by section without duplicating rows.

    Query-only Composer output can be schema-valid yet omit an entire section
    (the log cases lost all internships or skills while retaining education).
    The deterministic parser is deliberately less fluent but every value is
    evidence-gated. Use it to fill absent sections and compatible singleton
    rows; never replace richer Composer prose or append an ambiguous competing
    record to a populated section.
    """

    merged = resume.model_copy(deep=True)
    for field in ("name", "phone", "email", "target_role", "work_experience"):
        if not getattr(merged.meta, field) and getattr(fallback.meta, field):
            setattr(merged.meta, field, getattr(fallback.meta, field))

    section_fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, fields in section_fields.items():
        current = getattr(merged, section)
        recovery = getattr(fallback, section)
        if not current and recovery:
            setattr(merged, section, [item.model_copy(deep=True) for item in recovery])
            continue
        if len(current) != 1 or len(recovery) != 1:
            continue
        destination = current[0]
        source_record = recovery[0]
        conflicts = [
            field for field in fields
            if getattr(destination, field)
            and getattr(source_record, field)
            and _identity_value(getattr(destination, field))
            != _identity_value(getattr(source_record, field))
        ]
        if conflicts:
            continue
        for field in fields:
            if not getattr(destination, field) and getattr(source_record, field):
                setattr(destination, field, getattr(source_record, field))
        if hasattr(destination, "bullets"):
            for bullet in source_record.bullets:
                if not _bullet_is_represented(bullet, list(destination.bullets)):
                    destination.bullets.append(bullet)

    existing_skills = {_identity_value(item.name) for item in merged.skills.items}
    for item in fallback.skills.items:
        normalized = _identity_value(item.name)
        if normalized and normalized not in existing_skills:
            merged.skills.items.append(item.model_copy(deep=True))
            existing_skills.add(normalized)

    for section in (
        "awards", "publications", "patents", "certifications", "training", "teaching"
    ):
        values = getattr(merged, section)
        existing = {_identity_value(value) for value in values}
        for value in getattr(fallback, section):
            normalized = _identity_value(value)
            if normalized and normalized not in existing:
                values.append(value)
                existing.add(normalized)
    return merged


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
        allow_reordered_record_bullets=True,
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
    *,
    record_recovery_allowed: bool = True,
) -> VerifiedResult:
    """Run the V2 5-layer pipeline. Returns VerifiedResult or fallback."""
    t_start = time.perf_counter()
    pipeline_profile = resolve_pipeline_profile()
    logger.info(
        "V2 | Pipeline profile=%s compiler=%s structure_recovery=%s "
        "record_recovery=%s query_narrative=%s cv_narrative=%s",
        pipeline_profile.name,
        pipeline_profile.fact_compiler_mode,
        pipeline_profile.source_structure_recovery,
        pipeline_profile.record_fact_recovery,
        pipeline_profile.query_narrative,
        pipeline_profile.cv_narrative,
    )
    trace_event("pipeline_profile", profile=pipeline_profile.trace_payload())
    trace_event(
        "record_recovery_policy",
        allowed=record_recovery_allowed,
        reason=("stable_text_layout" if record_recovery_allowed else "ocr_layout_unstable"),
    )
    trace_event(
        "v2_input",
        cv_text=cv_text,
        query_text=query_text,
        jd_text=jd_text,
    )

    # ── No CV: generate structured framework from query + JD ──
    if not cv_text or not cv_text.strip():
        logger.info("V2 | No CV — generating framework from query+JD")
        t_gen = time.perf_counter()
        evidence_source = build_source_bundle("", query_text, jd_text)
        trace_event("source_bundle", source=evidence_source)
        candidate_source_blocks = candidate_blocks(evidence_source)
        candidate_evidence = "\n".join(
            block.text for block in candidate_source_blocks
        )
        used_fallback = False
        grounded_fallback = CanonicalResume()
        grounded_fallback_bindings: list = []
        grounded_fallback_removed: list[str] = []
        if candidate_evidence:
            raw_fallback = _deterministic_fallback("", query_text, jd_text)
            fallback_data = raw_fallback.model_dump()
            _ground_fixed_fields(fallback_data, candidate_evidence)
            _reclassify_non_work(fallback_data, candidate_evidence)
            grounded_fallback = CanonicalResume.model_validate(fallback_data)
            grounded_fallback = _ground_bullets(
                grounded_fallback,
                candidate_evidence,
            )
            grounded_fallback, _, grounded_fallback_removed = enforce_resume_evidence(
                grounded_fallback,
                evidence_source,
            )
            # ``补充信息`` is parser scratch space, not a safe semantic type.
            # Composer may still emit a named long-tail section when it can
            # classify one from the same source.
            grounded_fallback.additional_sections.pop("补充信息", None)
            grounded_fallback = _compact_canonical(grounded_fallback)
            grounded_fallback, fallback_placeholder_paths = sanitize_resume_placeholders(
                grounded_fallback,
            )
            # Compaction rebuilds the summary, so retain one binding set that
            # exactly matches the candidate inspected by the fast-path audit.
            grounded_fallback_bindings = bind_resume_evidence(
                grounded_fallback,
                evidence_source,
            )
            if fallback_placeholder_paths:
                trace_event(
                    "query_fallback_placeholder_cleanup",
                    paths=fallback_placeholder_paths,
                )

        query_compiler_fastpath = False
        resume = CanonicalResume()
        if (
            candidate_evidence
            and _fact_compiler_mode() == "on"
            and len(candidate_source_blocks)
            >= _QUERY_FACT_COMPILER_FASTPATH_MIN_BLOCKS
            and not _is_empty_resume(grounded_fallback)
        ):
            try:
                query_fastpath_resume = grounded_fallback
                query_fastpath_bindings = grounded_fallback_bindings
                query_fastpath_diagnostics = _audit_existing_scaffold(
                    query_fastpath_resume,
                    evidence_source,
                    query_fastpath_bindings,
                )
                direct_fastpath_recall = float(
                    query_fastpath_diagnostics.get("after_atomic_recall") or 0.0
                )
                if (
                    query_fastpath_diagnostics.get("accepted")
                    and direct_fastpath_recall < 1.0
                ):
                    (
                        compiled_query_resume,
                        compiled_query_bindings,
                        _compiled_query_removed,
                        compiled_query_diagnostics,
                    ) = _apply_fact_compiler_candidate(
                        query_fastpath_resume,
                        evidence_source,
                        scaffold=query_fastpath_resume,
                        allow_reordered_record_bullets=True,
                    )
                    compiled_recall = float(
                        compiled_query_diagnostics.get("after_atomic_recall") or 0.0
                    )
                    if (
                        compiled_query_diagnostics.get("accepted")
                        and compiled_recall > direct_fastpath_recall
                        and int(compiled_query_diagnostics.get("after_unsupported") or 0) == 0
                        and int(compiled_query_diagnostics.get("after_ownership_errors") or 0) == 0
                    ):
                        query_fastpath_resume = compiled_query_resume
                        query_fastpath_bindings = compiled_query_bindings
                        query_fastpath_diagnostics = compiled_query_diagnostics
                query_fastpath_recall = float(
                    query_fastpath_diagnostics.get("after_atomic_recall") or 0.0
                )
                query_allows_missing_school = not any(
                    fact.fact_eligible
                    and fact.section_hint == "education"
                    and _SCHOOL_SUFFIX.search(fact.verbatim_text)
                    and not _FALLBACK_PLACEHOLDER_TOKEN.search(fact.verbatim_text)
                    for fact in evidence_source.fact_units
                )
                query_fastpath_structure_safe = _fallback_is_structurally_safe(
                    query_fastpath_resume,
                    allow_period_only_experience=True,
                    allow_missing_education_school=query_allows_missing_school,
                    allow_partial_record_identity=True,
                )
                if (
                    query_fastpath_diagnostics.get("accepted")
                    and query_fastpath_recall
                    >= _QUERY_FACT_COMPILER_FASTPATH_MIN_ATOMIC_RECALL
                    and _has_candidate_profile(query_fastpath_resume)
                    and query_fastpath_structure_safe
                ):
                    resume = query_fastpath_resume
                    query_compiler_fastpath = True
                    used_fallback = True
                    trace_event(
                        "query_composer_skipped_for_fact_compiler",
                        candidate_blocks=len(candidate_source_blocks),
                        diagnostics=query_fastpath_diagnostics,
                    )
                    logger.warning(
                        "V2 | Dense query Composer skipped: audited scaffold "
                        "atomic recall %.1f%% across %d candidate blocks",
                        query_fastpath_recall * 100,
                        len(candidate_source_blocks),
                    )
                else:
                    logger.info(
                        "V2 | Dense query scaffold not promoted: accepted=%s "
                        "recall=%.1f%% structure_safe=%s",
                        bool(query_fastpath_diagnostics.get("accepted")),
                        query_fastpath_recall * 100,
                        query_fastpath_structure_safe,
                    )
            except Exception as exc:
                logger.warning("V2 | Dense query fact-compiler scaffold failed: %s", exc)

        if query_compiler_fastpath:
            # The scaffold was already compacted, evidence-gated, rebound and
            # independently audited above.  Optimize only a bounded set of
            # multi-clause records, then re-run the same independent audit.
            # This keeps the fast path cheap while allowing source fragments
            # from one record to become coherent accomplishment sentences.
            optimizer_changes: list[Change] = []
            optimizer_removed: list[str] = []
            narrative_record_keys = select_narrative_record_keys(resume)
            query_narrative_enabled = pipeline_profile.query_narrative
            if narrative_record_keys and query_narrative_enabled:
                before_optimizer = _rank_resume_content(
                    resume,
                    jd_text or resume.meta.target_role,
                )
                optimizer_bindings = bind_resume_evidence(
                    before_optimizer,
                    evidence_source,
                )
                optimization = optimize_resume_with_provenance(
                    before_optimizer,
                    jd_text,
                    evidence_bindings=optimizer_bindings,
                    record_keys=narrative_record_keys,
                )
                if optimization.accepted:
                    trusted_rewrites = dict(optimization.trusted_rewrites)
                    optimized_resume = _ground_optimizer_output(
                        before_optimizer,
                        optimization.resume,
                        query_text,
                        trusted_rewrites=trusted_rewrites,
                    )
                    optimized_resume = _ground_bullets(
                        optimized_resume,
                        query_text,
                        trusted_rewrites=trusted_rewrites,
                    )
                    optimized_resume = validate_resume(
                        optimized_resume,
                        source_text=query_text,
                    )
                    (
                        optimized_resume,
                        optimized_bindings,
                        optimized_removed,
                    ) = enforce_resume_evidence(
                        optimized_resume,
                        evidence_source,
                        trusted_rewrites=trusted_rewrites,
                        allow_reordered_record_bullets=True,
                    )
                    optimized_resume = _compact_canonical(optimized_resume)
                    optimized_bindings = bind_resume_evidence(
                        optimized_resume,
                        evidence_source,
                        trusted_rewrites=trusted_rewrites,
                    )
                    optimized_diagnostics = _audit_existing_scaffold(
                        optimized_resume,
                        evidence_source,
                        optimized_bindings,
                    )
                    baseline_recall = float(
                        query_fastpath_diagnostics.get("after_atomic_recall") or 0.0
                    )
                    if (
                        optimized_diagnostics.get("accepted")
                        and float(optimized_diagnostics.get("after_atomic_recall") or 0.0)
                        >= baseline_recall
                    ):
                        resume = optimized_resume
                        query_fastpath_bindings = optimized_bindings
                        optimizer_removed = optimized_removed
                        optimizer_changes = _bullet_rewrite_changes(
                            before_optimizer,
                            optimized_resume,
                        )
                        logger.info(
                            "V2 | Dense-query narrative pass accepted: records=%d "
                            "rewrites=%d recall=%.1f%%",
                            len(narrative_record_keys),
                            len(optimizer_changes),
                            float(optimized_diagnostics["after_atomic_recall"]) * 100,
                        )
                    else:
                        logger.warning(
                            "V2 | Dense-query narrative pass reverted: accepted=%s "
                            "recall=%.1f%% baseline=%.1f%%",
                            bool(optimized_diagnostics.get("accepted")),
                            float(optimized_diagnostics.get("after_atomic_recall") or 0.0) * 100,
                            baseline_recall * 100,
                        )
            elif narrative_record_keys:
                logger.info(
                    "V2 | Dense-query narrative pass disabled after no-gain "
                    "validation; retained audited source wording"
                )
            # Returning here prevents the normal no-CV recovery tail from
            # rebinding the same long profile and invoking the compiler again.
            resume_dict = _canonical_to_v1_format(resume)
            final_result = VerifiedResult(
                resume=resume,
                changes=[Change(
                    path="*",
                    action="replace",
                    reason=(
                        "Generated from an evidence-gated query fact scaffold; "
                        "free-form Composer was unnecessary"
                    ),
                )] + optimizer_changes + [
                    Change(
                        path=path,
                        action="remove",
                        reason="No candidate evidence binding",
                    )
                    for path in grounded_fallback_removed + optimizer_removed
                ],
                resume_dict=resume_dict,
                evidence_bindings=query_fastpath_bindings,
            )
            trace_event(
                "v2_final",
                result=final_result,
                evidence_removed=grounded_fallback_removed,
                query_scaffold_fastpath=True,
            )
            logger.info(
                "V2 | Total: %.1fs (audited dense-query scaffold)",
                time.perf_counter() - t_start,
            )
            return final_result

        if not query_compiler_fastpath:
            resume = compose_from_query(query_text, jd_text)
            trace_event("generate_composer_assembled", resume=resume)

        if candidate_evidence:
            before_recovery = resume.model_dump()
            resume = _merge_query_fallback_sections(resume, grounded_fallback)
            resume, structure_recovery = _recover_grounded_source_structure(
                resume,
                grounded_fallback,
                evidence_source,
            )
            used_fallback = resume.model_dump() != before_recovery
            if structure_recovery.total:
                logger.info(
                    "V2 | Query structure recovery: fields=%d bullets=%d records=%d values=%d",
                    structure_recovery.filled_fields,
                    structure_recovery.appended_bullets,
                    structure_recovery.appended_records,
                    structure_recovery.appended_values,
                )
        if _is_empty_resume(resume) and query_text.strip():
            logger.warning("Generate composer produced an empty resume; using deterministic query fallback")
            resume = grounded_fallback
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
        resume, atom_provenance = _atomize_resume_bullets(resume)
        trusted_rewrites = dict(atom_provenance)
        trusted_outputs = {
            path: value
            for path in trusted_rewrites
            if (value := _bullet_path_value(resume, path)) is not None
        }
        optimizer_changes: list[Change] = []
        if _needs_optimizer(
            resume,
            audited_source_scaffold=query_compiler_fastpath,
        ):
            before_optimizer = resume.model_copy(deep=True)
            optimization = optimize_resume_with_provenance(resume, jd_text)
            resume = optimization.resume
            for path, source_value in optimization.trusted_rewrites.items():
                trusted_rewrites[path] = _expand_optimizer_provenance(
                    path,
                    source_value,
                    before_optimizer,
                    atom_provenance,
                )
                output_value = _bullet_path_value(resume, path)
                if output_value is not None:
                    trusted_outputs[path] = output_value
            resume = _ground_optimizer_output(
                before_optimizer,
                resume,
                query_text,
                trusted_rewrites=trusted_rewrites,
            )
            trusted_rewrites = _filter_trusted_rewrites(
                resume, trusted_rewrites, trusted_outputs,
            )
            optimizer_changes = _bullet_rewrite_changes(before_optimizer, resume)
        else:
            if query_compiler_fastpath:
                logger.info(
                    "V2 | Optimizer skipped: audited dense-query scaffold "
                    "already preserves complete source clauses"
                )
            else:
                logger.info("V2 | Optimizer skipped: no factual bullets")
        resume = _ground_bullets(
            resume,
            query_text,
            trusted_rewrites=trusted_rewrites,
        )
        trusted_rewrites = _filter_trusted_rewrites(
            resume, trusted_rewrites, trusted_outputs,
        )
        resume = validate_resume(resume, source_text=query_text)
        trusted_rewrites = _filter_trusted_rewrites(
            resume, trusted_rewrites, trusted_outputs,
        )

        resume, evidence_bindings, evidence_removed = enforce_resume_evidence(
            resume,
            evidence_source,
            trusted_rewrites=trusted_rewrites,
        )
        if not _is_empty_resume(grounded_fallback):
            resume = _merge_query_fallback_sections(resume, grounded_fallback)
            resume, evidence_bindings, second_removed = enforce_resume_evidence(
                resume,
                evidence_source,
                trusted_rewrites=trusted_rewrites,
            )
            evidence_removed.extend(second_removed)
        before_ledger_coverage, _ = _deterministic_source_coverage(
            evidence_source,
            evidence_bindings,
        )
        before_ledger_resume = resume.model_copy(deep=True)
        ledger_resume, ledger_recovery, changed_paths = _recover_missing_record_facts(
            resume,
            evidence_source,
            evidence_bindings,
        )
        if ledger_recovery.total:
            ledger_trusted = dict(trusted_rewrites)
            for path in changed_paths:
                ledger_trusted.pop(path, None)
            ledger_resume, ledger_bindings, ledger_removed = enforce_resume_evidence(
                ledger_resume,
                evidence_source,
                trusted_rewrites=ledger_trusted,
            )
            ledger_coverage, _ = _deterministic_source_coverage(
                evidence_source,
                ledger_bindings,
            )
            if (
                ledger_coverage > before_ledger_coverage
                and _resume_claims_are_preserved(before_ledger_resume, ledger_resume)
            ):
                resume = ledger_resume
                evidence_bindings = ledger_bindings
                trusted_rewrites = ledger_trusted
                evidence_removed.extend(ledger_removed)
                logger.info(
                    "V2 | Query fact-ledger recovery: %.1f%% -> %.1f%%; appended=%d expanded=%d",
                    before_ledger_coverage * 100,
                    ledger_coverage * 100,
                    ledger_recovery.appended_bullets,
                    ledger_recovery.expanded_bullets,
                )
        # Finalize legacy structure and its evidence-derived summary before
        # the append-only compiler.  Compacting afterwards would rewrite the
        # accepted candidate and bypass the compiler's atomic safety audit.
        resume = _compact_canonical(resume)
        trusted_rewrites = _filter_trusted_rewrites(
            resume, trusted_rewrites, trusted_outputs,
        )
        compiler_resume, compiler_bindings, compiler_removed, compiler_diagnostics = (
            _apply_fact_compiler_candidate(
                resume,
                evidence_source,
                scaffold=(grounded_fallback if not _is_empty_resume(grounded_fallback) else None),
                trusted_rewrites=trusted_rewrites,
            )
        )
        if compiler_diagnostics.get("accepted"):
            resume = compiler_resume
            evidence_bindings = compiler_bindings
            evidence_removed.extend(compiler_removed)
            trusted_rewrites = _filter_trusted_rewrites(
                resume, trusted_rewrites, trusted_outputs,
            )
        # Recompute only after unsupported JD-derived records have been
        # removed. Otherwise temporary model output suppresses the framework
        # and produces an almost blank document.
        has_profile_records = _has_candidate_profile(resume)
        if not has_profile_records:
            resume.summary = ""
        evidence_bindings = bind_resume_evidence(
            resume,
            evidence_source,
            trusted_rewrites=trusted_rewrites,
        )
        logger.info("V2 | Evidence bindings: %d", len(evidence_bindings))
        resume_dict = _canonical_to_v1_format(resume)
        if not has_profile_records:
            resume_dict["framework"] = _empty_profile_framework(resume.meta.target_role)
        final_result = VerifiedResult(
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
        trace_event(
            "v2_final",
            result=final_result,
            evidence_removed=evidence_removed,
            framework_mode=not has_profile_records,
        )
        return final_result

    # ── Has CV: full Composer → Verifier → Optimizer pipeline ──
    source = build_source_bundle(cv_text, query_text, jd_text)
    trace_event("source_bundle", source=source)
    logger.info("V2 | SourceBundle: %d blocks (%.1fs)",
                len(source.blocks), time.perf_counter() - t_start)

    candidate_source_blocks = candidate_blocks(source)
    candidate_evidence = "\n".join(block.text for block in candidate_source_blocks)
    compiler_scaffold_result = None
    compiler_fastpath_result: VerifiedResult | None = None
    if _fact_compiler_mode() != "legacy":
        try:
            compiler_scaffold_result = _grounded_source_fallback(
                cv_text,
                query_text,
                jd_text,
                source,
                candidate_evidence,
            )
            trace_event(
                "fact_compiler_scaffold",
                source_coverage=compiler_scaffold_result[1],
                missing_source_facts=compiler_scaffold_result[2],
            )
            if (
                _fact_compiler_mode() == "on"
                and len(candidate_source_blocks) >= _FACT_COMPILER_FASTPATH_MIN_BLOCKS
            ):
                (
                    fastpath_resume,
                    fastpath_bindings,
                    _fastpath_removed,
                    fastpath_diagnostics,
                ) = _apply_fact_compiler_candidate(
                    compiler_scaffold_result[0].resume,
                    source,
                    scaffold=compiler_scaffold_result[0].resume,
                    allow_reordered_record_bullets=True,
                )
                fastpath_recall = float(
                    fastpath_diagnostics.get("after_atomic_recall") or 0.0
                )
                fastpath_has_history = _has_structured_history(fastpath_resume)
                fastpath_allows_missing_school = not any(
                    fact.fact_eligible
                    and fact.section_hint == "education"
                    and _SCHOOL_SUFFIX.search(fact.verbatim_text)
                    and not _FALLBACK_PLACEHOLDER_TOKEN.search(fact.verbatim_text)
                    for fact in source.fact_units
                )
                fastpath_structure_safe = _fallback_is_structurally_safe(
                    fastpath_resume,
                    allow_period_only_experience=True,
                    allow_missing_education_school=fastpath_allows_missing_school,
                    allow_partial_record_identity=True,
                )
                if (
                    fastpath_diagnostics.get("accepted")
                    and fastpath_recall >= _FACT_COMPILER_FASTPATH_MIN_ATOMIC_RECALL
                    and fastpath_has_history
                    and fastpath_structure_safe
                ):
                    compiler_fastpath_result = compiler_scaffold_result[0].model_copy(
                        deep=True,
                    )
                    compiler_fastpath_result.resume = fastpath_resume
                    compiler_fastpath_result.evidence_bindings = fastpath_bindings
                    compiler_fastpath_result.changes.insert(0, Change(
                        path="*",
                        action="replace",
                        reason=(
                            "Used audited fact-compiler scaffold before free-form "
                            "Composer for a long source"
                        ),
                    ))
                    trace_event(
                        "fact_compiler_fastpath",
                        candidate_blocks=len(candidate_source_blocks),
                        diagnostics=fastpath_diagnostics,
                    )
                    logger.warning(
                        "V2 | Long-source Composer skipped: audited scaffold "
                        "atomic recall %.1f%% across %d candidate blocks",
                        float(fastpath_diagnostics["after_atomic_recall"]) * 100,
                        len(candidate_source_blocks),
                    )
                else:
                    logger.warning(
                        "V2 | Long-source fact-compiler scaffold rejected: "
                        "accepted=%s recall=%.1f%% history=%s structure_safe=%s "
                        "allow_missing_school=%s candidate_blocks=%d",
                        bool(fastpath_diagnostics.get("accepted")),
                        fastpath_recall * 100,
                        fastpath_has_history,
                        fastpath_structure_safe,
                        fastpath_allows_missing_school,
                        len(candidate_source_blocks),
                    )
        except Exception as exc:
            logger.warning("V2 | Fact compiler scaffold failed: %s", exc)

    t_verifier = time.perf_counter()
    if compiler_fastpath_result is not None:
        result = compiler_fastpath_result
        trace_event("composer_skipped_for_fact_compiler", result=result)
    else:
        t_composer = time.perf_counter()
        draft = compose_resume(source)
        trace_event("composer_assembled_draft", draft=draft)
        logger.info("V2 | Composer done: %d edu, %d exp, %d res, %d proj (%.1fs)",
                    len(draft.education), len(draft.experience),
                    len(draft.research), len(draft.projects),
                    time.perf_counter() - t_composer)

        t_verifier = time.perf_counter()
        result = _deterministic_verify_draft(source, draft)
        trace_event(
            "deterministic_verifier_result",
            accepted=result is not None,
            result=result,
        )
        if result is None:
            fallback_candidate = compiler_scaffold_result
            if fallback_candidate is None:
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
                and _fallback_is_structurally_safe(fallback_candidate[0].resume)
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
    trace_event("verifier_selected_result", result=result)
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

    # Coverage cannot be recovered by a wording-only optimizer.  Ground the
    # selected draft first, then recover each safe source value independently.
    # A whole missing record no longer has to depress global recall by ten
    # points before it is considered, but it must have a unique source owner
    # plus explicit identity, period, and body evidence.
    grounded_before, pre_optimizer_bindings, pre_optimizer_removed = enforce_resume_evidence(
        result.resume,
        source,
        allow_reordered_record_bullets=True,
    )
    grounded_before = _compact_canonical(grounded_before)
    pre_optimizer_bindings = bind_resume_evidence(grounded_before, source)
    pre_optimizer_coverage, pre_optimizer_missing = _deterministic_source_coverage(
        source,
        pre_optimizer_bindings,
    )
    result.resume = grounded_before
    result.changes.extend(
        Change(path=path, action="remove", reason="No candidate evidence binding")
        for path in pre_optimizer_removed
    )
    if _missing_units_need_source_recovery(source, pre_optimizer_missing):
        try:
            fallback_result, fallback_coverage, _fallback_missing = _grounded_source_fallback(
                cv_text,
                query_text,
                jd_text,
                source,
                candidate_evidence,
            )
            recovered_structure, structure_recovery = _recover_grounded_source_structure(
                result.resume,
                fallback_result.resume,
                source,
            )
            if structure_recovery.total:
                recovered_structure, structure_bindings, structure_removed = enforce_resume_evidence(
                    recovered_structure,
                    source,
                    allow_reordered_record_bullets=True,
                )
                recovered_structure = _compact_canonical(recovered_structure)
                structure_bindings = bind_resume_evidence(recovered_structure, source)
                structure_coverage, _ = _deterministic_source_coverage(
                    source,
                    structure_bindings,
                )
                if (
                    structure_coverage >= pre_optimizer_coverage
                    and _resume_claims_are_preserved(result.resume, recovered_structure)
                ):
                    result.resume = recovered_structure
                    result.evidence_bindings = structure_bindings
                    result.changes.extend(
                        Change(path=path, action="remove", reason="No candidate evidence binding")
                        for path in structure_removed
                    )
                    logger.warning(
                        "V2 | Recovered grounded source structure before optimizer: %.1f%% -> %.1f%% (fallback %.1f%%); fields=%d bullets=%d records=%d values=%d",
                        pre_optimizer_coverage * 100,
                        structure_coverage * 100,
                        fallback_coverage * 100,
                        structure_recovery.filled_fields,
                        structure_recovery.appended_bullets,
                        structure_recovery.appended_records,
                        structure_recovery.appended_values,
                    )
        except Exception as exc:
            logger.warning("V2 | Pre-optimizer grounded structure recovery failed: %s", exc)

    # Recover record-owned source facts before wording optimization.  The old
    # post-optimizer-only placement preserved completeness, but the restored raw
    # clause never participated in record-level consolidation and therefore
    # looked visibly tacked on.  This pass is evidence-gated and cannot create
    # records or move a fact across source record boundaries.
    pre_ledger_bindings = bind_resume_evidence(result.resume, source)
    pre_ledger_coverage, _ = _deterministic_source_coverage(
        source,
        pre_ledger_bindings,
    )
    pre_ledger_resume = result.resume.model_copy(deep=True)
    recovered_resume, pre_ledger_recovery, _ = _recover_missing_record_facts(
        result.resume,
        source,
        pre_ledger_bindings,
    )
    if pre_ledger_recovery.total:
        recovered_resume, recovered_bindings, recovered_removed = enforce_resume_evidence(
            recovered_resume,
            source,
            allow_reordered_record_bullets=True,
        )
        recovered_resume = _compact_canonical(recovered_resume)
        recovered_bindings = bind_resume_evidence(recovered_resume, source)
        recovered_coverage, _ = _deterministic_source_coverage(
            source,
            recovered_bindings,
        )
        if (
            recovered_coverage > pre_ledger_coverage
            and _resume_claims_are_preserved(pre_ledger_resume, recovered_resume)
        ):
            result.resume = recovered_resume
            result.evidence_bindings = recovered_bindings
            result.changes.extend(
                Change(path=path, action="remove", reason="No candidate evidence binding")
                for path in recovered_removed
            )
            logger.warning(
                "V2 | Recovered record facts before optimizer: %.1f%% -> %.1f%%; appended=%d expanded=%d",
                pre_ledger_coverage * 100,
                recovered_coverage * 100,
                pre_ledger_recovery.appended_bullets,
                pre_ledger_recovery.expanded_bullets,
            )

    # Quality-v2 keeps the free-form Composer as the structural authority.
    # The compiler is allowed to close only uniquely owned record-body gaps,
    # cannot merge its fallback scaffold, cannot populate summary/skills or
    # create records, and still has to pass the atomic precision/ownership
    # transaction before the recovered clauses reach the wording optimizer.
    if pipeline_profile.record_compiler_recovery and record_recovery_allowed:
        (
            recovered_resume,
            recovered_bindings,
            recovered_removed,
            recovery_diagnostics,
        ) = _apply_fact_compiler_candidate(
            result.resume,
            source,
            scaffold=None,
            allow_reordered_record_bullets=True,
            mode_override="on",
            merge_scaffold=False,
            allowed_destinations=frozenset({
                "experience", "research", "activities", "projects",
            }),
            cleanup_presentation=False,
            require_identity_owner=True,
        )
        if recovery_diagnostics.get("accepted"):
            result.resume = recovered_resume
            result.evidence_bindings = recovered_bindings
            result.changes.extend(
                Change(path=path, action="remove", reason="No candidate evidence binding")
                for path in recovered_removed
            )
            logger.warning(
                "V2 | Quality-v2 recovered uniquely owned record facts before "
                "optimizer: %.1f%% -> %.1f%%; recovered=%d",
                float(recovery_diagnostics["before_atomic_recall"]) * 100,
                float(recovery_diagnostics["after_atomic_recall"]) * 100,
                int(recovery_diagnostics["recall_gain"]),
            )
        result.resume = _quality_v2_presentation_cleanup(result.resume)
    elif pipeline_profile.record_compiler_recovery:
        logger.info(
            "V2 | Record compiler recovery skipped: OCR text does not retain "
            "a trustworthy source-record layout"
        )
        trace_event(
            "record_compiler_recovery_skipped",
            reason="ocr_layout_unstable",
        )
    logger.info("V2 | Verifier done: %d edu, %d exp, %d res, %d changes (%.1fs)",
                len(result.resume.education), len(result.resume.experience),
                len(result.resume.research), len(result.changes),
                time.perf_counter() - t_verifier)

    # Freeze the evidence-gated records before any wording/ranking stage.  The
    # optimizer may rewrite bullets, but it is never authorized to make a whole
    # uniquely grounded source record disappear.
    record_guard_resume = result.resume.model_copy(deep=True)

    # Rank first so accepted rewrite paths remain stable in the final output.
    result.resume = _rank_resume_content(result.resume, jd_text or result.resume.meta.target_role)
    result.resume, atom_provenance = _atomize_resume_bullets(result.resume)
    t_optimizer = time.perf_counter()
    trusted_rewrites: dict[str, str] = dict(atom_provenance)
    trusted_outputs = {
        path: value
        for path in trusted_rewrites
        if (value := _bullet_path_value(result.resume, path)) is not None
    }
    # The bounded narrative planner is an experimental surface-realization
    # layer, not part of the factual compiler contract.  Keep it off by
    # default until a held-out paired run demonstrates a factuality-neutral
    # readability gain.  This also makes the safe compiler scaffold the
    # production fallback instead of letting a development bad-case rule alter
    # every dense CV.
    cv_narrative_enabled = pipeline_profile.cv_narrative
    narrative_record_keys = (
        select_narrative_record_keys(result.resume)
        if compiler_fastpath_result is not None and cv_narrative_enabled
        else None
    )
    if _needs_optimizer(
        result.resume,
        audited_source_scaffold=compiler_fastpath_result is not None,
        narrative_record_keys=narrative_record_keys,
    ):
        before_optimizer = result.resume.model_copy(deep=True)
        trace_event("optimizer_input_resume", resume=before_optimizer, jd_text=jd_text)
        optimizer_bindings = bind_resume_evidence(
            result.resume,
            source,
            trusted_rewrites=atom_provenance,
        )
        optimization = optimize_resume_with_provenance(
            result.resume,
            jd_text,
            evidence_bindings=optimizer_bindings,
            record_keys=narrative_record_keys,
        )
        trace_event("optimizer_outcome", outcome=optimization)
        result.resume = optimization.resume
        for path, source_value in optimization.trusted_rewrites.items():
            # A rewritten atom binds through the original compound source
            # bullet when atomization created it; otherwise use the optimizer's
            # verified pre-edit wording.
            trusted_rewrites[path] = _expand_optimizer_provenance(
                path,
                source_value,
                before_optimizer,
                atom_provenance,
            )
            output_value = _bullet_path_value(result.resume, path)
            if output_value is not None:
                trusted_outputs[path] = output_value
        result.resume = _ground_optimizer_output(
            before_optimizer,
            result.resume,
            candidate_evidence,
            trusted_rewrites=trusted_rewrites,
        )
        trusted_rewrites = _filter_trusted_rewrites(
            result.resume, trusted_rewrites, trusted_outputs,
        )
        result.changes.extend(_bullet_rewrite_changes(before_optimizer, result.resume))
    else:
        if compiler_fastpath_result is not None:
            logger.info(
                "V2 | Optimizer skipped: audited long-source scaffold has no "
                "bounded multi-clause record requiring narrative grouping"
            )
        else:
            logger.info("V2 | Optimizer skipped: no factual bullets")
    result.resume = _ground_bullets(
        result.resume,
        candidate_evidence,
        trusted_rewrites=trusted_rewrites,
    )
    trusted_rewrites = _filter_trusted_rewrites(
        result.resume, trusted_rewrites, trusted_outputs,
    )
    logger.info("V2 | Optimizer done (%.1fs)", time.perf_counter() - t_optimizer)

    # Deterministic repair uses candidate evidence only.  JD schools, employers
    # and dates must never backfill candidate fields.
    _source_for_validate = candidate_evidence
    result.resume = validate_resume(result.resume, source_text=_source_for_validate)
    trusted_rewrites = _filter_trusted_rewrites(
        result.resume, trusted_rewrites, trusted_outputs,
    )
    result.resume, post_gate_bindings, evidence_removed = enforce_resume_evidence(
        result.resume,
        source,
        trusted_rewrites=trusted_rewrites,
    )
    trace_event(
        "post_optimizer_evidence_gate",
        resume=result.resume,
        removed_paths=evidence_removed,
        trusted_rewrites=trusted_rewrites,
    )
    result.changes.extend(
        Change(path=path, action="remove", reason="No candidate evidence binding")
        for path in evidence_removed
    )
    guarded_resume, record_guard_recovery = _recover_grounded_source_structure(
        result.resume,
        record_guard_resume,
        source,
    )
    if record_guard_recovery.total:
        guarded_resume, guarded_bindings, guarded_removed = enforce_resume_evidence(
            guarded_resume,
            source,
            trusted_rewrites=trusted_rewrites,
            allow_reordered_record_bullets=True,
        )
        guarded_resume = _compact_canonical(guarded_resume)
        guarded_bindings = bind_resume_evidence(
            guarded_resume,
            source,
            trusted_rewrites=trusted_rewrites,
        )
        current_guard_coverage, _ = _deterministic_source_coverage(
            source,
            post_gate_bindings,
        )
        restored_guard_coverage, _ = _deterministic_source_coverage(
            source,
            guarded_bindings,
        )
        if (
            restored_guard_coverage >= current_guard_coverage
            and _resume_claims_are_preserved(result.resume, guarded_resume)
        ):
            logger.warning(
                "V2 | Restored grounded records after optimizer: %.1f%% -> %.1f%%; "
                "fields=%d bullets=%d records=%d values=%d",
                current_guard_coverage * 100,
                restored_guard_coverage * 100,
                record_guard_recovery.filled_fields,
                record_guard_recovery.appended_bullets,
                record_guard_recovery.appended_records,
                record_guard_recovery.appended_values,
            )
            result.resume = guarded_resume
            post_gate_bindings = guarded_bindings
            result.changes.extend(
                Change(path=path, action="remove", reason="No candidate evidence binding")
                for path in guarded_removed
            )
            trace_event(
                "grounded_record_guard",
                recovered=record_guard_recovery,
                source_coverage=restored_guard_coverage,
            )
    before_ledger_coverage, _ = _deterministic_source_coverage(
        source,
        post_gate_bindings,
    )
    before_ledger_resume = result.resume.model_copy(deep=True)
    ledger_resume, ledger_recovery, changed_paths = _recover_missing_record_facts(
        result.resume,
        source,
        post_gate_bindings,
    )
    if ledger_recovery.total:
        ledger_trusted = dict(trusted_rewrites)
        for path in changed_paths:
            ledger_trusted.pop(path, None)
        ledger_resume, ledger_bindings, ledger_removed = enforce_resume_evidence(
            ledger_resume,
            source,
            trusted_rewrites=ledger_trusted,
            allow_reordered_record_bullets=True,
        )
        ledger_resume = _compact_canonical(ledger_resume)
        ledger_bindings = bind_resume_evidence(
            ledger_resume,
            source,
            trusted_rewrites=ledger_trusted,
        )
        ledger_coverage, _ = _deterministic_source_coverage(
            source,
            ledger_bindings,
        )
        if (
            ledger_coverage > before_ledger_coverage
            and _resume_claims_are_preserved(before_ledger_resume, ledger_resume)
        ):
            result.resume = ledger_resume
            result.evidence_bindings = ledger_bindings
            trusted_rewrites = ledger_trusted
            result.changes.extend(
                Change(path=path, action="remove", reason="No candidate evidence binding")
                for path in ledger_removed
            )
            logger.warning(
                "V2 | Fact-ledger recovery: %.1f%% -> %.1f%%; appended=%d expanded=%d",
                before_ledger_coverage * 100,
                ledger_coverage * 100,
                ledger_recovery.appended_bullets,
                ledger_recovery.expanded_bullets,
            )
    result.resume = _compact_canonical(result.resume)
    trusted_rewrites = _filter_trusted_rewrites(
        result.resume, trusted_rewrites, trusted_outputs,
    )
    result.evidence_bindings = bind_resume_evidence(
        result.resume,
        source,
        trusted_rewrites=trusted_rewrites,
    )
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
        # Recover only the still-missing source facts.  Existing records and
        # accepted optimizer wording keep their positions and provenance.
        try:
            fallback_result, fallback_coverage, fallback_missing = _grounded_source_fallback(
                cv_text,
                query_text,
                jd_text,
                source,
                candidate_evidence,
            )
            if (
                fallback_coverage > coverage
                and _fallback_is_structurally_safe(
                    fallback_result.resume,
                    allow_period_only_experience=True,
                )
            ):
                merged_resume, recovery = _recover_grounded_source_structure(
                    result.resume,
                    fallback_result.resume,
                    source,
                )
                if recovery.total:
                    merged_trusted_rewrites = _filter_trusted_rewrites(
                        merged_resume,
                        trusted_rewrites,
                        trusted_outputs,
                    )
                    merged_resume, _, merged_removed = enforce_resume_evidence(
                        merged_resume,
                        source,
                        trusted_rewrites=merged_trusted_rewrites,
                        allow_reordered_record_bullets=True,
                    )
                    merged_resume = _compact_canonical(merged_resume)
                    merged_trusted_rewrites = _filter_trusted_rewrites(
                        merged_resume,
                        merged_trusted_rewrites,
                        trusted_outputs,
                    )
                    merged_bindings = bind_resume_evidence(
                        merged_resume,
                        source,
                        trusted_rewrites=merged_trusted_rewrites,
                    )
                    merged_coverage, merged_missing = _deterministic_source_coverage(
                        source,
                        merged_bindings,
                    )
                    if merged_coverage > coverage:
                        logger.warning(
                            "V2 | Recovered missing source facts after optimizer: %.1f%% -> %.1f%%; fields=%d bullets=%d records=%d values=%d",
                            coverage * 100,
                            merged_coverage * 100,
                            recovery.filled_fields,
                            recovery.appended_bullets,
                            recovery.appended_records,
                            recovery.appended_values,
                        )
                        result.resume = merged_resume
                        result.evidence_bindings = merged_bindings
                        trusted_rewrites = merged_trusted_rewrites
                        result.changes.extend(
                            Change(
                                path=path,
                                action="remove",
                                reason="No candidate evidence binding",
                            )
                            for path in merged_removed
                        )
                        coverage = merged_coverage
                        missing_blocks = merged_missing
        except Exception as exc:
            logger.warning("V2 | Source coverage fallback failed: %s", exc)

    compiler_resume, compiler_bindings, compiler_removed, compiler_diagnostics = (
        _apply_fact_compiler_candidate(
            result.resume,
            source,
            scaffold=(
                compiler_scaffold_result[0].resume
                if compiler_scaffold_result is not None
                else None
            ),
            trusted_rewrites=trusted_rewrites,
            allow_reordered_record_bullets=True,
        )
    )
    if compiler_diagnostics.get("accepted"):
        result.resume = compiler_resume
        result.evidence_bindings = compiler_bindings
        result.changes.extend(
            Change(path=path, action="remove", reason="No candidate evidence binding")
            for path in compiler_removed
        )
        trusted_rewrites = _filter_trusted_rewrites(
            result.resume, trusted_rewrites, trusted_outputs,
        )
        coverage, missing_blocks = _deterministic_source_coverage(
            source,
            result.evidence_bindings,
        )

    summary_restored, restored_summary_facts = _restore_attested_source_summary(
        result.resume,
        source,
    )
    if restored_summary_facts:
        result.resume = summary_restored
        result.evidence_bindings = bind_resume_evidence(
            result.resume,
            source,
            trusted_rewrites=trusted_rewrites,
        )
        coverage, missing_blocks = _deterministic_source_coverage(
            source,
            result.evidence_bindings,
        )
        trace_event(
            "attested_summary_recovery",
            restored=restored_summary_facts,
            source_coverage=coverage,
        )
        logger.info(
            "V2 | Restored %d exact source summary fact(s); coverage=%.1f%%",
            len(restored_summary_facts),
            coverage * 100,
        )
    # Template placeholders are never candidate facts and must not leak merely
    # because the optional fact compiler is disabled.  This cleanup is local
    # and deterministic, so apply it on every production path.
    final_resume, final_placeholder_paths = sanitize_resume_placeholders(
        result.resume,
    )
    if final_placeholder_paths:
        result.resume = final_resume
        result.evidence_bindings = bind_resume_evidence(
            result.resume,
            source,
            trusted_rewrites=trusted_rewrites,
        )
        coverage, missing_blocks = _deterministic_source_coverage(
            source,
            result.evidence_bindings,
        )
        trace_event(
            "final_placeholder_cleanup",
            paths=final_placeholder_paths,
            source_coverage=coverage,
        )
    if pipeline_profile.record_compiler_recovery:
        quality_cleaned = _quality_v2_presentation_cleanup(result.resume)
        if quality_cleaned != result.resume:
            result.resume = quality_cleaned
            trusted_rewrites = _filter_trusted_rewrites(
                result.resume, trusted_rewrites, trusted_outputs,
            )
            result.evidence_bindings = bind_resume_evidence(
                result.resume,
                source,
                trusted_rewrites=trusted_rewrites,
            )
            coverage, missing_blocks = _deterministic_source_coverage(
                source,
                result.evidence_bindings,
            )
            trace_event(
                "quality_v2_presentation_cleanup",
                source_coverage=coverage,
            )
    if missing_blocks:
        logger.info("V2 | Final source coverage %.1f%%, missing=%s", coverage * 100, missing_blocks[:8])
    result.changes = [
        change for change in result.changes
        if _change_path_exists(result.resume, change.path)
    ]
    result.resume_dict = _canonical_to_v1_format(result.resume)
    logger.info("V2 | Evidence bindings: %d", len(result.evidence_bindings))

    logger.info("V2 | Total: %.1fs (Composer+Verifier+Validate+Format)",
                time.perf_counter() - t_start)

    trace_event(
        "v2_final",
        result=result,
        source_coverage=coverage,
        missing_source_blocks=missing_blocks,
    )

    return result
