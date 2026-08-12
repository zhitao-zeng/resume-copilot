"""V2 Pipeline orchestration.

Layers: SourceAdapter → Composer → Verifier → Optimizer → Validator
"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from dataclasses import dataclass

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
import resume_product_logic as product_logic
from diagnostic_trace import trace_event

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
_SUMMARY_MAX_CHARS = 220
_SUMMARY_MAX_SENTENCES = 5
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
        for item in re.split(r"[\n。；;,，]+", str(text or ""))
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
        if (
            len(sentence) >= 12
            and not _SUMMARY_SUBJECTIVE.search(sentence)
            and not _INCOMPLETE_TEXT_TAIL.search(sentence)
        ):
            candidates.append(sentence)

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
    if experience_bits:
        seniority = resume.meta.work_experience.strip()
        if seniority and re.fullmatch(r"\d+(?:\.\d+)?\s*(?:年|个月|月)", seniority):
            seniority += "经验"
        prefix = f"有{seniority}，" if seniority else ""
        candidates.append(prefix + "曾任" + "、".join(experience_bits))
    if role_only_bits:
        candidates.append("工作或实习经历包括" + "、".join(role_only_bits))

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

    if not experience_bits and not role_only_bits and not research_bits:
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
    r"学生会|社团|委员会|事务所|律所|银行|基金会|工作室|团队|基地)(?:\d+)?$"
)
_SCHOOL_SUFFIX = re.compile(r"(?:大学|学院|学校|研究院)(?:\d+)?")
_NON_SCHOOL_SENTENCE = re.compile(
    r"(?:准备找|求职|岗位|工作|简历|马上要毕业|已经毕业|毕业了|开始准备|"
    r"相关的工作|专业硕士|专业博士)"
)
_NON_MAJOR_SENTENCE = re.compile(
    r"(?:准备找|求职|岗位|工作|简历|马上要毕业|已经毕业|毕业了|开始准备|"
    r"最近开始|具备|熟悉|掌握|负责|参与|经验|能力|环境|定位|营销)"
)


def _clean_structured_organization(value: str) -> str:
    """Remove narrative wrappers from an otherwise explicit organization."""

    text = re.sub(r"\s+", " ", str(value or "")).strip(" ，,。；;")
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


def _clean_role_title(value: str, *, explicit: bool = False) -> str:
    """Normalize role grammar and reject a task accidentally bound as a title.

    Evidence presence alone is insufficient for typed fields: ``单元测试`` may
    appear in a source bullet, but that does not make it a job title.  The guard
    is deliberately grammatical and narrow rather than an industry dictionary.
    Explicitly labelled fallback values still have their wrappers normalized;
    canonical model output receives the stricter semantic check below.
    """

    text = re.sub(r"\s+", " ", str(value or "")).strip(" ，,。；;|｜:：")
    text = re.sub(r"^(?:岗位|职位|角色|职务)\s*[:：]\s*", "", text)
    text = _ROLE_WRAPPER.sub("", text).strip(" ，,。；;|｜:：")
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
    school_match = _SCHOOL_SUFFIX.search(school)
    if school_match:
        school = school[:school_match.end()].strip()
    elif _NON_SCHOOL_SENTENCE.search(school) or len(school) > 50:
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


def _compact_canonical(resume: CanonicalResume) -> CanonicalResume:
    """Remove blank records/items left by model repair or leakage cleanup."""

    data = resume.model_dump()
    meta = data.get("meta") or {}
    if isinstance(meta, dict):
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
                            item["bullets"].insert(0, original_role)
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
        if section == "education":
            cleaned = _coalesce_compatible_education(cleaned)
            cleaned = _drop_subsumed_education(cleaned)
        data[section] = cleaned

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
    data["summary"] = str(data.get("summary", "") or "").strip()
    compacted = CanonicalResume.model_validate(data)
    compacted.summary = _build_evidence_summary(compacted)
    return compacted


_FALLBACK_PERIOD = re.compile(
    r"(?:(?:19|20)\d{2}(?:(?:[./-]\d{1,2})|(?:年\d{1,2}月?))?|(?:0?[1-9]|1[0-2])[-/](?:19|20)\d{2})"
    r"\s*(?:[-—~至到]\s*(?:(?:19|20)\d{2}(?:(?:[./-]\d{1,2})|(?:年\d{1,2}月?))?|"
    r"(?:0?[1-9]|1[0-2])[-/](?:19|20)\d{2}|今|至今|现在))?"
)
_FALLBACK_ORGANIZATION = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9·.&（）()_-]{0,40}?(?:大学|学院|学校|医院|公司|企业|集团|"
    r"中学|小学|幼儿园|研究院|实验室|中心|部门|协会|学会|学生会|社团|委员会|事务所|律所|银行|"
    r"基金会|工作室|团队|基地)(?:\d+)?"
)
_FALLBACK_ROLE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9/+.#_-]{0,24}(?:工程师|设计师|教师|老师|医生|医师|"
    r"护士|经理|主管|总监|主任|顾问|研究员|专员|助理|负责人|组长|队长|主席|"
    r"部长|干事|部员|成员|委员|志愿者|实习生|实习|见习|分析师|架构师|运营|产品|开发|测试|销售|讲师)"
)
_FALLBACK_DEGREE = re.compile(r"(?:博士研究生|硕士研究生|本科|硕士|博士|大专|专科|高中)(?:在读|毕业)?")
_FALLBACK_DATE_TOKEN = re.compile(
    r"^(?:19|20)\d{2}(?:(?:[./-]\d{1,2})|(?:年\d{1,2}月?))?\s*(?:至)?\s*$"
)
_FALLBACK_LABELED_SKILL = re.compile(
    r"^(?:[-•·]\s*)?(?P<label>技能|专业技能|工具|语言|语言能力)"
    r"\s*[:：]\s*(?P<value>.+)$"
)


def _first_match(pattern: re.Pattern[str], value: str) -> str:
    match = pattern.search(value)
    return match.group(0).strip() if match else ""


def _labeled_value(value: str, labels: tuple[str, ...]) -> str:
    joined = "|".join(re.escape(label) for label in labels)
    match = re.search(rf"(?:{joined})\s*[:：]\s*([^|｜，,；;\n]+)", value)
    return match.group(1).strip() if match else ""


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
    cleaned = re.sub(r"^[\s\-—~至到]+", "", cleaned)
    contextual = re.search(
        r"(?:就职于|任职于|供职于|在)\s*"
        r"([\u4e00-\u9fffA-Za-z0-9·.&（）()_-]{1,40}?(?:大学|学院|学校|医院|公司|企业|集团|"
        r"研究院|实验室|中心|部门|协会|学会|学生会|社团|委员会|事务所|银行|"
        r"基金会|工作室|团队))(?=工作|任职|担任|就职|[，,。；;\s])",
        cleaned,
    )
    if contextual:
        return contextual.group(1).strip()
    organization_match = _FALLBACK_ORGANIZATION.search(cleaned)
    organization = organization_match.group(0).strip() if organization_match else ""
    if (
        organization_match
        and organization.endswith("中心")
        and cleaned[organization_match.end():].startswith("医院")
    ):
        organization += "医院"
    # “做过两段律所实习” identifies an institution type, not a named
    # employer. Keep the role/duties and ask for the two firm names instead of
    # rendering a fake company called “律所”.
    if organization in {
        "律所", "事务所", "公司", "企业", "医院", "学校", "学院", "部门", "协会",
        "社团", "组织", "团队", "基地",
    }:
        return ""
    return organization


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
    return labeled or _first_match(_FALLBACK_PERIOD, joined)


def _role_from_text(value: str, organization: str = "") -> str:
    labeled = _labeled_value(value, ("岗位", "职位", "角色", "职务"))
    if labeled:
        return _clean_role_title(labeled, explicit=True)
    # Identity normally appears before the first duty clause. Strip the period
    # and already-grounded organization so a greedy role regex cannot absorb
    # them from OCR-compressed headers.
    header = re.split(r"[，,。；;]", value, maxsplit=1)[0]
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
    value = re.sub(r"^(?:(?:更)?适合|转向?|偏)\s*", "", str(value or "")).strip()
    value = re.sub(r"(?:相关)?(?:岗位|方向|工作)$", "", value).strip()
    return value


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
            and not re.match(r"^(?:\d{1,3}(?:[、)]|\.(?!\d))|[-*•·▪◦])\s*", line)
            and not _looks_like_record_body(line)
            and not _FALLBACK_PERIOD.fullmatch(line)
            and not _FALLBACK_ORGANIZATION.search(line)
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
    header_lines = [line for line in lines if line and not _looks_like_record_body(line)]
    header_text = " ｜ ".join(header_lines)
    period = _fallback_period_from_lines(
        lines,
        joined,
        ("时间", "任职时间", "起止时间", "项目时间"),
    )
    organization = _labeled_value(header_text, ("公司", "单位", "组织", "机构", "学校"))
    if not organization:
        organization = _organization_from_text(header_text)
    role = _clean_role_title(
        _labeled_value(header_text, ("岗位", "职位", "角色", "职务")),
        explicit=True,
    )
    if section == "activities" and not organization and not role:
        for line in header_lines:
            activity_identity = re.fullmatch(
                r"(.{2,40}?)(干事|部员|成员|委员)", line.strip(" ，,。；;|｜")
            )
            if activity_identity:
                organization = _clean_structured_organization(activity_identity.group(1))
                role = activity_identity.group(2)
                break
    if not role and section != "projects":
        for line in header_lines:
            role = _role_from_text(line, organization)
            if role:
                break
    identity_lines = [
        line.strip(" ，,。；;|｜")
        for line in header_lines
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
            (index for index, line in enumerate(header_lines) if role in line),
            len(header_lines),
        )
        preceding_identity = next(
            (
                candidate for candidate in identity_lines
                if header_lines.index(candidate) < role_line_index
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
        residual = re.sub(
            r"^(?:我|本人)?(?:目前|曾经|曾)?(?:在|于|就职于|任职于|供职于|担任|任职为|作为)\s*",
            "",
            residual,
        )
        residual = re.sub(r"^(?:我|本人)?(?:在|于|担任|任职|作为)+\s*$", "", residual)
        residual = re.sub(r"^[\s|｜,，;；:：\-—~至到]+|[\s|｜,，;；:：\-—~至到]+$", "", residual)
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
                and re.search(r"[。；;!?！？]$", residual)
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
        ]
        if header_candidates:
            candidate = header_candidates[0]
            candidate = candidate.replace(period, " ") if period else candidate
            candidate = candidate.replace(organization, " ") if organization else candidate
            candidate = candidate.replace(role, " ") if role else candidate
            name = re.sub(r"^[\s|｜,，;；:：\-]+|[\s|｜,，;；:：\-]+$", "", candidate)
    raw_name = name
    if raw_name in bullets:
        bullets.remove(raw_name)
    name = re.sub(
        r"^(?:我|本人)?(?:做过|参与|负责|主导|开发|设计|搭建|完成|开展)\s*",
        "",
        name,
    ).strip()
    if not role:
        for line in header_lines:
            if name and name in line:
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
    return {
        "name": name,
        "organization": organization,
        "role": role,
        "period": period,
        "bullets": bullets,
    }, []


def _fallback_education(lines: list[str]) -> tuple[dict, list[str]]:
    joined = re.sub(
        r"^(?:教育经历|教育背景|学历信息)\s*[:：]\s*",
        "",
        " ｜ ".join(lines),
    )
    school = _labeled_value(joined, ("学校", "院校")) or _organization_from_text(joined)
    degree = _labeled_value(joined, ("学历", "学位")) or _first_match(_FALLBACK_DEGREE, joined)
    major = _labeled_value(joined, ("专业",))
    if not major:
        major_match = re.search(
            r"(?:我是|就读于?|毕业于?|攻读)?\s*([^|｜，,。；;]{2,40}?)\s*专业(?:毕业|在读|学生)?",
            joined,
        )
        direction_match = re.search(
            r"(?:方向偏|研究方向(?:是|为)?|专业方向(?:是|为)?)\s*([^|｜，,。；;]{2,40})",
            joined,
        )
        if major_match:
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
        if education_extras:
            additional.setdefault("补充信息", []).extend(education_extras)

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
        for item in re.split(
            r"[、，,；;|｜]+|\s+/\s+|"
            r"(?<=[A-Za-z0-9\u4e00-\u9fff])(?:和|及|与)(?=[A-Za-z0-9\u4e00-\u9fff])",
            raw_values,
        ):
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
                and not _looks_like_record_body(item)
                and not re.search(r"[。；;]", item)
            ):
                skill_items.append({"name": item, "category": category})

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
        elif section in {"hobbies", "coursework"}:
            title = "兴趣爱好" if section == "hobbies" else "相关课程"
            cleaned = re.sub(r"^[^:：]{1,12}[:：]\s*", "", value).strip()
            if cleaned and not re.fullmatch(r"[-•·]?\s*[A-Za-z0-9._-]{3,}", cleaned):
                additional.setdefault(title, []).append(cleaned)
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


def _fallback_is_structurally_safe(
    resume: CanonicalResume,
    *,
    allow_period_only_experience: bool = False,
) -> bool:
    """Reject high-coverage fallbacks whose fields are only lexical matches."""

    required_fields = {
        "experience": ("organization", "role"),
        "research": ("institution", "topic"),
        "activities": ("organization", "role"),
        "projects": ("name",),
    }
    for education in resume.education:
        if not str(education.school or "").strip():
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
                if not period_only_experience:
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
    fallback_keys = _record_identity_keys(section, fallback_record)
    if not fallback_keys:
        return None
    for index, record in enumerate(records):
        if fallback_keys & _record_identity_keys(section, record):
            return index
    return None


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
    record_targets: dict[tuple[str, str], set[int]] = {}
    record_target_evidence: dict[tuple[str, int], dict[int, set[str]]] = {}
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
            if block.record_id:
                path = str(binding.path or "")
                priority = 3 if path.endswith(".period") else 2 if ".bullets[" in path else 1
                record_target_evidence.setdefault((section, index), {}).setdefault(
                    priority,
                    set(),
                ).add(block.record_id)
    for (section, index), by_priority in record_target_evidence.items():
        # Periods and already-grounded bullets identify record ownership more
        # reliably than repeated employer/title strings.  Using every identity
        # binding equally made two jobs at the same company contaminate each
        # other's target set and blocked otherwise safe recovery.
        strongest_ids = by_priority[max(by_priority)]
        if len(strongest_ids) != 1:
            continue
        record_id = next(iter(strongest_ids))
        record_targets.setdefault((section, record_id), set()).add(index)

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
) -> VerifiedResult:
    """Run the V2 5-layer pipeline. Returns VerifiedResult or fallback."""
    t_start = time.perf_counter()
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
        candidate_evidence = "\n".join(
            block.text for block in candidate_blocks(evidence_source)
        )
        resume = compose_from_query(query_text, jd_text)
        trace_event("generate_composer_assembled", resume=resume)
        used_fallback = False
        grounded_fallback = CanonicalResume()
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
            grounded_fallback, _, _ = enforce_resume_evidence(
                grounded_fallback,
                evidence_source,
            )
            # ``补充信息`` is parser scratch space, not a safe semantic type.
            # Composer may still emit a named long-tail section when it can
            # classify one from the same source.
            grounded_fallback.additional_sections.pop("补充信息", None)
            grounded_fallback = _compact_canonical(grounded_fallback)
            before_recovery = resume.model_dump()
            resume = _merge_query_fallback_sections(resume, grounded_fallback)
            used_fallback = resume.model_dump() != before_recovery
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
        if _needs_optimizer(resume):
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
        resume = _compact_canonical(resume)
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

    t_composer = time.perf_counter()
    draft = compose_resume(source)
    trace_event("composer_assembled_draft", draft=draft)
    logger.info("V2 | Composer done: %d edu, %d exp, %d res, %d proj (%.1fs)",
                len(draft.education), len(draft.experience),
                len(draft.research), len(draft.projects),
                time.perf_counter() - t_composer)

    t_verifier = time.perf_counter()
    candidate_evidence = "\n".join(block.text for block in candidate_blocks(source))
    result = _deterministic_verify_draft(source, draft)
    trace_event(
        "deterministic_verifier_result",
        accepted=result is not None,
        result=result,
    )
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

    # Coverage cannot be recovered by a wording-only optimizer.  Merge missing
    # deterministic facts before optimization, but never replace a valid draft
    # wholesale: doing so cancelled every useful Composer rewrite.
    pre_optimizer_bindings = bind_resume_evidence(result.resume, source)
    pre_optimizer_coverage, pre_optimizer_missing = _deterministic_source_coverage(
        source,
        pre_optimizer_bindings,
    )
    if pre_optimizer_missing and pre_optimizer_coverage < 0.80:
        try:
            fallback_result, fallback_coverage, _fallback_missing = _grounded_source_fallback(
                cv_text,
                query_text,
                jd_text,
                source,
                candidate_evidence,
            )
            if (
                fallback_coverage >= pre_optimizer_coverage + 0.10
                and _fallback_is_structurally_safe(
                    fallback_result.resume,
                    allow_period_only_experience=True,
                )
            ):
                merged_resume, recovery = _merge_source_recovery(
                    result.resume,
                    fallback_result.resume,
                )
                if recovery.total:
                    result.resume = merged_resume
                    result.evidence_bindings = bind_resume_evidence(result.resume, source)
                    logger.warning(
                        "V2 | Recovered missing source facts before optimizer: coverage %.1f%% -> fallback %.1f%%; fields=%d bullets=%d records=%d values=%d",
                        pre_optimizer_coverage * 100,
                        fallback_coverage * 100,
                        recovery.filled_fields,
                        recovery.appended_bullets,
                        recovery.appended_records,
                        recovery.appended_values,
                    )
        except Exception as exc:
            logger.warning("V2 | Pre-optimizer source fallback failed: %s", exc)

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
    logger.info("V2 | Verifier done: %d edu, %d exp, %d res, %d changes (%.1fs)",
                len(result.resume.education), len(result.resume.experience),
                len(result.resume.research), len(result.changes),
                time.perf_counter() - t_verifier)

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
    if _needs_optimizer(result.resume):
        before_optimizer = result.resume.model_copy(deep=True)
        trace_event("optimizer_input_resume", resume=before_optimizer, jd_text=jd_text)
        optimization = optimize_resume_with_provenance(result.resume, jd_text)
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
                fallback_coverage >= coverage + 0.10
                and _fallback_is_structurally_safe(
                    fallback_result.resume,
                    allow_period_only_experience=True,
                )
            ):
                merged_resume, recovery = _merge_source_recovery(
                    result.resume,
                    fallback_result.resume,
                    trusted_rewrites=trusted_rewrites,
                    allow_new_records=False,
                    restore_empty_sections=True,
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
