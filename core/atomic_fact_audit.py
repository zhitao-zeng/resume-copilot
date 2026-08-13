"""Deterministic atomic factuality and ownership auditing.

The audit is deliberately separate from generation.  It observes the final
canonical resume, exact source ``FactUnit`` objects and evidence bindings, but
never mutates resume content.  This separation lets the production pipeline
measure false additions, source omissions and cross-record attribution before
any repair policy is enabled.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from v2_schemas import CanonicalResume, EvidenceBinding, FactUnit, SourceBundle


_MAX_AUDIT_ITEMS = 12
_RECORD_PATH = re.compile(
    r"^(education|experience|research|activities|projects)\[(\d+)]"
)
_SENTENCE_SPLIT = re.compile(r"[。；;！？!?\r\n]+")
_LIST_SPLIT = re.compile(r"[，,、]+")
_CLAUSE_CONJUNCTION = re.compile(
    r"(?:并且|并|同时|此外|另外)(?=(?:负责|参与|主导|协助|支持|配合|推动|"
    r"组织|协调|执行|设计|开发|构建|实现|制定|管理|运营|分析|统计|策划|"
    r"培训|处理|研究|撰写|输出|交付|维护|优化|搭建|建立|开展|承担|提供|"
    r"跟进|编制|制作|诊断|治疗|授课|教学|复核|检索|调研|提升|提高|降低|"
    r"减少|增长|缩短|节省|达到|达成|获得|完成|形成|上线|发布|落地))"
)
_HARD_ANCHOR = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:%|万|亿|w|k|人|次|项|个|条|"
    r"篇|例|台|套|元|年|个月|月|日|小时|分钟|ms|s|qps|tps|fps|mb|gb)?|"
    r"[A-Za-z][A-Za-z0-9+.#/_-]{1,}",
    re.IGNORECASE,
)
_METRIC_ANCHOR = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:%|万|亿|w|k|人|次|项|个|条|"
    r"篇|例|台|套|元|小时|分钟|ms|s|qps|tps|fps|mb|gb)",
    re.IGNORECASE,
)
_LIST_MARKER = re.compile(
    r"^\s*(?:\(?\d{1,2}[.、．)）]|[一二三四五六七八九十]{1,3}[、.．)）])\s*"
)
_SUMMARY_DIRECTION = re.compile(
    r"^(?:求职|应聘|职业|目标)(?:方向|岗位|目标)|^希望(?:应聘|求职)",
    re.IGNORECASE,
)
_SUMMARY_PRESENTATION_SHELL = re.compile(
    r"^(?:代表成果|代表经历|相关经历|核心技能|教育背景|工作背景|项目背景|"
    r"工作或实习经历|工作经历|项目经历|校园经历|科研经历)\s*[:：]?\s*",
    re.IGNORECASE,
)
_DATE_TOKEN = re.compile(
    r"(?<!\d)(?:(?P<year1>(?:19|20)\d{2})\s*[-./年]\s*"
    r"(?P<month1>0?[1-9]|1[0-2])\s*月?|"
    r"(?P<month2>0?[1-9]|1[0-2])\s*[-./]\s*"
    r"(?P<year2>(?:19|20)\d{2})|"
    r"(?P<year_only>(?:19|20)\d{2})\s*年?)(?!\d)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AtomicClaim:
    atom_id: str
    path: str
    text: str
    record_key: str
    structural_category: str


@dataclass(frozen=True)
class AtomicMatch:
    status: str
    fact_ids: tuple[str, ...]
    confidence: float
    reason: str
    unmatched_anchors: tuple[str, ...] = ()


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^a-z0-9+.#/_\-\u4e00-\u9fff]+", "", text)


def _bigrams(value: str) -> set[str]:
    normalized = _normalize(value)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {
        normalized[index:index + 2]
        for index in range(len(normalized) - 1)
    }


def _anchors(value: str) -> set[str]:
    # A leading ``1.``/``（一）`` is document structure, not a candidate
    # metric.  Source fact units intentionally omit list markers, so treating
    # one as a hard anchor would reject an otherwise verbatim source bullet.
    # Only an explicit enumerator suffix is removed; dates and real quantities
    # such as ``2024年`` or ``100位患者`` remain immutable anchors.
    value = _LIST_MARKER.sub("", str(value or ""), count=1)
    result: set[str] = set()
    for match in _HARD_ANCHOR.finditer(value):
        anchor = _normalize(match.group(0))
        if not anchor:
            continue
        # OCR/native text commonly uses ``09-2022`` while the renderer emits
        # ``2022年9月``.  A leading zero on a one/two-digit integer is formatting,
        # not a different fact.  Units, percentages, years and identifiers with
        # letters keep their exact normalized form.
        if anchor.isdigit() and len(anchor) <= 2:
            anchor = str(int(anchor))
        result.add(anchor)
    return result


def _metric_anchors(value: str) -> list[str]:
    return list(dict.fromkeys(
        match.group(0).strip()
        for match in _METRIC_ANCHOR.finditer(str(value or ""))
        if match.group(0).strip()
    ))


def _date_signatures(value: str) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for match in _DATE_TOKEN.finditer(str(value or "")):
        year = match.group("year1") or match.group("year2") or match.group("year_only")
        month = match.group("month1") or match.group("month2") or ""
        result.add((str(year), str(int(month)) if month else ""))
    return result


def _date_component_anchors(value: str) -> set[str]:
    result: set[str] = set()
    for year, month in _date_signatures(value):
        result.update({year, f"{year}年"})
        if month:
            result.update({month, f"{month}月", month.zfill(2)})
    return result


def _unmatched_anchors(claim: str, source: str) -> tuple[str, ...]:
    unmatched = _anchors(claim) - _anchors(source)
    claim_dates = _date_signatures(claim)
    source_dates = _date_signatures(source)
    if claim_dates and claim_dates.issubset(source_dates):
        unmatched -= _date_component_anchors(claim)
    return tuple(sorted(unmatched))


def _date_only(value: str) -> bool:
    stripped = _DATE_TOKEN.sub("", str(value or ""))
    stripped = re.sub(r"(?:至今|现在|应届|至|到|[-—~年月日/.,，。；;()（）\s])", "", stripped)
    return not stripped


def _excerpt(value: str, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _record_key(path: str) -> str:
    match = _RECORD_PATH.match(path)
    return match.group(0) if match else ""


def _structural_category(path: str) -> str:
    if path.endswith(".organization") or path.endswith(".institution"):
        return "organization"
    if path.endswith(".role"):
        return "role"
    if path.endswith(".period") or path == "meta.work_experience":
        return "period"
    if path.startswith("education[") and not path.endswith(".period"):
        return "education"
    if path.startswith(("certifications[", "awards[")):
        return "credential"
    if path.startswith("skills.items["):
        return "skill_tool"
    return ""


def _atomic_segments(value: str, *, split_lists: bool) -> list[str]:
    """Split claims at explicit grammatical boundaries, not domain words."""

    result: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(str(value or "")):
        sentence = sentence.strip(" \t-•·▪◦")
        if not sentence:
            continue
        sentence = _CLAUSE_CONJUNCTION.sub("；", sentence)
        parts = _SENTENCE_SPLIT.split(sentence)
        for part in parts:
            candidates = _LIST_SPLIT.split(part) if split_lists else [part]
            for candidate in candidates:
                atom = candidate.strip(" \t-•·▪◦")
                if _normalize(atom):
                    result.append(atom)
    return result


def atomize_claim_text(value: str) -> list[str]:
    """Public, deterministic clause splitter shared by audit and repair."""

    return _atomic_segments(value, split_lists=True)


def _canonical_claims(resume: CanonicalResume) -> list[AtomicClaim]:
    raw: list[tuple[str, str, bool]] = []

    for key in ("name", "phone", "email", "work_experience"):
        raw.append((f"meta.{key}", str(getattr(resume.meta, key, "") or ""), False))

    for sentence_index, sentence in enumerate(
        item.strip()
        for item in _SENTENCE_SPLIT.split(resume.summary)
        if item.strip()
    ):
        raw.append((f"summary[{sentence_index}]", sentence, True))

    section_fields = {
        "education": ("school", "degree", "major", "period"),
        "experience": ("organization", "role", "period"),
        "research": ("institution", "topic", "period"),
        "activities": ("organization", "role", "period"),
        "projects": ("name", "organization", "role", "period"),
    }
    for section, fields in section_fields.items():
        for record_index, record in enumerate(getattr(resume, section)):
            for field in fields:
                raw.append((
                    f"{section}[{record_index}].{field}",
                    str(getattr(record, field, "") or ""),
                    False,
                ))
            for bullet_index, bullet in enumerate(getattr(record, "bullets", [])):
                raw.append((
                    f"{section}[{record_index}].bullets[{bullet_index}]",
                    str(bullet or ""),
                    True,
                ))

    for item_index, item in enumerate(resume.skills.items):
        raw.append((f"skills.items[{item_index}].name", item.name, True))
    for section in (
        "awards", "publications", "patents", "certifications", "training", "teaching",
    ):
        for item_index, value in enumerate(getattr(resume, section)):
            raw.append((f"{section}[{item_index}]", str(value or ""), True))
    for title, items in resume.additional_sections.items():
        for item_index, value in enumerate(items):
            raw.append((f"additional_sections.{title}[{item_index}]", str(value or ""), True))

    claims: list[AtomicClaim] = []
    for path, value, split_lists in raw:
        if not value.strip():
            continue
        segments = _atomic_segments(value, split_lists=split_lists)
        for atom_index, atom in enumerate(segments):
            # Target direction is allowed to come from the JD and is not a
            # candidate biography claim.  ``meta.target_role`` is excluded for
            # the same reason; keep that distinction when a deterministic
            # summary renders the direction in prose.
            if path.startswith("summary[") and _SUMMARY_DIRECTION.search(atom):
                continue
            claims.append(AtomicClaim(
                atom_id=f"{path}#atom[{atom_index}]",
                path=path,
                text=atom,
                record_key=_record_key(path),
                structural_category=_structural_category(path),
            ))
    return claims


def _claim_match_text(claim: AtomicClaim) -> str:
    """Remove deterministic presentation shells without deleting facts."""

    text = str(claim.text or "").strip()
    if not claim.path.startswith("summary["):
        return text
    shell_removed = bool(_SUMMARY_PRESENTATION_SHELL.search(text))
    text = _SUMMARY_PRESENTATION_SHELL.sub("", text, count=1).strip()
    if shell_removed:
        text = re.sub(r"^(?:包括|为)\s*", "", text, count=1).strip()
        section_wrapper = re.match(
            r"^(?:工作|项目|校园|科研|教育)经历(?:[（(][^）)]{1,80}[）)])?\s*[:：]\s*(.+)$",
            text,
        )
        if section_wrapper:
            text = section_wrapper.group(1).strip()
    text = re.sub(r"^(?:拥有|曾任)\s*", "", text, count=1).strip()
    contextual = re.fullmatch(r"在(.+?)担任(.+?)期间", text)
    if contextual:
        text = "".join(contextual.groups()).strip()
    role_contextual = re.fullmatch(r"(?:担任|任职为?|作为)(.+?)期间", text)
    if role_contextual:
        text = role_contextual.group(1).strip()
    return text or claim.text


def _binding_map(bindings: Iterable[EvidenceBinding]) -> dict[str, EvidenceBinding]:
    result: dict[str, EvidenceBinding] = {}
    for binding in bindings:
        result.setdefault(binding.path, binding)
    return result


def _fact_record_ids(binding: EvidenceBinding, facts: dict[str, FactUnit]) -> list[str]:
    return [
        facts[fact_id].record_id or ""
        for fact_id in binding.fact_ids
        if fact_id in facts and facts[fact_id].record_id
    ]


def _infer_record_owners(
    bindings: Iterable[EvidenceBinding],
    fact_by_id: dict[str, FactUnit],
) -> dict[str, str]:
    header_scores: dict[str, Counter[str]] = defaultdict(Counter)
    all_scores: dict[str, Counter[str]] = defaultdict(Counter)
    for binding in bindings:
        record = _record_key(binding.path)
        if not record:
            continue
        record_ids = list(dict.fromkeys(_fact_record_ids(binding, fact_by_id)))
        if not record_ids:
            continue
        identity = bool(re.search(
            r"\.(?:organization|institution|school|name|role|topic|period)$",
            binding.path,
        ))
        for record_id in record_ids:
            all_scores[record][record_id] += 1
            if identity:
                header_scores[record][record_id] += 4

    owners: dict[str, str] = {}
    for record in set(all_scores) | set(header_scores):
        scores = header_scores.get(record) or all_scores.get(record)
        if not scores:
            continue
        ranked = scores.most_common()
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            owners[record] = ranked[0][0]
    return owners


def _candidate_facts(
    claim: AtomicClaim,
    binding: EvidenceBinding | None,
    owner: str,
    eligible_facts: list[FactUnit],
    fact_by_id: dict[str, FactUnit],
) -> list[FactUnit]:
    result: list[FactUnit] = []
    seen: set[str] = set()

    if binding is not None:
        linked_blocks = set(binding.block_ids or [binding.block_id])
        for fact in eligible_facts:
            if fact.fact_id in binding.fact_ids or fact.block_id in linked_blocks:
                if fact.fact_id not in seen:
                    seen.add(fact.fact_id)
                    result.append(fact)
    if owner:
        for fact in eligible_facts:
            if fact.record_id == owner and fact.fact_id not in seen:
                seen.add(fact.fact_id)
                result.append(fact)
    if result:
        return result

    # Summary, skills and other distributed fields may legitimately draw from
    # any candidate record.  An unbound generated value is still checked
    # against every eligible fact so the audit does not equate a missing legacy
    # binding with fabrication.
    return list(eligible_facts)


def _individual_fact_scores(atom: str, facts: list[FactUnit]) -> list[tuple[float, FactUnit]]:
    atom_value = _normalize(atom)
    atom_bigrams = _bigrams(atom)
    atom_anchors = _anchors(atom)
    scored: list[tuple[float, FactUnit]] = []
    for fact in facts:
        source = fact.verbatim_text
        source_value = _normalize(source)
        if not source_value:
            continue
        source_anchors = _anchors(source)
        if atom_anchors and not atom_anchors.issubset(source_anchors):
            # A fact can still be one member of a multi-fact aggregate.  Keep a
            # lexical score, but do not let it independently support the atom.
            anchor_factor = 0.45
        else:
            anchor_factor = 1.0
        if atom_value == source_value:
            score = 1.0
        elif atom_value in source_value:
            score = min(1.0, len(atom_value) / max(1, len(source_value)) + 0.35)
        elif source_value in atom_value:
            score = min(1.0, len(source_value) / max(1, len(atom_value)) + 0.20)
        else:
            score = len(atom_bigrams & _bigrams(source)) / max(1, len(atom_bigrams))
        scored.append((round(score * anchor_factor, 4), fact))
    return sorted(scored, key=lambda item: item[0], reverse=True)


def _match_atom(
    claim: AtomicClaim,
    facts: list[FactUnit],
    binding: EvidenceBinding | None,
) -> AtomicMatch:
    if not facts:
        return AtomicMatch("unsupported", (), 0.0, "no_candidate_fact")

    match_text = _claim_match_text(claim)
    source_text = "；".join(fact.verbatim_text for fact in facts)
    if binding is not None and binding.source_claim:
        source_text += "；" + binding.source_claim
    source_value = _normalize(source_text)
    atom_value = _normalize(match_text)
    unmatched = _unmatched_anchors(match_text, source_text)
    if unmatched:
        return AtomicMatch(
            "unsupported", (), 0.0, "new_hard_anchor", unmatched,
        )

    scores = _individual_fact_scores(match_text, facts)
    best_score = scores[0][0] if scores else 0.0
    selection_floor = max(0.55, best_score - 0.12)
    selected = tuple(
        fact.fact_id for score, fact in scores
        if score >= selection_floor
    )
    if atom_value and atom_value in source_value:
        if not selected and scores:
            selected = (scores[0][1].fact_id,)
        return AtomicMatch("supported", selected, 1.0, "direct_source")
    if (
        _date_only(match_text)
        and _date_signatures(match_text)
        and _date_signatures(match_text).issubset(_date_signatures(source_text))
    ):
        if not selected and scores:
            selected = (scores[0][1].fact_id,)
        return AtomicMatch("supported", selected, 1.0, "equivalent_date_format")

    atom_bigrams = _bigrams(match_text)
    precision = len(atom_bigrams & _bigrams(source_text)) / max(1, len(atom_bigrams))
    direct_binding = bool(
        binding is not None
        and binding.mode in {"direct", "normalized"}
        and binding.similarity >= 0.9
    )
    trusted_rewrite = bool(
        binding is not None
        and binding.mode == "rewritten"
        and binding.source_claim
        and binding.similarity >= 0.75
    )
    threshold = 0.55 if trusted_rewrite else 0.68
    if direct_binding or precision >= threshold:
        if not selected and scores:
            selected = (scores[0][1].fact_id,)
        reason = "trusted_rewrite" if trusted_rewrite and precision < 0.68 else "lexical_entailment"
        return AtomicMatch(
            "supported", selected, round(max(precision, 0.75 if direct_binding else 0.0), 4), reason,
        )
    return AtomicMatch(
        "unsupported", (), round(precision, 4), "insufficient_source_overlap",
    )


def match_atomic_claim(
    text: str,
    facts: Iterable[FactUnit],
    *,
    path: str = "",
    binding: EvidenceBinding | None = None,
) -> AtomicMatch:
    """Classify one output clause against an explicit fact allow-list.

    Callers must choose the fact scope.  In particular, the repair layer passes
    facts from one uniquely owned source record, so a similar duty from another
    employer cannot legitimize or replace the clause.
    """

    claim = AtomicClaim(
        atom_id=f"{path or 'claim'}#atom[0]",
        path=path,
        text=str(text or "").strip(),
        record_key=_record_key(path),
        structural_category=_structural_category(path),
    )
    return _match_atom(claim, list(facts), binding)


def _fact_is_represented(
    fact: FactUnit,
    generated_text: str,
) -> bool:
    fact_value = _normalize(fact.verbatim_text)
    generated_value = _normalize(generated_text)
    if not fact_value:
        return True
    if fact_value in generated_value:
        return True
    if _unmatched_anchors(fact.verbatim_text, generated_text):
        return False
    if (
        _date_only(fact.verbatim_text)
        and _date_signatures(fact.verbatim_text)
        and _date_signatures(fact.verbatim_text).issubset(_date_signatures(generated_text))
    ):
        return True
    recall = len(_bigrams(fact.verbatim_text) & _bigrams(generated_text)) / max(
        1, len(_bigrams(fact.verbatim_text)),
    )
    return recall >= 0.58


def _source_structural_categories(fact: FactUnit) -> list[str]:
    mapping = {
        "organization": "organization",
        "role": "role",
        "period": "period",
        "education": "education",
        "credential": "credential",
        "skill": "skill_tool",
        "metric": "metric",
    }
    categories = [
        mapping[dimension]
        for dimension in fact.dimensions
        if dimension in mapping
        # Calendar dates contain digits but are period invariants, not outcome
        # metrics.  Keeping the categories disjoint avoids reporting a removed
        # employment date as both a date and a business number.
        and not (dimension == "metric" and "period" in fact.dimensions)
    ]
    return list(dict.fromkeys(categories))


def audit_atomic_facts(
    *,
    source: SourceBundle,
    resume: CanonicalResume,
    evidence_bindings: Iterable[EvidenceBinding],
) -> dict[str, dict[str, Any]]:
    """Return score-free atomic, ownership and invariant diagnostics."""

    bindings = list(evidence_bindings)
    binding_by_path = _binding_map(bindings)
    eligible_facts = [
        fact for fact in source.fact_units
        if fact.fact_eligible and fact.source_type != "jd"
    ]
    fact_by_id = {fact.fact_id: fact for fact in eligible_facts}
    owners = _infer_record_owners(bindings, fact_by_id)
    claims = _canonical_claims(resume)

    matches: dict[str, AtomicMatch] = {}
    generated_by_owner: dict[str, list[str]] = defaultdict(list)
    generated_by_fact: dict[str, list[str]] = defaultdict(list)
    unsupported_items: list[dict[str, Any]] = []
    for claim in claims:
        binding = binding_by_path.get(claim.path)
        candidates = _candidate_facts(
            claim,
            binding,
            owners.get(claim.record_key, ""),
            eligible_facts,
            fact_by_id,
        )
        match = _match_atom(claim, candidates, binding)
        matches[claim.atom_id] = match
        if claim.record_key:
            generated_by_owner[owners.get(claim.record_key, claim.record_key)].append(claim.text)
        if match.status == "supported":
            for fact_id in match.fact_ids:
                generated_by_fact[fact_id].append(claim.text)
        else:
            unsupported_items.append({
                "atom_id": claim.atom_id,
                "canonical_field_path": claim.path,
                "excerpt": _excerpt(claim.text),
                "reason": match.reason,
                "unmatched_anchors": list(match.unmatched_anchors),
            })

    represented_fact_ids: set[str] = set(generated_by_fact)
    all_generated = "；".join(claim.text for claim in claims)
    for fact in eligible_facts:
        if fact.fact_id in represented_fact_ids:
            continue
        candidate_texts: list[str] = []
        if fact.record_id:
            candidate_texts.extend(generated_by_owner.get(fact.record_id, []))
        if not candidate_texts:
            for binding in bindings:
                if fact.fact_id in binding.fact_ids or fact.block_id in (
                    binding.block_ids or [binding.block_id]
                ):
                    candidate_texts.append(binding.source_claim or binding.claim)
        generated_text = "；".join(item for item in candidate_texts if item)
        if not generated_text and not fact.record_id:
            generated_text = all_generated
        if generated_text and _fact_is_represented(fact, generated_text):
            represented_fact_ids.add(fact.fact_id)

    unrepresented_facts = [
        fact for fact in eligible_facts if fact.fact_id not in represented_fact_ids
    ]
    supported_count = sum(
        1 for match in matches.values() if match.status == "supported"
    )
    atom_count = len(claims)
    fact_count = len(eligible_facts)

    correct = 0
    incorrect = 0
    undetermined = 0
    ownership_issues: list[dict[str, Any]] = []
    for claim in claims:
        if not claim.record_key:
            continue
        match = matches[claim.atom_id]
        actual = {
            fact_by_id[fact_id].record_id
            for fact_id in match.fact_ids
            if fact_id in fact_by_id and fact_by_id[fact_id].record_id
        }
        expected = owners.get(claim.record_key, "")
        if match.status != "supported" or not expected or len(actual) != 1:
            undetermined += 1
            continue
        actual_owner = next(iter(actual))
        if actual_owner == expected:
            correct += 1
        else:
            incorrect += 1
            ownership_issues.append({
                "atom_id": claim.atom_id,
                "canonical_field_path": claim.path,
                "excerpt": _excerpt(claim.text),
                "expected_record_id": expected,
                "actual_record_id": actual_owner,
            })

    category_names = (
        "organization", "role", "period", "education", "credential", "metric", "skill_tool",
    )
    structural: dict[str, dict[str, Any]] = {}
    for category in category_names:
        additions: list[dict[str, Any]] = []
        for claim in claims:
            match = matches[claim.atom_id]
            claim_categories = {claim.structural_category} if claim.structural_category else set()
            if category == "metric" and claim.structural_category != "period" and _metric_anchors(claim.text):
                claim_categories.add("metric")
            if category in claim_categories and match.status != "supported":
                value = _metric_anchors(claim.text) if category == "metric" else [_excerpt(claim.text)]
                for item in value:
                    additions.append({
                        "canonical_field_path": claim.path,
                        "value": item,
                    })

        removals = [
            {
                "fact_id": fact.fact_id,
                "record_id": fact.record_id,
                "value": _excerpt(fact.verbatim_text),
                "source_spans": [span.model_dump() for span in fact.source_spans],
            }
            for fact in unrepresented_facts
            if category in _source_structural_categories(fact)
        ]
        structural[category] = {
            "added_count": len(additions),
            "missing_count": len(removals),
            "added": additions[:_MAX_AUDIT_ITEMS],
            "missing": removals[:_MAX_AUDIT_ITEMS],
            "truncated": (
                len(additions) > _MAX_AUDIT_ITEMS
                or len(removals) > _MAX_AUDIT_ITEMS
            ),
        }

    return {
        "atomic_factuality": {
            "generated_atom_count": atom_count,
            "supported_atom_count": supported_count,
            "unsupported_atom_count": atom_count - supported_count,
            "precision": round(supported_count / atom_count, 4) if atom_count else 1.0,
            "source_fact_count": fact_count,
            "represented_source_fact_count": len(represented_fact_ids),
            "unrepresented_source_fact_count": len(unrepresented_facts),
            "recall": round(len(represented_fact_ids) / fact_count, 4) if fact_count else 1.0,
            "unsupported_output": unsupported_items[:_MAX_AUDIT_ITEMS],
            "unrepresented_source_facts": [
                {
                    "fact_id": fact.fact_id,
                    "source_type": fact.source_type,
                    "record_id": fact.record_id,
                    "fact_type": fact.fact_type,
                    "dimensions": fact.dimensions,
                    "excerpt": _excerpt(fact.verbatim_text),
                    "source_spans": [span.model_dump() for span in fact.source_spans],
                }
                for fact in unrepresented_facts[:_MAX_AUDIT_ITEMS]
            ],
            "truncated": (
                len(unsupported_items) > _MAX_AUDIT_ITEMS
                or len(unrepresented_facts) > _MAX_AUDIT_ITEMS
            ),
        },
        "ownership_integrity": {
            "eligible_assignment_count": correct + incorrect + undetermined,
            "correct_assignment_count": correct,
            "incorrect_assignment_count": incorrect,
            "undetermined_assignment_count": undetermined,
            "integrity_rate": round(correct / (correct + incorrect), 4)
            if correct + incorrect else 1.0,
            "record_owners": owners,
            "issues": ownership_issues[:_MAX_AUDIT_ITEMS],
            "truncated": len(ownership_issues) > _MAX_AUDIT_ITEMS,
        },
        "structural_invariants": structural,
    }


__all__ = [
    "AtomicMatch",
    "atomize_claim_text",
    "audit_atomic_facts",
    "match_atomic_claim",
]
