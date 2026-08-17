"""Constrained narrative realization.

The first V3 implementation deliberately uses a deterministic connector
fallback.  A future LLM call can implement the same protocol, but it may only
rewrite a fixed group of fact IDs and must pass the same verifier afterwards.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re

from .contracts import FactGraph, FactUnit, FrozenResume, NarrativeGroup, RealizedClaim, RealizerResponse, ResumePlan


@dataclass(frozen=True)
class RealizerRequest:
    group: NarrativeGroup
    source_facts: tuple[str, ...]
    instruction: str = ""


def _placeholder(section: str) -> str:
    labels = {
        "contact": "姓名、联系方式",
        "summary": "个人概述",
        "experience": "工作/实习经历（公司、岗位、时间、职责与成果）",
        "projects": "项目经历（项目名称、角色、时间与成果）",
        "education": "教育经历（学校、专业、学历与时间）",
        "skills": "技能与工具",
    }
    return f"[待补充：{labels.get(section, section)}]"


def _source_slice(facts: list[FactUnit], graph: FactGraph) -> str | None:
    """Recompose semantic atoms from their exact shared transport span.

    Atom boundaries are semantic, not punctuation boundaries.  Joining atoms
    with a new semicolon turns ``action`` + ``using method`` into fragmented
    prose and can create false cross-atom entities.  When every atom came from
    one source unit, the original slice is the safest fluent realization.
    """

    if not facts:
        return None
    source_ids = {fact.source_id for fact in facts}
    base_ids = {fact.base_fact_id or fact.fact_id for fact in facts}
    if len(source_ids) != 1 or len(base_ids) != 1:
        return None
    flattened = [span for fact in facts for span in fact.spans]
    if not flattened or any(span.source_id != facts[0].source_id for span in flattened):
        return None
    ordered = sorted(flattened, key=lambda span: (span.char_start, span.char_end))
    if any(left.char_end > right.char_start for left, right in zip(ordered, ordered[1:])):
        return None
    document = graph.documents.get(facts[0].source_id, "")
    start, end = ordered[0].char_start, ordered[-1].char_end
    if not document or end > len(document):
        return None
    value = document[start:end]
    cursor = 0
    for fact in facts:
        location = value.find(fact.text, cursor)
        if location < 0:
            return None
        cursor = location + len(fact.text)
    return value


def _join_facts(facts: list[FactUnit], graph: FactGraph) -> str:
    source_value = _source_slice(facts, graph)
    if source_value is not None:
        return source_value
    texts = [fact.text for fact in facts]
    if len(texts) == 1:
        return texts[0]
    # Add a grammatical connector without normalizing even punctuation.  The
    # deterministic fallback is an exact-evidence path: each complete source
    # fact must remain a literal substring so the same hard verifier used for
    # generated prose cannot delete correctly sourced content.
    return "；".join(texts)


def realize_plan(plan: ResumePlan, fact_graph: FactGraph) -> FrozenResume:
    fact_map = fact_graph.fact_map()
    claims: list[RealizedClaim] = []
    sections: dict[str, list[RealizedClaim]] = {}
    for index, group in enumerate(plan.groups):
        facts = [fact_map[fact_id] for fact_id in group.fact_ids if fact_id in fact_map and fact_map[fact_id].eligible]
        if not facts:
            if not plan.skeleton:
                continue
            claim = RealizedClaim(
                claim_id=f"claim:{index}", section=group.section,
                field="item",
                text=_placeholder(group.section), fact_ids=[], record_id=group.record_id,
                generated=False, group_id=group.group_id,
            )
            claims.append(claim)
            sections.setdefault(group.section, []).append(claim)
            continue
        buckets: dict[tuple[str, str, str], list[object]] = {}
        for fact in facts:
            section = fact.destination_section or group.section
            field = fact.destination_field or "bullet"
            # Compose only atoms split from the same transport fact.  Joining
            # every bullet in one source record produced giant paragraphs,
            # hid exact bindings and could merge a later section when its
            # heading was missing.  The optional constrained realizer may
            # still combine compatible facts after preserving every fact ID.
            source_unit = fact.base_fact_id or fact.fact_id
            key = (section, field, source_unit)
            buckets.setdefault(key, []).append(fact)
        for subindex, ((section, field, _source_unit), bucket_items) in enumerate(buckets.items()):
            bucket = list(bucket_items)
            claim = RealizedClaim(
                claim_id=f"claim:{index}:{subindex}",
                section=section,
                field=field,  # type: ignore[arg-type]
                text=_join_facts(bucket, fact_graph),
                fact_ids=[fact.fact_id for fact in bucket],
                record_id=group.record_id,
                anchors=[anchor for fact in bucket for anchor in fact.anchors],
                generated=len(bucket) > 1,
                group_id=group.group_id,
            )
            claims.append(claim)
            sections.setdefault(section, []).append(claim)
    # A section can be encountered again after another group when semantic
    # atoms expose multiple fields.  Canonicalize the flat claim order to the
    # section mapping instead of weakening FrozenResume's consistency check.
    ordered_claims = [claim for values in sections.values() for claim in values]
    # R25: join OCR soft-wrapped fragments back into whole sentences before
    # freezing (exact source text, no content change).
    ordered_claims = merge_fragment_claims(ordered_claims, fact_graph)
    sections = {}
    for claim in ordered_claims:
        sections.setdefault(claim.section, []).append(claim)
    return FrozenResume(
        sections=sections,
        claims=ordered_claims,
        skeleton=plan.skeleton,
        template_mode=plan.template_mode,
    )


def validate_realized_claims(frozen: FrozenResume, fact_graph: FactGraph) -> list[str]:
    """Return protocol violations without mutating generated content."""
    fact_map = fact_graph.fact_map()
    section_by_id = {section.section_id: section.section_type for section in fact_graph.sections}
    violations: list[str] = []
    for claim in frozen.claims:
        summary_citation = (
            claim.section == "summary"
            and claim.field == "summary"
            and claim.group_id == "summary:profile"
        )
        if not claim.fact_ids and not claim.text.startswith("[待补充："):
            violations.append(f"{claim.claim_id}:factless_non_placeholder")
        for fact_id in claim.fact_ids:
            fact = fact_map.get(fact_id)
            if fact is None:
                violations.append(f"{claim.claim_id}:unknown_fact:{fact_id}")
            elif not fact.eligible:
                violations.append(f"{claim.claim_id}:ineligible_fact:{fact_id}")
            elif not summary_citation and fact.text not in claim.text:
                # Summary citations are synthesis surfaces: they are gated by
                # the Phase 4 summary verifier (bound fact IDs, no novel
                # numbers, no inferred tenure or unsupported adjectives), so
                # the atomic audit does not demand verbatim preservation.
                violations.append(f"{claim.claim_id}:source_text_not_preserved:{fact_id}")
            if fact is not None and not summary_citation and fact.record_id != claim.record_id:
                violations.append(f"{claim.claim_id}:record_mismatch:{fact_id}")
            if fact is not None and fact.section_id is not None:
                expected_section = section_by_id.get(fact.section_id, "other")
                if not summary_citation and expected_section != "other" and claim.section != expected_section:
                    violations.append(f"{claim.claim_id}:section_mismatch:{fact_id}")
            if fact is not None and not summary_citation and fact.destination_section is not None and claim.section != fact.destination_section:
                violations.append(f"{claim.claim_id}:destination_section_mismatch:{fact_id}")
            if fact is not None and not summary_citation and fact.destination_field is not None and claim.field != fact.destination_field:
                # Multiple atomic facts can form one narrative bullet.  Other
                # destination fields remain immutable render fields.
                if fact.destination_field != "bullet" or claim.field != "bullet":
                    violations.append(f"{claim.claim_id}:destination_field_mismatch:{fact_id}")
    return violations


_NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?%?|\d{4}[./年-]\d{1,2})(?![A-Za-z])")

# Connective-only residual allowlist: anything a claim adds around cited
# facts must be punctuation or one of these connectors; every other token
# must appear verbatim in the unit's source-line vocabulary (R25).
_CONNECTORS = frozenset({
    "的", "了", "和", "与", "及", "或", "并", "并且", "且", "而", "在", "为",
    "于", "以", "向", "从", "把", "将", "对", "通过", "根据", "围绕", "等",
    "以及", "同时", "期间", "其中", "相关", "负责", "参与", "协助", "支持",
    "完成", "实现", "进行", "担任", "兼任", "包括", "含", "主要", "具备", "拥有",
    # Framing nouns carry no entity/number/claim content of their own.
    "经验", "经历", "能力", "方面", "背景", "工作", "技能",
})
# English glue words allowed around cited facts (no content value).
_LATIN_CONNECTORS = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "as", "is", "are", "was", "were", "be", "been", "being",
    "from", "into", "across", "over", "within", "including", "via", "per",
    "experienced", "skilled", "proficient", "familiar", "responsible",
    "worked", "working", "focused", "based", "related", "strong", "solid",
})
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z+#.-]*")
_CJK_RUN_RE = re.compile(r"[一-鿿]+")
_PUNCT_ONLY_RE = re.compile(r"^[\s，。、；：,.;·•\-—–|｜/（）()【】「」《》\"'“”‘’]+$")


def _residual_escape_violations(
    claim_text: str,
    cited_texts: list[str],
    vocabulary: str,
) -> list[str]:
    """Residual text after removing cited facts must be connective only.

    The vocabulary is the exact source-line text of the cited facts and
    their base facts; any content token outside it is an atom-boundary
    escape (e.g. rewriting ``using`` as ``through`` around a quoted atom).
    """

    residual = claim_text
    for cited in sorted(cited_texts, key=len, reverse=True):
        residual = residual.replace(cited, "\x00", 1)
    violations: list[str] = []
    for segment in residual.split("\x00"):
        if not segment or _PUNCT_ONLY_RE.match(segment):
            continue
        probe = segment
        for connector in _CONNECTORS:
            probe = probe.replace(connector, " ")
        for word in _LATIN_WORD_RE.findall(probe):
            if word.casefold() in _LATIN_CONNECTORS:
                continue
            if word.casefold() not in vocabulary.casefold():
                violations.append(f"escape_latin:{word}")
        for run in _CJK_RUN_RE.findall(probe):
            if len(run) <= 1:
                continue
            if run not in vocabulary:
                violations.append(f"escape_cjk:{run[:12]}")
    return violations

_TERMINAL_PUNCT = "。！？；.!?;:：）)】」』”"
_ITEM_START = re.compile(r"^\s*(?:[A-Z0-9]|[•·\-*▪◦]|（?\d+[）.)])")
_LABEL_START = re.compile(r"^[^：:\n]{1,15}[：:]")
_DATE_RANGE = re.compile(r"\d{4}\s*[年./-]")


def _claims_are_soft_wrapped(left: "RealizedClaim", right: "RealizedClaim", fact_graph: FactGraph) -> bool:
    """True when right is an OCR soft-wrap continuation of left.

    Structural rule (no NLP guessing): both claims sit on adjacent source
    lines separated by a single newline, the left text has no sentence-final
    punctuation and is not a record header, and the right does not open a
    new item or label.  Joining restores the original sentence; the merged
    text is the exact source slice with the wrap newline removed, so every
    fact stays a literal substring and the hard verifiers still pass.
    """

    fact_map = fact_graph.fact_map()
    left_spans = [span for fid in left.fact_ids if fid in fact_map for span in fact_map[fid].spans]
    right_spans = [span for fid in right.fact_ids if fid in fact_map for span in fact_map[fid].spans]
    if not left_spans or not right_spans:
        return False
    # Never merge across records or plan groups: ownership stays exact.
    if left.record_id != right.record_id or left.group_id != right.group_id:
        return False
    source_ids = {span.source_id for span in left_spans + right_spans}
    if len(source_ids) != 1:
        return False
    source_id = next(iter(source_ids))
    document = fact_graph.documents.get(source_id, "")
    left_end = max(span.char_end for span in left_spans)
    right_start = min(span.char_start for span in right_spans)
    if right_start < left_end:
        return False
    gap = document[left_end:right_start]
    if gap not in {"\n", "\r\n"}:
        return False
    left_text = left.text.rstrip()
    right_text = right.text.lstrip()
    if not left_text or not right_text:
        return False
    if left_text[-1] in _TERMINAL_PUNCT:
        return False
    # A dated record header is complete on its own line; never merge content
    # up into it, and never merge a new header down into a fragment.
    if _DATE_RANGE.search(left_text) or _DATE_RANGE.search(right_text):
        return False
    if _ITEM_START.match(right_text) or _LABEL_START.match(right_text):
        return False
    return True


def merge_fragment_claims(claims: list["RealizedClaim"], fact_graph: FactGraph) -> list["RealizedClaim"]:
    """Join OCR soft-wrapped adjacent claims back into whole sentences.

    Only merges claims that are line-adjacent fragments of the same source
    sentence (``_claims_are_soft_wrapped``); everything else is untouched.
    Merged claims cite the union of fact IDs in order and keep the left
    claim's section/field/record/group, so ownership and destination checks
    are unaffected.
    """

    merged: list[RealizedClaim] = []
    merged_last = False
    for claim in claims:
        # No chain merges: a claim produced by a join is never the left side
        # of another join, so a fully-punctuated next line can never be
        # absorbed by accident.
        if merged and not merged_last and _claims_are_soft_wrapped(merged[-1], claim, fact_graph):
            left = merged[-1]
            text = left.text.rstrip() + claim.text.lstrip()
            merged[-1] = RealizedClaim(
                claim_id=left.claim_id,
                section=left.section,
                field=left.field,
                text=text,
                fact_ids=[*left.fact_ids, *claim.fact_ids],
                record_id=left.record_id,
                anchors=[*left.anchors, *claim.anchors],
                atomic=False,
                generated=True,
                group_id=left.group_id,
            )
            merged_last = True
            continue
        merged.append(claim)
        merged_last = False
    return merged


def validate_realizer_response(
    response: RealizerResponse,
    fact_graph: FactGraph,
    *,
    allowed_fact_ids: Iterable[str] | None = None,
    allowed_group_by_fact: dict[str, str] | None = None,
    allowed_summary_fact_ids: Iterable[str] | None = None,
) -> list[str]:
    """Validate the boundary of an optional constrained LLM response.

    This is deliberately conservative: it rejects unknown/out-of-request
    facts, record mixing, missing hard anchors and novel numeric tokens, while
    allowing ordinary connective words around source-backed text.
    """
    fact_map = fact_graph.fact_map()
    declared = set(response.request_fact_ids)
    requested = set(allowed_fact_ids) if allowed_fact_ids is not None else declared
    violations: list[str] = []
    if len(response.request_fact_ids) != len(declared):
        violations.append("request_fact_ids_not_unique")
    if allowed_fact_ids is not None and declared != requested:
        violations.append("declared_request_fact_ids_mismatch")
    for fact_id in sorted(requested):
        fact = fact_map.get(fact_id)
        if fact is None or not fact.eligible:
            violations.append(f"request_fact_not_eligible:{fact_id}")
    section_by_id = {section.section_id: section.section_type for section in fact_graph.sections}
    seen_claim_ids: set[str] = set()
    used_fact_ids: list[str] = []
    for claim in response.claims:
        summary_citation = (
            claim.section == "summary"
            and claim.field == "summary"
            and claim.group_id == "summary:profile"
        )
        if summary_citation and claim.record_id is not None:
            violations.append(f"{claim.claim_id}:summary_record_must_be_empty")
        if summary_citation and allowed_summary_fact_ids is not None:
            summary_allowed = set(allowed_summary_fact_ids)
            for fact_id in claim.fact_ids:
                if fact_id not in summary_allowed:
                    violations.append(f"{claim.claim_id}:summary_fact_not_allowed:{fact_id}")
        if claim.claim_id in seen_claim_ids:
            violations.append(f"duplicate_claim_id:{claim.claim_id}")
        seen_claim_ids.add(claim.claim_id)
        if not claim.fact_ids:
            if not claim.text.startswith("[待补充："):
                violations.append(f"{claim.claim_id}:factless_non_placeholder")
            continue
        if len(claim.fact_ids) != len(set(claim.fact_ids)):
            violations.append(f"{claim.claim_id}:duplicate_fact_id")
        claim_facts = [fact_map[fact_id] for fact_id in claim.fact_ids if fact_id in fact_map and fact_id in requested and fact_map[fact_id].eligible]
        if not summary_citation:
            used_fact_ids.extend(fact.fact_id for fact in claim_facts)
        allowed_numbers = {
            number
            for fact in claim_facts
            for number in _NUMBER_RE.findall(fact.text)
        }
        output_numbers = set(_NUMBER_RE.findall(claim.text))
        if not output_numbers <= allowed_numbers:
            violations.append(f"{claim.claim_id}:novel_numeric_anchor")
        for fact_id in claim.fact_ids:
            fact = fact_map.get(fact_id)
            if fact_id not in requested:
                violations.append(f"{claim.claim_id}:fact_not_requested:{fact_id}")
                continue
            if fact is None or not fact.eligible:
                violations.append(f"{claim.claim_id}:fact_not_eligible:{fact_id}")
                continue
            if not summary_citation and fact.record_id != claim.record_id:
                violations.append(f"{claim.claim_id}:record_mismatch:{fact_id}")
            if not summary_citation and allowed_group_by_fact is not None:
                expected_group = allowed_group_by_fact.get(fact_id)
                if expected_group is not None and claim.group_id != expected_group:
                    violations.append(f"{claim.claim_id}:group_mismatch:{fact_id}")
            if fact.text not in claim.text:
                violations.append(f"{claim.claim_id}:source_text_not_preserved:{fact_id}")
            if fact.section_id is not None:
                expected_section = section_by_id.get(fact.section_id, "other")
                if not summary_citation and expected_section != "other" and claim.section != expected_section:
                    violations.append(f"{claim.claim_id}:section_mismatch:{fact_id}")
            if not summary_citation and fact.destination_section is not None and claim.section != fact.destination_section:
                violations.append(f"{claim.claim_id}:destination_section_mismatch:{fact_id}")
            if not summary_citation and fact.destination_field is not None and claim.field != fact.destination_field:
                if fact.destination_field != "bullet" or claim.field != "bullet":
                    violations.append(f"{claim.claim_id}:destination_field_mismatch:{fact_id}")
            for anchor in fact.anchors:
                if anchor.text not in claim.text:
                    violations.append(f"{claim.claim_id}:missing_anchor:{anchor.text}")
    used = set(used_fact_ids)
    for fact_id in sorted(requested - used):
        violations.append(f"missing_requested_fact:{fact_id}")
    for fact_id in sorted({fact_id for fact_id in used if used_fact_ids.count(fact_id) > 1}):
        violations.append(f"duplicate_requested_fact:{fact_id}")
    # Atom-boundary escape check: connective tissue around cited facts must
    # come from punctuation, the connector allowlist, or the exact
    # source-line vocabulary of the cited facts and their base facts.
    for claim in response.claims:
        summary_citation = (
            claim.section == "summary"
            and claim.field == "summary"
            and claim.group_id == "summary:profile"
        )
        if summary_citation or not claim.fact_ids:
            continue
        cited_facts = [
            fact_map[fact_id] for fact_id in claim.fact_ids
            if fact_id in fact_map and fact_id in requested
        ]
        if not cited_facts:
            continue
        vocabulary_parts: list[str] = []
        for fact in cited_facts:
            vocabulary_parts.append(fact.text)
            base = fact_map.get(fact.base_fact_id or "")
            if base is not None:
                vocabulary_parts.append(base.text)
            else:
                # The base transport fact may be replaced by its atoms and
                # structural context facts; include every sibling derived
                # from the same source line (label prefixes live there).
                vocabulary_parts.extend(
                    sibling.text
                    for sibling in fact_graph.facts
                    if sibling.base_fact_id == (fact.base_fact_id or fact.fact_id)
                )
        for violation in _residual_escape_violations(
            claim.text,
            [fact.text for fact in cited_facts],
            "\n".join(vocabulary_parts),
        ):
            violations.append(f"{claim.claim_id}:{violation}")
    return violations


__all__ = ["RealizerRequest", "merge_fragment_claims", "realize_plan", "validate_realized_claims", "validate_realizer_response"]
