"""Profile summary compilation with claim-level verification (R24 Phase 4).

The summary is a high-risk synthesis surface, so every sentence must bind to
the exact fact IDs it was generated from and then pass the same class of
hard checks as the realizer, plus summary-specific anti-synthesis rules:

- at most 100 compact Chinese characters across all sentences
- no novel numeric tokens beyond the bound facts (the realizer rule)
- no computed tenure: an explicit ``N年`` duration only survives when that
  exact token is stated by a bound fact
- no positioning/seniority/ability adjectives or comparatives unless the
  exact wording appears in a bound fact
- no timeline or organization/role-label concatenation

Sentences failing verification are removed; the remainder is verified again.
When nothing verifiable remains, the summary is dropped (fail closed) — a
missing summary never blocks the resume.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
import time
from typing import Any, Callable, Iterable

from pydantic import Field

from .contracts import FactGraph, FrozenResume, RealizedClaim, RequirementGraph, V3Model
from .realizer import _NUMBER_RE
from .training_schema import SCHEMA_VERSION, SchemaVersion


_SUMMARY_TELEMETRY = logging.getLogger("v3.summary_telemetry")

SUMMARY_SYSTEM_PROMPT = """你是简历证据编译器的个人总结阶段。严格返回指定 JSON Schema。
schema_version 必须是 resume_compiler_v3.4。
任务：基于给定事实写一段不超过 100 字的个人总结（中文），最多 3 句。
硬约束：
1. 每句必须给出支撑它的 fact_id 列表（fact_ids），只能使用输入提供的事实。
2. 不得计算或推断工龄：只有事实原文明确写出"N年"时才允许出现该表述。
3. 不得使用事实原文中没有的定位、资深程度或能力形容词（如资深、经验丰富、精通、擅长、卓越）。
4. 不得使用最高级或比较级（最、第一、唯一），除非事实原文明确写出。
5. 不得拼接时间线、公司名或岗位标签；组织、岗位、时间、数字、学历、资质不得新增或改写。
6. 提供 JD 要求时优先选择有直接事实支撑的能力证据；没有 JD 时只总结既有范围与事实。
7. 不确定就不要写；宁可少一句，不可无依据。"""


class SummarySentenceDecision(V3Model):
    text: str = Field(min_length=1)
    fact_ids: list[str] = Field(min_length=1)


class ProfileSummaryResponse(V3Model):
    schema_version: SchemaVersion
    sentences: list[SummarySentenceDecision] = Field(min_length=1)


@dataclass(frozen=True)
class SummaryCompilationReport:
    status: str
    sentences: tuple[dict[str, Any], ...] = ()
    violations: tuple[str, ...] = ()
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "sentences": list(self.sentences),
            "violations": list(self.violations),
            "error": self.error,
        }


@dataclass(frozen=True)
class SummaryCompilationResult:
    frozen: FrozenResume
    report: SummaryCompilationReport


_TENURE_RE = re.compile(r"\d+(?:\.\d+)?\s*年(?:半|以上|多|余)?")
# A single range ("2019年1月 - 2021年6月") is legitimate evidence; stitching
# two or more ranges into one summary is timeline concatenation.
_RANGE_PAIR_RE = re.compile(
    r"\d{4}\s*[年./-]\s*\d{1,2}\s*月?\s*[-—~至到]+\s*\d{4}\s*[年./-]\s*\d{1,2}\s*月?"
    r"|\d{4}\s*[-—~至到]+\s*\d{4}"
)
_FORBIDDEN_SYNTHESIS = (
    "资深", "经验丰富", "精通", "擅长", "卓越", "优秀", "出色", "深厚",
    "专家", "领先", "一流", "多年", "数年", "十余年", "丰富",
)
_COMPARATIVES = ("最", "第一", "唯一", "首位", "顶级")
MAX_COMPACT_CHARS = 100


def _minimum_remaining_seconds() -> float:
    raw = os.getenv("V3_SUMMARY_MIN_REMAINING_SECONDS", "").strip()
    if not raw:
        return 60.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 60.0


def _compact_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _allowed_facts(graph: FactGraph) -> dict[str, Any]:
    return {
        fact.fact_id: fact
        for fact in graph.eligible_facts()
        if fact.fact_type not in {"identity", "contact"}
    }


def validate_summary_sentences(
    sentences: Iterable[dict[str, Any]],
    graph: FactGraph,
    allowed_fact_ids: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Split sentences into (verified, violations); never mutates input."""

    fact_map = graph.fact_map()
    allowed = (
        set(allowed_fact_ids)
        if allowed_fact_ids is not None
        else set(_allowed_facts(graph))
    )
    verified: list[dict[str, Any]] = []
    violations: list[str] = []
    total_chars = 0
    for index, sentence in enumerate(sentences):
        text = str(sentence.get("text") or "").strip()
        fact_ids = list(sentence.get("fact_ids") or [])
        label = f"sentence[{index}]"
        sentence_violations: list[str] = []
        if not text:
            sentence_violations.append(f"{label}:empty_text")
        if not fact_ids:
            sentence_violations.append(f"{label}:unbound_claim")
        cited = []
        for fact_id in fact_ids:
            fact = fact_map.get(fact_id)
            if fact_id not in allowed:
                sentence_violations.append(f"{label}:fact_not_allowed:{fact_id}")
                continue
            if fact is None or not fact.eligible:
                sentence_violations.append(f"{label}:fact_not_eligible:{fact_id}")
                continue
            cited.append(fact)
        cited_texts = [fact.text for fact in cited]
        blob = "\n".join(cited_texts)
        allowed_numbers = set()
        for fact_text in cited_texts:
            allowed_numbers.update(_NUMBER_RE.findall(fact_text))
        output_numbers = set(_NUMBER_RE.findall(text))
        if not output_numbers <= allowed_numbers:
            sentence_violations.append(f"{label}:novel_numeric_anchor")
        for match in _TENURE_RE.findall(text):
            if match.replace(" ", "") not in blob.replace(" ", ""):
                sentence_violations.append(f"{label}:computed_tenure:{match}")
        for word in _FORBIDDEN_SYNTHESIS:
            if word in text and word not in blob:
                sentence_violations.append(f"{label}:unsupported_adjective:{word}")
        for word in _COMPARATIVES:
            if word in text and word not in blob:
                sentence_violations.append(f"{label}:unsupported_comparative:{word}")
        if len(_RANGE_PAIR_RE.findall(text)) >= 2:
            sentence_violations.append(f"{label}:timeline_concatenation")
        total_chars += _compact_len(text)
        if sentence_violations:
            violations.extend(sentence_violations)
            continue
        verified.append({"text": text, "fact_ids": fact_ids})
    if total_chars > MAX_COMPACT_CHARS:
        violations.append(f"summary_exceeds_{MAX_COMPACT_CHARS}_chars")
        # Drop trailing sentences until within budget; never truncate mid-text.
        while verified and total_chars > MAX_COMPACT_CHARS:
            removed = verified.pop()
            total_chars -= _compact_len(removed["text"])
        if not verified:
            violations.append("summary_empty_after_length_repair")
    return verified, violations


def _summary_claim(sentences: list[dict[str, Any]], graph: FactGraph) -> RealizedClaim:
    fact_map = graph.fact_map()
    fact_ids = [fact_id for sentence in sentences for fact_id in sentence["fact_ids"]]
    seen: set[str] = set()
    ordered_ids = [fid for fid in fact_ids if not (fid in seen or seen.add(fid))]
    return RealizedClaim(
        claim_id="summary:profile/compiled",
        section="summary",
        field="summary",
        text="".join(sentence["text"] for sentence in sentences),
        fact_ids=ordered_ids,
        record_id=None,
        group_id="summary:profile",
        anchors=[anchor for fid in ordered_ids if fid in fact_map for anchor in fact_map[fid].anchors],
        atomic=False,
        generated=True,
    )


def _without_summary(frozen: FrozenResume) -> FrozenResume:
    claims = [
        claim for claim in frozen.claims
        if not (claim.section == "summary" and claim.field == "summary")
    ]
    sections: dict[str, list[RealizedClaim]] = {}
    for claim in claims:
        sections.setdefault(claim.section, []).append(claim)
    return FrozenResume(
        sections=sections,
        claims=[claim for values in sections.values() for claim in values],
        skeleton=frozen.skeleton,
        template_mode=frozen.template_mode,
    )


def compile_summary(
    frozen: FrozenResume,
    graph: FactGraph,
    requirements: RequirementGraph | None = None,
    *,
    use_llm: bool = True,
    llm_call: Callable[..., dict[str, Any]] | None = None,
    remaining_seconds: Callable[[], float | None] | None = None,
) -> SummaryCompilationResult:
    """Verify an existing summary or generate a verified one (fail closed)."""

    base = _without_summary(frozen)
    existing = [
        claim for claim in frozen.claims
        if claim.section == "summary" and claim.field == "summary"
    ]
    # Stage 1: re-verify whatever the realizer produced under Phase 4 rules.
    if existing:
        candidates = [
            {"text": claim.text, "fact_ids": list(claim.fact_ids)}
            for claim in existing
        ]
        verified, violations = validate_summary_sentences(candidates, graph)
        if verified:
            claim = _summary_claim(verified, graph)
            frozen_out = _with_summary(base, claim)
            if _passes_atomic_audit(frozen_out, graph, requirements):
                return SummaryCompilationResult(
                    frozen=frozen_out,
                    report=SummaryCompilationReport(
                        status="revalidated",
                        sentences=tuple(verified),
                        violations=tuple(violations),
                    ),
                )
            violations = violations + ["atomic_audit_rejected"]
        # Unverifiable generated text never survives; try fresh generation.
        base_violations = violations
    else:
        base_violations = []

    if base.skeleton:
        return SummaryCompilationResult(
            frozen=base,
            report=SummaryCompilationReport(status="skipped_skeleton"),
        )
    if not use_llm:
        return SummaryCompilationResult(
            frozen=base,
            report=SummaryCompilationReport(
                status="disabled",
                violations=tuple(base_violations),
            ),
        )
    if remaining_seconds is None:
        try:
            from server_runtime import remaining_request_seconds
        except ImportError:
            remaining_request_seconds = None  # type: ignore[assignment]
        remaining_seconds = remaining_request_seconds
    remaining = remaining_seconds() if remaining_seconds is not None else None
    minimum = _minimum_remaining_seconds()
    if remaining is not None and remaining < minimum:
        return SummaryCompilationResult(
            frozen=base,
            report=SummaryCompilationReport(
                status="budget_fallback",
                violations=tuple(base_violations),
                error=f"remaining_budget:{remaining:.2f}s_below:{minimum:.2f}s",
            ),
        )

    allowed = _allowed_facts(graph)
    if not allowed:
        return SummaryCompilationResult(
            frozen=base,
            report=SummaryCompilationReport(status="no_evidence"),
        )
    if llm_call is None:
        from server_runtime import call_llm_typed

        def _call_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return call_llm_typed(*args, allow_repair=False, **kwargs)

        llm_call = _call_once
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": "profile_summary",
        "max_chars": MAX_COMPACT_CHARS,
        "jd_requirements": [
            item.text for item in (requirements.requirements if requirements else [])
        ][:8],
        "evidence_facts": [
            {
                "fact_id": fact.fact_id,
                "source_text": fact.text,
                "fact_type": fact.fact_type,
                "destination_section": fact.destination_section,
            }
            for fact in allowed.values()
        ],
    }
    started = time.perf_counter()
    try:
        raw = llm_call(
            ProfileSummaryResponse,
            SUMMARY_SYSTEM_PROMPT,
            json.dumps(payload, ensure_ascii=False),
            temperature=0.1,
            max_tokens=max(512, int(os.getenv("V3_SUMMARY_MAX_TOKENS", "1024"))),
        )
        parsed = ProfileSummaryResponse.model_validate(raw)
        _SUMMARY_TELEMETRY.info(
            "summary_ok | sentences=%d | elapsed=%.3fs",
            len(parsed.sentences), time.perf_counter() - started,
        )
    except Exception as exc:
        _SUMMARY_TELEMETRY.warning(
            "summary_fail | elapsed=%.3fs | exc_type=%s",
            time.perf_counter() - started, type(exc).__name__,
        )
        return SummaryCompilationResult(
            frozen=base,
            report=SummaryCompilationReport(
                status="fallback",
                violations=tuple(base_violations),
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
    verified, violations = validate_summary_sentences(
        [sentence.model_dump(mode="json") for sentence in parsed.sentences],
        graph,
    )
    violations = base_violations + violations
    if not verified:
        return SummaryCompilationResult(
            frozen=base,
            report=SummaryCompilationReport(
                status="dropped_unverifiable",
                violations=tuple(violations),
            ),
        )
    claim = _summary_claim(verified, graph)
    frozen_out = _with_summary(base, claim)
    if not _passes_atomic_audit(frozen_out, graph, requirements):
        return SummaryCompilationResult(
            frozen=base,
            report=SummaryCompilationReport(
                status="dropped_atomic_audit",
                sentences=tuple(verified),
                violations=tuple(violations + ["atomic_audit_rejected"]),
            ),
        )
    return SummaryCompilationResult(
        frozen=frozen_out,
        report=SummaryCompilationReport(
            status="generated",
            sentences=tuple(verified),
            violations=tuple(violations),
        ),
    )


def _passes_atomic_audit(
    frozen: FrozenResume,
    graph: FactGraph,
    requirements: RequirementGraph | None,
) -> bool:
    """Fail closed: a compiled summary survives only if the full atomic
    verifier still accepts the frozen resume with the summary claim added."""

    from .atomic_verifier import audit_frozen_resume

    return audit_frozen_resume(frozen, graph, requirements).clean


def _with_summary(frozen: FrozenResume, claim: RealizedClaim) -> FrozenResume:
    sections: dict[str, list[RealizedClaim]] = {
        section: list(items) for section, items in frozen.sections.items()
    }
    sections.setdefault("summary", []).append(claim)
    return FrozenResume(
        sections=sections,
        claims=[item for values in sections.values() for item in values],
        skeleton=frozen.skeleton,
        template_mode=frozen.template_mode,
    )


__all__ = [
    "MAX_COMPACT_CHARS",
    "ProfileSummaryResponse",
    "SUMMARY_SYSTEM_PROMPT",
    "SummaryCompilationReport",
    "SummaryCompilationResult",
    "SummarySentenceDecision",
    "compile_summary",
    "validate_summary_sentences",
]
