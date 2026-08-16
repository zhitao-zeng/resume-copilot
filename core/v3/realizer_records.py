"""Record-local constrained realization (R24 Phase 3).

The logical isolation unit is one source record/experience (or one
section-scoped group for facts that own no record).  Physical requests may
pack multiple independent units under a character budget, but validation and
fallback decisions stay record-local:

- A semantic fallback fact degrades only its own unit; clean units keep LLM
  realization.  The whole-resume LLM realizer is never shut down because of
  one degraded record.
- A unit whose claims fail the hard verifier falls back to exact record-local
  source sentences; every other unit keeps its validated LLM prose.
- The optional profile summary is synthesized only when every unit is clean
  and fits a single physical request (the summary is a cross-record surface;
  the dedicated summary compiler is Phase 4 scope).

Every assembled claim is re-verified against its fact IDs and the immutable
organization/role/period/number/credential/ownership anchors by the same
``validate_realizer_response`` / ``validate_realized_claims`` verifiers used
for the all-resume boundary, plus the downstream atomic verifier.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
import json
import logging
import os
import time
from typing import Any, Callable, Iterable

from .contracts import (
    FactGraph,
    FactUnit,
    FrozenResume,
    NarrativeGroup,
    RealizedClaim,
    RealizerResponse,
    V3Model,
)
from .realizer import _join_facts, _placeholder, realize_plan, validate_realizer_response
from .realizer_llm import RealizationReport, RealizationResult, _minimum_remaining_seconds
from .training_schema import RealizerClaimDecision, SCHEMA_VERSION, SchemaVersion
from pydantic import Field


_REALIZER_TELEMETRY = logging.getLogger("v3.realizer_telemetry")

REALIZER_RECORD_SYSTEM_PROMPT = """你是简历证据编译器的表达阶段。严格返回指定 JSON Schema。
schema_version 必须是 resume_compiler_v3.4，request_fact_ids 必须与请求完全一致。
请求把事实分成若干独立 unit；每个 unit 对应一段来源经历或一个栏目，必须独立成文。
硬约束：
1. 每个请求 fact_id 必须出现在且只能出现在其所属 unit 的 claims 中一次；不得使用请求外或跨 unit 的 fact_id。
2. 每个输入 unit 必须在输出 units 中恰好出现一次，unit_id 与原请求一致。
3. claim 的 section、field、record_id、group_id 必须沿用输入约束，禁止跨经历归属。
4. 同一 unit 内，把相互支撑的背景、动作、方法、交付物或结果事实组织成少量连贯 bullet；
   所有 source_text 的实质内容必须逐字保留在 claim.text 中，只允许调整顺序、去重复标点、添加不含新事实的连接词。
5. 组织、岗位、时间、数字、学历、资质和工具不得改写、推断或新增。
6. 只有输入同时提供背景、动作、方法、交付物或结果时才可组合为 STAR，不补齐缺失维度。
7. 不为了篇幅删除事实，不输出解释、评分或待整理原始信息。
8. 仅当请求包含 optional_summary 时，才可额外输出最多一条 summary_claims：
   section=summary、field=summary、group_id=summary:profile、record_id=null，
   引用 summary_evidence_fact_ids 中的 fact_id，且同样逐字保留所引用事实的原文。"""


class RecordLocalUnitOutput(V3Model):
    """Claims for exactly one requested isolation unit."""

    unit_id: str = Field(min_length=1)
    claims: list[RealizerClaimDecision] = Field(min_length=1)


class RecordLocalPackResponse(V3Model):
    """One physical request, logically independent per-unit outputs."""

    schema_version: SchemaVersion
    request_fact_ids: list[str] = Field(min_length=1)
    units: list[RecordLocalUnitOutput] = Field(min_length=1)
    summary_claims: list[RealizerClaimDecision] = Field(default_factory=list)


@dataclass(frozen=True)
class RealizationUnit:
    unit_id: str
    groups: tuple[NarrativeGroup, ...]
    record_id: str | None
    section: str
    fact_ids: tuple[str, ...]
    degraded: bool = False


@dataclass(frozen=True)
class UnitReport:
    unit_id: str
    record_id: str | None
    section: str
    status: str
    fact_ids: tuple[str, ...]
    violations: tuple[str, ...] = ()
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "record_id": self.record_id,
            "section": self.section,
            "status": self.status,
            "fact_ids": list(self.fact_ids),
            "violations": list(self.violations),
            "error": self.error,
        }


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, "").strip() or default))
    except ValueError:
        return default


def partition_units(
    plan,
    graph: FactGraph,
    degraded_fact_ids: Iterable[str] = (),
) -> list[RealizationUnit]:
    """Split the plan into record-local isolation units in plan order."""

    fact_map = graph.fact_map()
    degraded = set(degraded_fact_ids)
    units: list[RealizationUnit] = []
    for group in plan.groups:
        fact_ids = tuple(
            fact_id for fact_id in group.fact_ids
            if fact_id in fact_map and fact_map[fact_id].eligible
        )
        unit_id = (
            f"record:{group.record_id}" if group.record_id is not None
            else f"group:{group.group_id}"
        )
        units.append(RealizationUnit(
            unit_id=unit_id,
            groups=(group,),
            record_id=group.record_id,
            section=group.section,
            fact_ids=fact_ids,
            degraded=bool(degraded.intersection(fact_ids)),
        ))
    return units


def _unit_payload(unit: RealizationUnit, fact_map: dict[str, FactUnit]) -> dict[str, Any]:
    groups = []
    for group in unit.groups:
        facts = []
        for fact_id in group.fact_ids:
            fact = fact_map.get(fact_id)
            if fact is None or not fact.eligible:
                continue
            facts.append({
                "fact_id": fact.fact_id,
                "source_text": fact.text,
                "fact_type": fact.fact_type,
                "destination_section": fact.destination_section or group.section,
                "destination_field": fact.destination_field or "bullet",
                "record_id": fact.record_id,
                "hard_anchors": [anchor.text for anchor in fact.anchors],
            })
        if facts:
            groups.append({
                "group_id": group.group_id,
                "section": group.section,
                "record_id": group.record_id,
                "purpose": group.purpose,
                "facts": facts,
            })
    return {
        "unit_id": unit.unit_id,
        "record_id": unit.record_id,
        "section": unit.section,
        "groups": groups,
    }


def _pack_units(
    units: list[RealizationUnit],
    fact_map: dict[str, FactUnit],
    max_chars: int,
) -> list[list[RealizationUnit]]:
    packs: list[list[RealizationUnit]] = []
    current: list[RealizationUnit] = []
    current_chars = 0
    for unit in units:
        size = len(json.dumps(_unit_payload(unit, fact_map), ensure_ascii=False))
        if current and current_chars + size > max_chars:
            packs.append(current)
            current, current_chars = [], 0
        current.append(unit)
        current_chars += size
    if current:
        packs.append(current)
    return packs


def _deterministic_unit_claims(
    unit: RealizationUnit,
    graph: FactGraph,
    *,
    skeleton: bool,
) -> list[RealizedClaim]:
    """Exact record-local source sentences for one unit (fail-closed path)."""

    fact_map = graph.fact_map()
    claims: list[RealizedClaim] = []
    for group in unit.groups:
        facts = [
            fact_map[fact_id] for fact_id in group.fact_ids
            if fact_id in fact_map and fact_map[fact_id].eligible
        ]
        if not facts:
            if skeleton:
                claims.append(RealizedClaim(
                    claim_id=f"{unit.unit_id}/placeholder:{group.group_id}",
                    section=group.section,
                    field="item",
                    text=_placeholder(group.section),
                    fact_ids=[],
                    record_id=group.record_id,
                    generated=False,
                    group_id=group.group_id,
                ))
            continue
        buckets: dict[tuple[str, str, str], list[FactUnit]] = {}
        for fact in facts:
            section = fact.destination_section or group.section
            field_name = fact.destination_field or "bullet"
            source_unit = fact.base_fact_id or fact.fact_id
            buckets.setdefault((section, field_name, source_unit), []).append(fact)
        for subindex, ((section, field_name, _src), bucket) in enumerate(buckets.items()):
            claims.append(RealizedClaim(
                claim_id=f"{unit.unit_id}/det:{group.group_id}:{subindex}",
                section=section,
                field=field_name,  # type: ignore[arg-type]
                text=_join_facts(list(bucket), graph),
                fact_ids=[fact.fact_id for fact in bucket],
                record_id=group.record_id,
                anchors=[anchor for fact in bucket for anchor in fact.anchors],
                generated=len(bucket) > 1,
                group_id=group.group_id,
            ))
    return claims


def _convert_unit_claims(
    unit: RealizationUnit,
    decisions: list[RealizerClaimDecision],
    fact_map: dict[str, FactUnit],
) -> list[RealizedClaim]:
    claims: list[RealizedClaim] = []
    for claim in decisions:
        claim_facts = [fact_map[fid] for fid in claim.fact_ids if fid in fact_map]
        claims.append(RealizedClaim(
            claim_id=f"{unit.unit_id}/{claim.claim_id}",
            section=claim.section,
            field=claim.field,
            text=claim.text,
            fact_ids=claim.fact_ids,
            record_id=claim.record_id,
            group_id=claim.group_id,
            anchors=[anchor for fact in claim_facts for anchor in fact.anchors],
            atomic=len(claim.fact_ids) == 1,
            generated=True,
        ))
    return claims


def _validate_unit(
    unit: RealizationUnit,
    claims: list[RealizedClaim],
    graph: FactGraph,
) -> list[str]:
    allowed_group_by_fact = {
        fact_id: group.group_id
        for group in unit.groups
        for fact_id in group.fact_ids
    }
    response = RealizerResponse(
        schema_version=SCHEMA_VERSION,
        request_fact_ids=list(unit.fact_ids),
        claims=claims,
    )
    return validate_realizer_response(
        response,
        graph,
        allowed_fact_ids=unit.fact_ids,
        allowed_group_by_fact=allowed_group_by_fact,
    )


def _validate_summary_claims(
    decisions: list[RealizerClaimDecision],
    graph: FactGraph,
    allowed_summary_fact_ids: set[str],
) -> tuple[list[RealizedClaim], list[str]]:
    """Summary citations reuse the hard numeric/anchor/verbatim checks."""

    fact_map = graph.fact_map()
    violations: list[str] = []
    if len(decisions) > 1:
        violations.append("summary_claims_exceed_one")
    converted: list[RealizedClaim] = []
    seen: set[str] = set()
    for claim in decisions[:1]:
        if not (
            claim.section == "summary"
            and claim.field == "summary"
            and claim.group_id == "summary:profile"
        ):
            violations.append(f"{claim.claim_id}:summary_shape_mismatch")
            continue
        if claim.record_id is not None:
            violations.append(f"{claim.claim_id}:summary_record_must_be_empty")
        claim_facts: list[FactUnit] = []
        for fact_id in claim.fact_ids:
            fact = fact_map.get(fact_id)
            if fact_id not in allowed_summary_fact_ids:
                violations.append(f"{claim.claim_id}:summary_fact_not_allowed:{fact_id}")
                continue
            if fact is None or not fact.eligible:
                violations.append(f"{claim.claim_id}:fact_not_eligible:{fact_id}")
                continue
            if fact_id in seen:
                violations.append(f"{claim.claim_id}:duplicate_fact_id")
            seen.add(fact_id)
            claim_facts.append(fact)
            if fact.text not in claim.text:
                violations.append(f"{claim.claim_id}:source_text_not_preserved:{fact_id}")
            for anchor in fact.anchors:
                if anchor.text not in claim.text:
                    violations.append(f"{claim.claim_id}:missing_anchor:{anchor.text}")
        from .realizer import _NUMBER_RE

        allowed_numbers = {
            number for fact in claim_facts for number in _NUMBER_RE.findall(fact.text)
        }
        if not set(_NUMBER_RE.findall(claim.text)) <= allowed_numbers:
            violations.append(f"{claim.claim_id}:novel_numeric_anchor")
        converted.append(RealizedClaim(
            claim_id=f"summary:profile/{claim.claim_id}",
            section=claim.section,
            field=claim.field,
            text=claim.text,
            fact_ids=claim.fact_ids,
            record_id=None,
            group_id=claim.group_id,
            anchors=[anchor for fact in claim_facts for anchor in fact.anchors],
            atomic=False,
            generated=True,
        ))
    if violations:
        return [], violations
    return converted, []


def _assemble(
    plan,
    units: list[RealizationUnit],
    claims_by_unit: dict[str, list[RealizedClaim]],
    summary_claims: list[RealizedClaim],
) -> FrozenResume:
    sections: dict[str, list[RealizedClaim]] = {}
    ordered: list[RealizedClaim] = []
    for unit in units:
        for claim in claims_by_unit.get(unit.unit_id, []):
            sections.setdefault(claim.section, []).append(claim)
    for claim in summary_claims:
        sections.setdefault(claim.section, []).append(claim)
    ordered = [claim for values in sections.values() for claim in values]
    return FrozenResume(
        sections=sections,
        claims=ordered,
        skeleton=plan.skeleton,
        template_mode=plan.template_mode,
    )


def realize_record_local(
    plan,
    graph: FactGraph,
    *,
    use_llm: bool = True,
    llm_call: Callable[..., dict[str, Any]] | None = None,
    remaining_seconds: Callable[[], float | None] | None = None,
    degraded_fact_ids: Iterable[str] = (),
) -> RealizationResult:
    """Realize each record-locally; degrade/fail per unit, never globally."""

    deterministic = realize_plan(plan, graph)
    units = partition_units(plan, graph, degraded_fact_ids)
    request_ids = tuple(plan.ledger.planned_fact_ids)

    def _result(
        status: str,
        claims_by_unit: dict[str, list[RealizedClaim]],
        unit_reports: list[UnitReport],
        summary_claims: list[RealizedClaim] | None = None,
        error: str = "",
        training_packs: tuple[tuple[dict[str, Any], dict[str, Any] | None], ...] = (),
    ) -> RealizationResult:
        frozen = _assemble(plan, units, claims_by_unit, summary_claims or [])
        first_input = training_packs[0][0] if training_packs else None
        first_output = training_packs[0][1] if training_packs else None
        return RealizationResult(
            frozen=frozen,
            report=RealizationReport(
                schema_version=SCHEMA_VERSION,
                status=status,
                request_fact_ids=request_ids,
                violations=tuple(
                    violation
                    for report in unit_reports
                    for violation in report.violations
                ),
                error=error,
                training_input=first_input,
                training_output=first_output,
                unit_reports=tuple(report.to_dict() for report in unit_reports),
            ),
        )

    def _all_deterministic(status: str, reason: str = "") -> RealizationResult:
        claims_by_unit: dict[str, list[RealizedClaim]] = {}
        reports: list[UnitReport] = []
        for unit in units:
            claims_by_unit[unit.unit_id] = _deterministic_unit_claims(
                unit, graph, skeleton=plan.skeleton,
            )
            reports.append(UnitReport(
                unit_id=unit.unit_id,
                record_id=unit.record_id,
                section=unit.section,
                status=status if unit.fact_ids or plan.skeleton else "empty",
                fact_ids=unit.fact_ids,
                error=reason,
            ))
        overall = status if status in {"disabled", "budget_fallback"} else "deterministic"
        return _result(overall, claims_by_unit, reports, error=reason)

    if not use_llm:
        return _all_deterministic("disabled")
    if plan.skeleton or not request_ids:
        return _all_deterministic("deterministic")

    if remaining_seconds is None:
        try:
            from server_runtime import remaining_request_seconds
        except ImportError:
            remaining_request_seconds = None  # type: ignore[assignment]
        remaining_seconds = remaining_request_seconds
    minimum = _minimum_remaining_seconds()
    remaining = remaining_seconds() if remaining_seconds is not None else None
    if remaining is not None and remaining < minimum:
        return _all_deterministic(
            "budget_fallback",
            f"remaining_budget:{remaining:.2f}s_below:{minimum:.2f}s",
        )

    fact_map = graph.fact_map()
    degraded_units = [unit for unit in units if unit.degraded]
    clean_units = [unit for unit in units if not unit.degraded and unit.fact_ids]
    empty_units = [unit for unit in units if not unit.degraded and not unit.fact_ids]

    claims_by_unit: dict[str, list[RealizedClaim]] = {}
    unit_reports: list[UnitReport] = []
    for unit in degraded_units:
        claims_by_unit[unit.unit_id] = _deterministic_unit_claims(
            unit, graph, skeleton=plan.skeleton,
        )
        unit_reports.append(UnitReport(
            unit_id=unit.unit_id,
            record_id=unit.record_id,
            section=unit.section,
            status="deterministic_degraded",
            fact_ids=unit.fact_ids,
        ))
    for unit in empty_units:
        claims_by_unit[unit.unit_id] = []
        unit_reports.append(UnitReport(
            unit_id=unit.unit_id,
            record_id=unit.record_id,
            section=unit.section,
            status="empty",
            fact_ids=(),
        ))
    if not clean_units:
        return _result(
            "deterministic",
            claims_by_unit,
            unit_reports,
            error="all_units_degraded",
        )

    max_chars = _env_int("V3_REALIZER_PACK_CHARS", 9000, 1000)
    packs = _pack_units(clean_units, fact_map, max_chars)
    # The optional cross-record summary stays single-request and clean-graph
    # only: a degraded unit means the model would synthesize from a partial
    # evidence view, so multi-pack or degraded resumes keep deterministic
    # summaries until the Phase 4 summary compiler.
    summary_enabled = not degraded_units and len(packs) == 1
    summary_fact_ids = {
        fact_id
        for unit in clean_units
        for fact_id in unit.fact_ids
        if fact_map[fact_id].fact_type not in {"identity", "contact"}
    }

    if llm_call is None:
        from server_runtime import call_llm_typed

        def _call_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return call_llm_typed(*args, allow_repair=False, **kwargs)

        llm_call = _call_once

    max_tokens = _env_int("V3_REALIZER_MAX_TOKENS", 6144, 1024)

    def _pack_payload(pack: list[RealizationUnit], include_summary: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "request_fact_ids": [fact_id for unit in pack for fact_id in unit.fact_ids],
            "units": [_unit_payload(unit, fact_map) for unit in pack],
        }
        if include_summary:
            payload["optional_summary"] = {
                "group_id": "summary:profile",
                "section": "summary",
                "field": "summary",
                "record_id": None,
                "summary_evidence_fact_ids": sorted(summary_fact_ids),
            }
        return payload

    def invoke(index: int, pack: list[RealizationUnit]) -> tuple[int, dict[str, Any] | Exception]:
        payload = _pack_payload(pack, include_summary=summary_enabled and index == 0)
        call_started = time.perf_counter()
        try:
            result = llm_call(
                RecordLocalPackResponse,
                REALIZER_RECORD_SYSTEM_PROMPT,
                json.dumps(payload, ensure_ascii=False),
                temperature=0.1,
                max_tokens=max_tokens,
            )
            _REALIZER_TELEMETRY.info(
                "realizer_pack_ok | pack=%d | units=%d | facts=%d | elapsed=%.3fs",
                index, len(pack), len(payload["request_fact_ids"]),
                time.perf_counter() - call_started,
            )
            return index, result
        except Exception as exc:
            _REALIZER_TELEMETRY.warning(
                "realizer_pack_fail | pack=%d | units=%d | elapsed=%.3fs | exc_type=%s | exc=%s",
                index, len(pack), time.perf_counter() - call_started,
                type(exc).__name__, str(exc)[:200],
            )
            return index, exc

    workers = max(1, min(_env_int("V3_REALIZER_CONCURRENCY", 2), len(packs)))
    pack_results: list[tuple[int, dict[str, Any] | Exception]] = []
    if workers == 1 or len(packs) == 1:
        for index, pack in enumerate(packs):
            pack_results.append(invoke(index, pack))
    else:
        futures: list[Future[tuple[int, dict[str, Any] | Exception]]] = []
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v3-realizer") as pool:
            for index, pack in enumerate(packs):
                context = copy_context()
                futures.append(pool.submit(context.run, invoke, index, pack))
            pack_results = [future.result() for future in futures]

    training_packs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    llm_unit_count = 0
    failed_unit_count = 0
    summary_claims: list[RealizedClaim] = []
    summary_violations: list[str] = []
    for index, raw_result in sorted(pack_results, key=lambda item: item[0]):
        pack = packs[index]
        payload = _pack_payload(pack, include_summary=summary_enabled and index == 0)
        if isinstance(raw_result, Exception):
            training_packs.append((payload, None))
            for unit in pack:
                claims_by_unit[unit.unit_id] = _deterministic_unit_claims(
                    unit, graph, skeleton=plan.skeleton,
                )
                failed_unit_count += 1
                unit_reports.append(UnitReport(
                    unit_id=unit.unit_id,
                    record_id=unit.record_id,
                    section=unit.section,
                    status="deterministic_fallback",
                    fact_ids=unit.fact_ids,
                    error=f"{type(raw_result).__name__}: {raw_result}",
                ))
            continue
        try:
            parsed = RecordLocalPackResponse.model_validate(raw_result)
        except Exception as exc:
            training_packs.append((payload, raw_result if isinstance(raw_result, dict) else None))
            for unit in pack:
                claims_by_unit[unit.unit_id] = _deterministic_unit_claims(
                    unit, graph, skeleton=plan.skeleton,
                )
                failed_unit_count += 1
                unit_reports.append(UnitReport(
                    unit_id=unit.unit_id,
                    record_id=unit.record_id,
                    section=unit.section,
                    status="deterministic_fallback",
                    fact_ids=unit.fact_ids,
                    error=f"schema_error:{type(exc).__name__}",
                ))
            continue
        training_packs.append((payload, parsed.model_dump(mode="json")))
        outputs_by_unit = {output.unit_id: output for output in parsed.units}
        requested_fact_ids = {fact_id for unit in pack for fact_id in unit.fact_ids}
        declared_mismatch = set(parsed.request_fact_ids) != requested_fact_ids
        for unit in pack:
            output = outputs_by_unit.get(unit.unit_id)
            if output is None:
                claims_by_unit[unit.unit_id] = _deterministic_unit_claims(
                    unit, graph, skeleton=plan.skeleton,
                )
                failed_unit_count += 1
                unit_reports.append(UnitReport(
                    unit_id=unit.unit_id,
                    record_id=unit.record_id,
                    section=unit.section,
                    status="deterministic_fallback",
                    fact_ids=unit.fact_ids,
                    violations=("unit_missing_from_response",),
                ))
                continue
            claims = _convert_unit_claims(unit, list(output.claims), fact_map)
            violations = _validate_unit(unit, claims, graph)
            if declared_mismatch:
                violations = violations + ["declared_request_fact_ids_mismatch"]
            if violations:
                claims_by_unit[unit.unit_id] = _deterministic_unit_claims(
                    unit, graph, skeleton=plan.skeleton,
                )
                failed_unit_count += 1
                unit_reports.append(UnitReport(
                    unit_id=unit.unit_id,
                    record_id=unit.record_id,
                    section=unit.section,
                    status="deterministic_fallback",
                    fact_ids=unit.fact_ids,
                    violations=tuple(violations),
                ))
                continue
            claims_by_unit[unit.unit_id] = claims
            llm_unit_count += 1
            unit_reports.append(UnitReport(
                unit_id=unit.unit_id,
                record_id=unit.record_id,
                section=unit.section,
                status="llm",
                fact_ids=unit.fact_ids,
            ))
        if summary_enabled and index == 0 and parsed.summary_claims:
            summary_claims, summary_violations = _validate_summary_claims(
                list(parsed.summary_claims), graph, summary_fact_ids,
            )
            if summary_violations:
                unit_reports.append(UnitReport(
                    unit_id="summary:profile",
                    record_id=None,
                    section="summary",
                    status="summary_rejected",
                    fact_ids=(),
                    violations=tuple(summary_violations),
                ))

    if llm_unit_count == 0:
        status = "fallback"
    elif failed_unit_count or degraded_units:
        status = "partial"
    else:
        status = "success"
    return _result(
        status,
        claims_by_unit,
        unit_reports,
        summary_claims=summary_claims,
        training_packs=tuple(training_packs),
    )


__all__ = [
    "REALIZER_RECORD_SYSTEM_PROMPT",
    "RecordLocalPackResponse",
    "RecordLocalUnitOutput",
    "RealizationUnit",
    "UnitReport",
    "partition_units",
    "realize_record_local",
]
