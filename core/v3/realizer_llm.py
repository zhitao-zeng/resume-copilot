"""One fixed-schema, source-locked LLM realization boundary."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Callable

from .contracts import FactGraph, FrozenResume, RealizedClaim, RealizerResponse, ResumePlan
from .realizer import realize_plan, validate_realizer_response
from .training_schema import ConstrainedRealizerResponse, SCHEMA_VERSION


_DEFAULT_REALIZER_MIN_REMAINING_SECONDS = 240.0


def _minimum_remaining_seconds() -> float:
    raw = os.getenv("V3_REALIZER_MIN_REMAINING_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_REALIZER_MIN_REMAINING_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_REALIZER_MIN_REMAINING_SECONDS


REALIZER_SYSTEM_PROMPT = """你是简历证据编译器的表达阶段。严格返回指定 JSON Schema。
schema_version 必须是 resume_compiler_v3.4，request_fact_ids 必须与请求完全一致。
硬约束：
1. 每个请求 fact_id 必须在主字段中且只能使用一次；不得使用请求外 fact_id。
   可额外输出最多一个 section=summary、field=summary、group_id=summary:profile、
   record_id=null 的总结 claim，并从 summary_evidence_fact_ids 复用已有 fact_id；复用不算新增事实。
2. claim 的 section、field、record_id、group_id 必须沿用输入约束，禁止跨经历归属。
3. 所有 source_text 的实质内容必须逐字保留在 claim.text 中；可以调整顺序、去掉重复标点并添加不含新事实的连接词。
4. 组织、岗位、时间、数字、学历、资质和工具不得改写、推断或新增。
5. 只有输入同时提供背景、动作、方法、交付物或结果时才可组合为 STAR，不补齐缺失维度。
6. 不为了篇幅删除事实，不输出解释、评分或待整理原始信息。"""


@dataclass(frozen=True)
class RealizationReport:
    schema_version: str
    status: str
    request_fact_ids: tuple[str, ...]
    violations: tuple[str, ...]
    error: str = ""
    training_input: dict[str, Any] | None = None
    training_output: dict[str, Any] | None = None
    unit_reports: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "request_fact_ids": list(self.request_fact_ids),
            "violations": list(self.violations),
            "error": self.error,
            "training_input": self.training_input,
            "training_output": self.training_output,
            "unit_reports": list(self.unit_reports),
        }


@dataclass(frozen=True)
class RealizationResult:
    frozen: FrozenResume
    report: RealizationReport


def _request_payload(plan: ResumePlan, graph: FactGraph) -> dict[str, Any]:
    fact_map = graph.fact_map()
    groups = []
    for group in plan.groups:
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
    request_fact_ids = [fact_id for group in plan.groups for fact_id in group.fact_ids]
    summary_evidence = [
        fact_id
        for fact_id in request_fact_ids
        if fact_id in fact_map and fact_map[fact_id].fact_type not in {"identity", "contact"}
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "request_fact_ids": request_fact_ids,
        "optional_summary": {
            "group_id": "summary:profile",
            "section": "summary",
            "field": "summary",
            "record_id": None,
            "summary_evidence_fact_ids": summary_evidence,
        },
        "groups": groups,
    }


def realize_with_llm(
    plan: ResumePlan,
    graph: FactGraph,
    *,
    use_llm: bool = True,
    llm_call: Callable[..., dict[str, Any]] | None = None,
    remaining_seconds: Callable[[], float | None] | None = None,
) -> RealizationResult:
    fallback = realize_plan(plan, graph)
    request_ids = tuple(plan.ledger.planned_fact_ids)
    payload = _request_payload(plan, graph)
    if not use_llm or plan.skeleton or not request_ids:
        return RealizationResult(
            frozen=fallback,
            report=RealizationReport(
                schema_version=SCHEMA_VERSION,
                status="disabled" if not use_llm else "deterministic",
                request_fact_ids=request_ids,
                violations=(),
                training_input=payload,
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
        return RealizationResult(
            frozen=fallback,
            report=RealizationReport(
                schema_version=SCHEMA_VERSION,
                status="budget_fallback",
                request_fact_ids=request_ids,
                violations=(),
                error=f"remaining_budget:{remaining:.2f}s_below:{minimum:.2f}s",
                training_input=payload,
            ),
        )
    if llm_call is None:
        from server_runtime import call_llm_typed

        # Do not turn base-model contract errors into another long repair call.
        # Exact deterministic realization is the safe fallback; the rejected
        # contract remains observable for later model fine-tuning.
        def _call_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return call_llm_typed(*args, allow_repair=False, **kwargs)

        llm_call = _call_once
    try:
        raw = llm_call(
            ConstrainedRealizerResponse,
            REALIZER_SYSTEM_PROMPT,
            json.dumps(payload, ensure_ascii=False),
            temperature=0.1,
            max_tokens=max(1024, int(os.getenv("V3_REALIZER_MAX_TOKENS", "6144"))),
        )
        parsed = ConstrainedRealizerResponse.model_validate(raw)
        fact_map = graph.fact_map()
        internal_claims = []
        for claim in parsed.claims:
            claim_facts = [fact_map[fact_id] for fact_id in claim.fact_ids if fact_id in fact_map]
            internal_claims.append(RealizedClaim(
                claim_id=claim.claim_id,
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
        response = RealizerResponse(
            schema_version=parsed.schema_version,
            request_fact_ids=parsed.request_fact_ids,
            claims=internal_claims,
        )
        allowed_groups = {
            fact_id: group.group_id
            for group in plan.groups
            for fact_id in group.fact_ids
        }
        violations = validate_realizer_response(
            response,
            graph,
            allowed_fact_ids=request_ids,
            allowed_group_by_fact=allowed_groups,
            allowed_summary_fact_ids=payload["optional_summary"]["summary_evidence_fact_ids"],
        )
        if violations:
            return RealizationResult(
                frozen=fallback,
                report=RealizationReport(
                    schema_version=SCHEMA_VERSION,
                    status="fallback",
                    request_fact_ids=request_ids,
                    violations=tuple(violations),
                    training_input=payload,
                    training_output=parsed.model_dump(mode="json"),
                ),
            )
        sections: dict[str, list[Any]] = {}
        claims = list(response.claims)
        for claim in claims:
            sections.setdefault(claim.section, []).append(claim)
        frozen = FrozenResume(
            sections=sections,
            claims=claims,
            skeleton=False,
            template_mode=plan.template_mode,
        )
        return RealizationResult(
            frozen=frozen,
            report=RealizationReport(
                schema_version=SCHEMA_VERSION,
                status="success",
                request_fact_ids=request_ids,
                violations=(),
                training_input=payload,
                training_output=parsed.model_dump(mode="json"),
            ),
        )
    except Exception as exc:
        return RealizationResult(
            frozen=fallback,
            report=RealizationReport(
                schema_version=SCHEMA_VERSION,
                status="fallback",
                request_fact_ids=request_ids,
                violations=(),
                error=f"{type(exc).__name__}: {exc}",
                training_input=payload,
            ),
        )


__all__ = [
    "REALIZER_SYSTEM_PROMPT", "RealizationReport", "RealizationResult",
    "realize_with_llm",
]
