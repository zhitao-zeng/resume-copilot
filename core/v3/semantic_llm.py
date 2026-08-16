"""Schema-driven semantic compilation for exact source facts.

This module is the trainable semantic boundary.  It deliberately contains no
occupation vocabulary or bad-case patch table.  Deterministic code validates
source identity, exact quotes, immutable layout ownership and schema version;
classification quality remains a model/data problem and is exposed in the
report instead of being hidden by corrective heuristics.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any, Callable

from pydantic import ValidationError

from .contracts import FactGraph, FactUnit, ResumeField, ResumeSection, SourceSpan
from .fact_graph import is_date_placeholder_range
from .training_schema import (
    SCHEMA_VERSION,
    SemanticAtomDecision,
    SemanticCompilationResponse,
    SemanticContextSpan,
    SemanticFactDecision,
)


SEMANTIC_SYSTEM_PROMPT = """你是简历证据编译器的语义阶段。只处理给出的候选原文，不写简历正文。
必须严格返回指定 JSON Schema，schema_version 必须是 resume_compiler_v3.4。
规则：
1. quote 必须逐字复制 candidate_text 的连续子串，不得改写、补字或从 JD 推断。
2. classification 与 atom.fact_type 是两套不同枚举：
   - 个人事实标 fact；
   - 纯字段标签、分隔符、占位符或重复标题标 context；方括号中的待填写字段，
     无论是姓名、学校、城市还是其他名称，均属于 placeholder，不是个人事实；
   - 请求/偏好标 intent，操作命令或元说明标 instruction；
   - 无法判断标 ambiguous。
   atom.fact_type 只能是 Schema 列出的 identity/contact/organization/role/period/action/
   method/deliverable/result/skill/education/credential/degree/major/metric/project/award/
   publication/training/teaching/other，绝不能填 intent、instruction 或 context。
3. fact 可拆成多个不重叠、按原文顺序排列的原子片段；不要丢失简历中的事实内容。
   不应写入简历的字段标签、分隔文本或重复内容必须逐字放入 context_spans；atoms 与
   context_spans 合计应覆盖原文中的所有实质字符。
   context_spans 只允许 label、separator、instruction、placeholder、duplicate；修饰词、
   能力程度、动作、方法和结果都不是 context，不得用 context 删除。
   时间、数字、联系方式等 hard_anchors 不能标为 context_span。
4. record_id 只能从该候选的 allowed_record_ids 选择；已有 locked_record_id 时必须原样返回。
5. destination_section/destination_field 使用通用 Schema，不发明行业字段。
6. 组织、岗位、时间、数字、学历、资质的事实类型必须按原文含义标注；不做常识补全。
7. context/intent/instruction 必须 atoms=[]，并用 context_spans 逐字覆盖全部实质字符；
   context 的 placeholder 使用 reason=placeholder。任何 hard_anchor 都不能进入 context_spans。
8. 输入若提供 structural_context_spans，它们是编译器从布局确定的标签上下文；不得作为
   atom 输出。

通用示例：
- candidate_text="3. 技能" -> classification=context, atoms=[],
  context_spans=[{"quote":"3. 技能","reason":"label"}]
- candidate_text="主要成就：" -> classification=context, atoms=[],
  context_spans=[{"quote":"主要成就：","reason":"label"}]
- candidate_text="[任意待填字段]" 或 "城市：[城市]" -> classification=context, atoms=[]，
  context_spans 必须逐字覆盖整段，reason=placeholder
- candidate_text="以下是我提供的全部个人信息。" -> classification=instruction, atoms=[],
  context_spans=[{"quote":"以下是我提供的全部个人信息。","reason":"instruction"}]
- candidate_text="- 将收款周期从145天缩短至50天" -> classification=fact；"- "放 separator
  context_span，其余逐字放 fact atoms，145天和50天必须保留在 atoms。"""


@dataclass(frozen=True)
class SemanticCompilationReport:
    schema_version: str
    status: str
    batch_count: int
    response_batch_count: int
    schema_valid_batch_count: int
    raw_decision_count: int
    valid_decision_count: int
    invalid_decision_count: int
    recovered_decision_count: int
    invalid_atom_count: int
    invalid_context_span_count: int
    input_fact_ids: tuple[str, ...]
    accepted_fact_ids: tuple[str, ...]
    fallback_fact_ids: tuple[str, ...]
    fail_closed_fact_ids: tuple[str, ...]
    non_fact_ids: tuple[str, ...]
    context_fact_ids: tuple[str, ...]
    errors: tuple[str, ...]
    training_inputs: tuple[dict[str, Any], ...] = ()
    training_outputs: tuple[dict[str, Any] | None, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "batch_count": self.batch_count,
            "response_batch_count": self.response_batch_count,
            "schema_valid_batch_count": self.schema_valid_batch_count,
            "raw_decision_count": self.raw_decision_count,
            "valid_decision_count": self.valid_decision_count,
            "invalid_decision_count": self.invalid_decision_count,
            "recovered_decision_count": self.recovered_decision_count,
            "invalid_atom_count": self.invalid_atom_count,
            "invalid_context_span_count": self.invalid_context_span_count,
            "input_fact_ids": list(self.input_fact_ids),
            "accepted_fact_ids": list(self.accepted_fact_ids),
            "fallback_fact_ids": list(self.fallback_fact_ids),
            "fail_closed_fact_ids": list(self.fail_closed_fact_ids),
            "non_fact_ids": list(self.non_fact_ids),
            "context_fact_ids": list(self.context_fact_ids),
            "errors": list(self.errors),
            "training_inputs": list(self.training_inputs),
            "training_outputs": list(self.training_outputs),
        }


@dataclass(frozen=True)
class SemanticCompilationResult:
    graph: FactGraph
    report: SemanticCompilationReport


def _structural_section(fact: FactUnit, graph: FactGraph) -> str:
    return next(
        (section.section_type for section in graph.sections if section.section_id == fact.section_id),
        "other",
    )


_KEY_VALUE_LINE_RE = re.compile(
    r"^(?P<label>[^\n:：]{1,48})(?P<separator>\s*[:：]\s*)(?P<value>\S.*)$"
)


def _structural_context_prefix(fact: FactUnit, graph: FactGraph) -> tuple[int, str] | None:
    """Return an exact ``label: value`` transport prefix.

    This is a layout/syntax boundary, not a vocabulary decision.  The label is
    still sent to the semantic model as context, but it can never be published
    as a standalone candidate biography fact.  Apply the same contract inside
    records so ``period: 2020-2022`` cannot become two bullets.  Short all-cap
    keys such as ``CRM: ...`` are retained because the key itself is commonly
    a factual acronym rather than a presentation label.
    """

    match = _KEY_VALUE_LINE_RE.fullmatch(fact.text)
    if match is None or "://" in fact.text:
        return None
    label = match.group("label").strip()
    value = match.group("value").strip()
    if (
        not label
        or not value
        or not _SUBSTANTIVE_RE.search(label)
        or not _SUBSTANTIVE_RE.search(value)
        or re.search(r"\d", label)
        or re.search(r"[,，。！？!?；;]", label)
        or re.fullmatch(r"[A-Z][A-Z0-9+#./_-]{1,7}", label)
    ):
        return None
    value_start = match.start("value")
    return value_start, fact.text[:value_start]


def _fallback_destination(fact: FactUnit, graph: FactGraph) -> tuple[ResumeSection, ResumeField]:
    section = _structural_section(fact, graph)
    if section not in {
        "contact", "summary", "experience", "projects", "research", "activities",
        "education", "skills", "credentials", "awards", "publications", "training",
        "teaching", "additional", "other",
    }:
        section = "other"
    if section == "other":
        section = {
            "identity": "contact", "contact": "contact", "education": "education",
            "skill": "skills", "credential": "credentials", "award": "awards",
            "publication": "publications", "training": "training", "teaching": "teaching",
            "project": "projects",
        }.get(fact.fact_type, "other")
    field: ResumeField = {
        "identity": "name", "contact": "item", "organization": "organization",
        "role": "role", "period": "period", "education": "item", "skill": "skill",
        "degree": "degree", "major": "major", "credential": "credential",
        "project": "title",
    }.get(fact.fact_type, "bullet")  # type: ignore[assignment]
    if section == "summary":
        field = "summary"
    return section, field  # type: ignore[return-value]


def _candidate_payload(fact: FactUnit, graph: FactGraph) -> dict[str, Any]:
    structural_section = _structural_section(fact, graph)
    allowed_records = [
        record.record_id
        for record in graph.records
        if record.source_id == fact.source_id
        and (fact.section_id is None or record.section_id == fact.section_id)
    ]
    payload = {
        "candidate_fact_id": fact.fact_id,
        "source_type": fact.source_type,
        "candidate_text": fact.text,
        "structural_section": structural_section,
        "locked_record_id": fact.record_id,
        "allowed_record_ids": allowed_records,
    }
    structural_prefix = _structural_context_prefix(fact, graph)
    if structural_prefix is not None:
        end, quote = structural_prefix
        payload["structural_context_spans"] = [{
            "quote": quote,
            "reason": "label",
            "char_start": 0,
            "char_end": end,
        }]
    return payload


def _batches(payloads: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    max_chars = max(1000, int(os.getenv("V3_SEMANTIC_BATCH_CHARS", "9000")))
    max_items = max(1, int(os.getenv("V3_SEMANTIC_BATCH_FACTS", "14")))
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0
    for payload in payloads:
        size = len(json.dumps(payload, ensure_ascii=False))
        if current and (len(current) >= max_items or current_chars + size > max_chars):
            batches.append(current)
            current, current_chars = [], 0
        current.append(payload)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def _map_local_span(fact: FactUnit, start: int, end: int) -> list[SourceSpan]:
    spans: list[SourceSpan] = []
    local_cursor = 0
    for parent in fact.spans:
        length = parent.char_end - parent.char_start
        overlap_start = max(start, local_cursor)
        overlap_end = min(end, local_cursor + length)
        if overlap_end > overlap_start:
            spans.append(SourceSpan(
                source_id=parent.source_id,
                char_start=parent.char_start + overlap_start - local_cursor,
                char_end=parent.char_start + overlap_end - local_cursor,
                page=parent.page,
                paragraph_id=parent.paragraph_id,
                node_id=parent.node_id,
            ))
        local_cursor += length
    if sum(span.char_end - span.char_start for span in spans) != end - start:
        raise ValueError("atom cannot be mapped back to the complete source span")
    return spans


_SUBSTANTIVE_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
_PLACEHOLDER_RE = re.compile(
    r"(?:\[[^\]\n]{1,80}(?:\]|$)|【[^】\n]{1,80}(?:】|$))"
)


def _is_placeholder_atom(value: str) -> bool:
    return bool(_PLACEHOLDER_RE.fullmatch(str(value or "").strip()))


def _compile_decision(
    fact: FactUnit,
    decision: SemanticFactDecision,
    graph: FactGraph,
) -> tuple[list[FactUnit], list[str]]:
    errors: list[str] = []
    if decision.candidate_fact_id != fact.fact_id:
        return [], [f"{fact.fact_id}:decision_id_mismatch"]
    if decision.classification == "ambiguous":
        return [], [f"{fact.fact_id}:ambiguous_requires_fallback"]
    if (
        fact.source_type in {"cv", "resume"}
        and decision.classification not in {"fact", "context"}
    ):
        return [], [f"{fact.fact_id}:candidate_cv_reclassified_non_fact"]
    if decision.classification != "fact":
        occupied: list[tuple[int, int]] = []
        for context_index, context in enumerate(decision.context_spans):
            search_from = 0
            location: tuple[int, int] | None = None
            while True:
                start = fact.text.find(context.quote, search_from)
                if start < 0:
                    break
                end = start + len(context.quote)
                if not any(not (end <= left or right <= start) for left, right in occupied):
                    location = (start, end)
                    break
                search_from = start + 1
            if location is None:
                return [], [f"{fact.fact_id}:non_fact_context_not_exact:{context_index}"]
            occupied.append(location)
            start, end = location
            mapped = _map_local_span(fact, start, end)
            if any(
                any(
                    span.char_start <= anchor.span.char_start
                    and anchor.span.char_end <= span.char_end
                    for span in mapped
                )
                for anchor in fact.anchors
            ):
                return [], [f"{fact.fact_id}:non_fact_context_hides_hard_anchor:{context_index}"]
        mask = [False] * len(fact.text)
        for start, end in occupied:
            mask[start:end] = [True] * (end - start)
        remainder = "".join(
            character for index, character in enumerate(fact.text) if not mask[index]
        )
        if _SUBSTANTIVE_RE.search(remainder):
            return [], [f"{fact.fact_id}:non_fact_context_not_complete"]
        classification = {
            "instruction": "instruction",
            "intent": "intent",
            "context": "ineligible",
        }[decision.classification]
        return [fact.model_copy(update={
            "eligible": False,
            "classification": classification,
            "schema_version": SCHEMA_VERSION,
        })], errors

    allowed_records = {
        record.record_id
        for record in graph.records
        if record.source_id == fact.source_id
        and (fact.section_id is None or record.section_id == fact.section_id)
    }
    record_id = fact.record_id
    if record_id is not None and decision.record_id not in {None, record_id}:
        errors.append(f"{fact.fact_id}:locked_record_changed")
    elif record_id is None and decision.record_id is not None:
        if decision.record_id not in allowed_records:
            return [], [f"{fact.fact_id}:record_not_allowed:{decision.record_id}"]
        record_id = decision.record_id

    structural_section = _structural_section(fact, graph)
    cursor = 0
    occupied: list[tuple[int, int]] = []
    eligible_atom_ranges: list[tuple[int, int]] = []
    eligible_destinations: set[tuple[str, str]] = set()
    atoms: list[FactUnit] = []
    structural_prefix = _structural_context_prefix(fact, graph)
    structural_prefix_end = structural_prefix[0] if structural_prefix is not None else 0
    if structural_prefix is not None:
        prefix_end, prefix_quote = structural_prefix
        occupied.append((0, prefix_end))
        atoms.append(FactUnit(
            fact_id=f"{fact.fact_id}:context:structural",
            base_fact_id=fact.fact_id,
            source_id=fact.source_id,
            source_type=fact.source_type,
            fact_type="other",
            text=prefix_quote,
            spans=_map_local_span(fact, 0, prefix_end),
            section_id=fact.section_id,
            record_id=record_id,
            anchors=[],
            eligible=False,
            confidence=fact.confidence,
            classification="ineligible",
            schema_version=SCHEMA_VERSION,
        ))
    for atom_index, atom in enumerate(decision.atoms):
        # Bracketed template values are a source-syntax invariant, not a
        # profession-specific semantic guess.  Recover an exact placeholder
        # child as ineligible evidence instead of discarding valid role/date
        # siblings from the same transport line.  A compound atom that mixes
        # placeholder and factual text remains invalid because its factual
        # boundary is unknown.
        placeholder_atom = _is_placeholder_atom(atom.quote)
        if _PLACEHOLDER_RE.search(atom.quote) and not placeholder_atom:
            return [], [f"{fact.fact_id}:compound_placeholder_atom:{atom_index}"]
        start = fact.text.find(atom.quote, cursor)
        if start < 0:
            return [], [f"{fact.fact_id}:atom_not_exact:{atom_index}"]
        end = start + len(atom.quote)
        raw_end = end
        if structural_prefix_end:
            if end <= structural_prefix_end:
                errors.append(
                    f"{fact.fact_id}:structural_label_emitted_as_fact:{atom_index}"
                )
                cursor = raw_end
                continue
            if start < structural_prefix_end:
                errors.append(
                    f"{fact.fact_id}:structural_label_split_from_fact:{atom_index}"
                )
                start = structural_prefix_end
                atom = atom.model_copy(update={"quote": fact.text[start:end]})
        if any(not (end <= left or right <= start) for left, right in occupied):
            return [], [f"{fact.fact_id}:atom_overlap:{atom_index}"]
        occupied.append((start, end))
        cursor = raw_end
        spans = _map_local_span(fact, start, end)
        if placeholder_atom:
            errors.append(f"{fact.fact_id}:placeholder_emitted_as_fact:{atom_index}")
            atoms.append(FactUnit(
                # Atom-derived context and explicit context spans have
                # independent indexes.  Keep their ID namespaces separate so
                # a placeholder atom at index 0 cannot collide with a label
                # or separator context span at index 0.
                fact_id=f"{fact.fact_id}:context:atom:{atom_index}",
                base_fact_id=fact.fact_id,
                source_id=fact.source_id,
                source_type=fact.source_type,
                fact_type="other",
                text=atom.quote,
                spans=spans,
                section_id=fact.section_id,
                record_id=record_id,
                anchors=[],
                eligible=False,
                confidence=fact.confidence,
                classification="ineligible",
                schema_version=SCHEMA_VERSION,
            ))
            continue
        if is_date_placeholder_range(atom.quote):
            # A date placeholder can establish a record boundary but cannot
            # become a public candidate period.  Preserve it as exact,
            # ineligible training evidence while retaining valid sibling
            # atoms from the same transport line.
            errors.append(f"{fact.fact_id}:date_placeholder_emitted_as_fact:{atom_index}")
            atoms.append(FactUnit(
                fact_id=f"{fact.fact_id}:context:atom:{atom_index}",
                base_fact_id=fact.fact_id,
                source_id=fact.source_id,
                source_type=fact.source_type,
                fact_type="other",
                text=atom.quote,
                spans=spans,
                section_id=fact.section_id,
                record_id=record_id,
                anchors=[],
                eligible=False,
                confidence=fact.confidence,
                classification="ineligible",
                schema_version=SCHEMA_VERSION,
            ))
            continue
        destination_section = atom.destination_section
        if structural_section != "other" and destination_section != structural_section:
            errors.append(f"{fact.fact_id}:structural_section_locked:{destination_section}->{structural_section}")
            destination_section = structural_section  # type: ignore[assignment]
        anchors = [
            anchor for anchor in fact.anchors
            if any(
                parent.char_start <= anchor.span.char_start
                and anchor.span.char_end <= parent.char_end
                for parent in spans
            )
        ]
        atoms.append(FactUnit(
            fact_id=f"{fact.fact_id}:atom:{atom_index}",
            base_fact_id=fact.fact_id,
            source_id=fact.source_id,
            source_type=fact.source_type,
            fact_type=atom.fact_type,
            text=atom.quote,
            spans=spans,
            section_id=fact.section_id,
            record_id=record_id,
            anchors=anchors,
            eligible=True,
            confidence=fact.confidence,
            classification="fact",
            destination_section=destination_section,
            destination_field=atom.destination_field,
            schema_version=SCHEMA_VERSION,
        ))
        eligible_atom_ranges.append((start, end))
        eligible_destinations.add((destination_section, atom.destination_field))
    if not atoms:
        return [], [f"{fact.fact_id}:no_atoms"]

    # Explicit context spans let the model account for labels/separators
    # without turning them into resume facts.  They are exact evidence too,
    # and must not overlap emitted atoms or one another.
    for context_index, context in enumerate(decision.context_spans):
        search_from = 0
        location: tuple[int, int] | None = None
        inside_structural_prefix = False
        while True:
            start = fact.text.find(context.quote, search_from)
            if start < 0:
                break
            end = start + len(context.quote)
            if structural_prefix_end and end <= structural_prefix_end:
                inside_structural_prefix = True
                break
            if not any(not (end <= left or right <= start) for left, right in occupied):
                location = (start, end)
                break
            search_from = start + 1
        if inside_structural_prefix:
            # The deterministic structural context already represents this
            # exact label/separator span; do not duplicate it.
            continue
        if location is None:
            return [], [f"{fact.fact_id}:context_not_exact_or_overlapping:{context_index}"]
        start, end = location
        if not is_date_placeholder_range(context.quote) and any(
            any(
                span.char_start <= anchor.span.char_start
                and anchor.span.char_end <= span.char_end
                for span in _map_local_span(fact, start, end)
            )
            for anchor in fact.anchors
        ):
            return [], [f"{fact.fact_id}:context_hides_hard_anchor:{context_index}"]
        occupied.append(location)
        atoms.append(FactUnit(
            fact_id=f"{fact.fact_id}:context:span:{context_index}",
            base_fact_id=fact.fact_id,
            source_id=fact.source_id,
            source_type=fact.source_type,
            fact_type="other",
            text=context.quote,
            spans=_map_local_span(fact, start, end),
            section_id=fact.section_id,
            record_id=record_id,
            anchors=[],
            eligible=False,
            confidence=fact.confidence,
            classification="instruction" if context.reason == "instruction" else "ineligible",
            schema_version=SCHEMA_VERSION,
        ))
    mask = [False] * len(fact.text)
    for start, end in occupied:
        mask[start:end] = [True] * (end - start)
    remainder = "".join(character for index, character in enumerate(fact.text) if not mask[index])
    if _SUBSTANTIVE_RE.search(remainder):
        # A schema-positive decision may split a sentence into semantic atoms
        # while leaving connective transport text between them.  The
        # deterministic realizer reconstructs the exact source slice from the
        # first atom through the last, so rejecting that whole line only loses
        # source facts.  Accept gaps strictly inside one same-destination atom
        # hull.  Unclaimed leading/trailing content and cross-field splits
        # still fail closed and remain explicit fine-tuning evidence.
        if len(eligible_atom_ranges) < 2 or len(eligible_destinations) != 1:
            return [], [f"{fact.fact_id}:substantive_source_not_covered"]
        hull_start = min(start for start, _end in eligible_atom_ranges)
        hull_end = max(end for _start, end in eligible_atom_ranges)
        outside_hull = "".join(
            character
            for index, character in enumerate(fact.text)
            if not mask[index] and not (hull_start <= index < hull_end)
        )
        if _SUBSTANTIVE_RE.search(outside_hull):
            return [], [f"{fact.fact_id}:substantive_source_not_covered"]
        errors.append(f"{fact.fact_id}:implicit_transport_gap_preserved")
    return atoms, errors


def _fallback_fact(fact: FactUnit, graph: FactGraph) -> FactUnit:
    section, field = _fallback_destination(fact, graph)
    # Query text is a mixed command/fact channel and therefore requires a
    # positive schema-valid semantic decision.  Likewise, a transport fact
    # containing a template placeholder is safe only after the model splits
    # its factual and context spans.  Exact fallback remains available for
    # ordinary CV facts, but these two ambiguous boundaries fail closed.
    fallback_eligible = bool(
        fact.eligible
        and fact.source_type not in {"query", "jd", "template"}
        and not _PLACEHOLDER_RE.search(fact.text)
    )
    return fact.model_copy(update={
        "base_fact_id": fact.base_fact_id or fact.fact_id,
        "destination_section": fact.destination_section or section,
        "destination_field": fact.destination_field or field,
        "eligible": fallback_eligible,
        "classification": fact.classification if fallback_eligible else "ineligible",
        "schema_version": SCHEMA_VERSION,
    })


def _call_semantic_batches(
    batches: list[list[dict[str, Any]]],
    llm_call: Callable[..., dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any] | Exception]]:
    """Invoke independent schema batches with bounded, ordered concurrency."""

    requests = [
        {"schema_version": SCHEMA_VERSION, "candidates": batch}
        for batch in batches
    ]
    if not requests:
        return requests, []
    try:
        configured = int(os.getenv("V3_SEMANTIC_CONCURRENCY", "2"))
    except ValueError:
        configured = 2
    workers = max(1, min(configured, len(requests)))
    requested_max_tokens = max(1024, int(os.getenv("V3_SEMANTIC_MAX_TOKENS", "6144")))
    import logging as _logging
    _sem_logger = _logging.getLogger("v3.semantic_telemetry")

    def invoke(request_payload: dict[str, Any]) -> dict[str, Any]:
        batch_size = len(request_payload.get("candidates", []))
        call_started = time.perf_counter()
        try:
            result = llm_call(
                SemanticCompilationResponse,
                SEMANTIC_SYSTEM_PROMPT,
                json.dumps(request_payload, ensure_ascii=False),
                temperature=0.0,
                max_tokens=requested_max_tokens,
            )
            call_elapsed = time.perf_counter() - call_started
            # Telemetry: success path
            json_complete = bool(
                isinstance(result, dict)
                and result.get("schema_version") == SCHEMA_VERSION
                and isinstance(result.get("decisions"), list)
            )
            decision_count = len(result.get("decisions", [])) if isinstance(result, dict) else 0
            _sem_logger.info(
                "semantic_batch_ok | batch_size=%d | requested_max_tokens=%d | elapsed=%.3fs | json_complete=%s | decision_count=%d",
                batch_size,
                requested_max_tokens,
                call_elapsed,
                json_complete,
                decision_count,
            )
            return result
        except Exception as exc:
            call_elapsed = time.perf_counter() - call_started
            exc_type = type(exc).__name__
            # Classify failure: truncation, deadline, schema, or other
            is_truncation = "truncated" in str(exc) or exc_type == "ContextBudgetError"
            is_deadline = exc_type == "LLMDeadlineExceeded"
            _sem_logger.warning(
                "semantic_batch_fail | batch_size=%d | requested_max_tokens=%d | elapsed=%.3fs | exc_type=%s | truncation=%s | deadline=%s | exc=%s",
                batch_size,
                requested_max_tokens,
                call_elapsed,
                exc_type,
                is_truncation,
                is_deadline,
                str(exc)[:200],
            )
            raise

    if workers == 1:
        results: list[dict[str, Any] | Exception] = []
        for request in requests:
            try:
                results.append(invoke(request))
            except Exception as exc:  # retained as fine-tuning evidence below
                results.append(exc)
        return requests, results

    futures: list[Future[dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v3-semantic") as pool:
        for request in requests:
            # Request deadlines are ContextVars.  Each worker needs its own
            # copied context; sharing one Context across concurrent threads is
            # illegal and losing it would allow calls past the 480s deadline.
            context = copy_context()
            futures.append(pool.submit(context.run, invoke, request))
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(exc)
    return requests, results


def compile_semantics(
    fact_graph: FactGraph,
    *,
    use_llm: bool = True,
    llm_call: Callable[..., dict[str, Any]] | None = None,
) -> SemanticCompilationResult:
    """Compile transport facts into exact semantic atoms using the fixed schema."""

    candidates = [
        fact for fact in fact_graph.facts
        if fact.source_type not in {"jd", "template"}
        and not (fact.classification == "ineligible" and not fact.eligible)
    ]
    payloads = [_candidate_payload(fact, fact_graph) for fact in candidates]
    decisions: dict[str, SemanticFactDecision] = {}
    errors: list[str] = []
    training_inputs: list[dict[str, Any]] = []
    # Keep one output slot per input batch.  Positional alignment is part of
    # the future training dataset contract, including failed/empty responses.
    training_outputs: list[dict[str, Any] | None] = []
    response_batch_count = 0
    schema_valid_batch_count = 0
    raw_decision_count = 0
    valid_decision_count = 0
    invalid_decision_count = 0
    recovered_decision_count = 0
    invalid_atom_count = 0
    invalid_context_span_count = 0
    batches = _batches(payloads) if use_llm and payloads else []
    if use_llm and llm_call is None:
        from server_runtime import call_llm_schema_json

        # A schema violation is valuable fine-tuning evidence, not a reason to
        # spend another full model call inside the request deadline.  The
        # compiler falls back to exact source facts after the first failure.
        def _call_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return call_llm_schema_json(*args, allow_repair=False, **kwargs)

        llm_call = _call_once
    if batches:
        assert llm_call is not None
        batch_requests, batch_results = _call_semantic_batches(batches, llm_call)
    else:
        batch_requests, batch_results = [], []
    training_inputs.extend(batch_requests)
    for batch_index, (batch, raw_result) in enumerate(zip(batches, batch_results)):
        if isinstance(raw_result, Exception):
            training_outputs.append(None)
            errors.append(
                f"batch:{batch_index}:llm_error:"
                f"{type(raw_result).__name__}:{raw_result}"
            )
            continue
        try:
            raw = raw_result
            if not isinstance(raw, dict) or not raw:
                raise ValueError("empty semantic response")
            response_batch_count += 1
            training_outputs.append(raw)
            expected = {item["candidate_fact_id"] for item in batch}
            if raw.get("schema_version") != SCHEMA_VERSION:
                invalid_decision_count += len(expected)
                errors.append(
                    f"batch:{batch_index}:schema_version:{raw.get('schema_version')!r}"
                )
                continue
            raw_decisions = raw.get("decisions")
            if not isinstance(raw_decisions, list) or not raw_decisions:
                invalid_decision_count += len(expected)
                errors.append(f"batch:{batch_index}:decisions_missing_or_empty")
                continue
            raw_decision_count += len(raw_decisions)
            try:
                SemanticCompilationResponse.model_validate(raw)
                schema_valid_batch_count += 1
            except ValidationError:
                # Item-level diagnostics below retain valid siblings and make
                # each rejected item directly usable as fine-tuning evidence.
                pass
            seen: set[str] = set()
            for decision_index, raw_decision in enumerate(raw_decisions):
                recovered = False
                try:
                    decision = SemanticFactDecision.model_validate(raw_decision)
                except ValidationError as exc:
                    candidate_id = (
                        raw_decision.get("candidate_fact_id", "unknown")
                        if isinstance(raw_decision, dict) else "unknown"
                    )
                    # Keep recovery structural and schema-driven: validate
                    # child atoms/spans independently, drop only malformed
                    # children, then revalidate the same frozen decision
                    # contract.  Exact source coverage in _compile_decision
                    # still forces whole-source fallback if a substantive
                    # fragment would otherwise disappear.
                    if not isinstance(raw_decision, dict):
                        invalid_decision_count += 1
                        errors.append(
                            f"batch:{batch_index}:invalid_decision:{decision_index}:"
                            f"{candidate_id}:{exc.error_count()}"
                        )
                        continue
                    raw_atoms = raw_decision.get("atoms")
                    raw_contexts = raw_decision.get("context_spans")
                    if not isinstance(raw_atoms, list) or not isinstance(raw_contexts, list):
                        invalid_decision_count += 1
                        errors.append(
                            f"batch:{batch_index}:invalid_decision:{decision_index}:"
                            f"{candidate_id}:{exc.error_count()}"
                        )
                        continue
                    valid_atoms: list[dict[str, Any]] = []
                    for atom_index, raw_atom in enumerate(raw_atoms):
                        try:
                            valid_atoms.append(
                                SemanticAtomDecision.model_validate(raw_atom).model_dump(mode="json")
                            )
                        except ValidationError as atom_exc:
                            invalid_atom_count += 1
                            errors.append(
                                f"batch:{batch_index}:invalid_atom:{decision_index}:"
                                f"{candidate_id}:{atom_index}:{atom_exc.error_count()}"
                            )
                    valid_contexts: list[dict[str, Any]] = []
                    for context_index, raw_context in enumerate(raw_contexts):
                        try:
                            valid_contexts.append(
                                SemanticContextSpan.model_validate(raw_context).model_dump(mode="json")
                            )
                        except ValidationError as context_exc:
                            invalid_context_span_count += 1
                            errors.append(
                                f"batch:{batch_index}:invalid_context_span:{decision_index}:"
                                f"{candidate_id}:{context_index}:{context_exc.error_count()}"
                            )
                    recovered_payload = dict(raw_decision)
                    recovered_payload["atoms"] = valid_atoms
                    recovered_payload["context_spans"] = valid_contexts
                    try:
                        decision = SemanticFactDecision.model_validate(recovered_payload)
                    except ValidationError:
                        invalid_decision_count += 1
                        errors.append(
                            f"batch:{batch_index}:invalid_decision:{decision_index}:"
                            f"{candidate_id}:{exc.error_count()}"
                        )
                        continue
                    recovered = True
                fact_id = decision.candidate_fact_id
                if fact_id not in expected:
                    invalid_decision_count += 1
                    errors.append(f"batch:{batch_index}:unknown_decision:{fact_id}")
                    continue
                if fact_id in seen or fact_id in decisions:
                    invalid_decision_count += 1
                    errors.append(f"batch:{batch_index}:duplicate_decision:{fact_id}")
                    continue
                seen.add(fact_id)
                decisions[fact_id] = decision
                valid_decision_count += 1
                if recovered:
                    recovered_decision_count += 1
            for missing in sorted(expected - seen):
                errors.append(f"batch:{batch_index}:missing_decision:{missing}")
        except Exception as exc:
            training_outputs.append(None)
            errors.append(f"batch:{batch_index}:llm_error:{type(exc).__name__}:{exc}")

    new_facts: list[FactUnit] = []
    accepted: list[str] = []
    fallback: list[str] = []
    fail_closed: list[str] = []
    non_fact: list[str] = []
    context_only: list[str] = []
    for fact in fact_graph.facts:
        if fact.classification == "ineligible" and not fact.eligible:
            new_facts.append(fact.model_copy(update={"schema_version": SCHEMA_VERSION}))
            non_fact.append(fact.fact_id)
            context_only.append(fact.fact_id)
            continue
        decision = decisions.get(fact.fact_id)
        if decision is None:
            fallback_fact = _fallback_fact(fact, fact_graph)
            new_facts.append(fallback_fact)
            fallback.append(fact.fact_id)
            if fact.eligible and not fallback_fact.eligible:
                fail_closed.append(fact.fact_id)
            continue
        compiled, fact_errors = _compile_decision(fact, decision, fact_graph)
        errors.extend(fact_errors)
        if not compiled:
            fallback_fact = _fallback_fact(fact, fact_graph)
            new_facts.append(fallback_fact)
            fallback.append(fact.fact_id)
            if fact.eligible and not fallback_fact.eligible:
                fail_closed.append(fact.fact_id)
            continue
        new_facts.extend(compiled)
        if any(item.eligible for item in compiled):
            accepted.append(fact.fact_id)
        else:
            non_fact.append(fact.fact_id)
            if decision.classification == "context":
                context_only.append(fact.fact_id)

    record_fact_ids: dict[str, list[str]] = {record.record_id: [] for record in fact_graph.records}
    for fact in new_facts:
        if fact.record_id in record_fact_ids:
            record_fact_ids[fact.record_id].append(fact.fact_id)
    records = [
        record.model_copy(update={"fact_ids": record_fact_ids[record.record_id]})
        for record in fact_graph.records
    ]
    graph = FactGraph(
        documents=fact_graph.documents,
        sections=fact_graph.sections,
        records=records,
        facts=new_facts,
    )
    if not use_llm:
        status = "disabled"
    elif not fallback and not errors:
        status = "success"
    elif accepted:
        status = "partial"
    else:
        status = "fallback"
    report = SemanticCompilationReport(
        schema_version=SCHEMA_VERSION,
        status=status,
        batch_count=len(batches),
        response_batch_count=response_batch_count,
        schema_valid_batch_count=schema_valid_batch_count,
        raw_decision_count=raw_decision_count,
        valid_decision_count=valid_decision_count,
        invalid_decision_count=invalid_decision_count,
        recovered_decision_count=recovered_decision_count,
        invalid_atom_count=invalid_atom_count,
        invalid_context_span_count=invalid_context_span_count,
        input_fact_ids=tuple(fact.fact_id for fact in candidates),
        accepted_fact_ids=tuple(accepted),
        fallback_fact_ids=tuple(fallback),
        fail_closed_fact_ids=tuple(fail_closed),
        non_fact_ids=tuple(non_fact),
        context_fact_ids=tuple(
            context_only
            + [fact.fact_id for fact in new_facts if ":context:" in fact.fact_id]
        ),
        errors=tuple(errors),
        training_inputs=tuple(training_inputs),
        training_outputs=tuple(training_outputs),
    )
    return SemanticCompilationResult(graph=graph, report=report)


__all__ = [
    "SEMANTIC_SYSTEM_PROMPT", "SemanticCompilationReport",
    "SemanticCompilationResult", "compile_semantics",
]
