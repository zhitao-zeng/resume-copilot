"""Evidence-preserving bullet optimizer for the V2 resume pipeline.

The optimizer returns patches instead of a complete resume.  Immutable fields
never cross the LLM boundary a second time, which prevents role/company/date
drift and makes a rejected bullet independent from the rest of the document.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from typing import Any

from llm_gateway import LLMDeadlineExceeded, parse_json_content
from server_runtime import call_llm_text, llm_enabled
from semantic_guard import review_entailment_batch
from v2_schemas import CanonicalResume
from diagnostic_trace import trace_event

logger = logging.getLogger(__name__)

_MAX_OPTIMIZER_CONCURRENCY = 2


@dataclass(frozen=True)
class OptimizationOutcome:
    """Optimizer result plus stable provenance for accepted bullet rewrites."""

    resume: CanonicalResume
    trusted_rewrites: dict[str, str] = field(default_factory=dict)
    proposed: int = 0
    accepted: int = 0
    semantic_reviewed: int = 0
    semantic_rejected: int = 0


@dataclass(frozen=True)
class _RewriteProposal:
    section: str
    record_index: int
    bullet_index: int
    before: str
    after: str
    # Grouped proposals are applied atomically per record. ``before`` joins
    # every source bullet referenced by the final claim with newlines so the
    # evidence layer can retain multi-block provenance.
    grouped: bool = False
    source_indices: tuple[int, ...] = ()

    @property
    def path(self) -> str:
        return f"{self.section}[{self.record_index}].bullets[{self.bullet_index}]"


_STRONG_ACTIONS = ("主导", "统筹", "牵头", "独立负责", "全权负责", "从0到1", "从零到一")
_MEDIUM_ACTIONS = ("负责", "组织", "推动", "管理", "设计", "开发", "构建", "实现", "制定")
_WEAK_ACTIONS = ("参与", "协助", "支持", "配合", "接触", "了解", "学习")
_UNSUPPORTED_RESULT_TERMS = (
    "显著提升", "大幅提升", "提升了", "降低了", "减少了", "增长了", "增强了",
    "确保", "保障", "关键依据", "高质量交付", "打通", "性能达标", "降低成本",
    "提高准确率", "提升准确率", "提升效率", "提升用户体验",
)

_FACT_INTRODUCER = re.compile(
    r"(?:使用|采用|基于|借助|运用|利用)\s*"
    r"(?P<fact>[^，。；,;]{2,24}?)(?=进行|开展|完成|实现|输出|分析|设计|优化|管理|，|。|；|,|;|$)"
)
_NAMED_CHINESE_FACT = re.compile(
    r"[A-Za-z0-9+.#/_-]*[\u4e00-\u9fffA-Za-z0-9+.#/_-]{2,18}"
    r"(?:方法|工具|技术|系统|平台|模型|框架|算法|软件|数据库|产品|项目)"
)


OPTIMIZER_SYSTEM_PROMPT = """你是一位保守的简历编辑，只输出经历记录内的 bullets 补丁，不得重写整份简历。

输出 JSON：
{
  "experience": [{"index": 0, "bullets": [{"text": "完整成果句", "source_indices": [0, 1]}]}],
  "research": [{"index": 0, "bullets": [{"text": "完整成果句", "source_indices": [0]}]}],
  "activities": [{"index": 0, "bullets": [{"text": "完整成果句", "source_indices": [0]}]}],
  "projects": [{"index": 0, "bullets": [{"text": "完整成果句", "source_indices": [0, 1]}]}]
}

硬约束：
1. source_indices 是输入 bullets 的 0 起始下标；每个输入下标必须出现且只能出现一次，不得遗漏、重复或越界。
2. 只有属于同一具体事项的动作、方法、交付物、已有结果才能合并。互不相关的职责保持分开；输出条数可以少于输入。
3. 每条尽量写成“动作 + 对象/方法 + 交付物/已有结果”的完整成果句，不要把一个完整事项拆成短语碎片，也不要机械套 STAR 模板。
4. 只能改 bullets；不得输出或修改 summary、公司、组织、岗位、学校、日期、技能、奖项。
5. 责任级别必须保持：原文“参与/协助/支持”不得升级，原文“独立负责/主导/负责”也不得降级；责任级别不同的动作不要合并。
6. 原文没有结果时，不得添加提升、降低、增长、确保、高质量交付等结果。
7. 不得新增数字、工具、技术、业务领域或项目事实。
8. 保留每条原文的关键动作、对象、过程、方法、交付物和结果，不得压成空泛短句。
9. 目标岗位只影响已有事实的排序和措辞，不是候选人事实来源。
只输出 JSON，不要解释。"""


def _numeric_facts(value: Any) -> Counter:
    text = value if isinstance(value, str) else str(value)
    return Counter(re.findall(
        r"(?<![A-Za-z])\d+(?:\.\d+)?(?:%|万|w|k|人|次|个|条|元|年|月|日)?",
        text,
        re.IGNORECASE,
    ))


def _action_level(text: str) -> int:
    if any(token in text for token in _STRONG_ACTIONS):
        return 3
    if any(token in text for token in _MEDIUM_ACTIONS):
        return 2
    if any(token in text for token in _WEAK_ACTIONS):
        return 1
    return 0


def _introduces_unsupported_fact(original: str, rewritten: str) -> bool:
    """Detect newly introduced Chinese methods/tools/entities.

    Character-overlap scores are unsafe for Chinese: a fabricated phrase can
    be short relative to a long otherwise-copied bullet. This guard focuses on
    explicit method/tool constructions while allowing punctuation and sentence
    restructuring.
    """

    original_compact = re.sub(r"[^\w\u4e00-\u9fff+.#/_-]+", "", original).casefold()
    for pattern in (_FACT_INTRODUCER, _NAMED_CHINESE_FACT):
        for match in pattern.finditer(rewritten):
            phrase = match.groupdict().get("fact") or match.group(0)
            compact = re.sub(r"[^\w\u4e00-\u9fff+.#/_-]+", "", phrase).casefold()
            if len(compact) >= 2 and compact not in original_compact:
                return True
    return False


def _safe_rewrite_diagnostics(original: str, rewritten: str) -> tuple[bool, list[str]]:
    original = str(original or "").strip()
    rewritten = str(rewritten or "").strip()
    reasons: list[str] = []
    if not original or not rewritten or len(rewritten) > max(220, len(original) * 3):
        reasons.append("empty_or_too_long")
    original_action = _action_level(original)
    rewritten_action = _action_level(rewritten)
    # Ownership is a candidate fact: it may neither be inflated nor weakened.
    if original_action and rewritten_action != original_action:
        reasons.append("ownership_level_changed")
    elif not original_action and rewritten_action:
        reasons.append("ownership_level_introduced")
    if len(original) >= 20 and len(rewritten) < max(12, int(len(original) * 0.58)):
        reasons.append("source_content_shrunk")
    original_numbers = _numeric_facts(original)
    if any(count > original_numbers[token] for token, count in _numeric_facts(rewritten).items()):
        reasons.append("new_numeric_fact")
    # Product/model/tool names are commonly Latin tokens.  A rewritten bullet
    # may normalize case, but it must not introduce a new named token that was
    # absent from its grounded input bullet.
    latin_pattern = re.compile(r"[A-Za-z][A-Za-z0-9+.#/_-]*")
    original_latin = {token.casefold() for token in latin_pattern.findall(original)}
    rewritten_latin = {token.casefold() for token in latin_pattern.findall(rewritten)}
    if not rewritten_latin.issubset(original_latin):
        reasons.append("new_latin_token")
    if _introduces_unsupported_fact(original, rewritten):
        reasons.append("new_named_method_or_tool")
    for term in _UNSUPPORTED_RESULT_TERMS:
        if term in rewritten and term not in original:
            reasons.append(f"new_result_claim:{term}")
    return not reasons, reasons


def _safe_rewrite(original: str, rewritten: str) -> bool:
    return _safe_rewrite_diagnostics(original, rewritten)[0]


def _fact_bigrams(value: str) -> set[str]:
    compact = re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "")).casefold()
    return {
        compact[index:index + 2]
        for index in range(max(0, len(compact) - 1))
    }


def _source_fact_represented(source: str, candidate: str) -> bool:
    """Conservative, industry-agnostic lossless check for regrouped facts."""

    source_compact = re.sub(
        r"[^\w\u4e00-\u9fff+.#/_-]+", "", str(source or ""),
    ).casefold()
    candidate_compact = re.sub(
        r"[^\w\u4e00-\u9fff+.#/_-]+", "", str(candidate or ""),
    ).casefold()
    if not source_compact or not candidate_compact:
        return False
    if source_compact in candidate_compact:
        return True

    anchor_pattern = re.compile(
        r"\d+(?:\.\d+)?(?:%|万|w|k|人|次|个|条|元|年|月|日)?|"
        r"[A-Za-z][A-Za-z0-9+.#/_-]*",
        re.IGNORECASE,
    )
    source_anchors = {item.casefold() for item in anchor_pattern.findall(source)}
    candidate_anchors = {item.casefold() for item in anchor_pattern.findall(candidate)}
    if not source_anchors.issubset(candidate_anchors):
        return False

    source_bigrams = _fact_bigrams(source)
    candidate_bigrams = _fact_bigrams(candidate)
    recall = len(source_bigrams & candidate_bigrams) / max(1, len(source_bigrams))
    return recall >= 0.52


def _normalize_grouped_surface(value: str) -> str:
    """Remove model-only spacing noise without changing lexical facts."""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=\d)", "", text)
    text = re.sub(
        r"(?<=\d)\s+(?=(?:%|万|w|k|人|次|个|条|元|年|月|日))",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _action_marker_for_level(values: list[str], level: int) -> str:
    tokens = {
        3: _STRONG_ACTIONS,
        2: _MEDIUM_ACTIONS,
        1: _WEAK_ACTIONS,
    }.get(level, ())
    matches = [
        (value.find(token), token)
        for value in values
        for token in tokens
        if token in value
    ]
    return min(matches, default=(0, ""))[1]


def _grouped_patch_proposals(
    resume: CanonicalResume,
    section: str,
    record_index: int,
    proposed: list[Any],
) -> list[_RewriteProposal] | None:
    """Parse one lossless variable-count record rewrite, or reject it whole."""

    records = getattr(resume, section)
    original = [str(item or "").strip() for item in records[record_index].bullets]
    if not original or not proposed or not all(isinstance(item, dict) for item in proposed):
        return None

    parsed: list[tuple[str, tuple[int, ...]]] = []
    used_indices: list[int] = []
    for item in proposed:
        text = str(item.get("text", "") or "").strip()
        raw_indices = item.get("source_indices")
        if (
            not text
            or not isinstance(raw_indices, list)
            or not raw_indices
            or not all(isinstance(index, int) for index in raw_indices)
        ):
            return None
        indices = tuple(raw_indices)
        if len(set(indices)) != len(indices) or any(
            index < 0 or index >= len(original) for index in indices
        ):
            return None
        parsed.append((text, indices))
        used_indices.extend(indices)

    # This is the central lossless contract. The model may regroup facts but
    # cannot make any input bullet disappear or count it twice.
    if sorted(used_indices) != list(range(len(original))):
        return None
    if len({re.sub(r"\W+", "", text).casefold() for text, _ in parsed}) != len(parsed):
        return None

    proposals: list[_RewriteProposal] = []
    for bullet_index, (after, indices) in enumerate(parsed):
        after = _normalize_grouped_surface(after)
        source_parts = [original[index] for index in indices]
        source_action_levels = {
            level for level in (_action_level(item) for item in source_parts) if level
        }
        # A single surface verb cannot safely preserve conflicting ownership
        # levels. Keep those responsibilities as separate output bullets.
        if len(source_action_levels) > 1:
            return None
        candidate_action = _action_level(after)
        if source_action_levels:
            expected_action = next(iter(source_action_levels))
            if not candidate_action:
                # The model sometimes keeps the exact action and object but
                # starts the sentence with a method phrase. Reinsert the
                # source's own ownership marker; this is deterministic source
                # preservation, not an inferred responsibility upgrade.
                marker = _action_marker_for_level(source_parts, expected_action)
                after = f"{marker}{after}" if marker else after
                candidate_action = _action_level(after)
            if candidate_action != expected_action:
                return None
        elif candidate_action:
            return None

        before = "\n".join(source_parts)
        safe, reasons = _safe_rewrite_diagnostics(before, after)
        # Lossless per-source recall below is more precise for consolidation
        # than a raw total-length ratio, so that one generic shrink diagnostic
        # may be ignored while every actual fact still has to survive.
        material_reasons = [
            reason for reason in reasons if reason != "source_content_shrunk"
        ]
        missing_sources = [
            index for index, source_part in zip(indices, source_parts)
            if not _source_fact_represented(source_part, after)
        ]
        # Unchanged bullets are valid members of an otherwise regrouped
        # record. Rejecting one would roll back the genuinely improved sibling
        # bullets because grouped records are intentionally atomic.
        accepted = not material_reasons and not missing_sources
        trace_event(
            "optimizer_grouped_hard_gate",
            path=f"{section}[{record_index}].bullets[{bullet_index}]",
            source_indices=list(indices),
            before=before,
            after=after,
            accepted=accepted,
            reasons=material_reasons + (
                [f"missing_source_indices:{missing_sources}"] if missing_sources else []
            ),
        )
        if not accepted:
            return None
        proposals.append(_RewriteProposal(
            section=section,
            record_index=record_index,
            bullet_index=bullet_index,
            before=before,
            after=after,
            grouped=True,
            source_indices=indices,
        ))
    return proposals


def _rewrite_requires_semantic_review(original: str, rewritten: str) -> bool:
    """Flag large but hard-rule-safe paraphrases for bounded LLM review."""

    def bigrams(value: str) -> set[str]:
        compact = re.sub(r"[^\w\u4e00-\u9fff]+", "", value).casefold()
        return {compact[index:index + 2] for index in range(max(0, len(compact) - 1))}

    source = bigrams(original)
    candidate = bigrams(rewritten)
    if not source or not candidate:
        return True
    generated_coverage = len(source & candidate) / len(candidate)
    source_recall = len(source & candidate) / len(source)
    # Moderate expansion is normal when a terse source bullet is turned into
    # a readable sentence.  Review only when most candidate wording is new or
    # a material part of the source disappeared; the previous 0.58 generated
    # coverage threshold classified ordinary connective wording as risky.
    return generated_coverage < 0.35 or source_recall < 0.45


def _semantic_guard_mode() -> str:
    value = os.getenv("LLM_SEMANTIC_GUARD_MODE", "high_risk").strip().lower()
    return value if value in {"off", "high_risk", "all"} else "high_risk"


def _section_patch_proposals(
    resume: CanonicalResume,
    section: str,
    patches: Any,
) -> list[_RewriteProposal]:
    records = getattr(resume, section, None)
    if not isinstance(records, list) or not isinstance(patches, list):
        return []
    proposals: list[_RewriteProposal] = []
    for patch in patches:
        if not isinstance(patch, dict) or not isinstance(patch.get("index"), int):
            continue
        record_index = patch["index"]
        if record_index < 0 or record_index >= len(records):
            continue
        proposed = patch.get("bullets")
        original = list(records[record_index].bullets)
        if isinstance(proposed, list) and proposed and all(
            isinstance(item, dict) for item in proposed
        ):
            grouped = _grouped_patch_proposals(
                resume,
                section,
                record_index,
                proposed,
            )
            if grouped is None:
                trace_event(
                    "optimizer_patch_shape_rejected",
                    section=section,
                    record_index=record_index,
                    original=original,
                    proposed=proposed,
                    reason="invalid_or_lossy_grouped_rewrite",
                )
                continue
            proposals.extend(grouped)
            continue
        if not isinstance(proposed, list) or len(proposed) != len(original):
            trace_event(
                "optimizer_patch_shape_rejected",
                section=section,
                record_index=record_index,
                original=original,
                proposed=proposed,
                reason="bullet_count_or_type_mismatch",
            )
            continue
        for bullet_index, (before, after) in enumerate(zip(original, proposed)):
            before_text = str(before or "").strip()
            after_text = str(after or "").strip()
            safe, reasons = _safe_rewrite_diagnostics(before_text, after_text)
            trace_event(
                "optimizer_proposal_hard_gate",
                path=f"{section}[{record_index}].bullets[{bullet_index}]",
                before=before_text,
                after=after_text,
                accepted=bool(after_text != before_text and safe),
                reasons=(reasons if after_text != before_text else ["unchanged"]),
            )
            if after_text == before_text or not safe:
                continue
            proposals.append(_RewriteProposal(
                section=section,
                record_index=record_index,
                bullet_index=bullet_index,
                before=before_text,
                after=after_text,
            ))
    return proposals


def _apply_section_patches(
    optimized: CanonicalResume,
    section: str,
    patches: Any,
) -> int:
    proposals = _section_patch_proposals(optimized, section, patches)
    records = getattr(optimized, section, [])
    for proposal in proposals:
        records[proposal.record_index].bullets[proposal.bullet_index] = proposal.after
    return len(proposals)


def _build_optimizer_batches(resume: CanonicalResume) -> list[dict[str, list[dict[str, Any]]]]:
    """Split long resumes into independently recoverable edit batches."""

    batches: list[dict[str, list[dict[str, Any]]]] = []
    current = {section: [] for section in ("experience", "research", "activities", "projects")}
    current_bullets = 0
    current_chars = 0
    for section in current:
        for index, record in enumerate(getattr(resume, section)):
            dumped = record.model_dump()
            dumped["index"] = index
            bullet_count = len(record.bullets)
            char_count = sum(len(str(value)) for value in record.bullets)
            if current_bullets and (
                current_bullets + bullet_count > 8 or current_chars + char_count > 2500
            ):
                batches.append(current)
                current = {name: [] for name in current}
                current_bullets = 0
                current_chars = 0
            current[section].append(dumped)
            current_bullets += bullet_count
            current_chars += char_count
    if current_bullets:
        batches.append(current)
    return batches


def _split_optimizer_batch(
    payload: dict[str, list[dict[str, Any]]],
) -> list[dict[str, list[dict[str, Any]]]]:
    records = [
        (section, record)
        for section in ("experience", "research", "activities", "projects")
        for record in payload.get(section, [])
    ]
    if len(records) < 2:
        return []
    middle = len(records) // 2
    result: list[dict[str, list[dict[str, Any]]]] = []
    for subset in (records[:middle], records[middle:]):
        child = {section: [] for section in payload}
        for section, record in subset:
            child[section].append(record)
        result.append(child)
    return result


def _optimizer_concurrency() -> int:
    """Return the bounded fan-out supported by the 40 GiB runtime profile."""

    try:
        configured = int(os.getenv("LLM_OPTIMIZER_CONCURRENCY", "2"))
    except (TypeError, ValueError):
        configured = 2
    return max(1, min(_MAX_OPTIMIZER_CONCURRENCY, configured))


def _build_optimizer_prompt(
    payload: dict[str, list[dict[str, Any]]],
    jd_text: str,
) -> str:
    prompt = "请优化以下已校验简历的文字。每条记录中的 index 是整份简历的稳定索引，输出时必须原样使用。\n\n"
    if jd_text.strip():
        prompt += f"【目标岗位，仅用于排序和措辞】\n{jd_text.strip()[:1600]}\n\n"
    return prompt + "【只读事实与原始 bullets】\n" + json.dumps(
        payload,
        ensure_ascii=False,
    )


def _optimize_batch(
    payload: dict[str, list[dict[str, Any]]],
    jd_text: str,
) -> dict[str, Any]:
    """Call and parse one independent optimizer batch in a worker thread."""
    user_prompt = _build_optimizer_prompt(payload, jd_text)
    trace_event(
        "optimizer_batch_request",
        payload=payload,
        jd_text=jd_text,
        system_prompt=OPTIMIZER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=4096,
    )
    content = call_llm_text(
        OPTIMIZER_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.0,
        max_tokens=4096,
    )
    parsed = parse_json_content(content)
    trace_event("optimizer_batch_response", content=content, parsed=parsed)
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(f"Optimizer batch JSON parse failed, len={len(content)}")
    return parsed


def _patches_for_payload(
    payload: dict[str, list[dict[str, Any]]],
    section: str,
    patches: Any,
) -> list[dict[str, Any]]:
    """Reject patches that target records outside their isolated batch."""

    if not isinstance(patches, list):
        return []
    allowed_indices = {
        record.get("index")
        for record in payload.get(section, [])
        if isinstance(record, dict) and isinstance(record.get("index"), int)
    }
    return [
        patch
        for patch in patches
        if isinstance(patch, dict) and patch.get("index") in allowed_indices
    ]


def optimize_resume_with_provenance(
    resume: CanonicalResume,
    jd_text: str = "",
) -> OptimizationOutcome:
    if not llm_enabled():
        return OptimizationOutcome(resume=resume)

    total_bullets = sum(
        len(item.bullets)
        for section in (resume.experience, resume.research, resume.activities, resume.projects)
        for item in section
    )
    if total_bullets < 1:
        logger.info("Optimizer skipped: only %d bullets", total_bullets)
        return OptimizationOutcome(resume=resume)

    optimized = resume.model_copy(deep=True)
    batches = _build_optimizer_batches(resume)
    initial_batch_count = len(batches)
    pending: list[tuple[tuple[int, ...], dict[str, list[dict[str, Any]]]]] = [
        ((index,), payload) for index, payload in enumerate(batches)
    ]
    completed: dict[
        tuple[int, ...],
        tuple[dict[str, list[dict[str, Any]]], dict[str, Any]],
    ] = {}
    attempted = 0
    worker_count = _optimizer_concurrency()
    deadline_reached = False
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="resume-optimizer",
    ) as executor:
        while pending and not deadline_reached:
            wave = pending[:worker_count]
            del pending[:len(wave)]
            attempted += len(wave)
            futures = [
                (
                    key,
                    payload,
                    executor.submit(
                        copy_context().run,
                        _optimize_batch,
                        payload,
                        jd_text,
                    ),
                )
                for key, payload in wave
            ]
            retries: list[
                tuple[tuple[int, ...], dict[str, list[dict[str, Any]]]]
            ] = []
            for key, payload, future in futures:
                try:
                    completed[key] = (payload, future.result())
                except LLMDeadlineExceeded as exc:
                    deadline_reached = True
                    logger.warning(
                        "Optimizer stopped at request deadline; kept remaining original wording: %s",
                        exc,
                    )
                except Exception as exc:
                    children = _split_optimizer_batch(payload)
                    if children:
                        retries.extend(
                            (key + (child_index,), child)
                            for child_index, child in enumerate(children)
                        )
                        logger.warning(
                            "Optimizer batch failed and was split into %d smaller batches: %s",
                            len(children),
                            exc,
                        )
                    else:
                        logger.warning(
                            "Optimizer single-record batch failed; kept original wording: %s",
                            exc,
                        )
            if not deadline_reached:
                # Retry failed child batches before later records. Successful
                # results are applied only after all workers finish, on the
                # caller thread and in stable source order.
                pending[0:0] = retries

    proposals: list[_RewriteProposal] = []
    for key in sorted(completed):
        payload, parsed = completed[key]
        for section in ("experience", "research", "activities", "projects"):
            proposals.extend(_section_patch_proposals(
                resume,
                section,
                _patches_for_payload(payload, section, parsed.get(section)),
            ))

    mode = _semantic_guard_mode()
    review_candidates = [
        proposal for proposal in proposals
        if mode == "all" or (
            mode == "high_risk"
            and _rewrite_requires_semantic_review(proposal.before, proposal.after)
        )
    ]
    reviewed = review_entailment_batch([
        (proposal.before, proposal.after) for proposal in review_candidates
    ]) if review_candidates else []
    review_by_path = (
        {proposal.path: verdict for proposal, verdict in zip(review_candidates, reviewed)}
        if reviewed is not None else {}
    )

    trusted_rewrites: dict[str, str] = {}
    semantic_rejected = 0
    accepted_proposal_ids: set[int] = set()
    for proposal in proposals:
        high_risk = _rewrite_requires_semantic_review(proposal.before, proposal.after)
        needs_review = mode == "all" or (mode == "high_risk" and high_risk)
        verdict = review_by_path.get(proposal.path) if needs_review else True
        # A high-risk rewrite must receive an affirmative review.  Low-risk
        # rewrites remain governed by deterministic rules when an optional
        # all-mode review is unavailable.
        if verdict is False or (high_risk and verdict is not True):
            semantic_rejected += 1
            trace_event(
                "optimizer_semantic_gate",
                path=proposal.path,
                before=proposal.before,
                after=proposal.after,
                high_risk=high_risk,
                verdict=verdict,
                accepted=False,
            )
            continue
        accepted_proposal_ids.add(id(proposal))
        trace_event(
            "optimizer_semantic_gate",
            path=proposal.path,
            before=proposal.before,
            after=proposal.after,
            high_risk=high_risk,
            verdict=verdict,
            accepted=True,
        )

    grouped_records: dict[tuple[str, int], list[_RewriteProposal]] = {}
    for proposal in proposals:
        if proposal.grouped:
            grouped_records.setdefault(
                (proposal.section, proposal.record_index), [],
            ).append(proposal)
            continue
        if id(proposal) not in accepted_proposal_ids:
            continue
        records = getattr(optimized, proposal.section)
        records[proposal.record_index].bullets[proposal.bullet_index] = proposal.after
        trusted_rewrites[proposal.path] = proposal.before

    for (section, record_index), record_proposals in grouped_records.items():
        # Regrouping changes positional meaning, so partial application would
        # be unsafe. One failed claim rolls back the entire record.
        if not all(id(proposal) in accepted_proposal_ids for proposal in record_proposals):
            trace_event(
                "optimizer_grouped_record_reverted",
                section=section,
                record_index=record_index,
                reason="semantic_review_rejected_or_unavailable",
            )
            continue
        ordered = sorted(record_proposals, key=lambda item: item.bullet_index)
        records = getattr(optimized, section)
        records[record_index].bullets = [proposal.after for proposal in ordered]
        for proposal in ordered:
            trusted_rewrites[proposal.path] = proposal.before

    accepted = len(trusted_rewrites)

    logger.info(
        "Optimizer patches applied: %d/%d bullets across %d initial/%d attempted batch(es); semantic_reviewed=%d rejected=%d",
        accepted,
        total_bullets,
        initial_batch_count,
        attempted,
        len(review_candidates) if reviewed is not None else 0,
        semantic_rejected,
    )
    return OptimizationOutcome(
        resume=optimized,
        trusted_rewrites=trusted_rewrites,
        proposed=len(proposals),
        accepted=accepted,
        semantic_reviewed=len(review_candidates) if reviewed is not None else 0,
        semantic_rejected=semantic_rejected,
    )


def optimize_resume(resume: CanonicalResume, jd_text: str = "") -> CanonicalResume:
    """Compatibility wrapper returning only the optimized resume."""

    return optimize_resume_with_provenance(resume, jd_text).resume
