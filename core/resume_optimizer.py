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


OPTIMIZER_SYSTEM_PROMPT = """你是一位保守的简历编辑，只输出局部文字补丁，不得重写整份简历。

输出 JSON：
{
  "experience": [{"index": 0, "bullets": ["与原数组一一对应"]}],
  "research": [{"index": 0, "bullets": ["与原数组一一对应"]}],
  "activities": [{"index": 0, "bullets": ["与原数组一一对应"]}],
  "projects": [{"index": 0, "bullets": ["与原数组一一对应"]}]
}

硬约束：
1. 每个 bullets 数组长度和顺序必须与输入完全相同，一条原文对应一条改写。
2. 只能改 bullets；不得输出或修改 summary、公司、组织、岗位、学校、日期、技能、奖项。
3. 责任级别必须保持：原文“参与/协助/支持”不得升级，原文“独立负责/主导/负责”也不得降级。
4. 原文没有结果时，不得添加提升、降低、增长、确保、高质量交付等结果。
5. 不得新增数字、工具、技术、业务领域或项目事实。
6. 重点是压缩重复、改善句式和按目标岗位突出已有事实；不需要为了 STAR 强行补结果。
7. 保留原文中的关键过程、方法、交付物和结果，不得把多项事实压成空泛短句。
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


def _safe_rewrite(original: str, rewritten: str) -> bool:
    original = str(original or "").strip()
    rewritten = str(rewritten or "").strip()
    if not original or not rewritten or len(rewritten) > max(220, len(original) * 3):
        return False
    original_action = _action_level(original)
    rewritten_action = _action_level(rewritten)
    # Ownership is a candidate fact: it may neither be inflated nor weakened.
    if original_action and rewritten_action != original_action:
        return False
    if len(original) >= 20 and len(rewritten) < max(12, int(len(original) * 0.58)):
        return False
    original_numbers = _numeric_facts(original)
    if any(count > original_numbers[token] for token, count in _numeric_facts(rewritten).items()):
        return False
    # Product/model/tool names are commonly Latin tokens.  A rewritten bullet
    # may normalize case, but it must not introduce a new named token that was
    # absent from its grounded input bullet.
    latin_pattern = re.compile(r"[A-Za-z][A-Za-z0-9+.#/_-]*")
    original_latin = {token.casefold() for token in latin_pattern.findall(original)}
    rewritten_latin = {token.casefold() for token in latin_pattern.findall(rewritten)}
    if not rewritten_latin.issubset(original_latin):
        return False
    if _introduces_unsupported_fact(original, rewritten):
        return False
    for term in _UNSUPPORTED_RESULT_TERMS:
        if term in rewritten and term not in original:
            return False
    return True


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
        if not isinstance(proposed, list) or len(proposed) != len(original):
            continue
        for bullet_index, (before, after) in enumerate(zip(original, proposed)):
            before_text = str(before or "").strip()
            after_text = str(after or "").strip()
            if after_text == before_text or not _safe_rewrite(before_text, after_text):
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

    content = call_llm_text(
        OPTIMIZER_SYSTEM_PROMPT,
        _build_optimizer_prompt(payload, jd_text),
        temperature=0.0,
        max_tokens=4096,
    )
    parsed = parse_json_content(content)
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
    for proposal in proposals:
        high_risk = _rewrite_requires_semantic_review(proposal.before, proposal.after)
        needs_review = mode == "all" or (mode == "high_risk" and high_risk)
        verdict = review_by_path.get(proposal.path) if needs_review else True
        # A high-risk rewrite must receive an affirmative review.  Low-risk
        # rewrites remain governed by deterministic rules when an optional
        # all-mode review is unavailable.
        if verdict is False or (high_risk and verdict is not True):
            semantic_rejected += 1
            continue
        records = getattr(optimized, proposal.section)
        records[proposal.record_index].bullets[proposal.bullet_index] = proposal.after
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
