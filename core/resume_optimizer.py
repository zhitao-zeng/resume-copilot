"""Evidence-preserving bullet optimizer for the V2 resume pipeline.

The optimizer returns patches instead of a complete resume.  Immutable fields
never cross the LLM boundary a second time, which prevents role/company/date
drift and makes a rejected bullet independent from the rest of the document.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any

from llm_gateway import parse_json_content
from server_runtime import call_llm_text, llm_enabled
from v2_schemas import CanonicalResume

logger = logging.getLogger(__name__)


_STRONG_ACTIONS = ("主导", "统筹", "牵头", "独立负责", "全权负责", "从0到1", "从零到一")
_MEDIUM_ACTIONS = ("负责", "组织", "推动", "管理", "设计", "开发", "构建", "实现", "制定")
_WEAK_ACTIONS = ("参与", "协助", "支持", "配合", "接触", "了解", "学习")
_UNSUPPORTED_RESULT_TERMS = (
    "显著提升", "大幅提升", "提升了", "降低了", "减少了", "增长了", "增强了",
    "确保", "保障", "关键依据", "高质量交付", "打通", "性能达标", "降低成本",
    "提高准确率", "提升准确率", "提升效率", "提升用户体验",
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
    for term in _UNSUPPORTED_RESULT_TERMS:
        if term in rewritten and term not in original:
            return False
    return True


def _apply_section_patches(
    optimized: CanonicalResume,
    section: str,
    patches: Any,
) -> int:
    records = getattr(optimized, section, None)
    if not isinstance(records, list) or not isinstance(patches, list):
        return 0
    accepted = 0
    for patch in patches:
        if not isinstance(patch, dict) or not isinstance(patch.get("index"), int):
            continue
        index = patch["index"]
        if index < 0 or index >= len(records):
            continue
        proposed = patch.get("bullets")
        original = list(records[index].bullets)
        if not isinstance(proposed, list) or len(proposed) != len(original):
            continue
        merged: list[str] = []
        for before, after in zip(original, proposed):
            after_text = str(after or "").strip()
            if _safe_rewrite(before, after_text):
                merged.append(after_text)
                accepted += int(after_text != before)
            else:
                merged.append(before)
        records[index].bullets = merged
    return accepted


def optimize_resume(resume: CanonicalResume, jd_text: str = "") -> CanonicalResume:
    if not llm_enabled():
        return resume

    total_bullets = sum(
        len(item.bullets)
        for section in (resume.experience, resume.research, resume.activities, resume.projects)
        for item in section
    )
    if total_bullets < 1:
        logger.info("Optimizer skipped: only %d bullets", total_bullets)
        return resume

    payload = {
        "experience": [item.model_dump() for item in resume.experience],
        "research": [item.model_dump() for item in resume.research],
        "activities": [item.model_dump() for item in resume.activities],
        "projects": [item.model_dump() for item in resume.projects],
    }
    prompt = "请优化以下已校验简历的文字。\n\n"
    if jd_text.strip():
        prompt += f"【目标岗位，仅用于排序和措辞】\n{jd_text.strip()[:1200]}\n\n"
    prompt += "【只读事实与原始 bullets】\n" + json.dumps(payload, ensure_ascii=False)

    try:
        content = call_llm_text(
            OPTIMIZER_SYSTEM_PROMPT,
            prompt,
            temperature=0.0,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.warning("Optimizer LLM call failed: %s", exc)
        return resume

    parsed = parse_json_content(content)
    if not isinstance(parsed, dict) or not parsed:
        logger.warning("Optimizer patch JSON parse failed, len=%d", len(content))
        return resume

    optimized = resume.model_copy(deep=True)
    accepted = 0
    for section in ("experience", "research", "activities", "projects"):
        accepted += _apply_section_patches(optimized, section, parsed.get(section))

    logger.info("Optimizer patches applied: %d/%d bullets", accepted, total_bullets)
    return optimized
