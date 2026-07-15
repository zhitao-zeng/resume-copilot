"""LLM-based repair functions replacing regex keyword rules.

Each function costs ~1 LLM call (~1-1.5s on Qwen3.5-9B), replacing
hundreds of lines of brittle keyword/regex maintenance.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from server_runtime import call_llm_typed, llm_enabled, logger, sanitize_user_text


# ── 1. Parse repair ─────────────────────────────────────────────────────────

_PARSE_REPAIR_SYSTEM = """你是简历数据修复专家。给定原始简历文本和初步解析的JSON，修复所有解析错误。

修复项：
1. education: 删除推荐人/指导教师/导师/研究员相关的虚假条目。保留真实的教育经历。
   如果同一学校有多个学位(本科+硕士)，保留两者。
2. projects: 删除纯英文例句(如"The car is moving very fastly")、空壳项目。
   合并重复项目(名称相似或bullet重叠)。
3. publications: 删除被误分类的技能行、教育行、枚举片段。
4. 填充education中缺失的degree、major、period（从原文提取）。
5. 修复meta字段：姓名、邮箱、电话等。
6. 删除任何LLM思考链/系统指令泄露到数据字段的内容。

只输出修复后的JSON，格式为：
{"education": [...], "projects": [...], "publications": [...], "meta": {...}}
不需要输出experience和skills（那些不需要修复）。
"""


def llm_repair_parse_errors(
    resume_data: dict[str, Any],
    raw_text: str,
) -> dict[str, Any]:
    """Replace ~350 lines of _repair_common_parse_errors with 1 LLM call."""
    if not llm_enabled():
        return resume_data

    _repair_input = json.dumps({
        "education": resume_data.get('education', []),
        "projects": resume_data.get('projects', []),
        "publications": resume_data.get('publications', []),
        "meta": resume_data.get('meta', {}),
    }, ensure_ascii=False)

    prompt = (
        "【原始简历文本】\n"
        f"{sanitize_user_text(raw_text)[:8000]}\n\n"
        "【当前解析JSON（需要修复的字段）】\n"
        f"{_repair_input}\n\n"
        "请修复上述解析错误，只输出修复后的JSON。"
    )

    try:
        result = call_llm_typed(  # type: ignore[no-untyped-call]
            _ParseRepairOutput,
            _PARSE_REPAIR_SYSTEM,
            prompt,
            temperature=0.1,
            max_tokens=4096,
        )
        if not isinstance(result, dict) or not result:
            logger.warning("LLM parse repair returned empty, keeping original")
            return resume_data

        # Merge repair results back
        repaired = dict(resume_data)
        for key in ("education", "projects", "publications", "meta"):
            if key in result and result[key]:
                repaired[key] = result[key]
        return repaired
    except Exception as exc:
        logger.warning("LLM parse repair failed: %s", exc)
        return resume_data


from pydantic import BaseModel, Field


class _ParseRepairOutput(BaseModel):
    education: list[dict[str, Any]] = Field(default_factory=list)
    projects: list[dict[str, Any]] = Field(default_factory=list)
    publications: list[dict[str, Any]] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


# ── 2. Fabrication check ────────────────────────────────────────────────────

_FABRICATION_SYSTEM = """你是简历事实核查专家。逐条对比原始简历文本和结构化JSON，只标记"明显不在原文中"的编造内容。

容忍度规则（——这些不算编造——）：
1. 大小写差异：Pytorch vs PyTorch, Sql vs SQL → 不算编造
2. 简写/全称：GEE vs Google Earth Engine → 不算编造
3. 常见技术栈推断：原文有"机器学习"→ JSON中有"scikit-learn" → 不算编造
4. 日期格式差异：05/2025 vs 05-2025 → 不算编造
5. 学校名在原文的任意位置出现过 → 不算编造
6. 项目名能匹配原文中任意5个以上的中文字符 → 不算编造

只有这些才算编造：
- 原文完全没有的公司名、学校名
- 原文完全没有的数字/百分比/指标
- 原文完全没有的项目名称（无法匹配任何原文片段）
- 学位/专业原文完全没有提及

只输出JSON对象(带details字段)。如果没有编造项，输出 {"details": []}。
"""


def llm_check_fabrication(
    source_text: str,
    resume_data: dict[str, Any],
) -> dict[str, Any]:
    """Replace ~200 lines of check_fabrication_heuristic with 1 LLM call.

    Returns: {"fabrication_found": bool, "details": [...]}
    """
    if not llm_enabled() or not source_text or not source_text.strip():
        return {"fabrication_found": False, "details": []}

    # Only send the fields that matter for fabrication checking
    check_data = {
        "experience": [
            {"company": e.get("company", ""), "role": e.get("role", ""),
             "period": e.get("period", ""), "function_description": str(e.get("function_description", ""))[:200],
             "result_description": str(e.get("result_description", ""))[:200]}
            for e in resume_data.get("experience", []) if isinstance(e, dict)
        ][:10],
        "education": [
            {"school": e.get("school", ""), "degree": e.get("degree", ""),
             "major": e.get("major", ""), "period": e.get("period", "")}
            for e in resume_data.get("education", []) if isinstance(e, dict)
        ][:8],
        "projects": [
            {"name": p.get("name", ""), "company": p.get("company", ""),
             "period": p.get("period", ""), "description": str(p.get("description", ""))[:200]}
            for p in resume_data.get("projects", []) if isinstance(p, dict)
        ][:8],
        "skills": resume_data.get("skills", {}),
        "meta": {
            k: v for k, v in resume_data.get("meta", {}).items()
            if k in ("name", "email", "phone", "target_role")
        },
    }

    prompt = (
        "【原始简历文本（事实来源）】\n"
        f"{sanitize_user_text(source_text)[:8000]}\n\n"
        "【结构化JSON（需要核查的字段）】\n"
        f"{json.dumps(check_data, ensure_ascii=False)}\n\n"
        "请逐条对比，列出所有不在原文中的字段。如果没有编造项，输出 {\"details\": []}。"
    )

    try:
        result = call_llm_typed(  # type: ignore[no-untyped-call]
            _FabricationOutput,
            _FABRICATION_SYSTEM,
            prompt,
            temperature=0.1,
            max_tokens=4096,
            prefill='{"details": [',
        )
        # 安全兜底: LLM可能仍输出[]，转换为dict
        if isinstance(result, list):
            result = {"details": result}
        details = result.get("details", []) if isinstance(result, dict) else []
        if not isinstance(details, list):
            details = []
        return {
            "fabrication_found": len(details) > 0,
            "details": details,
        }
    except Exception as exc:
        logger.debug("LLM fabrication check failed (non-critical): %s", exc)
        return {"fabrication_found": False, "details": []}


class _FabricationItem(BaseModel):
    type: str = ""
    content: str = ""
    reason: str = ""


class _FabricationOutput(BaseModel):
    details: list[_FabricationItem] = Field(default_factory=list)


# ── 3. Conflict resolution ──────────────────────────────────────────────────

_CONFLICT_SYSTEM = """你是简历时间线分析专家。判断简历中的时间冲突是否为真实冲突。

规则：
1. 实习/暑期/part-time工作与在校时间重叠 → 正常，不算冲突。
2. 同一人在同一期间有多个角色（如学生+实习） → 不算冲突。
3. 双学位/辅修与主修时间重叠且学校相同 → 不算冲突。
4. 教育经历之间时间有重叠但学校不同 → 可能是交换/暑校，不算冲突。
5. 只有明确排他性的冲突（如同一人同一时间在两个不同全职公司工作）才算真实冲突。

输入是已检测出重叠的字段列表。对每条，判定：
- is_real: true=真实冲突, false=正常情况
- reason: 简短说明

只输出JSON数组，没有真实冲突时输出 []。
"""


def llm_resolve_conflicts(
    conflicts: list[dict[str, Any]],
    resume_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Filter out false-positive conflicts using LLM judgment."""
    if not llm_enabled() or not conflicts:
        return conflicts

    prompt = (
        "【简历数据】\n"
        f"education: {json.dumps(resume_data.get('education', []), ensure_ascii=False)[:2000]}\n"
        f"experience: {json.dumps(resume_data.get('experience', []), ensure_ascii=False)[:2000]}\n\n"
        "【检测出的时间冲突】\n"
        f"{json.dumps(conflicts, ensure_ascii=False)}\n\n"
        "请判断每条冲突是否为真实冲突。"
    )

    try:
        result = call_llm_typed(  # type: ignore[no-untyped-call]
            _ConflictOutput,
            _CONFLICT_SYSTEM,
            prompt,
            temperature=0.1,
            max_tokens=1024,
        )
        resolved = result.get("resolved", []) if isinstance(result, dict) else []
        if not isinstance(resolved, list):
            return conflicts
        if not resolved:
            return []  # LLM says all are false positives
        # Return only real conflicts
        real = [r for r in resolved if isinstance(r, dict) and r.get("is_real")]
        return real
    except Exception as exc:
        logger.warning("LLM conflict resolution failed: %s", exc)
        return conflicts


class _ConflictItem(BaseModel):
    is_real: bool = False
    field: str = ""
    description: str = ""
    reason: str = ""


class _ConflictOutput(BaseModel):
    resolved: list[_ConflictItem] = Field(default_factory=list)


# ── 4. Missing field enhancement ────────────────────────────────────────────

_MISSING_SYSTEM = """你是简历审核专家。判断缺失字段的真正原因。

对每个缺失字段，判断：
1. "parse_lost": 原文有该信息但解析丢失 → 从原文提取并补充
2. "truly_missing": 原文确实没有 → 保留为missing
3. "not_applicable": 该字段不适用于此人（如学生无需"工作经验年限"）

只输出JSON数组。"""


def llm_enhance_missing_fields(
    missing_fields: list[dict[str, Any]],
    resume_data: dict[str, Any],
    raw_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate parse errors from truly missing fields using LLM.

    Returns: (still_missing, found_from_text)
    """
    if not llm_enabled() or not missing_fields:
        return missing_fields, []

    prompt = (
        "【原始简历文本】\n"
        f"{sanitize_user_text(raw_text)[:4000]}\n\n"
        "【缺失字段列表】\n"
        f"{json.dumps([{'field': m.get('field',''), 'label': m.get('label',''), 'reason': m.get('reason','')} for m in missing_fields if isinstance(m, dict)], ensure_ascii=False)}\n\n"
        "请判断每个缺失字段的原因，并尝试从原文提取。"
    )

    try:
        result = call_llm_typed(  # type: ignore[no-untyped-call]
            _MissingOutput,
            _MISSING_SYSTEM,
            prompt,
            temperature=0.1,
            max_tokens=2048,
        )
        verdicts = result.get("verdicts", []) if isinstance(result, dict) else []
        if not isinstance(verdicts, list):
            return missing_fields, []

        still_missing = []
        found_from_text = []
        for v in verdicts:
            if not isinstance(v, dict):
                continue
            if v.get("verdict") == "parse_lost" and v.get("extracted_value"):
                found_from_text.append({
                    "field": v.get("field", ""),
                    "label": v.get("label", ""),
                    "extracted_value": v.get("extracted_value", ""),
                })
            else:
                still_missing.append({
                    "field": v.get("field", ""),
                    "label": v.get("label", ""),
                    "reason": v.get("reason", v.get("reason_llm", "")),
                })
        return still_missing, found_from_text
    except Exception as exc:
        logger.warning("LLM missing field enhancement failed: %s", exc)
        return missing_fields, []


# ── 5. Issue type classification ─────────────────────────────────────────────

_ISSUE_CLASSIFY_SYSTEM = """你是简历审核issue分类器。判断issue能否由优化器直接改写，还是必须用户补充信息。

判定规则（严格）：
只有 issue 明确指向"原文不存在、用户必须提供"的信息时，才判 needs_data：
  - 原文没有的具体数字、百分比、指标数值
  - 原文没写的方法名、模型版本、算法参数
  - 原文没给的基线对比数据、测试环境配置

其他情况一律判 actionable：
  - 表达弱、措辞模糊、STAR结构缺失 → actionable（优化器能改写表达）
  - 职责边界不清 → actionable（优化器能改为主动语态）
  - 技术描述停留在工具层、缺少方案级细节 → actionable（优化器能重组原文技术素材）
  - 未说明具体实现细节但原文有相关内容 → actionable

如果无法确定，必须判为 actionable。宁可多给优化器一个它做不了的issue，也不要漏掉它能做的。

输出格式：{"results": [{"issue_index": 0, "type": "actionable"}]}。"""


def classify_audit_issues(issues: list[dict[str, Any]], source_text: str) -> list[str]:
    """Classify unclassified audit issues using a dedicated LLM call.

    Args:
        issues: List of audit issues missing issue_type
        source_text: Original resume text for context

    Returns:
        List of "actionable" or "needs_data" strings, one per issue
    """
    if not llm_enabled() or not issues:
        return ["actionable"] * len(issues)

    # Build a compact prompt with just the problem statements
    issues_text = "\n".join(
        f"[{i}] {issue.get('problem', '')[:200]}"
        for i, issue in enumerate(issues)
        if isinstance(issue, dict)
    )

    prompt = (
        "【原始简历文本（参考上下文）】\n"
        f"{sanitize_user_text(source_text)[:3000]}\n\n"
        "【待分类的issue列表】\n"
        f"{issues_text}\n\n"
        "请判断每条issue是actionable还是needs_data。"
    )

    try:
        result = call_llm_typed(
            _IssueClassifyOutput,
            _ISSUE_CLASSIFY_SYSTEM,
            prompt,
            temperature=0.1,
            max_tokens=512,
        )
        results = result.get("results", []) if isinstance(result, dict) else []
        if not isinstance(results, list) or not results:
            return ["actionable"] * len(issues)

        labels = []
        for issue in issues:
            idx = issues.index(issue)
            label = "actionable"
            for r in results:
                if isinstance(r, dict) and r.get("issue_index") == idx:
                    label = r.get("type", "actionable")
                    break
            labels.append(label)
        return labels
    except Exception as exc:
        logger.warning("LLM issue classification failed: %s", exc)
        return ["actionable"] * len(issues)


class _IssueClassifyItem(BaseModel):
    issue_index: int = 0
    type: str = "actionable"


class _IssueClassifyOutput(BaseModel):
    results: list[_IssueClassifyItem] = Field(default_factory=list)


class _MissingVerdict(BaseModel):
    field: str = ""
    label: str = ""
    verdict: str = "truly_missing"  # parse_lost | truly_missing | not_applicable
    reason_llm: str = ""
    extracted_value: str = ""
    reason: str = ""


class _MissingOutput(BaseModel):
    verdicts: list[_MissingVerdict] = Field(default_factory=list)
