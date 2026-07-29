"""ResumeOptimizer: LLM Call 3 — optimize verified resume for target role.

V2 Layer 4. Takes verified CanonicalResume + JD, outputs optimized version.
- Rewrites weak bullets into STAR format (action verb + context + result)
- Reorders experiences by JD relevance
- Does NOT fabricate new facts — only restructures/rephrases existing content
"""
from __future__ import annotations

import logging

from server_runtime import call_llm_text, llm_enabled
from llm_gateway import parse_json_content
from v2_schemas import CanonicalResume

logger = logging.getLogger(__name__)

OPTIMIZER_SYSTEM_PROMPT = """你是一位简历优化专家。对已校验的简历做针对性优化，输出优化后的完整 JSON。

【输出结构不变】：
{
  "meta": {"name": "", "phone": "", "email": "", "target_role": "", "work_experience": ""},
  "summary": "",
  "education": [{"school": "", "degree": "", "major": "", "period": ""}],
  "experience": [{"organization": "", "role": "", "period": "", "bullets": [""]}],
  "research": [{"institution": "", "topic": "", "period": "", "bullets": [""]}],
  "projects": [{"name": "", "organization": "", "role": "", "period": "", "bullets": [""]}],
  "skills": {"items": [{"name": "", "category": ""}]},
  "awards": [""]
}

【优化规则】
1. bullets 改写：
   - 弱 bullet（无动词开头、无具体成果）→ 改为"动词 + 做了什么 + 怎么做的 + 结果"
   - 例：「参与用户调研」→「主导用户调研，通过问卷+访谈覆盖N名用户，输出需求文档推动X功能上线」
   - 但绝不能编造原文没有的数字或成果。如果原文没有数据，用定性描述代替
   - 每条 bullet 控制在 1-2 句话

2. 相关性排序：
   - 与目标岗位最相关的经历排最前面
   - 明显无关的经历（如志愿者对技术岗）排到最后，但不要删除
   - projects 也按相关性排序

3. summary 优化：
   - 重写为 2-3 句，突出与目标岗位的匹配点
   - 用原文有的事实，不要编造

4. 硬约束：
   - 不能新增原文没有的经历、项目、技能
   - 不能编造数字（百分比、金额、人数）
   - 不能改变 organization/school/name 等事实字段
   - 不能删除任何经历或项目（只能调顺序）
   - awards 保持不变

输出 JSON，不要额外解释。"""


def optimize_resume(resume: CanonicalResume, jd_text: str = "") -> CanonicalResume:
    """Optimize verified resume for target role. Returns optimized copy."""
    if not llm_enabled():
        return resume

    # Skip optimization if resume is mostly empty
    total_bullets = sum(len(e.bullets) for e in resume.experience) + \
                    sum(len(e.bullets) for e in resume.projects) + \
                    sum(len(r.bullets) for r in resume.research)
    if total_bullets < 2:
        logger.info("Optimizer skipped: only %d bullets", total_bullets)
        return resume

    resume_json = resume.model_dump_json(exclude_none=True)

    prompt = "请优化以下简历，使其更匹配目标岗位。\n\n"
    if jd_text.strip():
        prompt += f"【目标岗位】\n{jd_text.strip()[:500]}\n\n"
    prompt += f"【待优化简历】\n{resume_json}"

    try:
        content = call_llm_text(
            OPTIMIZER_SYSTEM_PROMPT,
            prompt,
            temperature=0.2,
            max_tokens=4096,
        )
    except Exception as exc:
        logger.warning("Optimizer LLM call failed: %s", exc)
        return resume

    parsed = parse_json_content(content)
    if not isinstance(parsed, dict) or not parsed:
        logger.warning("Optimizer JSON parse failed, len=%d", len(content))
        return resume

    # Strip wrapper key if present
    if len(parsed) == 1:
        key = next(iter(parsed))
        if key in ("resume", "data", "result"):
            inner = parsed[key]
            if isinstance(inner, dict):
                parsed = inner

    try:
        optimized = CanonicalResume(**parsed)
    except Exception as exc:
        logger.warning("Optimizer output validation failed: %s", exc)
        return resume

    # Safety check: don't lose content
    orig_exp = len(resume.experience)
    opt_exp = len(optimized.experience)
    orig_proj = len(resume.projects)
    opt_proj = len(optimized.projects)
    if opt_exp < orig_exp or opt_proj < orig_proj:
        logger.warning("Optimizer lost content: exp %d→%d, proj %d→%d, reverting",
                       orig_exp, opt_exp, orig_proj, opt_proj)
        return resume

    logger.info("Optimizer done: exp %d, proj %d, bullets %d→%d",
                opt_exp, opt_proj, total_bullets,
                sum(len(e.bullets) for e in optimized.experience) +
                sum(len(e.bullets) for e in optimized.projects) +
                sum(len(r.bullets) for r in optimized.research))
    return optimized
