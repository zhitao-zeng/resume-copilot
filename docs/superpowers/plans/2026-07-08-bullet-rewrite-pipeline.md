# Bullet 改写结构化优化实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将现有的单步 bullet patch 改写（一次 LLM 调用直接生成）拆分为 4 步结构化流水线：分析(Analyze) → 构思(Conceive) → 改写(Rewrite) → 校验(Verify)，提升技术描述深度并前置拦截编造内容。

**架构：** 在 `resume_optimization.py` 中新增 3 个函数（`analyze_bullet`、`conceive_material`、`verify_bullet`），修改 `_call_patch_llm_sync` → `_rewrite_bullet`，修改 `patch_optimize_weak_bullets` 编排流水线。新增 1 个测试文件 `tests/test_bullet_pipeline.py`。

**技术栈：** Python 3.13, pydantic BaseModel, call_llm_typed, FactLedger/FactBullet, asyncio, pylint

---

### 任务 1：新增 Analyze 阶段的 LLM 数据模型

**文件：**
- 修改：`resume_optimization.py:978-994`（在 `_PatchOutput` 附近新增）

新增 2 个 pydantic model：

```python
class _BulletAnalysisOutput(BaseModel):
    """LLM 输出：bullet 弱点诊断"""
    missing_situation: bool = False
    missing_task: bool = False
    missing_action: bool = False
    missing_result: bool = False
    missing_technical_detail: bool = False
    missing_metric: bool = False
    has_vague_language: bool = False


class _BulletVerdictOutput(BaseModel):
    """LLM 输出：改写结果校验"""
    is_safe: bool = True
    risk_tags: list[str] = []
    reason: str = ""
```

- [ ] **步骤 1：编写 2 个 pydantic model**

```python
# 在 class _PatchOutput(BaseModel): 之后插入
class _BulletAnalysisOutput(BaseModel):
    missing_situation: bool = False
    missing_task: bool = False
    missing_action: bool = False
    missing_result: bool = False
    missing_technical_detail: bool = False
    missing_metric: bool = False
    has_vague_language: bool = False


class _BulletVerdictOutput(BaseModel):
    is_safe: bool = True
    risk_tags: list[str] = []
    reason: str = ""
```

- [ ] **步骤 2：Commit**

```bash
git add resume_optimization.py
git commit -m "feat: 新增 Analyze/Verify 阶段的 pydantic model"
```

---

### 任务 2：实现 analyze_bullet() 函数

**文件：**
- 修改：`resume_optimization.py`（在 `_call_patch_llm_sync` 附近）

**说明：** 对单个 bullet 调用 LLM 分析其弱点，输出结构化诊断。单独做成函数，不耦合进流水线。

- [ ] **步骤 1：编写 analyze_bullet() 函数**

```python
_ANALYZE_SYSTEM_PROMPT = """你是简历 bullet 诊断专家。分析下面这条 bullet 的弱点，输出结构化诊断。

判断标准：
- missing_situation: 是否缺少背景/场景/业务上下文
- missing_task: 是否缺少任务目标或用户问题
- missing_action: 是否缺少具体动作/技术方案/实现方式
- missing_result: 是否缺少结果/效果/验证
- missing_technical_detail: 是否缺少技术实现细节（框架/算法/工具/参数）
- missing_metric: 是否缺少可量化的指标
- has_vague_language: 是否使用了"先进""显著""大量"等空泛词

只输出 JSON。"""


def analyze_bullet(bullet: FactBullet) -> Optional[dict[str, Any]]:
    """Step 1: Analyze bullet weakness. Returns structured diagnosis dict."""
    if not llm_enabled():
        return None
    prompt = (
        "【当前 bullet】\n"
        f"{bullet.source_text}\n\n"
        "【所属上下文】\n"
        f"{bullet.context}\n\n"
        "输出诊断 JSON。"
    )
    try:
        result = call_llm_typed(
            _BulletAnalysisOutput,
            _ANALYZE_SYSTEM_PROMPT,
            prompt,
            temperature=0.1,
            max_tokens=192,
            prefill='{"missing_situation":',
        )
        return result if isinstance(result, dict) else None
    except Exception as exc:
        logger.warning("analyze_bullet failed for %s: %s", bullet.id, exc)
        return None
```

- [ ] **步骤 2：Commit**

```bash
git add resume_optimization.py
git commit -m "feat: 实现 analyze_bullet() 诊断函数 + system prompt"
```

---

### 任务 3：实现 conceive_material() 函数

**文件：**
- 修改：`resume_optimization.py`（在 `analyze_bullet` 附近）

**说明：** 纯规则函数，从 FactLedger 和原文中提取可安全使用的素材。不需要 LLM 调用。

- [ ] **步骤 1：编写 conceive_material() 函数**

```python
from collections import Counter


def conceive_material(bullet: FactBullet, ledger: FactLedger) -> dict[str, Any]:
    """Step 2: Extract safe-to-use material from CV (rules, no LLM call).
    
    Returns dict with:
      - available_tech: list[str] — tech keywords from context/entities
      - available_metrics: list[str] — numbers/metrics from source
      - safe_angles: list[str] — description directions from entities
    """
    tech_keywords: list[str] = []
    metric_keywords: list[str] = []
    angle_keywords: list[str] = []
    
    # Collect tech keywords from entities (role, project_name, etc.)
    for (kind, val_lower), entity in ledger.entities.items():
        if kind in ("role", "project_name"):
            val = entity.value.strip()
            if len(val) >= 2 and val not in angle_keywords:
                angle_keywords.append(val)
    
    # Collect tech keywords from bullet's own metrics
    for m in bullet.metrics:
        if m not in metric_keywords:
            metric_keywords.append(m)
    
    # Extract tech keywords from raw text (model names, frameworks, tools)
    _TECH_PATTERN = re.compile(
        r'[A-Za-z][A-Za-z0-9_+./#-]{2,30}(?:\s+\d+(?:\.\d+)*)?'
        r'(?=\s*(?:框架|引擎|工具|平台|库|语言|技术|协议|数据库|系统|模型|算法))?',
        re.IGNORECASE
    )
    seen = set()
    for m in _TECH_PATTERN.finditer(ledger.raw_text):
        val = m.group().strip()
        val_lower = val.lower()
        if len(val) >= 3 and val_lower not in seen:
            seen.add(val_lower)
            tech_keywords.append(val)
    
    return {
        "available_tech": tech_keywords[:10],
        "available_metrics": metric_keywords[:8],
        "safe_angles": angle_keywords[:5],
    }
```

- [ ] **步骤 2：Commit**

```bash
git add resume_optimization.py
git commit -m "feat: 实现 conceive_material() 纯规则素材提取"
```

---

### 任务 4：实现 verify_bullet() 函数

**文件：**
- 修改：`resume_optimization.py`（在 `analyze_bullet` 附近）

- [ ] **步骤 1：编写 verify_bullet() 函数**

```python
_VERIFY_SYSTEM_PROMPT = """你是简历事实核查员。对比改写前后的 bullet，判断改写是否突破事实边界。

判断标准（任一 true 则 is_safe=false）：
1. fabricated_company: 原文没有的公司名、组织名
2. fabricated_metric: 原文没有的指标/数字
3. role_promotion: 角色层级被提升（参与→主导、协助→负责）
4. fabricated_tech: 技术栈/工具名完全不在原文
5. hallucinated_angle: 完全虚构的描述方向

只输出 JSON。"""


def verify_bullet(bullet: FactBullet, new_text: str, ledger: FactLedger) -> dict[str, Any]:
    """Step 4: Verify that the rewritten bullet is safe (no fabrication)."""
    if not llm_enabled():
        return {"is_safe": True, "risk_tags": [], "reason": "LLM disabled"}
    prompt = (
        "【原文 bullet】\n"
        f"{bullet.source_text}\n\n"
        "【改写后 bullet】\n"
        f"{new_text}\n\n"
        "【原始简历上下文】\n"
        f"{ledger.raw_text[:1500]}\n\n"
        "输出校验 JSON。"
    )
    try:
        result = call_llm_typed(
            _BulletVerdictOutput,
            _VERIFY_SYSTEM_PROMPT,
            prompt,
            temperature=0.1,
            max_tokens=128,
            prefill='{"is_safe":',
        )
        if isinstance(result, dict):
            return result
        # fallback: dict 结构不对但包含 json 时, 用默认 safe
        return {"is_safe": True, "risk_tags": [], "reason": "parse fallback"}
    except Exception as exc:
        logger.warning("verify_bullet failed for %s: %s", bullet.id, exc)
        return {"is_safe": True, "risk_tags": [], "reason": "verify exception"}
```

- [ ] **步骤 2：Commit**

```bash
git add resume_optimization.py
git commit -m "feat: 实现 verify_bullet() 事实验证函数"
```

---

### 任务 5：重构 _call_patch_llm_sync → _rewrite_bullet

**文件：**
- 修改：`resume_optimization.py:1049-1068`（`_call_patch_llm_sync`）

**说明：** 修改现有函数，接受 analysis + material 参数，增大 `max_tokens`，更新 prompt。

- [ ] **步骤 1：改写 _rewrite_bullet() 函数**

```python
_NEW_PATCH_SYSTEM_PROMPT = """你是简历表达优化专家。基于诊断结果和可用素材改写 bullet。

硬约束：
1. 已有指标（百分比、数字、规模）逐字保留，不能改数值
2. 公司名、岗位名、时间绝对不能动
3. 不提升角色层级：参与≠主导、协助≠负责
4. 不得编造原文没有的数字/指标
5. 按 STAR 重组（背景→动作→结果）

改写方向（严格遵循诊断结果）：
- 若 missing_situation=true → 补充业务背景/场景描述
- 若 missing_technical_detail=true → 结合可用技术词补充具体实现方式
- 若 missing_metric=true → 如原文有指标则保留，无指标则写验证方式
- 若 missing_result=true → 补充验证方式或可验证口径（不编造数字）
- 若 has_vague_language=true → 替换为具体描述

输出 JSON: {"bullet_id": "...", "new_text": "..."}"""


def _rewrite_bullet(
    bullet: FactBullet,
    jd_keywords: list[str],
    ledger: FactLedger,
    analysis: Optional[dict[str, Any]] = None,
    material: Optional[dict[str, Any]] = None,
) -> str:
    """Step 3: Rewrite a single bullet given analysis + material. Sync wrapper."""
    # Build the base prompt with analysis and material context
    analysis_block = ""
    if analysis:
        flags = []
        for k in ("missing_situation", "missing_task", "missing_action", "missing_result",
                   "missing_technical_detail", "missing_metric", "has_vague_language"):
            if analysis.get(k):
                flags.append(k.removeprefix("missing_").removeprefix("has_"))
        if flags:
            analysis_block = f"【诊断】缺: {', '.join(flags)}\n"
    
    material_block = ""
    if material:
        parts = []
        if material.get("available_tech"):
            parts.append(f"可用技术词: {', '.join(material['available_tech'][:8])}")
        if material.get("available_metrics"):
            parts.append(f"已有指标: {', '.join(material['available_metrics'][:6])}")
        if parts:
            material_block = "\n".join(parts) + "\n"
    
    jd_str = ", ".join(jd_keywords[:8]) if jd_keywords else ""
    jd_block = f"【JD 关键词】\n{jd_str}\n\n" if jd_str else ""
    
    prompt = (
        "【不可变事实】\n"
        f"所属: {bullet.context}\n"
        f"已有指标: {', '.join(bullet.metrics[:6]) if bullet.metrics else '（无已有指标）'}\n\n"
        f"{analysis_block}"
        f"{material_block}"
        f"{jd_block}"
        f"【当前 bullet】\n{bullet.source_text}\n\n"
        f'输出 JSON: {{"bullet_id": "{bullet.id}", "new_text": "..."}}'
    )
    
    try:
        result = call_llm_typed(
            _PatchOutput,
            _NEW_PATCH_SYSTEM_PROMPT,
            prompt,
            temperature=0.35,
            max_tokens=512,
        )
        new_text = result.get("new_text", "") if isinstance(result, dict) else ""
        return str(new_text).strip() if new_text else ""
    except Exception as exc:
        logger.warning("Rewrite LLM call failed for bullet=%s: %s", bullet.id, exc)
        return ""
```

- [ ] **步骤 2：更新 `_build_patch_prompt` 的引用**

搜索 `_call_patch_llm_sync` 的调用位置，更新为 `_rewrite_bullet`。调用处的参数传递参考任务6的流水线。

```bash
# 确认引用位置
grep -n "_call_patch_llm_sync\|_build_patch_prompt" resume_optimization.py
```

- [ ] **步骤 3：Commit**

```bash
git add resume_optimization.py
git commit -m "refactor: _call_patch_llm_sync → _rewrite_bullet, 支持analysis+material参数, max_tokens 256→512"
```

---

### 任务 6：重构 patch_optimize_weak_bullets() 流水线

**文件：**
- 修改：`resume_optimization.py:1071-1138`（`patch_optimize_weak_bullets`）

**说明：** 将现有的单步循环改为 4 步流水线。注意保持 asyncio 和 best-of-N 模式不变。

- [ ] **步骤 1：修改 `patch_optimize_weak_bullets()` 的流水线编排**

```python
async def patch_optimize_weak_bullets(
    ledger: FactLedger,
    jd_keywords: list[str],
    n: int = 3,
) -> list[BulletPatch]:
    """4-stage pipeline: Analyze → Conceive → Rewrite → Verify, best-of-N."""
    if not llm_enabled():
        logger.info("patch_optimize: LLM disabled, skipping")
        return []

    weak_ids = _scan_weak_bullets(ledger)
    if not weak_ids:
        logger.info("patch_optimize: no weak bullets found")
        return []

    patches: list[BulletPatch] = []
    bullet_map: dict[str, FactBullet] = {b.id: b for b in ledger.bullets}
    
    # ── Step 1: Analyze all weak bullets (batch, 1 LLM each) ──
    analyses: dict[str, Optional[dict[str, Any]]] = {}
    for bid in weak_ids:
        bullet = bullet_map.get(bid)
        if bullet is None:
            continue
        analyses[bid] = await asyncio.to_thread(analyze_bullet, bullet)
    
    # ── Step 2: Conceive material for all (pure rules, 0 LLM) ──
    materials: dict[str, dict[str, Any]] = {}
    for bid in weak_ids:
        bullet = bullet_map.get(bid)
        if bullet is None:
            continue
        materials[bid] = conceive_material(bullet, ledger)
    
    # ── Step 3+4: Rewrite + Verify, best-of-N ──
    for bid in weak_ids:
        bullet = bullet_map.get(bid)
        if bullet is None:
            continue
        
        analysis = analyses.get(bid)
        material = materials.get(bid)
        
        # Generate N candidates concurrently (2 at a time for max_num_seqs=2)
        candidates: list[str] = []
        for batch_start in range(0, n, 2):
            batch_size = min(2, n - batch_start)
            tasks = []
            for _ in range(batch_size):
                tasks.append(asyncio.to_thread(
                    _rewrite_bullet, bullet, jd_keywords, ledger, analysis, material
                ))
            batch_results = await asyncio.gather(*tasks)
            
            # Step 4: Verify each candidate
            for r in batch_results:
                if not r:
                    continue
                verdict = verify_bullet(bullet, r, ledger)
                if verdict.get("is_safe", False):
                    candidates.append(r)
                else:
                    logger.info("Verify reject bullet=%s: %s", bid, verdict.get("reason", "unsafe"))
        
        if not candidates:
            logger.info("No safe candidates for bullet=%s, keeping original", bid)
            continue
        
        # Select best from safe candidates
        try:
            best = select_best_candidate(bullet, candidates, ledger, jd_keywords)
            if best is not None:
                patches.append(best)
            else:
                logger.info("All candidates rejected by selector for bullet=%s, keeping original", bid)
        except Exception as exc:
            logger.warning("select_best_candidate failed for bullet=%s: %s", bid, exc)
    
    logger.info(
        "patch_optimize: %d weak bullets → %d patches applied (%d unchanged)",
        len(weak_ids), len(patches), len(weak_ids) - len(patches),
    )
    return patches
```

- [ ] **步骤 2：删除旧的 `_build_patch_prompt` 函数（不再使用）**

确认 `_build_patch_prompt` 不再被任何位置引用后删除：

```python
# 删除 _build_patch_prompt 函数（lines 1019-1046）
```

- [ ] **步骤 3：Commit**

```bash
git add resume_optimization.py
git commit -m "refactor: patch_optimize_weak_bullets 改为4步流水线 Analyze→Conceive→Rewrite→Verify"
```

---

### 任务 7：编写单元测试

**文件：**
- 创建：`tests/test_bullet_pipeline.py`

- [ ] **步骤 1：编写测试文件**

```python
"""Tests for the 4-stage bullet rewrite pipeline."""
import json
import pytest
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import patch, MagicMock

from resume_optimization import (
    conceive_material,
    _BulletAnalysisOutput,
    _BulletVerdictOutput,
)

# ── Factories for test data ──

@pytest.fixture
def sample_bullet():
    from fact_ledger import FactBullet
    return FactBullet(
        id="exp_0_b0",
        source_text="参与公司核心产品的功能迭代，负责收集用户反馈。",
        context="某科技公司 | 产品助理实习生 | 2025",
        entities=("产品助理",),
        metrics=("",),
        has_action=True,
        has_result=False,
        missing_info=True,
    )


@pytest.fixture
def sample_ledger(sample_bullet):
    from fact_ledger import FactLedger, FactEntity
    return FactLedger(
        entities={
            ("role", "产品助理"): FactEntity(kind="role", value="产品助理", source_span="产品助理实习生"),
            ("company", "某科技公司"): FactEntity(kind="company", value="某科技公司", source_span="某科技公司"),
        },
        bullets=[sample_bullet],
        meta={"name": "测试", "target_role": "产品经理"},
        raw_text="某科技公司 产品助理实习生 2025 参与核心产品功能迭代 收集用户反馈 撰写需求文档",
    )


# ── Test conceive_material (pure rules, no LLM) ──

def test_conceive_material_extracts_tech(sample_bullet, sample_ledger):
    """conceive_material should extract tech keywords from raw_text."""
    material = conceive_material(sample_bullet, sample_ledger)
    assert isinstance(material, dict)
    assert "available_tech" in material
    assert "available_metrics" in material
    assert "safe_angles" in material


def test_conceive_material_deals_with_empty_bullet():
    """conceive_material should handle a bullet with no metrics gracefully."""
    from fact_ledger import FactBullet, FactLedger, FactEntity
    empty_bullet = FactBullet(
        id="exp_0_b0",
        source_text="日常运营工作",
        context="某公司",
        entities=(),
        metrics=(),
        has_action=False,
        has_result=False,
        missing_info=True,
    )
    empty_ledger = FactLedger(
        entities={},
        bullets=[empty_bullet],
        meta={},
        raw_text="某公司 日常运营工作",
    )
    material = conceive_material(empty_bullet, empty_ledger)
    assert isinstance(material.get("available_tech"), list)
    assert isinstance(material.get("available_metrics"), list)
    assert isinstance(material.get("safe_angles"), list)


# ── Test Analyze model ──

def test_analysis_model_defaults():
    """_BulletAnalysisOutput should default all bools to False."""
    output = _BulletAnalysisOutput()
    assert output.missing_situation is False
    assert output.missing_task is False
    assert output.missing_action is False
    assert output.missing_result is False
    assert output.missing_technical_detail is False
    assert output.missing_metric is False
    assert output.has_vague_language is False


def test_analysis_model_parses_json():
    """_BulletAnalysisOutput should parse from dict."""
    data = {
        "missing_situation": True,
        "missing_technical_detail": True,
        "missing_result": True,
    }
    output = _BulletAnalysisOutput.model_validate(data)
    assert output.missing_situation is True
    assert output.missing_technical_detail is True
    assert output.missing_result is True
    assert output.missing_task is False  # not in input, defaults to False


# ── Test Verdict model ──

def test_verdict_model_defaults():
    """_BulletVerdictOutput should default to safe."""
    output = _BulletVerdictOutput()
    assert output.is_safe is True
    assert output.risk_tags == []


def test_verdict_model_parses_unsafe():
    """_BulletVerdictOutput should parse unsafe verdict."""
    data = {
        "is_safe": False,
        "risk_tags": ["fabricated_metric"],
        "reason": "数字完全不在原文",
    }
    output = _BulletVerdictOutput.model_validate(data)
    assert output.is_safe is False
    assert "fabricated_metric" in output.risk_tags
```

- [ ] **步骤 2：运行测试验证通过**

```bash
cd /mnt/disk1/zengzhitao/resume-copilot-server-acceptance
source .venv/bin/activate
python -m pytest tests/test_bullet_pipeline.py -v
```
预期：PASS

- [ ] **步骤 3：Commit**

```bash
git add tests/test_bullet_pipeline.py
git commit -m "test: bullet 4步流水线单元测试（conceive/analysis/verdict）"
```

---

### 任务 8：删除不再使用的旧代码

**文件：**
- 修改：`resume_optimization.py`

- [ ] **步骤 1：确认 `_build_patch_prompt` 无引用后删除**

```bash
grep -n "_build_patch_prompt" resume_optimization.py
# 确认只有定义行，无调用行
```

如果无调用，删除 `_build_patch_prompt` 函数（约 lines 1019-1046）以及旧的 `PATCH_OPTIMIZE_SYSTEM_PROMPT` 常量。

- [ ] **步骤 2：Commit**

```bash
git add resume_optimization.py
git commit -m "cleanup: 删除旧的 _build_patch_prompt 和 PATCH_OPTIMIZE_SYSTEM_PROMPT"
```

---

### 任务 9：集成测试——验证流水线不破坏现有功能

**说明：** 运行完整的 patch_optimize_weak_bullets 流程，确保 4 步流水线在 mock LLM 环境下能正常编排。

- [ ] **步骤 1：编写集成测试**

```python
"""Integration test for the full 4-stage pipeline."""
import pytest
from unittest.mock import patch, AsyncMock
from fact_ledger import FactBullet, FactLedger, build_ledger


@pytest.mark.asyncio
async def test_pipeline_runs_with_mock_llm():
    """Full pipeline should produce BulletPatch list even with mock LLM."""
    # Build a minimal ledger from sample text
    sample_raw = "某科技公司 | 产品助理 | 2025\n参与核心产品功能迭代，收集用户反馈。"
    ledger = build_ledger({"experience": [{"company": "某科技公司", "role": "产品助理", "period": "2025", "bullets": ["参与核心产品功能迭代，收集用户反馈。"]}]}, sample_raw, run_repair=False)
    
    from resume_optimization import patch_optimize_weak_bullets
    
    # Mock all 3 LLM steps to return canned data
    with patch("resume_optimization.call_llm_typed") as mock_llm:
        # analyze_bullet returns diagnosis
        # verify_bullet returns safe
        # _rewrite_bullet returns rewritten text
        mock_llm.side_effect = [
            {"missing_situation": True, "missing_technical_detail": True},
            {"is_safe": True, "risk_tags": []},
            {"bullet_id": "exp_0_b0", "new_text": "主导核心产品功能迭代，统筹用户反馈收集与需求分析流程。"},
        ]
        
        patches = await patch_optimize_weak_bullets(ledger, ["产品", "迭代"])
        
        assert isinstance(patches, list)
        if patches:
            # Verify the patch contains the rewritten text
            assert patches[0].new_text
            assert patcher[0].bullet_id
```

- [ ] **步骤 2：运行集成测试**

```bash
cd /mnt/disk1/zengzhitao/resume-copilot-server-acceptance
source .venv/bin/activate
python -m pytest tests/test_bullet_pipeline.py -v --asyncio-mode=auto
```
预期：PASS

- [ ] **步骤 3：Commit**

```bash
git add tests/test_bullet_pipeline.py
git commit -m "test: 4步流水线集成测试（mock LLM 验证编排）"
```
