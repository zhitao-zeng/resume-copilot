# Bullet 改写结构化优化方案

## 问题背景

112 样本评测中，约 30% 的样本存在技术描述深度不够的问题：bullet 改写后仍然浮于表面（「使用了先进技术」「显著提升」等空泛表达）。同时，字段补全经常触发下游 fabric 检查红线，导致样本被归零。

当前 patch 优化是「一次 LLM 调用完成改写」，缺少结构化分工，导致：
1. LLM 不知道具体缺什么（缺过程？缺技术？缺结果？）
2. 没有显式的素材搜集环节，LLM 容易编造
3. 没有前置校验，全靠下游 fabric check 拦截

## 设计方案：分析 → 构思 → 改写 → 校验

对每个弱 bullet，执行以下 4 步流水线，做 3 次 best-of-N 后选最优：

```
                    ┌─────────────────┐
                    │   Step 1        │
                    │   Analyze (LLM) │  → 诊断结果 + STAR 成分分析
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Step 2        │
                    │   Conceive (规则)│  → 可用素材（技术词/指标/安全方向）
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Step 3        │
                    │   Rewrite (LLM) │  → 生成改写文本
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   Step 4        │
                    │   Verify (LLM)  │  → 校验是否突破事实边界
                    └────────┬────────┘
                             ↓
                       通过 → 加入候选池
                       不通过 → 丢弃

              重复 3 次 → select_best_candidate() 选最优
```

### Step 1: Analyze (LLM, 1 次调用)

对每条弱 bullet，分析其弱点。

**输入：**
- bullet 原文
- 所属上下文（公司/岗位/项目名称）

**输出结构（代码）：**
```python
class BulletAnalysis:
    # STAR 成分分析
    missing_situation: bool     # 缺背景/场景描述
    missing_task: bool          # 缺任务/目标
    missing_action: bool        # 缺具体动作/方案/实现
    missing_result: bool        # 缺结果/效果/验证
    
    # 技术深度
    missing_technical_detail: bool  # 缺技术实现细节
    missing_metric: bool            # 缺量化结果
    
    # 语言质量
    has_vague_language: bool    # 有空泛词（先进、显著、大量等）
    
    # 可安全使用的线索
    inferred_tech: list[str]    # 从上下文推断的可补充技术方向
```

**prompt 设计：**
```
你是一位简历诊断专家。分析以下 bullet 的弱点：
- 它属于哪个STAR成分？（Situation/Task/Action/Result）
- 缺了哪些成分？
- 是否有技术细节不足？
- 是否有空泛词？

输出 JSON 格式的诊断结果。
```

**耗时：** ~1s, max_tokens=128

### Step 2: Conceive (规则, 0 次 LLM 调用)

从原文上下文提取可安全使用的素材，无需 LLM。

**路径：**
- `FactLedger.entities` → 提取 entity 中的技术词（框架名、语言、工具）
- 原文 regex → 提取数值/指标（`\d+%`, `\d+万`, `\d+QPS` 等）
- `FactBullet.skills` → 提取 bullet 已有的技能标签

**输出结构（代码）：**
```python
class BulletMaterial:
    available_tech: list[str]       # 从原文提取的可安全引用的技术词
    available_metrics: list[str]    # 从原文提取的已有指标
    safe_angles: list[str]          # 可安全补充的描述方向
```

### Step 3: Rewrite (LLM, 1 次调用)

基于分析结果 + 可用素材，生成改写。

**输入：**
- 原 bullet
- Analyze 的诊断结果
- Conceive 的可用素材
- JD 关键词（如有）

**与当前 prompt 的关键差异：**

| 当前 | 改后 |
|------|------|
| 「按 STAR 重组」泛泛要求 | 明确指定缺哪一块（如「缺Situation，补充背景」）|
| 无技术细节引导 | 「缺技术细节 → 结合 available_tech 补充具体实现」|
| max_tokens=256 | max_tokens=512 |

**prompt 模板：**
```
【诊断结果】
缺Situation=true, 缺Action=false, 缺Result=true
缺技术细节=true
可用技术词: ["React", "Node.js", "MySQL"]
可用指标: ["50+ API接口"]

【改写要求】
1. 补充Situation：说明任务背景和业务场景
2. 补充Result：量化结果或验证方式
3. 技术细节：结合可用技术词补充具体实现方式
4. 不得编造原文没有的指标
5. 不提升角色层级

【原文】
{bullet.source_text}

输出 JSON: {"bullet_id": "...", "new_text": "..."}
```

不编造约束同当前（事实保真、不提升角色、不编造数字）。

### Step 4: Verify (LLM, 1 次调用)

校验改写结果是否会触发 fabric 红线。

**输入：**
- 改写前后对比
- 原文上下文

**输出结构：**
```python
class BulletVerdict:
    is_safe: bool                   # 是否安全
    risk_tags: list[str]            # 风险标签（如 fabricated_company, fabricated_metric）
    reason: str                     # 简要说明
```

**安全策略（具体标准）：**
- `is_safe=False` 的直接丢弃该候选，不应用
- 只有 3 个候选全部不安全时才保留原文
- 具体风险标签：
  - `fabricated_company` — 公司/组织名完全不在原文
  - `fabricated_metric` — 数字/百分比不在原文
  - `role_promotion` — 角色层级被提升（参与→主导、协助→负责）
  - `fabricated_tech` — 技术栈/工具名完全不在原文及上下文
  - `hallucinated_angle` — 完全虚构的描述方向
- 对于 `fabricated_metric` 但原文有相近数字（差异 <10%）→ 降级为 warning 而非 unsafe

**耗时：** ~0.5s, max_tokens=64

## 数据流

```
patch_optimize_weak_bullets()
  │
  ├─ for each weak bullet:
  │    ├─ analyze_bullet(bullet, context)           → BulletAnalysis    (LLM)
  │    ├─ conceive_material(bullet, ledger, raw)     → BulletMaterial    (规则)
  │    ├─ repeat ×3 (best-of-N):
  │    │    ├─ rewrite_bullet(analysis, material)    → new_text          (LLM)
  │    │    └─ verify_bullet(original, new, context) → BulletVerdict     (LLM)
  │    │         ├─ safe → add to candidates
  │    │         └─ unsafe → discard
  │    └─ select_best_candidate(candidates)          → BulletPatch | None
  │
  └─ return patches
```

## 耗时影响

| 阶段 | 当前 | 改后 | 变化 |
|------|------|------|------|
| Analyze | - | 1 LLM/bullet | +1 次 LLM |
| Conceive | - | 0 LLM | — |
| Rewrite | 3 LLM/bullet | 3 LLM/bullet | 不变 |
| Verify | - | 1 LLM/bullet | +1 次 LLM |
| **每 bullet 总计** | **3 LLM** | **5 LLM** | **+2 次** |

按每 bullet ~1s/次计算，单样本约 15 个 bullet → 增加 ~30s。当前平均 160s/样本 → ~190s。

**可选优化：** batch Analyze（一次分析所有 weak bullet），减少到 1 次 LLM 调用。

## 与现有 fabric check 的关系

| 拦截点 | 拦截对象 | 后果 |
|--------|---------|------|
| Step 4 Verify | 单个改写候选 | 丢弃不安全候选，保留原文 |
| 下游 `final_fact_guard()` | 整份简历 | 原逻辑不变，但 Verify 已前置拦截大部分问题 |

Verify 是**前置检查**，替代不了下游的 fabric check，但可以大幅减少 fabric false positive（因为大部分编造改写已在 Verify 阶段被丢弃）。

## 待今后考虑

- batch Analyze 优化：一次分析所有 weak bullet，减少 LLM 调用
- Verify 的规则兜底：当 LLM 校验超时/失败时，用规则做快速判断
- 行业感知：不同行业（技术 vs 非技术）对"技术深度"的要求不同
