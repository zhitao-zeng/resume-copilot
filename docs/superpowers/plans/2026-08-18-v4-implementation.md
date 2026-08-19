# V4 实施计划 — 证据驱动的 JD 定向改写（2026-08-18）

> **面向执行者：** 本计划自包含。**任务 0 是风险前置门，未通过不得进入任务 1。**
> 每任务独立 commit（禁 push），单元测试随修随跑，基线 `922 passed, 43 deselected`
> （2026-08-18 于 408517a 实测），不得净减少。**勾选验收前必须留可复核证据**
> （测试文件、结果 JSON 或命令输出路径）。遇到与本计划描述不符的代码现状，
> 停下来报告，不要自行扩大改动范围。

**设计来源：** 负责人《Resume Copilot 新架构设计总结》（2026-08-18）。本计划是其工程化
落地，不改变其结论；文档与本计划冲突时以文档为准，并在此登记差异。

## 一、为什么现在做 V4

**V3 从未上过平台。** R26 镜像 `77708bc-launch-r26` 内不含 `core/v3`，平台 61.1 分由 V2
路径产生（2026-08-18 实测，见 R28 计划任务 3.5 登记）。因此 V4 **没有生产包袱**，
是替换成本最低的时刻。

**V3 的病灶已量化。** 单份真实简历 280s 中 v3 流水线占 250s，语义编译墙钟约 215s，
改写器始终 `budget_fallback` ——**产品价值最高的一步从未执行**。这正是平台反馈
「基本没有对原简历的描述做太多改写」的成因。设计文档的结论
「Semantic Compiler 过重，安全编译消耗了本应留给简历优化的预算」由此得到实测支撑。

**V3 的资产必须保留。** span 逐字精确性、record 归属优先级、OCR 条件路由、原子事实安全门
是数轮迭代才调对的部分，V4 以 **import 复用**方式继承，不重写、不拷贝。

## 二、模块边界与复用清单

```
CV/Query/Template
  → 1 Structure Parser   → DocumentGraph
  → 2 Evidence Builder   → EvidenceGraph
  → 3 JD Grounder        → requirement × support-state × fact_ids
  → 4 Rewrite Planner    → RewritePlan
  → 5 STAR Realizer      → ResumeDraft claims
  → 6 Claim Verifier     → accepted / rejected / local fallback
  → 7 Assembler          → Resume + DOCX + reply + audit
```

**IR 收敛为四个**：`DocumentGraph` / `EvidenceGraph` / `RewritePlan` / `ResumeDraft`。
禁止再引入近义 Graph。

| V4 模块 | 复用自 V3 | 处置 |
|---|---|---|
| 1 Structure Parser | `document_graph.py` `input_adapters.py` `section_ontology.py` `numeric_guard.py` `text_integrity.py` | **原样 import**（含 R28 全部修复） |
| 2 Evidence Builder | `fact_graph.py` 的 record 归属与 span 绑定 | 保留归属逻辑，**弃用 `semantic_llm.py`（942 行）的重编译** |
| 3 JD Grounder | `jd_graph.py`（46 行，仅排序） | **新写** |
| 4 Rewrite Planner | `planner.py`（104 行） | **新写**，含 JD support / STAR 状态 / 槽位 / 改写目标 |
| 5 STAR Realizer | — | **新写**，单次调用；`realizer*.py`（1535 行）与 `summary_compiler.py`（525 行）弃用 |
| 6 Claim Verifier | `atomic_verifier.py` | **原样 import**，判据由「限制字面变化」转为「限制事实增加」 |
| 7 Assembler | `resume_adapter.py` `reply_builder.py` + 既有 renderer | **原样 import** |

新代码落在 `core/v4/`，经 `RESUME_PIPELINE_VERSION=v4` 选择。
**V2 保持生产默认；V3 保留为对照，直到 V4 在盲评上跑赢。**

## 三、边界政策（V4 宪法）

沿用 R28/主计划已定条款，并按设计文档增补：

**四态支撑模型**（JD Grounder 与 Planner 共用）：

| 状态 | 含义 | 允许的产出 |
|---|---|---|
| `SUPPORTED` | 有 evidence 直接支撑 | 正常写入，必须绑定 fact_ids |
| `PARTIAL` | 部分支撑 | 写已有部分，缺口留槽 |
| `MISSING` | 该经历缺关键 STAR 组件 | 正文留 `______`，reply **定点**追问 |
| `UNSUPPORTED` | 仅 JD 想要、候选人无证据 | **不得进入正文，也不得诱导用户补一段可能不存在的经历** |

`related_only` 归入 `UNSUPPORTED` 的正文禁令，但在 reply 中可作为方向建议。
设计文档判例：**TensorRT OCR 优化 ≠ LLM 推理优化**——相关不等于支撑。

**已定不变量（不得违反）：**
- **记录顺序不受 JD 影响**：工作/教育 record 保持时间倒序或源相对顺序。JD 只改变
  Summary、record 内 bullet 优先级与表达角度。
- 量化槽位形式为下划线 `______`，reply **逐处点名**（负责人 2026-08-18 拍板）。
- Summary 为岗位优势合成：≤100 字、2–3 条最强源支撑优势、禁清单式罗列、禁程度词。
- **不得新增关键词/短语/标签常量表做内容判定**（三问判据见主计划）。规则只管形式结构
  （电话、日期、Markdown、表格、heading、separator、record boundary、list marker）；
  能力、成果、职责、JD 支撑关系交给 LLM。
- 验收以**盲评**为准，本地分只作诊断。

## 四、任务 0：合并 Planner 调用可行性验证（风险前置门）

**这是 V4 最大的未验证假设。** 设计文档要求 Planner 一次 LLM 调用完成
JD grounding + STAR 完整度 + 槽位规划 + 改写规划。若一次调用无法稳定输出高质量结构化
结果，整个 V4 的调用预算不成立——**必须在写任何模块前证伪或证实**。

- [ ] **0.1** 从 dogfood 取 3 例真实输入（含 1 例带 JD、1 例长简历、1 例扫描件），
  **手工构造** EvidenceGraph 与 JD 输入（不依赖尚未实现的模块）。
- [ ] **0.2** 定义合并 Planner 的 JSON Schema：requirement×support-state×fact_ids、
  每 record 的 STAR 状态、missing slots、bullet 优先级、rewrite goal。
- [ ] **0.3** 对 3 例各调用 3 次（共 9 次），记录：**单次墙钟延迟**、schema 合规率、
  support 判定与人工判读的一致性、是否出现 `related_only` 误判为 strong。
- [ ] **0.4** 判定：
  - **通过** = 单次 ≤60s、schema 合规 ≥8/9、support 判定无「相关即支撑」错误 → 进任务 1
  - **降级** = 拆为 2 次调用（grounding / planning）重测；预算仍需 ≤90s 合计
  - **否决** = 两种都不达标 → **停止 V4，回报并重新设计**，不得硬推

**验收：** 9 次调用的原始输出与判读表落盘 `.codex/research-loop/artifacts/v4-task0/`，
结论写回本节。

## 五、任务 1：Evidence Builder（同时是最大的延迟收益）

**现状病灶：** `semantic_llm.py` 要求模型逐字输出 `context_spans` 覆盖所有非事实字符
（标签、分隔符、重复标题）。**这是 atoms 的补集，可由确定性代码计算**，让模型复述整份
简历是 215s 墙钟的主因。

- [ ] **1.1** 新建 `core/v4/evidence.py`：从 DocumentGraph 构建 EvidenceGraph
  （`fact_id` / `record_id` / `type` / `text` / `source_span` / `entity` / `metric`），
  **record 归属逻辑 import 自 `core/v3/fact_graph.py`，不重写**。
- [ ] **1.2** 语义调用只做「这段原文是不是个人事实、属于哪个 record、是什么类型」，
  **不再要求输出 context_spans**——非事实部分由代码取补集。
- [ ] **1.3** 保留 fail-closed：语义判定无效时回退源文本，绝不猜测。
- [ ] **1.4** 实测对比：同一份简历，V3 语义编译墙钟 vs V4 Evidence Builder 墙钟，
  记录 token 输出量差异。
- [ ] **1.5** 测试：span 逐字性、record 归属正确性、fail-closed 行为、补集覆盖完整性。

**验收：** EvidenceGraph 的 fact 集合与 V3 FactGraph 在同一输入上**事实层面等价**
（允许 context 表示不同）；语义阶段墙钟**至少减半**；922 基线不减。

## 六、任务 2：JD Grounder

- [ ] **2.1** 新建 `core/v4/jd_grounding.py`：JD → atomic requirements。
- [ ] **2.2** 对每个 requirement 判定 `support` 四态 + 支撑 `fact_ids`。
  **不得用 embedding/token overlap 直接当事实支撑**（现行 `support_rate 0/299` 的病灶）。
- [ ] **2.3** `related_only` 必须与 `strong` 区分并落盘，供 reply 使用。
- [ ] **2.4** 测试：含设计文档判例（TensorRT OCR vs LLM 推理部署必须判为 `related_only`）。

**验收：** dogfood 带 JD 的 case 上，`SUPPORTED` 项均可回指 fact_ids；无「相关即支撑」误判。

## 七、任务 3：Rewrite Planner（含任务 0 验证过的合并调用）

- [ ] **3.1** 新建 `core/v4/rewrite_plan.py`：RewritePlan IR —— Summary evidence、
  record-local bullet plan、fact 组合、rewrite goal、STAR 状态、missing slots、questions。
- [ ] **3.2** **固化记录顺序不变量**：Planner 输出 `preserve_record_order=true`，
  下游不得重排 record。写成断言，不是约定。
- [ ] **3.3** 按任务 0 结论实现合并调用（1 次或降级为 2 次）。
- [ ] **3.4** 测试：JD 变化时 record 顺序逐字节不变；bullet 优先级随 JD 变化。

## 八、任务 4：STAR Realizer

- [ ] **4.1** 新建 `core/v4/realizer.py`：输入 = 单 record facts + bullet plan +
  **已 grounding 的 requirement**（不让它自由读整份 JD 猜候选人满足什么）。
- [ ] **4.2** 单次调用同时产出：改写句 + **每句自证依据（引用的 fact_ids）**。
  蕴含校验消费该产物，**不另起 LLM 调用**（延迟预算不允许，见 R28 任务 3.5）。
- [ ] **4.3** 量化槽位：`MISSING` 时写 `______`，并向 reply 输出定点问题。
- [ ] **4.4** Summary 为岗位优势合成，同轨校验。
- [ ] **4.5** 失败回退该 record 源句（最坏 = 今日输出）。

## 九、任务 5：Claim Verifier 判据换轨

- [ ] **5.1** import `core/v3/atomic_verifier.py`，判据从「限制字面变化」转为
  **「限制事实增加」**：实体零新增、数字零新增、record ownership 正确、fact 覆盖、
  无源程度升级。
- [ ] **5.2** 跨语言（D10）随之解决：翻译不再违规，逐字要求只保留给硬锚点。
- [ ] **5.3** 测试：合法改写通过、新增实体/数字被拒、程度词升级被拒。

## 十、任务 6：Assembler 与 reply 一致性

- [ ] **6.1** import `resume_adapter.py` / `reply_builder.py`，接 ResumeDraft。
- [ ] **6.2** **Resume 与 reply 从同一个 RewritePlan/missing slots 生成**，
  杜绝正文与缺失提示互相矛盾（D1 类矛盾的根治）。
- [ ] **6.3** 测试：正文留槽处 reply 必有对应定点问题，反之亦然。

## 十一、任务 7：集成验证（唯一端到端验证点）

- [ ] **7.1** dogfood 16 例跑 V4，对照 r0/r1：真实可用三档、结构缺陷、姓名/摘要率
- [ ] **7.2** frozen full60：precision ≥0.99、编造 0、max latency ≤460s
- [ ] **7.3** **盲评门禁**：V4 vs V3 vs V2（含 ≥2 例真实简历），**必须赢或平**
- [ ] **7.4** **镜像交付验证**：构建候选镜像并断言 `core/v4` 确实在镜像内、
  且 `RESUME_PIPELINE_VERSION=v4` 能实际启动并处理一个真实请求。
  **不得只验证代码能在工作区跑通。**
  > 教训（2026-08-18）：`config/Dockerfile` 的 `COPY core/*.py` 只匹配顶层文件，
  > 导致 `core/v3` 从未进入任何镜像——五轮 V3 工作因此从未上过平台，而单测、健康检查、
  > 平台评分全部为绿，**没有任何信号指向它**。已修复并加构建期守卫，V4 必须复验同一条。
- [ ] **7.5** 登记结果数字 + artifact 路径 + commit id，交负责人总 review

## 十二、不在本计划范围

- V2 路径任何功能改动（冻结为回退基线）
- 真实 PDF 版面章节分类（R28 任务 0.2 已诊断，另行立项）
- SFT 27B、语义 JD embedding 匹配（主计划任务 4）
- Presentation Model / 原生 V4 Renderer（设计文档明确可后置）

## 十三、回退与止损

- V4 全程不改 V2/V3 代码路径；任一阶段失败可直接停用 `RESUME_PIPELINE_VERSION=v4`
- **任务 0 否决即停止**，不进入实现阶段
- 任务 1 若语义墙钟未能至少减半，说明延迟根因判断有误，**回到诊断而非继续搭模块**

## 十四、工作量与风险提示

粗算：可复用约 2000 行（且是最难调对的部分），需新写约 4300 行但替换物更简单
（单次调用 Realizer 显著简于现行 record-local 多层回退）。规模约等于已完成的
R24–R28 中的一到两轮。

**风险集中在任务 0。** 若合并调用不成立，V4 的调用预算与 V3 无异，届时应重新设计
而非硬推。**这是竞赛项目，中途换架构的时间成本需由负责人权衡后再启动任务 1。**

## 十五、参考

- 设计来源：负责人《Resume Copilot 新架构设计总结》（2026-08-18，`~/Downloads/resume_copilot_architecture_summary.docx`）
- 延迟实测与三层限流根因：`2026-08-18-real-input-hardening.md` 任务 3.5 登记
- 缺陷清单：`docs/quality-defects-2026-08-17.md`（D1–D23）
- 主计划（目标与验收口径）：`2026-08-18-rubric-aligned-overall-quality.md`
- 平台反馈：`docs/feedback/2026-08-18-platform-feedback.md`
