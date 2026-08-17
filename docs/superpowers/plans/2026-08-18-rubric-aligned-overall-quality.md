# 整体质量计划 — 对齐平台 Rubric（2026-08-18）

> **面向执行者：** 本计划自包含，不依赖外部技能或插件。按任务顺序逐项执行；每个任务内步骤用复选框（`- [ ]`）跟踪，全部步骤完成并通过该任务末尾的**验收**后再开始下一个任务。每次改动后运行 `.venv/bin/python -m pytest -m 'not integration' -q`，当前基线为 **858 passed, 43 deselected**，不得净减少。**勾选任何验收项前必须留下可复核证据**（测试文件、结果 JSON 或命令输出路径）——R27 计划曾因无证据勾选被整体重置，不要重演。遇到与本计划描述不符的代码现状，停下来报告，不要自行扩大改动范围。

## 一、目标定义

平台评分规则（简历助手基线榜-中文）已确认：5 大项 16 细项，编造否决；**单 case ≥ 80 分为"可用"**，可用度 = 可用 case / 全部 case。

| 指标 | 现状（2026-08-18） | 目标 |
|---|---|---|
| 可用度（case ≥80） | ≈ 0% | **≥ 50%** |
| 本地 darvin-r3 均分 | V3 56.36 / V2 56.05 | **≥ 70** |
| 平台均分 | V2 线 65.48（f507820，历史最好） | ≥ 80 |
| 编造否决 | 0 | 保持 0（不可回退） |
| 最大延迟 | 479.3s / 480s | **≤ 460s** |

**双量尺换算**：本地 darvin-component-evaluator-r3 与平台 rubric 逐项对齐（10/30/40/20，细项权重一致），但主观项上本地代理更严。唯一锚点：V2 线本地 56.05 ↔ 平台 65.48（offset ≈ +9.4）。V3 无平台数据点——任务 1 的提交即为校准。

**发版纪律（每次合入前）**：
1. full60 逐 case 打分，**可用度不下降**；
2. 任何细项均值 < 0.5 不出门（书面豁免除外）；
3. atomic precision ≥ 0.99、编造 = 0、max latency ≤ 460s；
4. 修复与架构改动不混提交（59056c0 教训）。

## 二、事实基线

R26 full60（`.codex/research-loop/artifacts/darvin-aligned-quality-20260816/quality-gate-r24-v3/r26-full60-reaudit.json` 及同目录 darvin-r3）：

- 门禁全绿：60/60 成功、编造 0、precision 0.9996、ownership 0.9925（另有 249 undetermined 不计入）、max 479.3s。
- 组件：readability 0.9993 / completeness 0.6846（summary **0.0222**）/ expression **0.3744**（star **0.1739**@w20、ability 0.2609@w10）/ reply 0.8831。
- generation_quality：star_complete 0.043、JD supported **0/299**、industry 误判 **35/60**、recall 0.8619（education 缺 102、period 缺 51）。
- V2 对照（同评测器）：总分 56.05；star 0.1353（**无原子约束仍更低** → STAR 瓶颈是源数据缺 Result + 组装，不是逐字约束本身）；industry 误判 36/60（与 V3 几乎逐例相同 → 分类器共用，坏的是组件）；metric 型事实仅 3 处丢失（→ 量化结果大多存在，只是没被组装成 R）。

已落地（工作树未提交，2026-08-18，858 passed）：
- **A1** `summary_compiler._passes_atomic_audit` 门禁作用域从全文档 `Audit.clean` 改为仅 summary 自身违规——`dropped_atomic_audit:32` 的根因。
- **A0** `realizer._LATIN_CONNECTORS` 摘除 experienced/skilled/proficient/familiar/strong/solid（英文侧无源能力断言缺口，与 `_FORBIDDEN_SYNTHESIS` 对齐）；D10 测试改为合法织入，并新增回归测试钉死该缺口。
- **A2** `pipeline` not_rendered 仅限 {dropped_unverifiable, dropped_atomic_audit, fallback, budget_fallback}。
- **A3** reply 冲突区先过滤再切片。
- **A4** 岗位匹配遍历全部需求并报总数。

## 三、改写与编造的边界（政策，任务 3 的宪法）

平台否决项原文：「是否无中生有、是否篡改」，例子全部为实体与事实（捏造公司/经历/教育/时间）。**否决的是"新增可核查的事实断言"，不是"改变字面表达"。**

> **一句话判据：改写可以改变信息的形态和位置，不可以增加信息的含量。检验：这句话里每一个 HR 打电话能核实的断言，源材料里找得到出处吗？**

| 档 | 政策 | 例 |
|---|---|---|
| ✅ 无条件允许 | 重排/合并/拆分；织入同 record 的岗位/公司/时间上下文；把源中已有的结果动词升格到 R 位置；同 record 内把散落 metric 拼回动作句；口语→公文、去冗余 | 「确保设施的清洁和维护标准」→「监督设施清洁与维护（A），确保持续符合规定标准（R）」 |
| ⚠️ 受限允许 | 能力抽象须被源事实**蕴含**（"使用Python完成数据清洗"→"具备Python数据处理能力"可；"精通Python"不可）；动词不许升级职级（负责→承担 可；负责→主导/牵头 不可，除非源有主导类表述） | `_FORBIDDEN_SYNTHESIS` 保留 |
| ❌ 禁止 | 新实体、新数字、新时间、新事件、无出处的效果因果与目的从句、程度断言。R 真缺时：A 写好写满 + reply 定点问，不给对冲式假结果 | 源无效果陈述却写"使转化率提升" |

当前逐字校验管的是「不许出现源字符串以外的字」，比否决线严一个量级（翻译都算违规）。任务 3 将验证线移到否决线上方：**锚点逐字 + 数字/实体零新增 + 蕴含校验**，逐字要求只保留给锚点。

## 四、任务列表

### 任务 0：STAR 天花板测量（已完成 2026-08-18）

脚本：`/tmp/star_ceiling.py`（gpu16），输入 R26 reaudit 60 case 的 resume_data。
按 record 分桶：A=已含结果内容（独立 metric 或结果动词）→ 纯组装可救；B=无结果但有时间/组织上下文；C=两者皆无。

- [x] 跑出 A/B/C 分布与逐场景占比（证据：本节下方数字，脚本 `/tmp/star_ceiling.py`）
- [x] 依据 A 桶占比确定任务 3 的 star 目标值：**A = 60.1% ≥ 60% → star 目标定 0.50**

**测量结果（268 个经历族 record，45 个有 record 的 case）：**

| 桶 | 数量 | 占比 |
|---|---|---|
| A（结果内容已存在，纯组装可救） | 161 | **60.1%** |
| B（无结果，有时间/组织上下文） | 79 | 29.5% |
| C（两者皆无） | 28 | 10.4% |

- 细分：25.7% record 含独立 metric，49.6% 含结果动词。
- case 级：14 个 case 全部 record 为 A；34/45（76%）≥半数 record 为 A；仅 3 个 case 零 A（HV2-S1-002/005/010）。
- 逐场景 A 占比：S1 0.53 / S2 0.66 / S3 0.65（S4 无经历族 record，框架/查询路径）。

**结论**：STAR 瓶颈确证为组装问题而非数据贫困——**不触碰编造红线，star 的物理天花板约 0.60**；任务 3 目标 0.50 有余量。B+C 桶（40%）走 A-only 规范写法 + reply 定点问。

### 任务 1：full60 基线重跑 + 平台校准提交

- [ ] 用当前工作树（R27+A 批）跑 full60 + darvin-r3 重审计
- [ ] 验证预测：summary 0.0222 → ≥0.6，总分 ≥59；若未达，查 summary 状态分布再修
- [ ] 结果登记到本文档；这是后续所有改动的对照基线
- [ ] （负责人拍板）提交平台一次，目的=拿 V3 平台数据点校准 offset，不是冲线

### 任务 2：Rubric 白捡分（全部确定性工程）

按 rubric 分值排序：

- [ ] **2.1 排序**：经历/教育按开始时间由近及远（rubric 明文）。先查 planner 现状；源无时间的 record 保持源相对顺序，排序键缺失不猜。
- [ ] **2.2 技能归类**（D17 复活，rubric 技能 5 分明文"按类别归类"）：languages/tools/certifications/domains 四类，用 fact_type/来源 section 结构信号，不建词表。
- [ ] **2.3 必填字段抢救**：period/教育字段/成果字段的 undetermined 降级呈现进对应 section 的补充位置；**只救 rubric 范围内字段**（荣誉奖项等范围外内容不写不扣分，不塞杂物间）。目标 education missing 102 → 个位数。
- [ ] **2.4 缺失提示具体化**：补充类提示按字段点名（"缺失您的联系电话和邮箱"式），逐段经历缺 start/end/成果的逐条点名。
- [ ] **2.5 跨经历时间重叠检测**（新验证器）：experience 家族 record 两两比较 period 区间，重叠即入确认类提示。现有 conflict 检测只查同 record 同类型文本不一致，查不到这个。
- [ ] **2.6 行业错报清零**：有 JD 从 RequirementGraph 推导；无 JD 不声称行业走通用模板。分类器与 V2 共用，一次修复两线受益。
- [ ] **2.7 杂项**：≤3 页检查；模板符合度路径复核；碎句处理从 `_DROP_DEFECTS` 丢弃改为"合并→回退同 record 源句"（缺陷清单 D5 原建议），并放开三行合并（`merge_fragment_claims` 的 no-chain 限制只防回环，不防延续）。
- [ ] **2.8 延迟余量**：收紧 admission gate 预留，full60 max ≤ 460s。

**验收**：full60 对照任务 1 基线——completeness ≥0.82、reply ≥0.93、结构缺陷 0、行业错报 0、教育/时间字段缺失清零或进提示、precision ≥0.99、可用度不降。

### 任务 3：表达层重建（决定可用度的战役）

前置：任务 0 的天花板数字、任务 1 的基线。

- [ ] **3.1 验证器换轨**：`_residual_escape_violations` 的词表+逐字包含判定 → span 覆盖/蕴含判定；summary 与 realizer 共用同一实现（当前 summary_compiler 与 realizer 各养一套是 32 例误杀的温床）。跨语言（D10）随之解决，`_LATIN_CONNECTORS`/`_CONNECTORS` 词表整体退役。**中文 `_CONNECTORS` 中的断言动词（担任/兼任/具备/拥有）在换轨时一并处理，换轨前不单独动**（爆炸半径大）。
- [ ] **3.2 重写式 realizer**：per-record unit 输入固定 fact_ids，LLM 重写为专业 STAR 表达；三层验证=硬锚点逐字在场且零新增（现有）+ 数字/实体零新增（现有）+ 蕴含校验（新，27B 自评、schema 化、每 unit 一次）；失败回退源句（现有 per-unit 机制，最坏=今日输出）。
- [ ] **3.3 选择性呈现**（rubric 能力凸显度的"冲突"条款）：与目标岗位冲突的能力描述移出总结/降权；无 JD 中性处理。判定用 JD 语义信号，不建冲突词表。
- [ ] **3.4 缺 R 的经历**：A-only 写规范 + reply 定点问量化结果（B/C 桶的出路）。
- [ ] **3.5 延迟供养**：平台按 40G 半卡记账（gpu-util 0.88×40≈35G）——评估关 enforce-eager、提 max-num-seqs；蕴含校验优先打包进现有 pack 调用。admission gate 管尾部：mean 附近 case 全流程，逼近 460s 降级为今日路径。

**验收**：star_richness ≥（任务 0 定的目标）、expression ≥0.55、precision ≥0.99、编造 0、max ≤460s、可用度显著 >0（首批 case 过 80）。每个子步骤独立 full60 消融，star 与 precision 同时记录。

### 任务 4：模型与匹配（与任务 3 后半并行）

- [ ] 4.1 用任务 3 积累的验证通过/拒绝样本在 gpu16 本地（7×A100-80G）SFT 27B；产物 AWQ 量化后经平台挂载路径部署（`ceph_customer:zengzhitao/resume-copilot/models/`）。目标：重写质量↑、蕴含拒绝率↓、重试次数↓。
- [ ] 4.2 JD 匹配从 token 重叠换语义匹配（embedding + 事实级支撑判定）；`_GENERIC_MATCH_TOKENS`、`_REQUIREMENT_CONTENT_SIGNAL` 词表随之退役。supported 0/299 → 有真实支撑。

### 任务 5：证明与提交

- [ ] 5.1 盲评：新版 vs V2、vs f507820 输出（R24 曾 0:4 输给 V2；这是"整体质量好"的直接证据，必须赢或平）
- [ ] 5.2 逐 case 打分报告：可用度、各 case 距 80 的短板分布
- [ ] 5.3 平台正式提交（负责人）；leaderboard.yml 更新镜像 tag
- [ ] 5.4 收尾：37+ 未推提交推送 origin；V2 路径逐字节不变复核；退役夹具端到端复跑

## 五、不在本计划范围

- D21 两栏扫描件阅读序（上游 OCR/版面，独立立项）
- D20 非数字 OCR 错字守卫（需与数字守卫同构的置信度方案）
- 多文档 career vault、训练数据扩采（SOTA 文档记录的方向，本计划不做）
- V2 线的任何功能投入（冻结为回退基线）

## 六、记录维护约定

- **本文档是唯一的计划事实源**；缺陷清单（`docs/quality-defects-2026-08-17.md`）与本计划互补覆盖，新缺陷两处同步。
- 每个任务完成时在对应小节登记：结果数字、artifact 路径、commit id。
- 历史账目待补：R27 计划中 D16 标签错误（59056c0 实修 D5 回归）、整体验收第 2-4 项证据缺失——在任务 1 完成时一并订正到 R27 计划文档。

## 七、参考

- 平台 rubric：简历助手基线榜-中文（2026-08-18 由产品方提供，全文见本计划第一节引用的分值表）
- 缺陷清单：`docs/quality-defects-2026-08-17.md`（D1–D23）
- 架构不变量：`docs/architecture/resume-evidence-compiler-v3.md`
- SOTA 扫描：`docs/research-sota-2026-08-17.md`（细粒度归因告诫：arXiv 2604.01432）
- R27 计划（前序）：`docs/superpowers/plans/2026-08-17-output-quality-remediation.md`
- 平台历史最好：commit `f507820`，镜像 `f507820-post-guard-consistency`，Darvin 65.48（84/84）
